"""
Model adapter for LEAD NavSim (LTFv6) — ported from BridgeSim.
Uses the self-contained ltfv6.py model definition bundled with the checkpoint.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS


@register_policy("lead_navsim")
class LEADNavsimAdapter(SensorPolicy):
    """
    Adapter for LEAD NavSim (LTFv6) model.
    Uses 4 cameras (front, front-left, front-right, back) with 1920x270 resolution.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config = None
        self.image_width = 1920
        self.image_height = 270

    def load_model(self):
        print("Loading LEAD NavSim (LTFv6) model...")
        checkpoint_dir = Path(self.checkpoint_path).parent
        sys.path.insert(0, str(checkpoint_dir))

        from ltfv6 import load_tf

        self.model = load_tf(self.checkpoint_path, torch.device(self.device))
        self.config = self.model.config

        type(self.config).torch_float_type = property(lambda self: torch.float32)
        self.model = self.model.float()
        print("Forced float32 mode for inference")

        if hasattr(self.config, 'final_image_width'):
            self.image_width = self.config.final_image_width
        if hasattr(self.config, 'final_image_height'):
            self.image_height = self.config.final_image_height

        print(f"Image dimensions: {self.image_width}x{self.image_height}")
        print("LEAD NavSim model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0')}

    def _preprocess_rgb(self, camera_images: Dict[str, np.ndarray]) -> torch.Tensor:
        cam_order = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_B0']
        processed_cams: list[np.ndarray] = []
        cam_width = self.image_width // 4

        for cam_name in cam_order:
            if cam_name not in camera_images:
                processed_cams.append(np.zeros((self.image_height, cam_width, 3), dtype=np.uint8))
                continue
            img = camera_images[cam_name]
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (cam_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            processed_cams.append(img)

        rgb = np.concatenate(processed_cams, axis=1)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return rgb_tensor

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        rgb = self._preprocess_rgb(images).unsqueeze(0).to(self.device)
        speed: float | np.floating[Any] = np.linalg.norm(ego_state['velocity'][:2])
        KICKOFF_SPEED = 2.0
        if speed < 0.5:
            speed = KICKOFF_SPEED

        acceleration: float | np.floating[Any]
        if 'acceleration' in ego_state:
            acceleration = np.linalg.norm(ego_state['acceleration'][:2])
        else:
            acceleration = 0.0

        command = ego_state.get('command', 1)
        cmd_onehot = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD).copy()

        return {
            'rgb': rgb,
            'speed': torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            'acceleration': torch.tensor([[acceleration]], dtype=torch.float32, device=self.device),
            'command': torch.from_numpy(cmd_onehot).unsqueeze(0).to(self.device),
        }

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            prediction = self.model(model_input)
        return prediction

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if model_output.pred_future_waypoints is not None:
            waypoints = model_output.pred_future_waypoints[0].float().cpu().numpy()
            waypoints_swapped = np.column_stack([waypoints[:, 1], waypoints[:, 0]])
        else:
            waypoints_swapped = np.zeros((8, 2), dtype=np.float32)
        return {'trajectory': waypoints_swapped}
