"""
Model adapter for TransFuser — ported from BridgeSim.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import torch
import numpy as np
import cv2
from typing import Dict, Any

from navsafe.modelzoo.navsim.transfuser.transfuser_model import TransfuserModel
from navsafe.modelzoo.common.trajectory_sampling import TrajectorySampling
from navsafe.modelzoo.navsim.transfuser.transfuser_config import TransfuserConfig

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS


@register_policy("transfuser")
class TransfuserAdapter(SensorPolicy):
    """
    Adapter for TransFuser model.
    TransFuser uses multi-camera images and LiDAR for BEV representation.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config: Any = None
        self.trajectory_sampling: Any = None

    def load_model(self):
        """Load TransFuser model from checkpoint."""
        print("Loading TransFuser model...")

        self.config = TransfuserConfig()
        self.trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

        self.model = TransfuserModel(self.trajectory_sampling, self.config)

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._transfuser_model.', '').replace('_transfuser_model.', '')
            clean_sd[new_key] = v

        self.model.load_state_dict(clean_sd, strict=False)
        self.model.to(self.device)
        self.model.eval()

        print("TransFuser model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """TransFuser uses 3 cameras (left, front, right) stitched together."""
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_L0', 'CAM_R0')}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for TransFuser model.
        TransFuser expects a stitched panoramic image (left + front + right) resized to 1024x256.
        """
        cam_l0 = images_dict.get('CAM_L0')
        cam_f0 = images_dict.get('CAM_F0')
        cam_r0 = images_dict.get('CAM_R0')

        dummy_h, dummy_w = 1080, 1920
        if cam_l0 is None:
            cam_l0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_f0 is None:
            cam_f0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_r0 is None:
            cam_r0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)

        # CameraManager delivers BGR; this checkpoint is an upstream NAVSIM
        # release trained on RGB (navsim loads with PIL). Without the swap the
        # model sees a red light as blue and nothing raises — see
        # navsafe/policy/sensor/utils/frames.py, and the measured cost on the
        # sibling DiffusionDrive adapter in
        # docs/experiments/table2_adapter_coverage.md Sec 3.7.
        cam_l0 = renderer_bgr_to_rgb(cam_l0)
        cam_f0 = renderer_bgr_to_rgb(cam_f0)
        cam_r0 = renderer_bgr_to_rgb(cam_r0)

        # The renderer delivers 1920x1120 where navsim logged 1920x1080. The
        # proportional crop below scales with resolution but cannot repair an
        # aspect difference, so the stitch comes out 4096x1062 instead of
        # upstream's 4096x1024 and the fixed 1024x256 resize squashes it 3.7 %
        # vertically. Dropping the 40 extra bottom rows first needs no
        # resampling -- see navsafe/policy/sensor/utils/frames.py.
        cam_l0 = crop_to_navsim_aspect(cam_l0)
        cam_f0 = crop_to_navsim_aspect(cam_f0)
        cam_r0 = crop_to_navsim_aspect(cam_r0)

        h, w = cam_f0.shape[:2]
        crop_tb = int(28 * h / 1080)
        crop_lr = int(416 * w / 1920)

        l0_cropped = cam_l0[crop_tb:-crop_tb, crop_lr:-crop_lr] if crop_tb > 0 else cam_l0[:, crop_lr:-crop_lr]
        f0_cropped = cam_f0[crop_tb:-crop_tb] if crop_tb > 0 else cam_f0
        r0_cropped = cam_r0[crop_tb:-crop_tb, crop_lr:-crop_lr] if crop_tb > 0 else cam_r0[:, crop_lr:-crop_lr]

        stitched_image = np.concatenate([l0_cropped, f0_cropped, r0_cropped], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))

        tensor_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float()
        tensor_image = tensor_image / 255.0

        return tensor_image

    def _create_lidar_bev(self) -> torch.Tensor:
        """Create dummy LiDAR BEV representation."""
        lidar_bev = torch.zeros(
            self.config.lidar_seq_len,
            self.config.lidar_resolution_height,
            self.config.lidar_resolution_width,
            dtype=torch.float32
        )
        return lidar_bev

    def _get_status_feature(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        """Create status feature tensor: [command(4), velocity(2), acceleration(2)]."""
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        if 'acceleration' in ego_state:
            acc = ego_state['acceleration'][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0])

        status = np.concatenate([cmd_vec, vel_local, acc_local]).astype(np.float32)
        return torch.from_numpy(status)

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for TransFuser model."""
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)
        lidar_feature = self._create_lidar_bev().unsqueeze(0).to(self.device)

        command = ego_state.get('command', 3)
        status_feature = self._get_status_feature(ego_state, command).unsqueeze(0).to(self.device)

        return {
            "camera_feature": camera_feature,
            "lidar_feature": lidar_feature,
            "status_feature": status_feature,
        }

    def run_inference(self, model_input: Any) -> Any:
        """Run TransFuser model inference."""
        with torch.no_grad():
            output = self.model(model_input)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse TransFuser output."""
        trajectory = model_output["trajectory"][0].cpu().numpy()  # (T, 3) -> (x, y, heading)
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
        return {'trajectory': traj_swapped}
