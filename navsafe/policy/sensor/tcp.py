"""
Model adapter for TCP (Trajectory-guided Control Prediction) — ported from BridgeSim.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from collections import OrderedDict
from torchvision import transforms as T

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy


@register_policy("tcp")
class TCPAdapter(SensorPolicy):
    """
    Adapter for TCP model.
    TCP uses only the front camera (CAM_FRONT) and predicts waypoints + control signals.
    """

    def __init__(self, checkpoint_path: str, planner_type: str = "only_traj", config_path: str | None = None, **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.planner_type = planner_type
        self.config = None
        self.img_normalize: Any = None

    def load_model(self):
        print(f"Loading TCP model with planner_type={self.planner_type}...")
        try:
            from navsafe.modelzoo.bench2drive.tcp.model import TCP
            from navsafe.modelzoo.bench2drive.tcp.config import GlobalConfig
        except ImportError as e:
            raise ImportError(
                f"TCP model requires its model architecture files. "
                f"Ensure navsafe.modelzoo.bench2drive.tcp is properly set up.\n"
                f"Original error: {e}"
            ) from e

        self.config = GlobalConfig()
        self.model = TCP(self.config)

        ckpt = torch.load(self.checkpoint_path, map_location="cuda")
        state_dict = ckpt["state_dict"]

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.cuda()
        self.model.eval()

        self.img_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print(f"TCP model loaded successfully (pred_len={self.config.pred_len})")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {
            'CAM_FRONT': {'x': 0.80, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_LEFT': {'x': 0.27, 'y': -0.55, 'z': 1.60, 'yaw': -55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
            'CAM_FRONT_RIGHT': {'x': 0.27, 'y': 0.55, 'z': 1.60, 'yaw': 55.0, 'pitch': 0.0, 'roll': 0.0, 'fov': 70, 'width': 1600, 'height': 900},
        }

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        img = images['CAM_FRONT']
        img_rgb = img[:, :, ::-1].copy()
        img_resized = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_resized = torch.nn.functional.interpolate(
            img_resized.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
        ).squeeze(0)
        img_normalized = self.img_normalize(img_resized)
        img_tensor = img_normalized.unsqueeze(0).cuda()

        delta_world = ego_state['waypoint'] - ego_state['position'][:2]
        cos_h = np.cos(-ego_state['heading'])
        sin_h = np.sin(-ego_state['heading'])
        target_ego_x = cos_h * delta_world[0] - sin_h * delta_world[1]
        target_ego_y = sin_h * delta_world[0] + cos_h * delta_world[1]
        target_point = torch.FloatTensor([[target_ego_x, target_ego_y]]).cuda()

        speed = np.linalg.norm(ego_state['velocity'])
        speed_normalized = torch.FloatTensor([[speed / 12.0]]).cuda()

        command = ego_state['command']
        if command < 0:
            command = 3
        cmd_one_hot = torch.zeros(1, 6).cuda()
        cmd_one_hot[0, command] = 1.0

        state = torch.cat([speed_normalized, target_point, cmd_one_hot], 1)
        return {'img': img_tensor, 'state': state, 'target_point': target_point}

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            pred = self.model(model_input['img'], model_input['state'], model_input['target_point'])

        PIXELS_PER_METER = 5.5
        pred_wp_meters = pred['pred_wp'] / PIXELS_PER_METER
        return {'pred_wp': pred_wp_meters, 'pred_speed': pred['pred_speed'], 'action_index': pred['action_index']}

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        pred_wp = model_output['pred_wp']
        plan_traj = pred_wp[0].cpu().numpy()
        return {'trajectory': plan_traj}

    def get_trajectory_time_horizon(self) -> float:
        return 2.0
