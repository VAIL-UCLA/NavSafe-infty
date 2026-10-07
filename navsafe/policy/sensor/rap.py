"""
Model adapter for RAP (Rasterized Autonomous Planner) — ported from BridgeSim.
RAP uses BEVFormer with DINO backbone. Requires mmcv for full model loading.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.utils.frames import renderer_bgr_to_rgb
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import (
    NAVSIM_CAMERA_PARAMS as camera_params,
    convert_camera_params_to_simple_format,
)

# Monkey patch for PyTorch-transformers compatibility
if not hasattr(torch.utils._pytree, "register_pytree_node"):
    _original_register = torch.utils._pytree._register_pytree_node

    def _register_pytree_node_wrapper(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None):
        return _original_register(cls, flatten_fn, unflatten_fn)

    # Deliberate monkeypatch: the wrapper's signature is a superset of torch's,
    # so mypy's assignment check against the original symbol does not apply.
    torch.utils._pytree.register_pytree_node = _register_pytree_node_wrapper  # type: ignore[assignment]

# RAP Model expects images in this order
CAM_ORDER = ['CAM_B0', 'CAM_F0', 'CAM_L0', 'CAM_R0']

# Image Normalization Constants (RGB)
IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(1, 3, 1, 1)
IMG_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(1, 3, 1, 1)
IMG_SCALE = 0.4


@register_policy("rap")
class RAPAdapter(SensorPolicy):
    """
    Adapter for RAP model.
    RAP uses BEVFormer with DINO backbone for multi-camera 3D perception.
    """

    # Lazily populated by ``prepare_input`` (guarded by ``hasattr``); declared
    # here so mypy knows its type without creating the attribute at runtime.
    _prev_velocity: np.ndarray

    def __init__(self, checkpoint_path: str, image_source: str = "metadrive",
                 scorer=None, num_proposals: int | None = None, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.image_source = image_source
        self.scorer = scorer
        self.num_proposals = num_proposals
        self._current_frame_id = 0
        self.config = None
        self.lidar2img_tensor = None
        self.img_shape_tensor = None

    def load_model(self):
        print(f"Loading RAP model (image_source={self.image_source})...")
        try:
            from navsafe.modelzoo.navsim.rap_dino.rap_model import RAPModel
            from navsafe.modelzoo.navsim.rap_dino.navsim_config import RAPConfig
            from navsafe.modelzoo.common.trajectory_sampling import TrajectorySampling
        except ImportError as e:
            raise ImportError(
                f"RAP model requires mmcv and related dependencies. "
                f"Install them with: pip install mmcv-full mmdet mmengine\n"
                f"Original error: {e}"
            ) from e

        self.config = RAPConfig()
        self.config.b2d = False
        self.config.num_poses = 10
        self.config.trajectory_sampling = TrajectorySampling(time_horizon=5.0, interval_length=0.5)

        self.model = RAPModel(self.config).to(self.device)
        self.model.progress = 0.0
        self.model.batch_size = 1

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {k.replace('agent._rap_model.', '').replace('_rap_model.', ''): v
                   for k, v in state_dict.items()}

        # Fix shape mismatches between DINOv3 checkpoint and standard Dinov2Model
        model_sd = self.model.state_dict()
        for key in list(clean_sd.keys()):
            if key in model_sd and clean_sd[key].shape != model_sd[key].shape:
                try:
                    clean_sd[key] = clean_sd[key].reshape(model_sd[key].shape)
                except RuntimeError:
                    print(f"  Warning: dropping {key} (shape {clean_sd[key].shape} vs {model_sd[key].shape})")
                    del clean_sd[key]

        self.model.load_state_dict(clean_sd, strict=False)
        self.model.eval()

        self._setup_calibration_features()
        print("RAP model loaded successfully.")

    def _setup_calibration_features(self):
        lidar2imgs = []
        img_shapes = []

        S = np.eye(4, dtype=np.float32)
        S[0, 0] = IMG_SCALE
        S[1, 1] = IMG_SCALE

        for cam_name in CAM_ORDER:
            params = camera_params[cam_name]
            s2l = np.eye(4, dtype=np.float32)
            s2l[:3, :3] = params['sensor2lidar_rotation']
            s2l[:3, 3] = params['sensor2lidar_translation']
            l2s = np.linalg.inv(s2l)
            K = np.eye(4, dtype=np.float32)
            K[:3, :3] = params['intrinsics']
            l2i = S @ K @ l2s
            lidar2imgs.append(l2i)
            h = int(1120 * IMG_SCALE)
            w = int(1920 * IMG_SCALE)
            img_shapes.append((h, w, 3))

        self.lidar2img_tensor = torch.from_numpy(np.stack(lidar2imgs)).float().unsqueeze(0).to(self.device)
        self.img_shape_tensor = torch.from_numpy(np.stack(img_shapes)).float().unsqueeze(0).to(self.device)

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        if self.image_source == "metadrive":
            return convert_camera_params_to_simple_format(
                camera_params, image_width=1920, image_height=1120, to_metadrive=True)
        return {}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        processed_imgs = []
        for cam_name in CAM_ORDER:
            img = images_dict.get(cam_name)
            if img is None:
                img = np.zeros((int(1120 * IMG_SCALE), int(1920 * IMG_SCALE), 3), dtype=np.uint8)
            else:
                # CameraManager delivers BGR; RAP wants RGB. Its own data
                # pipeline settles it — navsafe/modelzoo/navsim/rap_dino/
                # bevformer/bev_feature_build.py reads navsim's
                # ``cameras.cam_*.image`` (navsim builds that with
                # ``np.array(Image.open(...))``, so RGB) and normalises it with
                # ``NormalizeMultiviewImage(..., to_rgb=False)``, i.e. no swap
                # on already-RGB data, against the same IMG_MEAN/IMG_STD this
                # file repeats. Forwarding the renderer's array applied the red
                # mean to the blue channel and showed the model a red light as
                # blue. Nothing raises; see
                # navsafe/policy/sensor/utils/frames.py.
                img = renderer_bgr_to_rgb(img)
                h, w = img.shape[:2]
                new_h, new_w = int(h * IMG_SCALE), int(w * IMG_SCALE)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)
            processed_imgs.append(img)

        img_tensor = torch.from_numpy(np.stack(processed_imgs)).float()
        img_tensor = (img_tensor - IMG_MEAN.to(img_tensor.device)) / IMG_STD.to(img_tensor.device)

        _, _, h, w = img_tensor.shape
        pad_h = int(np.ceil(h / 32)) * 32
        pad_w = int(np.ceil(w / 32)) * 32
        if pad_h != h or pad_w != w:
            padding = torch.nn.ZeroPad2d((0, pad_w - w, 0, pad_h - h))
            img_tensor = padding(img_tensor)

        return img_tensor

    def _get_ego_status(self, velocity, acceleration, command):
        pose = torch.zeros(3, dtype=torch.float32)
        vel = torch.tensor(velocity, dtype=torch.float32)
        acc = torch.tensor(acceleration, dtype=torch.float32)
        cmd = torch.tensor(command, dtype=torch.float32)
        return torch.cat([pose, vel, acc, cmd], dim=0)

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        self._current_frame_id = frame_id
        img_tensor = self._preprocess_images(images).unsqueeze(0).to(self.device)

        velocity = ego_state['velocity']
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity[:2]

        if hasattr(self, '_prev_velocity'):
            acc_global = (velocity[:2] - self._prev_velocity[:2]) / 0.1
            acc_local = R @ acc_global
        else:
            acc_local = np.array([0.0, 0.0])
        self._prev_velocity = velocity

        command = ego_state.get('command', 3)  # default: LANE_FOLLOW
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        curr_status = self._get_ego_status(vel_local, acc_local, cmd_vec)
        ego_status_tensor = torch.stack([curr_status] * 4).unsqueeze(0).to(self.device)

        return {
            "camera_feature": img_tensor, "ego_status": ego_status_tensor,
            "camera_valid": torch.tensor([True]).to(self.device),
            "lidar2img": self.lidar2img_tensor, "img_shape": self.img_shape_tensor,
        }

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            if self.scorer is not None:
                output = self.model(model_input, targets=None, return_score=True)
                if self.num_proposals is not None:
                    k = self.num_proposals
                    output["trajectory"] = output["trajectory"][:, :k]
                    output["score"] = output["score"][:, :k]
            elif self.num_proposals is not None:
                output = self.model(model_input, targets=None, return_score=True)
                k = self.num_proposals
                proposals = output["trajectory"][:, :k]
                scores = output["score"][:, :k]
                best_idx = torch.argmax(scores, dim=1)
                batch_size = proposals.shape[0]
                output["trajectory"] = proposals[torch.arange(batch_size), best_idx]
            else:
                output = self.model(model_input, targets=None)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.scorer is not None:
            all_proposals = model_output["trajectory"]
            scorer_input = {"all_candidates": all_proposals}
            result = self.scorer.select_best(scorer_input, ego_state=ego_state, frame_idx=self._current_frame_id)
            trajectory = result["trajectory"][0].cpu().numpy()
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            parsed = {'trajectory': traj_swapped, 'best_idx': result["best_idx"][0].item(),
                      'num_candidates': all_proposals.shape[1]}
            all_cands = all_proposals[0].cpu().numpy()
            cands_swapped = np.stack([np.column_stack([c[:, 1], c[:, 0]]) for c in all_cands])
            parsed['trajectory_coarse'] = cands_swapped
            parsed['coarse_scores'] = result["scores"][0].cpu().numpy()
            return parsed

        pred_traj_full = model_output["trajectory"][0].cpu().numpy()
        if pred_traj_full.ndim == 2 and pred_traj_full.shape[1] >= 2:
            pred_traj_local = np.column_stack([pred_traj_full[:, 1], pred_traj_full[:, 0]])
        else:
            pred_traj_local = pred_traj_full
        return {'trajectory': pred_traj_local}

    def get_trajectory_time_horizon(self) -> float:
        return 5.0
