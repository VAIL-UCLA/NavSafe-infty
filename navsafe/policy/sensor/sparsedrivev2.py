# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""SparseDriveV2 adapter — vocabulary-based planner with factorized scoring.

SparseDriveV2 (Sun et al., ``swc-17/SparseDriveV2``) decomposes a trajectory
vocabulary into 1024 geometric paths x 256 velocity profiles, scores paths and
profiles separately, then scores a small set of composed trajectories and takes
the argmax. There is no diffusion head and no autoregressive decode: one
forward pass per replan, ResNet-34 backbone, ~537 MB checkpoint.

Unlike the VLA adapters this runs **in-process** — no subprocess server, no
second venv. The only native dependency is the repo's deformable-aggregation
CUDA op, which builds against this stack (verified against torch 2.10.0+cu128 /
nvcc 12.8) and is JIT-compiled on first use; see
``navsafe/modelzoo/navsim/sparsedrivev2/ops/_build.py``.

Two deviations from the upstream eval path, both forced and both benign:

* **Image geometry.** Upstream reads 1920x1080 OpenScene jpgs; the NuRec/NavSim
  rig here renders 1920x1120 (which is what makes ``cy = 560`` the true optical
  centre). The test-mode resize/crop is recomputed from the *actual* frame size
  with the upstream formula, so the crop still keeps the bottom ``final_dim``
  rows and the same matrix is folded into ``lidar2img``. The practical effect
  is that a little more sky is cropped than upstream.
* **Camera parameters.** Upstream takes intrinsics/extrinsics per frame from
  the NAVSIM ``AgentInput``; here they come from ``OPENSCENE_CAMERA_PARAMS``,
  which is the same sensor rig and is what the renderer is posed to.

The released NAVSIM-v1 checkpoint carries six metric heads
(``no_at_fault_collisions``, ``drivable_area_compliance``,
``driving_direction_compliance``, ``time_to_collision_within_bound``,
``comfort``, ``ego_progress``), so the config is pinned to
``dataset_version="v1"`` — the v2 default expects ``lane_keeping`` and
``history_comfort`` heads this checkpoint does not have, and selects the
trajectory with a different formula.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import torch
from PIL import Image

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import OPENSCENE_CAMERA_PARAMS
from navsafe.policy.sensor.utils.frames import renderer_bgr_to_rgb

# The trajectory/path/velocity vocabularies ship beside the checkpoint in the
# HF repo rather than inside it (they are `nn.Parameter`s with
# `requires_grad=False`, loaded from disk at construction and then overwritten
# by the checkpoint's own copies). Default to the checkpoint's directory.
ANCHOR_ENV = "NAVSAFE_SPARSEDRIVEV2_ANCHORS"
PATH_ANCHOR = "path_1024.npy"
VELOCITY_ANCHOR = "velocity_256.npy"
TRAJECTORY_ANCHOR = "trajectory_1024_256.npz"

# Lightning wraps the model as `agent._sparsedrive_model`.
CKPT_PREFIX = "agent._sparsedrive_model."

# NAVSIM-v1 head set, in the checkpoint's own order.
V1_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
)


@register_policy("sparsedrivev2")
class SparseDriveV2Adapter(SensorPolicy):
    """Adapter for SparseDriveV2 (NAVSIM-v1 release)."""

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config: Any = None
        self._cams: List[str] = []
        self._lidar2img: np.ndarray | None = None

    # -- setup --------------------------------------------------------------

    def _anchor_dir(self) -> Path:
        override = os.environ.get(ANCHOR_ENV)
        if override:
            return Path(override)
        return Path(self.checkpoint_path).resolve().parent

    def load_model(self):
        """Build the model at the checkpoint's configuration and load weights."""
        from navsafe.modelzoo.navsim.sparsedrivev2.config import SparseDriveConfig
        from navsafe.modelzoo.navsim.sparsedrivev2.model import SparseDriveModel

        anchors = self._anchor_dir()
        missing = [f for f in (PATH_ANCHOR, VELOCITY_ANCHOR, TRAJECTORY_ANCHOR)
                   if not (anchors / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"SparseDriveV2: vocabulary files {missing} not found in {anchors}. "
                f"They live beside the checkpoint in the HF repo; set {ANCHOR_ENV} "
                "to point at them.")

        self.config = SparseDriveConfig(
            path_anchor=str(anchors / PATH_ANCHOR),
            velocity_anchor=str(anchors / VELOCITY_ANCHOR),
            trajectory_anchor=str(anchors / TRAJECTORY_ANCHOR),
            dataset_version="v1",
            metrics=V1_METRICS,
            # Grid mask is a training augmentation; upstream leaves the flag on
            # and relies on `module.eval()` to disable it, which this port also
            # does, but turning it off makes that explicit.
            use_grid_mask=False,
        )
        self._cams = [c.upper() for c in self.config.cams]

        print("Loading SparseDriveV2 model...")
        self.model = SparseDriveModel(self.config)

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        clean = {k[len(CKPT_PREFIX):]: v for k, v in state_dict.items()
                 if k.startswith(CKPT_PREFIX)}
        if not clean:  # a re-exported checkpoint may already be unwrapped
            clean = state_dict
        missing_keys, unexpected = self.model.load_state_dict(clean, strict=False)
        # Reported rather than swallowed: a silent partial load here renders as
        # a plausible-but-wrong trajectory, which is the expensive failure.
        print(f"SparseDriveV2 loaded: {len(clean)} tensors, "
              f"{len(missing_keys)} missing, {len(unexpected)} unexpected.")

        self.model.to(self.device)
        self.model.eval()
        self._lidar2img = self._build_lidar2img()

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Three cameras: left, front, right — the upstream ``config.cams``."""
        from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS

        cams = self._cams or [c.upper() for c in ("cam_l0", "cam_f0", "cam_r0")]
        return {k: NAVSIM_CAM_CONFIGS[k] for k in cams}

    # -- geometry -----------------------------------------------------------

    def _build_lidar2img(self) -> np.ndarray:
        """``lidar2img`` per camera, exactly as upstream's ``get_camera_params``.

        Fixed for the whole episode: the rig is rigid, so this is built once at
        load rather than per frame.
        """
        mats = []
        for cam in self._cams:
            params = OPENSCENE_CAMERA_PARAMS[cam]
            rot = np.asarray(params["sensor2lidar_rotation"], dtype=np.float64)
            trans = np.asarray(params["sensor2lidar_translation"], dtype=np.float64)
            intrinsics = np.asarray(params["intrinsics"], dtype=np.float64)

            lidar2cam_r = np.linalg.inv(rot)
            lidar2cam_t = trans @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            viewpad = np.eye(4)
            viewpad[:intrinsics.shape[0], :intrinsics.shape[1]] = intrinsics
            mats.append(viewpad @ lidar2cam_rt.T)
        return np.stack(mats).astype(np.float32)

    def _img_transform(self, img: np.ndarray):
        """Upstream test-mode resize+crop, recomputed for this frame's size.

        Returns the transformed image and the 4x4 matrix that maps original
        pixel coordinates to transformed ones, which is folded into
        ``lidar2img`` so projection stays consistent with the crop.
        """
        fH, fW = self.config.final_dim
        H, W = img.shape[:2]

        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        new_w, new_h = resize_dims
        crop_h = int((1 - float(np.mean(self.config.bot_pct_lim))) * new_h) - fH
        crop_w = int(max(0, new_w - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        pil = Image.fromarray(img.astype(np.uint8)).resize(resize_dims).crop(crop)
        out = np.array(pil).astype(np.float32)

        transform = np.eye(3)
        transform[:2, :2] *= resize
        transform[:2, 2] -= np.array(crop[:2], dtype=np.float64)
        extend = np.eye(4)
        extend[:3, :3] = transform
        return out, extend

    def _preprocess_images(self, images: Dict[str, np.ndarray]):
        """(imgs, projection_mat, image_wh) for the three configured cameras."""
        mean = np.array(self.config.img_mean, dtype=np.float32)
        std = np.array(self.config.img_std, dtype=np.float32)

        imgs, mats, whs = [], [], []
        for i, cam in enumerate(self._cams):
            img = images.get(cam)
            if img is None:
                # A missing camera is a rig/render problem, not something to
                # paper over with zeros: the projection would still claim the
                # view exists and the planner would score against black pixels.
                raise KeyError(
                    f"SparseDriveV2 needs {cam}; got {sorted(images)}. "
                    "get_camera_configs() asks the renderer for all three.")
            # The renderer sends BGR; `to_bgr` says which order the model
            # wants. With the default RGB-ordered img_mean/img_std, forwarding
            # the frame unchanged applied the red mean to the blue channel.
            if not self.config.to_bgr:
                img = renderer_bgr_to_rgb(img)
            transformed, mat = self._img_transform(img)
            transformed = (transformed - mean) / std
            imgs.append(transformed.transpose(2, 0, 1))
            # _lidar2img is built at the end of load_model(), which the adapter
            # contract runs before any prepare_input/_preprocess_images call.
            mats.append(mat.astype(np.float32) @ cast(np.ndarray, self._lidar2img)[i])
            whs.append([transformed.shape[1], transformed.shape[0]])

        imgs_t = torch.from_numpy(np.ascontiguousarray(np.stack(imgs))).float()
        return (imgs_t,
                torch.from_numpy(np.stack(mats)).float(),
                torch.tensor(whs, dtype=torch.float32))

    def _get_status_feature(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        """``[driving_command(4), velocity(2), acceleration(2)]``, ego frame."""
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)

        velocity = np.asarray(ego_state["velocity"], dtype=np.float64)[:2]
        heading = float(ego_state["heading"])
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, s], [-s, c]])
        vel_local = rot @ velocity

        acc = np.asarray(ego_state.get("acceleration", [0.0, 0.0]), dtype=np.float64)[:2]
        acc_local = rot @ acc

        status = np.concatenate([cmd_vec, vel_local, acc_local]).astype(np.float32)
        return torch.from_numpy(status)

    # -- eval-loop contract -------------------------------------------------

    def prepare_input(self,
                      images: Dict[str, np.ndarray],
                      ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any],
                      frame_id: int) -> Any:
        if self.config is None:
            raise RuntimeError("SparseDriveV2Adapter: call load_model() first")

        imgs, projection_mat, image_wh = self._preprocess_images(images)
        status = self._get_status_feature(ego_state, ego_state.get("command", 3))
        return {
            "camera_feature": {
                "imgs": imgs.unsqueeze(0).to(self.device),
                "projection_mat": projection_mat.unsqueeze(0).to(self.device),
                "image_wh": image_wh.unsqueeze(0).to(self.device),
            },
            "status_feature": status.unsqueeze(0).to(self.device),
        }

    def run_inference(self, model_input: Any) -> Any:
        if self.model is None:
            raise RuntimeError("SparseDriveV2Adapter: call load_model() first")
        with torch.no_grad():
            # The module returns (output, loss_dict); the second is empty
            # outside training but is still part of the signature.
            output, _ = self.model(model_input, targets={})
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """(x forward, y left, heading) -> the eval loop's (lateral, forward)."""
        trajectory = model_output["trajectory"][0].cpu().numpy()
        return {"trajectory": np.column_stack([trajectory[:, 1], trajectory[:, 0]])}
