"""NavSafe adapter for the official SimScale GTRS-Dense checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from navsafe.evaluation.utils.constants import DEFAULT_CMD, NAVSIM_CMD_MAPPING
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.utils.frames import crop_to_navsim_aspect, renderer_bgr_to_rgb
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS


@register_policy("gtrs_dense")
class GTRSDenseAdapter(SensorPolicy):
    """Run GTRS-Dense baseline or SimScale checkpoints in closed loop."""

    def __init__(self, checkpoint_path: str, **kwargs: Any):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.backbone = os.environ.get("NAVSAFE_GTRS_BACKBONE", "auto").lower()
        if self.backbone not in {"auto", "resnet", "vov"}:
            raise ValueError(f"NAVSAFE_GTRS_BACKBONE must be auto, resnet or vov, got {self.backbone!r}")

    def load_model(self) -> None:
        from navsafe.modelzoo.gtrs_dense.hydra_config import HydraConfig
        from navsafe.modelzoo.gtrs_dense.hydra_model import HydraModel

        vocab_size = int(os.environ.get("NAVSAFE_GTRS_VOCAB_SIZE", "16384"))
        vocab_path = os.environ.get(
            "NAVSAFE_GTRS_VOCAB",
            str(Path(__file__).resolve().parents[2] / "modelzoo" / "gtrs_dense" / f"{vocab_size}.npy"),
        )
        vocabulary = np.load(vocab_path, allow_pickle=False)
        if vocabulary.shape != (vocab_size, 40, 3) or not np.isfinite(vocabulary).all():
            raise ValueError(
                f"GTRS vocabulary must contain {vocab_size} finite trajectories of shape (40, 3), "
                f"got {vocabulary.shape}"
            )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        raw = checkpoint["state_dict"]
        clean = {}
        for key, value in raw.items():
            if "model._trajectory_head.vocab" in key:
                continue
            prefix = "agent.model."
            clean[key[len(prefix):] if key.startswith(prefix) else key] = value

        inferred_backbone = "vov" if "downscale_layer.weight" in clean else "resnet"
        if self.backbone == "auto":
            self.backbone = inferred_backbone
        elif self.backbone != inferred_backbone:
            raise ValueError(f"Requested {self.backbone} backbone conflicts with checkpoint {inferred_backbone}")

        config = HydraConfig(
            vocab_path=vocab_path,
            vocab_size=vocab_size,
            lidar_seq_len=4,
            sigma=0.5,
            trajectory_imi_weight=1.0,
            progress_weight=2.0,
            vocab_dropout=True,
            normalize_vocab_pos=True,
            camera_width=2048,
            camera_height=512,
            img_vert_anchors=16,
            img_horz_anchors=64,
            backbone_type=self.backbone,
            bkb_path="",
        )

        self.model = HydraModel(config)

        missing, unexpected = self.model.load_state_dict(clean, strict=False)
        allowed_missing = {"_trajectory_head.vocab"}
        bad_missing = set(missing) - allowed_missing
        allowed_unexpected = {"_query_embedding.weight"}
        bad_unexpected = set(unexpected) - allowed_unexpected
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "GTRS-Dense checkpoint mismatch: "
                f"missing={sorted(bad_missing)[:12]}, unexpected={sorted(bad_unexpected)[:12]}"
            )
        self.model.to(self.device).eval()
        print(
            f"GTRS-Dense loaded exactly: backbone={self.backbone}, "
            f"checkpoint={self.checkpoint_path}, vocab={vocab_path}"
        )

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {name: NAVSIM_CAM_CONFIGS[name] for name in ("CAM_F0", "CAM_L0", "CAM_R0")}

    def get_waypoint_dt(self) -> float:
        return 0.1

    @staticmethod
    def _image_or_zeros(image: np.ndarray | None) -> np.ndarray:
        if image is None:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
        return image

    def _camera_feature(self, images: Dict[str, np.ndarray]) -> torch.Tensor:
        left = crop_to_navsim_aspect(renderer_bgr_to_rgb(self._image_or_zeros(images.get("CAM_L0"))))
        front = crop_to_navsim_aspect(renderer_bgr_to_rgb(self._image_or_zeros(images.get("CAM_F0"))))
        right = crop_to_navsim_aspect(renderer_bgr_to_rgb(self._image_or_zeros(images.get("CAM_R0"))))
        height, width = front.shape[:2]
        crop_tb = int(28 * height / 1080)
        crop_lr = int(416 * width / 1920)
        left = left[crop_tb:-crop_tb, crop_lr:-crop_lr]
        front = front[crop_tb:-crop_tb]
        right = right[crop_tb:-crop_tb, crop_lr:-crop_lr]
        stitched = np.concatenate([left, front, right], axis=1)
        resized = cv2.resize(stitched, (2048, 512))
        return torch.from_numpy(resized.transpose(2, 0, 1)).float().div_(255.0)

    @staticmethod
    def _status(ego_state: Dict[str, Any]) -> torch.Tensor:
        command = ego_state.get("command", 3)
        command_vector = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)
        heading = float(ego_state["heading"])
        rotation = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )
        velocity = rotation @ np.asarray(ego_state["velocity"][:2])
        acceleration = rotation @ np.asarray(ego_state.get("acceleration", [0.0, 0.0])[:2])
        value = np.concatenate([command_vector, velocity, acceleration]).astype(np.float32)
        return torch.from_numpy(value)

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Dict[str, Any]:
        camera = self._camera_feature(images).unsqueeze(0).to(self.device)
        status = self._status(ego_state).unsqueeze(0).to(self.device)
        return {"camera_feature": camera, "status_feature": [status]}

    def run_inference(self, model_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with torch.inference_mode():
            return self.model(model_input)

    def parse_output(
        self, model_output: Dict[str, torch.Tensor], ego_state: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        trajectory = model_output["trajectory"][0].detach().cpu().numpy()
        return {"trajectory": np.column_stack([trajectory[:, 1], trajectory[:, 0]])}
