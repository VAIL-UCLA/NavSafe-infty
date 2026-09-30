"""
Model adapter for DiffusionDrive — ported from BridgeSim.
DiffusionDrive uses diffusion-based trajectory prediction with TransFuser backbone.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.

This one adapter serves three benchmark-table rows, because the two
augmentation-based ones change *training* and not inference:

* **DiffusionDrive** — the authors' 88.1-PDMS release.
* **DiffusionDrive (BeyondDrive)** — hard-negative training (Wang et al., ECCV
  2026). Measured: its checkpoint matches this model 763/763, 0 missing and 0
  unexpected, so it is a checkpoint swap and nothing else.
* **DiffusionDrive (SimScale)** — sim-real co-training (Tian et al.). Same,
  with ONE difference that is not cosmetic: SimScale trains from the
  ``config.latent`` fork of the backbone, where the lidar branch is fed a
  learned ``_backbone.lidar_latent`` parameter instead of a lidar BEV. Loading
  that checkpoint against ``latent=False`` silently discards the tensor and
  runs the branch on the adapter's zero BEV — a different model from the one
  the authors measured. :meth:`DiffusionDriveAdapter.load_model` therefore
  detects the tensor and switches the config, which makes the load exact
  (764/764) and the topology theirs.
"""

import torch
import numpy as np
import cv2
from typing import Dict, Any

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)
from navsafe.policy.sensor.utils.state_dict import assert_state_dict_matches
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS

#: The tensor whose presence identifies a ``config.latent`` checkpoint.
LIDAR_LATENT_KEY = "_backbone.lidar_latent"


@register_policy("diffusiondrive")
class DiffusionDriveAdapter(SensorPolicy):
    """
    Adapter for DiffusionDrive model.
    Uses diffusion-based trajectory generation with TransFuser backbone.
    """

    #: Key mismatches tolerated by the load precondition, as exact names or
    #: ``prefix.`` subtrees. Empty on purpose: all three checkpoints this
    #: adapter serves load with 0 missing and 0 unexpected once ``latent`` is
    #: set correctly, so an entry here would be hiding a real mismatch.
    ALLOWED_MISSING_KEYS: tuple[str, ...] = ()
    ALLOWED_UNEXPECTED_KEYS: tuple[str, ...] = ()

    def __init__(self, checkpoint_path: str, plan_anchor_path: str | None = None,
                 scorer=None, num_groups: int = 1, num_proposals: int | None = None,
                 latent: bool | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config: Any = None
        self.plan_anchor_path = plan_anchor_path
        self.scorer = scorer
        self.num_groups = num_groups
        self.num_proposals = num_proposals
        #: ``None`` = read the topology off the checkpoint (see the module
        #: note); True/False forces it, which is what a test wants.
        self.latent = latent
        self._current_frame_id = 0

    def load_model(self):
        print("Loading DiffusionDrive model...")
        from navsafe.modelzoo.navsim.diffusiondrive.transfuser_config import TransfuserConfig
        from navsafe.modelzoo.navsim.diffusiondrive.transfuser_model_v2 import V2TransfuserModel

        self.config = TransfuserConfig()
        # Resolve the kmeans anchor file: explicit override wins, else
        # look for ``kmeans_navsim_traj_20.npy`` next to the checkpoint.
        # Without this, callers that don't pass ``--plan-anchor-path``
        # hit ``FileNotFoundError: ''`` because the config default is "".
        anchor = self.plan_anchor_path
        if not anchor:
            from pathlib import Path
            candidate = (
                Path(self.checkpoint_path).resolve().parent / "kmeans_navsim_traj_20.npy"
            )
            if candidate.is_file():
                anchor = str(candidate)
                print(f"[DiffusionDriveAdapter] Auto-detected plan anchor: {anchor}")
        if anchor:
            self.config.plan_anchor_path = anchor

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._transfuser_model.', '').replace('_transfuser_model.', '')
            clean_sd[new_key] = v

        # Topology before construction: a latent-lidar checkpoint needs the
        # parameter to exist on the model, so this cannot be fixed after the
        # fact by the load.
        latent = self.latent
        if latent is None:
            latent = LIDAR_LATENT_KEY in clean_sd
            if latent:
                print(
                    "[DiffusionDriveAdapter] Checkpoint carries "
                    f"{LIDAR_LATENT_KEY} — enabling the latent-lidar backbone "
                    "(SimScale-style co-training checkpoint)."
                )
        self.config.latent = bool(latent)

        self.model = V2TransfuserModel(self.config)
        assert_state_dict_matches(
            self.model,
            clean_sd,
            model_name="DiffusionDrive",
            checkpoint_path=str(self.checkpoint_path),
            allowed_missing=self.ALLOWED_MISSING_KEYS,
            allowed_unexpected=self.ALLOWED_UNEXPECTED_KEYS,
            hint=(
                f"A single {LIDAR_LATENT_KEY} mismatch is the latent-lidar "
                "topology; pass latent=True/False to force it. A mismatch on "
                "the diffusion head usually means the wrong kmeans anchor."
            ),
        )
        self.model.to(self.device)
        self.model.eval()

        print("DiffusionDrive model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_L0', 'CAM_R0') if k in NAVSIM_CAM_CONFIGS}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        cam_l0 = images_dict.get('CAM_L0')
        cam_f0 = images_dict.get('CAM_F0')
        cam_r0 = images_dict.get('CAM_R0')

        dummy_h, dummy_w = 1080, 1920
        if cam_l0 is None: cam_l0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_f0 is None: cam_f0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
        if cam_r0 is None: cam_r0 = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)

        # CameraManager delivers BGR; every DiffusionDrive release was trained
        # on RGB (navsim loads with PIL). Without this the model sees a red
        # light as blue and nothing raises — see
        # navsafe/policy/sensor/utils/frames.py. Measured on real rendered
        # frames: the swap moves the final planned waypoint 1.03 m
        # (BeyondDrive) / 1.29 m (SimScale) against a 0.17-0.21 m repeat-noise
        # floor, so it is not rounding.
        cam_l0 = renderer_bgr_to_rgb(cam_l0)
        cam_f0 = renderer_bgr_to_rgb(cam_f0)
        cam_r0 = renderer_bgr_to_rgb(cam_r0)

        # The renderer delivers 1920x1120 where navsim logged 1920x1080. The
        # proportional crop below scales with resolution but cannot repair an
        # aspect difference, so the stitch comes out 4096x1062 instead of
        # upstream's 4096x1024 and the fixed 1024x256 resize squashes it 3.7 %
        # vertically. Dropping the 40 extra bottom rows first needs no
        # resampling -- see navsafe/policy/sensor/utils/frames.py. This is one
        # adapter and three table rows (base, SimScale, BeyondDrive).
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
        tensor_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
        return tensor_image

    def _create_lidar_bev(self) -> torch.Tensor:
        return torch.zeros(self.config.lidar_seq_len, self.config.lidar_resolution_height,
                          self.config.lidar_resolution_width, dtype=torch.float32)

    def _get_status_feature(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)
        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # The ego's own speed is reported as measured. A previous version
        # substituted 2.0 m/s whenever |v| < 0.5, which meant the model could
        # never observe a stopped ego: ego_dynamics clips speed to >= 0, so the
        # substitution fired on every genuine halt and told the planner it was
        # rolling at 2 m/s. That inverts precisely the scenarios a closed-loop
        # benchmark stops for -- a red light, a yield, the back of a queue --
        # and it had no counterpart upstream (navsim passes
        # ego_status.ego_velocity through). It also made this adapter's status
        # vector incomparable with ltf/transfuser/sparsedrivev2, which never
        # did it. One adapter, three table rows.
        if 'acceleration' in ego_state:
            acc_local = R @ ego_state['acceleration'][:2]
        else:
            acc_local = np.array([0.0, 0.0])

        status = np.concatenate([cmd_vec, vel_local, acc_local]).astype(np.float32)
        return torch.from_numpy(status)

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        self._current_frame_id = frame_id
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)
        lidar_feature = self._create_lidar_bev().unsqueeze(0).to(self.device)
        command = ego_state.get('command', 3)
        status_feature = self._get_status_feature(ego_state, command).unsqueeze(0).to(self.device)
        return {"camera_feature": camera_feature, "lidar_feature": lidar_feature, "status_feature": status_feature}

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            if self.scorer is not None:
                output = self.model.forward_inference_scaling(
                    model_input, num_groups=self.num_groups)
                if self.num_proposals is not None:
                    k = self.num_proposals
                    output["all_candidates"] = output["all_candidates"][:, :k]
                    if output["confidence_scores"] is not None:
                        output["confidence_scores"] = output["confidence_scores"][:, :k]
            elif self.num_proposals is not None:
                output = self.model.forward_inference_scaling(
                    model_input, num_groups=self.num_groups)
                k = self.num_proposals
                candidates = output["all_candidates"][:, :k]
                scores = output["confidence_scores"][:, :k] if output["confidence_scores"] is not None else None
                if scores is not None:
                    best_idx = torch.argmax(scores, dim=1)
                else:
                    best_idx = torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)
                batch_size = candidates.shape[0]
                output["trajectory"] = candidates[torch.arange(batch_size), best_idx]
            else:
                output = self.model(model_input, targets=None)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.scorer is not None:
            result = self.scorer.select_best(model_output, ego_state=ego_state, frame_idx=self._current_frame_id)
            trajectory = result["trajectory"][0].cpu().numpy()
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            parsed = {'trajectory': traj_swapped, 'best_idx': result["best_idx"][0].item(),
                      'num_candidates': model_output["all_candidates"].shape[1]}
            all_cands = model_output["all_candidates"][0].cpu().numpy()
            cands_swapped = np.stack([np.column_stack([c[:, 1], c[:, 0]]) for c in all_cands])
            parsed['trajectory_coarse'] = cands_swapped
            parsed['coarse_scores'] = result["scores"][0].cpu().numpy()
            return parsed

        trajectory = model_output["trajectory"][0].cpu().numpy()
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
        return {'trajectory': traj_swapped}
