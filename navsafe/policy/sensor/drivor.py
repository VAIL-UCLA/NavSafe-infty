"""
Model adapter for DrivoR (Driving with Routing) — ported from BridgeSim.
DrivoR uses DINOv2 with LoRA for image encoding and transformer-based trajectory prediction.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.utils.detections import attach_detections
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS

# Image Normalization Constants (ImageNet RGB)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _keys_not_allowed(keys: Any, allowlist: tuple[str, ...]) -> list[str]:
    """Sorted ``keys`` minus those covered by ``allowlist``.

    An allowlist entry matches a key exactly, or matches every key under it as
    a dotted subtree — ``"trajectory_decoder"`` covers
    ``"trajectory_decoder.layers.0.w"`` but never ``"trajectory_decoder_v2.w"``,
    so an entry cannot swallow a sibling module by string prefix.
    """
    prefixes = tuple(a.removesuffix(".") for a in allowlist)
    return sorted(
        k for k in keys if not any(k == p or k.startswith(p + ".") for p in prefixes)
    )


def _sample_keys(keys: list[str], limit: int = 10) -> str:
    """``limit`` keys, with a count of the remainder — full lists are useless."""
    head = ", ".join(keys[:limit])
    return head + (f", … (+{len(keys) - limit} more)" if len(keys) > limit else "")


def _looks_like_pre_stage2_checkpoint(missing: list[str], unexpected: list[str]) -> bool:
    """True for a checkpoint trained against the half-loaded pre-fix model.

    Signature: it lacks exactly the tensors the corrected config restored — the
    upper scorer layers and the DINOv2 register tokens — and carries nothing
    the corrected model does not want. Recognising this case matters because
    the generic "fix the config" advice would, here, tell the operator to
    reinstate the defect.
    """
    if unexpected:
        return False
    if not missing:
        return False
    return all(
        "scorer_attention.layers." in k or k.endswith("lora_vit.reg_token") for k in missing
    )


def create_drivor_config(
    num_cameras: int = 4,
    image_size: tuple = (1148, 672),
    num_poses: int = 8,
    proposal_num: int = 64,
    use_lidar: bool = False,
) -> DictConfig:
    """Create a default DrivoR configuration."""
    if num_cameras == 4:
        cam_config = {
            "cam_f0": [3],
            "cam_l0": [3],
            "cam_l1": [],
            "cam_l2": [],
            "cam_r0": [3],
            "cam_r1": [],
            "cam_r2": [],
            "cam_b0": [3],
        }
    else:
        cam_config = {
            "cam_f0": [3],
            "cam_l0": [3],
            "cam_l1": [3],
            "cam_l2": [3],
            "cam_r0": [3],
            "cam_r1": [3],
            "cam_r2": [3],
            "cam_b0": [3],
        }

    config_dict = {
        **cam_config,
        "lidar_pc": [3] if use_lidar else [],
        "image_size": list(image_size),
        "lidar_image_size": [256, 256],
        "num_scene_tokens": 16,
        "tf_d_model": 256,
        "tf_d_ffn": 1024,
        "num_poses": num_poses,
        "proposal_num": proposal_num,
        "ref_num": 4,
        "scorer_ref_num": 4,
        "full_history_status": False,
        "one_token_per_traj": True,
        "b2d": True,
        "trajectory_sampling": {
            "num_poses": num_poses,
            "time_horizon": 4.0,
            "interval_length": 0.5,
        },
        # DINOv2 variant: the checkpoint carries a trained
        # `image_backbone.model.lora_vit.reg_token` of shape (1, 4, 384), the
        # registers of the *reg4* DINOv2. Plain `vit_small_patch14_dinov2` has
        # `reg_token = None`, so the encoder ran without them. Registers are
        # prepended after the scene tokens (dinov2_lora.timm_ViT._pos_embed:
        # [scene, cls, reg, patches]) while the encoder returns
        # `tokens[:, :num_scene_tokens]`, so this does not shift the
        # scene-token slice. `lidar_backbone` carries the same wrong name and
        # is corrected with it so enabling lidar cannot resurrect the bug.
        "image_backbone": {
            "model_name": "timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "model_weights": None,
            "use_lora": True,
            "lora_rank": 32,
            "finetune": False,
            "use_feature_pooling": False,
            "focus_front_cam": False,
            "compress_fc": False,
        },
        "lidar_backbone": {
            # Same correction as image_backbone above.
            "model_name": "timm/vit_small_patch14_reg4_dinov2.lvd142m",
            "model_weights": None,
            "use_lora": False,
            "lora_rank": 0,
            "finetune": False,
            "use_feature_pooling": False,
            "focus_front_cam": False,
            "compress_fc": False,
        },
        "lidar_min_x": -32,
        "lidar_max_x": 32,
        "lidar_min_y": -32,
        "lidar_max_y": 32,
        "lidar_max_height": 2.0,
        "lidar_split_height": 0.2,
        "lidar_use_ground_plane": True,
        "lidar_hist_max_per_pixel": 5,
        # Proposal-selection weights for the model's internal pdm_score
        # (drivor_model.py: sum of noc/dac/ddc·log σ + log(ttc·σ+ep·σ+c·σ)).
        # These must match upstream DrivoR inference (valeoai/DrivoR
        # pdm_score config: noc=1, dac=1, ddc=0, ttc=5, ep=5, comfort=2).
        # They previously sat at (5,5,5,2,2,2) — safety sigmoids raised to
        # the 5th power with progress at weight 2 — which made the argmax
        # pick the SHORTEST of the 64 proposals on every frame the sub-1
        # safety sigmoids differed at all: measured on a real rollout frame,
        # the pick was the set minimum (1.8 m of 1.8/7.1/12.6 min/med/max at
        # standstill) at every ego speed, locking closed-loop rollouts into a
        # 0.6-0.75 m/s crawl. With the upstream weights the same frame picks
        # 4.8 m at standstill and 7.1 m at 4 m/s.
        "noc": 1.0,
        "dac": 1.0,
        "ddc": 0.0,
        "ttc": 5.0,
        "ep": 5.0,
        "comfort": 2.0,
        "double_score": False,
        "agent_pred": False,
        "area_pred": False,
        "bev_map": False,
        "bev_agent": False,
        "long_trajectory_additional_poses": 0,
    }
    return OmegaConf.create(config_dict)


@register_policy("drivor")
class DrivoRAdapter(SensorPolicy):
    """
    Adapter for DrivoR model.
    Uses DINOv2 with LoRA for image encoding and transformer decoder
    for multi-proposal trajectory prediction with scoring.
    """

    MODEL_IMAGE_SIZE = (1148, 672)  # (width, height) - fixed to match checkpoint

    CAM_ORDER_4 = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")
    CAM_ORDER_8 = (
        "CAM_F0",
        "CAM_B0",
        "CAM_L0",
        "CAM_L1",
        "CAM_L2",
        "CAM_R0",
        "CAM_R1",
        "CAM_R2",
    )

    # Key mismatches `_load_state_dict_or_raise` tolerates, as exact names or
    # `prefix.` subtrees. The two directions are separate lists because they
    # are not the same risk: an ALLOWED_UNEXPECTED key is a tensor the
    # checkpoint carries and the model ignores (wasteful), while an
    # ALLOWED_MISSING key is a tensor the model USES and the checkpoint does
    # not supply — it runs randomly-initialised weights there, which is the
    # failure this guard exists to close. Both are empty on purpose: the
    # shipped checkpoint loads 0/0 against this config. Adding an entry
    # requires a comment beside it naming the checkpoint that needs it and why
    # that specific mismatch is benign.
    ALLOWED_MISSING_KEYS: tuple[str, ...] = ()
    ALLOWED_UNEXPECTED_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        num_cameras: int = 4,
        image_size: tuple | None = None,
        num_poses: int = 8,
        use_lidar: bool = False,
        scorer=None,
        num_proposals: int | None = None,
        **kwargs,
    ):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.num_cameras = num_cameras
        self.image_size = self.MODEL_IMAGE_SIZE
        self.num_poses = num_poses
        self.use_lidar = use_lidar
        self.scorer = scorer
        self.num_proposals = num_proposals
        self._current_frame_id = 0
        self.config: Any = None

        self.cam_order = list(self.CAM_ORDER_4 if num_cameras == 4 else self.CAM_ORDER_8)

    def load_model(self):
        print("Loading DrivoR model...")
        from navsafe.modelzoo.navsim.drivor.drivor_model import DrivoRModel

        if self.config_path and Path(self.config_path).exists():
            self.config = OmegaConf.load(self.config_path)
        else:
            self.config = create_drivor_config(
                num_cameras=self.num_cameras,
                image_size=self.image_size,
                num_poses=self.num_poses,
                use_lidar=self.use_lidar,
            )

        self.model = DrivoRModel(self.config)

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace("agent._drivor_model.", "").replace("_drivor_model.", "")
            clean_sd[new_key] = v

        self._load_state_dict_or_raise(clean_sd)

        self.model.to(self.device)
        self.model.eval()
        proposal_num = int(self.config.get("proposal_num", 64))
        if self.num_proposals is not None and self.num_proposals > proposal_num:
            # The model decodes a fixed proposal_num; parse_output's [:k]
            # slice silently returns the full set for larger requests.
            print(
                f"num_proposals={self.num_proposals} exceeds DrivoR's "
                f"proposal_num={proposal_num} — candidate consumers will "
                f"see {proposal_num} candidates"
            )
        print("DrivoR model loaded successfully.")

    def _load_state_dict_or_raise(self, clean_sd: Dict[str, Any]) -> None:
        """Load ``clean_sd`` into the model, raising on any key mismatch.

        Raises:
            RuntimeError: if any key is missing from, or unexpected by, the
                constructed model and is not covered by
                :attr:`ALLOWED_MISSING_KEYS` / :attr:`ALLOWED_UNEXPECTED_KEYS`.
                The message names the offending keys and the config keys that
                usually cause them.
        """
        missing_keys, unexpected_keys = self.model.load_state_dict(clean_sd, strict=False)
        blocked_missing = _keys_not_allowed(missing_keys, self.ALLOWED_MISSING_KEYS)
        blocked_unexpected = _keys_not_allowed(unexpected_keys, self.ALLOWED_UNEXPECTED_KEYS)

        if not blocked_missing and not blocked_unexpected:
            waived = (len(missing_keys) - len(blocked_missing)) + (
                len(unexpected_keys) - len(blocked_unexpected)
            )
            note = f", {waived} allowlisted" if waived else ""
            print(
                f"Checkpoint loaded: {len(clean_sd)} tensors, {len(missing_keys)} missing, "
                f"{len(unexpected_keys)} unexpected{note}"
            )
            return

        lines = [
            f"DrivoR checkpoint does not match the constructed model: "
            f"{len(blocked_missing)} missing, {len(blocked_unexpected)} unexpected "
            f"(checkpoint {self.checkpoint_path}).",
        ]
        if blocked_missing:
            lines.append(
                f"  missing (model wants, checkpoint lacks): {_sample_keys(blocked_missing)}"
            )
        if blocked_unexpected:
            lines.append(
                f"  unexpected (checkpoint has, model lacks): {_sample_keys(blocked_unexpected)}"
            )
        lines.append(
            "  strict=False drops these silently, so the model would run with randomly "
            "initialised weights in their place."
        )
        if _looks_like_pre_stage2_checkpoint(blocked_missing, blocked_unexpected):
            lines.append(
                "  DIAGNOSIS: this checkpoint predates the scorer_ref_num 2->4 / reg4-DINOv2 "
                "correction — it was trained against a model that had already discarded those "
                "same tensors, so its weights encode a 2-layer scorer and no register tokens. "
                "Do NOT revert the config to load it: that reinstates the defect. Either "
                "retrain from the corrected base, or, to reproduce a historical number "
                "deliberately, pass --config with a copy of create_drivor_config()'s dict "
                "overriding scorer_ref_num: 2 and both model_name entries to "
                "timm/vit_small_patch14_dinov2.lvd142m — and label the result as pre-fix, "
                "because it is not comparable with anything produced after it."
            )
        else:
            lines.append(
                "  Usual causes: a scorer_ref_num / ref_num that disagrees with the "
                "checkpoint's scorer_attention.layers.* / trajectory_decoder.layers.* depth, "
                "an image_backbone.model_name of the wrong DINOv2 variant (reg4 vs plain — "
                "look for model.lora_vit.reg_token). Fix the config, or add the key to "
                "DrivoRAdapter.ALLOWED_MISSING_KEYS / ALLOWED_UNEXPECTED_KEYS with a comment "
                "saying why the mismatch is benign."
            )
        raise RuntimeError("\n".join(lines))

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Cameras the renderer must produce, in this adapter's slot order.

        Derived from :attr:`cam_order` — the single declaration of the slot
        map — so the render request and the tensor layout cannot disagree.
        Consumers read ``images`` by key, so only the key *set* is load-bearing
        here; the order is carried anyway to keep the two surfaces readable as
        one statement.
        """
        return {k: NAVSIM_CAM_CONFIGS[k] for k in self.cam_order}

    def _preprocess_images(self, images_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        processed_imgs = []
        target_width, target_height = self.image_size

        for cam_name in self.cam_order:
            if cam_name in images_dict:
                img = images_dict[cam_name]
            else:
                img = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            # CameraManager delivers BGR (it swaps the replicator's RGB so the
            # evaluator's artifacts go straight through cv2.imwrite), but
            # IMG_MEAN/IMG_STD below are ImageNet's RGB-ordered constants and
            # DrivoR was trained on navsim frames loaded through PIL as RGB.
            # Forwarding the array unchanged shows the model an orange sky and
            # renders a red traffic light blue -- which inverts the cue a
            # red-light scenario exists to test. Nothing raises; the only
            # symptom is a score that is quietly too low.
            img = renderer_bgr_to_rgb(img)
            # The renderer delivers 1920x1120 where navsim logged 1920x1080,
            # and rows 0..1079 ARE the navsim frame (shared principal point,
            # cy 560), so the extra 40 sit at the bottom. Resizing straight to
            # 1148x672 folds them in and changes the vertical scale relative to
            # the horizontal: 672/1120 against 1148/1920 is a ratio of 1.0035
            # where upstream's 672/1080 gives 1.0407. Measured in the tensor the
            # model receives, that is a 3.6 % vertical squash and a horizon at
            # row 336 instead of 348. Trimming the extra rows first needs no
            # resampling -- same fix as the navsim-stitching adapters, see
            # navsafe/policy/sensor/utils/frames.py. Two table rows (DrivoR,
            # DrivoR + PriorEye).
            img = crop_to_navsim_aspect(img)
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - IMG_MEAN) / IMG_STD
            img = img.transpose(2, 0, 1)
            processed_imgs.append(img)

        return torch.from_numpy(np.stack(processed_imgs)).float()

    def _get_ego_status(self, ego_state: Dict[str, Any], command: int) -> torch.Tensor:
        pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        velocity = ego_state["velocity"][:2]
        heading = ego_state["heading"]
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        # Measured speed, not a floor -- see the note in
        # navsafe/policy/sensor/diffusiondrive.py's _get_status_feature: the
        # 2.0 m/s substitution this replaces hid every genuine standstill from
        # the model. Two table rows here (DrivoR, and DrivoR + PriorEye, which
        # subclasses this adapter).
        if "acceleration" in ego_state:
            acc = ego_state["acceleration"][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0], dtype=np.float32)

        cmd_vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)
        ego_status = np.concatenate([pose, vel_local, acc_local, cmd_vec]).astype(np.float32)

        if self.config.get("full_history_status", False):
            ego_status = np.tile(ego_status, (4, 1))

        return torch.from_numpy(ego_status)

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        self._current_frame_id = frame_id
        camera_feature = self._preprocess_images(images).unsqueeze(0).to(self.device)
        command = ego_state.get("command", 3)
        ego_status = (
            self._get_ego_status(ego_state, command).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        inputs = {"image": camera_feature, "ego_status": ego_status}

        if self.use_lidar:
            lidar_size = self.config.lidar_image_size
            lidar_feature = torch.zeros(
                1, 1, 2, lidar_size[0], lidar_size[1], dtype=torch.float32, device=self.device
            )
            inputs["lidar_feature"] = lidar_feature

        return inputs

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            output = self.model(model_input)
        return output

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.scorer is not None:
            all_proposals = model_output["proposals"]
            if self.num_proposals is not None:
                all_proposals = all_proposals[:, : self.num_proposals]
            scorer_input = {"all_candidates": all_proposals}
            result = self.scorer.select_best(
                scorer_input, ego_state=ego_state, frame_idx=self._current_frame_id
            )
            trajectory = result["trajectory"][0].cpu().numpy()
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            parsed = {
                "trajectory": traj_swapped,
                "best_idx": result["best_idx"][0].item(),
                "num_candidates": all_proposals.shape[1],
            }
            all_cands = all_proposals[0].cpu().numpy()
            cands_swapped = np.stack([np.column_stack([c[:, 1], c[:, 0]]) for c in all_cands])
            parsed["trajectory_coarse"] = cands_swapped
            parsed["coarse_scores"] = result["scores"][0].cpu().numpy()
            return attach_detections(parsed, model_output)

        candidate_keys = {}
        if (
            self.num_proposals is not None
            and "proposals" in model_output
            and "pdm_score" in model_output
        ):
            k = self.num_proposals
            proposals = model_output["proposals"][:, :k]
            pdm_score = model_output["pdm_score"][:, :k]
            selected_idx = torch.argmax(pdm_score, dim=1)
            batch_size = proposals.shape[0]
            trajectory = proposals[torch.arange(batch_size), selected_idx][0].cpu().numpy()
            # Publish bounded candidate confidence while preserving the model's
            # pdm_score argmax. DDC has zero weight in the upstream score;
            # including it here would change the candidate ranking.
            logit = model_output["pred_logit"]
            conf = (
                logit["no_at_fault_collisions"][:, :k].sigmoid()
                * logit["drivable_area_compliance"][:, :k].sigmoid()
                * (
                    5 * logit["time_to_collision_within_bound"][:, :k].sigmoid()
                    + 5 * logit["ego_progress"][:, :k].sigmoid()
                    + 2 * logit["comfort"][:, :k].sigmoid()
                )
                / 12.0
            )
            cands = proposals[0].cpu().numpy()
            # Candidate-set keys for trajectory selection and visualization:
            # trajectory_coarse is (N, T, 2) in the adapter frame
            # [lateral, forward] — same column swap as 'trajectory'.
            candidate_keys = {
                "trajectory_coarse": cands[:, :, [1, 0]],
                "coarse_scores": conf[0].cpu().numpy(),
                "best_idx": int(selected_idx[0].item()),
                "num_candidates": int(cands.shape[0]),
            }
        else:
            trajectory = model_output["trajectory"][0].cpu().numpy()
            selected_idx = model_output.get("token")

        trajectory_xy = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
        heading = trajectory[:, 2] if trajectory.shape[1] > 2 else None

        result = {"trajectory": trajectory_xy, **candidate_keys}
        if heading is not None:
            result["heading"] = heading
        if "proposals" in model_output:
            result["proposals"] = model_output["proposals"][0].cpu().numpy()
        if "pdm_score" in model_output:
            result["pdm_score"] = model_output["pdm_score"][0].cpu().numpy()
        if "proposal_list" in model_output and selected_idx is not None:
            # Trajectory lens: decode the SAME selected proposal at every refinement
            # step's traj_head, tracing how the committed trajectory evolves with depth.
            # `selected_idx` is a valid index into proposal_list's proposal dim in both
            # branches above (num_proposals truncation only slices a *prefix*, so the
            # index still refers to the same original proposal).
            idx = selected_idx[0].item()
            lens = [step[0, idx].cpu().numpy() for step in model_output["proposal_list"]]
            result["trajectory_lens"] = np.stack(
                [np.column_stack([step[:, 1], step[:, 0]]) for step in lens]
            )
        return attach_detections(result, model_output)
