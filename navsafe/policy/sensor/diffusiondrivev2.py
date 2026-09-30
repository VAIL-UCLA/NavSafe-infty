"""
Model adapter for DiffusionDriveV2 — ported from BridgeSim.
DiffusionDriveV2 uses diffusion-based trajectory prediction with scoring module.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import os
from pathlib import Path

import torch
import numpy as np
import cv2
from typing import Dict, Any

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.utils.frames import crop_to_navsim_aspect
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS


@register_policy("diffusiondrivev2")
class DiffusionDriveV2Adapter(SensorPolicy):
    """
    Adapter for DiffusionDriveV2 model.
    Uses diffusion-based trajectory generation with a scoring module for trajectory selection.
    """

    def __init__(self, checkpoint_path: str, plan_anchor_path: str | None = None,
                 enable_temporal_consistency: bool = False,
                 temporal_alpha: float = 1.5, temporal_lambda: float = 0.3,
                 temporal_max_history: int = 8, temporal_sigma: float = 5.0,
                 consensus_temperature: float = 1.0,
                 scorer=None, num_groups: int = 10, num_proposals: int | None = None, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config: Any = None
        self.plan_anchor_path = plan_anchor_path
        self.scorer = scorer
        self.num_groups = num_groups
        self.num_proposals = num_proposals
        self._current_frame_id = 0
        self.enable_temporal_consistency = enable_temporal_consistency
        self.temporal_alpha = temporal_alpha
        self.temporal_lambda = temporal_lambda
        self.temporal_max_history = temporal_max_history
        self.temporal_sigma = temporal_sigma
        self.consensus_temperature = consensus_temperature
        self.current_sim_time = 0.0

    def load_model(self):
        print("Loading DiffusionDriveV2 model...")
        from navsafe.modelzoo.navsim.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
        from navsafe.modelzoo.navsim.diffusiondrivev2.diffusiondrivev2_model_sel import V2TransfuserModel

        self.config = TransfuserConfig()
        if self.plan_anchor_path:
            self.config.plan_anchor_path = self.plan_anchor_path

        self.model = V2TransfuserModel(self.config)

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

        if self.enable_temporal_consistency:
            self.model.set_temporal_params(
                alpha=self.temporal_alpha, lambda_consist=self.temporal_lambda,
                max_history=self.temporal_max_history, sigma=self.temporal_sigma,
                consensus_temperature=self.consensus_temperature)

        print("DiffusionDriveV2 model loaded successfully.")

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

        # The renderer delivers 1920x1120 where navsim logged 1920x1080. The
        # proportional crop below scales with resolution but cannot repair an
        # aspect difference, so the stitch comes out 4096x1062 instead of
        # upstream's 4096x1024 and the resize squashes it 3.7 % vertically.
        # Dropping the 40 extra bottom rows first needs no resampling -- see
        # navsafe/policy/sensor/utils/frames.py.
        #
        # NOTE: this adapter still forwards the renderer's BGR unconverted,
        # unlike its three siblings. That is a separate defect, named in
        # docs/experiments/table2_adapter_coverage.md Sec 3.7 and deliberately
        # not bundled into this change.
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
        resized_image = cv2.resize(stitched_image, (self.config.camera_width, self.config.camera_height))
        self._maybe_dump_camera_input(
            {'CAM_L0': cam_l0, 'CAM_F0': cam_f0, 'CAM_R0': cam_r0},
            stitched_image, resized_image)
        tensor_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
        return tensor_image

    # Dump the exact camera input the model receives (raw cams, stitched
    # panorama, and the resized model tensor) for offline inspection of the
    # render-backend domain gap / side-camera geometry. Enabled by setting
    # NEXUSSIM_DUMP_CAM_INPUT=<dir>; capped so long evals don't fill the disk.
    _CAM_DUMP_LIMIT = 12

    def _maybe_dump_camera_input(self, cams: Dict[str, np.ndarray],
                                 stitched: np.ndarray, resized: np.ndarray) -> None:
        dump_dir = os.environ.get("NEXUSSIM_DUMP_CAM_INPUT")
        if not dump_dir:
            return
        count = getattr(self, "_cam_dump_count", 0)
        if count >= self._CAM_DUMP_LIMIT:
            return
        self._cam_dump_count = count + 1
        out = Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"frame_{self._current_frame_id:05d}"
        for name, img in cams.items():
            cv2.imwrite(str(out / f"{prefix}_{name}.png"), img)
        cv2.imwrite(str(out / f"{prefix}_stitched.png"), stitched)
        cv2.imwrite(str(out / f"{prefix}_model_input.png"), resized)

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

        # Measured speed, not a floor -- see the note in
        # navsafe/policy/sensor/diffusiondrive.py's _get_status_feature: the
        # 2.0 m/s substitution this replaces hid every genuine standstill from
        # the model.
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
                    if output.get("confidence_scores") is not None:
                        output["confidence_scores"] = output["confidence_scores"][:, :k]
                    if output.get("coarse_scores") is not None:
                        output["coarse_scores"] = output["coarse_scores"][:, :k]
            elif self.num_proposals is not None:
                output = self.model.forward_inference_scaling(
                    model_input, num_groups=self.num_groups)
                k = self.num_proposals
                candidates = output["all_candidates"][:, :k]
                scores = output.get("coarse_scores")
                if scores is not None:
                    scores = scores[:, :k]
                    best_idx = torch.argmax(scores, dim=1)
                else:
                    best_idx = torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)
                batch_size = candidates.shape[0]
                output["trajectory"] = candidates[torch.arange(batch_size), best_idx]
            elif self.enable_temporal_consistency:
                output = self.model.forward_temporal(
                    model_input, current_time=self.current_sim_time,
                    targets=None, cal_pdm=False)
            else:
                output = self.model(model_input, targets=None, cal_pdm=False)
        return output

    def set_simulation_time(self, sim_time: float):
        self.current_sim_time = sim_time

    def reset_temporal_history(self):
        if self.enable_temporal_consistency and self.model is not None:
            self.model.reset_temporal_history()
            self.current_sim_time = 0.0

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.scorer is not None:
            scorer_result = self.scorer.select_best(model_output, ego_state=ego_state, frame_idx=self._current_frame_id)
            trajectory = scorer_result["trajectory"][0].cpu().numpy()
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            return {'trajectory': traj_swapped, 'best_idx': scorer_result["best_idx"][0].item(),
                    'num_candidates': model_output["all_candidates"].shape[1]}

        result = {}
        if "trajectory" in model_output:
            trajectory = model_output["trajectory"][0].cpu().numpy()
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            result['trajectory'] = traj_swapped
        else:
            result['trajectory'] = np.zeros((8, 2), dtype=np.float32)

        if "trajectory_candidates" in model_output:
            candidates = model_output["trajectory_candidates"][0].cpu().numpy()
            result['trajectory_candidates'] = np.array([
                np.column_stack([c[:, 1], c[:, 0]]) for c in candidates])

        if "trajectory_topk" in model_output:
            topk = model_output["trajectory_topk"][0].cpu().numpy()
            result['trajectory_topk'] = np.array([np.column_stack([c[:, 1], c[:, 0]]) for c in topk])

        if "topk_scores" in model_output:
            result['topk_scores'] = model_output["topk_scores"][0].cpu().numpy()

        if "trajectory_coarse" in model_output:
            coarse = model_output["trajectory_coarse"][0].cpu().numpy()
            result['trajectory_coarse'] = np.array([np.column_stack([c[:, 1], c[:, 0]]) for c in coarse])
        elif "all_candidates" in model_output:
            # forward_inference_scaling names the candidate set "all_candidates";
            # surface it under the same key so consumers (uncertainty gate)
            # see one candidate-set convention regardless of inference path.
            coarse = model_output["all_candidates"][0].cpu().numpy()
            result['trajectory_coarse'] = np.array([np.column_stack([c[:, 1], c[:, 0]]) for c in coarse])

        if "coarse_scores" in model_output:
            result['coarse_scores'] = model_output["coarse_scores"][0].cpu().numpy()

        return result
