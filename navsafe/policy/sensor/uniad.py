"""UniAD — a :class:`SensorPolicy` adapter (split from ``uniad_vad_adapter.py``).

Phase 3 task 3.7 of the NavSafe Package Reorg spec splits the
pre-reorg single-file ``uniad_vad_adapter.py`` into two modules
under :mod:`navsafe.policy`: this file (UniAD) and
:mod:`navsafe.policy.vad` (VAD). Both adapters historically share
identical mmcv-based plumbing — only ``model_type`` and a few
trajectory-parsing branches differ — so the shared implementation
remains the :class:`UniADVADAdapter` class defined here, and
:mod:`navsafe.policy.vad` thinly subclasses it.

Backs:

* Requirement 3.3 — every first-party adapter inherits from exactly
  one of :class:`StatePolicy` / :class:`SensorPolicy`.
* Requirement 3.5 — UniAD is one of the 12 first-party
  sensor-modality adapters and inherits from
  :class:`SensorPolicy`.
* Requirement 3.7 — base-class choice is determined by what
  :meth:`prepare_input` consumes; UniAD consumes 6-camera
  Bench2Drive imagery, hence ``SensorPolicy``.
* Requirement 10.5 — :meth:`load_model`, :meth:`prepare_input`,
  :meth:`run_inference`, :meth:`parse_output` semantics are
  preserved verbatim from the pre-reorg ``UniADVADAdapter``.

The :func:`register_policy` decorator wires :class:`UniADAdapter`
into the process-wide registry under the name ``"uniad"`` so the
CLI (``navsafe eval --model-type uniad``) and the public API
resolve the same class against the same registry (Requirement 4.6).
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.utils.camera_utils import (
    BENCH2DRIVE_CAM_NAMES as CAM_NAMES,
    BENCH2DRIVE_LIDAR2IMG as LIDAR2IMG,
    BENCH2DRIVE_LIDAR2CAM as LIDAR2CAM,
    BENCH2DRIVE_LIDAR2EGO as LIDAR2EGO,
)


def _navsafe_package_root() -> Path:
    """The inner ``navsafe`` package directory, the one holding ``modelzoo/``.

    Resolved by walking up until the directory is named ``navsafe`` rather
    than by a fixed number of ``.parent`` hops, so moving this module between
    ``navsafe/policy/`` and ``navsafe/policy/sensor/`` cannot silently change
    where it looks. A hop count did exactly that once.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "navsafe":
            return parent
    raise RuntimeError(f"no 'navsafe' package directory above {here}")


# Check for mmcv availability
try:
    from pyquaternion import Quaternion
    _HAS_PYQUATERNION = True
except ImportError:
    _HAS_PYQUATERNION = False


class UniADVADAdapter(SensorPolicy):
    """
    Adapter for UniAD and VAD models.
    These models use mmcv-based architecture and inference pipeline.
    Requires mmcv, mmdet, mmengine to be installed.

    UniAD and VAD share identical mmcv-based plumbing — only
    ``model_type`` and a few trajectory-parsing branches differ — so
    this single class implements both. The Phase-3 reorg surfaces
    them through two thin subclasses (:class:`UniADAdapter` and
    :class:`navsafe.policy.vad.VADAdapter`) that pin ``model_type``
    and carry a unique ``@register_policy`` name (Requirement 3.7,
    Requirement 4.6).
    """

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 model_type: str = "uniad", **kwargs):
        super().__init__(checkpoint_path, config_path, **kwargs)
        self.model_type = model_type
        self.pipeline: Any = None
        self.cfg: Any = None

    def load_model(self):
        print(f"Loading {self.model_type.upper()} model...")

        if not _HAS_PYQUATERNION:
            raise ImportError(
                "UniAD/VAD adapter requires pyquaternion. Install with: pip install pyquaternion"
            )

        # Try to import mmcv — these models require it
        try:
            # Add bench2drive modelzoo to path for plugin loading.
            # This file is ``navsafe/policy/sensor/uniad.py``, so the inner
            # ``navsafe`` package (which holds ``modelzoo/``) is THREE parents
            # up: sensor -> policy -> navsafe. Anchoring on the package rather
            # than counting parents is what stops this drifting again -- the
            # count was last written for ``navsafe/policy/uniad.py`` and
            # silently pointed at ``navsafe/policy/modelzoo/bench2drive``,
            # which does not exist, from the day the file moved into
            # ``sensor/``. Nothing raised: a non-existent sys.path entry is
            # ignored, so the plugin import and the motion_head anchor below
            # failed later and elsewhere.
            _bench2drive_root = _navsafe_package_root() / "modelzoo" / "bench2drive"
            sys.path.insert(0, str(_bench2drive_root))
            from mmcv import Config
            from mmcv.models import build_model
            from mmcv.utils import load_checkpoint
            from mmcv.datasets.pipelines import Compose
        except ImportError as e:
            raise ImportError(
                f"UniAD/VAD models require mmcv and related dependencies. "
                f"Install them with: pip install mmcv-full mmdet mmengine\n"
                f"Original error: {e}"
            ) from e

        self.cfg = Config.fromfile(self.config_path)

        if 'motion_head' in self.cfg.model:
            anchor_path = self.cfg.model['motion_head']['anchor_info_path']
            self.cfg.model['motion_head']['anchor_info_path'] = str(_bench2drive_root / anchor_path)

        if hasattr(self.cfg, 'plugin') and self.cfg.plugin:
            import importlib
            if hasattr(self.cfg, 'plugin_dir'):
                plugin_dir = self.cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir).split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                importlib.import_module(_module_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = build_model(self.cfg.model, train_cfg=self.cfg.get('train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        load_checkpoint(self.model, self.checkpoint_path, map_location='cpu', strict=True)
        self.model.cuda()
        self.model.eval()

        inference_only_pipeline_cfg = []
        for pipeline_cfg in self.cfg.inference_only_pipeline:
            if pipeline_cfg["type"] not in ['LoadMultiViewImageFromFilesInCeph', 'LoadMultiViewImageFromFiles']:
                inference_only_pipeline_cfg.append(pipeline_cfg)
        self.pipeline = Compose(inference_only_pipeline_cfg)

        print(f"{self.model_type.upper()} model loaded successfully.")

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        from mmcv.core.bbox import get_box_type

        results: Dict[str, Any] = {}
        results['lidar2img'] = np.stack([LIDAR2IMG[cam] for cam in CAM_NAMES], axis=0)
        results['lidar2cam'] = np.stack([LIDAR2CAM[cam] for cam in CAM_NAMES], axis=0)
        results['img'] = [images[cam] for cam in CAM_NAMES]

        results['folder'] = ' '
        results['scene_token'] = ' '
        results['frame_idx'] = int(frame_id)
        results['timestamp'] = int(frame_id) / 20.0
        results['box_type_3d'], _ = get_box_type('LiDAR')

        ego_theta = ego_state['heading']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        can_bus = np.zeros(18)
        can_bus[0] = ego_state['position'][0]
        can_bus[1] = ego_state['position'][1]
        can_bus[2] = ego_state['position'][2] if len(ego_state['position']) > 2 else 0.0
        can_bus[3:7] = rotation
        can_bus[7] = np.linalg.norm(ego_state['velocity'])
        can_bus[10:13] = ego_state['acceleration']
        can_bus[13:16] = -ego_state['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180
        results['can_bus'] = can_bus

        command = ego_state['command']
        if command < 0:
            command = 4
        if self.model_type == "vad":
            one_hot = np.zeros(6, dtype=np.float64)
            one_hot[command] = 1.0
            results['command'] = one_hot
            results['ego_fut_cmd'] = one_hot
        else:
            results['command'] = command
            results['ego_fut_cmd'] = np.array([command], dtype=np.int64)

        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = can_bus[0:2]
        lidar2global = ego2world @ LIDAR2EGO
        results['l2g_r_mat'] = lidar2global[0:3, 0:3]
        results['l2g_t'] = lidar2global[0:3, 3]

        stacked_imgs = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape

        results = self.pipeline(results)
        return results

    def run_inference(self, model_input: Any) -> Any:
        from mmcv.parallel.collate import collate as mm_collate_to_batch_form

        input_data_batch = mm_collate_to_batch_form([model_input], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to("cuda")

        with torch.no_grad():
            output_data_batch = self.model(
                input_data_batch, return_loss=False, rescale=True)
        return output_data_batch[0]

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if 'planning' in model_output:
            plan_traj = model_output['planning']['result_planning']['sdc_traj'][0].detach().cpu().numpy()
        elif 'ego_fut_preds' in model_output:
            ego_fut_preds = model_output['ego_fut_preds']
            ego_fut_cmd = ego_state['command']
            plan_traj_idx = min(ego_fut_cmd, ego_fut_preds.shape[0] - 1)
            plan_traj = ego_fut_preds[plan_traj_idx].cpu().numpy()
            plan_traj = np.cumsum(plan_traj, axis=0)
        elif 'pts_bbox' in model_output and isinstance(model_output['pts_bbox'], dict):
            pts_bbox = model_output['pts_bbox']
            if 'ego_fut_preds' in pts_bbox:
                ego_fut_preds = pts_bbox['ego_fut_preds']
                ego_fut_cmd_container = pts_bbox['ego_fut_cmd']
                if hasattr(ego_fut_cmd_container, 'data'):
                    ego_fut_cmd_val = ego_fut_cmd_container.data
                else:
                    ego_fut_cmd_val = ego_fut_cmd_container
                if torch.is_tensor(ego_fut_cmd_val):
                    ego_fut_cmd_val = ego_fut_cmd_val.cpu().argmax().item() if ego_fut_cmd_val.numel() > 1 else ego_fut_cmd_val.cpu().item()
                elif isinstance(ego_fut_cmd_val, np.ndarray):
                    ego_fut_cmd_val = int(np.argmax(ego_fut_cmd_val)) if ego_fut_cmd_val.size > 1 else int(ego_fut_cmd_val.flat[0])
                else:
                    ego_fut_cmd_val = int(ego_fut_cmd_val)
                plan_traj_idx = min(ego_fut_cmd_val, ego_fut_preds.shape[0] - 1)
                plan_traj = ego_fut_preds[plan_traj_idx].cpu().numpy()
                plan_traj = np.cumsum(plan_traj, axis=0)
            else:
                raise KeyError(f"Cannot find ego_fut_preds in pts_bbox. Keys: {pts_bbox.keys()}")
        else:
            raise KeyError(f"Cannot find planning output. Keys: {model_output.keys()}")

        return {'trajectory': plan_traj}

    def get_trajectory_time_horizon(self) -> float:
        return 3.0


@register_policy("uniad")
class UniADAdapter(UniADVADAdapter):
    """UniAD-specific subclass of :class:`UniADVADAdapter`.

    Pins ``model_type="uniad"`` so the shared
    ``prepare_input``/``run_inference``/``parse_output`` plumbing
    routes through the UniAD branches. The class exists purely to
    carry the unique ``@register_policy("uniad")`` name (Requirement
    3.5, 4.6) and to give downstream code a clean
    ``UniADAdapter`` import target separate from the VAD subclass in
    :mod:`navsafe.policy.vad`.
    """

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 **kwargs):
        # Ignore any caller-supplied ``model_type`` to keep the
        # registration name and the runtime model_type in lockstep.
        kwargs.pop("model_type", None)
        super().__init__(
            checkpoint_path,
            config_path=config_path,
            model_type="uniad",
            **kwargs,
        )


__all__ = ["UniADAdapter", "UniADVADAdapter"]
