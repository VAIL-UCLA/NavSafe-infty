"""ResWorld adapter — temporal residual world model over a BEV encoder.

ResWorld (Zhang et al., ICLR 2026, ``mengtan00/ResWorld``) predicts the future
*spatial distribution of dynamic objects* by taking temporal residuals of BEV
scene representations: TR-World subtracts consecutive BEV token sets rather
than detecting and tracking, and a Future-Guided Trajectory Refinement module
attends a prior trajectory against the predicted future BEV.

Three things make this row different from every other world-model adapter here
(DriveLaW / SimWAM / DriveVLA-W0), and each one is a place where copying their
shape would be wrong:

1. **It is a nuScenes/mmdet3d model, not a navsim one.** The published stack is
   py3.8 + torch 1.9.1 + mmcv-full 1.4.0 + mmdet3d 0.17.1 with compiled CUDA
   ops -- the same legacy-stack situation as MindDrive, so it gets its own venv
   (``NAVSAFE_RESWORLD_PYTHON``) rather than the shared VLA one.
2. **Six surround cameras at 256x704**, the nuScenes rig
   (``data_config['cams']``), not navsim's front view. The renderer is asked for
   the eight-camera navsim rig's closest six and the server maps them into
   nuScenes slot order.
3. **A 3 s horizon, not 4 s.** ``valid_fut_ts = fut_ts = 6`` at 0.5 s spacing;
   ``ego_fut_preds`` is ``(bs, ego_fut_mode=3, 6, 2)`` of per-step **deltas**,
   which the reference selects by command index and then ``cumsum``s
   (``resworld.py::simple_test_pts``). Reporting 8 poses here would be
   fabricating two.

The reference's ``simple_test``/``simple_test_pts`` cannot be called
closed-loop: they compute planning metrics inline against ground-truth boxes,
maps and ``ego_fut_trajs``, none of which exist while driving. The server
therefore calls ``extract_feat`` + ``pts_bbox_head`` directly, which is the
same bypass ``drivelaw_server`` makes and for the same reason.

``checkpoint_path`` is the trained ResWorld ``.pth`` (the release is
``epoch_12_ema.pth``); ``config_path`` is the mmcv config that names the
architecture (``projects/configs/resworld/resworld_config.py``).
"""

from __future__ import annotations

import os
from navsafe.data_paths import model_path
import tempfile
from typing import Any, Dict

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.sensor.vla_client import (
    EgoPoseHistory,
    VLASubprocessClient,
    local_velocity_acceleration,
    vla_server_dir,
)
from navsafe.utils.camera_utils import (
    NAVSIM_CAM_CONFIGS,
    OPENSCENE_CAMERA_PARAMS,
)

DEFAULT_REPO_PATH = os.path.expanduser("~/.cache/navsafe/repos/ResWorld")
DEFAULT_CHECKPOINT = model_path('resworld/resworld_trained.pth')
DEFAULT_CONFIG = "projects/configs/resworld/resworld_config.py"

# ResWorld's own py3.8 stack (mmcv-full 1.4.0 + compiled mmdet3d ops). Kept
# separate from NAVSAFE_VLA_PYTHON on purpose: the VLA venv is py3.12/torch
# 2.x and cannot import mmcv 1.4.0 at all.
DEFAULT_RESWORLD_PYTHON = os.path.expanduser("~/.cache/navsafe/venvs/resworld/bin/python")

# fut_ts = valid_fut_ts = 6 at 0.5 s (resworld_config.py). Not a tunable: the
# head's `ego_fut_decoder` emits exactly fut_ts * 2 numbers per mode.
FUT_TS = 6
WAYPOINT_DT = 0.5

# nuScenes surround rig -> the closest camera in the navsim eight-camera rig.
# CAM_L0/CAM_R0 are the ~55 deg front-obliques and CAM_L2/CAM_R2 the ~140/218
# deg rear-obliques (see NAVSIM_CAM_CONFIGS yaws), which is the nuScenes
# FRONT_LEFT/FRONT_RIGHT/BACK_LEFT/BACK_RIGHT arrangement. The order of this
# dict IS the model's channel order and the server relies on it.
NUSCENES_CAM_SLOTS = {
    "CAM_FRONT_LEFT": "CAM_L0",
    "CAM_FRONT": "CAM_F0",
    "CAM_FRONT_RIGHT": "CAM_R0",
    "CAM_BACK_LEFT": "CAM_L2",
    "CAM_BACK": "CAM_B0",
    "CAM_BACK_RIGHT": "CAM_R2",
}


def resworld_python() -> str:
    return os.environ.get("NAVSAFE_RESWORLD_PYTHON", DEFAULT_RESWORLD_PYTHON)


@register_policy("resworld")
class ResWorldAdapter(SensorPolicy):
    """Adapter for ResWorld (TR-World BEV residual planner, ICLR 2026)."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT,
                 config_path: str | None = None, repo_path: str | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.repo_path = (repo_path
                          or os.environ.get("NAVSAFE_RESWORLD_REPO", DEFAULT_REPO_PATH))
        # The mmcv config is resolved against the repo when given as a relative
        # path, which is how the reference's own commands name it.
        cfg = config_path or os.environ.get("NAVSAFE_RESWORLD_CONFIG", DEFAULT_CONFIG)
        self.mmcv_config = (cfg if os.path.isabs(cfg)
                            else os.path.join(self.repo_path, cfg))
        self.client: VLASubprocessClient | None = None
        # Kept for its stride guard alone, NOT for its poses: the released
        # config sets ego_lcf_feat_idx=None, so the head ignores
        # ego_his_trajs entirely and this adapter sends none. What the guard
        # still buys is the can_bus deltas below, which are per-replan
        # differences and therefore only mean "one second of motion" at
        # replan-rate 5 -- at the harness default of 1 they would silently be
        # five times too small.
        self._stride_guard = EgoPoseHistory()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="resworld_")
        # Previous-sample pose, for the can_bus deltas. None = start of an
        # episode, which the reference represents as a zero delta.
        self._prev_pos: np.ndarray | None = None
        self._prev_angle_deg: float = 0.0
        self._calib: Dict[str, Dict[str, list]] | None = None

    def load_model(self):
        script = str(vla_server_dir() / "resworld_server.py")
        print(f"Starting ResWorld server (ckpt={self.checkpoint_path})...")
        self.client = VLASubprocessClient(
            resworld_python(), script,
            ["--repo", self.repo_path,
             "--config", self.mmcv_config,
             "--checkpoint", self.checkpoint_path])
        self.model = self.client
        print("ResWorld server ready.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {cam: NAVSIM_CAM_CONFIGS[cam] for cam in NUSCENES_CAM_SLOTS.values()}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        # Frame 0 is a fresh episode: drop the previous pose so the first
        # can_bus delta is zero rather than a jump from wherever the last
        # episode ended. EgoPoseHistory does the same internally.
        if frame_id == 0:
            self._prev_pos = None
            self._prev_angle_deg = 0.0
        self._stride_guard.update(ego_state, frame_id)

        # One .npy per camera, rewritten in place each replan: six 1920x1120
        # frames are ~38 MB, which does not belong in a JSON pipe (the same
        # reason ImageHistory writes files).
        #
        # Frames go out in the renderer's native BGR, NOT converted here: the
        # reference's mmlabNormalize calls imnormalize(..., to_rgb=True), which
        # does the swap itself. Converting first would swap twice -- the exact
        # silent-wrong-colour failure utils.frames warns about, arrived at from
        # the other direction.
        cam_npys: Dict[str, str] = {}
        for slot, cam in NUSCENES_CAM_SLOTS.items():
            img = images.get(cam)
            if img is None:
                # A missing camera is a render-request bug, not something to
                # paper over with zeros: a black surround view silently
                # changes what the BEV encoder sees.
                raise RuntimeError(
                    f"ResWorldAdapter: renderer returned no {cam} (nuScenes "
                    f"slot {slot}); get_camera_configs() asks for all six")
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(
                    f"ResWorldAdapter: expected an (H, W, 3) frame for {cam}, "
                    f"got shape {arr.shape}")
            path = os.path.join(self._tmpdir.name, f"{slot}.npy")
            np.save(path, np.ascontiguousarray(arr.astype(np.uint8)))
            cam_npys[slot] = path

        return {"cam_npys": cam_npys,
                "command": self._nuscenes_command(ego_state),
                "can_bus": self._can_bus(ego_state),
                "calibration": self._calibration()}

    def _calibration(self) -> Dict[str, Dict[str, list]]:
        """Per-camera intrinsics and sensor->lidar extrinsics, nuScenes slots.

        Sent in the payload rather than read by the server: the subprocess is
        started with ``env -u PYTHONPATH`` (vla_client), so it cannot import
        navsafe, and copying the rig table into the server file would fork
        the single definition every other surround adapter shares. The rig is
        rigid, so this is built once and the server caches the first copy.
        """
        if self._calib is None:
            self._calib = {
                slot: {
                    "intrinsics": np.asarray(
                        OPENSCENE_CAMERA_PARAMS[cam]["intrinsics"]).tolist(),
                    "sensor2lidar_rotation": np.asarray(
                        OPENSCENE_CAMERA_PARAMS[cam]["sensor2lidar_rotation"]).tolist(),
                    "sensor2lidar_translation": np.asarray(
                        OPENSCENE_CAMERA_PARAMS[cam]["sensor2lidar_translation"]).tolist(),
                }
                for slot, cam in NUSCENES_CAM_SLOTS.items()
            }
        return self._calib

    def _can_bus(self, ego_state: Dict[str, Any]) -> list:
        """The 18-vector ``canbus_mlp`` consumes, in the reference's layout.

        Assembled from ``_get_can_bus_info`` (the converter) plus the two
        overwrites the dataset applies afterwards:

            [0:3]   translation, rewritten to the DELTA against the previous
                    sample (``can_bus[:3] -= prev_pos``); zero on the first
                    frame of a sequence
            [3:7]   ego orientation quaternion (w, x, y, z)
            [7:16]  nuScenes CAN pose fields -- accel(3), rotation_rate(3),
                    vel(3), in the alphabetical key order ``last_pose.keys()``
                    yields after pos/orientation are popped
            [16]    yaw in radians
            [17]    yaw in degrees, rewritten to the DELTA against the
                    previous sample (``can_bus[-1] -= prev_angle``)

        The renderer gives world pose, velocity and acceleration, so [0:7],
        [7:16] and [16:18] are all real here -- rotation_rate is the one field
        the harness does not carry and it is left at zero. That is the same
        partial-information case the reference itself handles by returning
        ``np.zeros(18)`` for scenes with no CAN log, so a zero in one slot is
        in-distribution rather than novel.
        """
        pos = np.asarray(ego_state["position"], dtype=np.float64)
        if pos.shape[0] == 2:
            pos = np.array([pos[0], pos[1], 0.0])
        heading = float(ego_state["heading"])

        # patch_angle is wrapped into [0, 360) before the dataset takes its
        # delta, so the wrap has to happen before the subtraction here too.
        patch_angle = np.degrees(heading) % 360.0

        can_bus = np.zeros(18, dtype=np.float64)
        # Deltas, not absolutes: the first frame of an episode has no previous
        # pose, and the reference zeroes that case rather than seeding it.
        if self._prev_pos is not None:
            can_bus[0:3] = pos - self._prev_pos
            # Shortest signed turn: a straight subtraction of two wrapped
            # angles reads a 1 deg turn through due-east as -359.
            can_bus[17] = (patch_angle - self._prev_angle_deg + 180.0) % 360.0 - 180.0
        can_bus[3:7] = [np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)]

        vel, acc = local_velocity_acceleration(ego_state)
        can_bus[7:10] = [float(acc[0]), float(acc[1]), 0.0]
        # [10:13] rotation_rate: not carried by the harness, left zero.
        can_bus[13:16] = [float(vel[0]), float(vel[1]), 0.0]

        can_bus[16] = np.radians(patch_angle)

        self._prev_pos = pos
        self._prev_angle_deg = patch_angle
        return [float(v) for v in can_bus]

    @staticmethod
    def _nuscenes_command(ego_state: Dict[str, Any]) -> int:
        """ResWorld's 3-way command index (ego_fut_mode = 3).

        nuScenes/VAD order is [right, left, straight] -- the convention
        ResWorld inherits from VAD's dataset, where `ego_fut_cmd` is a
        one-hot the head indexes with `torch.nonzero(cmd)[0, 0]`. NavSafe's
        `command` follows navsim's [left, straight, right, unknown], so the
        two are remapped here rather than in the server: this is the only
        place that knows the harness convention.
        """
        from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD

        vec = NAVSIM_CMD_MAPPING.get(ego_state.get("command", 3), DEFAULT_CMD)
        navsim_idx = int(np.argmax(vec)) if np.any(vec) else 3
        # navsim left/straight/right/unknown -> VAD right/left/straight.
        # 'unknown' has no VAD counterpart; straight is the neutral choice and
        # is what the reference's own data uses when no turn is annotated.
        return {0: 1, 1: 2, 2: 0, 3: 2}[navsim_idx]

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("ResWorldAdapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        # (6, 2) cumulative waypoints in the nuScenes LIDAR frame, which is
        # **x = lateral (right positive), y = forward** -- NOT the x-forward
        # convention the navsim rows use. The reference states it: the VAD
        # converter reads `ego_fut_trajs[-1][0] >= 2` as "Turn Right", so
        # column 0 is the lateral channel. Measured too: on a go-straight
        # command the released checkpoint returns column 1 growing
        # 1.80 -> 9.30 m while column 0 stays under 0.11 m.
        #
        # So the NavSafe [lateral, forward] contract is this array as-is.
        # Column-swapping it (the right move for every navsim adapter here)
        # feeds forward distance in as lateral offset: the ego swerves off the
        # road within a second, which is exactly what the first closed-loop
        # run did -- steer -0.345 at frame 20 rising to -0.914 by frame 30,
        # off_drivable, while every other model held steer under 0.02 on the
        # same token.
        traj = np.asarray(model_output["trajectory"], dtype=np.float32)
        if traj.shape != (FUT_TS, 2):
            raise ValueError(
                f"ResWorld returned {traj.shape}, expected {(FUT_TS, 2)}")
        right, forward = traj[:, 0], traj[:, 1]
        # ...and the sign flips as well as the order. The harness reads column
        # 0 as `ego_left` (evaluator.py: `ego_left = traj_ego[:, 0]`, then
        # `world_x = x + cos_h*forward - sin_h*ego_left`), so it is
        # left-positive, while ResWorld's lateral channel is right-positive
        # (`ego_fut_trajs[-1][0] >= 2` == Turn Right). Without the negation
        # every turn comes out mirrored -- a failure that looks like bad
        # driving rather than a bad adapter.
        left = -right
        # ResWorld predicts positions only -- no heading channel. Deriving it
        # from consecutive waypoints is what the controller needs.
        heading = self._heading_from_waypoints(forward, right)
        return {"trajectory": np.column_stack([left, forward]),
                "heading": heading}

    @staticmethod
    def _heading_from_waypoints(forward: np.ndarray,
                                lateral: np.ndarray) -> np.ndarray:
        """Tangent heading per waypoint, ego frame (0 = straight ahead).

        Takes the two channels separately rather than a (n, 2) array, because
        which column is which is the whole subtlety of this adapter and an
        `arctan2(traj[:, 1], traj[:, 0])` on the wrong convention is silently
        off by 90 degrees.

        Heading is CCW-positive (+ = left) to match the harness, so the
        lateral channel is negated: ResWorld's is right-positive.

        A segment shorter than 1 cm carries the previous heading rather than
        the arctan2 of numerical noise, which would otherwise spin the yaw of
        a stopped ego.
        """
        prev_f = prev_l = 0.0
        heading = np.zeros(len(forward), dtype=np.float32)
        last = 0.0
        for i in range(len(forward)):
            df = float(forward[i]) - prev_f
            dl = float(lateral[i]) - prev_l
            if float(np.hypot(df, dl)) >= 0.01:
                last = float(np.arctan2(-dl, df))
            heading[i] = last
            prev_f, prev_l = float(forward[i]), float(lateral[i])
        return heading

    def get_waypoint_dt(self) -> float:
        return WAYPOINT_DT

    def get_trajectory_time_horizon(self) -> float:
        return FUT_TS * WAYPOINT_DT  # 3.0 s, not the navsim rows' 4.0
