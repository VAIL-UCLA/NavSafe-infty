"""MTDrive adapter — Qwen2.5-VL-7B text-out trajectory planner.

MTDrive released only checkpoints (HF ``chenchaoxNV/mtdrive-models``);
the server reimplements single-turn inference from the paper (the prompt
is ReCogDrive's, the answer is a ``[PT, (x, y, heading) x8]`` string).
``checkpoint_path`` points at the stage directory: ``mtdrive_sft`` or
``mtdrive_rl_best`` (mtGRPO).

When the model output fails to parse (no trajectory tuple in the text),
the adapter holds a zero trajectory — braking in place — and counts the
event; a policy that never parses scores as stationary rather than
crashing the run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.sensor.vla_client import (
    EgoPoseHistory,
    VLASubprocessClient,
    local_velocity_acceleration,
    navsim_command_one_hot,
    vla_python,
    vla_server_dir,
)
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)


@register_policy("mtdrive")
class MTDriveAdapter(SensorPolicy):
    """Adapter for MTDrive (SFT or mtGRPO stage via checkpoint_path)."""

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.client: VLASubprocessClient | None = None
        self._history = EgoPoseHistory()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="mtdrive_")
        self.parse_failures = 0
        # See parse_output: a zero plan is a command to stop, not a
        # neutral default, so the last good plan is held instead.
        self._last_plan = np.zeros((8, 2), dtype=np.float32)

    def load_model(self):
        script = str(vla_server_dir() / "mtdrive_server.py")
        print(f"Starting MTDrive server ({self.checkpoint_path})...")
        self.client = VLASubprocessClient(
            vla_python(), script, ["--model-path", self.checkpoint_path])
        self.model = self.client
        print("MTDrive server ready.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        self._history.update(ego_state, frame_id)
        img = images.get("CAM_F0")
        if img is None and images:
            img = next(iter(images.values()))
        if img is None:
            img = np.zeros((1120, 1920, 3), dtype=np.uint8)
        img_path = str(Path(self._tmpdir.name) / "cam_f0.npy")
        np.save(img_path, renderer_bgr_to_rgb(crop_to_navsim_aspect(img)))

        vel, acc = local_velocity_acceleration(ego_state)
        status = navsim_command_one_hot(ego_state) + [float(vel[0]), float(vel[1]),
                                                      float(acc[0]), float(acc[1])]
        return {"image_npy": img_path,
                "history": self._history.local_history(),
                "status": status}

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("MTDriveAdapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = model_output.get("trajectory")
        if traj is None:
            self.parse_failures += 1
            print(f"[MTDriveAdapter] unparseable model output "
                  f"(#{self.parse_failures}): {model_output.get('text', '')[:200]!r}")
            # Zeros collapse the plan onto the ego origin, which paces at
            # 0 m/s AND exhausts immediately -- so the executor force-replans
            # every frame, paying a full 7B generation each time on the same
            # frozen input that just failed. Repeat the last good plan instead.
            return {"trajectory": self._last_plan.copy()}
        traj = np.asarray(traj, dtype=np.float32)  # (8, 3) x fwd, y left
        self._last_plan = np.column_stack([traj[:, 1], traj[:, 0]])
        return {"trajectory": self._last_plan.copy(),
                "heading": traj[:, 2]}

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0
