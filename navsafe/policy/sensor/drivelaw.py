"""DriveLaW adapter — LTX video world model + latent diffusion planner.

DriveLaW (Xia et al., CVPR 2026) injects the latent of its video generator
into the planner, so the planner's input is a *video*, not a frame: the model
conditions on the past 2 s at 2 Hz and ``forward_test`` asserts exactly four
conditioning frames.  The adapter therefore keeps a 4-deep front-camera
history (``ImageHistory``) alongside the usual 4-pose ego history, and the
replan cadence is what makes that 2 Hz -- at replan-rate 5 and 10 Hz sim,
``prepare_input`` runs every 0.5 s.

The world model + planner run in a persistent subprocess under the VLA venv
(``navsafe/modelzoo/navsim/vla_server/drivelaw_server.py``).  Output is the
repo's own ``actions``: 8 x (x, y, heading) in the navsim ego frame after
``denorm_odo``, converted here to the NavSafe ``[lateral, forward]``
contract.

``checkpoint_path`` is the DriveLaW-Act inference config yaml (the repo takes
``config_file``, which names the weights it loads); ``repo_path`` locates the
checkout.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.sensor.vla_client import (
    EgoPoseHistory,
    ImageHistory,
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

DEFAULT_REPO_PATH = os.path.expanduser("~/.cache/navsafe/repos/DriveLaW")
# forward_test asserts T_cond == 4; this is not a tunable.
COND_FRAMES = 4


@register_policy("drivelaw")
class DriveLaWAdapter(SensorPolicy):
    """Adapter for DriveLaW (DriveLaW-Video latents -> DriveLaW-Act planner)."""

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 repo_path: str | None = None, view_mode: str = "front",
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.repo_path = (repo_path
                          or os.environ.get("NAVSAFE_DRIVELAW_REPO", DEFAULT_REPO_PATH))
        self.view_mode = view_mode
        self.client: VLASubprocessClient | None = None
        self._history = EgoPoseHistory()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="drivelaw_")
        self._frames = ImageHistory(self._tmpdir.name, maxlen=COND_FRAMES)

    def load_model(self):
        script = str(vla_server_dir() / "drivelaw_server.py")
        print(f"Starting DriveLaW server (config={self.checkpoint_path})...")
        self.client = VLASubprocessClient(
            vla_python(), script,
            ["--repo", self.repo_path,
             "--config", self.checkpoint_path,
             "--view-mode", self.view_mode])
        self.model = self.client
        print("DriveLaW server ready.")

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
        self._frames.update(
            renderer_bgr_to_rgb(crop_to_navsim_aspect(img)), frame_id)

        vel, acc = local_velocity_acceleration(ego_state)
        status = navsim_command_one_hot(ego_state) + [float(vel[0]), float(vel[1]),
                                                      float(acc[0]), float(acc[1])]
        return {"image_npys": self._frames.paths(),
                "history": self._history.local_history(),
                "status": status}

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("DriveLaWAdapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = np.asarray(model_output["trajectory"], dtype=np.float32)  # (8, 3)
        return {"trajectory": np.column_stack([traj[:, 1], traj[:, 0]]),
                "heading": traj[:, 2]}

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0
