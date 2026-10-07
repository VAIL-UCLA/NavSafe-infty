"""ReCogDrive adapter — InternVL3-2B VLM + DiT diffusion planner.

The VLM + planner run in a persistent subprocess under the VLA venv
(``navsafe/modelzoo/navsim/vla_server/recogdrive_server.py``); this
adapter feeds it the front camera, the 4-pose ego history and the 8-dim
status feature, and converts the returned 8x(x, y, heading) navsim-frame
trajectory into the NavSafe ``[lateral, forward]`` contract.

``checkpoint_path`` selects the planner stage:
``ReCogDrive_Diffusion_Planner_2B_IL.ckpt`` (SFT/IL) or ``..._RL.ckpt``
(DiffGRPO RLFT).  The VLM directory is shared by both stages
(``vlm_path`` kwarg / ``NAVSAFE_RECOGDRIVE_VLM`` env).
"""

from __future__ import annotations

import os
from navsafe.data_paths import model_path
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

DEFAULT_VLM_PATH = model_path("recogdrive/vlm2b")


@register_policy("recogdrive")
class ReCogDriveAdapter(SensorPolicy):
    """Adapter for ReCogDrive (IL or RL planner stage via checkpoint_path)."""

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 vlm_path: str | None = None, dit_type: str = "small",
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.vlm_path = (vlm_path
                         or os.environ.get("NAVSAFE_RECOGDRIVE_VLM", DEFAULT_VLM_PATH))
        self.dit_type = dit_type
        self.client: VLASubprocessClient | None = None
        self._history = EgoPoseHistory()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="recogdrive_")

    def load_model(self):
        script = str(vla_server_dir() / "recogdrive_server.py")
        print(f"Starting ReCogDrive server (planner={self.checkpoint_path})...")
        self.client = VLASubprocessClient(
            vla_python(), script,
            ["--vlm-path", self.vlm_path,
             "--planner-ckpt", self.checkpoint_path,
             "--dit-type", self.dit_type])
        self.model = self.client
        print("ReCogDrive server ready.")

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
        # The 1120-row render makes InternVL's dynamic tiling pick a 3x2
        # grid where navsim's 1080 rows give 4x2 -- 512 fewer image tokens,
        # leaving 18% of the 2800-token budget as unseen padding.
        np.save(img_path, renderer_bgr_to_rgb(crop_to_navsim_aspect(img)))

        vel, acc = local_velocity_acceleration(ego_state)
        status = navsim_command_one_hot(ego_state) + [float(vel[0]), float(vel[1]),
                                                      float(acc[0]), float(acc[1])]
        return {"image_npy": img_path,
                "history": self._history.local_history(),
                "status": status}

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("ReCogDriveAdapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = np.asarray(model_output["trajectory"], dtype=np.float32)  # (8, 3) x fwd, y left
        return {"trajectory": np.column_stack([traj[:, 1], traj[:, 0]]),
                "heading": traj[:, 2]}

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0
