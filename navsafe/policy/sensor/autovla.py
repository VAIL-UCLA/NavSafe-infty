"""AutoVLA adapter — Qwen2.5-VL-3B with physical action tokens.

The server (``vla_server/autovla_server.py``) ports ``AutoVLA.predict``:
three cameras (front / front-left / front-right) x 4 sequential frames at
2 Hz go in as three videos; 10 action tokens come back, decoded through
the repo codebook into a 5 s trajectory at 0.5 s steps.

Frame history: ``prepare_input`` runs once per replan (0.5 s at
replan-rate 5), so keeping the last 4 rendered frames per camera exactly
reproduces the 2 Hz context window.  The released checkpoint is the
GRPO-RFT model (``AutoVLA_PDMS_89.ckpt``); no pre-RL checkpoint is
public, so AutoVLA is final-policy evidence only (Table I ‡).
"""

from __future__ import annotations

import os
from navsafe.data_paths import model_path
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict

import cv2
import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.sensor.vla_client import (
    VLASubprocessClient,
    local_velocity_acceleration,
    navsim_command_one_hot,
    vla_python,
    vla_server_dir,
)
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
from navsafe.policy.sensor.utils.frames import crop_to_navsim_aspect

DEFAULT_BASE_MODEL = model_path("_base/qwen2.5-vl-3b-instruct")
DEFAULT_REPO = os.path.expanduser("~/data/navsafe_repos/AutoVLA")

# AutoVLA free-text instruction, indexed by the navsim command ONE-HOT
# ([left, straight, right, unknown]) — the same list and the same indexing
# recogdrive/mtdrive use. Deliberately not a dict keyed on the raw command int:
# that int is vis_utils' CMD_* (0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANEFOLLOW),
# whose order is NOT the one-hot's, and a dict written in one-hot order against
# it silently swaps RIGHT and STRAIGHT — which told the model "turn right" on
# every straight-ahead frame.
# Verbatim from the reference builder (navsim/agents/vla_agent.py), lowered
# by AutoVLA.get_prompt. Index 3 is UNKNOWN, not an assertion of "straight":
# NexusSim routes LANEFOLLOW there, so claiming "go straight" told the model
# something the route did not say.
_COMMAND_TEXT = ("turn left", "keep forward", "turn right", "unknown")

# The released checkpoint is the nuPlan one, and its builder filled the
# front_left / front_right video slots from `cam_l1` / `cam_r1` (yaw 111 deg /
# 248 deg), NOT `cam_l0` / `cam_r0` (yaw 55 deg / -56 deg) -- see
# navsim/agents/vla_agent.py and autovla_agent.py's `dataset_name == "nuplan"`
# branch. The prompt still calls them "front-left"/"front-right"; that is the
# reference's own labelling, and matching the pixels matters more than the word.
_CAM_TO_SLOT = {"CAM_F0": "front", "CAM_L1": "front_left", "CAM_R1": "front_right"}

#: Set NAVSAFE_AUTOVLA_RAW_DUMP=<dir> to keep every camera frame this adapter
#: hands the VLA server. Unset by default. The frames under <output_dir>/frames/
#: carry the visualiser's overlay, so they cannot answer what the model actually
#: saw -- asking it about those is asking it to describe our own annotations.
_RAW_DUMP_DIR = os.environ.get("NAVSAFE_AUTOVLA_RAW_DUMP", "")


@register_policy("autovla")
class AutoVLAAdapter(SensorPolicy):
    """Adapter for AutoVLA (released GRPO-RFT checkpoint)."""

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 base_model: str | None = None, repo: str | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.base_model = (base_model
                           or os.environ.get("NAVSAFE_AUTOVLA_BASE", DEFAULT_BASE_MODEL))
        self.repo = repo or os.environ.get("NAVSAFE_AUTOVLA_REPO", DEFAULT_REPO)
        self.client: VLASubprocessClient | None = None
        self._tmpdir = tempfile.TemporaryDirectory(prefix="autovla_")
        self._frames: Dict[str, Deque[str]] = {
            slot: deque(maxlen=4) for slot in _CAM_TO_SLOT.values()}
        self._frame_counter = 0
        self.decode_failures = 0
        # Held across replans so a decode failure repeats the last good
        # plan rather than commanding a stop (zeros).
        self._last_plan = np.zeros((10, 2), dtype=np.float32)

    def load_model(self):
        script = str(vla_server_dir() / "autovla_server.py")
        print(f"Starting AutoVLA server ({self.checkpoint_path})...")
        self.client = VLASubprocessClient(
            vla_python(), script,
            ["--base-model", self.base_model,
             "--checkpoint", self.checkpoint_path,
             "--repo", self.repo])
        self.model = self.client
        print("AutoVLA server ready.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {k: NAVSIM_CAM_CONFIGS[k] for k in _CAM_TO_SLOT}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        if frame_id == 0:
            for dq in self._frames.values():
                dq.clear()
        self._frame_counter += 1
        for cam, slot in _CAM_TO_SLOT.items():
            img = images.get(cam)
            if img is None and images:
                img = next(iter(images.values()))
            if img is None:
                img = np.zeros((1120, 1920, 3), dtype=np.uint8)
            path = str(Path(self._tmpdir.name)
                       / f"{slot}_{self._frame_counter % 8}.jpg")
            # The renderer frame is BGR and cv2.imwrite expects BGR, so it is
            # written as-is: the server reads the jpg back with PIL, which
            # decodes to the RGB the processor wants. Swapping here instead
            # wrote a colour-inverted jpg (see utils.frames).
            cropped = crop_to_navsim_aspect(img.astype(np.uint8))
            cv2.imwrite(path, cropped)
            if _RAW_DUMP_DIR:
                dump = Path(_RAW_DUMP_DIR) / ("%05d_%s.jpg" % (frame_id, slot))
                dump.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dump), cropped)
            self._frames[slot].append(path)

        frames = {}
        for slot, dq in self._frames.items():
            fr = list(dq)
            while len(fr) < 4:
                fr.insert(0, fr[0])
            frames[slot] = fr

        vel, acc = local_velocity_acceleration(ego_state)
        one_hot = navsim_command_one_hot(ego_state)
        return {
            "frames": frames,
            "velocity": float(np.linalg.norm(vel)),
            "acceleration": float(np.linalg.norm(acc)),
            "instruction": _COMMAND_TEXT[int(np.argmax(one_hot))],
        }

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("AutoVLAAdapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = model_output.get("trajectory")
        if traj is None:
            self.decode_failures += 1
            print(f"[AutoVLAAdapter] no action tokens in output "
                  f"(#{self.decode_failures}): {model_output.get('cot', '')[:200]!r}")
            return {"trajectory": self._last_plan.copy()}
        traj = np.asarray(traj, dtype=np.float32)  # (10, 3) x fwd, y left
        # The reference clamps to num_poses=10 (autovla_agent.py) and zero-pads
        # a short decode; an unclamped decode silently changes the horizon.
        traj = traj[:10]
        self._last_plan = np.column_stack([traj[:, 1], traj[:, 0]])
        out: Dict[str, Any] = {"trajectory": self._last_plan.copy()}
        if traj.shape[1] > 2:
            out["heading"] = traj[:, 2]
        cot = model_output.get("cot")
        if cot:
            out["reasoning"] = str(cot)
        return out

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 5.0
