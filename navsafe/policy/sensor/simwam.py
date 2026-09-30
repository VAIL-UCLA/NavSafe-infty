"""SimWAM adapter — the action expert of a world-action model.

SimWAM (Zhao et al., arXiv 2608.07468) co-trains a Wan2.2-5B video expert and a
lightweight action DiT with joint flow matching, then **skips the video branch
at inference**: an isolated attention mask keeps action tokens independent of
future frames, so the scored path is a self-contained planner that needs one
image and no frame history. That is the difference from the other world-model
rows — DriveLaW runs a video model on every replan, SimWAM does not.

Skipped, not removed. Both released checkpoints carry the full
``mixtures.video.*`` weights, and upstream's ``infer_joint`` denoises video and
action together and decodes the frames through the Wan2.2 VAE. Set
``NAVSAFE_SIMWAM_VIDEO_DIR`` to take that path and write ``wm_NNNNN.mp4`` per
replan. It is much slower, and the trajectory it returns is not the scored one
— upstream only asserts the two agree to atol/rtol 1e-2 — so a video run is for
figures, never for numbers.

``checkpoint_path`` selects the stage. The released pair on NAVSIM navtest:

    SimWAM.pt      joint video-action flow matching          90.3 PDMS
    SimWAM-RL.pt   + FlowGRPO against the NAVSIM PDM reward  91.5 PDMS

The RL checkpoint is the default here: it is the better one, and the gain is
where a closed-loop benchmark cares — EP 83.9 -> 86.4 and DAC 98.0 -> 98.7,
against 0.3 NC and 0.4 TTC given back.

The model runs in a persistent subprocess under its own venv
(``navsafe/modelzoo/navsim/vla_server/simwam_server.py``), which owns the
prompt construction, the 384x672 preprocessing and the ``denorm_odo``
denormalisation, all read off the released dataset code rather than guessed.
Output is 8 x (x, y, heading) in the navsim ego frame, converted here to the
NexusSim ``[lateral, forward]`` contract.
"""

from __future__ import annotations

import os
from navsafe.data_paths import model_path
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

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
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)

DEFAULT_REPO_PATH = os.path.expanduser("~/data/navsafe_repos/SimWAM")
DEFAULT_CHECKPOINT = model_path('simwam/weights/SimWAM-RL.pt')
# The supervised task config, which fixes the camera size (384x672) and
# trajectory_mode=absolute — the latter decides the denormalisation the server
# applies. It serves the RL checkpoint too: that release is merged, not a LoRA
# delta, so the plain factory loads it (see simwam_server.DEFAULT_TASK).
DEFAULT_TASK = "navsim_uncond_front_384x672_1e-4"
# The reference's own eval setting: configs/sim_navsim.yaml resolves
# num_inference_steps to eval_num_inference_steps = 10 (configs/train.yaml),
# which is also simwam_server.py's argparse default. The adapter always passes
# this value, so a different default here silently overrides the reference for
# both SimWAM rows -- it sat at 20, i.e. double the published flow-matching
# steps, for every recorded run.
DEFAULT_NUM_INFERENCE_STEPS = 10


def reference_num_inference_steps(repo_path: str) -> int | None:
    """``eval_num_inference_steps`` read out of the reference's own config.

    Preferred over the constant above, because an adapter-side constant that
    shadows the reference is exactly the defect this replaced: it agreed with
    the paper the day it was written and nothing would have said so when the
    reference moved. Returns None when the repo is not on this box (it lives on
    /data, which is not always mounted), in which case the caller falls back
    to :data:`DEFAULT_NUM_INFERENCE_STEPS`.

    Parsed with a regex rather than yaml because ``configs/sim_navsim.yaml``
    holds ``num_inference_steps: ${eval_num_inference_steps}`` -- an OmegaConf
    interpolation that only resolves inside the reference's own composition, so
    the value has to be read from where it is defined (``configs/train.yaml``).
    """
    cfg = Path(repo_path) / "configs" / "train.yaml"
    try:
        text = cfg.read_text()
    except OSError:
        return None
    m = re.search(r"^eval_num_inference_steps:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(m.group(1)) if m else None


@register_policy("simwam")
class SimWAMAdapter(SensorPolicy):
    """Adapter for SimWAM (action expert; RL checkpoint by default)."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT,
                 config_path: str | None = None, repo_path: str | None = None,
                 task: str | None = None,
                 num_inference_steps: int | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.repo_path = (repo_path
                          or os.environ.get("NAVSAFE_SIMWAM_REPO", DEFAULT_REPO_PATH))
        self.task = task or os.environ.get("NAVSAFE_SIMWAM_TASK", DEFAULT_TASK)
        # None means "whatever the reference evaluates at", which is read from
        # its config so the two cannot drift apart silently. An explicit value
        # still wins, for a deliberate step-count study.
        if num_inference_steps is None:
            ref = reference_num_inference_steps(self.repo_path)
            self.num_inference_steps = (ref if ref is not None
                                        else DEFAULT_NUM_INFERENCE_STEPS)
            print(f"SimWAM num_inference_steps={self.num_inference_steps} "
                  f"({'reference config' if ref is not None else 'fallback constant'})")
        else:
            self.num_inference_steps = int(num_inference_steps)
        self.client: VLASubprocessClient | None = None
        self._tmpdir = tempfile.TemporaryDirectory(prefix="simwam_")

    def load_model(self):
        script = str(vla_server_dir() / "simwam_server.py")
        print(f"Starting SimWAM server (ckpt={self.checkpoint_path})...")
        server_args = ["--repo", self.repo_path,
                       "--checkpoint", self.checkpoint_path,
                       "--task", self.task,
                       "--num-inference-steps", str(self.num_inference_steps)]
        # Future-frame video, opt-in. SimWAM omits frame generation at
        # deployment by design, but the video expert is present in both
        # released checkpoints, so this only costs speed -- and a trajectory
        # that is no longer the scored one (infer_joint vs infer_action agree
        # only to ~1e-2 upstream). Never set this on a scoring run.
        video_dir = os.environ.get("NAVSAFE_SIMWAM_VIDEO_DIR")
        if video_dir:
            server_args += ["--video-dir", video_dir]
            print(f"SimWAM future-frame video -> {video_dir} "
                  f"(trajectories will differ from the scored run)")
        self.client = VLASubprocessClient(vla_python(), script, server_args)
        self.model = self.client
        print("SimWAM server ready.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        img = images.get("CAM_F0")
        if img is None and images:
            img = next(iter(images.values()))
        if img is None:
            img = np.zeros((1120, 1920, 3), dtype=np.uint8)
        img_path = str(Path(self._tmpdir.name) / "cam_f0.npy")
        np.save(img_path, renderer_bgr_to_rgb(crop_to_navsim_aspect(img)))

        vel, acc = local_velocity_acceleration(ego_state)
        # Command first, matching every other adapter's status vector; the
        # server reorders into SimWAM's [velocity, acceleration, command].
        status = navsim_command_one_hot(ego_state) + [float(vel[0]), float(vel[1]),
                                                      float(acc[0]), float(acc[1])]
        return {"image_npy": img_path, "status": status}

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("SimWAMAdapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = np.asarray(model_output["trajectory"], dtype=np.float32)  # (8, 3)
        return {"trajectory": np.column_stack([traj[:, 1], traj[:, 0]]),
                "heading": traj[:, 2]}

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0
