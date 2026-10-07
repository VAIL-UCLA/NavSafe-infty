"""DriveVLA-W0 adapter — Emu3 VLM + flow-matching action expert.

DriveVLA-W0 (Li et al., ICLR 2026, ``BraveGroup/DriveVLA-W0``) trains a VLA
with future-image prediction as a dense self-supervision signal. The released
NAVSIM row this adapter serves is the VQ/autoregressive archetype:
``Emu3_Flow_Matching_Action_Expert_PDMS_87.2`` -- an Emu3 backbone over discrete
visual tokens with a flow-matching action expert on top
(``Emu3Pi0.sample_actions``).

**The offline/closed-loop gap this adapter has to close.** The released pipeline
never sees a picture at inference: ``Emu3DrivingVAVADataset`` reads *pre-computed*
VQ codes off disk (``np.load(path).reshape(1, 18, 32)``) that a separate offline
pass produced with ``Emu3VisionVQModel``. Closed loop there is no such pass, so
the server holds the vision tokenizer itself and encodes each rendered frame --
256x144, which the VQ's factor-8 downsample turns into exactly the 18x32 grid
the prompt format expects (``models/tokenizer/emu3_tokenizer_navsim.py``,
``SIZE = (256, 144)``).

**Prompt.** Reproduced from ``Emu3DrivingVAVADataset.__getitem__``: the sequence
is ``[pre-block | current-block]`` where the pre-block is
``bos + pre_prompt + pre_video + FAST(pre past actions)`` and the current block
is the same without ``bos``. The dataset then appends the *future* action tokens
-- those are training labels, and ``sample_actions`` reads only the prefix, so
the adapter stops before them. Emitting them would be feeding the model its own
answer.

**Two frames per block, at 1 Hz.** ``random_frames_to_tensor`` slices
``img_list[cur_idx - 2*(T-1) : cur_idx+1 : 2]`` with ``T = frames = 1``, i.e. the
current frame alone per block, and the ``pre_1s`` block is the same view one
second earlier. At replan-rate 5 (0.5 s) that is two replans back, which is what
``PRE_BLOCK_LAG_REPLANS`` encodes.

Output is 8 x (x, y, heading) after the reference's own denormalisation
``0.5 * (pred + 1) * (q99 - q01) + q01`` against ``norm_stats['libero']``.
"""

from __future__ import annotations

import os
from navsafe.data_paths import model_path
import tempfile
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.sensor.vla_client import (
    EgoPoseHistory,
    VLASubprocessClient,
    navsim_command_one_hot,
    vla_python,
    vla_server_dir,
)
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)

DEFAULT_REPO_PATH = os.path.expanduser("~/.cache/navsafe/repos/DriveVLA-W0")
DEFAULT_CHECKPOINT = model_path('drivevla_w0/Emu3_Flow_Matching_Action_Expert_PDMS_87.2')

# action_frames / action_dim from the inference script's DataArguments.
ACTION_FRAMES = 8
ACTION_DIM = 3
# cur_frame_idx = 3 -> three past actions go in the prompt as FAST tokens.
PRE_ACTION_FRAMES = 3
# inference.num_inference_steps default in inference/vla/config.py.
DEFAULT_NUM_INFERENCE_STEPS = 10
# The "pre_1s" block is one second back; at replan-rate 5 on a 10 Hz sim that
# is two replans. Named rather than inlined because it is the one number that
# silently changes what "pre_1s" means if the replan rate moves.
PRE_BLOCK_LAG_REPLANS = 2


def _integrate_relative_steps(steps: np.ndarray) -> np.ndarray:
    """Compose per-step ``(dx, dy, dtheta)`` into absolute ego-frame poses.

    A port of the reference's ``_integrate_xy_with_heading``
    (``inference/navsim/navsim/navsim/agents/external_agent.py``), which
    chains each step as an SE(2) transform::

        t2_to_t0 = t1_to_t0 @ t2_to_t1
        t3_to_t0 = t2_to_t0 @ t3_to_t2
        ...

    The first row is taken as an absolute pose (``xy[0]``/``heading[0]`` go
    straight into the output there), and every later row is relative to its
    predecessor. Written as a loop rather than the reference's eight unrolled
    lines, which is the same arithmetic.

    Returns ``(n, 3)`` of ``(x, y, heading)`` in the current ego frame.
    """
    def _mat(dx: float, dy: float, dtheta: float) -> np.ndarray:
        c, s = np.cos(dtheta), np.sin(dtheta)
        return np.array([[c, -s, dx], [s, c, dy], [0.0, 0.0, 1.0]])

    out = np.zeros((len(steps), 3), dtype=np.float32)
    acc = _mat(float(steps[0, 0]), float(steps[0, 1]), float(steps[0, 2]))
    out[0] = (steps[0, 0], steps[0, 1], steps[0, 2])
    for i in range(1, len(steps)):
        acc = acc @ _mat(float(steps[i, 0]), float(steps[i, 1]),
                         float(steps[i, 2]))
        out[i] = (acc[0, 2], acc[1, 2], np.arctan2(acc[1, 0], acc[0, 0]))
    return out


@register_policy("drivevla_w0")
class DriveVLAW0Adapter(SensorPolicy):
    """Adapter for DriveVLA-W0 (Emu3 VQ + flow-matching action expert)."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT,
                 config_path: str | None = None, repo_path: str | None = None,
                 vision_tokenizer_path: str | None = None,
                 vlm_model_path: str | None = None,
                 norm_stats_path: str | None = None,
                 action_tokenizer_path: str | None = None,
                 num_inference_steps: int | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.repo_path = (repo_path
                          or os.environ.get("NAVSAFE_DRIVEVLA_W0_REPO", DEFAULT_REPO_PATH))
        # The three assets that ship beside the checkpoint rather than inside
        # it. Defaulted next to the weights, which is how the HF release is
        # laid out, and overridable for a split install.
        ckpt_dir = os.path.dirname(self.checkpoint_path.rstrip("/"))
        self.vision_tokenizer_path = (
            vision_tokenizer_path
            or os.environ.get("NAVSAFE_DRIVEVLA_W0_VQ")
            or os.path.join(ckpt_dir, "Emu3-VisionTokenizer"))
        # Emu3Pi0 builds its VLM with Emu3MoE.from_pretrained on THIS path and
        # reads the tokenizer from it too, so it must be the base VLM the
        # checkpoint was fine-tuned from -- the checkpoint's own nested
        # vlm_config is a stub without the token ids Emu3Model needs.
        self.vlm_model_path = (
            vlm_model_path
            or os.environ.get("NAVSAFE_DRIVEVLA_W0_VLM")
            or os.path.join(ckpt_dir, "Emu3-Stage1"))
        self.norm_stats_path = (
            norm_stats_path
            or os.environ.get("NAVSAFE_DRIVEVLA_W0_NORM_STATS")
            or os.path.join(self.repo_path, "configs",
                            "normalizer_navsim_trainval", "norm_stats.json"))
        self.action_tokenizer_path = (
            action_tokenizer_path
            or os.environ.get("NAVSAFE_DRIVEVLA_W0_ACTION_TOKENIZER")
            or os.path.join(self.repo_path, "configs", "fast"))
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else os.environ.get("NAVSAFE_DRIVEVLA_W0_STEPS",
                                DEFAULT_NUM_INFERENCE_STEPS))

        self.client: VLASubprocessClient | None = None
        # Kept for its stride guard alone. Both of this adapter's temporal
        # assumptions -- that PRE_BLOCK_LAG_REPLANS replans is one second, and
        # that a past action spans 0.5 s -- hold only at replan-rate 5. At the
        # harness default of 1 the "pre_1s" block would be 0.2 s back and the
        # FAST past-action tokens five times too small, with nothing in the
        # output to show it.
        self._stride_guard = EgoPoseHistory()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="drivevla_w0_")
        # Rendered frames at the replan cadence, so the "pre_1s" block can be
        # served from a frame this episode actually saw rather than a repeat
        # of the current one.
        self._frames: Deque[str] = deque(maxlen=PRE_BLOCK_LAG_REPLANS + 1)
        # Past ego actions, in the (x, y, heading) form the FAST tokenizer
        # takes. Seeded with zeros so the prompt has its three slots from
        # frame 0 rather than only after 1.5 s of driving.
        self._past_actions: Deque[list] = deque(maxlen=PRE_ACTION_FRAMES)
        self._last_pose: Optional[np.ndarray] = None
        self._last_frame_id: Optional[int] = None
        self._slot = 0

    def load_model(self):
        script = str(vla_server_dir() / "drivevla_w0_server.py")
        print(f"Starting DriveVLA-W0 server (ckpt={self.checkpoint_path})...")
        self.client = VLASubprocessClient(
            vla_python(), script,
            ["--repo", self.repo_path,
             "--emu-hub", self.checkpoint_path,
             "--vision-tokenizer", self.vision_tokenizer_path,
             "--vlm-model", self.vlm_model_path,
             "--norm-stats", self.norm_stats_path,
             "--action-tokenizer", self.action_tokenizer_path,
             "--num-inference-steps", str(self.num_inference_steps)])
        self.model = self.client
        print("DriveVLA-W0 server ready.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        if frame_id == 0:
            self._reset()
        self._stride_guard.update(ego_state, frame_id)

        img = images.get("CAM_F0")
        if img is None and images:
            img = next(iter(images.values()))
        if img is None:
            img = np.zeros((1120, 1920, 3), dtype=np.uint8)

        if frame_id != self._last_frame_id:
            self._push_action(ego_state)
            path = os.path.join(
                self._tmpdir.name,
                f"frame_{self._slot % self._frames.maxlen}.npy")
            self._slot += 1
            # The crop is load-bearing, not hygiene: the server resizes to
            # 256x144 (1.7778), and navsim's 1920x1080 is exactly that ratio
            # while the renderer's raw 1920x1120 is 1.7143. Skipping it would
            # squash every frame vertically by 4% before the VQ ever saw it.
            np.save(path, renderer_bgr_to_rgb(crop_to_navsim_aspect(img)))
            self._frames.append(path)
            self._last_frame_id = frame_id

        frames = list(self._frames)
        # Before 1 s of history exists the oldest frame stands in for the
        # pre_1s view: the block still describes something the episode saw,
        # and the alternative (a black frame) is out of distribution.
        pre_path = frames[0]
        cur_path = frames[-1]

        # No velocity/acceleration in the payload: unlike the navsim rows,
        # Emu3Pi0.sample_actions takes only the token prefix, `pre_action` and
        # `cmd` -- the ego's speed reaches the model through `pre_action`, the
        # motion it just executed. Sending a status vector the server cannot
        # use would read as a contract this model does not have.
        return {"pre_image_npy": pre_path,
                "cur_image_npy": cur_path,
                "past_actions": [list(a) for a in self._past_actions],
                "command": navsim_command_one_hot(ego_state)}

    def _reset(self) -> None:
        self._frames.clear()
        self._past_actions.clear()
        self._past_actions.extend([[0.0, 0.0, 0.0]] * PRE_ACTION_FRAMES)
        self._last_pose = None
        self._last_frame_id = None
        self._slot = 0

    def _push_action(self, ego_state: Dict[str, Any]) -> None:
        """Append the pose delta since the last replan, in the ego frame.

        The dataset's ``action`` is an ego-relative (x, y, heading) step, so
        the closed-loop equivalent is the motion actually executed between
        replans expressed in the *previous* ego frame.
        """
        pos = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        heading = float(ego_state["heading"])
        if self._last_pose is not None:
            prev = self._last_pose
            c, s = np.cos(prev[2]), np.sin(prev[2])
            rot = np.array([[c, s], [-s, c]])
            d = rot @ (pos - prev[:2])
            dh = float(np.arctan2(np.sin(heading - prev[2]),
                                  np.cos(heading - prev[2])))
            self._past_actions.append([float(d[0]), float(d[1]), dh])
        self._last_pose = np.array([pos[0], pos[1], heading])

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("DriveVLAW0Adapter: call load_model() first")
        return self.client.request(model_input)

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = np.asarray(model_output["trajectory"], dtype=np.float32)
        if traj.shape != (ACTION_FRAMES, ACTION_DIM):
            raise ValueError(
                f"DriveVLA-W0 returned {traj.shape}, expected "
                f"{(ACTION_FRAMES, ACTION_DIM)}")
        # The model emits RELATIVE steps, not poses: each row is
        # (dx, dy, dtheta) in the previous waypoint's frame. Measured on the
        # released checkpoint -- a go-straight prompt returns column 0 at
        # ~4.3-4.7 for every one of the eight rows rather than accumulating,
        # i.e. ~4.5 m per 0.5 s step (~9 m/s), not a position 4.5 m ahead.
        #
        # They compose as SE(2) transforms, which is not the same as a cumsum:
        # a cumsum adds the steps in the FIRST frame and ignores the heading
        # the earlier steps turned through, so any curve comes out
        # progressively wrong. The reference is explicit about this -- its
        # `_construct_trajectory` calls `_integrate_xy_with_heading` and leaves
        # `xy = np.cumsum(xy, axis=0)` commented out beside it.
        poses = _integrate_relative_steps(traj)
        # navsim x-forward / y-left -> the NavSafe [lateral, forward] contract.
        return {"trajectory": np.column_stack([poses[:, 1], poses[:, 0]]),
                "heading": poses[:, 2]}

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return ACTION_FRAMES * 0.5
