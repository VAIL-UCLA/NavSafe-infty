"""Shared plumbing for VLA subprocess adapters (ReCogDrive / MTDrive / AutoVLA).

Follows the Alpamayo-R1 adapter pattern: the heavyweight VLM runs in a
persistent subprocess under a dedicated venv (``NAVSAFE_VLA_PYTHON``),
speaking one-JSON-per-line over stdin/stdout.  This keeps the IsaacSim
eval process free of transformers-version constraints.

Also hosts the ego-pose-history bookkeeping every navsim-style VLA needs:
the last 4 ego poses at 0.5 s spacing, expressed in the *current* ego
frame (x forward, y left, heading relative) — the ``ego_statuses[:4]``
contract of the navsim devkit that ReCogDrive/MTDrive prompts encode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import IO, Deque, Dict, List, Optional, Sequence, cast

import numpy as np

DEFAULT_VLA_PYTHON = os.path.expanduser("~/data/navsafe_venvs/vla/bin/python")


def vla_python() -> str:
    return os.environ.get("NAVSAFE_VLA_PYTHON", DEFAULT_VLA_PYTHON)


class VLASubprocessClient:
    """Persistent line-JSON subprocess wrapper."""

    def __init__(self, python: str, script: str, args: Sequence[str],
                 ready_timeout_s: float = 900.0):
        if not os.path.isfile(python):
            raise FileNotFoundError(
                f"VLA venv python not found: {python} "
                f"(set NAVSAFE_VLA_PYTHON)")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"VLA server script not found: {script}")
        # Strip the host's PYTHONPATH/PYTHONHOME: the server venv must not
        # inherit IsaacLab/navsafe paths (mismatched torch builds).
        cmd = ["env", "-u", "PYTHONPATH", "-u", "PYTHONHOME", "-u", "PYTHONUTF8"]
        # ...but a server venv that cannot be written to (a full or quota'd
        # filesystem) needs its extra packages somewhere else, so allow an
        # explicit side site-dir. Set *after* the -u, which is what makes it
        # win; it is opt-in precisely so the host's PYTHONPATH never leaks in.
        extra_pythonpath = os.environ.get("NAVSAFE_VLA_EXTRA_PYTHONPATH")
        if extra_pythonpath:
            cmd += [f"PYTHONPATH={extra_pythonpath}"]
        # Pin the server to its own GPU (default 7) so the VLM does not share
        # VRAM with IsaacSim/the renderer on the eval GPU.
        server_gpu = os.environ.get("NAVSAFE_VLA_GPU", "7")
        if server_gpu:
            cmd += [f"CUDA_VISIBLE_DEVICES={server_gpu}"]
        cmd += [python, script, *args]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True)
        # stdout=PIPE always yields a stream; typeshed only knows Optional.
        server_stdout = cast("IO[str]", self._proc.stdout)
        # Some model libraries (e.g. InternVL remote code) print banners to
        # stdout before the server's READY line — skim past them.
        for _ in range(200):
            line = server_stdout.readline()
            if not line:
                self._proc.kill()
                raise RuntimeError("VLA server exited before READY")
            if line.strip() == "READY":
                break
            print(f"[vla-server] {line.rstrip()}")
        else:
            self._proc.kill()
            raise RuntimeError("VLA server: no READY within 200 lines")

    def request(self, payload: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("VLA server process is not running")
        # Both pipes were opened with PIPE in __init__ (see the cast there).
        server_stdin = cast("IO[str]", self._proc.stdin)
        server_stdout = cast("IO[str]", self._proc.stdout)
        server_stdin.write(json.dumps(payload) + "\n")
        server_stdin.flush()
        line = server_stdout.readline()
        if not line:
            raise RuntimeError("VLA server closed stdout unexpectedly")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(
                f"VLA server error: {result['error']}\n{result.get('traceback', '')}")
        return result

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __del__(self):
        self.close()


# The six VLA rows condition on a 2 Hz history, which this module reproduces by
# appending once per distinct frame_id -- correct only while a replan lands
# every 5th sim frame (replan-rate 5 at 10 Hz). The harness default is 1, and at
# that rate the same code silently builds a 0.4 s history where the models were
# trained on 2.0 s, with nothing in the output to show it. So the spacing is an
# assumption, and this checks it once per episode: on the first measurable gap,
# which is early enough to fail before a cell burns a GPU hour.
HISTORY_STRIDE_FRAMES = 5


def _check_history_stride(prev: Optional[int], cur: Optional[int], what: str,
                          expected: int = HISTORY_STRIDE_FRAMES) -> None:
    """Raise unless successive history appends are ``expected`` frames apart.

    Checked on the FIRST gap of an episode only. A later ragged gap (a final
    partial step, a dropped frame) is not a protocol error and must not kill a
    run that is otherwise valid, whereas the wrong replan rate is wrong from the
    first gap onwards.

    Set ``NAVSAFE_ALLOW_HISTORY_STRIDE=1`` to run a deliberate protocol change;
    such a run is not comparable with any recorded number.
    """
    if prev is None or cur is None:
        return
    stride = cur - prev
    if stride == expected or os.environ.get("NAVSAFE_ALLOW_HISTORY_STRIDE"):
        return
    raise RuntimeError(
        f"{what}: history frames are {stride} sim frames apart, expected "
        f"{expected}. These models were trained on a "
        f"{expected / 10.0:.1f} s spacing, so pass --replan-rate {expected} "
        f"(the harness default is 1). Set NAVSAFE_ALLOW_HISTORY_STRIDE=1 to "
        f"override, which makes the run incomparable with recorded numbers.")


class EgoPoseHistory:
    """Last-4 ego poses at (roughly) 0.5 s spacing, in the current ego frame.

    ``prepare_input`` runs once per replan (replan-rate 5 at 10 Hz sim =
    0.5 s), so appending once per distinct frame_id reproduces the 2 Hz
    history the navsim devkit feeds these models.
    """

    def __init__(self, maxlen: int = 4):
        self._poses: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._last_frame_id: Optional[int] = None
        self._stride_checked = False

    def reset(self):
        self._poses.clear()
        self._last_frame_id = None
        self._stride_checked = False

    def update(self, ego_state: Dict, frame_id: int):
        if frame_id == 0:
            self.reset()
        if frame_id == self._last_frame_id:
            return
        if not self._stride_checked:
            _check_history_stride(self._last_frame_id, frame_id, "EgoPoseHistory")
            self._stride_checked = self._last_frame_id is not None
        self._last_frame_id = frame_id
        pos = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        heading = float(ego_state["heading"])
        self._poses.append(np.array([pos[0], pos[1], heading]))

    def local_history(self) -> List[List[float]]:
        """4 poses [x, y, heading] in the current ego frame (last = origin)."""
        poses = list(self._poses)
        if not poses:
            return [[0.0, 0.0, 0.0]] * 4
        while len(poses) < 4:
            poses.insert(0, poses[0])
        cur = poses[-1]
        c, s = np.cos(cur[2]), np.sin(cur[2])
        # world -> ego: R^T (p - p0); x forward, y left for CCW heading.
        rot = np.array([[c, s], [-s, c]])
        out = []
        for p in poses:
            d = rot @ (p[:2] - cur[:2])
            dh = float(np.arctan2(np.sin(p[2] - cur[2]), np.cos(p[2] - cur[2])))
            out.append([float(d[0]), float(d[1]), dh])
        return out


class ImageHistory:
    """Last-N camera frames at the replan cadence, oldest first.

    World-model policies condition on a short video, not a single frame:
    DriveLaW asserts exactly 4 conditioning frames.  ``prepare_input`` runs
    once per replan (replan-rate 5 at 10 Hz = 0.5 s), so appending once per
    distinct frame_id reproduces the 2 Hz history it was trained on.

    Frames are written to ``.npy`` files in ``dirpath`` and passed to the
    server as paths -- a 4-frame 1920x1120 history is 25 MB, which is not
    something to put through a JSON pipe.  The files are reused round-robin,
    so the directory holds ``maxlen`` files for the whole episode.
    """

    def __init__(self, dirpath: str, maxlen: int = 4):
        self._dir = Path(dirpath)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._maxlen = int(maxlen)
        self._paths: Deque[str] = deque(maxlen=self._maxlen)
        self._last_frame_id: Optional[int] = None
        self._stride_checked = False
        self._slot = 0

    def reset(self):
        self._paths.clear()
        self._last_frame_id = None
        self._stride_checked = False

    def update(self, image: np.ndarray, frame_id: int) -> None:
        if frame_id == 0:
            self.reset()
        if frame_id == self._last_frame_id:
            return
        if not self._stride_checked:
            _check_history_stride(self._last_frame_id, frame_id, "ImageHistory")
            self._stride_checked = self._last_frame_id is not None
        self._last_frame_id = frame_id
        path = str(self._dir / f"frame_{self._slot % self._maxlen}.npy")
        self._slot += 1
        np.save(path, np.ascontiguousarray(image.astype(np.uint8)))
        self._paths.append(path)

    def paths(self) -> List[str]:
        """``maxlen`` paths, oldest first; the first frame is repeated while
        the history is still filling, so the shape the model asserts on holds
        from frame 0 rather than only after 2 s of driving."""
        paths = list(self._paths)
        if not paths:
            return []
        while len(paths) < self._maxlen:
            paths.insert(0, paths[0])
        return paths


def local_velocity_acceleration(ego_state: Dict) -> tuple[np.ndarray, np.ndarray]:
    """World-frame velocity/acceleration rotated into the ego frame."""
    heading = float(ego_state["heading"])
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, s], [-s, c]])
    vel = rot @ np.asarray(ego_state["velocity"], dtype=np.float64)[:2]
    if "acceleration" in ego_state:
        acc = rot @ np.asarray(ego_state["acceleration"], dtype=np.float64)[:2]
    else:
        acc = np.zeros(2)
    return vel, acc


def navsim_command_one_hot(ego_state: Dict) -> List[float]:
    """4-dim navsim driving command one-hot [left, straight, right, unknown]."""
    from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD

    command = ego_state.get("command", 3)
    vec = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)
    return [float(x) for x in vec]


def vla_server_dir() -> Path:
    return (Path(__file__).resolve().parents[2]
            / "modelzoo" / "navsim" / "vla_server")
