#!/usr/bin/env python
"""DriveLaW inference server: LTX video world model + latent diffusion planner.

Port of the DriveLaW-Act eval path (xiaomi-research/drivelaw,
``navsim/agents/videodrive/videodrive_agent.py::VideoDriveAgent``) to a
closed-loop, one-request-per-replan server.

The repo's own path builds features from a navsim ``AgentInput`` inside
``compute_trajectory``.  Closed-loop there is no ``AgentInput`` and no
``Scene`` (``compute_trajectory`` also calls the *target* builder, which
needs ground-truth future poses -- unavailable when driving), so this server
calls ``forward_test`` directly with the feature dict
``VideoDriveFeatureBuilder`` produces.  That dict is small and fully
specified, and building it here is what keeps the model's own preprocessing
authoritative:

    images             (4, 3, 768, 1344) float, RGB, normalised to [-1, 1]
    history_trajectory (4, 3) ego poses in the current ego frame
    vel, acc           (2,) each, ego frame
    driving_command    (4,) navsim one-hot

The 4 conditioning frames are not a choice: ``forward_test`` asserts
``T_cond == 4`` -- the past 2 s at 2 Hz, which is the replan cadence.

``videodrive_agent`` imports ``nuplan.planning.simulation.trajectory``
for a ``TrajectorySampling`` dataclass it only stores.  The nuPlan devkit is
a heavy dependency that the inference path never calls into, so a minimal
stub is installed in ``sys.modules`` before the import.  The stub is
deliberately loud about what it is: if anything ever *uses* nuPlan beyond the
dataclass, it will fail here rather than silently return something wrong.

Run under the VLA venv:
    python drivelaw_server.py --repo <DriveLaW dir> --config <act config.yaml>

Request : {"image_npys": [p0, p1, p2, p3], "history": [[x,y,h]*4],
           "status": [cmd0..3, vx, vy, ax, ay]}
Response: {"trajectory": [[x, y, heading]*8]}   # navsim ego frame, dt 0.5 s
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The model's own resize target for the front view (videodrive_features.py).
FRONT_HW = (768, 1344)
SURROUND_HW = (256, 3072)
COND_FRAMES = 4          # forward_test asserts T_cond == 4


def _install_nuplan_stub() -> None:
    """Satisfy ``from nuplan...trajectory_sampling import TrajectorySampling``.

    DriveLaW stores the sampling spec and never uses it during inference; the
    devkit itself is not installed in the VLA venv (and pulls a database
    stack). Anything beyond the dataclass raises rather than silently
    degrading.
    """
    if "nuplan" in sys.modules:
        return
    try:
        # DriveLaW's navsim fork imports the devkit far beyond TrajectorySampling
        # (StateSE2, map layers, tracked-object types). Where the real devkit is
        # importable, defer to it -- a stub installed on top would shadow it and
        # turn those into "'nuplan' is not a package".
        import nuplan.common.actor_state.state_representation  # noqa: F401
        return
    except ImportError:
        pass

    @dataclass
    class TrajectorySampling:  # noqa: D401 - mirrors the devkit's fields
        num_poses: int = 8
        interval_length: float = 0.5
        time_horizon: float = 4.0

    def _missing(name):
        def _raise(attr: str):
            # Dunders must answer normally: torch's custom-op registration walks
            # sys.modules and probes `__file__`/`__spec__` on every entry, and a
            # stub that raises there takes down an unrelated import with a
            # message about the nuPlan devkit. Only real API access should fail.
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            raise ModuleNotFoundError(
                f"nuplan.{name}.{attr} is stubbed in drivelaw_server: the "
                "DriveLaW inference path is not supposed to reach the nuPlan "
                "devkit")
        return _raise

    for name in ("nuplan", "nuplan.planning", "nuplan.planning.simulation",
                 "nuplan.planning.simulation.trajectory"):
        mod = types.ModuleType(name)
        mod.__getattr__ = _missing(name)  # type: ignore[attr-defined]
        sys.modules[name] = mod
    leaf = types.ModuleType("nuplan.planning.simulation.trajectory.trajectory_sampling")
    leaf.TrajectorySampling = TrajectorySampling
    sys.modules[leaf.__name__] = leaf


def _load_frames(paths, view_mode: str) -> torch.Tensor:
    """(T, 3, H, W) in [-1, 1], resized as VideoDriveFeatureBuilder does."""
    import torch.nn.functional as F

    imgs = []
    for p in paths:
        arr = np.load(p)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{p}: expected an (H, W, 3) RGB frame, got {arr.shape}")
        t = torch.from_numpy(arr.astype(np.float32) / 255.0)
        imgs.append((t * 2.0 - 1.0).permute(2, 0, 1).contiguous())
    video = torch.stack(imgs, dim=0)
    h, w = SURROUND_HW if view_mode == "surround6" else FRONT_HW
    if video.shape[-2:] != (h, w):
        video = F.interpolate(video, size=(h, w), mode="bilinear",
                              align_corners=False)
    return video


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="DriveLaW checkout (the dir holding DriveLaW-Act)")
    ap.add_argument("--config", required=True,
                    help="DriveLaW-Act inference config yaml (config_file)")
    ap.add_argument("--view-mode", default="front", choices=["front", "surround6"])
    ap.add_argument("--num-poses", type=int, default=8)
    args = ap.parse_args()

    _install_nuplan_stub()
    act_root = Path(args.repo) / "DriveLaW-Act"
    if not act_root.is_dir():
        raise FileNotFoundError(f"DriveLaW-Act not found under {args.repo}")
    sys.path.insert(0, str(act_root))

    from nuplan.planning.simulation.trajectory.trajectory_sampling import (  # noqa: E402
        TrajectorySampling,
    )
    from navsim.agents.videodrive.videodrive_agent import VideoDriveAgent  # noqa: E402

    agent = VideoDriveAgent(
        trajectory_sampling=TrajectorySampling(
            num_poses=args.num_poses, interval_length=0.5,
            time_horizon=0.5 * args.num_poses),
        config_file=args.config,
        weight_dtype=torch.bfloat16,
        device="cuda",
        view_mode=args.view_mode,
    )
    agent.initialize()
    agent.eval()

    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            paths = req["image_npys"]
            if len(paths) != COND_FRAMES:
                raise ValueError(
                    f"DriveLaW conditions on exactly {COND_FRAMES} frames "
                    f"(forward_test asserts it); got {len(paths)}")
            status = [float(v) for v in req["status"]]
            features = {
                "images": _load_frames(paths, args.view_mode),
                "history_trajectory": torch.tensor(req["history"], dtype=torch.float32),
                "vel": torch.tensor(status[4:6], dtype=torch.float32),
                "acc": torch.tensor(status[6:8], dtype=torch.float32),
                "driving_command": torch.tensor(status[0:4], dtype=torch.float32),
            }
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            with torch.no_grad():
                out = agent.forward_test(features)
            actions = np.asarray(out["actions"], dtype=np.float32).reshape(-1, 3)
            print(json.dumps({"trajectory": actions.tolist()}), flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
