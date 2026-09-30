"""ScriptedPath — replay a pre-authored world-frame path as if it were a plan.

This adapter exists for **figure generation**, not for benchmarking. It
ignores every observation and emits, at each frame, the next slice of a
world-frame path loaded from a ``.npy`` file. It is the mechanism behind
NavSafe Figure 1: four hand-authored deviations from one scenario's logged
ego path, each constructed to end in a different taxonomy termination
(``off_drivable``, vehicle contact, pedestrian contact, ``wrong_way``).

Because the path is authored in *world* coordinates, "leave the drivable
area here" or "cross this actor's box at this frame" is expressible
directly; the adapter re-expresses the upcoming waypoints in the ego frame
each time the evaluator asks for a plan, so tracking stays correct under
any ``--replan-rate``.

Inheritance: :class:`StatePolicy` — ``prepare_input`` reads only ego pose.
No checkpoint is required; pass ``--checkpoint none`` and point
``NAVSAFE_SCRIPTED_PATH`` at the ``.npy``.

Path file format: ``(M, 2)`` or ``(M, 3)`` float array of world XY(Z)
positions, dense in time at the evaluator's ``sim_dt`` (0.1 s). The adapter
resamples by arc length onto its own ``0.5 s`` waypoint grid using the
ego's current speed, so the authored path defines *geometry*; the speed
profile follows from the path's own spacing.

Use with ``--execution-mode teleport`` for an exact reproduction of the
authored geometry, or the default ``controller`` mode to keep the LQR
tracker (and therefore realistic dynamics) in the loop.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
from navsafe.policy.registry import register_policy
from navsafe.policy.state_policy import StatePolicy

_NUM_POSES = 8      # 8 waypoints
_WAYPOINT_DT = 0.5  # at 0.5 s => 4 s horizon, the NavSafe convention
_SIM_DT = 0.1       # the evaluator's step; the authored path is dense at this rate


@register_policy("scripted_path")
class ScriptedPathAdapter(StatePolicy):
    """Emit slices of a pre-authored world-frame path as the plan."""

    def __init__(self, checkpoint_path: str = "none", path_npy: str | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.num_poses = _NUM_POSES
        self._path_npy = path_npy or os.environ.get("NAVSAFE_SCRIPTED_PATH")
        self._path: np.ndarray | None = None
        self._arc: np.ndarray | None = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the authored path. No learned weights."""
        if self._loaded:
            return
        if not self._path_npy:
            raise ValueError(
                "scripted_path needs a path: set NAVSAFE_SCRIPTED_PATH=/path/to.npy "
                "or pass path_npy=")
        p = np.load(self._path_npy)
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 2:
            raise ValueError(
                f"scripted path {self._path_npy} must be (M,>=2) with M>=2, got {p.shape}")
        self._path = p[:, :2]
        # Cumulative arc length, so the plan can be resampled by distance
        # rather than by index — index pacing would couple the emitted plan
        # to the authoring resolution.
        seg = np.linalg.norm(np.diff(self._path, axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(seg)])
        self._loaded = True
        print(f"[scripted_path] loaded {self._path.shape[0]} pts, "
              f"{self._arc[-1]:.1f} m from {self._path_npy}")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """Request CAM_F0 so the evaluator renders the real reconstruction.

        An adapter that asks for no cameras gets the evaluator's synthetic
        agent-bbox fallback -- flat coloured rectangles, not the neural
        render. That is correct for a policy which never looks at images,
        but this adapter exists to produce FIGURE frames, so it needs the
        photoreal view even though `prepare_input` ignores it. Same reason
        pdm_closed (also a blind planner) requests CAM_F0.
        """
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    def get_waypoint_dt(self) -> float:
        return _WAYPOINT_DT

    def get_trajectory_time_horizon(self) -> float:
        return _NUM_POSES * _WAYPOINT_DT

    def _project(self, xy: np.ndarray) -> float:
        """Arc length of the path point nearest ``xy`` (the ego's progress)."""
        d = np.linalg.norm(self._path - xy[None, :2], axis=1)
        i = int(np.argmin(d))
        # Refine within the neighbouring segment so progress is continuous
        # rather than quantised to the authoring resolution.
        best = self._arc[i]
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self._path):
                continue
            a, b = self._path[j], self._path[j + 1]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 <= 1e-12:
                continue
            t = float(np.clip((xy[:2] - a) @ ab / L2, 0.0, 1.0))
            proj = a + t * ab
            if np.linalg.norm(xy[:2] - proj) <= d[i] + 1e-9:
                best = self._arc[j] + t * np.sqrt(L2)
        return best

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        pos = np.asarray(ego_state["position"], dtype=np.float64)
        vel = np.asarray(ego_state.get("velocity", [0.0, 0.0]), dtype=np.float64)
        return {
            "position": pos[:2],
            "heading": float(ego_state["heading"]),
            "speed": float(np.linalg.norm(vel[:2])),
            "frame": int(frame_id),
        }

    def run_inference(self, model_input: Any) -> Any:
        """Slice the authored path ahead of the ego. No network involved.

        Pacing is by FRAME INDEX, not by speed. The authored path is dense at
        the sim's own 0.1 s grid, so index i is "where the ego should be at
        frame i" -- that timing is part of the authoring, because a trajectory
        has to meet a specific moving actor at a specific frame.

        Speed-based pacing was measured to run 15-19 frames behind the authored
        schedule by the end of an episode (progress ratio 0.63-0.84): the ego
        tracked the right curve to within 0.25 m but arrived late, so it met
        different actors than the ones the path was designed to reach. Emitting
        the waypoints the path itself specifies for the frames ahead removes
        that drift entirely.
        """
        pos = model_input["position"]
        frame = int(model_input.get("frame", 0))

        # Waypoints at frame + k*(waypoint_dt/sim_dt), i.e. the authored poses
        # at the times this plan spans. Clipped at the end of the path.
        stride = max(1, int(round(_WAYPOINT_DT / _SIM_DT)))
        idx = frame + stride * np.arange(1, _NUM_POSES + 1)
        idx = np.clip(idx, 0, len(self._path) - 1)

        w = self._path[idx]
        return {"world": w, "position": pos, "heading": model_input["heading"]}

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]
                     ) -> Dict[str, np.ndarray]:
        """World waypoints -> ego frame ``[lateral, forward]``.

        The evaluator documents the parsed plan as ``(N, 2)`` in the ego
        frame ordered ``[lateral, forward]`` and applies the inverse of
        this rotation to get back to world, so the two must agree exactly.
        """
        world = model_output["world"]
        pos = np.asarray(model_output["position"], dtype=np.float64)[:2]
        h = float(model_output["heading"])
        c, s = np.cos(h), np.sin(h)
        d = world - pos[None, :]
        forward = c * d[:, 0] + s * d[:, 1]
        lateral = -s * d[:, 0] + c * d[:, 1]
        return {"trajectory": np.column_stack([lateral, forward])}


__all__ = ["ScriptedPathAdapter"]
