"""IDM-Centerline — single-trajectory IDM planner following the nearest lane.

This adapter is the *legacy* rule-based baseline that was previously
mis-labelled as ``pdm_closed`` in NexusSim. It is **not** PDM-Closed:
the canonical PDM-Closed algorithm (Dauner et al., CoRL 2023; reference
implementation: ``tuplan_garage.planning.simulation.planner.pdm_planner``)
generates a small batch of trajectory proposals (``5 lateral offsets ×
3 IDM policies``), forward-simulates each one with a kinematic bicycle
+ pure-pursuit tracker, scores every proposal with the PDM scorer, and
selects the best. None of that is done here — this adapter generates a
*single* trajectory by following the nearest centerline at the speed
chosen by a single hard-coded IDM policy.

It is kept as a fast, deterministic baseline so the faithful PDM-Closed
implementation under :mod:`navsafe.policy.state.pdm_closed` has a
clean A/B comparator. New code should prefer ``--model-type pdm_closed``;
``--model-type idm_centerline`` is the regression baseline.

Inheritance: :class:`StatePolicy` because ``prepare_input`` reads only
ego state and map/route information (no camera or LiDAR tensors). No
checkpoint is required — pass ``--checkpoint none`` on the CLI.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.state_policy import StatePolicy


# ── IDM parameters (Treiber, Hennecke, Helbing, 2000) ────────────────────────
# These constants match the historical pre-refactor ``pdm_closed`` adapter
# byte-for-byte so this file is a behaviourally-identical rename. The
# faithful PDM-Closed implementation in :mod:`navsafe.policy.state.pdm_closed`
# uses a *bank* of three IDM policies (aggressive / nominal / conservative)
# rather than a single fixed policy.
_IDM_DESIRED_SPEED = 11.0  # m/s (~40 km/h urban)
_IDM_TIME_HEADWAY = 1.5  # s
_IDM_MIN_GAP = 2.0  # m
_IDM_ACCEL = 1.5  # m/s²
_IDM_DECEL = 3.0  # m/s² (comfortable braking)
_IDM_DELTA = 4  # acceleration exponent


@register_policy("idm_centerline")
class IDMCenterlineAdapter(StatePolicy):
    """Single-trajectory IDM planner following the nearest centerline.

    Output contract (matches every other state policy in NexusSim):
        ``parse_output`` returns ``{"trajectory": (N, 2)}`` in ego frame
        with column order ``[lateral, forward]`` per the convention
        established by :class:`navsafe.policy.state.ego_mlp.EgoStatusMLPAdapter`.

    Trajectory shape: ``(num_poses, 2) = (8, 2)``, sampled at
    ``dt = 0.5 s`` for a 4 s horizon. ``get_waypoint_dt`` and
    ``get_trajectory_time_horizon`` return those values respectively.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.num_poses = 8  # 4 s at 0.5 s intervals
        self.dt = 0.5

    # ── BasePolicyAdapter overrides ───────────────────────────────────────────

    def load_model(self) -> None:
        """No learned weights — nothing to load."""
        # Idempotent: the evaluator may call ``load_model`` multiple times.
        # Avoid noisy reprints by guarding on the ``model`` slot we inherit
        # from ``BasePolicyAdapter`` (it stays ``None`` for rule-based
        # planners, which is fine — ``model`` is documented as optional in
        # the base class docstring).
        if getattr(self, "_loaded", False):
            return
        self._loaded = True
        print("IDM-Centerline: rule-based planner (no checkpoint required).")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {}

    def get_waypoint_dt(self) -> float:
        return self.dt

    def get_trajectory_time_horizon(self) -> float:
        return self.num_poses * self.dt

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        # Extract ego pose (XY in world frame, heading in radians, planar speed).
        pos = np.asarray(ego_state["position"][:2], dtype=np.float64)
        heading = float(ego_state["heading"])
        vel = np.asarray(ego_state["velocity"][:2], dtype=np.float64)
        speed = float(np.linalg.norm(vel))

        # Build centerline from nearest lane aligned with the ego heading.
        centerline = self._get_nearest_centerline(pos, heading, scenario_data)

        # Find lead-vehicle distance/speed for the IDM kernel.
        lead_dist, lead_speed = self._get_lead_vehicle(
            pos, heading, scenario_data, frame_id
        )

        return {
            "position": pos,
            "heading": heading,
            "speed": speed,
            "centerline": centerline,
            "lead_dist": lead_dist,
            "lead_speed": lead_speed,
        }

    def run_inference(self, model_input: Any) -> Any:
        pos: np.ndarray = model_input["position"]
        heading: float = model_input["heading"]
        speed: float = model_input["speed"]
        centerline = model_input["centerline"]
        lead_dist = model_input["lead_dist"]
        lead_speed = model_input["lead_speed"]

        # IDM longitudinal acceleration — single policy, no proposal scoring.
        accel = self._idm_accel(speed, lead_dist, lead_speed)

        # Generate a single trajectory by forward-simulating along the
        # centerline at constant acceleration. This is the *non-PDM*
        # baseline — see :mod:`navsafe.policy.state.pdm_closed` for the
        # proposal/score/select pipeline.
        trajectory = self._plan_along_centerline(
            pos, heading, speed, accel, centerline
        )
        return {"trajectory": trajectory}

    def parse_output(
        self, model_output: Any, ego_state: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        traj_world = model_output["trajectory"]  # (N, 2) world frame XY

        pos = np.asarray(ego_state["position"][:2], dtype=np.float64)
        heading = float(ego_state["heading"])

        # World → ego frame (rotate by -heading around ego origin).
        c, s = np.cos(-heading), np.sin(-heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        relative = traj_world - pos
        traj_ego = (R @ relative.T).T  # (N, 2) ego frame in [forward, left]

        # Output column order is (lateral, forward) — i.e. (left, forward) —
        # matching the NexusSim policy contract verified by every other
        # state policy and the evaluator's world-frame transform.
        traj_out = np.column_stack([traj_ego[:, 1], traj_ego[:, 0]])
        return {"trajectory": traj_out}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _idm_accel(
        self, speed: float, lead_dist: float | None, lead_speed: float
    ) -> float:
        """Single-policy IDM acceleration.

        Mirrors the original adapter's behaviour: when no lead is present
        (``lead_dist`` is ``None`` or > 100 m) we fall back to the
        free-road term only.
        """
        if lead_dist is None or lead_dist > 100.0:
            return _IDM_ACCEL * (1.0 - (speed / _IDM_DESIRED_SPEED) ** _IDM_DELTA)

        delta_v = speed - lead_speed
        s_star = _IDM_MIN_GAP + max(
            0.0,
            speed * _IDM_TIME_HEADWAY
            + speed * delta_v / (2.0 * np.sqrt(_IDM_ACCEL * _IDM_DECEL)),
        )
        gap = max(lead_dist, 0.1)
        return _IDM_ACCEL * (
            1.0 - (speed / _IDM_DESIRED_SPEED) ** _IDM_DELTA - (s_star / gap) ** 2
        )

    def _plan_along_centerline(
        self,
        pos: np.ndarray,
        heading: float,
        speed: float,
        accel: float,
        centerline: np.ndarray | None,
    ) -> np.ndarray:
        """Forward-simulate along the centerline with a constant-accel speed profile."""
        if centerline is None or len(centerline) < 2:
            # Fallback: drive straight along the current heading.
            direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
            waypoints = []
            for i in range(1, self.num_poses + 1):
                t = i * self.dt
                dist = max(speed * t + 0.5 * accel * t * t, 0.0)
                waypoints.append(pos + direction * dist)
            return np.asarray(waypoints, dtype=np.float64)

        # Project ego onto the centerline.
        diffs = centerline - pos
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = int(np.argmin(dists))

        cl = centerline[nearest_idx:]
        if len(cl) < 2:
            cl = centerline[max(0, nearest_idx - 1):]
        seg_lengths = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        waypoints = []
        for i in range(1, self.num_poses + 1):
            t = i * self.dt
            dist = max(speed * t + 0.5 * accel * t * t, 0.0)

            if dist >= cum_lengths[-1]:
                # Extrapolate beyond centerline end along last segment direction.
                # NOTE: kept ``+ 1e-8`` denominator scheme rather than
                # ``max(norm, 1e-8)`` so this rename is byte-faithful to
                # the pre-refactor ``pdm_closed`` adapter (Stage 0 of the
                # PDM-Closed refactor; see tests/policy/test_idm_centerline_stage0.py).
                direction = cl[-1] - cl[-2]
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                wp = cl[-1] + direction * (dist - cum_lengths[-1])
            else:
                idx = int(np.searchsorted(cum_lengths, dist)) - 1
                idx = max(0, min(idx, len(cl) - 2))
                # Same byte-faithfulness note as above for the segment
                # denominator.
                frac = (dist - cum_lengths[idx]) / (seg_lengths[idx] + 1e-8)
                wp = cl[idx] + frac * (cl[idx + 1] - cl[idx])
            waypoints.append(wp)

        return np.asarray(waypoints, dtype=np.float64)

    def _get_nearest_centerline(
        self,
        pos: np.ndarray,
        heading: float,
        scenario_data: Dict[str, Any],
    ) -> np.ndarray | None:
        """Find the nearest lane centerline aligned with the ego heading."""
        map_features = scenario_data.get("map_features", {})
        if not map_features:
            return None

        from navsafe.scenario.type import MetaDriveType

        best_lane: np.ndarray | None = None
        best_score = float("inf")
        ego_dir = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)

        for feat_data in map_features.values():
            if not MetaDriveType.is_lane(feat_data.get("type", "")):
                continue
            polyline = feat_data.get("polyline")
            if polyline is None or len(polyline) < 2:
                continue
            polyline = np.asarray(polyline, dtype=np.float64)[:, :2]

            dists = np.linalg.norm(polyline - pos, axis=1)
            nearest_idx = int(np.argmin(dists))
            dist = float(dists[nearest_idx])

            if nearest_idx < len(polyline) - 1:
                seg_dir = polyline[nearest_idx + 1] - polyline[nearest_idx]
            else:
                seg_dir = polyline[nearest_idx] - polyline[nearest_idx - 1]
            norm = float(np.linalg.norm(seg_dir))
            if norm < 1e-8:
                continue
            seg_dir = seg_dir / norm
            alignment = float(np.dot(ego_dir, seg_dir))

            if alignment < 0.0:
                # Opposing-direction lane — skip.
                continue
            score = dist - 5.0 * alignment

            if score < best_score:
                best_score = score
                best_lane = polyline

        return best_lane

    def _get_lead_vehicle(
        self,
        pos: np.ndarray,
        heading: float,
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> tuple[float | None, float]:
        """Find the nearest in-lane vehicle ahead of the ego."""
        tracks = scenario_data.get("tracks", {})
        sdc_id = scenario_data.get("metadata", {}).get("sdc_id", None)
        ego_dir = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        ego_left = np.array([-ego_dir[1], ego_dir[0]], dtype=np.float64)

        best_dist: float | None = None
        best_speed = 0.0

        for track_id, track_data in tracks.items():
            if track_id == sdc_id:
                continue
            states = track_data.get("state", {})
            positions = states.get("position")
            if positions is None or frame_id >= len(positions):
                continue
            # NOTE: the legacy adapter silently used the position even when
            # ``state['valid'][frame_id]`` was False (i.e. stale data). The
            # rename is byte-faithful: do *not* add a valid-frame filter here.
            # The faithful PDM-Closed planner under
            # :mod:`navsafe.policy.state.pdm_closed` handles validity
            # correctly via ``predict_agents_constant_velocity``.
            agent_pos = np.asarray(positions[frame_id][:2], dtype=np.float64)

            diff = agent_pos - pos
            longitudinal = float(np.dot(diff, ego_dir))
            lateral = abs(float(np.dot(diff, ego_left)))

            if longitudinal < 0.0 or lateral > 3.0:
                continue
            if best_dist is None or longitudinal < best_dist:
                best_dist = longitudinal
                velocities = states.get("velocity")
                if velocities is not None and frame_id < len(velocities):
                    agent_vel = np.asarray(
                        velocities[frame_id][:2], dtype=np.float64
                    )
                    best_speed = float(np.linalg.norm(agent_vel))
                else:
                    best_speed = 0.0

        return best_dist, best_speed


__all__ = ["IDMCenterlineAdapter"]
