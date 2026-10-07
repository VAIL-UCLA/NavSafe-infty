"""RouteManager - the route/waypoint subsystem of :class:`Evaluator`.

Generates the BridgeSim-style route (a deque of waypoints derived from the GT
trajectory) and consumes it during evaluation (``get_next_waypoint`` pops
waypoints as the ego advances). Extracted from the ``Evaluator`` God class.

Ownership: this manager OWNS the route state (``route`` / ``full_route``); it is
the only writer. The evaluator's ``_reset_state`` delegates to :meth:`reset`, and
exposes ``route`` / ``full_route`` read-only properties so the artifact writer
(which reads ``full_route``) keeps working. The transient per-step
``_current_waypoint`` (the *result* of get_next_waypoint, consumed by the step
loop) stays on the evaluator. Shared reads (env, config, scenario data, the
trajectory helpers) forward to the evaluator via ``__getattr__``.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Optional, cast

import numpy as np

try:
    from navsafe.evaluation import vis_utils
except ImportError:  # pragma: no cover — vis_utils imports cv2 (viz extra)
    vis_utils = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class RouteManager:
    """Route generation + waypoint consumption for an :class:`Evaluator`."""

    def __init__(self, evaluator: Any) -> None:
        self._eval = evaluator
        self.reset()

    def reset(self) -> None:
        """Clear the route state (called by the evaluator's _reset_state)."""
        self.route: Optional[deque] = None        # deque of (position, command, frame_idx)
        self.full_route: Optional[list] = None   # list copy for visualisation (never consumed)
        # Live goal test state: a monotone cursor over the DENSE ego path.
        # Not full_route — those waypoints are subsampled at 1 m for navigation,
        # and testing the goal against them quantises the cursor to that
        # spacing: the ego 0.41 m from the last waypoint snapped onto it and
        # read exactly 100 %, while one frame earlier (0.74 m) it dropped to
        # 92.91 % and stayed flat there over the next several frames. Episode
        # termination was being decided by waypoint spacing. The scorer walks
        # the dense logged path, so this walks the same array.
        self._goal_path: Optional[np.ndarray] = None
        self._goal_arc: Optional[np.ndarray] = None
        self._goal_cursor: int = 0
        self._goal_reached: bool = False

    @property
    def goal_latched(self) -> bool:
        """Whether :meth:`goal_reached` has fired at any point this episode.

        Read at teardown, when the env may already be gone and there is no ego
        pose left to test.
        """
        return self._goal_reached

    def goal_reached(self, ego_xy: Any) -> bool:
        """Whether the ego has completed the route, by the scoring rule.

        Calls the SAME predicate as the post-hoc scorer
        (:func:`navsafe.benchmark.scoring.route.at_goal`) over the SAME dense
        path, so a live ending and the post-hoc verdict cannot disagree about
        whether the route was finished. They used to share only the two
        constants while walking different polylines, which is not the same
        thing and did disagree.

        The cursor only advances, matching Bench2Drive's monotone
        ``RouteCompletionTest``: a policy that leaves the route and returns
        resumes where it left, and stopping beside a late waypoint does not
        retro-credit the stretch it never drove.

        Latches: once the goal is reached it stays reached, so a policy that
        overshoots past the goal radius cannot un-finish the route.
        """
        if self._goal_reached:
            return True
        route = self._goal_path
        if route is None or len(route) < 2:
            return False
        from navsafe.benchmark.scoring.route import at_goal

        # _goal_arc is only ever written in the same statement pair that writes
        # _goal_path, so a non-None route (checked above) proves a non-None arc.
        arc = cast(np.ndarray, self._goal_arc)
        total = float(arc[-1])
        if total <= 0:
            return False
        p = np.asarray(ego_xy, dtype=np.float64)[:2]
        j = self._goal_cursor + int(np.argmin(
            np.linalg.norm(route[self._goal_cursor:] - p, axis=1)))
        self._goal_cursor = max(self._goal_cursor, j)
        if at_goal(float(arc[self._goal_cursor]), total,
                   float(np.linalg.norm(p - route[-1]))):
            self._goal_reached = True
        return self._goal_reached

    def policy_context(self, ego_xy: Any) -> dict[str, Any]:
        """Expose the existing completion predicate and remaining mission.

        Endpoint distance alone is insufficient on loops and nearby parallel
        route legs. The monotone cursor and goal predicate stay authoritative.
        """
        route, arc = self._goal_path, self._goal_arc
        if route is None or arc is None or len(route) < 2:
            return {}
        xy = np.asarray(ego_xy, dtype=np.float64).reshape(-1)[:2]
        if len(xy) != 2 or not np.isfinite(xy).all():
            return {}
        complete = self.goal_reached(xy)
        return {
            "_execution_route_complete": complete,
            "_execution_route_remaining_m": max(0.0, float(arc[-1] - arc[self._goal_cursor])),
            "_execution_goal_distance_m": float(np.linalg.norm(xy - route[-1])),
        }

    def __getattr__(self, name: str) -> Any:
        if name == "_eval":
            raise AttributeError(name)
        return getattr(self._eval, name)

    def generate_route(self) -> None:
        """Generate static route waypoints and driving commands from GT trajectory.

        Ports BridgeSim's generate_route(): subsamples ego GT trajectory at 1m
        spacing, computes heading-based commands (LEFT/RIGHT/STRAIGHT/LANEFOLLOW)
        using a 3° threshold between consecutive waypoints.

        Stores:
            self.route: deque of (position_2d, command, frame_idx) — consumed during eval
            self.full_route: list copy for visualization (never consumed)
        """
        from collections import deque
        from navsafe.scenario.scenario_description import ScenarioDescription as SD

        scenario = getattr(self.env, "current_scenario", None)
        if scenario is None:
            logger.warning("No scenario loaded — cannot generate route")
            return

        metadata = scenario.get(SD.METADATA, {})
        sdc_id = str(metadata.get(SD.SDC_ID, ""))
        tracks = scenario.get(SD.TRACKS, {})
        ego_track = tracks.get(sdc_id)
        if ego_track is None:
            logger.warning(f"Ego track {sdc_id} not found — cannot generate route")
            return

        positions = ego_track.get(SD.STATE, {}).get("position")
        headings = ego_track.get(SD.STATE, {}).get("heading")
        if positions is None or headings is None:
            logger.warning("Ego track missing position/heading — cannot generate route")
            return

        # The goal test walks the FULL-resolution logged path, not the 1 m
        # waypoints below: the scorer walks this same array, and subsampling
        # first would quantise the goal decision to the waypoint spacing.
        dense = np.asarray(positions, dtype=np.float64)[:, :2]
        if len(dense) >= 2:
            self._goal_path = dense
            self._goal_arc = np.concatenate(
                [[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])

        waypoint_spacing = 1.0
        turn_threshold_deg = 3.0

        # Subsample GT trajectory at waypoint_spacing intervals
        waypoints: list[dict[str, Any]] = []
        last_pos = None
        for i in range(len(positions)):
            pos = np.asarray(positions[i])[:2]
            if last_pos is None or np.linalg.norm(pos - last_pos) >= waypoint_spacing:
                waypoints.append({
                    "position": pos.copy(),
                    "heading": float(headings[i]),
                    "frame_idx": i,
                })
                last_pos = pos

        # Generate commands based on heading changes between consecutive waypoints
        self.route = deque()
        for i in range(len(waypoints)):
            wp = waypoints[i]
            if i < len(waypoints) - 1:
                heading_diff = np.rad2deg(waypoints[i + 1]["heading"] - wp["heading"])
                # Normalize to [-180, 180]
                while heading_diff > 180:
                    heading_diff -= 360
                while heading_diff < -180:
                    heading_diff += 360

                if abs(heading_diff) < turn_threshold_deg:
                    command = vis_utils.CMD_STRAIGHT
                elif heading_diff > turn_threshold_deg:
                    command = vis_utils.CMD_LEFT
                else:
                    command = vis_utils.CMD_RIGHT
            else:
                command = vis_utils.CMD_LANEFOLLOW

            self.route.append((wp["position"], command, wp["frame_idx"]))

        self.full_route = list(self.route)
        import sys
        sys.stderr.write(f"[ROUTE] Generated {len(self.route)} route waypoints from GT trajectory\n")
        sys.stderr.flush()

    def get_next_waypoint(self, current_position: np.ndarray):
        """Get next navigation waypoint and command by popping reached waypoints.

        Ports BridgeSim's get_next_waypoint(): pops waypoints within 4m of
        current position, returns the next upcoming waypoint + command.

        Args:
            current_position: (2,) current ego XY position.

        Returns:
            (waypoint_pos, command, frame_idx) or None if no route.
        """
        if self.route is None or len(self.route) == 0:
            return None

        min_distance = 4.0
        max_distance = 50.0

        if len(self.route) == 1:
            return self.route[0]

        # Find how many waypoints have been reached (within min_distance)
        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > max_distance:
                break
            cumulative_distance += float(np.linalg.norm(
                self.route[i][0] - self.route[i - 1][0]
            ))
            distance = float(np.linalg.norm(self.route[i][0] - current_position[:2]))
            if distance <= min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        # Pop reached waypoints (keep at least 2)
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        # Return next waypoint (index 1 if available, else 0)
        next_wp = self.route[1] if len(self.route) > 1 else self.route[0]
        return next_wp
