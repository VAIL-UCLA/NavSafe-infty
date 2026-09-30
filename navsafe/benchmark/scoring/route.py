"""Route progress and off-route distance, in Bench2Drive's terms.

Two quantities that every Driving Score needs and that a NavSafe trace does not
store directly, because both are *route*-relative and the trace records world
state (schema rule 1: facts, not derived verdicts).  Both are pure functions of
stored arrays.

* **Route completion** (``RouteCompletionTest``, ``atomic_criteria.py``) is the
  fraction of the route the ego actually traversed, and it is **monotone**: the
  test walks a cursor forward along the route waypoints and never lets it go
  back, so driving away and returning cannot be double-counted, and ending near
  a late waypoint does not retro-credit the route in between.  A nearest-point
  query over the whole route -- the obvious implementation -- is *not* this:
  it awards a policy that leaves the route and stops beside its far end the
  full 100 %.
* **Outside-route-lanes** (``OutsideRouteLanesTest`` /
  ``statistics_manager.py`` L31-38) is the percentage of driven distance spent
  off the drivable surface, with a 0.5 m allowance, and it is the only
  *proportional* Driving Score penalty: ``score_penalty *= 1 - pct/100``.

Both are reported here with their own denominators so a caller can see what
they were computed over, rather than only their effect on a score.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

# RouteCompletionTest treats the route as complete at > 99 % traversed, and the
# leaderboard requires the ego to actually be at the target (10 m).
COMPLETION_PCT_FOR_DONE = 99.0
GOAL_RADIUS_M = 10.0
# ...but 99 % is a *fraction*, and the leaderboard sized it for km-scale routes.
# A NavSafe seed is a ~10 s event, and its route is the logged ego path, which
# can be very short: across navhard421 they run 15 m to 155 m, so the same 1 %
# demands 0.15 m of arc precision on one seed and 1.55 m on another. The strict
# end of that range is below what the tracker can hold, which made the goal
# unreachable on short seeds — a policy stopping 0.6 m from the goal scored
# 94.35 % and was recorded as a route_timeout rather than a completion.
#
# So the arc gate is metres-based, with the percentage kept only as a floor for
# long routes. 2.0 m is half a vehicle length: stopping half a car short of
# where the logged ego stopped is completing the route, not failing it. Taking
# the MAX means this can only ever loosen the test — no episode that completed
# under the pure-percentage rule can stop completing under this one.
GOAL_ARC_TOL_M = 2.0
# OutsideRouteLanesTest ALLOWED_OUT_DISTANCE (atomic_criteria.py): distance off
# the drivable surface below this is not counted as wrong distance.
ALLOWED_OUT_DISTANCE_M = 0.5


def goal_arc_tolerance_m(total_arc_m: float) -> float:
    """Metres of un-driven route still counted as "reached the goal"."""
    return max(GOAL_ARC_TOL_M,
               (1.0 - COMPLETION_PCT_FOR_DONE / 100.0) * float(total_arc_m))


def at_goal(arc_done_m: float, total_arc_m: float, dist_to_goal_m: float) -> bool:
    """The one goal predicate, shared by the live and post-hoc tests.

    Both the evaluator (``RouteManager.goal_reached``, live, per step) and the
    scorer (:func:`route_progress`, post-hoc) answer "did the ego finish the
    route". They MUST agree — a run that ends on `goal_reached` and is then
    scored as `budget_expired` fails success() twice over on an episode that
    completed. Keeping the rule in one function is what makes that structural
    rather than a comment.
    """
    if total_arc_m <= 0:
        return False
    remaining = float(total_arc_m) - float(arc_done_m)
    return (remaining <= goal_arc_tolerance_m(total_arc_m)
            and float(dist_to_goal_m) < GOAL_RADIUS_M)


@dataclass
class RouteProgress:
    completion_pct: float          # 0-100, monotone, at the last frame
    completed: bool                # see at_goal(): arc remaining within
                                   #   tolerance AND within GOAL_RADIUS_M
    dist_to_goal_m: float
    goal_frame: int | None         # first frame at which ``completed`` held
    per_frame_pct: list[float]     # monotone non-decreasing, one per ego frame

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("per_frame_pct")
        return d

    def at(self, frame: int) -> float:
        """Completion as of ``frame`` -- what an episode that ended there earned."""
        if not self.per_frame_pct:
            return 0.0
        i = max(0, min(int(frame), len(self.per_frame_pct) - 1))
        return self.per_frame_pct[i]


def route_progress(route_xy, ego_xy) -> RouteProgress:
    """Monotone route completion of ``ego_xy`` against the route ``route_xy``.

    ``route_xy`` is the reference path (for a NavSafe seed, the logged ego
    trajectory); ``ego_xy`` is the driven path, one row per frame.  The cursor
    only ever advances, so completion is non-decreasing -- a policy that drives
    off and comes back resumes from where it left the route rather than
    collecting the distance it never drove.
    """
    route = np.asarray(route_xy, dtype=np.float64)[:, :2]
    ego = np.asarray(ego_xy, dtype=np.float64)[:, :2]
    if len(route) < 2 or len(ego) == 0:
        return RouteProgress(0.0, False, float("inf"), None, [])

    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))])
    total = float(arc[-1])
    if total <= 0:
        return RouteProgress(0.0, False, float("inf"), None, [])

    per_frame: list[float] = []
    goal_frame: int | None = None
    cursor = 0
    for f, p in enumerate(ego):
        # Cursor walks forward only (RouteCompletionTest's monotone index).
        j = cursor + int(np.argmin(np.linalg.norm(route[cursor:] - p, axis=1)))
        cursor = max(cursor, j)
        pct = 100.0 * float(arc[cursor]) / total
        per_frame.append(pct)
        if goal_frame is None and at_goal(
                float(arc[cursor]), total, float(np.linalg.norm(p - route[-1]))):
            goal_frame = f

    dist_to_goal = float(np.linalg.norm(ego[-1] - route[-1]))
    completed = goal_frame is not None
    return RouteProgress(
        completion_pct=100.0 if completed else per_frame[-1],
        completed=completed,
        dist_to_goal_m=dist_to_goal,
        goal_frame=goal_frame,
        per_frame_pct=per_frame,
    )


def outside_route_lanes_pct(ego_xy, on_drivable: Sequence[bool], *,
                            allowed_m: float = ALLOWED_OUT_DISTANCE_M) -> float:
    """Percentage of driven distance spent off the drivable surface.

    Bench2Drive's ``outside_route_lanes`` value, which enters the Driving Score
    as ``score_penalty *= 1 - pct/100``.  The first ``allowed_m`` of an
    excursion is free (``ALLOWED_OUT_DISTANCE``); an excursion shorter than
    that contributes nothing.
    """
    ego = np.asarray(ego_xy, dtype=np.float64)[:, :2]
    flags = list(on_drivable)
    if len(ego) < 2 or len(flags) != len(ego):
        return 0.0
    step = np.hypot(*np.diff(ego, axis=0).T)
    total = float(step.sum())
    if total <= 0:
        return 0.0

    wrong = 0.0
    run = 0.0
    for k, d in enumerate(step):
        # A step counts as off-route when the frame it ends on is off-drivable.
        if not flags[k + 1]:
            run += float(d)
        elif run:
            wrong += max(0.0, run - allowed_m)
            run = 0.0
    if run:
        wrong += max(0.0, run - allowed_m)
    return 100.0 * wrong / total
