"""Route progress and off-route distance (navsafe/benchmark/scoring/route.py)."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.scoring import route as R


def _straight(n=101, step=1.0):
    return np.stack([np.arange(n) * step, np.zeros(n)], axis=1)


def test_full_traversal_completes():
    route = _straight()
    p = R.route_progress(route, route)
    assert p.completed
    assert p.completion_pct == pytest.approx(100.0)
    assert p.dist_to_goal_m == pytest.approx(0.0)
    assert p.goal_frame is not None


def test_half_traversal_is_half():
    route = _straight()
    ego = route[:51]
    p = R.route_progress(route, ego)
    assert not p.completed
    assert p.completion_pct == pytest.approx(50.0, abs=1.0)
    assert p.dist_to_goal_m == pytest.approx(50.0)


def test_progress_is_per_frame_and_monotone():
    """Completion must be readable *as of a frame*, because an episode ends at
    its termination frame and the motion after that is not the episode's.

    A final-position-only implementation (what the ad-hoc scorer did) cannot
    answer that question at all: it has one number, computed from wherever the
    ego happened to stop.
    """
    route = _straight()
    ego = np.concatenate([route[:11],
                          np.stack([np.linspace(10, 100, 40),
                                    np.linspace(0, 40, 40)], axis=1)])
    p = R.route_progress(route, ego)
    assert all(b >= a - 1e-9 for a, b in zip(p.per_frame_pct, p.per_frame_pct[1:]))
    # As of the frame it left the route, it had earned 10 %, not its final value.
    assert p.at(10) == pytest.approx(10.0, abs=1.0)
    assert p.at(10) < p.at(len(ego) - 1)


def test_backtracking_does_not_lose_progress():
    route = _straight()
    ego = np.concatenate([route[:61], route[60::-1]])   # drive out and back
    p = R.route_progress(route, ego)
    assert p.at(len(ego) - 1) == pytest.approx(60.0, abs=1.0)


def test_goal_needs_both_completion_and_radius():
    route = _straight()
    # Ends 20 m off to the side of the final waypoint: > GOAL_RADIUS_M.
    ego = route.copy()
    ego[-5:, 1] = 20.0
    p = R.route_progress(route, ego)
    assert not p.completed and p.goal_frame is None


def test_goal_tolerance_is_metres_not_a_fraction_on_short_routes():
    """A 15 m route must not demand 0.15 m of arc precision.

    The navhard421 seeds run 15-155 m, so a pure 99 % rule made the goal 10x
    harder to reach on the shortest seed than the longest. Measured case: an
    ego that stopped 0.87 m short of a 15.36 m route scored 94.35 % and was
    recorded as a route_timeout rather than a completion.
    """
    route = _straight(n=16)                      # 15 m
    ego = route[:-1].copy()                      # stops 1.0 m short of the end
    p = R.route_progress(route, ego)
    assert p.completed and p.goal_frame is not None
    # 1 % of 15 m is 0.15 m, which this ego is nowhere near.
    assert R.goal_arc_tolerance_m(15.0) == pytest.approx(R.GOAL_ARC_TOL_M)


def test_goal_tolerance_keeps_the_percentage_on_long_routes():
    """Above ~200 m the 1 % rule is the looser of the two and still governs."""
    assert R.goal_arc_tolerance_m(1000.0) == pytest.approx(10.0)   # 1 %, not 2 m
    assert R.goal_arc_tolerance_m(50.0) == pytest.approx(R.GOAL_ARC_TOL_M)


def test_goal_tolerance_never_tightens():
    """The fix may only loosen: nothing that completed before can stop doing so."""
    for total in (5.0, 15.36, 41.8, 100.0, 155.0, 400.0, 2000.0):
        pct_only = (1.0 - R.COMPLETION_PCT_FOR_DONE / 100.0) * total
        assert R.goal_arc_tolerance_m(total) >= pct_only - 1e-9


def test_goal_still_needs_the_radius_on_a_short_route():
    """The metres floor loosens the ARC gate only — not the 'be there' gate."""
    route = _straight(n=16)
    ego = route.copy()
    ego[-3:, 1] = 20.0                           # ends 20 m off to the side
    p = R.route_progress(route, ego)
    assert not p.completed and p.goal_frame is None


def test_stopping_well_short_is_still_not_a_completion():
    route = _straight()                          # 100 m
    ego = route[:81].copy()                      # 20 m short, way past tolerance
    p = R.route_progress(route, ego)
    assert not p.completed and p.goal_frame is None


def test_outside_route_lanes_allowance():
    ego = _straight(n=101)                      # 100 m driven
    on = [True] * 101
    assert R.outside_route_lanes_pct(ego, on) == pytest.approx(0.0)

    # A 0.4 m excursion is inside ALLOWED_OUT_DISTANCE and costs nothing.
    short = _straight(n=101, step=0.2)          # 20 m driven, 0.2 m per step
    flags = [True] * 101
    flags[50] = flags[51] = False               # 0.4 m off
    assert R.outside_route_lanes_pct(short, flags) == pytest.approx(0.0)

    # 10 m of a 100 m route off-drivable -> (10 - 0.5) / 100.
    on = [True] * 101
    for i in range(50, 60):
        on[i] = False
    assert R.outside_route_lanes_pct(ego, on) == pytest.approx(9.5, abs=0.6)
