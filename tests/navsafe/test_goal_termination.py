# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Finishing the route must end the episode.

`GOAL_REACHED` existed in the taxonomy and `LiveMonitor.update` accepted a
`goal_reached` flag, but the eval loop never passed one -- so an episode that
completed its route kept driving past the end until something else stopped it,
and whatever it hit out there was charged to the policy. One observed run
finished 99.7 % of its route and was then scored as an at-fault collision.
"""

from __future__ import annotations

import unittest

import numpy as np

from navsafe.evaluation.route_manager import RouteManager
from navsafe.benchmark.termination import LiveMonitor, TerminationReason
from navsafe.benchmark.scoring import route as route_mod
from navsafe.benchmark.scoring.route import GOAL_ARC_TOL_M, GOAL_RADIUS_M


def _route_manager(route_xy) -> RouteManager:
    """A RouteManager with a canned route and no live evaluator behind it."""
    rm = RouteManager.__new__(RouteManager)      # skip __init__'s evaluator wiring
    rm._eval = None
    rm.reset()
    rm.full_route = [(np.asarray(p, dtype=np.float64), "STRAIGHT", i)
                     for i, p in enumerate(route_xy)]
    # The goal test walks the dense path, not the 1 m navigation waypoints;
    # generate_route() populates these two together.
    dense = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    if len(dense) >= 2:
        rm._goal_path = dense
        rm._goal_arc = np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(dense, axis=0).T))])
    return rm


# A straight 100 m route along +x, one waypoint per metre.
STRAIGHT = [(float(x), 0.0) for x in range(101)]


class TestRouteManagerGoal(unittest.TestCase):

    def test_start_of_route_is_not_the_goal(self) -> None:
        rm = _route_manager(STRAIGHT)
        self.assertFalse(rm.goal_reached((0.0, 0.0)))
        self.assertFalse(rm.goal_latched)

    def test_driving_the_route_reaches_the_goal(self) -> None:
        rm = _route_manager(STRAIGHT)
        reached_at = None
        for x in range(101):
            if rm.goal_reached((float(x), 0.0)):
                reached_at = x
                break
        self.assertIsNotNone(reached_at)
        # Within GOAL_ARC_TOL_M of the end of a 100 m route (the 1 % rule would
        # be 1 m here, so the 2 m floor governs), and inside GOAL_RADIUS_M.
        self.assertGreaterEqual(reached_at, 100 - int(GOAL_ARC_TOL_M))

    def test_goal_latches_once_reached(self) -> None:
        """Overshooting past the goal radius must not un-finish the route."""
        rm = _route_manager(STRAIGHT)
        for x in range(101):
            rm.goal_reached((float(x), 0.0))
        self.assertTrue(rm.goal_latched)
        far = (100.0 + 5 * GOAL_RADIUS_M, 0.0)
        self.assertTrue(rm.goal_reached(far))

    def test_near_the_end_but_short_is_not_the_goal(self) -> None:
        """95 % traversed is below COMPLETION_PCT_FOR_DONE."""
        rm = _route_manager(STRAIGHT)
        for x in range(96):
            rm.goal_reached((float(x), 0.0))
        self.assertFalse(rm.goal_latched)

    def test_no_route_is_never_the_goal(self) -> None:
        rm = _route_manager([])
        self.assertFalse(rm.goal_reached((0.0, 0.0)))


class TestLiveAndPostHocAgree(unittest.TestCase):
    """The live goal test and the scorer's must return the same verdict.

    They used to share only the two constants while walking different
    polylines -- the evaluator a 1 m subsample, the scorer the dense logged
    path -- which is not the same rule. Measured divergence on
    navhard421/028613e11f415422: the live test read exactly 100.00 % (its
    cursor snapped onto the final waypoint) while the scorer read 94.35 %, so
    the run ended `goal_reached` and was scored `budget_expired` +
    route_timeout. One frame earlier the live test collapsed to 92.91 % and
    stayed flat for several frames -- termination decided by waypoint spacing.
    """

    def _both(self, route_xy, ego_xy):
        rm = _route_manager(route_xy)
        live = False
        for p in ego_xy:
            live = rm.goal_reached(p) or live
        post = route_mod.route_progress(
            np.asarray(route_xy, float), np.asarray(ego_xy, float)).completed
        return live, post

    def test_agree_when_stopping_just_short_on_a_short_route(self) -> None:
        route = [(float(x) * 0.1, 0.0) for x in range(154)]   # 15.3 m
        ego = route[:-9]                                      # ~0.9 m short
        live, post = self._both(route, ego)
        self.assertTrue(live)
        self.assertEqual(live, post)

    def test_agree_across_a_sweep_of_stopping_distances(self) -> None:
        """No stopping point may make the two disagree, on either scale."""
        for n, step in ((154, 0.1), (101, 1.0), (201, 1.0)):
            route = [(float(i) * step, 0.0) for i in range(n)]
            for cut in range(0, 40, 3):
                ego = route[:len(route) - cut]
                if len(ego) < 2:
                    continue
                live, post = self._both(route, ego)
                self.assertEqual(
                    live, post,
                    f"live={live} post={post} for a {(n-1)*step:.1f} m route "
                    f"cut {cut} points short")


class TestGoalEndsTheEpisode(unittest.TestCase):

    @staticmethod
    def _monitor() -> LiveMonitor:
        # No logged path: the deviation columns are nan, which no rule reads.
        return LiveMonitor(None, warmup=0, dt=0.1, t_max_s=20.0)

    def test_goal_reached_terminates(self) -> None:
        m = self._monitor()
        self.assertIsNone(m.update(ego_xy=(0.0, 0.0), ego_speed=5.0))
        ended = m.update(ego_xy=(1.0, 0.0), ego_speed=5.0, goal_reached=True)
        self.assertIsNotNone(ended)
        self.assertIs(ended.reason, TerminationReason.GOAL_REACHED)

    def test_without_the_flag_the_episode_keeps_running(self) -> None:
        """The regression: the same frames, goal never announced."""
        m = self._monitor()
        for _ in range(50):
            self.assertIsNone(m.update(ego_xy=(1.0, 0.0), ego_speed=5.0))

    def test_earlier_contact_still_wins(self) -> None:
        """A crash on the way is not erased by finishing afterwards."""
        m = self._monitor()
        m.update(ego_xy=(0.0, 0.0), ego_speed=5.0,
                 contacts=[{"at_fault": True, "kind": "front", "agent_id": "a1"}])
        self.assertIs(m.termination.reason, TerminationReason.CONTACT_AT_FAULT)
        ended = m.update(ego_xy=(1.0, 0.0), ego_speed=5.0, goal_reached=True)
        self.assertIs(ended.reason, TerminationReason.CONTACT_AT_FAULT)

    def test_same_frame_contact_masks_the_goal(self) -> None:
        m = self._monitor()
        ended = m.update(ego_xy=(1.0, 0.0), ego_speed=5.0, goal_reached=True,
                         contacts=[{"at_fault": True, "kind": "front",
                                    "agent_id": "a1"}])
        self.assertIs(ended.reason, TerminationReason.CONTACT_AT_FAULT)

    def test_goal_beats_budget_expiry_at_teardown(self) -> None:
        """An episode stopped by the frame cap after finishing is a completion."""
        m = self._monitor()
        for _ in range(10):
            m.update(ego_xy=(1.0, 0.0), ego_speed=5.0)
        # 10 frames under a 20 s ceiling: the frames ran out, not the budget,
        # which is TRACE_EXHAUSTED and not the policy's.
        self.assertIs(m.final().reason, TerminationReason.TRACE_EXHAUSTED)
        self.assertFalse(m.final().reason.policy_attributed)
        self.assertIs(m.final(goal_reached=True).reason,
                      TerminationReason.GOAL_REACHED)


if __name__ == "__main__":
    unittest.main()
