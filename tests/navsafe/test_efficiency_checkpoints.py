# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Efficiency checkpoints follow the route, not the ego's odometer.

Bench2Drive: "Speed check is now performed every 5% of the total route length"
and "if the ego vehicle fails to pass the initial 5% checkpoint, this route is
not included in the final driving efficiency metric calculation."

Binning by driven distance instead re-normalises every run onto its own
trajectory: a policy that left the road after 12% of the route still collected
all 20 checks, and no run could ever fail the initial gate.
"""

from __future__ import annotations

import unittest

import numpy as np

from navsafe.benchmark.scoring.metrics import (
    EFFICIENCY_CHECKPOINTS, route_efficiency,
)


def _flat(n: int, ego: float, bg: float, pct_end: float, pct_start: float = 0.0):
    """n frames at constant speeds, covering pct_start..pct_end of the route."""
    return (np.full(n, ego), np.full(n, bg),
            np.linspace(pct_start, pct_end, n))


class TestInitialCheckpointGate(unittest.TestCase):

    def test_route_short_of_the_first_checkpoint_is_excluded(self) -> None:
        """<5% of the route -> the route contributes nothing, not a number."""
        ego, bg, pct = _flat(30, 10.0, 5.0, pct_end=4.9)
        self.assertIsNone(route_efficiency(ego, bg, pct))

    def test_just_past_the_first_checkpoint_is_scored(self) -> None:
        ego, bg, pct = _flat(30, 10.0, 5.0, pct_end=5.5)
        self.assertAlmostEqual(route_efficiency(ego, bg, pct), 200.0, places=6)

    def test_a_stalled_ego_is_excluded_not_scored_zero(self) -> None:
        """The die-immediately case the gate exists for."""
        ego, bg, pct = _flat(40, 0.0, 6.0, pct_end=1.0)
        self.assertIsNone(route_efficiency(ego, bg, pct))


class TestCheckpointsAreRouteRelative(unittest.TestCase):

    def test_only_reached_checkpoints_contribute(self) -> None:
        """12% of the route is 2 checkpoints, not 20.

        Ego is fast over the first 5% and slow after, so a 2-checkpoint mean
        and a 20-checkpoint mean over the same frames differ.
        """
        pct = np.linspace(0.0, 12.0, 240)
        ego = np.where(pct < 5.0, 20.0, 5.0)
        bg = np.full_like(pct, 10.0)
        got = route_efficiency(ego, bg, pct)
        # Checkpoint 0 (0-5%) is all fast -> 200%; checkpoint 1 (5-10%) all
        # slow -> 50%. The 10-12% tail never reached checkpoint 2, so it is
        # not a checkpoint of its own.
        self.assertAlmostEqual(got, (200.0 + 50.0) / 2, places=6)

    def test_driven_distance_binning_would_have_used_all_20(self) -> None:
        """Guards the actual regression: same frames, route- vs self-relative.

        Under the old odometer binning these 12% of route were stretched
        across all 20 checkpoints, so the fast opening stretch was diluted
        across many bins instead of owning exactly one.
        """
        pct = np.linspace(0.0, 12.0, 240)
        ego = np.where(pct < 5.0, 20.0, 5.0)
        bg = np.full_like(pct, 10.0)
        route_relative = route_efficiency(ego, bg, pct)
        # Self-relative: rescale so this run's 12% looks like a whole route.
        self_relative = route_efficiency(ego, bg, pct * (100.0 / 12.0))
        self.assertNotAlmostEqual(route_relative, self_relative, places=3)

    def test_full_route_uses_every_checkpoint(self) -> None:
        pct = np.linspace(0.0, 100.0, 2000)
        ego = np.full_like(pct, 12.0)
        bg = np.full_like(pct, 10.0)
        self.assertAlmostEqual(route_efficiency(ego, bg, pct), 120.0, places=6)


class TestPreservedBench2DriveRules(unittest.TestCase):

    def test_outlier_checkpoints_above_1000_pct_are_dropped(self) -> None:
        pct = np.linspace(0.0, 10.0, 200)
        ego = np.where(pct < 5.0, 10_000.0, 10.0)   # absurd spike in cp 0
        bg = np.full_like(pct, 10.0)
        self.assertAlmostEqual(route_efficiency(ego, bg, pct), 100.0, places=6)

    def test_checkpoint_without_background_traffic_is_skipped(self) -> None:
        pct = np.linspace(0.0, 10.0, 200)
        ego = np.full_like(pct, 10.0)
        bg = np.where(pct < 5.0, np.nan, 5.0)       # no traffic in cp 0
        self.assertAlmostEqual(route_efficiency(ego, bg, pct), 200.0, places=6)

    def test_no_surviving_checkpoint_returns_none(self) -> None:
        pct = np.linspace(0.0, 10.0, 200)
        ego = np.full_like(pct, 10.0)
        bg = np.full_like(pct, np.nan)
        self.assertIsNone(route_efficiency(ego, bg, pct))

    def test_checkpoint_count_is_bench2drives_twenty(self) -> None:
        self.assertEqual(EFFICIENCY_CHECKPOINTS, 20)


if __name__ == "__main__":
    unittest.main()
