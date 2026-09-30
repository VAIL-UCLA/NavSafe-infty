# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock regression guard for ``EPDMSTrajectoryScorer_Fast.score_candidates``.

The batch EPDMS engine was measurably sped up (2.16x, 2026-08-18: shared
per-frame precompute + AABB pre-filtering); this smoke keeps that win from
silently eroding — an accidental O(N x agents x horizon) shapely loop would
blow the budget immediately.

Anti-flake design (deliberate, do not "tighten" casually):

* **Skipped when the ``CI`` env var is set.** Shared CI runners have unknown
  and highly variable speed; an absolute wall-clock bound there WILL flake.
  The regression this guards is a dev/GPU-box concern (the scorer's hot use
  is planner-side proposal scoring on sim machines), so it runs everywhere
  except CI. A two-point relative comparison (e.g. batch-vs-loop) was
  rejected: it measures batching efficiency, not the absolute cost the
  speedup reduced, and same-process ratios still flake under CPU contention.
* **min-of-3 after a warmup.** The minimum over repeats is the standard
  noise-robust wall-clock estimator (contention only ever ADDS time).
* **Generous absolute budget.** Measured on the dev GPU box on 2026-08-18:
  82-87 ms per 24-candidate call (min 82.1 ms over 5 repeats). The budget is
  1.0 s — ~12x the measured minimum — so only a genuine algorithmic
  regression (not machine noise, not a slower box) can cross it.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSTrajectoryScorer_Fast,
)

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="wall-clock smoke is meaningless on shared CI runners of unknown "
    "speed; it guards dev/GPU boxes (see module docstring)",
)

DT = 0.5  # scorer planner_dt default
T = 8  # 4 s horizon
N_CANDIDATES = 24
BUDGET_S = 1.0  # ~12x the 82 ms measured on the dev box (2026-08-18)


def _track(x: float, y: float, *, track_type: str = "VEHICLE",
           length: float = 4.5, width: float = 1.8,
           heading: float = 0.0, n: int = 120) -> dict:
    return {
        "type": track_type,
        "state": {
            "position": np.tile(np.array([x, y, 0.0]), (n, 1)),
            "velocity": np.zeros((n, 3)),
            "heading": np.full(n, heading),
            "valid": np.ones(n, dtype=bool),
            "length": np.full(n, length),
            "width": np.full(n, width),
        },
    }


def _scenario() -> dict:
    """Synthetic straight-lane scene, built like
    tests/evaluation/test_epdms_metric_regressions.py, plus a handful of
    agents so every metric loop does real polygon work."""
    n = 120
    xs = np.linspace(-100.0, 300.0, 401)
    tracks = {
        "sdc": {
            "type": "VEHICLE",
            "state": {
                "position": np.zeros((n, 3)),
                "velocity": np.zeros((n, 3)),
                "heading": np.zeros(n),
                "valid": np.ones(n, dtype=bool),
                "length": np.full(n, 4.515),
                "width": np.full(n, 1.852),
            },
        },
    }
    for i in range(6):
        tracks[f"veh{i}"] = _track(25.0 + 12.0 * i, 3.5 * (i % 2))
    tracks["ped"] = _track(40.0, -3.0, track_type="PEDESTRIAN",
                           length=0.6, width=0.6)
    return {
        "metadata": {"sdc_id": "sdc", "ts": np.arange(n) * 0.1},
        "length": n,
        "tracks": tracks,
        "map_features": {
            "lane_0": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.column_stack(
                    [xs, np.zeros_like(xs), np.zeros_like(xs)]),
                "speed_limit_mps": 15.0,
            }
        },
        "dynamic_map_states": {},
    }


def _candidates() -> np.ndarray:
    """(24, T, 3) deterministic spread of speeds with mild lateral wander."""
    rng = np.random.default_rng(0)
    cands = np.zeros((N_CANDIDATES, T, 3))
    for i, speed in enumerate(np.linspace(2.0, 12.0, N_CANDIDATES)):
        cands[i, :, 0] = speed * DT * np.arange(1, T + 1)
        cands[i, :, 1] = 0.2 * rng.standard_normal(T).cumsum()
    return cands


def test_score_candidates_24_batch_stays_inside_wall_clock_budget():
    scorer = EPDMSTrajectoryScorer_Fast(verbose=False)
    scorer.initialize(_scenario(), env=None)
    cands = _candidates()
    ego = {
        "position": np.array([0.0, 0.0, 0.0]),
        "heading": 0.0,
        "velocity": np.array([5.0, 0.0, 0.0]),
        "acceleration": np.zeros(3),
        "angular_velocity": np.zeros(3),
    }

    # Warmup (first call pays one-time per-frame precompute/caches).
    scorer.score_candidates(cands, ego, frame_idx=0, return_metrics=True)

    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        scores, metrics = scorer.score_candidates(
            cands, ego, frame_idx=0, return_metrics=True)
        best = min(best, time.perf_counter() - t0)

    # Sanity: the call actually scored the batch (a no-op would be "fast").
    assert scores.shape == (N_CANDIDATES,)
    assert len(metrics) == N_CANDIDATES
    assert np.count_nonzero(scores) > 0, "all-zero scores: scorer no-op'd"

    assert best < BUDGET_S, (
        f"score_candidates took {best * 1000:.0f} ms for {N_CANDIDATES} "
        f"candidates (budget {BUDGET_S * 1000:.0f} ms ~= 12x the 82 ms "
        "measured 2026-08-18). This is a real perf regression in the batch "
        "EPDMS engine, not noise — the budget is deliberately generous."
    )
