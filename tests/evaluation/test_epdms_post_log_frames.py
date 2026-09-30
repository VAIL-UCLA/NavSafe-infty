# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Scoring must survive a frame past the end of the log.

A NavSafe policy episode is indefinite: it ends on a taxonomy event or the
60 s safety ceiling, not when the 20 s bundle runs out. Past that point the
ego is simulated rather than replayed, and the SDC track has no recorded
validity for the frame.

Both entry points here gated on ``sdc_track['state']['valid'][frame_idx]``
with the raw frame, so the first episode to outlive its log raised::

    IndexError: index 205 is out of bounds for axis 0 with size 201"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely")

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (  # noqa: E402
    EPDMSTrajectoryScorer_Fast,
)

LOG_FRAMES = 201


def _scenario(valid_all: bool = True) -> dict:
    """One straight lane, ego driving down it, ``LOG_FRAMES`` long."""
    t = np.arange(LOG_FRAMES) * 0.1
    position = np.zeros((LOG_FRAMES, 3))
    position[:, 0] = 2.0 + 2.0 * t
    valid = np.ones(LOG_FRAMES, dtype=bool)
    if not valid_all:
        valid[:] = False
    return {
        "metadata": {"sdc_id": "ego", "scenario_id": "post_log_probe"},
        "length": LOG_FRAMES,
        "dynamic_map_states": {},
        "map_features": {
            "lane_0": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array([[0.0, 0.0], [120.0, 0.0]]),
                "polygon": np.array([[0.0, 3.5], [120.0, 3.5],
                                     [120.0, -3.5], [0.0, -3.5]]),
            },
        },
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": position,
                    "heading": np.zeros(LOG_FRAMES),
                    "valid": valid,
                },
            },
        },
    }


def _ego_state(x: float = 60.0) -> dict:
    return {
        "position": np.array([x, 0.0, 0.0]),
        "heading": 0.0,
        "velocity": np.array([5.0, 0.0, 0.0]),
        "acceleration": np.zeros(3),
        "angular_velocity": np.zeros(3),
    }


def _candidates(n: int = 3) -> np.ndarray:
    """``(n, 8, 2)`` forward candidates in ``[forward, lateral]``."""
    out = np.zeros((n, 8, 2))
    steps = np.arange(1, 9) * 0.5
    for i in range(n):
        out[i, :, 0] = 5.0 * (1.0 - 0.1 * i) * steps
    return out


def _scorer(scenario: dict) -> EPDMSTrajectoryScorer_Fast:
    scorer = EPDMSTrajectoryScorer_Fast(verbose=False)
    scorer.initialize(scenario, None)
    return scorer


class TestPastTheLog:
    def test_scores_a_frame_beyond_the_log(self):
        """The frame that used to raise now scores like any other."""
        scorer = _scorer(_scenario())
        scores, metrics = scorer.score_candidates(
            _candidates(), _ego_state(), LOG_FRAMES + 4, return_metrics=True)
        assert scores.shape == (3,)
        assert np.isfinite(scores).all()
        # Not the all-zero early return: a post-log frame is scorable, and a
        # zero grid here would read as uniformly terrible driving.
        assert len(metrics) == 3

    def test_in_log_invalid_frame_still_returns_zeros(self):
        """The clamp must not swallow the case the gate exists for."""
        scorer = _scorer(_scenario(valid_all=False))
        scores, metrics = scorer.score_candidates(
            _candidates(), _ego_state(), 10, return_metrics=True)
        assert scores.shape == (3,)
        assert not scores.any()
        assert metrics == []

    def test_the_boundary_frame_is_past_the_log(self):
        """``frame_idx == len(valid)`` is the first post-log frame, not the last
        logged one — an off-by-one here re-raises the original IndexError."""
        scorer = _scorer(_scenario())
        scores, _ = scorer.score_candidates(
            _candidates(), _ego_state(), LOG_FRAMES, return_metrics=False)
        assert np.isfinite(scores).all()
