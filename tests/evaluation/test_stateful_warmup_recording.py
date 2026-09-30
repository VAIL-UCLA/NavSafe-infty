# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Replay warm-up recording must not require a stateful policy plan."""

from __future__ import annotations

import numpy as np

from navsafe.evaluation.evaluator import Evaluator


def test_replay_pose_supplies_history_without_a_planner_trajectory():
    evaluator = object.__new__(Evaluator)
    evaluator._current_trajectory = None

    evaluator._record_replay_pose({
        "position": np.array([12.0, -3.0, 0.4]),
        "velocity": np.array([4.5, 0.25, 0.0]),
    })

    np.testing.assert_allclose(
        evaluator._current_trajectory,
        [[12.0, -3.0, 0.4, 4.5, 0.25]],
    )


def test_replay_pose_is_refreshed_instead_of_carrying_stale_history():
    evaluator = object.__new__(Evaluator)
    evaluator._current_trajectory = np.full((2, 5), -99.0)

    evaluator._record_replay_pose({
        "position": np.array([2.0, 7.0]),
        "velocity": np.array([1.0, -0.5]),
    })

    np.testing.assert_allclose(
        evaluator._current_trajectory,
        [[2.0, 7.0, 0.0, 1.0, -0.5]],
    )
