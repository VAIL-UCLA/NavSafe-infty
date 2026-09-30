# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The shared ``gt_stride`` derivation and every path that must use it.

``gt_stride`` was reimplemented per scorer and once recomputed inline by
the planner, which silently bypassed the >=1 clamp on the planner's path
while the evaluator's looked fine (the frozen-world bug). These tests pin
the one audited derivation and that the fast scorer routes through it.
"""

from __future__ import annotations

import logging

import pytest

from navsafe.evaluation.scorers.gt_stride import gt_stride


def test_nominal_stride() -> None:
    assert gt_stride(0.5, 0.1) == 5


def test_planner_faster_than_scenario_clamps_to_one(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert gt_stride(0.05, 0.1) == 1
    assert any("Clamping to 1" in r.message for r in caplog.records), (
        "the clamp must WARN — a silent clamp is how the frozen-world "
        "bug stayed invisible"
    )


@pytest.mark.parametrize("bad_dt", [0.0, -0.1])
def test_degenerate_scenario_dt_falls_back(bad_dt) -> None:
    assert gt_stride(0.5, bad_dt) == 5


def test_misparsed_absolute_timestamp_would_clamp_not_freeze() -> None:
    # The historical failure: scenario_dt read as an absolute us stamp.
    # Even if the misparse ever regressed, the stride must never be 0.
    assert gt_stride(0.5, 3.16e14) >= 1


def test_rounding() -> None:
    assert gt_stride(0.5, 0.2) == 2   # 2.5 rounds to even
    assert gt_stride(0.5, 0.3) == 2   # 1.67 -> 2


def test_fast_scorer_routes_through_shared_helper() -> None:
    """The fast scorer's property must be the shared derivation, not a copy."""
    torch = pytest.importorskip("torch")  # noqa: F841 — heavy import gate
    from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
        EPDMSTrajectoryScorer_Fast,
    )

    scorer = EPDMSTrajectoryScorer_Fast.__new__(EPDMSTrajectoryScorer_Fast)
    for planner_dt, scenario_dt in [
        (0.5, 0.1), (0.05, 0.1), (0.5, 0.0), (0.5, 3.16e14), (0.5, 0.2),
    ]:
        scorer.planner_dt = planner_dt
        scorer.scenario_dt = scenario_dt
        assert scorer.gt_stride == gt_stride(
            planner_dt, scenario_dt
        ), f"drift at planner_dt={planner_dt}, scenario_dt={scenario_dt}"

    with pytest.raises(AttributeError):
        scorer.gt_stride = 3
