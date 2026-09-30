# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""No-op eval knobs are refused, not silently accepted.

``eval_mode="open_loop"`` was parsed by every eval surface and validated by
``EvaluationConfig`` — but nothing ever read it: the evaluator ran the closed
loop regardless, so the flag mislabelled runs. The value is now refused with
a pointer at the control that actually exists (``ego_replay_frames >=
eval_frames`` for a pure log-replay run).
"""

from __future__ import annotations

import pytest

from navsafe.evaluation.evaluator import EvaluationConfig


def test_open_loop_is_refused_with_pointer():
    with pytest.raises(NotImplementedError, match="ego_replay_frames"):
        EvaluationConfig(eval_mode="open_loop")


def test_unknown_eval_mode_still_a_value_error():
    with pytest.raises(ValueError, match="Invalid eval_mode"):
        EvaluationConfig(eval_mode="bogus")


def test_closed_loop_stays_the_default_and_valid():
    assert EvaluationConfig().eval_mode == "closed_loop"
    assert EvaluationConfig(eval_mode="closed_loop").eval_mode == "closed_loop"


def test_zero_route_time_limit_means_unlimited():
    assert EvaluationConfig(route_time_limit_s=0).route_time_limit_s is None


def test_idm_traffic_mode_is_refused_with_pointer():
    """The env refuses idm (unwired silent no-op); accepting it here would
    stamp 'idm' into metrics.json while the run scored something else."""
    with pytest.raises(NotImplementedError, match="semi_reactive"):
        EvaluationConfig(traffic_mode="idm")


def test_wired_traffic_modes_stay_valid():
    for mode in ("no_traffic", "log_replay", "semi_reactive", "navsafe"):
        assert EvaluationConfig(traffic_mode=mode).traffic_mode == mode
    with pytest.raises(ValueError, match="Invalid traffic_mode"):
        EvaluationConfig(traffic_mode="bogus")
