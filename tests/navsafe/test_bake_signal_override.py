# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Which V-1 bundles can be asked to hold behind a stop line.

The trap this pins: the distances that decide it belong to the HAND-OFF, not to
the window start. `run_bundle_eval.sh policy` replays 20 logged frames before
the policy drives, which at 6.4 m/s is 12.8 m of road — past several of these
stop lines. Measured from frame 0 the bake called six of the nine traffic-light
bundles usable; measured from the hand-off it is three, and five of the six
would have authored a red into a scenario where the ego is already through the
junction before the policy gets the wheel.

The second test is stoppability. A line closer than the ego's braking distance
cannot be honoured by any policy, and a Table-2 row where every policy scores
zero is measuring the scenario.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "navsafe_bake_signal_override",
    REPO / "scripts/tools/navsafe_bake_signal_override.py")
bake = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(bake)


def test_a_stopping_distance_of_road_in_front_is_usable():
    ok, why = bake.feasibility(stop_s=20.0, handoff_arc=10.0, v=5.0)
    assert ok, why


def test_a_stationary_ego_well_short_of_the_line_is_usable():
    """9e34c53444bd518c: 35.3 m to the line, 0.0 m/s at hand-off."""
    ok, _ = bake.feasibility(stop_s=35.3, handoff_arc=0.0, v=0.0)
    assert ok


def test_an_ego_already_past_the_line_at_handoff_is_refused():
    """0f622aef14545f59: line at 5.3 m, hand-off at 10.3 m."""
    ok, why = bake.feasibility(stop_s=5.3, handoff_arc=10.3, v=6.4)
    assert not ok
    assert "already 5.0 m past" in why


def test_a_line_measured_from_frame_zero_can_still_be_behind_the_policy():
    """The bug this rule fixes, stated as a test.

    From the window start there is 5.3 m of approach, which reads as usable;
    from the hand-off the ego is 5.0 m beyond the line.
    """
    assert bake.feasibility(stop_s=5.3, handoff_arc=0.0, v=6.4)[0] is True
    assert bake.feasibility(stop_s=5.3, handoff_arc=10.3, v=6.4)[0] is False


def test_too_little_approach_left_is_refused():
    """9157902936a456bf: 0.9 m in front of the line at hand-off."""
    ok, why = bake.feasibility(stop_s=2.8, handoff_arc=1.9, v=0.5)
    assert not ok
    assert "0.9 m in front" in why


def test_a_line_inside_the_braking_distance_is_refused():
    """6.4 m/s needs 4.2 m at nuPlan's comfort bound; 3.0 m is unstoppable."""
    ok, why = bake.feasibility(stop_s=13.0, handoff_arc=10.0, v=6.4)
    assert not ok
    assert "unstoppable" in why


def test_the_braking_bound_is_nuplans_comfort_acceleration():
    from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import MAX_ACCEL

    assert bake.MAX_DECEL_MS2 == MAX_ACCEL, (
        "the braking authority asked of a policy must stay the same number the "
        "comfort subscore grades it against, or the bake and the score disagree")


def test_the_warmup_matches_the_eval_driver():
    driver = (REPO / "navsafe/benchmark/eval/run_bundle_eval.sh").read_text()
    assert f"--ego-replay-frames {bake.WARMUP_FRAMES}" in driver, (
        "the hand-off this bake measures from must be the one the policy eval "
        "actually uses")
