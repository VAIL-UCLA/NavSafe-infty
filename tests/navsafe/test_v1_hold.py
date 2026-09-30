# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""V-1 scores a hold, not a traversal.

The trap this pins: on every other leaf the objective is to reach the goal, and
the whole chain encodes that -- ``completed`` is ``reason is GOAL_REACHED``, the
fallthrough when nothing happens is ``BUDGET_EXPIRED``, and that adds
``route_timeout``, which fails success(). Point that chain at a red light and an
ego that behaves perfectly fails: measured on ``00c1e4eb4a045f20``/pdm_closed,
``budget_expired`` / ``{route_timeout: 1}`` / SR false / DS 89.1.

Two of these tests would also have passed before the change, and are here
because the mechanism they rest on was broken: ``goal_reached`` used to be
appended to the event list directly instead of going through ``first()``, so
``disabled_terminations={"goal_reached"}`` was accepted and ignored -- the same
no-op the C-3 lane-change gate shipped with (`ec7602e`).
"""

from __future__ import annotations

import pytest

from navsafe.benchmark import scenario_rules as sr
from navsafe.benchmark import termination as term


def _frames(n: int, *, speed: float = 0.0, warmup: int = 0, **cols):
    """A trace of ``n`` frames, all scored past ``warmup``.

    ``cols`` maps a column name to either a scalar (every frame) or a list.
    """
    out = []
    for i in range(n):
        f = {"phase": "warmup" if i < warmup else "scored",
             "ego_speed": speed, "ego_x": 0.0, "ego_y": 0.0,
             "on_drivable": True, "driving_direction_ok": True,
             "contacts": []}
        for key, val in cols.items():
            f[key] = val[i] if isinstance(val, list) else val
        out.append(f)
    return out


#: The hold only applies where the red was authored, so every V-1 hold fixture
#: carries the manifest key that says so.
_OVERRIDE = {"lane": "53726", "state": "LANE_STATE_STOP", "frames": "all"}


def _v1_rules():
    return sr.rules_for_scenario({"taxonomy_leaves": ["V-1"],
                                  "signal_override": _OVERRIDE})


def _stop_sign_rules():
    return sr.rules_for_scenario({"taxonomy_leaves": ["V-1"],
                                  "scenario_types": ["on_stopline_stop_sign"]})


# --- routing -------------------------------------------------------------

def test_v1_leaf_gets_the_hold_even_with_no_scenario_types():
    """`00c1e4eb4a045f20` publishes an empty scenario_types list."""
    rules = sr.rules_for_scenario({"taxonomy_leaves": ["V-1"],
                                   "scenario_types": [],
                                   "signal_override": _OVERRIDE})
    assert rules.hold_region == "stop_line"
    assert rules.stop_sign_dwell_s is None


def test_a_stop_sign_type_replaces_the_hold_rather_than_adding_to_it():
    rules = _stop_sign_rules()
    assert rules.hold_region is None, "a stop sign is stop-then-go, not a hold"
    assert rules.stop_sign_dwell_s == sr.STOP_SIGN_DWELL_S
    assert "goal_reached" not in rules.disabled_terminations


def test_other_leaves_are_untouched():
    assert sr.rules_for_scenario({"taxonomy_leaves": ["V-5"]}) is sr.DEFAULT_RULES
    assert sr.rules_for_leaf("I-2").hold_region is None


# --- termination ---------------------------------------------------------

def test_a_held_episode_completes_and_charges_nothing():
    t = term.classify(_frames(200, dist_to_stopline_m=8.0), goal_reached=False,
                      t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.HOLD_SATISFIED
    route = term.to_route_result("r", 100.0, t, {}, _v1_rules())
    assert route.completed and route.success()
    assert route.infractions == {}, "no route_timeout: the clock IS the success"


def test_crossing_on_red_ends_the_episode_and_is_the_reason():
    frames = _frames(100, signal_state=["unknown"] * 40 + ["red"] * 60)
    t = term.classify(frames, goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.RED_LIGHT_RUN
    assert t.frame == 40


def test_the_crossing_reason_adds_no_second_count():
    """`red_light` prices it from the TL column; the reason must not re-charge."""
    frames = _frames(100, signal_state=["unknown"] * 40 + ["red"] * 60)
    t = term.classify(frames, goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    route = term.to_route_result("r", 40.0, t, {"red_light": 1.0}, _v1_rules())
    assert route.infractions == {"red_light": 1.0}
    assert not route.completed and not route.success()


def test_reaching_the_goal_does_not_end_a_hold_episode():
    t = term.classify(_frames(200, dist_to_stopline_m=8.0), goal_reached=True,
                      goal_frame=50, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.HOLD_SATISFIED


def test_the_goal_suppression_is_not_a_no_op_on_its_own():
    """Without the hold, `disabled_terminations` alone must still take."""
    rules = sr.ScenarioRules(leaf="X", disabled_terminations=frozenset({"goal_reached"}))
    t = term.classify(_frames(200, speed=5.0), goal_reached=True, goal_frame=50,
                      t_max_s=20.0, rules=rules)
    assert t.reason is not term.TerminationReason.GOAL_REACHED


def test_standing_still_at_a_red_is_not_a_deadlock():
    """200 stationary frames is 20 s against a 5 s rule."""
    t = term.classify(_frames(200, dist_to_stopline_m=8.0), goal_reached=False,
                      t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.HOLD_SATISFIED


def test_a_contact_during_the_hold_still_ends_it():
    frames = _frames(200, dist_to_stopline_m=8.0)
    frames[60]["contacts"] = [{"at_fault": True, "kind": "rear_end",
                               "agent_id": "a1"}]
    t = term.classify(frames, goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.CONTACT_AT_FAULT
    assert t.frame == 60


def test_a_crossing_by_geometry_when_the_artifact_has_no_tl_column():
    dists = [10.0 - 0.2 * i for i in range(100)]     # crosses zero at frame 50
    t = term.classify(_frames(100, speed=2.0, dist_to_stopline_m=dists),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.RED_LIGHT_RUN
    assert t.frame == 50


def test_a_trace_that_starts_past_the_line_is_neither_crossing_nor_hold():
    """Never in front of it means never approached: not checked, not failed."""
    t = term.classify(_frames(100, speed=2.0, dist_to_stopline_m=-5.0),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason not in (term.TerminationReason.RED_LIGHT_RUN,
                            term.TerminationReason.HOLD_SATISFIED)


def test_an_unwitnessed_hold_is_not_a_completion():
    """The regression that made this rule dangerous.

    On `00c1e4eb4a045f20` neither witness existed -- the logged light is never
    red and the connector entry projects to the window's first frame, so
    `dist_to_stopline_m` is NaN throughout. An ego that drove 130.6 m at
    5-7.6 m/s scored hold_satisfied / SR true / DS 100.0.
    """
    t = term.classify(_frames(200, speed=6.0), goal_reached=False,
                      t_max_s=20.0, rules=_v1_rules())
    assert t.reason is not term.TerminationReason.HOLD_SATISFIED
    route = term.to_route_result("r", 12.0, t, {}, _v1_rules())
    assert not route.completed and not route.success()


def test_a_bundle_with_no_authored_red_keeps_the_default_rules():
    """The nine published V-1 bundles log GO/UNKNOWN and never STOP."""
    rules = sr.rules_for_scenario({"taxonomy_leaves": ["V-1"]})
    assert rules is sr.DEFAULT_RULES


# --- hold fraction -------------------------------------------------------

def test_hold_fraction_is_100_when_nothing_was_violated():
    assert sr.hold_fraction(_frames(50), None) == 100.0


def test_hold_fraction_is_the_window_survived():
    assert sr.hold_fraction(_frames(100), 60) == pytest.approx(60.0)


def test_hold_fraction_counts_scored_frames_only():
    """A warm-up the policy did not drive cannot be credited to it."""
    frames = _frames(100, warmup=20)
    assert sr.hold_fraction(frames, 60) == pytest.approx(50.0)


# --- stop sign -----------------------------------------------------------

def test_a_three_second_stop_before_the_line_is_compliant():
    speeds = [3.0] * 20 + [0.0] * 30 + [3.0] * 50
    dists = [20.0 - 0.5 * i if i < 20 else 10.0 for i in range(50)] + \
            [10.0 - 0.5 * (i - 50) for i in range(50, 100)]
    violated, detail = sr.stop_sign_violation(
        _frames(100, ego_speed=speeds, dist_to_stopline_m=dists),
        _stop_sign_rules())
    assert violated is False, detail


def test_rolling_through_the_line_is_a_stop_infraction():
    speeds = [3.0] * 100
    dists = [10.0 - 0.5 * i for i in range(100)]
    violated, detail = sr.stop_sign_violation(
        _frames(100, ego_speed=speeds, dist_to_stopline_m=dists),
        _stop_sign_rules())
    assert violated is True, detail


def test_a_stop_shorter_than_the_dwell_is_a_stop_infraction():
    speeds = [3.0] * 10 + [0.0] * 20 + [3.0] * 70      # 2.0 s, needs 3.0
    dists = [10.0 - 0.2 * i for i in range(100)]       # crosses at frame 50
    violated, _ = sr.stop_sign_violation(
        _frames(100, ego_speed=speeds, dist_to_stopline_m=dists),
        _stop_sign_rules())
    assert violated is True


def test_no_approach_in_the_window_is_skipped_not_scored():
    """`9311d9a2409c5224` is tagged accelerating_at_stop_sign."""
    violated, why = sr.stop_sign_violation(
        _frames(100, ego_speed=5.0, dist_to_stopline_m=-3.0),
        _stop_sign_rules())
    assert violated is None
    assert "no approach" in why


def test_a_missing_stopline_column_is_skipped_not_scored():
    violated, _ = sr.stop_sign_violation(_frames(100, ego_speed=5.0),
                                         _stop_sign_rules())
    assert violated is None


def test_waiting_longer_than_the_deadlock_rule_at_a_stop_sign_is_not_frozen():
    """A cautious 8 s stop at the line must not terminate as deadlock."""
    frames = _frames(100, ego_speed=0.0, dist_to_stopline_m=2.0)
    t = term.classify(frames, goal_reached=False, t_max_s=10.0,
                      rules=_stop_sign_rules())
    assert t.reason is not term.TerminationReason.DEADLOCK


def test_freezing_far_from_the_line_is_still_a_deadlock():
    frames = _frames(100, ego_speed=0.0, dist_to_stopline_m=40.0)
    t = term.classify(frames, goal_reached=False, t_max_s=10.0,
                      rules=_stop_sign_rules())
    assert t.reason is term.TerminationReason.DEADLOCK


def test_the_live_monitor_does_not_end_a_hold_on_its_first_frame():
    """Mid-episode, "held the whole window" only means "has not crossed yet".

    `LiveMonitor.update` ends the episode on the frame an ending first exists,
    so a hold reported per frame would stop a V-1 episode at frame 1 with a
    perfect score. HOLD_SATISFIED is therefore a no-event fallback: real only
    once stepping has stopped, which is what `final()` evaluates.
    """
    import numpy as np

    mon = term.LiveMonitor(np.zeros((200, 2)), warmup=0, dt=0.1, t_max_s=20.0,
                           rules=_v1_rules())
    for i in range(10):
        out = mon.update(ego_xy=(0.0, 0.0), ego_speed=0.0,
                         dist_to_stopline_m=8.0)
        assert out is None, f"episode ended at frame {i}"


def test_the_hold_fraction_denominator_is_the_budget_not_the_trace():
    """Early termination must not inflate the score it caused.

    A crossing ends the episode, so the recorded window shrinks with the very
    failure being measured. Held against the trace, one simwam run that crossed
    at frame 28 scored 88.9 (8 of the 9 frames recorded) while the same
    behaviour, before the live monitor could end it, scored 1.3 of 599.
    """
    short = _frames(30)          # episode ended at the crossing
    long = _frames(600)          # same crossing, no live terminator
    assert sr.hold_fraction(short, 28, window_frames=180) == pytest.approx(
        sr.hold_fraction(long, 28, window_frames=180))
    assert sr.hold_fraction(short, 28, window_frames=180) == pytest.approx(15.56, abs=0.01)


def test_the_hold_fraction_cannot_exceed_100():
    assert sr.hold_fraction(_frames(300), 250, window_frames=100) == 100.0


def test_the_signal_verdict_outranks_the_stop_line_geometry():
    """They do not agree to the frame, and the scorer is the authority.

    The scorer waits until the ego centre is inside the red connector's
    polygon, ~0.8 m past the arc where `dist_to_stopline_m` crosses zero: on
    05d0a1a763fc5334/simwam that was frame 30 against geometry's 28. Whichever
    fires first ends the episode, so geometry winning truncated the trace two
    frames before the TL drop and `red_light` went uncharged.
    """
    dists = [10.0 - 0.2 * i for i in range(100)]           # crosses at 50
    states = ["unknown"] * 52 + ["red"] * 48               # scorer convicts at 52
    t = term.classify(_frames(100, speed=2.0, dist_to_stopline_m=dists,
                              signal_state=states),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.RED_LIGHT_RUN
    assert t.frame == 52, "geometry must not pre-empt the signal"


def test_geometry_still_convicts_when_no_tl_column_exists():
    dists = [10.0 - 0.2 * i for i in range(100)]
    t = term.classify(_frames(100, speed=2.0, dist_to_stopline_m=dists),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.RED_LIGHT_RUN
    assert t.frame == 50


def test_a_compliant_signal_verdict_is_not_overruled():
    """`unknown` means the scorer looked and saw no violation."""
    dists = [10.0 - 0.2 * i for i in range(100)]
    t = term.classify(_frames(100, speed=2.0, dist_to_stopline_m=dists,
                              signal_state="unknown"),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is not term.TerminationReason.RED_LIGHT_RUN


def test_a_hold_the_trace_refutes_is_not_a_completion():
    """No violation reported, ego 10 m past the line: an authoring gap.

    `_green_way_through` exempts a crossing while a non-red signalized
    connector still offers a way through, which is what happens when an
    override reddens only part of an approach. Crediting a hold there would
    give SR true to a policy that drove through the junction.
    """
    dists = [10.0 - 0.2 * i for i in range(100)]   # ends 10 m beyond
    t = term.classify(_frames(100, speed=2.0, dist_to_stopline_m=dists,
                              signal_state="unknown"),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is not term.TerminationReason.HOLD_SATISFIED
    route = term.to_route_result("r", 10.0, t, {}, _v1_rules())
    assert not route.completed and not route.success()


def test_a_genuine_hold_is_still_a_completion_with_a_compliant_signal():
    t = term.classify(_frames(200, dist_to_stopline_m=6.0, signal_state="unknown"),
                      goal_reached=False, t_max_s=20.0, rules=_v1_rules())
    assert t.reason is term.TerminationReason.HOLD_SATISFIED
