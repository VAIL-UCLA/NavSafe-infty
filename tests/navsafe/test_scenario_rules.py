# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Per-leaf scoring rules, read from the bundle's manifest.

Two leaves depart from the defaults because the default contradicts what the
scenario tests: I-2 (work-zone bypass needs the opposing lane) and C-3
(sideswipe is the lane change; going straight declines it). Everything else
must be bit-identical to before this layer existed, and that is what most of
this file pins.
"""

from __future__ import annotations

import pytest

from navsafe.benchmark import scenario_rules as SR
from navsafe.benchmark import termination as T


#: A manifest for the C-3 precursor where the ego is the one changing lane.
_C3_META = {"taxonomy_leaves": ["C-3"],
            "scenario_types": ["changing_lane_with_trail"]}


def _frame(phase="scored", speed=5.0, lane="lane_a", offset=0.1, **kw):
    f = {"phase": phase, "ego_speed": speed, "on_drivable": True,
         "driving_direction_ok": True, "signal_hold": False, "contacts": [],
         "lane_id": lane, "lateral_offset_m": offset}
    f.update(kw)
    return f


# --- resolution ------------------------------------------------------------

def test_an_unlisted_leaf_gets_the_defaults():
    for leaf in (None, "", "V-9", "C-1", "R-2", "I-3", "nonsense"):
        assert SR.rules_for_leaf(leaf).is_default, leaf


def test_i2_is_a_leaf_rule_and_c3_is_not():
    """I-2 is decided by the leaf; C-3 is decided by the scenario TYPE.

    C-3 covers both "the ego changes lane into an occupied lane" and "a fast
    car passes alongside". Only the first requires the ego to change lane, and
    22 of the corpus's 23 C-3 rows are the second, so a leaf-keyed rule would
    have applied the wrong goal to almost all of them.
    """
    assert not SR.rules_for_leaf("I-2").is_default
    assert SR.rules_for_leaf("C-3").is_default
    # Case and whitespace come from a manifest, not from code.
    assert SR.rules_for_leaf(" i-2 ") is SR.rules_for_leaf("I-2")


@pytest.mark.parametrize("scenario_type,expect_gate", [
    ("changing_lane_with_lead", True),
    ("changing_lane_with_trail", True),
    # The ego is overtaken here; holding the lane is correct driving.
    ("near_high_speed_vehicle", False),
])
def test_c3_lane_change_gate_is_keyed_on_the_scenario_type(scenario_type,
                                                           expect_gate):
    meta = {"taxonomy_leaves": ["C-3"], "scenario_types": [scenario_type]}
    rules = SR.rules_for_scenario(meta)
    assert rules.require_lane_change is expect_gate
    if expect_gate:
        assert rules.leaf == "C-3"
    else:
        assert rules.is_default


def test_rules_come_from_the_manifests_taxonomy_leaves():
    meta = {"token": "x", "taxonomy_leaves": ["I-2"],
            "scenario_types": ["near_construction_zone_sign"]}
    assert SR.rules_for_scenario(meta).leaf == "I-2"
    assert SR.rules_for_scenario({"taxonomy_leaves": ["V-9"]}).is_default
    assert SR.rules_for_scenario({}).is_default
    assert SR.rules_for_scenario(None).is_default


def test_a_default_leaf_alongside_an_override_still_resolves():
    """nuPlan tags a window with several types, so several leaves is normal."""
    meta = {"taxonomy_leaves": ["V-9", "C-3", "I-3"],
            "scenario_types": ["changing_lane_with_lead", "stationary"]}
    assert SR.rules_for_scenario(meta).leaf == "C-3"


def test_two_conflicting_overrides_raise_instead_of_picking_one():
    """Scoring a bundle under both rule sets is not expressible; say so.

    Silently taking the first would make the number depend on manifest key
    order, which is exactly the kind of invisible dependency this benchmark
    keeps getting bitten by.
    """
    with pytest.raises(ValueError, match="conflicting scoring rules"):
        SR.rules_for_scenario({
            "taxonomy_leaves": ["C-3", "I-2"],
            "scenario_types": ["changing_lane_with_lead",
                               "near_construction_zone_sign"]})


# --- I-2: wrong-way is the manoeuvre --------------------------------------

def test_i2_does_not_end_the_episode_on_wrong_way():
    """The work-zone bypass has to be allowed to keep driving.

    Suppressing only the infraction would not help: the terminator fires first
    and the episode is over before any infraction filter runs.
    """
    frames = [_frame(driving_direction_ok=(i >= 5)) for i in range(30)]
    # The default: wrong-way at frame 0 ends it.
    default = T.classify(frames, goal_reached=False, t_max_s=3.0, dt=0.1)
    assert default.reason is T.TerminationReason.WRONG_WAY

    i2 = T.classify(frames, goal_reached=False, t_max_s=3.0, dt=0.1,
                    rules=SR.rules_for_leaf("I-2"))
    assert i2.reason is not T.TerminationReason.WRONG_WAY
    assert i2.reason is T.TerminationReason.BUDGET_EXPIRED


def test_i2_records_no_wrong_way_infraction():
    rules = SR.rules_for_leaf("I-2")
    t = T.Termination(T.TerminationReason.BUDGET_EXPIRED, 20)
    r = T.to_route_result("r", 90.0, t, {"wrong_way": 1}, rules)
    assert "wrong_way" not in r.infractions
    # And the filter is not a blanket amnesty: a contact still counts.
    r2 = T.to_route_result("r", 90.0, t,
                           {"wrong_way": 1, "collisions_vehicle": 1}, rules)
    assert r2.infractions.get("collisions_vehicle") == 1


def test_i2_still_terminates_on_everything_else():
    """Only wrong-way is waived; the leaf is not exempt from driving badly."""
    rules = SR.rules_for_leaf("I-2")
    frames = [_frame(on_drivable=False) for _ in range(30)]
    t = T.classify(frames, goal_reached=False, t_max_s=3.0, dt=0.1, rules=rules)
    assert t.reason is T.TerminationReason.OFF_DRIVABLE

    hit = [_frame() for _ in range(10)]
    hit[3]["contacts"] = [{"at_fault": True, "kind": "sideswipe", "agent_id": "a"}]
    t2 = T.classify(hit, goal_reached=False, t_max_s=3.0, dt=0.1, rules=rules)
    assert t2.reason is T.TerminationReason.CONTACT_AT_FAULT


# --- C-3: the goal requires the lane change -------------------------------

def _hold(lane, offset=0.1, n=SR.LANE_CHANGE_HOLD_FRAMES):
    """Enough frames in `lane` to satisfy the persistence requirement."""
    return [_frame(lane=lane, offset=offset) for _ in range(n)]


def test_lane_change_needs_an_offset_sign_flip():
    """A lateral crossing flips the signed offset; a successor handoff does not.

    Both change lane_id, which is why the id alone cannot be the signal.
    """
    crossing = [_frame(lane="a", offset=1.4)] + _hold("b", -1.3)
    assert SR.changed_lane(crossing)

    successor = [_frame(lane="a", offset=0.1)] + _hold("a_next", 0.1)
    assert not SR.changed_lane(successor)

    # A sign flip WITHOUT enough magnitude is centreline jitter, not a crossing.
    jitter = [_frame(lane="a", offset=0.05)] + _hold("b", -0.05)
    assert not SR.changed_lane(jitter)


def test_nearest_lane_flapping_at_a_split_is_not_a_lane_change():
    """The writer reports the NEAREST lane, so two parallel lanes can alternate.

    Each flap carries a real offset sign flip, so without the persistence
    requirement every one of them would be credited as a lane change -- a false
    positive, which for C-3 means honouring a goal the ego did not earn.
    """
    flap = []
    for k in range(12):
        flap.append(_frame(lane="a" if k % 2 == 0 else "b",
                           offset=1.4 if k % 2 == 0 else -1.3))
    assert not SR.changed_lane(flap)


def test_a_change_too_close_to_the_end_of_the_trace_is_not_confirmed():
    """Unconfirmable is not confirmed, and it fails safe (no completion)."""
    short = [_frame(lane="a", offset=1.4)] + _hold(
        "b", -1.3, n=SR.LANE_CHANGE_HOLD_FRAMES - 1)
    assert not SR.changed_lane(short)


def test_a_warmup_lane_change_is_not_credited_to_the_policy():
    frames = ([_frame(phase="warmup", lane="a", offset=1.4)]
              + [_frame(phase="warmup", lane="b", offset=-1.3)
                 for _ in range(SR.LANE_CHANGE_HOLD_FRAMES)]
              + _hold("b", 0.1))
    assert not SR.changed_lane(frames)
    assert SR.changed_lane(frames, scored_only=False)


def test_c3_withholds_the_goal_when_the_ego_never_changed_lane():
    """Driving straight past reaches the goal region and declines the scenario."""
    straight = [_frame(lane="a", offset=0.1) for _ in range(30)]
    default = T.classify(straight, goal_reached=True, t_max_s=3.0, dt=0.1)
    assert default.reason is T.TerminationReason.GOAL_REACHED

    c3 = T.classify(straight, goal_reached=True, t_max_s=3.0, dt=0.1,
                    rules=SR.rules_for_scenario(_C3_META))
    assert c3.reason is not T.TerminationReason.GOAL_REACHED
    # Withheld, not terminated early: it falls through to the budget.
    assert c3.reason is T.TerminationReason.BUDGET_EXPIRED


def test_c3_honours_the_goal_after_a_real_lane_change():
    frames = [_frame(lane="a", offset=1.4) for _ in range(5)]
    frames += [_frame(lane="b", offset=-1.3)]
    frames += [_frame(lane="b", offset=0.1) for _ in range(24)]
    assert SR.changed_lane(frames)
    t = T.classify(frames, goal_reached=True, t_max_s=3.0, dt=0.1,
                   rules=SR.rules_for_scenario(_C3_META))
    assert t.reason is T.TerminationReason.GOAL_REACHED
    assert "lane change" in t.detail


def test_c3_ignores_a_lane_change_after_the_goal_frame():
    """The manoeuvre has to happen on the way, not after arriving."""
    frames = [_frame(lane="a", offset=0.1) for _ in range(10)]
    frames += [_frame(lane="a", offset=1.4)] + _hold("b", -1.3)
    t = T.classify(frames, goal_reached=True, goal_frame=5, t_max_s=1.6, dt=0.1,
                   rules=SR.rules_for_scenario(_C3_META))
    assert t.reason is not T.TerminationReason.GOAL_REACHED


def test_c3_without_lane_columns_is_not_checked_rather_than_failed():
    """Absent data means "not checked" here, as it does for DAC and DDC.

    A harness that does not populate lane_id would otherwise report every C-3
    episode as a declined manoeuvre, which is indistinguishable from a policy
    that really declined it.
    """
    bare = [{"phase": "scored", "ego_speed": 5.0, "on_drivable": True,
             "driving_direction_ok": True, "contacts": []} for _ in range(30)]
    assert not SR.has_lane_data(bare)
    t = T.classify(bare, goal_reached=True, t_max_s=3.0, dt=0.1,
                   rules=SR.rules_for_scenario(_C3_META))
    assert t.reason is T.TerminationReason.GOAL_REACHED


# --- the default path must be untouched -----------------------------------

@pytest.mark.parametrize("goal", [True, False])
def test_passing_default_rules_changes_nothing(goal):
    frames = [_frame() for _ in range(30)]
    frames[7]["on_drivable"] = False
    frames[8]["on_drivable"] = False
    frames[9]["on_drivable"] = False
    frames[10]["on_drivable"] = False
    frames[11]["on_drivable"] = False
    a = T.classify(frames, goal_reached=goal, t_max_s=3.0, dt=0.1)
    b = T.classify(frames, goal_reached=goal, t_max_s=3.0, dt=0.1,
                   rules=SR.DEFAULT_RULES)
    assert (a.reason, a.frame) == (b.reason, b.frame)


def test_live_monitor_applies_the_same_rules_as_the_post_hoc_pass():
    """The live ending and the re-scored ending must not disagree on a leaf."""
    mon = T.LiveMonitor(None, warmup=0, dt=0.1, t_max_s=3.0,
                        rules=SR.rules_for_leaf("I-2"))
    end = None
    for _ in range(30):
        end = mon.update(ego_xy=(0.0, 0.0), ego_speed=5.0,
                         driving_direction_ok=False)
        if end is not None:
            break
    assert end is None or end.reason is not T.TerminationReason.WRONG_WAY
    assert mon.final().reason is not T.TerminationReason.WRONG_WAY
