"""Tests for the NavSafe termination taxonomy (navsafe/benchmark/termination.py)."""

from __future__ import annotations

import pytest

from navsafe.benchmark import termination as T
from navsafe.benchmark.trace.schema import PHASE_SCORED, PHASE_WARMUP, empty_frame


def _frames(n=100, warmup=5, speed=5.0):
    out = []
    for i in range(n):
        f = empty_frame()
        f["frame"] = i
        f["phase"] = PHASE_WARMUP if i < warmup else PHASE_SCORED
        f["ego_speed"] = speed
        f["on_drivable"] = True
        f["driving_direction_ok"] = True
        f["contacts"] = []
        out.append(f)
    return out


def test_goal_reached():
    t = T.classify(_frames(), goal_reached=True, t_max_s=10.0)
    assert t.reason is T.TerminationReason.GOAL_REACHED
    assert t.reason.policy_attributed


def test_budget_expiry_when_moving():
    # 95 scored frames = 9.5 s, so a 9 s budget is genuinely used up.
    t = T.classify(_frames(), goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.BUDGET_EXPIRED
    assert t.reason.policy_attributed


def test_deadlock_needs_sustained_hold():
    fr = _frames()
    for f in fr[20:68]:          # 48 reconstructed frames: below 1-frame tolerance
        f["ego_speed"] = 0.0
    t = T.classify(fr, goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.BUDGET_EXPIRED


def test_reconstructed_deadlock_has_one_frame_measurement_tolerance():
    fr = _frames()
    for f in fr[20:69]:
        f["ego_speed"] = 0.0
    t = T.classify(fr, goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.DEADLOCK
    assert t.frame == 68

    for f in fr[20:71]:          # 51 frames >= 5.0 s
        f["ego_speed"] = 0.0
    t = T.classify(fr, goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.DEADLOCK


def test_signal_hold_frames_do_not_count_toward_deadlock():
    """Waiting at a red light is not a freeze: frames flagged ``signal_hold``
    (the evaluator's "red ahead on the ego's lane" state fact) reset the
    standstill hold. A genuinely frozen ego near a red light still ends on
    budget expiry, so the exemption cannot manufacture a success."""
    fr = _frames()
    for f in fr[20:80]:          # 60 stationary frames: a deadlock without the hold
        f["ego_speed"] = 0.0
        f["signal_hold"] = True
    t = T.classify(fr, goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.BUDGET_EXPIRED
    # The same stop with the light gone (green) is the ordinary deadlock.
    for f in fr[20:80]:
        f["signal_hold"] = False
    assert T.classify(fr, goal_reached=False, t_max_s=9.0).reason is \
        T.TerminationReason.DEADLOCK
    # The hold resets the count: a 30-frame stop after a held stop is
    # measured from the release.
    for f in fr[20:50]:
        f["signal_hold"] = True
    assert T.classify(fr, goal_reached=False, t_max_s=9.0).reason is \
        T.TerminationReason.BUDGET_EXPIRED


def test_live_monitor_threads_the_signal_hold():
    m = T.LiveMonitor(None, warmup=0, dt=0.1, t_max_s=100.0)
    for _ in range(70):
        assert m.update(ego_xy=(0.0, 0.0), ego_speed=0.0, signal_hold=True) is None
    ended = None
    for _ in range(70):
        ended = m.update(ego_xy=(0.0, 0.0), ego_speed=0.0)
        if ended is not None:
            break
    assert ended is not None and ended.reason is T.TerminationReason.DEADLOCK


def test_deadlock_distinct_from_budget_expiry():
    # crawling (slow but moving) is budget expiry, not deadlock
    t = T.classify(_frames(speed=0.3), goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.BUDGET_EXPIRED


def test_contact_fault_split():
    fr = _frames()
    fr[30]["contacts"] = [{"agent_id": "a1", "at_fault": False,
                           "kind": "rear_end", "rel_speed": 2.0}]
    # Strict protocol (2026-09-01): ANY contact ends the episode; fault only
    # decides whether the ending is charged to the policy.
    t = T.classify(fr, goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.CONTACT_NOT_AT_FAULT
    assert not t.reason.policy_attributed

    fr[30]["contacts"][0]["at_fault"] = True
    t = T.classify(fr, goal_reached=False, t_max_s=9.0)
    assert t.reason is T.TerminationReason.CONTACT_AT_FAULT
    assert "a1" in t.detail


def test_wrong_way_masks_later_contact():
    """Committing to an opposing lane at 2 s is the episode; what it hits at 3 s
    happened after the run was already over."""
    fr = _frames()
    fr[20]["driving_direction_ok"] = False
    fr[30]["contacts"] = [{"agent_id": "a1", "at_fault": True,
                           "kind": "angle", "rel_speed": 1.0}]
    t = T.classify(fr, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.WRONG_WAY
    assert t.frame == 20
    assert t.reason.policy_attributed


def test_wrong_way_terminates_as_an_infraction_and_fails_success():
    """It is the ego's fault, but Bench2Drive has no coefficient for it, so it
    terminates and fails SR instead of multiplying an invented penalty."""
    ww = T.Termination(T.TerminationReason.WRONG_WAY, 20)
    r = T.to_route_result("r", 60.0, ww, {})
    assert r is not None                       # scored, never excluded
    assert r.infractions == {"wrong_way": 1}
    assert r.driving_score() == pytest.approx(60.0)   # no penalty factor
    assert r.success() is False


def test_contact_masks_goal():
    fr = _frames()
    fr[30]["contacts"] = [{"agent_id": "a1", "at_fault": True,
                           "kind": "vru", "rel_speed": 1.0}]
    t = T.classify(fr, goal_reached=True, t_max_s=10.0)
    assert t.reason is T.TerminationReason.CONTACT_AT_FAULT


def test_off_drivable():
    fr = _frames()
    for f in fr[40:45]:
        f["on_drivable"] = False
    t = T.classify(fr, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.OFF_DRIVABLE
    assert t.frame == 44


def test_transient_drivable_boundary_sample_does_not_end_episode():
    """Strict DAC still scores the sample; termination debounces map seams."""
    fr = _frames()
    for f in fr[40:44]:
        f["on_drivable"] = False
    # A valid sample resets the consecutive hold before it reaches 0.5 s.
    fr[44]["on_drivable"] = True
    t = T.classify(fr, goal_reached=True, goal_frame=60, t_max_s=10.0)
    assert t.reason is T.TerminationReason.GOAL_REACHED


def test_infra_failure_and_empty_trace():
    t = T.classify(_frames(), goal_reached=True, t_max_s=10.0,
                   infra_error="renderer died")
    assert t.reason is T.TerminationReason.INFRA_FAILURE
    t2 = T.classify([], goal_reached=False, t_max_s=10.0)
    assert t2.reason is T.TerminationReason.INFRA_FAILURE


def test_to_route_result_mapping():
    goal = T.Termination(T.TerminationReason.GOAL_REACHED, 99)
    r = T.to_route_result("r", 100.0, goal, {})
    assert r.completed and r.success()

    # Deadlock adds NO infraction (2026-08-28): it is a reason, and the episode
    # already fails success() because it did not complete. The old
    # `vehicle_blocked` was a second name for one event, and a NON_PENALTY_KEY
    # besides, so it never moved DS either -- which this pins.
    dead = T.Termination(T.TerminationReason.DEADLOCK, 50)
    r = T.to_route_result("r", 30.0, dead, {})
    assert r.infractions == {}
    assert not r.completed and not r.success()
    assert r.driving_score() == pytest.approx(30.0)

    slow = T.Termination(T.TerminationReason.BUDGET_EXPIRED, 99)
    r = T.to_route_result("r", 80.0, slow, {})
    assert r.infractions == {"route_timeout": 1}
    assert r.driving_score() == pytest.approx(80.0)  # timeout: SR-fail, no DS hit

    # not-at-fault contact terminates without adding an infraction
    naf = T.Termination(T.TerminationReason.CONTACT_NOT_AT_FAULT, 30)
    r = T.to_route_result("r", 45.0, naf, {})
    assert r.infractions == {} and r.driving_score() == pytest.approx(45.0)

    # benchmark endings are excluded, not zeroed. ENVELOPE_EXIT is retired and
    # classify never returns it; a stored trace that already carries it must
    # still be excluded rather than folded in as a score.
    for reason in (T.TerminationReason.ENVELOPE_EXIT,
                   T.TerminationReason.INFRA_FAILURE):
        assert T.to_route_result("r", 45.0, T.Termination(reason, 20), {}) is None


# --- earliest-event-wins (taxonomy §0.1: exactly one reason per episode) ----

def test_goal_before_contact_is_a_completion():
    """A goal reached at 4 s is not undone by a contact at 9 s -- the episode
    ended at the goal, and the frames after it are not the episode's."""
    fr = _frames()
    fr[90]["contacts"] = [{"agent_id": "a1", "at_fault": True,
                           "kind": "angle", "rel_speed": 1.0}]
    t = T.classify(fr, goal_reached=True, goal_frame=40, t_max_s=10.0)
    assert t.reason is T.TerminationReason.GOAL_REACHED
    assert t.frame == 40


def test_deadlock_before_wrong_way_is_the_deadlock():
    """Earliest event wins. Freezing at 3 s and only then ending up against the
    traffic direction at 8 s is a frozen policy."""
    fr = _frames()
    for f in fr[5:60]:
        f["ego_speed"] = 0.0
    fr[80]["driving_direction_ok"] = False
    t = T.classify(fr, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.DEADLOCK
    assert t.reason.policy_attributed


def test_contact_before_wrong_way_is_charged():
    fr = _frames()
    fr[10]["contacts"] = [{"agent_id": "a1", "at_fault": True,
                           "kind": "rear_end", "rel_speed": 3.0}]
    fr[40]["driving_direction_ok"] = False
    t = T.classify(fr, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.CONTACT_AT_FAULT
    assert t.frame == 10


def test_same_frame_precedence_prefers_the_contact():
    """Both on frame 30: the contact is the more specific fact about it, and
    both are the ego's, so the ending names what actually happened."""
    fr = _frames()
    fr[30]["driving_direction_ok"] = False
    fr[30]["contacts"] = [{"agent_id": "a1", "at_fault": True,
                           "kind": "angle", "rel_speed": 1.0}]
    t = T.classify(fr, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.CONTACT_AT_FAULT


def test_short_window_is_not_reported_as_budget_expiry():
    """A 9.5 s window under a 15 s ceiling ran out of FRAMES, not of time.

    That is the harness stopping, not a policy blowing its budget, so it gets
    its own reason and is not attributed to the policy -- previously both
    landed on `budget_expired` and were told apart only by the detail string.
    """
    t = T.classify(_frames(n=100, warmup=5), goal_reached=False, t_max_s=15.0)
    assert t.reason is T.TerminationReason.TRACE_EXHAUSTED
    assert not t.reason.policy_attributed
    assert "short of" in t.detail
    t2 = T.classify(_frames(n=100, warmup=5), goal_reached=False, t_max_s=9.0)
    assert "elapsed" in t2.detail


def test_dead_end_mission_walk_extension():
    """05ee09cb class: a mission walk that dead-ends short of the mission
    is extended along the logged mission; covered walks are untouched."""
    import numpy as np
    from navsafe.policy.state.pdm_closed_planner.route import (
        _extend_dead_end_walk_with_mission)

    n = 200
    xs = np.linspace(0.0, 100.0, n)
    sd = {"metadata": {"sdc_id": "ego"},
          "tracks": {"ego": {"state": {
              "position": np.stack([xs, np.zeros(n)], axis=1),
              "valid": np.ones(n, bool)}}}}
    # Walk covers 0..40 m; mission runs to 100 m -> extended past the wall.
    walk = np.stack([np.linspace(0, 40, 41), np.zeros(41)], axis=1)
    diag = {}
    out = _extend_dead_end_walk_with_mission(walk, sd, 0, diagnostics=diag)
    assert out[-1][0] > 95.0 and diag["route_dead_end_extension_m"] > 50
    # Walk already covers the mission -> byte-identical, no diagnostic.
    full = np.stack([np.linspace(0, 100, 101), np.zeros(101)], axis=1)
    diag2 = {}
    out2 = _extend_dead_end_walk_with_mission(full, sd, 0, diagnostics=diag2)
    assert np.array_equal(out2, full) and "route_dead_end_extension_m" not in diag2
    # A shortfall past the goal latch's 2 m slack extends even when small —
    # 05ee09cb ended 5.8 m short and the old 8 m floor declined it.
    short = np.stack([np.linspace(0, 94.5, 95), np.zeros(95)], axis=1)
    diag3 = {}
    out3 = _extend_dead_end_walk_with_mission(short, sd, 0, diagnostics=diag3)
    assert out3[-1][0] > 99.0 and diag3["route_dead_end_extension_m"] >= 5.0
    # Mission nowhere near the walk end (off-route dead end) -> untouched.
    off = np.stack([np.linspace(0, 40, 41), np.full(41, 30.0)], axis=1)
    assert np.array_equal(
        _extend_dead_end_walk_with_mission(off, sd, 0, diagnostics={}), off)


class TestStationaryCeiling:
    """The signal / stop-line deadlock exemptions are bounded
    (0e272e003af65a71: a stop-line park behind frozen traffic stepped past
    frame 11,000 under the clock-free protocol)."""

    @staticmethod
    def _held(n, warmup=5, signal=True):
        fs = _frames(n=n, warmup=warmup)
        for i in range(warmup, n):
            fs[i]["ego_speed"] = 0.0
            fs[i]["signal_hold"] = signal
        return fs

    def test_exempt_hold_ends_at_the_ceiling(self):
        n = 5 + int(T.STATIONARY_CEILING_S * 10) + 5
        t = T.classify(self._held(n), goal_reached=False, t_max_s=float("inf"))
        assert t.reason is T.TerminationReason.DEADLOCK
        assert "bounded" in t.detail

    def test_exempt_hold_below_the_ceiling_is_still_exempt(self):
        n = 5 + int(T.STATIONARY_CEILING_S * 10) - 50
        t = T.classify(self._held(n), goal_reached=False, t_max_s=float("inf"))
        assert t.reason is not T.TerminationReason.DEADLOCK

    def test_unexempt_freeze_still_ends_at_five_seconds(self):
        fs = self._held(200, signal=False)
        t = T.classify(fs, goal_reached=False, t_max_s=float("inf"))
        assert t.reason is T.TerminationReason.DEADLOCK
        assert "5 s" in t.detail


def test_terminal_lane_divergence_snap():
    """107a64d4 class: a mission walk whose final stretch commits to the
    neighbouring lane is spliced back onto the mission; on-mission walks
    are untouched."""
    import numpy as np
    from navsafe.policy.state.pdm_closed_planner.route import (
        _snap_terminal_lane_divergence_to_mission)

    n = 200
    xs = np.linspace(0.0, 100.0, n)
    sd = {"metadata": {"sdc_id": "ego"},
          "tracks": {"ego": {"state": {
              "position": np.stack([xs, np.zeros(n)], axis=1),
              "valid": np.ones(n, bool)}}}}
    # Walk follows the mission to x=60, then forks a lane left (y=3.5) and
    # runs parallel to x=95 — the 107a geometry (3.4 m lateral, ends near
    # the mission's end).
    on = np.stack([np.linspace(0, 60, 61), np.zeros(61)], axis=1)
    fork = np.stack([np.linspace(61, 95, 35), np.full(35, 3.5)], axis=1)
    walk = np.vstack([on, fork])
    diag = {}
    out = _snap_terminal_lane_divergence_to_mission(walk, sd, 0,
                                                    diagnostics=diag)
    assert diag["route_terminal_lane_snap_m"] >= 3.0
    # The spliced route ends at the mission's end, on the mission lane.
    assert abs(out[-1][1]) < 1e-9 and out[-1][0] > 99.0
    # Nothing after the splice anchor sits on the diverged lane.
    assert float(np.abs(out[:, 1]).max()) < 3.5
    # A walk that stays on the mission lane is byte-identical.
    full = np.stack([np.linspace(0, 100, 101), np.zeros(101)], axis=1)
    diag2 = {}
    out2 = _snap_terminal_lane_divergence_to_mission(full, sd, 0,
                                                     diagnostics=diag2)
    assert np.array_equal(out2, full) and not diag2
    # Mild wobble (0.9 m, the measured passing runs) is untouched.
    wob = np.stack([np.linspace(0, 98, 99), np.full(99, 0.9)], axis=1)
    assert np.array_equal(
        _snap_terminal_lane_divergence_to_mission(wob, sd, 0,
                                                  diagnostics={}), wob)
    # Divergence far from the mission's end (mid-route bypass) is not this
    # rule's business.
    mid = np.vstack([np.stack([np.linspace(0, 20, 21), np.zeros(21)], axis=1),
                     np.stack([np.linspace(21, 50, 30),
                               np.full(30, 3.5)], axis=1)])
    assert np.array_equal(
        _snap_terminal_lane_divergence_to_mission(mid, sd, 0,
                                                  diagnostics={}), mid)
