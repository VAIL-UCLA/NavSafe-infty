# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Direct unit coverage for EPDMSLiveScorer — the live metric of record.

The live half historically had no unit tests of its own (it was pinned only
transitively, through the md-vs-fast window parity harness that retired with
the scorer consolidation). These tests pin the live semantics directly on a
fake MetaDrive-style env: the at-fault conventions, the
pre-existing-contact exclusion, the lane-keeping streak counter, the
comfort/EC state machine, the dac-gates, and the exact no-ep aggregation.
"""

import numpy as np
import pytest

pytest.importorskip("shapely")

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (  # noqa: E402
    EPDMSLiveScorer,
    W_EXTENDED_COMFORT,
    W_HISTORY_COMFORT,
    W_LANE_KEEPING,
    W_NO_EP_TOTAL,
    W_TTC,
)


class _FakeObj:
    def __init__(self, name, pos, heading=0.0, vel=(0.0, 0.0),
                 length=4.5, width=1.9):
        self.name = name
        self.position = np.asarray(pos, dtype=float)
        self.heading_theta = float(heading)
        self.velocity = np.asarray(vel, dtype=float)
        self.top_down_length = length
        self.top_down_width = width


class _FakeEnv:
    """MetaDrive-style env: agent/engine are self, no map_manager."""

    def __init__(self):
        self.agent = self
        self.engine = self
        self.name = "__ego__"
        self.position = np.array([0.0, 0.0])
        self.heading_theta = 0.0
        self.velocity = np.array([0.0, 0.0])
        self._objects = {}

    def get_objects(self):
        return dict(self._objects)

    @property
    def map_manager(self):
        raise AttributeError("no map_manager on _FakeEnv")


def _scenario(with_lane=False, n_frames=60):
    map_features = {}
    if with_lane:
        xs = np.linspace(0.0, 200.0, 200)
        map_features["lane_0"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": np.stack([xs, np.zeros(200), np.zeros(200)], axis=1),
        }
    return {
        "map_features": map_features,
        "tracks": {},
        "metadata": {"sdc_id": "ego", "timestep": 0.1},
        "dynamic_map_states": {},
        "length": n_frames,
    }


def _make(with_lane=False):
    env = _FakeEnv()
    scorer = EPDMSLiveScorer(_scenario(with_lane=with_lane), env)
    scorer.reset_live_state()
    return env, scorer


def _step(env, scorer, frame, pos, heading=0.0, vel=(0.0, 0.0), objs=()):
    env.position = np.asarray(pos, dtype=float)
    env.heading_theta = float(heading)
    env.velocity = np.asarray(vel, dtype=float)
    env._objects = {o.name: o for o in objs}
    return scorer.score_frame_live(frame)


def test_at_fault_frontal_contact_zeroes_nc_and_ttc():
    env, scorer = _make()
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(3.0, 0.0),
              objs=[_FakeObj("lead", (3.0, 0.0))])
    assert r["no_at_fault_collisions"] == 0.0
    # nc gates ttc: an at-fault contact zeroes both.
    assert r["time_to_collision_within_bound"] == 0.0


def test_stopped_ego_struck_is_not_at_fault():
    env, scorer = _make()
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(0.0, 0.0),
              objs=[_FakeObj("striker", (3.0, 0.0), vel=(-4.0, 0.0))])
    assert r["no_at_fault_collisions"] == 1.0


def test_rear_contact_is_not_at_fault():
    env, scorer = _make()
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(3.0, 0.0),
              objs=[_FakeObj("tailgater", (-3.0, 0.0), vel=(5.0, 0.0))])
    assert r["no_at_fault_collisions"] == 1.0


def test_persisting_contact_is_charged_once():
    env, scorer = _make()
    lead = _FakeObj("lead", (3.0, 0.0))
    r0 = _step(env, scorer, 0, (0.0, 0.0), vel=(3.0, 0.0), objs=[lead])
    assert r0["no_at_fault_collisions"] == 0.0
    # Same contact next frame: excluded via the previous frame's contact
    # set — a persisting contact must not re-zero nc every frame.
    r1 = _step(env, scorer, 1, (0.2, 0.0), vel=(3.0, 0.0), objs=[lead])
    assert r1["no_at_fault_collisions"] == 1.0
    # Contact cleared for a frame -> the exclusion lapses with it.
    r2 = _step(env, scorer, 2, (50.0, 0.0), vel=(3.0, 0.0), objs=[])
    assert r2["no_at_fault_collisions"] == 1.0
    r3 = _step(env, scorer, 3, (50.0, 0.0), vel=(3.0, 0.0),
               objs=[_FakeObj("lead", (53.0, 0.0))])
    assert r3["no_at_fault_collisions"] == 0.0


def test_ttc_projects_ahead_without_current_contact():
    env, scorer = _make()
    # 4 m gap closing at 8 m/s: contact inside the 1 s TTC horizon but not
    # at the current frame.
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(8.0, 0.0),
              objs=[_FakeObj("lead", (9.0, 0.0), vel=(0.0, 0.0))])
    assert r["no_at_fault_collisions"] == 1.0
    assert r["time_to_collision_within_bound"] == 0.0


def test_lane_keeping_streak_zeroes_after_two_seconds():
    env, scorer = _make(with_lane=True)
    # 0.8 m lateral offset: > LANE_DEVIATION_LIMIT (0.5). scenario_dt=0.1,
    # LANE_KEEPING_WINDOW=2.0 -> the 21st consecutive deviating frame fails.
    results = [
        _step(env, scorer, f, (5.0 + 0.1 * f, 0.8), vel=(1.0, 0.0))
        for f in range(21)
    ]
    assert all(r["lane_keeping"] == 1.0 for r in results[:20])
    assert results[20]["lane_keeping"] == 0.0
    # Returning to the lane centre resets the streak.
    r = _step(env, scorer, 21, (7.2, 0.0), vel=(1.0, 0.0))
    assert r["lane_keeping"] == 1.0


def test_dac_zero_forces_ddc_and_lk_to_one():
    env, scorer = _make(with_lane=False)  # no lanes -> corners never inside
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(3.0, 0.0))
    assert r["drivable_area_compliance"] == 0.0
    assert r["driving_direction_compliance"] == 1.0
    assert r["lane_keeping"] == 1.0


def test_comfort_state_machine_and_reset():
    env, scorer = _make()
    # Frame 0: no history -> comfortable by construction.
    r0 = _step(env, scorer, 0, (0.0, 0.0), vel=(5.0, 0.0))
    assert r0["history_comfort"] == 1.0 and r0["extended_comfort"] == 1.0
    # Frame 1: 5 -> 6 m/s in 0.1 s = 10 m/s^2 accel (> 4.89): hc fails; the
    # accel jump (10 > 0.7) also breaks EC consistency.
    r1 = _step(env, scorer, 1, (0.5, 0.0), vel=(6.0, 0.0))
    assert r1["history_comfort"] == 0.0
    assert r1["extended_comfort"] == 0.0
    # reset_live_state clears the kinematic history AND the contact memory.
    scorer.reset_live_state()
    assert scorer.prev_velocity is None
    assert scorer._live_contact_names == set()
    r2 = _step(env, scorer, 2, (1.0, 0.0), vel=(5.0, 0.0))
    assert r2["history_comfort"] == 1.0 and r2["extended_comfort"] == 1.0


def test_no_ep_aggregation_formula():
    """Non-vacuous pin of the exact aggregation: gate 1, ttc 0 → 6/11.

    The previous version ran on the no-lane fixture, where dac=0 gated the
    score to 0.0 and the assertion reduced to 0.0 == 0.0 — any aggregation
    bug passed. Here the multiplicative gate is 1 and only ttc is lost (the
    constant-velocity projection hits a static lead inside the 1 s horizon,
    with no current contact), so the score must be bit-equal to
    ``multi_prod * (weighted_sum / W_NO_EP_TOTAL)`` — the live scorer's
    documented association order — which is exactly 6/11.
    """
    env, scorer = _make(with_lane=True)
    r = _step(env, scorer, 0, (10.0, 0.0), vel=(8.0, 0.0),
              objs=[_FakeObj("lead", (19.0, 0.0))])
    # The lead zeroes exactly one weighted term (ttc); every gate and
    # every other weighted term is 1.
    assert r["no_at_fault_collisions"] == 1.0
    assert r["time_to_collision_within_bound"] == 0.0
    assert r["drivable_area_compliance"] == 1.0
    assert r["driving_direction_compliance"] == 1.0
    assert r["traffic_light_compliance"] == 1.0
    assert r["lane_keeping"] == 1.0
    assert r["history_comfort"] == 1.0
    assert r["extended_comfort"] == 1.0

    multi_prod = (
        r["no_at_fault_collisions"] * r["drivable_area_compliance"]
        * r["driving_direction_compliance"] * r["traffic_light_compliance"]
    )
    weighted_sum = (
        W_TTC * r["time_to_collision_within_bound"]
        + W_LANE_KEEPING * r["lane_keeping"]
        + W_HISTORY_COMFORT * r["history_comfort"]
        + W_EXTENDED_COMFORT * r["extended_comfort"]
    )
    expected = multi_prod * (weighted_sum / W_NO_EP_TOTAL)
    assert r["score"] == expected
    assert r["score"] == 6.0 / 11.0
    assert r["valid"] is True


def test_malformed_dynamic_map_states_fails_loudly_at_init():
    """A malformed traffic-light entry must raise at construction.

    _check_tlc_live indexes ``entry['state']['object_state'][frame]`` every
    frame; before validation a malformed entry raised INSIDE the
    evaluator's per-frame catch, silently un-scoring the whole episode.
    The evaluator fails loudly on scorer-init errors, so init is where the
    data problem must surface.
    """
    env = _FakeEnv()
    sd = _scenario(with_lane=True)
    sd["dynamic_map_states"] = {"lane_0": {"state": {"wrong_key": []}}}
    with pytest.raises(ValueError, match="dynamic_map_states"):
        EPDMSLiveScorer(sd, env)


def test_none_dynamic_map_states_tolerated_as_empty():
    env = _FakeEnv()
    sd = _scenario(with_lane=True)
    sd["dynamic_map_states"] = None
    scorer = EPDMSLiveScorer(sd, env)
    scorer.reset_live_state()
    r = _step(env, scorer, 0, (10.0, 0.0), vel=(2.0, 0.0))
    assert r["traffic_light_compliance"] == 1.0
    assert r["valid"] is True


def test_failed_frame_commits_no_cross_frame_state():
    """Exception atomicity: a frame that raises mid-scoring must leave ALL
    cross-frame state untouched — contact memory, comfort history, and the
    lane-keeping streak — so the next healthy frame still charges a
    persisting collision and differentiates kinematics over one dt.
    """
    env, scorer = _make()
    lead = _FakeObj("lead", (3.0, 0.0))
    orig_tlc = scorer._check_tlc_live
    scorer._check_tlc_live = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("injected mid-frame failure"))
    with pytest.raises(RuntimeError, match="injected"):
        _step(env, scorer, 0, (0.0, 0.0), vel=(3.0, 0.0), objs=[lead])
    # Nothing committed by the failed frame.
    assert scorer._live_contact_names == set()
    assert scorer.prev_velocity is None
    assert scorer.prev_acceleration is None
    assert scorer.consecutive_lane_deviation == 0
    # Next healthy frame: the still-present collision IS charged (the
    # contact was never consumed into the exclusion memory) …
    scorer._check_tlc_live = orig_tlc
    r = _step(env, scorer, 1, (0.0, 0.0), vel=(3.0, 0.0), objs=[lead])
    assert r["no_at_fault_collisions"] == 0.0
    # … and the healthy frame's commit went through.
    assert scorer._live_contact_names == {"lead"}
    assert scorer.prev_velocity is not None


def test_lk_streak_resets_across_dac_gated_frames():
    """REPLACES ``test_lk_streak_freezes_across_dac_gated_frames``
    (2026-08-18 approved LK consecutive-semantics fix; the replaced test
    was added earlier the same day and never committed, so it has no git
    history — this docstring is its record).

    The replaced test pinned a defect: with the counter FROZEN across
    dac == 0 frames, deviations separated by seconds of off-drivable
    driving accumulated into one "consecutive" streak (reproduced: lk
    tripped after 6 rather than 21 consecutive deviating frames). Lane
    keeping is not evaluable on a dac-gated frame, so the 2 s window must
    restart: the streak now RESETS.
    """
    env, scorer = _make(with_lane=True)
    for f in range(10):
        r = _step(env, scorer, f, (5.0 + 0.1 * f, 0.8), vel=(1.0, 0.0))
        assert r["lane_keeping"] == 1.0
    assert scorer.consecutive_lane_deviation == 10
    # Off-drivable frame: dac == 0 → lk reported 1.0, streak RESET.
    r = _step(env, scorer, 10, (6.0, 50.0), vel=(1.0, 0.0))
    assert r["drivable_area_compliance"] == 0.0
    assert r["lane_keeping"] == 1.0
    assert scorer.consecutive_lane_deviation == 0
    # Post-gap deviations start a NEW streak: 20 more deviating frames
    # stay 1.0 (under the replaced freeze semantics the 11th tripped),
    # and only the 21st consecutive deviating frame trips the 2 s window.
    for i in range(20):
        r = _step(env, scorer, 11 + i, (6.5 + 0.1 * i, 0.8), vel=(1.0, 0.0))
        assert r["lane_keeping"] == 1.0, f"frame {11 + i}"
    r = _step(env, scorer, 31, (8.6, 0.8), vel=(1.0, 0.0))
    assert scorer.consecutive_lane_deviation == 21
    assert r["lane_keeping"] == 0.0


def test_lk_streak_resets_when_no_best_lane():
    """REPLACES ``test_lk_streak_freezes_when_no_best_lane``
    (2026-08-18 approved LK consecutive-semantics fix; the replaced test
    was added earlier the same day and never committed — see the note on
    the dac-gate test above).

    The replaced test pinned the second freeze path: with best_lane=None
    frames freezing the counter, deviations on either side of e.g. a
    wrong-way excursion accumulated into one streak. A frame with no
    best lane is not evaluable for lane keeping, so the streak now
    RESETS and a fresh streak starts when a lane matches again.
    """
    env, scorer = _make(with_lane=True)
    for f in range(3):
        _step(env, scorer, f, (5.0 + 0.1 * f, 0.8), vel=(1.0, 0.0))
    assert scorer.consecutive_lane_deviation == 3
    # Wrong-way heading at speed: dac still 1 (the lk check DID run), but
    # the heading-gated best-lane query returns None → RESET.
    r = _step(env, scorer, 3, (5.4, 0.8), heading=np.pi, vel=(-2.0, 0.0))
    assert r["drivable_area_compliance"] == 1.0
    assert r["lane_keeping"] == 1.0
    assert scorer.consecutive_lane_deviation == 0
    # Aligned again: a fresh streak starts at 1, not 4.
    _step(env, scorer, 4, (5.5, 0.8), vel=(1.0, 0.0))
    assert scorer.consecutive_lane_deviation == 1


# ---------------------------------------------------------------------------
# FIX 1 (2026-08-18): warm-up comfort-history seeding
# ---------------------------------------------------------------------------


def _observe(env, scorer, pos, heading=0.0, vel=(0.0, 0.0)):
    """Drive the env to a warm-up state and observe (no scoring)."""
    env.position = np.asarray(pos, dtype=float)
    env.heading_theta = float(heading)
    env.velocity = np.asarray(vel, dtype=float)
    scorer.observe_frame_kinematics()


def test_warmup_seeding_makes_first_scored_frame_measured():
    """A violent warm-up→policy handoff must be MEASURED, not forced-passed.

    Pre-fix, the first scored frame always had ``prev_velocity=None`` and
    reported hc = ec = 1.0 constitutively — masking above-bound handoff
    kinematics in ~95% of real episodes. With one warm-up frame observed,
    a 0 → 5 m/s jump over one dt (50 m/s² ≫ 4.89) drops hc and ec to 0.
    """
    env, scorer = _make()
    _observe(env, scorer, (0.0, 0.0), vel=(0.0, 0.0))
    assert scorer.prev_velocity is not None
    r = _step(env, scorer, 1, (0.5, 0.0), vel=(5.0, 0.0))
    assert r["history_comfort"] == 0.0
    assert r["extended_comfort"] == 0.0


def test_warmup_seeding_feeds_jerk_chain():
    """Two observed warm-up frames seed the accel history, so the first
    scored frame's JERK is measured too (pre-fix, frame-0's placeholder
    ``prev_acceleration=0.0`` fabricated the ordinal-1 jerk).
    """
    env, scorer = _make()
    _observe(env, scorer, (0.0, 0.0), vel=(0.0, 0.0))
    # 0 → 1 m/s over dt stages a 10 m/s² accel into the history.
    _observe(env, scorer, (0.1, 0.0), vel=(1.0, 0.0))
    assert scorer.prev_acceleration == pytest.approx(10.0)
    # Scored frame holds 1 m/s: accel 0, jerk |0 − 10| / 0.1 = 100 > 8.37.
    r = _step(env, scorer, 2, (0.2, 0.0), vel=(1.0, 0.0))
    assert r["history_comfort"] == 0.0


def test_no_warmup_first_frame_stays_vacuous():
    """Legacy behavior preserved when there is NO warm-up
    (``ego_replay_frames=0``): the truly history-less first scored frame
    has nothing to measure against, so hc = ec = 1.0 exactly as before —
    the approved defect was ignoring AVAILABLE history, not this case.
    """
    env, scorer = _make()
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(9.0, 0.0))
    assert r["history_comfort"] == 1.0
    assert r["extended_comfort"] == 1.0


def test_reset_clears_seeded_history():
    """``reset_live_state`` drops seeded warm-up history: the next scored
    frame is history-less again (legacy vacuous pass)."""
    env, scorer = _make()
    _observe(env, scorer, (0.0, 0.0), vel=(0.0, 0.0))
    scorer.reset_live_state()
    assert scorer.prev_velocity is None
    r = _step(env, scorer, 0, (0.0, 0.0), vel=(9.0, 0.0))
    assert r["history_comfort"] == 1.0
    assert r["extended_comfort"] == 1.0


def test_observe_seeds_only_comfort_history():
    """Scope guard: warm-up seeding advances ONLY the comfort-history
    chain — NOT the contact set (a separate, unapproved change) and NOT
    the lane-keeping streak. A contact present during warm-up must still
    be charged on the first scored frame.
    """
    env, scorer = _make(with_lane=True)
    env._objects = {"lead": _FakeObj("lead", (7.0, 0.8))}
    _observe(env, scorer, (5.0, 0.8), vel=(1.0, 0.0))
    assert scorer.prev_velocity is not None
    assert scorer._live_contact_names == set()
    assert scorer.consecutive_lane_deviation == 0
    # First scored frame, same overlapping lead: charged — the warm-up
    # observation did NOT pre-exclude it as a pre-existing contact.
    r = _step(env, scorer, 1, (5.0, 0.8), vel=(1.0, 0.0),
              objs=[_FakeObj("lead", (7.0, 0.8))])
    assert r["no_at_fault_collisions"] == 0.0


# ---------------------------------------------------------------------------
# FIX 3 (2026-08-18): traffic-light frame alignment
# ---------------------------------------------------------------------------


def _scenario_with_red_from(transition, n_states=10):
    """Lane fixture plus a red-light transition at index ``transition``.

    The light-bearing lane is an intersection CONNECTOR: its START (x=0)
    is the stop line (see TL_STOP_LINE_ZONE_M).
    """
    sd = _scenario(with_lane=True)
    sd["dynamic_map_states"] = {
        "lane_0": {"state": {"object_state": [
            "LANE_STATE_GO" if i < transition else "LANE_STATE_STOP"
            for i in range(n_states)]}}}
    return sd


def _tlc_at(frame_idx, transition=5, n_states=10):
    """Score one frame CROSSING the connector's stop line at speed: the ego
    centre was before the start on the previous (warm-up) frame and is
    0.3 m past it now."""
    env = _FakeEnv()
    scorer = EPDMSLiveScorer(_scenario_with_red_from(transition, n_states), env)
    scorer.reset_live_state()
    # The fixture lane has no map polygon; LaneProxy buffers its centerline
    # with FLAT caps, so the footprint starts exactly at x=0: (-3, 0) is
    # outside, (0.3, 0) is just across the line.
    _observe(env, scorer, (-3.0, 0.0), vel=(3.0, 0.0))
    r = _step(env, scorer, frame_idx, (0.3, 0.0), vel=(3.0, 0.0))
    return r["traffic_light_compliance"]


def test_tlc_charged_on_transition_frame_not_one_late():
    """The evaluator scores the POST-step world: after ``env.step`` the
    ego/agents are at scenario timestep f+1, and since the 2026-08-18 TL
    alignment fix it passes that timestep (``_scored_world_timestep``) to
    ``score_frame_live``. For a light turning red at timestep 5, scoring
    the timestep-5 world with frame_idx=5 charges the violation ON the
    transition frame.
    """
    assert _tlc_at(5) == 0.0
    # Pre-fix the evaluator passed the PRE-step index (4) for that same
    # world, looked up the still-green logged state, and charged the
    # violation one frame (0.1 s) late. Pinned as documentation: index 4
    # is green in the log, so a caller passing 4 sees no violation.
    assert _tlc_at(4) == 1.0


def test_tlc_holds_last_logged_state_past_log_end():
    """Last-frame bound: past the end of the logged light states the world
    the scorer reads is frozen at the last logged frame (the env replay
    clamps ``t = min(timestep, len-1)``), so the TL lookup clamps the same
    way and holds the final state. Pre-fix, out-of-range read as "no state
    → not red", silently clearing a red light at the log boundary.
    """
    assert _tlc_at(10, transition=5, n_states=6) == 0.0


# --- 2026-08-23: the stop line is the connector's START, and the charge is
# the CROSSING (see TL_STOP_LINE_ZONE_M in the scorer module).


def _red_scorer():
    env = _FakeEnv()
    scorer = EPDMSLiveScorer(_scenario_with_red_from(0), env)
    scorer.reset_live_state()
    return env, scorer


def test_tlc_is_not_charged_at_the_connectors_end():
    """The old rule charged the LAST 5 m of any red polygon the ego box
    touched -- on connectors, the exit, which the sibling connectors merging
    into the same junction exit share with the ego's own green lane. An ego
    that entered elsewhere and is leaving the red polygon is not running it."""
    env, scorer = _red_scorer()
    _observe(env, scorer, (196.0, 0.0), vel=(3.0, 0.0))
    r = _step(env, scorer, 1, (197.0, 0.0), vel=(3.0, 0.0))
    assert r["traffic_light_compliance"] == 1.0


def test_tlc_is_not_charged_while_already_inside():
    """Crossing, not presence: the centre was inside on the previous frame
    (a scored phase starting inside a connector the warm-up entered, or a
    light that turned red after entry), so no crossing happens here."""
    env, scorer = _red_scorer()
    _observe(env, scorer, (1.0, 0.0), vel=(3.0, 0.0))
    r = _step(env, scorer, 1, (1.3, 0.0), vel=(3.0, 0.0))
    assert r["traffic_light_compliance"] == 1.0


def test_tlc_is_not_charged_for_a_nose_over_the_line():
    """Containment is tested on the CENTRE: a car braking to the line with
    its bonnet over it at 1.5 m/s has not crossed (half length 2.26 m, so a
    centre at -1.0 m puts the nose 1.26 m past the line)."""
    env, scorer = _red_scorer()
    _observe(env, scorer, (-2.5, 0.0), vel=(1.5, 0.0))
    r = _step(env, scorer, 1, (-1.0, 0.0), vel=(1.5, 0.0))
    assert r["traffic_light_compliance"] == 1.0


def test_tlc_is_not_charged_at_walking_pace():
    env, scorer = _red_scorer()
    _observe(env, scorer, (-3.0, 0.0), vel=(0.8, 0.0))
    r = _step(env, scorer, 1, (0.3, 0.0), vel=(0.8, 0.0))
    assert r["traffic_light_compliance"] == 1.0


def test_tlc_crossing_is_charged_once_then_the_car_is_inside():
    # A centre ON the boundary (x=0) is not "contained"; judge poses that
    # are clearly outside, then clearly inside.
    env, scorer = _red_scorer()
    _observe(env, scorer, (-3.0, 0.0), vel=(5.0, 0.0))
    assert _step(env, scorer, 1, (0.5, 0.0), vel=(5.0, 0.0))[
        "traffic_light_compliance"] == 0.0
    assert _step(env, scorer, 2, (1.0, 0.0), vel=(5.0, 0.0))[
        "traffic_light_compliance"] == 1.0


def test_tlc_side_entry_far_from_the_start_is_not_a_crossing():
    """Entering the red polygon through its side 50 m in (a merge) is not
    crossing its stop line."""
    env, scorer = _red_scorer()
    _observe(env, scorer, (50.0, 2.5), vel=(0.0, -3.0))
    r = _step(env, scorer, 1, (50.0, 1.0), heading=-np.pi / 2, vel=(0.0, -3.0))
    assert r["traffic_light_compliance"] == 1.0


def test_tlc_without_history_falls_back_to_the_zone_test():
    """No warm-up and a first frame: nothing to cross from, so the stateless
    zone test applies (centre within 5 m of the start, moving)."""
    env, scorer = _red_scorer()
    assert _step(env, scorer, 1, (2.0, 0.0), vel=(3.0, 0.0))[
        "traffic_light_compliance"] == 0.0
    scorer.reset_live_state()
    assert _step(env, scorer, 1, (8.0, 0.0), vel=(3.0, 0.0))[
        "traffic_light_compliance"] == 1.0


def test_signal_hold_is_published_as_a_state_fact():
    """``signal_hold``: a red stop line ahead within 30 m on the ego's
    heading, or just crossed -- at any speed. It never enters the score."""
    env, scorer = _red_scorer()
    ahead = _step(env, scorer, 1, (-10.0, 0.0), vel=(0.0, 0.0))
    assert ahead["signal_hold"] == 1.0
    assert ahead["traffic_light_compliance"] == 1.0
    far = _step(env, scorer, 1, (-40.0, 0.0), vel=(0.0, 0.0))
    assert far["signal_hold"] == 0.0
    beside = _step(env, scorer, 1, (-10.0, 6.0), vel=(0.0, 0.0))
    assert beside["signal_hold"] == 0.0
    # Facing away from the line: not held by it.
    away = _step(env, scorer, 1, (-10.0, 0.0), heading=np.pi, vel=(0.0, 0.0))
    assert away["signal_hold"] == 0.0
    # Just across the line (an IDM standstill can leave the centre there).
    over = _step(env, scorer, 1, (0.4, 0.0), vel=(0.0, 0.0))
    assert over["signal_hold"] == 1.0
    # Green: no hold.
    env2 = _FakeEnv()
    green = EPDMSLiveScorer(_scenario_with_red_from(100), env2)
    green.reset_live_state()
    assert _step(env2, green, 1, (-10.0, 0.0), vel=(0.0, 0.0))["signal_hold"] == 0.0


def test_evaluator_scored_world_timestep_helper():
    """The evaluator-side half of the TL alignment: the frame passed to
    ``score_frame_live`` is the env-reported post-step scenario timestep,
    falling back to frame+1 when the env does not report one. (The
    emitted ``epdms_metrics['frame']`` stamp is unchanged — the evaluator
    adds it separately.)
    """
    from navsafe.evaluation.evaluator import Evaluator

    assert Evaluator._scored_world_timestep(7, {"scenario_timestep": 8}) == 8
    # Env counter wins even if it disagrees with frame+1 (e.g. loop_replay
    # wraparound): the scorer must see the world actually being scored.
    assert Evaluator._scored_world_timestep(7, {"scenario_timestep": 3}) == 3
    assert Evaluator._scored_world_timestep(7, {}) == 8
    assert Evaluator._scored_world_timestep(7, None) == 8
