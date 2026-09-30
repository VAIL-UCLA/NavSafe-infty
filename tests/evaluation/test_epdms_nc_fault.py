# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A stationary ego must not be charged an at-fault collision.

Two independent bugs made the at-fault guard unreachable:

1. ``nc`` read ``states['speed'][t]`` — the CANDIDATE's savgol-smoothed speed
   profile, which measures ~0.31 m/s for a genuinely stationary ego — instead
   of the ego's actual speed, so ``ego_v < STOPPED_SPEED_THRESHOLD`` (0.05)
   never fired.
2. There was no pre-existing-contact exclusion, so an agent already touching
   the ego at t=0 (which no candidate caused and none can avoid) zeroed ``nc``
   on every proposal.

Observed together at b040d87a f35-f45: a 0.00 m/s ego charged an at-fault
collision for 11 consecutive frames against a lead already overlapping it and
RECEDING at 3.5 m/s. 49 of 111 gate-failing frames were this artifact.

Calling convention: since the 2026-08-18 terminal-pose fix,
``_calculate_metrics(states, horizon, ...)`` scores ``horizon + 1`` poses
(prepended current pose + waypoints), so these tests pass ``horizon = n - 1``
for their length-``n`` arrays — the exact same n scored poses (and the same
expected values) as before the fix, just under the new contract.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely")

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (  # noqa: E402
    EPDMSTrajectoryScorer_Fast,
    PDM_STOPPED_SPEED_THRESHOLD,
    STOPPED_SPEED_THRESHOLD,
    VEHICLE_LENGTH,
)


def _stationary_states(n=8, x=0.0, y=0.0, smoothed_speed=0.31):
    """A stopped ego whose SMOOTHED speed profile reads non-zero.

    This is the crux: the candidate's profile says it is creeping, the ego
    itself is not moving.
    """
    return _with_comfort({
        "x": np.full(n, x),
        "y": np.full(n, y),
        "heading": np.zeros(n),
        "speed": np.full(n, smoothed_speed),
    }, n)


def _with_comfort(states, n):
    """Add the comfort-term arrays _calculate_metrics expects."""
    for k in ("acceleration", "yaw_rate", "jerk", "lon_accel", "lon_jerk",
              "yaw_accel", "lat_accel"):
        states.setdefault(k, np.zeros(n))
    return states


def _ego(speed):
    """Enriched ego_state shape the scorer expects (see collector._enrich_ego_state)."""
    return {
        "speed": speed,
        "acceleration": np.zeros(3),
        "angular_velocity": np.zeros(3),
    }


def _scorer():
    s = EPDMSTrajectoryScorer_Fast(verbose=False)
    s.planner_dt = 0.5
    # Stub the lane state normally built by initialize(): these tests exercise
    # nc only, and an empty lane set keeps the dac path inert.
    s.all_lanes = []
    s.all_lane_bounds = np.zeros((0, 4), dtype=float)
    return s


def _agent_at(scorer, x, y, obj_id="agent0"):
    poly = scorer._get_agent_polygon(x, y, 0.0, 4.0, 1.8)
    return (poly, poly.bounds, x, y, obj_id)


def test_smoothed_speed_alone_would_have_cleared_the_guard():
    """Pins the bug's premise: the profile value passes the stopped test."""
    assert 0.31 > STOPPED_SPEED_THRESHOLD


def test_stopped_ego_hit_by_agent_ahead_is_not_at_fault():
    scorer = _scorer()
    n = 8
    states = _stationary_states(n)
    # Agent arrives MID-horizon (clear of the ego at t=0, so the
    # pre-existing-contact exclusion cannot decide this test — review
    # showed the original overlapping-at-t0 version passed even with the
    # speed guard fully disabled) and strikes the stationary ego from
    # ahead at t>=4.
    agents_per_t = [
        [_agent_at(scorer, 20.0 - 4.0 * i, 0.0)] for i in range(n)
    ]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(0.0),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 1.0, (
        "stationary ego charged an at-fault collision — the guard is reading "
        "the candidate's smoothed speed again")


def test_preexisting_contact_is_not_charged_even_when_ego_moves():
    """The b040d87a case: contact already exists at t=0.

    No candidate caused it and none can avoid it, so it must not zero nc.
    """
    scorer = _scorer()
    n = 8
    states = _with_comfort({
        "x": np.arange(n, dtype=float),  # ego genuinely moving
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 2.0),
    }, n)
    # Agent overlaps the ego at t=0 and stays ahead.
    agents_per_t = [[_agent_at(scorer, 0.5 + i, 0.0)] for i in range(n)]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(2.0),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 1.0, "pre-existing contact was charged to the candidate"


def test_moving_ego_striking_a_new_agent_ahead_is_still_at_fault():
    """The guard must not become a blanket amnesty."""
    scorer = _scorer()
    n = 8
    states = _with_comfort({
        "x": np.arange(n, dtype=float) * 2.0,
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 5.0),
    }, n)
    # No contact at t=0 (agent far ahead); ego drives into it later.
    agents_per_t = [
        [_agent_at(scorer, 10.0, 0.0)] for _ in range(n)
    ]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(5.0),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 0.0, "a moving ego rear-ending a new agent must be faulted"


def test_carl_full_state_speed_closes_net_displacement_blind_spot():
    """CaRL reads PDM's instantaneous simulated velocity at contact.

    This is the generalized form of the 3a0e regression: an early tracker
    wobble has almost zero net displacement even though the simulated PDM
    state is moving.  The generic XY fallback may call it stopped; the
    PDM/CaRL path must classify the same front-edge contact as at fault.
    """
    scorer = _scorer()
    scorer.planner_dt = 0.1
    n = 4
    states = _with_comfort({
        "x": np.array([0.0, 0.0002, 0.0004, 0.0006]),
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 0.16665),
    }, n)
    # A moving actor crosses the ego front edge at t=0.2 s.  It is clear at
    # t=0, so the independent pre-existing-contact exclusion cannot decide.
    agents_per_t = [
        [_agent_at(scorer, 2.5, y)]
        for y in (5.0, 2.5, 0.0, -2.5)
    ]

    generic = scorer._calculate_metrics(
        states, n - 1, 0, ego_state=_ego(0.16665),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    faithful = scorer._calculate_metrics(
        states, n - 1, 0, ego_state=_ego(0.16665),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
        carl_collision_classifier=True,
    )
    assert generic["nc"] == 1.0
    assert faithful["nc"] == 0.0
    assert faithful["ttc"] == 0.0
    assert faithful["collision_time_s"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("ego_speed", "expected_nc"),
    [(STOPPED_SPEED_THRESHOLD, 1.0),
     (STOPPED_SPEED_THRESHOLD + 1e-6, 0.0)],
)
def test_carl_stopped_threshold_and_stopped_track_ordering(
    ego_speed, expected_nc,
):
    """CaRL uses ``<= 5e-2`` for the NC ego-stop gate, then stopped-track.

    This test previously asserted ``5e-2``'s sibling ``5e-3`` and cited it as
    CaRL's ego gate. It is not. The reference has two distinct constants and
    annotates them itself:

    * ``pdm_scorer_utils.get_collision_type(..., stopped_speed_threshold=5e-02)``
      is the NC classifier's ego gate, and ``pdm_scorer.py:318`` calls it
      WITHOUT overriding the default. ``is_track_stopped`` also defaults to
      ``5e-02``.
    * ``pdm_scorer.py:55`` declares ``STOPPED_SPEED_THRESHOLD = 5e-03`` with the
      inline comment ``# [m/s] (ttc)``, and its only use is line 401, inside
      ``_calculate_ttc``.

    So 5e-3 belongs to the TTC term alone. Pinning it here certified a gate the
    reference does not have, and it made the port punitive on an ego creeping
    between 0.005 and 0.05 m/s.
    """
    scorer = _scorer()
    n = 4
    states = _with_comfort({
        "x": np.array([0.0, 0.1, 0.2, 0.3]),
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, ego_speed),
    }, n)
    agents_per_t = [[_agent_at(scorer, 4.5, 0.0)] for _ in range(n)]
    metrics = scorer._calculate_metrics(
        states, n - 1, 0, ego_state=_ego(ego_speed),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
        carl_collision_classifier=True,
    )
    assert metrics["nc"] == expected_nc


def test_creep_from_standstill_into_agent_is_at_fault():
    """The stationary-ego exemption must expire once the CANDIDATE moves.

    The first fix substituted the ego's frame-0 actual speed at EVERY
    timestep, so a candidate that creeps off from a standstill and drives
    into an agent mid-horizon was never at fault — a blind spot where
    creep proposals collide for free. A per-timestep speed derived from
    the candidate's own scored positions closes it.
    """
    scorer = _scorer()
    n = 8
    # Creep: 0.5 m per 0.5 s step -> 3.5 m by the end of the horizon.
    states = _with_comfort({
        "x": np.arange(n, dtype=float) * 0.5,
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 1.0),
    }, n)
    # Agent parked 6 m ahead (clear of the ego's bumper at t=0, so not a
    # pre-existing contact), never moving; the creep reaches it at ~t=4.
    agents_per_t = [[_agent_at(scorer, 6.0, 0.0)] for _ in range(n)]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(0.0),   # frame-0 actual speed: standstill
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 0.0, (
        "a creep candidate that drives into a parked agent must be at "
        "fault even though the ego's frame-0 speed was zero")


def test_stationary_jitter_stays_exempt():
    """Savgol jitter oscillates but goes nowhere — exemption must hold."""
    scorer = _scorer()
    n = 8
    jitter = 0.1 * np.sin(np.arange(n))   # ±0.1 m, zero net displacement
    states = _with_comfort({
        "x": jitter,
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 0.31),   # the measured smoothed-profile value
    }, n)
    # Crossing agent drives THROUGH the stationary ego mid-horizon.
    agents_per_t = [
        [_agent_at(scorer, 1.0, 20.0 - 5.0 * i)] for i in range(n)
    ]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(0.0),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 1.0, (
        "a crossing agent striking a jittering-but-stationary ego was "
        "charged to the ego")


def test_hard_brake_then_struck_is_not_at_fault():
    """A candidate braked to a stop mid-horizon is stopped at collision time.

    The frame-0 actual speed was reused at EVERY timestep, so a candidate
    braking from speed to a standstill could never earn the stopped-ego
    exemption (frame-0 speed frozen, moved_m > 0.5) and was faulted for a
    frontal collision it suffered while parked. Upstream nuPlan judges by
    ego speed at collision time.
    """
    scorer = _scorer()
    n = 8
    # Braking 4 -> 0 m/s: stops at x=5.0 by t=4 and stays.
    states = _with_comfort({
        "x": np.array([0.0, 2.0, 3.5, 4.5, 5.0, 5.0, 5.0, 5.0]),
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 2.0),
    }, n)
    # Agent closes in from far ahead and strikes the parked ego at t=6.
    agents_per_t = [[_agent_at(scorer, 30.0 - 4.0 * i, 0.0)] for i in range(n)]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(4.0),   # frame-0 actual speed: still moving
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 1.0, (
        "hard-brake candidate struck after stopping was charged — the "
        "exemption is reading the frame-0 speed at every timestep again")


def test_creep_contact_inside_half_metre_is_at_fault():
    """The moved_m < 0.5 guard exempted ANY contact within the first 0.5 m.

    A parked ego's creep proposal that reaches a non-touching lead inside
    its first 0.5 m and halts in contact was NEVER at fault at any t —
    nc=1.0 for driving into the car directly ahead. Speed at contact
    decides now.
    """
    scorer = _scorer()
    n = 8
    # Creep 0.4 m/s for two steps (0.4 m total), then hold in contact.
    states = _with_comfort({
        "x": np.array([0.0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 0.4),
    }, n)
    # Lead parked 0.25 m beyond the ego's front bumper: clear at t=0 (not a
    # pre-existing contact), first touched at t=2 while the ego is moving.
    lead_x = VEHICLE_LENGTH / 2 + 2.0 + 0.25
    agents_per_t = [[_agent_at(scorer, lead_x, 0.0)] for _ in range(n)]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(0.0),   # frame-0 actual speed: standstill
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 0.0, (
        "a creep that reaches the lead inside its first 0.5 m and halts in "
        "contact was exempted — the moved_m blind spot is back")


def test_generic_ttc_is_computed_even_after_nc_fails():
    """TTC remains independent of the no-collision score."""
    scorer = _scorer()
    n = 8
    states = _with_comfort({
        "x": np.arange(n, dtype=float) * 2.0,
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 4.0),
    }, n)
    # Contact happens at t=6, outside the projection window at earlier poses.
    # At the contact pose tau=0 must still make TTC fail even though NC already
    # failed; the old generic path skipped all TTC work once nc == 0.
    agents_per_t = [
        [_agent_at(scorer, 14.0, 0.0)] if i >= 6 else [] for i in range(n)
    ]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(4.0),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 0.0
    assert m["ttc"] == 0.0


def test_preexisting_contact_spares_ttc_too():
    """Review finding: excluding pre-existing contacts from nc alone left
    the SAME artifact zeroing the ttc multiplier — the smoothed 0.31 m/s
    profile keeps the ttc loop live for a stationary ego, and the
    projected polygons still hit the overlapping agent. The b040d87a
    frames must keep BOTH terms."""
    scorer = _scorer()
    n = 8
    states = _stationary_states(n)
    # Agent overlapping the ego at t=0, ahead, receding slowly enough to
    # stay inside the projected-polygon sweep for the whole horizon.
    agents_per_t = [[_agent_at(scorer, 1.0 + 0.1 * i, 0.0)] for i in range(n)]
    m = scorer._calculate_metrics(
        states, n - 1, 0,
        ego_state=_ego(0.0),
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
    )
    assert m["nc"] == 1.0
    assert m["ttc"] == 1.0, (
        "pre-existing contact still zeroes ttc — the exclusion only "
        "half-landed (nc spared, ttc charged)")


def test_historical_collision_id_spares_later_nc_and_ttc() -> None:
    """CaRL observation remembers contacts even after overlap has ended."""
    scorer = _scorer()
    n = 8
    states = _with_comfort({
        "x": np.arange(n, dtype=float) * 2.0,
        "y": np.zeros(n),
        "heading": np.zeros(n),
        "speed": np.full(n, 4.0),
    }, n)
    # Clear at the current pose, then the candidate reaches the same actor.
    agents_per_t = [[_agent_at(scorer, 10.0, 0.0, "old")]
                    for _ in range(n)]
    ego = _ego(4.0)
    ego["_pdm_collided_track_ids"] = ("old",)
    metrics = scorer._calculate_metrics(
        states, n - 1, 0, ego_state=ego,
        agents_per_t=agents_per_t,
        red_lanes_per_t=[set() for _ in range(n)],
        carl_collision_classifier=True,
    )
    assert metrics["nc"] == 1.0
    assert metrics["ttc"] == 1.0


# The md-vs-fast window parity harness that used to live here (simplify.md
# consolidation Phase 1) completed its job: the md scorer's window half was
# deleted in Phase 2 and its live half now lives beside the fast engine as
# EPDMSLiveScorer. The fast-side behavior of every parity case remains
# pinned by the fast-only tests above and test_epdms_metric_regressions.py.
