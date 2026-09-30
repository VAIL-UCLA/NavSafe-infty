"""Rebuilding trace-shaped frames from evaluator artifacts
(navsafe/benchmark/trace/from_eval.py)."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark import termination as T
from navsafe.benchmark.trace import from_eval


def _route(n=60):
    return np.stack([np.arange(n) * 1.0, np.zeros(n)], axis=1)


def _agent(aid="a1", n=60, x0=30.0, y0=0.0, dx=0.0, parked=False):
    xy = np.stack([x0 + np.arange(n) * dx, np.full(n, y0)], axis=1)
    return {"id": aid, "cls": "vehicle", "xy": xy,
            "valid": np.ones(n, dtype=bool), "heading": np.zeros(n),
            "length": 4.5, "width": 1.9, "parked": parked}


def test_phase_split_and_schema_completeness():
    route = _route()
    frames = from_eval.build_frames(route, route, [], {}, warmup_frames=10,
                                    dt=0.1, drivable_known=False)
    assert len(frames) == len(route)
    assert [f["phase"] for f in frames[:10]] == ["warmup"] * 10
    assert all(f["phase"] == "scored" for f in frames[10:])
    # Every schema column present, so a partial writer cannot shift the schema.
    from navsafe.benchmark.trace.schema import FRAME_SCHEMA
    for col in FRAME_SCHEMA.names:
        assert col in frames[0]


def test_deviation_from_the_logged_path_is_recorded_but_ends_nothing():
    """The render-validity envelope is gone: distance from the logged path is a
    measurement, not a verdict. An ego 20 m off the log on drivable road is a
    policy that drove somewhere else, which is scored, not excluded."""
    route = _route()
    ego = route.copy()
    ego[40:, 1] = 20.0
    frames = from_eval.build_frames(ego, route, [], {}, warmup_frames=5,
                                    dt=0.1, drivable_known=False)
    assert all("render_valid" not in f for f in frames)
    assert abs(frames[50]["ego_dev_lat_m"]) == pytest.approx(20.0, abs=0.5)
    t = T.classify(frames, goal_reached=False, t_max_s=10.0)
    assert t.reason is not T.TerminationReason.ENVELOPE_EXIT
    assert t.reason.policy_attributed or t.reason.is_no_event_fallback


def test_ddc_zero_ends_the_episode_as_the_egos_fault():
    """DDC 0.0 is >6 m driven against the local traffic direction."""
    route = _route()
    per_frame = {i: {"DAC": 1.0, "NC": 1.0,
                     "DDC": 0.0 if i >= 30 else 1.0} for i in range(len(route))}
    frames = from_eval.build_frames(route, route, [], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert all(f["driving_direction_ok"] for f in frames[:30])
    assert not any(f["driving_direction_ok"] for f in frames[30:])
    t = T.classify(frames, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.WRONG_WAY
    assert t.frame == 30
    assert t.reason.policy_attributed


def test_tl_hold_column_exempts_the_deadlock_rule():
    """TL_HOLD is the evaluator's persisted "red light ahead" state fact; a
    stored run re-scored post hoc must apply the same deadlock exemption the
    live monitor did, or the two verdicts disagree."""
    route = _route(120)
    ego = route.copy()
    ego[20:, :] = ego[20]                       # parked from frame 20 on
    per_frame = {i: {"DAC": 1.0, "NC": 1.0, "TL": 1.0, "TL_HOLD": 1.0}
                 for i in range(len(route))}
    frames = from_eval.build_frames(ego, route, [], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert all(f["signal_hold"] for f in frames)
    t = T.classify(frames, goal_reached=False, t_max_s=10.0)
    assert t.reason is not T.TerminationReason.DEADLOCK
    # Without the column: the plain rule, as before.
    per_frame = {i: {"DAC": 1.0, "NC": 1.0, "TL": 1.0} for i in range(len(route))}
    frames = from_eval.build_frames(ego, route, [], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert not any(f["signal_hold"] for f in frames)
    t = T.classify(frames, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.DEADLOCK


def test_half_ddc_is_not_a_termination():
    """0.5 is 2-6 m against the direction -- a brush, not a committed wrong-way
    run. Ending the episode there would kill ordinary wide turns."""
    route = _route()
    per_frame = {i: {"DAC": 1.0, "NC": 1.0, "DDC": 0.5} for i in range(len(route))}
    frames = from_eval.build_frames(route, route, [], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert all(f["driving_direction_ok"] for f in frames)
    assert T.classify(frames, goal_reached=False, t_max_s=10.0).reason \
        is not T.TerminationReason.WRONG_WAY


def test_missing_map_does_not_read_as_wrong_way():
    """No lane graph -> the scorer reports DDC 0 for want of anything to
    measure. Terminating there would fail every policy on every mapless
    reconstruction, exactly as the DAC case would."""
    route = _route()
    per_frame = {i: {"DAC": 0.0, "DDC": 0.0, "NC": 1.0} for i in range(len(route))}
    frames = from_eval.build_frames(route, route, [], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=False)
    assert all(f["driving_direction_ok"] for f in frames)
    assert T.classify(frames, goal_reached=False, t_max_s=10.0).reason \
        is not T.TerminationReason.WRONG_WAY


def test_missing_map_does_not_read_as_off_drivable():
    """DAC is 0 on a scenario with no map because there is nothing to test
    against; reading that as 'left the road' would fail every policy."""
    route = _route()
    per_frame = {i: {"DAC": 0.0, "NC": 1.0} for i in range(len(route))}
    frames = from_eval.build_frames(route, route, [], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=False)
    assert all(f["on_drivable"] for f in frames)
    assert T.classify(frames, goal_reached=False, t_max_s=10.0).reason \
        is not T.TerminationReason.OFF_DRIVABLE

    known = from_eval.build_frames(route, route, [], per_frame,
                                   warmup_frames=5, dt=0.1,
                                   drivable_known=True)
    assert not any(f["on_drivable"] for f in known)
    assert T.classify(known, goal_reached=False, t_max_s=10.0).reason \
        is T.TerminationReason.OFF_DRIVABLE


def test_contacts_come_from_the_per_frame_at_fault_column():
    route = _route()
    per_frame = {i: {"NC": 1.0, "DAC": 1.0} for i in range(len(route))}
    per_frame[30]["NC"] = 0.0
    frames = from_eval.build_frames(route, route, [_agent()], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert [i for i, f in enumerate(frames) if f["contacts"]] == [30]
    c = frames[30]["contacts"][0]
    assert c["at_fault"] and c["agent_id"] == "a1"
    t = T.classify(frames, goal_reached=False, t_max_s=10.0)
    assert t.reason is T.TerminationReason.CONTACT_AT_FAULT and t.frame == 30

    # warm-up frames are replayed ground truth and never carry a contact
    per_frame[2] = {"NC": 0.0, "DAC": 1.0}
    frames = from_eval.build_frames(route, route, [_agent()], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert not frames[2]["contacts"]


def test_parked_vehicles_are_excluded_from_the_efficiency_background():
    from navsafe.benchmark.scoring import metrics as navsafe_metrics

    route = _route()
    moving = _agent("moving", x0=20.0, dx=0.5)      # 5 m/s
    parked = _agent("parked", x0=10.0, y0=6.0, dx=0.0, parked=True)
    frames = from_eval.build_frames(route, route, [moving, parked], {},
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=False)
    e = navsafe_metrics.efficiency_inputs_from_trace(frames)
    bg = e["background_mean_speed"]
    assert np.isfinite(bg).any()
    # Only the moving vehicle counts: the parked one would halve the mean.
    assert np.nanmax(bg) == pytest.approx(5.0, abs=0.2)


def _sd(tracks, ego_id="ego", n=60):
    """A minimal ScenarioDescription for :func:`_agent_tracks`."""
    out = {"tracks": {ego_id: {"type": "VEHICLE", "state": {
        "position": np.stack([np.arange(n) * 1.0, np.zeros(n), np.zeros(n)], axis=1)}}}}
    for aid, xy in tracks.items():
        out["tracks"][aid] = {"type": "VEHICLE", "state": {
            "position": np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)}}
    return out


def test_crawling_vehicles_count_as_parked():
    """A queued car that inches past the displacement bar is still not driving.

    Measured on navhard421/028613e11f415422: 29 of the 35 vehicles that passed
    the 5 m net-displacement filter never exceeded 1 m/s. They dragged the
    background mean down, and since Efficiency is a per-checkpoint ego/bg
    ratio, that put it in the hundreds of percent.
    """
    n = 201                                   # 20 s at 10 Hz
    crawl = np.stack([np.arange(n) * 0.03, np.zeros(n)], axis=1)   # 0.3 m/s, 6 m
    drive = np.stack([np.arange(n) * 0.50, np.zeros(n)], axis=1)   # 5 m/s
    tracks = from_eval._agent_tracks(_sd({"crawl": crawl, "drive": drive}), "ego", dt=0.1)
    by = {t["id"]: t for t in tracks}
    # The crawler moved 6 m -- past PARKED_NET_DISPLACEMENT_M -- so displacement
    # alone kept it. Peak speed is what excludes it.
    assert by["crawl"]["peak_speed"] == pytest.approx(0.3, abs=0.05)
    assert by["crawl"]["parked"] is True
    assert by["drive"]["parked"] is False


def test_a_car_that_drives_then_stops_stays_in_the_background():
    """Only never-moved vehicles are dropped; stopping is real traffic."""
    n = 201
    xy = np.stack([np.concatenate([np.arange(60) * 0.5,
                                   np.full(n - 60, 59 * 0.5)]), np.zeros(n)], axis=1)
    tracks = from_eval._agent_tracks(_sd({"stops": xy}), "ego", dt=0.1)
    assert tracks[0]["parked"] is False


def test_kinematics_are_finite_and_start_at_rest_free():
    route = _route()
    frames = from_eval.build_frames(route, route, [], {}, warmup_frames=0,
                                    dt=0.1, drivable_known=False)
    speeds = np.array([f["ego_speed"] for f in frames])
    assert np.isfinite(speeds).all()
    assert speeds.mean() == pytest.approx(10.0, abs=0.5)   # 1 m per 0.1 s


def test_not_at_fault_contact_is_recoverable_from_coll_columns():
    """COLL/COLL_AF carry fault; EPDMS's NC alone cannot express a contact the
    ego did not cause, so that termination state was unreachable before."""
    route = _route()
    per_frame = {i: {"NC": 1.0, "DAC": 1.0, "COLL": 0.0, "COLL_AF": 0.0}
                 for i in range(len(route))}
    per_frame[30] = {"NC": 1.0, "DAC": 1.0, "COLL": 1.0, "COLL_AF": 0.0}
    frames = from_eval.build_frames(route, route, [_agent()], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    c = frames[30]["contacts"][0]
    assert c["at_fault"] is False
    # Strict protocol (2026-09-01): the exonerated contact ends the episode
    # without being charged to the policy.
    t = T.classify(frames, goal_reached=False, t_max_s=1.0)
    assert t.reason is T.TerminationReason.CONTACT_NOT_AT_FAULT
    assert not t.reason.policy_attributed


def test_at_fault_contact_from_coll_columns():
    route = _route()
    per_frame = {i: {"NC": 1.0, "DAC": 1.0, "COLL": 0.0, "COLL_AF": 0.0}
                 for i in range(len(route))}
    per_frame[30] = {"NC": 0.0, "DAC": 1.0, "COLL": 1.0, "COLL_AF": 1.0}
    frames = from_eval.build_frames(route, route, [_agent()], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert frames[30]["contacts"][0]["at_fault"] is True
    assert T.classify(frames, goal_reached=False, t_max_s=10.0).reason \
        is T.TerminationReason.CONTACT_AT_FAULT


def test_legacy_runs_without_coll_columns_still_read_nc():
    route = _route()
    per_frame = {i: {"NC": 1.0, "DAC": 1.0} for i in range(len(route))}
    per_frame[30]["NC"] = 0.0
    frames = from_eval.build_frames(route, route, [_agent()], per_frame,
                                    warmup_frames=5, dt=0.1,
                                    drivable_known=True)
    assert frames[30]["contacts"][0]["at_fault"] is True
