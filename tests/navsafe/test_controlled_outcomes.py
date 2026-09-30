# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""controlled_outcomes: the per-episode hazard description used by 4.3.2."""

from __future__ import annotations

import math

import pytest

from navsafe.benchmark.scoring.controlled import (
    BRAKE_ACCEL_MS2,
    STOPPED_SPEED_MS,
    controlled_outcomes,
)

DT = 0.1


def frame(i, *, speed=5.0, accel=0.0, agents=(), min_clear=None, x=0.0, y=0.0):
    return {
        "frame": i, "ego_speed": speed, "ego_accel": accel,
        "ego_x": x, "ego_y": y,
        "min_clearance_m": float("nan") if min_clear is None else min_clear,
        "agents": list(agents),
    }


def agent(aid, *, ttc=float("inf"), clear=99.0, cls="PEDESTRIAN"):
    return {"id": aid, "cls": cls, "ttc_s": ttc, "clearance_m": clear}


def test_empty_trace_says_so_instead_of_inventing_zeros():
    out = controlled_outcomes([], dt=DT)
    assert out == {"scored_frames": 0}


def test_warmup_frames_are_excluded():
    # A hard brake during warm-up is the replayed log, not the policy.
    frames = [frame(i, accel=-5.0, speed=0.0) for i in range(5)]
    frames += [frame(i, accel=0.0, speed=5.0) for i in range(5, 10)]
    out = controlled_outcomes(frames, dt=DT, scored_from=5)
    assert out["scored_frames"] == 5
    assert out["braked"] is False
    assert out["braking_onset_frame"] is None
    assert out["stopped_frames"] == 0


def test_braking_onset_is_the_first_frame_past_the_threshold():
    frames = [
        frame(0, accel=-0.2),                    # coasting, not braking
        frame(1, accel=BRAKE_ACCEL_MS2 + 0.01),  # just under
        frame(2, accel=BRAKE_ACCEL_MS2),         # exactly at -> counts
        frame(3, accel=-4.0),
    ]
    out = controlled_outcomes(frames, dt=DT)
    assert out["braked"] is True
    assert out["braking_onset_frame"] == 2
    assert out["braking_onset_s"] == pytest.approx(0.2)
    assert out["min_accel_ms2"] == pytest.approx(-4.0)


def test_stopped_duration_totals_and_longest_run_differ():
    # stop, move, stop again for longer: total 5, longest 3
    speeds = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 5.0]
    frames = [frame(i, speed=s) for i, s in enumerate(speeds)]
    out = controlled_outcomes(frames, dt=DT)
    assert out["stopped_frames"] == 5
    assert out["stopped_duration_s"] == pytest.approx(0.5)
    assert out["longest_stop_s"] == pytest.approx(0.3)
    assert out["min_speed_ms"] == pytest.approx(0.0)


def test_residual_creep_still_counts_as_stopped():
    frames = [frame(i, speed=STOPPED_SPEED_MS) for i in range(3)]
    out = controlled_outcomes(frames, dt=DT)
    assert out["stopped_frames"] == 3


def test_ttc_and_clearance_split_target_from_everything_else():
    frames = [
        frame(0, agents=[agent("hazard", ttc=9.0, clear=8.0),
                         agent("other", ttc=2.0, clear=1.0)]),
        frame(1, agents=[agent("hazard", ttc=3.0, clear=0.4),
                         agent("other", ttc=5.0, clear=6.0)]),
    ]
    out = controlled_outcomes(frames, dt=DT, target_ids=["hazard"])
    # the closest thing overall was `other`, but the TARGET minima are the
    # hazard's own -- mixing them would hide the manipulation.
    assert out["min_ttc_s"] == pytest.approx(2.0)
    assert out["min_ttc_to_target_s"] == pytest.approx(3.0)
    assert out["min_clearance_to_target_m"] == pytest.approx(0.4)
    assert out["target_frames_seen"] == 2
    assert out["target_ids"] == ["hazard"]


def test_absent_target_reports_none_not_zero():
    # An insert that silently did not render must not read as "clearance 0".
    frames = [frame(0, agents=[agent("other", ttc=1.0, clear=0.5)])]
    out = controlled_outcomes(frames, dt=DT, target_ids=["hazard"])
    assert out["min_clearance_to_target_m"] is None
    assert out["min_ttc_to_target_s"] is None
    assert out["target_frames_seen"] is None


def test_infinite_ttc_is_dropped_not_treated_as_a_number():
    frames = [frame(0, agents=[agent("hazard", ttc=float("inf"))])]
    out = controlled_outcomes(frames, dt=DT, target_ids=["hazard"])
    assert out["min_ttc_to_target_s"] is None
    assert out["min_ttc_s"] is None


def test_conflict_zone_passed_when_the_ego_gets_within_the_radius():
    frames = [frame(0, x=0.0, y=0.0), frame(1, x=10.0, y=0.0),
              frame(2, x=20.0, y=0.0)]
    near = controlled_outcomes(frames, dt=DT, conflict_xy=(10.5, 0.0))
    assert near["passed_conflict_zone"] is True
    assert near["closest_approach_to_conflict_m"] == pytest.approx(0.5)
    far = controlled_outcomes(frames, dt=DT, conflict_xy=(60.0, 0.0))
    assert far["passed_conflict_zone"] is False
    assert far["closest_approach_to_conflict_m"] == pytest.approx(40.0)


def test_no_conflict_point_leaves_the_progress_fields_unset():
    out = controlled_outcomes([frame(0)], dt=DT)
    assert out["passed_conflict_zone"] is None
    assert out["conflict_xy"] is None


def test_nan_and_missing_values_never_become_numbers():
    frames = [
        {"frame": 0, "ego_speed": None, "ego_accel": "x", "agents": [{"id": "h"}]},
        frame(1, speed=float("nan"), accel=float("nan")),
    ]
    out = controlled_outcomes(frames, dt=DT, target_ids=["h"])
    assert out["min_accel_ms2"] is None
    assert out["min_ttc_to_target_s"] is None
    assert out["min_clearance_to_target_m"] is None
    # the agent row WAS seen, even though it carried no usable numbers
    assert out["target_frames_seen"] == 1


def test_output_is_json_serialisable():
    import json
    frames = [frame(0, agents=[agent("h", ttc=2.0, clear=1.0)])]
    out = controlled_outcomes(frames, dt=DT, target_ids=["h"], conflict_xy=(1.0, 2.0))
    json.dumps(out)  # must not raise
