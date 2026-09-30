# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the EPDMS fast-scorer metric fixes.

Pins the scorer-side bug fixes from the PDM-Closed audit:

* prepended ego position comes from ``ego_state``, not the GT log
  (closed-loop drift corrupted every proposal's kinematics),
* TTC is computed from projected-ego polygons (it was only ever zeroed
  together with NC, so it never discriminated),
* DDC is reachable and graded (the aligned-only lane pre-filter made
  wrong-way driving unpenalisable),
* pedestrians participate in NC with type-sized (not car-sized) boxes,
* HC enforces the full nuPlan comfort bounds (signed lon accel etc.).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSTrajectoryScorer_Fast,
)

DT = 0.5  # scorer planner_dt default
T = 8  # 4 s horizon


#: Planner-facing metadata carried alongside the metrics. Not metrics, so
#: the "everything is zeroed" contract below does not apply to them.
_NON_METRIC_KEYS = frozenset({
    "collision_actor_id", "collision_at_fault_ids", "ttc_actor_id",
    "ttc_time_s", "ttc_probe_time_s", "ttc_projection_s",
})


def _scenario(extra_tracks: dict | None = None) -> dict:
    n = 120
    zeros = np.zeros(n)
    xs = np.linspace(-100.0, 300.0, 401)
    tracks = {
        "sdc": {
            "type": "VEHICLE",
            "state": {
                "position": np.zeros((n, 3)),
                "velocity": np.zeros((n, 3)),
                "heading": zeros.copy(),
                "valid": np.ones(n, dtype=bool),
                "length": np.full(n, 4.515),
                "width": np.full(n, 1.852),
            },
        },
    }
    if extra_tracks:
        tracks.update(extra_tracks)
    return {
        "metadata": {"sdc_id": "sdc", "ts": np.arange(n) * 0.1},
        "length": n,
        "tracks": tracks,
        "map_features": {
            "lane_0": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]),
                "speed_limit_mps": 15.0,
            }
        },
        "dynamic_map_states": {},
    }


def _static_track(x: float, y: float, *, track_type: str = "VEHICLE",
                  length: float = 4.5, width: float = 1.8,
                  heading: float = 0.0) -> dict:
    n = 120
    return {
        "type": track_type,
        "state": {
            "position": np.tile(np.array([x, y, 0.0]), (n, 1)),
            "velocity": np.zeros((n, 3)),
            "heading": np.full(n, heading),
            "valid": np.ones(n, dtype=bool),
            "length": np.full(n, length),
            "width": np.full(n, width),
        },
    }


def _scorer(scenario: dict) -> EPDMSTrajectoryScorer_Fast:
    s = EPDMSTrajectoryScorer_Fast(verbose=False)
    s.initialize(scenario, env=None)
    return s


def _ego_state(x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> dict:
    return {
        "position": np.array([x, y, 0.0]),
        "heading": heading,
        "velocity": np.array([5.0, 0.0, 0.0]),
        "acceleration": np.zeros(3),
        "angular_velocity": np.zeros(3),
    }


def _constant_speed_candidate(speed: float = 5.0) -> np.ndarray:
    """(1, T, 3) ego-frame [forward, lateral] samples at ``speed``."""
    cand = np.zeros((1, T, 3))
    cand[0, :, 0] = speed * DT * np.arange(1, T + 1)
    return cand


def test_prepended_point_uses_actual_ego_not_gt_log() -> None:
    """Ego drift from the GT log must not corrupt proposal kinematics.

    The GT sdc sits at x=0; the actual ego has drifted 3 m ahead. The
    proposal is a smooth constant 5 m/s roll-out, so HC must pass.
    Prepending the log position injected a phantom first segment
    (jump = drift) whose finite differences blew the jerk bound for
    every proposal at once.
    """
    scorer = _scorer(_scenario())
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(), _ego_state(x=3.0), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["hc"] == 1.0


def test_ttc_zeroes_for_tailgating_without_collision() -> None:
    """A proposal closing hard on a lead loses TTC while keeping NC."""
    scenario = _scenario({"lead": _static_track(25.0, 0.0)})
    scorer = _scorer(scenario)
    # 5 m/s for 4 s → ego centre ends at x=20: front bumper ≈ 22.26,
    # lead rear bumper ≈ 22.75 — no contact, but the 0.9 s constant-
    # velocity projection reaches ~24.5 and overlaps the lead.
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0, "No actual collision expected"
    assert metrics[0]["ttc"] == 0.0, (
        "Projected-ego TTC must flag tailgating even without contact"
    )


def test_ttc_passes_with_safe_gap() -> None:
    scenario = _scenario({"lead": _static_track(60.0, 0.0)})
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0
    assert metrics[0]["ttc"] == 1.0


def test_ddc_zeroes_for_wrong_way_driving() -> None:
    """Driving against the lane direction must zero DDC.

    The old code pre-filtered candidate lanes to heading-aligned ones,
    which made the |diff| > π/2 violation branch unreachable.
    """
    scorer = _scorer(_scenario())
    cand = np.zeros((1, T, 3))
    # Backwards along the lane at 5 m/s (ego heading flips to ~π).
    cand[0, :, 0] = -5.0 * DT * np.arange(1, T + 1)
    _, metrics = scorer.score_candidates(
        cand, _ego_state(x=50.0), frame_idx=0, return_metrics=True,
    )
    assert metrics[0]["ddc"] == 0.0


def test_ddc_passes_for_forward_driving() -> None:
    scorer = _scorer(_scenario())
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(x=50.0), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["ddc"] == 1.0


def test_pedestrian_collision_zeroes_nc() -> None:
    """Pedestrians must participate in no-collision scoring."""
    scenario = _scenario({
        "ped": _static_track(10.0, 0.0, track_type="PEDESTRIAN",
                             length=0.6, width=0.6),
    })
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_traffic_cone_collision_zeroes_nc() -> None:
    """Static traffic objects are real contact actors, not scenery."""
    scenario = _scenario({
        "cone": _static_track(11.0, 0.0, track_type="TRAFFIC_CONE",
                              length=0.4, width=0.4),
    })
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_logged_other_actor_collision_zeroes_nc() -> None:
    """The verifier must retain generic objects before live state is wired."""
    scenario = _scenario({
        "generic": _static_track(4.8, 0.0, track_type="OTHER",
                                  length=0.325, width=0.323),
    })
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_vehicle_collision_is_still_scored_after_log_end() -> None:
    """Replay holds final poses; post-log verification must do the same."""
    scenario = _scenario({"held": _static_track(11.0, 0.0)})
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=140,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_live_actor_snapshot_overrides_stale_logged_collision() -> None:
    """A semi-reactive actor that moved away must not remain a phantom."""
    scenario = _scenario({"held": _static_track(11.0, 0.0)})
    scorer = _scorer(scenario)
    ego = _ego_state()
    ego["_execution_agent_states"] = [{
        "id": "held", "type": "VEHICLE", "position": [80.0, 0.0, 0.0],
        "velocity": [2.0, 0.0], "heading": 0.0,
        "length": 4.5, "width": 1.8, "valid": True,
    }]
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), ego, frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0


def test_live_actor_snapshot_adds_collision_absent_from_log() -> None:
    """A diverged live actor must be scored at its current, not logged, pose."""
    scorer = _scorer(_scenario())
    ego = _ego_state()
    ego["_execution_agent_states"] = [{
        "id": "live", "type": "VEHICLE", "position": [11.0, 0.0, 0.0],
        "velocity": [0.0, 0.0], "heading": 0.0,
        "length": 4.5, "width": 1.8, "valid": True,
    }]
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), ego, frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_live_other_actor_collision_zeroes_nc() -> None:
    """Unknown semantics cannot hide a physical generic-object collision."""
    scorer = _scorer(_scenario())
    ego = _ego_state()
    ego["_execution_agent_states"] = [{
        "id": "generic", "type": "OTHER", "position": [4.8, 0.0, 0.0],
        "velocity": [0.0, 0.0], "heading": 0.0,
        "length": 0.325, "width": 0.323, "valid": True,
    }]
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), ego, frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_explicit_empty_live_actor_snapshot_is_authoritative() -> None:
    scenario = _scenario({"held": _static_track(11.0, 0.0)})
    scorer = _scorer(scenario)
    ego = _ego_state()
    ego["_execution_agent_states"] = []
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), ego, frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0


def test_pedestrian_at_terminal_pose_zeroes_nc() -> None:
    """The 4.0 s (terminal) pose must be safety-scored (2026-08-18 fix).

    ``score_candidates`` prepends the current pose, so state arrays carry
    ``T + 1`` entries; the safety loops used to run ``range(horizon)`` and
    never saw index ``T`` — a candidate ending ON TOP of a pedestrian kept
    nc = 1.0 while earning full EP credit for the distance to that pose.

    Geometry: constant 5 m/s → terminal ego centre x = 20.0, front bumper
    ≈ 22.26. Pedestrian centred at x = 22.5 (rear edge 22.2): clear of the
    3.5 s pose (front bumper 19.76) by ~2.4 m, overlapped only at 4.0 s.
    """
    scenario = _scenario({
        "ped": _static_track(22.5, 0.0, track_type="PEDESTRIAN",
                             length=0.6, width=0.6),
    })
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0, (
        "collision at the terminal (4.0 s) pose was not charged — the "
        "safety loops stopped at range(horizon) again"
    )


def test_pedestrian_at_penultimate_pose_still_zeroes_nc() -> None:
    """Boundary guard for the 3.5 s pose (last one the OLD loops scored).

    Pedestrian centred at x = 19.9 (rear edge 19.6): overlapped by the
    3.5 s ego front bumper (≈ 19.76), clear of the 3.0 s one (≈ 17.26).
    Must stay charged after the terminal-pose extension.
    """
    scenario = _scenario({
        "ped": _static_track(19.9, 0.0, track_type="PEDESTRIAN",
                             length=0.6, width=0.6),
    })
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 0.0


def test_offroad_terminal_pose_lowers_dac() -> None:
    """The 4.0 s pose must count in DAC's ratio AND denominator (2026-08-18).

    Lane polyline ends at x = 21 (flat-capped strip + 0.3 m seam buffer →
    drivable surface ends at x = 21.3): the terminal ego front bumper
    (≈ 22.26) runs off the end while the 3.5 s pose's (≈ 19.76) is well
    inside. Pre-fix DAC never looked at the terminal pose and scored 1.0;
    now it is one failed pose out of ``horizon + 1 = 9`` scored.
    """
    scenario = _scenario()
    xs = np.linspace(-100.0, 21.0, 122)
    scenario["map_features"]["lane_0"]["polyline"] = np.column_stack(
        [xs, np.zeros_like(xs), np.zeros_like(xs)])
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0
    assert metrics[0]["dac"] == 8 / 9, (
        "terminal pose off the drivable surface must cost exactly one of "
        "the horizon + 1 DAC poses"
    )


def test_roadside_pedestrian_is_not_car_sized() -> None:
    """A pedestrian 2 m off the path must NOT collide.

    Building agent boxes with ego dims turned roadside pedestrians
    into 4.5 x 1.85 m phantom obstacles.
    """
    scenario = _scenario({
        "ped": _static_track(10.0, 2.0, track_type="PEDESTRIAN",
                             length=0.6, width=0.6),
    })
    scorer = _scorer(scenario)
    _, metrics = scorer.score_candidates(
        _constant_speed_candidate(5.0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0


def test_hc_enforces_longitudinal_accel_bound() -> None:
    """Sustained accel above nuPlan's 2.40 m/s² lon bound fails HC.

    3.5 m/s² stays below the 4.89 magnitude bound the old code
    checked, so only the new signed lon-accel bound catches it.
    """
    scorer = _scorer(_scenario())
    cand = np.zeros((1, T, 3))
    t = DT * np.arange(1, T + 1)
    cand[0, :, 0] = 5.0 * t + 0.5 * 3.5 * t**2
    _, metrics = scorer.score_candidates(
        cand, _ego_state(), frame_idx=0, return_metrics=True,
    )
    assert metrics[0]["hc"] == 0.0


def test_contact_with_agent_behind_ego_is_not_at_fault() -> None:
    """Overlap with an agent at/behind the ego centre must not zero NC.

    The at-fault filter exempts ``longitudinal <= 0`` — rear-ends by
    the other agent and pure side contact the ego did not initiate.
    (The old ``< -1.0`` cutoff faulted every sideswipe.)
    """
    scenario = _scenario({
        "tail": _static_track(-3.0, 0.0),  # centre 3 m behind ego
    })
    scorer = _scorer(scenario)
    cand = np.zeros((1, T, 3))
    cand[0, :, 0] = 0.2 * DT * np.arange(1, T + 1)  # slow creep forward
    _, metrics = scorer.score_candidates(
        cand, _ego_state(), frame_idx=0, return_metrics=True,
    )
    assert metrics[0]["nc"] == 1.0, (
        "Contact with an agent behind the ego centre is a rear-end by "
        "the other agent — not at fault."
    )


def test_non_finite_candidate_scores_zero_without_killing_batch() -> None:
    """A NaN row scores 0.0 with zeroed metrics; finite rows must be
    BIT-identical to a batch that never contained the NaN row (a single
    non-finite candidate used to raise a GEOSException and kill the whole
    batch)."""
    good_a = _constant_speed_candidate(5.0)
    good_b = _constant_speed_candidate(4.0)
    bad = _constant_speed_candidate(5.0)
    bad[0, 3, 0] = np.nan
    mixed = np.concatenate([good_a, bad, good_b], axis=0)

    control_scores, control_metrics = _scorer(_scenario()).score_candidates(
        np.concatenate([good_a, good_b], axis=0), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    scores, metrics = _scorer(_scenario()).score_candidates(
        mixed, _ego_state(), frame_idx=0, return_metrics=True,
    )

    assert scores[1] == 0.0
    # Numeric score channels are zeroed; the optional actor identifier is
    # metadata and remains absent for a non-finite candidate.
    assert metrics[1]["collision_actor_id"] is None
    assert metrics[1]["ttc_actor_id"] is None
    assert metrics[1]["ttc_time_s"] is None
    assert all(v == 0.0 for k, v in metrics[1].items()
               if k not in _NON_METRIC_KEYS)
    # Exact (bit) equality against the no-NaN control.
    assert scores[0] == control_scores[0]
    assert scores[2] == control_scores[1]
    assert metrics[0] == control_metrics[0]
    assert metrics[2] == control_metrics[1]


def test_non_finite_batch_logs_single_warning(caplog) -> None:
    scorer = _scorer(_scenario())
    bad = _constant_speed_candidate()
    bad[0, 0, 0] = np.inf
    mixed = np.concatenate([_constant_speed_candidate(), bad], axis=0)
    with caplog.at_level(logging.WARNING):
        scorer.score_candidates(mixed, _ego_state(), frame_idx=0)
    records = [r for r in caplog.records if "non-finite" in r.getMessage()]
    assert len(records) == 1
    assert "1/2" in records[0].getMessage()


def test_all_nan_batch_returns_zeros_without_raising() -> None:
    scorer = _scorer(_scenario())
    bad = np.full((3, T, 3), np.nan)
    scores, metrics = scorer.score_candidates(
        bad, _ego_state(), frame_idx=0, return_metrics=True,
    )
    assert scores.shape == (3,)
    assert np.all(scores == 0.0)
    assert len(metrics) == 3
    assert all(m and m["collision_actor_id"] is None for m in metrics)
    assert all(all(v == 0.0 for k, v in m.items()
                   if k not in _NON_METRIC_KEYS) for m in metrics)


def test_missing_ego_acceleration_uses_zero_prior() -> None:
    """No 'acceleration' key must not KeyError; it scores exactly like an
    explicit zero-acceleration prior (which _ego_state supplies)."""
    control_scores, control_metrics = _scorer(_scenario()).score_candidates(
        _constant_speed_candidate(), _ego_state(), frame_idx=0,
        return_metrics=True,
    )
    ego_no_acc = {k: v for k, v in _ego_state().items()
                  if k != "acceleration"}
    scores, metrics = _scorer(_scenario()).score_candidates(
        _constant_speed_candidate(), ego_no_acc, frame_idx=0,
        return_metrics=True,
    )
    assert scores[0] == control_scores[0]
    assert metrics[0] == control_metrics[0]
