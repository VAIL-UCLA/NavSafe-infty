"""Regression tests for the Bench2Drive metric port.

The comfort path was validated against the reference implementation
(``tools/efficiency_smoothness_benchmark.py``) directly: 200 randomized
episodes, ``b2d_compat=True`` vs ``seg_compute_comfort_metric``, 200/200
exact matches.  These tests pin that behaviour without importing the
reference, plus hand-computed values for every closed-form formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.scoring import metrics as navsafe_metrics
from navsafe.benchmark.trace.schema import PHASE_SCORED, PHASE_WARMUP, empty_frame


# --- Driving Score ---------------------------------------------------------

def test_penalty_coefficients_are_leaderboard_20():
    assert navsafe_metrics.PENALTY_COEFFICIENTS == {
        "collisions_pedestrian": 0.50,
        "collisions_vehicle": 0.60,
        "collisions_layout": 0.65,
        "red_light": 0.70,
        "scenario_timeouts": 0.70,
        "yield_emergency_vehicle_infractions": 0.70,
        "stop_infraction": 0.80,
    }


def test_driving_score_multiplies_per_occurrence():
    r = navsafe_metrics.RouteResult("r", 80.0, False,
                        {"collisions_vehicle": 2, "red_light": 1,
                         "outside_route_lanes": 10.0})
    assert r.driving_score() == pytest.approx(80.0 * 0.6 ** 2 * 0.7 * 0.9)


def test_offroad_percentage_is_proportional_not_a_coefficient():
    # penalty_value = 0 in PENALTY_PERC_DICT means the FULL (1 - pct/100)
    # applies -- the reference's "ignored" comment is wrong about its own code.
    r = navsafe_metrics.RouteResult("r", 100.0, True, {"outside_route_lanes": 25.0})
    assert r.driving_score() == pytest.approx(75.0)


def test_benchmark_ds_uses_declared_denominator():
    routes = [navsafe_metrics.RouteResult("a", 100.0, True, {}),
              navsafe_metrics.RouteResult("b", 50.0, False, {})]
    # merge_route_json.py divides by 220 regardless of how many routes ran;
    # a missing episode therefore counts as 0.
    assert navsafe_metrics.driving_score(routes, denominator=4) == pytest.approx(150.0 / 4)
    assert navsafe_metrics.driving_score(routes) == pytest.approx(75.0)
    with pytest.raises(ValueError):
        navsafe_metrics.driving_score(routes, denominator=1)


# --- Success Rate ----------------------------------------------------------

def test_success_exempts_only_min_speed():
    ok = navsafe_metrics.RouteResult("r", 100.0, True, {"min_speed_infractions": 7})
    assert ok.success()
    for key in ("collisions_vehicle", "red_light", "stop_infraction",
                "route_timeout", "route_dev", "vehicle_blocked",
                "scenario_timeouts"):
        assert not navsafe_metrics.RouteResult("r", 100.0, True, {key: 1}).success(), key


def test_success_requires_completion():
    assert not navsafe_metrics.RouteResult("r", 99.0, False, {}).success()


def test_success_rate_denominator():
    routes = [navsafe_metrics.RouteResult("a", 100.0, True, {}),
              navsafe_metrics.RouteResult("b", 100.0, True, {"red_light": 1})]
    assert navsafe_metrics.success_rate(routes, denominator=4) == pytest.approx(0.25)


# --- Efficiency ------------------------------------------------------------

def _pct(n):
    """Route completion per frame for a run that drives the whole route.

    route_efficiency bins by route position (0-100 %), not by odometer: the
    checkpoints are every 5 % of the ROUTE, so a test that means "a full route
    at a steady speed" has to say so on that axis.
    """
    return np.linspace(0.0, 100.0, n)


def test_efficiency_is_speed_ratio_percent():
    n = 400
    v = navsafe_metrics.route_efficiency(np.full(n, 5.0), np.full(n, 10.0), _pct(n))
    assert v == pytest.approx(50.0)


def test_efficiency_can_exceed_100():
    n = 400
    v = navsafe_metrics.route_efficiency(np.full(n, 12.0), np.full(n, 10.0), _pct(n))
    assert v == pytest.approx(120.0)


def test_efficiency_drops_outlier_checkpoints():
    n = 400
    bg = np.full(n, 10.0)
    bg[:20] = 1e-4                      # ratio > 1000 % in the first checkpoint
    v = navsafe_metrics.route_efficiency(np.full(n, 5.0), bg, _pct(n))
    assert v == pytest.approx(50.0)


def test_efficiency_none_without_background():
    n = 400
    assert navsafe_metrics.route_efficiency(np.full(n, 5.0), np.full(n, np.nan), _pct(n)) is None
    assert navsafe_metrics.efficiency([None, None]) is None
    assert navsafe_metrics.efficiency([50.0, None, 100.0]) == pytest.approx(75.0)


# --- Comfort ---------------------------------------------------------------

def _calm(n):
    z = np.zeros(n)
    return z, z, z, z


def test_comfort_calm_episode_passes():
    assert navsafe_metrics.route_comfort(*_calm(100)) == 1.0


def test_comfort_hard_brake_fails_lon_accel():
    n = 100
    lon = np.zeros(n)
    lon[40:60] = -5.0                  # beyond the -4.05 bound
    lat, mag, yr = np.zeros(n), np.abs(lon), np.zeros(n)
    score = navsafe_metrics.route_comfort(lon, lat, mag, yr)
    # 5 segments of 20 frames; the brake exactly fills segment 2 (frames
    # 40-59), and segmentation hides the jerk at both boundaries -- only that
    # one segment fails.
    assert score == pytest.approx(4 / 5)


def test_comfort_score_is_segment_fraction():
    n = 60                              # 3 segments
    lon = np.zeros(n)
    lon[0:20] = 3.0                     # segment 0 violates +2.40
    score = navsafe_metrics.route_comfort(lon, np.zeros(n), np.abs(lon), np.zeros(n))
    assert score == pytest.approx(2 / 3)


def test_comfort_short_episode_is_binary():
    assert navsafe_metrics.route_comfort(*_calm(10)) == 1.0


def test_comfort_fixed_mode_catches_yaw_spin_compat_does_not():
    # |yaw rate| < 0.95 rad/s throughout, but its derivative peaks at
    # 2.7 rad/s^2 > 1.93.  The reference's yaw-acceleration channel is the
    # smoothed yaw rate (no derivative), so compat mode cannot see it.
    t = np.arange(40) * navsafe_metrics.COMFORT_DT
    yr = 0.9 * np.sin(3.0 * t)
    z = np.zeros(40)
    assert navsafe_metrics.route_comfort(z, z, z, yr, b2d_compat=False) == 0.0
    assert navsafe_metrics.route_comfort(z, z, z, yr, b2d_compat=True) == 1.0


def test_comfort_bounds_are_strict():
    # A signal exactly at the bound fails (reference uses > and <).
    n = 20
    lon = np.full(n, 2.40)
    assert navsafe_metrics.route_comfort(lon, np.zeros(n), lon, np.zeros(n)) == 0.0


# --- Trace adapters --------------------------------------------------------

def _trace(n_warm=5, n_scored=45, speed=5.0, bg_speed=10.0):
    frames = []
    for i in range(n_warm + n_scored):
        f = empty_frame()
        f["frame"] = i
        f["phase"] = PHASE_WARMUP if i < n_warm else PHASE_SCORED
        f["ego_x"], f["ego_y"] = i * speed * 0.1, 0.0
        f["ego_yaw"] = 0.0
        f["ego_speed"] = speed
        f["ego_accel"] = 0.0
        f["ego_lat_accel"] = 0.0
        f["agents"] = [{"id": "a1", "cls": "vehicle", "policy": "idm",
                        "speed": bg_speed},
                       {"id": "p1", "cls": "pedestrian", "policy": "replay",
                        "speed": 1.0}]
        f["contacts"] = []
        frames.append(f)
    return frames


def test_comfort_inputs_use_scored_phase_only():
    sig = navsafe_metrics.comfort_inputs_from_trace(_trace())
    assert len(sig["lon_accel"]) == 45
    assert navsafe_metrics.route_comfort(**sig) == 1.0


def test_efficiency_inputs_filter_to_vehicles():
    inp = navsafe_metrics.efficiency_inputs_from_trace(_trace())
    # pedestrian excluded: background mean must be the idm vehicle's speed
    assert np.allclose(inp["background_mean_speed"], 10.0)
    v = navsafe_metrics.route_efficiency(
        inp["ego_speed"], inp["background_mean_speed"], _pct(len(inp["ego_speed"])))
    assert v == pytest.approx(50.0)


def test_route_result_from_termination_mapping():
    from navsafe.scenario.constants import TerminationState as TS

    ok = navsafe_metrics.route_result_from_termination("r", 100.0, {TS.SUCCESS: True})
    assert ok.completed and ok.success() and ok.driving_score() == 100.0

    idle = navsafe_metrics.route_result_from_termination("r", 40.0, {TS.IDLE: True})
    assert idle.infractions == {"vehicle_blocked": 1}
    assert not idle.success()
    # blocked is a terminator, not a penalty: DS is completion untouched
    assert idle.driving_score() == pytest.approx(40.0)

    oor = navsafe_metrics.route_result_from_termination("r", 60.0, {TS.OUT_OF_ROAD: True})
    assert oor.infractions == {"route_dev": 1}

    to = navsafe_metrics.route_result_from_termination("r", 70.0, {}, truncated=True)
    assert to.infractions == {"route_timeout": 1}
    # route_timeout fails SR but is NOT a DS coefficient in Bench2Drive
    assert to.driving_score() == pytest.approx(70.0)

    crash = navsafe_metrics.route_result_from_termination(
        "r", 55.0, {TS.CRASH_VEHICLE: True, TS.CRASH_OBJECT: True})
    assert crash.infractions == {"collisions_vehicle": 1,
                                 "collisions_layout": 1}
    assert crash.driving_score() == pytest.approx(55.0 * 0.6 * 0.65)

    # success beats a stale MAX_STEP flag; rubric-layer extras merge in
    both = navsafe_metrics.route_result_from_termination(
        "r", 100.0, {TS.SUCCESS: True, TS.MAX_STEP: True},
        extra_infractions={"red_light": 1})
    assert both.completed and both.infractions == {"red_light": 1}
    assert both.driving_score() == pytest.approx(70.0)


def test_infractions_from_trace_maps_kind_and_fault():
    frames = _trace()
    hit = {"agent_id": "a1", "at_fault": True, "kind": "rear_end",
           "rel_speed": 3.0}
    not_fault = {"agent_id": "a2", "at_fault": False, "kind": "angle",
                 "rel_speed": 1.0}
    vru = {"agent_id": "p1", "at_fault": True, "kind": "vru", "rel_speed": 2.0}
    # same contact pair persisting over frames counts once
    frames[10]["contacts"] = [hit, not_fault]
    frames[11]["contacts"] = [hit]
    frames[20]["contacts"] = [vru]
    counts = navsafe_metrics.infractions_from_trace(frames)
    assert counts == {"collisions_vehicle": 1, "collisions_pedestrian": 1}
    both = navsafe_metrics.infractions_from_trace(frames, at_fault_only=False)
    assert both["collisions_vehicle"] == 2


def test_efficiency_excluded_when_the_background_never_moves():
    """A queued street has no denominator, so the route reports nothing.

    Measured on navhard421/0fde069313a35062: 9-14 background vehicles with a
    median speed of 0.41 m/s put all 19 reached checkpoints between 769 % and
    2393 %, for a route value of 948 %. The inflation is uniform, so B2D's
    1000 % outlier drop never fires. That is an absent denominator, not a fast
    policy.
    """
    n = 200
    ego = np.full(n, 5.4)                       # the ego is moving
    bg = np.full(n, 0.41)                       # the street is not
    pct = np.linspace(0.0, 99.0, n)
    assert navsafe_metrics.route_efficiency(ego, bg, pct) is None


def test_efficiency_reported_when_the_background_does_move():
    n = 200
    ego = np.full(n, 5.0)
    bg = np.full(n, 5.0)
    pct = np.linspace(0.0, 99.0, n)
    assert navsafe_metrics.route_efficiency(ego, bg, pct) == pytest.approx(100.0)


def test_one_moving_car_cannot_carry_a_stopped_street():
    """The gate is a median, so a single passing vehicle is not a background."""
    n = 200
    ego = np.full(n, 5.4)
    bg = np.full(n, 0.3)
    bg[:20] = 12.0                              # one car sweeps past early
    pct = np.linspace(0.0, 99.0, n)
    assert navsafe_metrics.route_efficiency(ego, bg, pct) is None
