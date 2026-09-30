# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Every lane must carry a speed target, and it must say where it came from.

Argoverse 2 posts no speed limit on any lane (measured: 0 of 169/130/141/146/
124/69/101/120 lane features across the eight repair-plan scenes). Before
:mod:`navsafe.scenario.speed_target` existed, ``IDMPolicy.target_velocity_for``
therefore returned ``fraction x 15 m/s`` on every lane of every scene — a
residential street and a downtown junction got the same target, and the only
thing holding the teacher below a context-free 15 m/s was the plan-execution
pacing gain. These tests pin the replacement: what the target is derived from,
which rule produced it, and that the record says so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navsafe.scenario.speed_target import (
    PROVENANCE_KEY,
    SPEED_LIMIT_KEY,
    SPEED_LIMIT_SOURCE_KEY,
    SpeedTargetParams,
    annotate_speed_targets,
    speed_target_mode,
)

DT = 0.1


def _lane(x0=0.0, y0=0.0, length=60.0, heading=0.0, **extra):
    s = np.arange(0.0, length + 0.5, 0.5)
    poly = np.column_stack(
        [x0 + s * math.cos(heading), y0 + s * math.sin(heading)])
    return {"type": "LANE_SURFACE_STREET", "polyline": poly, **extra}


def _track(speeds, *, y=0.0, heading=0.0, x0=0.0, obj_type="VEHICLE"):
    """A vehicle driving along y=const at the given per-frame speeds."""
    speeds = np.asarray(speeds, dtype=np.float64)
    n = len(speeds)
    x = x0 + np.concatenate([[0.0], np.cumsum(speeds[:-1] * DT)])
    pos = np.column_stack([
        x * math.cos(heading) - y * math.sin(heading),
        x * math.sin(heading) + y * math.cos(heading),
        np.zeros(n)])
    return {
        "type": obj_type,
        "state": {
            "position": pos,
            "heading": np.full(n, heading),
            "velocity": np.column_stack(
                [speeds * math.cos(heading), speeds * math.sin(heading)]),
            "valid": np.ones(n, dtype=bool),
        },
    }


def _sd(lanes, tracks, sdc_id="ego"):
    return {"map_features": dict(lanes), "tracks": dict(tracks),
            "metadata": {"sdc_id": sdc_id}}


# ── the estimator ─────────────────────────────────────────────────────────


class TestObservedTarget:
    def test_lane_target_follows_the_traffic_that_used_it(self):
        sd = _sd({"L": _lane()}, {"v1": _track([10.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert sd["map_features"]["L"][SPEED_LIMIT_KEY] == pytest.approx(
            10.0, rel=1e-3)
        assert sd["map_features"]["L"][SPEED_LIMIT_SOURCE_KEY] == "observed"
        assert prov["counts"]["observed"] == 1

    def test_a_slow_lane_and_a_fast_lane_get_different_targets(self):
        # The whole point: one number per scene cannot be right for both.
        sd = _sd({"slow": _lane(y0=0.0), "fast": _lane(y0=50.0)},
                 {"a": _track([4.0] * 40, y=0.0),
                  "b": _track([13.0] * 40, y=50.0)})
        annotate_speed_targets(sd)
        assert sd["map_features"]["slow"][SPEED_LIMIT_KEY] == pytest.approx(
            4.0, rel=1e-3)
        assert sd["map_features"]["fast"][SPEED_LIMIT_KEY] == pytest.approx(
            13.0, rel=1e-3)

    def test_one_idling_vehicle_cannot_outvote_the_traffic_by_sitting_there(
            self):
        # Aggregating raw samples would let a bus that crept for 200 frames
        # define the street. The study aggregates per vehicle first, so the
        # crawler is ONE opinion among two, not 200.
        crawler = _track([1.0] * 200, y=0.0)
        driver = _track([12.0] * 20, y=0.5, x0=5.0)
        sd = _sd({"L": _lane()}, {"slow": crawler, "fast": driver})
        annotate_speed_targets(sd)
        target = sd["map_features"]["L"][SPEED_LIMIT_KEY]
        assert target > 6.0, (
            f"target {target:.2f} was dragged down by sample count, not by "
            f"vehicle count")

    def test_stopped_samples_are_not_evidence_about_the_road(self):
        # A queue says what the traffic is doing, not what the road allows.
        moving_only = _sd({"L": _lane()}, {"v": _track([8.0] * 30)})
        with_stop = _sd({"L": _lane()},
                        {"v": _track([0.0] * 100 + [8.0] * 30)})
        annotate_speed_targets(moving_only)
        annotate_speed_targets(with_stop)
        assert (with_stop["map_features"]["L"][SPEED_LIMIT_KEY]
                == pytest.approx(
                    moving_only["map_features"]["L"][SPEED_LIMIT_KEY],
                    rel=1e-6))

    def test_oncoming_traffic_is_not_attributed_to_this_lane(self):
        # Opposing centerlines sit metres apart on a divided road; without a
        # heading test the fast oncoming carriageway sets this lane's target.
        sd = _sd({"L": _lane()},
                 {"onc": _track([15.0] * 40, heading=math.pi)})
        prov = annotate_speed_targets(sd)
        assert prov["counts"]["observed"] == 0
        assert SPEED_LIMIT_KEY not in sd["map_features"]["L"]

    def test_traffic_on_a_parallel_street_is_not_attributed(self):
        sd = _sd({"L": _lane()}, {"far": _track([15.0] * 40, y=8.0)})
        prov = annotate_speed_targets(sd)
        assert prov["counts"]["observed"] == 0

    def test_a_single_frame_is_a_coordinate_not_a_speed_study(self):
        sd = _sd({"L": _lane()}, {"blip": _track([12.0] * 2)})
        prov = annotate_speed_targets(sd)
        assert prov["counts"]["observed"] == 0

    def test_a_malformed_track_is_skipped_not_raised_on(self):
        # One track with a missing field must not fail the whole scenario
        # conversion — the scene still has a map and other traffic.
        sd = _sd({"L": _lane()},
                 {"broken": {"type": "VEHICLE", "state": {"position": None}},
                  "ok": _track([9.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert prov["counts"]["observed"] == 1

    def test_non_vehicles_do_not_set_the_speed_limit(self):
        sd = _sd({"L": _lane()},
                 {"p": _track([2.0] * 40, obj_type="PEDESTRIAN")})
        prov = annotate_speed_targets(sd)
        assert prov["counts"]["observed"] == 0


class TestFallbackChain:
    def test_a_lane_with_no_traffic_inherits_from_its_neighbour(self):
        lanes = {
            "L0": _lane(length=40.0, exit_lanes=["L1"]),
            "L1": _lane(x0=40.0, length=40.0, entry_lanes=["L0"]),
        }
        # 20 frames x 9 m/s x 0.1 s = 18 m: the vehicle never leaves L0.
        sd = _sd(lanes, {"v": _track([9.0] * 20)})
        annotate_speed_targets(sd)
        src = {k: v[SPEED_LIMIT_SOURCE_KEY] for k, v in
               sd["map_features"].items()}
        assert "observed" in src.values()
        inherited = [k for k, v in src.items() if v.startswith("propagated")]
        assert inherited, f"nothing propagated across the lane graph: {src}"
        assert (sd["map_features"][inherited[0]][SPEED_LIMIT_KEY]
                == pytest.approx(
                    sd["map_features"][
                        [k for k, v in src.items()
                         if v == "observed"][0]][SPEED_LIMIT_KEY]))

    def test_a_disconnected_lane_falls_back_to_the_scene_median(self):
        lanes = {"driven": _lane(), "island": _lane(x0=500.0, y0=500.0)}
        sd = _sd(lanes, {"v": _track([7.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert (sd["map_features"]["island"][SPEED_LIMIT_SOURCE_KEY]
                == "scene_median")
        assert sd["map_features"]["island"][SPEED_LIMIT_KEY] == pytest.approx(
            prov["scene_median_mps"])

    def test_a_scene_with_no_moving_traffic_annotates_nothing(self):
        # Better an explicit "no evidence" than a fabricated limit: the
        # planner's own fallback then applies, and the record says why.
        sd = _sd({"L": _lane()}, {"v": _track([0.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert prov["method"] == "none"
        assert prov["counts"]["unset"] == 1
        assert SPEED_LIMIT_KEY not in sd["map_features"]["L"]

    def test_annotating_twice_does_not_promote_an_estimate_to_a_posted_fact(
            self):
        # The reconstruction assembler re-projects scenarios it has already
        # built. Reading "has a limit" as "the map posted a limit" would
        # relabel every inferred target ``dataset`` on the second pass and
        # freeze it — a silent promotion no downstream reader could detect.
        sd = _sd({"L": _lane(), "island": _lane(x0=500.0)},
                 {"v": _track([7.0] * 40)})
        first = annotate_speed_targets(sd)
        second = annotate_speed_targets(sd)
        assert second["counts"] == first["counts"]
        assert second["counts"]["dataset"] == 0
        assert second["target_mps"] == first["target_mps"]

    def test_a_partially_posted_map_falls_back_to_its_own_limits(self):
        # Some maps post limits on arterials only. The unposted residential
        # lanes should inherit the map's own scale, not a global constant.
        lanes = {"posted": _lane(speed_limit_mps=17.0),
                 "quiet": _lane(x0=500.0, y0=500.0)}
        sd = _sd(lanes, {})
        prov = annotate_speed_targets(sd)
        assert prov["scene_median_source"] == "dataset"
        assert sd["map_features"]["quiet"][SPEED_LIMIT_KEY] == pytest.approx(
            17.0)

    def test_dataset_limits_win_and_are_left_untouched(self):
        lanes = {"posted": _lane(speed_limit_mps=11.18)}
        sd = _sd(lanes, {"v": _track([4.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert sd["map_features"]["posted"][SPEED_LIMIT_KEY] == 11.18
        assert sd["map_features"]["posted"][SPEED_LIMIT_SOURCE_KEY] == "dataset"
        assert prov["counts"]["dataset"] == 1
        assert prov["counts"]["observed"] == 0


class TestClamping:
    def test_a_crawl_does_not_pin_the_teacher_at_walking_pace(self):
        sd = _sd({"L": _lane()}, {"v": _track([0.8] * 40)})
        prov = annotate_speed_targets(sd)
        assert sd["map_features"]["L"][SPEED_LIMIT_KEY] == pytest.approx(
            SpeedTargetParams().min_target_mps)
        assert prov["clamped"] == 1

    def test_a_tracking_outlier_cannot_post_a_racetrack(self):
        sd = _sd({"L": _lane(length=400.0)}, {"v": _track([80.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert sd["map_features"]["L"][SPEED_LIMIT_KEY] == pytest.approx(
            SpeedTargetParams().max_target_mps)
        assert prov["clamped"] == 1


# ── the record ────────────────────────────────────────────────────────────


class TestProvenance:
    def test_every_lane_is_accounted_for_exactly_once(self):
        lanes = {"a": _lane(), "b": _lane(x0=500.0, y0=500.0),
                 "c": _lane(speed_limit_mps=20.0, x0=-500.0)}
        sd = _sd(lanes, {"v": _track([6.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert sum(prov["counts"].values()) == len(lanes)
        assert set(prov["target_mps"]) == set(lanes)

    def test_the_record_carries_the_parameters_that_produced_it(self):
        # A target that can only be reproduced by guessing the percentile is
        # not auditable. The record must be self-contained.
        sd = _sd({"L": _lane()}, {"v": _track([6.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert sd["metadata"][PROVENANCE_KEY] is prov
        assert prov["params"]["lane_percentile"] == 85.0
        assert prov["params"]["track_percentile"] == 90.0
        assert prov["params"]["include_ego"] is True
        assert prov["version"] >= 1

    def test_ego_contribution_is_recorded_because_it_is_an_oracle(self):
        # The target is derived from a log the policy has not driven yet.
        # A reader must be able to see which lanes leaned on the ego.
        sd = _sd({"L": _lane()}, {"ego": _track([6.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert prov["evidence"]["L"]["ego_contributed"] is True
        assert prov["evidence"]["L"]["n_tracks"] == 1

    def test_the_ego_free_variant_drops_the_oracle(self):
        sd = _sd({"L": _lane()}, {"ego": _track([6.0] * 40)})
        prov = annotate_speed_targets(
            sd, params=SpeedTargetParams(include_ego=False))
        assert prov["counts"]["observed"] == 0
        assert prov["params"]["include_ego"] is False


class TestRobustness:
    def test_a_non_finite_lane_vertex_does_not_abort_the_scenario(self):
        # cKDTree rejects NaN. This annotation sits on the critical path of
        # every scenario load, so one bad vertex in one lane must not take
        # down a conversion — still less a shard of a bulk cache build.
        bad = _lane()
        bad["polyline"] = bad["polyline"].copy()
        bad["polyline"][3, 1] = np.nan
        sd = _sd({"bad": bad, "good": _lane(y0=50.0)},
                 {"v": _track([9.0] * 40, y=50.0)})
        prov = annotate_speed_targets(sd)
        assert prov["counts"]["observed"] == 1
        assert SPEED_LIMIT_KEY not in sd["map_features"]["bad"]
        assert prov["n_lane_features"] == 1
        assert prov["n_lane_features_total"] == 2

    def test_a_non_finite_track_sample_does_not_abort_the_scenario(self):
        good = _track([9.0] * 40)
        bad = _track([9.0] * 40)
        bad["state"]["position"] = bad["state"]["position"].copy()
        bad["state"]["position"][5, 0] = np.inf
        sd = _sd({"L": _lane()}, {"bad": bad, "good": good})
        assert annotate_speed_targets(sd)["counts"]["observed"] == 1

    def test_non_string_lane_keys_are_not_a_crash(self):
        # Keys index back into map_features; stringifying them for lookup
        # turned every non-str producer into a KeyError.
        sd = _sd({7: _lane()}, {"v": _track([9.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert sd["map_features"][7][SPEED_LIMIT_KEY] == pytest.approx(
            9.0, rel=1e-3)
        assert prov["target_mps"]["7"] == pytest.approx(9.0, rel=1e-3)

    def test_the_record_is_json_serializable(self):
        # A harness dumps this to the run directory; a leaked np.float64
        # would fail the run, not the estimator.
        import json

        lanes = {"a": _lane(), "b": _lane(x0=500.0, y0=500.0),
                 "c": _lane(speed_limit_mps=20.0, x0=-500.0)}
        sd = _sd(lanes, {"v": _track([6.0] * 40)})
        json.dumps(annotate_speed_targets(sd))


class TestPropagation:
    @staticmethod
    def _chain(n, **first):
        lanes = {}
        for i in range(n):
            lanes[f"L{i}"] = _lane(x0=40.0 * i, length=40.0,
                                   entry_lanes=[f"L{i - 1}"] if i else [],
                                   exit_lanes=[f"L{i + 1}"] if i < n - 1
                                   else [])
        lanes["L0"].update(first)
        return lanes

    def test_inheritance_stops_at_max_graph_hops(self):
        # 6 lanes in a chain, traffic only on L0: L1..L3 inherit, L4/L5 are
        # beyond 3 hops and must fall back to the scene median instead of
        # silently inheriting from arbitrarily far away.
        sd = _sd(self._chain(6), {"v": _track([9.0] * 20)})
        annotate_speed_targets(sd)
        src = {k: v[SPEED_LIMIT_SOURCE_KEY]
               for k, v in sd["map_features"].items()}
        assert src["L0"] == "observed"
        assert src["L1"] == "propagated_1hop"
        assert src["L3"] == "propagated_3hop"
        assert src["L4"] == "scene_median"

    def test_a_donor_is_counted_once_however_often_the_map_lists_the_edge(
            self):
        # A symmetric edge is normally declared twice (A lists B as an exit,
        # B lists A as an entry). Weighting the median by how often the map
        # spelled an edge out would let map verbosity move the target.
        lanes = {
            "slow": _lane(y0=0.0, length=30.0, exit_lanes=["mid", "mid"]),
            "fast": _lane(y0=60.0, length=30.0, exit_lanes=["mid"]),
            "mid": _lane(x0=200.0, y0=200.0, length=30.0,
                         entry_lanes=["slow", "fast"]),
        }
        sd = _sd(lanes, {"a": _track([4.0] * 20, y=0.0),
                         "b": _track([12.0] * 20, y=60.0)})
        annotate_speed_targets(sd)
        assert sd["map_features"]["mid"][SPEED_LIMIT_KEY] == pytest.approx(
            8.0, rel=0.05), "the doubly-listed donor outvoted the other lane"

    def test_the_record_says_which_lanes_a_target_was_inherited_from(self):
        sd = _sd(self._chain(3), {"v": _track([9.0] * 20)})
        prov = annotate_speed_targets(sd)
        assert prov["evidence"]["L1"]["inherited_from"] == ["L0"]
        assert prov["evidence"]["L1"]["hops"] == 1


class TestMode:
    def test_off_annotates_nothing_and_says_so(self):
        sd = _sd({"L": _lane()}, {"v": _track([9.0] * 40)})
        prov = annotate_speed_targets(sd, mode="off")
        assert SPEED_LIMIT_KEY not in sd["map_features"]["L"]
        assert prov["mode"] == "off"
        assert prov["counts"]["unset"] == 1

    def test_off_still_passes_a_dataset_limit_through(self):
        sd = _sd({"L": _lane(speed_limit_mps=13.4)}, {})
        prov = annotate_speed_targets(sd, mode="off")
        assert sd["map_features"]["L"][SPEED_LIMIT_KEY] == 13.4
        assert prov["counts"]["dataset"] == 1

    @pytest.mark.parametrize("raw,expected", [
        ("off", "off"), ("OFF", "off"), ("0", "off"), ("none", "off"),
        ("", "inferred"), ("inferred", "inferred"), ("1", "inferred"),
    ])
    def test_env_values_resolve_predictably(self, raw, expected):
        assert speed_target_mode(raw) == expected

    def test_env_var_selects_the_mode(self, monkeypatch):
        monkeypatch.setenv("NEXUSSIM_SPEED_TARGET", "off")
        sd = _sd({"L": _lane()}, {"v": _track([9.0] * 40)})
        assert annotate_speed_targets(sd)["mode"] == "off"

    def test_an_unrecognized_value_warns_instead_of_failing_open_silently(
            self, caplog):
        # A typo'd arm labelled "targets off" that ran with targets on is a
        # contaminated result nobody would ever look for.
        with caplog.at_level("WARNING"):
            assert speed_target_mode("of") == "inferred"
        assert "not recognized" in caplog.text

    def test_off_reports_targets_a_previous_pass_left_on_the_lane(self):
        # ``off`` does not erase; the planner will still read whatever is
        # there. Calling those lanes "unset" is how a record starts
        # disagreeing with the scenario it describes.
        sd = _sd({"L": _lane()}, {"v": _track([9.0] * 40)})
        annotate_speed_targets(sd)
        prov = annotate_speed_targets(sd, mode="off")
        assert prov["counts"]["unset"] == 0
        assert prov["counts"]["preexisting"] == 1
        assert prov["target_mps"]["L"] == pytest.approx(
            sd["map_features"]["L"][SPEED_LIMIT_KEY])
        assert prov["evidence"]["L"]["recorded_source"] == "observed"

    def test_the_ego_can_be_dropped_from_the_env(self, monkeypatch):
        # An oracle-free arm must be runnable without editing code.
        monkeypatch.setenv("NEXUSSIM_SPEED_TARGET_INCLUDE_EGO", "0")
        sd = _sd({"L": _lane()}, {"ego": _track([9.0] * 40)})
        prov = annotate_speed_targets(sd)
        assert prov["params"]["include_ego"] is False
        assert prov["counts"]["observed"] == 0


class TestTheWiringItself:
    """The two lines that make this feature exist at all.

    Every other test in this file calls the estimator directly. Without these,
    deleting the converter's call to ``annotate_speed_targets`` — or renaming
    ``SPEED_LIMIT_KEY`` out from under the planner that hardcodes the literal —
    leaves the whole suite green and every lane back on ``fraction x 15 m/s``.
    """

    def test_the_converter_annotates_the_scenarios_it_builds(self):
        from navsafe.scenario.py123d_scenario_description import (
            scenario_description_from_nexus_log,
        )
        from navsafe.scenario.training_schema import (
            NexusEgoState,
            NexusFrameState,
            NexusLaneState,
            NexusMapState,
            NexusScenarioLog,
        )

        n = 40
        s = np.arange(0.0, 60.0, 0.5)
        zeros = np.zeros_like(s)
        lane = NexusLaneState(
            lane_id="L", lane_type="surface_street", lane_group_id=None,
            left_lane_id=None, right_lane_id=None, predecessor_ids=(),
            successor_ids=(), speed_limit_mps=None,
            centerline=np.column_stack([s, zeros]),
            left_boundary=np.column_stack([s, zeros + 1.8]),
            right_boundary=np.column_stack([s, zeros - 1.8]),
            polygon=np.array([[0.0, -1.8], [60.0, -1.8], [60.0, 1.8],
                              [0.0, 1.8]]))
        frames = [
            NexusFrameState(
                scenario_id="wiring", iteration=t, timestamp_us=int(t * 1e5),
                ego=NexusEgoState(
                    timestamp_us=int(t * 1e5), x=float(t), y=0.0, z=0.0,
                    heading=0.0, vx=10.0, vy=0.0, vz=0.0, ax=0.0, ay=0.0,
                    az=0.0, yaw_rate=0.0, steering_angle=0.0,
                    length=4.5, width=1.8, height=1.5),
                actors=(), traffic_lights=())
            for t in range(n)
        ]
        log = NexusScenarioLog(
            scenario_id="wiring", dataset="test", split="test",
            location="test", log_name="wiring",
            timestamps_us=[int(f.timestamp_us or 0) for f in frames],
            frames=frames,
            map_state=NexusMapState(location="test", lanes={"L": lane}))

        sd = scenario_description_from_nexus_log(log)
        feat = sd["map_features"]["L"]
        assert SPEED_LIMIT_KEY in feat, (
            "the converter did not annotate speed targets — every lane of "
            "every AV2 scene is back on the context-free 15 m/s fallback")
        assert feat[SPEED_LIMIT_SOURCE_KEY] == "observed"
        assert sd["metadata"][PROVENANCE_KEY]["counts"]["observed"] == 1

    def test_the_planner_reads_the_key_this_module_writes(self):
        # Producer defines a constant, consumer hardcodes the literal. A
        # rename would silently revert the teacher to the 15 m/s fallback
        # with every test still green, so pin the contract across the seam.
        pytest.importorskip("torch")
        from navsafe.policy.state.pdm_closed_planner.planner import (
            _speed_limit_to_mps,
        )
        assert _speed_limit_to_mps({SPEED_LIMIT_KEY: 12.0}) == pytest.approx(
            12.0)


class TestConverterForwardsDatasetLimit:
    def test_a_source_map_speed_limit_reaches_the_lane_feature(self):
        # py123d has carried ``Lane.speed_limit_mps`` all along; the converter
        # dropped it, so even a source that HAS limits (nuPlan) reached the
        # planner as "no annotation".
        from navsafe.scenario.py123d_scenario_description import (
            _map_features_from_map_state,
        )

        class _Lane:
            lane_type = "SURFACE_STREET"
            centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            left_boundary = None
            right_boundary = None
            polygon = None
            speed_limit_mps = 13.4

        class _MapState:
            lanes = {"lane_0": _Lane()}
            road_edges: dict = {}
            road_lines: dict = {}

        feats = _map_features_from_map_state(_MapState())
        assert feats["lane_0"][SPEED_LIMIT_KEY] == pytest.approx(13.4)
        assert feats["lane_0"][SPEED_LIMIT_SOURCE_KEY] == "dataset"

    def test_a_source_without_a_limit_emits_no_key(self):
        from navsafe.scenario.py123d_scenario_description import (
            _map_features_from_map_state,
        )

        class _Lane:
            lane_type = "SURFACE_STREET"
            centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
            left_boundary = None
            right_boundary = None
            polygon = None
            speed_limit_mps = None

        class _MapState:
            lanes = {"lane_0": _Lane()}
            road_edges: dict = {}
            road_lines: dict = {}

        assert SPEED_LIMIT_KEY not in _map_features_from_map_state(
            _MapState())["lane_0"]
