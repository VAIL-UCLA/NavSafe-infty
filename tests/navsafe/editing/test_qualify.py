# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Host qualification: leaf predicates over synthetic hosts of known shape."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.author import AuthoringError, author_recipe
from navsafe.benchmark.editing.placement.probe import HostProbe, PlacementError
from navsafe.benchmark.editing.qualify import (
    PREDICATES,
    qualify_host,
)
from navsafe.benchmark.leaves import load_all

T = 80
DT = 0.1
EGO_Z = 10.0
Z_TO_GROUND = 1.4
ROAD_Z = EGO_Z - Z_TO_GROUND


def _lane(xs, ys, *, z=ROAD_Z):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return np.stack([xs, ys, np.full_like(xs, z)], axis=1)


def _seg(a, b, *, y=0.0, reverse=False):
    xs = np.arange(a, b + 1e-9, 2.0)
    if reverse:
        xs = xs[::-1]
    return _lane(xs, np.full(len(xs), y))


def _vehicle(pos_xy, *, heading=0.0):
    pos = np.zeros((T, 3), np.float64)
    pos[:, :2] = pos_xy
    pos[:, 2] = ROAD_Z + 0.8
    return {
        "type": "VEHICLE",
        "state": {
            "position": pos,
            "length": np.full(T, 4.6, np.float32),
            "width": np.full(T, 1.9, np.float32),
            "height": np.full(T, 1.6, np.float32),
            "heading": np.full(T, heading),
            "velocity": np.zeros((T, 2)),
            "valid": np.ones(T, bool),
        },
        "metadata": {},
    }


def _host_sd(
    *,
    opposing_y=None,  # y of an opposing-direction lane, or None for one-way
    curve=False,  # bend the ego route so no straight stretch exists
    merge_lane=False,  # a same-direction lane converging into the route
    oncoming_track=False,  # a real vehicle driving the opposing direction
    same_direction_ys=(),  # extra same-direction lanes, by y
    bike_lane_y=None,  # y of a MAPPED bike lane (nuPlan carries these)
    moving_vehicles=0,  # logged vehicles that actually drive
    crosswalk_x=None,  # x of a mapped crosswalk on the route
) -> dict:
    pos = np.zeros((T, 3), np.float64)
    pos[:, 0] = np.arange(T) * 1.5
    if curve:
        # A constant-radius arc (R = 60 m): EVERY 25 m window drifts 24 deg,
        # so no straight stretch exists anywhere. (A parabola would not do:
        # its drift-per-arc decays downstream and grazes the threshold.)
        theta = np.arange(T) * 1.5 / 60.0
        pos[:, 0] = 60.0 * np.sin(theta)
        pos[:, 1] = 60.0 * (1.0 - np.cos(theta))
    pos[:, 2] = EGO_Z
    map_features = {
        "lane_ego": {"type": "LANE_SURFACE_STREET", "polyline": _seg(-20.0, 140.0)},
    }
    if opposing_y is not None:
        map_features["lane_opposing"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": _seg(-20.0, 140.0, y=opposing_y, reverse=True),
        }
    for i, y in enumerate(same_direction_ys):
        map_features[f"lane_same_{i}"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": _seg(-20.0, 140.0, y=float(y)),
        }
    if bike_lane_y is not None:
        if abs(float(bike_lane_y)) < 1e-9:
            # The ego is riding the painted lane itself.
            map_features["lane_ego"]["type"] = "LANE_BIKE_LANE"
        else:
            map_features["lane_bike"] = {
                "type": "LANE_BIKE_LANE",
                "polyline": _seg(-20.0, 140.0, y=float(bike_lane_y)),
            }
    if crosswalk_x is not None:
        map_features["walk_1"] = {
            "type": "CROSSWALK",
            "polyline": _lane([crosswalk_x, crosswalk_x], [-4.0, 4.0]),
        }
    if merge_lane:
        # Converges from 4 m beside the route down to 0.5 m over 60 m.
        xs = np.arange(10.0, 90.0, 2.0)
        ys = np.clip(4.0 - (xs - 10.0) * (3.5 / 60.0), 0.5, 4.0)
        map_features["lane_merge"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": _lane(xs, ys),
        }
    tracks = {
        "ego": {
            "type": "VEHICLE",
            "state": {
                "position": pos,
                "heading": np.zeros(T),
                "velocity": np.zeros((T, 2)),
                "valid": np.ones(T, bool),
            },
            "metadata": {"object_id": "ego"},
        }
    }
    for i in range(int(moving_vehicles)):
        xy = np.stack([np.arange(T) * 1.2 + 20.0 * i, np.full(T, -3.5)], axis=1)
        tracks[f"mover_{i}"] = _vehicle(xy)
    if oncoming_track:
        xy = np.stack([90.0 - np.arange(T) * 1.0, np.full(T, opposing_y or 3.5)], axis=1)
        tracks["oncoming"] = _vehicle(xy, heading=np.pi)
    return {
        "metadata": {
            "sdc_id": "ego",
            "ts": (np.arange(T) * DT * 1e6).astype(np.float64),
            "coordinate": "local_frame0",
            "route_lane_ids": ["lane_ego"],
        },
        "tracks": tracks,
        "map_features": map_features,
    }


def _probe(sd) -> HostProbe:
    probe = HostProbe(sd, ego_z_to_ground_m=Z_TO_GROUND)
    probe.after_frame = 8
    return probe


class TestPredicates:
    def test_two_way_road_needs_an_opposing_lane(self):
        assert PREDICATES["two_way_road"](_probe(_host_sd(opposing_y=3.5))).ok
        assert not PREDICATES["two_way_road"](_probe(_host_sd(opposing_y=None))).ok

    def test_a_same_direction_neighbour_is_not_two_way(self):
        sd = _host_sd(opposing_y=None)
        sd["map_features"]["lane_parallel"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": _seg(-20.0, 140.0, y=3.5),  # same direction
        }
        assert not PREDICATES["two_way_road"](_probe(sd)).ok

    def test_narrow_corridor_gates_on_centreline_gap(self):
        assert PREDICATES["narrow_corridor"](_probe(_host_sd(opposing_y=3.5))).ok
        assert not PREDICATES["narrow_corridor"](_probe(_host_sd(opposing_y=7.5))).ok

    def test_oncoming_vehicle_in_log(self):
        hit = PREDICATES["oncoming_vehicle_in_log"](
            _probe(_host_sd(opposing_y=3.5, oncoming_track=True))
        )
        assert hit.ok and "oncoming" in hit.evidence
        assert not PREDICATES["oncoming_vehicle_in_log"](
            _probe(_host_sd(opposing_y=3.5))
        ).ok

    def test_straight_stretch_rejects_a_bend(self):
        assert PREDICATES["straight_stretch"](_probe(_host_sd())).ok
        assert not PREDICATES["straight_stretch"](_probe(_host_sd(curve=True))).ok

    def test_merge_convergence_wants_actual_convergence(self):
        assert PREDICATES["merge_convergence"](_probe(_host_sd(merge_lane=True))).ok
        # A parallel same-direction lane at constant offset is NOT a merge.
        sd = _host_sd()
        sd["map_features"]["lane_parallel"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": _seg(-20.0, 140.0, y=3.5),
        }
        assert not PREDICATES["merge_convergence"](_probe(sd)).ok

    def test_multi_lane_counts_same_direction_only(self):
        # Two extra same-direction lanes + the ego's own = 3.
        assert PREDICATES["multi_lane"](
            _probe(_host_sd(same_direction_ys=(3.5, 7.0)))).ok
        # An OPPOSING lane widens the road but is not somewhere the ego may go.
        assert not PREDICATES["multi_lane"](
            _probe(_host_sd(opposing_y=-3.5, same_direction_ys=(3.5,)))).ok

    def test_light_traffic_is_a_density_on_the_ego_s_own_carriageway(self):
        # The route here is ~118 m, so the ceiling of 4.0 per 100 m is ~4 cars.
        assert PREDICATES["light_traffic"](_probe(_host_sd(moving_vehicles=4))).ok
        assert not PREDICATES["light_traffic"](_probe(_host_sd(moving_vehicles=9))).ok

    def test_parked_cars_do_not_block_a_lane_change(self):
        sd = _host_sd()
        for i in range(20):
            sd["tracks"][f"parked_{i}"] = _vehicle(
                np.tile([10.0 + 3.0 * i, -4.0], (T, 1)))
        assert PREDICATES["light_traffic"](_probe(sd)).ok

    def test_traffic_elsewhere_in_the_scene_is_not_counted(self):
        # 40 moving vehicles across a city block is an ordinary 20 s of urban
        # nuPlan; counting them said "not light traffic" about every host.
        sd = _host_sd()
        for i in range(40):
            xy = np.stack([np.arange(T) * 1.2, np.full(T, 200.0 + 5.0 * i)], axis=1)
            sd["tracks"][f"far_{i}"] = _vehicle(xy)
        assert PREDICATES["light_traffic"](_probe(sd)).ok

    def test_oncoming_traffic_cannot_block_the_manoeuvre(self):
        # It is not somewhere the ego was going to go.
        sd = _host_sd(opposing_y=3.5)
        for i in range(12):
            xy = np.stack([120.0 - np.arange(T) * 1.2 - 4.0 * i, np.full(T, 3.5)], axis=1)
            sd["tracks"][f"onc_{i}"] = _vehicle(xy, heading=np.pi)
        assert PREDICATES["light_traffic"](_probe(sd)).ok

    def test_the_same_count_on_a_short_route_is_dense(self):
        # Nine vehicles over 244 m and fifteen over 70 m are opposite
        # situations; a raw count cannot tell them apart, which is why this is
        # normalised by route length. Four cars is light over ~118 m and a
        # queue over ~24 m, and only the density says so.
        def _four_movers(route_step):
            sd = _host_sd()
            pos = np.asarray(sd["tracks"]["ego"]["state"]["position"])
            pos[:, 0] = np.arange(T) * route_step
            span = float(pos[-1, 0])
            for i in range(4):
                start = span * (i + 0.5) / 5.0
                xy = np.stack([start + np.linspace(0, 4.0, T), np.full(T, -3.5)], axis=1)
                sd["tracks"][f"mover_{i}"] = _vehicle(xy)
            return PREDICATES["light_traffic"](_probe(sd))

        assert _four_movers(1.5).ok            # ~118 m -> 3.4 per 100 m
        assert not _four_movers(0.3).ok        # ~24 m  -> 17 per 100 m

    def test_bike_permitted_lane_names_a_mapped_bike_lane_when_it_is_the_egos(self):
        # The ego is riding the painted lane itself, so sharing it IS sharing a
        # bike lane and the evidence should say so.
        mapped = PREDICATES["bike_permitted_lane"](_probe(_host_sd(bike_lane_y=0.0)))
        assert mapped.ok and "dedicated" in mapped.evidence
        # No bike lane: the ego's own lane still qualifies, and the evidence
        # says which answer it is — they are different scenarios.
        plain = PREDICATES["bike_permitted_lane"](_probe(_host_sd(same_direction_ys=(3.5,))))
        assert plain.ok and "own lane" in plain.evidence

    def test_bike_permitted_lane_is_the_egos_lane_not_the_rightmost(self):
        # A parking lane / second carriageway 3.5 m to the RIGHT is where a
        # cyclist rides, and it is not where THIS ego drives — riders put
        # there are never in its way, so there is no overtake to judge. The
        # answer has to be the lane the ego is in.
        probe = _probe(_host_sd(same_direction_ys=(-3.5,)))
        lane = probe.bike_lane()
        assert lane["lane_id"] == "lane_ego"
        assert abs(lane["lateral_m"]) < 0.5

    def test_crosswalk_on_route_gates_the_crossing_leaves(self):
        assert PREDICATES["crosswalk_on_route"](_probe(_host_sd(crosswalk_x=60.0))).ok
        assert not PREDICATES["crosswalk_on_route"](_probe(_host_sd())).ok
        # A HARD gate for both: the actors walk the crossing's own geometry, so
        # a host without one cannot carry the leaf. Mid-block is a different
        # scenario, not a fallback.
        for leaf in ("R-3", "R-4"):
            man = load_all()[leaf]
            assert "crosswalk_on_route" in man.qualify

    def test_a_crossing_along_the_route_is_not_one_the_ego_drives_over(self):
        # A junction carries the side street's crossing a few metres from its
        # own. Taking the nearest without the angle test puts a pedestrian
        # walking down the ego's lane instead of across it.
        sd = _host_sd(crosswalk_x=60.0)
        sd["map_features"]["walk_along"] = {
            "type": "CROSSWALK",
            "polyline": _lane([50.0, 70.0], [0.5, 0.5]),   # parallel to the route
        }
        found = _probe(sd).crosswalks_on_route()
        assert [f["feature_id"] for f in found] == ["walk_1"]


class TestNewReferences:
    def test_merging_lane_chain_is_the_lane_the_gate_passed_on(self):
        # One search, so the gate and the placement cannot disagree about which
        # lane merged — the failure mode that motivated sharing it.
        probe = _probe(_host_sd(merge_lane=True))
        merge = probe.merging_lane()
        assert merge is not None and merge["lane_id"] == "lane_merge"
        assert PREDICATES["merge_convergence"](probe).evidence.startswith("lane lane_merge")
        chain = probe.resolve_reference("merging_lane_chain")
        assert chain.total > 0

    def test_merging_lane_chain_refuses_a_host_without_a_merge(self):
        with pytest.raises(PlacementError, match="no same-direction lane converges"):
            _probe(_host_sd()).resolve_reference("merging_lane_chain")

    def test_bike_lane_chain_falls_back_to_the_rightmost_lane(self):
        # Laterals are +LEFT, so the RIGHTMOST of {0.0, +3.5} is the ego's own.
        probe = _probe(_host_sd(same_direction_ys=(3.5,)))
        lane = probe.bike_lane()
        assert lane is not None and not lane["dedicated"]
        assert probe.resolve_reference("bike_lane_chain").total > 0

    def test_bike_lane_chain_stays_in_the_egos_lane_past_a_separated_bike_lane(self):
        # A painted bike lane 3 m off the ego's line is where a cyclist
        # belongs and not where this leaf's event is: riders put there are in
        # a lane the ego never enters, so it has nothing to overtake.
        probe = _probe(_host_sd(bike_lane_y=-3.0, same_direction_ys=(3.5,)))
        lane = probe.bike_lane()
        assert lane["lane_id"] == "lane_ego" and not lane["dedicated"]

    def test_arrive_with_ego_times_against_the_merge_point(self):
        # V-10's whole mechanism. The merging lane never CROSSES the ego route,
        # so there is no intersection to time against; the conflict is where the
        # two converge, and the spawn is back-solved from it.
        sd = _host_sd(merge_lane=True)
        spec = {
            "recipe_id": "V-10/unsafe_merge/syn/001",
            "leaf": "V-10",
            "scenario": "unsafe_merge",
            "host": {"scene": "syn", "world_version": "test@0"},
            "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
            "actors": {
                "mainline_vehicle": {
                    "op": "insert",
                    "track": {"type": "VEHICLE"},
                    "asset": {"dims": [4.6, 1.9, 1.6]},
                    "authored": {
                        "template": "dynamic",
                        "reference": "merging_lane_chain",
                        "arrive_with_ego": True,
                        "speed": 8.0,
                        "blind_to_ego": True,
                    },
                }
            },
            "requires": {"min_reaction_s": 0.0},
            "pair": {"e0_removes": ["mainline_vehicle"], "pair_id": "v10"},
        }
        recipe, _ = author_recipe(sd, spec, use_registry=False)
        actor = recipe.actors["mainline_vehicle"]
        solved = actor.authored["solved"]
        # The lane closes on the route between x=10 and x=70, so the conflict
        # lands downstream of the spawn and the car is placed BEFORE it.
        assert solved["conflict_xy"][0] > 40.0
        assert np.asarray(actor.spawn["position"], np.float64)[0] < solved["conflict_xy"][0]
        # Same direction as the ego, and deaf to it — else IDM opens the gap.
        assert actor.policy["kind"] == "idm" and actor.policy["v0"] > 0.0
        assert actor.policy["blind_to_ego"] is True

    def test_arrive_with_ego_honours_the_leafs_reaction_floor(self):
        # Co-arrival at a conflict the ego is ALREADY at is contact on the
        # first scored frame, not a merge to judge: 13c555e68671524f ended at
        # frame 0, rear-ended by the mainline car, with nothing to score. The
        # leaf declares the reaction it needs; the layout path honoured it and
        # this one did not.
        sd = _host_sd(merge_lane=True)

        def solve(reaction_s):
            spec = {
                "recipe_id": "V-10/unsafe_merge/syn/001",
                "leaf": "V-10", "scenario": "unsafe_merge",
                "host": {"scene": "syn", "world_version": "test@0"},
                "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
                "actors": {
                    "mainline_vehicle": {
                        "op": "insert",
                        "track": {"type": "VEHICLE"},
                        "asset": {"dims": [4.6, 1.9, 1.6]},
                        "authored": {
                            "template": "dynamic",
                            "reference": "merging_lane_chain",
                            "arrive_with_ego": True,
                            "speed": 8.0,
                            "blind_to_ego": True,
                        },
                    }
                },
                "requires": {"min_reaction_s": reaction_s},
                "pair": {"e0_removes": ["mainline_vehicle"], "pair_id": "v10"},
            }
            recipe, _ = author_recipe(sd, spec, use_registry=False)
            return recipe.actors["mainline_vehicle"].authored["solved"]

        base, floored = solve(0.0), solve(6.0)
        assert floored["ego_arrival_frame"] > base["ego_arrival_frame"]
        assert floored["arrive_offset_s"] > 0.0

    def test_a_short_approach_slows_the_actor_instead_of_moving_it_off_the_road(self):
        # The intent is a TIME -- be at the conflict when the ego is -- so a
        # reference that cannot supply the approach at the authored speed makes
        # the actor start at the reference's own beginning and travel it slower,
        # rather than being extrapolated off the end of its own path.
        #
        # `Polyline.offset` extrapolates past its own ends without complaining,
        # and that is what baked 13c555e68671524f's mainline car 18.2 m OFF the
        # path it was then driven along -- which began 2.3 m from the ego. The
        # episode ended on frame 0, rear-ended by its own inserted car,
        # `infra_failure`, nothing to score. 49a0d29c7058501c: -20.0 m, same
        # failure. The spawn assertion at the end of this test is the one that
        # would have caught it.
        sd = _host_sd(merge_lane=True)
        spec = {
            "recipe_id": "V-10/unsafe_merge/syn/001",
            "leaf": "V-10", "scenario": "unsafe_merge",
            "host": {"scene": "syn", "world_version": "test@0"},
            "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
            "actors": {
                "mainline_vehicle": {
                    "op": "insert",
                    "track": {"type": "VEHICLE"},
                    "asset": {"dims": [4.6, 1.9, 1.6]},
                    "authored": {
                        "template": "dynamic",
                        "reference": "merging_lane_chain",
                        "arrive_with_ego": True,
                        "speed": 25.0,
                        "blind_to_ego": True,
                    },
                }
            },
            "requires": {"min_reaction_s": 0.0},
            "pair": {"e0_removes": ["mainline_vehicle"], "pair_id": "v10"},
        }
        recipe, _ = author_recipe(sd, spec, use_registry=False)
        actor = recipe.actors["mainline_vehicle"]
        assert actor.authored.get("speed_clamped_by_reference") is True
        assert 2.0 <= actor.policy["v0"] < 25.0
        # ...and the spawn is ON the path it will be driven along.
        spawn = np.asarray(actor.spawn["position"], np.float64)[:2]
        path = np.asarray(actor.policy["path_polyline"], np.float64)
        assert float(np.linalg.norm(path - spawn, axis=1).min()) < 1.0

    def test_a_dynamic_actor_rides_its_own_line_not_the_lane_centre(self):
        # The spawn is placed at the actor's `lateral`; handing the policy the
        # raw centreline made the first controlled step a jump back into the
        # middle of the lane. R-2's cyclists were spawned 1.2-1.4 m out against
        # the kerb and snapped in on frame 2.
        sd = _host_sd(same_direction_ys=(-3.5,))
        spec = {
            "recipe_id": "R-2/bicycle_crossing/syn/001",
            "leaf": "R-2", "scenario": "bicycle_crossing",
            "host": {"scene": "syn", "world_version": "test@0"},
            "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
            "actors": {
                "cyclist_1": {
                    "op": "insert",
                    "track": {"type": "CYCLIST"},
                    "asset": {"dims": [1.7, 0.6, 1.7]},
                    "authored": {
                        "template": "dynamic",
                        "reference": "bike_lane_chain",
                        "arc": 10.0,
                        "lateral": -1.4,
                        "speed": 3.0,
                    },
                }
            },
            "pair": {"e0_removes": ["cyclist_1"], "pair_id": "r2"},
        }
        recipe, _ = author_recipe(sd, spec, use_registry=False)
        actor = recipe.actors["cyclist_1"]
        spawn = np.asarray(actor.spawn["position"], np.float64)[:2]
        path = np.asarray(actor.policy["path_polyline"], np.float64)
        assert float(np.linalg.norm(path - spawn, axis=1).min()) < 0.2

    def test_an_unknown_reference_names_the_new_ones(self):
        with pytest.raises(PlacementError, match="merging_lane_chain"):
            _probe(_host_sd()).resolve_reference("motorway_shoulder")


class TestManifest:
    def test_every_required_predicate_exists(self):
        for leaf, man in load_all().items():
            for pred in list(man.qualify) + list(man.qualify_info):
                assert pred in PREDICATES, f"{leaf} names unknown predicate {pred!r}"

    def test_v10_records_the_physical_merge_check_without_gating_on_it(self):
        # This check lived only in qualify.py's own manifest while
        # `leaves/V-10.yaml` declared none — the drift that motivated
        # collapsing the two declarations. Pin it so it cannot vanish again.
        #
        # It is INFORMATION now, not a gate. The event V-10 builds is "a car
        # beside the ego takes its lane", which needs a lane alongside, not a
        # convergence the map has drawn — and gating on the convergence cost
        # two hosts that plainly carry the scenario (63c145828c3b5fd8 and
        # be36f75d360c502f, where the merge a reviewer can see is modelled as
        # parallel lanes) while adding nothing the cut-in depends on. What is
        # pinned is that the check is still COMPUTED and recorded beside the
        # scores, and that V-10 gates on nothing.
        manifest = load_all()["V-10"]
        assert manifest.qualify == []
        assert "merge_pressure" in manifest.qualify_info

    def test_oncoming_traffic_is_what_separates_c10_from_c7(self):
        # Both leaves want a two-way road, and the LOG decides which one the
        # host can carry. C-10 is judged on the ego driving the wrong way into
        # traffic that is already there, so a host without that traffic cannot
        # produce the crash and is refused. C-7 builds its own oncoming car, so
        # the same host is fine — and a host that already HAS the traffic is
        # C-10's, not C-7's, which is why C-7 only reports the check.
        empty = {
            v.leaf: v
            for v in qualify_host(
                _probe(_host_sd(opposing_y=3.5)), scene="syn", leaves=["C-10", "V-11", "C-7"]
            )
        }
        assert not empty["C-10"].ok
        assert empty["V-11"].ok and empty["C-7"].ok

        busy = {
            v.leaf: v
            for v in qualify_host(
                _probe(_host_sd(opposing_y=3.5, oncoming_track=True)),
                scene="syn",
                leaves=["C-10", "V-11", "C-7"],
            )
        }
        assert busy["C-10"].ok and busy["V-11"].ok and busy["C-7"].ok

    def test_c7_never_leaves_its_event_to_the_log(self):
        # The insert used to be conditional on the log lacking oncoming traffic,
        # which left C-7's one frozen recipe with zero actors. The head-on car
        # is the leaf; it is not optional, and the tier that made it optional
        # no longer exists.
        man = load_all()["C-7"]
        assert man.tier == "constructed"
        assert man.rule.cast and man.rule.cast[0].authored["blind_to_ego"] is True

    def test_every_leaf_that_supplies_its_own_event_is_deaf_to_the_ego(self):
        # IDM brakes rather than hits. An oncoming or mainline car handed the
        # ego as a leader yields, the shared strip of road is never contested,
        # and the leaf measures nothing.
        for leaf in ("C-7", "V-10"):
            for slot in load_all()[leaf].rule.cast:
                assert slot.authored.get("blind_to_ego") is True, \
                    f"{leaf}/{slot.slot} would yield to the ego"

    def test_one_way_host_is_rejected_for_the_wrong_way_leaves(self):
        verdicts = {
            v.leaf: v
            for v in qualify_host(_probe(_host_sd()), scene="syn", leaves=["C-10", "V-11"])
        }
        assert not verdicts["C-10"].ok and not verdicts["V-11"].ok

    def test_unknown_leaf_raises(self):
        with pytest.raises(KeyError, match="unknown leaf"):
            qualify_host(_probe(_host_sd()), scene="syn", leaves=["Z-99"])


class TestOpposingLaneChain:
    def test_chain_runs_the_opposing_direction(self):
        probe = _probe(_host_sd(opposing_y=3.5))
        chain = probe.opposing_lane_chain()
        # The ego drives +x; the opposing chain must run -x.
        assert chain.xy[-1, 0] < chain.xy[0, 0]

    def test_one_way_host_refuses_with_guidance(self):
        with pytest.raises(PlacementError, match="two-way host"):
            _probe(_host_sd()).opposing_lane_chain()

    def test_c10_template_semantics_bake_normal_oncoming_traffic(self):
        sd = _host_sd(opposing_y=3.5)
        spec = {
            "recipe_id": "C-10/wrong_way_crash/syn/001",
            "leaf": "C-10",
            "scenario": "wrong_way_crash",
            "host": {"scene": "syn", "world_version": "test@0"},
            "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
            "actors": {
                "oncoming_vehicle": {
                    "op": "insert",
                    "track": {"type": "VEHICLE"},
                    "asset": {"dims": [4.6, 1.9, 1.6]},
                    "authored": {
                        "template": "dynamic",
                        "reference": "opposing_lane_chain",
                        "arc": -60.0,
                        "speed": 10.0,
                    },
                }
            },
            "requires": {"min_reaction_s": 0.0},
            "pair": {"e0_removes": ["oncoming_vehicle"], "pair_id": "c10"},
        }
        recipe, _ = author_recipe(sd, spec, use_registry=False)
        actor = recipe.actors["oncoming_vehicle"]
        spawn = np.asarray(actor.spawn["position"], np.float64)
        # Spawns ego-forward on the opposing lane, correctly oriented, and is
        # handed an IDM path that runs its lane's own direction (-x) — so a
        # POSITIVE desired speed is ordinary oncoming traffic.
        assert spawn[0] > 40.0
        assert spawn[1] == pytest.approx(3.5, abs=0.5)
        assert abs(abs(float(actor.spawn["heading"])) - np.pi) < 0.2
        path = np.asarray(actor.policy["path_polyline"], np.float64)
        assert path[-1, 0] < path[0, 0]
        assert actor.policy["kind"] == "idm" and actor.policy["v0"] > 0.0


class TestLayoutAt:
    """Where a bank of dart-outs is laid out — the R-3 / R-4 crosswalk rule."""

    @staticmethod
    def _spec(at, *, names=("ped_1", "ped_2")):
        return {
            "recipe_id": "R-3/micromobility_crossing/syn/001",
            "leaf": "R-3",
            "scenario": "micromobility_crossing",
            "host": {"scene": "syn", "world_version": "test@0"},
            "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
            "layout": {"at": at, "span_m": 6.0, "margin_m": 2.0, "min_spacing_m": 1.0},
            "actors": {
                name: {
                    "op": "insert",
                    "track": {"type": "PEDESTRIAN"},
                    "asset": {"dims": [0.7, 0.7, 1.75]},
                    "authored": {
                        "template": "dart_out",
                        "reference": "ego_route",
                        "conflict_arc": "auto",
                        "start_lateral": -5.0,
                        "end_lateral": 5.0,
                        "speed": 1.4,
                    },
                }
                for name in names
            },
            "requires": {"min_reaction_s": 0.0},
            "pair": {"e0_removes": list(names), "pair_id": "r3"},
        }

    def _conflict_arcs(self, sd, at):
        recipe, _ = author_recipe(sd, self._spec(at), use_registry=False)
        return [float(a.authored["conflict_arc"]) for a in recipe.actors.values()]

    def test_the_crowd_crosses_at_the_crosswalk_when_the_host_has_one(self):
        sd = _host_sd(crosswalk_x=70.0)
        arcs = self._conflict_arcs(sd, "crosswalk_1|straight")
        # The window is CENTRED on the crosswalk, so the crowd straddles it.
        assert min(arcs) < 70.0 < max(arcs) + 6.0
        assert all(abs(arc - 70.0) < 10.0 for arc in arcs)

    def test_the_same_rule_falls_back_to_the_straight_stretch(self):
        # No crosswalk on this host: the identical spec still bakes, mid-block.
        arcs = self._conflict_arcs(_host_sd(), "crosswalk_1|straight")
        assert len(arcs) == 2 and arcs[0] < arcs[1]

    def test_the_fallback_is_not_silent_when_nothing_resolves(self):
        # A chain that does NOT end in `straight` must refuse rather than
        # quietly placing the crowd somewhere else — which host carries the
        # landmark is a selection finding.
        with pytest.raises(Exception, match="none of these resolve"):
            self._conflict_arcs(_host_sd(), "crosswalk_1|crosswalk_2")

    def test_default_is_the_straight_stretch(self):
        sd = _host_sd(crosswalk_x=70.0)
        assert self._conflict_arcs(sd, "straight") != self._conflict_arcs(
            sd, "crosswalk_1|straight"
        )
