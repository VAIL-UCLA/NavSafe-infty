# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Authoring: a short intent spec, resolved against a host, into a frozen recipe."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from navsafe.benchmark.editing.author import AuthoringError, author_recipe
from navsafe.benchmark.editing.ground_z import (
    CONSUMER_ENV,
    export_to_consumers,
    resolve_z_to_ground,
)
from navsafe.benchmark.editing.placement.probe import HostProbe, PlacementError
from navsafe.benchmark.editing.recipe import edits_from_recipe_file, freeze_recipe
from navsafe.scenario.edits import apply_scenario_edits

T = 60
DT = 0.1
EGO_Z = 55.0
Z_TO_GROUND = 1.4
ROAD_Z = EGO_Z - Z_TO_GROUND


def _lane(xs, ys, *, z=ROAD_Z):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return np.stack([xs, ys, np.full_like(xs, z)], axis=1)


def _host_sd(*, with_boxes: bool = True) -> dict:
    """Ego heading east; a three-lane chain along it, and a cross street north."""
    pos = np.zeros((T, 3), np.float64)
    pos[:, 0] = np.arange(T) * 1.5  # 15 m/s -> 90 m of route
    pos[:, 2] = EGO_Z
    seg = lambda a, b: _lane(np.arange(a, b + 1e-9, 2.0), np.zeros(len(np.arange(a, b + 1e-9, 2.0))))
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
        },
        "host_car": {
            "type": "VEHICLE",
            "state": {
                "position": np.tile([70.0, 3.5, ROAD_Z + 0.9], (T, 1)),
                "length": np.full(T, 4.6, np.float32),
                "width": np.full(T, 1.9, np.float32),
                "height": np.full(T, 1.8, np.float32),
                "heading": np.zeros(T),
                "velocity": np.zeros((T, 2)),
                "valid": np.ones(T, bool),
            },
            "metadata": {"object_id": "host_car"},
        },
    }
    if not with_boxes:
        for key in ("length", "width", "height"):
            tracks["host_car"]["state"].pop(key)
    cross_ys = np.arange(-40.0, 40.0, 2.0)
    return {
        "metadata": {
            "sdc_id": "ego",
            "ts": (np.arange(T) * DT * 1e6).astype(np.float64),
            "coordinate": "local_frame0",
            "route_lane_ids": ["lane_b"],
        },
        "tracks": tracks,
        "map_features": {
            "lane_a": {"type": "LANE_SURFACE_STREET", "polyline": seg(-60.0, -20.0),
                       "exit_lanes": ["lane_b"]},
            "lane_b": {
                "type": "LANE_SURFACE_STREET",
                "polyline": seg(-20.0, 60.0),
                "entry_lanes": ["lane_a"],
                "exit_lanes": ["lane_c"],
                "left_boundaries": np.stack([np.arange(-20.0, 61.0, 2.0),
                                             np.full(41, 1.75)], axis=1),
                "right_boundaries": np.stack([np.arange(-20.0, 61.0, 2.0),
                                              np.full(41, -1.75)], axis=1),
            },
            "lane_c": {"type": "LANE_SURFACE_STREET", "polyline": seg(60.0, 160.0),
                       "entry_lanes": ["lane_b"]},
            "cross": {
                "type": "LANE_SURFACE_STREET",
                "polyline": _lane(np.full(len(cross_ys), 45.0), cross_ys),
            },
            # A lane centreline whose height column is all zeros — the shape
            # this data actually ships.
            "flat_lane": {
                "type": "LANE_SURFACE_STREET",
                "polyline": _lane(np.arange(0.0, 80.0, 2.0), np.full(40, -3.5), z=0.0),
            },
        },
    }


def _spec(**overrides) -> dict:
    spec = {
        "recipe_id": "C-10/wrong_way_approach/synthetic/001",
        "leaf": "C-10",
        "scenario": "wrong_way_approach",
        "host": {"scene": "synthetic", "world_version": "test@0"},
        "ego": {"replay_frames": 8, "cam_height": "waymo", "z_to_ground": Z_TO_GROUND},
        "actors": {
            "oncoming_vehicle": {
                "op": "relocate",
                "source_track_id": "host_car",
                "keep_appearance": True,
                "track": {"type": "VEHICLE"},
                "asset": {"registry_key": "host_vehicle", "dims": [4.6, 1.9, 1.8]},
                "authored": {
                    "template": "dynamic",
                    "reference": "ego_lane_chain",
                    "arc": 40.0,
                    "speed": -8.0,
                },
            }
        },
        "pair": {"e0_removes": ["oncoming_vehicle"], "pair_id": "c10_001"},
        "review": {"status": "pending", "checklist": "leaves/C-10.md"},
    }
    spec.update(overrides)
    return spec


def _author(sd, spec, **kwargs):
    """Author without the asset registry — these tests are about geometry.

    Registry resolution has its own tests in ``test_assets.py``; wiring it in
    here would make every trajectory test depend on which PLYs happen to be on
    disk.
    """
    kwargs.setdefault("use_registry", False)
    return author_recipe(sd, spec, **kwargs)


class TestAuthorEndToEnd:
    def test_no_arrival_window_is_claimed_any_more(self):
        # It used to be solved from the baked trajectory against the LOG-REPLAY
        # ego, and carried on the Recipe. Both sides of that comparison are
        # gone: the actor reacts, and the ego is the policy under test. Claiming
        # a window would be claiming a rollout that will not happen — "when does
        # the ego get there" is a fact about the episode trace, not a property
        # of the file, so the field does not exist.
        recipe, _ = _author(_host_sd(), _spec())
        assert not hasattr(recipe, "assumed_ego_arrival_window_s")
        assert "assumed_ego_arrival_window_s" not in recipe.to_dict()

    def test_c10_authors_freezes_and_replays(self, tmp_path):
        sd = _host_sd()
        recipe, diagnostics = _author(sd, _spec())
        assert recipe.frames.T == T
        assert recipe.frames.after_frame == 8
        assert list(recipe.actors) == ["oncoming_vehicle"]
        assert recipe.route_polyline  # the reference the arcs were measured against

        path = freeze_recipe(recipe, tmp_path / "c10.yaml")
        _, edits = edits_from_recipe_file(path)
        out = apply_scenario_edits(_host_sd(), edits)
        # A relocate re-tasks the host's actor, so its logged track is gone and
        # the recipe's actor stands in its place.
        assert "host_car" not in out["tracks"]
        state = out["tracks"]["navsafe_oncoming_vehicle"]["state"]
        position = np.asarray(state["position"])
        ego = np.asarray(out["tracks"]["ego"]["state"]["position"])
        # The DECLARATION holds the spawn pose for the episode; the closing is
        # the policy's job at run time, so what is checked here is that the
        # actor starts ahead of the ego, facing it.
        assert position[0, 0] > ego[0, 0]
        assert float(np.asarray(state["heading"])[0]) == pytest.approx(np.pi, abs=1e-3)
        spec_actor = out["metadata"]["navsafe_reactive"][0]
        assert spec_actor["policy"]["kind"] == "idm"
        assert diagnostics["actors"]["oncoming_vehicle"]["speed_range_mps"][0] == pytest.approx(8.0)

    def test_road_height_is_cross_checked_against_the_host(self, caplog):
        spec = _spec()
        spec["ego"]["z_to_ground"] = 5.0  # nonsense on purpose
        with caplog.at_level("WARNING"):
            _, diagnostics = _author(_host_sd(), spec)
        assert diagnostics["host"]["road_drift_measured"]["samples"] > 0
        assert "recon-road drift" in caplog.text

    def test_a_matching_road_height_warns_about_nothing(self, caplog):
        # The fixture puts the actor boxes' road exactly Z_TO_GROUND below the
        # ego z, which is what the spec declares — so there is nothing to say.
        with caplog.at_level("WARNING"):
            _author(_host_sd(), _spec())
        assert "recon-road drift" not in caplog.text

    def test_no_actors_is_refused(self):
        with pytest.raises(AuthoringError, match="no actors"):
            _author(_host_sd(), _spec(actors={}))


class TestTimingIntent:
    def test_arrive_with_ego_solves_a_cross_street_dart(self):
        spec = _spec()
        spec["actors"] = {
            "cyclist": {
                "op": "insert",
                "track": {"type": "CYCLIST"},
                "asset": {"registry_key": "bicycle_rider", "dims": [1.8, 0.6, 1.7]},
                "authored": {
                    "template": "dart_out",
                    "reference": "lane:cross",
                    "start_lateral": -6.0,
                    "end_lateral": 6.0,
                    "speed": 5.0,
                    "arrive_with_ego": True,
                },
            }
        }
        spec["pair"] = {"e0_removes": ["cyclist"], "pair_id": "x"}
        recipe, diagnostics = _author(_host_sd(), spec)
        solved = recipe.actors["cyclist"].authored["solved"]
        # The ego crosses x = 45 at 15 m/s -> frame 30.
        assert solved["conflict_xy"] == [45.0, 0.0]
        assert solved["ego_arrival_frame"] == 30
        assert recipe.actors["cyclist"].authored["conflict_frame"] == 30
        # …and the cyclist is aimed ACROSS the ego's path. Where it actually
        # is at frame 30 is no longer decided here — a social-force actor gets
        # there when it gets there, which is the point of the change.
        actor = recipe.actors["cyclist"]
        assert actor.policy["kind"] == "social_force"
        goal = np.asarray(actor.policy["goal"], float)
        spawn = np.asarray(actor.spawn["position"], float)[:2]
        assert float(np.linalg.norm(goal - spawn)) > 1.0
        assert diagnostics["actors"]["cyclist"]["template"] == "dart_out"

    def test_arrive_with_ego_needs_a_crossing_or_a_merge(self):
        # A parallel lane is NOT a conflict however close it runs: there is no
        # moment at which the two arrive anywhere together, so there is nothing
        # to time against. Only a CONVERGING pair gets the merge fallback.
        spec = _spec()
        spec["actors"]["oncoming_vehicle"]["authored"]["arrive_with_ego"] = True
        with pytest.raises(AuthoringError, match="CONVERGES"):
            _author(_host_sd(), spec)


class TestActorShapes:
    def test_static_actor_stores_one_pose(self, tmp_path, monkeypatch):
        spec = _spec()
        spec["actors"] = {
            "sign": {
                "op": "insert",
                "track": {"type": "TRAFFIC_CONE"},
                "asset": {"registry_key": "one_way_sign", "dims": [0.15, 0.66, 2.0]},
                "nurec_asset_id": "/assets/one_way_3dgs.ply",
                "authored": {
                    "template": "static",
                    "reference": "ego_route",
                    "arc": 30.0,
                    "lateral": -4.0,
                    "yaw_offset_deg": 180.0,
                },
            }
        }
        spec["pair"] = {"e0_removes": ["sign"], "pair_id": "x"}
        recipe, _ = _author(_host_sd(), spec)
        # A thing that never moves is a POLICY, not a file-format special case.
        assert recipe.actors["sign"].policy["kind"] == "static"
        assert np.asarray(recipe.actors["sign"].spawn["position"]).shape == (3,)

        path = freeze_recipe(recipe, tmp_path / "sign.yaml")
        # Replay refuses an unset NAVSAFE_ASSET_BANK -- the frozen
        # "/assets/..." above is the authoring machine's -- so stand in for the
        # reader's unpacked bank and let the asset rebase onto it.
        import navsafe.benchmark.config as cfg
        bank = tmp_path / "bank"
        bank.mkdir()
        (bank / "one_way_3dgs.ply").write_bytes(b"ply")
        monkeypatch.setattr(cfg, "ASSET_BANK", bank)
        _, edits = edits_from_recipe_file(path)
        out = apply_scenario_edits(_host_sd(), edits)
        track = out["tracks"]["navsafe_sign"]
        assert np.asarray(track["state"]["position"]).shape == (T, 3)
        assert track["metadata"]["nurec_asset_id"] == str(bank / "one_way_3dgs.ply")

    def test_remove_needs_no_geometry(self):
        spec = _spec()
        spec["actors"]["clutter"] = {"op": "remove", "source_track_id": "host_car"}
        spec["actors"].pop("oncoming_vehicle")
        spec["pair"] = {"e0_removes": ["clutter"], "pair_id": "x"}
        recipe, _ = _author(_host_sd(), spec)
        assert recipe.actors["clutter"].op == "remove"
        assert recipe.actors["clutter"].spawn == {} and recipe.actors["clutter"].policy == {}


class TestReferences:
    def test_lane_chain_extends_past_the_route(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        probe.after_frame = 8
        chain = probe.ego_lane_chain(min_length_m=150.0)
        # lane_a + lane_b + lane_c, i.e. the whole road, not just the 80 m lane.
        assert chain.total > probe.ego_route.total
        assert chain.total == pytest.approx(220.0, abs=1.0)

    def test_a_zero_height_lane_borrows_the_road_surface(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        flat = probe.lane("flat_lane")
        # The map says z = 0, which means "no height was written", not "the
        # road is at 0" — the ego's own road surface is used instead.
        assert float(flat.z[5]) == pytest.approx(ROAD_Z, abs=1e-6)

    def test_route_lanes_refuses_a_lane_change_seam(self):
        sd = _host_sd()
        sd["metadata"]["route_lane_ids"] = ["lane_b", "flat_lane"]  # parallel, not successive
        probe = HostProbe(sd, ego_z_to_ground_m=Z_TO_GROUND)
        with pytest.raises(PlacementError, match="lane CHANGE"):
            probe.route_lanes()

    def test_route_lanes_accepts_a_genuine_chain(self):
        sd = _host_sd()
        sd["metadata"]["route_lane_ids"] = ["lane_a", "lane_b", "lane_c"]
        probe = HostProbe(sd, ego_z_to_ground_m=Z_TO_GROUND)
        assert probe.route_lanes().total == pytest.approx(220.0, abs=1.0)

    def test_road_drift_unmeasurable_without_boxes(self):
        probe = HostProbe(_host_sd(with_boxes=False), ego_z_to_ground_m=Z_TO_GROUND)
        assert probe.measure_road_drift() is None

    def test_road_drift_reads_the_actor_boxes(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        measured = probe.measure_road_drift()
        # host_car's box centre sits 0.9 m above ROAD_Z with height 1.8, so its
        # base IS the road, and the residual is the fixture's pose lift.
        assert measured["actor_road_median"] == pytest.approx(ROAD_Z, abs=1e-3)
        assert measured["residual_m"] == pytest.approx(Z_TO_GROUND, abs=1e-3)


class TestGroundZResolution:
    """The one resolver every road-height consumer shares."""

    def test_explicit_wins(self, tmp_path):
        value, reason = resolve_z_to_ground(0.25, tmp_path / "root")
        assert (value, reason) == (0.25, "explicit")

    def test_registry_is_keyed_by_the_data_root_parent(self, tmp_path):
        root = tmp_path / "scene_abc" / "arrow"
        root.mkdir(parents=True)
        registry = tmp_path / "calib.json"
        registry.write_text(json.dumps({"scene_abc": -0.32}))
        value, reason = resolve_z_to_ground(None, root, registry_path=registry)
        assert value == pytest.approx(-0.32)
        assert "scene_abc" in reason

    def test_default_is_zero_because_the_ego_z_is_the_ground(self, tmp_path):
        value, reason = resolve_z_to_ground(None, tmp_path / "unlisted" / "arrow")
        assert value == 0.0
        assert "ground-referenced" in reason

    def test_an_unreadable_registry_falls_back_loudly(self, tmp_path, caplog):
        registry = tmp_path / "broken.json"
        registry.write_text("{not json")
        with caplog.at_level("WARNING"):
            value, _ = resolve_z_to_ground(None, tmp_path / "x" / "arrow", registry_path=registry)
        assert value == 0.0
        assert "unreadable" in caplog.text

    def test_consumers_all_get_the_same_number(self, monkeypatch):
        # Swap in a copy of the environment: ``export_to_consumers`` writes
        # through ``os.environ.setdefault``, which monkeypatch cannot track and
        # would otherwise leak the value into every later test in the session.
        monkeypatch.setattr(os, "environ", dict(os.environ))
        for name in CONSUMER_ENV:
            os.environ.pop(name, None)
        export_to_consumers(-0.4)
        assert {os.environ[name] for name in CONSUMER_ENV} == {"-0.4"}

    def test_an_operator_pin_is_not_overwritten(self, monkeypatch):
        monkeypatch.setattr(os, "environ", dict(os.environ))
        os.environ[CONSUMER_ENV[0]] = "0.99"
        export_to_consumers(-0.4)
        assert os.environ[CONSUMER_ENV[0]] == "0.99"


class TestSpeedIsRelativeToThisHostsEgo:
    """A speed that has to be slower than the ego cannot be an absolute number.

    R-2 stages an overtake, and shipped with ``speed: [5.0, 4.4]`` — honest
    urban cycling speeds that were at or above the ego's own on every host in
    the set (measured 2.4-8.2 m/s, median 5.3). The riders drew away from the
    hand-off and the ego finished the episode never having seen them.
    """

    def _ride(self, **authored):
        spec = _spec(actors={"rider": {
            "op": "insert",
            "track": {"type": "CYCLIST"},
            "asset": {"registry_key": "bike", "dims": [1.8, 0.6, 1.7]},
            "authored": {"template": "dynamic", "reference": "ego_lane_chain",
                         "arc": 20.0, **authored},
        }}, pair={"e0_removes": ["rider"], "pair_id": "r2_001"})
        recipe, _ = _author(_host_sd(), spec)
        policy = recipe.to_dict()["actors"]["rider"]["policy"]
        # A cyclist rides a lane, so `dynamic` gives it IDM, whose desired
        # speed is `v0`; a walker gets social force, whose is `desired_speed`.
        return policy["v0"] if "v0" in policy else policy["desired_speed"]

    def test_the_fixture_ego_cruises_at_fifteen(self):
        probe = HostProbe(_host_sd())
        assert probe.ego_cruise_speed() == pytest.approx(15.0, abs=0.01)

    def test_a_factor_of_the_ego_s_own_pace(self):
        assert self._ride(speed="ego*0.6") == pytest.approx(9.0)

    def test_a_deficit_from_the_ego_s_own_pace(self):
        # "The ego closes at 2 m/s" is the statement an overtake leaf wants to
        # make, and it survives a host whose ego drives at a different speed.
        assert self._ride(speed="ego-2.0") == pytest.approx(13.0)

    def test_bare_ego_matches_it(self):
        assert self._ride(speed="ego") == pytest.approx(15.0)

    def test_the_band_the_actor_is_physically_capable_of_wins(self):
        # 0.6 x 15 m/s is 9 m/s, which is not a bicycle. The clamp is what
        # keeps an arithmetic result a real rider.
        got = self._ride(speed="ego*0.6", speed_max=6.5)
        assert got == pytest.approx(6.5)

    def test_a_crawling_ego_does_not_produce_a_crawling_rider(self):
        sd = _host_sd()
        sd["tracks"]["ego"]["state"]["position"][:, 0] = np.arange(T) * 0.24   # 2.4 m/s
        spec = _spec(actors={"rider": {
            "op": "insert",
            "track": {"type": "CYCLIST"},
            "asset": {"registry_key": "bike", "dims": [1.8, 0.6, 1.7]},
            "authored": {"template": "dynamic", "reference": "ego_lane_chain",
                         "arc": 5.0, "speed": "ego*0.65", "speed_min": 2.0},
        }}, pair={"e0_removes": ["rider"], "pair_id": "r2_002"})
        recipe, _ = _author(sd, spec)
        assert recipe.to_dict()["actors"]["rider"]["policy"]["v0"] == pytest.approx(2.0)

    def test_an_absolute_speed_still_works(self):
        assert self._ride(speed=4.4) == pytest.approx(4.4)

    def test_an_unreadable_expression_is_refused(self):
        with pytest.raises(AuthoringError, match="ego"):
            self._ride(speed="fast")
        with pytest.raises(AuthoringError, match="ego"):
            self._ride(speed="ego/2")


class TestLayoutWindow:
    """Where an ``arc: auto`` group lands, and in whose coordinates."""

    def _arcs(self, layout, **authored):
        actors = {
            f"a{i}": {
                "op": "insert",
                "track": {"type": "VEHICLE"},
                "asset": {"registry_key": "car", "dims": [4.6, 1.9, 1.8]},
                "authored": {"template": "static", "reference": "ego_route",
                             "arc": "auto", **authored},
            } for i in (1, 2)
        }
        spec = _spec(actors=actors, layout=layout,
                     pair={"e0_removes": ["a1"], "pair_id": "lay_001"})
        recipe, _ = _author(_host_sd(), spec)
        # Ego heads east from the origin, so its x IS its route arc.
        return sorted(a["spawn"]["position"][0]
                      for a in recipe.to_dict()["actors"].values())

    def test_handoff_puts_the_group_beside_the_ego(self):
        # The hand-off is frame 8, i.e. 12 m along. `straight` scans forward for
        # a junction-free stretch and can start anywhere; `handoff` says the
        # group's business is with the ego, so it starts where the ego is.
        near, far = self._arcs({"at": "handoff", "span_m": 16.0, "margin_m": 10.0})
        assert near == pytest.approx(12.0 + 10.0, abs=0.5)
        assert far == pytest.approx(12.0 + 10.0 + 16.0, abs=0.5)

    def test_the_hand_off_is_not_counted_twice(self):
        # `absolute_arc = anchor_arc + arc` and `anchor_arc` is already the
        # hand-off on the actor's reference, so a layout arc has to be
        # RELATIVE. Returning the absolute route arc put the group at
        # 2 x 12 + margin.
        near, _ = self._arcs({"at": "handoff", "span_m": 16.0, "margin_m": 0.0})
        assert near == pytest.approx(12.0, abs=0.5)

    def test_an_unknown_window_names_what_the_host_offers(self):
        with pytest.raises(AuthoringError, match="crosswalk_9"):
            self._arcs({"at": "crosswalk_9", "span_m": 16.0, "margin_m": 4.0})


class TestClearPath:
    """The logged traffic between the ego and a STATIC insert.

    Recorded traffic was driving a road with no incident on it, so it passes
    through an inserted wreck — and hides it on the way, which is how
    03b66343e1ac5d68 produced an ego that met the blockage only after a phantom
    lead car had driven over it.
    """

    def _spec_with(self, clear=None, arc=60.0):
        # `arc` is measured from the hand-off (route arc 12 on this fixture),
        # and `host_car` stands at route arc 70.
        actors = {
            "blocker_1": {
                "op": "insert",
                "track": {"type": "VEHICLE"},
                "asset": {"registry_key": "car", "dims": [4.6, 1.9, 1.8]},
                "authored": {"template": "static", "reference": "ego_route", "arc": arc},
            }
        }
        spec = _spec(actors=actors, pair={"e0_removes": ["blocker_1"], "pair_id": "i3_001"})
        if clear is not None:
            spec["clear_path"] = clear
        return spec

    def test_a_car_standing_on_the_wreck_is_removed(self):
        # `host_car` sits at x=70 — 10 m past the blocker, inside the margin.
        recipe, _ = _author(_host_sd(), self._spec_with({"corridor_m": 4.5, "margin_m": 12.0}))
        removed = {a["source_track_id"] for a in recipe.to_dict()["actors"].values()
                   if a["op"] == "remove"}
        assert removed == {"host_car"}

    def test_without_the_declaration_nothing_is_removed(self):
        # The phantom-lead-car problem belongs to leaves whose insert is
        # static; a leaf that does not ask keeps its host's traffic.
        recipe, _ = _author(_host_sd(), self._spec_with(None))
        assert not [a for a in recipe.to_dict()["actors"].values() if a["op"] == "remove"]

    def test_traffic_in_the_next_lane_is_left_alone(self):
        # Going round the wreck is only a decision if there is traffic to
        # judge; a corridor that swallows the neighbouring lane removes it.
        recipe, _ = _author(_host_sd(), self._spec_with({"corridor_m": 2.0, "margin_m": 12.0}))
        assert not [a for a in recipe.to_dict()["actors"].values() if a["op"] == "remove"]

    def test_traffic_past_the_margin_is_left_alone(self):
        # Blocker at route arc 12+18=30, margin 2 -> the window ends at 32 and
        # `host_car` at 70 is well beyond the incident, not standing on it.
        recipe, _ = _author(
            _host_sd(), self._spec_with({"corridor_m": 4.5, "margin_m": 2.0}, arc=18.0))
        assert not [a for a in recipe.to_dict()["actors"].values() if a["op"] == "remove"]

    def test_the_resolved_ids_are_recorded(self):
        # A frozen recipe rebuilds one exact scene, so the detection runs once
        # at authoring time and its answer is written down.
        _, diag = _author(_host_sd(), self._spec_with({"corridor_m": 4.5, "margin_m": 12.0}))
        assert diag["clear_path"]["removed"][0]["track_id"] == "host_car"


class TestHowFarAheadIsATime:
    """`margin_m` alone makes a static insert a different test on every host.

    I-3 declares `min_reaction_s: 2.0` and laid its wrecks 22 m ahead. That is
    3.7 s at 6 m/s and 1.5 s at 14 m/s, and on 76da778ff251508d the ego hit
    them at frame 27 with no room to have done anything else. Nothing read
    `requires.min_reaction_s` — six leaves declared it and it gated nothing.
    """

    def _blocker_arc(self, ego_mps, *, margin_m=22.0, reaction_s=None):
        sd = _host_sd()
        sd["tracks"]["ego"]["state"]["position"][:, 0] = np.arange(T) * ego_mps * DT
        spec = _spec(
            actors={"blocker": {
                "op": "insert",
                "track": {"type": "VEHICLE"},
                "asset": {"registry_key": "car", "dims": [4.6, 1.9, 1.8]},
                "authored": {"template": "static", "reference": "ego_route", "arc": "auto"},
            }},
            layout={"at": "handoff", "span_m": 8.0, "margin_m": margin_m},
            pair={"e0_removes": ["blocker"], "pair_id": "i3_001"})
        if reaction_s is not None:
            spec["requires"] = {"min_reaction_s": reaction_s}
        recipe, _ = _author(sd, spec)
        # Ego heads east from the origin, so x IS the route arc; the hand-off
        # is frame 8.
        handoff = 8 * ego_mps * DT
        return recipe.to_dict()["actors"]["blocker"]["spawn"]["position"][0] - handoff

    def test_a_fast_ego_gets_the_seconds_the_leaf_asked_for(self):
        gap = self._blocker_arc(14.0, reaction_s=2.0)
        assert gap / 14.0 == pytest.approx(2.0, abs=0.15)
        assert gap > 22.0                      # the metres were not enough here

    def test_a_slow_ego_keeps_the_metres_as_a_floor(self):
        # 22 m is already 3.7 s at 6 m/s; asking for 2 s must not pull the
        # wreck closer than the leaf's own stated distance.
        gap = self._blocker_arc(6.0, reaction_s=2.0)
        assert gap == pytest.approx(22.0, abs=0.6)

    def test_without_a_declared_reaction_the_metres_stand(self):
        gap = self._blocker_arc(14.0, reaction_s=None)
        assert gap == pytest.approx(22.0, abs=0.6)


class TestACrossingTooCloseToReactTo:
    """The reaction filter has to use a RATE, not one frame's difference.

    On 02379e524f105926 the ego is slow at the hand-off frame (1.65 m/s) and
    cruises at 6.96; the one-frame speed shrank a 1.5 s window to 2.5 m, a
    crossing 3.6 m ahead survived it, and R-3 put four walkers where the ego
    reached them 0.9 s after taking over.
    """

    @staticmethod
    def _host_with_crossing_at(x, *, crawl_at_handoff):
        sd = _host_sd()
        pos = sd["tracks"]["ego"]["state"]["position"]
        pos[:, 0] = np.arange(T) * 0.7                      # 7 m/s cruise
        if crawl_at_handoff:
            # Creeping through the hand-off, then back up to cruise — exactly
            # the shape that fools a one-frame speed.
            pos[8:10, 0] = pos[8, 0] + np.arange(2) * 0.165
            pos[10:, 0] = pos[9, 0] + np.arange(T - 10) * 0.7
        ys = np.arange(-8.0, 8.0, 2.0)
        sd["map_features"]["xwalk"] = {
            "type": "CROSSWALK",
            "polyline": np.stack([np.full(len(ys), float(x)), ys,
                                  np.full(len(ys), ROAD_Z)], axis=1),
        }
        return sd

    def _arcs(self, sd):
        probe = HostProbe(sd)
        probe.after_frame = 8
        return [c["arc_m"] for c in probe.crosswalks_on_route(8)]

    def test_a_crossing_inside_the_reaction_window_is_dropped(self):
        # Hand-off at x=5.6; a crossing at 9 m is 3.4 m ahead, under the 10.5 m
        # that 1.5 s at 7 m/s needs.
        assert self._arcs(self._host_with_crossing_at(9.0, crawl_at_handoff=True)) == []

    def test_a_crossing_beyond_it_survives(self):
        arcs = self._arcs(self._host_with_crossing_at(40.0, crawl_at_handoff=True))
        assert len(arcs) == 1 and arcs[0] > 10.0

    def test_the_crawl_does_not_change_the_answer(self):
        # The whole point: the same road gives the same verdict whether or not
        # the ego happened to be creeping through the hand-off frame.
        assert (self._arcs(self._host_with_crossing_at(9.0, crawl_at_handoff=True))
                == self._arcs(self._host_with_crossing_at(9.0, crawl_at_handoff=False)))


class TestACrossingHasToCrossTheEgosLine:
    """A junction's other arm carries crossings the ego never drives over.

    On 225eb6e22af55972 the nearest crossing by arc runs from 16.6 m to 6.8 m
    LEFT of the route -- entirely to one side. It was picked anyway (its centre
    is inside `max_lateral_m`), R-3's walkers were sent along it, and
    `arrive_with_ego` reported that their path never meets the ego's: an
    accurate message about the wrong crossing, on a host with three good ones.
    """

    @staticmethod
    def _host(*crossings):
        sd = _host_sd()
        sd["tracks"]["ego"]["state"]["position"][:, 0] = np.arange(T) * 0.7
        for i, (x, y0, y1) in enumerate(crossings):
            ys = np.linspace(y0, y1, 8)
            sd["map_features"][f"xwalk_{i}"] = {
                "type": "CROSSWALK",
                "polyline": np.stack([np.full(len(ys), float(x)), ys,
                                      np.full(len(ys), ROAD_Z)], axis=1),
            }
        probe = HostProbe(sd)
        probe.after_frame = 8
        return [c["feature_id"] for c in probe.crosswalks_on_route(8)]

    def test_one_that_straddles_the_route_is_kept(self):
        # The ego runs along y=0, so -6..+6 spans it.
        assert self._host((40.0, -6.0, 6.0)) == ["xwalk_0"]

    def test_one_wholly_to_one_side_is_dropped(self):
        # Centre at +11 m is inside `max_lateral_m` and it is still a crossing
        # on another road.
        assert self._host((40.0, 6.0, 16.0)) == []

    def test_the_straddling_one_wins_even_when_it_is_further_along(self):
        near_but_beside, far_but_crossing = (30.0, 7.0, 17.0), (50.0, -6.0, 6.0)
        assert self._host(near_but_beside, far_but_crossing) == ["xwalk_1"]


class TestMergePressureHasTwoCauses:
    """A merge the map draws, and a merge the cones make.

    nuPlan's lane graph does not know a lane is coned off, so a contraflow
    reads as an ordinary multi-lane road: `merging_lane` finds nothing and
    three reviewer-picked hosts were refused as "no merge here" while the
    recorded ego was plainly threading a closure. The closure is in the data as
    one track per cone and per barrier.
    """

    @staticmethod
    def _host(*, cones=0, cone_lat=5.0, cone_from=20.0):
        sd = _host_sd()
        sd["tracks"]["ego"]["state"]["position"][:, 0] = np.arange(T) * 0.8
        for i in range(cones):
            x = cone_from + i * 4.0
            sd["tracks"][f"cone_{i}"] = {
                "type": "TRAFFIC_CONE",
                "state": {
                    "position": np.tile([x, cone_lat, ROAD_Z], (T, 1)),
                    "heading": np.zeros(T), "velocity": np.zeros((T, 2)),
                    "valid": np.ones(T, bool),
                },
                "metadata": {"object_id": f"cone_{i}"},
            }
        probe = HostProbe(sd)
        probe.after_frame = 8
        return probe

    def test_a_bare_road_offers_no_merge_of_either_kind(self):
        assert self._host().merge_pressure_lane() is None

    def test_a_few_cones_are_litter_not_a_closure(self):
        assert self._host(cones=3).workzone_pinch() is None

    def test_a_run_of_them_is_a_closure(self):
        pinch = self._host(cones=8).workzone_pinch()
        assert pinch is not None
        assert pinch["count"] >= 5
        assert pinch["side"] > 0            # they are to the ego's left

    def test_the_closure_names_its_cause(self):
        found = self._host(cones=8).merge_pressure_lane()
        assert found is not None and found["cause"] == "workzone"

    def test_a_map_merge_still_wins_when_there_is_one(self):
        # `_host_sd` has no converging lane, so this asserts the ORDER rather
        # than the geometry: a host with both is a map merge, because that is
        # the lane the map itself says joins.
        probe = self._host(cones=8)
        probe.merging_lane = lambda *a, **k: {
            "lane_id": "lane_b", "far_m": 3.0, "near_m": 0.5, "merge_arc_m": 40.0}
        found = probe.merge_pressure_lane()
        assert found["cause"] == "map_merge" and found["lane_id"] == "lane_b"

    def test_the_egos_own_lane_is_never_the_answer(self):
        # A lane within half a lane-width of the route IS the ego's, whatever
        # the map calls it; an actor placed there is placed on top of the ego.
        found = self._host(cones=8, cone_lat=1.0).merge_pressure_lane()
        assert found is None or abs(found["detail"]["lateral_m"]) >= 2.0
