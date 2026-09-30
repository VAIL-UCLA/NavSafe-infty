# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Semantic anchors: landmarks detected from the host's own lane geometry."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.author import author_recipe
from navsafe.benchmark.editing.placement.anchors import (
    find_anchors,
    is_anchor_name,
    resolve_anchor,
)
from navsafe.benchmark.editing.placement.probe import HostProbe, PlacementError

T = 80
DT = 0.1
EGO_Z = 10.0
Z_TO_GROUND = 1.4
ROAD_Z = EGO_Z - Z_TO_GROUND


def _lane(xs, ys, *, z=ROAD_Z):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return np.stack([xs, ys, np.full_like(xs, z)], axis=1)


def _seg(a, b, *, y=0.0):
    xs = np.arange(a, b + 1e-9, 2.0)
    return _lane(xs, np.full(len(xs), y))


def _host_sd(*, cross_at=(50.0,), with_crosswalk: bool = False) -> dict:
    """Ego drives east 120 m; a north-south street crosses at each ``cross_at``."""
    pos = np.zeros((T, 3), np.float64)
    pos[:, 0] = np.arange(T) * 1.5  # 15 m/s
    pos[:, 2] = EGO_Z
    map_features = {
        "lane_ego": {"type": "LANE_SURFACE_STREET", "polyline": _seg(-20.0, 140.0)},
        # A parallel neighbour must NOT read as a crossing.
        "lane_parallel": {"type": "LANE_SURFACE_STREET", "polyline": _seg(-20.0, 140.0, y=3.5)},
    }
    for i, x in enumerate(cross_at):
        ys = np.arange(-40.0, 40.0, 2.0)
        map_features[f"cross_{i}"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": _lane(np.full(len(ys), float(x)), ys),
        }
    if with_crosswalk:
        map_features["cw_1"] = {
            "type": "CROSSWALK",
            "polyline": _lane(np.full(5, float(cross_at[0]) - 6.0), np.linspace(-4, 4, 5)),
        }
    return {
        "metadata": {
            "sdc_id": "ego",
            "ts": (np.arange(T) * DT * 1e6).astype(np.float64),
            "coordinate": "local_frame0",
            "route_lane_ids": ["lane_ego"],
        },
        "tracks": {
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
        },
        "map_features": map_features,
    }


def _probe(sd, after_frame: int = 8) -> HostProbe:
    probe = HostProbe(sd, ego_z_to_ground_m=Z_TO_GROUND)
    probe.after_frame = after_frame
    return probe


class TestDetection:
    def test_one_cross_street_becomes_entry_and_exit(self):
        anchors = {a.name: a for a in find_anchors(_probe(_host_sd()))}
        assert "handoff" in anchors and "route_end" in anchors
        assert "intersection_1_entry" in anchors and "intersection_1_exit" in anchors
        # Hand-off is at x = 8 * 1.5 = 12 m; the cross street at x = 50, so the
        # junction straddles arc ~38 from the hand-off (± the 8 m merge/lateral
        # reach of the detector).
        entry, exit_ = anchors["intersection_1_entry"], anchors["intersection_1_exit"]
        assert entry.arc_m < exit_.arc_m
        assert 25.0 <= entry.arc_m <= 40.0
        assert 36.0 <= exit_.arc_m <= 55.0

    def test_parallel_lane_is_not_an_intersection(self):
        anchors = find_anchors(_probe(_host_sd(cross_at=())))
        assert [a.kind for a in anchors] == ["handoff", "route_end"]

    def test_two_junctions_number_in_route_order(self):
        anchors = {a.name: a for a in find_anchors(_probe(_host_sd(cross_at=(40.0, 90.0))))}
        assert anchors["intersection_1_entry"].arc_m < anchors["intersection_2_entry"].arc_m

    def test_crosswalk_feature_is_surfaced_when_present(self):
        anchors = {a.name: a for a in find_anchors(_probe(_host_sd(with_crosswalk=True)))}
        assert "crosswalk_1" in anchors
        assert anchors["crosswalk_1"].arc_m < anchors["intersection_1_entry"].arc_m + 8.0

    def test_anchor_arcs_follow_the_handoff_convention(self):
        # Same host, later hand-off => every arc shrinks by the difference.
        early = {a.name: a.arc_m for a in find_anchors(_probe(_host_sd(), after_frame=0))}
        late = {a.name: a.arc_m for a in find_anchors(_probe(_host_sd(), after_frame=20))}
        shift = early["intersection_1_entry"] - late["intersection_1_entry"]
        assert shift == pytest.approx(20 * 1.5, abs=0.5)


class TestResolution:
    def test_aliases_resolve(self):
        probe = _probe(_host_sd())
        assert resolve_anchor(probe, "first_intersection_exit").name == "intersection_1_exit"

    def test_a_missing_anchor_lists_what_the_host_offers(self):
        with pytest.raises(PlacementError, match="intersection_1_entry"):
            resolve_anchor(_probe(_host_sd()), "intersection_2_entry")

    def test_name_shape(self):
        assert is_anchor_name("intersection_3_exit")
        assert is_anchor_name("first_crosswalk")
        assert not is_anchor_name("lane:123")


class TestAuthoringIntegration:
    def _spec(self, authored: dict) -> dict:
        return {
            "recipe_id": "V-11/driving_wrong_way/synthetic/001",
            "leaf": "V-11",
            "scenario": "driving_wrong_way",
            "host": {"scene": "synthetic", "world_version": "test@0"},
            "ego": {"replay_frames": 8, "z_to_ground": Z_TO_GROUND},
            "actors": {
                "dne_sign": {
                    "op": "insert",
                    "track": {"type": "TRAFFIC_CONE"},
                    "asset": {"dims": [0.31, 1.06, 2.2]},
                    "authored": {"template": "static", "reference": "ego_route", **authored},
                }
            },
            "review": {"status": "pending"},
        }

    def test_anchored_static_lands_past_the_junction(self):
        sd = _host_sd()
        recipe, _ = author_recipe(
            sd, self._spec({"anchor": "intersection_1_exit", "arc": 5.0, "lateral": -3.0}),
            use_registry=False,
        )
        actor = recipe.actors["dne_sign"]
        solved = actor.authored["anchor_solved"]
        assert solved["name"] == "intersection_1_exit"
        # The route starts at x=0, so route arc == x: the sign sits 5 m past
        # the junction's far edge (cross street at x=50 + the detector's reach).
        x = float(np.asarray(actor.spawn["position"])[0])
        assert x == pytest.approx(solved["reference_arc_m"] + 5.0, abs=0.1)
        assert 55.0 <= x <= 70.0

    def test_anchored_arc_equals_handoff_arc_plus_offset_when_anchor_is_handoff(self):
        sd = _host_sd()
        plain, _ = author_recipe(sd, self._spec({"arc": 30.0}), use_registry=False)
        anchored, _ = author_recipe(
            sd, self._spec({"anchor": "handoff", "arc": 30.0}), use_registry=False
        )
        np.testing.assert_allclose(
            np.asarray(plain.actors["dne_sign"].spawn["position"]),
            np.asarray(anchored.actors["dne_sign"].spawn["position"]),
        )

    def test_unknown_anchor_refuses_with_the_hosts_own_list(self):
        with pytest.raises(PlacementError, match="host has no anchor"):
            author_recipe(
                _host_sd(), self._spec({"anchor": "intersection_9_exit", "arc": 5.0}),
                use_registry=False,
            )
