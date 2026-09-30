# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for scenario-edit tools (navsafe/scenario/edits.py)."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.scenario.edits import (
    _DEFAULT_OBSTACLE_TYPES,
    _OBSTACLE_ARCHETYPES,
    apply_scenario_edits,
    place_static_obstacles,
    relocate_agent,
)


def _sd(T: int = 60, step_m: float = 2.0, ego_z: float = 55.0) -> dict:
    """Minimal ScenarioDescription: straight-line ego at absolute city z."""
    pos = np.zeros((T, 3), np.float64)
    pos[:, 0] = np.arange(T) * step_m
    pos[:, 2] = ego_z
    return {
        "metadata": {"sdc_id": "ego"},
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": pos,
                    "heading": np.zeros(T),
                    "valid": np.ones(T, bool),
                },
            },
        },
    }


def _injected(sd: dict) -> dict:
    return {
        k: v for k, v in sd["tracks"].items() if (v.get("metadata") or {}).get("injected_obstacle")
    }


class TestConeArchetype:
    def test_cone_track_type_and_dims(self):
        sd = place_static_obstacles(_sd(), count=3, seed=7, types=["cone"])
        inj = _injected(sd)
        assert len(inj) == 3
        for tr in inj.values():
            assert tr["type"] == "TRAFFIC_CONE"
            assert tr["metadata"]["type"] == "TRAFFIC_CONE"
            assert tr["metadata"]["obstacle_archetype"] == "cone"
            st = tr["state"]
            l, w, h = _OBSTACLE_ARCHETYPES["cone"]
            assert float(st["length"][0]) == pytest.approx(l)
            assert float(st["width"][0]) == pytest.approx(w)
            assert float(st["height"][0]) == pytest.approx(h)

    def test_default_pool_stays_vehicle_only(self):
        # Legacy behaviour: without ``types``, no cone is ever sampled.
        assert "cone" not in _DEFAULT_OBSTACLE_TYPES
        sd = place_static_obstacles(_sd(T=120), count=20, seed=0)
        for tr in _injected(sd).values():
            assert tr["metadata"]["obstacle_archetype"] in _DEFAULT_OBSTACLE_TYPES
            assert tr["type"] == "VEHICLE"

    def test_ground_z_from_ego_track(self):
        sd = place_static_obstacles(_sd(ego_z=55.0), count=1, seed=1, types=["cone"])
        (tr,) = _injected(sd).values()
        z = float(np.asarray(tr["state"]["position"])[0, 2])
        assert abs(z - (55.0 - 1.7)) < 1e-6  # ego IMU z minus default offset


class TestNurecAssetIds:
    def test_asset_id_metadata_plumbed(self):
        sd = place_static_obstacles(
            _sd(),
            count=2,
            seed=3,
            types=["cone"],
            nurec_asset_ids={"cone": "/assets/cone_3dgs.ply"},
        )
        for tr in _injected(sd).values():
            assert tr["metadata"]["nurec_asset_id"] == "/assets/cone_3dgs.ply"
            assert tr["metadata"]["nurec_semantic_class"] == "cone"

    def test_no_asset_id_no_metadata(self):
        sd = place_static_obstacles(_sd(), count=2, seed=3, types=["cone"])
        for tr in _injected(sd).values():
            assert "nurec_asset_id" not in tr["metadata"]

    def test_unmapped_archetype_gets_no_asset_id(self):
        sd = place_static_obstacles(
            _sd(), count=4, seed=3, types=["car"], nurec_asset_ids={"cone": "/assets/cone_3dgs.ply"}
        )
        for tr in _injected(sd).values():
            assert "nurec_asset_id" not in tr["metadata"]


class TestApplyScenarioEdits:
    def test_apply_via_registry_with_asset_ids(self):
        sd = apply_scenario_edits(
            _sd(),
            [
                {
                    "tool": "place_static_obstacles",
                    "count": 1,
                    "seed": 5,
                    "types": ["cone"],
                    "nurec_asset_ids": {"cone": "bank_cone_01"},
                }
            ],
        )
        (tr,) = _injected(sd).values()
        assert tr["metadata"]["nurec_asset_id"] == "bank_cone_01"
        assert sd["metadata"]["_scenario_edits_applied"]

    def test_idempotent(self):
        spec = [{"tool": "place_static_obstacles", "count": 2, "seed": 5, "types": ["cone"]}]
        sd = apply_scenario_edits(_sd(), spec)
        n1 = len(_injected(sd))
        sd = apply_scenario_edits(sd, spec)
        assert len(_injected(sd)) == n1


class TestRelocateDynamicProfiles:
    @staticmethod
    def _with_actor() -> dict:
        sd = _sd(T=80, step_m=1.0)
        sd["tracks"]["actor"] = {
            "type": "VEHICLE",
            "state": {
                "position": np.zeros((80, 3), np.float32),
                "heading": np.zeros(80, np.float32),
                "velocity": np.zeros((80, 2), np.float32),
                "valid": np.ones(80, bool),
            },
            "metadata": {"object_id": "actor"},
        }
        return sd

    def test_crossing_profile_moves_across_route(self):
        sd = relocate_agent(
            self._with_actor(),
            relocations={
                "actor": {
                    "mode": "crossing",
                    "arc": 20.0,
                    "start_lateral": -5.0,
                    "end_lateral": 5.0,
                    "conflict_frame": 30,
                    "maneuver_duration_frames": 20,
                }
            },
            dt_s=0.1,
        )
        state = sd["tracks"]["actor"]["state"]
        position = np.asarray(state["position"])
        assert position[20, 1] == pytest.approx(-5.0)
        assert position[30, 1] == pytest.approx(0.0)
        assert position[40, 1] == pytest.approx(5.0)
        assert np.max(np.linalg.norm(state["velocity"], axis=1)) > 0

    def test_braking_profile_reaches_zero_speed(self):
        sd = relocate_agent(
            self._with_actor(),
            relocations={
                "actor": {
                    "mode": "braking",
                    "arc": 10.0,
                    "speed": 4.0,
                    "deceleration_mps2": 4.0,
                    "maneuver_start_frame": 5,
                }
            },
            dt_s=0.1,
        )
        velocity = np.asarray(sd["tracks"]["actor"]["state"]["velocity"])
        assert np.linalg.norm(velocity[0]) == pytest.approx(4.0)
        assert np.linalg.norm(velocity[-1]) == pytest.approx(0.0)
