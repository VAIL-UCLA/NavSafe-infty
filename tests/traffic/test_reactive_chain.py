# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The whole chain from a recipe edit to the pose the world sees.

An inserted reactive actor passes through five hands:

    spawn_reactive_actor  ->  a track (the declaration: asset, dims, class)
                          ->  scenario metadata (the specs)
                          ->  NavSafeTraffic.adopt  (build the drivers)
                          ->  driver.step           (the motion)
                          ->  pose_overrides -> agent list -> metrics/renderer

Any one of those links failing produces the same symptom — an actor that
stands still, or renders on a rail — so each is pinned separately, and the
whole chain once.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.recipe.schema import spec_digest
from navsafe.scenario.edits import apply_scenario_edits, spawn_reactive_actor
from navsafe.traffic.navsafe import NavSafeTraffic
from navsafe.traffic.stage_free import WorldView

T = 40
DT = 0.1
ROAD = [[float(x), 0.0] for x in range(0, 301, 5)]


def _sd():
    xs = np.linspace(-30.0, 10.0, T)
    return {
        "metadata": {"sdc_id": "ego", "ts": (np.arange(T) * 100_000).tolist()},
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": np.stack([xs, np.zeros(T), np.zeros(T)], axis=1),
                    "heading": np.zeros(T, np.float32),
                    "velocity": np.stack([np.full(T, 10.0), np.zeros(T)], axis=1),
                    "valid": np.ones(T, bool),
                    "length": np.full(T, 4.5, np.float32),
                    "width": np.full(T, 1.8, np.float32),
                    "height": np.full(T, 1.5, np.float32),
                },
            }
        },
    }


def _actor(**policy):
    policy.setdefault("kind", "idm")
    policy.setdefault("path_polyline", ROAD)
    return _signed({
        "name": "oncoming_vehicle",
        "op": "insert",
        "track_type": "VEHICLE",
        "dims": [4.6, 2.0, 1.5],
        "registry_key": "hb_car_3",
        "nurec_asset_id": "asset/harvester_calib/car_3.ply",
        "semantic_class": "pedestrian",
        "spawn": {"position": [50.0, 0.0, 0.0], "heading": 0.0, "velocity": [8.0, 0.0]},
        "policy": policy,
    })


def _signed(spec: dict) -> dict:
    """Stamp the digest the frozen recipe would have carried."""
    spec["sha256"] = spec_digest(spec)
    return spec


class TestDeclaration:
    def test_the_track_carries_the_asset_and_its_class(self):
        # This half is unchanged from the baked path on purpose: asset
        # insertion, dims for collision and the renderer's track list all
        # read a track, and none of them should know about policies.
        sd = spawn_reactive_actor(_sd(), actors=[_actor()], frames={"T": T, "dt_s": DT})
        track = sd["tracks"]["navsafe_oncoming_vehicle"]
        meta = track["metadata"]
        assert meta["nurec_asset_id"].endswith("car_3.ply")
        # The render class the car2sim recon actually accepts — a VEHICLE-class
        # insert is rejected server-side after the PLY loads.
        assert meta["nurec_semantic_class"] == "pedestrian"
        assert meta["injected_obstacle"] is True
        assert track["state"]["position"].shape == (T, 3)

    def test_it_spawns_holding_its_pose_not_guessing_a_trajectory(self):
        # If the manager never runs, the actor must visibly stand still.
        # Seeding a constant-velocity guess would make a broken run look like
        # a working one.
        sd = spawn_reactive_actor(_sd(), actors=[_actor()], frames={"T": T, "dt_s": DT})
        pos = sd["tracks"]["navsafe_oncoming_vehicle"]["state"]["position"]
        assert np.allclose(pos[0], pos[-1])

    def test_the_specs_are_left_for_the_manager(self):
        sd = spawn_reactive_actor(_sd(), actors=[_actor()], frames={"T": T, "dt_s": DT})
        (spec,) = sd["metadata"]["navsafe_reactive"]
        assert spec["track_id"] == "navsafe_oncoming_vehicle"
        assert spec["policy"]["kind"] == "idm"

    def test_a_host_at_the_wrong_rate_is_refused(self):
        # dt is the integration step for IDM; a mismatched host runs the same
        # scenario at the wrong speed.
        with pytest.raises(ValueError, match="dt="):
            spawn_reactive_actor(_sd(), actors=[_actor()], frames={"T": T, "dt_s": 0.05})

    def test_an_actor_with_no_policy_is_refused(self):
        bad = _actor()
        bad["policy"] = {}
        bad["sha256"] = spec_digest(bad)
        with pytest.raises(ValueError, match="policy.kind"):
            spawn_reactive_actor(_sd(), actors=[bad], frames={"T": T, "dt_s": DT})

    def test_an_unsigned_spec_is_refused_at_the_last_gate(self):
        # replay.py is not the only way into the env, so the integrity check
        # cannot live only there.
        bad = _actor()
        bad.pop("sha256")
        with pytest.raises(ValueError, match="carries no sha256"):
            spawn_reactive_actor(_sd(), actors=[bad], frames={"T": T, "dt_s": DT})

    def test_a_tampered_spec_is_refused_at_the_last_gate(self):
        bad = _actor()
        bad["policy"]["v0"] = 99.0          # signed before this edit
        with pytest.raises(ValueError, match="sha256 mismatch"):
            spawn_reactive_actor(_sd(), actors=[bad], frames={"T": T, "dt_s": DT})

    def test_it_is_reachable_as_an_edit_tool(self):
        sd = apply_scenario_edits(_sd(), [{
            "tool": "spawn_reactive_actor", "actors": [_actor()],
            "frames": {"T": T, "dt_s": DT}, "recipe_id": "C-10/x/y/001",
        }])
        assert "navsafe_oncoming_vehicle" in sd["tracks"]


class TestTheWholeChain:
    def _run(self, **policy):
        sd = apply_scenario_edits(_sd(), [{
            "tool": "spawn_reactive_actor", "actors": [_actor(**policy)],
            "frames": {"T": T, "dt_s": DT},
        }])
        mgr = NavSafeTraffic()
        mgr.adopt(sd["metadata"]["navsafe_reactive"])

        class _Env:
            scenario_timestep = 0
            current_scenario = sd
            agent_states: list = []

            def get_ego_state(self):
                k = self.scenario_timestep
                p = sd["tracks"]["ego"]["state"]["position"][k]
                return {"position": p, "heading": 0.0,
                        "velocity": np.array([10.0, 0.0]),
                        "length": 4.5, "width": 1.8}

        env = _Env()
        mgr.reset(env)
        for k in range(20):
            env.scenario_timestep = k
            mgr.step(env, DT)
        return mgr

    def test_the_actor_actually_moves(self):
        mgr = self._run()
        pose = mgr.pose_overrides["navsafe_oncoming_vehicle"]
        # Spawned at x=50 doing 8 m/s; after 2 s it must be well past that.
        assert pose["position"][0] > 60.0

    def test_the_override_is_keyed_by_track_id(self):
        # The agent list the metrics read is keyed by track id, so an
        # override keyed by the actor's own name would silently never merge.
        mgr = self._run()
        assert set(mgr.pose_overrides) == {"navsafe_oncoming_vehicle"}

    def test_blind_to_ego_survives_the_whole_chain(self):
        # The flag has to travel recipe -> edit -> metadata -> driver. It is
        # the one bit that decides whether C-10 is a crash scenario.
        mgr = self._run(blind_to_ego=True)
        driver = mgr.drivers["navsafe_oncoming_vehicle"]
        assert driver.blind_to_ego is True
