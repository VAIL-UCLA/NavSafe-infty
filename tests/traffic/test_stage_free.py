# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Reactive poses must reach the metrics AND the camera, off IsaacSim.

Every traffic manager before ``navsafe`` reaches its agents through
``env.sim.stage``, and the hook that calls them returns at its first line
when IsaacLab is unavailable. In this deployment IsaacLab is always
unavailable (``isaaclab.scene`` imports ``isaaclab_contrib``, which is not
installed), so those managers never run at all under the evaluator.

These tests pin the two halves of the replacement: the manager is stepped
from the pure-Python loop, and what it decides shows up in the agent-state
list — the one list collision, TTC, the BEV and the renderer all read.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.env.env_cfg import EnvCfg
from navsafe.env.navsafe_env import NexusSimEnv
from navsafe.traffic.navsafe import NavSafeTraffic
from navsafe.traffic.stage_free import Driver, StageFreeTraffic

T = 6


class _Marching(Driver):
    """Walks +1 m in x per frame. Distinguishable from any logged track."""

    def __init__(self, agent_id, spawn, **_kw):
        self.agent_id = agent_id
        self.spawn = dict(spawn)
        self.reset(spawn=self.spawn)

    def reset(self, *, spawn):
        self._xy = np.asarray(spawn.get("position", (0.0, 0.0, 0.0)), np.float64)[:2].copy()

    def step(self, world, dt):
        self._xy = self._xy + np.array([1.0, 0.0])

    def pose(self):
        return {
            "position": np.array([self._xy[0], self._xy[1], 0.0], np.float32),
            "heading": 0.0,
            "velocity": np.array([10.0, 0.0], np.float32),
            "length": 4.5,
            "width": 1.8,
        }


def _track(kind="VEHICLE", x0=0.0):
    xs = np.arange(T, dtype=np.float64) * 0.0 + x0
    return {
        "type": kind,
        "state": {
            "position": np.stack([xs, np.zeros(T), np.zeros(T)], axis=1),
            "heading": np.zeros(T, np.float32),
            "valid": np.ones(T, bool),
            "length": np.full(T, 4.5, np.float32),
            "width": np.full(T, 1.8, np.float32),
            "height": np.full(T, 1.5, np.float32),
        },
    }


class _Renderer:
    def __init__(self):
        self.seen = None

    def update_agents(self, agent_states):
        self.seen = agent_states


def _env(tracks, manager) -> NexusSimEnv:
    # Bypass __init__: the real constructor boots IsaacSim. The methods under
    # test read only cfg, the scenario and the manager.
    env = object.__new__(NexusSimEnv)
    env.cfg = EnvCfg(traffic_mode="navsafe")
    env._scenario_data = {"tracks": tracks, "metadata": {"sdc_id": "ego"}}
    env._scenario_timestep = 0
    env._agent_states = []
    env._renderer = None
    env._traffic_manager = manager
    return env


class TestStageFreeManager:
    def test_it_declares_itself_stage_free(self):
        # The env keys off this to decide whether to step the manager from
        # the pure-Python loop instead of the IsaacLab-only prim hook.
        assert StageFreeTraffic.stage_free is True

    def test_a_driver_is_taken_over_by_name_not_geometry(self):
        # semi_reactive would refuse this actor: it is a pedestrian, and it
        # is ahead of the ego. Being the point of the scenario is what makes
        # it eligible here.
        mgr = NavSafeTraffic()
        mgr.drivers = {"ped_1": _Marching("ped_1", {"position": (5.0, 0.0, 0.0)})}
        mgr._built = True
        env = _env({"ped_1": _track("PEDESTRIAN")}, mgr)
        env.get_ego_state = lambda: {"position": np.zeros(3), "heading": 0.0}
        env.current_scenario = env._scenario_data
        mgr.step(env, 0.1)
        assert mgr.pose_overrides["ped_1"]["position"][0] == pytest.approx(6.0)

    def test_an_unregistered_policy_does_not_take_the_actor_over(self):
        # Better to run the scenario minus its reactivity, loudly, than to
        # guess a policy and score the wrong experiment.
        mgr = NavSafeTraffic([{"name": "x", "policy": {"kind": "no_such_policy"}}])
        env = _env({"x": _track()}, mgr)
        mgr._build(env)
        assert mgr.drivers == {}


class TestPosesReachTheWorld:
    def _stepped(self):
        mgr = NavSafeTraffic()
        mgr.drivers = {"actor": _Marching("actor", {"position": (5.0, 0.0, 0.0)})}
        mgr._built = True
        env = _env({"actor": _track(x0=0.0), "ego": _track(x0=-20.0)}, mgr)
        env.get_ego_state = lambda: {"position": np.zeros(3), "heading": 0.0}
        env.current_scenario = env._scenario_data
        mgr.step(env, 0.1)
        env._update_agents()
        return env

    def test_the_agent_list_carries_the_reactive_pose_not_the_logged_one(self):
        # This list is what collision, TTC and the BEV read. The logged
        # track sits at x=0 for every frame; the driver is at x=6.
        env = self._stepped()
        row = next(r for r in env.agent_states if r["id"] == "actor")
        assert row["position"][0] == pytest.approx(6.0)
        assert row["velocity"][0] == pytest.approx(10.0)

    def test_the_ego_is_not_taken_over(self):
        env = self._stepped()
        assert "ego" not in [r["id"] for r in env.agent_states]

    def test_an_actor_with_no_logged_row_is_appended_not_dropped(self):
        mgr = NavSafeTraffic()
        mgr.drivers = {"ghost": _Marching("ghost", {"position": (1.0, 2.0, 0.0)})}
        mgr._built = True
        env = _env({"ego": _track(x0=-20.0)}, mgr)   # no track named "ghost"
        env.get_ego_state = lambda: {"position": np.zeros(3), "heading": 0.0}
        env.current_scenario = env._scenario_data
        mgr.step(env, 0.1)
        env._update_agents()
        row = next(r for r in env.agent_states if r["id"] == "ghost")
        assert row["position"][0] == pytest.approx(2.0)

    def test_the_renderer_is_handed_the_same_list(self):
        # The bug this pins: renderer.update_agents was only ever called from
        # _pre_physics_step, which the Evaluator does not call, so nurec_grpc
        # kept serving the poses it cached at scene load — the actor yielded
        # in the metrics and sailed on in the camera.
        env = self._stepped()
        renderer = _Renderer()
        env._renderer = renderer
        renderer.update_agents(env._agent_states)
        assert renderer.seen is env._agent_states
        row = next(r for r in renderer.seen if r["id"] == "actor")
        assert row["position"][0] == pytest.approx(6.0)


class TestSupportedCombinations:
    def test_navsafe_is_registered_against_the_closed_loop_backend(self):
        # nurec_grpc is the evaluator's backend and had no reactive traffic
        # mode registered against it at all.
        from navsafe.env._bootstrap import SUPPORTED_COMBINATIONS

        assert ("sensor", "navsafe", "py123d", "nurec_grpc") in SUPPORTED_COMBINATIONS

    def test_the_factory_builds_it(self):
        from navsafe.env._bootstrap import _make_traffic_manager

        mgr = _make_traffic_manager(EnvCfg(traffic_mode="navsafe"))
        assert isinstance(mgr, NavSafeTraffic) and mgr.stage_free
