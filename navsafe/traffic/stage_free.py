# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Reactive traffic that runs without a USD stage.

Every traffic manager shipped before this one is welded to IsaacSim:
``LogReplayTraffic`` and ``SemiReactiveTraffic`` both call
``env.agent_manager.update_agents(stage=env.sim.stage, ...)`` and
``set_agent_pose(env.sim.stage, ...)``. That is fine for the IsaacLab driver
and useless for the closed-loop evaluator, which runs the **pure-Python** step
loop — and, in this deployment, is the only loop that runs at all:

    isaaclab/scene/interactive_scene.py:40 imports isaaclab_contrib
    unconditionally; isaaclab_contrib is not installed in the venv; so
    `from isaaclab.envs import DirectRLEnv` raises, `_HAS_ISAACLAB` is False,
    `_advance_replay_agent_prims` returns at its first line, and
    `_traffic_manager.step()` is never called.

So a reactive actor needs a manager that owns no prims. This one computes
poses and publishes them into ``pose_overrides`` — the dict the env already
merges over logged track state before handing it to collision, TTC, the BEV
and the renderer. Nothing else is needed to be seen by the metrics.

The split this makes explicit: **advancing an agent is a decision, posing a
prim is a rendering detail**. `_advance_replay_agent_prims` conflated the two,
which is why the decision half was unreachable off IsaacSim.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from navsafe.traffic.base import TrafficManager

logger = logging.getLogger(__name__)


class Driver:
    """One agent's controller. Owns its own pose; sees the world each step."""

    #: set by the manager when the driver is created
    agent_id: Any = None

    def reset(self, *, spawn: Dict[str, Any]) -> None:
        raise NotImplementedError

    def step(self, world: "WorldView", dt: float) -> None:
        raise NotImplementedError

    def pose(self) -> Dict[str, Any]:
        """``{position, heading, velocity, length, width}`` for this frame."""
        raise NotImplementedError


class WorldView:
    """What a driver may look at this frame.

    Deliberately a snapshot rather than the env: a driver that reaches into
    the env can read the future (the whole logged track array is right there),
    and a scenario built on an actor that knows the ego's future is not
    testing the policy, it is testing nothing.
    """

    __slots__ = ("t", "dt", "ego", "agents", "ego_id")

    def __init__(self, *, t: int, dt: float, ego: Dict[str, Any],
                 agents: List[Dict[str, Any]], ego_id: Any):
        self.t = t
        self.dt = dt
        self.ego = ego
        self.agents = agents          # every OTHER agent, logged + reactive
        self.ego_id = ego_id

    def others(self, exclude: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        drop = set(exclude)
        return [a for a in self.agents if a.get("id") not in drop]

    def ego_distance(self, xy) -> float:
        p = np.asarray(self.ego.get("position", (0.0, 0.0, 0.0)), np.float64)[:2]
        return float(np.linalg.norm(p - np.asarray(xy, np.float64)[:2]))


class StageFreeTraffic(TrafficManager):
    """Base for managers that publish ``pose_overrides`` and touch no USD.

    Subclasses fill :attr:`drivers`. The env recognises ``stage_free`` and
    steps the manager from its own loop instead of from the IsaacLab-only
    prim hook.
    """

    #: the env checks this to decide where to call `step` from
    stage_free = True

    def __init__(self) -> None:
        self.drivers: Dict[Any, Driver] = {}
        self.pose_overrides: Dict[Any, Dict[str, Any]] = {}
        self._t = 0

    # -- TrafficManager ------------------------------------------------
    def reset(self, env: Any) -> None:
        self.pose_overrides = {}
        self._t = 0
        for driver in self.drivers.values():
            driver.reset(spawn=getattr(driver, "spawn", {}) or {})

    def step(self, env: Any, dt: float) -> None:
        if not self.drivers:
            return
        world = self._world(env, dt)
        overrides: Dict[Any, Dict[str, Any]] = {}
        for agent_id, driver in self.drivers.items():
            try:
                driver.step(world, dt)
            except Exception:  # noqa: BLE001 — one bad driver must not kill the episode
                if getattr(driver, "strict", False):
                    raise  # Controlled experiments must never silently freeze hazards.
                logger.exception("traffic: driver %r failed at t=%d; freezing it",
                                 agent_id, world.t)
                continue
            overrides[agent_id] = driver.pose()
        self.pose_overrides = overrides
        self._t += 1

    # -- helpers -------------------------------------------------------
    def _world(self, env: Any, dt: float) -> WorldView:
        t = int(getattr(env, "scenario_timestep", self._t))
        try:
            ego = env.get_ego_state() or {}
        except Exception:  # noqa: BLE001
            ego = {}
        sd = getattr(env, "current_scenario", None) or {}
        ego_id = str((sd.get("metadata") or {}).get("sdc_id", "ego"))
        # The env's own agent list is the logged world at this frame, already
        # dims-resolved. Reactive agents are layered on by the manager, so a
        # driver sees its peers' CURRENT poses, not their logged ones.
        agents = list(getattr(env, "agent_states", None) or [])
        for agent_id, pose in self.pose_overrides.items():
            merged = False
            for row in agents:
                if row.get("id") == agent_id:
                    row.update(pose)
                    merged = True
                    break
            if not merged:
                agents.append({"id": agent_id, **pose})
        return WorldView(t=t, dt=float(dt), ego=ego, agents=agents, ego_id=ego_id)


__all__ = ["Driver", "StageFreeTraffic", "WorldView"]
