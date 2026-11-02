# Copyright (c) 2022-2025, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Log-replay traffic manager.

Selected by ``EnvCfg(traffic_mode="log_replay")``.  Drives non-ego
agents along the trajectories recorded in the loaded ScenarioNet
scenario — the canonical mode for closed-loop ego evaluation against
a fixed traffic log.

The agent list itself is built by the env from the recorded tracks
(:meth:`~navsafe.env.navsafe_env.NavSafeEnv._update_agents`); this
manager advances the replay cursor of the env's ``agent_manager``.
"""

from __future__ import annotations

from typing import Any

from navsafe.traffic.base import TrafficManager


class LogReplayTraffic(TrafficManager):
    """Replay non-ego agents from the loaded scenario log.

    The manager is a thin orchestrator over the env's
    ``agent_manager`` (a :class:`~navsafe.manager.scenario_replay_manager.ScenarioReplayManager`):
    it advances the per-frame agent state from the recorded tracks,
    optionally skipping the ego (when an external policy is overriding
    ego state via ``set_ego_override``).

    Attributes:
        skip_ego: When ``True``, the ego agent is not updated by the
            log-replay step — typically set by the env when a model is
            actively planning the ego trajectory.
    """

    def __init__(self, *, skip_ego: bool = False) -> None:
        self.skip_ego = skip_ego

    # ------------------------------------------------------------------
    # TrafficManager interface
    # ------------------------------------------------------------------

    def step(self, env: Any, dt: float) -> None:  # noqa: ARG002 - dt unused
        """Advance non-ego agents along the recorded scenario log.

        Reads ``env.scenario_timestep`` and replays every track to that
        frame via ``env.agent_manager.update_agents``.  ``dt`` is unused
        because timestep advancement is the env's responsibility (the
        env increments ``scenario_timestep`` each frame).

        Args:
            env: A :class:`~navsafe.env.navsafe_env.NavSafeEnv` exposing
                ``agent_manager`` and ``scenario_timestep``.
            dt: Simulation timestep (s) — unused by log replay.
        """
        # Honor an env-level ego-override flag if the env exposes one.
        skip_ego = self.skip_ego or bool(getattr(env, "_ego_override_active", False))

        env.agent_manager.update_agents(
            timestep=env.scenario_timestep,
            skip_ego=skip_ego,
        )

    def reset(self, env: Any) -> None:
        """Reset agents to the first frame of the scenario log.

        Delegates to ``env.agent_manager.reset(timestep=0)``.
        """
        env.agent_manager.reset(timestep=0)


__all__ = ["LogReplayTraffic"]
