# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Abstract interface for log-replay, no-traffic and semi-reactive managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrafficManager(ABC):
    """Episode-level orchestrator for non-ego traffic.

    A ``TrafficManager`` is constructed once per episode (or once and
    re-used across episodes by calling :meth:`reset`).  The env calls
    :meth:`step` once per simulation timestep, before physics, to give
    the manager an opportunity to update non-ego agent state.

    Subclasses are free to read from and write to the env's agent state
    via the documented hooks (``env.agent_manager``, ``env.current_scenario``,
    ``env.scenario_timestep``); the ABC does not constrain how state is
    stored.  See :class:`~navsafe.traffic.semi_reactive.SemiReactiveTraffic`
    for the canonical (wired) reactive-traffic shape and
    :class:`~navsafe.traffic.log_replay.LogReplayTraffic` for the log-replay
    shape.
    """

    @abstractmethod
    def step(self, env: Any, dt: float) -> None:
        """Advance non-ego agents by one simulation timestep.

        Called from the env's pre-physics hook.  Implementations mutate
        agent state on ``env`` (or its ``agent_manager``) in place.

        Args:
            env: The owning :class:`~navsafe.env.navsafe_env.NexusSimEnv`
                (or, during the migration window, a retired env class).
                Subclasses access ``env.agent_manager``,
                ``env.current_scenario``, ``env.scenario_timestep``,
                ``env.get_ego_state()``, etc.
            dt: Simulation timestep in seconds.
        """

    @abstractmethod
    def reset(self, env: Any) -> None:
        """Reset manager state at the start of a new episode.

        Called from the env's reset hook.  Implementations clear any
        per-episode caches (e.g. lead-vehicle assignments, replay
        cursors) and rebuild from the env's current scenario.
        """


__all__ = ["TrafficManager"]
