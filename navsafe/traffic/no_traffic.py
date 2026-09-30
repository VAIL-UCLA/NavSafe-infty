# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""No-op traffic manager.

Selected by ``EnvCfg(traffic_mode="no_traffic")``.  The env still spawns
non-ego agents at the scenario's initial frame (the manager's
:meth:`reset` may delegate to the agent manager for that), but no
per-step traffic update is applied — useful for ablations that isolate
ego behavior from reactive traffic, and for the smallest-possible env
configurations used in unit tests.
"""

from __future__ import annotations

from typing import Any

from navsafe.traffic.base import TrafficManager


class NoTrafficManager(TrafficManager):
    """No-op manager: every :meth:`step` is a no-op.

    Reset is also a no-op by default.  Subclasses that want
    "spawn at frame 0 then freeze" can override :meth:`reset` to call
    ``env.agent_manager.reset(env.sim.stage, timestep=0)`` while
    keeping :meth:`step` empty.
    """

    def step(self, env: Any, dt: float) -> None:  # noqa: ARG002
        """No-op."""
        return

    def reset(self, env: Any) -> None:  # noqa: ARG002
        """No-op."""
        return


__all__ = ["NoTrafficManager"]
