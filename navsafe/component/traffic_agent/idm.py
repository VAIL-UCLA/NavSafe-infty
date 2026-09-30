# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""IDM parameters and per-vehicle acceleration/lead updates for reactive traffic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class IDMParams:
    """Parameters for the Intelligent Driver Model.

    Attributes:
        v0: Desired velocity (m/s).
        s0: Minimum gap (m).
        T: Desired time headway (s).
        a: Maximum acceleration (m/s²).
        b: Comfortable deceleration (m/s²).
        delta: Acceleration exponent (dimensionless, typically 4).
    """

    v0: float = 30.0
    s0: float = 2.0
    T: float = 1.5
    a: float = 1.0
    b: float = 1.5
    delta: float = 4.0

    def __post_init__(self) -> None:
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if self.s0 < 0:
            raise ValueError(f"s0 must be non-negative, got {self.s0}")
        if self.T < 0:
            raise ValueError(f"T must be non-negative, got {self.T}")
        if self.a <= 0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.b <= 0:
            raise ValueError(f"b must be positive, got {self.b}")
        if self.delta <= 0:
            raise ValueError(f"delta must be positive, got {self.delta}")


class IDMActor:
    """Per-vehicle IDM car-following actor.

    Stateless across calls — every method takes the agent state as input
    and returns the updated state.  Holds only the IDM parameter struct.

    The actor exposes two layers:

    * :meth:`compute_acceleration` — the analytical IDM formula for a single
      (v, delta_v, s) triple.  This is the unit-testable kernel.
    * :meth:`update_agents` — multi-agent integration loop: per agent,
      look up the lead vehicle's velocity, compute acceleration via the
      kernel, integrate to a new velocity (clamped to >= 0), and advance
      the position along the heading.

    Episode-level orchestration (reading agent state from the env, assigning
    leads from proximity, writing state back) lives on the matching manager
    class :class:`~navsafe.traffic.semi_reactive.SemiReactiveTraffic`.
    """

    def __init__(self, params: IDMParams | None = None) -> None:
        self.params = params if params is not None else IDMParams()

    # ------------------------------------------------------------------
    # Core IDM formula
    # ------------------------------------------------------------------

    def compute_acceleration(self, v: float, delta_v: float, s: float) -> float:
        """Compute IDM acceleration for a single agent.

        Args:
            v: Current velocity of the agent (m/s), must be >= 0.
            delta_v: Approach rate to lead vehicle (m/s).
                     Positive means closing in on the lead.
            s: Bumper-to-bumper gap to lead vehicle (m).
               Use ``float('inf')`` for free-road (no lead vehicle).

        Returns:
            Acceleration (m/s²).  The resulting velocity after applying
            this acceleration for one timestep is guaranteed >= 0 via
            clamping of the acceleration value at the
            :meth:`update_agents` integration step.
        """
        p = self.params

        # Free-road term: a * [1 - (v/v0)^delta]
        v_ratio = v / p.v0
        free_road_term = 1.0 - v_ratio ** p.delta

        if math.isinf(s) and s > 0:
            # No lead vehicle — pure free-road acceleration
            return p.a * free_road_term

        # Desired gap s*
        interaction = v * delta_v / (2.0 * math.sqrt(p.a * p.b))
        s_star = p.s0 + max(0.0, v * p.T + interaction)

        # Full IDM acceleration
        if s <= 0:
            # Emergency: gap is zero or negative — apply max deceleration
            return -p.a
        gap_term = (s_star / s) ** 2
        accel = p.a * (free_road_term - gap_term)

        return accel

    # ------------------------------------------------------------------
    # Multi-agent update
    # ------------------------------------------------------------------

    def update_agents(
        self,
        agent_states: List[Dict[str, Any]],
        ego_state: Dict[str, Any],
        dt: float,
    ) -> List[Dict[str, Any]]:
        """Update all non-ego agents for one timestep.

        Each agent's acceleration is computed based on its own lead vehicle
        (identified by ``lead_id`` and ``gap`` in the agent state dict).

        Args:
            agent_states: List of agent state dicts.  Each dict must contain
                at minimum ``id``, ``position``, ``velocity``, ``heading``.
                Optional: ``lead_id``, ``gap``.
            ego_state: Ego vehicle state dict with ``id``, ``position``,
                ``velocity``, ``heading``.
            dt: Simulation timestep (s).

        Returns:
            Updated list of agent state dicts with new velocities and
            positions.
        """
        # Build lookup for velocities (include ego)
        velocity_lookup: Dict[int, float] = {ego_state["id"]: ego_state["velocity"]}
        for agent in agent_states:
            velocity_lookup[agent["id"]] = agent["velocity"]

        updated: List[Dict[str, Any]] = []
        for agent in agent_states:
            agent = dict(agent)  # shallow copy
            v = agent["velocity"]
            lead_id = agent.get("lead_id")
            gap = agent.get("gap")

            if lead_id is not None and gap is not None:
                lead_v = velocity_lookup.get(lead_id, v)
                delta_v = v - lead_v  # positive = closing
                accel = self.compute_acceleration(v, delta_v, gap)
            else:
                # Free road — no lead vehicle
                accel = self.compute_acceleration(v, 0.0, float("inf"))

            # Integrate velocity and clamp to >= 0
            new_v = max(0.0, v + accel * dt)
            agent["velocity"] = new_v

            # Update position along heading
            heading = agent["heading"]
            avg_v = (v + new_v) / 2.0
            dx = avg_v * dt * math.cos(heading)
            dy = avg_v * dt * math.sin(heading)
            pos = np.array(agent["position"], dtype=np.float64)
            pos[0] += dx
            pos[1] += dy
            agent["position"] = pos

            updated.append(agent)

        return updated

    # ------------------------------------------------------------------
    # Lead assignment (per-vehicle proximity calculation, no env state)
    # ------------------------------------------------------------------

    def assign_leads(
        self,
        agent_states: List[Dict[str, Any]],
        ego_state: Dict[str, Any],
        ego_id: Any,
    ) -> None:
        """Assign lead vehicle for each agent based on proximity ahead.

        Modifies agent dicts in-place, setting ``lead_id`` and ``gap``.

        This is per-vehicle geometry (each agent looks at all others and
        picks the closest one strictly ahead along its heading), so it
        belongs to the actor.  Manager-level concerns (where the agent
        list comes from, what happens between timesteps) live on
        :class:`~navsafe.traffic.semi_reactive.SemiReactiveTraffic`.
        """
        ego_pos = np.asarray(ego_state["position"][:2], dtype=np.float64)

        # All potential leads: agents + ego
        all_vehicles = []
        for a in agent_states:
            all_vehicles.append({
                "id": a["id"],
                "position": np.asarray(a["position"][:2], dtype=np.float64),
                "heading": a["heading"],
                "length": a.get("length", 4.5),
            })
        all_vehicles.append({
            "id": ego_id,
            "position": ego_pos,
            "heading": float(ego_state.get("heading", 0.0)),
            "length": 4.5,
        })

        for agent in agent_states:
            pos = np.asarray(agent["position"][:2], dtype=np.float64)
            heading = agent["heading"]
            fwd = np.array([math.cos(heading), math.sin(heading)])

            best_gap = float("inf")
            best_lead_id = None

            for other in all_vehicles:
                if other["id"] == agent["id"]:
                    continue
                diff = np.asarray(other["position"][:2], dtype=np.float64) - pos
                along = float(np.dot(diff, fwd))
                if along <= 0:
                    continue  # behind or at same position
                # Bumper-to-bumper gap
                gap = along - agent.get("length", 4.5) / 2.0 - other.get("length", 4.5) / 2.0
                if gap < best_gap:
                    best_gap = gap
                    best_lead_id = other["id"]

            if best_lead_id is not None:
                agent["lead_id"] = best_lead_id
                agent["gap"] = max(0.0, best_gap)


__all__ = ["IDMActor", "IDMParams"]
