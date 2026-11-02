# Copyright (c) 2022-2025, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Replay agent manager for ScenarioDescription format.

Indexes the scenario's tracks and answers per-timestep state queries for the
env (collision, TTC and renderer agent lists) and the traffic managers. The
NuRec renderer draws every agent from the reconstruction, so nothing here
builds or poses scene geometry.
"""

import logging
import numpy as np
from typing import Dict, Optional

from navsafe.scenario.scenario_description import ScenarioDescription as SD

logger = logging.getLogger(__name__)


class ScenarioReplayManager:
    """
    Per-timestep access to the replay agents of a ScenarioDescription.

    Handles:
    - Looking up an agent's logged state at a timestep
    - Tracking the replay cursor the traffic managers advance
    """

    def __init__(self, scenario: SD, num_envs: int = 1):
        """
        Args:
            scenario: ScenarioDescription object
            num_envs: Number of parallel environments
        """
        self.scenario = scenario
        self.num_envs = num_envs

        self.tracks = scenario[SD.TRACKS]
        self.metadata = scenario[SD.METADATA]
        self.scenario_length = scenario[SD.LENGTH]

        # Agent data structures
        self.agent_ids = list(self.tracks.keys())
        self.num_agents = len(self.agent_ids)

        # Mapping from agent_id to index
        self.agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}

        # Current timestep
        self.current_timestep = 0

        # SDC / ego vehicle ID from scenario metadata
        self.ego_agent_id = str(self.metadata.get(SD.SDC_ID, "")) or None

        print(f"[ReplayManager] Initialized with {self.num_agents} agents, {self.scenario_length} timesteps")

    def _get_initial_valid_state(self, track_data: Dict) -> Optional[Dict]:
        """Get first valid state for initialization."""
        state = track_data[SD.STATE]
        valid = state.get('valid', np.ones(len(state['position'])))

        for t in range(len(valid)):
            if valid[t]:
                return self._get_state_at_timestep(track_data, t)

        return None

    def _get_state_at_timestep(self, track_data: Dict, timestep: int) -> Optional[Dict]:
        """Extract state at given timestep."""
        state = track_data[SD.STATE]

        if timestep < 0 or timestep >= len(state['position']):
            return None

        result = {
            'position': np.array(state['position'][timestep]),
            'heading': float(state['heading'][timestep]),
            'valid': state.get('valid', np.ones(len(state['position'])))[timestep]
        }

        if 'velocity' in state:
            result['velocity'] = np.array(state['velocity'][timestep])

        if 'length' in state:
            result['length'] = float(state['length'][timestep])
        if 'width' in state:
            result['width'] = float(state['width'][timestep])
        if 'height' in state:
            result['height'] = float(state['height'][timestep])

        return result

    def get_agent_state(self, agent_id: str, timestep: int) -> Optional[Dict]:
        """Return the world-frame state dict for ``agent_id`` at ``timestep``.

        Thin public wrapper around :meth:`_get_state_at_timestep` that
        resolves the agent id to the underlying track first, so callers
        don't have to reach into private track storage.

        Args:
            agent_id: Track id as it appears in ``scenario_data['tracks']``.
            timestep: Scenario timestep.

        Returns:
            ``{'position': ndarray, 'heading': float, 'valid': bool, ...}``
            or ``None`` if the agent is unknown or the timestep is out of
            range.
        """
        track = self.tracks.get(agent_id) if hasattr(self, "tracks") else None
        if track is None:
            return None
        return self._get_state_at_timestep(track, timestep)

    def get_initial_state(self, agent_id: str) -> Optional[Dict]:
        """First-valid-timestep state for ``agent_id`` (or ``None``).

        Public wrapper around :meth:`_get_initial_valid_state` for callers
        that need spawn dims without reaching into private track storage.
        """
        track = self.tracks.get(agent_id) if hasattr(self, "tracks") else None
        if track is None:
            return None
        return self._get_initial_valid_state(track)

    def update_agents(self, timestep: int, skip_ego: bool = False,
                      skip_agents=None):
        """
        Advance the replay cursor to ``timestep``.

        Args:
            timestep: Current timestep
            skip_ego: The ego is controlled externally (kept for the traffic
                managers' call signature; the cursor is shared)
            skip_agents: Agent ids driven by a traffic manager (same)
        """
        if timestep >= self.scenario_length:
            # Agents hold their final logged pose past the end of the log.
            if not getattr(self, "_overrun_logged", False):
                self._overrun_logged = True
                print(f"[ReplayManager] episode ran past the log "
                      f"(timestep {timestep} >= length {self.scenario_length});"
                      f" agents hold their final logged pose")
            timestep = max(0, self.scenario_length - 1)
        self.current_timestep = timestep

    def get_agent_info(self) -> Dict:
        """Get information about the replay agents."""
        info = {
            'total_agents': self.num_agents,
            'scenario_length': self.scenario_length,
            'current_timestep': self.current_timestep,
            'num_envs': self.num_envs
        }

        # Count by type
        type_counts: dict[str, int] = {}
        for track_data in self.tracks.values():
            agent_type = track_data[SD.TYPE]
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

        info['agents_by_type'] = type_counts

        return info

    def reset(self, timestep: int = 0):
        """Reset the replay cursor to given timestep."""
        self.update_agents(timestep)
