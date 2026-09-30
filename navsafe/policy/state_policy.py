"""StatePolicy — abstract base for policies consuming structured state.

Requirement 3.2: NexusSim exposes StatePolicy as an abstract subclass
of BasePolicyAdapter for policies whose prepare_input consumes
structured ego/agent/map state vectors.
"""

from __future__ import annotations

from navsafe.policy.base import BasePolicyAdapter


class StatePolicy(BasePolicyAdapter):
    """Abstract base for policies that consume structured state observations.

    Subclass this when your policy's ``prepare_input`` reads ego status,
    agent positions, road graph, or other structured vectors — NOT raw
    camera/LiDAR tensors.

    The only first-party StatePolicy adapter is ``ego_mlp``.
    """

    pass


__all__ = ["StatePolicy"]
