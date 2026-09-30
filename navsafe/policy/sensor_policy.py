"""SensorPolicy — abstract base for policies consuming sensor observations.

Requirement 3.2: NexusSim exposes SensorPolicy as an abstract subclass
of BasePolicyAdapter for policies whose prepare_input consumes camera,
depth, semseg, or LiDAR observations.
"""

from __future__ import annotations

from navsafe.policy.base import BasePolicyAdapter


class SensorPolicy(BasePolicyAdapter):
    """Abstract base for policies that consume sensor observations.

    Subclass this when your policy's ``prepare_input`` reads camera
    images, depth maps, semantic segmentation, or LiDAR point clouds.

    All first-party adapters except ``ego_mlp`` are SensorPolicy
    subclasses.
    """

    pass


__all__ = ["SensorPolicy"]
