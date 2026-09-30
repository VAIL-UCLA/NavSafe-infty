"""Abstract base class for scene rendering backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class SceneRenderer(ABC):
    """Abstract base class for scene rendering backends.

    All renderers must implement setup, get_camera_images, and update_agents.
    The close method is optional and defaults to a no-op.
    """

    @abstractmethod
    def setup(self, scenario_data: dict, env: Any = None) -> None:
        """Initialize the renderer with scenario data.

        Args:
            scenario_data: Dict containing map_features, traffic_lights, etc.
            env: Optional simulation environment (e.g. IsaacSim).
        """

    @abstractmethod
    def get_camera_images(
        self,
        ego_state: dict,
        cam_configs: dict,
        agent_states: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """Render camera images for the current ego state.

        Args:
            ego_state: dict with position (3,), heading (float), velocity (3,).
            cam_configs: dict mapping cam_name -> config dict with keys
                x, y, z, yaw, pitch, roll, fov, width, height.
            agent_states: optional list of dicts with position, heading,
                length, width, height, type.

        Returns:
            Dict mapping cam_name -> (H, W, 3) uint8 BGR image.
        """

    @abstractmethod
    def update_agents(self, agent_states: list) -> None:
        """Update agent positions for rendering.

        Args:
            agent_states: list of dicts with position, heading, length,
                width, height, type.
        """

    # ------------------------------------------------------------------
    # Per-agent capture (default loop; concrete renderers may override)
    # ------------------------------------------------------------------
    def get_agent_camera_images(
        self,
        agent_id: str,
        observer_pose: dict,
        cam_configs: dict,
        agent_states: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """Render cameras for an arbitrary observer agent.

        The default implementation ignores ``agent_id`` and reuses
        :meth:`get_camera_images` with the observer's pose in the
        ``ego_state`` slot, which is correct for stateless renderers.
        Subclasses should override when a meaningful per-agent distinction
        is needed, such as excluding the observer's own body from the image.

        Args:
            agent_id: Observer agent id (opaque; may be used by overrides
                to route the capture to a per-agent prim).
            observer_pose: ``{position, heading, ...}`` in world frame.
            cam_configs: Rig dict keyed by camera name.
            agent_states: Scene content (other agents).  When the observer
                is a non-ego agent, it is that override's responsibility
                to exclude the observer from this list.

        Returns:
            ``{cam_name: (H, W, 3) uint8 BGR}``.
        """
        return self.get_camera_images(
            ego_state=observer_pose,
            cam_configs=cam_configs,
            agent_states=agent_states,
        )

    def set_timestep(self, sim_step: int) -> None:
        """Tell the renderer which scenario timestep to render next.

        Optional hook with a no-op default. Time-varying backends (e.g. the
        4D Gaussian-splat renderer) override this to select the frame matching
        the current sim step; backends without a temporal state may ignore it.
        Envs should call this every capture so backends never have to reach back
        into env state to discover the timestep.

        Args:
            sim_step: The current scenario timestep index (env-owned).
        """

    def close(self) -> None:
        """Clean up resources. Override if needed."""
