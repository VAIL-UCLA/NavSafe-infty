"""Ego-mounted ray-cast LiDAR configuration.

Moved from ``navsafe/envs/scenario_replay_env_cfg.py`` to consolidate
sensor configurations under ``navsafe/component/sensors/``.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class EgoLidarCfg:
    """Configuration for an ego-mounted ray-cast LiDAR.

    Wires :class:`isaaclab.sensors.RayCasterCfg` with a Velodyne-style
    azimuth/elevation pattern by default.  The actual sensor is
    instantiated by :class:`ScenarioReplayEnv` only when this cfg is
    non-None.

    Attributes:
        prim_path: USD prim path of the body the LiDAR is attached to.
            Defaults to the ego vehicle's chassis prim under the env
            namespace.  ``"{ENV_REGEX_NS}"`` placeholders are honored by
            IsaacLab.
        offset_pos: ``(x, y, z)`` offset from the parent body in metres.
        update_period: Sensor update period in seconds (0.0 = every step).
        max_distance: Maximum return range in metres.
        horizontal_fov_deg: Total horizontal FoV (degrees).
        horizontal_res_deg: Angular resolution per beam (degrees).
        vertical_channels: Number of beam rows (e.g. 32 = HDL-32E,
            64 = HDL-64E, 128 = VLS-128).
        vertical_fov_min_deg / vertical_fov_max_deg: Vertical span.
        attach_yaw_only: Whether to track yaw of the parent (default
            True so the sensor follows ego steering but ignores pitch/roll).
        debug_vis: Render rays in IsaacSim viewport when True.
    """
    prim_path: str = "{ENV_REGEX_NS}/ego_vehicle/chassis"
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 1.85)
    update_period: float = 0.1
    max_distance: float = 100.0
    horizontal_fov_deg: float = 360.0
    horizontal_res_deg: float = 0.4
    vertical_channels: int = 32
    vertical_fov_min_deg: float = -30.0
    vertical_fov_max_deg: float = 10.0
    attach_yaw_only: bool = True
    debug_vis: bool = False
