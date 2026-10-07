"""Ego-mounted contact sensor configuration.

Moved from ``navsafe/envs/scenario_replay_env_cfg.py`` to consolidate
sensor configurations under ``navsafe/component/sensors/``.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class EgoContactSensorCfg:
    """Configuration for an ego-mounted contact sensor.

    Wires :class:`isaaclab.sensors.ContactSensorCfg` so the evaluator
    can replace the heuristic cuboid-IoU collision detection with real
    physics contact reports.

    Attributes:
        prim_path: Body prim to attach the sensor to.  Defaults to the
            ego chassis; you can broaden it (e.g. ``".*chassis|.*bumper"``)
            to capture contacts at multiple parts.
        update_period: Sensor update period in seconds (0.0 = every step).
        history_length: Number of past contact frames to retain (the
            base impl rolls a ring buffer).
        track_pose: Persist contact location in world frame (uses more
            memory but useful for visualisation).
        track_air_time: Track airborne time per body (rarely needed in
            kinematic replay; left off by default).
        debug_vis: Render contact points in viewport when True.
    """
    prim_path: str = "{ENV_REGEX_NS}/ego_vehicle/chassis"
    update_period: float = 0.0  # every sim step
    history_length: int = 1
    track_pose: bool = True
    track_air_time: bool = False
    debug_vis: bool = False
