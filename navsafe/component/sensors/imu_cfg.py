"""Ego-mounted IMU configuration.

Moved from ``navsafe/envs/scenario_replay_env_cfg.py`` to consolidate
sensor configurations under ``navsafe/component/sensors/``.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class EgoImuCfg:
    """Configuration for an ego-mounted IMU.

    Wires :class:`isaaclab.sensors.ImuCfg`.  Outputs (per step) consumed
    by :meth:`ScenarioReplayEnv.get_imu_state`:

    * ``lin_acc_b`` — body-frame linear acceleration (m/s²).
    * ``ang_vel_b`` — body-frame angular velocity (rad/s).
    * ``quat_w`` — orientation as a (w, x, y, z) quaternion.
    """
    prim_path: str = "{ENV_REGEX_NS}/ego_vehicle/chassis"
    update_period: float = 0.0  # every sim step
    gravity_bias: tuple[float, float, float] = (0.0, 0.0, 9.81)
    debug_vis: bool = False
