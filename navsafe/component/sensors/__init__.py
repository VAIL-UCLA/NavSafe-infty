"""Sensor components for NexusSim.

This subpackage consolidates sensor configurations and implementations
attached to driving agents (ego or non-ego):

**Other sensors (IsaacLab wrappers):**
- :class:`EgoLidarCfg` — ray-cast LiDAR with Velodyne-style patterns.
- :class:`EgoImuCfg` — body-frame linear/angular acceleration + orientation.
- :class:`EgoContactSensorCfg` — physics-based contact detection.
- :class:`EgoFrameTransformerCfg` — SE(3) pose tracking ego-to-world.
"""

import logging as _logging

_logger = _logging.getLogger(__name__)

# Sensor configurations (always available — pure dataclasses)
from navsafe.component.sensors.contact_sensor_cfg import EgoContactSensorCfg
from navsafe.component.sensors.frame_transformer_cfg import EgoFrameTransformerCfg
from navsafe.component.sensors.imu_cfg import EgoImuCfg
from navsafe.component.sensors.lidar_cfg import EgoLidarCfg

__all__ = [
    # Other sensors
    "EgoContactSensorCfg",
    "EgoFrameTransformerCfg",
    "EgoImuCfg",
    "EgoLidarCfg",
]
