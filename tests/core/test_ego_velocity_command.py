# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for EgoDynamics.compute_velocity_command (physics mode T2.1).

The analytic guarantee behind the physics execution mode: writing the
commanded velocities to a body and integrating for one dt must reproduce
``EgoDynamics.step``'s pose delta exactly. These tests integrate the command
in pure Python (x += vx*dt, heading += wz*dt) and compare against step().
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg


def test_full_lock_has_physical_symmetric_bicycle_turning_radius():
    """The wheelbase is the axle-to-axle length, not twice that length."""
    cfg = EgoDynamicsCfg(wheelbase=2.8, max_steer_angle=0.6)
    ego = EgoDynamics(cfg)
    ego.reset(speed=1.0)
    cmd = ego.compute_velocity_command(steer=-1.0, accel=0.0)
    turning_radius = ego.speed / abs(cmd["yaw_rate"])
    beta = np.arctan(0.5 * np.tan(cfg.max_steer_angle))
    expected = cfg.wheelbase / (2.0 * np.sin(beta))
    assert turning_radius == pytest.approx(expected)
    assert turning_radius < 4.4


@pytest.mark.parametrize("steer,accel,speed,heading", [
    (0.0, 0.5, 3.0, 0.0),      # straight, accelerating
    (0.4, 0.0, 5.0, 1.2),      # left-biased turn at cruise
    (-0.8, -0.5, 8.0, -2.0),   # hard right, braking
    (1.0, 1.0, 0.0, 0.5),      # full lock from standstill
    (0.2, -1.0, 14.9, 3.0),    # near max speed, full brake
])
def test_velocity_command_integrates_to_step_pose(steer, accel, speed, heading):
    cfg = EgoDynamicsCfg(dt=0.1)
    ego_a = EgoDynamics(cfg)
    ego_a.reset(x=2.0, y=-3.0, heading=heading, speed=speed)
    ego_b = EgoDynamics(cfg)
    ego_b.reset(x=2.0, y=-3.0, heading=heading, speed=speed)

    cmd = ego_a.compute_velocity_command(steer, accel)
    # compute_velocity_command must not mutate the ego.
    assert ego_a.x == 2.0 and ego_a.y == -3.0 and ego_a.speed == speed

    ego_b.step(steer, accel)

    # Integrate the command for one dt (what PhysX does with the velocities).
    x = 2.0 + cmd["lin_vel_w"][0] * cfg.dt
    y = -3.0 + cmd["lin_vel_w"][1] * cfg.dt
    h = heading + cmd["yaw_rate"] * cfg.dt

    assert x == pytest.approx(ego_b.x, abs=1e-12)
    assert y == pytest.approx(ego_b.y, abs=1e-12)
    assert h == pytest.approx(ego_b.heading, abs=1e-12)
    assert cmd["speed"] == pytest.approx(ego_b.speed, abs=1e-12)


def test_apply_external_state_bookkeeping():
    ego = EgoDynamics(EgoDynamicsCfg(dt=0.1))
    frames_before = ego.frame
    traj_before = len(ego.trajectory)
    state = ego.apply_external_state(1.0, 2.0, 0.3, 4.0)
    assert ego.x == 1.0 and ego.y == 2.0
    assert ego.heading == 0.3 and ego.speed == 4.0
    assert ego.frame == frames_before + 1
    assert len(ego.trajectory) == traj_before + 1
    assert state["speed_kmh"] == pytest.approx(4.0 * 3.6)


def test_apply_external_state_clips_speed():
    ego = EgoDynamics(EgoDynamicsCfg(dt=0.1, max_speed=15.0))
    ego.apply_external_state(0.0, 0.0, 0.0, 99.0)
    assert ego.speed == 15.0
