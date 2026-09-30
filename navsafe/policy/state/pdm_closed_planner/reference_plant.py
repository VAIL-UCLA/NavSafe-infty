# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The reference PDM-Closed plant, for the SCORED copy of every proposal.

A single-vehicle transcription of CaRL's ``BatchKinematicBicycleModel``
(``carl_nuplan/.../simulation/batch_kinematic_bicycle.py``, commit
``2677d14``), which is nuPlan's ``KinematicBicycleModel`` in array form:

* the reference point is the **rear axle**; ``heading_dot = v tan(δ) / L``;
* first-order actuator lags on acceleration (0.2 s) and steering angle
  (0.05 s), applied in ``_update_commands``: the pose propagates under the
  OLD steering angle, the velocity under the UPDATED (filtered) acceleration;
* the steering angle is integrated with the filtered rate and clipped to
  ``±max_steering_angle`` AFTER the filter; the recorded steering rate is the
  unclipped filtered rate;
* no velocity floor, no acceleration saturation, no speed cap — the model
  is exactly ``forward_integrate`` on every channel.

The eleven channels follow the reference ``StateIndex`` order: x, y,
heading, velocity_x, velocity_y, acceleration_x, acceleration_y,
steering_angle, steering_rate, angular_velocity, angular_acceleration
(velocities/accelerations are body-frame longitudinal/lateral; the lateral
ones are identically zero).

NexusSim carries the ego pose at the wheelbase CENTRE and the shared
:class:`~navsafe.core.ego_dynamics.EgoDynamics` is a centre-referenced
symmetric bicycle that every policy executes through. The planner only
needs the reference plant for the copy it scores, so this class lives in
the planner package and :func:`~.forward_sim.simulate_proposal` converts
between the two conventions at its boundary
(``centre = rear_axle + (L/2)·[cos h, sin h]``). Measured against the
reference simulator on identical inputs, the tracked copy is then
bit-identical (max |Δ| 9e-16 over 4 s); with the centre plant the
residual was 0.1 m (lane change) to 0.3 m (R = 30 m) — the
centre-of-wheelbase kinematics and the course-heading adaptation it
needed, not the tracker.
"""

from __future__ import annotations

import math

import numpy as np

# Reference ``StateIndex`` channel positions (``utils/pdm_enums.py``).
X, Y, HEADING = 0, 1, 2
VELOCITY_X, VELOCITY_Y = 3, 4
ACCELERATION_X, ACCELERATION_Y = 5, 6
STEERING_ANGLE, STEERING_RATE = 7, 8
ANGULAR_VELOCITY, ANGULAR_ACCELERATION = 9, 10
STATE_SIZE = 11


def _principal_value(angle: float) -> float:
    """nuPlan ``principal_value``: wrap into ``[-π, π)``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class ReferenceBicyclePlant:
    """Rear-axle kinematic bicycle with the reference's actuator lags.

    Args:
        wheelbase: Distance between the axles (m).
        max_steering_angle: Steering-angle bound (rad), applied after the
            filter like ``propagate_state``.
        accel_time_constant: Acceleration low-pass time constant (s).
        steering_angle_time_constant: Steering low-pass time constant (s).
    """

    def __init__(
        self,
        *,
        wheelbase: float,
        max_steering_angle: float,
        accel_time_constant: float = 0.2,
        steering_angle_time_constant: float = 0.05,
    ) -> None:
        if wheelbase <= 0.0:
            raise ValueError(f"wheelbase must be positive, got {wheelbase}")
        if max_steering_angle <= 0.0:
            raise ValueError(
                f"max_steering_angle must be positive, got {max_steering_angle}")
        self.wheelbase = float(wheelbase)
        self.max_steering_angle = float(max_steering_angle)
        self.accel_time_constant = float(accel_time_constant)
        self.steering_angle_time_constant = float(steering_angle_time_constant)
        self.state = np.zeros(STATE_SIZE, dtype=np.float64)

    # ------------------------------------------------------------------
    # Seeding (``ego_state_to_state_array``)
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        rear_x: float,
        rear_y: float,
        heading: float,
        velocity: float,
        acceleration: float = 0.0,
        steering_angle: float = 0.0,
        steering_rate: float = 0.0,
        angular_velocity: float = 0.0,
        angular_acceleration: float = 0.0,
    ) -> None:
        """Seed the eleven channels from the live ego state.

        Mirrors ``ego_state_to_state_array``: rear-axle pose, longitudinal
        velocity and acceleration (lateral channels zero), tire steering
        angle and rate, angular velocity and acceleration.
        """
        s = self.state
        s[:] = 0.0
        s[X], s[Y], s[HEADING] = float(rear_x), float(rear_y), float(heading)
        s[VELOCITY_X] = float(velocity)
        s[ACCELERATION_X] = float(acceleration)
        s[STEERING_ANGLE] = float(steering_angle)
        s[STEERING_RATE] = float(steering_rate)
        s[ANGULAR_VELOCITY] = float(angular_velocity)
        s[ANGULAR_ACCELERATION] = float(angular_acceleration)

    # ------------------------------------------------------------------
    # One reference step (``_update_commands`` + ``propagate_state``)
    # ------------------------------------------------------------------

    def step(self, *, acceleration_cmd: float, steering_rate_cmd: float,
             dt: float) -> np.ndarray:
        """Advance one step under ``(acceleration, steering_rate)`` commands.

        Returns the new state array (also kept in :attr:`state`).
        """
        s = self.state
        dt = float(dt)
        accel = float(s[ACCELERATION_X])
        steering_angle = float(s[STEERING_ANGLE])

        # _update_commands: first-order lags. The ideal steering angle is
        # the commanded rate integrated for one step on top of the current
        # angle; the command itself is never clipped.
        ideal_accel = float(acceleration_cmd)
        ideal_steering_angle = dt * float(steering_rate_cmd) + steering_angle
        updated_accel = (
            dt / (dt + self.accel_time_constant) * (ideal_accel - accel) + accel
        )
        updated_steering_angle = (
            dt / (dt + self.steering_angle_time_constant)
            * (ideal_steering_angle - steering_angle) + steering_angle
        )
        updated_steering_rate = (updated_steering_angle - steering_angle) / dt

        # get_state_dot on the propagating state: the pose derivative reads
        # the OLD steering angle and the OLD velocity; the velocity
        # derivative is the UPDATED acceleration.
        v = float(s[VELOCITY_X])
        heading = float(s[HEADING])
        x_dot = v * math.cos(heading)
        y_dot = v * math.sin(heading)
        heading_dot = v * math.tan(steering_angle) / self.wheelbase

        out = s.copy()
        out[X] = s[X] + x_dot * dt
        out[Y] = s[Y] + y_dot * dt
        out[HEADING] = _principal_value(heading + heading_dot * dt)
        out[VELOCITY_X] = v + updated_accel * dt
        out[VELOCITY_Y] = 0.0
        out[STEERING_ANGLE] = float(np.clip(
            steering_angle + updated_steering_rate * dt,
            -self.max_steering_angle, self.max_steering_angle))
        out[ANGULAR_VELOCITY] = (
            out[VELOCITY_X] * math.tan(out[STEERING_ANGLE]) / self.wheelbase
        )
        out[ACCELERATION_X] = updated_accel
        out[ACCELERATION_Y] = 0.0
        out[ANGULAR_ACCELERATION] = (
            out[ANGULAR_VELOCITY] - s[ANGULAR_VELOCITY]) / dt
        out[STEERING_RATE] = updated_steering_rate
        self.state = out
        return out

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def rear_x(self) -> float:
        return float(self.state[X])

    @property
    def rear_y(self) -> float:
        return float(self.state[Y])

    @property
    def heading(self) -> float:
        return float(self.state[HEADING])

    @property
    def velocity(self) -> float:
        """Signed rear-axle longitudinal velocity (may dip below zero)."""
        return float(self.state[VELOCITY_X])

    @property
    def steering_angle(self) -> float:
        return float(self.state[STEERING_ANGLE])

    def centre(self) -> tuple[float, float]:
        """Wheelbase-centre position: ``rear_axle + (L/2)·[cos h, sin h]``."""
        half = 0.5 * self.wheelbase
        return (
            self.rear_x + half * math.cos(self.heading),
            self.rear_y + half * math.sin(self.heading),
        )


__all__ = [
    "ReferenceBicyclePlant",
    "STATE_SIZE",
    "X", "Y", "HEADING", "VELOCITY_X", "VELOCITY_Y",
    "ACCELERATION_X", "ACCELERATION_Y", "STEERING_ANGLE", "STEERING_RATE",
    "ANGULAR_VELOCITY", "ANGULAR_ACCELERATION",
]
