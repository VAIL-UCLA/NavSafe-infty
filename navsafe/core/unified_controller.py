"""
navsafe.core.unified_controller

Unified vehicle controller supporting PID and Pure Pursuit control strategies.

Converts high-level trajectories to low-level control commands.
Adapted from BridgeSim controllers with IsaacSim parameter tuning.

Relocated from ``navsafe.primitives.controller.unified_controller`` as part
of Phase 3 task 3.13 (NexusSim Package Reorg). This module is pure Python
(no IsaacSim imports) and lives in ``navsafe.core``.
"""

import numpy as np
from typing import Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ControllerType(Enum):
    """Available controller types."""
    PID = "pid"
    PURE_PURSUIT = "pure_pursuit"


class UnifiedController:
    """Unified vehicle controller supporting PID and Pure Pursuit.

    Converts high-level trajectory to low-level control commands [steering, throttle, brake].

    This controller uses the following strategies:
    1. **Pure Pursuit**: Geometric method - tracks a look-ahead point on the trajectory
    2. **PID**: Control-theoretic method - minimizes steering error using proportional-integral-derivative

    Both methods convert trajectory waypoints to steering commands while managing speed.

    Example:
        >>> controller = UnifiedController(controller_type="pure_pursuit")
        >>> trajectory = np.array([[1, 0, 0, 1, 0], [2, 0, 0, 1, 0], ...])
        >>> action = controller.compute_action(
        ...     trajectory,
        ...     current_position=np.array([0, 0, 0]),
        ...     current_heading=0.0,
        ...     current_speed=0.5
        ... )
        >>> print(action)  # [steering, throttle/brake, reserved]
    """

    def __init__(
        self,
        controller_type: str = "pure_pursuit",
        vehicle_wheelbase: float = 2.7,
        max_steering_angle: float = 0.6,
        dt: float = 0.1,
        **kwargs
    ):
        """Initialize controller.

        Args:
            controller_type: 'pid' or 'pure_pursuit'
            vehicle_wheelbase: Vehicle wheelbase in meters (default 2.7m for typical sedan)
            max_steering_angle: Maximum steering angle in radians (default ~35 degrees)
            dt: Simulation timestep in seconds (default 0.1s)
            **kwargs: Additional controller-specific parameters
        """
        self.controller_type = ControllerType(controller_type)
        self.wheelbase = vehicle_wheelbase
        self.max_steering = max_steering_angle
        self.dt = dt

        # PID controller state
        self.speed_error_integral = 0.0
        self.steering_error_integral = 0.0

        # PID gains (tuned for IsaacSim physics)
        # These values have been calibrated for realistic vehicle behavior
        self.kp_speed = kwargs.get("kp_speed", 0.8)    # Proportional gain for speed
        self.ki_speed = kwargs.get("ki_speed", 0.1)    # Integral gain for speed
        self.kd_speed = kwargs.get("kd_speed", 0.2)    # Derivative gain for speed

        self.kp_steering = kwargs.get("kp_steering", 0.5)      # Proportional gain for steering
        self.ki_steering = kwargs.get("ki_steering", 0.05)     # Integral gain for steering
        self.kd_steering = kwargs.get("kd_steering", 0.15)     # Derivative gain for steering

        # Pure Pursuit specific parameters
        self.lookahead_distance = kwargs.get("lookahead_distance", 5.0)  # Default 5 meter lookahead
        self.speed_dependent_lookahead = kwargs.get("speed_dependent_lookahead", True)

        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_action(
        self,
        trajectory: np.ndarray,
        current_position: np.ndarray,
        current_heading: float,
        current_speed: float,
        current_frame: int = 0,
    ) -> np.ndarray:
        """Compute control action from trajectory.

        Args:
            trajectory: Planned trajectory (T, 5) with columns [x, y, z, vx, vy]
            current_position: Current position (x, y, z) as numpy array
            current_heading: Current heading in radians
            current_speed: Current speed in m/s
            current_frame: Current frame number (for logging/debugging)

        Returns:
            np.ndarray: Control action [steering, throttle/brake, reserved] each in [-1, 1]
                - steering: Steering angle normalized to [-1, 1]
                - throttle/brake: Acceleration command (-1 = max brake, +1 = max throttle)
                - reserved: Future use
        """
        # Handle empty trajectory gracefully
        if trajectory is None or len(trajectory) == 0:
            self.logger.warning(f"Frame {current_frame}: Empty trajectory, returning zero action")
            return np.array([0.0, 0.0, 0.0])

        # Ensure trajectory is numpy array
        trajectory = np.asarray(trajectory)
        current_position = np.asarray(current_position)

        # Find reference point on trajectory
        ref_point = self._find_reference_point(trajectory, current_position)

        # Compute steering based on controller type
        if self.controller_type == ControllerType.PURE_PURSUIT:
            steering = self._compute_pure_pursuit_steering(
                current_position, current_heading, ref_point
            )
        else:  # PID
            steering = self._compute_pid_steering(
                current_position, current_heading, ref_point
            )

        # Compute speed control
        target_speed = ref_point[3] if len(ref_point) > 3 else 1.0  # vx from trajectory
        throttle, brake = self._compute_speed_control(current_speed, target_speed)

        # Clamp outputs to valid ranges
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        # Normalize steering to [-1, 1] range
        steering_normalized = steering / self.max_steering if self.max_steering != 0 else 0.0

        # Combine throttle and brake into single command
        # Positive = throttle, Negative = brake
        throttle_brake = throttle - brake

        return np.array([steering_normalized, throttle_brake, 0.0])

    def _find_reference_point(
        self,
        trajectory: np.ndarray,
        current_position: np.ndarray,
    ) -> np.ndarray:
        """Find reference point on trajectory for control.

        Strategy:
        1. Find closest point on trajectory to current position
        2. Look ahead by either fixed distance or speed-dependent distance
        3. Return the point at that lookahead distance

        Args:
            trajectory: Planned trajectory (T, 5)
            current_position: Current position (3,)

        Returns:
            np.ndarray: Reference point (5,) or trajectory[0] if lookahead fails
        """
        if len(trajectory) == 0:
            return np.array([0., 0., 0., 1., 0.])

        # Find point on trajectory closest to current position
        positions = trajectory[:, :3]
        distances_to_current = np.linalg.norm(positions - current_position, axis=1)
        closest_idx = int(np.argmin(distances_to_current))

        # Compute lookahead distance
        if self.speed_dependent_lookahead:
            # Increase lookahead with speed (1 second lookahead)
            lookahead = max(self.lookahead_distance * 0.5, current_position[0] * 1.0) if len(current_position) > 0 else self.lookahead_distance
        else:
            lookahead = self.lookahead_distance

        # Find point at lookahead distance
        # Simple heuristic: use trajectory point at closest_idx + lookahead_frames
        lookahead_frames = max(1, int(lookahead / 1.0))  # Assuming 1m per frame
        lookahead_idx = min(closest_idx + lookahead_frames, len(trajectory) - 1)

        return trajectory[lookahead_idx]

    def _compute_pure_pursuit_steering(
        self,
        current_position: np.ndarray,
        current_heading: float,
        reference_point: np.ndarray,
    ) -> float:
        """Compute steering using Pure Pursuit control.

        Pure Pursuit is a geometric tracking method that:
        1. Identifies a look-ahead point on the desired trajectory
        2. Computes the steering angle needed to pass through that point
        3. Uses the kinematic bicycle model: tan(delta) = 2*L*sin(alpha) / ld
           where delta=steering angle, L=wheelbase, alpha=heading error, ld=lookahead distance

        Args:
            current_position: Current position (x, y, z)
            current_heading: Current heading in radians
            reference_point: Target point on trajectory

        Returns:
            float: Steering angle in radians
        """
        ref_x = reference_point[0]
        ref_y = reference_point[1]
        curr_x = current_position[0]
        curr_y = current_position[1]

        # Vector from current to reference
        dx = ref_x - curr_x
        dy = ref_y - curr_y
        distance = np.sqrt(dx**2 + dy**2)

        # If very close to reference, don't steer
        if distance < 0.1:
            return 0.0

        # Angle to reference point
        angle_to_ref = np.arctan2(dy, dx)

        # Cross-track error (heading error)
        heading_error = angle_to_ref - current_heading

        # Normalize heading error to [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # Pure Pursuit steering law
        # From bicycle model: sin(delta) = sin(alpha) * 2*L / ld
        steering = np.arctan2(2.0 * self.wheelbase * np.sin(heading_error), distance)

        return steering

    def _compute_pid_steering(
        self,
        current_position: np.ndarray,
        current_heading: float,
        reference_point: np.ndarray,
    ) -> float:
        """Compute steering using PID control.

        PID (Proportional-Integral-Derivative) is a feedback control method that:
        1. Computes cross-track error (deviation from desired heading)
        2. Applies proportional, integral, and derivative terms to minimize error
        3. More responsive than Pure Pursuit but requires tuning

        Args:
            current_position: Current position (x, y, z)
            current_heading: Current heading in radians
            reference_point: Target point on trajectory

        Returns:
            float: Steering angle in radians
        """
        ref_x = reference_point[0]
        ref_y = reference_point[1]
        curr_x = current_position[0]
        curr_y = current_position[1]

        # Desired heading to reference point
        desired_heading = np.arctan2(ref_y - curr_y, ref_x - curr_x)

        # Heading error
        heading_error = desired_heading - current_heading

        # Normalize to [-π, π]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # PID update
        self.steering_error_integral += heading_error * self.dt
        # Prevent integral windup
        self.steering_error_integral = np.clip(self.steering_error_integral, -1.0, 1.0)

        # PID control law
        steering = (
            self.kp_steering * heading_error +
            self.ki_steering * self.steering_error_integral
        )

        return steering

    def _compute_speed_control(
        self,
        current_speed: float,
        target_speed: float,
    ) -> tuple:
        """Compute throttle and brake to reach target speed.

        Args:
            current_speed: Current speed in m/s
            target_speed: Target speed in m/s

        Returns:
            Tuple of (throttle, brake) both in [0, 1]
        """
        # Speed error
        speed_error = target_speed - current_speed

        # PID update
        self.speed_error_integral += speed_error * self.dt
        # Prevent integral windup
        self.speed_error_integral = np.clip(self.speed_error_integral, -1.0, 1.0)

        # PID control law
        control = (
            self.kp_speed * speed_error +
            self.ki_speed * self.speed_error_integral
        )

        # Split into throttle (positive) and brake (negative)
        throttle = max(0.0, control)
        brake = max(0.0, -control)

        return throttle, brake

    def reset(self):
        """Reset controller state.

        Should be called at the beginning of each new scenario or episode.
        Clears integral accumulation to prevent integral windup.
        """
        self.speed_error_integral = 0.0
        self.steering_error_integral = 0.0
        self.logger.debug("Controller state reset")

    def set_gains(
        self,
        kp_speed: Optional[float] = None,
        ki_speed: Optional[float] = None,
        kd_speed: Optional[float] = None,
        kp_steering: Optional[float] = None,
        ki_steering: Optional[float] = None,
        kd_steering: Optional[float] = None,
    ) -> None:
        """Update PID gains at runtime.

        Useful for tuning and experimentation during evaluation.

        Args:
            kp_speed: Proportional gain for speed control
            ki_speed: Integral gain for speed control
            kd_speed: Derivative gain for speed control
            kp_steering: Proportional gain for steering control
            ki_steering: Integral gain for steering control
            kd_steering: Derivative gain for steering control
        """
        if kp_speed is not None:
            self.kp_speed = kp_speed
        if ki_speed is not None:
            self.ki_speed = ki_speed
        if kd_speed is not None:
            self.kd_speed = kd_speed
        if kp_steering is not None:
            self.kp_steering = kp_steering
        if ki_steering is not None:
            self.ki_steering = ki_steering
        if kd_steering is not None:
            self.kd_steering = kd_steering

        self.logger.info(f"Updated PID gains: speed=({self.kp_speed}, {self.ki_speed}), steering=({self.kp_steering}, {self.ki_steering})")
