"""Pure-Python ego dynamics and collision detection.

Extracted from ``navsafe.envs.base_driving_env`` as part of Phase 3
task 3.25 (NavSafe Package Reorg). This module is pure Python with
NumPy/Shapely dependencies only — no IsaacSim imports.

Provides:
- ``EgoDynamics``: Bicycle-model ego vehicle dynamics
- ``CollisionDetector``: Shapely-based polygon collision detection
- ``BaseDrivingEnvCfg``: Configuration dataclass for ego physics params

Requirements: 2.4 (fold base_driving_env into NavSafeEnv core).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan, cos, sin, tan
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class EgoDynamicsCfg:
    """Configuration for ego vehicle dynamics (bicycle model).

    Extracted from ``BaseDrivingEnvCfg`` — the physics/dynamics subset
    that belongs in ``navsafe.core`` (pure Python, no IsaacSim).
    """

    dt: float = 0.1
    """Simulation timestep in seconds (10 Hz default)."""

    max_speed: float = 15.0
    """Maximum ego speed in m/s."""

    max_steer_angle: float = 0.6
    """Maximum steering angle in radians."""

    max_accel: float = 3.0
    """Maximum acceleration in m/s²."""

    max_brake: float = 5.0
    """Maximum braking deceleration in m/s²."""

    wheelbase: float = 2.8
    """Vehicle wheelbase in metres."""

    ego_length: float = 4.5
    """Ego vehicle length in metres."""

    ego_width: float = 1.8
    """Ego vehicle width in metres."""

    ego_height: float = 1.5
    """Ego vehicle height in metres."""


class EgoDynamics:
    """Bicycle-model ego vehicle dynamics.

    Pure-Python implementation of the kinematic bicycle model used for
    ego vehicle simulation. Extracted from ``BaseDrivingEnv.step_ego``.

    Example:
        >>> cfg = EgoDynamicsCfg(dt=0.1, wheelbase=2.8)
        >>> ego = EgoDynamics(cfg)
        >>> state = ego.step(steer=0.5, accel=0.3)
        >>> print(state["x"], state["y"])
    """

    def __init__(self, cfg: EgoDynamicsCfg | None = None):
        self.cfg = cfg or EgoDynamicsCfg()
        self.x: float = 0.0
        self.y: float = 0.0
        self.heading: float = 0.0
        self.speed: float = 0.0
        self.frame: int = 0
        self.trajectory: List[Dict] = []

    def step(self, steer: float, accel: float) -> Dict[str, Any]:
        """Advance ego one timestep using bicycle model.

        Args:
            steer: Normalized steering input (-1 to 1), negative=left, positive=right.
            accel: Normalized accel input (-1 to 1), negative=brake, positive=throttle.

        Returns:
            Updated ego state dict.
        """
        steer = float(np.clip(steer, -1.0, 1.0))
        accel_input = float(np.clip(accel, -1.0, 1.0))

        # Map to physical values (negate steer so positive=right)
        steer_angle = -steer * self.cfg.max_steer_angle
        if accel_input >= 0:
            accel_val = accel_input * self.cfg.max_accel
        else:
            accel_val = accel_input * self.cfg.max_brake

        # Bicycle model
        dt = self.cfg.dt
        beta = atan(0.5 * tan(steer_angle))
        self.x += self.speed * cos(self.heading + beta) * dt
        self.y += self.speed * sin(self.heading + beta) * dt
        # Symmetric kinematic bicycle, with the state at the wheelbase
        # centre.  beta = atan(l_r / L * tan(delta)) and yaw rate is
        # v / l_r * sin(beta).  Here l_r = L / 2, hence the factor 2 / L.
        # Omitting it doubles the minimum turning radius (8.55 m instead of
        # 4.27 m for NavSafe's L=2.8 m, delta=0.6 rad) and makes tight but
        # physically valid logged turns impossible to reproduce.
        self.heading += (2.0 * self.speed / self.cfg.wheelbase) * sin(beta) * dt
        self.speed += accel_val * dt
        self.speed = float(np.clip(self.speed, 0.0, self.cfg.max_speed))

        self.frame += 1

        state = self.get_state()
        self.trajectory.append(state)
        return state

    def compute_velocity_command(self, steer: float, accel: float) -> Dict[str, Any]:
        """Bicycle-model velocity command for one step — without integrating.

        Mirrors :meth:`step` exactly (same clipping, sign conventions and
        slip-angle beta) but returns the commanded world-frame velocities
        instead of mutating the pose. The physics execution mode writes
        these to the PhysX rigid body and lets the solver integrate; a
        PhysX step of ``dt`` then reproduces :meth:`step`'s pose delta.

        Returns:
            ``{"lin_vel_w": (vx, vy, 0.0), "yaw_rate": wz, "speed": v_new}``.
            ``lin_vel_w``/``yaw_rate`` are computed at the CURRENT speed
            (matching step()'s integrate-then-update-speed order) — write
            them to the body and step PhysX. ``speed`` is the NEW
            post-accel speed: the physics caller must pass it (not the
            PhysX-measured speed) to :meth:`apply_external_state`,
            otherwise commanded acceleration never takes effect.
        """
        steer = float(np.clip(steer, -1.0, 1.0))
        accel_input = float(np.clip(accel, -1.0, 1.0))
        steer_angle = -steer * self.cfg.max_steer_angle
        if accel_input >= 0:
            accel_val = accel_input * self.cfg.max_accel
        else:
            accel_val = accel_input * self.cfg.max_brake

        beta = atan(0.5 * tan(steer_angle))
        new_speed = float(np.clip(self.speed + accel_val * self.cfg.dt,
                                  0.0, self.cfg.max_speed))
        # step() integrates position with the CURRENT speed and updates the
        # speed afterwards — command the current speed so one PhysX step of
        # dt reproduces step()'s pose delta exactly; report the new speed.
        lin_vel = (
            self.speed * cos(self.heading + beta),
            self.speed * sin(self.heading + beta),
            0.0,
        )
        yaw_rate = (2.0 * self.speed / self.cfg.wheelbase) * sin(beta)
        return {"lin_vel_w": lin_vel, "yaw_rate": yaw_rate, "speed": new_speed}

    def apply_external_state(self, x: float, y: float, heading: float,
                             speed: float) -> Dict[str, Any]:
        """Adopt an externally integrated pose (PhysX read-back) as ego state.

        Mirrors the bookkeeping :meth:`step` performs after integrating
        (frame counter + trajectory history) so ``env.frame`` and recorded
        trajectories stay truthful in physics execution mode.
        """
        self.x = float(x)
        self.y = float(y)
        self.heading = float(heading)
        self.speed = float(np.clip(speed, 0.0, self.cfg.max_speed))
        self.frame += 1
        state = self.get_state()
        self.trajectory.append(state)
        return state

    def get_state(self) -> Dict[str, Any]:
        """Return current ego state."""
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "speed": self.speed,
            "speed_kmh": self.speed * 3.6,
            "frame": self.frame,
            "position": np.array([self.x, self.y, 0.0], dtype=np.float32),
            "velocity": np.array([
                self.speed * cos(self.heading),
                self.speed * sin(self.heading),
                0.0,
            ], dtype=np.float32),
        }

    def reset(self, x: float = 0.0, y: float = 0.0,
              heading: float = 0.0, speed: float = 0.0) -> None:
        """Reset ego state."""
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = speed
        self.frame = 0
        self.trajectory = []


class CollisionDetector:
    """Shapely-based polygon collision detection.

    Extracted from ``BaseDrivingEnv.check_collision`` and
    ``BaseDrivingEnv._make_box_polygon``.
    """

    @staticmethod
    def make_box_polygon(x: float, y: float, heading: float,
                         length: float, width: float) -> ShapelyPolygon:
        """Create a rotated rectangle polygon for collision detection."""
        cos_h, sin_h = cos(heading), sin(heading)
        hl, hw = length / 2, width / 2
        corners = np.array([
            [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]
        ])
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated = (R @ corners.T).T + np.array([x, y])
        return ShapelyPolygon(rotated)

    @staticmethod
    def check_collision(
        ego_x: float, ego_y: float, ego_heading: float,
        ego_length: float, ego_width: float,
        agent_states: List[Dict],
    ) -> bool:
        """Check if ego collides with any agent using shapely polygon overlap.

        Args:
            ego_x: Ego x position.
            ego_y: Ego y position.
            ego_heading: Ego heading in radians.
            ego_length: Ego vehicle length.
            ego_width: Ego vehicle width.
            agent_states: List of agent state dicts with position, heading,
                length, width keys.

        Returns:
            True if ego collides with any agent.
        """
        collided, _ = CollisionDetector.check_collision_at_fault(
            ego_x, ego_y, ego_heading, ego_length, ego_width,
            agent_states, ego_speed=1.0,
        )
        return collided

    @staticmethod
    def check_collision_at_fault(
        ego_x: float, ego_y: float, ego_heading: float,
        ego_length: float, ego_width: float,
        agent_states: List[Dict],
        ego_speed: float,
    ) -> Tuple[bool, bool]:
        """Collision check with NavSim-style at-fault classification.

        A collision is *at fault* only when the ego is moving and the
        colliding agent's centre is ahead of or beside the ego. A rear-end by
        a following agent (centre behind the ego) or any contact while the
        ego is stopped is recorded but not at fault — mirrors the at-fault
        filtering in the PDMS scorer and the PDM-Closed planner.

        Args:
            ego_speed: Ego speed in m/s (below the stopped threshold no
                collision is attributed to the ego).

        Returns:
            (collided, at_fault) — ``collided`` is True on any overlap;
            ``at_fault`` is True only for ego-attributable overlaps.
        """
        ego_poly = CollisionDetector.make_box_polygon(
            ego_x, ego_y, ego_heading, ego_length, ego_width,
        )
        forward = np.array([cos(ego_heading), sin(ego_heading)])
        ego_stopped = ego_speed < 0.05
        collided = False
        at_fault = False
        for agent in agent_states:
            # _collect_agent_states_for_renderer keeps the ego's own logged
            # track in the list (the renderer needs it); colliding the ego
            # with its own box would flag every replayed frame.
            if agent.get("is_ego"):
                continue
            agent_poly = CollisionDetector.make_box_polygon(
                agent["position"][0], agent["position"][1],
                agent["heading"],
                agent.get("length", 4.5), agent.get("width", 1.8),
            )
            if not ego_poly.intersects(agent_poly):
                continue
            collided = True
            delta = np.asarray(agent["position"][:2], dtype=np.float64) - np.array(
                [ego_x, ego_y]
            )
            # Strictly ahead (> 0.0): aligned to the shared scorer/planner
            # convention where rear-ends AND pure side contact (agent centre
            # exactly beside the ego) are not the ego's fault.
            if not ego_stopped and float(np.dot(delta, forward)) > 0.0:
                at_fault = True
        return collided, at_fault
