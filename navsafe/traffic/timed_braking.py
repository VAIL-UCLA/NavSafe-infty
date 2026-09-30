"""Time-controlled lane motion for paired hazard experiments.

Unlike reactive traffic, a timed actor publishes the same state at a given
absolute simulation time for every ego policy. The profile includes recovery
so braking does not permanently obstruct event completion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from navsafe.traffic.geometry import Route
from navsafe.traffic.navsafe import register_driver
from navsafe.traffic.stage_free import Driver, WorldView


@dataclass(frozen=True)
class BrakeProfile:
    speed: float
    onset_s: float = 2.0
    deceleration: float = 3.0
    speed_reduction: float = 6.0
    hold_s: float = 1.0
    recovery_acceleration: float = 2.0
    recovery_speed: float | None = None
    enabled: bool = True

    def __post_init__(self):
        values = (self.speed, self.onset_s, self.deceleration,
                  self.speed_reduction, self.hold_s, self.recovery_acceleration,
                  *(() if self.recovery_speed is None else (self.recovery_speed,)))
        if not all(math.isfinite(x) for x in values):
            raise ValueError("profile parameters must be finite")
        if min(self.speed, self.onset_s, self.speed_reduction, self.hold_s) < 0:
            raise ValueError("speed, time and speed reduction must be nonnegative")
        if min(self.deceleration, self.recovery_acceleration) <= 0:
            raise ValueError("braking and recovery magnitudes must be positive")
        low = self.speed - min(self.speed_reduction, self.speed)
        if self.recovery_speed is not None and self.recovery_speed < low:
            raise ValueError("recovery speed must not be below low speed")

    def sample(self, time_s: float) -> tuple[float, float, float]:
        """Absolute travel, speed, acceleration; piecewise exact integration."""
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("time must be finite and nonnegative")
        if not self.enabled or time_s < self.onset_s:
            return self.speed * time_s, self.speed, 0.0
        delta = min(self.speed_reduction, self.speed)
        low = self.speed - delta
        target = self.speed if self.recovery_speed is None else self.recovery_speed
        brake_s = delta / self.deceleration
        recover_s = (target - low) / self.recovery_acceleration
        elapsed = time_s - self.onset_s
        distance = self.speed * self.onset_s
        if elapsed < brake_s:
            return (distance + self.speed * elapsed - self.deceleration * elapsed**2 / 2,
                    self.speed - self.deceleration * elapsed, -self.deceleration)
        distance += (self.speed + low) * brake_s / 2
        elapsed -= brake_s
        if elapsed < self.hold_s:
            return distance + low * elapsed, low, 0.0
        distance += low * self.hold_s
        elapsed -= self.hold_s
        if elapsed < recover_s:
            return (distance + low * elapsed + self.recovery_acceleration * elapsed**2 / 2,
                    low + self.recovery_acceleration * elapsed, self.recovery_acceleration)
        distance += (low + target) * recover_s / 2
        elapsed -= recover_s
        return distance + target * elapsed, target, 0.0


@register_driver("timed_braking")
class TimedBrakingDriver(Driver):
    """Sample a frozen speed profile on a resolved lane at world.t * dt."""

    strict = True

    def __init__(self, agent_id, spawn, *, policy=None, **_params):
        self.agent_id = agent_id
        self.spawn = dict(spawn)
        policy = dict(policy or {})
        self.route = Route(policy["path_polyline"])
        raw = np.asarray(policy["path_polyline"], dtype=float)
        self._height_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw[:,:2], axis=0), axis=1))]
        self._height_z = raw[:,2] if raw.shape[1]>2 else None
        self.profile = BrakeProfile(**policy["profile"])
        self.reset(spawn=self.spawn)

    def reset(self, *, spawn):
        position = np.asarray(spawn["position"], dtype=float)
        self.start_s, lateral = self.route.local_coordinates(position[:2])
        if abs(lateral) > 0.05:
            raise ValueError("timed actor spawn must lie on its frozen route")
        self.z = float(position[2]) if len(position) > 2 else 0.0
        self.length = float(spawn.get("length", 4.5))
        self.width = float(spawn.get("width", 1.8))
        self.time_s = 0.0
        self.s = self.start_s
        self.speed = self.profile.speed
        self.acceleration = 0.0

    def step(self, world: WorldView, dt: float):
        # NexusSim increments scenario_timestep before stepping traffic.
        self.time_s = float(world.t) * float(dt)
        distance, self.speed, self.acceleration = self.profile.sample(self.time_s)
        self.s = self.start_s + distance
        if self.s > self.route.length + 1e-6:
            raise ValueError("timed actor exhausted its qualified route")

    def pose(self):
        xy = self.route.position_at(self.s)
        z = self.z if self._height_z is None else float(np.interp(self.s, self._height_s, self._height_z))
        heading = self.route.heading_at(self.s)
        return {
            "position": np.array([xy[0], xy[1], z], dtype=float),
            "heading": heading,
            "velocity": self.speed * np.array([math.cos(heading), math.sin(heading)]),
            "length": self.length, "width": self.width,
        }
