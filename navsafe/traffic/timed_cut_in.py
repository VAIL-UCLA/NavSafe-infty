"""Deterministic cut-in and lane-return motion for paired event experiments."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from navsafe.traffic.navsafe import register_driver
from navsafe.traffic.timed_braking import TimedBrakingDriver


@dataclass(frozen=True)
class CutInProfile:
    lateral_m: float
    onset_s: float = 2.0
    transition_s: float = 2.0
    hold_s: float = 2.0
    recovery_s: float = 2.0
    enabled: bool = True

    def __post_init__(self):
        values = (self.lateral_m, self.onset_s, self.transition_s,
                  self.hold_s, self.recovery_s)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("cut-in parameters must be finite")
        if min(self.onset_s, self.hold_s) < 0:
            raise ValueError("onset and hold must be nonnegative")
        if min(self.transition_s, self.recovery_s) <= 0:
            raise ValueError("transition and recovery must be positive")

    def sample(self, time_s):
        """Lateral offset and velocity; positive points left of source lane."""
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("time must be finite and nonnegative")
        elapsed = time_s - self.onset_s
        if not self.enabled or elapsed <= 0:
            return 0.0, 0.0
        def smooth(t, duration):
            u = min(1.0, max(0.0, t / duration))
            return (10*u**3 - 15*u**4 + 6*u**5,
                    30*u**2 * (1-u)**2 / duration)
        if elapsed < self.transition_s:
            q, dq = smooth(elapsed, self.transition_s)
            return self.lateral_m*q, self.lateral_m*dq
        elapsed -= self.transition_s
        if elapsed < self.hold_s:
            return self.lateral_m, 0.0
        elapsed -= self.hold_s
        q, dq = smooth(elapsed, self.recovery_s)
        return self.lateral_m*(1-q), -self.lateral_m*dq


@register_driver("timed_cut_in")
class TimedCutInDriver(TimedBrakingDriver):
    """Follow the frozen source lane, enter the ego lane, then leave it.

    The source route must have sufficient paved lateral clearance throughout
    the maneuver. Recipe preparation must verify swept actor footprints.
    """

    def __init__(self, agent_id, spawn, *, policy=None, **params):
        policy = dict(policy or {})
        self.cut_in = CutInProfile(**policy["cut_in"])
        super().__init__(agent_id, spawn, policy=policy, **params)

    def _xy_at(self, time_s):
        distance, _, _ = self.profile.sample(time_s)
        s = self.start_s + distance
        if s > self.route.length + 1e-6:
            raise ValueError("cut-in actor exhausted its qualified route")
        heading = self.route.heading_at(s)
        lateral, _ = self.cut_in.sample(time_s)
        return self.route.position_at(s) + lateral*np.array(
            [-math.sin(heading), math.cos(heading)])

    def pose(self):
        result = super().pose()
        xy = self._xy_at(self.time_s)
        # Differentiate the actual offset curve, including lane curvature.
        lo = max(0.0, self.time_s - 1e-3)
        hi = self.time_s + 1e-3
        if self.start_s + self.profile.sample(hi)[0] > self.route.length:
            hi = self.time_s
        velocity = ((self._xy_at(hi)-self._xy_at(lo))/(hi-lo)
                    if hi > lo else np.zeros(2))
        result["position"][:2] = xy
        result["velocity"] = velocity
        if np.linalg.norm(velocity) > 1e-6:
            result["heading"] = math.atan2(velocity[1], velocity[0])
        return result
