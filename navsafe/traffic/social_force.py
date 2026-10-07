# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Social force for the pedestrians, animals and robots a recipe inserts.

Where IDM answers "how fast do I go down this lane", a social-force model
answers "which way do I lean" — a vulnerable road user has no lane, only a
goal, some obstacles and other people to avoid. The model is Helbing's:
a *desired* force pulling toward the goal, a *social* force pushing away from
others, and an *obstacle* force pushing away from geometry.

Implementation is the vendored PySocialForce (MIT, `third_party/`). Three
adaptations matter and all three are easy to get silently wrong:

* **`step_width` defaults to 1.0 s.** The host runs at 0.1 s. Left alone, every
  pedestrian moves ten times too far per frame and the scenario looks like a
  teleport.
* **Desired speed comes from the *initial* speed** (`max_speeds =
  max_speed_multiplier * initial_speeds`, sampled at construction). Since a
  fresh simulator is built each frame from the actor's *current* velocity,
  the desired speed would drift with whatever the actor happens to be doing,
  so it is pinned from the policy every frame instead.
* **Neighbours must be present but unmoved.** The ego and the logged traffic
  are put in the state array so the actor avoids them, with goals set straight
  ahead so they behave as though nothing is in their way — they are being
  replayed, and a replayed agent does not get to react.

Trigger: a dart-out is not a trajectory but it is not "walks from frame 0"
either. The actor waits at its spawn until the ego is within
``trigger.ego_distance_m``, then goes. That keeps the event type meaningful whatever
speed the policy under test drives at, without scripting a single pose.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from navsafe.traffic.navsafe import register_driver
from navsafe.traffic.stage_free import Driver, WorldView

logger = logging.getLogger(__name__)

#: Neighbours further than this cannot plausibly deflect the actor, and every
#: extra row costs an O(n^2) force evaluation.
NEIGHBOUR_RADIUS_M = 25.0
#: How far ahead a replayed neighbour's goal is placed, so it keeps its heading.
NEIGHBOUR_GOAL_HORIZON_S = 5.0
#: How much a pair's RELATIVE VELOCITY steers the interaction, in the Moussaid
#: force PySocialForce actually runs (`SocialForce`; the Helbing `PedRepulsiveForce`
#: in the same module is never instantiated). Its angular term is
#: exp(-(n_prime * B * theta)^2) with B proportional to |lambda * dv|, so a fast
#: encounter narrows the acceptance cone until nothing registers: at the shipped
#: 2.0, two people closing at 2.8 m/s measured EXACTLY 0.000 force at every
#: separation from 0.5 m to 3.0 m — the model was silently off for the one
#: geometry this event type is about, people crossing a road in opposite directions.
#: 0.5 restores it (0.428 at 1.0 m, 0.173 at 1.5 m) without touching the shape
#: of the force for the head-on case it already handled.
SOCIAL_LAMBDA_IMPORTANCE = 0.5


def _vendored_psf():
    """Import the PySocialForce package bundled with NavSafe."""
    from navsafe._vendor import pysocialforce as psf
    return psf


@register_driver("social_force")
class SocialForceDriver(Driver):
    """Walks toward a goal, avoiding the ego, the traffic and the kerbs."""

    def __init__(self, agent_id: Any, spawn: Dict[str, Any], *,
                 policy: Optional[Dict[str, Any]] = None,
                 desired_speed: float = 1.4, **_ignored) -> None:
        policy = dict(policy or {})
        self.agent_id = agent_id
        self.spawn = dict(spawn)
        self.psf = _vendored_psf()

        goal = policy.get("goal")
        if goal is None or len(goal) < 2:
            raise ValueError(
                f"actor {agent_id!r}: a social-force driver needs `policy.goal` — "
                f"where it is trying to get to. Without a goal there is no desired "
                f"force and the actor only drifts.")
        self.goal = np.asarray(goal, np.float64)[:2]
        self.desired_speed = float(policy.get("desired_speed", desired_speed))
        # PySocialForce's obstacle format is INTERLEAVED (startx, endx, starty,
        # endy), not (x0, y0, x1, y1). Getting this wrong yields walls at right
        # angles to the real ones, which looks like the model misbehaving.
        self.obstacles: List[List[float]] = [
            [float(v) for v in seg] for seg in (policy.get("obstacles") or [])
        ]
        trigger = dict(policy.get("trigger") or {})
        self.trigger_distance_m = float(trigger.get("ego_distance_m", 0.0) or 0.0)
        self.reset(spawn=self.spawn)

    # -- Driver --------------------------------------------------------
    def reset(self, *, spawn: Dict[str, Any]) -> None:
        p = np.asarray(spawn.get("position", (0.0, 0.0, 0.0)), np.float64)
        self._xy = np.array([p[0], p[1]], np.float64)
        self._z = float(p[2]) if len(p) > 2 else 0.0
        self._v = np.zeros(2, np.float64)
        self._heading = float(spawn.get("heading", 0.0))
        self.length = float(spawn.get("length", 0.7))
        self.width = float(spawn.get("width", 0.7))
        self._started = self.trigger_distance_m <= 0.0
        self._arrived = False
        self._frames = 0
        self._path_m = 0.0
        self._reported = False

    def step(self, world: WorldView, dt: float) -> None:
        if not self._started:
            if world.ego_distance(self._xy) > self.trigger_distance_m:
                return
            self._started = True
            logger.info("social_force: %s triggered — ego within %.1f m",
                        self.agent_id, self.trigger_distance_m)
        if self._arrived:
            self._v[:] = 0.0
            return

        rows = [self._row_self()]
        rows.extend(self._neighbour_rows(world))
        sim = self.psf.Simulator(np.asarray(rows, np.float64),
                                 obstacles=self.obstacles or None)
        sim.peds.step_width = float(dt)
        # Pin the desired speed instead of inheriting it from whatever the
        # actor happens to be doing this frame (see the module docstring).
        speeds = np.full(len(rows), self.desired_speed, np.float64)
        sim.peds.initial_speeds = speeds
        # Cap AT the desired speed, not at PySocialForce's 1.3x headroom. The
        # repulsive terms add to the goal-seeking one, so the velocity saturates
        # against whatever the cap is rather than settling at the desired speed:
        # every R-3 pedestrian walked at exactly 1.3x the speed its recipe
        # states (1.30 -> 1.60 m/s measured). A recipe number that the run does
        # not honour is worse than no number.
        sim.peds.max_speeds = speeds
        for force in sim.forces:
            if type(force).__name__ == "SocialForce":
                force.config.from_dict(
                    {"lambda_importance": SOCIAL_LAMBDA_IMPORTANCE})
        sim.step_once()
        state = sim.peds.state[0]

        self._path_m += float(np.linalg.norm(np.array([state[0], state[1]]) - self._xy))
        self._frames += 1
        self._xy = np.array([state[0], state[1]], np.float64)
        self._v = np.array([state[2], state[3]], np.float64)
        if np.linalg.norm(self._v) > 1e-3:
            self._heading = math.atan2(self._v[1], self._v[0])
        if np.linalg.norm(self.goal - self._xy) <= 0.3:
            self._arrived = True
        # The one line that makes a wrong clock obvious. A stage-free manager
        # stepped twice per frame doubles the realised speed without changing
        # anything a still frame would show, and the run still "looks"
        # plausible — it was found by reading this off against the recipe.
        # Reported 3 s in rather than on arrival because an episode routinely
        # ends first: this scenario is cut short by a logged vehicle at frame
        # 101, before any pedestrian finishes crossing.
        if not self._reported and (self._arrived or self._frames == 30):
            self._reported = True
            logger.info("social_force: %s walked %.2f m in %d frames = "
                        "%.2f m/s (policy asks %.2f)%s", self.agent_id,
                        self._path_m, self._frames,
                        self._path_m / max(self._frames * 0.1, 1e-6),
                        self.desired_speed, " [arrived]" if self._arrived else "")

    def pose(self) -> Dict[str, Any]:
        return {
            "position": np.array([self._xy[0], self._xy[1], self._z], np.float32),
            "heading": float(self._heading),
            "velocity": self._v.astype(np.float32),
            "length": self.length,
            "width": self.width,
        }

    # -- state rows ----------------------------------------------------
    def _row_self(self) -> List[float]:
        return [self._xy[0], self._xy[1], self._v[0], self._v[1],
                self.goal[0], self.goal[1]]

    def _neighbour_rows(self, world: WorldView) -> List[List[float]]:
        """The ego and nearby traffic, each aimed straight ahead.

        They are replayed, so they must not be deflected by the actor — giving
        them a goal along their own heading is how a force model expresses
        "this one is not negotiating with you".
        """
        rows: List[List[float]] = []
        candidates = list(world.agents)
        ego = dict(world.ego or {})
        if ego:
            ego.setdefault("id", world.ego_id)
            candidates.append(ego)
        for obj in candidates:
            if obj.get("id") == self.agent_id:
                continue
            p = np.asarray(obj.get("position", (0.0, 0.0, 0.0)), np.float64)[:2]
            if float(np.linalg.norm(p - self._xy)) > NEIGHBOUR_RADIUS_M:
                continue
            v = np.asarray(obj.get("velocity", (0.0, 0.0)), np.float64)[:2]
            g = p + v * NEIGHBOUR_GOAL_HORIZON_S
            rows.append([p[0], p[1], v[0], v[1], g[0], g[1]])
        return rows


__all__ = ["NEIGHBOUR_RADIUS_M", "SocialForceDriver"]
