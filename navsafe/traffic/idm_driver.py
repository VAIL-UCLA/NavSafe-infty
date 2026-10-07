# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""IDM for an actor a NavSafe recipe inserted.

IDM is a purely **longitudinal** car-following model: given how fast I am, how
fast the thing in front of me is, and how big the gap is, it returns an
acceleration. Three consequences shape everything below.

* **It does not decide where to go.** The path is an input. Lateral position
  is whatever the path says, so a recipe hands the driver the lane chain it
  resolved at bake time.
* **It only sees what is on its own path.** An object off the corridor is
  invisible to it. For C-10 that is not a limitation but the point: a car in
  the opposing carriageway does not expect anyone to be coming the wrong way
  down it, and modelling it as if it did would delete the event type's whole event.
* **It never crashes.** IDM is a *safe* following model — it will brake rather
  than hit the thing ahead. So an actor whose job is to supply a collision
  cannot simply be handed the ego as a leader and left to it; see
  ``blind_to_ego``.

The kernel is :class:`navsafe.component.traffic_agent.idm.IDMActor`, the same
one the background-traffic manager uses. What is new here is the takeover rule
(by name, from the recipe) and that blindness switch.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navsafe.component.traffic_agent.idm import IDMActor, IDMParams
from navsafe.traffic.geometry import ROUTE_WIDTH_M, Route, corners
from navsafe.traffic.navsafe import register_driver
from navsafe.traffic.stage_free import Driver, WorldView

logger = logging.getLogger(__name__)

#: How far ahead to look for a leader. Beyond this IDM's gap term is
#: negligible anyway, and the scan is O(agents) per frame.
LOOKAHEAD_M = 40.0

#: Lateral half-band, beyond the route corridor, in which the EGO alone is
#: still treated as something to brake for.
#:
#: The corridor test asks "is it in my lane", which is the right question for
#: logged traffic and the wrong one for a closed-loop ego: the ego is by
#: definition somewhere the log never put it, so it can be converging on this
#: actor while sitting outside a 2 m lane. Measured on
#: 05d0a1a763fc5334/drivor: the ego was 2.72 m lateral and 5.97 m ahead of a
#: replayed vehicle, i.e. ~1.8 m outside the +/-1.0 m corridor, so `_leader`
#: returned None, IDM accelerated freely, and it rear-ended the ego at
#: 7.6 m/s. The episode ended `contact_not_at_fault` -- a collision the ego
#: could not have avoided and the actor never braked for.
#:
#: 3.0 m is half an actor width plus half an ego width plus ~1.2 m of margin:
#: wide enough to cover a neighbouring-lane conflict, narrow enough that a car
#: two lanes over is still ignored. Widening the corridor itself would do this
#: too, but it would also make actors brake for ordinary overtaking traffic in
#: the next lane, which is a behaviour change in every scenario rather than a
#: fix for this one.
EGO_CONFLICT_LAT_M = 3.0


@register_driver("idm")
class IDMDriver(Driver):
    """Follows a path at a desired speed, yielding to whatever is in front.

    Args:
        blind_to_ego: drop the ego from the leader search. The actor still
            follows its path and still yields to ordinary traffic — it simply
            does not treat the ego as an obstacle. This is what makes C-10 and
            C-7 testable: the wrong-way ego meets a car that is driving
            normally and is not expecting it, so crossing the centreline has
            a consequence. Without it IDM brakes, the collision never happens,
            and the event type silently stops measuring what it is named after.
    """

    def __init__(self, agent_id: Any, spawn: Dict[str, Any], *,
                 policy: Optional[Dict[str, Any]] = None,
                 v0: float = 8.0, s0: float = 2.0, T: float = 1.5,
                 a: float = 1.0, b: float = 1.5, delta: float = 4.0,
                 blind_to_ego: bool = False,
                 route_width_m: float = ROUTE_WIDTH_M,
                 **_ignored) -> None:
        policy = dict(policy or {})
        self.agent_id = agent_id
        self.spawn = dict(spawn)
        self.blind_to_ego = bool(policy.get("blind_to_ego", blind_to_ego))
        self.params = IDMParams(v0=float(v0), s0=float(s0), T=float(T),
                                a=float(a), b=float(b), delta=float(delta))
        self.actor = IDMActor(self.params)

        polyline = policy.get("path_polyline")
        if polyline is None or len(polyline) < 2:
            raise ValueError(
                f"actor {agent_id!r}: an IDM driver needs `policy.path_polyline` "
                f"— the lane the recipe resolved at bake time. Storing the "
                f"resolved polyline (not just the name 'opposing_lane_chain') is "
                f"what makes the same recipe rebuild the same path later."
            )
        self.route = Route(np.asarray(polyline, dtype=np.float64), width=route_width_m)
        # Hold at the spawn until the ego is this close, then drive. Without
        # it every inserted vehicle sets off at frame 0 and has cleared the
        # area before the ego arrives — the scenario runs, renders, and asks
        # the policy for nothing. Staggering the distance across a group is
        # what turns "five bicycles somewhere on the road" into "five things
        # that demand a reaction, one after another".
        trigger = dict(policy.get("trigger") or {})
        self.trigger_distance_m = float(trigger.get("ego_distance_m", 0.0) or 0.0)
        self.reset(spawn=self.spawn)

    # -- Driver --------------------------------------------------------
    def reset(self, *, spawn: Dict[str, Any]) -> None:
        p = np.asarray(spawn.get("position", (0.0, 0.0, 0.0)), np.float64)[:2]
        self.s, _lat = self.route.local_coordinates(p)
        vel = np.asarray(spawn.get("velocity", (0.0, 0.0)), np.float64)
        # Spawn speed defaults to the desired speed: an actor that has been
        # driving down this road for a while is already at cruise, and starting
        # every one from rest turns "a car is coming" into "a car pulls away".
        self.v = float(np.linalg.norm(vel)) if vel.any() else float(self.params.v0)
        self.length = float(spawn.get("length", 4.5))
        self.width = float(spawn.get("width", 1.8))
        self._accel = 0.0
        self._done = False
        self._started = self.trigger_distance_m <= 0.0

    def step(self, world: WorldView, dt: float) -> None:
        if not self._started:
            here = self.route.position_at(self.s)
            if world.ego_distance(here) > self.trigger_distance_m:
                self.v = 0.0
                return
            self._started = True
            logger.info("idm: %s triggered — ego within %.1f m",
                        self.agent_id, self.trigger_distance_m)
        if self._done:
            self.v = 0.0
            return
        lead, gap = self._leader(world)
        if lead is None:
            accel = self.actor.compute_acceleration(self.v, 0.0, float("inf"))
        else:
            accel = self.actor.compute_acceleration(self.v, self._closing(lead), gap)
        self._accel = float(accel)
        self.v = max(0.0, self.v + self._accel * dt)
        self.s += self.v * dt
        if self.s >= self.route.length:
            # Ran out of road. Freeze at the end rather than extrapolating off
            # the map — an actor teleporting past the reconstruction's edge is
            # a rendering artefact, not a scenario.
            self.s = self.route.length
            self._done = True

    def pose(self) -> Dict[str, Any]:
        xy = self.route.position_at(self.s)
        heading = self.route.heading_at(self.s)
        return {
            "position": np.array([xy[0], xy[1], 0.0], dtype=np.float32),
            "heading": float(heading),
            "velocity": np.array([self.v * math.cos(heading),
                                  self.v * math.sin(heading)], dtype=np.float32),
            "length": self.length,
            "width": self.width,
        }

    # -- perception ----------------------------------------------------
    def _candidates(self, world: WorldView) -> List[Dict[str, Any]]:
        rows = [r for r in world.agents if r.get("id") != self.agent_id]
        if self.blind_to_ego:
            # Not "cannot see it" — "is not expecting it". The ego is simply
            # absent from this driver's world model.
            return [r for r in rows if r.get("id") != world.ego_id]
        ego = dict(world.ego or {})
        if not ego:
            return rows
        ego.setdefault("id", world.ego_id)
        ego.setdefault("length", 4.5)
        ego.setdefault("width", 1.8)
        return rows + [ego]

    def _leader(self, world: WorldView) -> Tuple[Optional[Dict[str, Any]], float]:
        """Nearest object ahead **inside my corridor**, centre to centre.

        The corridor test is what makes IDM's blindness principled rather than
        accidental: an object is a leader because it is in my lane, not
        because it happens to be nearby.
        """
        here = self.route.position_at(self.s)
        best, best_gap = None, LOOKAHEAD_M
        # The ego when it is near-but-not-in the corridor, kept separately so a
        # real in-lane leader always wins.
        conflict, conflict_gap = None, LOOKAHEAD_M
        for obj in self._candidates(world):
            p = np.asarray(obj.get("position", (0.0, 0.0, 0.0)), np.float64)[:2]
            if float(np.hypot(*(p - here))) > LOOKAHEAD_M:
                continue
            box = corners(float(p[0]), float(p[1]), float(obj.get("heading", 0.0)),
                          float(obj.get("length", 4.5)), float(obj.get("width", 1.8)))
            long, lat = self.route.local_coordinates(p)
            gap = long - self.s
            if not any(self.route.point_on_lane(c) for c in box):
                # Outside my lane. Only the ego earns a second look, and only
                # ahead of me: `blind_to_ego` actors stay blind, because C-10
                # and C-7 depend on the actor NOT expecting the ego.
                if (not self.blind_to_ego
                        and str(obj.get("id")) == str(world.ego_id)
                        and abs(lat) <= EGO_CONFLICT_LAT_M
                        and 0.0 < gap < conflict_gap):
                    conflict, conflict_gap = obj, gap
                continue
            if 0.0 < gap < best_gap:
                best, best_gap = obj, gap
        if best is None and conflict is not None and self._closing(conflict) > 0.0:
            # Closing only: braking for an ego that is pulling away would stall
            # the actor for no reason, and a stalled actor is its own artefact.
            logger.debug("idm: braking for the ego %.1f m ahead, %.1f m lateral "
                         "(outside my %.1f m corridor)", conflict_gap,
                         float(self.route.local_coordinates(
                             np.asarray(conflict["position"], np.float64)[:2])[1]),
                         self.route.width)
            return conflict, conflict_gap
        return best, best_gap

    def _closing(self, lead: Dict[str, Any]) -> float:
        """Closing rate along my heading (IDM's ``delta_v``)."""
        heading = self.route.heading_at(self.s)
        v_lead = np.asarray(lead.get("velocity", (0.0, 0.0)), np.float64)[:2]
        mine = np.array([self.v * math.cos(heading), self.v * math.sin(heading)])
        rel = mine - v_lead
        return float(rel[0] * math.cos(heading) + rel[1] * math.sin(heading))


__all__ = ["IDMDriver", "LOOKAHEAD_M"]
