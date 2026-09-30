# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared per-timestep ego speed for the EPDMS at-fault classifiers.

Upstream nuPlan judges an at-fault collision by the ego's speed AT COLLISION
TIME. Neither speed the scorers have at hand can decide that directly:

* ``states['speed']`` -- finite differences of the savgol-filtered candidate
  path -- has a ~0.31 m/s floor for a genuinely stationary ego (measured),
  so the 0.05 m/s stopped threshold never fires and a parked ego is blamed
  for collisions it suffers (the 11-consecutive-frames artifact);
* the ego's REAL frame-0 speed (``ego_state['speed']``) describes t=0 only:
  frozen across the horizon it denies the exemption to a candidate that
  brakes to a stop mid-horizon (still "moving" at collision time) and, with
  a displacement guard, grants it to a creep that reaches a lead within the
  guard distance (still "stopped" while driving into the car ahead).

:func:`at_fault_ego_speed` derives the speed at scored pose ``t`` from the
candidate's own positions and returns the smaller of two estimates:

* segment speed ``|p[t] - p[t-1]| / dt`` -- current motion, so a candidate
  braked to a stop reads ~0 by collision time;
* net-displacement speed ``|p[t] - p[0]| / (t * dt)`` -- robust to the
  savgol floor: a stationary candidate's filtered path oscillates but goes
  nowhere, so this decays below the threshold, while a real creep keeps it
  above however slowly it rolls.

The min is below the stopped threshold iff the candidate is either stopped
at ``t`` or never really went anywhere -- exactly the poses that must not be
charged. ``t == 0`` returns ``ego_v0`` (the real frame-0 speed) when the
caller has one; t=0 contacts are normally consumed by the scorers'
pre-existing-contact exclusion anyway.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def at_fault_ego_speed(xs: Sequence[float], ys: Sequence[float], t: int,
                       dt: float, ego_v0: Optional[float] = None) -> float:
    """Ego speed at scored pose ``t`` for the stopped-ego at-fault exemption.

    Args:
        xs, ys: the candidate's scored positions (savgol-filtered where the
            caller filters -- use the same arrays the collision polygons are
            built from).
        t: index of the scored pose being classified.
        dt: spacing of the scored poses in seconds (the scorers' planner_dt).
        ego_v0: the ego's REAL frame-0 speed, used at ``t == 0`` where no
            backward difference exists. Falls back to the first forward
            segment when unavailable.
    """
    if t <= 0:
        if ego_v0 is not None:
            return abs(float(ego_v0))
        if len(xs) > 1:
            return math.hypot(float(xs[1]) - float(xs[0]),
                              float(ys[1]) - float(ys[0])) / dt
        return 0.0
    seg_v = math.hypot(float(xs[t]) - float(xs[t - 1]),
                       float(ys[t]) - float(ys[t - 1])) / dt
    net_v = math.hypot(float(xs[t]) - float(xs[0]),
                       float(ys[t]) - float(ys[0])) / (t * dt)
    return min(seg_v, net_v)
