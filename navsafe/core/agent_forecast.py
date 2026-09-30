# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Constant-velocity forecast conventions shared by planner and scorer.

Lives in ``core`` (no IsaacSim, no torch) because both
:mod:`navsafe.policy.state.pdm_closed_planner.forward_sim` and
:mod:`navsafe.evaluation.scorers.epdms_trajectory_scorer_fast` forecast
other agents, and the scorer cannot import the planner package without a
cycle. The two forecasts must agree: the planner picks IDM leads and the
scorer grades NC/TTC against the same predicted world.
"""

from __future__ import annotations

import math
from typing import Tuple

#: Types the reference treats as AGENTS (``TrackedObjectType`` VEHICLE,
#: PEDESTRIAN, BICYCLE): they are forecast at constant velocity and may be
#: classified as moving in the at-fault test. Every other tracked type is a
#: STATIC object in ``PDMObjectManager``: never moved, always "stopped".
#: ``CYCLIST`` is py123d's name for nuPlan's BICYCLE.
MOVING_AGENT_TYPES = frozenset({"VEHICLE", "PEDESTRIAN", "BICYCLE", "CYCLIST"})


def is_moving_agent_type(actor_type: object) -> bool:
    """Whether the reference would forecast/classify this type as an agent."""
    return str(actor_type).upper() in MOVING_AGENT_TYPES


#: ``PDMObjectManager.MAX_DYNAMIC_OBJECTS`` / ``MAX_STATIC_OBJECTS``: after the
#: 50 m admission the reference keeps only the nearest k objects per class,
#: ranked by centre distance to the ego centre at the current frame.
NEAREST_K_CAPS = {"VEHICLE": 50, "PEDESTRIAN": 25, "BICYCLE": 10, "CYCLIST": 10}
NEAREST_K_STATIC_CAP = 50


def nearest_k_admitted(
    actors: "list[tuple[str, str, float]]",
) -> "set[str]":
    """Ids surviving the reference's per-class nearest-k caps.

    ``actors`` holds ``(actor_id, actor_type, distance_to_ego_centre)``.
    Every non-agent type shares the single static-object cap.
    """
    by_class: dict[str, list[tuple[float, str]]] = {}
    for actor_id, actor_type, distance in actors:
        kind = str(actor_type).upper()
        key = kind if kind in NEAREST_K_CAPS else "__static__"
        by_class.setdefault(key, []).append((float(distance), str(actor_id)))
    keep: set[str] = set()
    for key, items in by_class.items():
        cap = NEAREST_K_CAPS.get(key, NEAREST_K_STATIC_CAP)
        # argsort is stable in the reference (numpy default quicksort is
        # not, but ties at equal distance are measure-zero); sort by
        # distance then id for determinism.
        items.sort()
        keep.update(actor_id for _d, actor_id in items[:cap])
    return keep


def forecast_velocity(
    vx: float, vy: float, heading: float, actor_type: object,
) -> Tuple[float, float]:
    """Constant-velocity forecast the reference applies to ``actor_type``.

    Agents: :func:`heading_aligned_velocity`. Static objects (cones,
    barriers, generic objects, ...): zero — ``PDMObjectManager`` keeps their
    box where it is for the whole horizon and ``_get_leading_agent_velocity``
    reports 0 for them, whatever their measured velocity says (logged static
    objects jitter by a few cm/s per frame).
    """
    if not is_moving_agent_type(actor_type):
        return 0.0, 0.0
    return heading_aligned_velocity(vx, vy, heading)


def heading_aligned_velocity(
    vx: float, vy: float, heading: float,
) -> Tuple[float, float]:
    """The velocity CaRL's PDM-Closed forecasts an agent with.

    ``PDMObjectManager.add_object`` (``carl_nuplan/.../pdm_object_manager.py``)
    keeps only the velocity's magnitude and applies it along the box heading,
    flipped by π when the vector points backwards (a reversing vehicle):

        agent_drives_forward = |normalize(heading - atan2(vy, vx))| < π/2
        dxy = |v| * [cos, sin](heading or heading + π)

    Every consumer of the constant-velocity forecast — the lead search and
    the NC/TTC scoring — sees this velocity, not the raw vector. Logged
    vehicles are heading-aligned on 98-100 % of moving frames (NavSafe
    bundles, measured), so the difference is confined to pedestrians and
    the odd sliding box; it is kept for parity, not for effect.
    """
    speed = math.hypot(float(vx), float(vy))
    if speed <= 0.0:
        return 0.0, 0.0
    rel = (math.atan2(float(vy), float(vx)) - float(heading) + math.pi) % (
        2.0 * math.pi) - math.pi
    track_heading = (
        float(heading) if abs(rel) < 0.5 * math.pi else float(heading) + math.pi
    )
    return speed * math.cos(track_heading), speed * math.sin(track_heading)


__all__ = [
    "MOVING_AGENT_TYPES", "NEAREST_K_CAPS", "NEAREST_K_STATIC_CAP",
    "forecast_velocity", "heading_aligned_velocity", "is_moving_agent_type",
    "nearest_k_admitted",
]
