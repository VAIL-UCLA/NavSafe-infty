# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Bake: route-frame motion + reference -> the four per-frame arrays.

Producing those four arrays is the whole job::

    track["state"] = {
        "position": (T, 3) float32,   # x, y, z
        "heading":  (T,)   float32,   # rad
        "velocity": (T, 2) float32,   # vx, vy, world frame
        "valid":    (T,)   bool,      # does the actor exist this frame
    }

All four are baked, not just position: ``velocity`` because TTC, BEV and the
agent observations read it, and recomputing it from finite differences at
replay would give different numbers; ``valid`` because an actor need not exist
at every frame.

The z convention depends on ``keep_appearance``. Kept-appearance actors are
ground-clamped by the renderer, so their z stays at the reference's own pose
height; swapped-in assets are base-origin and have their base pinned to road z.
Recording which one applies is what keeps the object from floating or sinking
on rebuild.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from navsafe.benchmark.editing.placement.solve import Polyline
from navsafe.benchmark.editing.recipe.schema import bake_state
from navsafe.benchmark.editing.trajectory.templates import Motion
from navsafe.errors import NavSafeError

logger = logging.getLogger(__name__)

# How far a trajectory may run past either end of its reference before the bake
# refuses. Sampling extrapolates along the end tangent, which is a reasonable
# few metres and nonsense beyond that.
DEFAULT_EXTRAPOLATION_TOL_M = 2.0


class BakeError(NavSafeError, ValueError):
    """The motion cannot be baked against this reference."""


@dataclass
class BakedTrack:
    """Baked arrays plus the numbers a reviewer sanity-checks them with."""

    state: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Normalise to (-pi, pi] so a stored heading is comparable across recipes."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def bake_actor_state(
    reference: Polyline,
    motion: Motion,
    *,
    ego_z_to_ground_m: float = 0.0,
    keep_appearance: bool = False,
    strict_reference: bool = True,
    extrapolation_tol_m: float = DEFAULT_EXTRAPOLATION_TOL_M,
) -> BakedTrack:
    """Turn one :class:`Motion` into storable per-frame arrays.

    Args:
        reference: the polyline the motion was authored against.
        motion: route-frame series from a template.
        ego_z_to_ground_m: how far the host's ego pose sits above the road.
        keep_appearance: the actor keeps its baked gaussians (a pure pose
            override), so its z stays at the reference's pose height rather
            than being dropped to road z.
        strict_reference: refuse a trajectory that leaves its reference. Off
            only when the author deliberately runs past the end.
        extrapolation_tol_m: slack allowed at either end before refusing.

    Raises:
        BakeError: the trajectory starts outside the reference, or leaves it by
            more than the tolerance.
    """
    s = np.asarray(motion.s, np.float64)
    lateral = np.asarray(motion.lateral, np.float64)
    if strict_reference:
        if not (0.0 <= s[0] <= reference.total):
            raise BakeError(
                f"spawn arc {s[0]:.1f} m lies outside the reference "
                f"{reference.name or '<polyline>'} (0 .. {reference.total:.1f} m), so the actor "
                f"would start at the end of the route rather than where the recipe says"
            )
        if not reference.covers(s, tol=extrapolation_tol_m):
            raise BakeError(
                f"trajectory runs from {s.min():.1f} to {s.max():.1f} m on a "
                f"{reference.total:.1f} m reference (tolerance {extrapolation_tol_m:.1f} m). "
                f"Shorten the episode's reach — lower |speed| or move the spawn arc — or pass a "
                f"longer reference."
            )

    x, y, _, tangent = reference.offset(s, lateral)
    road_z = reference.road_z(s, ego_z_to_ground_m=ego_z_to_ground_m)
    # Kept appearance -> the renderer ground-clamps the baked gaussians, so the
    # actor keeps the reference's own pose height. Swapped-in asset -> it is
    # base-origin, so its base goes on the road.
    z = road_z + (float(ego_z_to_ground_m) if keep_appearance else 0.0)

    heading = _wrap(tangent + np.asarray(motion.heading_offset, np.float64))
    s_dot = np.asarray(motion.s_dot, np.float64)
    lateral_dot = np.asarray(motion.lateral_dot, np.float64)
    vx = s_dot * np.cos(tangent) - lateral_dot * np.sin(tangent)
    vy = s_dot * np.sin(tangent) + lateral_dot * np.cos(tangent)

    position = np.stack([x, y, z], axis=1)
    velocity = np.stack([vx, vy], axis=1)
    state = bake_state(position, heading, velocity, np.asarray(motion.valid, bool))

    speed = np.hypot(vx, vy)
    diagnostics = {
        "template": motion.template,
        "reference": reference.name or "polyline",
        "reference_length_m": round(reference.total, 2),
        "spawn_arc_m": round(float(s[0]), 2),
        "end_arc_m": round(float(s[-1]), 2),
        "lateral_range_m": [round(float(lateral.min()), 2), round(float(lateral.max()), 2)],
        "speed_range_mps": [round(float(speed.min()), 2), round(float(speed.max()), 2)],
        "start_xy": [round(float(x[0]), 2), round(float(y[0]), 2)],
        "end_xy": [round(float(x[-1]), 2), round(float(y[-1]), 2)],
        "keep_appearance": bool(keep_appearance),
        "valid_frames": int(np.count_nonzero(motion.valid)),
    }
    diagnostics.update(motion.meta)
    logger.debug("bake_actor_state: %s", diagnostics)
    return BakedTrack(state=state, diagnostics=diagnostics)
