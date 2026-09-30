# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The three trajectory templates, in the route frame.

A template turns authored intent into four per-frame series in the reference's
own frame — arc ``s``, lateral offset, and their rates — plus the heading offset
from the reference tangent. Turning those into world coordinates is
:mod:`~navsafe.benchmark.editing.trajectory.bake`'s job; nothing here knows about
a host, a map or a scenario.

Only frame 0 of that survives into a recipe: an inserted actor is REACTIVE, so
what is stored is its spawn plus the controller
:func:`~navsafe.benchmark.editing.author._policy_from_template` derives. There
are three templates because there are three controllers, and the map between
them is one-to-one:

=================== =============================================== ===============================
template            motion                          controller      key parameters
=================== =============================================== ===============================
``static``          a fixed pose                    ``static``      ``arc``, ``lateral``,
                                                                    ``yaw_offset_deg``
``dynamic``         constant speed along the        ``idm``         + ``speed`` (signed)
                    reference, either direction
``dart_out``        holds one arc position and      ``social_force``+ ``start_lateral``,
                    traverses laterally across                      ``end_lateral``, ``speed``,
                    the reference                                   ``conflict_frame``
=================== =============================================== ===============================

There were two more. ``lateral_schedule`` named a lateral offset per FRAME, and
a reactive actor's frame *k* is not knowable in advance — a cut-in has to be a
gap-acceptance policy, not a schedule. ``group`` was N ``dynamic`` actors with a
declared gap pattern, and once the actors became reactive its policy was
byte-identical to ``dynamic``'s while its gap arithmetic described a platoon the
sim would never hold. Neither was reachable from any leaf.

Constant speed is enough throughout: no leaf needs an acceleration or braking
profile. A standing start is already what ``dart_out`` does — the actor holds
its start offset until its trigger — and where the ego's reaction time matters
it is set by ``speed`` and ``arc``, not by how quickly the actor reaches speed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np
from navsafe.errors import NexusSimError
from navsafe.benchmark.editing.recipe.schema import TEMPLATES

# Below this the actor is standing still and its velocity vector carries no
# heading information, so the last meaningful heading is held instead.
_MOVING_EPS = 1e-6


class TemplateError(NexusSimError, ValueError):
    """The authored parameters do not describe a usable trajectory."""


@dataclass
class Motion:
    """One actor's motion in the reference frame, per scenario frame.

    ``heading_offset`` is carried explicitly rather than derived at bake time
    because the templates disagree about it on purpose: a ``dart_out`` holds its
    crossing heading through the still frames at either end, so the animal does
    not rotate on the spot when it starts and stops, while a ``dynamic`` actor
    simply points where its velocity points.
    """

    s: np.ndarray  # (T,) arc length on the reference, absolute
    lateral: np.ndarray  # (T,) +left metres from the centreline
    s_dot: np.ndarray  # (T,) m/s along the reference (signed)
    lateral_dot: np.ndarray  # (T,) m/s across it
    heading_offset: np.ndarray  # (T,) rad from the reference tangent
    valid: np.ndarray  # (T,) bool
    template: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.s.shape[0])


def _zeros(T: int) -> np.ndarray:
    return np.zeros(int(T), np.float64)


def _velocity_heading(s_dot: np.ndarray, lateral_dot: np.ndarray) -> np.ndarray:
    """Heading offset from the velocity vector, holding it while stopped.

    A signed ``s_dot`` is what makes an oncoming actor expressible: ``atan2(0,
    -8)`` is ``pi``, so the heading reverses with the speed and nothing extra is
    needed to turn the actor around.
    """
    moving = (np.abs(s_dot) + np.abs(lateral_dot)) > _MOVING_EPS
    offset = np.where(moving, np.arctan2(lateral_dot, s_dot), np.nan)
    if not np.any(moving):
        return _zeros(len(s_dot))
    # Hold the nearest known heading through stopped stretches, forwards then
    # backwards, so a standing start does not snap the actor to the tangent.
    idx = np.where(moving, np.arange(len(offset)), -1)
    np.maximum.accumulate(idx, out=idx)
    offset = np.where(idx >= 0, offset[np.maximum(idx, 0)], np.nan)
    first = int(np.argmax(moving))
    offset[np.isnan(offset)] = float(np.arctan2(lateral_dot[first], s_dot[first]))
    return offset


def build_motion(
    template: str,
    *,
    T: int,
    dt_s: float,
    anchor_arc: float,
    params: Dict[str, Any],
) -> Motion:
    """Build one actor's route-frame motion from authored parameters.

    Args:
        template: one of :data:`TEMPLATES`.
        T: the host's scenario frame count. Arrays index scenario frames, so
            they cover ``0 .. T-1`` including the pre-hand-off replay stretch.
        dt_s: the host's frame interval, from its own timestamps.
        anchor_arc: arc 0 for the authored ``arc``, i.e. the hand-off projected
            onto the reference.
        params: the template's parameters.

    Raises:
        TemplateError: unknown template, or parameters that do not describe a
            usable trajectory.
    """
    T = int(T)
    if T < 2:
        raise TemplateError(f"T must be >= 2, got {T}")
    if dt_s <= 0:
        raise TemplateError(f"dt_s must be positive, got {dt_s}")
    if template not in TEMPLATES:
        raise TemplateError(f"unknown template {template!r}; expected one of {list(TEMPLATES)}")
    arc = float(params.get("arc", 0.0))
    lateral0 = float(params.get("lateral", 0.0))
    yaw_offset = np.deg2rad(float(params.get("yaw_offset_deg", 0.0)))
    start_s = anchor_arc + arc
    k = np.arange(T, dtype=np.float64)

    if template == "static":
        motion = Motion(
            s=np.full(T, start_s),
            lateral=np.full(T, lateral0),
            s_dot=_zeros(T),
            lateral_dot=_zeros(T),
            heading_offset=np.full(T, yaw_offset),
            valid=np.ones(T, bool),
            template=template,
        )
        return motion

    if template == "dynamic":
        speed = float(params.get("speed", 0.0))
        s = start_s + speed * dt_s * k
        s_dot = np.full(T, speed)
        lateral = np.full(T, lateral0)
        lateral_dot = _zeros(T)
        heading = _velocity_heading(s_dot, lateral_dot) + yaw_offset
        if abs(speed) <= _MOVING_EPS:
            heading = np.full(T, yaw_offset)
        return Motion(
            s=s,
            lateral=lateral,
            s_dot=s_dot,
            lateral_dot=lateral_dot,
            heading_offset=heading,
            valid=np.ones(T, bool),
            template=template,
            meta={"speed": speed},
        )

    # dart_out — holds one arc position and moves purely across the reference.
    start_lateral = float(params["start_lateral"]) if "start_lateral" in params else None
    end_lateral = float(params["end_lateral"]) if "end_lateral" in params else None
    if start_lateral is None or end_lateral is None:
        raise TemplateError("dart_out needs both `start_lateral` and `end_lateral`")
    speed = float(params.get("speed", 0.0))
    if speed <= 0:
        raise TemplateError(
            "dart_out `speed` is a crossing speed and must be positive; the direction "
            "comes from start_lateral -> end_lateral"
        )
    span = end_lateral - start_lateral
    if abs(span) < 1e-6:
        raise TemplateError("dart_out start_lateral and end_lateral are the same — nothing crosses")
    conflict_frame = int(params.get("conflict_frame", T // 2))
    # duration = |end - start| / (speed * dt); the crossing is centred on the
    # conflict frame, so conflict_frame IS the frame the actor reaches the
    # centre — which is only true while the sweep stays symmetric about it.
    duration = int(round(abs(span) / (speed * dt_s)))
    if duration < 1:
        raise TemplateError(
            f"dart_out crossing lasts {duration} frames at {speed} m/s over {abs(span):.2f} m; "
            f"lower the speed or widen the sweep"
        )
    crossing_start = conflict_frame - duration // 2
    alpha = np.clip((k - crossing_start) / duration, 0.0, 1.0)
    lateral = start_lateral + alpha * span
    lateral_dot = _zeros(T)
    inside = (k >= crossing_start) & (k < crossing_start + duration)
    lateral_dot[inside] = span / (duration * dt_s)
    # Hold the crossing heading through the still frames at either end, so the
    # actor does not rotate on the spot when it starts and stops.
    crossing_heading = float(np.arctan2(np.sign(span) * speed, 0.0))
    return Motion(
        s=np.full(T, start_s),
        lateral=lateral,
        s_dot=_zeros(T),
        lateral_dot=lateral_dot,
        heading_offset=np.full(T, crossing_heading + yaw_offset),
        valid=np.ones(T, bool),
        template=template,
        meta={
            "speed": speed,
            "duration_frames": duration,
            "crossing_start_frame": int(crossing_start),
            "conflict_frame": conflict_frame,
        },
    )
