# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Can the ego actually see it, and in time?

A constructed scenario is worthless if the actor the event type is *about* spends the
approach hidden behind another vehicle and only emerges a moment before
contact. Nothing could react to that: not the policy, and not a human driver.
It scores as a failure while measuring nothing.

The characteristic shape is a lead vehicle: the ego follows a car, the inserted
actor closes head-on in the same lane, and the lead car eclipses it until the
two are metres apart — at which point the actor appears to "pop through" the
lead car into the gap. Geometrically the trajectory is fine; as a test it is
meaningless.

So this module answers two questions from the baked geometry alone, before any
render:

* **when does the actor first become visible** to the ego, and does it *stay*
  visible (a flicker through a gap is not visibility); and
* **how long is it visible before the closest approach** — the reaction window.

The occlusion model is deliberately coarse and 2-D: an actor is occluded when
another agent's box crosses the ego->actor sight line *and* sits closer to the
ego. It ignores camera FOV, elevation and partial occlusion, so it is a
screening tool, not a substitute for the reviewer's eyes. It is tuned to catch
the one failure that matters — a fully eclipsed actor — and to stay quiet
otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: A policy needs at least this long between the actor becoming visible and the
#: moment of closest approach. Below it the episode measures reaction time the
#: scenario never granted. The event type checklists ask for ">= 1.5 s to initiate
#: evasion"; this is that number.
DEFAULT_MIN_REACTION_S = 1.5

#: Visibility shorter than this is a flicker through a gap between vehicles, not
#: a sighting, and is not counted as the actor "appearing".
_MIN_SUSTAINED_S = 0.3


@dataclass
class VisibilityReport:
    """Per-frame occlusion of one actor, plus the numbers a reviewer needs."""

    visible: np.ndarray  # (T,) bool
    blockers: List[Optional[str]] = field(default_factory=list)
    first_visible_frame: Optional[int] = None
    closest_frame: int = 0
    closest_gap_m: float = float("inf")
    reaction_s: float = 0.0
    dt_s: float = 0.1
    occluded_fraction: float = 0.0

    @property
    def ok(self) -> bool:
        return self.reaction_s >= DEFAULT_MIN_REACTION_S

    def describe(self) -> str:
        first = "never" if self.first_visible_frame is None else f"frame {self.first_visible_frame}"
        return (
            f"visible from {first}; closest approach {self.closest_gap_m:.1f} m at frame "
            f"{self.closest_frame} (t={self.closest_frame * self.dt_s:.1f}s); "
            f"reaction window {self.reaction_s:.2f}s; "
            f"occluded {100 * self.occluded_fraction:.0f}% of the approach"
        )


def _boxes(track: dict, frame: int):
    """(centre_xy, half_length, half_width, heading) of a track at a frame."""
    state = track.get("state", {})
    pos = np.asarray(state.get("position"))
    if pos.ndim != 2 or frame >= pos.shape[0]:
        return None
    valid = np.asarray(state.get("valid", np.ones(pos.shape[0], bool)))
    if not bool(valid[frame]):
        return None
    def _at(key, default):
        arr = np.asarray(state.get(key, [default])).reshape(-1)
        return float(arr[min(frame, len(arr) - 1)]) or default
    return (
        pos[frame, :2].astype(np.float64),
        _at("length", 4.5) / 2.0,
        _at("width", 1.9) / 2.0,
        float(np.asarray(state.get("heading", np.zeros(pos.shape[0]))).reshape(-1)[frame]),
    )


def _segment_hits_box(a: np.ndarray, b: np.ndarray, centre, half_l, half_w, heading) -> bool:
    """Does segment a->b cross an oriented box? (2-D, in the box's own frame.)"""
    cos_t, sin_t = np.cos(-heading), np.sin(-heading)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    p, q = rot @ (a - centre), rot @ (b - centre)
    # Liang-Barsky against the axis-aligned box in the rotated frame.
    d = q - p
    t0, t1 = 0.0, 1.0
    for axis, half in ((0, half_l), (1, half_w)):
        if abs(d[axis]) < 1e-12:
            if abs(p[axis]) > half:
                return False
            continue
        for sign in (-1.0, 1.0):
            num = sign * half - p[axis]
            t = num / d[axis]
            if d[axis] * sign > 0:
                t1 = min(t1, t)
            else:
                t0 = max(t0, t)
        if t0 > t1:
            return False
    return True


def occlusion_report(
    sd: dict,
    actor_position: np.ndarray,
    *,
    dt_s: float,
    ignore_track_ids: Optional[List[str]] = None,
    actor_id: Optional[str] = None,
) -> VisibilityReport:
    """When can the ego see ``actor_position``, and for how long before contact?

    Args:
        sd: the host scenario (its ego and other agents supply the occluders).
        actor_position: the actor's baked ``(T, >=2)`` positions.
        dt_s: frame interval.
        ignore_track_ids: tracks that must not count as occluders — at minimum
            the actor's own source track, which a ``relocate`` is about to
            overwrite or delete.
        actor_id: the actor's own track id, ignored as an occluder.

    Returns:
        A :class:`VisibilityReport`.
    """
    meta = sd.get("metadata", {}) or {}
    tracks = sd.get("tracks", {}) or {}
    sdc_id = str(meta.get("sdc_id", "ego"))
    ego_pos = np.asarray(tracks[sdc_id]["state"]["position"], np.float64)
    actor_position = np.asarray(actor_position, np.float64)
    T = min(len(ego_pos), len(actor_position))
    skip = {sdc_id, *(ignore_track_ids or [])}
    if actor_id:
        skip.add(actor_id)

    visible = np.ones(T, bool)
    blockers: List[Optional[str]] = [None] * T
    gaps = np.linalg.norm(actor_position[:T, :2] - ego_pos[:T, :2], axis=1)

    for frame in range(T):
        eye, target = ego_pos[frame, :2], actor_position[frame, :2]
        reach = float(np.linalg.norm(target - eye))
        for tid, track in tracks.items():
            if tid in skip:
                continue
            box = _boxes(track, frame)
            if box is None:
                continue
            centre, half_l, half_w, heading = box
            # Only something BETWEEN ego and actor can hide it.
            if float(np.linalg.norm(centre - eye)) >= reach:
                continue
            if _segment_hits_box(eye, target, centre, half_l, half_w, heading):
                visible[frame] = False
                blockers[frame] = str(tid)
                break

    closest = int(np.argmin(gaps)) if T else 0
    # Sustained visibility only: a one-frame glimpse through a gap is not a
    # sighting, so short visible runs before the event are discarded.
    min_run = max(1, int(round(_MIN_SUSTAINED_S / max(dt_s, 1e-6))))
    first_visible: Optional[int] = None
    run_start = None
    for frame in range(closest + 1):
        if visible[frame]:
            if run_start is None:
                run_start = frame
            elif frame - run_start + 1 >= min_run:
                first_visible = run_start
                break
        else:
            run_start = None
    if first_visible is None and run_start is not None and closest - run_start + 1 >= min_run:
        first_visible = run_start

    reaction = 0.0 if first_visible is None else max(0.0, (closest - first_visible) * dt_s)
    approach = slice(0, closest + 1)
    return VisibilityReport(
        visible=visible,
        blockers=blockers,
        first_visible_frame=first_visible,
        closest_frame=closest,
        closest_gap_m=float(gaps[closest]) if T else float("inf"),
        reaction_s=float(reaction),
        dt_s=float(dt_s),
        occluded_fraction=float(1.0 - visible[approach].mean()) if closest >= 0 else 0.0,
    )


def dominant_blocker(report: VisibilityReport) -> Optional[str]:
    """The track that does most of the hiding — usually the ego's lead vehicle."""
    counts: Dict[str, int] = {}
    for frame, tid in enumerate(report.blockers[: report.closest_frame + 1]):
        if tid:
            counts[tid] = counts.get(tid, 0) + 1
    return max(counts, key=counts.get) if counts else None


def summarise(report: VisibilityReport) -> Dict[str, Any]:
    return {
        "first_visible_frame": report.first_visible_frame,
        "reaction_s": round(report.reaction_s, 2),
        "closest_gap_m": round(report.closest_gap_m, 2),
        "closest_frame": report.closest_frame,
        "occluded_pct_of_approach": round(100 * report.occluded_fraction, 1),
        "dominant_blocker": dominant_blocker(report),
        "ok": report.ok,
    }
