# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Route-frame geometry: arc length, lateral offset, and conflict points.

A trajectory is authored as ``(reference polyline, template, parameters)`` —
arc length ``s`` along the reference, lateral offset, heading offset, speed —
because that is how a human states the intent ("52 m ahead, in my lane, facing
me, 8 m/s"). World coordinates are the output, never the input. This module is
the conversion, and nothing in it knows about scenarios, hosts or recipes.

The sign conventions, fixed once here:

* ``s`` grows along the reference's own direction.
* lateral offset is **+left** of that direction.
* heading is measured from the reference tangent, so a heading offset of 0 is
  "with the reference" and ``pi`` is "against it".
* **speed is signed**: positive travels with the reference, negative against
  it. That single convention is what makes an oncoming actor expressible at
  all — the heading follows from the velocity vector, so nothing extra is
  needed to turn it around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Polyline:
    """An (N, 2) path with per-vertex height and a cumulative arc table.

    Sampling **extrapolates** past either end along the end segment's tangent
    rather than clamping. Clamping is the more dangerous default: it silently
    parks an actor at the end of the route and the recipe still looks correct.
    Callers that care (they should) check :meth:`covers`.
    """

    xy: np.ndarray  # (N, 2) float64
    z: np.ndarray  # (N,)   float64
    arc: np.ndarray  # (N,) float64, arc[0] == 0
    name: str = ""
    # Is ``z`` the ROAD SURFACE, or a pose height above it? The ego track's z
    # is lifted ~1.4-1.7 m above the road (it is an IMU/rig pose), while map
    # polylines are already road-referenced. Getting this wrong floats or
    # buries every actor baked against the reference, so it is carried
    # explicitly rather than inferred.
    z_is_road_surface: bool = True

    # -- construction ----------------------------------------------------
    @classmethod
    def from_points(
        cls, points, *, z=None, name: str = "", z_is_road_surface: bool = True
    ) -> "Polyline":
        """Build from an (N, 2) or (N, 3) array, dropping consecutive duplicates."""
        pts = np.asarray(points, np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            raise ValueError(f"a reference polyline needs >= 2 points of >= 2 columns, got {pts.shape}")
        xy = pts[:, :2]
        if z is not None:
            zs = np.asarray(z, np.float64).reshape(-1)
            if zs.shape[0] != xy.shape[0]:
                raise ValueError(f"z has {zs.shape[0]} entries for {xy.shape[0]} points")
        elif pts.shape[1] >= 3:
            zs = pts[:, 2]
        else:
            zs = np.zeros(xy.shape[0])
        # Repeated vertices give a zero-length segment and a NaN tangent.
        keep = np.concatenate([[True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9])
        xy, zs = xy[keep], zs[keep]
        if xy.shape[0] < 2:
            raise ValueError("reference polyline collapsed to a single point")
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        return cls(xy=xy, z=zs, arc=arc, name=name, z_is_road_surface=z_is_road_surface)

    # -- queries ---------------------------------------------------------
    @property
    def total(self) -> float:
        """Arc length of the whole reference, in metres."""
        return float(self.arc[-1])

    def covers(self, s, *, tol: float = 0.0) -> bool:
        """Is every arc value in ``s`` inside the reference (within ``tol``)?"""
        s = np.asarray(s, np.float64)
        return bool(np.all(s >= -tol) and np.all(s <= self.total + tol))

    def sample(self, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Position, height and tangent heading at arc length(s) ``s``.

        Returns:
            ``(x, y, z, heading)``, each shaped like ``s``. Heading is the
            reference's own tangent — which is why the choice of reference
            decides the shape of the motion: a turn-lane centreline through a
            junction produces a turning vehicle, and a sidewalk polyline
            produces an actor that travels along the kerb.
        """
        s = np.atleast_1d(np.asarray(s, np.float64))
        j = np.clip(np.searchsorted(self.arc, s), 1, len(self.arc) - 1)
        span = np.maximum(self.arc[j] - self.arc[j - 1], 1e-9)
        f = (s - self.arc[j - 1]) / span  # not clipped: extrapolates past the ends
        delta = self.xy[j] - self.xy[j - 1]
        p = self.xy[j - 1] + f[:, None] * delta
        z = self.z[j - 1] + f * (self.z[j] - self.z[j - 1])
        heading = np.arctan2(delta[:, 1], delta[:, 0])
        return p[:, 0], p[:, 1], z, heading

    def offset(self, s, lateral) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample at ``s`` then step ``lateral`` metres to the **left**."""
        x, y, z, heading = self.sample(s)
        lateral = np.broadcast_to(np.asarray(lateral, np.float64), x.shape)
        return x - np.sin(heading) * lateral, y + np.cos(heading) * lateral, z, heading

    def road_z(self, s, *, ego_z_to_ground_m: float) -> np.ndarray:
        """Road-surface height at ``s``, whatever convention ``z`` is in."""
        _, _, z, _ = self.sample(s)
        return z if self.z_is_road_surface else z - float(ego_z_to_ground_m)

    def project(self, point) -> Tuple[float, float]:
        """Nearest point on the reference to ``point``.

        Returns:
            ``(s, lateral)`` — arc length of the foot of the perpendicular, and
            the signed (+left) distance to it.
        """
        p = np.asarray(point, np.float64).reshape(2)
        a, b = self.xy[:-1], self.xy[1:]
        d = b - a
        length2 = np.maximum(np.sum(d * d, axis=1), 1e-12)
        t = np.clip(np.sum((p - a) * d, axis=1) / length2, 0.0, 1.0)
        foot = a + t[:, None] * d
        dist = np.linalg.norm(p - foot, axis=1)
        i = int(np.argmin(dist))
        s = float(self.arc[i] + t[i] * np.linalg.norm(d[i]))
        # +left: the z component of tangent x (point - foot).
        cross = d[i, 0] * (p[1] - foot[i, 1]) - d[i, 1] * (p[0] - foot[i, 0])
        lateral = float(np.sign(cross) * dist[i])
        return s, lateral


def closest_approach(
    a: "Polyline", b: "Polyline", *, after_arc_a: float = 0.0
) -> Tuple[float, float, "np.ndarray", float]:
    """Where two references come nearest, at or after ``after_arc_a`` on ``a``.

    A MERGE is the case this exists for. Two lanes that merge converge to
    within a lane width and then run together; they need never cross, so
    :func:`intersect_polylines` returns nothing and "arrive at the same place at
    the same time" has no crossing to hang on. Their point of closest approach
    is that place.

    Sampled on ``a``'s own vertices, which are the map's, so the resolution is
    the map's own — good to well under a metre on this data and not worth
    refining further, since the answer feeds a spawn arc.

    Returns:
        ``(s_a, s_b, xy, distance_m)`` — the arc on each, the point midway
        between them, and how far apart they actually get.
    """
    xy_a, arc_a = a.xy, a.arc
    keep = arc_a >= float(after_arc_a)
    if not np.any(keep):
        keep = np.ones(len(arc_a), bool)
    best = None
    for idx in np.flatnonzero(keep):
        s_b, lateral = b.project(xy_a[idx])
        d = abs(float(lateral))
        if best is None or d < best[3]:
            bx, by, _, _ = b.sample([s_b])
            mid = 0.5 * (xy_a[idx] + np.array([float(bx[0]), float(by[0])]))
            best = (float(arc_a[idx]), float(s_b), mid, d)
    return best


def intersect_polylines(
    a: Polyline, b: Polyline, *, after_arc_a: float = 0.0
) -> Optional[Tuple[float, float, np.ndarray]]:
    """First crossing of two references, at or after ``after_arc_a`` on ``a``.

    This is how a cross-street conflict point is found: where the actor's own
    polyline actually meets the ego's route, rather than an offset guessed from
    the ego's path.

    Returns:
        ``(s_a, s_b, xy)`` at the crossing, or ``None`` if they never cross.
    """
    p, r = a.xy[:-1], np.diff(a.xy, axis=0)
    q, s_vec = b.xy[:-1], np.diff(b.xy, axis=0)
    best: Optional[Tuple[float, float, np.ndarray]] = None
    for i in range(len(r)):
        denom = r[i, 0] * s_vec[:, 1] - r[i, 1] * s_vec[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = q - p[i]
            t = (diff[:, 0] * s_vec[:, 1] - diff[:, 1] * s_vec[:, 0]) / denom
            u = (diff[:, 0] * r[i, 1] - diff[:, 1] * r[i, 0]) / denom
        hit = np.isfinite(t) & np.isfinite(u) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if not np.any(hit):
            continue
        seg_len_a = float(np.linalg.norm(r[i]))
        for jdx in np.flatnonzero(hit):
            s_a = float(a.arc[i] + t[jdx] * seg_len_a)
            if s_a < after_arc_a:
                continue
            s_b = float(b.arc[jdx] + u[jdx] * float(np.linalg.norm(s_vec[jdx])))
            if best is None or s_a < best[0]:
                best = (s_a, s_b, p[i] + t[jdx] * r[i])
    return best


def solve_start_arc_for_conflict(
    reference: Polyline,
    *,
    conflict_arc: float,
    speed: float,
    conflict_frame: int,
    dt_s: float,
) -> float:
    """Back-solve the arc an actor must start at to reach a point on cue.

    Timing intents are solved into geometry rather than authored as arc values:
    a leaf says "arrive as the ego reaches the conflict point", and this turns
    that into the arc the actor occupies at frame 0. ``conflict_frame`` is read
    off the LOG-REPLAY ego, the only one that exists at bake time. What the
    recipe keeps is the arc this returns — a spawn position — not a promise
    about when the policy under test will get there.

    Args:
        reference: the actor's own reference.
        conflict_arc: arc length on ``reference`` of the conflict point.
        speed: signed m/s along the reference.
        conflict_frame: scenario frame at which the actor should be there.
        dt_s: the host's frame interval.

    Returns:
        The arc length the actor occupies at frame 0.
    """
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    del reference  # kept in the signature: the arc is only meaningful against it
    return float(conflict_arc - speed * dt_s * int(conflict_frame))
