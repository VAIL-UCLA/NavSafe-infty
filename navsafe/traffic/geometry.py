# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Route-frame geometry shared by the traffic managers.

An agent driven along a path needs three things from that path: where am I on
it (arc length), where does arc ``s`` put me in the world, and is that other
car in my lane. Both the MetaDrive-derived ``semi_reactive`` manager and the
recipe-driven ``navsafe`` one need exactly this, so it lives here rather than
being written twice with two sets of sign conventions.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

#: How wide the corridor is when asking "is that object in my lane".
ROUTE_WIDTH_M = 2.0

#: Shortest polyline segment :class:`Route` will keep, in metres.
#:
#: A route seeded from a logged track (``semi_reactive._decide`` passes the
#: vehicle's own logged positions) contains the stretches where that vehicle was
#: STOPPED. At 10 Hz a stopped vehicle's consecutive samples differ by
#: localization noise, not motion, and a heading read off such a pair is
#: therefore random. Measured on bundle ``01a58976a2e45a3d``: of 43
#: takeover-eligible agents, six had route segments of 0.0000-0.0055 m with
#: 16-94 of their ~100-200 segments under 5 cm, and sampling ``heading_at``
#: along those routes swung up to **179.3 deg** — the vehicle's rendered box
#: flipping end-for-end as ``s`` crossed a noise-scale segment. That is the
#: field of splayed, crosswise vehicles seen in a crowded junction.
#:
#: 0.20 m sits above the noise (cm-scale) and below one frame of real motion
#: for even a slow vehicle (1 m/s at 10 Hz = 0.10 m... so a crawling vehicle's
#: points merge, which is correct: it has no reliable heading of its own either,
#: and holding the last good one is what a stopped car does). Dropping points
#: closer than 20 cm leaves a lane polyline's shape intact.
#:
#: ``_extend_route`` in ``semi_reactive`` already applied this insight to the
#: direction of the route's straight extension; it was never applied to the
#: route body, which is where the artifact came from.
MIN_SEGMENT_M = 0.20


class Route:
    """A polyline with arc length, projection and a lane corridor."""

    __slots__ = ("xy", "s", "width", "min_segment_m", "fallback_heading")

    def __init__(self, xy, width: float = ROUTE_WIDTH_M,
                 min_segment_m: float = MIN_SEGMENT_M,
                 fallback_heading: float | None = None) -> None:
        """``fallback_heading`` is what ``heading_at`` reports when the polyline
        carries no usable direction at all — a caller that knows the object's
        heading from somewhere other than its own positions (a logged track has
        a ``heading`` field) should pass it. Without it the fallback can only be
        read off the same noise that made the question unanswerable.
        """
        xy = np.asarray(xy, dtype=np.float64)[:, :2]
        if len(xy) < 2:
            raise ValueError("a route needs at least two points")
        self.fallback_heading = (None if fallback_heading is None
                                 else float(fallback_heading))
        # Kept so ``heading_at`` searches with the same floor the points were
        # filtered against: a caller that opts out with ``min_segment_m=0``
        # would otherwise still have its short segments skipped when reading a
        # heading, which is the opposite of what opting out asks for.
        self.min_segment_m = float(min_segment_m)
        self.xy = _drop_noise_points(xy, self.min_segment_m)
        seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.width = float(width)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def end(self) -> np.ndarray:
        return self.xy[-1]

    def local_coordinates(self, p) -> Tuple[float, float]:
        """``(longitudinal, lateral)`` of ``p``, on the nearest segment.

        Lateral is signed **+left** of the direction of travel — the same
        convention the NavSafe placement layer uses, so a kerb offset means
        the same thing on both sides of the pipeline.
        """
        p = np.asarray(p, dtype=np.float64)[:2]
        a = self.xy[:-1]
        ab = self.xy[1:] - a
        seg_len2 = np.einsum("ij,ij->i", ab, ab)
        seg_len2 = np.where(seg_len2 < 1e-12, 1e-12, seg_len2)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / seg_len2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - p, axis=1)
        i = int(np.argmin(d))
        seg_len = math.sqrt(seg_len2[i])
        long = float(self.s[i] + t[i] * seg_len)
        ux, uy = ab[i, 0] / seg_len, ab[i, 1] / seg_len
        lat = float(-uy * (p[0] - a[i, 0]) + ux * (p[1] - a[i, 1]))
        return long, lat

    def point_on_lane(self, p) -> bool:
        """Within half a width laterally and inside the route's own extent."""
        long, lat = self.local_coordinates(p)
        return abs(lat) <= 0.5 * self.width and -1e-6 <= long <= self.length + 1e-6

    def position_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        return np.array([float(np.interp(s, self.s, self.xy[:, 0])),
                         float(np.interp(s, self.s, self.xy[:, 1]))])

    def heading_at(self, s: float) -> float:
        """Heading of the segment containing ``s``.

        The constructor removes noise-scale segments, so the segment under
        ``s`` normally carries real motion. The outward search below covers what
        it cannot: a route whose LAST kept segment is short (the final point is
        always kept, however close it fell), and a fully degenerate route.

        The old guard was ``> 1e-6`` — one micron — with a fallback of 0.0 rad,
        i.e. due east. Both failure modes were silent: nothing raises, the
        vehicle is simply drawn and simulated facing the wrong way.
        """
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.s, s, side="right"))
        i = max(1, min(i, len(self.s) - 1))
        n = len(self.xy)
        floor = max(self.min_segment_m, 1e-9)
        # Walk outward from the containing segment to the nearest usable one.
        for j in range(0, n):
            for k in (i - j, i + j):
                if 1 <= k <= n - 1:
                    d = self.xy[k] - self.xy[k - 1]
                    if float(np.linalg.norm(d)) >= floor:
                        return math.atan2(float(d[1]), float(d[0]))
        # Every segment is below the floor: the whole polyline is one noise
        # cluster, and nothing in it answers the question. Prefer what the
        # caller knows from outside the polyline; otherwise report the gross
        # first-to-last direction, so a caller still gets the object's axis
        # rather than due east.
        if self.fallback_heading is not None:
            return self.fallback_heading
        d = self.xy[-1] - self.xy[0]
        if float(np.linalg.norm(d)) > 0.0:
            return math.atan2(float(d[1]), float(d[0]))
        return 0.0


def _drop_noise_points(xy: np.ndarray, min_segment_m: float) -> np.ndarray:
    """``xy`` with points closer than ``min_segment_m`` to the last kept one removed.

    The first and last points are always kept, so a route never loses its
    extent and never drops below two points. A polyline that is entirely one
    noise cluster therefore survives as its two endpoints, with a near-zero
    length. ``semi_reactive`` never builds one: such a track is classified
    static (``STATIC_THRESHOLD_M``) and is ineligible for takeover, so it stays
    on log replay where it belongs. Note the ``route.length <= 0`` guard in
    ``_decide`` would NOT catch it — a few centimetres of noise is still
    positive — and ``heading_at`` handles the case directly rather than
    relying on the caller.
    """
    if min_segment_m <= 0.0 or len(xy) < 3:
        return xy
    keep = [0]
    for i in range(1, len(xy) - 1):
        if float(np.linalg.norm(xy[i] - xy[keep[-1]])) >= min_segment_m:
            keep.append(i)
    keep.append(len(xy) - 1)
    return xy[keep]


def corners(x: float, y: float, yaw: float, length: float,
            width: float) -> List[Tuple[float, float]]:
    """The four bbox corners, so a lane test uses the footprint not the centre."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))]


__all__ = ["MIN_SEGMENT_M", "ROUTE_WIDTH_M", "Route", "corners"]


def body(pose):
    """Convex footprint polygon for a pose with length and width."""
    from shapely.geometry import Polygon
    position = pose["position"]
    return Polygon(corners(float(position[0]), float(position[1]),
                           float(pose["heading"]), float(pose["length"]),
                           float(pose["width"]))).convex_hull
