# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Semantic anchors: name the places on a host that placement rules talk about.

Event type placement rules are written against road *landmarks* — "the start line
after the first intersection the ego enters, a few metres on" — while the
authoring layer measures everything as an arc from the hand-off. Bridging the
two by hand meant probing cross-sections until the landmark was found, once
per host. This module finds the landmarks instead, so a spec can say

.. code-block:: yaml

    authored:
      anchor: intersection_1_exit   # a named place on THIS host
      arc: 5.0                      # metres past it, along the reference

and baking resolves the name against the host's own geometry. The resolved
numbers are recorded in the frozen recipe (``authored.anchor_solved``), so a
recipe still rebuilds one exact scene — the name is an authoring convenience,
never a replay-time lookup.

**How intersections are found.** This data's maps carry no INTERSECTION
polygon (the py123d conversion emits lanes, boundary lines, crosswalks — see
``_map_lane_type`` / ``_map_line_type``), so intersections are inferred from
what defines them: stretches of the ego route that other lanes *cross* at a
real angle. Stations along the route are tested for crossing lane centrelines;
contiguous crossing stretches merge into intervals; each interval's ends
become ``intersection_<n>_entry`` / ``_exit``. Crosswalk features, where the
map has them, are surfaced as ``crosswalk_<n>`` — often the literal "start
line" a rule means.

All anchor arcs follow the authoring convention: **metres from the hand-off**
(``after_frame``), negative behind it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from navsafe.benchmark.editing.placement.probe import HostProbe, PlacementError
from navsafe.scenario.scenario_description import ScenarioDescription as SD

logger = logging.getLogger(__name__)

# Station spacing along the ego route. Finer buys nothing: lane polylines are
# themselves ~2 m resolution.
STATION_STEP_M = 2.0

# A lane "crosses" when its tangent is 45°-135° off the route's: |cos| below
# this. The surveyed R-2 side street meets the route at 127° (|cos| = 0.60).
CROSSING_COS_MAX = 0.7

# A crossing lane's centreline must pass this close to the route to belong to
# the same junction (half a wide road).
CROSSING_LATERAL_MAX_M = 8.0

# Crossing stretches closer than this merge into one intersection; a stretch
# shorter than this is a skewed driveway, not a junction.
INTERSECTION_MERGE_GAP_M = 8.0
INTERSECTION_MIN_SPAN_M = 4.0

_CROSSWALK_HINT = "CROSSWALK"

# Aliases an author may reach for. ``first_`` reads better in an event type rule than
# ``_1_`` and means the same thing.
_ALIASES = {
    "first_intersection_entry": "intersection_1_entry",
    "first_intersection_exit": "intersection_1_exit",
    "first_crosswalk": "crosswalk_1",
}


@dataclass
class Anchor:
    """One named place on the host, on the ego route."""

    name: str
    kind: str  # handoff | intersection_entry | intersection_exit | crosswalk |
               # workzone_taper | route_end
    arc_m: float  # metres from the hand-off along the ego route (negative = behind)
    xy: Tuple[float, float]  # ego frame-0
    detail: str = ""

    def describe(self) -> str:
        return (
            f"{self.name:<24} arc {self.arc_m:+8.1f} m   xy ({self.xy[0]:8.1f}, "
            f"{self.xy[1]:8.1f})   {self.detail}"
        )


def find_anchors(probe: HostProbe, *, after_frame: Optional[int] = None) -> List[Anchor]:
    """Every anchor this host offers, in route order.

    Args:
        probe: the host, with its map loaded.
        after_frame: the hand-off frame anchor arcs are measured from; defaults
            to ``probe.after_frame``.
    """
    k = int(probe.after_frame if after_frame is None else after_frame)
    route = probe.ego_route
    handoff_arc = probe.anchor_arc(route, k)

    anchors: List[Anchor] = [
        Anchor(
            name="handoff",
            kind="handoff",
            arc_m=0.0,
            xy=_xy_at(route, handoff_arc),
            detail=f"replay hand-off (after_frame={k}); today's default anchor",
        )
    ]
    anchors += _intersection_anchors(probe, handoff_arc)
    anchors += _crosswalk_anchors(probe, handoff_arc)
    anchors += _workzone_anchors(probe, handoff_arc)
    anchors.append(
        Anchor(
            name="route_end",
            kind="route_end",
            arc_m=round(route.total - handoff_arc, 1),
            xy=_xy_at(route, route.total),
            detail="end of the logged ego route — the reconstruction ends near here",
        )
    )
    anchors.sort(key=lambda a: a.arc_m)
    return anchors


def resolve_anchor(probe: HostProbe, name: str, *, after_frame: Optional[int] = None) -> Anchor:
    """Look one anchor up by name.

    Raises:
        PlacementError: no such anchor on this host. The message lists what the
            host actually offers, because "intersection_2_entry" failing on a
            host with one intersection is a host-selection finding, not a typo.
    """
    wanted = _ALIASES.get(str(name), str(name))
    anchors = find_anchors(probe, after_frame=after_frame)
    for anchor in anchors:
        if anchor.name == wanted:
            return anchor
    known = ", ".join(a.name for a in anchors)
    raise PlacementError(
        f"host has no anchor {name!r}. It offers: {known}. If the rule needs a landmark this "
        f"host lacks (a second intersection, a crosswalk), that is a host-selection problem — "
        f"pick a different clip rather than approximating the arc by hand."
    )


# ----------------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------------


def _xy_at(route, arc: float) -> Tuple[float, float]:
    x, y, _, _ = route.sample([float(arc)])
    return (round(float(x[0]), 2), round(float(y[0]), 2))


def _route_lane_ids(probe: HostProbe) -> set:
    """Lanes the route itself runs along — never counted as crossings."""
    ids = set(probe.route_lane_ids())
    if not ids:
        # No recorded route lanes: sample a few stations and take the nearest
        # aligned lane under each.
        route = probe.ego_route
        for arc in np.linspace(0.0, route.total, 8):
            x, y, _, heading = route.sample([float(arc)])
            lane_id, _ = probe.nearest_lane(
                np.array([float(x[0]), float(y[0])]), heading=float(heading[0])
            )
            if lane_id is not None:
                ids.add(lane_id)
    return ids


def _near_route_lane_ids(probe: HostProbe) -> List[str]:
    """Lanes whose raw polyline comes near the route at all — cheap prefilter.

    Real hosts carry thousands of lane features across the whole city tile;
    testing each against every station via :meth:`Polyline.project` would also
    *load* each (which back-fills z from the ego route, point by point). A
    bounding-box test on the raw feature polyline costs nothing and discards
    the distant ones before any of that.
    """
    route_xy = probe.ego_route.xy
    lo = route_xy.min(axis=0) - CROSSING_LATERAL_MAX_M
    hi = route_xy.max(axis=0) + CROSSING_LATERAL_MAX_M
    out: List[str] = []
    for lane_id in probe.lane_ids():
        pts = np.asarray(probe.map_features[lane_id][SD.POLYLINE], np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        if np.any(np.all((pts[:, :2] >= lo) & (pts[:, :2] <= hi), axis=1)):
            out.append(lane_id)
    return out


def _crossing_stations(probe: HostProbe) -> Tuple[np.ndarray, List[List[str]]]:
    """For each station along the route, the lane ids crossing it there.

    Endpoint-clamped projections are kept deliberately: a side street's lanes
    START or END inside the junction, so the clamped point *is* the crossing
    and its tangent still carries the crossing angle.
    """
    route = probe.ego_route
    stations = np.arange(0.0, route.total + 1e-9, STATION_STEP_M)
    xs, ys, _, headings = route.sample(stations)
    own = _route_lane_ids(probe)
    crossing: List[List[str]] = [[] for _ in stations]
    for lane_id in _near_route_lane_ids(probe):
        if lane_id in own:
            continue
        try:
            lane = probe.lane(lane_id)
        except PlacementError:
            continue
        for i, (x, y, theta) in enumerate(zip(xs, ys, headings)):
            s_l, lateral = lane.project(np.array([float(x), float(y)]))
            if abs(lateral) > CROSSING_LATERAL_MAX_M:
                continue
            _, _, _, tangent = lane.sample([s_l])
            if abs(np.cos(float(tangent[0]) - float(theta))) < CROSSING_COS_MAX:
                crossing[i].append(lane_id)
    return stations, crossing


def _intersection_anchors(probe: HostProbe, handoff_arc: float) -> List[Anchor]:
    stations, crossing = _crossing_stations(probe)
    if not len(stations):
        return []
    # Contiguous crossing stretches -> [start_arc, end_arc, lane ids] intervals.
    intervals: List[List] = []
    for arc, lanes in zip(stations, crossing):
        if not lanes:
            continue
        if intervals and arc - intervals[-1][1] <= INTERSECTION_MERGE_GAP_M:
            intervals[-1][1] = float(arc)
            intervals[-1][2].update(lanes)
        else:
            intervals.append([float(arc), float(arc), set(lanes)])
    out: List[Anchor] = []
    n = 0
    for start, end, lanes in intervals:
        if end - start < INTERSECTION_MIN_SPAN_M:
            continue
        n += 1
        shown = sorted(lanes)
        listed = ", ".join(shown[:4]) + ("…" if len(shown) > 4 else "")
        route = probe.ego_route
        out.append(
            Anchor(
                name=f"intersection_{n}_entry",
                kind="intersection_entry",
                arc_m=round(start - handoff_arc, 1),
                xy=_xy_at(route, start),
                detail=f"crossing lanes first reach the route ({listed})",
            )
        )
        out.append(
            Anchor(
                name=f"intersection_{n}_exit",
                kind="intersection_exit",
                arc_m=round(end - handoff_arc, 1),
                xy=_xy_at(route, end),
                detail="last crossing-lane station — the far side; wrong-way traffic for the "
                "crossing road starts past here",
            )
        )
    return out


def _crosswalk_anchors(probe: HostProbe, handoff_arc: float) -> List[Anchor]:
    route = probe.ego_route
    out: List[Anchor] = []
    n = 0
    for key, feat in probe.map_features.items():
        type_name = str(feat.get(SD.TYPE, "")).upper()
        if _CROSSWALK_HINT not in type_name and _CROSSWALK_HINT not in key.upper():
            continue
        pts = feat.get(SD.POLYLINE)
        if pts is None:
            continue
        pts = np.asarray(pts, np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        centre = pts[:, :2].mean(axis=0)
        arc, lateral = route.project(centre)
        if abs(lateral) > CROSSING_LATERAL_MAX_M or not route.covers(arc, tol=1.0):
            continue
        n += 1
        out.append(
            Anchor(
                name=f"crosswalk_{n}",
                kind="crosswalk",
                arc_m=round(arc - handoff_arc, 1),
                xy=_xy_at(route, arc),
                detail=f"map feature {key}",
            )
        )
    out.sort(key=lambda a: a.arc_m)
    # Number in route order, not map-dict order.
    for i, anchor in enumerate(out, start=1):
        anchor.name = f"crosswalk_{i}"
    return out


_ANCHOR_NAME_RE = re.compile(
    r"^(handoff|route_end|(?:intersection_\d+_(?:entry|exit))|(?:crosswalk_\d+)"
    r"|first_intersection_(?:entry|exit)|first_crosswalk)$"
)


def is_anchor_name(name: str) -> bool:
    """Whether ``name`` is shaped like an anchor this module can resolve."""
    return bool(_ANCHOR_NAME_RE.match(str(name)))


__all__ = [
    "Anchor",
    "find_anchors",
    "is_anchor_name",
    "resolve_anchor",
]


#: A work zone is logged as ORDINARY TRACKS -- one per cone, one per barrier --
#: so it is found the way a cone line is seen from a car: several of them, not
#: moving, strung along one side of the road. Two is a couple of stray objects;
#: this many in a row is a closure.
WORKZONE_MIN_RUN = 5
#: How far off the ego's line the clutter still belongs to its carriageway.
#: Wider than a lane, because the taper starts at the shoulder and walks in.
WORKZONE_MAX_LATERAL_M = 12.0
#: A gap longer than this ends one run and starts another, so two separate
#: closures on one route do not merge into a single 200 m "work zone".
WORKZONE_MAX_GAP_M = 25.0
_WORKZONE_TYPES = ("TRAFFIC_CONE", "TRAFFIC_BARRIER")


def _workzone_anchors(probe: HostProbe, handoff_arc: float) -> List[Anchor]:
    """Where a lane closure begins, per run of cones/barriers along the route.

    This is the anchor a merge-under-pressure scenario needs, and it is a fact
    about the LOG rather than about the map: nuPlan's lane graph does not know
    the lane is coned off, so `merging_lane_chain` finds nothing and the host
    reads as "no merge here" while the recorded ego is plainly threading a
    contraflow. Three reviewer-picked hosts were refused that way; all three
    carry 13-179 barrier tracks within a lane's width of the route.

    The anchor is the START of the run -- the taper -- because that is the
    point the ego has to be merged by, not the length of cones it then drives
    past.
    """
    route = probe.ego_route
    sd = getattr(probe, "_sd", {}) or {}
    hits = []
    for tid, track in (sd.get(SD.TRACKS) or {}).items():
        if tid == probe.sdc_id:
            continue
        if str(track.get("type", "")).upper() not in _WORKZONE_TYPES:
            continue
        state = track.get("state", {})
        pos = np.asarray(state.get("position"), np.float64)
        if pos.ndim != 2 or len(pos) < 1:
            continue
        valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
        if not valid.any():
            continue
        xy = pos[valid][0, :2]
        arc, lateral = route.project(xy)
        if abs(lateral) > WORKZONE_MAX_LATERAL_M or not route.covers(arc, tol=2.0):
            continue
        hits.append((float(arc), float(lateral)))
    if len(hits) < WORKZONE_MIN_RUN:
        return []
    hits.sort()
    out: List[Anchor] = []
    run = [hits[0]]
    for h in hits[1:] + [(float("inf"), 0.0)]:
        if h[0] - run[-1][0] <= WORKZONE_MAX_GAP_M:
            run.append(h)
            continue
        if len(run) >= WORKZONE_MIN_RUN:
            start = run[0][0]
            side = float(np.median([r[1] for r in run]))
            out.append(Anchor(
                name=f"workzone_{len(out) + 1}",
                kind="workzone_taper",
                arc_m=round(start - handoff_arc, 2),
                xy=_xy_at(route, start),
                detail=(f"{len(run)} cones/barriers over "
                        f"{run[-1][0] - start:.0f} m, "
                        f"{'left' if side > 0 else 'right'} side "
                        f"({side:+.1f} m); taper begins here"),
            ))
        run = [h]
    return out
