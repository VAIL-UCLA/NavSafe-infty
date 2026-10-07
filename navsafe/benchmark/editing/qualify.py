# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Host qualification: which event types can THIS host carry?

Not every event type needs a constructed scene. The taxonomy splits into three
tiers, and the split is a property of what triggers the event:

* **mine-only** (V-8, V-10, V-11) — the road itself is the test. Nothing is
  inserted; a metric judges the ego's own behaviour (an illegal turn, an
  unsafe merge, drifting into the opposing lane).
* **mine + conditional insert** (C-10, C-7) — the road is the test, but a
  consequence must be present: oncoming traffic in the opposing lane. If the
  log already has it, nothing is inserted.
* **insert-driven** (R-2, R-3, R-4, I-3) — the event is the inserted actor;
  the host only has to offer the geometry the placement rule needs.

Either way the pipeline's first question is the same: *does this host offer
the road the event type needs?* This module answers it with predicates built on
:class:`~navsafe.benchmark.editing.placement.probe.HostProbe` — the same
queries authoring uses, so "qualified" means exactly "the spec's references
and anchors will resolve".

Scope: geometry on the converted Arrow log. The *tag* tier of mining (nuPlan
``scenario_tag`` runs, 8-camera coverage) lives in ``navsafe/benchmark/mining``
and runs upstream, over the whole split, before anything is converted. The
coarse nuPlan-map roadblock predicate V-10 wants (a lane whose roadblock has
two predecessors) belongs there too and is not implemented here — this
module's ``merge_convergence`` is the *physical* check that runs once a
candidate is converted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from navsafe.benchmark.editing.placement.anchors import find_anchors
from navsafe.benchmark.editing.placement.probe import (
    MERGE_FAR_M,
    MERGE_NEAR_M,
    HostProbe,
)
from navsafe.scenario.scenario_description import ScenarioDescription as SD

logger = logging.getLogger(__name__)

# ── thresholds ──────────────────────────────────────────────────────────────

# A road is "two-way here" when an opposing lane runs beside a majority of the
# probed stations, not just at one point — a single station can clip a
# junction's cross traffic.
TWO_WAY_STATIONS_M = (0.0, 10.0, 20.0)
TWO_WAY_MIN_HITS = 2

# C-7's corridor: the opposing centreline is close enough that two vehicles
# must negotiate. 5.5 m between centrelines ≈ 2.75 m per direction.
NARROW_CENTRELINE_MAX_M = 5.5

# An in-log oncoming vehicle counts when it actually drives (not parked) and
# its heading opposes the route where it passes.
ONCOMING_MIN_TRAVEL_M = 5.0
ONCOMING_MAX_LATERAL_M = 9.0
ONCOMING_HEADING_COS_MAX = -0.5

# I-3's blockage needs a straight stretch: heading drift below this across the
# window, so two static cars + debris read as an in-lane incident, not a bend.
STRAIGHT_WINDOW_M = 25.0
STRAIGHT_MAX_HEADING_DRIFT_RAD = np.deg2rad(15.0)

# V-10's physical merge signature: the thresholds live with the search that
# uses them, in `placement.probe`, so the gate and the placement cannot
# disagree about which lane merged.

# I-3 asks for "three or more lanes and light traffic": the incident has to
# block the ego without walling the road off, so the ego needs somewhere legal
# to go, and traffic thin enough that the manoeuvre is the ego's decision
# rather than the queue's.
MULTI_LANE_MIN = 3

# Traffic is measured on the EGO'S OWN CARRIAGEWAY and per 100 m of route, not
# as a count over the whole scene. Both corrections came from the data: the
# first two hosts a reviewer chose for I-3 carry 42 and 38 moving vehicles,
# which is unremarkable for 20 s of urban nuPlan and says nothing about whether
# the ego can change lanes — only 9 and 14 of those are anywhere near its
# route. And a raw count cannot compare a 244 m route with a 70 m one: the same
# 9-vs-15 becomes 3.7 vs 21 per 100 m, which is the difference between an open
# road and a queue.
#
# The threshold is a lane-change question. A car every ~25 m of your own
# carriageway still leaves gaps to move into; a car every ~10 m does not. Over
# the 29 converted hosts measured, the distribution is median 1.1,
# p75 3.7, max 11.2 per 100 m, so 4.0 sits at the top of the ordinary range and
# rejects only the genuinely dense.
LIGHT_TRAFFIC_MAX_PER_100M = 4.0
MOVING_TRACK_MIN_TRAVEL_M = 3.0
NEAR_ROUTE_LATERAL_M = 12.0


# ── verdicts ────────────────────────────────────────────────────────────────


@dataclass
class Check:
    """One predicate's answer for one host."""

    name: str
    ok: bool
    evidence: str = ""

    def describe(self) -> str:
        return f"{'PASS' if self.ok else 'fail'}  {self.name:<24} {self.evidence}"


@dataclass
class LeafVerdict:
    """Whether one host can carry one event type, with the numbers that say why."""

    leaf: str
    scene: str
    ok: bool
    checks: List[Check] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        head = f"{self.leaf:<6} {'QUALIFIED' if self.ok else 'rejected '}  {self.scene}"
        lines = [head] + [f"    {c.describe()}" for c in self.checks]
        lines += [f"    note: {n}" for n in self.notes]
        return "\n".join(lines)


# ── predicates ──────────────────────────────────────────────────────────────
#
# Each takes a probe and returns a Check. They are pure queries: nothing here
# writes, and a predicate failing is a *finding about the host*, not an error.


def two_way_road(probe: HostProbe) -> Check:
    """An opposing-direction lane runs beside the ego's own road, consistently.

    Measured across a STRAIGHT window (see ``HostProbe.straight_window``), not
    at the hand-off: a junction station reports the branch lanes around it, so
    the hand-off test passed hosts whose "opposing carriageway" was a different
    street. Consistency across stations is the second half of that guard — a
    genuine opposing lane holds a steady offset, a connector does not.
    """
    window = probe.straight_window()
    if window is None:
        return Check(
            "two_way_road", False,
            "the ego route never holds its heading for "
            f"{probe.STRAIGHT_WINDOW_M:.0f} m past the hand-off, so 'the road beside it' "
            "is not a well-posed question on this host (all turning / junction)",
        )
    stations = np.linspace(window[0], window[1], 3)
    hits: List[dict] = []
    for arc in stations:
        opposing = [
            lane
            for lane in probe.parallel_lanes(probe.ego_route, float(arc))
            if not lane["same_direction"]
        ]
        if opposing:
            hits.append(min(opposing, key=lambda lane: abs(lane["lateral_m"])))
    if len(hits) < TWO_WAY_MIN_HITS:
        return Check(
            "two_way_road", False,
            f"opposing lane at {len(hits)}/{len(stations)} stations of the straight "
            f"stretch (need >= {TWO_WAY_MIN_HITS}) — one-way road",
        )
    laterals = [abs(h["lateral_m"]) for h in hits]
    spread = max(laterals) - min(laterals)
    if spread > 2.0:
        return Check(
            "two_way_road", False,
            f"the nearest opposing lane jumps {spread:.1f} m across the straight "
            f"stretch ({[h['lane_id'] for h in hits]}) — junction connectors, not a "
            f"steady opposing carriageway",
        )
    nearest = min(hits, key=lambda lane: abs(lane["lateral_m"]))
    return Check(
        "two_way_road", True,
        f"opposing lane {nearest['lane_id']} at {nearest['lateral_m']:+.1f} m, steady "
        f"(±{spread:.1f} m) over a {window[1] - window[0]:.0f} m straight stretch at "
        f"+{window[0] - probe.anchor_arc(probe.ego_route, probe.after_frame):.0f} m",
    )


def narrow_corridor(probe: HostProbe) -> Check:
    """Two-way AND the opposing centreline close enough to force negotiation."""
    base = two_way_road(probe)
    if not base.ok:
        return Check("narrow_corridor", False, base.evidence)
    lane = probe.opposing_lane()
    gap = abs(lane["lateral_m"]) if lane else float("inf")
    if gap > NARROW_CENTRELINE_MAX_M:
        return Check(
            "narrow_corridor", False,
            f"opposing centreline {gap:.1f} m away (> {NARROW_CENTRELINE_MAX_M}) — "
            f"wide enough to pass without negotiating",
        )
    return Check("narrow_corridor", True, f"opposing centreline {gap:.1f} m away")


def oncoming_vehicle_in_log(probe: HostProbe) -> Check:
    """A real vehicle already drives the opposing direction near the route.

    This is what separates C-10 from C-7 on the same two-way road. C-10 gates
    on it: the ego is the wrong-way driver, so the traffic it meets has to be
    traffic the log already has. C-7 only reports it — it builds its own
    oncoming car, and a host that already has one is C-10's host, not its.
    """
    sd = probe._sd  # noqa: SLF001 - qualification is a probe-family module
    route = probe.ego_route
    for tid, track in (sd.get("tracks") or {}).items():
        if tid == probe.sdc_id or str(track.get("type", "")) != "VEHICLE":
            continue
        state = track.get("state", {})
        pos = np.asarray(state.get("position"), np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2:
            continue
        valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
        if valid.sum() < 2:
            continue
        good = pos[valid]
        if float(np.linalg.norm(good[-1, :2] - good[0, :2])) < ONCOMING_MIN_TRAVEL_M:
            continue
        mid = good[len(good) // 2, :2]
        arc, lateral = route.project(mid)
        if abs(lateral) > ONCOMING_MAX_LATERAL_M or not route.covers(arc, tol=5.0):
            continue
        _, _, _, route_heading = route.sample([arc])
        travel = good[-1, :2] - good[0, :2]
        travel_heading = float(np.arctan2(travel[1], travel[0]))
        if np.cos(travel_heading - float(route_heading[0])) <= ONCOMING_HEADING_COS_MAX:
            return Check(
                "oncoming_vehicle_in_log", True,
                f"track {tid} drives opposing at {lateral:+.1f} m lateral",
            )
    return Check(
        "oncoming_vehicle_in_log", False,
        "no in-log vehicle drives the opposing direction near the route",
    )


def crossing_street(probe: HostProbe) -> Check:
    """A street crosses the ego route ahead — the R-2/R-3 crossing geometry."""
    intersections = [
        a for a in find_anchors(probe) if a.kind == "intersection_entry" and a.arc_m > 0
    ]
    if not intersections:
        return Check(
            "crossing_street", False,
            "no intersection ahead of the hand-off (cli anchors agrees)",
        )
    first = intersections[0]
    return Check(
        "crossing_street", True,
        f"{first.name} at +{first.arc_m:.1f} m — {first.detail}",
    )


def straight_stretch(probe: HostProbe) -> Check:
    """A straight window ahead long enough to stage an in-lane incident (I-3)."""
    route = probe.ego_route
    start = probe.anchor_arc(route, probe.after_frame)
    arcs = np.arange(start, route.total - STRAIGHT_WINDOW_M, 5.0)
    for arc in arcs:
        stations = np.arange(arc, arc + STRAIGHT_WINDOW_M + 1e-9, 5.0)
        _, _, _, headings = route.sample(stations)
        drift = float(np.max(np.abs(np.unwrap(headings) - headings[0])))
        if drift <= STRAIGHT_MAX_HEADING_DRIFT_RAD:
            return Check(
                "straight_stretch", True,
                f"{STRAIGHT_WINDOW_M:.0f} m window at +{arc - start:.0f} m "
                f"(heading drift {np.rad2deg(drift):.0f} deg)",
            )
    return Check(
        "straight_stretch", False,
        f"no {STRAIGHT_WINDOW_M:.0f} m stretch under "
        f"{np.rad2deg(STRAIGHT_MAX_HEADING_DRIFT_RAD):.0f} deg of heading drift",
    )


def merge_convergence(probe: HostProbe) -> Check:
    """A same-direction lane converges into the route: the physical merge test.

    The search itself is :meth:`HostProbe.merging_lane`, shared with the
    ``merging_lane_chain`` reference so that the lane this gate passed on is
    exactly the lane V-10's actor is placed in. It used to be a second copy
    here, and a gate that agrees with the placement only by coincidence is not
    a gate.
    """
    merge = probe.merging_lane()
    if merge is None:
        if probe.straight_window() is None:
            return Check(
                "merge_convergence", False,
                "no straight stretch past the hand-off — a converging lane here would be a "
                "junction turn-fan, not a merge",
            )
        return Check(
            "merge_convergence", False,
            f"no same-direction lane converges {MERGE_FAR_M} -> {MERGE_NEAR_M} m along the "
            f"route",
        )
    return Check(
        "merge_convergence", True,
        f"lane {merge['lane_id']} closes {merge['far_m']:.1f} -> {merge['near_m']:.1f} m, "
        f"merging at {merge['merge_arc_m']:+.0f} m",
    )


def multi_lane(probe: HostProbe) -> Check:
    """At least ``MULTI_LANE_MIN`` same-direction lanes beside the ego (I-3).

    An incident the ego cannot get past is a wall, not a test. Counting
    same-direction lanes rather than "is there an evasion lane" is deliberate:
    the event type wants a road wide enough that going round is an ordinary
    manoeuvre, which is a different question from whether one gap exists.
    """
    window = probe.straight_window()
    arc = (0.5 * (window[0] + window[1]) if window is not None
           else probe.anchor_arc(probe.ego_route, probe.after_frame))
    same = [lane for lane in probe.parallel_lanes(probe.ego_route, float(arc))
            if lane["same_direction"]]
    ok = len(same) >= MULTI_LANE_MIN
    return Check(
        "multi_lane", ok,
        f"{len(same)} same-direction lane(s) at +{arc:.0f} m "
        f"(need {MULTI_LANE_MIN}): {[l['lateral_m'] for l in same]}",
    )


def light_traffic(probe: HostProbe) -> Check:
    """Few enough moving vehicles on the ego's own carriageway to change lanes (I-3).

    Three things are deliberately NOT counted, and each was a way to get the
    wrong answer:

    * **parked cars** — a kerb full of them changes nothing about whether the
      ego can steer around a blockage, and on this data they are most tracks;
    * **traffic elsewhere in the scene** — 40 moving vehicles across a city
      block is an ordinary 20 s of urban nuPlan and says nothing about the
      ego's lane;
    * **the raw count** — nine vehicles over 244 m of route and fifteen over
      70 m are opposite situations, so this reports a density.

    Oncoming traffic is excluded too: it is not somewhere the ego was going to
    go, so it cannot block the manoeuvre this event type is about.
    """
    route = probe.ego_route
    sd = probe._sd  # noqa: SLF001 - qualification is a probe-family module
    near = 0
    for tid, track in (sd.get(SD.TRACKS) or {}).items():
        if tid == probe.sdc_id:
            continue
        if str(track.get("type", "")).upper() not in ("VEHICLE", "CAR", "TRUCK", "BUS"):
            continue
        state = track.get("state") or {}
        pos = np.asarray(state.get("position"), np.float64)
        if pos.ndim != 2:
            continue
        valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
        if valid.sum() < 2:
            continue
        xy = pos[valid][:, :2]
        travel = xy[-1] - xy[0]
        if float(np.linalg.norm(travel)) < MOVING_TRACK_MIN_TRAVEL_M:
            continue                                    # parked
        arc, lateral = route.project(xy[len(xy) // 2])
        if abs(lateral) > NEAR_ROUTE_LATERAL_M or not route.covers(arc, tol=5.0):
            continue                                    # elsewhere in the scene
        _, _, _, route_heading = route.sample([arc])
        if np.cos(float(np.arctan2(travel[1], travel[0])) - float(route_heading[0])) <= 0.5:
            continue                                    # oncoming, not in the way
        near += 1
    length_m = max(route.total, 1.0)
    density = near / length_m * 100.0
    ok = density <= LIGHT_TRAFFIC_MAX_PER_100M
    return Check(
        "light_traffic", ok,
        f"{density:.1f} moving vehicle(s) per 100 m on the ego's carriageway "
        f"({near} over {length_m:.0f} m; max {LIGHT_TRAFFIC_MAX_PER_100M})",
    )


def bike_permitted_lane(probe: HostProbe) -> Check:
    """Somewhere a cyclist would legitimately be, beside the ego (R-2).

    A mapped bike lane is the strong answer and nuPlan's maps do carry them,
    but most streets have none — so the ego's own carriageway lane counts too,
    because that is where a cyclist rides on a street without one, and it is
    the only place the ego has to do anything about them. Both pass; the
    evidence says which, because "dedicated bike lane" and "sharing the
    kerbside lane" are different scenarios to look at.
    """
    lane = probe.bike_lane()
    if lane is None:
        return Check(
            "bike_permitted_lane", False,
            "no same-direction lane beside the ego — nowhere a cyclist belongs",
        )
    kind = "dedicated bike lane" if lane["dedicated"] else "the ego's own lane"
    return Check(
        "bike_permitted_lane", True,
        f"{kind} {lane['lane_id']} at {lane['lateral_m']:+.1f} m",
    )


def crosswalk_on_route(probe: HostProbe) -> Check:
    """A marked crossing the ego drives OVER (R-3 / R-4).

    The search is :meth:`HostProbe.crosswalks_on_route`, shared with the
    ``crosswalk_chain`` reference so the crossing this gate passed on is the
    crossing the actors are placed on. It drops crossings shallower than
    ``CROSSWALK_MIN_ANGLE_DEG`` to the route: a real host carries the side
    street's crossing a few metres from its own, and that one runs ALONGSIDE
    the ego rather than across it.

    This is a hard gate for both event types, not a report. A crowd sent across the
    ego's route mid-block is a different scenario from a crowd on a crossing,
    and letting one stand in for the other hides which was tested.
    """
    found = probe.crosswalks_on_route()
    if not found:
        return Check(
            "crosswalk_on_route", False,
            "no marked crossing on the ego's route past the hand-off (a crossing here "
            "would be mid-block)",
        )
    first = found[0]
    return Check(
        "crosswalk_on_route", True,
        f"{len(found)} crossing(s); first {first['feature_id']} at +{first['arc_m']:.0f} m, "
        f"{first['length_m']:.0f} m long at {first['angle_deg']:.0f} deg to the route",
    )


def merge_pressure(probe: HostProbe) -> Check:
    """Somebody needs the strip of road the ego is on — from either cause.

    A MAP merge is two carriageways drawn as joining. A WORK-ZONE merge is a
    lane the map still shows as open and the log has coned off; nuPlan carries
    no closure layer, so `merge_convergence` cannot see it and three
    reviewer-picked hosts were refused as "no merge here" while their recorded
    ego was plainly threading a contraflow.

    The event type's question is the same either way, so this gate admits both and
    NAMES which — the cause is recorded in the recipe, so a score can still be
    read against the kind of merge it came from rather than against a mixture.
    """
    found = probe.merge_pressure_lane()
    if found is None:
        return Check(
            "merge_pressure", False,
            f"no lane converges {MERGE_FAR_M} -> {MERGE_NEAR_M} m along the route, and no "
            f"work zone closes one beside it",
        )
    d = found["detail"]
    if found["cause"] == "map_merge":
        why = (f"map merge: lane {found['lane_id']} closes {d['far_m']:.1f} -> "
               f"{d['near_m']:.1f} m at {d['merge_arc_m']:+.0f} m")
    else:
        why = (f"work zone: lane {found['lane_id']} at {d['lateral_m']:+.1f} m "
               f"({d['side']}), pinch at {d['pinch_arc_m']:+.0f} m")
    return Check("merge_pressure", True, why)


PREDICATES: Dict[str, Callable[[HostProbe], Check]] = {
    "merge_pressure": merge_pressure,
    "two_way_road": two_way_road,
    "narrow_corridor": narrow_corridor,
    "oncoming_vehicle_in_log": oncoming_vehicle_in_log,
    "crossing_street": crossing_street,
    "straight_stretch": straight_stretch,
    "merge_convergence": merge_convergence,
    "multi_lane": multi_lane,
    "light_traffic": light_traffic,
    "bike_permitted_lane": bike_permitted_lane,
    "crosswalk_on_route": crosswalk_on_route,
}


# ── running them for an event type ─────────────────────────────────────────────────


def qualify_host(
    probe: HostProbe, *, scene: str, leaves: Optional[List[str]] = None
) -> List[LeafVerdict]:
    """Run every requested event type's predicates against one host.

    Which predicates an event type gates on is read from ``leaves/<LEAF>.yaml`` — the
    same declaration `navsafe mine` and `navsafe bake` read. This module used
    to carry its own ``LEAF_MANIFEST`` saying the same thing, and the two had
    already drifted apart: it gated V-10 on ``merge_convergence`` while the
    event type manifest declared no gate at all, and it listed a C-9 that no event type
    file has. One declaration, so that cannot happen again.
    """
    from navsafe.benchmark.event_types import load_all

    manifests = load_all()
    verdicts: List[LeafVerdict] = []
    for leaf in leaves or sorted(manifests):
        man = manifests.get(leaf)
        if man is None:
            raise KeyError(f"unknown event type {leaf!r}; known: {sorted(manifests)}")
        checks: List[Check] = []
        ok = True
        for pred in man.qualify:
            check = PREDICATES[pred](probe)
            checks.append(check)
            ok = ok and check.ok
        for pred in man.qualify_info:
            check = PREDICATES[pred](probe)
            check.name += " (info)"
            checks.append(check)
        notes = [man.notes] if man.notes else []
        if not man.qualify:
            notes.append("no geometry gate — qualification happens upstream (see note)")
        verdicts.append(LeafVerdict(leaf=leaf, scene=scene, ok=ok, checks=checks, notes=notes))
    return verdicts


__all__ = [
    "Check",
    "LeafVerdict",
    "PREDICATES",
    "qualify_host",
]
