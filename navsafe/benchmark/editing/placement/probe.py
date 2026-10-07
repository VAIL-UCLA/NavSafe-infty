# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Host probe: what a given reconstructed scene actually offers.

Lateral offsets are measured from a reference centreline, so whether −5.0 m is
the shoulder, the verge or the oncoming lane is **a property of the host, not a
constant**. The probe is where that gets measured, so an author states intent
("the right shoulder") and the recipe records the metre value this host makes
that mean.

Three reference kinds, and the choice decides the shape of the motion because
heading follows the reference:

``ego_route``     the ego's own logged route — only as long as the ego drove.
``ego_lane_chain`` the lane the ego is in at the hand-off, extended along the
                  map's own successor graph: the same road, running well past
                  either end of the route, which is what an actor closing
                  head-on for a whole episode needs.
``route_lanes``   the ego's traversed-lane chain, stitched — refuses when the
                  chain contains a lane change rather than a successor.
``opposing_lane_chain`` the chain through the opposing carriageway beside the
                  hand-off, in ITS travel direction: ordinary oncoming traffic
                  authors as positive speed, ego-forward spawns as negative arc.
``crossing_lane_chain[:N]`` a lane that crosses the ego's route ahead — the
                  side street an actor rides in from.
``merging_lane_chain`` the same-direction lane that CONVERGES with the ego's
                  route: the mainline an ego on a slip road has to enter, or
                  the slip road entering the ego's mainline. An actor on it and
                  an ego on the route arrive at one strip of road, which is the
                  whole of V-10.
``bike_lane_chain`` the lane bicycles belong in beside the ego — a mapped bike
                  lane where the host has one, else the ego's own lane.
``crosswalk_N``   a marked pedestrian crossing the ego drives over, as the line
                  an actor WALKS: kerb to kerb, in the crossing's own
                  direction. Numbered in route order, so ``crosswalk_1`` is the
                  first one ahead of the hand-off.
``lane:<id>``     a lane centreline from the map. A turn-lane centreline
                  through a junction produces a turning vehicle.
``polyline``      an inline polyline, for geometry the map does not name.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from navsafe.benchmark.editing.placement.solve import Polyline
from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.scenario_description import scenario_dt_seconds
from navsafe.errors import NavSafeError

logger = logging.getLogger(__name__)

# Map feature types that bound the drivable surface. Names come from
# ``_map_lane_type`` / ``_map_line_type`` in py123d_scenario_description.py;
# matching on a substring keeps this working across their spellings.
_ROAD_EDGE_HINT = "ROAD_EDGE"
_LANE_HINT = "LANE"
_BIKE_LANE_HINT = "BIKE"
_CROSSWALK_HINT = "CROSSWALK"

# A crossing at less than this angle to the ego's route is the crossing of the
# OTHER street at a junction — the ego drives alongside it, not over it. Real
# hosts carry both, a few metres apart (0dc54a8c8203567b has one at 83 deg and
# one at 5 deg within 6 m), so taking "the nearest crosswalk" without the angle
# puts a pedestrian walking down the ego's own lane instead of across it.
CROSSWALK_MIN_ANGLE_DEG = 45.0

# A merge is not a topology fact. "A lane with >= 2 incoming edges" fires on
# half of nuPlan because lanes split at every segment boundary; what a split
# cannot fake is a SEPARATE lane whose centreline closes on the route as arc
# advances. These are that test, and they live here rather than in `qualify`
# because the predicate and the `merging_lane_chain` reference have to agree
# about which lane merged — two searches would drift, and the event type would gate
# on one lane and place its actor on another.
MERGE_FAR_M = 2.5
MERGE_NEAR_M = 1.0
#: And no FURTHER than this at its far end. `far >= MERGE_FAR_M` alone has no
#: upper bound, so a road a hundred metres away that eventually joins passes
#: the far-then-near test as readily as the slip road beside the ego -- on
#: 13c555e68671524f it picked a lane starting 65 m off and spawned the actor
#: 134 m to the side, out of frame, so the event type staged nothing. A feeder lane
#: is a road width or two away where it starts, not a different road.
MERGE_FAR_MAX_M = 20.0
# 1 m, not 5. A NavSafe host's route past the hand-off is 65-130 m, and at 5 m
# a converging lane landed on one to three stations -- too few for a
# far-then-near profile to exist at all, so the test reported "no merge" on
# hosts that plainly have one. Measured on 6c9e40634f705f56 and
# 34ac200e359653b5: 5 m gave n=1..3 and no verdict, 1 m gives n=7..54 and both
# qualify on the same thresholds. The cost is 5x more projections over a
# 100 m route, which is nothing next to loading the scenario.
MERGE_STATION_STEP_M = 1.0


class PlacementError(NavSafeError, ValueError):
    """The host cannot support the placement the author asked for."""


@dataclass
class CrossSection:
    """The road's shape at one point, in lateral metres from the reference.

    Every value is **+left of the reference direction**, so a right-hand
    shoulder is negative. ``None`` means the host's map does not carry it —
    a real answer for a reconstruction, and better than a plausible default.
    """

    arc: float
    lane_id: Optional[str] = None
    lane_width_m: Optional[float] = None
    left_boundary_m: Optional[float] = None
    right_boundary_m: Optional[float] = None
    left_road_edge_m: Optional[float] = None
    right_road_edge_m: Optional[float] = None

    def describe(self) -> str:
        """One line a reviewer can sanity-check the placement numbers against."""

        def fmt(value, unit="m"):
            return "n/a" if value is None else f"{value:+.2f}{unit}"

        return (
            f"arc {self.arc:.1f}m  lane={self.lane_id or 'n/a'} "
            f"width={'n/a' if self.lane_width_m is None else f'{self.lane_width_m:.2f}m'} "
            f"boundaries L{fmt(self.left_boundary_m)} R{fmt(self.right_boundary_m)} "
            f"road-edge L{fmt(self.left_road_edge_m)} R{fmt(self.right_road_edge_m)}"
        )


class HostProbe:
    """Read-only view of one host scenario, in its own ego frame-0 coordinates."""

    def __init__(self, sd: dict, *, ego_z_to_ground_m: float = 0.0):
        self._sd = sd
        meta = sd.get(SD.METADATA, {}) or {}
        self.ego_z_to_ground_m = float(ego_z_to_ground_m)
        self.sdc_id = str(meta.get(SD.SDC_ID, "ego"))
        ego = (sd.get(SD.TRACKS) or {}).get(self.sdc_id)
        if not ego:
            raise PlacementError(f"host has no ego track {self.sdc_id!r}")
        state = ego.get("state", {})
        pos = np.asarray(state.get("position"), np.float64)
        if pos.ndim != 2 or pos.shape[0] < 2:
            raise PlacementError("host ego track is too short to define a route")
        self.T = int(pos.shape[0])
        self.dt_s = float(scenario_dt_seconds(meta, default=0.1))
        ts = meta.get(SD.TIMESTEP)
        self.timestamps_us: List[int] = (
            [int(round(float(t))) for t in np.asarray(ts, np.float64).reshape(-1)]
            if ts is not None and np.asarray(ts).size == self.T
            else []
        )
        self.ego_position = pos
        valid = np.asarray(state.get("valid", np.ones(self.T, bool))).astype(bool)
        if valid.sum() < 2:
            raise PlacementError("host ego track has fewer than two valid frames")
        # The ego track's z is a rig pose, not the road surface.
        self.ego_route = Polyline.from_points(
            pos[valid][:, :3], name="ego_route", z_is_road_surface=False
        )
        self._lane_cache: Dict[str, Polyline] = {}
        # Which frame "arc 0" and ``ego_lane_chain`` mean; the authoring layer
        # sets this from the recipe's ego.replay_frames.
        self.after_frame = 0

    # -- frame grid ------------------------------------------------------
    def frames_dict(self, after_frame: int) -> Dict[str, Any]:
        """The ``frames`` block a recipe records, taken from the host itself."""
        if not (0 <= int(after_frame) < self.T):
            raise PlacementError(
                f"after_frame {after_frame} is outside this host's episode (T={self.T})"
            )
        out: Dict[str, Any] = {
            "T": self.T,
            "dt_s": round(self.dt_s, 6),
            "after_frame": int(after_frame),
        }
        if self.timestamps_us:
            out["timestamps_us"] = list(self.timestamps_us)
        return out

    # -- references ------------------------------------------------------
    @property
    def map_features(self) -> Dict[str, dict]:
        return self._sd.get(SD.MAP_FEATURES) or {}

    def lane_ids(self) -> List[str]:
        """Every lane in the host's map, by id."""
        return [
            key
            for key, feat in self.map_features.items()
            if _LANE_HINT in str(feat.get(SD.TYPE, "")).upper()
            and feat.get(SD.POLYLINE) is not None
        ]

    def route_lane_ids(self) -> List[str]:
        """Lanes the logged ego actually drove through, in order (if derived)."""
        return [str(x) for x in (self._sd.get(SD.METADATA, {}) or {}).get("route_lane_ids", [])]

    def lane(self, lane_id: str) -> Polyline:
        """A lane's centreline as a reference polyline."""
        lane_id = str(lane_id)
        if lane_id in self._lane_cache:
            return self._lane_cache[lane_id]
        feat = self.map_features.get(lane_id)
        if feat is None:
            raise PlacementError(
                f"host map has no lane {lane_id!r}; it carries {len(self.lane_ids())} lanes"
            )
        polyline = feat.get(SD.POLYLINE)
        if polyline is None:
            raise PlacementError(f"map feature {lane_id!r} has no centreline polyline")
        pts = np.asarray(polyline, np.float64)
        # Map lines can be 2-D, and this data's lane centrelines routinely carry
        # a height column of exact zeros — which is not "the road is at 0", it
        # is "no height was written". Baking against it puts every actor on a
        # flat plane several metres off the reconstructed road, so the ego's own
        # road surface is borrowed instead.
        if pts.shape[1] < 3 or float(np.max(np.abs(pts[:, 2]))) < 1e-6:
            z = self._road_z_from_ego(pts[:, :2])
            pts = np.concatenate([pts[:, :2], z.reshape(-1, 1)], axis=1)
        line = Polyline.from_points(pts, name=f"lane:{lane_id}")
        self._lane_cache[lane_id] = line
        return line

    # Lanes in a chain must actually join: beyond this the "chain" is a lane
    # CHANGE, not a successor, and stitching it would insert a sideways jump
    # an actor would teleport across.
    SEAM_GAP_TOL_M = 3.0
    # A successor must both start near the current lane's end and continue in
    # roughly the same direction; intersections offer several, including ones
    # that double back.
    SUCCESSOR_ALIGN = 0.5  # cos(60 deg)

    def route_lanes(self) -> Polyline:
        """The ego's traversed-lane chain, stitched into one long reference.

        Raises:
            PlacementError: consecutive lanes do not join. ``route_lane_ids``
                records the lanes the ego drove *through*, which includes a
                lane CHANGE — and two parallel lanes stitched end-to-start
                produce a reference that jumps sideways at the seam. Use
                :meth:`lane_chain` (which walks real successors) instead.
        """
        if "__route_lanes__" in self._lane_cache:
            return self._lane_cache["__route_lanes__"]
        lane_ids = self.route_lane_ids()
        if not lane_ids:
            raise PlacementError(
                "host carries no route_lane_ids, so the traversed-lane chain cannot be "
                "stitched; name a lane explicitly with 'lane:<id>'"
            )
        points: List[np.ndarray] = []
        for lane_id in lane_ids:
            lane = self.lane(lane_id)
            xyz = np.concatenate([lane.xy, lane.z.reshape(-1, 1)], axis=1)
            if points:
                previous_end = points[-1][-1, :2]
                gaps = np.linalg.norm(xyz[:, :2] - previous_end, axis=1)
                start = int(np.argmin(gaps))
                if float(gaps[start]) > self.SEAM_GAP_TOL_M:
                    raise PlacementError(
                        f"route lanes {lane_ids} do not form a continuous reference: the seam "
                        f"into lane {lane_id} is {float(gaps[start]):.1f} m wide, which is a lane "
                        f"CHANGE rather than a successor. Use 'ego_lane_chain' or "
                        f"'lane_chain:<id>', which walk the map's own successor graph."
                    )
                xyz = xyz[start + 1 :]
                if xyz.shape[0] == 0:
                    continue
            points.append(xyz)
        line = Polyline.from_points(np.concatenate(points, axis=0), name="route_lanes")
        self._lane_cache["__route_lanes__"] = line
        return line

    def _lane_endpoints(self, lane_id: str):
        lane = self.lane(lane_id)
        start_tangent = lane.xy[1] - lane.xy[0]
        end_tangent = lane.xy[-1] - lane.xy[-2]
        return lane, start_tangent / max(np.linalg.norm(start_tangent), 1e-9), end_tangent / max(
            np.linalg.norm(end_tangent), 1e-9
        )

    def _neighbour(self, lane_id: str, *, forward: bool) -> Optional[str]:
        """Best genuine successor (or predecessor) of ``lane_id``, if any."""
        feat = self.map_features.get(str(lane_id)) or {}
        candidates = [str(x) for x in (feat.get(SD.EXIT if forward else SD.ENTRY) or [])]
        lane, start_tangent, end_tangent = self._lane_endpoints(lane_id)
        anchor_xy = lane.xy[-1] if forward else lane.xy[0]
        tangent = end_tangent if forward else start_tangent
        best, best_score = None, -np.inf
        for cand in candidates:
            if cand not in self.map_features:
                continue
            try:
                other, other_start, other_end = self._lane_endpoints(cand)
            except (PlacementError, ValueError):
                continue
            join_xy = other.xy[0] if forward else other.xy[-1]
            if float(np.linalg.norm(join_xy - anchor_xy)) > self.SEAM_GAP_TOL_M:
                continue
            # TWO TANGENTS, TWO JOBS. The candidate's SEAM tangent gates the
            # join (a branch that meets this lane at an angle is not a
            # continuation of it); its FAR tangent scores it, because that is
            # what says whether the branch is the same road or another one
            # leaving the junction. Junction connectors are drawn to meet
            # smoothly, so on 13c555e68671524f all three predecessors of the
            # lane the ego merges into scored 1.0000 at the seam and the
            # tie-break took 49696 -- the ego's OWN approach. V-10's mainline
            # car was then placed on the road the ego is turning off, 2.3 m
            # behind it and turning the same way, and rear-ended it on frame 0.
            # The through road is 53330, which comes in from the north-west at
            # 0.9998 against this lane's own heading and 0.31 for the ego's.
            seam = float(np.dot(tangent, other_start if forward else other_end))
            if seam < self.SUCCESSOR_ALIGN:
                continue
            align = float(np.dot(tangent, other_end if forward else other_start))
            if align <= best_score:
                continue
            best, best_score = cand, align
        return best

    def lane_chain(self, lane_id: str, *, min_length_m: float = 150.0) -> Polyline:
        """Walk the map's successor graph out from ``lane_id`` into one reference.

        A single lane centreline is 30-40 m in this data, which is shorter than
        an actor closing head-on for a whole episode needs. Walking real
        successors and predecessors extends it along the road the lane is part
        of, so the reference's own curvature still supplies the motion's shape
        and nothing is invented at a seam.
        """
        key = f"__chain__{lane_id}__{min_length_m}"
        if key in self._lane_cache:
            return self._lane_cache[key]
        chain: List[str] = [str(lane_id)]
        seen = {str(lane_id)}
        length = self.lane(lane_id).total
        while length < min_length_m:
            grew = False
            for forward in (True, False):
                if length >= min_length_m:
                    break
                nxt = self._neighbour(chain[-1] if forward else chain[0], forward=forward)
                if nxt is None or nxt in seen:
                    continue
                seen.add(nxt)
                (chain.append if forward else chain.insert)(*( (nxt,) if forward else (0, nxt)))
                length += self.lane(nxt).total
                grew = True
            if not grew:
                break
        points: List[np.ndarray] = []
        for member in chain:
            lane = self.lane(member)
            xyz = np.concatenate([lane.xy, lane.z.reshape(-1, 1)], axis=1)
            if points:
                gaps = np.linalg.norm(xyz[:, :2] - points[-1][-1, :2], axis=1)
                start = int(np.argmin(gaps))
                xyz = xyz[start + 1 :]
                if xyz.shape[0] == 0:
                    continue
            points.append(xyz)
        line = Polyline.from_points(
            np.concatenate(points, axis=0), name=f"lane_chain:{lane_id}"
        )
        logger.info(
            "lane_chain: %s -> %s (%.1f m over %d lanes)", lane_id, chain, line.total, len(chain)
        )
        self._lane_cache[key] = line
        return line

    def ego_lane_chain(self, after_frame: Optional[int] = None, *, min_length_m: float = 150.0):
        """The chain through the lane the ego occupies at the hand-off."""
        k = int(self.after_frame if after_frame is None else after_frame)
        k = int(np.clip(k, 0, self.T - 1))
        heading = float(
            np.arctan2(*(self.ego_position[min(k + 1, self.T - 1), :2] - self.ego_position[k, :2])[::-1])
        )
        lane_id, _ = self.nearest_lane(self.ego_position[k, :2], heading=heading)
        if lane_id is None:
            raise PlacementError(
                f"no lane found under the ego at frame {k}; name one with 'lane:<id>'"
            )
        return self.lane_chain(lane_id, min_length_m=min_length_m)

    # A road-geometry question ("is this two-way?", "does a lane merge in?")
    # must be asked on a stretch where the ego is going STRAIGHT. Asked inside a
    # junction it answers about the junction's connectors instead: the fan of
    # turn lanes reads as a converging lane, and another branch's lane reads as
    # the opposing carriageway. Both misreads shipped once (V-10 picked a
    # turn-fan; C-10 put its "oncoming" car on a different branch), and both
    # came from evaluating at the hand-off, which on many hosts sits in a
    # junction.
    STRAIGHT_WINDOW_M = 20.0
    STRAIGHT_MAX_DRIFT_RAD = float(np.deg2rad(15.0))

    def straight_window(
        self,
        *,
        after_frame: Optional[int] = None,
        length_m: Optional[float] = None,
        max_drift_rad: Optional[float] = None,
    ) -> Optional[tuple]:
        """First ``(start_arc, end_arc)`` past the hand-off where the ego route
        holds its heading — the junction-free stretch to measure road geometry on.

        Returns ``None`` when the whole route past the hand-off is turning; that
        is itself the finding (this host cannot answer a two-way / merge
        question, so it should not be qualified for an event type that asks one).
        """
        span = float(self.STRAIGHT_WINDOW_M if length_m is None else length_m)
        drift_max = float(self.STRAIGHT_MAX_DRIFT_RAD if max_drift_rad is None else max_drift_rad)
        k = int(self.after_frame if after_frame is None else after_frame)
        start = self.anchor_arc(self.ego_route, k)
        total = self.ego_route.total
        if total - start < span:
            return None
        for a0 in np.arange(start, total - span + 1e-9, 5.0):
            stations = np.arange(a0, a0 + span + 1e-9, 5.0)
            _, _, _, headings = self.ego_route.sample(stations)
            unwrapped = np.unwrap(np.asarray(headings, np.float64))
            if float(np.max(np.abs(unwrapped - unwrapped[0]))) > drift_max:
                continue
            # Extend while the heading still holds: a merge converges over
            # 40-60 m, so a window clipped to the minimum span would measure
            # only the approach and miss the merge itself.
            a1 = a0 + span
            while a1 + 5.0 <= total:
                _, _, _, h = self.ego_route.sample(np.arange(a0, a1 + 5.0 + 1e-9, 5.0))
                u = np.unwrap(np.asarray(h, np.float64))
                if float(np.max(np.abs(u - u[0]))) > drift_max:
                    break
                a1 += 5.0
            return (float(a0), float(a1))
        return None

    def opposing_lane(
        self,
        after_frame: Optional[int] = None,
        *,
        arc: Optional[float] = None,
        max_lateral_m: float = 9.0,
    ) -> Optional[dict]:
        """The nearest opposing-direction lane beside the ego's own road.

        This is what makes a road *two-way* — the precondition for the
        wrong-way event types (C-10, V-11) and the corridor half of C-7.

        Args:
            arc: where to measure, as an arc on the ego route. Defaults to the
                middle of :meth:`straight_window` (falling back to the hand-off
                only when the route never straightens), because a junction
                station answers about connectors rather than about the road.

        Returns:
            The nearest ``{lane_id, lateral_m, same_direction: False}`` entry,
            or ``None`` when every nearby parallel lane runs the ego's way.
        """
        if arc is None:
            window = self.straight_window(after_frame=after_frame)
            if window is not None:
                arc = 0.5 * (window[0] + window[1])
            else:
                k = int(self.after_frame if after_frame is None else after_frame)
                arc = self.anchor_arc(self.ego_route, k)
                logger.warning(
                    "opposing_lane: no straight stretch past the hand-off on this host; "
                    "measuring at the hand-off, which may sit inside a junction and pick "
                    "another branch's lane"
                )
        opposing = [
            lane
            for lane in self.parallel_lanes(
                self.ego_route, float(arc), max_lateral_m=max_lateral_m
            )
            if not lane["same_direction"]
        ]
        if not opposing:
            return None
        return min(opposing, key=lambda lane: abs(lane["lateral_m"]))

    def opposing_lane_chain(
        self, after_frame: Optional[int] = None, *, min_length_m: float = 150.0
    ) -> Polyline:
        """The chain through the opposing carriageway beside the hand-off.

        The chain runs in the OPPOSING lane's own travel direction, so an
        actor authored on it with positive ``speed`` is ordinary oncoming
        traffic — correctly oriented, driving its own lane. Two consequences
        for authoring against it:

        * its arc increases toward the ego and past it, so a spawn that is
          ego-forward ("60 m up the road, closing") is a **negative** ``arc``;
        * ``speed`` stays positive — a negative speed here would author a
          wrong-way driver on the *opposing* carriageway, i.e. a car in the
          ego's lane of travel facing away. Almost never what an event type means.
        """
        lane = self.opposing_lane(after_frame)
        if lane is None:
            raise PlacementError(
                "no opposing-direction lane runs beside the ego at the hand-off, so this "
                "host's road is one-way there. 'opposing_lane_chain' needs a two-way host — "
                "qualify hosts first (cli qualify), or name a lane explicitly with "
                "'lane_chain:<id>'."
            )
        return self.lane_chain(lane["lane_id"], min_length_m=min_length_m)

    def crossing_lane_chain(
        self, after_frame: Optional[int] = None, *, index: int = 0,
        min_length_m: float = 150.0
    ) -> Polyline:
        """The chain through a lane that CROSSES the ego's route ahead.

        This is R-2's side street — the road a cyclist rides in from — named by
        what it IS rather than by a map id. The per-host specs pinned
        ``lane_chain:52991``, which is correct for exactly one clip and silently
        wrong everywhere else.

        ``index`` selects among the crossing lanes at the first intersection
        ahead, ordered by how far along the route they cross: a four-way
        junction offers two, and an event type that wants both (R-2 puts a cyclist on
        each) asks for 0 and 1.
        """
        from navsafe.benchmark.editing.placement.anchors import _crossing_stations

        after = self.after_frame if after_frame is None else after_frame
        handoff_arc = float(self.ego_arc()[int(np.clip(after, 0, self.T - 1))])
        stations, crossing = _crossing_stations(self)
        seen: List[str] = []
        for arc, lane_ids in zip(stations, crossing):
            if arc < handoff_arc:
                continue
            for lane_id in lane_ids:
                if lane_id not in seen:
                    seen.append(lane_id)
            if len(seen) > index:
                break
        if len(seen) <= index:
            raise PlacementError(
                f"this host offers {len(seen)} lane(s) crossing the ego's route after the "
                f"hand-off, so crossing_lane_chain index {index} does not exist. The event type "
                f"needs a host with a crossing street — `navsafe mine` reports "
                f"`crossing_street` — or the actor belongs on 'ego_route'."
            )
        return self.lane_chain(seen[index], min_length_m=min_length_m)

    #: A cone/barrier run is a lane closure rather than litter at this many
    #: objects; matches the anchor detector, which reads the same tracks.
    WORKZONE_MIN_RUN = 5
    WORKZONE_MAX_LATERAL_M = 12.0
    #: Half a lane. Closer than this to the route and it is the ego's own lane,
    #: whatever the map calls it.
    NEIGHBOUR_LANE_MIN_M = 2.0
    #: How far past the merge to ask "which lane is the ego in now". At the
    #: merge point itself both roads are within a lane width of each other.
    MERGE_TARGET_LOOKAHEAD_M = 12.0

    def workzone_pinch(self, after_frame: Optional[int] = None, *,
                       reachable_s: float = 18.0, bin_m: float = 10.0,
                       tight_m: float = 4.0) -> Optional[dict]:
        """Where a work zone squeezes the ego's road hardest.

        nuPlan's lane graph does not know a lane is coned off, so a contraflow
        reads as an ordinary multi-lane road: `merging_lane` finds nothing and
        the host is refused as "no merge here" while the recorded ego is
        plainly threading a closure. The closure IS in the data, as one track
        per cone and per barrier.

        The pinch — the closest the clutter comes to the ego's line — is the
        point of maximum pressure, and that is what an actor should be timed
        against. The START of the run is not: on bbee1ab465af50c2 the cones
        begin 8 m BEHIND the hand-off and continue for 144 m, so "the taper the
        ego approaches" does not exist; the ego is inside the closure the whole
        clip and the question is who else wants the surviving lane.

        Returns:
            ``{arc_m, lateral_m, side, count}`` — ``arc_m`` from the hand-off,
            ``side`` +1 when the closure is to the ego's left — or ``None``.
        """
        route = self.ego_route
        after = self.after_frame if after_frame is None else after_frame
        start = self.anchor_arc(route, int(after))
        hits = []
        for tid, track in (self._sd.get(SD.TRACKS) or {}).items():
            if tid == self.sdc_id:
                continue
            if str(track.get("type", "")).upper() not in ("TRAFFIC_CONE", "TRAFFIC_BARRIER"):
                continue
            state = track.get("state", {})
            pos = np.asarray(state.get("position"), np.float64)
            if pos.ndim != 2 or len(pos) < 1:
                continue
            valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
            if not valid.any():
                continue
            arc, lateral = route.project(pos[valid][0, :2])
            if abs(lateral) > self.WORKZONE_MAX_LATERAL_M or not route.covers(arc, tol=2.0):
                continue
            hits.append((float(arc), float(lateral)))
        if len(hits) < self.WORKZONE_MIN_RUN:
            return None
        side = 1.0 if float(np.median([h[1] for h in hits])) > 0 else -1.0
        # Ahead of the hand-off, and inside what the ego can reach: clutter it
        # has already passed cannot put it under pressure, and a closure 130 m
        # up a road the ego covers 100 m of is not in this episode.
        reach = start + self.ego_cruise_speed(int(after)) * float(reachable_s)
        ahead = [h for h in hits if start <= h[0] <= reach]
        if len(ahead) < self.WORKZONE_MIN_RUN:
            return None
        # BINNED, not nearest-single. One cone lying in the carriageway — a
        # stray, or one the logged ego drove around — reads as a 0.04 m pinch
        # on 07846b829b3a575e and would put the whole scenario there. The
        # closure is the RUN, so each 10 m of route votes with its median and
        # the pinch is the first stretch that actually gets tight.
        edges = np.arange(start, reach + bin_m, bin_m)
        best = None
        for lo, hi in zip(edges[:-1], edges[1:]):
            lat = [abs(h[1]) for h in ahead if lo <= h[0] < hi]
            if len(lat) < 2:
                continue
            med = float(np.median(lat))
            if med <= tight_m:
                best = (0.5 * (lo + hi), med)
                break
            if best is None or med < best[1]:
                best = (0.5 * (lo + hi), med)
        if best is None:
            return None
        return {"arc_m": round(best[0] - start, 2),
                "lateral_m": round(side * best[1], 2),
                "side": side, "count": len(ahead)}

    def merging_lane(self, after_frame: Optional[int] = None) -> Optional[dict]:
        """The same-direction lane that converges into the ego's route.

        The physical merge signature: a *separate* lane whose lateral distance
        from the route falls from more than :data:`MERGE_FAR_M` to less than
        :data:`MERGE_NEAR_M` as arc advances.

        Measured across the WHOLE route past the hand-off, junctions included.
        It used to be limited to :meth:`straight_window` because the fan of turn
        connectors inside a junction converges on the route by construction, and
        a protected-turn clip once qualified as a merge on that. But the event type's
        own definition is broader than that guard allowed: a signalised right
        turn onto a main road IS the merge V-10 is about, and excluding
        junctions rejected 9 of the 11 hosts a reviewer had picked for it.

        What still keeps a plain turn from qualifying is the far-then-near
        profile below: a connector the ego itself drives along is never far and
        then near, it is near throughout. A lane that starts more than
        MERGE_FAR_M away and ends within MERGE_NEAR_M is a *separate*
        carriageway joining this one, which is the physical event either way.

        Each candidate is profiled only at stations whose projection lands
        INSIDE the lane: an endpoint-clamped projection measures distance to the
        lane's tip, which rises again past the merge point and would scramble
        the far-then-near ordering the test depends on.

        Returns:
            ``{lane_id, far_m, near_m, merge_arc_m}`` — ``merge_arc_m`` measured
            from the hand-off along the ego route — or ``None`` if no lane
            converges. Both the ``merge_convergence`` predicate and the
            ``merging_lane_chain`` reference read this one answer.
        """
        from navsafe.benchmark.editing.placement.anchors import _near_route_lane_ids

        route = self.ego_route
        after = self.after_frame if after_frame is None else after_frame
        start = self.anchor_arc(route, int(after))
        # The whole route past the hand-off, not just its junction-free part.
        window = (start, route.total)
        stations = np.arange(window[0], window[1] + 1e-9, MERGE_STATION_STEP_M)
        if len(stations) < 3:
            return None
        xs, ys, _, thetas = route.sample(stations)
        for lane_id in _near_route_lane_ids(self):
            try:
                lane = self.lane(lane_id)
            except Exception:  # noqa: BLE001 - a malformed lane is not a merge
                continue
            profile: List[Tuple[float, float]] = []
            for arc, x, y, theta in zip(stations, xs, ys, thetas):
                s_l, lateral = lane.project(np.array([float(x), float(y)]))
                if not lane.covers(s_l, tol=-0.5):  # endpoint clamp — see docstring
                    continue
                _, _, _, tangent = lane.sample([s_l])
                align = float(np.cos(float(tangent[0]) - float(theta)))
                # Direction is checked at the MERGE POINT, not at every station.
                # A lane joining from a side road is still turning while it is
                # far away, so a per-station same-direction filter threw away
                # exactly the "far" half of the profile and the far-then-near
                # test could never fire. What makes it a merge is that it ENDS
                # up running with the ego; how it got there is the turn.
                # Oncoming traffic is still excluded, by the alignment test on
                # the nearest station below.
                profile.append((float(arc), abs(float(lateral)), align))
            if len(profile) < 2:
                continue
            laterals = [lat for _, lat, _ in profile]
            far, near = max(laterals), min(laterals)
            i_near = laterals.index(near)
            # Same-direction WHERE IT MERGES. Oncoming traffic converges on the
            # route too — that is C-7's event, not V-10's — so the lane has to
            # be running with the ego by the time it arrives.
            if profile[i_near][2] < 0.7:
                continue
            if MERGE_FAR_M <= far <= MERGE_FAR_MAX_M and near <= MERGE_NEAR_M \
                    and i_near > laterals.index(far):
                return {
                    "lane_id": lane_id,
                    "far_m": round(far, 2),
                    "near_m": round(near, 2),
                    "merge_arc_m": round(profile[i_near][0] - start, 1),
                }
        return None

    def merging_lane_chain(
        self, after_frame: Optional[int] = None, *, min_length_m: float = 150.0
    ) -> Polyline:
        """The chain through the lane that merges with the ego's route.

        Runs in the ego's own direction, so an actor on it authors like ordinary
        traffic: positive ``speed``, and a spawn upstream of the merge point is
        a negative ``arc``. Because the chain meets the ego route at the merge,
        ``arrive_with_ego`` resolves its conflict point there without anyone
        having to measure it — which is the whole of an unsafe-merge leaf: the
        actor holds the lane the ego wants, and the ego has to yield or not.
        """
        merge = self.merging_lane(after_frame)
        if merge is None:
            raise PlacementError(
                "no same-direction lane converges with the ego's route past the hand-off, "
                f"so this host has no merge ({MERGE_FAR_M} -> {MERGE_NEAR_M} m is the test). "
                "'merging_lane_chain' needs a host `navsafe mine --event-type V-10` selected, or "
                "name the lane explicitly with 'lane_chain:<id>'."
            )
        return self.lane_chain(merge["lane_id"], min_length_m=min_length_m)

    def bike_lane(self, after_frame: Optional[int] = None,
                  *, max_lateral_m: float = 9.0) -> Optional[dict]:
        """The lane a bicycle belongs in beside the ego, and why it was picked.

        nuPlan's maps do carry bike lanes (lane type 1 in its own encoding,
        which py123d converts to ``LANE_BIKE_LANE``), but most streets have
        none. So this asks the event type's question — "where would a cyclist
        legitimately be here?" — rather than a map question, and answers it in
        two steps: a mapped bike lane if the host has one running the ego's way,
        otherwise the EGO'S OWN carriageway lane, which is where a cyclist
        rides on a street without one and the only place the ego has to pass
        them.

        Returns:
            ``{lane_id, lateral_m, dedicated}`` — ``dedicated`` False when the
            answer is the ego's own lane — or ``None`` if no same-direction
            lane runs beside the ego at all.
        """
        # AT THE HAND-OFF, which is where the riders go. This asked the
        # question at the MIDDLE of the host's straight stretch instead --
        # 67 m up the road on 12f96c65436e56bf, 97 m on 04cdd9195f885ac6 --
        # and the lane beside the ego there is not the lane beside the ego
        # here. The riders came out one lane over (measured -3 to -9 m off the
        # ego's line across the event type) or, where the chain started downstream,
        # 40-100 m ahead of it. The straight stretch is what the LAYOUT wants;
        # which lane the ego is in is a property of where the ego is.
        k = int(self.after_frame if after_frame is None else after_frame)
        arc = self.anchor_arc(self.ego_route, k)
        # `__logged_drivable_support_*` is a synthesised stand-in for drivable
        # area, laid along the ego's OWN track where the map has no lane. It
        # therefore sits at lateral 0.00 and is picked as "the rightmost lane"
        # on every host that has one -- and it has no carriageway either side,
        # so the kerbside lateral the event type asks for resolves against nothing.
        # On 048eb7efa08354e3 that is what put the two riders 1.1 m apart on
        # different lines, one of them past the kerb. Same exclusion as
        # `closure_lane` and `merge_target_lane_chain` make.
        same = [lane for lane in self.parallel_lanes(
            self.ego_route, float(arc), max_lateral_m=max_lateral_m)
            if lane["same_direction"]
            and not str(lane["lane_id"]).startswith("__")]
        if not same:
            return None
        # THE EGO'S OWN LANE, not the rightmost one in the map. "Rightmost"
        # is the right answer to "where does a cyclist ride on this street"
        # and the wrong answer to the event type's question, which is whether THIS
        # ego has room to pass: on a road with a parking lane or a second
        # carriageway the rightmost same-direction lane is 3-8 m off the ego's
        # line, so the riders sit in traffic the ego never meets (measured
        # across all ten R-2 hosts: -2.5 to -7.8 m, and on 02f1ad081f41550e
        # that lane is live, so logged cars drove through the riders). Nearest
        # to the ego's line, with the event type's kerbside `lateral` putting them
        # against its right edge, is the overtake the event type is named for.
        nearest = min(same, key=lambda ln: abs(ln["lateral_m"]))
        # A MAPPED BIKE LANE ONLY IF IT IS THAT LANE. Preferring one wherever
        # the host had it is where a cyclist belongs and not what this event type
        # tests: on 12f96c65436e56bf the mapped bike lane is 3.9 m off the
        # ego's line, beyond a parking lane, so the riders sat in a lane the
        # ego never enters and there was nothing to overtake. Sharing the
        # carriageway is the event; `dedicated` still records when the lane
        # they share happens to be painted for them.
        return {"lane_id": nearest["lane_id"], "lateral_m": nearest["lateral_m"],
                "dedicated": _BIKE_LANE_HINT in str(
                    self.map_features[nearest["lane_id"]].get(SD.TYPE, "")).upper()}

    def closure_lane(self, after_frame: Optional[int] = None,
                     *, max_lateral_m: float = 12.0) -> Optional[dict]:
        """The same-direction lane the work zone is closing, if there is one.

        The cones sit IN the lane that is being taken away, so the lane nearest
        them on their own side is the one whose traffic has to merge out — into
        the ego's. That is where an actor goes to put the ego under merge
        pressure: not beside it as ordinary traffic, but in a lane that runs
        out.

        Returns ``{lane_id, lateral_m, pinch_arc_m}`` or ``None`` when this
        host has no closure (then the merge, if any, is a map merge and
        :meth:`merging_lane` answers instead).
        """
        pinch = self.workzone_pinch(after_frame)
        if pinch is None:
            return None
        after = self.after_frame if after_frame is None else after_frame
        arc = self.anchor_arc(self.ego_route, int(after)) + float(pinch["arc_m"])
        side = float(pinch["side"])
        # A NEIGHBOURING lane, so two exclusions. `__logged_drivable_support_*`
        # is a synthesised polyline standing in for drivable area where the map
        # has no lane, not a lane traffic uses; and anything within half a lane
        # of the route IS the ego's own lane, so an actor placed there is
        # placed on top of the ego. Both were selected before this guard.
        parallel = [ln for ln in self.parallel_lanes(
            self.ego_route, float(arc), max_lateral_m=max_lateral_m)
            if ln["same_direction"]
            and not str(ln["lane_id"]).startswith("__")
            and abs(ln["lateral_m"]) >= self.NEIGHBOUR_LANE_MIN_M]
        if not parallel:
            return None
        # Nearest to the ego on the closure's side: the lane between the ego
        # and the cones is the one that ends, so its traffic is what has to
        # come across.
        closing = [ln for ln in parallel if ln["lateral_m"] * side > 0]
        if closing:
            lane = min(closing, key=lambda ln: abs(ln["lateral_m"]))
            return {"lane_id": lane["lane_id"], "lateral_m": lane["lateral_m"],
                    "pinch_arc_m": float(pinch["arc_m"]), "side": "closed_lane"}
        # Nothing on that side: the closure is eating the EGO'S OWN lane, as on
        # be36f75d360c502f where the barrier runs 2.3 m off its line with no
        # lane beyond it. Then the pressure is the other way round — the ego is
        # squeezed between the barrier and the traffic beside it — so the actor
        # goes in the lane on the open side and the ego has nowhere to go.
        lane = min(parallel, key=lambda ln: abs(ln["lateral_m"]))
        return {"lane_id": lane["lane_id"], "lateral_m": lane["lateral_m"],
                "pinch_arc_m": float(pinch["arc_m"]), "side": "open_side"}

    def merge_pressure_lane(self, after_frame: Optional[int] = None) -> Optional[dict]:
        """The lane whose traffic has to take the ego's, whatever causes it.

        Two causes, one question. A MAP merge is two carriageways drawn as
        joining; a WORK-ZONE merge is a lane the map still shows as open and
        the log has coned off. V-10 is the same test either way -- somebody
        needs the strip of road the ego is on -- so the event type gates and places
        on this, and the ``cause`` it returns is recorded in the recipe so the
        two can still be told apart when the scores are read.
        """
        merge = self.merging_lane(after_frame)
        if merge is not None:
            return {"cause": "map_merge", "lane_id": merge["lane_id"],
                    "conflict_arc_m": merge["merge_arc_m"], "detail": merge}
        closure = self.closure_lane(after_frame)
        if closure is not None:
            return {"cause": "workzone", "lane_id": closure["lane_id"],
                    "conflict_arc_m": closure["pinch_arc_m"], "detail": closure}
        return None

    #: How much the ego's own heading has to swing across the merge before it
    #: is the one joining rather than the one being joined. A slip road onto a
    #: main road turns; a main road through a junction does not.
    MERGE_EGO_TURN_DEG = 25.0
    #: Over how much arc that swing is measured.
    MERGE_TURN_SPAN_M = 20.0

    def ego_turns_into_merge(self, after_frame: Optional[int] = None) -> bool:
        """Does the EGO turn through the merge point?

        The distinction the event type needs: a turning ego is joining a road, and
        the conflict is the traffic already on it; a straight ego is being
        joined, and the conflict is the traffic coming across.
        """
        found = self.merge_pressure_lane(after_frame)
        if found is None:
            return False
        after = self.after_frame if after_frame is None else after_frame
        arc = (self.anchor_arc(self.ego_route, int(after))
               + float(found["conflict_arc_m"]))
        # Measured over the APPROACH, not just the merge point: a slip road
        # curves in over 20-40 m, so a window one lookahead wide catches only
        # part of it. On 49a0d29c7058501c the swing is 21.9 deg over +/-12 m
        # and 58.5 over +/-20 -- the same turn, and only the wider window sees
        # it for what it is.
        span = self.MERGE_TURN_SPAN_M
        lo = max(arc - span, 0.0)
        hi = min(arc + span, self.ego_route.total)
        _, _, _, th = self.ego_route.sample([lo, hi])
        swing = abs(math.degrees(math.atan2(
            math.sin(float(th[1]) - float(th[0])),
            math.cos(float(th[1]) - float(th[0])))))
        return swing >= self.MERGE_EGO_TURN_DEG

    def merge_target_lane_chain(
        self, after_frame: Optional[int] = None, *, min_length_m: float = 150.0
    ) -> Polyline:
        """The road the EGO merges into, chained back the way its traffic comes.

        `merging_lane` finds the FEEDER -- the lane that joins the ego's route
        -- and putting the actor there stages "somebody merges into the ego".
        When the ego is the one joining, the event type wants the opposite: a car
        already travelling the road the ego is trying to enter, so the ego's
        own merge is the unsafe act. That is the lane the ego's route runs
        along just PAST the merge point, chained upstream.

        Refuses when the map has no real lane there. Some junctions are covered
        only by `__logged_drivable_support_*`, a synthesised stand-in for
        drivable area with no connectivity, and a chain built from it runs
        wherever the ego drove rather than down the other road.
        """
        found = self.merge_pressure_lane(after_frame)
        if found is None:
            raise PlacementError(
                "no merge on this host, so there is no road for the ego to merge INTO "
                "('merge_target_lane_chain'). Pick a host `navsafe qualify` passes for V-10."
            )
        after = self.after_frame if after_frame is None else after_frame
        arc = (self.anchor_arc(self.ego_route, int(after))
               + float(found["conflict_arc_m"]) + self.MERGE_TARGET_LOOKAHEAD_M)
        # NOT `nearest_lane`: `__logged_drivable_support_*` is a synthesised
        # stand-in for drivable area laid ALONG the ego's own track, so it sits
        # at lateral 0.00 and wins every nearest-lane query on a road the map
        # covers perfectly well. On 13c555e68671524f it hid lane 52241, which
        # is 0.1 m from the route and is the road the ego is merging onto.
        candidates = [ln for ln in self.parallel_lanes(
            self.ego_route, float(arc), max_lateral_m=6.0)
            if ln["same_direction"] and not str(ln["lane_id"]).startswith("__")]
        if not candidates:
            raise PlacementError(
                "the road the ego merges into is not in the lane graph here. A car cannot "
                "be placed on a lane that does not exist; this host needs the actor "
                "re-tasked from logged traffic, or a different host."
            )
        lane_id = min(candidates, key=lambda ln: abs(ln["lateral_m"]))["lane_id"]
        return self.lane_chain(lane_id, min_length_m=min_length_m)

    def merge_pressure_lane_chain(
        self, after_frame: Optional[int] = None, *, min_length_m: float = 150.0
    ) -> Polyline:
        """The chain an actor drives to put the ego under merge pressure."""
        # WHICH SIDE OF THE MERGE IS THE EGO ON? Both readings are "two roads
        # become one", and they need the actor in opposite places. When the
        # EGO turns through the merge it is the one joining, so the car the
        # event type is about is already on the road it wants -- put the actor there.
        # When the ego holds its line, the other road is the feeder and its
        # traffic comes across into the ego's.
        if self.ego_turns_into_merge(after_frame):
            try:
                return self.merge_target_lane_chain(
                    after_frame, min_length_m=min_length_m)
            except PlacementError as exc:
                logger.info("merge target unusable (%s); placing on the feeder instead", exc)
        found = self.merge_pressure_lane(after_frame)
        if found is None:
            raise PlacementError(
                "no lane converges with the ego's route and no work zone closes one beside "
                f"it ({MERGE_FAR_M} -> {MERGE_NEAR_M} m is the map test; a closure needs "
                f"{self.WORKZONE_MIN_RUN} cones/barriers within "
                f"{self.WORKZONE_MAX_LATERAL_M:.0f} m). This host stages no merge of either "
                "kind — pick one that does (navsafe qualify), or name the lane explicitly "
                "with 'lane_chain:<id>'."
            )
        return self.lane_chain(found["lane_id"], min_length_m=min_length_m)

    def crosswalks_on_route(self, after_frame: Optional[int] = None,
                            *, max_lateral_m: float = 12.0,
                            min_angle_deg: float = CROSSWALK_MIN_ANGLE_DEG,
                            reaction_s: float = 1.5) -> List[dict]:
        """Marked crossings the ego drives over, in route order.

        A crosswalk feature's ``polyline`` is its long axis — kerb to kerb —
        so this is the line an actor crosses ON, not merely a place along the
        route. That distinction is the whole point: an actor sent across the
        ego's route perpendicular to it crosses wherever the layout put it,
        which is not where people cross.

        Returns:
            One entry per crossing ahead of the hand-off:
            ``{feature_id, arc_m, lateral_m, length_m, angle_deg}`` — ``arc_m``
            from the hand-off, ``angle_deg`` the crossing's angle to the route
            (90 deg is square across it). Crossings shallower than
            ``min_angle_deg`` are dropped: see
            :data:`CROSSWALK_MIN_ANGLE_DEG`.

            Crossings the ego reaches within ``reaction_s`` of the hand-off are
            dropped too, measured at the EGO'S OWN SPEED rather than a fixed
            distance. This host's first crossing is 3.6 m past the hand-off:
            an actor triggered on the ego's approach has no approach to be
            triggered by, and the episode is over before the crossing starts.
            The threshold is a distance because the ego's speed is what makes
            3.6 m either "already there" or "a comfortable pause".
        """
        route = self.ego_route
        after = self.after_frame if after_frame is None else after_frame
        k = int(np.clip(after, 0, self.T - 1))
        handoff = self.anchor_arc(route, k)
        # The ego's own pace past the hand-off. NOT the one-frame difference
        # that stood here: on 02379e524f105926 the ego happens to be slow at
        # the hand-off frame (1.65 m/s against a 6.96 m/s cruise), so a 1.5 s
        # reaction window shrank to 2.5 m and a crossing 3.6 m ahead passed the
        # filter. R-3 then put four walkers in front of an ego that reached
        # them 0.9 s after taking over — the episode ended at frame 17 with too
        # few frames to score. Same defect as the one that made `heading_at`
        # read positioning noise: one frame of a logged track is not a rate.
        ego_speed = self.ego_cruise_speed(k)
        min_arc = float(reaction_s) * ego_speed
        out: List[dict] = []
        for key, feat in self.map_features.items():
            if _CROSSWALK_HINT not in str(feat.get(SD.TYPE, "")).upper():
                continue
            pts = feat.get(SD.POLYLINE)
            if pts is None:
                continue
            pts = np.asarray(pts, np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue
            centre = pts[:, :2].mean(axis=0)
            arc, lateral = route.project(centre)
            if abs(lateral) > max_lateral_m or not route.covers(arc, tol=1.0):
                continue
            # IT HAS TO CROSS THE EGO'S LINE, not merely lie near it. A
            # junction's other arm carries crossings whose centres are within
            # `max_lateral_m` and which the ego never drives over: on
            # 225eb6e22af55972 the nearest one by arc runs from -16.6 m to
            # -6.8 m, entirely to the ego's left. It was picked, R-3's walkers
            # were sent along it, and `arrive_with_ego` correctly reported that
            # their path never meets the ego's route -- an accurate message
            # about the wrong crossing, on a host that has three good ones.
            ends = np.asarray([route.project(pts[0, :2])[1],
                               route.project(pts[-1, :2])[1]], np.float64)
            if ends[0] * ends[1] > 0.0:
                continue
            if arc - handoff < min_arc:
                # Behind the ego, or so close ahead that it arrives before an
                # actor triggered on its approach could have moved.
                continue
            span = pts[-1, :2] - pts[0, :2]
            _, _, _, heading = route.sample([arc])
            angle = abs(np.degrees(np.arctan2(
                np.sin(np.arctan2(span[1], span[0]) - float(heading[0])),
                np.cos(np.arctan2(span[1], span[0]) - float(heading[0])))))
            if min(angle, 180.0 - angle) < float(min_angle_deg):
                continue                       # the side street's crossing
            out.append({
                "feature_id": key,
                "arc_m": round(float(arc - handoff), 1),
                "lateral_m": round(float(lateral), 2),
                "length_m": round(float(np.linalg.norm(span)), 1),
                "angle_deg": round(float(min(angle, 180.0 - angle)), 0),
            })
        out.sort(key=lambda d: d["arc_m"])
        return out

    def crosswalk_chain(self, index: int = 0, after_frame: Optional[int] = None) -> Polyline:
        """The ``index``-th crossing ahead of the hand-off, as a walkable line.

        Falls back to a MID-BLOCK crossing when the host carries no marked one.
        R-3 and R-4 are about something entering the road in front of the ego;
        a painted crossing is where that most often happens, not a precondition.
        Requiring one rejected four reviewer-picked hosts outright, so a host
        without paint gets a line built perpendicular to the route instead —
        the same walk, at a place the ego actually drives through.
        """
        found = self.crosswalks_on_route(after_frame)
        if len(found) <= index:
            return self._midblock_crossing(index, after_frame)
        key = found[index]["feature_id"]
        pts = np.asarray(self.map_features[key][SD.POLYLINE], np.float64)
        if pts.shape[1] < 3 or float(np.max(np.abs(pts[:, 2]))) < 1e-6:
            z = self._road_z_from_ego(pts[:, :2])
            pts = np.concatenate([pts[:, :2], z.reshape(-1, 1)], axis=1)
        return Polyline.from_points(pts, name=f"crosswalk_chain:{key}")

    #: How far past the hand-off a fabricated crossing is placed, and how wide
    #: it is built. The arc gives the ego room to react (a crossing 2 m ahead is
    #: not a scenario); the half-width spans a typical two-way carriageway plus
    #: verge, so the walk starts and ends off the road.
    MIDBLOCK_ARC_M = 35.0
    MIDBLOCK_HALF_WIDTH_M = 9.0

    def _midblock_crossing(self, index: int = 0,
                           after_frame: Optional[int] = None) -> Polyline:
        """A crossing line perpendicular to the ego route, for a host with no paint.

        Built rather than found, so it is named `midblock_crossing` and never
        mistaken for a mapped feature. Successive indices step further along the
        route, so an event type asking for `crosswalk_chain:2` still gets two distinct
        places rather than the same line twice.
        """
        route = self.ego_route
        after = self.after_frame if after_frame is None else after_frame
        start = self.anchor_arc(route, int(after))
        arc = start + self.MIDBLOCK_ARC_M * (index + 1)
        if arc >= route.total:
            raise PlacementError(
                f"a mid-block crossing {self.MIDBLOCK_ARC_M * (index + 1):.0f} m past the "
                f"hand-off runs off the end of this host's {route.total:.0f} m route. The "
                f"host is too short for the crossing this event type wants.")
        xs, ys, _, thetas = route.sample(np.asarray([arc]))
        cx, cy, th = float(xs[0]), float(ys[0]), float(thetas[0])
        # Perpendicular to the route: +left of travel to -right of it.
        nx, ny = -math.sin(th), math.cos(th)
        h = self.MIDBLOCK_HALF_WIDTH_M
        xy = np.asarray([[cx + nx * h, cy + ny * h], [cx - nx * h, cy - ny * h]], np.float64)
        z = self._road_z_from_ego(xy)
        pts = np.concatenate([xy, z.reshape(-1, 1)], axis=1)
        logger.info("crosswalk_chain: host has no marked crossing; built a mid-block one "
                    "%.0f m past the hand-off, %.0f m wide", arc - start, 2 * h)
        return Polyline.from_points(pts, name="midblock_crossing")

    def bike_lane_chain(
        self, after_frame: Optional[int] = None, *, min_length_m: float = 150.0
    ) -> Polyline:
        """The chain through :meth:`bike_lane`, in the ego's own direction."""
        lane = self.bike_lane(after_frame)
        if lane is None:
            raise PlacementError(
                "no same-direction lane runs beside the ego at the hand-off, so there is "
                "nowhere on this host a cyclist would legitimately be. 'bike_lane_chain' "
                "needs a host with a carriageway — `navsafe mine --event-type R-2` reports "
                "`bike_permitted_lane`."
            )
        return self.lane_chain(lane["lane_id"], min_length_m=min_length_m)

    def nearest_lane(self, point, *, heading: Optional[float] = None, max_dist_m: float = 6.0):
        """Lane whose centreline is closest to ``point`` (optionally aligned).

        Returns:
            ``(lane_id, lateral_m)`` or ``(None, None)`` when nothing qualifies.
            Intersections stack overlapping lanes that share XY, so a heading
            gate is the difference between picking the lane an actor is on and
            picking a crossing connector.
        """
        best = (None, None, np.inf)
        for lane_id in self.lane_ids():
            try:
                line = self.lane(lane_id)
            except PlacementError:
                continue
            s, lateral = line.project(point)
            if abs(lateral) > max_dist_m or abs(lateral) >= best[2]:
                continue
            if heading is not None:
                _, _, _, tangent = line.sample(s)
                if abs(np.cos(float(tangent[0]) - float(heading))) < 0.5:  # > 60 deg off
                    continue
            best = (lane_id, lateral, abs(lateral))
        return best[0], best[1]

    def resolve_reference(self, reference: Any) -> Polyline:
        """Turn a recipe's ``reference`` field into a polyline.

        Accepts ``"ego_route"``, ``"lane:<id>"``, or a mapping/sequence holding
        an inline polyline.
        """
        return resolve_reference(self, reference)

    def ego_cruise_speed(self, after_frame: Optional[int] = None) -> float:
        """The ego's own pace past the hand-off, in m/s.

        An event type that stages an OVERTAKE has to know this: "a realistic urban
        cyclist does 15-22 km/h" is true and still wrong here, because a nuPlan
        city log's ego averages about 18 km/h itself. Two cyclists authored at
        an absolute 5.0 / 4.4 m/s therefore rode away from a 5.3 m/s ego on
        every host in the R-2 set, and the ego finished the episode never
        having seen them.

        Measured as the MEDIAN per-frame speed rather than one frame's
        difference (which is positioning noise) or the mean (which a stop at a
        light drags to zero).
        """
        k = int(np.clip(self.after_frame if after_frame is None else after_frame,
                        0, self.T - 1))
        pos = np.asarray(self.ego_position)[k:, :2]
        if len(pos) < 2:
            return 0.0
        return float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1))
                     / max(self.dt_s, 1e-6))

    def anchor_arc(self, reference: Polyline, after_frame: int) -> float:
        """Arc 0 for authored ``arc`` values: the hand-off, on this reference.

        ``arc`` parameters are measured from the hand-off, while the state
        arrays index scenario frames — two different origins, and both are
        recorded in the recipe. Projecting the ego's hand-off pose onto the
        reference makes the same convention work for a lane centreline or an
        inline polyline, and reduces exactly to ``arc[after_frame]`` on the ego
        route itself.
        """
        k = int(np.clip(after_frame, 0, self.T - 1))
        return float(reference.project(self.ego_position[k, :2])[0])

    #: How far a logged vehicle can be from a query point and still say what
    #: the road is under it. A city block's worth of carriageway is flat to
    #: within a few centimetres; beyond that a real gradient starts to show.
    ROAD_Z_RADIUS_M = 30.0

    def _road_samples(self):
        """``(xy, road_z)`` for every logged vehicle with a real box.

        A vehicle's box centre sits half its height above the road it stands
        on, so each one MEASURES the road where it is. That is worth more than
        the ego pose: the ego z is documented as road-referenced, but that was
        established on WOD, and on these nuPlan clips it wanders — on
        5d12ad55fdd858e1 it falls 3.1 m over 209 m of demonstrably flat road,
        which is what buried an inserted car 1.7 m into the tarmac.
        """
        if getattr(self, "_road_sample_cache", None) is not None:
            return self._road_sample_cache
        xy, z = [], []
        for tid, track in (self._sd.get(SD.TRACKS) or {}).items():
            if tid == self.sdc_id or str(track.get("type", "")).upper() != "VEHICLE":
                continue
            state = track.get("state", {})
            pos = np.asarray(state.get("position"), np.float64)
            h = np.asarray(state.get("height", [0.0]), np.float64).reshape(-1)
            if pos.ndim != 2 or len(pos) < 1 or float(h[0]) <= 0.1:
                continue
            valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
            if not valid.any():
                continue
            k = int(np.flatnonzero(valid)[0])
            xy.append(pos[k, :2])
            z.append(float(pos[k, 2]) - 0.5 * float(h[0]))
        self._road_sample_cache = (
            (np.asarray(xy, np.float64), np.asarray(z, np.float64))
            if xy else (np.zeros((0, 2)), np.zeros(0)))
        return self._road_sample_cache

    def _road_z_from_ego(self, xy: np.ndarray) -> np.ndarray:
        """Road-surface height under each point.

        Measured from the logged vehicles standing on it where there are any
        within :data:`ROAD_Z_RADIUS_M`, and taken from the ego's route where
        there are not. The ego route is the fallback rather than the source
        because its z is a POSE height that drifts along a clip; the boxes do
        not drift, they are observations of the road itself.
        """
        pts = np.asarray(xy, np.float64)
        arcs = [self.ego_route.project(pt)[0] for pt in pts]
        from_ego = np.asarray(
            self.ego_route.road_z(arcs, ego_z_to_ground_m=self.ego_z_to_ground_m), np.float64
        )
        sxy, sz = self._road_samples()
        if len(sz) < 2:
            return from_ego
        out = from_ego.copy()
        for i, pt in enumerate(pts):
            d = np.linalg.norm(sxy - pt[None, :2], axis=1)
            near = sz[d <= self.ROAD_Z_RADIUS_M]
            if len(near) >= 2:
                out[i] = float(np.median(near))
        return out

    def measure_road_drift(self, *, near_ego_m: float = 30.0) -> Optional[dict]:
        """Check the host's road height against the ego z, from the log itself.

        Since ``_extract_ego`` takes the ego's z from the ground reference
        (rear axle / IMU) rather than the bbox centre, **the ego z already means
        the road under the ego** and the correct drop is 0. What this measures
        is the RESIDUAL: actor perception boxes are ground-referenced with z at
        the box centre, so ``centre - height/2`` is the road under each actor,
        and it should agree with the ego's z.

        A residual much beyond the documented ±0.4 m camera-only recon drift
        means one of two things, and they need different fixes:

        * the reconstruction's road drifts from the real one — calibrate it per
          scene in the ``NUREC_GROUND_Z_CALIB`` registry;
        * this source's rear-axle pose does not sit on the road. ``rear_axle_se3
          ≈ imu ≈ on the road`` was established for WOD; a different source can
          put the axle half a metre up, and then every host from it shares one
          constant offset.

        ``if_box_is_base`` is the same measurement under the other reading of
        the actor box z, because the two differ by half a vehicle height and
        that is the same order as the residual itself.

        Returns:
            ``{"samples", "ego_z_median", "actor_road_median", "residual_m",
            "if_box_is_base_m"}``, or ``None`` when no actor carries usable
            dimensions.
        """
        centres: List[float] = []
        heights: List[float] = []
        ego_zs: List[float] = []
        for tid, track in (self._sd.get(SD.TRACKS) or {}).items():
            if tid == self.sdc_id:
                continue
            state = track.get("state", {})
            pos = np.asarray(state.get("position"), np.float64)
            height = np.asarray(state.get("height", []), np.float64).reshape(-1)
            if pos.ndim != 2 or height.size == 0 or float(np.max(height)) <= 0.1:
                continue
            valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
            n = min(len(pos), self.T, len(height))
            near = (
                np.linalg.norm(pos[:n, :2] - self.ego_position[:n, :2], axis=1) <= near_ego_m
            ) & valid[:n] & (height[:n] > 0.1)
            if not np.any(near):
                continue
            centres.extend(pos[:n, 2][near].tolist())
            heights.extend(height[:n][near].tolist())
            ego_zs.extend(self.ego_position[:n, 2][near].tolist())
        if not centres:
            return None
        centre = np.asarray(centres)
        height = np.asarray(heights)
        ego_z = float(np.median(ego_zs))
        actor_road = float(np.median(centre - 0.5 * height))
        return {
            "samples": len(centres),
            "ego_z_median": round(ego_z, 3),
            "actor_road_median": round(actor_road, 3),
            "residual_m": round(ego_z - actor_road, 3),
            "if_box_is_base_m": round(ego_z - float(np.median(centre)), 3),
        }

    # -- picking an actor to re-task --------------------------------------
    def vehicles_over_arc(
        self,
        arc_from: float,
        arc_to: float,
        *,
        corridor_m: float = 4.5,
        track_type: str = "VEHICLE",
    ) -> List[dict]:
        """Logged vehicles that occupy the ego's own path over ``[from, to]``.

        A STATIC insert has no way to defend itself: the logged traffic ahead
        of the ego was recorded on a road that had no incident on it, so it
        drives straight through an inserted wreck — and while it is doing that
        it also SITS IN FRONT OF IT, so the ego first sees the obstruction when
        the phantom lead car has already passed over it. The scenario then
        tests a surprise rather than a decision.

        The fix is to take those vehicles out of the scenario, which is what
        this finds: any track whose recorded path enters the ego's corridor
        anywhere in the arc window, at any frame. "Enters at any frame" rather
        than "is there at the hand-off" is deliberate — a car 60 m ahead at
        frame 0 is still the car that will be standing on the incident when the
        ego arrives.

        Returns:
            ``[{"track_id", "arc_m", "lateral_m", "frame"}]`` at each track's
            closest approach to the window, nearest first.
        """
        route = self.ego_route
        lo, hi = (min(arc_from, arc_to), max(arc_from, arc_to))
        found: List[dict] = []
        for tid, track in (self._sd.get(SD.TRACKS) or {}).items():
            if tid == self.sdc_id or str(track.get("type", "")) != track_type:
                continue
            state = track.get("state", {})
            pos = np.asarray(state.get("position"), np.float64)
            if pos.ndim != 2 or pos.shape[0] < 1:
                continue
            valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
            best = None
            for k in np.flatnonzero(valid):
                arc, lateral = route.project(pos[k, :2])
                if not (lo <= arc <= hi) or abs(lateral) > corridor_m:
                    continue
                if best is None or abs(lateral) < abs(best["lateral_m"]):
                    best = {"track_id": str(tid), "arc_m": round(float(arc), 2),
                            "lateral_m": round(float(lateral), 2), "frame": int(k)}
            if best is not None:
                found.append(best)
        found.sort(key=lambda d: d["arc_m"])
        return found

    def select_relocation_target(
        self,
        *,
        track_type: str = "VEHICLE",
        length_range_m: tuple = (3.0, 6.5),
        min_travel_m: float = 5.0,
        min_valid_fraction: float = 1.0,
    ) -> Optional[dict]:
        """Pick the host actor best suited to being re-tasked.

        Relocating beats inserting because the actor keeps its baked gaussians,
        but only some of a host's tracks are worth re-tasking. Wanted, in order:

        * **real dimensions** — a great many tracks carry ``[0, 0, 0]`` boxes,
          and an actor with no box has no usable appearance either;
        * **valid for the whole episode** — a track that blinks out mid-way
          cannot carry a scenario that runs to the end;
        * **a car that actually drove** — a parked car is reconstructed from one
          pose and smears badly the moment it is moved, while a moving one was
          seen from many angles.

        Returns:
            ``{"track_id", "dims", "travelled_m", "valid_frames", "reason"}`` for
            the best candidate, or ``None`` when the host offers none.
        """
        best = None
        for tid, track in (self._sd.get(SD.TRACKS) or {}).items():
            if tid == self.sdc_id or str(track.get("type", "")) != track_type:
                continue
            state = track.get("state", {})
            pos = np.asarray(state.get("position"), np.float64)
            if pos.ndim != 2 or pos.shape[0] < 2:
                continue
            dims = [
                float(np.asarray(state.get(k, [0.0])).reshape(-1)[0])
                for k in ("length", "width", "height")
            ]
            if not (length_range_m[0] <= dims[0] <= length_range_m[1]) or dims[1] <= 0:
                continue
            valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
            fraction = float(valid.sum()) / max(1, min(len(valid), self.T))
            if fraction < min_valid_fraction:
                continue
            travelled = float(np.linalg.norm(pos[valid][-1, :2] - pos[valid][0, :2]))
            if travelled < min_travel_m:
                continue
            score = (fraction, travelled)
            if best is None or score > best["_score"]:
                best = {
                    "track_id": str(tid),
                    "dims": [round(d, 2) for d in dims],
                    "travelled_m": round(travelled, 1),
                    "valid_frames": int(valid.sum()),
                    "_score": score,
                }
        if best is None:
            return None
        best.pop("_score")
        best["reason"] = (
            f"{track_type} with real dims, valid {best['valid_frames']}/{self.T} frames, "
            f"drove {best['travelled_m']} m"
        )
        return best

    # -- the log-replay ego ----------------------------------------------
    def ego_arc(self) -> np.ndarray:
        """Arc length along the ego route at each scenario frame, (T,)."""
        return np.array(
            [self.ego_route.project(self.ego_position[k, :2])[0] for k in range(self.T)],
            np.float64,
        )

    def ego_arrival_frame(self, point) -> int:
        """First frame at which the LOG-REPLAY ego reaches ``point``.

        Under a non-reactive design a timing solve needs an assumed ego
        arrival, and this is it. The policy under test may drive faster or
        slower, which is exactly why the recipe records the window it assumed —
        so a mistimed episode is detectable rather than silently scored.
        """
        target, _ = self.ego_route.project(point)
        arcs = self.ego_arc()
        ahead = np.flatnonzero(arcs >= target)
        return int(ahead[0]) if ahead.size else int(self.T - 1)

    def ego_arrival_window_s(
        self, positions: Sequence, *, event_radius_m: float = 15.0
    ) -> Optional[List[float]]:
        """Seconds during which the log-replay ego is within reach of an actor.

        Returns ``None`` when the actor never comes close, which is itself the
        finding: the scenario-defining event never meets the ego.
        """
        best: Optional[List[float]] = None
        for pos in positions:
            arr = np.asarray(pos, np.float64)[:, :2]
            n = min(arr.shape[0], self.T)
            dist = np.linalg.norm(arr[:n] - self.ego_position[:n, :2], axis=1)
            near = np.flatnonzero(dist <= float(event_radius_m))
            if near.size == 0:
                continue
            span = [float(near[0]) * self.dt_s, float(near[-1]) * self.dt_s]
            best = span if best is None else [min(best[0], span[0]), max(best[1], span[1])]
        return None if best is None else [round(best[0], 2), round(best[1], 2)]

    def parallel_lanes(
        self, reference: Polyline, arc: float, *, max_lateral_m: float = 9.0,
        align_cos: float = 0.7,
    ) -> List[dict]:
        """Lanes running alongside ``reference`` at ``arc``.

        An event type like C-10 is only fair if the ego has somewhere to go: "a legal
        evasion exists — shoulder or free adjacent lane — so the episode is not
        unavoidable at spawn". A single-carriageway host cannot offer that, so
        the number of parallel lanes is a host-qualification question, not a
        placement detail.

        Returns:
            One entry per lane whose centreline passes within ``max_lateral_m``
            and runs roughly parallel: ``{lane_id, lateral_m, same_direction}``.
            ``lateral_m`` is +left of the reference; ``same_direction`` is False
            for an opposing carriageway, which is NOT a legal evasion.
        """
        x, y, _, heading = reference.sample([float(arc)])
        point = np.array([float(x[0]), float(y[0])])
        theta = float(heading[0])
        out: List[dict] = []
        for lane_id in self.lane_ids():
            try:
                lane = self.lane(lane_id)
            except PlacementError:
                continue
            s_l, lateral = lane.project(point)
            if abs(lateral) > max_lateral_m:
                continue
            _, _, _, tangent = lane.sample([s_l])
            dot = float(np.cos(float(tangent[0]) - theta))
            if abs(dot) < align_cos:      # crossing lane, not a parallel one
                continue
            out.append({
                "lane_id": lane_id,
                # lane.project gives the offset of the REFERENCE point from the
                # lane; flip it so the sign reads "lane is this far left of us".
                "lateral_m": round(-lateral, 2),
                "same_direction": dot > 0,
            })
        out.sort(key=lambda d: d["lateral_m"])
        return out

    def evasion_lanes(self, reference: Polyline, arc: float) -> List[dict]:
        """Parallel lanes an ego could legally swerve into (same direction, not its own)."""
        return [
            l for l in self.parallel_lanes(reference, arc)
            if l["same_direction"] and abs(l["lateral_m"]) > 1.0
        ]

    # -- cross-section ---------------------------------------------------
    def cross_section(self, reference: Polyline, arc: float) -> CrossSection:
        """Measure the road either side of ``reference`` at ``arc``.

        This is what turns "the right shoulder" into a number the recipe can
        pin: offsets are measured from the reference centreline, and what −5.0 m
        lands on is a property of this host.
        """
        x, y, _, heading = reference.sample([float(arc)])
        point = np.array([float(x[0]), float(y[0])])
        theta = float(heading[0])
        section = CrossSection(arc=float(arc))

        lane_id, _ = self.nearest_lane(point, heading=theta)
        if lane_id is not None:
            section.lane_id = lane_id
            feat = self.map_features.get(lane_id, {})
            left = self._lateral_of(feat.get(SD.LEFT_BOUNDARIES), point, theta)
            right = self._lateral_of(feat.get(SD.RIGHT_BOUNDARIES), point, theta)
            section.left_boundary_m = left
            section.right_boundary_m = right
            if left is not None and right is not None:
                section.lane_width_m = abs(left - right)

        edges: List[float] = []
        for key, feat in self.map_features.items():
            if _ROAD_EDGE_HINT not in str(feat.get(SD.TYPE, "")).upper() and not key.startswith(
                "road_edge"
            ):
                continue
            lateral = self._lateral_of(feat.get(SD.POLYLINE), point, theta)
            if lateral is not None:
                edges.append(lateral)
        if edges:
            left_edges = [e for e in edges if e > 0]
            right_edges = [e for e in edges if e < 0]
            section.left_road_edge_m = min(left_edges) if left_edges else None
            section.right_road_edge_m = max(right_edges) if right_edges else None
        return section

    @staticmethod
    def _lateral_of(polyline, point: np.ndarray, theta: float) -> Optional[float]:
        """Signed (+left) lateral distance from ``point`` to a map polyline."""
        if polyline is None:
            return None
        pts = np.asarray(polyline, np.float64)
        if pts.ndim != 2 or pts.shape[0] < 1:
            return None
        delta = pts[:, :2] - point
        # Project onto the reference's own normal; keep the nearest vertex by
        # longitudinal distance so a long boundary is measured beside the point,
        # not at whichever end happens to be closest in 2-D.
        longitudinal = delta[:, 0] * np.cos(theta) + delta[:, 1] * np.sin(theta)
        lateral = -delta[:, 0] * np.sin(theta) + delta[:, 1] * np.cos(theta)
        i = int(np.argmin(np.abs(longitudinal)))
        if abs(longitudinal[i]) > 10.0:  # boundary does not reach this station
            return None
        return float(lateral[i])


def resolve_reference(probe: HostProbe, reference: Any) -> Polyline:
    """A recipe's ``reference`` name -> :class:`Polyline`.

    The named references are the module docstring's list; ``"lane:<id>"`` and an
    inline polyline are the escape hatches for geometry no name covers.
    """
    if isinstance(reference, Polyline):
        return reference
    if isinstance(reference, dict):
        if "polyline" in reference:
            return Polyline.from_points(reference["polyline"], name="polyline")
        if "lane" in reference:
            return probe.lane(reference["lane"])
        raise PlacementError(f"reference mapping needs 'polyline' or 'lane', got {sorted(reference)}")
    if isinstance(reference, Sequence) and not isinstance(reference, str):
        return Polyline.from_points(reference, name="polyline")
    text = str(reference or "ego_route")
    # ALTERNATIVES, first that resolves. An event type whose event has two shapes on
    # the ground -- V-10's car is on the lane the ego joins where the map draws
    # one, and beside the ego where it does not -- can otherwise only be
    # written for the shape the majority of its hosts happen to have, and the
    # rest are dropped or forced. `at:` already reads `a|b` this way.
    if "|" in text:
        tried = []
        for alt in [v.strip() for v in text.split("|") if v.strip()]:
            try:
                out = resolve_reference(probe, alt)
                out.chosen_alternative = alt          # the author records which
                return out
            except Exception as exc:                  # noqa: BLE001
                tried.append(f"{alt}: {exc}")
        raise PlacementError(
            "no alternative in reference " + repr(text) + " resolved on this host — "
            + "; ".join(tried))
    if text == "ego_route":
        return probe.ego_route
    if text == "route_lanes":
        return probe.route_lanes()
    if text == "ego_lane_chain":
        return probe.ego_lane_chain()
    if text == "opposing_lane_chain":
        return probe.opposing_lane_chain()
    if text == "crossing_lane_chain":
        return probe.crossing_lane_chain()
    if text.startswith("crossing_lane_chain:"):
        return probe.crossing_lane_chain(index=int(text.split(":", 1)[1]))
    if text == "merging_lane_chain":
        return probe.merging_lane_chain()
    if text == "merge_target_lane_chain":
        return probe.merge_target_lane_chain()
    if text == "merge_pressure_lane_chain":
        return probe.merge_pressure_lane_chain()
    if text == "bike_lane_chain":
        return probe.bike_lane_chain()
    if text == "crosswalk_chain":
        return probe.crosswalk_chain()
    if text.startswith("crosswalk_chain:"):
        return probe.crosswalk_chain(index=int(text.split(":", 1)[1]))
    if text.startswith("lane_chain:"):
        return probe.lane_chain(text.split(":", 1)[1])
    if text.startswith("lane:"):
        return probe.lane(text.split(":", 1)[1])
    raise PlacementError(
        f"unknown reference {text!r}; expected 'ego_route', 'ego_lane_chain', "
        f"'opposing_lane_chain', 'crossing_lane_chain[:N]', 'merging_lane_chain', "
        f"'bike_lane_chain', 'crosswalk_chain[:N]', 'route_lanes', 'lane:<id>', "
        f"'lane_chain:<id>', or an inline polyline"
    )
