# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Privileged traffic-light handling for PDM-Closed: red lights as obstacles.

Upstream ``tuplan_garage`` PDM-Closed never scores a red light. It complies
with one because ``PDMObservation.update`` adds every RED lane connector on
the route to the occupancy map (``red_light_<id>`` tokens) and
``PDMGenerator._update_leading_agents`` then treats that polygon as a
stationary leading obstacle, so every IDM proposal brakes to a standstill at
the connector's entry — the stop line — and drives on once the token is gone.
The scorer skips those tokens, so a red light is a *longitudinal constraint*,
never a collision.

NavSafe's port lacked the observation half: the planner read no light
state at all, and on a NavSafe cell whose own connector is red for 16 s it
drove straight through (measured: ``0f622aef14545f59``, connector 69940 red
at frames 0-163, ego flagged at the stop line at frames 85-97). This module
supplies that half from the scenario's logged ``dynamic_map_states`` — the
same privileged per-timestep light states the evaluator's traffic-light term
reads — so the planner and the metric agree on what is red, when.

Geometry, pinned by the data rather than assumed: in py123d/nuPlan logs the
lanes carrying a light state are intersection **connectors**, whose START is
the stop line. The ego must therefore be brought to rest before a red
connector's first polyline point. A red connector is on the route when the
route passes through that start point, aligned with the connector's initial
heading, and the route DRIVES that connector: by id when the route came from
a lane-graph walk (``diagnostics["route_walk_lane_ids"]`` — upstream's
``route_lane_dict``), otherwise because the route keeps following the
connector beyond the line (sampled at :data:`STOP_LINE_FOLLOW_SAMPLE_M`).
That membership test is what keeps the planner from stopping for the *other*
connectors that share the same entry: a protected-left connector that is red
while the straight-ahead one is green starts at the same point with nearly
the same tangent and only separates a few metres in (13 of the 24 benchmark
bundles hold such a mixed-phase shared start somewhere in their log).
Connectors the ego merely crosses or merges into (cross traffic's reds, whose
starts lie on other approaches) never qualify.

Everything here is pure geometry over ``scenario_data``; no simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from navsafe.policy.state.pdm_closed_planner.config import PDMConfig
from navsafe.policy.state.pdm_closed_planner.forward_sim import AgentPrediction

#: ``dynamic_map_states`` value meaning "red" (ScenarioNet / py123d vocabulary;
#: ``LANE_STATE_CAUTION`` and ``LANE_STATE_UNKNOWN`` do not constrain).
RED_LANE_STATE = "LANE_STATE_STOP"
#: ``AgentPrediction.type`` of a red-light obstacle. Deliberately outside
#: ``COLLIDABLE_ACTOR_TYPES`` so nothing that filters real actors by type can
#: mistake the stop line for traffic.
RED_LIGHT_TYPE = "RED_LIGHT"
#: ``AgentPrediction.track_id`` prefix — the upstream ``red_light_<id>`` token.
RED_LIGHT_TOKEN_PREFIX = "red_light_"

#: How far laterally the route may pass from a connector's polyline before it
#: is no longer "following" that connector. Half a 3.5 m urban lane.
STOP_LINE_LATERAL_TOL_M = 1.75
#: Minimum cosine between the route tangent at the stop line and the
#: connector's initial tangent (60 degrees): excludes crossing connectors.
STOP_LINE_HEADING_COS_MIN = 0.5
#: The route must stay within the lateral tolerance of the connector at these
#: arc-length samples past its start (as far as both extend). Separates the
#: connector the route follows from a sibling that shares its stop line.
STOP_LINE_FOLLOW_SAMPLE_M = (5.0, 10.0, 15.0)
#: Box the IDM brakes for. Its REAR face sits on the stop line; the length
#: only has to exceed the forward-simulation step so the corridor sweep cannot
#: step over it. Width covers the lane so every lateral offset proposal sees it.
RED_LIGHT_OBSTACLE_LENGTH_M = 1.0
RED_LIGHT_OBSTACLE_WIDTH_M = 3.5
RED_LIGHT_HOLD_DISTANCE_M = 30.0
RED_LIGHT_HOLD_GRACE_S = 3.0


@dataclass(frozen=True)
class RedLightStopLine:
    """A red connector's stop line, located on the route."""

    lane_id: str
    #: Arc length along the route centerline (m from its first point).
    s_route: float
    #: Stop line position (route point at ``s_route``).
    xy: Tuple[float, float]
    #: Unit route tangent at the stop line.
    tangent: Tuple[float, float]
    #: Distance from the ego's route position to the stop line (m, >= 0).
    distance_m: float
    #: Length of the red connector's polyline (m): the obstacle spans it.
    connector_length_m: float = RED_LIGHT_OBSTACLE_LENGTH_M


def red_lane_ids_at(scenario_data: Dict[str, Any], frame_id: int,
                    *, lookahead_frames: int = 0) -> Set[str]:
    """Lane ids whose logged light state is red at ``frame_id`` — or, with
    ``lookahead_frames``, at any timestep in ``[frame_id, frame_id +
    lookahead_frames]``.

    Past the end of an entry's log the lookup CLAMPS to the last logged
    state — the evaluator's ``_check_tlc_live`` does the same, because the
    replay world freezes at its last logged frame. A planner that read
    "no state" there would drive through a light the metric still holds red."""
    states = scenario_data.get("dynamic_map_states") or {}
    red: Set[str] = set()
    lo = max(0, int(frame_id))
    hi = lo + max(0, int(lookahead_frames))
    for lane_id, entry in states.items():
        try:
            seq = entry["state"]["object_state"]
        except (KeyError, TypeError):
            continue
        if seq is None or len(seq) == 0:
            continue
        last = len(seq) - 1
        start = min(lo, last)
        stop = min(hi, last)
        if any(seq[i] == RED_LANE_STATE for i in range(start, stop + 1)):
            red.add(str(lane_id))
    return red


def traffic_light_state_evidence(
    scenario_data: Dict[str, Any], lane_id: str, frame_id: int,
) -> Dict[str, Any]:
    """Describe the current signal lookup without changing its semantics.

    UNKNOWN can mean an explicit observation or a missing detection. Older
    inputs did not retain that distinction. A clamped last sample is not a
    new observation and says nothing about a future permissive phase.
    """
    entry = (scenario_data.get("dynamic_map_states") or {}).get(str(lane_id), {})
    state = entry.get("state") or {}
    seq = state.get("object_state")
    requested = max(0, int(frame_id))
    if seq is None or len(seq) == 0:
        return {"lane_id": str(lane_id), "requested_frame": requested,
                "source": "no_signal_sequence"}
    sample = min(requested, len(seq) - 1)
    present = state.get("observation_present")
    observed = None
    if ((isinstance(present, (list, tuple))
         or isinstance(present, np.ndarray) and present.ndim == 1)
            and len(present) == len(seq)
            and isinstance(present[sample], (bool, np.bool_))):
        observed = bool(present[sample])
    source = ("held_last_sample" if requested >= len(seq)
              else "missing_observation" if observed is False
              else "recorded_observation" if observed is True
              else "legacy_observation_presence_unknown")
    return {
        "lane_id": str(lane_id), "status": seq[sample], "source": source,
        "requested_frame": requested, "sample_frame": sample,
        "last_available_frame": len(seq) - 1,
        "sample_age_frames": requested - sample,
        "observation_present": observed,
    }


class _RoutePolyline:
    """Arc-length projection onto a (M, 2) polyline."""

    def __init__(self, pts: np.ndarray) -> None:
        self.pts = np.asarray(pts, dtype=np.float64)[:, :2]
        self.seg = np.diff(self.pts, axis=0)
        self.seg_lens = np.linalg.norm(self.seg, axis=1)
        safe = np.where(self.seg_lens > 1e-12, self.seg_lens, 1.0)
        self.tangents = self.seg / safe[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_lens)])
        self._denom = np.maximum(self.seg_lens ** 2, 1e-24)

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def project(self, p: Any) -> Tuple[float, float, np.ndarray]:
        """``(s, distance, unit tangent)`` of the closest polyline point.

        ``p`` is any 2-vector (tuple, list or array; extra components
        ignored).
        """
        q = np.asarray(p, dtype=np.float64)[:2]
        rel = q - self.pts[:-1]
        t = np.clip(np.einsum("ij,ij->i", rel, self.seg) / self._denom, 0.0, 1.0)
        proj = self.pts[:-1] + t[:, None] * self.seg
        d = np.linalg.norm(proj - q, axis=1)
        i = int(np.argmin(d))
        return (float(self.cum[i] + t[i] * self.seg_lens[i]), float(d[i]),
                self.tangents[i])

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.searchsorted(self.cum, s, side="right") - 1)
        i = max(0, min(i, len(self.seg_lens) - 1))
        if self.seg_lens[i] <= 1e-12:
            return self.pts[i].copy()
        frac = (s - self.cum[i]) / self.seg_lens[i]
        return self.pts[i] + frac * self.seg[i]


def _lane_polyline(feature: Dict[str, Any]) -> Optional[np.ndarray]:
    polyline = feature.get("polyline")
    if polyline is None:
        return None
    arr = np.asarray(polyline, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def find_red_light_stop_lines(
    scenario_data: Dict[str, Any],
    route_centerline: np.ndarray,
    *,
    ego_x: float,
    ego_y: float,
    frame_id: int,
    lookahead_frames: int = 0,
    route_lane_ids: Optional[Sequence[str]] = None,
    lateral_tol_m: float = STOP_LINE_LATERAL_TOL_M,
    heading_cos_min: float = STOP_LINE_HEADING_COS_MIN,
    release_margin_m: float = 0.0,
) -> List[RedLightStopLine]:
    """Stop lines of the red connectors the route is about to enter.

    Sorted nearest first (ties by lane id, so the result is deterministic
    across processes). A connector is dropped once the ego's route position
    is more than ``release_margin_m`` past its start — pass half the ego
    length for upstream's rule: ``PDMObservation.update`` keeps a red
    connector as a zero-gap lead until the ego box lies ``within`` its
    polygon (rear bumper past the stop line), and only then latches it as
    collided. A merely straddling ego therefore waits for green upstream; an
    ego released at the centre line (the port's earlier rule) drove on.
    ``lookahead_frames`` widens "red" to the coming window (see
    :func:`red_lane_ids_at`).

    Membership — is this connector the one the route drives? — is decided
    by id when ``route_lane_ids`` (the lane chain the route polyline was cut
    from: ``diagnostics["route_walk_lane_ids"]``, upstream's
    ``route_lane_dict``) is given. Without it the geometry decides: the
    route must keep tracking the connector at :data:`STOP_LINE_FOLLOW_SAMPLE_M`
    past the line. When neither the route nor the connector reaches the
    first sample the follow test is UNDECIDED, and a stop line is then
    accepted only if every signalized connector sharing that start is red —
    a red protected-left beside a green straight must not stop a short-route
    ego that cannot yet be told apart from a turning one.
    """
    if not (np.isfinite(ego_x) and np.isfinite(ego_y)):
        return []
    route = np.asarray(route_centerline, dtype=np.float64)
    if route.ndim != 2 or route.shape[0] < 2 or not np.all(np.isfinite(route)):
        return []
    red = red_lane_ids_at(scenario_data, frame_id,
                          lookahead_frames=lookahead_frames)
    if not red:
        return []
    poly = _RoutePolyline(route)
    if poly.length <= 1e-6:
        return []
    s_ego, _, _ = poly.project((ego_x, ego_y))
    map_features = scenario_data.get("map_features") or {}
    signalized = set(str(k) for k in (scenario_data.get("dynamic_map_states") or {}))
    known_ids = (set(str(x) for x in route_lane_ids)
                 if route_lane_ids is not None else None)

    found: List[RedLightStopLine] = []
    for lane_id in sorted(red):
        feature = map_features.get(lane_id)
        if feature is None:
            continue
        lane = _lane_polyline(feature)
        if lane is None or not np.all(np.isfinite(lane)):
            continue
        lane_poly = _RoutePolyline(lane)
        if lane_poly.length <= 1e-6:
            continue
        start = lane[0]
        s_sl, d_sl, route_tan = poly.project(start)
        if d_sl > lateral_tol_m:
            continue
        if s_sl <= s_ego - float(release_margin_m):
            continue
        # The connector's initial heading: over its first metres, not its
        # first (possibly sub-centimetre) segment.
        probe = lane_poly.point_at(min(2.0, lane_poly.length))
        lane_tan = probe - start
        norm = float(np.linalg.norm(lane_tan))
        if norm < 1e-9:
            continue
        lane_tan = lane_tan / norm
        if float(np.dot(route_tan, lane_tan)) < heading_cos_min:
            continue
        if known_ids is not None:
            if lane_id not in known_ids:
                continue
        else:
            verdict = _route_follows_connector(poly, lane_poly, s_sl, lateral_tol_m)
            if verdict is False:
                continue
            if verdict is None and not _every_sibling_red(
                    map_features, signalized, red, lane_id, start, lateral_tol_m):
                continue
        xy = poly.point_at(s_sl)
        found.append(RedLightStopLine(
            lane_id=str(lane_id), s_route=float(s_sl),
            xy=(float(xy[0]), float(xy[1])),
            tangent=(float(route_tan[0]), float(route_tan[1])),
            distance_m=float(max(0.0, s_sl - s_ego)),
            connector_length_m=float(max(RED_LIGHT_OBSTACLE_LENGTH_M,
                                         lane_poly.length))))
    found.sort(key=lambda sl: (sl.s_route, sl.lane_id))
    return found


def _route_follows_connector(route: _RoutePolyline, lane: _RoutePolyline,
                             s_sl: float, lateral_tol_m: float) -> Optional[bool]:
    """``True``/``False`` when a follow sample could be judged, ``None`` when
    neither the route nor the connector reaches the first one."""
    judged = False
    for ds in STOP_LINE_FOLLOW_SAMPLE_M:
        if ds >= lane.length or s_sl + ds >= route.length:
            break
        judged = True
        _, d_sample, _ = route.project(lane.point_at(ds))
        if d_sample > lateral_tol_m:
            return False
    return True if judged else None


def _every_sibling_red(map_features: Dict[str, Any], signalized: Set[str],
                       red: Set[str], lane_id: str, start: np.ndarray,
                       tol_m: float) -> bool:
    """No green way through this stop line: every signalized connector whose
    start lies within ``tol_m`` of ``start`` is red."""
    for other in signalized:
        if other == lane_id or other in red:
            continue
        feature = map_features.get(other)
        if feature is None:
            continue
        other_lane = _lane_polyline(feature)
        if other_lane is None:
            continue
        if float(np.linalg.norm(other_lane[0] - start)) <= tol_m:
            return False
    return True


def red_light_obstacles(
    stop_lines: Sequence[RedLightStopLine],
    cfg: PDMConfig,
) -> Dict[str, AgentPrediction]:
    """Stationary obstacle predictions with their rear face on each stop line.

    Shaped exactly like the real-agent predictions
    (:func:`~.forward_sim.predict_agents_constant_velocity`) so the proposal
    generator's corridor sweep, the initial lead search and the brake guard
    consume them unchanged. IDM then settles the ego ``min_gap`` (1 m) short
    of the line.

    The box spans the whole connector length along the route tangent at the
    stop line (upstream's obstacle is the connector POLYGON, whose centroid
    sits mid-connector): a proposal that overruns the line keeps the box
    "ahead" of its rear axle and overlapping — zero gap, IDM holds it at
    rest — instead of shrugging off a 1 m box half a car length in. Straight
    along the initial tangent is an approximation of a curved connector's
    polygon; the near face (the stop line) is exact.
    """
    n = int(round(cfg.agent_prediction_horizon_s / cfg.sim_dt)) + 1
    out: Dict[str, AgentPrediction] = {}
    for sl in stop_lines:
        length = float(max(RED_LIGHT_OBSTACLE_LENGTH_M, sl.connector_length_m))
        half = 0.5 * length
        tx, ty = sl.tangent
        cx = sl.xy[0] + tx * half
        cy = sl.xy[1] + ty * half
        heading = math.atan2(ty, tx)
        token = f"{RED_LIGHT_TOKEN_PREFIX}{sl.lane_id}"
        out[token] = AgentPrediction(
            track_id=token,
            x=np.full(n, cx, dtype=np.float64),
            y=np.full(n, cy, dtype=np.float64),
            heading=np.full(n, heading, dtype=np.float64),
            vx=np.zeros(n, dtype=np.float64),
            vy=np.zeros(n, dtype=np.float64),
            valid=np.ones(n, dtype=bool),
            type=RED_LIGHT_TYPE,
            length=length,
            width=RED_LIGHT_OBSTACLE_WIDTH_M,
        )
    return out


def is_red_light_token(track_id: str) -> bool:
    return str(track_id).startswith(RED_LIGHT_TOKEN_PREFIX)


__all__ = [
    "RED_LANE_STATE",
    "RED_LIGHT_HOLD_DISTANCE_M",
    "RED_LIGHT_HOLD_GRACE_S",
    "RED_LIGHT_OBSTACLE_LENGTH_M",
    "RED_LIGHT_OBSTACLE_WIDTH_M",
    "RED_LIGHT_TOKEN_PREFIX",
    "RED_LIGHT_TYPE",
    "RedLightStopLine",
    "STOP_LINE_FOLLOW_SAMPLE_M",
    "STOP_LINE_HEADING_COS_MIN",
    "STOP_LINE_LATERAL_TOL_M",
    "find_red_light_stop_lines",
    "is_red_light_token",
    "red_lane_ids_at",
    "red_light_obstacles",
]
