"""THE EPDMS module — one file owns both public behaviors of the metric.

* :class:`EPDMSTrajectoryScorer_Fast` — the batch engine: candidate scoring
  and selection (``score_candidates``/``select_best``), parent of
  the PDM planner scorer subclasses. Optimized with per-frame
  precompute shared across candidates, AABB prefilters before Shapely
  intersections, and vectorized trajectory-state computation.
* :class:`EPDMSLiveScorer` — the live per-frame METRIC OF RECORD scored on
  executed sim state (every ``metrics.json`` number), absorbed from the
  deleted ``evaluation/utils/epdms_scorer_md.py`` (simplify.md, scorer
  consolidation Phase 2).

Both share this module's constants and term-semantics conventions; their
deliberate behavioral divergences are enumerated in the
:class:`EPDMSLiveScorer` docstring.
"""

import logging
import math
import time
import numpy as np
from typing import Dict, Any, FrozenSet, Optional, List, Tuple
from shapely.geometry import LineString, Point, Polygon
from shapely import affinity
from scipy.signal import savgol_filter

from navsafe.core.agent_forecast import (
    forecast_velocity,
    is_moving_agent_type,
    nearest_k_admitted,
)
from navsafe.core.collision_actor_types import COLLIDABLE_ACTOR_TYPES
from navsafe.evaluation.scorers.at_fault import at_fault_ego_speed
from navsafe.evaluation.scorers.gt_stride import gt_stride as _gt_stride
from navsafe.scenario.scenario_description import scenario_dt_seconds

from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M
from navsafe.core.track_dims import track_dims
from navsafe.evaluation.scorers.base_scorer import BaseTrajectoryScorer
from navsafe.evaluation.utils.lane_proxy import (
    DrivableAreaProxy,
    corners_in_drivable_area,
)

logger = logging.getLogger(__name__)

# --- Constants ---
VEHICLE_LENGTH = EGO_LENGTH_M
VEHICLE_WIDTH = EGO_WIDTH_M
VEHICLE_HALF_DIAG = math.sqrt((VEHICLE_LENGTH / 2) ** 2 + (VEHICLE_WIDTH / 2) ** 2)

W_PROGRESS, W_TTC, W_LANE_KEEPING, W_HISTORY_COMFORT, W_EXTENDED_COMFORT = 5.0, 5.0, 2.0, 2.0, 2.0
W_TOTAL = W_PROGRESS + W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT
EP_NORM_M = 30.0


def combine_epdms_terms(m: Dict[str, float], *, use_ep: bool,
                        use_ec: bool) -> float:
    """THE EPDMS aggregation: multiplicative gate × weighted average.

    Single shared implementation for the metric path (score_candidates below)
    and the VerifierScorer judge — a weight or term edit in one place used to
    be able to fork the two objectives silently.
    """
    multi_prod = m['nc'] * m['dac'] * m['ddc'] * m['tlc']
    # Summation ORDER is load-bearing: float addition is non-associative
    # (a reorder measures 1-ulp drift on ~36% of random term rows — enough
    # to flip a near-tied candidate argmax in select_best). This order —
    # ttc/lk/hc, then ec, then ep — reproduces bit-for-bit the accumulation
    # every scored campaign run used since the use_ec/use_ep parameters
    # landed. (Committed HEAD once summed ep first; that parity was already
    # given up by that change, and consistency with the scored baselines is
    # the one that matters.)
    weighted_sum = (W_TTC * m['ttc'] + W_LANE_KEEPING * m['lk']
                    + W_HISTORY_COMFORT * m['hc'])
    denom = W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT
    if use_ec:
        weighted_sum += W_EXTENDED_COMFORT * m['ec']
        denom += W_EXTENDED_COMFORT
    if use_ep:
        weighted_sum += W_PROGRESS * m['ep']
        denom += W_PROGRESS
    return float(multi_prod * weighted_sum / denom)
# Denominator when the caller scores WITHOUT the ego-progress term (use_ep=False).
# ``ep`` here is ABSOLUTE displacement normalized to 30 m over the plan horizon
# (see metrics['ep'] below), so it needs ~7.5 m/s to saturate: a perfectly safe
# but slow or stopped ego scores only 0.6875 with EP included. A takeover gate
# thresholded on that fires on being slow rather than on being unsafe, and it
# can never be cleared by a stalled student — the gate latches into permanent
# takeover. Dropping EP matches the weighting of the reported `epdms_no_ep`
# metric, so a gate threshold means the same thing as the number in the table.
W_NO_EP_TOTAL = W_TTC + W_LANE_KEEPING + W_HISTORY_COMFORT + W_EXTENDED_COMFORT
LANE_DEVIATION_LIMIT = 0.5
LANE_KEEPING_WINDOW = 2.0
TTC_HORIZON = 1.0
STOPPED_SPEED_THRESHOLD = 5e-2
# CaRL PDMScorer's own threshold is 5e-3. The generic EPDMS path retains its
# historical threshold above; full-state PDM proposals use this exact value.
PDM_STOPPED_SPEED_THRESHOLD = 5e-3
# Pacifica geometry used by CaRL's StateSE2 (rear axle) TTC angle test:
# (front_length 4.049 - rear_length 1.127) / 2 = 1.461 m.
PDM_REAR_AXLE_TO_CENTER_M = 1.461
MAX_ACCEL, MAX_JERK, MAX_YAW_RATE = 4.89, 8.37, 0.95
# MAX_ACCEL carries nuPlan's lateral-acceleration bound VALUE (4.89), but
# both scorers deliberately apply it to the TOTAL acceleration magnitude
# (|accel vec|), not the lateral component alone — a conservative
# approximation that needs no lane frame. Same for MAX_JERK (jerk
# magnitude). Do not "fix" the comparison to lateral-only without a
# bit-parity sign-off: it would move every scored table.
# Remaining nuPlan ``ego_is_comfortable`` bounds: signed longitudinal
# acceleration, longitudinal jerk, and yaw acceleration.
MIN_LON_ACCEL, MAX_LON_ACCEL = -4.05, 2.40
MAX_LON_JERK = 4.13
MAX_YAW_ACCEL = 1.93
EC_ACCEL_THRESH, EC_JERK_THRESH, EC_YAW_RATE_THRESH = 0.7, 0.5, 0.1


def _carl_ttc_actor_is_at_fault(
    rear_x: float,
    rear_y: float,
    ego_heading: float,
    actor_x: float,
    actor_y: float,
    *,
    widened_cone: bool,
) -> bool:
    """CaRL/nuPlan TTC angle predicate (30° ahead, 150° behind)."""
    dx = float(actor_x) - float(rear_x)
    dy = float(actor_y) - float(rear_y)
    distance = math.hypot(dx, dy)
    if distance <= 1e-12:
        angle = 0.0
    else:
        along = (
            dx * math.cos(ego_heading) + dy * math.sin(ego_heading)
        ) / distance
        angle = math.acos(float(np.clip(along, -1.0, 1.0)))
    is_ahead = angle < math.radians(30.0)
    is_behind = angle > math.radians(150.0)
    return bool(is_ahead or (widened_cone and not is_behind))


def _carl_nc_actor_is_behind(
    rear_x: float,
    rear_y: float,
    ego_heading: float,
    actor_x: float,
    actor_y: float,
) -> bool:
    """nuPlan ``is_agent_behind``: angle > 150 deg from the ego's REAR AXLE.

    ``pdm_scorer_utils.get_collision_type`` calls this with the rear-axle pose
    and the actor polygon's centroid to decide ACTIVE_REAR, which is the one
    active category upstream does NOT charge. A half-plane through the ego
    centre (``longitudinal_dist < 0``) is not the same predicate: it exempts
    the whole 90-150 deg band, i.e. actors beside and slightly behind the ego,
    which upstream routes to ACTIVE_LATERAL and charges whenever the ego is
    straddling lanes or off the drivable area.
    """
    dx = float(actor_x) - float(rear_x)
    dy = float(actor_y) - float(rear_y)
    distance = math.hypot(dx, dy)
    if distance <= 0.0:
        return False
    along = (
        dx * math.cos(ego_heading) + dy * math.sin(ego_heading)
    ) / distance
    angle = math.acos(float(np.clip(along, -1.0, 1.0)))
    return bool(angle > math.radians(150.0))


#: ``pdm_scorer_utils.CollisionType`` categories, in the reference's own
#: classification order. Only the charged/uncharged split matters here, so
#: these are plain strings rather than an enum mirroring nuPlan's ints.
CARL_STOPPED_EGO = "stopped_ego"
CARL_STOPPED_TRACK = "stopped_track"
CARL_ACTIVE_REAR = "active_rear"
CARL_ACTIVE_FRONT = "active_front"
CARL_ACTIVE_LATERAL = "active_lateral"


def _actor_speeds_from_ego_state(
    ego_state: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """``{actor_id: speed}`` from the live agent snapshot, or ``{}``.

    Upstream's ``is_track_stopped`` reads the tracked object's own velocity.
    The snapshot the evaluator already exports for actor TYPES carries it too
    (``_execution_agent_states[i]["velocity"]`` is a 2-vector), so the
    reference's quantity is available and the port does not have to infer speed
    from consecutive positions.
    """
    if not ego_state:
        return {}
    states = ego_state.get("_execution_agent_states")
    if not states:
        return {}
    out: Dict[str, float] = {}
    for i, state in enumerate(states):
        if state.get("is_ego", False):
            continue
        actor_id = str(state.get("id", i))
        if not is_moving_agent_type(state.get("type", "VEHICLE")):
            # ``is_track_stopped``: a non-Agent tracked object is stopped by
            # definition, whatever its measured velocity.
            out[actor_id] = 0.0
            continue
        velocity = state.get("velocity")
        if velocity is None:
            continue
        vector = np.asarray(velocity, dtype=np.float64).ravel()
        if vector.size < 2 or not np.all(np.isfinite(vector[:2])):
            continue
        out[actor_id] = float(
            math.hypot(float(vector[0]), float(vector[1])))
    return out


def _carl_collision_type(
    *,
    ego_speed: float,
    rear_x: float,
    rear_y: float,
    ego_heading: float,
    actor_x: float,
    actor_y: float,
    actor_speed: Optional[float],
    front_edge_hits: bool,
) -> str:
    """``pdm_scorer_utils.get_collision_type``, branch for branch.

    ``actor_speed`` is the actor's OWN speed, which is what upstream's
    ``is_track_stopped`` reads. ``None`` means the port could recover neither a
    logged velocity nor a previous sample to difference; that is a port-only
    state with no reference counterpart, and it must NOT be reported as
    stopped. Claiming 'stopped' charges a moving vehicle first seen at the
    contact step, which upstream would classify ACTIVE_REAR and exempt. The
    classifier therefore falls through to the geometric branches, which is what
    upstream does for any moving object.
    """
    if float(ego_speed) <= STOPPED_SPEED_THRESHOLD:
        return CARL_STOPPED_EGO
    if actor_speed is not None and float(actor_speed) <= STOPPED_SPEED_THRESHOLD:
        return CARL_STOPPED_TRACK
    if _carl_nc_actor_is_behind(rear_x, rear_y, ego_heading, actor_x, actor_y):
        return CARL_ACTIVE_REAR
    if front_edge_hits:
        return CARL_ACTIVE_FRONT
    return CARL_ACTIVE_LATERAL


def _carl_collision_is_at_fault(
    collision_type: str, *, straddling_or_offroad: bool
) -> bool:
    """``pdm_scorer.py:324-334``'s at-fault predicate."""
    if collision_type in (CARL_ACTIVE_FRONT, CARL_STOPPED_TRACK):
        return True
    return bool(
        straddling_or_offroad and collision_type == CARL_ACTIVE_LATERAL
    )

# DDC (driving-direction compliance) thresholds — upstream pdm_scorer:
# < 2 m against traffic → 1.0, 2–6 m → 0.5, > 6 m → 0.0.
DDC_HALF_VIOLATION_M = 2.0
DDC_FULL_VIOLATION_M = 6.0

# TLC (traffic-light compliance) — "ran a red light" semantics.
#
# The lanes that carry a light state in py123d / nuPlan logs are intersection
# CONNECTORS, and a connector's stop line is its START. Running the light is
# therefore the CROSSING event: the ego CENTRE enters a red connector's
# polygon on this frame (it was outside on the previous one) near the
# connector's start, moving faster than TL_MOVING_SPEED_MPS. CARLA's
# RunningRedLightTest, transposed to the per-frame column.
#
# "Near the start" is ``s <= TL_STOP_LINE_ZONE_M + speed * dt``: the
# crossing pose can be up to one sample past the line, and the bound is what
# keeps a SIDE entry -- the ego's own lane overlapping a red sibling that
# merges into the same exit -- from counting (those entries happen metres
# from the sibling's END).
#
# Previously the rule looked at the connector's END (``dist_to_end <
# 5``) on any red polygon the ego BOX touched, as if the lane ended at the
# stop line. On connectors that charged the ego for the last metres of the
# sibling connectors merging into the same exit as its own green one.
# Measured on the benchmark scenarios (rescoring the stored closed-loop trajectories of
# both 24-cell arms): every red-light flag in the corpus -- 0f622aef14545f59
# (ego on green 69070; red 69940/68807 merge into its exit),
# 99a98a7ffb075389 (green 47169; red 52786/52338) and 0ba54149d1575f95
# (green 52621; red 48590) -- was such a sibling, and the logged HUMAN is
# flagged on all three. The crossing rule charges none of them; it also
# does not charge a run whose SCORED phase begins inside a connector the
# replayed warm-up entered (0ba54149 again: connector 47570 entered at
# warm-up frame 11), because no crossing happens on a scored frame. A
# caller with no previous centre at all (no warm-up, first frame) falls
# back to the stateless zone test: centre within TL_STOP_LINE_ZONE_M of the
# start.
TL_STOP_LINE_ZONE_M = 5.0
TL_MOVING_SPEED_MPS = 1.0
# SIGNAL HOLD — "a red light is what is holding the ego". True when a red
# connector's stop line lies ahead of the ego along its heading within
# SIGNAL_HOLD_LOOKAHEAD_M (laterally within SIGNAL_HOLD_LATERAL_M, heading
# aligned), or the ego has just crossed one (centre inside the red connector
# within the stop-line zone). Published per frame as ``signal_hold`` so the
# NavSafe deadlock detector can tell a vehicle waiting at a signal from a
# frozen one; it never enters the score. Generous lookahead on purpose: a
# policy that is genuinely frozen near a red light still ends on the budget.
# Known limit: a queue more than SIGNAL_HOLD_LOOKAHEAD_M behind a red light
# is held by its lead vehicle, not by a line this rule can see, and is still
# classified by the plain deadlock rule. The planner's own hold distance
# (pdm_closed_planner.traffic_lights.RED_LIGHT_HOLD_DISTANCE_M) matches it.
SIGNAL_HOLD_LOOKAHEAD_M = 30.0
SIGNAL_HOLD_LATERAL_M = 2.0
SIGNAL_HOLD_HEADING_COS_MIN = 0.5

# Ego footprint for the LIVE scorer's shapely-affinity polygon construction
# (EPDMSLiveScorer below). The batch scorer builds corners with cos/sin
# arithmetic instead (_get_agent_polygon); both describe the same rectangle
# but round differently at the ulp level, so the live scorer keeps the
# affinity path its scored history was produced with.
VEHICLE_POLYGON_COORDS = np.array([
    [VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2],
    [VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, -VEHICLE_WIDTH / 2],
    [-VEHICLE_LENGTH / 2, VEHICLE_WIDTH / 2]
])


def _light_id(lane: Any) -> str:
    """The id a lane is keyed by in ``dynamic_map_states`` (MetaDrive tuple
    indices carry it in slot 2)."""
    idx = lane.index
    if isinstance(idx, (tuple, list)) and len(idx) > 2:
        return str(idx[2])
    return str(idx)


def _red_lane_ids(traffic_lights: Dict[Any, Any], frame_idx: int) -> set:
    """Lane ids red at ``frame_idx``, clamped to the last logged state.

    ``max(0, ·)``: a (caller-bug) negative index must not silently read from
    the end of the log. Past the end the world — and the light — is frozen
    at its last logged frame.
    """
    red = set()
    for lane_id, tl_data in traffic_lights.items():
        s_list = tl_data['state']['object_state']
        if len(s_list) == 0:
            continue
        if s_list[max(0, min(frame_idx, len(s_list) - 1))] == "LANE_STATE_STOP":
            red.add(str(lane_id))
    return red


def _lane_polyline_start(lane: Any) -> Optional[np.ndarray]:
    """First polyline point through the lane interface, ``None`` if the
    object exposes neither ``polyline`` nor ``position`` (MetaDrive lanes)."""
    polyline = getattr(lane, "polyline", None)
    if polyline is not None and len(polyline) > 0:
        return np.asarray(polyline[0], dtype=np.float64)[:2]
    position = getattr(lane, "position", None)
    if callable(position):
        try:
            return np.asarray(position(0.0, 0.0), dtype=np.float64)[:2]
        except Exception:  # noqa: BLE001 — interface probe
            return None
    return None


def _ran_red_light(lane: Any, lane_polygon: Any, centre_now: Point,
                   centre_prev: Optional[Point], max_entry_s: float) -> bool:
    """Did the ego cross THIS red lane's stop line on this sample?

    ``lane`` is a LaneProxy-shaped object (``local_coordinates``);
    ``lane_polygon`` its footprint; the caller has already established that
    the lane is red and the ego is moving faster than TL_MOVING_SPEED_MPS,
    and passes the longitudinal bound it wants on the crossing pose
    (TL_STOP_LINE_ZONE_M plus the sample's travel; the bare zone when there
    is no previous sample). ``centre_prev`` is the ego centre one sample
    earlier (``None`` when there is none). See TL_STOP_LINE_ZONE_M.
    """
    if not lane_polygon.contains(centre_now):
        return False
    if centre_prev is not None and lane_polygon.contains(centre_prev):
        return False                      # already inside: no crossing here
    s_along, _ = lane.local_coordinates(
        np.array([centre_now.x, centre_now.y]))
    return bool(s_along <= max_entry_s)


def _heading_aligned(lane: Any, centre: Point, heading: float,
                     cos_min: float = 0.5) -> bool:
    """Is the lane's direction at the ego's projection within ~60 deg of
    the ego heading? (Excludes the crossing connectors a junction box
    overlaps with.)"""
    s_along, _ = lane.local_coordinates(np.array([centre.x, centre.y]))
    tangent = lane.heading_at(max(0.0, min(float(s_along), float(lane.length))))
    return float(tangent[0] * math.cos(heading)
                 + tangent[1] * math.sin(heading)) >= cos_min


def _green_way_through(nearby_lanes: Any, lane_polygons: Any,
                       red_lane_ids: set, signalized_ids: set,
                       centre: Point, heading: float) -> bool:
    """A non-red SIGNALIZED connector, heading-aligned, also contains the
    centre: the ego had a green way through this stop line.

    Connectors fanning out of one approach (straight / left / right) share
    the stop line and overlap for their first metres; nuPlan lights them
    per connector, so a protected-left can be red while straight is green.
    The per-frame column cannot know which one the ego is taking at the
    crossing, so it charges only when EVERY signalized connector the ego
    enters there is red — a legal manoeuvre is never charged, at the price
    of a red-arrow turn beside a green straight going uncharged. A lane
    with no light entry never exempts: it says nothing about this signal.
    """
    for lane, polygon in zip(nearby_lanes, lane_polygons):
        lid = _light_id(lane)
        if lid in red_lane_ids or lid not in signalized_ids:
            continue
        if not polygon.contains(centre):
            continue
        if _heading_aligned(lane, centre, heading):
            return True
    return False


class EPDMSTrajectoryScorer_Fast(BaseTrajectoryScorer):
    """
    Optimized EPDMS trajectory scorer.

    Pre-computes per-frame data (agent polygons, traffic lights) shared across
    all N candidates, caches per-candidate data (ego polygons, nearby lanes),
    and uses AABB pre-filtering to minimize expensive Shapely operations.

    Two modes of lane / coordinate-frame initialization are supported (see
    :meth:`initialize`):

    1. **Env-backed mode** (the historical default): ``env`` is a
       MetaDrive/IsaacSim env with a populated ``engine.map_manager``. Lanes
       are read from the engine's road network and the world→sim coordinate
       offset is calibrated against ``env.agent.position``.
    2. **Env-less mode**: ``env`` is ``None`` (or has no usable map_manager).
       Lanes are built from ``scenario_data['map_features']`` via
       :func:`navsafe.evaluation.utils.lane_proxy.build_lanes_from_scenario`,
       and the world→sim offset defaults to ``(0, 0)`` because the caller is
       providing trajectories already in scenario-world coordinates.

    The env-less mode lets rule-based planners
    (e.g. :class:`navsafe.policy.state.pdm_closed.PDMClosedAdapter`) score
    their own proposals during inference without instantiating a MetaDrive
    runtime. The metric formulas are identical in both modes; only the
    sources of lane geometry and the coordinate-frame offset differ.
    """

    def __init__(self, *, verbose: bool = True):
        """Construct an uninitialized scorer.

        Args:
            verbose: When ``True`` (the default for backward compatibility
                with the 12 sensor adapters that consume this scorer via the
                evaluator), the scorer prints a one-line summary on
                :meth:`initialize` and a per-frame summary on
                :meth:`select_best`. Pass ``False`` for callers that score
                many proposals per frame internally (e.g. PDM-Closed) — the
                evaluator-level summary is usually enough.
        """
        self._initialized = False
        self.scenario_data: Optional[dict] = None
        self.env = None
        self.sdc_id = None
        self.all_lanes: List[Any] = []
        self.all_lane_bounds: Optional[np.ndarray] = None  # (num_lanes, 4) array of [minx, miny, maxx, maxy]
        self.traffic_lights: Dict[Any, Any] = {}
        self.world_to_sim_offset = np.array([0.0, 0.0])
        self.scenario_dt = 0.1
        self.planner_dt = 0.5
        self.prev_frame_idx = None
        self.verbose = bool(verbose)
        # Opt-in: build FastLaneProxy lanes instead of the reference
        # LaneProxy (see navsafe/evaluation/utils/lane_proxy_fast.py);
        # PDMClosedFastAdapter ("pdm_closed_fast") switches it on. Since the
        # 2026-08 perf pass vectorised LaneProxy.local_coordinates itself
        # (bit-identical to the old scalar loop; 74-frame banked-corpus
        # replay + 163k-query sweep), the default path is the FASTER one
        # (164.5 vs 185.9 ms/frame on that corpus) — a probed default flip
        # was declined: FastLaneProxy scored the corpus bit-identically but
        # its np.hypot distance differs from the reference at 1 ulp on
        # ~17% of pointwise inputs, so flipping would trade away exact
        # semantics for negative perf. The flag stays as an escape hatch.
        self.use_fast_lanes = False
        # (key, value) memo for _precompute_frame_data — see its docstring.
        self._frame_data_cache: Optional[Tuple[Any, Any]] = None
        # Drivable-area proxy for DAC, built lazily from ``all_lanes`` on
        # first use and reset by ``initialize`` (see _drivable_area).
        self._drivable_area_proxy: Optional[DrivableAreaProxy] = None
        self._drivable_area_built = False
        # ``None`` keeps the generic evaluator's whole-map behavior. The
        # PDM wrapper sets CaRL's configured 50 m radius explicitly.
        self.map_radius_m: Optional[float] = None
        # See ``_calculate_metrics``: 1 = exact per-step agents (default),
        # 2 = the reference's 0.2 s occupancy-map sampling.
        self.observation_sample_res: int = 1
        # Ego footprint the batch scorer builds polygons with. The module
        # constants are the platform vehicle; a planner with its own
        # ``PDMConfig`` dims overrides them so it scores the car it models.
        self.ego_length_m: float = VEHICLE_LENGTH
        self.ego_width_m: float = VEHICLE_WIDTH
        self.rear_axle_to_center_m: float = PDM_REAR_AXLE_TO_CENTER_M
        #: Lane ids forming the current route, or ``None`` when the
        #: caller has not supplied them. Upstream's driving-direction
        #: term needs route IDENTITY, not just a centreline; ``None``
        #: keeps the legacy heading-only test rather than treating
        #: every nearby lane as on-route.
        self.route_lane_ids: Optional[FrozenSet[str]] = None

    def initialize(self, scenario_data: dict, env):
        """Initialize scorer with scenario data and environment."""
        self.scenario_data = scenario_data
        self.env = env
        self.sdc_id = scenario_data['metadata']['sdc_id']

        # Time step alignment. This used to parse metadata['ts'] inline as
        # ``float(ts.flat[0])`` -- i.e. it read the FIRST TIMESTAMP as if it
        # were an interval. On py123d data ``ts`` is an array of absolute
        # MICROSECOND stamps (~3.16e14), so scenario_dt became 3.16e14 and
        # gt_stride collapsed to int(round(0.5 / 3.16e14)) == 0. With stride 0
        # every scored pose read ``agents_per_t`` at the SAME sim_frame: the
        # world was frozen for the whole 4 s horizon, making nc/ttc judge
        # candidates against agents that never moved. Parse it in one audited
        # place instead (handles scalar dt, absolute-us and relative-s series).
        metadata = scenario_data.get('metadata', {})
        self.scenario_dt = scenario_dt_seconds(metadata, default=0.1)
        self.planner_dt = 0.5

        # Map extraction + pre-compute lane bounds for vectorized nearby query
        self.all_lanes = []
        self.traffic_lights = scenario_data.get('dynamic_map_states', {})

        # Path 1: env-backed lanes (MetaDrive engine road network).
        # Wrapped in try/except so an env that lacks engine/map_manager
        # silently falls through to the scenario_data path. This mirrors
        # ``EPDMSTrajectoryScorer.initialize`` so both scorers have the
        # same env-less behaviour.
        try:
            if self.env is not None and self.env.engine.map_manager.current_map:
                road_network = self.env.engine.map_manager.current_map.road_network
                if hasattr(road_network, 'get_all_lanes'):
                    for lane in road_network.get_all_lanes():
                        if hasattr(lane, 'shapely_polygon'):
                            self.all_lanes.append((lane, lane.shapely_polygon))
                else:
                    for start_node, end_dict in road_network.graph.items():
                        for end_node, lanes in end_dict.items():
                            for lane in lanes:
                                if hasattr(lane, 'shapely_polygon'):
                                    self.all_lanes.append((lane, lane.shapely_polygon))
        except (AttributeError, TypeError):
            # env lacks the MetaDrive runtime objects (e.g. NavSafe env=None
            # mode, or a stripped IsaacSim env). Fall through to the
            # scenario_data fallback below.
            pass

        # Path 2: scenario_data fallback. Build LaneProxy objects from
        # ``scenario_data['map_features']``. Used when env was unavailable
        # or its road network was empty. With ``use_fast_lanes`` the lanes
        # are FastLaneProxy (vectorised local_coordinates, same numerics).
        if not self.all_lanes and 'map_features' in scenario_data:
            if self.use_fast_lanes:
                from navsafe.evaluation.utils.lane_proxy_fast import (
                    build_fast_lanes_from_scenario,
                )
                self.all_lanes = build_fast_lanes_from_scenario(scenario_data)
            else:
                from navsafe.evaluation.utils.lane_proxy import (
                    build_lanes_from_scenario,
                )
                self.all_lanes = build_lanes_from_scenario(scenario_data)

        # Pre-compute lane bounds as numpy array for vectorized nearby queries
        if self.all_lanes:
            bounds = np.array([poly.bounds for _, poly in self.all_lanes])  # (num_lanes, 4)
            self.all_lane_bounds = bounds  # [minx, miny, maxx, maxy]
        else:
            self.all_lane_bounds = np.empty((0, 4))

        # Non-lane map surfaces CaRL's PDM-Closed reads from the nuPlan map
        # (planner path only): CARPARK_AREA polygons join the drivable-area
        # union (``get_drivable_area_map``), INTERSECTION polygons are the TTC
        # fault-cone predicate (``map_api.is_in_layer(rear_axle, INTERSECTION)``).
        # Emitted by ``py123d_scenario_description`` as polygon features.
        self.carpark_polygons: List[Polygon] = []
        self.intersection_polygons: List[Polygon] = []
        for feature in (scenario_data.get('map_features') or {}).values():
            ftype = str(feature.get('type', '')).upper()
            if ftype not in ('CARPARK_AREA', 'INTERSECTION'):
                continue
            ring = feature.get('polygon')
            if ring is None:
                continue
            ring = np.asarray(ring, dtype=np.float64)
            if ring.ndim != 2 or ring.shape[0] < 3:
                continue
            poly = Polygon(ring[:, :2])
            if not poly.is_valid:
                poly = poly.buffer(0.0)
            if poly.is_empty:
                continue
            (self.carpark_polygons if ftype == 'CARPARK_AREA'
             else self.intersection_polygons).append(poly)
        self.intersection_bounds = (
            np.array([poly.bounds for poly in self.intersection_polygons])
            if self.intersection_polygons else np.empty((0, 4)))

        # Coordinate calibration (world -> sim).
        # Three sub-paths:
        # - env present + valid SDC frame 0 + ``env.agent.position`` available:
        #   classic BridgeSim path; offset = sim_pos - world_pos.
        # - env present but SDC frame 0 invalid: cannot calibrate; offset = 0.
        # - env=None: trajectories are already in scenario-world coordinates,
        #   no offset is needed.
        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        if self.env is None:
            self.world_to_sim_offset = np.array([0.0, 0.0])
        elif sdc_track['state']['valid'][0]:
            try:
                start_pos_world = sdc_track['state']['position'][0][:2]
                start_pos_sim = self.env.agent.position
                self.world_to_sim_offset = np.array(start_pos_sim) - np.array(start_pos_world)
                if self.verbose:
                    print(f"[EPDMS Fast] Calibrated World->Sim Offset: {self.world_to_sim_offset}")
            except AttributeError:
                # env without ``.agent.position`` — treat like env=None.
                self.world_to_sim_offset = np.array([0.0, 0.0])
        else:
            self.world_to_sim_offset = np.array([0.0, 0.0])
            if self.verbose:
                print("[EPDMS Fast] Could not calibrate coordinates (Frame 0 invalid).")

        self.prev_frame_idx = None
        self._frame_data_cache = None
        # Invalidate the drivable-area union: the lane set just changed.
        self._drivable_area_proxy = None
        self._drivable_area_built = False
        # Re-arm the lane-less warning: it is per-scenario diagnostics.
        self._warned_no_lanes = False
        self._initialized = True
        if self.verbose:
            print(f"[EPDMS Fast] Initialized with {len(self.all_lanes)} lanes")

    # ---------------------------------------------------------------
    # Vectorized helpers
    # ---------------------------------------------------------------

    def _to_sim_frame(self, points: np.ndarray) -> np.ndarray:
        return points + self.world_to_sim_offset

    def _query_nearby_lanes_vec(self, x: float, y: float, radius: float) -> List[int]:
        """Return indices into self.all_lanes using vectorized bounds check."""
        assert self.all_lane_bounds is not None  # set in initialize()
        if len(self.all_lane_bounds) == 0:
            return []
        bounds = self.all_lane_bounds
        mask = ((bounds[:, 0] - radius < x) & (x < bounds[:, 2] + radius) &
                (bounds[:, 1] - radius < y) & (y < bounds[:, 3] + radius))
        return np.where(mask)[0].tolist()

    def _active_map_context(
        self, ego_state: Dict[str, Any]
    ) -> Tuple[Optional[frozenset[int]], Optional[DrivableAreaProxy]]:
        """CaRL's per-plan proximal drivable map around the current ego."""
        radius = self.map_radius_m
        if radius is None or radius <= 0.0:
            return None, None
        centre_world = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        centre = centre_world + self.world_to_sim_offset
        point = Point(float(centre[0]), float(centre[1]))
        coarse = self._query_nearby_lanes_vec(
            float(centre[0]), float(centre[1]), float(radius)
        )
        active = frozenset(
            idx for idx in coarse
            if self.all_lanes[idx][1].distance(point) <= float(radius)
        )
        # ``get_drivable_area_map``: lanes of the proximal roadblocks and
        # connectors PLUS the proximal CARPARK_AREA polygons.
        carparks = [poly for poly in getattr(self, 'carpark_polygons', [])
                    if poly.distance(point) <= float(radius)]
        area = DrivableAreaProxy.from_lane_polygons(
            [self.all_lanes[idx][1] for idx in active] + carparks
        )
        return active, area

    def _filter_agents_to_map_radius(
        self,
        agents_per_t: list,
        ego_state: Dict[str, Any],
    ) -> list:
        """Filter on each actor's current center, as PDMObservation does."""
        radius = self.map_radius_m
        if radius is None or radius <= 0.0 or not agents_per_t:
            return agents_per_t
        centre = (
            np.asarray(ego_state["position"], dtype=np.float64)[:2]
            + self.world_to_sim_offset
        )
        keep = {
            item[4]
            for item in agents_per_t[0]
            if math.hypot(float(item[2]) - float(centre[0]),
                          float(item[3]) - float(centre[1])) <= float(radius)
        }
        return [
            [item for item in timestep if item[4] in keep]
            for timestep in agents_per_t
        ]

    @property
    def gt_stride(self) -> int:
        """Frames the world advances per scored pose. NEVER less than 1.

        Callers that re-tune ``planner_dt`` after ``initialize`` (e.g.
        ``pdm_closed_planner.scoring.PDMScorer.initialize``, which sets it to
        ``cfg.sim_dt``) get the clamped stride automatically -- recomputing
        the division inline (or caching a stale stride) is how the clamp got
        bypassed on the planner's own path while the evaluator's looked fine.

        stride 0 means ``sim_frame = frame_idx + t*0``: every scored pose reads
        the agents at the SAME frame, i.e. the world is frozen for the whole
        horizon and nc/ttc are meaningless. That was live for every py123d run
        (scenario_dt misparsed as 3.16e14). It is also reachable legitimately
        when planner_dt < scenario_dt, so clamp rather than raise.

        Thin wrapper over the shared :func:`gt_stride` derivation — this used
        to be a local reimplementation, i.e. one more copy of the exact
        per-scorer drift the shared helper exists to kill.
        """
        return _gt_stride(self.planner_dt, self.scenario_dt)

    # ---------------------------------------------------------------
    # Drivable area (DAC) — shared with EPDMSLiveScorer
    # ---------------------------------------------------------------

    def _drivable_area(self) -> Optional[DrivableAreaProxy]:
        """The prepared union-of-lanes drivable surface, built once.

        Lazy rather than eager in ``initialize`` so a caller that assigns
        ``all_lanes`` directly (tests, ad-hoc probes) still gets a proxy;
        ``initialize`` resets the memo when the lane set changes.
        """
        if not self._drivable_area_built:
            self._drivable_area_proxy = DrivableAreaProxy.from_lane_polygons(
                poly for _, poly in self.all_lanes)
            self._drivable_area_built = True
        return self._drivable_area_proxy

    def _corners_in_drivable_area(self, x: float, y: float,
                                  heading: float) -> bool:
        """All four corners of the ego box at this pose inside drivable area."""
        return self._footprint_in_drivable_area(
            self._get_ego_polygon(x, y, heading), x, y)

    def _footprint_in_drivable_area(self, ego_poly: Polygon,
                                    x: float, y: float) -> bool:
        """``_corners_in_drivable_area`` for an already-built ego polygon.

        The DAC loop builds one polygon per horizon step for NC/TLC anyway,
        so this entry point avoids rebuilding it. ``x``/``y`` are only used
        by the per-lane fallback's nearby-lane prefilter.
        """
        return corners_in_drivable_area(
            ego_poly.exterior.coords[:-1],
            self._drivable_area(),
            lambda: (self.all_lanes[idx][1]
                     for idx in self._query_nearby_lanes_vec(x, y, 15.0)),
        )

    def _get_ego_polygon(self, x: float, y: float, heading: float) -> Polygon:
        return self._get_agent_polygon(
            x, y, heading, self.ego_length_m, self.ego_width_m
        )

    def _get_ego_aabb(self, x: float, y: float) -> Tuple[float, float, float, float]:
        """Fast AABB for ego vehicle (conservative circle-based)."""
        half_diag = 0.5 * math.hypot(self.ego_length_m, self.ego_width_m)
        return (x - half_diag, y - half_diag, x + half_diag, y + half_diag)

    @staticmethod
    def _get_agent_polygon(x: float, y: float, heading: float,
                           length: float, width: float) -> Polygon:
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        ex, ey = 0.5 * length, 0.5 * width
        corners = np.empty((4, 2))
        for i, (cx, cy) in enumerate(((ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey))):
            corners[i, 0] = x + cx * cos_h - cy * sin_h
            corners[i, 1] = y + cx * sin_h + cy * cos_h
        return Polygon(corners)

    @staticmethod
    def _track_dims(track: Dict[str, Any], sim_frame: int) -> Tuple[float, float]:
        """Track dims via the shared resolver (``navsafe.core.track_dims``).

        Shared with the planner's agent predictions so the emergency
        brake and the scorer agree on collision geometry.
        """
        return track_dims(
            track, sim_frame, fallback=(VEHICLE_LENGTH, VEHICLE_WIDTH)
        )

    @staticmethod
    def _aabb_overlap(a: Tuple, b: Tuple) -> bool:
        """Check if two AABBs (minx, miny, maxx, maxy) overlap."""
        return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]

    def _get_best_lane(self, x: float, y: float, heading: float,
                       nearby_indices: List[int], speed: Optional[float] = None,
                       *, heading_gate: bool = True):
        """Nearest lane to ``(x, y)``.

        With ``heading_gate`` (default) only lanes within ±π/2 of
        ``heading`` qualify — right for lane-keeping style queries.
        Pass ``heading_gate=False`` for queries that must be able to
        return an opposing lane (e.g. DDC's wrong-way detection, which
        is vacuous if the candidate set is pre-filtered to aligned
        lanes).
        """
        if not nearby_indices:
            return None
        candidates = []
        is_stopped = (speed is not None) and (speed < 1.0)
        pos = np.array([x, y])

        for idx in nearby_indices:
            lane = self.all_lanes[idx][0]
            dist = lane.distance(pos)
            if not heading_gate:
                # Distance-only query: the centerline projection and
                # heading comparison below are never consumed, so skip
                # them (they dominate the per-lane cost).
                candidates.append((lane, dist))
                continue
            s, r = lane.local_coordinates(pos)
            s_clamped = max(0, min(s, lane.length))
            lane_heading_vec = lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi

            if is_stopped or (abs(diff) < (np.pi / 2)):
                candidates.append((lane, dist))

        if candidates:
            return min(candidates, key=lambda c: c[1])[0]
        return None

    # ---------------------------------------------------------------
    # Pre-computation (called once per select_best, shared by all N)
    # ---------------------------------------------------------------

    def _precompute_frame_data(self, frame_idx: int, horizon: int):
        """Pre-compute per-timestep agent polygons and traffic light states.

        Builds ``horizon + 1`` timesteps (indices ``0..horizon``): the scored
        state arrays carry the prepended current pose PLUS the ``horizon``
        waypoints, and ``_calculate_metrics`` checks every one of them —
        including the terminal (4.0 s) pose — against the world (upstream
        ``train_pdm_scorer`` convention: ``range(num_poses + 1)``). Timesteps
        past the scenario's end hold the last logged actor poses and light
        states (the replay world is frozen there)."""
        key = (int(frame_idx), int(horizon), int(self.gt_stride))
        if self._frame_data_cache is not None and self._frame_data_cache[0] == key:
            return self._frame_data_cache[1]
        assert self.scenario_data is not None  # set in initialize()
        tracks = self.scenario_data['tracks']
        scenario_length = self.scenario_data['length']

        # Agent data: list of lists, one per timestep
        # Each entry: (poly, aabb, x, y, obj_id). obj_id is carried so NC can
        # exclude agents already in contact with the ego at t=0 — a collision
        # that exists before the candidate starts was not caused by it.
        agents_per_t = []
        for t in range(horizon + 1):
            sim_frame = frame_idx + (t * self.gt_stride)
            agents_at_t = []
            if scenario_length > 0:
                # The evaluator holds replay actors at their final logged
                # poses after the bundle ends. The verifier must score the
                # same occupied world instead of switching to an empty one.
                actor_frame = min(sim_frame, scenario_length - 1)
                for obj_id, track in tracks.items():
                    if obj_id == self.sdc_id:
                        continue
                    # Static traffic objects are physical collision actors in
                    # NavSafe too. Excluding them gave every proposal a 1.0
                    # NC/TTC at 20cc... f235, immediately before the driven
                    # trajectory hit a logged cone at f240.
                    if str(track.get('type', '')).upper() not in (
                            COLLIDABLE_ACTOR_TYPES):
                        continue
                    pos_arr = track['state']['position']
                    valid_arr = track['state']['valid']
                    if (actor_frame >= len(pos_arr)
                            or not valid_arr[actor_frame]):
                        continue
                    obj_pos_world = pos_arr[actor_frame][:2]
                    ax, ay = obj_pos_world + self.world_to_sim_offset
                    ah = track['state']['heading'][actor_frame]
                    a_len, a_wid = self._track_dims(track, actor_frame)
                    poly = self._get_agent_polygon(ax, ay, ah, a_len, a_wid)
                    aabb = poly.bounds  # (minx, miny, maxx, maxy)
                    agents_at_t.append((poly, aabb, ax, ay, obj_id))
            agents_per_t.append(agents_at_t)

        # Traffic light red lane IDs per timestep. Past the end of an entry's
        # log the lookup CLAMPS to the last logged state, exactly like the
        # live scorer's _check_tlc_live and the actor poses above: the
        # replay world is frozen at its last logged frame, so the light is
        # too. (It used to read "no state -> not red" out there, so the
        # verifier cleared candidates the live metric still charged.)
        red_lanes_per_t = [
            _red_lane_ids(self.traffic_lights, frame_idx + (t * self.gt_stride))
            for t in range(horizon + 1)]

        self._frame_data_cache = (key, (agents_per_t, red_lanes_per_t))
        return agents_per_t, red_lanes_per_t

    def _live_agents_per_t(
        self, agent_states: List[Dict[str, Any]], horizon: int,
    ) -> List[List[Tuple[Any, Tuple[float, float, float, float],
                              float, float, str]]]:
        """Forecast the evaluator's live actor snapshot for candidate scoring.

        Semi-reactive actors can diverge from their logged tracks, while the
        environment snapshot already contains *all* currently collidable
        actors (including held cones/barriers).  When that snapshot is
        present it is therefore authoritative, just as it is for the planner,
        controller collision probe, renderer, and live metric.  Missing the
        key retains historical log scoring; an explicitly empty snapshot
        means the live world has no actors.
        """
        times = np.arange(horizon + 1, dtype=np.float64) * self.planner_dt
        out: List[List[Tuple[Any, Tuple[float, float, float, float],
                                  float, float, str]]] = [
            [] for _ in range(horizon + 1)
        ]
        # Reference admission order: the 50 m radius, then the per-class
        # nearest-k caps, both on centre distance at the current frame. The
        # radius is re-applied by the caller (``_filter_agents_to_map_radius``);
        # the caps need the types, so they live here.
        ego_xy: Optional[np.ndarray] = None
        for state in agent_states:
            if state.get("is_ego", False):
                pos = np.asarray(state.get("position", []), dtype=np.float64).reshape(-1)
                if pos.size >= 2 and np.isfinite(pos[:2]).all():
                    ego_xy = pos[:2]
                break
        admitted: Optional[set] = None
        if ego_xy is not None:
            candidates = []
            for index, state in enumerate(agent_states):
                if state.get("is_ego", False) or not state.get("valid", True):
                    continue
                pos = np.asarray(state.get("position", []), dtype=np.float64).reshape(-1)
                if pos.size < 2 or not np.isfinite(pos[:2]).all():
                    continue
                dist = float(math.hypot(pos[0] - ego_xy[0], pos[1] - ego_xy[1]))
                radius = self.map_radius_m
                if radius is not None and radius > 0.0 and dist > float(radius):
                    continue
                candidates.append((index, str(state.get("type", "VEHICLE")).upper(), dist))
            admitted = nearest_k_admitted([
                (str(i), t, d) for i, t, d in candidates])
        seen_ids: Dict[str, int] = {}
        for index, state in enumerate(agent_states):
            if state.get("is_ego", False) or not state.get("valid", True):
                continue
            if admitted is not None and str(index) not in admitted:
                continue
            actor_type = str(state.get("type", "VEHICLE")).upper()
            if actor_type not in COLLIDABLE_ACTOR_TYPES:
                continue
            position = np.asarray(
                state.get("position", []), dtype=np.float64).reshape(-1)
            if position.size < 2 or not np.isfinite(position[:2]).all():
                continue
            heading = float(state.get("heading", 0.0))
            velocity = np.asarray(
                state.get("velocity", []), dtype=np.float64).reshape(-1)
            if velocity.size >= 2:
                vx, vy = float(velocity[0]), float(velocity[1])
            elif velocity.size == 1:
                speed = float(velocity[0])
                vx = speed * math.cos(heading)
                vy = speed * math.sin(heading)
            else:
                vx = vy = 0.0
            length = float(state.get("length", VEHICLE_LENGTH))
            width = float(state.get("width", VEHICLE_WIDTH))
            if (not np.isfinite([heading, vx, vy, length, width]).all()
                    or length <= 0.0 or width <= 0.0):
                continue
            # Same forecast the planner's lead search uses: |v| along the box
            # heading for agents, frozen for static types (CaRL
            # ``PDMObjectManager``), see ``core.agent_forecast``.
            vx, vy = forecast_velocity(vx, vy, heading, actor_type)
            raw_id = str(state.get("id", index))
            duplicate = seen_ids.get(raw_id, 0)
            seen_ids[raw_id] = duplicate + 1
            actor_id = raw_id if duplicate == 0 else f"{raw_id}#{duplicate}"
            for t, seconds in enumerate(times):
                ax = float(position[0] + vx * seconds
                           + self.world_to_sim_offset[0])
                ay = float(position[1] + vy * seconds
                           + self.world_to_sim_offset[1])
                poly = self._get_agent_polygon(
                    ax, ay, heading, length, width)
                out[t].append((poly, poly.bounds, ax, ay, actor_id))
        return out

    # ---------------------------------------------------------------
    # Batched trajectory state computation
    # ---------------------------------------------------------------

    def _get_all_trajectory_states(self, all_paths: np.ndarray, dt: float):
        """
        Vectorized trajectory states for all N candidates.

        Args:
            all_paths: (N, T, 2) array of paths in sim frame
            dt: time step between waypoints

        Returns:
            dict of arrays, each (N, T):
                x, y, heading, speed, acceleration, jerk, yaw_rate
        """
        N, T, _ = all_paths.shape
        paths = all_paths.copy()

        # Savgol filtering per candidate (can't fully vectorize due to per-row filter)
        window_size = min(7, T if T % 2 != 0 else T - 1)
        if window_size >= 3:
            for i in range(N):
                try:
                    paths[i, :, 0] = savgol_filter(paths[i, :, 0], window_length=window_size, polyorder=2)
                    paths[i, :, 1] = savgol_filter(paths[i, :, 1], window_length=window_size, polyorder=2)
                except ValueError:
                    # The window/polyorder guard above rules out parameter
                    # errors; the only remaining ValueError is scipy's
                    # "array must not contain infs or NaNs" on a non-finite
                    # row (e.g. a non-finite ego position — non-finite
                    # CANDIDATES are already filtered in score_candidates).
                    # Keep the historical behavior (row stays unsmoothed)
                    # but say so instead of swallowing silently.
                    logger.warning(
                        "[EPDMS Fast] savgol_filter rejected candidate %d "
                        "(non-finite path values); row left unsmoothed", i)

        # Vectorized finite differences across all candidates: (N, T-1, 2)
        dpath = np.diff(paths, axis=1) / dt
        # Pad last row: (N, T, 2)
        vel_vec = np.concatenate([dpath, dpath[:, -1:, :]], axis=1)

        heading = np.arctan2(vel_vec[:, :, 1], vel_vec[:, :, 0])  # (N, T)
        heading = np.unwrap(heading, axis=1)

        dacc = np.diff(vel_vec, axis=1) / dt
        acc_vec = np.concatenate([dacc, dacc[:, -1:, :]], axis=1)
        acc_mag = np.linalg.norm(acc_vec, axis=2)  # (N, T)

        djerk = np.diff(acc_vec, axis=1) / dt
        jerk_vec = np.concatenate([djerk, djerk[:, -1:, :]], axis=1)
        jerk_mag = np.linalg.norm(jerk_vec, axis=2)  # (N, T)

        yaw_rate_raw = np.diff(heading, axis=1) / dt
        yaw_rate = np.concatenate([yaw_rate_raw, yaw_rate_raw[:, -1:]], axis=1)  # (N, T)

        speed = np.linalg.norm(vel_vec, axis=2)  # (N, T)

        # Signed longitudinal kinematics for the nuPlan comfort bounds:
        # lon accel = d(speed)/dt, lon jerk = d(lon accel)/dt, and yaw
        # accel = d(yaw rate)/dt.
        dlon = np.diff(speed, axis=1) / dt
        lon_accel = np.concatenate([dlon, dlon[:, -1:]], axis=1)
        dlon_jerk = np.diff(lon_accel, axis=1) / dt
        lon_jerk = np.concatenate([dlon_jerk, dlon_jerk[:, -1:]], axis=1)
        dyaw_acc = np.diff(yaw_rate, axis=1) / dt
        yaw_accel = np.concatenate([dyaw_acc, dyaw_acc[:, -1:]], axis=1)

        stopped_mask = speed < STOPPED_SPEED_THRESHOLD
        acc_mag[stopped_mask] = 0.0
        jerk_mag[stopped_mask] = 0.0
        yaw_rate[stopped_mask] = 0.0
        lon_accel[stopped_mask] = 0.0
        lon_jerk[stopped_mask] = 0.0
        yaw_accel[stopped_mask] = 0.0

        return {
            "x": paths[:, :, 0],  # (N, T)
            "y": paths[:, :, 1],
            "heading": heading,
            "speed": speed,
            "acceleration": acc_mag,
            "jerk": jerk_mag,
            "yaw_rate": yaw_rate,
            "lon_accel": lon_accel,
            "lon_jerk": lon_jerk,
            "yaw_accel": yaw_accel,
        }

    # ---------------------------------------------------------------
    # Per-candidate metric evaluation (with caches + precomputed data)
    # ---------------------------------------------------------------

    def _calculate_metrics(self, states: dict, horizon: int, frame_idx: int,
                           ego_state: Optional[Dict[str, Any]] = None,
                           n_execute: Optional[int] = None,
                           agents_per_t: Optional[list] = None,
                           red_lanes_per_t: Optional[list] = None,
                           carl_collision_classifier: bool = False,
                           active_lane_indices: Optional[frozenset[int]] = None,
                           active_drivable_area: Optional[DrivableAreaProxy] = None,
                           direction_route_lane_ids: Optional[FrozenSet[str]] = None) -> dict:
        """EPDMS terms for ONE candidate over ``horizon + 1`` scored poses.

        Contract (terminal-pose fix): ``states`` arrays carry
        ``horizon + 1`` entries — the prepended current pose (index 0) plus
        the ``horizon`` planned waypoints — and ``agents_per_t`` /
        ``red_lanes_per_t`` (from :meth:`_precompute_frame_data`) carry
        ``horizon + 1`` timesteps. Every per-pose safety loop (DAC, NC, TTC,
        DDC, LK) runs ``range(horizon + 1)`` so the terminal (4.0 s)
        pose — the one EP's progress is measured to — is checked against the
        world too, matching the vendored upstream ``train_pdm_scorer``'s
        ``range(num_poses + 1)`` convention. Previously the loops stopped at
        ``horizon``: a candidate could earn full progress credit for ending
        on top of a pedestrian or off the drivable surface. DAC's
        denominator is ``horizon + 1`` accordingly, so an early violation's
        per-pose weight shifts slightly even when the terminal pose is clean
        (accepted — the upstream convention). Index 0 (the current pose,
        identical across candidates) stays in the loops, also per upstream.
        Degenerate ``horizon == 0`` (no waypoints, unreachable from any
        production caller) keeps its historical ``dac = 1.0`` carve-out
        rather than scoring the lone current pose.
        """
        metrics: Dict[str, Any] = {
            "nc": 1.0, "dac": 1.0, "ddc": 1.0, "tlc": 1.0,
            "ep": 0.0, "ttc": 1.0, "lk": 1.0, "hc": 1.0, "ec": 1.0,
            # Planner-facing metadata. Ordinary EPDMS aggregation ignores it.
            "collision_time_s": float("inf"),
            "collision_actor_id": None,
            "ttc_actor_id": None,
            "ttc_time_s": None,
            "ttc_probe_time_s": None,
            "ttc_projection_s": None,
            # Every at-fault contact id, earliest first. Upstream's NC is the
            # minimum over all of them, which the wrapper cannot compute from
            # a single id (pdm_scorer.py:331-337).
            "collision_at_fault_ids": [],
        }
        # The scored arrays must be exactly the terminal-pose-inclusive
        # length (upstream asserts ``num_poses + 1`` the same way). A LONGER
        # array would silently reintroduce the blind spot's mirror image:
        # EP reads ``states['x'][-1]`` while the safety loops stop at
        # ``horizon``.
        assert len(states['x']) == horizon + 1, (
            f"states arrays carry {len(states['x'])} entries; need exactly "
            f"horizon+1 = {horizon + 1} (prepended pose + waypoints)")

        # --- Pre-build caches for this candidate ---
        # Ego polygons: create once, reuse for DAC, NC, TLC. Index
        # ``horizon`` is the terminal pose; its heading/speed come from the
        # padded (repeated-last-segment) finite differences in
        # _get_all_trajectory_states — an approximation we accept.
        ego_polys = [self._get_ego_polygon(states['x'][t], states['y'][t], states['heading'][t])
                     for t in range(horizon + 1)]
        ego_aabbs = [self._get_ego_aabb(states['x'][t], states['y'][t])
                     for t in range(horizon + 1)]

        # Reference occupancy maps are sampled every ``observation_sample_res``
        # steps (``PDMObservation``: 2 → 0.2 s) and ``__getitem__(time_idx)``
        # reads map ``time_idx // res``, so NC/TTC at an odd 0.1 s step see
        # the agents where they were one step earlier. The planner's scorer
        # sets ``observation_sample_res = 2`` for bit-parity with upstream
        # (measured: NC 13 / TTC 16 of 1800 synthetic proposals differ
        # without it); every other consumer keeps exact per-step sampling.
        obs_res = max(1, int(getattr(self, 'observation_sample_res', 1)))
        n_obs = len(agents_per_t) if agents_per_t is not None else 0

        def _obs_index(t: int) -> int:
            return min((t // obs_res) * obs_res, n_obs - 1)

        # Nearby lanes cache: key = (t, radius) -> list of lane indices
        nearby_cache = {}

        def get_nearby(t, radius):
            key = (t, radius)
            if key not in nearby_cache:
                found = self._query_nearby_lanes_vec(
                    states['x'][t], states['y'][t], radius)
                nearby_cache[key] = (
                    found if active_lane_indices is None
                    else [idx for idx in found if idx in active_lane_indices]
                )
            return nearby_cache[key]

        def footprint_in_drivable(t: int) -> bool:
            if active_lane_indices is None:
                return self._footprint_in_drivable_area(
                    ego_polys[t], states['x'][t], states['y'][t]
                )
            return corners_in_drivable_area(
                ego_polys[t].exterior.coords[:-1],
                active_drivable_area,
                lambda: (
                    self.all_lanes[idx][1] for idx in get_nearby(t, 15.0)
                ),
            )

        ego_area_cache: Dict[int, Tuple[bool, bool, bool]] = {}

        def ego_area_flags(t: int) -> Tuple[bool, bool, bool]:
            """CaRL MULTIPLE_LANES, NON_DRIVABLE and INTERSECTION flags."""
            if t in ego_area_cache:
                return ego_area_cache[t]
            corners = [
                Point(xy) for xy in ego_polys[t].exterior.coords[:4]
            ]
            lane_corner_counts = []
            nearby = get_nearby(t, 15.0)
            for lane_idx in nearby:
                lane_poly = self.all_lanes[lane_idx][1]
                lane_corner_counts.append(sum(
                    bool(lane_poly.contains(corner)) for corner in corners
                ))
            in_multiple_lanes = (
                sum(count > 0 for count in lane_corner_counts) > 1
                and not any(count == 4 for count in lane_corner_counts)
            )
            in_nondrivable = not footprint_in_drivable(t)
            heading = float(states['heading'][t])
            rear_x = float(states['x'][t]) - (
                self.rear_axle_to_center_m * math.cos(heading)
            )
            rear_y = float(states['y'][t]) - (
                self.rear_axle_to_center_m * math.sin(heading)
            )
            rear_point = Point(rear_x, rear_y)
            intersections = getattr(self, 'intersection_polygons', [])
            if intersections:
                # ``map_api.is_in_layer(ego_rear_axle, INTERSECTION)`` on the
                # junction footprint itself (gaps between connectors included).
                b = self.intersection_bounds
                hits = np.where((b[:, 0] <= rear_x) & (rear_x <= b[:, 2])
                                & (b[:, 1] <= rear_y) & (rear_y <= b[:, 3]))[0]
                in_intersection = any(
                    intersections[i].covers(rear_point) for i in hits)
            else:
                # Fallback for maps without intersection polygons: the
                # connector lanes flagged ``is_intersection``.
                in_intersection = any(
                    bool(getattr(self.all_lanes[idx][0], "is_intersection", False))
                    and self.all_lanes[idx][1].covers(rear_point)
                    for idx in nearby
                )
            value = (in_multiple_lanes, in_nondrivable, in_intersection)
            ego_area_cache[t] = value
            return value

        # 1. DAC (ratio of scored poses — current pose + all waypoints
        # incl. the terminal one — inside drivable area).
        # Scored against the SAME union+buffer surface as the reported
        # metric (``corners_in_drivable_area``). This used to be an inline
        # per-lane-piece test with no seam tolerance, so a candidate whose
        # footprint overhangs its 3.5 m lane strip by a few centimetres was
        # charged a violation the metric of record does not see: 11.5% of
        # logged poses shifted 1 m right and 3.1% shifted 1 m left over 30
        # scenes / 4748 frames, and 0% of un-shifted ones. Since dac
        # MULTIPLIES, that devalued or zeroed the lateral third of the
        # proposal grid against the metric the planner is judged by (and is
        # the "dac 36/36" standstill deadlock in BRAKE_DEADLOCK_REVIEW.md).
        dac_valid = 0
        for t in range(horizon + 1):
            if footprint_in_drivable(t):
                dac_valid += 1
        metrics['dac'] = dac_valid / (horizon + 1) if horizon > 0 else 1.0

        # 2. NC (with AABB pre-filter + pre-computed agent polys).
        # At-fault approximation of nuPlan's classification: front
        # collisions (agent centre ahead of ego centre) are at fault;
        # rear-end *by* the other agent and pure side contact (agent
        # alongside — e.g. an agent merging into the ego) are not.
        # The previous ``longitudinal < -1.0`` cutoff faulted every
        # sideswipe the ego did not initiate.
        # Agents already overlapping the ego at t=0 are PRE-EXISTING
        # contacts: every candidate starts from the same ego pose, so no
        # candidate caused them and none can avoid them. Charging them
        # zeroed nc on every proposal of a frame (observed at b040d87a
        # f35-f45: a stopped ego "at fault" for 11 consecutive frames
        # against a lead that was already touching it and receding at
        # 3.5 m/s). Upstream's classifier excludes these via
        # ``already_collided_ids``; the inline approximation here lost it.
        # Shared by NC and TTC: excluding them from nc alone left every
        # candidate of such a frame losing its ttc term instead — the same
        # artifact through the other multiplier (found in review; the
        # smoothed 0.31 m/s profile keeps the ttc loop live even for a
        # genuinely stationary ego).
        preexisting_ids = set()
        if ego_state is not None:
            historical = ego_state.get("_pdm_collided_track_ids", ())
            if historical is not None:
                try:
                    preexisting_ids.update(historical)
                except TypeError:
                    logger.warning(
                        "invalid _pdm_collided_track_ids=%r; ignoring",
                        historical,
                    )
        if agents_per_t and agents_per_t[0]:
            for a_poly, a_aabb, _ax, _ay, a_id in agents_per_t[0]:
                if (self._aabb_overlap(ego_aabbs[0], a_aabb)
                        and ego_polys[0].intersects(a_poly)):
                    preexisting_ids.add(a_id)

        if agents_per_t is not None:
            # At-fault requires the EGO to be moving AT COLLISION TIME
            # (upstream nuPlan classifies by ego speed when contact happens).
            # ``states['speed'][t]`` cannot decide that: the savgol-smoothed
            # profile reads ~0.31 m/s for a genuinely stationary ego
            # (measured), so the 0.05 m/s threshold never fired. The ego's
            # REAL frame-0 speed is only right at t=0 — frozen across the
            # horizon it kept a hard-braking candidate "moving" at collision
            # time, and the old moved_m < 0.5 m guard exempted any contact a
            # creep made within its first 0.5 m. ``at_fault_ego_speed``
            # derives a per-timestep speed from the candidate's own scored
            # positions instead (min of segment and net-displacement speed —
            # see its module docstring for the savgol-floor robustness).
            ego_v_actual = None
            if ego_state is not None:
                raw_speed = ego_state.get('speed')
                if raw_speed is not None:
                    try:
                        ego_v_actual = abs(float(raw_speed))
                    except (TypeError, ValueError):
                        ego_v_actual = None

            # FIRST-CONTACT latch (upstream nuPlan's already_collided_ids,
            # generalized): a collision is classified once, at first
            # contact. A not-at-fault first contact (stopped ego, rear-end
            # by the agent, or the agent OVERTAKING the ego from behind)
            # latches the agent into ``excluded_ids`` for the remaining
            # horizon — without the latch, a faster replay agent plowing
            # through a slower-than-log ego was re-classified as an
            # at-fault FRONT collision once its centre swept past the
            # ego's centre. The overtaking test (agent's along-heading
            # speed exceeds the ego's) covers the coarse-sampling case
            # where 0.5 s samples only ever observe the front phase of a
            # fast pass-through.
            actor_speeds = _actor_speeds_from_ego_state(ego_state)
            excluded_ids = set(preexisting_ids)
            prev_agent_pos: Dict[Any, Tuple[float, float]] = {}
            # Every at-fault contact, earliest first. Upstream scores the
            # minimum over all of them; the wrapper needs the actor types to
            # resolve that, so the ids travel out with the metrics.
            at_fault_ids: List[str] = metrics['collision_at_fault_ids']
            for t in range(horizon + 1):
                # PDM-Closed supplies the coherent tracker-simulated state
                # array that CaRL scores.  In that path, use its instantaneous
                # speed directly: CaRL reads hypot(VELOCITY_X, VELOCITY_Y),
                # not a displacement estimate.  Generic XY-only candidates
                # retain the robust fallback because SavGol gives a truly
                # stationary path a measured ~0.31 m/s velocity floor.
                ego_v = (
                    abs(float(states['speed'][t]))
                    if carl_collision_classifier
                    else at_fault_ego_speed(
                        states['x'], states['y'], t, self.planner_dt,
                        ego_v0=ego_v_actual)
                )
                ego_aabb = ego_aabbs[t]
                ego_hx = math.cos(states['heading'][t])
                ego_hy = math.sin(states['heading'][t])
                curr_agent_pos: Dict[Any, Tuple[float, float]] = {}
                for agent_poly, agent_aabb, ax, ay, a_id in agents_per_t[_obs_index(t)]:
                    curr_agent_pos[a_id] = (ax, ay)
                    if a_id in excluded_ids:
                        continue
                    # AABB pre-filter
                    if not self._aabb_overlap(ego_aabb, agent_aabb):
                        continue
                    if ego_polys[t].intersects(agent_poly):
                        dx, dy = ax - states['x'][t], ay - states['y'][t]
                        longitudinal_dist = dx * ego_hx + dy * ego_hy
                        if carl_collision_classifier:
                            # Exact CaRL ordering from pdm_scorer_utils, via
                            # _carl_collision_type so the branch order and the
                            # 150 deg rear cone are unit-testable against the
                            # reference rather than inlined here.
                            # Upstream reads the tracked object's OWN
                            # velocity (is_track_stopped), not a positional
                            # difference, so prefer the logged velocity and
                            # keep the finite difference only as a fallback.
                            actor_speed = actor_speeds.get(str(a_id))
                            if actor_speed is None and a_id in prev_agent_pos:
                                pax, pay = prev_agent_pos[a_id]
                                actor_speed = math.hypot(
                                    ax - pax, ay - pay
                                ) / self.planner_dt
                            half_l = 0.5 * self.ego_length_m
                            half_w = 0.5 * self.ego_width_m
                            front_x = states['x'][t] + half_l * ego_hx
                            front_y = states['y'][t] + half_l * ego_hy
                            left_x, left_y = -ego_hy, ego_hx
                            front_edge = LineString([
                                (front_x + half_w * left_x,
                                 front_y + half_w * left_y),
                                (front_x - half_w * left_x,
                                 front_y - half_w * left_y),
                            ])
                            collision_type = _carl_collision_type(
                                ego_speed=ego_v,
                                rear_x=(states['x'][t]
                                        - self.rear_axle_to_center_m * ego_hx),
                                rear_y=(states['y'][t]
                                        - self.rear_axle_to_center_m * ego_hy),
                                ego_heading=float(states['heading'][t]),
                                actor_x=ax,
                                actor_y=ay,
                                actor_speed=actor_speed,
                                front_edge_hits=front_edge.intersects(
                                    agent_poly),
                            )
                            if collision_type == CARL_ACTIVE_LATERAL:
                                (in_multiple_lanes,
                                 in_nondrivable,
                                 _in_intersection) = ego_area_flags(t)
                                straddling_or_offroad = bool(
                                    in_multiple_lanes or in_nondrivable)
                            else:
                                straddling_or_offroad = False
                            is_at_fault = _carl_collision_is_at_fault(
                                collision_type,
                                straddling_or_offroad=straddling_or_offroad)
                        else:
                            is_at_fault = True
                            if ego_v < STOPPED_SPEED_THRESHOLD:
                                is_at_fault = False
                            if longitudinal_dist <= 0.0:
                                is_at_fault = False
                        if (not carl_collision_classifier and is_at_fault
                                and a_id in prev_agent_pos):
                            pax, pay = prev_agent_pos[a_id]
                            agent_v_long = (
                                (ax - pax) * ego_hx + (ay - pay) * ego_hy
                            ) / self.planner_dt
                            if agent_v_long > ego_v + 0.5:
                                # Overlapping agent moving faster than the
                                # ego along the ego's heading ran over the
                                # ego from behind — not the ego's fault.
                                is_at_fault = False
                        if is_at_fault:
                            metrics['nc'] = 0.0
                            if not at_fault_ids:
                                metrics['collision_time_s'] = (
                                    t * self.planner_dt)
                                metrics['collision_actor_id'] = str(a_id)
                            if str(a_id) not in at_fault_ids:
                                at_fault_ids.append(str(a_id))
                            if not carl_collision_classifier:
                                if t * self.planner_dt <= TTC_HORIZON:
                                    metrics['ttc'] = 0.0
                                    metrics['ttc_actor_id'] = str(a_id)
                                    metrics['ttc_time_s'] = t * self.planner_dt
                                    metrics['ttc_probe_time_s'] = t * self.planner_dt
                                    metrics['ttc_projection_s'] = 0.0
                                break
                            # CaRL path: upstream takes the MINIMUM over every
                            # at-fault collision (pdm_scorer.py:331-337), so a
                            # 0.5 static hit must not mask a later 0.0 agent
                            # hit. Keep scanning; the wrapper resolves the min
                            # from ``collision_at_fault_ids``. Upstream's
                            # _calculate_ttc never reads the collision term, so
                            # nothing is written to ``ttc`` here.
                            continue
                        # Not-at-fault first contact — latch.
                        excluded_ids.add(a_id)
                prev_agent_pos = curr_agent_pos
                if metrics['nc'] == 0.0 and not carl_collision_classifier:
                    break

        if agents_per_t is not None:
            ttc_taus = (0.0, 0.3, 0.6, 0.9)
            ttc_excluded_ids = set(preexisting_ids)
            for t in range(horizon + 1):
                ego_v = (
                    abs(float(states['speed'][t]))
                    if carl_collision_classifier
                    else at_fault_ego_speed(
                        states['x'], states['y'], t, self.planner_dt,
                        ego_v0=ego_v_actual,
                    )
                )
                if ego_v < PDM_STOPPED_SPEED_THRESHOLD:
                    continue
                hx = math.cos(states['heading'][t])
                hy = math.sin(states['heading'][t])
                rear_x = (
                    float(states['x'][t])
                    - self.rear_axle_to_center_m * hx
                )
                rear_y = (
                    float(states['y'][t])
                    - self.rear_axle_to_center_m * hy
                )
                (in_multiple_lanes,
                 in_nondrivable,
                 in_intersection) = ego_area_flags(t)
                widened_cone = bool(
                    in_multiple_lanes or in_nondrivable or in_intersection
                )
                found = False
                for tau in ttc_taus:
                    future_step = int(round(tau / self.planner_dt))
                    actor_t = t + future_step
                    if actor_t >= len(agents_per_t):
                        continue
                    px = states['x'][t] + hx * ego_v * tau
                    py = states['y'][t] + hy * ego_v * tau
                    proj_poly = None
                    proj_aabb = self._get_ego_aabb(px, py)
                    for (agent_poly, agent_aabb, ax, ay,
                         a_id) in agents_per_t[_obs_index(actor_t)]:
                        if a_id in ttc_excluded_ids:
                            continue
                        if not self._aabb_overlap(proj_aabb, agent_aabb):
                            continue
                        if proj_poly is None:
                            proj_poly = self._get_ego_polygon(
                                px, py, states['heading'][t])
                        if not proj_poly.intersects(agent_poly):
                            continue
                        # Official nuPlan helpers are angle cones measured
                        # from ego REAR AXLE to actor centroid: ahead <30deg,
                        # behind >150deg. In an intersection or while
                        # straddling/off-road, CaRL widens eligibility to
                        # every actor that is not behind.
                        if not _carl_ttc_actor_is_at_fault(
                            rear_x,
                            rear_y,
                            float(states['heading'][t]),
                            float(ax),
                            float(ay),
                            widened_cone=widened_cone,
                        ):
                            ttc_excluded_ids.add(a_id)
                            continue
                        metrics['ttc'] = 0.0
                        # Witness of this exact TTC projection, not an
                        # assertion that the nominal trajectory collides.
                        metrics['ttc_actor_id'] = str(a_id)
                        metrics['ttc_time_s'] = t * self.planner_dt + tau
                        metrics['ttc_probe_time_s'] = t * self.planner_dt
                        metrics['ttc_projection_s'] = tau
                        found = True
                        break
                    if found:
                        break
                if found:
                    break

        # 3. DDC — distance driven against the local traffic direction,
        # graded per upstream: < 2 m → 1.0, 2–6 m → 0.5, > 6 m → 0.0.
        # The nearest lane is selected WITHOUT the heading gate:
        # ``_get_best_lane``'s aligned-only pre-filter made the old
        # ``|diff| > π/2`` violation test unreachable (DDC was
        # constitutively 1.0 and wrong-way driving went unpenalised).
        if metrics['dac'] > 0:
            route_ids = getattr(self, 'route_lane_ids', None)
            if carl_collision_classifier and route_ids:
                # Upstream quantity: ONCOMING_TRAFFIC is "ego centre not
                # inside any drivable polygon that is ON ROUTE"
                # (pdm_scorer.py:282-284), and the grade is the MAXIMUM over
                # contiguous oncoming excursions, not their sum
                # (pdm_scorer.py:487-497). There is no speed gate upstream.
                #
                # This is a route-adherence constraint, not a wrong-way test:
                # a correctly-oriented but off-route lane is charged. That is
                # what makes DDC bite on the +-1 m lateral offsets at narrow
                # lanes and turns, which the heading-only test never did.
                on_route: List[bool] = []
                for t in range(horizon + 1):
                    px, py = float(states['x'][t]), float(states['y'][t])
                    inside = False
                    for idx in get_nearby(t, 15.0):
                        lane, poly = self.all_lanes[idx]
                        if str(lane.index) not in route_ids:
                            continue
                        if poly.covers(Point(px, py)):
                            inside = True
                            break
                    on_route.append(inside)
                # Centre displacement per step, masked to oncoming steps, then
                # the longest contiguous run — upstream splits ``cum_progress``
                # wherever the mask flips and takes the max of the sums.
                worst_run = 0.0
                run = 0.0
                for t in range(1, horizon + 1):
                    step_m = math.hypot(
                        float(states['x'][t]) - float(states['x'][t - 1]),
                        float(states['y'][t]) - float(states['y'][t - 1]),
                    )
                    if on_route[t]:
                        run = 0.0
                        continue
                    run += step_m
                    worst_run = max(worst_run, run)
                if worst_run >= DDC_FULL_VIOLATION_M:
                    metrics['ddc'] = 0.0
                elif worst_run >= DDC_HALF_VIOLATION_M:
                    metrics['ddc'] = 0.5
            else:
                oncoming_m = 0.0
                for t in range(horizon + 1):
                    # Outcome pinned: past the full-violation threshold ddc is
                    # 0.0 regardless of later steps — skip the remaining
                    # (expensive) per-step best-lane scans.
                    if oncoming_m > DDC_FULL_VIOLATION_M:
                        break
                    speed = states['speed'][t]
                    if speed <= 1.0:
                        continue
                    nearby_idx = get_nearby(t, 15.0)
                    if direction_route_lane_ids:
                        # Physical continuation alone supplies explicit mission
                        # identity to disambiguate overlapping junction lanes.
                        # Require actual centre containment; a nearby aligned
                        # route lane cannot excuse opposing-lane travel. Select
                        # by distance WITHOUT an ego-heading filter so reverse
                        # travel on the route remains a violation. Other callers
                        # keep the original query through the default None.
                        point = Point(float(states['x'][t]), float(states['y'][t]))
                        route_nearby = [
                            idx for idx in nearby_idx
                            if str(self.all_lanes[idx][0].index) in direction_route_lane_ids
                            and self.all_lanes[idx][1].covers(point)
                        ]
                        if route_nearby:
                            nearby_idx = route_nearby
                    best_lane = self._get_best_lane(
                        states['x'][t], states['y'][t], states['heading'][t],
                        nearby_idx, speed=speed, heading_gate=False)
                    if best_lane is None:
                        continue
                    s, r = best_lane.local_coordinates(
                        np.array([states['x'][t], states['y'][t]]))
                    s_clamped = max(0, min(s, best_lane.length))
                    lane_heading_vec = best_lane.heading_at(s_clamped)
                    lane_heading = math.atan2(
                        lane_heading_vec[1], lane_heading_vec[0])
                    diff = abs(states['heading'][t] - lane_heading)
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    if abs(diff) > np.pi / 2:
                        oncoming_m += speed * self.planner_dt
                if oncoming_m > DDC_FULL_VIOLATION_M:
                    metrics['ddc'] = 0.0
                elif oncoming_m > DDC_HALF_VIOLATION_M:
                    metrics['ddc'] = 0.5

        # 4. TLC (with pre-computed red lane IDs) — see TL_STOP_LINE_ZONE_M.
        # Pose 0 is the ego's CURRENT pose (prepended by the caller): it is
        # where the candidate starts, already charged live if it ran a
        # light, so crossings are looked for from pose 1 on. Speed and the
        # entry bound come from the displacement of the two poses being
        # judged (``states['speed']`` is the FORWARD difference).
        if red_lanes_per_t is not None:
            signalized_ids = set(str(k) for k in self.traffic_lights)
            for t in range(1, horizon + 1):
                red_ids = red_lanes_per_t[t]
                if not red_ids:
                    continue
                travel = math.hypot(float(states['x'][t]) - float(states['x'][t - 1]),
                                    float(states['y'][t]) - float(states['y'][t - 1]))
                if travel / max(float(self.planner_dt), 1e-6) <= TL_MOVING_SPEED_MPS:
                    continue
                nearby_idx = get_nearby(t, 5.0)
                centre = Point(float(states['x'][t]), float(states['y'][t]))
                centre_prev = Point(float(states['x'][t - 1]),
                                    float(states['y'][t - 1]))
                max_entry_s = TL_STOP_LINE_ZONE_M + travel
                crossed_red = False
                for idx in nearby_idx:
                    lane = self.all_lanes[idx][0]
                    if _light_id(lane) not in red_ids:
                        continue
                    if _ran_red_light(lane, self.all_lanes[idx][1], centre,
                                      centre_prev, max_entry_s):
                        crossed_red = True
                        break
                if crossed_red and not _green_way_through(
                        [self.all_lanes[i][0] for i in nearby_idx],
                        [self.all_lanes[i][1] for i in nearby_idx],
                        red_ids, signalized_ids, centre,
                        float(states['heading'][t])):
                    metrics['tlc'] = 0.0
                    break

        # 5. Lane Keeping
        consecutive_bad = 0
        if metrics['dac'] > 0:
            for t in range(horizon + 1):
                nearby_idx = get_nearby(t, 10.0)
                speed = states['speed'][t]
                best_lane = self._get_best_lane(
                    states['x'][t], states['y'][t], states['heading'][t],
                    nearby_idx, speed=speed)
                if best_lane:
                    s_lk, lat = best_lane.local_coordinates(
                        np.array([states['x'][t], states['y'][t]]))
                    if s_lk <= 0.0 or s_lk >= best_lane.length:
                        # Projection clamped at a lane end — the point
                        # is beyond the lane's longitudinal extent
                        # (junction interior not covered by lane
                        # polylines), so |lat| is dominated by the
                        # overshoot, not by lateral deviation. Not a
                        # lane-keeping signal; skip the sample.
                        continue
                    if abs(lat) > LANE_DEVIATION_LIMIT:
                        consecutive_bad += 1
                    else:
                        consecutive_bad = 0
                if consecutive_bad * self.planner_dt > LANE_KEEPING_WINDOW:
                    metrics['lk'] = 0.0
                    break

        # 6. HC — nuPlan ``ego_is_comfortable``: magnitude bounds plus
        # the signed longitudinal / yaw-acceleration bounds (the
        # original triple only covered lateral accel, jerk magnitude,
        # and yaw rate). The signed arrays are optional so external
        # callers that build ``states`` by hand keep working.
        max_acc = np.max(states['acceleration'])
        max_jerk = np.max(states['jerk'])
        max_yr = np.max(np.abs(states['yaw_rate']))
        if max_acc > MAX_ACCEL or max_jerk > MAX_JERK or max_yr > MAX_YAW_RATE:
            metrics['hc'] = 0.0
        else:
            lon_accel = states.get('lon_accel')
            lon_jerk = states.get('lon_jerk')
            yaw_accel = states.get('yaw_accel')
            if lon_accel is not None and (
                np.max(lon_accel) > MAX_LON_ACCEL
                or np.min(lon_accel) < MIN_LON_ACCEL
            ):
                metrics['hc'] = 0.0
            elif lon_jerk is not None and np.max(np.abs(lon_jerk)) > MAX_LON_JERK:
                metrics['hc'] = 0.0
            elif yaw_accel is not None and np.max(np.abs(yaw_accel)) > MAX_YAW_ACCEL:
                metrics['hc'] = 0.0

        # 7. EC (smoothness from current ego state through executed waypoints)
        if ego_state is not None:
            ego_acc_vec = ego_state.get('acceleration')
            if ego_acc_vec is None:
                # Evaluator-fed states always carry it (_enrich_ego_state);
                # a hand-built ego_state without it used to KeyError the
                # whole batch. Zero-accel prior keeps EC scored with the
                # same semantics as an ego at rest.
                if not getattr(self, '_warned_missing_accel', False):
                    self._warned_missing_accel = True
                    logger.warning(
                        "[EPDMS Fast] ego_state has no 'acceleration' — "
                        "using a zero-acceleration prior for the EC term "
                        "(warned once)")
                ego_acc_vec = np.zeros(3)
            ego_acc = np.linalg.norm(ego_acc_vec[:2])
            ego_yr = abs(ego_state.get('angular_velocity', np.zeros(3))[2])
            n_exec = min(n_execute if n_execute is not None else horizon, len(states['acceleration']))
            acc_seq = np.concatenate([[ego_acc], states['acceleration'][:n_exec]])
            yr_seq = np.concatenate([[ego_yr], np.abs(states['yaw_rate'][:n_exec])])
            diff_acc = np.abs(np.diff(acc_seq))
            diff_yr = np.abs(np.diff(yr_seq))
            comfortable = (diff_acc <= EC_ACCEL_THRESH) & (diff_yr <= EC_YAW_RATE_THRESH)
            metrics['ec'] = float(comfortable.sum()) / len(comfortable) if len(comfortable) > 0 else 1.0

        # 8. Progress
        dist = math.sqrt((states['x'][-1] - states['x'][0]) ** 2 +
                         (states['y'][-1] - states['y'][0]) ** 2)
        # Raw chord metres alongside the normalized term: reference-normalized
        # judges (VerifierScorer) rescale progress against a reference plan
        # and need the unclipped distance — ep alone saturates at EP_NORM_M.
        metrics['ep_m'] = dist
        metrics['ep'] = min(dist / EP_NORM_M, 1.0)

        return metrics

    @staticmethod
    def _zeroed_metrics() -> Dict[str, Any]:
        """All-zero metrics dict (same keys ``_calculate_metrics`` emits).

        Used for candidates that cannot be scored at all (non-finite
        values): every multiplicative gate and every weighted term is 0.0,
        so the combined score is 0.0 under any use_ep/use_ec setting.
        """
        return {
            "nc": 0.0, "dac": 0.0, "ddc": 0.0, "tlc": 0.0,
            "ep": 0.0, "ttc": 0.0, "lk": 0.0, "hc": 0.0, "ec": 0.0,
            "ep_m": 0.0,
            "collision_time_s": 0.0,
            "collision_actor_id": None,
            "ttc_actor_id": None,
            "ttc_time_s": None,
            "ttc_probe_time_s": None,
            "ttc_projection_s": None,
            "collision_at_fault_ids": [],
        }

    # ---------------------------------------------------------------
    # Coordinate transforms
    # ---------------------------------------------------------------

    def _ego_to_world(self, candidates_np: np.ndarray, ego_state: Dict[str, Any]) -> np.ndarray:
        """Vectorized ego→world transform. (N, 8, 3) → (N, 8, 2)."""
        ego_x, ego_y = ego_state['position'][:2]
        ego_heading = ego_state['heading']
        cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)
        x_fwd = candidates_np[:, :, 0]
        y_lat = candidates_np[:, :, 1]
        world_x = ego_x + x_fwd * cos_h - y_lat * sin_h
        world_y = ego_y + x_fwd * sin_h + y_lat * cos_h
        return np.stack([world_x, world_y], axis=-1)

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------

    def score_candidates(
        self,
        candidates_np: np.ndarray,
        ego_state: Dict[str, Any],
        frame_idx: int,
        *,
        n_execute: Optional[int] = None,
        return_metrics: bool = False,
        use_ep: bool = True,
        use_ec: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """Score ``(N, T, >=2)`` model-frame ``[forward, lateral, ...]`` candidates.

        .. note:: **Comparability cut.** Candidate scores
           produced before and after this date are NOT directly comparable
           (same class of cut as the c9c0aad behavior changes). Mechanism:
           the per-pose safety loops in ``_calculate_metrics`` (DAC, NC,
           TTC, DDC, TLC, LK) previously ran ``range(horizon)`` over
           ``horizon + 1`` state entries, so the terminal (4.0 s) pose
           earned EP progress while being invisible to every safety term —
           and DAC's denominator was ``horizon``. Both now cover
           ``horizon + 1`` poses (upstream ``train_pdm_scorer`` convention).
           On the 74-frame banked loop-2 corpus this changes 327/1110
           combined candidate scores and flips 15/74 frame argmaxes.

        Returns:
            ``(scores, metrics)`` — ``scores`` is ``(N,)``; ``metrics`` has one
            EPDMS-term dict per candidate when ``return_metrics`` else ``[]``.
            Both are all-zero / empty when the SDC frame is invalid."""
        if not self._initialized:
            raise RuntimeError("EPDMSTrajectoryScorer_Fast not initialized.")
        assert self.scenario_data is not None  # set in initialize()

        candidates_np = np.asarray(candidates_np, dtype=np.float64)
        N = candidates_np.shape[0]
        horizon = candidates_np.shape[1]

        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        _sdc_valid = sdc_track['state']['valid']
        _past_log = int(frame_idx) >= len(_sdc_valid)
        if not _past_log and not _sdc_valid[int(frame_idx)]:
            return np.zeros(N), []

        # Lane-less scenario: DAC has no drivable surface, so every
        # candidate is gated to 0.0. That is a data problem, not a scoring
        # verdict — say so loudly (once per scorer) instead of publishing
        # an all-zero grid that looks like uniformly terrible driving.
        if not self.all_lanes and not getattr(self, '_warned_no_lanes', False):
            self._warned_no_lanes = True
            logger.warning(
                "[EPDMS Fast] scenario %s has NO lanes — dac=0 gates every "
                "candidate's score to 0.0 (check map_features / lane build)",
                self.scenario_data.get('metadata', {}).get(
                    'scenario_id', '<unknown>'))

        # Non-finite candidates: a single NaN/inf row used to raise a
        # GEOSException from polygon construction and kill the WHOLE batch.
        # Such rows score 0.0 with zeroed metrics; finite rows are scored
        # exactly as they would be in a fully-finite batch.
        finite_mask = np.isfinite(candidates_np).all(axis=(1, 2))
        n_finite = int(finite_mask.sum())
        if n_finite < N:
            logger.warning(
                "[EPDMS Fast] %d/%d candidates contain non-finite values at "
                "frame %d — scoring them 0.0 with zeroed metrics",
                N - n_finite, N, frame_idx)

        scores = np.zeros(N)
        all_metrics: List[Dict[str, float]] = []
        if n_finite == 0:
            if return_metrics:
                all_metrics = [self._zeroed_metrics() for _ in range(N)]
            return scores, all_metrics
        scored_np = candidates_np if n_finite == N else candidates_np[finite_mask]

        # --- Step 1: Pre-compute per-frame data (shared across all candidates) ---
        logged_agents_per_t, red_lanes_per_t = self._precompute_frame_data(
            frame_idx, horizon)
        live_agent_states = ego_state.get("_execution_agent_states")
        agents_per_t = (
            logged_agents_per_t
            if live_agent_states is None
            else self._live_agents_per_t(live_agent_states, horizon)
        )

        # --- Step 2: Transform all candidates to world frame ---
        candidates_world = self._ego_to_world(scored_np, ego_state)  # (n, T, 2)

        # --- Step 3: Build all full paths (prepend ego pos) and transform to sim ---
        # Prepend the ACTUAL ego position (same frame the candidates
        # were expressed in via _ego_to_world). Prepending the GT log
        # position corrupts the first finite-difference segment of
        # every candidate by the ego's drift from the log — speed
        # ~10x drift, jerk ~1000x drift — zeroing HC and rotating the
        # early ego polygons for all proposals at once.
        current_pos = np.asarray(ego_state['position'], dtype=np.float64)[:2]
        # (n, 1, 2) ego pos broadcast + (n, T, 2) candidates → (n, T+1, 2) full paths
        ego_pos_tiled = np.broadcast_to(current_pos[None, None, :], (n_finite, 1, 2))
        all_paths_world = np.concatenate([ego_pos_tiled, candidates_world], axis=1)
        all_paths_sim = all_paths_world + self.world_to_sim_offset  # (n, T+1, 2)

        # --- Step 4: Vectorized trajectory states ---
        all_states = self._get_all_trajectory_states(all_paths_sim, self.planner_dt)

        # --- Step 5: Score each candidate ---
        metrics_by_idx: Dict[int, Dict[str, float]] = {}
        for j, i in enumerate(np.flatnonzero(finite_mask)):
            # Extract per-candidate states (views, no copy)
            states_i = {key: arr[j] for key, arr in all_states.items()}

            metrics = self._calculate_metrics(
                states_i, horizon, frame_idx,
                ego_state=ego_state, n_execute=n_execute,
                agents_per_t=agents_per_t, red_lanes_per_t=red_lanes_per_t,
            )

            scores[i] = combine_epdms_terms(metrics, use_ep=use_ep,
                                            use_ec=use_ec)
            if return_metrics:
                metrics_by_idx[int(i)] = metrics

        if return_metrics:
            all_metrics = [
                metrics_by_idx.get(i, self._zeroed_metrics())
                for i in range(N)
            ]

        return scores, all_metrics

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Deferred: torch is only needed to wrap the returned tensors, and
        # model_output already carries torch tensors on every call path. Keeps
        # the module importable on torch-free machines (CI runs the numpy
        # scoring logic without the heavy stack).
        import torch

        if not self._initialized:
            raise RuntimeError("EPDMSTrajectoryScorer_Fast not initialized.")
        assert self.scenario_data is not None  # set in initialize()

        t_start = time.time()

        ego_state = kwargs['ego_state']
        frame_idx = kwargs['frame_idx']

        # Compute n_execute from replan interval
        if self.prev_frame_idx is not None:
            replan_interval_s = (frame_idx - self.prev_frame_idx) * self.scenario_dt
            n_execute = max(1, int(round(replan_interval_s / self.planner_dt)))
        else:
            n_execute = None

        all_candidates = model_output["all_candidates"]  # (B, N, 8, 3) tensor
        candidates_np = all_candidates[0].cpu().numpy()  # (N, 8, 3)
        N = candidates_np.shape[0]

        sdc_track = self.scenario_data['tracks'][self.sdc_id]
        # Same clamp as score_candidates above: past the log the SDC carries no
        # recorded validity, and returning candidate 0 unscored there would
        # stop selection exactly where the world stops being logged.
        _sdc_valid = sdc_track['state']['valid']
        _past_log = int(frame_idx) >= len(_sdc_valid)
        if not _past_log and not _sdc_valid[int(frame_idx)]:
            # No valid ego position — return zeros
            best_traj = all_candidates[0, 0]
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.zeros(1, N),
                "best_idx": torch.tensor([0]),
            }

        scores, _ = self.score_candidates(
            candidates_np, ego_state, frame_idx, n_execute=n_execute)

        best_idx = int(np.argmax(scores))
        self.prev_frame_idx = frame_idx

        elapsed = time.time() - t_start
        if self.verbose:
            print(f"[EPDMS Fast] Frame {frame_idx}: best_idx={best_idx}/{N}, "
                  f"score={scores[best_idx]:.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}, min={scores.min():.4f}, "
                  f"time={elapsed:.2f}s ({elapsed/N*1000:.1f}ms/cand)")

        best_traj = all_candidates[0, best_idx]
        return {
            "trajectory": best_traj.unsqueeze(0),
            "scores": torch.from_numpy(scores).unsqueeze(0).float(),
            "best_idx": torch.tensor([best_idx]),
        }


class EPDMSLiveScorer:
    """Live per-frame EPDMS on executed sim state — the METRIC OF RECORD.

    Every ``metrics.json`` number in every scored table comes from this
    class's :meth:`score_frame_live`, fed once per executed frame by
    ``evaluation/evaluator.py`` through its ``_EnvProxy`` bridge. Moved
    verbatim from ``evaluation/utils/epdms_scorer_md.py`` (scorer
    consolidation, simplify.md Phase 2) so the batch engine above and the
    live metric share one module, one set of constants, and one set of
    term semantics conventions (``at_fault_ego_speed`` stopped/behind
    exemptions, pre-existing-contact exclusion, thresholds).

    .. note:: **Comparability cut.** ``metrics.json`` EPDMS
       produced before and after this date are NOT directly comparable.
       Three approved metric-semantics fixes landed together:

       * **Warm-up comfort seeding** — the evaluator now feeds the
         comfort-history chain during its open-loop replay phase via
         :meth:`observe_frame_kinematics`, so the FIRST scored frame's
         hc/ec are MEASURED against the real handoff kinematics instead
         of forced to 1.0 by an empty history (which masked above-bound
         handoff kinematics in ~95% of audited episodes). With no
         warm-up (``ego_replay_frames=0``) or after
         :meth:`reset_live_state` the legacy vacuous first frame is
         unchanged.
       * **LK consecutive semantics** — the deviation streak now RESETS
         on frames where lane keeping is not evaluable (``dac == 0`` or
         no best lane); it previously FROZE, so deviations separated by
         seconds of off-drivable / wrong-way driving accumulated toward
         the 2 s rule (observed tripping after 6 instead of 21
         consecutive deviating frames).
       * **TL frame alignment** — the evaluator passes the POST-step
         scenario timestep (the world actually being scored) and
         :meth:`_check_tlc_live` clamps to the last logged light state,
         so TL transitions are charged/cleared on the transition frame
         instead of one frame (0.1 s) late.

    .. note:: **Comparability cut.** The TL term now charges
       the CROSSING of a red connector's stop line (its START) rather than
       presence in the last 5 m of any red polygon the ego box touched —
       see ``TL_STOP_LINE_ZONE_M``. ``traffic_light_compliance`` (and the
       NavSafe ``red_light`` count) before and after this date are not
       comparable; every other term is unchanged.

       Measured magnitude (pdm_closed directional runs, scenes 2/5/8 of
       py123d_val_sample, 40 scored frames, 8 warm-up): episode
       EPDMS_no_ep moved by −0.0227 / −0.0182 / −0.0136 (scene 2/5/8) —
       entirely from hc/ec at the first 2–3 scored frames dropping
       1.0 → 0.0 once the replay→policy handoff kinematics were
       measured; every other per-frame value was byte-identical (those
       scenes contain no TL data and no LK evaluability gap, so the LK /
       TL fixes are exercised by the unit tests and the consolidation
       parity harness instead).

    Deliberate divergences from the batch scorer, kept because the live
    metric must stay bit-identical to every previously scored table:

    * **Aggregation association**: the final score is
      ``multi_prod * (weighted_sum / denom)`` — NOT
      ``combine_epdms_terms``'s ``(multi_prod * weighted_sum) / denom``.
      Same terms, same order, different float association: swapping in the
      shared combine would drift the metric by 1 ulp on some frames (the
      class of change an 800k-row bit-parity check already caught once).
    * **ec** is frame-to-frame consistency of the executed kinematics
      (scalar deltas vs the previous frame), not the batch scorer's
      intra-window plan-consistency fraction — executed motion has no
      plan window.
    * Polygons are built with shapely affinity transforms
      (``VEHICLE_POLYGON_COORDS``), not the batch scorer's cos/sin corner
      arithmetic — same rectangle, different ulp-level rounding. This is
      the ONLY remaining difference in the ``dac`` term: since Stage 4 both
      classes test their corners with the shared
      :func:`~navsafe.evaluation.utils.lane_proxy.corners_in_drivable_area`
      against the same union+buffer surface.
    * **ddc live is constitutively 1.0**: it selects its lane via the
      heading-GATED ``_get_best_lane``, so at speed only lanes within π/2
      of the ego heading qualify and the ``> π/2`` violation branch is
      unreachable. The batch scorer names and fixes this same defect with
      ``heading_gate=False`` — do NOT "fix" it here: that silently moves
      the metric of record every scored table was produced with.
    * Lesser structural differences, same bit-parity reason: live TTC
      projects ego AND agents at 5×0.2 s steps (batch: ego-only at taus
      0.3/0.6/0.9); live NC has no overtake first-contact latch (its
      pre-existing-contact exclusion is the previous frame's contact set).

    Duplication kept deliberately (NOT divergences — behavior matches the
    batch scorer): the ``__init__`` lane extraction, ``_query_nearby_lanes``
    and ``_get_best_lane`` mirror the batch implementations on a
    lane-object rather than index calling convention. They stay duplicated
    so this consolidation phase's diff is purely ADDITIVE for the batch
    class — candidate-ranking parity holds by construction. Extracting
    shared helpers is a follow-up that must run behind the
    candidate-ranking parity gate (simplify.md, "the hard acceptance
    gate"), not a drive-by.

    Construction needs the scenario dict (traffic lights, lane fallback,
    timestep) and a MetaDrive-style env exposing ``agent.position/
    heading_theta/velocity/name`` and ``engine.get_objects()``.
    """

    def __init__(self, scenario_data: dict, env):
        self.scenario_data = scenario_data
        self.env = env

        # Timestep: parsed in one audited place. On py123d, metadata['ts']
        # is an array of absolute MICROSECOND stamps — never read it as a dt
        # (scenario_dt=3.16e14 froze agents for a whole scored horizon once).
        metadata = scenario_data.get('metadata', {})
        self.scenario_dt = scenario_dt_seconds(metadata, default=0.1)

        # --- Map extraction ---
        self.all_lanes: List[Tuple[Any, Polygon]] = []
        # Tolerate an explicit ``dynamic_map_states: None`` (some converters
        # emit it), and validate the structure UP FRONT: _check_tlc_live
        # indexes ``entry['state']['object_state'][frame]`` on every frame,
        # so a malformed entry used to raise per frame inside the
        # evaluator's per-frame catch — silently un-scoring the whole
        # episode. The evaluator fails loudly on scorer-INIT errors, which
        # is where a data problem belongs.
        self.traffic_lights = scenario_data.get('dynamic_map_states') or {}
        for _tl_id, _tl_data in self.traffic_lights.items():
            try:
                _s_list = _tl_data['state']['object_state']
                len(_s_list)
            except (KeyError, TypeError, IndexError) as e:
                raise ValueError(
                    f"dynamic_map_states[{_tl_id!r}] is malformed ({e!r}): "
                    "each entry must expose ['state']['object_state'] as a "
                    "sequence of per-frame lane states — otherwise every "
                    "frame's traffic-light check fails and the episode is "
                    "silently un-scored") from e

        # Try MetaDrive road_network first (BridgeSim / MetaDrive env)
        try:
            if self.env and self.env.engine.map_manager.current_map:
                road_network = self.env.engine.map_manager.current_map.road_network
                if hasattr(road_network, 'get_all_lanes'):
                    for lane in road_network.get_all_lanes():
                        if hasattr(lane, 'shapely_polygon'):
                            self.all_lanes.append((lane, lane.shapely_polygon))
                else:
                    for start_node, end_dict in road_network.graph.items():
                        for end_node, lanes in end_dict.items():
                            for lane in lanes:
                                if hasattr(lane, 'shapely_polygon'):
                                    self.all_lanes.append((lane, lane.shapely_polygon))
        except (AttributeError, TypeError):
            pass  # env doesn't have MetaDrive road_network (e.g. navsafe/IsaacSim)

        # Fallback: build lanes from scenario_data map_features
        if not self.all_lanes and 'map_features' in scenario_data:
            from navsafe.evaluation.utils.lane_proxy import build_lanes_from_scenario
            self.all_lanes = build_lanes_from_scenario(scenario_data)

        # Drivable-area proxy for DAC: the UNION of all lane polygons with a
        # small tolerance buffer, built by the SHARED DrivableAreaProxy the
        # batch scorer above also scores against. Testing corners against
        # individual lane pieces (buffered centerline strips from LaneProxy)
        # fails whenever the ego footprint lands on the seam between two
        # pieces, in the gap at a junction connector, or slightly over the
        # edge of the strip it legitimately overhangs — MetaDrive's
        # road_network polygons cover those seams, the per-piece proxy does
        # not. NavSim's DAC checks the drivable-surface layer, which this
        # union approximates.
        self._drivable_union = DrivableAreaProxy.from_lane_polygons(
            poly for _, poly in self.all_lanes)

        # --- Live scoring state ---
        # Motion history for comfort metrics
        self.prev_velocity: Optional[np.ndarray] = None
        self.prev_heading: Optional[float] = None
        self.prev_acceleration: Optional[float] = None
        self.prev_jerk: Optional[float] = None
        self.prev_yaw_rate: Optional[float] = None

        # Lane keeping streak counter — see _check_lk_live for the exact
        # cross-frame semantics (consecutive over EVALUABLE frames; resets
        # whenever lane keeping is not evaluable).
        self.consecutive_lane_deviation = 0

        # Agent names in contact with the ego on the PREVIOUS live frame —
        # the live analog of the trajectory scorers' ``preexisting_ids``
        # (upstream charges a collision once, via already_collided_ids).
        self._live_contact_names: set = set()
        # Ego centre on the previous observed frame (warm-up or scored) —
        # the red-light CROSSING rule's other endpoint (TL_STOP_LINE_ZONE_M).
        self._prev_centre_xy: Optional[np.ndarray] = None

    def _get_ego_polygon(self, x, y, heading) -> Polygon:
        base_poly = Polygon(VEHICLE_POLYGON_COORDS)
        rotated_poly = affinity.rotate(base_poly, heading, origin=(0, 0), use_radians=True)
        return affinity.translate(rotated_poly, xoff=x, yoff=y)

    def _query_nearby_lanes(self, x, y, radius=50.0):
        # x, y are already in the sim frame (live positions come straight
        # from the env; this class carries no world→sim calibration).
        nearby = []
        for lane, poly in self.all_lanes:
            minx, miny, maxx, maxy = poly.bounds
            if (minx - radius < x < maxx + radius) and (miny - radius < y < maxy + radius):
                nearby.append(lane)
        return nearby

    def _get_best_lane(self, x, y, heading, nearby_lanes, speed=None):
        if not nearby_lanes: return None
        candidates = []
        is_stopped = (speed is not None) and (speed < 1.0)

        for lane in nearby_lanes:
            s, r = lane.local_coordinates(np.array([x, y]))
            s_clamped = max(0, min(s, lane.length))
            lane_heading_vec = lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            dist = lane.distance(np.array([x, y]))

            if is_stopped or (abs(diff) < (np.pi / 2)):
                candidates.append((lane, dist))

        if candidates:
            return min(candidates, key=lambda c: c[1])[0]
        return None

    def _get_current_agents_from_sim(self) -> List[dict]:
        """
        Query current agent states from simulation environment.
        Returns list of dicts with position, heading, and bounding box info.
        """
        agents = []
        ego_name = self.env.agent.name

        for obj in self.env.engine.get_objects().values():
            if obj.name == ego_name:
                continue
            # Only consider objects with position and heading (vehicles, cyclists, etc.)
            if not hasattr(obj, 'position') or not hasattr(obj, 'heading_theta'):
                continue

            # Get bounding box dimensions
            length = getattr(obj, 'top_down_length', getattr(obj, 'LENGTH', VEHICLE_LENGTH))
            width = getattr(obj, 'top_down_width', getattr(obj, 'WIDTH', VEHICLE_WIDTH))

            agents.append({
                'name': obj.name,
                'position': np.array(obj.position[:2]),  # [x, y]
                'heading': obj.heading_theta,
                'length': length,
                'width': width,
                # [vx, vy] — needed by the TTC forward projection.
                'velocity': (np.array(obj.velocity[:2])
                             if hasattr(obj, 'velocity')
                             else np.zeros(2)),
            })

        return agents

    def _get_agent_polygon(self, agent: dict) -> Polygon:
        """Create a polygon for an agent given its state dict."""
        half_l = agent['length'] / 2
        half_w = agent['width'] / 2
        coords = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])
        base_poly = Polygon(coords)
        rotated_poly = affinity.rotate(base_poly, agent['heading'], origin=(0, 0), use_radians=True)
        return affinity.translate(rotated_poly, xoff=agent['position'][0], yoff=agent['position'][1])

    def _check_dac_live(self, ego_pos: np.ndarray, ego_heading: float) -> float:
        """
        Check Drivable Area Compliance using current ego state.
        Returns 1.0 if all corners are in drivable area, 0.0 otherwise.
        """
        return 1.0 if self._corners_in_drivable_area(
            ego_pos[0], ego_pos[1], ego_heading) else 0.0

    def _corners_in_drivable_area(self, x: float, y: float, heading: float) -> bool:
        """All four ego corners inside the drivable-area proxy.

        Uses the cached union of lane polygons (see __init__) so a footprint
        spanning adjacent lane pieces or a junction connector is compliant;
        falls back to the per-lane corner test when the union is unavailable.
        Both branches live in the shared
        :func:`~navsafe.evaluation.utils.lane_proxy.corners_in_drivable_area`
        so the batch scorer's proposal DAC cannot diverge from this one.
        """
        ego_poly = self._get_ego_polygon(x, y, heading)
        return corners_in_drivable_area(
            ego_poly.exterior.coords[:-1],
            self._drivable_union,
            lambda: (lane.shapely_polygon
                     for lane in self._query_nearby_lanes(x, y, radius=15.0)),
        )

    def _check_nc_live(self, ego_pos: np.ndarray, ego_heading: float,
                       ego_speed: float, current_agents: List[dict],
                       excluded_names: Optional[set] = None,
                       agent_polys: Optional[List[Polygon]] = None) -> float:
        """
        Check No Collision using current simulation state.
        Returns 1.0 if no at-fault collision, 0.0 otherwise.
        (Port of BridgeSim's _check_nc_live, plus the pre-existing-contact
        exclusion: ``excluded_names`` are agents already in contact when the
        frame began — a persisting contact must not re-zero nc every frame.)

        ``agent_polys``, when given, must be parallel to ``current_agents``
        (one polygon per agent, same order) — score_frame_live already
        builds them for the contact-set update, so this avoids rebuilding
        the identical polygons.
        """
        ego_poly = self._get_ego_polygon(ego_pos[0], ego_pos[1], ego_heading)

        for agent_idx, agent in enumerate(current_agents):
            if excluded_names and agent['name'] in excluded_names:
                continue
            agent_poly = (agent_polys[agent_idx] if agent_polys is not None
                          else self._get_agent_polygon(agent))

            if ego_poly.intersects(agent_poly):
                is_at_fault = True

                # If ego is stopped, not at fault. ``ego_speed`` is the LIVE
                # speed from the sim (not the savgol profile), i.e. already
                # the ego's speed at collision time.
                if ego_speed < STOPPED_SPEED_THRESHOLD:
                    is_at_fault = False

                # If agent is behind or beside ego, not at fault — aligned
                # to the shared <= 0.0 convention (fast/slow scorers,
                # planner sweep): the old < -1.0 cutoff faulted every
                # sideswipe the ego did not initiate.
                dx = agent['position'][0] - ego_pos[0]
                dy = agent['position'][1] - ego_pos[1]
                ego_hx = np.cos(ego_heading)
                ego_hy = np.sin(ego_heading)
                longitudinal_dist = dx * ego_hx + dy * ego_hy
                if longitudinal_dist <= 0.0:
                    is_at_fault = False

                if is_at_fault:
                    return 0.0

        return 1.0

    @staticmethod
    def _is_agent_behind(ego_position: np.ndarray, ego_heading: float,
                         agent_position: np.ndarray) -> bool:
        """NavSim/nuPlan behind test: negative projection on ego forward.

        Port of bridgesim.evaluation.utils.collision_classifier.is_agent_behind.
        """
        delta = agent_position[:2] - ego_position[:2]
        ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        return float(np.dot(delta, ego_forward)) < 0

    def _check_ttc_live(self, ego_pos: np.ndarray, ego_heading: float,
                        ego_speed: float, ego_velocity: np.ndarray,
                        current_agents: List[dict],
                        excluded_names: Optional[set] = None) -> float:
        """
        Check Time to Collision by projecting ego and agents forward at
        constant velocity for TTC_HORIZON (1s), following NAVSIM style.
        (Port of BridgeSim's _check_ttc_live.)

        At-fault logic actually applied (per projected contact):
        - Ego stopped (< STOPPED_SPEED_THRESHOLD) → not at-fault, skip
        - Projected agent NOT behind the projected ego → at-fault → 0.0
        - Projected agent behind → not at-fault, skip

        The BridgeSim source also consulted an "ego in multiple lanes or
        non-drivable area" test, but its condition —
        ``agent_ahead or (ego_in_wrong_area and not agent_behind)`` with
        ``agent_ahead = not agent_behind`` — reduces to ``not
        agent_behind``: the wrong-area term could never change the
        outcome, so it is not computed here (float-identity proven by the
        consolidation parity replay).

        Returns 1.0 if no predicted at-fault collision, 0.0 otherwise.
        """
        n_steps = 5
        step_dt = TTC_HORIZON / n_steps

        for i in range(1, n_steps + 1):
            t = i * step_dt
            future_ego_pos = ego_pos + ego_velocity * t
            # Depends only on the ego — hoisted out of the agent loop.
            future_ego_poly = self._get_ego_polygon(
                future_ego_pos[0], future_ego_pos[1], ego_heading)

            for agent in current_agents:
                # Same pre-existing-contact exclusion as NC: an agent already
                # touching the ego must not cost the ttc term every frame.
                if excluded_names and agent['name'] in excluded_names:
                    continue
                agent_vel = agent.get('velocity', np.zeros(2))
                future_agent_pos = agent['position'] + agent_vel * t

                future_agent = {**agent, 'position': future_agent_pos}
                future_agent_poly = self._get_agent_polygon(future_agent)

                if not future_ego_poly.intersects(future_agent_poly):
                    continue

                # Ego stopped → not at-fault
                if ego_speed < STOPPED_SPEED_THRESHOLD:
                    continue

                agent_behind = self._is_agent_behind(
                    future_ego_pos, ego_heading, future_agent_pos)
                if not agent_behind:
                    return 0.0

        return 1.0

    def _check_ddc_live(self, ego_pos: np.ndarray, ego_heading: float, ego_speed: float) -> float:
        """
        Check Driving Direction Compliance using current ego state.
        Returns 1.0 if heading aligns with lane direction, 0.0 otherwise.
        """
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=15.0)
        best_lane = self._get_best_lane(ego_pos[0], ego_pos[1], ego_heading, nearby, speed=ego_speed)

        if best_lane and ego_speed > 1.0:
            s, r = best_lane.local_coordinates(np.array([ego_pos[0], ego_pos[1]]))
            s_clamped = max(0, min(s, best_lane.length))
            lane_heading_vec = best_lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(ego_heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi

            if abs(diff) > np.pi / 2:
                return 0.0

        return 1.0

    def _check_tlc_live(self, ego_pos: np.ndarray, ego_heading: float,
                        ego_speed: float, frame_idx: int,
                        prev_pos: Optional[np.ndarray] = None) -> float:
        """
        Check Traffic Light Compliance using current ego state.

        ``prev_pos`` is the ego centre one scenario step earlier (the
        previous scored or warm-up frame) — the crossing rule needs it; see
        TL_STOP_LINE_ZONE_M. ``None`` falls back to the stateless zone test.

        ``frame_idx`` is the scenario timestep of the WORLD STATE being
        scored — since the TL alignment fix the evaluator
        passes its post-step timestep (``Evaluator._scored_world_timestep``),
        so the logged light state is read at the same timestep the ego /
        agent poses reflect (it previously ran one frame behind).
        Returns 1.0 if compliant, 0.0 if running a red light.
        """
        # Red lane IDs from the logged light states at the scored timestep.
        # Past the end of an entry's log the lookup CLAMPS to the last
        # logged state: the world the evaluator scores out
        # there is frozen at the last logged frame (the env replay clamps
        # ``t = min(timestep, len - 1)``), so the light must freeze the
        # same way. Out-of-range previously read as "no state → not red".
        red_lane_ids = _red_lane_ids(self.traffic_lights, frame_idx)

        if not red_lane_ids:
            return 1.0

        if ego_speed <= TL_MOVING_SPEED_MPS:
            return 1.0
        centre = Point(float(ego_pos[0]), float(ego_pos[1]))
        centre_prev = (Point(float(prev_pos[0]), float(prev_pos[1]))
                       if prev_pos is not None else None)
        # The entry bound grows with the sample's own travel (exact under a
        # teleport double-step too); with no previous sample the bare zone.
        travel = (math.hypot(centre.x - centre_prev.x, centre.y - centre_prev.y)
                  if centre_prev is not None else 0.0)
        max_entry_s = TL_STOP_LINE_ZONE_M + travel
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=5.0)

        crossed_red = False
        for lane in nearby:
            if _light_id(lane) not in red_lane_ids:
                continue
            # Running the light = crossing the red connector's ENTRY at
            # speed (see TL_STOP_LINE_ZONE_M); same rule as the batch scorer.
            if _ran_red_light(lane, lane.shapely_polygon, centre, centre_prev,
                              max_entry_s):
                crossed_red = True
                break
        if crossed_red and not _green_way_through(
                nearby, [lane.shapely_polygon for lane in nearby],
                red_lane_ids, set(str(k) for k in self.traffic_lights),
                centre, float(ego_heading)):
            return 0.0
        return 1.0

    def _red_lane_ids_at(self, frame_idx: int) -> set:
        """Lane ids red at ``frame_idx``, clamped to the last logged state."""
        return _red_lane_ids(self.traffic_lights, frame_idx)

    def _signal_hold_live(self, ego_pos: np.ndarray, ego_heading: float,
                          frame_idx: int) -> bool:
        """Is a red light what is (or would be) holding the ego here?

        See SIGNAL_HOLD_LOOKAHEAD_M. A state fact, independent of speed:
        the deadlock detector combines it with the standstill itself.
        """
        red_lane_ids = self._red_lane_ids_at(frame_idx)
        if not red_lane_ids:
            return False
        ex, ey = float(ego_pos[0]), float(ego_pos[1])
        hx, hy = math.cos(ego_heading), math.sin(ego_heading)
        centre = Point(ex, ey)
        for lane in self._query_nearby_lanes(ex, ey, radius=SIGNAL_HOLD_LOOKAHEAD_M):
            if _light_id(lane) not in red_lane_ids:
                continue
            tangent = lane.heading_at(min(2.0, lane.length))
            if float(tangent[0] * hx + tangent[1] * hy) < SIGNAL_HOLD_HEADING_COS_MIN:
                continue
            start = _lane_polyline_start(lane)
            if start is None:
                continue
            dx, dy = float(start[0]) - ex, float(start[1]) - ey
            forward = dx * hx + dy * hy
            lateral = abs(-dx * hy + dy * hx)
            if 0.0 <= forward <= SIGNAL_HOLD_LOOKAHEAD_M and lateral <= SIGNAL_HOLD_LATERAL_M:
                return True
            # Just across the line (the IDM standstill can leave the centre
            # a few centimetres past it): still the signal's hold.
            if lane.shapely_polygon.contains(centre):
                s_along, _ = lane.local_coordinates(np.array([ex, ey]))
                if s_along < TL_STOP_LINE_ZONE_M:
                    return True
        return False

    def _check_lk_live(self, ego_pos: np.ndarray, ego_heading: float,
                       ego_speed: float) -> Tuple[float, int]:
        """
        Check Lane Keeping using current ego state.

        Returns ``(lk, new_streak)``: the term value and the NEW value of
        the cross-frame deviation streak counter. This method does NOT
        write ``self.consecutive_lane_deviation`` — score_frame_live
        commits the returned streak only after the whole frame scores
        successfully (exception atomicity).

        Cross-frame streak semantics (approved fix — true
        CONSECUTIVE semantics; pinned by tests): the counter counts
        consecutive frames on which lane keeping was EVALUATED as
        deviating. On any frame where lane keeping is not evaluable, the
        streak RESETS to 0:

        * no best lane found (e.g. a moving ego whose heading is > π/2
          from every nearby lane, or no lane within 10 m) — reset here;
        * dac == 0 — score_frame_live skips this check entirely,
          reports lk = 1.0, and resets the streak at its dac gate.

        Before the fix the counter FROZE across both kinds of frame, so
        deviations separated by seconds of off-drivable / wrong-way
        driving accumulated into one "consecutive" streak (reproduced:
        the 2 s rule tripped after 6 rather than 21 consecutive frames).
        """
        nearby = self._query_nearby_lanes(ego_pos[0], ego_pos[1], radius=10.0)
        best_lane = self._get_best_lane(ego_pos[0], ego_pos[1], ego_heading, nearby, speed=ego_speed)

        if best_lane:
            _, lat = best_lane.local_coordinates(np.array([ego_pos[0], ego_pos[1]]))
            if abs(lat) > LANE_DEVIATION_LIMIT:
                new_streak = self.consecutive_lane_deviation + 1
            else:
                new_streak = 0
        else:
            # Not evaluable this frame → the deviation run is broken.
            new_streak = 0

        # Check if consecutive deviation exceeds threshold
        # Using scenario_dt as the time step between frames
        if new_streak * self.scenario_dt > LANE_KEEPING_WINDOW:
            return 0.0, new_streak

        return 1.0, new_streak

    def _check_comfort_live(
            self, ego_velocity: np.ndarray, ego_heading: float,
    ) -> Tuple[float, float, Tuple[np.ndarray, float, float, float, float]]:
        """
        Check History Comfort (HC) and Extended Comfort (EC) using actual motion.
        Computes acceleration, jerk, yaw_rate from motion history.

        Returns ``(hc, ec, staged)`` where ``staged`` is the new comfort
        history ``(prev_velocity, prev_heading, prev_acceleration,
        prev_jerk, prev_yaw_rate)``. This method does NOT write the
        ``self.prev_*`` fields — score_frame_live commits ``staged`` only
        after the whole frame scores successfully (exception atomicity).

        First-frame note: when every ``prev_*`` is ``None`` (a truly
        history-less frame — construction, or the first frame after
        ``reset_live_state``, with no warm-up observed), acceleration /
        jerk / yaw_rate are 0.0 and EC's consistency test is skipped —
        hc = ec = 1.0 vacuously, as there is nothing to measure against.
        Since the warm-up seeding fix the evaluator feeds this
        history during its replay phase via
        :meth:`observe_frame_kinematics`, so with warm-up the first
        SCORED frame is measured (the handoff kinematics are no longer
        masked); with ``ego_replay_frames=0`` the vacuous first frame is
        the unchanged legacy behavior.
        """
        hc, ec = 1.0, 1.0
        dt = self.scenario_dt

        # Current speed
        speed = np.linalg.norm(ego_velocity)
        is_stopped = speed < STOPPED_SPEED_THRESHOLD

        # Compute current acceleration
        if self.prev_velocity is not None:
            acceleration_vec = (ego_velocity - self.prev_velocity) / dt
            acceleration = float(np.linalg.norm(acceleration_vec))
        else:
            acceleration = 0.0

        # Compute current jerk
        if self.prev_acceleration is not None:
            jerk = abs(acceleration - self.prev_acceleration) / dt
        else:
            jerk = 0.0

        # Compute current yaw rate
        if self.prev_heading is not None:
            heading_diff = ego_heading - self.prev_heading
            # Wrap to [-pi, pi]
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
            yaw_rate = abs(heading_diff) / dt
        else:
            yaw_rate = 0.0

        # Filter out noise when stopped
        if is_stopped:
            acceleration = 0.0
            jerk = 0.0
            yaw_rate = 0.0

        # HC: Check if current values exceed thresholds
        if acceleration > MAX_ACCEL or jerk > MAX_JERK or yaw_rate > MAX_YAW_RATE:
            hc = 0.0

        # EC: Check consistency with previous values
        if self.prev_acceleration is not None and self.prev_jerk is not None and self.prev_yaw_rate is not None:
            diff_acc = abs(acceleration - self.prev_acceleration)
            diff_jerk = abs(jerk - self.prev_jerk)
            diff_yr = abs(yaw_rate - self.prev_yaw_rate)

            if diff_acc > EC_ACCEL_THRESH or diff_jerk > EC_JERK_THRESH or diff_yr > EC_YAW_RATE_THRESH:
                ec = 0.0

        # Staged history for the next frame — committed by the caller.
        staged = (ego_velocity.copy(), ego_heading, acceleration, jerk,
                  yaw_rate)

        return hc, ec, staged

    def observe_frame_kinematics(self) -> None:
        """Advance the comfort-history chain from the current env state
        WITHOUT scoring (warm-up seeding fix).

        During the evaluator's open-loop replay phase
        (``frame < ego_replay_frames``) the ego executes logged actions;
        those frames are not scored, but their kinematics are real
        history. Before this method existed the first scored frame began
        with an empty history (``prev_* = None``), which forced
        hc = ec = 1.0 regardless of the actual replay→policy handoff
        kinematics — masking above-bound handoffs in ~95% of audited
        episodes. The evaluator calls this once per warm-up frame at the
        same post-step point where scoring would occur, so the seeded
        samples are spaced exactly one ``scenario_dt`` apart, matching
        the scored cadence.

        Reads the same env-agent state ``score_frame_live`` reads
        (velocity / heading for the comfort chain; position for the
        red-light crossing rule's previous centre) and stages + commits
        ONLY the comfort history (velocity → acceleration → jerk staging,
        heading → yaw rate) and ``_prev_centre_xy``. It deliberately does
        NOT touch the live contact set (seeding it is a separate,
        unapproved change), the lane-keeping streak, or any term
        evaluation.

        When there is no warm-up this method is simply never called and
        the first scored frame keeps its legacy vacuous-comfort behavior
        (a truly history-less frame has nothing to measure against);
        likewise after :meth:`reset_live_state`.
        """
        ego_heading = self.env.agent.heading_theta
        ego_velocity = (np.array(self.env.agent.velocity[:2])
                        if hasattr(self.env.agent, 'velocity')
                        else np.array([0.0, 0.0]))
        # Reuse the scoring path's staging math so seeded history is
        # bit-identical to what a scored frame would have committed.
        _hc, _ec, staged = self._check_comfort_live(ego_velocity, ego_heading)
        (self.prev_velocity, self.prev_heading, self.prev_acceleration,
         self.prev_jerk, self.prev_yaw_rate) = staged
        # The crossing rule's previous endpoint: a warm-up that ends inside a
        # connector must not read as a crossing on the first scored frame.
        self._prev_centre_xy = np.array(self.env.agent.position[:2],
                                        dtype=np.float64)

    def score_frame_live(self, frame_idx: int) -> Dict[str, float]:
        """
        Score current frame using LIVE simulation state for ALL metrics.

        This method queries the current ego state and other agents directly from
        the simulation environment, rather than using predicted trajectories or
        logged agent data.

        Use this for closed-loop evaluation where actual execution matters.

        Args:
            frame_idx: Scenario timestep of the world state being scored
                (used only for the traffic light state lookup). The
                evaluator scores AFTER ``env.step`` advanced the sim, so
                it passes its post-step timestep
                (``Evaluator._scored_world_timestep`` — TL
                alignment fix); a caller whose env already reflects
                ``frame_idx`` passes it as-is.

        Returns:
            Dict with all metric scores and overall score.
        """
        # 1. Get CURRENT ego state from simulation
        ego_pos = np.array(self.env.agent.position[:2])
        ego_heading = self.env.agent.heading_theta
        ego_velocity = np.array(self.env.agent.velocity[:2]) if hasattr(self.env.agent, 'velocity') else np.array([0.0, 0.0])
        ego_speed = float(np.linalg.norm(ego_velocity))

        # 2. Get CURRENT other agents from simulation
        current_agents = self._get_current_agents_from_sim()

        # Pre-existing contacts: agents already touching the ego on the
        # previous frame were not newly caused by this frame's motion —
        # without the exclusion a single contact re-zeroes nc/ttc for its
        # whole duration (the live analog of the trajectory scorers'
        # ``preexisting_ids``; upstream charges a collision once).
        # The NEW contact set is computed here but only COMMITTED after
        # every term check succeeds (see the commit block below): a frame
        # that raises mid-scoring must not consume the contact memory, or
        # the still-present collision is never charged.
        prev_contacts = self._live_contact_names
        ego_poly_now = self._get_ego_polygon(ego_pos[0], ego_pos[1], ego_heading)
        agent_polys = [self._get_agent_polygon(agent)
                       for agent in current_agents]
        new_contacts = {
            agent['name']
            for agent, agent_poly in zip(current_agents, agent_polys)
            if ego_poly_now.intersects(agent_poly)
        }

        # 3. Compute all metrics using live state
        # NC: current-frame collision check (reuses the agent polygons
        # built for the contact set above)
        nc = self._check_nc_live(ego_pos, ego_heading, ego_speed, current_agents,
                                 excluded_names=prev_contacts,
                                 agent_polys=agent_polys)
        # TTC: constant-velocity forward projection (if already at-fault
        # colliding, TTC is also 0) — matches BridgeSim's ordering.
        ttc = (self._check_ttc_live(ego_pos, ego_heading, ego_speed,
                                    ego_velocity, current_agents,
                                    excluded_names=prev_contacts)
               if nc > 0 else 0.0)

        # DAC: Drivable area compliance
        dac = self._check_dac_live(ego_pos, ego_heading)

        # DDC: Driving direction compliance
        ddc = self._check_ddc_live(ego_pos, ego_heading, ego_speed) if dac > 0 else 1.0

        # TLC: Traffic light compliance (crossing rule — needs last centre)
        tlc = self._check_tlc_live(ego_pos, ego_heading, ego_speed, frame_idx,
                                   prev_pos=self._prev_centre_xy)
        # Signal hold: a red light ahead on the ego's lane (state fact, not
        # a score term) — consumed by the NavSafe deadlock detector.
        signal_hold = self._signal_hold_live(ego_pos, ego_heading, frame_idx)

        # LK: Lane keeping. A dac-gated frame is not evaluable for lane
        # keeping, so the deviation streak RESETS (consecutive-
        # semantics fix; it previously froze — see _check_lk_live's
        # docstring). The reset is committed with the rest of the frame's
        # state in the commit block below.
        if dac > 0:
            lk, lk_streak = self._check_lk_live(ego_pos, ego_heading, ego_speed)
        else:
            lk, lk_streak = 1.0, 0

        # HC & EC: Comfort metrics from actual motion
        hc, ec, comfort_staged = self._check_comfort_live(
            ego_velocity, ego_heading)

        # 4. Final score: same terms/weights as combine_epdms_terms with
        # use_ep=False/use_ec=True, but the ASSOCIATION is kept verbatim —
        # multi_prod * (weighted_sum / denom), not (multi_prod *
        # weighted_sum) / denom. See the class docstring: this is the
        # metric of record and must not move by even 1 ulp.
        multi_prod = nc * dac * ddc * tlc
        weighted_sum = (W_TTC * ttc + W_LANE_KEEPING * lk +
                        W_HISTORY_COMFORT * hc + W_EXTENDED_COMFORT * ec)
        # W_NO_EP_TOTAL is the same sum in the same association order as the
        # original inline denominator — exact small-integer floats, so this
        # is the one shared-constant swap that cannot move a bit.
        weighted_avg = weighted_sum / W_NO_EP_TOTAL
        score = multi_prod * weighted_avg

        # 5. COMMIT cross-frame state — every term check above succeeded.
        # All mutation is confined to this block so a frame that raises
        # anywhere above leaves the scorer exactly as the previous frame
        # left it: the contact memory still charges a persisting collision,
        # and the comfort history spans one dt, not two.
        self._live_contact_names = new_contacts
        self.consecutive_lane_deviation = lk_streak
        (self.prev_velocity, self.prev_heading, self.prev_acceleration,
         self.prev_jerk, self.prev_yaw_rate) = comfort_staged
        self._prev_centre_xy = np.array(ego_pos[:2], dtype=np.float64)

        return {
            "no_at_fault_collisions": nc,
            "drivable_area_compliance": dac,
            "driving_direction_compliance": ddc,
            "traffic_light_compliance": tlc,
            "time_to_collision_within_bound": ttc,
            "lane_keeping": lk,
            "history_comfort": hc,
            "extended_comfort": ec,
            "score": score,
            "signal_hold": 1.0 if signal_hold else 0.0,
            "valid": True
        }

    def reset_live_state(self):
        """
        Reset the live scoring state. Call this at the start of each episode.
        """
        self.prev_velocity = None
        self.prev_heading = None
        self.prev_acceleration = None
        self.prev_jerk = None
        self.prev_yaw_rate = None
        self.consecutive_lane_deviation = 0
        self._live_contact_names = set()
        self._prev_centre_xy = None
