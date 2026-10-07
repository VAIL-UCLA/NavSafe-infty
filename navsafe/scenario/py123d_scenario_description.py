# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Project a py123d ScenarioState into a ScenarioNet ``ScenarioDescription``.

The evaluation/log-replay env consumes a ScenarioNet-format ``ScenarioDescription``
dict (``tracks`` / ``map_features`` / ``dynamic_map_states`` / ``metadata``).
py123d arrow logs are first *absorbed* into the py123d-native
:class:`~navsafe.scenario.py123d_schema.Py123DScenarioData` / and the
per-frame :class:`~navsafe.scenario.training_schema.NexusScenarioLog`, then
*projected* here into the universal runtime dict — so the env, scorers and
visualizers stay source-agnostic and unchanged.

This is the bridge that lets the eval pipeline read arrow logs directly instead
of ScenarioNet pickles, while keeping the py123d ScenarioState as the carrier
that absorbs all py123d information.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from navsafe.scenario.py123d_schema import Py123DScenarioData
from navsafe.scenario.py123d_training_extractor import training_scenario_from_py123d
from navsafe.scenario.route_lane_chain import derive_route_lane_ids
from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.speed_target import (
    SPEED_LIMIT_KEY,
    SPEED_LIMIT_SOURCE_KEY,
    annotate_speed_targets,
)
from navsafe.scenario.training_schema import NexusMapState, NexusScenarioLog
from navsafe.scenario.type import MetaDriveType

logger = logging.getLogger(__name__)

# Unmapped actor labels already warned about — warn once per label, not per actor.
_WARNED_UNKNOWN_LABELS: set[str] = set()


# py123d emits raw lane/line type strings (e.g. "surface_street",
# "double_solid_yellow"); the ScenarioDescription contract (and every
# downstream consumer — map builder, EPDMS drivable-area scorer, lane proxy)
# expects MetaDriveType-canonical types (e.g. "LANE_SURFACE_STREET"). Without
# this mapping ``MetaDriveType.is_lane`` rejects every lane, so 0 lanes are
# built and drivable-area compliance collapses to 0. Unknown types fall back
# to a sensible default rather than being dropped.
_PY123D_LANE_TYPE_TO_MD = {
    "surface_street": MetaDriveType.LANE_SURFACE_STREET,
    "bike_lane": MetaDriveType.LANE_BIKE_LANE,
    "freeway": MetaDriveType.LANE_FREEWAY,
    "highway": MetaDriveType.LANE_FREEWAY,
    "unstructured": MetaDriveType.LANE_SURFACE_UNSTRUCTURE,
}
_PY123D_LINE_TYPE_TO_MD = {
    "solid_white": MetaDriveType.LINE_SOLID_SINGLE_WHITE,
    "broken_white": MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
    "double_solid_white": MetaDriveType.LINE_SOLID_DOUBLE_WHITE,
    "solid_yellow": MetaDriveType.LINE_SOLID_SINGLE_YELLOW,
    "broken_yellow": MetaDriveType.LINE_BROKEN_SINGLE_YELLOW,
    "double_solid_yellow": MetaDriveType.LINE_SOLID_DOUBLE_YELLOW,
    "double_broken_yellow": MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW,
    "passing_double_yellow": MetaDriveType.LINE_PASSING_DOUBLE_YELLOW,
}
_PY123D_EDGE_TYPE_TO_MD = {
    "road_edge": MetaDriveType.BOUNDARY_LINE,
    "median": MetaDriveType.BOUNDARY_MEDIAN,
    "sidewalk": MetaDriveType.BOUNDARY_SIDEWALK,
    "guardrail": MetaDriveType.GUARDRAIL,
    "crosswalk": MetaDriveType.CROSSWALK,
}


def _map_lane_type(raw: Optional[str]) -> str:
    """py123d lane type → MetaDriveType (default: surface street — still a lane)."""
    if not raw:
        return MetaDriveType.LANE_SURFACE_STREET
    return _PY123D_LANE_TYPE_TO_MD.get(str(raw).lower(), MetaDriveType.LANE_SURFACE_STREET)


def _map_line_type(raw: Optional[str]) -> str:
    """py123d line/edge type → MetaDriveType (default: unknown line)."""
    if not raw:
        return MetaDriveType.LINE_UNKNOWN
    key = str(raw).lower()
    return (
        _PY123D_LINE_TYPE_TO_MD.get(key)
        or _PY123D_EDGE_TYPE_TO_MD.get(key)
        or MetaDriveType.LINE_UNKNOWN
    )


# Argoverse2 / py123d actor labels → MetaDriveType agent types. The replay
# manager dispatches on these canonical types (``MetaDriveType.VEHICLE`` etc.);
# without this mapping the raw av2 labels (``regular_vehicle``, ``bicyclist``,
# ``pedestrian`` …) match no branch, so every non-ego actor falls through to
# the generic spawner and fails — leaving the scene with only the ego.
_PY123D_AGENT_TYPE_TO_MD = {
    # vehicles
    "regular_vehicle": MetaDriveType.VEHICLE,
    "vehicle": MetaDriveType.VEHICLE,
    "car": MetaDriveType.VEHICLE,
    "large_vehicle": MetaDriveType.VEHICLE,
    "truck": MetaDriveType.VEHICLE,
    "truck_cab": MetaDriveType.VEHICLE,
    "box_truck": MetaDriveType.VEHICLE,
    "bus": MetaDriveType.VEHICLE,
    "school_bus": MetaDriveType.VEHICLE,
    "articulated_bus": MetaDriveType.VEHICLE,
    "vehicular_trailer": MetaDriveType.VEHICLE,
    "message_board_trailer": MetaDriveType.VEHICLE,
    "railed_vehicle": MetaDriveType.VEHICLE,
    # cyclists / two-wheelers (CYCLIST renders a rider; manager reuses a person USD)
    "bicyclist": MetaDriveType.CYCLIST,
    "bicycle": MetaDriveType.CYCLIST,
    "motorcyclist": MetaDriveType.CYCLIST,
    "motorcycle": MetaDriveType.CYCLIST,
    "wheeled_rider": MetaDriveType.CYCLIST,
    "two_wheeler": MetaDriveType.CYCLIST,  # py123d/123D DefaultBoxDetectionLabel.TWO_WHEELER
    # pedestrians
    "pedestrian": MetaDriveType.PEDESTRIAN,
    "wheelchair": MetaDriveType.PEDESTRIAN,
    "stroller": MetaDriveType.PEDESTRIAN,
    "official_signaler": MetaDriveType.PEDESTRIAN,
    "wheeled_device": MetaDriveType.PEDESTRIAN,
    # static traffic objects
    "construction_cone": MetaDriveType.TRAFFIC_CONE,
    "construction_barrel": MetaDriveType.TRAFFIC_CONE,
    "bollard": MetaDriveType.TRAFFIC_BARRIER,
    "construction_barrier": MetaDriveType.TRAFFIC_BARRIER,
    "sign": MetaDriveType.TRAFFIC_BARRIER,
    "stop_sign": MetaDriveType.TRAFFIC_STOP_SIGN,
    "mobile_pedestrian_crossing_sign": MetaDriveType.TRAFFIC_BARRIER,
}


def _map_agent_type(raw: Optional[str], *, strict: bool = False) -> str:
    """py123d/av2 actor label → MetaDriveType agent type.

    Exact lookup first, then a keyword fallback so unseen av2 categories still
    route to a sensible type. A label matching neither maps to
    ``MetaDriveType.OTHER`` (not silently to VEHICLE, which would corrupt
    training data) and is warned once. ``strict=True`` raises instead — use it
    for baseline-grade dataset builds where an unmapped label must fail loudly.
    """
    key = str(raw or "").lower()
    if key in _PY123D_AGENT_TYPE_TO_MD:
        return _PY123D_AGENT_TYPE_TO_MD[key]
    if any(s in key for s in ("cycl", "bike", "bicycle")):
        return MetaDriveType.CYCLIST
    if any(s in key for s in ("ped", "wheelchair", "stroller", "person")):
        return MetaDriveType.PEDESTRIAN
    if "cone" in key or "barrel" in key:
        return MetaDriveType.TRAFFIC_CONE
    if any(s in key for s in ("barrier", "bollard", "sign")):
        return MetaDriveType.TRAFFIC_BARRIER
    if strict:
        raise ValueError(f"unmapped py123d actor label {raw!r}; no MetaDriveType mapping (strict mode)")
    if key not in _WARNED_UNKNOWN_LABELS:
        _WARNED_UNKNOWN_LABELS.add(key)
        logger.warning("py123d actor label %r has no MetaDriveType mapping; using OTHER", raw)
    return MetaDriveType.OTHER

# Default id assigned to the ego (self-driving car) track. ScenarioNet keys the
# ego track by ``metadata.sdc_id``; py123d ego_state_se3 has no token of its own.
EGO_ID = "ego"

# py123d traffic-light status names → ScenarioNet ``object_state`` strings.
# Unknown / unmapped statuses fall back to LANE_STATE_UNKNOWN.
_TL_STATUS_MAP = {
    "GREEN": "LANE_STATE_GO",
    "GO": "LANE_STATE_GO",
    "RED": "LANE_STATE_STOP",
    "STOP": "LANE_STATE_STOP",
    "YELLOW": "LANE_STATE_CAUTION",
    "AMBER": "LANE_STATE_CAUTION",
    "CAUTION": "LANE_STATE_CAUTION",
}


def py123d_to_scenario_description(
    scenario: Py123DScenarioData,
    *,
    sdc_id: str = EGO_ID,
    strict_labels: bool = False,
) -> SD:
    """Convert an absorbed py123d ScenarioState into a ``ScenarioDescription``.

    Args:
        scenario: The py123d ScenarioState (``Py123DScenarioData``) produced by
            :func:`~navsafe.scenario.py123d_adapter.scenario_from_py123d_scene`.
        sdc_id: Track id to assign to the ego vehicle.
        strict_labels: When True, raise on an actor label with no MetaDriveType
            mapping instead of warning and using ``OTHER`` (baseline-grade builds).

    Returns:
        A ScenarioNet-format :class:`ScenarioDescription` dict the env consumes.
    """
    log = training_scenario_from_py123d(scenario)
    return scenario_description_from_nexus_log(log, sdc_id=sdc_id, strict_labels=strict_labels)


def scenario_description_from_nexus_log(
    log: NexusScenarioLog,
    *,
    sdc_id: str = EGO_ID,
    strict_labels: bool = False,
) -> SD:
    """Project a per-frame ``NexusScenarioLog`` into a ``ScenarioDescription``."""
    num_steps = len(log.frames)

    tracks = {sdc_id: _empty_track(num_steps, "VEHICLE", sdc_id)}
    actor_tracks: dict[str, dict] = {}
    dynamic_map_states: dict[str, dict] = {}

    for t, frame in enumerate(log.frames):
        if frame.ego is not None:
            _fill_state(tracks[sdc_id]["state"], t, frame.ego)
        for actor in frame.actors:
            token = actor.track_token or f"obj_{id(actor):x}"
            track = actor_tracks.get(token)
            if track is None:
                track = _empty_track(num_steps, _map_agent_type(actor.label, strict=strict_labels), token)
                actor_tracks[token] = track
            _fill_state(track["state"], t, actor)
        _fill_traffic_lights(dynamic_map_states, t, num_steps, frame)

    tracks.update(actor_tracks)

    out = SD()
    out[SD.ID] = log.scenario_id
    out[SD.VERSION] = "py123d"
    out[SD.LENGTH] = num_steps
    out[SD.TRACKS] = tracks
    out[SD.DYNAMIC_MAP_STATES] = dynamic_map_states
    # A V-1 scenario's red light is authored, not logged (see
    # navsafe/benchmark/signal_override.py). Applied here because every
    # py123d-derived description passes through this function, so the evaluator,
    # the planners and the trace cannot end up reading different light states.
    from navsafe.benchmark import signal_override as _sig

    try:
        _sig.apply(out, _sig.from_env())
    except Exception as exc:                                 # noqa: BLE001
        # Loud: a malformed override means the scenario silently poses the
        # question the log posed, and the run would look like a policy result.
        raise ValueError(f"invalid {_sig.ENV_VAR}: {exc}") from exc
    out[SD.MAP_FEATURES] = _map_features_from_map_state(log.map_state)
    out[SD.METADATA] = {
        SD.SDC_ID: sdc_id,
        SD.COORDINATE: "world",
        SD.TIMESTEP: np.asarray(log.timestamps_us, dtype=np.float64),
        "dataset": log.dataset,
        "split": log.split,
        "scenario_id": log.scenario_id,
        "source": "py123d",
    }
    # Route intent — the ordered lane ids the log ego drove through. AV2
    # carries no route field (nuPlan-sourced py123d logs get one via the
    # "scenario" custom modality; ours have zero entries), so the branch
    # choice at each fork is reconstructed from the log trajectory here,
    # once, at conversion. This is the analog of nuPlan's
    # ``route_roadblock_ids``: which way to go, not a path, not speeds.
    # ``route_source="lane_graph_route"`` (route.py) consumes it.
    out[SD.METADATA]["route_lane_ids"] = derive_route_lane_ids(out)
    # Recenter to the ego's first pose so the whole scene lives in a small local
    # frame (see _recenter_to_frame0). Gated: the absolute-frame render path
    # (nurec_grpc origin_offset) still works when this is off.
    if (os.environ.get("PY123D_RECENTER", "1") != "0"):
        _recenter_to_frame0(out, sdc_id)
    # Per-lane speed targets. AV2 posts no speed limit on any lane, so without
    # this the IDM bank of every state planner collapses to a context-free
    # ``fraction x 15 m/s`` and the plan-execution pacing constant ends up
    # standing in for the missing map attribute (see speed_target's module
    # docstring). Derived once, here, and recorded in metadata so a run's
    # targets are read off the scenario rather than re-derived downstream.
    annotate_speed_targets(out)
    # Some source maps omit an intersection/turn connector even though the
    # logged expert traverses it.  In that case the metric's lane-polygon union
    # declares the expert off-road, making every policy impossible to score:
    # 891953217984568c had no drivable polygon for its U-turn and terminated on
    # the first policy frame despite a ground-truth replay.  Calibrate only the
    # proven hole with the expert vehicle's swept footprint.  This is much
    # narrower than growing every lane or exempting off-road poses.  It is
    # deliberately AFTER route + speed annotation: adding a low-speed support
    # lane before calibration changed 2792 unrelated fallback speed targets
    # from 11.18 to 3.98 m/s in the first 891 rerun.
    _add_logged_drivable_support(out, sdc_id)
    return out


def _add_logged_drivable_support(out: SD, sdc_id: str) -> int:
    """Add expert-centered connector polygons when the source map has holes.

    Returns the number of support polygons added.  A completely missing map is
    deliberately not repaired: that remains a data error.  With a real map,
    There are two distinct source-map defects and they must not share one
    width.  If the logged footprint itself leaves the source surface, a true
    connector is missing; reconstruct its vehicle turning envelope.  If the
    log fits but its declared closed-loop tracking tolerance does not, fill
    only that narrow seam.  The distinction preserves the wide U-turn support
    required by 891953... without widening every ordinary lane, while still
    repairing 0b125...'s acute junction seam.
    """
    # This calibration is for the nuPlan-derived NavSafe corpus.  Generated
    # PG/text2sim logs deliberately use arbitrary heading/motion combinations
    # in round-trip fixtures and require exact map cardinality; treating those
    # as map defects both invents semantics and breaks lossless Arrow IO.
    if str(out.get(SD.METADATA, {}).get("dataset", "")).lower() != "nuplan":
        return 0

    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    features = out.get(SD.MAP_FEATURES, {})
    lane_polys = []
    for feature in features.values():
        if not MetaDriveType.is_lane(feature.get(SD.TYPE)):
            continue
        points = feature.get(SD.POLYGON)
        if points is None or len(points) < 3:
            continue
        poly = Polygon(np.asarray(points, dtype=np.float64)[:, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            lane_polys.append(poly)
    if not lane_polys:
        return 0

    track = out.get(SD.TRACKS, {}).get(sdc_id, {})
    state = track.get(SD.STATE, {})
    positions = np.asarray(state.get(SD.POSITION, []), dtype=np.float64)
    headings = np.asarray(state.get(SD.HEADING, []), dtype=np.float64).reshape(-1)
    lengths = np.asarray(state.get("length", []), dtype=np.float64).reshape(-1)
    widths = np.asarray(state.get("width", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
    n = min(len(positions), len(headings), len(lengths), len(widths), len(valid))
    if n == 0:
        return 0

    footprints = []
    centers = []
    for i in range(n):
        if (not valid[i] or positions.shape[1] < 2
                or not np.all(np.isfinite(positions[i, :2]))
                or not np.isfinite(headings[i])
                or lengths[i] <= 0.0 or widths[i] <= 0.0):
            continue
        x, y = (float(positions[i, 0]), float(positions[i, 1]))
        c, s = np.cos(headings[i]), np.sin(headings[i])
        ex, ey = 0.5 * float(lengths[i]), 0.5 * float(widths[i])
        footprints.append(Polygon([
            (x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey))
        ]))
        centers.append(positions[i].copy())
    if not footprints:
        return 0

    centerline = np.asarray(centers, dtype=np.float64)
    tracking_tolerance_m = 0.75
    valid_dimensions = (
        valid[:n]
        & np.isfinite(lengths[:n]) & (lengths[:n] > 0.0)
        & np.isfinite(widths[:n]) & (widths[:n] > 0.0)
    )
    # This is the same 0.3 m seam tolerance as DrivableAreaProxy.  Keep the
    # value local to avoid importing the evaluation/scorer dependency tree.
    certified = unary_union(lane_polys).buffer(0.3)
    exact_log_covered = all(certified.covers(footprint)
                            for footprint in footprints)

    if not exact_log_covered:
        # A real connector is absent.  During a tight turn, the vehicle can
        # occupy its half-diagonal laterally relative to the centerline.  The
        # logged line proves the topology; the vehicle dimensions plus the
        # measured planner->LQR seam bound the surface needed to track it.
        # 891953... needs this 6.6 m U-turn envelope; a standard 3.5 m strip
        # and the narrower footprint-margin repair both deadlock there.
        if np.any(valid_dimensions):
            half_diagonal = np.hypot(
                0.5 * lengths[:n][valid_dimensions],
                0.5 * widths[:n][valid_dimensions],
            )
            support_half_width = max(
                1.75,
                float(np.percentile(half_diagonal, 95.0))
                + tracking_tolerance_m,
            )
        else:  # guarded above in normal data
            support_half_width = 1.75
        support_surface = unary_union([
            *footprints,
            LineString(centerline[:, :2]).buffer(
                support_half_width, cap_style=2, join_style=1),
        ])
        repair_mode = "missing_turning_envelope"
        reason = "logged ego footprint leaves source lane-polygon union"
    else:
        # The expert fits, so this is not permission to create a broad lane.
        # Grow its oriented sweep only by the measured controller seam and
        # materialize the portion not already certified by the source map.
        if np.any(valid_dimensions):
            support_half_width = (
                float(np.percentile(
                    0.5 * widths[:n][valid_dimensions], 95.0))
                + tracking_tolerance_m)
        else:  # guarded above in normal data
            support_half_width = 0.925 + tracking_tolerance_m
        required_sweep = unary_union([
            *(footprint.buffer(tracking_tolerance_m, join_style=1)
              for footprint in footprints),
            LineString(centerline[:, :2]).buffer(
                support_half_width, cap_style=2, join_style=1),
        ])
        if certified.covers(required_sweep):
            return 0
        support_surface = required_sweep.difference(certified)
        repair_mode = "tracking_tolerance_seam"
        reason = (
            "logged ego tracking envelope leaves source lane-polygon union")

    polygons = ([support_surface] if support_surface.geom_type == "Polygon"
                else [g for g in getattr(support_surface, "geoms", ())
                      if g.geom_type == "Polygon" and not g.is_empty])
    if not polygons:
        return 0

    velocity = np.asarray(state.get("velocity", []), dtype=np.float64)
    support_speed = None
    if velocity.ndim == 2 and velocity.shape[1] >= 2 and len(velocity):
        speeds = np.linalg.norm(velocity[:n, :2], axis=1)
        speeds = speeds[valid[:n] & np.isfinite(speeds) & (speeds > 0.1)]
        if len(speeds):
            support_speed = float(np.percentile(speeds, 85.0))
    for i, poly in enumerate(polygons):
        exterior = np.asarray(poly.exterior.coords[:-1], dtype=np.float64)
        feature = {
            SD.TYPE: MetaDriveType.LANE_SURFACE_UNSTRUCTURE,
            SD.POLYLINE: centerline,
            SD.POLYGON: exterior,
            "provenance": "logged_ego_standard_lane_connector",
        }
        if support_speed is not None:
            feature[SPEED_LIMIT_KEY] = support_speed
            feature[SPEED_LIMIT_SOURCE_KEY] = "logged_drivable_support_p85"
        features[f"__logged_drivable_support_{i}"] = feature
    out[SD.METADATA]["logged_drivable_support"] = {
        "version": 5,
        "polygon_count": len(polygons),
        "connector_width_m": 2.0 * support_half_width,
        "controller_tracking_tolerance_m": tracking_tolerance_m,
        "repair_mode": repair_mode,
        "reason": reason,
    }
    return len(polygons)


def _recenter_to_frame0(out: SD, sdc_id: str) -> None:
    """Shift every xy coordinate so the ego's first valid pose is the origin.

    nuPlan coords are absolute UTM (~3.3e5 easting, ~4.7e6 northing). float32 at
    that magnitude has a ~0.5 m ULP, and the env/renderer/map all cast positions
    and lane polylines to float32 -> ego, actors, and lanes snap to a 0.5 m grid
    (ego-camera sawtooth + jagged lane overlays). Working in a local frame keeps
    float32 sub-mm. Records the offset so consumers can recover UTM
    (utm = local + scenario_origin_xy). Relative fields (heading/velocity) and z
    are untouched. Idempotent per SD (called once at build)."""
    tracks = out.get(SD.TRACKS, {})
    ego = tracks.get(sdc_id)
    if ego is None:
        return
    pos = np.asarray(ego[SD.STATE][SD.POSITION], dtype=np.float64)
    valid = ego[SD.STATE].get("valid")
    idx = 0
    if valid is not None:
        nz = np.flatnonzero(np.asarray(valid))
        if len(nz):
            idx = int(nz[0])
    ox, oy = float(pos[idx, 0]), float(pos[idx, 1])

    def shift(arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim >= 1 and a.shape[-1] >= 2:
            a = a.copy()
            a[..., 0] -= ox
            a[..., 1] -= oy
        return a

    for tr in tracks.values():
        st = tr.get(SD.STATE, {})
        if SD.POSITION in st:
            st[SD.POSITION] = shift(st[SD.POSITION])
    for feat in out.get(SD.MAP_FEATURES, {}).values():
        for key in (SD.POLYLINE, SD.POLYGON, SD.LEFT_BOUNDARIES, SD.RIGHT_BOUNDARIES):
            if feat.get(key) is not None:
                feat[key] = shift(feat[key])
    for dm in out.get(SD.DYNAMIC_MAP_STATES, {}).values():
        if dm.get("stop_point") is not None:
            dm["stop_point"] = shift(dm["stop_point"])
    out[SD.METADATA]["scenario_origin_xy"] = [ox, oy]
    out[SD.METADATA][SD.COORDINATE] = "local_frame0"


# ---------------------------------------------------------------------------
# Track assembly
# ---------------------------------------------------------------------------

def _empty_track(num_steps: int, obj_type: str, object_id: str) -> dict:
    """ScenarioNet object-track template (mirrors the converters' layout)."""
    return {
        SD.TYPE: obj_type,
        SD.STATE: {
            # float64: nuPlan positions are absolute UTM (~4.7e6 northing), where
            # float32's ULP is ~0.5 m. Keeping them float64 preserves sub-mm ego
            # precision so the eval render path can re-reference (pos - origin_offset)
            # into the recon's local frame without inheriting a 0.5 m grid.
            SD.POSITION: np.zeros([num_steps, 3], dtype=np.float64),
            "length": np.zeros([num_steps], dtype=np.float32),
            "width": np.zeros([num_steps], dtype=np.float32),
            "height": np.zeros([num_steps], dtype=np.float32),
            SD.HEADING: np.zeros([num_steps], dtype=np.float32),
            "velocity": np.zeros([num_steps, 2], dtype=np.float32),
            "velocity_valid": np.zeros([num_steps], dtype=bool),
            "valid": np.zeros([num_steps], dtype=bool),
        },
        SD.METADATA: {
            "track_length": num_steps,
            "type": obj_type,
            "object_id": object_id,
        },
    }


def _fill_state(state: dict, t: int, obj) -> None:
    """Write one (ego or actor) state into the per-step arrays at index ``t``."""
    state[SD.POSITION][t] = (float(obj.x), float(obj.y), float(obj.z))
    state[SD.HEADING][t] = float(obj.heading)
    state["velocity"][t] = (float(obj.vx), float(obj.vy))
    state["velocity_valid"][t] = (
        getattr(obj, "velocity_valid", False) is True
        and np.isfinite(state["velocity"][t]).all())
    state["length"][t] = float(obj.length)
    state["width"][t] = float(obj.width)
    state["height"][t] = float(obj.height)
    state["valid"][t] = True


def _fill_traffic_lights(
    dynamic_map_states: dict,
    t: int,
    num_steps: int,
    frame,
) -> None:
    """Accumulate per-step traffic-light status keyed by controlled lane id."""
    for tl in frame.traffic_lights:
        lane_id = str(tl.lane_id)
        entry = dynamic_map_states.get(lane_id)
        if entry is None:
            entry = {
                SD.TYPE: "TRAFFIC_LIGHT",
                SD.STATE: {
                    "object_state": ["LANE_STATE_UNKNOWN"] * num_steps,
                    # Missing detections and explicit UNKNOWN detections share
                    # the runtime state, but are different source evidence.
                    "observation_present": [False] * num_steps,
                },
                "lane": lane_id,
                "stop_point": np.zeros([3], dtype=np.float32),
                SD.METADATA: {"track_length": num_steps, "object_id": lane_id},
            }
            dynamic_map_states[lane_id] = entry
        status = str(tl.status or "").upper()
        entry[SD.STATE]["object_state"][t] = _TL_STATUS_MAP.get(status, "LANE_STATE_UNKNOWN")
        entry[SD.STATE]["observation_present"][t] = True


# ---------------------------------------------------------------------------
# Map features
# ---------------------------------------------------------------------------

def _map_features_from_map_state(map_state: Optional[NexusMapState]) -> dict:
    """Project map lanes/lines into ScenarioNet ``map_features`` polylines."""
    features: dict[str, dict] = {}
    if map_state is None:
        return features

    intersection_lane_groups = {
        str(group_id)
        for group_id, group in map_state.lane_groups.items()
        if getattr(group, "intersection_id", None) is not None
    }
    for lane_id, lane in map_state.lanes.items():
        polyline = _as_polyline(getattr(lane, "centerline", None))
        if polyline is None:
            continue
        feature: dict[str, object] = {
            SD.TYPE: _map_lane_type(lane.lane_type), SD.POLYLINE: polyline}
        if getattr(lane, "lane_group_id", None) is not None:
            # Preserve the parent topology needed to adapt CaRL's
            # roadblock-connector loop removal at the correct level.
            feature["lane_group_id"] = str(lane.lane_group_id)
        if (getattr(lane, "lane_group_id", None) is not None
                and str(lane.lane_group_id) in intersection_lane_groups):
            # Preserve the nuPlan semantic-map INTERSECTION predicate needed
            # by PDM-Closed's TTC fault cone. Without this, every connector
            # was treated as ordinary road and the 30-degree cone was never
            # widened to the reference's not-behind (<=150-degree) rule.
            feature["is_intersection"] = True
        # The lane's true footprint. py123d ships it (Lane.shapely_polygon ->
        # NexusLaneState.polygon) but it was never forwarded here, so every
        # downstream consumer saw map_features with ZERO polygons and fell back
        # to synthesising one: LaneProxy extrudes centerline.buffer(1.75), i.e.
        # a uniform 3.5 m ribbon. Real lanes in this data measure 3.09-7.33 m
        # wide (median 3.72) and flare through turns, so the fallback is up to
        # 2x too narrow and the wrong shape. EPDMS ``dac`` tests all four ego
        # corners against these polygons, so a car legitimately tracking a wide
        # or flaring lane was scored as leaving the drivable area (bbb77289
        # f40: every proposal dac=0.250 while the path centreline sat 11/11
        # inside the lanes, EPDMS 0.184 -> gate fail).
        poly = _as_polyline(getattr(lane, "polygon", None))
        if poly is not None and poly.shape[0] >= 3:
            feature[SD.POLYGON] = poly
        # The lane's posted speed limit. py123d carries it
        # (``Lane.speed_limit_mps`` -> ``NexusLaneState.speed_limit_mps``) but
        # it was never forwarded, so even a source that HAS limits (nuPlan)
        # reached the planner as "no annotation" and every IDM policy fell back
        # to ``fraction x 15 m/s``. AV2 populates None here — that scene's
        # targets come from ``speed_target.annotate_speed_targets`` instead,
        # which leaves a dataset limit untouched wherever one exists.
        limit = getattr(lane, "speed_limit_mps", None)
        if limit is not None and np.isfinite(limit) and float(limit) > 0.0:
            feature[SPEED_LIMIT_KEY] = float(limit)
            feature[SPEED_LIMIT_SOURCE_KEY] = "dataset"
        # Lane-graph connectivity. py123d ships it (Lane.successor_ids /
        # predecessor_ids -> NexusLaneState) but it was never forwarded, so
        # map_features carried ZERO exit_lanes on every scene. That silently
        # disabled ``route_source="lane_graph_search"`` -- the documented
        # oracle-free route mode (route.py:28-35) needs exit_lanes to walk the
        # successor graph. With none, it degraded to _polyline_from_nearest_lane
        # (a SINGLE lane's centerline), which is where the planner's stub routes
        # came from: at 9c380aeb f25 a 6.65 m/s ego got 9.7 m of route, tracked
        # it to the end and parked, collapsing ep and failing the gate. The
        # gt_future fallback masked this whenever the hint was usable.
        entry = tuple(str(x) for x in (getattr(lane, "predecessor_ids", ()) or ()))
        exit_ = tuple(str(x) for x in (getattr(lane, "successor_ids", ()) or ()))
        if entry:
            feature[SD.ENTRY] = list(entry)
        if exit_:
            feature[SD.EXIT] = list(exit_)
        left = _as_polyline(getattr(lane, "left_boundary", None))
        right = _as_polyline(getattr(lane, "right_boundary", None))
        if left is not None:
            feature[SD.LEFT_BOUNDARIES] = left
        if right is not None:
            feature[SD.RIGHT_BOUNDARIES] = right
        features[str(lane_id)] = feature

    # Lanes keep their raw id as the key so dynamic_map_states traffic lights
    # (which reference the controlled lane id) still resolve. Road edges/lines
    # are namespaced by layer: py123d numbers object ids per-layer from 0, so
    # without a prefix a lane, an edge, and a line all id ``0`` would collide in
    # this flat dict and overwrite one another.
    for prefix, line_dict in (("road_edge", map_state.road_edges), ("road_line", map_state.road_lines)):
        for line_id, line in line_dict.items():
            polyline = _as_polyline(getattr(line, "polyline", None))
            if polyline is None:
                continue
            features[f"{prefix}_{line_id}"] = {
                SD.TYPE: _map_line_type(getattr(line, "semantic_type", None)
                                        or getattr(line, "layer", None)),
                SD.POLYLINE: polyline,
            }

    # Crosswalks. py123d parses them (nuPlan's own ``crosswalks`` layer ->
    # ``MapLayer.CROSSWALK`` -> ``NexusMapState.crosswalks``) and a converted
    # host carries hundreds — 354 on 17b0992157365222, 233 on 03b66343e1ac5d68
    # — but this function emitted lanes and lines only, so every consumer saw a
    # world with no pedestrian crossings in it. A scenario whose event IS
    # someone crossing the road then had nowhere real to put them: the actor
    # was placed mid-block and sent perpendicular to the ego's route, which is
    # not where people cross.
    #
    # POLYGON is the footprint; POLYLINE is the crossing's LONG AXIS, kerb to
    # kerb, because that is the line an actor walks and it is what a reference
    # has to be. Taking the axis rather than the polygon outline matters: the
    # outline's first edge is an arbitrary corner-to-corner segment, and
    # walking it would send the actor along the kerb instead of across.
    for cw_id, crosswalk in (getattr(map_state, "crosswalks", None) or {}).items():
        axis = _surface_long_axis(getattr(crosswalk, "polygon", None))
        if axis is None:
            continue
        crosswalk_feature: dict[str, object] = {
            SD.TYPE: MetaDriveType.CROSSWALK,
            SD.POLYLINE: axis,
        }
        polygon = _as_polyline(getattr(crosswalk, "polygon", None))
        if polygon is not None and polygon.shape[0] >= 3:
            crosswalk_feature[SD.POLYGON] = polygon
        lane_ids = tuple(str(x) for x in (getattr(crosswalk, "lane_ids", ()) or ()))
        if lane_ids:
            # Which lanes this crossing spans — the ones an actor on it enters.
            crosswalk_feature[SD.ENTRY] = list(lane_ids)
        features[f"crosswalk_{cw_id}"] = crosswalk_feature

    # Carparks and intersections. Both are py123d layers the extractor already
    # carries (``NexusMapState.carparks`` / ``.intersections``) that were never
    # forwarded. CaRL's PDM-Closed reads both from the nuPlan map:
    # ``get_drivable_area_map`` unions CARPARK_AREA polygons into the drivable
    # surface (a proposal over a parking apron is NOT off-road), and the TTC
    # at-fault test widens its 30-degree cone whenever the ego rear axle is
    # inside an INTERSECTION polygon (``map_api.is_in_layer``). Polygon-only
    # features; the closed ring doubles as ``polyline`` for consumers that
    # index it unguarded.
    for cp_id, carpark in (getattr(map_state, "carparks", None) or {}).items():
        polygon = _as_polyline(getattr(carpark, "polygon", None))
        if polygon is None or polygon.shape[0] < 3:
            continue
        features[f"carpark_{cp_id}"] = {
            SD.TYPE: MetaDriveType.CARPARK_AREA,
            SD.POLYGON: polygon,
            SD.POLYLINE: np.vstack([polygon, polygon[:1]]),
        }
    for ix_id, intersection in (getattr(map_state, "intersections", None) or {}).items():
        polygon = _as_polyline(getattr(intersection, "polygon", None))
        if polygon is None or polygon.shape[0] < 3:
            continue
        features[f"intersection_{ix_id}"] = {
            SD.TYPE: MetaDriveType.INTERSECTION,
            SD.POLYGON: polygon,
            SD.POLYLINE: np.vstack([polygon, polygon[:1]]),
            "lane_group_ids": [
                str(x) for x in (getattr(intersection, "lane_group_ids", ()) or ())],
        }

    return features


def _surface_long_axis(polygon) -> Optional[np.ndarray]:
    """A surface polygon -> the 2-point line along its LONGEST dimension.

    A crosswalk is a rectangle-ish polygon and the useful line through it is
    kerb to kerb, not along the kerb. The principal axis of the vertices gives
    that without assuming the polygon has exactly four corners or that its
    vertices start anywhere in particular — real ones from this data have
    4 to 20+ vertices and no consistent winding.

    Height comes from the polygon's own z where it has one, so an actor placed
    on this line sits on the road rather than at z=0.
    """
    pts = _as_polyline(polygon)
    if pts is None or pts.shape[0] < 3:
        return None
    xy = np.asarray(pts[:, :2], np.float64)
    centre = xy.mean(axis=0)
    centred = xy - centre
    # Principal axis: the eigenvector of the covariance with the larger
    # eigenvalue, i.e. the direction the polygon is longest in.
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vh[0]
    t = centred @ axis
    lo, hi = float(t.min()), float(t.max())
    if hi - lo < 1.0:                      # degenerate: not a crossing
        return None
    ends = np.stack([centre + lo * axis, centre + hi * axis])
    if pts.shape[1] >= 3:
        z = float(np.median(pts[:, 2]))
        ends = np.concatenate([ends, np.full((2, 1), z)], axis=1)
    return ends.astype(pts.dtype)


def _as_polyline(arr) -> Optional[np.ndarray]:
    """Return an (N, >=2) polyline, or None if empty/degenerate.

    float32 by default, but float64 under PY123D_RECENTER: map polylines are
    absolute UTM (~4.7e6), where float32's ~0.5 m ULP snaps every lane vertex to
    a grid (the jagged-lane overlay). The recenter pass then shifts them into a
    local frame where even a later float32 cast stays sub-mm.
    """
    if arr is None:
        return None
    _dt = np.float64 if (os.environ.get("PY123D_RECENTER", "1") != "0") else np.float32
    poly = np.asarray(arr, dtype=_dt)
    if poly.ndim != 2 or poly.shape[0] == 0 or poly.shape[1] < 2:
        return None
    return poly


__all__ = [
    "EGO_ID",
    "py123d_to_scenario_description",
    "scenario_description_from_nexus_log",
]
