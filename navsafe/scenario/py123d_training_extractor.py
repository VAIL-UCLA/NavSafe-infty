"""Extract NavSafe training objects from py123d-backed scenario data."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

import numpy as np

from navsafe.scenario.py123d_schema import Py123DScenarioData, to_serializable
from navsafe.scenario.training_schema import (
    NexusActorState,
    NexusEgoState,
    NexusFrameState,
    NexusIntersectionState,
    NexusLaneGroupState,
    NexusLaneState,
    NexusLineState,
    NexusMapState,
    NexusRouteState,
    NexusScenarioLog,
    NexusSensorReference,
    NexusSurfaceState,
    NexusTrafficLightState,
)


SURFACE_LAYER_ATTRS = {
    "crosswalk": "crosswalks",
    "walkway": "walkways",
    "carpark": "carparks",
    "generic_drivable": "generic_drivable",
    "stop_zone": "stop_zones",
    "speed_bump": "speed_bumps",
}

LINE_LAYER_ATTRS = {
    "road_edge": "road_edges",
    "road_line": "road_lines",
}

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Missing-required-field accounting
# ---------------------------------------------------------------------------
#
# ``_float_attr`` coerces an absent/non-numeric value to a default. That is
# legitimate for OPTIONAL fields (SE2 poses carry no z; many datasets carry no
# acceleration, angular velocity, or tire steering angle; SE2 boxes carry no
# height) but silently corrupting for REQUIRED ones — a missing velocity field
# used to train as "stationary" with no trace. Required call sites now pass
# ``field=`` so the drop is counted and surfaced; optional ones keep the
# documented silent default. Returned values are unchanged in every case: this
# is telemetry, never a behavior change for well-formed data.
#
# Surfacing has two channels:
#   * warn-once per field name per process (deliberately process-global — that
#     is what "once" means; tests reset via ``reset_missing_field_state``),
#     with running totals queryable via ``missing_field_totals``;
#   * a per-extraction ``Counter`` scoped by the ``collect_missing_fields``
#     context manager (ContextVar-backed, so nothing leaks across extractions
#     or threads), which callers such as the BC cache builder log/report.

_missing_fields_var: ContextVar[Optional[Counter[str]]] = ContextVar(
    "navsafe_py123d_missing_fields", default=None
)
_warned_missing_fields: set[str] = set()
_missing_field_totals: Counter[str] = Counter()


@contextmanager
def collect_missing_fields() -> Iterator[Counter[str]]:
    """Collect required-field drop counts for the enclosed extraction calls.

    Yields a ``Counter`` mapping field name (e.g. ``"ego.vx"``) to the number
    of times that required field was missing/non-numeric and defaulted while
    the context was active. Nested contexts each see only their own drops.
    """
    counter: Counter[str] = Counter()
    token = _missing_fields_var.set(counter)
    try:
        yield counter
    finally:
        _missing_fields_var.reset(token)


def missing_field_totals() -> dict[str, int]:
    """Process-wide counts of required-field drops, for logging/monitoring."""
    return dict(_missing_field_totals)


def reset_missing_field_state() -> None:
    """Reset warn-once markers and totals (test isolation)."""
    _warned_missing_fields.clear()
    _missing_field_totals.clear()


def _record_missing_field(field: str) -> None:
    _missing_field_totals[field] += 1
    active = _missing_fields_var.get()
    if active is not None:
        active[field] += 1
    if field not in _warned_missing_fields:
        _warned_missing_fields.add(field)
        _logger.warning(
            "py123d extraction: required field %r missing or non-numeric; defaulting to 0.0. "
            "Warning once per field; occurrences are counted (missing_field_totals() / "
            "collect_missing_fields()).",
            field,
        )


def training_scenario_from_py123d(scenario: Py123DScenarioData) -> NexusScenarioLog:
    """Materialize a single py123d scene into NavSafe training objects.

    This is scene-local by design. Dataset-scale loaders should call this one
    scene at a time or use ``training_frame_from_py123d`` for sampled frames.
    """
    map_state = training_map_from_py123d(scenario)
    window = scenario.extras.get("frame_window")
    start, stop = (0, scenario.number_of_iterations) if window is None else window
    start = max(0, int(start))  # NexusScenarioLog does not expose history frames.
    stop = min(int(stop), scenario.number_of_iterations)
    if start >= stop:
        raise ValueError(f"no runtime frames in py123d window {window}")
    frames = [training_frame_from_py123d(scenario, idx) for idx in range(start, stop)]
    if window is not None:
        # The sparse py123d carrier retains absolute keys, while runtime
        # tracks and NexusScenarioLog use contiguous, zero-based iterations.
        from dataclasses import replace

        frames = [replace(frame, iteration=i) for i, frame in enumerate(frames)]
    log = NexusScenarioLog(
        scenario_id=scenario.scenario_id,
        dataset=scenario.dataset,
        split=scenario.split,
        location=scenario.location,
        log_name=scenario.log_name,
        timestamps_us=list(scenario.timestamps_us[start:stop]),
        map_state=map_state,
        frames=frames,
        sensor_metadata=_sensor_metadata_summary(scenario),
        source_metadata={
            "scene_metadata": to_serializable(scenario.scene_metadata),
            "log_metadata": to_serializable(scenario.log_metadata),
            "map_metadata": to_serializable(scenario.map_metadata),
            "modality_keys": scenario.modality_keys,
            **({"frame_window": [start, stop]} if window is not None else {}),
        },
    )
    log.validate()
    return log


def training_frame_from_py123d(scenario: Py123DScenarioData, iteration: int) -> NexusFrameState:
    py_frame = scenario.get_frame_state(iteration)
    # Ego is the only hard requirement (it becomes the sdc track). Box and
    # traffic-light modalities are dataset-dependent — AV2 sensor, e.g., has no
    # traffic lights — so a missing stream yields an empty list, not an error.
    if py_frame.ego_state is None:
        raise ValueError(f"missing ego_state_se3 at frame {iteration}")

    custom_modalities = _canonical_custom_modalities(py_frame.custom_modalities)
    custom = _custom_payload_data(custom_modalities.get("custom.scenario"))
    return NexusFrameState(
        scenario_id=scenario.scenario_id,
        iteration=iteration,
        timestamp_us=py_frame.timestamp_us,
        ego=_extract_ego(py_frame.ego_state),
        actors=tuple(_extract_actor(actor) for actor in _iter_box_detections(py_frame.box_detections)),
        traffic_lights=tuple(_extract_traffic_light(det) for det in _iter_traffic_lights(py_frame.traffic_light_detections)),
        route=_route_state_from_custom(custom, "custom.scenario.route_roadblock_ids"),
        sensor_references=_sensor_references_from_custom(custom, py_frame.timestamp_us),
        custom_modalities=custom_modalities,
    )


def training_frame_from_py123d_scene_api(scene: Any, scenario_id: str, iteration: int) -> NexusFrameState:
    """Extract one training frame directly from a py123d SceneAPI.

    This avoids materializing every frame into ``Py123DScenarioData`` and is the
    preferred path for dataset-scale validation or DataLoader sampling.
    """
    ego_state = _call(scene, "get_ego_state_se3_at_iteration", iteration)
    box_detections = _call(scene, "get_box_detections_se3_at_iteration", iteration)
    traffic_lights = _call(scene, "get_traffic_light_detections_at_iteration", iteration)
    # Ego is the only hard requirement; box/traffic-light streams are
    # dataset-dependent (AV2 sensor has no traffic lights) → empty, not error.
    if ego_state is None:
        raise ValueError(f"missing ego_state_se3 at frame {iteration}")
    timestamp_us = _timestamp_us(_attr(ego_state, "timestamp"))
    custom_modalities = _custom_modalities_from_scene_api(scene, iteration)
    custom = _custom_payload_data(custom_modalities.get("custom.scenario"))
    return NexusFrameState(
        scenario_id=scenario_id,
        iteration=iteration,
        timestamp_us=timestamp_us,
        ego=_extract_ego(ego_state),
        actors=tuple(_extract_actor(actor) for actor in _iter_box_detections(box_detections)),
        traffic_lights=tuple(_extract_traffic_light(det) for det in _iter_traffic_lights(traffic_lights)),
        route=_route_state_from_custom(custom, "custom.scenario.route_roadblock_ids"),
        sensor_references=_sensor_references_from_custom(custom, timestamp_us),
        custom_modalities=custom_modalities,
    )


def training_map_from_py123d(scenario: Py123DScenarioData) -> NexusMapState:
    map_state = NexusMapState(location=scenario.location)
    for layer in _available_layers(scenario):
        layer_name = str(layer)
        for object_id, obj in _iter_map_objects(scenario, layer_name):
            if layer_name == "lane":
                lane = _extract_lane(object_id, obj)
                map_state.lanes[lane.lane_id] = lane
            elif layer_name == "lane_group":
                lane_group = _extract_lane_group(object_id, obj)
                map_state.lane_groups[lane_group.lane_group_id] = lane_group
            elif layer_name == "intersection":
                intersection = _extract_intersection(object_id, obj)
                map_state.intersections[intersection.intersection_id] = intersection
            elif layer_name in SURFACE_LAYER_ATTRS:
                surface = _extract_surface(object_id, layer_name, obj)
                getattr(map_state, SURFACE_LAYER_ATTRS[layer_name])[surface.object_id] = surface
            elif layer_name in LINE_LAYER_ATTRS:
                line = _extract_line(object_id, layer_name, obj)
                getattr(map_state, LINE_LAYER_ATTRS[layer_name])[line.object_id] = line

    map_state.lane_point_features = _build_lane_point_features(map_state.lanes.values())
    return map_state


def _extract_ego(ego: Any) -> NexusEgoState:
    # x/y/heading come from the bounding-box CENTRE (center_se3): everything
    # downstream (origin_offset, the recon origin anchor, the overlay) is keyed
    # to it, and moving to the rear axle would shift the ego ~half a wheelbase
    # back. But the CENTRE z sits ~height/2 above the road, which lifted the
    # rendered camera and every route-anchored placement by ~0.85 m (the
    # "ego-z-to-ground" saga). Take z from the GROUND reference instead
    # (rear_axle_se3 ≈ imu ≈ on the road for WOD), so the ego z means the road
    # under the ego — matching the lidar-supervised recon road and removing the
    # magic 0.85 drop. Falls back to center z if no ground pose exists.
    pose = _first_attr(ego, "center_se3", "center_se2", "rear_axle_se3", "rear_axle_se2", "imu_se3", "imu_se2")
    ground_pose = _first_attr(ego, "rear_axle_se3", "imu_se3", "center_se3",
                              "rear_axle_se2", "imu_se2", "center_se2")
    # SE2 egos carry ``dynamic_state_se2`` — read it too so a real SE2
    # velocity is neither dropped nor falsely counted as a missing field.
    # (All first-party paths fetch SE3 egos; this is a forward guard.)
    dyn = _first_attr(ego, "dynamic_state_se3", "dynamic_state_se2")
    velocity = _first_attr(dyn, "velocity_3d", "velocity_2d")
    acceleration = _first_attr(dyn, "acceleration_3d", "acceleration_2d")
    angular_velocity = _attr(dyn, "angular_velocity")
    # SE3 dynamic states carry a vector angular velocity (yaw rate is its z);
    # SE2 ones carry a bare scalar. Read both so an SE2 yaw rate is not
    # silently zeroed by asking a float for a ``.z`` attribute.
    if angular_velocity is not None and not hasattr(angular_velocity, "z"):
        scalar_yaw_rate = _optional_float(angular_velocity)
        yaw_rate = 0.0 if scalar_yaw_rate is None else scalar_yaw_rate
    else:
        yaw_rate = _float_attr(angular_velocity, "z")
    bbox = _first_attr(ego, "bounding_box_se3", "bounding_box_se2")
    # Required (counted+warned when missing): pose x/y/heading, planar
    # velocity, box length/width. Optional (documented silent defaults): z
    # (SE2 poses carry none), vz (2D velocity), acceleration, yaw_rate,
    # steering_angle (dataset-dependent), height (SE2 boxes carry none).
    return NexusEgoState(
        timestamp_us=_timestamp_us(_attr(ego, "timestamp")),
        x=_float_attr(pose, "x", field="ego.x"),
        y=_float_attr(pose, "y", field="ego.y"),
        z=_float_attr(ground_pose, "z"),
        heading=_float_attr(pose, "yaw", field="ego.heading"),
        vx=_float_attr(velocity, "x", field="ego.vx"),
        vy=_float_attr(velocity, "y", field="ego.vy"),
        vz=_float_attr(velocity, "z"),
        ax=_float_attr(acceleration, "x"),
        ay=_float_attr(acceleration, "y"),
        az=_float_attr(acceleration, "z"),
        yaw_rate=yaw_rate,
        steering_angle=_float_attr(ego, "tire_steering_angle"),
        length=_float_attr(bbox, "length", field="ego.length"),
        width=_float_attr(bbox, "width", field="ego.width"),
        height=_float_attr(bbox, "height"),
    )


def _extract_actor(actor: Any) -> NexusActorState:
    bbox = _first_attr(actor, "bounding_box_se3", "bounding_box_se2")
    pose = _first_attr(actor, "center_se3", "center_se2", "bounding_box_se3", "bounding_box_se2")
    velocity = _first_attr(actor, "velocity_3d", "velocity_2d")
    planar_velocity = (_optional_float(_attr(velocity, "x")),
                       _optional_float(_attr(velocity, "y")))
    velocity_valid = all(v is not None and np.isfinite(v) for v in planar_velocity)
    attrs = _attr(actor, "attributes")
    # Required/optional split mirrors ``_extract_ego`` (see comment there).
    return NexusActorState(
        track_token=_optional_str(_attr(attrs, "track_token")) or "",
        label=_enum_name(_attr(attrs, "label")),
        x=_float_attr(pose, "x", field="actor.x"),
        y=_float_attr(pose, "y", field="actor.y"),
        z=_float_attr(pose, "z"),
        heading=_float_attr(pose, "yaw", field="actor.heading"),
        vx=_float_attr(velocity, "x", field="actor.vx"),
        vy=_float_attr(velocity, "y", field="actor.vy"),
        vz=_float_attr(velocity, "z"),
        length=_float_attr(bbox, "length", field="actor.length"),
        width=_float_attr(bbox, "width", field="actor.width"),
        height=_float_attr(bbox, "height"),
        num_lidar_points=_optional_int(_attr(attrs, "num_lidar_points")),
        velocity_valid=velocity_valid,
    )


def _extract_traffic_light(det: Any) -> NexusTrafficLightState:
    return NexusTrafficLightState(
        lane_id=str(_attr(det, "lane_id")),
        status=_enum_name(_attr(det, "status")),
    )


def _extract_lane(object_id: Any, obj: Any) -> NexusLaneState:
    return NexusLaneState(
        lane_id=str(object_id),
        lane_type=_enum_name(_attr(obj, "lane_type")),
        lane_group_id=_optional_str(_attr(obj, "lane_group_id")),
        left_lane_id=_optional_str(_attr(obj, "left_lane_id")),
        right_lane_id=_optional_str(_attr(obj, "right_lane_id")),
        predecessor_ids=_str_tuple(_attr(obj, "predecessor_ids", [])),
        successor_ids=_str_tuple(_attr(obj, "successor_ids", [])),
        speed_limit_mps=_optional_float(_attr(obj, "speed_limit_mps")),
        centerline=_polyline_array(_attr(obj, "centerline")),
        left_boundary=_polyline_array(_attr(obj, "left_boundary")),
        right_boundary=_polyline_array(_attr(obj, "right_boundary")),
        polygon=_polygon_array(_first_attr(obj, "shapely_polygon", "polygon")),
    )


def _extract_lane_group(object_id: Any, obj: Any) -> NexusLaneGroupState:
    return NexusLaneGroupState(
        lane_group_id=str(object_id),
        lane_ids=_str_tuple(_attr(obj, "lane_ids", [])),
        intersection_id=_optional_str(_attr(obj, "intersection_id")),
        predecessor_ids=_str_tuple(_attr(obj, "predecessor_ids", [])),
        successor_ids=_str_tuple(_attr(obj, "successor_ids", [])),
        polygon=_polygon_array(_first_attr(obj, "shapely_polygon", "polygon")),
    )


def _extract_intersection(object_id: Any, obj: Any) -> NexusIntersectionState:
    return NexusIntersectionState(
        intersection_id=str(object_id),
        intersection_type=_enum_name(_attr(obj, "intersection_type")),
        lane_group_ids=_str_tuple(_attr(obj, "lane_group_ids", [])),
        polygon=_polygon_array(_first_attr(obj, "shapely_polygon", "polygon")),
    )


def _extract_surface(object_id: Any, layer: str, obj: Any) -> NexusSurfaceState:
    semantic_attr = {
        "stop_zone": "stop_zone_type",
        "speed_bump": "speed_bump_type",
    }.get(layer)
    return NexusSurfaceState(
        object_id=str(object_id),
        layer=layer,
        polygon=_polygon_array(_first_attr(obj, "shapely_polygon", "polygon")),
        semantic_type=_enum_name(_attr(obj, semantic_attr)) if semantic_attr else None,
        lane_ids=_str_tuple(_attr(obj, "lane_ids", [])),
    )


def _extract_line(object_id: Any, layer: str, obj: Any) -> NexusLineState:
    semantic_attr = {
        "road_edge": "road_edge_type",
        "road_line": "road_line_type",
    }.get(layer)
    return NexusLineState(
        object_id=str(object_id),
        layer=layer,
        polyline=_polyline_array(_attr(obj, "polyline")),
        semantic_type=_enum_name(_attr(obj, semantic_attr)) if semantic_attr else None,
    )


def _build_lane_point_features(lanes: Iterable[NexusLaneState]) -> np.ndarray:
    rows: list[list[float]] = []
    for lane_index, lane in enumerate(lanes):
        centerline = lane.centerline
        if centerline.size == 0:
            continue
        speed_limit = lane.speed_limit_mps if lane.speed_limit_mps is not None else 0.0
        for idx, point in enumerate(centerline):
            if len(point) < 2:
                continue
            if idx + 1 < len(centerline):
                nxt = centerline[idx + 1]
            elif idx > 0:
                nxt = point
                point = centerline[idx - 1]
            else:
                nxt = point
            heading = float(np.arctan2(nxt[1] - point[1], nxt[0] - point[0]))
            rows.append([float(centerline[idx][0]), float(centerline[idx][1]), heading, float(speed_limit), float(lane_index)])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _available_layers(scenario: Py123DScenarioData) -> list[str]:
    if scenario.map.available_layers:
        return list(scenario.map.available_layers)
    map_api = scenario.map.raw_map_api
    layers = _call(map_api, "get_available_map_layers") or []
    return [_id_to_str(layer) for layer in layers]


def _iter_map_objects(scenario: Py123DScenarioData, layer: str) -> Iterable[tuple[Any, Any]]:
    objects = scenario.map.objects_by_layer.get(layer, {})
    if objects:
        for object_id, obj in objects.items():
            yield object_id, obj
        return

    map_api = scenario.map.raw_map_api
    object_ids = scenario.map.object_ids_by_layer.get(layer)
    if object_ids is None:
        object_ids = list(_call(map_api, "get_all_map_object_ids_in_layer", layer) or [])
    for object_id in object_ids:
        obj = _call(map_api, "get_map_object_in_layer", object_id, layer)
        if obj is not None:
            yield object_id, obj


def _iter_box_detections(container: Any) -> Iterable[Any]:
    return _attr(container, "box_detections", []) or []


def _iter_traffic_lights(container: Any) -> Iterable[Any]:
    return _attr(container, "detections", []) or []


def _canonical_custom_modalities(custom_modalities: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, payload in custom_modalities.items():
        canonical_key = str(key) if str(key).startswith("custom.") else f"custom.{key}"
        output[canonical_key] = dict(_custom_payload_data(payload))
    return output


def _custom_modalities_from_scene_api(scene: Any, iteration: int) -> dict[str, Any]:
    metadatas = _call(scene, "get_all_custom_modality_metadatas") or {}
    output: dict[str, Any] = {}
    for custom_id in metadatas.keys():
        custom_id_s = _id_to_str(custom_id)
        payload = _call(scene, "get_custom_modality_at_iteration", iteration, custom_id)
        if payload is None and custom_id_s != custom_id:
            payload = _call(scene, "get_custom_modality_at_iteration", iteration, custom_id_s)
        if payload is None:
            continue
        output[f"custom.{custom_id_s}"] = dict(_custom_payload_data(payload))
    return output


def _custom_payload_data(custom_payload: Any) -> Mapping[str, Any]:
    if custom_payload is None:
        return {}
    data = _attr(custom_payload, "data")
    if isinstance(data, Mapping):
        return data
    payload = _attr(custom_payload, "payload")
    if isinstance(payload, Mapping):
        return payload
    if isinstance(custom_payload, Mapping):
        return custom_payload
    return {}


def _route_state_from_custom(custom: Mapping[str, Any], source_key: str) -> NexusRouteState:
    roadblock_ids = _str_tuple(custom.get("route_roadblock_ids", ()))
    return NexusRouteState(
        roadblock_ids=roadblock_ids,
        source_key=source_key if roadblock_ids else None,
        support="supported" if roadblock_ids else "unsupported",
        metadata={"source_modality": "custom.scenario", "source_field": "route_roadblock_ids"}
        if roadblock_ids
        else {},
    )


def _sensor_references_from_custom(custom: Mapping[str, Any], timestamp_us: Optional[int]) -> dict[str, NexusSensorReference]:
    lidar_token = _optional_str(custom.get("lidar_token"))
    if lidar_token is None:
        return {}
    key = "custom.scenario.lidar_token"
    return {
        key: NexusSensorReference(
            key=key,
            modality_type="lidar_token",
            modality_id="lidar_token",
            timestamp_us=timestamp_us,
            metadata={"lidar_token": lidar_token, "source_key": key},
        )
    }


def _sensor_metadata_summary(scenario: Py123DScenarioData) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, record in scenario.modalities.items():
        if record.modality_type in {"camera", "lidar"}:
            output[key] = record.to_summary_dict()
    return output


def _polyline_array(polyline: Any) -> np.ndarray:
    if polyline is None:
        return np.zeros((0, 3), dtype=np.float64)
    array = _attr(polyline, "array")
    if array is None:
        linestring = _attr(polyline, "linestring")
        if linestring is not None:
            array = np.asarray(linestring.coords)
    return _as_xyz_array(array)


def _polygon_array(polygon: Any) -> np.ndarray:
    if polygon is None:
        return np.zeros((0, 3), dtype=np.float64)
    exterior = _attr(polygon, "exterior")
    if exterior is not None:
        return _as_xyz_array(np.asarray(exterior.coords))
    array = _attr(polygon, "array")
    return _as_xyz_array(array)


def _as_xyz_array(value: Any) -> np.ndarray:
    """(N, 3) map geometry, in float64.

    NOT float32: these coordinates are still absolute UTM at this point, where
    float32's ULP is 0.031 m at a nuPlan easting and 0.5 m at a northing. Casting
    here snapped every lane vertex to that lattice -- the jagged lane overlay --
    and nothing downstream could undo it, because recentring to a local frame
    moves a lattice rather than removing one. The source WKB in the arrow is
    clean float64; keep it that way and let the recentre pass hand consumers a
    local frame where a later float32 cast is sub-mm.
    """
    if value is None:
        return np.zeros((0, 3), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] == 2:
        zeros = np.zeros((array.shape[0], 1), dtype=np.float64)
        array = np.concatenate([array, zeros], axis=1)
    if array.shape[1] > 3:
        array = array[:, :3]
    return array


def _first_attr(obj: Any, *attrs: str) -> Any:
    for attr in attrs:
        value = _attr(obj, attr)
        if value is not None:
            return value
    return None


def _attr(obj: Any, attr: Optional[str], default: Any = None) -> Any:
    if obj is None or attr is None:
        return default
    try:
        return getattr(obj, attr)
    except Exception:
        if isinstance(obj, Mapping):
            return obj.get(attr, default)
        return default


def _call(obj: Any, method: str, *args: Any) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn(*args)
    except Exception:
        return None


def _float_attr(obj: Any, attr: str, default: float = 0.0, *, field: Optional[str] = None) -> float:
    """Float attribute with a default; ``field`` marks it REQUIRED.

    A required field that is missing or non-numeric still returns ``default``
    (downstream guards own the raise/skip decision) but is counted and warned
    via :func:`_record_missing_field`. Call sites without ``field`` are
    genuinely optional and keep the silent documented default.
    """
    result = _optional_float(_attr(obj, attr))
    if result is None:
        if field is not None:
            _record_missing_field(field)
        return default
    return result


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text and text != "None" else None


def _str_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return tuple(part for part in values.split() if part)
    try:
        return tuple(str(value) for value in values if str(value))
    except TypeError:
        return (str(values),)


def _enum_name(value: Any) -> str:
    if value is None:
        return "unknown"
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    serialized = getattr(value, "serialize", None)
    if callable(serialized):
        try:
            return str(serialized()).lower()
        except Exception:
            pass
    return str(value).split(".")[-1].lower()


def _id_to_str(value: Any) -> str:
    serialized = getattr(value, "serialize", None)
    if callable(serialized):
        try:
            return str(serialized())
        except Exception:
            pass
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    return str(value)


def _timestamp_us(timestamp: Any) -> Optional[int]:
    if timestamp is None:
        return None
    for attr in ("time_us", "timestamp_us", "microseconds"):
        value = _attr(timestamp, attr)
        if value is not None:
            return int(value)
    return None
