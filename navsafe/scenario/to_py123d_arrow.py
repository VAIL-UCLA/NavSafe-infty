# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Serialize a NexusSim ``ScenarioDescription`` to a py123d Apache Arrow log.

This is the *write* inverse of the read pipeline in
:mod:`navsafe.scenario.py123d_scenario_description` /
:mod:`navsafe.scenario.py123d_training_extractor`: it materializes a
ScenarioNet-format ``ScenarioDescription`` dict as a py123d-native Arrow log
directory that :func:`py123d.api.get_filtered_scenes` rediscovers, so the same
runtime can round-trip PG / text2sim scenes through py123d's storage format.

py123d is imported lazily inside the writer so this module stays import-safe
where py123d is absent (the conda test suite stays collectable); a missing
install raises a clear error rather than degrading silently.

Conventions mirrored from the read side (invert these exactly):

* Heading is yaw about +Z. py123d stores orientation as an SE3 quaternion, so a
  scalar yaw becomes ``(qw, qz) = (cos(yaw/2), sin(yaw/2))`` with ``qx=qy=0``.
* The ego is stored as ``EgoStateSE3`` keyed off its *center* pose. We give the
  ego metadata identity center/rear-axle extrinsics so center == imu and the
  reader's ``center_se3`` reproduces the input x/y/heading without an offset.
* Non-ego actors are ``BoxDetectionSE3`` whose center pose carries x/y/heading
  and whose ``BoundingBoxSE3`` carries length/width/height. Both box and ego
  velocity are stored in the global frame — the reader reads ego
  ``dynamic_state.velocity`` directly as world, identically to box velocity.
* Lanes are written to a per-log ``map.arrow`` (``map_is_per_log=True``) so the
  log directory is self-contained and ``get_map_api_for_log`` finds it without a
  configured global maps root. The ScenarioNet polyline becomes the lane
  centerline; left/right boundaries are synthesized by lateral offset since the
  reader requires all three to be the same polyline type.

Known fidelity limitations (read-side, not writer bugs):

* Stop-sign agent types do not round-trip (the reader's ``_map_agent_type``
  has no distinct stop-sign inverse). CYCLIST round-trips via ``two_wheeler``.
* The ego must be valid at every frame; py123d treats ego as mandatory per
  iteration, so a scenario with an invalid ego mid-scene will not read back.
* Rediscovery layout: ``get_filtered_scenes(data_root)`` scans
  ``data_root/logs/<split>/<log_name>/sync.arrow``. The split name must contain
  ``train``/``val``/``test``, so we write under ``<dataset>_train``. ``log_dir``
  is therefore treated as a logical id: ``data_root = log_dir.parent`` and
  ``log_name = log_dir.name``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.type import MetaDriveType
from navsafe.utils.hashing import stable_hash

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from py123d.datatypes import EgoStateSE3, EgoStateSE3Metadata


_MICROS_PER_SECOND = 1_000_000
_DEFAULT_ITERATION_S = 0.1  # fallback timestep when metadata carries no timestamps

# Fallback box dimensions (length, width, height) in metres, used per-field when
# a track state omits a dimension array (e.g. PG tracks carry no "height").
# Default actor dimensions fill gaps without overriding source values.
_DEFAULT_DIMS = {
    MetaDriveType.VEHICLE: (4.7, 1.8, 1.5),
    MetaDriveType.PEDESTRIAN: (0.5, 0.5, 1.75),
    MetaDriveType.CYCLIST: (1.8, 0.7, 1.7),
}
_FALLBACK_DIMS = (4.0, 2.0, 1.5)


def _dims(state: dict, obj_type: str, t: int) -> tuple[float, float, float]:
    """(length, width, height) at frame ``t``, per-field-defaulted by type."""
    dl, dw, dh = _DEFAULT_DIMS.get(obj_type, _FALLBACK_DIMS)
    return (
        float(state["length"][t]) if "length" in state else dl,
        float(state["width"][t]) if "width" in state else dw,
        float(state["height"][t]) if "height" in state else dh,
    )


def _velocity(state: dict, t: int) -> tuple[float, float]:
    """Planar (vx, vy) at frame ``t``; (0, 0) when the track carries no velocity."""
    if "velocity" not in state:
        return 0.0, 0.0
    vx, vy = state["velocity"][t]
    return float(vx), float(vy)

# Canonical MetaDriveType -> py123d DefaultBoxDetectionLabel table. This is the
# structural inverse of ``_map_agent_type`` in py123d_scenario_description: that
# reader keys on ``label.name.lower()``, so each target enum is chosen such that
# its lowercased name routes back to the same MetaDriveType. VEHICLE, PEDESTRIAN,
# CYCLIST (via ``two_wheeler``), TRAFFIC_CONE and TRAFFIC_BARRIER round-trip
# exactly; stop-signs cannot (the reader has no distinct stop-sign inverse).
def _metadrive_type_to_py123d_label(md_type: str) -> Any:
    """Map a MetaDriveType agent type to a py123d ``DefaultBoxDetectionLabel``."""
    from py123d.datatypes import DefaultBoxDetectionLabel as L

    table = {
        MetaDriveType.VEHICLE: L.VEHICLE,
        MetaDriveType.PEDESTRIAN: L.PERSON,
        MetaDriveType.CYCLIST: L.TWO_WHEELER,
        MetaDriveType.TRAFFIC_CONE: L.TRAFFIC_CONE,
        MetaDriveType.TRAFFIC_BARRIER: L.BARRIER,
        MetaDriveType.TRAFFIC_STOP_SIGN: L.TRAFFIC_SIGN,
    }
    return table.get(md_type, L.GENERIC_OBJECT)


def _traffic_light_status(object_state: str) -> Any:
    """Map a ScenarioNet ``object_state`` string to a ``TrafficLightStatus``.

    Inverse of ``_TL_STATUS_MAP`` (GO/STOP/CAUTION) used by the reader.
    """
    from py123d.datatypes import TrafficLightStatus

    state = str(object_state or "").upper()
    if "GO" in state:
        return TrafficLightStatus.GREEN
    if "STOP" in state:
        return TrafficLightStatus.RED
    if "CAUTION" in state:
        return TrafficLightStatus.YELLOW
    return TrafficLightStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _pose_from_xy_heading(x: float, y: float, z: float, heading: float) -> Any:
    """Build a ``PoseSE3`` from a planar pose; heading is yaw about +Z."""
    from py123d.geometry import PoseSE3

    half = 0.5 * float(heading)
    return PoseSE3(x=float(x), y=float(y), z=float(z), qw=np.cos(half), qx=0.0, qy=0.0, qz=np.sin(half))


def _lateral_boundaries(centerline: np.ndarray, half_width: float = 1.75) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize left/right boundary polylines by offsetting the centerline.

    Boundaries are required by the py123d ``Lane`` constructor but are not part
    of the round-trip fidelity contract (only the centerline is compared). The
    offset is along the per-point normal of the centerline tangent.
    """
    xy = centerline[:, :2]
    tangents = np.gradient(xy, axis=0) if len(xy) > 1 else np.array([[1.0, 0.0]])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = tangents / norms
    normals = np.stack([-unit[:, 1], unit[:, 0]], axis=1)  # rotate tangent +90deg
    left = centerline.copy()
    right = centerline.copy()
    left[:, :2] = xy + half_width * normals
    right[:, :2] = xy - half_width * normals
    return left, right


# ---------------------------------------------------------------------------
# Modality construction (datatype building, no orchestration)
# ---------------------------------------------------------------------------

def _ego_metadata(length: float, width: float, height: float) -> "EgoStateSE3Metadata":
    """Ego metadata with identity extrinsics so center == imu == rear axle."""
    from py123d.datatypes import EgoStateSE3Metadata
    from py123d.geometry import PoseSE3

    return EgoStateSE3Metadata(
        vehicle_name="navsafe_ego",
        width=float(width),
        length=float(length),
        height=float(height),
        wheel_base=max(float(length) * 0.6, 0.1),
        center_to_imu_se3=PoseSE3.identity(),
        rear_axle_to_imu_se3=PoseSE3.identity(),
    )


def _build_ego_modality(state: dict, t: int, metadata: "EgoStateSE3Metadata", timestamp: Any) -> "EgoStateSE3":
    """Build the per-frame ``EgoStateSE3`` from a track state at index ``t``."""
    from py123d.datatypes import EgoStateSE3
    from py123d.datatypes.vehicle_state import DynamicStateSE3
    from py123d.geometry import Vector3D

    pos = state[SD.POSITION][t]
    heading = float(state[SD.HEADING][t])
    center = _pose_from_xy_heading(pos[0], pos[1], pos[2], heading)

    vx, vy = _velocity(state, t)
    # World-frame velocity: the reader reads ego ``dynamic_state.velocity.x/y``
    # directly as world (py123d_training_extractor), identically to box velocity,
    # so no body-frame rotation is applied here.
    dynamic_state = DynamicStateSE3(
        velocity=Vector3D(vx, vy, 0.0),
        acceleration=Vector3D(0.0, 0.0, 0.0),
        angular_velocity=Vector3D(0.0, 0.0, 0.0),
    )
    return EgoStateSE3.from_center(
        center_se3=center, metadata=metadata, timestamp=timestamp, dynamic_state_se3=dynamic_state
    )


def _build_box_detection(track_id: str, track: dict, t: int) -> Any:
    """Build a single ``BoxDetectionSE3`` for one actor track at index ``t``."""
    from py123d.datatypes import BoxDetectionAttributes, BoxDetectionSE3
    from py123d.geometry import BoundingBoxSE3, Vector3D

    state = track[SD.STATE]
    pos = state[SD.POSITION][t]
    heading = float(state[SD.HEADING][t])
    center = _pose_from_xy_heading(pos[0], pos[1], pos[2], heading)
    length, width, height = _dims(state, track[SD.TYPE], t)
    bbox = BoundingBoxSE3(center_se3=center, length=length, width=width, height=height)
    vx, vy = _velocity(state, t)
    return BoxDetectionSE3(
        attributes=BoxDetectionAttributes(
            label=_metadrive_type_to_py123d_label(track[SD.TYPE]),
            track_token=str(track_id),
        ),
        bounding_box_se3=bbox,
        velocity_3d=Vector3D(vx, vy, 0.0),  # box velocity is global frame
    )


def _build_box_detections(actor_tracks: dict, t: int, timestamp: Any, metadata: Any) -> Any:
    """Build the ``BoxDetectionsSE3`` container for all actors valid at ``t``."""
    from py123d.datatypes import BoxDetectionsSE3

    detections = [
        _build_box_detection(track_id, track, t)
        for track_id, track in actor_tracks.items()
        if bool(track[SD.STATE]["valid"][t])
    ]
    return BoxDetectionsSE3(box_detections=detections, timestamp=timestamp, metadata=metadata)


def _build_traffic_lights(dynamic_map_states: dict, t: int, timestamp: Any) -> Optional[Any]:
    """Build ``TrafficLightDetections`` for frame ``t``, or None when absent."""
    from py123d.datatypes import TrafficLightDetection, TrafficLightDetections

    detections = []
    for lane_id, entry in dynamic_map_states.items():
        states = entry.get(SD.STATE, {}).get("object_state")
        if not states or t >= len(states):
            continue
        # py123d keys traffic lights by an integer lane id; fall back to a stable
        # hash when the controlled-lane id is not numeric. ``stable_hash`` (not
        # builtin ``hash``) keeps this id stable across processes — builtin hash
        # is salted per process, which would renumber lanes every run.
        try:
            numeric_lane_id = int(lane_id)
        except (TypeError, ValueError):
            numeric_lane_id = stable_hash(str(lane_id)) % (2**31)
        detections.append(TrafficLightDetection(lane_id=numeric_lane_id, status=_traffic_light_status(states[t])))
    if not detections:
        return None
    return TrafficLightDetections(detections=detections, timestamp=timestamp)


def _build_lane(lane_id: str, feature: dict) -> Optional[Any]:
    """Build a py123d ``Lane`` from a ScenarioNet lane ``map_feature``."""
    from py123d.datatypes import Lane, LaneType
    from py123d.geometry import Polyline3D

    polyline = np.asarray(feature.get(SD.POLYLINE), dtype=np.float64)
    if polyline.ndim != 2 or len(polyline) < 2:
        return None
    if polyline.shape[1] == 2:  # lift to 3D; py123d lane polylines are Polyline3D
        polyline = np.column_stack([polyline, np.zeros(len(polyline))])

    left_arr = feature.get(SD.LEFT_BOUNDARIES)
    right_arr = feature.get(SD.RIGHT_BOUNDARIES)
    if left_arr is not None and right_arr is not None:
        left = _as_xyz(left_arr)
        right = _as_xyz(right_arr)
    else:
        left, right = _lateral_boundaries(polyline)

    # Carry a POSTED speed limit across the round trip. py123d's ``Lane`` has
    # the field and the converter back (``py123d_training_extractor._extract_
    # lane``) reads it, so dropping it here is silently lossy in the one
    # direction that cannot be detected downstream: on re-read,
    # ``speed_target.annotate_speed_targets`` sees a lane with no limit and
    # replaces the map's own number with an ESTIMATE labelled ``observed``.
    # An inferred target is deliberately NOT written back — it is a property
    # of the log this scenario was derived from, not of the map.
    from navsafe.scenario.speed_target import (
        SPEED_LIMIT_KEY, SPEED_LIMIT_SOURCE_KEY,
    )
    posted = feature.get(SPEED_LIMIT_KEY)
    if str(feature.get(SPEED_LIMIT_SOURCE_KEY, "dataset")) != "dataset":
        posted = None
    try:
        speed_limit = float(posted) if posted is not None else None
    except (TypeError, ValueError):
        speed_limit = None
    if speed_limit is not None and not (np.isfinite(speed_limit)
                                        and speed_limit > 0.0):
        speed_limit = None

    return Lane(
        object_id=str(lane_id),
        lane_type=LaneType.SURFACE_STREET,
        left_boundary=Polyline3D.from_array(left),
        right_boundary=Polyline3D.from_array(right),
        centerline=Polyline3D.from_array(polyline),
        speed_limit_mps=speed_limit,
    )


def _py123d_road_line_type(md_type: str) -> Any:
    """Best-effort MetaDriveType line → ``RoadLineType`` (color/style; else UNKNOWN).

    Geometry round-trips exactly; the exact line subtype is best-effort.
    """
    from py123d.datatypes import RoadLineType as R

    t = str(md_type).upper()
    broken = "BROKEN" in t or "DASH" in t
    double = "DOUBLE" in t
    if "YELLOW" in t:
        return (R.DOUBLE_DASH_YELLOW if broken else R.DOUBLE_SOLID_YELLOW) if double else (
            R.DASHED_YELLOW if broken else R.SOLID_YELLOW
        )
    if "WHITE" in t:
        return (R.DOUBLE_DASH_WHITE if broken else R.DOUBLE_SOLID_WHITE) if double else (
            R.DASHED_WHITE if broken else R.SOLID_WHITE
        )
    return R.UNKNOWN


def _build_road_line_or_edge(feature_id: str, feature: dict) -> Optional[Any]:
    """Build a py123d ``RoadLine`` (lane markings) or ``RoadEdge`` (boundaries).

    Non-lane map features are split by :meth:`MetaDriveType.is_road_line`;
    everything else non-lane (boundaries/edges/unknown) becomes a ``RoadEdge``.
    """
    from py123d.datatypes import RoadEdge, RoadEdgeType, RoadLine
    from py123d.geometry import Polyline3D

    md_type = feature.get(SD.TYPE)
    polyline = np.asarray(feature.get(SD.POLYLINE), dtype=np.float64)
    if polyline.ndim != 2 or len(polyline) < 2:
        return None
    if polyline.shape[1] == 2:
        polyline = np.column_stack([polyline, np.zeros(len(polyline))])
    geom = Polyline3D.from_array(polyline)

    if MetaDriveType.is_road_line(md_type):
        return RoadLine(object_id=str(feature_id), road_line_type=_py123d_road_line_type(cast(str, md_type)), polyline=geom)
    edge_type = RoadEdgeType.ROAD_EDGE_MEDIAN if "MEDIAN" in str(md_type).upper() else RoadEdgeType.ROAD_EDGE_BOUNDARY
    return RoadEdge(object_id=str(feature_id), road_edge_type=edge_type, polyline=geom)


def _as_xyz(array: Any) -> np.ndarray:
    """Coerce a polyline-like array to float64 (N, 3), zero-padding 2D input."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr))])
    return arr[:, :3]




# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _split_actors(sd: SD, sdc_id: str) -> tuple[dict, dict]:
    """Partition tracks into (ego_track, actor_tracks) by the sdc id."""
    tracks = sd[SD.TRACKS]
    ego_track = tracks.get(sdc_id)
    actor_tracks = {tid: tr for tid, tr in tracks.items() if tid != sdc_id}
    return ego_track, actor_tracks


# A per-frame stamp SPACING (or lone stamp value) at or below this is only
# plausible for a seconds-valued series; microsecond stamps at any real frame
# rate are spaced >= 10^4. Mirrors ``_DT_MAX_S`` in ``scenario_description``
# (see the ``SD.TIMESTEP`` convention table there).
_TS_MAX_SECONDS_SPACING_S = 10.0


def _stamps_are_seconds(stamps: np.ndarray) -> bool:
    """True when a per-frame stamp series is second-valued, False for microseconds.

    Classified by frame SPACING (median of positive diffs, robust to dropped
    frames); a series with no positive spacing (single frame, stationary clock)
    falls back to magnitude — any real microsecond stamp dwarfs a relative
    seconds one.
    """
    diffs = np.diff(stamps)
    positive = diffs[diffs > 0]
    if positive.size:
        return float(np.median(positive)) <= _TS_MAX_SECONDS_SPACING_S
    return float(np.max(np.abs(stamps), initial=0.0)) <= _TS_MAX_SECONDS_SPACING_S


def _timestamps_us(sd: SD, num_frames: int) -> list[int]:
    """Derive per-frame microsecond timestamps (monotonic increasing).

    ``metadata[TIMESTEP]`` ("ts") is overloaded by three producers (see the
    convention table above ``scenario_dt_seconds`` in ``scenario_description``):

    1. scalar dt in SECONDS (procgen / ``from_scenario_state``),
    2. per-frame ABSOLUTE MICROSECOND stamps (``py123d_scenario_description``),
    3. per-frame RELATIVE SECOND stamps (``[0, dt, 2dt, …]``).

    Family (3) used to be misread as (2), collapsing every stamp to
    ``int(round(i*dt))`` → 0,0,…,1,1 — sub-frame timestamps that broke
    round-tripped scenic logs. A per-frame array is now classified by
    :func:`_stamps_are_seconds` and second-valued series are scaled to
    microseconds. ``np.atleast_1d`` guards the 0-d scalar that ``len()``
    cannot handle; a lone second-plausible value is a scalar dt, not a stamp.
    """
    raw = sd.get(SD.METADATA, {}).get(SD.TIMESTEP)
    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)) if raw is not None else None
    is_stamp_series = (
        arr is not None
        and num_frames > 0
        and arr.size >= num_frames
        and not (arr.size == 1 and abs(float(arr[0])) <= _TS_MAX_SECONDS_SPACING_S)
    )
    if is_stamp_series:
        assert arr is not None  # narrowed by is_stamp_series
        stamps = arr[:num_frames]
        scale = float(_MICROS_PER_SECOND) if _stamps_are_seconds(stamps) else 1.0
        return [int(round(float(v) * scale)) for v in stamps]
    step_s = float(arr[0]) if arr is not None and arr.size == 1 else _DEFAULT_ITERATION_S
    step = int(round(step_s * _MICROS_PER_SECOND))
    return [i * step for i in range(num_frames)]


def _write_map(sd: SD, *, dataset: str, split: str, log_name: str, logs_root: Path, maps_root: Path) -> None:
    """Write all map_features (lanes + road lines/edges) to a per-log ``map.arrow``.

    py123d numbers object ids per-layer from 0; the reader namespaces non-lane
    features by layer (``road_edge_*`` / ``road_line_*``), so lanes, edges, and
    lines no longer collide on read and all round-trip.
    """
    from py123d.api.map import ArrowMapWriter
    from py123d.datatypes import MapMetadata

    objects = []
    for feature_id, feature in sd[SD.MAP_FEATURES].items():
        if MetaDriveType.is_lane(feature.get(SD.TYPE)):
            obj = _build_lane(feature_id, feature)
        else:
            obj = _build_road_line_or_edge(feature_id, feature)
        if obj is not None:
            objects.append(obj)
    if not objects:
        return

    map_metadata = MapMetadata(
        dataset=dataset, location=dataset, map_has_z=True, map_is_per_log=True, split=split, log_name=log_name
    )
    writer = ArrowMapWriter(force_map_conversion=True, maps_root=maps_root, logs_root=logs_root)
    if writer.reset(map_metadata):
        for obj in objects:
            writer.write_map_object(obj)
    writer.close()


def _write_log(
    sd: SD,
    ego_track: dict,
    actor_tracks: dict,
    *,
    dataset: str,
    split: str,
    log_name: str,
    location: str,
    logs_root: Path,
    sensors_root: Path,
    num_frames: int,
) -> None:
    """Write ego + box + traffic-light modalities as one synchronized Arrow log."""
    from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter, LogWriterConfig
    from py123d.datatypes import (
        BoxDetectionsSE3Metadata,
        DefaultBoxDetectionLabel,
        LogMetadata,
        MapMetadata,
        Timestamp,
        TrafficLightDetectionsMetadata,
    )
    from py123d.parser.base_dataset_parser import ModalitiesSync

    ego_state = ego_track[SD.STATE]
    ego_metadata = _ego_metadata(*_dims(ego_state, ego_track[SD.TYPE], 0))
    box_metadata = BoxDetectionsSE3Metadata(box_detection_label_class=DefaultBoxDetectionLabel)
    tl_metadata = TrafficLightDetectionsMetadata()
    dynamic_map_states = sd.get(SD.DYNAMIC_MAP_STATES, {})

    map_metadata = MapMetadata(
        dataset=dataset, location=location, map_has_z=True, map_is_per_log=True, split=split, log_name=log_name
    )
    log_metadata = LogMetadata(
        dataset=dataset, split=split, log_name=log_name, location=location, map_metadata=map_metadata
    )

    # Computed before the writer is opened so a timestamp error surfaces directly
    # rather than being masked by writer.close()'s deferred-sync path.
    timestamps_us = _timestamps_us(sd, num_frames)

    writer = ArrowLogWriter(
        log_writer_config=LogWriterConfig(), logs_root=logs_root, sensors_root=sensors_root
    )
    if not writer.reset(log_metadata):
        return
    try:
        for t in range(num_frames):
            timestamp = Timestamp.from_us(timestamps_us[t])
            modalities: list = []
            if bool(ego_state["valid"][t]):
                modalities.append(_build_ego_modality(ego_state, t, ego_metadata, timestamp))
            modalities.append(_build_box_detections(actor_tracks, t, timestamp, box_metadata))
            traffic_lights = _build_traffic_lights(dynamic_map_states, t, timestamp)
            if traffic_lights is not None:
                modalities.append(traffic_lights)
            writer.write_sync(ModalitiesSync(timestamp=timestamp, modalities=modalities))
    finally:
        writer.close()


def write_scenario_description_arrow(sd: SD, log_dir: str | Path) -> Path:
    """Write a ``ScenarioDescription`` as a py123d Arrow log under ``log_dir``.

    The log is rediscoverable via ``get_filtered_scenes(data_root=log_dir.parent)``.
    ``log_dir`` is treated as a logical identity: its parent is the py123d data
    root and its name is the log name (the data physically lands under
    ``<parent>/logs/<dataset>_train/<log_name>/``).

    :param sd: The ScenarioNet-format scenario to serialize.
    :param log_dir: Logical log directory; ``log_dir.parent`` is the data root.
    :returns: ``log_dir`` (the unique logical identity of the written log); its
        parent is the data root to pass to ``get_filtered_scenes``.
    :raises ImportError: If py123d is not installed.
    :raises ValueError: If the scenario has no ego (sdc) track.
    """
    try:
        import py123d  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without py123d
        raise ImportError(
            "py123d is required to write Arrow logs. Install it with `uv sync` "
            "(it is a core dependency)."
        ) from exc

    log_dir = Path(log_dir)
    data_root = log_dir.parent
    log_name = log_dir.name

    sdc_id = sd[SD.METADATA][SD.SDC_ID]
    ego_track, actor_tracks = _split_actors(sd, sdc_id)
    if ego_track is None:
        raise ValueError(f"ScenarioDescription has no ego track under sdc_id={sdc_id!r}")

    num_frames = int(sd[SD.LENGTH])
    dataset = str(sd.get(SD.METADATA, {}).get("dataset") or "navsafe")
    # The split name must embed train/val/test for py123d log discovery to find it.
    split = f"{dataset}_train"
    location = dataset

    logs_root = data_root / "logs"
    maps_root = data_root / "maps"
    sensors_root = data_root / "sensors"

    _write_map(sd, dataset=dataset, split=split, log_name=log_name, logs_root=logs_root, maps_root=maps_root)
    _write_log(
        sd,
        ego_track,
        actor_tracks,
        dataset=dataset,
        split=split,
        log_name=log_name,
        location=location,
        logs_root=logs_root,
        sensors_root=sensors_root,
        num_frames=num_frames,
    )
    return log_dir


__all__ = ["write_scenario_description_arrow"]
