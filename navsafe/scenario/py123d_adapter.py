"""Adapter from py123d SceneAPI objects into NexusSim py123d schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from navsafe.scenario.py123d_schema import (
    Py123DFrameModality,
    Py123DMapRecord,
    Py123DModalityRecord,
    Py123DScenarioData,
    timestamp_to_us,
    to_serializable,
)


STANDARD_STATE_MODALITIES = {
    "ego_state_se3",
    "box_detections_se3",
    "traffic_light_detections",
}


@dataclass(frozen=True)
class Py123DAdapterConfig:
    """Controls how much payload is loaded eagerly from a py123d scene."""

    include_history: bool = False
    load_state_payloads: bool = True
    load_custom_payloads: bool = True
    load_sensor_payloads: bool = False
    load_map_objects: bool = True
    data_root: Optional[str] = None
    require_map: bool = True
    #: Half-open ``[start, stop)`` iteration window to materialise. ``None``
    #: loads the whole scene, which is what every caller did before this
    #: existed -- and what made memory scale with LOG LENGTH rather than with
    #: episode length: this conversion emits one scene per nuPlan log
    #: (650-5320 iterations), so a 156-frame episode was paying to hold
    #: thousands of frames of agent tracks it never simulated. Measured peak
    #: was 34-50 GB RSS per worker, scaling with the log drawn rather than
    #: with the work done.
    #:
    #: Frames stay keyed by ABSOLUTE iteration (``record.frames`` is a dict,
    #: and py123d_schema validates each key against the scene range), so a
    #: windowed scene is a sparse scene, not a renumbered one -- no consumer
    #: has to learn about the offset.
    frame_window: Optional[tuple[int, int]] = None


def scenario_from_py123d_scene(scene: Any, config: Optional[Py123DAdapterConfig] = None) -> Py123DScenarioData:
    """Convert a py123d ``SceneAPI`` object into ``Py123DScenarioData``.

    The adapter is intentionally duck-typed: tests can use a fake SceneAPI and
    NexusSim can import this module even when py123d is not installed.  Actual
    py123d objects are preserved as raw payloads in the returned dataclasses.
    """
    cfg = config or Py123DAdapterConfig()

    scene_metadata = _call(scene, "get_scene_metadata")
    log_metadata = _call(scene, "get_log_metadata")
    map_metadata = _call(scene, "get_map_metadata")

    timestamps = [ts for ts in (_call(scene, "get_all_iteration_timestamps", cfg.include_history) or [])]
    timestamps_us = [time_us for ts in timestamps if (time_us := timestamp_to_us(ts)) is not None]
    number_of_iterations = int(_attr(scene, "number_of_iterations", len(timestamps_us)))
    history_iterations = int(_attr(scene, "number_of_history_iterations", 0))

    scenario = Py123DScenarioData(
        dataset=_attr(scene, "dataset", _attr(log_metadata, "dataset", None)),
        split=_attr(scene, "split", _attr(log_metadata, "split", None)),
        location=_attr(scene, "location", _attr(log_metadata, "location", None)),
        log_name=_attr(scene, "log_name", _attr(log_metadata, "log_name", None)),
        scene_uuid=_attr(scene, "scene_uuid", _attr(scene_metadata, "initial_uuid", None)),
        number_of_iterations=number_of_iterations,
        number_of_history_iterations=history_iterations,
        timestamps_us=timestamps_us,
        scene_metadata=scene_metadata,
        log_metadata=log_metadata,
        map_metadata=map_metadata,
        raw_scene_api=scene,
    )

    scenario.modalities.update(_discover_modalities(scene, cfg))
    _load_requested_frames(scene, scenario, cfg)
    map_api = _resolve_map_api(scene, scenario, cfg)
    scenario.map = _snapshot_map(map_api, map_metadata, cfg.load_map_objects)
    scenario.validate()
    return scenario


def _resolve_map_api(scene: Any, scenario: Py123DScenarioData, cfg: Py123DAdapterConfig) -> Any:
    map_api = _call(scene, "get_map_api")
    if _map_api_has_layers(map_api):
        return map_api

    explicit_map_api = _load_explicit_map_api(cfg.data_root, scenario.dataset, scenario.location)
    if _map_api_has_layers(explicit_map_api):
        return explicit_map_api

    if cfg.require_map:
        expected_paths = _candidate_map_paths(cfg.data_root, scenario.dataset, scenario.location)
        expected = ", ".join(str(path) for path in expected_paths) or "<no data_root/dataset/location available>"
        raise RuntimeError(
            "py123d map API is unavailable or has no layers "
            f"for dataset={scenario.dataset!r}, location={scenario.location!r}, log_name={scenario.log_name!r}. "
            f"Expected map path candidates: {expected}. "
            "Pass Py123DAdapterConfig(data_root=...) or configure py123d map root before training."
        )

    return map_api


def _map_api_has_layers(map_api: Any) -> bool:
    if map_api is None:
        return False
    layers = _call(map_api, "get_available_map_layers") or []
    return bool(layers)


def _load_explicit_map_api(data_root: Optional[str], dataset: Optional[str], location: Optional[str]) -> Any:
    for map_path in _candidate_map_paths(data_root, dataset, location):
        if not map_path.exists():
            continue
        try:
            from py123d.api.map.arrow.arrow_map_api import ArrowMapAPI
        except ImportError as exc:
            raise RuntimeError("py123d is required to load explicit Arrow map files") from exc
        return ArrowMapAPI(map_path)
    return None


def _candidate_map_paths(data_root: Optional[str], dataset: Optional[str], location: Optional[str]) -> list[Path]:
    if not data_root or not dataset or not location:
        return []
    root = Path(data_root)
    file_name = f"{dataset}_{location}.arrow"
    if root.suffix == ".arrow":
        return [root]
    return [
        root / "maps" / dataset / file_name,
        root / dataset / file_name,
        root / file_name,
    ]


def _discover_modalities(scene: Any, cfg: Py123DAdapterConfig) -> dict[str, Py123DModalityRecord]:
    records: dict[str, Py123DModalityRecord] = {}

    for key, metadata in (_call(scene, "get_all_modality_metadatas") or {}).items():
        modality_type, modality_id = _metadata_modality_identity(key, metadata)
        canonical_key = _canonical_key(modality_type, modality_id, str(key))
        records[canonical_key] = Py123DModalityRecord(
            key=canonical_key,
            modality_type=modality_type,
            modality_id=modality_id,
            metadata=metadata,
            timestamps_us=_generic_timestamps(scene, modality_type, modality_id, cfg.include_history),
            raw_reader_info={"source_key": str(key)},
        )

    _ensure_record(
        records,
        key="ego_state_se3",
        modality_type="ego_state_se3",
        modality_id=None,
        metadata=_call(scene, "get_ego_state_se3_metadata"),
        timestamps=_call(scene, "get_all_ego_state_se3_timestamps", cfg.include_history),
    )
    _ensure_record(
        records,
        key="box_detections_se3",
        modality_type="box_detections_se3",
        modality_id=None,
        metadata=_call(scene, "get_box_detections_se3_metadata"),
        timestamps=_call(scene, "get_all_box_detections_se3_timestamps", cfg.include_history),
    )
    _ensure_record(
        records,
        key="traffic_light_detections",
        modality_type="traffic_light_detections",
        modality_id=None,
        metadata=_call(scene, "get_traffic_light_detections_metadata"),
        timestamps=_call(scene, "get_all_traffic_light_detections_timestamps", cfg.include_history),
    )

    for camera_id, metadata in (_call(scene, "get_camera_metadatas") or {}).items():
        camera_id_s = _id_to_str(camera_id)
        _ensure_record(
            records,
            key=f"camera:{camera_id_s}",
            modality_type="camera",
            modality_id=camera_id_s,
            metadata=metadata,
            timestamps=_call(scene, "get_all_camera_timestamps", camera_id, cfg.include_history),
        )

    for lidar_id, metadata in (_call(scene, "get_lidar_metadatas") or {}).items():
        lidar_id_s = _id_to_str(lidar_id)
        _ensure_record(
            records,
            key=f"lidar:{lidar_id_s}",
            modality_type="lidar",
            modality_id=lidar_id_s,
            metadata=metadata,
            timestamps=_call(scene, "get_all_lidar_timestamps", lidar_id, cfg.include_history),
        )

    for custom_id, metadata in (_call(scene, "get_all_custom_modality_metadatas") or {}).items():
        custom_id_s = _id_to_str(custom_id)
        _ensure_record(
            records,
            key=f"custom:{custom_id_s}",
            modality_type="custom",
            modality_id=custom_id_s,
            metadata=metadata,
            timestamps=_call(scene, "get_all_custom_modality_timestamps", custom_id, cfg.include_history),
        )

    return records


def _load_requested_frames(scene: Any, scenario: Py123DScenarioData, cfg: Py123DAdapterConfig) -> None:
    start = -scenario.number_of_history_iterations if cfg.include_history else 0
    end = scenario.number_of_iterations
    if cfg.frame_window is not None:
        # Clamp rather than trust: a mined window comes from nuPlan's clock and
        # a scene's iteration count comes from the conversion, so an
        # off-by-a-frame at either end must narrow the load, never index past
        # the scene (which would surface as a confusing payload-is-None loop).
        window_start, window_stop = cfg.frame_window
        start = max(start, int(window_start))
        end = min(end, int(window_stop))
        if start >= end:
            raise ValueError(
                f"frame_window {cfg.frame_window} selects no iterations of a "
                f"scene with {scenario.number_of_iterations} (history "
                f"{scenario.number_of_history_iterations})")
        # Payloads remain keyed by absolute source iteration. Dense runtime
        # projection must know which interval exists instead of requesting
        # unloaded frame zero (or the unloaded tail of a prefix window).
        scenario.extras["frame_window"] = [start, end]
    for iteration in range(start, end):
        for record in scenario.modalities.values():
            should_load = (
                (cfg.load_state_payloads and record.modality_type in STANDARD_STATE_MODALITIES)
                or (cfg.load_custom_payloads and record.modality_type == "custom")
                or (cfg.load_sensor_payloads and record.modality_type in {"camera", "lidar"})
            )
            if not should_load:
                continue
            payload = _read_payload(scene, record, iteration)
            if payload is None:
                continue
            timestamp_us = timestamp_to_us(getattr(payload, "timestamp", None))
            if timestamp_us is None and 0 <= iteration < len(scenario.timestamps_us):
                timestamp_us = scenario.timestamps_us[iteration]
            record.frames[iteration] = Py123DFrameModality(
                iteration=iteration,
                timestamp_us=timestamp_us,
                data=payload,
                data_summary=_modality_summary(payload),
            )


def _read_payload(scene: Any, record: Py123DModalityRecord, iteration: int) -> Any:
    if record.modality_type == "ego_state_se3":
        return _call(scene, "get_ego_state_se3_at_iteration", iteration)
    if record.modality_type == "box_detections_se3":
        return _call(scene, "get_box_detections_se3_at_iteration", iteration)
    if record.modality_type == "traffic_light_detections":
        return _call(scene, "get_traffic_light_detections_at_iteration", iteration)
    if record.modality_type == "camera":
        return _call(scene, "get_camera_at_iteration", iteration, record.modality_id)
    if record.modality_type == "lidar":
        return _call(scene, "get_lidar_at_iteration", iteration, record.modality_id)
    if record.modality_type == "custom":
        return _call(scene, "get_custom_modality_at_iteration", iteration, record.modality_id)
    return None


def _snapshot_map(map_api: Any, map_metadata: Any, load_objects: bool) -> Py123DMapRecord:
    if map_api is None:
        return Py123DMapRecord(metadata=map_metadata)

    layers = [_id_to_str(layer) for layer in (_call(map_api, "get_available_map_layers") or [])]
    object_ids_by_layer: dict[str, list[Any]] = {}
    objects_by_layer: dict[str, dict[str, Any]] = {}
    for layer in layers:
        object_ids = list(_call(map_api, "get_all_map_object_ids_in_layer", layer) or [])
        object_ids_by_layer[layer] = object_ids
        if not load_objects:
            continue
        objects_by_layer[layer] = {}
        for object_id in object_ids:
            obj = _call(map_api, "get_map_object_in_layer", object_id, layer)
            if obj is not None:
                objects_by_layer[layer][str(object_id)] = obj

    return Py123DMapRecord(
        metadata=_call(map_api, "get_map_metadata") or map_metadata,
        available_layers=layers,
        object_ids_by_layer=object_ids_by_layer,
        objects_by_layer=objects_by_layer,
        raw_map_api=map_api,
    )


def _ensure_record(
    records: dict[str, Py123DModalityRecord],
    *,
    key: str,
    modality_type: str,
    modality_id: Optional[str],
    metadata: Any,
    timestamps: Optional[Iterable[Any]],
) -> None:
    timestamps_us = [time_us for ts in (timestamps or []) if (time_us := timestamp_to_us(ts)) is not None]
    if key in records:
        record = records[key]
        record.modality_type = modality_type
        record.modality_id = modality_id
        record.metadata = metadata if metadata is not None else record.metadata
        record.timestamps_us = timestamps_us or record.timestamps_us
        return
    if metadata is None and not timestamps_us:
        return
    records[key] = Py123DModalityRecord(
        key=key,
        modality_type=modality_type,
        modality_id=modality_id,
        metadata=metadata,
        timestamps_us=timestamps_us,
    )


def _metadata_modality_identity(key: str, metadata: Any) -> tuple[str, Optional[str]]:
    source_key = str(key)
    source_type = source_key
    source_id = None
    for sep in (":", "."):
        if sep in source_key:
            source_type, source_id = source_key.split(sep, 1)
            break
    metadata_type = _attr(metadata, "modality_type", None)
    metadata_id = _attr(metadata, "modality_id", None)
    modality_type = _id_to_str(metadata_type if metadata_type is not None else source_type)
    modality_id = metadata_id if metadata_id is not None else source_id
    return modality_type, _id_to_str(modality_id) if modality_id is not None else None


def _canonical_key(modality_type: str, modality_id: Optional[str], source_key: str) -> str:
    if modality_type in {"camera", "lidar", "custom"} and modality_id is not None:
        return f"{modality_type}:{modality_id}"
    if modality_type in STANDARD_STATE_MODALITIES:
        return modality_type
    return source_key


def _generic_timestamps(scene: Any, modality_type: str, modality_id: Optional[str], include_history: bool) -> list[int]:
    timestamps = _call(scene, "get_all_modality_timestamps", modality_type, modality_id, include_history) or []
    return [time_us for ts in timestamps if (time_us := timestamp_to_us(ts)) is not None]


def _modality_summary(payload: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"class": type(payload).__name__}
    timestamp_us = timestamp_to_us(getattr(payload, "timestamp", None))
    if timestamp_us is not None:
        summary["timestamp_us"] = timestamp_us
    for attr in ("box_detections", "detections"):
        values = getattr(payload, attr, None)
        if values is not None:
            try:
                summary[f"num_{attr}"] = len(values)
            except TypeError:
                pass
    if hasattr(payload, "image"):
        image = getattr(payload, "image")
        summary["image"] = to_serializable(getattr(image, "shape", None))
    if hasattr(payload, "points"):
        points = getattr(payload, "points")
        summary["points"] = to_serializable(getattr(points, "shape", None))
    return summary


def _id_to_str(value: Any) -> str:
    serialize = getattr(value, "serialize", None)
    if callable(serialize):
        try:
            return str(serialize())
        except Exception:
            pass
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    return str(value)


def _attr(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def _call(obj: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return None
