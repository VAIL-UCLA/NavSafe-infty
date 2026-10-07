"""py123d-native scenario containers for NavSafe training.

These dataclasses are intentionally separate from ``ScenarioDescription`` and
the text2sim ``ScenarioState`` IR.  Evaluation keeps using those legacy schemas;
BC and online RL can consume this py123d-aligned representation instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


def timestamp_to_us(timestamp: Any) -> Optional[int]:
    """Convert a py123d ``Timestamp``-like object to integer microseconds."""
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, np.integer)):
        return int(timestamp)
    for attr in ("time_us", "timestamp_us", "microseconds"):
        value = getattr(timestamp, attr, None)
        if value is not None:
            return int(value)
    to_us = getattr(timestamp, "to_us", None)
    if callable(to_us):
        return int(to_us())
    return None


def to_serializable(value: Any) -> Any:
    """Best-effort, lossy serialization for metadata summaries.

    Raw py123d objects are retained in the dataclasses.  This helper is only for
    JSON/debug summaries and test assertions, so unknown objects fall back to a
    small class/repr record instead of failing.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value) and not isinstance(value, type):
        return to_serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_serializable(v) for v in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_serializable(to_dict())
        except Exception:
            pass

    serialize = getattr(value, "serialize", None)
    if callable(serialize):
        try:
            return to_serializable(serialize())
        except Exception:
            pass

    time_us = timestamp_to_us(value)
    if time_us is not None:
        return time_us

    geometry = getattr(value, "wkt", None)
    if isinstance(geometry, str):
        return {"__class__": type(value).__name__, "wkt": geometry}

    return {"__class__": type(value).__name__, "repr": repr(value)}


@dataclass
class Py123DFrameModality:
    """One modality payload at one py123d scene iteration."""

    iteration: int
    timestamp_us: Optional[int]
    data: Any = None
    data_summary: dict[str, Any] = field(default_factory=dict)

    def has_data(self) -> bool:
        return self.data is not None


@dataclass
class Py123DModalityRecord:
    """All NavSafe knows about one py123d modality stream."""

    key: str
    modality_type: str
    modality_id: Optional[str] = None
    metadata: Any = None
    timestamps_us: list[int] = field(default_factory=list)
    frames: dict[int, Py123DFrameModality] = field(default_factory=dict)
    raw_reader_info: dict[str, Any] = field(default_factory=dict)

    @property
    def loaded_iterations(self) -> list[int]:
        return sorted(self.frames)

    def frame(self, iteration: int) -> Optional[Py123DFrameModality]:
        return self.frames.get(iteration)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "modality_type": self.modality_type,
            "modality_id": self.modality_id,
            "metadata": to_serializable(self.metadata),
            "num_timestamps": len(self.timestamps_us),
            "timestamps_us": list(self.timestamps_us),
            "loaded_iterations": self.loaded_iterations,
            "raw_reader_info": to_serializable(self.raw_reader_info),
        }


@dataclass
class Py123DMapRecord:
    """Map snapshot preserving py123d MapAPI layers and raw objects."""

    metadata: Any = None
    available_layers: list[str] = field(default_factory=list)
    object_ids_by_layer: dict[str, list[Any]] = field(default_factory=dict)
    objects_by_layer: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_map_api: Any = None

    def get_object(self, layer: str, object_id: Any) -> Any:
        return self.objects_by_layer.get(str(layer), {}).get(str(object_id))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "metadata": to_serializable(self.metadata),
            "available_layers": list(self.available_layers),
            "object_ids_by_layer": {
                layer: [to_serializable(obj_id) for obj_id in obj_ids]
                for layer, obj_ids in self.object_ids_by_layer.items()
            },
            "num_objects_by_layer": {
                layer: len(obj_ids) for layer, obj_ids in self.object_ids_by_layer.items()
            },
        }


@dataclass
class Py123DFrameState:
    """Per-frame runtime view derived from ``Py123DScenarioData``."""

    scenario_id: str
    iteration: int
    timestamp_us: Optional[int]
    ego_state: Any = None
    box_detections: Any = None
    traffic_light_detections: Any = None
    cameras: dict[str, Any] = field(default_factory=dict)
    lidars: dict[str, Any] = field(default_factory=dict)
    custom_modalities: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Py123DScenarioData:
    """Lossless-ish py123d scene container for BC and online RL.

    The normalized fields make common training queries cheap.  ``raw_scene_api``
    and the raw objects stored in records preserve py123d-native access for
    modalities that state-based RL does not consume yet, such as camera/lidar.
    """

    source: str = "py123d"
    dataset: Optional[str] = None
    split: Optional[str] = None
    location: Optional[str] = None
    log_name: Optional[str] = None
    scene_uuid: Optional[str] = None
    number_of_iterations: int = 0
    number_of_history_iterations: int = 0
    timestamps_us: list[int] = field(default_factory=list)
    scene_metadata: Any = None
    log_metadata: Any = None
    map_metadata: Any = None
    modalities: dict[str, Py123DModalityRecord] = field(default_factory=dict)
    map: Py123DMapRecord = field(default_factory=Py123DMapRecord)
    raw_scene_api: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        return self.scene_uuid or self.log_name or "unknown_py123d_scene"

    @property
    def modality_keys(self) -> list[str]:
        return sorted(self.modalities)

    def has_modality(self, key: str) -> bool:
        return key in self.modalities

    def get_modality(self, key: str) -> Optional[Py123DModalityRecord]:
        return self.modalities.get(key)

    def get_frame_state(self, iteration: int) -> Py123DFrameState:
        if iteration < 0 or (self.number_of_iterations and iteration >= self.number_of_iterations):
            raise IndexError(f"iteration {iteration} outside scene range [0, {self.number_of_iterations})")

        timestamp_us = self.timestamps_us[iteration] if iteration < len(self.timestamps_us) else None

        def data(key: str) -> Any:
            record = self.modalities.get(key)
            frame = record.frame(iteration) if record else None
            return frame.data if frame else None

        cameras = {
            record.modality_id or record.key: frame.data
            for record in self.modalities.values()
            if record.modality_type == "camera" and (frame := record.frame(iteration)) is not None
        }
        lidars = {
            record.modality_id or record.key: frame.data
            for record in self.modalities.values()
            if record.modality_type == "lidar" and (frame := record.frame(iteration)) is not None
        }
        custom = {
            record.modality_id or record.key: frame.data
            for record in self.modalities.values()
            if record.modality_type == "custom" and (frame := record.frame(iteration)) is not None
        }
        return Py123DFrameState(
            scenario_id=self.scenario_id,
            iteration=iteration,
            timestamp_us=timestamp_us,
            ego_state=data("ego_state_se3"),
            box_detections=data("box_detections_se3"),
            traffic_light_detections=data("traffic_light_detections"),
            cameras=cameras,
            lidars=lidars,
            custom_modalities=custom,
        )

    def validate(self) -> None:
        if self.number_of_iterations < 0:
            raise ValueError("number_of_iterations must be non-negative")
        if self.timestamps_us and len(self.timestamps_us) != self.number_of_iterations:
            raise ValueError("timestamps_us length must match number_of_iterations")
        for key, record in self.modalities.items():
            if key != record.key:
                raise ValueError(f"modality dict key {key!r} does not match record.key {record.key!r}")
            for iteration in record.frames:
                if iteration < 0 or (self.number_of_iterations and iteration >= self.number_of_iterations):
                    raise ValueError(f"modality {key!r} has frame outside scene range: {iteration}")

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset": self.dataset,
            "split": self.split,
            "location": self.location,
            "log_name": self.log_name,
            "scene_uuid": self.scene_uuid,
            "number_of_iterations": self.number_of_iterations,
            "number_of_history_iterations": self.number_of_history_iterations,
            "timestamps_us": list(self.timestamps_us),
            "scene_metadata": to_serializable(self.scene_metadata),
            "log_metadata": to_serializable(self.log_metadata),
            "map_metadata": to_serializable(self.map_metadata),
            "modalities": {key: record.to_summary_dict() for key, record in self.modalities.items()},
            "map": self.map.to_summary_dict(),
            "extras": to_serializable(self.extras),
        }
