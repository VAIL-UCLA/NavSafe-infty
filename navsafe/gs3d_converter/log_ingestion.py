# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Log ingestion: py123d sensor logs → IngestedScene.

Supports any dataset exposed by py123d (NCore, WOD, AV2, …) via the
common ModalitiesSync interface.  Dataset-specific readers subclass the
shared _Py123dLogReader base and supply a label-info mapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, cast

import numpy as np

from py123d.datatypes import BoxDetectionsSE3, EgoStateSE3, TrafficLightDetections
from py123d.datatypes.sensors.base_camera import BaseCameraMetadata
from py123d.datatypes.sensors.lidar import Lidar, LidarMergedMetadata
from py123d.parser.base_dataset_parser import ModalitiesSync, ParsedCamera, ParsedLidar
from py123d.parser.registry import (
    AV2SensorBoxDetectionLabel,
    PhysicalAIAVBoxDetectionLabel,
    WODPerceptionBoxDetectionLabel,
)

try:
    from navsafe.scenario.type import MetaDriveType
except ImportError:
    class MetaDriveType:  # type: ignore[no-redef]
        VEHICLE = "VEHICLE"
        PEDESTRIAN = "PEDESTRIAN"
        CYCLIST = "CYCLIST"

# ── NCore label taxonomy ──────────────────────────────────────────────────────

_L = PhysicalAIAVBoxDetectionLabel

LABEL_TO_METADRIVE: dict = {
    _L.AUTOMOBILE:        MetaDriveType.VEHICLE,
    _L.HEAVY_TRUCK:       MetaDriveType.VEHICLE,
    _L.BUS:               MetaDriveType.VEHICLE,
    _L.TROLLEY_BUS:       MetaDriveType.VEHICLE,
    _L.TRAIN_OR_TRAM_CAR: MetaDriveType.VEHICLE,
    _L.OTHER_VEHICLE:     MetaDriveType.VEHICLE,
    _L.TRAILER:           MetaDriveType.VEHICLE,
    _L.PERSON:            MetaDriveType.PEDESTRIAN,
    _L.STROLLER:          MetaDriveType.PEDESTRIAN,
    _L.RIDER:             MetaDriveType.CYCLIST,
    _L.ANIMAL:            MetaDriveType.VEHICLE,
    _L.PROTRUDING_OBJECT: MetaDriveType.VEHICLE,
}

LABEL_TO_DYNAMIC_KIND: dict = {
    _L.AUTOMOBILE:        "vehicle",
    _L.HEAVY_TRUCK:       "vehicle",
    _L.BUS:               "vehicle",
    _L.TROLLEY_BUS:       "vehicle",
    _L.TRAIN_OR_TRAM_CAR: "vehicle",
    _L.OTHER_VEHICLE:     "vehicle",
    _L.TRAILER:           "vehicle",
    _L.PERSON:            "pedestrian",
    _L.STROLLER:          "pedestrian",
    _L.RIDER:             "cyclist",
    _L.ANIMAL:            "vehicle",
    _L.PROTRUDING_OBJECT: "vehicle",
}

LABEL_DEFAULT_SIZES: dict = {
    _L.AUTOMOBILE:        (4.5, 1.8, 1.5),
    _L.HEAVY_TRUCK:       (8.0, 2.5, 3.5),
    _L.BUS:               (12.0, 2.5, 3.5),
    _L.PERSON:            (0.5, 0.5, 1.75),
    _L.STROLLER:          (0.8, 0.6, 1.0),
    _L.RIDER:             (1.8, 0.7, 1.7),
}

_NCORE_LABEL_INFO: dict = {
    lbl: (LABEL_TO_METADRIVE.get(lbl, MetaDriveType.VEHICLE),
          LABEL_TO_DYNAMIC_KIND.get(lbl, "vehicle"),
          LABEL_DEFAULT_SIZES.get(lbl, (4.5, 1.8, 1.5)))
    for lbl in PhysicalAIAVBoxDetectionLabel
    if lbl in LABEL_TO_METADRIVE
}

# ── WOD label taxonomy ────────────────────────────────────────────────────────

_W = WODPerceptionBoxDetectionLabel

_WOD_LABEL_INFO: dict = {
    _W.TYPE_VEHICLE:    (MetaDriveType.VEHICLE,    "vehicle",    (4.5, 1.8, 1.5)),
    _W.TYPE_PEDESTRIAN: (MetaDriveType.PEDESTRIAN, "pedestrian", (0.5, 0.5, 1.75)),
    _W.TYPE_CYCLIST:    (MetaDriveType.CYCLIST,    "cyclist",    (1.8, 0.7, 1.7)),
    _W.TYPE_SIGN:       (MetaDriveType.VEHICLE,    None,         (0.5, 0.5, 1.5)),
    _W.TYPE_UNKNOWN:    (MetaDriveType.VEHICLE,    "vehicle",    (4.5, 1.8, 1.5)),
}

_DEFAULT_LABEL_INFO: Tuple = (MetaDriveType.VEHICLE, "vehicle", (4.5, 1.8, 1.5))

# ── AV2 sensor label taxonomy ─────────────────────────────────────────────────

_A = AV2SensorBoxDetectionLabel

_AV2_LABEL_INFO: dict = {
    _A.REGULAR_VEHICLE:   (MetaDriveType.VEHICLE,    "vehicle",    (4.5, 1.8, 1.5)),
    _A.LARGE_VEHICLE:     (MetaDriveType.VEHICLE,    "vehicle",    (8.0, 2.5, 3.0)),
    _A.BOX_TRUCK:         (MetaDriveType.VEHICLE,    "vehicle",    (8.0, 2.5, 3.5)),
    _A.TRUCK:             (MetaDriveType.VEHICLE,    "vehicle",    (8.0, 2.5, 3.5)),
    _A.TRUCK_CAB:         (MetaDriveType.VEHICLE,    "vehicle",    (6.0, 2.5, 3.0)),
    _A.BUS:               (MetaDriveType.VEHICLE,    "vehicle",    (12.0, 2.5, 3.5)),
    _A.ARTICULATED_BUS:   (MetaDriveType.VEHICLE,    "vehicle",    (18.0, 2.5, 3.5)),
    _A.SCHOOL_BUS:        (MetaDriveType.VEHICLE,    "vehicle",    (12.0, 2.5, 3.5)),
    _A.RAILED_VEHICLE:    (MetaDriveType.VEHICLE,    "vehicle",    (18.0, 2.7, 3.8)),
    _A.VEHICULAR_TRAILER: (MetaDriveType.VEHICLE,    "vehicle",    (6.0, 2.3, 2.5)),
    _A.MOTORCYCLE:        (MetaDriveType.CYCLIST,    "cyclist",    (2.2, 0.9, 1.4)),
    _A.MOTORCYCLIST:      (MetaDriveType.CYCLIST,    "cyclist",    (2.2, 0.9, 1.7)),
    _A.BICYCLE:           (MetaDriveType.CYCLIST,    "cyclist",    (1.8, 0.7, 1.2)),
    _A.BICYCLIST:         (MetaDriveType.CYCLIST,    "cyclist",    (1.8, 0.7, 1.7)),
    _A.WHEELED_RIDER:     (MetaDriveType.CYCLIST,    "cyclist",    (1.2, 0.6, 1.7)),
    _A.PEDESTRIAN:        (MetaDriveType.PEDESTRIAN, "pedestrian", (0.5, 0.5, 1.75)),
    _A.OFFICIAL_SIGNALER: (MetaDriveType.PEDESTRIAN, "pedestrian", (0.5, 0.5, 1.75)),
    _A.STROLLER:          (MetaDriveType.PEDESTRIAN, "pedestrian", (0.8, 0.6, 1.0)),
    _A.WHEELCHAIR:        (MetaDriveType.PEDESTRIAN, "pedestrian", (1.0, 0.7, 1.3)),
    _A.WHEELED_DEVICE:    (MetaDriveType.PEDESTRIAN, "pedestrian", (1.0, 0.5, 1.2)),
    _A.DOG:               (MetaDriveType.PEDESTRIAN, "pedestrian", (0.8, 0.4, 0.6)),
    _A.ANIMAL:            (MetaDriveType.PEDESTRIAN, "pedestrian", (0.8, 0.4, 0.6)),
    # Static street furniture: keep as non-dynamic vehicles so NRE sees the cuboid
    # but NavSafe does not try to drive it.
    _A.BOLLARD:                         (MetaDriveType.VEHICLE, None, (0.3, 0.3, 1.0)),
    _A.CONSTRUCTION_CONE:               (MetaDriveType.VEHICLE, None, (0.3, 0.3, 0.7)),
    _A.CONSTRUCTION_BARREL:             (MetaDriveType.VEHICLE, None, (0.6, 0.6, 1.0)),
    _A.SIGN:                            (MetaDriveType.VEHICLE, None, (0.5, 0.5, 1.5)),
    _A.STOP_SIGN:                       (MetaDriveType.VEHICLE, None, (0.5, 0.5, 1.5)),
    _A.MOBILE_PEDESTRIAN_CROSSING_SIGN: (MetaDriveType.VEHICLE, None, (0.5, 0.5, 1.5)),
    _A.MESSAGE_BOARD_TRAILER:           (MetaDriveType.VEHICLE, None, (3.0, 2.0, 2.5)),
    _A.TRAFFIC_LIGHT_TRAILER:           (MetaDriveType.VEHICLE, None, (3.0, 2.0, 2.5)),
}


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TLFrame:
    """Traffic light detections at one lidar-rate timestep."""
    frame_idx:    int
    timestamp_us: int
    detections:   list


@dataclass
class AgentTrack:
    """Per-agent trajectory across all T frames of a clip."""
    track_id:      str
    label:         object          # raw py123d label (any dataset)
    metadrive_type: str            # pre-resolved MetaDriveType string
    dynamic_kind:  Optional[str]   # "vehicle" / "pedestrian" / "cyclist" / None
    positions:     np.ndarray      # (T, 3) float32 world frame; zeros where valid=False
    headings:      np.ndarray      # (T,)   float32 yaw radians
    sizes:         np.ndarray      # (T, 3) float32 [length, width, height] forward-filled
    valid:         np.ndarray      # (T,)   bool


@dataclass
class IngestedScene:
    """All data extracted from one clip, ready for NCore conversion."""
    clip_id:        str
    ncore_root:     Optional[Path]  # set for NCore clips; None for other datasets
    n_frames:       int
    dt_us:          int             # median lidar sweep interval microseconds
    timestamps_us:  np.ndarray      # (T,) int64 absolute Unix microseconds

    ego_positions:  np.ndarray      # (T, 3) float32 world frame absolute
    ego_headings:   np.ndarray      # (T,)   float32 yaw radians
    ego_velocities: np.ndarray      # (T, 3) float32 world frame

    agent_tracks:   List[AgentTrack]
    traffic_lights: List[TLFrame]

    camera_metas:   dict            # CameraID → camera metadata (any py123d type)
    lidar_meta:     object          # LidarMergedMetadata

    # Kept for NCoreBridge to re-iterate raw sensor data when ncore_root is None.
    log_parser:     object = field(default=None, repr=False)


# ── shared reader base ────────────────────────────────────────────────────────

class _Py123dLogReader:
    """Build IngestedScene objects from any py123d BaseDatasetParser."""

    def __init__(
        self,
        parser,                        # any py123d BaseDatasetParser
        ncore_root: Optional[Path],    # Path for NCore; None for other datasets
    ) -> None:
        self._parser = parser
        self._ncore_root = ncore_root

    def iter_scenes(self) -> Iterator[IngestedScene]:
        for log_parser in self._parser.get_log_parsers():
            scene = self._build_scene(log_parser)
            if scene is not None:
                yield scene

    def _label_info(self, label: object) -> Tuple:
        """Return (metadrive_type, dynamic_kind, default_lwh) for any py123d label."""
        raise NotImplementedError

    def _build_scene(self, log_parser) -> Optional[IngestedScene]:
        clip_id = log_parser.get_log_metadata().log_name

        ego_pos_list:  List[np.ndarray] = []
        ego_yaw_list:  List[float]      = []
        ego_vel_list:  List[np.ndarray] = []
        ts_list:       List[int]        = []

        track_frames: dict  = {}
        tl_frames:    List[TLFrame] = []
        camera_metas: dict  = {}
        lidar_meta          = None

        for frame_idx, sync in enumerate(log_parser.iter_modalities_sync()):
            ts_us = sync.timestamp.time_us
            ts_list.append(ts_us)

            ego:    Optional[EgoStateSE3]      = None
            boxes:  Optional[BoxDetectionsSE3] = None
            tl_det                             = None

            for mod in sync.modalities:
                if isinstance(mod, EgoStateSE3):
                    ego = mod
                elif isinstance(mod, BoxDetectionsSE3):
                    boxes = mod
                elif isinstance(mod, TrafficLightDetections):
                    tl_det = mod
                elif isinstance(mod, ParsedCamera):
                    # ParsedCamera.metadata is declared BaseModalityMetadata but is
                    # always a camera metadata (its ctor only accepts those).
                    cid = cast(BaseCameraMetadata, mod.metadata).camera_id
                    if cid not in camera_metas:
                        camera_metas[cid] = mod.metadata
                elif isinstance(mod, (ParsedLidar, Lidar)):
                    if lidar_meta is None:
                        lidar_meta = getattr(mod, "metadata", None)

            if ego is None:
                continue

            pose = ego.imu_se3
            ego_pos_list.append(np.array([pose.x, pose.y, pose.z], np.float32))
            ego_yaw_list.append(float(pose.yaw))

            dyn = ego.dynamic_state_se3
            if dyn is not None:
                v = dyn.velocity_3d
                ego_vel_list.append(np.array([v.array[0], v.array[1], v.array[2]], np.float32))
            else:
                ego_vel_list.append(np.zeros(3, np.float32))

            if boxes is not None:
                for det in boxes.box_detections:
                    tok = det.attributes.track_token
                    if tok not in track_frames:
                        track_frames[tok] = []
                    track_frames[tok].append((frame_idx, det))

            if tl_det is not None:
                tl_frames.append(TLFrame(
                    frame_idx=frame_idx,
                    timestamp_us=ts_us,
                    detections=list(tl_det.detections),
                ))

        if not ts_list:
            return None

        T = len(ts_list)
        timestamps = np.array(ts_list, dtype=np.int64)
        diffs = np.diff(timestamps)
        dt_us = int(np.median(diffs)) if len(diffs) > 0 else 100_000

        agent_tracks = _build_agent_tracks(track_frames, T, self._label_info)

        return IngestedScene(
            clip_id=clip_id,
            ncore_root=self._ncore_root,
            n_frames=T,
            dt_us=dt_us,
            timestamps_us=timestamps,
            ego_positions=np.stack(ego_pos_list, axis=0).astype(np.float32),
            ego_headings=np.array(ego_yaw_list, dtype=np.float32),
            ego_velocities=np.stack(ego_vel_list, axis=0).astype(np.float32),
            agent_tracks=agent_tracks,
            traffic_lights=tl_frames,
            camera_metas=camera_metas,
            lidar_meta=lidar_meta,
            log_parser=log_parser,
        )


# ── NCore reader ──────────────────────────────────────────────────────────────

class WaymoLogReader(_Py123dLogReader):
    """Iterate NCore clips via py123d and yield one IngestedScene per clip.

    Name kept for backward compatibility. Uses NCoreParser internally.
    """

    def __init__(
        self,
        data_root: "str | Path",
        splits: List[str] | None = None,
        max_clips: Optional[int] = None,
    ) -> None:
        from py123d.parser.ncore.ncore_parser import NCoreParser

        self._data_root = Path(data_root)
        parser = NCoreParser(
            splits=splits or ["ncore_train"],
            ncore_data_root=self._data_root,
            max_clips=max_clips,
        )
        super().__init__(parser, ncore_root=self._data_root)

    def _label_info(self, label: object) -> Tuple:
        return _NCORE_LABEL_INFO.get(label, _DEFAULT_LABEL_INFO)


# ── WOD reader ────────────────────────────────────────────────────────────────

class WODLogReader(_Py123dLogReader):
    """Iterate Waymo Open Dataset clips via py123d and yield one IngestedScene per clip."""

    def __init__(
        self,
        data_root: "str | Path",
        splits: List[str] | None = None,
        max_clips: Optional[int] = None,
    ) -> None:
        from py123d.parser.wod.wod_perception_parser import WODPerceptionParser

        parser = WODPerceptionParser(
            splits=splits or ["wod-perception_train"],
            wod_perception_data_root=str(data_root),
        )
        if max_clips is not None:
            parser._split_tf_record_pairs = parser._split_tf_record_pairs[:max_clips]

        super().__init__(parser, ncore_root=None)

    def _label_info(self, label: object) -> Tuple:
        return _WOD_LABEL_INFO.get(label, _DEFAULT_LABEL_INFO)


# ── NavSim / OpenScene label taxonomy ─────────────────────────────────────────
# NavSim IS nuPlan/OpenScene data; NavsimParser emits native NuPlanBoxDetectionLabel
# enums. Static clutter (cones/barriers/signs/misc) → dynamic_kind None so it is
# kept as scene geometry but not treated as a dynamic traffic agent.

def _nuplan_label_info() -> dict:
    from py123d.parser.registry import NuPlanBoxDetectionLabel as _N
    return {
        _N.VEHICLE:        (MetaDriveType.VEHICLE,    "vehicle",    (4.5, 1.8, 1.5)),
        _N.BICYCLE:        (MetaDriveType.CYCLIST,    "cyclist",    (1.8, 0.7, 1.7)),
        _N.PEDESTRIAN:     (MetaDriveType.PEDESTRIAN, "pedestrian", (0.5, 0.5, 1.75)),
        _N.TRAFFIC_CONE:   (MetaDriveType.VEHICLE,    None,         (0.3, 0.3, 0.6)),
        _N.BARRIER:        (MetaDriveType.VEHICLE,    None,         (1.0, 0.3, 1.0)),
        _N.CZONE_SIGN:     (MetaDriveType.VEHICLE,    None,         (0.5, 0.5, 1.5)),
        _N.GENERIC_OBJECT: (MetaDriveType.VEHICLE,    None,         (0.5, 0.5, 0.5)),
    }


_NAVSIM_LABEL_INFO: dict = _nuplan_label_info()


# ── NavSim / OpenScene reader ─────────────────────────────────────────────────

class NavsimLogReader(_Py123dLogReader):
    """Iterate NavSim/OpenScene scenes via the in-repo NavsimParser and yield one
    IngestedScene per scene (~40 frames @ 2 Hz ≈ 20 s).

    ``data_root`` is the directory of per-log ``<log>.pkl`` files (e.g.
    ``/data/navsim/test_navsim_logs/test``).  ``sensor_root`` is the
    sensor-blob tree (e.g. ``/data/navsim/test_sensor_blobs/test``); when
    omitted it is inferred by swapping ``*_navsim_logs/<split>`` →
    ``*_sensor_blobs/<split>``.
    """

    def __init__(
        self,
        data_root: "str | Path",
        sensor_root: "str | Path | None" = None,
        splits: List[str] | None = None,
        log_names: List[str] | None = None,
        scene_tokens: List[str] | None = None,
        max_clips: Optional[int] = None,
    ) -> None:
        from navsafe.gs3d_converter.navsim_parser import NavsimParser

        data_root = Path(data_root)
        if sensor_root is None:
            # test_navsim_logs/test → test_sensor_blobs/test
            split_dir = data_root.name
            base = data_root.parent.name.replace("_navsim_logs", "_sensor_blobs")
            sensor_root = data_root.parent.parent / base / split_dir
        split = (splits or ["navsim_test"])[0]

        parser = NavsimParser(
            navsim_data_root=data_root,
            sensor_root=sensor_root,
            split=split,
            log_names=log_names,
            scene_tokens=scene_tokens,
            max_clips=max_clips,
        )
        super().__init__(parser, ncore_root=None)

    def _label_info(self, label: object) -> Tuple:
        return _NAVSIM_LABEL_INFO.get(label, _DEFAULT_LABEL_INFO)


# ── raw nuPlan reader (native 10 Hz sensors from the .db) ─────────────────────
# For a SHARP recon, ingest the full-rate nuPlan data (10 Hz cameras, 20 Hz
# lidar, native ego/boxes from the .db) instead of NavSim's 2 Hz subsample. Each
from navsafe.gs3d_converter._windowed_sync_patch import _SYNC_WINDOW  # perf: window sync reads


# navhard scene is a ~20 s window of a nuPlan log, so we slice the log parser to
# the scene's [t0, t1]. Labels are native NuPlanBoxDetectionLabel (reuse the
# nuPlan label table). See [[navhard-integration-progress]].

class _WindowedNuplanLogParser:
    """Wrap a py123d NuplanLogParser, exposing only the frames in [t0, t1] and
    renaming the log to the scene token (so the recon clip is per-window)."""

    def __init__(self, inner, scene_token: str, t0_us: int, t1_us: int) -> None:
        self._inner = inner
        self._token = scene_token
        self._t0 = int(t0_us)
        self._t1 = int(t1_us)
        self._sync_cache = None

    def get_log_metadata(self):
        from py123d.datatypes import LogMetadata
        md = self._inner.get_log_metadata()
        return LogMetadata(dataset=md.dataset, split=md.split, log_name=self._token,
                           location=md.location, map_metadata=md.map_metadata)

    def iter_modalities_sync(self):
        # NCore preparation needs the same window three times (scene metadata,
        # structured lidar calibration, shard write). Cache the finite window
        # after its first expensive NuPlan ORM traversal.
        if self._sync_cache is None:
            self._sync_cache = []
            _SYNC_WINDOW.win = (self._t0, self._t1)  # perf: skip out-of-window frames in inner ORM loop
            try:
                for sync in self._inner.iter_modalities_sync():
                    t = sync.timestamp.time_us
                    if t < self._t0:
                        continue
                    if t > self._t1:
                        break  # frames are time-ordered
                    self._sync_cache.append(sync)
            finally:
                _SYNC_WINDOW.win = None
        yield from self._sync_cache

    def iter_modalities_async(self):
        """Expose native-rate modalities around the window.

        The bridge consumes the leading native ego-pose stream to cover each
        50 ms lidar sweep. Keep one 100 ms sample of padding on both sides so
        it can interpolate exact sequence-boundary poses.
        """
        padded_start = self._t0 - 100_000
        padded_stop = self._t1 + 100_000
        for modality in self._inner.iter_modalities_async():
            stamp = getattr(modality, "timestamp", None)
            if stamp is None:
                stamp = getattr(modality, "start_timestamp", None)
            if stamp is None:
                continue
            timestamp_us = int(stamp.time_us)
            if padded_start <= timestamp_us <= padded_stop:
                yield modality

class NuplanLogReader(_Py123dLogReader):
    """Yield one IngestedScene per navhard scene from raw nuPlan (10 Hz).

    ``scenes`` is a list of ``(log_name, scene_token, t0_us, t1_us)`` — the
    nuPlan log and the scene's time window (from the navhard/navsim token).
    """

    def __init__(
        self,
        nuplan_data_root: "str | Path",
        sensor_root: "str | Path",
        maps_root: "str | Path",
        scenes: List[Tuple],
        split: str = "nuplan_test",
    ) -> None:
        from py123d.parser.nuplan.nuplan_parser import NuplanParser

        self._scenes = list(scenes)
        log_names = sorted({s[0] for s in self._scenes})
        parser = NuplanParser(
            splits=[split],
            nuplan_data_root=str(nuplan_data_root),
            nuplan_maps_root=str(maps_root),
            nuplan_sensor_root=str(sensor_root),
            log_names=log_names,
        )
        self._nuplan_parser = parser
        super().__init__(parser, ncore_root=None)

    def get_map_parsers(self):
        return self._nuplan_parser.get_map_parsers()

    def iter_scenes(self) -> Iterator[IngestedScene]:
        by_log = {lp.get_log_metadata().log_name: lp
                  for lp in self._nuplan_parser.get_log_parsers()}
        for (log_name, token, t0, t1) in self._scenes:
            inner = by_log.get(log_name)
            if inner is None:
                continue
            scene = self._build_scene(_WindowedNuplanLogParser(inner, token, int(t0), int(t1)))
            if scene is not None:
                yield scene

    def _label_info(self, label: object) -> Tuple:
        return _NAVSIM_LABEL_INFO.get(label, _DEFAULT_LABEL_INFO)  # native nuPlan labels


# ── AV2 sensor reader ─────────────────────────────────────────────────────────

class AV2LogReader(_Py123dLogReader):
    """Iterate Argoverse 2 Sensor clips via py123d and yield one IngestedScene per clip.

    ``data_root`` is the AV2 *sensor* root — the directory containing
    ``train/ val/ test/`` with one ``<log_uuid>/`` folder per clip
    (feather calibration + ``sensors/cameras|lidar``).
    """

    def __init__(
        self,
        data_root: "str | Path",
        splits: List[str] | None = None,
        max_clips: Optional[int] = None,
    ) -> None:
        from py123d.parser.av2.av2_sensor_parser import Av2SensorParser

        parser = Av2SensorParser(
            splits=splits or ["av2-sensor_train"],
            av2_sensor_root=str(data_root),
        )
        if max_clips is not None:
            parser._log_paths_and_split = parser._log_paths_and_split[:max_clips]

        super().__init__(parser, ncore_root=None)

    def _label_info(self, label: object) -> Tuple:
        return _AV2_LABEL_INFO.get(label, _DEFAULT_LABEL_INFO)


# ── converted Arrow-log reader ────────────────────────────────────────────────

class _ArrowSceneLogParser:
    """Adapter: py123d ``ArrowSceneAPI`` → the ``BaseLogParser`` surface the
    ingestion + NCoreBridge paths consume (``get_log_metadata`` +
    ``iter_modalities_sync``).

    Cameras are surfaced as ``ParsedCamera`` carrying the JPEG bytes stored in
    the Arrow table (no decode/re-encode round trip); lidar is surfaced as the
    in-memory py123d ``Lidar`` datatype (ego-frame xyz + intensity), which
    NCoreBridge handles alongside the path-backed ``ParsedLidar``.
    """

    def __init__(self, scene) -> None:
        self._scene = scene
        self._sensor_root: "Path | None" = None

    def get_log_metadata(self):
        return self._scene.get_log_metadata()

    def _read_sensor_file(self, relative_path: str) -> "bytes | None":
        """Resolve a path-variant sensor reference against the dataset sensor
        root (same convention as py123d's path-backed lidar) and return the raw
        file bytes, or None if unresolvable."""
        if self._sensor_root is None:
            from py123d.common.runtime import get_dataset_paths

            dataset = str(getattr(self.get_log_metadata(), "dataset", "") or "")
            self._sensor_root = get_dataset_paths().get_sensor_root(dataset)
        if self._sensor_root is None:
            return None
        p = Path(self._sensor_root) / relative_path
        return p.read_bytes() if p.exists() else None

    def _lidar_ids(self) -> List[str]:
        ids: List[str] = []
        for key in self._scene.get_all_modality_metadatas():
            if key.startswith("lidar."):
                ids.append(key.split(".", 1)[1])
        return ids

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        scene = self._scene
        cam_metas = scene.get_camera_metadatas()
        lidar_ids = self._lidar_ids()
        n = len(scene.get_all_iteration_timestamps())

        for i in range(n):
            ts = scene.get_timestamp_at_iteration(i)
            mods: list = []

            ego = scene.get_ego_state_se3_at_iteration(i)
            if ego is not None:
                mods.append(ego)

            boxes = scene.get_box_detections_se3_at_iteration(i)
            if boxes is not None:
                mods.append(boxes)

            tl = scene.get_traffic_light_detections_at_iteration(i)
            if tl is not None:
                mods.append(tl)

            for cam_id, meta in cam_metas.items():
                jpeg = scene.get_modality_column_at_iteration(
                    i, column="data", modality_type="camera", modality_id=cam_id,
                )
                if isinstance(jpeg, str):
                    # Path-variant Arrow log: the data column stores a path
                    # relative to the dataset sensor root (same convention as
                    # path-backed lidar). Read the JPEG bytes from disk — still
                    # no decode/re-encode round trip.
                    jpeg = self._read_sensor_file(jpeg)
                if not isinstance(jpeg, (bytes, bytearray)):
                    continue  # missing frame, or an mp4-codec log
                pose = scene.get_modality_column_at_iteration(
                    i, column="camera_to_global_se3", modality_type="camera",
                    modality_id=cam_id, deserialize=True,
                )
                cam_ts = scene.get_modality_column_at_iteration(
                    i, column="timestamp_us", modality_type="camera",
                    modality_id=cam_id, deserialize=True,
                )
                mods.append(ParsedCamera(
                    metadata=meta,
                    timestamp=cam_ts if cam_ts is not None else ts,
                    camera_to_global_se3=pose,
                    byte_string=bytes(jpeg),
                ))

            for lid_id in lidar_ids:
                lidar = scene.get_lidar_at_iteration(i, lid_id)
                if lidar is not None:
                    mods.append(lidar)

            yield ModalitiesSync(timestamp=ts, modalities=mods)


class ArrowLogReader(_Py123dLogReader):
    """Iterate converted py123d Arrow logs and yield one IngestedScene per log.

    ``data_root`` is a py123d Arrow data root (a directory containing
    ``logs/<split>/<log_id>/*.arrow``) — the same root the eval env loads.
    Dataset-agnostic: the box-label taxonomy is picked per log from the log
    metadata's dataset name.
    """

    _LABEL_MAPS: List[Tuple[str, dict]] = []  # filled lazily below

    def __init__(
        self,
        data_root: "str | Path",
        max_clips: Optional[int] = None,
    ) -> None:
        from py123d.api import SceneFilter, get_filtered_scenes

        scenes = list(get_filtered_scenes(SceneFilter(), data_root=str(data_root)))
        scenes.sort(key=lambda s: s.get_log_metadata().log_name)
        if max_clips is not None:
            scenes = scenes[:max_clips]
        self._scenes = scenes
        self._ncore_root = None
        self._active_label_map: dict = {}

    def iter_scenes(self) -> Iterator[IngestedScene]:
        for arrow_scene in self._scenes:
            self._active_label_map = self._label_map_for(
                str(getattr(arrow_scene.get_log_metadata(), "dataset", "") or "")
            )
            scene = self._build_scene(_ArrowSceneLogParser(arrow_scene))
            if scene is not None:
                yield scene

    @staticmethod
    def _label_map_for(dataset: str) -> dict:
        if dataset.startswith("av2"):
            return _AV2_LABEL_INFO
        if dataset.startswith("wod"):
            return _WOD_LABEL_INFO
        if dataset.startswith(("ncore", "physical_ai", "pai")):
            return _NCORE_LABEL_INFO
        return {}

    def _label_info(self, label: object) -> Tuple:
        return self._active_label_map.get(label, _DEFAULT_LABEL_INFO)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_agent_tracks(
    track_frames: dict,
    T: int,
    label_info_fn: Callable,
) -> List[AgentTrack]:
    tracks: List[AgentTrack] = []

    for tok, frame_det_list in track_frames.items():
        if not frame_det_list:
            continue

        first_det = frame_det_list[0][1]
        label = first_det.attributes.label
        md_type, dyn_kind, default_lwh = label_info_fn(label)

        positions = np.zeros((T, 3), np.float32)
        headings  = np.zeros(T, np.float32)
        sizes     = np.zeros((T, 3), np.float32)
        valid     = np.zeros(T, bool)

        for frame_idx, det in frame_det_list:
            if frame_idx >= T:
                continue
            bb = det.bounding_box_se3
            c  = bb.center_se3
            positions[frame_idx] = [c.x, c.y, c.z]
            headings[frame_idx]  = float(c.yaw)
            sizes[frame_idx]     = [bb.length, bb.width, bb.height]
            valid[frame_idx]     = True

        # Forward-fill sizes from last valid frame.
        last_size = np.array(default_lwh, np.float32)
        for t in range(T):
            if valid[t]:
                last_size = sizes[t].copy()
            else:
                sizes[t] = last_size

        tracks.append(AgentTrack(
            track_id=tok,
            label=label,
            metadrive_type=md_type,
            dynamic_kind=dyn_kind,
            positions=positions,
            headings=headings,
            sizes=sizes,
            valid=valid,
        ))

    return tracks
