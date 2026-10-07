# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""py123d parser for NavSim / OpenScene (nuPlan-derived) sensor logs.

NavSim ships nuPlan re-packaged as self-contained per-log pickles plus a
sensor-blob tree, so — unlike py123d's ``nuplan`` parser — this reader needs
**no** nuPlan devkit / sqlite dependency.  It reads directly from:

    <navsim_data_root>/<log>.pkl                       (list of 2 Hz frame dicts)
    <sensor_root>/<log>/<CAM>/<token>.jpg              (8 pinhole cameras)
    <sensor_root>/<log>/MergedPointCloud/<token>.pcd   (merged lidar sweep)

Each frame dict carries everything we need (verified against the navhard
`test` split): ``ego2global_translation/rotation`` (scalar-first quaternion),
``ego_dynamic_state``, per-camera ``sensor2lidar_*`` + ``cam_intrinsic`` +
``distortion``, and ``anns`` (``gt_boxes`` in the ego/lidar frame — lidar2ego
is identity in NavSim — plus ``gt_names`` / ``track_tokens`` / ``gt_velocity_3d``).

The unit of a "log" here is one NavSim **scene** (``scene_token``): 40 frames
@ 2 Hz ≈ 20 s.  That makes each reconstruction clip a single ~20 s scene, which
matches the navhard evaluation window (and BridgeSim's ``--num-future-frames-extract 40``).

Downstream, ``navsafe.gs3d_converter.log_ingestion.NavsimLogReader`` wraps this
parser to build an ``IngestedScene`` per scene, feeding both the NuRec
reconstruction and the py123d Arrow scenario source.
"""

from __future__ import annotations

import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, cast

import numpy as np

from py123d.datatypes import (
    BoxDetectionAttributes,
    BoxDetectionSE3,
    BoxDetectionsSE3,
    CameraID,
    DynamicStateSE3,
    EgoStateSE3,
    LidarID,
    LidarMetadata,
    LogMetadata,
    PinholeCameraMetadata,
    PinholeDistortion,
    PinholeIntrinsics,
    Timestamp,
    TrafficLightDetections,
)
from py123d.datatypes.metadata.map_metadata import MapMetadata
from py123d.datatypes.modalities.base_modality import BaseModality
from py123d.datatypes.sensors.lidar import LidarMergedMetadata
from py123d.geometry import BoundingBoxSE3, EulerAngles, PoseSE3, Vector3D
from py123d.geometry.transform.transform_se3 import rel_to_abs_se3
from py123d.geometry.utils.constants import DEFAULT_PITCH, DEFAULT_ROLL
from py123d.parser.base_dataset_parser import (
    BaseDatasetParser,
    BaseLogParser,
    ModalitiesSync,
    ParsedCamera,
    ParsedLidar,
)
from py123d.parser.registry import NuPlanBoxDetectionLabel
# NavSim/OpenScene IS nuPlan data, so its ``gt_names`` are the nuPlan category
# strings; reuse py123d's own nuPlan name→label map (single source of truth,
# identical to py123d's nuplan parser).
from py123d.parser.nuplan.utils.nuplan_constants import NUPLAN_DETECTION_NAME_DICT

# Reuse nuPlan's generic modality-metadata singletons — they are pure py123d
# constants (no nuplan-devkit import) describing the ego / box modalities.
from py123d.parser.nuplan.utils.nuplan_constants import (
    NUPLAN_BOX_DETECTIONS_SE3_METADATA,
    NUPLAN_EGO_STATE_SE3_METADATA,
)

logger = logging.getLogger(__name__)

# NavSim camera-folder name → py123d CameraID.  NavSim uses the nuPlan 8-camera
# rig; PCAM_* ids keep us aligned with the nuplan parser's conventions.
NAVSIM_CAMERA_MAPPING: "OrderedDict[str, CameraID]" = OrderedDict(
    [
        ("CAM_F0", CameraID.PCAM_F0),
        ("CAM_B0", CameraID.PCAM_B0),
        ("CAM_L0", CameraID.PCAM_L0),
        ("CAM_L1", CameraID.PCAM_L1),
        ("CAM_L2", CameraID.PCAM_L2),
        ("CAM_R0", CameraID.PCAM_R0),
        ("CAM_R1", CameraID.PCAM_R1),
        ("CAM_R2", CameraID.PCAM_R2),
    ]
)

# Nominal merged-lidar sweep duration (NavSim is 2 Hz; the merged cloud is an
# accumulated instant, so the exact end stamp is not load-bearing).
NAVSIM_LIDAR_SWEEP_DURATION_US = 100_000


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate a heading, taking the shortest way around the ±π wrap."""
    d = (b - a + np.pi) % (2 * np.pi) - np.pi
    return a + d * t


def _interpolate_frames(frames: List[dict], target_dt_us: int) -> List[dict]:
    """Upsample native 2 Hz NavSim frames to a ``target_dt_us`` timeline.

    Interpolates ego pose (translation lerp + quaternion slerp), ego dynamics,
    and per-track boxes (position/size lerp + heading angle-lerp), matched by
    ``track_token``. Traffic lights hold from the earlier bracketing frame.
    Cameras/lidar are NOT interpolated (only 2 Hz jpgs exist) — this mode is for
    the scenario-source arrow, which excludes sensor modalities. Returns virtual
    frame dicts with the same keys the modality builders read.
    """
    from scipy.spatial.transform import Rotation as _R, Slerp as _Slerp

    frames = sorted(frames, key=lambda f: int(f["timestamp"]))
    times = np.array([int(f["timestamp"]) for f in frames], dtype=np.int64)
    if len(frames) < 2:
        return frames

    trans = np.array([np.asarray(f["ego2global_translation"], np.float64) for f in frames])
    quats_wxyz = np.array([np.asarray(f["ego2global_rotation"], np.float64) for f in frames])
    ego_R = _R.from_quat(quats_wxyz[:, [1, 2, 3, 0]])  # scipy wants xyzw
    slerp = _Slerp(times.astype(np.float64), ego_R)
    dyn = np.array([np.asarray(f.get("ego_dynamic_state", np.zeros(4)), np.float64).reshape(-1)[:4]
                    if np.asarray(f.get("ego_dynamic_state", np.zeros(4))).size >= 4 else np.zeros(4)
                    for f in frames])

    # Per-track box tables keyed by frame index for interpolation.
    track_at = []  # list of dict: track_token -> (box7, name, vel3)
    for f in frames:
        anns = f.get("anns") or {}
        gt_boxes = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))), np.float64).reshape(-1, 7)
        gt_names = list(anns.get("gt_names", []))
        gt_vel = np.asarray(anns.get("gt_velocity_3d", np.zeros((len(gt_boxes), 3))), np.float64).reshape(-1, 3)
        toks = list(anns.get("track_tokens", []))
        d = {}
        for i in range(len(gt_boxes)):
            tok = str(toks[i]) if i < len(toks) else f"navsim_{i}"
            d[tok] = (gt_boxes[i], str(gt_names[i]) if i < len(gt_names) else "vehicle",
                      gt_vel[i] if i < len(gt_vel) else np.zeros(3))
        track_at.append(d)

    out: List[dict] = []
    t = int(times[0])
    t_end = int(times[-1])
    while t <= t_end:
        j = int(np.searchsorted(times, t, side="right"))
        if j <= 0:
            ia, ib, alpha = 0, 0, 0.0
        elif j >= len(times):
            ia, ib, alpha = len(times) - 1, len(times) - 1, 0.0
        else:
            ia, ib = j - 1, j
            span = float(times[ib] - times[ia]) or 1.0
            alpha = float(t - times[ia]) / span

        q = slerp([float(t)]).as_quat()[0]  # xyzw
        tr = trans[ia] * (1 - alpha) + trans[ib] * alpha
        dd = dyn[ia] * (1 - alpha) + dyn[ib] * alpha

        # interpolate boxes: union of tracks in the two bracket frames.
        da, db = track_at[ia], track_at[ib]
        boxes, names, vels, toks = [], [], [], []
        for tok in list(da.keys()) + [k for k in db if k not in da]:
            if tok in da and tok in db and ia != ib:
                ba, na, va = da[tok]; bb, _, vb = db[tok]
                box = ba * (1 - alpha) + bb * alpha
                box[6] = _lerp_angle(float(ba[6]), float(bb[6]), alpha)
                vel = va * (1 - alpha) + vb * alpha
                boxes.append(box); names.append(na); vels.append(vel); toks.append(tok)
            else:
                # tok comes from da's keys plus db-only keys, so at least one
                # of the two lookups always hits; None is unreachable here.
                bx, nm, vv = cast(Any, da.get(tok, db.get(tok)))
                boxes.append(bx); names.append(nm); vels.append(vv); toks.append(tok)

        out.append({
            "timestamp": int(t),
            "map_location": frames[ia].get("map_location", "unknown"),
            "ego2global_translation": tr,
            "ego2global_rotation": np.array([q[3], q[0], q[1], q[2]]),  # back to wxyz
            "ego_dynamic_state": dd,
            "anns": {
                "gt_boxes": np.array(boxes, np.float64).reshape(-1, 7),
                "gt_names": names,
                "gt_velocity_3d": np.array(vels, np.float64).reshape(-1, 3),
                "track_tokens": toks,
            },
            "traffic_lights": frames[ia].get("traffic_lights"),
            "cams": {},  # sensors excluded from the arrow; not interpolated
        })
        t += target_dt_us
    return out


def _pose_from_translation_quat(translation: np.ndarray, quat_wxyz: np.ndarray) -> PoseSE3:
    """Build a PoseSE3 from a (3,) translation and a scalar-first (w,x,y,z) quaternion."""
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return PoseSE3(x=float(t[0]), y=float(t[1]), z=float(t[2]),
                   qw=float(q[0]), qx=float(q[1]), qy=float(q[2]), qz=float(q[3]))


def _image_size(image_path: Path) -> "tuple[int, int] | None":
    """Return (width, height) of a JPEG by reading only its SOF header — no full
    decode, no PIL dependency."""
    try:
        with open(image_path, "rb") as fh:
            data = fh.read()
        i = 2  # skip SOI (0xFFD8)
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            # SOF0..SOF15 (except DHT=C4, DNL=C8, DAC=CC) carry frame dims.
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height = (data[i + 5] << 8) | data[i + 6]
                width = (data[i + 7] << 8) | data[i + 8]
                return int(width), int(height)
            seg_len = (data[i + 2] << 8) | data[i + 3]
            i += 2 + seg_len
    except Exception:  # noqa: BLE001
        return None
    return None


def _camera_metadata_from_frame(
    cam_name: str, cam: dict, image_path: "Path | None" = None
) -> PinholeCameraMetadata:
    """Static pinhole metadata (intrinsics/distortion/extrinsic) for one camera.

    NavSim's ``sensor2lidar_*`` are the camera→lidar extrinsics, and lidar2ego is
    identity, so sensor2lidar == camera→ego (== camera→imu in py123d terms).
    """
    K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
    intrinsics = PinholeIntrinsics.from_camera_matrix(K)
    distortion = PinholeDistortion.from_array(
        np.asarray(cam.get("distortion", np.zeros(5)), dtype=np.float64), copy=False
    )
    R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
    t = np.asarray(cam["sensor2lidar_translation"], dtype=np.float64)
    camera_to_imu = PoseSE3.from_R_t(rotation=R, translation=t)

    # Read the TRUE image size from the jpg header — NavSim's principal point is
    # NOT centred (cy≈560 for a 1080-tall image), so cy×2 would give a wrong
    # height (1120) and break aux/recon. Fall back to intrinsics×2 if unreadable.
    wh = _image_size(image_path) if image_path is not None else None
    if wh is not None:
        width, height = wh
    else:
        width = int(round(float(K[0, 2]) * 2)) or 1920
        height = int(round(float(K[1, 2]) * 2)) or 1080

    return PinholeCameraMetadata(
        camera_name=cam_name,
        camera_id=NAVSIM_CAMERA_MAPPING[cam_name],
        intrinsics=intrinsics,
        distortion=distortion,
        width=width,
        height=height,
        camera_to_imu_se3=camera_to_imu,
    )


class NavsimSceneLogParser(BaseLogParser):
    """A single NavSim scene (``scene_token``, ~40 frames @ 2 Hz) as a py123d log."""

    def __init__(
        self,
        *,
        split: str,
        scene_token: str,
        frames: List[dict],
        sensor_root: Path,
        source_log_name: str,
        interpolate_hz: Optional[int] = None,
    ) -> None:
        self._split = split
        self._scene_token = scene_token
        self._frames = frames
        self._sensor_root = Path(sensor_root)
        self._source_log_name = source_log_name
        self._interpolate_hz = interpolate_hz
        self._location = frames[0].get("map_location", "unknown") if frames else "unknown"

    def get_log_metadata(self) -> LogMetadata:
        return LogMetadata(
            dataset="navsim",
            split=self._split,
            log_name=self._scene_token,
            location=self._location,
            map_metadata=MapMetadata(
                # Must match the LOG dataset ("navsim"): the py123d adapter
                # resolves the map path as maps/<log.dataset>/<log.dataset>_
                # <location>.arrow. The map itself is the nuPlan HD map, written
                # under the same "navsim" dataset name by _NavsimMapParser below.
                dataset="navsim",
                location=self._location,
                map_has_z=False,
                map_is_per_log=False,
            ),
        )

    # -- per-frame modality construction --------------------------------------

    def _ego_state(self, frame: dict, ts: Timestamp) -> EgoStateSE3:
        imu_pose = _pose_from_translation_quat(
            frame["ego2global_translation"], frame["ego2global_rotation"]
        )
        dyn = np.asarray(frame.get("ego_dynamic_state", np.zeros(4)), dtype=np.float64).reshape(-1)
        vx, vy = (float(dyn[0]), float(dyn[1])) if dyn.size >= 2 else (0.0, 0.0)
        ax, ay = (float(dyn[2]), float(dyn[3])) if dyn.size >= 4 else (0.0, 0.0)
        dynamic_state = DynamicStateSE3(
            velocity=Vector3D(x=vx, y=vy, z=0.0),
            acceleration=Vector3D(x=ax, y=ay, z=0.0),
            angular_velocity=Vector3D(x=0.0, y=0.0, z=0.0),
        )
        return EgoStateSE3.from_imu(
            imu_se3=imu_pose,
            metadata=NUPLAN_EGO_STATE_SE3_METADATA,
            dynamic_state_se3=dynamic_state,
            timestamp=ts,
        )

    def _box_detections(self, frame: dict, ego_pose: PoseSE3, ts: Timestamp) -> BoxDetectionsSE3:
        anns = frame.get("anns") or {}
        gt_boxes = np.asarray(anns.get("gt_boxes", np.zeros((0, 7))), dtype=np.float64).reshape(-1, 7)
        gt_names = list(anns.get("gt_names", []))
        gt_vel = np.asarray(anns.get("gt_velocity_3d", np.zeros((len(gt_boxes), 3))), dtype=np.float64).reshape(-1, 3)
        track_tokens = list(anns.get("track_tokens", []))

        detections: List[BoxDetectionSE3] = []
        for i in range(len(gt_boxes)):
            x, y, z, length, width, height, yaw = gt_boxes[i]
            q = EulerAngles(roll=DEFAULT_ROLL, pitch=DEFAULT_PITCH, yaw=float(yaw)).quaternion
            local = PoseSE3(x=float(x), y=float(y), z=float(z),
                            qw=q.qw, qx=q.qx, qy=q.qy, qz=q.qz)
            # NavSim boxes are in the ego/lidar frame → lift to global.
            center_global = rel_to_abs_se3(origin=ego_pose, pose_se3=local)
            bbox = BoundingBoxSE3(
                center_se3=center_global,
                length=float(length), width=float(width), height=float(height),
            )
            vel = gt_vel[i] if i < len(gt_vel) else np.zeros(3)
            tok = str(track_tokens[i]) if i < len(track_tokens) else f"navsim_{i}"
            name = str(gt_names[i]) if i < len(gt_names) else "vehicle"
            label = NUPLAN_DETECTION_NAME_DICT.get(name, NuPlanBoxDetectionLabel.GENERIC_OBJECT)
            detections.append(
                BoxDetectionSE3(
                    attributes=BoxDetectionAttributes(label=label, track_token=tok),
                    bounding_box_se3=bbox,
                    velocity_3d=Vector3D(x=float(vel[0]), y=float(vel[1]), z=float(vel[2])),
                )
            )
        return BoxDetectionsSE3(box_detections=detections, timestamp=ts,
                                metadata=NUPLAN_BOX_DETECTIONS_SE3_METADATA)

    def _cameras(self, frame: dict, ego_pose: PoseSE3, ts: Timestamp,
                 cam_metas: Dict[str, PinholeCameraMetadata]) -> List[ParsedCamera]:
        cams = frame.get("cams") or {}
        out: List[ParsedCamera] = []
        for cam_name, meta in cam_metas.items():
            cam = cams.get(cam_name)
            if not cam:
                continue
            rel = cam.get("data_path")
            if not rel:
                continue
            if not (self._sensor_root / rel).exists():
                continue
            camera_to_global = rel_to_abs_se3(origin=ego_pose, pose_se3=meta.camera_to_imu_se3)
            out.append(
                ParsedCamera(
                    metadata=meta,
                    timestamp=ts,
                    camera_to_global_se3=camera_to_global,
                    dataset_root=self._sensor_root,
                    relative_path=rel,
                )
            )
        return out

    def _lidar(self, frame: dict, ts: Timestamp, lidar_meta: LidarMergedMetadata) -> Optional[ParsedLidar]:
        rel = frame.get("lidar_path")
        if not rel or not (self._sensor_root / rel).exists():
            return None
        return ParsedLidar(
            metadata=lidar_meta,
            start_timestamp=ts,
            end_timestamp=Timestamp.from_us(ts.time_us + NAVSIM_LIDAR_SWEEP_DURATION_US),
            dataset_root=self._sensor_root,
            relative_path=rel,
        )

    def iter_modalities_sync(self) -> Iterator[ModalitiesSync]:
        if not self._frames:
            return

        # ── interpolated (10 Hz) scenario-source mode: ego + boxes + TL only ──
        # NavSim is native 2 Hz; the sim/eval steps at 10 Hz, so replaying 0.5 s
        # frames at 0.1 s makes the ego 5× too fast. Upsample the trajectory here
        # (no sensors — the arrow excludes them; the grpc render supplies pixels).
        if self._interpolate_hz:
            target_dt_us = int(round(1_000_000 / self._interpolate_hz))
            for frame in _interpolate_frames(self._frames, target_dt_us):
                ts = Timestamp.from_us(int(frame["timestamp"]))
                ego = self._ego_state(frame, ts)
                yield ModalitiesSync(timestamp=ts, modalities=[
                    ego,
                    self._box_detections(frame, ego.imu_se3, ts),
                    TrafficLightDetections(detections=[], timestamp=ts),
                ])
            return

        # ── native 2 Hz mode (recon path): full sensors ──────────────────────
        # Static, per-scene sensor metadata (built once from the first frame).
        # Pass the first jpg so the true image size is read from its header.
        cam_metas: Dict[str, PinholeCameraMetadata] = {}
        for cam_name, cam in (self._frames[0].get("cams") or {}).items():
            if cam_name in NAVSIM_CAMERA_MAPPING and cam:
                rel = cam.get("data_path")
                img_path = (self._sensor_root / rel) if rel else None
                cam_metas[cam_name] = _camera_metadata_from_frame(cam_name, cam, img_path)
        lidar_meta = LidarMergedMetadata(
            {LidarID.LIDAR_MERGED: LidarMetadata(
                lidar_name=LidarID.LIDAR_MERGED.serialize(),
                lidar_id=LidarID.LIDAR_MERGED,
                lidar_to_imu_se3=PoseSE3.identity(),
            )}
        )

        for frame in self._frames:
            ts = Timestamp.from_us(int(frame["timestamp"]))
            ego = self._ego_state(frame, ts)
            ego_pose = ego.imu_se3
            modalities: List[BaseModality] = [
                ego,
                self._box_detections(frame, ego_pose, ts),
                TrafficLightDetections(detections=[], timestamp=ts),
            ]
            modalities.extend(self._cameras(frame, ego_pose, ts, cam_metas))
            parsed_lidar = self._lidar(frame, ts, lidar_meta)
            if parsed_lidar is not None:
                modalities.append(parsed_lidar)
            yield ModalitiesSync(timestamp=ts, modalities=modalities)


def _make_navsim_map_parser_cls():
    """Build a NuplanMapParser subclass that reports dataset='navsim'.

    The nuPlan gpkg reading is unchanged (keyed by ``location``); only the
    MapMetadata dataset label is overridden so ArrowMapWriter writes the map to
    ``maps/navsim/navsim_<location>.arrow`` — where the py123d adapter looks
    (it resolves the map from the LOG's dataset, which is 'navsim').
    """
    from py123d.parser.nuplan.nuplan_map_parser import NuplanMapParser

    class _NavsimMapParser(NuplanMapParser):
        def get_map_metadata(self) -> MapMetadata:
            md = super().get_map_metadata()
            return MapMetadata(
                dataset="navsim",
                location=md.location,
                map_has_z=getattr(md, "map_has_z", False),
                map_is_per_log=getattr(md, "map_is_per_log", False),
            )

    return _NavsimMapParser


_NavsimMapParser = _make_navsim_map_parser_cls()


class NavsimParser(BaseDatasetParser):
    """Discover NavSim/OpenScene scenes and expose one log parser per scene.

    Parameters
    ----------
    navsim_data_root : dir of per-log ``<log>.pkl`` files (e.g.
        ``/data/navsim/test_navsim_logs/test``).
    sensor_root : root of the sensor-blob tree (e.g.
        ``/data/navsim/test_sensor_blobs/test``); camera ``data_path`` and
        ``lidar_path`` are resolved relative to it.
    split : py123d split tag written into every LogMetadata.
    log_names : optional whitelist of log stems to scan (defaults to all).
    scene_tokens : optional whitelist of ``scene_token`` values to keep
        (e.g. the navhard stage-1 tokens).
    max_clips : cap on the number of scenes yielded.
    """

    def __init__(
        self,
        *,
        navsim_data_root: "str | Path",
        sensor_root: "str | Path",
        maps_root: "str | Path | None" = None,
        split: str = "navsim_test",
        log_names: Optional[Sequence[str]] = None,
        scene_tokens: Optional[Sequence[str]] = None,
        max_clips: Optional[int] = None,
        interpolate_hz: Optional[int] = None,
    ) -> None:
        self._data_root = Path(navsim_data_root)
        self._sensor_root = Path(sensor_root)
        from navsafe.benchmark.config import NUPLAN_MAPS_DEVKIT
        self._maps_root = Path(maps_root) if maps_root is not None else NUPLAN_MAPS_DEVKIT
        self._split = split
        self._log_names = set(log_names) if log_names else None
        self._scene_tokens = set(scene_tokens) if scene_tokens else None
        self._max_clips = max_clips
        # For the scenario-source arrow, upsample the native 2 Hz trajectory to
        # this rate (e.g. 10) so the sim's 10 Hz replay runs at real speed. Leave
        # None for the recon path (sensors only exist at 2 Hz).
        self._interpolate_hz = interpolate_hz

    def _scene_locations(self) -> "set[str]":
        """Distinct nuPlan map locations across the selected scenes (read the
        first frame of each log — cheap, avoids full-pkl scans per location)."""
        locs: "set[str]" = set()
        for pkl in self._log_pkls():
            try:
                frames = pickle.load(open(pkl, "rb"))
            except Exception:  # noqa: BLE001
                continue
            for fr in frames:
                if self._scene_tokens is None or str(fr.get("scene_token")) in self._scene_tokens:
                    loc = fr.get("map_location")
                    if loc:
                        locs.add(loc)
        return locs

    def get_map_parsers(self):  # type: ignore[override]
        """One NuplanMapParser per map location in the selected scenes.

        NavSim IS nuPlan, so its HD maps are the nuPlan gpkg maps
        (``/data/nuplan/maps``), keyed by ``map_location`` (e.g.
        ``us-nv-las-vegas-strip``). NuplanMapParser reads the gpkg directly — no
        nuPlan devkit dependency.
        """
        return [_NavsimMapParser(nuplan_maps_root=self._maps_root, location=loc)
                for loc in sorted(self._scene_locations())]

    def _log_pkls(self) -> List[Path]:
        pkls = sorted(self._data_root.glob("*.pkl"))
        if self._log_names is not None:
            pkls = [p for p in pkls if p.stem in self._log_names]
        return pkls

    def get_log_parsers(self) -> List[NavsimSceneLogParser]:  # type: ignore[override]
        parsers: List[NavsimSceneLogParser] = []
        for pkl in self._log_pkls():
            try:
                frames = pickle.load(open(pkl, "rb"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load NavSim log %s: %s", pkl.name, exc)
                continue

            scenes: "OrderedDict[str, List[dict]]" = OrderedDict()
            for fr in frames:
                scenes.setdefault(str(fr.get("scene_token", pkl.stem)), []).append(fr)

            for scene_token, scene_frames in scenes.items():
                if self._scene_tokens is not None and scene_token not in self._scene_tokens:
                    continue
                parsers.append(
                    NavsimSceneLogParser(
                        split=self._split,
                        scene_token=scene_token,
                        frames=scene_frames,
                        sensor_root=self._sensor_root,
                        source_log_name=pkl.stem,
                        interpolate_hz=self._interpolate_hz,
                    )
                )
                if self._max_clips is not None and len(parsers) >= self._max_clips:
                    return parsers
        return parsers


class NavhardNuplanParser(BaseDatasetParser):
    """py123d dataset parser that yields navhard scenes from raw nuPlan, windowed
    to each scene's [t0,t1] and renamed to the scene token — so ``py123d-conversion``
    writes an arrow whose ``log_name`` == the recon clip id (grpc scene id match).

    ``scenes``: list of ``[log_name, scene_token, t0_us, t1_us]``.
    """

    def __init__(self, *, nuplan_data_root, nuplan_sensor_root, nuplan_maps_root,
                 scenes, split: str = "nuplan_test") -> None:
        from py123d.parser.nuplan.nuplan_parser import NuplanParser
        self._scenes = [tuple(s) for s in scenes]
        logs = sorted({s[0] for s in self._scenes})
        self._np = NuplanParser(
            splits=[split],
            nuplan_data_root=str(nuplan_data_root),
            nuplan_maps_root=str(nuplan_maps_root),
            nuplan_sensor_root=str(nuplan_sensor_root),
            log_names=logs,
        )
        self._log_parsers_cache = None

    def get_log_parsers(self):
        from navsafe.gs3d_converter.log_ingestion import _WindowedNuplanLogParser
        if self._log_parsers_cache is None:
            by_log = {lp.get_log_metadata().log_name: lp for lp in self._np.get_log_parsers()}
            out = []
            for (log, tok, t0, t1) in self._scenes:
                inner = by_log.get(log)
                if inner is not None:
                    out.append(_WindowedNuplanLogParser(inner, str(tok), int(t0), int(t1)))
            self._log_parsers_cache = out
        return self._log_parsers_cache

    def get_map_parsers(self):
        locs = {lp.get_log_metadata().location for lp in self.get_log_parsers()}
        return [mp for mp in self._np.get_map_parsers()
                if mp.get_map_metadata().location in locs]
