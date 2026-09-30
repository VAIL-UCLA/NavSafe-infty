# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""NCoreBridge: prepare NCore V4 zarr.itar for NuRec ingestion.

Two paths:
  NCore clips  (scene.ncore_root is set)  → cheap symlink / copy.
  Other datasets (scene.log_parser set)   → convert via nvidia-ncore writer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, cast

import numpy as np

from navsafe.gs3d_converter.log_ingestion import IngestedScene

logger = logging.getLogger(__name__)

_LIDAR_SENSOR_ID = "lidar_top_360fov"
# Second lidar component: nuPlan's TOP sensor with a recovered beam model.
# Separate from the merged cloud above so initialisation is untouched.
_STRUCT_LIDAR_ID = "lidar_top_structured"
_RIG_FRAME_ID    = "rig"
_WORLD_FRAME_ID  = "world"

# Store all poses as float32. NCore's PosesComponent round-trips the stored
# dtype (components.py: `{"pose": ..., "dtype": str(pose.dtype)}` → read back as
# that dtype), and NRE's lidar point pipeline does an uncast `T_sensor_nre @
# xyz.T` where the points are float32. float64 poses make T_sensor_nre a double
# and torch raises "expected mat1 and mat2 to have the same dtype, double !=
# float". Real NCore datasets store float32 poses; matching that keeps it happy.
_POSE_DTYPE = np.float32


class NCoreBridge:
    """Ensure a clip's NCore zarr.itar files are available under ``work_dir``."""

    @staticmethod
    def prepare(scene: IngestedScene, work_dir: Path) -> Path:
        """Return the path to the clip directory ready for NuRec.

        For NCore-sourced clips (``scene.ncore_root`` is set) this is a cheap
        symlink — no re-encoding is needed.

        For other dataset sources (WOD, AV2, …) the raw sensor data is read from
        ``scene.log_parser`` and re-encoded as NCore V4 zarr.itar using
        ``nvidia-ncore`` (``pip install nvidia-ncore``).

        Returns
        -------
        Path
            ``work_dir/clips/<clip_id>/`` containing the three NCore files.
        """
        if scene.ncore_root is not None:
            return _symlink_ncore_clip(scene, work_dir)

        if scene.log_parser is None:
            raise RuntimeError(
                "NCoreBridge: scene.ncore_root is None and scene.log_parser is None. "
                "Cannot prepare NCore data."
            )

        return _convert_to_ncore(scene, work_dir)


# ── NCore symlink path ────────────────────────────────────────────────────────

def _symlink_ncore_clip(scene: IngestedScene, work_dir: Path) -> Path:
    ncore_root = scene.ncore_root
    assert ncore_root is not None  # caller (prepare) guards ncore_root is not None
    clip_dir = work_dir / "clips" / scene.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    src_dir = ncore_root / "clips" / scene.clip_id
    if not src_dir.exists():
        src_dir = ncore_root

    needed = [
        f"pai_{scene.clip_id}.json",
        f"pai_{scene.clip_id}.ncore4.zarr.itar",
        f"pai_{scene.clip_id}.ncore4-{_LIDAR_SENSOR_ID}.zarr.itar",
    ]
    for fname in needed:
        src = src_dir / fname
        dst = clip_dir / fname
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(f"NCoreBridge: expected NCore file not found: {src}")
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)

    return clip_dir


# ── py123d → NCore conversion path ───────────────────────────────────────────

def _convert_to_ncore(scene: IngestedScene, work_dir: Path) -> Path:
    """Re-encode a py123d log as NCore V4 zarr.itar files."""
    from ncore.data.v4 import (
        CameraSensorComponent,
        CuboidsComponent,
        IntrinsicsComponent,
        LidarSensorComponent,
        MasksComponent,
        PosesComponent,
        SequenceComponentGroupsReader,
        SequenceComponentGroupsWriter,
    )
    from ncore.impl.data.types import LabelSource, OpenCVPinholeCameraModelParameters, ShutterType
    from ncore.impl.data.v4.components import HalfClosedInterval
    from ncore.data import BBox3, CuboidTrackObservation

    from py123d.datatypes import BoxDetectionsSE3, EgoStateSE3
    from py123d.datatypes.sensors.base_camera import BaseCameraMetadata
    from py123d.datatypes.sensors.lidar import Lidar
    from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata
    from py123d.parser.base_dataset_parser import ParsedCamera, ParsedLidar
    from py123d.geometry.transform.transform_se3 import abs_to_rel_points_3d_array

    clip_dir = work_dir / "clips" / scene.clip_id
    manifest_path = clip_dir / f"pai_{scene.clip_id}.json"
    base_shard = clip_dir / f"pai_{scene.clip_id}.ncore4.zarr.itar"
    # Cache check: the manifest JSON is written LAST (after writer.finalize()
    # flushes every shard), so manifest + base shard both present means a prior
    # conversion completed — reuse it instead of re-encoding ~hundreds of MB.
    if manifest_path.exists() and base_shard.exists():
        logger.info("NCoreBridge: NCore data already exists at %s, skipping conversion", clip_dir)
        return clip_dir
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Structured lidar (real per-beam model + per-ray timing). WOD supplies one
    # from its tfrecord; nuPlan and other datasets fall back to flattened points.
    structured_lidar = None

    top_lidar_ext = _top_lidar_extrinsic(scene.lidar_meta)
    if structured_lidar is None and top_lidar_ext is not None:
        try:
            from navsafe.gs3d_converter.wod_structured_lidar import (
                WodStructuredLidar, find_wod_tfrecord)
            tfrecord = find_wod_tfrecord(scene.clip_id)
            if tfrecord is not None:
                structured_lidar = WodStructuredLidar(tfrecord)
                logger.info("NCoreBridge: structured lidar from %s "
                            "(%d sweeps, %d beams)", tfrecord,
                            len(structured_lidar._frames), structured_lidar.n_rows)
        except Exception as exc:  # noqa: BLE001 - structured path is best-effort
            logger.warning("NCoreBridge: structured WOD lidar unavailable (%s)", exc)
            structured_lidar = None

    # nuPlan carries no structured sweep, but its merged cloud keeps `ring` and
    # `lidar_info`, which is enough to recover the TOP sensor's beam model (see
    # nuplan_structured_lidar). That model is what NRE's get_lidar_data_batch
    # needs, so recovering it is the difference between lidar-supervised
    # training being possible on nuPlan and not. It goes into its OWN component:
    # the merged cloud below stays byte-identical, so Gaussian initialisation is
    # unchanged and only supervision gains a source.
    nuplan_lidar = None
    if (structured_lidar is None
            and not os.environ.get("NUREC_NO_LIDAR")
            and os.environ.get("NUPLAN_STRUCTURED_LIDAR", "1") != "0"):
        try:
            from navsafe.gs3d_converter.nuplan_structured_lidar import (
                NuplanStructuredLidar)
            probe = _first_lidar_modality(scene)
            if probe is not None:
                cand = NuplanStructuredLidar()
                if cand.calibrate(probe):
                    nuplan_lidar = cand
        except Exception as exc:  # noqa: BLE001 - best-effort, same as WOD above
            logger.warning("NCoreBridge: nuPlan structured lidar unavailable (%s)", exc)
            nuplan_lidar = None

    # Native-rate (~100 Hz) ego trajectory. A structured lidar needs it twice
    # over. Its rays run past the frame stamp, so the sequence interval widens by
    # a sweep and the stored dynamic poses must bracket the WIDER interval -- with
    # only the 10 Hz frame poses NRE dies at step 0 with
    #   RuntimeError: T_sensor_world_startend_allviews: data pointer is invalid
    # because it cannot build the sweep-start/end sensor-to-world pair. And
    # per-ray timestamps are only worth having if the poses they interpolate
    # against are fine enough to resolve them. Collected here, before the
    # interval is fixed, so the interval can depend on whether it succeeded.
    rig_poses: List[np.ndarray] = []
    rig_ts_us: List[int]        = []
    native_pose_trajectory = False
    async_iter = getattr(cast(Any, scene.log_parser), "iter_modalities_async", None)
    if ((structured_lidar is not None or nuplan_lidar is not None)
            and async_iter is not None):
        for modality in async_iter():
            if isinstance(modality, EgoStateSE3):
                # Keep rig->world in float64 here; it carries the absolute UTM
                # translation (~4.7e6 in nuPlan northing) whose precision under
                # float32 is ~0.5 m. Re-referencing to frame-0 and the final
                # float32 cast both happen at store time (see NCORE_REREF_FRAME0).
                rig_poses.append(
                    np.asarray(modality.imu_se3.transformation_matrix, dtype=np.float64)
                )
                rig_ts_us.append(int(modality.timestamp.time_us))
            elif rig_poses:
                break
        native_pose_trajectory = len(rig_poses) >= 2

    if nuplan_lidar is not None and not native_pose_trajectory:
        logger.warning("NCoreBridge: no native ego trajectory available; a "
                       "structured lidar would emit rays the stored poses cannot "
                       "bracket, so falling back to the flattened path")
        nuplan_lidar = None

    ts0 = int(scene.timestamps_us[0])
    # Half-open [ts0, ts1). Structured lidar rays extend beyond the frame start;
    # include the full final sweep and later interpolate rig poses at both exact
    # interval boundaries.
    lidar_tail_us = int(max(getattr(structured_lidar, "sweep_us", 1),
                            getattr(nuplan_lidar, "sweep_us", 1)))
    ts1 = int(scene.timestamps_us[-1]) + lidar_tail_us
    interval = HalfClosedInterval(start=ts0, stop=ts1)

    store_base = f"pai_{scene.clip_id}"
    writer = SequenceComponentGroupsWriter(
        output_dir_path=clip_dir,
        store_base_name=store_base,
        sequence_id=scene.clip_id,
        sequence_timestamp_interval_us=interval,
        generic_meta_data={"dataset": "py123d_converted"},
        store_type="itar",
    )

    # ── register component writers ────────────────────────────────────────────
    # The component INSTANCE NAME (2nd arg) is what NRE's SequenceLoaderV4 looks
    # up: open_component_readers() keys readers by instance name, and the loader
    # defaults poses/intrinsics/masks/cuboids group names to "default". Naming
    # these anything else (e.g. "rig_poses") makes nre-tools/training fail with
    # "PosesComponent group 'default' not found". group_name stays None so they
    # land in the main pai_<clip>.ncore4 store.
    poses_w  = writer.register_component_writer(PosesComponent.Writer,  "default")
    intr_w   = writer.register_component_writer(IntrinsicsComponent.Writer, "default")
    cuboid_w = writer.register_component_writer(CuboidsComponent.Writer, "default")
    # NRE's SequenceLoaderV4 (used by nre-tools ncore-aux-data) asserts a masks
    # component group "default" exists, even though ncore-aux-data is what fills
    # in real segmentation/ego masks. Register an empty one so the loader passes;
    # aux populates it. (Its __init__ creates the 'cameras' group → valid+loadable.)
    masks_w  = writer.register_component_writer(MasksComponent.Writer, "default")
    # Camera-only source (e.g. navsim arrow, whose lidar can't be arrow-encoded):
    # skip the lidar component entirely, else it's registered but never written
    # -> empty/1-D store -> every NRE reader IndexErrors. NUREC_NO_LIDAR set by
    # run_real2sim for the camera-only path.
    _no_lidar = bool(os.environ.get("NUREC_NO_LIDAR"))
    lidar_w = None if _no_lidar else writer.register_component_writer(
        LidarSensorComponent.Writer, _LIDAR_SENSOR_ID, group_name=_LIDAR_SENSOR_ID
    )
    # Second, structured component (nuPlan only). Registered ONLY once the beam
    # model has been fitted, because a registered-but-never-written component
    # leaves a 1-D store that every NRE reader IndexErrors on.
    lidar_struct_w = None if nuplan_lidar is None else writer.register_component_writer(
        LidarSensorComponent.Writer, _STRUCT_LIDAR_ID, group_name=_STRUCT_LIDAR_ID
    )
    # NCore lidar points are sensor-frame (direction/distance are rays from the
    # sensor) and lidar→rig places that sensor in the rig. The recovered nuPlan
    # model supplies its fitted TOP optical origin; calibrated datasets use the
    # source extrinsic. Identity remains only as an unstructured fallback.
    if structured_lidar is not None and hasattr(structured_lidar, "extrinsic"):
        lidar_pose_mat = np.asarray(structured_lidar.extrinsic, dtype=_POSE_DTYPE)
    elif top_lidar_ext is not None:
        lidar_pose_mat = np.asarray(top_lidar_ext.transformation_matrix, dtype=_POSE_DTYPE)
    else:
        lidar_pose_mat = np.eye(4, dtype=_POSE_DTYPE)
    if not _no_lidar:
        poses_w.store_static_pose(_LIDAR_SENSOR_ID, _RIG_FRAME_ID, lidar_pose_mat)
    if lidar_struct_w is not None:
        poses_w.store_static_pose(
            _STRUCT_LIDAR_ID, _RIG_FRAME_ID,
            np.asarray(nuplan_lidar.extrinsic, dtype=_POSE_DTYPE))

    n_structured_frames = 0
    n_nuplan_struct_frames = 0
    # Successive frame windows must have strictly increasing, unique end
    # stamps, so cap each window at the next stored frame's stamp.
    _seq_ts = [int(t) for t in scene.timestamps_us]
    _next_ts = {t: _seq_ts[i + 1] for i, t in enumerate(_seq_ts[:-1])}

    cam_writers: dict = {}  # cam_name → CameraSensorComponent.Writer

    # ── write static calibration (intrinsics + camera-to-rig poses) ───────────
    for cam_id, meta in scene.camera_metas.items():
        cam_name = _cam_id_to_name(cam_id)
        # Full sensor id (instance name == group name == shard suffix), mirroring
        # the lidar registration and NVIDIA's convention where camera_ids carry
        # the "camera_" prefix. Intrinsics/pose/frame stores all key off this so
        # NRE's dataset.camera_ids=[camera_<name>] resolves consistently.
        cam_sid = f"camera_{cam_name}"

        if isinstance(meta, PinholeCameraMetadata):
            intr = meta.intrinsics
            if intr is None:
                raise RuntimeError(
                    f"NCoreBridge: camera {cam_sid} has no pinhole intrinsics; "
                    "cannot write NCore calibration."
                )
            dist = meta.distortion if hasattr(meta, "distortion") else None
            k1, k2, p1, p2, k3 = (0.0, 0.0, 0.0, 0.0, 0.0)
            if dist is not None:
                k1 = getattr(dist, "k1", 0.0)
                k2 = getattr(dist, "k2", 0.0)
                p1 = getattr(dist, "p1", 0.0)
                p2 = getattr(dist, "p2", 0.0)
                k3 = getattr(dist, "k3", 0.0)
            cam_params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([meta.width, meta.height], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                external_distortion_parameters=None,
                principal_point=np.array([intr.cx, intr.cy], dtype=np.float32),
                focal_length=np.array([intr.fx, intr.fy], dtype=np.float32),
                radial_coeffs=np.array([k1, k2, k3, 0.0, 0.0, 0.0], dtype=np.float32),
                tangential_coeffs=np.array([p1, p2], dtype=np.float32),
                thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            )
            intr_w.store_camera_intrinsics(cam_sid, cam_params)

            # static pose: camera-frame → rig
            if hasattr(meta, "camera_to_imu_se3") and meta.camera_to_imu_se3 is not None:
                mat = np.asarray(meta.camera_to_imu_se3.transformation_matrix, dtype=_POSE_DTYPE)
                poses_w.store_static_pose(cam_sid, _RIG_FRAME_ID, mat)

        cam_writers[cam_name] = writer.register_component_writer(
            CameraSensorComponent.Writer, cam_sid, group_name=cam_sid
        )
        # Create an (empty) per-camera masks entry so nre-tools segmentation can
        # call get_mask_images().get("ego") -> None (full mask) instead of
        # KeyError on a missing camera group. aux overwrites with real masks.
        masks_w.store_camera_masks(cam_sid, {})

    # ── iterate frames ────────────────────────────────────────────────────────
    all_cuboids:  List             = []
    _raw_cuboids: List             = []  # raw box tuples; built into
    #                                      observations after the loop,
    #                                      once the frame-0 offset is known

    # Trim the native trajectory to the (now final) interval, interpolating exact
    # endpoints. Raising here would mean the collected poses do not span the
    # sequence, which cannot happen for a log whose frames define the interval.
    if native_pose_trajectory:
        rig_poses, rig_ts_us = _clip_pose_trajectory(
            rig_poses, rig_ts_us, ts0, ts1 - 1
        )

    # Frame-0 xy offset for the re-reference (NCore convention; see the store
    # block below). Computed here when a native ego trajectory is already
    # available; otherwise recomputed after the frame loop. Cuboid centroids are
    # deliberately NOT shifted inline: rig_poses[0] (hence the offset) is not yet
    # known for datasets without a native trajectory, so the boxes are built into
    # observations after the loop, once the offset is definitive. float64.
    _reref_offset = None
    if (os.environ.get("NCORE_REREF_FRAME0", "1") != "0") and rig_poses:
        _reref_offset = np.asarray(rig_poses[0], dtype=np.float64)[:2, 3].copy()

    # log_parser is an untyped py123d parser (IngestedScene types it as object).
    for frame_idx, sync in enumerate(cast(Any, scene.log_parser).iter_modalities_sync()):
        ts_us = sync.timestamp.time_us

        for mod in sync.modalities:
            if isinstance(mod, EgoStateSE3):
                if native_pose_trajectory:
                    continue
                mat = np.asarray(mod.imu_se3.transformation_matrix, dtype=np.float64)
                rig_poses.append(mat)
                rig_ts_us.append(ts_us)

            elif isinstance(mod, BoxDetectionsSE3):
                for det in mod.box_detections:
                    bb  = det.bounding_box_se3
                    c   = bb.center_se3
                    # Collect raw world-frame (absolute UTM) box data; the
                    # observations are built after the frame loop so their
                    # centroids can be shifted into the SAME frame-0 local frame
                    # as the rig poses (the offset is only definitive post-loop
                    # for datasets without a native ego trajectory).
                    _raw_cuboids.append((
                        str(det.attributes.track_token),
                        str(det.attributes.label),
                        ts_us,
                        float(c.x), float(c.y), float(c.z),
                        float(bb.length), float(bb.width), float(bb.height),
                        float(c.yaw),
                    ))

            elif isinstance(mod, ParsedCamera):
                # ParsedCamera.metadata is declared BaseModalityMetadata but is
                # always a camera metadata (its ctor only accepts those).
                cam_name = _cam_id_to_name(cast(BaseCameraMetadata, mod.metadata).camera_id)
                image_bytes = mod._byte_string
                if image_bytes is None and mod.has_jpeg_file_path:
                    # AV2 (and other on-disk datasets) hand back a file path
                    # instead of an in-memory bytestring; read it lazily here.
                    # has_jpeg_file_path implies both path parts are set.
                    image_bytes = (
                        Path(cast("str | Path", mod._dataset_root))
                        / cast("str | Path", mod._relative_path)
                    ).read_bytes()
                if cam_name in cam_writers and image_bytes is not None:
                    cam_writers[cam_name].store_frame(
                        image_binary_data=image_bytes,
                        image_format="JPEG",
                        frame_timestamps_us=np.array([ts_us, ts_us], dtype=np.uint64),
                        generic_data={},
                        generic_meta_data={},
                    )

            elif isinstance(mod, (ParsedLidar, Lidar)):
                if structured_lidar is not None:
                    rec = structured_lidar.decode(int(ts_us))
                    if rec is not None:
                        # NCore requires every stored timestamp inside the
                        # sequence interval [ts0, ts1); cap a sweep only when
                        # the next stored frame begins earlier.
                        _ft = int(rec["frame_timestamp_us"])
                        _cap = _next_ts.get(_ft)
                        _end = min(int(rec.get(
                                       "frame_end_timestamp_us",
                                       _ft + structured_lidar.sweep_us - 1)),
                                   (_cap - 1) if _cap else ts1 - 1,
                                   ts1 - 1)
                        lidar_w.store_frame(
                            direction=rec["direction"],
                            timestamp_us=np.minimum(
                                rec["timestamp_us"], np.uint64(_end)),
                            model_element=rec["model_element"],
                            distance_m=rec["distance_m"][None, :],
                            intensity=rec["intensity"][None, :],
                            frame_timestamps_us=np.array(
                                [rec["frame_timestamp_us"], _end],
                                dtype=np.uint64),
                            generic_data={},
                            generic_meta_data={},
                        )
                        n_structured_frames += 1
                        continue
                    logger.warning("NCoreBridge: no tfrecord sweep near ts=%d; "
                                   "flattened fallback for this frame", ts_us)
                xyz, dist_m, intensity = _load_lidar(mod)
                if xyz is not None and len(xyz):
                    # py123d returns ego-frame points; NCore expects sensor-frame
                    # direction/distance (paired with the TOP lidar→rig extrinsic
                    # registered above). Transform ego → TOP-lidar frame so the
                    # rays originate at the real sensor. Geometry is unchanged
                    # (NRE re-applies the extrinsic), but NRE's sensor-centric
                    # ground/normal math now sees the correct sensor origin.
                    if top_lidar_ext is not None:
                        xyz = abs_to_rel_points_3d_array(
                            top_lidar_ext, xyz.astype(np.float64)
                        ).astype(np.float32)
                    # NCore lidar writer wants unit-norm directions and (R, N)
                    # per-return arrays (R = number of returns, here 1).
                    norm = np.linalg.norm(xyz, axis=1).astype(np.float32)
                    safe = norm.clip(min=1e-6)
                    direction = (xyz / safe[:, None]).astype(np.float32)
                    per_ray_ts = np.full(len(xyz), ts_us, dtype=np.uint64)
                    lidar_w.store_frame(
                        direction=direction,
                        timestamp_us=per_ray_ts,
                        model_element=None,
                        distance_m=norm[None, :],                          # (1, N)
                        intensity=intensity.astype(np.float32)[None, :],   # (1, N)
                        frame_timestamps_us=np.array([ts_us, ts_us], dtype=np.uint64),
                        generic_data={},
                        generic_meta_data={},
                    )
                if lidar_struct_w is not None:
                    srec = nuplan_lidar.decode(mod, int(ts_us))
                    if srec is not None:
                        _cap = _next_ts.get(int(ts_us))
                        _end = min(int(srec["frame_end_timestamp_us"]),
                                   (_cap - 1) if _cap else ts1 - 1, ts1 - 1)
                        lidar_struct_w.store_frame(
                            direction=srec["direction"],
                            timestamp_us=np.minimum(
                                srec["timestamp_us"], np.uint64(_end)),
                            model_element=srec["model_element"],
                            distance_m=srec["distance_m"][None, :],
                            intensity=srec["intensity"][None, :],
                            frame_timestamps_us=np.array(
                                [int(ts_us), _end], dtype=np.uint64),
                            generic_data={},
                            generic_meta_data={},
                        )
                        n_nuplan_struct_frames += 1

    # ── re-reference to frame-0 (NCore convention) ────────────────────────────
    # nuPlan poses are absolute UTM (~3.3e5 easting, ~4.7e6 northing). Stored as
    # float32 that northing has a ~0.5 m ULP, so the rig->world translation gets
    # snapped to a 0.5 m grid — quantizing BOTH the training cameras (recon blur)
    # and, downstream, the eval camera (a visible per-frame sawtooth). Subtracting
    # the frame-0 xy in float64 BEFORE the float32 cast moves every coordinate to
    # ~0..few-hundred m, where float32 is sub-mm. The offset is recorded so the
    # eval render path can shift its (absolute) ego/actor xy into this same local
    # frame (nurec_grpc origin_offset). Gated so in-flight absolute-frame stores
    # are untouched; a re-referenced store is self-describing via the sidecar.
    # (_reref_offset was computed before the frame loop; cuboid centroids were
    # already shifted inline as they were built.)
    if _reref_offset is None and (os.environ.get("NCORE_REREF_FRAME0", "1") != "0") and rig_poses:
        # Fallback (no native lidar trajectory): offset not known before the
        # frame loop, so cuboids stayed absolute — but still re-reference the
        # rig poses so the recon/eval camera is in the local frame.
        _reref_offset = np.asarray(rig_poses[0], dtype=np.float64)[:2, 3].copy()
    if rig_poses:
        poses_arr = np.stack(rig_poses, axis=0).astype(np.float64)
        if _reref_offset is not None:
            poses_arr[:, 0, 3] -= _reref_offset[0]
            poses_arr[:, 1, 3] -= _reref_offset[1]
        poses_w.store_dynamic_pose(
            source_frame_id=_RIG_FRAME_ID,
            target_frame_id=_WORLD_FRAME_ID,
            poses=poses_arr.astype(_POSE_DTYPE),
            timestamps_us=np.array(rig_ts_us, dtype=np.uint64),
        )

    # Persist the frame-0 xy offset next to the store so the eval/serve path can
    # recover the local<->UTM shift (absolute UTM = local + offset).
    if _reref_offset is not None:
        try:
            (clip_dir / "nurec_origin_offset.json").write_text(json.dumps(
                {"offset_xy_utm": [float(_reref_offset[0]), float(_reref_offset[1])],
                 "frame": "rig_world_frame0", "note": "absolute_utm = local + offset"}))
        except Exception as _e:  # noqa: BLE001
            logger.warning("NCoreBridge: could not write origin_offset sidecar: %s", _e)

    # Build cuboid observations now that the frame-0 offset is definitive, so
    # their centroids land in the SAME re-referenced local frame as the rig
    # poses above. Building inline during the frame loop left them in absolute
    # UTM (~4.7e6 m from the local-frame scene) whenever the offset was not yet
    # known, which zeroed every dynamic/static track gaussian layer (all objects
    # baked into background). reref disabled -> offset None -> centroids stay absolute.
    _cub_ox = float(_reref_offset[0]) if _reref_offset is not None else 0.0
    _cub_oy = float(_reref_offset[1]) if _reref_offset is not None else 0.0
    for (_tid, _cls, _cts, _cx, _cy, _cz, _L, _W, _H, _yaw) in _raw_cuboids:
        all_cuboids.append(CuboidTrackObservation(
            track_id=_tid,
            class_id=_cls,
            timestamp_us=_cts,
            reference_frame_id=_WORLD_FRAME_ID,
            reference_frame_timestamp_us=_cts,
            bbox3=BBox3(
                centroid=(_cx - _cub_ox, _cy - _cub_oy, _cz),
                dim=(_L, _W, _H),
                rot=(0.0, 0.0, _yaw),
            ),
            source=LabelSource.GT_ANNOTATION,
        ))

    # Always store, even with zero observations (AV2 test split has no labels):
    # NRE train reads the "cuboids" zarr group unconditionally and raises
    # KeyError if the bridge never created it.
    cuboid_w.store_observations(all_cuboids)

    if structured_lidar is not None and n_structured_frames:
        # Registered AFTER the frame loop: n_columns is learned from the first
        # decoded sweep. The model + per-ray model_element is exactly what
        # NRE's get_lidar_data_batch requires for lidar-supervised training.
        intr_w.store_lidar_intrinsics(
            _LIDAR_SENSOR_ID, structured_lidar.spinning_params())
        logger.info("NCoreBridge: %d structured lidar sweeps + spinning model "
                    "registered", n_structured_frames)

    if lidar_struct_w is not None and n_nuplan_struct_frames:
        intr_w.store_lidar_intrinsics(
            _STRUCT_LIDAR_ID, nuplan_lidar.spinning_params())
        logger.info("NCoreBridge: %d nuPlan structured TOP sweeps as '%s' "
                    "(%d beams x %d columns, beam fit %.4f deg, deskew=%s)",
                    n_nuplan_struct_frames, _STRUCT_LIDAR_ID,
                    nuplan_lidar.n_rows, nuplan_lidar.n_columns,
                    nuplan_lidar.fit_residual_deg, nuplan_lidar.deskew)

    store_paths = writer.finalize()
    logger.info("NCoreBridge: wrote %d stores to %s", len(store_paths), clip_dir)

    # ── write JSON manifest (pai_<clip_id>.json) ──────────────────────────────
    reader = SequenceComponentGroupsReader([p for p in store_paths])
    seq_meta = reader.get_sequence_meta()
    manifest_path.write_text(json.dumps(seq_meta.to_dict(), indent=2))
    logger.info("NCoreBridge: manifest written to %s", manifest_path)

    return clip_dir


# ── lidar loading ─────────────────────────────────────────────────────────────

def _top_lidar_extrinsic(lidar_meta):
    """Return the TOP lidar→imu(rig) ``PoseSE3``, or None if unavailable.

    ``lidar_meta`` is the py123d ``LidarMergedMetadata`` (a Mapping[LidarID,
    LidarMetadata]) captured during ingestion; for single-lidar datasets it may
    be a plain ``LidarMetadata``. The extrinsic itself can be None when the
    source omits lidar calibration — callers fall back to identity.
    """
    if lidar_meta is None:
        return None
    try:
        from py123d.datatypes.sensors.lidar import LidarID, LidarMetadata
        if isinstance(lidar_meta, LidarMetadata):
            return lidar_meta.lidar_to_imu_se3
        return lidar_meta[LidarID.LIDAR_TOP].lidar_to_imu_se3
    except (KeyError, AttributeError, TypeError) as exc:
        logger.warning("NCoreBridge: no TOP lidar extrinsic (%s); using identity", exc)
        return None


def _load_pcd_xyz_intensity(path):
    """Parse a binary/ascii PCD (v0.7) → (xyz float32 (N,3), intensity float32 (N,)).

    Handles NavSim/OpenScene MergedPointCloud (fields ``x y z intensity
    lidar_info ring``). Intensity is normalized to [0, 1]; returns (None, None)
    on unsupported layouts.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    # Header is ASCII up to and including the DATA line.
    hdr_end = raw.find(b"DATA")
    line_end = raw.find(b"\n", hdr_end)
    header = raw[:line_end].decode("ascii", "replace").splitlines()
    fields, sizes, types, counts = [], [], [], []
    npts = 0
    data_kind = "binary"
    for line in header:
        toks = line.split()
        if not toks:
            continue
        key = toks[0].upper()
        if key == "FIELDS":
            fields = toks[1:]
        elif key == "SIZE":
            sizes = [int(x) for x in toks[1:]]
        elif key == "TYPE":
            types = toks[1:]
        elif key == "COUNT":
            counts = [int(x) for x in toks[1:]]
        elif key == "POINTS":
            npts = int(toks[1])
        elif key == "DATA":
            data_kind = toks[1].lower() if len(toks) > 1 else "binary"

    if not counts:
        counts = [1] * len(fields)
    # Only COUNT==1 fields are supported (true for NavSim MergedPointCloud).
    _npmap = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2",
              ("U", 4): "<u4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}

    if data_kind == "ascii":
        arr = np.array(raw[line_end + 1:].decode("ascii", "replace").split(),
                       dtype=np.float32).reshape(-1, sum(counts))
        idx, j = {}, 0
        for f, c in zip(fields, counts):
            idx[f] = j
            j += c
        xyz = arr[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
        inten = arr[:, idx["intensity"]].astype(np.float32) if "intensity" in idx \
            else np.ones(len(xyz), np.float32)
    else:
        dt = np.dtype([(f, _npmap[(t, s)]) for f, s, t in zip(fields, sizes, types)])
        rec = np.frombuffer(raw[line_end + 1: line_end + 1 + dt.itemsize * npts],
                            dtype=dt, count=npts)
        xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
        inten = (rec["intensity"].astype(np.float32) if "intensity" in fields
                 else np.ones(len(xyz), np.float32))

    if inten.size and float(inten.max()) > 1.0:
        inten = inten / 255.0
    return xyz, inten.clip(0.0, 1.0).astype(np.float32)


def _first_lidar_modality(scene):
    """The first lidar sweep of the log, or None.

    A beam model has to be fitted BEFORE the component writer is registered --
    registering one and then never writing it leaves a 1-D store that every NRE
    reader IndexErrors on. So peek at one sweep up front; the sync iterator is
    abandoned as soon as it yields one, which costs a single ~1.3 MB PCD read.
    """
    from py123d.datatypes.sensors.lidar import Lidar
    from py123d.parser.base_dataset_parser import ParsedLidar
    try:
        for sync in cast(Any, scene.log_parser).iter_modalities_sync():
            for mod in sync.modalities:
                if isinstance(mod, (ParsedLidar, Lidar)):
                    return mod
    except Exception as exc:  # noqa: BLE001 - probe is best-effort
        logger.warning("NCoreBridge: lidar probe failed (%s)", exc)
    return None


def _load_lidar(parsed_lidar):
    """Return (xyz, distance_m, intensity) from a ParsedLidar.

    Supports WOD (tfrecord range images), AV2 sensor (feather sweeps), and
    NavSim/OpenScene (binary PCD); returns (None, None, None) on failure or
    unknown formats. All loaders yield ego-frame points, matching the
    sensor-frame transform in the caller.
    """
    try:
        # In-memory py123d Lidar datatype (Arrow-log path): ego-frame xyz +
        # optional intensity are already decoded — no file to read.
        xyz_mem = getattr(parsed_lidar, "xyz", None)
        if xyz_mem is not None:
            xyz = np.asarray(xyz_mem, dtype=np.float32)
            dist_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
            raw_int = getattr(parsed_lidar, "intensity", None)
            if raw_int is None:
                intensity = np.ones(len(xyz), np.float32)
            else:
                intensity = np.asarray(raw_int, dtype=np.float32)
                if intensity.size and float(intensity.max()) > 1.0:
                    intensity = intensity / 255.0
                intensity = intensity.clip(0.0, 1.0).astype(np.float32)
            return xyz, dist_m, intensity

        ds_root  = getattr(parsed_lidar, "_dataset_root", None)
        rel_path = getattr(parsed_lidar, "_relative_path", None)
        iter_idx = getattr(parsed_lidar, "_iteration", None)

        if ds_root is None or rel_path is None:
            return None, None, None

        full_path = Path(ds_root) / rel_path

        if str(rel_path).endswith(".pcd"):
            # NavSim/OpenScene MergedPointCloud: binary PCD v0.7 with fields
            # (x,y,z float32, intensity/lidar_info/ring uint8), points already in
            # the ego frame (lidar2ego is identity) — return as-is.
            xyz, intensity = _load_pcd_xyz_intensity(full_path)
            if xyz is None or not len(xyz):
                return None, None, None
            dist_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
            return xyz.astype(np.float32), dist_m, intensity.astype(np.float32)

        if str(rel_path).endswith(".feather"):
            # AV2 sensor sweep: ego-frame xyz + uint8 intensity in one feather.
            from py123d.datatypes.sensors.lidar import LidarFeature
            from py123d.parser.av2.av2_sensor_io import (
                load_av2_sensor_point_cloud_data_from_path,
            )
            xyz, features = load_av2_sensor_point_cloud_data_from_path(full_path)
            dist_m  = np.linalg.norm(xyz, axis=1).astype(np.float32)
            raw_int = features.get(
                LidarFeature.INTENSITY.serialize(), np.ones(len(xyz), np.float32)
            ).astype(np.float32)
            intensity = (raw_int / 255.0).clip(0.0, 1.0).astype(np.float32)
            return xyz.astype(np.float32), dist_m, intensity

        if iter_idx is None:
            return None, None, None

        from py123d.parser.wod.wod_perception_sensor_io import (
            load_wod_perception_point_cloud_data_from_path,
        )
        xyz, features = load_wod_perception_point_cloud_data_from_path(
            full_path, iter_idx, keep_polar_features=True
        )
        dist_m    = features.get("RANGE", np.linalg.norm(xyz, axis=1).astype(np.float32))
        raw_int   = features.get("INTENSITY", np.ones(len(xyz), np.float32))
        intensity = (raw_int / 255.0).clip(0.0, 1.0).astype(np.float32)
        return xyz.astype(np.float32), dist_m.astype(np.float32), intensity

    except Exception as exc:
        logger.warning("NCoreBridge: failed to load lidar frame: %s", exc)
        return None, None, None


# ── helpers ───────────────────────────────────────────────────────────────────

def _clip_pose_trajectory(
    poses: List[np.ndarray],
    timestamps_us: List[int],
    start_us: int,
    end_us: int,
) -> tuple[List[np.ndarray], List[int]]:
    """Clip native poses and interpolate exact half-closed interval endpoints.

    Returns float64. Casting to ``_POSE_DTYPE`` here would be a silent
    catastrophe on nuPlan: at this point the poses still carry ABSOLUTE UTM
    (~4.69e6 northing), where float32's ULP is 0.5 m, and the frame-0
    re-reference that makes float32 safe has not happened yet. The store-time
    code subtracts the offset first and casts after. WOD never noticed because
    its coordinates are segment-local and small; nuPlan turned the ego track
    into a 0.5 m staircase (speed 0 -> 53 m/s between samples).
    """
    from scipy.spatial.transform import Rotation, Slerp

    order = np.argsort(np.asarray(timestamps_us, dtype=np.int64))
    stamps = np.asarray(timestamps_us, dtype=np.int64)[order]
    matrices = np.stack(poses, axis=0)[order]
    unique = np.r_[True, stamps[1:] != stamps[:-1]]
    stamps, matrices = stamps[unique], matrices[unique]
    if stamps[0] > start_us or stamps[-1] < end_us:
        raise ValueError(
            f"native pose trajectory [{stamps[0]}, {stamps[-1]}] does not "
            f"bracket sequence [{start_us}, {end_us}]"
        )

    def interpolate(timestamp_us: int) -> np.ndarray:
        exact = np.flatnonzero(stamps == timestamp_us)
        if exact.size:
            return np.asarray(matrices[int(exact[0])], dtype=np.float64)
        right = int(np.searchsorted(stamps, timestamp_us))
        left = right - 1
        alpha = float(timestamp_us - stamps[left]) / float(stamps[right] - stamps[left])
        result = np.eye(4, dtype=np.float64)
        result[:3, 3] = (
            (1.0 - alpha) * matrices[left, :3, 3]
            + alpha * matrices[right, :3, 3]
        )
        rotations = Rotation.from_matrix(matrices[[left, right], :3, :3])
        result[:3, :3] = Slerp(
            [float(stamps[left]), float(stamps[right])], rotations
        )([float(timestamp_us)]).as_matrix()[0]
        return result

    inside = (stamps > start_us) & (stamps < end_us)
    clipped_poses = [interpolate(start_us)]
    clipped_poses.extend(np.asarray(m, dtype=np.float64) for m in matrices[inside])
    clipped_poses.append(interpolate(end_us))
    clipped_stamps = [start_us, *stamps[inside].astype(int).tolist(), end_us]
    return clipped_poses, clipped_stamps


def _cam_id_to_name(cam_id) -> str:
    """Convert a py123d CameraID to a lowercase string safe for NCore component names."""
    name = str(cam_id)
    # Strip enum class prefix if present (e.g. "CameraID.PCAM_F0" → "pcam_f0")
    if "." in name:
        name = name.split(".")[-1]
    return name.lower().replace("-", "_")
