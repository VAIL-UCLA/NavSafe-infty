"""NuRec gRPC render backend.

Requests the policy's camera images from a NuRec render server
(``serve-grpc`` of the NVIDIA NRE image), which draws them from the scenario's
reconstructions.

Server
------
::

    serve-grpc --host 0.0.0.0 --enable-editing-actors \
        --enable-harmonizer --harmonizer-cache <writable-dir> \
        --artifact-glob '<dataset>/full_test/*/*.usdz'

``--enable-editing-actors`` is required: it lets the client move the
reconstructed actors to their simulated poses. The benchmark is rendered with
the harmonizer enabled; its weights are downloaded once into the cache
directory. Each ``.usdz`` is served as a scene named after the file.

The client needs ``grpcio`` and ``protobuf``; the protocol bindings are in
``navsafe._vendor.nurec_grpc`` and are imported on first use.

Scenario metadata read: ``nurec_grpc_host`` (else ``NUREC_GRPC_HOST``),
``nurec_grpc_port`` (else ``NUREC_GRPC_PORT``, 8080), ``nurec_grpc_scene_id``
and ``gaussian_splat_origin_offset``.

Camera pose
-----------
``camera_to_world = rig_to_world @ camera_to_rig``. The intrinsics and
``camera_to_rig`` come from the server; this module builds ``rig_to_world``
from the ego state (x forward, y left, z up), with two additions:

* *Attitude.* The simulated ego is planar, but the road is not. Pitch and
  roll are taken from the server's recorded rig trajectory; yaw is the
  simulator's, since the policy may steer differently from the log.
* *Anchor.* The scenario's ego position is its bounding-box centre, while the
  reconstruction is anchored at the rear-axle ego pose, about 1.45 m behind
  it along the vehicle axis. See ``_ego_anchor_lever``.

Handoff
-------
A 20 s scenario is four consecutive 5 s reconstructions.
``NUREC_GRPC_HANDOFF`` lists them with their origin offsets and time windows,
and the client switches scene as the ego advances along the logged path.

Actors
------
Every actor the server exposes is sent each frame at its simulated pose;
actors absent from the simulation are moved out of view.

*Inserted assets.* Tracks whose metadata has ``nurec_asset_id`` are registered
with the ``edit_assets`` RPC at setup and then posed like any other actor.
The id is a path to a 3D Gaussian ``.ply`` that the server can read. The
server expects y-up files and scales an asset by its object size, so metric
assets are handed over pre-divided by that factor (``_server_asset_path``).

*Gait.* An inserted asset is rigid. A track with ``nurec_pose_bank`` names a
directory of posed copies spanning one gait cycle; each is inserted as its own
server track, and one is shown per frame, chosen from the distance travelled
so that the feet do not slide. Each pose costs server memory once per window.

*Harvested replacements.* ``NUREC_GRPC_ASSET_REPLACE`` names a manifest of
harvested assets that replace the reconstruction's own actors
(``_replace_harvested_assets``).

``close()`` restores the scenes, undoing inserts and replacements.

Diagnostics
-----------
``NUREC_GRPC_CAM_LATERAL_M`` shifts the camera rig sideways in the ego frame
(positive to the left) without changing the simulation, to inspect the
reconstruction away from the recorded trajectory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from navsafe.engine.registry import register_renderer
from navsafe.render.base import SceneRenderer
from navsafe.render.ply_utils import scale_ply
from navsafe.render.usdz_utils import resolve_usdz_path

logger = logging.getLogger(__name__)


class _ActorEditingDisabledError(RuntimeError):
    """The renderer cannot keep traffic actors aligned with simulation."""


# NAVSIM camera name -> NuRec training camera logical_id.
_CAM_LOGICAL = {
    "CAM_F0": "camera_pcam_f0", "CAM_B0": "camera_pcam_b0",
    "CAM_L0": "camera_pcam_l0", "CAM_L1": "camera_pcam_l1",
    "CAM_L2": "camera_pcam_l2", "CAM_R0": "camera_pcam_r0",
    "CAM_R1": "camera_pcam_r1", "CAM_R2": "camera_pcam_r2",
}


def _run_id_from_meta(meta: dict) -> Optional[str]:
    """Parse the NuRec run-id (== gRPC scene_id) from the usdz/run-dir path."""
    p = resolve_usdz_path(meta) or meta.get("nurec_run_dir")
    if not p:
        return None
    parts = Path(str(p)).parts
    if "artifacts" in parts:
        i = parts.index("artifacts")
        if i >= 1:
            return parts[i - 1]
    if "output" in parts:
        i = parts.index("output")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None



#: Calibration fields copied from a CameraSpec by :func:`_write_camera_spec_dump`.
#: Scalars and sequences both appear across backends and spec versions, and a
#: given spec carries only some of them, so each is probed rather than assumed.
_SPEC_SCALAR_FIELDS = (
    "resolution_w", "resolution_h", "fx", "fy", "cx", "cy",
    "fov_x", "fov_y", "model", "camera_model",
)
_SPEC_SEQUENCE_FIELDS = ("intrinsics", "distortion", "distortion_coeffs", "k", "d")


def _jsonable(value: Any) -> Any:
    """``value`` as JSON: a float, a list of floats, or its string form."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hasattr(value, "__len__") and not isinstance(value, str):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _write_camera_spec_dump(path: str, *, spec: Any, rig: Any, step: int,
                            scene_id: Optional[str], request_hw: tuple,
                            cam2rig: Any, c2w: Any, rig2world: Any) -> bool:
    """Write one camera's full calibration to ``path``. True if it landed.

    Best effort by design: this is a diagnostic, and a render must not fail
    because a debug dump could not be written or a spec lacks a field.
    """
    record: Dict[str, Any] = {
        "rig": str(rig),
        "step": step,
        "scene_id": scene_id,
        "request_hw": list(request_hw),
        "cam2rig": np.asarray(cam2rig, dtype=float).tolist(),
        "c2w": np.asarray(c2w, dtype=float).tolist(),
        "rig2world": np.asarray(rig2world, dtype=float).tolist(),
    }
    for field in _SPEC_SCALAR_FIELDS + _SPEC_SEQUENCE_FIELDS:
        if hasattr(spec, field):
            record[field] = _jsonable(getattr(spec, field))
    # The repr is the backstop for anything the field list above does not name;
    # bounded because some specs embed their whole distortion table in it.
    record["spec_repr"] = str(spec)[:4000]
    try:
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
    except OSError as exc:
        logger.warning("nurec_grpc: could not write camera spec dump %r: %s", path, exc)
        return False
    logger.info("nurec_grpc: wrote camera spec dump %s", path)
    return True


@register_renderer("nurec_grpc")
class NuRecGrpcSceneRenderer(SceneRenderer):
    """Render each frame by calling a warm NuRec gRPC render server."""

    def __init__(self, timeout_s: Optional[float] = None) -> None:
        # The first request for a scene makes the server load its
        # reconstruction, which can take well over a minute when
        # several clients start at once. The deadline is therefore
        # configurable through NUREC_GRPC_TIMEOUT_S.
        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get("NUREC_GRPC_TIMEOUT_S", 300.0))
            except ValueError:
                logger.warning("NUREC_GRPC_TIMEOUT_S=%r is not a number — using 300 s",
                               os.environ.get("NUREC_GRPC_TIMEOUT_S"))
                timeout_s = 300.0
        self._timeout = float(timeout_s)
        self._camera_rig = self._camera_rig_mode()
        # grpc channel/stub and the proto bundle are dynamic objects loaded
        # lazily in _connect(); Any keeps mypy honest about that without
        # inventing stubs for generated grpc code.
        self._channel: Any = None
        self._stub: Any = None
        self._g: Any = None            # lazily-imported proto bundle
        self._scene_id: Optional[str] = None
        # Bare ndarray on purpose: initialised float32 here, but the handoff
        # and re-reference paths below overwrite it with a float64 array
        # (pre-existing; every consumer converts, so nothing depends on it).
        self._origin_offset: np.ndarray = np.zeros(3, np.float32)
        self._sd_ego_z = 0.0
        self._sd_ego_z_arr = None  # per-frame recorded ego z (road altitude)
        self._sd_ego_xy_arr = None  # per-frame recorded ego xy (same frame as runtime pos)
        # A 3DGS reconstruction has finite novel-view coverage. Closed-loop
        # policies can leave it even though the simulator itself remains
        # numerically valid; the result is the familiar crystal/splat mush.
        # The guard is render-only and reports whenever it constrains a pose.
        self._manifold_clamp_warned = False
        self._manifold_max_deviation_m = 0.0
        self._cam_intrinsics: Dict[str, Any] = {}    # logical_id -> CameraSpec
        self._cam_to_rig: Dict[str, np.ndarray] = {}  # logical_id -> 4x4 SE3
        self._template_fallback_warned: set[str] = set()
        self._renderable = False
        self._timestamps_us: List[int] = []
        self._cur_step = 0
        #: NUREC_GRPC_SPECDUMP writes one camera's calibration and then stops.
        self._specdumped = False
        # NUREC_GRPC_RENDER_STEPS: render only at these simulation
        # steps and reuse the previous frame otherwise. Useful when
        # only some frames are consumed but the episode must step at
        # the log rate. Only the render call is skipped; scene
        # selection and actor poses are still updated every step. The
        # value is a list of integers, or "@<path>" to read one from
        # a file.
        self._render_steps = self._render_step_filter()
        if self._render_steps is not None:
            logger.info("nurec_grpc: rendering only %d of the episode's steps "
                        "(NUREC_GRPC_RENDER_STEPS)", len(self._render_steps))
        self._last_images: Optional[Dict[str, np.ndarray]] = None
        # scene_id -> (timestamps_us, tilt 3x3 array) from the training rig
        # trajectory; see _recorded_tilt.
        self._tilt_cache: Dict[str, Any] = {}
        # centre->rig lever arm in the EGO frame; see _ego_anchor_lever.
        self._anchor_lever: Optional[np.ndarray] = None
        self._anchor_logged = False
        # Multi-recon handoff table, filled from NUREC_GRPC_HANDOFF once the
        # scenario resolves. Initialised here because the insert/restore paths
        # enumerate it to reach EVERY served subclip, and they can run before
        # (or without) a scenario ever being set.
        self._handoff: Optional[list] = None
        self._ho_cur = 0
        # Editable dynamic actors (Level A). The env feeds per-frame symbolic
        # agent poses via update_agents(); we forward them to the server as
        # RGBRenderRequest.dynamic_objects so actors are driven by the sim
        # (move/remove) instead of the artifact's baked sequence tracks.
        self._agent_states: List[Dict[str, Any]] = []
        # Track ids the server exposes for this scene, identical to
        # the scenario's track tokens. Only these actors can be
        # moved. A server with actor editing disabled is rejected
        # during setup.
        self._actor_editing_enabled = True  # legacy servers may lack capability discovery
        self._dyn_track_ids: set = set()
        # {scene_id: {track_id}} — a handoff enumerates one set per
        # window; _dyn_track_ids stays their union.
        self._dyn_track_ids_by_scene: dict = {}
        # {scene_id: {track_id: (size_x, size_y, size_z)}} — the AABB the
        # SERVER holds for each baked track. Kept because an asset replacement
        # is scaled to a box, and the server's own box is the only one that
        # cannot disagree with the reconstruction it is replacing geometry in.
        self._dyn_track_size_by_scene: dict = {}
        # Baked tracks whose gaussians were swapped for a harvested asset
        # (Level C). Like _inserted_asset_ids, non-empty => close() must
        # restore_model_parameters, or the swap leaks onto the next eval
        # sharing this warm server.
        self._replaced_asset_ids: set = set()
        # Tracks we inserted into the served scene via edit_assets (Level B:
        # scenario tracks carrying metadata.nurec_asset_id). Driven by OUR
        # insert config, not a get_dynamic_objects round-trip. Non-empty =>
        # close() must restore_model_parameters to undo the edit server-side.
        self._inserted_asset_ids: set = set()
        # Per-frame states of the tracks this renderer inserted
        # (position, heading, valid). They are the pose source when
        # the environment's agent states do not carry an inserted
        # track, as in pure log replay.
        self._injected_track_states: dict = {}
        # Injected tracks carry their true ground z in the scenario frame, but
        # agent_states clamps z to 0 — pin the ground z per id so short assets
        # (a cone) don't fall back to the ego IMU altitude (~1.7 m up).
        self._injected_ground_z: Dict[str, float] = {}
        # Injected tracks that carry a GAIT rather than one static asset:
        # {logical tid: {"ids": [<tid>#ph0, ...], "stride_m": float}}. The
        # phase ids are what the server knows; the logical tid is what
        # agent_states and the scenario keep using.
        self._pose_banks: Dict[str, dict] = {}
        # {logical tid: {"step", "xy", "travel"}} — the odometer the phase is
        # clocked off. Keyed on the sim step so the several camera renders of
        # one frame cannot advance the gait several times.
        self._bank_clock: Dict[str, dict] = {}
        # Half-height (m) per baked dynamic actor, from the server's object_size.
        # Baked actors are placed at their box CENTRE, so a road-level pose sinks
        # them by this much; _actor_to_world adds it back. Injected assets use
        # the base-origin PLY path instead (_injected_ground_z), so they are not
        # listed here.
        self._dyn_track_half_h: Dict[str, float] = {}
        # Semantic class strings seen on the scene's own tracks, with
        # counts. The server rejects an insert whose class it cannot
        # parse, so inserted assets use a class the scene is known to
        # accept. See _resolve_semantic_class().
        self._dyn_sem_counter: Dict[str, int] = {}
        # frame-0 anchors: the env feeds ego_state in a recentered LOCAL frame,
        # so we map eval-local -> NuRec world via the recorded sd frame-0 pose:
        #   nre = (eval_pos - eval_pos0) + sd_pos0 - origin_offset
        self._sd_pos0: Optional[np.ndarray] = None    # sd-world xy at frame 0
        self._eval_pos0: Optional[np.ndarray] = None  # eval-local xy at frame 0

    # ------------------------------------------------------------------
    def _import(self):
        if self._g is not None:
            return self._g
        import grpc  # noqa
        from navsafe._vendor.nurec_grpc import common_pb2 as cpb
        from navsafe._vendor.nurec_grpc import sensorsim_pb2 as sspb
        from navsafe._vendor.nurec_grpc.common_pb2 import Pose, Vec3, Quat
        from navsafe._vendor.nurec_grpc.sensorsim_pb2 import (
            AvailableCamerasRequest, AvailableDynamicObjectsRequest, DynamicObject,
            ImageFormat, PosePair, RGBRenderRequest)
        from navsafe._vendor.nurec_grpc.sensorsim_pb2_grpc import SensorsimServiceStub
        # Module refs kept so edit_assets-era messages (DynamicObjectTrack,
        # EditAssetsRequest, Trajectory, ...) resolve via _find_msg without
        # hard-failing the whole backend on an older stub vintage.
        self._g = dict(grpc=grpc, sspb=sspb, cpb=cpb,
                       Pose=Pose, Vec3=Vec3, Quat=Quat,
                       AvailableCamerasRequest=AvailableCamerasRequest,
                       AvailableDynamicObjectsRequest=AvailableDynamicObjectsRequest,
                       DynamicObject=DynamicObject,
                       ImageFormat=ImageFormat, PosePair=PosePair,
                       RGBRenderRequest=RGBRenderRequest, Stub=SensorsimServiceStub)
        return self._g

    def _find_msg(self, name: str):
        """Resolve a proto message class from the sensorsim/common stubs."""
        for key in ("sspb", "cpb"):
            cls = getattr(self._g[key], name, None)
            if cls is not None:
                return cls
        raise AttributeError(
            f"proto message {name!r} not found in vendored nre stubs — "
            "asset insertion needs an edit_assets-capable NRE proto vintage")

    def _read_actor_editing_capability(self) -> bool:
        """Honor an explicit server editing prohibition; retain legacy fallback."""
        try:
            response = self._stub.get_server_config(
                self._find_msg("Empty")(), timeout=self._timeout)
            value = dict(response.server_config).get("enable_editing_actors")
        except Exception as exc:  # older servers/protos do not expose this RPC
            logger.info("nurec_grpc: actor editing capability unavailable: %s", exc)
            return True
        if value is None:
            return True
        normalized = str(value).strip().lower()
        if normalized in {"false", "0", "no", "off"}:
            logger.info("nurec_grpc: server disables actor editing")
            return False
        if normalized in {"true", "1", "yes", "on"}:
            return True
        raise ValueError(f"Invalid server enable_editing_actors value: {value!r}")

    def _check_cache_asset_retention(self, scenario_data: dict) -> None:
        """Refuse configured count eviction when setup needs mutable assets.

        The launcher must forward NAVSAFE_NUREC_CACHE_SIZE to the server's
        --cache-size. This checks that declared configuration, not server state
        or OOM eviction. Baked actors and their per-frame poses do not require
        edit_assets; injected geometry and harvested replacements do.
        """
        raw = os.environ.get("NAVSAFE_NUREC_CACHE_SIZE")
        evidence = {
            "status": "not_checked", "configured_cache_size": None,
            "unique_scene_ids": None, "configured_count_eviction": None,
            "inserted_asset_track_ids": None,
            "replacement_requested": bool(os.environ.get("NUREC_GRPC_ASSET_REPLACE")),
            "scope": "Declared cache count only; server settings and OOM eviction are not verified",
        }
        self._cache_retention_evidence = evidence
        if raw is None:
            evidence["reason"] = "cache size was not explicitly declared"
            logger.info("nurec_grpc: asset cache check %s", evidence)
            return
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) <= 0:
            raise ValueError("NAVSAFE_NUREC_CACHE_SIZE must be a positive integer")
        cache_size = int(raw)
        scene_ids = [row[0] for row in self._handoff] if self._handoff else [self._scene_id]
        if not scene_ids or any(not isinstance(sid, str) or not sid for sid in scene_ids):
            raise RuntimeError("nurec_grpc: cache retention requires resolved scene IDs")
        scene_ids = list(dict.fromkeys(scene_ids))
        evidence.update(configured_cache_size=cache_size, unique_scene_ids=scene_ids,
                        configured_count_eviction=cache_size < len(scene_ids))
        if cache_size >= len(scene_ids):
            evidence["status"] = "no_configured_count_eviction"
            logger.info("nurec_grpc: asset cache check %s", evidence)
            return
        tracks = scenario_data.get("tracks") if isinstance(scenario_data, Mapping) else None
        if not isinstance(tracks, Mapping):
            raise RuntimeError("nurec_grpc: undersized cache requires known scenario tracks")
        inserted = []
        for track_id, track in tracks.items():
            if not isinstance(track, Mapping):
                raise RuntimeError("nurec_grpc: undersized cache requires mapping tracks")
            metadata = track.get("metadata")
            # Missing/None metadata has the same empty meaning as the insertion
            # path. Other malformed metadata cannot establish absence of edits.
            if metadata is None:
                continue
            if not isinstance(metadata, Mapping):
                raise RuntimeError("nurec_grpc: undersized cache requires valid track metadata")
            if metadata.get("nurec_asset_id"):
                inserted.append(str(track_id))
        evidence["inserted_asset_track_ids"] = inserted
        if inserted or evidence["replacement_requested"]:
            evidence["status"] = "refused_mutable_asset_eviction"
            raise RuntimeError(
                "nurec_grpc: configured cache is smaller than the unique scene count "
                "and would permit eviction of requested edit_assets geometry; "
                "increase NAVSAFE_NUREC_CACHE_SIZE to at least " + str(len(scene_ids)))
        evidence["status"] = "no_mutable_asset_edits_requested"
        logger.info("nurec_grpc: asset cache check %s", evidence)

    def setup(self, scenario_data: dict, env: Any = None) -> None:
        self._renderable = False
        self._cache_retention_evidence = {
            "status": "not_checked", "reason": "setup scene is unresolved"}
        meta = scenario_data.get("metadata", {}) if scenario_data else {}
        self._scene_id = meta.get("nurec_grpc_scene_id") or _run_id_from_meta(meta)
        if not self._scene_id:
            logger.warning("nurec_grpc: no scene_id resolved; rendering black.")
            return
        # --- handoff: multi-recon scene switching, see _handoff_clock for how
        # the active window is chosen (position-based by default) ---
        self._handoff = None
        _ho = os.environ.get("NUREC_GRPC_HANDOFF")
        if _ho:
            import json as _hj
            # heterogeneous record rows: [scene_id, offset_xyz, traj|None, window|None]
            hs: list[list[Any]] = []
            for _part in _ho.split(";"):
                if not _part.strip():
                    continue
                _f = [x.strip() for x in _part.split(",")]
                _sid, _oj = _f[0], _f[1]
                _tw = (int(_f[2]), int(_f[3])) if len(_f) >= 4 else None
                _off = _hj.loads(open(_oj).read()).get("offset_xy_utm")
                hs.append([_sid,
                           np.asarray([float(_off[0]), float(_off[1]), 0.0], np.float64),
                           None, _tw])
            # handoff offsets relative to arrow frame-0: the scenario ego is
            # recentered (arrow-local), so shift each recon by offset_i minus the
            # arrow origin (scenario_origin_xy; fallback = first scene's offset).
            if hs:
                _sox = meta.get('scenario_origin_xy')
                if _sox and len(_sox) >= 2:
                    _ref = np.asarray([float(_sox[0]), float(_sox[1]), 0.0], np.float64)
                else:
                    _ref = hs[0][1].copy()
                for _h in hs:
                    _h[1] = _h[1] - _ref
            if hs:
                self._handoff = hs
                self._scene_id = hs[0][0]  # discover cameras/anchor from a served scene
                self._ho_cur = 0
        # Checked outside the fallback handlers below: an undersized
        # server cache must be reported, not silently drop edited
        # geometry.
        self._check_cache_asset_retention(scenario_data)
        self._origin_offset = np.asarray(
            meta.get("gaussian_splat_origin_offset", [0.0, 0.0, 0.0]), np.float32)
        self._sd_ego_z = float(self._read_sd_ego_z(scenario_data, meta))
        self._sd_ego_z_arr, self._sd_ego_xy_arr = self._read_sd_ego_track(
            scenario_data, meta)
        # The env recenters the track XY to a local frame (ego frame-0 == origin),
        # so the scenario_data tracks give local (0,0), NOT sd-world. The true
        # sd-world ego start is -origin_offset (the r2s pipeline sets the offset
        # to recenter the reconstruction on the ego start). Use that as the map
        # anchor. (Z is preserved in the tracks, handled by _read_sd_ego_z.)
        self._sd_pos0 = -self._origin_offset[:2].astype(np.float64)
        # Ego frame-0 xy in the scenario's OWN frame: (0,0) for the r2s pkl
        # (pre-recentered) but the ego WORLD start for native py123d arrows
        # (world-frame). Subtracting it recenters both sources to a common
        # local frame before the -origin_offset map, so the camera/actors
        # track motion instead of being thrown ~|origin_offset| off-scene.
        _ep0 = self._read_sd_pos0(scenario_data, meta)
        self._eval_pos0 = _ep0 if _ep0 is not None else np.zeros(2, np.float64)
        # A reconstruction converted with NCORE_REREF_FRAME0 lives in
        # a local frame whose origin is the first rig position.
        # Runtime positions are absolute, so the offset from the
        # sidecar file is subtracted, in float64. Reconstructions in
        # the absolute frame have no sidecar.
        _off_file = os.environ.get("NUREC_GRPC_ORIGIN_OFFSET_FILE")
        if _off_file and os.path.exists(_off_file):
            try:
                import json as _json
                _off = _json.loads(open(_off_file).read()).get("offset_xy_utm")
                if _off and len(_off) >= 2:
                    self._origin_offset = np.asarray(
                        [float(_off[0]), float(_off[1]), 0.0], np.float64)
                    self._eval_pos0 = np.zeros(2, np.float64)
                    self._sd_pos0 = -self._origin_offset[:2].astype(np.float64)
                    logger.info("nurec_grpc: re-referenced recon; origin_offset="
                                "[%.3f, %.3f] (local = utm - offset)",
                                self._origin_offset[0], self._origin_offset[1])
            except Exception as _e:  # noqa: BLE001
                logger.warning("nurec_grpc: origin_offset file %s unreadable: %s",
                               _off_file, _e)
        if self._handoff:
            self._origin_offset = self._handoff[0][1].copy()
            self._eval_pos0 = np.zeros(2, np.float64)
        t_start = int(meta.get("real2sim_start_timestamp_us", 0) or 0)
        # frame timestamps live in metadata["ts"] (relative seconds), NOT at the
        # scenario_data top level. Without them the request sends frame_us=0,
        # outside the scene's timestamp range -> the server renders empty.
        ts = meta.get("nurec_frame_timestamps")
        if ts is None:
            ts = meta.get("ts")
        if isinstance(ts, dict):
            ts = [ts[k] for k in sorted(ts)]
        if ts is not None:
            self._timestamps_us = [t_start + int(round(float(t) * 1e6)) for t in np.asarray(ts).reshape(-1)]
        logger.info("nurec_grpc timestamps: t_start=%s n_frames=%d first=%s",
                       t_start, len(self._timestamps_us), self._timestamps_us[:1])
        host = meta.get("nurec_grpc_host") or os.environ.get("NUREC_GRPC_HOST", "nurec-grpc")
        # NB: a k8s Service named "nurec-grpc" auto-injects NUREC_GRPC_PORT as
        # "tcp://<ip>:<port>", so parse tolerantly (take the trailing port).
        raw_port = meta.get("nurec_grpc_port", os.environ.get("NUREC_GRPC_PORT", 8080))
        port = int(str(raw_port).rsplit(":", 1)[-1])
        # Keep only the small parsed XY arrays. Group each scene's metadata
        # requests so a bounded server cache need not reload every GPU model
        # separately for reset, actor discovery, and trajectory discovery.
        trajectory_xy_by_scene: dict[str, np.ndarray | None] = {}
        anchor_trajectory_xy = None
        anchor_requested = False
        try:
            g = self._import()
            opts = [("grpc.max_receive_message_length", 64 << 20),
                    ("grpc.max_send_message_length", 64 << 20)]
            self._channel = g["grpc"].insecure_channel(f"{host}:{port}", options=opts)
            self._stub = g["Stub"](self._channel)
            self._actor_editing_enabled = self._read_actor_editing_capability()
            if not self._actor_editing_enabled:
                raise _ActorEditingDisabledError(
                    "NuRec server has enable_editing_actors=False. Evaluation "
                    "requires simulation-driven actor poses; baked trajectories "
                    "would disagree with simulation, BEV and collision states. "
                    "Start the render server with --enable-editing-actors."
                )
            avail = self._stub.get_available_cameras(
                g["AvailableCamerasRequest"](scene_id=self._scene_id), timeout=self._timeout)
            for c in avail.available_cameras:
                self._cam_intrinsics[c.logical_id] = c.intrinsics
                self._cam_to_rig[c.logical_id] = self._pose_to_se3(c.rig_to_camera)
            logger.info("nurec_grpc: scene=%s camera_rig=%s cameras=%s",
                        self._scene_id, self._camera_rig,
                        list(self._cam_intrinsics))
            self._renderable = len(self._cam_intrinsics) > 0
            # List the actors the server can move. With a
            # handoff, each window is its own scene holding only
            # the actors present during its seconds, so ids are
            # collected per scene; sending one scene's ids to
            # another is an error.
            scene_ids = ([sid for sid, _off, _t0, _t1 in self._handoff]
                         if self._handoff else [self._scene_id])
            # Restore each scene to its reconstructed state
            # first. Asset edits live in the server, so edits
            # left by an earlier run that did not close cleanly
            # would otherwise still be applied and rendered.
            self._dyn_track_ids_by_scene = {}
            for sid in scene_ids:
                self._reset_server_scenes([sid])
                try:
                    dyn = self._stub.get_dynamic_objects(
                        g["AvailableDynamicObjectsRequest"](scene_id=sid),
                        timeout=self._timeout).dynamic_objects
                except Exception as exc:  # pragma: no cover — live only
                    logger.info("nurec_grpc: get_dynamic_objects unavailable for "
                                "scene %s (%s); its actors stay baked.", sid, exc)
                    self._dyn_track_ids_by_scene[sid] = set()
                else:
                    ids = {(getattr(o, "track_id", None) or getattr(o, "id", None))
                           for o in dyn}
                    ids.discard(None)
                    self._dyn_track_ids_by_scene[sid] = {str(i) for i in ids}
                    sizes: Dict[str, tuple] = {}
                    for o in dyn:
                        tid = getattr(o, "track_id", None) or getattr(o, "id", None)
                        sc = str(getattr(o, "semantic_class", "") or "")
                        if sc:
                            self._dyn_sem_counter[sc] = self._dyn_sem_counter.get(sc, 0) + 1
                        osz = getattr(o, "object_size", None)
                        sz = float(getattr(osz, "size_z", 0.0) or 0.0) if osz is not None else 0.0
                        if tid is not None and sz > 0.0:
                            self._dyn_track_half_h[str(tid)] = sz / 2.0
                        if tid is not None and osz is not None:
                            sizes[str(tid)] = (float(getattr(osz, "size_x", 0.0) or 0.0),
                                               float(getattr(osz, "size_y", 0.0) or 0.0),
                                               sz)
                    self._dyn_track_size_by_scene[sid] = sizes
                # A failed actor query must still discover this trajectory.
                # Keep the existing first-scene retry semantics: if its anchor
                # request fails, handoff discovery makes one separate attempt.
                if self._renderable and sid == self._scene_id and not anchor_requested:
                    anchor_requested = True
                    try:
                        anchor_trajectory_xy = self._server_trajectory_xy(sid)
                        trajectory_xy_by_scene[sid] = anchor_trajectory_xy
                    except Exception as exc:  # pragma: no cover — live RPC
                        logger.info("nurec_grpc: trajectory anchor request unavailable "
                                    "(%s).", exc)
                if self._handoff and self._renderable and sid not in trajectory_xy_by_scene:
                    try:
                        trajectory_xy_by_scene[sid] = self._server_trajectory_xy(sid)
                        if sid == self._scene_id and anchor_trajectory_xy is None:
                            # A successful retry supplies authoritative anchor
                            # evidence too; never skip a known mismatch merely
                            # because the first request was unavailable.
                            anchor_trajectory_xy = trajectory_xy_by_scene[sid]
                    except Exception as exc:  # pragma: no cover — live RPC
                        logger.warning("nurec_grpc handoff: no trajectory for %s (%s)", sid, exc)
                        trajectory_xy_by_scene[sid] = None
            # Kept as the union so callers that only ask "is anything
            # controllable here" behave as before.
            self._dyn_track_ids = set().union(
                *self._dyn_track_ids_by_scene.values()) if self._dyn_track_ids_by_scene else set()
            logger.info("nurec_grpc: %d controllable dynamic actors exposed by "
                        "server across %d scene(s): %s",
                        len(self._dyn_track_ids), len(self._dyn_track_ids_by_scene),
                        {s: len(v) for s, v in self._dyn_track_ids_by_scene.items()})
        except _ActorEditingDisabledError:
            # A semantic mismatch must never become a scored episode, even if
            # the caller permits black frames with NUREC_GRPC_STRICT=0.
            self._renderable = False
            raise
        except Exception as exc:  # pragma: no cover — needs live server + stubs
            logger.warning("nurec_grpc: setup failed (%s:%s scene=%s): %s",
                           host, port, self._scene_id, exc)
            self._renderable = False
            if os.environ.get("NUREC_GRPC_STRICT", "1").lower() not in {
                    "0", "false", "no", "off"}:
                raise RuntimeError(
                    f"nurec_grpc setup failed for scene {self._scene_id}: "
                    f"{exc}. Frames would be BLACK. Set "
                    "NUREC_GRPC_STRICT=0 to proceed with black frames anyway."
                ) from exc
        # Cross-check the frame of reference against the server's own
        # rig trajectory. Poses sent in the wrong frame produce valid
        # images from a camera far from the scene, which the black-
        # frame check cannot detect.
        if self._renderable and anchor_trajectory_xy is not None:
            self._validate_frame_anchor(anchor_trajectory_xy)
        elif self._renderable:
            logger.info("nurec_grpc: trajectory anchor check unavailable; skipping.")
        if self._handoff and self._renderable:
            for _hsc in self._handoff:
                _hsc[2] = trajectory_xy_by_scene.get(_hsc[0])
            logger.info("nurec_grpc handoff: %d scenes armed (%s)",
                        len(self._handoff), ", ".join(h[0] for h in self._handoff))
        # Level B: register scenario tracks that carry a NuRec asset reference
        # (metadata.nurec_asset_id, e.g. injected cones) into the served scene.
        # Failure here must not kill rendering — the eval still runs, with the
        # injected assets scored but missing from the frames (logged loudly).
        if self._renderable:
            try:
                self._insert_injected_assets(scenario_data)
            except Exception as exc:  # pragma: no cover — live only
                logger.warning(
                    "nurec_grpc: edit_assets insertion failed — injected assets "
                    "will be MISSING from the render (sim state keeps them): %s", exc)
        # Replace logged actors with harvested assets. A failure here
        # fails the run: rendering the original actors after a
        # replacement was requested would be indistinguishable in the
        # metrics.
        if self._renderable:
            from navsafe.render.harvest_takeover import required_harvest_tracks
            self._required_harvest_tracks = required_harvest_tracks(scenario_data)
            self._replace_harvested_assets()

    # ------------------------------------------------------------------
    @staticmethod
    def _min_traj_dist_m(nre0_xy: np.ndarray, traj_xy: np.ndarray) -> float:
        """Distance (m) from a computed frame-0 xy to the nearest baked rig
        trajectory point. Min over the whole trajectory, not point 0: a
        scenario cut from mid-clip still starts ON the trajectory, just not at
        its first pose."""
        d = np.asarray(traj_xy, np.float64) - np.asarray(nre0_xy, np.float64)
        return float(np.sqrt(np.einsum("ij,ij->i", d, d).min()))

    def _server_trajectory_xy(self, scene_id: str) -> np.ndarray:
        """Read the renderer's authoritative trajectory as a small CPU array."""
        req = self._find_msg("AvailableTrajectoriesRequest")(scene_id=scene_id)
        resp = self._stub.get_available_trajectories(req, timeout=self._timeout)
        return np.asarray(
            [(p.pose.vec.x, p.pose.vec.y)
             for t in resp.available_trajectories
             for p in t.trajectory.poses], np.float64)

    def _validate_frame_anchor(self, trajectory_xy: np.ndarray | None = None) -> None:
        """Fail closed when the computed frame-0 rig pose is off-trajectory.

        Renders from a mis-anchored frame (e.g. a recentered scenario mapped
        onto an absolute-frame reconstruction with a zero origin offset) are
        valid non-black images of splat mush — every downstream consumer
        (policy input, LLM prompt frames, eval artifacts) accepts them
        silently. Compare the frame-0 camera-rig xy this renderer would use
        against the server's baked rig trajectory; beyond
        ``NUREC_GRPC_ANCHOR_TOL_M`` (default 10 m) raise under
        ``NUREC_GRPC_STRICT`` (else warn). Skips quietly when the server
        does not expose trajectories (older NRE vintages).
        """
        try:
            traj_xy = (self._server_trajectory_xy(self._scene_id)
                       if trajectory_xy is None else trajectory_xy)
        except Exception as exc:  # pragma: no cover — live only
            logger.info("nurec_grpc: trajectory anchor check unavailable "
                        "(%s); skipping.", exc)
            return
        if traj_xy.size == 0:
            logger.info("nurec_grpc: server exposes no trajectory poses; "
                        "anchor check skipped.")
            return
        # Frame-0 rig xy exactly as _ego_rig_to_world would compute it.
        e0 = self._eval_pos0 if self._eval_pos0 is not None else np.zeros(2)
        p0 = (self._sd_ego_xy_arr[0]
              if self._sd_ego_xy_arr is not None and len(self._sd_ego_xy_arr)
              else np.asarray(e0, np.float64))
        nre0 = (np.asarray(p0, np.float64) - np.asarray(e0, np.float64)
                - np.asarray(self._origin_offset[:2], np.float64))
        dist = self._min_traj_dist_m(nre0, traj_xy)
        tol = float(os.environ.get("NUREC_GRPC_ANCHOR_TOL_M", "10"))
        if dist <= tol:
            logger.info("nurec_grpc: frame anchor OK (%.2f m from baked "
                        "trajectory, tol %.1f m)", dist, tol)
            return
        msg = (f"nurec_grpc frame anchor MISMATCH for scene {self._scene_id}: "
               f"computed frame-0 rig xy {nre0.tolist()} is {dist:.1f} m from "
               f"the server's baked rig trajectory (tolerance {tol:.1f} m). "
               "The scenario and reconstruction disagree on the coordinate "
               "frame (origin offset wrong or missing) — renders would come "
               "from off-scene and look like smeared splat mush. Check "
               "metadata gaussian_splat_origin_offset / scenario_origin_xy / "
               "the ncore re-reference sidecar. Set NUREC_GRPC_STRICT=0 to "
               "render anyway.")
        strict = os.environ.get("NUREC_GRPC_STRICT", "1").lower() not in {
            "0", "false", "no", "off"}
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    def _handoff_clock(self, t_now: int, ego_state: dict) -> int:
        """The timestamp the handoff pick runs on. POSITION-based by default.

        The sim clock lies about coverage: a policy that runs late (or early)
        relative to the log crosses each recon's boundary at a different time
        than the log did, and a time pick then renders it from a recon that
        never baked the road it is on. What actually decides coverage is
        WHERE the ego is — so project the live ego xy onto the recorded
        (log-replay) ego track, take the nearest recorded frame, and use THAT
        frame's timestamp as the pick clock. The window table stays the
        single source of clip ownership; only the clock changes.

        ``NUREC_GRPC_HANDOFF_MODE=time`` restores the wall-clock pick;
        position mode also falls back to it when the scenario carries no ego
        track or timestamps to match against.
        """
        mode = os.environ.get("NUREC_GRPC_HANDOFF_MODE", "position").strip().lower()
        # An empty ego track is not None, and argmin over it raises rather than
        # falling back — same for an ego_state that carries no usable xy.
        if (mode != "position" or self._sd_ego_xy_arr is None
                or not len(self._sd_ego_xy_arr) or not self._timestamps_us):
            return t_now
        pos = np.asarray(
            ego_state.get("position", (0.0, 0.0)), np.float64).reshape(-1)
        if pos.shape[0] < 2:
            return t_now
        d = np.linalg.norm(
            np.asarray(self._sd_ego_xy_arr, np.float64)[:, :2] - pos[None, :2],
            axis=1)
        k = int(np.argmin(d))
        k = min(k, len(self._timestamps_us) - 1)
        return int(self._timestamps_us[k])

    def _render_pose_on_manifold(self, ego_state: dict) -> dict:
        """Constrain only the camera pose to the reconstruction's coverage.

        NuRec was trained from cameras on the recorded ego trajectory. Beyond
        a few metres of that trajectory there is no observation support, so
        asking it to render the physical closed-loop pose produces crystal
        structures rather than a meaningful image. The simulator, metrics,
        collision checks and policy state keep the real pose; only the render
        request is capped to a configurable translation/yaw envelope around
        the nearest recorded camera pose.

        ``NUREC_GRPC_MAX_NOVEL_VIEW_M=0`` disables the guard. The default 5 m
        is based on the failing 20fffc4 run: imagery remained coherent through
        about 5 m, became visibly unstable past 10 m, and was fully crystalline
        at 40-100 m.
        """
        track = getattr(self, "_sd_ego_xy_arr", None)
        try:
            limit = float(os.environ.get(
                "NUREC_GRPC_MAX_NOVEL_VIEW_M", "5"))
            yaw_limit = math.radians(float(os.environ.get(
                "NUREC_GRPC_MAX_NOVEL_YAW_DEG", "30")))
        except ValueError:
            logger.warning("invalid NuRec novel-view limit; using 5 m / 30 deg")
            limit, yaw_limit = 5.0, math.radians(30.0)
        if limit <= 0.0 or track is None or not len(track):
            return ego_state
        pos = np.asarray(
            ego_state.get("position", (0.0, 0.0)), np.float64).reshape(-1)
        xy = np.asarray(track, np.float64)
        if pos.size < 2 or xy.ndim != 2 or xy.shape[1] < 2:
            return ego_state
        delta = pos[:2][None, :] - xy[:, :2]
        distances = np.linalg.norm(delta, axis=1)
        k = int(np.argmin(distances))
        deviation = float(distances[k])
        self._manifold_max_deviation_m = max(
            float(getattr(self, "_manifold_max_deviation_m", 0.0)), deviation)
        if not np.isfinite(deviation) or deviation <= limit:
            return ego_state

        # Cap the magnitude of the deviation while keeping its
        # direction, so the camera stays within the measured envelope
        # without snapping to the logged pose.
        render_pos = pos.copy()
        render_pos[:2] = xy[k, :2] + delta[k] * (limit / deviation)

        # Heading is another novel-view axis. Estimate the recorded tangent
        # using the closest non-repeated neighbours, then cap the live yaw's
        # difference from it. Repeated stop points are common at lights.
        lo, hi = k - 1, k + 1
        while lo >= 0 and np.linalg.norm(xy[k, :2] - xy[lo, :2]) < 1e-6:
            lo -= 1
        while hi < len(xy) and np.linalg.norm(xy[hi, :2] - xy[k, :2]) < 1e-6:
            hi += 1
        a = xy[lo, :2] if lo >= 0 else xy[k, :2]
        b = xy[hi, :2] if hi < len(xy) else xy[k, :2]
        heading = float(ego_state.get("heading", 0.0))
        if np.linalg.norm(b - a) > 1e-6 and yaw_limit >= 0.0:
            track_heading = math.atan2(float(b[1] - a[1]),
                                       float(b[0] - a[0]))
            yaw_error = (heading - track_heading + math.pi) % (2 * math.pi) - math.pi
            heading = track_heading + float(np.clip(
                yaw_error, -yaw_limit, yaw_limit))

        adjusted = dict(ego_state)
        adjusted["position"] = render_pos
        adjusted["heading"] = heading
        if not getattr(self, "_manifold_clamp_warned", False):
            self._manifold_clamp_warned = True
            logger.warning(
                "nurec_grpc: live ego is %.1f m from the recorded camera "
                "manifold; constraining RENDER ONLY to %.1f m / %.0f deg to "
                "prevent invalid crystal-like 3DGS output (scene=%s). "
                "Simulation and metrics retain the physical pose. Set "
                "NUREC_GRPC_MAX_NOVEL_VIEW_M=0 to disable.",
                deviation, limit, math.degrees(yaw_limit), self._scene_id)
        return adjusted

    def _pick_handoff(self, t_now):
        """handoff: pick the scene whose [t0,t1) window contains ``t_now``
        (fallback: nearest window). ``t_now`` comes from :meth:`_handoff_clock`
        — position-matched by default, wall-clock under HANDOFF_MODE=time —
        so temporally-tiled recons, which each bake dynamic state only for
        their own window, render the stretch of road the ego is actually on."""
        best = 0; bestd = None
        for _i, _h in enumerate(self._handoff):
            _tw = _h[3] if len(_h) > 3 else None
            if _tw and _tw[0] <= t_now < _tw[1]:
                self._ho_cur = _i; return _h[0], _h[1]
            if _tw:
                _d = min(abs(t_now - _tw[0]), abs(t_now - _tw[1]))
                if bestd is None or _d < bestd: bestd = _d; best = _i
        self._ho_cur = best
        return self._handoff[best][0], self._handoff[best][1]

    def _clamp_to_window(self, t_now: int) -> int:
        """Keep the request inside the window of the scene that was picked.

        The window table is ARITHMETIC -- `t0 + i * window_us` -- while the
        log's own pose timestamps are not exact multiples of the sample
        interval. The drift is tens of microseconds and it lands entirely on
        the last window: on 2ccebcdb0da25be5 the final logged frame sits 30 us
        past the end of window 4, so no window contained it, `_pick_handoff`
        fell through to its nearest-window branch, and the picked scene was
        then asked for a frame 30 us beyond what it baked. The server answered
        OUT_OF_RANGE and the episode died at frame 179 of 208 as a camera
        failure.

        Clamping is the right answer rather than widening the window: the road
        is static, so the nearest baked frame IS the frame, and a request that
        falls outside every window by more than a sample interval still lands
        on the boundary rather than being silently served from the wrong recon.
        """
        cur = getattr(self, "_ho_cur", None)
        if cur is None or not self._handoff or cur >= len(self._handoff):
            return int(t_now)
        window = self._handoff[cur][3] if len(self._handoff[cur]) > 3 else None
        if not window:
            return int(t_now)
        lo, hi = int(window[0]), int(window[1])
        out = min(max(int(t_now), lo), max(lo, hi - 1))
        if out != int(t_now):
            logger.debug("nurec_grpc handoff: clamped request %d -> %d into %s [%d, %d)",
                         t_now, out, self._handoff[cur][0], lo, hi)
        return out

    def get_camera_images(
        self, ego_state: dict, cam_configs: dict,
        agent_states: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        if agent_states is not None:
            self._agent_states = agent_states
        render_ego_state = self._render_pose_on_manifold(ego_state)
        t0 = 0
        if self._timestamps_us:
            i = min(max(self._cur_step, 0), len(self._timestamps_us) - 1)
            t0 = self._timestamps_us[i]
        if self._handoff:
            _prev = self._scene_id
            # Use one clock for choosing the scene and for the
            # requested timestamp. Under a handoff the scene is
            # chosen from the ego's position along the logged
            # path; requesting the simulation clock's timestamp
            # from that scene would run past its time range
            # whenever the policy drives slower than the log.
            t0 = self._handoff_clock(int(t0), render_ego_state)
            _sid, _off = self._pick_handoff(t0)
            t0 = self._clamp_to_window(t0)
            if _sid != _prev:
                logger.info(
                    "nurec_grpc handoff: scene %s -> %s (step %d, %s pick)",
                    _prev, _sid, self._cur_step,
                    os.environ.get("NUREC_GRPC_HANDOFF_MODE", "position"))
            self._scene_id = _sid
            self._origin_offset = _off
        rig2world = self._ego_rig_to_world(render_ego_state, int(t0))
        # Per-frame actor pose overrides (camera-independent): build once, reuse
        # across the rig. Empty when no controllable actors -> server uses its
        # baked sequence tracks (unchanged behavior).
        dyn_objs = self._build_dynamic_objects()
        # Everything above has run for this step; only the render is skipped.
        # Copies, not the cached arrays themselves: the visualiser annotates
        # the frames it is handed in place, which would otherwise corrupt the
        # cache for every later skipped step.
        reuse = self._cached_frames_for(self._cur_step)
        if reuse is not None:
            return reuse
        images: Dict[str, np.ndarray] = {}
        for name, cfg in cam_configs.items():
            h, w = int(cfg.get("height", 900)), int(cfg.get("width", 1600))
            rgb = None
            # Same fail-closed contract as setup(): a renderer that quietly
            # hands back black frames corrupts training data and eval metrics
            # without ever surfacing an error.
            strict = os.environ.get("NUREC_GRPC_STRICT", "1").lower() not in {
                "0", "false", "no", "off"}
            if not self._renderable and strict:
                raise RuntimeError(
                    f"nurec_grpc is not renderable (scene={self._scene_id}); "
                    f"camera '{name}' would be BLACK. Set NUREC_GRPC_STRICT=0 "
                    "to accept black frames.")
            if self._renderable:
                try:
                    rgb = self._render_one(name, cfg, rig2world, t0, dyn_objs)
                except Exception as exc:  # pragma: no cover — live only
                    logger.debug("nurec_grpc render %s failed: %s", name, exc)
                    if strict:
                        raise RuntimeError(
                            f"nurec_grpc render failed for {name}"
                        ) from exc
            images[name] = (rgb[:, :, ::-1].copy() if rgb is not None
                            else np.zeros((h, w, 3), np.uint8))  # RGB->BGR contract
        if self._render_steps is not None:
            self._last_images = {k: v.copy() for k, v in images.items()}
        return images

    def _render_one(self, name, cfg, rig2world, t0, dyn_objs=None) -> Optional[np.ndarray]:
        g = self._g
        logical = _CAM_LOGICAL.get(name, name)

        # Camera rig (NUREC_GRPC_CAM_RIG, default "recon"). "recon"
        # or "native" reuses the reconstruction's own camera model
        # and extrinsics, including distortion. "navsim" builds a
        # distortion-free pinhole camera from the policy's camera
        # configuration.
        rig = self._camera_rig
        base_spec = self._base_spec_for(name, cfg, rig)
        if base_spec is None:
            return None
        cam2rig: Optional[np.ndarray]
        if rig == "navsim" and self._cfg_has_geometry(cfg):
            spec = self._navsim_spec(base_spec, cfg)
            cam2rig = self._navsim_cam2rig(cfg)
            # NavSim camera heights are measured from the ground,
            # while the rig origin is the ego pose above the
            # road. Lower the camera by that offset.
            cam2rig = cam2rig.copy()
            cam2rig[2, 3] -= float(os.environ.get(
                "NUREC_GRPC_EGO_Z_TO_GROUND", "1.4"))
        else:
            cam2rig = self._cam_to_rig.get(logical)
            if cam2rig is None:
                return None
            spec = base_spec
        # Novel-view drift probe (see the module docstring): slide the whole
        # camera rig sideways in the ego frame, +left, metres. Copy first —
        # under the recon rig `cam2rig` is the cached per-camera extrinsic.
        _lat = float(os.environ.get("NUREC_GRPC_CAM_LATERAL_M", "0") or 0)
        if _lat:
            cam2rig = cam2rig.copy()
            cam2rig[1, 3] += _lat
        # Optional camera offsets for diagnostics, all zero by
        # default: vertical and longitudinal translation, and
        # NUREC_GRPC_CAM_YAW_DEG, a rotation about the rig's up axis
        # in degrees (positive to the left). They are applied to the
        # camera-to-rig transform, so the rendered view and any
        # projection derived from it agree.
        _yaw_d = float(os.environ.get("NUREC_GRPC_CAM_YAW_DEG", "0") or 0)
        if _yaw_d:
            _a = math.radians(_yaw_d)
            _Rz = np.array([[math.cos(_a), -math.sin(_a), 0.0, 0.0],
                            [math.sin(_a), math.cos(_a), 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
            cam2rig = _Rz @ np.asarray(cam2rig, dtype=float)
        _vert = float(os.environ.get("NUREC_GRPC_CAM_VERTICAL_M", "0") or 0)
        _fwd = float(os.environ.get("NUREC_GRPC_CAM_FORWARD_M", "0") or 0)
        if _vert or _fwd:
            cam2rig = cam2rig.copy()
            cam2rig[2, 3] += _vert
            cam2rig[0, 3] += _fwd
        h, w = self._request_resolution(spec, cfg, rig)
        c2w = rig2world @ cam2rig
        if os.environ.get("NUREC_GRPC_CAMDUMP") and name == "CAM_F0":
            try:
                with open(os.environ["NUREC_GRPC_CAMDUMP"], "a") as _cf:
                    _cf.write("%d %.5f %.5f %.5f %.5f %.5f\n" % (self._cur_step, c2w[0,3], c2w[1,3], c2w[2,3], rig2world[0,3], rig2world[1,3]))
            except Exception:
                pass
        # NUREC_GRPC_SPECDUMP: write the full calibration of CAM_F0
        # once, as JSON. Under the recon rig the intrinsics and
        # extrinsics come from the reconstruction, so an offline
        # projection needs them from here.
        specdump = os.environ.get("NUREC_GRPC_SPECDUMP")
        if specdump and name == "CAM_F0" and not self._specdumped:
            self._specdumped = _write_camera_spec_dump(
                specdump,
                spec=spec,
                rig=rig,
                step=int(self._cur_step),
                scene_id=getattr(self, "_scene_id", None),
                request_hw=(int(h), int(w)),
                cam2rig=cam2rig,
                c2w=c2w,
                rig2world=rig2world,
            )
        if os.environ.get("NUREC_GRPC_DEBUG_CAM") and name == "CAM_F0" and self._cur_step % 8 == 0:
            _drop = float(os.environ.get("NUREC_GRPC_EGO_Z_TO_GROUND", "1.4"))
            _ego_z = self._ego_z_now(c2w[:2, 3])
            print(f"nurec_grpc[cam] step={self._cur_step} {name} "
                  f"cam_world=[{c2w[0,3]:.2f}, {c2w[1,3]:.2f}, {c2w[2,3]:.3f}] "
                  f"rig_origin_z={rig2world[2,3]:.3f} local_ego_track_z={_ego_z:.3f} "
                  f"recon_road_z={_ego_z - _drop:.3f} "
                  f"cam_height_above_recon_road={c2w[2,3] - (_ego_z - _drop):.3f}", flush=True)
        # NUREC_GRPC_TS_LOG: log one line per rendered frame with
        # step, timestamp and scene. Under a handoff the timestamp
        # follows the ego's position, so the step index alone does
        # not identify the rendered moment.
        _tslog = os.environ.get("NUREC_GRPC_TS_LOG")
        if _tslog and name == "CAM_F0":
            try:
                with open(_tslog, "a") as _tf:
                    _tf.write(f"{self._cur_step}\t{int(t0)}\t{self._scene_id}\n")
            except Exception:  # a diagnostic must never fail a render
                pass
        pose = self._se3_to_pose(c2w)
        req = g["RGBRenderRequest"](
            scene_id=self._scene_id,
            resolution_h=h, resolution_w=w,
            camera_intrinsics=spec,
            frame_start_us=int(t0), frame_end_us=int(t0) + 1,
            sensor_pose=g["PosePair"](start_pose=pose, end_pose=pose),
            image_format=g["ImageFormat"].JPEG, image_quality=95,
            dynamic_objects=dyn_objs or [])
        return self._decode(self._send_render(req, int(t0)))

    def _send_render(self, req, t0: int) -> bytes:
        """One render call, with bounded retries for known recoverable failures.

        A recon's BAKED coverage can end before the log's last frame:
        2ccebcdb0da25be5's fourth sub-clip stops 75-100 us short, so the final
        steps ask for a frame it never trained and the server answers
        OUT_OF_RANGE -- surfacing as `render failed for CAM_F0` at frame 179 of
        208, which reads as a camera fault rather than as coverage. Clamping to
        the handoff WINDOW does not help: that table is arithmetic
        (`t0 + i * window_us`) and is wider than the bake.

        The last timestamp this scene actually served is a frame it certainly
        has, and the road is static, so re-asking at that frame is the same
        picture. Retried once; then the error stands, because a scene that has
        never served anything is a real failure rather than a boundary.
        """
        try:
            ret = self._stub.render_rgb(req, timeout=self._timeout)
        except Exception as exc:
            # `grpc` is imported lazily in setup() (the eval env vendors it),
            # so the status is read off the exception rather than compared to
            # an enum this module cannot name here.
            code = getattr(exc, "code", None)
            status = getattr(code() if callable(code) else code, "name", "")
            details_fn = getattr(exc, "details", None)
            details = str(details_fn() if callable(details_fn) else details_fn or exc)
            # The first render on a cold server can outlast the
            # HTTP/2 keep-alive while the backend compiles, and
            # the call then fails with UNAVAILABLE and "ping
            # timeout". Rendering is idempotent, so retry once
            # for exactly this error.
            if status == "UNAVAILABLE" and "ping timeout" in details.lower():
                logger.warning(
                    "nurec_grpc: first render lost its HTTP/2 transport during "
                    "cold JIT (ping timeout); reconnecting once")
                ret = self._stub.render_rgb(req, timeout=self._timeout)
                if not hasattr(self, "_last_ok_us"):
                    self._last_ok_us = {}
                self._last_ok_us[self._scene_id] = int(t0)
                return ret.image_bytes
            last_ok = getattr(self, "_last_ok_us", {}).get(self._scene_id)
            if status != "OUT_OF_RANGE" or last_ok is None or int(last_ok) == int(t0):
                raise
            logger.warning(
                "nurec_grpc: %s has no frame at %d (its bake ends earlier); "
                "re-asking at %d, the last frame it served.",
                self._scene_id, int(t0), int(last_ok))
            req.frame_start_us, req.frame_end_us = int(last_ok), int(last_ok) + 1
            ret = self._stub.render_rgb(req, timeout=self._timeout)
            t0 = int(last_ok)
        if not hasattr(self, "_last_ok_us"):
            self._last_ok_us = {}
        self._last_ok_us[self._scene_id] = int(t0)
        return ret.image_bytes

    @staticmethod
    def _render_step_filter() -> Optional[set]:
        """``NUREC_GRPC_RENDER_STEPS`` as a set of sim steps, or None for all.

        Accepts a comma/whitespace list, or ``@<path>`` to read one from a
        file — a fidelity shard passes one file per scenario and they are far
        too long for an environment variable.
        """
        raw = os.environ.get("NUREC_GRPC_RENDER_STEPS", "").strip()
        if not raw:
            return None
        if raw.startswith("@"):
            with open(raw[1:]) as fh:
                raw = fh.read()
        return {int(x) for x in re.split(r"[,\s]+", raw.strip()) if x}

    def _cached_frames_for(self, step: int) -> Optional[Dict[str, np.ndarray]]:
        """The previous frame set for a step that is not being rendered.

        None means "render this step" — either no filter is active, this step
        is in it, or nothing has been rendered yet to reuse.

        The frames are COPIED. The visualiser annotates in place whatever it
        is handed, so returning the cached arrays themselves would let one
        skipped step corrupt the cache for every later one.
        """
        if (self._render_steps is None or step in self._render_steps
                or self._last_images is None):
            return None
        return {k: v.copy() for k, v in self._last_images.items()}

    @staticmethod
    def _camera_rig_mode() -> str:
        """Resolve the camera mode once per renderer instance.

        ``native`` is a clearer public spelling for the historical ``recon``
        value; normalize it so the rest of the code has only two branches.
        """
        rig = os.environ.get("NUREC_GRPC_CAM_RIG", "recon").strip().lower()
        if rig == "native":
            rig = "recon"
        if rig not in {"recon", "navsim"}:
            raise ValueError(
                "NUREC_GRPC_CAM_RIG must be 'recon'/'native' or 'navsim', "
                f"got {rig!r}")
        return rig

    @staticmethod
    def _request_resolution(spec, cfg: dict, rig: str) -> tuple[int, int]:
        """Return ``(height, width)`` for the render request.

        Native mode must use the advertised training-camera dimensions. Passing
        the policy canvas height with a baked spec silently asks NRE for another
        resampled view and defeats the purpose of the native-camera path.
        """
        if rig == "navsim":
            return (int(cfg.get("height", 0)) or 1120,
                    int(cfg.get("width", 0)) or 1920)
        return (int(getattr(spec, "resolution_h", 0)) or
                int(cfg.get("height", 0)) or 1080,
                int(getattr(spec, "resolution_w", 0)) or
                int(cfg.get("width", 0)) or 1920)

    def _base_spec_for(self, name: str, cfg: dict, rig: str):
        """Resolve the protobuf template used to describe a rendered camera.

        In ``navsim`` mode the output is a novel virtual view: policy geometry
        supplies both intrinsics and extrinsics, while an advertised server
        ``CameraSpec`` is only a protobuf template. Reconstructions from rigs
        without a rear camera therefore need not feed a black ``CAM_B0`` to a
        four-camera policy; borrow an available spec and overwrite its
        geometry in :meth:`_navsim_spec`. ``recon`` mode still requires the
        exact baked logical camera.
        """
        logical = _CAM_LOGICAL.get(name, name)
        spec = self._cam_intrinsics.get(logical)
        if (spec is None and rig == "navsim" and self._cfg_has_geometry(cfg)
                and self._cam_intrinsics):
            spec = (self._cam_intrinsics.get("camera_pcam_f0")
                    or next(iter(self._cam_intrinsics.values())))
            if name not in self._template_fallback_warned:
                logger.warning(
                    "nurec_grpc: scene=%s lacks %s; rendering synthetic %s "
                    "from policy geometry using an available CameraSpec template",
                    self._scene_id, logical, name)
                self._template_fallback_warned.add(name)
        return spec

    @staticmethod
    def _cfg_has_geometry(cfg: dict) -> bool:
        """True when ``cfg`` carries a full pinhole rig (NavSim sensor config)."""
        return (all(k in cfg for k in ("x", "y", "z"))
                and ("fov_h" in cfg or "fov" in cfg))

    def _navsim_spec(self, base_spec, cfg):
        """Pinhole ``CameraSpec`` from the model's NavSim intrinsics.

        Copies the server spec (to keep logical_id / trajectory_idx / shutter)
        then overwrites the pinhole params: focal from cfg's horizontal/vertical
        fov, principal point at the image centre, zero distortion, resolution
        from cfg. Distortion is intentionally zero — the policy expects an
        undistorted pinhole and the overlay projects with the same model.
        """
        w = int(cfg.get("width", 0)) or 1920
        h = int(cfg.get("height", 0)) or 1120
        fov_h = float(cfg.get("fov_h") or cfg.get("fov") or 70.0)
        fov_v = float(cfg.get("fov_v") or 0.0) or math.degrees(
            2.0 * math.atan((h / w) * math.tan(math.radians(fov_h) / 2.0)))
        fx = w / (2.0 * math.tan(math.radians(fov_h) / 2.0))
        fy = h / (2.0 * math.tan(math.radians(fov_v) / 2.0))
        spec = base_spec.__class__()
        spec.CopyFrom(base_spec)
        op = spec.opencv_pinhole_param
        op.Clear()
        op.focal_length_x = fx
        op.focal_length_y = fy
        op.principal_point_x = w / 2.0
        op.principal_point_y = h / 2.0
        op.radial_coeffs.extend([0.0] * 6)
        op.tangential_coeffs.extend([0.0] * 2)
        op.thin_prism_coeffs.extend([0.0] * 4)
        spec.resolution_h = h
        spec.resolution_w = w
        return spec

    @staticmethod
    def _navsim_cam2rig(cfg) -> np.ndarray:
        """4x4 camera->rig SE3 from the NavSim extrinsic.

        Translation = cfg (x-fwd, y-left, z-up in the rig/FLU frame). Rotation =
        the FLU mounting rotation ``Rz(yaw) Ry(pitch) Rx(roll)`` composed with the
        optical->FLU base ``R_base`` (camera z->rig +x forward, x->rig -y right,
        y->rig -z down). Calibrated against the server's WOD front-cam quat
        (match to 3e-3, i.e. its residual mounting pitch).
        """
        x = float(cfg.get("x", 0.0)); y = float(cfg.get("y", 0.0)); z = float(cfg.get("z", 0.0))
        yaw = math.radians(float(cfg.get("yaw", 0.0)))
        pitch = math.radians(float(cfg.get("pitch", 0.0)))
        roll = math.radians(float(cfg.get("roll", 0.0)))
        cz, sz = math.cos(yaw), math.sin(yaw)
        cy, sy = math.cos(pitch), math.sin(pitch)
        cx, sx = math.cos(roll), math.sin(roll)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        m = np.eye(4)
        m[:3, :3] = (Rz @ Ry @ Rx) @ R_base
        m[:3, 3] = [x, y, z]
        return m

    def _build_dynamic_objects(self) -> list:
        """Authoritative per-frame ``DynamicObject`` list — the render mirrors
        the sim's agent set exactly when server-side editing is enabled.

        Setup rejects explicitly disabled editing. The capability guard also
        prevents pose overrides after a failed setup. Ego camera motion is
        independent of actor editing.

        EVERY server-exposed controllable actor is sent: those present in the
        env's ``agent_states`` at their current world pose, those ABSENT
        relocated far off-screen. So the camera stays consistent with the sim
        STATE — an actor dropped from the scenario (e.g. via
        ``EnvCfg.remove_agents``) disappears from the render as well as from the
        BEV / collision / observation. An empty ``agent_states`` therefore means
        every actor off-screen (background only). The proto has no delete flag,
        so "remove" == relocate out of view; static separation put parked
        cars/signs into the controllable set too, so they follow the same rule.

        Scoped to the scene that owns the current frame: under a handoff
        ``_pick_handoff`` has already swapped ``_scene_id``, and an override
        naming an actor that scene does not hold is an INVALID_ARGUMENT.
        """
        if not getattr(self, "_actor_editing_enabled", True):
            return []
        scene_ids = self._dyn_track_ids_by_scene.get(
            self._scene_id, self._dyn_track_ids)
        if not scene_ids:
            return []
        g = self._g
        far = np.eye(4)
        far[:3, 3] = [1.0e5, 1.0e5, 0.0]
        far_pose = self._se3_to_pose(far)
        real = {}
        for st in (self._agent_states or []):
            tid = str(st.get("id"))
            # A gait-banked actor is on the server as its phases, never under
            # its own id, so it resolves through the phase clock. The pose is
            # the actor's either way: only WHICH geometry shows up differs.
            if tid in self._pose_banks:
                pid = self._bank_phase_id(tid, st)
                if pid in scene_ids:
                    real[pid] = self._se3_to_pose(self._actor_to_world(st))
            elif tid in scene_ids:
                real[tid] = self._se3_to_pose(self._actor_to_world(st))
        # Fallback for tracks this renderer inserted when the agent
        # states do not carry them: follow the recipe's baked arrays,
        # which are what collisions are scored against.
        k = min(max(self._cur_step, 0), 10**9)
        for tid, arrs in self._injected_track_states.items():
            banked = tid in self._pose_banks
            if not banked and (tid in real or tid not in scene_ids):
                continue
            n = len(arrs["position"])
            i = min(k, n - 1)
            if not bool(arrs["valid"][i]):
                continue  # invalid frame -> stays off-screen, as authored
            state = {"id": tid,
                     "position": arrs["position"][i],
                     "heading": float(arrs["heading"][i])}
            key = self._bank_phase_id(tid, state) if banked else tid
            if key in real or key not in scene_ids:
                continue
            real[key] = self._se3_to_pose(self._actor_to_world(state))
        return [g["DynamicObject"](
                    track_id=str(tid),
                    pose_pair=g["PosePair"](start_pose=real.get(tid, far_pose),
                                            end_pose=real.get(tid, far_pose)))
                for tid in scene_ids]

    def _actor_to_world(self, state: dict) -> np.ndarray:
        """4x4 actor-to-world in NuRec world from a symbolic agent state.

        Same eval-local -> NuRec-world mapping as the ego (recentered xy minus
        the reconstruction origin offset). z comes from one of three places,
        and the server wants the box CENTRE in every case:

        * an injected asset uses its pinned base z (see below);
        * a 3D agent state is used verbatim -- under log_replay that is the
          actor's own logged altitude, already a centre because py123d reads
          ``center_se3``, so no half-height is added;
        * a planar state falls back to an estimate, which only reactive
          traffic hits.
        """
        pos = np.asarray(state.get("position", (0.0, 0.0, 0.0)), np.float64)
        heading = float(state.get("heading", 0.0))
        ox, oy, _ = self._origin_offset.tolist()
        tid = str(state.get("id"))
        if tid in self._injected_ground_z:
            # Inserted assets are placed at the ground height
            # recorded in the recipe, which was measured at the
            # spawn. It is kept constant: estimating the current
            # road height from the ego's track is wrong wherever
            # the actor is not near the ego. The value is the
            # road height itself, since an inserted asset's
            # origin is at its base, whereas a logged actor's z
            # is its box centre.
            z = self._injected_ground_z[tid]
        elif pos.shape[0] >= 3 and pos[2] != 0.0:
            z = float(pos[2])
        else:
            # Reactive traffic only. Its pose is planar, so the
            # height is estimated from the ego's recorded track
            # plus half the actor's box. This is approximate,
            # because the road under the actor can differ from
            # the road under the ego.
            z = self._ego_z_now(pos[:2]) + self._dyn_track_half_h.get(tid, 0.0)
        Rz = np.array([[math.cos(heading), -math.sin(heading), 0.0],
                       [math.sin(heading),  math.cos(heading), 0.0],
                       [0.0, 0.0, 1.0]])
        m = np.eye(4)
        m[:3, :3] = Rz
        e0 = self._eval_pos0 if self._eval_pos0 is not None else np.zeros(2)
        m[:3, 3] = [(float(pos[0]) - float(e0[0])) - ox,
                    (float(pos[1]) - float(e0[1])) - oy, z]
        return m

    # ------------------------------------------------------------------
    # Level B: server-side asset insertion (edit_assets)
    # ------------------------------------------------------------------
    @staticmethod
    def _proto_field_names(msg_cls) -> set:
        return {f.name for f in msg_cls.DESCRIPTOR.fields}

    def _make_pose_at_time(self, PoseAtTime, pose, t_us: int):
        """Build a PoseAtTime binding fields by TYPE, not name.

        Field names vary across NRE proto vintages; the message shape does
        not — one Pose-typed field, one scalar timestamp field.
        """
        kw: Dict[str, Any] = {}
        for f in PoseAtTime.DESCRIPTOR.fields:
            if f.message_type is not None and f.message_type.name == "Pose":
                kw[f.name] = pose
            elif f.message_type is None:
                kw[f.name] = int(t_us)
        return PoseAtTime(**kw)

    @staticmethod
    def _make_trajectory(Trajectory, pose_at_times: list):
        fld = next(f for f in Trajectory.DESCRIPTOR.fields
                   if f.label == f.LABEL_REPEATED and f.message_type is not None)
        traj = Trajectory()
        getattr(traj, fld.name).extend(pose_at_times)
        return traj

    def _resolve_semantic_class(self, meta: dict) -> str:
        """Map an injected track's type to a semantic_class the server accepts.

        The server parses ``DynamicObjectTrack.semantic_class`` against the
        artifact's label enum (e.g. ``WODPerceptionBoxDetectionLabel.
        TYPE_VEHICLE``); an unknown literal fails the entire insert (verified
        against a live server). A metadata value that already looks like an
        enum literal (contains a dot) passes through verbatim; otherwise pick
        the closest class observed on the scene's own tracks — keyword match
        first (cone -> *CONE*, pedestrian -> *PEDESTRIAN*, ...), then any
        *VEHICLE*, then the scene's most common class. *SIGN* is deliberately
        NOT a cone candidate: sign-class inserts are rejected by the live
        server (signs take a special static-separation path), while VEHICLE /
        PEDESTRIAN inserts succeed.
        """
        want = str(meta.get("nurec_semantic_class")
                   or meta.get("type") or "").strip()
        if "." in want:
            return want
        observed = sorted(self._dyn_sem_counter.items(), key=lambda kv: -kv[1])
        if not observed:
            return want or "vehicle"
        keywords = {
            "cone": ("CONE",), "traffic_cone": ("CONE",),
            "pedestrian": ("PEDESTRIAN",), "cyclist": ("CYCLIST",),
            # Robotic micromobility family + stroller are inserted into the
            # actor (deformable) layer, which many recons (e.g. the WOD
            # car2sim recipe) permit ONLY for a PEDESTRIAN-class track — a
            # VEHICLE-class insert is rejected server-side and renders
            # invisibly. Route these archetypes to *PEDESTRIAN.
            "delivery_robot": ("PEDESTRIAN",), "robot_dog": ("PEDESTRIAN",),
            "humanoid_robot": ("PEDESTRIAN",), "scooter": ("PEDESTRIAN",),
            "stroller": ("PEDESTRIAN",), "escooter_rider": ("PEDESTRIAN",),
        }.get(want.lower(), ())
        # Insertable classes are the ones this reconstruction actually trained
        # an actor layer for. GENERIC_OBJECT never is (the server refuses it
        # even on scenes whose only dynamic object IS one), so it is not a
        # candidate at any priority.
        usable = [sc for sc, _ in observed if "GENERIC_OBJECT" not in sc.upper()]
        # Try the requested class first, even if no logged track uses
        # it: what a reconstruction accepts depends on its trained
        # actor layers, not on its recorded traffic.
        # `_edit_assets_with_class_fallback` handles a reconstruction
        # without such a layer.
        for kw in keywords:
            sem = self._class_literal(kw)
            if sem is not None:
                return sem
        for sc in usable:
            if "VEHICLE" in sc.upper():
                return sc
        for kw in ("PEDESTRIAN", "CYCLIST", "VEHICLE"):
            sem = self._class_literal(kw)
            if sem is not None:
                logger.warning("nurec_grpc: %r is not among this scene\u2019s classes "
                               "%s; inserting as %s instead", want, usable, sem)
                return sem
        if usable:
            return usable[0]
        raise RuntimeError(
            f"this reconstruction exposes no insertable actor class (observed: "
            f"{[sc for sc, _ in observed] or None}), so no asset can be added to "
            f"it. Pick a host whose recon has dynamic actors, or retrain this one.")

    def _make_aabb(self, AABB, length: float, width: float, height: float):
        """Actor-local AABB, base on the ground plane (pose z rides the road).

        The shipped NRE proto carries plain extents (``size_x/y/z``, verified
        against the vendored stubs); the min/max and center/size branches
        cover other proto vintages.
        """
        g = self._g
        names = self._proto_field_names(AABB)
        if {"size_x", "size_y", "size_z"} <= names:
            return AABB(size_x=float(length), size_y=float(width),
                        size_z=float(height))
        hl, hw = length / 2.0, width / 2.0
        if {"min", "max"} <= names:
            return AABB(min=g["Vec3"](x=-hl, y=-hw, z=0.0),
                        max=g["Vec3"](x=hl, y=hw, z=height))
        if {"center", "size"} <= names:
            return AABB(center=g["Vec3"](x=0.0, y=0.0, z=height / 2.0),
                        size=g["Vec3"](x=length, y=width, z=height))
        logger.warning("nurec_grpc: unrecognized AABB fields %s; sending empty "
                       "object_size", sorted(names))
        return AABB()

    @staticmethod
    def _asset_box_extent(dims=None) -> List[float]:
        """The box extent, above ``dims_offset``, that an inserted asset needs.

        Never below 1.0 per axis: the box doubles as the asset's extent
        server-side and squashes anything larger onto it. Measured on a live
        server, a 2.34 m sign in the default 1.25 m box came back compressed in
        z by 1.87x, so the box has to admit the real dims.
        """
        if dims is None:
            return [1.0, 1.0, 1.0]
        return [max(1.0, float(d)) for d in dims]

    def _asset_scale_factor(self, dims=None) -> float:
        """The uniform scale the server will apply to a loaded asset's raw
        coordinates: ``(object_size - dims_offset).max()``.

        Sizing the box to the real dims (which the squash above forces) makes
        this the asset's own longest dimension, never 1.0 — hence the
        pre-division in :meth:`_server_asset_path`.
        """
        return max(self._asset_box_extent(dims))

    def _grpc_object_size(self, AABB, dims=None):
        """``object_size`` for an inserted 3DGS PLY.

        The box is not a free cull margin — it is the same knob as the scale:
        the server sizes the asset by it *and* scales by
        ``(object_size - asset_bank.dims_offset).max()``. So it is sized to the
        track's real dims, and :meth:`_server_asset_path` cancels the scale
        that sizing implies.

        Args:
            AABB: the resolved proto message class.
            dims: the track's real ``(length, width, height)`` in metres, or
                None to keep the legacy fixed box.

        dims_offset is server-side config (NRE default ``[1.0, 1.0, 0.25]``,
        logged by the server on every insert as "Applying dims_offset: ...");
        override ``NUREC_GRPC_ASSET_DIMS_OFFSET`` if the deployment differs. NB
        the real track dims still reach sim state / scoring via the scenario
        track — this AABB only feeds the server.
        """
        raw = os.environ.get("NUREC_GRPC_ASSET_DIMS_OFFSET", "1.0,1.0,0.25")
        off = [float(v) for v in raw.split(",")]
        extent = self._asset_box_extent(dims)
        return self._make_aabb(
            AABB, off[0] + extent[0], off[1] + extent[1], off[2] + extent[2])

    def _server_asset_path(self, asset_id: str, dims=None) -> str:
        """The asset the server should load: our metric PLY pre-divided by the
        scale the server is about to apply to it.

        NRE's scale convention assumes AssetHarvester PLYs, which ship
        normalised to unit max extent with ``object_size`` supplying the real
        size. This library is authored in METRES instead — ``assets calibrate``
        measures an asset's visual extent and writes it out life-size — so the
        size was applied twice: a 1.75 m bicycle in a box asking for 1.75 m
        rendered 3.05 m tall. Every inserted asset was affected, each scaled by
        its own longest dimension (cars by 4.6x), while the sim box stayed
        correct — so the pixels the policy drives on disagreed with the physics
        that scored it.

        Dividing by exactly the factor :meth:`_asset_scale_factor` is about to
        send cancels it for any asset, whatever its dims and however they were
        measured. The scaled copy is cached beside the source, keyed by the
        source's content hash, so re-authoring an asset invalidates it;
        ``NUREC_GRPC_ASSET_CACHE`` relocates the cache when the library is
        mounted read-only. The cache must be readable by the sensorsim
        container, which is why it defaults to sitting next to the asset.

        An ``asset_id`` that is not a PLY on disk is an AssetBank track id
        baked into the artifact; the bank sizes those itself and there is no
        file to divide, so it is passed through.
        """
        factor = self._asset_scale_factor(dims)
        if factor <= 1.0:
            return asset_id
        src = Path(asset_id)
        if src.suffix.lower() != ".ply" or not src.is_file():
            logger.warning(
                "nurec_grpc: asset %r is not a PLY on disk, so it cannot be "
                "pre-divided by the server's %.3fx scale; it will render %.3fx "
                "its authored size if it is metric.", asset_id, factor, factor)
            return asset_id
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        root = os.environ.get("NUREC_GRPC_ASSET_CACHE")
        cache = Path(root) if root else src.parent / "_nurec_prescaled"
        dst = cache / f"{src.stem}.{digest}.div{factor:.4f}.ply"
        if not dst.is_file():
            scale_ply(src, dst, 1.0 / factor)
            logger.info("nurec_grpc: wrote %s (source divided by %.3fx to "
                        "cancel the server's scale)", dst, factor)
        return str(dst)

    @staticmethod
    def _track_dims(tr: dict):
        """The track's own ``(length, width, height)`` in metres, or None.

        The track's per-frame box sizes the ``object_size`` the server gets, so
        an asset larger than the default box is not silently squashed onto it.
        """
        st = tr.get("state") or {}
        keys = ("length", "width", "height")
        if not all(k in st for k in keys):
            return None
        return [float(np.asarray(st[k]).reshape(-1)[0]) for k in keys]

    def _resolve_assets(self, tid: str, meta: dict, dims) -> List[tuple]:
        """The (server track id, asset) pairs one scenario track needs.

        A plain track needs one. A track carrying ``nurec_pose_bank`` needs one
        per baked gait phase, because geometry cannot be changed after the
        insert — swapping which phase is on screen is the only way limbs move.
        Registering the bank here (rather than at render time) is what lets
        :meth:`_build_dynamic_objects` stay a pure per-frame lookup.

        A declared but unreadable bank raises rather than falling back to the
        static asset: the fallback renders a pedestrian that slides instead of
        walks, which is a wrong result that looks like a working one.
        """
        bank_dir = meta.get("nurec_pose_bank")
        if not bank_dir:
            return [(tid, self._server_asset_path(
                str(meta.get("nurec_asset_id")), dims))]

        import json
        root = Path(str(bank_dir))
        manifest = root / "bank.json"
        try:
            spec = json.loads(manifest.read_text())
            phases = [root / n for n in spec["phases"]]
            stride = float(spec["stride_m"])
        except Exception as exc:
            raise RuntimeError(
                f"track {tid!r} declares nurec_pose_bank={bank_dir!r} but its "
                f"manifest {manifest} is unusable ({exc}). Bake it with "
                f"`navsafe assets animate`, or drop the key to go back to the "
                f"static asset.") from exc
        missing = [p for p in phases if not p.is_file()]
        if missing or not phases:
            raise RuntimeError(
                f"track {tid!r}: pose bank {root} lists {len(phases)} phase(s) "
                f"but {len(missing)} are absent (e.g. {missing[:2]})")

        out = [(f"{tid}#ph{j}", self._server_asset_path(str(p), dims))
               for j, p in enumerate(phases)]
        self._pose_banks[tid] = {"ids": [i for i, _ in out], "stride_m": stride}
        logger.info("nurec_grpc: %s walks — %d gait phases, stride %.3f m (%s)",
                    tid, len(out), stride, root)
        return out

    def _bank_phase_id(self, tid: str, state: dict) -> str:
        """Which phase of ``tid``'s gait belongs on screen this frame.

        The odometer integrates the actor's own path, so the gait is tied to
        ground covered rather than to time and the feet stay planted whatever
        speed the policy picked. It advances once per sim step: a rig renders
        several cameras per frame and each asks for the pose list again.
        """
        bank = self._pose_banks[tid]
        pos = np.asarray(state.get("position", (0.0, 0.0)), np.float64).reshape(-1)
        xy = pos[:2]
        clk = self._bank_clock.get(tid)
        if clk is None:
            clk = {"step": self._cur_step, "xy": xy.copy(), "travel": 0.0}
            self._bank_clock[tid] = clk
        elif clk["step"] != self._cur_step:
            clk["travel"] += float(np.linalg.norm(xy - clk["xy"]))
            clk["xy"] = xy.copy()
            clk["step"] = self._cur_step
        n = len(bank["ids"])
        stride = bank["stride_m"] or 1.0
        phase = int(clk["travel"] / stride * n) % n
        # A gait stuck on phase 0 renders a pedestrian that slides exactly like
        # the static asset it replaced — a silent failure with no visual tell
        # beyond "the legs never moved", which is what this was built to fix.
        # NUREC_GRPC_DEBUG_GAIT prints the odometer so it can be checked.
        if os.environ.get("NUREC_GRPC_DEBUG_GAIT"):
            logger.info("nurec_grpc[gait] step=%d %s travel=%.3fm phase=%d/%d",
                        self._cur_step, tid, clk["travel"], phase, n)
        return bank["ids"][phase]

    def _insert_injected_assets(self, scenario_data: dict) -> None:
        """Register tracks carrying ``metadata.nurec_asset_id`` into the served
        scene via the ``edit_assets`` RPC.

        The asset_id must resolve ON THE SERVER: either an AssetBank track id
        baked into the artifact's ``external_assets``, or a filesystem path to
        a **3DGS PLY** readable inside the sensorsim container (vertex attrs
        ``opacity``/``f_dc_*``/``scale_*``/``rot_*``; a mesh GLB/USD will not
        load). Successful ids join ``_dyn_track_ids_by_scene`` for the scene
        that took them, so the per-frame authoritative mirror
        (:meth:`_build_dynamic_objects`) drives them like any other
        controllable actor — present in ``agent_states`` => rendered at the sim
        pose, absent => relocated off-screen. The render therefore stays
        grounded by sim state in both directions.

        **Every handoff scene is inserted into, not just the current one.** A
        bundle is served as one scene per subclip and each window renders from
        its own; inserting into ``_scene_id`` alone put the asset in the first
        subclip and nowhere else, and registering it only in the flat
        ``_dyn_track_ids`` meant the per-frame mirror — which reads the
        per-scene map — never drove it at all. Both together render an inserted
        asset invisible on any multi-subclip bundle while the server still
        answers "Completed insertion" (repro: scenario 0122ce98b2735558, 4
        subclips). Single-artifact hosts never showed it.

        ``edit_assets`` snapshots model parameters server-side before applying;
        :meth:`close` calls ``restore_model_parameters`` to undo the insert so
        the warm shared server does not leak edits across evals.
        """
        tracks = (scenario_data or {}).get("tracks") or {}
        todo = []
        for tid, tr in tracks.items():
            meta = (tr.get("metadata") or {})
            if meta.get("nurec_asset_id"):
                todo.append((str(tid), tr, meta))
        if not todo:
            return
        Track = self._find_msg("DynamicObjectTrack")
        Req = self._find_msg("EditAssetsRequest")
        Trajectory = self._find_msg("Trajectory")
        PoseAtTime = self._find_msg("PoseAtTime")
        AABB = self._find_msg("AABB")

        t0 = int(self._timestamps_us[0]) if self._timestamps_us else 0
        t1 = int(self._timestamps_us[-1]) if self._timestamps_us else t0 + 1

        # Resolve each track's server ids before the checks below. A
        # track with a gait bank exists on the server as its pose
        # phases, not under its own id.
        assets = {tid: self._resolve_assets(tid, meta, self._track_dims(tr))
                  for tid, tr, meta in todo}

        # Self-healing: our injected ids already existing server-side means a
        # previous run died without close() (restore never ran). Roll the
        # scene back once and re-enumerate before inserting — otherwise every
        # insert would collide and the leak would compound across runs.
        stale = [sid for pairs in assets.values() for sid, _ in pairs
                 if sid in self._dyn_track_ids]
        if stale:
            logger.warning("nurec_grpc: stale injected track(s) %s found on "
                           "server (previous run not closed?); calling "
                           "restore_model_parameters before inserting.", stale)
            self._restore_server_edits()
            g2 = self._g
            dyn = self._stub.get_dynamic_objects(
                g2["AvailableDynamicObjectsRequest"](scene_id=self._scene_id),
                timeout=self._timeout).dynamic_objects
            self._dyn_track_ids = {
                (getattr(o, "track_id", None) or getattr(o, "id", None))
                for o in dyn}
            self._dyn_track_ids.discard(None)

        inserts, ids = [], []
        for tid, tr, meta in todo:
            clash = [sid for sid, _ in assets[tid] if sid in self._dyn_track_ids]
            if clash:
                # The server rejects inserts whose id collides with an existing
                # scene track — skip rather than fail the whole request.
                logger.error("nurec_grpc: injected id(s) %r collide with an "
                             "existing scene track; skipping insert", clash)
                continue
            st = tr.get("state") or {}
            p = np.asarray(st.get("position"), np.float64)
            p0 = p.reshape(-1, p.shape[-1])[0] if p.ndim >= 2 else p
            heading = float(np.asarray(st.get("heading", 0.0)).reshape(-1)[0])
            if p0.shape[0] >= 3 and p0[2] != 0.0:
                self._injected_ground_z[tid] = float(p0[2])
            # Keep the full baked arrays: the replay-phase pose source for
            # _build_dynamic_objects when agent_states omits this track.
            if p.ndim >= 2:
                T_arr = p.reshape(-1, p.shape[-1])
                self._injected_track_states[tid] = {
                    "position": T_arr.copy(),
                    "heading": np.asarray(
                        st.get("heading", np.zeros(len(T_arr))), np.float64
                    ).reshape(-1).copy(),
                    "valid": np.asarray(
                        st.get("valid", np.ones(len(T_arr), bool))
                    ).astype(bool).reshape(-1).copy(),
                }
            world = self._actor_to_world(
                {"id": tid, "position": p0, "heading": heading})
            pose = self._se3_to_pose(world)
            traj = self._make_trajectory(
                Trajectory,
                [self._make_pose_at_time(PoseAtTime, pose, t) for t in (t0, t1)])
            sem = self._resolve_semantic_class(meta)
            track_dims = self._track_dims(tr)
            # One entry for a static asset, one per gait phase for a pose bank.
            # Every phase shares the track's semantic class, box and trajectory
            # — they are the same actor, differing only in limb pose.
            for sub_id, asset_id in assets[tid]:
                logger.info("nurec_grpc: inserting %s as semantic_class=%r "
                            "asset=%s dims=%s scale=%.3fx", sub_id, sem,
                            asset_id, track_dims,
                            self._asset_scale_factor(track_dims))
                inserts.append(Track(
                    id=sub_id,
                    semantic_class=sem,
                    trajectory=traj,
                    object_size=self._grpc_object_size(AABB, dims=track_dims),
                    asset_id=asset_id))
                ids.append(sub_id)
        # Remember how to rebuild each Track under a different class, for the
        # retry below.
        rebuild = {
            "Track": Track, "AABB": AABB,
            "specs": [(t.id, t.trajectory, t.object_size, t.asset_id) for t in inserts],
        }
        if not inserts:
            return
        # One insert per served scene: under a handoff each subclip is its own
        # scene, and an asset inserted into one is absent from the others.
        scene_ids = ([sid for sid, _off, _t0, _t1 in self._handoff]
                     if self._handoff else [self._scene_id])
        # Try every handoff window. Each is a separate reconstruction
        # and may accept or refuse the insert independently; stopping
        # at the first refusal would leave the actor visible in some
        # windows only. What was placed and what was refused is
        # reported.
        placed, refused = [], []
        # With NAVSAFE_INSERT_CLASS_PER_SCENE each window negotiates
        # the class starting from the one the recipe asked for,
        # instead of inheriting the class an earlier window fell back
        # to.
        _per_scene = os.environ.get(
            "NAVSAFE_INSERT_CLASS_PER_SCENE", "").strip() in ("1", "true", "yes")
        _orig = list(inserts)
        skipped_empty = []
        for sid in scene_ids:
            # Skip a window that has no controllable actors: it
            # has no actor layer for an inserted node, and
            # rendering it would fail. The actor is then
            # invisible in that window while the simulation still
            # has it.
            if not self._dyn_track_ids_by_scene.get(sid):
                skipped_empty.append(sid)
                continue
            resp, got = self._edit_assets_with_class_fallback(
                Req, sid, list(_orig) if _per_scene else inserts, rebuild)
            if not _per_scene:
                inserts = got
            if not getattr(resp, "success", True):
                refused.append((sid, getattr(resp, "message", "") or str(resp)[:200]))
                continue
            # Register the inserted ids per scene. The per-scene
            # enumeration ran before the insert, and
            # _build_dynamic_objects reads the per-scene sets.
            self._dyn_track_ids_by_scene.setdefault(sid, set()).update(ids)
            placed.append(sid)
        if skipped_empty:
            logger.warning(
                "nurec_grpc: %d hand-off window(s) hold ZERO controllable actors "
                "(%s); the insert is skipped there, because a window with no actor "
                "layer crashes the renderer on its first frame. The asset is "
                "INVISIBLE in those windows while sim state and the BEV keep it.",
                len(skipped_empty), skipped_empty)
        if not placed:
            raise RuntimeError(
                "edit_assets placed the insert in NO scene: "
                f"refused={refused or 'none'}, "
                f"skipped as zero-actor={skipped_empty or 'none'}")
        self._dyn_track_ids.update(ids)
        self._inserted_asset_ids.update(ids)
        if refused:
            logger.error(
                "nurec_grpc: %d of %d hand-off window(s) refused the insert — the actor "
                "renders in %s and is INVISIBLE in %s while sim state and the BEV keep it "
                "everywhere. Refusals: %s",
                len(refused), len(scene_ids), placed, [sid for sid, _ in refused],
                "; ".join(f"{sid}: {msg}" for sid, msg in refused))
        logger.info("nurec_grpc: edit_assets inserted %d asset track(s) %s into "
                    "%d scene(s) %s", len(ids), ids, len(placed), placed)

    # ------------------------------------------------------------------
    # Level C: swap a baked actor's gaussians for a harvested asset
    # ------------------------------------------------------------------
    def _reset_server_scenes(self, scene_ids) -> list:
        """Roll every scene this run will touch back to how it was reconstructed.

        Returns the scenes that were actually reset. A failure is logged and not
        raised: an older proto vintage without the RPC, or a server that refuses
        it, should not stop a render — but it must be visible, because what it
        means is that an earlier run's edits are still applied.
        """
        done = []
        Req = None
        try:
            Req = self._find_msg("RestoreModelParametersRequest")
        except Exception as exc:  # pragma: no cover — old stubs only
            logger.info("nurec_grpc: no restore_model_parameters in these stubs "
                        "(%s); edits left by an earlier run stay applied", exc)
            return done
        for sid in scene_ids:
            try:
                self._stub.restore_model_parameters(Req(scene_id=sid),
                                                    timeout=self._timeout)
                done.append(sid)
            except Exception as exc:  # pragma: no cover — live only
                logger.warning("nurec_grpc: could not reset scene %s before setup "
                               "(%s); any asset an earlier run edited in is STILL "
                               "APPLIED and will render as if it were baked",
                               sid, exc)
        return done

    def _replace_harvested_assets(self) -> None:
        """Re-skin logged actors from a NavSafe harvested-asset bank.

        A reconstruction fits each actor to the views the LOGGED ego had. Drive
        the scenario differently -- and a closed-loop policy always does -- and
        the same car is seen from angles no training view covered, which renders
        as a smear; worst on oncoming traffic, and worst again just after a
        handoff, where the incoming 5 s model never saw that car from here at
        all. ``edit_assets(replace=...)`` swaps the actor's gaussians for a
        view-consistent asset harvested from the same clip
        (``navsafe/harvest/``), so its appearance stops depending on where the
        logged ego drove.

        **Appearance only.** The track keeps its id, its box and its per-frame
        pose from :meth:`_build_dynamic_objects`; sim state is untouched and no
        metric moves because of this. That is the whole difference from
        ``the evaluator --replace-agent-ids``, which deletes the logged actor
        and inserts a different one in its place.

        ``NUREC_GRPC_ASSET_REPLACE`` names the manifest. The manifest lists
        TRACKS, not scenes, because a 20 s scenario is served as four 5 s scenes
        and each holds a different subset of the same cars; the intersection is
        taken here, per scene, against what ``get_dynamic_objects`` reported.
        A track the manifest covers but this scene does not hold is normal and
        silent. NOT ONE track matching in ANY scene is fatal: it means the bank
        was harvested from a different reconstruction, and proceeding would
        render the un-replaced smears the run was set up to avoid while
        reporting success.

        The AABB sent per replacement is the server's own box for that track.
        The alternative, the cuboid recorded at harvest time, is the same number
        via a longer path -- and where the two ever disagree, the server's is
        the one the reconstruction is actually posed against.

        ``replacement_id`` is a PLY path readable INSIDE the render container,
        not an id baked into the artifact: NVIDIA's documented route repackages
        the USDZ with ``export-external-assets`` first, which at NavSafe scale
        is a rewrite of a 2.1 GB artifact per 5 s window. The gRPC path skips
        it: a bad path answers ``PLYGaussianLoader
        provided path ... not a file``.
        """
        spec = os.environ.get("NUREC_GRPC_ASSET_REPLACE")
        required = getattr(self, "_required_harvest_tracks", set())
        if not spec:
            if required:
                raise RuntimeError("Takeover requires --asset-harvester-replace for original tracks: " + ", ".join(sorted(required)))
            return
        # Imported here, not at module scope: a generic render backend should
        # not pull a benchmark package in on every import, and this is the only
        # path that needs it. Reading the manifest through its own module rather
        # than re-parsing the JSON keeps one definition of what a bank is,
        # including the checks that make a half-resolvable bank fatal.
        from navsafe.benchmark.harvest import manifest as _manifest

        doc = _manifest.read(Path(spec))
        plys = _manifest.ply_by_track(doc)
        # Limit how many actors are replaced. Each replaced actor is
        # held once per scene that contains it, in addition to the
        # four reconstructions, and a camera render needs over 1 GiB
        # of transient memory. Nearest actors first, since distant
        # ones cost the same memory for few pixels.
        Req = self._find_msg("EditAssetsRequest")
        Action = self._find_msg("ReplaceAssetAction")
        AABB = self._find_msg("AABB")

        scene_ids = ([sid for sid, _off, _t0, _t1 in self._handoff]
                     if self._handoff else [self._scene_id])
        plys = self._fit_replacements_to_vram(doc, plys, scene_ids)
        # NAVSAFE_REPLACE_SCOPE restricts where a replacement is
        # applied: "source" applies it only in each asset's source
        # window, "first" only in the scene that is live at setup. By
        # default it is applied in every scene holding the track. In
        # rare bundles a replacement in some window makes that window
        # fail to render; a restricted scope avoids that, but the run
        # is then not comparable with an unrestricted one.
        _scope = os.environ.get("NAVSAFE_REPLACE_SCOPE", "").strip().lower()
        _src_of = {tid: (a or {}).get("source_window")
                   for tid, a in (doc.get("assets") or {}).items()} \
            if isinstance(doc, dict) else {}
        if required and _scope:
            raise RuntimeError("Required harvester takeover cannot use partial NAVSAFE_REPLACE_SCOPE")
        applied: Dict[str, int] = {}
        applied_tracks = {}
        for sid in scene_ids:
            here = self._dyn_track_ids_by_scene.get(sid, set())
            sizes = self._dyn_track_size_by_scene.get(sid, {})
            actions = []
            for tid in sorted(here & set(plys)):
                if _scope == "first" and sid != scene_ids[0]:
                    continue
                if _scope == "source" and _src_of.get(tid) not in (None, sid):
                    continue
                kw = {"original_id": tid, "replacement_id": plys[tid]}
                sz = sizes.get(tid)
                if sz and all(v > 0.0 for v in sz):
                    kw["object_size"] = AABB(size_x=sz[0], size_y=sz[1], size_z=sz[2])
                actions.append(Action(**kw))
            if not actions:
                continue
            resp = self._stub.edit_assets(
                Req(scene_id=sid, replace=actions), timeout=self._timeout)
            if not getattr(resp, "success", True):
                raise RuntimeError(
                    f"edit_assets replace rejected for scene {sid}: "
                    f"{getattr(resp, 'message', '') or resp}")
            self._replaced_asset_ids.update(a.original_id for a in actions)
            applied[sid] = len(actions)
            applied_tracks[sid] = [a.original_id for a in actions]
        if not applied:
            raise RuntimeError(
                f"asset-harvester replace: none of the {len(plys)} manifest "
                f"track id(s) exist in any served scene ({', '.join(scene_ids)}). "
                f"The bank at {spec} was harvested from a different "
                f"reconstruction than the one being rendered.")
        if required:
            replaced = set().union(*(set(v) for v in applied_tracks.values()))
            missing = required - replaced
            if missing:
                raise RuntimeError(
                    "Takeover was not replaced in any scene: " + str(sorted(missing)))
            report = {"manifest": str(spec), "required_tracks": sorted(required),
                      "budget_instances": int(os.environ.get("NUREC_GRPC_ASSET_REPLACE_MAX", "10") or 0),
                      "applied_tracks_by_scene": applied_tracks,
                      "asset_by_track": {t: plys[t] for t in sorted(required)}}
            logger.warning("nurec_grpc: verified original-track harvester takeover: %s", report)
            audit_path = os.environ.get("NUREC_GRPC_ASSET_REPLACE_REPORT")
            if audit_path:
                import json
                audit = Path(audit_path)
                audit.parent.mkdir(parents=True, exist_ok=True)
                audit.write_text(json.dumps(report, indent=2))
        logger.info("nurec_grpc: replaced %d baked actor(s) with harvested "
                    "assets across %d scene(s): %s",
                    len(self._replaced_asset_ids), len(applied), applied)

    def _fit_replacements_to_vram(self, doc, plys, scene_ids):
        """Drop the far replacements that will not fit on the card.

        ``NUREC_GRPC_ASSET_REPLACE_MAX`` is a budget in **instances**, not in
        manifest entries, because that is the unit VRAM is spent in: a 20 s
        scenario is served as four 5 s scenes and a car that appears in three
        of them is loaded three times. Counting entries instead is what let a
        cap of 12 mean anywhere from 12 to 48 resident assets, and 9 of 51
        scenarios then died mid-episode with ``CUDA out of memory`` while their
        un-replaced halves, on the same card, finished clean.

        Nearest-first by closest ego approach, and it STOPS at the first actor
        that does not fit rather than skipping to a cheaper one further away:
        "the nearest N that fit" is a rule you can state, whereas best-fit
        packing produces a set nobody can predict from the manifest.

        The default is a REAL budget rather than "no limit", because a bank
        that does not fit is the normal case, not the exceptional one: harvest
        keeps up to 10 tracks per scenario and each is resident in every served
        scene that holds it, so an ordinary bank asks for more than the card
        has. Left unlimited, that surfaces as ``CUDA out of memory`` twenty
        minutes into an episode with the frames already written, rather than as
        a line in the log saying which far actors stayed baked. Set ``0``
        explicitly for no limit; raise it on a card bigger than 24 GB.

        Returns the surviving ``{track: ply}``.
        """
        cap = int(os.environ.get("NUREC_GRPC_ASSET_REPLACE_MAX", "10") or 0)
        required = getattr(self, "_required_harvest_tracks", set())
        if required:
            from navsafe.render.harvest_takeover import select_takeover_replacements
            return select_takeover_replacements(doc, plys, self._dyn_track_ids_by_scene,
                                                scene_ids, cap, required)
        if cap <= 0:
            return plys

        def instances(tid):
            return sum(1 for sid in scene_ids
                       if tid in self._dyn_track_ids_by_scene.get(sid, set()))

        near = sorted(((t, r) for t, r in doc["assets"].items() if t in plys),
                      key=lambda kv: kv[1].get("min_ego_dist_m") or 1e9)
        kept, used, dropped = set(), 0, []
        for tid, rec in near:
            n = instances(tid)
            if n == 0:
                continue          # in the manifest, in no served scene: free
            if used + n > cap:
                dropped = [t for t, _ in near if t not in kept]
                break
            kept.add(tid)
            used += n
        if not dropped:
            return plys
        far = doc["assets"][dropped[0]].get("min_ego_dist_m")
        logger.warning(
            "nurec_grpc: NUREC_GRPC_ASSET_REPLACE_MAX=%d instance(s) — "
            "replacing the %d nearest of %d harvested actors (%d instance(s) "
            "across %d scene(s)); %d further away stay baked "
            "(nearest dropped: %s m)", cap, len(kept), len(plys), used,
            len(scene_ids), len(dropped), far)
        return {t: p for t, p in plys.items() if t in kept}

    # ------------------------------------------------------------------
    # Classes to try, in order, when the server refuses the requested one. A
    # rigid-bodied asset is worth trying as a vehicle before a bicycle; the
    # deformable layer (PEDESTRIAN) is last because it accepts arbitrary
    # geometry and would otherwise mask a wrong first guess.
    _INSERT_CLASS_FALLBACKS = ("VEHICLE", "BICYCLE", "CYCLIST", "PEDESTRIAN")
    #: Classes that may be NAMED for a scene whose own tracks never use
    #: them. The road-user members every label enum carries; anything
    #: else has to be observed before it can be asked for.
    _SYNTHESISABLE_CLASSES = ("VEHICLE", "PEDESTRIAN", "BICYCLE", "CYCLIST")

    def _class_literal(self, keyword: str) -> "str | None":
        """The scene's own name for ``keyword``, or one built from its enum.

        The classes a recon will accept an insert AS are the actor layers it
        was TRAINED with, and nothing in the API reports them — the only list
        is in the server's own log. What the client can enumerate is the
        classes this scene's LOGGED TRACKS use, which is a different set: a
        clip whose recorded traffic is all vehicles can still have a trained
        pedestrian layer. Offering only observed classes therefore never tried
        PEDESTRIAN on such a clip, and R-3's walkers went in as VEHICLE and
        were refused outright (45ebc34cbe405c3e, 8164612a623156ae: "Failed to
        insert track_id='navsafe_ped_1#ph0'", whole insert dropped, empty
        pavement rendered while sim state kept the walkers).

        The literal has to parse against the artifact's label enum, so the
        prefix is taken from a class the scene demonstrably uses
        (``NuPlanBoxDetectionLabel.VEHICLE`` -> ``NuPlanBoxDetectionLabel.``)
        and only the member name is supplied.
        """
        for sc, _n in sorted(self._dyn_sem_counter.items(), key=lambda kv: -kv[1]):
            if keyword in sc.upper() and "GENERIC_OBJECT" not in sc.upper():
                return sc
        if keyword not in self._SYNTHESISABLE_CLASSES:
            # Only the road-user classes every label enum carries. Inventing a
            # member the enum lacks (WOD has no CONE) fails the whole insert on
            # parse, which is the failure this is here to avoid.
            return None
        for sc, _n in sorted(self._dyn_sem_counter.items(), key=lambda kv: -kv[1]):
            if "." not in sc:
                continue
            enum, member = sc.rsplit(".", 1)
            # Keep the enum's own member style: WOD writes TYPE_VEHICLE, nuPlan
            # writes VEHICLE, and only one of the two parses per artifact.
            stem = "TYPE_" if member.upper().startswith("TYPE_") else ""
            return f"{enum}.{stem}{keyword}"
        return None

    def _edit_assets_with_class_fallback(self, Req, sid, inserts, rebuild):
        """``edit_assets``, retrying under a different ``semantic_class``.

        Which classes a scene will accept an insert AS is a property of the
        reconstruction — it is the set of actor layers that recon was trained
        with — and it is NOT the set of classes its logged tracks use. A clip
        full of logged pedestrians can still refuse a PEDESTRIAN insert because
        it only ever trained ``dynamic_rigids``::

            Insertion of NuPlanBoxDetectionLabel.PEDESTRIAN for dynamic_rigids
            is not allowed. Allowed label classes: ['...VEHICLE', '...BICYCLE']

        That list is only in the SERVER's log, never in the response, so it
        cannot be read and obeyed — the one thing the client can do is offer
        another class and see. Without this the whole insert is dropped, the
        episode renders an empty road, and sim state still scores against an
        actor nobody can see: a passing run with an invisible hazard, which is
        worse than a failure.

        The class only picks the layer; the gaussians are what render, so an
        asset inserted under a different class still looks like itself.

        Returns the response and the (possibly re-classed) insert list.
        """
        def attempt(tracks):
            """The RPC either answers ``success=False`` or raises, depending on
            where in the server the rejection happens. Both are just 'no'."""
            try:
                r = self._stub.edit_assets(
                    Req(scene_id=sid, insert=tracks), timeout=self._timeout)
                return r, bool(getattr(r, "success", True))
            except Exception as exc:  # noqa: BLE001 — gRPC error type varies
                return exc, False

        resp, ok = attempt(inserts)
        if ok:
            return resp, inserts

        Track = rebuild["Track"]
        tried = {getattr(t, "semantic_class", "") for t in inserts}
        observed = [sc for sc, _ in sorted(self._dyn_sem_counter.items(),
                                           key=lambda kv: -kv[1])
                    if "GENERIC_OBJECT" not in sc.upper()]
        for keyword in self._INSERT_CLASS_FALLBACKS:
            sem = self._class_literal(keyword)
            if sem is None or sem in tried:
                continue
            tried.add(sem)
            retry = [Track(id=i, semantic_class=sem, trajectory=tr,
                           object_size=osz, asset_id=aid)
                     for i, tr, osz, aid in rebuild["specs"]]
            logger.warning(
                "nurec_grpc: scene %s refused the insert under %s; retrying as "
                "%s (the recon's actor layers, not its logged tracks, decide "
                "this)", sid, sorted(tried - {sem}), sem)
            resp, ok = attempt(retry)
            if ok:
                logger.info("nurec_grpc: insert accepted as semantic_class=%r", sem)
                return resp, retry
        # Out of classes: hand back the last answer so the caller raises with
        # the server's own words rather than a summary of them.
        if isinstance(resp, Exception):
            raise resp
        return resp, inserts

    # ------------------------------------------------------------------
    def _recorded_tilt(self, t_us: Optional[int]) -> Optional[np.ndarray]:
        """The recorded rig's PITCH AND ROLL at ``t_us``, yaw removed.

        ``ego_state`` carries position and heading and nothing else -- the
        simulator's ego is planar -- so a rig built from it renders perfectly
        level. The reconstruction was not fit level: on 3067f3d3d5a75989s1 its
        rig pitches -1.025 deg and rolls -0.948 deg at frame 0, and at fx=1545
        a 1.025 deg pitch is 27.6 px of vertical image shift.

        The attitude comes from the server's own training rig trajectory --
        the same poses `nre render --replicate-training-views` walks, and
        bit-identical to the Arrow's ego quaternion (max component difference
        6.1e-09). Only the TILT is taken: yaw stays the simulator's heading,
        because a closed-loop policy may steer differently from the log, while
        pitch and roll describe the road surface and it may not.

        Precedent: ``_ego_z_now`` already takes the rig's z from the recorded
        track rather than from ego_state. This is the same data, one component
        further.

        Returns None -- and the caller stays level -- only when the server has
        no trajectory for the scene.
        """
        sid = self._scene_id
        if sid is None or t_us is None:
            return None
        if sid not in self._tilt_cache:
            self._tilt_cache[sid] = self._fetch_tilt_table(sid)
        table = self._tilt_cache[sid]
        if table is None:
            return None
        ts, tilts = table
        i = int(np.argmin(np.abs(ts - int(t_us))))
        return tilts[i]

    @staticmethod
    def _strip_yaw(mats: np.ndarray) -> np.ndarray:
        """``Rz(-yaw) @ R`` for a stack of rotations — the pitch/roll that is left.

        ``yaw = atan2(R[1,0], R[0,0])`` is the exact inverse of the Rz the
        caller applies, so ``Rz(yaw) @ _strip_yaw(R) == R``. Do this on the
        matrix, never via euler angles: ``as_euler("xyz")`` and
        ``from_euler("zyx", ...)`` are not an inverse pair, and composing them
        silently substitutes a different attitude.
        """
        yaw = np.arctan2(mats[:, 1, 0], mats[:, 0, 0])
        c, s = np.cos(-yaw), np.sin(-yaw)
        rz_inv = np.zeros_like(mats)
        rz_inv[:, 0, 0] = c
        rz_inv[:, 0, 1] = -s
        rz_inv[:, 1, 0] = s
        rz_inv[:, 1, 1] = c
        rz_inv[:, 2, 2] = 1.0
        return rz_inv @ mats

    def _fetch_tilt_table(self, scene_id: str):
        """(timestamps_us, Nx3x3 tilt) for a scene, or None if unavailable.

        Yaw is stripped with ``Rz(-yaw)`` where ``yaw = atan2(R[1,0], R[0,0])``
        -- the exact inverse of the Rz the caller then applies, so the two
        compose back to the recorded attitude when the policy is on the logged
        heading.
        """
        from scipy.spatial.transform import Rotation as R
        try:
            g = self._import()
            ret = self._stub.get_available_trajectories(
                g["sspb"].AvailableTrajectoriesRequest(scene_id=scene_id),
                timeout=self._timeout)
            poses = list(ret.available_trajectories[0].trajectory.poses)
        except Exception as exc:
            logger.warning("nurec_grpc: no training trajectory for %s (%s) — "
                           "the rig will render level", scene_id, exc)
            return None
        if not poses:
            return None
        ts = np.array([int(p.timestamp_us) for p in poses], np.int64)
        quats = np.array([[p.pose.quat.x, p.pose.quat.y, p.pose.quat.z,
                           p.pose.quat.w] for p in poses], np.float64)
        mats = R.from_quat(quats).as_matrix()
        tilts = self._strip_yaw(mats)
        pr = R.from_matrix(tilts).as_euler("ZYX", degrees=True)
        logger.info("nurec_grpc: rig attitude from the training trajectory for "
                    "%s (%d poses, frame-0 pitch %+.3f deg roll %+.3f deg)",
                    scene_id, len(poses), pr[0, 1], pr[0, 2])
        return ts, tilts

    def _ego_rig_to_world(self, ego_state: dict,
                          t_us: Optional[int] = None) -> np.ndarray:
        """4x4 ego rig-to-world in NuRec world (FLU: x-fwd, y-left, z-up).

        Maps the env's recentered local ego xy into NuRec world by anchoring to
        the recorded frame-0 pose (see _sd_pos0/_eval_pos0), then subtracting the
        reconstruction origin offset. Orientation is the simulator's heading
        composed with the recorded pitch/roll (see :meth:`_recorded_tilt`).
        """
        pos = np.asarray(ego_state.get("position", (0.0, 0.0, 0.0)), np.float64)
        heading = float(ego_state.get("heading", 0.0))
        # The ego track (both the sd pickle and the eval) is already recentered
        # (frame-0 xy == origin, z preserved). NuRec world = recentered - offset.
        ox, oy, _ = self._origin_offset.tolist()
        e0 = self._eval_pos0 if self._eval_pos0 is not None else np.zeros(2)
        nre_x = (float(pos[0]) - float(e0[0])) - ox
        nre_y = (float(pos[1]) - float(e0[1])) - oy
        th = heading
        Rz = np.array([[math.cos(th), -math.sin(th), 0.0],
                       [math.sin(th),  math.cos(th), 0.0],
                       [0.0, 0.0, 1.0]])
        lever = self._ego_anchor_lever(th)
        if lever is not None:
            # The pose is that of the bounding-box centre, while
            # the camera extrinsics are relative to the rig
            # origin. Step back along the vehicle axis to the rig
            # origin.
            nre_x -= float(Rz[0, 0] * lever[0] + Rz[0, 1] * lever[1])
            nre_y -= float(Rz[1, 0] * lever[0] + Rz[1, 1] * lever[1])
        m = np.eye(4)
        tilt = self._recorded_tilt(t_us)
        m[:3, :3] = Rz if tilt is None else Rz @ tilt
        m[:3, 3] = [nre_x, nre_y, self._ego_z_now(pos[:2])]
        return m

    def _ego_anchor_lever(self, heading: float) -> Optional[np.ndarray]:
        """Centre-to-rig-origin offset in the EGO frame, metres, or None.

        The scenario's ego x/y is the BOUNDING-BOX CENTRE
        (``py123d_training_extractor._extract_ego`` reads ``center_se3``
        first), while the reconstruction's origin sidecar records the nuPlan
        ego_pose -- the Arrow's ``imu_se3``, bit-identical to the training rig
        trajectory (max component difference 6.1e-09). The two sit 1.4527 m
        apart along the vehicle axis on 3067f3d3d5a75989, so ``camera_to_rig``
        has to be composed onto the rig origin, not onto the centre.

        The lever is derived, not fitted: ``origin_offset`` already carries
        ``imu_0 - centre_0`` in world axes, and rotating it into the ego frame
        by the frame-0 heading gives the body-frame offset -- which then
        rotates back with the heading at every step, as a lever arm must. A
        world-frame constant would only agree at frame 0.

        An ``origin_offset`` of zero makes this a no-op: an absolute-frame
        reconstruction has no centre/origin split to correct.
        """
        if self._anchor_lever is None:
            # s1's offset is imu_0 - centre_0. Later sub-clips re-reference to
            # their own window start, so only the first one measures the
            # vehicle; the lever is a property of the car, not of the clip.
            base = (self._handoff[0][1] if self._handoff
                    else self._origin_offset)
            d_world = -np.asarray(base, np.float64)[:2]
            if not np.any(np.abs(d_world) > 1e-9):
                self._anchor_lever = np.zeros(2)
            else:
                c, s_ = math.cos(heading), math.sin(heading)
                self._anchor_lever = np.array(
                    [c * d_world[0] + s_ * d_world[1],
                     -s_ * d_world[0] + c * d_world[1]], np.float64)
        if not self._anchor_logged:
            self._anchor_logged = True
            logger.info("nurec_grpc: ego anchor centre->rig lever = "
                        "[%+.4f, %+.4f] m in the ego frame (|%.4f|); the "
                        "scenario ego is the bbox centre, the recon origin is "
                        "the nuPlan ego_pose",
                        self._anchor_lever[0], self._anchor_lever[1],
                        float(np.linalg.norm(self._anchor_lever)))
        return self._anchor_lever

    def _pose_to_se3(self, p) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R
        m = np.eye(4)
        m[:3, :3] = R.from_quat([p.quat.x, p.quat.y, p.quat.z, p.quat.w]).as_matrix()
        m[:3, 3] = [p.vec.x, p.vec.y, p.vec.z]
        return m

    def _se3_to_pose(self, m: np.ndarray):
        from scipy.spatial.transform import Rotation as R
        g = self._g
        q = R.from_matrix(m[:3, :3]).as_quat(canonical=False)
        return g["Pose"](vec=g["Vec3"](x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
                         quat=g["Quat"](x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])))

    @staticmethod
    def _decode(image_bytes: bytes) -> Optional[np.ndarray]:
        if not image_bytes:
            return None
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    @staticmethod
    def _find_sdc(scenario_data: dict, meta: dict):
        """Locate the ego (self-driving car) track id."""
        tracks = scenario_data.get("tracks", {}) if scenario_data else {}
        sdc = meta.get("sdc_id") or scenario_data.get("sdc_id")
        if sdc is not None and sdc in tracks:
            return sdc
        for k, v in tracks.items():
            if v.get("is_sdc") or str(v.get("type", "")).lower() in ("ego", "sdc"):
                return k
        return next(iter(tracks), None)

    def _ego_z_now(self, pos_xy=None) -> float:
        """Ground altitude for the current camera/actor pose.

        Altitude is a function of POSITION, not time: when a live xy is given,
        return the z of the nearest recorded-track point, searched in a
        +-50-frame window around the replay cursor (so self-crossing routes /
        overpasses cannot snap to the wrong branch). Exact in replay (nearest
        point == the cursor frame); in closed loop it follows where the ego
        actually IS — a slower/faster policy sits tens of meters from the
        recorded position at the same step, which on a grade is a visible
        camera-height error. Without xy, falls back to cursor indexing; without
        the track, to the frame-0 scalar. Riding the altitude profile at all is
        what keeps the rendered ground attached to the overlay's ground plane
        (the old frame-0 pinning floated it by >100 px near-field at dz~0.8 m).
        """
        z_arr = self._sd_ego_z_arr
        if z_arr is None or not len(z_arr):
            return self._sd_ego_z
        i = min(max(self._cur_step, 0), len(z_arr) - 1)
        xy_arr = self._sd_ego_xy_arr
        if pos_xy is not None and xy_arr is not None and len(xy_arr) == len(z_arr):
            lo, hi = max(0, i - 50), min(len(z_arr), i + 51)
            d = xy_arr[lo:hi] - np.asarray(pos_xy, np.float64)
            i = lo + int(np.argmin(np.einsum("ij,ij->i", d, d)))
        return float(z_arr[i])

    @classmethod
    def _read_sd_ego_track(cls, scenario_data: dict, meta: dict):
        """Per-frame recorded ego (z, xy) arrays, (None, None) when unavailable.

        The xy is in the scenario's own frame — the same frame the runtime
        ego_state positions use (the env poses come from these tracks), so a
        live position can be compared against it directly.
        """
        try:
            sdc = cls._find_sdc(scenario_data, meta)
            pos = np.asarray(scenario_data["tracks"][sdc]["state"]["position"],
                             np.float64)
            if pos.ndim >= 2 and pos.shape[-1] >= 3:
                pos = pos.reshape(-1, pos.shape[-1])
                return pos[:, 2].copy(), pos[:, :2].copy()
        except Exception:
            pass
        return None, None

    @classmethod
    def _read_sd_ego_z(cls, scenario_data: dict, meta: dict) -> float:
        try:
            sdc = cls._find_sdc(scenario_data, meta)
            pos = scenario_data["tracks"][sdc]["state"]["position"]
            return float(np.asarray(pos)[..., 2].reshape(-1)[0])
        except Exception:
            return float(meta.get("sd_ego_z", 0.0) or 0.0)

    @classmethod
    def _read_sd_pos0(cls, scenario_data: dict, meta: dict) -> Optional[np.ndarray]:
        """Recorded sd-world ego xy at frame 0 (the map anchor)."""
        try:
            sdc = cls._find_sdc(scenario_data, meta)
            pos = np.asarray(scenario_data["tracks"][sdc]["state"]["position"])
            p0 = pos[0] if pos.ndim == 2 else pos
            return np.asarray([float(p0[0]), float(p0[1])], np.float64)
        except Exception:
            return None

    # ------------------------------------------------------------------
    def update_agents(self, agent_states: list) -> None:
        # Store the env's per-frame symbolic agent poses; _build_dynamic_objects
        # forwards them to the server as RGBRenderRequest.dynamic_objects so
        # actors are driven by the sim. get_camera_images also refreshes this
        # from its argument.
        self._agent_states = agent_states or []

    def set_timestep(self, sim_step: int) -> None:
        self._cur_step = int(sim_step)

    def _restore_server_edits(self) -> None:
        """Roll back all edit_assets state on the server.

        Whole-model parameter snapshot rollback ("undoes any asset editing
        operations") — the per-eval isolation mechanism for the warm shared
        server. Every handoff scene is rolled back, because every one of them
        was inserted into; restoring only ``_scene_id`` would leak the asset
        into later evals of the other subclips.
        """
        Req = self._find_msg("RestoreModelParametersRequest")
        if "scene_id" not in self._proto_field_names(Req):
            self._stub.restore_model_parameters(Req(), timeout=self._timeout)
            return
        scene_ids = ([sid for sid, _off, _t0, _t1 in self._handoff]
                     if self._handoff else [self._scene_id])
        for sid in scene_ids:
            self._stub.restore_model_parameters(
                Req(scene_id=sid), timeout=self._timeout)

    def close(self) -> None:
        # Undo this run's asset edits on the server. Inserted tracks
        # and replaced actors are restored by the same call. This
        # requires close() to be called; edits left by a run that was
        # killed are cleared at the next setup.
        edited = self._inserted_asset_ids or self._replaced_asset_ids
        if edited and self._stub is not None and self._g is not None:
            try:
                self._restore_server_edits()
                logger.info("nurec_grpc: restore_model_parameters undid %d "
                            "inserted and %d replaced asset track(s)",
                            len(self._inserted_asset_ids),
                            len(self._replaced_asset_ids))
            except Exception as exc:  # pragma: no cover — live only
                logger.warning("nurec_grpc: restore_model_parameters failed (%s); "
                               "edited assets may persist on the warm server "
                               "and pollute later evals of scene %s",
                               exc, self._scene_id)
            self._inserted_asset_ids = set()
            self._replaced_asset_ids = set()
            self._injected_track_states = {}
        try:
            if self._channel is not None:
                self._channel.close()
        except Exception:
            pass


__all__ = ["NuRecGrpcSceneRenderer"]
