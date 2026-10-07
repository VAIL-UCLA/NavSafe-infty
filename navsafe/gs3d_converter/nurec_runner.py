# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA Neural Reconstruction Engine (NRE / "NuRec") Docker pipeline wrappers.

This module drives the **real, public NRE containers** documented at
https://github.com/NVIDIA/nurec-skills and on the NGC catalog:

    nvcr.io/nvidia/nre/nre-ga:26.04        — train, render, export, evaluate
    nvcr.io/nvidia/nre/nre-tools-ga:26.04  — auxiliary-data generation

GA (``-ga``) are the **publicly available** repos — readable with any NGC API
key, no purchase. (The plain ``nre`` / ``nre-tools`` repos are the *private,
enterprise-entitled* variants and return ``DENIED: Payment Required``; do not
use them here.)

.. warning::
   **Tag pulls fail on Docker's classic image store.** Each tag (``26.04`` /
   ``latest``) is an OCI image **index** bundling the linux/amd64 image with an
   ``unknown/unknown`` cosign attestation child. ``docker pull ...:26.04`` then
   fails with ``error from registry: Incorrect Repository Format`` (and
   ``--platform`` does not help). Two fixes:

   1. **Pull by digest** (what this module does — see ``NRE_IMAGE`` below)::

          docker pull nvcr.io/nvidia/nre/nre-ga@sha256:6e0caa70a914...

   2. Or enable the **containerd image store** (then tags work normally)::

          # /etc/docker/daemon.json
          { "features": { "containerd-snapshotter": true } }   # then restart docker

   Docs: https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html

It replaces an earlier placeholder that invoked a non-existent
``nvcr.io/nvidia/nurec`` image with imaginary ``nurec aux|train|export
--temporal`` sub-commands. None of that ever matched a shipping container.

Prerequisites
-------------
* Docker + NVIDIA Container Toolkit (``docker run --gpus all`` must work).
* ``docker login nvcr.io`` (username ``$oauthtoken``, password = NGC API key).
* ``NGC_API_KEY`` exported in the environment — passed into every container.
  The ``-ga`` repos are public; any valid NGC API key can pull them.
* RAM/shm: the toolkit uses ``--shm-size=64g`` for train/export and ``2g`` for
  the aux tools.

Pipeline → real sub-command mapping
-----------------------------------
* :func:`run_aux`    → ``nre-tools ncore-aux-data`` (semantic seg, lidar
  seg/visibility, ego mask, metadata). Aux signals are written back into the
  NCore shards in place so training finds them via ``dataset.aux_data=True``.
* :func:`run_train`  → ``nre <config> mode=train dataset.path=... out_dir=...``
  Produces ``<out_dir>/<RUN-ID>/`` with ``config/parsed.yaml``,
  ``checkpoints/last.ckpt``, ``usd-out/last.usdz`` (+ ``map.xodr``,
  ``sequence_tracks.json``, …), and ``val/metrics.yaml``.
* :func:`run_export` → ``nre export-gaussian-plys`` (per-layer 3DGS ``.ply``)
  plus ``nre export-sequence-tracks`` (actor tracks JSON). Returns a
  :class:`NuRecExport` describing the real artifacts.

Every function checks whether its output already exists before launching Docker,
so reruns after a partial failure are idempotent. All mounts use absolute paths.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── container images ──────────────────────────────────────────────────────────

# Pinned by digest, NOT by tag — see the module note. Each tag (``26.04`` /
# ``latest``) is an OCI image index bundling the linux/amd64 image with an
# ``unknown/unknown`` cosign attestation; Docker's classic (non-containerd)
# image store rejects that index with "error from registry: Incorrect Repository
# Format". The digests below point straight at the linux/amd64 image manifest,
# so ``docker run`` works regardless of image-store config.
#
# These are the linux/amd64 children of the ``:26.04`` (NRE release 26.04) tags.
# To refresh for a new release: read the tag's image index and take the
# architecture=="amd64" child digest (see git history / repo notes for the
# one-liner), then update both constants here.
NRE_IMAGE = "nvcr.io/nvidia/nre/nre-ga@sha256:6e0caa70a9148490552520c3dde9ee665c8d094ca10b814601cb8bb306567c90"
NRE_TOOLS_IMAGE = "nvcr.io/nvidia/nre/nre-tools-ga@sha256:46d78256219a6541601fd16517c7edca3e7a6ec9634f1858f7977279bd98f9ff"

# ── default Hydra training config (container-bundled) ─────────────────────────
# Exact ``--config-name`` from the NRE AV docs:
#   https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html
# NCoreBridge always emits NCore/PAI shards (even for WOD source), so NRE always
# consumes the NCore format — the car2sim (Hyperion) recipe applies regardless of
# the original dataset. Override per-clip via ``run_train(config_name=...)``.
_DEFAULT_CONFIG = {
    "ncore": "configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml",
    # WOD -> the Waymo DYNAMIC recipe: it builds real per-actor Gaussian nodes
    # (background + dynamic_rigids + dynamic_deformables) so moving agents are
    # separable/editable. car2sim_6cam is a static recipe (background + road
    # only) that bakes agents into the background — verified it produces
    # "Has rigid tracks: False". run_train applies recipe-specific overrides.
    "wod":   "configs/apps/AV/Waymo/3dgut_dynamic.yaml",
    # NavSim/OpenScene IS nuPlan data. car2sim_6cam is the best recipe on it
    # (dynamic_rigids + dynamic_deformables + road + MCMC + temporal appearance +
    # 40k). It DOES have dynamic layers — the "Has rigid tracks: False" note above
    # is only the DEFAULT config (NVIDIA class names don't match our labels);
    # run_train's class routing populates them. run_train also adds the car2sim
    # road-init swap + ground-mesh disable, and (static_separation) uses the
    # car2sim_6cam_static overlay.
    "navsim": "configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml",
}

# ── per-dataset Gaussian-layer label routing ──────────────────────────────────
# NCoreBridge stores each cuboid's class_id as ``str(det.attributes.label)`` — the
# py123d label enum for that dataset. NRE routes each tracked object into a layer
# by matching that string against the layer's ``tracks.label_classes``. The enums
# differ per dataset, so this MUST be specified per dataset. Moving-vs-static
# within ``static_rigids`` is decided automatically by pose speed
# (``dataset.cuboid_tracks_params``); we only list which CLASSES are eligible.
#   dynamic_rigids       — moving rigid actors (vehicles, cyclists)
#   dynamic_deformables  — non-rigid actors (pedestrians)
#   static_rigids        — parked/static boxed objects (only with static_separation)
_LAYER_LABEL_CLASSES: dict = {
    "wod": {
        "dynamic_rigids":      ["WODPerceptionBoxDetectionLabel.TYPE_VEHICLE"],
        "dynamic_deformables": ["WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN"],
        "static_rigids":       ["WODPerceptionBoxDetectionLabel.TYPE_VEHICLE",
                                "WODPerceptionBoxDetectionLabel.TYPE_SIGN"],
    },
    # NavSim IS nuPlan data → NavsimParser emits NuPlanBoxDetectionLabel.
    "navsim": {
        "dynamic_rigids":      ["NuPlanBoxDetectionLabel.VEHICLE",
                                "NuPlanBoxDetectionLabel.BICYCLE"],
        "dynamic_deformables": ["NuPlanBoxDetectionLabel.PEDESTRIAN"],
        "static_rigids":       ["NuPlanBoxDetectionLabel.VEHICLE",
                                "NuPlanBoxDetectionLabel.TRAFFIC_CONE",
                                "NuPlanBoxDetectionLabel.BARRIER",
                                "NuPlanBoxDetectionLabel.CZONE_SIGN",
                                "NuPlanBoxDetectionLabel.GENERIC_OBJECT"],
    },
}


def _label_classes(dataset: str, layer: str) -> str:
    """Hydra list literal of label classes for a layer, e.g. ``[A,B]``.

    Falls back to the WOD taxonomy for datasets without an explicit entry
    (ncore/av2 currently reuse the WOD-style routing — add entries as needed).
    """
    routing = _LAYER_LABEL_CLASSES.get(dataset, _LAYER_LABEL_CLASSES["wod"])
    return "[" + ",".join(routing[layer]) + "]"

# Container-internal mount points used by NRE.
_C_DATASET = "/workdir/dataset"
_C_OUTPUT = "/workdir/output"

# Static-separation overlay recipe shipped with the package, and the container
# config dir Hydra resolves it from. The NRE click entrypoint rejects
# --config-dir, so run_train bind-mounts the recipe into the bundled tree.
_STATIC_RECIPE = Path(__file__).parent / "configs" / "3dgut_dynamic_static.yaml"
_C_WAYMO_CFG_DIR = (
    "/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles"
    "/_main/configs/apps/AV/Waymo"
)
# car2sim static-separation overlay (adds a static_rigids layer on top of
# car2sim_6cam so parked/static boxed objects are their own editable node), and
# the container config dir it is bind-mounted into.
_CAR2SIM_STATIC_RECIPE = Path(__file__).parent / "configs" / "car2sim_6cam_static.yaml"
_C_HYPERION_CFG_DIR = (
    "/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles"
    "/_main/configs/apps/prod/Hyperion-8.1"
)


# ── output dataclass ──────────────────────────────────────────────────────────

@dataclass
class NuRecExport:
    """Paths and metadata produced by :func:`run_export` (real NRE artifacts)."""

    # ── real NRE outputs ──────────────────────────────────────────────────────
    run_dir:              Path                  # <out_dir>/<RUN-ID>/
    usdz_path:            Optional[Path] = None  # run_dir/usd-out/last.usdz
    parsed_config:        Optional[Path] = None  # run_dir/config/parsed.yaml
    checkpoint:           Optional[Path] = None  # run_dir/checkpoints/last.ckpt
    plys_dir:             Optional[Path] = None  # run_dir/plys/ (export-gaussian-plys)
    ply_paths:            List[Path] = field(default_factory=list)  # per-layer PLYs
    sequence_tracks_json: Optional[Path] = None  # export-sequence-tracks (actors)
    ncore_tracks_json:    Optional[Path] = None  # export-ncore-tracks (per-frame poses)
    map_xodr:             Optional[Path] = None  # run_dir/usd-out/map.xodr if present



# ── pipeline functions ────────────────────────────────────────────────────────

def run_aux(
    clip_dir: Path,
    work_dir: Path,
    camera_ids: Optional[List[str]] = None,
    lidar_ids: Optional[List[str]] = None,
    segmentation_backend: str = "mask2former",
    lidar_seg_camvis: bool = True,
    ego_mask: bool = True,
    depth_backend: str = "none",
    zarr_store_type: str = "itar",
    nurec_image: str = NRE_TOOLS_IMAGE,
    extra_args: Optional[List[str]] = None,
) -> Path:
    """Generate auxiliary training signals via ``nre-tools ncore-aux-data``.

    NRE training (``dataset.aux_data=True``) consumes per-frame aux signals:
    semantic segmentation (sky / dynamic masks), lidar segmentation +
    point-in-camera visibility, and an ego self-occlusion mask. ``ncore-aux-data``
    writes these back into the NCore shards under ``--output-dir``; we point that
    at the clip directory itself so ``run_train`` finds them in place.

    The exact CLI is from ``nre-tools-ga:26.04 ncore-aux-data --help``.

    Parameters
    ----------
    clip_dir:
        NCore clip directory from :class:`~navsafe.gs3d_converter.ncore_bridge.NCoreBridge`,
        containing ``pai_<clip>.json`` and its ``.zarr.itar`` shard(s).
    work_dir:
        Per-clip working directory (used for reruns / sibling NRE outputs).
    camera_ids / lidar_ids:
        Sensors to process (``--camera-id`` / ``--lidar-id``, repeatable).
        ``None`` processes all sensors in the shard.
    segmentation_backend:
        ``"mask2former"`` (semantic seg → sky/dynamic masks) or ``"none"``.
    lidar_seg_camvis:
        Run lidar segmentation + point-in-camera visibility (``--lidar-seg-camvis``).
    ego_mask:
        Estimate the ego self-occlusion mask (``--ego-mask``).
    depth_backend:
        ``"depthanythingv2"`` for monocular depth supervision, or ``"none"``.
    zarr_store_type:
        ``"itar"`` (matches NCoreBridge shards) or ``"directory"``.
    nurec_image:
        ``nre-tools`` image (digest-pinned by default).
    extra_args:
        Extra raw flags appended to the container command.

    Returns
    -------
    Path
        ``clip_dir`` — the dataset directory, now populated with aux signals.
    """
    clip_dir = Path(clip_dir).resolve()
    manifest = _ncore_manifest(clip_dir)

    # Idempotency: skip if aux artifacts already sit beside the shard.
    if any(clip_dir.glob("*.aux.*")):
        logger.info("run_aux: aux artifacts already present in %s, skipping", clip_dir)
        _link_aux_to_manifest_base(clip_dir, manifest)
        return clip_dir

    # Aux historically ran as container root, but root-squashing filesystems
    # (NFS work dirs) deny root writes to the bind mount. NAVSAFE_NUREC_AUX_USER=1
    # pins the host uid:gid instead, matching the train/export stages.
    aux_as_user = os.environ.get("NAVSAFE_NUREC_AUX_USER", "").strip() == "1"
    cmd = _docker_base(shm="2g", with_user=aux_as_user) + [
        "-v", f"{clip_dir}:{_C_DATASET}",          # read-write: aux lands here
        "-v", f"{clip_dir}:{_C_OUTPUT}",
        nurec_image,
        "ncore-aux-data",
        f"--dataset-path={_C_DATASET}/{manifest.name}",
        f"--output-dir={_C_OUTPUT}",
        f"--segmentation-backend={segmentation_backend}",
        f"--zarr-store-type={zarr_store_type}",
        "--store-meta",
    ]
    cmd.append("--lidar-seg-camvis" if lidar_seg_camvis else "--no-lidar-seg-camvis")
    cmd.append("--ego-mask" if ego_mask else "--no-ego-mask")
    if depth_backend and depth_backend != "none":
        cmd.append(f"--depth-backend={depth_backend}")
    for cid in camera_ids or []:
        cmd.append(f"--camera-id={cid}")
    for lid in lidar_ids or []:
        cmd.append(f"--lidar-id={lid}")
    cmd += extra_args or []

    _run(cmd, step="aux")
    _link_aux_to_manifest_base(clip_dir, manifest)
    return clip_dir


def run_train(
    dataset_dir: Path,
    work_dir: Path,
    dataset: str = "ncore",
    config_name: Optional[str] = None,
    camera_ids: Optional[List[str]] = None,
    lidar_ids: Optional[List[str]] = None,
    aux_data: bool = True,
    run_id: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    subsample: int = 2,
    static_separation: bool = False,
    lidar_supervision: bool = False,
    nurec_image: str = NRE_IMAGE,
) -> Path:
    """Train a 3DGUT reconstruction with the ``nre`` container (default ``main``).

    Mirrors the documented "Standard Training" invocation:
    https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html ::

        docker run ... nre-ga:26.04 mode=train out_dir=/workdir/output \\
            --config-name=configs/apps/prod/Hyperion-8.1/car2sim_6cam.yaml \\
            dataset.path=/workdir/dataset/pai_<CLIP>.json \\
            dataset.camera_ids=[...] dataset.lidar_ids=[lidar_top_360fov] \\
            dataset.aux_data=True

    Parameters
    ----------
    dataset_dir:
        Directory with the NCore manifest + shards + aux artifacts
        (the return value of :func:`run_aux`).
    work_dir:
        Outputs are written under ``work_dir/output/<RUN-ID>/``.
    dataset:
        ``"ncore"`` / ``"wod"`` — selects the default ``--config-name`` (both
        currently map to the NCore car2sim recipe; see :data:`_DEFAULT_CONFIG`).
    config_name:
        Override the Hydra ``--config-name`` (e.g. a different camera-count rig).
    camera_ids:
        Sensor camera IDs. Applied to ``dataset.camera_ids`` AND
        ``dataset.train_camera_ids`` / ``dataset.val_camera_ids`` (the recipe
        keeps them as separate lists). ``None`` auto-derives from the NCore
        shard names (e.g. ``camera_pcam_f0``).
    lidar_ids:
        Sensor lidar IDs. Applied to ``dataset.{lidar_ids,train_lidar_ids,
        val_lidar_ids}``. ``None`` auto-derives, falling back to
        ``["lidar_top_360fov"]``.
    aux_data:
        Sets ``dataset.aux_data`` (requires :func:`run_aux` to have run).
    run_id:
        Force the ``<RUN-ID>`` subdir (Hydra ``logger.run_id``). Defaults to the
        dataset directory name for deterministic, idempotent reruns. If NRE
        ignores this key, the actual run dir is discovered after training.
    overrides:
        Extra Hydra ``key=value`` overrides appended verbatim.
    subsample:
        Image subsample factor for camera supervision (pixel sampler + val +
        sequential images). Must divide every camera's width and height.
        ``2`` (default) halves WOD resolution; ``1`` trains at native res.
    static_separation:
        Train with the shipped ``3dgut_dynamic_static`` overlay: STATIC
        tracked objects (parked cars, signs) get their own ``static_rigids``
        Gaussian node instead of baking into ``background``, so every boxed
        object is movable/removable through the ``nurec_grpc`` render
        backend (see ``docs/navsafe_eval.md``). The recipe file is
        bind-mounted into the container config tree automatically.
    nurec_image:
        ``nre`` image (digest-pinned by default).

    Returns
    -------
    Path
        The run directory (``work_dir/output/<RUN-ID>/``, or the discovered dir).
    """
    dataset_dir = Path(dataset_dir).resolve()
    work_dir = Path(work_dir).resolve()
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    rid = _sanitize_run_id(run_id or dataset_dir.name)
    run_dir = output_dir / rid

    # Idempotency: skip if any run dir already has a trained checkpoint.
    existing = _discover_run_dir(output_dir, prefer=run_dir)
    if existing is not None:
        logger.info("run_train: checkpoint exists at %s, skipping", existing)
        return existing

    manifest = _ncore_manifest(dataset_dir)
    cfg = config_name or _DEFAULT_CONFIG.get(dataset, _DEFAULT_CONFIG["ncore"])
    if static_separation and config_name is None:
        cfg = (f"configs/apps/prod/Hyperion-8.1/{_CAR2SIM_STATIC_RECIPE.name}"
               if "car2sim" in cfg else
               f"configs/apps/AV/Waymo/{_STATIC_RECIPE.name}")

    # Sensor IDs: use caller-provided, else derive from the NCore shard names
    # (NCoreBridge names stores ``pai_<clip>.ncore4-<sensor_id>.zarr.itar``).
    disc_cams, disc_lids = _discover_sensor_ids(dataset_dir)
    cams = camera_ids if camera_ids is not None else disc_cams
    lids = lidar_ids if lidar_ids is not None else (disc_lids or ["lidar_top_360fov"])
    logger.info("run_train: camera_ids=%s lidar_ids=%s", cams, lids)

    cam_list = "[" + ",".join(cams) + "]"
    lid_list = "[" + ",".join(lids) + "]"

    # The Waymo 3dgut_dynamic recipe builds per-actor dynamic nodes and needs a
    # different override set than the static car2sim_6cam recipe (class routing,
    # camera-based dynamic seeding, dummy logger; no road layer / no train-val
    # sensor split). Branch on the resolved config name.
    # car2sim (Hyperion) is ALSO a dynamic recipe — it composes dynamic_rigids +
    # dynamic_deformables + a road layer. It needs the dynamic override set (class
    # routing, camera-based seeding) PLUS a road-init swap + ground-mesh disable
    # (added in the is_car2sim block below).
    is_car2sim = "car2sim" in cfg
    is_dynamic = "3dgut_dynamic" in cfg or is_car2sim

    cmd = _docker_base(shm="64g", with_user=True) + [
        "-v", f"{dataset_dir}:{_C_DATASET}",
        "-v", f"{output_dir}:{_C_OUTPUT}",
    ]
    if static_separation:
        # Bind-mount the shipped overlay into Hydra's bundled config tree so
        # --config-name resolves it (the entrypoint rejects --config-dir).
        if is_car2sim:
            cmd += ["-v",
                    f"{_CAR2SIM_STATIC_RECIPE.resolve()}:"
                    f"{_C_HYPERION_CFG_DIR}/{_CAR2SIM_STATIC_RECIPE.name}:ro"]
        else:
            cmd += ["-v",
                    f"{_STATIC_RECIPE.resolve()}:"
                    f"{_C_WAYMO_CFG_DIR}/{_STATIC_RECIPE.name}:ro"]
    cmd += [
        nurec_image,
        f"--config-name={cfg}",
        "mode=train",
    ]
    if is_dynamic:
        # 3dgut_dynamic defaults its logger to wandb, which aborts a
        # non-interactive run with "No API key configured". Force the local dummy
        # logger. Must precede logger.run_id below (switching the logger group
        # resets its subtree, dropping an earlier run_id).
        cmd.append("logger=dummy")
    cmd += [
        f"dataset.path={_C_DATASET}/{manifest.name}",
        f"out_dir={_C_OUTPUT}",
        # Single-quote so Hydra keeps it a string: an all-digit run_id with
        # underscores (e.g. a WOD clip id "1020..._7625_000_7645_000") is
        # otherwise parsed as an int (underscores are digit separators), and
        # logger.dummy.run_id then fails pydantic's str validation.
        f"logger.run_id='{rid}'",
        f"dataset.aux_data={'True' if aux_data else 'False'}",
        # Point every sensor list at the IDs actually in our NCore shard. The
        # recipe defaults assume a different rig, so train/val would otherwise
        # select sensors that don't exist. (train/val *splits* are car2sim-only,
        # added in the branch below — 3dgut_dynamic uses single camera/lidar
        # lists via the Waymo mixin.)
        f"dataset.lidar_ids={lid_list}",
        # Lidar SUPERVISION is opt-in: it requires the NCore shard to carry a
        # structured spinning-lidar model + per-ray model_element=(row,col)
        # (NRE's get_lidar_data_batch hard-requires both). NCoreBridge writes
        # them when the source WOD tfrecord is reachable (R2S_WOD_TFRECORD[_DIR];
        # see wod_structured_lidar.py); legacy flattened shards must keep
        # rays=0 or the sampler asserts. With supervision on we also mirror the
        # Alpasim recipe's loss weight and embed the lidar-derived ground mesh
        # into the artifact — the per-(x,y) ground-z oracle that retires the
        # ground_z_calib registry (see memory: ground-z saga). NOTE: do NOT
        # empty train_lidar_ids instead of zeroing rays — the sampler then
        # samples 0 sensors and asserts "len==1".
        *(("dataset.n_train_sample_lidar_rays=2048",
           "loss.lidar.lambda_=0.03",
           "checkpoint.artifact.mesh.ground.enabled=true")
          if lidar_supervision else
          ("dataset.n_train_sample_lidar_rays=0",)),
        # Dynamic-actor tracks: NRE filters cuboid observations by source via
        # cuboid_tracks_params.track_label_sources (car2sim default [AUTOLABEL],
        # for the Hyperion auto-label pipeline). NCoreBridge writes WOD's
        # ground-truth boxes as source=GT_ANNOTATION, so the default filters out
        # every observation → "Assemble Tracks: 0" → no dynamic Gaussian layers
        # (sequence_tracks.json empty) even though the scene has moving agents.
        # Select our source so the tracks are kept.
        "dataset.cuboid_tracks_params.track_label_sources=[GT_ANNOTATION]",
        # Image subsample factor must divide every camera's resolution. car2sim
        # defaults to 4 (uniform Hyperion rig), but WOD cameras are heterogeneous
        # — front is 1920x1280 (÷4 ok) while side cameras are 1920x886, and
        # 886 = 2x443 is not divisible by 4 → image_crop asserts
        # "Subsample factor 4 invalid, resolution is 1920x886". 2 is the largest
        # factor >1 dividing all WOD heights (1280 and 886) and the width (1920);
        # 1 trains at native resolution (~4x pixels/step).
        #
        # car2sim is EXEMPT: the Hyperion rig is uniform-resolution, so the recipe
        # composes its own subsample (4) that divides every nuPlan camera fine.
        # Forcing 2 here would double pixels/step vs the validated car2sim run
        # (unvalidated VRAM/perf on a 3090 at 40k iters). Let the recipe default
        # stand; the caller can still override via `overrides`.
        *(() if is_car2sim else (
            f"dataset.samplers.batch_sampler.camera_pixel_sampler.subsample={subsample}",
            f"dataset.n_val_image_subsample={subsample}",
            f"dataset.n_train_sequential_image_subsample={subsample}",
        )),
    ]
    if is_dynamic:
        # ── 3dgut_dynamic: per-actor nodes (background + dynamic_rigids +
        #    dynamic_deformables). Adapt the Waymo recipe to our NCore/WOD data. ──
        if cams:
            cmd.append(f"dataset.camera_ids={cam_list}")
        # (a) Class routing. Each dynamic node whitelists label classes via
        #     `tracks.label_classes`; the recipe ships NVIDIA names ("Car", "bus",
        #     …) that never match NCoreBridge's WOD class strings, so both dynamic
        #     nodes would assemble 0 tracks. Route our strings: vehicles -> rigid,
        #     pedestrians -> deformable. (Cyclists: add TYPE_CYCLIST if present.)
        cmd += [
            "model.layers.dynamic_rigids.tracks.label_classes="
            + _label_classes(dataset, "dynamic_rigids"),
            "model.layers.dynamic_deformables.tracks.label_classes="
            + _label_classes(dataset, "dynamic_deformables"),
        ]
        # (b) Seed BOTH dynamic nodes from cameras, not the default
        #     lidar-dynamic-tracks: WOD's lidar is flattened to an unstructured
        #     cloud through py123d (no per-ray beam grid), so per-cuboid lidar
        #     seeding yields ~0 points and the nodes train empty. camera-dynamic-
        #     tracks + fill_with_random_points guarantees every track gets a seed
        #     from the (good) cameras, so actors reconstruct as their own nodes.
        for _layer in ("dynamic_rigids", "dynamic_deformables"):
            _di = f"model.layers.{_layer}.initialization"
            cmd += [
                f"{_di}.name=camera-dynamic-tracks",
                # ++ (force-override) not + (add-only): car2sim_lidarfree already
                # defines these init keys, so + raises ConfigCompositionException
                # ("already at ...camera_ids"); ++ is safe whether or not preset.
                f"++{_di}.camera_ids=null",
                f"++{_di}.step_frame=1",
                f"++{_di}.fill_with_random_points=true",
            ]
        if static_separation and not is_car2sim:
            # (c) static_rigids values. The Waymo overlay only mounts the layer
            #     structure; camera_dynamic_tracks already defines these keys, so
            #     they must be ++ (force-override), not +. (car2sim_6cam_static
            #     bakes these values into the overlay itself — nothing to add here.)
            _si = "model.layers.static_rigids"
            cmd += [
                f"++{_si}.tracks.is_dynamic=false",
                f"++{_si}.tracks.label_classes="
                + _label_classes(dataset, "static_rigids"),
                f"++{_si}.initialization.symmetric_axis=Y",
                f"++{_si}.initialization.camera_ids=null",
                f"++{_si}.initialization.step_frame=1",
                f"++{_si}.initialization.fill_with_random_points=true",
            ]
        if is_car2sim:
            # car2sim adds a `road` layer + needs train/val sensor splits, a road
            # init swap (its lidar_ground_mesh_road init fails on flattened nuPlan
            # lidar), and the ground-mesh ARTIFACT disabled (it crashes at
            # checkpoint save on our degenerate road surface — car2sim saves only
            # one checkpoint at the final step, so that wastes the whole run).
            cmd += [
                f"dataset.train_camera_ids={cam_list}",
                f"dataset.val_camera_ids={cam_list}",
                f"dataset.train_lidar_ids={lid_list}",
                f"dataset.val_lidar_ids={lid_list}",
                "model.layers.road.initialization.name=lidar-rig-trajectory",
                "model.layers.road.initialization.default_scale=0.1",
                "+model.layers.road.initialization.num_near_points=100000",
                "+model.layers.road.initialization.num_far_points=100000",
                "+model.layers.road.initialization.far_radius_factor=20",
                "+model.layers.road.initialization.observation_scale_factor=0.01",
                "+model.layers.road.initialization.lidar_ids=null",
                "+model.layers.road.initialization.camera_ids=null",
                "+model.layers.road.initialization.non_dynamic_points_only=true",
                "checkpoint.artifact.mesh.ground.enabled=false",
            ]
    else:
        # ── car2sim_6cam (legacy static recipe): separate train/val sensor lists
        #    + a road layer whose ground-mesh init fails on WOD's sparse lidar. ──
        cmd += [
            f"dataset.train_lidar_ids={lid_list}",
            f"dataset.val_lidar_ids={lid_list}",
        ]
        if cams:
            cmd += [
                f"dataset.camera_ids={cam_list}",
                f"dataset.train_camera_ids={cam_list}",
                f"dataset.val_camera_ids={cam_list}",
            ]
        # The road layer's default init (lidar-ground-mesh-road) reconstructs a
        # road mesh from camera→lidar road labels. WOD's forward-only cameras
        # label far too few lidar points/frame for that pipeline (empty mesh →
        # AxisError, or degenerate frames → Delaunay assert). Swap to
        # lidar-rig-trajectory (the background layer's working init), which skips
        # the ground-mesh machinery. See PLAN.md "NRE training gotchas".
        _ri = "model.layers.road.initialization"
        cmd += [
            f"{_ri}.name=lidar-rig-trajectory",
            f"{_ri}.default_scale=0.1",
            f"+{_ri}.num_near_points=100000",
            f"+{_ri}.num_far_points=100000",
            f"+{_ri}.far_radius_factor=20",
            f"+{_ri}.observation_scale_factor=0.01",
            f"+{_ri}.lidar_ids=null",
            f"+{_ri}.camera_ids=null",
            f"+{_ri}.non_dynamic_points_only=true",
        ]
    cmd += overrides or []

    _run(cmd, step="train")

    # Prefer the requested run dir; fall back to whatever NRE actually created.
    produced = _discover_run_dir(output_dir, prefer=run_dir)
    if produced is None:
        raise FileNotFoundError(
            f"run_train: no <RUN-ID>/checkpoints/last.ckpt found under {output_dir} "
            f"after training (expected {run_dir})."
        )
    return produced


def run_export(
    run_dir: Path,
    work_dir: Path,
    ply_format: str = "_3DGS",
    export_ncore_tracks: bool = False,
    export_usdz: bool = True,
    dataset_dir: Optional[Path] = None,
    nurec_image: str = NRE_IMAGE,
) -> NuRecExport:
    """Export Gaussian PLYs + actor tracks + IsaacSim USDZ from a trained run.

    Calls ``export-gaussian-plys`` (per-layer ``.ply`` in 3DGS format),
    ``export-sequence-tracks`` (actor cuboid tracks JSON, world frame), and
    ``export-usdz-artifact`` (the IsaacSim-renderable USD bundle carrying the
    Gaussian neural volume). Optionally calls ``export-ncore-tracks`` for
    per-sensor per-frame poses (requires ``dataset_dir`` for the shard glob).

    Parameters
    ----------
    run_dir:
        Run directory from :func:`run_train` (``.../output/<RUN-ID>/``).
    work_dir:
        Per-clip working directory; the ``output`` root is derived from it so
        container mounts line up with ``run_train``.
    ply_format:
        ``"_3DGS"`` (third-party-viewer compatible) or ``"_3DGRT"`` — the exact
        values ``export-gaussian-plys --format`` accepts. Case/underscore are
        normalised, so ``"3dgs"`` also works.
    export_ncore_tracks:
        Also run ``export-ncore-tracks`` (needs ``dataset_dir``).
    export_usdz:
        Also run ``export-usdz-artifact`` to write the IsaacSim-renderable USDZ
        (default ``True``). The training-time ``artifacts/last.usdz`` is a
        render-only bundle WITHOUT the Gaussian volume (``car2sim_6cam`` trains
        with ``artifact.nrend`` disabled), so IsaacSim renders only the ground
        mesh; this re-emits the checkpoint as a USD carrying the
        ``OmniNuRecVolumeAPI`` volume IsaacSim's NuRec extension renders.
    dataset_dir:
        NCore dataset directory (for ``export-ncore-tracks`` shard glob).
    nurec_image:
        ``nre`` image:tag.

    Returns
    -------
    NuRecExport
    """
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir.parent
    rid = run_dir.name

    parsed_config = run_dir / "config" / "parsed.yaml"
    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if not parsed_config.exists():
        raise FileNotFoundError(f"run_export: parsed.yaml not found: {parsed_config}")

    c_parsed = f"{_C_OUTPUT}/{rid}/config/parsed.yaml"
    c_ckpt = f"{_C_OUTPUT}/{rid}/checkpoints/last.ckpt"
    plys_dir = run_dir / "plys"
    seq_tracks = run_dir / "sequence_tracks.json"

    # export-gaussian-plys / export-sequence-tracks load the dataset referenced
    # by parsed.yaml's dataset.path (=/workdir/dataset/pai_<clip>.json), so the
    # dataset dir must be mounted just like in run_train — else NRE asserts
    # "provided path .../pai_<clip>.json not a file".
    if dataset_dir is None:
        raise ValueError("run_export: dataset_dir is required (export reads the NCore dataset)")
    dataset_dir = Path(dataset_dir).resolve()
    ds_mount = ["-v", f"{dataset_dir}:{_C_DATASET}"]

    # ── export-gaussian-plys ────────────────────────────────────────────────
    # --format accepts exactly _3DGS / _3DGRT; normalise "3dgs"/"3dgrt"/etc.
    fmt = "_" + ply_format.lstrip("_").upper()
    if plys_dir.exists() and any(plys_dir.glob("*.ply")):
        logger.info("run_export: PLYs already exist at %s, skipping", plys_dir)
    else:
        cmd = _docker_base(shm="64g", with_user=True) + ds_mount + [
            "-v", f"{output_dir}:{_C_OUTPUT}",
            nurec_image,
            "export-gaussian-plys",
            "--config-name", c_parsed,
            "--checkpoint-name", "last.ckpt",
            "--output-dir", f"{_C_OUTPUT}/{rid}/plys",
            "--format", fmt,
        ]
        _run(cmd, step="export-gaussian-plys")

    # ── export-sequence-tracks ──────────────────────────────────────────────
    if seq_tracks.exists():
        logger.info("run_export: sequence_tracks.json exists, skipping")
    else:
        cmd = _docker_base(shm="64g", with_user=True) + ds_mount + [
            "-v", f"{output_dir}:{_C_OUTPUT}",
            nurec_image,
            "export-sequence-tracks",
            "--config-name", c_parsed,
            "--checkpoint-path", c_ckpt,
            "--output-dir", f"{_C_OUTPUT}/{rid}",
            "--world-frame",
            "--format", "json",
        ]
        _run(cmd, step="export-sequence-tracks")

    # ── export-usdz-artifact (IsaacSim-renderable USD + neural volume) ───────
    # Reads the dataset spec from parsed.yaml, so the dataset dir must be mounted
    # (same as export-gaussian-plys). Output filename is release-dependent
    # (last.usdz / export_last.usdz), so we discover it by glob afterwards.
    usd_out_dir = run_dir / "usd-out"
    if export_usdz:
        existing_usdz = _find_isaac_usdz(usd_out_dir)
        if existing_usdz is not None:
            logger.info("run_export: IsaacSim USDZ exists at %s, skipping", existing_usdz)
        else:
            cmd = _docker_base(shm="64g", with_user=True) + ds_mount + [
                "-v", f"{output_dir}:{_C_OUTPUT}",
                nurec_image,
                "export-usdz-artifact",
                "--config-name", c_parsed,
                "--checkpoint-name", "last.ckpt",
                "--output-dir", f"{_C_OUTPUT}/{rid}/usd-out",
            ]
            _run(cmd, step="export-usdz-artifact")

    # ── export-gaussian-usd-asset (renderable Gaussian USD bundle) ───────────
    # export-usdz-artifact's bundle is checkpoint + ground mesh + trajectories
    # only — referencing it in IsaacSim shows an untextured ground mesh and no
    # Gaussians. The bundle IsaacSim actually renders (background/road Gaussian
    # usdc prims + sky HDRI domelight + baked PPISP shader) comes from
    # export-gaussian-usd-asset (usd-asset/pai_*.usdz).
    usd_asset_dir = run_dir / "usd-asset"
    if export_usdz:
        if any(usd_asset_dir.glob("*.usdz")):
            logger.info("run_export: Gaussian USD asset exists at %s, skipping", usd_asset_dir)
        else:
            cmd = _docker_base(shm="64g", with_user=True) + ds_mount + [
                "-v", f"{output_dir}:{_C_OUTPUT}",
                nurec_image,
                "export-gaussian-usd-asset",
                "--config-name", c_parsed,
                "--output-dir", f"{_C_OUTPUT}/{rid}/usd-asset",
                "--usd-format", "usdz",
                "--ppisp-frame-idx", "0",
                "--skip-gaussian-deformation",
                "--half-precision",
                "--export-summary",
            ]
            _run(cmd, step="export-gaussian-usd-asset")

    # ── export-ncore-tracks (optional) ──────────────────────────────────────
    ncore_tracks: Optional[Path] = None
    if export_ncore_tracks:
        if dataset_dir is None:
            raise ValueError("run_export: export_ncore_tracks=True requires dataset_dir")
        dataset_dir = Path(dataset_dir).resolve()
        manifest = _ncore_manifest(dataset_dir)
        shard_glob = f"{_C_DATASET}/{manifest.stem}.zarr.itar"
        ncore_tracks = run_dir / "ncore_tracks.json"
        if not ncore_tracks.exists():
            cmd = _docker_base(shm="64g", with_user=True) + [
                "-v", f"{dataset_dir}:{_C_DATASET}",
                "-v", f"{output_dir}:{_C_OUTPUT}",
                nurec_image,
                "export-ncore-tracks",
                "--shard-file-pattern", shard_glob,
                "--model-tracks-json", f"{_C_OUTPUT}/{rid}/sequence_tracks.json",
                "--output-dir", f"{_C_OUTPUT}/{rid}",
            ]
            _run(cmd, step="export-ncore-tracks")

    # ── assemble result ──────────────────────────────────────────────────────
    return load_nurec_export(run_dir, ncore_tracks=ncore_tracks)


def run_render(
    run_dir: Path,
    camera_ids: List[str],
    image_scale: float = 0.5,
    frame_step: int = 1,
    export_video: bool = True,
    custom_rig_trajectory: Optional[str] = None,
    nurec_image: str = NRE_IMAGE,
) -> Path:
    """Render camera frames (+ optional MP4) from a trained run's USDZ artifact.

    Wraps ``nre render`` against ``artifacts/last.usdz``, which is self-contained
    (model checkpoint + tracks + config), so no NCore dataset mount is needed.
    Renders along the original rig trajectory by default — a good way to *see*
    the reconstruction (do dynamic actors appear? road/floater quality?). Pass
    ``custom_rig_trajectory`` (a path inside the run dir) for novel views.

    Parameters
    ----------
    run_dir:
        Trained run directory (``.../output/<RUN-ID>/``) containing
        ``artifacts/last.usdz``.
    camera_ids:
        Cameras to render (NRE errors if none are given). e.g. ``camera_pcam_f0``.
    image_scale:
        Output resolution as a fraction of each camera's native resolution.
    frame_step:
        Render every ``frame_step``-th frame along the trajectory.
    export_video:
        Also write an MP4 (H.264) per camera.
    custom_rig_trajectory:
        Optional path (container-visible, under the run dir) to a custom rig
        trajectory for novel-view rendering.

    Returns
    -------
    Path
        Host render output directory (``run_dir/render``).
    """
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir.parent
    rid = run_dir.name
    if not camera_ids:
        raise ValueError("run_render: at least one camera_id is required")
    artifact = run_dir / "artifacts" / "last.usdz"
    if not artifact.exists():
        raise FileNotFoundError(f"run_render: USDZ artifact not found: {artifact}")
    render_dir = run_dir / "render"

    cmd = _docker_base(shm="64g", with_user=True) + [
        "-v", f"{output_dir}:{_C_OUTPUT}",
        nurec_image,
        "render",
        "--artifact-path", f"{_C_OUTPUT}/{rid}/artifacts/last.usdz",
        "--output-dir", f"{_C_OUTPUT}/{rid}/render",
        "--image-scale", str(image_scale),
        "--frame-step", str(frame_step),
    ]
    for cid in camera_ids:
        cmd += ["--camera-id", cid]
    if export_video:
        cmd.append("--export-video")
    if custom_rig_trajectory:
        cmd += ["--custom-rig-trajectory", custom_rig_trajectory]

    _run(cmd, step="render")
    logger.info("run_render: frames/video written to %s", render_dir)
    return render_dir


def run_serve_grpc(
    work_dir: Path,
    host_port: int = 8080,
    artifact_glob: str = "*/output*/*/usd-out/*.usdz",
    nurec_image: str = NRE_IMAGE,
    enable_harmonizer: bool = True,
) -> None:
    """Serve reconstructions over the sensorsim gRPC API (blocking).

    Wraps ``serve-grpc --enable-editing-actors``: every reconstructed actor
    (moving, plus parked cars / signs when trained with
    ``static_separation=True``) is exposed as a controllable node the
    ``nurec_grpc`` render backend can move or remove per frame. Point the
    eval at it via ``NUREC_GRPC_HOST`` / ``NUREC_GRPC_PORT`` — see
    ``docs/navsafe_eval.md``. Blocks until interrupted.

    Parameters
    ----------
    work_dir:
        Host dir containing the NRE run dirs (the ``work_dir`` given to
        :func:`run_train`); scanned with ``artifact_glob``.
    host_port:
        Host port publishing the container's gRPC port 8080.
    artifact_glob:
        Glob relative to ``work_dir`` selecting the usd-out USDZs to serve.
    nurec_image:
        ``nre`` image (digest-pinned by default).
    enable_harmonizer:
        Run the server's DiffusionHarmonizer postprocessing pass on every
        rendered frame (``--enable-harmonizer``). Default ON — it measurably
        improves perceptual quality. The checkpoint is fetched from
        HuggingFace on first use into a host cache dir
        (``~/.cache/nre/harmonizer``, override ``NAVSAFE_HARMONIZER_CACHE``)
        mounted into the container, so the download is paid once, not per
        server start.
    """
    work_dir = Path(work_dir).resolve()
    cmd = _docker_base(shm="16g", with_user=True)
    harmonizer_args: List[str] = []
    if enable_harmonizer:
        # The container runs -u uid:gid with no writable $HOME, so the
        # server's default cache (~/.cache/nre/harmonizer inside the
        # container) is unwritable AND ephemeral — the cache must live on a
        # mounted host dir.
        cache = Path(os.environ.get(
            "NAVSAFE_HARMONIZER_CACHE",
            str(Path.home() / ".cache/nre/harmonizer"))).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        cmd += ["-v", f"{cache}:/harmonizer-cache"]
        harmonizer_args = ["--enable-harmonizer",
                           "--harmonizer-cache", "/harmonizer-cache"]
    cmd += [
        "-p", f"{host_port}:8080",
        "-v", f"{work_dir}:{_C_OUTPUT}",
        nurec_image,
        "serve-grpc", "--host", "0.0.0.0",
        "--enable-editing-actors", "--renderer", "default",
        *harmonizer_args,
        "--artifact-glob", f"{_C_OUTPUT}/{artifact_glob}",
    ]
    _run(cmd, step="serve-grpc")


def load_nurec_export(run_dir: Path, ncore_tracks: Optional[Path] = None) -> NuRecExport:
    """Build a :class:`NuRecExport` by discovering artifacts in an existing run dir.

    Used by :func:`run_export` after Docker completes, and by ``--skip-nurec``
    reruns that reconstruct the export from outputs already on disk (no Docker).
    """
    run_dir = Path(run_dir).resolve()
    plys_dir = run_dir / "plys"
    seq_tracks = run_dir / "sequence_tracks.json"
    # Prefer the renderable Gaussian bundle (export-gaussian-usd-asset) over the
    # checkpoint-only export-usdz-artifact output.
    usd_asset = sorted((run_dir / "usd-asset").glob("*.usdz")) if (run_dir / "usd-asset").exists() else []
    usdz = usd_asset[0] if usd_asset else _find_isaac_usdz(run_dir / "usd-out")
    map_xodr = run_dir / "usd-out" / "map.xodr"
    parsed_config = run_dir / "config" / "parsed.yaml"
    checkpoint = run_dir / "checkpoints" / "last.ckpt"

    if ncore_tracks is None:
        nct = run_dir / "ncore_tracks.json"
        ncore_tracks = nct if nct.exists() else None

    ply_paths = sorted(plys_dir.glob("*.ply")) if plys_dir.exists() else []

    return NuRecExport(
        run_dir=run_dir,
        usdz_path=usdz,
        parsed_config=parsed_config if parsed_config.exists() else None,
        checkpoint=checkpoint if checkpoint.exists() else None,
        plys_dir=plys_dir if plys_dir.exists() else None,
        ply_paths=ply_paths,
        sequence_tracks_json=seq_tracks if seq_tracks.exists() else None,
        ncore_tracks_json=ncore_tracks if (ncore_tracks and ncore_tracks.exists()) else None,
        map_xodr=map_xodr if map_xodr.exists() else None,
    )


# ── internal helpers ──────────────────────────────────────────────────────────

def _docker_base(shm: str, with_user: bool) -> List[str]:
    """Common ``docker run`` prefix: GPUs, shm, NGC key, optional uid:gid.

    ``NAVSAFE_NUREC_GPUS`` restricts the container to specific GPU indices
    (docker ``--gpus device=...`` syntax, e.g. ``7`` or ``6,7``); unset → all.
    """
    gpus = os.environ.get("NAVSAFE_NUREC_GPUS", "").strip()
    gpu_arg = f"\"device={gpus}\"" if gpus else "all"
    cmd = [
        "docker", "run", "--rm", "--gpus", gpu_arg,
        f"--shm-size={shm}",
        "-e", f"NGC_API_KEY={_ngc_api_key()}",
        # NRE apps run under Hydra, which by default truncates tracebacks to the
        # bare exception message. Surface the full stack so container-side errors
        # are diagnosable from the host log.
        "-e", "HYDRA_FULL_ERROR=1",
    ]
    if with_user:
        # Pin uid:gid so container-written outputs are owned by the host user.
        cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]
    return cmd


def _ngc_api_key() -> str:
    key = os.environ.get("NGC_API_KEY")
    if not key:
        raise RuntimeError(
            "NGC_API_KEY is not set. Export your NGC API key and run "
            "`docker login nvcr.io` (username '$oauthtoken') before invoking NuRec."
        )
    return key


def _find_isaac_usdz(usd_out_dir: Path) -> Optional[Path]:
    """Locate the IsaacSim-renderable USDZ written by ``export-usdz-artifact``.

    The exact filename is release-dependent (``last.usdz`` / ``export_last.usdz``),
    so glob ``usd-out`` and prefer the conventional names. Returns ``None`` if the
    export has not run.
    """
    if not usd_out_dir.exists():
        return None
    candidates = sorted(usd_out_dir.glob("*.usdz"))
    if not candidates:
        return None
    for name in ("last.usdz", "export_last.usdz"):
        for c in candidates:
            if c.name == name:
                return c
    return candidates[0]


def _ncore_manifest(clip_dir: Path) -> Path:
    """Return the NCore JSON manifest (``pai_<clip>.json``) inside ``clip_dir``."""
    candidates = sorted(p for p in clip_dir.glob("*.json") if not p.name.endswith(".aux.json"))
    if not candidates:
        raise FileNotFoundError(f"No NCore JSON manifest found in {clip_dir}")
    # Prefer the conventional pai_*.json if present.
    for p in candidates:
        if p.name.startswith("pai_"):
            return p
    return candidates[0]


def _link_aux_to_manifest_base(clip_dir: Path, manifest: Path) -> None:
    """Symlink ``ncore-aux-data`` outputs to the dataset manifest's base name.

    NRE training (``dataset.aux_data=True``) discovers aux stores by the manifest
    base name (``<manifest_stem>.aux.*.zarr.itar``), but ``ncore-aux-data`` names
    its outputs after the sequence_id, which omits NCoreBridge's ``pai_`` file
    prefix (it writes ``<sequence_id>.aux.sseg.zarr.itar`` etc.). Without a
    matching name training fails with ``No semantic segmentation data found for
    <camera>``. Create ``<manifest_stem>.aux.*`` symlinks so the stores resolve.

    Only the ``.zarr.itar`` aux stores are linked. The ``.aux-meta.json`` is
    deliberately skipped: a ``pai_*.aux-meta.json`` would sort ahead of the real
    ``pai_<clip>.json`` in :func:`_ncore_manifest`'s glob and shadow it.
    """
    base = manifest.name[: -len(".json")]  # e.g. "pai_<clip>"
    for store in clip_dir.glob("*.aux.*.zarr.itar"):
        if store.name.startswith(base + ".aux."):
            continue  # already the manifest-base name (or a link we made)
        marker = ".aux."
        suffix = store.name[store.name.index(marker):]  # ".aux.sseg.zarr.itar"
        link = clip_dir / f"{base}{suffix}"
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(store.name)  # relative link, stays valid if dir moves
        logger.info("run_aux: linked aux store %s -> %s", link.name, store.name)


def _sanitize_run_id(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


def _discover_sensor_ids(dataset_dir: Path):
    """Derive (camera_ids, lidar_ids) from NCore shard filenames.

    NCoreBridge writes sensor stores as ``pai_<clip>.ncore4-<sensor_id>.zarr.itar``
    where ``<sensor_id>`` is exactly the ID NRE's ``dataset.{camera,lidar}_ids``
    expect (``camera_pcam_f0``, ``lidar_top_360fov``, …). The plain
    ``pai_<clip>.ncore4.zarr.itar`` (no ``-``) is the non-sensor store and is
    skipped. Returns sorted, de-duplicated lists.
    """
    cams, lids = set(), set()
    for p in Path(dataset_dir).glob("*.ncore4-*.zarr.itar"):
        sid = p.name.split(".ncore4-", 1)[1]
        if sid.endswith(".zarr.itar"):
            sid = sid[: -len(".zarr.itar")]
        if sid.startswith("camera_"):
            cams.add(sid)
        elif sid.startswith("lidar"):
            lids.add(sid)
    return sorted(cams), sorted(lids)


def _discover_run_dir(output_dir: Path, prefer: Optional[Path] = None) -> Optional[Path]:
    """Return a run dir under ``output_dir`` that holds ``checkpoints/last.ckpt``.

    NRE names the run subdirectory from ``logger.run_id`` when honoured, but may
    fall back to its own scheme. We prefer ``prefer`` if it has a checkpoint,
    else return the most-recently-modified run dir that does.
    """
    if prefer is not None and (prefer / "checkpoints" / "last.ckpt").exists():
        return prefer
    if not output_dir.exists():
        return None
    candidates = [
        d for d in output_dir.iterdir()
        if d.is_dir() and (d / "checkpoints" / "last.ckpt").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _run(cmd: List[str], step: str) -> None:
    # Redact the NGC key from logs.
    safe = []
    redact_next = False
    for tok in cmd:
        if redact_next:
            safe.append("NGC_API_KEY=***")
            redact_next = False
        elif tok == "-e":
            safe.append(tok)
            redact_next = True
        else:
            safe.append(tok)
    logger.info("nurec_runner [%s]: %s", step, " ".join(safe))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"NuRec {step} failed with exit code {result.returncode} "
            f"(image/sub-command: {step})."
        )
