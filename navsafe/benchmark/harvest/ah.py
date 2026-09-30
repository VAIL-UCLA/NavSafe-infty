# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Driving NVIDIA Asset Harvester: the four steps, and what each one needs.

Asset Harvester is a third-party checkout with its own conda environment
(``cfg.AH_REPO`` / ``cfg.AH_PYTHON``); nothing here imports it. Every step is a
subprocess, which is also what keeps its pinned torch 2.10 / gsplat build out of
the NexusSim venv.

    parse    NCore V4 clip  ->  per-track 512x512 multi-view crops + masks
    lift     crops          ->  16 synthesised views  ->  gaussians.ply
    orient   PLY            ->  NuRec's axis convention
    describe PLYs           ->  metadata.yaml (NVIDIA's external-assets format)

Three things about *our* clips that the upstream defaults get wrong:

**Camera ids.** Upstream defaults to the Hyperion rig
(``camera_front_wide_120fov`` and friends). Our clips are converted from nuPlan
and name their sensors ``camera_pcam_f0``, ``camera_pcam_l0`` ... so the ids are
read out of the clip's own manifest instead of being assumed. A wrong id list is
not an error upstream -- it silently parses nothing.

**Windows.** A ``<token>_20s`` scenario is four separate 5 s NCore clips, one
per reconstruction. A car that crosses a boundary is one track with one id in
all of them, so parsing is per window and selection collapses the duplicates
(``select.choose``): one asset per track, harvested from the window that saw it
closest.

**Orientation.** NuRec expects a vehicle PLY with top toward -Y, front toward
-X, right toward +Z, unit-scaled, origin at the box centroid. Asset Harvester
does not emit that directly; ``orient_gaussians_for_nurec`` rotates 90 degrees
about Y to get there, and skipping it puts every replaced car sideways.

Scale is not predicted by Asset Harvester at all -- the PLY is unit-scaled and
the real dimensions come from the clip's cuboid. That is why nothing here
rescales geometry: the render server is told the AABB at replace time and does
the scaling itself.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from navsafe.benchmark import config as cfg
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

SEG_CKPT = "AH_object_seg_jit.pt"
DIFFUSION_CKPT = "AH_multiview_diffusion.safetensors"
LIFTING_CKPT = "AH_tokengs_lifting.safetensors"

DEFAULT_NUM_STEPS = 30
DEFAULT_CFG_SCALE = 2.0


class HarvestError(NexusSimError, RuntimeError):
    """Asset Harvester is not installed, or one of its steps failed."""


# --------------------------------------------------------------------------
# environment


def check_install() -> None:
    """Fail early and by name, rather than inside a subprocess ten minutes in."""
    missing = []
    if not cfg.AH_PYTHON.is_file():
        missing.append(f"interpreter {cfg.AH_PYTHON}")
    if not (cfg.AH_REPO / "run_inference.py").is_file():
        missing.append(f"checkout {cfg.AH_REPO}")
    for name in (SEG_CKPT, DIFFUSION_CKPT, LIFTING_CKPT):
        if not (cfg.AH_REPO / "checkpoints" / name).is_file():
            missing.append(f"checkpoint {name}")
    # A pod can be given a GPU that CUDA cannot open -- one bad device on an
    # otherwise healthy node. Caught here it costs two seconds and names itself;
    # caught where it used to be, it surfaced as a torch deserialization error
    # inside the segmentation model, three stack frames into a third party.
    if not missing:
        probe = subprocess.run(
            [str(cfg.AH_PYTHON), "-c",
             "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
            capture_output=True)
        if probe.returncode != 0:
            raise HarvestError(
                f"no usable CUDA device on {os.environ.get('NODE_NAME', 'this node')} "
                f"(torch.cuda.is_available() is False). Harvesting needs a GPU; "
                f"the pod was scheduled onto one that cannot be opened.")
    if missing:
        raise HarvestError(
            "Asset Harvester is not installed here: " + "; ".join(missing) +
            ". Point NAVSAFE_AH_HOME at the install, or create it with "
            "`bash setup.sh` in a clone of github.com/NVIDIA/asset-harvester "
            "plus `hf download nvidia/asset-harvester --local-dir checkpoints` "
            "(the model card is gated; HF_TOKEN must have accepted it).")


def _run(argv: Sequence[str], *, log: Optional[Path] = None, cwd: Optional[Path] = None) -> None:
    """One Asset Harvester step, with its output kept.

    Its logs are the only account of why a track produced no asset (occluded in
    every view, guard rejected the crops, lifting diverged), and those are
    per-track skips that leave the run "successful" with fewer assets.
    """
    pretty = " ".join(str(a) for a in argv)
    logger.info("ah: %s", pretty)
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(cfg.AH_HOME / "hf"))
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(f"\n$ {pretty}\n")
            fh.flush()
            rc = subprocess.call(
                [str(a) for a in argv], stdout=fh, stderr=subprocess.STDOUT,
                cwd=str(cwd or cfg.AH_REPO), env=env)
    else:
        rc = subprocess.call([str(a) for a in argv], cwd=str(cwd or cfg.AH_REPO), env=env)
    dt = time.time() - t0
    if rc != 0:
        raise HarvestError(
            f"asset-harvester step failed (exit {rc}, {dt:.0f}s): {pretty}"
            + (f" -- see {log}" if log else ""))
    logger.info("ah: done in %.0fs", dt)


# --------------------------------------------------------------------------
# step 1 -- parse


def clip_manifest(window: str, corpus: Optional[Path] = None) -> Path:
    """The NCore V4 clip manifest for one 5 s reconstruction window."""
    path = cfg.clips_dir(window, corpus) / f"pai_{window}.json"
    if not path.is_file():
        raise HarvestError(f"no NCore clip manifest for {window} at {path}")
    return path


def camera_ids(manifest: Path) -> List[str]:
    """The clip's own camera sensor ids, read from its manifest.

    Upstream's default list is the Hyperion rig and matches nothing in a
    nuPlan-converted clip; a mismatch parses zero tracks and reports success.
    """
    doc = json.loads(Path(manifest).read_text())
    ids: List[str] = []
    for store in doc.get("component_stores", []):
        for cam in (store.get("components", {}).get("cameras") or {}):
            if cam not in ids:
                ids.append(cam)
    if not ids:
        raise HarvestError(f"{manifest} declares no cameras")
    return ids


def window_motion(
    window: str,
    out_json: Path,
    *,
    corpus: Optional[Path] = None,
    log: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Per-track displacement for one window, from its clip's cuboids.

    A subprocess like every other step here: reading NCore cuboids needs
    ``ncore`` and ``asset_harvester``, which live in the harvester's environment
    and must not become a NexusSim dependency. See ``harvest/motion.py`` for why
    the selection needs this at all.
    """
    out_json = Path(out_json)
    _run([
        cfg.AH_PYTHON, "-m", "navsafe.benchmark.harvest.motion",
        "--manifest", clip_manifest(window, corpus),
        "--out", out_json,
    ], log=log, cwd=cfg.NAVSAFE_ROOT)
    return json.loads(out_json.read_text())


def parse_window(
    window: str,
    out_dir: Path,
    *,
    corpus: Optional[Path] = None,
    cameras: Optional[Sequence[str]] = None,
    log: Optional[Path] = None,
) -> Path:
    """Crop every track in one window's clip into per-track multi-view samples."""
    manifest = clip_manifest(window, corpus)
    cams = list(cameras) if cameras else camera_ids(manifest)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([
        cfg.AH_PYTHON, "-m", "asset_harvester.ncore_parser",
        "--component-store", manifest,
        "--output-path", out_dir,
        "--camera-ids", ",".join(cams),
        "--segmentation-ckpt", cfg.AH_REPO / "checkpoints" / SEG_CKPT,
    ], log=log)
    return out_dir


# --------------------------------------------------------------------------
# step 2 -- lift


def lift(
    data_root: Path,
    out_dir: Path,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    offload: bool = False,
    log: Optional[Path] = None,
) -> Path:
    """Multi-view diffusion + TokenGS lifting for every sample under ``data_root``.

    This is the expensive step and the only one that is per-asset rather than
    per-clip, which is why selection happens before it and not after.
    """
    argv = [
        cfg.AH_PYTHON, cfg.AH_REPO / "run_inference.py",
        "--diffusion_checkpoint", cfg.AH_REPO / "checkpoints" / DIFFUSION_CKPT,
        "--lifting_checkpoint", cfg.AH_REPO / "checkpoints" / LIFTING_CKPT,
        "--data_root", data_root,
        "--output_dir", out_dir,
        "--num_steps", str(int(num_steps)),
        "--cfg_scale", str(float(cfg_scale)),
    ]
    if offload:
        argv.append("--offload_model_to_cpu")
    _run(argv, log=log)
    return Path(out_dir)


# --------------------------------------------------------------------------
# steps 3 and 4 -- orient, describe


def orient(staging_dir: Path, *, degrees: float = 90.0, log: Optional[Path] = None) -> None:
    """Rotate the PLYs under ``staging_dir`` into NuRec's convention, in place.

    In place because the un-oriented PLY has no use: nothing reads it, and
    keeping a copy would double the only sizeable output this produces.

    **This is not idempotent** -- a second call rotates another 90 degrees and
    leaves every replaced car facing sideways, with nothing in the output to say
    so. Hence the staging directory: :func:`lift` writes there, this rotates
    exactly what was just written, and :func:`promote` moves the result into the
    permanent bank. A resumed harvest cannot re-rotate an asset it kept, because
    the kept assets were never in the staging directory.
    """
    _run([
        cfg.AH_PYTHON, "-m", "asset_harvester.utils.orient_gaussians_for_nurec",
        "--input-dir", staging_dir, "--in-place", "--degrees", str(float(degrees)),
    ], log=log)


def ply_vertex_count(path: Path) -> int:
    """Vertices declared in a PLY header, or 0 if it has none."""
    with Path(path).open("rb") as fh:
        for _ in range(64):
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header" or not line:
                break
    return 0


def drop_degenerate(staging_dir: Path) -> List[str]:
    """Remove lifted samples whose PLY has no vertices, before orienting.

    TokenGS occasionally emits an empty cloud. Orientation then dies with
    ``No vertices found in PLY header`` -- and because that is a subprocess
    failure, it took down the whole scenario's harvest rather than the one bad
    asset (seen on 8b20ada64fe8512a_20s). One asset is allowed to fail; a
    scenario is not.
    """
    dropped = []
    for ply in sorted(Path(staging_dir).rglob("gaussians.ply")):
        if ply_vertex_count(ply) < 1:
            logger.warning("lifted an empty cloud for %s; dropping it",
                           ply.parent.name)
            dropped.append(ply.parent.name)
            shutil.rmtree(ply.parent, ignore_errors=True)
    return dropped


def promote(staging_dir: Path, lifted_dir: Path) -> List[str]:
    """Move freshly oriented ``<class>/<track>`` samples into the bank."""
    staging_dir, lifted_dir = Path(staging_dir), Path(lifted_dir)
    moved: List[str] = []
    for ply in sorted(staging_dir.rglob("gaussians.ply")):
        sample = ply.parent
        dst = lifted_dir / sample.parent.name / sample.name
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sample), str(dst))
        moved.append(sample.name)
    clear(staging_dir)
    return moved


def describe(lifted_dir: Path, *, log: Optional[Path] = None) -> Path:
    """Write NVIDIA's ``metadata.yaml`` beside the PLYs.

    Our renderer does not read it -- it goes through ``replace_manifest.json``
    and reads the AABB off the server. It is written anyway because it is the
    format the documented offline route (``export-external-assets`` into a
    repackaged USDZ) consumes, so the same directory stays usable there without
    re-harvesting.
    """
    _run([
        cfg.AH_PYTHON, cfg.AH_REPO / "asset_harvester/utils/generate_external_assets_metadata.py",
        "--input-dir", lifted_dir,
    ], log=log)
    out = Path(lifted_dir) / "metadata.yaml"
    if not out.is_file():
        raise HarvestError(f"metadata generation produced no {out}")
    return out


def ply_extent(path: Path) -> List[float]:
    """The asset's own (x, y, z) extent, in its unit-scaled PLY coordinates.

    Percentile rather than min/max: a lifted gaussian cloud has a faint tail
    reaching well past the body, and a raw bounding box measures that halo
    instead of the car.
    """
    import numpy as np  # local: everything else in this module is subprocess-only

    with Path(path).open("rb") as fh:
        n, props = 0, []
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        buf = fh.read(n * 4 * len(props))
    arr = np.frombuffer(buf, dtype="<f4").reshape(n, len(props))
    xyz = arr[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)
    lo, hi = np.percentile(xyz, [1.0, 99.0], axis=0)
    return [float(v) for v in (hi - lo)]


def aspect_error(extent: Sequence[float], dims: Sequence[float]) -> float:
    """How far a lifted asset's SHAPE is from the cuboid the clip recorded.

    The server scales an asset onto the track's box, so a wrong aspect ratio is
    not a small asset -- it is a car stretched or squashed onto the right
    footprint, which is more visibly wrong than the smear it replaced. The
    diffusion can produce one: a mask that caught the neighbouring car, or a
    track seen from one angle only.

    Compared in NuRec's post-orientation convention -- x is length, y height,
    z width -- and on ratios, because the PLY is unit-scaled and the cuboid is
    metric. Returns the worst per-axis relative error, or 0.0 when there is
    nothing to compare against.
    """
    if not dims or len(dims) < 3 or max(extent) <= 0:
        return 0.0
    length, width, height = (float(v) for v in dims[:3])
    want = [length, height, width]
    if min(want) <= 0:
        return 0.0
    scale = max(want) / max(extent)
    return max(abs(e * scale - w) / w for e, w in zip(extent, want))


def lifted_assets(lifted_dir: Path, *, measure: bool = False) -> Dict[str, Dict[str, object]]:
    """What actually came out: track id -> PLY path, class and dimensions.

    Read from the tree rather than from what was asked for, because a sample
    can be skipped mid-run (image guard, no usable views) and the difference
    between requested and produced is exactly what the caller must report.

    ``measure`` also opens each PLY for its extent and aspect error, which is
    what the quality gate reads. It is off by default because the resume check
    calls this once per run just to ask which tracks already exist.
    """
    out: Dict[str, Dict[str, object]] = {}
    for ply in sorted(Path(lifted_dir).rglob("gaussians.ply")):
        sample = ply.parent
        lwh_file = sample / "multiview" / "lwh.txt"
        dims: List[float] = []
        if lwh_file.is_file():
            dims = [float(v) for v in lwh_file.read_text().split()]
        rec: Dict[str, object] = {
            "ply": str(ply.resolve()),
            "label_class": sample.parent.name,
            "cuboids_dims": dims,
        }
        if measure:
            extent = ply_extent(ply)
            rec["extent"] = [round(v, 4) for v in extent]
            rec["aspect_error"] = round(aspect_error(extent, dims), 4)
        out[sample.name] = rec
    return out


def clear(path: Path) -> None:
    """Remove a scratch tree, tolerating its absence."""
    shutil.rmtree(Path(path), ignore_errors=True)
