"""Deployment paths for the NavSafe benchmark, in one place.

Following the convention in ``docs/reconstruct_navsim_nuplan.md``: the code
ships no data paths of its own, it reads environment variables and every CLI
flag can override them.  Defaults use the current user's home/cache directories. Set these variables
for the dataset and repository mounts used in your deployment.

Three kinds of location, and keeping them apart is the point:

    NAVSAFE_CORPUS  per-scenario and permanent -- one directory per scenario,
    NAVSAFE_NAVHARD_CORPUS   holding its clip, its recon and its Arrow
    NAVSAFE_RUNS    per-run and disposable -- one directory per render/eval
    NAVSAFE_WORK    working state shared across runs (index, seeds, recipes)

    NUPLAN_ROOT    raw nuPlan: nuplan-v1.1/{splits,maps,sensor_blobs}
    NAVSAFE_ROOT       this repository, as mounted inside cluster jobs
    NAVSAFE_LOCAL_DB  container-local staging for nuPlan log dbs

``NAVSAFE_LOCAL_DB`` deserves the note it gets in ``run_ncore.py``: the nuPlan
devkit's ORM issues millions of small page reads against a single log db, which
is pathological on a network filesystem, so it must point at real local disk and
not at the shared PVC.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

# --- roots ----------------------------------------------------------------
WORK = Path(os.environ.get("NAVSAFE_WORK", str(Path.home() / ".cache/navsafe/work")))
NUPLAN_ROOT = Path(os.environ.get("NUPLAN_ROOT", str(Path.home() / "data/nuplan")))
NAVSAFE_ROOT = Path(os.environ.get("NAVSAFE_ROOT", str(Path(__file__).resolve().parents[2])))
LOCAL_DB = Path(os.environ.get("NAVSAFE_LOCAL_DB", "/tmp/nuplan_local"))

# --- derived artifact tree ------------------------------------------------
INDEX = WORK / "index"
CANDIDATES = WORK / "candidates"
SEEDS = WORK / "seeds"
EXPORT = WORK / "export"
EVAL = WORK / "eval"
UV_CACHE = WORK / ".uv-cache"
HOST_INDEX = WORK / "host_index.json"
# DiffusionHarmonizer checkpoint cache for serve-grpc (fetched from
# HuggingFace); on the PVC so only the first eval job pays the download.
HARMONIZER_CACHE = WORK / ".harmonizer-cache"

# --- scenario corpora -----------------------------------------------------
# A corpus is the permanent, per-scenario side of the deployment: one directory
# per scenario, named for its scene id, holding everything derived from that
# scenario exactly once.  It is not where a render or an eval writes — those
# are per-run and fork with every experiment, so they go somewhere else.  Two
# corpora exist; they differ in clip length and camera rig, not in shape.
#
#     <corpus>/<scene_id>/clips/<scene_id>/                    NCore V4 in
#     <corpus>/<scene_id>/output_5cam/<scene_id>/artifacts/    NRE recon out
#     <corpus>/<scene_id>/arrow/                               py123d Arrow
#
# A ``<token>_20s`` scene id is a stitched host: it carries only ``arrow``, and
# its reconstructions are the four ``<token>s1..s4`` siblings beside it.
CORPUS = Path(os.environ.get("NAVSAFE_CORPUS", str(WORK / "corpus")))
NAVHARD_CORPUS = Path(os.environ.get("NAVSAFE_NAVHARD_CORPUS", str(WORK / "navhard")))

# Derived from a corpus but not part of one: previews are regenerable.
#
# There used to be a SERVE_DIR here as well, naming a hand-maintained tree of
# symlinks that serve-grpc was pointed at. It is gone: the renderer now globs
# the training tree directly, so a scenario is servable exactly when it is
# trained and there is nothing to register. Nothing should reintroduce it --
# a symlink whose target is missing killed the whole server at start-up.
MINING_BEV = Path(os.environ.get("NAVSAFE_MINING_BEV", str(WORK / "mining_bev")))

# Preserve the directory layout of c13752hz/NavSafe on Hugging Face.
_data_root = os.environ.get("NAVSAFE_DATA_ROOT")
DATA_ROOT: Path | None = Path(_data_root).expanduser() if _data_root else None

def data_path(name: str, override: str) -> Path | None:
    value = os.environ.get(override)
    return Path(value).expanduser() if value else (DATA_ROOT / name if DATA_ROOT else None)

ASSET_BANK = data_path("asset", "NAVSAFE_ASSET_BANK")
MODEL_ZOO = data_path("model_zoo", "NAVSAFE_MODEL_ZOO")

# --- gait ------------------------------------------------------------------
# An inserted pedestrian is one static gaussian asset, and the render server's
# only per-frame lever is a rigid pose, so a walking actor has to be baked as a
# short bank of posed copies (``editing/assets/animate.py``).  Two inputs:
#
#     KIMODO      the motion generator's checkout, for its SOMA skinned mesh
#                 (``skin_standard.npz``: bind vertices, LBS weights, the bind
#                 rig) and its shipped example motions.  Nothing is trained
#                 here; the repo is read for assets only.
#     GAIT_BANK   where the baked pose banks are written -- one directory per
#                 asset, holding the phases of one gait cycle plus the
#                 ``bank.json`` that records the stride they were baked at.
#
# A pose bank is DERIVED FROM AN ASSET and not from a scenario or a run, so it
# lives beside the asset library rather than in a corpus or a run dir.
KIMODO = Path(os.environ.get("NAVSAFE_KIMODO", str(Path.home() / "tools/kimodo")))
GAIT_BANK = data_path("gait_bank", "NAVSAFE_GAIT_BANK")

# The quadruped template, for animals.  kimodo ships a skinned human and no
# animal, so SMAL (3D Menagerie, CVPR 2017) supplies the four-legged one --
# a separate, gated download from https://smal.is.tue.mpg.de.  Point this at
# the unpacked ``smal_online_V1.0`` folder.
SMAL = Path(os.environ.get("NAVSAFE_SMAL", str(Path.home() / "data/smal_online_V1.0")))

# The skinned human the gait is applied through, and the motions available to
# bake from.  Both ship inside the kimodo checkout.
SOMA_SKIN = KIMODO / "kimodo/assets/skeletons/somaskel77/skin_standard.npz"
MOTIONS = KIMODO / "kimodo/assets/demo/examples"


def motion_npz(name: str = "kimodo-soma-rp/05_root_path") -> Path:
    """A generated motion, by example name under :data:`MOTIONS`.

    The default is the shipped 10 s "casually walking forward slowly" sample,
    which needs no model download and no gated text encoder.  Generating a new
    one with a custom prompt writes the same ``motion.npz`` layout, so it drops
    in here by path.
    """
    p = Path(name)
    return (p if p.is_absolute() else MOTIONS / p / "motion.npz")

# Where the published scenario bundles were downloaded to -- one self-contained
# directory per token, holding its Arrow, its offsets, its four usdz and a
# manifest.  There is deliberately no fallback: a bundle tree is something the
# reader fetches, so there is no deployment path to guess at.
_bundles = os.environ.get("NAVSAFE_BUNDLES")
BUNDLES = data_path("full_test", "NAVSAFE_BUNDLES")

# --- per-run outputs ------------------------------------------------------
# The other half of the deployment: what a render or an eval writes, which is
# new every time it runs.  A run is not a property of a scenario, so it must
# not be stored inside one, and it is not a variant of a corpus, so it must not
# be spelled by suffixing a corpus directory -- that is how `eval-clip`,
# `eval-lr5cam`, `recon-clip` and `navhard_logreplay_nuplan_base3dgut_<id>`
# came to exist, each an experiment encoded in a directory name.
#
#     <runs>/<YYYYMMDD>-<slug>/
#     ├── run.yaml     what produced this: config, git sha, seed ids
#     ├── eval/<seed>/<shape>/    exactly the tree run_eval.py already writes
#     ├── logs/  metrics/  render/
#     └── STATUS       queued | running | done | failed
#
# `runner/collect.py` takes `--eval-root`, so pointing it at
# `<run>/eval` needs no change: a run stays one directory that can be kept,
# published or deleted whole, and two runs can never overwrite each other.
RUNS = Path(os.environ.get("NAVSAFE_RUNS", str(WORK / "runs")))

# A slug has to survive being read six months later in a results table.
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHORTCODE = re.compile(r"^[a-z]{1,2}[0-9]{1,3}$")


def run_dir(slug: str, *, date: str | None = None, runs: Path | None = None) -> Path:
    """The directory for one render/eval campaign.

    ``date`` defaults to today as ``YYYYMMDD``; pass it explicitly to address a
    run made earlier.  The slug must say what the run was for -- ``j1``, ``b2``
    and friends are rejected on purpose, because a directory named after a
    counter is unreadable by the time the result matters.
    """
    if not _SLUG.match(slug) or len(slug) < 4 or _SHORTCODE.match(slug):
        raise ValueError(
            f"run slug {slug!r} is not descriptive: use lowercase words joined "
            f"by hyphens, at least 4 characters, saying what the run tests "
            f"(e.g. 'epdms-no-extcomfort', 'c10-oncoming-insert')")
    day = date or _dt.date.today().strftime("%Y%m%d")
    return (runs or RUNS) / f"{day}-{slug}"


def scene_dir(scene_id: str, corpus: Path | None = None) -> Path:
    """Everything belonging to one scenario, forever."""
    return (corpus or CORPUS) / scene_id


def arrow_dir(scene_id: str, corpus: Path | None = None) -> Path:
    """The scenario's Arrow, beside its recon rather than in a parallel tree.

    Keeping it here is what lets ``leaves/hosts.py`` resolve a scene id by
    globbing for ``arrow`` and reading the parent directory's name.
    """
    return scene_dir(scene_id, corpus) / "arrow"


def clips_dir(scene_id: str, corpus: Path | None = None) -> Path:
    """The NCore V4 clip the reconstruction was trained from.

    The scene id repeats inside; that nesting is historical and 2.7 TB of
    corpus is not worth restructuring to remove, so it is absorbed here.
    """
    return scene_dir(scene_id, corpus) / "clips" / scene_id


def ah_assets_dir(scene_id: str, corpus: Path | None = None) -> Path:
    """Per-object assets harvested from this scenario, and the replace manifest.

    Asset Harvester output is DERIVED FROM ONE SCENARIO and never forked by
    experiment, so it belongs in the scenario's own directory beside its clip,
    its recon and its Arrow -- not in a run dir and not in a parallel tree.

    For a stitched ``<token>_20s`` host, pass the host's scene id: one bank
    covers all four 5 s windows, because a car that crosses the window boundary
    is ONE track with one id and harvesting it four times would be four times
    the diffusion cost for the same asset.
    """
    return scene_dir(scene_id, corpus) / "ah_assets"


def recon_usdz(scene_id: str, corpus: Path | None = None) -> Path | None:
    """The trained reconstruction for one 5 s window, or None if absent.

    Two layouts are in the wild: car2sim writes under ``output_5cam/<clip>/``
    (it emits ``last.usdz`` mid-training, with no export stage), while a
    re-referenced or hand-placed recon sits flat beside it.  Callers asking
    "is this trained yet" must accept both, so neither may hard-code one.
    """
    base = scene_dir(scene_id, corpus)
    for candidate in (base / "output_5cam" / scene_id / "artifacts" / "last.usdz",
                      base / "artifacts" / "last.usdz"):
        if candidate.is_file():
            return candidate
    return None


# --- nuPlan layout --------------------------------------------------------
NUPLAN_V11 = NUPLAN_ROOT / "nuplan-v1.1"
NUPLAN_SPLITS = NUPLAN_V11 / "splits"
NUPLAN_SENSORS = NUPLAN_V11 / "sensor_blobs"
NUPLAN_MAPS = NUPLAN_V11 / "maps"
# The devkit's own map root differs from the NCore one on this deployment.
NUPLAN_MAPS_DEVKIT = Path(os.environ.get("NUPLAN_MAPS_ROOT", str(NUPLAN_ROOT / "maps")))

# --- cluster -------------------------------------------------------------
NAMESPACE = os.environ.get("NAVSAFE_NAMESPACE", "default")
POD = os.environ.get("NAVSAFE_POD", "navsafe-dev")
# Explicit checkpoint paths still override the published default.
WEIGHTS = Path(os.environ.get("NAVSAFE_WEIGHTS", str(
    MODEL_ZOO or Path.home() / "data/NavSafe/model_zoo")))
DEFAULT_CHECKPOINT = os.environ.get(
    "NAVSAFE_CHECKPOINT", str(WEIGHTS / "drivor/drivor_Nav1_25epochs.pth"))
# NRE Hydra overlay for the recommended recipe (docs/reconstruct_navsim_nuplan.md §5).
NRE_OVERLAY = Path(os.environ.get(
    "NAVSAFE_NRE_OVERLAY", str(Path(__file__).resolve().parents[1] / "gs3d_converter/configs/car2sim_6cam_static.yaml")))
# Import stubs the eval client needs on PYTHONPATH inside the NRE image.
NRE_STUBS = Path(os.environ.get("NAVSAFE_NRE_STUBS", str(Path(__file__).resolve().parents[2])))

# --- Asset Harvester ------------------------------------------------------
# NVIDIA Asset Harvester (github.com/NVIDIA/asset-harvester, Apache-2.0) turns
# the multi-view observations of one logged actor into a view-consistent 3DGS
# asset.  It is a third party with its own conda env (python 3.10, torch
# 2.10/cu128, gsplat from a pinned commit) that cannot share the NexusSim venv,
# so what lives here is a checkout, an env and the model checkpoints -- all on
# the PVC, because the pod running them is cycled every ~6 h and a 30 GB env
# must not be rebuilt each time.
#
#     <AH_HOME>/repo/            the checkout, incl. checkpoints/ and run.sh
#     <AH_HOME>/conda/envs/asset-harvester/   the env its scripts run in
AH_HOME = Path(os.environ.get("NAVSAFE_AH_HOME", str(Path.home() / "tools/asset_harvester")))
AH_REPO = Path(os.environ.get("NAVSAFE_AH_REPO", str(AH_HOME / "repo")))
AH_PYTHON = Path(os.environ.get(
    "NAVSAFE_AH_PYTHON", str(AH_HOME / "conda/envs/asset-harvester/bin/python")))


def describe() -> str:
    return "\n".join(f"{k:22} {v}" for k, v in (
        ("NAVSAFE_WORK", WORK), ("NAVSAFE_CORPUS", CORPUS),
        ("NAVSAFE_NAVHARD_CORPUS", NAVHARD_CORPUS), ("NAVSAFE_RUNS", RUNS),
        ("NUPLAN_ROOT", NUPLAN_ROOT),
        ("NAVSAFE_ROOT", NAVSAFE_ROOT), ("NAVSAFE_LOCAL_DB", LOCAL_DB),
        ("namespace", NAMESPACE), ("pod", POD),
    ))


if __name__ == "__main__":
    print(describe())
