"""Mode A: emit Kubernetes Jobs for the GPU stages of a NavSafe seed.

The CPU stages (NCore, Arrow) run wherever a NavSafe env exists; the GPU
stages run in NVIDIA's gated NRE containers, which is what makes Mode A a
split-service deployment.  One Job per (seed, stage):

    aux     nre-tools-ga   ncore-aux-data                    ~30 min
    train   nre-ga         car2sim_6cam_static, 160k iters   ~5.5 h on one 3090
    export  nre-ga         usdz (+ plys, tracks)             ~5 min
    eval    nre-ga         serve-grpc + closed-loop rollout, both on one GPU

Container conventions are taken from the reconstruction jobs already proven on
this cluster (``navsafe-train-*``), not invented here:

* ``nre-tools-ga`` is invoked through its own entrypoint -- ``args`` only, no
  shell and no ``/app/run`` prefix (that binary does not exist in the tools
  image).
* ``nre-ga`` runs under ``/opt/nvidia/nvidia_entrypoint.sh bash -c`` so a
  multi-step script can prepare the Hydra overlay before calling ``/app/run``.
* ``ncore-aux-data`` writes ``<clip>.aux.*.zarr.itar`` but NRE training
  discovers ``pai_<clip>.aux.*`` -- the train script symlinks them.
* ``logger.run_id=<seed>`` pins the run directory name so export does not have
  to glob for it.

Every stage is idempotent: it short-circuits when its output already exists, so
a Job can be resubmitted after a pre-emption without redoing work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg
from navsafe.benchmark.world import deployment


NRE_IMAGE = ("nvcr.io/nvidia/nre/nre-ga@sha256:"
             "6e0caa70a9148490552520c3dde9ee665c8d094ca10b814601cb8bb306567c90")
NRE_TOOLS_IMAGE = ("nvcr.io/nvidia/nre/nre-tools-ga@sha256:"
                   "46d78256219a6541601fd16517c7edca3e7a6ec9634f1858f7977279bd98f9ff")

NRE_ENTRYPOINT = ["/opt/nvidia/nvidia_entrypoint.sh", "bash", "-c"]
C_HYPERION_CFG_DIR = ("/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles"
                      "/_main/configs/apps/prod/Hyperion-8.1")
OVERLAY_SRC = str(cfg.NRE_OVERLAY)

CAMS = ("[camera_pcam_f0,camera_pcam_b0,camera_pcam_l0,camera_pcam_l1,"
        "camera_pcam_l2,camera_pcam_r0,camera_pcam_r1,camera_pcam_r2]")

# GPU nodes the proven reconstruction jobs ran on.
NODES = deployment.nodes()


def _pod_spec(image: str, *, command=None, args, gpus=1, cpu="4", mem="32Gi",
              ephemeral="120Gi", nodes=None) -> dict:
    container = {
        "name": "gpu-container",
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "args": args,
        "env": [{"name": "HYDRA_FULL_ERROR", "value": "1"}],
        "resources": {
            "limits": {"nvidia.com/gpu": str(gpus), "cpu": cpu, "memory": mem,
                       "ephemeral-storage": ephemeral},
            "requests": {"nvidia.com/gpu": str(gpus), "cpu": cpu, "memory": mem,
                         "ephemeral-storage": ephemeral},
        },
        "volumeMounts": [
            {"name": "dshm", "mountPath": "/dev/shm"},
            # Policy checkpoints live here; eval cannot score anything without
            # them, and the reconstruction stages simply ignore the mount.
        ],
    }
    if command:
        container["command"] = command
    return deployment.configure({
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
        ],
    }, nodes)


def _job(name: str, spec: dict) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": cfg.NAMESPACE,
                     "labels": {"app": "navsafe-dev", "k8s-app": name}},
        "spec": {"backoffLimit": 1, "ttlSecondsAfterFinished": 1209600,
                 "template": {"metadata": {"labels": {"k8s-app": name}}, "spec": spec}},
    }


def _clipdir(seed: dict) -> str:
    tok = seed["seed_id"]
    return f"{seed['artifacts']['ncore']}/clips/{tok}"


def build_aux(seed: dict):
    """nre-tools entrypoint takes the subcommand + flags as bare args."""
    tok, clip = seed["seed_id"], _clipdir(seed)
    args = [
        "ncore-aux-data",
        f"--dataset-path={clip}/pai_{tok}.json",
        f"--output-dir={clip}",
        "--segmentation-backend=mask2former",
        "--zarr-store-type=itar",
        "--store-meta",
        "--lidar-seg-camvis",
        "--ego-mask",
    ]
    return NRE_TOOLS_IMAGE, None, args, {"cpu": "2", "mem": "12Gi", "ephemeral": "40Gi"}


def build_train(seed: dict):
    tok, clip = seed["seed_id"], _clipdir(seed)
    out = seed["artifacts"]["recon"]
    script = f"""set -ex
C={tok}; CLIPDIR={clip}; OUT={out}
if [ -f "$OUT/$C/checkpoints/last.ckpt" ]; then echo "TRAIN_SKIP $C (checkpoint exists)"; exit 0; fi
CFGDIR={C_HYPERION_CFG_DIR}
cp {OVERLAY_SRC} "$CFGDIR/"
# aux-data writes <C>.aux.*.zarr.itar; NRE discovers pai_<C>.aux.* -- symlink.
( cd "$CLIPDIR" && for f in $C.aux.*.zarr.itar; do ln -sf "$f" "pai_$f"; done )
/app/run --config-name=configs/apps/prod/Hyperion-8.1/car2sim_6cam_static.yaml \\
  mode=train logger=dummy \\
  dataset.path=$CLIPDIR/pai_$C.json \\
  out_dir=$OUT \\
  logger.run_id="$C" \\
  dataset.aux_data=True \\
  dataset.camera_ids={CAMS} \\
  dataset.train_camera_ids={CAMS} \\
  dataset.val_camera_ids={CAMS} \\
  dataset.lidar_ids=[lidar_top_360fov] \\
  dataset.train_lidar_ids=[lidar_top_360fov] \\
  dataset.val_lidar_ids=[lidar_top_360fov] \\
  dataset.n_train_sample_lidar_rays=0 \\
  dataset.cuboid_tracks_params.track_label_sources=[GT_ANNOTATION] \\
  model.layers.dynamic_rigids.tracks.label_classes=[NuPlanBoxDetectionLabel.VEHICLE,NuPlanBoxDetectionLabel.BICYCLE] \\
  model.layers.dynamic_deformables.tracks.label_classes=[NuPlanBoxDetectionLabel.PEDESTRIAN] \\
  model.layers.dynamic_rigids.initialization.name=camera-dynamic-tracks \\
  +model.layers.dynamic_rigids.initialization.camera_ids=null \\
  +model.layers.dynamic_rigids.initialization.step_frame=1 \\
  +model.layers.dynamic_rigids.initialization.fill_with_random_points=true \\
  model.layers.dynamic_deformables.initialization.name=camera-dynamic-tracks \\
  +model.layers.dynamic_deformables.initialization.camera_ids=null \\
  +model.layers.dynamic_deformables.initialization.step_frame=1 \\
  +model.layers.dynamic_deformables.initialization.fill_with_random_points=true \\
  model.layers.road.initialization.name=lidar-rig-trajectory \\
  model.layers.road.initialization.default_scale=0.1 \\
  +model.layers.road.initialization.num_near_points=100000 \\
  +model.layers.road.initialization.num_far_points=100000 \\
  +model.layers.road.initialization.far_radius_factor=20 \\
  +model.layers.road.initialization.observation_scale_factor=0.01 \\
  +model.layers.road.initialization.lidar_ids=null \\
  +model.layers.road.initialization.camera_ids=null \\
  +model.layers.road.initialization.non_dynamic_points_only=true \\
  checkpoint.artifact.mesh.ground.enabled=false
test -f $OUT/$C/checkpoints/last.ckpt
echo "TRAIN_DONE $C"
"""
    return NRE_IMAGE, NRE_ENTRYPOINT, [script], {"cpu": "4", "mem": "32Gi", "ephemeral": "120Gi"}


def build_export(seed: dict):
    tok = seed["seed_id"]
    run = f"{seed['artifacts']['recon']}/{tok}"
    exp = seed["artifacts"]["export"]
    script = f"""set -ex
C={tok}; RUN={run}; EXP={exp}
test -f "$RUN/checkpoints/last.ckpt"
mkdir -p "$EXP/usd-out"
# Training already writes artifacts/last.usdz; re-exporting it costs ~5 min for
# an identical file, so only run export-usdz-artifact when it is absent.
if [ -f "$RUN/artifacts/last.usdz" ]; then
  cp "$RUN/artifacts/last.usdz" "$EXP/usd-out/$C.usdz"
elif ! ls "$EXP"/usd-out/*.usdz >/dev/null 2>&1; then
  /app/run export-usdz-artifact --config-name "$RUN/config/parsed.yaml" --checkpoint-name last.ckpt --output-dir "$RUN/usd-out"
  cp "$RUN"/usd-out/*.usdz "$EXP/usd-out/$C.usdz"
fi
# PLYs and world-frame tracks are for actor editing / asset work downstream,
# not for serve-grpc, so a failure here must not cost the usdz.
if [ ! -d "$EXP/plys" ]; then
  /app/run export-gaussian-plys  --config-name "$RUN/config/parsed.yaml" --checkpoint-name last.ckpt --output-dir "$RUN/plys" --format _3DGS || true
  /app/run export-sequence-tracks --config-name "$RUN/config/parsed.yaml" --checkpoint-path "$RUN/checkpoints/last.ckpt" --output-dir "$RUN" --world-frame --format json || true
  cp -r "$RUN"/plys "$EXP"/ 2>/dev/null || true
  cp "$RUN"/*track*.json "$EXP"/ 2>/dev/null || true
fi
ls -la "$EXP/usd-out"
test -s "$EXP/usd-out/$C.usdz"
echo "EXPORT_DONE $C"
"""
    return NRE_IMAGE, NRE_ENTRYPOINT, [script], {"cpu": "4", "mem": "32Gi", "ephemeral": "120Gi"}


def build_eval(seed: dict, *, run: str, regime: str = "log_replay",
               policy: str = "drivor",
               checkpoint: str = "",
               enable_harmonizer: bool = True) -> tuple:
    """Closed-loop eval with its own renderer, co-located on one GPU.

    The doc's high-throughput shape: rather than every eval client sharing one
    long-lived ``serve-grpc``, each job runs its own against a single seed's
    usdz (serve + rollout + the harmonizer's diffusion model together on one
    24 GB card), so jobs share no renderer -- no artifact-glob to switch
    between scenes, and one job dying takes nothing else with it. The one
    piece of shared state is the harmonizer checkpoint cache on the PVC.

    The NRE image supplies ``/app/run serve-grpc``; the eval client is this
    repo, built into a venv beside it.  The uv cache lives on the PVC so only
    the first eval job pays the wheel-download cost.

    ``enable_harmonizer=False`` serves without the DiffusionHarmonizer
    postprocessing pass -- the ablation arm for "does harmonizer move eval
    scores". Keep arms consistent within a comparison.
    """
    tok = seed["seed_id"]
    exp = seed["artifacts"]["export"]
    # The job's own success check reads this path, and run_eval.py writes it,
    # so the two must be built the same way. Passing --tag {regime} below
    # pins run_eval's output directory to the regime rather than to the
    # episode shape, which is what this path has always assumed.
    out = f"{cfg.run_dir(run)}/eval/{tok}/{regime}"
    work = str(Path(seed["artifacts"]["ncore"]).parent.parent)
    checkpoint = checkpoint or cfg.DEFAULT_CHECKPOINT
    uv_cache, navsafe = cfg.UV_CACHE, cfg.NAVSAFE_ROOT
    # Checkpoint cache on the PVC: the NRE downloader stages into a temp file
    # and renames (HF-hub machinery), so a concurrent first wave of jobs at
    # worst downloads redundantly rather than corrupting the cache; every
    # later wave reuses it.
    harmonizer_setup = (
        "# Harmonizer postproc default-on; checkpoint cache on the PVC.\n"
        f"mkdir -p {cfg.HARMONIZER_CACHE}\n") if enable_harmonizer else ""
    harmonizer_flags = (
        f"  --enable-harmonizer --harmonizer-cache {cfg.HARMONIZER_CACHE} \\\n"
        if enable_harmonizer else "")
    script = f"""set -eo pipefail
C={tok}
trap 'echo "=== serve-grpc tail ==="; tail -80 /tmp/serve.log 2>/dev/null || true' EXIT

# Fail before the ~20 min of bootstrap + scene load if an input is missing.
test -s {checkpoint} || {{ echo "FATAL: checkpoint not found: {checkpoint}"; exit 1; }}
test -s {exp}/usd-out/$C.usdz || {{ echo "FATAL: no usdz for $C"; exit 1; }}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl python3-venv \
  libglu1-mesa libxt6 libgl1 libglx0 libegl1 libglib2.0-0 \
  libxrandr2 libxinerama1 libxcursor1 libxi6 libxext6 libxrender1 \
  libx11-6 libxfixes3 libxdamage1 libsm6 libice6 libgomp1 >/dev/null

export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
# Wheel cache on the PVC: the first eval job downloads the stack, the rest
# reuse it. Without this every job re-fetches torch and IsaacSim.
export UV_CACHE_DIR={uv_cache}
export UV_PROJECT_ENVIRONMENT=/opt/navsafe/venv
export VIRTUAL_ENV=/opt/navsafe/venv
for i in 1 2 3 4; do
  command -v uv >/dev/null && break
  python -m pip install --quiet --user uv 2>/dev/null || \
    pip install --quiet --user uv 2>/dev/null || \
    curl -LsSf https://astral.sh/uv/install.sh | sh
  hash -r; sleep 10
done
command -v uv >/dev/null || {{ echo "FATAL: uv unavailable"; exit 1; }}

# Build against the checkout on the PVC (so edits are live) but install into
# /root, never into the shared repo -- concurrent eval jobs must not race on
# a .venv inside it.
cd {navsafe}
if ! /opt/navsafe/venv/bin/python -c "import isaacsim, isaaclab, navsafe" 2>/dev/null; then
  uv sync --all-extras --python 3.12
  [ -d /root/IsaacLab/.git ] || git clone --depth 1 --branch v3.0.0-beta \
    https://github.com/isaac-sim/IsaacLab.git /root/IsaacLab
  for ext in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    d=/root/IsaacLab/source/$ext
    [ -d "$d" ] && uv pip install --python /opt/navsafe/venv/bin/python --no-deps -e "$d"
  done
  uv pip install --python /opt/navsafe/venv/bin/python lazy_loader einops
fi
/opt/navsafe/venv/bin/python -m pip install --quiet grpcio protobuf 2>&1 | tail -1 || true

# --- renderer, this seed only -------------------------------------------
ART={exp}/usd-out/$C.usdz
test -s "$ART"
{harmonizer_setup}CUDA_VISIBLE_DEVICES=0 /app/run serve-grpc --host 0.0.0.0 \
  --enable-editing-actors --renderer default \
{harmonizer_flags}  --artifact-glob "$ART" > /tmp/serve.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 120); do
  kill -0 $SERVE_PID 2>/dev/null || {{ echo "serve-grpc exited early"; exit 1; }}
  (exec 3<>/dev/tcp/localhost/8080) 2>/dev/null && {{ echo "serve up after ${{i}}0s"; break; }}
  sleep 10
done

# --- rollout -------------------------------------------------------------
mkdir -p {out}
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=0 \
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  /opt/navsafe/venv/bin/python \
    {navsafe}/navsafe/benchmark/world/run_eval.py \
    --seed {work}/seeds/$C \
    --grpc-host localhost \
    --run {run} \
    --tag {regime} \
    --traffic-mode {regime} \
    --model-type {policy} \
    --checkpoint {checkpoint}
kill $SERVE_PID 2>/dev/null || true
test -f {out}/metrics.json
echo "EVAL_DONE $C {regime}"
"""
    return NRE_IMAGE, NRE_ENTRYPOINT, [script], {"cpu": "8", "mem": "48Gi",
                                                 "ephemeral": "150Gi"}


# The nodes the workspace pod's own affinity list enables, which is where the
# campaign is asked to stay. `NODES` above is what the reconstruction stages
# have always used and is left alone; a fan-out of eval jobs picks from here.
EDIT_EVAL_NODES = deployment.nodes()


def build_editing_eval(target: str, *, run: str, policy: str = "drivor",
                       checkpoint: str = "", eval_frames: int = 200,
                       corpus: str = "", recipes: str = "",
                       out_root: str = "") -> tuple:
    """One edited NavSafe scenario, renderer and rollout on the same GPU.

    Same shape as :func:`build_eval` and for the same reason -- a job that owns
    its renderer shares no state, so one dying takes nothing else with it --
    but for an EDITED scenario rather than a mined seed. Three things differ:

    * the host is a stitched 20 s clip, so the renderer is pointed at its four
      sub-clip reconstructions and the rollout needs the hand-off table that
      says which window each belongs to;
    * the scene comes from a frozen recipe, so the client is the evaluator
      with ``--recipe`` rather than ``run_eval.py`` with a regime;
    * the renderer's ``--artifact-glob`` matches four files instead of the
      whole corpus. A shared server globs 1636 scenes and spends ~7 minutes
      scanning before it will answer, which is also the window in which an
      eval that connects too early dies; four files load in seconds.

    ``target`` is ``<LEAF>.<token>``, the recipe's own basename.
    """
    leaf, _, tok = str(target).partition(".")
    if not leaf or not tok:
        raise ValueError(f"target {target!r} must be '<LEAF>.<token>'")
    corpus = corpus or str(cfg.CORPUS)
    recipes = recipes or f"{cfg.WORK}/recipes/editing90"
    checkpoint = checkpoint or cfg.DEFAULT_CHECKPOINT
    # `out_root` exists so a fan-out can land in a campaign directory that is
    # ALREADY OPEN: `cfg.run_dir` stamps today's date, so a batch started on a
    # second day would otherwise split one campaign across two directories and
    # `collect.py` would aggregate half of it.
    out = f"{out_root or cfg.run_dir(run)}/{target}"
    navsafe = cfg.NAVSAFE_ROOT
    shared_venv = f"{cfg.WORK}/ns-venv"
    script = f"""set -eo pipefail
T={tok}
trap 'echo "=== serve-grpc tail ==="; tail -60 /tmp/serve.log 2>/dev/null || true' EXIT

test -s {checkpoint} || {{ echo "FATAL: no checkpoint at {checkpoint}"; exit 1; }}
test -s {recipes}/{target}.yaml || {{ echo "FATAL: no recipe {target}.yaml"; exit 1; }}
test -d {corpus}/${{T}}_20s/arrow || {{ echo "FATAL: no arrow for $T"; exit 1; }}
N=$(ls {corpus}/${{T}}s?/output_5cam/*/artifacts/last.usdz 2>/dev/null | wc -l)
[ "$N" = 4 ] || {{ echo "FATAL: $T has $N/4 reconstructions"; exit 1; }}

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl python3-venv \
  libglu1-mesa libxt6 libgl1 libglx0 libegl1 libglib2.0-0 \
  libxrandr2 libxinerama1 libxcursor1 libxi6 libxext6 libxrender1 \
  libx11-6 libxfixes3 libxdamage1 libsm6 libice6 libgomp1 >/dev/null

export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export UV_CACHE_DIR={cfg.UV_CACHE}
# A SHARED venv on the PVC, or build one. Building it per job costs ~40
# minutes of wall clock with a GPU sitting idle: the uv cache is on CephFS
# while the venv goes on the pod's own disk, so uv cannot hardlink and copies
# all 360 packages -- torch and IsaacSim included -- once per job. The
# The shared copy has to be SELF-CONTAINED: a venv built by uv points its
# `bin/python` and its `pyvenv.cfg` at a uv-managed interpreter under $HOME,
# and $HOME is /root in the workspace pod but /home in this image, so the
# symlink dangles here. Its interpreter therefore lives on the PVC beside it.
# Falls back to building when the shared one is absent or does not import, so
# a missing PVC copy costs time rather than the run.
VENV=/opt/navsafe/venv
if [ -x {shared_venv}/bin/python ]; then
  # ACCEPT_EULA here as well as at the rollout: `import isaacsim` PROMPTS for
  # the Omniverse licence, and with no tty it fails on EOF -- so the probe
  # answered "this venv does not work" for a venv that works perfectly, and
  # every job silently rebuilt its own.
  if ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES {shared_venv}/bin/python \
       -c "import isaacsim, isaaclab, navsafe" 2>/dev/null; then
    VENV={shared_venv}
    echo "using the shared venv at {shared_venv}"
  fi
fi
export UV_PROJECT_ENVIRONMENT=$VENV VIRTUAL_ENV=$VENV
for i in 1 2 3 4; do
  command -v uv >/dev/null && break
  python -m pip install --quiet --user uv 2>/dev/null || \
    curl -LsSf https://astral.sh/uv/install.sh | sh
  hash -r; sleep 10
done
command -v uv >/dev/null || {{ echo "FATAL: uv unavailable"; exit 1; }}
cd {navsafe}
if ! $VENV/bin/python -c "import isaacsim, isaaclab, navsafe" 2>/dev/null; then
  uv sync --all-extras --python 3.12
  [ -d /root/IsaacLab/.git ] || git clone --depth 1 --branch v3.0.0-beta \
    https://github.com/isaac-sim/IsaacLab.git /root/IsaacLab
  for ext in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    d=/root/IsaacLab/source/$ext
    [ -d "$d" ] && uv pip install --python $VENV/bin/python --no-deps -e "$d"
  done
  uv pip install --python $VENV/bin/python lazy_loader einops
fi
$VENV/bin/python -m pip install --quiet grpcio protobuf 2>&1 | tail -1 || true

# --- renderer: this host's four sub-clips only ---------------------------
CUDA_VISIBLE_DEVICES=0 /app/run serve-grpc --host 0.0.0.0 \
  --enable-editing-actors --renderer default \
  --artifact-glob "{corpus}/$T""s?/output_5cam/*/artifacts/last.usdz" \
  > /tmp/serve.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 120); do
  kill -0 $SERVE_PID 2>/dev/null || {{ echo "serve-grpc exited early"; exit 1; }}
  (exec 3<>/dev/tcp/localhost/8080) 2>/dev/null && {{ echo "serve up after ${{i}}0s"; break; }}
  sleep 10
done

# --- rollout -------------------------------------------------------------
# Import NavSafe and its bundled renderer protocol package from the checkout.
export PYTHONPATH={navsafe}
export NUREC_GRPC_HOST=localhost NUREC_GRPC_CAM_RIG=${{NUREC_GRPC_CAM_RIG:-recon}} NUREC_GRPC_TIMEOUT_S=600
export PY123D_RECENTER=1 NAVSAFE_NO_CAM_MAP_LINES=1 NAVSAFE_WORK={cfg.WORK}
# The hand-off says which of the four recons owns each frame. Derived here
# rather than baked into the Job, because it carries absolute paths that are
# only valid on the machine that reads them.
H=$($VENV/bin/python -m navsafe.benchmark handoff --token $T 2>/dev/null | tail -1)
case "$H" in *NUREC_GRPC_HANDOFF*) eval "$H";; *) echo "FATAL: no hand-off for $T"; exit 1;; esac

mkdir -p {out}
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 CUDA_VISIBLE_DEVICES=0 \
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  $VENV/bin/python {navsafe}/navsafe/cli/eval_entry.py \
    --scenario-source py123d --py123d-data-root {corpus}/${{T}}_20s/arrow \
    --py123d-scene-index 0 --nurec-work-dir {corpus} \
    --render-backend nurec_grpc \
    --model-type {policy} --checkpoint {checkpoint} \
    --recipe {recipes}/{target}.yaml \
    --traffic-mode navsafe --ego-replay-frames 8 --eval-frames {eval_frames} \
    --terminate-on-collision --execution-mode controller --controller lqr \
    --replan-rate 5 --camera-resolution-scale 1.0 --enable-vis \
    --vis-cameras CAM_L0,CAM_B0 --nurec-cameras CAM_F0,CAM_B0,CAM_L0 \
    --log-level INFO --output-dir {out}
kill $SERVE_PID 2>/dev/null || true
test -f {out}/navsafe_metrics.json
echo "EVAL_DONE {target}"
"""
    return NRE_IMAGE, NRE_ENTRYPOINT, [script], {"cpu": "8", "mem": "48Gi",
                                                 "ephemeral": "150Gi",
                                                 "nodes": EDIT_EVAL_NODES}


BUILDERS = {"aux": build_aux, "train": build_train, "export": build_export,
            "eval": build_eval}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", help="seed dir or seed.json (every stage but "
                                   "editing-eval)")
    # An EDITED scenario has no seed: it is a frozen recipe against a host in
    # the corpus, so it is named by the recipe rather than by a mined seed dir.
    ap.add_argument("--target", help="editing-eval only: '<LEAF>.<token>', the "
                                     "recipe's own basename")
    ap.add_argument("--eval-frames", type=int, default=200,
                    help="editing-eval only: episode length")
    ap.add_argument("--out-root", default="",
                    help="editing-eval only: write into this campaign "
                         "directory instead of a fresh cfg.run_dir(<run>). Use "
                         "when the fan-out joins a campaign already in flight.")
    ap.add_argument("--stage", required=True,
                    choices=sorted(set(BUILDERS) | {"editing-eval"}))
    ap.add_argument("--out", default="-")
    ap.add_argument("--suffix", default="")
    # eval only: which interaction regime and which policy to score.
    ap.add_argument("--regime", default="log_replay",
                    choices=["log_replay", "idm"])
    ap.add_argument("--policy", default="drivor")
    ap.add_argument("--run", default="",
                    help="eval only, REQUIRED: the campaign these jobs belong "
                         "to. All their output lands under "
                         "NAVSAFE_RUNS/<today>-<slug>/, so a sweep is one "
                         "directory and a later sweep cannot overwrite it.")
    ap.add_argument("--checkpoint", default=cfg.DEFAULT_CHECKPOINT)
    ap.add_argument("--no-harmonizer", action="store_true",
                    help="eval only -- serve without DiffusionHarmonizer "
                         "postprocessing (ablation; default is harmonizer ON)")
    args = ap.parse_args()

    # Argument validation before any I/O, so a missing --run reports itself
    # rather than surfacing as whatever the seed read happens to fail on.
    if args.stage in ("eval", "editing-eval") and not args.run:
        ap.error(f"--run is required for --stage {args.stage}: name the campaign "
                 "these jobs belong to (e.g. --run epdms-no-extcomfort)")
    if args.stage == "editing-eval":
        if not args.target:
            ap.error("--stage editing-eval needs --target '<LEAF>.<token>'")
    elif not args.seed:
        ap.error(f"--stage {args.stage} needs --seed")

    import yaml

    if args.stage == "editing-eval":
        image, command, cargs, res = build_editing_eval(
            args.target, run=args.run, policy=args.policy,
            checkpoint=args.checkpoint, eval_frames=args.eval_frames,
            out_root=args.out_root)
        # k8s names are lowercase and take no dots; the recipe's own name is
        # kept otherwise so a job is traceable to the scenario it ran.
        slug = args.target.replace(".", "-").lower()
        name = f"navsafe-edit-{slug}{args.suffix}"
        job = _job(name, _pod_spec(image, command=command, args=cargs, **res))
        text = yaml.safe_dump(job, sort_keys=False, width=100_000)
        if args.out == "-":
            sys.stdout.write(text)
        else:
            Path(args.out).write_text(text)
            print(f"wrote {args.out}  (job {name})")
        return 0

    p = Path(args.seed)
    seed = json.loads((p / "seed.json" if p.is_dir() else p).read_text())
    kw = {}
    if args.stage == "eval":
        kw = {"run": args.run, "regime": args.regime, "policy": args.policy,
              "checkpoint": args.checkpoint,
              "enable_harmonizer": not args.no_harmonizer}
    image, command, cargs, res = BUILDERS[args.stage](seed, **kw)
    # The regime is part of the job identity: the same seed is evaluated under
    # each of them, and they must not collide on one job name. Underscores are
    # not legal in a k8s object name, so the regime is hyphenated here while
    # staying underscored everywhere it names a directory or a traffic mode.
    tag = f"-{args.regime.replace('_', '-')}" if args.stage == "eval" else ""
    name = f"navsafe-dev-{args.stage}-{seed['seed_id']}{tag}{args.suffix}"
    job = _job(name, _pod_spec(image, command=command, args=cargs, **res))
    text = yaml.safe_dump(job, sort_keys=False, width=100_000, default_style=None)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}  (job {name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
