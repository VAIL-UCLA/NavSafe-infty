#!/usr/bin/env bash
# Serve one navhard421 bundle and run it: closed-loop policy eval, or the
# full ego-replay render that isolates reconstruction quality from policy
# behaviour.
#
#   run_bundle_eval.sh TOKEN [MODE] [MODEL] [CHECKPOINT] [extra eval_py123d args...]
#     MODE   policy (default) | replay
#
# Anything after CHECKPOINT is appended verbatim to the eval_py123d.py command
# line, so an execution-mode or controller sweep does not need a second copy of
# this script. Later flags win, argparse-style, which is what lets an extra
# --replan-rate override the one set below.
#
# policy : 20 warm-up + semantic termination, semi_reactive traffic, HUD overlay on
# replay : 200 replayed frames, no policy control, cam_f0 written unannotated
#
# The output directory holds the NavSafe result and the render only; the eval log
# goes to "$OUT.log" beside it and the other metric families' artifacts are
# pruned (see PRUNE below, NAVSAFE_KEEP_ARTIFACTS=1 to keep them).
#
# Starts serve-grpc on the bundle, waits for its scene list, runs the eval, and
# stops the server again. The scene list is CHECKED, not assumed: a renderer
# that does not hold a scene id falls back to raster silently, which reads as a
# reconstruction-quality problem rather than a configuration one.
set -euo pipefail

TOKEN="${1:?token}"
MODE="${2:-policy}"
MODEL="${3:-drivor}"
# Deployment paths are the caller's, never this file's: see
# docs/navsafe_eval.md for the variables and what each one is.
CKPT="${4:-${NAVSAFE_CHECKPOINT:?set NAVSAFE_CHECKPOINT (or pass the checkpoint as arg 4)}}"
# Consume the positionals; whatever remains is passed through to the eval.
shift $(( $# < 4 ? $# : 4 ))
EXTRA=("$@")

REPO="${NAVSAFE_ROOT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
BUNDLES="$(python -c 'from navsafe.benchmark.config import BUNDLES; import sys; sys.exit("Set NAVSAFE_DATA_ROOT or NAVSAFE_BUNDLES") if BUNDLES is None else print(BUNDLES)')"
B="$BUNDLES/$TOKEN"
PORT="${NUREC_PORT:-8088}"
GPU="${NAVSAFE_GPU:-0}"
IMAGE="${NRE_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"   # untagged pulls :latest (~28 GB)
# Either export NGC_API_KEY yourself or point NGC_ENV_FILE at a file that does.
NGC_ENV="${NGC_ENV_FILE:-}"
PY="${NAVSAFE_PYTHON:-python}"
NAME="nre_bundle_$PORT"

[ -f "$B/manifest.json" ] || { echo "no bundle at $B (run bundle.py first)" >&2; exit 2; }
[ -d "$B/arrow" ] || { echo "$B has no arrow/ (run make_arrow.sh first)" >&2; exit 2; }

case "$MODE" in
  # --terminate-on-collision is the NavSafe convention (termination.py:
  # CONTACT_NOT_AT_FAULT ends the episode). Strict-protocol ruling
  # (2026-09-01): ANY contact ends the run; fault only decides whether the
  # ending is charged to the policy. Only the policy mode takes it; a replay
  # run follows the log and should render all 200 frames regardless.
  # --execution-mode controller: the ego is driven by the tracker + bicycle
  # model rather than teleported along the plan, so a trajectory the vehicle
  # cannot actually track shows up as tracking error instead of being executed
  # exactly. Pure pursuit is the shared, plant-adapted outer protocol for both
  # arms. It is not advertised as bit-identical to BridgeSim's PDM-specific
  # PID adapter; it prevents the outer LQR seam from being mistaken for a PDM
  # planner failure (0ebb... was RC100/SR-false under LQR and RC100/SR-true
  # under this controller with the selected PDM planner held fixed).
  # NAVSAFE_EXECUTION_MODE=teleport reproduces the pre-2026-08-12 cells, which
  # additionally need --controller pure_pursuit.
  # semi_reactive is the NavSafe default and the only reactive mode that is
  # wired: background traffic takes over from the log and responds to the ego,
  # so a policy that blocks a lane is answered instead of driven through.
  # eval_py123d.py's own default is still log_replay (every other scenario
  # source uses it), which is why this is set here rather than assumed;
  # NAVSAFE_TRAFFIC=log_replay reproduces a non-reactive run.
  policy) OUT="$REPO/output/navhard421_$TOKEN"
          EVAL_WINDOW=()
          # Omitted means EvaluationConfig.eval_frames=None: route execution
          # continues until a semantic terminal condition rather than a frame
          # budget. Callers can still request a bounded diagnostic window.
          if [ -n "${NAVSAFE_EVAL_FRAMES:-}" ]; then
            EVAL_WINDOW=(--eval-frames "$NAVSAFE_EVAL_FRAMES")
          fi
          SHAPE=(--traffic-mode "${NAVSAFE_TRAFFIC:-semi_reactive}"
                 "${EVAL_WINDOW[@]}"
                 --route-time-limit-s "${NAVSAFE_ROUTE_TIME_LIMIT_S:-0}"
                 --ego-replay-frames 8
                 --terminate-on-collision
                 --controller "${NAVSAFE_CONTROLLER:-pure_pursuit}"
                 --execution-mode "${NAVSAFE_EXECUTION_MODE:-controller}") ;;
  # A replay render follows the logged ego, so its traffic follows the log too:
  # reactive background here would deviate from the trajectory the
  # reconstruction was trained on, which is the one thing this mode exists to
  # hold fixed. Overridable, but that is no longer a reconstruction check.
  replay) OUT="$REPO/output/navhard421_${TOKEN}_replay"
          SHAPE=(--traffic-mode "${NAVSAFE_TRAFFIC:-log_replay}"
                 --controller pure_pursuit
                 --ego-replay-frames 200 --eval-frames 0) ;;
  *) echo "MODE must be policy or replay" >&2; exit 2 ;;
esac
# The default path is per-token, not per-model, so a second policy on the same
# token would silently overwrite the first. NAVSAFE_OUT keeps them side by side.
OUT="${NAVSAFE_OUT:-$OUT}"
# Front camera keeps its HUD, plan ribbon and route dots; only the projected
# grey map polylines come off, because a photoreal reconstruction already has
# the road in the pixels. NEXUSSIM_NO_OVERLAY=1 strips the camera annotations
# entirely (for video); the top-down is annotated either way.
export NEXUSSIM_NO_CAM_MAP_LINES="${NEXUSSIM_NO_CAM_MAP_LINES:-1}"
export NEXUSSIM_NO_OVERLAY="${NEXUSSIM_NO_OVERLAY:-0}"
# Vis-only extra cameras (NAVSAFE_VIS_CAMS=CAM_B0 adds the rear view). These are
# rendered on top of the policy's own camera set and never reach inference, so
# adding one changes the artifacts and the render cost, not the score.
VIS_CAMS=()
if [ -n "${NAVSAFE_VIS_CAMS:-}" ]; then
  VIS_CAMS=(--vis-cameras "$NAVSAFE_VIS_CAMS")
fi
# Score-only sweeps do not need hundreds of JPEGs and four GIFs per cell.
# Visualization is on by default; callers may disable only artifact generation
# without changing observations, planning, traffic, control, or scoring.
ENABLE_VIS=(--enable-vis)
case "${NAVSAFE_ENABLE_VIS:-1}" in
  0|false|False|no|off) ENABLE_VIS=() ;;
esac
# NAVSAFE_LOG_LEVEL=INFO surfaces navsafe's INFO lines (traffic takeovers,
# scenario setup) in eval.log; the default root level is WARNING, which drops
# them entirely.
LOG_LEVEL=()
if [ -n "${NAVSAFE_LOG_LEVEL:-}" ]; then
  LOG_LEVEL=(--log-level "$NAVSAFE_LOG_LEVEL")
fi
# NAVSAFE_TAKEOVER=spawn reverts semi_reactive to MetaDrive's spawn-time-only
# takeover test, for reproducing cells recorded before 2026-08-11.
TAKEOVER=()
if [ -n "${NAVSAFE_TAKEOVER:-}" ]; then
  TAKEOVER=(--traffic-takeover "$NAVSAFE_TAKEOVER")
fi
# The output directory holds the NavSafe result and the render, nothing else:
# navsafe_metrics.json, vehicle_states.npy, trajectory.npy, frames/,
# visualization/. metrics.json (EPDMS), driving_score_summary.csv and
# run_meta.json are pruned once navsafe_metrics.json exists, and this log lives
# BESIDE the directory rather than in it. NAVSAFE_KEEP_ARTIFACTS=1 keeps all of
# them, which is what re-scoring with per-frame EPDMS flags needs.
PRUNE=(--navsafe-prune-artifacts)
[ "${NAVSAFE_KEEP_ARTIFACTS:-0}" = "1" ] && PRUNE=()
# Sibling, not child — and named after the run, so parallel tokens do not share
# it. A sweep's completion test greps this file for the DONE line.
LOG="${NAVSAFE_EVAL_LOG:-$OUT.log}"
mkdir -p "$OUT" "$(dirname "$LOG")"

# A scene-editing recipe's `nurec_asset_id` is a filesystem path that the
# SERVER open()s (nurec_grpc._insert_injected_assets -> edit_assets), so the
# asset library has to be visible inside this container at the identical path.
# NAVSAFE_ASSET_MOUNT is a colon-separated list of host directories to bind
# read-only. Without it an inserted asset fails the insert outright — loudly,
# but a long way from the missing mount that caused it.
ASSET_MOUNTS=()
if [ -n "${NAVSAFE_ASSET_MOUNT:-}" ]; then
  IFS=':' read -r -a _asset_dirs <<< "$NAVSAFE_ASSET_MOUNT"
  for d in "${_asset_dirs[@]}"; do
    [ -d "$d" ] || { echo "NAVSAFE_ASSET_MOUNT: no directory $d" >&2; exit 2; }
    ASSET_MOUNTS+=(-v "$d":"$d":ro)
  done
fi
# Same requirement, one directory the wrapper can find by itself: a harvested
# asset bank (--asset-harvester-replace) lives inside the bundle, and the
# bundle is mounted at /workdir/bundle rather than at its own path — so the
# manifest's absolute PLY paths would not resolve in the container even though
# the files are right there. Bind it at its host path as well. Costs nothing
# when no bank exists, and removes the failure where a replace run renders the
# baked actors it was run to avoid.
if [ -d "$B/ah_assets" ]; then
  ASSET_MOUNTS+=(-v "$B/ah_assets":"$B/ah_assets":ro)
fi

# --- serve ---------------------------------------------------------------
[ -n "$NGC_ENV" ] && { set -a; . "$NGC_ENV"; set +a; }
: "${NGC_API_KEY:?export NGC_API_KEY (or point NGC_ENV_FILE at a file that does)}"
# The harmonizer weights are fetched once and cached; without a cache directory
# every run re-downloads them, so it is opt-in rather than silently slow.
HARMONIZER=()
if [ -n "${NAVSAFE_ROOT_HARMONIZER_CACHE:-}" ]; then
  mkdir -p "$NAVSAFE_ROOT_HARMONIZER_CACHE"
  HARMONIZER=(-v "$NAVSAFE_ROOT_HARMONIZER_CACHE":/harmonizer-cache)
  HARMONIZER_ARGS=(--enable-harmonizer --harmonizer-cache /harmonizer-cache)
else
  HARMONIZER_ARGS=()
fi
# NRE unpacks runtime extensions below /tmp.  On shared hosts Docker's writable
# layer can be full even when the host tmpfs is not, so allow a caller-owned
# directory to be bind-mounted for this container only.  This is deliberately
# opt-in: it neither prunes shared Docker state nor changes other campaigns.
CONTAINER_TMP=()
if [ -n "${NAVSAFE_CONTAINER_TMP:-}" ]; then
  mkdir -p "$NAVSAFE_CONTAINER_TMP"
  [ -d "$NAVSAFE_CONTAINER_TMP" ] \
    || { echo "NAVSAFE_CONTAINER_TMP: not a directory: $NAVSAFE_CONTAINER_TMP" >&2; exit 2; }
  CONTAINER_TMP=(-v "$NAVSAFE_CONTAINER_TMP":/tmp)
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$GPU\"" --shm-size 16g \
  -u "$(id -u):$(id -g)" -p "$PORT":8080 -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$B":/workdir/bundle ${ASSET_MOUNTS[@]+"${ASSET_MOUNTS[@]}"} \
  ${CONTAINER_TMP[@]+"${CONTAINER_TMP[@]}"} \
  ${HARMONIZER[@]+"${HARMONIZER[@]}"} \
  "$IMAGE" serve-grpc --host 0.0.0.0 --enable-editing-actors --renderer default \
  --artifact-glob '/workdir/bundle/*.usdz' \
  ${HARMONIZER_ARGS[@]+"${HARMONIZER_ARGS[@]}"} >/dev/null
# Keep the renderer's own log before removing it. A render that dies mid-run
# reaches the eval only as `nurec_grpc render failed for CAM_F0` — the reason
# ("CUDA being in bad state: NVIDIA L40S", a CUDA OOM, an evicted scene) is
# server-side, and `docker rm -f` used to take it with the container. Cells
# then had to be diagnosed by inference. Written beside the eval log, and only
# the tail: a healthy run's log is large and says nothing new.
_keep_serve_log() {
  local dest="${NAVSAFE_SERVE_LOG:-$OUT.serve-container.log}"
  docker logs "$NAME" > "$dest" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap _keep_serve_log EXIT

echo "[serve] waiting for $NAME on :$PORT"
for _ in $(seq 1 120); do
  docker logs "$NAME" 2>&1 | grep -q "Available scenes" && break
  sleep 5
done
SCENES=$(docker logs "$NAME" 2>&1 | grep "Available scenes" | tail -1)
[ -n "$SCENES" ] || { echo "[serve] never published a scene list" >&2; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
echo "[serve] $SCENES"
for S in 1 2 3 4; do
  echo "$SCENES" | grep -q "${TOKEN}s${S}" \
    || { echo "[serve] MISSING scene ${TOKEN}s${S} — would render as raster" >&2; exit 1; }
done

# The scene list is NOT readiness: serve-grpc prints it while still loading the
# harmonizer weights, and only binds the port ~2 min later ("Serving on
# 0.0.0.0:8080"). Connecting in that gap dies with UNAVAILABLE / "Connection
# reset by peer" from deep inside renderer setup, which reads as a broken
# bundle. Wait for the bind line, then for the socket to actually accept.
for _ in $(seq 1 120); do
  docker logs "$NAME" 2>&1 | grep -q "Serving on" && break
  sleep 5
done
docker logs "$NAME" 2>&1 | grep -q "Serving on" \
  || { echo "[serve] never bound its port" >&2; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
for _ in $(seq 1 60); do
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && { exec 3<&- 3>&-; break; }
  sleep 2
done
echo "[serve] accepting connections on :$PORT"

# --- eval ----------------------------------------------------------------
# Derived from the manifest against THIS bundle's path, never read from a file:
# the renderer open()s the offsets JSON directly, so a handoff baked at build
# time points at the builder's machine and dies in renderer init everywhere
# else. Stdlib-only python3 on purpose — importing navsafe here would bootstrap
# Omniverse and write its EULA prompt into the captured value.
export NUREC_GRPC_HANDOFF="$(python3 - "$B" <<'PY'
import json, sys
b = sys.argv[1]
m = json.load(open(b + "/manifest.json"))
print(";".join("%s,%s/offsets/%s.json,%d,%d"
               % (c["scene_id"], b, c["scene_id"],
                  c["t_start_us"], c["t_stop_us"])
               for c in m["subclips"]))
PY
)"
export NUREC_GRPC_HOST=127.0.0.1 NUREC_GRPC_PORT=$PORT
# recon, not navsim: the reconstruction's own training camera. The
# navsim rig rebuilds a zero-distortion pinhole from seven scalars and
# renders rays the recon was never fit on; nurec_grpc.py calls it a
# legacy compatibility path and not the quality default. Overridable
# (NUREC_GRPC_CAM_RIG=navsim) so the A/B stays available.
export NUREC_GRPC_CAM_RIG=${NUREC_GRPC_CAM_RIG:-recon} NUREC_GRPC_TIMEOUT_S=600
export PY123D_RECENTER=1
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
# The bundle's own arrow/ carries the map, so NUPLAN_MAPS_ROOT and ISAACLAB_PATH
# are only exported when the caller has them — a default would point at the
# machine this script was written on and fail everywhere else.
export PYTHONPATH="$REPO${ISAACLAB_PATH:+:$ISAACLAB_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES UV_NO_SYNC=1
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export CUDA_VISIBLE_DEVICES=$GPU NEXUSSIM_NUREC_GPUS=$GPU

echo "[eval] $MODE  model=$MODEL  -> $OUT  (log $LOG)"
"$PY" "$REPO/scripts/tools/eval_py123d.py" \
  --scenario-source py123d --py123d-data-root "$B/arrow" --py123d-scene-index 0 \
  --render-backend nurec_grpc \
  --model-type "$MODEL" --checkpoint "$CKPT" \
  "${SHAPE[@]}" --replan-rate "${NAVSAFE_REPLAN_RATE:-5}" \
  --camera-resolution-scale 1.0 ${ENABLE_VIS[@]+"${ENABLE_VIS[@]}"} \
  "${VIS_CAMS[@]}" "${LOG_LEVEL[@]}" "${TAKEOVER[@]}" ${PRUNE[@]+"${PRUNE[@]}"} \
  --output-dir "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG" 2>&1
echo "[eval] done -> $OUT ($(find "$OUT" -name cam_f0.jpg | wc -l) frames)"
