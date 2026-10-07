#!/usr/bin/env bash
# Evaluate one scenario bundle: closed-loop policy evaluation, or a full
# ego-replay render that shows reconstruction quality without a policy.
#
#   run_bundle_eval.sh TOKEN [MODE] [MODEL] [CHECKPOINT] [extra evaluator args...]
#     MODE   policy (default) | replay
#
# Arguments after CHECKPOINT are appended to the evaluator command line, where
# a later flag overrides an earlier one.
#
# The renderer keeps four reconstructions resident (about 18 GiB, more with
# inserted and harvested assets), so on 24 GB GPUs it gets one to itself and the
# simulator and policy use another. One larger GPU can serve both.
#
#   NAVSAFE_DATA_ROOT    dataset root; the bundle is full_test/<TOKEN> below it
#   NAVSAFE_GPU          evaluator GPU (default 0)
#   NAVSAFE_RENDER_GPU   renderer GPU (default NAVSAFE_GPU)
#   NAVSAFE_RENDERER     docker (default): start serve-grpc for this bundle and
#                        stop it afterwards.
#                        existing: use the serve-grpc already running at
#                        NUREC_GRPC_HOST:NUREC_GRPC_PORT that serves this bundle.
#   NAVSAFE_OUT          output directory (default <repo>/output/bundle_<TOKEN>)
#
# See docs/navsafe_eval.md for the remaining variables.
set -euo pipefail

TOKEN="${1:?token}"
MODE="${2:-policy}"
MODEL="${3:-drivor}"
CKPT="${4:-${NAVSAFE_CHECKPOINT:?set NAVSAFE_CHECKPOINT (or pass the checkpoint as arg 4)}}"
shift $(( $# < 4 ? $# : 4 ))
EXTRA=("$@")

REPO="${NAVSAFE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${NAVSAFE_PYTHON:-python}"
BUNDLES="${NAVSAFE_BUNDLES:-${NAVSAFE_DATA_ROOT:+$NAVSAFE_DATA_ROOT/full_test}}"
[ -n "$BUNDLES" ] || { echo "set NAVSAFE_DATA_ROOT (or NAVSAFE_BUNDLES)" >&2; exit 2; }
B="$BUNDLES/$TOKEN"
PORT="${NUREC_PORT:-8088}"
GPU="${NAVSAFE_GPU:-0}"
RENDER_GPU="${NAVSAFE_RENDER_GPU:-$GPU}"
RENDERER="${NAVSAFE_RENDERER:-docker}"
IMAGE="${NRE_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"
NGC_ENV="${NGC_ENV_FILE:-}"
NAME="nre_bundle_$PORT"

[ -f "$B/manifest.json" ] || { echo "no bundle at $B" >&2; exit 2; }
[ -d "$B/arrow" ] || { echo "$B has no arrow/" >&2; exit 2; }

case "$MODE" in
  # Any contact ends the episode; fault attribution only decides whether the
  # ending is charged to the policy. The ego is driven through a tracker and a
  # bicycle model, so a plan the vehicle cannot follow shows up as tracking
  # error. Background traffic is semi-reactive: it leaves the log to respond to
  # the ego.
  policy) OUT="$REPO/output/bundle_$TOKEN"
          EVAL_WINDOW=()
          # Unset means no frame cap: the episode runs to a terminal condition.
          if [ -n "${NAVSAFE_EVAL_FRAMES:-}" ]; then
            EVAL_WINDOW=(--eval-frames "$NAVSAFE_EVAL_FRAMES")
          fi
          SHAPE=(--traffic-mode "${NAVSAFE_TRAFFIC:-semi_reactive}"
                 ${EVAL_WINDOW[@]+"${EVAL_WINDOW[@]}"}
                 --route-time-limit-s "${NAVSAFE_ROUTE_TIME_LIMIT_S:-0}"
                 --ego-replay-frames 8
                 --terminate-on-collision
                 --controller "${NAVSAFE_CONTROLLER:-pure_pursuit}"
                 --execution-mode "${NAVSAFE_EXECUTION_MODE:-controller}") ;;
  # A replay follows the logged ego, so traffic follows the log as well.
  replay) OUT="$REPO/output/bundle_${TOKEN}_replay"
          SHAPE=(--traffic-mode "${NAVSAFE_TRAFFIC:-log_replay}"
                 --controller pure_pursuit
                 --ego-replay-frames 200 --eval-frames 0) ;;
  *) echo "MODE must be policy or replay" >&2; exit 2 ;;
esac
# The default is per token, not per model: set NAVSAFE_OUT to keep two
# policies' results on one scenario apart.
OUT="${NAVSAFE_OUT:-$OUT}"

# The front camera keeps its HUD and plan overlay but not the projected map
# lines. NAVSAFE_NO_OVERLAY=1 removes every camera annotation.
export NAVSAFE_NO_CAM_MAP_LINES="${NAVSAFE_NO_CAM_MAP_LINES:-1}"
export NAVSAFE_NO_OVERLAY="${NAVSAFE_NO_OVERLAY:-0}"
# Extra cameras are rendered for the artifacts only and never reach the policy.
VIS_CAMS=()
if [ -n "${NAVSAFE_VIS_CAMS:-}" ]; then
  VIS_CAMS=(--vis-cameras "$NAVSAFE_VIS_CAMS")
fi
# Visualization renders every simulation step and writes images and GIFs.
ENABLE_VIS=(--enable-vis)
case "${NAVSAFE_ENABLE_VIS:-1}" in
  0|false|False|no|off) ENABLE_VIS=() ;;
esac
LOG_LEVEL=()
if [ -n "${NAVSAFE_LOG_LEVEL:-}" ]; then
  LOG_LEVEL=(--log-level "$NAVSAFE_LOG_LEVEL")
fi
# NAVSAFE_TAKEOVER=spawn tests traffic takeover once per vehicle, at spawn.
TAKEOVER=()
if [ -n "${NAVSAFE_TAKEOVER:-}" ]; then
  TAKEOVER=(--traffic-takeover "$NAVSAFE_TAKEOVER")
fi
# The other metric families' artifacts are removed once navsafe_metrics.json
# exists. NAVSAFE_KEEP_ARTIFACTS=1 keeps them, which later re-scoring needs.
PRUNE=(--navsafe-prune-artifacts)
[ "${NAVSAFE_KEEP_ARTIFACTS:-0}" = "1" ] && PRUNE=()
# The log sits beside the output directory, named after the run.
LOG="${NAVSAFE_EVAL_LOG:-$OUT.log}"
mkdir -p "$OUT" "$(dirname "$LOG")"

# --- renderer ------------------------------------------------------------
if [ "$RENDERER" = existing ]; then
  RHOST="${NUREC_GRPC_HOST:?NAVSAFE_RENDERER=existing needs NUREC_GRPC_HOST}"
  RPORT="${NUREC_GRPC_PORT:-8080}"
  # A renderer that has just started accepts connections some minutes after
  # it lists its scenes.
  echo "[serve] existing renderer at $RHOST:$RPORT"
  for _ in $(seq 1 "${NAVSAFE_RENDERER_WAIT_TRIES:-360}"); do
    (exec 3<>"/dev/tcp/$RHOST/$RPORT") 2>/dev/null && { exec 3<&- 3>&-; READY=1; break; }
    sleep 5
  done
  [ "${READY:-0}" = 1 ] || { echo "[serve] $RHOST:$RPORT never accepted a connection" >&2; exit 1; }
else
  [ -n "$NGC_ENV" ] && { set -a; . "$NGC_ENV"; set +a; }
  : "${NGC_API_KEY:?export NGC_API_KEY (or point NGC_ENV_FILE at a file that does)}"

  # The renderer opens inserted assets by the absolute paths the recipe
  # resolves to, so each directory in NAVSAFE_ASSET_MOUNT (colon-separated) is
  # bound read-only at its own path. The bundle's harvested-asset bank is bound
  # the same way.
  ASSET_MOUNTS=()
  if [ -n "${NAVSAFE_ASSET_MOUNT:-}" ]; then
    IFS=':' read -r -a _asset_dirs <<< "$NAVSAFE_ASSET_MOUNT"
    for d in "${_asset_dirs[@]}"; do
      [ -d "$d" ] || { echo "NAVSAFE_ASSET_MOUNT: no directory $d" >&2; exit 2; }
      ASSET_MOUNTS+=(-v "$d":"$d":ro)
    done
  fi
  if [ -d "$B/ah_assets" ]; then
    ASSET_MOUNTS+=(-v "$B/ah_assets":"$B/ah_assets":ro)
  fi
  # The harmonizer is enabled when given a directory to cache its weights in.
  HARMONIZER=(); HARMONIZER_ARGS=()
  if [ -n "${NAVSAFE_HARMONIZER_CACHE:-}" ]; then
    mkdir -p "$NAVSAFE_HARMONIZER_CACHE"
    HARMONIZER=(-v "$NAVSAFE_HARMONIZER_CACHE":/harmonizer-cache)
    HARMONIZER_ARGS=(--enable-harmonizer --harmonizer-cache /harmonizer-cache)
  fi
  # NAVSAFE_CONTAINER_TMP gives the container its own /tmp when Docker's
  # writable layer is short of space.
  CONTAINER_TMP=()
  if [ -n "${NAVSAFE_CONTAINER_TMP:-}" ]; then
    mkdir -p "$NAVSAFE_CONTAINER_TMP"
    CONTAINER_TMP=(-v "$NAVSAFE_CONTAINER_TMP":/tmp)
  fi

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --gpus "\"device=$RENDER_GPU\"" --shm-size 16g \
    -u "$(id -u):$(id -g)" -p "$PORT":8080 -e NGC_API_KEY="$NGC_API_KEY" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$B":/workdir/bundle ${ASSET_MOUNTS[@]+"${ASSET_MOUNTS[@]}"} \
    ${CONTAINER_TMP[@]+"${CONTAINER_TMP[@]}"} \
    ${HARMONIZER[@]+"${HARMONIZER[@]}"} \
    "$IMAGE" serve-grpc --host 0.0.0.0 --enable-editing-actors --renderer default \
    --artifact-glob '/workdir/bundle/*.usdz' \
    ${HARMONIZER_ARGS[@]+"${HARMONIZER_ARGS[@]}"} >/dev/null
  # The reason a render failed is in the renderer's log, so keep it.
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
  # A scene the renderer does not hold would be rendered without the
  # reconstruction, which looks like poor quality rather than a missing file.
  for S in 1 2 3 4; do
    echo "$SCENES" | grep -q "${TOKEN}s${S}" \
      || { echo "[serve] missing scene ${TOKEN}s${S}" >&2; exit 1; }
  done
  # The port is bound some minutes after the scene list is printed.
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
  RHOST=127.0.0.1 RPORT=$PORT
fi

# --- eval ----------------------------------------------------------------
# The handoff names each window's offsets file by absolute path, so it is
# built from the manifest against this bundle's location.
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
export NUREC_GRPC_HOST=$RHOST NUREC_GRPC_PORT=$RPORT
# recon renders with the reconstruction's own calibrated cameras.
export NUREC_GRPC_CAM_RIG=${NUREC_GRPC_CAM_RIG:-recon} NUREC_GRPC_TIMEOUT_S="${NUREC_GRPC_TIMEOUT_S:-600}"
export PY123D_RECENTER=1
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$REPO${ISAACLAB_PATH:+:$ISAACLAB_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES UV_NO_SYNC=1
_LIBSTDCXX=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
[ -n "${LD_PRELOAD:-}" ] || [ ! -e "$_LIBSTDCXX" ] || export LD_PRELOAD="$_LIBSTDCXX"
export CUDA_VISIBLE_DEVICES=$GPU NAVSAFE_NUREC_GPUS=$GPU

echo "[eval] $MODE  model=$MODEL  -> $OUT  (log $LOG)"
"$PY" -m navsafe.cli.eval_entry \
  --scenario-source py123d --py123d-data-root "$B/arrow" --py123d-scene-index 0 \
  --render-backend nurec_grpc \
  --model-type "$MODEL" --checkpoint "$CKPT" \
  "${SHAPE[@]}" --replan-rate "${NAVSAFE_REPLAN_RATE:-5}" \
  --camera-resolution-scale 1.0 ${ENABLE_VIS[@]+"${ENABLE_VIS[@]}"} \
  ${VIS_CAMS[@]+"${VIS_CAMS[@]}"} ${LOG_LEVEL[@]+"${LOG_LEVEL[@]}"} \
  ${TAKEOVER[@]+"${TAKEOVER[@]}"} ${PRUNE[@]+"${PRUNE[@]}"} \
  --output-dir "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG" 2>&1
echo "[eval] done -> $OUT ($(find "$OUT" -name cam_f0.jpg | wc -l) frames)"
