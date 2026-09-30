#!/usr/bin/env bash
# Render N published full_test/ bundles as 200-frame ego log-replays, 4
# DrivoR cameras each (front/back/left/right), and write one combined.gif
# per scenario. One serve-grpc for the whole batch: the ~20 min bootstrap
# (venv/deps + harmonizer weights + first scene load) is a fixed cost per
# process, not per scenario, so batching amortizes it across TOKENS instead
# of paying it once per scenario -- same reasoning as k8s_jobs.py:build_eval,
# whose serve-then-eval-then-kill shape this follows. Runs INSIDE the nre-ga
# container itself (no docker-in-docker): /app/run serve-grpc is backgrounded
# as a plain process, same as build_eval's script does for one seed.
#
#   render_gif_batch.sh TOKEN [TOKEN...]
#
# Env:
#   NAVSAFE_CHECKPOINT   required -- eval_py123d.py always loads a policy
#                        model even in pure replay mode (--eval-frames 0
#                        just means it's never called to act)
#   NAVSAFE_GIF_OUT      where combined.gif per token lands; default
#                        $HOME/.cache/navsafe/gifs
#   NAVSAFE_WORK         scratch root for the venv, downloaded bundles, and
#                        per-token render output; default $HOME/.cache/navsafe/gifwork
#   NAVSAFE_GPU          GPU device index; default 0
#   NAVSAFE_CAM_LATERAL_M  comma list of camera lateral offsets in metres
#                        (+left), default "0". Each token is rendered once
#                        per offset against the SAME loaded scene -- the ego
#                        still drives its logged path, only the camera rig
#                        slides sideways, so the set of gifs is a novel-view
#                        drift sweep of the reconstruction, not N episodes.
#                        A nonzero offset lands in combined_lat<v>m.gif; 0
#                        keeps the plain combined.gif.
#   NEXUSSIM_HARMONIZER_CACHE  persistent harmonizer checkpoint cache;
#                        default $HOME/.cache/navsafe/harmonizer
#   UV_CACHE_DIR         persistent uv wheel cache; default
#                        $HOME/.cache/uv
set -eo pipefail

[ $# -ge 1 ] || { echo "usage: render_gif_batch.sh TOKEN [TOKEN...]" >&2; exit 2; }
TOKENS=("$@")

NAVSAFE_ROOT="${NAVSAFE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
WORK="${NAVSAFE_WORK:-$HOME/.cache/navsafe/gifwork}"
DATA_ROOT="${NAVSAFE_DATA_ROOT:-$HOME/data/NavSafe}"
BUNDLES="$DATA_ROOT/full_test"
OUT_ROOT="$WORK/eval"
GIF_OUT="${NAVSAFE_GIF_OUT:-$HOME/.cache/navsafe/gifs}"
GPU="${NAVSAFE_GPU:-0}"
CKPT="${NAVSAFE_CHECKPOINT:?set NAVSAFE_CHECKPOINT}"
DRIVOR_EXTRA_CAMS="CAM_B0,CAM_L0,CAM_R0"   # CAM_F0 is the policy's own input, always rendered
IFS=',' read -r -a LATS <<< "${NAVSAFE_CAM_LATERAL_M:-0}"

mkdir -p "$BUNDLES" "$OUT_ROOT" "$GIF_OUT"
trap 'echo "=== serve-grpc tail ==="; tail -80 /tmp/serve.log 2>/dev/null || true; \
      [ -n "${SERVE_PID:-}" ] && kill "$SERVE_PID" 2>/dev/null || true' EXIT

# --- 0. skip tokens that already have a combined.gif, so a resubmitted job
# after a partial failure doesn't redo the whole batch. ---------------------
# A token counts as done only when EVERY requested lateral offset is on disk,
# so adding an offset to the sweep re-renders that offset instead of reading
# the un-drifted gif as proof the token is finished.
gif_name() {  # $1 = offset -> the file that offset writes
  case "$1" in
    *[!0.\ -]*) echo "combined_lat${1}m.gif" ;;
    *)           echo "combined.gif" ;;
  esac
}
TODO=()
for T in "${TOKENS[@]}"; do
  MISSING_LAT=()
  for LAT in "${LATS[@]}"; do
    [ -f "$GIF_OUT/$T/$(gif_name "$LAT")" ] || MISSING_LAT+=("$LAT")
  done
  if [ ${#MISSING_LAT[@]} -eq 0 ]; then
    echo "[skip] $T: all ${#LATS[@]} render(s) already in $GIF_OUT/$T"
  else
    TODO+=("$T")
  fi
done
[ ${#TODO[@]} -gt 0 ] || { echo "[render] nothing to do, all ${#TOKENS[@]} token(s) already rendered"; exit 0; }
echo "[render] ${#TODO[@]}/${#TOKENS[@]} token(s) to render: ${TODO[*]}"

# --- 1. Use the environment installed by the public installation guide.
PY="${NAVSAFE_PYTHON:-python}"
"$PY" -c "import isaacsim, isaaclab, navsafe, huggingface_hub"
cd "$NAVSAFE_ROOT"

# --- 2. Fetch missing bundles without deleting the user's snapshot.
PYTHONPATH="$NAVSAFE_ROOT" "$PY" -m navsafe.benchmark.eval.fetch_bundle --out "$DATA_ROOT" \
  $(printf -- '--token %s ' "${TODO[@]}")

for T in "${TODO[@]}"; do
  [ -f "$BUNDLES/$T/manifest.json" ] || { echo "fetch produced no manifest for $T" >&2; exit 2; }
  [ -d "$BUNDLES/$T/arrow" ] || { echo "fetch produced no arrow/ for $T" >&2; exit 2; }
done

# --- 3. serve the whole batch's usdz from one process, in-container --------
HARMONIZER_CACHE="${NAVSAFE_ROOT_HARMONIZER_CACHE:-$HOME/.cache/navsafe/harmonizer}"
mkdir -p "$HARMONIZER_CACHE"

CUDA_VISIBLE_DEVICES=$GPU /app/run serve-grpc --host 0.0.0.0 \
  --enable-editing-actors --renderer default \
  --enable-harmonizer --harmonizer-cache "$HARMONIZER_CACHE" \
  --cache-size 4 \
  --artifact-glob "$BUNDLES/*/*.usdz" > /tmp/serve.log 2>&1 &
SERVE_PID=$!

echo "[serve] waiting on pid $SERVE_PID (batch of ${#TODO[@]} -- first load is the ~20 min cost)"
for i in $(seq 1 180); do
  kill -0 $SERVE_PID 2>/dev/null || { echo "serve-grpc exited early" >&2; exit 1; }
  grep -q "Available scenes" /tmp/serve.log 2>/dev/null && break
  sleep 10
done
SCENES=$(grep "Available scenes" /tmp/serve.log 2>/dev/null | tail -1)
[ -n "$SCENES" ] || { echo "[serve] never published a scene list" >&2; tail -60 /tmp/serve.log; exit 1; }
echo "[serve] $SCENES"
MISSING=()
for T in "${TODO[@]}"; do
  for S in 1 2 3 4; do
    echo "$SCENES" | grep -q "${T}s${S}" || MISSING+=("${T}s${S}")
  done
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "[serve] MISSING scene(s), would render as raster: ${MISSING[*]}" >&2
  exit 1
fi

for i in $(seq 1 120); do
  kill -0 $SERVE_PID 2>/dev/null || { echo "serve-grpc exited early" >&2; exit 1; }
  (exec 3<>/dev/tcp/localhost/8080) 2>/dev/null && { exec 3<&- 3>&-; echo "serve up after ${i}0s"; break; }
  sleep 10
done
echo "[serve] accepting connections -- rendering ${#TODO[@]} token(s) against it"

# --- 4. one eval_py123d.py ego-replay per token x lateral offset, same
# server throughout ------------------------------------------------------
export NUREC_GRPC_HOST=localhost NUREC_GRPC_PORT=8080
# recon, not navsim: the reconstruction's own training camera. The
# navsim rig rebuilds a zero-distortion pinhole from seven scalars and
# renders rays the recon was never fit on; nurec_grpc.py calls it a
# legacy compatibility path and not the quality default. Overridable
# (NUREC_GRPC_CAM_RIG=navsim) so the A/B stays available.
export NUREC_GRPC_CAM_RIG=${NUREC_GRPC_CAM_RIG:-recon} NUREC_GRPC_TIMEOUT_S=600
export PY123D_RECENTER=1
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export PYTHONPATH="$NAVSAFE_ROOT${ISAACLAB_PATH:+:$ISAACLAB_PATH}"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES UV_NO_SYNC=1
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export CUDA_VISIBLE_DEVICES=$GPU NEXUSSIM_NUREC_GPUS=$GPU
# Unannotated cameras (video, not the HUD/plan-ribbon debug view); topdown
# keeps its own annotation either way, per run_bundle_eval.sh's convention.
export NEXUSSIM_NO_OVERLAY="${NAVSAFE_ROOT_NO_OVERLAY:-1}"
export NEXUSSIM_NO_CAM_MAP_LINES="${NAVSAFE_ROOT_NO_CAM_MAP_LINES:-1}"

FAILED=()
for T in "${TODO[@]}"; do
  B="$BUNDLES/$T"

  # Derived per-token, against THIS token's own offsets/ -- bundle.py --handoff
  # is manifest-only (stdlib, no navsafe import needed at the call site, per
  # docs/navsafe_eval.md §4), so this stays a plain subprocess call rather
  # than re-deriving the format inline.
  export NUREC_GRPC_HANDOFF="$(python3 "$NAVSAFE_ROOT/navsafe/benchmark/eval/bundle.py" --handoff "$B")"

  # Every lateral offset reuses the scenes this token already loaded into
  # serve-grpc, so a drift sweep costs one render pass each, not one server.
  for LAT in "${LATS[@]}"; do
    # "0", "0.0", "-0" are all the undrifted render, and it keeps the
    # name every existing caller already looks for.
    NAME="$(gif_name "$LAT")"
    case "$LAT" in
      *[!0.\ -]*) TAG="$T@${LAT}m"; OUT="$OUT_ROOT/$T/lat$LAT" ;;
      *)           TAG="$T";         OUT="$OUT_ROOT/$T" ;;
    esac
    mkdir -p "$OUT"
    export NUREC_GRPC_CAM_LATERAL_M="$LAT"
    echo "[eval] $TAG -> $OUT"

    if "$PY" "$NAVSAFE_ROOT/scripts/tools/eval_py123d.py" \
        --scenario-source py123d --py123d-data-root "$B/arrow" --py123d-scene-index 0 \
        --render-backend nurec_grpc \
        --model-type drivor --checkpoint "$CKPT" \
        --traffic-mode log_replay --controller pure_pursuit \
        --ego-replay-frames 200 --eval-frames 0 \
        --camera-resolution-scale 1.0 --enable-vis \
        --vis-cameras "$DRIVOR_EXTRA_CAMS" \
        --output-dir "$OUT" > "$OUT/eval.log" 2>&1
    then
      GIF=$(find "$OUT" -name combined.gif | head -1)
      if [ -z "$GIF" ]; then
        echo "[eval] $TAG: eval_py123d.py exited 0 but wrote no combined.gif" >&2
        FAILED+=("$TAG")
        continue
      fi
      mkdir -p "$GIF_OUT/$T"
      cp "$GIF" "$GIF_OUT/$T/$NAME"
      echo "[eval] $TAG: OK -> $GIF_OUT/$T/$NAME"
    else
      echo "[eval] $TAG: eval_py123d.py failed, see $OUT/eval.log" >&2
      tail -30 "$OUT/eval.log" >&2 || true
      FAILED+=("$TAG")
    fi
  done

  # This token is done either way -- its 4 scenes are about to be evicted
  # from serve-grpc's --cache-size 4 by the NEXT token's 4 anyway (never
  # revisited within a batch), so the usdz on disk has no further use.
  # Defense-in-depth against a bundle pile-up eviction: the real fix is
  # sizing ephemeral-storage for one batch (gif_batch_jobs.py), but this
  # keeps a batch's disk footprint from ever exceeding roughly one token's
  # worth beyond that even if a batch is ever run oversized by mistake.
  # Keep downloaded bundles and episode artifacts for reuse and inspection.
done

N_RENDERS=$(( ${#TODO[@]} * ${#LATS[@]} ))
echo "[render] batch done: $((N_RENDERS - ${#FAILED[@]}))/$N_RENDERS succeeded"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[render] failed: ${FAILED[*]}" >&2
  exit 1
fi
