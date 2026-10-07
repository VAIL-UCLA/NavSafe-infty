#!/usr/bin/env bash
# Evaluate one policy on every downloaded NavSafe bundle -- the whole published
# full_test set (280 scenarios), edited and unedited alike -- in one command.
#
#   run_full_benchmark.sh MODEL CHECKPOINT [extra the evaluator args...]
#
# A loop over run_bundle_eval.sh, and nothing else, so a cell here follows
# exactly the single-bundle command in docs/navsafe_eval.md. What this adds is
# the enumeration, the edit decision, the benchmark protocol, retries,
# resumability and the final table.
#
# On 24 GB GPUs use two: one for the renderer, one for the simulator and policy.
# A GPU with enough VRAM for both can be given to both.
#
#   NAVSAFE_DATA_ROOT   required: the HF snapshot root (full_test/, asset/, gait_bank/)
#   NAVSAFE_OUT_ROOT    run directory (default <repo>/output/full_benchmark_<MODEL>)
#   NAVSAFE_RENDERER    docker (default): a renderer container per scenario on
#                       NAVSAFE_RENDER_GPU (default 0), evaluator on NAVSAFE_GPU
#                       (default 1).
#                       existing: a serve-grpc already running at
#                       NUREC_GRPC_HOST:NUREC_GRPC_PORT that serves every bundle
#                       (--artifact-glob '<data-root>/full_test/*/*.usdz'
#                       --cache-size 4 --enable-editing-actors). No Docker
#                       needed; this is the Kubernetes path.
#   NAVSAFE_SHARD       i/n: run only every n-th scenario starting at i (0-based),
#                       for n parallel lanes, each with its own renderer and
#                       GPU pair, sharing one NAVSAFE_OUT_ROOT
#   NAVSAFE_EDITS       auto (default) | off
#   NAVSAFE_TOKENS      file with one token per line, to run a subset
#   NAVSAFE_EVAL_SEED   policy seed (default 0)
#   NAVSAFE_RETRIES     attempts per scenario until it is scored (default 2)
#   NAVSAFE_ENABLE_VIS  1 writes images/GIFs per scenario (default 0 here)
#   NAVSAFE_FORCE       1 re-runs scenarios that are already scored
#   NAVSAFE_DRY_RUN     1 prints the plan and exits; needs no GPU and no Docker
#
# Everything run_bundle_eval.sh reads (NGC_ENV_FILE, NRE_IMAGE, NAVSAFE_REPLAN_RATE,
# NAVSAFE_KEEP_ARTIFACTS, ...) passes through untouched.
#
# The edit decision is the evaluator's, not this script's. NAVSAFE_EDITS=auto
# forwards --recipe-dir, and the evaluator resolves each bundle against the
# frozen recipes in navsafe/benchmark/recipes/benchmark: a scenario whose recipe
# inserts actors runs edited under navsafe traffic, every other scenario runs
# its host unedited. NAVSAFE_EDITS=off runs all of them unedited, which is the
# no-edit baseline rather than the benchmark.
#
# One scenario at a time per renderer, never two: closing an episode restores
# the renderer's scene, which would remove another episode's inserted assets.
set -euo pipefail

MODEL="${1:?model type, e.g. drivor}"
CKPT="${2:?checkpoint path (or the adapter sentinel, e.g. none for pdm_closed)}"
shift 2
EXTRA=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${NAVSAFE_REPO:-$(cd "$HERE/../../.." && pwd)}"
PY="${NAVSAFE_PYTHON:-python}"
BUNDLES="${NAVSAFE_BUNDLES:-${NAVSAFE_DATA_ROOT:+$NAVSAFE_DATA_ROOT/full_test}}"
[ -n "$BUNDLES" ] || { echo "set NAVSAFE_DATA_ROOT (or NAVSAFE_BUNDLES)" >&2; exit 2; }
[ -d "$BUNDLES" ] || { echo "no bundle directory at $BUNDLES" >&2; exit 2; }
RECIPES="${NAVSAFE_RECIPES:-$REPO/navsafe/benchmark/recipes/benchmark}"
OUT_ROOT="${NAVSAFE_OUT_ROOT:-$REPO/output/full_benchmark_$MODEL}"
EDITS="${NAVSAFE_EDITS:-auto}"
# Renderer start-up is ~3 min and a full episode ~10; this only has to catch a
# wedged cell so that it costs one scenario instead of the rest of the sweep.
EPISODE_TIMEOUT="${NAVSAFE_EPISODE_TIMEOUT:-45m}"

case "$EDITS" in
  auto) EDIT_ARGS=(--recipe-dir "$RECIPES")
        [ -d "$RECIPES" ] || { echo "no recipe directory at $RECIPES" >&2; exit 2; } ;;
  off)  EDIT_ARGS=() ;;
  *) echo "NAVSAFE_EDITS must be auto or off" >&2; exit 2 ;;
esac

# The benchmark protocol, stated here rather than inherited from the wrapper:
# LQR tracking and a 120 s route budget (1200 frames at 10 Hz).
export NAVSAFE_CONTROLLER="${NAVSAFE_CONTROLLER:-lqr}"
export NAVSAFE_ROUTE_TIME_LIMIT_S="${NAVSAFE_ROUTE_TIME_LIMIT_S:-120}"
export NAVSAFE_EVAL_FRAMES="${NAVSAFE_EVAL_FRAMES:-1200}"

export NAVSAFE_RENDERER="${NAVSAFE_RENDERER:-docker}"
case "$NAVSAFE_RENDERER" in
  docker)   export NAVSAFE_RENDER_GPU="${NAVSAFE_RENDER_GPU:-0}" NAVSAFE_GPU="${NAVSAFE_GPU:-1}"
            # The benchmark is rendered with the harmonizer; the wrapper only
            # enables it when given somewhere to cache the weights.
            export NAVSAFE_HARMONIZER_CACHE="${NAVSAFE_HARMONIZER_CACHE:-$HOME/.cache/navsafe/harmonizer}"
            LANE="renderer gpu $NAVSAFE_RENDER_GPU, eval gpu $NAVSAFE_GPU" ;;
  existing) : "${NUREC_GRPC_HOST:?NAVSAFE_RENDERER=existing needs NUREC_GRPC_HOST}"
            export NAVSAFE_GPU="${NAVSAFE_GPU:-0}"
            LANE="renderer $NUREC_GRPC_HOST:${NUREC_GRPC_PORT:-8080}, eval gpu $NAVSAFE_GPU" ;;
  *) echo "NAVSAFE_RENDERER must be docker or existing" >&2; exit 2 ;;
esac
SHARD="${NAVSAFE_SHARD:-0/1}"
SHARD_I="${SHARD%/*}"; SHARD_N="${SHARD#*/}"
[ "$SHARD_N" -ge 1 ] 2>/dev/null && [ "$SHARD_I" -ge 0 ] 2>/dev/null && [ "$SHARD_I" -lt "$SHARD_N" ] \
  || { echo "NAVSAFE_SHARD must be i/n with 0 <= i < n" >&2; exit 2; }
RETRIES="${NAVSAFE_RETRIES:-2}"
SEED="${NAVSAFE_EVAL_SEED:-0}"

# A bundle is a directory with a manifest; anything else under full_test/ (a
# partial download, a stray file) is reported instead of failing mid-sweep.
TOKENS=()
if [ -n "${NAVSAFE_TOKENS:-}" ]; then
  while read -r t _; do
    [ -n "$t" ] && [ "${t#\#}" = "$t" ] && TOKENS+=("$t")
  done < "$NAVSAFE_TOKENS"
else
  for m in "$BUNDLES"/*/manifest.json; do
    [ -f "$m" ] && TOKENS+=("$(basename "$(dirname "$m")")")
  done
fi
[ "${#TOKENS[@]}" -gt 0 ] || { echo "no bundles under $BUNDLES" >&2; exit 2; }

echo "=== NavSafe full benchmark: ${#TOKENS[@]} scenario(s), shard $SHARD, model=$MODEL, edits=$EDITS, $LANE"
echo "=== -> $OUT_ROOT"
[ "${#TOKENS[@]}" -eq 280 ] || [ -n "${NAVSAFE_TOKENS:-}" ] \
  || echo "[warn] the published set has 280 bundles; found ${#TOKENS[@]} under $BUNDLES" >&2

if [ "${NAVSAFE_DRY_RUN:-0}" = "1" ]; then
  # The recipe column is the file that exists for the token, not the verdict:
  # a recipe that inserts nothing still runs unedited (editing/autoselect.py).
  for t in "${TOKENS[@]}"; do
    r=-
    if [ "$EDITS" = auto ]; then
      for f in "$RECIPES"/*."$t".yaml; do [ -f "$f" ] && r="$(basename "$f")"; done
    fi
    [ -f "$BUNDLES/$t/manifest.json" ] && s=ok || s=NO_BUNDLE
    printf '%s\t%s\t%s\n' "$t" "$s" "$r"
  done
  exit 0
fi

mkdir -p "$OUT_ROOT/scenarios" "$OUT_ROOT/logs"
# The data root holds asset/ and gait_bank/, which the renderer open()s at the
# recipe's absolute paths. Defaulted here because an edited cell without it
# fails at insert time, a long way from the missing mount.
export NAVSAFE_ASSET_MOUNT="${NAVSAFE_ASSET_MOUNT:-${NAVSAFE_DATA_ROOT:-}}"
export NAVSAFE_ENABLE_VIS="${NAVSAFE_ENABLE_VIS:-0}"

# An infra failure still exits 0 and writes a navsafe_metrics.json carrying
# status=excluded, so neither the exit code nor the file's existence says a
# scenario is finished. status == "scored" does.
is_scored() {
  [ -f "$1/navsafe_metrics.json" ] || return 1
  python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status")=="scored" else 1)' \
    "$1/navsafe_metrics.json" 2>/dev/null
}

for i in "${!TOKENS[@]}"; do
  [ $(( i % SHARD_N )) -eq "$SHARD_I" ] || continue
  t="${TOKENS[$i]}"
  cell="$OUT_ROOT/scenarios/$t"
  if [ "${NAVSAFE_FORCE:-0}" != "1" ] && is_scored "$cell"; then
    echo "[$t] already scored (NAVSAFE_FORCE=1 to redo)"
    continue
  fi
  attempt=0
  while [ "$attempt" -lt "$RETRIES" ]; do
    attempt=$(( attempt + 1 ))
    echo "[$t] start (attempt $attempt/$RETRIES, $(date -u +%H:%M:%S))"
    # An earlier result is set aside, never deleted: it may be the only copy
    # of a scored run, and a stale one must not pass for this attempt's.
    if [ -e "$cell" ] && ! mv "$cell" "$cell.prev-$(date -u +%Y%m%dT%H%M%SZ)-$$-$attempt"; then
      echo "[$t] cannot set the previous result aside; leaving this scenario for a later run" >&2
      break
    fi
    # One failing scenario must not abandon the rest; the reason is in its log.
    NAVSAFE_OUT="$cell" NAVSAFE_EVAL_LOG="$OUT_ROOT/logs/$t.log" \
    NAVSAFE_SERVE_LOG="$OUT_ROOT/logs/$t.serve-container.log" \
      timeout --kill-after=60 "$EPISODE_TIMEOUT" \
      bash "$HERE/run_bundle_eval.sh" "$t" policy "$MODEL" "$CKPT" --eval-seed "$SEED" \
        ${EDIT_ARGS[@]+"${EDIT_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$OUT_ROOT/logs/$t.wrapper.log" 2>&1 || true
    is_scored "$cell" && break
    echo "[$t] attempt $attempt not scored (see $OUT_ROOT/logs/$t.*log)"
  done
  is_scored "$cell" && echo "[$t] scored" || echo "[$t] NOT SCORED after $RETRIES attempt(s)"
done

# Written from what is on disk rather than accumulated by the loop, so a
# resumed or sharded sweep's summary covers every scenario. The exit status
# covers only this shard's, so a lane does not fail for another lane's cells.
SUMMARY="$OUT_ROOT/summary.tsv"
printf 'token\tstatus\tedited\ttermination\n' > "$SUMMARY"
MISSING=0; OWN=0
for i in "${!TOKENS[@]}"; do
  t="${TOKENS[$i]}"
  cell="$OUT_ROOT/scenarios/$t"
  if [ -f "$cell/navsafe_metrics.json" ]; then
    python3 - "$t" "$cell/navsafe_metrics.json" "$OUT_ROOT/logs/$t.log" >> "$SUMMARY" <<'PY'
import json, sys
tok, metrics, log = sys.argv[1:]
d = json.load(open(metrics))
try:
    edited = "yes" if "-> EDITED" in open(log, errors="replace").read() else "no"
except OSError:
    edited = "?"
print(tok, d.get("status"), edited,
      (d.get("termination") or {}).get("reason"), sep="\t")
PY
  else
    printf '%s\tMISSING\t?\t-\n' "$t" >> "$SUMMARY"
  fi
  [ $(( i % SHARD_N )) -eq "$SHARD_I" ] || continue
  OWN=$(( OWN + 1 ))
  is_scored "$cell" || MISSING=$(( MISSING + 1 ))
done

"$PY" "$REPO/navsafe/tools/report.py" --run "$OUT_ROOT" || true
echo
echo "=== shard $SHARD: $(( OWN - MISSING ))/$OWN scenario(s) scored"
echo "=== summary: $SUMMARY   report: $OUT_ROOT/report.md"
[ "$MISSING" -eq 0 ]
