#!/usr/bin/env bash
# Evaluate one policy on the state-perturbation set: 56 controlled events on
# 28 full_test scenarios, defined by recipes/proxy_set_state_perturbation.
#
#   run_perturbation_set.sh MODEL CHECKPOINT [extra the evaluator args...]
#
# The counterpart of run_full_benchmark.sh for this set, and the same shape: a
# loop over run_bundle_eval.sh, one cell per (scenario, event). The cells, their
# recipes and their hashes come from the set's index.json; a recipe whose bytes
# no longer match its recorded prepared_sha256 is refused rather than run.
#
#   NAVSAFE_DATA_ROOT   required: the HF snapshot root (full_test/, asset/, gait_bank/)
#   NAVSAFE_OUT_ROOT    run directory (default <repo>/output/perturbation_<MODEL>)
#   NAVSAFE_VARIANT     event (default) | baseline | both
#   NAVSAFE_EVAL_SEED   policy seed (default 0)
#   NAVSAFE_RENDERER, NAVSAFE_RENDER_GPU, NAVSAFE_GPU, NAVSAFE_SHARD,
#   NAVSAFE_RETRIES, NAVSAFE_FORCE, NAVSAFE_DRY_RUN
#                       as in run_full_benchmark.sh
#
# Protocol: navsafe traffic, LQR tracking, 600 scored frames (60 s), seed 0,
# the hand-off frame each recipe freezes, and visualization ON with the left,
# right and rear cameras. Visualization is part of the protocol here, not a
# convenience: it renders every simulation step instead of only the steps the
# policy consumes, so turning it off changes the render cadence.
#
# Fifteen events move a logged vehicle onto an authored path while keeping its
# identity. Those cells run with --asset-harvester-replace, and count as done
# only with a harvester_takeover_audit.json showing that vehicle's own
# harvested asset was applied; a scored episode alone does not show that.
set -euo pipefail

MODEL="${1:?model type, e.g. drivor}"
CKPT="${2:?checkpoint path (or the adapter sentinel, e.g. none for pdm_closed)}"
shift 2
EXTRA=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${NAVSAFE_REPO:-$(cd "$HERE/../../.." && pwd)}"
: "${NAVSAFE_DATA_ROOT:?set NAVSAFE_DATA_ROOT}"
SET="${NAVSAFE_PERTURBATION_SET:-$REPO/navsafe/benchmark/recipes/proxy_set_state_perturbation}"
[ -f "$SET/index.json" ] || { echo "no index.json under $SET" >&2; exit 2; }
OUT_ROOT="${NAVSAFE_OUT_ROOT:-$REPO/output/perturbation_$MODEL}"
VARIANT="${NAVSAFE_VARIANT:-event}"
EPISODE_TIMEOUT="${NAVSAFE_EPISODE_TIMEOUT:-45m}"
RETRIES="${NAVSAFE_RETRIES:-2}"
SEED="${NAVSAFE_EVAL_SEED:-0}"

export NAVSAFE_CONTROLLER="${NAVSAFE_CONTROLLER:-lqr}"
export NAVSAFE_EVAL_FRAMES="${NAVSAFE_EVAL_FRAMES:-600}"
export NAVSAFE_ENABLE_VIS="${NAVSAFE_ENABLE_VIS:-1}"
export NAVSAFE_VIS_CAMS="${NAVSAFE_VIS_CAMS-CAM_L0,CAM_R0,CAM_B0}"
export NAVSAFE_RENDERER="${NAVSAFE_RENDERER:-docker}"
case "$NAVSAFE_RENDERER" in
  docker)   export NAVSAFE_RENDER_GPU="${NAVSAFE_RENDER_GPU:-0}" NAVSAFE_GPU="${NAVSAFE_GPU:-1}"
            export NAVSAFE_HARMONIZER_CACHE="${NAVSAFE_HARMONIZER_CACHE:-$HOME/.cache/navsafe/harmonizer}" ;;
  existing) : "${NUREC_GRPC_HOST:?NAVSAFE_RENDERER=existing needs NUREC_GRPC_HOST}"
            export NAVSAFE_GPU="${NAVSAFE_GPU:-0}" ;;
  *) echo "NAVSAFE_RENDERER must be docker or existing" >&2; exit 2 ;;
esac
SHARD="${NAVSAFE_SHARD:-0/1}"
SHARD_I="${SHARD%/*}"; SHARD_N="${SHARD#*/}"
[ "$SHARD_N" -ge 1 ] 2>/dev/null && [ "$SHARD_I" -ge 0 ] 2>/dev/null && [ "$SHARD_I" -lt "$SHARD_N" ] \
  || { echo "NAVSAFE_SHARD must be i/n with 0 <= i < n" >&2; exit 2; }

# One row per cell: name, token, recipe file, harvester manifest (or -), and
# the logged tracks the recipe takes over (or -). Every recipe is hash-checked
# first, and the list is written whole before anything runs: a failure on a
# later recipe must not leave the earlier cells queued.
CELL_LIST="$(mktemp)"
trap 'rm -f "$CELL_LIST"' EXIT
python3 - "$SET" "$VARIANT" > "$CELL_LIST" <<'CELLS_PY' || { echo "cell list not built; nothing was run" >&2; exit 2; }
import hashlib, json, sys
from pathlib import Path
root, want = Path(sys.argv[1]), sys.argv[2]
if want not in ("event", "baseline", "both"):
    sys.exit("NAVSAFE_VARIANT must be event, baseline or both")
for e in json.loads((root / "index.json").read_text())["events"]:
    for v in e["variants"]:
        if want != "both" and v["variant"] != want:
            continue
        got = hashlib.sha256((root / v["recipe"]).read_bytes()).hexdigest()
        if got != v["prepared_sha256"]:
            sys.exit(f"{v['recipe']}: sha256 {got} is not the recorded {v['prepared_sha256']}")
        name = f"{e['leaf']}.{e['token']}.{e['event']}.{v['variant']}"
        tracks = ",".join(sorted(t["source_track_id"] for t in v.get("takeovers") or []))
        print(name, e["token"], v["recipe"], v.get("asset_harvester_manifest") or "-",
              tracks or "-", sep="\t")
CELLS_PY
CELLS=()
while IFS= read -r row; do CELLS+=("$row"); done < "$CELL_LIST"
[ "${#CELLS[@]}" -gt 0 ] || { echo "no cells for NAVSAFE_VARIANT=$VARIANT" >&2; exit 2; }

echo "=== NavSafe state-perturbation set: ${#CELLS[@]} cell(s), variant=$VARIANT, shard $SHARD, model=$MODEL, seed=$SEED"
echo "=== -> $OUT_ROOT"

if [ "${NAVSAFE_DRY_RUN:-0}" = "1" ]; then
  for row in "${CELLS[@]}"; do
    IFS=$'\t' read -r name t recipe ah tracks <<< "$row"
    [ -f "$NAVSAFE_DATA_ROOT/full_test/$t/manifest.json" ] && s=ok || s=NO_BUNDLE
    printf '%s\t%s\t%s\t%s\n' "$name" "$s" "$ah" "$tracks"
  done
  exit 0
fi

mkdir -p "$OUT_ROOT/scenarios" "$OUT_ROOT/logs"
export NAVSAFE_ASSET_MOUNT="${NAVSAFE_ASSET_MOUNT:-$NAVSAFE_DATA_ROOT}"

# Done = scored, and for a takeover cell also a valid audit: exactly the
# recipe's tracks were required, each was replaced in at least one scene, and
# each has its asset recorded.
is_done() {
  python3 - "$1" "$2" <<'DONE_PY' 2>/dev/null
import json, sys
from pathlib import Path
cell, tracks = Path(sys.argv[1]), sys.argv[2]
try:
    if json.loads((cell / "navsafe_metrics.json").read_text()).get("status") != "scored":
        sys.exit(1)
    if tracks != "-":
        need = set(tracks.split(","))
        a = json.loads((cell / "harvester_takeover_audit.json").read_text())
        applied = {t for v in a["applied_tracks_by_scene"].values() for t in v}
        if set(a["required_tracks"]) != need or not need <= applied \
                or not need <= set(a["asset_by_track"]):
            sys.exit(1)
except Exception:
    sys.exit(1)
DONE_PY
}

for i in "${!CELLS[@]}"; do
  [ $(( i % SHARD_N )) -eq "$SHARD_I" ] || continue
  IFS=$'\t' read -r name t recipe ah tracks <<< "${CELLS[$i]}"
  cell="$OUT_ROOT/scenarios/$name"
  if [ "${NAVSAFE_FORCE:-0}" != "1" ] && is_done "$cell" "$tracks"; then
    echo "[$name] already done (NAVSAFE_FORCE=1 to redo)"
    continue
  fi
  AH=()
  if [ "$ah" != "-" ]; then
    [ -f "$NAVSAFE_DATA_ROOT/$ah" ] || { echo "[$name] NOT DONE: missing $NAVSAFE_DATA_ROOT/$ah" >&2; continue; }
    AH=(--asset-harvester-replace "$NAVSAFE_DATA_ROOT/$ah")
  fi
  attempt=0
  while [ "$attempt" -lt "$RETRIES" ]; do
    attempt=$(( attempt + 1 ))
    echo "[$name] start (attempt $attempt/$RETRIES, $(date -u +%H:%M:%S))"
    # An earlier result is set aside, never deleted: it may be the only copy
    # of a scored run, and a stale one must not pass for this attempt's.
    if [ -e "$cell" ] && ! mv "$cell" "$cell.prev-$(date -u +%Y%m%dT%H%M%SZ)-$$-$attempt"; then
      echo "[$name] cannot set the previous result aside; leaving this scenario for a later run" >&2
      break
    fi
    NAVSAFE_OUT="$cell" NAVSAFE_EVAL_LOG="$OUT_ROOT/logs/$name.log" \
    NAVSAFE_SERVE_LOG="$OUT_ROOT/logs/$name.serve-container.log" \
      timeout --kill-after=60 "$EPISODE_TIMEOUT" \
      bash "$HERE/run_bundle_eval.sh" "$t" policy "$MODEL" "$CKPT" \
        --traffic-mode navsafe --recipe "$SET/$recipe" --eval-seed "$SEED" \
        ${AH[@]+"${AH[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$OUT_ROOT/logs/$name.wrapper.log" 2>&1 || true
    is_done "$cell" "$tracks" && break
    echo "[$name] attempt $attempt not done (see $OUT_ROOT/logs/$name.*log)"
  done
  is_done "$cell" "$tracks" && echo "[$name] done" || echo "[$name] NOT DONE after $RETRIES attempt(s)"
done

# Two events share a token, so the per-token report of the full sweep would
# overwrite one with the other; this table is keyed by cell instead. It covers
# every cell, while the exit status covers only this shard's, so a lane does
# not fail for cells another lane owns.
SUMMARY="$OUT_ROOT/summary.tsv"
printf 'cell\tstatus\tdriving_score\tsuccess\ttermination\ttakeover_audit\n' > "$SUMMARY"
MISSING=0; OWN=0
for i in "${!CELLS[@]}"; do
  IFS=$'\t' read -r name _ _ _ tracks <<< "${CELLS[$i]}"
  cell="$OUT_ROOT/scenarios/$name"
  audit=-
  if [ "$tracks" != "-" ]; then is_done "$cell" "$tracks" && audit=ok || audit=INVALID; fi
  if [ -f "$cell/navsafe_metrics.json" ]; then
    python3 - "$name" "$cell/navsafe_metrics.json" "$audit" >> "$SUMMARY" <<'ROW_PY'
import json, sys
d = json.load(open(sys.argv[2])); m = d.get("metrics") or {}
print(sys.argv[1], d.get("status"), m.get("driving_score"), m.get("success"),
      (d.get("termination") or {}).get("reason"), sys.argv[3], sep="\t")
ROW_PY
  else
    printf '%s\tMISSING\t-\t-\t-\t%s\n' "$name" "$audit" >> "$SUMMARY"
  fi
  [ $(( i % SHARD_N )) -eq "$SHARD_I" ] || continue
  OWN=$(( OWN + 1 ))
  is_done "$cell" "$tracks" || MISSING=$(( MISSING + 1 ))
done
echo "=== shard $SHARD: $(( OWN - MISSING ))/$OWN cell(s) done; summary: $SUMMARY"
[ "$MISSING" -eq 0 ]
