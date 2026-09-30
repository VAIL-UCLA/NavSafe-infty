#!/usr/bin/env bash
# Score every frozen NavSafe recipe against one long-lived renderer.
#
#   navsafe/benchmark/eval/run_recipe_sweep.sh <run-slug> [extra eval_py123d args...]
#
# One closed-loop episode per recipe, sequentially, into a single run dir. This
# is the whole-benchmark counterpart to a hand-run eval_py123d.py: the only
# thing that varies between recipes is the token, its Arrow, the recipe path
# and the output dir, so everything else below is a constant.
#
# Two scenario layouts, detected per token rather than configured:
#
#   bundle   the published NavSafe distribution, one self-contained directory
#            per token -- arrow/, offsets/, <token>s1..s4.usdz, manifest.json.
#            Point NAVSAFE_BUNDLES at where you downloaded them. The handoff
#            comes from the manifest, resolved against your own path, because
#            the renderer open()s the offsets JSON directly and a baked
#            absolute path fails on every machine but the one that wrote it.
#   corpus   a reconstruction tree built in place (NAVSAFE_CORPUS): the Arrow
#            under <token>_20s/ with the four 5 s recons as siblings. The
#            handoff comes from `navsafe handoff`, and the renderer needs
#            --nurec-work-dir to find each recon's origin-offset sidecar.
#
# NOT the same tool as scripts/tools/run_navsafe_recipes.sh: that one renders
# ego-replay frames (no policy, no score) and starts its own `docker run`
# serve-grpc per token, which suits a workstation with docker and the
# navhard421 bundle layout. Here the renderer is a long-lived Kubernetes
# Deployment that already holds every host's scenes, so nothing is started or
# torn down and the sweep is only a loop.
#
# Four things are deliberate:
#
#   SEQUENTIAL       Two evals against one server corrupt each other:
#       close() calls restore_model_parameters, which rolls the whole scene
#       back and takes the other episode's inserted assets with it. Never
#       background these, and do not launch a second sweep alongside one.
#
#   keyed by RECIPE NAME, not token
#       C-10 and R-4 are both built on 0bcae698fd905226. Keying output by
#       token would have the second silently overwrite the first.
#
#   run dir from cfg.run_dir
#       One campaign is one directory that can be kept or dropped whole. The
#       slug has to say what the run tested; `cfg.run_dir` rejects counters.
#
#   e_plus only
#       The built scenario. The `e_zero` counterfactual is a separate sweep
#       (pass --recipe-variant e_zero as an extra arg) rather than an inner
#       loop, so the two never interleave in one run dir.
set -euo pipefail

SLUG="${1:?run slug, e.g. asset-scale-fix}"
shift
EXTRA=("$@")

REPO="${NAVSAFE_ROOT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RECIPES="${NAVSAFE_RECIPES:-$REPO/navsafe/benchmark/recipes}"
PY="${NAVSAFE_PYTHON:-python}"
# The recon rig -- the reconstruction's own calibrated camera -- is what
# every NavSafe number is rendered under. Stated here rather than inherited:
# a caller that happened to have NUREC_GRPC_CAM_RIG=navsim exported rendered
# three campaigns under the synthetic pinhole with nothing in the command
# saying so. Override it explicitly for an A/B.
export NUREC_GRPC_CAM_RIG="${NUREC_GRPC_CAM_RIG:-recon}"
MODEL="${NAVSAFE_MODEL:-drivor}"
# 160 frames = the 16 s the scorer's t_max already expects of a 20 s scenario.
# Empty by default: an indefinite episode. Set NAVSAFE_EVAL_FRAMES to cap the
# scored window again (collection, or reproducing a pre-ceiling run).
EVAL_FRAMES="${NAVSAFE_EVAL_FRAMES:-}"
# A wedged episode is not hypothetical: one spun at 100% CPU for 49 minutes
# after frame 180, writing nothing, and stalled the whole sweep behind it.
# render_rgb's own 300 s gRPC deadline does not cover it, because the spin is
# client-side. A full 200-frame episode is ~10 min, so this is loose enough to
# never fire on a healthy run and tight enough that one bad leaf costs one leaf.
EPISODE_TIMEOUT="${NAVSAFE_EPISODE_TIMEOUT:-45m}"
BUNDLES="${NAVSAFE_BUNDLES:-}"

# THE HAND-OFF COMES FROM THE RECIPE, and nothing here overrides it.
#
# Every leaf freezes `ego.replay_frames`, and that frame is the scenario: the
# hazard was solved to arrive as the policy takes over. Forcing a later
# hand-off replays the LOGGED ego straight through the event -- R-2's logged
# ego drove into the inserted bicycle at frame 62 of an 80-frame replay, so the
# policy never drove and the trace had no scored frames at all, while the other
# eight scored 2 s of an intended 16 s and reported `budget_expired`.
#
# `--keep-ego-replay-frames` is what makes eval_py123d honour a CLI value over
# the recipe's, and its own help restricts it to "pure log-replay renders
# (--ego-replay-frames >= --eval-frames)". A scored sweep is not that.
# NAVSAFE_REPLAY_FRAMES is here for those renders, and it drags the flag along
# because the two are only ever correct together.
# NAVSAFE_EVAL_FRAMES caps the scored window; unset (the default now) leaves the
# episode indefinite, ending on the taxonomy or the 60 s ceiling.
SHAPE=()
if [ -n "$EVAL_FRAMES" ]; then
  SHAPE+=(--eval-frames "$EVAL_FRAMES")
fi
if [ -n "${NAVSAFE_REPLAY_FRAMES:-}" ]; then
  SHAPE+=(--ego-replay-frames "$NAVSAFE_REPLAY_FRAMES" --keep-ego-replay-frames)
fi

# Every path comes from config.py, never from a literal here — it is the single
# place deployment paths are allowed to live, it reads them from the
# environment, and it also validates the slug.
read -r RUN_ROOT CORPUS CKPT < <("$PY" - "$SLUG" <<'PY'
import sys
from navsafe.benchmark import config as cfg
print(cfg.run_dir(sys.argv[1]), cfg.CORPUS, cfg.DEFAULT_CHECKPOINT)
PY
)

mkdir -p "$RUN_ROOT/eval" "$RUN_ROOT/logs"
SUMMARY="$RUN_ROOT/summary.tsv"
[ -f "$SUMMARY" ] || printf 'recipe\ttoken\tstatus\tframes\tout\n' > "$SUMMARY"
echo running > "$RUN_ROOT/STATUS"

# Leaves to leave out, space separated against the recipe's leaf prefix
# (NAVSAFE_SKIP="V-10 V-11"). Skipping every leaf is the loop's smoke test:
# it exercises the token parsing and the summary without touching a GPU.
SKIP="${NAVSAFE_SKIP:-}"

# `find` on a directory that does not exist yet exits non-zero, which under
# `set -e -o pipefail` would kill the sweep before its first episode.
frames_in() {
  [ -d "$1" ] || { echo 0; return; }
  find "$1" -name 'cam_f0.jpg' 2>/dev/null | wc -l
}

echo "=== sweep $SLUG -> $RUN_ROOT"
FAILED=0
shopt -s nullglob
for RECIPE in "$RECIPES"/*.yaml; do
  NAME="$(basename "$RECIPE" .yaml)"           # e.g. R-4.0bcae698fd905226_20s
  LEAF="${NAME%%.*}"                           # e.g. R-4
  # <LEAF>.<16 hex token>[_<window>] — the window suffix is optional, because
  # not every recipe carries one and requiring it silently left TOKEN set to
  # the whole filename, which then failed as an unknown token.
  TOKEN="$(sed -E 's/^[^.]+\.([0-9a-f]{16}).*$/\1/' <<< "$NAME")"
  OUT="$RUN_ROOT/eval/$NAME"

  if [ -n "$SKIP" ] && [[ " $SKIP " == *" $LEAF "* ]]; then
    printf '%s\t%s\tSKIPPED\t0\t-\n' "$NAME" "$TOKEN" >> "$SUMMARY"
    echo "[skip] $NAME: $LEAF excluded by NAVSAFE_SKIP"
    continue
  fi

  # Re-runnable: a FINISHED episode is left alone, so an interrupted sweep can
  # be relaunched without repeating work. Finished means eval_py123d printed
  # its DONE line -- not "wrote some frames", which is also true of an episode
  # killed halfway. The renderer pod is ephemeral and takes the sweep with it
  # when it goes, so a half-episode is the normal interruption, and counting
  # frames silently accepted one (18 of ~160) as complete.
  N=$(frames_in "$OUT")
  if [ "${NAVSAFE_FORCE:-0}" != "1" ] \
     && grep -qs "^\[eval_py123d\] DONE\." "$OUT/eval.log"; then
    printf '%s\t%s\tALREADY\t%s\t%s\n' "$NAME" "$TOKEN" "$N" "$OUT" >> "$SUMMARY"
    echo "[done] $NAME: $N frames already there (NAVSAFE_FORCE=1 to redo)"
    continue
  fi

  mkdir -p "$OUT"
  echo "=== $NAME  (token $TOKEN) -> $OUT"

  # All four 5 s reconstructions are armed, always — the renderer picks between
  # them per frame by projecting the live ego onto the logged track, so a
  # handoff covering only the frames this shape reaches renders the road from
  # the wrong model as soon as someone lengthens the episode.
  B="${BUNDLES:+$BUNDLES/$TOKEN}"
  WORKDIR=()
  if [ -n "$B" ] && [ -f "$B/manifest.json" ]; then
    ARROW="$B/arrow"
    if ! HANDOFF="$("$PY" -m navsafe.benchmark.eval.bundle --handoff "$B" \
                    2>"$OUT/handoff.log")"; then
      printf '%s\t%s\tNO_HANDOFF\t0\t-\n' "$NAME" "$TOKEN" >> "$SUMMARY"
      echo "    NO_HANDOFF (see $OUT/handoff.log)"; FAILED=1; continue
    fi
    export NUREC_GRPC_HANDOFF="$HANDOFF"
  else
    ARROW="$CORPUS/${TOKEN}_20s/arrow"
    WORKDIR=(--nurec-work-dir "$CORPUS")
    # `navsafe handoff` warns on stderr when a recon is missing from the served
    # pool — the failure that otherwise shows up as a silent raster fallback.
    if navsafe handoff --token "$TOKEN" \
         >"$OUT/handoff.out" 2>"$OUT/handoff.log"; then
      HANDOFF="$(tail -1 "$OUT/handoff.out")"
    else
      # It reports an unresolvable token on STDOUT, which used to be captured
      # into the handoff string and discarded with it — leaving an empty
      # handoff.log and no stated reason for the skip.
      cat "$OUT/handoff.out" >> "$OUT/handoff.log"
      printf '%s\t%s\tNO_HANDOFF\t0\t-\n' "$NAME" "$TOKEN" >> "$SUMMARY"
      echo "    NO_HANDOFF (see $OUT/handoff.log)"; FAILED=1; continue
    fi
    eval "$HANDOFF"
  fi
  [ -s "$OUT/handoff.log" ] && cat "$OUT/handoff.log" || true

  # `|| true`: one episode failing must not abandon the rest of the sweep. The
  # per-episode eval.log holds the reason and summary.tsv records the status.
  timeout --signal=KILL "$EPISODE_TIMEOUT" \
    "$PY" "$REPO/scripts/tools/eval_py123d.py" \
      --scenario-source py123d \
      --py123d-data-root "$ARROW" \
      --py123d-scene-index 0 \
      ${WORKDIR[@]+"${WORKDIR[@]}"} \
      --render-backend nurec_grpc \
      --model-type "$MODEL" --checkpoint "$CKPT" \
      --recipe "$RECIPE" --recipe-variant e_plus \
      --traffic-mode navsafe "${SHAPE[@]}" \
      --enable-vis --log-level INFO \
      --output-dir "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} \
      > "$OUT/eval.log" 2>&1 || true

  # Same completion test as the skip above, and for the same reason: an
  # episode killed at frame 182 of 200 leaves plenty of frames behind but no
  # metrics.json, and calling that OK once sent a whole leaf's missing score
  # unnoticed into a results table.
  N=$(frames_in "$OUT")
  if grep -qs "^\[eval_py123d\] DONE\." "$OUT/eval.log"; then
    STATUS=OK
  else
    STATUS=FAILED; FAILED=1
  fi
  echo "    $STATUS ($N frames)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$NAME" "$TOKEN" "$STATUS" "$N" "$OUT" >> "$SUMMARY"
done

[ "$FAILED" -eq 0 ] && echo done > "$RUN_ROOT/STATUS" || echo failed > "$RUN_ROOT/STATUS"
echo
echo "=== summary ($RUN_ROOT/STATUS: $(cat "$RUN_ROOT/STATUS")) ==="
column -t -s$'\t' "$SUMMARY"
