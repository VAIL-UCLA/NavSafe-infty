#!/bin/bash
# Render one scenario as a stitched 20 s clip: 4x5s time-based handoff, full
# ego-replay, harmonizer on, pure gRPC frames (no overlay, no BEV). Writes
# $NAVSAFE_STITCH_OUT/<CAT>-<BASE>.mp4 — the shape the website wants.
#
#   stitch_eval.sh CAT BASE T0_US
#
# Deployment paths are the caller's; this file names none of its own. Set:
#
#   NAVSAFE_ROOT             repo root
#   NAVSAFE_CORPUS       corpus holding <BASE>/arrow and <BASE>sN/clips/...
#   NAVSAFE_CHECKPOINT   policy weights
#   NAVSAFE_STITCH_OUT   where the .mp4 goes
#
# Optional: NAVSAFE_PYTHON, NAVSAFE_NRE_STUBS (NRE gRPC stubs, if they are not
# already importable), NUREC_GRPC_HOST, NUPLAN_MAPS_ROOT, CUDA_VISIBLE_DEVICES.
set -uo pipefail

: "${NAVSAFE_ROOT:?set NAVSAFE_ROOT to the repo root}"
: "${NAVSAFE_CORPUS:?set NAVSAFE_CORPUS to the corpus holding this scenario}"
: "${NAVSAFE_CHECKPOINT:?set NAVSAFE_CHECKPOINT to the policy weights}"
: "${NAVSAFE_STITCH_OUT:?set NAVSAFE_STITCH_OUT to the directory for the .mp4}"
PY="${NAVSAFE_PYTHON:-python}"

CAT=$1; BASE=$2; T0=$3; SEG=5000000
W1=$((T0+SEG)); W2=$((T0+2*SEG)); W3=$((T0+3*SEG)); W4=$((T0+4*SEG))
off(){ echo "$NAVSAFE_CORPUS/${BASE}s$1/clips/${BASE}s$1/nurec_origin_offset.json"; }
HANDOFF="${BASE}s1,$(off 1),$T0,$W1;${BASE}s2,$(off 2),$W1,$W2;${BASE}s3,$(off 3),$W2,$W3;${BASE}s4,$(off 4),$W3,$W4"
DR=/tmp/dr_$BASE; rm -rf $DR; mkdir -p $DR/logs/nuplan_test
ln -s "$NAVSAFE_CORPUS/$BASE/arrow/logs/nuplan_test/$BASE" $DR/logs/nuplan_test/$BASE
ln -s "$NAVSAFE_CORPUS/$BASE/arrow/maps" $DR/maps
OUT=/tmp/stitch_$BASE; rm -rf $OUT; mkdir -p $OUT
echo "$(date +%H:%M) === stitch $CAT ($BASE): 4x5s handoff, 20s ego-replay, no overlay ==="
NUREC_GRPC_HANDOFF="$HANDOFF" NAVSAFE_NO_OVERLAY=1 \
PYTHONPATH="${NAVSAFE_NRE_STUBS:+$NAVSAFE_NRE_STUBS:}$NAVSAFE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  NUREC_GRPC_CAM_RIG="${NUREC_GRPC_CAM_RIG:-recon}" \
  PY123D_RECENTER=1 NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}" \
  ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  timeout 1800 "$PY" "$NAVSAFE_ROOT/navsafe/cli/eval_entry.py" \
    --scenario-source py123d --py123d-data-root $DR --py123d-scene-index 0 \
    --render-backend nurec_grpc --model-type drivor --checkpoint "$NAVSAFE_CHECKPOINT" \
    --traffic-mode log_replay --controller pure_pursuit \
    --ego-replay-frames 200 --eval-frames 0 --replan-rate 5 --camera-resolution-scale 1.0 --enable-vis \
    --output-dir $OUT > /tmp/stitch_$BASE.log 2>&1
fd=$(ls -d $OUT/*/frames 2>/dev/null | head -1)
n=$(ls $fd/*/cam_f0.jpg 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  SEQ=/tmp/seq_$BASE; rm -rf $SEQ; mkdir -p $SEQ; j=0
  for d in $(ls -d $fd/*/ 2>/dev/null | sort); do [ -f "${d}cam_f0.jpg" ] && ln -s "${d}cam_f0.jpg" "$SEQ/$(printf %05d $j).jpg" && j=$((j+1)); done
  DEST="$NAVSAFE_STITCH_OUT/${CAT}-${BASE}.mp4"
  mkdir -p "$(dirname "$DEST")"
  ffmpeg -y -framerate 10 -i $SEQ/%05d.jpg -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$DEST" >/tmp/ffmpeg_$BASE.log 2>&1
  rm -rf $SEQ
  [ -f "$DEST" ] && echo "$(date +%H:%M) MP4_OK ${CAT}-${BASE} ($j frames, $(du -h "$DEST"|cut -f1))" || { echo "$(date +%H:%M) MP4_FAIL $BASE"; tail -3 /tmp/ffmpeg_$BASE.log; }
else
  echo "$(date +%H:%M) EVAL_FAIL $BASE"; tail -12 /tmp/stitch_$BASE.log
fi
