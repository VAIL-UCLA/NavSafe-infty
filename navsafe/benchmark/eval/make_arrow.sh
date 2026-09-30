#!/usr/bin/env bash
# Build the Arrow scenario source (ego, boxes, map, route) for one navhard
# scenario window, from the raw nuPlan log.
#
# The NuRec models render pixels; they carry no scenario. Without this Arrow
# there is nothing to drive — see PIPELINE.md.
#
#   make_arrow.sh TOKEN LOG_NAME T0_US T1_US OUT_ROOT
#
# Writes OUT_ROOT/logs/nuplan_test/<TOKEN>/*.arrow + OUT_ROOT/maps/nuplan/*.arrow
# The Arrow's log_name is the TOKEN, so it matches the grpc scene ids.
set -euo pipefail

TOKEN="${1:?token}"; LOG="${2:?log_name}"; T0="${3:?t0_us}"; T1="${4:?t1_us}"
OUT="${5:?out_root}"

REPO="${NAVSAFE_ROOT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VENV="${NAVSAFE_CONVERT_PYTHON:-${HOME}/.venvs/navsafe-convert/bin/python}"
# nuPlan roots. The devkit reads the log DB directly; navhard logs are the test
# split. Maps are the shared nuplan-maps-v1.0 tree.
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-${HOME}/data/nuplan}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${HOME}/data/nuplan/maps}"
export NUPLAN_SENSOR_ROOT="${NUPLAN_SENSOR_ROOT:-${HOME}/data/nuplan/nuplan-v1.1/sensor_blobs}"

if [ ! -x "$VENV" ]; then
  echo "convert venv not found: $VENV" >&2
  echo "create it once:" >&2
  echo "  uv venv --python 3.11 <dir> && uv pip install --python <dir>/bin/python py123d nuplan-devkit" >&2
  exit 2
fi

# py123d resolves dataset configs from its own package dir, so the config is
# installed there rather than passed by path (hydra's config_path is baked into
# the entry point). Copying is idempotent and keeps the repo the source of truth.
CFG_DIR="$("$VENV" - <<'PY'
import pathlib, py123d.script as s
print(pathlib.Path(s.__file__).parent / "config" / "conversion" / "dataset")
PY
)"
cp "$(dirname "${BASH_SOURCE[0]}")/nuplan-navhard.yaml" "$CFG_DIR/nuplan-navhard.yaml"

mkdir -p "$OUT"
echo "[arrow] $TOKEN  $LOG  [$T0, $T1]  -> $OUT"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PY123D_DATA_ROOT="$OUT" \
  "$(dirname "$VENV")/py123d-conversion" dataset=nuplan-navhard \
    "dataset.parser.scenes=[[$LOG,$TOKEN,$T0,$T1]]"

ls "$OUT/logs/nuplan_test/$TOKEN"/*.arrow >/dev/null 2>&1 \
  && echo "[arrow] OK  $(ls "$OUT/logs/nuplan_test/$TOKEN" | tr '\n' ' ')" \
  || { echo "[arrow] FAIL: no arrow written for $TOKEN" >&2; exit 1; }
