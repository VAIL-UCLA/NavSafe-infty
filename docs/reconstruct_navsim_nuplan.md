# Reconstructing new clips

This guide turns a window of a raw nuPlan log into a NuRec reconstruction and a matching scenario that NavSafe can evaluate. Evaluating the published dataset does not need any of it: start from the [downloaded bundles](navsafe_eval.md).

The steps are: convert the raw clip to NCore, generate auxiliary signals, fit and export the reconstruction, build the Arrow scenario for the same window, then serve and evaluate. Conversion needs the nuPlan devkit and the `nvidia-ncore` writer in the NavSafe environment. Auxiliary signals, fitting and export run in NVIDIA NRE containers on a host with Docker and a GPU.

## 1. Select a clip

A clip is a log name, a token to name the result, and a time window in microseconds.

```bash
export NUPLAN_ROOT="<absolute-nuplan-root>"     # contains nuplan-v1.1/{splits,maps,sensor_blobs}
export WORK="<absolute-work-directory>"         # NCore clips and reconstructions are written here
export TOKEN="<scenario-token>"
export NUPLAN_LOG="<nuplan-log-name>"           # database name without .db
export NUPLAN_SPLIT="<py123d-nuplan-split>"     # for example nuplan_test
export T0_US="<start-timestamp-microseconds>"
export T1_US="<end-timestamp-microseconds>"
export NCORE_REREF_FRAME0=1                     # express the clip relative to the first ego pose
export PY123D_RECENTER=1                        # the matching convention at evaluation time
```

Keep the log databases on a local disk during conversion; the devkit's many small reads are slow on a network filesystem.

## 2. Convert to NCore

```bash
python - <<'PY'
import os
from pathlib import Path
from navsafe.gs3d_converter.log_ingestion import NuplanLogReader
from navsafe.gs3d_converter.ncore_bridge import NCoreBridge
root = Path(os.environ["NUPLAN_ROOT"])
reader = NuplanLogReader(
    nuplan_data_root=root,
    sensor_root=root / "nuplan-v1.1/sensor_blobs",
    maps_root=root / "nuplan-v1.1/maps",
    scenes=[(os.environ["NUPLAN_LOG"], os.environ["TOKEN"],
             int(os.environ["T0_US"]), int(os.environ["T1_US"]))],
    split=os.environ["NUPLAN_SPLIT"],
)
scene = next(iter(reader.iter_scenes()))
clip_dir = NCoreBridge.prepare(scene, Path(os.environ["WORK"]) / "ncore" / os.environ["TOKEN"])
print(clip_dir)
PY
```

`scenes` takes `(log_name, token, start_us, end_us)` tuples. The clip is written to `<work>/ncore/<token>/clips/<token>/` as a `pai_<token>.json` manifest with camera and lidar shards, plus a file recording the clip's origin offset. Keep that file: evaluation uses it to align the reconstruction with the scenario.

## 3. Generate auxiliary signals

Log Docker in to the NVIDIA registry with your [NGC API key](https://org.ngc.nvidia.com/setup/api-key). The user name is the literal string `$oauthtoken`.

```bash
export NGC_API_KEY="<ngc-api-key>"
printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

```bash
export NAVSAFE_NUREC_GPUS="<gpu-index-or-list>" # default: all visible GPUs
export CLIP_DIR="$WORK/ncore/$TOKEN/clips/$TOKEN"
python - <<'PY'
import os
from pathlib import Path
from navsafe.gs3d_converter.nurec_runner import run_aux
run_aux(clip_dir=Path(os.environ["CLIP_DIR"]), work_dir=Path(os.environ["WORK"]))
PY
```

The signals are written into the clip.

| `run_aux` argument | Default | Meaning |
| :--- | :--- | :--- |
| `camera_ids`, `lidar_ids` | all | Restrict the sensors processed. |
| `segmentation_backend` | `mask2former` | Source of the semantic masks used in fitting. |
| `ego_mask` | `True` | Generate the mask of the ego vehicle's own body. |
| `depth_backend` | `none` | Monocular depth backend, when depth supervision is wanted. |
| `nurec_image` | pinned in `nurec_runner.py` | NRE tools image. |

## 4. Fit and export

```bash
export RECON_WORK="$WORK/recon/$TOKEN"
python - <<'PY'
import os
from pathlib import Path
from navsafe.gs3d_converter.nurec_runner import run_train, run_export
clip = Path(os.environ["CLIP_DIR"])
work = Path(os.environ["RECON_WORK"])
run = run_train(
    dataset_dir=clip, work_dir=work,
    static_separation=True, lidar_supervision=True,
)
exported = run_export(run_dir=run, work_dir=work, dataset_dir=clip, export_usdz=True)
print(run)
print(exported)
PY
```

| `run_train` argument | Default | Meaning |
| :--- | :--- | :--- |
| `static_separation` | `False` | Put parked and boxed objects in their own layers so they can be edited, instead of fusing them into the background. Enable it for scenes that will be edited. |
| `lidar_supervision` | `False` | Supervise geometry with lidar. |
| `config_name` | chosen by the helper | NRE training configuration. An explicit one must match the sensor rig. |
| `aux_data` | `True` | Use the auxiliary signals from step 3. |
| `subsample` | `2` | Image subsampling during fitting. |
| `overrides` | none | Additional Hydra overrides. Record them with the reconstruction. |

| `run_export` argument | Default | Meaning |
| :--- | :--- | :--- |
| `export_usdz` | `True` | Write the `.usdz` file the renderer serves. |
| `dataset_dir` | none | Source clip; needed with `export_ncore_tracks`. |
| `export_ncore_tracks` | `False` | Also export per-sensor poses. |

Resolution, cameras and training budget affect quality, GPU memory and run time; one configuration does not suit every source dataset.

## 5. Build the Arrow scenario

The reconstruction only renders. The simulator also needs the ego states, actor tracks, route and map for the same window, as py123d Arrow.

```bash
export NAVSAFE_REPO="<absolute-repository-directory>"
export NAVSAFE_CONVERT_PYTHON="$NAVSAFE_REPO/.venv/bin/python"
export NUPLAN_DATA_ROOT="$NUPLAN_ROOT"
export NUPLAN_MAPS_ROOT="$NUPLAN_ROOT/nuplan-v1.1/maps"
export NUPLAN_SENSOR_ROOT="$NUPLAN_ROOT/nuplan-v1.1/sensor_blobs"
bash navsafe/benchmark/eval/make_arrow.sh "$TOKEN" "$NUPLAN_LOG" "$T0_US" "$T1_US" "<arrow-output-root>"
```

The script writes `logs/nuplan_test/<token>/` and `maps/` below the output root. It installs the bundled `nuplan-navhard` conversion configuration into the py123d package, which therefore must be writable, and it reads logs of the nuPlan test split; another split needs its own conversion configuration.

For a scenario described by a mined `seed.json`, `python -m navsafe.benchmark.world.run_ncore --seed "<seed-json-or-directory>"` performs the NCore conversion of step 2 from the seed's paths and timestamps. `--nuplan-local` sets where the log database is staged on local disk, and `--force` repeats a conversion that already exists.

## 6. Serve and evaluate

```bash
export NUREC_PORT="<renderer-port>"
python - <<'PY'
import os
from pathlib import Path
from navsafe.gs3d_converter.nurec_runner import run_serve_grpc
run_serve_grpc(
    work_dir=Path(os.environ["WORK"]) / "recon",
    host_port=int(os.environ["NUREC_PORT"]),
    artifact_glob="*/output*/*/usd-out/*.usdz",
    enable_harmonizer=True,
)
PY
```

`artifact_glob` is relative to `work_dir` and must match the exported `.usdz` files. `enable_harmonizer=False` turns off DiffusionHarmonizer post-processing, which changes the images.

Then evaluate against this renderer as in [existing renderer](navsafe_eval.md#existing-renderer), using the Arrow root from step 5. A scenario made of several consecutive reconstructions also needs the handoff descriptor described in the [bundle notes](../navsafe/benchmark/eval/PIPELINE.md).
