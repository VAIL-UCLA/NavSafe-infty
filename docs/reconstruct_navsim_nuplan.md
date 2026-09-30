# Reconstruct NavSim / nuPlan clips with NuRec

This workflow creates new reconstructions. Evaluating the published dataset starts from [downloaded bundles](navsafe_eval.md) and does not require these preparation steps.

The examples use the installed NavSafe environment and the Python APIs present in `navsafe/gs3d_converter`. The raw nuPlan ingestion path additionally needs the upstream nuPlan parser/devkit and the `nvidia-ncore` writer available in that environment. NRE auxiliary-data generation, fitting and export run in NVIDIA containers on GPU-enabled Docker hosts.

## 1. Select source data and an output root

A source clip is identified by a log name, token and timestamp window in microseconds. Its sensor calibration and map must match that window. Keep the symbolic Arrow scenario and reconstructed camera scene in the same coordinate frame.

```bash
export NUPLAN_ROOT="<absolute-nuplan-root>" # Required here: raw dataset root containing nuplan-v1.1/splits, maps and sensor_blobs.
export WORK="<absolute-reconstruction-work-directory>" # Required here: persistent output root for NCore stores and fitted reconstructions.
export TOKEN="<scenario-token>" # Required here: identifier used for the converted clip and output directories.
export NUPLAN_LOG="<nuplan-log-name>" # Required here: source database name without its .db suffix.
export NUPLAN_SPLIT="<py123d-nuplan-split>" # Required here: parser split corresponding to your source logs, such as nuplan_test.
export T0_US="<start-timestamp-microseconds>" # Required here: beginning of the source clip window.
export T1_US="<end-timestamp-microseconds>" # Required here: end of the source clip window, greater than T0_US.
export NCORE_REREF_FRAME0=1 # Optional, enabled here: re-reference reconstruction coordinates to frame-zero ego for numerical precision.
export PY123D_RECENTER=1 # Optional, enabled here: use the matching local coordinate convention during evaluation.
```

The raw reader expects a root containing `nuplan-v1.1`. Store log databases on local disk for ingestion where possible; SQLite's small reads can be expensive over shared storage.

## 2. Convert raw data to NCore

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

- `nuplan_data_root`, `sensor_root`, `maps_root` — **required reader arguments**; database hierarchy, sensor files and maps respectively.
- `scenes` — **required**; list of `(log_name, token, start_us, end_us)` tuples. These select the exact data to reconstruct.
- `split` — **optional API argument**, default `nuplan_test`; explicitly select the split matching your source database.
- `NCoreBridge.prepare(scene, work_dir)` — **both arguments required**; writes the selected scene under `work_dir/clips/<token>/` and returns that clip directory.

The output contains a `pai_<token>.json` manifest and camera/lidar shards. Preserve its origin-offset sidecar. Native nuPlan lidar includes structured-return geometry as well as points used for supervision; retain the source sensor metadata when converting.

## 3. Generate auxiliary signals

Set your NGC API key and authenticate Docker before launching these helpers. They use the image references defined in `nurec_runner.py`.

```bash
export NGC_API_KEY="<ngc-api-key>" # Required by NRE helpers: your registry/runtime credential; keep its real value out of committed scripts.
printf '%s' "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

- `printf '%s' "$NGC_API_KEY"` — **required pipeline input**; supplies the key on stdin without putting it in Docker's command-line arguments.
- `login nvcr.io` — **required Docker subcommand and registry positional**; authenticates to NVIDIA's container registry.
- `-u '$oauthtoken'` — **required for this authentication method**; literal NGC token username, not a shell variable.
- `--password-stdin` — **required for this form**; reads the key from the pipe rather than prompting interactively.

```bash
export NEXUSSIM_NUREC_GPUS="<gpu-index-or-list>" # Optional, default all visible host GPUs: devices passed to the NRE Docker containers.
export CLIP_DIR="$WORK/ncore/$TOKEN/clips/$TOKEN" # Required here: NCore clip directory created above.
python - <<'PY'
import os
from pathlib import Path
from navsafe.gs3d_converter.nurec_runner import run_aux
run_aux(clip_dir=Path(os.environ["CLIP_DIR"]), work_dir=Path(os.environ["WORK"]))
PY
```

- `clip_dir` — **required API argument**; NCore manifest/shard directory. Auxiliary signals are written into the clip dataset.
- `work_dir` — **required API argument**; working directory supplied to the helper.
- `camera_ids`, `lidar_ids` — **optional**, default `None`; restrict sensors when explicitly supplied.
- `segmentation_backend` — **optional**, default `mask2former`; supplies semantic masks used by fitting.
- `ego_mask` — **optional**, default `True`; generates the ego occlusion mask.
- `depth_backend` — **optional**, default `none`; enable an available upstream backend only when monocular depth supervision is intended.
- `nurec_image` — **optional**; defaults to the helper's NRE tools image. Changing the image changes the external tool version.

## 4. Fit and export a reconstruction

```bash
export RECON_WORK="$WORK/recon/$TOKEN" # Required here: per-clip directory for training runs and exported assets.
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

- `run_train.dataset_dir` — **required**; NCore clip including the auxiliary signals.
- `run_train.work_dir` — **required**; parent of `output/<run-id>/` training products.
- `config_name` — **optional API argument**, omitted here to let the helper select its default static-separation configuration. An explicit override must itself define the intended static-object layers and match the sensor rig. Its iteration budget, cameras and Gaussian limits affect quality, GPU memory and runtime.
- `static_separation` — **optional**, default `False`, enabled here; creates separate static-object layers so parked/boxed objects can be edited instead of being fused into the background.
- `lidar_supervision` — **optional**, default `False`, enabled here; adds lidar geometry supervision and requires compatible lidar data/configuration.
- `aux_data` — **optional**, default `True`; expects auxiliary signals from the preceding step.
- `subsample` — **optional**, default `2`; image subsampling for fitting. Changing it changes training resolution and cost.
- `overrides` — **optional**, default `None`; list of additional Hydra configuration overrides. Record them with the reconstruction.
- `run_export.run_dir`, `run_export.work_dir` — **required**; trained run and matching work directory returned/used above.
- `dataset_dir` — **optional export argument**; source clip, needed when exporting NCore tracks.
- `export_usdz` — **optional**, default `True`; writes the artifact used for NuRec rendering in addition to other exported products.
- `export_ncore_tracks` — **optional**, default `False`; adds per-sensor pose exports when requested.

`static_separation` controls editability; sensor choice, resolution and fitting budget still need to match the intended scene. Do not assume that one training recipe or fixed budget is optimal for all source datasets.

## 5. Prepare the symbolic Arrow scene

Evaluation also needs ego state, actor tracks, routes and maps over the same window. If you have a mined NavSafe `seed.json`, the repository contains these preparation commands:

```bash
python -m navsafe.benchmark.world.run_ncore --seed "<seed-json-or-directory>" --nuplan-local "<local-db-staging-directory>"
```

- `--seed` — **required**; seed JSON or its directory, containing source paths, timestamps and output destinations.
- `--nuplan-local` — **optional**, default `NAVSAFE_LOCAL_DB` or `/tmp/nuplan_local`; stages the source SQLite database on local disk.
- `--force` — **optional**, off by default; recomputes an existing conversion instead of skipping its manifest.

For a selected raw test-split clip, build Arrow for the same token and time window with the bundled conversion helper:

```bash
export NAVSAFE_ROOT_REPO="<absolute-repository-directory>" # Required here: repository root added to the conversion process's Python path.
export NAVSAFE_CONVERT_PYTHON="$NAVSAFE_ROOT_REPO/.venv/bin/python" # Required here: use the installed NavSafe environment's Python and sibling py123d-conversion command.
export NUPLAN_DATA_ROOT="$NUPLAN_ROOT" # Required here: raw nuPlan database hierarchy read by the bundled converter configuration.
export NUPLAN_MAPS_ROOT="$NUPLAN_ROOT/nuplan-v1.1/maps" # Required here: directory holding the nuPlan map files; adjust to your raw dataset layout.
export NUPLAN_SENSOR_ROOT="$NUPLAN_ROOT/nuplan-v1.1/sensor_blobs" # Required here: raw camera/lidar sensor directory.
bash navsafe/benchmark/eval/make_arrow.sh "$TOKEN" "$NUPLAN_LOG" "$T0_US" "$T1_US" "<arrow-output-root>"
```

- Script path — **required positional**; installs the bundled `nuplan-navhard` conversion configuration into the active py123d package, then runs conversion.
- `TOKEN` — **required positional**; identifier for the output scene, matching the reconstructed clip.
- `NUPLAN_LOG` — **required positional**; test-split log database name without `.db`.
- `T0_US`, `T1_US` — **required positionals**; start and end timestamps in microseconds, matching the reconstruction window.
- `<arrow-output-root>` — **required positional**; writes `logs/nuplan_test/<token>/` and `maps/` here.

The environment needs the upstream raw-nuPlan parser/devkit dependencies and write access to its py123d configuration directory. This helper uses the test-split configuration; another source split requires an appropriate converter configuration. A USDZ alone is not a substitute for the Arrow scene and maps.

## 6. Serve and evaluate

```bash
export NUREC_PORT="<renderer-port>" # Required in this example: host port exposing the renderer's gRPC endpoint.
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

- `work_dir` — **required API argument**; root containing the reconstructed per-token work directories.
- `host_port` — **optional API argument**, default `8080`; mapped host gRPC port.
- `artifact_glob` — **optional**; glob relative to `work_dir`, with the shown pattern also being the default. It must match exported USDZ files and preserve distinct scene identities.
- `enable_harmonizer` — **optional**, default `True`; applies DiffusionHarmonizer. Turning it off changes images and rendering cost.
- `nurec_image` — **optional**; NRE renderer image pinned by the helper.

In another shell, activate the same NavSafe environment and use the [NuRec evaluation command](navsafe_eval.md#existing-renderer) with `--render-backend nurec_grpc`. Point the client to this service and use matching time windows and coordinate frames. Published multi-window bundles additionally use their handoff descriptor.
