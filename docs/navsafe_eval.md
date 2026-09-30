# Evaluate the published NavSafe dataset

Install the [environment](installation.md). This guide runs an underlying scene; use the [edited-scenario guide](navsafe_eval_edited_scene.md) when a benchmark entry has a frozen recipe.

## Download a bundle

```bash
export NAVSAFE_DATA_ROOT="<absolute-dataset-directory>" # Required here: HF snapshot root containing full_test, asset, gait_bank and model_zoo.
export TOKEN="<scenario-token>" # Required here: the bundle token from the dataset.
python -m navsafe.benchmark.eval.fetch_bundle --token "$TOKEN" --out "$NAVSAFE_DATA_ROOT"
```

- `--token` — **required, repeatable**; downloads `full_test/<token>/` for each supplied token.
- `--out` — **required**; dataset root. The downloader preserves the `full_test/<token>/` layout.
- `--repo` — **optional**, default `c13752hz/NavSafe`; HF dataset repository.

## Download assets and a model

```bash
export MODEL_DIRECTORY="<model-directory>" # Required here: directory name under the dataset's model_zoo; it need not equal the adapter name.
export DATASET_REVISION="<dataset-revision>" # Required here: HF commit or revision to use for reproducible downloads.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="c13752hz/NavSafe",
    repo_type="dataset",
    revision=os.environ["DATASET_REVISION"],
    local_dir=os.environ["NAVSAFE_DATA_ROOT"],
    allow_patterns=["asset/**", "gait_bank/**", f"model_zoo/{os.environ['MODEL_DIRECTORY']}/**"],
)
PY
```

- `repo_id` — **required API argument**; dataset repository to download.
- `repo_type="dataset"` — **required here**; selects a dataset rather than a model repository.
- `revision` — **optional API argument, explicitly set here**; pins the dataset snapshot. Omitting it follows the repository's default revision.
- `local_dir` — **optional API argument, required for this layout**; materializes the snapshot at the dataset root rather than only in the Hub cache.
- `allow_patterns` — **optional API filter**; downloads global actor assets, gait banks and one model directory. Omit a pattern to skip that resource. Omitting the filter downloads the entire dataset.

For all 280 bundles, replace `allow_patterns` with `["full_test/**"]` in the same command. To pin a single bundle too, use `[f"full_test/{os.environ['TOKEN']}/**"]`. The `fetch_bundle` CLI above has no revision flag and follows the dataset's default revision.

Use an external checkpoint path for models absent from this dataset. Keep PLY files and `bank.json` unchanged: frozen recipes verify their hashes. See [data layout](public-data-layout.md) for resource-path overrides.

## Existing renderer

The service must serve the chosen bundle's USDZ files; edited scenes also require actor editing. Renderer and client must see assets at identical absolute paths. The following exports are consumed by the evaluator or its renderer client:

```bash
export BUNDLE="$NAVSAFE_DATA_ROOT/full_test/$TOKEN" # Required here: selected bundle; derived from the dataset root and token.
export NUREC_GRPC_HOST="<renderer-host>" # Required for a remote service; gRPC server hostname or IP (default nurec-grpc).
export NUREC_GRPC_PORT="<renderer-port>" # Optional: gRPC port; use the service's port (default 8080).
export NUREC_GRPC_HANDOFF="$(python -m navsafe.benchmark.eval.bundle --handoff "$BUNDLE")" # Required for bundled multi-clip rendering: resolves the bundle's handoff specification.
export ACCEPT_EULA=Y # Required for unattended simulator startup after accepting the NVIDIA license.
export OMNI_KIT_ACCEPT_EULA=YES # Required for unattended Kit startup after accepting the NVIDIA license.
```

- Embedded `python -m navsafe.benchmark.eval.bundle` — runs the bundle utility; **`--handoff <bundle-directory>` is required for this operation** and prints the handoff specification used by the renderer client.

```bash
navsafe-eval \
  --scenario-source py123d --py123d-data-root "$BUNDLE/arrow" \
  --py123d-scene-index 0 --render-backend nurec_grpc \
  --model-type "<model-type>" --checkpoint "<checkpoint-path>" \
  --traffic-mode semi_reactive --execution-mode controller --controller lqr \
  --ego-replay-frames "<warmup-frames>" --replan-rate "<replan-interval-frames>" \
  --eval-seed "<seed>" --terminate-on-collision \
  --output-dir "<unique-output-directory>"
```

- `--scenario-source` — **Optional**, default `py123d`; the only supported input format is py123d Arrow.
- `--py123d-data-root` — **Required for this workflow**, CLI default `data`; Arrow root for the selected bundle, not the HF snapshot root.
- `--py123d-scene-index` — **Optional**, default `0`; zero-based scene index within the Arrow root. A single-scenario bundle normally uses `0`.
- `--render-backend` — **Optional**, default `nurec_grpc`; requests images from the connected NuRec service.
- `--model-type` — **Required for this workflow**, CLI default `transfuser`; registered policy adapter name. Choose the adapter that matches the checkpoint.
- `--checkpoint` — **Required**; checkpoint path expected by the selected adapter. For a checkpoint-free adapter, pass its supported sentinel, such as `none` for `pdm_closed`.
- `--traffic-mode` — **Required for the intended protocol**, CLI default `log_replay`; `semi_reactive` lets background vehicles react to ego, while `navsafe` executes recipe actor controllers. This changes traffic behavior and scores.
- `--execution-mode` — **Optional**, default `controller`; controls how plans move ego. `controller` uses a tracker and bicycle model, `physics` uses PhysX, and `teleport` places ego along the plan and excludes meaningful comfort scoring.
- `--controller` — **Optional**, default `lqr`; trajectory tracker for controller/physics execution. Alternatives are `pure_pursuit` and `pid`; changing it changes tracking behavior.
- `--ego-replay-frames` — **Optional**, default `8`; logged-ego warm-up frames before policy takeover. Changing this moves the handoff point and evaluation starting state.
- `--replan-rate` — **Optional**, default `5`; simulation frames between policy calls, **not Hz**. At the default 10 Hz simulation rate, `5` means 2 Hz replanning. Set it for your model.
- `--eval-seed` — **Optional**, default unset; seeds Python, NumPy and Torch for policy sampling. It does not add noise to deterministic traffic or control.
- `--terminate-on-collision` — **Optional CLI flag, enabled for the NavSafe protocol**; ends the episode on any ego-box contact, including not-at-fault contact. Fault attribution still determines the penalty.
- `--output-dir` — **Required**; directory for this run's metrics and artifacts. Use a distinct directory per model, scenario, configuration and seed.

### Optional controls

These are additions to the full command above; none is required to run a policy:

- `--enable-vis` — off by default; writes images/GIFs and renders at 10 Hz instead of only at observation-consumption timesteps.
- `--vis-cameras "<camera-list>"` — default unset; comma-separated extra visualization cameras. Requires `--enable-vis`; adds rendering/storage without adding those views to policy inference.
- `--eval-frames "<scored-frame-cap>"` — default unset; bounds a diagnostic run. A truncated run is not automatically a valid full-episode score.
- `--route-time-limit-s "<seconds>"` — default `60`; route time budget when no frame cap is set. `0` disables the clock; semantic termination still applies.
- `--camera-resolution-scale "<scale>"` — default `0.5`; changes input resolution, cost and potentially policy behavior.
- `--config "<model-config-path>"` — default unset; additional configuration required by some adapters.
- `--recipe-dir "<recipe-directory>"` — default unset; selects recipes using bundle metadata. Explicit `--recipe` takes precedence. Use the directory containing the frozen benchmark recipes.

Optional environment overrides:

```bash
export NUREC_GRPC_TIMEOUT_S="<rpc-timeout-seconds>" # Optional, default 300: per-request deadline; increasing it tolerates slow/cold rendering without increasing render speed.
export NUREC_GRPC_CAM_RIG="<camera-rig>" # Optional, default recon: recon/native uses reconstruction calibration; navsim uses policy virtual-camera geometry and changes images.
export NEXUSSIM_NO_OVERLAY=1 # Optional, default 0: removes camera visualization annotations; does not disable rendering or the BEV view.
```

## Local Docker wrapper

The wrapper starts a renderer, evaluates one bundle and stops that renderer. Set your NGC credential file and use absolute data paths that Docker can mount:

```bash
export NGC_ENV_FILE="<absolute-ngc-env-file>" # Required unless NGC_API_KEY is already exported; shell file defining your NGC_API_KEY.
export NAVSAFE_GPU="<gpu-index>" # Optional, default 0: GPU exposed to the wrapper's renderer and evaluator.
export NAVSAFE_ASSET_MOUNT="$NAVSAFE_DATA_ROOT" # Required for inserted assets: bind-mounts this directory into the renderer at the same absolute path.
export NAVSAFE_ENABLE_VIS=0 # Optional, wrapper default 1: disables image/GIF generation and full-rate visualization rendering.
export NAVSAFE_OUT="<unique-output-directory>" # Optional but recommended: per-run output path; prevents different model runs sharing the wrapper's per-token default.
export NAVSAFE_KEEP_ARTIFACTS=1 # Optional, default 0: retains auxiliary metrics and run metadata for later re-scoring.
bash navsafe/benchmark/eval/run_bundle_eval.sh "$TOKEN" policy "<model-type>" \
  "<checkpoint-path>" --eval-seed "<seed>" --replan-rate "<replan-interval-frames>"
```

- Script path — **required positional**; launches the bundled Docker/evaluation wrapper from the checkout root.
- `TOKEN` — **required positional**; bundle directory name below the configured `full_test` root.
- `policy` — **optional mode positional**, default `policy`; runs closed-loop evaluation. `replay` follows logged ego instead.
- `<model-type>` — **explicitly set here**, wrapper default `drivor`; registered adapter name.
- `<checkpoint-path>` — **required unless `NAVSAFE_CHECKPOINT` is set**; policy checkpoint. Supply it as the fourth positional argument before additional evaluator flags.
- `--eval-seed` — **optional forwarded evaluator argument**; policy RNG seed, default unset.
- `--replan-rate` — **optional forwarded evaluator argument**; frames between policy calls, default `5`.
- Additional evaluator flags — **optional**, forwarded after the checkpoint. They override the wrapper's earlier values. The policy wrapper otherwise selects `semi_reactive` traffic, 8 replay frames, `pure_pursuit` control, contact termination and a disabled route time limit; use explicit overrides to match your protocol.

For separate renderer/policy GPUs or repeated scenarios, use an existing renderer and the direct evaluator command instead of starting a renderer for each run.

## Results and full sweeps

Inspect `navsafe_metrics.json`, including status, exclusion and terminal reason. Count only valid scored episodes in aggregates; record failed/excluded runs separately. For a full sweep, enumerate downloaded bundle manifests and resolve their recipes rather than assuming every scenario is unedited. Keep the dataset revision, checkpoint identity, recipe, seed, camera settings and execution settings with the results.

[GTRS-specific checkpoint settings](gtrs.md) are documented separately. Other adapters retain their upstream checkpoint formats.

## Validate completed outputs

```bash
python scripts/evaluator/validate_eval_output.py "<episode-output-directory>"
```

- `<episode-output-directory>` — **required positional**; directory containing `navsafe_metrics.json`.
- `--require-vis` — **optional**, disabled by default; additionally requires front-camera and BEV images and GIFs. Any images present are decoded and checked even without this flag.
- `--model-dir "<sweep-output-directory>"` — **optional alternative to the positional path**; recursively checks discovered episode outputs, including runs with evaluator artifacts but no final metrics. An empty scan fails. This checks existing outputs; it does not establish that every planned scenario or seed ran.

Validation requires `status=scored`, a nonempty consistent scored frame window, a termination reason, and valid NavSafe metrics. A zero driving score or `success=false` is a valid evaluation outcome. Excluded episodes fail completion validation; unavailable efficiency and comfort values may be `null`. Visualization is not required for ordinary runs, and no fixed image resolution or file size is imposed. The command returns a nonzero exit code if any episode fails validation.
