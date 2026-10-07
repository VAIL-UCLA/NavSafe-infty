# Policy evaluation

This guide downloads the data, starts the renderer and evaluates a policy, on one scenario or on the whole benchmark. Install the [environment](installation.md) first.

The NuRec renderer can use up to 24 GB of VRAM for one scenario, and the simulator and policy need their own. We evaluate on two 24 GB GPUs, one for each; a single larger GPU also works.

## Download the data

The dataset is [c13752hz/NavSafe](https://huggingface.co/datasets/c13752hz/NavSafe); its layout is described in [data layout](public-data-layout.md).

```bash
export NAVSAFE_DATA_ROOT="<absolute-dataset-directory>"
hf download c13752hz/NavSafe --repo-type dataset --local-dir "$NAVSAFE_DATA_ROOT" \
  --include "full_test/*" "asset/*" "gait_bank/*" "model_zoo/<model-directory>/*"
```

| Directory | Size | Contents |
| :--- | ---: | :--- |
| `full_test/` | 2.2 TB | The 280 scenario bundles, about 8 GB each. |
| `asset/`, `gait_bank/` | 0.2 GB | Actor assets and gait banks used by recipes. |
| `model_zoo/` | 178 GB | Policy weights, one directory per model. Download only the ones you evaluate. |
| `proxy/` | 218 GB | Optional subset used for authoring. Not needed for evaluation. |

Leave the downloaded `.ply` files and `bank.json` manifests unmodified, because recipes verify their hashes.

### One scenario

```bash
export TOKEN="<scenario-token>"
navsafe fetch --token "$TOKEN" --out "$NAVSAFE_DATA_ROOT"
```

`--token` can be repeated. The bundle is written to `full_test/<token>/` below `--out`. `--repo` selects another dataset repository with the same layout. Edited scenarios also need `asset/` and `gait_bank/`.

## Evaluate one scenario

### Local Docker wrapper

The wrapper starts a renderer container for the bundle, evaluates the policy and stops the container. It needs Docker with NVIDIA GPU support and an [NGC API key](https://org.ngc.nvidia.com/setup/api-key) for the renderer image.

```bash
export NGC_ENV_FILE="<absolute-ngc-env-file>"   # shell file that sets NGC_API_KEY
export NAVSAFE_RENDER_GPU=0 NAVSAFE_GPU=1       # renderer GPU, evaluator GPU (may be the same)
export NAVSAFE_ASSET_MOUNT="$NAVSAFE_DATA_ROOT" # lets the renderer read asset/ and gait_bank/
navsafe bundle-eval "$TOKEN" policy "<model-type>" "<checkpoint-path>"
```

The positional arguments are the token, the mode (`policy`, or `replay` to render the logged ego without a policy), the registered policy adapter and its checkpoint. Any further arguments are passed to the evaluator and override the wrapper's own.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `NAVSAFE_GPU` | `0` | GPU for the simulator and policy. |
| `NAVSAFE_RENDER_GPU` | `NAVSAFE_GPU` | GPU for the renderer container. Use a second GPU unless one GPU has enough VRAM for both. |
| `NAVSAFE_RENDERER` | `docker` | `existing` skips Docker and uses the renderer at `NUREC_GRPC_HOST:NUREC_GRPC_PORT`. |
| `NAVSAFE_OUT` | `output/bundle_<token>` | Output directory. Set it per model, seed and configuration. |
| `NAVSAFE_ASSET_MOUNT` | unset | Colon-separated directories bound into the renderer at the same path. Required for edited scenarios. |
| `NAVSAFE_HARMONIZER_CACHE` | unset | Directory for the harmonizer weights. Setting it enables the harmonizer. |
| `NAVSAFE_ENABLE_VIS` | `1` | `0` disables images and GIFs. |
| `NAVSAFE_VIS_CAMS` | unset | Extra cameras to render for the artifacts, for example `CAM_B0`. |
| `NAVSAFE_CONTROLLER` | `pure_pursuit` | Trajectory tracker: `pure_pursuit`, `lqr` or `pid`. |
| `NAVSAFE_ROUTE_TIME_LIMIT_S` | `0` | Route time budget in seconds; `0` disables it. |
| `NAVSAFE_EVAL_FRAMES` | unset | Cap on scored frames. |
| `NAVSAFE_REPLAN_RATE` | `5` | Simulation frames between policy calls. |
| `NAVSAFE_KEEP_ARTIFACTS` | `0` | `1` keeps the auxiliary metrics that later re-scoring needs. |
| `NRE_IMAGE` | `nvcr.io/nvidia/nre/nre-ga:26.04` | Renderer image. |

### Existing renderer

Without Docker, or to keep one renderer running across evaluations, start `serve-grpc` yourself and connect the evaluator to it. The renderer must serve the bundle's `.usdz` files with `--enable-editing-actors`, so that rendered traffic follows the simulation, and must see bundles, assets and gait banks at the same absolute paths as the evaluator.

```bash
export BUNDLE="$NAVSAFE_DATA_ROOT/full_test/$TOKEN"
export NUREC_GRPC_HOST="<renderer-host>"
export NUREC_GRPC_PORT="<renderer-port>"        # default 8080
export NUREC_GRPC_HANDOFF="$(navsafe bundle --handoff "$BUNDLE")"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES   # after accepting the NVIDIA terms

navsafe eval \
  --py123d-data-root "$BUNDLE/arrow" --render-backend nurec_grpc \
  --model-type "<model-type>" --checkpoint "<checkpoint-path>" \
  --traffic-mode semi_reactive --terminate-on-collision \
  --eval-seed "<seed>" --output-dir "<unique-output-directory>"
```

`NUREC_GRPC_HANDOFF` tells the renderer client which reconstruction covers which part of the scenario; recompute it for each bundle. The wrapper accepts the same setup through `NAVSAFE_RENDERER=existing`.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--py123d-data-root` | `data` | The bundle's `arrow/` directory. |
| `--py123d-scene-index` | `0` | Scene within the Arrow root. A bundle has one. |
| `--model-type` | `transfuser` | Registered policy adapter; it must match the checkpoint. See [policies](models.md). |
| `--checkpoint` | required | Checkpoint path, or the adapter's sentinel such as `none` for `pdm_closed`. |
| `--config` | unset | Model configuration, for adapters that need one. |
| `--traffic-mode` | `log_replay` | `semi_reactive` lets background vehicles react to the ego; `navsafe` runs recipe actors. |
| `--execution-mode` | `controller` | `controller` tracks the plan with a bicycle model, `physics` uses PhysX, `teleport` places the ego on the plan. |
| `--controller` | `lqr` | Tracker for `controller` and `physics`: `lqr`, `pure_pursuit` or `pid`. |
| `--ego-replay-frames` | `8` | Logged warm-up frames before the policy takes over. |
| `--replan-rate` | `5` | Simulation frames between policy calls: `5` is 2 Hz at the 10 Hz simulation rate. |
| `--eval-seed` | unset | Seeds Python, NumPy and Torch for policy sampling. |
| `--terminate-on-collision` | off | End the episode on any ego contact. The benchmark uses it. |
| `--eval-frames` | unset | Cap on scored frames. |
| `--route-time-limit-s` | `60` | Route time budget when no frame cap is set; `0` disables it. |
| `--camera-resolution-scale` | `0.5` | Scale of the policy's camera input. |
| `--enable-vis` | off | Write images and GIFs; renders every simulation step. |
| `--vis-cameras` | unset | Extra cameras for the artifacts; requires `--enable-vis`. |
| `--recipe`, `--recipe-dir` | unset | Apply a recipe; see [edited scenarios](navsafe_eval_edited_scene.md). |
| `--output-dir` | required | Directory for this run's metrics and artifacts. |

Renderer client variables:

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `NUREC_GRPC_TIMEOUT_S` | `300` | Deadline of one render request. |
| `NUREC_GRPC_CAM_RIG` | `recon` | `recon` renders with the reconstruction's calibrated cameras; `navsim` uses the policy's virtual camera geometry. |
| `NAVSAFE_NO_OVERLAY` | `0` | `1` removes the annotations drawn on camera images. |

## Full benchmark sweep

One command evaluates a policy on every bundle under `$NAVSAFE_DATA_ROOT/full_test`, edited and unedited. It needs the whole dataset: all 280 bundles, `asset/` and `gait_bank/`.

```bash
export NGC_ENV_FILE="<absolute-ngc-env-file>"   # shell file that sets NGC_API_KEY
export NAVSAFE_RENDER_GPU=0 NAVSAFE_GPU=1       # renderer GPU, evaluator GPU (may be the same)
navsafe benchmark "<model-type>" "<checkpoint-path>"
```

A scenario whose recipe inserts actors runs edited under `navsafe` traffic; every other scenario runs its logged scene. Arguments after the checkpoint are passed to every evaluation.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `NAVSAFE_OUT_ROOT` | `output/full_benchmark_<model-type>` | Run directory. Reuse it to resume. |
| `NAVSAFE_RENDERER` | `docker` | `docker` starts a renderer container per scenario. `existing` uses a running renderer; see below. |
| `NAVSAFE_RENDER_GPU`, `NAVSAFE_GPU` | `0`, `1` | Renderer GPU and evaluator GPU in `docker` mode. Set both to the same index to share one large GPU. |
| `NAVSAFE_SHARD` | `0/1` | `i/n` runs every n-th scenario starting at the i-th. Use it to split the sweep over n lanes, each with its own renderer and GPU pair, sharing one run directory. |
| `NAVSAFE_EDITS` | `auto` | `off` runs all scenarios unedited: a no-edit baseline, not the benchmark. |
| `NAVSAFE_TOKENS` | unset | File with one token per line, to run a subset. |
| `NAVSAFE_EVAL_SEED` | `0` | Policy seed. |
| `NAVSAFE_RETRIES` | `2` | Attempts per scenario until it is scored. |
| `NAVSAFE_ENABLE_VIS` | `0` | `1` writes images and GIFs for every scenario. |
| `NAVSAFE_FORCE` | `0` | `1` re-runs scenarios that are already scored. The previous result is moved to `scenarios/<token>.prev-<timestamp>`, not deleted. |
| `NAVSAFE_EPISODE_TIMEOUT` | `45m` | Stops a stuck scenario so the sweep continues. |
| `NAVSAFE_DRY_RUN` | `0` | `1` lists the scenarios and their recipe files and exits. |

Protocol: semi-reactive traffic, LQR tracking, a 120 s route budget (1200 frames), 8 warm-up frames, contact termination, seed 0, full camera resolution, visualization off and the harmonizer on. The protocol variables of the [wrapper](#local-docker-wrapper) override it; record any change with the results.

The run directory contains `scenarios/<token>/` (one evaluation each), `logs/`, `summary.tsv` (status, edited or not, and terminal reason per scenario) and the aggregate `report.{md,tsv,json}`. The command exits nonzero if a scenario of its shard is not scored.

### Sweep with an existing renderer

On Kubernetes, or any host without Docker, run one renderer that serves every bundle and point the sweep at it. The renderer keeps four reconstructions loaded and loads the others on demand.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
serve-grpc --host 0.0.0.0 --port 8080 --enable-editing-actors --renderer default \
  --cache-size 4 --enable-harmonizer --harmonizer-cache "<harmonizer-cache-directory>" \
  --artifact-glob "$NAVSAFE_DATA_ROOT/full_test/*/*.usdz"
```

```bash
export NAVSAFE_RENDERER=existing
export NUREC_GRPC_HOST="<renderer-host>" NUREC_GRPC_PORT="<renderer-port>"
navsafe benchmark "<model-type>" "<checkpoint-path>"
```

Run one sweep per renderer. Two evaluations on one renderer remove each other's inserted actors. The [Kubernetes guide](../deploy/kubernetes/README.md) shows this setup as a Job with the renderer as a sidecar.

## State-perturbation set

The 56 controlled events of the [state-perturbation set](../navsafe/benchmark/recipes/proxy_set_state_perturbation/README.md) have their own command, which takes the same renderer, shard and resume variables:

```bash
navsafe perturbation "<model-type>" "<checkpoint-path>"
```

## Results

Each evaluation writes `navsafe_metrics.json` with the episode's status, terminal reason and metrics. Aggregate only episodes with `status: scored`; report excluded and failed ones separately. Keep the checkpoint, recipe, seed and protocol settings with the results. [Scoring and termination](navsafe_termination_and_success.md) defines the terminal reasons and the success criterion.

`navsafe report --run "<run-directory>"` summarizes a sweep; `--baseline "<other-run>"` compares two runs on the scenarios both scored.

## Validate completed outputs

```bash
navsafe validate "<episode-output-directory>"
```

The check requires `status: scored`, a consistent scored frame window, a terminal reason and valid metrics. A driving score of zero or `success: false` is a valid outcome; an excluded episode fails the check. `--require-vis` also requires the front-camera and bird's-eye images and GIFs. `--model-dir "<sweep-output-directory>"` checks every episode found below a directory; it verifies the outputs that exist, not that every planned scenario ran.
