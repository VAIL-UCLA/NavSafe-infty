# Harvested actor assets

A scenario's reconstruction is fit to what the logged ego saw. When a policy drives differently, it can meet a logged vehicle from a direction the log never observed, and the reconstruction has nothing to draw there: the vehicle smears or disappears. Oncoming traffic and vehicles near a boundary between two reconstruction windows are affected most.

[NVIDIA Asset Harvester](https://github.com/NVIDIA/asset-harvester) lifts one actor's observations into a view-consistent 3D Gaussian asset. NavSafe can ask the renderer to draw a logged track from such an asset instead of from the scene reconstruction, so the actor renders from any direction.

Replacement changes appearance only. The track keeps its id, its box and its simulated pose, so the scenario's geometry and scoring are unchanged. Different images can still change a vision policy's decisions, so record whether a run used it.

This guide covers [using a published bank](#evaluate-with-a-bank), [building one](#build-a-bank) and [installing Asset Harvester](#install-asset-harvester).

## When replacement helps

A harvested asset is generated from a few image crops; a reconstructed actor is fit to real pixels. Where the reconstruction has coverage it looks better than the asset: a truck whose lettering is legible in the reconstruction comes back as a plain box truck. Replacement pays off only where the reconstruction has no coverage, such as a logged car met at close range from a new direction, which otherwise does not render at all.

For that reason replacement is an option of an evaluation, not a default, and a bank holds only vehicles that moved. A parked car is passed by the ego and seen over a wide arc, so its reconstruction is already good. A moving car keeps a nearly constant bearing to the ego and is the one that breaks.

## Evaluate with a bank

Every published bundle carries a bank in `full_test/<token>/ah_assets/`. Add one option to the evaluation:

```bash
navsafe bundle-eval "$TOKEN" policy "<model-type>" "<checkpoint-path>" \
  --asset-harvester-replace
```

`--asset-harvester-replace` without a value uses the bank beside the bundle's `arrow/` directory; a manifest path can follow it. It works with the `nurec_grpc` render backend, because the renderer performs the swap.

The renderer opens the asset files itself, so it must see the bank at the same absolute path as the evaluator. The Docker wrapper binds `<bundle>/ah_assets` for this. If the renderer cannot open the bank, or none of the bank's tracks exist in the served scenes, the evaluation fails instead of rendering the original actors.

The number of replaced tracks is capped by `NUREC_GRPC_ASSET_REPLACE_MAX` (default 10), nearest to the ego first.

### Renderer VRAM

A replaced actor's asset is held once for every reconstruction window that contains the track, in addition to the four reconstructions and the harmonizer. Measured on one 24 GB GPU with one scenario's four windows:

| State | VRAM used | Free |
| :--- | ---: | ---: |
| Idle (CUDA context and harmonizer) | 9.8 GiB | |
| Four reconstructions loaded | 18.4 GiB | 6.2 GiB |
| + 8 nearest tracks replaced (29 instances) | 21.4 GiB | 3.1 GiB |
| + 16 nearest (54 instances) | 23.5 GiB | 1.0 GiB |
| + all 26 (75 instances) | 24.1 GiB | 0.5 GiB |

That is about 76 MiB per replaced instance. One camera render needs over 1 GiB of transient memory, so about 20 replaced tracks is the practical limit on a 24 GB GPU.

Run the renderer with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without it, memory fragmentation makes the renderer fail after about 11 replaced tracks even though memory is free.

### Recipes that require a bank

A recipe can move a logged vehicle onto a new path while keeping its identity (`op: relocate` with `keep_appearance: true`). If its controller sets `require_harvested_asset: true`, the evaluator refuses to run without `--asset-harvester-replace`, always replaces that vehicle regardless of the cap, and writes `harvester_takeover_audit.json` to the output directory, listing the tracks replaced in each window. The [state-perturbation set](../navsafe/benchmark/recipes/proxy_set_state_perturbation/README.md) uses this.

## Build a bank

Building a bank needs the scenario's source clips in NCore format and its reconstructions, as produced by the [reconstruction guide](reconstruct_navsim_nuplan.md); the published Arrow bundle alone is not enough. It also needs a GPU with 16 GB (or `--offload`) and an [Asset Harvester installation](#install-asset-harvester).

```bash
export NAVSAFE_CORPUS="<absolute-source-corpus>"            # per-scene clips and reconstructions
export NAVSAFE_AH_HOME="<absolute-asset-harvester-directory>"
navsafe harvest harvest "<scene-id>"
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--max-assets` | `10` | Number of tracks to harvest, nearest to the ego first. |
| `--min-motion-m` | `2` | Net displacement a track needs to count as moving. |
| `--include-parked` | off | Also harvest stationary vehicles. |
| `--max-aspect-error` | `0.35` | Tolerance of the shape check described below. |
| `--orient-degrees` | `90` | Rotation applied to bring assets into the renderer's convention. |
| `--offload` | off | Lower GPU memory use at the cost of speed. |
| `--keep-parse` | off | Keep the intermediate crops. |
| `--force` | off | Redo work that is already on disk. |

For a 20 s scenario the four 5 s windows are found and processed together. The output is written into the scenario's directory:

```text
<corpus>/<scene-id>/ah_assets/
├── lifted/<class>/<track-id>/gaussians.ply   one asset per track
├── lifted/metadata.yaml                      Asset Harvester's own description
├── replace_manifest.json                     read by the evaluator
└── harvest.log                               why a track produced no asset
```

`replace_manifest.json` stores paths relative to itself, so a bank can be moved or published as a directory.

### Steps

1. **Parse** each window's clip into per-track image crops and masks.
2. **Measure motion** from the clip's boxes, as net displacement from first to last observation. Summing per-frame steps would count annotation jitter as travel.
3. **Select** the moving vehicles closest to the ego, one entry per track. A vehicle that appears in several windows is harvested once, from the window that saw it closest.
4. **Lift** each track with Asset Harvester's multi-view diffusion and Gaussian reconstruction. This is the expensive step, about 40 s per asset on an RTX 3090.
5. **Orient** the asset into the renderer's axis convention.
6. **Check shape**: discard an asset whose proportions disagree with the track's box. The renderer scales an asset onto the box, so a wrong shape would render as a stretched vehicle. Rejected assets stay on disk but are left out of the manifest.

Pedestrians and cyclists are not harvested: a rigid asset cannot walk. Animated actors use the gait banks instead.

A scenario costs about 20 minutes and 190 MB. The yield depends on the scenario: only vehicles that the cameras saw well enough to crop from at least two views can be lifted, so a quiet street may give one asset and a busy one more than twenty. A re-run skips windows and tracks that are already done.

### Other commands

```bash
navsafe harvest status ["<scene-id>"]
navsafe harvest verify "<scene-id>"
navsafe harvest reorient "<scene-id>" --degrees "<rotation-degrees>"
navsafe harvest scenarios [--all]
navsafe harvest batch --scenes-file "<scene-list.txt>" --workers "<worker-count>" \
  --image "<harvester-container-image>" --out "<harvest-manifest.yaml>"
navsafe harvest pack --repo "<dataset-repository>" --prefix "<dataset-prefix>" --strict
```

| Command | Purpose |
| :--- | :--- |
| `status` | List existing banks and their asset counts. |
| `verify` | Compare a bank's track ids with those of a running renderer, per window. Some tracks matching no window is normal; none matching anywhere means the bank belongs to a different reconstruction. Needs `NUREC_GRPC_HOST`. |
| `reorient` | Rotate an existing bank about the vertical axis without lifting again. |
| `scenarios` | List scenarios that can be harvested and have no bank yet; `--all` includes those that have one. |
| `batch` | Write a Kubernetes Job that harvests many scenarios, one worker per GPU. It does not submit the Job; configure it as in the [Kubernetes guide](../deploy/kubernetes/README.md). |
| `pack` | Validate banks and write an index and an upload plan for publishing them. It does not upload. |

## Install Asset Harvester

Asset Harvester is a separate tool with its own Conda environment, which cannot share the NavSafe environment. Its checkpoints need about 12 GB.

```bash
export NAVSAFE_AH_HOME="<absolute-asset-harvester-directory>"
export CONDA_ENVS_DIRS="$NAVSAFE_AH_HOME/conda/envs"
export CONDA_PKGS_DIRS="$NAVSAFE_AH_HOME/conda/pkgs"
git clone https://github.com/NVIDIA/asset-harvester "$NAVSAFE_AH_HOME/repo"
cd "$NAVSAFE_AH_HOME/repo"
bash setup.sh
hf download nvidia/asset-harvester --local-dir checkpoints
```

The model is gated: accept its terms on Hugging Face and set `HF_TOKEN` before downloading. `harvest` checks the checkpoints and the interpreter before it starts.

NavSafe reads camera names from each clip's own manifest and matches object classes by substring, because the nuPlan-derived clips use different names from Asset Harvester's defaults.

NVIDIA advises reconstructing a scene with PPISP disabled if assets will be inserted into it; otherwise the assets look over-saturated against the scene.
