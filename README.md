# NavSafe-infty

NavSafe benchmarks closed-loop safety of driving policies in photorealistic reconstructed environments. It includes scenario mining, reconstruction preparation, asset harvesting, recipe authoring and replay, policy evaluation, visualization, scoring, and result aggregation.

## Install

Use Linux x86-64, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/NavSafe-infty.git "<repository-directory>"
cd "<repository-directory>"
uv sync --locked --python 3.12
source .venv/bin/activate
```

See [installation details](docs/installation.md) for renderer prerequisites.

## Download data

The published dataset is [c13752hz/NavSafe](https://huggingface.co/datasets/c13752hz/NavSafe).

```bash
export NAVSAFE_DATA_ROOT="<absolute-dataset-directory>" # Required here: destination root for the published dataset layout.
export TOKEN="<scenario-token>" # Required here: token of the scenario bundle to download.
python -m navsafe.benchmark.eval.fetch_bundle --token "$TOKEN" --out "$NAVSAFE_DATA_ROOT"
```

- `--token` — **required, repeatable**; chooses a scenario token. Repeat it to download several bundles.
- `--out` — **required**; dataset root, not the individual scenario directory. Files go under `full_test/<token>/`.
- `--repo` — **optional**, default `c13752hz/NavSafe`; selects another dataset repository with the same bundle layout.

```text
<NAVSAFE_DATA_ROOT>/
├── full_test/<scenario-token>/{manifest.json,arrow/,offsets/,ah_assets/,*.usdz}
├── asset/
├── gait_bank/
├── model_zoo/<model-directory>/
└── proxy/                         # Optional authoring subset
```

## Evaluate

Follow the [evaluation guide](docs/navsafe_eval.md) to download assets and model weights, start or connect to the NuRec gRPC renderer, and run a policy.

- [Edited scenarios](docs/navsafe_eval_edited_scene.md): apply a frozen recipe to its matching bundle.
- [Asset harvesting](docs/navsafe_harvested_actors.md): build and use actor replacement banks.
- [Reconstruction](docs/reconstruct_navsim_nuplan.md): prepare and reconstruct new source clips.
- [Kubernetes deployment](deploy/nautilus/README.md): generate a renderer manifest for your cluster.

## Models

Adapters are under `navsafe/policy`; inference components are under `navsafe/modelzoo`. Pass your adapter name and checkpoint path to evaluation. Model weights are downloaded separately.
