<div align="center">
  <img src="assets/readme/navsafe-logo.png" alt="NavSafe logo" width="110">
  <h1>NavSafe-∞: Benchmarking Closed-Loop Driving Safety<br>in Photorealistic Environments</h1>
  <p>
    Yuxin Bao<sup>1,*</sup>, Hongwei Ruan<sup>2,*</sup>, Luobin Wang<sup>2,†</sup>, Seth Z. Zhao<sup>1,†</sup>, Ziyang Leng<sup>1</sup>,<br>
    Zihan Zhang<sup>2</sup>, Yu Zeng<sup>3</sup>, Rowan McAllister<sup>3</sup>, Henrik Christensen<sup>2</sup>, Bolei Zhou<sup>1</sup>
  </p>
  <p><sup>1</sup>UCLA &nbsp;·&nbsp; <sup>2</sup>UCSD &nbsp;·&nbsp; <sup>3</sup>Toyota Research Institute<br>
  <sub>* Equal contribution; order by last name &nbsp; † Corresponding authors</sub></p>
  <p>
    <a href="https://navsafe-vail.github.io/"><img src="assets/readme/website.svg" alt="Project website" height="20"></a>
    <a href="https://arxiv.org/abs/2609.26618"><img src="assets/readme/paper.svg" alt="Read the paper on arXiv" height="20"></a>
    <a href="https://huggingface.co/datasets/c13752hz/NavSafe"><img src="assets/readme/dataset.svg" alt="Dataset on Hugging Face" height="20"></a>
  </p>
  <p><a href="#overview">Overview</a> · <a href="#quick-start">Quick Start</a> · <a href="#documentation">Documentation</a> · <a href="https://navsafe-vail.github.io/#leaderboard">Leaderboard</a> · <a href="#citation">Citation</a></p>
</div>

<p align="center">
  <img src="assets/readme/teaser.png" alt="Open-loop trajectories lead to different closed-loop outcomes: animal and vehicle collisions, roadway departure, direction noncompliance, or reaching the goal." width="1000">
  <br>
  <em>Strong open-loop scores do not guarantee safe closed-loop behavior.</em>
</p>

## Overview

**NavSafe-∞** benchmarks closed-loop driving safety in photorealistic reconstructed environments. The benchmark contains **280 scenarios across 28 safety event types**, with event-specific success and failure criteria. Camera observations reflect the evolving ego and actor states, allowing evaluation of how a policy interacts, recovers, and handles compounding errors.

This repository provides the toolbox for scenario mining, reconstruction preparation, asset harvesting, recipe authoring and replay, policy evaluation, visualization, scoring, and result aggregation.

### What you can do

- **Evaluate driving policies in closed loop** with reconstructed camera observations, configurable camera rigs, and policy adapters.
- **Create and replay safety events** using editable actor assets, traffic behavior, and frozen scenario recipes.
- **Diagnose policy failures** with camera and bird’s-eye visualizations, per-run artifacts, and event-aware scoring.
- **Prepare new environments** through source-clip reconstruction and actor asset harvesting.

<p align="center">
  <img src="assets/readme/framework.png" alt="NavSafe closed-loop pipeline: event layout and actor states are rendered into camera inputs, the driving policy predicts a plan, and the environment updates before the next observation." width="1000">
  <br>
  <em>Reconstructed scenes and editable actors connect policy planning to simulation feedback.</em>
</p>

### Benchmark at a glance

| Safety category | Capability evaluated |
| :--- | :--- |
| **Traffic Crashes (TC)** | Respond to vehicle conflicts and collision hazards. |
| **Vulnerable Road User Crashes (VRUC)** | Interact safely with pedestrians and other vulnerable road users. |
| **Traffic Violations (TV)** | Follow traffic controls and event-specific rules. |
| **Traffic Incidents (TI)** | Handle work zones, obstructions, and unexpected conflicts. |

Evaluation reports **driving score (DS)**, **success rate (SR)**, **driving efficiency (DE)**, and **comfort**, together with category-level capability scores. See the [leaderboard](https://navsafe-vail.github.io/#leaderboard) for policy comparisons and the [scoring guide](docs/navsafe_termination_and_success.md) for termination and success semantics.

## Roadmap

NavSafe-∞ is under active development. We maintain the benchmark, fix reported issues, and keep extending it. The following is planned:

- [ ] **Higher rendering quality.** Improve the fidelity of the reconstructed scenes under large deviations from the logged trajectory. We follow progress in the vision and 3D Gaussian splatting communities closely and will keep incorporating new advances into the benchmark.
- [ ] **More data.** Extend the benchmark beyond the current 280 scenarios with additional scenarios and source datasets.
- [ ] **Automatic reasoning labeling.** Provide an automatic labeling pipeline that annotates scenarios with reasoning about the scene, to support the training and evaluation of reasoning-capable driving policies.

NavSafe-∞ is meant to be a shared resource for closed-loop driving safety, and we welcome contributions from the community. If you have questions about the project, please [open an issue](https://github.com/VAIL-UCLA/NavSafe-infty/issues). Your feedback directly shapes what we build next and helps move the field forward.

## Quick Start

### 1. Install

Use **Linux x86-64**, **Python 3.12**, [uv](https://docs.astral.sh/uv/), and NVIDIA GPUs with a compatible driver. The renderer can use up to 24 GB of VRAM for one scenario, and the simulator and policy need their own on top. We ran the benchmark on two 24 GB RTX 3090s, one for the renderer and one for the simulator and policy; a single GPU with enough VRAM for both also works.

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/NavSafe-infty.git
cd NavSafe-infty
uv sync --locked --python 3.12
source .venv/bin/activate
```

See the [installation guide](docs/installation.md) for simulator prerequisites, renderer setup, and existing checkouts.

### 2. Download the dataset

The dataset is on [Hugging Face](https://huggingface.co/datasets/c13752hz/NavSafe). Download the 280 scenarios with the shared actor assets and gait banks (about 2.2 TB), and the weights of the policy you want to evaluate:

```bash
export NAVSAFE_DATA_ROOT="<absolute-dataset-directory>"
hf download c13752hz/NavSafe --repo-type dataset --local-dir "$NAVSAFE_DATA_ROOT" \
  --include "full_test/*" "asset/*" "gait_bank/*" "model_zoo/<model-directory>/*"
```

Keep the directory tree as downloaded; `NAVSAFE_DATA_ROOT` always points at its root. See [data layout](docs/public-data-layout.md).

```text
<NAVSAFE_DATA_ROOT>/
├── full_test/<scenario-token>/    one bundle per scenario, about 8 GB each
├── asset/                         actor assets inserted by recipes
├── gait_bank/                     posed copies of walking actors
└── model_zoo/<model-directory>/   policy weights
```

<details>
<summary>Download a single scenario instead</summary>

Pick a token from the dataset's `full_test/` directory. `--token` can be repeated.

```bash
export TOKEN="<scenario-token>"
navsafe fetch --token "$TOKEN" --out "$NAVSAFE_DATA_ROOT"
hf download c13752hz/NavSafe --repo-type dataset --local-dir "$NAVSAFE_DATA_ROOT" \
  --include "asset/*" "gait_bank/*" "model_zoo/<model-directory>/*"
```

</details>

### 3. Run a policy

Follow the [evaluation guide](docs/navsafe_eval.md) to connect to an existing NuRec gRPC renderer or launch the local Docker wrapper, then run your policy with its matching adapter and checkpoint.

For benchmark entries with edits, use the [edited-scenario guide](docs/navsafe_eval_edited_scene.md) to apply the frozen recipe to its matching bundle. Preserve the recipe and evaluation settings when comparing policies.

```bash
# Inspect evaluator options in the installed environment.
navsafe eval --help

# Validate a completed episode’s metrics and artifacts.
navsafe validate "<episode-output-directory>"
```

The [results guide](docs/navsafe_eval.md#results) explains how to inspect `navsafe_metrics.json` and aggregate valid scored episodes.

### 4. Run the full benchmark

With the dataset downloaded, one command evaluates a policy on all 280 scenarios, edited and unedited:

```bash
export NGC_ENV_FILE="<absolute-ngc-env-file>"   # shell file defining NGC_API_KEY
export NAVSAFE_RENDER_GPU=0 NAVSAFE_GPU=1       # renderer GPU, evaluator GPU (may be the same)
navsafe benchmark "<model-type>" "<checkpoint-path>"
```

A scenario whose frozen recipe inserts actors runs edited; every other scenario runs its logged scene. Results go to `output/full_benchmark_<model-type>/` (`report.md`, `summary.tsv`, one directory per scenario), and rerunning the command resumes where it stopped. Without Docker, for example on [Kubernetes](deploy/kubernetes/README.md), point the same command at a running renderer with `NAVSAFE_RENDERER=existing`. See [full benchmark sweep](docs/navsafe_eval.md#full-benchmark-sweep) for the options, the protocol and running several lanes in parallel.

## Documentation

| Guide | What it covers |
| :--- | :--- |
| [Getting started](docs/navsafe_getting_started.md) | The path from installation to a scored episode. |
| [Installation](docs/installation.md) | Environment, simulator, and renderer prerequisites. |
| [Data layout](docs/public-data-layout.md) | Published bundles, shared assets, and resource paths. |
| [Policy evaluation](docs/navsafe_eval.md) | Downloads, renderer connection, inference, and output validation. |
| [Policies](docs/models.md) | Supported models, their checkpoints and options. |
| [Edited scenarios](docs/navsafe_eval_edited_scene.md) | Frozen recipes and matching scene bundles. |
| [Recipe editor](navsafe/benchmark/editor/README.md) | Authoring and inspecting scenario edits. |
| [Asset harvesting](docs/navsafe_harvested_actors.md) | Building and using actor replacement banks. |
| [Reconstruction](docs/reconstruct_navsim_nuplan.md) | Preparing and reconstructing new source clips. |
| [Scoring and termination](docs/navsafe_termination_and_success.md) | Episode status, termination reasons, and success criteria. |
| [Kubernetes deployment](deploy/kubernetes/README.md) | Running the renderer and the benchmark sweep on a cluster. |

## Policy Integration

Policy adapters live in [`navsafe/policy`](navsafe/policy), and the model code they load lives in [`navsafe/modelzoo`](navsafe/modelzoo). Pass the adapter name and the matching checkpoint to the evaluator; weights are downloaded from the dataset's `model_zoo/`. The [policies guide](docs/models.md) lists every supported model, its checkpoint and its options, and explains how to add your own.

## Citation

If NavSafe-∞ supports your research, please cite our [paper](https://arxiv.org/abs/2609.26618):

```bibtex
@article{bao2026navsafe,
  title={{NavSafe-$\infty$: Benchmarking Closed-Loop Driving Safety in Photorealistic Environments}},
  author={Bao, Yuxin and Ruan, Hongwei and Wang, Luobin and Zhao, Seth Z. and Leng, Ziyang
          and Zhang, Zihan and Zeng, Yu and McAllister, Rowan and Christensen, Henrik and Zhou, Bolei},
  journal={arXiv preprint arXiv:2609.26618},
  year={2026}
}
```

## License and Acknowledgements

This repository is licensed under [Apache 2.0](LICENSE). Third-party components, model weights, and datasets retain their respective licenses.

We thank the authors and maintainers of [Isaac Sim](https://github.com/isaac-sim/IsaacSim), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [NVIDIA NuRec](docs/reconstruct_navsim_nuplan.md), [py123d](https://github.com/kesai-labs/py123d), and the integrated driving policies for their tools and open research resources.
