# Installation

## Requirements

| | |
| :--- | :--- |
| System | Linux x86-64 with an NVIDIA driver |
| CPU and memory | At least 4 cores and 32 GB of RAM for the evaluator; about 16 GB more for the renderer |
| GPUs | One 24 GB GPU for the renderer together with a policy such as DrivoR. Larger policies can need a second GPU for the simulator and the policy. |
| Disk | About 25 GB for the Python environment, 28 GB for the renderer image and 1.4 GB per scenario bundle (0.43 TB for all 280) |
| Software | Python 3.12 and [uv](https://docs.astral.sh/uv/); Docker with NVIDIA GPU support, or an existing NuRec gRPC renderer |

Each scenario has one reconstruction, and the renderer keeps up to `--cache-size` of them loaded. The renderer and the evaluator can share one GPU: give both the same GPU index. Larger policies, or a larger `--cache-size`, can need a second GPU.

## Environment

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/NavSafe-infty.git
cd NavSafe-infty
uv sync --locked --python 3.12
source .venv/bin/activate
```

In an existing checkout, run `git submodule update --init --recursive` before `uv sync`. The locked environment contains CUDA PyTorch and the policy dependencies. Keep `.venv` on a local disk rather than a network filesystem. The vision-language and world-model policies run in separate environments and fetch their upstream code as extra submodules; see [models that run outside the NavSafe environment](models.md#models-that-run-outside-the-navsafe-environment).

Check the installation:

```bash
navsafe --help
navsafe eval --help
navsafe event-types        # lists the event-type definitions
```

### System packages

A minimal container image, such as `ubuntu:22.04`, needs these libraries installed:

```bash
apt-get update && apt-get install -y git curl ca-certificates libgl1 libglib2.0-0 libgomp1
```

## Renderer

The simulation state is stepped in Python; no simulator is installed. Camera images come from an NVIDIA NuRec renderer, which runs in the NRE container image. Either:

- install Docker with NVIDIA GPU support and get an [NGC API key](https://org.ngc.nvidia.com/setup/api-key), and let the [evaluation scripts](navsafe_eval.md#local-docker-wrapper) start the renderer; or
- run the renderer yourself, for example as a [Kubernetes sidecar](../deploy/kubernetes/README.md), and [connect to it](navsafe_eval.md#existing-renderer).

The renderer and the evaluator must read the dataset at the same absolute paths.

Continue with [data download and evaluation](navsafe_eval.md). Building new actor assets additionally needs [Asset Harvester](navsafe_harvested_actors.md#install-asset-harvester).
