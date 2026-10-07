# Installation

## Requirements

| | |
| :--- | :--- |
| System | Linux x86-64 with an NVIDIA driver |
| CPU and memory | At least 4 cores and 32 GB of RAM for the evaluator; about 16 GB more for the renderer |
| GPUs | Up to 24 GB of VRAM for the renderer, plus what the simulator and policy need. Tested on two 24 GB RTX 3090s, one for each. |
| Disk | About 25 GB for the Python environment, 28 GB for the renderer image and 8 GB per scenario bundle (2.2 TB for all 280) |
| Software | Python 3.12 and [uv](https://docs.astral.sh/uv/); Docker with NVIDIA GPU support, or an existing NuRec gRPC renderer |

The renderer keeps a scenario's four reconstructions loaded, which takes about 18 GiB, and inserted assets, gait banks and harvested vehicles bring it to 24 GB or more ([measurements](navsafe_harvested_actors.md#renderer-vram)). On 24 GB GPUs it therefore needs a GPU to itself, with the simulator and policy on a second one. A single GPU with enough VRAM for both also works: give the renderer and the evaluator the same GPU index.

## Environment

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/NavSafe-infty.git
cd NavSafe-infty
uv sync --locked --python 3.12
source .venv/bin/activate
```

In an existing checkout, run `git submodule update --init --recursive` before `uv sync`. The locked environment contains CUDA PyTorch, Isaac Sim, Isaac Lab and the policy dependencies. Keep `.venv` on a local disk rather than a network filesystem.

Check the installation:

```bash
navsafe --help
navsafe eval --help
navsafe event types        # lists the event-type definitions
```

### System packages

A desktop Linux installation already has the libraries Isaac Sim loads. A minimal container image, such as `ubuntu:22.04`, needs them installed:

```bash
apt-get update && apt-get install -y git curl ca-certificates \
  libglu1-mesa libxt6 libgl1 libglx0 libegl1 libglib2.0-0 libxrandr2 libxinerama1 \
  libxcursor1 libxi6 libxext6 libxrender1 libx11-6 libxfixes3 libxdamage1 libsm6 \
  libice6 libgomp1
```

In a container, also set `NVIDIA_DRIVER_CAPABILITIES=all` so that the graphics libraries of the driver are available.

## Simulator and renderer

Isaac Sim starts unattended once you have accepted the NVIDIA terms and set:

```bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
```

Camera images come from an NVIDIA NuRec renderer, which runs in the NRE container image. Either:

- install Docker with NVIDIA GPU support and get an [NGC API key](https://org.ngc.nvidia.com/setup/api-key), and let the [evaluation scripts](navsafe_eval.md#local-docker-wrapper) start the renderer; or
- run the renderer yourself, for example as a [Kubernetes sidecar](../deploy/kubernetes/README.md), and [connect to it](navsafe_eval.md#existing-renderer).

The renderer and the evaluator must read the dataset at the same absolute paths.

Continue with [data download and evaluation](navsafe_eval.md). Building new actor assets additionally needs [Asset Harvester](navsafe_harvested_actors.md#install-asset-harvester).
