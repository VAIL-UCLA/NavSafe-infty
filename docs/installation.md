# Install NavSafe

Requirements: Linux x86-64, Python 3.12 and uv. Simulation and policy inference require NVIDIA hardware and a compatible driver. NuRec rendering runs in an NVIDIA NRE container, accessed through Docker or a deployed gRPC service.

## Checkout and environment

```bash
git clone --recurse-submodules https://github.com/VAIL-UCLA/NavSafe-infty.git "<repository-directory>"
cd "<repository-directory>"
uv sync --locked --python 3.12
source .venv/bin/activate
```

For an existing checkout, initialize its submodules before `uv sync`:

```bash
git submodule update --init --recursive
```

The locked environment includes CUDA PyTorch, IsaacSim, IsaacLab, model dependencies and visualization tools. Keep `.venv` on local disk where possible.

## Inspect available commands

```bash
navsafe --help
navsafe-eval --help
navsafe leaves
```

`navsafe leaves` lists the benchmark leaf definitions.

## Simulator and renderer setup

After accepting the NVIDIA terms for the installed simulator, set:

```bash
export ACCEPT_EULA=Y # Required for unattended IsaacSim startup after accepting its license.
export OMNI_KIT_ACCEPT_EULA=YES # Required for unattended Kit startup after accepting its license.
```

For local NuRec rendering, install Docker with NVIDIA GPU support and authenticate to NGC. Use the [evaluation guide](navsafe_eval.md#local-docker-wrapper) for the wrapper or [Kubernetes guide](../deploy/nautilus/README.md) for a service. Renderer and client must be able to read shared assets at identical absolute paths.

Continue with [data download and evaluation](navsafe_eval.md). New asset generation uses the separately maintained Asset Harvester tool; its setup is described in [asset harvesting](navsafe_harvested_actors.md#5-installing-asset-harvester).
