# Kubernetes renderer deployment

Run the generator in the installed NavSafe environment. The mounted checkout and data must already exist on the specified PVCs.

## Environment

```bash
export NAVSAFE_NAMESPACE="<namespace>" # Required here: namespace that owns your workloads, PVCs and image-pull secret.
export NAVSAFE_DATA_ROOT="<absolute-dataset-mount>" # Required here: dataset path inside renderer and evaluator containers.
export NAVSAFE_ROOT="<absolute-checkout-mount>" # Required for code-running jobs: checkout path inside their containers.
export NAVSAFE_WORK="<absolute-work-mount>" # Required here: persistent work directory, including the renderer's harmonizer cache.
export NAVSAFE_PVC_MOUNTS='[{"claim":"<data-pvc>","mountPath":"<absolute-data-parent>"},{"claim":"<code-pvc>","mountPath":"<absolute-code-parent>"}]' # Required for this PVC workflow: JSON mapping of existing claims to container paths.
export NAVSAFE_IMAGE_PULL_SECRET="<image-pull-secret>" # Required when the registry needs authentication: existing Kubernetes Secret name, not its credential value.
export TOKEN="<scenario-token>" # Required here: scenario whose USDZ files should be served.
```

The data-parent mount must contain `NAVSAFE_DATA_ROOT`; the code-parent mount must contain `NAVSAFE_ROOT`. Add a work PVC to `NAVSAFE_PVC_MOUNTS` if `NAVSAFE_WORK` is not below either mount.

Optional placement controls:

```bash
export NAVSAFE_NODES="<node-a>,<node-b>" # Optional, default unrestricted: limits placement to these hostnames and may delay scheduling.
export NAVSAFE_TOLERATIONS='[{"key":"<reservation-key>","operator":"Equal","value":"<reservation-value>","effect":"NoSchedule"}]' # Optional, default none: allows your authorized reservation taint; it does not reserve a node.
```

## Generate a manifest

```bash
python -m navsafe.benchmark.world.serve_grpc \
  --name "<renderer-service-name>" \
  --artifact-glob "$NAVSAFE_DATA_ROOT/full_test/$TOKEN/*.usdz" \
  --out "<renderer-manifest.yaml>"
```

- `--name` — **optional**, default `navsafe-dev-nurec-grpc`; name of both resources and the service hostname used by clients.
- `--artifact-glob` — **required for this dataset layout**; USDZ glob inside the container. Quote it so the local shell does not expand it. Default is the work directory's export tree.
- `--out` — **optional**, default `-` (stdout); destination YAML file.
- `--renderer` — **optional**, default `default`; NRE renderer implementation.
- `--no-harmonizer` — **optional flag**, omitted by default; disables DiffusionHarmonizer postprocessing. Changes rendered images and may change policy behavior; record it with results.

The generated renderer requests one GPU, 4 CPUs, 24 GiB memory and 40 GiB ephemeral storage, and exposes gRPC port 8080. Adjust the generated manifest for your deployment's actual resource requirements.

## Validate and submit

```bash
kubectl apply --dry-run=server -n "$NAVSAFE_NAMESPACE" -f "<renderer-manifest.yaml>"
kubectl apply -n "$NAVSAFE_NAMESPACE" -f "<renderer-manifest.yaml>"
```

Connect the evaluator to `<renderer-service-name>:8080` using the [evaluation guide](../../docs/navsafe_eval.md#existing-renderer). Clients in another namespace need a namespace-qualified service hostname. Recreate or roll out the renderer when the served artifact set changes; it discovers scenes at startup.
