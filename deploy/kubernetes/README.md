# Kubernetes deployment

NavSafe runs on a cluster in two ways: a long-lived renderer Service that evaluators connect to, or a Job that runs the benchmark sweep with its own renderer. Both need the dataset on a volume that the renderer and the evaluator mount at the same path, and a Secret for pulling the renderer image from `nvcr.io`.

## Renderer Service

A generator in the installed NavSafe environment writes a Deployment and a Service for the renderer. It reads the cluster settings from the environment:

| Variable | Meaning |
| :--- | :--- |
| `NAVSAFE_NAMESPACE` | Namespace of the workloads, volumes and Secrets. |
| `NAVSAFE_DATA_ROOT` | Dataset path inside the containers. |
| `NAVSAFE_WORK` | Persistent work directory; holds the harmonizer weights. |
| `NAVSAFE_PVC_MOUNTS` | JSON list of `{"claim": "<pvc>", "mountPath": "<absolute-path>"}`. The mounts must contain `NAVSAFE_DATA_ROOT` and `NAVSAFE_WORK`. |
| `NAVSAFE_IMAGE_PULL_SECRET` | Name of the image-pull Secret, if the registry needs one. |
| `NAVSAFE_NODES` | Optional comma-separated hostnames to restrict placement to. |
| `NAVSAFE_TOLERATIONS` | Optional JSON list of tolerations. |

```bash
navsafe serve-manifest \
  --name "<renderer-service-name>" \
  --artifact-glob "$NAVSAFE_DATA_ROOT/full_test/<scenario-token>/*.usdz" \
  --out "<renderer-manifest.yaml>"
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--name` | `navsafe-dev-nurec-grpc` | Name of the Deployment and Service, and the hostname clients use. |
| `--artifact-glob` | the work directory's export tree | `.usdz` files to serve, as a path inside the container. Quote it so the local shell does not expand it. |
| `--out` | stdout | Destination file. |
| `--renderer` | `default` | NRE renderer implementation. |
| `--no-harmonizer` | off | Disable harmonizer post-processing. This changes the images; record it with the results. |

The generated renderer requests one GPU, 4 CPUs, 24 GiB of memory and 40 GiB of ephemeral storage, and listens on port 8080. Adjust the manifest to your cluster. The renderer lists its scenes at start-up, so restart it when the served files change.

## Full benchmark as a Job

The benchmark sweep needs a renderer and an evaluator, each with its own GPU. One way to run it on Kubernetes is a Job whose Pod has two containers: the renderer as a sidecar and the evaluator as the main container. They share the Pod's network, so the evaluator reaches the renderer at `127.0.0.1:8080`, and both mount the dataset at the same path.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: <job-name>
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets: [{name: <ngc-image-pull-secret>}]
      initContainers:
        - name: renderer            # sidecar: restarts on failure, stops with the Job
          restartPolicy: Always
          image: nvcr.io/nvidia/nre/nre-ga:26.04
          command: ["/opt/nvidia/nvidia_entrypoint.sh", "bash", "-c"]
          args:
            - |
              exec /app/run serve-grpc --host 0.0.0.0 --port 8080 \
                --enable-editing-actors --renderer default --cache-size 4 \
                --enable-harmonizer --harmonizer-cache <harmonizer-cache-directory> \
                --artifact-glob "<dataset-mount>/full_test/*/*.usdz"
          env:
            - name: NGC_API_KEY
              valueFrom: {secretKeyRef: {name: <ngc-key-secret>, key: NGC_API_KEY}}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
          resources:
            limits: {nvidia.com/gpu: "1", cpu: "4", memory: 24Gi, ephemeral-storage: 40Gi}
          volumeMounts:
            - {name: data, mountPath: <dataset-mount-parent>}
            - {name: dshm, mountPath: /dev/shm}
      containers:
        - name: eval
          image: <image-with-navsafe-installed>
          command: ["bash", "-c"]
          args:
            - |
              export NAVSAFE_DATA_ROOT=<dataset-mount> ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
              export NAVSAFE_RENDERER=existing NUREC_GRPC_HOST=127.0.0.1 NUREC_GRPC_PORT=8080
              export NAVSAFE_OUT_ROOT=<run-directory-on-a-volume> NAVSAFE_SHARD=<i>/<n>
              navsafe benchmark "<model-type>" "<checkpoint-path>"
          env:
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: all}
          resources:
            limits: {nvidia.com/gpu: "1", cpu: "8", memory: 48Gi, ephemeral-storage: 150Gi}
          volumeMounts:
            - {name: data, mountPath: <dataset-mount-parent>}
            - {name: dshm, mountPath: /dev/shm}
      volumes:
        - name: data
          persistentVolumeClaim: {claimName: <data-pvc>}
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 16Gi}
```

The renderer takes several minutes to index all bundles before it accepts connections; the sweep waits for it. To split the sweep over several Jobs, give each a different `NAVSAFE_SHARD` and the same `NAVSAFE_OUT_ROOT`. Write the run directory to a volume, since the Pod's own storage is removed with it. The example gives each container its own GPU, which is what 24 GB GPUs require. Give the evaluator at least 4 CPU cores: Isaac Sim needs them while it starts, and with fewer an episode may not begin within the sweep's per-scenario timeout.

The evaluator image can be a plain Ubuntu image that [installs NavSafe](../../docs/installation.md) when the container starts; see the system packages listed there.

## Validate and submit

```bash
kubectl apply --dry-run=server -n "$NAVSAFE_NAMESPACE" -f "<manifest.yaml>"
kubectl apply -n "$NAVSAFE_NAMESPACE" -f "<manifest.yaml>"
```

Connect an evaluator to a renderer Service at `<renderer-service-name>:8080` as described in [existing renderer](../../docs/navsafe_eval.md#existing-renderer). A client in another namespace needs the namespace-qualified hostname.
