"""Mode A renderer service: a long-lived ``serve-grpc`` Deployment + Service.

The doc's split-service deployment: one NRE process holds the exported
reconstructions in VRAM and answers render requests over the network, so eval
clients (which need their own GPU for the policy) stay independent of it.

A scene is keyed by its clip id, so one serve can hold several *distinct*
seeds at once -- but never two artifacts with the same id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg
from navsafe.benchmark.world import deployment


NRE_IMAGE = ("nvcr.io/nvidia/nre/nre-ga@sha256:"
             "6e0caa70a9148490552520c3dde9ee665c8d094ca10b814601cb8bb306567c90")
NODES = deployment.nodes()
GRPC_PORT = 8080


def manifest(name: str, artifact_glob: str, *, renderer="default", gpus=1,
             cpu="4", mem="24Gi", nodes=None,
             enable_harmonizer: bool = True) -> list[dict]:
    # Harmonizer postproc is our standard config (better perceptual quality).
    # Checkpoint cache on the PVC so a Deployment restart re-uses the
    # HuggingFace download instead of repeating it.
    harmonizer_setup = f"mkdir -p {cfg.HARMONIZER_CACHE}\n" if enable_harmonizer else ""
    harmonizer_flags = (
        f"  --enable-harmonizer --harmonizer-cache {cfg.HARMONIZER_CACHE} \\\n"
        if enable_harmonizer else "")
    script = f"""set -ex
echo "artifacts:"; ls -la {artifact_glob} || {{ echo "NO ARTIFACTS MATCHED"; exit 1; }}
{harmonizer_setup}exec /app/run serve-grpc --host 0.0.0.0 --enable-editing-actors \\
{harmonizer_flags}  --renderer {renderer} --artifact-glob "{artifact_glob}"
"""
    pod = {
        "restartPolicy": "Always",
        "containers": [{
            "name": "serve",
            "image": NRE_IMAGE,
            "imagePullPolicy": "IfNotPresent",
            "command": ["/opt/nvidia/nvidia_entrypoint.sh", "bash", "-c"],
            "args": [script],
            "ports": [{"containerPort": GRPC_PORT, "name": "grpc"}],
            "resources": {
                "limits": {"nvidia.com/gpu": str(gpus), "cpu": cpu, "memory": mem,
                           "ephemeral-storage": "40Gi"},
                "requests": {"nvidia.com/gpu": str(gpus), "cpu": cpu, "memory": mem,
                             "ephemeral-storage": "40Gi"},
            },
            "volumeMounts": [
                {"name": "dshm", "mountPath": "/dev/shm"},
            ],
            # The renderer is ready once it accepts a TCP connection; until then
            # the Service must not route eval traffic to it.
            "readinessProbe": {"tcpSocket": {"port": GRPC_PORT},
                               "initialDelaySeconds": 30, "periodSeconds": 10,
                               "failureThreshold": 60},
        }],
        "volumes": [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
        ],
    }
    deployment.configure(pod, nodes)
    deploy = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": cfg.NAMESPACE,
                     "labels": {"app": "navsafe-dev", "k8s-app": name}},
        "spec": {"replicas": 1,
                 "selector": {"matchLabels": {"k8s-app": name}},
                 "template": {"metadata": {"labels": {"k8s-app": name,
                                                      "app": "navsafe-dev"}},
                              "spec": pod}},
    }
    svc = {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": cfg.NAMESPACE,
                     "labels": {"app": "navsafe-dev"}},
        "spec": {"selector": {"k8s-app": name},
                 "ports": [{"name": "grpc", "port": GRPC_PORT,
                            "targetPort": GRPC_PORT, "protocol": "TCP"}]},
    }
    return [deploy, svc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="navsafe-dev-nurec-grpc")
    ap.add_argument("--artifact-glob",
                    default=str(cfg.EXPORT / "*/usd-out/*.usdz"))
    ap.add_argument("--renderer", default="default")
    ap.add_argument("--no-harmonizer", action="store_true",
                    help="Ablation: serve without DiffusionHarmonizer "
                         "postprocessing (default is harmonizer ON).")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    import yaml

    docs = manifest(args.name, args.artifact_glob, renderer=args.renderer,
                    enable_harmonizer=not args.no_harmonizer)
    text = "---\n".join(yaml.safe_dump(d, sort_keys=False, width=100_000) for d in docs)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}  (deployment+service {args.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
