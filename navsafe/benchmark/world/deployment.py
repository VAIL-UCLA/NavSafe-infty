"""User-supplied Kubernetes configuration. No cluster-specific defaults."""
import json
import os

def nodes():
    return [n.strip() for n in os.environ.get("NAVSAFE_NODES", "").split(",") if n.strip()]

def configure(pod, selected_nodes=None):
    mounts = json.loads(os.environ.get("NAVSAFE_PVC_MOUNTS", "[]"))
    for i, mount in enumerate(mounts):
        name = f"data-{i}"
        claim, path = mount["claim"], mount["mountPath"]
        if not claim or not path.startswith("/"):
            raise ValueError("Each PVC mount needs claim and absolute mountPath")
        pod.setdefault("volumes", []).append({
            "name": name, "persistentVolumeClaim": {"claimName": claim}})
        for container in pod["containers"]:
            container.setdefault("volumeMounts", []).append({
                "name": name, "mountPath": path})
    selected = nodes() if selected_nodes is None else list(selected_nodes)
    if selected:
        pod["affinity"] = {"nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{"matchExpressions": [{
                    "key": "kubernetes.io/hostname", "operator": "In",
                    "values": selected}]}]}}}
    tolerations = json.loads(os.environ.get("NAVSAFE_TOLERATIONS", "[]"))
    if tolerations:
        pod["tolerations"] = tolerations
    secret = os.environ.get("NAVSAFE_IMAGE_PULL_SECRET")
    if secret:
        pod["imagePullSecrets"] = [{"name": secret}]
    # Explicit path settings are safe to propagate; do not copy arbitrary env.
    keys = ("NAVSAFE_DATA_ROOT", "NAVSAFE_ASSET_BANK", "NAVSAFE_GAIT_BANK",
            "NAVSAFE_BUNDLES", "NAVSAFE_MODEL_ZOO", "NAVSAFE_ROOT",
            "NAVSAFE_WORK", "NAVSAFE_RUNS", "NAVSAFE_CORPUS",
            "NAVSAFE_NAVHARD_CORPUS", "NAVSAFE_CHECKPOINT", "NUPLAN_ROOT")
    for container in pod["containers"]:
        env = container.setdefault("env", [])
        existing = {e["name"] for e in env}
        for key in keys:
            if key in os.environ and key not in existing:
                env.append({"name": key, "value": os.environ[key]})
    return pod
