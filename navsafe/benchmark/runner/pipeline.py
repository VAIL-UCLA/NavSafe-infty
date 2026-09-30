"""Drive NavSafe seeds through the world-building stages.

    ncore -> arrow           CPU, inside the work pod (needs py123d + devkit)
    aux -> train -> export   Kubernetes Jobs in the NRE containers (Mode A)

The two halves live on opposite sides of a permission boundary: the pod holds
the data and the Python environment but its service account may not create
Jobs, while the operator's kubeconfig may create Jobs but cannot see the PVC.
So this runs *outside* the pod and reaches into it with ``kubectl exec`` for
filesystem state and CPU stages, and talks to the API server directly for the
GPU stages.

Every stage is idempotent and judged by its output on disk, not by job history,
so an interrupted run resumes by re-invoking the same command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from navsafe.benchmark import config as cfg
from navsafe.errors import NexusSimError


STAGE_ORDER = ["ncore", "arrow", "aux", "train", "export", "eval"]
K8S_STAGES = {"aux", "train", "export", "eval"}
# Eval is a Job like the other GPU stages, with its own serve-grpc co-located
# on the same GPU. An earlier version ran it in the work pod, which was wrong on
# three counts: the pod is reclaimed on a 6 h deadline, there is only one of it
# so evals could not run in parallel, and a shared renderer would have to swap
# artifact globs between scenes. serve_grpc.py remains for interactive use.
WORLD = f"{cfg.NAVSAFE_ROOT}/navsafe/benchmark/world"


class PodUnreachable(NexusSimError, RuntimeError):
    """The pod could not be reached, so nothing can be concluded about disk."""


class Pod:
    """Thin ``kubectl exec`` wrapper for state queries and CPU stages."""

    def __init__(self, name: str, namespace: str):
        self.name, self.ns = name, namespace

    def sh(self, script: str, timeout: int | None = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", "exec", self.name, "-n", self.ns, "--", "bash", "-lc", script],
            capture_output=True, text=True, timeout=timeout,
        )

    def exists(self, path: str, glob: bool = False) -> bool:
        """True/False only when the check actually ran.

        A pod that has gone away (Nautilus reclaims these on a 6 h deadline)
        makes every ``kubectl exec`` fail.  Reading that as "the artifact is
        missing" would make the orchestrator conclude that finished work never
        happened and resubmit GPU stages over good outputs, so an unreachable
        pod raises instead of answering.
        """
        q = f"ls -d {path} >/dev/null 2>&1" if glob else f"test -e {path!r}"
        r = self.sh(f"{q} && echo Y || echo N")
        out = r.stdout.strip()
        if out.endswith(("Y", "N")):
            return out.endswith("Y")
        raise PodUnreachable(
            f"cannot query {self.name}: {(r.stderr or r.stdout).strip()[:200]}")

    def cat(self, path: str) -> str:
        return self.sh(f"cat {path!r}").stdout

    def start_stage(self, stage: str, seed_dir: str, log: str, extra: str = "") -> None:
        script = {"ncore": "run_ncore.py", "arrow": "run_arrow.py"}[stage]
        self.sh(f"cd {cfg.NAVSAFE_ROOT} && nohup python3 {WORLD}/{script} "
                f"--seed {seed_dir} {extra} > {log} 2>&1 & echo started")

    def stage_running(self, stage: str, seed_id: str) -> bool:
        script = {"ncore": "run_ncore.py", "arrow": "run_arrow.py"}[stage]
        out = self.sh(f"pgrep -fa '{script} --seed .*{seed_id}' | grep -v pgrep | wc -l")
        return out.stdout.strip().isdigit() and int(out.stdout.strip()) > 0


# --- completion predicates: what "done" means on disk ----------------------

def done_ncore(pod, seed) -> bool:
    tok = seed["seed_id"]
    # The re-reference sidecar, not the manifest: a store built without
    # NCORE_REREF_FRAME0 is silently jittery and must not count as done.
    return pod.exists(f"{seed['artifacts']['ncore']}/clips/{tok}/nurec_origin_offset.json")


def done_arrow(pod, seed) -> bool:
    return (pod.exists(f"{seed['artifacts']['arrow_log']}/*.arrow", glob=True)
            and pod.exists(f"{seed['artifacts']['arrow']}/maps/*/*.arrow", glob=True))


# ncore-aux-data writes its shards incrementally, so "some .aux.* exists" is
# true long before the stage is finished -- a glob test there will start
# training on half-generated masks.  A finished run has exactly these five,
# and the meta json is written last.
AUX_OUTPUTS = ("aux-meta.json", "aux.egomask.zarr.itar", "aux.lidar-camvis.zarr.itar",
               "aux.lidar-sseg.zarr.itar", "aux.sseg.zarr.itar")


def done_aux(pod, seed) -> bool:
    tok = seed["seed_id"]
    clip = f"{seed['artifacts']['ncore']}/clips/{tok}"
    return all(pod.exists(f"{clip}/{tok}.{suffix}") for suffix in AUX_OUTPUTS)


def done_train(pod, seed) -> bool:
    return pod.exists(f"{seed['artifacts']['recon']}/{seed['seed_id']}/checkpoints/last.ckpt")


def done_export(pod, seed) -> bool:
    return pod.exists(f"{seed['artifacts']['export']}/usd-out/*.usdz", glob=True)


def done_eval(pod, seed, regime: str = "log_replay") -> bool:
    # run_eval.py copies the nested metrics.json up to the regime root, so the
    # top-level path is the one to trust; the glob covers runs from before that.
    root = f"{seed['artifacts']['eval']}/{regime}"
    return pod.exists(f"{root}/metrics.json") or pod.exists(f"{root}/*/metrics.json", glob=True)


DONE = {"ncore": done_ncore, "arrow": done_arrow, "aux": done_aux,
        "train": done_train, "export": done_export, "eval": done_eval}


def job_name(seed, stage, regime: str = "log_replay") -> str:
    # k8s object names are RFC 1123 subdomains: no underscores.
    tag = f"-{regime.replace('_', '-')}" if stage == "eval" else ""
    return f"navsafe-dev-{stage}-{seed['seed_id']}{tag}"


def job_status(name: str, ns: str) -> str:
    """'succeeded' | 'failed' | 'active' | 'absent'"""
    r = subprocess.run(
        ["kubectl", "get", "job", name, "-n", ns, "-o",
         "jsonpath={.status.succeeded}|{.status.failed}|{.status.active}"],
        capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return "absent"
    s, f, a = (r.stdout.strip().split("|") + ["", "", ""])[:3]
    if s and s != "0":
        return "succeeded"
    if a and a != "0":
        return "active"
    if f and f != "0":
        return "failed"
    return "active"


def submit_k8s(pod: Pod, seed, stage: str, work: str,
               regime: str = "log_replay") -> None:
    name = job_name(seed, stage, regime)
    st = job_status(name, pod.ns)
    if st in ("active", "succeeded"):
        print(f"    job {name}: {st}")
        return
    if st == "failed":
        print(f"    job {name}: failed before -- deleting and resubmitting")
        subprocess.run(["kubectl", "delete", "job", name, "-n", pod.ns],
                       capture_output=True, text=True)
    yaml_path = f"{work}/jobs/{name}.yaml"
    extra = f"--regime {regime}" if stage == "eval" else ""
    gen = pod.sh(f"mkdir -p {work}/jobs && cd {cfg.NAVSAFE_ROOT} && "
                 f"python3 {WORLD}/k8s_jobs.py --seed {work}/seeds/{seed['seed_id']} "
                 f"--stage {stage} {extra} --out {yaml_path}")
    if gen.returncode != 0:
        print(f"    !! could not render {stage} job: {gen.stderr.strip()[:300]}")
        return
    manifest = pod.cat(yaml_path)
    r = subprocess.run(["kubectl", "apply", "-f", "-"], input=manifest,
                       capture_output=True, text=True)
    print(f"    submit {name}: {(r.stdout or r.stderr).strip()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", default=cfg.POD)
    ap.add_argument("--namespace", default=cfg.NAMESPACE)
    ap.add_argument("--work", default=str(cfg.WORK))
    ap.add_argument("--seeds", default="")
    ap.add_argument("--until", default="export", choices=STAGE_ORDER)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--regime", default="log_replay", choices=["log_replay", "idm"],
                    help="interaction regime for the eval stage")
    args = ap.parse_args()

    pod = Pod(args.pod, args.namespace)
    listing = pod.sh(f"ls {args.work}/seeds").stdout.split()
    wanted = [s for s in args.seeds.split(",") if s] or listing
    seeds = []
    for s in wanted:
        txt = pod.cat(f"{args.work}/seeds/{s}/seed.json")
        if txt.strip():
            seeds.append(json.loads(txt))
    if not seeds:
        print("no seeds found", file=sys.stderr)
        return 1

    upto = STAGE_ORDER.index(args.until)
    while True:
        pending = 0
        try:
            _probe = pod.exists(args.work)
        except PodUnreachable as e:
            # Stop rather than guess. GPU stages already in flight are
            # unaffected -- they run in their own pods.
            print(f"!! {e}\n!! halting; recreate the pod and rerun this command",
                  file=sys.stderr)
            return 2
        for seed in seeds:
            tok = seed["seed_id"]
            state = [(st, DONE[st](pod, seed, args.regime) if st == "eval"
                      else DONE[st](pod, seed))
                     for st in STAGE_ORDER[: upto + 1]]
            marks = " ".join(f"{st}{'+' if ok else '-'}" for st, ok in state)
            nxt = next((st for st, ok in state if not ok), None)
            print(f"  {tok} [{seed['family']:<26}] {marks}", flush=True)
            if nxt is None:
                continue
            pending += 1
            if nxt in K8S_STAGES:
                # A running Job is authoritative over the disk: its outputs
                # appear progressively, so never let a partially written stage
                # look finished to the stage that follows it.
                st = job_status(job_name(seed, nxt, args.regime), pod.ns)
                if st == "active":
                    print(f"    {nxt} job still running")
                    continue
                submit_k8s(pod, seed, nxt, args.work, args.regime)
            elif pod.stage_running(nxt, tok):
                print(f"    {nxt} already running in the pod")
            else:
                log = f"{args.work}/logs/{nxt}-{tok}.log"
                pod.sh(f"mkdir -p {args.work}/logs")
                pod.start_stage(nxt, f"{args.work}/seeds/{tok}", log)
                print(f"    started {nxt} in the pod -> {log}")
        if not pending:
            print("all seeds complete")
            return 0
        if not args.watch:
            return 0
        print(f"  -- {pending} seed(s) pending; sleeping {args.poll}s", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
