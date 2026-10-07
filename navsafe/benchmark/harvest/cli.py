# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``navsafe harvest`` -- build and check an asset bank.

    harvest   <scene_id>   parse, select, lift, orient, write the manifest
    verify    <scene_id>   do the manifest's track ids exist in the served recon
    reorient  <scene_id>   spin an existing bank about Y, without re-lifting it
    scenarios              which scenarios can be harvested, and which still need it
    batch     <ids...>     a Kubernetes Job that harvests many, N GPUs wide
    pack                   validate every bank, write an index and an upload plan
    status    [<scene_id>] which scenarios have a bank, and how big

The corpus-wide loop is `scenarios` piped into `batch`:

    navsafe harvest scenarios > /tmp/todo.txt
    navsafe harvest batch --scenes-file /tmp/todo.txt \\
        --workers 6 --out /tmp/harvest.yaml
    kubectl apply -f /tmp/harvest.yaml

``harvest`` is the only expensive one and it is resumable at the step boundary:
the lifted tree is kept in the scenario's own ``ah_assets/`` and a re-run skips
any track whose ``gaussians.ply`` is already there, so a pod cycled out mid-run
(the cluster caps a GPU lease at ~6 h) is restarted rather than restarted from
scratch. Pass ``--force`` to harvest a track again anyway.

``verify`` needs a running render server and is the check worth doing before a
sweep: the manifest names logged track ids, the reconstruction names its own,
and the whole mechanism is silently a no-op if the two ever stop matching.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from navsafe.benchmark import config as cfg
from navsafe.benchmark.harvest import ah, manifest, select

logger = logging.getLogger(__name__)

from navsafe.benchmark.world import deployment
DEFAULT_NODES = tuple(deployment.nodes())


def windows_for(scene_id: str, corpus: Optional[Path] = None) -> List[str]:
    """The 5 s reconstruction windows behind one scenario id.

    A stitched ``<token>_20s`` host holds only Arrow; its reconstructions are
    the ``<token>s1..s4`` siblings beside it (``leaves/hosts.py``). Any other id
    is its own single window.
    """
    root = corpus or cfg.CORPUS
    if scene_id.endswith("_20s"):
        token = scene_id[: -len("_20s")]
        found = sorted(p.name for p in root.glob(f"{token}s*") if p.is_dir())
        if not found:
            raise ah.HarvestError(
                f"{scene_id} is a stitched host but no {token}s* windows exist "
                f"under {root}")
        return found
    return [scene_id]


# --------------------------------------------------------------------------


def cmd_harvest(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus) if args.corpus else None
    try:
        ah.check_install()
    except ah.HarvestError as exc:
        # Exit 3 = this MACHINE cannot harvest, as opposed to 1 = this scenario
        # failed. A sweep must tell them apart: a pod that landed on a node with
        # a broken device plugin would otherwise burn through its whole slice
        # two seconds at a time, reporting seventeen scenario failures for one
        # node fault (observed on a failed GPU).
        print(f"[harvest] environment unusable: {exc}")
        return 3
    scene_id = args.scene_id
    windows = windows_for(scene_id, corpus)
    assets_dir = Path(args.out) if args.out else cfg.ah_assets_dir(scene_id, corpus)
    lifted = assets_dir / "lifted"
    log = assets_dir / "harvest.log"
    parse_root = Path(args.parse_root or (cfg.WORK / "ah_parse" / scene_id))

    print(f"[harvest] {scene_id}: {len(windows)} window(s) {', '.join(windows)}")
    print(f"[harvest] assets -> {assets_dir}")

    # 1. parse every window (cheap relative to lifting, and the only source of
    #    the distances selection ranks on).
    for window in windows:
        done = parse_root / window / "sample_paths.json"
        if done.is_file() and not args.force:
            print(f"[parse] {window}: already parsed, skipping")
            continue
        print(f"[parse] {window}")
        ah.parse_window(window, parse_root / window, corpus=corpus, log=log)

    # 2. which tracks actually drove. Cheap (it reads cuboids, not pixels) and
    #    it decides whether this feature helps or hurts: a parked car is one the
    #    reconstruction saw from every angle as the ego drove past, so replacing
    #    it only loses detail, while a moving car is one the reconstruction
    #    barely has an angular baseline on.
    motion: dict = {}
    if not args.include_parked:
        for window in windows:
            mfile = parse_root / window / "motion.json"
            if mfile.is_file() and not args.force:
                window_motion = json.loads(mfile.read_text())
            else:
                window_motion = ah.window_motion(window, mfile, corpus=corpus, log=log)
            # Keep the record with maximum per-clip net displacement.
            # A later stop must not erase motion observed in an earlier clip.
            for tid, rec in window_motion.items():
                if (tid not in motion or
                        select.displacement_of(window_motion, tid) >
                        select.displacement_of(motion, tid)):
                    motion[tid] = rec

    # 3. select.
    candidates = select.scan(parse_root, windows)
    chosen = select.choose(
        candidates,
        max_assets=args.max_assets,
        classes=(tuple(args.classes.split(",")) if args.classes else select.DEFAULT_CLASSES),
        motion=(None if args.include_parked else motion),
        min_motion_m=args.min_motion_m,
    )
    if not chosen:
        # A quiet scenario with no moving vehicle worth replacing is an ANSWER,
        # not an error: exit 2 so a sweep can count it apart from a genuine
        # failure. Returning 1 here once failed a 20-worker fleet outright --
        # one such scenario tripped `backoffLimit: 0` and terminated the other
        # nineteen workers mid-scenario.
        print(f"[select] nothing to harvest in {scene_id}: "
              f"{len(candidates)} track(s) parsed, none of them a vehicle that "
              f"moved at least {args.min_motion_m:g} m in any clip")
        # A manifest from an earlier, wider selection must not survive this.
        # It would keep naming actors -- parked ones, under the old rules --
        # that this run has just decided should not be replaced, and it would be
        # published as if it were current.
        stale_manifest = manifest.manifest_path(assets_dir)
        if stale_manifest.is_file():
            stale_manifest.unlink()
            print(f"[select] removed the previous manifest: its selection is no "
                  f"longer what this scenario should ship")
        # Say so on disk. "Ran and found nothing" and "still running" leave the
        # same absence of a manifest, and leftover PLYs from an earlier, wider
        # selection make the directory look half-built. A marker settles it for
        # anyone reading the tree later, and for `pack`.
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / manifest.EMPTY_MARKER).write_text(
            f"{len(candidates)} track(s) parsed, none of them a vehicle that "
            f"moved at least {args.min_motion_m:g} m in any clip. Nothing to replace here.\n")
        return 2
    parsed = len({c.track_id for c in candidates})
    moved = (sum(1 for t in {c.track_id for c in candidates}
                 if select.displacement_of(motion, t) >= args.min_motion_m)
             if motion else parsed)
    print(f"[select] {len(chosen)}/{parsed} track(s) "
          f"({moved} of them moved at least {args.min_motion_m:g} m in any clip), "
          f"nearest {chosen[0].min_dist_m:.1f} m, furthest kept "
          f"{chosen[-1].min_dist_m:.1f} m")

    # 4. lift only what is not already lifted. Asset Harvester has no resume of
    #    its own; staging just the missing samples is the resume.
    #
    #    Lifting writes to a STAGING directory, not straight into the bank, so
    #    that the orientation fix-up below applies exactly once per asset: it is
    #    a 90-degree rotation in place, and running it twice over a resumed
    #    bank would turn every previously harvested car sideways silently.
    have = ah.lifted_assets(lifted) if lifted.is_dir() else {}
    todo = [c for c in chosen if args.force or c.track_id not in have]
    if todo:
        data_root = assets_dir / "_selected"
        staging = assets_dir / "_lifting"
        ah.clear(data_root)
        ah.clear(staging)
        select.write_sample_paths(todo, data_root)
        print(f"[lift] {len(todo)} track(s) ({len(chosen) - len(todo)} already lifted)")
        ah.lift(data_root, staging, num_steps=args.num_steps,
                cfg_scale=args.cfg_scale, offload=args.offload, log=log)
        empty_plys = ah.drop_degenerate(staging)
        if empty_plys:
            print(f"[lift] {len(empty_plys)} track(s) lifted to an empty cloud "
                  f"and were dropped: {', '.join(empty_plys)}")
        ah.orient(staging, degrees=args.orient_degrees, log=log)
        promoted = ah.promote(staging, lifted)
        print(f"[lift] {len(promoted)} new asset(s) oriented and banked")
        ah.clear(data_root)
    else:
        print(f"[lift] all {len(chosen)} track(s) already lifted")

    # 5. describe the whole bank, then the manifest.
    ah.describe(lifted, log=log)
    produced = ah.lifted_assets(lifted, measure=True)
    by_track = {c.track_id: c for c in chosen}
    entries, misshapen, stale = {}, [], []
    for tid, rec in produced.items():
        # The bank on disk is cumulative; the manifest is not. An earlier run
        # with a wider selection (or --include-parked) leaves its PLYs behind,
        # and shipping them would replace actors this run deliberately did not
        # choose — silently, since a manifest carries no record of why a track
        # is in it. Keep the files, list only what was selected.
        if tid not in by_track:
            stale.append(tid)
            continue
        # Quality gate. The server scales an asset onto the track's box, so an
        # asset whose SHAPE disagrees with the clip's cuboid does not render
        # small -- it renders as a car stretched onto the right footprint,
        # which is more visibly wrong than the smear it replaced. Measured on
        # 2b7bf25209dd5705: 26 of 28 agreed within 20%, one was 93% too wide
        # (the mask had caught the neighbouring car).
        err = float(rec.get("aspect_error") or 0.0)
        if err > args.max_aspect_error:
            misshapen.append((tid, err))
            continue
        cand = by_track.get(tid)
        entries[tid] = {
            **rec,
            "min_ego_dist_m": round(cand.min_dist_m, 2) if cand else None,
            "displacement_m": round(select.displacement_of(motion, tid), 2) or None,
            "source_window": cand.window if cand else None,
            "n_views": cand.n_views if cand else None,
        }
    if stale:
        print(f"[bank] {len(stale)} asset(s) on disk are not in this selection "
              f"and were left out of the manifest (kept for re-selection)")
    if misshapen:
        print(f"[gate] {len(misshapen)} asset(s) kept on disk but LEFT OUT of the "
              f"manifest — lifted shape disagrees with the clip cuboid by more "
              f"than {args.max_aspect_error:.0%}:")
        for tid, err in sorted(misshapen, key=lambda t: -t[1]):
            print(f"        {tid}  {err:.0%}")
    (assets_dir / manifest.EMPTY_MARKER).unlink(missing_ok=True)
    out = manifest.write(
        assets_dir, scene_id, entries, windows=windows,
        provenance={
            "harvester": "NVIDIA/asset-harvester",
            "repo": str(cfg.AH_REPO),
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "max_assets": args.max_assets,
            "motion_aggregation": "max_clip_net_displacement",
            "selected": len(chosen),
        })
    skipped = sorted(set(by_track) - set(produced))
    if skipped:
        print(f"[harvest] {len(skipped)} selected track(s) produced no asset "
              f"(see {log}): {', '.join(skipped)}")
    print(f"[harvest] {len(entries)} asset(s) -> {out}")
    if not args.keep_parse:
        ah.clear(parse_root)
    return 0


# --------------------------------------------------------------------------


def cmd_reorient(args: argparse.Namespace) -> int:
    """Spin an existing bank about Y, without re-lifting it.

    Orientation is the one thing in this pipeline that cannot be settled
    offline. The extents of a lifted PLY pin which axis is the car's length and
    which is its height, and a render pins which way is up -- but not which END
    of the length axis is the front, and a 180-degree error there renders every
    replaced actor driving backwards while passing every numeric check.

    So the answer comes from a rendered frame, and this is what applies it:
    re-lifting a bank to change a rotation would be ~20 min of diffusion per
    scenario to fix a matrix multiply.
    """
    corpus = Path(args.corpus) if args.corpus else None
    assets_dir = Path(args.out) if args.out else cfg.ah_assets_dir(args.scene_id, corpus)
    lifted = assets_dir / "lifted"
    if not lifted.is_dir():
        print(f"[reorient] no bank at {lifted}")
        return 1
    print(f"[reorient] {lifted}: {args.degrees:+g} degrees about Y")
    ah.orient(lifted, degrees=args.degrees, log=assets_dir / "harvest.log")
    print("[reorient] done — the manifest is unchanged (it names paths, not poses)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Cross-check the manifest against what the render server actually serves."""
    corpus = Path(args.corpus) if args.corpus else None
    scene_id = args.scene_id
    assets_dir = Path(args.out) if args.out else cfg.ah_assets_dir(scene_id, corpus)
    doc = manifest.read(assets_dir)
    print(f"[verify] {assets_dir}: {len(doc['assets'])} asset(s)")

    import grpc  # noqa: PLC0415 -- optional, only this command needs it
    from navsafe._vendor.nurec_grpc import sensorsim_pb2 as ss  # noqa: PLC0415
    from navsafe._vendor.nurec_grpc import sensorsim_pb2_grpc as ss_grpc  # noqa: PLC0415

    host = os.environ.get("NUREC_GRPC_HOST", "nurec-grpc")
    # A k8s Service named "nurec-grpc" makes the kubelet inject
    # NUREC_GRPC_PORT="tcp://<ip>:<port>" into every pod in the namespace, so
    # the obvious read of this variable produces "host:tcp://ip:8080" and every
    # window comes back UNAVAILABLE. Take the trailing port, as
    # ``render/nurec_grpc.py`` already does.
    port = str(os.environ.get("NUREC_GRPC_PORT", "8080")).rsplit(":", 1)[-1]
    stub = ss_grpc.SensorsimServiceStub(grpc.insecure_channel(f"{host}:{port}"))

    served: set = set()
    reachable = 0
    windows = windows_for(scene_id, corpus)
    for window in windows:
        try:
            dyn = stub.get_dynamic_objects(
                ss.AvailableDynamicObjectsRequest(scene_id=window), timeout=600)
        except grpc.RpcError as exc:
            print(f"[verify] {window}: NOT SERVED ({exc.code().name})")
            continue
        reachable += 1
        ids = {o.id for o in dyn.dynamic_objects}
        hit = sorted(set(doc["assets"]) & ids)
        served |= ids
        print(f"[verify] {window}: {len(ids)} track(s) served, {len(hit)} replaceable")
    # Two different failures that must not be reported as one. A server that
    # answered nothing says nothing about the bank -- and serve-grpc binds about
    # two minutes AFTER it prints its scene list, so "connect during the gap" is
    # the common case.
    if reachable == 0:
        print(f"[verify] INCONCLUSIVE: none of {len(windows)} window(s) answered "
              f"at {host}:{port}. Check the server holds this scenario "
              f"(`kubectl logs ... | grep 'Available scenes'`) and that it has "
              f"printed 'Serving on' -- it binds ~2 min after the scene list.")
        return 2
    missing = manifest.unmatched(doc, served)
    if len(missing) == len(doc["assets"]):
        print(f"[verify] FAILED: {reachable} window(s) answered and not one "
              f"manifest track id is among their tracks. The bank does not "
              f"belong to this reconstruction.")
        return 1
    if missing:
        print(f"[verify] {len(missing)} manifest track(s) in no window "
              f"(normal -- a bank covers 20 s, a window is 5 s)")
    print("[verify] OK")
    return 0


# --------------------------------------------------------------------------


def cmd_scenarios(args: argparse.Namespace) -> int:
    """Which scenarios *can* be harvested, and which still need it.

    Two things gate it and both are silent otherwise: a scenario needs its NCore
    clips (some corpus directories no longer have them, and without them there
    is nothing to harvest from) and it needs the reconstruction those clips
    trained, or there is nothing to replace actors in.
    """
    corpus = Path(args.corpus) if args.corpus else cfg.CORPUS
    # Enumerate by TOKEN, from the reconstruction windows themselves -- not by
    # the `<token>_20s` directory. That directory is created by the Arrow
    # conversion, which is an EVAL prerequisite; harvesting needs only the NCore
    # clips. Going by `_20s` reported 87 harvestable scenarios out of 407 and
    # silently hid 320 that were ready. The bank still belongs at
    # `<token>_20s/ah_assets`, which is where a later Arrow conversion lands, so
    # it is already in the right place when the scenario becomes evaluatable.
    hosts = sorted({re.sub(r"s[1-4]$", "", p.name) + "_20s"
                    for p in corpus.glob("*s[1-4]") if p.is_dir()}
                   | {p.name for p in corpus.glob("*_20s") if p.is_dir()})
    ready, blocked = [], []
    for scene_id in hosts:
        windows = windows_for(scene_id, corpus)
        no_clip = [w for w in windows
                   if not (cfg.clips_dir(w, corpus) / f"pai_{w}.json").is_file()]
        no_recon = [w for w in windows if cfg.recon_usdz(w, corpus) is None]
        # "Done" means a manifest written by the CURRENT rules. A bank from an
        # older schema still resolves in place but is not portable, and one
        # selected under older rules may hold parked cars -- both need the
        # scenario re-run, so neither counts as done.
        adir = cfg.ah_assets_dir(scene_id, corpus)
        mpath = manifest.manifest_path(adir)
        # A scenario judged to have nothing to replace is DONE. Without this
        # every sweep re-parses all of them forever -- 35 scenarios of pure
        # waste per pass -- and they show up as outstanding work that is not.
        done = (adir / manifest.EMPTY_MARKER).is_file()
        if not done and mpath.is_file():
            try:
                done = json.loads(mpath.read_text()).get("schema") == manifest.SCHEMA_VERSION
            except json.JSONDecodeError:
                done = False
        if no_clip or no_recon:
            blocked.append((scene_id, len(no_clip), len(no_recon)))
        elif not done or args.all:
            ready.append(scene_id)
    for scene_id in ready:
        print(scene_id)
    print(f"# {len(ready)} harvestable, {len(blocked)} blocked, "
          f"{len(hosts)} stitched host(s) under {corpus}", file=sys.stderr)
    for scene_id, nc, nr in blocked[:10]:
        print(f"#   {scene_id}: {nc} window(s) without a clip, "
              f"{nr} without a reconstruction", file=sys.stderr)
    return 0


_BATCH_JOB = """\
apiVersion: batch/v1
kind: Job
metadata: {{name: {name}, namespace: {ns}}}
spec:
  completions: {workers}
  parallelism: {workers}
  completionMode: Indexed
  # One retry per worker. A pod can land on a GPU that CUDA cannot open -- one
  # bad device on an otherwise healthy node, observed on a failed GPU -- and with
  # backoffLimit 0 that single pod failed the whole Job and terminated the other
  # nineteen mid-scenario. Every stage is resumable and skips what exists, so a
  # retried index costs a re-parse and nothing else. It is still bounded: a
  # systematically broken fleet exhausts this and stops.
  backoffLimit: {workers}
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: harvest
          image: {image}
          command: ["bash", "-lc"]
          args:
            - |
              set -uo pipefail
              # Indexed completions: worker N takes every Nth scenario, so one
              # slow scenario does not leave a worker idle at the end.
              W=$JOB_COMPLETION_INDEX
              i=0; ok=0; empty=0; bad=0
              for S in {scenes}; do
                if [ $(( i % {workers} )) -eq "$W" ]; then
                  echo "[w$W] $S"
                  PYTHONPATH={navsafe} {python} -m navsafe.benchmark.harvest \\
                      harvest "$S" --max-assets {max_assets}
                  case $? in
                    0) ok=$(( ok + 1 )) ;;
                    2) empty=$(( empty + 1 )); echo "[w$W] $S: nothing to harvest" ;;
                    3) echo "[w$W] ABORTING SLICE: this node cannot harvest"; exit 1 ;;
                    *) bad=$(( bad + 1 ));   echo "[w$W] $S: FAILED" ;;
                  esac
                fi
                i=$(( i + 1 ))
              done
              echo "[w$W] done: $ok harvested, $empty empty, $bad failed"
              # One bad scenario must not take the fleet down with it
              # (backoffLimit is 0, so a non-zero worker fails the whole Job and
              # terminates the others mid-scenario). Fail only if this worker
              # accomplished nothing at all, which is what a systematic problem
              # -- a missing install, an unreadable corpus -- actually looks
              # like. Per-scenario failures are named above and in the Job log.
              [ $(( ok + empty )) -gt 0 ] || exit 1
              exit 0
          env:
            # Named in the error when a pod gets a GPU CUDA cannot open.
            - {{name: NODE_NAME, valueFrom: {{fieldRef: {{fieldPath: spec.nodeName}}}}}}
            - {{name: NAVSAFE_AH_HOME, value: "{ah_home}"}}
            - {{name: HF_HOME, value: "{ah_home}/hf"}}
            # Without this the per-stage prints sit in a block buffer until the
            # worker exits, so `kubectl logs` shows one line for an hour of work
            # and there is no way to tell a slow scenario from a stuck one.
            - {{name: PYTHONUNBUFFERED, value: "1"}}
          # RIGHT-SIZED, not padded: a cluster's utilisation policy compares a
          # pod's ACTUAL cpu/mem against its REQUESTS and then denies the whole
          # account EVERY new pod -- GPU is not part of that comparison, and
          # the running pods are the cause, so it cannot be waited out. A
          # harvest measured 0.89 cpu / 7.6 GiB, so this is ~2x observed.
          resources:
            requests: {{cpu: "3", memory: "16Gi", nvidia.com/gpu: "1", ephemeral-storage: 40Gi}}
            limits: {{cpu: "3", memory: "16Gi", nvidia.com/gpu: "1", ephemeral-storage: 40Gi}}
          volumeMounts:
            - {{name: dshm, mountPath: /dev/shm}}
      volumes:
        - {{name: dshm, emptyDir: {{medium: Memory, sizeLimit: 8Gi}}}}
"""


def cmd_batch(args: argparse.Namespace) -> int:
    """Emit one Kubernetes Job that harvests many scenarios across N GPUs.

    A Job rather than a pod per scenario: harvesting is minutes to an hour of
    pure GPU work with no coordination between scenarios, and a controller-owned
    workload is also the only kind some clusters allow past a per-pod resource cap on
    bare pods.

    ``backoffLimit: 0`` on purpose. Every stage is resumable (a re-run skips
    parsed windows and lifted tracks), so the response to a pre-empted worker is
    to resubmit the Job and let it skip what is done -- not to have Kubernetes
    silently retry a container whose failure might be a real one.
    """
    scenes = [s for s in (args.scenes or []) if s]
    if args.scenes_file:
        scenes += [ln.strip() for ln in Path(args.scenes_file).read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")]
    if not scenes:
        print("no scenarios: pass ids, or --scenes-file (see the `scenarios` "
              "command for what is harvestable)", file=sys.stderr)
        return 2
    workers = max(1, min(args.workers, len(scenes)))
    doc = _BATCH_JOB.format(
        name=args.name, ns=cfg.NAMESPACE, workers=workers, image=args.image,
        scenes=" ".join(scenes), navsafe=cfg.NAVSAFE_ROOT, python=cfg.AH_PYTHON,
        max_assets=args.max_assets, ah_home=cfg.AH_HOME,
        nodes="[" + ", ".join(args.nodes.split(",")) + "]")
    import yaml
    parsed = yaml.safe_load(doc)
    deployment.configure(parsed["spec"]["template"]["spec"],
                         [n for n in args.nodes.split(",") if n])
    doc = yaml.safe_dump(parsed, sort_keys=False)
    if args.out and args.out != "-":
        Path(args.out).write_text(doc)
        print(f"{len(scenes)} scenario(s) over {workers} worker(s) -> {args.out}",
              file=sys.stderr)
    else:
        sys.stdout.write(doc)
    return 0



_PACK_README = """\
# NavSafe harvested actor assets

One directory per scenario, holding a view-consistent 3D Gaussian asset for each
of that scenario's **moving** vehicles, plus the manifest that says which logged
track each asset replaces.

    <scene_id>/ah_assets/
    ├── replace_manifest.json          what the renderer reads
    ├── lifted/metadata.yaml           the same assets in NVIDIA's external-assets format
    ├── lifted/<class>/<track_id>/
    │   ├── gaussians.ply              the asset (~5.4 MB, ~100k gaussians)
    │   ├── multiview/                 the 16 synthesised views it was lifted from
    │   ├── input/                     the real crops those were conditioned on
    │   └── {{multiview,3d_lifted}}.mp4  previews
    └── harvest.log                    why a track produced no asset

`replace_manifest.json` records **paths relative to itself**, so a bank works
wherever it is unpacked. Every entry also carries the provenance that decides
whether it should have been harvested at all: how close the actor came to the
ego (`min_ego_dist_m`), how far it drove (`displacement_m`), which 5 s window
the crops came from, and how many views were usable.

## Using it

Drop the directory beside the scenario it belongs to and pass one flag:

    --asset-harvester-replace          # finds <scenario>/ah_assets/ automatically

It needs `--render-backend nurec_grpc`, and the render server must run with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. On a 24 GB card the
practical ceiling is ~12 replaced actors per episode
(`NUREC_GRPC_ASSET_REPLACE_MAX`). See `docs/navsafe_harvested_actors.md`.

## What is deliberately NOT in here

Parked cars. The ego drives past them, so the reconstruction saw them across a
wide arc and beats a lifted asset; only actors that actually drove are
harvested. Assets whose lifted shape disagreed with the clip's cuboid by more
than 35% are left out of the manifest as well.

Assets are produced with NVIDIA Asset Harvester (Apache-2.0),
<https://github.com/NVIDIA/asset-harvester>.
"""


def cmd_pack(args: argparse.Namespace) -> int:
    """Validate every bank and emit an index plus an upload plan.

    Deliberately copies nothing. A bank is already laid out the way it should be
    published -- inside the scenario it belongs to, which is where the eval flag
    looks for it -- so packing means checking that each one is complete and
    portable, and writing down what to upload where. Copying 10+ GB to a staging
    tree would only create a second copy to keep in step.
    """
    corpus = Path(args.corpus) if args.corpus else cfg.CORPUS
    # `--dest`, not the top-level `--out`: that one names a scenario's asset
    # directory, and reusing it here would mean two different things by the
    # same flag depending on the subcommand.
    out = Path(args.dest) if args.dest else cfg.run_dir("ah-assets-pack")
    out.mkdir(parents=True, exist_ok=True)

    banks = sorted(p.parent for p in corpus.glob("*/ah_assets") if p.is_dir())
    index, broken, unfinished, empty = {}, [], [], []
    total_assets = total_bytes = 0
    for scene in banks:
        adir = scene / "ah_assets"
        # Three different things look the same from outside, and calling them
        # all "failed" would bury the one that matters:
        #   * a harvest that RAN and found nothing to replace -- a quiet
        #     scenario with no moving vehicle the log saw well enough to lift.
        #     That is an answer, and there is nothing to publish for it.
        #   * a harvest still in progress (normal while a fleet runs).
        #   * a manifest that exists and does not validate. Only this is a bug.
        if not manifest.manifest_path(adir).is_file():
            (empty if (adir / manifest.EMPTY_MARKER).is_file()
             else unfinished).append(scene.name)
            continue
        try:
            doc = manifest.read(adir)
        except manifest.ManifestError as exc:
            broken.append((scene.name, str(exc).replace(str(corpus), "<corpus>")))
            continue
        plys = [Path(a["ply"]) for a in doc["assets"].values()]
        nbytes = sum(p.stat().st_size for p in plys)
        total_assets += len(plys)
        total_bytes += nbytes
        index[scene.name] = {
            "scene_id": doc["scene_id"],
            "windows": doc["windows"],
            "created": doc.get("created"),
            "n_assets": len(plys),
            "ply_bytes": nbytes,
            "bank_bytes": sum(f.stat().st_size for f in adir.rglob("*") if f.is_file()),
            "provenance": doc.get("provenance", {}),
            "assets": {t: {k: v for k, v in a.items() if k != "ply"}
                       for t, a in doc["assets"].items()},
        }

    (out / "index.json").write_text(json.dumps(
        {"corpus": str(corpus), "n_scenarios": len(index),
         "n_assets": total_assets, "ply_bytes": total_bytes,
         "empty_scenarios": sorted(empty), "scenarios": index},
        indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(_PACK_README)

    lines = ["#!/usr/bin/env bash",
             "# Upload every harvested bank into the dataset, one scenario at a time.",
             "# Each lands beside the scenario it belongs to, which is where",
             "# --asset-harvester-replace looks for it with no configuration.",
             "#",
             "# Add --exclude '*/multiview/*' '*/input/*' '*.mp4' to ship only the",
             "# PLYs and the manifests (~82% of the bytes are the PLYs).",
             "set -euo pipefail",
             f'REPO="${{1:-{args.repo}}}"',
             f'PREFIX="${{2:-{args.prefix}}}"', ""]
    for name in index:
        lines.append(f'hf upload "$REPO" "{corpus / name / "ah_assets"}" '
                     f'"$PREFIX/{name}/ah_assets" --repo-type dataset')
    lines.append('echo "uploaded %d scenario bank(s)"' % len(index))
    plan = out / "upload.sh"
    plan.write_text("\n".join(lines) + "\n")
    plan.chmod(0o755)

    if broken:
        (out / "broken.txt").write_text(
            "\n".join(f"{n}: {e}" for n, e in broken) + "\n")
    if unfinished:
        (out / "unfinished.txt").write_text("\n".join(unfinished) + "\n")
    if empty:
        (out / "empty.txt").write_text("\n".join(empty) + "\n")

    print(f"[pack] {len(index)} bank(s) validated, {total_assets} asset(s), "
          f"{total_bytes / 1e9:.1f} GB of PLY")
    if empty:
        print(f"[pack] {len(empty)} scenario(s) have nothing to harvest -- no "
              f"moving vehicle the log saw well enough to lift -> "
              f"{out / 'empty.txt'}")
    if unfinished:
        print(f"[pack] {len(unfinished)} bank(s) have no manifest yet "
              f"(harvest not finished) -> {out / 'unfinished.txt'}")
    if broken:
        print(f"[pack] {len(broken)} bank(s) FAILED validation -> {out / 'broken.txt'}")
        for n, e in broken[:5]:
            print(f"        {n}: {e[:110]}")
    print(f"[pack] index    {out / 'index.json'}")
    print(f"[pack] readme   {out / 'README.md'}")
    print(f"[pack] upload   {plan}")
    return 1 if (broken or unfinished) and args.strict else 0


def cmd_status(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus) if args.corpus else cfg.CORPUS
    scenes = [args.scene_id] if args.scene_id else sorted(
        p.parent.name for p in corpus.glob("*/ah_assets") if p.is_dir())
    if not scenes:
        print(f"no harvested banks under {corpus}")
        return 0
    for scene_id in scenes:
        path = manifest.manifest_path(cfg.ah_assets_dir(scene_id, corpus))
        if not path.is_file():
            print(f"{scene_id:<24} (no manifest)")
            continue
        doc = json.loads(path.read_text())
        dists = [a.get("min_ego_dist_m") for a in doc["assets"].values()
                 if a.get("min_ego_dist_m")]
        span = f"{min(dists):.0f}-{max(dists):.0f} m" if dists else "-"
        print(f"{scene_id:<24} {len(doc['assets']):>3} asset(s)  ego dist {span}"
              f"  {doc.get('created', '')}")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="navsafe harvest",
        description="Harvest per-actor 3D assets for a NavSafe scenario and "
                    "record which logged track each one replaces.")
    ap.add_argument("--corpus", default=None, help="override NAVSAFE_CORPUS")
    ap.add_argument("--out", default=None,
                    help="override the asset directory (default: the "
                         "scenario's own ah_assets/)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="build the asset bank for one scenario")
    h.add_argument("scene_id")
    h.add_argument("--max-assets", type=int, default=select.DEFAULT_MAX_ASSETS,
                   help="cap on harvested actors, nearest first (default "
                        f"{select.DEFAULT_MAX_ASSETS}; every one is resident in "
                        "renderer VRAM beside four reconstructions)")
    h.add_argument("--classes", default=None,
                   help="comma-separated label classes to consider (default "
                        f"{','.join(select.DEFAULT_CLASSES)}; deformables are "
                        "excluded because a rigid asset cannot walk)")
    h.add_argument("--max-aspect-error", type=float, default=0.35,
                   help="drop an asset from the manifest when its lifted shape "
                        "disagrees with the clip's cuboid by more than this "
                        "(default 0.35). The PLY is kept so it can be looked at")
    h.add_argument("--min-motion-m", type=float, default=select.DEFAULT_MIN_MOTION_M,
                   help="minimum net displacement in ANY clip for a "
                        f"track to count as moving (default {select.DEFAULT_MIN_MOTION_M:g}). "
                        "Parked cars are excluded because the reconstruction "
                        "saw them from every angle as the ego drove past, so "
                        "replacing one only loses detail")
    h.add_argument("--include-parked", action="store_true",
                   help="harvest stationary actors too. Only useful for "
                        "diagnosing: it spends the asset budget and the "
                        "renderer's VRAM where the reconstruction is already "
                        "better than the harvester")
    h.add_argument("--orient-degrees", type=float, default=90.0,
                   help="Y rotation applied to each lifted PLY (default 90, the "
                        "upstream value). If replaced actors render facing "
                        "backwards, fix an existing bank with `reorient "
                        "--degrees 180` rather than re-harvesting")
    h.add_argument("--num-steps", type=int, default=ah.DEFAULT_NUM_STEPS)
    h.add_argument("--cfg-scale", type=float, default=ah.DEFAULT_CFG_SCALE)
    h.add_argument("--offload", action="store_true",
                   help="halve peak VRAM on a <16 GB card, at a latency cost")
    h.add_argument("--parse-root", default=None,
                   help="where the multi-view crops are staged (default: "
                        "NAVSAFE_WORK/ah_parse/<scene_id>, deleted on success)")
    h.add_argument("--keep-parse", action="store_true",
                   help="keep the crops, e.g. to re-select without re-parsing")
    h.add_argument("--force", action="store_true",
                   help="re-parse and re-lift tracks that already have assets")
    h.set_defaults(fn=cmd_harvest)

    r = sub.add_parser("reorient",
                       help="rotate an existing bank about Y (no re-lifting)")
    r.add_argument("scene_id")
    r.add_argument("--degrees", type=float, required=True,
                   help="e.g. 180 when replaced actors render facing backwards")
    r.set_defaults(fn=cmd_reorient)

    v = sub.add_parser("verify", help="check the manifest against a live server")
    v.add_argument("scene_id")
    v.set_defaults(fn=cmd_verify)

    n = sub.add_parser("scenarios", help="which scenarios can be harvested")
    n.add_argument("--all", action="store_true",
                   help="include ones that already have a bank")
    n.set_defaults(fn=cmd_scenarios)

    b = sub.add_parser("batch", help="emit a Kubernetes Job harvesting many scenarios")
    b.add_argument("scenes", nargs="*", help="scene ids (or use --scenes-file)")
    b.add_argument("--scenes-file", default=None,
                   help="one scene id per line, e.g. the output of `scenarios`")
    b.add_argument("--name", default="navsafe-harvest")
    b.add_argument("--workers", type=int, default=4,
                   help="GPUs to spread the scenarios over (default 4)")
    b.add_argument("--nodes", default=",".join(DEFAULT_NODES),
                   help="comma-separated hostnames the fleet may land on "
                        "(default: NAVSAFE_NODES; otherwise any schedulable node)")
    b.add_argument("--max-assets", type=int, default=select.DEFAULT_MAX_ASSETS)
    b.add_argument("--image", default="robinwangucsd/metabench:latest",
                   help="any image with conda and the PVCs mounted; the Asset "
                        "Harvester env itself lives on the PVC, not in the image")
    b.add_argument("--out", default="-")
    b.set_defaults(fn=cmd_batch)

    k = sub.add_parser("pack", help="validate every bank and write an upload plan")
    k.add_argument("--dest", default=None,
                   help="where the index, README and upload plan go "
                        "(default: a dated run directory)")
    k.add_argument("--repo", default="c13752hz/NavSafe",
                   help="HuggingFace dataset the upload plan targets")
    k.add_argument("--prefix", default="full_test",
                   help="path inside the dataset the banks go under")
    k.add_argument("--strict", action="store_true",
                   help="exit non-zero if any bank fails validation")
    k.set_defaults(fn=cmd_pack)

    s = sub.add_parser("status", help="list harvested banks")
    s.add_argument("scene_id", nargs="?", default=None)
    s.set_defaults(fn=cmd_status)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
