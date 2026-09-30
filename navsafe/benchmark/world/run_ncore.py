"""Stage 1 (CPU): NavSafe seed -> NCore V4 store.

Reads the frozen ``seed.json`` and converts the seed's **reconstruction**
window (not the scored window -- reconstruction needs the wider extent so the
closed-loop camera never sits at the edge of the trained time range) into an
NCore V4 store via ``NuplanLogReader`` + ``NCoreBridge``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from navsafe.benchmark import config as cfg


# py123d's nuPlan parser resolves splits through its own naming; nuPlan's
# on-disk ``val`` logs live under ``splits/trainval`` as far as py123d is
# concerned, so a val seed must be staged there before it can be converted.
PY123D_SPLIT = {"test": "nuplan_test", "val": "nuplan_val", "mini": "nuplan-mini_test"}
PY123D_DIR = {"nuplan_test": "test", "nuplan_val": "trainval", "nuplan-mini_test": "mini"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, help="path to seed.json or the seed dir")
    # Container-local disk, NOT the shared PVC.  The nuPlan devkit's ORM issues
    # ~17 M page reads against a single 55 MB log db; on CephFS that is tens of
    # GB of network round-trips and the process sits in D-state for half an
    # hour, whereas the same work is page-cache-hot on local disk.
    ap.add_argument("--nuplan-local", default=str(cfg.LOCAL_DB),
                    help="local staging tree (nuplan-v1.1/splits/<dir>/<log>.db); "
                         "the db is copied here from the PVC if absent")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seed_path = Path(args.seed)
    if seed_path.is_dir():
        seed_path = seed_path / "seed.json"
    seed = json.loads(seed_path.read_text())

    token = seed["seed_id"]
    prov = seed["provenance"]
    win = seed["window"]
    out = Path(seed["artifacts"]["ncore"])
    manifest = out / "clips" / token / f"pai_{token}.json"
    if manifest.exists() and not args.force:
        print(f"[ncore] {token}: already converted -> {manifest}")
        return 0

    split = PY123D_SPLIT.get(prov["split"])
    if split is None:
        raise SystemExit(f"unsupported nuPlan split {prov['split']!r}")
    staged = Path(args.nuplan_local) / "nuplan-v1.1" / "splits" / PY123D_DIR[split] / f"{prov['log_name']}.db"
    if not staged.exists():
        src = Path(seed["source_paths"]["db"])
        if not src.exists():
            raise SystemExit(f"source db missing: {src}")
        staged.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ncore] staging {src.name} -> {staged.parent} "
              f"({src.stat().st_size / 1e6:.0f} MB)", flush=True)
        shutil.copy2(src, staged)

    # Re-reference the store to the ego pose at frame 0.  nuPlan poses are
    # absolute UTM (~4.7e6 m); stored as float32 their ULP is ~0.5 m, which
    # shows up downstream as ego sawtooth and jagged lane geometry.  NCoreBridge
    # gates this on an env var, so set it here rather than relying on the
    # caller's environment -- a store built without it is silently jittery, and
    # the mistake only becomes visible after a 50-minute training run.
    # The local<->UTM shift is written to nurec_origin_offset.json beside the
    # store.  At eval time the scenario side must be recentred to the SAME
    # frame-0 origin (PY123D_RECENTER=1); because both origins are the same
    # nuPlan ego pose they then align at offset 0, so the render bridge's
    # NUREC_GRPC_ORIGIN_OFFSET_FILE override must be left UNSET or the shift is
    # subtracted twice.  The sidecar is for the non-recentred scenario path.
    os.environ["NCORE_REREF_FRAME0"] = "1"

    from navsafe.gs3d_converter.log_ingestion import NuplanLogReader
    from navsafe.gs3d_converter.ncore_bridge import NCoreBridge

    t0, t1 = win["recon_t0_us"], win["recon_t1_us"]
    print(f"[ncore] {token}  {prov['log_name']}  {(t1 - t0) / 1e6:.1f}s  ({prov['location']})", flush=True)

    reader = NuplanLogReader(
        nuplan_data_root=str(args.nuplan_local),
        sensor_root=seed["source_paths"]["sensor_root"],
        maps_root=seed["source_paths"]["maps_root"],
        scenes=[(prov["log_name"], token, t0, t1)],
        split=split,
    )
    started = time.time()
    scene = next(iter(reader.iter_scenes()))
    out.mkdir(parents=True, exist_ok=True)
    NCoreBridge.prepare(scene, out)
    print(f"[ncore] {token}: done in {time.time() - started:.0f}s -> {manifest}", flush=True)
    if not manifest.exists():
        raise SystemExit(f"[ncore] {token}: expected manifest not written: {manifest}")
    offset = manifest.parent / "nurec_origin_offset.json"
    if not offset.exists():
        raise SystemExit(
            f"[ncore] {token}: {offset.name} missing -- the store was NOT "
            f"re-referenced and will be jittery; do not train on it")
    print(f"[ncore] {token}: origin offset {json.loads(offset.read_text())['offset_xy_utm']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
