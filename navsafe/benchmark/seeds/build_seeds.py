"""Freeze chosen event-window candidates into NavSafe seed directories.

A *seed* is one mined real event plus everything needed to rebuild and score it
(taxonomy doc Table III).  This step writes the immutable descriptor; the
sensor clip / map slice / agent logs are materialised downstream by the NCore
and Arrow stages, which both read their window from this file.

The seed id is the nuPlan trigger ``lidar_pc`` token -- i.e. the nuPlan
scenario id itself, so provenance back to the source dataset is exact.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg


NAVSAFE_SEED_VERSION = "0.1.0"

# nuPlan on-disk split -> the split name py123d writes under.
PY123D_SPLIT = {"test": "nuplan_test", "val": "nuplan_val",
                "mini": "nuplan-mini_test"}


def build_seed(cand: dict, out_root: Path, nuplan_root: Path, work_root: Path) -> Path:
    token = cand["trigger_token"].lower()
    seed_dir = out_root / token
    seed_dir.mkdir(parents=True, exist_ok=True)

    seed = {
        "seed_version": NAVSAFE_SEED_VERSION,
        "seed_id": token,
        "family": cand["family"],
        "doc_family": cand["doc_family"],
        # --- provenance (split hygiene, taxonomy doc II-C) -----------------
        "provenance": {
            "dataset": "nuplan-v1.1",
            "split": cand["split"],
            "log_name": cand["log_name"],
            "location": cand["location"],
            "map_version": cand["map_version"],
            "scenario_type": cand["scenario_type"],
            "trigger_lidar_pc_token": cand["trigger_token"],
            "excluded_from_baseline_training": cand["split"] in ("test", "val"),
        },
        # --- the event contract (taxonomy doc II-B) -----------------------
        "event": {
            "t_trig_us": cand["t_trig_us"],
            "t_term_us": cand["t_term_us"],
            "terminal_reason": cand["terminal_reason"],
            "event_duration_s": cand["event_duration_s"],
        },
        "window": {
            "scored_t0_us": cand["t0_us"],
            "scored_t1_us": cand["t1_us"],
            "recon_t0_us": cand["recon_t0_us"],
            "recon_t1_us": cand["recon_t1_us"],
            "window_duration_s": cand["window_duration_s"],
        },
        "selection": {"score": cand["score"], "score_parts": cand["score_parts"],
                      "sensor": cand.get("sensor", {})},
        # --- where each downstream stage writes ---------------------------
        "artifacts": {
            "ncore": str(work_root / "ncore" / token),
            "aux": str(work_root / "aux" / token),
            # py123d-native layout: one shared root so the city maps are
            # converted once and reused, with per-seed logs inside it.
            "arrow": str(work_root / "arrow"),
            "arrow_log": str(work_root / "arrow" / "logs"
                             / PY123D_SPLIT[cand["split"]] / token),
            "recon": str(work_root / "recon" / token),
            "export": str(work_root / "export" / token),
            "eval": str(work_root / "eval" / token),
        },
        "source_paths": {
            "nuplan_root": str(nuplan_root),
            "db": str(nuplan_root / "nuplan-v1.1" / "splits" / cand["split"] / f"{cand['log_name']}.db"),
            "sensor_root": str(nuplan_root / "nuplan-v1.1" / "sensor_blobs"),
            "maps_root": str(nuplan_root / "maps"),
        },
    }
    (seed_dir / "seed.json").write_text(json.dumps(seed, indent=2))
    return seed_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(cfg.CANDIDATES))
    ap.add_argument("--work", default=str(cfg.WORK))
    ap.add_argument("--nuplan-root", default=str(cfg.NUPLAN_ROOT))
    ap.add_argument("--pick", action="append", required=True,
                    help="family=log_name[:t_trig_us] -- repeatable")
    ap.add_argument("--stage-db", action="store_true",
                    help="copy the source .db to a local-ish staging dir (faster reads)")
    args = ap.parse_args()

    work = Path(args.work)
    seeds_root = work / "seeds"
    picked = []
    for spec in args.pick:
        fam, rest = spec.split("=", 1)
        log, _, t_trig = rest.partition(":")
        rows = [json.loads(l) for l in (Path(args.candidates) / f"{fam}.jsonl").read_text().splitlines()]
        matches = [r for r in rows if r["log_name"] == log and (not t_trig or str(r["t_trig_us"]) == t_trig)]
        if not matches:
            print(f"!! no candidate for {spec}", file=sys.stderr)
            return 1
        picked.append(max(matches, key=lambda r: r["score"]))

    for c in picked:
        d = build_seed(c, seeds_root, Path(args.nuplan_root), work)
        print(f"{c['family']:<26} {d}  win={c['window_duration_s']:.1f}s  {c['location']}")
        if args.stage_db:
            split = c["split"]
            dst = work / "nuplan_local" / "nuplan-v1.1" / "splits" / split
            dst.mkdir(parents=True, exist_ok=True)
            src = Path(args.nuplan_root) / "nuplan-v1.1" / "splits" / split / f"{c['log_name']}.db"
            if not (dst / src.name).exists():
                print(f"  staging db {src.name} ...", flush=True)
                shutil.copy2(src, dst / src.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
