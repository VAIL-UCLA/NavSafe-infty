"""Turn indexed scenario-tag runs into ranked NavSafe seed candidates.

For every eligible tag run of a requested family this:
  1. derives the event window via the family's trigger/terminal predicates
     (``windowing.py``),
  2. checks that the *reconstruction* window is fully covered by all 8 nuPlan
     cameras (both in the db and on disk -- a seed whose sensor clip has holes
     cannot be reconstructed),
  3. scores the candidate so the best ones surface first.

Output: ``candidates/<family>.jsonl`` plus a short console report.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from navsafe.benchmark.mining.windowing import (  # noqa: E402

    FAMILIES,
    US,
    build_window,
    load_ego_track,
    window_asdict,
)

from navsafe.benchmark import config as cfg

CAMERAS = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2")
ELIGIBLE_SPLITS = ("test", "val")
NOMINAL_CAM_HZ = 10.0


def sensor_coverage(con: sqlite3.Connection, blob_root: Path, log_name: str,
                    t0: int, t1: int) -> dict:
    """Per-camera frame counts over [t0, t1] + an on-disk spot check."""
    rows = con.execute(
        "SELECT c.channel, COUNT(*), MIN(i.timestamp), MAX(i.timestamp) "
        "FROM image i JOIN camera c ON i.camera_token = c.token "
        "WHERE i.timestamp BETWEEN ? AND ? GROUP BY c.channel",
        (t0, t1),
    ).fetchall()
    counts = {ch: (n, lo, hi) for ch, n, lo, hi in rows}
    dur_s = (t1 - t0) / US
    expected = max(1, int(dur_s * NOMINAL_CAM_HZ * 0.8))

    per_cam = {}
    missing_files = 0
    for cam in CAMERAS:
        n, lo, hi = counts.get(cam, (0, None, None))
        ok = n >= expected and lo is not None and lo <= t0 + US and hi >= t1 - US
        per_cam[cam] = {"n": n, "ok": bool(ok)}
        if n:
            fn = con.execute(
                "SELECT i.filename_jpg FROM image i JOIN camera c ON i.camera_token=c.token "
                "WHERE c.channel=? AND i.timestamp BETWEEN ? AND ? LIMIT 1",
                (cam, t0, t1),
            ).fetchone()
            # nuPlan's filename_jpg is already "<log>/<CHANNEL>/<hash>.jpg",
            # i.e. relative to sensor_blobs/ -- do not prefix the log again.
            if fn and not (blob_root / fn[0]).exists():
                missing_files += 1
                per_cam[cam]["ok"] = False
    n_ok = sum(1 for v in per_cam.values() if v["ok"])
    return {
        "per_camera": per_cam,
        "n_cameras_ok": n_ok,
        "complete": n_ok == len(CAMERAS) and missing_files == 0,
        "missing_files": missing_files,
        "expected_per_cam": expected,
    }


def score_candidate(w, cov, ego, runs_by_type) -> tuple[float, dict]:
    """Higher is better.  Prefers complete sensor coverage, a window in the
    sweet spot for a single-GPU reconstruction, and a scene with some traffic
    (an empty scene makes a poor interaction test)."""
    parts = {}
    parts["coverage"] = 40.0 if cov["complete"] else -100.0
    # 8-14 s is the sweet spot: long enough to contain the event + recovery,
    # short enough to reconstruct on one 3090.
    d = w.window_duration_s
    parts["duration"] = 25.0 - abs(d - 11.0) * 2.0
    # richer tag context around the trigger == a more interactive moment
    ctx = sum(
        1
        for ty, runs in runs_by_type.items()
        for s, e in runs
        if s <= w.t1_us and e >= w.t0_us
    )
    parts["context"] = min(ctx, 25)
    i0, i1 = ego.idx_at(w.t0_us), ego.idx_at(w.t1_us)
    dist = sum(
        ((ego.x[k + 1] - ego.x[k]) ** 2 + (ego.y[k + 1] - ego.y[k]) ** 2) ** 0.5
        for k in range(i0, min(i1, len(ego.x) - 1))
    )
    parts["travel_m"] = round(dist, 1)
    # a purely stationary clip gives the reconstruction no parallax
    parts["parallax"] = min(dist, 60.0) * 0.25
    return sum(v for k, v in parts.items() if k != "travel_m"), parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(cfg.INDEX))
    ap.add_argument("--nuplan-root", default=str(cfg.NUPLAN_ROOT))
    ap.add_argument("--out", default=str(cfg.CANDIDATES))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--max-per-family", type=int, default=40,
                    help="stop after this many accepted candidates per family")
    ap.add_argument("--top", type=int, default=10, help="report top N")
    args = ap.parse_args()

    root = Path(args.nuplan_root) / "nuplan-v1.1"
    blob_root = root / "sensor_blobs"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Push the split-hygiene filter into the parquet read: 3.0 M runs overall,
    # but only ~60 k are on eligible sensor-bearing logs, and materialising the
    # rest as Python objects costs GBs for nothing.
    t = pq.read_table(
        Path(args.index) / "runs.parquet",
        filters=[("has_sensor", "==", True), ("split", "in", list(ELIGIBLE_SPLITS))],
    )
    print(f"eligible runs: {t.num_rows}", flush=True)
    rows = t.to_pylist()
    del t

    # runs grouped per log, for the tag-composition terminals
    per_log_runs: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_log_runs[r["log_name"]][r["scenario_type"]].append(
            (r["start_timestamp_us"], r["end_timestamp_us"])
        )

    fams = [FAMILIES[f] for f in args.families.split(",") if f in FAMILIES]
    db_cache: dict[str, str] = {}
    for split_dir in (root / "splits").iterdir():
        if split_dir.is_dir() and split_dir.name in ELIGIBLE_SPLITS:
            for db in split_dir.glob("*.db"):
                db_cache[db.stem] = str(db)

    for fam in fams:
        cands = []
        rejects: dict[str, int] = defaultdict(int)
        pool = [
            r for r in rows
            if r["scenario_type"] in fam.nuplan_tags
            and r["has_sensor"]
            and r["split"] in ELIGIBLE_SPLITS
        ]
        # deterministic order, and spread across logs so one log cannot
        # dominate a family's candidate pool
        pool.sort(key=lambda r: (r["log_name"], r["start_timestamp_us"]))
        print(f"\n=== {fam.key}  ({fam.nuplan_tags}) : {len(pool)} eligible runs", flush=True)

        ego_cache: dict[str, object] = {}
        seen_log = defaultdict(int)
        for r in pool:
            if len(cands) >= args.max_per_family:
                break
            if seen_log[r["log_name"]] >= 3:
                rejects["log quota"] += 1
                continue
            db_path = db_cache.get(r["log_name"])
            if not db_path:
                rejects["db not found"] += 1
                continue
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                ego = ego_cache.get(r["log_name"])
                if ego is None:
                    ego = load_ego_track(con)
                    ego_cache.clear()          # one log at a time; tracks are big
                    ego_cache[r["log_name"]] = ego
                w, why = build_window(fam, r, ego, per_log_runs[r["log_name"]])
                if w is None:
                    rejects[why.split(" (")[0][:44]] += 1
                    continue
                cov = sensor_coverage(con, blob_root, r["log_name"], w.recon_t0_us, w.recon_t1_us)
                if not cov["complete"]:
                    rejects[f"sensor coverage ({cov['n_cameras_ok']}/8 cams)"] += 1
                    continue
                score, parts = score_candidate(w, cov, ego, per_log_runs[r["log_name"]])
                d = window_asdict(w)
                d["score"] = round(score, 1)
                d["score_parts"] = {k: round(v, 1) for k, v in parts.items()}
                d["sensor"] = {"n_cameras_ok": cov["n_cameras_ok"], "expected_per_cam": cov["expected_per_cam"]}
                cands.append(d)
                seen_log[r["log_name"]] += 1
            finally:
                con.close()

        cands.sort(key=lambda d: -d["score"])
        path = out / f"{fam.key}.jsonl"
        path.write_text("".join(json.dumps(c) + "\n" for c in cands))
        print(f"accepted {len(cands)} -> {path}")
        print("  rejects:", dict(sorted(rejects.items(), key=lambda kv: -kv[1])[:6]))
        for c in cands[: args.top]:
            print(
                f"  {c['score']:6.1f}  {c['log_name']}  "
                f"win={c['window_duration_s']:5.1f}s event={c['event_duration_s']:5.1f}s "
                f"travel={c['score_parts']['travel_m']:6.1f}m  {c['location']}  "
                f"[{c['terminal_reason']}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
