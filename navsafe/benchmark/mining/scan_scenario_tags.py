"""Scan every nuPlan ``.db`` and index its ``scenario_tag`` rows.

nuPlan tags individual ``lidar_pc`` frames (20 Hz), so a single semantic event
shows up as a *run* of consecutively tagged frames.  This scanner emits both
views:

``runs.parquet``    one row per contiguous tagged run, for **every** log --
                    this is the event-instance view the NavSafe miner consumes.
``frames/*.parquet``  one row per tagged frame, for logs that have sensor
                    blobs only (the reconstructable universe) -- this is the
                    per-``lidar_pc``-token "scenario id" view.

Run:
    python scan_scenario_tags.py --nuplan-root "$NUPLAN_ROOT" \
        --out "$NAVSAFE_WORK/index" --workers 16
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from navsafe.benchmark import config as cfg


# 20 Hz lidar_pc -> 50 ms spacing.  Allow three dropped frames before a run
# is considered broken.
RUN_GAP_US = 200_000

TAG_QUERY = """
SELECT st.type,
       hex(st.lidar_pc_token),
       hex(st.agent_track_token),
       lp.timestamp,
       hex(lp.scene_token)
FROM scenario_tag st
JOIN lidar_pc lp ON st.lidar_pc_token = lp.token
"""

RUN_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("log_name", pa.string()),
        ("location", pa.string()),
        ("map_version", pa.string()),
        ("has_sensor", pa.bool_()),
        ("scenario_type", pa.string()),
        ("run_index", pa.int32()),
        ("start_token", pa.string()),
        ("end_token", pa.string()),
        ("start_timestamp_us", pa.int64()),
        ("end_timestamp_us", pa.int64()),
        ("duration_s", pa.float32()),
        ("n_frames", pa.int32()),
        ("agent_track_token", pa.string()),
        ("scene_token", pa.string()),
    ]
)

FRAME_SCHEMA = pa.schema(
    [
        ("split", pa.string()),
        ("log_name", pa.string()),
        ("scenario_type", pa.string()),
        ("lidar_pc_token", pa.string()),
        ("timestamp_us", pa.int64()),
        ("agent_track_token", pa.string()),
        ("scene_token", pa.string()),
        ("run_index", pa.int32()),
    ]
)


def scan_one(db_path: str, split: str, has_sensor: bool):
    """Return (run_rows, frame_rows, error) for a single log db."""
    runs: list[tuple] = []
    frames: list[tuple] = []
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.execute("PRAGMA query_only=ON")
        log_row = con.execute(
            "SELECT logfile, location, map_version FROM log"
        ).fetchone()
        log_name = (log_row[0] if log_row else Path(db_path).stem) or Path(db_path).stem
        location = (log_row[1] if log_row else "") or ""
        map_version = (log_row[2] if log_row else "") or ""

        by_type: dict[str, list] = {}
        for typ, tok, agent_tok, ts, scene_tok in con.execute(TAG_QUERY):
            by_type.setdefault(typ, []).append((int(ts), tok, agent_tok or "", scene_tok or ""))
        con.close()

        for typ, rows in by_type.items():
            rows.sort()
            run_index = 0
            i = 0
            n = len(rows)
            while i < n:
                j = i
                while j + 1 < n and rows[j + 1][0] - rows[j][0] <= RUN_GAP_US:
                    j += 1
                start, end = rows[i], rows[j]
                runs.append(
                    (
                        split,
                        log_name,
                        location,
                        map_version,
                        has_sensor,
                        typ,
                        run_index,
                        start[1],
                        end[1],
                        start[0],
                        end[0],
                        (end[0] - start[0]) / 1e6,
                        j - i + 1,
                        start[2],
                        start[3],
                    )
                )
                if has_sensor:
                    for k in range(i, j + 1):
                        ts, tok, agent_tok, scene_tok = rows[k]
                        frames.append(
                            (split, log_name, typ, tok, ts, agent_tok, scene_tok, run_index)
                        )
                run_index += 1
                i = j + 1
        return runs, frames, None
    except Exception as exc:  # noqa: BLE001 - one bad db must not kill the sweep
        return [], [], f"{db_path}: {type(exc).__name__}: {exc}"


def _worker(chunk):
    out_runs, out_frames, errs = [], [], []
    for db_path, split, has_sensor in chunk:
        r, f, e = scan_one(db_path, split, has_sensor)
        out_runs.extend(r)
        out_frames.extend(f)
        if e:
            errs.append(e)
    return out_runs, out_frames, errs


def _write(rows, schema, path: Path):
    if not rows:
        return
    cols = list(zip(*rows))
    table = pa.Table.from_arrays(
        [pa.array(c, type=schema.field(i).type) for i, c in enumerate(cols)],
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuplan-root", default=str(cfg.NUPLAN_ROOT))
    ap.add_argument("--out", default=str(cfg.INDEX))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--splits", default="", help="comma list; default = all")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N dbs")
    args = ap.parse_args()

    root = Path(args.nuplan_root) / "nuplan-v1.1"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sensor_logs = set(os.listdir(root / "sensor_blobs"))
    print(f"sensor blobs: {len(sensor_logs)} logs", flush=True)

    wanted = {s for s in args.splits.split(",") if s} or None
    tasks = []
    for split_dir in sorted((root / "splits").iterdir()):
        if not split_dir.is_dir():
            continue
        if wanted and split_dir.name not in wanted:
            continue
        for db in sorted(split_dir.glob("*.db")):
            tasks.append((str(db), split_dir.name, db.stem in sensor_logs))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"dbs to scan: {len(tasks)}", flush=True)

    chunks = [tasks[i : i + args.chunk] for i in range(0, len(tasks), args.chunk)]
    t0 = time.time()
    done = 0
    all_errs: list[str] = []
    run_shard: list[tuple] = []
    frame_shard: list[tuple] = []
    frame_id = 0
    run_id = 0
    n_runs = 0
    # Flush to disk rather than accumulating ~5M run / ~25M frame rows in the
    # parent: the pod is capped at 32 GiB and Python tuples are not cheap.
    SHARD_ROWS = 2_000_000

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_worker, c) for c in chunks]
        for fut in as_completed(futs):
            runs, frames, errs = fut.result()
            run_shard.extend(runs)
            frame_shard.extend(frames)
            n_runs += len(runs)
            all_errs.extend(errs)
            done += 1
            if len(frame_shard) >= SHARD_ROWS:
                _write(frame_shard, FRAME_SCHEMA, out / "frames" / f"part-{frame_id:04d}.parquet")
                frame_shard = []
                frame_id += 1
            if len(run_shard) >= SHARD_ROWS:
                _write(run_shard, RUN_SCHEMA, out / "runs" / f"part-{run_id:04d}.parquet")
                run_shard = []
                run_id += 1
            if done % 50 == 0 or done == len(chunks):
                el = time.time() - t0
                print(
                    f"[{done}/{len(chunks)} chunks] {el:.0f}s "
                    f"runs={n_runs} frames_buf={len(frame_shard)} "
                    f"eta={el / done * (len(chunks) - done):.0f}s",
                    flush=True,
                )

    _write(frame_shard, FRAME_SCHEMA, out / "frames" / f"part-{frame_id:04d}.parquet")
    _write(run_shard, RUN_SCHEMA, out / "runs" / f"part-{run_id:04d}.parquet")

    # Single consolidated runs.parquet, streamed shard-by-shard.
    shards = sorted((out / "runs").glob("part-*.parquet"))
    if shards:
        writer = pq.ParquetWriter(out / "runs.parquet", RUN_SCHEMA, compression="zstd")
        for s in shards:
            writer.write_table(pq.read_table(s, schema=RUN_SCHEMA))
        writer.close()

    if all_errs:
        (out / "scan_errors.txt").write_text("\n".join(all_errs))
    print(f"errors: {len(all_errs)}", flush=True)
    print(f"total runs: {len(run_shard)}  elapsed {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
