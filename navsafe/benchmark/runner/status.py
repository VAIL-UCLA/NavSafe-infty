"""One-screen view of the NavSafe development tree.

Run inside the work pod (it reads the PVC directly)::

    python3 navsafe/benchmark/runner/status.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from navsafe.benchmark import config as cfg


STAGES = [
    # (label, path template relative to the seed's artifacts, glob?)
    ("ncore", "{ncore}/clips/{tok}/nurec_origin_offset.json", False),
    ("arrow", "{arrow_log}/ego_state_se3.arrow", False),
    ("aux", "{ncore}/clips/{tok}/{tok}.aux-meta.json", False),
    ("train", "{recon}/{tok}/checkpoints/last.ckpt", False),
    ("export", "{export}/usd-out/*.usdz", True),
    ("eval", "{eval}/*/metrics.json", True),
]


def _hit(pattern: str, is_glob: bool) -> Path | None:
    if is_glob:
        p = Path(pattern)
        hits = sorted(p.parent.parent.glob("/".join(p.parts[-2:]))) if p.parent.parent.exists() else []
        return hits[0] if hits else None
    p = Path(pattern)
    return p if p.exists() else None


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}P"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(cfg.WORK))
    args = ap.parse_args()
    work = Path(args.work)

    idx = work / "index" / "summary.json"
    if idx.exists():
        s = json.loads(idx.read_text())
        print(f"index: {s['total_runs']:,} tag runs, {len(s['types'])} scenario types, "
              f"eligible splits {s['eligible_splits']}")
    cand = work / "candidates"
    if cand.exists():
        for f in sorted(cand.glob("*.jsonl")):
            n = sum(1 for _ in f.open())
            print(f"  candidates {f.stem:<28} {n:>5}")

    seeds_root = work / "seeds"
    if not seeds_root.exists():
        return
    print(f"\n{'seed':<18}{'family':<26}{'city':<28}{'win':>6}  stages")
    print("-" * 108)
    for sd in sorted(seeds_root.iterdir()):
        sj = sd / "seed.json"
        if not sj.exists():
            continue
        seed = json.loads(sj.read_text())
        a = dict(seed["artifacts"], tok=seed["seed_id"])
        marks = []
        for label, tmpl, is_glob in STAGES:
            hit = _hit(tmpl.format(**a), is_glob)
            marks.append(f"{label}{'+' if hit else '-'}")
        print(f"{seed['seed_id']:<18}{seed['family']:<26}"
              f"{seed['provenance']['location']:<28}"
              f"{seed['window']['window_duration_s']:>5.1f}s  {' '.join(marks)}")

    print()
    for name in ("ncore", "aux", "arrow", "recon", "export", "eval"):
        d = work / name
        if not d.exists():
            continue
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"  {name:<10}{human(total):>8}")


if __name__ == "__main__":
    main()
