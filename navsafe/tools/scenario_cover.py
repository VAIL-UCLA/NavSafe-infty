#!/usr/bin/env python3
"""Pick the Table-2 re-sweep scenario set: k scenarios per event type.

Builds an event type index from the published bundle manifests on the HuggingFace
dataset (``full_test/<token>/manifest.json``, key ``scenario_meta``), caches it,
then selects k scenarios for every event type the corpus represents.

``full_test/`` is the kept allocation — as re-cut it holds one
event type per bundle and ten bundles per event type, so selection is stratified sampling,
not set cover. ``unused_scen/`` holds the bundles moved out that day as "not in
the PDF allocation"; those are multi-event-type and are only counted under
``--include-unused``, which reproduces the older 417-manifest numbers via greedy
set cover.

    PYTHONPATH=<repo> python navsafe/tools/scenario_cover.py --k 2

Writes ``<out-dir>/leaf_index.jsonl`` (one row per bundle) and, for the chosen
k, ``<out-dir>/cover_k<k>.txt`` (one token per line, the driver's input).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

DATASET = "c13752hz/NavSafe"
KEPT_PREFIX = "full_test"
UNUSED_PREFIX = "unused_scen"
DEFAULT_OUT = Path(os.environ.get("NAVSAFE_WORK", str(Path.home() / ".cache/navsafe"))) / "table2_cover"


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def list_bundles(prefixes: Iterable[str]) -> dict[str, list[str]]:
    """Map prefix -> tokens that have a top-level manifest.json under it."""
    files = list(_api().list_repo_files(DATASET, repo_type="dataset"))
    out: dict[str, list[str]] = {}
    for prefix in prefixes:
        tokens = sorted(
            f.split("/")[1]
            for f in files
            if f.startswith(f"{prefix}/")
            and f.count("/") == 2
            and f.endswith("/manifest.json")
        )
        out[prefix] = tokens
    return out


def fetch_meta(prefix: str, token: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DATASET,
        f"{prefix}/{token}/manifest.json",
        repo_type="dataset",
    )
    manifest = json.loads(Path(path).read_text())
    meta = manifest.get("scenario_meta") or {}
    return {
        "token": token,
        "split": prefix,
        "log": manifest.get("log"),
        "duration_s": manifest.get("duration_s"),
        "leaves": list(meta.get("taxonomy_leaves") or []),
        "leaf_names": list(meta.get("taxonomy_leaf_names") or []),
        "scenario_types": list(meta.get("scenario_types") or []),
        "has_inserted_actors": meta.get("has_inserted_actors"),
        "has_scenario_meta": bool(meta),
    }


def build_index(prefixes: Iterable[str], out_path: Path, workers: int) -> list[dict]:
    per_prefix = list_bundles(prefixes)
    jobs = [(p, t) for p, tokens in per_prefix.items() for t in tokens]
    print(f"fetching {len(jobs)} manifests " + ", ".join(f"{p}={len(t)}" for p, t in per_prefix.items()))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda job: fetch_meta(*job), jobs))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def load_index(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def all_leaves() -> list[str]:
    from navsafe.benchmark.eval.scenario_taxonomy import LEAF_NAMES

    return sorted(LEAF_NAMES)


def stratified(rows: list[dict], k: int) -> list[str]:
    """Take the k lowest tokens of every event type. Deterministic, event type-balanced.

    Correct only while a bundle carries a single event type, which is what the kept
    allocation guarantees; a multi-event-type bundle would be charged to each of its
    event types and inflate the selection.
    """
    by_leaf: dict[str, list[str]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["token"]):
        for leaf in row["leaves"]:
            by_leaf[leaf].append(row["token"])
    picked: list[str] = []
    for leaf in sorted(by_leaf):
        picked.extend(by_leaf[leaf][:k])
    return sorted(set(picked))


def greedy_cover(rows: list[dict], k: int) -> tuple[list[str], dict[str, int]]:
    """Pick tokens so every event type present in ``rows`` is covered k times.

    Event types whose total frequency is below k are covered as often as the corpus
    allows; ties are broken by token for reproducibility.
    """
    freq = Counter(leaf for row in rows for leaf in row["leaves"])
    need = {leaf: min(k, count) for leaf, count in freq.items()}
    remaining = dict(need)
    by_token = {row["token"]: set(row["leaves"]) for row in rows if row["leaves"]}
    picked: list[str] = []
    while any(v > 0 for v in remaining.values()):
        best = max(
            by_token.items(),
            key=lambda kv: (sum(1 for leaf in kv[1] if remaining.get(leaf, 0) > 0), -len(kv[1]), kv[0]),
        )
        token, leaves = best
        gain = sum(1 for leaf in leaves if remaining.get(leaf, 0) > 0)
        if gain == 0:
            break
        picked.append(token)
        for leaf in leaves:
            if remaining.get(leaf, 0) > 0:
                remaining[leaf] -= 1
        del by_token[token]
    return picked, dict(freq)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=2, help="times each represented event type must appear")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh", action="store_true", help="re-fetch manifests instead of using the cached index")
    parser.add_argument("--include-unused", action="store_true", help="also count unused_scen/ bundles")
    parser.add_argument("--rows", type=int, default=16, help="Table-2 rows, for the cell count")
    parser.add_argument("--lanes", type=int, default=2, help="concurrent eval lanes, for the wall-clock estimate")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    index_path = args.out_dir / "leaf_index.jsonl"
    prefixes = [KEPT_PREFIX] + ([UNUSED_PREFIX] if args.include_unused else [])

    if args.refresh or not index_path.exists():
        rows = build_index([KEPT_PREFIX, UNUSED_PREFIX], index_path, args.workers)
    else:
        rows = load_index(index_path)
        print(f"loaded cached index {index_path} ({len(rows)} bundles)")

    rows = [row for row in rows if row["split"] in prefixes]
    no_meta = [row["token"] for row in rows if not row["has_scenario_meta"]]
    if no_meta:
        print(f"WARNING {len(no_meta)} bundles have no scenario_meta: {no_meta[:5]}")

    multi_leaf = [row["token"] for row in rows if len(row["leaves"]) > 1]
    select = (lambda k: greedy_cover(rows, k)[0]) if multi_leaf else (lambda k: stratified(rows, k))
    freq = Counter(leaf for row in rows for leaf in row["leaves"])
    picked = select(args.k)
    taxonomy = all_leaves()
    absent = [leaf for leaf in taxonomy if leaf not in freq]
    singletons = sorted(leaf for leaf, count in freq.items() if count == 1)
    unknown = sorted(leaf for leaf in freq if leaf not in taxonomy)

    print(f"\ncorpus: {len(rows)} bundles over {sorted(prefixes)}")
    print(f"selection: {'greedy set cover' if multi_leaf else 'stratified, k per leaf'}"
          f" ({len(multi_leaf)} multi-event-type bundles)")
    print(f"event types represented: {len(freq)} of {len(taxonomy)}")
    print(f"event types with no scenario ({len(absent)}): {' '.join(absent) or '-'}")
    print(f"event types with exactly one ({len(singletons)}): {' '.join(singletons) or '-'}")
    if unknown:
        print(f"event type ids not in the taxonomy: {' '.join(unknown)}")

    print("\nleaf frequency:")
    for leaf, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {leaf:<5} {count}")

    print("\ncost (per-cell wall clock 8 min incl. renderer start; 219 s is compute-only median):")
    max_k = max(freq.values(), default=0)
    for k in sorted({1, 2, 3, args.k, max_k}):
        cover = select(k)
        cells = len(cover) * args.rows
        print(
            f"  k={k:<3} {len(cover):>3} scenarios  {cells:>5} cells"
            f"  {cells * 8 / 60 / args.lanes:>6.0f} h on {args.lanes} lanes"
            f"  ({cells * 219 / 3600 / args.lanes:.0f} h compute-only)"
            f"  {len(cover) * 8.4:>6.0f} GB fetched"
        )

    corpus_tag = "417" if args.include_unused else "full_test"
    cover_path = args.out_dir / f"cover_k{args.k}_{corpus_tag}.txt"
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_text("\n".join(picked) + "\n")
    print(f"\nwrote {len(picked)} tokens for k={args.k} to {cover_path}")


if __name__ == "__main__":
    main()
