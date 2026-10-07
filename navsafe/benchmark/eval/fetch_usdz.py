# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Fetch NuRec scenario artifacts from a Hugging Face dataset repo.

Each scenario is four ~2 GB `.usdz` (`<token>s1..s4`), stored under the
`intermediate_data/` folder of the dataset, so a whole repo is well over
100 GB: the default is to fetch **one scenario**, and `--all` has to be
asked for. Downloads are resumable and de-duplicated by `huggingface_hub`'s
cache, so re-running costs nothing for what is already present. Fetched
files are hardlinked into a flat `<out>/<token>/<token>sN.usdz` layout
locally, matching what `bundle.py` expects.

Usage::

    python -m navsafe.benchmark.eval.fetch_usdz --token 00c1e4eb4a045f20 \\
        --out "$NAVSAFE_BUNDLES"
    python -m navsafe.benchmark.eval.fetch_usdz --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from navsafe.benchmark import config as cfg

DEFAULT_REPO = "c13752hz/NavSafe"
# Trained-model clips start under `intermediate_data/`; once a scenario is
# published as a full bundle its four usdz move to `full_test/<token>/` beside
# the manifest and Arrow. Both are searched, in this order, so a token that has
# been promoted is still fetchable by the same command.
HF_PREFIXES = ("intermediate_data", "full_test")
SUBCLIPS = ("s1", "s2", "s3", "s4")


def scenario_index(repo: str) -> dict[str, str]:
    """token -> the top-level folder its usdz live under."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    found: dict[str, str] = {}
    for prefix in HF_PREFIXES:
        p = prefix + "/"
        for f in files:
            if f.startswith(p) and f.endswith(".usdz"):
                found.setdefault(f[len(p):].split("/")[0], prefix)
    return found


def fetch(repo: str, token: str, out: Path, prefix: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    paths = []
    for s in SUBCLIPS:
        name = f"{prefix}/{token}/{token}{s}.usdz"
        print(f"[fetch] {name}", flush=True)
        got = Path(hf_hub_download(repo, name, repo_type="dataset",
                                   local_dir=str(out)))
        # local layout stays flat (<token>/<token>sN.usdz): bundle.py and
        # everything downstream expect that, not the HF folder prefix.
        dst = out / token / got.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.hardlink_to(got)
        paths.append(dst)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", type=Path, default=cfg.BUNDLES,
                    help="write <token>/<token>sN.usdz here; defaults to "
                         "$NAVSAFE_BUNDLES")
    ap.add_argument("--token", action="append", default=None,
                    help="scenario token; repeatable")
    ap.add_argument("--all", action="store_true",
                    help="every scenario in the repo (>100 GB)")
    ap.add_argument("--list", action="store_true", help="list tokens and exit")
    args = ap.parse_args()

    index = scenario_index(args.repo)
    tokens = sorted(index)
    if args.list:
        print(f"{len(tokens)} scenario(s) in {args.repo}:")
        for t in tokens:
            print(f"  {t}  ({index[t]})")
        return 0

    wanted = tokens if args.all else (args.token or [])
    if not wanted:
        raise SystemExit("nothing to do: pass --token TOKEN, --all, or --list")
    if args.out is None:
        raise SystemExit("--out is required (or set NAVSAFE_BUNDLES)")
    unknown = [t for t in wanted if t not in tokens]
    if unknown:
        raise SystemExit(f"not in {args.repo}: {', '.join(unknown)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for t in wanted:
        got = fetch(args.repo, t, args.out, index[t])
        total = sum(p.stat().st_size for p in got) / 1e9
        print(f"[fetch] {t}: {len(got)} files, {total:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
