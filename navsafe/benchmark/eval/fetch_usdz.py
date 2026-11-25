# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Fetch NuRec scenario artifacts from a Hugging Face dataset repo.

A scenario's reconstruction is one ~1.4 GB `.usdz` (`<token>.usdz`, 20 s);
the first release (`--revision v1-4x5s`) has four ~2 GB `.usdz` per scenario
(`<token>s1..s4`). Every `.usdz` of the scenario is fetched, so both work.
The default is to fetch **one scenario**, and `--all` has to be asked for.
Downloads are resumable and de-duplicated by `huggingface_hub`'s cache, so
re-running costs nothing for what is already present. Fetched files are
hardlinked into a flat `<out>/<token>/<name>.usdz` layout locally.

This fetches the reconstructions only; `navsafe fetch` (`fetch_bundle.py`)
fetches whole bundles, which is what evaluation needs.

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
# published as a full bundle its usdz move to `full_test/<token>/` beside the
# manifest and Arrow. Both are searched, in this order, so a token that has
# been promoted is still fetchable by the same command.
HF_PREFIXES = ("intermediate_data", "full_test")


def scenario_index(repo: str, revision: str | None = None) -> dict[str, tuple[str, list[str]]]:
    """token -> (the top-level folder its usdz live under, their repo paths)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    found: dict[str, tuple[str, list[str]]] = {}
    for prefix in HF_PREFIXES:
        p = prefix + "/"
        for f in sorted(files):
            if f.startswith(p) and f.endswith(".usdz"):
                tok = f[len(p):].split("/")[0]
                if found.setdefault(tok, (prefix, []))[0] == prefix:
                    found[tok][1].append(f)
    return found


def fetch(repo: str, token: str, out: Path, names: list[str],
          revision: str | None = None) -> list[Path]:
    from huggingface_hub import hf_hub_download

    paths = []
    for name in names:
        print(f"[fetch] {name}", flush=True)
        got = Path(hf_hub_download(repo, name, repo_type="dataset",
                                   local_dir=str(out), revision=revision))
        # local layout stays flat (<token>/<name>.usdz), not the HF folder prefix.
        dst = out / token / got.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.hardlink_to(got)
        paths.append(dst)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=None,
                    help="dataset revision, e.g. v1-4x5s for the first release")
    ap.add_argument("--out", type=Path, default=cfg.BUNDLES,
                    help="write <token>/<name>.usdz here; defaults to "
                         "$NAVSAFE_BUNDLES")
    ap.add_argument("--token", action="append", default=None,
                    help="scenario token; repeatable")
    ap.add_argument("--all", action="store_true",
                    help="every scenario in the repo (0.4 TB; 2.2 TB for v1-4x5s)")
    ap.add_argument("--list", action="store_true", help="list tokens and exit")
    args = ap.parse_args()

    index = scenario_index(args.repo, args.revision)
    tokens = sorted(index)
    if args.list:
        print(f"{len(tokens)} scenario(s) in {args.repo}:")
        for t in tokens:
            print(f"  {t}  ({index[t][0]}, {len(index[t][1])} usdz)")
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
        got = fetch(args.repo, t, args.out, index[t][1], args.revision)
        total = sum(p.stat().st_size for p in got) / 1e9
        print(f"[fetch] {t}: {len(got)} files, {total:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
