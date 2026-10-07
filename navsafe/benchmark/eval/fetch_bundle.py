# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Fetch whole published bundle directories from the `full_test/` prefix of a
Hugging Face dataset repo -- everything `bundle.py --out <root> --token
<TOKEN>` wrote (four `.usdz`, `offsets/`, `arrow/`, `manifest.json`,
`README.md`), not just the raw usdz `fetch_usdz.py` pulls.

Needed because rendering off a *published* bundle -- `bundle.py --handoff`,
`serve-grpc --artifact-glob`, `NUREC_GRPC_HANDOFF` -- reads `manifest.json`
and `offsets/*.json` directly off local disk; the usdz alone are not enough.

Usage::

    navsafe fetch --token 00c1e4eb4a045f20 \\
        --out "$NAVSAFE_DATA_ROOT"
    navsafe fetch --token A --token B --out ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "c13752hz/NavSafe"
HF_PREFIX = "full_test"


def fetch(repo: str, token: str, out: Path) -> Path:
    """Download into <out>/full_test/<token>, preserving the HF layout."""
    from huggingface_hub import snapshot_download

    dest = out / HF_PREFIX / token
    got = Path(snapshot_download(
        repo, repo_type="dataset", local_dir=str(out),
        allow_patterns=[f"{HF_PREFIX}/{token}/*", f"{HF_PREFIX}/{token}/**"]))
    src = got / HF_PREFIX / token
    if not src.is_dir():
        raise SystemExit(f"{repo}: no {HF_PREFIX}/{token}/ found after download")
    if not (dest / "manifest.json").exists():
        raise SystemExit(f"{repo}: {HF_PREFIX}/{token}/ has no manifest.json -- not a bundle?")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", type=Path, required=True,
                    help="dataset root: write full_test/<token>/ here")
    ap.add_argument("--token", action="append", required=True,
                    help="scenario token; repeatable")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for t in args.token:
        print(f"[fetch-bundle] {t}", flush=True)
        dest = fetch(args.repo, t, args.out)
        n = sum(1 for _ in dest.rglob("*") if _.is_file())
        size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e9
        print(f"[fetch-bundle] {t}: {n} files, {size:.2f} GB -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
