# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Upload a finished `bundle.py` output directory to the `full_test/` folder
of the `c13752hz/NavSafe` HF dataset, as ONE commit per bundle.

Input is exactly what `bundle.py --out <root> --token <TOKEN>` writes to
`<root>/<TOKEN>/`: the four `.usdz`, `offsets/`, `arrow/`, `manifest.json`,
`README.md`. All files in the bundle go into a single `create_commit` call
(`CommitOperationAdd` per file) rather than one `upload_file` call per file —
with many concurrent bundle jobs, per-file commits blew through HF's 128/hour
commit rate limit within minutes (588 pods failed on 429 before this fix).

Usage::

    python -m navsafe.benchmark.eval.upload_bundle \\
        --bundle-dir /tmp/bundles/00c1e4eb4a045f20 --token 00c1e4eb4a045f20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "c13752hz/NavSafe"
HF_PREFIX = "full_test"
INTERMEDIATE_PREFIX = "intermediate_data"


def already_published(token: str, *, repo_id: str = REPO_ID) -> bool:
    """True if full_test/<token>/manifest.json already exists on the repo.

    Checked before uploading so a resubmitted job (retry, or a bulk re-run
    over all tokens) does not redo an ~8 GB copy + Arrow conversion + upload
    for a scenario that already has a bundle. Existence of manifest.json is
    the same completion signal write_bundle() itself uses.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    return f"{HF_PREFIX}/{token}/manifest.json" in set(
        api.list_repo_files(repo_id, repo_type="dataset"))


def _bundle_files(bundle_dir: Path) -> list:
    """Every file under ``bundle_dir``, following symlinked DIRECTORIES.

    `Path.rglob` does not descend into a symlinked directory, and a bundle's
    `arrow/` routinely contains them -- `bundle_5s500.py` links `arrow/maps`
    and `arrow/logs/nuplan_test/<token>` back at the raw scenario tree rather
    than copying ~GBs twice. With `rglob` the six `.arrow` files under them
    were dropped from the commit and the upload still reported success: the
    four `.usdz` uploaded fine (`is_file()` follows symlinked FILES, just not
    directories), so the only symptom was a bundle with 9 files instead of 15.

    That bundle is unusable. `arrow/` is where the evaluator reads the
    scenario from -- the usdz carry the reconstruction and no scenario
    definition -- so the failure surfaces at load time, on someone else's
    machine, long after the upload said DONE.

    `os.walk(followlinks=True)` descends. Bundles are a fixed shallow shape
    written by our own bundler, so the cycle risk that flag carries in general
    does not arise here.
    """
    out = []
    for root, _dirs, files in os.walk(bundle_dir, followlinks=True):
        for name in files:
            p = Path(root) / name
            if p.is_file():
                out.append(p)
    return sorted(out)


def upload_bundle(bundle_dir: Path, token: str, *, repo_id: str = REPO_ID) -> int:
    """Upload every file under ``bundle_dir`` in ONE commit.

    Originally called ``api.upload_file`` per file (~16/bundle): with many
    concurrent bundle jobs that blew through HF's commit rate limit (128/hour)
    within minutes -- confirmed the hard way, 588 pods failing on 429 Too Many
    Requests, only 11/407 bundles actually completing. One `create_commit`
    with a `CommitOperationAdd` per file is the fix HF's own error message
    points at, and drops this to 1 commit/bundle regardless of file count.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    dest_prefix = f"{HF_PREFIX}/{token}"
    paths = _bundle_files(bundle_dir)
    ops = []
    for p in paths:
        rel = p.relative_to(bundle_dir)
        dest = f"{dest_prefix}/{rel.as_posix()}"
        print(f"[upload] {dest}  ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
        ops.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)))
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"publish bundle {token} ({len(ops)} files)",
    )
    return len(ops)


def delete_intermediate(token: str, *, repo_id: str = REPO_ID) -> int:
    """Remove intermediate_data/<token>/ once full_test/<token>/ has its own
    verified copy of the same usdz -- intermediate_data becomes pure
    duplication at that point. Call only after upload_bundle has succeeded.
    """
    from huggingface_hub import CommitOperationDelete, HfApi

    api = HfApi()
    prefix = f"{INTERMEDIATE_PREFIX}/{token}/"
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset") if f.startswith(prefix)]
    if not files:
        print(f"[cleanup] no files under {prefix} (already removed or never existed)")
        return 0
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in files],
        commit_message=f"cleanup: remove {INTERMEDIATE_PREFIX}/{token} (superseded by {HF_PREFIX}/{token})",
    )
    print(f"[cleanup] removed {len(files)} file(s) under {prefix}")
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", required=True, type=Path,
                    help="a bundle.py output directory, e.g. <out>/<TOKEN>")
    ap.add_argument("--token", required=True)
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--delete-intermediate", action="store_true",
                    help="after a successful upload, remove intermediate_data/<token>/ "
                         "from the same repo (full_test/<token>/ has its own copy)")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="if full_test/<token>/manifest.json already exists on the repo, "
                         "skip the upload entirely instead of overwriting it")
    args = ap.parse_args()

    if not args.bundle_dir.is_dir():
        raise SystemExit(f"not a directory: {args.bundle_dir}")
    if not (args.bundle_dir / "manifest.json").exists():
        raise SystemExit(f"{args.bundle_dir}: no manifest.json — not a bundle.py output?")

    if args.skip_if_exists and already_published(args.token, repo_id=args.repo):
        print(f"UPLOAD_BUNDLE_SKIPPED {args.token}: {args.repo}/{HF_PREFIX}/{args.token} already exists")
        return 0

    n = upload_bundle(args.bundle_dir, args.token, repo_id=args.repo)
    print(f"UPLOAD_BUNDLE_DONE {args.token}: {n} file(s) -> {args.repo}/{HF_PREFIX}/{args.token}")

    if args.delete_intermediate:
        delete_intermediate(args.token, repo_id=args.repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
