# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Record what produced a run, next to its outputs.

Long sweeps are routinely launched from a dirty working tree, so their results
correspond to no commit and cannot be reproduced or even interpreted after the
fact. :func:`write_provenance` drops a ``run_meta.json`` into the output dir
capturing the code state (git SHA + the full working diff), the resolved
config, seeds, and the environment.

Design constraints:

* **Never fails a run.** Every git call is ``check=False`` with a fallback, and
  the whole function is exception-guarded — a pip-installed copy outside a
  repo, or a machine without git, must still evaluate.
* **Captures the diff, not just the SHA.** A dirty tree's SHA is a lie on its
  own; ``git_diff`` is what makes a dirty-tree run reconstructable.
* **Stays local.** Unlike the ``--wandb`` config dump this writes to disk only."""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# The distributions whose exact versions the docs pin (CLAUDE.md /
# docs/installation.md) plus the heavy-stack markers. Resolved via package
# METADATA (never imported: importing torch/warp here would drag the heavy
# stack into every light caller and can have side effects).
_PINNED_DISTS = ("numpy", "warp-lang", "websockets", "torch", "isaacsim")


def _dist_version(dist: str) -> Optional[str]:
    """Installed version of ``dist`` from metadata, or ``None``."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist)
    except PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001 — provenance never fails a run
        logger.warning("could not read version of %s: %s", dist, exc)
        return None

# A whole-tree diff can be huge (vendored trees, notebooks); past this we keep
# the head and record the truncation rather than writing a multi-MB metadata
# file next to every run.
_MAX_DIFF_CHARS = 2_000_000


def content_digest(path: str | Path, patterns: tuple = ("*",),
                   *, max_files: int = 10_000) -> Dict[str, Any]:
    """Order-independent content hash of a directory (or single file).

    Campaign arms are only comparable if the *inputs* are pinned as
    tightly as the code: two runs quoting the same bank path can still
    have read different bytes. Hashes file contents, not mtimes or paths
    alone, so a re-materialised bank with identical contents compares
    equal and a silently mutated one does not.

    Returns ``{"path", "sha256", "n_files", "bytes"}``; ``sha256`` is
    ``None`` when the path is missing, which is recorded rather than
    raised — provenance must never fail a run.
    """
    import hashlib

    p = Path(path)
    try:
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(
                {f for pat in patterns for f in p.rglob(pat) if f.is_file()})
        else:
            return {"path": str(p), "sha256": None, "n_files": 0, "bytes": 0}
        if len(files) > max_files:
            files = files[:max_files]
        h = hashlib.sha256()
        total = 0
        for f in files:
            # Relative name participates so a renamed file changes the
            # digest even when its bytes do not.
            h.update(str(f.relative_to(p) if p.is_dir() else f.name).encode())
            # Chunked, never read_bytes(): banks carry multi-GB pickles and
            # a whole-file read spikes RSS by the largest file's size.
            fh_hash = hashlib.sha256()
            with open(f, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    total += len(chunk)
                    fh_hash.update(chunk)
            h.update(fh_hash.digest())
        return {"path": str(p), "sha256": h.hexdigest(),
                "n_files": len(files), "bytes": total}
    except Exception as exc:  # noqa: BLE001 — provenance never fails a run
        logger.warning("content_digest(%s) failed: %s", p, exc)
        return {"path": str(p), "sha256": None, "error": str(exc)}


def model_contract(cfg: Any = None, *, execution_mode: Optional[str] = None,
                   traffic_mode: Optional[str] = None,
                   representation: Optional[str] = None) -> Dict[str, Any]:
    """The modelling choices an arm's numbers are only valid under.

    Recorded together because comparing across any one of them silently
    is the failure mode: a brake-contract claim, a drivable-area figure
    and a route-completion rate each depend on a different subset, and a
    table that omits them reads as if they were held fixed.
    """
    return {
        "emergency_brake_mode": getattr(cfg, "emergency_brake_mode", None),
        # The brake THRESHOLD belongs here beside its mode. Both are settable
        # per-process via NEXUSSIM_EMERGENCY_BRAKE_* (see pdm_config_from_env),
        # so without it two planners that brake differently — 0.0 releases a
        # held brake to any epsilon-score proposal, 0.15 freezes a comfortable
        # low-DAC creep — hash to the SAME judge identity, and a table can
        # compare them as one teacher. Recording the mode alone captured half
        # the contract.
        "emergency_brake_threshold": getattr(cfg, "emergency_brake_threshold", None),
        "route_source": getattr(cfg, "route_source", None),
        "agent_forecast": getattr(cfg, "agent_forecast", None),
        "tracker": getattr(cfg, "tracker", None),
        "sim_dt": getattr(cfg, "sim_dt", None),
        "output_dt": getattr(cfg, "output_dt", None),
        "num_trajectory_poses": getattr(cfg, "num_trajectory_poses", None),
        "execution_mode": execution_mode,
        "traffic_mode": traffic_mode,
        # Which trajectory representation the reported scores were
        # computed on — unresolved project-wide, see
        # REPRESENTATION_MISMATCH.md, so it must travel with the numbers.
        "score_representation": representation,
    }


def _git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    """Run a git command, returning stripped stdout or ``None`` on any failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def collect_provenance(args: Any = None,
                       extra: Optional[Dict[str, Any]] = None,
                       *, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Build the provenance record (pure; does no I/O beyond git queries)."""
    root = repo_root or Path(__file__).resolve().parents[2]
    sha = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", cwd=root)
    diff = _git("diff", "HEAD", cwd=root)
    truncated = False
    if diff and len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS]
        truncated = True

    if args is None:
        args_dict: Dict[str, Any] = {}
    elif isinstance(args, dict):
        args_dict = dict(args)
    else:
        args_dict = {k: v for k, v in vars(args).items()
                     if not k.startswith("_")}

    record: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        # None means "not a git checkout"; distinguish that from a clean tree.
        "git_dirty": None if status is None else bool(status),
        "git_status_porcelain": status,
        "git_diff": diff,
        "git_diff_truncated": truncated,
        "argv": list(sys.argv),
        "args": args_dict,
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        # Record behavior-changing options not represented in argv,
        # restricted to project prefixes rather than the whole environment.
        "env_navsafe": {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith(("NEXUSSIM_", "NUREC_", "HYPOTHESIS_",
                             "PY123D_"))},
    }
    # Resolved versions of the documented pins (numpy==2.3.1,
    # warp-lang==1.12.0, websockets==12.0, IsaacSim 6.x) — a run whose
    # numbers disagree with another's often differs HERE, not in argv.
    record["pinned_versions"] = {d: _dist_version(d) for d in _PINNED_DISTS}
    # Backwards-compatible flat keys (pre-existing run_meta.json consumers).
    record["numpy_version"] = record["pinned_versions"]["numpy"]
    record["torch_version"] = record["pinned_versions"]["torch"]
    if extra:
        record["extra"] = extra
    return record


def write_provenance(output_dir: str | Path, args: Any = None,
                     extra: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Write ``run_meta.json`` into ``output_dir``; return its path.

    Returns ``None`` (and logs a warning) on any failure — provenance must
    never be the reason a run dies.
    """
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "run_meta.json"
        payload = collect_provenance(args, extra)
        # Write-then-rename so a crash mid-write cannot leave a truncated file
        # that later tooling would parse as authoritative.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)
        if payload.get("git_dirty"):
            logger.warning(
                "run launched from a DIRTY tree (sha=%s): the full diff is "
                "recorded in %s — this run corresponds to no commit.",
                payload.get("git_sha"), path)
        return path
    except Exception as exc:  # noqa: BLE001 — never fail the run
        logger.warning("could not write provenance to %s (%s)", output_dir, exc)
        return None


__all__ = ["collect_provenance", "write_provenance"]
