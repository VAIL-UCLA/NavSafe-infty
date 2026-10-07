"""Resolve NuRec reconstruction assets from scenario metadata."""
import glob
import os
from typing import Optional

_USDZ_GLOBS = ("usd-asset/pai_*.usdz", "usd-asset/*.usdz", "usd-out/last.usdz")


def resolve_usdz_path(meta: dict) -> Optional[str]:
    """Return the usdz path from ``meta``, discovering it under ``nurec_run_dir``.

    Returns ``None`` when no USDZ can be found.
    """
    existing = meta.get("nurec_usdz_path") or ""
    if existing and os.path.isfile(existing):
        return existing
    run_dir = meta.get("nurec_run_dir") or ""
    if run_dir:
        for pattern in _USDZ_GLOBS:
            hits = sorted(glob.glob(os.path.join(run_dir, pattern)))
            if hits:
                return hits[0]
    return None

