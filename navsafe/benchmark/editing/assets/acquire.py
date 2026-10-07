# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Acquisition: make the PLY a recipe names actually exist.

Each source needs a different chain, and only some of them can run here:

``procedural``     the converter builds it outright (cones, sign plates). Runs
                   offline, no network, no GPU.
``composed``       merged from parts already in the registry. Runs offline.
``urbanverse``     needs the UrbanVerse SDK and the network to fetch the GLB,
                   then ``convert_mesh_to_3dgs`` to turn it into a splat.
``assetharvester`` needs a GPU and an NCore clip containing the object. Not a
                  step this module can take — it prints what to run.
``host``           nothing to acquire; the actor keeps its baked gaussians.

By default this **plans** rather than runs: it prints the exact command for each
missing asset and touches nothing. Acquisition downloads third-party assets and
writes into a shared library, so it happens when a human asks for it, not as a
side effect of baking a recipe.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from navsafe.benchmark.editing.assets.registry import AssetEntry, AssetError, AssetRegistry
from navsafe.errors import NavSafeError

logger = logging.getLogger(__name__)

_CONVERTER = "navsafe.tools.convert_mesh_to_3dgs"
_DOWNLOADER = "navsafe.tools.download_targeted_assets"


class AcquireError(NavSafeError, ValueError):
    """The asset cannot be acquired here."""


def plan_acquisition(entry: AssetEntry, registry: AssetRegistry) -> Dict[str, Any]:
    """What it would take to put ``entry``'s PLY on disk.

    Returns:
        ``{"key", "source", "runnable_here", "steps": [[argv], ...], "note"}``.
        ``runnable_here`` is False when the chain needs the network, a GPU, or a
        human decision — the steps are still printed so the person who can run
        them has the exact command.
    """
    out: Dict[str, Any] = {
        "key": entry.key,
        "source": entry.source,
        "ply": entry.ply,
        "runnable_here": False,
        "steps": [],
        "note": "",
    }
    if entry.source == "host":
        out["note"] = (
            "nothing to acquire: this actor keeps its baked gaussians. Use op 'relocate' "
            "with keep_appearance."
        )
        out["runnable_here"] = True
        return out
    if entry.present:
        out["note"] = "already on disk"
        out["runnable_here"] = True
        return out
    if not entry.ply:
        raise AcquireError(f"asset {entry.key!r} declares no output `ply` path")

    if entry.source == "procedural":
        kind = str(entry.procedural.get("kind", ""))
        flag = {"cone": "--procedural-cone", "sign": "--procedural-sign"}.get(kind)
        if not flag:
            raise AcquireError(
                f"asset {entry.key!r}: procedural.kind must be 'cone' or 'sign', got {kind!r}"
            )
        argv = [sys.executable, "-m", _CONVERTER, flag, entry.ply]
        height = entry.procedural.get("target_height") or (
            entry.dims[2] if entry.dims else None
        )
        if height:
            argv += ["--target-height", str(float(height))]
        if entry.procedural.get("yaw_deg"):
            argv += ["--yaw-deg", str(float(entry.procedural["yaw_deg"]))]
        out["steps"].append(argv)
        out["runnable_here"] = True
        return out

    if entry.source == "composed":
        missing = [
            part.get("asset")
            for part in (entry.compose.get("parts") or [])
            if part.get("asset") and not registry.get(part["asset"]).present
        ]
        if missing:
            out["note"] = f"parts not on disk yet: {missing} — acquire those first"
            return out
        out["steps"].append(
            [sys.executable, "-m", "navsafe.benchmark.editing.cli", "assets", "compose", entry.key]
        )
        out["runnable_here"] = True
        return out

    if entry.source == "urbanverse":
        if not entry.uid:
            out["note"] = (
                "no UrbanVerse uid recorded — pick one from the library first "
                "(navsafe.tools.download_urbanverse_sdk), then record it here so the "
                "recipe can pin the asset by identity rather than by name"
            )
            return out
        glb = str(Path(entry.ply).with_suffix(".glb"))
        out["steps"] = [
            [sys.executable, "-m", _DOWNLOADER, "--uid", entry.uid, "--out", glb],
            [sys.executable, "-m", _CONVERTER, glb, entry.ply]
            + (["--target-height", str(entry.dims[2])] if entry.dims else []),
        ]
        out["note"] = "needs the UrbanVerse SDK and network access"
        return out

    if entry.source == "assetharvester":
        out["note"] = (
            "reconstruct from an NCore clip that contains the object: this needs a GPU and "
            "the asset-harvester pipeline, which produces gaussians.ply directly. Copy the "
            "result to the path above and re-run `assets status`."
        )
        return out

    raise AcquireError(f"asset {entry.key!r}: unhandled source {entry.source!r}")


def acquire_asset(
    key: str,
    registry: AssetRegistry,
    *,
    execute: bool = False,
    cwd: Optional["str | Path"] = None,
) -> Dict[str, Any]:
    """Plan (and optionally run) the chain that puts one asset on disk.

    Args:
        key: registry key.
        registry: the loaded registry.
        execute: actually run the steps. Off by default — acquisition fetches
            third-party assets and writes into a shared library.
        cwd: working directory for the steps.

    Returns:
        The plan, with ``"executed"`` and ``"ok"`` added when ``execute``.
    """
    entry = registry.get(key)
    plan = plan_acquisition(entry, registry)
    if not execute or not plan["steps"]:
        return plan
    if not plan["runnable_here"]:
        raise AcquireError(
            f"asset {key!r} cannot be acquired here: {plan['note']}. The steps are printed "
            f"so they can be run where they belong."
        )
    Path(entry.ply).parent.mkdir(parents=True, exist_ok=True)
    for argv in plan["steps"]:
        logger.info("acquire_asset: %s", shlex.join(argv))
        result = subprocess.run(argv, cwd=str(cwd) if cwd else None)
        if result.returncode != 0:
            plan.update({"executed": True, "ok": False, "failed_step": shlex.join(argv)})
            return plan
    plan.update({"executed": True, "ok": Path(entry.ply).is_file()})
    return plan


def acquire_missing(
    registry: AssetRegistry, *, leaf: Optional[str] = None, execute: bool = False
) -> List[Dict[str, Any]]:
    """Plan acquisition for every declared-but-missing asset (optionally one event type's)."""
    entries = registry.missing()
    if leaf:
        entries = [e for e in entries if leaf in e.leaves]
    plans = []
    for entry in entries:
        try:
            plans.append(
                acquire_asset(entry.key, registry, execute=execute)
                if execute
                else plan_acquisition(entry, registry)
            )
        except (AcquireError, AssetError) as exc:
            plans.append(
                {"key": entry.key, "source": entry.source, "runnable_here": False,
                 "steps": [], "note": f"ERROR: {exc}"}
            )
    return plans
