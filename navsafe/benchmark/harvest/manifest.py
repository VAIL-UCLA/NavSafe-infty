# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``replace_manifest.json``: the one file the renderer reads.

One manifest per scenario, listing which logged track gets which harvested PLY.
It is deliberately **scene-agnostic**: a 20 s scenario is served as four
separate 5 s scenes and a given car appears in some of them and not others, so
the manifest names tracks and lets the renderer intersect that with what each
served scene actually holds (``get_dynamic_objects``). Anything else would need
the manifest rewritten whenever the windowing changed.

It carries no poses and no dimensions to apply. Pose comes from sim state as it
always did; the AABB sent with a replace is read off the server, which is the
only authority on the box the reconstruction actually uses. What the manifest
records beyond the path is provenance -- what was harvested, from where, how
close it ever came to the ego -- so a rendered frame can be traced back to the
run that produced the asset.

Paths are stored **relative to the manifest's own directory** and resolved
against it on read. That is what makes a bank relocatable: the harvester writes
it under ``/data/...`` on the cluster, and the same directory downloaded
into a published scenario bundle resolves to wherever the reader put it, with no
rebasing step and nothing to configure. What the RENDER SERVER receives is
always the resolved absolute path, because it opens the file itself.

Schema 1 stored absolute paths and is still readable, so older banks keep
working in place; they are simply not portable until re-written.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from navsafe.errors import NavSafeError

logger = logging.getLogger(__name__)

MANIFEST_NAME = "replace_manifest.json"
# Written instead of a manifest when a harvest ran and found nothing to replace.
# Without it, "ran and found nothing" and "still running" are the same absence,
# and leftover PLYs from an earlier, wider selection make the directory look
# half-built rather than deliberately empty.
EMPTY_MARKER = "NOTHING_TO_HARVEST"
SCHEMA_VERSION = 2


class ManifestError(NavSafeError, ValueError):
    """The manifest is absent, malformed, or names assets that are not there."""


def manifest_path(assets_dir: Path) -> Path:
    return Path(assets_dir) / MANIFEST_NAME


def write(
    assets_dir: Path,
    scene_id: str,
    entries: Dict[str, Dict[str, Any]],
    *,
    windows: Iterable[str] = (),
    provenance: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the manifest for one scenario.

    ``entries`` maps a logged track id to at least ``{"ply": <path>}``; the
    other keys (``label_class``, ``cuboids_dims``, ``min_ego_dist_m``,
    ``source_window``) are provenance and are written through unchanged.
    """
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    root = assets_dir.resolve()
    clean: Dict[str, Dict[str, Any]] = {}
    for tid, rec in sorted(entries.items()):
        ply = Path(rec["ply"]).resolve()
        if not ply.is_file():
            raise ManifestError(f"track {tid}: no PLY at {ply}")
        try:
            rel = ply.relative_to(root)
        except ValueError as exc:
            raise ManifestError(
                f"track {tid}: {ply} is outside the bank at {root}; a manifest "
                f"records paths relative to itself so the bank can be moved or "
                f"published") from exc
        clean[str(tid)] = {**rec, "ply": str(rel)}
    doc = {
        "schema": SCHEMA_VERSION,
        "scene_id": scene_id,
        "windows": list(windows),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "provenance": provenance or {},
        "assets": clean,
    }
    out = manifest_path(assets_dir)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    logger.info("wrote %s (%d asset(s))", out, len(clean))
    return out


def read(path: Path) -> Dict[str, Any]:
    """Load and validate a manifest. Raises rather than degrading silently.

    A missing asset is fatal here on purpose: the alternative is an eval that
    quietly renders half its actors from the reconstruction and half from
    harvested assets, which is two different fidelity regimes inside one
    scored episode and no way to tell from the numbers which is which.
    """
    path = Path(path)
    if path.is_dir():
        path = manifest_path(path)
    if not path.is_file():
        raise ManifestError(f"no replace manifest at {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    schema = doc.get("schema")
    if schema not in (1, SCHEMA_VERSION):
        raise ManifestError(
            f"{path}: schema {schema!r}, this code reads 1 and {SCHEMA_VERSION}")
    assets = doc.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ManifestError(f"{path}: no assets")
    # Resolve to absolute HERE, once: everything downstream (the renderer, the
    # checks below) deals in paths the server can open, while the file on disk
    # stays relocatable.
    root = path.parent.resolve()
    for rec in assets.values():
        rec["ply"] = str((root / rec["ply"]).resolve()) if schema >= 2 else str(rec["ply"])
    missing = [t for t, r in assets.items() if not Path(r.get("ply", "")).is_file()]
    if missing:
        raise ManifestError(
            f"{path}: {len(missing)} asset PLY(s) missing, e.g. "
            f"{assets[missing[0]].get('ply')!r}. "
            + ("The bank is incomplete -- a PLY named in the manifest is not in "
               "the directory beside it (an interrupted download, or a partial "
               "copy)." if schema >= 2 else
               "This is a schema-1 bank, which records ABSOLUTE paths: it only "
               "resolves on the machine that harvested it. Re-run `harvest` to "
               "rewrite it as a relocatable schema-2 bank."))
    return doc


def ply_by_track(doc: Dict[str, Any]) -> Dict[str, str]:
    """The renderer's view of a manifest: track id -> PLY path, nothing else."""
    return {str(t): str(r["ply"]) for t, r in doc["assets"].items()}


def unmatched(doc: Dict[str, Any], served_track_ids: Iterable[str]) -> List[str]:
    """Manifest tracks that no served scene holds.

    Never empty in normal operation and not an error: a manifest covers the
    whole 20 s scenario while any one served scene is 5 s of it. It IS an error
    when it is the whole manifest -- that means the ids do not match the
    reconstruction at all (wrong corpus, wrong token, re-ingested clip).
    """
    served = {str(t) for t in served_track_ids}
    return sorted(t for t in doc["assets"] if t not in served)
