# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Composition: two assets that only exist together.

Some event types need an object the library does not contain as one mesh. R-2 is the
sharpest case — 118 bicycles against 2 cyclist-person meshes — so the rider has
to be composed onto the bike, and the result is reported as composed rather
than retrieved. R-3's ``wheelchair + user`` is the same problem.

Composition is a rigid merge in the y-up file frame: each part is scaled, yawed
about the vertical, translated, and its gaussians concatenated. That is enough
for "a person seated on a bike" and deliberately not more — no rigging, no
deformation, no pose fitting. The plausibility of the result is a reviewer
check, and it is the characteristic failure of both leaves:

    "the rider is composed onto the bike plausibly — seated, correctly scaled,
     feet at the pedals. This is the event type's characteristic failure"

so the composer prints the numbers (each part's extent and where its base
lands) rather than asserting the result is right.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from navsafe.benchmark.editing.assets.ply_io import (
    PlyError,
    bounding_dims,
    concat_gaussians,
    read_3dgs_ply,
    transform_gaussians,
    write_3dgs_ply,
)
from navsafe.errors import NavSafeError

logger = logging.getLogger(__name__)


class ComposeError(NavSafeError, ValueError):
    """The parts cannot be composed as specified."""


def compose_assets(
    parts: List[Dict[str, Any]],
    out_path: "str | Path",
    *,
    sh_policy: str = "strict",
    ground: bool = True,
) -> Dict[str, Any]:
    """Merge several 3DGS PLYs into one composed asset.

    Args:
        parts: one mapping per part::

            {"ply": "<path>",           # required
             "translate": [x, y, z],    # metres in the y-up file frame
             "yaw_deg": 0.0,            # about the vertical (+y)
             "scale": 1.0}

        out_path: where to write the composed PLY.
        sh_policy: ``strict`` | ``drop`` | ``keep`` — see
            :func:`~navsafe.benchmark.editing.assets.ply_io.transform_gaussians`.
        ground: after merging, drop the whole asset so its lowest point sits at
            ``y = 0``. Composed assets are base-origin, and a rider lifted onto
            a bike leaves the merged origin somewhere arbitrary otherwise.

    Returns:
        A report with the merged dims, each part's extent, and the output path
        — the numbers a reviewer needs to judge "seated, correctly scaled".

    Raises:
        ComposeError: fewer than two parts, or a part cannot be read.
    """
    if len(parts) < 2:
        raise ComposeError("composition needs at least two parts")

    loaded = []
    report_parts = []
    for i, spec in enumerate(parts):
        ply = spec.get("ply")
        if not ply:
            raise ComposeError(f"part {i} has no `ply`")
        try:
            gaussians = read_3dgs_ply(ply)
        except (OSError, PlyError) as exc:
            raise ComposeError(f"part {i} ({ply}): {exc}") from exc
        before = bounding_dims(gaussians)
        moved = transform_gaussians(
            gaussians,
            translate=spec.get("translate", (0.0, 0.0, 0.0)),
            yaw_deg=float(spec.get("yaw_deg", 0.0)),
            scale=float(spec.get("scale", 1.0)),
            sh_policy=sh_policy,
        )
        after = bounding_dims(moved)
        loaded.append(moved)
        report_parts.append(
            {
                "ply": str(ply),
                "gaussians": int(len(gaussians["xyz"])),
                "extent_before_lwh": [round(before[k], 3) for k in ("length", "width", "height")],
                "extent_after_lwh": [round(after[k], 3) for k in ("length", "width", "height")],
                "base_y_after": round(after["base_y"], 3),
                "translate": list(spec.get("translate", (0.0, 0.0, 0.0))),
                "yaw_deg": float(spec.get("yaw_deg", 0.0)),
                "scale": float(spec.get("scale", 1.0)),
            }
        )

    merged = concat_gaussians(loaded)
    if ground:
        base = bounding_dims(merged)["base_y"]
        merged = transform_gaussians(merged, translate=(0.0, -base, 0.0), sh_policy="keep")
    dims = bounding_dims(merged)
    path = write_3dgs_ply(merged, out_path)
    report = {
        "output": str(path),
        "gaussians": int(len(merged["xyz"])),
        "dims_lwh": [round(dims[k], 3) for k in ("length", "width", "height")],
        "base_y": round(dims["base_y"], 3),
        "parts": report_parts,
        "composed": True,
    }
    logger.info(
        "compose_assets: %d parts -> %s (%d gaussians, dims %s)",
        len(parts),
        path,
        report["gaussians"],
        report["dims_lwh"],
    )
    return report


def compose_from_registry(
    key: str,
    registry,
    *,
    out_path: Optional["str | Path"] = None,
    sh_policy: str = "strict",
) -> Dict[str, Any]:
    """Build a ``source: composed`` registry entry from its declared parts.

    The entry's ``compose`` block names the parts by registry key::

        bicycle_rider_01:
          source: composed
          ply: <library_root>/bicycle_rider_01_3dgs.ply
          compose:
            parts:
              - {asset: bicycle_01}
              - {asset: cyclist_person_01, translate: [-0.05, 0.62, 0.0], scale: 0.98}
    """
    entry = registry.get(key)
    if entry.source != "composed":
        raise ComposeError(f"asset {key!r} has source {entry.source!r}, not 'composed'")
    declared = entry.compose.get("parts") or []
    if not declared:
        raise ComposeError(f"asset {key!r} declares no compose.parts")
    parts = []
    for spec in declared:
        spec = dict(spec)
        part_key = spec.pop("asset", None)
        if part_key:
            spec["ply"] = str(registry.resolve(part_key).ply)
        parts.append(spec)
    target = Path(out_path or entry.ply)
    if not target:
        raise ComposeError(f"asset {key!r} has no output `ply` path and none was given")
    return compose_assets(
        parts, target, sh_policy=str(entry.compose.get("sh_policy", sh_policy))
    )
