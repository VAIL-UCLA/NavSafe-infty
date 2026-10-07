# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Which of a clip's tracks actually moved, read from its cuboids.

Run as a subprocess under the Asset Harvester interpreter -- it imports
``ncore`` and ``asset_harvester``, which are not in the NavSafe environment and
must never become a dependency of it::

    <AH_PYTHON> -m navsafe.benchmark.harvest.motion --manifest <pai_*.json> --out <json>

**Why the harvest cares.** Replacing a *parked* car is a losing trade. The ego
drives past it, so the reconstruction sees it across a wide arc and has real
pixels for every angle a policy is likely to want; a harvested asset can only be
worse there, and measured on 2b7bf25209dd5705 it is -- a FedEx truck whose
branding is legible in the reconstruction comes back as a plain grey box truck.
A *moving* car is the opposite: it travels with or against the ego, so the
relative geometry barely changes for the whole clip and the reconstruction has
almost no angular coverage. That is the actor that falls apart the moment a
policy arrives early, late or wide, and the one worth spending an asset and a
GB of renderer VRAM on.

**Net displacement, not path length.** Cuboid annotations jitter, and summing
per-frame steps turns that jitter into metres: measured on one 5 s window, a
parked car accumulated 3.8 m of path while its start and end differed by 0.7 m,
against 41.6 m and 41.4 m for a car genuinely driving through. Start-to-end
displacement separates the two cleanly; path length does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def track_motion(manifest: Path) -> dict:
    """``{track_id: {"class", "n_obs", "displacement_m", "path_m"}}`` for one clip."""
    import numpy as np
    from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4

    from asset_harvester.ncore_parser.ncore_object_parser import (
        build_tracks_from_observations,
    )

    manifest = Path(manifest)
    doc = json.loads(manifest.read_text())
    stores = [Path((manifest.parent / s["path"]).resolve())
              for s in doc.get("component_stores", [])]
    loader = SequenceLoaderV4(SequenceComponentGroupsReader(stores))
    tracks = build_tracks_from_observations(
        loader.get_cuboid_track_observations(), loader.pose_graph)

    out = {}
    for tid, track in tracks.items():
        xyz = np.asarray(track.poses)[:, :3, 3]
        if len(xyz) < 2:
            disp = path = 0.0
        else:
            disp = float(np.linalg.norm(xyz[-1] - xyz[0]))
            path = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
        out[str(tid)] = {
            "class": str(track.label_class),
            "n_obs": int(len(xyz)),
            "displacement_m": round(disp, 3),
            "path_m": round(path, 3),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m navsafe.benchmark.harvest.motion",
        description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="the clip's pai_<window>.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    motion = track_motion(Path(args.manifest))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(motion, indent=2, sort_keys=True) + "\n")
    moving = sum(1 for m in motion.values() if m["displacement_m"] > 2.0)
    print(f"{len(motion)} track(s), {moving} moved more than 2 m -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
