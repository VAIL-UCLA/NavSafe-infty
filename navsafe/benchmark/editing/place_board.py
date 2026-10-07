"""A bird's-eye board for standing a static prop by hand.

Where a sign belongs is a judgement about a junction, and judgement is what an
author has and a solver does not. V-8's ten plates were solved five times over
-- against camera framing, distance from the ego, the side the turn goes, the
carriageway polygons, terrain height -- and each rule was right on its own
terms while none of them produced a placement the reviewer accepted. Every
round cost a full render to discover that. Putting the map in front of the
person who can see the answer settled all ten in one pass.

So this is not a V-8 tool. It takes frozen recipes and emits a self-contained
page carrying what the decision needs -- the drivable surface, the lane
centrelines, the logged ego frame by frame, the hand-off pose the arcs are
measured from, the junction anchors, and the actor where it currently stands --
and gives back a ``hosts:`` block to paste into the event type.

The board reports ``arc`` along the ego route from the hand-off, ``lateral``
positive to the LEFT (the convention :meth:`Polyline.offset` documents and the
one a recipe's ``lateral`` is read in), and ``yaw_offset_deg`` measured from
the route tangent there -- the same three fields the event type's ``authored`` block
speaks, so nothing is translated between seeing and freezing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

#: Lanes further than this from the ego's own path are dropped. A bounding box
#: round a 100 m route sweeps in whole city blocks the driver never sees: on the
#: V-8 hosts that was ~600 lanes each and 3 MB of embedded JSON, against ~200
#: and under 1 MB at this radius.
NEAR_M = 45.0

#: Coordinates are rounded to this many decimals. Centimetres are below what a
#: hand placement resolves and the file is embedded in the page.
ND = 1


def project_to_route(route, xy, *, handoff_arc: float) -> Dict[str, float]:
    """``(arc, lateral, tangent)`` for a point, in the recipe's own terms.

    ``arc`` is measured from the hand-off, not from the route's start, because
    that is what ``authored: {anchor: handoff, arc: ...}`` means. ``lateral``
    is positive to the LEFT.

    The board recomputes this in JavaScript so the readout tracks the cursor;
    keeping the definition here is what lets a test pin the two together --
    a point placed at ``(arc, lateral)`` must come back as ``(arc, lateral)``.
    """
    pts = np.asarray(route.xy, np.float64)[:, :2]
    arcs = np.asarray(route.arc, np.float64)
    p = np.asarray(xy, np.float64)[:2]
    best = None
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        v = b - a
        L2 = float(v @ v) or 1e-9
        t = float(np.clip((p - a) @ v / L2, 0.0, 1.0))
        q = a + t * v
        d2 = float((p - q) @ (p - q))
        if best is None or d2 < best[0]:
            tang = math.atan2(float(v[1]), float(v[0]))
            d = p - q
            best = (d2,
                    float(arcs[i] + t * (arcs[i + 1] - arcs[i])),
                    float(-d[0] * math.sin(tang) + d[1] * math.cos(tang)),
                    tang)
    _, arc, lat, tang = best
    return {"arc": arc - handoff_arc, "lateral": lat, "tangent": tang}


def _round(points, nd: int = ND) -> List[List[float]]:
    return [[round(float(x), nd), round(float(y), nd)]
            for x, y in np.asarray(points, np.float64)[:, :2]]


def host_payload(probe, recipe: Dict[str, Any], *, after_frame: int,
                 near_m: float = NEAR_M) -> Dict[str, Any]:
    """Everything the board draws for one host."""
    from .placement.anchors import find_anchors

    pos = np.asarray(probe.ego_position, np.float64)[:, :2]
    T = len(pos)
    heading = [
        math.atan2(*(pos[min(T - 1, k + 2)] - pos[max(0, k - 2)])[::-1])
        for k in range(T)
    ]

    lanes: List[Dict[str, Any]] = []
    for lane_id, feat in probe.map_features.items():
        line = feat.get("polyline")
        if line is None:
            continue
        line = np.asarray(line, np.float64)[:, :2]
        if len(line) < 2:
            continue
        step = max(1, len(line) // 40)
        near = float(np.min(np.linalg.norm(
            line[::step][:, None, :] - pos[None, ::4, :], axis=2)))
        if near > near_m:
            continue
        poly = feat.get("polygon")
        lanes.append({
            "id": str(lane_id),
            "c": _round(line[::max(1, len(line) // 24)]),
            "p": (_round(np.asarray(poly, np.float64)[::max(1, len(poly) // 28)])
                  if poly is not None and len(poly) >= 3 else None),
        })

    route = probe.ego_route
    actor = next(iter((recipe.get("actors") or {}).values()), None)
    spawn = (actor or {}).get("spawn", {})
    return {
        "ego": [[round(float(x), 2), round(float(y), 2), round(float(h), 4)]
                for (x, y), h in zip(pos, heading)],
        "after_frame": int(after_frame),
        "lanes": lanes,
        "route": _round(np.asarray(route.xy, np.float64)),
        "route_arc": [round(float(a), 2) for a in np.asarray(route.arc, np.float64)],
        "handoff_arc": round(float(probe.anchor_arc(route, after_frame)), 2),
        "anchors": [
            {"name": a.name, "arc": round(float(a.arc_m), 1),
             "xy": [round(float(a.xy[0]), 2), round(float(a.xy[1]), 2)]}
            for a in find_anchors(probe, after_frame=after_frame)
            if a.kind.startswith("intersection")
        ],
        "sign": {
            "xy": [round(float(spawn.get("position", [0, 0, 0])[0]), 2),
                   round(float(spawn.get("position", [0, 0, 0])[1]), 2)],
            "heading": round(float(spawn.get("heading", 0.0)), 4),
            "key": str(((actor or {}).get("asset") or {}).get("registry_key", "actor")),
        },
        "ego_wl": [4.6, 2.0],
    }


def build_payload(recipe_paths: Sequence[Path], *, corpus: Path,
                  after_frame: int = 8, near_m: float = NEAR_M) -> Dict[str, Any]:
    """Payload for every recipe, keyed by host token.

    A recipe with no actors is skipped rather than drawn empty: there is
    nothing to place on it, and a board of blanks is a board nobody reads.
    """
    from .host import load_host_scenario
    from .placement.probe import HostProbe

    out: Dict[str, Any] = {}
    for path in recipe_paths:
        recipe = yaml.safe_load(Path(path).read_text()) or {}
        if not (recipe.get("actors") or {}):
            continue
        token = Path(path).stem.split(".", 1)[-1]
        root = Path(corpus) / f"{token}_20s" / "arrow"
        if not root.is_dir():
            root = Path(corpus) / token / "arrow"
        sd, _ = load_host_scenario(str(root), scene_index=0, require_map=True)
        probe = HostProbe(sd, ego_z_to_ground_m=0.0)
        probe.after_frame = int(after_frame)
        out[token] = host_payload(probe, recipe, after_frame=after_frame, near_m=near_m)
    return out


def render_board(payload: Dict[str, Any], *, title: str = "Sign Placement Board") -> str:
    """The self-contained page. The data is embedded, so the file opens
    anywhere -- no server, no data root, no cluster."""
    tpl = Path(__file__).with_name("place_board.html").read_text()
    return (tpl.replace("__TITLE__", title)
               .replace("__DATA__", json.dumps(payload, separators=(",", ":"))))


def write_board(recipe_paths: Sequence[Path], out: Path, *, corpus: Path,
                after_frame: int = 8, title: str = "Sign Placement Board") -> Path:
    payload = build_payload(recipe_paths, corpus=corpus, after_frame=after_frame)
    if not payload:
        raise ValueError(
            f"none of the {len(recipe_paths)} recipes has an actor to place; "
            f"a board needs at least one")
    out = Path(out)
    out.write_text(render_board(payload, title=title))
    return out
