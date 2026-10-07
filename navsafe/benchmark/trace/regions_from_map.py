"""Resolve a scenario's junction regions from the bundle alone.

:mod:`navsafe.benchmark.seeds.resolve_regions` is the canonical resolver, and it
stays canonical: it freezes ``regions.json`` beside a seed at build time, which
is what makes two policies provably score against the same exit lane.  But it
needs the nuPlan sqlite db to find the ``on_intersection`` tag run, and a
NavSafe *bundle* carries no db back-reference -- its manifest says so outright
("offsets from arrow imu_se3, not nuPlan db").  Pointed at a bundle it falls
through to ``route_lookahead``, which walks the route graph to the first
signal-controlled lane; on the V-8 hosts that guessed a junction the logged ego
never enters (0 of 200 frames inside the polygon it returned).

Everything the resolution actually needs is in the bundle:

* the **arrow map** (``arrow/maps/nuplan/*.arrow``) carries real lane polygons
  and each lane's entry/exit topology, and
* the **logged ego** says which of those lanes it drove.

A *junction lane* is one with more than one entry or more than one exit -- the
same property the map itself uses to mark a merge or a diverge, so this is read
off the map rather than inferred from geometry.  The *exit lane* is the junction
lane the ego occupies at the end of its manoeuvre.  On a V-8 host the logged
manoeuvre IS the turn the prohibitory plate forbids, so that lane is exactly the
region a compliant policy must stay out of.

This module derives regions and never freezes them, which is a real weakening of
the guarantee ``resolve_regions`` gives: two runs of the same scenario resolve
independently.  They are deterministic functions of the bundle -- same map, same
logged ego, same answer -- so the exposure is a bundle changing under a
comparison, not run-to-run drift.  Where a frozen ``regions.json`` exists it
should be preferred; this is the fallback that lets a non-seed scenario be
scored on region predicates at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _is_junction_lane(feat: Mapping[str, Any]) -> bool:
    """A lane the map itself marks as a merge/diverge."""
    return (len(feat.get("entry_lanes") or ()) > 1
            or len(feat.get("exit_lanes") or ()) > 1)


def _lane_polys(sd: Mapping) -> dict[str, np.ndarray]:
    """``lane_id -> boundary polygon``, for LANE* features that have one."""
    out: dict[str, np.ndarray] = {}
    for lid, feat in (sd.get("map_features") or {}).items():
        if not str(feat.get("type", "")).startswith("LANE"):
            continue
        poly = feat.get("polygon")
        if poly is None:
            continue
        arr = np.asarray(poly, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 3:
            out[str(lid)] = arr[:, :2]
    return out


def _occupied(pos: np.ndarray, polys: Mapping[str, np.ndarray]) -> list[set[str]]:
    """Which lane polygons contain the ego at each frame."""
    from shapely.geometry import Point, Polygon

    shp = {lid: Polygon(p) for lid, p in polys.items()}
    out: list[set[str]] = []
    for p in pos:
        pt = Point(float(p[0]), float(p[1]))
        out.append({lid for lid, g in shp.items() if g.contains(pt)})
    return out


#: Junction regions marked by hand on the BEV board, keyed by token.
#: ``{token: {"x": float, "y": float, "r": float}}`` in world coordinates.
#:
#: Automated resolution failed on these hosts in three distinct ways, each of
#: which looked right until it was drawn: lane topology returned whole 90 m
#: streets (a nuPlan lane IS a street, not a junction); pairwise lane overlap
#: returned nothing where the junction lanes only abut; crosswalk spread
#: straddled TWO junctions on 14c0a657, whose route crosses more than one. The
#: sign placements above were settled the same way and for the same reason --
#: judgement is what an author has and a solver does not.
HAND_MARKED: dict[str, dict[str, float]] = {
    # Marked on the BEV board. Radii run 4.6-9.4 m: a junction box,
    # not a street. Every automatic attempt returned tens of metres because a
    # nuPlan lane IS a street -- which is how `intersection` came to cover the
    # whole south-west arm on 14c0a657 and report "never entered" for a policy
    # that drove straight through the junction.
    "0980dd9cdc0e543a": {"x": 7.1,   "y": -4.4,  "r": 7.0},
    "14c0a657ac3e5bb7": {"x": 25.4,  "y": -6.6,  "r": 9.4},
    "61b3149306a3501a": {"x": 0.9,   "y": -25.6, "r": 4.6},
    "67eb77e9004a503f": {"x": -10.8, "y": 6.6,   "r": 5.6},
    "7c00774138ae5805": {"x": -5.6,  "y": -8.7,  "r": 8.4},
    "8decf23c348f5b65": {"x": -4.6,  "y": -9.1,  "r": 6.7},
    "b66f23071e675170": {"x": 1.7,   "y": -10.4, "r": 6.5},
    "c8600743d9e050a9": {"x": 4.8,   "y": -17.3, "r": 7.7},
    "e1526251bd0a53ee": {"x": 4.9,   "y": -9.5,  "r": 5.3},
    "e9567d54464052f7": {"x": 31.1,  "y": -17.7, "r": 5.7},
}

#: ``token -> host.scene`` for the V-8 hosts, read off their frozen recipes.
#: A scenario knows only its recon UUID; the marks are keyed by token.
SCENE_BY_TOKEN: dict[str, str] = {
    "0980dd9cdc0e543a": "90cb1e01-9201-53ed-93fd-0dc868cc9152",
    "14c0a657ac3e5bb7": "de7fda37-74e5-59c4-b7e2-a240794ad4e1",
    "61b3149306a3501a": "c4fca2ce-14f2-5d2c-b16e-f4f0d2275d70",
    "67eb77e9004a503f": "2af994f6-b988-5d6d-8404-75186bb2fe05",
    "7c00774138ae5805": "fda6acb4-3ce3-5767-b103-f9e05a5ec9c7",
    "8decf23c348f5b65": "1cdc7ebc-d838-5230-b027-b7a377e98fb1",
    "b66f23071e675170": "891cfde3-b89f-5bb3-a69d-c720d3978bc7",
    "c8600743d9e050a9": "7c0eb2a8-1ede-52b3-a5b5-ee9d696f73b9",
    "e1526251bd0a53ee": "adc57f77-c87d-52e5-9bb5-4520abd94159",
    "e9567d54464052f7": "51c441df-1d4e-546e-8750-78f07c0951f8",
}


def _hand_marked(sd: Mapping) -> dict[str, Any] | None:
    """The marked junction for this scenario, as a region, or ``None``."""
    # A bundle identifies itself by `scenario_id`, which is the recon's UUID
    # (`de7fda37-...`), NOT the benchmark token (`14c0a657...`) the marks are
    # keyed by. Nothing in the scenario carries the token, so the recipe's
    # `host.scene` is what bridges them -- SCENE_BY_TOKEN is read off the ten
    # V-8 recipes. A token key is still accepted, for a producer exposing one.
    md = sd.get("metadata", {}) or {}
    cand = " ".join(str(md.get(k) or "") for k in
                    ("token", "scenario_token", "scenario_id", "log_name", "scenario_name"))
    m = None
    for tok, region in HAND_MARKED.items():
        scene = SCENE_BY_TOKEN.get(tok, "")
        if (tok and tok in cand) or (scene and scene in cand):
            m = region
            break
    if not m:
        return None
    import math
    cx, cy, r = float(m["x"]), float(m["y"]), float(m["r"])
    ring = [[cx + r * math.cos(t), cy + r * math.sin(t)]
            for t in (i * math.tau / 48 for i in range(49))]
    return {"polygons": {"intersection": ring}, "source": "hand_marked"}


def resolve_from_scenario(sd: Mapping) -> dict[str, Any] | None:
    """Junction + exit-lane polygons for ``sd``, or ``None`` when unresolvable.

    ``None`` means "not checked" and must be propagated as such: a scenario with
    no map, or one whose logged ego never touches a junction lane, has to leave
    the region columns absent rather than emit ``False``, which a predicate
    would read as "checked, and the ego stayed out".
    """

    # A hand-marked junction is checked FIRST and returns immediately. It exists
    # precisely because the automatic resolution is unreliable here, so letting
    # any of that code path's `return None` guards run ahead of it would discard
    # the one answer a reviewer has actually looked at. The exit lane is still
    # derived below when the mark is absent.
    hm = _hand_marked(sd)
    if hm:
        auto = _resolve_auto(sd)
        ex = (auto or {}).get('polygons', {}).get('exit_lane')
        if ex:
            hm['polygons']['exit_lane'] = ex
            hm['lanes'] = (auto or {}).get('lanes', {})
        return hm

    # No mark for this host: fall back to deriving both regions from the map.
    return _resolve_auto(sd)


def _resolve_auto(sd: Mapping) -> dict[str, Any] | None:
    """Junction box + exit lane derived from the map, with no hand mark."""
    if not sd.get("map_features"):
        return None
    ego_id = (sd.get("metadata", {}) or {}).get("sdc_id") or sd.get("sdc_id")
    if not ego_id:
        return None
    try:
        pos = np.asarray(sd["tracks"][ego_id]["state"]["position"],
                         dtype=np.float64)[:, :2]
    except (KeyError, TypeError, ValueError):
        return None
    if pos.shape[0] < 2:
        return None

    polys = _lane_polys(sd)
    if not polys:
        return None
    occ = _occupied(pos, polys)
    mf = sd["map_features"]

    junction = {lid for frame in occ for lid in frame
                if _is_junction_lane(mf.get(lid, {}))}
    if not junction:
        return None

    entry = set(occ[0])
    # The manoeuvre's destination: the junction lane(s) the ego ends in. Falling
    # back to the last occupied lane keeps a scenario whose final pose has
    # already left the junction from resolving to nothing.
    exit_lanes = {lid for lid in occ[-1] if lid in junction} or set(occ[-1])
    # Drop the approach: the ego sits in it before the event, and leaving it in
    # would make `in_intersection` true while merely approaching.
    ix_lanes = (junction - entry) or junction

    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def _merge(ids: Sequence[str]) -> list[list[float]]:
        geoms = [Polygon(polys[l]) for l in ids if l in polys]
        if not geoms:
            return []
        u = unary_union(geoms)
        g = max(u.geoms, key=lambda x: x.area) if hasattr(u, "geoms") else u
        return [[float(x), float(y)] for x, y in g.exterior.coords]

    # The junction BOX, not the junction lanes. A lane the map marks as a
    # merge/diverge can be 60 m of approach road -- on 14c0a657 that made
    # `intersection` a long ribbon down the south-west arm, so an ego that
    # drove straight through the actual junction scored 0 frames inside it and
    # the event type reported "never entered" for a policy that plainly had.
    #
    # A real junction is where lanes CROSS, so take the pairwise overlaps of
    # the junction lanes and union those. An approach lane overlaps nothing and
    # drops out on its own; the crossing region survives. Same principle as
    # seeds/resolve_regions.py restricting its corridor to the lanes the ego
    # actually occupies, resolved geometrically because a bundle has no
    # `on_intersection` tag run to slice by.
    def _merge_overlaps(ids, polys) -> list[list[float]]:
        """Union of the pairwise intersections -- the crossing region."""
        from itertools import combinations
        geoms = [Polygon(polys[l]) for l in ids if l in polys]
        geoms = [g for g in geoms if g.is_valid and not g.is_empty]
        overlaps = []
        for a, b in combinations(geoms, 2):
            if a.intersects(b):
                inter = a.intersection(b)
                if not inter.is_empty and inter.area > 1.0:
                    overlaps.append(inter)
        if not overlaps:
            return []
        u = unary_union(overlaps)
        g = max(u.geoms, key=lambda x: x.area) if hasattr(u, "geoms") else u
        if g.is_empty or not hasattr(g, "exterior"):
            return []
        return [[float(x), float(y)] for x, y in g.exterior.coords]

    # A hand-marked junction wins: it is the only one of the four resolutions
    # tried here that a reviewer has actually looked at.
    ix_poly = (_merge_overlaps(sorted(ix_lanes), polys)
               or _merge(sorted(ix_lanes)))
    # anywhere along it is the violation, not only at its mouth.
    ex_poly = _merge(sorted(exit_lanes))
    if not ix_poly or not ex_poly:
        return None

    return {
        "polygons": {"intersection": ix_poly, "exit_lane": ex_poly},
        "lanes": {"entry": sorted(entry), "junction": sorted(junction),
                  "exit": sorted(exit_lanes)},
        "source": "bundle_lane_occupancy",
    }


def region_flags(pos: np.ndarray,
                 regions: Mapping[str, Any] | None) -> dict[str, list[bool]] | None:
    """Per-frame ``in_intersection`` / ``in_exit_lane`` for an ego path."""
    if not regions:
        return None
    from shapely.geometry import Point, Polygon

    polys = regions.get("polygons") or {}
    ix = polys.get("intersection")
    ex = polys.get("exit_lane")
    if not ix or not ex:
        return None
    ixg, exg = Polygon(ix), Polygon(ex)
    pts = [Point(float(p[0]), float(p[1])) for p in np.asarray(pos)[:, :2]]
    return {
        "in_intersection": [bool(ixg.contains(p)) for p in pts],
        "in_exit_lane": [bool(exg.contains(p)) for p in pts],
    }
