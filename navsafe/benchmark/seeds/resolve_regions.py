"""Resolve a seed's rubric regions to concrete geometry, once, and freeze them.

A rubric says ``reach(exit_lane_polygon)`` and ``clear(intersection_conflict_zone)``.
Something has to turn those names into geometry, and *when* it happens matters:

* Resolved **here**, at seed build time, the regions are part of the frozen
  contract -- auditable, diffable, and identical for every policy, regime and
  re-scoring pass.
* Resolved at eval time they would be re-derived per run, so two policies could
  silently be scored against different exit lanes.

So this writes ``regions.json`` beside ``seed.json`` and nothing downstream
re-derives it.

Two ways an event meets an intersection, and both occur in the pilot seeds:

* **The ego drives through it** (F3).  nuPlan's own ``on_intersection`` tag run
  gives the interval; the lanes occupied inside it, less the approach and
  departure lanes, are the conflict zone.  Reusing nuPlan's map-derived tag
  instead of re-deriving "is this an intersection?" keeps this consistent with
  how the event window was defined (``mining/windowing.py``).
* **The ego stops short of it** (F1 -- stationary at a light).  There is no tag
  run at all, because the ego never enters.  The intersection is then found by
  walking the route forward to the first signal-controlled lane.

Finally, and most importantly, this checks that **the logged human satisfies the
rubric's reach target**.  A seed whose own log never reaches the target would
fail every policy including a perfect one, which breaks the paper's solvability
guarantee.  That check is reported per seed and recorded in the output.

Coordinates are the recentred scenario frame (``PY123D_RECENTER=1``), the same
one the evaluator runs in; ``scenario_origin_xy`` is recorded so UTM is
recoverable.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

from navsafe.benchmark import config as cfg


# Ego poses are 20 Hz in the db; allow three dropped frames before a tagged
# stretch counts as broken (same rule as mining/windowing.py).
RUN_GAP_US = 200_000
# How far along the route to look for the intersection an ego is stopped at.
MAX_ROUTE_LOOKAHEAD = 6


def _load_scenario(seed: dict):
    """Load this seed's scenario offline -- no GPU, no IsaacSim."""
    tok = seed["seed_id"]
    arrow = Path(seed["artifacts"]["arrow"])
    log_src = Path(seed["artifacts"]["arrow_log"])
    root = Path(tempfile.mkdtemp()) / f"py123d_{tok}"
    (root / "logs" / log_src.parent.name).mkdir(parents=True)
    (root / "logs" / log_src.parent.name / tok).symlink_to(log_src)
    (root / "maps").symlink_to(arrow / "maps")
    os.environ.setdefault("PY123D_RECENTER", "1")
    from navsafe.scenario.py123d_dataset import scenario_description_by_index
    return scenario_description_by_index(root, 0)


def _lane_polys(sd) -> dict[str, np.ndarray]:
    out = {}
    for lid, feat in sd["map_features"].items():
        if not str(feat.get("type", "")).startswith("LANE"):
            continue
        poly = feat.get("polygon")
        if poly is not None and len(poly) >= 3:
            out[str(lid)] = np.asarray(poly, dtype=np.float64)[:, :2]
    return out


def _occupied_lanes(points, polys) -> list[set[str]]:
    """Lane ids whose polygon contains each point (several where lanes overlap)."""
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree
    ids = list(polys)
    shapes = [Polygon(polys[i]) for i in ids]
    tree = STRtree(shapes)
    out = []
    for x, y in points:
        p = Point(float(x), float(y))
        out.append({ids[k] for k in tree.query(p) if shapes[k].contains(p)})
    return out


def _intersection_interval(seed: dict) -> tuple[int, int] | None:
    """[t0, t1] of the ``on_intersection`` run overlapping the event, or None."""
    db = seed["source_paths"]["db"]
    t_trig, t_term = seed["event"]["t_trig_us"], seed["event"]["t_term_us"]
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = sorted(int(t) for (t,) in con.execute(
            "SELECT lp.timestamp FROM scenario_tag st "
            "JOIN lidar_pc lp ON st.lidar_pc_token = lp.token "
            "WHERE st.type = 'on_intersection'"))
        con.close()
    except Exception:
        return None
    best, i = None, 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] - rows[j] <= RUN_GAP_US:
            j += 1
        lo, hi = rows[i], rows[j]
        if hi >= t_trig and lo <= t_term:
            span = min(hi, t_term) - max(lo, t_trig)
            if best is None or span > best[2]:
                best = (lo, hi, span)
        i = j + 1
    return (best[0], best[1]) if best else None


def _arc_length(path: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def _lanes_ahead(sd, start_lanes: set[str], controlled: set[str]) -> set[str]:
    """Walk lane successors until a signal-controlled lane is found.

    This is the "ego stopped short of the intersection" case: the geometry the
    rubric cares about is ahead of everything the ego ever occupied, so it can
    only come from the lane graph, not from the trajectory.
    """
    mf = sd["map_features"]
    frontier, seen = set(start_lanes), set(start_lanes)
    for _ in range(MAX_ROUTE_LOOKAHEAD):
        nxt: set[str] = set()
        for lid in frontier:
            for s in map(str, mf.get(lid, {}).get("exit_lanes") or ()):
                if s not in seen:
                    nxt.add(s)
        if not nxt:
            break
        hit = nxt & controlled
        if hit:
            return hit
        seen |= nxt
        frontier = nxt
    return set()


def resolve(seed: dict) -> dict:
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    sd = _load_scenario(seed)
    md = sd["metadata"]
    ego = sd["tracks"][md["sdc_id"]]["state"]
    pos = np.asarray(ego["position"], dtype=np.float64)[:, :2]
    ts = np.asarray(md["ts"], dtype=np.int64)
    if ts.size != pos.shape[0]:
        raise SystemExit(f"ego steps {pos.shape[0]} != timestamps {ts.size}")

    polys = _lane_polys(sd)
    occ = _occupied_lanes(pos, polys)
    s = _arc_length(pos)
    ev = seed["event"]
    controlled = set(map(str, sd.get("dynamic_map_states", {})))

    # --- entry: where the ego was when the event triggered --------------
    pre = np.flatnonzero(ts <= ev["t_trig_us"])
    entry_lanes = set(occ[pre[-1]]) if pre.size else set(occ[0])

    # --- intersection ----------------------------------------------------
    iv = _intersection_interval(seed)
    if iv is not None:
        m_ix = (ts >= iv[0]) & (ts <= iv[1])
        traversed = set().union(*[occ[i] for i in np.flatnonzero(m_ix)]) if m_ix.any() else set()
        post = np.flatnonzero(ts > iv[1])
        exit_lanes = set(occ[post[0]]) if post.size else set(occ[-1])
        # Drop the approach and departure lanes: the ego is inside them at the
        # interval's edges, and leaving them in would make `in_conflict_zone`
        # true while merely approaching, so `clear` could never hold.
        ix_lanes = traversed - entry_lanes - exit_lanes
        source = "on_intersection_tag"
    else:
        # Never entered one: the ego is stopped short. Find it on the graph.
        ix_lanes = _lanes_ahead(sd, entry_lanes, controlled)
        exit_lanes = set()
        for lid in ix_lanes:
            exit_lanes |= set(map(str, sd["map_features"].get(lid, {}).get("exit_lanes") or ()))
        source = "route_lookahead"

    ix_lanes &= set(polys)
    exit_lanes &= set(polys)

    # The conflict zone is the ego's own corridor through the junction, not the
    # whole junction. `clear` asks whether a *conflicting* agent shared the
    # space the ego needed; scoring the entire intersection would count every
    # car legitimately crossing in its own lane and no signalised traversal
    # could ever pass.
    conflict_lanes = ix_lanes

    # --- stopline as an arc position along the logged path ---------------
    # An arc position, not a line segment: distances then stay signed and
    # monotone even for a policy that deviates, because they are measured by
    # projecting onto this path.
    ix_poly = unary_union([Polygon(polys[i]) for i in ix_lanes]) if ix_lanes else None
    s_entry = s_exit = float("nan")
    inside = np.zeros(len(pos), dtype=bool)
    if ix_poly is not None and not ix_poly.is_empty:
        inside = np.array([ix_poly.contains(Point(float(x), float(y))) for x, y in pos])
        if inside.any():
            idx = np.flatnonzero(inside)
            s_entry, s_exit = float(s[idx[0]]), float(s[idx[-1]])
        else:
            # The log stops short: the stopline is where the path comes closest.
            d = np.array([ix_poly.distance(Point(float(x), float(y))) for x, y in pos])
            s_entry = float(s[int(np.argmin(d))])

    # Restrict the corridor to the intersection lanes the ego actually occupies.
    if inside.any():
        occupied_ix = set().union(*[occ[i] for i in np.flatnonzero(inside)]) & ix_lanes
        if occupied_ix:
            conflict_lanes = occupied_ix

    # --- signal governing entry ------------------------------------------
    # It must be the light facing the EGO's approach. At a four-way junction
    # the lights controlling cross traffic also sit on intersection lanes, and
    # picking one of those reads red while the ego's own light is green -- which
    # is how the logged human first "ran a red" in every pilot seed.
    # Preference order: the approach lane, then a route lane, then a lane the
    # ego actually occupies inside the junction.
    route_ids = [str(x) for x in (md.get("route_lane_ids") or [])]
    ego_ix = set().union(*[occ[i] for i in np.flatnonzero(inside)]) if inside.any() else set()
    signal = None
    for pool in (entry_lanes, set(route_ids), ego_ix & ix_lanes):
        hit = sorted(pool & controlled)
        if hit:
            signal = hit[0]
            break

    # --- solvability of the reach target, by the log itself ---------------
    # If the logged human never reaches it, no policy can, and the seed would
    # fail everything for reasons that are not the policy's.
    exit_poly = unary_union([Polygon(polys[i]) for i in exit_lanes]) if exit_lanes else None
    scored = (ts >= seed["window"]["scored_t0_us"])
    log_reaches_exit = bool(exit_poly is not None and not exit_poly.is_empty and any(
        exit_poly.contains(Point(float(x), float(y)))
        for (x, y), m in zip(pos, scored) if m))
    log_enters_intersection = bool(inside[scored].any()) if inside.size else False

    def union_xy(lane_ids):
        if not lane_ids:
            return []
        u = unary_union([Polygon(polys[i]) for i in lane_ids if i in polys])
        g = u.convex_hull if u.geom_type != "Polygon" else u
        return [[round(float(x), 3), round(float(y), 3)] for x, y in g.exterior.coords]

    return {
        "regions_version": "0.3.0",
        "seed_id": seed["seed_id"],
        "family": seed["family"],
        "frame": "scenario_recentred",
        "scenario_origin_xy": list(md.get("scenario_origin_xy") or []),
        "source": source,
        "lanes": {
            "entry": sorted(entry_lanes),
            "intersection": sorted(ix_lanes),
            "conflict_zone": sorted(conflict_lanes),
            "exit": sorted(exit_lanes),
            "route": [str(x) for x in (md.get("route_lane_ids") or [])],
        },
        "signal": {"controlled_lane": signal} if signal else None,
        "path": {
            "xy": [[round(float(x), 3), round(float(y), 3)] for x, y in pos],
            "ts_us": [int(t) for t in ts],
            "s_m": [round(float(v), 3) for v in s],
            "s_intersection_entry_m": None if s_entry != s_entry else round(s_entry, 3),
            "s_intersection_exit_m": None if s_exit != s_exit else round(s_exit, 3),
        },
        "polygons": {
            "intersection": union_xy(ix_lanes),
            "conflict_zone": union_xy(conflict_lanes),
            "exit_lane": union_xy(exit_lanes),
        },
        # The seed-level solvability record. A rubric whose reach target the log
        # never achieves is a rubric/seed mismatch, not a hard scenario.
        "log_check": {
            "log_reaches_exit": log_reaches_exit,
            "log_enters_intersection": log_enters_intersection,
            "n_frames_in_intersection": int(inside.sum()),
        },
        "counts": {"lanes_in_map": len(polys), "signals_in_map": len(controlled)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(cfg.SEEDS))
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    want = {s for s in args.only.split(",") if s}
    rc = 0
    for sd_dir in sorted(Path(args.seeds).iterdir()):
        if not (sd_dir / "seed.json").exists() or (want and sd_dir.name not in want):
            continue
        out = sd_dir / "regions.json"
        if out.exists() and not args.force:
            print(f"{sd_dir.name}: regions.json exists (use --force)")
            continue
        seed = json.loads((sd_dir / "seed.json").read_text())
        try:
            r = resolve(seed)
        except Exception as e:  # noqa: BLE001
            print(f"{sd_dir.name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
            continue
        out.write_text(json.dumps(r, indent=2))
        L, C = r["lanes"], r["log_check"]
        flag = "" if C["log_reaches_exit"] else "   <-- LOG NEVER REACHES EXIT"
        print(f"{sd_dir.name}  [{r['family']}]  src={r['source']}")
        print(f"    entry={len(L['entry'])} ix={len(L['intersection'])} exit={len(L['exit'])}"
              f"  signal={(r['signal'] or {}).get('controlled_lane', '-')}"
              f"  s_entry={r['path']['s_intersection_entry_m']}"
              f"  in_ix_frames={C['n_frames_in_intersection']}{flag}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
