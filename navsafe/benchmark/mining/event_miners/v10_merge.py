# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""V-10 (Unsafe Merge / Entry): the roadblock-graph miner.

nuPlan tags no merge and no on-ramp, so this event type cannot be tag-mined. What
defines it is a route property:

    the ego drives OFF one road ONTO a road that two roads feed

Three things had to be learned the hard way, and each is a guard here:

1. **Roadblocks, not lanes.** nuPlan splits lanes at every segment boundary,
   so "a lane with >= 2 incoming edges" fires on 1016 of 1628 scenes. The
   lane-*group* graph does not split that way.
2. **Follow the connectors.** Every roadblock transition runs
   road -> connector -> road, so the feeding ROADS are two hops back; asking
   for plain-roadblock predecessors can never fire.
3. **A junction is not a merge.** Every junction exit is also fed by several
   roads. Two geometric guards separate them: the merge point must be outside
   an INTERSECTION polygon, and the feeding roads must converge within
   ``max_converge_angle_deg`` — an on-ramp meets at a shallow angle, a cross
   street does not.

Plus the obvious: the ego must be MOVING, and must actually cross INTO the
merge inside the window (an earlier version reported clips where the merge sat
at t+0 and the ego drove away from it).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from navsafe.benchmark.mining import nuplan_map as nm
from navsafe.benchmark.mining.event_miners.base import (
    Candidate,
    Scenario,
    register_miner,
    trained,
    window_of,
)

logger = logging.getLogger(__name__)

SAMPLE_EVERY_US = 500_000   # 2 Hz along the trajectory is plenty
EDGE_MARGIN_US = 1_000_000  # real approach and real continuation either side


def _scan_one(scn: Scenario, *, min_speed_mps: float, max_converge_angle_deg: float,
              exclude_types: List[str], data_root: str, maps_root: str) -> Candidate:
    ev: Dict[str, Any] = {"funnel": {"transitions": 0, "lt2_feeders": 0,
                                     "not_from_feeder": 0, "geometry_reject": 0,
                                     "at_window_edge": 0}}
    excluded = [t for t in exclude_types if t and t in (scn.types or "").lower()]
    try:
        db = nm.find_log_db(scn.log, data_root=data_root)
        ts, xy = nm.ego_expert_track(db, scn.t0, scn.t1)
        if len(ts) < 4:
            return Candidate(scn, False, "roadblock_merge", ev, note=f"only {len(ts)} ego poses")
        map_api, location = nm.map_api_for(db, maps_root=maps_root)
    except Exception as exc:  # noqa: BLE001 — one unreadable log must not stop a sweep
        return Candidate(scn, False, "roadblock_merge", ev, note=f"{type(exc).__name__}: {exc}")

    ev["location"] = location
    dt_s = float(ts[-1] - ts[0]) / 1e6
    speed = float(np.linalg.norm(xy[-1] - xy[0]) / max(dt_s, 1e-6))
    ev["mean_speed_mps"] = round(speed, 2)

    # sample the trajectory, remembering the last ROAD (not connector) the ego
    # was on: the crossing is road -> connector -> road, so the road it came
    # from is one sample further back than the previous sample.
    hits: List[Dict[str, Any]] = []
    last_t = None
    last_road = None
    for t, (x, y) in zip(ts, xy):
        if last_t is not None and t - last_t < SAMPLE_EVERY_US:
            continue
        last_t = int(t)
        block, is_conn = nm.roadblock_at(map_api, x, y)
        if block is None or is_conn:
            continue
        if last_road is None or str(block.id) == str(last_road.id):
            last_road = block
            continue
        prev_road, last_road = last_road, block
        ev["funnel"]["transitions"] += 1

        feeders = nm.feeding_roads(map_api, block)
        if len(feeders) < 2:
            ev["funnel"]["lt2_feeders"] += 1
            continue
        if str(prev_road.id) not in feeders:
            ev["funnel"]["not_from_feeder"] += 1
            continue
        if int(t) - int(ts[0]) < EDGE_MARGIN_US or int(ts[-1]) - int(t) < EDGE_MARGIN_US:
            ev["funnel"]["at_window_edge"] += 1
            continue
        if nm.in_intersection(map_api, x, y):
            ev["funnel"]["geometry_reject"] += 1
            continue
        heads = [h for h in (nm.block_heading(rb) for rb, _ in feeders.values()) if h is not None]
        if len(heads) < 2:
            ev["funnel"]["geometry_reject"] += 1
            continue
        spread = nm.heading_spread_deg(heads)
        if spread > max_converge_angle_deg:
            ev["funnel"]["geometry_reject"] += 1
            continue
        hits.append({
            "t_us": int(t), "t_offset_s": round((int(t) - scn.t0) / 1e6, 1),
            "merge_roadblock": str(block.id), "from_roadblock": str(prev_road.id),
            "other_feeders": sorted(set(feeders) - {str(prev_road.id)}),
            "converge_angle_deg": round(spread, 1),
            "xy_utm": [round(float(x), 1), round(float(y), 1)],
        })

    ev["hits"] = hits[:6]
    ev["n_hits"] = len(hits)
    windows = sorted({window_of(h["t_us"], scn.t0) for h in hits})
    ok = bool(hits) and speed >= min_speed_mps and not excluded
    note = ""
    if excluded:
        note = f"excluded by type {excluded}"
    elif hits and speed < min_speed_mps:
        note = f"ego too slow ({speed:.1f} < {min_speed_mps} m/s) to be merging"
    elif hits:
        h = hits[0]
        note = (f"enters {h['merge_roadblock']} from {h['from_roadblock']} at "
                f"t+{h['t_offset_s']}s, feeders converge {h['converge_angle_deg']} deg")
    return Candidate(scn, ok, "roadblock_merge", ev, windows,
                     trained(scn.token, windows) if ok else [], note)


@register_miner("roadblock_merge")
def mine(scenarios: List[Scenario], *, min_speed_mps: float = 2.0,
         max_converge_angle_deg: float = 45.0, exclude_types: List[str] = (),
         data_root: str = nm.DATA_ROOT, maps_root: str = nm.MAPS_ROOT,
         progress_every: int = 25) -> List[Candidate]:
    """Scan scenarios for the roadblock-merge signature."""
    out: List[Candidate] = []
    for i, scn in enumerate(scenarios, 1):
        cand = _scan_one(scn, min_speed_mps=min_speed_mps,
                         max_converge_angle_deg=max_converge_angle_deg,
                         exclude_types=list(exclude_types or []),
                         data_root=data_root, maps_root=maps_root)
        out.append(cand)
        if cand.qualifies:
            logger.info("%s", cand.describe())
        if progress_every and i % progress_every == 0:
            logger.info("  ... %d/%d scanned", i, len(scenarios))
    return out


__all__ = ["mine"]
