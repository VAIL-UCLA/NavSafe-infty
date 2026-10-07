# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Two-way road, mined from the nuPlan map (C-10, C-7, V-11).

These event types ask a question about ROAD SHAPE — is there an opposing
carriageway beside the ego, and how close is it — which the editing tier
answers with ``navsafe qualify`` on a CONVERTED host. That is the authoritative
check, but it costs an Arrow conversion per clip, and most clips are not
two-way at all. This miner asks the same question of the raw nuPlan map so a
whole pool can be filtered before anything is converted.

The predicate mirrors ``qualify.two_way_road`` deliberately:

* measured on a STRAIGHT stretch of the ego's route, never at the hand-off —
  inside a junction the "opposing lane" is another branch's connector, which is
  how an earlier C-10 put its oncoming car on a different street;
* the opposing lane must be there CONSISTENTLY along that stretch, and at a
  steady offset, because a connector drifts;
* the offset itself is reported, because it is what separates the leaves:
  C-7 wants a corridor narrow enough to force negotiation, C-10 wants the
  opposing lane close enough that crossing the centreline means a head-on, and
  a 7 m dual carriageway is neither.

What it cannot do is replace the qualify step: this reads the map's own
geometry, while qualify reads the converted scenario the eval will actually
drive. Mine here, confirm there.
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
)

logger = logging.getLogger(__name__)

STRAIGHT_WINDOW_M = 25.0
STRAIGHT_MAX_DRIFT_DEG = 15.0
STATION_STEP_M = 5.0
SEARCH_RADIUS_M = 20.0
MAX_LATERAL_M = 9.0
OPPOSING_COS = -0.7        # travel direction must genuinely oppose
MIN_HITS_FRACTION = 0.6    # present along most of the stretch, not at one point
MAX_OFFSET_SPREAD_M = 2.5  # a steady carriageway, not a drifting connector


def _lane_dir_and_lateral(lane, p: np.ndarray, heading: float):
    """(cos of angle to the ego's heading, signed +left lateral) for a lane."""
    try:
        pts = np.asarray([[s.x, s.y] for s in lane.baseline_path.discrete_path], np.float64)
    except Exception:  # noqa: BLE001 — a lane without a baseline cannot answer
        return None, None
    if len(pts) < 2:
        return None, None
    d = np.linalg.norm(pts - p[None, :], axis=1)
    i = int(np.argmin(d))
    j = min(i + 1, len(pts) - 1)
    k = max(i - 1, 0)
    tangent = pts[j] - pts[k]
    n = float(np.linalg.norm(tangent))
    if n < 1e-6:
        return None, None
    tangent = tangent / n
    cos = float(np.cos(np.arctan2(tangent[1], tangent[0]) - heading))
    rel = pts[i] - p
    lateral = float(-rel[0] * np.sin(heading) + rel[1] * np.cos(heading))
    return cos, lateral


def _scan_one(scn: Scenario, *, max_lateral_m: float, data_root: str, maps_root: str) -> Candidate:
    ev: Dict[str, Any] = {}
    try:
        db = nm.find_log_db(scn.log, data_root=data_root)
        ts, xy = nm.ego_expert_track(db, scn.t0, scn.t1)
        if len(ts) < 8:
            return Candidate(scn, False, "two_way", ev, note=f"only {len(ts)} ego poses")
        map_api, location = nm.map_api_for(db, maps_root=maps_root)
    except Exception as exc:  # noqa: BLE001
        return Candidate(scn, False, "two_way", ev, note=f"{type(exc).__name__}: {exc}")
    ev["location"] = location

    # arc + heading along the logged route
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    head = np.arctan2(*np.diff(xy, axis=0).T[::-1])
    ev["route_m"] = round(float(arc[-1]), 1)
    if arc[-1] < STRAIGHT_WINDOW_M:
        return Candidate(scn, False, "two_way", ev, note=f"route only {arc[-1]:.0f} m")

    # first straight window (the ego holds its heading)
    start_i = None
    for i in range(len(head)):
        j = int(np.searchsorted(arc, arc[i] + STRAIGHT_WINDOW_M))
        if j >= len(head):
            break
        h = np.unwrap(head[i:j + 1])
        if len(h) and float(np.max(np.abs(h - h[0]))) <= np.deg2rad(STRAIGHT_MAX_DRIFT_DEG):
            start_i = i
            break
    if start_i is None:
        return Candidate(scn, False, "two_way", ev,
                         note="ego never holds its heading for 25 m (all turning / junction)")
    ev["straight_from_m"] = round(float(arc[start_i]), 1)

    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    stations = np.arange(arc[start_i], min(arc[start_i] + STRAIGHT_WINDOW_M, arc[-1]), STATION_STEP_M)
    laterals: List[float] = []
    lane_ids: List[str] = []
    for a in stations:
        i = int(np.clip(np.searchsorted(arc, a), 1, len(head)))
        p, h = xy[i], float(head[min(i, len(head) - 1)])
        best = None
        try:
            lanes = map_api.get_proximal_map_objects(
                Point2D(*p), SEARCH_RADIUS_M, [SemanticMapLayer.LANE])[SemanticMapLayer.LANE]
        except Exception:  # noqa: BLE001
            lanes = []
        for lane in lanes:
            cos, lat = _lane_dir_and_lateral(lane, p, h)
            if cos is None or cos > OPPOSING_COS or abs(lat) > max_lateral_m:
                continue
            if best is None or abs(lat) < abs(best[0]):
                best = (lat, str(lane.id))
        if best is not None:
            laterals.append(best[0])
            lane_ids.append(best[1])

    hits = len(laterals)
    ev["stations"] = len(stations)
    ev["opposing_hits"] = hits
    ok = hits >= max(2, int(MIN_HITS_FRACTION * len(stations)))
    note = ""
    if ok:
        spread = max(laterals) - min(laterals)
        nearest = min(laterals, key=abs)
        ev["opposing_lateral_m"] = round(float(nearest), 2)
        ev["offset_spread_m"] = round(float(spread), 2)
        ev["opposing_lane_ids"] = sorted(set(lane_ids))[:4]
        if spread > MAX_OFFSET_SPREAD_M:
            ok = False
            note = (f"the nearest opposing lane drifts {spread:.1f} m across the straight "
                    f"stretch — junction connectors, not a steady carriageway")
        else:
            note = (f"opposing carriageway {abs(nearest):.1f} m away, steady (±{spread:.1f} m) "
                    f"over {STRAIGHT_WINDOW_M:.0f} m from +{ev['straight_from_m']:.0f} m")
    else:
        note = f"opposing lane at only {hits}/{len(stations)} stations — one-way here"
    windows = ["s1", "s2", "s3", "s4"] if ok else []
    return Candidate(scn, ok, "two_way", ev, windows,
                     trained(scn.token, windows) if ok else [], note)


@register_miner("two_way")
def mine(scenarios: List[Scenario], *, max_lateral_m: float = MAX_LATERAL_M,
         max_corridor_m: float = 0.0, data_root: str = nm.DATA_ROOT,
         maps_root: str = nm.MAPS_ROOT, progress_every: int = 25, **_ignored) -> List[Candidate]:
    """Scan for a steady opposing carriageway beside a straight stretch.

    Args:
        max_corridor_m: when set, only keep hosts whose opposing lane is within
            this distance — C-7's "narrow enough that two vehicles must
            negotiate", as opposed to C-10's merely-two-way.
    """
    out: List[Candidate] = []
    for i, scn in enumerate(scenarios, 1):
        cand = _scan_one(scn, max_lateral_m=max_lateral_m, data_root=data_root, maps_root=maps_root)
        if cand.qualifies and max_corridor_m:
            lat = abs(float(cand.evidence.get("opposing_lateral_m", 99)))
            if lat > max_corridor_m:
                cand.qualifies = False
                cand.trained_windows = []
                cand.note += f" — wider than the {max_corridor_m:.1f} m corridor this leaf needs"
        out.append(cand)
        if cand.qualifies:
            logger.info("%s", cand.describe())
        if progress_every and i % progress_every == 0:
            logger.info("  ... %d/%d scanned", i, len(scenarios))
    return out


__all__ = ["mine"]
