# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Derive the log ego's traversed-lane chain from a ScenarioDescription.

BridgeSim / navsim's PDM planner is route-conditioned: nuPlan supplies
``route_roadblock_ids`` (which ROADS to take — turn intent at each fork,
not a path, not speeds) and the planner runs Dijkstra over the on-route
lane graph. Our AV2-sensor source carries no such field (py123d reserves
a ``"scenario"`` custom modality for it and populates it only for
nuPlan-sourced logs), so the lane-graph walk in
:mod:`navsafe.policy.state.pdm_closed_planner.route` had no branch
intent and could only guess "most aligned" at every fork.

This module reconstructs the legitimate analog offline: project the log
ego's full trajectory onto the map lanes and record, in order, the lane
ids the human actually drove through. That encodes which branch was
taken at each fork — the same information a nav system supplies in a
real car — and nothing else. It is still log-derived (a *weak* oracle:
branch intent only), which is strictly less leakage than the
``gt_future`` route hint (the ego's actual future positions).

Robustness, tuned for where it matters (intersections):

* Candidate lanes must be within :data:`_CHAIN_MAX_LANE_DIST_M` of the
  ego AND heading-aligned to :data:`_CHAIN_MIN_ALIGNMENT` (within 60°).
  Intersections contain overlapping and crossing lanes that share XY —
  a bare nearest-lane pick jitters between them exactly at the forks
  this chain exists to disambiguate.
* Candidates are ranked by fit against the NEXT
  :data:`_CHAIN_LOOKAHEAD_FRAMES` of trajectory, not the single frame.
  Frame-local scoring cannot tell a straight-through lane from a turn
  connector whose start tangent is also straight — measured on scene
  05f5e760 the frame-local chain contained a −104° turn connector the
  log never drove (verified by ablation: lookahead=1 re-inserts lane
  42362683). A lane that curves away from where the ego actually goes
  next accumulates lateral offset over the window and loses. NOTE:
  the current walk happens to be robust to that specific corruption
  (its own gates reject the spurious connector); the window ranking is
  kept because ``route_lane_ids`` is emitted dataset metadata and must
  be correct for ANY consumer, not merely tolerable to today's walk.
* A lane enters the chain only after winning
  :data:`_CHAIN_MIN_DWELL_FRAMES` consecutive frames. Consecutive-id
  dedupe alone cannot remove A-B-A flicker between overlapping lanes;
  a dwell requirement can.
* Lanes that start at the same point with the same initial direction are
  competing branch alternatives, not consecutive road segments.  A long,
  gradual fork can make each alternative win the lookahead for several
  frames, defeating the dwell filter and producing A-B-A-B.  Such a run is
  collapsed to its final winner.  Real A-B-A loops are retained when B does
  not share A's origin.

The converter (:mod:`navsafe.scenario.py123d_scenario_description`)
emits the result as ``metadata["route_lane_ids"]``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from navsafe.evaluation.utils.lane_proxy import LaneProxy, build_lanes_from_scenario

# A lane farther than this from the ego at a frame is not the lane being
# driven. Matches route.py's _ROUTE_MAX_EGO_OFFSET_M.
_CHAIN_MAX_LANE_DIST_M = 5.0
# Require real alignment (within ~60°) between the ego heading and the
# lane tangent at the projection point. Perpendicular crossing lanes at
# intersections can be the *nearest* lane while the ego drives over them.
_CHAIN_MIN_ALIGNMENT = 0.5
# A lane must be the best candidate for this many consecutive frames
# before it is appended. At 10 Hz this is 0.3 s — longer than projection
# flicker between overlapping lanes, far shorter than any real traversal.
_CHAIN_MIN_DWELL_FRAMES = 3
# Same distance-vs-alignment trade-off as route.py's lane ranking.
_CHAIN_ALIGNMENT_WEIGHT = 5.0
# Trajectory window each candidate must fit: 1.5 s at 10 Hz. Long enough
# that a turn connector diverges from a straight-driving ego by several
# metres (and vice versa), short enough to switch lanes promptly.
_CHAIN_LOOKAHEAD_FRAMES = 15
# Two directed lanes whose starts coincide this closely and whose initial
# tangents agree are alternative arms of one fork.  They cannot be traversed
# consecutively: taking the second restarts at the first lane's origin.
_CHAIN_SHARED_ORIGIN_M = 1.0
_CHAIN_SHARED_ORIGIN_MIN_ALIGNMENT = 0.5

# Memoization of the derivation. The chain is a pure function of the
# scene log, but the converter re-runs on EVERY env reset (reset_to_scene
# nulls the env's scenario data, forcing a full reconversion) and the
# derivation is the dominant cost: a Python loop over frames x all lanes
# with full vertex-distance arrays. Cache keyed by stable scene identity;
# bounded FIFO so long sweeps over many scenes cannot grow unboundedly.
# All call sites are single-threaded per process (no ThreadPool around
# the loaders/converters; DDP workers are separate processes), so a
# plain dict — insertion-ordered, giving FIFO eviction — needs no lock.
_CACHE_MAX_ENTRIES = 64
_route_chain_cache: Dict[tuple, List[str]] = {}


def _cache_key(scenario_data: Dict[str, Any]) -> tuple | None:
    """Stable identity of the scene, or ``None`` when caching is unsafe.

    The converter stamps ``metadata`` with ``dataset`` / ``split`` /
    ``scenario_id`` before deriving the chain; together those name the
    source log uniquely. Frame count and number of valid ego frames are
    included defensively so a differently-trimmed reconversion of the
    same id can never alias a stale entry. Without a ``scenario_id``
    (e.g. hand-built dicts) there is no stable identity — skip caching.
    """
    metadata = scenario_data.get("metadata") or {}
    scenario_id = metadata.get("scenario_id")
    if not scenario_id:
        return None
    sdc_id = metadata.get("sdc_id")
    state = scenario_data.get("tracks", {}).get(sdc_id, {}).get("state", {})
    positions = state.get("position")
    n_frames = 0 if positions is None else int(np.asarray(positions).shape[0])
    valid = state.get("valid")
    n_valid = (
        n_frames if valid is None else int(np.asarray(valid).astype(bool).sum())
    )
    return (
        metadata.get("dataset"),
        metadata.get("split"),
        str(scenario_id),
        n_frames,
        n_valid,
    )


def derive_route_lane_ids(scenario_data: Dict[str, Any]) -> List[str]:
    """Ordered lane ids the log ego traversed, one entry per lane.

    Deterministic per scene log; results are memoized (see
    :data:`_route_chain_cache`) so repeated reconversions of the same
    scene — one per env reset — pay the derivation once per process.

    Args:
        scenario_data: ScenarioNet-style dict. Reads
            ``tracks[metadata['sdc_id']]['state']`` (position / heading /
            valid) and ``map_features`` (via
            :func:`build_lanes_from_scenario`).

    Returns:
        Lane ids in traversal order, consecutive duplicates removed.
        Empty list when the scenario has no lanes or no valid ego frames
        — callers must treat that as "no route available", never as an
        error.
    """
    key = _cache_key(scenario_data)
    if key is None:
        return _derive_route_lane_ids_uncached(scenario_data)
    cached = _route_chain_cache.get(key)
    if cached is not None:
        return list(cached)
    chain = _derive_route_lane_ids_uncached(scenario_data)
    if len(_route_chain_cache) >= _CACHE_MAX_ENTRIES:
        # FIFO eviction: drop the oldest insertion (dicts are ordered).
        _route_chain_cache.pop(next(iter(_route_chain_cache)))
    _route_chain_cache[key] = chain
    # Return a copy so a caller mutating its list cannot poison the cache.
    return list(chain)


def _derive_route_lane_ids_uncached(scenario_data: Dict[str, Any]) -> List[str]:
    """The actual derivation — see :func:`derive_route_lane_ids`."""
    lane_pairs = build_lanes_from_scenario(scenario_data)
    if not lane_pairs:
        return []
    lanes: List[LaneProxy] = [lp for lp, _poly in lane_pairs]

    metadata = scenario_data.get("metadata", {})
    sdc_id = metadata.get("sdc_id")
    if sdc_id is None:
        return []
    state = scenario_data.get("tracks", {}).get(sdc_id, {}).get("state", {})
    positions = state.get("position")
    headings = state.get("heading")
    if positions is None or headings is None:
        return []
    positions = np.asarray(positions, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[0] == 0:
        return []
    valid = state.get("valid")
    valid = (
        np.ones(positions.shape[0], dtype=bool)
        if valid is None
        else np.asarray(valid).astype(bool)
    )

    valid_xy = positions[valid, :2]
    chain: List[str] = []
    pending_id: str | None = None
    pending_count = 0
    vi = -1  # index into valid_xy of the current frame
    for t in range(positions.shape[0]):
        if not valid[t]:
            continue
        vi += 1
        window = valid_xy[vi : vi + _CHAIN_LOOKAHEAD_FRAMES]
        best_id = _best_lane_id(lanes, positions[t, :2], headings[t], window)
        if best_id is None:
            # Off-map / intersection-interior gap: no vote either way.
            # Deliberately does NOT reset the dwell counter — a one-frame
            # gap must not make the current candidate start over.
            continue
        if best_id == pending_id:
            pending_count += 1
        else:
            pending_id = best_id
            pending_count = 1
        if pending_count >= _CHAIN_MIN_DWELL_FRAMES:
            if not chain or chain[-1] != pending_id:
                chain.append(pending_id)
    return _collapse_shared_origin_alternatives(chain, lanes)


def _collapse_shared_origin_alternatives(
    chain: List[str], lanes: List[LaneProxy]
) -> List[str]:
    """Remove persistent projection oscillation between fork siblings.

    Dwell removes short nearest-lane blips, but not the measured 0ebb case:
    two turn arms share an exact origin and overlap before diverging, so the
    1.5 s lookahead selects each for several frames and emits
    ``A, B, A, B``.  Feeding that cyclic chain to the route walker makes its
    branch depend on millimetre-scale start-lane ranking and can flip the
    reference route between replans.

    Same-origin, same-direction lanes are mutually exclusive alternatives.
    Replace the preceding alternative with the later winner.  A legitimate
    loop such as ``A, B, A`` remains intact when B begins elsewhere.
    """
    lane_by_id = {lane.index: lane for lane in lanes}
    collapsed: List[str] = []
    for lane_id in chain:
        lane = lane_by_id.get(lane_id)
        if lane is None or not collapsed:
            collapsed.append(lane_id)
            continue
        previous = lane_by_id.get(collapsed[-1])
        if previous is None:
            collapsed.append(lane_id)
            continue
        start_delta = np.asarray(lane.polyline[0]) - np.asarray(
            previous.polyline[0]
        )
        previous_tangent = previous.polyline[1] - previous.polyline[0]
        lane_tangent = lane.polyline[1] - lane.polyline[0]
        previous_norm = float(np.linalg.norm(previous_tangent))
        lane_norm = float(np.linalg.norm(lane_tangent))
        aligned = (
            previous_norm > 1e-9
            and lane_norm > 1e-9
            and float(np.dot(
                previous_tangent / previous_norm,
                lane_tangent / lane_norm,
            )) >= _CHAIN_SHARED_ORIGIN_MIN_ALIGNMENT
        )
        if (float(np.linalg.norm(start_delta)) <= _CHAIN_SHARED_ORIGIN_M
                and aligned):
            collapsed[-1] = lane_id
        elif collapsed[-1] != lane_id:
            collapsed.append(lane_id)
    return collapsed


def _best_lane_id(
    lanes: List[LaneProxy],
    pos: np.ndarray,
    heading: float,
    window: np.ndarray,
) -> str | None:
    """Best lane for one ego pose given its short-term future, or ``None``.

    Gates are frame-local (distance cap + heading alignment); the
    RANKING is by mean lateral offset of the lookahead ``window`` to the
    lane, so a connector that curves away from where the ego actually
    goes next cannot outrank the lane the ego keeps following.
    """
    ego_dir = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    best_score: float | None = None
    best_id: str | None = None
    for lane in lanes:
        polyline = lane.polyline
        if len(polyline) < 2:
            continue
        dists = np.linalg.norm(polyline - pos, axis=1)
        nearest_idx = int(np.argmin(dists))
        dist = float(dists[nearest_idx])
        if dist > _CHAIN_MAX_LANE_DIST_M:
            continue
        if nearest_idx < len(polyline) - 1:
            seg = polyline[nearest_idx + 1] - polyline[nearest_idx]
        else:
            seg = polyline[nearest_idx] - polyline[nearest_idx - 1]
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-8:
            continue
        alignment = float(np.dot(ego_dir, seg / seg_norm))
        if alignment < _CHAIN_MIN_ALIGNMENT:
            continue
        # Mean offset of the lookahead window to the lane. Min over
        # vertices (not a projection) is fine at the ~1 m vertex spacing
        # these maps ship, and points past the lane's end measure to the
        # end vertex — penalising a lane the ego is about to leave is
        # the desired behaviour, it just hands over slightly earlier.
        #
        # Exact band reduction before the (window x polyline) matrix:
        # only vertices near the ego can win the per-point min. For any
        # window point p, the nearest vertex v* satisfies
        #   |p - v*| <= |p - v_nearest| <= |p - pos| + dist,
        # hence |v* - pos| <= |v* - p| + |p - pos| <= dist + 2|p - pos|
        #                  <= dist + 2*reach,
        # where reach = max_p |p - pos|. Dropping vertices farther than
        # that from pos (using the ``dists`` array already computed for
        # the nearest-vertex scan) cannot change any per-point min, so
        # the score is bit-identical; the pairwise matrix just shrinks.
        # (An arc-length band would NOT be exact: arc distance bounds
        # Euclidean distance from above, so it could exclude a near
        # vertex on a lane that loops back toward the ego.)
        reach = float(np.max(np.linalg.norm(window - pos, axis=1)))
        band = dists <= dist + 2.0 * reach + 1e-6
        poly_band = polyline if band.all() else polyline[band]
        w_off = np.linalg.norm(
            window[:, None, :] - poly_band[None, :, :], axis=2
        ).min(axis=1)
        score = float(w_off.mean()) - _CHAIN_ALIGNMENT_WEIGHT * alignment
        if best_score is None or score < best_score:
            best_score = score
            best_id = lane.index
    return best_id


__all__ = ["derive_route_lane_ids"]
