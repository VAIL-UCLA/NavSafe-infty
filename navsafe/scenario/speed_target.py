# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Per-lane speed targets for scenarios whose source map carries none.

**Why this exists.** Every longitudinal decision PDM-Closed makes is
``speed_limit_fraction × lane_speed_limit``
(:mod:`navsafe.policy.state.pdm_closed_planner.idm`), and the planner reads
that limit off the lane feature it is driving on
(``planner._active_lane_speed_limit_mps`` → ``_speed_limit_to_mps``).  When the
lane carries no annotation the IDM bank degenerates to
``fraction × 15 m/s`` — one context-free number for a residential street and a
downtown junction alike.

**What this module does.** It gives every lane a target, in this precedence:

1. ``dataset`` — the source map's own limit, forwarded by the converter
   (nuPlan-sourced py123d logs have one; AV2 does not).
2. ``observed`` — a *spot-speed study* over the log, the same construction
   traffic engineering uses to set a limit where none is posted: for each
   (lane, vehicle) pair take that vehicle's high-percentile speed while it is
   on the lane, then take the 85th percentile over vehicles.  Per-vehicle
   aggregation first is what keeps one bus idling for 200 frames from
   redefining the street.  Read the "85th percentile" honestly: on a quiet
   side street a lane is often observed by ONE vehicle, and a percentile of
   one sample is that sample.  ``evidence[lane]['n_tracks']`` says which case
   a given lane is, and it is frequently 1.
3. ``propagated`` — a lane with no traffic of its own inherits the median of
   its nearest neighbours in the lane graph (``entry_lanes`` / ``exit_lanes``),
   breadth-first, up to :attr:`SpeedTargetParams.max_graph_hops`.
4. ``scene_median`` — the median of every observed lane target on the scene.

A scene with no moving traffic at all gets **no** annotation; the planner then
falls back exactly as before, and the provenance says so.

**The ego is one vehicle among many, and that is recorded.** Its speeds are
evidence like any other track's (``include_ego``), which on a quiet scene is
the only evidence there is.  This is an oracle in closed loop — the target is
derived from a log the policy has not driven yet — so
:func:`annotate_speed_targets` records, per lane, how many tracks contributed
and whether the ego was one of them; ``include_ego=False`` (or
``NEXUSSIM_SPEED_TARGET_INCLUDE_EGO=0``) produces the ego-free variant
without a code edit.  A run's targets must be readable off the scenario, never
re-derived later by a reader who may parameterise it differently: that is what
``metadata['speed_target']`` is for."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Lane-feature key the planner reads (``planner._speed_limit_to_mps``).
SPEED_LIMIT_KEY = "speed_limit_mps"
#: Lane-feature key recording which precedence rule produced the value.
SPEED_LIMIT_SOURCE_KEY = "speed_limit_source"
#: ``metadata`` key holding the full derivation record.
PROVENANCE_KEY = "speed_target"
#: Bumped whenever the derivation changes in a way that moves numbers.
DERIVATION_VERSION = 1

#: Env override: ``off`` disables inference entirely (dataset limits still
#: pass through), restoring the pre-Stage-1b context-free IDM fallback.
#: Unset means "infer"; an unrecognized value warns and infers.
ENV_MODE = "NEXUSSIM_SPEED_TARGET"
#: Env override: ``0`` drops the logged ego from the observed traffic, so a
#: run can be made oracle-free without a code edit. See ``include_ego``.
ENV_INCLUDE_EGO = "NEXUSSIM_SPEED_TARGET_INCLUDE_EGO"

_OFF_VALUES = frozenset({"off", "0", "none", "false", "no"})
_ON_VALUES = frozenset({"", "on", "1", "inferred", "infer", "true", "yes"})


@dataclass(frozen=True)
class SpeedTargetParams:
    """Knobs of the spot-speed study, recorded verbatim in the provenance.

    Attributes:
        lane_percentile: Percentile over per-vehicle speeds that becomes the
            lane target. 85 is the traffic-engineering convention for setting
            a limit from observed speeds.
        track_percentile: Percentile over one vehicle's samples on one lane,
            i.e. that vehicle's unimpeded speed there. Not the max, which is
            a single noisy sample.
        moving_speed_min_mps: Samples at or below this contribute nothing —
            a queue says what the traffic is doing, not what the road allows.
        max_lateral_offset_m: A sample must lie within this distance of a lane
            centerline to be attributed to it. Roughly a half lane width, so
            curb-parked and cross-lane vehicles are not attributed.
        heading_tolerance_rad: Maximum |angle| between the vehicle heading and
            the lane direction. Rejects oncoming traffic on a divided road
            whose centerlines are close.
        min_samples_per_track: Fewest moving samples required BOTH of a
            track overall and of that track on one lane before it
            contributes a value. One frame is a coordinate, not a speed
            study.
        max_graph_hops: How far a lane with no traffic may inherit a target
            through ``entry_lanes`` / ``exit_lanes``.
        min_target_mps / max_target_mps: Clamp. The floor keeps a lane whose
            only evidence is a crawl from pinning the teacher at walking pace;
            the ceiling sits well above any urban speed seen in this data
            (the fastest AV2 background track on the measurement set peaks at
            22 m/s) so it only ever catches tracking blow-ups, not traffic.
            Clamped lanes are counted in the provenance.
        include_ego: Whether the logged ego is one of the observed vehicles.
    """

    lane_percentile: float = 85.0
    track_percentile: float = 90.0
    moving_speed_min_mps: float = 0.5
    max_lateral_offset_m: float = 2.0
    heading_tolerance_rad: float = math.pi / 4.0
    min_samples_per_track: int = 5
    max_graph_hops: int = 3
    min_target_mps: float = 2.0
    max_target_mps: float = 35.0
    include_ego: bool = True


DEFAULT_PARAMS = SpeedTargetParams()

_LANE_TOKEN = "LANE"
_VEHICLE_TYPES = frozenset({"VEHICLE"})


def speed_target_mode(explicit: Optional[str] = None) -> str:
    """Resolve the inference mode: ``"inferred"`` or ``"off"``.

    An unrecognized value infers — but says so. Failing open silently on a
    typo (``NEXUSSIM_SPEED_TARGET=of``) would leave a research arm labelled
    "targets off" running with targets on.
    """
    raw = str(explicit if explicit is not None
              else os.environ.get(ENV_MODE, "")).strip().lower()
    if raw in _OFF_VALUES:
        return "off"
    if raw not in _ON_VALUES:
        logger.warning(
            "%s=%r is not recognized (off: %s); inferring speed targets",
            ENV_MODE, raw, "/".join(sorted(_OFF_VALUES)))
    return "inferred"


def default_params() -> SpeedTargetParams:
    """:data:`DEFAULT_PARAMS`, with the env overrides applied."""
    raw = os.environ.get(ENV_INCLUDE_EGO)
    if raw is None:
        return DEFAULT_PARAMS
    include = str(raw).strip().lower() not in _OFF_VALUES
    return replace(DEFAULT_PARAMS, include_ego=include)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _lane_polylines(map_features: Mapping[str, Any],
                    ) -> Tuple[List[Any], List[np.ndarray]]:
    """Lane keys (as stored) and their (N, 2) centerlines, in a stable order.

    Keys are returned unchanged rather than stringified: they index back into
    ``map_features``, and a producer using non-string ids would otherwise turn
    every lookup into a ``KeyError``. Only the provenance record stringifies,
    because it has to be JSON.

    Non-finite geometry is dropped rather than raised on. ``cKDTree`` rejects
    NaN/inf, and this annotation sits on the critical path of every scenario
    load — one bad vertex in one lane must not abort a conversion, still less
    a shard of a bulk cache build, for the sake of an advisory speed target.
    """
    ids: List[Any] = []
    lines: List[np.ndarray] = []
    for fid, feat in map_features.items():
        if not isinstance(feat, dict):
            continue
        if _LANE_TOKEN not in str(feat.get("type", "")).upper():
            continue
        poly = feat.get("polyline")
        if poly is None:
            continue
        try:
            arr = np.asarray(poly, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        arr = arr[:, :2]
        if not np.isfinite(arr).all():
            continue
        ids.append(fid)
        lines.append(arr)
    return ids, lines


def _vertex_table(lines: Sequence[np.ndarray],
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten lane centerlines into (points, direction, lane index) arrays.

    The direction at a vertex is the forward difference (the backward one at
    the last vertex), normalized; degenerate segments inherit the previous
    direction so every vertex carries a usable heading.
    """
    pts: List[np.ndarray] = []
    dirs: List[np.ndarray] = []
    owner: List[np.ndarray] = []
    for idx, line in enumerate(lines):
        d = np.diff(line, axis=0)
        d = np.vstack([d, d[-1:]])
        norm = np.linalg.norm(d, axis=1)
        good = norm > 1e-9
        # Carry the last usable direction forward across degenerate segments.
        if not good.all():
            last = np.array([1.0, 0.0])
            fixed = np.empty_like(d)
            for i in range(len(d)):
                if good[i]:
                    last = d[i] / norm[i]
                fixed[i] = last
            d = fixed
        else:
            d = d / norm[:, None]
        pts.append(line)
        dirs.append(d)
        owner.append(np.full(len(line), idx, dtype=np.int64))
    if not pts:
        return (np.zeros((0, 2)), np.zeros((0, 2)),
                np.zeros(0, dtype=np.int64))
    return (np.vstack(pts), np.vstack(dirs), np.concatenate(owner))


# ----------------------------------------------------------------------
# Sample collection
# ----------------------------------------------------------------------


def _track_samples(track: Mapping[str, Any], params: SpeedTargetParams,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions, headings, speeds) of a track's moving, valid frames."""
    empty = (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    state = track.get("state") or {}
    # A track missing any of the three is not evidence; it must also not be
    # an exception, or one malformed track fails the whole conversion.
    if any(state.get(k) is None for k in ("position", "heading", "velocity")):
        return empty
    pos = np.asarray(state["position"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] < 2:
        return empty
    head = np.asarray(state["heading"], dtype=np.float64).reshape(-1)
    vel = np.asarray(state["velocity"], dtype=np.float64)
    if vel.ndim != 2 or vel.shape[1] < 2:
        return empty
    valid = state.get("valid")
    keep = (np.asarray(valid, dtype=bool).reshape(-1) if valid is not None
            else np.ones(len(pos), dtype=bool))
    n = min(len(pos), len(head), len(vel), len(keep))
    if n == 0:
        return empty
    speed = np.hypot(vel[:n, 0], vel[:n, 1])
    keep = keep[:n] & (speed > params.moving_speed_min_mps)
    return pos[:n, :2][keep], head[:n][keep], speed[keep]


def _assign(points: np.ndarray, headings: np.ndarray, tree: Any,
            vert_dir: np.ndarray, vert_owner: np.ndarray,
            params: SpeedTargetParams) -> np.ndarray:
    """Lane index per sample, or -1 when no lane claims it."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    dist, idx = tree.query(points, k=1,
                           distance_upper_bound=params.max_lateral_offset_m)
    idx = np.asarray(idx, dtype=np.int64)
    hit = np.isfinite(dist) & (idx < len(vert_owner))
    out = np.full(len(points), -1, dtype=np.int64)
    if not hit.any():
        return out
    d = vert_dir[idx[hit]]
    ego_dir = np.stack([np.cos(headings[hit]), np.sin(headings[hit])], axis=1)
    cos = np.clip(np.einsum("ij,ij->i", d, ego_dir), -1.0, 1.0)
    aligned = cos >= math.cos(params.heading_tolerance_rad)
    sel = np.flatnonzero(hit)[aligned]
    out[sel] = vert_owner[idx[sel]]
    return out


# ----------------------------------------------------------------------
# Derivation
# ----------------------------------------------------------------------


def _observed_targets(sd: Mapping[str, Any], lines: Sequence[np.ndarray],
                      params: SpeedTargetParams,
                      ) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]]]:
    """Per-lane observed target plus the evidence behind it."""
    from scipy.spatial import cKDTree

    verts, vert_dir, vert_owner = _vertex_table(lines)
    if len(verts) == 0:
        return {}, {}
    tree = cKDTree(verts)

    sdc_id = str((sd.get("metadata") or {}).get("sdc_id", ""))
    # lane index -> list of (per-vehicle value, is_ego, n_samples)
    per_lane: Dict[int, List[Tuple[float, bool, int]]] = {}

    for tid, track in (sd.get("tracks") or {}).items():
        if not isinstance(track, dict):
            continue
        is_ego = str(tid) == sdc_id
        if is_ego and not params.include_ego:
            continue
        if str(track.get("type", "")).upper() not in _VEHICLE_TYPES:
            continue
        pos, head, speed = _track_samples(track, params)
        if len(pos) < params.min_samples_per_track:
            continue
        if not (np.isfinite(pos).all() and np.isfinite(head).all()):
            continue
        lane_of = _assign(pos, head, tree, vert_dir, vert_owner, params)
        claimed = lane_of >= 0
        if not claimed.any():
            continue
        for lane_idx in np.unique(lane_of[claimed]):
            sel = speed[lane_of == lane_idx]
            if len(sel) < params.min_samples_per_track:
                continue
            per_lane.setdefault(int(lane_idx), []).append(
                (float(np.percentile(sel, params.track_percentile)),
                 bool(is_ego), int(len(sel))))

    targets: Dict[int, float] = {}
    evidence: Dict[int, Dict[str, Any]] = {}
    for lane_idx, rows in per_lane.items():
        values = np.asarray([r[0] for r in rows], dtype=np.float64)
        targets[lane_idx] = float(np.percentile(values, params.lane_percentile))
        evidence[lane_idx] = {
            "n_tracks": len(rows),
            "n_samples": int(sum(r[2] for r in rows)),
            "ego_contributed": bool(any(r[1] for r in rows)),
            "track_values_mps": [round(float(v), 4) for v in values],
        }
    return targets, evidence


def _propagate(targets: Mapping[int, float], lane_ids: Sequence[Any],
               map_features: Mapping[str, Any], max_hops: int,
               ) -> Dict[int, Tuple[float, int, List[int]]]:
    """Breadth-first inheritance along the lane graph.

    Returns ``lane index -> (value, hops, donor indices)`` for lanes with no
    observation of their own. Neighbours are taken in both directions
    (``entry_lanes`` and ``exit_lanes``): a lane's free-flow speed is a
    property of the road it is part of, which does not have a direction.

    Adjacency is a SET per lane. A symmetric edge is normally declared twice
    (A lists B as an exit, B lists A as an entry), and a list would then weight
    that donor twice in the median — so how often the map happened to spell an
    edge out would move the target.
    """
    index_of = {str(lid): i for i, lid in enumerate(lane_ids)}
    adjacency: Dict[int, set] = {i: set() for i in range(len(lane_ids))}
    for i, lid in enumerate(lane_ids):
        feat = map_features.get(lid) or {}
        for key in ("entry_lanes", "exit_lanes"):
            for other in (feat.get(key) or ()):
                j = index_of.get(str(other))
                if j is not None and j != i:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
    neighbours = {i: sorted(js) for i, js in adjacency.items()}

    out: Dict[int, Tuple[float, int, List[int]]] = {}
    known = {int(i): float(v) for i, v in targets.items()}
    frontier: List[int] = sorted(known)
    for hop in range(1, int(max_hops) + 1):
        contributions: Dict[int, List[Tuple[int, float]]] = {}
        for i in frontier:
            for j in neighbours.get(i, ()):
                if j in known:
                    continue
                contributions.setdefault(j, []).append((i, known[i]))
        if not contributions:
            break
        # Resolve the whole hop before it becomes evidence, so a lane at hop
        # h never inherits from a sibling resolved in the same pass.
        for j, donors in contributions.items():
            out[j] = (float(np.median([v for _, v in donors])), hop,
                      [d for d, _ in donors])
        for j in contributions:
            known[j] = out[j][0]
        frontier = sorted(contributions)
    return out


def annotate_speed_targets(
    sd: Any,
    *,
    params: Optional[SpeedTargetParams] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Give every lane in ``sd`` a speed target and record how it was derived.

    Mutates ``sd`` in place: each lane feature gains ``speed_limit_mps`` and
    ``speed_limit_source``, and ``sd['metadata']['speed_target']`` gains the
    full derivation record (parameters, per-lane values, per-lane evidence).
    Lanes that already carry a dataset limit keep it.

    Args:
        sd: A ``ScenarioDescription``-shaped mapping with ``map_features``,
            ``tracks`` and ``metadata``.
        params: Study parameters; defaults to :data:`DEFAULT_PARAMS`.
        mode: ``"inferred"`` (default) or ``"off"``; ``None`` reads
            :data:`ENV_MODE` from the environment.

    Returns:
        The provenance record, also stored on ``sd['metadata']``.
    """
    par = params if params is not None else default_params()
    resolved_mode = speed_target_mode(mode)
    map_features = sd.get("map_features") or {}
    lane_ids, lines = _lane_polylines(map_features)

    n_lane_total = sum(
        1 for f in map_features.values()
        if isinstance(f, dict) and _LANE_TOKEN in str(f.get("type", "")).upper())
    provenance: Dict[str, Any] = {
        "version": DERIVATION_VERSION,
        "mode": resolved_mode,
        "params": asdict(par),
        # Lanes the study could work with vs. lanes the map declares. They
        # differ when a lane's centerline is missing, degenerate or non-finite;
        # ``counts`` sums to the former, so both are recorded.
        "n_lane_features": len(lane_ids),
        "n_lane_features_total": n_lane_total,
        "counts": {"dataset": 0, "observed": 0, "propagated": 0,
                   "scene_median": 0, "preexisting": 0, "unset": 0},
        "clamped": 0,
        "scene_median_mps": None,
        "scene_median_source": "none",
        "target_mps": {},
        "evidence": {},
    }

    # A dataset limit is one the SOURCE MAP posted, not one an earlier call to
    # this function inferred. Without the source test, annotating twice (the
    # reconstruction assembler re-projects a scenario it has already built)
    # would relabel every inferred target as "dataset" and freeze it — a
    # silent promotion of an estimate to a posted fact.
    dataset_idx = {
        i for i, lid in enumerate(lane_ids)
        if _is_dataset_limit(map_features.get(lid) or {})
    }
    provenance["counts"]["dataset"] = len(dataset_idx)

    off = resolved_mode == "off" or not lane_ids
    if off:
        observed: Dict[int, float] = {}
        evidence: Dict[int, Dict[str, Any]] = {}
    else:
        observed, evidence = _observed_targets(sd, lines, par)
        observed = {i: v for i, v in observed.items() if i not in dataset_idx}
    provenance["method"] = (
        "log_observed_spot_speed_p85" if observed
        else "dataset_only" if dataset_idx else "none")

    propagated = ({} if off else
                  _propagate(observed, lane_ids, map_features,
                             par.max_graph_hops))
    # Last resort for a lane the traffic never touched and the lane graph
    # never reaches. Observed lanes first; a partially-posted map (some
    # nuPlan maps post limits on arterials only) falls back to its own
    # posted values rather than to a global constant.
    if observed:
        scene_median: Optional[float] = float(
            np.median(list(observed.values())))
        median_source = "observed"
    elif dataset_idx and not off:
        scene_median = float(np.median([
            _dataset_limit_mps(map_features[lane_ids[i]]) or 0.0
            for i in dataset_idx]))
        median_source = "dataset"
    else:
        scene_median, median_source = None, "none"
    provenance["scene_median_mps"] = scene_median
    provenance["scene_median_source"] = median_source

    for i, lid in enumerate(lane_ids):
        feat = map_features[lid]
        key = str(lid)
        if i in dataset_idx:
            feat[SPEED_LIMIT_SOURCE_KEY] = "dataset"
            posted = _dataset_limit_mps(feat)
            if posted is not None:
                provenance["target_mps"][key] = posted
            continue
        if i in observed:
            value, source, bucket = observed[i], "observed", "observed"
        elif i in propagated:
            value = propagated[i][0]
            source = f"propagated_{propagated[i][1]}hop"
            bucket = "propagated"
            provenance["evidence"][key] = {
                "inherited_from": [str(lane_ids[d]) for d in propagated[i][2]],
                "hops": propagated[i][1],
            }
        elif scene_median is not None:
            value, source, bucket = scene_median, "scene_median", "scene_median"
        else:
            # This call derived nothing for this lane. It may still CARRY a
            # target an earlier call inferred, which is what the planner will
            # read — so report what is on the lane, not what this call did.
            # Calling that "unset" is how a record starts disagreeing with the
            # scenario it describes.
            existing = _positive_float(feat.get(SPEED_LIMIT_KEY))
            if existing is None:
                provenance["counts"]["unset"] += 1
                continue
            provenance["target_mps"][key] = existing
            provenance["counts"]["preexisting"] += 1
            provenance["evidence"][key] = {
                "recorded_source": str(feat.get(SPEED_LIMIT_SOURCE_KEY, "")),
            }
            continue
        clamped = float(np.clip(value, par.min_target_mps, par.max_target_mps))
        if abs(clamped - value) > 1e-9:
            provenance["clamped"] += 1
        feat[SPEED_LIMIT_KEY] = clamped
        feat[SPEED_LIMIT_SOURCE_KEY] = source
        provenance["target_mps"][key] = clamped
        provenance["counts"][bucket] += 1
        if i in evidence:
            provenance["evidence"][key] = evidence[i]

    _store(sd, provenance)
    counts = provenance["counts"]
    logger.info(
        "speed targets (%s v%d): %d lanes — %d dataset / %d observed / "
        "%d propagated / %d scene-median / %d unset; scene median %s m/s",
        provenance["method"], DERIVATION_VERSION, len(lane_ids),
        counts["dataset"], counts["observed"], counts["propagated"],
        counts["scene_median"], counts["unset"],
        "n/a" if scene_median is None else f"{scene_median:.2f}")
    return provenance


#: Every key ``planner._speed_limit_to_mps`` will read. A posted limit in ANY
#: of them outranks an estimate — and since the planner checks ``_mps`` first,
#: writing an inferred ``_mps`` onto a lane that posts ``_kmh`` would let the
#: estimate silently outrank the posted value.
_DATASET_LIMIT_KEYS = (SPEED_LIMIT_KEY, "speed_limit_kmh", "speed_limit_mph",
                       "speedLimit")


#: Unit conversions to m/s, mirroring ``planner._speed_limit_to_mps``.
_LIMIT_TO_MPS = {SPEED_LIMIT_KEY: 1.0, "speed_limit_kmh": 1.0 / 3.6,
                 "speed_limit_mph": 0.44704, "speedLimit": 1.0}


def _dataset_limit_mps(feat: Mapping[str, Any]) -> Optional[float]:
    """The lane's posted limit in m/s, in the planner's own key precedence."""
    for k in _DATASET_LIMIT_KEYS:
        value = _positive_float(feat.get(k))
        if value is not None:
            return value * _LIMIT_TO_MPS[k]
    return None


def _is_dataset_limit(feat: Mapping[str, Any]) -> bool:
    """Whether this lane already carries a limit from the source map."""
    if any(_positive_float(feat.get(k)) is not None
           for k in _DATASET_LIMIT_KEYS[1:]):
        return True
    if _positive_float(feat.get(SPEED_LIMIT_KEY)) is None:
        return False
    source = feat.get(SPEED_LIMIT_SOURCE_KEY)
    return source is None or str(source) == "dataset"


def _positive_float(value: Any) -> Optional[float]:
    """``float(value)`` when it is finite and positive, else ``None``."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0.0 else None


def _store(sd: Any, provenance: Dict[str, Any]) -> None:
    meta = sd.get("metadata")
    if meta is None:
        meta = {}
        sd["metadata"] = meta
    meta[PROVENANCE_KEY] = provenance


__all__ = [
    "DEFAULT_PARAMS",
    "DERIVATION_VERSION",
    "ENV_INCLUDE_EGO",
    "ENV_MODE",
    "PROVENANCE_KEY",
    "SPEED_LIMIT_KEY",
    "SPEED_LIMIT_SOURCE_KEY",
    "SpeedTargetParams",
    "annotate_speed_targets",
    "default_params",
    "speed_target_mode",
]
