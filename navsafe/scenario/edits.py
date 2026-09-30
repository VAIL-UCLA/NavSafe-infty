# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-time scenario-edit tools.

Each tool is a pure function ``tool(sd, **params) -> sd`` operating on a
runtime ``ScenarioDescription`` dict, applied after the loader returns and
before the env builds the scene. Tools are registered by name in
``EDIT_TOOLS`` and driven declaratively via ``EnvCfg.scenario_edits``::

    cfg.scenario_edits = [
        {"tool": "place_static_obstacles", "count": 6, "seed": 42},
    ]

The four edit ops map onto these tools: ``insert``
(``place_static_obstacles``), ``replace`` (``replace_agent_with_asset``),
``relocate`` (``relocate_agent``) and ``remove`` (``remove_tracks``) solve the
geometry live from parameters. ``spawn_reactive_actor`` is what a frozen NavSafe
recipe produces instead: it declares the actor and hands its spawn and
controller to the traffic manager, which owns the motion from there. Prefer a
*relocate* over an *insert* where a suitable actor exists: with no archetype
swap it keeps the actor's baked appearance, which is the most reliable path,
because the server can silently drop a freshly inserted asset.

There used to be a fifth, ``apply_baked_track``, which wrote the per-frame
arrays a recipe had resolved in advance. Recipes stopped carrying trajectories
when their actors became reactive, so nothing could produce its input.

Injected tracks carry ``metadata.injected_obstacle=True`` so downstream
consumers can special-case them — e.g. the replay manager spawns their visual
prims even in the ego-only visualization mode (a reconstruction
cannot bake an object that was never in the log), while symbolic scoring
(collision / TTC / EPDMS) picks them up like any other track.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np

from navsafe.scenario.scenario_description import scenario_dt_seconds

logger = logging.getLogger(__name__)

# (length, width, height) metres per obstacle archetype; sampled uniformly.
_OBSTACLE_ARCHETYPES: Dict[str, tuple] = {
    "car": (4.6, 1.9, 1.6),
    "suv": (5.0, 2.0, 1.8),
    "truck": (8.0, 2.5, 3.2),
    "cone": (0.4, 0.4, 0.7),
    # Tall channelizer-style cone — visually on par with construction-zone
    # drums/panels; use when a 0.7 m cone reads too small in the scene.
    "cone_tall": (0.4, 0.4, 1.0),
    # UrbanVerse scanned sedan (uid 946ec7c0…, annotation dims) — pair with
    # --obstacle-nurec-asset sedan_uv=<ASSETS>/sedan_uv_3dgs.ply
    "sedan_uv": (3.96, 1.65, 1.45),
    # Harvested sedan (ah_assets/car_1.ply), dims MEASURED off the calibrated
    # PLY and mirrored from the NavSafe asset registry's `hb_car_1`. Listed
    # rather than reusing "car" (4.6x1.9x1.6) because these dims become the
    # replacement track's collision box: the generic archetype would put a
    # box a metre longer than the car anyone can see, which brakes the ego
    # early and draws a too-large BEV box.
    "car_1": (3.566, 1.594, 1.163),
    # Roadside warning sign (thin plate on a post) — pair with the procedural
    # asset: convert_mesh_to_3dgs --procedural-sign sign_3dgs.ply --target-height 2.0
    "sign": (0.15, 0.66, 2.0),
    # Long-tail classes — UrbanVerse meshes converted to 3DGS (dims from the
    # converted PLY bbox: L=forward, W=lateral, H=up). Pair each with
    # --obstacle-nurec-asset <name>=<ASSETS>/<name>_uv_3dgs.ply
    "barricade": (1.63, 0.98, 1.00),
    "drum": (1.00, 0.87, 0.95),
    "bollard": (0.73, 0.32, 1.00),
    "stop_sign": (0.68, 0.07, 2.20),
    "work_vehicle": (7.67, 3.05, 3.20),
    "trash_bin": (1.03, 0.53, 1.10),
    "dog": (0.87, 0.23, 0.55),
    "wheelchair": (0.54, 0.31, 1.35),
    "police_car": (4.65, 2.00, 1.50),
    # Robotic micromobility family + stroller (UrbanVerse-100K → mesh2gs 3DGS).
    # Dims (L=forward, W=lateral, H=up) from real-world class size; pair each
    # with --obstacle-nurec-asset <name>=<ASSETS>/<name>_uv_3dgs.ply.
    # See docs/research/longtailed_assets.md + download_targeted_assets.py.
    "delivery_robot": (0.70, 0.60, 1.00),
    "robot_dog": (0.80, 0.35, 0.60),
    "humanoid_robot": (0.50, 0.40, 1.60),
    "scooter": (1.10, 0.50, 1.20),
    "stroller": (0.90, 0.55, 1.05),
    "escooter_rider": (0.90, 0.55, 1.70),  # rider standing on an e-scooter
}

# Default sampling pool when ``types`` is not given — the legacy vehicle-only
# behaviour. "cone" is opt-in via ``types=["cone"]`` so existing eval configs
# keep their obstacle mix.
_DEFAULT_OBSTACLE_TYPES = ["car", "suv", "truck"]

# Track type per archetype (default VEHICLE). The type drives sim-side spawn
# and observation semantics — TRAFFIC_CONE enters NavSim-style observations as
# a static object and takes the replay manager's cone spawn path.
_ARCHETYPE_TRACK_TYPE: Dict[str, str] = {
    "cone": "TRAFFIC_CONE",
    "cone_tall": "TRAFFIC_CONE",
    "sign": "TRAFFIC_CONE",
    # static long-tail props
    "barricade": "TRAFFIC_CONE",
    "drum": "TRAFFIC_CONE",
    "bollard": "TRAFFIC_CONE",
    "stop_sign": "TRAFFIC_CONE",
    "trash_bin": "TRAFFIC_CONE",
    "work_vehicle": "TRAFFIC_CONE",
    # robotic micromobility + stroller:
    # insert as static props for the
    # visual/scoring test (dynamic actor
    # variants can be added later).
    "delivery_robot": "TRAFFIC_CONE",
    "robot_dog": "TRAFFIC_CONE",
    "humanoid_robot": "TRAFFIC_CONE",
    "scooter": "TRAFFIC_CONE",
    "stroller": "TRAFFIC_CONE",
    "escooter_rider": "TRAFFIC_CONE",
}
# (dynamic actors dog / wheelchair / police_car keep the default VEHICLE type)


def place_static_obstacles(
    sd: dict,
    count: int = 1,
    seed: int = 42,
    start_offset_m: float = 15.0,
    end_margin_m: float = 5.0,
    lateral_jitter_m: float = 0.0,
    after_frame: int = 0,
    ego_z_to_ground_m: float = 1.7,
    types: List[str] | None = None,
    nurec_asset_ids: Dict[str, str] | None = None,
    arc_positions: List[float] | None = None,
) -> dict:
    """Drop ``count`` static obstacles on the ego's logged route.

    Positions are arc-length samples of the ego ground-truth trajectory (the
    road/lane centre the log ego actually drove), evenly spaced over
    ``[anchor + start_offset_m, route_end - end_margin_m]`` with seeded jitter;
    heading follows the local route tangent so the obstacle sits aligned with
    the lane. ``after_frame`` anchors the offset at the ego's position at that
    frame (pass the warm-up/replay hand-off frame so the policy gets a
    reaction window — with 0 the first obstacle can sit inside the replayed
    stretch and the ego rams it before it ever plans).

    The obstacles exist for the whole scenario (static, valid at every frame),
    so a closed-loop policy must brake for or steer around them.

    ``types`` selects the archetype pool (default vehicles only); ``"cone"``
    injects a TRAFFIC_CONE-typed track. ``nurec_asset_ids`` maps archetype ->
    NuRec server asset_id (an AssetBank track id, or a filesystem path to a
    3DGS PLY readable inside the sensorsim container). When set, the injected
    track's metadata carries ``nurec_asset_id`` + ``nurec_semantic_class`` and
    the ``nurec_grpc`` renderer registers it into the served scene via the
    ``edit_assets`` RPC, so the obstacle appears in the NuRec render as well
    as in sim state. Without it, the obstacle is scored but (on ``nurec_grpc``)
    not drawn.
    """
    tracks = sd.get("tracks", {})
    meta = sd.get("metadata", {})
    sdc_id = str(meta.get("sdc_id", "ego"))
    ego = tracks.get(sdc_id)
    if not ego:
        logger.warning("place_static_obstacles: no ego track %r; skipping", sdc_id)
        return sd

    state = ego.get("state", {})
    pos = np.asarray(state.get("position"))
    valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
    if pos is None or len(pos) < 2:
        logger.warning("place_static_obstacles: ego track too short; skipping")
        return sd

    T = len(pos)
    route = pos[valid][:, :2]
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    anchor = float(arc[min(max(0, int(after_frame)), len(arc) - 1)])
    lo = anchor + start_offset_m
    hi = max(lo + 1.0, total - end_margin_m)
    if total <= lo:
        logger.warning(
            "place_static_obstacles: route only %.1fm (< anchor %.1fm + offset); skipping",
            total,
            anchor,
        )
        return sd

    rng = np.random.default_rng(int(seed))
    # Even spacing + jitter of up to a quarter slot keeps obstacles ordered and
    # non-overlapping without a rejection loop.
    slots = np.linspace(lo, hi, int(count))
    slot_w = (hi - lo) / max(1, count - 1)
    if arc_positions:
        # Deterministic placement: metres past the hand-off anchor, no jitter.
        # Lets an eval pin obstacles onto a clean stretch of the reconstruction
        # (the sampled spread once landed a car inside the baked dust cloud at
        # the intersection, where it renders veiled and near-invisible).
        arcs = np.clip(
            np.asarray([anchor + float(a) for a in arc_positions], np.float64)[:count],
            lo * 0.0,
            arc[-1] - 1.0,
        )
    else:
        arcs = np.clip(slots + rng.uniform(-0.25, 0.25, count) * slot_w, lo, hi)

    type_names = list(types or _DEFAULT_OBSTACLE_TYPES)
    n_injected = 0
    for i, s in enumerate(arcs):
        j = int(np.searchsorted(arc, s).clip(1, len(arc) - 1))
        frac = (s - arc[j - 1]) / max(1e-6, arc[j] - arc[j - 1])
        p = route[j - 1] + frac * (route[j] - route[j - 1])
        tangent = route[j] - route[j - 1]
        heading = float(np.arctan2(tangent[1], tangent[0]))
        if lateral_jitter_m > 0.0:
            normal = np.array([-tangent[1], tangent[0]])
            normal /= max(1e-6, np.linalg.norm(normal))
            p = p + normal * rng.uniform(-lateral_jitter_m, lateral_jitter_m)

        name = type_names[int(rng.integers(len(type_names)))]
        length, width, height = _OBSTACLE_ARCHETYPES.get(name, _OBSTACLE_ARCHETYPES["car"])
        # Ground z: nearest ego sample minus the pose height above the road
        # (dataset-specific: ~1.7 m for AV2, ~1.4 m for WOD py123d arrows —
        # verified with a z-lift render study; too large a value buries short
        # obstacles in the NuRec road gaussian shell).
        z = (float(pos[valid][j][2]) - ego_z_to_ground_m) if pos.shape[1] > 2 else 0.0

        track_type = _ARCHETYPE_TRACK_TYPE.get(name, "VEHICLE")
        obj_id = f"injected_obstacle_{i}"
        track: dict[str, Any] = {
            "type": track_type,
            "state": {
                "position": np.tile(np.array([p[0], p[1], z], np.float32), (T, 1)),
                "length": np.full(T, length, np.float32),
                "width": np.full(T, width, np.float32),
                "height": np.full(T, height, np.float32),
                "heading": np.full(T, heading, np.float32),
                "velocity": np.zeros((T, 2), np.float32),
                "valid": np.ones(T, bool),
            },
            "metadata": {
                "track_length": T,
                "type": track_type,
                "object_id": obj_id,
                "injected_obstacle": True,
                "obstacle_archetype": name,
            },
        }
        asset_id = (nurec_asset_ids or {}).get(name)
        if asset_id:
            track["metadata"]["nurec_asset_id"] = str(asset_id)
            track["metadata"]["nurec_semantic_class"] = name
        tracks[obj_id] = track
        n_injected += 1
        logger.info(
            "place_static_obstacles: %s (%s) at arc %.1fm pos=(%.1f, %.1f) heading=%.2f",
            obj_id,
            name,
            s,
            p[0],
            p[1],
            heading,
        )

    logger.info(
        "place_static_obstacles: injected %d/%d obstacles (seed=%d)", n_injected, count, seed
    )
    return sd


def replace_agent_with_asset(
    sd: dict,
    replacements: "Dict[str, str] | None" = None,
    nurec_asset_ids: "Dict[str, str] | None" = None,
    ego_z_to_ground_m: float = 1.4,
    extra_z_drop: float = 0.0,
    **_ignored,
) -> dict:
    """Swap existing scenario agents for injected NuRec assets, in place.

    ``replacements`` maps an existing ``track_id`` -> obstacle archetype name
    (e.g. ``"sedan_uv"``). For each: the original track is DELETED (so it
    vanishes from BOTH the render — the nurec_grpc mirror relocates any actor
    absent from agent_states off-screen — and the sim state / scoring), and a
    new injected track is created that COPIES the original's per-frame
    trajectory (position xy / heading / valid) but carries the replacement
    archetype's dims and, via ``nurec_asset_ids[archetype]``, a NuRec asset to
    render.

    Ground z (differs from place_static_obstacles): the perception box z is the
    box CENTRE, and — unlike the EGO track z, which py123d lifts ~0.85 m above
    the road — actor boxes are already GROUND-referenced (base on the road). So
    the replacement asset's BASE goes at ``box_centre_z - original_height/2``
    (centre -> base). We do NOT subtract ``ego_z_to_ground_m`` here: that offset
    only corrects the lifted EGO z (used by the camera and by route-anchored
    obstacle placement), not actor boxes. ``extra_z_drop`` is an optional manual
    nudge (default 0) for residual recon-road waviness.
    """
    if not replacements:
        return sd
    tracks = sd.get("tracks", {})
    meta = sd.get("metadata", {})
    sdc_id = meta.get("sdc_id") or sd.get("sdc_id") or "ego"
    n = 0
    for i, (target_id, archetype) in enumerate(dict(replacements).items()):
        tid = str(target_id)
        if tid == sdc_id or tid not in tracks:
            logger.warning("replace_agent_with_asset: track %r not found; skipping", tid)
            continue
        src = tracks[tid]
        st = src.get("state", {})
        pos = np.asarray(st.get("position"), np.float64)
        if pos.ndim != 2 or pos.shape[0] < 1:
            logger.warning("replace_agent_with_asset: track %r has no trajectory; skipping", tid)
            continue
        T = pos.shape[0]
        heading = np.asarray(st.get("heading", np.zeros(T)), np.float64).reshape(-1)
        if heading.shape[0] != T:
            heading = np.full(T, float(heading[0]) if heading.size else 0.0)
        valid = np.asarray(st.get("valid", np.ones(T, bool))).astype(bool).reshape(-1)
        if valid.shape[0] != T:
            valid = np.ones(T, bool)
        length, width, height = _OBSTACLE_ARCHETYPES.get(archetype, _OBSTACLE_ARCHETYPES["car"])
        # Original car's per-frame HEIGHT (perception box). box_z is the centre;
        # base = centre - h/2 sits on the road (actor boxes are ground-refd).
        orig_h = np.asarray(st.get("height", np.full(T, height)), np.float64).reshape(-1)
        if orig_h.shape[0] != T:
            orig_h = np.full(T, float(orig_h[0]) if orig_h.size else float(height))
        centre_z = pos[:, 2] if pos.shape[1] > 2 else np.zeros(T)
        z_series = (centre_z - 0.5 * orig_h - float(extra_z_drop)).astype(np.float32)
        track_type = _ARCHETYPE_TRACK_TYPE.get(archetype, "VEHICLE")
        obj_id = f"replaced_agent_{i}"
        new: dict[str, Any] = {
            "type": track_type,
            "state": {
                "position": np.stack(
                    [pos[:, 0].astype(np.float32), pos[:, 1].astype(np.float32), z_series], axis=1
                ),
                "length": np.full(T, length, np.float32),
                "width": np.full(T, width, np.float32),
                "height": np.full(T, height, np.float32),
                "heading": heading.astype(np.float32),
                "velocity": np.zeros((T, 2), np.float32),
                "valid": valid,
            },
            "metadata": {
                "track_length": T,
                "type": track_type,
                "object_id": obj_id,
                "injected_obstacle": True,
                "obstacle_archetype": archetype,
                "replaced_track_id": tid,
            },
        }
        asset_id = (nurec_asset_ids or {}).get(archetype)
        if asset_id:
            new["metadata"]["nurec_asset_id"] = str(asset_id)
            new["metadata"]["nurec_semantic_class"] = archetype
        del tracks[tid]
        tracks[obj_id] = new
        n += 1
        logger.info(
            "replace_agent_with_asset: %s -> %s (%s) at (%.1f, %.1f), asset=%s",
            tid,
            obj_id,
            archetype,
            float(pos[0, 0]),
            float(pos[0, 1]),
            asset_id,
        )
    logger.info("replace_agent_with_asset: replaced %d agent(s)", n)
    return sd


def relocate_agent(
    sd: dict,
    relocations: Dict[str, Dict[str, Any]] | None = None,
    nurec_asset_ids: Dict[str, str] | None = None,
    after_frame: int = 0,
    ego_z_to_ground_m: float = 0.0,
    dt_s: float = 0.1,
    deceleration_mps2: float = 2.5,
    maneuver_start_frame: int | None = None,
    maneuver_duration_frames: int = 20,
    start_lateral: float | None = None,
    end_lateral: float | None = None,
    conflict_frame: int | None = None,
) -> dict:
    """Move EXISTING reconstructed actors onto a synthetic route-anchored
    trajectory — the safety-critical scenario-authoring primitive.

    For each ``track_id`` in ``relocations`` the actor is re-posed along the
    ego's logged route (``arc`` metres past the closed-loop hand-off
    ``after_frame``); ``mode`` picks the motion:

      * ``static``  — a fixed point (stalled car / sudden obstacle in the path);
      * ``dynamic`` — advances along the route at ``speed`` m/s;
      * ``braking`` — advances then decelerates to a stop;
      * ``cut_in`` — advances while interpolating from ``start_lateral`` to
        ``end_lateral``;
      * ``crossing`` — traverses laterally across the ego route and reaches the
        route centre at ``conflict_frame``.

    Per-entry spec keys: ``mode`` (static|dynamic|braking|cut_in|crossing),
    ``arc`` (m, default 20),
    ``speed`` (m/s, dynamic, default 3), ``lateral`` (+left m from the
    centreline, default 0), ``yaw_offset_deg`` (rotate heading off the route
    tangent, default 0), ``archetype`` (optional).

    Appearance:
      * no ``archetype`` → the ORIGINAL track id is kept and only its state is
        rewritten, so the server keeps rendering the actor's BAKED gaussians at
        the new pose. This is a pure per-frame pose override — exactly what the
        nurec_grpc mirror already applies every frame — so it renders reliably
        (unlike a fresh asset insert).
      * with ``archetype`` → the source track is DELETED and a new asset track
        (dims from the archetype, ``nurec_asset_ids[archetype]`` for the NuRec
        render) is created on the synthetic trajectory, i.e. a moving/positioned
        imported asset (e.g. a driving UrbanVerse sedan).

    z: baked actors ground at road + half-height in the renderer (their sim z is
    clamped), so it is left at the route z; injected assets are base-origin, so
    their base is pinned to the route road z (matching place_static_obstacles).
    """
    if not relocations:
        return sd
    tracks = sd.get("tracks", {})
    meta = sd.get("metadata", {})
    sdc_id = str(meta.get("sdc_id", "ego"))
    ego = tracks.get(sdc_id)
    if not ego:
        logger.warning("relocate_agent: no ego track %r; skipping", sdc_id)
        return sd
    estate = ego.get("state", {})
    epos = np.asarray(estate.get("position"), np.float64)
    evalid = np.asarray(estate.get("valid", np.ones(len(epos), bool))).astype(bool)
    if epos.ndim != 2 or len(epos) < 2:
        logger.warning("relocate_agent: ego track too short; skipping")
        return sd
    route = epos[evalid][:, :2]
    route_z = epos[evalid][:, 2] if epos.shape[1] > 2 else np.zeros(len(route))
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    anchor = float(arc[min(max(0, int(after_frame)), len(arc) - 1)])

    def _sample(s: float):
        """(x, y, road_z, heading) at arc-length ``s`` along the ego route."""
        s = float(np.clip(s, 0.0, total))
        j = int(np.clip(np.searchsorted(arc, s), 1, len(arc) - 1))
        f = (s - arc[j - 1]) / max(1e-6, arc[j] - arc[j - 1])
        p = route[j - 1] + f * (route[j] - route[j - 1])
        z = route_z[j - 1] + f * (route_z[j] - route_z[j - 1])
        tang = route[j] - route[j - 1]
        return float(p[0]), float(p[1]), float(z), float(np.arctan2(tang[1], tang[0]))

    if dt_s <= 0:
        raise ValueError("relocate_agent dt_s must be positive")
    if maneuver_duration_frames < 1:
        raise ValueError("relocate_agent maneuver_duration_frames must be positive")
    n = 0
    for src_id, spec in dict(relocations).items():
        src_id = str(src_id)
        if src_id == sdc_id or src_id not in tracks:
            logger.warning("relocate_agent: track %r not found; skipping", src_id)
            continue
        src = tracks[src_id]
        st = src.get("state", {})
        pos = np.asarray(st.get("position"), np.float64)
        if pos.ndim != 2 or pos.shape[0] < 1:
            logger.warning("relocate_agent: track %r has no trajectory; skipping", src_id)
            continue
        T = pos.shape[0]
        mode = str(spec.get("mode", "static")).lower()
        if mode not in {"static", "dynamic", "braking", "cut_in", "crossing"}:
            raise ValueError(f"relocate_agent unsupported mode {mode!r}")
        arc_off = float(spec.get("arc", 20.0))
        speed = float(spec.get("speed", 3.0))
        if speed < 0:
            raise ValueError("relocate_agent speed must be non-negative")
        lateral = float(spec.get("lateral", 0.0))
        yaw_off = float(spec.get("yaw_offset_deg", 0.0)) * np.pi / 180.0
        archetype = spec.get("archetype")
        start_s = anchor + arc_off

        local_start_frame = int(
            spec.get(
                "maneuver_start_frame",
                after_frame if maneuver_start_frame is None else maneuver_start_frame,
            )
        )
        local_duration = int(spec.get("maneuver_duration_frames", maneuver_duration_frames))
        if local_duration < 1:
            raise ValueError("relocate_agent maneuver_duration_frames must be positive")
        local_start_lateral = float(
            spec.get(
                "start_lateral",
                lateral if start_lateral is None else start_lateral,
            )
        )
        local_end_lateral = float(
            spec.get(
                "end_lateral",
                0.0 if end_lateral is None else end_lateral,
            )
        )
        local_conflict_frame = int(
            spec.get(
                "conflict_frame",
                local_start_frame + local_duration // 2
                if conflict_frame is None
                else conflict_frame,
            )
        )
        local_deceleration = float(spec.get("deceleration_mps2", deceleration_mps2))
        if local_deceleration <= 0:
            raise ValueError("relocate_agent deceleration_mps2 must be positive")
        if bool(spec.get("align_to_ego_at_conflict", False)):
            conflict_index = min(max(0, local_conflict_frame), len(arc) - 1)
            start_s = float(arc[conflict_index]) + arc_off

        # Bare ndarray: the branches below rebind ``longitudinal`` to a
        # broadcast product whose static shape/dtype params differ from
        # ``np.zeros``' precise ones (identical float64 array at runtime).
        longitudinal: np.ndarray = np.zeros(T, np.float64)
        longitudinal_speed = np.zeros(T, np.float64)
        lateral_series = np.full(T, lateral, np.float64)
        lateral_speed = np.zeros(T, np.float64)
        if mode == "dynamic":
            longitudinal = speed * dt_s * np.arange(T)
            longitudinal_speed.fill(speed)
        elif mode == "braking":
            current_speed = speed
            distance = 0.0
            for k in range(T):
                longitudinal[k] = distance
                longitudinal_speed[k] = current_speed
                distance += current_speed * dt_s
                if k >= local_start_frame:
                    current_speed = max(0.0, current_speed - local_deceleration * dt_s)
        elif mode == "cut_in":
            longitudinal = speed * dt_s * np.arange(T)
            longitudinal_speed.fill(speed)
            for k in range(T):
                alpha = np.clip((k - local_start_frame) / local_duration, 0.0, 1.0)
                lateral_series[k] = local_start_lateral + float(alpha) * (
                    local_end_lateral - local_start_lateral
                )
                if local_start_frame <= k < local_start_frame + local_duration:
                    lateral_speed[k] = (local_end_lateral - local_start_lateral) / (
                        local_duration * dt_s
                    )
        elif mode == "crossing":
            crossing_start = local_conflict_frame - local_duration // 2
            for k in range(T):
                alpha = np.clip((k - crossing_start) / local_duration, 0.0, 1.0)
                lateral_series[k] = local_start_lateral + float(alpha) * (
                    local_end_lateral - local_start_lateral
                )
                if crossing_start <= k < crossing_start + local_duration:
                    lateral_speed[k] = (local_end_lateral - local_start_lateral) / (
                        local_duration * dt_s
                    )

        xs = np.empty(T)
        ys = np.empty(T)
        zs = np.empty(T)
        hd = np.empty(T)
        vx = np.zeros(T)
        vy = np.zeros(T)
        for k in range(T):
            s = start_s + longitudinal[k]
            x, y, z, route_h = _sample(s)
            current_lateral = lateral_series[k]
            x += -np.sin(route_h) * current_lateral
            y += np.cos(route_h) * current_lateral
            actor_h = route_h + yaw_off
            if abs(longitudinal_speed[k]) + abs(lateral_speed[k]) > 1e-6:
                actor_h = route_h + np.arctan2(lateral_speed[k], longitudinal_speed[k]) + yaw_off
            xs[k], ys[k], zs[k], hd[k] = x, y, z, actor_h
            vx[k] = longitudinal_speed[k] * np.cos(route_h) - lateral_speed[k] * np.sin(route_h)
            vy[k] = longitudinal_speed[k] * np.sin(route_h) + lateral_speed[k] * np.cos(route_h)

        if not archetype:
            # Keep the baked appearance: rewrite the source track in place.
            st["position"] = np.stack([xs, ys, zs], axis=1).astype(np.float32)
            st["heading"] = hd.astype(np.float32)
            st["velocity"] = np.stack([vx, vy], axis=1).astype(np.float32)
            st["valid"] = np.ones(T, bool)
            src.setdefault("metadata", {})["relocated"] = mode
            logger.info(
                "relocate_agent: %s -> baked %s at arc %.1fm start=(%.1f,%.1f) "
                "speed=%.1f lateral=%.1f",
                src_id,
                mode,
                start_s,
                xs[0],
                ys[0],
                speed if mode != "static" else 0.0,
                float(lateral_series[0]),
            )
        else:
            # Swap in an imported asset on the synthetic trajectory.
            length, width, height = _OBSTACLE_ARCHETYPES.get(archetype, _OBSTACLE_ARCHETYPES["car"])
            track_type = _ARCHETYPE_TRACK_TYPE.get(archetype, "VEHICLE")
            obj_id = f"relocated_{n}"
            new: dict[str, Any] = {
                "type": track_type,
                "state": {
                    # base-origin asset: base pinned to the route road z.
                    "position": np.stack([xs, ys, (zs - float(ego_z_to_ground_m))], axis=1).astype(
                        np.float32
                    ),
                    "length": np.full(T, length, np.float32),
                    "width": np.full(T, width, np.float32),
                    "height": np.full(T, height, np.float32),
                    "heading": hd.astype(np.float32),
                    "velocity": np.stack([vx, vy], axis=1).astype(np.float32),
                    "valid": np.ones(T, bool),
                },
                "metadata": {
                    "track_length": T,
                    "type": track_type,
                    "object_id": obj_id,
                    "injected_obstacle": True,
                    "obstacle_archetype": archetype,
                    "relocated_from": src_id,
                },
            }
            asset_id = (nurec_asset_ids or {}).get(archetype)
            if asset_id:
                new["metadata"]["nurec_asset_id"] = str(asset_id)
                new["metadata"]["nurec_semantic_class"] = archetype
            del tracks[src_id]
            tracks[obj_id] = new
            logger.info(
                "relocate_agent: %s -> %s (%s %s) at arc %.1fm start=(%.1f,%.1f) "
                "speed=%.1f asset=%s",
                src_id,
                obj_id,
                mode,
                archetype,
                start_s,
                xs[0],
                ys[0],
                speed if mode != "static" else 0.0,
                asset_id,
            )
        n += 1
    logger.info("relocate_agent: relocated %d actor(s)", n)
    return sd


def remove_tracks(sd: dict, track_ids: List[str] | None = None, **_ignored) -> dict:
    """Drop named non-ego agents from the scenario.

    The fourth edit op. Deleting the track removes the actor from BOTH the
    render (the nurec_grpc mirror relocates any actor absent from
    ``agent_states`` off-screen) and the sim state / scoring, so render and sim
    stay consistent. The ego is never removable.
    """
    if not track_ids:
        return sd
    tracks = sd.get("tracks", {})
    sdc_id = str((sd.get("metadata") or {}).get("sdc_id", "ego"))
    n = 0
    for raw in track_ids:
        tid = str(raw)
        if tid == sdc_id:
            raise ValueError("remove_tracks refuses to delete the ego track")
        if tid not in tracks:
            logger.warning("remove_tracks: track %r not found; skipping", tid)
            continue
        del tracks[tid]
        n += 1
    logger.info("remove_tracks: removed %d/%d track(s)", n, len(track_ids))
    return sd


def _host_frame_grid(sd: dict, frames: Dict[str, Any] | None, *,
                     label: str, why: str = "") -> tuple:
    """``(T, dt)`` of the host, refused if it disagrees with the recipe's.

    Shared by both recipe edit tools. The checks are not paperwork: ``T``
    decides whether an actor exists for the whole episode, and ``dt`` is the
    integration step for every speed-derived quantity — a baked velocity, or
    an IDM acceleration. A host that runs at a different dt plays the same
    scenario at the wrong speed, which is exactly the kind of error that
    survives review because everything still runs.
    """
    meta = sd.get("metadata", {})
    tracks = sd.get("tracks", {})
    sdc_id = str(meta.get("sdc_id", "ego"))
    ego = tracks.get(sdc_id)
    if not ego:
        raise ValueError(f"{label}: no ego track {sdc_id!r} in the scenario")
    host_T = int(np.asarray(ego.get("state", {}).get("position")).shape[0])

    frames = dict(frames or {})
    recipe_T = int(frames.get("T", host_T))
    if recipe_T != host_T:
        raise ValueError(
            f"{label} was baked for T={recipe_T} frames but this host has {host_T}. "
            f"{why or 'The episode length is part of the scenario'}, so replaying it here "
            f"would silently truncate or run off the end of the episode."
        )
    recipe_dt = frames.get("dt_s")
    host_dt = scenario_dt_seconds(meta, default=float(recipe_dt) if recipe_dt else 0.1)
    if recipe_dt is not None and abs(float(recipe_dt) - host_dt) > 1e-4:
        raise ValueError(
            f"{label} was baked at dt={float(recipe_dt):.4f}s but this host runs at "
            f"dt={host_dt:.4f}s. Every speed-derived quantity scales with dt, so the "
            f"scenario would play out at the wrong speed."
        )
    recipe_ts = frames.get("timestamps_us")
    host_ts = meta.get("ts")
    if recipe_ts and host_ts is not None:
        host_arr = np.asarray(host_ts, np.float64).reshape(-1)
        recipe_arr = np.asarray(recipe_ts, np.float64).reshape(-1)
        if host_arr.size == recipe_arr.size and not np.allclose(host_arr, recipe_arr, atol=1.0):
            raise ValueError(
                f"{label} frame timestamps disagree with this host's (max delta "
                f"{float(np.max(np.abs(host_arr - recipe_arr))):.0f} us). This is a "
                f"different scenario window than the one the recipe was baked against."
            )
    return host_T, host_dt


def spawn_reactive_actor(
    sd: dict,
    actors: List[Dict[str, Any]] | None = None,
    frames: Dict[str, Any] | None = None,
    recipe_id: str = "",
    variant: str = "",
    **_ignored,
) -> dict:
    """Declare the actors a recipe inserted, and hand their motion to a policy.

    The tool a frozen NavSafe recipe produces, for actors whose trajectory is
    not knowable in advance because it depends on what the ego does. The two
    split the same job along a seam that did not exist before:

    * this writes the **declaration** — the actor exists, it is this asset,
      this big, this class, present for these frames. Every downstream
      consumer (asset insertion into the reconstruction, dims for collision,
      the renderer's track list) reads that and is unchanged.
    * the traffic manager writes the **motion**, each frame, into
      ``pose_overrides``.

    The track therefore spawns holding its initial pose. That is not a
    trajectory and must not be read as one: if the manager never runs, the
    actor stands still, which is a visible failure rather than a plausible
    wrong answer. That was the deliberate choice over seeding it with a
    constant-velocity guess, which would look like a working scenario.

    The recipe stores only ``spawn`` and ``policy``; the specs are parked in
    the scenario metadata under ``navsafe_reactive`` for the env to hand to
    the manager once the scene is loaded.
    """
    if not actors:
        return sd

    # Local import: keeps the ``navsafe`` package out of the scenario
    # package's import graph (this module is pulled in at every env reset).
    from navsafe.benchmark.editing.recipe.schema import (
        FROZEN_ASSET_ID_KEY,
        RecipeError,
        spec_digest,
    )

    tracks = sd.setdefault("tracks", {})
    label = recipe_id or "<recipe>"
    host_T, _dt = _host_frame_grid(
        sd, frames, label=f"spawn_reactive_actor: {label}",
        why="The actor's declaration spans the episode")
    reactive: List[Dict[str, Any]] = []

    for n, spec in enumerate(actors):
        spec = dict(spec)
        name = str(spec.get("name", f"actor_{n}"))
        where = f"{label} actor {name!r}"
        op = str(spec.get("op", "insert"))
        src_id = str(spec.get("source_track_id", ""))

        # Re-checked HERE, at the last gate before the sim, so a spec
        # assembled by hand — bypassing replay.py — cannot slip past.
        #
        # Hash the path the recipe was FROZEN with, not the one that will be
        # opened. `replay_spec` puts `nurec_asset_id` in the payload, and
        # NAVSAFE_ASSET_BANK rewrites it to the reader's own bank, so hashing
        # the live value made every relocated bank look like tampering and
        # broke every actor-bearing recipe at env reset. The asset's CONTENT is
        # still pinned, by `asset_sha256` inside this same payload — moving a
        # byte-identical file cannot pass off different gaussians.
        frozen_asset_id = spec.pop(FROZEN_ASSET_ID_KEY, None)
        if op != "remove":
            recorded = str(spec.get("sha256", ""))
            if not recorded:
                raise RecipeError(f"spawn_reactive_actor: {where} carries no sha256; "
                                  f"refusing to build an unverified scenario")
            as_frozen = spec
            if frozen_asset_id:
                as_frozen = dict(spec)
                as_frozen["nurec_asset_id"] = frozen_asset_id
            actual = spec_digest(as_frozen)
            if actual != recorded:
                raise RecipeError(
                    f"spawn_reactive_actor: {where} sha256 mismatch — recorded "
                    f"{recorded[:12]}…, computed {actual[:12]}…. The spawn, the "
                    f"policy or the asset was edited after freezing.")

        if op == "remove":
            remove_tracks(sd, [src_id])
            logger.info("spawn_reactive_actor: %s removed %s", where, src_id)
            continue
        if op in ("replace", "relocate"):
            # The host's own actor is re-tasked: it stops being itself and
            # becomes this one, driven by a policy. Its logged track goes, so
            # the two cannot both be in the scene.
            if src_id not in tracks:
                raise ValueError(f"spawn_reactive_actor: {where} source track {src_id!r} not found")
            del tracks[src_id]
        elif op != "insert":
            raise ValueError(f"spawn_reactive_actor: {where} unknown op {op!r}")

        spawn = dict(spec.get("spawn") or {})
        policy = dict(spec.get("policy") or {})
        if not policy.get("kind"):
            raise ValueError(f"spawn_reactive_actor: {where} declares no policy.kind")

        position = np.asarray(spawn.get("position", (0.0, 0.0, 0.0)), np.float64)
        if position.shape[-1] < 3:
            position = np.array([position[0], position[1], 0.0])
        heading = float(spawn.get("heading", 0.0))
        velocity = np.asarray(spawn.get("velocity", (0.0, 0.0)), np.float64)[:2]

        dims = spec.get("dims")
        if dims is None:
            raise ValueError(
                f"spawn_reactive_actor: {where} needs asset dims [length, width, height]")
        length, width, height = (float(v) for v in dims)

        track_type = str(spec.get("track_type", "VEHICLE"))
        obj_id = f"navsafe_{name}"
        track_meta = {
            "track_length": host_T,
            "type": track_type,
            "object_id": obj_id,
            "injected_obstacle": True,
            "navsafe_actor": name,
            "navsafe_recipe_id": recipe_id,
            "navsafe_op": op,
            "navsafe_policy": str(policy.get("kind")),
        }
        if src_id:
            track_meta["replaced_track_id"] = src_id
        if variant:
            track_meta["navsafe_variant"] = variant
        if spec.get("registry_key"):
            track_meta["obstacle_archetype"] = str(spec["registry_key"])
        asset_id = spec.get("nurec_asset_id")
        if asset_id:
            track_meta["nurec_asset_id"] = str(asset_id)
            track_meta["nurec_semantic_class"] = str(
                spec.get("semantic_class") or spec.get("registry_key") or name
            )
        # A walking actor names a bank of posed copies instead of relying on
        # its single static asset; the renderer swaps between them per frame.
        # Carried alongside nurec_asset_id, never instead of it: the static
        # asset stays the actor's identity and its content hash.
        if spec.get("nurec_pose_bank"):
            track_meta["nurec_pose_bank"] = str(spec["nurec_pose_bank"])
        tracks[obj_id] = {
            "type": track_type,
            "state": {
                "position": np.repeat(position[None, :].astype(np.float32), host_T, axis=0),
                "length": np.full(host_T, length, np.float32),
                "width": np.full(host_T, width, np.float32),
                "height": np.full(host_T, height, np.float32),
                "heading": np.full(host_T, heading, np.float32),
                "velocity": np.repeat(velocity[None, :].astype(np.float32), host_T, axis=0),
                "valid": np.ones(host_T, bool),
            },
            "metadata": track_meta,
        }
        reactive.append({
            "name": name,
            # Keyed by TRACK id, not actor name: the manager publishes
            # pose_overrides into the agent list, which is keyed by track id.
            "track_id": obj_id,
            "policy": policy,
            "spawn": {**spawn, "position": position.tolist(), "heading": heading,
                      "length": length, "width": width},
        })
        logger.info("spawn_reactive_actor: %s -> %s policy=%s at (%.1f, %.1f), asset=%s",
                    where, obj_id, policy.get("kind"),
                    float(position[0]), float(position[1]), asset_id)

    meta = sd.setdefault("metadata", {})
    meta.setdefault("navsafe_reactive", []).extend(reactive)
    logger.info("spawn_reactive_actor: %s [%s] declared %d reactive actor(s) over T=%d",
                label, variant or "e_plus", len(reactive), host_T)
    return sd


EDIT_TOOLS: Dict[str, Callable[..., dict]] = {
    "spawn_reactive_actor": spawn_reactive_actor,
    "place_static_obstacles": place_static_obstacles,
    "replace_agent_with_asset": replace_agent_with_asset,
    "relocate_agent": relocate_agent,
    "remove_tracks": remove_tracks,
}


def apply_scenario_edits(sd: dict, edits: List[Dict[str, Any]] | None) -> dict:
    """Apply a list of ``{"tool": name, **params}`` edit specs to ``sd``.

    Idempotent per scenario: a marker in ``sd["metadata"]`` prevents double
    application when the env resets more than once with the same dict.
    """
    if not edits:
        return sd
    meta = sd.setdefault("metadata", {})
    if meta.get("_scenario_edits_applied"):
        return sd
    for spec in edits:
        spec = dict(spec)
        name = spec.pop("tool", None)
        tool = EDIT_TOOLS.get(str(name))
        if tool is None:
            raise ValueError(f"Unknown scenario edit tool {name!r}; known: {sorted(EDIT_TOOLS)}")
        sd = tool(sd, **spec)
    meta["_scenario_edits_applied"] = True
    return sd
