# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode outcomes for the Section 4.3.2 controlled hazard experiment.

The four NavSafe headline metrics (``driving_score`` / ``success`` /
``efficiency_pct`` / ``comfort``) do not say *how* an episode went past the
hazard, and 4.3.2 needs exactly that: did the ego brake, how early, did it
stop and for how long, how close did it come to the manipulated actor, and did
it ever get past the conflict point. Those are what separate "improved hazard
resilience" from "more conservative behaviour" -- an RL checkpoint that avoids
the collision by stopping dead is not the same result as one that threads past.

Everything here is derived from the stored trace, so it can be recomputed on a
finished run without re-executing a policy. Nothing in this module changes a
score; it only describes.

The per-frame trace already carries what is needed
(``navsafe/benchmark/trace/from_eval.py``): each frame has ``ego_speed``,
``ego_accel`` and ``min_clearance_m``, and each agent row inside it carries
``id``, ``cls``, ``ttc_s``, ``clearance_m`` and ``dist_to_ego``.

**Target actor.** The controlled experiment manipulates ONE inserted actor. Its
recipe id is passed in as ``target_ids``; every "to_target" quantity is
restricted to those rows, and is ``None`` when the actor never appears (which
is itself a finding -- an insert that silently did not render).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = ["controlled_outcomes", "outcomes_from_dir",
           "BRAKE_ACCEL_MS2", "STOPPED_SPEED_MS"]

#: Longitudinal deceleration that counts as "braking". Below this the ego is
#: coasting: the tracker's own speed control produces small negative accels
#: continuously (``_speed_control`` clips the error at zero and has no
#: proportional band), so a 0-threshold would report braking on nearly every
#: frame of every episode and the onset would be meaningless.
BRAKE_ACCEL_MS2 = -1.0

#: Below this the ego counts as stopped. Not zero: the LQR tracker leaves a
#: few cm/s of residual even when commanded to a halt.
STOPPED_SPEED_MS = 0.2

#: How close the ego centre must come to the conflict point to count as having
#: reached it. The conflict point is where the actor's reference line crosses
#: the ego route, so passing "through" it means passing within roughly a lane.
CONFLICT_PASS_RADIUS_M = 3.0


def _finite(values: Iterable[float]) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _agent_rows(frame: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = frame.get("agents")
    return rows if isinstance(rows, (list, tuple)) else ()


def controlled_outcomes(
    frames: Sequence[Mapping[str, Any]],
    *,
    dt: float,
    scored_from: int = 0,
    target_ids: Optional[Iterable[str]] = None,
    conflict_xy: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Describe how one episode handled its hazard.

    Args:
        frames: the trace rows, in order.
        dt: seconds per frame.
        scored_from: first scored frame index; warm-up frames are replayed and
            say nothing about the policy, so they are excluded from every
            quantity here.
        target_ids: recipe ids of the manipulated actor(s).
        conflict_xy: where the actor's reference line crosses the ego route.

    Returns:
        A dict of plain JSON types. A quantity that cannot be measured is
        ``None`` rather than a sentinel, so an unmeasured episode is never
        averaged in as a zero.
    """
    scored = [f for f in frames if int(f.get("frame", -1)) >= scored_from]
    if not scored:
        return {"scored_frames": 0}

    targets = {str(t) for t in (target_ids or ())}

    speeds = np.asarray(_finite(f.get("ego_speed") for f in scored), float)
    accels = np.asarray(_finite(f.get("ego_accel") for f in scored), float)

    # --- braking -----------------------------------------------------------
    brake_onset_frame = None
    brake_onset_s = None
    for f in scored:
        a = f.get("ego_accel")
        try:
            a = float(a)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and a <= BRAKE_ACCEL_MS2:
            brake_onset_frame = int(f.get("frame", -1))
            brake_onset_s = round((brake_onset_frame - scored_from) * dt, 3)
            break

    # --- stopped time ------------------------------------------------------
    stopped_frames = int(np.count_nonzero(speeds <= STOPPED_SPEED_MS)) if speeds.size else 0
    longest = run = 0
    for f in scored:
        s = f.get("ego_speed")
        try:
            s = float(s)
        except (TypeError, ValueError):
            s = float("inf")
        if math.isfinite(s) and s <= STOPPED_SPEED_MS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0

    # --- clearance and TTC -------------------------------------------------
    any_ttc: list[float] = []
    tgt_ttc: list[float] = []
    any_clear: list[float] = []
    tgt_clear: list[float] = []
    target_seen = 0
    for f in scored:
        mc = f.get("min_clearance_m")
        try:
            mc = float(mc)
        except (TypeError, ValueError):
            mc = float("nan")
        if math.isfinite(mc):
            any_clear.append(mc)
        for a in _agent_rows(f):
            t = a.get("ttc_s")
            c = a.get("clearance_m")
            is_target = str(a.get("id")) in targets
            if is_target:
                target_seen += 1
            try:
                tf = float(t)
                if math.isfinite(tf):
                    any_ttc.append(tf)
                    if is_target:
                        tgt_ttc.append(tf)
            except (TypeError, ValueError):
                pass
            try:
                cf = float(c)
                if math.isfinite(cf) and is_target:
                    tgt_clear.append(cf)
            except (TypeError, ValueError):
                pass

    # --- did the ego get past the conflict point ---------------------------
    passed = None
    closest_to_conflict = None
    if conflict_xy is not None and len(conflict_xy) >= 2:
        cx, cy = float(conflict_xy[0]), float(conflict_xy[1])
        d = [math.hypot(float(f.get("ego_x", np.nan)) - cx,
                        float(f.get("ego_y", np.nan)) - cy) for f in scored]
        d = _finite(d)
        if d:
            closest_to_conflict = round(min(d), 3)
            passed = bool(min(d) <= CONFLICT_PASS_RADIUS_M)

    def _r(x, n=3):
        return None if x is None else round(float(x), n)

    return {
        "scored_frames": len(scored),
        # braking
        "braking_onset_frame": brake_onset_frame,
        "braking_onset_s": brake_onset_s,
        "braked": brake_onset_frame is not None,
        "min_accel_ms2": _r(float(accels.min())) if accels.size else None,
        # stopping
        "stopped_frames": stopped_frames,
        "stopped_duration_s": round(stopped_frames * dt, 3),
        "longest_stop_s": round(longest * dt, 3),
        "min_speed_ms": _r(float(speeds.min())) if speeds.size else None,
        # proximity -- the manipulation check for the spatial axis
        "min_clearance_m": _r(min(any_clear)) if any_clear else None,
        "min_clearance_to_target_m": _r(min(tgt_clear)) if tgt_clear else None,
        "min_ttc_s": _r(min(any_ttc)) if any_ttc else None,
        "min_ttc_to_target_s": _r(min(tgt_ttc)) if tgt_ttc else None,
        # was the manipulated actor even there?
        "target_ids": sorted(targets) or None,
        "target_frames_seen": target_seen or None,
        # progress past the hazard
        "conflict_xy": [round(float(conflict_xy[0]), 2),
                        round(float(conflict_xy[1]), 2)] if conflict_xy is not None
                       and len(conflict_xy) >= 2 else None,
        "closest_approach_to_conflict_m": closest_to_conflict,
        "passed_conflict_zone": passed,
    }


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------

def _load_agent_states(eval_dir) -> Optional[dict]:
    """``agent_states.json``, or None when the run was not asked to write it."""
    import json
    from pathlib import Path

    p = Path(eval_dir) / "agent_states.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001 -- a truncated dump is "no record"
        return None


def _frames_from_agent_states(doc: Mapping[str, Any],
                              ego_xy: np.ndarray) -> list[dict]:
    """Trace-shaped rows built from the per-frame actor dump plus ego path.

    ``agent_states.json`` is the ONLY record of where a recipe-inserted actor
    actually went (the scenario's ``tracks`` hold its spawn pose repeated), so
    clearance and TTC to a manipulated hazard have to be recomputed from it
    rather than from the scenario-derived trace.

    Clearance here is centre-to-centre minus half the two footprints along the
    line joining them -- the same quantity the live trace calls
    ``clearance_m``, to within the box-vs-circle difference. TTC is
    range/closing-rate, ``inf`` when not closing, which is how the trace
    defines it too.
    """
    rows: list[dict] = []
    prev: dict[str, tuple[float, float]] = {}
    dt = float(doc.get("dt", 0.1) or 0.1)
    # ego half-extent: the NavSafe ego box, used only for the footprint margin
    EGO_HALF = 1.0
    for i, fr in enumerate(doc.get("frames", [])):
        k = int(fr.get("frame", i))
        if k >= len(ego_xy):
            break
        ex, ey = float(ego_xy[k][0]), float(ego_xy[k][1])
        epx, epy = (float(ego_xy[k - 1][0]), float(ego_xy[k - 1][1])) if k > 0 else (ex, ey)
        agents = []
        for a in fr.get("agents", ()):
            try:
                ax, ay = float(a["x"]), float(a["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(ax) and math.isfinite(ay)):
                continue
            aid = str(a.get("id", ""))
            d = math.hypot(ax - ex, ay - ey)
            half = 0.5 * max(float(a.get("width") or 0.0), float(a.get("length") or 0.0))
            clearance = max(d - EGO_HALF - half, 0.0)
            # closing rate from the previous frame's separation
            ttc = float("inf")
            if aid in prev:
                pd = math.hypot(prev[aid][0] - epx, prev[aid][1] - epy)
                rate = (d - pd) / dt
                if rate < -1e-3:
                    ttc = d / -rate
            prev[aid] = (ax, ay)
            agents.append({"id": aid, "x": ax, "y": ay,
                           "clearance_m": clearance, "ttc_s": ttc})
        rows.append({"frame": k, "ego_x": ex, "ego_y": ey, "agents": agents,
                     "min_clearance_m": min((g["clearance_m"] for g in agents),
                                            default=float("nan"))})
    return rows


def outcomes_from_dir(eval_dir, *, warmup_frames: int = 0,
                      target_ids: Optional[Iterable[str]] = None,
                      conflict_xy: Optional[Sequence[float]] = None,
                      trace_frames: Optional[Sequence[Mapping[str, Any]]] = None,
                      dt: float = 0.1) -> dict[str, Any]:
    """Controlled outcomes for a finished run directory.

    Ego kinematics come from ``trace_frames`` when the caller already built
    the trace (it carries the smoothed accel the braking test needs);
    proximity to the manipulated actor comes from ``agent_states.json``, which
    is the only place an inserted actor's driven path exists.

    Returns a dict with ``agent_states: "missing"`` when the run was not asked
    to dump them -- an absent record must not read as "never got close".
    """
    from pathlib import Path

    doc = _load_agent_states(eval_dir)
    out: dict[str, Any] = {}
    if trace_frames:
        out = controlled_outcomes(trace_frames, dt=dt, scored_from=warmup_frames,
                                  target_ids=target_ids, conflict_xy=conflict_xy)
    if doc is None:
        out.setdefault("scored_frames", 0)
        out["agent_states"] = "missing"
        # Without the dump these three cannot be measured for an INSERTED
        # actor; say so rather than reporting the scenario-track value, which
        # is that actor's spawn pose repeated.
        for k in ("min_clearance_to_target_m", "min_ttc_to_target_s",
                  "target_frames_seen"):
            out[k] = None
        return out

    ego = np.load(Path(eval_dir) / "vehicle_states.npy")[:, :2]
    rows = _frames_from_agent_states(doc, ego)
    prox = controlled_outcomes(rows, dt=float(doc.get("dt", dt) or dt),
                               scored_from=warmup_frames,
                               target_ids=target_ids, conflict_xy=conflict_xy)
    if not out:
        out = prox
    else:
        # ego-derived quantities stay from the trace; proximity from the dump
        for k in ("min_clearance_m", "min_clearance_to_target_m",
                  "min_ttc_s", "min_ttc_to_target_s", "target_ids",
                  "target_frames_seen"):
            out[k] = prox.get(k)
    out["agent_states"] = "recorded"
    out["agent_state_frames"] = len(rows)
    return out
