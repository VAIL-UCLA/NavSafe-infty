"""Reconstruct trace-shaped frames from a finished the evaluator run.

The canonical trace is written live by :mod:`navsafe.benchmark.trace.writer`,
but it only runs when ``NAVSAFE_TRACE`` points at a seed -- i.e. on NavSafe
seeds with frozen regions. Other evaluations leave the evaluator's own
artifacts, which this module converts for the same scoring chain.

This module closes that gap: it rebuilds the subset of the trace schema that
those artifacts support, so **one** scoring chain -- termination, infractions,
NavSafe metrics -- runs over both kinds of run.

What it can and cannot recover, stated rather than papered over:

* **Ego kinematics** are derived from ``vehicle_states.npy`` (positions only),
  so heading, speed, acceleration and jerk are finite differences of a
  position grid, not measurements.  Positions are recorded post-teleport, so
  this is the same construction the live writer uses.
* **On-drivable** comes from the per-frame ``DAC`` column of
  ``driving_score_summary.csv``.  When the scenario carries no map
  (``map_features`` empty),
  DAC is 0 for every frame *because there is no drivable area to test
  against*, which is not the same fact as "the ego left the road".  In that
  case drivability is marked **unknown**: ``on_drivable`` stays True, and
  ``drivable_known=False`` is reported so no caller can quietly read the
  absence of a map as compliance.
* **Driving direction** comes from the per-frame ``DDC`` column, gated the same
  way: 0.0 there means the ego drove past the full-violation distance against
  the local traffic direction, which ENDS the episode as ``wrong_way`` and is
  charged to the policy. Without a map there is no DDC to read and the frame is
  marked compliant, because "not checked" must never read as "wrong way".
* **Contacts** come from the per-frame ``COLL`` / ``COLL_AF`` columns, which
  carry the contact and its fault attribution, and from ``COLL_ID`` /
  ``COLL_KIND``, which carry WHICH agent it was with and what kind.  Only when
  those two are absent is the pair inferred from the nearest agent's class, as
  the live hook does -- a guess that can name the wrong vehicle.  Runs recorded before
  those columns existed fall back to ``NC`` (EPDMS ``no_at_fault_collisions``),
  which is an at-fault flag only — for those, a not-at-fault contact leaves no
  trace and ``contact_not_at_fault`` can never be reported.
* **Regions** (intersection, conflict zone, exit lane) are seed concepts with
  no equivalent here; they stay at their schema defaults, so rubric predicates
  that need them must not be run over these frames.  The NavSafe metrics and
  the termination taxonomy do not use them.
* **Signals** are the one exception: ``signal_state`` is set to ``"red"`` on
  frames the evaluator's ``TL`` column already flagged as a red-light
  violation, so the ``red_light`` penalty channel is measurable instead of
  silently skipped.  It is the evaluator's verdict, not the light's colour --
  ``predicates.comply`` still must not be run over these frames, because its
  t_react window needs the real per-frame signal.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from navsafe.benchmark.trace import regions_from_map

from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M
from navsafe.benchmark.trace.schema import PHASE_SCORED, PHASE_WARMUP, empty_frame
from navsafe.benchmark.trace.writer import (
    AGENT_RADIUS_M, CLOSING_EPS_MPS, _box_distance, lane_centerlines, nearest_lane,
)

# py123d / MetaDrive track types -> the trace's agent classes.
_CLS = {"VEHICLE": "vehicle", "CAR": "vehicle", "TRUCK": "vehicle",
        "BUS": "vehicle", "PEDESTRIAN": "pedestrian", "CYCLIST": "bicycle",
        "BICYCLE": "bicycle", "MOTORCYCLE": "vehicle"}
# A track whose net displacement is below this is parked; Bench2Drive's
# background set is traffic-manager traffic, so parked cars must not drag the
# background mean speed towards zero.
PARKED_NET_DISPLACEMENT_M = 5.0
# A track also counts as parked if it never exceeds walking pace during the
# episode. Queued vehicles that creep forward pass the displacement test but
# would pull the background mean speed towards zero and inflate efficiency. A
# vehicle that drives and then stops at a light stays in the pool.
PARKED_PEAK_SPEED_MS = 1.0


@dataclass
class EvalRun:
    """Everything the scoring chain needs from one evaluator output directory."""

    frames: list[dict]
    route_xy: np.ndarray                  # the logged ego path (the route)
    ego_xy: np.ndarray                    # the driven path
    dt: float
    warmup_frames: int
    drivable_known: bool
    collision_count: int
    # The evaluator's own metrics.json, kept so a caller can quote the EPDMS
    # numbers beside the Bench2Drive ones instead of conflating them.
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def scored(self) -> list[dict]:
        return [f for f in self.frames if f["phase"] == PHASE_SCORED]


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

#: Per-frame columns that are text, not scores, and must survive the float()
#: filter below: the contact's agent id and kind, as the evaluator recorded
#: them.
_TEXT_COLS = ("COLL_ID", "COLL_KIND")


def read_per_frame_epdms(eval_dir: Path) -> dict[int, dict[str, Any]]:
    """Per-frame EPDMS subscores from ``driving_score_summary.csv``.

    Returns ``{frame_id: {"NC": .., "DAC": .., ...}}``; missing file -> empty,
    which the caller must treat as "unknown", never as "clean".

    ``COLL_ID`` / ``COLL_KIND`` come through as strings when present.
    """
    path = Path(eval_dir) / "driving_score_summary.csv"
    if not path.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            fid = row.get("frame_id", "")
            if not str(fid).isdigit():
                continue                     # the AVERAGE row
            vals: dict[str, Any] = {}
            for k, v in row.items():
                if k in ("frame_id", "valid") or v in ("", None):
                    continue
                if k in _TEXT_COLS:
                    vals[k] = str(v)
                    continue
                try:
                    vals[k] = float(v)
                except ValueError:
                    continue
            out[int(fid)] = vals
    return out


def _agent_tracks(sd: Mapping, ego_id: str, *, dt: float = 0.1) -> list[dict]:
    """Non-ego tracks as arrays, with the parked ones flagged."""
    out = []
    for aid, tr in (sd.get("tracks") or {}).items():
        if aid == ego_id:
            continue
        st = tr.get("state", {})
        P = np.asarray(st.get("position", []), dtype=np.float64)
        if P.ndim != 2 or len(P) < 2:
            continue
        valid = np.asarray(st.get("valid", [True] * len(P)), dtype=bool)
        head = np.asarray(st.get("heading", np.zeros(len(P))), dtype=np.float64).reshape(-1)
        moved = float(np.linalg.norm(P[-1, :2] - P[0, :2]))
        # Peak speed over consecutively-valid frames, by the same finite
        # difference build_frames uses for the per-frame speed. Steps across an
        # invalid gap are skipped: a track that disappears and reappears
        # elsewhere would otherwise show one enormous spurious step.
        step_ok = valid[:-1] & valid[1:]
        peak = 0.0
        if step_ok.any():
            steps = np.hypot(*np.diff(P[:, :2], axis=0).T)[step_ok]
            peak = float(steps.max() / dt) if len(steps) else 0.0
        out.append({
            "id": str(aid),
            "cls": _CLS.get(str(tr.get("type", "VEHICLE")).upper(), "vehicle"),
            "xy": P[:, :2], "valid": valid, "heading": head,
            "length": float(np.asarray(st.get("length", [4.5])).reshape(-1)[0]),
            "width": float(np.asarray(st.get("width", [1.9])).reshape(-1)[0]),
            "parked": (moved < PARKED_NET_DISPLACEMENT_M
                       or peak < PARKED_PEAK_SPEED_MS),
            "peak_speed": peak,
        })
    return out


# --------------------------------------------------------------------------
# frame construction
# --------------------------------------------------------------------------

def _ego_kinematics(xy: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Heading, speed, accel, jerk and lateral accel from positions alone."""
    from scipy.signal import savgol_filter

    n = len(xy)
    if n < 2:
        z = np.zeros(max(n, 1))
        return {"yaw": z, "speed": z, "accel": z, "jerk": z, "lat_accel": z,
                "yaw_rate": z}
    d = np.diff(xy, axis=0)
    heading = np.unwrap(np.concatenate(
        [[math.atan2(d[0, 1], d[0, 0])], np.arctan2(d[:, 1], d[:, 0])]))
    win = min(11, n - (1 - n % 2))
    if win >= 5:
        vx = savgol_filter(xy[:, 0], win, 3, deriv=1, delta=dt)
        vy = savgol_filter(xy[:, 1], win, 3, deriv=1, delta=dt)
        speed = np.hypot(vx, vy)
        yaw_rate = np.gradient(savgol_filter(heading, win, 3), dt)
    else:
        step = np.hypot(*d.T) / dt
        speed = np.concatenate([[step[0]], step])
        yaw_rate = np.gradient(heading, dt)
    accel = np.gradient(speed, dt)
    return {"yaw": heading, "speed": speed, "accel": accel,
            "jerk": np.gradient(accel, dt), "lat_accel": speed * yaw_rate,
            "yaw_rate": yaw_rate}


def _deviation(route: np.ndarray, route_head: np.ndarray, p: np.ndarray,
               yaw: float) -> tuple[float, float, float]:
    """(lateral, longitudinal, yaw-deg) deviation from the nearest route pose."""
    j = int(np.argmin(np.linalg.norm(route - p, axis=1)))
    h = float(route_head[min(j, len(route_head) - 1)])
    dx, dy = float(p[0] - route[j, 0]), float(p[1] - route[j, 1])
    lon = math.cos(h) * dx + math.sin(h) * dy
    lat = -math.sin(h) * dx + math.cos(h) * dy
    dyaw = math.degrees((yaw - h + math.pi) % (2 * math.pi) - math.pi)
    return lat, lon, dyaw


#: How close a signalled connector's entry has to come to the logged route for
#: that signal to be the one governing this ego. Junctions carry a light per
#: approach and a map holds many junctions; without a bound, an unrelated light
#: across the map would supply the stop line.
STOPLINE_ROUTE_TOL_M = 6.0


def _arc(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline."""
    if len(path) < 2:
        return np.zeros(len(path))
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def _arc_at(path: np.ndarray, arc: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(arc length, distance) of the point on ``path`` nearest ``p``."""
    d = np.linalg.norm(path - np.asarray(p, dtype=np.float64)[:2], axis=1)
    i = int(np.argmin(d))
    return float(arc[i]), float(d[i])


def stopline_arc(sd: Mapping,
                 route_xy: np.ndarray,
                 lane_center: Mapping[str, np.ndarray] | None) -> float | None:
    """Route arc length of the stop line this ego approaches, or ``None``.

    nuPlan publishes no stop-line layer, so the line is the ENTRY of the
    signalled connector — the same geometry the evaluator's red-light rule uses
    (``_ran_red_light`` against the lane's entry plus ``TL_STOP_LINE_ZONE_M``),
    which is what keeps this column and the ``TL`` subscore talking about one
    place. Signalled lanes come from ``dynamic_map_states``; the one taken is
    the nearest entry ahead of the ego's start, within
    :data:`STOPLINE_ROUTE_TOL_M` of the route.

    ``None`` whenever that cannot be resolved (no lights, no centrelines, none
    near the route), and the column then stays NaN — "not checked", never a
    stop line at the origin.
    """
    lights = sd.get("dynamic_map_states") or {}
    if not lights or not lane_center or len(route_xy) < 2:
        return None
    arc = _arc(route_xy)
    best: float | None = None
    for lane_id in lights:
        centre = lane_center.get(str(lane_id))
        if centre is None or len(centre) == 0:
            continue
        centre = np.asarray(centre, dtype=np.float64)[:, :2]
        # Decide whether a signal governs the ego from the whole
        # connector, not its first vertex, which lies at the lane
        # edge and can be several metres from a route that follows
        # the lane.
        gap = float(np.min(np.linalg.norm(
            route_xy[:, None, :] - centre[None, :, :], axis=2)))
        if gap > STOPLINE_ROUTE_TOL_M:
            continue
        s, _ = _arc_at(route_xy, arc, centre[0])
        if s <= 0.0:
            continue
        best = s if best is None else min(best, s)
    return best


def build_frames(ego_xy: np.ndarray, route_xy: np.ndarray,
                 agents: Sequence[dict], per_frame: Mapping[int, Mapping[str, float]],
                 *, warmup_frames: int, dt: float,
                 drivable_known: bool,
                 lane_center: Mapping[str, np.ndarray] | None = None,
                 stopline_s: float | None = None,
                 region_flags: Mapping[str, Sequence[bool]] | None = None) -> list[dict]:
    """Trace-shaped frames for one run.  Pure function of the arrays it is given.

    ``lane_center`` are the map's lane centrelines
    (:func:`trace.writer.lane_centerlines`). With them the frames carry
    ``lane_id`` and the signed ``lateral_offset_m``, which is what the C-3
    lane-change gate (``navsafe/scenario_rules.py``) reads; without them those
    columns stay empty and the gate correctly reports "not checked" rather than
    failing every episode. Same function as the live writer uses, so the two
    paths cannot disagree about which lane the ego is in or about the sign.

    ``region_flags`` are per-frame ``in_intersection`` / ``in_exit_lane``
    booleans (:mod:`trace.regions_from_map`), for the event types whose rubric is a
    region rather than a goal -- V-8, whose objective is to NOT enter the turn
    lane a prohibitory plate forbids. Omitted, the columns are ABSENT rather
    than False: ``scenario_rules.hold_violation`` distinguishes the two, and a
    False there would read as "checked, and the ego stayed out" -- convicting
    nothing and acquitting everything. The same rule the DDC and drivable
    columns above follow: not checked must never read as compliant.
    """
    ego_xy = np.asarray(ego_xy, dtype=np.float64)[:, :2]
    route_xy = np.asarray(route_xy, dtype=np.float64)[:, :2]
    kin = _ego_kinematics(ego_xy, dt)
    rd = np.diff(route_xy, axis=0)
    route_head = np.concatenate(
        [[math.atan2(rd[0, 1], rd[0, 0])], np.arctan2(rd[:, 1], rd[:, 0])]) \
        if len(route_xy) > 1 else np.zeros(len(route_xy))

    route_arc = _arc(route_xy)

    rows: list[dict] = []
    for i, p in enumerate(ego_xy):
        r = empty_frame()
        yaw = float(kin["yaw"][i])
        speed = float(kin["speed"][i])
        scored = i >= warmup_frames
        lat, lon, dyaw = _deviation(route_xy, route_head, p, yaw)

        if stopline_s is not None:
            # Signed distance to the stop line along the logged
            # route, positive before the line; the same
            # convention as `trace/writer.py`.
            s_ego, _ = _arc_at(route_xy, route_arc, p)
            r["dist_to_stopline_m"] = float(stopline_s) - s_ego

        if lane_center:
            lane_id, lane_head, lane_off = nearest_lane(
                float(p[0]), float(p[1]), lane_center)
            r["lane_id"] = lane_id
            r["lane_heading"] = lane_head
            r["lateral_offset_m"] = lane_off

        rows_agents, min_clear = [], float("inf")
        for a in agents:
            if i + 1 >= len(a["xy"]) or not bool(a["valid"][i]):
                continue
            ax, ay = float(a["xy"][i, 0]), float(a["xy"][i, 1])
            d = math.dist((float(p[0]), float(p[1])), (ax, ay))
            if d > AGENT_RADIUS_M:
                continue
            # Difference only against a valid neighbouring frame.
            # Invalid frames hold positions in another coordinate
            # frame and would yield absurd speeds.
            if bool(a["valid"][i + 1]):
                avx = float(a["xy"][i + 1, 0] - ax) / dt
                avy = float(a["xy"][i + 1, 1] - ay) / dt
            elif i > 0 and bool(a["valid"][i - 1]):
                avx = float(ax - a["xy"][i - 1, 0]) / dt
                avy = float(ay - a["xy"][i - 1, 1]) / dt
            else:
                avx = avy = 0.0   # one isolated valid frame: no velocity exists
            evx, evy = speed * math.cos(yaw), speed * math.sin(yaw)
            ux, uy = ((ax - p[0]) / d, (ay - p[1]) / d) if d > 1e-6 else (0.0, 0.0)
            rate = (avx - evx) * ux + (avy - evy) * uy
            ego_closing = bool(-rate > CLOSING_EPS_MPS
                               and ((ax - p[0]) * evx + (ay - p[1]) * evy) > 0)
            ayaw = float(a["heading"][min(i, len(a["heading"]) - 1)])
            clearance = _box_distance(float(p[0]), float(p[1]), yaw,
                                      EGO_LENGTH_M, EGO_WIDTH_M,
                                      ax, ay, ayaw, a["length"], a["width"])
            min_clear = min(min_clear, clearance)
            fwd = math.cos(yaw) * (ax - p[0]) + math.sin(yaw) * (ay - p[1])
            side = -math.sin(yaw) * (ax - p[0]) + math.cos(yaw) * (ay - p[1])
            rows_agents.append({
                "id": a["id"], "cls": a["cls"],
                # Every agent replays its log here: the takeover set of the
                # semi-reactive manager is not recorded in these artifacts, so
                # claiming a reactive policy would be an invention.
                "policy": "parked" if a["parked"] else "replay",
                "x": ax, "y": ay, "yaw": ayaw, "speed": math.hypot(avx, avy),
                "length": a["length"], "width": a["width"],
                "dist_to_ego": d, "clearance_m": clearance,
                "is_lead": bool(fwd > 0.0 and abs(side) < 1.75),
                "in_conflict_zone": False,
                "ttc_s": float(d / -rate) if rate < -CLOSING_EPS_MPS else float("inf"),
                "ego_is_closing": ego_closing,
            })

        pf = per_frame.get(i, {})
        # DAC is a compliance score in [0, 1]; anything below 1 means the frame
        # was not fully compliant. Only trusted when a map exists.
        on_drivable = True
        if drivable_known and "DAC" in pf:
            on_drivable = pf["DAC"] >= 1.0
        # Driving-direction compliance scores 1.0 below 2 m driven
        # against traffic, 0.5 from 2 to 6 m and 0.0 beyond; only the
        # last ends the episode. Without a lane graph the score is
        # always 1.0, and `drivable_known` guards the same case here.
        driving_direction_ok = True
        if drivable_known and "DDC" in pf:
            driving_direction_ok = pf["DDC"] > 0.0
        # TL is the evaluator's per-frame traffic-light compliance
        # (1.0 compliant, 0.0 crossed on red at speed). It is read
        # back because this module has no map to recompute it.
        # `signal_state` therefore marks frames the evaluator
        # flagged, not frames on which a light was red.
        signal_state = ""
        if "TL" in pf:
            signal_state = "red" if pf["TL"] < 1.0 else "unknown"
        # TL_HOLD is the evaluator's "red light ahead on the ego's lane" state
        # fact (EPDMSLiveScorer._signal_hold_live). classify() exempts such
        # frames from the deadlock hold; an artifact without the column keeps
        # the plain rule, the same way the live monitor does without it.
        signal_hold = bool(pf.get("TL_HOLD", 0.0) > 0.0)
        # COLL/COLL_AF (written by the evaluator) carry the contact AND its
        # fault, so a not-at-fault contact is recoverable. Runs recorded before
        # those columns existed fall back to EPDMS's NC, which is an at-fault
        # flag only -- there, `contact_not_at_fault` can never fire, and that
        # is a property of the artifact, not of the episode.
        contacts = []
        has_coll_cols = "COLL" in pf
        hit = (pf.get("COLL", 0.0) > 0 if has_coll_cols
               else ("NC" in pf and pf["NC"] < 1.0))
        if scored and hit:
            near = min(rows_agents, key=lambda a: a["dist_to_ego"], default=None)
            # Use the evaluator's own record of which agent was
            # hit and how, when present. Guessing from the
            # nearest agent is the fallback for artifacts without
            # those columns.
            agent_id = str(pf.get("COLL_ID") or "")
            if not agent_id:
                agent_id = near["id"] if near else ""
            # rel_speed is only meaningful against the agent actually named.
            named = next((a for a in rows_agents if a["id"] == agent_id), near)
            # The penalty channel depends on what was hit, not on
            # how the boxes met: a pedestrian struck head-on is a
            # pedestrian collision. The evaluator's geometric
            # kinds are therefore mapped by the class of the
            # other agent first.
            geometry = str(pf.get("COLL_KIND") or "")
            if not geometry and near is not None:
                geometry = "rear_end" if near["is_lead"] else "angle"
            if (named or {}).get("cls") in ("pedestrian", "bicycle"):
                kind = "vru"
            elif not agent_id:
                kind = "single"          # nothing else was there: layout
            else:
                kind = geometry or "angle"
            at_fault = (pf.get("COLL_AF", 0.0) > 0 if has_coll_cols else True)
            contacts.append({
                "agent_id": agent_id,
                "at_fault": bool(at_fault),
                "kind": kind,
                "rel_speed": abs(speed - (named or {}).get("speed", 0.0)),
            })

        r.update({
            "frame": i, "t_sim_s": i * dt, "t_log_us": 0,
            "phase": PHASE_SCORED if scored else PHASE_WARMUP,
            "ego_x": float(p[0]), "ego_y": float(p[1]), "ego_z": 0.0,
            "ego_yaw": yaw, "ego_speed": speed,
            "ego_accel": float(kin["accel"][i]), "ego_jerk": float(kin["jerk"][i]),
            "ego_lat_accel": float(kin["lat_accel"][i]), "ego_steer": 0.0,
            "on_drivable": on_drivable,
            "driving_direction_ok": driving_direction_ok,
            "signal_state": signal_state,
            "signal_hold": signal_hold,
            "agents": rows_agents, "contacts": contacts,
            "min_clearance_m": min_clear if min_clear != float("inf") else float("nan"),
            # Deviation from the logged pose, recorded as facts.
            # It does not end the episode or affect the score.
            "ego_dev_lat_m": lat, "ego_dev_lon_m": lon, "ego_dev_yaw_deg": dyaw,
        })
        # Region columns only when they were actually resolved. Absent, not
        # False -- see the region_flags paragraph in this function's docstring.
        if region_flags:
            for col, vals in region_flags.items():
                if i < len(vals):
                    r[col] = bool(vals[i])
        rows.append(r)
    return rows


def scenario_from_arrow(data_root: str | Path, scene_index: int = 0) -> Mapping:
    """ScenarioDescription from a py123d/Arrow scene directory.

    Lets a bundle be scored from what the NavSafe HF dataset actually ships --
    the ``.usdz`` files plus ``arrow/``. Same loader the env uses, so the route and
    agent tracks are the ones the policy drove against; no simulator is built.
    """
    from navsafe.scenario.py123d_adapter import (
        Py123DAdapterConfig,
        scenario_from_py123d_scene,
    )
    from navsafe.scenario.py123d_scenario_description import (
        py123d_to_scenario_description,
    )
    from navsafe.scenario.py123d_scenes import enumerate_scenes

    data_root = Path(data_root)
    scenes = enumerate_scenes(data_root)
    if not scenes:
        raise FileNotFoundError(f"no py123d scenes under {data_root}")
    if scene_index >= len(scenes):
        raise IndexError(
            f"scene index {scene_index} out of range: {len(scenes)} scene(s) "
            f"under {data_root}")
    scenario = scenario_from_py123d_scene(
        scenes[scene_index],
        Py123DAdapterConfig(
            load_state_payloads=True,
            load_custom_payloads=False,
            # Map objects are what make drivable-area compliance measurable;
            # without them the scorer reports those terms null rather than 0.
            load_map_objects=True,
            load_sensor_payloads=False,
            data_root=str(data_root),
            require_map=False,
        ),
    )
    sd = py123d_to_scenario_description(scenario)
    # Scoring reads the override from the bundle rather than from the
    # environment: a re-score run days later, by someone who never saw the
    # driver's env, must reconstruct the same lights the policy drove against.
    from navsafe.benchmark import signal_override as _sig

    _sig.apply(sd, _sig.from_manifest(data_root))
    return sd


def load_from_arrow(eval_dir: str | Path, data_root: str | Path, *,
                    scene_index: int = 0, warmup_frames: int = 0,
                    dt: float = 0.1) -> EvalRun:
    """Read an evaluator output directory, scenario taken from Arrow."""
    return load_with_scenario(
        eval_dir, scenario_from_arrow(data_root, scene_index),
        warmup_frames=warmup_frames, dt=dt)


def load_with_scenario(eval_dir: str | Path, sd: Mapping, *,
                       warmup_frames: int = 0, dt: float = 0.1) -> EvalRun:
    """Same, for a ScenarioDescription already in memory.

    A py123d/Arrow run has no scenario pickle to point at, and the evaluator
    holds the description anyway, so scoring should not have to round-trip
    through the filesystem to reach it.
    """
    eval_dir = Path(eval_dir)
    ego_xyz = np.load(eval_dir / "vehicle_states.npy")
    metrics: dict[str, Any] = {}
    mp = eval_dir / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())

    ego_id = (sd.get("metadata", {}) or {}).get("sdc_id") or sd.get("sdc_id")
    route_xy = np.asarray(sd["tracks"][ego_id]["state"]["position"],
                          dtype=np.float64)[:, :2]
    drivable_known = bool(sd.get("map_features"))

    notes: list[str] = []
    if not drivable_known:
        notes.append(
            "scenario carries no map_features: drivable-area compliance is "
            "unknown, so off-drivable termination and the outside_route_lanes "
            "penalty cannot be evaluated (they are reported as null, not 0)")
    per_frame = read_per_frame_epdms(eval_dir)
    if not per_frame:
        notes.append(
            "no driving_score_summary.csv: per-frame collision and drivability "
            "flags are unavailable; contacts fall back to the run's total")

    lane_center = lane_centerlines(sd)
    # Junction regions, for the event types scored on a region rather than a goal.
    # Derived from the bundle's own map + logged ego, so this needs no seed and
    # no nuPlan db; unresolvable scenarios return None and the columns stay
    # absent. Cheap to skip: no map means no regions, which is most of the
    # reconstruction pilots.
    regions = regions_from_map.resolve_from_scenario(sd) if drivable_known else None
    rflags = regions_from_map.region_flags(ego_xyz[:, :2], regions)
    if regions is None:
        notes.append("junction regions unresolved: region predicates "
                     "(in_intersection / in_exit_lane) report not-checked")

    frames = build_frames(ego_xyz[:, :2], route_xy, _agent_tracks(sd, ego_id, dt=dt),
                          per_frame, warmup_frames=warmup_frames, dt=dt,
                          drivable_known=drivable_known,
                          lane_center=lane_center,
                          stopline_s=stopline_arc(sd, route_xy, lane_center),
                          region_flags=rflags)

    n_coll = int(metrics.get("collision_count", metrics.get("collisions", 0)) or 0)
    if n_coll and not any(f["contacts"] for f in frames):
        notes.append(
            f"metrics.json reports {n_coll} collision(s) that no per-frame NC "
            "row places: counted as an infraction, not as a termination")
    return EvalRun(frames=frames, route_xy=route_xy,
                   ego_xy=np.asarray(ego_xyz, dtype=np.float64)[:, :2], dt=dt,
                   warmup_frames=warmup_frames, drivable_known=drivable_known,
                   collision_count=n_coll, metrics=metrics, notes=notes)
