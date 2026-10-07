"""Turn a running episode into the canonical trace the rubric scores.

This is the join between the simulator and the benchmark.  Everything the
scoring chain needs is assembled here, once per frame, and nothing downstream
touches the simulator again -- which is what makes re-scoring a stored trace
possible without re-rendering anything.

Three responsibilities, in order of subtlety:

1. **Map context.**  The rubric's regions were frozen at seed build time
   (``seeds/resolve_regions.py``); here they are only *tested against*.  A
   writer that re-derived "which lane is the exit lane" could disagree with the
   rubric it is feeding.
2. **Fault.**  Recorded per contact, because the fault rule is regime-dependent
   (doc Table VI) and re-deciding it later must remain possible.  Under a
   non-reactive regime an agent that closes on a correctly-behaving ego owns
   the contact, so ``ego_is_closing`` is computed from the relative velocity
   and stored per agent rather than inferred later from geometry.
3. **Deviation from the logged pose is a measurement, not a gate.**  Closed-loop
   control moves the camera off the recorded trajectory, and the reconstruction
   is only trained near it.  ``ego_dev_lat_m`` / ``ego_dev_lon_m`` /
   ``ego_dev_yaw_deg`` record how far, every frame.  They used to feed a
   render-validity envelope that ENDED the episode (``envelope_exit``, excluded
   from every denominator); that bound was never certified, fired on ordinary
   closed-loop driving, and was removed. Certifying one from
   held-out rendering metrics at controlled offsets remains possible precisely
   because these columns are still here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M
from navsafe.benchmark.trace.schema import (
    FRAME_SCHEMA, PHASE_SCORED, PHASE_WARMUP, EpisodeMeta, empty_frame,
)

# An agent is "closing" when the range rate is negative by more than noise.
CLOSING_EPS_MPS = 0.15
# Above this, a reported velocity is not a road-vehicle velocity and is ignored
# in favour of the finite difference. 60 m/s is ~216 km/h.
MAX_PLAUSIBLE_SPEED_MPS = 60.0
# Beyond this there is no interaction worth recording; keeps traces small.
AGENT_RADIUS_M = 60.0


def _rect(cx: float, cy: float, yaw: float, length: float, width: float):
    """The oriented footprint of a body, as a shapely polygon."""
    from shapely.geometry import Polygon
    hl, hw = 0.5 * length, 0.5 * width
    c, s = math.cos(yaw), math.sin(yaw)
    return Polygon([(cx + c * dx - s * dy, cy + s * dx + c * dy)
                    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def _box_distance(x1, y1, yaw1, l1, w1, x2, y2, yaw2, l2, w2) -> float:
    """Distance between two oriented rectangles; 0.0 when they overlap.

    Exact rather than a support-function subtraction along the line of sight:
    that approximation over-counts at an angle and reported 0.00 m clearance
    between vehicles 3 m apart, which failed a hard gate on the logged human.
    """
    return float(_rect(x1, y1, yaw1, l1, w1).distance(
        _rect(x2, y2, yaw2, l2, w2)))


def _poly(points) -> Any:
    from shapely.geometry import Polygon
    return Polygon(points) if points and len(points) >= 3 else None


#: Feature-id namespace for lanes NavSafe synthesised rather than read from
#: the source map. ``_add_logged_drivable_support``
#: (scenario/py123d_scenario_description.py) adds
#: ``__logged_drivable_support_<i>`` polygons, typed as lanes, wherever the
#: source map has a hole the logged ego drives through. They exist to make
#: drivable-area compliance measurable, and they must stay in the drivable set
#: -- but they are NOT lanes: they follow the logged path, carry no nuPlan id
#: and no ``exit_lanes``.
#:
#: Excluded here for the same reason ``policy/sensor/utils/geospatial_priors``
#: excludes them from ``closest_lane`` (see its ``SYNTHETIC_ID_PREFIX``, where
#: including them cost 0 of 20 retrieval bins on all 41 replans). Measured on
#: ``4528c271d89c53e1``: 4 support polygons against 4583 real lanes, and the
#: support lane was the NEAREST for 155 of 190 frames, so ``lane_id`` tracked
#: the logged path instead of the lane graph and ``lateral_offset_m`` never
#: left +/-0.26 m. That is not merely noisy -- it makes a lane change
#: undetectable, so C-3's gate would have withheld the goal from every policy
#: including one that performed the manoeuvre.
SYNTHETIC_LANE_PREFIX = "__"


def lane_centerlines(scenario_data: Mapping) -> dict[str, "np.ndarray"]:
    """``{lane_id: (N, 2) centreline}`` for the map's REAL lanes.

    Module-level because two paths need the SAME lane geometry and sign
    convention: this writer (live) and ``trace/from_eval.py`` (post-hoc, over a
    finished eval directory). They diverged silently once -- the eval path
    carried no lane columns at all, which turned the C-3 lane-change gate into
    a no-op on every real run while its unit tests passed on synthetic frames.

    Synthesised support lanes (:data:`SYNTHETIC_LANE_PREFIX`) are excluded.
    """
    return {
        str(lid): np.asarray(f["polyline"], dtype=np.float64)[:, :2]
        for lid, f in (scenario_data.get("map_features") or {}).items()
        if str(f.get("type", "")).startswith("LANE")
        and not str(lid).startswith(SYNTHETIC_LANE_PREFIX)
        and f.get("polyline") is not None
    }


def nearest_lane(x: float, y: float, lane_center: Mapping[str, "np.ndarray"],
                 candidates: Sequence[str] | None = None,
                 *, early_exit_m: float | None = None):
    """``(lane_id, lane_heading, signed lateral offset)`` for the nearest lane.

    The offset is signed with left of the lane direction positive, which is what
    makes a lateral crossing detectable: leaving lane A towards one edge and
    entering lane B from the opposite edge flips the sign, while driving into a
    successor lane happens near both centrelines and does not.

    ``early_exit_m`` stops the scan at the first lane closer than that, which is
    only sound when the caller has no candidate set and wants speed over the
    exact argmin; pass None for the true nearest.
    """
    best = ("", float("nan"), float("nan"), float("inf"))
    for lid in (candidates or lane_center.keys()):
        c = lane_center.get(str(lid))
        if c is None or len(c) < 2:
            continue
        d = np.linalg.norm(c - np.array([x, y]), axis=1)
        i = int(np.argmin(d))
        if d[i] >= best[3]:
            continue
        j = min(i + 1, len(c) - 1)
        k = max(i - 1, 0)
        tang = c[j] - c[k]
        head = math.atan2(tang[1], tang[0])
        rel = np.array([x, y]) - c[i]
        lat = float(-math.sin(head) * rel[0] + math.cos(head) * rel[1])
        best = (str(lid), head, lat, float(d[i]))
        if early_exit_m is not None and d[i] < early_exit_m:
            break
    return best[0], best[1], best[2]


class TraceWriter:
    """Accumulates per-frame rows; writes parquet + meta json on close."""

    def __init__(self, *, scenario_data: dict, regions: dict, meta: EpisodeMeta,
                 warmup_frames: int, sim_dt: float = 0.1):
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree

        self.meta = meta
        self.warmup = int(warmup_frames)
        self.dt = float(sim_dt)
        self.rows: list[dict] = []
        # Ego kinematics are derived from the position history rather than from
        # the env's speed reading: see the module docstring on teleport.
        self._xy_hist: list[tuple[float, float]] = []
        self._speed_hist: list[float] = []
        self._accel_hist: list[float] = []
        # Per-agent last position, so agent velocity comes from motion too.
        self._agent_prev: dict[str, tuple[float, float]] = {}
        self._warned_agent_vel = False

        self.regions = regions
        self.ix_poly = _poly(regions.get("polygons", {}).get("intersection"))
        # Falls back to the whole junction only for regions.json written before
        # the corridor existed.
        self.cz_poly = (_poly(regions.get("polygons", {}).get("conflict_zone"))
                        or self.ix_poly)
        self.exit_poly = _poly(regions.get("polygons", {}).get("exit_lane"))
        self.signal_lane = (regions.get("signal") or {}).get("controlled_lane")

        # Logged path, for the deviation columns + arc-length distances.
        path = regions.get("path", {})
        self.path_xy = np.asarray(path.get("xy") or [], dtype=np.float64)
        self.path_s = np.asarray(path.get("s_m") or [], dtype=np.float64)
        self.s_stop = path.get("s_intersection_entry_m")

        # Full lane set: on_drivable is about the road, not about the route.
        self.lane_ids: list[str] = []
        shapes = []
        for lid, feat in scenario_data.get("map_features", {}).items():
            if not str(feat.get("type", "")).startswith("LANE"):
                continue
            p = feat.get("polygon")
            if p is not None and len(p) >= 3:
                self.lane_ids.append(str(lid))
                shapes.append(Polygon(np.asarray(p)[:, :2]))
        self.lane_shapes = shapes
        self.lane_tree = STRtree(shapes) if shapes else None
        self.lane_center = lane_centerlines(scenario_data)
        self.signals = scenario_data.get("dynamic_map_states", {})
        # Logged ego heading, for the yaw deviation column.
        sdc = scenario_data.get("metadata", {}).get("sdc_id")
        st = scenario_data.get("tracks", {}).get(sdc, {}).get("state", {})
        h = st.get("heading")
        self.log_heading = (np.asarray(h, dtype=np.float64).reshape(-1)
                            if h is not None else np.empty(0))

    # --- geometry helpers -------------------------------------------------

    def _lanes_at(self, x: float, y: float) -> list[str]:
        from shapely.geometry import Point
        if self.lane_tree is None:
            return []
        p = Point(x, y)
        return [self.lane_ids[k] for k in self.lane_tree.query(p)
                if self.lane_shapes[k].contains(p)]

    def _lane_frame(self, x: float, y: float, lanes: Sequence[str]):
        """(lane_id, lane_heading, signed lateral offset) for the nearest lane.

        Delegates to the module-level :func:`nearest_lane` so the live trace and
        the post-hoc one (``trace/from_eval.py``) cannot drift apart. The
        early exit is kept for the no-candidate scan, which is what this call
        used to do.
        """
        lane_id, head, lat = nearest_lane(x, y, self.lane_center, lanes,
                                          early_exit_m=None if lanes else 0.5)
        if not lane_id and lanes:
            # The containing polygons were all synthesised support lanes (they
            # are excluded from lane_center, on purpose -- see
            # SYNTHETIC_LANE_PREFIX). The ego is still somewhere, and the
            # nearest REAL lane is the honest answer; leaving the column empty
            # would drop the frame out of every lane-based check for a reason
            # that has nothing to do with the ego.
            lane_id, head, lat = nearest_lane(x, y, self.lane_center,
                                              early_exit_m=0.5)
        return lane_id, head, lat

    def _project(self, x: float, y: float):
        """(arc length, lateral offset, index) of (x, y) on the logged path."""
        if self.path_xy.size == 0:
            return float("nan"), float("nan"), 0
        d = np.linalg.norm(self.path_xy - np.array([x, y]), axis=1)
        i = int(np.argmin(d))
        j = min(i + 1, len(self.path_xy) - 1)
        k = max(i - 1, 0)
        tang = self.path_xy[j] - self.path_xy[k]
        head = math.atan2(tang[1], tang[0])
        rel = np.array([x, y]) - self.path_xy[i]
        lon = float(math.cos(head) * rel[0] + math.sin(head) * rel[1])
        lat = float(-math.sin(head) * rel[0] + math.cos(head) * rel[1])
        return float(self.path_s[i]) + lon, lat, i

    def _signal_state(self, frame: int) -> tuple[str, str]:
        if not self.signal_lane:
            return "", "unknown"
        st = self.signals.get(str(self.signal_lane))
        if not st:
            return str(self.signal_lane), "unknown"
        states = st.get("state", {}).get("object_state") or st.get("state") or []
        try:
            raw = str(states[min(frame, len(states) - 1)])
        except Exception:
            return str(self.signal_lane), "unknown"
        u = raw.upper()
        if "GO" in u or "GREEN" in u:
            return str(self.signal_lane), "green"
        if "STOP" in u or "RED" in u:
            return str(self.signal_lane), "red"
        if "CAUTION" in u or "YELLOW" in u:
            return str(self.signal_lane), "yellow"
        return str(self.signal_lane), "unknown"

    # --- the per-frame entry point ---------------------------------------

    def on_frame(self, *, frame: int, t_sim_s: float, t_log_us: int,
                 ego: dict, agents: Sequence[dict], contacts: Sequence[dict] = (),
                 signal_hold: bool = False) -> None:
        from shapely.geometry import Point

        r = empty_frame()
        # The evaluator's state fact, not re-derived here (see schema.py):
        # absent means "not checked", which keeps the plain deadlock rule.
        r["signal_hold"] = bool(signal_hold)
        x, y = float(ego["x"]), float(ego["y"])
        yaw, speed = float(ego.get("yaw", 0.0)), float(ego.get("speed", 0.0))

        # The handover is a discontinuity: the ego stops replaying the log and
        # starts following the policy, so derivatives must not span it.
        if frame == self.warmup and self.warmup > 0:
            self._xy_hist.clear()
            self._speed_hist.clear()
            self._accel_hist.clear()
        self._xy_hist.append((x, y))
        speed_m, accel, jerk = self._kinematics()

        lanes = self._lanes_at(x, y)
        lane_id, lane_head, lat_off = self._lane_frame(x, y, lanes)
        s_ego, dev_lat, path_i = self._project(x, y)
        pt = Point(x, y)

        sig_id, sig_state = self._signal_state(frame)

        dev_lon = 0.0
        if self.path_s.size:
            dev_lon = float(s_ego - self.path_s[min(path_i, len(self.path_s) - 1)])
        dev_yaw = 0.0
        if self.log_heading.size:
            h = float(self.log_heading[min(path_i, len(self.log_heading) - 1)])
            dev_yaw = math.degrees((yaw - h + math.pi) % (2 * math.pi) - math.pi)

        rows_agents = []
        min_clear = float("inf")
        for a in agents:
            ax, ay = float(a.get("x", 0.0)), float(a.get("y", 0.0))
            d = math.dist((x, y), (ax, ay))
            if d > AGENT_RADIUS_M:
                continue
            avx, avy = self._agent_velocity(a)
            evx, evy = speed_m * math.cos(yaw), speed_m * math.sin(yaw)
            # Range rate along the line of sight: who is closing the gap.
            ux, uy = ((ax - x) / d, (ay - y) / d) if d > 1e-6 else (0.0, 0.0)
            rate = (avx - evx) * ux + (avy - evy) * uy
            ego_closing = bool(-rate > CLOSING_EPS_MPS
                               and ((ax - x) * evx + (ay - y) * evy) > 0)
            # Range-rate TTC: closing speed along the line of sight. This is
            # NOT a collision-course test -- vehicles passing in adjacent
            # opposing lanes close fast at short range without ever
            # conflicting -- so it is sound for lead-vehicle and same-lane
            # geometry and optimistic-to-wrong for crossing/oncoming. A
            # predicate that needs true conflict (rubric `clear` with an
            # oncoming band) must wait for a path-intersection TTC.
            ttc = float(d / -rate) if rate < -CLOSING_EPS_MPS else float("inf")
            alen = float(a.get("length", 4.5))
            awid = float(a.get("width", 1.9))
            clearance = _box_distance(
                x, y, yaw, EGO_LENGTH_M, EGO_WIDTH_M,
                ax, ay, float(a.get("yaw", 0.0)), alen, awid)
            min_clear = min(min_clear, clearance)
            rows_agents.append({
                "id": str(a.get("id", "")), "cls": str(a.get("cls", "")),
                "policy": str(a.get("policy", "")),
                "x": ax, "y": ay, "yaw": float(a.get("yaw", 0.0)),
                "speed": math.hypot(avx, avy),
                "length": alen, "width": awid,
                "dist_to_ego": d, "clearance_m": clearance,
                "is_lead": bool(a.get("is_lead", False)),
                "in_conflict_zone": bool(self.cz_poly.contains(Point(ax, ay)))
                                    if self.cz_poly is not None else False,
                "ttc_s": ttc, "ego_is_closing": ego_closing,
            })

        closing = {a["id"]: a["ego_is_closing"] for a in rows_agents}
        rows_contacts = [{
            "agent_id": str(c.get("agent_id", "")),
            # Fault follows who was closing; the regime decides whether the
            # gate honours it (predicates.no_collision).
            "at_fault": bool(c.get("at_fault", closing.get(str(c.get("agent_id", "")), True))),
            "kind": str(c.get("kind", "unknown")),
            "rel_speed": float(c.get("rel_speed", 0.0)),
        } for c in contacts]

        r.update({
            "frame": int(frame), "t_sim_s": float(t_sim_s), "t_log_us": int(t_log_us),
            "phase": PHASE_WARMUP if frame < self.warmup else PHASE_SCORED,
            "ego_x": x, "ego_y": y, "ego_z": float(ego.get("z", 0.0)),
            # ego_speed is the MEASURED speed (from motion); the env's reported
            # speed is unusable under teleport execution.
            "ego_yaw": yaw, "ego_speed": speed_m, "ego_accel": accel, "ego_jerk": jerk,
            "ego_lat_accel": float(speed_m * speed_m * abs(float(ego.get("yaw_rate", 0.0)))),
            "ego_steer": float(ego.get("steer", 0.0)),
            "on_drivable": bool(lanes),
            "lane_id": lane_id, "lane_heading": lane_head, "lateral_offset_m": lat_off,
            "dist_to_stopline_m": (float(self.s_stop - s_ego)
                                   if self.s_stop is not None else float("nan")),
            "in_intersection": bool(self.ix_poly.contains(pt)) if self.ix_poly is not None else False,
            "in_conflict_zone": bool(self.cz_poly.contains(pt)) if self.cz_poly is not None else False,
            "in_exit_lane": bool(self.exit_poly.contains(pt)) if self.exit_poly is not None else False,
            "signal_id": sig_id, "signal_state": sig_state,
            "agents": rows_agents, "contacts": rows_contacts,
            "min_clearance_m": min_clear if min_clear != float("inf") else float("nan"),
            # The seed writer has lane geometry but not the EPDMS DDC subscore
            # the wrong-way rule reads, so it states "not checked" rather than
            # guessing. from_eval fills it from the evaluator's DDC column.
            "driving_direction_ok": True,
            "ego_dev_lat_m": dev_lat, "ego_dev_lon_m": dev_lon, "ego_dev_yaw_deg": dev_yaw,
        })
        self.rows.append(r)

    def _agent_velocity(self, a: dict) -> tuple[float, float]:
        """(vx, vy) from this agent's motion, not from ``a["velocity"]``.

        Deliberate: under teleport execution bodies advance by placement, so
        only a motion-derived velocity is consistent with the positions this
        trace scores against -- ``clearance_m`` and ``ttc_s`` depend on that.
        Switching to the reported field would break them silently.
        """
        aid = str(a.get("id", ""))
        ax, ay = float(a.get("x", 0.0)), float(a.get("y", 0.0))
        prev = self._agent_prev.get(aid)
        self._agent_prev[aid] = (ax, ay)
        if prev is not None:
            return (ax - prev[0]) / self.dt, (ay - prev[1]) / self.dt
        rvx, rvy = float(a.get("vx", 0.0)), float(a.get("vy", 0.0))
        if math.hypot(rvx, rvy) > MAX_PLAUSIBLE_SPEED_MPS:
            if not self._warned_agent_vel:
                self._warned_agent_vel = True
                print(f"[trace] implausible agent velocity "
                      f"{math.hypot(rvx, rvy):.3g} m/s for {aid!r}; using "
                      f"motion-derived velocity instead",
                      flush=True)
            return 0.0, 0.0
        return rvx, rvy

    def _kinematics(self) -> tuple[float, float, float]:
        """(speed, accel, jerk) from the position history, central differences.

        Central rather than backward differences because a replan spike is
        one-sided: a backward difference attributes the whole jump to the frame
        after it, a central one spreads it over the neighbours and keeps the
        signal usable for the comfort envelope.
        """
        h = self._xy_hist
        if len(h) < 2:
            self._speed_hist.append(0.0)
            self._accel_hist.append(0.0)
            return 0.0, 0.0, 0.0
        (x0, y0), (x1, y1) = h[-2], h[-1]
        speed = math.dist((x0, y0), (x1, y1)) / self.dt
        self._speed_hist.append(speed)
        s = self._speed_hist
        accel = ((s[-1] - s[-3]) / (2 * self.dt)) if len(s) >= 3 else 0.0
        self._accel_hist.append(accel)
        a = self._accel_hist
        jerk = ((a[-1] - a[-3]) / (2 * self.dt)) if len(a) >= 3 else 0.0
        return speed, accel, jerk

    # --- output ------------------------------------------------------------

    def close(self, out_dir: str | Path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # `in_exit_lane` is a region flag the rubric resolves by name; it is not
        # part of the fixed schema, so carry it as an extra column.
        schema = FRAME_SCHEMA.append(pa.field("in_exit_lane", pa.bool_()))
        table = pa.Table.from_pylist(self.rows, schema=schema)
        path = out / "trace.parquet"
        pq.write_table(table, path, compression="zstd")

        n_scored = sum(1 for r in self.rows if r["phase"] == PHASE_SCORED)
        meta = self.meta.to_dict()
        meta["execution_mode"] = os.environ.get("NAVSAFE_EXECUTION_MODE", "")
        # `n_render_invalid` is kept at 0 rather than dropped: consumers written
        # against the envelope (runner/collect, show_verdict, rubric/evaluator)
        # read this key, and 0 is now the truth — no frame is excluded for
        # deviation any more.
        meta["coverage"] = {"n_frames": len(self.rows), "n_scored": n_scored,
                            "n_render_invalid": 0}
        (out / "trace_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[trace] {path}  frames={len(self.rows)} scored={n_scored}",
              flush=True)
        return path


def enabled() -> bool:
    """Opt-in, so an eval that does not want a trace is unaffected."""
    return os.environ.get("NAVSAFE_TRACE", "") not in ("", "0", "false")
