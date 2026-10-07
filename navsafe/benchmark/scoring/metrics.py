"""NavSafe's four headline metrics, recomputed from a NavSafe trace.

Driving Score, Success Rate, Efficiency and Comfort exactly as Bench2Drive
computes them, so a NavSafe number can sit in a table next to a published
Bench2Drive one.  Every formula cites the line of the Bench2Drive codebase it
reproduces (github.com/Thinklab-SJTU/Bench2Drive @ main, 2026-08-01):

* Driving Score  -- ``leaderboard/leaderboard/utils/statistics_manager.py``
  L352-363 (percentage penalties), L388-394 (multiplicative coefficients),
  L413 (``score_composed = score_route * score_penalty``);
  benchmark level: ``tools/merge_route_json.py`` L38 (``sum / 220``).
* Success Rate   -- ``tools/merge_route_json.py`` L20-27: status Completed or
  Perfect, and every infraction list empty except ``min_speed_infractions``.
* Efficiency     -- ``srunner/.../atomic_criteria.py`` ``MinimumSpeedRouteTest``
  L1957-2086: per-checkpoint mean ego speed over mean background speed, 20
  checkpoints per route; aggregation ``tools/efficiency_smoothness_benchmark.py``
  L263-274 (values > 1000 % dropped, route mean, then mean over routes).
* Comfort        -- ``tools/efficiency_smoothness_benchmark.py`` L40-166:
  six Savitzky-Golay-smoothed kinematic channels checked against nuPlan
  bounds per 20-frame segment; route score is the fraction of segments with
  every channel in bounds.

Two defects in the reference comfort code are reproduced only on request
(``b2d_compat=True``), because published Bench2Drive comfort numbers were
produced with them:

1. **Yaw acceleration is not a derivative.**  The reference smooths the yaw
   rate twice (no ``deriv=1`` on the second call), so its "yaw acceleration"
   channel is the yaw-rate channel again.
2. **Degrees against radian bounds.**  The reference feeds CARLA's
   ``get_angular_velocity()`` -- deg/s -- to bounds specified in rad/s.

The default (``b2d_compat=False``) fixes both: yaw acceleration is the
Savitzky-Golay first derivative of the yaw rate, and all angular inputs are
taken to be rad/s.  The trace records yaw in radians, so NavSafe traces are
correct by construction; the compat mode exists for validating this port
against the reference implementation, not for scoring.

Everything here is a pure function of stored data -- the NavSafe architectural
rule.  No simulator state, no clocks, no I/O.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------
# Driving Score
# --------------------------------------------------------------------------

# statistics_manager.py L21-30 -- identical to CARLA Leaderboard 2.0.
PENALTY_COEFFICIENTS: Mapping[str, float] = {
    "collisions_pedestrian": 0.50,
    "collisions_vehicle": 0.60,
    "collisions_layout": 0.65,
    "red_light": 0.70,
    "scenario_timeouts": 0.70,
    "yield_emergency_vehicle_infractions": 0.70,
    "stop_infraction": 0.80,
}

# statistics_manager.py L31-38.  ``outside_route_lanes`` carries a percentage
# and applies score_penalty *= 1 - (1 - value) * pct/100 with value = 0, i.e.
# the full proportional penalty (1 - pct/100).  The in-code comment claims the
# event is "ignored"; the formula at L358 says otherwise.  min_speed is
# genuinely disabled ('unused', L37).
OUTSIDE_LANES_KEY = "outside_route_lanes"
MIN_SPEED_KEY = "min_speed_infractions"

# Infraction keys that end a route instead of multiplying the penalty; they
# affect success through the status field. `wrong_way` is NavSafe's own:
# Bench2Drive defines no coefficient for it, so it ends the episode and fails
# success() without entering the score product.
NON_PENALTY_KEYS = ("route_dev", "vehicle_blocked", "route_timeout", "wrong_way")


@dataclass
class RouteResult:
    """One route's outcome, mirroring a Bench2Drive RouteRecord.

    ``infractions`` maps infraction name -> occurrence count (or, for
    ``outside_route_lanes``, the single percentage value 0-100).
    ``completed`` is Bench2Drive's status in {'Completed','Perfect'}:
    route completion reached 100 % (RouteCompletionTest: > 99 % of the route
    traversed and within 10 m of the target).
    """

    route_id: str
    completion_pct: float                       # score_route, 0-100
    completed: bool                             # status Completed/Perfect
    infractions: dict[str, float] = field(default_factory=dict)

    def penalty(self) -> float:
        """score_penalty: statistics_manager.py L388-394 + L352-363."""
        p = 1.0
        for key, coeff in PENALTY_COEFFICIENTS.items():
            n = int(self.infractions.get(key, 0))
            p *= coeff ** n
        pct_outside = float(self.infractions.get(OUTSIDE_LANES_KEY, 0.0))
        p *= 1.0 - pct_outside / 100.0
        return p

    def driving_score(self) -> float:
        """score_composed: statistics_manager.py L413.  Range 0-100."""
        return max(self.completion_pct * self.penalty(), 0.0)

    def success(self) -> bool:
        """merge_route_json.py L20-27.

        Success iff the route completed AND no infraction of any kind except
        ``min_speed_infractions`` occurred.  ``route_timeout`` is not exempt:
        a route that finished but blew the time budget fails.
        """
        if not self.completed:
            return False
        for key, value in self.infractions.items():
            if key == MIN_SPEED_KEY:
                continue
            if key == OUTSIDE_LANES_KEY:
                if value > 0:
                    return False
                continue
            if value:
                return False
        return True


def driving_score(routes: Sequence[RouteResult], *,
                  denominator: int | None = None) -> float:
    """Benchmark Driving Score: merge_route_json.py L38.

    Bench2Drive hard-codes the divisor at 220 and thereby counts missing
    episodes as 0.  NavSafe must declare its own denominator; pass it
    explicitly, or omit it to divide by ``len(routes)`` (only correct when
    every episode ran).
    """
    n = denominator if denominator is not None else len(routes)
    if n <= 0:
        raise ValueError("denominator must be positive")
    if len(routes) > n:
        raise ValueError(f"{len(routes)} routes but denominator {n}")
    return sum(r.driving_score() for r in routes) / n


def success_rate(routes: Sequence[RouteResult], *,
                 denominator: int | None = None) -> float:
    """Benchmark Success Rate: merge_route_json.py L39, same denominator rule."""
    n = denominator if denominator is not None else len(routes)
    if n <= 0:
        raise ValueError("denominator must be positive")
    if len(routes) > n:
        raise ValueError(f"{len(routes)} routes but denominator {n}")
    return sum(1 for r in routes if r.success()) / n


# --------------------------------------------------------------------------
# Efficiency
# --------------------------------------------------------------------------

EFFICIENCY_OUTLIER_PCT = 1000.0   # efficiency_smoothness_benchmark.py L269
EFFICIENCY_CHECKPOINTS = 20       # route_scenario.py L427
# Efficiency is the ego's speed relative to background traffic, so it is
# defined only where the background is moving. In a replayed log the whole
# street can be queued, and the ratio then diverges. Below this background
# median speed the route reports no efficiency and is left out of the mean.
# The value separates queued from flowing scenarios in the benchmark with a
# clear gap.
EFFICIENCY_MIN_BACKGROUND_MS = 1.5


def route_efficiency(ego_speed: np.ndarray,
                     background_mean_speed: np.ndarray,
                     route_pct: np.ndarray,
                     *,
                     checkpoints: int = EFFICIENCY_CHECKPOINTS,
                     min_background_ms: float = EFFICIENCY_MIN_BACKGROUND_MS,
                     ) -> float | None:
    """One route's Efficiency, per MinimumSpeedRouteTest (atomic_criteria.py
    L1957-2086) + the aggregation in efficiency_smoothness_benchmark.py
    L263-274.

    Inputs are per-frame arrays over the scored phase:

    * ``ego_speed``  -- ego speed, m/s.
    * ``background_mean_speed`` -- mean speed of surrounding background
      traffic this frame, m/s; NaN when no background vehicle exists.  In
      Bench2Drive the background set is every actor with
      ``role_name == 'background'`` -- locality comes from the traffic
      manager's spawn radius, not from the metric.  From a NavSafe trace use
      the mean speed over ``agents`` with ``policy != 'replay'`` (reactive
      traffic), or all non-ego vehicles under log replay.
    * ``route_pct`` -- monotone route completion at each frame, 0-100
      (``route.route_progress(...).per_frame_pct``).

    Checkpoints sit every ``100 / checkpoints`` percent **of the total route**,
    which is what "speed check is performed every 5 % of the total route
    length" means -- not every 5 % of the distance this ego happened to cover.
    Binning by driven distance instead re-normalises each run onto its own
    trajectory, so a policy that left the road after 12 % of the route still
    collected all 20 checks (one every 0.64 % of route) and no run could ever
    fail to reach the first checkpoint.

    A route whose ego never passes the FIRST checkpoint is excluded entirely
    (returns None), per "if the ego vehicle fails to pass the initial 5 %
    checkpoint, this route is not included in the final driving efficiency
    metric calculation".  A route whose background never drives -- median
    background speed below ``min_background_ms`` -- is excluded the same way:
    the ratio has no denominator to speak of, and reporting several hundred
    percent there states a result the data does not contain.  Only checkpoints the ego actually reached contribute;
    a checkpoint with no background traffic is skipped (the reference records
    100 there only via its always-fired terminate path, and those all-default
    checkpoints are what the outlier drop and the 'no infraction -> route
    skipped' rule prune at aggregation).  Values above 1000 % are dropped
    (L269) -- the reference's guard against speed spikes such as an ego
    falling through the map.  Returns the mean over surviving checkpoints, or
    None when none survive; the reference likewise excludes such routes from
    the benchmark mean rather than scoring them 100.
    """
    ego = np.asarray(ego_speed, dtype=float)
    bg = np.asarray(background_mean_speed, dtype=float)
    pct = np.asarray(route_pct, dtype=float)
    if not (len(ego) == len(bg) == len(pct)):
        raise ValueError("per-frame arrays must have equal length")
    if len(ego) == 0:
        return None

    span = 100.0 / checkpoints          # 5 % of the route, at 20 checkpoints
    # The initial-checkpoint gate: an ego that never got 5 % along the route
    # contributes no route to the benchmark mean at all.
    if float(np.nanmax(pct)) < span:
        return None
    # The moving-background gate: see EFFICIENCY_MIN_BACKGROUND_MS. Median, not
    # mean, so one vehicle driving past a stopped street cannot carry the whole
    # scored window over the floor.
    finite_bg = bg[np.isfinite(bg)]
    if finite_bg.size and float(np.median(finite_bg)) < min_background_ms:
        return None
    seg_of_frame = np.floor(pct / span).astype(int).clip(0, checkpoints - 1)
    # Only checkpoints the ego reached exist; the rest were never driven.
    reached = int(np.floor(float(np.nanmax(pct)) / span))

    values: list[float] = []
    for seg in range(min(reached, checkpoints)):
        mask = (seg_of_frame == seg) & np.isfinite(bg) & (bg > 0)
        if not mask.any():
            continue
        pct = 100.0 * float(ego[mask].mean()) / float(bg[mask].mean())
        if pct > EFFICIENCY_OUTLIER_PCT:
            continue
        values.append(pct)
    if not values:
        return None
    return float(np.mean(values))


def efficiency(route_values: Sequence[float | None]) -> float | None:
    """Benchmark Efficiency: mean over routes that produced a value
    (efficiency_smoothness_benchmark.py L285)."""
    vals = [v for v in route_values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


# --------------------------------------------------------------------------
# Comfort
# --------------------------------------------------------------------------

# efficiency_smoothness_benchmark.py L9-26 -- nuPlan bounds.
COMFORT_BOUNDS = {
    "lon_accel": (-4.05, 2.40),      # m/s^2
    "lat_accel": (-4.89, 4.89),      # m/s^2
    "mag_jerk": (-8.37, 8.37),       # m/s^3
    "lon_jerk": (-4.13, 4.13),       # m/s^3
    "yaw_accel": (-1.93, 1.93),      # rad/s^2
    "yaw_rate": (-0.95, 0.95),       # rad/s
}
COMFORT_WINDOW = 7                   # savgol window_length
COMFORT_POLY = 2                     # savgol polyorder
COMFORT_DT = 0.1                     # s
COMFORT_SEGMENT_FRAMES = 20          # seg_compute_comfort_metric per_step


def _smooth(x: np.ndarray, *, deriv: int = 0) -> np.ndarray:
    window = min(COMFORT_WINDOW, len(x))
    if window <= COMFORT_POLY:
        # Too short to filter; fall back to the raw signal / finite difference.
        if deriv == 0:
            return x
        return np.gradient(x, COMFORT_DT)
    return savgol_filter(x, window_length=window, polyorder=COMFORT_POLY,
                         deriv=deriv, delta=COMFORT_DT)


def _within(x: np.ndarray, lo: float, hi: float) -> bool:
    # Strict inequalities, matching _within_bound (L211).
    return bool(np.all((x > lo) & (x < hi)))


def segment_comfort(lon_accel: np.ndarray,
                    lat_accel: np.ndarray,
                    mag_accel: np.ndarray,
                    yaw_rate: np.ndarray,
                    *,
                    b2d_compat: bool = False) -> bool:
    """All six channels in bounds over one segment (compute_comfort_metric,
    L65-166).  Inputs are raw per-frame signals; smoothing happens here.

    ``yaw_rate`` must be rad/s.  With ``b2d_compat=True`` the two reference
    defects are reproduced: yaw acceleration is the smoothed yaw rate (no
    derivative), and the caller is expected to pass deg/s as the reference
    does -- the function itself never converts.
    """
    lon = _smooth(np.asarray(lon_accel, dtype=float))
    lat = _smooth(np.asarray(lat_accel, dtype=float))
    mag = _smooth(np.asarray(mag_accel, dtype=float))
    yr = _smooth(np.asarray(yaw_rate, dtype=float))
    mag_jerk = _smooth(np.asarray(mag_accel, dtype=float), deriv=1)
    lon_jerk = _smooth(np.asarray(lon_accel, dtype=float), deriv=1)
    if b2d_compat:
        ya = yr                      # defect 1: no derivative
    else:
        ya = _smooth(np.asarray(yaw_rate, dtype=float), deriv=1)

    b = COMFORT_BOUNDS
    return (_within(lon, *b["lon_accel"])
            and _within(lat, *b["lat_accel"])
            and _within(mag_jerk, *b["mag_jerk"])
            and _within(lon_jerk, *b["lon_jerk"])
            and _within(ya, *b["yaw_accel"])
            and _within(yr, *b["yaw_rate"]))


def route_comfort(lon_accel: np.ndarray,
                  lat_accel: np.ndarray,
                  mag_accel: np.ndarray,
                  yaw_rate: np.ndarray,
                  *,
                  b2d_compat: bool = False,
                  segment_frames: int = COMFORT_SEGMENT_FRAMES) -> float:
    """One route's comfort score: the fraction of ``segment_frames``-long
    segments with every channel in bounds (seg_compute_comfort_metric,
    L40-62).  Segments shorter than ``segment_frames`` at the tail are
    dropped, as in the reference; an episode shorter than one segment is
    scored as its single pass/fail."""
    arrays = [np.asarray(a, dtype=float)
              for a in (lon_accel, lat_accel, mag_accel, yaw_rate)]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("per-frame arrays must have equal length")
    if n == 0:
        raise ValueError("empty episode")

    if n <= segment_frames:
        return 1.0 if segment_comfort(*arrays, b2d_compat=b2d_compat) else 0.0

    verdicts: list[bool] = []
    for start in range(0, n, segment_frames):
        seg = [a[start:start + segment_frames] for a in arrays]
        if len(seg[0]) < segment_frames:
            continue
        verdicts.append(segment_comfort(*seg, b2d_compat=b2d_compat))
    return verdicts.count(True) / len(verdicts)


def comfort(route_values: Sequence[float]) -> float:
    """Benchmark comfort ('Driving Smoothness'): mean over routes (L286)."""
    if not route_values:
        raise ValueError("no routes")
    return float(np.mean(route_values))


# --------------------------------------------------------------------------
# Trace adapters
# --------------------------------------------------------------------------

def _scored(frames: Sequence[Mapping]) -> list[Mapping]:
    return [f for f in frames if f.get("phase") == "scored"]


def comfort_inputs_from_trace(frames: Sequence[Mapping],
                              *, dt: float = COMFORT_DT) -> dict[str, np.ndarray]:
    """Per-frame comfort signals from NavSafe trace rows (scored phase only).

    The trace stores ``ego_accel`` (longitudinal), ``ego_lat_accel`` and
    ``ego_yaw`` (radians); yaw rate is the wrapped finite difference of yaw.
    """
    rows = _scored(frames)
    if not rows:
        raise ValueError("trace has no scored frames")
    lon = np.array([f["ego_accel"] for f in rows], dtype=float)
    lat = np.array([f["ego_lat_accel"] for f in rows], dtype=float)
    mag = np.hypot(lon, lat)
    yaw = np.unwrap(np.array([f["ego_yaw"] for f in rows], dtype=float))
    yaw_rate = np.gradient(yaw, dt)
    return {"lon_accel": lon, "lat_accel": lat,
            "mag_accel": mag, "yaw_rate": yaw_rate}


# Agent ``policy`` values that are not background traffic.  Bench2Drive's
# background set is what the traffic manager spawned, so a car parked for the
# whole log is not in it -- including parked cars drags the background mean
# towards zero and inflates every policy's Efficiency.
NON_BACKGROUND_POLICIES = ("static", "parked")


def efficiency_inputs_from_trace(frames: Sequence[Mapping]) -> dict[str, np.ndarray]:
    """Per-frame efficiency inputs from NavSafe trace rows (scored phase only).

    Background set: non-replay vehicle agents when any exist (the reactive
    traffic Bench2Drive's ``role_name == 'background'`` filter selects),
    otherwise every vehicle agent (log replay).  Agents whose policy marks them
    as parked are excluded from both pools.
    """
    rows = _scored(frames)
    if not rows:
        raise ValueError("trace has no scored frames")
    ego = np.array([f["ego_speed"] for f in rows], dtype=float)

    def frame_bg(f: Mapping) -> float:
        vehicles = [a for a in f.get("agents", [])
                    if a.get("cls") == "vehicle"
                    and a.get("policy") not in NON_BACKGROUND_POLICIES]
        reactive = [a for a in vehicles
                    if a.get("policy") not in ("", "replay")]
        pool = reactive or vehicles
        if not pool:
            return float("nan")
        return float(np.mean([a["speed"] for a in pool]))

    bg = np.array([frame_bg(f) for f in rows], dtype=float)
    xy = np.array([[f["ego_x"], f["ego_y"]] for f in rows], dtype=float)
    steps = np.hypot(*np.diff(xy, axis=0).T)
    accum = np.concatenate([[0.0], np.cumsum(steps)])
    # accum_dist is the ego's own odometer -- kept for callers that report it,
    # but NOT the checkpoint axis: route_efficiency bins by route position, and
    # odometer is not route progress (driving in a circle advances one and not
    # the other). The caller supplies route_pct from route.route_progress.
    return {"ego_speed": ego, "background_mean_speed": bg, "accum_dist": accum,
            "ego_xy": xy}


def route_result_from_termination(route_id: str,
                                  completion_pct: float,
                                  info: Mapping,
                                  *,
                                  truncated: bool = False,
                                  extra_infractions: Mapping[str, float] | None = None,
                                  ) -> RouteResult:
    """Compatibility bridge: build a RouteResult from a MetaDrive-style
    termination ``info`` dict (``navsafe.scenario.constants.TerminationState``
    keys), for episodes run through the legacy MetaDrive-shaped envs.  New
    NavSafe traces should use ``navsafe.benchmark.termination`` instead — the
    benchmark's own taxonomy (fault-split contacts, wrong-way
    exits, deadlock vs budget expiry), classified from the trace.

    State mapping, against the Bench2Drive terminators it stands in for:

    ==================  =========================================  ===========
    TerminationState    Bench2Drive terminator                     RouteResult
    ==================  =========================================  ===========
    SUCCESS             route completion 100 % (> 99 % + < 10 m)   completed
    IDLE                agent blocked (< 0.1 m/s for 60 s)         vehicle_blocked
    OUT_OF_ROAD         route deviation (> 30 m / > 30 % unsafe)   route_dev
    MAX_STEP/truncated  route timeout (300 s + Σd/(0.1·v_lim))     route_timeout
    CRASH_HUMAN         --                                         collisions_pedestrian
    CRASH_VEHICLE       --                                         collisions_vehicle
    CRASH_OBJECT/                                                  collisions_layout
      BUILDING/SIDEWALK
    ==================  =========================================  ===========

    Two declared semantic gaps, to configure in the env rather than paper
    over here: MetaDrive's IDLE threshold must be set to Bench2Drive's
    (speed < 0.1 m/s sustained 60 s), and MAX_STEP is a flat step cap, so a
    faithful route timeout needs max_step derived per route from
    ``300 s + Σ Δd / (0.1 · v_limit)`` (timer.py ``RouteTimeoutBehavior``)
    rather than one global constant.  Note ``route_timeout`` is an SR-failing
    infraction but NOT a DS penalty in Bench2Drive -- the legacy
    ``statistics_manager_md`` multiplies 0.7 for it; this module deliberately
    does not.
    """
    from navsafe.scenario.constants import TerminationState as TS

    infractions: dict[str, float] = dict(extra_infractions or {})

    def bump(key: str) -> None:
        infractions[key] = infractions.get(key, 0) + 1

    if info.get(TS.CRASH_HUMAN):
        bump("collisions_pedestrian")
    if info.get(TS.CRASH_VEHICLE):
        bump("collisions_vehicle")
    for k in (TS.CRASH_OBJECT, TS.CRASH_BUILDING, TS.CRASH_SIDEWALK):
        if info.get(k):
            bump("collisions_layout")

    completed = bool(info.get(TS.SUCCESS))
    if not completed:
        if info.get(TS.IDLE):
            bump("vehicle_blocked")
        elif info.get(TS.OUT_OF_ROAD):
            bump("route_dev")
        elif truncated or info.get(TS.MAX_STEP):
            bump("route_timeout")

    return RouteResult(route_id=route_id,
                       completion_pct=float(completion_pct),
                       completed=completed,
                       infractions=infractions)


#: Contact kind -> penalty channel. `vru` is the class of what was hit and
#: the others are impact geometry; `from_eval` resolves the class first.
#: `physical_impulse` means no other agent was involved, which is a layout
#: collision.
_CONTACT_KIND_TO_INFRACTION = {
    "vru": "collisions_pedestrian",
    "rear_end": "collisions_vehicle",
    "angle": "collisions_vehicle",
    "sideswipe": "collisions_vehicle",
    "front": "collisions_vehicle",
    "single": "collisions_layout",
    "physical_impulse": "collisions_layout",
}


def infractions_from_trace(frames: Sequence[Mapping],
                           *, at_fault_only: bool = True) -> dict[str, float]:
    """Contact-derived infraction counts from a trace.

    Covers the collision family only -- red light, stop sign, blocked and
    timeout verdicts come from the rubric layer, which owns those thresholds;
    merge its output into the returned dict before building a RouteResult.
    ``at_fault_only`` keeps the NavSafe fault attribution (a replay follower
    rear-ending a correct ego is not the policy's infraction); Bench2Drive
    itself has no fault model and counts every contact.
    """
    counts: dict[str, float] = {}
    seen: set[tuple[str, str]] = set()
    for f in _scored(frames):
        for c in f.get("contacts", []):
            if at_fault_only and not c.get("at_fault", False):
                continue
            key = (c.get("agent_id", ""), c.get("kind", ""))
            if key in seen:      # one infraction per contact pair, not per frame
                continue
            seen.add(key)
            name = _CONTACT_KIND_TO_INFRACTION.get(c.get("kind", ""),
                                                   "collisions_vehicle")
            counts[name] = counts.get(name, 0) + 1
    return counts
