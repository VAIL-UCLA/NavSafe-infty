"""Event windowing: nuPlan scenario tags -> NavSafe event windows.

Implements the extraction contract of the taxonomy doc II-B::

    window = [t_trig - t_pre,  t_term + t_post]

``t_trig`` comes from the family's trigger, ``t_term`` from its terminal
predicate -- *not* from the tag itself.  nuPlan's ``starting_*`` tags mark an
instant (p90 run length is ~0 s), so the terminal has to be derived; the
``stationary_*`` tags do span time but their end is "still stationary", not
"event over".

The P0 terminal predicates below are **map-free**: they compose ego kinematics
(from ``ego_pose``) with *other tag runs on the same log* -- notably
``on_intersection`` / ``on_traffic_light_intersection``, which nuPlan already
computes against the map.  The map-based refinement (exit-lane polygon, heading
tolerance against the lane's own heading) lands with the map slice in P2.

A window additionally carries a wider ``recon`` extent: reconstruction needs
context beyond the scored interval so the closed-loop ego camera never sits at
the edge of the trained time range (temporal appearance is time-indexed).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, asdict
from typing import Callable

US = 1_000_000

# --- contract constants (taxonomy doc II-B; values fixed in P0, to be
# --- re-calibrated from log statistics in P1) ------------------------------
T_PRE_S = 2.0          # context + policy warm-up (NAVSIM-family needs 1.5 s history)
T_POST_S = 2.0         # observe recovery past the terminal
RECON_MARGIN_S = 1.5   # reconstruction-only margin on each side, never scored
MIN_WINDOW_S = 4.0
MAX_WINDOW_S = 20.0    # single-3090 training budget
TERMINAL_SEARCH_S = 40.0


@dataclass
class EgoTrack:
    ts: list[int]
    x: list[float]
    y: list[float]
    yaw: list[float]
    speed: list[float]

    def idx_at(self, t_us: int) -> int:
        lo, hi = 0, len(self.ts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.ts[mid] < t_us:
                lo = mid + 1
            else:
                hi = mid
        return lo


def load_ego_track(con: sqlite3.Connection) -> EgoTrack:
    ts, xs, ys, yaws, spd = [], [], [], [], []
    q = "SELECT timestamp, x, y, qw, qx, qy, qz, vx, vy FROM ego_pose ORDER BY timestamp"
    for t, x, y, qw, qx, qy, qz, vx, vy in con.execute(q):
        # yaw from quaternion (z-up)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        ts.append(int(t))
        xs.append(float(x))
        ys.append(float(y))
        yaws.append(yaw)
        spd.append(math.hypot(float(vx or 0.0), float(vy or 0.0)))
    return EgoTrack(ts, xs, ys, yaws, spd)


def _unwrapped_yaw_change(ego: EgoTrack, i0: int, i1: int) -> float:
    total = 0.0
    for k in range(i0, min(i1, len(ego.yaw) - 1)):
        d = ego.yaw[k + 1] - ego.yaw[k]
        total += (d + math.pi) % (2 * math.pi) - math.pi
    return total


def _covering_run_end(runs_by_type: dict[str, list[tuple[int, int]]],
                      types: tuple[str, ...], t_us: int) -> int | None:
    """End timestamp of the run of any of ``types`` that covers ``t_us``.

    nuPlan's ``on_intersection`` family is map-derived, so "ego is still inside
    the intersection" is available without loading the map ourselves.
    """
    best = None
    for ty in types:
        for s, e in runs_by_type.get(ty, ()):
            if s - US <= t_us <= e + US:  # 1 s slack: the tags are not frame-aligned
                best = e if best is None else max(best, e)
    return best


# --- family terminal predicates -------------------------------------------
# Each returns (t_term_us, reason) or (None, why_rejected).

INTERSECTION_TYPES = ("on_intersection", "on_traffic_light_intersection",
                      "traversing_intersection", "traversing_traffic_light_intersection")


def terminal_turn(ego: EgoTrack, t_trig: int, runs_by_type, min_turn_deg=60.0):
    """F3 turn: ego exits the intersection AND its heading has stabilised.

    Terminal = the later of (a) the end of the covering intersection run and
    (b) the first time after the turn where the yaw rate settles -- "ego exits
    into target lane, heading aligned".
    """
    i0 = ego.idx_at(t_trig)
    i_end = ego.idx_at(t_trig + int(TERMINAL_SEARCH_S * US))
    turn = _unwrapped_yaw_change(ego, i0, i_end)
    if abs(math.degrees(turn)) < min_turn_deg:
        return None, f"yaw change {math.degrees(turn):.0f} deg < {min_turn_deg}"

    # first index where the accumulated turn reaches 90% of its total and the
    # heading then stays within 5 deg for 0.5 s
    target = 0.9 * turn
    t_aligned = None
    for k in range(i0, i_end):
        if abs(_unwrapped_yaw_change(ego, i0, k)) >= abs(target):
            k2 = ego.idx_at(ego.ts[k] + US // 2)
            if abs(math.degrees(_unwrapped_yaw_change(ego, k, k2))) < 5.0:
                t_aligned = ego.ts[k2]
                break
    t_exit = _covering_run_end(runs_by_type, INTERSECTION_TYPES, t_trig)
    cands = [t for t in (t_aligned, t_exit) if t is not None]
    if not cands:
        return None, "no aligned/exit terminal found"
    return max(cands), f"turn={math.degrees(turn):.0f}deg aligned={t_aligned is not None} exit={t_exit is not None}"


def terminal_straight(ego: EgoTrack, t_trig: int, runs_by_type, max_turn_deg=25.0):
    """F3 straight traversal: ego exits the intersection, heading unchanged."""
    t_exit = _covering_run_end(runs_by_type, INTERSECTION_TYPES, t_trig)
    if t_exit is None:
        return None, "no covering intersection run"
    i0, i1 = ego.idx_at(t_trig), ego.idx_at(t_exit)
    turn = abs(math.degrees(_unwrapped_yaw_change(ego, i0, i1)))
    if turn > max_turn_deg:
        return None, f"yaw change {turn:.0f} deg > {max_turn_deg} (not straight)"
    return t_exit, f"exit, turn={turn:.0f}deg"


def terminal_pull_away(ego: EgoTrack, t_trig: int, runs_by_type,
                       speed_thresh=1.5, hold_s=1.0):
    """F1 stationary at a light: terminal = ego pulls away and stays moving.

    "ego clears stopline at green" -- speed above threshold sustained, so a
    creep forward in the queue does not terminate the event.
    """
    i0 = ego.idx_at(t_trig)
    i_end = ego.idx_at(t_trig + int(TERMINAL_SEARCH_S * US))
    for k in range(i0, i_end):
        if ego.speed[k] > speed_thresh:
            k2 = ego.idx_at(ego.ts[k] + int(hold_s * US))
            if all(ego.speed[m] > speed_thresh for m in range(k, min(k2, len(ego.speed)))):
                return ego.ts[k2], f"pull-away at +{(ego.ts[k] - t_trig) / US:.1f}s"
    return None, "never pulled away within search horizon"


@dataclass(frozen=True)
class Family:
    key: str
    nuplan_tags: tuple[str, ...]
    terminal: Callable
    doc_family: str          # F1..F10 in the taxonomy doc


FAMILIES: dict[str, Family] = {
    "F1_stationary_light_lead": Family(
        "F1_stationary_light_lead",
        ("stationary_at_traffic_light_with_lead",),
        terminal_pull_away,
        "F1",
    ),
    "F3_left_turn": Family(
        "F3_left_turn", ("starting_left_turn",), terminal_turn, "F3",
    ),
    "F3_straight_traversal": Family(
        "F3_straight_traversal",
        ("starting_straight_traffic_light_intersection_traversal",),
        terminal_straight,
        "F3",
    ),
}


@dataclass
class EventWindow:
    family: str
    doc_family: str
    scenario_type: str
    split: str
    log_name: str
    location: str
    map_version: str
    trigger_token: str
    t_trig_us: int
    t_term_us: int
    t0_us: int          # scored window start  (t_trig - t_pre)
    t1_us: int          # scored window end    (t_term + t_post)
    recon_t0_us: int
    recon_t1_us: int
    event_duration_s: float
    window_duration_s: float
    terminal_reason: str


def build_window(fam: Family, run: dict, ego: EgoTrack, runs_by_type) -> tuple[EventWindow | None, str]:
    t_trig = int(run["start_timestamp_us"])
    t_term, reason = fam.terminal(ego, t_trig, runs_by_type)
    if t_term is None:
        return None, reason
    t0 = t_trig - int(T_PRE_S * US)
    t1 = t_term + int(T_POST_S * US)
    dur = (t1 - t0) / US
    if dur < MIN_WINDOW_S:
        t1 = t0 + int(MIN_WINDOW_S * US)
        dur = MIN_WINDOW_S
    if dur > MAX_WINDOW_S:
        return None, f"window {dur:.1f}s > MAX_WINDOW_S {MAX_WINDOW_S}"
    r0 = t0 - int(RECON_MARGIN_S * US)
    r1 = t1 + int(RECON_MARGIN_S * US)
    if r0 < ego.ts[0] or r1 > ego.ts[-1]:
        return None, "window (with recon margin) runs past the log bounds"
    return (
        EventWindow(
            family=fam.key,
            doc_family=fam.doc_family,
            scenario_type=run["scenario_type"],
            split=run["split"],
            log_name=run["log_name"],
            location=run.get("location", ""),
            map_version=run.get("map_version", ""),
            trigger_token=run["start_token"],
            t_trig_us=t_trig,
            t_term_us=t_term,
            t0_us=t0,
            t1_us=t1,
            recon_t0_us=r0,
            recon_t1_us=r1,
            event_duration_s=round((t_term - t_trig) / US, 2),
            window_duration_s=round(dur, 2),
            terminal_reason=reason,
        ),
        "ok",
    )


def window_asdict(w: EventWindow) -> dict:
    return asdict(w)
