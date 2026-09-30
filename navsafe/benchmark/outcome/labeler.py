"""Outcome labelling: what actually happened, read off the trace.

Two rules the taxonomy doc is explicit about, and both are easy to violate:

* Labels are assigned **from the simulator trace, never from scenario intent**.
  A scenario built to provoke a rear-end that instead ends in a deadlock gets
  the deadlock label.  Labelling by intent would make the coverage claim
  circular -- every crash cell would be "populated" by construction.
* Labelling is **multi-label**.  One episode can contain a near conflict, a
  missed yield and a late-braking collision; compressing that into one name
  loses exactly the detail a failure analysis needs.

The eight labels are the paper's outcome axis (doc VI, Table VIII); the
RoadSafe365 / ANSI D16 crash cell is derived from the contact kind, so the
coverage claim can be broken down per crash type rather than reported only in
aggregate.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from navsafe.benchmark.rubric import predicates as P

# --- the paper's eight outcome labels (doc Table VIII) ---------------------
CONTACT = "contact"
NEAR_CONFLICT = "near_conflict"
ROAD_RULE_VIOLATION = "road_rule_violation"
MISSED_YIELD = "missed_yield"
UNNECESSARY_STOP = "unnecessary_stop"
DEADLOCK = "deadlock"
COMFORT_VIOLATION = "comfort_violation"
DELAYED_RECOVERY = "delayed_recovery"

# --- thresholds. Diagnostic, not safety gates: these decide how an episode is
# *described*, and are deliberately separate from the rubric's pass/fail floors
# so relabelling never silently changes who passed.
TTC_NEAR_CONFLICT_S = 1.5
CLEARANCE_NEAR_CONFLICT_M = 1.0
STOP_SPEED_MPS = 0.3
UNNECESSARY_STOP_S = 3.0     # stopped this long with nothing to stop for
DEADLOCK_S = 5.0             # no progress this long in a solvable scene
DEVIATION_M = 1.5            # lateral departure from the route that counts
JERK_LIMIT = 5.0             # m/s^3
LAT_ACCEL_LIMIT = 4.0        # m/s^2

# --- RoadSafe365 L1 categories, keyed off the contact kind recorded in the
# trace (doc Table VII).  L2 is the contact kind itself.
CRASH_L1 = {
    "rear_end": "traffic_crashes",
    "angle": "traffic_crashes",
    "sideswipe": "traffic_crashes",
    "single": "traffic_crashes",
    "vru": "vru_crashes",
}


def label_episode(frames: Sequence[dict], rubric_result=None, *,
                  comfort_measurable: bool = True) -> dict[str, Any]:
    """Return the outcome labels plus the evidence behind each.

    ``comfort_measurable=False`` suppresses the comfort label. Under teleport
    execution the ego is placed at waypoints rather than driven, so jerk is an
    artifact of the placement cadence; labelling that as a comfort violation
    would put an unsupported entry on the outcome axis the coverage claim rests
    on.
    """
    fs = P.scorable(frames)
    labels: dict[str, dict] = {}

    def mark(name: str, **evidence) -> None:
        labels[name] = evidence

    # --- contact -----------------------------------------------------------
    contacts = [(f, c) for f in fs for c in f.get("contacts", ())]
    if contacts:
        at_fault = [c for _, c in contacts if c.get("at_fault")]
        kinds = sorted({c.get("kind", "unknown") for _, c in contacts})
        mark(CONTACT,
             n=len(contacts), at_fault=len(at_fault), kinds=kinds,
             crash_l1=sorted({CRASH_L1.get(k, "unclassified") for k in kinds}),
             first_t_s=contacts[0][0]["t_sim_s"])

    # --- near conflict: severity precursor, no contact ----------------------
    worst_ttc, worst_clear = float("inf"), float("inf")
    for f in fs:
        for a in f.get("agents", ()):
            if not a.get("ego_is_closing", True):
                continue
            t = a.get("ttc_s", float("inf"))
            if t == t and t < worst_ttc:
                worst_ttc = t
            d = a.get("clearance_m")
            if d is None or d != d:
                d = a.get("dist_to_ego", float("inf"))
            if d == d and d < worst_clear:
                worst_clear = d
    if not contacts and (worst_ttc < TTC_NEAR_CONFLICT_S
                         or worst_clear < CLEARANCE_NEAR_CONFLICT_M):
        mark(NEAR_CONFLICT, min_ttc_s=_fin(worst_ttc), min_clearance_m=_fin(worst_clear))

    # --- road/rule violation ------------------------------------------------
    ran_red = [f for f in fs if f.get("signal_state") == "red"
               and f.get("in_intersection") and f["ego_speed"] > 0.5]
    off_road = [f for f in fs if not f.get("on_drivable", True)]
    if ran_red or off_road:
        mark(ROAD_RULE_VIOLATION,
             red_running_frames=len(ran_red),
             off_drivable_frames=len(off_road),
             first_t_s=(ran_red or off_road)[0]["t_sim_s"])

    # --- missed yield --------------------------------------------------------
    # Same notion of "conflicting" as predicates.clear: crossing or oncoming,
    # not merely co-present. Sharing a junction with same-direction traffic is
    # normal driving, and labelling it as a missed yield would put the label on
    # every signalised traversal including the logged human's.
    def _conflicting(f, a) -> bool:
        if not a.get("in_conflict_zone"):
            return False
        d = abs(((math.degrees(a.get("yaw", 0.0) - f["ego_yaw"]) + 180) % 360) - 180)
        if not 45.0 <= d <= 135.0:          # crossing only, for the label
            return False
        ttc = a.get("ttc_s", float("inf"))
        gap = a.get("clearance_m")
        if gap is None or gap != gap:
            gap = a.get("dist_to_ego", float("inf"))
        return (ttc == ttc and ttc < TTC_NEAR_CONFLICT_S) or (gap == gap and gap < 2.0)

    shared = [f for f in fs if f.get("in_conflict_zone")
              and any(_conflicting(f, a) for a in f.get("agents", ()))]
    if shared:
        mark(MISSED_YIELD, n_frames=len(shared), first_t_s=shared[0]["t_sim_s"])

    # --- stopped-when-it-should-not-be --------------------------------------
    # Split deliberately: an unnecessary stop is over-conservatism that
    # resolves, a deadlock never resolves. They are different failures and the
    # paper lists them separately.
    stops = _stalls(fs)
    for start, end, blocked, permitted in stops:
        dur = end - start
        if dur >= DEADLOCK_S and end >= fs[-1]["t_sim_s"] - 1e-6:
            mark(DEADLOCK, stalled_s=round(dur, 2), from_t_s=round(start, 2))
        elif dur >= UNNECESSARY_STOP_S and not blocked and permitted:
            mark(UNNECESSARY_STOP, stopped_s=round(dur, 2), from_t_s=round(start, 2))

    # --- comfort ------------------------------------------------------------
    if comfort_measurable:
        max_jerk = max((abs(f.get("ego_jerk", 0.0)) for f in fs), default=0.0)
        max_lat = max((abs(f.get("ego_lat_accel", 0.0)) for f in fs), default=0.0)
        if max_jerk > JERK_LIMIT or max_lat > LAT_ACCEL_LIMIT:
            mark(COMFORT_VIOLATION, max_abs_jerk=round(max_jerk, 2),
                 max_lat_accel=round(max_lat, 2))

    # --- delayed / failed recovery ------------------------------------------
    # Deviation is measured against the ROUTE (the logged path), not against a
    # lane centreline: real lanes are 3.1-7.3 m wide and flare through turns, so
    # a human tracking a wide lane sits >1 m off centre for most of a junction
    # and would be labelled as deviating for driving normally.
    dev = next((f for f in fs if abs(f.get("ego_dev_lat_m", 0.0)) > DEVIATION_M), None)
    if dev is not None:
        back = next((f for f in fs
                     if f["t_sim_s"] > dev["t_sim_s"]
                     and abs(f.get("ego_dev_lat_m", 0.0)) < DEVIATION_M / 2), None)
        if back is None:
            mark(DELAYED_RECOVERY, deviated_at_s=round(dev["t_sim_s"], 2),
                 recovered=False)
        elif back["t_sim_s"] - dev["t_sim_s"] > 3.0:
            mark(DELAYED_RECOVERY, deviated_at_s=round(dev["t_sim_s"], 2),
                 recovered=True,
                 recovery_s=round(back["t_sim_s"] - dev["t_sim_s"], 2))

    return {
        "labels": sorted(labels),
        "evidence": labels,
        "crash_cells": sorted(labels.get(CONTACT, {}).get("crash_l1", [])),
    }


def _fin(v: float):
    return None if v == float("inf") else round(v, 3)


def _stalls(fs: Sequence[dict]):
    """Yield (start_s, end_s, blocked, permitted) for each stopped stretch.

    ``blocked`` -- something was actually in the way, so stopping was correct.
    ``permitted`` -- the signal allowed proceeding, so stopping was not.
    """
    out = []
    start = None
    blocked = permitted = False
    for f in fs:
        stopped = f["ego_speed"] < STOP_SPEED_MPS
        if stopped and start is None:
            start = f["t_sim_s"]
            blocked = permitted = False
        if stopped:
            lead = next((a for a in f.get("agents", ()) if a.get("is_lead")), None)
            if lead is not None and lead.get("dist_to_ego", 1e9) < 8.0:
                blocked = True
            if f.get("signal_state") == "green":
                permitted = True
        if not stopped and start is not None:
            out.append((start, f["t_sim_s"], blocked, permitted))
            start = None
    if start is not None and fs:
        out.append((start, fs[-1]["t_sim_s"], blocked, permitted))
    return out
