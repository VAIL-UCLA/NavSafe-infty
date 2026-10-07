"""The rubric predicate library -- pure functions over an episode trace.

Every predicate takes the scored frames and its declared parameters and returns
a :class:`Verdict`: the boolean, plus the evidence that produced it.  The
evidence matters as much as the verdict; "failed `reach`" is not a finding,
"entered the exit lane at 18.3 s against a 15.8 s budget" is.

Two invariants hold throughout:

* **Warm-up frames are never scored.**  During warm-up the ego replays logged
  actions, so scoring them would measure the log, not the policy.
* **Frames the reconstruction cannot certify are never scored.**  Rendering
  outside the validity envelope can fail a policy for reasons that are not the
  policy's; those frames are dropped from evaluation and counted separately so
  the exclusion stays visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

Frame = dict[str, Any]


@dataclass
class Verdict:
    name: str
    passed: bool
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def scorable(frames: Sequence[Frame]) -> list[Frame]:
    """Frames the policy is answerable for.

    Warm-up is replayed ground truth, so it is never scored. There is no second
    test any more: the render-validity envelope that also excluded frames here
    was removed (see trace/writer.py rule 3), and a filter no
    producer can trigger reads as a guarantee the code no longer makes.
    """
    return [f for f in frames if f.get("phase") == "scored"]


def _t0(frames: Sequence[Frame]) -> float:
    return frames[0]["t_sim_s"] if frames else 0.0


# --- success predicates (taxonomy doc IV-B) --------------------------------

def reach(frames, *, region: str, t_max: float, **_) -> Verdict:
    """Ego enters ``region`` within the time budget."""
    fs = scorable(frames)
    if not fs:
        return Verdict("reach", False, "no scorable frames")
    t0 = _t0(fs)
    for f in fs:
        if f.get(_region_flag(region)):
            dt = f["t_sim_s"] - t0
            return Verdict("reach", dt <= t_max,
                           f"entered {region} at {dt:.1f}s (budget {t_max:.1f}s)",
                           {"t_reach_s": dt, "t_max_s": t_max})
    held = fs[-1]["t_sim_s"] - t0
    return Verdict("reach", False,
                   f"never entered {region} in {held:.1f}s (budget {t_max:.1f}s)",
                   {"t_reach_s": None, "t_max_s": t_max, "elapsed_s": held})


def stabilize(frames, *, region: str, tau: float, **_) -> Verdict:
    """Ego stays inside ``region`` for ``tau`` continuous seconds."""
    fs = scorable(frames)
    flag = _region_flag(region)
    best = run_start = 0.0
    run = None
    for f in fs:
        if f.get(flag):
            run = f["t_sim_s"] if run is None else run
            best = max(best, f["t_sim_s"] - run)
        else:
            run = None
    return Verdict("stabilize", best >= tau,
                   f"longest continuous stay in {region} {best:.1f}s (need {tau:.1f}s)",
                   {"longest_s": best, "tau_s": tau})


def align(frames, *, lane: str, heading_tol: float, **_) -> Verdict:
    """Ego heading matches its lane within tolerance, at the end of the episode.

    Evaluated on the final scorable frame: alignment is a property of having
    *completed* the maneuver, not of every instant during it.
    """
    fs = scorable(frames)
    if not fs:
        return Verdict("align", False, "no scorable frames")
    f = fs[-1]
    err = abs(_wrap_deg(math.degrees(f["ego_yaw"] - f["lane_heading"])))
    return Verdict("align", err <= heading_tol,
                   f"final heading error {err:.1f} deg (tol {heading_tol:.1f})",
                   {"heading_err_deg": err, "tol_deg": heading_tol,
                    "lane_id": f.get("lane_id", "")})


def stop_before(frames, *, line: str, v_eps: float, **_) -> Verdict:
    """Ego comes to a near-stop before crossing ``line``."""
    fs = scorable(frames)
    before = [f for f in fs if f.get("dist_to_stopline_m", 0.0) > 0]
    if not before:
        return Verdict("stop_before", False, f"never approached {line}")
    vmin = min(f["ego_speed"] for f in before)
    return Verdict("stop_before", vmin <= v_eps,
                   f"min speed before {line} {vmin:.2f} m/s (need <= {v_eps:.2f})",
                   {"v_min_mps": vmin, "v_eps": v_eps})


def clear(frames, *, conflict_zone: str,
          conflicting_heading_deg: Sequence[float] = (45.0, 135.0),
          min_ttc_s: float = 2.0, min_gap_m: float = 2.0, **_) -> Verdict:
    """Ego traverses the conflict zone with no *conflicting* agent inside it.

    Which relative headings count as conflicting is a property of the maneuver,
    so the band is declared in the rubric rather than fixed here:

    * going straight, the conflict is **crossing** traffic (~90 deg).  Oncoming
      traffic (~180 deg) stays in the opposing lanes and is not a conflict --
      gating on it would fail every straight traversal ever recorded, the
      logged human's included.
    * turning left across traffic, **oncoming** is exactly the conflict; that
      is what makes the turn unprotected, so the band extends to 180 deg.

    Same-direction traffic is never a conflict here: longitudinal separation
    from a lead vehicle is what :func:`maintain_gap` is for.

    Co-presence alone does not make a conflict either.  A human turning left
    does it in a gap, with oncoming vehicles present in the junction the whole
    time; a human going straight passes cross traffic waiting at its own line.
    Gating on occupancy would fail both, so the test is **spatio-temporal**: a
    conflicting agent must also come within an unsafe time-to-contact or
    clearance.  This is where the doc's prose ("no simultaneous occupancy by a
    conflicting agent") has to be made operational, and the floors are declared
    per family in the rubric rather than fixed here.
    """
    lo, hi = float(conflicting_heading_deg[0]), float(conflicting_heading_deg[1])
    fs = scorable(frames)
    bad, ids = [], set()
    for f in fs:
        if not f.get("in_conflict_zone"):
            continue
        hit = []
        for a in f.get("agents", ()):
            if not a.get("in_conflict_zone"):
                continue
            d = abs(_wrap_deg(math.degrees(a.get("yaw", 0.0) - f["ego_yaw"])))
            if not (lo <= d <= hi):
                continue
            ttc = a.get("ttc_s", float("inf"))
            gap = a.get("clearance_m")
            if gap is None or gap != gap:
                gap = a.get("dist_to_ego", float("inf"))
            if (ttc == ttc and ttc < min_ttc_s) or (gap == gap and gap < min_gap_m):
                hit.append(a)
        if hit:
            bad.append(f)
            ids.update(a["id"] for a in hit)
    band = {"conflicting_heading_deg": [lo, hi],
            "min_ttc_s": min_ttc_s, "min_gap_m": min_gap_m}
    if not bad:
        return Verdict("clear", True, f"no conflicting agent in {conflict_zone}", band)
    return Verdict("clear", False,
                   f"shared {conflict_zone} with {len(ids)} conflicting agent(s) "
                   f"over {len(bad)} frame(s)",
                   {"n_frames": len(bad), "agent_ids": sorted(ids)[:8], **band})


def maintain_gap(frames, *, d_min: float, agent: str = "*",
                 fault: str = "at_fault", **_) -> Verdict:
    """Distance to ``agent`` (or any agent) stays above the floor.

    Like :func:`no_collision`, this respects fault by default.  A non-reactive
    replayed follower closing on a correctly-behaving ego is the replay's
    doing, not the policy's, so only approaches the ego is *causing* are gated
    (``fault="any"`` gates on every approach and is what an analyst uses to
    count near misses regardless of blame).
    """
    fs = scorable(frames)
    worst, worst_id, worst_t = math.inf, "", None
    excluded = 0
    for f in fs:
        for a in f.get("agents", ()):
            if agent not in ("*", a["id"]):
                continue
            if fault != "any" and not a.get("ego_is_closing", True):
                excluded += 1
                continue
            # Clearance, not centre distance: a 0.5 m floor on centres would
            # mean the boxes overlap. Falls back to centre distance only for
            # traces written before clearance existed.
            d = a.get("clearance_m")
            if d is None or d != d:
                d = a.get("dist_to_ego", math.inf)
            if d < worst:
                worst, worst_id, worst_t = d, a["id"], f["t_sim_s"]
    if worst is math.inf:
        return Verdict("maintain_gap", True,
                       "no ego-caused approach to gate on",
                       {"n_excluded_approaches": excluded})
    return Verdict("maintain_gap", worst >= d_min,
                   f"closest clearance {worst:.2f} m to {worst_id} "
                   f"at t={worst_t:.1f}s (floor {d_min:.2f})",
                   {"min_gap_m": worst, "agent_id": worst_id, "t_s": worst_t,
                    "d_min": d_min, "n_excluded_approaches": excluded})


def comply(frames, *, signal: str = "", t_react: float = 3.0, **_) -> Verdict:
    """No motion through red; move off within ``t_react`` of green.

    Two failures, deliberately in one predicate: running a red is unsafe and
    sitting through a green is the over-conservatism the outcome axis names.
    """
    fs = scorable(frames)
    ran_red = [f for f in fs
               if f.get("signal_state") == "red" and f.get("in_intersection")
               and f["ego_speed"] > 0.5]
    if ran_red:
        return Verdict("comply", False,
                       f"entered on red at t={ran_red[0]['t_sim_s']:.1f}s",
                       {"violation": "red_running",
                        "t_s": ran_red[0]["t_sim_s"]})
    green_from = next((f["t_sim_s"] for f in fs if f.get("signal_state") == "green"), None)
    if green_from is None:
        return Verdict("comply", True, "signal never permitted; no red run")
    moved = next((f["t_sim_s"] for f in fs
                  if f["t_sim_s"] >= green_from and f["ego_speed"] > 0.5), None)
    if moved is None:
        held = fs[-1]["t_sim_s"] - green_from
        return Verdict("comply", False,
                       f"never moved off after {held:.1f}s of green",
                       {"violation": "failed_to_proceed", "green_held_s": held})
    delay = moved - green_from
    return Verdict("comply", delay <= t_react,
                   f"moved off {delay:.1f}s after green (react budget {t_react:.1f}s)",
                   {"reaction_s": delay, "t_react": t_react})


# --- hard gates ------------------------------------------------------------

def no_collision(frames, *, fault: str = "at_fault", **_) -> Verdict:
    """No contact attributable to the ego under the regime's fault rule.

    ``fault="any"`` gates on every contact; ``at_fault`` only on the ego's --
    under log-replay a replayed follower cannot brake, so rear-ending the ego
    is not the ego's failure (the nuPlan precedent, doc Table VI).
    """
    fs = scorable(frames)
    hits = [(f, c) for f in fs for c in f.get("contacts", ())
            if fault == "any" or c.get("at_fault")]
    if not hits:
        return Verdict("no_collision", True, "no gated contact")
    f, c = hits[0]
    return Verdict("no_collision", False,
                   f"{c.get('kind', 'contact')} with {c['agent_id']} "
                   f"at t={f['t_sim_s']:.1f}s",
                   {"n_contacts": len(hits), "kind": c.get("kind"),
                    "agent_id": c["agent_id"], "t_s": f["t_sim_s"],
                    "rel_speed": c.get("rel_speed")})


def on_drivable(frames, **_) -> Verdict:
    fs = scorable(frames)
    bad = [f for f in fs if not f.get("on_drivable", True)]
    if not bad:
        return Verdict("on_drivable", True, "stayed on drivable area")
    return Verdict("on_drivable", False,
                   f"off drivable for {len(bad)} frame(s), first at "
                   f"t={bad[0]['t_sim_s']:.1f}s",
                   {"n_frames": len(bad), "t_first_s": bad[0]["t_sim_s"]})


# --- registry --------------------------------------------------------------

SUCCESS: dict[str, Callable[..., Verdict]] = {
    "reach": reach, "stabilize": stabilize, "align": align,
    "stop_before": stop_before, "clear": clear, "maintain_gap": maintain_gap,
    "comply": comply,
}
GATES: dict[str, Callable[..., Verdict]] = {
    "no_collision": no_collision, "on_drivable": on_drivable,
    "maintain_gap": maintain_gap,
}


# --- helpers ---------------------------------------------------------------

_REGION_FLAGS = {
    "exit_lane_polygon": "in_exit_lane",
    "exit_region_polygon": "in_exit_lane",
    "intersection_polygon": "in_intersection",
    "intersection_conflict_zone": "in_conflict_zone",
}


def _region_flag(region: str) -> str:
    """Map a rubric region name to its per-frame boolean column.

    Regions are resolved to booleans by the trace writer (which has the map),
    not here: a predicate that re-derived geometry would be evaluating a
    different map than the simulator used.
    """
    return _REGION_FLAGS.get(region, f"in_{region}")


def _wrap_deg(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0
