# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The NavSafe scoring chain, as one pure function over a stored run.

``navsafe/tools/score_run.py`` and the evaluator both need exactly
this: rebuilt trace frames in, the four NavSafe metrics out. It lives here
so the CLI and the in-run path cannot drift into two subtly different
definitions of "driving score" — which is the failure this benchmark exists to
avoid.

    frames -> termination.classify        why the episode ended (one reason)
           -> infractions_from_trace      fault-respecting contact counts
           -> route.route_progress        monotone route completion
           -> termination.to_route_result RouteResult, or None
           -> DS / SR / Efficiency / Comfort

Pure post-processing over ``EvalRun``: no simulator, no GPU, no clock.
"""

from __future__ import annotations

import json
import math

from typing import Any, Mapping, Optional

import numpy as np

from navsafe.benchmark import termination as term
from navsafe.benchmark import scenario_rules
from navsafe.benchmark.scoring import metrics as navsafe_metrics
from navsafe.benchmark.scoring import route as route_mod
from navsafe.benchmark.trace import from_eval

# t_max is the logged human completion time times this factor, with a 10 s
# floor.
T_MAX_SLACK = 1.3
T_MAX_FLOOR_S = 10.0

#: Hard ceiling on an episode, in seconds. An episode otherwise ends on an
#: event (goal, contact, leaving the drivable surface, deadlock); the
#: ceiling stops a policy that keeps driving on drivable road without
#: finishing.
SAFETY_CEILING_S = 60.0

# Cap the budget at the length of the scenario. A bundle is a 20 s window
# that the logged drive fills, so 1.3 times the human time would exceed the
# frames that exist, and every episode would end short of its budget.
CAP_T_MAX_AT_SCENARIO = True

MIN_SCORABLE_FRAMES = 10



#: The penalty channels that can be evaluated here, and their sources.
#:
#: Collisions come from the per-frame contact record and
#: `outside_route_lanes` from the off-route distance. `red_light` is
#: deliberately not scored from the logged light states: they are not
#: reliable enough to justify a 0.70 multiplier, and a wrong state cannot be
#: told apart from a real violation afterwards. Authored red-light holds
#: (V-1) are unaffected; they are judged against a per-bundle region by
#: `scenario_rules.hold_violation`.
#:
#: Channels with no source are reported as skipped, not as zero, so that an
#: unchecked channel does not read as clean.
_CHANNEL_SOURCES = {
    "collisions_pedestrian": "contact kind 'vru'",
    "collisions_vehicle": "contact kinds rear_end/angle/sideswipe, or the "
                          "evaluator's collision count",
    "collisions_layout": "contact kind 'single'",
    navsafe_metrics.OUTSIDE_LANES_KEY: "off-route distance (needs a map)",
}
_SKIPPED_CHANNELS = tuple(
    k for k in navsafe_metrics.PENALTY_COEFFICIENTS if k not in _CHANNEL_SOURCES)


#: The channel exists only where the scenario asked for it, so it is described
#: here but gated at the call site rather than being unconditionally scored.
_STOP_SIGN_SOURCE = ("dwell below the stop-sign speed floor before the ego "
                     "crossed the line (V-1 stop-sign scenarios only)")


def _penalty_channels(infr: Mapping[str, float],
                      drivable_known: bool,
                      stop_sign_checked: bool = False) -> tuple[list[dict], float]:
    """Itemise score_penalty so the driving score can be recomputed by hand.

    Returns the per-channel rows and their product. Only channels this path
    actually evaluates appear; see ``_CHANNEL_SOURCES``. ``red_light`` is not
    one of them on any run — see that table for why.
    """
    rows: list[dict] = []
    penalty = 1.0
    for key, coeff in navsafe_metrics.PENALTY_COEFFICIENTS.items():
        source = _CHANNEL_SOURCES.get(key)
        if key == "stop_infraction" and stop_sign_checked:
            source = _STOP_SIGN_SOURCE
        if source is None:
            continue
        n = int(infr.get(key, 0))
        factor = coeff ** n
        penalty *= factor
        rows.append({"channel": key, "coefficient": coeff, "count": n,
                     "factor": round(factor, 6),
                     "source": source})
    if drivable_known:
        pct = float(infr.get(navsafe_metrics.OUTSIDE_LANES_KEY, 0.0))
        factor = 1.0 - pct / 100.0
        penalty *= factor
        rows.append({"channel": navsafe_metrics.OUTSIDE_LANES_KEY,
                     "coefficient": "1 - pct/100", "pct": round(pct, 3),
                     "factor": round(factor, 6),
                     "source": _CHANNEL_SOURCES[navsafe_metrics.OUTSIDE_LANES_KEY]})
    return rows, penalty



def scenario_meta_for(py123d_data_root: Any) -> dict:
    """``scenario_meta`` from the bundle that a py123d data root belongs to.

    The root is ``<bundle>/arrow``, so the manifest is its parent's. Both meta
    schemas are accepted: two early bundles carry singular ``scenario_type`` /
    ``taxonomy_leaf`` keys, everything since carries the plural tuples. Returns
    ``{}`` when there is no manifest, so a run on a bare scenario still scores.
    """
    from pathlib import Path

    root = Path(str(py123d_data_root))
    man = (root.parent if root.name == "arrow" else root) / "manifest.json"
    if not man.is_file():
        return {}
    try:
        manifest = json.loads(man.read_text())
    except Exception:                                    # noqa: BLE001
        return {}
    meta = manifest.get("scenario_meta") or {}
    if not meta:
        return {}

    def pick(plural: str, singular: str) -> list:
        v = meta.get(plural)
        if v:
            return list(v)
        one = meta.get(singular)
        return [one] if one else []

    return {
        "token": meta.get("token"),
        "scenario_types": pick("scenario_types", "scenario_type"),
        "taxonomy_leaves": pick("taxonomy_leaves", "taxonomy_leaf"),
        "taxonomy_leaf_names": pick("taxonomy_leaf_names", "taxonomy_leaf_name"),
        "has_inserted_actors": meta.get("has_inserted_actors"),
        # A signal override belongs to the scenario as authored. It
        # can also be given through the environment, for a bundle
        # whose manifest cannot be edited; a manifest entry takes
        # precedence.
        "signal_override": (manifest.get("signal_override")
                            or _sidecar_signal_override()),
    }


def _sidecar_signal_override():
    from navsafe.benchmark import signal_override as _sig
    try:
        return _sig.from_env()
    except Exception:                                        # noqa: BLE001
        return None


def score_run(run: from_eval.EvalRun, *, name: str, warmup_frames: int = 0,
              dt: float = 0.1, t_max: Optional[float] = None,
              source: str = "",
              scenario_meta: Optional[Mapping[str, Any]] = None,
              live_termination: Optional[str] = None) -> dict[str, Any]:
    """The four NavSafe metrics for one finished run.

    Raises ``ValueError`` when there is not enough of an episode to score --
    the caller decides whether that is fatal (the CLI) or a note on an
    otherwise-successful run (the evaluator).

    ``live_termination`` is the reason the EVALUATOR ended the episode, when
    the caller has it. This pass reconstructs the ending from the stored
    trace alone, and a trace that stops early looks identical whether the
    policy ran out of frames or the renderer died: measured on
    ``00c1e4eb4a045f20`` — ``nurec_grpc render failed for CAM_F0`` at frame
    117, which the evaluator recorded as ``infra_failure`` and this pass
    re-read as ``trace_exhausted``, publishing ``status: scored`` and a
    driving score of 45.8 for an episode no policy was responsible for. A
    benchmark failure folded into the mean is the one error the two-cause
    rule exists to prevent, so the live reason wins when it is a benchmark
    reason.
    """
    frames = run.frames
    # A hard floor: two scored frames are needed before any derivative exists.
    if len(frames) <= warmup_frames + 1:
        raise ValueError(
            f"only {len(frames)} frames recorded (warmup {warmup_frames}) "
            "— nothing to score")
    # Whether a short trace can be scored depends on why it is short. One
    # that simply stops cannot; one that ends on a decisive event
    # attributable to the policy, such as crossing a stop line a second
    # after hand-off, is a complete result.
    short = len(frames) <= warmup_frames + MIN_SCORABLE_FRAMES

    # --- route progress (monotone) ---------------------------------------
    prog = route_mod.route_progress(run.route_xy, run.ego_xy)
    goal_frame = (prog.goal_frame
                  if prog.goal_frame is not None and prog.goal_frame >= warmup_frames
                  else None)
    goal_reached = goal_frame is not None
    # Fall back to the evaluator's own goal verdict if the recomputation
    # disagrees, for example when the two used different route sources.
    # The goal frame stays unknown, so classify() places the goal at the
    # last scored frame and any earlier event still takes precedence.
    reported_goal = str(
        run.metrics.get("termination_reason") or ""
    ) == term.TerminationReason.GOAL_REACHED.value
    goal_from_report = reported_goal and not goal_reached
    if goal_from_report:
        goal_reached = True

    # --- time budget -----------------------------------------------------
    human_s = max(len(run.route_xy) - 1 - warmup_frames, 0) * dt
    # The budget is the safety ceiling, not a multiple of the human's drive.
    # Capping at `human_s` (CAP_T_MAX_AT_SCENARIO) was right while an episode
    # could not outlive its bundle; now that it can, capping there would end
    # every run at the 20 s mark again.
    t_max_s = t_max if t_max is not None else SAFETY_CEILING_S
    time_limited = math.isfinite(t_max_s)
    window_s = max(len(frames) - warmup_frames, 0) * dt
    # A window shorter than the budget means the harness cap ended the run, not
    # the clock: `budget_expired` then says nothing about the policy's pace.
    window_short = time_limited and window_s < t_max_s - 0.5 * dt

    # --- scenario rules ---
    # The event type in the bundle manifest selects the termination and
    # success rules. Event types without an entry in scenario_rules use
    # the defaults.
    rules = scenario_rules.rules_for_scenario(scenario_meta)

    # --- termination -----------------------------------------------------
    t = term.classify(frames, goal_reached=goal_reached, goal_frame=goal_frame,
                      t_max_s=t_max_s, dt=dt, rules=rules)
    if short and (t.reason.is_no_event_fallback
                  or t.reason is term.TerminationReason.BUDGET_EXPIRED):
        # Nothing decisive happened in the few frames there are, so the trace
        # cannot say what the policy did. `hold_satisfied` is among the
        # fallbacks here for the same reason it is one live: holding for 0.9 s
        # is not holding.
        raise ValueError(
            f"only {len(frames)} frames recorded (warmup {warmup_frames}) and "
            f"they end in {t.reason.value} — nothing to score")
    # V-8 succeeds by not entering the prohibited lane, so a trace in
    # which nothing happens would look like a perfect hold. Success
    # therefore requires that the ego was inside the intersection at some
    # point; otherwise the episode is unscorable. A contact before the
    # intersection is still a decisive outcome and is scored normally.
    decisive_pre_event_outcome = t.reason in {
        term.TerminationReason.CONTACT_AT_FAULT,
        term.TerminationReason.CONTACT_NOT_AT_FAULT,
    }
    if (rules.hold_region == "no_turn_lane"
            and t.reason is not term.TerminationReason.ILLEGAL_TURN
            and not decisive_pre_event_outcome):
        seen = [f for f in frames[:t.frame + 1]
                if f.get("phase") == from_eval.PHASE_SCORED]
        if not any("in_intersection" in f for f in seen):
            raise ValueError(
                "V-8 requires the in_intersection column and this trace has "
                "none — cannot tell whether the ego met the event")
        if not any(f.get("in_intersection") for f in seen):
            raise ValueError(
                f"ego never entered the intersection in {len(seen)} scored "
                f"frames (ended {t.reason.value}) — the event was never met, "
                "so not entering the turn lane is not evidence of compliance")

    # The episode ended at t.frame; nothing after it is the episode's.
    scored_idx = [i for i, f in enumerate(frames[:t.frame + 1])
                  if f["phase"] == from_eval.PHASE_SCORED]
    scored_frames = [frames[i] for i in scored_idx]
    completion = prog.at(t.frame)
    if t.reason is term.TerminationReason.GOAL_REACHED:
        completion = 100.0
    if t.reason in (term.TerminationReason.HOLD_SATISFIED,
                    term.TerminationReason.RED_LIGHT_RUN,
                    term.TerminationReason.ILLEGAL_TURN):
        # On a hold event type the route runs through the
        # intersection the ego must not enter, so route completion
        # would measure the opposite of compliance. The fraction of
        # the window for which the hold was kept is used instead.
        completion = scenario_rules.hold_fraction(
            frames,
            t.frame if t.reason in (term.TerminationReason.RED_LIGHT_RUN,
                                    term.TerminationReason.ILLEGAL_TURN) else None,
            # Measure against the budget, not the trace: a
            # crossing ends the episode and shortens the trace.
            # Without a time budget, the recorded window is the
            # window.
            window_frames=(int(round(t_max_s / dt))
                           if math.isfinite(t_max_s)
                           else max(len(frames) - warmup_frames, 1)))

    # --- infractions -----------------------------------------------------
    infractions = navsafe_metrics.infractions_from_trace(
        scored_frames, at_fault_only=True)
    # Count a collision that the per-frame record cannot place, but only
    # if there are no contact columns at all. If the columns exist and
    # attribute the contact to another agent, the ego is not charged.
    placed_contacts = any(f.get("contacts") for f in frames)
    if run.collision_count and not infractions and not placed_contacts:
        infractions["collisions_vehicle"] = float(run.collision_count)
    # No `red_light` infraction is raised (see _CHANNEL_SOURCES). The
    # stop-sign check is tri-state: None means the trace could not
    # answer, and the channel stays skipped.
    stop_sign_checked = False
    stop_note = ""
    if rules.stop_sign_dwell_s is not None:
        violated, stop_note = scenario_rules.stop_sign_violation(
            scored_frames, rules)
        if violated is not None:
            stop_sign_checked = True
            if violated:
                infractions["stop_infraction"] = 1.0
    off_road_pct = None
    if run.drivable_known and scored_frames:
        xy = np.array([[f["ego_x"], f["ego_y"]] for f in scored_frames])
        off_road_pct = route_mod.outside_route_lanes_pct(
            xy, [f["on_drivable"] for f in scored_frames])
        if off_road_pct > 0:
            infractions[navsafe_metrics.OUTSIDE_LANES_KEY] = off_road_pct

    route = term.to_route_result(name, completion, t, infractions, rules)

    # --- NavSafe metrics -------------------------------------------------
    notes = list(run.notes)
    if goal_from_report:
        notes.append(
            f"goal taken from the evaluator's metrics.json: this run's own "
            f"geometry says {prog.completion_pct:.2f} % of the route driven, "
            f"{prog.dist_to_goal_m:.2f} m from the goal, which route.at_goal() "
            f"does not call reached. The two tests are meant to agree — treat a "
            f"disagreement as a bug in one of them, not as a scoring detail.")
    eff = comfort = None
    if scored_frames:
        try:
            e = navsafe_metrics.efficiency_inputs_from_trace(scored_frames)
            # Checkpoints are every 5 % of the ROUTE, so the axis is the
            # monotone route completion at each scored frame, not the ego's
            # odometer. prog.at() is the same monotone cursor the route
            # completion metric uses, so both agree on where the ego was.
            route_pct = np.array([prog.at(i) for i in scored_idx], dtype=float)
            eff = navsafe_metrics.route_efficiency(
                e["ego_speed"], e["background_mean_speed"], route_pct)
            if eff is None:
                # Say why efficiency is missing, so that a
                # null value next to a finished route is not
                # mistaken for an error.
                bg_ok = e["background_mean_speed"][
                    np.isfinite(e["background_mean_speed"])]
                bg_med = float(np.median(bg_ok)) if bg_ok.size else float("nan")
                notes.append(
                    f"efficiency not reported: background traffic median "
                    f"{bg_med:.2f} m/s is below the "
                    f"{navsafe_metrics.EFFICIENCY_MIN_BACKGROUND_MS:.1f} m/s "
                    f"floor (or no checkpoint had moving traffic), so the "
                    f"ratio has no denominator — this scenario cannot measure "
                    f"efficiency, it is not a slow policy")
            c = navsafe_metrics.comfort_inputs_from_trace(scored_frames, dt=dt)
            comfort = navsafe_metrics.route_comfort(
                c["lon_accel"], c["lat_accel"], c["mag_accel"], c["yaw_rate"])
        except ValueError as exc:
            notes.append(f"metrics unavailable: {exc}")
    if stop_note:
        notes.append(f"stop sign: {stop_note}")
    if window_short:
        need = int(round(t_max_s / dt))
        notes.append(
            f"scored window is {window_s:.1f} s but t_max is {t_max_s:.1f} s — "
            f"no policy was tested against its budget here; run with "
            f"--eval-frames {need} or declare the window as the budget")

    excluded = route is None or t.reason is term.TerminationReason.TRACE_EXHAUSTED
    # The live reason wins when it is a BENCHMARK reason. Reconstructing the
    # ending from the trace cannot see a renderer that died mid-episode: the
    # frames simply stop, which is indistinguishable from running out of them.
    live = str(live_termination or "").strip()
    if live:
        try:
            live_reason = term.TerminationReason(live)
        except ValueError:
            live_reason = None
            notes.append(f"evaluator reported an unknown termination {live!r}")
        # The evaluator's live ending takes precedence. Recomputing
        # from stored positions can place a stationary hold one frame
        # differently.
        if live_reason is not None and live_reason is not t.reason:
            post_hoc = t.reason.value
            t = term.Termination(
                live_reason, t.frame,
                f"evaluator ended the episode: {live_reason.value} "
                f"(post-hoc trace read as {post_hoc})")
            notes.append(
                f"used evaluator termination {live_reason.value}; the stored "
                f"trace alone reconstructed {post_hoc}")
        if live_reason in (term.TerminationReason.INFRA_FAILURE,
                           term.TerminationReason.ENVELOPE_EXIT,
                           term.TerminationReason.TRACE_EXHAUSTED):
            excluded = True
        elif live_reason is not None and route is not None:
            excluded = False
    channels, penalty = _penalty_channels(
        infractions if excluded else route.infractions, run.drivable_known,
        stop_sign_checked=stop_sign_checked)
    # List the channels that were not measured, so that they do not read
    # as clean.
    skipped = _SKIPPED_CHANNELS
    # The stop-sign channel is skipped by default; remove it from the
    # skipped list when it was measured.
    if stop_sign_checked:
        skipped = tuple(k for k in skipped if k != "stop_infraction")
    return {
        "source": source,
        # metrics.json in the same directory carries an EPDMS `driving_score`
        # (EPDMS_no_EP x RC over the FULL run) — same word, different metric,
        # different window. Never put them in one column.
        "metric_family": "navsafe",
        # Excluded episodes are kept out of every denominator instead
        # of being scored 0.
        "status": "excluded" if excluded else "scored",
        "scenario": dict(scenario_meta or {}),
        # Which scoring rules ran. Omitted when the defaults did, so a
        # metrics file that carries this key is saying "this event type was
        # NOT scored like the rest of the benchmark" -- which anyone
        # comparing it against another event type has to know.
        **({} if rules.is_default else {"scenario_rules": rules.to_dict()}),
        "frames": {
            "total": len(frames),
            "scored": len(scored_frames),
            "warmup_excluded": warmup_frames,
            "scored_window_s": round(window_s, 2),
            "t_max_s": round(t_max_s, 2) if time_limited else None,
        },
        "termination": {
            "reason": t.reason.value,
            "frame": t.frame,
            "detail": t.detail,
            "policy_attributed": t.reason.policy_attributed,
            "collision_count": run.collision_count,
        },
        # The four NavSafe (Bench2Drive-vocabulary) metrics.
        "metrics": {
            "driving_score": None if excluded else round(route.driving_score(), 3),
            "success": None if excluded else bool(route.success()),
            "efficiency_pct": None if eff is None else round(eff, 2),
            "comfort": None if comfort is None else round(comfort, 4),
        },
        "driving_score_breakdown": {
            "route_completion_pct": round(completion, 2),
            "penalty": round(penalty, 6),
            "formula": "driving_score = route_completion_pct * penalty",
            "channels": channels,
            "penalty_channels_evaluated": len(channels),
            "penalty_channels_skipped": skipped,
            # route_dev / vehicle_blocked / route_timeout TERMINATE a route
            # instead of multiplying the penalty, and min_speed is disabled
            # upstream — but they still fail success(), so a run can show
            # penalty 1.0 and success false with nothing else to explain it.
            "not_in_penalty": {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in (infractions if excluded else route.infractions).items()
                if k in navsafe_metrics.NON_PENALTY_KEYS
                or k == navsafe_metrics.MIN_SPEED_KEY} or None,
        },
        "notes": notes,
    }
