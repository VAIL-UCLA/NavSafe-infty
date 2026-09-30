# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The NavSafe scoring chain, as one pure function over a stored run.

``scripts/tools/navsafe_from_eval.py`` and ``eval_py123d.py`` both need exactly
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

# Taxonomy §0.1: t_max is 0.95-quantile human completion time x 1.3, floor 10 s.
# One scenario has one human drive, so its own duration is the quantile.
T_MAX_SLACK = 1.3
T_MAX_FLOOR_S = 10.0

#: Hard ceiling on an episode, in seconds. An episode is otherwise INDEFINITE:
#: it ends when something in the taxonomy ends it (the goal, a contact, leaving
#: the drivable surface, a deadlock), not when a frame count runs out. Something
#: still has to stop a policy that drives in circles on drivable road forever,
#: and that is this.
#:
#: 60 s is the user-set budget. It exceeds every published bundle's 20 s window
#: on purpose: reactive actors keep driving past the log (see
#: traffic/semi_reactive.ROUTE_EXTENSION_M) and nothing stops the ego for being
#: far from the logged path, so it can keep going, and `budget_expired`
#: now means "used a full minute and still did not finish" instead of "the
#: bundle ran out", which is what made it the commonest reason in the corpus.
SAFETY_CEILING_S = 60.0

# ...but a budget is only real if the scenario can supply it. A published
# bundle is a 20 s window cut around its event and the logged human drives the
# whole of it, so `1.3 x human_s` asks for ~26 s of a world that ends at 20 —
# unreachable on EVERY token, not an edge case. The episode then ran out of
# frames at about human time and was reported `budget_expired ... short of
# t_max`, which reads as a slow policy when nothing was ever tested against a
# budget (05d0a1a763fc5334/pdm_closed: 91.3 % of route, zero infractions,
# 18.0 s of a promised 23.3 s).
#
# So the budget is capped at the drivable window: the policy gets the whole
# scenario after the hand-off, and `budget_expired` means "used all the time
# that exists and did not finish" — a verdict rather than an artifact.
# `window_shorter_than_t_max` goes back to flagging real truncation (a killed
# run, a crashed sim) instead of firing on every healthy episode.
CAP_T_MAX_AT_SCENARIO = True

MIN_SCORABLE_FRAMES = 10



#: The penalty channels this path can evaluate, and where each comes from.
#:
#: Bench2Drive's score_penalty has eight channels; ``infractions_from_trace``
#: covers the collision family, and ``outside_route_lanes`` comes from the
#: off-route distance when a map exists.
#:
#: ``red_light`` is NOT among them, by decision rather than by omission. It was
#: measured from the evaluator's per-frame ``TL`` column, which the EPDMS
#: scorer computes live against the lane graph and ``dynamic_map_states``. The
#: logged light states that column reads are not reliable enough to price a
#: 0.70 multiplier off — a wrong `red` turns a clean episode into a 70 % run,
#: and there is no way to tell that case apart from a genuine violation after
#: the fact. An unreliable source scored anyway is worse than an unmeasured
#: one, so the channel joins the SKIPPED list for EVERY run and no
#: ``red_light`` infraction is ever raised here. That also keeps it out of
#: ``RouteResult.success()``, which fails on any infraction it can see.
#:
#: This does NOT disable the authored hold leaves. ``RED_LIGHT_RUN`` comes from
#: ``scenario_rules.hold_violation`` against a per-bundle ``hold_region`` (V-1),
#: not from the logged column, so crossing an authored stop line still ends the
#: episode. Only the log-derived penalty is gone.
#:
#: ``stop_infraction`` still has no source (``predicates.stop_before`` needs
#: stop-line regions this data does not carry), and ``scenario_timeouts`` /
#: ``yield_emergency_vehicle_infractions`` have no implementation at all. Those
#: are reported as SKIPPED rather than as zero counts: a zero would read as
#: "clean" when nothing was checked. ``red_light`` now reads the same way, and
#: means the same thing — nobody checked.
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
        # Top-level, not inside scenario_meta: it is a property of the scenario
        # as authored, and V-1's hold rules only apply where it is present
        # (scenario_rules.rules_for_scenario). The environment channel is the
        # same one the scenario builder honours (signal_override.ENV_VAR) so a
        # sidecar-authored red — a read-only bundle whose manifest cannot be
        # patched — scores under the identical light the planner drove under;
        # a manifest declaration still wins because it travels with the data.
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
    # Beyond that, whether a short trace is scorable depends on WHY it is short,
    # which is not known until the ending is classified below. A stub that just
    # stops is unscorable; a stub that ends on a decisive, policy-attributable
    # event is the whole result. V-1 makes this common rather than exotic: an
    # ego that crosses the line 0.9 s after the hand-off produces 9 scored
    # frames, and refusing them would drop the clearest possible failure out of
    # the denominator instead of scoring it.
    short = len(frames) <= warmup_frames + MIN_SCORABLE_FRAMES

    # --- route progress (monotone) ---------------------------------------
    prog = route_mod.route_progress(run.route_xy, run.ego_xy)
    goal_frame = (prog.goal_frame
                  if prog.goal_frame is not None and prog.goal_frame >= warmup_frames
                  else None)
    goal_reached = goal_frame is not None
    # Fallback: honour the evaluator's own goal verdict when the recompute here
    # disagrees. Since both now call route.at_goal() over the same dense path,
    # they should not — so reaching this branch is NEWS, not routine, and it
    # says so in the notes. It stays for runs scored from artifacts written
    # before that unification, and for any case where the scorer's route source
    # (the Arrow scenario) is not the array the evaluator drove.
    #
    # goal_frame stays None: metrics.json records the reason, not the frame,
    # and classify() then dates the goal at the last scored frame, so any
    # earlier contact/off-drivable/deadlock event still masks it. Honouring the
    # report cannot turn a crash into a clean completion.
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

    # --- scenario rules --------------------------------------------------
    # The taxonomy leaf, from the bundle's own manifest, decides which
    # termination/success rules apply. Every leaf not named in
    # scenario_rules.LEAF_RULES gets the defaults, so this is a no-op for all
    # but I-2 (work-zone: wrong-way is the manoeuvre) and C-3 (sideswipe: the
    # goal requires the lane change the scenario is built around).
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
    # V-8 only: success requires that the ego actually MET the event. The leaf
    # succeeds by not entering the prohibited lane, so every trace in which
    # nothing happens -- a policy that froze at the hand-off, a clip that ran
    # out early -- reads as a perfect hold and would score 100. The generic
    # MIN_SCORABLE_FRAMES floor does not catch it: 10 frames is 1.0 s, and an
    # ego stationary for 1.5 s clears it.
    #
    # So the trace must show the ego inside the intersection at some point.
    # Reaching it and leaving without touching the turn lane is the compliant
    # manoeuvre; never reaching it means the trace cannot say whether the plate
    # was read, which is unscorable rather than a pass.
    # Any contact is already a decisive episode outcome.  It can happen on the
    # approach, before the ego reaches the intersection, and must remain in the
    # denominator through the ordinary fault-aware contact/route score rather
    # than be mislabeled as an unmet V-8 event.  The guard still rejects
    # uneventful holds (freeze, deadlock, trace exhaustion), which are the
    # false passes it was introduced to prevent.
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
        # On a hold leaf the route runs THROUGH the intersection the ego is
        # correctly not entering, so route completion measures the opposite of
        # compliance: a perfect hold completes ~0 % of it and DS
        # (completion x penalty) would report a compliant policy as a total
        # failure. The base becomes the fraction of the window the hold
        # survived. This is why the field means something different here, and
        # why the scenario_rules block below has to travel with the number.
        completion = scenario_rules.hold_fraction(
            frames,
            t.frame if t.reason in (term.TerminationReason.RED_LIGHT_RUN,
                                    term.TerminationReason.ILLEGAL_TURN) else None,
            # The budget, not the trace: a crossing ends the episode, so the
            # recorded window shrinks with the very failure being measured.
            # Under a clock-free protocol (t_max infinite) there is no budget
            # to read — the harness frame cap is what ended the run, so the
            # recorded window IS the window (a V-1 cell holds for exactly
            # NAVSAFE_EVAL_FRAMES and int(inf) was a crash, not a score).
            window_frames=(int(round(t_max_s / dt))
                           if math.isfinite(t_max_s)
                           else max(len(frames) - warmup_frames, 1)))

    # --- infractions -----------------------------------------------------
    infractions = navsafe_metrics.infractions_from_trace(
        scored_frames, at_fault_only=True)
    # A collision the per-frame record cannot PLACE still counts once -- but
    # only when the record genuinely cannot place it. An empty `infractions`
    # means one of two very different things: no COLL/COLL_AF columns at all
    # (an old artifact; the evaluator's count is the only evidence there is), or
    # columns that did place the contact and attributed it to someone else.
    # Testing `not infractions` alone conflated them, so a NOT-AT-FAULT contact
    # was charged 0.6x anyway -- the fault model overridden by its own fallback.
    # Seen on 05d0a1a763fc5334/drivor: a semi-reactive agent hit the ego at
    # frame 158, the episode correctly ended `contact_not_at_fault` with
    # policy_attributed=False, and the score still dropped 65.2 -> 39.1.
    placed_contacts = any(f.get("contacts") for f in frames)
    if run.collision_count and not infractions and not placed_contacts:
        infractions["collisions_vehicle"] = float(run.collision_count)
    # No `red_light` infraction is raised, whatever the TL column says: the
    # logged light states are not trustworthy enough to price a 0.70 penalty
    # (see _CHANNEL_SOURCES). The column is still written to the trace and is
    # still what `scenario_rules` reads for the authored V-1 hold, so nothing
    # below this line loses information — only the penalty is gone.
    # Stop sign: crossing the line without the dwell. Tri-state on purpose --
    # `None` means the trace could not answer (no approach to the line in the
    # window, or no stop-line column) and the channel stays SKIPPED, because a
    # zero here would read as "checked and clean" for an ego whose stop simply
    # happened before the window opened.
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
                # Distinguish "not measurable" from "not computed": a null
                # Efficiency beside a finished route otherwise reads as a bug.
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
        # The evaluator owns the live ending. Post-hoc finite differences can
        # reconstruct a stationary hold one frame differently, but must never
        # silently turn a live deadlock into trace_exhausted (or vice versa).
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
    # An unmeasured channel has to appear SOMEWHERE, or the file reads as
    # "clean" on something nobody checked. `red_light` is now permanently in
    # this list because it dropped out of _CHANNEL_SOURCES.
    skipped = _SKIPPED_CHANNELS
    # stop_infraction is skipped everywhere by default (nothing computes it), so
    # a scenario where the dwell WAS measured has to drop out of that list or
    # the file would report the same channel as both scored and unmeasured.
    if stop_sign_checked:
        skipped = tuple(k for k in skipped if k != "stop_infraction")
    return {
        "source": source,
        # metrics.json in the same directory carries an EPDMS `driving_score`
        # (EPDMS_no_EP x RC over the FULL run) — same word, different metric,
        # different window. Never put them in one column.
        "metric_family": "navsafe",
        # Excluded episodes are reported '—' and kept out of every denominator
        # (taxonomy §0.1 two-cause rule), never scored 0.
        "status": "excluded" if excluded else "scored",
        "scenario": dict(scenario_meta or {}),
        # Which scoring rules ran. Omitted when the defaults did, so a
        # metrics file that carries this key is saying "this leaf was
        # NOT scored like the rest of the benchmark" -- which anyone
        # comparing it against another leaf has to know.
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
