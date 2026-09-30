"""NavSafe's episode termination taxonomy.

Neither reference vocabulary fits a benchmark of ~10 s reconstructed events,
so NavSafe defines its own, using both as references:

* **MetaDrive** (``navsafe.scenario.constants.TerminationState``) mixes crash
  *types* into the termination enum (``CRASH_VEHICLE`` …), so "why the episode
  ended" and "what the ego hit" are one value — and it has no notion of fault
  and its ``SUCCESS`` is a destination radius, not a rubric goal.
* **Bench2Drive / CARLA LB 2.0** splits terminators from infractions but sizes
  every threshold for km-scale routes, and terminating contacts are still
  fault-blind (every collision is the ego's problem).

NavSafe's rules, in order of importance:

1. **One reason per episode.**  Termination answers "why did stepping stop";
   everything countable (collisions, red lights, off-lane meters) lives in the
   trace and is scored as infractions.  The reason is a fact about the
   episode, not a verdict about the policy.
2. **Fault splits the contact state.**  Under log replay a non-reactive
   follower can drive into a correctly-behaving ego; ending the episode is
   right (the world is no longer meaningful), but the *reason* must record
   that the ego was not at fault, or scoring inherits the replay's physics.
3. **Driving the wrong way ends the episode, and it is the ego's.**  Past
   ``DDC_FULL_VIOLATION_M`` of travel against the local traffic direction the
   episode is over: an ego committed to an opposing lane is not going to
   produce a meaningful remainder, and every metre it drives out there is
   scored against a route it is no longer on.  Unlike leaving the drivable
   surface this can happen entirely on-road, so it is its own reason.
4. **Budget expiry and deadlock are different failures.**  Still driving at
   ``t_max`` (too slow / too cautious) is not the same behaviour as never
   moving (frozen); collapsing them, as a bare step-cap does, hides the
   distinction every over-conservatism analysis needs.  Neither is an
   *infraction*: a deadlock is named by its reason and priced by the route
   completion it failed to reach, and adding Bench2Drive's ``vehicle_blocked``
   on top was one event wearing two names (see ``_REASON_TO_INFRACTION``).

5. **Some leaves need different rules, and the scenario says which.**  A
   work-zone bypass legitimately drives into the opposing lane, and a sideswipe
   scenario is not exercised at all by an ego that never changes lane.  Those
   are properties of the taxonomy leaf, not of the policy, so they are read
   from the bundle's ``scenario_meta`` and applied through
   :mod:`navsafe.benchmark.scenario_rules` rather than being hardcoded here.
   Everything not named there keeps the default rules exactly.

Scale defaults (declared, not inherited): episodes are ~10 s events with a
per-family time budget ``t_max`` (``rubric/calibrate_tmax.py``: 0.95-quantile
human completion time × 1.3, floor 10 s); the deadlock detector is
speed < 0.1 m/s sustained 5 s — half the budget floor, so a freeze cannot
silently consume an episode the way a 60 s CARLA blocked-test never firing
inside a 10 s window would. Frames on which the evaluator reports a red
signal ahead on the ego's lane (``signal_hold``) do not count toward that
hold: waiting at a light is not a freeze (2026-08-23; see ``classify``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Sequence

from navsafe.benchmark.scoring.metrics import RouteResult

if TYPE_CHECKING:      # import at call time below; this module must stay light
    from navsafe.benchmark.scenario_rules import ScenarioRules

# Deadlock detector: speed floor shared with Bench2Drive, duration scaled to
# the episode (half the 10 s MIN_BUDGET_S floor of calibrate_tmax).
DEADLOCK_SPEED_MS = 0.1
DEADLOCK_HOLD_S = 5.0
# The signal / stop-line exemptions to the 5 s rule are BOUNDED. They exist so
# a lawful wait is not convicted as a freeze — but under the clock-free
# protocol (route_time_limit_s=0, no frame window) an exemption with no bound
# makes an episode immortal: on 0e272e003af65a71 the baseline parked at a stop
# line behind end-of-log frozen cross traffic and stepped to frame 11,330+
# (three launch attempts, ~10 GPU-hours) with the deadlock counter reset every
# frame by the stop-line zone. No lawful hold in a ~20-30 s scenario
# approaches two minutes; a vehicle stationary that long is not waiting, it is
# parked, whatever geometry it is parked on.
STATIONARY_CEILING_S = 120.0

# NavSim's DAC is intentionally strict: one ego-box corner outside the union
# of lane polygons makes that frame score zero.  That is appropriate for the
# metric, but it is too brittle as an immediate episode terminator at acute
# connectors and lane-union seams.  Keep scoring every such frame, while only
# declaring that the vehicle has *left* the drivable surface after the signal
# persists for half a second.  At 10 Hz this permits four boundary samples and
# terminates on the fifth; a genuine departure is still stopped promptly.
OFF_DRIVABLE_HOLD_S = 0.5


class TerminationReason(str, Enum):
    """Why a NavSafe episode stopped stepping.  Exactly one per episode."""

    # -- the episode ran to a conclusion --------------------------------
    GOAL_REACHED = "goal_reached"          # rubric goal region entered within t_max
    BUDGET_EXPIRED = "budget_expired"      # t_max elapsed, ego still active
    DEADLOCK = "deadlock"                  # < 0.1 m/s sustained 5 s
    #: The objective was to HOLD, and the ego held for the whole window (V-1
    #: red light). A completion, exactly like GOAL_REACHED: on those leaves the
    #: goal region sits past the line the ego must not cross, so the episode
    #: succeeds by not reaching it and the clock running out is the success
    #: condition rather than BUDGET_EXPIRED's failure.
    HOLD_SATISFIED = "hold_satisfied"
    #: The ego crossed the line it had to hold behind. Ends the episode: past
    #: the line the hold is gone and the route it is now driving was never the
    #: objective, so further frames would score something nobody asked for. Adds
    #: NO infraction of its own -- `red_light` already prices the crossing from
    #: the evaluator's TL column, the same division of labour CONTACT_AT_FAULT
    #: has with `collisions_*`.
    RED_LIGHT_RUN = "red_light_run"
    #: The ego entered a turn the scenario prohibits (V-8). The mirror of
    #: RED_LIGHT_RUN: there a plate is not what forbids the manoeuvre, a signal
    #: is, but the shape is identical -- the objective is to NOT enter, so
    #: entering ends the episode and everything after it belongs to a route
    #: nobody asked for. Adds no infraction of its own; `illegal_turn` prices
    #: the entry from the trace, the division of labour RED_LIGHT_RUN has with
    #: `red_light` and CONTACT_AT_FAULT has with `collisions_*`.
    ILLEGAL_TURN = "illegal_turn"

    # -- the world stopped being drivable/meaningful --------------------
    CONTACT_AT_FAULT = "contact_at_fault"          # ego-caused contact
    CONTACT_NOT_AT_FAULT = "contact_not_at_fault"  # e.g. replayed follower
    OFF_DRIVABLE = "off_drivable"          # left the certified drivable surface
    WRONG_WAY = "wrong_way"                # drove against the traffic direction
                                           #   past the full-violation distance

    ENVELOPE_EXIT = "envelope_exit"
    INFRA_FAILURE = "infra_failure"        # simulator/renderer error
    TRACE_EXHAUSTED = "trace_exhausted"    # frames ran out before the ceiling:
                                           #   a harness cap, not an outcome

    @property
    def is_no_event_fallback(self) -> bool:
        """True for the reasons `classify` returns when NOTHING ended the run.

        `classify` is total -- it must name a reason for any trace -- so it ends
        with fallbacks that mean "no event stopped this". Live, a fallback is
        only an ending once the budget is genuinely gone: TRACE_EXHAUSTED
        mid-episode just means frames are still arriving.

        This lives on the enum because `LiveMonitor.update` used to hardcode the
        single name it knew about, so adding a second fallback ended every
        episode on its first scored frame. Asking the taxonomy makes the next
        one safe by construction.

        HOLD_SATISFIED is the second one, and it is exactly the failure that
        rule was written for: mid-episode "held for the whole window" only means
        "has not crossed yet", and returning it from `update` would end a V-1
        episode on its first scored frame with a perfect score.
        """
        return self in (TerminationReason.TRACE_EXHAUSTED,
                        TerminationReason.HOLD_SATISFIED)

    @property
    def policy_attributed(self) -> bool:
        """Whether the ending is chargeable to the policy.

        INFRA_FAILURE is a benchmark ending: the episode is reported `—` /
        rerun per the §8.6 two-cause rule, never folded into a mean as 0.
        CONTACT_NOT_AT_FAULT ends the episode but the contact is not the ego's
        infraction; what *is* chargeable is whatever the trace recorded up to
        that frame.
        """
        return self in (TerminationReason.GOAL_REACHED,
                        TerminationReason.HOLD_SATISFIED,
                        TerminationReason.RED_LIGHT_RUN,
                        TerminationReason.ILLEGAL_TURN,
                        TerminationReason.BUDGET_EXPIRED,
                        TerminationReason.DEADLOCK,
                        TerminationReason.CONTACT_AT_FAULT,
                        TerminationReason.OFF_DRIVABLE,
                        TerminationReason.WRONG_WAY)
        # TRACE_EXHAUSTED joins INFRA_FAILURE: the run stopped because the
        # harness stopped stepping, which says nothing about the policy and must
        # not be averaged in as a completed episode.


@dataclass(frozen=True)
class Termination:
    reason: TerminationReason
    frame: int                 # frame index at which the episode ended
    detail: str = ""           # free-text fact ("hit agent a17 rear_end", ...)


# Tie-break order when two events land on the SAME frame: hard endings mask
# soft ones. A contact is the most specific thing that can be said about a
# frame, and leaving the road or the lane direction outranks a goal reached on
# the same frame as either — an episode that ended up somewhere it should not be
# is not a clean completion.
_SAME_FRAME_PRECEDENCE = (
    TerminationReason.CONTACT_AT_FAULT,
    TerminationReason.CONTACT_NOT_AT_FAULT,
    TerminationReason.OFF_DRIVABLE,
    TerminationReason.WRONG_WAY,
    # A crossing outranks both endings that could share its frame: it masks
    # GOAL_REACHED for the same reason OFF_DRIVABLE does (an episode that ended
    # up where it must not be is not a clean completion), and it masks
    # HOLD_SATISFIED because a hold broken on the last frame was still broken.
    TerminationReason.RED_LIGHT_RUN,
    TerminationReason.ILLEGAL_TURN,
    TerminationReason.GOAL_REACHED,
    TerminationReason.DEADLOCK,
    TerminationReason.HOLD_SATISFIED,
)


def classify(frames: Sequence[Mapping], *,
             goal_reached: bool,
             t_max_s: float,
             dt: float = 0.1,
             goal_frame: int | None = None,
             infra_error: str = "",
             rules: "ScenarioRules | None" = None) -> Termination:
    """Derive the termination reason from a trace (pure function, no sim state).

    **Earliest event wins.**  Termination answers "why did stepping stop", so
    the event that would have stopped it first is the reason — a goal reached
    at 4 s is not undone by a contact at 9 s, and a deadlock at 3 s is not
    relabelled by a wrong-way exit at 8 s.  Two events on the same frame are
    broken by ``_SAME_FRAME_PRECEDENCE``.

    ``goal_frame`` is when the goal region was entered.  Callers that only know
    *whether* the goal was reached pass ``goal_reached=True`` alone; the goal
    then counts as having happened at the last scored frame, so any earlier
    event still masks it (the conservative reading — an episode that hit
    something on the way cannot be called a clean completion).

    ``rules`` are the taxonomy leaf's departures from the defaults
    (:mod:`navsafe.benchmark.scenario_rules`); ``None`` means the defaults, so
    every existing caller keeps its behaviour exactly. A leaf can suppress a
    termination reason (I-2 does not end on wrong-way) or add a precondition to
    the goal (C-3 requires a lane change).
    """
    from navsafe.benchmark import scenario_rules as _rules_mod

    rules = rules or _rules_mod.DEFAULT_RULES
    if infra_error:
        return Termination(TerminationReason.INFRA_FAILURE,
                           len(frames) - 1 if frames else 0, infra_error)

    scored = [(i, f) for i, f in enumerate(frames) if f.get("phase") == "scored"]
    if not scored:
        return Termination(TerminationReason.INFRA_FAILURE, 0,
                           "trace has no scored frames")

    # (frame, reason, detail) for every ending the trace supports; the earliest
    # is the episode's, so each rule reports only its FIRST occurrence.
    events: list[tuple[int, TerminationReason, str]] = []

    def first(reason: TerminationReason, i: int, detail: str) -> None:
        # A leaf-disabled reason is not merely unrecorded, it must not end the
        # episode: I-2's work-zone bypass has to be allowed to keep driving
        # after it crosses into the opposing lane, or the terminator decides the
        # outcome before the infraction filter ever sees it.
        if not _rules_mod.allows_termination(reason, rules):
            return
        if not any(e[1] is reason for e in events):
            events.append((i, reason, detail))

    hold = 0
    stationary = 0
    off_drivable_hold = 0
    # Exit-region bookkeeping (V-8). `cleared_intersection` records that the ego
    # actually MET the event; `entered_no_go` that it broke the hold. Both are
    # read off the same region columns hold_violation uses, so a trace without
    # them leaves both False and the gate can never fire -- not checked stays
    # not checked.
    cleared_intersection = False
    entered_no_go = False
    for i, f in scored:
        # DDC (driving-direction compliance) is the EPDMS subscore graded by
        # metres driven against the local traffic direction: 1.0 under 2 m, 0.5
        # from 2-6 m, 0.0 past 6 m. Only the 0.0 case ends the episode — the
        # half score is a mid-turn brush against an opposing lane, which is a
        # penalty, not a terminated route. `driving_direction_ok` is False only
        # where the frame carried a DDC to judge, so a run with no lane graph
        # never fires it (the same rule `on_drivable` follows for DAC).
        if not f.get("driving_direction_ok", True):
            first(TerminationReason.WRONG_WAY, i,
                  f"drove against the traffic direction at frame {i}")
        for c in f.get("contacts", []) or []:
            # ANY contact ends the episode — the strict protocol ruling
            # (2026-09-01): a collision is a collision. Fault still splits
            # the reason, so a replayed follower rear-ending a correct ego
            # ends the run as CONTACT_NOT_AT_FAULT (not the ego's
            # infraction, scored on the route driven up to that frame)
            # rather than as the ego's failure.
            reason = (TerminationReason.CONTACT_AT_FAULT if c.get("at_fault")
                      else TerminationReason.CONTACT_NOT_AT_FAULT)
            first(reason, i,
                  f"{c.get('kind') or 'contact'} with {c.get('agent_id') or '?'}")
        if f.get("in_intersection"):
            cleared_intersection = True
        if f.get("in_exit_lane"):
            entered_no_go = True
        if not f.get("on_drivable", True):
            off_drivable_hold += 1
            if off_drivable_hold * dt >= OFF_DRIVABLE_HOLD_S - 1e-9:
                # An exit-region leaf (V-8) is DONE once the ego has cleared the
                # junction without entering the region it had to stay out of.
                # Compliance there means leaving the logged route -- the route
                # IS the prohibited turn -- and a route-departing ego runs out
                # of certified surface within seconds, so charging it
                # off_drivable would fail the very behaviour the leaf asks for.
                # The gate is deliberately narrow: it needs the event to have
                # been MET (the ego was inside the junction) and the hold to
                # have been KEPT (it never entered the prohibited lane). An ego
                # that drove off the road before reaching the junction, or one
                # that entered the lane, gets the ordinary reason.
                if (rules.exit_region_ends_episode
                        and cleared_intersection and not entered_no_go):
                    first(TerminationReason.HOLD_SATISFIED, i,
                          "cleared the junction without entering the "
                          f"prohibited lane, then left the mapped surface at "
                          f"frame {i} -- the manoeuvre was complete")
                else:
                    first(
                        TerminationReason.OFF_DRIVABLE, i,
                        f"off drivable for {OFF_DRIVABLE_HOLD_S:.1f} s "
                        f"through frame {i}")
        else:
            off_drivable_hold = 0
        # A vehicle waiting at a red signal is not deadlocked. ``signal_hold``
        # is the evaluator's per-frame state fact "a red light is ahead on
        # the ego's lane" (EPDMSLiveScorer._signal_hold_live; TL_HOLD in
        # driving_score_summary.csv), read from the same logged light states
        # the red_light penalty is measured from. The 5 s rule was calibrated
        # for ~10 s signal-free events -- Bench2Drive's own blocked test is
        # 180 s precisely because of traffic lights -- and on navhard421 a
        # stop-compliant ego waits 13 s on 0f622aef14545f59 (connector red
        # for frames 0-163). Absent or false, the rule is unchanged. A policy
        # frozen near a red light still ends on budget_expired.
        # A stop-sign episode has no signal to set `signal_hold`, so the same
        # exemption is taken from geometry: within STOP_LINE_ZONE_M of the line
        # a stationary ego is obeying the sign, not frozen. Bounded to the zone
        # so a policy that stops 40 m short is still caught.
        if f.get("signal_hold", False) or (
                rules.stop_sign_dwell_s is not None
                and _rules_mod.near_stop_line(f)):
            hold = 0
        else:
            hold = hold + 1 if f.get("ego_speed", 0.0) < DEADLOCK_SPEED_MS else 0
        # The exemptions above are bounded (STATIONARY_CEILING_S): this
        # counter ignores them, so a hold no lawful wait can explain still
        # ends the episode instead of stepping forever under the clock-free
        # protocol.
        stationary = (stationary + 1
                      if f.get("ego_speed", 0.0) < DEADLOCK_SPEED_MS else 0)
        if stationary * dt >= STATIONARY_CEILING_S - dt:
            first(TerminationReason.DEADLOCK, i,
                  f"stationary {STATIONARY_CEILING_S:.0f} s — the signal/"
                  "stop-line hold exemption is bounded")
        # One frame of tolerance, and it is a measurement artefact, not slack.
        # LiveMonitor is fed the env's REPORTED speed and ends the episode on
        # the frame that completes the hold; `from_eval` re-derives speed by
        # finite difference of the stored positions, where the sample at which
        # motion ceases still carries the last moving step. A hold measured
        # from positions is therefore systematically ONE sample shorter than
        # the same hold measured live, and on a hard `>=` the post-hoc pass
        # lands one frame short of its own threshold: a live DEADLOCK came back
        # `trace_exhausted`, an ending the taxonomy charges to nobody, for the
        # one event that is squarely the policy's. Measured on
        # 0bcae698fd905226/pdm_closed: live deadlock at frame 230, 49
        # reconstructed stationary frames against a 50-frame rule.
        if hold * dt >= DEADLOCK_HOLD_S - dt:
            first(TerminationReason.DEADLOCK, i,
                  f"< {DEADLOCK_SPEED_MS} m/s for {DEADLOCK_HOLD_S:.0f} s")

    goal_detail = "goal region entered"
    if (goal_reached and rules.require_lane_change
            and _rules_mod.has_lane_data(frames)):
        # C-3: reaching the goal region while never leaving the lane means the
        # scenario's conflict was never entered. Withhold the completion rather
        # than end the episode -- it falls through to whatever the trace does
        # support (usually budget_expired), which is the honest reading of
        # "declined the manoeuvre".
        changes = _rules_mod.lane_change_frames(frames)
        goal_frame_eff = (scored[-1][0] if goal_frame is None else int(goal_frame))
        if not any(c <= goal_frame_eff for c in changes):
            goal_reached = False
        else:
            goal_detail = (f"goal region entered after a lane change at frame "
                           f"{min(c for c in changes if c <= goal_frame_eff)}")

    if goal_reached:
        # Through `first()`, not appended directly: a leaf that disables the
        # goal (V-1 holds behind a line the goal region sits past) needs the
        # suppression to actually take, and a directly-appended event skips
        # `allows_termination` entirely.
        first(TerminationReason.GOAL_REACHED,
              scored[-1][0] if goal_frame is None else int(goal_frame),
              goal_detail)

    # A hold leaf has two endings of its own and no fallthrough: either the ego
    # crossed the line it had to stay behind, or it held to the end of the
    # window. Neither can be reached by a leaf without `hold_region`.
    if rules.hold_region:
        violation, detail = _rules_mod.hold_violation(frames, rules)
        contradicted, _ = _rules_mod.hold_contradicted(frames, rules)
        if violation is not None:
            # Which ending a broken hold is depends on WHAT was held: V-1 holds
            # behind a stop line and breaking it is a red-light run; V-8 holds
            # out of a prohibited turn lane and breaking it is an illegal turn.
            # Both end the episode identically and are priced by their own
            # infraction channel; only the name differs.
            first(TerminationReason.ILLEGAL_TURN
                  if rules.hold_region == "no_turn_lane"
                  else TerminationReason.RED_LIGHT_RUN,
                  violation, detail)
        elif not contradicted and _rules_mod.hold_measurable(frames, rules):
            first(TerminationReason.HOLD_SATISFIED, scored[-1][0],
                  f"held behind the {rules.hold_region.replace('_', ' ')} for "
                  f"the whole {len(scored) * dt:.1f} s window")
        # else: nothing witnessed the hold either way, so the episode falls
        # through to budget_expired / trace_exhausted. Not a completion.

    if events:
        i, reason, detail = min(
            events, key=lambda e: (e[0], _SAME_FRAME_PRECEDENCE.index(e[1])))
        return Termination(reason, i, detail)

    # Nothing ended it: the ego was still driving when the trace ran out. Two
    # very different situations, and conflating them under `budget_expired` is
    # what made that reason the commonest outcome in the corpus while saying
    # nothing about any policy.
    #
    #   elapsed >= t_max  the ego really used its whole budget (the safety
    #                     ceiling) without finishing the route. A verdict, and
    #                     the policy's.
    #   elapsed <  t_max  the frames ran out first -- a harness cap, an
    #                     interrupted run, a bundle shorter than the ceiling.
    #                     Not an outcome: reported as its own reason and
    #                     excluded, so it cannot be read as "too slow".
    elapsed = len(scored) * dt
    if elapsed >= t_max_s - 0.5 * dt:
        return Termination(TerminationReason.BUDGET_EXPIRED, scored[-1][0],
                           f"t_max {t_max_s:.1f} s elapsed")
    return Termination(
        TerminationReason.TRACE_EXHAUSTED, scored[-1][0],
        f"trace ended at {elapsed:.1f} s, short of the {t_max_s:.1f} s ceiling "
        f"— the harness stopped stepping, the ego was still driving")


# Bench2Drive vocabulary each reason maps onto when a RouteResult is built.
# CONTACT_* adds no infraction here: the contacts themselves are already
# counted (with fault respected) by ``infractions_from_trace``.
_REASON_TO_INFRACTION: dict[TerminationReason, str | None] = {
    TerminationReason.GOAL_REACHED: None,
    # Holding is the objective, so it adds nothing; crossing is already priced
    # by `red_light` from the evaluator's TL column, and adding a second count
    # here would be the deadlock/`vehicle_blocked` mistake again (716f1f3) --
    # one event wearing two names.
    TerminationReason.HOLD_SATISFIED: None,
    TerminationReason.RED_LIGHT_RUN: None,
    # Same shape as RED_LIGHT_RUN, and None for a stronger reason. There is no
    # evaluator column for a prohibitory plate, so nothing else is counting the
    # entry -- but the entry is ALREADY priced, and priced proportionally, by
    # the hold fraction that becomes route completion on this leaf (see
    # scoring/from_run.py): an ego that enters the turn lane early completes
    # almost none of the window and its DS collapses accordingly. Adding a
    # multiplicative infraction on top would charge one event twice, and
    # inventing a Bench2Drive coefficient for a channel Bench2Drive never
    # defined is exactly what WRONG_WAY's comment refuses to do. success() is
    # failed regardless, because `completed` is False.
    TerminationReason.ILLEGAL_TURN: None,
    TerminationReason.BUDGET_EXPIRED: "route_timeout",
    # Deadlock is a REASON, not an infraction (2026-08-28). It used to add
    # Bench2Drive's `vehicle_blocked`, which double-counted the same fact: the
    # episode already fails success() because `completed` is False, and
    # `vehicle_blocked` is a NON_PENALTY_KEY so it never entered the DS product
    # either. All it did was put a second name on one event, in a taxonomy whose
    # first rule is that termination answers "why did stepping stop" while
    # everything countable lives in the trace. A frozen ego is now reported by
    # its reason and its route completion alone.
    TerminationReason.DEADLOCK: None,
    TerminationReason.CONTACT_AT_FAULT: None,
    TerminationReason.CONTACT_NOT_AT_FAULT: None,
    TerminationReason.OFF_DRIVABLE: "route_dev",
    # Wrong-way is the ego's, and it TERMINATES rather than multiplying: like
    # route_dev it is a non-penalty key, so it fails success() without pretending
    # Bench2Drive has a coefficient for a channel it never defined.
    TerminationReason.WRONG_WAY: "wrong_way",
    TerminationReason.ENVELOPE_EXIT: None,     # retired; kept for old artifacts
    TerminationReason.INFRA_FAILURE: None,
    # No infraction: the harness stopped, the ego did nothing wrong.
    TerminationReason.TRACE_EXHAUSTED: None,
}


def to_route_result(route_id: str,
                    completion_pct: float,
                    termination: Termination,
                    infractions: Mapping[str, float],
                    rules: "ScenarioRules | None" = None) -> RouteResult | None:
    """Fold a NavSafe termination into a Bench2Drive-vocabulary RouteResult.

    Returns None for benchmark-attributed endings (INFRA_FAILURE, and the
    retired ENVELOPE_EXIT for a stored trace that still carries it): those
    episodes are reported `—` with their reason and are excluded from DS/SR
    denominators per the §8.6 two-cause rule — returning a zero-score
    RouteResult here would fold a benchmark failure into the policy's mean.
    """
    if not termination.reason.policy_attributed and termination.reason in (
            TerminationReason.ENVELOPE_EXIT, TerminationReason.INFRA_FAILURE):
        return None

    from navsafe.benchmark import scenario_rules as _rules_mod

    rules = rules or _rules_mod.DEFAULT_RULES
    counts = dict(infractions)
    extra = _REASON_TO_INFRACTION[termination.reason]
    if extra is not None:
        counts[extra] = counts.get(extra, 0) + 1
    # Applied last, so a leaf-disabled key is dropped whether it came from the
    # trace or from the termination reason above.
    counts = _rules_mod.filter_infractions(counts, rules)
    return RouteResult(route_id=route_id,
                       completion_pct=float(completion_pct),
                       completed=termination.reason in (
                           TerminationReason.GOAL_REACHED,
                           TerminationReason.HOLD_SATISFIED),
                       infractions=counts)


# ---------------------------------------------------------------------------
# Live driver
# ---------------------------------------------------------------------------

class LiveMonitor:
    """Run :func:`classify` inside the eval loop instead of over a finished trace.

    Same rules, same thresholds, same precedence — the only difference is that
    frames arrive one at a time and the answer is wanted as soon as it exists.
    Everything the taxonomy can end an episode for (off-drivable, wrong-way,
    deadlock, contact) is computed by the loop already; what was missing was
    anything asking. A run that commits to an opposing lane at frame 62 and
    keeps stepping to 107 spends 45 frames scoring a route it is no longer
    driving, then reports the result.

    ``update`` returns the :class:`Termination` on the frame an ending first
    exists, and ``None`` while the episode is still live. ``BUDGET_EXPIRED``
    is never returned early: mid-episode it only means "the trace has not run
    out yet", and the frame cap is the caller's to enforce.
    """

    def __init__(self, log_path_xy, *, warmup: int = 0, dt: float = 0.1,
                 t_max_s: float = 0.0,
                 rules: "ScenarioRules | None" = None) -> None:
        import numpy as np

        from navsafe.benchmark import scenario_rules as _rules_mod

        self.warmup = int(warmup)
        self.dt = float(dt)
        self.t_max_s = float(t_max_s)
        #: The leaf's rule departures, so the live ending matches what the
        #: post-hoc pass will conclude from the same trace.
        self.rules = rules or _rules_mod.DEFAULT_RULES
        self._path = (np.asarray(log_path_xy, dtype=np.float64).reshape(-1, 2)
                      if log_path_xy is not None else np.empty((0, 2)))
        self._frames: list[dict] = []
        self.termination: Termination | None = None

    def deviation_m(self, ego_xy) -> float:
        """Planar distance to the logged path; ``nan`` without one.

        No longer a termination input — it is recorded as a diagnostic, which is
        all it ever was measured well enough to be.
        """
        import numpy as np

        if self._path.size == 0:
            return float("nan")
        return float(np.min(np.linalg.norm(
            self._path - np.asarray(ego_xy, dtype=np.float64)[:2], axis=1)))

    def update(self, *, ego_xy, ego_speed: float, on_drivable: bool = True,
               contacts=(), goal_reached: bool = False,
               driving_direction_ok: bool = True,
               signal_hold: bool = False,
               lane_id: str | None = None,
               lateral_offset_m: float | None = None,
               dist_to_stopline_m: float | None = None,
               signal_violation: bool | None = None) -> Termination | None:
        """Append one frame and re-classify. Returns the ending, or None.

        ``driving_direction_ok`` defaults True so a caller with no DDC to hand
        (no lane graph, or an older harness) cannot accidentally end every
        episode as wrong-way. ``signal_hold`` (a red light ahead on the ego's
        lane) defaults False, so a caller without the scorer's state fact
        keeps the plain deadlock rule.

        ``lane_id`` / ``lateral_offset_m`` are the columns the C-3 lane-change
        gate reads. They default to None, and a trace with none of them is
        treated as "not checked" rather than "no lane change" -- otherwise a
        harness that does not supply them would report every C-3 episode as a
        declined manoeuvre.
        """
        dev = self.deviation_m(ego_xy)
        self._frames.append({
            "phase": "warmup" if len(self._frames) < self.warmup else "scored",
            "ego_speed": float(ego_speed),
            "on_drivable": bool(on_drivable),
            "driving_direction_ok": bool(driving_direction_ok),
            "signal_hold": bool(signal_hold),
            "contacts": list(contacts),
            "ego_dev_m": dev,
            "lane_id": lane_id,
            "lateral_offset_m": (None if lateral_offset_m is None
                                 else float(lateral_offset_m)),
            # V-1's hold reads this. None keeps the column absent, which
            # `hold_measurable` treats as "cannot witness a hold" rather than as
            # a line at the origin.
            "dist_to_stopline_m": (None if dist_to_stopline_m is None
                                   else float(dist_to_stopline_m)),
            # The evaluator's traffic-light verdict for this frame, written into
            # the SAME column `trace/from_eval.py` fills from the stored TL
            # subscore, so live and post-hoc convict on one rule and one frame.
            # Three states, not two: None means the caller has no traffic-light
            # verdict at all, and only THAT leaves the stop-line geometry as the
            # witness. A verdict of "compliant" has to reach the column as
            # "unknown", or a scorer that says the ego did not run the light
            # would be silently overruled by geometry two frames earlier.
            "signal_state": ("" if signal_violation is None
                             else "red" if signal_violation else "unknown"),
            "ego_x": float(ego_xy[0]),
            "ego_y": float(ego_xy[1]),
        })
        if self.termination is not None:
            return self.termination
        # Nothing to classify until the policy is actually driving: with only
        # warm-up frames `classify` has no scored frames and reports
        # INFRA_FAILURE, which live means "not started yet", not "broken".
        if not any(f["phase"] == "scored" for f in self._frames):
            return None
        t = classify(self._frames, goal_reached=goal_reached,
                     t_max_s=self.t_max_s, dt=self.dt, rules=self.rules)
        # A no-event fallback means "nothing has ended it YET" while frames are
        # still arriving. BUDGET_EXPIRED is NOT one: reaching the ceiling is a
        # real ending. Asking the enum rather than naming reasons here is what
        # keeps a newly added fallback from ending every episode on frame 1.
        if t.reason.is_no_event_fallback:
            return None
        self.termination = t
        return t

    def final(self, *, goal_reached: bool = False) -> Termination:
        """The episode's ending once stepping has stopped for any reason."""
        if self.termination is not None:
            return self.termination
        return classify(self._frames, goal_reached=goal_reached,
                        t_max_s=self.t_max_s, dt=self.dt, rules=self.rules)
