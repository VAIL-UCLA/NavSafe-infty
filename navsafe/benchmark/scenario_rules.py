# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Per-leaf overrides to the default termination / success rules.

Most of the 28 event types are scored by the same rules, and that default
is the point: a benchmark whose every event type has bespoke scoring cannot be
summed. This module holds the exceptions, and only the exceptions — an event type
absent from :data:`LEAF_RULES` gets :data:`DEFAULT_RULES`, which changes
nothing about how ``termination.classify`` and ``RouteResult.success`` already
behave.

The event type comes from the bundle's own ``manifest.json`` (``scenario_meta.
taxonomy_leaves``), which ``scoring/from_run.scenario_meta_for`` already reads
and records. Nothing new has to be plumbed to the scenario; the metadata was
being carried and not consulted.

Two event types need it today, both because the *default* rule contradicts what the
scenario is built to test:

**I-2 Work-Zone Incident** (``near_construction_zone_sign``). Bypassing a
construction zone means crossing into the opposing lane — that is the correct
manoeuvre, and often the only one. The default treats
``DDC_FULL_VIOLATION_M`` of travel against the local traffic direction as
``WRONG_WAY``: it ends the episode and fails success(). On this event type that
scores the solution as the failure, so ``wrong_way`` is disabled as both a
terminator and an infraction. Everything else — contacts, off-drivable, the
time budget — is unchanged, so an ego that bypasses the zone *and* hits
something is still charged for the contact.

**C-3 Sideswipe**, but only for two of its three precursors. The event type covers
two different mechanisms and they need different goals:

* ``changing_lane_with_lead`` / ``changing_lane_with_trail`` -- **the ego** is
  the one changing lane, into a lane already occupied ahead or behind. An ego
  that holds its lane and drives straight past never enters the conflict, yet
  the default goal (the rubric goal region, which sits ahead along the route)
  is satisfied by exactly that, so the event type would report success for a policy
  that declined the scenario. ``require_lane_change`` gates ``goal_reached`` on
  trace evidence that the ego actually changed lane.
* ``near_high_speed_vehicle`` -- a fast vehicle passes **alongside**. The ego is
  not required to change lane at all; holding the lane while a car overtakes is
  the correct behaviour, and demanding a lane change here would invert the
  test. These keep the default goal.

That split is not cosmetic. Of the 23 C-3 rows in the 417-scenario corpus, 22
are ``near_high_speed_vehicle`` and 1 is ``changing_lane_with_trail``, so
keying ``require_lane_change`` on the event type alone would apply the wrong goal to
22 of 23 scenarios. The rule is therefore keyed on the nuPlan scenario *type*
(:data:`LANE_CHANGE_REQUIRED_TYPES`), which the manifest carries beside the
event type.

Withholding the goal does not end the episode early: it simply is not a
completion, and the run falls through to ``BUDGET_EXPIRED`` / whatever else the
trace supports, which is the honest reading of "did not attempt the manoeuvre".

Detecting the lane change map-free
----------------------------------
``classify`` is a pure function of the trace, with no map access, and the trace
carries ``lane_id`` and a signed ``lateral_offset_m`` per frame. A *lane
change* has to be told apart from driving forward into the next lane along the
same corridor, which also changes ``lane_id``.

The discriminator is the offset sign. Crossing a lane boundary laterally means
leaving lane A towards one of its edges and entering lane B from the opposite
edge, so ``lateral_offset_m`` flips sign across the transition. Driving into a
successor lane happens at the centre of both, where the offset is near zero and
keeps its sign. :func:`lane_change_frames` reports transitions where the id
changes and the signed offset flips by at least
:data:`LANE_CHANGE_MIN_OFFSET_M` on each side, which no longitudinal handoff
does.

This is evidence, not ground truth, and it can err in BOTH directions, so both
are bounded:

* *Missed* -- a change whose new lane is not held for
  :data:`LANE_CHANGE_HOLD_FRAMES` (including one in the last few frames of a
  trace) is not counted. The episode then reports "no completion", which is the
  safe direction.
* *Falsely credited* -- the writer reports the NEAREST lane, so at a split two
  parallel lanes can alternate and each flap carries a real offset sign flip.
  This would credit a manoeuvre the ego never made, which is the unsafe
  direction, and it is what the hold requirement exists to reject.

Asking the map would settle both, and is not available to a pure trace
classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Minimum absolute signed lateral offset on each side of a ``lane_id``
#: transition for it to read as a lateral crossing rather than a longitudinal
#: handoff. Lane half-widths on nuPlan maps are ~1.6-1.9 m, so a boundary
#: crossing shows well over a metre on both sides while a successor transition
#: sits near the centreline.
LANE_CHANGE_MIN_OFFSET_M = 0.5

#: Frames a new lane must be held before it counts as a lane change. The
#: trace reports the nearest lane centreline, which can alternate between
#: two parallel lanes at a split or a wide merge; requiring 0.5 s of
#: persistence rejects that flicker.
LANE_CHANGE_HOLD_FRAMES = 5

#: Half-window around the stop line inside which a stop-sign episode's ego is
#: exempt from the deadlock rule. Mirrors the evaluator's
#: ``TL_STOP_LINE_ZONE_M``: the same 5 m that decides whether a light was run
#: decides whether a stationary ego is "at the line" or merely frozen.
STOP_LINE_ZONE_M = 5.0

#: What a stop-sign stop is: this long, below this speed. Rules of the road
#: rather than tuned thresholds, hence constants and not per-scenario fields.
STOP_SIGN_DWELL_S = 3.0
STOP_SIGN_SPEED_MS = 0.3


@dataclass(frozen=True)
class ScenarioRules:
    """How one event type departs from the default scoring rules.

    Attributes:
        leaf: The event type these rules came from, or ``None`` for the
            default. Carried so a metrics file can say which rules ran.
        disabled_terminations: Termination reason *values* (the enum's string
            form, so this module does not import the taxonomy) that must not
            end an episode on this event type.
        disabled_infractions: Infraction keys that must not be recorded on this
            event type, whether they came from a termination reason or from the trace.
        require_lane_change: Gate ``goal_reached`` on trace evidence of a
            completed lane change.
        hold_region: What the ego must stay behind for the episode to be a
            success -- ``"stop_line"`` (the signalled connector's entry, read
            from ``dist_to_stopline_m``) or ``"start"`` (within
            :attr:`hold_tolerance_m` of the first scored pose). ``None`` for
            every event type whose objective is to reach somewhere rather than hold.
        hold_tolerance_m: Radius of the ``"start"`` form's hold region.
        exit_region_ends_episode: Whether leaving the drivable surface after a
            kept hold is a completion rather than ``off_drivable``.
        stop_sign_dwell_s: Seconds below :data:`STOP_SIGN_SPEED_MS` the ego must
            accumulate before crossing the line. ``None`` disables the check,
            which is what every event type but V-1's stop-sign scenarios gets: the
            ``stop_infraction`` channel then stays SKIPPED rather than
            reporting a zero that would read as "checked and clean".
        note: One line for the metrics file, explaining the departure to
            whoever reads the number later.
    """

    leaf: str | None = None
    disabled_terminations: frozenset[str] = field(default_factory=frozenset)
    disabled_infractions: frozenset[str] = field(default_factory=frozenset)
    require_lane_change: bool = False
    hold_region: str | None = None
    hold_tolerance_m: float = 2.0
    #: Leaving the mapped surface ends the episode as a completion, once
    #: the ego has cleared the junction without entering the region it
    #: must stay out of. Used where the compliant behaviour is to leave
    #: the logged route (V-8).
    exit_region_ends_episode: bool = False
    stop_sign_dwell_s: float | None = None
    note: str = ""

    @property
    def is_default(self) -> bool:
        """True when these rules change nothing."""
        return (not self.disabled_terminations
                and not self.disabled_infractions
                and not self.require_lane_change
                and self.hold_region is None
                and self.stop_sign_dwell_s is None)

    def to_dict(self) -> dict[str, Any]:
        """Provenance for ``metrics.json`` — omitted entirely when default."""
        return {
            "leaf": self.leaf,
            "disabled_terminations": sorted(self.disabled_terminations),
            "disabled_infractions": sorted(self.disabled_infractions),
            "require_lane_change": self.require_lane_change,
            "hold_region": self.hold_region,
            "hold_tolerance_m": self.hold_tolerance_m,
            "stop_sign_dwell_s": self.stop_sign_dwell_s,
            "note": self.note,
        }


DEFAULT_RULES = ScenarioRules()

#: Hold regions that apply only when the scenario authors a signal state. A
#: hold posed by a sign in the recipe (V-8) does not depend on
#: `signal_override`.
_SIGNAL_HOLD_REGIONS = frozenset({"stop_line"})

LEAF_RULES: Mapping[str, ScenarioRules] = {
    "I-2": ScenarioRules(
        leaf="I-2",
        disabled_terminations=frozenset({"wrong_way"}),
        disabled_infractions=frozenset({"wrong_way"}),
        note=("Work-Zone Incident: bypassing the construction zone requires "
              "the opposing lane, so wrong-way is the manoeuvre and not the "
              "violation. Contacts, off-drivable and the time budget are "
              "unchanged."),
    ),
    "V-8": ScenarioRules(
        leaf="V-8",
        # `goal_reached` is disabled because the logged route makes
        # the prohibited turn, so the goal region lies inside the
        # lane the ego must not enter. `deadlock` and `wrong_way`
        # stay: the ego may legally go straight or right, so standing
        # still or turning around is not compliance.
        disabled_terminations=frozenset({"goal_reached"}),
        # `outside_route_lanes` measures distance off the logged
        # route, which here is the prohibited turn. A compliant
        # policy must leave it.
        disabled_infractions=frozenset({"outside_route_lanes"}),
        hold_region="no_turn_lane",
        # A compliant ego leaves the logged route and soon runs out
        # of mapped surface; that ends the episode as a completed
        # hold.
        exit_region_ends_episode=True,
        note=("Illegal Turn: a prohibitory plate makes the logged turn illegal, "
              "so the episode succeeds by NOT entering the turn lane -- going "
              "straight, turning the other way, or stopping short are all "
              "compliant. Entering ends it as illegal_turn and is priced by the "
              "illegal_turn channel. Success additionally REQUIRES that the ego "
              "reached the intersection: a trace that ends before the ego met "
              "the event cannot say whether the plate was read, so it is "
              "unscorable rather than a free pass."),
    ),
    "V-1": ScenarioRules(
        leaf="V-1",
        # `goal_reached` is disabled because the goal region lies
        # past the intersection, so entering it is the violation.
        # `deadlock` is disabled because standing at a red light is
        # correct.
        disabled_terminations=frozenset({"goal_reached", "deadlock"}),
        hold_region="stop_line",
        note=("Red-Light: the objective is to hold behind the stop line while "
              "the signal is red, so the episode succeeds by NOT reaching the "
              "goal. Crossing ends it as red_light_run and is priced by the "
              "red_light channel; contacts, off-drivable and wrong-way are "
              "unchanged."),
    ),
    # C-3 is NOT a plain event type entry: only the two precursors where the EGO is
    # the one changing lane need the stricter goal (see the module docstring and
    # LANE_CHANGE_REQUIRED_TYPES). `rules_for_scenario` builds it from the
    # scenario types, so a C-3 bundle that is `near_high_speed_vehicle` keeps
    # the default goal.
}

#: V-1's stop-sign precursors. A stop sign is not a hold: the ego must stop for
#: :data:`STOP_SIGN_DWELL_S` and then GO, so the default goal and the default
#: success rule are both correct and only the dwell has to be added. Keying this
#: on the type rather than the event type is what keeps the nine traffic-light
#: scenarios on the hold rules.
STOP_SIGN_TYPES: frozenset[str] = frozenset({
    "on_stopline_stop_sign",
    "starting_straight_stop_sign_intersection_traversal",
    "stopping_at_stop_sign_without_lead",
    "stopping_at_stop_sign_no_crosswalk",
    "accelerating_at_stop_sign",
    "accelerating_at_stop_sign_no_crosswalk",
})

_STOP_SIGN_RULES = ScenarioRules(
    leaf="V-1",
    # Deadlock stays enabled. Waiting at the line is exempted frame by
    # frame in `classify`, within STOP_LINE_ZONE_M, so that an ego
    # stopped far short of the line is still caught.
    stop_sign_dwell_s=STOP_SIGN_DWELL_S,
    note=("Stop-Sign: default goal and default success, plus stop_infraction "
          f"when the ego crosses the line without {STOP_SIGN_DWELL_S:.0f} s "
          f"below {STOP_SIGN_SPEED_MS} m/s first. The check is SKIPPED, not "
          "passed, on a trace with no approach to the line."),
)

#: nuPlan scenario types on which the ego itself must change lane for the
#: scenario to have been attempted. C-3's third precursor,
#: `near_high_speed_vehicle`, is deliberately absent: there the fast vehicle
#: passes alongside and holding the lane is correct.
LANE_CHANGE_REQUIRED_TYPES: frozenset[str] = frozenset({
    "changing_lane_with_lead",
    "changing_lane_with_trail",
})

#: The rules a lane-change-required scenario gets. Built once so the identity
#: is stable for the conflict check in `rules_for_scenario`.
_LANE_CHANGE_RULES = ScenarioRules(
    leaf="C-3",
    require_lane_change=True,
    note=("Sideswipe, ego-initiated lane change: holding the lane and driving "
          "straight past is a declined scenario, not a completion, so "
          "goal_reached requires trace evidence of a lane change. Not applied "
          "to near_high_speed_vehicle, where the ego is overtaken and keeping "
          "its lane is correct."),
)


def rules_for_leaf(leaf: str | None) -> ScenarioRules:
    """Rules for one event type; :data:`DEFAULT_RULES` when not overridden."""
    if not leaf:
        return DEFAULT_RULES
    return LEAF_RULES.get(str(leaf).strip().upper(), DEFAULT_RULES)


def rules_for_scenario(scenario_meta: Mapping[str, Any] | None) -> ScenarioRules:
    """Rules for a bundle, from its ``manifest.json`` ``scenario_meta``.

    A scenario can carry several event types (``taxonomy_leaves`` is a list, and
    nuPlan tags a window with several types). At most one may override the
    defaults: two different event type overrides on one bundle would be a scoring
    ambiguity the metrics file could not express, so this raises rather than
    silently picking the first. A bundle with no meta, or whose event types are all
    unlisted, gets the defaults.

    Raises:
        ValueError: the scenario carries two or more overriding event types.
    """
    meta = scenario_meta or {}
    leaves = list(meta.get("taxonomy_leaves") or [])
    types = {str(x).strip() for x in (meta.get("scenario_types") or [])}
    overrides = []
    for leaf in leaves:
        rules = rules_for_leaf(leaf)
        # Applies to signal-backed holds only. A hold posed by a sign
        # does not use `signal_override`.
        if (rules.hold_region in _SIGNAL_HOLD_REGIONS
                and not meta.get("signal_override")):
            # Without an authored red light there is no hold to
            # judge: the logged lights in the published V-1
            # bundles are never red. Such bundles keep the
            # default rules.
            continue
        if not rules.is_default and rules not in overrides:
            overrides.append(rules)
    # C-3's stricter goal is keyed on the scenario TYPE, not the leaf: two of
    # its three precursors have the ego changing lane and one has the ego being
    # overtaken. Keyed on the event type it would have applied the wrong goal to 22
    # of the corpus's 23 C-3 rows.
    if types & LANE_CHANGE_REQUIRED_TYPES and _LANE_CHANGE_RULES not in overrides:
        overrides.append(_LANE_CHANGE_RULES)
    # A stop sign has no hold, so for a V-1 scenario of a stop-sign type
    # the stop-sign rules replace the hold. The hold is the default
    # because a bundle may list no scenario types.
    if types & STOP_SIGN_TYPES:
        overrides = [r for r in overrides if r.leaf != "V-1"]
        overrides.append(_STOP_SIGN_RULES)
    if not overrides:
        return DEFAULT_RULES
    if len(overrides) > 1:
        raise ValueError(
            "scenario carries two event types with conflicting scoring rules "
            f"({[r.leaf for r in overrides]}); one bundle cannot be scored "
            "under both. Split the scenario or give the pair an explicit rule.")
    return overrides[0]


# ---------------------------------------------------------------------------
# Lane-change evidence (C-3)
# ---------------------------------------------------------------------------

def lane_change_frames(frames: Sequence[Mapping[str, Any]],
                       *, scored_only: bool = True) -> list[int]:
    """Frame indices at which the ego completed a lateral lane change.

    Two tests, and both are needed:

    * the signed ``lateral_offset_m`` flips sign with at least
      :data:`LANE_CHANGE_MIN_OFFSET_M` on each side -- what separates a lateral
      crossing from a longitudinal successor handoff (see the module docstring);
    * the new lane is then held for :data:`LANE_CHANGE_HOLD_FRAMES` -- what
      separates a crossing from nearest-lane flapping at a split.

    Args:
        frames: The trace, oldest first.
        scored_only: Consider only ``phase == "scored"`` frames, so a lane
            change performed by the warm-up's ground-truth replay is not
            credited to the policy.

    Returns:
        Indices (into ``frames``) of the frame on which each change completed.
        A candidate whose new lane is not held long enough is dropped, and so is
        one at the very end of the trace with too few frames left to confirm --
        unconfirmable is not confirmed.
    """
    # (index, lane) for the frames under consideration, so the hold check can
    # look ahead without re-filtering.
    seq = [(i, f) for i, f in enumerate(frames)
           if not (scored_only and f.get("phase") != "scored")]
    lanes = [(f.get("lane_id") or None) for _, f in seq]

    prev_lane: str | None = None
    prev_offset: float | None = None
    out: list[int] = []
    for pos, (i, f) in enumerate(seq):
        lane = lanes[pos]
        offset = f.get("lateral_offset_m")
        offset = None if offset is None else float(offset)
        if (prev_lane is not None and lane is not None and lane != prev_lane
                and prev_offset is not None and offset is not None
                and abs(prev_offset) >= LANE_CHANGE_MIN_OFFSET_M
                and abs(offset) >= LANE_CHANGE_MIN_OFFSET_M
                and (prev_offset > 0) != (offset > 0)):
            window = lanes[pos:pos + LANE_CHANGE_HOLD_FRAMES]
            if (len(window) >= LANE_CHANGE_HOLD_FRAMES
                    and all(w == lane for w in window)):
                out.append(i)
        if lane is not None:
            prev_lane = lane
        if offset is not None:
            prev_offset = offset
    return out


def changed_lane(frames: Sequence[Mapping[str, Any]], **kwargs: Any) -> bool:
    """Whether the trace shows at least one completed lane change."""
    return bool(lane_change_frames(frames, **kwargs))


def has_lane_data(frames: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the trace carries the lane columns the C-3 gate reads.

    Follows this package's standing rule that absent data means "not checked",
    never a verdict — the same reason ``driving_direction_ok`` and
    ``on_drivable`` default True in ``termination.classify``. Without it, a
    harness that does not populate ``lane_id`` (the live monitor, an older
    artifact) would fail the lane-change gate on every C-3 episode and report
    every one of them as a non-completion, which looks exactly like a policy
    that declined the manoeuvre.
    """
    for f in frames:
        if f.get("lane_id") and f.get("lateral_offset_m") is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Hold evidence (V-1 red light)
# ---------------------------------------------------------------------------

def _scored(frames: Sequence[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]]:
    return [(i, f) for i, f in enumerate(frames) if f.get("phase") == "scored"]


def hold_violation(frames: Sequence[Mapping[str, Any]],
                   rules: ScenarioRules) -> tuple[int | None, str]:
    """First frame at which the ego left the region it had to hold behind.

    Three tests, tried in this order because they differ in authority:

    1. ``signal_state == "red"``. In the eval path that column is set from the
       evaluator's TL subscore dropping below 1.0 (``trace/from_eval.py``), i.e.
       the scorer's own crossing rule -- real lane polygons, the 5 m entry zone,
       the green-way-through exemption. Nothing here can do better, so where it
       exists it decides. (The live writer fills the same column with the raw
       light COLOUR; a trace from that producer would read a compliant wait as a
       violation, which is why this only runs for event types that opted in.)
    2. ``dist_to_stopline_m`` crossing zero, for artifacts with no TL column.
       Requires the ego to have been in front of the line first: a trace that
       starts past it never approached, and "not checked" is the honest verdict.
    3. Displacement from the first scored pose beyond ``hold_tolerance_m``, the
       ``"start"`` form, for scenarios where no stop line resolves.

    Returns:
        ``(frame index, detail)``, or ``(None, "")`` when the hold was kept --
        which includes the case where nothing could be measured.
    """
    if not rules.hold_region:
        return None, ""
    scored = _scored(frames)
    if not scored:
        return None, ""

    # V-8's hold is a region, not a signal, so it is answered before the
    # traffic-light tests. `in_exit_lane` is the exit lane of the logged
    # route, which on a V-8 scenario is the lane the sign forbids.
    if rules.hold_region == "no_turn_lane":
        if not any("in_exit_lane" in f for _, f in scored):
            return None, ""     # column absent: not checked, not "kept"
        for i, f in scored:
            if f.get("in_exit_lane"):
                return i, f"entered the prohibited turn lane at frame {i}"
        return None, ""

    for i, f in scored:
        if f.get("signal_state") == "red":
            return i, f"crossed the stop line against a red signal at frame {i}"

    # If the traffic-light column exists and reports no violation, that
    # is the answer. The geometry test below fires about two frames
    # earlier than the scorer and must not overrule it.
    if any(f.get("signal_state") for _, f in scored):
        return None, ""

    if rules.hold_region == "stop_line":
        approached = False
        for i, f in scored:
            d = f.get("dist_to_stopline_m")
            d = float("nan") if d is None else float(d)
            if d != d:          # NaN: the column was never populated
                continue
            if d > 0:
                approached = True
            elif approached:
                return i, f"passed the stop line at frame {i} ({d:.1f} m beyond)"
        return None, ""

    x0, y0 = scored[0][1].get("ego_x"), scored[0][1].get("ego_y")
    if x0 is None or y0 is None:
        return None, ""
    tol = float(rules.hold_tolerance_m)
    for i, f in scored:
        dx = float(f.get("ego_x", x0)) - float(x0)
        dy = float(f.get("ego_y", y0)) - float(y0)
        moved = (dx * dx + dy * dy) ** 0.5
        if moved > tol:
            return i, (f"moved {moved:.1f} m from the start pose at frame {i} "
                       f"(hold radius {tol:.1f} m)")
    return None, ""


def hold_measurable(frames: Sequence[Mapping[str, Any]],
                    rules: ScenarioRules) -> bool:
    """Whether this trace can witness a hold being KEPT.

    Not symmetric with :func:`hold_violation`, and deliberately: a crossing can
    be convicted by either witness, while "held" needs positive evidence that
    the ego was in front of the line and stayed there. Without that, an ego
    that drove straight through an intersection the artifact cannot locate would
    be reported as a perfect hold -- measured on ``00c1e4eb4a045f20``, where an
    ego that travelled 130.6 m at 5-7.6 m/s scored ``hold_satisfied``, SR true,
    DS 100.0 because neither the light nor the stop line resolved. Absence of
    evidence is "not checked", which on this event type must fall through to the
    ordinary endings rather than become a completion.

    The ``"start"`` form is always measurable: displacement from the first
    scored pose needs no map.
    """
    if not rules.hold_region:
        return False
    if rules.hold_region != "stop_line":
        return bool(_scored(frames))
    for _, f in _scored(frames):
        d = f.get("dist_to_stopline_m")
        if d is None:
            continue
        d = float(d)
        if d == d and d > 0:
            return True
    return False


#: Distance past the stop line beyond which a "no violation" signal verdict
#: is not accepted as a hold. The scorer exempts a crossing when another
#: non-red connector offers a way through, which can also happen when an
#: override reddened only some connectors of an approach.
HOLD_CONTRADICTED_M = 5.0


def hold_contradicted(frames: Sequence[Mapping[str, Any]],
                      rules: ScenarioRules) -> tuple[bool, str]:
    """Whether the trace refutes a hold that the signal did not convict.

    Guards the case where both witnesses are present and disagree in the unsafe
    direction: the TL column reports compliance while the ego is metres past the
    line. Reporting that as ``hold_satisfied`` would hand a policy that drove
    through the junction SR true, so the episode falls through to the ordinary
    endings instead -- a non-completion, which is the safe reading of "the
    scenario could not pose its question".
    """
    if not rules.hold_region:
        return False, ""
    worst = 0.0
    for _, f in _scored(frames):
        d = f.get("dist_to_stopline_m")
        if d is None:
            continue
        d = float(d)
        if d == d:
            worst = min(worst, d)
    if -worst >= HOLD_CONTRADICTED_M:
        return True, (f"the signal reported no violation, but the ego ended "
                      f"{-worst:.1f} m past the stop line — check that the "
                      f"override reddened every connector of this approach")
    return False, ""


def hold_fraction(frames: Sequence[Mapping[str, Any]],
                  violation_frame: int | None,
                  *, window_frames: int | None = None) -> float:
    """Percent of the episode's window the hold survived.

    Replaces route completion as the DS base on a hold event type, because the route
    runs through the intersection the ego is correctly not entering: a perfect
    hold completes 0 % of it and would score ~0. A clean hold is 100.0.

    ``window_frames`` is the window the episode was ENTITLED to, and passing it
    matters as soon as a crossing ends the episode early. Measured against the
    recorded trace instead, a simwam run that crossed at frame 28 scored 88.9 --
    it held 8 of the 9 frames that got recorded -- while the identical behaviour
    before the live monitor could end the episode scored 1.3 out of 599. Same
    policy, same scenario, differing only in when the harness stopped stepping.
    The budget is the denominator; the trace is not.
    """
    scored = _scored(frames)
    if not scored:
        return 0.0
    denom = max(int(window_frames) if window_frames else len(scored), 1)
    if violation_frame is None:
        return 100.0
    held = sum(1 for i, _ in scored if i < violation_frame)
    return 100.0 * min(held / denom, 1.0)


# ---------------------------------------------------------------------------
# Stop-sign dwell (V-1 stop-sign precursors)
# ---------------------------------------------------------------------------

def stop_sign_violation(frames: Sequence[Mapping[str, Any]],
                        rules: ScenarioRules) -> tuple[bool | None, str]:
    """Whether the ego crossed the line without the required stop.

    Returns:
        ``(True, detail)`` for a violation, ``(False, detail)`` for a compliant
        stop, and ``(None, reason)`` when the trace cannot answer -- no dwell
        configured, no ``dist_to_stopline_m`` column, or no approach to the line
        in the window. The third case is why this returns a tri-state:
        ``9311d9a2409c5224`` is tagged ``accelerating_at_stop_sign``, so its ego
        is already leaving the line and its stop happened before the window
        opened. Scoring that as a violation would charge the policy for a stop
        the artifact cannot see.
    """
    if rules.stop_sign_dwell_s is None:
        return None, "no stop-sign dwell configured for this scenario"
    scored = _scored(frames)
    approach = [(i, f) for i, f in scored
                if (f.get("dist_to_stopline_m") is not None
                    and float(f["dist_to_stopline_m"]) == float(f["dist_to_stopline_m"])
                    and float(f["dist_to_stopline_m"]) > 0)]
    if not approach:
        return None, ("trace carries no approach to the stop line "
                      "(no dist_to_stopline_m > 0 on a scored frame)")
    crossing = next((i for i, f in scored
                     if i > approach[-1][0]
                     and f.get("dist_to_stopline_m") is not None
                     and float(f["dist_to_stopline_m"]) == float(f["dist_to_stopline_m"])
                     and float(f["dist_to_stopline_m"]) <= 0), None)
    dt = 0.1
    best = run = 0
    for i, f in approach:
        speed = float(f.get("ego_speed") or 0.0)
        run = run + 1 if speed < STOP_SIGN_SPEED_MS else 0
        best = max(best, run)
    dwell_s = best * dt
    needed = float(rules.stop_sign_dwell_s)
    if crossing is None:
        # Still short of the line when the window ended: nothing was violated
        # yet, and calling it compliant would credit a stop that may never come.
        return None, (f"ego never crossed the line in the window "
                      f"(longest stop {dwell_s:.1f} s)")
    if dwell_s + 1e-9 >= needed:
        return False, (f"stopped {dwell_s:.1f} s before crossing at frame "
                       f"{crossing} (needed {needed:.1f} s)")
    return True, (f"crossed at frame {crossing} after only {dwell_s:.1f} s "
                  f"below {STOP_SIGN_SPEED_MS} m/s (needed {needed:.1f} s)")


def near_stop_line(frame: Mapping[str, Any]) -> bool:
    """Whether this frame sits inside the stop line's deadlock-exempt zone."""
    d = frame.get("dist_to_stopline_m")
    if d is None:
        return False
    d = float(d)
    if d != d:
        return False
    return abs(d) <= STOP_LINE_ZONE_M


def filter_infractions(infractions: Mapping[str, float],
                       rules: ScenarioRules) -> dict[str, float]:
    """Drop the infraction keys this event type disables."""
    return {k: v for k, v in infractions.items()
            if k not in rules.disabled_infractions}


def allows_termination(reason: Any, rules: ScenarioRules) -> bool:
    """Whether ``reason`` (a ``TerminationReason`` or its value) may end an
    episode under ``rules``."""
    value = getattr(reason, "value", reason)
    return str(value) not in rules.disabled_terminations


__all__ = [
    "DEFAULT_RULES",
    "LANE_CHANGE_HOLD_FRAMES",
    "LANE_CHANGE_MIN_OFFSET_M",
    "LANE_CHANGE_REQUIRED_TYPES",
    "LEAF_RULES",
    "STOP_LINE_ZONE_M",
    "STOP_SIGN_DWELL_S",
    "STOP_SIGN_SPEED_MS",
    "STOP_SIGN_TYPES",
    "ScenarioRules",
    "allows_termination",
    "changed_lane",
    "filter_infractions",
    "has_lane_data",
    "hold_fraction",
    "hold_violation",
    "lane_change_frames",
    "near_stop_line",
    "rules_for_leaf",
    "rules_for_scenario",
    "stop_sign_violation",
]
