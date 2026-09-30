# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Authoring: a leaf's intent in, a frozen recipe out.

The intent — "52 m ahead, in my lane, facing me, 8 m/s" — is declared once per
leaf and expanded against a host by ``leaves/rule.py``. This module resolves
that into the absolute numbers a recipe pins, and is the only place the three
lower layers meet:

    authoring spec (short, parameterised)
      │  HostProbe          — what this host's road actually looks like
      │  templates + bake   — route-frame intent -> a world-frame pose
      ▼
    Recipe (spawn + policy per actor)  ->  freeze_recipe()  ->  the file

One intent is solved into geometry here rather than authored as a number:
``arrive_with_ego`` — "arrive as the ego reaches the conflict point". The
conflict point is resolved (for two different references, the intersection of
the actor's polyline with the ego's route) and the start arc or conflict frame
is back-solved from it, against the LOG-REPLAY ego. That is the only ego
available at bake time; the one the scenario will actually meet is the policy
under test, so the solve places the actor and does not promise a meeting.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from navsafe.benchmark.editing.assets.registry import AssetError, AssetRegistry
from navsafe.benchmark.editing.placement.anchors import resolve_anchor
from navsafe.benchmark.editing.placement.probe import (
    MERGE_FAR_M,
    HostProbe,
    PlacementError,
)
from navsafe.benchmark.editing.placement.visibility import (
    DEFAULT_MIN_REACTION_S,
    dominant_blocker,
    occlusion_report,
    summarise as summarise_visibility,
)
from navsafe.benchmark.editing.placement.solve import (
    Polyline,
    closest_approach,
    intersect_polylines,
    solve_start_arc_for_conflict,
)
from navsafe.benchmark.editing.recipe.schema import (
    ActorRecipe,
    AssetRef,
    EgoSpec,
    Frames,
    HostSpec,
    Recipe,
)
from navsafe.benchmark.editing.trajectory.bake import BakedTrack, bake_actor_state
from navsafe.benchmark.editing.trajectory.templates import build_motion
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

# How close the log-replay ego must come for the event to count as "met".
DEFAULT_EVENT_RADIUS_M = 15.0

# The documented camera-only recon-road drift band. Beyond this the road is
# somewhere other than the recipe thinks.
ROAD_DRIFT_TOL_M = 0.4

# Keys consumed by the authoring layer itself; everything else in ``authored``
# is a template parameter.
_CONTROL_KEYS = {
    "template", "reference", "anchor", "arrive_with_ego", "arrive_offset_s",
    "conflict_arc", "reference_polyline", "speed_min", "speed_max",
}

# Lateral offsets an author can name instead of measuring. A dart-out crosses
# from one side of the road to the other, and where those sides ARE is a
# property of the host at that arc — `probe.cross_section` already measures it.
# Writing metres instead means every host needs its own hand-tuned spec, and a
# number that was the verge on one clip is the oncoming lane on the next.
# Beyond this, a "road edge" is another road's polyline that happened to be
# nearest, not this carriageway's kerb.
LATERAL_SANITY_M = 15.0

# How far apart an `auto`-placed group has to stay for each member to be its
# own encounter rather than part of one wall. Roughly a second of driving at
# urban speed: closer than this and the ego meets them as a single event.
MIN_GROUP_SPACING_M = 6.0

# How near two references have to come before "they never cross" is a merge
# rather than a mistake. A lane width: closer than this and one road's traffic
# is in the other's, which is the conflict a merge leaf is about; further and
# the reference is simply the wrong one for this actor.
CONFLICT_MERGE_MAX_M = 4.0

#: Below this a gait-banked actor is no longer crossing in front of
#: anything, so a bank whose cap falls here is a bank to re-bake.
MIN_GAIT_SPEED_MS = 0.8

#: Below this an `arrive_with_ego` actor is not travelling, it is loitering.
#: A reference too short to be covered at any believable speed is a
#: host-selection finding, so the solve refuses rather than crawling.
MIN_ARRIVAL_SPEED_MS = 2.0

#: How close the ego may be when a held actor steps off. `arrive_offset_s` is a
#: TIME, so on a host where the ego crawls it folds down to almost no distance:
#: R-4 asks its animal to reach the ego's line 0.5 s ahead, and on
#: 225eb6e22af55972 (ego 2.1 m/s) that solved to 3.6 m and on 02015675e4585611
#: (3.0 m/s) to 3.3 m -- 0.6 s and 1.0 s of approach against the 1.5 s the leaf
#: declares, i.e. an animal appearing in the bumper rather than a yield the
#: policy can be asked about. Raising the threshold can only make an actor leave
#: EARLIER: the ego reached the smaller distance, so it passes through this one
#: first, and no held actor is turned into one that waits forever.
MIN_TRIGGER_DISTANCE_M = 5.0

LATERAL_FEATURES = {
    "road_edge_left": "left_road_edge_m",
    "road_edge_right": "right_road_edge_m",
    "boundary_left": "left_boundary_m",
    "boundary_right": "right_boundary_m",
    "lane_centre": None,   # 0.0
    "lane_center": None,
}

# Laterals are +LEFT, so a feature whose NAME states a side must measure on
# that side. The map does not guarantee it: on 0bcae698fd905226 the nearest
# thing tagged "left road edge" sat 2.26 m to the RIGHT of the ego's line, and
# the resolver took it because it was neither missing nor implausibly far. The
# cost was silent — R-4's four animals were authored to cross from the left
# verge to beyond the right kerb, and baked as four actors that START on the
# right, two of them stepping 1.7 m onto the ego's centreline and stopping
# there. A dart-out that never crosses renders fine and tests nothing.
# A wrong-side hit means this is not the feature the spec asked for, so it is
# treated exactly like a missing one: fall through to the next alternative.
LATERAL_SIDE = {
    "road_edge_left": +1.0, "boundary_left": +1.0,
    "road_edge_right": -1.0, "boundary_right": -1.0,
}


def _resolve_lateral(probe, reference, value, arc: float, *, where: str) -> float:
    """Turn ``road_edge_right`` (or ``a|b|c``, or ``name+0.5``) into metres HERE.

    Real maps are patchy along a route: this data has stretches with a left
    boundary and no right edge, and stretches with the opposite, on the same
    clip. A spec naming one feature therefore works at 30 m and fails at 65 m,
    which is not a spec problem — so alternatives are separated by ``|`` and
    tried in order:

        start_lateral: road_edge_left|boundary_left|lane_centre+4.5

    reads as "the kerb if this host has one here, else the lane boundary, else
    4.5 m left of the centreline". ``lane_centre`` always resolves, so a chain
    ending in it always succeeds — and says plainly that the last resort is a
    stated offset rather than a measured feature.

    Raises:
        AuthoringError: every alternative is missing, or the one that resolved
            is implausibly far (another road's polyline).
    """
    if not isinstance(value, str):
        return float(value)
    attempts = []
    for alt in [v.strip() for v in value.split("|") if v.strip()]:
        name, margin = alt, 0.0
        for sign in ("+", "-"):
            if sign in alt[1:]:
                head_, _, tail = alt.rpartition(sign)
                try:
                    margin, name = float(f"{sign}{tail}"), head_
                    break
                except ValueError:
                    pass
        name = name.strip()
        if name not in LATERAL_FEATURES:
            raise AuthoringError(
                f"{where}: unknown lateral {alt!r} in {value!r}; use a number, or one of "
                f"{sorted(LATERAL_FEATURES)} (with an optional '+0.5' margin), or several "
                f"separated by '|'")
        attr = LATERAL_FEATURES[name]
        if attr is None:
            return 0.0 + margin
        section = probe.cross_section(reference, float(arc))
        measured = getattr(section, attr)
        if measured is None:
            attempts.append(f"{name}: not in this host's map here")
            continue
        if abs(float(measured)) > LATERAL_SANITY_M:
            # A kerb 130 m from the lane is another road's polyline that happened
            # to be nearest, not this carriageway's edge. Taking it put two
            # animals 56 m and 70 m from the ego: a recipe that bakes clean and
            # renders an empty road.
            attempts.append(f"{name}: {measured:+.1f} m — another road's line")
            continue
        side = LATERAL_SIDE.get(name)
        if side is not None and float(measured) * side <= 0.0:
            attempts.append(
                f"{name}: {measured:+.1f} m is on the "
                f"{'right' if float(measured) < 0 else 'left'}, not the "
                f"{name.rsplit('_', 1)[1]}")
            continue
        return float(measured) + margin
    raise AuthoringError(
        f"{where}: none of {value!r} resolves at arc {arc:.1f} m ({'; '.join(attempts)}). "
        f"Add a fallback (…|lane_centre+4.0), give metres, or pick a host whose map "
        f"covers this stretch.")


def _resolve_speed(probe, value, *, lo, hi, where: str) -> float:
    """Turn ``ego*0.75`` (or ``ego-1.8``, or a number) into m/s on THIS host.

    A speed that must sit in a relationship to the ego's — slower, so it gets
    overtaken; faster, so it pulls away — cannot be written as an absolute
    number, because the ego's own pace is a property of the clip. The R-2 set
    is the demonstration: 5.0 and 4.4 m/s are honest cycling speeds and were
    at or above the ego's own on every host, so the riders drew away from the
    hand-off and the leaf tested nothing.

    ``speed_min`` / ``speed_max`` keep the answer a real cyclist rather than an
    arithmetic result: a 2.4 m/s ego times 0.65 is a pace nobody rides. The
    clamp is the author's statement of the band the actor is physically capable
    of, so it belongs in the spec beside the expression. Two scalars rather
    than one ``[lo, hi]`` pair because a list under ``authored:`` is indexed
    PER ACTOR (:func:`~navsafe.benchmark.leaves.rule._per_slot`), so a range
    written as a pair would silently become one bound each.
    """
    if not isinstance(value, str):
        return float(value)
    text = value.strip()
    if not text.startswith("ego"):
        raise AuthoringError(
            f"{where}: unknown speed {value!r}; give metres per second, or an expression on "
            f"the ego's own pace — 'ego', 'ego*0.75', 'ego-1.8'.")
    ego = float(probe.ego_cruise_speed())
    rest = text[3:].strip()
    try:
        if not rest:
            out = ego
        elif rest[0] == "*":
            out = ego * float(rest[1:])
        elif rest[0] in "+-":
            out = ego + float(rest)
        else:
            raise ValueError(rest)
    except ValueError:
        raise AuthoringError(
            f"{where}: cannot read {value!r}; expected 'ego', 'ego*<factor>' or "
            f"'ego<+/-><m/s>'.") from None
    if lo is not None or hi is not None:
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        clamped = min(max(out, lo), hi)
        if abs(clamped - out) > 1e-6:
            logger.info("%s: %s = %.2f m/s on a %.2f m/s ego, clamped to %.2f (range %.1f-%.1f)",
                        where, value, out, ego, clamped, lo, hi)
        out = clamped
    logger.info("%s: %s -> %.2f m/s (ego cruises at %.2f m/s)", where, value, out, ego)
    return round(out, 3)


class AuthoringError(NexusSimError, ValueError):
    """The authoring spec cannot be resolved against this host."""


def _authored_reference(probe: HostProbe, authored: Optional[Dict[str, Any]]):
    """Resolve an actor's reference from its own ``authored`` block.

    ``_documented`` records an INLINE polyline as the literal string
    ``"polyline"`` and freezes the geometry in ``reference_polyline``, so
    re-resolving the string alone raises PlacementError. Callers that need the
    reference after baking must go through here.
    """
    authored = authored or {}
    ref = authored.get("reference", "ego_route")
    if ref == "polyline":
        frozen = authored.get("reference_polyline")
        if not frozen:
            raise AuthoringError(
                "authored.reference is 'polyline' but no reference_polyline was frozen "
                "with it, so the actor's own path cannot be recovered."
            )
        return probe.resolve_reference(frozen)
    return probe.resolve_reference(ref)


def _conflict_arc(
    probe: HostProbe, reference: Polyline, params: Dict[str, Any]
) -> Tuple[float, np.ndarray]:
    """Where the actor's path meets the ego's, as an arc on ``reference``.

    A CROSSING is the usual answer — a side street meets the ego's route at a
    point, and that point is the conflict. A MERGE has no crossing: two lanes
    that merge converge to within a lane width and then run together, so the
    conflict is where they come nearest. Both are "the ego and the actor want
    the same strip of road", which is what ``arrive_with_ego`` times against.
    """
    if "conflict_arc" in params:
        arc = float(params["conflict_arc"])
        x, y, _, _ = reference.sample([arc])
        return arc, np.array([float(x[0]), float(y[0])])
    hit = intersect_polylines(reference, probe.ego_route)
    if hit is not None:
        return hit[0], hit[2]
    near = _merge_conflict(probe, reference)
    if near is not None:
        return near
    # Ask the probe, which already found this merge when the leaf's gate ran.
    # Two reasons the geometry above can miss what the gate passed on:
    #
    #   A CLOSURE is not in the map at all — nuPlan has no work-zone layer, so
    #   the cones that end a lane leave its centreline running parallel to the
    #   ego's for ever, and no amount of geometry will call that a conflict.
    #
    #   A MAP MERGE is profiled by `merging_lane` on the LANE, while this runs
    #   on the 150 m CHAIN built through it; the chain's far ends break the
    #   far-then-near ordering the convergence test depends on, so a merge that
    #   qualified can fail here. 63c145828c3b5fd8 did exactly that.
    #
    # The leaf's own contract says the gate and the placement must read one
    # answer, or the gate passes on one lane while the actor goes in another.
    # This is that answer.
    pressure = probe.merge_pressure_lane()
    if pressure is not None:
        arc_on_route = (probe.anchor_arc(probe.ego_route, probe.after_frame)
                        + float(pressure["conflict_arc_m"]))
        rx, ry, _, _ = probe.ego_route.sample([arc_on_route])
        xy = np.array([float(rx[0]), float(ry[0])])
        arc = float(reference.project(xy)[0])
        logger.info(
            "conflict: from the %s gate at route arc %+.0f m (reference arc %.1f m), "
            "lane %s", pressure["cause"], float(pressure["conflict_arc_m"]), arc,
            pressure["lane_id"])
        return arc, xy
    raise AuthoringError(
        "arrive_with_ego needs a conflict point, and this actor's reference neither crosses "
        "the ego's route nor CONVERGES with it. Two lanes running parallel are not a "
        "conflict however close they are — nothing decides when the two arrive at one "
        "place. Give an explicit `conflict_arc` (metres along the actor's own reference), "
        "or author the arc directly."
    )


def _merge_conflict(probe: HostProbe, reference: Polyline):
    """The conflict point of a MERGE: where two converging paths come nearest.

    Two lanes that merge need never cross, so ``intersect_polylines`` finds
    nothing and ``arrive_with_ego`` has no crossing to time against. Their point
    of closest approach is that place — but only if they actually converge.
    Proximity alone is not a conflict: an opposing carriageway 3.5 m away is
    close for its whole length and there is no moment at which the two arrive
    anywhere together. So this asks for the same signature the ``merge_convergence``
    gate asks for, on this actor's own reference:

    * they end up within :data:`CONFLICT_MERGE_MAX_M` — a lane width, i.e. one
      road's traffic is now in the other's;
    * they were at least :data:`MERGE_FAR_M` apart earlier, and the near point
      comes AFTER the far one, so the separation is closing rather than opening;
    * they run the same way where they meet, because a head-on pair sharing a
      strip of road is C-7, not a merge, and is timed by crossing instead.

    Returns:
        ``(arc_on_reference, xy)``, or ``None`` when this is not a merge.
    """
    near = closest_approach(reference, probe.ego_route)
    if near is None or near[3] > CONFLICT_MERGE_MAX_M:
        return None
    s_ref, s_ego, xy, gap = near
    _, _, _, ref_heading = reference.sample([s_ref])
    _, _, _, ego_heading = probe.ego_route.sample([s_ego])
    if np.cos(float(ref_heading[0]) - float(ego_heading[0])) < 0.7:
        return None
    laterals = [abs(probe.ego_route.project(pt)[1]) for pt in reference.xy]
    far = max(laterals)
    if far < MERGE_FAR_M or laterals.index(min(laterals)) <= laterals.index(far):
        return None
    logger.info(
        "author: %s and the ego route never cross but converge %.1f -> %.2f m; timing "
        "against that closest approach, at reference arc %.1f m",
        reference.name or "the actor's reference", far, gap, s_ref,
    )
    return s_ref, xy


def _clamp_to_gait(name: str, bank_dir, resolved: Dict[str, Any]) -> None:
    """Hold an actor to the speed its own gait bank can be seen walking at.

    A pose bank moves limbs by swapping which posed copy is on screen, clocked
    off ground covered: ``phase = travel / stride``. Past about half a cycle per
    rendered frame the phases stop reading as a walk and start reading as
    flicker -- and flicker at 10 Hz reads as no gait at all, which is what R-4
    came back with. ``quadruped.max_speed`` has said so since the banks were
    baked and nothing called it: every R-4 animal was authored at 3.0 m/s, and
    the dog's 0.24 m cycle supports 1.19.

    Clamped rather than refused, and clamped PER ACTOR rather than by lowering
    the leaf's speed: the species is chosen per host by ``--variety``, so one
    number in the leaf would slow the cow and the horse -- which a reviewer
    confirmed look right at 3.0 -- to keep the dog legible. The clamp is
    recorded in the recipe and logged, so a dog trotting at 1.2 m/s where the
    leaf asked for 3.0 is visible in the diff rather than a surprise in the
    render. Below ``MIN_GAIT_SPEED_MS`` the actor is not crossing any more, so
    that is an error instead: the leaf's event would be gone.
    """
    import json

    from navsafe.benchmark.editing.assets.quadruped import max_speed

    try:
        spec = json.loads((Path(str(bank_dir)) / "bank.json").read_text())
        stride = float(spec["stride_m"])
    except Exception:  # noqa: BLE001 — a bad bank is reported where it is read
        return
    speed = float(resolved.get("speed") or 0.0)
    cap = max_speed(stride)
    if speed <= cap:
        return
    if cap < MIN_GAIT_SPEED_MS:
        raise AuthoringError(
            f"actor {name!r} would have to move at {cap:.2f} m/s for its gait bank "
            f"{Path(str(bank_dir)).name} ({stride:.2f} m cycle) to read as a gait, which is "
            f"too slow to cross in front of anything. Bake a longer-strided gait for this "
            f"species, or drop `gait_bank:` and let it slide."
        )
    logger.warning(
        "%s moves at %.1f m/s and its gait bank %s has a %.2f m cycle, which reads as a gait "
        "only up to %.2f m/s (%.1fx over); holding it to %.2f m/s. Past that the phase clock "
        "advances more than half a cycle per rendered frame and the limbs flicker rather than "
        "walk — indistinguishable from the static asset.",
        name, speed, Path(str(bank_dir)).name, stride, cap, speed / cap, cap)
    resolved["speed"] = round(float(cap), 3)
    resolved["speed_clamped_by_gait"] = True



def _resolve_timing(
    probe: HostProbe,
    reference: Polyline,
    template: str,
    params: Dict[str, Any],
    *,
    anchor_arc: float,
    dt_s: float,
    reaction_s: float = 0.0,
) -> Dict[str, Any]:
    """Turn an ``arrive_with_ego`` intent into an arc or a conflict frame."""
    if params.get("cut_in_frame") is not None:
        # WHERE THE EGO IS AT THAT FRAME, which is the only thing "cut in at
        # frame 50" can mean. Solved here because it needs the probe; the path
        # builder only sees the reference.
        k = int(params["cut_in_frame"])
        out = dict(params)
        out["_cut_in_arc"] = round(float(probe.anchor_arc(probe.ego_route, k)), 3)
        params = out
    if params.get("behind_ego_m") is not None or params.get("behind_ego_s") is not None:
        # PLACE IT, do not time it. `arrive_with_ego` solves for a moment at a
        # shared conflict point, and on a merge that put the inserted car far
        # up the road it is joining -- past the ego and gone -- long before the
        # ego turned in. What makes the merge unsafe is much simpler: at the
        # hand-off the car is a few metres BEHIND where the ego is about to be,
        # so the ego pulls out in front of it.
        #
        # `anchor_arc` is the ego's hand-off pose projected onto this reference,
        # so the target is `anchor - behind`; the actor has been running since
        # frame 0, so its spawn is that minus the distance it covers first.
        # A GAP IS A TIME, and it is measured between BUMPERS. `behind_ego_m`
        # was neither: 5.0 m is centre-to-centre, and with a 4.6 m car against a
        # 4.6 m ego that is 0.4 m of clear road -- every V-10 insert sat on the
        # ego's bumper at the hand-off with nothing for a policy to decide.
        # `behind_ego_s` is seconds of following distance at this actor's own
        # speed, so it means the same thing on a host doing 4 m/s and one doing
        # 11, and the car's own length is added because that is the half the
        # driver cannot use.
        back = float(params.get("behind_ego_m") or 0.0)
        if params.get("behind_ego_s") is not None:
            back = float(params["behind_ego_s"]) * float(params.get("speed", 0.0) or 0.0)
        back += float(params.get("_actor_length") or 0.0)
        speed = float(params.get("speed", 0.0) or 0.0)
        travelled = speed * float(probe.after_frame) * float(dt_s)
        out = dict(params)
        out["arc"] = round(-back - travelled, 4)
        out["_solved"] = {
            "placed_behind_ego_m": back,
            "handoff_frame": int(probe.after_frame),
            "travelled_before_handoff_m": round(travelled, 2),
        }
        return out
    if not params.get("arrive_with_ego"):
        return params
    resolved = dict(params)
    conflict_arc, conflict_xy = _conflict_arc(probe, reference, params)
    arrival_frame = probe.ego_arrival_frame(conflict_xy)
    # `arrive_with_ego` aims the actor AT the ego, which is co-arrival — i.e. a
    # collision for the log-replay ego, which does not react. That ends the
    # episode on contact and leaves a policy nothing to decide. `arrive_offset_s`
    # turns co-arrival into a NEAR MISS: negative crosses ahead of the ego
    # (it watches someone dart across and must slow), positive crosses behind.
    offset_s = float(params.get("arrive_offset_s", 0.0) or 0.0)
    if offset_s:
        shifted = int(round(arrival_frame + offset_s / max(dt_s, 1e-6)))
        arrival_frame = int(np.clip(shifted, 0, probe.T - 1))
    # AND THE SAME FLOOR THE LAYOUT PATH ALREADY HONOURS. `requires:
    # {min_reaction_s}` was read only where actors are laid out by `arc: auto`,
    # so an actor timed by `arrive_with_ego` got none of it. Where the conflict
    # sits AT the hand-off -- V-10's two hosts whose ego is already merging
    # when it takes over -- co-arrival is contact on the first scored frame:
    # 13c555e68671524f ended at frame 0, `rear_end` with the mainline car,
    # `infra_failure`, nothing to score. The leaf asking for 1.5 s of reaction
    # means the actor must not be at the ego's line before then, whichever way
    # its arrival was solved.
    floor = int(round(int(probe.after_frame) + max(reaction_s, 0.0) / max(dt_s, 1e-6)))
    if reaction_s > 0.0 and arrival_frame < floor:
        logger.info(
            "arrive_with_ego: the ego reaches the conflict %.1f s into the episode, inside the "
            "%.1f s this leaf requires; moving the arrival to frame %d so the ego has the "
            "approach the leaf is about.",
            max(arrival_frame - int(probe.after_frame), 0) * dt_s, reaction_s, floor)
        offset_s += (floor - arrival_frame) * dt_s
        arrival_frame = int(np.clip(floor, 0, probe.T - 1))
    resolved["_solved"] = {
        "conflict_arc_m": round(float(conflict_arc), 2),
        "conflict_xy": [round(float(conflict_xy[0]), 2), round(float(conflict_xy[1]), 2)],
        "ego_arrival_frame": int(arrival_frame),
        "arrive_offset_s": offset_s,
    }
    # Two ways to hit a moment in time: move WHERE the actor starts, or move
    # WHEN it leaves. Which one is available depends on whether the spec pinned
    # the start.
    #
    # `dart_out` always pins it (the actor stands at the verge by definition),
    # and so does any actor carrying an `anchor:` -- a crosswalk crossing is
    # authored as "start at THIS kerb", and back-solving a start arc would slide
    # it upstream off the crossing, which is the one thing the anchor exists to
    # prevent. Both therefore keep their position and get a `conflict_frame`,
    # which `_hold_until_trigger` turns into an ego-distance to wait for.
    #
    # Without a pinned start the actor is free to begin further back along its
    # own reference, and the start arc is solved instead.
    pins_start = template == "dart_out" or bool(params.get("anchor"))
    if pins_start:
        if template == "dart_out":
            # The actor holds one arc position, so the timing lands on the frame
            # it reaches the lane centre.
            resolved["arc"] = round(float(conflict_arc - anchor_arc), 4)
        resolved["conflict_frame"] = int(arrival_frame)
    else:
        speed = float(resolved.get("speed", 0.0))
        if abs(speed) < 1e-6:
            raise AuthoringError(
                f"arrive_with_ego on template {template!r} needs a non-zero speed to solve a "
                f"start arc from"
            )
        start_arc = solve_start_arc_for_conflict(
            reference, conflict_arc=conflict_arc, speed=speed, conflict_frame=arrival_frame,
            dt_s=dt_s,
        )
        # THE ROAD HAS TO REACH THAT FAR BACK. `Polyline.offset` extrapolates
        # past its own ends without complaining, so a start arc upstream of the
        # reference produced a spawn beside the road rather than on it: on
        # 13c555e68671524f the solve wanted -18.2 m and the chain begins at 0,
        # so the car was baked 18.2 m OFF its own path while the path the
        # policy drives still began at the chain -- 2.3 m from the ego. The
        # episode ended on frame 0, rear-ended by its own inserted car,
        # `infra_failure`, nothing to score. Same on 49a0d29c7058501c at
        # -20.0 m. The shortfall is a property of the map (these lanes have no
        # mapped predecessor, so there is no upstream road to come down), so it
        # is a host-selection finding and says so.
        if not reference.covers(float(start_arc), tol=0.5):
            # START WHERE THE ROAD STARTS, AND SLOW DOWN TO STILL ARRIVE WITH
            # THE EGO. The intent is a time, not a speed: what the actor has to
            # do is be at the conflict when the ego is. Refusing here would
            # cost the leaf both of the hosts whose merge the reviewer picked,
            # over a map property (no mapped predecessor) rather than anything
            # about the scenario. The speed that follows is recorded, so a
            # mainline car crawling at 4 m/s is visible in the recipe rather
            # than a surprise in the render.
            available = (float(conflict_arc) if start_arc < 0.0
                         else reference.total - float(conflict_arc))
            travel_s = max(int(arrival_frame) * dt_s, 1e-6)
            slowed = abs(available) / travel_s
            if slowed < MIN_ARRIVAL_SPEED_MS:
                raise AuthoringError(
                    f"arrive_with_ego needs {abs(float(start_arc) - conflict_arc):.0f} m of "
                    f"approach at {speed:.1f} m/s and {reference.name} offers {abs(available):.0f} m, "
                    f"so the actor would have to travel it at {slowed:.1f} m/s to arrive with the "
                    f"ego -- slower than anything on a road. The lane has no mapped predecessor: "
                    f"pick a host whose merge has approach road in the map."
                )
            logger.warning(
                "arrive_with_ego: %s offers %.0f m of approach and this actor wanted %.0f m at "
                "%.1f m/s; starting it at the reference's own beginning and slowing it to "
                "%.1f m/s so it still arrives with the ego. `Polyline.offset` would otherwise "
                "have extrapolated the spawn %.0f m off the road while the path stayed on it.",
                reference.name, abs(available), abs(float(start_arc) - conflict_arc), speed,
                slowed, abs(float(start_arc)))
            resolved["speed"] = round(float(slowed), 3)
            resolved["speed_clamped_by_reference"] = True
            start_arc = 0.0 if start_arc < 0.0 else reference.total
        resolved["arc"] = round(float(start_arc - anchor_arc), 4)
    logger.info(
        "author: solved arrive_with_ego -> %s (template=%s)", resolved["_solved"], template
    )
    return resolved


def _template_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if k not in _CONTROL_KEYS and not k.startswith("_")}


def _pin_asset(registry, asset, nurec_asset_id, track_type, *, declared_type, keep_appearance, name):
    """Resolve a ``registry_key`` into the uid + sha256 the recipe pins.

    Naming an asset by key is a convenience; a recipe pins it by CONTENT, so a
    library that later re-converts the same key cannot silently change what a
    frozen recipe rebuilds.

    Returns ``(asset, nurec_asset_id, track_type, gait)``, where ``gait`` is
    ``(bank dir, bank digest)`` or None. The bank comes from the REGISTRY, not
    from the spec: it used to reach a recipe only by someone hand-editing a
    frozen file, so every re-bake dropped it and the actor silently reverted to
    sliding -- which is what happened to R-3's six pedestrians and R-4's animals
    the moment their hosts were re-authored onto crosswalks.
    """
    entry = registry.get(asset.registry_key)
    if entry.source == "host":
        if not keep_appearance:
            raise AuthoringError(
                f"actor {name!r}: asset {asset.registry_key!r} is source 'host', which has no "
                f"file — it only means 'keep the relocated actor's baked gaussians'. Set "
                f"keep_appearance: true, or name a real asset."
            )
        if asset.dims is None and entry.dims:
            asset.dims = list(entry.dims)
        return asset, nurec_asset_id, (track_type if declared_type else entry.track_type), None
    try:
        resolved = registry.resolve(asset.registry_key)
    except AssetError as exc:
        raise AuthoringError(
            f"actor {name!r}: {exc} Freezing a recipe against an absent file would pin a "
            f"checksum of nothing."
        ) from exc
    asset.uid = asset.uid or resolved.entry.uid
    asset.sha256 = resolved.sha256
    if asset.dims is None:
        asset.dims = list(resolved.entry.dims) if resolved.entry.dims else None
    return (
        asset,
        nurec_asset_id or str(resolved.ply),
        track_type if declared_type else resolved.entry.track_type,
        registry.resolve_gait_bank(asset.registry_key),
    )


def _preflight_render_class(actors, *, world_version: str) -> None:
    """Refuse an insert the render server is known to reject.

    A car2sim reconstruction exposes its actor (deformable) layer only to a
    PEDESTRIAN-class track. A VEHICLE-class insert is accepted by the client,
    the PLY is loaded, and then the server raises

        Failed to insert track_id='navsafe_oncoming_vehicle' with asset_id='.../car_3.ply'

    — after which the episode renders perfectly, minus the car. That failure
    cost a full 8 s render and a frame-by-frame hunt before anyone read the
    server log, so it is caught here, before any GPU is booked.

    The fix is one field: name a render class (``pedestrian``) on the cast slot
    or on the asset's registry row. The RENDER class is what loads the
    gaussians; ``track.type`` stays VEHICLE, so collision and
    driving-direction still score a car.
    """
    if not str(world_version).startswith("car2sim"):
        return
    for name, actor in actors.items():
        if actor.op != "insert" or actor.track_type != "VEHICLE":
            continue
        if actor.semantic_class:
            continue
        raise AuthoringError(
            f"actor {name!r}: a VEHICLE-class insert into a {world_version} recon is "
            f"rejected by the render server — its actor layer is open only to a "
            f"PEDESTRIAN-class track, and the failure happens AFTER the PLY loads, so "
            f"the episode renders without the actor and looks merely empty. Set a render "
            f"class (`render_class: pedestrian` on the cast slot, or `render_class:` on "
            f"the asset's registry row); `track.type` stays VEHICLE for scoring."
        )


def _actor_from_spec(
    probe: HostProbe,
    name: str,
    spec: Dict[str, Any],
    *,
    frames: Frames,
    registry: Optional[AssetRegistry] = None,
    reaction_s: float = 0.0,
) -> Tuple[List[ActorRecipe], List[BakedTrack]]:
    """Bake one authored actor into one (or, for ``group``, several) recipes."""
    op = str(spec.get("op", "insert"))
    authored = dict(spec.get("authored") or {})
    if op == "remove":
        actor = ActorRecipe(
            name=name, op="remove", source_track_id=str(spec.get("source_track_id", ""))
        )
        return [actor], []

    template = str(authored.get("template", ""))
    reference_spec = authored.get("reference", "ego_route")
    reference = probe.resolve_reference(reference_spec)
    # WHICH shape this host got. A cut-in is defined against the ego's own line
    # -- `lateral_to: 0.0` means "the ego's lane" -- so those parameters are
    # meaningless on a merging lane, where the actor is already going where the
    # ego is headed and the timing is an arrival instead. A leaf may carry both
    # sets; the reference that resolved decides which is read.
    chosen = getattr(reference, "chosen_alternative", None) or (
        reference_spec if isinstance(reference_spec, str) else "polyline")
    if str(chosen) != "ego_route":
        for k in ("lateral_to", "transition_m", "cut_in_frame"):
            authored.pop(k, None)
    else:
        for k in ("arrive_with_ego", "arrive_offset_s"):
            authored.pop(k, None)
    # ``arc`` is measured from the hand-off unless the spec names a semantic
    # anchor — then it is measured from where THAT landmark projects onto this
    # actor's reference, and the resolved numbers are recorded so the frozen
    # recipe still rebuilds one exact scene without re-running the detection.
    anchor_name = str(authored.get("anchor", "") or "")
    anchor_solved: Optional[Dict[str, Any]] = None
    if anchor_name == "reference_start":
        # Arc 0 is the REFERENCE's own start, not the hand-off projected onto
        # it. For a lane the hand-off is the right origin — "40 m past where
        # the ego takes over" is a real statement about a road. For a crossing
        # it is not: the hand-off projects to the middle of the crossing, where
        # the ego's own lane is, so `arc: 0` put a pedestrian in the road
        # already and halved the walk (6.3 m of a 15.6 m crossing). A crossing
        # is authored kerb to kerb, and its kerb is its own arc 0.
        anchor_arc = 0.0
        anchor_solved = {"name": "reference_start", "kind": "reference",
                         "reference_arc_m": 0.0}
    elif anchor_name and anchor_name != "handoff":
        landmark = resolve_anchor(probe, anchor_name, after_frame=frames.after_frame)
        anchor_arc = float(reference.project(np.asarray(landmark.xy, np.float64))[0])
        anchor_solved = {
            "name": landmark.name,
            "kind": landmark.kind,
            "route_arc_m": landmark.arc_m,
            "xy": [float(landmark.xy[0]), float(landmark.xy[1])],
            "reference_arc_m": round(anchor_arc, 2),
        }
        logger.info(
            "author: actor %r anchored at %s (route arc %+.1f m -> reference arc %.1f m)",
            name, landmark.name, landmark.arc_m, anchor_arc,
        )
    else:
        anchor_arc = probe.anchor_arc(reference, frames.after_frame)
    # Speed is resolved BEFORE the timing solve, which uses it to place the
    # spawn; the laterals below cannot be, because the arc they are measured at
    # is only known once that solve has run.
    speed_spec = dict(authored)
    # The actor's own length, for the gap solve: a following distance is
    # between BUMPERS, and the solver works in arc along the reference, so it
    # needs to know how much of that arc the car itself occupies.
    _a = dict(spec.get("asset") or {})
    _dims = _a.get("dims")
    if _dims is None and registry is not None and _a.get("registry_key"):
        _e = registry.entries.get(str(_a["registry_key"]))
        _dims = getattr(_e, "dims", None) if _e is not None else None
    if _dims:
        speed_spec["_actor_length"] = float(_dims[0])
    if isinstance(speed_spec.get("speed"), str):
        speed_spec["speed"] = _resolve_speed(
            probe, speed_spec["speed"],
            lo=authored.get("speed_min"), hi=authored.get("speed_max"),
            where=f"actor {name!r} speed")
    resolved = _resolve_timing(
        probe, reference, template, speed_spec, anchor_arc=anchor_arc, dt_s=frames.dt_s,
        reaction_s=reaction_s,
    )
    # Laterals may name road features; the arc they are measured at is known
    # only after the timing solve, so resolve them here.
    absolute_arc = anchor_arc + float(resolved.get("arc", 0.0))
    for field in ("lateral", "start_lateral", "end_lateral"):
        if isinstance(resolved.get(field), str):
            resolved[field] = round(_resolve_lateral(
                probe, reference, resolved[field], absolute_arc,
                where=f"actor {name!r} {field}"), 3)
    params = _template_params(resolved)

    source_track_id = str(spec.get("source_track_id", ""))
    if source_track_id == "auto":
        # Resolve now and RECORD the id: a recipe must rebuild the same scene,
        # so "auto" is an authoring convenience, never a replay-time choice.
        picked = probe.select_relocation_target()
        if picked is None:
            raise AuthoringError(
                f"actor {name!r}: source_track_id is 'auto' but this host offers no actor "
                f"worth re-tasking (needs a VEHICLE with real dims, full validity and real "
                f"motion). Name one explicitly, or insert an asset instead."
            )
        source_track_id = picked["track_id"]
        spec["source_track_id"] = source_track_id
        spec.setdefault("_auto_pick", picked)
        logger.info("author: actor %r auto-selected %s — %s", name, source_track_id, picked["reason"])

    keep_appearance = bool(spec.get("keep_appearance", False))
    asset = AssetRef.from_dict(spec.get("asset"))
    nurec_asset_id = str(spec.get("nurec_asset_id", ""))
    track = spec.get("track") or {}
    track_type = str(track.get("type", "VEHICLE"))
    semantic_class = str(track.get("semantic_class", ""))
    gait = None
    if registry is not None and asset.registry_key:
        asset, nurec_asset_id, track_type, gait = _pin_asset(
            registry, asset, nurec_asset_id, track_type, declared_type=bool(track.get("type")),
            keep_appearance=keep_appearance, name=name,
        )
    if gait is not None:
        _clamp_to_gait(name, gait[0], resolved)
    # The bake is only used to SOLVE the spawn now — frames 1..T-1 are thrown
    # away, because a reactive actor will not follow them. So the guard that
    # refuses a trajectory running off the end of its reference is checking a
    # path that no longer exists: a car doing 8 m/s for 20 s needs 160 m of
    # lane whether or not it will ever drive them, and refusing on that basis
    # rejects perfectly good spawns. What still has to be on the reference is
    # frame 0, which `bake_actor_state` places by projection either way.
    bake_kwargs = dict(
        ego_z_to_ground_m=probe.ego_z_to_ground_m,
        keep_appearance=keep_appearance,
        strict_reference=bool(spec.get("strict_reference", False)),
    )

    def _documented(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = dict(resolved)
        doc.pop("_solved", None)
        doc["template"] = template
        doc["reference"] = reference_spec if isinstance(reference_spec, str) else "polyline"
        if anchor_solved is not None:
            doc["anchor_solved"] = dict(anchor_solved)
        if resolved.get("_solved"):
            doc["solved"] = resolved["_solved"]
        if reference.name != "ego_route":
            # Freeze the reference too: a lane the map renumbers later must not
            # silently change the trajectory this recipe rebuilds.
            doc["reference_polyline"] = [
                [round(float(x), 3), round(float(y), 3)] for x, y in reference.xy
            ]
        if extra:
            doc.update(extra)
        return doc

    motion = build_motion(
        template, T=frames.T, dt_s=frames.dt_s, anchor_arc=anchor_arc, params=params
    )
    baked = bake_actor_state(reference, motion, **bake_kwargs)
    policy = _policy_from_template(template, params, reference, anchor_arc, asset,
                                   track_type=track_type)
    # Every actor the solve pinned in place, not just dart_out. A crosswalk
    # crossing without this leaves at frame 0: R-4's animals were across and
    # standing on the far kerb 5 s before the ego arrived, so the leaf rendered
    # an empty road and scored a clean run -- the same failure the dart_out
    # trigger was written for, one template over.
    if "conflict_frame" in resolved and "trigger" not in policy:
        held = _hold_until_trigger(probe, baked, policy, resolved, frames.dt_s, name=name)
        if held is not None:
            policy["trigger"] = {"ego_distance_m": round(held, 1)}
            logger.info("%s %s: holds until the ego is %.1f m away (to be at the "
                        "conflict point by frame %s)", template, name, held,
                        resolved.get("conflict_frame"))
    actor = ActorRecipe(
        name=name,
        op=op,
        source_track_id=source_track_id,
        keep_appearance=keep_appearance,
        track_type=track_type,
        semantic_class=semantic_class,
        nurec_asset_id=nurec_asset_id,
        nurec_pose_bank=str(gait[0]) if gait else "",
        pose_bank_sha256=gait[1] if gait else "",
        leaf_variety_slot=str(spec.get("leaf_variety_slot", "")),
        asset=asset,
        authored=_documented(),
        spawn=_spawn_from(baked, asset),
        policy=policy,
    )
    return [actor], [baked]


def _spawn_from(baked: BakedTrack, asset) -> Dict[str, Any]:
    """The actor's frame-0 pose — all a reactive actor needs to start.

    The template still solves the placement; what changed is how much of the
    solve is kept. Frame 0 is the initial condition; frames 1..T-1 were a
    prediction of what a non-reactive actor would have done, and a reactive
    one will not do it.
    """
    state = baked.state
    pos = np.asarray(state["position"], np.float64)[0]
    vel = np.asarray(state["velocity"], np.float64)[0]
    out: Dict[str, Any] = {
        "position": [round(float(v), 3) for v in pos],
        "heading": round(float(np.asarray(state["heading"], np.float64)[0]), 5),
        "velocity": [round(float(v), 3) for v in vel],
    }
    if asset is not None and getattr(asset, "dims", None):
        out["length"], out["width"] = (float(asset.dims[0]), float(asset.dims[1]))
    return out


#: Track types driven by social force rather than IDM. IDM is longitudinal
#: car-following on a lane; it has no meaning for someone walking. Animals are
#: tracked as PEDESTRIAN (the taxonomy's VRU class), so this covers them too.
_WALKS_RATHER_THAN_DRIVES = {"PEDESTRIAN"}


def _policy_from_template(template: str, params: Dict[str, Any], reference,
                          anchor_arc: float, asset,
                          track_type: str = "VEHICLE") -> Dict[str, Any]:
    """Which controller a template's intent becomes.

    The templates were trajectory generators; they are now initial-condition
    generators plus this mapping. The correspondence is close because the
    parameters were always about intent — ``dynamic``'s ``speed`` was a desired
    speed, ``dart_out``'s ``end_lateral`` was a goal — it is only the middle
    step, integrating them into a fixed path, that is gone.

    The template alone does not decide the controller: ``dynamic`` means
    "travel along this reference", and how you travel depends on what you ARE.
    A cyclist in a bike lane follows the lane (IDM); a pedestrian on a crossing
    walks to the far kerb (social force). Handing a walker IDM gives them a
    car-following model with no car in front and no lane to keep — they stand
    still or slide down the centreline.
    """
    speed = abs(float(params.get("speed", 0.0) or 0.0))
    if template == "static":
        return {"kind": "static"}
    if template == "dynamic" and str(track_type).upper() in _WALKS_RATHER_THAN_DRIVES:
        # Walk the reference end to end. On a `crosswalk_chain` that is kerb to
        # kerb, which is the whole point: the crossing's own geometry decides
        # where the actor starts, which way it faces and where it stops, rather
        # than a lateral offset guessed against the ego's route.
        signed = float(params.get("speed", 0.0) or 0.0)
        lateral = float(params.get("lateral", 0.0))
        goal_arc = reference.total if signed >= 0 else 0.0
        gx, gy, _gz, _gh = reference.offset(np.array([goal_arc]), np.array([lateral]))
        policy: Dict[str, Any] = {
            "kind": "social_force",
            "goal": [round(float(gx[0]), 3), round(float(gy[0]), 3)],
            "desired_speed": speed,
        }
        trigger = params.get("trigger_ego_distance_m")
        if trigger is not None:
            policy["trigger"] = {"ego_distance_m": float(trigger)}
        return policy
    if template == "dynamic":
        # THE LINE IT RIDES, NOT THE LANE'S CENTRE. The spawn is placed at the
        # actor's `lateral`, and handing the policy the raw centreline made the
        # first controlled step a jump back to the middle: R-2's cyclists were
        # spawned 1.2-1.4 m out against the kerb and snapped into the lane on
        # frame 2 (00c1e4eb4a045f20, 20cc0fdb7e2d5c3f). A rider holds its line
        # down the road, so the path is the reference offset by the same
        # amount, sampled at the reference's own resolution.
        lateral = float(params.get("lateral", 0.0))
        # A CUT-IN is the same path with a lateral that MOVES. `lateral_to`
        # names where the actor ends up -- 0.0 is the reference's own line, so
        # on `ego_route` that is the ego's lane -- and `transition_m` how much
        # road it takes to get there, centred on the conflict point so the lane
        # change lands where the two are about to meet rather than somewhere up
        # the road. Without this the offset is a constant and the actor drives
        # a parallel lane forever: V-10 could stage a car beside the ego but
        # never one taking its lane, which is the manoeuvre the leaf is about.
        lateral_to = params.get("lateral_to")
        arcs = np.asarray(reference.arc, np.float64)
        if lateral_to is not None:
            span = max(float(params.get("transition_m", 20.0)), 1e-6)
            solved = params.get("_solved") or {}
            if params.get("_cut_in_arc") is not None:
                centre = float(params["_cut_in_arc"])
            elif solved.get("conflict_arc_m") is not None:
                centre = float(solved["conflict_arc_m"])
            else:
                centre = float(anchor_arc + float(params.get("arc", 0.0)))
            # A smoothstep, not a ramp: a linear lateral has a corner at each
            # end, and the heading the sim derives from consecutive points
            # jumps there -- the car visibly snaps into the lane.
            u = np.clip((arcs - (centre - span / 2.0)) / span, 0.0, 1.0)
            lat = lateral + (float(lateral_to) - lateral) * (u * u * (3.0 - 2.0 * u))
            px, py, _pz, _ph = reference.offset(arcs, lat)
            path = np.stack([px, py], axis=1)
        elif abs(lateral) > 1e-6:
            px, py, _pz, _ph = reference.offset(arcs, np.full_like(arcs, lateral))
            path = np.stack([px, py], axis=1)
        else:
            path = np.asarray(reference.xy, np.float64)
        policy: Dict[str, Any] = {
            "kind": "idm",
            "v0": speed,
            "blind_to_ego": bool(params.get("blind_to_ego", False)),
            # The NAME of the reference is a query against this host's map;
            # the resolved polyline is what makes the recipe rebuildable.
            "path_polyline": [[round(float(x), 3), round(float(y), 3)]
                              for x, y in path],
        }
        trigger = params.get("trigger_ego_distance_m")
        if trigger is not None:
            policy["trigger"] = {"ego_distance_m": float(trigger)}
        return policy
    if template == "dart_out":
        arc = anchor_arc + float(params.get("arc", 0.0))
        end_lateral = float(params.get("end_lateral", 0.0))
        gx, gy, _gz, _gh = reference.offset(np.array([arc]), np.array([end_lateral]))
        policy: Dict[str, Any] = {
            "kind": "social_force",
            "goal": [round(float(gx[0]), 3), round(float(gy[0]), 3)],
            "desired_speed": speed,
        }
        trigger = params.get("trigger_ego_distance_m")
        if trigger is not None:
            # An AUTHOR'S OVERRIDE. Left alone, the trigger is derived from
            # `arrive_offset_s` (see _dart_out_trigger) so every dart-out on
            # every host gets one; typing a distance here says "this leaf wants
            # a particular staggering", not "otherwise there is none".
            policy["trigger"] = {"ego_distance_m": float(trigger)}
        return policy
    raise AuthoringError(
        f"template {template!r} has no reactive policy. There are three templates "
        f"because there are three controllers — static, idm, social_force — and a "
        f"fourth would need a fourth driver, not another entry here."
    )


def _hold_until_trigger(probe, baked, policy, resolved, dt_s: float, *, name: str):
    """When an actor with a pinned start should leave, stated as an ego distance.

    ``arrive_with_ego`` / ``arrive_offset_s`` are solved here into
    ``conflict_frame`` — the frame the actor is wanted at the ego's line — and
    were then dropped on the floor, because a reactive actor has no frame
    schedule to carry them. What the actor actually did was leave at frame 0:
    R-4's elephant had 1.7 m to walk and was standing at its goal 1 s in, while
    the ego was not due for another 12 s. Every dart-out leaf rendered an empty
    road and scored a clean run.

    A distance is the replay-time form of the same intent. Back the actor's own
    travel time off ``conflict_frame``, ask where the ego was at that moment,
    and let the actor hold until the ego is that close. Unlike a frame number
    this survives a policy that drives faster or slower than the logged ego —
    which is the whole reason the actor is reactive.

    Returns:
        The trigger distance in metres, never below
        :data:`MIN_TRIGGER_DISTANCE_M`, or None when the intent cannot be
        expressed as one (no solved arrival, or the actor cannot make it in
        time even leaving at frame 0).
    """
    conflict_frame = resolved.get("conflict_frame")
    speed = float(policy.get("desired_speed", 0.0) or 0.0)
    if conflict_frame is None or speed <= 0.0:
        return None
    spawn = np.asarray(baked.state["position"], np.float64)[0, :2]
    # To the CONFLICT POINT, not to the goal. `arrive_with_ego` means "be where
    # the ego is passing, when it passes" -- the goal is wherever the actor
    # eventually stops, which for a crosswalk is the far kerb several metres
    # PAST the ego's lane. Timing on the goal made the walk look impossible: a
    # pedestrian needing 14 s to reach the far kerb was refused against an ego
    # 2 s away, when reaching the ego's own lane takes a fraction of that. The
    # goal remains the fallback for a solve that produced no conflict point.
    solved = resolved.get("_solved") or {}
    target = solved.get("conflict_xy")
    target = (np.asarray(target, np.float64) if target is not None
              else np.asarray(policy["goal"], np.float64))
    travel_s = float(np.linalg.norm(target - spawn)) / speed
    start_frame = int(round(int(conflict_frame) - travel_s / max(dt_s, 1e-6)))
    if start_frame <= 0:
        # Not "it arrives early" -- the opposite. Leaving at frame 0 is the
        # earliest it can go, and it still reaches the ego's line `late_s`
        # AFTER the ego has passed, so it crosses behind rather than in front.
        # That is a property of the HOST (the conflict sits too few seconds
        # into the route for anyone to walk into it), not of this solve, and no
        # trigger distance can fix it -- hence None, meaning "leave at frame 0".
        late_s = travel_s - int(conflict_frame) * dt_s
        logger.warning(
            "arrive_with_ego %s: needs %.1f s to reach the ego's line, which the ego passes at "
            "%.1f s (frame %d). Even leaving at frame 0 it gets there %.1f s late and crosses "
            "BEHIND the ego, so this host stages no conflict. Raise `speed`, start it closer to "
            "the ego's lane, or pick a crossing further along the route.",
            name, travel_s, int(conflict_frame) * dt_s, int(conflict_frame), late_s)
        return None
    k = int(np.clip(start_frame, 0, probe.T - 1))
    held = float(np.linalg.norm(np.asarray(probe.ego_position)[k, :2] - spawn))
    if held < MIN_TRIGGER_DISTANCE_M:
        logger.info(
            "arrive_with_ego %s: the solved hold puts the ego %.1f m away when the actor "
            "steps off, which is a spawn in the bumper rather than a conflict; holding to "
            "%.1f m instead, so it leaves a little earlier.",
            name, held, MIN_TRIGGER_DISTANCE_M)
        held = MIN_TRIGGER_DISTANCE_M
    return held


def _layout_window(probe, at: str, *, span_m: float) -> Tuple[float, float]:
    """Where on the ego route a bank of actors is laid out.

    ``at`` is ``straight`` (the host's own straight stretch, the default) or one
    or more anchor names with ``|`` fallbacks, e.g. ``crosswalk_1|straight``.
    An anchor gives a POINT, so the window is centred on it — a crowd crossing
    at a crosswalk straddles it rather than starting there.

    Raises:
        AuthoringError: every alternative is missing on this host. That is a
            host-selection finding, so the message says what the host does
            offer rather than quietly falling back to somewhere else.
    """
    attempts = []
    for alt in [v.strip() for v in str(at).split("|") if v.strip()]:
        if alt == "handoff":
            # Beside the ego, not "wherever this route happens to straighten".
            # `straight` scans FORWARD from the hand-off for a junction-free
            # stretch, and on 2575048779565f0b that stretch began 36 m along,
            # so a bank meant to share the ego's carriageway was laid out 80 m
            # up the road. A leaf whose event is the ego's relationship to the
            # actor has to start where the ego is.
            # From the ego onward, not a `span_m` slice of it: `span_m` is how
            # far the GROUP is strung out and the margin comes out of the same
            # window, so a window exactly one span long would squeeze the two
            # together by whatever the margin was.
            return (probe.anchor_arc(probe.ego_route, probe.after_frame),
                    probe.ego_route.total)
        if alt == "straight":
            window = probe.straight_window()
            if window is None:
                attempts.append("straight: this route never holds its heading past the hand-off")
                continue
            return window
        try:
            landmark = resolve_anchor(probe, alt, after_frame=probe.after_frame)
        except PlacementError as exc:
            attempts.append(f"{alt}: {exc}".split(". It offers")[0])
            continue
        centre = probe.anchor_arc(probe.ego_route, probe.after_frame) + float(landmark.arc_m)
        half = 0.5 * float(span_m)
        logger.info("layout: centred on %s (route arc %+.1f m)", landmark.name, landmark.arc_m)
        return (max(centre - half, 0.0), min(centre + half, probe.ego_route.total))
    raise AuthoringError(
        f"layout.at {at!r}: none of these resolve on this host — "
        + "; ".join(attempts)
        + ". Pick a host that carries the landmark (navsafe qualify / navsafe anchors), or "
          "end the chain in `straight`, which any host with a straight stretch answers."
    )


def _resolve_layout(probe, spec: Dict[str, Any]) -> Dict[str, float]:
    """Assign an arc to every actor whose placement says ``auto``.

    Where a group of dart-outs goes is a property of the HOST, not of the
    author: they belong on the stretch the ego drives straight, spaced so each
    is its own encounter. Hand-picking the arcs made the spec host-specific
    (46/53/60/... only meant anything on one clip) and made "spread them out"
    versus "pack them together" a retype rather than a parameter.

    ``layout: {span_m: 35, margin_m: 8}`` — the group occupies ``span_m`` of the
    straight stretch, starting ``margin_m`` into it so the first one is not on
    top of the hand-off. Actors keep their spec order.

    ``layout: {at: crosswalk_1|straight}`` moves that window onto a named
    landmark instead, with the usual ``|`` fallbacks. This is what lets one
    R-3 / R-4 rule say "cross at the crosswalk where the host has one, else
    anywhere on the ego's route": pedestrians at a marked crossing and
    pedestrians mid-block are different tests, and which one a host can offer
    is a fact about the host, not something to retype per clip.
    """
    layout = dict(spec.get("layout") or {})
    autos = [n for n, a in (spec.get("actors") or {}).items()
             if str((a.get("authored") or {}).get("conflict_arc", "")) == "auto"
             or str((a.get("authored") or {}).get("arc", "")) == "auto"]
    if not autos:
        return {}
    span = float(layout.get("span_m", 35.0))
    window = _layout_window(probe, str(layout.get("at", "straight")), span_m=span)
    # HOW FAR AHEAD IS A TIME, NOT A DISTANCE. `margin_m: 22` is the same 22 m
    # on every host, so the ego's approach to a STATIC insert was 1.5 s where
    # it drives at 14 m/s and 3.7 s where it drives at 6 m/s — and I-3 declares
    # `min_reaction_s: 2.0`. That declaration went unenforced (nothing read
    # `requires.min_reaction_s`; the gate that once did measured a baked
    # trajectory against a log-replay ego and went when both did), so the
    # scenario silently became a different test on every clip: on
    # 76da778ff251508d the ego hit the wreck at frame 27 with no room to have
    # done anything else.
    #
    # The metres stay as a floor — a slow ego still needs enough road to see
    # past its own bonnet — and the seconds raise it on a fast one.
    margin = float(layout.get("margin_m", 8.0))
    reaction_s = float((spec.get("requires") or {}).get("min_reaction_s", 0.0))
    margin_s = float(layout.get("margin_s", reaction_s))
    if margin_s > 0.0:
        ego_speed = float(probe.ego_cruise_speed())
        needed = margin_s * ego_speed
        if needed > margin:
            logger.info(
                "layout: margin %.0f m is %.1f s at this host's %.1f m/s; raising to %.0f m "
                "for the %.1f s the leaf asks for", margin, margin / max(ego_speed, 0.1),
                ego_speed, needed, margin_s)
            margin = needed
    min_spacing = float(layout.get("min_spacing_m", MIN_GROUP_SPACING_M))
    gaps = max(len(autos) - 1, 1)
    needed = min_spacing * (len(autos) - 1)
    window_m = window[1] - window[0]
    # SPACING IS THE POINT, MARGIN IS AN ALLOWANCE. Ranking them the other way
    # is what shipped R-4 with four animals inside one metre: `margin_m: 26`
    # against this host's 25 m straight stretch left -1 m of room, the old code
    # clamped the span to a 1 m floor, and the group arrived 0.7 s apart at the
    # same point. It warned, and a warning inside a bake that otherwise
    # succeeds is not a signal. So the margin yields first, and a group that
    # still cannot be spread is refused rather than piled — a bank of dart-outs
    # that overlap is not the leaf, it is one obstacle wearing four assets.
    if window_m < needed:
        raise AuthoringError(
            f"layout: {len(autos)} actors need {needed:.0f} m to stay {min_spacing:.0f} m "
            f"apart, and this host's straight stretch is {window_m:.0f} m. Pick a host with "
            f"a longer straight (navsafe qualify), lower `layout.min_spacing_m` if they are "
            f"meant to be a cluster, or author fewer actors.")
    if margin > window_m - needed:
        logger.warning(
            "layout: margin_m %.0f would leave less than the %.0f m this group needs on a "
            "%.0f m straight stretch; using %.0f m instead", margin, needed, window_m,
            max(window_m - needed, 0.0))
        margin = max(window_m - needed, 0.0)
    start = window[0] + margin
    span = min(span, window[1] - start)
    step = span / gaps
    # Arcs on the EGO ROUTE. The two fields that can say `auto` read them in
    # different frames, so the conversion belongs at the point of use, not
    # here — see the loop in `author_recipe`.
    arcs = {n: round(start + i * step, 2) for i, n in enumerate(autos)}
    logger.info("layout: %d actors over %.0f m of straight stretch (%.1f m apart): %s",
                len(autos), span, step, arcs)
    return arcs


def author_recipe(
    sd: dict,
    spec: Dict[str, Any],
    *,
    event_radius_m: float = DEFAULT_EVENT_RADIUS_M,
    registry: "AssetRegistry | str | None" = None,
    use_registry: bool = True,
) -> Tuple[Recipe, Dict[str, Any]]:
    """Resolve an authoring spec against a host scenario into a Recipe.

    The returned recipe is complete but **not yet frozen**: its checksums are
    stamped by :func:`~navsafe.benchmark.editing.recipe.freeze.freeze_recipe`,
    which is also what makes the file immutable.

    Args:
        sd: the host ScenarioDescription, in its own ego frame-0 coordinates.
        spec: the authoring file — the same shape as a recipe, but with
            ``authored`` parameters instead of baked ``state``.
        event_radius_m: how close the log-replay ego must come for the
            scenario-defining event to count as met.

    Returns:
        ``(recipe, diagnostics)``. The diagnostics carry the placement numbers a
        reviewer sanity-checks (lane width, spawn arc, closing speed) and are
        what the review card prints.

    Raises:
        AuthoringError: the spec cannot be resolved against this host.
        PlacementError / TemplateError / BakeError: from the layer that failed.
    """
    if not use_registry:
        registry = None
    elif registry is None or isinstance(registry, (str, Path)):
        registry = AssetRegistry.load(registry)
    ego_spec = EgoSpec.from_dict(spec.get("ego") or {})
    probe = HostProbe(sd, ego_z_to_ground_m=ego_spec.z_to_ground)
    # ``ego_lane_chain`` and every ``arc`` are measured from the hand-off.
    probe.after_frame = int(ego_spec.replay_frames)
    frames = Frames.from_dict(probe.frames_dict(after_frame=ego_spec.replay_frames))

    # The ego z is ground-referenced at the loader, so the correct drop is 0
    # and anything else is recon-road drift. Check that against the host's own
    # boxes rather than trusting the number in the spec.
    measured = probe.measure_road_drift()
    if measured and abs(measured["residual_m"] - ego_spec.z_to_ground) > ROAD_DRIFT_TOL_M:
        logger.warning(
            "author: ego.z_to_ground is %.2f m, but this host's own perception boxes put the "
            "road %.2f m below the ego z (%d samples; %.2f m if the box z is the base, not the "
            "centre). Since the loader already reports the ego z at the ground, a residual this "
            "size is either recon-road drift — calibrate it per scene in the "
            "NUREC_GROUND_Z_CALIB registry — or this source's rear-axle pose not sitting on the "
            "road. An inserted base-origin asset will float or sink by the difference; a "
            "kept-appearance relocate is unaffected, because the renderer ground-clamps it.",
            ego_spec.z_to_ground,
            measured["residual_m"],
            measured["samples"],
            measured["if_box_is_base_m"],
        )

    actors: Dict[str, ActorRecipe] = {}
    diagnostics: Dict[str, Any] = {
        "host": {
            "T": frames.T,
            "dt_s": frames.dt_s,
            "after_frame": frames.after_frame,
            "route_length_m": round(probe.ego_route.total, 2),
            "lanes": len(probe.lane_ids()),
            "z_to_ground_declared_m": ego_spec.z_to_ground,
            "road_drift_measured": measured,
        },
        "actors": {},
        "registry": str(getattr(registry, "path", "")) if registry else None,
    }
    baked_positions: Dict[str, np.ndarray] = {}

    raw_actors = spec.get("actors") or {}
    provenance = str(spec.get("provenance", "constructed"))
    if not raw_actors and provenance != "mined":
        raise AuthoringError(
            "authoring spec has no actors. A leaf judged on the ROAD itself is written "
            "`provenance: mined` (and must carry `selection` evidence); anything else "
            "with no actors is an unfinished spec.")
    auto_arcs = _resolve_layout(probe, spec)
    # `_resolve_layout` answers in ego-route arcs, and the two fields that can
    # say `auto` do not read that frame the same way:
    #   `arc`          is measured FROM THE HAND-OFF (`probe.anchor_arc` is
    #                  added to it later), so the hand-off comes off here or it
    #                  is counted twice. That stayed invisible while every such
    #                  leaf laid out on the `straight` window of a host whose
    #                  hand-off sits a few metres from arc 0; a window anchored
    #                  ON the hand-off doubles it outright.
    #   `conflict_arc` is an absolute arc on the actor's own reference and is
    #                  used as given.
    _handoff_arc = probe.anchor_arc(probe.ego_route, frames.after_frame)
    for _name, _arcv in auto_arcs.items():
        _auth = raw_actors[_name].setdefault("authored", {})
        for _field, _value in (("conflict_arc", _arcv),
                               ("arc", round(_arcv - _handoff_arc, 2))):
            if str(_auth.get(_field, "")) == "auto":
                _auth[_field] = _value
    for name, actor_spec in raw_actors.items():
        try:
            built, baked = _actor_from_spec(
                probe, str(name), dict(actor_spec), frames=frames, registry=registry,
                reaction_s=float((spec.get("requires") or {}).get("min_reaction_s", 0.0)),
            )
        except (PlacementError, ValueError) as exc:
            raise type(exc)(f"actor {name!r}: {exc}") from exc
        for actor, track in zip(built, baked or [None] * len(built)):
            actors[actor.name] = actor
            if track is None:
                continue
            section = probe.cross_section(
                _authored_reference(probe, actor.authored),
                float(track.diagnostics["spawn_arc_m"]),
            )
            diagnostics["actors"][actor.name] = {
                **track.diagnostics,
                "op": actor.op,
                "cross_section": section.describe(),
                "lane_width_m": section.lane_width_m,
            }
            # The SPAWN point, which is all the recipe now determines. The
            # gates that needed a whole trajectory (closest approach, reaction
            # window, arrival window) are gone with it: a reactive actor's path
            # is a function of the ego's behaviour, so there is nothing to
            # measure here that would still be true at run time. What survives
            # is `evasion_lanes`, which asks a question about the ROAD at the
            # spawn point, and that road does not move.
            baked_positions[actor.name] = np.asarray(
                [actor.spawn.get("position", (0.0, 0.0, 0.0))], np.float64
            )

    # A static insert cannot defend itself against the logged traffic, so the
    # leaf may ask for that traffic to be taken out between the ego and the
    # event. Resolved HERE because it needs the actors' arcs, and RECORDED as
    # explicit `op: remove` entries because a frozen recipe has to rebuild one
    # exact scene rather than re-run a detection that a later map or a later
    # probe could answer differently.
    clear = dict(spec.get("clear_path") or {})
    if clear and baked_positions:
        arcs = [probe.ego_route.project(np.asarray(p_, np.float64)[0, :2])[0]
                for p_ in baked_positions.values()]
        handoff = probe.anchor_arc(probe.ego_route, frames.after_frame)
        window = (handoff, max(arcs) + float(clear.get("margin_m", 10.0)))
        blocked = probe.vehicles_over_arc(
            *window, corridor_m=float(clear.get("corridor_m", 4.5)))
        for i, hit in enumerate(blocked, 1):
            name = f"cleared_{i}"
            actors[name] = ActorRecipe(name=name, op="remove",
                                       source_track_id=hit["track_id"])
            diagnostics["actors"][name] = {"op": "remove", "cleared": hit}
        logger.info(
            "clear_path: removed %d logged vehicle(s) from the ego's corridor over "
            "route arc %.0f-%.0f m: %s", len(blocked), window[0], window[1],
            ", ".join(f"{h['track_id']} at {h['arc_m']:.0f} m" for h in blocked) or "none")
        diagnostics["clear_path"] = {
            "window_arc_m": [round(window[0], 2), round(window[1], 2)],
            "corridor_m": float(clear.get("corridor_m", 4.5)),
            "removed": blocked,
        }

    # Three gates used to live here — closest approach, reaction window, and
    # the assumed ego arrival window — and all three were measurements of a
    # BAKED trajectory against the LOG-REPLAY ego. Neither side of that
    # comparison survives: the actor's path is now produced at run time by a
    # policy reacting to the ego under test, and the ego under test is not the
    # logged one.
    #
    # Keeping them would have meant scoring a scenario against a rollout that
    # will not happen. They move downstream instead, onto the episode trace,
    # where "how close did it actually come" is a fact rather than a forecast.
    # That is a real loss of an early warning: `min_clearance_m` was added
    # after three recipes froze clean with 0.00 / 0.14 / 0.40 m closest
    # approaches, and it now costs a GPU render to notice that again.
    #
    # `evasion_lanes` stays, because it asks about the ROAD at the spawn point
    # and the road does not react to anything.
    defining = [n for n in (spec.get("pair", {}).get("e0_removes") or []) if n in baked_positions]
    requires = dict(spec.get("requires") or {})
    need_evasion = int(requires.get("evasion_lanes", 0))
    for name in (defining or list(baked_positions)):
        pos = baked_positions.get(name)
        if pos is None:
            continue
        entry = diagnostics["actors"].setdefault(name, {})
        if need_evasion:
            ref = _authored_reference(probe, actor.authored)
            spawn_arc = probe.ego_route.project(pos[0, :2])[0]
            lanes = probe.evasion_lanes(probe.ego_route, spawn_arc)
            entry["evasion_lanes"] = lanes
            if len(lanes) < need_evasion:
                raise AuthoringError(
                    f"actor {name!r}: this host offers {len(lanes)} same-direction lane(s) to "
                    f"swerve into at the spawn point, but the leaf needs >= {need_evasion}. "
                    f"On a single carriageway the episode is unavoidable at spawn, so it "
                    f"tests nothing. Pick a multi-lane host."
                )

    _preflight_render_class(actors, world_version=str((spec.get("host") or {}).get(
        "world_version", "")))

    recipe = Recipe(
        recipe_id=str(spec.get("recipe_id", "")),
        leaf=str(spec.get("leaf", "")),
        scenario=str(spec.get("scenario", "")),
        nuplan_type=spec.get("nuplan_type"),
        # Both were silently dropped: a constructed recipe took the default
        # provenance whatever the spec said, and a mined-then-inserted leaf
        # lost its selection evidence the moment an actor appeared.
        provenance=provenance,
        selection=dict(spec.get("selection") or {}),
        host=HostSpec.from_dict(spec.get("host") or {}),
        frames=frames,
        ego=ego_spec,
        route_polyline=[
            [round(float(x), 3), round(float(y), 3)] for x, y in probe.ego_route.xy
        ],
        actors=actors,
        background_traffic=str(spec.get("background_traffic", "keep")),
        pair=dict(spec.get("pair") or {}),
        review=dict(spec.get("review") or {"status": "pending"}),
    )
    recipe.validate()
    return recipe, diagnostics
