# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""PDM-Closed planner — orchestration of route, proposal, score, select.

Per ``plan(...)`` call, the planner runs the canonical PDM-Closed pipeline:

1. Extract a route-conditioned centerline polyline.
2. Read the active lane's speed limit (used to scale each IDM
   policy's target velocity per ``tuplan_garage``).
3. Predict every non-ego agent forward at constant velocity.
4. Generate the ``3 × 5 = 15`` proposal grid; forward-simulate each
   one with kinematic bicycle + LQR (or pure pursuit on opt-in).
5. Score every proposal with the PDMS metric family (no EC).
6. Select the best by ``argmax``.
7. **Emergency-brake guard**: brake when the selected proposal collides
   with any agent within :attr:`PDMConfig.infraction_horizon_s` (default
   ``2.0 s``), or when :func:`score_brake_reason` names a reason. Both
   are gated on :attr:`PDMConfig.emergency_brake_max_ego_speed`.
   :func:`score_brake_reason` is the single home of the score brake's
   rationale — read it there, not here. A third, ungated red-light floor
   (guard 6r) is retained only as the explicit NexusSim extension
   :attr:`PDMConfig.red_light_brake_floor`; it is disabled by default because
   CaRL has no such emergency guard. Red connectors still enter IDM as
   stationary leads through :attr:`PDMConfig.traffic_light_obstacles`.

Output: ``(num_trajectory_poses, 2)`` ego-frame
``[lateral, forward]`` waypoints. The default is ``(8, 2)`` at 4 s
horizon (NexusSim's adapter contract). Set
``cfg.trajectory_horizon_s = 8.0`` and ``cfg.num_trajectory_poses
= 16`` for the upstream horizon while retaining the adapter's 0.5 s
downsampling; it is not the upstream 10 Hz output contract.

Determinism: no RNG. The same
``(scenario_data, ego_state, frame_id, cfg)`` always produces the
same plan output.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from navsafe.core.agent_forecast import nearest_k_admitted
from navsafe.evaluation.utils.lane_proxy import LaneProxy, build_lanes_from_scenario
from navsafe.scenario.scenario_description import scenario_dt_seconds
from navsafe.policy.state.pdm_closed_planner.config import PDMConfig
from navsafe.policy.state.pdm_closed_planner.forward_sim import (
    AgentPrediction,
    predict_agents_constant_velocity,
    predict_agents_from_live_states,
    predict_agents_log_replay,
    simulate_proposal,
)
from navsafe.policy.state.pdm_closed_planner.proposals import (
    SAFETY_BRAKE_PROPOSAL,
    STOCK_PROPOSAL,
    SYNTHETIC_BRAKE_EXECUTION,
    Proposal,
    build_brake_proposal,
    generate_proposals,
)
from navsafe.policy.state.pdm_closed_planner.route import extract_route_centerline
from navsafe.policy.state.pdm_closed_planner.scoring import PDMScorer
from navsafe.policy.state.pdm_closed_planner.traffic_lights import (
    RED_LIGHT_HOLD_DISTANCE_M,
    RED_LIGHT_HOLD_GRACE_S,
    RedLightStopLine,
    find_red_light_stop_lines,
    red_light_obstacles,
    traffic_light_state_evidence,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------


@dataclass
class PDMPlanResult:
    """Full plan() output."""

    trajectory: np.ndarray
    best_idx: int
    scores: Optional[np.ndarray]
    proposals: List[Proposal]
    emergency_brake_triggered: bool
    route_source: str
    # Longitudinal velocity attached to each returned future pose. Upstream
    # trajectories carry this state explicitly; reconstructing it from XY
    # distance mistakes lateral-offset path projection for forward speed and
    # loses the zero-velocity contract of PDMEmergencyBrake.
    trajectory_speeds_mps: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Emergency brake
# ----------------------------------------------------------------------


def emergency_brake_trajectory(
    ego_speed: float,
    cfg: PDMConfig,
    *,
    ego_longitudinal_accel: float = 0.0,
) -> np.ndarray:
    """Faithful port of ``PDMEmergencyBrake._generate_trajectory``."""
    n = cfg.num_trajectory_poses
    dt = cfg.output_dt

    out = np.zeros((n, 2), dtype=np.float64)
    if ego_speed > 0.2:
        command = float(np.clip(-10.0 * ego_speed,
                                -cfg.emergency_brake_decel, 2.4))
        correcting_velocity = 1.1 * (ego_speed + command)
    else:
        command = 4.0 * (-ego_speed) - ego_longitudinal_accel
        correcting_velocity = float(np.clip(
            command, -cfg.emergency_brake_decel, 2.4))
    for i in range(n):
        out[i, 1] = correcting_velocity * ((i + 1) * dt)
    return out


def emergency_brake_execution_trajectory(
    trajectory: np.ndarray,
) -> np.ndarray:
    """Project upstream emergency-brake states onto NexusSim's XY contract.

    ``emergency_brake_trajectory`` intentionally preserves upstream's pose
    correction signal byte-for-byte.  Those poses are *not* a path to drive:
    upstream gives every one the ego's current heading and zero velocity,
    acceleration, and steering.  NexusSim's policy contract carries only XY,
    so a generic path consumer would infer reverse motion and a pi-radian
    heading from the correction poses.  The faithful XY-only representation
    of the dynamic state is therefore a stationary trajectory.  Controller
    execution still applies braking because the accompanying target-speed
    profile is zero; teleport execution holds the zero-velocity state.

    Keeping this projection separate is important: planner scoring and the
    emergency-brake formula retain the exact upstream poses, while only the
    adapter/execution boundary removes semantics XY cannot encode.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(
            "trajectory must have shape (N, >=2), "
            f"got {traj.shape}"
        )
    return np.zeros_like(traj, dtype=np.float64)


def result_execution_trajectory(
    result: PDMPlanResult,
    cfg: PDMConfig,
) -> np.ndarray:
    """Return the trajectory generic execution/scoring consumers must see.

    The planner result deliberately retains upstream's raw emergency
    controller correction for formula-level auditability. Any consumer that
    models *executed motion*—controller rollout, verifier, progress reference,
    or teleport—must instead observe its zero-state XY projection. Candidate
    mode already contains a physical proposal and is returned unchanged.
    """
    trajectory = np.asarray(result.trajectory, dtype=np.float64)
    if (bool(result.emergency_brake_triggered)
            and cfg.emergency_brake_mode == "trajectory"):
        return emergency_brake_execution_trajectory(trajectory)
    return trajectory.copy()


#: The ``no_valid_proposal`` arm of guard 6a — see :func:`score_brake_reason`.
BRAKE_ARM_NO_VALID_PROPOSAL = "no_valid_proposal"
#: The ``stopping_scores_better`` arm of guard 6a.
BRAKE_ARM_STOPPING_BETTER = "stopping_scores_better"
#: The red-light floor (guard 6r) — see :func:`red_light_brake_reach_m`.
BRAKE_ARM_RED_LIGHT = "red_light"
#: Slack past the pure stopping distance inside which guard 6r still fires:
#: half the ego (the line is judged against the centre) plus a car length.
RED_LIGHT_BRAKE_REACH_MARGIN_M = 5.0


def red_light_brake_reach_m(ego_speed: float, cfg: PDMConfig) -> float:
    """How far ahead a red stop line may be for guard 6r to stop for it.

    Guard 6r is the red-light floor under the proposal grid: when EVERY
    proposal physically crosses a red stop line, the score argmax is
    irrelevant because upstream's scorer deliberately ignores red-light
    tokens. Upstream's observation/IDM coupling prevents such a grid; this
    adapter guard handles the port's residual integration failure directly
    from geometry. The floor applies only when the line is within an
    emergency stop's reach
    (``v² / 2·decel`` plus :data:`RED_LIGHT_BRAKE_REACH_MARGIN_M`), so a
    far-off red that happens to coincide with an all-invalid grid for some
    other reason does not turn into an ungated stop at speed.
    """
    decel = max(float(cfg.emergency_brake_decel), 1e-6)
    return (0.5 * float(ego_speed) ** 2 / decel
            + 0.5 * float(cfg.ego_length) + RED_LIGHT_BRAKE_REACH_MARGIN_M)


def proposal_crosses_red_stop_line(
    proposal: Proposal,
    stop_line: RedLightStopLine,
    cfg: PDMConfig,
) -> bool:
    """Whether a proposal's ego footprint reaches a current-red stop line.

    The stop line is represented by its route tangent and point.  A proposal
    overruns it once the ego centre comes within half an ego length of that
    plane (the front bumper has reached the line).  This deliberately uses
    proposal geometry, not the PDM scalar: the upstream scorer skips
    ``red_light_*`` observation tokens, while the observation/IDM path is
    responsible for stopping.
    """
    state = getattr(proposal, "state", None)
    if state is None:
        return False
    x = np.asarray(getattr(state, "x", []), dtype=np.float64).reshape(-1)
    y = np.asarray(getattr(state, "y", []), dtype=np.float64).reshape(-1)
    if x.size == 0 or x.shape != y.shape:
        return False
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return False
    xy = np.column_stack([x[finite], y[finite]])
    line_xy = np.asarray(stop_line.xy, dtype=np.float64)
    tangent = np.asarray(stop_line.tangent, dtype=np.float64)
    signed_center = (xy - line_xy) @ tangent
    return bool(np.any(signed_center >= -0.5 * float(cfg.ego_length)))


def score_brake_reason(
    *,
    best_score: float,
    brake_score: Optional[float],
    threshold: float,
    arms: str = "both",
) -> Optional[str]:
    """Why guard 6a fires at this state — or ``None`` when it does not.

    THE canonical rationale for the score brake. ``planner``'s module
    docstring, :attr:`PDMConfig.emergency_brake_threshold` and
    ``tests/policy/pdm_closed/test_emergency_brake_reason.py`` all point
    here rather than restating it, so a future change to the guard has
    exactly one place to edit.

    The returned string is ``"<arm>: <detail>"``; the arm alone is also
    published as ``diagnostics["emergency_brake_arm"]``
    (:data:`BRAKE_ARM_NO_VALID_PROPOSAL` /
    :data:`BRAKE_ARM_STOPPING_BETTER`) so an aggregator counting fire
    rates by arm never has to parse prose.

    Guard 6a used to be a bare magnitude test, ``best_score <=
    emergency_brake_threshold``, and no scalar value is defensible.
    ``score = (nc·dac·ddc)·(5·ep + 5·ttc + 2·hc)/12`` multiplies a
    *fractional* validity product into a weighted merit sum, so one
    number cannot separate "nothing here is valid" from "everything here
    is merely mediocre":

    * at the shipped ``0.0`` the guard fires only on an exact zero. From
      a stopped ego the first candidate to cross zero is the one with a
      near-minimal ``dac`` — ``dac = dac_valid / horizon``, so the
      archetype at ``20965a96`` f55 (``nc=1, dac=0.125, ddc=1, ep=1,
      ttc=0, hc=0``, recorded in ``BRAKE_DEADLOCK_EVIDENCE.json``)
      scores ``0.052`` on 1 of 8 waypoints inside the drivable area, and
      a safe stop releases into a parked car;
    * any positive floor ``f`` brakes every proposal whose ``dac < f``
      however good it otherwise is, because ``score <= multi_prod <=
      dac``. The freeze floor is ``dac · 2/12``, not ``2/12``.
    * measured, the knob is inert in healthy driving: an A/B at 0.0 vs
      0.15 on scene 2 is bit-identical. It only ever arbitrates
      degenerate states, which is precisely where a magnitude is
      meaningless.

    So the guard fires for one of two *stated* reasons, and the caller
    records which:

    ``no_valid_proposal``
        ``best_score <= threshold``. Default ``0.0`` = "every proposal
        is fully disqualified by the multiplicative terms". Retained
        rather than deleted because it is the only arm that covers the
        case where the stop is degenerate too (an ego stopped off the
        drivable surface scores the brake candidate ``0`` as well), and
        because it keeps :attr:`PDMConfig.emergency_brake_threshold`
        live, env-pinnable and un-disableable.

    ``stopping_scores_better``
        ``best_score < brake_score``: the argmax is worse than the
        max-deceleration stop, judged by the planner's OWN scorer inside
        the same batch. Scale-free — no constant to tune, and immune to
        the relative-progress renormalisation (and to any scorer-variant
        weight change) because both sides are normalised together.
        Strict ``<`` so an exact tie RELEASES: when nothing in the set
        makes progress the scorer's degenerate branch hands every
        candidate ``ep = 1.0``, including the stop, and a stopped ego's
        zero-offset base proposals are kinematically identical to the
        stop, so they tie it exactly. ``<=`` would make that state
        absorbing.

    **Both arms are STICKY on a static cause, by design.** Below the
    speed gate the ego decelerates to rest and the next replan sees the
    same pose at the same speed, so if the reason is static — the route
    leaving the drivable union (low ``dac``), or a parked blocker giving
    ``ttc = 0`` — the guard re-fires indefinitely. That is not new: the
    shipped scalar is absorbing in exactly the same way, and it is what
    ``20965a96`` f45-f75 records (seven consecutive replans). The
    stopping-vs-going comparison widens the set of states that can
    absorb, so ``diagnostics["score_brake_streak"]`` counts consecutive
    fires and makes the hold legible. No release-after-K is added: a
    counter that overrides a guard whose own judge still says stopping
    is better would reintroduce exactly the untunable magic constant
    this predicate removes, and the standstill remedy identified by
    ``BRAKE_DEADLOCK_REVIEW.md`` (RC3) is add-only proposal generation —
    give the ego a candidate worth taking — not a weaker brake.

    The rejected third formulation, for the record: "fire when every
    proposal has ``nc·dac·ddc·tlc == 0``" is what the old docstring
    claimed the guard meant and is anchor-invariant — but ``dac`` is
    fractional, so the archetype's product is ``0.125``, not zero. A
    pure validity predicate releases exactly the proposal this guard
    exists to hold against.

    Args:
        best_score: Score of the best NOMINAL proposal (the safety brake
            is a floor, never an option, so it is excluded). A
            non-finite value fires the floor arm: it means the scored
            batch is corrupt, and ``nan`` compares False against
            everything, which would otherwise disable BOTH arms at once.
        brake_score: The safety-brake candidate's own score from the
            same scored batch, or ``None`` when no brake candidate was
            built (above the speed gate the planner skips it, since 6a
            cannot fire there). A missing or non-finite reference
            degrades to the floor arm alone — a guard that has to stop
            the car must not depend on an optional input, and must not
            fire merely because that input is absent.
        threshold: :attr:`PDMConfig.emergency_brake_threshold`.
            ``PDMConfig.__post_init__`` rejects negatives; a caller
            constructing this argument some other way must not pass one,
            since a negative threshold makes the floor arm unreachable.

    Returns:
        ``"<arm>: <detail>"`` for ``diagnostics["emergency_brake_
        reason"]``, or ``None`` when guard 6a does not fire. The caller
        still applies the speed gate.
    """
    if arms == "none":
        return None
    if not math.isfinite(best_score) or best_score <= threshold:
        return (f"{BRAKE_ARM_NO_VALID_PROPOSAL}: best_score={best_score:.6f} "
                f"<= emergency_brake_threshold={threshold:.6f}")
    if (arms == "both"
            and brake_score is not None
            and math.isfinite(brake_score)
            and best_score < brake_score):
        return (f"{BRAKE_ARM_STOPPING_BETTER}: best_score={best_score:.6f} < "
                f"safety_brake_score={brake_score:.6f}")
    return None


def is_candidate_fallback(cfg: Any, execution: str) -> bool:
    """Did ``candidate`` mode have to synthesise a stop?

    A campaign arm declaring the candidate contract must record **zero**
    of these. One shared predicate so the planner and the collector
    cannot disagree about what counts as a violation.
    """
    return (getattr(cfg, "emergency_brake_mode", "trajectory") == "candidate"
            and execution != "candidate")


def resolve_brake(
    cfg: PDMConfig,
    proposals: Sequence[Proposal],
    *,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    agents: Optional[Dict[str, AgentPrediction]] = None,
    speed_limit_mps: Optional[float] = None,
    ego_longitudinal_accel: float = 0.0,
) -> Tuple[np.ndarray, int, str]:
    """``(trajectory, chosen_idx, execution)`` for a fired brake guard.

    ``trajectory`` mode returns the upstream-faithful integrated profile.
    ``candidate`` mode returns the forward-simulated safety proposal, and
    falls back to the profile when that proposal is absent — a guard that
    has fired must still stop the car."""
    # getattr: this is called with configs that predate the field (frozen
    # records, test doubles). Missing means the faithful profile, and it
    # must not raise — a guard that has fired still has to stop the car.
    if getattr(cfg, "emergency_brake_mode", "trajectory") == "candidate":
        idx = next((i for i, p in enumerate(proposals)
                    if getattr(p, "proposal_kind", None)
                    == SAFETY_BRAKE_PROPOSAL), -1)
        if idx >= 0:
            traj = _extend_trajectory_to_horizon(
                proposals[idx], ego_x, ego_y, ego_heading, ego_speed,
                agents or {}, speed_limit_mps, cfg)
            return np.asarray(traj, dtype=np.float64), idx, "candidate"
        # Reaching here means candidate mode synthesised a stop anyway —
        # a violation of the arm's own contract, not a benign default.
        # Loud, because a silent fallback is exactly how the narrowed-
        # proposal leak survived: the arm still braked, so nothing looked
        # wrong. Campaign arms must assert this count is zero.
        logger.warning(
            "[brake] candidate mode found no safety proposal among %d "
            "candidates — SYNTHESISED a stop (contract violation)",
            len(proposals))
    return (
        np.asarray(emergency_brake_trajectory(
            ego_speed, cfg, ego_longitudinal_accel=ego_longitudinal_accel),
            dtype=np.float64),
        -1, "trajectory")


# ----------------------------------------------------------------------
# Time-to-collision (emergency-brake guard + loop-1 imminent trigger)
# ----------------------------------------------------------------------


def _rect_corners(
    cx: float, cy: float, heading: float, length: float, width: float
) -> np.ndarray:
    """Corners of a rotated rectangle centred at ``(cx, cy)``."""
    ex = 0.5 * length
    ey = 0.5 * width
    c = math.cos(heading)
    s = math.sin(heading)
    body = np.array(
        [[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]], dtype=np.float64
    )
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return body @ rot.T + np.array([cx, cy])


def _aabb_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _aabb(corners: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(corners[:, 0].min()),
        float(corners[:, 1].min()),
        float(corners[:, 0].max()),
        float(corners[:, 1].max()),
    )


def _polys_intersect(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
    """Cheap SAT (Separating Axis Theorem) on two rectangles.

    Opposite edges of a rectangle are parallel, so each contributes only
    two unique separating axes — testing the first two edges per rectangle
    covers all four axis directions.
    """
    for poly in (corners_a, corners_b):
        for i in range(2):
            edge = poly[i + 1] - poly[i]
            n = math.hypot(edge[0], edge[1])
            if n < 1e-12:
                continue
            axis = np.array([-edge[1] / n, edge[0] / n], dtype=np.float64)
            pa = corners_a @ axis
            pb = corners_b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _currently_overlapping_agent_ids(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    agents: Dict[str, AgentPrediction],
    cfg: PDMConfig,
) -> set[str]:
    """Track ids touching the actual ego at the current observation.

    CaRL's ``PDMObservation.update`` appends these ids to
    ``collided_track_ids`` and excludes them from later object managers.
    This is actual-observation state only: proposal-predicted contacts must
    never mutate the cross-replan latch.
    """
    ego_corners = _rect_corners(
        ego_x, ego_y, ego_heading, cfg.ego_length, cfg.ego_width
    )
    ego_aabb = _aabb(ego_corners)
    overlapping: set[str] = set()
    for track_id, prediction in agents.items():
        if not len(prediction.x) or not bool(prediction.valid[0]):
            continue
        actor_corners = _rect_corners(
            float(prediction.x[0]),
            float(prediction.y[0]),
            float(prediction.heading[0]),
            float(prediction.length),
            float(prediction.width),
        )
        if (_aabb_overlap(ego_aabb, _aabb(actor_corners))
                and _polys_intersect(ego_corners, actor_corners)):
            overlapping.add(str(track_id))
    return overlapping


def _sweep_time_to_collision(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_heading: np.ndarray,
    ego_speed: np.ndarray,
    agents: Dict[str, AgentPrediction],
    cfg: PDMConfig,
) -> float:
    """Shared ego-vs-agent box sweep with the per-agent first-contact latch.

    Steps ``i`` over the ego poses on the ``cfg.sim_dt`` grid, building a
    rotated rectangle per step and intersecting it against each agent's
    predicted rectangle at the same index (each agent provides ``x[k]``,
    ``y[k]``, ``heading[k]``, ``length``, ``width``). Returns
    ``i * cfg.sim_dt`` of the first at-fault contact, ``+inf`` if none.

    Per-agent FIRST-CONTACT latch (mirrors upstream nuPlan's
    ``already_collided_ids``): a collision is classified once, at first
    contact. An agent whose first overlap is not-at-fault — pre-existing
    at step 0, rear-end by the agent, side contact, or ego stopped — stays
    exempt for the remainder of the sweep. Without the latch, a faster
    agent overrunning the ego from behind was re-classified as an
    at-fault FRONT collision the moment its centre swept past the ego's
    centre mid-plow-through (non-reactive log replay drives THROUGH a
    slower-than-log ego), which kept the emergency brake and the loop-2
    override veto firing forever on stopped-ego frames.
    """
    L = float(cfg.ego_length)
    W = float(cfg.ego_width)
    # Bounding-circle pre-reject: skip all rectangle work for agents that
    # cannot possibly overlap (the overwhelming majority at every step).
    r_ego = 0.5 * math.hypot(L, W)

    exempt: set[str] = set()
    n_agents = len(agents)
    for i in range(len(ego_x)):
        if len(exempt) == n_agents:
            return float("inf")
        ex = float(ego_x[i])
        ey = float(ego_y[i])
        eh = float(ego_heading[i])
        ev = float(ego_speed[i])

        ego_corners: Optional[np.ndarray] = None
        ego_aabb: Optional[Tuple[float, float, float, float]] = None

        ego_stopped = ev < 0.05
        cos_h = math.cos(eh)
        sin_h = math.sin(eh)

        for agent_id, pred in agents.items():
            if agent_id in exempt:
                continue
            if i >= pred.x.shape[0] or not bool(pred.valid[i]):
                continue
            ax = float(pred.x[i])
            ay = float(pred.y[i])
            r_sum = r_ego + 0.5 * math.hypot(pred.length, pred.width)
            if (ax - ex) ** 2 + (ay - ey) ** 2 > r_sum * r_sum:
                continue
            if ego_corners is None:
                ego_corners = _rect_corners(ex, ey, eh, L, W)
                ego_aabb = _aabb(ego_corners)
            ah = float(pred.heading[i])
            agent_corners = _rect_corners(ax, ay, ah, pred.length, pred.width)
            agent_aabb = _aabb(agent_corners)
            if not _aabb_overlap(ego_aabb, agent_aabb):  # type: ignore[arg-type]
                continue
            if not _polys_intersect(ego_corners, agent_corners):
                continue

            # Not-at-fault first contacts latch the agent exempt for the
            # whole sweep: step-0 overlaps are PRE-EXISTING (every
            # candidate starts from the same pose, so none caused or can
            # avoid them — upstream's already_collided_ids), and the
            # at-fault approximation of nuPlan's classification counts
            # only FRONT collisions (agent centre ahead of ego centre)
            # by a moving ego — rear-ends by the other agent and pure
            # side contact are exempt (the old ``< -1.0`` cutoff faulted
            # every sideswipe and fed the phantom-brake stall loop).
            longitudinal = (ax - ex) * cos_h + (ay - ey) * sin_h
            if i == 0 or ego_stopped or longitudinal <= 0.0:
                exempt.add(agent_id)
                continue
            return float(i * cfg.sim_dt)

    return float("inf")


def proposal_time_to_collision(
    proposal: Proposal,
    agents: Dict[str, AgentPrediction],
    cfg: PDMConfig,
) -> float:
    """Return the earliest at-fault collision time (s) along ``proposal``.

    Walks the proposal step-by-step, builds a rotated rectangle for
    the ego at each step, intersects with each agent's rotated
    rectangle (also stepped from the agent's constant-velocity
    prediction). Returns ``+inf`` if no collision is detected.

    This implementation mirrors the at-fault filter the EPDMS scorer
    applies: a collision is considered at-fault unless

    * ego is essentially stopped (speed < 0.05 m/s), or
    * the agent's centre is at or behind the ego's centre along the
      ego heading (rear-end by the other agent, or side contact the
      ego did not initiate).

    The check is deliberately stricter than the scorer's NC metric
    so the brake fires for *imminent* hits rather than only on
    horizon-aggregated collision probability. Matches the spirit of
    ``tuplan_garage/utils/pdm_emergency_brake.py``.
    """
    if not agents:
        return float("inf")

    n_steps = cfg.num_sim_steps
    horizon_steps = int(round(cfg.infraction_horizon_s / cfg.sim_dt))
    horizon_steps = min(horizon_steps, n_steps)

    # Fancy indexing (not a slice) deliberately preserves the historical
    # IndexError on proposals shorter than the brake horizon.
    ks = np.arange(horizon_steps + 1)
    return _sweep_time_to_collision(
        np.asarray(proposal.state.x, dtype=np.float64)[ks],
        np.asarray(proposal.state.y, dtype=np.float64)[ks],
        np.asarray(proposal.state.heading, dtype=np.float64)[ks],
        np.asarray(proposal.state.speed, dtype=np.float64)[ks],
        agents,
        cfg,
    )


# Below this per-step displacement (m) a trajectory segment is degenerate:
# its direction is numeric noise, so the previous step's heading carries.
_DEGENERATE_STEP_M = 1e-3


def trajectory_time_to_collision(
    traj_world_xy: np.ndarray,
    ego_state: Dict[str, Any],
    agents: Dict[str, AgentPrediction],
    cfg: PDMConfig,
    *,
    waypoint_dt: Optional[float] = None,
) -> float:
    """Earliest at-fault collision time (s) along a raw waypoint trajectory.

    The trajectory analog of :func:`proposal_time_to_collision` — same
    box sweep, same at-fault filter, same per-agent first-contact latch —
    for callers that hold only a planner-output trajectory instead of a
    forward-simulated :class:`Proposal` (the loop-1 imminent-collision
    intervention trigger sweeps the student's would-execute candidate).

    Args:
        traj_world_xy: ``(N, >=2)`` world-frame ``(x, y)`` waypoints at
            the planner output cadence (``waypoint_dt`` apart, first
            waypoint at ``t = waypoint_dt``).
        ego_state: Current ego state; reads ``position``, ``heading``
            and ``speed`` for the sweep's step-0 pose (so pre-existing
            contacts latch exactly like the proposal sweep's ``k=0``).
        agents: Per-agent predictions indexed at ``cfg.sim_dt``
            (:func:`~.forward_sim.predict_agents_log_replay` output).
        cfg: Ego dims / cadences — same fields the proposal sweep uses.
        waypoint_dt: Cadence (s) of the input waypoints. ``None`` →
            ``cfg.output_dt``. Callers whose trajectories come from a
            scorer/verifier with its own cadence (``planner_dt``) pass
            it here so the interpolation grid matches the data.

    The waypoint polyline is linearly interpolated onto the ``cfg.sim_dt``
    grid and swept densely — waypoint-cadence sampling (0.5 s by default)
    provably misses fast crossings whose overlap window is shorter than a
    waypoint interval, and quantizes the returned TTC to the waypoint
    cadence. Per-step heading is derived from the interpolated segments
    (falling back to the previous step's heading — ultimately the ego
    heading — on degenerate/stationary segments) and per-step speed from
    displacement over ``sim_dt``, so a stationary trajectory sweeps as a
    stopped ego and is never at fault. Returns ``+inf`` when no at-fault
    collision occurs within the trajectory horizon.
    """
    if not agents:
        return float("inf")
    traj = np.asarray(traj_world_xy, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
        return float("inf")

    ego_x0 = float(ego_state["position"][0])
    ego_y0 = float(ego_state["position"][1])
    ego_h0 = float(ego_state.get("heading", 0.0))
    ego_v0 = float(ego_state.get("speed", 0.0))
    wp_dt = float(waypoint_dt) if waypoint_dt is not None else float(cfg.output_dt)
    sim_dt = float(cfg.sim_dt)

    # Prepend the current pose: step 0 must alias agent prediction index 0
    # so pre-existing contacts latch exempt (the proposal sweep's k=0 rule).
    pts = np.concatenate([[[ego_x0, ego_y0]], traj[:, :2]], axis=0)
    wp_times = np.arange(pts.shape[0]) * wp_dt

    # Densify: linear interpolation onto the sim_dt grid (the agent
    # prediction cadence), so the sweep sees every sim step instead of
    # one pose per waypoint.
    n = int(math.floor(wp_times[-1] / sim_dt + 1e-9)) + 1
    step_times = np.arange(n) * sim_dt
    dense_x = np.interp(step_times, wp_times, pts[:, 0])
    dense_y = np.interp(step_times, wp_times, pts[:, 1])
    dx = np.diff(dense_x)
    dy = np.diff(dense_y)
    dists = np.hypot(dx, dy)

    speeds: np.ndarray = np.empty(n, dtype=np.float64)
    speeds[0] = ego_v0
    speeds[1:] = dists / sim_dt

    headings: np.ndarray = np.empty(n, dtype=np.float64)
    headings[0] = ego_h0
    for i in range(1, n):
        if dists[i - 1] >= _DEGENERATE_STEP_M:
            headings[i] = math.atan2(dy[i - 1], dx[i - 1])
        else:
            headings[i] = headings[i - 1]

    # The dense grid IS the prediction grid: step i ↔ prediction index i
    # (indices past pred_len are skipped by the sweep's validity check).
    return _sweep_time_to_collision(
        dense_x, dense_y, headings, speeds, agents, cfg,
    )


# ----------------------------------------------------------------------
# Synthetic-route fallback
# ----------------------------------------------------------------------


def _synthetic_route_along_heading(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    length_m: float = 80.0,
    spacing_m: float = 1.0,
) -> np.ndarray:
    """Emit a straight-ahead route ~``length_m`` long."""
    n = max(2, int(round(length_m / spacing_m)) + 1)
    s = np.linspace(0.0, length_m, n)
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    return np.column_stack([ego_x + cos_h * s, ego_y + sin_h * s])


# ----------------------------------------------------------------------
# Speed-limit extraction
# ----------------------------------------------------------------------


def _lane_speed_limit_if_allowed(
    feat: Dict[str, Any], *, allow_inferred: bool,
) -> Optional[float]:
    """A lane feature's posted limit in m/s, or ``None``.

    ``speed_limit_source`` is added by NexusSim's log-derived spot speed
    annotator. Upstream consumes posted map limits, not future motion from
    the scenario being evaluated, so inferred limits are skipped unless
    explicitly allowed.
    """
    source = str(feat.get("speed_limit_source", "dataset")).lower()
    if not allow_inferred and source not in ("", "dataset"):
        return None
    return _speed_limit_to_mps(feat)


def _starting_lane_speed_limit_mps(
    scenario_data: Dict[str, Any],
    ego_position: np.ndarray,
    ego_heading: float,
    route_lane_ids: Sequence[str],
    *,
    rear_axle_offset_m: float,
    ego_length: float,
    ego_width: float,
    allow_inferred: bool = False,
) -> Optional[float]:
    """Reference ``_get_starting_lane`` → ``speed_limit_mps`` on route lanes.

    ``AbstractPDMPlanner._get_starting_lane``: the on-route lanes whose
    polygon contains the ego REAR-AXLE point are the candidates, and the one
    with the smallest heading error (lane heading at the nearest baseline
    vertex vs the ego heading) wins; when no on-route lane contains the
    point, the on-route lane nearest to the ego footprint polygon does. Its
    ``speed_limit_mps`` (``None`` → the IDM fallback) sets every policy's
    target. Upstream's ``route_lane_dict`` holds every lane of the route
    roadblocks; the lane-graph walk's chain is the closest thing py123d
    offers (no roadblock grouping), so a neighbour lane the walk did not
    traverse resolves through the nearest-polygon fallback.
    """
    from shapely.geometry import Point, Polygon

    map_features = scenario_data.get("map_features", {})
    if not map_features or not route_lane_ids:
        return None
    pos2 = np.asarray(ego_position, dtype=np.float64).reshape(-1)[:2]
    cos_h = float(np.cos(ego_heading))
    sin_h = float(np.sin(ego_heading))
    rear = Point(pos2[0] - rear_axle_offset_m * cos_h,
                 pos2[1] - rear_axle_offset_m * sin_h)
    footprint = Polygon(_rect_corners(
        float(pos2[0]), float(pos2[1]), float(ego_heading),
        float(ego_length), float(ego_width)))

    best: Optional[Dict[str, Any]] = None
    best_heading_error = float("inf")
    nearest: Optional[Dict[str, Any]] = None
    nearest_dist = float("inf")
    for lane_id in route_lane_ids:
        feat = map_features.get(lane_id)
        if feat is None:
            feat = map_features.get(str(lane_id))
        if feat is None:
            continue
        polyline = feat.get("polyline")
        polygon = feat.get("polygon")
        if polyline is None:
            continue
        polyline = np.asarray(polyline, dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[0] < 2:
            continue
        shape = None
        if polygon is not None:
            poly = np.asarray(polygon, dtype=np.float64)
            if poly.ndim == 2 and poly.shape[0] >= 3:
                shape = Polygon(poly[:, :2])
                if not shape.is_valid:
                    shape = shape.buffer(0.0)
        if shape is not None and shape.covers(rear):
            d = np.linalg.norm(polyline[:, :2] - np.array([rear.x, rear.y]), axis=1)
            j = int(np.argmin(d))
            seg = (polyline[min(j + 1, polyline.shape[0] - 1), :2]
                   - polyline[max(j - 1, 0), :2])
            if float(np.linalg.norm(seg)) > 1e-9:
                lane_heading = math.atan2(float(seg[1]), float(seg[0]))
                err = abs((lane_heading - float(ego_heading) + math.pi)
                          % (2.0 * math.pi) - math.pi)
                if err < best_heading_error:
                    best_heading_error, best = err, feat
        dist = (float(shape.distance(footprint)) if shape is not None
                else float(np.min(np.linalg.norm(polyline[:, :2] - pos2, axis=1))))
        if dist < nearest_dist:
            nearest_dist, nearest = dist, feat
    chosen = best if best is not None else nearest
    if chosen is None:
        return None
    return _lane_speed_limit_if_allowed(chosen, allow_inferred=allow_inferred)


def _active_lane_speed_limit_mps(
    scenario_data: Dict[str, Any],
    ego_position: np.ndarray,
    ego_heading: float,
    *,
    allow_inferred: bool = False,
) -> Optional[float]:
    """Return the speed limit (m/s) of the nearest heading-aligned lane.

    Fallback used when the planner has no route lane identity (synthetic /
    ``nearest`` routes, offline callers); with a lane-graph route the
    reference rule in :func:`_starting_lane_speed_limit_mps` applies
    instead. Measured on the 24 NavSafe bundles, this heuristic and the
    reference rule disagree on 11 of 834 replans (a neighbouring lane with a
    different posted limit), which is why the route-aware rule exists.

    The returned value is in m/s (converted from kmh / mph if the
    scenario carries either of those). Returns ``None`` when:

    * the scenario has no map features; or
    * no lane within 5 m of the ego carries a speed-limit annotation.
    """
    map_features = scenario_data.get("map_features", {})
    if not map_features:
        return None

    pos2 = np.asarray(ego_position, dtype=np.float64).reshape(-1)[:2]
    cos_h = float(np.cos(ego_heading))
    sin_h = float(np.sin(ego_heading))
    ego_dir = np.array([cos_h, sin_h])

    best_dist = float("inf")
    best_speed_mps: Optional[float] = None

    for lane_id, feat in map_features.items():
        feat_type = feat.get("type", "")
        if "LANE" not in str(feat_type).upper():
            continue
        polyline = feat.get("polyline")
        if polyline is None:
            continue
        polyline = np.asarray(polyline, dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[0] < 2:
            continue

        # Distance from ego to the polyline.
        d = np.linalg.norm(polyline[:, :2] - pos2, axis=1)
        nearest = int(np.argmin(d))
        dist = float(d[nearest])
        if dist > 5.0 or dist >= best_dist:
            continue

        # Heading alignment: skip lanes pointing the wrong way.
        if nearest < polyline.shape[0] - 1:
            seg = polyline[nearest + 1, :2] - polyline[nearest, :2]
        else:
            seg = polyline[nearest, :2] - polyline[nearest - 1, :2]
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-9:
            continue
        seg_unit = seg / seg_norm
        if float(np.dot(seg_unit, ego_dir)) < 0.0:
            continue

        # Pull a speed-limit field from the lane feature (posted limits
        # only, unless inferred ones are explicitly allowed).
        sl_mps = _lane_speed_limit_if_allowed(feat, allow_inferred=allow_inferred)
        if sl_mps is None:
            continue
        best_dist = dist
        best_speed_mps = sl_mps

    return best_speed_mps


def _speed_limit_to_mps(feat: Dict[str, Any]) -> Optional[float]:
    """Read a lane's speed-limit field and convert to m/s.

    Tries ``speed_limit_mps``, ``speed_limit_kmh``,
    ``speed_limit_mph``, ``speedLimit`` (in that order) — covers the
    Bench2Drive / Waymo / NavSim ScenarioNet variants.
    """
    if "speed_limit_mps" in feat:
        v = feat["speed_limit_mps"]
    elif "speed_limit_kmh" in feat:
        v = float(feat["speed_limit_kmh"]) / 3.6
    elif "speed_limit_mph" in feat:
        v = float(feat["speed_limit_mph"]) * 0.44704
    elif "speedLimit" in feat:
        # Assume m/s when units are unannotated.
        v = feat["speedLimit"]
    else:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v


# ----------------------------------------------------------------------
# Trajectory extension (4 s → 8 s for tuplan_garage parity)
# ----------------------------------------------------------------------


def _extend_trajectory_to_horizon(
    proposal: Proposal,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    agents: Dict[str, AgentPrediction],
    speed_limit_mps: Optional[float],
    cfg: PDMConfig,
) -> np.ndarray:
    """Extend the selected ideal IDM-on-path proposal to output horizon."""
    # XY-only callers do not need to inspect the proposal's dense velocity
    # state when it already has the requested horizon. Keeping this fast path
    # also preserves compatibility with archived/duck-typed proposals that
    # predate explicit speed metadata.
    out = np.asarray(proposal.output_xy_ego, dtype=np.float64)
    if out.shape[0] == cfg.num_trajectory_poses:
        return out.copy()
    return _extend_trajectory_and_speeds_to_horizon(
        proposal, ego_x, ego_y, ego_heading, ego_speed, agents,
        speed_limit_mps, cfg)[0]


def _sample_output_speeds(state: Any, cfg: PDMConfig) -> np.ndarray:
    """Sample a proposal state's explicit velocity at output timestamps."""
    indices = np.arange(1, cfg.num_output_poses + 1) * cfg.output_stride
    indices = np.minimum(indices, len(state.speed) - 1)
    return np.asarray(state.speed[indices], dtype=np.float64).copy()


def _extend_trajectory_and_speeds_to_horizon(
    proposal: Proposal,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    agents: Dict[str, AgentPrediction],
    speed_limit_mps: Optional[float],
    cfg: PDMConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the extended ideal poses and their longitudinal velocities."""
    out = proposal.output_xy_ego  # (N_short, 2)
    n_target = cfg.num_trajectory_poses
    if out.shape[0] == n_target:
        ideal_state = getattr(proposal, "ideal_state", None)
        source = ideal_state if ideal_state is not None else proposal.state
        return out.copy(), _sample_output_speeds(source, cfg)
    extended_cfg = replace(
        cfg,
        horizon_s=cfg.trajectory_horizon_s,
        num_output_poses=cfg.num_trajectory_poses,
    )
    # The safety brake's commanded deceleration must survive the re-sim.
    # This extension runs whenever the output horizon differs from the
    # proposal horizon (the documented 8 s tuplan_garage parity config),
    # and re-simulating the brake as an ordinary IDM proposal under
    # idm_policies[0] produced a "brake" that decelerated briefly and then
    # CRUISED at the IDM target speed indefinitely — while diagnostics
    # claimed emergency_brake_execution="candidate". The default 4 s
    # config was safe only via the early return above.
    longitudinal_override = (
        -float(cfg.emergency_brake_decel)
        if getattr(proposal, "proposal_kind", None) == SAFETY_BRAKE_PROPOSAL
        else None)
    extended_state = simulate_proposal(
        initial_x=ego_x,
        initial_y=ego_y,
        initial_heading=ego_heading,
        initial_speed=ego_speed,
        path_xy=proposal.path_world_xy,
        policy=proposal.idm,
        cfg=extended_cfg,
        agents=agents,
        speed_limit_mps=speed_limit_mps,
        longitudinal_accel_mps2=longitudinal_override,
        ideal_kinematics=(longitudinal_override is None),
    )
    return (
        extended_state.output_xy_ego.copy(),
        _sample_output_speeds(extended_state, extended_cfg),
    )


# ----------------------------------------------------------------------
# PDMPlanner
# ----------------------------------------------------------------------


def scenario_identity_key(scenario_data: dict) -> Tuple[Any, ...]:
    """Content-derived identity of a scenario dict.

    Used by :meth:`PDMPlanner.plan` to decide whether the caller handed it a
    new scenario (reset) or the same one as last call (keep calibration).

    Why not ``id(scenario_data)``: CPython recycles object addresses, so a
    freshly loaded scenario can land on the address of a garbage-collected
    predecessor and *skip* the reset — keeping stale scorer calibration, a
    stale committed frame id. Mutating one dict in
    place had the mirror problem: a genuinely different scenario at the same
    address. Both failures were silent and unreproducible.

    Composition (mirrors ``route_lane_chain._cache_key``): the py123d
    converter stamps ``metadata`` with ``dataset`` / ``split`` /
    ``scenario_id``, which together name the source log uniquely. ``sdc_id``,
    frame count, track count, map-feature count and the ego's first/last
    planar pose are cheap structural tiebreaks, so an in-place edit of the
    same dict reads as a different scenario.

    Fallback when no id is stamped: hand-built dicts (unit tests, procgen,
    Scenic) carry neither ``metadata['scenario_id']`` nor top-level ``id``.
    Identity then rests on the structural fields alone, so two *different*
    scenarios agreeing on every one of them (same frame/track/map counts and
    the same ego endpoints) would be treated as the same scenario. That
    aliasing is deterministic and content-bounded — unlike ``id()`` reuse it
    cannot vary run to run — and it is the reason converted logs are always
    preferable. The fallback never degrades to ``id()``.

    Non-finite ego coordinates are folded to ``0.0``: ``nan != nan`` would
    otherwise make every key mismatch and reset the planner on every frame.
    """
    metadata = scenario_data.get("metadata") or {}
    tracks = scenario_data.get("tracks") or {}
    sdc_id = metadata.get("sdc_id")
    state = (tracks.get(sdc_id) or {}).get("state") or {}
    positions = state.get("position")

    ego_anchor: Tuple[float, ...] = ()
    n_frames = 0
    if positions is not None:
        arr = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        n_frames = int(arr.shape[0])
        if n_frames and arr.shape[1] >= 2:
            ends = np.nan_to_num(
                np.round(np.stack([arr[0, :2], arr[-1, :2]]), 3),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            ego_anchor = tuple(float(v) for v in ends.ravel())
    if not n_frames:
        try:
            n_frames = int(scenario_data.get("length") or 0)
        except (TypeError, ValueError):
            n_frames = 0

    scenario_id = metadata.get("scenario_id") or scenario_data.get("id") or ""
    return (
        str(metadata.get("dataset")),
        str(metadata.get("split")),
        str(scenario_id),
        str(sdc_id),
        n_frames,
        len(tracks),
        len(scenario_data.get("map_features") or {}),
        ego_anchor,
    )


class PDMPlanner:
    """Faithful PDM-Closed planner.

    Lifecycle:
        1. Construct: ``planner = PDMPlanner()`` (or with a custom
           :class:`PDMConfig`).
        2. Reset on a new scenario: ``planner.reset(scenario_data, env=None)``.
        3. Per replan: ``result = planner.plan(scenario_data, ego_state, frame_id)``.

    Thread-safety: a single planner instance is **not** thread-safe.
    Use one planner per parallel rollout.
    """

    def __init__(self, cfg: Optional[PDMConfig] = None) -> None:
        self.cfg = cfg if cfg is not None else PDMConfig()
        self._scorer = PDMScorer(self.cfg, verbose=False)
        self._scenario_key: Optional[Tuple[Any, ...]] = None
        self._last_committed_frame_id: Optional[int] = None
        #: Consecutive replans ended by guard 6a. Diagnostic only — it
        #: never releases the guard (see :func:`score_brake_reason`) —
        #: but a sticky brake is otherwise invisible in a dump.
        self._score_brake_streak: int = 0
        # Upstream creates its proposal paths at iteration zero and reuses
        # them.  Re-extracting a lane-graph route every replan can change the
        # mission branch as the simulated ego drifts from the log.
        self._route_centerline: Optional[np.ndarray] = None
        self._route_source: Optional[str] = None
        self._route_diagnostics: Dict[str, Any] = {}
        # NexusSim's ordered lane surrogate can contain map gaps that prevent
        # the first extraction from consuming the whole mission. Upstream's
        # cached object is complete; freezing an incomplete local substitute
        # makes the ego park at its artificial tail. Complete routes retain
        # strict iteration-zero reuse, while incomplete ones may be
        # re-anchored before the ego exhausts their remaining coverage.
        self._route_incomplete: bool = False
        self._route_refresh_count: int = 0
        self._route_refresh_attempt_count: int = 0
        self._route_refresh_anchor_xy: Optional[np.ndarray] = None
        # CaRL PDMObservation remembers real objects already contacted and
        # removes them from subsequent observations. Proposal-only contacts do
        # not enter this set.
        self._collided_track_ids: set[str] = set()
        # Lane-group index for the scorer's on-route set (built lazily).
        self._lane_group_index: Optional[Dict[str, List[str]]] = None
        self._lane_group_of: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, scenario_data: dict, env: Optional[object] = None) -> None:
        """Bind the planner to a new scenario.

        """
        self._scorer.initialize(scenario_data, env=env)
        self._scenario_key = scenario_identity_key(scenario_data)
        self._last_committed_frame_id = None
        self._score_brake_streak = 0
        self._route_centerline = None
        self._route_source = None
        self._route_diagnostics = {}
        self._route_incomplete = False
        self._route_refresh_count = 0
        self._route_refresh_attempt_count = 0
        self._route_refresh_anchor_xy = None
        self._collided_track_ids = set()
        self._lane_group_index = None
        self._lane_group_of = {}

    # ------------------------------------------------------------------
    # Route roadblocks (lane groups) for the scorer's on-route set
    # ------------------------------------------------------------------

    def _route_roadblock_lane_ids(
        self, scenario_data: dict, walk_ids: Sequence[Any],
    ) -> frozenset[str]:
        """All lanes sharing a lane group with any walked lane (+ the walk).

        The lane-group index is built once per scenario (``reset`` clears it).
        """
        if self._lane_group_index is None:
            index: Dict[str, List[str]] = {}
            for lane_id, feat in (scenario_data.get("map_features") or {}).items():
                group = feat.get("lane_group_id")
                if group is not None:
                    index.setdefault(str(group), []).append(str(lane_id))
            self._lane_group_index = index
            self._lane_group_of = {
                lane: group for group, lanes in index.items() for lane in lanes}
        out = set()
        for lane_id in walk_ids:
            lane = str(lane_id)
            out.add(lane)
            group = self._lane_group_of.get(lane)
            if group is not None:
                out.update(self._lane_group_index.get(group, ()))
        return frozenset(out)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _incomplete_route_needs_refresh(
        self, ego_position: np.ndarray, ego_speed: float,
    ) -> bool:
        """Whether an incomplete cached route lacks planning coverage.

        The threshold covers the route-hint horizon at the simulated ego's
        current speed, with the same 30 m standstill floor used by route
        extraction. Complete mission caches never enter this path.
        """
        route = self._route_centerline
        if not self._route_incomplete or route is None or len(route) < 2:
            return False
        point = np.asarray(ego_position, dtype=np.float64).reshape(-1)[:2]
        # A failed extraction at the same pose deterministically produces the
        # same route. Retry only after lane anchoring can materially change.
        if (self._route_refresh_anchor_xy is not None
                and float(np.linalg.norm(
                    point - self._route_refresh_anchor_xy)) < 10.0):
            return False
        xy = np.asarray(route, dtype=np.float64)[:, :2]
        segments = np.diff(xy, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid = lengths > 1e-9
        if not np.any(valid):
            return True
        rel = point - xy[:-1]
        denom = np.maximum(lengths * lengths, 1e-12)
        frac = np.clip(
            np.einsum("ij,ij->i", rel, segments) / denom, 0.0, 1.0)
        projections = xy[:-1] + frac[:, None] * segments
        nearest = int(np.argmin(np.linalg.norm(projections - point, axis=1)))
        consumed = float(
            np.sum(lengths[:nearest]) + frac[nearest] * lengths[nearest])
        remaining = max(0.0, float(np.sum(lengths)) - consumed)
        required = max(
            30.0, float(ego_speed) * float(self.cfg.route_horizon_s))
        # Match the extractor's 90%-coverage boundary exactly.
        return remaining < 0.9 * required

    def _score_with_brake_reference(
        self,
        proposals: List[Proposal],
        ego_state: dict,
        *,
        frame_id: int,
        route: np.ndarray,
        ego_x: float,
        ego_y: float,
        ego_heading: float,
        ego_speed: float,
        ego_longitudinal_accel: float,
        ego_steering_angle: float,
        ego_angular_velocity: float,
        agents: Dict[str, AgentPrediction],
        speed_limit_mps: Optional[float],
    ) -> Tuple[np.ndarray, Optional[float], bool]:
        """Score ``proposals``, plus the safety brake guard 6a compares to.

        Returns ``(scores, brake_reference_score, brake_in_proposals)``
        where ``scores`` is aligned 1:1 with ``proposals``.

        Skipped entirely above ``emergency_brake_max_ego_speed``, where
        6a cannot fire; the reference is then ``None`` and the guard
        degrades to its floor arm, which is also gated off. Same
        shortcut 6b already takes for its TTC sweep. This keeps the
        default mode's extra cost (one forward sim + one scored
        proposal, ~5% of a replan) off every above-gate frame.

        Appending is score-neutral for the base grid: progress
        normalisation is anchored to ``n_base = cfg.num_proposals``
        (``scoring.py``), which the extra entry sits past, and the
        appended stop can never be the masked-progress maximum because
        the base grid contains a zero-offset proposal that tracks the
        same route at greater-or-equal speed.
        Pinned by
        ``test_appending_the_brake_reference_does_not_move_base_scores``."""
        scoring_ego_state = dict(ego_state)
        scoring_ego_state["_pdm_collided_track_ids"] = tuple(
            sorted(self._collided_track_ids)
        )
        if self.cfg.emergency_brake_score_arms == "none":
            raw, _ = self._scorer.score_proposals(
                proposals, scoring_ego_state,
                frame_id=frame_id, centerline=route)
            return np.asarray(raw, dtype=np.float64), None, False

        brake_ref_idx = next(
            (i for i, p in enumerate(proposals)
             if p.proposal_kind == SAFETY_BRAKE_PROPOSAL), -1)
        brake_in_proposals = brake_ref_idx >= 0
        scored: List[Proposal] = proposals
        if not brake_in_proposals:
            if ego_speed > self.cfg.emergency_brake_max_ego_speed:
                raw, _ = self._scorer.score_proposals(
                    proposals, scoring_ego_state,
                    frame_id=frame_id, centerline=route)
                return np.asarray(raw, dtype=np.float64), None, False
            scored = list(proposals) + [build_brake_proposal(
                ego_x=ego_x, ego_y=ego_y, ego_heading=ego_heading,
                ego_speed=ego_speed, path_world_xy=route, cfg=self.cfg,
                proposal_idx=len(proposals), agents=agents,
                speed_limit_mps=speed_limit_mps,
                ego_acceleration=ego_longitudinal_accel,
                ego_steering_angle=ego_steering_angle,
                ego_angular_velocity=ego_angular_velocity,
            )]
            brake_ref_idx = len(proposals)

        all_scores = np.asarray(
            self._scorer.score_proposals(
                scored, scoring_ego_state,
                frame_id=frame_id, centerline=route)[0],
            dtype=np.float64)
        brake_ref_score: Optional[float] = (
            float(all_scores[brake_ref_idx])
            if 0 <= brake_ref_idx < len(all_scores) else None)
        return all_scores[: len(proposals)], brake_ref_score, brake_in_proposals

    def plan(
        self,
        scenario_data: dict,
        ego_state: dict,
        frame_id: int,
    ) -> PDMPlanResult:
        """Plan one waypoint trajectory for the given frame."""
        if scenario_identity_key(scenario_data) != self._scenario_key:
            self.reset(scenario_data)
        elif (self._last_committed_frame_id is not None
                and frame_id < self._last_committed_frame_id):
            # Same scenario, strictly earlier frame: only an episode restart
            # moves time backwards (within-episode re-plans reuse the SAME
            # frame id — the gated collector's probe memo and the stock
            # ep-reference both do). Without this, episode k's committed
            # frame survives into a back-to-back episode k+1 on the same
            # scene (episodes_per_iter=1, rotation off), pinning
            # ``prev_frame_idx = None`` — full-horizon comfort scoring —
            # for every replan at or below the stale frame, so expert
            # labels differ between otherwise identical episodes.
            self.reset(scenario_data)

        ego_x = float(ego_state["position"][0])
        ego_y = float(ego_state["position"][1])
        ego_heading = float(ego_state["heading"])
        ego_speed = float(ego_state.get("speed", 0.0))
        if not np.isfinite(ego_speed):
            ego_speed = float(np.linalg.norm(np.asarray(ego_state["velocity"][:2])))
        ego_speed = max(0.0, ego_speed)
        ego_accel_vec = np.asarray(ego_state.get("acceleration", np.zeros(3)), dtype=np.float64)
        ego_longitudinal_accel = float(
            ego_accel_vec[0] * math.cos(ego_heading) + ego_accel_vec[1] * math.sin(ego_heading)
        ) if ego_accel_vec.size >= 2 else 0.0
        ego_angular_velocity_vec = np.asarray(
            ego_state.get("angular_velocity", np.zeros(3)), dtype=np.float64)
        ego_angular_velocity = float(
            ego_angular_velocity_vec[2]
            if ego_angular_velocity_vec.size >= 3 else 0.0)
        explicit_steering = ego_state.get("_execution_steering_angle_rad")
        if explicit_steering is not None and np.isfinite(float(explicit_steering)):
            ego_steering_angle = float(explicit_steering)
        elif ego_speed > 1e-3 and np.isfinite(ego_angular_velocity):
            # NexusSim's pure-pursuit tracker has no persistent tire-angle
            # field. Recover the angle that produced the observed yaw rate
            # under its symmetric bicycle instead of resetting PDM's
            # simulated actuator state to zero at every replan.
            sin_beta = float(np.clip(
                ego_angular_velocity * self.cfg.wheelbase
                / (2.0 * ego_speed), -0.999999, 0.999999))
            beta = math.asin(sin_beta)
            ego_steering_angle = float(np.clip(
                math.atan(2.0 * math.tan(beta)),
                -self.cfg.max_steering_angle_rad,
                self.cfg.max_steering_angle_rad))
        else:
            ego_steering_angle = 0.0

        diagnostics: Dict[str, Any] = {}

        # 1. Route. Try the configured route source; fall back to a
        # synthetic straight-ahead route if it fails.
        refresh_incomplete = self._incomplete_route_needs_refresh(
            ego_state["position"], ego_speed)
        if self._route_centerline is None or refresh_incomplete:
            candidate_diagnostics: Dict[str, Any] = {}
            if refresh_incomplete:
                self._route_refresh_attempt_count += 1
                self._route_refresh_anchor_xy = np.array(
                    [ego_x, ego_y], dtype=np.float64)
                candidate_diagnostics[
                    "route_refresh_attempt_count"] = (
                        self._route_refresh_attempt_count)
            candidate_route = extract_route_centerline(
                scenario_data,
                ego_position=ego_state["position"],
                ego_heading=ego_heading,
                frame_id=frame_id,
                route_horizon_s=self.cfg.route_horizon_s,
                densify_spacing_m=self.cfg.route_densify_spacing_m,
                route_source=self.cfg.route_source,
                lane_graph_max_depth=self.cfg.lane_graph_max_depth,
                lane_graph_max_length_m=self.cfg.lane_graph_max_length_m,
                ego_speed=ego_speed,
                diagnostics=candidate_diagnostics,
                rear_axle_offset_m=0.5 * float(self.cfg.wheelbase),
            )
            mission_ids = (
                list((scenario_data.get("metadata", {}) or {}).get(
                    "route_lane_ids") or [])
                if self.cfg.route_source == "lane_graph_route" else [])

            if refresh_incomplete:
                # A refresh is allowed to change the cached surrogate only
                # when it advances toward the ordered mission endpoint. A
                # failed/same extraction must never replace a valid mission
                # route with the synthetic heading fallback.
                old_goal_distance = float(self._route_diagnostics.get(
                    "route_mission_goal_distance_m", float("inf")))
                candidate_goal_distance = float(candidate_diagnostics.get(
                    "route_mission_goal_distance_m", float("inf")))
                old_goal_covered = bool(self._route_diagnostics.get(
                    "route_mission_goal_covered", False))
                candidate_goal_covered = bool(candidate_diagnostics.get(
                    "route_mission_goal_covered", False))
                improves = bool(
                    candidate_route is not None
                    and (
                        (candidate_goal_covered and not old_goal_covered)
                        or candidate_goal_distance < old_goal_distance - 1.0
                    )
                )
                if improves:
                    route = np.asarray(candidate_route, dtype=np.float64)
                    route_source = self.cfg.route_source
                    diagnostics.update(candidate_diagnostics)
                    self._route_refresh_count += 1
                    diagnostics["route_refreshed_incomplete_cache"] = True
                    diagnostics["route_refresh_accepted"] = True
                    self._route_refresh_anchor_xy = None
                else:
                    # ``refresh_incomplete`` implies a non-empty cache.
                    assert self._route_centerline is not None
                    route = np.array(self._route_centerline, copy=True)
                    route_source = self._route_source or self.cfg.route_source
                    diagnostics.update(self._route_diagnostics)
                    diagnostics["route_refresh_rejected_no_progress"] = True
                    diagnostics["route_refresh_attempt_count"] = (
                        self._route_refresh_attempt_count)
                    if np.isfinite(candidate_goal_distance):
                        diagnostics[
                            "route_refresh_candidate_goal_distance_m"] = (
                                candidate_goal_distance)
            else:
                diagnostics.update(candidate_diagnostics)
                if candidate_route is None:
                    route = _synthetic_route_along_heading(
                        ego_x, ego_y, ego_heading,
                        length_m=max(
                            80.0,
                            ego_speed * self.cfg.horizon_s + 20.0),
                        spacing_m=self.cfg.route_densify_spacing_m,
                    )
                    route_source = "synthetic"
                    diagnostics[
                        "route_fallback"] = "synthetic_straight_ahead"
                else:
                    route = np.asarray(candidate_route, dtype=np.float64)
                    route_source = self.cfg.route_source

            consumed = int(diagnostics.get("route_walk_on_route_lanes", 0))
            expected = int(diagnostics.get(
                "route_full_mission_lane_count", len(mission_ids)))
            # Raw lane-count equality is not mission completeness: repeated
            # ids and harmless omitted connector lanes made 11/24 routes look
            # incomplete although most already reached the destination.
            self._route_incomplete = bool(
                mission_ids
                and not diagnostics.get("route_mission_goal_covered", False))
            diagnostics["route_cache_incomplete"] = self._route_incomplete
            if mission_ids:
                diagnostics["route_cache_mission_lanes_consumed"] = consumed
                diagnostics["route_cache_mission_lane_count"] = expected
            self._route_centerline = np.array(route, copy=True)
            self._route_source = route_source
            diagnostics["route_refresh_count"] = self._route_refresh_count
            self._route_diagnostics = dict(diagnostics)
        else:
            route = np.array(self._route_centerline, copy=True)
            route_source = self._route_source or self.cfg.route_source
            diagnostics.update(self._route_diagnostics)
            diagnostics["route_reused_from_cached_extraction"] = True
            if self._route_refresh_count == 0:
                diagnostics["route_reused_from_iteration_zero"] = True
            diagnostics["route_refresh_count"] = self._route_refresh_count

        # 2. Speed limit of the active lane — the reference's starting-lane
        # rule over the route's lanes when the route carries lane identity
        # (``_get_starting_lane``: on-route lane containing the rear axle,
        # min heading error; else nearest on-route lane); the
        # nearest-aligned-lane heuristic only without one.
        walk_lane_ids = diagnostics.get("route_walk_lane_ids")
        if walk_lane_ids:
            speed_limit_mps = _starting_lane_speed_limit_mps(
                scenario_data, ego_state["position"], ego_heading,
                [str(lane_id) for lane_id in walk_lane_ids],
                rear_axle_offset_m=0.5 * float(self.cfg.wheelbase),
                ego_length=float(self.cfg.ego_length),
                ego_width=float(self.cfg.ego_width),
                allow_inferred=self.cfg.allow_inferred_speed_limits,
            )
            diagnostics["speed_limit_rule"] = "reference_starting_lane"
        else:
            speed_limit_mps = _active_lane_speed_limit_mps(
                scenario_data, ego_state["position"], ego_heading,
                allow_inferred=self.cfg.allow_inferred_speed_limits,
            )
            diagnostics["speed_limit_rule"] = "nearest_aligned_lane"
        diagnostics["speed_limit_mps"] = speed_limit_mps

        # 3. Agent predictions over the horizon. A live environment snapshot
        # is authoritative when present: semi-reactive actors can leave their
        # raw log, so forecasting from scenario_data would score a different
        # world from the evaluator. Offline callers retain the configured
        # constant-velocity/log-replay behavior.
        live_agent_states = ego_state.get("_execution_agent_states")
        # Source is otherwise cfg.agent_forecast: constant velocity
        # (upstream-faithful, no oracle) or log replay (the
        # replay env's exact future and the scorer's agent model — an oracle;
        # see the config field's docstring for the trade-off).
        if live_agent_states is not None:
            agents = predict_agents_from_live_states(live_agent_states, self.cfg)
            diagnostics["agent_state_source"] = "live_environment"
        elif self.cfg.agent_forecast == "log_replay":
            agents = predict_agents_log_replay(
                scenario_data, frame_id=frame_id, cfg=self.cfg
            )
            diagnostics["agent_state_source"] = "scenario_log"
        else:
            agents = predict_agents_constant_velocity(
                scenario_data, frame_id=frame_id, cfg=self.cfg
            )
            diagnostics["agent_state_source"] = "scenario_log"
        # CaRL PDMObservation admits tracked objects by their current center
        # only when they are within ``map_radius`` of the current ego. Keep
        # the admitted subset for its full forecast horizon; do not filter
        # future samples independently (an actor does not disappear merely
        # because its prediction later crosses the radius).
        agents_before_radius = len(agents)
        map_radius = float(self.cfg.map_radius_m)
        if map_radius > 0.0:
            agents = {
                track_id: prediction
                for track_id, prediction in agents.items()
                if math.hypot(
                    float(prediction.x[0]) - ego_x,
                    float(prediction.y[0]) - ego_y,
                ) <= map_radius
            }
        diagnostics["map_radius_m"] = map_radius
        diagnostics["num_agents_outside_map_radius"] = (
            agents_before_radius - len(agents)
        )
        # ``PDMObjectManager`` then keeps only the nearest 50 vehicles, 25
        # pedestrians, 10 bicycles and 50 static objects (centre distance at
        # the current frame). Inert below those counts; it matters in the
        # densest scenes (one NavSafe bundle averages 122 actors in 50 m).
        admitted = nearest_k_admitted([
            (track_id, prediction.type,
             math.hypot(float(prediction.x[0]) - ego_x,
                        float(prediction.y[0]) - ego_y))
            for track_id, prediction in agents.items()
        ])
        diagnostics["num_agents_beyond_nearest_k"] = len(agents) - len(admitted)
        if len(admitted) < len(agents):
            agents = {track_id: prediction for track_id, prediction in agents.items()
                      if track_id in admitted}
        newly_collided = _currently_overlapping_agent_ids(
            ego_x, ego_y, ego_heading, agents, self.cfg
        )
        self._collided_track_ids.update(newly_collided)
        if self._collided_track_ids:
            agents = {
                track_id: prediction
                for track_id, prediction in agents.items()
                if str(track_id) not in self._collided_track_ids
            }
        diagnostics["new_collided_track_ids"] = sorted(newly_collided)
        diagnostics["collided_track_ids"] = sorted(self._collided_track_ids)
        diagnostics["num_predicted_agents"] = len(agents)
        diagnostics["agent_forecast"] = self.cfg.agent_forecast

        # 3b. Privileged traffic lights. Every red connector the route is
        # about to enter becomes a stationary obstacle with its rear face on
        # the stop line — upstream's observation semantics (see
        # ``traffic_lights.py``). The obstacles join the predictions that the
        # proposal generator, the lead search, the brake guards and the rule
        # context consume; the SCORER never sees them (it reads scenario
        # tracks / live actor states directly), so a red light constrains
        # the motion and is never charged as a collision — exactly as
        # upstream's ``red_light_`` tokens are skipped by its scorer.
        red_stop_lines: List[RedLightStopLine] = []
        red_obstacles: Dict[str, AgentPrediction] = {}
        scenario_dt = scenario_dt_seconds(
            scenario_data.get("metadata", {}), default=self.cfg.sim_dt)
        if self.cfg.traffic_light_obstacles:
            red_stop_lines = find_red_light_stop_lines(
                scenario_data, route, ego_x=ego_x, ego_y=ego_y,
                frame_id=frame_id, lookahead_frames=0,
                # The lane chain the route was cut from, when the route came
                # from a lane-graph walk: membership by id, not geometry.
                route_lane_ids=diagnostics.get("route_walk_lane_ids"),
                # Upstream drops a red connector only once the ego box is
                # ``within`` it: rear bumper past the stop line.
                release_margin_m=0.5 * float(self.cfg.ego_length))
            if red_stop_lines:
                red_obstacles = red_light_obstacles(red_stop_lines, self.cfg)
                agents = {**agents, **red_obstacles}
        diagnostics["red_light_stop_lines"] = [
            {"lane_id": sl.lane_id, "distance_m": round(sl.distance_m, 3)}
            for sl in red_stop_lines]
        diagnostics["red_light_state_evidence"] = [
            {**traffic_light_state_evidence(scenario_data, sl.lane_id, frame_id),
             "stop_boundary_source": "lane_connector_start"}
            for sl in red_stop_lines]
        diagnostics["red_light_distance_m"] = (
            float(red_stop_lines[0].distance_m) if red_stop_lines else None)
        hold_lines = red_stop_lines
        if self.cfg.traffic_light_obstacles:
            grace_frames = int(round(
                RED_LIGHT_HOLD_GRACE_S / max(float(scenario_dt), 1e-6)))
            if grace_frames > 0:
                hold_lines = find_red_light_stop_lines(
                    scenario_data, route, ego_x=ego_x, ego_y=ego_y,
                    frame_id=max(0, int(frame_id) - grace_frames),
                    # Past-to-present only; never inspect a future light.
                    lookahead_frames=min(grace_frames, int(frame_id)),
                    route_lane_ids=diagnostics.get("route_walk_lane_ids"))
        diagnostics["red_light_hold"] = bool(
            hold_lines
            and hold_lines[0].distance_m <= RED_LIGHT_HOLD_DISTANCE_M)

        # 4. Proposals + forward simulation (with per-step gap update).
        # Scoring reads a stable snapshot of the route and predicted agents.
        route.setflags(write=False)
        for _pred in agents.values():
            for _arr in (_pred.x, _pred.y, _pred.heading, _pred.vx,
                         _pred.vy, _pred.valid):
                _arr.setflags(write=False)
        proposals = generate_proposals(
            ego_x=ego_x,
            ego_y=ego_y,
            ego_heading=ego_heading,
            ego_speed=ego_speed,
            route_centerline=route,
            cfg=self.cfg,
            agents=agents,
            speed_limit_mps=speed_limit_mps,
            ego_acceleration=ego_longitudinal_accel,
            ego_steering_angle=ego_steering_angle,
            ego_angular_velocity=ego_angular_velocity,
        )
        # Keep the stock grid and safety-brake diagnostics separate.
        n_stock = sum(1 for p in proposals if p.proposal_kind == STOCK_PROPOSAL)
        n_safety = sum(1 for p in proposals
                       if p.proposal_kind == SAFETY_BRAKE_PROPOSAL)
        diagnostics["num_stock_proposals"] = n_stock
        diagnostics["num_safety_proposals"] = n_safety

        # 5. Scoring.
        # Upstream's driving-direction term asks whether the ego centre is
        # inside an ON-ROUTE drivable polygon (pdm_scorer.py:282-284), so the
        # scorer needs the route's lane identity, not just its centreline.
        # Same channel as ``prev_frame_idx`` below; ``None`` means "unknown",
        # which the scorer treats as "do not grade DDC" rather than silently
        # grading every lane as on-route.
        walk_ids = diagnostics.get("route_walk_lane_ids")
        # Upstream's on-route polygon set is ``route_lane_dict``: EVERY lane
        # of every route roadblock and roadblock-connector, not just the lanes
        # the centreline runs through. A same-direction neighbour lane is
        # on-route there; charging it as oncoming (the walk-only set) made
        # DDC bite on the ±1 m offsets at narrow lanes and turns, which
        # upstream never does. py123d's lane group is the roadblock analogue.
        route_set = (
            self._route_roadblock_lane_ids(scenario_data, walk_ids)
            if walk_ids else None)
        cast(Any, self._scorer.scorer).route_lane_ids = route_set
        diagnostics["route_on_route_lane_count"] = (
            len(route_set) if route_set is not None else 0)

        if (
            self._last_committed_frame_id is None
            or frame_id <= self._last_committed_frame_id
        ):
            self._scorer.scorer.prev_frame_idx = None
        else:
            # ``prev_frame_idx`` is inherited from EPDMSTrajectoryScorer_Fast,
            # which initialises it to ``None`` (so mypy infers a ``None`` type),
            # but it genuinely holds an int frame index. Cast the holder to set
            # it without a spurious assignment error.
            cast(Any, self._scorer.scorer).prev_frame_idx = self._last_committed_frame_id

        scores, brake_ref_score, brake_in_proposals = (
            self._score_with_brake_reference(
                proposals, ego_state, frame_id=frame_id, route=route,
                ego_x=ego_x, ego_y=ego_y, ego_heading=ego_heading,
                ego_speed=ego_speed,
                ego_longitudinal_accel=ego_longitudinal_accel,
                ego_steering_angle=ego_steering_angle,
                ego_angular_velocity=ego_angular_velocity,
                agents=agents,
                speed_limit_mps=speed_limit_mps))

        if (
            self._last_committed_frame_id is None
            or frame_id > self._last_committed_frame_id
        ):
            self._last_committed_frame_id = frame_id

        # Population stats over NOMINAL proposals only. Folding the safety
        # brake in changed the population behind score_max/min/mean between
        # brake modes (cross-arm dump comparisons compared different
        # populations), and let score_max exceed the best_score that gated
        # 6a — a reader sees "max > threshold, why did the brake fire?".
        # The brake's own score is reported under its own key instead.
        nominal = [i for i, p in enumerate(proposals)
                   if p.proposal_kind != SAFETY_BRAKE_PROPOSAL]
        # The base grid is guaranteed non-empty: generate_proposals raises
        # unless it produced exactly cfg.num_proposals stock proposals
        # (proposals.py), and __post_init__ forces non-empty offset/policy
        # axes. The brake is appended after that check, so `nominal` can
        # only be empty if that invariant breaks in another module — in
        # which case executing a brake through the NORMAL selection path
        # (no emergency_brake_triggered flag) must not happen silently.
        assert nominal, "base proposal grid is empty — invariant broken"
        nominal_scores = np.asarray(scores)[nominal]
        diagnostics["score_max"] = float(nominal_scores.max())
        diagnostics["score_min"] = float(nominal_scores.min())
        diagnostics["score_mean"] = float(nominal_scores.mean())
        diagnostics["safety_brake_score"] = brake_ref_score
        # Whether that score belongs to a proposal a reader can find in
        # ``result.proposals`` (candidate mode) or to the scoring-only
        # reference (trajectory mode). Without this a dump shows a
        # safety_brake_score with no matching proposal and looks corrupt.
        diagnostics["safety_brake_in_proposals"] = brake_in_proposals

        # 6. Selection + emergency-brake guard.
        # The safety brake is scored and logged like any candidate, but it
        # is a FLOOR, not an option: letting it win the nominal argmax
        # would make the planner choose to stop whenever stopping happens
        # to score well, which is a behaviour change, not a safety net.
        # It is selected only by the guard branches below. Unconditional
        # now that the brake is always scored: with no safety proposal in
        # the set this is identical to the scorer's own argmax (both take
        # the first index attaining the maximum).
        best_idx = nominal[int(np.argmax(nominal_scores))]
        best_score = float(scores[best_idx])

        def _brake_result(reason: str) -> PDMPlanResult:
            """Execute the brake via the one mode-aware helper."""
            brake_traj, chosen_idx, execution = resolve_brake(
                self.cfg, proposals, ego_x=ego_x, ego_y=ego_y,
                ego_heading=ego_heading, ego_speed=ego_speed, agents=agents,
                speed_limit_mps=speed_limit_mps,
                ego_longitudinal_accel=ego_longitudinal_accel)
            diagnostics["emergency_brake_execution"] = execution
            diagnostics["emergency_brake_candidate_fallback"] = (
                is_candidate_fallback(self.cfg, execution))
            diagnostics["emergency_brake_reason"] = reason
            # Arm as a bare code so an aggregator counting fire rates per
            # arm never prefix-parses prose. 6b has no score arm.
            diagnostics["emergency_brake_arm"] = (
                reason.split(":", 1)[0]
                if reason.startswith((BRAKE_ARM_NO_VALID_PROPOSAL,
                                      BRAKE_ARM_STOPPING_BETTER,
                                      BRAKE_ARM_RED_LIGHT))
                else "imminent_collision")
            diagnostics["selected_proposal_kind"] = (
                proposals[chosen_idx].proposal_kind if chosen_idx >= 0
                else SYNTHETIC_BRAKE_EXECUTION)
            return PDMPlanResult(
                trajectory=brake_traj,
                best_idx=chosen_idx,
                scores=scores,
                proposals=proposals,
                emergency_brake_triggered=True,
                route_source=route_source,
                # Upstream PDMEmergencyBrake deliberately gives every
                # generated EgoState zero center velocity. Its displaced
                # poses are a controller correction signal, not a speed
                # profile. Candidate mode is a NexusSim extension and keeps
                # the safety proposal's actual simulated velocity instead.
                trajectory_speeds_mps=(
                    _extend_trajectory_and_speeds_to_horizon(
                        proposals[chosen_idx], ego_x, ego_y, ego_heading,
                        ego_speed, agents, speed_limit_mps, self.cfg)[1]
                    if execution == "candidate" and chosen_idx >= 0
                    else np.zeros(len(brake_traj), dtype=np.float64)
                ),
                diagnostics=diagnostics,
                )

        # 6r. Optional NexusSim red-light floor (see
        # red_light_brake_reach_m): every nominal
        # proposal overruns a red stop line within an emergency stop's reach.
        # Checked before 6a and at ANY speed — the 6a/6b speed gate is what
        # would otherwise let the argmax of an all-zero grid drive through
        # the light — so the arm label does not depend on the ego speed.
        # Test the physical condition directly. The previous ``score <= 0``
        # proxy stopped working once the port correctly matched upstream's
        # scorer, which skips red-light observation tokens. On real 0bcae698
        # frame 62 all 15 proposals crossed the live-red line while their PDM
        # scores remained 0.663--0.833, so that proxy let the ego drive on.
        red_grid_overruns = bool(
            red_stop_lines
            and all(proposal_crosses_red_stop_line(
                proposals[i], red_stop_lines[0], self.cfg) for i in nominal)
        )
        diagnostics["red_light_all_nominal_overrun"] = red_grid_overruns
        if (self.cfg.red_light_brake_floor
                and red_grid_overruns
                and red_stop_lines[0].distance_m
                <= red_light_brake_reach_m(ego_speed, self.cfg)):
            self._score_brake_streak += 1
            diagnostics["score_brake_streak"] = self._score_brake_streak
            return _brake_result(
                f"{BRAKE_ARM_RED_LIGHT}: no valid proposal with the red stop "
                f"line {red_stop_lines[0].distance_m:.1f} m ahead "
                f"(lane {red_stop_lines[0].lane_id}) at "
                f"ego_speed={ego_speed:.2f} m/s; all {len(nominal)} nominal "
                f"proposals overrun; best_score={best_score:.6f}")

        # 6a. Score-based fallback, fired on a REASON rather than on a
        # score magnitude: either no proposal is valid at all, or the
        # argmax is worse than simply stopping — both decided by
        # :func:`score_brake_reason`, which owns the rationale.
        # Speed-gated like 6b: upstream executes the argmax proposal
        # even when every score is 0 (no score-based brake exists
        # there at all), and an ungated brake at speed emits a
        # dead-straight stop plan regardless of road curvature. Above
        # the gate we fall through and execute the argmax proposal.
        score_reason = score_brake_reason(
            best_score=best_score,
            brake_score=brake_ref_score,
            threshold=float(self.cfg.emergency_brake_threshold),
            arms=str(getattr(self.cfg, "emergency_brake_score_arms",
                             "both")),
        )
        if (
            score_reason is not None
            and ego_speed <= self.cfg.emergency_brake_max_ego_speed
        ):
            # Both arms are sticky on a static cause (see
            # score_brake_reason): the ego stops, the next replan reads
            # the same pose, and the guard re-fires. Count consecutive
            # fires so a hold is legible in a dump instead of showing up
            # as an unexplained zero-progress episode.
            self._score_brake_streak += 1
            diagnostics["score_brake_streak"] = self._score_brake_streak
            return _brake_result(score_reason)
        self._score_brake_streak = 0
        diagnostics["score_brake_streak"] = 0

        chosen = proposals[best_idx]

        # 6b. The only upstream emergency guard: inspect the selected
        # proposal's collision time recorded by the SAME scorer invocation.
        # A second geometry/TTC approximation here can disagree with the
        # argmax scorer and is not how PDMEmergencyBrake is invoked upstream.
        if ego_speed > self.cfg.emergency_brake_max_ego_speed:
            diagnostics["selected_ttc"] = float("inf")
            ttc = float("inf")
        else:
            collision_times = self._scorer.last_collision_times
            ttc = (float(collision_times[best_idx])
                   if best_idx < len(collision_times) else float("inf"))
            diagnostics["selected_ttc"] = ttc
        if (
            ttc <= self.cfg.infraction_horizon_s
            and ego_speed <= self.cfg.emergency_brake_max_ego_speed
        ):
            return _brake_result(
                f"selected_ttc={ttc:.3f}s <= "
                f"infraction_horizon={self.cfg.infraction_horizon_s:.3f}s "
                f"and ego_speed={ego_speed:.3f}m/s <= "
                f"max_brake_speed={self.cfg.emergency_brake_max_ego_speed:.3f}m/s"
            )

        # Normal path.
        trajectory, trajectory_speeds = _extend_trajectory_and_speeds_to_horizon(
            chosen,
            ego_x,
            ego_y,
            ego_heading,
            ego_speed,
            agents,
            speed_limit_mps,
            self.cfg,
        )
        diagnostics["selected_offset"] = float(chosen.lateral_offset)
        diagnostics["selected_policy"] = chosen.idm.name
        diagnostics["selected_longitudinal_idx"] = int(chosen.longitudinal_idx)
        diagnostics["selected_lateral_idx"] = int(chosen.lateral_idx)
        diagnostics["selected_proposal_kind"] = chosen.proposal_kind

        return PDMPlanResult(
            trajectory=trajectory,
            best_idx=int(best_idx),
            scores=scores,
            proposals=proposals,
            emergency_brake_triggered=False,
            route_source=route_source,
            trajectory_speeds_mps=trajectory_speeds,
            diagnostics=diagnostics,
        )


__all__ = [
    "PDMPlanResult",
    "PDMPlanner",
    "emergency_brake_execution_trajectory",
    "emergency_brake_trajectory",
    "result_execution_trajectory",
    "is_candidate_fallback",
    "proposal_time_to_collision",
    "resolve_brake",
    "scenario_identity_key",
    "score_brake_reason",
    "BRAKE_ARM_NO_VALID_PROPOSAL",
    "BRAKE_ARM_STOPPING_BETTER",
    "BRAKE_ARM_RED_LIGHT",
    "proposal_crosses_red_stop_line",
    "red_light_brake_reach_m",
    "trajectory_time_to_collision",
]
