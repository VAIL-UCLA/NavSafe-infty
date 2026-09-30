# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""PDM-Closed proposal generation.

Per replan, PDM-Closed generates ``len(lateral_offsets) × len(idm_policies)``
trajectory proposals (3 × 5 = 15 by default), forward-simulates each
one with :func:`simulate_proposal`, and hands the bundle to the scorer
for selection.

This module owns the *generation* half of that pipeline:

* :func:`offset_polyline` — laterally shift a centerline polyline by
  ``offset`` metres (positive = left of travel direction).
* :func:`find_lead_along_centerline` — project every agent's initial
  position onto the centerline; return the closest in-lane lead's
  longitudinal gap and centerline-aligned speed. This replaces the
  legacy rectangular-gate detector (``find_lead_vehicle``) which is
  retained for backward compatibility but is *not* used by
  :func:`generate_proposals` after the faithful refactor.
* :class:`Proposal` — output container.
* :func:`generate_proposals` — the entry point.

Iteration order matches upstream ``tuplan_garage`` / ``CaRL``
:class:`PDMProposalManager` exactly: **outer = lateral offset
(centerline first), inner = IDM policy**, producing
``proposal_idx = longitudinal_idx + num_longitudinal * lateral_idx``
— the same value as upstream's ``lateral_idx * num_longitudinal +
longitudinal_idx``. Centerline-first is behaviourally load-bearing on
exact score ties: ``np.argmax`` returns the first maximiser, so ties
resolve toward the centerline and then the slower IDM policy, as
upstream.

The proposal-grid layout (default offsets ``(0.0, -1.0, +1.0)``,
policies ``P_20..P_100``)::

    proposal_idx | lateral | longitudinal
    -------------+---------+--------------
    0..4         |   0.0   | P_20..P_100
    5..9         |  -1.0   | P_20..P_100
    10..14       |  +1.0   | P_20..P_100

(Pinned by ``test_proposal_idx_mapping_matches_upstream`` in
``tests/policy/pdm_closed/test_proposals.py``.)

Determinism: no RNG, no parallelism with non-deterministic ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from navsafe.policy.state.pdm_closed_planner.config import PDMConfig
from navsafe.policy.state.pdm_closed_planner.forward_sim import (
    AgentPrediction,
    ProposalState,
    simulate_proposal,
)
from navsafe.policy.state.pdm_closed_planner.idm import IDMPolicy



# ----------------------------------------------------------------------
# Public output container
# ----------------------------------------------------------------------

#: Proposal types distinguish the stock grid from the safety brake.
STOCK_PROPOSAL = "stock"
SAFETY_BRAKE_PROPOSAL = "safety_brake"
#: Diagnostics-only pseudo-kind: the value ``selected_proposal_kind``
#: reports when a fired guard SYNTHESISED the integrated stop profile
#: instead of selecting a proposal (``trajectory`` mode, or a candidate-
#: mode fallback). Never a valid ``Proposal.proposal_kind`` — defined
#: beside the taxonomy so diagnostics consumers have one authoritative
#: vocabulary instead of a magic string.
SYNTHETIC_BRAKE_EXECUTION = "synthetic_brake"


@dataclass
class Proposal:
    """One generated and forward-simulated trajectory proposal.

    Attributes:
        proposal_idx: Stable index into the proposal list. Used by the
            scorer to write back per-proposal scores. Indexing matches
            ``tuplan_garage``'s lateral-major convention:
            ``proposal_idx = longitudinal_idx + num_longitudinal *
            lateral_idx`` (see the module docstring's layout table).
        longitudinal_idx: Index of the IDM policy in
            ``cfg.idm_policies`` (0 = slowest, 4 = fastest by default).
        lateral_idx: Index of the lateral offset in
            ``cfg.lateral_offsets``.
        lateral_offset: Lateral offset (m). Positive = left of route
            direction.
        idm: The IDM policy that owns this proposal's speed profile.
        path_world_xy: ``(M, 2)`` world-frame polyline (the
            laterally-offset centerline).
        state: :class:`ProposalState` from the forward simulation.
        proposal_kind: ``"stock"`` for the base grid or ``"safety_brake"``
            for the forward-simulated maximum-deceleration stop.
    """

    proposal_idx: int
    longitudinal_idx: int
    lateral_idx: int
    lateral_offset: float
    idm: IDMPolicy
    path_world_xy: np.ndarray
    state: ProposalState
    # Ideal IDM-on-path proposal returned to execution. ``state`` is the
    # dynamically simulated copy used only for scoring.
    ideal_state: Optional[ProposalState] = None
    proposal_kind: str = STOCK_PROPOSAL

    @property
    def output_xy_ego(self) -> np.ndarray:
        """Convenience accessor — ego-frame ``[lateral, forward]`` output."""
        source = self.ideal_state if self.ideal_state is not None else self.state
        return source.output_xy_ego


# A safety brake is outside the lateral/longitudinal stock grid.
BRAKE_GRID_IDX = -1


def build_brake_proposal(
    *,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    path_world_xy: np.ndarray,
    cfg: PDMConfig,
    proposal_idx: int,
    agents: Dict[str, AgentPrediction] | None = None,
    speed_limit_mps: float | None = None,
    lead_dist: float | None = None,
    lead_speed: float = 0.0,
    ego_acceleration: float = 0.0,
    ego_steering_angle: float = 0.0,
    ego_angular_velocity: float = 0.0,
) -> Proposal:
    """A maximum-deceleration stop, as an ordinary forward-simulated candidate.

    Commands ``-cfg.emergency_brake_decel`` through the same actuator
    caps, tracker and bicycle model every other proposal uses, so the
    executed motion is what the vehicle can physically do. This is the
    safety floor: it keeps braking available to selection — preserving
    a physically feasible stopping option — without any component writing
    a position profile straight to the ego.

    Steering still tracks ``path_world_xy``, so the stop follows the road
    rather than the dead-straight line an ego-frame profile produces
    regardless of curvature.

    Two callers, one object. ``generate_proposals`` appends it under
    ``emergency_brake_mode == "candidate"``, where it is a selectable
    floor. :meth:`PDMPlanner.plan` builds one below the brake speed gate
    in EVERY mode as the reference guard 6a compares the argmax against
    (:func:`~navsafe.policy.state.pdm_closed_planner.planner.
    score_brake_reason`) — in ``trajectory`` mode for scoring only, kept
    out of ``result.proposals``.

    ``lead_dist`` / ``lead_speed`` seed the lead observation. The
    commanded ``longitudinal_accel_mps2`` replaces the IDM *longitudinal*
    term outright, but under the default LQR tracker the seed also feeds
    the lateral controller's velocity profile — so the two call sites
    agree only because both pass ``agents``, which makes
    ``simulate_proposal`` recompute the gap along ``path_world_xy`` from
    step 0 and discard the seed. **Pass ``agents``.** Calling this with
    ``agents=None`` and no seed yields a laterally different stop, and
    trajectory-mode guard 6a would then compare against a stop that
    candidate mode would not execute.
    """
    state = simulate_proposal(
        initial_x=float(ego_x),
        initial_y=float(ego_y),
        initial_heading=float(ego_heading),
        initial_speed=float(ego_speed),
        path_xy=path_world_xy,
        # Unused for the longitudinal command (overridden below) but the
        # proposal record requires a policy; the slowest is the least
        # misleading label.
        policy=cfg.idm_policies[0],
        cfg=cfg,
        lead_dist=lead_dist,
        lead_speed=lead_speed,
        agents=agents,
        speed_limit_mps=speed_limit_mps,
        longitudinal_accel_mps2=-float(cfg.emergency_brake_decel),
        initial_acceleration=float(ego_acceleration),
        initial_steering_angle=float(ego_steering_angle),
        initial_angular_velocity=float(ego_angular_velocity),
    )
    return Proposal(
        proposal_idx=int(proposal_idx),
        longitudinal_idx=BRAKE_GRID_IDX,
        lateral_idx=BRAKE_GRID_IDX,
        lateral_offset=0.0,
        idm=cfg.idm_policies[0],
        path_world_xy=np.asarray(path_world_xy, dtype=np.float64),
        state=state,
        proposal_kind=SAFETY_BRAKE_PROPOSAL,
    )


# ----------------------------------------------------------------------
# Lateral offset
# ----------------------------------------------------------------------


def offset_polyline(polyline: np.ndarray, offset: float) -> np.ndarray:
    """Shift each vertex of ``polyline`` by ``offset`` along its left normal.

    Reference ``parallel_discrete_path``: every vertex moves along
    ``heading + π/2`` where the vertex heading is the one the map carries —
    the forward-segment tangent, the last vertex repeating the one before
    (nuPlan ``maps/nuplan_map/utils.py``). The port used to average the two
    adjacent tangents (1–5 cm different on 1 m-densified arcs of R = 50–10 m).

    Args:
        polyline: ``(M, 2)`` float array.
        offset: Lateral offset (m). Positive = left of travel direction.

    Returns:
        ``(M, 2)`` offset polyline. Same vertex count as input.
    """
    p = np.asarray(polyline, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
        raise ValueError(f"polyline must have shape (M>=2, 2), got {p.shape}")
    if offset == 0.0:
        return p.copy()

    seg = np.diff(p, axis=0)
    seg_lens = np.linalg.norm(seg, axis=1)
    seg_lens_safe = np.where(seg_lens > 1e-12, seg_lens, 1.0)
    tangents = seg / seg_lens_safe[:, None]

    vertex_tangents = np.empty_like(p)
    vertex_tangents[:-1] = tangents
    vertex_tangents[-1] = tangents[-1]

    # Left normal: rotate tangent 90° CCW: (tx, ty) -> (-ty, tx).
    left_normals = np.column_stack([-vertex_tangents[:, 1], vertex_tangents[:, 0]])
    return p + offset * left_normals


# ----------------------------------------------------------------------
# Lead-vehicle detection — centerline-based (faithful)
# ----------------------------------------------------------------------


def find_lead_along_centerline(
    agents: Dict[str, AgentPrediction],
    centerline: np.ndarray,
    ego_x: float,
    ego_y: float,
    *,
    lane_half_width_m: float = 1.75,
    max_longitudinal_m: float = 100.0,
    ego_length: float = 0.0,
) -> tuple[float | None, float]:
    """Return ``(gap_m, lead_speed_m_s)`` for the nearest in-lane lead
    along the centerline.

    ``gap_m`` approximates the bumper-to-bumper distance: the
    centre-to-centre arc-length delta minus half the ego length
    (``ego_length / 2``, pass ``cfg.ego_length``) and half the agent
    length. Centre-to-centre gaps overstate the true gap by ~4.5 m for
    two cars, causing unsafe following distances.

    Faithful version of lead detection: every agent is projected onto
    the centerline arc-length axis. An agent is considered "in lane"
    when its perpendicular distance to the centerline is below
    ``lane_half_width_m``. The lead is the in-lane agent with the
    smallest positive ``s_agent - s_ego`` (closest agent ahead of the
    ego along the path). Lead speed is the agent's velocity projected
    onto the centerline tangent at the agent's projection point.

    The legacy rectangular-gate version
    (:func:`find_lead_vehicle`) is retained for backward compatibility
    but is *not* used by :func:`generate_proposals` after the faithful
    refactor.

    Args:
        agents: Agent predictions from
            :func:`predict_agents_constant_velocity`. Only the
            ``t=0`` state is read.
        centerline: ``(M, 2)`` route centerline polyline.
        ego_x / ego_y: Ego XY world position.
        lane_half_width_m: Half the lane width (m). Agents farther
            laterally are excluded. Default 1.75 m corresponds to a
            3.5 m lane (NavSim urban) — adjust if your scenarios use
            wider lanes.
        max_longitudinal_m: Beyond this distance the IDM kernel sees
            a free road regardless.

    Returns:
        ``(gap, speed)``. ``gap`` is ``None`` when no lead is in
        range. ``speed`` is the lead's centerline-tangent-aligned
        speed (signed; negative for oncoming traffic).
    """
    if not agents:
        return None, 0.0
    if centerline.shape[0] < 2:
        return None, 0.0

    # Pre-compute centerline cumulative arc length and tangents.
    seg = np.diff(centerline, axis=0)
    seg_lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    seg_lens_safe = np.where(seg_lens > 1e-12, seg_lens, 1.0)
    tangents = seg / seg_lens_safe[:, None]

    def _project(px: float, py: float) -> tuple[float, float, np.ndarray]:
        """Return ``(s, |r|, tangent)`` of point on centerline closest to ``(px, py)``."""
        # Project the point onto every segment; pick the segment with
        # the smallest perpendicular distance.
        rel = np.array([px, py], dtype=np.float64) - centerline[:-1]
        # Per-segment scalar projection ``t in [0, 1]``.
        t_unclamped = np.einsum("ij,ij->i", rel, seg) / np.maximum(seg_lens ** 2, 1e-24)
        t = np.clip(t_unclamped, 0.0, 1.0)
        proj = centerline[:-1] + t[:, None] * seg
        d = np.linalg.norm(proj - np.array([px, py]), axis=1)
        idx = int(np.argmin(d))
        return (
            float(cum[idx] + t[idx] * seg_lens[idx]),
            float(d[idx]),
            tangents[idx],
        )

    s_ego, _r_ego, _tan_ego = _project(ego_x, ego_y)

    best_gap: float | None = None
    best_speed: float = 0.0

    for pred in agents.values():
        if not bool(pred.valid[0]):
            continue
        ax = float(pred.x[0])
        ay = float(pred.y[0])
        s_a, r_a, tan_a = _project(ax, ay)
        # Skip agents off-lane.
        if r_a > lane_half_width_m:
            continue
        # Skip agents behind the ego (centre-to-centre along the path).
        center_gap = s_a - s_ego
        if center_gap <= 0.0:
            continue
        if center_gap > max_longitudinal_m:
            continue
        # Bumper-to-bumper approximation along the centerline.
        gap = max(
            0.0, center_gap - 0.5 * ego_length - 0.5 * float(pred.length)
        )
        if best_gap is None or gap < best_gap:
            best_gap = gap
            # Lead speed projected onto the centerline tangent at the
            # agent's projection point. Signed: negative for oncoming
            # traffic so IDM's closing-rate term brakes harder.
            v = np.array([float(pred.vx[0]), float(pred.vy[0])])
            best_speed = float(np.dot(v, tan_a))

    return best_gap, best_speed


def find_lead_vehicle(
    agents: Dict[str, AgentPrediction],
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    *,
    max_lateral_m: float = 2.5,
    max_longitudinal_m: float = 100.0,
) -> tuple[float | None, float]:
    """Legacy rectangular-gate lead detection.

    Retained so callers that imported the old API continue to work,
    but :func:`generate_proposals` uses
    :func:`find_lead_along_centerline` instead. See that function for
    the faithful implementation.
    """
    if not agents:
        return None, 0.0

    cos_h = float(np.cos(ego_heading))
    sin_h = float(np.sin(ego_heading))

    best_gap: float | None = None
    best_speed: float = 0.0

    for pred in agents.values():
        if not bool(pred.valid[0]):
            continue
        dx = float(pred.x[0]) - ego_x
        dy = float(pred.y[0]) - ego_y
        longitudinal = dx * cos_h + dy * sin_h
        lateral = -dx * sin_h + dy * cos_h
        if longitudinal <= 0.0 or longitudinal > max_longitudinal_m:
            continue
        if abs(lateral) > max_lateral_m:
            continue
        if best_gap is None or longitudinal < best_gap:
            best_gap = longitudinal
            forward_speed = float(pred.vx[0]) * cos_h + float(pred.vy[0]) * sin_h
            best_speed = max(0.0, forward_speed)

    return best_gap, best_speed


# ----------------------------------------------------------------------
# Proposal grid
# ----------------------------------------------------------------------


def generate_proposals(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    route_centerline: np.ndarray,
    cfg: PDMConfig,
    *,
    agents: Dict[str, AgentPrediction] | None = None,
    speed_limit_mps: float | None = None,
    ego_acceleration: float = 0.0,
    ego_steering_angle: float = 0.0,
    ego_angular_velocity: float = 0.0,
) -> List[Proposal]:
    """Generate the ``N_offsets × N_idm`` proposal grid.

    Iteration order: outer loop is the lateral offset (centerline
    first), inner loop is the IDM policy — the same lateral-major
    convention as upstream ``tuplan_garage`` / ``CaRL``'s
    :class:`PDMProposalManager`. The resulting ``proposal_idx`` is
    ``longitudinal_idx + num_longitudinal * lateral_idx``. See the
    module docstring for the full layout table; the ordering is
    behaviourally relevant on exact score ties (``np.argmax`` takes
    the first maximiser).

    Args:
        ego_x / ego_y / ego_heading / ego_speed: Ego state at the
            planning frame.
        route_centerline: ``(M, 2)`` route polyline.
        cfg: PDM configuration.
        agents: Agent predictions used to find the in-lane lead.
        speed_limit_mps: Lane speed limit (m/s) used to resolve each
            policy's target velocity. ``None`` falls back to each
            policy's ``fallback_target_velocity``.

    Returns:
        List of :class:`Proposal` of length ``cfg.num_proposals``.

    Raises:
        ValueError: If ``route_centerline`` is too short or
            ``ego_speed`` is negative.
    """
    if route_centerline is None:
        raise ValueError("route_centerline must not be None")
    if route_centerline.shape[0] < 2:
        raise ValueError(
            f"route_centerline must have >=2 vertices, got {route_centerline.shape}"
        )
    if ego_speed < 0.0:
        raise ValueError(f"ego_speed must be non-negative, got {ego_speed}")

    # Faithful lead detection — projection onto the centerline.
    lead_gap, lead_speed = find_lead_along_centerline(
        agents or {},
        centerline=route_centerline,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_length=float(cfg.ego_length),
    )


    proposals: List[Proposal] = []
    num_longitudinal = len(cfg.idm_policies)
    # Upstream ordering is lateral-major with the centerline first. This is
    # observable whenever scores tie because np.argmax selects the first.
    for lateral_idx, offset in enumerate(cfg.lateral_offsets):
        for longitudinal_idx, policy in enumerate(cfg.idm_policies):
            offset_path = offset_polyline(route_centerline, float(offset))
            ideal_state = simulate_proposal(
                initial_x=float(ego_x),
                initial_y=float(ego_y),
                initial_heading=float(ego_heading),
                initial_speed=float(ego_speed),
                path_xy=offset_path,
                policy=policy,
                cfg=cfg,
                lead_dist=lead_gap,
                lead_speed=lead_speed,
                agents=agents,
                speed_limit_mps=speed_limit_mps,
                ideal_kinematics=True,
                initial_acceleration=float(ego_acceleration),
                initial_steering_angle=float(ego_steering_angle),
                initial_angular_velocity=float(ego_angular_velocity),
            )
            state = simulate_proposal(
                initial_x=float(ego_x),
                initial_y=float(ego_y),
                initial_heading=float(ego_heading),
                initial_speed=float(ego_speed),
                path_xy=offset_path,
                policy=policy,
                cfg=cfg,
                agents=agents,
                speed_limit_mps=speed_limit_mps,
                reference_state=ideal_state,
                initial_acceleration=float(ego_acceleration),
                initial_steering_angle=float(ego_steering_angle),
                initial_angular_velocity=float(ego_angular_velocity),
            )
            proposal_idx = longitudinal_idx + num_longitudinal * lateral_idx
            proposals.append(
                Proposal(
                    proposal_idx=proposal_idx,
                    longitudinal_idx=longitudinal_idx,
                    lateral_idx=lateral_idx,
                    lateral_offset=float(offset),
                    idm=policy,
                    path_world_xy=offset_path,
                    state=state,
                    ideal_state=ideal_state,
                )
            )

    if len(proposals) != cfg.num_proposals:
        raise RuntimeError(
            f"generate_proposals produced {len(proposals)} proposals, "
            f"expected {cfg.num_proposals}"
        )

    # Defensive sort by proposal_idx so the output is in
    # canonical lexicographic order regardless of insertion order.
    proposals.sort(key=lambda p: p.proposal_idx)

    if cfg.emergency_brake_mode == "candidate":
        proposals.append(build_brake_proposal(
            ego_x=ego_x, ego_y=ego_y, ego_heading=ego_heading,
            ego_speed=ego_speed, path_world_xy=route_centerline, cfg=cfg,
            proposal_idx=len(proposals), agents=agents,
            speed_limit_mps=speed_limit_mps, lead_dist=lead_gap,
            lead_speed=lead_speed,
            ego_acceleration=ego_acceleration,
            ego_steering_angle=ego_steering_angle,
            ego_angular_velocity=ego_angular_velocity))
    return proposals


__all__ = [
    "Proposal",
    "STOCK_PROPOSAL",
    "SAFETY_BRAKE_PROPOSAL",
    "SYNTHETIC_BRAKE_EXECUTION",
    "build_brake_proposal",
    "offset_polyline",
    "find_lead_along_centerline",
    "find_lead_vehicle",
    "generate_proposals",
]
