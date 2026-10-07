# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Forward simulation for PDM-Closed proposals.

Per-proposal pipeline:

1. Track an *offset path* with either an LQR tracker (faithful
   default, matching ``tuplan_garage/batch_lqr.py``) or a pure-pursuit
   tracker (legacy / opt-in).
2. Generate the ideal speed with IDM and CaRL's two-step lead refresh
   cadence (including its zero-initialized first lead row), then track the
   ideal pose-derived velocity profile with CaRL's longitudinal LQR/stopping
   controller.
3. Return :class:`ProposalState` — every quantity the scorer needs.

The bicycle model is :class:`navsafe.core.ego_dynamics.EgoDynamics`,
the canonical NavSafe implementation, so PDM-Closed's simulated
proposals are dynamically consistent with what the env would execute
under the same control inputs.

Other-agent prediction: constant velocity over the horizon, matching
``tuplan_garage``'s constant-velocity branch.
"""

from __future__ import annotations

import logging
import math
import pickle
import struct
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Optional, ParamSpec, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon

from navsafe.core.agent_forecast import forecast_velocity
from navsafe.core.collision_actor_types import COLLIDABLE_ACTOR_TYPES_TUPLE
from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg
# The lateral path followers (`_project_point_on_path`,
# `_pure_pursuit_steer_norm`, `_lqr_steer`) were written here as faithful
# BridgeSim / tuplan_garage ports and are re-imported from their new home in
# `navsafe.core.steering` — they moved down into core (verbatim) so that
# `navsafe.core.controllers` can share them without a core → policy import
# cycle. Import them from here or from core; both names are the same objects.
from navsafe.core.steering import (
    _lqr_steer,
    _project_point_on_path,
    _pure_pursuit_steer_norm,
)
from navsafe.core.track_dims import track_dims
from navsafe.policy.state.pdm_closed_planner.config import PDMConfig
from navsafe.policy.state.pdm_closed_planner.dynamics import (
    ActuatorFilterParams,
    FilteredEgoDynamics,
)
from navsafe.policy.state.pdm_closed_planner.idm import IDMPolicy, idm_accel
from navsafe.policy.state.pdm_closed_planner.reference_plant import (
    ReferenceBicyclePlant,
)
from navsafe.policy.state.pdm_closed_planner.lqr_profiles import (
    generate_profile,
    longitudinal_lqr_acceleration,
    reference_slice,
    velocity_curvature_profiles_from_poses,
)
from navsafe.scenario.scenario_description import scenario_dt_seconds

_logger = logging.getLogger(__name__)

# Shapely 2 geometry values are immutable. This cache exists only inside one
# refinement bank; all candidate-specific corridor and safety checks still run.
_VEHICLE_POLYGON_CACHE_LIMIT = 8192
_active_vehicle_polygon_cache: ContextVar[Optional[OrderedDict[bytes, Polygon]]] = ContextVar(
    "refinement_vehicle_polygon_cache", default=None)


# ----------------------------------------------------------------------
# Geometry helpers (Shapely-based, faithful to CaRL)
# ----------------------------------------------------------------------


def _vehicle_polygon(
    cx: float, cy: float, heading: float, length: float, width: float
) -> Polygon:
    """Oriented rectangle for a vehicle footprint at ``(cx, cy, heading)``.

    Mirrors CaRL's ``CarFootprint.build_from_rear_axle(...).oriented_box.geometry``
    in shape (a 4-corner box aligned with ``heading``). NavSafe's
    centroid convention places the rectangle centre at ``(cx, cy)``
    rather than the rear axle; this is consistent with how the rest of
    NavSafe (and ScenarioNet) carries vehicle poses.
    """
    cache = _active_vehicle_polygon_cache.get()
    key = None
    if cache is not None:
        values = (cx, cy, heading, length, width)
        # Restrict reuse to exact builtin floats. Other scalar types and
        # nonfinite inputs retain the original constructor/error behavior.
        if all(type(value) is float and math.isfinite(value) for value in values):
            key = struct.pack("!5d", *values)  # Includes every bit and signed zero.
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached
    hx = 0.5 * length
    hy = 0.5 * width
    c = math.cos(heading)
    s = math.sin(heading)
    body = ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))
    corners = [(cx + c * x - s * y, cy + s * x + c * y) for x, y in body]
    polygon = Polygon(corners)
    if key is not None:
        cache[key] = polygon
        if len(cache) > _VEHICLE_POLYGON_CACHE_LIMIT:
            cache.popitem(last=False)
    return polygon


def _vertex_headings(tangents: np.ndarray) -> np.ndarray:
    """Per-vertex headings the way the reference map carries them.

    nuPlan's discrete baseline gives every vertex the heading of its forward
    segment, the last vertex repeating the one before
    (``maps/nuplan_map/utils.py``), and ``PDMPath`` unwraps them before
    interpolating (``pdm_path.py:35``). Returns ``len(tangents) + 1`` values.
    """
    seg_headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    return np.unwrap(np.concatenate([seg_headings, seg_headings[-1:]]))


def _pose_on_path(
    s: float,
    path: np.ndarray,
    seg: np.ndarray,
    seg_lengths_safe: np.ndarray,
    cum_lengths: np.ndarray,
    vertex_headings: np.ndarray,
) -> Tuple[float, float, float]:
    """``(x, y, heading)`` at arc length ``s`` along ``path`` (clamped).

    ``PDMPath.interpolate``: linear in x, y and the UNWRAPPED vertex
    heading, then the heading is normalised. A piecewise-constant segment
    heading (the port's previous choice) turned the reference's curvature
    ramp into a staircase (±45 % ripple in the fitted curvature on a 1 m
    densified R = 30 m arc) and rotated the ideal lead-search polygons.
    """
    idx = int(np.searchsorted(cum_lengths, s, side="right") - 1)
    idx = max(0, min(idx, len(seg) - 1))
    frac = float(np.clip((s - float(cum_lengths[idx]))
                         / float(seg_lengths_safe[idx]), 0.0, 1.0))
    heading = float(vertex_headings[idx]
                    + frac * (vertex_headings[idx + 1] - vertex_headings[idx]))
    return (
        float(path[idx, 0] + frac * seg[idx, 0]),
        float(path[idx, 1] + frac * seg[idx, 1]),
        float(math.atan2(math.sin(heading), math.cos(heading))),
    )


def _path_substring(
    path: np.ndarray, cum_lengths: np.ndarray, start_s: float, end_s: float,
) -> np.ndarray:
    """``PDMPath.substring``: the path between two arc lengths.

    Reference fast path first — the vertices whose progress lies inside
    ``[start, end]`` (inclusive), returned as-is — and shapely's exact
    ``substring`` (interpolated end points) only when fewer than two
    vertices qualify. The ends are therefore NOT interpolated in the common
    case: the corridor reaches to the last vertex inside the interval, less
    than one vertex spacing short of the exact bound, exactly as upstream.
    """
    total = float(cum_lengths[-1])
    start_s = float(np.clip(start_s, 0.0, total))
    end_s = float(np.clip(end_s, 0.0, total))
    inside = (start_s <= cum_lengths) & (cum_lengths <= end_s)
    if int(np.count_nonzero(inside)) > 1:
        return path[inside]

    def _at(s: float) -> np.ndarray:
        idx = int(np.searchsorted(cum_lengths, s, side="right") - 1)
        idx = max(0, min(idx, len(path) - 2))
        seg_len = float(cum_lengths[idx + 1] - cum_lengths[idx])
        frac = (s - float(cum_lengths[idx])) / seg_len if seg_len > 1e-12 else 0.0
        return path[idx] + float(np.clip(frac, 0.0, 1.0)) * (path[idx + 1] - path[idx])

    if end_s <= start_s + 1e-9:
        # Degenerate interval (ego at or past the path end): a two-vertex
        # stub keeps LineString construction valid and the corridor a
        # zero-length square cap.
        idx = int(np.searchsorted(cum_lengths, start_s, side="right") - 1)
        idx = max(0, min(idx, len(path) - 2))
        return path[idx:idx + 2]
    strictly_inside = (cum_lengths > start_s + 1e-9) & (cum_lengths < end_s - 1e-9)
    return np.vstack([_at(start_s)[None, :], path[strictly_inside], _at(end_s)[None, :]])


#: The reference builds the lead-search corridor as far ahead as the fastest
#: IDM policy can drive over its OUTPUT trajectory horizon: 80 poses × 0.1 s.
_CORRIDOR_REACH_S = 8.0


def _corridor_reach_m(cfg: PDMConfig, speed_limit_mps: Optional[float]) -> float:
    """``|max_target_velocity| * trajectory horizon`` from ``_get_driving_corridor``."""
    v_max = max(
        abs(policy.target_velocity_for(speed_limit_mps))
        for policy in cfg.idm_policies
    )
    return float(v_max) * _CORRIDOR_REACH_S


# ----------------------------------------------------------------------
# Output containers
# ----------------------------------------------------------------------


@dataclass
class ProposalState:
    """Forward-simulated state along one proposal."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    accel: np.ndarray
    steering: np.ndarray
    sim_dt: float
    output_xy_ego: np.ndarray
    path_world_xy: np.ndarray
    # CoRL PDMScorer consumes the simulator's complete 11-channel state,
    # not kinematics reconstructed later from XY.  Channel order matches
    # tuplan_garage ``StateIndex``: x, y, heading, longitudinal/lateral
    # velocity, longitudinal/lateral acceleration, steering angle/rate,
    # angular velocity/acceleration.
    full_state: Optional[np.ndarray] = None


@dataclass
class AgentPrediction:
    """Constant-velocity prediction for a single non-ego agent."""

    track_id: str
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    valid: np.ndarray
    type: str
    length: float
    width: float


# ----------------------------------------------------------------------
# Forward simulation
# ----------------------------------------------------------------------


class SimulationCache:
    """Reuse pure simulations across adjacent banks of one intervention.

    Only raw plant outputs are cached: every bank still scores and verifies
    its own candidates. Keep the previous and current bank, so repeated
    refinement cannot grow this cache with the number of model rounds.
    Input values include the complete ego/controller state, configuration,
    actor forecasts and reference simulation. Mutable results are isolated.
    """

    def __init__(self) -> None:
        self._previous: Dict[bytes, ProposalState] = {}
        self._current: Dict[bytes, ProposalState] = {}
        self.hits = 0
        self.misses = 0

    @contextmanager
    def bank(self) -> Iterator[None]:
        token = _active_simulation_cache.set(self)
        polygons: OrderedDict[bytes, Polygon] = OrderedDict()
        polygon_token = _active_vehicle_polygon_cache.set(polygons)
        try:
            yield
        finally:
            try:
                self._previous = self._current
                self._current = {}
                _active_simulation_cache.reset(token)
            finally:
                polygons.clear()
                _active_vehicle_polygon_cache.reset(polygon_token)


_active_simulation_cache: ContextVar[Optional[SimulationCache]] = ContextVar(
    "refinement_simulation_cache", default=None)
_SimParams = ParamSpec("_SimParams")


def _reuse_simulation(
        simulate: Callable[_SimParams, ProposalState],
) -> Callable[_SimParams, ProposalState]:
    parameters = signature(simulate)

    @wraps(simulate)
    def wrapped(*args: _SimParams.args, **kwargs: _SimParams.kwargs) -> ProposalState:
        cache = _active_simulation_cache.get()
        if cache is None:
            return simulate(*args, **kwargs)
        bound = parameters.bind(*args, **kwargs)
        bound.apply_defaults()
        policy = bound.arguments["policy"]
        if type(policy) is IDMPolicy:
            # Names encode the candidate's position in a bank; the IDM
            # kernel never reads them. Preserve subclass semantics.
            bound.arguments["policy"] = replace(policy, name="")
        try:
            # Exact serialized values, not object identity or rounded
            # geometry. Unserializable extensions simply run uncached.
            key = pickle.dumps(bound.arguments, protocol=5)
        except (pickle.PickleError, TypeError, AttributeError):
            return simulate(*args, **kwargs)
        cached = cache._current.get(key)
        if cached is None:
            cached = cache._previous.get(key)
        if cached is not None:
            cache.hits += 1
            cache._current[key] = cached
            return deepcopy(cached)
        cache.misses += 1
        result = simulate(*args, **kwargs)
        cache._current[key] = deepcopy(result)
        return result

    return wrapped


@_reuse_simulation
def simulate_proposal(
    initial_x: float,
    initial_y: float,
    initial_heading: float,
    initial_speed: float,
    path_xy: np.ndarray,
    policy: IDMPolicy,
    cfg: PDMConfig,
    *,
    lead_dist: Optional[float] = None,
    lead_speed: float = 0.0,
    agents: Optional[Dict[str, AgentPrediction]] = None,
    speed_limit_mps: Optional[float] = None,
    longitudinal_accel_mps2: Optional[float] = None,
    ideal_kinematics: bool = False,
    reference_state: Optional[ProposalState] = None,
    initial_acceleration: float = 0.0,
    initial_steering_angle: float = 0.0,
    initial_angular_velocity: float = 0.0,
) -> ProposalState:
    """Forward-simulate one proposal along an offset path.

    Args:
        initial_x / initial_y / initial_heading / initial_speed: Ego
            state at ``t=0``.
        path_xy: ``(M, 2)`` polyline (world frame) the proposal
            tracks. Densified to ~1 m.
        policy: IDM policy that owns this proposal.
        cfg: Static PDM configuration.
        lead_dist / lead_speed: Initial lead-vehicle observation. The
            simulation also receives ``agents`` so it can refine the
            gap each step (faithful to ``tuplan_garage``); when
            ``agents`` is ``None`` the simulation falls back to a
            constant-lead approximation seeded by these values.
        agents: Constant-velocity predictions for non-ego agents. When
            provided, the per-step gap update reads each agent's
            position at the current sim step from ``pred.x[k]`` /
            ``pred.y[k]``.
        speed_limit_mps: Lane speed limit (m/s). Forwarded to the IDM
            kernel so the target velocity is
            ``policy.fraction * speed_limit_mps``.
        longitudinal_accel_mps2: When given, replaces the IDM
            longitudinal command with this constant acceleration
            (negative to brake), still clipped by the actuator caps and
            integrated by the same bicycle model. Lets a
            maximum-deceleration stop be expressed as an ORDINARY
            candidate that is forward-simulated like every other
            proposal, instead of an integrated position profile handed
            straight to execution. ``None`` (the default) leaves the IDM
            path bit-identical.

    Returns:
        :class:`ProposalState` with the simulated trajectory.
    """
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError(f"path_xy must have shape (M>=2, 2), got {path.shape}")
    if initial_speed < 0.0:
        raise ValueError(
            f"initial_speed must be non-negative, got {initial_speed}"
        )

    n_steps = cfg.num_sim_steps
    dt = cfg.sim_dt
    use_lqr = cfg.tracker == "lqr"

    # Bicycle dynamics. Use the actuator-filtered wrapper when the
    # planner config asks for it (faithful to upstream's
    # ``BatchKinematicBicycleModel``); otherwise fall back to the bare
    # :class:`EgoDynamics` which is bit-identical to the legacy path
    # this module shipped with. The filter is *only* engaged here, so
    # any other consumer of :class:`EgoDynamics` (env loop, training
    # envs, sensor adapters, ...) is completely unaffected.
    ego_cfg = EgoDynamicsCfg(
        dt=dt,
        max_speed=cfg.max_speed,
        max_steer_angle=cfg.max_steering_angle_rad,
        max_accel=cfg.max_accel,
        max_brake=cfg.max_brake,
        wheelbase=cfg.wheelbase,
        ego_length=cfg.ego_length,
        ego_width=cfg.ego_width,
    )
    ego: EgoDynamics | FilteredEgoDynamics
    if getattr(cfg, "use_actuator_filter", False) and not ideal_kinematics:
        ego = FilteredEgoDynamics(
            ego_cfg,
            params=ActuatorFilterParams(
                accel_time_constant_s=float(
                    getattr(cfg, "actuator_accel_time_constant_s", 0.0)
                ),
                steering_time_constant_s=float(
                    getattr(cfg, "actuator_steering_time_constant_s", 0.0)
                ),
            ),
        )
    else:
        ego = EgoDynamics(ego_cfg)
    if isinstance(ego, FilteredEgoDynamics):
        ego.reset(
            x=float(initial_x),
            y=float(initial_y),
            heading=float(initial_heading),
            speed=float(initial_speed),
            acceleration=float(initial_acceleration),
            steering_angle=float(initial_steering_angle),
        )
    else:
        ego.reset(
            x=float(initial_x),
            y=float(initial_y),
            heading=float(initial_heading),
            speed=float(initial_speed),
        )

    # The SCORED copy runs on the reference plant (rear-axle bicycle with the
    # reference's actuator lags and no floors — see ``reference_plant.py``),
    # seeded like ``ego_state_to_state_array``. The centre-referenced
    # ``EgoDynamics`` above is kept only as the pose mirror the lead loop
    # reads (``ego.x/y/heading/speed`` are synced from the plant each step)
    # and for the ideal / pure-pursuit / unfiltered paths.
    half_wb = 0.5 * float(cfg.wheelbase)
    plant: Optional[ReferenceBicyclePlant] = None
    if use_lqr and isinstance(ego, FilteredEgoDynamics):
        beta0 = math.atan(0.5 * math.tan(float(initial_steering_angle)))
        plant = ReferenceBicyclePlant(
            wheelbase=float(cfg.wheelbase),
            max_steering_angle=float(cfg.max_steering_angle_rad),
            accel_time_constant=float(
                getattr(cfg, "actuator_accel_time_constant_s", 0.2)),
            steering_angle_time_constant=float(
                getattr(cfg, "actuator_steering_time_constant_s", 0.05)),
        )
        plant.reset(
            rear_x=float(initial_x) - half_wb * math.cos(float(initial_heading)),
            rear_y=float(initial_y) - half_wb * math.sin(float(initial_heading)),
            heading=float(initial_heading),
            # NavSafe's ``speed`` is the centre speed; the rear axle of the
            # same bicycle moves at ``v·cos β``.
            velocity=float(initial_speed) * math.cos(beta0),
            acceleration=float(initial_acceleration),
            steering_angle=float(initial_steering_angle),
            angular_velocity=float(initial_angular_velocity),
        )

    xs = np.empty(n_steps + 1, dtype=np.float64)
    ys = np.empty(n_steps + 1, dtype=np.float64)
    headings = np.empty(n_steps + 1, dtype=np.float64)
    speeds = np.empty(n_steps + 1, dtype=np.float64)
    accels = np.zeros(n_steps + 1, dtype=np.float64)
    steerings = np.zeros(n_steps + 1, dtype=np.float64)
    steering_rates = np.zeros(n_steps + 1, dtype=np.float64)
    angular_velocities = np.zeros(n_steps + 1, dtype=np.float64)
    angular_accelerations = np.zeros(n_steps + 1, dtype=np.float64)

    xs[0] = ego.x
    ys[0] = ego.y
    headings[0] = ego.heading
    # Channel 3 is the rear-axle longitudinal velocity throughout
    # (``ego_state_to_state_array`` seeds ``rear_axle_velocity_2d.x``).
    speeds[0] = plant.velocity if plant is not None else ego.speed
    accels[0] = float(initial_acceleration)
    steerings[0] = float(initial_steering_angle)
    angular_velocities[0] = float(initial_angular_velocity)

    # Pre-compute path arc length and tangents for both LQR and
    # per-step gap projection.
    seg = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(seg, axis=1)
    seg_lengths_safe = np.where(seg_lengths > 1e-12, seg_lengths, 1.0)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    tangents = seg / seg_lengths_safe[:, None]
    vertex_headings = _vertex_headings(tangents)

    # Constant-lead initial values (used when ``agents`` is None — the
    # legacy back-compat path).
    initial_lead_gap = lead_dist
    initial_lead_speed = float(lead_speed) if lead_dist is not None else 0.0

    path_total_length = float(cum_lengths[-1])
    ideal_progress, _ = _project_point_on_path(
        float(initial_x), float(initial_y), path, seg, seg_lengths_safe,
        cum_lengths)

    if ideal_kinematics:
        # Reference ``PDMGenerator._initialize_states``: the ideal proposal's
        # pose 0 is the ego PROJECTED ONTO the path
        # (``path.interpolate([ego_progress])``), with the path heading — not
        # the raw ego pose. The simulated copy still starts from the real ego
        # state (``PDMSimulator.simulate_proposals`` seeds it with
        # ``ego_state_to_state_array``), so only this ideal trajectory moves.
        # It matters because the ideal poses are the LQR's reference: the
        # velocity/curvature profile is a least-squares fit of THESE poses,
        # and a raw pose 0 (off the path by the whole lateral offset) fed the
        # ego's own heading error into the first fitted curvature sample —
        # measured κ₀ = −0.031 on a straight road for a 0.05 rad error.
        x0, y0, h0 = _pose_on_path(
            ideal_progress, path, seg, seg_lengths_safe, cum_lengths,
            vertex_headings)
        assert isinstance(ego, EgoDynamics)
        ego.x, ego.y, ego.heading = x0, y0, h0
        xs[0], ys[0], headings[0] = x0, y0, h0

    # ------------------------------------------------------------------
    # Faithful per-step lead detection (CaRL-style)
    # ------------------------------------------------------------------
    # CaRL's ``PDMGenerator._update_leading_agents`` does three things:
    #   1. Builds a *driving corridor* polygon = the proposal path
    #      buffered by ``vehicle_width / 2``, capped square.
    #      (``_get_driving_corridor``)
    #   2. Picks agents intersecting that corridor as candidate leads.
    #      (``_get_intersecting_objects``)
    #   3. Computes the gap as
    #        ``ego_polygon.distance(agent_polygon)``
    #      (Shapely polygon-to-polygon Euclidean distance) — a true
    #      bumper-to-bumper measurement, *not* a centerline projection.
    #   4. Refreshes the leading-agent state only every
    #      ``leading_agent_update_rate`` sim steps (default 2).
    #   5. When no agent is ahead, falls back to an "end-of-path"
    #      virtual lead.  NavSafe places its equilibrium at the route
    #      endpoint because NavSafe judges the ego CENTER against that goal;
    #      CaRL's obstacle-style half-length offset made completion
    #      mathematically unreachable on short routes.
    #
    # All four match the per-step block below.
    corridor: Optional[Polygon] = None
    filtered_agents: Dict[str, AgentPrediction] = {}
    agent_polys: Dict[str, List[Optional[Polygon]]] = {}
    if agents and path.shape[0] >= 2:
        try:
            # ``cap_style=3`` is Shapely's SQUARE cap — matches CaRL's
            # ``CAP_STYLE.square``. The corridor is the path AHEAD of the
            # initial ego progress, as far as the fastest IDM policy could
            # travel over the reference's 8 s trajectory horizon
            # (``_get_driving_corridor``: ``path.substring(ego_distance,
            # ego_distance + max_target_velocity * 80 * 0.1)``) — not the
            # whole route.
            corridor = LineString(_path_substring(
                path, cum_lengths, ideal_progress,
                ideal_progress + _corridor_reach_m(cfg, speed_limit_mps),
            )).buffer(0.5 * float(cfg.ego_width), cap_style=3)
        except Exception:
            # Losing the corridor silently disables lead-vehicle detection
            # for this proposal: no real lead can be found, so IDM sees only
            # the end-of-path wall and the ego free-drives into traffic.
            # Never let that pass unnoticed.
            _logger.warning(
                "PDM forward-sim: driving-corridor construction failed for a "
                "%d-point path (ego_width=%.3f); lead-vehicle detection is "
                "disabled for this proposal.",
                path.shape[0],
                float(cfg.ego_width),
                exc_info=True,
            )
            corridor = None

        if corridor is not None:
            for tid, pred in agents.items():
                steps_to_use = min(pred.x.shape[0], n_steps + 1)
                polys: List[Optional[Polygon]] = [None] * steps_to_use
                intersects_any = False
                for k_pre in range(steps_to_use):
                    if not bool(pred.valid[k_pre]):
                        continue
                    poly = _vehicle_polygon(
                        float(pred.x[k_pre]),
                        float(pred.y[k_pre]),
                        float(pred.heading[k_pre]),
                        float(pred.length),
                        float(pred.width),
                    )
                    polys[k_pre] = poly
                    if not intersects_any and corridor.intersects(poly):
                        intersects_any = True
                if intersects_any:
                    filtered_agents[tid] = pred
                    agent_polys[tid] = polys

    # Cached leading-agent state — refreshed every
    # ``cfg.leading_agent_update_rate`` sim steps to match CaRL's
    # ``leading_agent_update_rate=2`` cadence.
    # CaRL's proposal array is zero-initialized. Its generator starts at
    # ``time_idx=1`` and, with the default update rate of two, copies the
    # zero leading-agent row before the first real refresh at time_idx=2.
    # IDM therefore commands its maximum deceleration on every proposal's
    # first step. Mapping our zero-based loop k to time_idx=k+1 reproduces
    # that otherwise easy-to-miss transient.
    #
    # What is HELD between refreshes is the lead's absolute path progress
    # (``LeadingAgentIndex.PROGRESS = ego_progress + distance`` at the
    # refresh), not the bumper gap: ``BatchIDMPolicy.propagate`` recomputes
    # ``x_lead - x_agent`` every step, so on the un-refreshed odd step the
    # gap has shrunk by the ego's own advance. Holding the gap instead
    # under-braked on half of all IDM steps (measured 0.24 m / 0.10 m/s over
    # one 4 s unroll behind a 3 m/s lead from 10 m/s). The zero row is the
    # progress ``0.0``: ``s_alpha = max(0 - x_ego, s0) = s0``, as upstream.
    cached_lead_progress: Optional[float] = 0.0 if agents is not None else None
    cached_lead_speed: float = 0.0 if agents is not None else initial_lead_speed
    update_rate = max(1, int(getattr(cfg, "leading_agent_update_rate", 1)))
    # Faithful lead handling (per-step corridor leads + end-of-path
    # wall) applies whenever the caller supplied predictions — an
    # *empty* dict means "no agents around", not "skip the wall".
    # ``None`` preserves the legacy seed-verbatim path for old tests.
    use_faithful_lead = agents is not None

    # LQR state: maintains a steering-angle integrator across the
    # tracking horizon so the LQR controller can produce smooth
    # steering rates.
    cur_steering_angle = float(initial_steering_angle)  # physical units (rad)
    reference_velocity_profile: Optional[np.ndarray] = None
    reference_curvature_profile: Optional[np.ndarray] = None
    if reference_state is not None:
        reference_poses = np.column_stack(
            [reference_state.x, reference_state.y, reference_state.heading]
        )[None, ...]
        if plant is not None:
            # The reference tracks REAR-AXLE poses (its ideal states are the
            # rear axle on the baseline); the ideal trajectory here is the
            # centre on the path, so shift it back half a wheelbase.
            reference_poses = reference_poses.copy()
            reference_poses[0, :, 0] -= half_wb * np.cos(reference_poses[0, :, 2])
            reference_poses[0, :, 1] -= half_wb * np.sin(reference_poses[0, :, 2])
        fitted_velocity, fitted_curvature = (
            velocity_curvature_profiles_from_poses(
                reference_poses,
                discretization_time=cfg.sim_dt,
                jerk_penalty=cfg.lqr_jerk_penalty,
                curvature_rate_penalty=cfg.lqr_curvature_rate_penalty,
            )
        )
        reference_velocity_profile = fitted_velocity[0]
        reference_curvature_profile = fitted_curvature[0]

    for k in range(n_steps):
        # --- 1. Lead gap (CaRL-faithful per-step refinement) ---
        if use_faithful_lead and ((k + 1) % update_rate) == 0:
            ego_poly = _vehicle_polygon(
                ego.x, ego.y, ego.heading,
                cfg.ego_length, cfg.ego_width,
            )
            ego_s, _ = _project_point_on_path(
                ego.x, ego.y, path, seg, seg_lengths_safe, cum_lengths
            )
            # The reference's "agent ahead" test compares the agent centroid's
            # path progress against the ego's REAR-AXLE progress
            # (``_initialize_states`` projects ``ego_state.rear_axle``;
            # ``_update_leading_agents``: ``progress > current_ego_progress``).
            # NavSafe's pose is the wheelbase centre, so the rear axle sits
            # half a wheelbase behind it. The gap itself is polygon-to-polygon
            # and does not depend on which point defines progress; only this
            # comparison does. The end-of-path wall below keeps the centre
            # projection (NavSafe's goal is judged against the centre).
            half_wb = 0.5 * float(cfg.wheelbase)
            ego_s_rear, _ = _project_point_on_path(
                ego.x - half_wb * math.cos(ego.heading),
                ego.y - half_wb * math.sin(ego.heading),
                path, seg, seg_lengths_safe, cum_lengths
            )

            cur_gap: Optional[float] = None
            cur_lead_speed: float = 0.0

            # 1a. Real agents in the driving corridor AT THIS TIMESTEP.
            # ``filtered_agents`` is only a cheap superset prefilter
            # (any-step intersection); canonical PDM-Closed queries the
            # forecasted occupancy per step (tuplan_garage / CaRL
            # ``_get_intersecting_objects``: ``self._observation[time_idx]
            # .intersects(driving_corridor)``), so membership must be
            # re-checked at step k — otherwise a crossing/oncoming agent
            # whose sweep clips the corridor seconds from now reads as an
            # immediate lead and IDM phantom-brakes from step 0.
            # The reference samples agents at the DESTINATION step:
            # ``pdm_generator`` reads ``observation[time_idx]`` while ego
            # progress comes from ``time_idx - 1``. Our loop ``k`` maps to
            # ``time_idx = k + 1``, so the agent index is ``k + 1``
            # (clamped to the forecast tail, matching held-final-state
            # predictions); reading ``k`` left every lead refresh one sim
            # step (0.1 s) stale relative to the reference.
            for tid, pred in filtered_agents.items():
                polys = agent_polys[tid]
                if not polys:
                    continue
                ai = min(k + 1, len(polys) - 1)
                agent_poly = polys[ai]
                if agent_poly is None:
                    continue
                if corridor is not None and not corridor.intersects(agent_poly):
                    continue
                ai_p = min(k + 1, len(pred.x) - 1)
                ax = float(pred.x[ai_p])
                ay = float(pred.y[ai_p])
                s_a, _ = _project_point_on_path(
                    ax, ay, path, seg, seg_lengths_safe, cum_lengths
                )
                if s_a <= ego_s_rear:
                    # Agent behind the ego's rear axle along the path — not
                    # a lead (reference semantics, see ``ego_s_rear``).
                    continue
                # Bumper-to-bumper Euclidean gap via Shapely
                # polygon-to-polygon distance. Matches CaRL's
                # ``ego_polygon.distance(agent_polygon)`` exactly
                # (subject to the centroid-vs-rear-axle convention
                # noted in ``_vehicle_polygon``).
                bumper_gap = float(ego_poly.distance(agent_poly))
                if cur_gap is None or bumper_gap < cur_gap:
                    cur_gap = bumper_gap
                    # Lead speed projected onto ego heading — matches
                    # CaRL's ``_get_leading_agent_velocity``
                    # (``|v| * cos(relative_heading)``).
                    v_mag = math.hypot(
                        float(pred.vx[ai_p]), float(pred.vy[ai_p])
                    )
                    rel_heading = (
                        float(pred.heading[ai_p]) - float(ego.heading))
                    rel_heading = (
                        (rel_heading + math.pi) % (2.0 * math.pi) - math.pi
                    )
                    # Signed: negative for oncoming traffic, so IDM's
                    # closing-rate term ``v * (v - v_lead)`` brakes
                    # roughly twice as hard for a head-on approach.
                    # Clamping this at 0 (treating oncoming as parked)
                    # systematically under-brakes.
                    cur_lead_speed = v_mag * math.cos(rel_heading)

            # 1b. End-of-path virtual lead — ONLY in the free-driving
            # branch, exactly like upstream (tuplan_garage / CaRL
            # ``_update_leading_agents``: the ``else`` of "agents
            # ahead"). When a real lead exists it governs IDM alone;
            # letting the wall compete via min() made the ego brake for
            # the route end THROUGH a real moving lead, which on the
            # truncated routes py123d data produces (no exit_lanes →
            # single-lane fallback) suppressed progress in every scene.
            # CaRL models the path end as a physical obstacle and subtracts
            # half the ego length. NavSafe models it as a center-position
            # GOAL. Combining CaRL's half-length offset with IDM's 1 m
            # standstill gap stopped the ego about 3.4 m before the goal while
            # the shared live/post-hoc goal predicate permits only 2.0 m of
            # remaining arc. Measured on 20cc, the prior run settled 3.22 m
            # short and could never latch goal_reached.
            #
            # Put IDM equilibrium (gap == policy.min_gap at v == 0) exactly at
            # path_total_length instead. It still brakes a fast ego before a
            # short path ends, but no longer converts a target plane into an
            # unreachable bumper wall.
            if cur_gap is None and path_total_length > 0.0:
                cur_gap = max(
                    0.0,
                    path_total_length - ego_s + float(policy.min_gap),
                )
                cur_lead_speed = 0.0  # stationary wall

            # Reference ``relative_distance = current_ego_progress + dist``:
            # the ego progress of the state the IDM propagates FROM.
            ego_progress_now = (
                ideal_progress if ideal_kinematics
                else _project_point_on_path(
                    ego.x, ego.y, path, seg, seg_lengths_safe, cum_lengths)[0])
            cached_lead_progress = (
                None if cur_gap is None else ego_progress_now + cur_gap)
            cached_lead_speed = cur_lead_speed

        if use_faithful_lead:
            if cached_lead_progress is None:
                gap_for_idm: Optional[float] = None
            else:
                ego_progress_now = (
                    ideal_progress if ideal_kinematics
                    else _project_point_on_path(
                        ego.x, ego.y, path, seg, seg_lengths_safe,
                        cum_lengths)[0])
                gap_for_idm = cached_lead_progress - ego_progress_now
            lead_speed_for_idm = cached_lead_speed
        else:
            # Legacy back-compat: no agents passed at all → use the
            # caller-supplied seed verbatim (matches pre-faithful
            # behaviour exactly for tests that call simulate_proposal
            # without agents).
            gap_for_idm = initial_lead_gap
            lead_speed_for_idm = initial_lead_speed

        # --- 2. Longitudinal control (IDM, or a commanded override) ---
        lqr_reference_velocity: Optional[float] = None
        lqr_curvature_slice: Optional[np.ndarray] = None
        if reference_state is not None:
            # Track the fitted ideal-proposal profile with CaRL's actual
            # one-step longitudinal LQR. The old deadbeat follower divided
            # the next stored speed error by dt and ignored every configured
            # longitudinal LQR/stopping/profile-fit parameter.
            assert reference_velocity_profile is not None
            assert reference_curvature_profile is not None
            lqr_reference_velocity, _ = reference_slice(
                reference_velocity_profile, k, cfg.lqr_tracking_horizon
            )
            _, lqr_curvature_slice = reference_slice(
                reference_curvature_profile, k, cfg.lqr_tracking_horizon
            )
            a_idm = longitudinal_lqr_acceleration(
                plant.velocity if plant is not None else ego.speed,
                lqr_reference_velocity,
                discretization_time=cfg.sim_dt,
                tracking_horizon=cfg.lqr_tracking_horizon,
                q_longitudinal=cfg.lqr_q_longitudinal,
                r_longitudinal=cfg.lqr_r_longitudinal,
                stopping_velocity=cfg.lqr_stopping_velocity,
                stopping_proportional_gain=cfg.lqr_stopping_proportional_gain,
            )
        elif longitudinal_accel_mps2 is None:
            a_idm = idm_accel(
                speed=max(0.0, ego.speed),
                lead_dist=gap_for_idm,
                lead_speed=lead_speed_for_idm,
                policy=policy,
                speed_limit_mps=speed_limit_mps,
            )
        else:
            # A commanded longitudinal profile — used to express a
            # maximum-deceleration stop as an ORDINARY forward-simulated
            # candidate. It goes through the same normalisation, the same
            # actuator caps and the same bicycle model as every IDM
            # proposal, so the resulting motion is what the vehicle can
            # actually do rather than an integrated profile handed
            # straight to execution. Once stopped, command zero: holding
            # a negative accel at rest is what a physical brake does, but
            # the dynamics model would otherwise keep integrating it.
            a_idm = (float(longitudinal_accel_mps2)
                     if ego.speed > 0.0 else 0.0)
        # Cap by the ego's physical limits.
        if a_idm >= 0.0:
            accel_norm = float(np.clip(a_idm / cfg.max_accel, -1.0, 1.0))
        else:
            accel_norm = float(np.clip(a_idm / cfg.max_brake, -1.0, 1.0))

        # Ideal proposal generation is an IDM progress unroll constrained
        # exactly to the path. Upstream scores a separately simulated copy,
        # then returns this ideal representation.
        if ideal_kinematics:
            # Ideal generation never constructs the filtered wrapper; it is
            # the kinematic reference later tracked by the filtered scoring
            # simulation. Keep that invariant explicit for both runtime and
            # static checking before mutating the reference state in place.
            assert isinstance(ego, EgoDynamics)
            ideal_progress = min(path_total_length,
                                 ideal_progress + ego.speed * dt)
            old_speed = ego.speed
            # The IDEAL unroll keeps a speed floor the reference lacks
            # (``BatchIDMPolicy.propagate`` integrates ``v + dt * v_dot``
            # unfloored). With a stopped lead the IDM equilibrium is v = 0
            # from above, so the floor only binds for an ONCOMING lead
            # (negative ``v_lead``), where the reference dips a few mm/s
            # below zero and recovers. The SCORED copy runs on the reference
            # plant with no floor at all (``reference_plant.py``); this
            # unroll is what the executor consumes and must not reverse.
            ego.speed = float(np.clip(old_speed + a_idm * dt,
                                      0.0, cfg.max_speed))
            ego.x, ego.y, ego.heading = _pose_on_path(
                ideal_progress, path, seg, seg_lengths_safe, cum_lengths,
                vertex_headings)
            xs[k + 1], ys[k + 1] = ego.x, ego.y
            headings[k + 1], speeds[k + 1] = ego.heading, ego.speed
            accels[k + 1] = a_idm
            steerings[k + 1] = 0.0
            angular_velocities[k + 1] = (
                (headings[k + 1] - headings[k] + math.pi)
                % (2.0 * math.pi) - math.pi
            ) / dt
            angular_accelerations[k + 1] = (
                angular_velocities[k + 1] - angular_velocities[k]
            ) / dt
            continue

        # --- 3. Lateral control ---
        if use_lqr:
            # Faithful to upstream BatchLQRTracker: feed a per-step
            # velocity profile into the lateral linearization. We
            # predict ``cfg.lqr_tracking_horizon`` steps of IDM speed
            # forward from the current ego state, holding the lead's
            # gap and speed at the current values for the prediction
            # window. Matches the spirit of upstream's
            # ``_compute_reference_velocity_and_curvature_profile``,
            # adapted to NavSafe's IDM-driven longitudinal control.
            if reference_state is not None:
                assert lqr_reference_velocity is not None
                assert lqr_curvature_slice is not None
                should_stop = bool(
                    (plant.velocity if plant is not None else ego.speed)
                    <= cfg.lqr_stopping_velocity
                    and lqr_reference_velocity <= cfg.lqr_stopping_velocity
                )
                # CaRL linearizes lateral dynamics around the velocity
                # profile produced by holding this longitudinal LQR command
                # constant for the lookahead horizon (including the current
                # velocity as the first element).
                velocity_profile = generate_profile(
                    np.array([plant.velocity if plant is not None else ego.speed],
                             dtype=np.float64),
                    np.full(
                        (1, cfg.lqr_tracking_horizon),
                        a_idm,
                        dtype=np.float64,
                    ),
                    cfg.sim_dt,
                )[0, : cfg.lqr_tracking_horizon]
            else:
                should_stop = False
                velocity_profile = _predict_idm_velocity_profile(
                    initial_speed=max(0.0, ego.speed),
                    lead_dist=gap_for_idm,
                    lead_speed=lead_speed_for_idm,
                    policy=policy,
                    num_steps=cfg.lqr_tracking_horizon,
                    dt=cfg.sim_dt,
                    max_speed=cfg.max_speed,
                    max_accel=cfg.max_accel,
                    max_brake=cfg.max_brake,
                    speed_limit_mps=speed_limit_mps,
                )
            if should_stop:
                # Reference stopping controller emits zero steering rate,
                # which holds (rather than zeros) the current wheel angle.
                steering_rate_cmd = 0.0
                steer_norm = float(np.clip(
                    -cur_steering_angle / cfg.max_steering_angle_rad,
                    -1.0,
                    1.0,
                ))
            else:
                if plant is not None:
                    # Reference measurement: rear-axle position, body
                    # heading, the plant's own (clipped, filtered) angle.
                    lqr_x, lqr_y = plant.rear_x, plant.rear_y
                    lqr_speed = plant.velocity
                    lqr_heading_mode = "body"
                    lqr_ref_pose = (
                        np.array([
                            reference_poses[0, k, 0],
                            reference_poses[0, k, 1],
                            reference_poses[0, k, 2],
                        ], dtype=np.float64)
                        if reference_state is not None else None)
                else:
                    lqr_x, lqr_y = ego.x, ego.y
                    lqr_speed = ego.speed
                    lqr_heading_mode = "course"
                    lqr_ref_pose = (
                        np.array([
                            reference_state.x[k],
                            reference_state.y[k],
                            reference_state.heading[k],
                        ], dtype=np.float64)
                        if reference_state is not None else None)
                steer_norm, new_steering_angle = _lqr_steer(
                    ego_x=lqr_x,
                    ego_y=lqr_y,
                    ego_heading=ego.heading,
                    ego_speed=lqr_speed,
                    ego_steering_angle=cur_steering_angle,
                    path=path,
                    seg=seg,
                    seg_lengths=seg_lengths,
                    seg_lengths_safe=seg_lengths_safe,
                    cum_lengths=cum_lengths,
                    tangents=tangents,
                    cfg=cfg,
                    velocity_profile=velocity_profile,
                    curvature_profile=lqr_curvature_slice,
                    reference_pose=lqr_ref_pose,
                    # The reference tracker emits an unclipped steering
                    # rate; the PLANT clips the angle after its low-pass
                    # filter (``propagate_state``). The filtered plant below
                    # does the same, so the command must reach it raw.
                    clip_command=not isinstance(ego, FilteredEgoDynamics),
                    heading_mode=lqr_heading_mode,
                )
                # ``BatchLQRTracker`` hands the plant a steering RATE; the
                # integrated angle is only its own bookkeeping.
                steering_rate_cmd = (new_steering_angle - cur_steering_angle) / dt
                cur_steering_angle = new_steering_angle
        else:
            steer_norm = _pure_pursuit_steer_norm(
                ego_x=ego.x,
                ego_y=ego.y,
                ego_heading=ego.heading,
                path=path,
                cum_lengths=cum_lengths,
                lookahead_m=max(
                    cfg.controller_lookahead_min_m,
                    cfg.controller_lookahead_gain_s * ego.speed,
                ),
                wheelbase=cfg.wheelbase,
                max_steer_angle=cfg.max_steering_angle_rad,
            )

        if plant is not None:
            # Reference step: raw acceleration and steering-rate commands,
            # actuator lags inside the plant. Channels
            # are the reference ``StateIndex`` array; the recorded pose is
            # converted back to NavSafe's wheelbase-centre convention.
            state = plant.step(acceleration_cmd=a_idm,
                               steering_rate_cmd=steering_rate_cmd, dt=dt)
            if state[3] < 0.0:
                # This simulator accepts forward-only proposals. Braking,
                # including residual actuator lag at rest, must end at zero
                # speed as it does in live execution. Keep the filtered
                # acceleration so brake release retains its original lag.
                state[3] = 0.0
                state[9] = 0.0
                state[10] = -angular_velocities[k] / dt
                plant.state = state
            cx, cy = plant.centre()
            assert isinstance(ego, FilteredEgoDynamics)
            ego.mirror_pose(x=cx, y=cy, heading=plant.heading,
                            speed=plant.velocity)
            xs[k + 1], ys[k + 1] = cx, cy
            headings[k + 1] = plant.heading
            speeds[k + 1] = state[3]
            accels[k + 1] = state[5]
            steerings[k + 1] = state[7]
            steering_rates[k + 1] = state[8]
            angular_velocities[k + 1] = state[9]
            angular_accelerations[k + 1] = state[10]
            # ``BatchLQRTracker`` reads the plant's clipped, filtered angle.
            cur_steering_angle = float(state[7])
            continue

        previous_steering = (
            ego.filtered_steering_phys
            if isinstance(ego, FilteredEgoDynamics)
            else steerings[k]
        )
        ego.step(steer=steer_norm, accel=accel_norm)

        xs[k + 1] = ego.x
        ys[k + 1] = ego.y
        headings[k + 1] = ego.heading
        speeds[k + 1] = ego.speed
        if isinstance(ego, FilteredEgoDynamics):
            # BatchKinematicBicycle stores the actuator-filtered state that
            # actually propagated the vehicle.  Recording the requested
            # command here made CoRL comfort score a different trajectory.
            steerings[k + 1] = ego.filtered_steering_phys
            accels[k + 1] = ego.filtered_accel_phys
        else:
            steerings[k + 1] = -steer_norm * cfg.max_steering_angle_rad
            if accel_norm >= 0.0:
                accels[k + 1] = accel_norm * cfg.max_accel
            else:
                accels[k + 1] = accel_norm * cfg.max_brake
        if use_lqr:
            # CaRL's next BatchLQR call reads STEERING_ANGLE from the state
            # produced by BatchKinematicBicycleModel.  That state contains the
            # actuator-filtered angle which actually propagated the vehicle,
            # not the controller's requested/integrated angle.  Keeping the
            # request here desynchronised controller and plant on every step.
            cur_steering_angle = float(steerings[k + 1])
        steering_rates[k + 1] = (
            steerings[k + 1] - previous_steering
        ) / dt
        angular_velocities[k + 1] = (
            (headings[k + 1] - headings[k] + math.pi)
            % (2.0 * math.pi) - math.pi
        ) / dt
        angular_accelerations[k + 1] = (
            angular_velocities[k + 1] - angular_velocities[k]
        ) / dt

    # Build the output sample at the configured stride. Output is
    # in *ego* frame [lateral, forward] per the NavSafe contract.
    stride = cfg.output_stride
    n_out = cfg.num_output_poses
    out_indices = np.arange(1, n_out + 1) * stride
    out_indices = np.minimum(out_indices, n_steps)
    sampled_xy_world = np.column_stack([xs[out_indices], ys[out_indices]])
    output_xy_ego = _world_to_ego_lateral_forward(
        sampled_xy_world,
        initial_x=initial_x,
        initial_y=initial_y,
        initial_heading=initial_heading,
    )

    full_state = np.zeros((n_steps + 1, 11), dtype=np.float64)
    full_state[:, 0] = xs
    full_state[:, 1] = ys
    full_state[:, 2] = headings
    # Upstream's StateIndex velocities/accelerations are vehicle-frame
    # longitudinal/lateral channels, despite the x/y names.
    full_state[:, 3] = speeds
    full_state[:, 5] = accels
    full_state[:, 7] = steerings
    full_state[:, 8] = steering_rates
    full_state[:, 9] = angular_velocities
    full_state[:, 10] = angular_accelerations

    return ProposalState(
        x=xs,
        y=ys,
        heading=headings,
        speed=speeds,
        accel=accels,
        steering=steerings,
        sim_dt=dt,
        output_xy_ego=output_xy_ego,
        path_world_xy=np.column_stack([xs, ys]),
        full_state=full_state,
    )


# ----------------------------------------------------------------------
# Constant-velocity agent prediction
# ----------------------------------------------------------------------


def _track_dims(
    track: Dict[str, Any], frame_id: int, cfg: PDMConfig
) -> tuple[float, float]:
    """Track dims via the shared resolver (see ``navsafe.core.track_dims``)."""
    return track_dims(
        track, frame_id, fallback=(cfg.ego_length, cfg.ego_width)
    )


def predict_agents_constant_velocity(
    scenario_data: Dict[str, Any],
    frame_id: int,
    cfg: PDMConfig,
    *,
    include_types: tuple[str, ...] = COLLIDABLE_ACTOR_TYPES_TUPLE,
) -> Dict[str, AgentPrediction]:
    """Roll non-ego agents forward at constant velocity over the planning horizon."""
    metadata = scenario_data.get("metadata", {})
    sdc_id = metadata.get("sdc_id")
    tracks = scenario_data.get("tracks", {})

    n_steps = int(round(cfg.agent_prediction_horizon_s / cfg.sim_dt))
    dt = cfg.sim_dt

    # Consecutive LOG frames are ``scenario_dt`` apart, which need not equal
    # the sim step. Only the finite-difference velocity fallback below uses
    # it; the forward extrapolation stays on the sim-step grid consumers
    # index by. Mirrors ``predict_agents_log_replay``.
    scenario_dt = scenario_dt_seconds(metadata, default=cfg.sim_dt)

    out: Dict[str, AgentPrediction] = {}

    for track_id, track in tracks.items():
        if track_id == sdc_id:
            continue
        track_type = track.get("type", "")
        if track_type not in include_types:
            continue

        state = track.get("state", {})
        positions = state.get("position")
        if positions is None:
            continue
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim < 2 or positions.shape[0] == 0:
            continue
        sample_frame = min(int(frame_id), positions.shape[0] - 1)
        past_log = int(frame_id) >= positions.shape[0]

        valid_arr = state.get("valid")
        if valid_arr is not None:
            valid_arr = np.asarray(valid_arr, dtype=bool)
        else:
            valid_arr = np.ones(positions.shape[0], dtype=bool)
        if not bool(valid_arr[sample_frame]):
            continue

        init_xy = positions[sample_frame, :2].astype(np.float64)
        velocities = state.get("velocity")
        if past_log:
            # The replay environment holds actors at their last logged pose.
            init_vxvy = np.zeros(2, dtype=np.float64)
        elif velocities is not None and len(velocities) > sample_frame:
            init_vxvy = np.asarray(
                velocities[sample_frame][:2], dtype=np.float64)
        else:
            if (sample_frame + 1 < positions.shape[0]
                    and bool(valid_arr[sample_frame + 1])):
                next_xy = positions[sample_frame + 1, :2]
                init_vxvy = (next_xy - init_xy) / max(scenario_dt, 1e-6)
            else:
                init_vxvy = np.zeros(2, dtype=np.float64)

        headings_arr = state.get("heading")
        if headings_arr is not None and len(headings_arr) > sample_frame:
            init_heading = float(headings_arr[sample_frame])
        else:
            init_heading = float(np.arctan2(init_vxvy[1], init_vxvy[0]))
        # Reference forecast: |v| along the box heading for agents, frozen
        # for static object types (see ``core.agent_forecast``).
        init_vxvy = np.asarray(forecast_velocity(
            float(init_vxvy[0]), float(init_vxvy[1]), init_heading, track_type),
            dtype=np.float64)

        xs = np.empty(n_steps + 1, dtype=np.float64)
        ys = np.empty(n_steps + 1, dtype=np.float64)
        vxs = np.empty(n_steps + 1, dtype=np.float64)
        vys = np.empty(n_steps + 1, dtype=np.float64)
        headings = np.empty(n_steps + 1, dtype=np.float64)
        valid = np.zeros(n_steps + 1, dtype=bool)

        xs[0] = init_xy[0]
        ys[0] = init_xy[1]
        vxs[0] = init_vxvy[0]
        vys[0] = init_vxvy[1]
        headings[0] = init_heading
        valid[0] = True

        for k in range(1, n_steps + 1):
            xs[k] = xs[0] + init_vxvy[0] * (k * dt)
            ys[k] = ys[0] + init_vxvy[1] * (k * dt)
            vxs[k] = init_vxvy[0]
            vys[k] = init_vxvy[1]
            headings[k] = init_heading
            valid[k] = True

        length, width = _track_dims(track, sample_frame, cfg)

        out[str(track_id)] = AgentPrediction(
            track_id=str(track_id),
            x=xs,
            y=ys,
            heading=headings,
            vx=vxs,
            vy=vys,
            valid=valid,
            type=track_type,
            length=length,
            width=width,
        )

    return out


def predict_agents_from_live_states(
    agent_states: List[Dict[str, Any]],
    cfg: PDMConfig,
    *,
    include_types: tuple[str, ...] = COLLIDABLE_ACTOR_TYPES_TUPLE,
) -> Dict[str, AgentPrediction]:
    """Forecast the environment's current actor snapshot at constant velocity.

    Semi-reactive traffic can leave its logged trajectory and publishes the
    resulting pose through ``env.agent_states``.  Re-reading ``scenario_data``
    at that point forecasts a different world from the one the evaluator is
    scoring (and, past the bundle, can create a permanently parked phantom).
    This adapter makes the live environment snapshot the authoritative t=0
    state while retaining the planner's ordinary constant-velocity model.
    """
    n_steps = int(round(cfg.agent_prediction_horizon_s / cfg.sim_dt))
    times = np.arange(n_steps + 1, dtype=np.float64) * float(cfg.sim_dt)
    out: Dict[str, AgentPrediction] = {}
    seen_ids: Dict[str, int] = {}

    for index, state in enumerate(agent_states):
        if state.get("is_ego", False) or not state.get("valid", True):
            continue
        actor_type = str(state.get("type", "VEHICLE")).upper()
        if actor_type not in include_types:
            continue
        position = np.asarray(state.get("position", []), dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.isfinite(position[:2]).all():
            continue
        heading = float(state.get("heading", 0.0))
        velocity = np.asarray(state.get("velocity", []), dtype=np.float64).reshape(-1)
        if velocity.size >= 2:
            vx, vy = float(velocity[0]), float(velocity[1])
        elif velocity.size == 1:
            speed = float(velocity[0])
            vx, vy = speed * math.cos(heading), speed * math.sin(heading)
        else:
            vx = vy = 0.0
        if not np.isfinite([heading, vx, vy]).all():
            continue
        # Reference forecast: |v| along the box heading for agents, frozen
        # for static object types (``core.agent_forecast``).
        vx, vy = forecast_velocity(vx, vy, heading, actor_type)

        raw_id = str(state.get("id", index))
        # Duplicate/missing ids must not silently erase an actor from scoring;
        # same ``#<count>`` suffix rule as the scorer's ``_live_agents_per_t``
        # so the collided-track latch keys match.
        duplicate = seen_ids.get(raw_id, 0)
        seen_ids[raw_id] = duplicate + 1
        track_id = raw_id if duplicate == 0 else f"{raw_id}#{duplicate}"
        xs = position[0] + vx * times
        ys = position[1] + vy * times
        out[track_id] = AgentPrediction(
            track_id=track_id,
            x=xs,
            y=ys,
            heading=np.full(n_steps + 1, heading, dtype=np.float64),
            vx=np.full(n_steps + 1, vx, dtype=np.float64),
            vy=np.full(n_steps + 1, vy, dtype=np.float64),
            valid=np.ones(n_steps + 1, dtype=bool),
            type=actor_type,
            length=float(state.get("length", cfg.ego_length)),
            width=float(state.get("width", cfg.ego_width)),
        )

    return out


def predict_agents_log_replay(
    scenario_data: Dict[str, Any],
    frame_id: int,
    cfg: PDMConfig,
    *,
    include_types: tuple[str, ...] = COLLIDABLE_ACTOR_TYPES_TUPLE,
) -> Dict[str, AgentPrediction]:
    """Non-ego agent futures read straight from the scenario log.

    NavSafe environments replay logged agents, so the log *is* the
    future — the same agent model the EPDMS scorer uses. Use these for
    the emergency-brake TTC check so the brake and the scorer agree:
    constant-velocity extrapolation sweeps a turning agent's box
    across the ego's route and fires the brake for conflicts that
    never happen (upstream sources the brake's TTC from the scoring
    metrics for exactly this reason).

    Actors that become invalid within the log disappear. Once evaluation
    outlives the whole bundle, actors valid at the final frame are held there,
    matching the replay environment.
    """
    metadata = scenario_data.get("metadata", {})
    sdc_id = metadata.get("sdc_id")
    tracks = scenario_data.get("tracks", {})

    n_steps = int(round(cfg.agent_prediction_horizon_s / cfg.sim_dt))

    scenario_dt = scenario_dt_seconds(metadata, default=cfg.sim_dt)

    out: Dict[str, AgentPrediction] = {}

    for track_id, track in tracks.items():
        if track_id == sdc_id:
            continue
        track_type = track.get("type", "")
        if track_type not in include_types:
            continue

        state = track.get("state", {})
        positions = state.get("position")
        if positions is None:
            continue
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim < 2 or positions.shape[0] == 0:
            continue

        valid_arr = state.get("valid")
        if valid_arr is not None:
            valid_arr = np.asarray(valid_arr, dtype=bool)
        else:
            valid_arr = np.ones(positions.shape[0], dtype=bool)
        sample_frame = min(int(frame_id), positions.shape[0] - 1)
        past_log = int(frame_id) >= positions.shape[0]
        if not bool(valid_arr[sample_frame]):
            continue

        headings_arr = state.get("heading")
        velocities = state.get("velocity")

        n = n_steps + 1
        xs = np.zeros(n, dtype=np.float64)
        ys = np.zeros(n, dtype=np.float64)
        vxs = np.zeros(n, dtype=np.float64)
        vys = np.zeros(n, dtype=np.float64)
        headings = np.zeros(n, dtype=np.float64)
        valid = np.zeros(n, dtype=bool)

        # Consumers read prediction index k as sim time ``k * cfg.sim_dt``
        # (per-step gap update, TTC sweeps), while the log is sampled at the
        # SCENARIO dt — the two are equal on py123d (0.1 s) but nothing
        # guarantees it, and this repo added ``scenario_dt_seconds`` exactly
        # because producers disagree. Resample the log onto the sim-step
        # grid instead of assuming the strides match (with matching dts the
        # index array below is the old contiguous slice, bit for bit).
        stride = cfg.sim_dt / scenario_dt if scenario_dt > 0.0 else 1.0
        idxs = sample_frame + np.round(np.arange(n) * stride).astype(np.int64)
        n_frames = positions.shape[0]
        # Once evaluation outlives the bundle, replay actors hold their final
        # logged pose. Forecast the same occupied world instead of declaring
        # it empty; within the log, actors that become invalid still vanish.
        in_range = np.ones(n, dtype=bool) if past_log else idxs < n_frames
        safe = np.clip(idxs, 0, n_frames - 1)
        varr = np.asarray(valid_arr, dtype=bool)
        vmask = in_range & varr[safe]
        valid[:] = vmask
        xs[:] = np.where(vmask, positions[safe, 0], 0.0)
        ys[:] = np.where(vmask, positions[safe, 1], 0.0)
        if headings_arr is not None:
            harr = np.asarray(headings_arr, dtype=np.float64).reshape(-1)
            hmask = vmask & (safe < len(harr))
            hsafe = np.clip(safe, 0, max(0, len(harr) - 1))
            if len(harr) > 0:
                headings[:] = np.where(hmask, harr[hsafe], 0.0)
        if velocities is not None:
            vel = np.asarray(velocities, dtype=np.float64)
            wmask = vmask & (safe < len(vel))
            wsafe = np.clip(safe, 0, max(0, len(vel) - 1))
            if len(vel) > 0:
                vxs[:] = np.where(wmask, vel[wsafe, 0], 0.0)
                vys[:] = np.where(wmask, vel[wsafe, 1], 0.0)
        else:
            # Finite-difference fallback over consecutive LOG frames (the
            # log's own dt, not the sim step): needs frame f AND f+1 valid.
            nxt_ok = (safe + 1 < n_frames)
            nsafe = np.clip(safe + 1, 0, n_frames - 1)
            nmask = vmask & nxt_ok & varr[nsafe]
            dx = (positions[nsafe, 0] - positions[safe, 0]) / scenario_dt
            dy = (positions[nsafe, 1] - positions[safe, 1]) / scenario_dt
            vxs[:] = np.where(nmask, dx, 0.0)
            vys[:] = np.where(nmask, dy, 0.0)

        if past_log:
            vxs[:] = 0.0
            vys[:] = 0.0

        length, width = _track_dims(track, sample_frame, cfg)

        out[str(track_id)] = AgentPrediction(
            track_id=str(track_id),
            x=xs,
            y=ys,
            heading=headings,
            vx=vxs,
            vy=vys,
            valid=valid,
            type=track_type,
            length=length,
            width=width,
        )

    return out


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _predict_idm_velocity_profile(
    *,
    initial_speed: float,
    lead_dist: Optional[float],
    lead_speed: float,
    policy: IDMPolicy,
    num_steps: int,
    dt: float,
    max_speed: float,
    max_accel: float,
    max_brake: float,
    speed_limit_mps: Optional[float],
) -> np.ndarray:
    """Predict the ego's IDM-driven velocity profile for the next ``num_steps``.

    Used by the LQR's lateral linearization to feed in a per-step
    velocity (faithful to upstream's LTV LQR formulation). Holds the
    lead's gap and speed at the current values for the prediction
    window — a constant-lead approximation that is consistent with
    NavSafe's per-step IDM update and that matches the granularity
    upstream's reference-velocity profile gets from the proposal's
    pre-simulated states.

    Acceleration is clipped to the same physical bounds the
    EgoDynamics step applies, so the predicted velocity profile
    matches what the bicycle model will actually produce given the
    same control input. The profile is therefore a reliable
    linearization point for the LQR.

    Returns an array of length ``num_steps`` with the predicted
    velocity at each LQR step (i.e. the velocity at the *start* of
    that step).
    """
    if num_steps <= 0:
        return np.array([initial_speed], dtype=np.float64)

    velocities = np.empty(num_steps, dtype=np.float64)
    cur_speed = float(max(0.0, initial_speed))
    cur_gap = (
        float(lead_dist)
        if lead_dist is not None and np.isfinite(lead_dist) and lead_dist <= 1e9
        else None
    )
    cur_lead_speed = float(lead_speed) if cur_gap is not None else 0.0

    for k in range(num_steps):
        velocities[k] = cur_speed
        a_idm = idm_accel(
            speed=cur_speed,
            lead_dist=cur_gap,
            lead_speed=cur_lead_speed,
            policy=policy,
            speed_limit_mps=speed_limit_mps,
        )
        # Clip to ego's physical bounds (matches what EgoDynamics will
        # actually apply for the corresponding throttle/brake norm).
        a_clipped = max(-max_brake, min(max_accel, a_idm))
        cur_speed = max(0.0, min(max_speed, cur_speed + a_clipped * dt))

        # Constant-lead approximation: gap shrinks at the closing rate
        # if a lead is present. Floor at 0 to avoid negative gaps.
        if cur_gap is not None:
            cur_gap = max(0.0, cur_gap - (cur_speed - cur_lead_speed) * dt)

    return velocities


def _world_to_ego_lateral_forward(
    points_world: np.ndarray,
    *,
    initial_x: float,
    initial_y: float,
    initial_heading: float,
) -> np.ndarray:
    """Transform world-frame points into ego frame ``[lateral, forward]``."""
    rel = points_world - np.array([initial_x, initial_y], dtype=np.float64)
    cos_h = float(np.cos(-initial_heading))
    sin_h = float(np.sin(-initial_heading))
    R = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    rotated = (R @ rel.T).T  # (N, 2) in (forward, left) order
    return np.column_stack([rotated[:, 1], rotated[:, 0]])  # (lateral, forward)


__all__ = [
    "AgentPrediction",
    "ProposalState",
    "predict_agents_constant_velocity",
    "predict_agents_from_live_states",
    "predict_agents_log_replay",
    "simulate_proposal",
]
