# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy utilities for traversing a policy's planned trajectory: model-dt →
sim-dt interpolation, cumulative arc lengths, ego-anchored pacing speed,
paced-point lookup, and noise-robust heading extraction. They live in
``navsafe.core`` so shared control code does not depend on evaluation internals. The :class:`~navsafe.evaluation.evaluator.Evaluator`
re-exports them as static methods for backward compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

PLAN_PACING_FACTOR: float = 0.80

# Distance below which an ego-frame plan is already considered to start at
# the ego origin (see ``ego_plan_with_origin``).
_ORIGIN_TOL_M: float = 1e-9


def cumulative_arc_length(path: np.ndarray) -> np.ndarray:
    """(N,) cumulative arc length along an (N, 2) polyline (arc[0] == 0)."""
    if len(path) < 2:
        return np.zeros(len(path))
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def plan_pacing(world_traj: np.ndarray, sim_dt: float) -> Tuple[np.ndarray, float]:
    """Cumulative arc lengths + pacing speed for a plan.

    The pace is ``PLAN_PACING_FACTOR`` × the plan's average speed, applied
    immediately (see that constant for where the factor comes from).
    ``world_traj[0]`` must be the ego pose the plan was produced from
    (``ego_plan_with_origin``), otherwise the returned ``arc`` mislabels
    the ego→waypoint[0] gap as zero and every replan frame executes it for
    free on top of its paced step.

    Args:
        world_traj: (N, 2) sim_dt-interpolated world-frame plan, starting
            at the ego pose.
        sim_dt: Simulation timestep in seconds.

    Returns:
        (arc, speed) — ``arc`` is the (N,) cumulative arc length along the
        plan; ``speed`` the pacing speed in m/s (0 for degenerate plans).
    """
    if len(world_traj) < 2:
        return np.zeros(len(world_traj)), 0.0
    arc = cumulative_arc_length(world_traj)
    speed = PLAN_PACING_FACTOR * float(arc[-1]) / ((len(arc) - 1) * sim_dt)
    return arc, speed


def point_at_arc(world_traj: np.ndarray, arc: np.ndarray,
                 s: float) -> Tuple[np.ndarray, int]:
    """Point at arc length ``s`` along a polyline plan.

    Returns the interpolated (2,) point and the index of the segment's
    start waypoint (for heading lookahead). Clamps to the plan ends.
    """
    if len(world_traj) == 0:
        raise ValueError("empty plan")
    if len(world_traj) == 1 or s <= 0.0:
        return world_traj[0].copy(), 0
    total = float(arc[-1])
    if s >= total:
        return world_traj[-1].copy(), len(world_traj) - 1
    i = int(np.searchsorted(arc, s, side="right")) - 1
    seg_len = arc[i + 1] - arc[i]
    t = (s - arc[i]) / seg_len if seg_len > 0 else 0.0
    point = world_traj[i] + t * (world_traj[i + 1] - world_traj[i])
    return point, i


def teleport_plan_step(
    world_traj: np.ndarray,
    arc: np.ndarray,
    arc_s: float,
    speed: float,
    sim_dt: float,
    current_position: np.ndarray,
    current_heading: float,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Advance one production teleport-execution step.

    Returns ``(position, heading, velocity_xy, next_arc_s)``.  Keeping this
    state transition in the shared execution module prevents diagnostic
    rollouts from approximating a replan boundary with average displacement
    and thereby feeding a different heading/speed into the next plan.
    """
    if sim_dt <= 0.0:
        raise ValueError(f"sim_dt must be positive, got {sim_dt}")
    next_arc_s = float(arc_s) + float(speed) * float(sim_dt)
    target, seg_idx = point_at_arc(world_traj, arc, next_arc_s)
    heading = heading_from_plan(
        world_traj[seg_idx:], float(current_heading))
    current_xy = np.asarray(current_position, dtype=np.float64)[:2]
    velocity_xy = (target - current_xy) / float(sim_dt)
    return target, heading, velocity_xy, next_arc_s


def heading_from_plan(
    world_traj_subset: np.ndarray,
    current_heading: float,
    lookahead: int = 5,
    min_disp: float = 0.2,
) -> float:
    """Target heading from a plan segment long enough to be noise-free.

    Adjacent sim_dt-interpolated waypoints are only centimetres apart at
    low speed, so their atan2 is dominated by noise and can flip the ego
    heading by up to 180° (which then corrupts the next plan's
    ego→world transform). Use a waypoint ``lookahead`` steps ahead
    (~0.5 s at sim_dt=0.1) and keep the current heading when the plan is
    near-stationary.

    Args:
        world_traj_subset: (N, 2) world-frame waypoints, N >= 1.
        current_heading: Ego heading to fall back to (radians).
        lookahead: Index of the waypoint used for the direction vector.
        min_disp: Displacement (metres) below which the plan is treated
            as stationary and ``current_heading`` is returned.
    """
    j = min(lookahead, len(world_traj_subset) - 1)
    if j < 1:
        return float(current_heading)
    d = world_traj_subset[j] - world_traj_subset[0]
    if float(np.hypot(d[0], d[1])) < min_disp:
        return float(current_heading)
    return float(np.arctan2(d[1], d[0]))


def interpolate_plan(trajectory: np.ndarray, model_dt: float,
                     sim_dt: float) -> np.ndarray:
    """Interpolate trajectory from model_dt to sim_dt intervals.

    The trajectory is in ego frame where ego is at (0, 0) at time 0, which
    is included as the resampled plan's first sample: arc-length execution
    measures from index 0 and resets ``arc_s`` to 0 at each replan, so
    index 0 must be the pose the plan was produced from. Sampling from
    ``t = sim_dt`` instead (as this did before) put index 0 one sim step
    ahead of the ego while ``cumulative_arc_length`` still called it arc 0,
    so every replan frame executed two steps' worth of travel.

    The ``model_dt <= sim_dt`` pass-through returns the model's own
    waypoints, which start at ``t = model_dt`` and therefore do *not*
    include the origin; callers normalize with ``ego_plan_with_origin``.

    Args:
        trajectory: (N, 2) waypoints at model_dt intervals [lateral, forward].
        model_dt: Time between model waypoints (e.g. 0.5s).
        sim_dt: Simulation timestep (e.g. 0.1s).

    Returns:
        Interpolated (M, 2) trajectory at sim_dt intervals, starting at the
        ego origin (0, 0).
    """
    if model_dt <= sim_dt:
        return trajectory

    n = len(trajectory)
    if n == 0:
        return trajectory

    # Prepend ego origin at t=0
    origin = np.array([[0.0, 0.0]])
    traj_with_origin = np.vstack([origin, trajectory])

    t_original = np.arange(0, n + 1) * model_dt
    t_max = t_original[-1]
    t_interp = np.arange(0.0, t_max + sim_dt / 2, sim_dt)

    interp_x = np.interp(t_interp, t_original, traj_with_origin[:, 0])
    interp_y = np.interp(t_interp, t_original, traj_with_origin[:, 1])

    return np.stack([interp_x, interp_y], axis=1)


@dataclass(frozen=True)
class ControllerReference:
    """One time-aligned controller contract shared by every execution path."""

    trajectory: np.ndarray
    speeds_mps: np.ndarray


def interpolate_controller_plan(trajectory: np.ndarray, model_dt: float,
                                sim_dt: float) -> np.ndarray:
    """Resample future-only controller references without an ego-origin chord.

    Policy waypoints describe future poses at ``model_dt, 2*model_dt, ...``.
    :func:`interpolate_plan` intentionally adds the physical ego pose at
    ``t=0`` for arc-length/teleport execution.  That composition is invalid
    for a laterally projected controller reference: joining ``(0, 0)`` to the
    first offset waypoint invents a diagonal segment which is neither part of
    the planner path nor physically trackable.

    Controller references therefore keep their future-only geometry.  Samples
    after the first model waypoint are ordinary linear interpolation.  Samples
    needed before it are linearly extrapolated from the first *two* planner
    waypoints, preserving the planner's initial tangent instead of drawing a
    chord from the ego.  The result is sampled at ``sim_dt, 2*sim_dt, ...`` so
    its indices retain the production controller's elapsed-time semantics.

    The helper works for any ``(N, D)`` future sample array, including scalar
    speed profiles represented as ``(N, 1)``.  A single waypoint cannot define
    a tangent and is returned unchanged.
    """
    if model_dt <= 0.0 or sim_dt <= 0.0:
        raise ValueError(
            f"model_dt and sim_dt must be positive, got {model_dt}, {sim_dt}")
    samples = np.asarray(trajectory, dtype=np.float64)
    if model_dt <= sim_dt or samples.ndim != 2 or len(samples) <= 1:
        return samples.copy()

    n = len(samples)
    t_original = np.arange(1, n + 1, dtype=np.float64) * float(model_dt)
    t_interp: np.ndarray = np.arange(
        float(sim_dt), t_original[-1] + float(sim_dt) / 2.0,
        float(sim_dt), dtype=np.float64)
    result = np.stack([
        np.interp(t_interp, t_original, samples[:, dim])
        for dim in range(samples.shape[1])
    ], axis=1)

    before_first = t_interp < t_original[0]
    if np.any(before_first):
        slope = (samples[1] - samples[0]) / float(model_dt)
        result[before_first] = (
            samples[0]
            + (t_interp[before_first, None] - t_original[0]) * slope)
    return result


def controller_waypoint_speeds(trajectory: np.ndarray,
                               model_dt: float) -> np.ndarray:
    """Per-waypoint speeds for a future-only projected controller path.

    The first policy waypoint is reached over ``model_dt`` from the physical
    ego, but its perpendicular displacement may be a lane projection rather
    than travel.  Use the first planner segment as the local tangent and
    project the ego-to-first vector onto it.  This preserves legitimate
    forward pace without turning lateral offset into speed or steering
    geometry.  Later intervals use ordinary waypoint distances.
    """
    path = np.asarray(trajectory, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 2 or len(path) == 0:
        return np.zeros(0, dtype=np.float64)
    if model_dt <= 0.0:
        raise ValueError(f"model_dt must be positive, got {model_dt}")

    xy = path[:, :2]
    if len(xy) == 1:
        first_distance = float(np.linalg.norm(xy[0]))
    else:
        first_segment = xy[1] - xy[0]
        segment_norm = float(np.linalg.norm(first_segment))
        if segment_norm > 1e-9:
            tangent = first_segment / segment_norm
            first_distance = max(0.0, float(np.dot(xy[0], tangent)))
        else:
            first_distance = float(np.linalg.norm(xy[0]))
    distances = np.concatenate([
        np.array([first_distance], dtype=np.float64),
        np.linalg.norm(np.diff(xy, axis=0), axis=1),
    ])
    return distances / float(model_dt)


def controller_speed_profile(trajectory: np.ndarray, model_dt: float,
                             sim_dt: float) -> np.ndarray:
    """Dense speeds derived from projected controller-path intervals."""
    if model_dt <= 0.0 or sim_dt <= 0.0:
        raise ValueError(
            f"model_dt and sim_dt must be positive, got {model_dt}, {sim_dt}")
    path = np.asarray(trajectory, dtype=np.float64)
    interval_speeds = controller_waypoint_speeds(path, model_dt)
    if len(interval_speeds) == 0:
        return interval_speeds
    sample_times: np.ndarray = np.arange(
        float(sim_dt), len(path) * float(model_dt) + float(sim_dt) / 2.0,
        float(sim_dt), dtype=np.float64)
    interval_indices = np.minimum(
        np.ceil(sample_times / float(model_dt)).astype(np.int64) - 1,
        len(interval_speeds) - 1)
    return interval_speeds[interval_indices]


def controller_reference(
    trajectory: np.ndarray,
    model_dt: float,
    sim_dt: float,
    explicit_speeds_mps: Optional[np.ndarray] = None,
) -> ControllerReference:
    """Build the shared future-only geometry and time-aligned speed profile.

    Explicit policy velocities take precedence.  XY-only plans use the same
    tangent-projected first interval everywhere, so live execution and
    counterfactual verification cannot assign different speeds to one edit.
    """
    path = np.asarray(trajectory, dtype=np.float64)
    geometry = interpolate_controller_plan(path[:, :2], model_dt, sim_dt)
    speeds: Optional[np.ndarray] = None
    if explicit_speeds_mps is not None:
        explicit = np.asarray(
            explicit_speeds_mps, dtype=np.float64).reshape(-1)
        if len(explicit) == len(path):
            speeds = interpolate_controller_plan(
                explicit[:, None], model_dt, sim_dt)[:, 0]
        elif len(explicit) == len(geometry):
            speeds = explicit.copy()
    if speeds is None:
        speeds = controller_speed_profile(path[:, :2], model_dt, sim_dt)
    return ControllerReference(
        trajectory=geometry,
        speeds_mps=np.maximum(speeds, 0.0),
    )


def ego_plan_with_origin(trajectory: np.ndarray) -> np.ndarray:
    """Ego-frame plan guaranteed to start at the ego origin (0, 0).

    ``plan_pacing`` measures arc length from index 0 and ``teleport_plan_step``
    advances from ``arc_s = 0`` after every replan, so index 0 must BE the pose
    the plan was produced from. A model's raw waypoints start one ``model_dt``
    ahead of the ego: ``interpolate_plan`` supplies the ``t = 0`` sample when it
    resamples, but returns those raw waypoints unchanged on the
    ``model_dt <= sim_dt`` pass-through, which carries the identical
    off-by-one.

    **Why here and not inside the pass-through branch.** "Index 0 is the ego
    pose" belongs to the arc-length *execution* contract, not to resampling.
    The execution seam is the one place that knows the concrete ego pose the
    plan is anchored to — it is the same pose the ego→world transform on the
    next line uses — so normalizing there makes the invariant hold for every
    route into pacing, including plans that never touch ``interpolate_plan``,
    rather than for the two branches that happen to exist today. Putting it
    inside the pass-through would also silently rewrite a caller's array for
    callers that want resampling only.

    Idempotent, so it composes with the resampled branch (which already starts
    at the origin) without inserting a zero-length first segment that would
    give ``point_at_arc`` duplicate arc entries.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
        return traj
    if float(np.hypot(traj[0, 0], traj[0, 1])) <= _ORIGIN_TOL_M:
        return traj
    return np.vstack([np.zeros((1, traj.shape[1]), dtype=traj.dtype), traj])


__all__ = [
    "ControllerReference",
    "PLAN_PACING_FACTOR",
    "controller_reference",
    "controller_speed_profile",
    "controller_waypoint_speeds",
    "cumulative_arc_length",
    "ego_plan_with_origin",
    "plan_pacing",
    "point_at_arc",
    "teleport_plan_step",
    "heading_from_plan",
    "interpolate_controller_plan",
    "interpolate_plan",
]
