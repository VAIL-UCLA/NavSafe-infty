# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared lateral-steering primitives (pure pursuit + one-step LQR).

Pure-numpy path followers shared by two consumers on opposite sides of the
core/policy boundary:

* :mod:`navsafe.core.controllers` — the controller-in-the-loop trajectory
  trackers (``PurePursuitTracker`` / ``LQRTracker``);
* :mod:`navsafe.policy.state.pdm_closed_planner.forward_sim` — PDM-Closed's
  proposal forward simulation, where these functions were originally written
  as faithful ports of BridgeSim pure pursuit and ``tuplan_garage``'s batch
  LQR.

They were relocated here (verbatim) from ``forward_sim.py`` to break the
``core → policy`` import cycle: ``navsafe.core`` is documented as pure-Python
with **no upward dependencies** (so it imports without the heavy stack), so the shared code lives *down* here
and ``forward_sim`` imports it, not the other way around.

``_lqr_steer`` reads its tuning from any object satisfying
:class:`LateralLQRParams` — in practice either the planner's ``PDMConfig``
(policy side) or the :class:`LateralLQRConfig` defined below (core side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np


class LateralLQRParams(Protocol):
    """Structural type for the parameters :func:`_lqr_steer` reads.

    Satisfied by ``navsafe.policy.state.pdm_closed_planner.config.PDMConfig``
    (which carries many more fields) and by :class:`LateralLQRConfig`.
    Members are declared as read-only properties so frozen dataclasses and
    plain attributes both conform.
    """

    @property
    def sim_dt(self) -> float: ...

    @property
    def wheelbase(self) -> float: ...

    @property
    def max_steering_angle_rad(self) -> float: ...

    @property
    def lqr_q_lateral(self) -> Tuple[float, float, float]: ...

    @property
    def lqr_r_lateral(self) -> float: ...

    @property
    def lqr_tracking_horizon(self) -> int: ...

    @property
    def lqr_stopping_velocity(self) -> float: ...


@dataclass(frozen=True)
class LateralLQRConfig:
    """Minimal :class:`LateralLQRParams` bag for core-side callers.

    The ``lqr_*`` tuning defaults mirror ``PDMConfig``'s (both sourced from
    ``tuplan_garage/batch_lqr.py``) except ``lqr_q_lateral``, retuned for the
    execution tracker (see the field comment). Keep the remaining defaults equal to
    ``PDMConfig``'s.
    """

    sim_dt: float = 0.1
    # NOTE: the vehicle-geometry defaults here match the tracker
    # constructors in ``navsafe/core/controllers.py`` (which always pass
    # them explicitly), NOT ``PDMConfig`` — its default wheelbase is the
    # tuplan_garage 2.7 m. Only the ``lqr_*`` fields are the pinned pair.
    wheelbase: float = 2.8
    max_steering_angle_rad: float = 0.6
    # Lateral-position weight raised 1 -> 10 after an offline
    # log-following sweep over all 270 navsafe-loop full_test bundles (the
    # tracker follows the logged ego through the bicycle model, replanning
    # every 5 frames as the evaluator does): mean lateral error 0.111 m ->
    # 0.069 m, p95 0.329 m -> 0.187 m, lower on 98 % of bundles, with the
    # along-track lag and speed error unchanged. Heading weight, R and the
    # horizon stay at nuPlan's values; horizon 5 tightened the mean but
    # fattened the tail (p95 0.36 m) and was rejected. This is the ONE field
    # that deliberately differs from ``PDMConfig`` (the planner's internal
    # forward-sim tracker keeps the faithful tuplan_garage tuning).
    lqr_q_lateral: Tuple[float, float, float] = (10.0, 10.0, 0.0)
    lqr_r_lateral: float = 1.0
    lqr_tracking_horizon: int = 10
    lqr_stopping_velocity: float = 0.2

    def __post_init__(self) -> None:
        """Reject degenerate parameters (mirrors ``PDMConfig.__post_init__``).

        ``LQRTracker`` previously built a ``PDMConfig``, whose validation
        caught these; a zero ``sim_dt`` freezes the steering integrator and a
        zero ``max_steering_angle_rad`` divides by zero in ``_lqr_steer``.
        """
        if self.sim_dt <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {self.sim_dt}")
        if self.wheelbase <= 0.0:
            raise ValueError(f"wheelbase must be positive, got {self.wheelbase}")
        if self.max_steering_angle_rad <= 0.0:
            raise ValueError(
                f"max_steering_angle_rad must be positive, "
                f"got {self.max_steering_angle_rad}"
            )
        if self.lqr_tracking_horizon < 1:
            raise ValueError(
                f"lqr_tracking_horizon must be >= 1, "
                f"got {self.lqr_tracking_horizon}"
            )


def _project_point_on_path(
    px: float,
    py: float,
    path: np.ndarray,
    seg: np.ndarray,
    seg_lengths_safe: np.ndarray,
    cum_lengths: np.ndarray,
) -> Tuple[float, int]:
    """Project ``(px, py)`` onto ``path`` and return ``(arc_length_s, segment_idx)``."""
    rel = np.array([px, py], dtype=np.float64) - path[:-1]
    t_unclamped = np.einsum("ij,ij->i", rel, seg) / np.maximum(
        seg_lengths_safe ** 2, 1e-24
    )
    t = np.clip(t_unclamped, 0.0, 1.0)
    proj = path[:-1] + t[:, None] * seg
    d = np.linalg.norm(proj - np.array([px, py]), axis=1)
    idx = int(np.argmin(d))
    s = float(cum_lengths[idx] + t[idx] * seg_lengths_safe[idx])
    return s, idx


def _pure_pursuit_steer_norm(
    *,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    path: np.ndarray,
    cum_lengths: np.ndarray,
    lookahead_m: float,
    wheelbase: float,
    max_steer_angle: float,
) -> float:
    """Return a steering input in ``[-1, 1]`` for ``EgoDynamics.step``."""
    diffs = path - np.array([ego_x, ego_y], dtype=np.float64)
    dists = np.linalg.norm(diffs, axis=1)
    nearest_idx = int(np.argmin(dists))

    seg_idx = max(0, min(nearest_idx, len(path) - 2))
    a = path[seg_idx]
    b = path[seg_idx + 1]
    seg_vec = b - a
    seg_len = float(np.linalg.norm(seg_vec))
    if seg_len < 1e-9:
        s_ego = float(cum_lengths[seg_idx])
    else:
        t = float(np.dot(np.array([ego_x, ego_y]) - a, seg_vec) / (seg_len * seg_len))
        t = max(0.0, min(1.0, t))
        s_ego = float(cum_lengths[seg_idx] + t * seg_len)

    s_target = s_ego + lookahead_m
    if s_target >= cum_lengths[-1]:
        last_vec = path[-1] - path[-2]
        last_norm = float(np.linalg.norm(last_vec))
        if last_norm < 1e-9:
            target = path[-1].copy()
        else:
            target = path[-1] + last_vec / last_norm * (s_target - cum_lengths[-1])
    else:
        idx = int(np.searchsorted(cum_lengths, s_target, side="right")) - 1
        idx = max(0, min(idx, len(path) - 2))
        seg_len_i = max(float(cum_lengths[idx + 1] - cum_lengths[idx]), 1e-12)
        frac = (s_target - cum_lengths[idx]) / seg_len_i
        target = path[idx] + frac * (path[idx + 1] - path[idx])

    dx = float(target[0] - ego_x)
    dy = float(target[1] - ego_y)
    cos_h = float(np.cos(ego_heading))
    sin_h = float(np.sin(ego_heading))
    forward = dx * cos_h + dy * sin_h
    left = -dx * sin_h + dy * cos_h
    L_d = float(np.hypot(forward, left))
    if L_d < 1e-6:
        return 0.0
    sin_alpha = left / L_d
    delta_phys = float(np.arctan2(2.0 * wheelbase * sin_alpha, L_d))

    delta_phys = float(np.clip(delta_phys, -max_steer_angle, max_steer_angle))
    return float(np.clip(-delta_phys / max_steer_angle, -1.0, 1.0))


def _lqr_steer(
    *,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    ego_speed: float,
    ego_steering_angle: float,
    path: np.ndarray,
    seg: np.ndarray,
    seg_lengths: np.ndarray,
    seg_lengths_safe: np.ndarray,
    cum_lengths: np.ndarray,
    tangents: np.ndarray,
    cfg: LateralLQRParams,
    velocity_profile: Optional[np.ndarray] = None,
    curvature_profile: Optional[np.ndarray] = None,
    reference_pose: Optional[np.ndarray] = None,
    clip_command: bool = True,
    heading_mode: str = "course",
) -> Tuple[float, float]:
    """Single-step LQR-style lateral controller.

    ``heading_mode`` selects the heading whose error the lateral state
    carries: ``"course"`` (``θ + atan(½ tan δ)``) is the matched measurement
    for :class:`~navsafe.core.ego_dynamics.EgoDynamics`' centre-of-wheelbase
    plant (the execution tracker); ``"body"`` (``θ``) is the reference
    ``BatchLQRTracker`` measurement for a rear-axle plant, which the
    planner's scored copy uses.

    ``clip_command`` bounds the integrated angle (and the normalised steer)
    to ``±max_steering_angle_rad`` before returning. The execution tracker
    keeps that (its integrator has no plant feedback and must not wind up);
    the proposal forward-sim passes ``False`` because the reference
    ``BatchLQRTracker`` emits an unclipped steering rate and its PLANT clips
    the angle after the actuator filter — the filtered plant here does the
    same and feeds the clipped angle back as the next call's
    ``ego_steering_angle``.

    Faithful adaptation of ``tuplan_garage/batch_lqr.py`` for the
    proposal forward-simulation loop. We solve a one-step LQR problem
    over a tracking horizon ``N = cfg.lqr_tracking_horizon`` steps:

    * Lateral state: ``[lateral_error, heading_error, steering_angle]``
    * Input: ``steering_rate``
    * Output: ``EgoDynamics.step``-compatible normalised steer

    ``heading_error`` is the error of the **course** heading
    ``theta + atan(0.5 * tan(delta))``, not of the body heading
    ``theta``. The plant is :class:`~navsafe.core.ego_dynamics.
    EgoDynamics`, a symmetric centre-of-wheelbase bicycle: the centre
    translates along ``theta + beta`` and ``theta_dot = (2v/L) *
    sin(beta) ~= v * delta / L``. The measurement and the A-matrix
    gain below are therefore a **matched pair** — measure body heading
    with the ``v/L`` gain and the
    slip angle becomes an un-modelled disturbance that shows up as a
    steady-state lateral bias on every curve.

    Args:
        velocity_profile: Optional length-``N`` array of predicted
            velocities over the tracking horizon. When ``None``,
            ``ego_speed`` is held constant. The faithful behaviour is
            to pass the IDM-predicted speeds for the next ``N``
            sim-steps (matching ``tuplan_garage``'s linear-time-varying
            linearization). ``tuplan_garage`` models a rear-axle
            bicycle and so writes the lateral A-matrix as
            ``[[0, v_n, 0], [0, 0, v_n/L], [0, 0, 0]]``; NavSafe's
            corrected symmetric bicycle has the same first-order yaw gain.

    Returns ``(steer_norm, new_steering_angle)``. The new steering
    angle is the integrated steering after applying ``steering_rate
    * dt`` for one step — the caller threads it back as
    ``ego_steering_angle`` for the next call so the LQR sees the true
    steering integrator state.
    """
    # Project ego onto the path. Legacy/core callers use that projection as
    # the reference state. PDM-Closed passes the ideal proposal's pose at the
    # current simulation index, matching CaRL BatchLQRTracker's exact
    # proposal-state comparison rather than silently sliding the reference
    # forward/backward to the closest path point.
    s_ego, seg_idx = _project_point_on_path(
        ego_x, ego_y, path, seg, seg_lengths_safe, cum_lengths
    )
    if reference_pose is None:
        ref_xy = path[seg_idx]
        tan = tangents[seg_idx]
        ref_heading = float(np.arctan2(tan[1], tan[0]))
    else:
        ref = np.asarray(reference_pose, dtype=np.float64).reshape(-1)
        if ref.size < 3:
            raise ValueError("reference_pose must contain x, y, heading")
        ref_xy = ref[:2]
        ref_heading = float(ref[2])
        tan = np.array(
            [math.cos(ref_heading), math.sin(ref_heading)], dtype=np.float64
        )

    # Lateral error (signed, +ve to the LEFT of the reference direction).
    rel = np.array([ego_x - ref_xy[0], ego_y - ref_xy[1]])
    left_normal = np.array([-tan[1], tan[0]])
    lateral_error = float(np.dot(rel, left_normal))

    # Heading error normalised into (-pi, pi].
    #
    # The state we track is the COURSE heading, not the body heading.
    # ``EgoDynamics`` is a centre-slip bicycle: the centre of mass
    # translates along ``heading + beta`` with
    # ``beta = atan(0.5 * tan(delta))``, so the lateral-error rate is
    # ``v * sin(heading + beta - ref)``, not ``v * sin(heading - ref)``.
    # Measuring body heading here while the A-matrix below uses the
    # symmetric-bicycle gain ``v/L`` leaves the loop internally
    # inconsistent and produces a steady-state lateral bias on every
    # curve (the slip angle is exactly the un-modelled term).
    if heading_mode == "body":
        measured_heading = float(ego_heading)
    elif heading_mode == "course":
        measured_heading = ego_heading + math.atan(
            0.5 * math.tan(float(ego_steering_angle))
        )
    else:
        raise ValueError(
            f"heading_mode must be 'course' or 'body', got {heading_mode!r}")
    head_err = measured_heading - ref_heading
    head_err = (head_err + np.pi) % (2.0 * np.pi) - np.pi

    # Discrete-time linearised lateral dynamics over N steps
    # (``heading_error`` is the COURSE-heading error — see the
    # docstring; ``wheelbase`` is the symmetric-bicycle gain):
    #   lateral_error_dot   = velocity * heading_error
    #   heading_error_dot   = velocity * (steering_angle / wheelbase
    #                                     - curvature)
    #   steering_angle_dot  = steering_rate
    # When ``velocity_profile`` is provided, use the per-step velocity
    # for the linearization (faithful to upstream's LTV LQR). When
    # ``None`` (legacy / fallback path), hold ``ego_speed`` constant.
    N = int(cfg.lqr_tracking_horizon)
    dt_lqr = float(cfg.sim_dt)
    L = float(cfg.wheelbase)

    if velocity_profile is None:
        velocities = np.full(N, float(ego_speed), dtype=np.float64)
    else:
        vp = np.asarray(velocity_profile, dtype=np.float64).reshape(-1)
        if vp.shape[0] >= N:
            velocities = vp[:N]
        else:
            # Pad short profiles with the last value.
            velocities = np.concatenate(
                [vp, np.full(N - vp.shape[0], vp[-1] if vp.size > 0 else float(ego_speed))]
            )

    if curvature_profile is not None:
        cp = np.asarray(curvature_profile, dtype=np.float64).reshape(-1)
        if cp.shape[0] >= N:
            curvatures = cp[:N]
        else:
            curvatures = np.pad(
                cp,
                (0, N - cp.shape[0]),
                mode="edge" if cp.size else "constant",
            )
    else:
        # Fallback for core callers without a CaRL-fitted proposal profile.
        curvatures = np.zeros(N, dtype=np.float64)
        for n in range(N):
            idx_n = min(seg_idx + n, len(tangents) - 1)
            idx_np1 = min(idx_n + 1, len(tangents) - 1)
            if idx_n == idx_np1:
                curvatures[n] = 0.0
            else:
                t_a = tangents[idx_n]
                t_b = tangents[idx_np1]
                ang_a = np.arctan2(t_a[1], t_a[0])
                ang_b = np.arctan2(t_b[1], t_b[0])
                dtheta = (ang_b - ang_a + np.pi) % (2.0 * np.pi) - np.pi
                ds = max(float(seg_lengths[idx_n]), 1e-9)
                curvatures[n] = float(dtheta / ds)

    # Build per-step linearised dynamics matrices and propagate to
    # step N. Continuous → Euler discrete:
    #   x_{n+1} = (I + dt * Ac_n) @ x_n + dt * Bc @ u + dt * gc_n
    # with x = [lateral_error, heading_error, steering_angle], u = steering_rate.
    # The ``v`` in ``Ac_n`` is now ``velocities[n]`` (LTV faithful).
    I3 = np.eye(3, dtype=np.float64)
    Bc = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)  # steering_rate enters steering_angle

    A_total = I3.copy()
    B_total = np.zeros((3, 1), dtype=np.float64)
    g_total = np.zeros(3, dtype=np.float64)

    for n in range(N):
        v_n = float(velocities[n])
        # Steering gain matched to the actual symmetric bicycle:
        # heading_dot = (2v/L)·sin(atan(0.5·tan δ)) ≈ v·δ/L.
        Ac = np.array(
            [
                [0.0, v_n, 0.0],
                [0.0, 0.0, v_n / L],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        gc = np.array([0.0, -v_n * curvatures[n], 0.0], dtype=np.float64)
        A_step = I3 + dt_lqr * Ac
        B_step = dt_lqr * Bc
        g_step = dt_lqr * gc

        A_total = A_step @ A_total
        B_total = A_step @ B_total + B_step
        g_total = A_step @ g_total + g_step

    x0 = np.array(
        [lateral_error, head_err, ego_steering_angle], dtype=np.float64
    )

    # One-step LQR: minimise (x_N)^T Q x_N + u^T R u  s.t.
    #   x_N = A_total x_0 + B_total u + g_total
    Q = np.diag(np.asarray(cfg.lqr_q_lateral, dtype=np.float64))
    R = float(cfg.lqr_r_lateral)
    state_zero_input = A_total @ x0 + g_total

    # Faithful to ``tuplan_garage/batch_lqr.py::_solve_one_step_lateral_lqr``:
    # wrap heading_error and steering_angle entries via
    # ``arctan2(sin(angle), cos(angle))`` BEFORE solving the LQR. This
    # guarantees the LQR treats angle errors near ±π consistently
    # (e.g., a heading error of 3.14 rad and -3.14 rad are the same
    # physical state and must produce the same steering rate).
    HEADING_ERROR_IDX = 1
    STEERING_ANGLE_IDX = 2
    for idx in (HEADING_ERROR_IDX, STEERING_ANGLE_IDX):
        a = float(state_zero_input[idx])
        state_zero_input[idx] = float(np.arctan2(np.sin(a), np.cos(a)))

    # u* = -(B^T Q B + R)^{-1} B^T Q (A x_0 + g)
    BtQ = B_total.T @ Q
    denom_arr = BtQ @ B_total + R
    denom = float(np.asarray(denom_arr).reshape(-1)[0])
    if abs(denom) < 1e-12:
        steering_rate = 0.0
    else:
        num_arr = -(BtQ @ state_zero_input) / denom
        steering_rate = float(np.asarray(num_arr).reshape(-1)[0])

    # Apply for one sim step (we re-solve the LQR every step).
    new_steering_angle = ego_steering_angle + dt_lqr * steering_rate
    if clip_command:
        new_steering_angle = float(
            np.clip(
                new_steering_angle,
                -cfg.max_steering_angle_rad,
                cfg.max_steering_angle_rad,
            )
        )

    # EgoDynamics expects normalised steer in [-1, 1] with the
    # convention "positive steer = right". The physical-left-positive
    # value therefore needs the sign flip applied here too.
    steer_norm = -new_steering_angle / cfg.max_steering_angle_rad
    if clip_command:
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))
    return float(steer_norm), float(new_steering_angle)


__all__ = [
    "LateralLQRConfig",
    "LateralLQRParams",
]
