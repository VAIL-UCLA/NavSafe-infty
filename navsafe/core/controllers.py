# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-tracking controllers for controller-in-the-loop execution.

Turns a policy's planned world-frame trajectory into normalized
``(steer, accel)`` actions for :class:`~navsafe.core.ego_dynamics.EgoDynamics`
(and, in physics execution mode, for the PhysX-integrated ego). This makes
the previously metadata-only ``controller_type`` evaluation config real:
instead of teleporting the ego along the plan, a tracker chases it the way
BridgeSim's PurePursuit + speed PID chased plans on a MetaDrive vehicle.

Control wraps the faithful LQR/path-follower ports shared with PDM-Closed's
proposal simulation (``navsafe/core/steering.py``, originally written in
``pdm_closed_planner/forward_sim.py``):

* :class:`PurePursuitTracker` — BridgeSim-style pure pursuit with a
  speed-adaptive lookahead.
* :class:`LQRTracker` — the tuplan_garage batch-LQR adaptation, with the
  steering-angle integrator threaded across steps.

Pure-pursuit longitudinal control starts from BridgeSim's
``PurePursuitController.control_pid`` (controller_md.py): P-control toward the
plan's pacing speed and full brake for a stop/gross overshoot. NavSafe adds
proportional braking for ordinary negative error because its kinematic bicycle
has no passive drag; otherwise a requested speed reduction never occurs.

Future (Phase 3) execution modes plug in below this interface: a PhysX wheel
articulation (``navsafe/component/robot/metadrive_vehicle.py``) or the
``omni.physxvehicle`` vehicle SDK consume the same ``(steer, accel)`` —
mapped to wheel/steer joint targets — so trackers stay execution-agnostic.

Relationship to :class:`~navsafe.core.unified_controller.UnifiedController`:
that earlier BridgeSim adaptation reads its speed target from the
trajectory's column 3, which the evaluator populates with per-step
position *gradients* (metres/step), not velocities — one reason it was
instantiated but never called at the execution seam. These trackers
supersede it for closed-loop execution (explicit pacing-speed input, the
faithful PDM path followers, LQR support); ``UnifiedController`` remains
for ``scripts/sensor_smoke_no_isaacsim.py``.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import numpy as np

from navsafe.core.plan_execution import cumulative_arc_length
from navsafe.core.steering import (
    LateralLQRConfig,
    _lqr_steer,
    _pure_pursuit_steer_norm,
)

# BridgeSim controller_md.py parity constants (PurePursuitController defaults).
SPEED_KP: float = 5.0
# BridgeSim can coast down through its vehicle model. NavSafe's bicycle
# executor has no drag, so a negative speed error otherwise holds forever.
# A separate, gentler brake gain makes requested speed reductions real without
# turning every small overshoot into the full-brake discontinuity.
SPEED_BRAKE_KP: float = 0.5
CLIP_DELTA: float = 1.0
# A positive crawl is a real command, not a stop.  The old 0.5 m/s cutoff
# made the verifier's only on-road 0.29 m/s tight-turn trajectory impossible
# to execute on 891... f65: the harness selected/braked exactly as if the
# model had requested zero.  Keep a narrow near-zero stop band for numerical
# plan noise; exact/near-exact stop plans and gross overshoots still full-brake.
BRAKE_SPEED: float = 0.25  # m/s — target below this ⇒ full brake
BRAKE_RATIO: float = 2.0   # ego overshooting target by this factor ⇒ full brake
AIM_DIST_BASE: float = 4.0     # m — pure-pursuit lookahead at standstill
AIM_DIST_GAIN_S: float = 0.1   # s — lookahead grows with speed

# nuPlan/tuplan_garage LQR defaults.  The reference velocity is sampled one
# tracking horizon in the future and a constant physical acceleration is
# chosen to minimize the velocity error there.  Keep these in the core
# tracker because this is the trajectory-execution controller, not a policy
# scoring knob.
LQR_Q_LONGITUDINAL: float = 10.0
LQR_R_LONGITUDINAL: float = 1.0
LQR_STOPPING_GAIN: float = 0.5


def _speed_control(current_speed: float, target_speed: float) -> float:
    """BridgeSim-parity longitudinal control → normalized accel in [-1, 1].

    P-control on positive speed error follows BridgeSim. Negative error uses
    a gentler proportional brake because NavSafe's kinematic bicycle has no
    passive drag: copying BridgeSim's ``clip(error, 0, ...)`` made every
    moderate speed reduction a permanent coast at the old speed. Full brake
    remains reserved for a stop target or a gross overshoot.
    """
    # A low but strictly increasing target is the first step of a legitimate
    # from-rest velocity ramp, not a stop. With replanning every simulator
    # frame, PDM repeatedly exposes that first ~0.05 m/s sample; treating all
    # targets below BRAKE_SPEED as stops made the controller command full
    # brake forever after a red light. BridgeSim avoids this because it uses
    # a whole-plan desired speed. Preserve the time-indexed PDM profile while
    # giving its positive ramp the equivalent ability to start. Exact zero,
    # a requested low-speed reduction, and gross overshoot still brake.
    positive_ramp = (
        target_speed > 1e-6 and current_speed <= target_speed + 1e-9)
    brake = (
        (target_speed < BRAKE_SPEED and not positive_ramp)
        or (target_speed > 1e-6 and current_speed / target_speed > BRAKE_RATIO)
    )
    if brake:
        return -1.0
    error = float(target_speed - current_speed)
    if error < 0.0:
        return float(np.clip(SPEED_BRAKE_KP * error, -1.0, 0.0))
    delta = float(np.clip(error, 0.0, CLIP_DELTA))
    return float(np.clip(SPEED_KP * delta, 0.0, 1.0))


class TrajectoryTracker:
    """Base tracker: world-frame plan → normalized ``(steer, accel)``.

    Args:
        wheelbase: Ego wheelbase in metres (matches ``EgoDynamicsCfg``).
        max_steer_angle: Max physical steering angle in radians.
        sim_dt: Simulation timestep in seconds.
    """

    def __init__(self, *, wheelbase: float = 2.8, max_steer_angle: float = 0.6,
                 sim_dt: float = 0.1) -> None:
        self.wheelbase = float(wheelbase)
        self.max_steer_angle = float(max_steer_angle)
        self.sim_dt = float(sim_dt)

    def reset(self) -> None:
        """Clear per-episode controller state (integrators)."""

    @property
    def speed_lookahead_steps(self) -> int:
        """Timed-plan offset used to select the longitudinal reference.

        Pure pursuit retains its immediate BridgeSim pacing target.  The LQR
        override returns nuPlan's tracking horizon (10 steps = 1 second).
        """
        return 0

    def compute(self, ego_state: Dict[str, Any], world_traj: np.ndarray,
                target_speed: float) -> Tuple[float, float]:
        """Compute normalized ``(steer, accel)`` toward the plan.

        Args:
            ego_state: Dict with ``position`` (2+,), ``heading`` (rad) and
                either ``speed`` or ``velocity``.
            world_traj: (N, 2) sim_dt-interpolated world-frame plan.
                Degenerate plans (< 2 distinct points) yield a hold-and-brake
                action. Arc lengths are recomputed internally after
                near-duplicate points are dropped.
            target_speed: Longitudinal target in m/s (the plan's pacing
                speed, e.g. ``plan_execution.plan_pacing``).
        """
        raise NotImplementedError

    @staticmethod
    def _clean_path(world_traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Drop near-duplicate waypoints and recompute arc lengths.

        Near-stationary plans repeat waypoints; zero-length segments give the
        LQR unbounded curvature estimates (dtheta / ~0) and give pure pursuit
        meaningless tangents. Recomputing the arc internally also removes any
        dependence on the caller's arc being consistent with the path.
        Returns (path, arc); path may have < 2 points (caller handles).
        """
        path = np.asarray(world_traj, dtype=np.float64)
        if len(path) >= 2:
            seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
            keep = np.concatenate([[True], seg_len > 1e-3])
            path = path[keep]
        return path, cumulative_arc_length(path)

    @staticmethod
    def _degenerate_action(speed: float, target_speed: float) -> Tuple[float, float]:
        """Action for a degenerate (<2 point) plan: hold the wheel, track speed.

        The wrapped path followers index ``path[-2]`` / segment arrays and
        would crash on a single-point plan (a stop or a broken model output);
        with no direction to follow, steer 0 and let the longitudinal
        controller brake toward the (typically zero) pacing speed.
        """
        return 0.0, _speed_control(speed, target_speed)

    @staticmethod
    def _ego_scalars(ego_state: Dict[str, Any]) -> Tuple[float, float, float, float]:
        pos = np.asarray(ego_state["position"], dtype=np.float64)
        heading = float(ego_state["heading"])
        if "speed" in ego_state:
            speed = float(ego_state["speed"])
        else:
            vel = np.asarray(ego_state.get("velocity", np.zeros(2)), dtype=np.float64)
            speed = float(np.hypot(vel[0], vel[1]))
        return float(pos[0]), float(pos[1]), heading, speed


class PurePursuitTracker(TrajectoryTracker):
    """BridgeSim-style pure pursuit (speed-adaptive lookahead) + speed P."""

    def compute(self, ego_state: Dict[str, Any], world_traj: np.ndarray,
                target_speed: float) -> Tuple[float, float]:
        x, y, heading, speed = self._ego_scalars(ego_state)
        path, clean_arc = self._clean_path(world_traj)
        if len(path) < 2:
            return self._degenerate_action(speed, float(target_speed))
        steer = _pure_pursuit_steer_norm(
            ego_x=x, ego_y=y, ego_heading=heading,
            path=path,
            cum_lengths=clean_arc,
            lookahead_m=AIM_DIST_BASE + AIM_DIST_GAIN_S * speed,
            wheelbase=self.wheelbase,
            max_steer_angle=self.max_steer_angle,
        )
        return steer, _speed_control(speed, float(target_speed))


class LQRTracker(TrajectoryTracker):
    """nuPlan/tuplan_garage LQR trajectory tracker.

    Threads the steering-angle integrator across ``compute`` calls; call
    :meth:`reset` at episode start.
    """

    def __init__(self, *, wheelbase: float = 2.8, max_steer_angle: float = 0.6,
                 sim_dt: float = 0.1) -> None:
        super().__init__(wheelbase=wheelbase, max_steer_angle=max_steer_angle,
                         sim_dt=sim_dt)
        # LQR tuning defaults come from LateralLQRConfig, which match
        # PDMConfig except for the retuned lateral-position weight documented
        # there.
        # Optional experiment overrides (defaults stay pinned to PDMConfig):
        #   NAVSAFE_LQR_Q_LATERAL="5,20,0"  NAVSAFE_LQR_R_LATERAL=1
        #   NAVSAFE_LQR_HORIZON=5
        q_env = os.environ.get("NAVSAFE_LQR_Q_LATERAL")
        r_env = os.environ.get("NAVSAFE_LQR_R_LATERAL")
        h_env = os.environ.get("NAVSAFE_LQR_HORIZON")
        overrides: Dict[str, Any] = {}
        if q_env:
            overrides["lqr_q_lateral"] = tuple(
                float(v) for v in q_env.split(","))
        if r_env:
            overrides["lqr_r_lateral"] = float(r_env)
        if h_env:
            overrides["lqr_tracking_horizon"] = int(h_env)
        self._cfg = LateralLQRConfig(
            sim_dt=sim_dt, wheelbase=wheelbase,
            max_steering_angle_rad=max_steer_angle, **overrides,
        )
        if overrides:
            sys.stderr.write(f"[LQRTracker] overrides: {overrides}\n")
        self._steering_angle = 0.0

    @property
    def speed_lookahead_steps(self) -> int:
        return int(self._cfg.lqr_tracking_horizon)

    def _longitudinal_control(
        self, current_speed: float, reference_speed: float,
    ) -> Tuple[float, np.ndarray]:
        """Return normalized acceleration and LQR linearization velocities.

        This is the scalar longitudinal subsystem from nuPlan's
        ``LQRTracker``.  With constant acceleration over ``N`` steps,
        ``v_N = v_0 + N*dt*a``; solving its one-step quadratic objective
        gives the closed form below.  nuPlan passes the resulting constant-
        acceleration velocity profile into the lateral LQR, so do the same.
        """
        current = float(current_speed)
        reference = max(0.0, float(reference_speed))
        stopping = (current <= self._cfg.lqr_stopping_velocity
                    and reference <= self._cfg.lqr_stopping_velocity)
        if stopping:
            physical_accel = -LQR_STOPPING_GAIN * (current - reference)
        else:
            horizon_s = self.speed_lookahead_steps * self.sim_dt
            denom = (LQR_Q_LONGITUDINAL * horizon_s * horizon_s
                     + LQR_R_LONGITUDINAL)
            physical_accel = (
                LQR_Q_LONGITUDINAL * horizon_s * (reference - current)
                / denom
            )

        # EgoDynamics accepts normalized asymmetric throttle/brake, whereas
        # nuPlan's tracker emits physical acceleration in m/s^2.
        max_accel = 3.0
        max_brake = 5.0
        scale = max_accel if physical_accel >= 0.0 else max_brake
        accel_norm = float(np.clip(physical_accel / scale, -1.0, 1.0))

        # Equivalent to nuPlan's
        # _generate_profile_from_initial_condition_and_derivatives(... )[:N].
        steps = np.arange(self.speed_lookahead_steps, dtype=np.float64)
        velocities = np.maximum(
            0.0, current + steps * self.sim_dt * physical_accel)
        return accel_norm, velocities

    def reset(self) -> None:
        self._steering_angle = 0.0

    def compute(self, ego_state: Dict[str, Any], world_traj: np.ndarray,
                target_speed: float) -> Tuple[float, float]:
        x, y, heading, speed = self._ego_scalars(ego_state)
        path, clean_arc = self._clean_path(world_traj)
        if len(path) < 2:
            return self._degenerate_action(speed, float(target_speed))
        accel, velocity_profile = self._longitudinal_control(
            speed, float(target_speed))
        seg = np.diff(path, axis=0)
        seg_lengths = np.linalg.norm(seg, axis=1)
        seg_lengths_safe = np.where(seg_lengths > 1e-12, seg_lengths, 1.0)
        tangents = seg / seg_lengths_safe[:, None]
        # The steering-angle integrator persists across replans by design:
        # it models the physical steering state of the vehicle, which is
        # continuous when the reference plan changes.
        steer, self._steering_angle = _lqr_steer(
            ego_x=x, ego_y=y, ego_heading=heading, ego_speed=speed,
            ego_steering_angle=self._steering_angle,
            path=path, seg=seg, seg_lengths=seg_lengths,
            seg_lengths_safe=seg_lengths_safe,
            cum_lengths=clean_arc,
            tangents=tangents, cfg=self._cfg,
            velocity_profile=velocity_profile,
        )
        return float(steer), accel


def create_tracker(controller_type: str, *, wheelbase: float = 2.8,
                   max_steer_angle: float = 0.6,
                   sim_dt: float = 0.1) -> TrajectoryTracker:
    """Factory keyed by the evaluation config's ``controller_type``."""
    kwargs = dict(wheelbase=wheelbase, max_steer_angle=max_steer_angle,
                  sim_dt=sim_dt)
    if controller_type in ("pure_pursuit", "pid"):
        return PurePursuitTracker(**kwargs)
    if controller_type == "lqr":
        return LQRTracker(**kwargs)
    raise ValueError(
        f"Unknown controller_type '{controller_type}' "
        "(expected 'pure_pursuit', 'pid', or 'lqr')")


__all__ = [
    "TrajectoryTracker",
    "PurePursuitTracker",
    "LQRTracker",
    "SPEED_BRAKE_KP",
    "create_tracker",
]
