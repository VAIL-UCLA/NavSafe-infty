# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Actuator-filtered ego dynamics for PDM-Closed proposal simulation.

This module is **PDM-Closed-private**. It exists to make the planner's
proposal forward-simulation byte-closer to ``tuplan_garage`` /
``CaRL``'s ``BatchKinematicBicycleModel`` without modifying the
shared :class:`navsafe.core.ego_dynamics.EgoDynamics` (which is also
consumed by the env loop, the smoke-test scripts, and the docs
example).

What this adds
--------------

A first-order low-pass filter on the *commanded* actuator inputs
(acceleration and steering angle), exactly mirroring CaRL's
``_update_commands``::

    updated_accel    = dt / (dt + tau_a) * (cmd_accel    - filtered_accel)    + filtered_accel
    updated_steer    = dt / (dt + tau_s) * (cmd_steer    - filtered_steer)    + filtered_steer

Defaults (``tau_a = 0.2``, ``tau_s = 0.05``) match
``BatchKinematicBicycleModel.__init__`` line by line. The filter is
opt-in via :class:`PDMConfig`'s
``actuator_accel_time_constant_s`` / ``actuator_steering_time_constant_s``
knobs and is *only* used inside the planner's per-proposal
forward-sim — the env loop, training envs, sensor adapters, and any
other consumer of :class:`EgoDynamics` are completely untouched.

Why composition, not subclassing
--------------------------------

An earlier draft attempted to add the filter as a flag on
:class:`EgoDynamicsCfg` defaulting to ``0.0``. Even backward-compatible,
that change would have crossed a module boundary — modifying a
core/runtime class to satisfy a planner-internal need. The
composition pattern keeps the change strictly inside
``navsafe.policy.state.pdm_closed_planner`` and is therefore
guaranteed not to affect any other model adapter or env path.

Convention & units
------------------

:class:`EgoDynamics` ``step`` takes *normalised* inputs in ``[-1, 1]``
where ``negative steer = left`` and ``negative accel = brake``. The
filter operates on *physical* values (m/s² for accel, rad for steer
angle) — that's how upstream operates — so this wrapper:

1. Converts the normalised command to physical units using the same
   linear maps :class:`EgoDynamics` uses internally (
   ``physical_accel = norm * max_accel`` if ``norm >= 0``, else
   ``norm * max_brake``; ``physical_steer = -norm * max_steer_angle``).
2. Applies the per-channel low-pass filter on the physical values.
3. Converts the filtered physical values back to normalised inputs
   and forwards to :meth:`EgoDynamics.step` unchanged.

That last step is the price we pay for keeping :class:`EgoDynamics`
untouched: the round-trip through normalisation could in principle
introduce a sign-convention bug. We guard against that with the
unit tests in
``tests/policy/test_pdm_closed_actuator_filter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg


# ----------------------------------------------------------------------
# Default time constants — pinned to ``CaRL/.../batch_kinematic_bicycle.py``
# ----------------------------------------------------------------------

#: Default acceleration low-pass filter time constant (seconds).
#:
#: Reference: ``carl_nuplan/.../batch_kinematic_bicycle.py``
#: ``BatchKinematicBicycleModel.__init__`` (``accel_time_constant=0.2``).
DEFAULT_ACCEL_TIME_CONSTANT_S: float = 0.2

#: Default steering-angle low-pass filter time constant (seconds).
#:
#: Reference: ``carl_nuplan/.../batch_kinematic_bicycle.py``
#: ``BatchKinematicBicycleModel.__init__``
#: (``steering_angle_time_constant=0.05``).
DEFAULT_STEERING_TIME_CONSTANT_S: float = 0.05


@dataclass(frozen=True)
class ActuatorFilterParams:
    """Tunables for the per-channel first-order low-pass filter.

    Attributes:
        accel_time_constant_s: Time constant ``tau_a`` (s) for the
            acceleration channel. ``0.0`` disables filtering on
            acceleration. The faithful default is
            :data:`DEFAULT_ACCEL_TIME_CONSTANT_S` (= 0.2 s).
        steering_time_constant_s: Time constant ``tau_s`` (s) for the
            steering-angle channel. ``0.0`` disables filtering on
            steering. The faithful default is
            :data:`DEFAULT_STEERING_TIME_CONSTANT_S` (= 0.05 s).
    """

    accel_time_constant_s: float = DEFAULT_ACCEL_TIME_CONSTANT_S
    steering_time_constant_s: float = DEFAULT_STEERING_TIME_CONSTANT_S

    def __post_init__(self) -> None:
        if self.accel_time_constant_s < 0.0:
            raise ValueError(
                "accel_time_constant_s must be non-negative, "
                f"got {self.accel_time_constant_s}"
            )
        if self.steering_time_constant_s < 0.0:
            raise ValueError(
                "steering_time_constant_s must be non-negative, "
                f"got {self.steering_time_constant_s}"
            )

    @property
    def is_no_op(self) -> bool:
        """``True`` when both time constants are zero (filter is a pass-through)."""
        return (
            self.accel_time_constant_s == 0.0
            and self.steering_time_constant_s == 0.0
        )


class FilteredEgoDynamics:
    """Composition wrapper around :class:`EgoDynamics` that adds a low-pass
    filter on the *commanded* actuator inputs.

    Public surface mirrors :class:`EgoDynamics` so callers can use this
    interchangeably:

    * :meth:`reset` — re-initialises ego pose AND filter state.
    * :meth:`step` — applies the filter, then forwards to
      :meth:`EgoDynamics.step`.
    * :meth:`get_state` — proxies to the wrapped instance.
    * Read-only attribute proxies for ``x``, ``y``, ``heading``,
      ``speed``, ``frame``, ``trajectory``, ``cfg`` so existing
      forward-sim code that reads ``ego.x`` etc. continues to work.

    The wrapper tracks two pieces of internal state (the filtered
    physical accel and the filtered physical steering angle), updated
    each :meth:`step` call. :meth:`reset` seeds both from the live ego
    state, matching CaRL's initial 11-channel simulator state.

    Thread-safety: a single instance is **not** thread-safe. Each
    parallel proposal forward-simulation must use its own instance.

    Example:
        >>> from navsafe.core.ego_dynamics import EgoDynamicsCfg
        >>> ego = FilteredEgoDynamics(EgoDynamicsCfg(dt=0.1))
        >>> ego.reset(x=0.0, y=0.0, heading=0.0, speed=5.0)
        >>> for _ in range(10):
        ...     state = ego.step(steer=0.0, accel=1.0)
        >>> # The first few accel values are below 1.0 * max_accel
        >>> # because of the low-pass filter ramp; without the filter
        >>> # the response would be instant.
    """

    def __init__(
        self,
        cfg: EgoDynamicsCfg | None = None,
        params: ActuatorFilterParams | None = None,
    ) -> None:
        """Construct a filter-wrapped ego dynamics instance.

        Args:
            cfg: Bicycle-model configuration. Defaults to a
                :class:`EgoDynamicsCfg` with default values (which
                differ from PDM-Closed defaults — pass an explicit
                ``cfg`` from the planner).
            params: Filter time constants. Defaults to the
                ``CaRL``-pinned constants (0.2 s on accel, 0.05 s on
                steering).
        """
        self._inner = EgoDynamics(cfg)
        self._params = params if params is not None else ActuatorFilterParams()

        # Filter state (physical units). ``_filtered_accel_phys`` is in
        # m/s² and ``_filtered_steer_phys`` is in rad. ``reset`` seeds
        # them from the live ego state.
        self._filtered_accel_phys: float = 0.0
        self._filtered_steer_phys: float = 0.0

    # ------------------------------------------------------------------
    # Public API — mirror EgoDynamics
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> EgoDynamicsCfg:
        """Bicycle-model configuration (read-only proxy to inner instance)."""
        return self._inner.cfg

    @property
    def x(self) -> float:
        return self._inner.x

    @property
    def y(self) -> float:
        return self._inner.y

    @property
    def heading(self) -> float:
        return self._inner.heading

    @property
    def speed(self) -> float:
        return self._inner.speed

    @property
    def frame(self) -> int:
        return self._inner.frame

    @property
    def trajectory(self) -> list:
        return self._inner.trajectory

    @property
    def filter_params(self) -> ActuatorFilterParams:
        """The :class:`ActuatorFilterParams` this wrapper was constructed with."""
        return self._params

    @property
    def filtered_accel_phys(self) -> float:
        """Current filtered acceleration in m/s² (mainly for tests / diagnostics)."""
        return self._filtered_accel_phys

    @property
    def filtered_steering_phys(self) -> float:
        """Current filtered steering angle in rad (mainly for tests / diagnostics)."""
        return self._filtered_steer_phys

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        speed: float = 0.0,
        *,
        acceleration: float = 0.0,
        steering_angle: float = 0.0,
    ) -> None:
        """Reset ego pose and seed the physical actuator state.

        The filter state is a hidden integrator; failing to reset it
        between proposal forward-sim runs would leak commanded-input
        history from one proposal into the next, breaking
        determinism. The PDM-Closed forward-sim path therefore
        constructs a fresh :class:`FilteredEgoDynamics` per proposal.
        Reuse is also safe because :meth:`reset` overwrites both states
        with the new live initial condition.
        """
        self._inner.reset(x=x, y=y, heading=heading, speed=speed)
        self._filtered_accel_phys = float(acceleration)
        self._filtered_steer_phys = float(steering_angle)

    def get_state(self) -> Dict[str, Any]:
        """Return the wrapped ego's current state dict."""
        return self._inner.get_state()

    def mirror_pose(self, *, x: float, y: float, heading: float,
                    speed: float) -> None:
        """Overwrite the wrapped pose without touching the filter state.

        Used when another plant (the planner's reference rear-axle model)
        propagates the vehicle and this object only mirrors its
        centre-referenced pose for readers of ``x``/``y``/``heading``/``speed``.
        """
        self._inner.x = float(x)
        self._inner.y = float(y)
        self._inner.heading = float(heading)
        self._inner.speed = float(speed)

    # ------------------------------------------------------------------
    # Step — the only behaviour change vs EgoDynamics
    # ------------------------------------------------------------------

    def step(self, steer: float, accel: float) -> Dict[str, Any]:
        """Advance one timestep with low-pass-filtered actuator inputs.

        Args:
            steer: Normalised steering input ``[-1, 1]``. Convention
                matches :class:`EgoDynamics`: ``negative = left``,
                ``positive = right``.
            accel: Normalised throttle/brake input ``[-1, 1]``.
                Convention matches :class:`EgoDynamics`:
                ``positive = throttle``, ``negative = brake``.

        Returns:
            Same dict :meth:`EgoDynamics.step` returns.
        """
        # Acceleration is clipped to the normalised range before the filter
        # (the caps are the interface's own, see ``PDMConfig.max_accel``).
        # The steering COMMAND is deliberately NOT clipped here: the
        # reference (``BatchKinematicBicycleModel``) low-pass filters the
        # unclipped ideal angle (``_update_commands``) and clips the
        # integrated angle afterwards (``propagate_state``:
        # ``np.clip(forward_integrate(...), -max, max)``). Clipping the
        # command first made the filtered angle approach the cap more
        # slowly than the reference whenever the LQR asked for more than
        # the cap (0.998 vs 1.047 rad from 0.9 rad at one saturated step).
        steer_norm = float(steer)
        accel_norm = float(np.clip(accel, -1.0, 1.0))

        cfg = self._inner.cfg
        dt = float(cfg.dt)

        # Convert normalised → physical (matches EgoDynamics' internal
        # mapping).
        steer_phys_cmd = -steer_norm * float(cfg.max_steer_angle)
        if accel_norm >= 0.0:
            accel_phys_cmd = accel_norm * float(cfg.max_accel)
        else:
            accel_phys_cmd = accel_norm * float(cfg.max_brake)

        # Per-channel first-order low-pass filter:
        #   filtered <- alpha * cmd + (1 - alpha) * filtered
        # with alpha = dt / (dt + tau). When tau == 0 we degenerate to
        # alpha == 1 (instant response, identical to no filter).
        tau_a = float(self._params.accel_time_constant_s)
        if tau_a > 0.0:
            alpha_a = dt / (dt + tau_a)
            self._filtered_accel_phys = (
                alpha_a * accel_phys_cmd
                + (1.0 - alpha_a) * self._filtered_accel_phys
            )
        else:
            self._filtered_accel_phys = accel_phys_cmd

        # Reference phase order (``batch_kinematic_bicycle.propagate_state``):
        # ``_update_commands`` advances only STEERING_RATE, and
        # ``get_state_dot`` reads the *un-updated* STEERING_ANGLE, so the
        # pose propagates under the PREVIOUS steering state and the updated
        # angle takes effect one step later. Acceleration is the opposite —
        # the updated value drives ``v_dot`` in the same step. Feeding the
        # freshly filtered angle into ``step`` here gave the scored plant a
        # one-step (dt) steering phase lead over the reference.
        steer_phys_used = self._filtered_steer_phys
        tau_s = float(self._params.steering_time_constant_s)
        if tau_s > 0.0:
            alpha_s = dt / (dt + tau_s)
            self._filtered_steer_phys = (
                alpha_s * steer_phys_cmd
                + (1.0 - alpha_s) * self._filtered_steer_phys
            )
        else:
            self._filtered_steer_phys = steer_phys_cmd
        # Reference order: filter, THEN clip the integrated angle to the
        # physical bound (see the note above).
        self._filtered_steer_phys = float(np.clip(
            self._filtered_steer_phys,
            -float(cfg.max_steer_angle), float(cfg.max_steer_angle)))

        # Convert filtered physical values back to normalised inputs
        # for EgoDynamics. Use the same physical→normalised maps that
        # are the *inverse* of the conversion above. The inverse of
        # the brake/accel split depends on the sign of the filtered
        # value, not the original command — important when the filter
        # is in transition (e.g. cmd switches from accel to brake).
        if self._filtered_accel_phys >= 0.0:
            accel_norm_filtered = self._filtered_accel_phys / float(cfg.max_accel)
        else:
            accel_norm_filtered = self._filtered_accel_phys / float(cfg.max_brake)
        steer_norm_filtered = -steer_phys_used / float(cfg.max_steer_angle)

        # EgoDynamics will re-clip these to [-1, 1] internally, which
        # makes the wrapper safe even when the time constant is so
        # small that the filter overshoots transiently.
        return self._inner.step(
            steer=float(steer_norm_filtered),
            accel=float(accel_norm_filtered),
        )


__all__ = [
    "DEFAULT_ACCEL_TIME_CONSTANT_S",
    "DEFAULT_STEERING_TIME_CONSTANT_S",
    "ActuatorFilterParams",
    "FilteredEgoDynamics",
]
