# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""IDM policy bank for PDM-Closed proposal generation.

Faithful to Dauner et al., CoRL 2023 ("Parting with Misconceptions
about Learning-based Vehicle Motion Planning"). The reference
``tuplan_garage`` implementation parameterises proposals as 5
*longitudinal* policies (target velocity = ``speed_limit_fraction *
lane_speed_limit``) × 3 *lateral* offsets (``±1 m`` and ``0 m``).

Key facts pinned by this module:

* Five named policies — ``P_20``, ``P_40``, ``P_60``, ``P_80``,
  ``P_100`` — each parameterised by a fraction of the current lane's
  speed limit.
* The IDM acceleration exponent **δ is 10**, matching
  ``tuplan_garage``'s hardcoded value
  (``batch_idm_policy.py``, line 156). The original IDM paper used δ=4;
  the CoRL 2023 implementation found δ=10 produced sharper saturation
  near target velocity and used that constant in the winning planner.
* The IDM kernel is implemented inline (``idm_accel``) rather than
  delegating to :class:`navsafe.component.traffic_agent.idm.IDMActor`
  because the traffic-agent path uses δ=4 (per the IDM paper) and the
  two callers must not share a δ value. This is enforced by
  ``tests/policy/test_pdm_closed_idm_delta.py`` (regression guard).

Mapping to the reference (`tuplan_garage` Hydra config
``pdm_closed_planner.yaml``)::

    speed_limit_fraction: [0.2, 0.4, 0.6, 0.8, 1.0]   # 5 policies
    fallback_target_velocity: 15.0                     # m/s
    min_gap_to_lead_agent: 1.0                         # m
    headway_time: 1.5                                  # s
    accel_max: 1.5                                     # m/s²
    decel_max: 3.0                                     # m/s²

These are the exact numerical values the 2023 nuPlan winning planner
shipped with. They are exposed as keyword arguments on
:func:`build_default_idm_bank` so a downstream user can swap them
without forking this module.

Where the speed limit comes from
-------------------------------

``fallback_target_velocity`` is a *last* resort, not the normal path. The
planner resolves the active lane's limit from the scenario
(``planner._active_lane_speed_limit_mps`` → ``_speed_limit_to_mps``), and
:mod:`navsafe.scenario.speed_target` guarantees every lane of a py123d
scenario carries one — the dataset's own where the source map posts it,
otherwise a per-lane target inferred from the log's observed speeds and
recorded in ``metadata['speed_target']``.

Backward-compatibility shim: the legacy ``IDM_AGGRESSIVE`` /
``IDM_NOMINAL`` / ``IDM_CONSERVATIVE`` symbols are retained as
aliases for ``P_100`` / ``P_60`` / ``P_20`` (fastest / median /
slowest) so any caller that built a custom ``PDMConfig`` keyed on
those names continues to import. The names are documented as
deprecated and tests should migrate to ``P_*`` instead."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# IDM constants — MUST match tuplan_garage/batch_idm_policy.py line 156.
# Changing this is a behavioral break; the regression test
# ``test_idm_acceleration_exponent_is_ten`` asserts the value.
# ----------------------------------------------------------------------

ACCELERATION_EXPONENT: float = 10.0
"""IDM acceleration exponent δ used by PDM-Closed.

The original IDM paper (Treiber, Hennecke, Helbing 2000) uses δ=4. The
CoRL 2023 PDM-Closed implementation in ``tuplan_garage`` hardcodes δ=10
in ``batch_idm_policy.py`` (no Hydra knob, no public API). Faithful
reproduction therefore pins δ=10 here. The IDM traffic agent
(:class:`navsafe.component.traffic_agent.idm.IDMActor`) keeps δ=4 for
its own per-vehicle dynamics — the two paths are intentionally
independent.
"""


# ----------------------------------------------------------------------
# Policy dataclass
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IDMPolicy:
    """A named IDM policy used to generate one proposal speed profile.

    Faithful to ``tuplan_garage``'s :class:`BatchIDMPolicy`. Each
    policy stores a *speed-limit fraction* and a *fallback target
    velocity*; the actual target velocity at run time is
    ``fraction * lane_speed_limit_mps`` when the lane carries a
    speed-limit annotation, or ``fraction * fallback`` otherwise.

    Attributes:
        name: Human-readable identifier — used in logs and proposal
            metadata. Convention: ``"P_<percent>"``, e.g. ``P_60``.
        speed_limit_fraction: Fraction of the lane speed limit to
            target. Must be in ``(0, 1]`` for the standard bank.
        fallback_target_velocity: Free-traffic fallback velocity in
            m/s, used when the lane has no speed-limit annotation.
            Default: 15 m/s (matches ``tuplan_garage``).
        headway: Desired time headway ``T`` (s).
        min_gap: Minimum bumper-to-bumper gap ``s0`` (m).
        accel: Maximum comfortable acceleration ``a`` (m/s²).
        decel: Comfortable deceleration ``b`` (m/s²) — positive value.
        delta: Acceleration exponent. Pinned to
            :data:`ACCELERATION_EXPONENT` (= 10) for faithful
            reproduction; exposed as a field so tests can override it
            without monkey-patching.
    """

    name: str
    speed_limit_fraction: float
    fallback_target_velocity: float = 15.0
    headway: float = 1.5
    min_gap: float = 1.0
    accel: float = 1.5
    decel: float = 3.0
    delta: float = ACCELERATION_EXPONENT

    def __post_init__(self) -> None:
        if self.speed_limit_fraction <= 0.0:
            raise ValueError(
                f"speed_limit_fraction must be positive, got {self.speed_limit_fraction}"
            )
        if self.fallback_target_velocity <= 0.0:
            raise ValueError(
                f"fallback_target_velocity must be positive, got {self.fallback_target_velocity}"
            )
        if self.headway < 0.0:
            raise ValueError(f"headway must be non-negative, got {self.headway}")
        if self.min_gap < 0.0:
            raise ValueError(f"min_gap must be non-negative, got {self.min_gap}")
        if self.accel <= 0.0:
            raise ValueError(f"accel must be positive, got {self.accel}")
        if self.decel <= 0.0:
            raise ValueError(f"decel must be positive, got {self.decel}")
        if self.delta <= 0.0:
            raise ValueError(f"delta must be positive, got {self.delta}")

    # ------------------------------------------------------------------
    # Speed-limit resolution
    # ------------------------------------------------------------------

    def target_velocity_for(self, speed_limit_mps: float | None) -> float:
        """Resolve this policy's target velocity for a given lane speed limit.

        Args:
            speed_limit_mps: Lane speed limit in m/s, or ``None`` /
                non-positive when the lane has no annotation.

        Returns:
            ``fraction * speed_limit`` when the speed limit is positive
            and finite, else ``fraction * fallback``.

        The resolution rule mirrors ``tuplan_garage``'s
        ``BatchIDMPolicy.update`` exactly: when the speed limit is
        unknown the planner uses ``fraction * fallback``, *not* the
        fallback alone (so the 5 policies still differ when no speed
        limit is available — they degenerate to ``[3, 6, 9, 12, 15]``
        m/s with the default fallback of 15).
        """
        if (
            speed_limit_mps is not None
            and math.isfinite(speed_limit_mps)
            and speed_limit_mps > 0.0
        ):
            return float(self.speed_limit_fraction) * float(speed_limit_mps)
        return float(self.speed_limit_fraction) * float(self.fallback_target_velocity)

    # ------------------------------------------------------------------
    # Backward-compat: the legacy code referred to a single
    # ``target_speed`` field. Provide it as a property defaulting to
    # the no-speed-limit case.
    # ------------------------------------------------------------------

    @property
    def target_speed(self) -> float:
        """Default target speed when no lane speed limit is available."""
        return self.target_velocity_for(None)


# ----------------------------------------------------------------------
# Default 5-policy bank — matches tuplan_garage/pdm_closed_planner.yaml
# ----------------------------------------------------------------------


def build_default_idm_bank(
    *,
    fractions: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    fallback_target_velocity: float = 15.0,
    headway: float = 1.5,
    min_gap: float = 1.0,
    accel: float = 1.5,
    decel: float = 3.0,
    delta: float = ACCELERATION_EXPONENT,
) -> Tuple[IDMPolicy, ...]:
    """Construct the default 5-policy IDM bank.

    All keyword arguments default to the ``tuplan_garage`` Hydra config
    pinned in :mod:`navsafe.policy.state.pdm_closed_planner`. The bank
    is returned as a tuple so it can be used directly as the default
    for :attr:`PDMConfig.idm_policies`.
    """
    return tuple(
        IDMPolicy(
            name=f"P_{int(round(f * 100))}",
            speed_limit_fraction=float(f),
            fallback_target_velocity=fallback_target_velocity,
            headway=headway,
            min_gap=min_gap,
            accel=accel,
            decel=decel,
            delta=delta,
        )
        for f in fractions
    )


# Module-level instance so callers can ``from ... import DEFAULT_IDM_POLICIES``.
DEFAULT_IDM_POLICIES: Tuple[IDMPolicy, ...] = build_default_idm_bank()

# Convenience aliases for individual policies — covers downstream code
# that may want to reference one fraction directly.
P_20: IDMPolicy = DEFAULT_IDM_POLICIES[0]
P_40: IDMPolicy = DEFAULT_IDM_POLICIES[1]
P_60: IDMPolicy = DEFAULT_IDM_POLICIES[2]
P_80: IDMPolicy = DEFAULT_IDM_POLICIES[3]
P_100: IDMPolicy = DEFAULT_IDM_POLICIES[4]


# ----------------------------------------------------------------------
# Backward-compat aliases (deprecated; kept so external configs that
# imported the legacy "aggressive/nominal/conservative" names continue
# to import cleanly while we migrate tests)
# ----------------------------------------------------------------------

IDM_AGGRESSIVE: IDMPolicy = P_100
"""Deprecated alias for :data:`P_100` (highest target speed)."""

IDM_NOMINAL: IDMPolicy = P_60
"""Deprecated alias for :data:`P_60` (median target speed)."""

IDM_CONSERVATIVE: IDMPolicy = P_20
"""Deprecated alias for :data:`P_20` (lowest target speed)."""


# ----------------------------------------------------------------------
# IDM kernel — inline implementation, NOT delegating to IDMActor
# ----------------------------------------------------------------------


def idm_accel(
    speed: float,
    lead_dist: float | None,
    lead_speed: float,
    policy: IDMPolicy,
    *,
    speed_limit_mps: float | None = None,
) -> float:
    """Compute IDM acceleration for a single ``(speed, lead, policy)`` configuration.

    Implements the IDM formula directly so this module is independent
    of :class:`navsafe.component.traffic_agent.idm.IDMActor` (which
    must keep δ=4 for the traffic agent path).

    Formula (Treiber, Hennecke, Helbing 2000) with the
    ``tuplan_garage`` δ=10 modification, and the ``tuplan_garage``
    "no-max-on-s_star" variant::

        v0     = policy.target_velocity_for(speed_limit_mps)
        s_star = s0 + v * T + v * (v - v_lead) / (2 sqrt(a * b))
        a_idm  = a * (1 - (v / v0)**delta - (s_star / s_alpha)**2)

    where ``s_alpha = max(s - l_r_lead, s0)`` and ``s`` is the
    bumper-to-bumper gap. The exact form (clamping ``s_alpha`` from
    below by ``s0`` to avoid division by very small values; *no*
    ``max()`` on ``s_star``) matches the reference implementation
    byte-for-byte. The canonical IDM paper has ``max(0, ...)`` around
    the kinematic term in ``s_star``; CoRL 2023 PDM-Closed dropped
    that wrap so we drop it here too.

    Args:
        speed: Current ego speed (m/s), must be ``>= 0``.
        lead_dist: Bumper-to-bumper gap to the lead vehicle (m). Pass
            ``None`` (or any value > 1e9) to indicate no lead.
        lead_speed: Lead-vehicle speed (m/s) — projected onto the ego
            heading. Signed: negative for oncoming traffic, which
            enlarges the closing-rate term and brakes harder (matches
            upstream's signed ``_get_leading_agent_velocity``).
        policy: The IDM policy whose parameters drive the kernel.
        speed_limit_mps: Optional lane speed limit. When provided, the
            policy's target velocity becomes
            ``fraction * speed_limit_mps``; otherwise the fallback
            applies.

    Returns:
        Acceleration (m/s²). Can be negative (braking).

    Raises:
        ValueError: If ``speed`` is negative.
    """
    if speed < 0.0:
        raise ValueError(f"speed must be non-negative, got {speed}")

    target_velocity = policy.target_velocity_for(speed_limit_mps)
    if target_velocity <= 0.0:
        # Defensive: guard the (v / v0) ** delta division below.
        return -policy.decel

    v_ratio = speed / target_velocity
    free_road_term = 1.0 - v_ratio ** policy.delta

    if lead_dist is None or not np.isfinite(lead_dist) or lead_dist > 1e9:
        # Free-road branch.
        accel = policy.accel * free_road_term
    else:
        # Following branch.
        s = float(lead_dist)
        delta_v = float(speed) - float(lead_speed)  # closing rate (positive=approaching)
        interaction = (speed * delta_v) / (2.0 * math.sqrt(policy.accel * policy.decel))
        # Faithful to ``tuplan_garage/batch_idm_policy.py``: ``s_star``
        # is *not* clamped at zero. The canonical IDM paper uses
        # ``s0 + max(0, v*T + interaction)``, but the CoRL 2023
        # implementation drops the ``max()`` and lets ``s_star`` go
        # below ``s0`` (or even negative) when the lead is opening
        # up faster than the ego. The squared term ``(s_star/s_alpha)**2``
        # then still penalises positively, slightly reducing accel
        # in that regime — a known PDM-Closed property we must match.
        s_star = policy.min_gap + speed * policy.headway + interaction
        # Clamp s_alpha from below by min_gap to avoid numerical blow-up
        # at penetration. Matches tuplan_garage's `np.maximum(...)` line
        # on the gap (not on s_star).
        s_alpha = max(s, policy.min_gap)
        gap_term = (s_star / s_alpha) ** 2
        accel = policy.accel * (free_road_term - gap_term)

    # Reference implementation clips the acceleration into
    # ``[-decel, accel]``. Mirror that here so downstream forward-sim
    # never sees super-comfort decel/accel from the IDM kernel.
    return float(np.clip(accel, -policy.decel, policy.accel))


def idm_speed_profile(
    initial_speed: float,
    lead_dist: float | None,
    lead_speed: float,
    policy: IDMPolicy,
    horizon_s: float,
    dt: float,
    *,
    constant_lead: bool = True,
    max_speed: float | None = None,
    speed_limit_mps: float | None = None,
) -> np.ndarray:
    """Closed-form IDM speed rollout.

    Two prediction models for the lead vehicle are supported:

    * ``constant_lead=True`` (default): the lead's gap and speed are
      held constant at the initial values for the whole rollout.
      Useful as a cheap *seed* for the forward-simulation loop.
    * ``constant_lead=False``: the lead is propagated at constant
      velocity each step (the gap therefore evolves with the closing
      rate ``speed - lead_speed``).

    The faithful PDM-Closed forward simulation does *per-step gap
    refinement* (see :mod:`forward_sim`), so neither mode is the
    "right" answer for the planner; this function is retained as a
    primitive for unit tests and as a fallback when the per-step path
    is unavailable.

    Args:
        initial_speed: Ego speed at ``t=0`` (m/s).
        lead_dist: Bumper-to-bumper gap to the lead at ``t=0`` (m), or
            ``None`` for free road.
        lead_speed: Lead-vehicle speed at ``t=0`` (m/s).
        policy: IDM policy to integrate.
        horizon_s: Total rollout duration (s).
        dt: Integration step (s); typically the planner's ``sim_dt``.
        constant_lead: See above.
        max_speed: Optional hard cap on the resulting speeds (m/s).
            ``None`` = no cap.
        speed_limit_mps: Lane speed limit, forwarded to
            :func:`idm_accel`.

    Returns:
        Array of shape ``(num_steps + 1,)`` with the speed at each
        time ``0, dt, 2*dt, …, num_steps * dt``.
    """
    if horizon_s < 0.0:
        raise ValueError(f"horizon_s must be non-negative, got {horizon_s}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if initial_speed < 0.0:
        raise ValueError(f"initial_speed must be non-negative, got {initial_speed}")

    num_steps = int(round(horizon_s / dt))
    speeds = np.zeros(num_steps + 1, dtype=np.float64)
    speeds[0] = float(initial_speed)

    cap = float(max_speed) if max_speed is not None else float("inf")
    if cap < 0.0:
        raise ValueError(f"max_speed must be non-negative, got {max_speed}")

    cur_speed = float(initial_speed)
    cur_gap = (
        float(lead_dist)
        if lead_dist is not None and np.isfinite(lead_dist) and lead_dist <= 1e9
        else None
    )
    cur_lead_speed = float(lead_speed) if cur_gap is not None else 0.0

    for k in range(num_steps):
        a = idm_accel(
            cur_speed,
            cur_gap,
            cur_lead_speed,
            policy,
            speed_limit_mps=speed_limit_mps,
        )
        cur_speed = max(0.0, min(cap, cur_speed + a * dt))
        speeds[k + 1] = cur_speed

        if cur_gap is not None and not constant_lead:
            # Constant-velocity lead: gap evolves with closing rate.
            cur_gap = cur_gap - (cur_speed - cur_lead_speed) * dt
            if cur_gap < 0.0:
                cur_gap = 0.0

    return speeds


__all__ = [
    "ACCELERATION_EXPONENT",
    "IDMPolicy",
    "DEFAULT_IDM_POLICIES",
    "P_20",
    "P_40",
    "P_60",
    "P_80",
    "P_100",
    "IDM_AGGRESSIVE",
    "IDM_NOMINAL",
    "IDM_CONSERVATIVE",
    "build_default_idm_bank",
    "idm_accel",
    "idm_speed_profile",
]
