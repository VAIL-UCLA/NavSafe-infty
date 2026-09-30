# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""PDM-Closed configuration — every tunable knob lives in one dataclass.

Defaults match the ``tuplan_garage`` Hydra config that won the 2023
nuPlan planning challenge. Every value comes from
``tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml``:

* Upstream ``trajectory_sampling`` is ``80`` poses at ``0.1 s``.  The
  NexusSim/NavSafe execution adapter deliberately exposes ``8`` poses at
  ``0.5 s`` (a ``4 s`` horizon), while proposal simulation and scoring remain
  on the upstream ``0.1 s`` grid.
* ``proposal_sampling: num_poses=40, interval_length=0.1`` → ``4 s``
  proposal forward simulation at ``10 Hz``.
* ``lateral_offsets: [-1.0, 1.0]`` (the reference prepends ``0.0``
  internally → ``3`` lateral proposals).
* ``idm_policies.speed_limit_fraction: [0.2, 0.4, 0.6, 0.8, 1.0]`` →
  ``5`` longitudinal proposals.
* Vehicle dimensions: matching the EPDMS scorer's collision polygon
  and the MetaDrive/NexusSim default vehicle (length 4.515 m, width
  1.852 m, wheelbase 2.7 m — smaller than nuPlan's Pacifica).
* Emergency brake decel ``4.05 m/s²`` and infraction horizon ``2.0 s``
  match ``tuplan_garage/utils/pdm_emergency_brake.py``.

Output contract: NexusSim adapters consume ``(N, 2)`` ego-frame
``[lateral, forward]`` waypoints. Proposals are sampled for 4 s at 10 Hz;
the campaign adapter returns eight 0.5 s-spaced poses over that same 4 s.
This output-horizon/downsampling difference from CaRL is intentional and
must be disclosed rather than described as upstream-identical.

Consequence when comparing scores: with ``output_dt=0.5`` the executor's
reference for the first four 0.1 s steps of each replan window is
BACKWARD-EXTRAPOLATED from the 0.5 s and 1.0 s waypoints
(``core/plan_execution.py``), so it is not a planner-computed pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M


# ----------------------------------------------------------------------
# Defaults pinned to the ``tuplan_garage`` Hydra config
# ----------------------------------------------------------------------

# Lateral offsets (m). The reference YAML lists ``[-1.0, 1.0]`` and the
# proposal manager prepends ``0.0`` internally; we list all three
# explicitly so :attr:`PDMConfig.lateral_offsets` is the single source
# of truth.
DEFAULT_LATERAL_OFFSETS: Tuple[float, ...] = (0.0, -1.0, 1.0)

# Vehicle dimensions (m) — must match the EPDMS scorer's
# ``VEHICLE_LENGTH`` / ``VEHICLE_WIDTH``. Note these are the
# MetaDrive/NexusSim default-vehicle dims, NOT nuPlan's Pacifica
# (5.176 x 2.297 m, wheelbase 3.089 m) — they must match the vehicle
# the simulator actually spawns, and a regression in collision
# geometry is a scoring-validity bug, not a planner-only bug.
VEHICLE_LENGTH: float = EGO_LENGTH_M
VEHICLE_WIDTH: float = EGO_WIDTH_M
# One wheelbase for the reference plant AND the LQR tracker of the scored
# copy. Note the reference code freezes its TRACKER wheelbase at Pacifica's
# 3.089 m (``BatchLQRTracker.__init__``; ``PDMSimulator`` updates only the
# motion model's vehicle), which is a no-op on nuPlan's own Pacifica but
# would put a 3.089 m tracker over a 2.7 m plant here. The self-consistent
# reading — tracker == plant — is what the reference effectively runs.
VEHICLE_WHEELBASE: float = 2.7


@dataclass(frozen=True)
class PDMConfig:
    """Static configuration for the PDM-Closed planner.

    Frozen so the same config can be safely shared by parallel rollouts
    (e.g. CaRL-style PPO post-training) without accidental mutation.

    Attributes:
        horizon_s: Forward-simulation horizon for proposals in seconds.
            Matches ``tuplan_garage``'s ``proposal_sampling`` (4 s).
        sim_dt: Inner-loop simulation timestep in seconds. Matches
            ``tuplan_garage``'s ``interval_length=0.1``.
        output_dt: Spacing between the output waypoints in seconds.
            ``0.5 s`` matches every other NexusSim adapter.
        num_output_poses: Number of output waypoints (8 by NexusSim
            convention; 16 gives an 8 s adapter horizon but retains 0.5 s
            spacing rather than CaRL's 10 Hz output contract).
        trajectory_horizon_s: Total horizon of the *output* trajectory
            in seconds. Defaults to :attr:`horizon_s` (4 s) to keep the
            adapter contract unchanged. Set to ``8.0`` for full
            ``tuplan_garage`` horizon parity (the selected proposal is
            extended by re-running its IDM policy on the same path).
        num_trajectory_poses: Number of output waypoints. Must satisfy
            ``num_trajectory_poses * output_dt == trajectory_horizon_s``.
        lateral_offsets: Lateral offsets (m). Default
            :data:`DEFAULT_LATERAL_OFFSETS` (3 values).
        idm_policies: IDM policies that produce the longitudinal
            proposal speeds. Default = 5 policies, one per
            ``speed_limit_fraction`` in
            ``[0.2, 0.4, 0.6, 0.8, 1.0]``.
        agent_prediction_horizon_s: How far forward to predict other
            agents with constant-velocity rollout. Must be ≥
            :attr:`trajectory_horizon_s`.
        infraction_horizon_s: Time-to-collision threshold for the
            emergency-brake fallback. ``2.0 s`` matches
            ``tuplan_garage``'s ``PDMEmergencyBrake``.
        emergency_brake_threshold: The ``no_valid_proposal`` arm of the
            *secondary* (score-based) brake guard — fires when the best
            proposal scores at or below this threshold. Default ``0.0``
            = "every proposal is fully disqualified by the
            multiplicative metrics".

            It is **not** the whole guard — no scalar can be, and
            :func:`~navsafe.policy.state.pdm_closed_planner.planner.
            score_brake_reason` is the single home of that argument and
            of the second, scale-free arm (``stopping_scores_better``).
            The two arms are OR-ed, so this value can never disable the
            guard; it is kept rather than deleted because it is the only
            arm covering a state where the stop itself scores ``0``.

            Must be non-negative: scores are ``>= 0``, so a negative
            value makes this arm unreachable — the one failure mode
            nothing downstream would surface. No upper bound: ``>= 1.0``
            (tests use 1e9) deliberately forces the score brake whenever
            the ego is below the speed gate, and its effect is loud.
            Overridable per-process with
            ``NEXUSSIM_EMERGENCY_BRAKE_THRESHOLD`` (see
            :func:`pdm_config_from_env`), which is what pins a teacher
            across a default change.
        emergency_brake_decel: Deceleration (m/s²) used by the
            emergency-brake fallback. ``4.05`` matches
            ``tuplan_garage``'s ``PDMEmergencyBrake``.
        wheelbase / ego_length / ego_width: Vehicle geometry.
        max_speed: Hard upper bound on speed during forward
            simulation.
        max_steering_angle_rad: Maximum steering angle (rad).
        max_accel / max_brake: Normalisation scale of the proposal
            forward-sim plant's throttle/brake. Deliberately far above any
            tracker command (the reference plant is unsaturated); the
            execution plant keeps its own 3 / 5 m/s² caps.
        traffic_light_obstacles: Brake for red lights on the route by
            placing a stationary obstacle on each red connector's stop
            line, as upstream's observation does (default ``True``; the
            light states are the scenario's logged, privileged ones).
        tracker: ``"lqr"`` (faithful default) or ``"pure_pursuit"``
            (legacy / opt-in). The proposal forward-simulation uses
            this to track the offset path; the env's outer controller
            is independent.
        controller_lookahead_min_m / controller_lookahead_gain_s:
            Pure-pursuit lookahead (used only when
            ``tracker == "pure_pursuit"``).
        lqr_q_lateral: Q-matrix diagonal for the LQR lateral
            subsystem (lateral_error, heading_error, steering_angle).
            Default ``[1.0, 10.0, 0.0]`` from
            ``tuplan_garage/batch_lqr.py``.
        lqr_r_lateral: R-matrix scalar for the LQR lateral subsystem
            (steering_rate). Default ``1.0``.
        lqr_q_longitudinal: Q-matrix scalar for the LQR longitudinal
            subsystem (velocity). Default ``10.0``.
        lqr_r_longitudinal: R-matrix scalar for the LQR longitudinal
            subsystem (acceleration). Default ``1.0``.
        lqr_tracking_horizon: How many discrete time steps ahead the
            LQR considers. Default ``10``.
        lqr_stopping_velocity: Velocity (m/s) below which the LQR
            falls back to a P controller. Default ``0.2``.
        lqr_stopping_proportional_gain: Proportional gain for the
            stopping P controller. Default ``0.5``.
        lqr_jerk_penalty: Jerk smoothing weight for velocity profile.
            Default ``1e-4``.
        lqr_curvature_rate_penalty: Curvature-rate smoothing weight
            for the curvature profile. Default ``1e-2``.
        route_horizon_s: Horizon (s) of the route hint when route
            extraction depends on the SDC's ground-truth future.
        route_densify_spacing_m: Target spacing between centerline
            vertices after densification.
        route_source: Default route extraction strategy.
            ``"lane_graph_route"`` (default) — the successor-graph walk
            conditioned on ``metadata['route_lane_ids']`` (branch intent
            derived from the log at conversion, the analog of upstream's
            ``route_roadblock_ids``); degrades to ``"lane_graph_search"``
            when a scenario carries no route chain.
            ``"gt_future"`` (previous default) fits lanes to the SDC's
            ground-truth future — an oracle leak whose coverage gate
            rejects the hint on most closed-loop frames, silently
            degrading to an UNANCHORED walk that re-picks the mission
            every replan: measured flipping a route to a U-turn
            mid-episode (head-on collision), swapping fork arms between
            otherwise-identical runs, and abandoning a logged turn.
            Keep it only for open-loop navhard scoring.
            ``"lane_graph_search"`` (faithful, no route conditioning);
            ``"nearest"`` (legacy fallback).
        map_radius_m: Maximum distance from the ego at which lanes /
            agents are considered. Matches
            ``tuplan_garage``'s ``map_radius`` (50 m).
    """

    # --- Time-stepping ----------------------------------------------------
    horizon_s: float = 4.0
    sim_dt: float = 0.1
    # CaRL-derived NexusSim adapter contract used by the original NavSafe
    # baseline: proposals are still simulated/scored at ``sim_dt=0.1``, but
    # the trajectory exposed to the outer controller is sampled every 0.5 s.
    output_dt: float = 0.5
    num_output_poses: int = 8

    # --- Output trajectory shape (separate from proposal horizon) ---------
    trajectory_horizon_s: float = 4.0
    num_trajectory_poses: int = 8

    # --- Proposal grid ----------------------------------------------------
    lateral_offsets: Tuple[float, ...] = DEFAULT_LATERAL_OFFSETS
    idm_policies: Tuple = field(default=())  # populated in __post_init__

    # --- Other-agent prediction ------------------------------------------
    # Must be >= trajectory_horizon_s (8 s for upstream horizon parity).
    agent_prediction_horizon_s: float = 8.0

    # --- Lead-agent update cadence (faithful to CaRL) --------------------
    # CaRL's ``PDMGenerator`` updates the per-proposal leading-agent
    # state every ``leading_agent_update_rate`` sim steps (default 2 →
    # update at 5 Hz when sim_dt=0.1). Between updates, the previous
    # step's leading-agent state is reused. NexusSim's forward-sim
    # follows the same cadence so per-step IDM behaviour matches.
    # Reference: ``carl_nuplan/.../proposal/pdm_generator.py`` line 46
    # (``leading_agent_update_rate: int = 2``).
    leading_agent_update_rate: int = 2

    # --- Emergency brake --------------------------------------------------
    infraction_horizon_s: float = 2.0
    emergency_brake_threshold: float = 0.0
    #: Score-based brake extensions are not part of upstream PDM-Closed.
    #: ``"none"`` is therefore the faithful default. ``"threshold_only"``
    #: and ``"both"`` retain the NexusSim safety experiments as explicit
    #: opt-ins. Env: ``NEXUSSIM_EMERGENCY_BRAKE_SCORE_ARMS``.
    emergency_brake_score_arms: str = "none"
    #: NexusSim-only red-light emergency floor. CaRL represents red
    #: connectors as stationary IDM leads but has no independent red-light
    #: brake guard, so the faithful default is disabled. Env:
    #: ``NEXUSSIM_RED_LIGHT_BRAKE_FLOOR``.
    red_light_brake_floor: bool = False
    emergency_brake_mode: str = "trajectory"
    emergency_brake_decel: float = 4.05  # m/s² — matches tuplan_garage
    emergency_brake_max_ego_speed: float = 5.0

    # --- Vehicle / dynamics ----------------------------------------------
    wheelbase: float = VEHICLE_WHEELBASE
    ego_length: float = VEHICLE_LENGTH
    ego_width: float = VEHICLE_WIDTH
    # Physical speed clamp for the proposal forward-sim only. Upstream
    # has no such cap (targets are ``fraction × speed_limit``,
    # unbounded); keep this comfortably above any urban/highway speed
    # limit so the 80%/100% IDM policies stay distinct — 15 m/s capped
    # both at 54 km/h and collapsed two of the five longitudinal
    # policies into one on faster roads.
    max_speed: float = 30.0
    # CaRL BatchKinematicBicycleModel default.  The previous 0.6 rad adapter
    # value allowed only ~39% of the reference's maximum bicycle curvature.
    #
    # WARNING (unresolved, 2026-08-29): this makes the PROPOSAL SIMULATION
    # reference-faithful but widens the plant/model gap. ``forward_sim`` builds
    # its bicycle with this value (forward_sim.py:214/680), while the vehicle
    # that actually executes is capped by the controller at 0.6 rad
    # (``core/controllers.py:125``, exported as
    # ``_execution_max_steer_angle_rad``; measured 0.6 in the archived runs).
    # So the scorer now credits ~1.75x the curvature authority the ego has, and
    # tight-intersection proposals will be selected and then under-tracked.
    # Resolving it means either raising the controller/EgoDynamics cap too
    # (which changes every NexusSim policy, not just PDM) or returning both to
    # 0.6 (self-consistent, diverges from CaRL). Left at the reference value
    # because this tree's contract is fidelity of the planner; flagged so the
    # choice is explicit rather than inherited.
    max_steering_angle_rad: float = math.pi / 3.0
    # Longitudinal actuator caps of the PROPOSAL forward-simulation plant.
    # The reference plant (``BatchKinematicBicycleModel`` driven by
    # ``BatchLQRTracker``) applies no acceleration saturation at all, so a
    # faithful scored copy must not clip the tracker's command. The values
    # only exist because :class:`EgoDynamics` takes normalised ``[-1, 1]``
    # throttle/brake; they are far above anything the LQR asks for while
    # tracking an IDM profile (|a| ≲ 3.5 m/s²), so the old 3.0 / 5.0 caps
    # were inert on the stock grid and are kept ONLY by the execution plant
    # (``core/controllers.py``, ``EgoDynamicsCfg`` defaults), which is a
    # separate, declared deviation.
    max_accel: float = 100.0
    max_brake: float = 100.0

    # --- Tracker ----------------------------------------------------------
    tracker: str = "lqr"

    # Pure-pursuit (used only when tracker == "pure_pursuit").
    controller_lookahead_min_m: float = 4.0
    controller_lookahead_gain_s: float = 0.5

    # --- Actuator low-pass filter (proposal-only, opt-in faithful default) ---
    # Mirrors ``CaRL/.../batch_kinematic_bicycle.py``'s
    # ``BatchKinematicBicycleModel`` first-order actuator delay
    # (``accel_time_constant=0.2``, ``steering_angle_time_constant=0.05``).
    # The filter is applied **only** inside the planner's per-proposal
    # forward-simulation. The shared :class:`EgoDynamics` (used by the
    # env loop, training envs, sensor adapters, etc.) is unaffected
    # regardless of these values; see
    # :mod:`navsafe.policy.state.pdm_closed_planner.dynamics` for
    # the wrapper.
    use_actuator_filter: bool = True
    actuator_accel_time_constant_s: float = 0.2
    actuator_steering_time_constant_s: float = 0.05

    # LQR (used when tracker == "lqr"; values from tuplan_garage/batch_lqr.py).
    lqr_q_lateral: Tuple[float, float, float] = (1.0, 10.0, 0.0)
    lqr_r_lateral: float = 1.0
    lqr_q_longitudinal: float = 10.0
    lqr_r_longitudinal: float = 1.0
    lqr_tracking_horizon: int = 10
    lqr_stopping_velocity: float = 0.2
    lqr_stopping_proportional_gain: float = 0.5
    lqr_jerk_penalty: float = 1e-4
    lqr_curvature_rate_penalty: float = 1e-2

    # --- Route extraction -------------------------------------------------
    route_horizon_s: float = 8.0
    route_densify_spacing_m: float = 1.0
    route_source: str = "lane_graph_route"
    map_radius_m: float = 50.0

    # --- Agent forecasting ------------------------------------------------
    # How the planner predicts non-ego agents for proposal simulation and
    # scoring. "constant_velocity" is upstream-faithful (a deployed
    # planner cannot see the future) but disagrees with the EPDMS scorer's
    # log-replay world on ~18% of agents by >2 m at the 4 s horizon.
    # "log_replay" reads the log — the exact future of NexusSim's
    # non-reactive replay environments and the same agent model the scorer
    # judges with, at the price of being an oracle (label any numbers
    # produced with it accordingly). The emergency-brake TTC check always
    # uses log replay regardless (it must agree with the scorer; see
    # planner step 6b).
    agent_forecast: str = "constant_velocity"

    # Dataset/map speed limits are legitimate planner input.  Limits inferred
    # from future logged actor/ego motion are not.  Keep the latter available
    # only as an explicitly labelled experiment.
    allow_inferred_speed_limits: bool = False

    # --- Lane-graph search (used when route_source == "lane_graph_search") --
    # Matches upstream ``carl_nuplan/.../abstract_pdm_planner.py``
    # ``_get_discrete_centerline``'s ``search_depth=30``.
    lane_graph_max_depth: int = 30
    # Stop chaining an UNCONDITIONED lane successor walk once cumulative
    # polyline length exceeds this threshold.  In ``lane_graph_route`` mode
    # the planner walks and caches the complete mission ``route_lane_ids``
    # chain instead; truncating that fixed cache here can strand an unlimited
    # episode at a 200 m artificial endpoint.
    lane_graph_max_length_m: float = 200.0

    # --- Scoring variant --------------------------------------------------
    # ``"upstream"`` (default): NC*DAC*DDC * (5*EP + 5*TTC + 2*HC) / 12.
    #   Matches ``tuplan_garage/pdm_planner/scoring/pdm_scorer.py`` and
    #   CaRL's ``carl_nuplan/.../pdm_scorer.py`` exactly.
    # ``"navsim"`` (opt-in): NC*DAC*DDC*TLC * (5*EP + 5*TTC + 2*LK + 2*HC) / 14.
    #   NavSim-extended PDMS — adds TLC to the multiplicative product
    #   and Lane-Keeping to the weighted sum. Use when planner-internal
    #   scoring should align with NavSim leaderboard semantics.
    scorer_variant: str = "upstream"

    # --- Traffic lights (privileged) ---------------------------------------
    # Upstream PDM-Closed complies with red lights through its OBSERVATION:
    # every red lane connector on the route enters the occupancy map as a
    # stationary obstacle, so IDM brakes to the stop line and releases once
    # the light changes (``PDMObservation._get_traffic_light_geometries`` +
    # ``PDMGenerator._update_leading_agents``). The scorer never sees them.
    # ``True`` reproduces that from the scenario's logged per-timestep light
    # states (``dynamic_map_states`` — the same privileged signal the
    # evaluator's traffic-light term reads); see ``traffic_lights.py``.
    # ``False`` is the pre-2026-08-23 port, which read no light state at all.
    # Overridable per-process with ``NEXUSSIM_PDM_TRAFFIC_LIGHTS=0|1``.
    traffic_light_obstacles: bool = True

    def __post_init__(self) -> None:
        # Cross-field invariants. The dataclass is frozen so we use
        # ``object.__setattr__`` to populate the deferred default for
        # ``idm_policies``.
        if self.num_output_poses <= 0:
            raise ValueError(
                f"num_output_poses must be positive, got {self.num_output_poses}"
            )
        if self.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be positive, got {self.horizon_s}")
        if self.sim_dt <= 0.0:
            raise ValueError(f"sim_dt must be positive, got {self.sim_dt}")
        if self.output_dt <= 0.0:
            raise ValueError(f"output_dt must be positive, got {self.output_dt}")
        if self.output_dt < self.sim_dt:
            raise ValueError(
                f"output_dt ({self.output_dt}) must be >= sim_dt ({self.sim_dt})"
            )

        # Proposal horizon arithmetic — output count × dt must equal
        # the proposal sim horizon when ``num_output_poses ==
        # num_trajectory_poses`` (the default 4 s mode). When the
        # caller picks the 8 s parity mode (16 trajectory poses), the
        # invariant moves to ``num_trajectory_poses * output_dt ==
        # trajectory_horizon_s``.
        expected_traj = self.num_trajectory_poses * self.output_dt
        if abs(expected_traj - self.trajectory_horizon_s) > 1e-6:
            raise ValueError(
                f"num_trajectory_poses ({self.num_trajectory_poses}) * "
                f"output_dt ({self.output_dt}) = {expected_traj}, "
                f"but trajectory_horizon_s = {self.trajectory_horizon_s}"
            )
        # When the caller does not customise the output count we still
        # require the legacy contract: ``num_output_poses * output_dt
        # == horizon_s``. Catches a config that forgot to update both
        # output_dt and horizon_s in lockstep.
        expected_horizon = self.num_output_poses * self.output_dt
        if abs(expected_horizon - self.horizon_s) > 1e-6:
            raise ValueError(
                f"num_output_poses ({self.num_output_poses}) * "
                f"output_dt ({self.output_dt}) = {expected_horizon}, "
                f"but horizon_s = {self.horizon_s}. The adapter contract "
                f"requires these to agree."
            )

        if self.agent_prediction_horizon_s < self.trajectory_horizon_s:
            raise ValueError(
                f"agent_prediction_horizon_s ({self.agent_prediction_horizon_s}) "
                f"must be >= trajectory_horizon_s ({self.trajectory_horizon_s})"
            )

        if self.leading_agent_update_rate <= 0:
            raise ValueError(
                f"leading_agent_update_rate must be positive, "
                f"got {self.leading_agent_update_rate}"
            )

        if not self.lateral_offsets:
            raise ValueError("lateral_offsets must be non-empty")

        if self.emergency_brake_decel <= 0.0:
            raise ValueError(
                f"emergency_brake_decel must be positive, got {self.emergency_brake_decel}"
            )
        if self.infraction_horizon_s <= 0.0:
            raise ValueError(
                f"infraction_horizon_s must be positive, got {self.infraction_horizon_s}"
            )
        # Negative silently disables the no_valid_proposal arm of 6a even
        # for all-zero proposal sets — the one failure mode nothing
        # downstream would surface, and the one state where the
        # stopping_scores_better arm cannot cover for it (a degenerate
        # set scores the brake candidate 0 as well).
        # NaN does the same by a different route (`score <= nan` is always
        # False), so the predicate is written to reject both.
        # No upper bound: >= 1.0 (tests use 1e9) deliberately forces the
        # score brake whenever the ego is below the speed gate, and its
        # effect is loud.
        if not self.emergency_brake_threshold >= 0.0:
            raise ValueError(
                f"emergency_brake_threshold must be non-negative, "
                f"got {self.emergency_brake_threshold}"
            )

        if self.tracker not in ("lqr", "pure_pursuit"):
            raise ValueError(
                f"tracker must be 'lqr' or 'pure_pursuit', got {self.tracker!r}"
            )

        # Actuator filter fields — non-negative, time-constants in seconds.
        if self.actuator_accel_time_constant_s < 0.0:
            raise ValueError(
                "actuator_accel_time_constant_s must be non-negative, "
                f"got {self.actuator_accel_time_constant_s}"
            )
        if self.actuator_steering_time_constant_s < 0.0:
            raise ValueError(
                "actuator_steering_time_constant_s must be non-negative, "
                f"got {self.actuator_steering_time_constant_s}"
            )
        if self.route_source not in (
            "gt_future", "lane_graph_search", "lane_graph_route", "nearest"
        ):
            raise ValueError(
                f"route_source must be one of "
                "{'gt_future', 'lane_graph_search', 'lane_graph_route', "
                "'nearest'}, "
                f"got {self.route_source!r}"
            )
        if self.emergency_brake_score_arms not in (
            "none", "both", "threshold_only"
        ):
            raise ValueError(
                f"emergency_brake_score_arms must be 'none', 'both' or "
                f"'threshold_only', got {self.emergency_brake_score_arms!r}")
        if self.emergency_brake_mode not in ("trajectory", "candidate"):
            raise ValueError(
                f"emergency_brake_mode must be 'trajectory' or 'candidate', "
                f"got {self.emergency_brake_mode!r}"
            )
        if self.agent_forecast not in ("constant_velocity", "log_replay"):
            raise ValueError(
                f"agent_forecast must be 'constant_velocity' or 'log_replay', "
                f"got {self.agent_forecast!r}"
            )
        if self.lane_graph_max_depth <= 0:
            raise ValueError(
                f"lane_graph_max_depth must be positive, got {self.lane_graph_max_depth}"
            )
        if self.lane_graph_max_length_m <= 0.0:
            raise ValueError(
                f"lane_graph_max_length_m must be positive, "
                f"got {self.lane_graph_max_length_m}"
            )
        if self.scorer_variant not in ("upstream", "navsim"):
            raise ValueError(
                f"scorer_variant must be 'upstream' or 'navsim', "
                f"got {self.scorer_variant!r}"
            )

        # Resolve the default IDM policy bank lazily — see field comment.
        if not self.idm_policies:
            from navsafe.policy.state.pdm_closed_planner.idm import (
                DEFAULT_IDM_POLICIES,
            )
            object.__setattr__(self, "idm_policies", DEFAULT_IDM_POLICIES)

        if not self.idm_policies:
            raise ValueError("idm_policies must be non-empty")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_proposals(self) -> int:
        """Total number of proposals = ``len(lateral_offsets) * len(idm_policies)``."""
        return len(self.lateral_offsets) * len(self.idm_policies)

    @property
    def num_sim_steps(self) -> int:
        """Inner-loop step count covering the full *proposal* horizon."""
        return int(round(self.horizon_s / self.sim_dt))

    @property
    def num_trajectory_sim_steps(self) -> int:
        """Inner-loop step count covering the full *output* horizon."""
        return int(round(self.trajectory_horizon_s / self.sim_dt))

    @property
    def output_stride(self) -> int:
        """How many sim steps between two consecutive output samples."""
        return int(round(self.output_dt / self.sim_dt))


def pdm_config_from_env(base: "PDMConfig | None" = None) -> "PDMConfig":
    """A :class:`PDMConfig` with the campaign env overrides applied.

    ``NEXUSSIM_ROUTE_SOURCE`` / ``NEXUSSIM_EMERGENCY_BRAKE_MODE`` /
    ``NEXUSSIM_EMERGENCY_BRAKE_THRESHOLD`` select strategy per-process
    without plumbing flags through every driver. The
    SINGLE application point for those overrides: the adapter used this
    logic while the meta-campaign lanes constructed ``PDMPlanner()`` with
    bare defaults, so with the env set, adapter-driven planning honoured
    the campaign contract while the judge/replay/activation lanes silently
    ran ``trajectory`` mode — and the provenance env snapshot then claimed
    a contract those lanes never executed.

    Overrides are applied one at a time via :func:`dataclasses.replace` so
    each field keeps its declared type and ``__post_init__`` re-validates —
    an invalid env value fails loudly at construction, never silently.

    Returns ``(config, overrides)``-style provenance via the companion
    :func:`pdm_env_overrides`; this function returns only the config.
    """
    import os
    from dataclasses import replace as _replace

    cfg = PDMConfig() if base is None else base
    route_source = os.environ.get("NEXUSSIM_ROUTE_SOURCE", "").strip()
    if route_source:
        cfg = _replace(cfg, route_source=route_source)
    brake_mode = os.environ.get("NEXUSSIM_EMERGENCY_BRAKE_MODE", "").strip()
    if brake_mode:
        cfg = _replace(cfg, emergency_brake_mode=brake_mode)
    # Pin for cross-cut comparability: a campaign that changes the score
    # brake must be reproducible from the env alone — replay tooling and
    # paired-arm relaunches must not silently mix teachers across it.
    brake_thr = os.environ.get(
        "NEXUSSIM_EMERGENCY_BRAKE_THRESHOLD", "").strip()
    if brake_thr:
        try:
            parsed = float(brake_thr)
        except ValueError:
            # float()'s own message never names the variable, so an
            # operator reads a traceback to learn which export was wrong.
            raise ValueError(
                "NEXUSSIM_EMERGENCY_BRAKE_THRESHOLD must be a number, got "
                f"{brake_thr!r}"
            ) from None
        cfg = _replace(cfg, emergency_brake_threshold=parsed)
    brake_arms = os.environ.get(
        "NEXUSSIM_EMERGENCY_BRAKE_SCORE_ARMS", "").strip()
    if brake_arms:
        cfg = _replace(cfg, emergency_brake_score_arms=brake_arms)
    tl = os.environ.get("NEXUSSIM_PDM_TRAFFIC_LIGHTS", "").strip()
    if tl:
        cfg = _replace(cfg, traffic_light_obstacles=_parse_env_bool(
            "NEXUSSIM_PDM_TRAFFIC_LIGHTS", tl))
    red_floor = os.environ.get("NEXUSSIM_RED_LIGHT_BRAKE_FLOOR", "").strip()
    if red_floor:
        cfg = _replace(cfg, red_light_brake_floor=_parse_env_bool(
            "NEXUSSIM_RED_LIGHT_BRAKE_FLOOR", red_floor))
    return cfg


def _parse_env_bool(name: str, raw: str) -> bool:
    """``0/1``, ``true/false``, ``on/off``, ``yes/no`` — anything else is loud."""
    value = raw.strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if value in ("0", "false", "off", "no"):
        return False
    raise ValueError(f"{name} must be 0/1 (or true/false), got {raw!r}")


def pdm_env_overrides() -> dict:
    """Which campaign env overrides are currently set (for provenance)."""
    import os

    overrides = {}
    route_source = os.environ.get("NEXUSSIM_ROUTE_SOURCE", "").strip()
    if route_source:
        overrides["route_source"] = route_source
    brake_mode = os.environ.get("NEXUSSIM_EMERGENCY_BRAKE_MODE", "").strip()
    if brake_mode:
        overrides["emergency_brake_mode"] = brake_mode
    brake_thr = os.environ.get(
        "NEXUSSIM_EMERGENCY_BRAKE_THRESHOLD", "").strip()
    if brake_thr:
        overrides["emergency_brake_threshold"] = brake_thr
    brake_arms = os.environ.get(
        "NEXUSSIM_EMERGENCY_BRAKE_SCORE_ARMS", "").strip()
    if brake_arms:
        overrides["emergency_brake_score_arms"] = brake_arms
    tl = os.environ.get("NEXUSSIM_PDM_TRAFFIC_LIGHTS", "").strip()
    if tl:
        overrides["traffic_light_obstacles"] = tl
    red_floor = os.environ.get("NEXUSSIM_RED_LIGHT_BRAKE_FLOOR", "").strip()
    if red_floor:
        overrides["red_light_brake_floor"] = red_floor
    return overrides


__all__ = [
    "PDMConfig",
    "DEFAULT_LATERAL_OFFSETS",
    "VEHICLE_LENGTH",
    "VEHICLE_WIDTH",
    "VEHICLE_WHEELBASE",
    "pdm_config_from_env",
    "pdm_env_overrides",
]
