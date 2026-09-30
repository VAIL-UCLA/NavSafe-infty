# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Faithful PDM-Closed planner — building blocks.

This subpackage implements the PDM-Closed rule-based planner from
Dauner, Hallgarten, Geiger, Chitta — *Parting with Misconceptions
about Learning-based Vehicle Motion Planning* (CoRL 2023). The
reference implementation lives in
``tuplan_garage/planning/simulation/planner/pdm_planner/pdm_closed_planner.py``.

Faithful reproduction details:

* **Proposal grid**: 3 lateral offsets (``-1, 0, +1`` m) × 5
  longitudinal IDM policies (target speeds at ``20%, 40%, 60%, 80%,
  100%`` of the lane speed limit) → 15 proposals per replan.
* **IDM acceleration exponent δ = 10**, hardcoded in
  ``tuplan_garage/batch_idm_policy.py``. NexusSim's *traffic-agent*
  IDM stays at δ=4 (per the IDM paper); the two paths are
  intentionally independent.
* **EPDMS scoring** with the Predictive Driver Model (PDMS) variant
  (no Extended Comfort) — matches the original CoRL 2023 paper's
  scoring rather than the NavSim EPDMS extension.
* **2-second emergency-brake guard**: the selected proposal is
  re-checked for an imminent collision; if TTC < 2 s, the planner
  emits an emergency-brake trajectory at ``4.05 m/s²`` decel.
* **LQR tracker** (default) for proposal forward simulation, matching
  ``tuplan_garage/batch_lqr.py``. Pure pursuit is retained as an
  opt-in for the legacy ``UnifiedController``-aligned mode.
* **Per-step gap refinement** during forward simulation: lead
  position evolves with constant velocity, the IDM kernel sees the
  current gap each step.
* **Red lights as connector obstacles** (upstream's observation
  semantics): every red connector the route drives becomes a stationary
  box spanning the connector from its stop line, read from the scenario's
  logged light state at the current frame — privileged, like upstream's,
  and with no lookahead (``lookahead_frames=0``). It is released only once
  the ego box is fully inside the connector (upstream's ``within`` latch).
  The scorer never sees the obstacle. Two NexusSim-only extensions stay
  OFF by default: folding TLC into the validity product
  (``emergency_brake_score_arms``) and the red-light brake floor
  (guard 6r, ``red_light_brake_floor``). ``PDMConfig.
  traffic_light_obstacles`` / ``NEXUSSIM_PDM_TRAFFIC_LIGHTS``.

Module map:

* :mod:`navsafe.policy.state.pdm_closed_planner.config` —
  :class:`PDMConfig` (every tunable knob).
* :mod:`navsafe.policy.state.pdm_closed_planner.idm` — IDM policy
  bank and inline kernel (δ=10).
* :mod:`navsafe.policy.state.pdm_closed_planner.route` —
  centerline extraction.
* :mod:`navsafe.policy.state.pdm_closed_planner.forward_sim` —
  per-proposal kinematic-bicycle forward simulation with LQR tracking
  (or pure pursuit on opt-in).
* :mod:`navsafe.policy.state.pdm_closed_planner.proposals` —
  proposal grid generation, lateral offset, lead detection.
* :mod:`navsafe.policy.state.pdm_closed_planner.scoring` — PDMS
  scorer (no EC) wrapping :class:`PDMSTrajectoryScorerFast`.
* :mod:`navsafe.policy.state.pdm_closed_planner.traffic_lights` —
  red connectors on the route → stop-line obstacles.
* :mod:`navsafe.policy.state.pdm_closed_planner.planner` —
  :class:`PDMPlanner` orchestration + emergency-brake guard.
"""

from __future__ import annotations

from navsafe.policy.state.pdm_closed_planner.config import (
    DEFAULT_LATERAL_OFFSETS,
    PDMConfig,
    VEHICLE_LENGTH,
    VEHICLE_WHEELBASE,
    VEHICLE_WIDTH,
    pdm_config_from_env,
    pdm_env_overrides,
)
from navsafe.policy.state.pdm_closed_planner.dynamics import (
    DEFAULT_ACCEL_TIME_CONSTANT_S,
    DEFAULT_STEERING_TIME_CONSTANT_S,
    ActuatorFilterParams,
    FilteredEgoDynamics,
)
from navsafe.policy.state.pdm_closed_planner.idm import (
    ACCELERATION_EXPONENT,
    DEFAULT_IDM_POLICIES,
    IDM_AGGRESSIVE,
    IDM_CONSERVATIVE,
    IDM_NOMINAL,
    IDMPolicy,
    P_20,
    P_40,
    P_60,
    P_80,
    P_100,
    build_default_idm_bank,
    idm_accel,
    idm_speed_profile,
)
from navsafe.policy.state.pdm_closed_planner.route import extract_route_centerline
from navsafe.policy.state.pdm_closed_planner.forward_sim import (
    AgentPrediction,
    ProposalState,
    predict_agents_constant_velocity,
    simulate_proposal,
)
from navsafe.policy.state.pdm_closed_planner.proposals import (
    Proposal,
    find_lead_along_centerline,
    find_lead_vehicle,
    generate_proposals,
    offset_polyline,
)
from navsafe.policy.state.pdm_closed_planner.scoring import PDMScorer
from navsafe.policy.state.pdm_closed_planner.traffic_lights import (
    RedLightStopLine,
    find_red_light_stop_lines,
    is_red_light_token,
    red_lane_ids_at,
    red_light_obstacles,
)
from navsafe.policy.state.pdm_closed_planner.planner import (
    PDMPlanResult,
    PDMPlanner,
    emergency_brake_execution_trajectory,
    emergency_brake_trajectory,
    result_execution_trajectory,
)

__all__ = [
    # config
    "PDMConfig",
    "DEFAULT_LATERAL_OFFSETS",
    "VEHICLE_LENGTH",
    "VEHICLE_WIDTH",
    "VEHICLE_WHEELBASE",
    "pdm_config_from_env",
    "pdm_env_overrides",
    # dynamics (PDM-Closed-private actuator filter wrapper)
    "ActuatorFilterParams",
    "FilteredEgoDynamics",
    "DEFAULT_ACCEL_TIME_CONSTANT_S",
    "DEFAULT_STEERING_TIME_CONSTANT_S",
    # idm
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
    # route
    "extract_route_centerline",
    # forward sim
    "AgentPrediction",
    "ProposalState",
    "predict_agents_constant_velocity",
    "simulate_proposal",
    # proposals
    "Proposal",
    "offset_polyline",
    "find_lead_along_centerline",
    "find_lead_vehicle",
    "generate_proposals",
    # scoring
    "PDMScorer",
    # traffic lights
    "RedLightStopLine",
    "find_red_light_stop_lines",
    "is_red_light_token",
    "red_lane_ids_at",
    "red_light_obstacles",
    # planner
    "PDMPlanResult",
    "PDMPlanner",
    "emergency_brake_execution_trajectory",
    "emergency_brake_trajectory",
    "result_execution_trajectory",
]
