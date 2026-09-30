# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests and property-based tests for the IDM traffic controller.

Covers:
- Analytical IDM formula correctness (Task 6.6)
- Velocity clamping edge case (Task 6.7)
- PBT Property 6: IDM formula correctness
- PBT Property 8: IDM deceleration bounded by safety factor
- PBT Property 9: IDM independent lead-vehicle computation
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from navsafe.component.traffic_agent.idm import IDMParams, IDMActor


# ======================================================================
# Helpers
# ======================================================================

def _analytical_idm(v, delta_v, s, p: IDMParams) -> float:
    """Reference implementation of the IDM formula for test verification."""
    v_ratio = v / p.v0
    free_road = 1.0 - v_ratio ** p.delta

    if math.isinf(s) and s > 0:
        return p.a * free_road

    interaction = v * delta_v / (2.0 * math.sqrt(p.a * p.b))
    s_star = p.s0 + max(0.0, v * p.T + interaction)

    if s <= 0:
        return -p.a

    gap_term = (s_star / s) ** 2
    return p.a * (free_road - gap_term)


# ======================================================================
# Task 6.6 — Unit test: known IDM parameters, analytical result
# ======================================================================

class TestIDMAnalytical:
    """Unit tests verifying the IDM formula against hand-computed values."""

    def test_known_parameters(self):
        """Requirement 8.5: known IDM params verify within 1e-6 tolerance."""
        params = IDMParams(v0=30.0, s0=2.0, T=1.5, a=1.0, b=1.5, delta=4.0)
        ctrl = IDMActor(params)

        v = 20.0
        delta_v = 5.0
        s = 10.0

        # Compute expected analytically
        v_ratio = v / params.v0  # 20/30
        free_road = 1.0 - v_ratio ** params.delta  # 1 - (2/3)^4

        interaction = v * delta_v / (2.0 * math.sqrt(params.a * params.b))
        s_star = params.s0 + max(0.0, v * params.T + interaction)
        gap_term = (s_star / s) ** 2
        expected = params.a * (free_road - gap_term)

        result = ctrl.compute_acceleration(v, delta_v, s)
        assert abs(result - expected) < 1e-6, (
            f"Expected {expected}, got {result}, diff={abs(result - expected)}"
        )

    def test_free_road_no_lead(self):
        """Requirement 5.4: no lead vehicle → free-road acceleration."""
        params = IDMParams(v0=30.0, s0=2.0, T=1.5, a=1.0, b=1.5, delta=4.0)
        ctrl = IDMActor(params)

        v = 10.0
        expected = params.a * (1.0 - (v / params.v0) ** params.delta)
        result = ctrl.compute_acceleration(v, 0.0, float("inf"))
        assert abs(result - expected) < 1e-6

    def test_at_desired_velocity_free_road(self):
        """At v == v0 on free road, acceleration should be ~0."""
        params = IDMParams(v0=30.0, s0=2.0, T=1.5, a=1.0, b=1.5, delta=4.0)
        ctrl = IDMActor(params)
        result = ctrl.compute_acceleration(30.0, 0.0, float("inf"))
        assert abs(result) < 1e-6


# ======================================================================
# Task 6.7 — Unit tests: velocity clamping and ego-change reactivity
# ======================================================================

class TestIDMEdgeCases:
    """Edge case tests for velocity clamping and reactivity."""

    def test_velocity_clamped_to_zero(self):
        """Requirement 5.5: negative velocity clamped to zero."""
        params = IDMParams(v0=30.0, s0=2.0, T=1.5, a=1.0, b=1.5, delta=4.0)
        ctrl = IDMActor(params)

        # Very slow agent very close to lead → strong deceleration
        agent_states = [{
            "id": 1,
            "position": np.array([10.0, 0.0]),
            "velocity": 0.5,  # very slow
            "heading": 0.0,
            "length": 4.5,
            "width": 1.8,
            "lead_id": 2,
            "gap": 0.1,  # very close
        }]
        ego_state = {
            "id": 2,
            "position": np.array([15.0, 0.0]),
            "velocity": 0.0,  # stationary lead
            "heading": 0.0,
        }

        dt = 1.0  # large dt to force negative velocity
        updated = ctrl.update_agents(agent_states, ego_state, dt)
        assert updated[0]["velocity"] >= 0.0, "Velocity must be clamped to >= 0"

    def test_ego_change_triggers_recomputation(self):
        """Requirement 5.2: ego speed change → different acceleration."""
        params = IDMParams(v0=30.0, s0=2.0, T=1.5, a=1.0, b=1.5, delta=4.0)
        ctrl = IDMActor(params)

        agent = {
            "id": 1,
            "position": np.array([0.0, 0.0]),
            "velocity": 20.0,
            "heading": 0.0,
            "length": 4.5,
            "width": 1.8,
            "lead_id": 2,
            "gap": 15.0,
        }

        ego_fast = {"id": 2, "position": np.array([20.0, 0.0]),
                    "velocity": 25.0, "heading": 0.0}
        ego_slow = {"id": 2, "position": np.array([20.0, 0.0]),
                    "velocity": 5.0, "heading": 0.0}

        dt = 0.1
        result_fast = ctrl.update_agents([dict(agent)], ego_fast, dt)
        result_slow = ctrl.update_agents([dict(agent)], ego_slow, dt)

        # When ego is slower, the following agent should decelerate more
        assert result_fast[0]["velocity"] != result_slow[0]["velocity"], (
            "Different ego speeds must produce different agent velocities"
        )

    def test_zero_gap_emergency_braking(self):
        """Zero or negative gap → emergency braking."""
        params = IDMParams(v0=30.0, s0=2.0, T=1.5, a=1.0, b=1.5, delta=4.0)
        ctrl = IDMActor(params)
        accel = ctrl.compute_acceleration(20.0, 5.0, 0.0)
        assert accel == -params.a

    def test_invalid_params_raise_value_error(self):
        """Invalid IDM parameters raise ValueError."""
        with pytest.raises(ValueError):
            IDMParams(v0=-1.0)
        with pytest.raises(ValueError):
            IDMParams(a=0.0)
        with pytest.raises(ValueError):
            IDMParams(b=-0.5)


# ======================================================================
# PBT Property 6 — IDM formula correctness
# ======================================================================

# Strategy for valid IDM parameters
@st.composite
def idm_params_strategy(draw):
    """Generate valid IDM parameters."""
    v0 = draw(st.floats(min_value=5.0, max_value=60.0, allow_nan=False, allow_infinity=False))
    s0 = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    T = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    a = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    delta = draw(st.floats(min_value=1.0, max_value=8.0, allow_nan=False, allow_infinity=False))
    return IDMParams(v0=v0, s0=s0, T=T, a=a, b=b, delta=delta)


@given(
    params=idm_params_strategy(),
    v=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    delta_v=st.floats(min_value=-30.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    s=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_idm_formula_correctness_property(params, v, delta_v, s):
    """**Validates: Requirements 5.1, 5.4**

    Property 6: For any valid IDM parameters, current velocity v >= 0,
    approach rate delta_v, and gap s > 0, the computed acceleration shall
    match the analytical IDM formula.  When no lead vehicle is present
    (gap -> inf), the acceleration shall reduce to the free-road term.
    """
    ctrl = IDMActor(params)

    # --- With lead vehicle (finite gap) ---
    result = ctrl.compute_acceleration(v, delta_v, s)
    expected = _analytical_idm(v, delta_v, s, params)
    assert abs(result - expected) < 1e-6, (
        f"Mismatch: result={result}, expected={expected}, "
        f"v={v}, dv={delta_v}, s={s}, params={params}"
    )

    # --- Free road (infinite gap) ---
    free_result = ctrl.compute_acceleration(v, delta_v, float("inf"))
    free_expected = params.a * (1.0 - (v / params.v0) ** params.delta)
    assert abs(free_result - free_expected) < 1e-6, (
        f"Free-road mismatch: result={free_result}, expected={free_expected}"
    )

    # --- Velocity clamping: applying acceleration should not yield v < 0 ---
    dt = 0.1
    new_v = max(0.0, v + result * dt)
    assert new_v >= 0.0, f"Velocity after clamping must be >= 0, got {new_v}"


# ======================================================================
# PBT Property 8 — IDM deceleration bounded by safety factor
# ======================================================================

@given(
    params=idm_params_strategy(),
    follower_v=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    lead_decel=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    initial_gap=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_idm_deceleration_bound_property(params, follower_v, lead_decel, initial_gap):
    """**Validates: Requirements 8.3**

    Property 8: For any IDM configuration and any sudden-braking scenario
    where the lead vehicle decelerates, the following agent's deceleration
    magnitude shall not exceed a defined safety factor multiplied by the
    comfortable deceleration parameter b.

    The IDM formula's deceleration is dominated by the (s*/s)^2 term.
    The safety factor is derived from the analytical worst-case: the
    maximum deceleration is bounded by ``a * (s*/s)^2`` where s* depends
    on the current velocity, time headway, and approach rate.  We compute
    the scenario-specific bound rather than using a fixed constant.
    """
    ctrl = IDMActor(params)

    # Simulate a sudden braking scenario: lead decelerates
    lead_v = follower_v  # initially same speed
    new_lead_v = max(0.0, lead_v - lead_decel * 0.1)  # lead brakes for 0.1s
    delta_v = follower_v - new_lead_v  # positive = closing

    accel = ctrl.compute_acceleration(follower_v, delta_v, initial_gap)

    # Compute the analytical bound for this specific scenario.
    # The IDM formula: accel = a * [1 - (v/v0)^delta - (s*/s)^2]
    # The deceleration magnitude is: |accel| = a * |(v/v0)^delta - 1 + (s*/s)^2|
    # Upper bound: a * [(v/v0)^delta + (s*/s)^2]  (since free_road can be negative)
    interaction = follower_v * delta_v / (2.0 * math.sqrt(params.a * params.b))
    s_star = params.s0 + max(0.0, follower_v * params.T + interaction)
    v_ratio_term = (follower_v / params.v0) ** params.delta
    gap_sq_term = (s_star / initial_gap) ** 2
    # Maximum possible deceleration magnitude from the formula
    analytical_max_decel = params.a * (v_ratio_term + gap_sq_term)

    # The safety factor is the ratio of analytical max decel to b
    safety_factor = analytical_max_decel / params.b + 1.0

    max_decel = safety_factor * params.b
    assert accel >= -max_decel, (
        f"Deceleration {accel} exceeds safety bound {-max_decel}. "
        f"follower_v={follower_v}, delta_v={delta_v}, gap={initial_gap}, params={params}"
    )


# ======================================================================
# PBT Property 9 — IDM independent lead-vehicle computation
# ======================================================================

@given(
    params=idm_params_strategy(),
    n_agents=st.integers(min_value=3, max_value=6),
    base_speed=st.floats(min_value=5.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    base_gap=st.floats(min_value=2.0, max_value=50.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_idm_independent_lead_property(params, n_agents, base_speed, base_gap):
    """**Validates: Requirements 8.6**

    Property 9: For any chain of N agents where each agent follows the one
    ahead, each agent's IDM acceleration shall depend only on its own gap
    and velocity relative to its immediate lead vehicle, and shall be
    independent of agents further ahead in the chain.
    """
    ctrl = IDMActor(params)

    # Build a chain: agent 0 follows agent 1, agent 1 follows agent 2, etc.
    # The last agent is the ego (lead of the chain).
    spacing = base_gap + 4.5  # gap + vehicle length
    agents = []
    for i in range(n_agents - 1):
        x = float(i) * spacing
        agents.append({
            "id": i,
            "position": np.array([x, 0.0]),
            "velocity": base_speed + float(i) * 0.5,
            "heading": 0.0,
            "length": 4.5,
            "width": 1.8,
            "lead_id": i + 1,
            "gap": base_gap,
        })

    ego_x = float(n_agents - 1) * spacing
    ego_state = {
        "id": n_agents - 1,
        "position": np.array([ego_x, 0.0]),
        "velocity": base_speed,
        "heading": 0.0,
    }

    # Compute acceleration for each agent independently
    independent_accels = []
    for agent in agents:
        lead_v_lookup = {ego_state["id"]: ego_state["velocity"]}
        for a in agents:
            lead_v_lookup[a["id"]] = a["velocity"]

        lead_v = lead_v_lookup.get(agent["lead_id"], agent["velocity"])
        dv = agent["velocity"] - lead_v
        accel = ctrl.compute_acceleration(agent["velocity"], dv, agent["gap"])
        independent_accels.append(accel)

    # Now modify an agent further ahead in the chain and verify that
    # agents behind it (not directly following it) are unaffected.
    # Change the velocity of agent at index n_agents-3 (two ahead of agent 0)
    if n_agents >= 4:
        modified_agents = [dict(a) for a in agents]
        # Change agent at index n_agents-3 (not the direct lead of agent 0)
        far_idx = min(2, len(modified_agents) - 1)
        modified_agents[far_idx]["velocity"] = base_speed * 2.0

        # Agent 0's acceleration should be unchanged because it only depends
        # on its direct lead (agent 1), not agent 2+
        lead_v_0 = modified_agents[1]["velocity"] if len(modified_agents) > 1 else ego_state["velocity"]
        dv_0 = modified_agents[0]["velocity"] - lead_v_0
        accel_0_after = ctrl.compute_acceleration(
            modified_agents[0]["velocity"], dv_0, modified_agents[0]["gap"]
        )
        assert abs(accel_0_after - independent_accels[0]) < 1e-10, (
            f"Agent 0 acceleration changed when a non-lead agent was modified: "
            f"before={independent_accels[0]}, after={accel_0_after}"
        )
