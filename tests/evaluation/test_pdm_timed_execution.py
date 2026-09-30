# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""PDM kinematics must survive the adapter-to-controller seam."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg
from navsafe.evaluation.evaluator import EvaluationConfig, Evaluator
from navsafe.policy.state.pdm_closed import PDMClosedAdapter


class _TimedAdapter:
    def prepare_input(self, **kwargs):
        return kwargs

    def run_inference(self, model_input):
        return {
            # One metre left is path projection, not forward travel.
            "trajectory": np.array(
                [[-1.0, 1.0], [-1.0, 2.0], [-1.0, 3.0]], dtype=float),
            "trajectory_speeds_mps": np.full(3, 10.0),
        }

    def parse_output(self, output, ego_state):
        return output

    def get_waypoint_dt(self):
        return 0.1

    def preserves_trajectory_timing(self):
        return True


class _Controller:
    _steering_angle = 0.0
    wheelbase = 2.8
    max_steer_angle = 0.6

    def compute(self, ego_state, trajectory, target_speed):
        self.trajectory = np.asarray(trajectory).copy()
        self.target_speed = float(target_speed)
        return 0.0, 0.0


class _Env:
    agent_states = []
    _ego = SimpleNamespace(cfg=SimpleNamespace(
        max_speed=30.0, max_accel=3.0, max_brake=5.0,
        ego_length=4.515, ego_width=1.852))

    def clear_ego_override(self):
        pass

    def set_ego_override(self, **kwargs):
        self.override = kwargs


def _evaluator():
    evaluator = object.__new__(Evaluator)
    evaluator.adapter = _TimedAdapter()
    evaluator.controller = _Controller()
    evaluator.env = _Env()
    evaluator.config = EvaluationConfig(
        execution_mode="controller", sim_dt=0.1, ego_replay_frames=0, replan_rate=1,
        controller_type="pure_pursuit")
    evaluator._route = SimpleNamespace(policy_context=lambda position: {})
    evaluator.scenario_data = {}
    evaluator.scenario_id = "pdm-offset"
    evaluator.frame = 0
    evaluator.trajectory_scorer = None
    evaluator._warned_backward_plan = False
    return evaluator


def test_lateral_projection_is_not_converted_to_forward_speed():
    evaluator = _evaluator()
    ego = {
        "position": np.zeros(3), "heading": 0.0, "speed": 10.0,
        "velocity": np.array([10.0, 0.0, 0.0]),
    }

    evaluator._run_inference_and_cache(ego, {})
    evaluator._execute_controller_step(ego, replan_offset=0)

    # Teleport pacing retains its physical-ego origin, while LQR receives the
    # projected PDM reference directly. Most importantly, the 1 m lateral
    # projection cannot inflate sqrt(1^2 + 1^2)/0.1 into 14.14 m/s.
    np.testing.assert_allclose(evaluator._cached_world_traj[0], [0.0, 0.0])
    np.testing.assert_allclose(evaluator.controller.trajectory[0], [1.0, -1.0])
    assert evaluator.controller.target_speed == pytest.approx(10.0)


def test_half_second_offset_path_keeps_controller_tangent_and_speed_timing():
    evaluator = _evaluator()
    evaluator.adapter.get_waypoint_dt = lambda: 0.5
    evaluator.adapter.run_inference = lambda _: {
        "trajectory": np.array([
            [-1.0, 1.0], [-1.0, 2.0], [-1.0, 3.0], [-1.0, 4.0],
        ]),
        "trajectory_speeds_mps": np.array([2.0, 3.0, 4.0, 5.0]),
    }
    ego = {
        "position": np.zeros(3), "heading": 0.0, "speed": 2.0,
        "velocity": np.array([2.0, 0.0, 0.0]),
    }

    evaluator._run_inference_and_cache(ego, {})
    evaluator._execute_controller_step(ego, replan_offset=0)

    controller = evaluator.controller.trajectory
    # World axes are [forward, left].  All controller segments follow the
    # planner's straight offset line; the teleport plan alone starts at ego.
    np.testing.assert_allclose(controller[:, 1], -1.0, atol=1e-12)
    assert np.all(np.diff(controller[:, 0]) > 0.0)
    np.testing.assert_allclose(evaluator._cached_world_traj[0], [0.0, 0.0])
    assert evaluator.controller.target_speed == pytest.approx(1.2)
    # Back-extrapolation places the supplied 2 m/s value at its true t=.5
    # index; it does not collapse 0.5-second samples onto simulator frames.
    assert evaluator._cached_controller_plan_speeds[4] == pytest.approx(2.0)


def test_xy_only_edit_uses_projected_first_interval_at_every_live_step():
    evaluator = _evaluator()
    evaluator.adapter.get_waypoint_dt = lambda: 0.5
    evaluator.adapter.run_inference = lambda _: {
        "trajectory": np.array([
            [-1.0, 1.0], [-1.0, 3.0], [-1.0, 5.0],
        ]),
    }
    ego = {
        "position": np.zeros(3), "heading": 0.0, "speed": 2.0,
        "velocity": np.array([2.0, 0.0, 0.0]),
    }

    evaluator._run_inference_and_cache(ego, {})
    np.testing.assert_allclose(
        evaluator._cached_controller_plan_speeds[:5], np.full(5, 2.0))
    live_targets = []
    for offset in range(5):
        evaluator._execute_controller_step(ego, replan_offset=offset)
        live_targets.append(evaluator.controller.target_speed)
    np.testing.assert_allclose(live_targets, np.full(5, 2.0))


def test_zero_velocity_brake_profile_reaches_controller_unchanged():
    evaluator = _evaluator()
    evaluator.adapter.run_inference = lambda _: {
        "trajectory": np.array([[0.0, -0.1], [0.0, -0.2]]),
        "trajectory_speeds_mps": np.zeros(2),
    }
    ego = {
        "position": np.zeros(3), "heading": 0.0, "speed": 2.0,
        "velocity": np.array([2.0, 0.0, 0.0]),
    }

    evaluator._run_inference_and_cache(ego, {})
    evaluator._execute_controller_step(ego, replan_offset=0)

    assert evaluator.controller.target_speed == 0.0


def test_future_stationary_pose_is_not_mistaken_for_time_zero_in_teleport():
    evaluator = _evaluator()
    evaluator.adapter.run_inference = lambda _: {
        # Both samples are future-only: t=.1 is stationary at the origin and
        # t=.2 moves. The first simulator step must execute t=.1, not t=.2.
        "trajectory": np.array([[0.0, 0.0], [0.0, 1.0]]),
        "trajectory_speeds_mps": np.array([0.0, 10.0]),
    }
    ego = {
        "position": np.zeros(3), "heading": 0.0, "speed": 0.0,
        "velocity": np.zeros(3),
    }

    evaluator._run_inference_and_cache(ego, {})
    evaluator._execute_teleport_step(
        ego, images={}, needs_replan=True, replan_offset=0)

    assert len(evaluator._cached_world_traj) == 3
    np.testing.assert_allclose(evaluator.env.override["position"], [0.0, 0.0])


def test_external_scorer_preserves_selected_pdm_brake_semantics():
    evaluator = _evaluator()
    adapter = PDMClosedAdapter(checkpoint_path="none")
    raw_brake = np.array([[0.0, -0.3355], [0.0, -0.671]])
    adapter.run_inference = lambda _: {
        "trajectory": raw_brake,
        "trajectory_speeds_mps": np.zeros(2),
        "emergency_brake_triggered": True,
        "best_idx": 0,
        "scores": np.array([0.0]),
        "proposals": [SimpleNamespace(output_xy_ego=raw_brake.copy())],
        "route_source": "lane_graph_route",
        "all_candidates": torch.from_numpy(
            raw_brake[:, [1, 0]][None, None, ...]),
    }
    evaluator.adapter = adapter

    class _KeepCandidateZero:
        def select_best(self, output, **kwargs):
            return {
                "trajectory": output["all_candidates"][:, 0],
                "scores": torch.tensor([[1.0]]),
                "best_idx": torch.tensor([0]),
            }

    evaluator.trajectory_scorer = _KeepCandidateZero()
    ego = {
        "position": np.zeros(3), "heading": 0.0, "speed": 1.0,
        "velocity": np.array([1.0, 0.0, 0.0]),
    }
    evaluator._run_inference_and_cache(ego, {})
    evaluator._execute_controller_step(ego, replan_offset=0)

    # The .5 s output is resampled to the simulator's .1 s controller grid;
    # every resampled pose must remain a stop (never the reverse geometry).
    np.testing.assert_array_equal(
        evaluator.controller.trajectory,
        np.zeros_like(evaluator.controller.trajectory))
    assert evaluator.controller.target_speed == 0.0


class _ClockStopAdapter(_TimedAdapter):
    def __init__(self):
        from navsafe.core.stopping import make_stop_command

        forward = np.arange(1, 13) * 3.0
        self.reference = np.column_stack([0.025 * forward**2, forward])
        self.stop_command = make_stop_command(0.5)
        self.calls = 0

    def prepare_input(self, **kwargs):
        self.planning_state = copy.deepcopy(kwargs["ego_state"])
        return kwargs

    def run_inference(self, model_input):
        self.calls += 1
        return {"trajectory": self.reference.copy(),
                "stop_command": self.stop_command,
                "stop_reference_trajectory": self.reference.copy()}

    def get_waypoint_dt(self):
        return 0.5


@pytest.fixture
def clock_runner(tmp_path):
    def build(dt=0.1):
        plant = EgoDynamics(EgoDynamicsCfg(dt=dt))
        plant.reset(x=20.0, y=0.0, heading=0.0, speed=8.0)
        env = SimpleNamespace(_ego=plant, dt=dt, agent_states=[],
                              clear_ego_override=Mock())
        adapter = _ClockStopAdapter()
        evaluator = Evaluator(env, adapter, EvaluationConfig(
            sim_dt=dt, ego_replay_frames=0, replan_rate=5,
            controller_type="lqr", execution_mode="controller",
            output_dir=tmp_path))
        evaluator._route = SimpleNamespace(policy_context=lambda _: {},
                                           reset=lambda: None)
        evaluator._log_execution_debug = Mock()
        return evaluator, env, adapter

    return build


def test_execution_clock_mismatch_rejected_before_provider(clock_runner):
    evaluator, env, adapter = clock_runner()
    # A scene changes the actual clock while the evaluation configuration
    # still describes 0.1 s. Neither a new owner nor a replan may conceal it.
    env._ego.cfg.dt = env.dt = 0.2
    with pytest.raises(ValueError, match="configured execution timestep"):
        Evaluator(env, adapter, evaluator.config)
    with pytest.raises(ValueError, match="configured execution timestep"):
        evaluator._run_inference_and_cache(env._ego.get_state(), {})
    assert adapter.calls == 0
    env.clear_ego_override.assert_not_called()


def test_cached_clock_change_refused_before_actuator_or_replan(clock_runner):
    evaluator, env, adapter = clock_runner()
    evaluator._run_inference_and_cache(env._ego.get_state(), {})
    steering = evaluator.controller._steering_angle
    position = env._ego.get_state()["position"].copy()
    # Matching all current clocks still cannot reuse a plan measured under
    # the old clock, or silently change the episode's accumulated time.
    env._ego.cfg.dt = env.dt = evaluator.config.sim_dt = 0.2
    with pytest.raises(ValueError, match="execution timestep changed"):
        evaluator._execute_controller_step(env._ego.get_state(), replan_offset=0)
    with pytest.raises(ValueError, match="execution timestep changed"):
        evaluator._run_inference_and_cache(env._ego.get_state(), {})
    assert adapter.calls == 1
    assert evaluator.controller._steering_angle == steering
    np.testing.assert_array_equal(env._ego.get_state()["position"], position)
    env.clear_ego_override.assert_not_called()
