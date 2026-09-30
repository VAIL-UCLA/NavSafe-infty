"""Regression tests for closed-loop trajectory-execution fixes.

Covers the two execution bugs that tanked closed-loop driving scores:

1. Heading flips — ego heading was recomputed from the atan2 of two
   *adjacent* sim_dt-interpolated waypoints (centimetres apart at low
   speed), so noise could flip the heading by up to 180° and corrupt the
   next plan's ego→world transform. ``Evaluator.heading_from_plan`` now
   uses a ~0.5 s lookahead and falls back to the current heading for
   near-stationary plans.

2. Double integration — the evaluator teleports the ego via
   ``set_ego_override`` (which syncs ``_ego.speed`` to the injected
   velocity), and ``NexusSimEnv.step`` then advanced the bicycle model
   *again* from the teleported pose, overshooting one waypoint per replan
   frame and freezing on the frames in between. ``step`` now treats the
   override pose as authoritative and skips bicycle-model integration.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from navsafe.env.env_cfg import EnvCfg
from navsafe.env.navsafe_env import NexusSimEnv
from navsafe.evaluation.evaluator import Evaluator
from navsafe.core import plan_execution


class TestHeadingFromPlan:
    def test_moving_plan_returns_route_direction(self) -> None:
        # Straight-line plan heading northeast at 45°.
        t = np.linspace(0.0, 4.0, 41)
        traj = np.stack([t, t], axis=1)
        h = Evaluator.heading_from_plan(traj, current_heading=0.0)
        assert h == pytest.approx(np.pi / 4)

    def test_near_stationary_plan_keeps_current_heading(self) -> None:
        # 4 cm of total displacement — below min_disp: heading must not
        # jump to the (noise-dominated) waypoint direction.
        traj = np.array([[0.0, 0.0], [-0.01, 0.0], [-0.02, 0.0],
                         [-0.03, 0.0], [-0.04, 0.0], [-0.04, 0.0]])
        h = Evaluator.heading_from_plan(traj, current_heading=2.966)
        assert h == pytest.approx(2.966)

    def test_noisy_adjacent_waypoints_do_not_flip_heading(self) -> None:
        # Slow forward plan with a backward first segment (model wobble).
        # The adjacent-pair atan2 would have returned ~pi (a 180° flip);
        # the lookahead pair sees net forward motion.
        traj = np.array([[0.0, 0.0], [-0.05, 0.0], [0.1, 0.0],
                         [0.25, 0.0], [0.4, 0.0], [0.55, 0.0]])
        h = Evaluator.heading_from_plan(traj, current_heading=0.1)
        assert abs(h) < np.pi / 2  # forward-ish, not flipped

    def test_short_subset_clamps_lookahead(self) -> None:
        traj = np.array([[0.0, 0.0], [1.0, 0.0]])
        h = Evaluator.heading_from_plan(traj, current_heading=1.0)
        assert h == pytest.approx(0.0)

    def test_single_point_returns_current_heading(self) -> None:
        traj = np.array([[3.0, 4.0]])
        assert Evaluator.heading_from_plan(traj, current_heading=0.7) == 0.7


class TestPlanPacing:
    """Plan pacing: the ego advances along the plan at
    ``PLAN_PACING_FACTOR`` × the plan's average speed instead of replaying
    the plan's ramp-up waypoint-by-waypoint."""

    def test_plan_pacing_factor_is_pinned(self) -> None:
        """The constant is a literal, pinned to a measured sweep.

        Every other assertion in this file reads the constant back
        (``speed == PLAN_PACING_FACTOR * mean``), which is tautological —
        verified: the whole file passes with the constant set to 0.5. That
        is how a 0.8 -> 0.96 retune once landed unguarded.

        0.80 is behaviour-preserving: with the plan-origin fix in place,
        PDM-Closed reproduces the pre-fix baseline's plan speeds to within
        1% (median ratio 0.993/0.994/0.990/1.002 over scenes 0-3). Raising
        it is not a pace change but a closed-loop gain change — executed
        speed seeds the next plan, so d ln v / d ln f = 1/(1 - f), and at
        0.96 the teacher escaped to the IDM fallback target (AV2 carries no
        speed limits) and drove 2.26x the logged human's mean on scene 1."""
        assert plan_execution.PLAN_PACING_FACTOR == 0.80

    def test_pacing_speed_is_the_factor_times_plan_average(self) -> None:
        # 40 waypoints at 0.1s, uniform 0.3 m spacing → 3.0 m/s plan speed.
        traj = np.stack([np.arange(41) * 0.3, np.zeros(41)], axis=1)
        arc, speed = Evaluator.plan_pacing(traj, sim_dt=0.1)
        assert speed == pytest.approx(plan_execution.PLAN_PACING_FACTOR * 3.0)
        assert arc[-1] == pytest.approx(12.0)

    def test_ramp_plan_paces_at_average_not_start(self) -> None:
        # Ramp from 0.5 to 3.5 m/s: literal first-waypoint execution would
        # move at ~0.5 m/s; pacing must use the average (~2 m/s) instead.
        speeds = np.linspace(0.5, 3.5, 40)
        x = np.concatenate([[0.0], np.cumsum(speeds * 0.1)])
        traj = np.stack([x, np.zeros_like(x)], axis=1)
        _, speed = Evaluator.plan_pacing(traj, sim_dt=0.1)
        assert speed == pytest.approx(
            plan_execution.PLAN_PACING_FACTOR * 2.0, rel=0.02)

    def test_degenerate_plan_paces_zero(self) -> None:
        _, speed = Evaluator.plan_pacing(np.array([[1.0, 2.0]]), sim_dt=0.1)
        assert speed == 0.0

    def test_point_at_arc_interpolates_and_clamps(self) -> None:
        # All four cases live HERE: the mid-segment interpolations and both
        # clamps. (A refactor once relocated the last three into the middle
        # of an unrelated test body, where they passed only by coincidence
        # of shared fixtures — and deleting that test would have silently
        # deleted clamp coverage.)
        traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        arc = np.array([0.0, 1.0, 2.0])
        p, i = Evaluator.point_at_arc(traj, arc, 0.5)
        np.testing.assert_allclose(p, [0.5, 0.0])
        assert i == 0
        p, i = Evaluator.point_at_arc(traj, arc, 1.5)
        np.testing.assert_allclose(p, [1.0, 0.5])
        assert i == 1
        p, _ = Evaluator.point_at_arc(traj, arc, 99.0)   # clamp to end
        np.testing.assert_allclose(p, [1.0, 1.0])
        p, i = Evaluator.point_at_arc(traj, arc, -1.0)   # clamp to start
        np.testing.assert_allclose(p, [0.0, 0.0])
        assert i == 0

    def test_shared_teleport_step_uses_final_step_velocity_and_heading(self) -> None:
        """Boundary probes must reproduce production step-by-step state.

        Average origin-to-boundary displacement is not the final ego
        velocity and gives the wrong next-plan heading on a curved path.

        ``pos = traj[0]`` is the production convention: a cached plan's
        index 0 IS the ego pose it was produced from (``arc[0] == 0``), and
        ``TestPlanExecutionComposition`` pins that end to end.
        """
        traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        arc = plan_execution.cumulative_arc_length(traj)
        pos = traj[0].copy()
        heading = 0.0
        s = 0.0
        for _ in range(2):
            pos, heading, velocity, s = plan_execution.teleport_plan_step(
                traj, arc, s, speed=1.0, sim_dt=1.0,
                current_position=pos, current_heading=heading)
        assert np.allclose(pos, [1.0, 1.0])
        assert np.allclose(velocity, [0.0, 1.0])
        assert heading == pytest.approx(np.pi / 2)


class TestPlanExecutionComposition:
    """The whole execution chain, composed as production composes it:
    ``interpolate_plan → ego_plan_with_origin → plan_pacing →
    teleport_plan_step``.

    The pieces were each tested in isolation and the composition was not,
    which is how the two-step replan frame survived: ``interpolate_plan``
    sampled from ``t = sim_dt``, so the cached plan's index 0 sat one sim
    step ahead of the ego while ``cumulative_arc_length`` labelled it
    ``arc[0] = 0``. Every replan reset ``arc_s`` to 0 and then advanced a
    paced step from there, so the replan frame executed the ego→waypoint[0]
    gap for free — a 2× displacement spike once every ``replan_rate``
    frames (measured max/min phase ratio ≈ 2.0 on all eight Stage 0
    baseline runs).
    """

    SIM_DT = 0.1
    MODEL_DT = 0.5
    REPLAN_RATE = 5

    @staticmethod
    def _model_plan(n: int = 8, model_dt: float = 0.5) -> np.ndarray:
        """(n, 2) ego-frame [lateral, forward] waypoints at ``model_dt``.

        A ramp (1 → 5 m/s), like a real plan: the first waypoint is closer
        than the mean spacing, which is exactly the gap the bug executed
        for free.
        """
        speeds = np.linspace(1.0, 5.0, n)
        forward = np.cumsum(speeds * model_dt)
        return np.stack([np.zeros(n), forward], axis=1)

    def _cache_plan(self, ego_xy: np.ndarray, model_dt: float | None = None):
        """Production's ``_run_inference_and_cache`` / ``_cache_execution_plan``
        for an ego at ``ego_xy`` with heading 0."""
        dt_model = self.MODEL_DT if model_dt is None else model_dt
        traj = plan_execution.interpolate_plan(
            self._model_plan(model_dt=dt_model), dt_model, self.SIM_DT)
        traj = plan_execution.ego_plan_with_origin(traj)
        # heading 0 → world = ego + [forward, lateral]
        world = np.stack([ego_xy[0] + traj[:, 1], ego_xy[1] + traj[:, 0]], axis=1)
        arc, speed = plan_execution.plan_pacing(world, self.SIM_DT)
        return world, arc, speed

    def test_cached_plan_starts_at_the_ego_pose(self) -> None:
        ego = np.array([12.0, -3.0])
        world, arc, _ = self._cache_plan(ego)
        np.testing.assert_allclose(world[0], ego, atol=1e-12)
        assert arc[0] == 0.0

    def test_first_executed_target_is_one_paced_step_from_the_ego(self) -> None:
        ego = np.array([12.0, -3.0])
        world, arc, speed = self._cache_plan(ego)
        target, _, velocity, next_s = plan_execution.teleport_plan_step(
            world, arc, arc_s=0.0, speed=speed, sim_dt=self.SIM_DT,
            current_position=ego, current_heading=0.0)
        step = float(np.linalg.norm(target - ego))
        assert step == pytest.approx(speed * self.SIM_DT, rel=1e-6)
        assert next_s == pytest.approx(speed * self.SIM_DT)
        # The reported velocity is that same step over sim_dt, not double it.
        assert float(np.linalg.norm(velocity)) == pytest.approx(speed, rel=1e-6)

    def test_pass_through_branch_is_also_anchored_at_the_ego(self) -> None:
        """``model_dt <= sim_dt`` returns the model's raw waypoints, whose
        [0] is at t=+model_dt — the identical off-by-one, covered by the
        seam anchor rather than by the interpolation branch."""
        ego = np.zeros(2)
        world, arc, speed = self._cache_plan(ego, model_dt=self.SIM_DT)
        np.testing.assert_allclose(world[0], ego, atol=1e-12)
        target, _, _, _ = plan_execution.teleport_plan_step(
            world, arc, arc_s=0.0, speed=speed, sim_dt=self.SIM_DT,
            current_position=ego, current_heading=0.0)
        assert float(np.linalg.norm(target - ego)) == pytest.approx(
            speed * self.SIM_DT, rel=1e-6)

    def _rollout(self, frames: int = 30) -> np.ndarray:
        """Per-frame displacement over a replanning teleport rollout."""
        pos = np.zeros(2)
        heading = 0.0
        world = arc = None
        speed = 0.0
        arc_s = 0.0
        displacement = []
        for frame in range(frames):
            if frame % self.REPLAN_RATE == 0:
                world, arc, speed = self._cache_plan(pos)
                arc_s = 0.0  # production resets the cursor at every replan
            prev = pos
            pos, heading, _, arc_s = plan_execution.teleport_plan_step(
                world, arc, arc_s, speed, self.SIM_DT, prev, heading)
            displacement.append(float(np.linalg.norm(pos - prev)))
        return np.asarray(displacement)

    def test_no_period_r_spike_in_per_frame_displacement(self) -> None:
        step = self._rollout()
        idx = np.arange(len(step))
        phase = np.array(
            [step[idx % self.REPLAN_RATE == r].mean()
             for r in range(self.REPLAN_RATE)])
        # Baseline runs measured 2.03 here; a uniform-speed plan replanned
        # from the executed pose must be flat to numerical precision.
        assert phase.max() / phase.min() < 1.01, f"phase means {phase}"

    def test_executed_pace_is_the_pacing_factor_times_plan_speed(self) -> None:
        """No compensating error left in the chain: mean executed speed is
        the documented fraction of the plan's own average speed."""
        _, arc, _ = self._cache_plan(np.zeros(2))
        plan_speed = float(arc[-1]) / ((len(arc) - 1) * self.SIM_DT)
        executed = self._rollout().mean() / self.SIM_DT
        assert executed == pytest.approx(
            plan_execution.PLAN_PACING_FACTOR * plan_speed, rel=1e-6)


class TestEgoPlanWithOrigin:
    def test_prepends_when_plan_starts_ahead_of_the_ego(self) -> None:
        traj = np.array([[0.0, 2.0], [0.0, 4.0]])
        out = plan_execution.ego_plan_with_origin(traj)
        np.testing.assert_allclose(out, [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])

    def test_idempotent_on_a_plan_already_at_the_origin(self) -> None:
        traj = np.array([[0.0, 0.0], [0.0, 2.0]])
        out = plan_execution.ego_plan_with_origin(traj)
        np.testing.assert_allclose(out, traj)
        np.testing.assert_allclose(
            plan_execution.ego_plan_with_origin(out), traj)

    def test_degenerate_inputs_pass_through(self) -> None:
        empty = np.zeros((0, 2))
        np.testing.assert_allclose(
            plan_execution.ego_plan_with_origin(empty), empty)
        one_d = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            plan_execution.ego_plan_with_origin(one_d), one_d)

    def test_no_zero_length_first_segment_after_composition(self) -> None:
        """A duplicated origin would give ``point_at_arc`` duplicate arc
        entries; the idempotence guard is what prevents it."""
        traj = plan_execution.interpolate_plan(
            np.stack([np.zeros(8), np.arange(1, 9) * 2.0], axis=1),
            model_dt=0.5, sim_dt=0.1)
        arc = plan_execution.cumulative_arc_length(
            plan_execution.ego_plan_with_origin(traj))
        assert float(np.diff(arc).min()) > 0.0


class TestInterpolatePlanOrigin:
    def test_resampled_plan_includes_the_t0_sample(self) -> None:
        # 8 waypoints at 0.5 s → 4.0 s horizon → 41 samples at 0.1 s,
        # the first of which is the ego origin.
        traj = np.stack([np.zeros(8), np.arange(1, 9) * 2.0], axis=1)
        out = plan_execution.interpolate_plan(traj, model_dt=0.5, sim_dt=0.1)
        assert len(out) == 41
        np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-12)
        # Spacing is the model speed (4 m/s → 0.4 m per 0.1 s step) from the
        # very first step: the ego is not one step ahead of its own plan.
        np.testing.assert_allclose(out[1], [0.0, 0.4], atol=1e-12)
        np.testing.assert_allclose(out[-1], traj[-1], atol=1e-12)


class TestControllerPlanInterpolation:
    @pytest.mark.parametrize("model_dt,sim_dt", [(0.0, 0.1), (0.1, 0.0)])
    def test_invalid_dt_is_rejected_before_pass_through(
            self, model_dt: float, sim_dt: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            plan_execution.interpolate_controller_plan(
                np.zeros((2, 2)), model_dt=model_dt, sim_dt=sim_dt)

    def test_lateral_reference_never_gains_an_origin_chord(self) -> None:
        # A one-metre projected lane sampled every 0.5 s.  Joining the ego
        # origin would create a fake diagonal; controller resampling must keep
        # every segment parallel to the planner's straight reference.
        raw = np.array([
            [-1.0, 1.0], [-1.0, 2.0], [-1.0, 3.0], [-1.0, 4.0],
        ])
        out = plan_execution.interpolate_controller_plan(
            raw, model_dt=0.5, sim_dt=0.1)

        assert len(out) == 20
        assert np.all(out[:, 0] == pytest.approx(-1.0))
        np.testing.assert_allclose(out[:5, 1], np.arange(1, 6) * 0.2)
        np.testing.assert_allclose(out[-1], raw[-1])
        headings = np.arctan2(np.diff(out[:, 0]), np.diff(out[:, 1]))
        np.testing.assert_allclose(headings, 0.0, atol=1e-12)

    def test_speed_profile_uses_the_same_future_time_base(self) -> None:
        speeds = np.array([[2.0], [3.0], [4.0]])
        out = plan_execution.interpolate_controller_plan(
            speeds, model_dt=0.5, sim_dt=0.1)[:, 0]

        assert len(out) == 15
        assert out[4] == pytest.approx(2.0)   # t=0.5
        assert out[9] == pytest.approx(3.0)   # t=1.0
        assert out[14] == pytest.approx(4.0)  # t=1.5

    def test_geometry_speed_uses_forward_tangent_not_lateral_chord(self) -> None:
        raw = np.array([
            [-1.0, 1.0], [-1.0, 2.0], [-1.0, 3.0], [-1.0, 4.0],
        ])
        speed = plan_execution.controller_speed_profile(
            raw, model_dt=0.5, sim_dt=0.1)

        # One metre of forward travel in each half second is 2 m/s.  The
        # persistent one-metre lateral projection contributes no fake speed.
        np.testing.assert_allclose(speed, 2.0)


class TestEgoOverrideNoDoubleIntegration:
    """Exercises the pure-Python step path (shared with the IsaacLab path's
    ego-override logic). ``_HAS_ISAACLAB`` is forced off so the env is
    constructable under pytest even on machines with the full stack installed
    (DirectRLEnv requires a booted sim app)."""

    @pytest.fixture(autouse=True)
    def _pure_python_env_path(self):
        with patch("navsafe.env.navsafe_env._HAS_ISAACLAB", False):
            yield

    def _stepped_env(self) -> NexusSimEnv:
        env = NexusSimEnv(EnvCfg())
        env.reset()
        return env

    def test_step_lands_exactly_on_override_pose(self) -> None:
        env = self._stepped_env()
        target = np.array([10.0, 20.0], dtype=np.float32)
        # Inject a large velocity: before the fix, step() re-integrated it
        # through the bicycle model, drifting speed*dt past the target.
        env.set_ego_override(
            position=target, heading=0.5,
            velocity=np.array([3.0, 1.0], dtype=np.float32),
        )
        env.step(None)
        state = env.get_ego_state()
        np.testing.assert_allclose(state["position"][:2], target, atol=1e-6)
        assert state["heading"] == pytest.approx(0.5)
        # Speed is preserved for the model's status feature / renderer.
        assert state["speed"] == pytest.approx(float(np.hypot(3.0, 1.0)))

    def test_consecutive_overrides_advance_one_waypoint_per_step(self) -> None:
        # Emulate the evaluator consuming an interpolated plan: each frame
        # teleports to the next 0.1s waypoint. Executed positions must match
        # the plan exactly (no overshoot frames, no frozen frames).
        env = self._stepped_env()
        plan = np.stack([np.linspace(0.5, 2.0, 4), np.zeros(4)], axis=1)
        prev = np.zeros(2)
        executed = []
        for wp in plan:
            vel = (wp - prev) / 0.1
            env.set_ego_override(
                position=wp.astype(np.float32), heading=0.0,
                velocity=np.array([vel[0], vel[1], 0.0], dtype=np.float32),
            )
            env.step(None)
            executed.append(env.get_ego_state()["position"][:2].copy())
            prev = wp
        np.testing.assert_allclose(np.asarray(executed), plan, atol=1e-6)

    def test_step_without_override_still_integrates(self) -> None:
        env = self._stepped_env()
        env._ego.reset(x=0.0, y=0.0, heading=0.0, speed=5.0)
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        state = env.get_ego_state()
        # Bicycle model must still advance the ego when no override is active.
        assert state["position"][0] == pytest.approx(0.5, abs=1e-3)


class TestExecutionModeConfig:
    """EvaluationConfig execution-mode axis (T1.2)."""

    def _cfg(self, **kw):
        from navsafe.evaluation.evaluator import EvaluationConfig
        return EvaluationConfig(**kw)

    def test_default_is_teleport(self) -> None:
        assert self._cfg().execution_mode == "teleport"

    def test_controller_and_physics_accepted(self) -> None:
        assert self._cfg(execution_mode="controller").execution_mode == "controller"
        assert self._cfg(execution_mode="physics").execution_mode == "physics"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="execution_mode"):
            self._cfg(execution_mode="warp_drive")

    def test_lqr_controller_type_accepted(self) -> None:
        assert self._cfg(controller_type="lqr").controller_type == "lqr"


class TestEnvExecutionModeConfig:
    """EnvCfg execution-mode axis (T2.1)."""

    def test_default_is_kinematic(self) -> None:
        assert EnvCfg().execution_mode == "kinematic"

    def test_physics_accepted_by_validation(self) -> None:
        from navsafe.env.navsafe_env import _validate_cfg
        _validate_cfg(EnvCfg(execution_mode="physics"))  # must not raise

    def test_invalid_mode_rejected(self) -> None:
        from navsafe.env.navsafe_env import _validate_cfg
        with pytest.raises(ValueError, match="execution_mode"):
            _validate_cfg(EnvCfg(execution_mode="antigravity"))


class TestEgoExternallyDriven:
    """set_ego_externally_driven marks the ego as action-driven (T1.2)."""

    @pytest.fixture(autouse=True)
    def _pure_python_env_path(self):
        with patch("navsafe.env.navsafe_env._HAS_ISAACLAB", False):
            yield

    def test_flag_set_and_reset(self) -> None:
        env = NexusSimEnv(EnvCfg())
        assert env._ego_externally_driven is False
        env.set_ego_externally_driven(True)
        assert env._ego_externally_driven is True
        env.set_ego_externally_driven(False)
        assert env._ego_externally_driven is False


class TestControllerModeRollout:
    """Controller execution advances the ego without teleporting."""


# TestPDMSScorerAtFaultParity was removed with metrics/pdms_scorer.py — NC
# at-fault behaviour is owned by the EPDMS live scorer
# (scorers/epdms_trajectory_scorer_fast.py::EPDMSLiveScorer._check_nc_live).


class TestAtFaultCollisionTermination:
    """Only ego-attributable collisions may terminate the episode: log-replay
    traffic does not react to the ego, so a rear-end by a follower must be
    recorded but not end the run (NavSim at-fault convention)."""

    @pytest.fixture(autouse=True)
    def _pure_python_env_path(self):
        with patch("navsafe.env.navsafe_env._HAS_ISAACLAB", False):
            yield

    def _agent(self, x: float, y: float, heading: float = 0.0) -> dict:
        return {"position": np.array([x, y, 0.0]), "heading": heading,
                "length": 4.5, "width": 1.8}

    def _env_with_agent(self, agent: dict) -> NexusSimEnv:
        env = NexusSimEnv(EnvCfg())
        env.reset()
        env._ego.reset(x=0.0, y=0.0, heading=0.0, speed=2.0)
        # Traffic manager is a no-op without scenario data, so the injected
        # agent list survives the step's agent update.
        env._agent_states = [agent]
        env._update_agents = lambda: None  # keep injected agents
        return env

    def test_rear_end_by_follower_records_but_does_not_terminate(self) -> None:
        env = self._env_with_agent(self._agent(-4.0, 0.0))  # behind, overlapping
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["collision"] is True
        assert info["collision_at_fault"] is False
        assert not terminated.any()

    def test_frontal_collision_terminates(self) -> None:
        env = self._env_with_agent(self._agent(4.0, 0.0))  # ahead, overlapping
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["collision"] is True
        assert info["collision_at_fault"] is True
        assert terminated.any()

    def test_contact_while_stopped_is_not_at_fault(self) -> None:
        env = self._env_with_agent(self._agent(4.0, 0.0))
        env._ego.speed = 0.0
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["collision"] is True
        assert info["collision_at_fault"] is False
        assert not terminated.any()

    def test_no_overlap_no_collision(self) -> None:
        env = self._env_with_agent(self._agent(20.0, 0.0))
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert info["collision"] is False
        assert not terminated.any()
