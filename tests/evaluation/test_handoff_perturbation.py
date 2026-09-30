# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The hand-off perturbation, and the plan record that explains an arm.

A perturbation sweep asks one question — how much does WHERE the policy is
handed the car change what it plans? — and the answer is only readable if two
things hold:

  * the displacement is in the EGO's frame, so ``lateral=+1`` means "one metre
    to its left" on a north-bound and a west-bound scenario alike, and it lands
    exactly once, on the hand-off frame. A world-frame offset, or one re-applied
    every frame, would not be a change of initial condition at all;
  * the candidate set, the pick and the executed pose survive to disk as
    numbers. ``cam_f0_candidates.png`` already draws them, but a distribution
    plot cannot be made from pixels.

Both are geometry and bookkeeping, so both are tested here without a simulator.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from navsafe.evaluation.evaluator import EvaluationConfig, Evaluator


class _StubEnv:
    """Just enough env to answer ``get_ego_state`` and take an override."""

    def __init__(self, position=(10.0, 20.0, 0.0), heading=0.0,
                 velocity=(5.0, 0.0, 0.0)):
        self.state = {
            "position": np.asarray(position, dtype=np.float64),
            "heading": float(heading),
            "velocity": np.asarray(velocity, dtype=np.float64),
            "speed": float(np.linalg.norm(np.asarray(velocity)[:2])),
        }
        self.overrides: list = []

    def get_ego_state(self):
        return dict(self.state)

    def set_ego_override(self, position, heading, velocity=None):
        self.overrides.append({
            "position": np.asarray(position, dtype=np.float64).copy(),
            "heading": float(heading),
            "velocity": (None if velocity is None
                         else np.asarray(velocity, dtype=np.float64).copy()),
        })
        self.state["position"] = np.asarray(position, dtype=np.float64)
        self.state["heading"] = float(heading)
        if velocity is not None:
            self.state["velocity"] = np.asarray(velocity, dtype=np.float64)


class _StubAdapter:
    pass


def _evaluator(tmp_path, env=None, **perturb) -> Evaluator:
    cfg = EvaluationConfig(
        execution_mode="controller", controller_type="lqr",
        ego_replay_frames=20, output_dir=tmp_path, **perturb)
    return Evaluator(env=env or _StubEnv(), model_adapter=_StubAdapter(),
                     config=cfg)


class TestHandoffPerturbation:
    def test_lateral_is_the_egos_left_not_the_worlds_y(self, tmp_path):
        # Ego pointing due WEST: +1 m to its left is SOUTH (-y), which a
        # world-frame implementation would have put north.
        env = _StubEnv(position=(0.0, 0.0, 0.0), heading=math.pi,
                       velocity=(-5.0, 0.0, 0.0))
        ev = _evaluator(tmp_path, env, ego_perturb_lateral_m=1.0)
        ev.frame = 20
        ev._apply_handoff_perturbation()
        pos = env.overrides[-1]["position"]
        assert abs(pos[0]) < 1e-9
        assert pos[1] == pytest.approx(-1.0)

    def test_longitudinal_is_along_the_heading(self, tmp_path):
        env = _StubEnv(position=(0.0, 0.0, 0.0), heading=math.pi / 2,
                       velocity=(0.0, 5.0, 0.0))
        ev = _evaluator(tmp_path, env, ego_perturb_longitudinal_m=1.5)
        ev.frame = 20
        ev._apply_handoff_perturbation()
        pos = env.overrides[-1]["position"]
        assert abs(pos[0]) < 1e-9
        assert pos[1] == 1.5

    def test_yaw_rotates_the_velocity_with_the_body(self, tmp_path):
        # A ego yawed 90 deg whose velocity stayed in the world frame would be
        # handed to the policy travelling fully sideways.
        env = _StubEnv(position=(0.0, 0.0, 0.0), heading=0.0,
                       velocity=(5.0, 0.0, 0.0))
        ev = _evaluator(tmp_path, env, ego_perturb_yaw_deg=90.0)
        ev.frame = 20
        ev._apply_handoff_perturbation()
        ov = env.overrides[-1]
        assert ov["heading"] == math.pi / 2
        assert abs(ov["velocity"][0]) < 1e-9
        assert ov["velocity"][1] == 5.0
        # Speed is preserved: this moves and turns the car, it does not
        # accelerate it.
        assert np.linalg.norm(ov["velocity"][:2]) == 5.0

    def test_applied_once_even_if_the_frame_is_revisited(self, tmp_path):
        env = _StubEnv()
        ev = _evaluator(tmp_path, env, ego_perturb_lateral_m=1.0)
        ev.frame = 20
        ev._apply_handoff_perturbation()
        ev._apply_handoff_perturbation()
        assert len(env.overrides) == 1

    def test_zero_perturbation_touches_nothing(self, tmp_path):
        env = _StubEnv()
        ev = _evaluator(tmp_path, env)
        ev.frame = 20
        ev._apply_handoff_perturbation()
        assert env.overrides == []
        assert ev._perturb_record is None

    def test_record_names_the_pose_it_moved(self, tmp_path):
        env = _StubEnv(position=(10.0, 20.0, 0.5), heading=0.0)
        ev = _evaluator(tmp_path, env, ego_perturb_lateral_m=-0.5,
                        ego_perturb_yaw_deg=5.0)
        ev.frame = 20
        ev._apply_handoff_perturbation()
        rec = ev._perturb_record
        assert rec["frame"] == 20
        assert rec["lateral_m"] == -0.5 and rec["yaw_deg"] == 5.0
        assert rec["position_before"][:2] == [10.0, 20.0]
        assert rec["position_after"][1] == 19.5
        assert rec["heading_after"] == math.radians(5.0)


class TestPlanRecords:
    def test_plan_list_unwraps_a_batch_axis(self):
        arr = np.zeros((1, 3, 8, 2))
        assert np.asarray(Evaluator._plan_list(arr, 3)).shape == (3, 8, 2)

    def test_plan_list_refuses_wrong_rank_and_non_finite(self):
        assert Evaluator._plan_list(np.zeros((8, 2)), 3) is None
        assert Evaluator._plan_list(None, 2) is None
        bad = np.zeros((4, 2))
        bad[1, 1] = np.nan
        assert Evaluator._plan_list(bad, 2) is None

    def test_record_keeps_candidates_scores_and_the_pick(self, tmp_path):
        ev = _evaluator(tmp_path)
        ev.frame = 25
        ev._cached_world_traj = np.array([[0.0, 0.0], [1.0, 0.0]])
        parsed = {
            "trajectory": np.array([[0.0, 0.0], [0.0, 1.0]]),
            "trajectory_coarse": np.zeros((4, 8, 2)),
            "coarse_scores": np.array([0.1, 0.4, 0.2, 0.3]),
        }
        ev._record_plan(
            {"position": np.array([1.0, 2.0, 0.0]), "heading": 0.25},
            parsed, {"best_idx": np.array([1])}, parsed["trajectory"])
        rec = ev._plan_records[-1]
        assert rec["frame"] == 25 and rec["is_warmup"] is False
        assert np.asarray(rec["candidates_ego"]).shape == (4, 8, 2)
        assert rec["candidate_scores"] == [0.1, 0.4, 0.2, 0.3]
        assert rec["selected_index"] == 1
        assert rec["ego_heading"] == 0.25

    def test_single_trajectory_policy_records_no_candidates(self, tmp_path):
        ev = _evaluator(tmp_path)
        ev.frame = 5
        ev._cached_world_traj = np.array([[0.0, 0.0]])
        traj = np.array([[0.0, 0.0], [0.0, 1.0]])
        ev._record_plan({"position": np.zeros(3), "heading": 0.0},
                        {"trajectory": traj}, {}, traj)
        rec = ev._plan_records[-1]
        assert rec["candidates_ego"] is None
        assert rec["selected_index"] is None
        assert rec["is_warmup"] is True          # frame 5 < ego_replay_frames

    def test_saved_file_carries_the_arm_and_the_executed_path(self, tmp_path):
        env = _StubEnv()
        ev = _evaluator(tmp_path, env, ego_perturb_lateral_m=1.0)
        ev.scenario_id = "probe"
        ev.frame = 20
        ev._apply_handoff_perturbation()
        ev._history["vehicle_states"] = [
            {"position": np.array([0.0, 0.0, 0.0]), "heading": 0.0, "speed": 5.0},
            {"position": np.array([0.5, 0.0, 0.0]), "heading": 0.0, "speed": 5.0},
        ]
        ev._history["timestamps"] = [0.0, 0.1]
        ev._history["metrics"] = [{}, {"collision": True}]
        ev._save_plan_records(tmp_path)

        out = json.loads((tmp_path / "plan_records.json").read_text())
        assert out["handoff_frame"] == 20
        assert out["perturbation_requested"]["lateral_m"] == 1.0
        assert out["perturbation"]["frame"] == 20
        assert len(out["executed"]) == 2
        assert out["executed"][1]["collision"] is True
        assert out["executed"][1]["position"][0] == 0.5

    def test_unperturbed_arm_says_so_rather_than_omitting_the_field(self, tmp_path):
        ev = _evaluator(tmp_path)
        ev.scenario_id = "probe"
        ev._save_plan_records(tmp_path)
        out = json.loads((tmp_path / "plan_records.json").read_text())
        assert out["perturbation"] is None
        assert out["perturbation_requested"]["yaw_deg"] == 0.0
