import json
from types import SimpleNamespace
import zipfile

import numpy as np

from navsafe.evaluation.reactivity_trace import ReactivityTraceRecorder


class Adapter:
    def get_waypoint_dt(self):
        return 0.5


class Env:
    def __init__(self):
        self._ego = SimpleNamespace(cfg=SimpleNamespace(
            ego_length=4.8, ego_width=2.0, wheelbase=2.8))
        # Match the minimal runtime/wrapper shape seen by the real probe:
        # AuthoredMotion's effective onset is an attribute even if the source
        # policy mapping is unavailable.
        motion = SimpleNamespace(
            policy={}, kind="lead_brake", onset=0.2, enabled=True)
        self.driver = SimpleNamespace(motion=motion, time_s=0.0)
        # Real edit environments create drivers lazily after recorder setup.
        self._traffic_manager = SimpleNamespace(drivers={})
        self.agent_states = [{
            "id": "lead", "type": "VEHICLE",
            "position": np.array([10.0, 0.0, 0.0]),
            "velocity": np.array([8.0, 0.0]), "heading": 0.0,
            "length": 4.5, "width": 1.9,
        }]


class Config:
    record_reactivity_trace = True
    reactivity_condition = "hazard"
    reactivity_group_id = "probe-triplet"
    reactivity_metadata = {"model_type": "probe", "checkpoint": "none"}
    reactivity_trace_filename = "reactivity_trace.zip"
    traffic_mode = "navsafe"
    execution_mode = "controller"
    controller_type = "lqr"
    sim_dt = 0.1
    replan_rate = 5
    ego_replay_frames = 0
    eval_frames = 2
    enable_vis = False


def _row(archive, name):
    return [json.loads(line) for line in archive.read(name).decode().splitlines()]


def test_trace_buffers_then_writes_one_archive(tmp_path):
    env = Env()
    evaluator = SimpleNamespace(
        config=Config(), adapter=Adapter(), env=env,
        scenario_id="probe", scenario_path=tmp_path/"arrow",
        scenario_data={"metadata": {"map_name": "probe-map"}, "map_features": {}},
        full_route=[(np.array([0.0, 0.0, 0.0]), 3, 0),
                    (np.array([20.0, 0.0, 0.0]), 3, 1)],
        _current_command=3,
    )
    evaluator.config.output_dir = tmp_path
    recorder = ReactivityTraceRecorder(evaluator)
    assert list(tmp_path.iterdir()) == []
    env._traffic_manager.drivers["lead"] = env.driver

    precise = np.array([[0.123456789, 1.987654321], [0.2, 3.0]])
    recorder.record_plan(
        frame=0, ego_state={"position": [0, 0, 0], "heading": 0},
        raw_traj_ego=precise, selected_world=precise,
        controller_world=precise, selected_speeds=[8.0, 7.5],
        controller_speeds=[8.0, 7.5], candidates_ego=precise[None],
        candidate_scores=[0.75], selected_index=0, emergency_brake=False,
        model_dt_s=0.5, inference_wall_s=0.0123)

    before = {"position": [0, 0, 0], "velocity": [5, 0],
              "speed": 5, "heading": 0}
    after = {"position": [0.5, 0, 0], "velocity": [4.5, 0],
             "speed": 4.5, "heading": 0.01}
    env.driver.time_s = 0.1
    recorder.record_step(
        frame=0, ego_before=before, ego_after=after,
        action=[0.1, -1.0, 0], control={"target_speed_mps": 5.0},
        info={}, frame_metrics={}, signal_hold=False,
        signal_violation=None, episode_done=False)
    env.driver.time_s = 0.2
    env.agent_states[0]["velocity"] = np.array([7.0, 0.0])
    recorder.record_step(
        frame=1, ego_before=after,
        ego_after={**after, "position": [0.9, 0, 0], "velocity": [4, 0], "speed": 4},
        action=[0.1, -1.0, 0], control={"target_speed_mps": 4.0},
        info={"collision": True},
        frame_metrics={"collision": True, "collision_at_fault": True,
                       "contact_detail": {"agent_id": "lead"}},
        signal_hold=False, signal_violation=None, episode_done=True)

    assert list(tmp_path.iterdir()) == []
    output = recorder.finalize(
        results={"total_frames": 2, "metrics": {"success": False}},
        termination={"reason": "contact_at_fault"})
    assert list(tmp_path.iterdir()) == [output]

    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert not any("image" in name or "mask" in name for name in names)
        assert {"manifest.json", "completion.json", "plans.jsonl",
                "ego_states.jsonl", "actor_states.jsonl"} <= names
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["storage"] == {
            "buffering": "memory_until_finalize", "pvc_writes": 1,
            "images_recorded": False}
        completion = json.loads(archive.read("completion.json"))
        assert completion["trace_complete"] is True
        assert completion["recorded_frames"] == 2
        assert completion["recorded_queries"] == completion["recorded_plans"] == 1
        plans = _row(archive, "plans.jsonl")
        assert plans[0]["selected_ego"][0][0] == 0.123456789
        onsets = [r for r in _row(archive, "intervention.jsonl")
                  if r["kind"] == "actual_onset"]
        assert onsets == [{"kind": "actual_onset", "actor_id": "lead",
                           "event_kind": "lead_brake",
                           "frame": 1, "actual_onset_s": 0.2,
                           "configured_onset_s": 0.2}]
        actors = _row(archive, "actor_states.jsonl")
        assert len(actors) == 2
        assert actors[-1]["acceleration"][0] == -10.0
        safety = _row(archive, "safety_events.jsonl")
        assert any(r["kind"] == "contact_begin" for r in safety)
