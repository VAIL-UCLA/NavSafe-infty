from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from navsafe.scenario.py123d_adapter import Py123DAdapterConfig, scenario_from_py123d_scene
from navsafe.scenario.py123d_training_extractor import (
    training_frame_from_py123d,
    training_frame_from_py123d_scene_api,
    training_scenario_from_py123d,
)
from navsafe.scenario.training_schema import NexusScenarioLog, NexusFrameState, TrainingScenarioLog


@dataclass
class FakeTimestamp:
    time_us: int


@dataclass
class FakeMetadata:
    name: str
    modality_type: str | None = None
    modality_id: str | None = None


class FakePayload:
    def __init__(self, timestamp, data):
        self.timestamp = timestamp
        self.data = data
        self.payload = data


class FakeMapAPI:
    def __init__(self):
        polygon = SimpleNamespace(array=np.asarray([[0, -2, 0], [20, -2, 0], [20, 2, 0], [0, 2, 0]], dtype=np.float32))
        self.objects = {
            "lane": {
                "lane_0": SimpleNamespace(
                    lane_type="surface_street",
                    lane_group_id="group_0",
                    left_lane_id=None,
                    right_lane_id=None,
                    predecessor_ids=[],
                    successor_ids=["lane_1"],
                    speed_limit_mps=12.0,
                    centerline=SimpleNamespace(array=np.asarray([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float32)),
                    left_boundary=SimpleNamespace(array=np.asarray([[0, 2, 0], [20, 2, 0]], dtype=np.float32)),
                    right_boundary=SimpleNamespace(array=np.asarray([[0, -2, 0], [20, -2, 0]], dtype=np.float32)),
                    polygon=polygon,
                )
            },
            "lane_group": {
                "group_0": SimpleNamespace(
                    lane_ids=["lane_0"],
                    intersection_id="intersection_0",
                    predecessor_ids=[],
                    successor_ids=[],
                    polygon=polygon,
                )
            },
            "intersection": {
                "intersection_0": SimpleNamespace(
                    intersection_type="traffic_light",
                    lane_group_ids=["group_0"],
                    polygon=polygon,
                )
            },
            "stop_zone": {
                "stop_0": SimpleNamespace(stop_zone_type="traffic_light", lane_ids=["lane_0"], polygon=polygon)
            },
            "road_line": {
                "line_0": SimpleNamespace(road_line_type="dashed_white", polyline=SimpleNamespace(array=np.asarray([[0, 1, 0], [20, 1, 0]], dtype=np.float32)))
            },
        }

    def get_map_metadata(self):
        return FakeMetadata("map")

    def get_available_map_layers(self):
        return list(self.objects)

    def get_all_map_object_ids_in_layer(self, layer):
        return list(self.objects[str(layer)])

    def get_map_object_in_layer(self, object_id, layer):
        return self.objects[str(layer)][str(object_id)]


class FakeTrainingScene:
    number_of_iterations = 2
    number_of_history_iterations = 0
    dataset = "nuplan"
    split = "train"
    location = "fake_city"
    log_name = "fake_log"
    scene_uuid = "fake_scene"

    def __init__(self):
        self.timestamps = [FakeTimestamp(0), FakeTimestamp(100_000)]

    def get_scene_metadata(self):
        return FakeMetadata("scene")

    def get_log_metadata(self):
        return FakeMetadata("log")

    def get_map_metadata(self):
        return FakeMetadata("map")

    def get_map_api(self):
        return FakeMapAPI()

    def get_all_iteration_timestamps(self, include_history=False):
        return self.timestamps

    def get_all_modality_metadatas(self):
        return {
            "ego_state_se3": FakeMetadata("ego", "ego_state_se3"),
            "box_detections_se3": FakeMetadata("boxes", "box_detections_se3"),
            "traffic_light_detections": FakeMetadata("tls", "traffic_light_detections"),
            "custom.scenario": FakeMetadata("scenario", "custom"),
        }

    def get_all_modality_timestamps(self, *args):
        return self.timestamps

    def get_ego_state_se3_metadata(self):
        return FakeMetadata("ego", "ego_state_se3")

    def get_all_ego_state_se3_timestamps(self, include_history=False):
        return self.timestamps

    def get_ego_state_se3_at_iteration(self, iteration):
        pose = SimpleNamespace(x=float(iteration), y=0.0, z=0.0, yaw=0.1 * iteration)
        velocity = SimpleNamespace(x=5.0 + iteration, y=0.0, z=0.0)
        accel = SimpleNamespace(x=1.0, y=0.0, z=0.0)
        angular = SimpleNamespace(x=0.0, y=0.0, z=0.2)
        dyn = SimpleNamespace(velocity_3d=velocity, acceleration_3d=accel, angular_velocity=angular)
        bbox = SimpleNamespace(length=5.0, width=2.0, height=1.5)
        return SimpleNamespace(
            timestamp=self.timestamps[iteration],
            center_se3=pose,
            dynamic_state_se3=dyn,
            bounding_box_se3=bbox,
            tire_steering_angle=0.05,
        )

    def get_box_detections_se3_metadata(self):
        return FakeMetadata("boxes", "box_detections_se3")

    def get_all_box_detections_se3_timestamps(self, include_history=False):
        return self.timestamps

    def get_box_detections_se3_at_iteration(self, iteration):
        actor = SimpleNamespace(
            center_se3=SimpleNamespace(x=10.0, y=1.0, z=0.0, yaw=0.0),
            bounding_box_se3=SimpleNamespace(length=4.5, width=1.8, height=1.6),
            velocity_3d=SimpleNamespace(x=3.0, y=0.0, z=0.0),
            attributes=SimpleNamespace(track_token="actor_0", label="vehicle", num_lidar_points=8),
        )
        return SimpleNamespace(timestamp=self.timestamps[iteration], box_detections=[actor])

    def get_traffic_light_detections_metadata(self):
        return FakeMetadata("tls", "traffic_light_detections")

    def get_all_traffic_light_detections_timestamps(self, include_history=False):
        return self.timestamps

    def get_traffic_light_detections_at_iteration(self, iteration):
        return SimpleNamespace(timestamp=self.timestamps[iteration], detections=[SimpleNamespace(lane_id="lane_0", status="red")])

    def get_camera_metadatas(self):
        return {}

    def get_lidar_metadatas(self):
        return {}

    def get_all_custom_modality_metadatas(self):
        return {
            "scenario": FakeMetadata("scenario", "custom", "scenario"),
            "validity": FakeMetadata("validity", "custom", "validity"),
        }

    def get_all_custom_modality_timestamps(self, *args):
        return self.timestamps

    def get_custom_modality_at_iteration(self, iteration, modality_id):
        if str(modality_id) == "scenario":
            return FakePayload(self.timestamps[iteration], {"lidar_token": f"lidar_{iteration}", "route_roadblock_ids": ["rb0"]})
        if str(modality_id) == "validity":
            return FakePayload(self.timestamps[iteration], {"sdc_valid": True, "frame_index": iteration})
        return None


def test_py123d_training_extractor_materializes_training_log():
    py123d_scenario = scenario_from_py123d_scene(FakeTrainingScene(), Py123DAdapterConfig(require_map=True))
    training_log = training_scenario_from_py123d(py123d_scenario)

    assert isinstance(training_log, NexusScenarioLog)
    assert isinstance(training_log, TrainingScenarioLog)
    assert training_log.scenario_id == "fake_scene"
    assert training_log.number_of_iterations == 2
    assert training_log.route_supported
    assert set(training_log.map_state.available_layers) == {"lane", "lane_group", "intersection", "stop_zone", "road_line"}
    assert training_log.map_state.lanes["lane_0"].speed_limit_mps == 12.0
    assert training_log.map_state.stop_zones["stop_0"].semantic_type == "traffic_light"

    frame = training_log.frame(0)
    assert frame.ego.x == 0.0
    assert frame.actors[0].track_token == "actor_0"
    assert frame.actors[0].label == "vehicle"
    assert frame.traffic_lights[0].status == "red"
    assert frame.route.roadblock_ids == ("rb0",)
    assert frame.route.source_key == "custom.scenario.route_roadblock_ids"
    assert frame.sensor_references["custom.scenario.lidar_token"].metadata["lidar_token"] == "lidar_0"
    assert frame.custom_modalities["custom.scenario"]["route_roadblock_ids"] == ["rb0"]
    assert frame.custom_modalities["custom.validity"] == {"sdc_valid": True, "frame_index": 0}

    future = training_log.future_ego_trajectory(0, horizon=4)
    assert future.shape == (1, 3)
    assert np.allclose(future[0, :2], [1.0, 0.0])

    action = training_log.approximate_action(0)
    assert action is not None
    assert action.approximate
    assert action.accel_mps2 > 0.0


def test_windowed_runtime_projection_uses_loaded_absolute_frames():
    """Regression: [1,3) used to ask for the unloaded ego at frame zero."""
    from navsafe.scenario.py123d_scenario_description import py123d_to_scenario_description

    scene = FakeTrainingScene()
    scene.number_of_iterations = 3
    scene.timestamps.append(FakeTimestamp(200_000))
    sparse = scenario_from_py123d_scene(
        scene, Py123DAdapterConfig(require_map=True, frame_window=(1, 3)))
    assert sparse.get_frame_state(0).ego_state is None
    assert sparse.get_frame_state(1).ego_state is not None
    log = training_scenario_from_py123d(sparse)
    assert len(log.frames) == 2
    assert log.frames[0].iteration == 0
    assert log.frames[0].ego.x == 1.0
    assert log.timestamps_us == [100_000, 200_000]
    assert log.source_metadata["frame_window"] == [1, 3]
    description = py123d_to_scenario_description(sparse)
    assert description["length"] == 2
    assert description["metadata"]["ts"].tolist() == [100_000, 200_000]


def test_prefix_window_does_not_project_unloaded_tail():
    sparse = scenario_from_py123d_scene(
        FakeTrainingScene(), Py123DAdapterConfig(require_map=True, frame_window=(0, 1)))
    log = training_scenario_from_py123d(sparse)
    assert len(log.frames) == 1
    assert log.timestamps_us == [0]


def test_py123d_training_frame_can_be_extracted_without_materializing_whole_log():
    py123d_scenario = scenario_from_py123d_scene(FakeTrainingScene(), Py123DAdapterConfig(require_map=True))
    frame = training_frame_from_py123d(py123d_scenario, 1)

    assert isinstance(frame, NexusFrameState)
    assert frame.iteration == 1
    assert frame.ego.speed == 6.0
    assert len(frame.actors) == 1
    assert len(frame.traffic_lights) == 1


def test_py123d_streaming_frame_reads_all_custom_modalities():
    scene = FakeTrainingScene()
    frame = training_frame_from_py123d_scene_api(scene, "fake_scene", 0)

    assert frame.route.roadblock_ids == ("rb0",)
    assert frame.sensor_references["custom.scenario.lidar_token"].metadata["lidar_token"] == "lidar_0"
    assert frame.custom_modalities["custom.scenario"]["route_roadblock_ids"] == ["rb0"]
    assert frame.custom_modalities["custom.validity"] == {"sdc_valid": True, "frame_index": 0}


# ─────────────────────── missing-required-field telemetry ───────────────────────

import logging

import pytest

from navsafe.scenario import py123d_training_extractor as extractor_module
from navsafe.scenario.py123d_training_extractor import (
    _extract_actor,
    _extract_ego,
    collect_missing_fields,
    missing_field_totals,
    reset_missing_field_state,
)


@pytest.fixture(autouse=True)
def _reset_missing_field_state():
    reset_missing_field_state()
    yield
    reset_missing_field_state()


def _ego(**overrides):
    pose = SimpleNamespace(x=10.0, y=5.0, z=0.1, yaw=0.3)
    fields = dict(
        center_se3=pose,
        rear_axle_se3=pose,
        bounding_box_se3=SimpleNamespace(length=4.7, width=1.9, height=1.6),
        timestamp=SimpleNamespace(time_us=123),
        dynamic_state_se3=SimpleNamespace(
            velocity_3d=SimpleNamespace(x=3.0, y=0.5, z=0.0),
            acceleration_3d=SimpleNamespace(x=0.1, y=0.0, z=0.0),
            angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.02),
        ),
        tire_steering_angle=0.05,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_well_formed_ego_counts_nothing() -> None:
    with collect_missing_fields() as missing:
        state = _extract_ego(_ego())
    assert dict(missing) == {}
    assert missing_field_totals() == {}
    assert state.vx == 3.0 and state.vy == 0.5


def test_missing_velocity_is_counted_and_still_defaults_to_zero(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=extractor_module.__name__):
        with collect_missing_fields() as missing:
            state = _extract_ego(_ego(dynamic_state_se3=None))
    # Values unchanged (telemetry only): still the documented 0.0 default.
    assert state.vx == 0.0 and state.vy == 0.0
    assert missing["ego.vx"] == 1 and missing["ego.vy"] == 1
    # Optional fields (vz/accel/yaw_rate) missing too, but NOT counted.
    assert "ego.vz" not in missing and "ego.ax" not in missing and "ego.yaw_rate" not in missing
    assert any("ego.vx" in rec.message for rec in caplog.records)


def test_nan_required_field_counts_as_missing() -> None:
    ego = _ego()
    ego.dynamic_state_se3.velocity_3d.x = float("nan")
    with collect_missing_fields() as missing:
        state = _extract_ego(ego)
    assert state.vx == 0.0
    assert missing["ego.vx"] == 1


def test_warns_once_per_field_but_counts_every_occurrence(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=extractor_module.__name__):
        for _ in range(3):
            _extract_ego(_ego(dynamic_state_se3=None))
    assert missing_field_totals()["ego.vx"] == 3
    vx_warnings = [rec for rec in caplog.records if "ego.vx" in rec.message]
    assert len(vx_warnings) == 1


def test_per_extraction_counts_do_not_leak_across_contexts() -> None:
    with collect_missing_fields() as first:
        _extract_ego(_ego(dynamic_state_se3=None))
    with collect_missing_fields() as second:
        _extract_ego(_ego())
    assert first["ego.vx"] == 1
    assert dict(second) == {}
    # Process-wide totals keep accumulating regardless of contexts.
    assert missing_field_totals()["ego.vx"] == 1


def test_actor_without_velocity_counts_actor_fields() -> None:
    actor = SimpleNamespace(
        bounding_box_se3=SimpleNamespace(
            center_se3=SimpleNamespace(x=1.0, y=2.0, z=0.0, yaw=0.1), length=4.0, width=2.0, height=1.5
        ),
        center_se3=SimpleNamespace(x=1.0, y=2.0, z=0.0, yaw=0.1),
        attributes=SimpleNamespace(track_token="tok", label="vehicle", num_lidar_points=10),
    )
    with collect_missing_fields() as missing:
        state = _extract_actor(actor)
    assert state.vx == 0.0 and state.vy == 0.0
    assert missing["actor.vx"] == 1 and missing["actor.vy"] == 1
    assert "actor.x" not in missing


def test_scalar_se2_angular_velocity_becomes_yaw_rate() -> None:
    # SE2 dynamic states carry angular velocity as a bare float; it must reach
    # yaw_rate instead of being zeroed by a ``.z`` lookup on a float.
    ego = _ego()
    ego.dynamic_state_se3.angular_velocity = 0.25
    state = _extract_ego(ego)
    assert state.yaw_rate == 0.25
