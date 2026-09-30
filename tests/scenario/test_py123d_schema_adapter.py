from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import navsafe.scenario.py123d_adapter as py123d_adapter
from navsafe.scenario.py123d_adapter import Py123DAdapterConfig, scenario_from_py123d_scene
from navsafe.scenario.py123d_schema import Py123DScenarioData, to_serializable


@dataclass
class FakeTimestamp:
    time_us: int


@dataclass
class FakeMetadata:
    name: str
    modality_type: str | None = None
    modality_id: str | None = None

    def to_dict(self):
        return {"name": self.name, "modality_type": self.modality_type, "modality_id": self.modality_id}


@dataclass
class FakePayload:
    timestamp: FakeTimestamp
    payload: object


class FakeMapAPI:
    def get_map_metadata(self):
        return FakeMetadata("map")

    def get_available_map_layers(self):
        return ["lane", "stop_zone"]

    def get_all_map_object_ids_in_layer(self, layer):
        return {"lane": ["lane_0", "lane_1"], "stop_zone": ["stop_0"]}.get(str(layer), [])

    def get_map_object_in_layer(self, object_id, layer):
        return {"object_id": object_id, "layer": layer, "geometry": np.array([1.0, 2.0])}




class FakeEmptyMapAPI:
    def get_map_metadata(self):
        return FakeMetadata("empty_map")

    def get_available_map_layers(self):
        return []

class FakeSceneAPI:
    number_of_iterations = 2
    number_of_history_iterations = 0
    dataset = "fake_dataset"
    split = "train"
    location = "fake_city"
    log_name = "fake_log"
    scene_uuid = "fake_scene"

    def __init__(self):
        self.timestamps = [FakeTimestamp(100), FakeTimestamp(200)]

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
            "custom.scenario": FakeMetadata("scenario_custom_raw", "custom"),
            "future_modality": FakeMetadata("future", "future_modality"),
        }

    def get_all_modality_timestamps(self, modality_type, modality_id=None, include_history=False):
        return self.timestamps

    def get_ego_state_se3_metadata(self):
        return FakeMetadata("ego", "ego_state_se3")

    def get_all_ego_state_se3_timestamps(self, include_history=False):
        return self.timestamps

    def get_ego_state_se3_at_iteration(self, iteration):
        return FakePayload(self.timestamps[iteration], {"x": iteration})

    def get_box_detections_se3_metadata(self):
        return FakeMetadata("boxes", "box_detections_se3")

    def get_all_box_detections_se3_timestamps(self, include_history=False):
        return self.timestamps

    def get_box_detections_se3_at_iteration(self, iteration):
        payload = FakePayload(self.timestamps[iteration], [f"agent_{iteration}"])
        payload.box_detections = payload.payload
        return payload

    def get_traffic_light_detections_metadata(self):
        return FakeMetadata("tls", "traffic_light_detections")

    def get_all_traffic_light_detections_timestamps(self, include_history=False):
        return self.timestamps

    def get_traffic_light_detections_at_iteration(self, iteration):
        payload = FakePayload(self.timestamps[iteration], [f"tl_{iteration}"])
        payload.detections = payload.payload
        return payload

    def get_camera_metadatas(self):
        return {"front": FakeMetadata("front_camera", "camera", "front")}

    def get_all_camera_timestamps(self, camera_id, include_history=False):
        return self.timestamps

    def get_camera_at_iteration(self, iteration, camera_id):
        payload = FakePayload(self.timestamps[iteration], np.zeros((2, 3, 3), dtype=np.uint8))
        payload.image = payload.payload
        return payload

    def get_lidar_metadatas(self):
        return {"top": FakeMetadata("top_lidar", "lidar", "top")}

    def get_all_lidar_timestamps(self, lidar_id, include_history=False):
        return self.timestamps

    def get_lidar_at_iteration(self, iteration, lidar_id):
        payload = FakePayload(self.timestamps[iteration], np.zeros((4, 3), dtype=np.float32))
        payload.points = payload.payload
        return payload

    def get_all_custom_modality_metadatas(self):
        return {"scenario": FakeMetadata("scenario_custom", "custom", "scenario")}

    def get_all_custom_modality_timestamps(self, modality_id, include_history=False):
        return self.timestamps

    def get_custom_modality_at_iteration(self, iteration, modality_id):
        return FakePayload(self.timestamps[iteration], {"route_roadblock_ids": ["rb0", "rb1"]})


class FakeSceneWithEmptyMapAPI(FakeSceneAPI):
    def get_map_api(self):
        return FakeEmptyMapAPI()


def test_py123d_adapter_preserves_all_discovered_modalities_without_loading_sensors():
    scenario = scenario_from_py123d_scene(FakeSceneAPI())

    assert isinstance(scenario, Py123DScenarioData)
    assert scenario.dataset == "fake_dataset"
    assert scenario.scene_uuid == "fake_scene"
    assert scenario.timestamps_us == [100, 200]
    assert set(scenario.modality_keys) == {
        "box_detections_se3",
        "camera:front",
        "custom:scenario",
        "ego_state_se3",
        "future_modality",
        "lidar:top",
        "traffic_light_detections",
    }

    assert scenario.get_modality("ego_state_se3").loaded_iterations == [0, 1]
    assert scenario.get_modality("box_detections_se3").frame(1).data.payload == ["agent_1"]
    assert scenario.get_modality("traffic_light_detections").frame(0).data_summary["num_detections"] == 1
    assert scenario.get_modality("custom:scenario").frame(0).data.payload["route_roadblock_ids"] == ["rb0", "rb1"]

    assert scenario.get_modality("camera:front").loaded_iterations == []
    assert scenario.get_modality("lidar:top").loaded_iterations == []
    assert scenario.get_modality("custom:scenario").raw_reader_info["source_key"] == "custom.scenario"
    assert scenario.get_modality("future_modality").metadata.name == "future"

    assert scenario.map.available_layers == ["lane", "stop_zone"]
    assert scenario.map.object_ids_by_layer["lane"] == ["lane_0", "lane_1"]
    assert scenario.map.get_object("stop_zone", "stop_0")["object_id"] == "stop_0"

    frame = scenario.get_frame_state(1)
    assert frame.ego_state.payload == {"x": 1}
    assert frame.cameras == {}
    assert frame.lidars == {}
    assert frame.custom_modalities["scenario"].payload["route_roadblock_ids"] == ["rb0", "rb1"]

    scenario.validate()


def test_py123d_adapter_can_load_sensor_payloads_when_requested():
    scenario = scenario_from_py123d_scene(
        FakeSceneAPI(), Py123DAdapterConfig(load_sensor_payloads=True)
    )

    assert scenario.get_modality("camera:front").loaded_iterations == [0, 1]
    assert scenario.get_modality("lidar:top").loaded_iterations == [0, 1]

    frame = scenario.get_frame_state(0)
    assert frame.cameras["front"].payload.shape == (2, 3, 3)
    assert frame.lidars["top"].payload.shape == (4, 3)


def test_py123d_schema_summary_is_json_ready_and_does_not_expose_raw_scene():
    scenario = scenario_from_py123d_scene(FakeSceneAPI())
    summary = scenario.to_summary_dict()

    assert summary["scene_uuid"] == "fake_scene"
    assert summary["modalities"]["camera:front"]["num_timestamps"] == 2
    assert "raw_scene_api" not in summary
    assert to_serializable(np.array([1, 2])).copy() == [1, 2]


def test_py123d_adapter_uses_explicit_map_fallback_when_scene_map_is_empty(monkeypatch, tmp_path):
    def fake_loader(data_root, dataset, location):
        assert data_root == str(tmp_path)
        assert dataset == "fake_dataset"
        assert location == "fake_city"
        return FakeMapAPI()

    monkeypatch.setattr(py123d_adapter, "_load_explicit_map_api", fake_loader)

    scenario = scenario_from_py123d_scene(
        FakeSceneWithEmptyMapAPI(),
        Py123DAdapterConfig(data_root=str(tmp_path)),
    )

    assert scenario.map.available_layers == ["lane", "stop_zone"]
    assert scenario.map.object_ids_by_layer["lane"] == ["lane_0", "lane_1"]


def test_py123d_adapter_fails_fast_when_required_map_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(py123d_adapter, "_load_explicit_map_api", lambda *args: None)

    with pytest.raises(RuntimeError, match="py123d map API is unavailable") as exc_info:
        scenario_from_py123d_scene(
            FakeSceneWithEmptyMapAPI(),
            Py123DAdapterConfig(data_root=str(tmp_path)),
        )

    message = str(exc_info.value)
    assert "fake_dataset" in message
    assert "fake_city" in message
    assert "fake_log" in message
    assert "maps/fake_dataset/fake_dataset_fake_city.arrow" in message
