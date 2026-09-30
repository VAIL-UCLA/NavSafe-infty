# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative metadata discovery and strict anchors with a bounded cache."""

from collections import Counter, OrderedDict
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from navsafe.render.nurec_grpc import NuRecGrpcSceneRenderer
from tests.render.test_nurec_grpc_anchor_check import _Request, _traj_response


class _DiscoveryStub:
    def __init__(self, cache_size, first_trajectories=(), actor_failure=None):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.events = []
        self.loads = 0
        self.trajectory_calls = Counter()
        self.first_trajectories = list(first_trajectories)
        self.actor_failure = actor_failure

    def _touch(self, method, scene):
        self.events.append((method, scene))
        if scene not in self.cache:
            self.loads += 1
        self.cache[scene] = None
        self.cache.move_to_end(scene)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def get_available_cameras(self, req, timeout=None):
        self._touch("cameras", req.scene_id)
        return NS(available_cameras=[NS(logical_id="CAM_F0", intrinsics="fixture",
                                        rig_to_camera=None)])

    def restore_model_parameters(self, req, timeout=None):
        self._touch("reset", req.scene_id)

    def get_dynamic_objects(self, req, timeout=None):
        self._touch("actors", req.scene_id)
        if req.scene_id == self.actor_failure:
            raise RuntimeError("actor RPC unavailable")
        height = float(req.scene_id[1:])
        return NS(dynamic_objects=[NS(id="actor_"+req.scene_id, semantic_class="vehicle",
                                      object_size=NS(size_x=4.5, size_y=1.9, size_z=height))])

    def get_available_trajectories(self, req, timeout=None):
        self._touch("trajectory", req.scene_id)
        self.trajectory_calls[req.scene_id] += 1
        if req.scene_id == "s1" and self.first_trajectories:
            response = self.first_trajectories.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        return _traj_response([(10.0*(int(req.scene_id[1:])-1), 0.0)])


def _setup_renderer(tmp_path, monkeypatch, *, cache_size=1,
                    first_trajectories=(), actor_failure=None, scenes=("s1", "s2", "s3", "s4")):
    entries = []
    for i, scene in enumerate(scenes):
        offset = tmp_path/f"offset_{i}.json"
        offset.write_text(json.dumps({"offset_xy_utm": [100.0+10*i, 200.0]}))
        entries.append(f"{scene},{offset},{i*1000000},{(i+1)*1000000}")
    monkeypatch.setenv("NUREC_GRPC_HANDOFF", ";".join(entries))
    monkeypatch.setenv("NUREC_GRPC_ORIGIN_OFFSET_FILE", "")
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    monkeypatch.setenv("NUREC_GRPC_ANCHOR_TOL_M", "10")
    monkeypatch.setenv("NUREC_GRPC_CAM_RIG", "recon")
    monkeypatch.delenv("NUREC_GRPC_RENDER_STEPS", raising=False)
    renderer = NuRecGrpcSceneRenderer(timeout_s=1)
    stub = _DiscoveryStub(cache_size, first_trajectories, actor_failure)
    renderer._g = {
        "sspb": NS(AvailableTrajectoriesRequest=_Request, RestoreModelParametersRequest=_Request),
        "cpb": NS(), "grpc": NS(insecure_channel=lambda *a, **kw: None),
        "Stub": lambda channel: stub, "AvailableCamerasRequest": _Request,
        "AvailableDynamicObjectsRequest": _Request,
    }
    monkeypatch.setattr(renderer, "_read_sd_ego_z", lambda *a: 0.8)
    monkeypatch.setattr(renderer, "_read_sd_ego_track", lambda *a: (np.array([0.8]), np.array([[0., 0.]])))
    monkeypatch.setattr(renderer, "_read_sd_pos0", lambda *a: np.zeros(2))
    monkeypatch.setattr(renderer, "_pose_to_se3", lambda pose: np.eye(4))
    monkeypatch.setattr(renderer, "_insert_injected_assets", lambda data: stub.events.append(("insert", None)))
    monkeypatch.setattr(renderer, "_replace_harvested_assets", lambda: stub.events.append(("replace", None)))
    data = {"metadata": {"nurec_grpc_scene_id": "s1", "nurec_grpc_host": "fixture",
                          "scenario_origin_xy": [100.0, 200.0], "ts": [0.0, 0.1]}}
    return renderer, stub, data


@pytest.mark.parametrize("cache_size", [1, 10])
def test_discovery_keeps_actor_truth_with_one_load_per_scene(tmp_path, monkeypatch, cache_size):
    renderer, stub, data = _setup_renderer(tmp_path, monkeypatch, cache_size=cache_size)
    renderer.setup(data)
    assert stub.loads == 4
    assert renderer._dyn_track_ids_by_scene == {f"s{i}": {f"actor_s{i}"} for i in range(1, 5)}
    for i in range(1, 5):
        scene, actor = f"s{i}", f"actor_s{i}"
        assert stub.events.index(("reset", scene)) < stub.events.index(("actors", scene))
        assert renderer._dyn_track_size_by_scene[scene][actor] == (4.5, 1.9, float(i))
        assert renderer._dyn_track_half_h[actor] == i/2
        np.testing.assert_array_equal(renderer._handoff[i-1][2], [[10.0*(i-1), 0.0]])
    assert all(stub.events.index(("reset", f"s{i}")) < stub.events.index(("insert", None)) for i in range(1,5))
    assert stub.events[-2:] == [("insert", None), ("replace", None)]
    assert stub.trajectory_calls["s1"] == 1
    np.testing.assert_array_equal(renderer._origin_offset, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(renderer._eval_pos0, [0.0, 0.0])


def test_actor_failure_does_not_skip_scene_trajectory(tmp_path, monkeypatch):
    renderer, stub, data = _setup_renderer(tmp_path, monkeypatch, actor_failure="s2")
    renderer.setup(data)
    assert renderer._dyn_track_ids_by_scene["s2"] == set()
    assert stub.trajectory_calls["s2"] == 1
    np.testing.assert_array_equal(renderer._handoff[1][2], [[10.0, 0.0]])


def test_successful_anchor_retry_rejects_known_mismatch_before_edits(tmp_path, monkeypatch):
    renderer, stub, data = _setup_renderer(tmp_path, monkeypatch,
        first_trajectories=[RuntimeError("temporary"), _traj_response([(100.0, 0.0)])])
    with pytest.raises(RuntimeError, match="frame anchor MISMATCH"):
        renderer.setup(data)
    assert stub.trajectory_calls["s1"] == 2
    assert not any(method in {"insert", "replace"} for method, _ in stub.events)


def test_unavailable_anchor_and_retry_preserve_older_server_behavior(tmp_path, monkeypatch):
    renderer, stub, data = _setup_renderer(tmp_path, monkeypatch,
        first_trajectories=[RuntimeError("unavailable"), RuntimeError("unavailable")])
    renderer.setup(data)
    assert stub.trajectory_calls["s1"] == 2
    assert renderer._handoff[0][2] is None


def test_duplicate_scene_cannot_overwrite_first_successful_anchor(tmp_path, monkeypatch):
    renderer, stub, data = _setup_renderer(tmp_path, monkeypatch, scenes=("s1", "s2", "s1"),
        first_trajectories=[_traj_response([(100.0, 0.0)]), _traj_response([(0.0, 0.0)])])
    with pytest.raises(RuntimeError, match="frame anchor MISMATCH"):
        renderer.setup(data)
    assert stub.trajectory_calls["s1"] == 1
    assert not any(method in {"insert", "replace"} for method, _ in stub.events)
