# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed frame-anchor validation in the NuRec gRPC renderer.

A mis-anchored scenario (wrong/missing origin offset) renders valid non-black
frames from a camera kilometres off-scene — the black-frame strict guard never
fires. ``_validate_frame_anchor`` compares the computed frame-0 rig xy against
the server's baked rig trajectory and raises under ``NUREC_GRPC_STRICT``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from navsafe.render.nurec_grpc import NuRecGrpcSceneRenderer


def _traj_response(points):
    poses = [SimpleNamespace(pose=SimpleNamespace(
        vec=SimpleNamespace(x=float(x), y=float(y), z=-23.3)))
        for x, y in points]
    traj = SimpleNamespace(trajectory=SimpleNamespace(poses=poses))
    return SimpleNamespace(available_trajectories=[traj])


class _Request:
    def __init__(self, scene_id):
        self.scene_id = scene_id


def _renderer(traj_points, *, origin_offset, eval_pos0, ego_xy0,
              fail_rpc=False):
    r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    r._scene_id = "scene"
    r._timeout = 1.0
    r._g = {"sspb": SimpleNamespace(AvailableTrajectoriesRequest=_Request),
            "cpb": SimpleNamespace()}
    r._origin_offset = np.asarray(list(origin_offset) + [0.0], np.float64)
    r._eval_pos0 = (np.asarray(eval_pos0, np.float64)
                    if eval_pos0 is not None else None)
    r._sd_ego_xy_arr = (np.asarray([ego_xy0], np.float64)
                        if ego_xy0 is not None else None)

    class _Stub:
        @staticmethod
        def get_available_trajectories(req, timeout=None):
            if fail_rpc:
                raise RuntimeError("UNIMPLEMENTED")
            return _traj_response(traj_points)

    r._stub = _Stub()
    return r


TRAJ = [(867.2, 1669.5), (868.0, 1669.5), (869.0, 1669.6)]


def test_min_traj_dist_uses_whole_trajectory():
    d = NuRecGrpcSceneRenderer._min_traj_dist_m(
        np.asarray([869.0, 1669.6]), np.asarray(TRAJ))
    assert d == pytest.approx(0.0)


def test_correct_anchor_passes(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    # local scenario (pos0 = eval_pos0 = 0) with the city-frame offset
    # restored: frame-0 rig lands on the trajectory.
    r = _renderer(TRAJ, origin_offset=(-868.0, -1669.5),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0))
    r._validate_frame_anchor()  # must not raise


def test_missing_offset_fails_closed(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    # The regression: zero offset maps the camera to (0, 0), ~1.9 km away.
    r = _renderer(TRAJ, origin_offset=(0.0, 0.0),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0))
    with pytest.raises(RuntimeError, match="frame anchor MISMATCH"):
        r._validate_frame_anchor()


def test_mismatch_warns_when_not_strict(monkeypatch, caplog):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "0")
    r = _renderer(TRAJ, origin_offset=(0.0, 0.0),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0))
    with caplog.at_level("WARNING"):
        r._validate_frame_anchor()
    assert any("frame anchor MISMATCH" in m for m in caplog.messages)


def test_tolerance_env_override(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    monkeypatch.setenv("NUREC_GRPC_ANCHOR_TOL_M", "5000")
    r = _renderer(TRAJ, origin_offset=(0.0, 0.0),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0))
    r._validate_frame_anchor()  # 1.9 km < 5 km tolerance


def test_rpc_unavailable_skips_quietly(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    r = _renderer(TRAJ, origin_offset=(0.0, 0.0),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0), fail_rpc=True)
    r._validate_frame_anchor()  # older server vintage: no check, no crash


def test_empty_trajectory_skips(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    r = _renderer([], origin_offset=(0.0, 0.0),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0))
    r._validate_frame_anchor()


def test_mid_clip_scenario_start_passes(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_STRICT", "1")
    # Scenario cut from mid-clip: frame-0 sits on a LATER trajectory point.
    r = _renderer(TRAJ, origin_offset=(-869.0, -1669.6),
                  eval_pos0=(0.0, 0.0), ego_xy0=(0.0, 0.0))
    r._validate_frame_anchor()
