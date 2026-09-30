# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NuRecGrpcSceneRenderer's edit_assets insertion (Level B).

The real gRPC stubs are vendored only in the eval environment, so these tests
drive :meth:`_insert_injected_assets` / :meth:`close` against a minimal fake
proto bundle + stub that mimics the NRE message shapes (DESCRIPTOR.fields with
name / message_type / label), verifying the client-side translation:

* which scenario tracks are selected (``metadata.nurec_asset_id``),
* id-collision skip against server-exposed tracks,
* trajectory poses at the scene's first/last timestamps,
* ground-z pinning for injected tracks (agent_states clamps z to 0),
* inserted ids joining ``_dyn_track_ids`` (per-frame authoritative mirror),
* the metric PLY being pre-divided by the scale the server will apply,
* ``restore_model_parameters`` cleanup on close.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from navsafe.render.ply_utils import load_ply
from navsafe.render.nurec_grpc import NuRecGrpcSceneRenderer
from tests.fixtures.gaussian_ply import write_min_ply


# ---------------------------------------------------------------------------
# Fake proto machinery (shape-compatible with generated protobuf classes)
# ---------------------------------------------------------------------------

class _FakeMsgType:
    def __init__(self, name: str):
        self.name = name


class _FakeField:
    LABEL_OPTIONAL = 1
    LABEL_REPEATED = 3

    def __init__(self, name, message_type=None, repeated=False):
        self.name = name
        self.message_type = _FakeMsgType(message_type) if message_type else None
        self.label = self.LABEL_REPEATED if repeated else self.LABEL_OPTIONAL


def _make_msg_cls(cls_name: str, fields):
    desc = types.SimpleNamespace(fields=fields)

    class Msg:
        DESCRIPTOR = desc

        def __init__(self, **kw):
            for f in desc.fields:
                setattr(self, f.name, [] if f.label == _FakeField.LABEL_REPEATED
                        else None)
            for k, v in kw.items():
                if not any(f.name == k for f in desc.fields):
                    raise ValueError(f"{cls_name} has no field {k!r}")
                setattr(self, k, v)

        def __repr__(self):
            vals = {f.name: getattr(self, f.name) for f in desc.fields}
            return f"{cls_name}({vals})"

    Msg.__name__ = cls_name
    return Msg


class _Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _Pose:
    def __init__(self, vec=None, quat=None):
        self.vec, self.quat = vec, quat


def _fake_bundle():
    """Fake sensorsim/common stub modules mirroring the probed NRE shapes."""
    # Field names/shapes mirror the REAL vendored NRE stubs (verified via
    # DESCRIPTOR introspection on the NRE stubs).
    PoseAtTime = _make_msg_cls("PoseAtTime", [
        _FakeField("pose", message_type="Pose"),
        _FakeField("timestamp_us"),                  # scalar timestamp
    ])
    Trajectory = _make_msg_cls("Trajectory", [
        _FakeField("poses", message_type="PoseAtTime", repeated=True),
    ])
    AABB = _make_msg_cls("AABB", [
        _FakeField("size_x"), _FakeField("size_y"), _FakeField("size_z"),
    ])
    DynamicObjectTrack = _make_msg_cls("DynamicObjectTrack", [
        _FakeField("id"), _FakeField("semantic_class"),
        _FakeField("trajectory", message_type="Trajectory"),
        _FakeField("object_size", message_type="AABB"),
        _FakeField("asset_id"),
    ])
    ReplaceAssetAction = _make_msg_cls("ReplaceAssetAction", [
        _FakeField("original_id"), _FakeField("replacement_id"),
        _FakeField("object_size", message_type="AABB"),
    ])
    EditAssetsRequest = _make_msg_cls("EditAssetsRequest", [
        _FakeField("scene_id"),
        _FakeField("replace", message_type="ReplaceAssetAction", repeated=True),
        _FakeField("insert", message_type="DynamicObjectTrack", repeated=True),
    ])
    RestoreModelParametersRequest = _make_msg_cls(
        "RestoreModelParametersRequest", [_FakeField("scene_id")])
    AvailableDynamicObjectsRequest = _make_msg_cls(
        "AvailableDynamicObjectsRequest", [_FakeField("scene_id")])

    sspb = types.SimpleNamespace(
        PoseAtTime=PoseAtTime, Trajectory=Trajectory, AABB=AABB,
        DynamicObjectTrack=DynamicObjectTrack,
        ReplaceAssetAction=ReplaceAssetAction,
        EditAssetsRequest=EditAssetsRequest,
        RestoreModelParametersRequest=RestoreModelParametersRequest,
        AvailableDynamicObjectsRequest=AvailableDynamicObjectsRequest)
    cpb = types.SimpleNamespace()
    return dict(sspb=sspb, cpb=cpb, Pose=_Pose, Vec3=_Vec3, Quat=_Quat,
                AvailableDynamicObjectsRequest=AvailableDynamicObjectsRequest)


class _FakeStub:
    def __init__(self):
        self.edit_requests = []
        self.restore_requests = []
        # ids reported by get_dynamic_objects AFTER a restore (the baked set)
        self.baked_ids = ["baked_actor_0"]

    def edit_assets(self, req, timeout=None):
        self.edit_requests.append(req)
        return "ok"

    def restore_model_parameters(self, req, timeout=None):
        self.restore_requests.append(req)
        return "ok"

    def get_dynamic_objects(self, req, timeout=None):
        return types.SimpleNamespace(dynamic_objects=[
            types.SimpleNamespace(id=i, track_id=None, semantic_class="X.T_V")
            for i in self.baked_ids])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _renderer(sem_counter=None) -> NuRecGrpcSceneRenderer:
    r = NuRecGrpcSceneRenderer()
    r._g = _fake_bundle()
    r._stub = _FakeStub()
    r._scene_id = "test_scene"
    r._timestamps_us = [1_000_000, 1_100_000, 1_200_000]
    r._origin_offset = np.asarray([100.0, 200.0, 0.0], np.float32)
    r._eval_pos0 = np.zeros(2, np.float64)
    r._dyn_track_ids = {"baked_actor_0"}
    r._dyn_track_ids_by_scene = {"test_scene": {"baked_actor_0"}}
    # semantic classes observed on the scene's own tracks (live servers only
    # accept these enum literals; see _resolve_semantic_class)
    r._dyn_sem_counter = sem_counter if sem_counter is not None else {
        "WODPerceptionBoxDetectionLabel.TYPE_VEHICLE": 90,
        "WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN": 30,
    }
    return r


def _cone_track(pos=(30.0, 0.0, 53.3), asset_id="/assets/cone_3dgs.ply"):
    T = 3
    meta = {"injected_obstacle": True, "obstacle_archetype": "cone",
            "type": "TRAFFIC_CONE", "nurec_semantic_class": "cone"}
    if asset_id:
        meta["nurec_asset_id"] = asset_id
    return {
        "type": "TRAFFIC_CONE",
        "state": {
            "position": np.tile(np.asarray(pos, np.float32), (T, 1)),
            "heading": np.zeros(T, np.float32),
            "length": np.full(T, 0.4, np.float32),
            "width": np.full(T, 0.4, np.float32),
            "height": np.full(T, 0.7, np.float32),
            "valid": np.ones(T, bool),
        },
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInsertInjectedAssets:
    def test_insert_selected_and_mirrored(self):
        r = _renderer()
        sd = {"tracks": {
            "injected_obstacle_0": _cone_track(),
            "plain_agent": _cone_track(asset_id=None),   # no asset -> skipped
        }}
        r._insert_injected_assets(sd)

        (req,) = r._stub.edit_requests
        assert req.scene_id == "test_scene"
        assert len(req.insert) == 1
        trk = req.insert[0]
        assert trk.id == "injected_obstacle_0"
        assert trk.asset_id == "/assets/cone_3dgs.ply"
        # "cone" has no CONE/SIGN class in the observed set -> falls back to
        # the scene's VEHICLE literal (a class the server provably accepts)
        assert trk.semantic_class == "WODPerceptionBoxDetectionLabel.TYPE_VEHICLE"
        # id joins the controllable set -> _build_dynamic_objects mirrors it
        assert "injected_obstacle_0" in r._dyn_track_ids
        assert r._inserted_asset_ids == {"injected_obstacle_0"}

    def test_trajectory_spans_scene_timestamps(self):
        r = _renderer()
        r._insert_injected_assets({"tracks": {"c0": _cone_track()}})
        trk = r._stub.edit_requests[0].insert[0]
        poses = trk.trajectory.poses
        assert [p.timestamp_us for p in poses] == [1_000_000, 1_200_000]
        # NuRec world = eval-local - origin_offset
        assert poses[0].pose.vec.x == pytest.approx(30.0 - 100.0)
        assert poses[0].pose.vec.y == pytest.approx(0.0 - 200.0)
        assert poses[0].pose.vec.z == pytest.approx(53.3)

    def test_aabb_extents(self):
        """``dims_offset`` + the extent, floored at 1.0 per axis.

        The floor is what keeps the implied scale — ``(object_size -
        dims_offset).max()`` — at exactly 1.0 for anything cone-sized.
        """
        r = _renderer()
        r._insert_injected_assets({"tracks": {"c0": _cone_track()}})
        box = r._stub.edit_requests[0].insert[0].object_size
        assert (box.size_x, box.size_y, box.size_z) == pytest.approx(
            (2.0, 2.0, 1.25))
        assert r._asset_scale_factor([0.4, 0.4, 0.7]) == pytest.approx(1.0)

    def test_semantic_class_resolution(self):
        r = _renderer()
        # explicit enum literal passes through verbatim
        assert r._resolve_semantic_class(
            {"nurec_semantic_class": "Foo.TYPE_SIGN"}) == "Foo.TYPE_SIGN"
        # keyword match against observed classes
        assert r._resolve_semantic_class(
            {"nurec_semantic_class": "pedestrian"}
        ) == "WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN"
        # cone: no CONE class observed -> VEHICLE fallback
        assert r._resolve_semantic_class(
            {"nurec_semantic_class": "cone"}
        ) == "WODPerceptionBoxDetectionLabel.TYPE_VEHICLE"
        # SIGN must NOT be picked for cone even when observed — sign-class
        # inserts are rejected by the live server (special static path)
        r._dyn_sem_counter["WODPerceptionBoxDetectionLabel.TYPE_SIGN"] = 500
        assert r._resolve_semantic_class(
            {"nurec_semantic_class": "cone"}
        ) == "WODPerceptionBoxDetectionLabel.TYPE_VEHICLE"
        # a genuine CONE class is preferred when present
        r._dyn_sem_counter["SomeLabel.TYPE_CONE"] = 1
        assert r._resolve_semantic_class(
            {"nurec_semantic_class": "cone"}) == "SomeLabel.TYPE_CONE"
        # no observed classes at all -> raw value passthrough
        r._dyn_sem_counter = {}
        assert r._resolve_semantic_class(
            {"nurec_semantic_class": "cone"}) == "cone"

    def test_a_class_the_scenes_tracks_never_use_is_still_asked_for(self):
        # What a recon accepts is its trained actor layers, not its recorded
        # traffic. A clip whose logged tracks are all vehicles can still have a
        # pedestrian layer, and R-3's walkers were refused outright because the
        # client would only name classes it had seen. The literal is built in
        # the artifact's own enum style so it still parses.
        r = _renderer(sem_counter={"NuPlanBoxDetectionLabel.VEHICLE": 9})
        assert r._resolve_semantic_class({"type": "pedestrian"}) == (
            "NuPlanBoxDetectionLabel.PEDESTRIAN")
        wod = _renderer(sem_counter={"WODPerceptionBoxDetectionLabel.TYPE_VEHICLE": 9})
        assert wod._resolve_semantic_class({"type": "pedestrian"}) == (
            "WODPerceptionBoxDetectionLabel.TYPE_PEDESTRIAN")
        # ...but a member no label enum carries is never invented: an
        # unparseable literal fails the whole insert on the server.
        assert wod._class_literal("CONE") is None

    def test_server_rejection_raises_and_keeps_state_clean(self):
        r = _renderer()

        class _Reject:
            success = False
            message = "asset not found"

        r._stub.edit_assets = lambda req, timeout=None: _Reject()
        with pytest.raises(RuntimeError, match="asset not found"):
            r._insert_injected_assets({"tracks": {"c0": _cone_track()}})
        assert "c0" not in r._dyn_track_ids
        assert r._inserted_asset_ids == set()

    def test_id_collision_skipped(self):
        r = _renderer()
        sd = {"tracks": {
            "baked_actor_0": _cone_track(),   # collides with a BAKED track
            "c1": _cone_track(),
        }}
        r._insert_injected_assets(sd)
        # the baked collision triggers the stale-heal restore once, but the
        # id still exists after re-enumeration -> skipped, only c1 inserted
        (req,) = r._stub.edit_requests
        assert [t.id for t in req.insert] == ["c1"]
        assert r._inserted_asset_ids == {"c1"}

    def test_stale_injected_tracks_healed_by_restore(self):
        """A dead previous run left injected ids on the server: restore runs
        once, ids re-enumerate to the baked set, insert then succeeds."""
        r = _renderer()
        r._dyn_track_ids = {"baked_actor_0", "injected_obstacle_0"}  # stale
        r._insert_injected_assets(
            {"tracks": {"injected_obstacle_0": _cone_track()}})
        assert len(r._stub.restore_requests) == 1          # healed first
        (req,) = r._stub.edit_requests
        assert [t.id for t in req.insert] == ["injected_obstacle_0"]
        assert "injected_obstacle_0" in r._dyn_track_ids

    def test_no_asset_tracks_no_rpc(self):
        r = _renderer()
        r._insert_injected_assets({"tracks": {"a": _cone_track(asset_id=None)}})
        assert r._stub.edit_requests == []
        assert r._inserted_asset_ids == set()

    def test_ground_z_pinned_for_render_poses(self):
        """agent_states clamps z to 0; injected ids must keep the track ground z."""
        r = _renderer()
        r._insert_injected_assets({"tracks": {"c0": _cone_track(pos=(30.0, 0.0, 53.3))}})
        assert r._injected_ground_z["c0"] == pytest.approx(53.3)
        # per-frame path: z=0 in the state (as agent_states delivers it)
        m = r._actor_to_world({"id": "c0", "position": (31.0, 0.5, 0.0),
                               "heading": 0.0})
        assert m[2, 3] == pytest.approx(53.3)


class TestAssetPreScaling:
    """The server multiplies a loaded asset by ``(object_size -
    dims_offset).max()``; the library is metric, so the copy it loads must be
    pre-divided by that factor or the size lands twice (the 1.75 m bicycle that
    rendered 3.05 m tall while its sim box stayed 1.75 m)."""

    @staticmethod
    def _bike_track(tmp_path, dims=(1.8, 0.675, 1.713)):
        src = tmp_path / "bicycle_1.ply"
        # A metric, base-grounded, y-up asset: the library's own convention.
        write_min_ply(src, [[-0.9, 0.0, -0.34], [0.9, 1.713, 0.34]])
        trk = _cone_track(asset_id=str(src))
        T = len(trk["state"]["heading"])
        for key, value in zip(("length", "width", "height"), dims):
            trk["state"][key] = np.full(T, value, np.float32)
        return trk, src

    def test_metric_ply_divided_by_the_factor_the_server_applies(self, tmp_path):
        r = _renderer()
        trk, src = self._bike_track(tmp_path)
        r._insert_injected_assets({"tracks": {"bike": trk}})

        sent = r._stub.edit_requests[0].insert[0]
        assert sent.asset_id != str(src)
        scaled = tmp_path / "_nurec_prescaled"
        assert Path(sent.asset_id).parent == scaled

        # factor = max(object_size - dims_offset) = the longest dim, 1.8
        factor = 1.8
        before = load_ply(src).means
        after = load_ply(sent.asset_id).means
        assert after == pytest.approx(before / factor, rel=1e-5)
        # object_size still describes the real asset, so factor * the divided
        # asset is the metric asset again.
        box = sent.object_size
        assert max(box.size_x - 1.0, box.size_y - 1.0,
                   box.size_z - 0.25) == pytest.approx(factor)

    def test_log_sigmas_follow_the_resize(self, tmp_path):
        """Centres alone would shrink the cloud but keep fat blobs."""
        r = _renderer()
        trk, src = self._bike_track(tmp_path)
        r._insert_injected_assets({"tracks": {"bike": trk}})
        sent = r._stub.edit_requests[0].insert[0]
        assert load_ply(sent.asset_id).scales == pytest.approx(
            load_ply(src).scales / 1.8, rel=1e-5)

    def test_cache_is_reused_and_keyed_by_content(self, tmp_path):
        r = _renderer()
        trk, src = self._bike_track(tmp_path)
        r._insert_injected_assets({"tracks": {"bike": trk}})
        first = r._stub.edit_requests[0].insert[0].asset_id
        mtime = Path(first).stat().st_mtime_ns

        r2 = _renderer()
        r2._insert_injected_assets({"tracks": {"bike": trk}})
        assert r2._stub.edit_requests[0].insert[0].asset_id == first
        assert Path(first).stat().st_mtime_ns == mtime   # not rewritten

        # re-authoring the source must not be served from the stale copy
        write_min_ply(src, [[-0.5, 0.0, -0.2], [0.5, 1.0, 0.2]])
        r3 = _renderer()
        r3._insert_injected_assets({"tracks": {"bike": trk}})
        assert r3._stub.edit_requests[0].insert[0].asset_id != first

    def test_small_asset_passes_through_untouched(self, tmp_path):
        """A cone fits inside dims_offset + 1, so the factor is already 1.0."""
        r = _renderer()
        src = tmp_path / "cone.ply"
        write_min_ply(src, [[0.0, 0.0, 0.0], [0.1, 0.7, 0.1]])
        r._insert_injected_assets({"tracks": {"c0": _cone_track(asset_id=str(src))}})
        assert r._stub.edit_requests[0].insert[0].asset_id == str(src)
        assert not (tmp_path / "_nurec_prescaled").exists()

    def test_asset_bank_id_passes_through(self):
        """Not a file: an AssetBank track id the bank sizes itself."""
        r = _renderer()
        trk = _cone_track(asset_id="bank_track_17")
        T = len(trk["state"]["heading"])
        trk["state"]["length"] = np.full(T, 4.6, np.float32)
        r._insert_injected_assets({"tracks": {"a": trk}})
        assert r._stub.edit_requests[0].insert[0].asset_id == "bank_track_17"

    def test_cache_root_override(self, tmp_path, monkeypatch):
        """A read-only asset library still evaluates."""
        elsewhere = tmp_path / "cache"
        monkeypatch.setenv("NUREC_GRPC_ASSET_CACHE", str(elsewhere))
        r = _renderer()
        trk, _src = self._bike_track(tmp_path)
        r._insert_injected_assets({"tracks": {"bike": trk}})
        assert Path(r._stub.edit_requests[0].insert[0].asset_id).parent == elsewhere


class TestCloseRestores:
    def test_restore_called_once_with_scene_id(self):
        r = _renderer()
        r._insert_injected_assets({"tracks": {"c0": _cone_track()}})
        r.close()
        (restore_req,) = r._stub.restore_requests
        assert restore_req.scene_id == "test_scene"
        assert r._inserted_asset_ids == set()

    def test_no_restore_without_inserts(self):
        r = _renderer()
        r.close()
        assert r._stub.restore_requests == []
