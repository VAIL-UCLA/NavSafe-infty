# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free contracts for the NuRec gRPC renderer."""

from types import SimpleNamespace

import numpy as np
import pytest

from navsafe.render.nurec_grpc import NuRecGrpcSceneRenderer


_GEOMETRY = {"x": 0.0, "y": 0.0, "z": 1.5, "fov": 70.0}


def _renderer(cameras):
    renderer = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    renderer._cam_intrinsics = cameras
    renderer._template_fallback_warned = set()
    renderer._scene_id = "scene"
    return renderer


def test_navsim_virtual_rear_camera_borrows_available_spec():
    front = object()
    renderer = _renderer({"camera_pcam_f0": front,
                          "camera_pcam_l0": object()})

    assert renderer._base_spec_for("CAM_B0", _GEOMETRY, "navsim") is front
    assert renderer._template_fallback_warned == {"CAM_B0"}


def test_recon_rig_does_not_invent_missing_camera():
    renderer = _renderer({"camera_pcam_f0": object()})

    assert renderer._base_spec_for("CAM_B0", _GEOMETRY, "recon") is None


def test_exact_camera_spec_wins_over_fallback():
    rear = object()
    renderer = _renderer({"camera_pcam_f0": object(), "camera_pcam_b0": rear})

    assert renderer._base_spec_for("CAM_B0", _GEOMETRY, "navsim") is rear
    assert not renderer._template_fallback_warned


def test_native_camera_is_the_quality_default(monkeypatch):
    monkeypatch.delenv("NUREC_GRPC_CAM_RIG", raising=False)

    assert NuRecGrpcSceneRenderer._camera_rig_mode() == "recon"


def test_native_camera_alias(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_CAM_RIG", "native")

    assert NuRecGrpcSceneRenderer._camera_rig_mode() == "recon"


def test_invalid_camera_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_CAM_RIG", "typo")

    with pytest.raises(ValueError, match="NUREC_GRPC_CAM_RIG"):
        NuRecGrpcSceneRenderer._camera_rig_mode()


def test_native_request_uses_advertised_training_resolution():
    spec = SimpleNamespace(resolution_h=1080, resolution_w=1920)
    policy_cfg = {"height": 1120, "width": 1920}

    assert NuRecGrpcSceneRenderer._request_resolution(
        spec, policy_cfg, "recon") == (1080, 1920)


def test_synthetic_request_keeps_policy_resolution():
    spec = SimpleNamespace(resolution_h=1080, resolution_w=1920)
    policy_cfg = {"height": 1120, "width": 1920}

    assert NuRecGrpcSceneRenderer._request_resolution(
        spec, policy_cfg, "navsim") == (1120, 1920)


def _manifold_renderer():
    renderer = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    renderer._sd_ego_xy_arr = np.array(
        [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    renderer._manifold_clamp_warned = False
    renderer._manifold_max_deviation_m = 0.0
    renderer._scene_id = "scene"
    return renderer


def test_camera_pose_inside_3dgs_coverage_is_unchanged(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_MAX_NOVEL_VIEW_M", "5")
    renderer = _manifold_renderer()
    state = {"position": np.array([5.0, 3.0, 1.0]), "heading": 0.4}

    assert renderer._render_pose_on_manifold(state) is state


def test_out_of_coverage_camera_is_capped_without_mutating_sim_pose(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_MAX_NOVEL_VIEW_M", "5")
    monkeypatch.setenv("NUREC_GRPC_MAX_NOVEL_YAW_DEG", "30")
    renderer = _manifold_renderer()
    original = np.array([5.0, 20.0, 1.0])
    state = {"position": original.copy(), "heading": np.pi}

    rendered = renderer._render_pose_on_manifold(state)

    assert np.array_equal(state["position"], original)
    assert np.allclose(rendered["position"], [5.0, 5.0, 1.0])
    assert abs(rendered["heading"]) == pytest.approx(np.deg2rad(30))
    assert renderer._manifold_max_deviation_m == pytest.approx(20.0)


def test_novel_view_guard_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_MAX_NOVEL_VIEW_M", "0")
    renderer = _manifold_renderer()
    state = {"position": np.array([5.0, 100.0]), "heading": 0.0}

    assert renderer._render_pose_on_manifold(state) is state


class TestHandoffRequestStaysInsideItsWindow:
    """The window table is arithmetic; the log's timestamps are not.

    Windows are `t0 + i * window_us`, while nuPlan pose timestamps are not
    exact multiples of the sample interval. The drift lands on the last
    window: on 2ccebcdb0da25be5 the final logged frame sat 30 us past the end
    of window 4, no window contained it, the nearest-window fallback picked
    that scene anyway, and the server refused a frame it had not baked —
    killing the episode at frame 179 of 208 as a camera failure.
    """

    @staticmethod
    def _renderer(windows):
        r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
        r._handoff = [(f"s{i + 1}", f"/off/s{i + 1}.json", None, w)
                      for i, w in enumerate(windows)]
        return r

    def test_a_frame_past_the_last_window_lands_on_its_last_baked_step(self):
        r = self._renderer([(100, 200), (200, 300)])
        r._pick_handoff(330)                  # nearest-window fallback -> s2
        assert r._ho_cur == 1
        assert r._clamp_to_window(330) == 299

    def test_a_frame_before_the_first_window_lands_on_its_first(self):
        r = self._renderer([(100, 200), (200, 300)])
        r._pick_handoff(40)
        assert r._clamp_to_window(40) == 100

    def test_a_frame_inside_its_window_is_untouched(self):
        r = self._renderer([(100, 200), (200, 300)])
        r._pick_handoff(250)
        assert r._ho_cur == 1
        assert r._clamp_to_window(250) == 250

    def test_no_handoff_table_is_a_no_op(self):
        r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
        r._handoff = []
        r._ho_cur = None
        assert r._clamp_to_window(1234) == 1234


class TestAFrameTheReconNeverBaked:
    """A recon's baked coverage can end before the log's last frame.

    2ccebcdb0da25be5's fourth sub-clip stops ~75 us short of the last logged
    pose, so the closing steps ask for a frame it never trained and the server
    answers OUT_OF_RANGE. Clamping to the handoff WINDOW does not help: that
    table is arithmetic and is wider than the bake.
    """

    class _Status:
        def __init__(self, name): self.name = name

    class _RpcError(Exception):
        def __init__(self, name): self.name = name
        def code(self): return TestAFrameTheReconNeverBaked._Status(self.name)

    @staticmethod
    def _renderer(fail_on):
        r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
        r._scene_id, r._timeout, r._asked = "s1", 1, []

        class _Stub:
            def render_rgb(_s, req, timeout=None):
                r._asked.append(req.frame_start_us)
                if req.frame_start_us in fail_on:
                    raise TestAFrameTheReconNeverBaked._RpcError("OUT_OF_RANGE")
                return type("R", (), {"image_bytes": b"jpeg"})()

        r._stub = _Stub()
        r._decode = lambda b: b
        return r

    def _req(self, t):
        return type("Req", (), {"frame_start_us": t, "frame_end_us": t + 1})()

    def test_it_re_asks_at_the_last_frame_that_scene_served(self, monkeypatch):
        r = self._renderer(fail_on={200})
        # A good frame first, so the renderer knows one this scene has.
        assert r._send_render(self._req(100), 100) == b"jpeg"
        assert r._send_render(self._req(200), 200) == b"jpeg"
        assert r._asked == [100, 200, 100]

    def test_with_nothing_known_good_the_error_stands(self):
        r = self._renderer(fail_on={200})
        with pytest.raises(TestAFrameTheReconNeverBaked._RpcError):
            r._send_render(self._req(200), 200)

    def test_a_different_status_is_not_retried(self):
        r = self._renderer(fail_on=set())

        class _Stub:
            def render_rgb(_s, req, timeout=None):
                raise TestAFrameTheReconNeverBaked._RpcError("UNAVAILABLE")

        r._stub = _Stub()
        r._last_ok_us = {"s1": 100}
        with pytest.raises(TestAFrameTheReconNeverBaked._RpcError):
            r._send_render(self._req(200), 200)


# ── NUREC_GRPC_RENDER_STEPS ──────────────────────────────────────────────────
# A fidelity campaign scores the 2 Hz navsim timestamps but has to STEP at the
# log's 10 Hz for the replayed ego and actors to be where the log put them.
# Rendering all ten and keeping two is 5x the GPU time for frames nothing reads.


def test_render_step_filter_is_off_by_default(monkeypatch):
    monkeypatch.delenv("NUREC_GRPC_RENDER_STEPS", raising=False)

    assert NuRecGrpcSceneRenderer._render_step_filter() is None


def test_render_step_filter_parses_a_list(monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_RENDER_STEPS", "0,5, 10\n15")

    assert NuRecGrpcSceneRenderer._render_step_filter() == {0, 5, 10, 15}


def test_render_step_filter_reads_a_file(monkeypatch, tmp_path):
    p = tmp_path / "steps.txt"
    p.write_text("0,5,10\n")
    monkeypatch.setenv("NUREC_GRPC_RENDER_STEPS", f"@{p}")

    assert NuRecGrpcSceneRenderer._render_step_filter() == {0, 5, 10}


def _filtered(steps, last):
    r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    r._render_steps = steps
    r._last_images = last
    return r


def test_unfiltered_renderer_never_reuses_a_frame():
    r = _filtered(None, {"CAM_F0": np.ones((2, 2, 3), np.uint8)})

    assert r._cached_frames_for(7) is None


def test_a_listed_step_is_rendered_not_reused():
    r = _filtered({0, 5}, {"CAM_F0": np.ones((2, 2, 3), np.uint8)})

    assert r._cached_frames_for(5) is None


def test_first_step_renders_even_when_unlisted():
    """Nothing to reuse yet — a skipped first step would hand back nothing."""
    r = _filtered({5}, None)

    assert r._cached_frames_for(0) is None


def test_unlisted_step_reuses_the_previous_frame():
    img = np.ones((2, 2, 3), np.uint8)
    r = _filtered({0, 5}, {"CAM_F0": img})

    out = r._cached_frames_for(3)

    assert set(out) == {"CAM_F0"}
    assert np.array_equal(out["CAM_F0"], img)


def test_reused_frames_are_copies():
    """The visualiser annotates in place; sharing the array would let one
    skipped step corrupt the cache for every later one."""
    img = np.ones((2, 2, 3), np.uint8)
    r = _filtered({0}, {"CAM_F0": img})

    out = r._cached_frames_for(1)
    out["CAM_F0"][:] = 9

    assert np.array_equal(r._last_images["CAM_F0"], img)
    assert not np.shares_memory(out["CAM_F0"], img)


# ── recorded rig attitude ────────────────────────────────────────────────────
# ego_state carries position and heading only, so a rig built from it renders
# level; the reconstruction was not fit level. Yaw stays the simulator's, pitch
# and roll come from the training trajectory.


def _rot(yaw, pitch, roll):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()


def _rz(yaw_deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("Z", yaw_deg, degrees=True).as_matrix()


def test_strip_yaw_round_trips():
    """Rz(yaw) @ strip_yaw(R) == R. The euler detour that broke this once
    (as_euler("xyz") then from_euler("zyx")) is not an inverse pair."""
    mats = np.stack([_rot(50.079, -1.02511, -0.94771),
                     _rot(-170.0, 3.5, -7.25),
                     _rot(0.0, 0.0, 0.0)])

    tilts = NuRecGrpcSceneRenderer._strip_yaw(mats)

    for m, t in zip(mats, tilts):
        yaw = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
        assert np.allclose(_rz(yaw) @ t, m, atol=1e-12)


def test_strip_yaw_leaves_no_yaw():
    tilts = NuRecGrpcSceneRenderer._strip_yaw(np.stack([_rot(50.079, -1.025, -0.948)]))

    assert abs(np.degrees(np.arctan2(tilts[0][1, 0], tilts[0][0, 0]))) < 1e-9


def test_strip_yaw_keeps_pitch_and_roll():
    from scipy.spatial.transform import Rotation
    tilts = NuRecGrpcSceneRenderer._strip_yaw(np.stack([_rot(50.079, -1.025, -0.948)]))

    _, pitch, roll = Rotation.from_matrix(tilts[0]).as_euler("ZYX", degrees=True)

    assert pitch == pytest.approx(-1.025, abs=1e-6)
    assert roll == pytest.approx(-0.948, abs=1e-6)


def _tilted(tilt, scene="s", ts=None, tilts=None):
    r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    r._scene_id = scene
    r._tilt_cache = {scene: (ts, tilts)} if ts is not None else {scene: tilt}
    return r


def test_rig_attitude_is_unconditional():
    """No switch: a level rig is the bug, not an alternative rendering."""
    want = _rot(0, 5, 5)
    r = _tilted(None, ts=np.array([10]), tilts=np.stack([want]))

    assert np.allclose(r._recorded_tilt(10), want)


def test_rig_attitude_picks_the_nearest_timestamp():
    a, b = _rot(0, 1, 0), _rot(0, 2, 0)
    r = _tilted(None, ts=np.array([100, 200]), tilts=np.stack([a, b]))

    assert np.allclose(r._recorded_tilt(140), a)
    assert np.allclose(r._recorded_tilt(160), b)


def test_rig_stays_level_when_the_server_has_no_trajectory():
    r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    r._scene_id = "s"
    r._tilt_cache = {"s": None}

    assert r._recorded_tilt(10) is None


# ── ego anchor: bbox centre vs rig origin ────────────────────────────────────
# The scenario's ego x/y is the bounding-box CENTRE; the reconstruction's world
# origin is the nuPlan ego_pose. Composing cam->rig onto the centre puts the
# camera ~1.45 m up the road.


def _anchored(base, handoff=None):
    r = NuRecGrpcSceneRenderer.__new__(NuRecGrpcSceneRenderer)
    r._origin_offset = np.asarray(base, np.float64)
    r._handoff = handoff
    r._anchor_lever = None
    r._anchor_logged = True
    return r


def test_ego_anchor_lever_is_purely_along_the_vehicle_axis():
    """origin_offset is imu0 - centre0 in WORLD axes; rotated by the frame-0
    heading it must come out as a forward offset with no lateral part."""
    import math
    th = math.radians(50.07934)
    r = _anchored([-0.9265, -1.1189, 0.0])

    lever = r._ego_anchor_lever(th)

    assert lever[0] == pytest.approx(1.4527, abs=2e-3)   # forward
    assert lever[1] == pytest.approx(0.0, abs=1e-2)      # not sideways


def test_ego_anchor_lever_is_cached_not_recomputed_per_heading():
    import math
    r = _anchored([-0.9265, -1.1189, 0.0])

    first = r._ego_anchor_lever(math.radians(50.07934)).copy()
    later = r._ego_anchor_lever(math.radians(180.0))

    assert np.allclose(first, later)


def test_ego_anchor_prefers_the_first_subclip_offset():
    """Later sub-clips re-reference to their own window start; only s1 measures
    the vehicle."""
    import math
    ho = [["s1", np.array([-0.9265, -1.1189, 0.0])],
          ["s2", np.array([-40.0, -50.0, 0.0])]]
    r = _anchored([-40.0, -50.0, 0.0], handoff=ho)

    lever = r._ego_anchor_lever(math.radians(50.07934))

    assert lever[0] == pytest.approx(1.4527, abs=2e-3)


def test_zero_offset_means_no_lever():
    """An absolute-frame recon has no centre/origin split to correct."""
    import math
    r = _anchored([0.0, 0.0, 0.0])

    assert np.allclose(r._ego_anchor_lever(math.radians(50.0)), [0.0, 0.0])
