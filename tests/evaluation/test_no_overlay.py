"""NEXUSSIM_NO_OVERLAY: cam_f0.jpg written unannotated (eval_artifacts)."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.evaluation.eval_artifacts import EvalArtifactWriter, _no_overlay


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("no", False), ("off", False), ("1", True), ("true", True),
    ("yes", True), ("ON", True),
])
def test_flag_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("NEXUSSIM_NO_OVERLAY", raising=False)
    else:
        monkeypatch.setenv("NEXUSSIM_NO_OVERLAY", value)
    assert _no_overlay() is expected


def _write(tmp_path, img):
    """The artifact writer's own imwrite, so the test covers the real path."""
    ok = EvalArtifactWriter._safe_imwrite(tmp_path / "cam_f0.jpg", img)
    assert ok
    return tmp_path / "cam_f0.jpg"


def test_raw_frame_round_trips_through_the_writer(tmp_path):
    """The unannotated path writes the renderer's frame as-is (jpeg, so exact
    equality is not the test — that it is the same picture is)."""
    import cv2

    img = np.zeros((64, 96, 3), dtype=np.uint8)
    img[:, :48] = (200, 30, 30)          # a distinctly asymmetric picture
    path = _write(tmp_path, img)
    back = cv2.imread(str(path))
    assert back.shape == img.shape
    # Left half stays the coloured half — channels and orientation preserved.
    assert back[:, :48].mean() > back[:, 48:].mean() + 50


# --- NEXUSSIM_NO_CAM_MAP_LINES: keep the overlays, drop the map polylines ---

def _scene_with_map():
    """A scenario whose map has one lane straight ahead of the ego."""
    from navsafe.scenario.scenario_description import ScenarioDescription as SD

    from navsafe.scenario.type import MetaDriveType

    # A road *line* (not a lane surface): that is what the camera projection
    # draws, per MetaDriveType.is_road_line.
    poly = np.array([[x, 1.75, 0.0] for x in np.linspace(2.0, 60.0, 40)],
                    dtype=np.float32)
    return {SD.MAP_FEATURES: {"line_0": {SD.TYPE: MetaDriveType.LINE_SOLID_SINGLE_WHITE,
                                         SD.POLYLINE: poly}}}


def _front_cam(**over):
    from navsafe.evaluation import vis_utils
    from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS

    kwargs = dict(
        image=np.full((360, 640, 3), 90, dtype=np.uint8),
        # [lateral, forward] — the NexusSim adapter convention.
        plan_traj_ego=np.array([[0.0, float(i) * 2.0] for i in range(1, 9)]),
        cam_config=NAVSIM_CAM_CONFIGS["CAM_F0"],
        frame_id=7, model_name="test", speed_kmh=10.0, collision=False,
        driving_command=2, sim_dt=0.1,
        ego_position=np.array([0.0, 0.0, 0.0]), ego_heading=0.0,
        scenario_data=_scene_with_map(),
    )
    kwargs.update(over)
    return vis_utils.render_front_cam(**kwargs)


def test_map_lines_flag_removes_only_the_map(monkeypatch):
    monkeypatch.delenv("NEXUSSIM_NO_CAM_MAP_LINES", raising=False)
    with_map = _front_cam()

    monkeypatch.setenv("NEXUSSIM_NO_CAM_MAP_LINES", "1")
    without_map = _front_cam()

    raw = np.full((360, 640, 3), 90, dtype=np.uint8)
    # The map lines are gone ...
    assert not np.array_equal(with_map, without_map)
    # ... but the frame is still annotated (HUD, plan ribbon, waypoints).
    assert not np.array_equal(without_map, raw)


def test_map_lines_flag_leaves_the_plan_ribbon(monkeypatch):
    """The ribbon is drawn outside the guarded block, so it must survive."""
    monkeypatch.setenv("NEXUSSIM_NO_CAM_MAP_LINES", "1")
    with_plan = _front_cam()
    no_plan = _front_cam(plan_traj_ego=None)
    assert not np.array_equal(with_plan, no_plan)


# --- NEXUSSIM_NO_OVERLAY, as render_front_cam reads it ----------------------

@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("no", False), ("off", False), ("1", True), ("true", True),
    ("yes", True), ("ON", True),
])
def test_vis_utils_flag_parsing(monkeypatch, value, expected):
    """Must agree with eval_artifacts._no_overlay on every spelling.

    "0" is the one that matters: a launcher pinning the default with
    ``export NEXUSSIM_NO_OVERLAY=0`` puts the *string* "0" in the environment,
    which a bare truth test reads as on.
    """
    from navsafe.evaluation.vis_utils import _no_overlay as vis_no_overlay

    if value is None:
        monkeypatch.delenv("NEXUSSIM_NO_OVERLAY", raising=False)
    else:
        monkeypatch.setenv("NEXUSSIM_NO_OVERLAY", value)
    assert vis_no_overlay() is expected


def test_explicit_zero_keeps_the_camera_annotated(monkeypatch):
    monkeypatch.delenv("NEXUSSIM_NO_CAM_MAP_LINES", raising=False)
    raw = np.full((360, 640, 3), 90, dtype=np.uint8)

    monkeypatch.setenv("NEXUSSIM_NO_OVERLAY", "0")
    assert not np.array_equal(_front_cam(), raw), "0 must mean 'draw the overlay'"

    monkeypatch.setenv("NEXUSSIM_NO_OVERLAY", "1")
    assert np.array_equal(_front_cam(), raw), "1 must return the frame untouched"
