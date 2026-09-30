# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-trajectory overlays — the front-camera projection contract.

Candidates are produced at a replan frame and reused until the next one, so
between replans they must be re-anchored from the *prediction-time* ego frame
into the *current* ego frame before projection. These tests pin:

* :func:`reanchor_candidates_to_ego` is exactly the composition of
  ``render_bev_candidates``' ego→world transform with ``EvalArtifacts``'
  world→ego transform (i.e. the camera and BEV overlays agree).
* :func:`render_front_cam_candidates` projects sanely and degrades quietly on
  degenerate input — it is visualization-only and must never fail a run.
"""

from __future__ import annotations

import numpy as np

from navsafe.evaluation.vis_utils import (
    project_ego_to_camera,
    reanchor_candidates_to_ego,
    render_front_cam_candidates,
)


CAM_CONFIG = {
    "x": 1.5, "y": 0.0, "z": 1.5, "yaw": 0.0,
    "fov": 70.0, "width": 640, "height": 360,
}


def _blank(h: int = 360, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ----------------------------------------------------------------------
# reanchor_candidates_to_ego
# ----------------------------------------------------------------------


def test_reanchor_is_identity_at_the_prediction_pose() -> None:
    """On a replan frame the ego *is* the prediction pose — candidates unchanged."""
    cands = np.random.default_rng(0).normal(size=(4, 8, 2)) * 5.0
    pos, heading = np.array([10.0, -3.0, 0.0]), 0.7

    out = reanchor_candidates_to_ego(cands, pos, heading, pos, heading)
    np.testing.assert_allclose(out, cands, atol=1e-9)


def test_reanchor_matches_bev_and_eval_artifacts_transforms() -> None:
    """The camera overlay must place candidates exactly where the BEV does.

    Independently recomputes ``render_bev_candidates``' prediction-ego→world
    step and ``EvalArtifacts._render_frame_vis``' world→current-ego step; the
    helper must equal their composition.
    """
    cands = np.random.default_rng(1).normal(size=(3, 8, 2)) * 4.0
    pred_pos, pred_h = np.array([6765.08, 1599.16, 0.0]), 2.010
    ego_pos, ego_h = np.array([6763.36, 1602.89, 0.0]), 1.984

    out = reanchor_candidates_to_ego(cands, pred_pos, pred_h, ego_pos, ego_h)

    # prediction-ego → world (vis_utils.render_bev_candidates)
    cos_h, sin_h = np.cos(pred_h), np.sin(pred_h)
    wx = pred_pos[0] + cos_h * cands[..., 1] - sin_h * cands[..., 0]
    wy = pred_pos[1] + sin_h * cands[..., 1] + cos_h * cands[..., 0]

    # world → current ego (eval_artifacts._render_frame_vis)
    dx, dy = wx - ego_pos[0], wy - ego_pos[1]
    c, s = np.cos(-ego_h), np.sin(-ego_h)
    expected = np.stack([s * dx + c * dy, c * dx - s * dy], axis=-1)

    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_reanchor_shifts_candidates_backward_as_ego_advances() -> None:
    """Ego drives 5 m forward → the (world-fixed) candidates recede by 5 m."""
    cands = np.array([[[0.0, 10.0], [0.0, 20.0]]])  # straight ahead
    out = reanchor_candidates_to_ego(
        cands,
        prediction_position=np.array([0.0, 0.0, 0.0]), prediction_heading=0.0,
        ego_position=np.array([5.0, 0.0, 0.0]), ego_heading=0.0,
    )
    np.testing.assert_allclose(out, [[[0.0, 5.0], [0.0, 15.0]]], atol=1e-9)


# ----------------------------------------------------------------------
# render_front_cam_candidates
# ----------------------------------------------------------------------


def test_straight_candidate_projects_to_image_centerline() -> None:
    """A zero-lateral candidate lands on the vertical centerline, receding upward."""
    straight = np.stack([np.zeros(8), np.arange(5.0, 45.0, 5.0)], axis=1)
    px, valid = project_ego_to_camera(straight, CAM_CONFIG, (360, 640))

    assert valid.all()
    np.testing.assert_allclose(px[:, 0], 320.0, atol=1e-6)
    # Ground plane sits below the camera, so farther points rise toward the horizon.
    assert np.all(np.diff(px[:, 1]) < 0)


def test_render_draws_without_mutating_input() -> None:
    cands = np.stack([np.zeros(8), np.arange(5.0, 45.0, 5.0)], axis=1)[None]
    img = _blank()
    out = render_front_cam_candidates(img, cands, CAM_CONFIG, scores=np.array([1.0]))

    assert out.shape == img.shape
    assert out is not img
    assert not img.any(), "input image was mutated"
    assert out.any(), "nothing was drawn"


def test_render_tolerates_degenerate_candidates() -> None:
    """Empty / wrong-rank candidate sets are a no-op, never an exception."""
    img = _blank()
    assert render_front_cam_candidates(img, np.zeros((0, 8, 2)), CAM_CONFIG).shape == img.shape
    assert render_front_cam_candidates(img, np.zeros((8, 2)), CAM_CONFIG).shape == img.shape


def test_render_without_scores_still_draws() -> None:
    """``scores=None`` (and mismatched lengths) fall back to an index ramp."""
    cands = np.random.default_rng(2).normal(size=(5, 8, 2))
    cands[..., 1] = np.abs(cands[..., 1]) + 5.0  # keep them in front of the camera

    assert render_front_cam_candidates(_blank(), cands, CAM_CONFIG, scores=None).any()
    # A score vector that does not index the candidate set must not raise.
    assert render_front_cam_candidates(
        _blank(), cands, CAM_CONFIG, scores=np.zeros(15)).any()


def test_selected_trajectory_drawn_in_cyan_over_the_fan() -> None:
    """The executed plan is re-drawn thick cyan (BGR 255,255,0) on top.

    Only the image *above* the legend band is inspected — the legend's own
    "Cyan = selected trajectory" caption is drawn in the same colour.
    """
    cands = np.stack([np.full(8, 3.0), np.arange(5.0, 45.0, 5.0)], axis=1)[None]
    plan = np.stack([np.zeros(8), np.arange(5.0, 45.0, 5.0)], axis=1)

    without = render_front_cam_candidates(_blank(), cands, CAM_CONFIG)
    with_plan = render_front_cam_candidates(_blank(), cands, CAM_CONFIG, plan_traj_ego=plan)

    cyan = np.array([255, 255, 0], dtype=np.uint8)
    scene = slice(0, 360 - 50)  # above the legend overlay
    assert not (without[scene] == cyan).all(axis=2).any()
    assert (with_plan[scene] == cyan).all(axis=2).any()
