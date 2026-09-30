# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The BEV must draw the ego at its TRUE footprint.

The footprint constants live in ``navsafe.core.ego_dims`` and are shared
with the planner config; these tests pin their values and text-check that
the planner config still sources them from the shared module (without
importing the planner package, which pulls torch + an IsaacSim bootstrap)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from navsafe.evaluation import vis_utils  # noqa: E402


def _ego_blob_extent(img) -> tuple[int, int]:
    """(width_px, height_px) of the ego marker at the image centre.

    The legend carries an ``Ego``-coloured swatch, so a naive colour mask spans
    legend->ego; take the connected component containing the centre pixel.
    """
    mask = np.all(img == np.array(vis_utils._CLR_EGO, dtype=np.uint8),
                  axis=-1).astype(np.uint8)
    assert mask.any(), "ego marker not drawn"
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    assert n > 1, "ego marker not drawn"
    c = vis_utils.BEV_SIZE / 2.0
    # The heading tick is drawn in black from the centre pixel, so the centre
    # itself is not ego-coloured — pick the component closest to it (the
    # legend swatch is the only other one, far away in the corner).
    best = min(range(1, n),
               key=lambda i: (centroids[i][0] - c) ** 2
               + (centroids[i][1] - c) ** 2)
    x, y, w, h = (stats[best, cv2.CC_STAT_LEFT], stats[best, cv2.CC_STAT_TOP],
                  stats[best, cv2.CC_STAT_WIDTH], stats[best, cv2.CC_STAT_HEIGHT])
    assert abs(centroids[best][0] - c) < 20 and abs(centroids[best][1] - c) < 20, (
        "nearest ego-coloured blob is not at the BEV centre")
    return int(w), int(h)


def test_ego_footprint_constants_match_planner_config():
    """The shared dims MUST stay the canonical MetaDrive-vehicle values.

    vis_utils and the planner config both import ``navsafe.core.ego_dims``;
    pin the values so an accidental edit of the shared constant is caught,
    and text-check (the planner package drags in torch + a kit bootstrap,
    so it cannot be imported here) that config.py still sources its dims
    from the shared module rather than re-hardcoding them.
    """
    from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M

    assert EGO_LENGTH_M == 4.515
    assert EGO_WIDTH_M == 1.852
    assert vis_utils.EGO_LENGTH_M == EGO_LENGTH_M
    assert vis_utils.EGO_WIDTH_M == EGO_WIDTH_M

    src = (
        "navsafe/policy/state/pdm_closed_planner/config.py"
    )
    text = open(src).read()
    assert "VEHICLE_LENGTH: float = EGO_LENGTH_M" in text
    assert "VEHICLE_WIDTH: float = EGO_WIDTH_M" in text


def test_ego_is_drawn_at_true_scale_not_a_fixed_triangle():
    """Ego pixel extent must track its real footprint at the BEV scale."""
    img = vis_utils.render_bev(
        ego_position=np.array([0.0, 0.0]),
        ego_heading=0.0,  # +X, so length maps to image width
        frame_id=0,
    )
    width_px, height_px = _ego_blob_extent(img)

    expect_len_px = vis_utils.EGO_LENGTH_M * vis_utils.BEV_PPM
    expect_wid_px = vis_utils.EGO_WIDTH_M * vis_utils.BEV_PPM
    # Generous tolerance: fill + AA outline, and the heading tick is excluded
    # from the colour mask. The point is scale, not exact rasterisation.
    assert width_px == pytest.approx(expect_len_px, abs=4), (
        f"ego length {width_px}px != true {expect_len_px:.1f}px — the old "
        f"fixed-size triangle bug is back")
    assert height_px == pytest.approx(expect_wid_px, abs=4), (
        f"ego width {height_px}px != true {expect_wid_px:.1f}px")


def test_ego_box_rotates_with_heading():
    """At 90 deg the footprint's long axis must map to image height."""
    img = vis_utils.render_bev(
        ego_position=np.array([0.0, 0.0]),
        ego_heading=np.pi / 2.0,
        frame_id=0,
    )
    width_px, height_px = _ego_blob_extent(img)
    assert height_px > width_px, "ego box did not rotate with heading"


def test_ego_and_route_are_visually_distinct():
    """Ego and route dots must not share a colour.

    They were both red, which is the one confusion a spatial reasoner cannot
    afford: the marker it must localise first looked like the route cluster.
    """
    assert vis_utils._CLR_EGO != vis_utils._CLR_ROUTE
