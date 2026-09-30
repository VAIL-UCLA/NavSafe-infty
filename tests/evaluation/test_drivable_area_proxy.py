# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""THE drivable-area predicate — one implementation, two scorers.

The trap this pins: the batch (proposal-scoring) EPDMS scorer used to test
each ego corner against the raw lane polygons ONE PIECE AT A TIME, while the
reported metric tested a buffered union of all of them. The per-piece test has
no tolerance for seams or overhang — lane polygons fall back to a 3.5 m
flat-capped centerline strip — so a candidate shifted 1 m laterally puts its
outer corner a few tens of centimetres past the strip edge and is charged a
violation. Over 30 py123d scenes / 4748 logged ego frames the two proxies
disagreed on 11.5% of poses shifted 1 m right and 3.1% shifted 1 m left, and
agreed on 100% of un-shifted poses — the penalty falls on LATERAL candidates
only. ``dac`` multiplies into the score, so the lateral third of the teacher's
proposal grid was devalued or zeroed against the metric it is graded by.

These tests are import-light on purpose (shapely only): they guard the shared
predicate itself. The scorer-level parity test lives in ``test_lane_proxy.py``
with the rest of the scorer-backed suite.
"""

from __future__ import annotations

import math
import pickle

import pytest

pytest.importorskip("shapely")

from shapely.geometry import LineString, Polygon  # noqa: E402

from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M  # noqa: E402
from navsafe.evaluation.utils.lane_proxy import (  # noqa: E402
    DRIVABLE_AREA_BUFFER_M,
    DrivableAreaProxy,
    corners_in_drivable_area,
)

# The real ego footprint the scorers use (1.852 m wide), not a round number:
# the margins under test are 10-30 cm, so a stand-in box would move them.
EGO_LENGTH = EGO_LENGTH_M
EGO_WIDTH = EGO_WIDTH_M
LANE_HALF_WIDTH = 1.75  # lane_proxy's centerline-buffer fallback (3.5 m lane)


def _strip(y_center: float, half_width: float = LANE_HALF_WIDTH,
           x0: float = -50.0, x1: float = 50.0) -> Polygon:
    """A lane piece: a flat-capped buffer of a straight centerline."""
    return LineString([(x0, y_center), (x1, y_center)]).buffer(
        half_width, cap_style=2)


def _ego_corners(x: float, y: float, heading: float = 0.0):
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    ex, ey = EGO_LENGTH / 2, EGO_WIDTH / 2
    return [
        (x + cx * cos_h - cy * sin_h, y + cx * sin_h + cy * cos_h)
        for cx, cy in ((ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey))
    ]


def _no_fallback():
    raise AssertionError("fallback must not be consulted when a union exists")


def _legacy_per_piece(corners, polygons) -> bool:
    """The pre-Stage-4 batch test, verbatim: each corner in SOME lane piece.

    Kept here (rather than deleted with the production copy) so every case
    below states which predicate it is contrasting, instead of asserting a
    remembered behaviour.
    """
    from shapely.geometry import Point
    for cx, cy in corners:
        pt = Point(cx, cy)
        if not any(p.contains(pt) for p in polygons):
            return False
    return True


class TestBuild:
    def test_empty_lane_set_has_no_proxy(self):
        assert DrivableAreaProxy.from_lane_polygons([]) is None

    def test_none_polygons_are_skipped(self):
        assert DrivableAreaProxy.from_lane_polygons([None, None]) is None
        proxy = DrivableAreaProxy.from_lane_polygons([None, _strip(0.0)])
        assert proxy is not None
        assert proxy.contains_point(0.0, 0.0)

    def test_buffer_tolerance_is_the_documented_one(self):
        # The reported metric has been scored with 0.3 m since the live
        # scorer's union landed; the batch scorer now shares it. Changing
        # this constant moves every historical DAC number.
        assert DRIVABLE_AREA_BUFFER_M == 0.3

    def test_union_is_grown_by_the_buffer(self):
        proxy = DrivableAreaProxy.from_lane_polygons([_strip(0.0)])
        assert proxy is not None
        # 1.75 m strip + 0.3 m tolerance.
        assert proxy.contains_point(0.0, 2.0)
        assert not proxy.contains_point(0.0, 2.1)

    def test_survives_a_pickle_round_trip(self):
        proxy = DrivableAreaProxy.from_lane_polygons([_strip(0.0)])
        restored = pickle.loads(pickle.dumps(proxy))
        assert restored.contains_point(0.0, 0.0)
        assert not restored.contains_point(0.0, 5.0)

    def test_degenerate_geometry_falls_back_rather_than_raising(self):
        class _Exploding:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        assert DrivableAreaProxy.from_lane_polygons([_Exploding()]) is None


class TestSharedPredicate:
    """The regression itself, on the geometry that produced it."""

    def test_laterally_shifted_footprint_overhanging_its_lane_is_drivable(self):
        # THE sweep case: a single 3.5 m lane, ego shifted 1 m laterally.
        # The outer corners land at 1.93 m — 0.18 m past the strip edge,
        # squarely inside the 0.15-0.35 m median overhang measured on the
        # real divergent frames.
        lanes = [_strip(0.0)]
        corners = _ego_corners(0.0, 1.0)
        assert max(abs(cy) for _, cy in corners) == pytest.approx(
            1.0 + EGO_WIDTH / 2)
        assert LANE_HALF_WIDTH < 1.0 + EGO_WIDTH / 2 < (
            LANE_HALF_WIDTH + DRIVABLE_AREA_BUFFER_M)

        assert not _legacy_per_piece(corners, lanes)
        proxy = DrivableAreaProxy.from_lane_polygons(lanes)
        assert corners_in_drivable_area(corners, proxy, _no_fallback)

    def test_corner_in_the_gap_between_two_parallel_lanes_is_drivable(self):
        # Lane centres 4.0 m apart, 3.5 m wide: a 0.5 m unpaved-by-proxy gap
        # that no real road has. A corner in it is inside no piece; the two
        # buffered strips overlap, so the union covers it.
        lanes = [_strip(0.0), _strip(4.0)]
        corners = _ego_corners(0.0, 1.0)  # outer corner at y = 2.0, in the gap
        assert not _legacy_per_piece(corners, lanes)
        proxy = DrivableAreaProxy.from_lane_polygons(lanes)
        assert corners_in_drivable_area(corners, proxy, _no_fallback)

    def test_corner_on_a_junction_seam_is_drivable(self):
        # One corridor split into two butt-jointed halves. ``contains`` is
        # strict, so a corner exactly on the shared edge is inside neither
        # piece; unioning dissolves the seam.
        left = _strip(0.0, x0=-50.0, x1=0.0)
        right = _strip(0.0, x0=0.0, x1=50.0)
        corners = _ego_corners(EGO_LENGTH / 2, 0.0)  # rear corners at x = 0
        assert not _legacy_per_piece(corners, [left, right])
        proxy = DrivableAreaProxy.from_lane_polygons([left, right])
        assert corners_in_drivable_area(corners, proxy, _no_fallback)

    def test_footprint_off_the_map_is_not_drivable(self):
        # The buffer is a seam tolerance, not an amnesty: a genuinely
        # off-road footprint must still fail, or DAC stops being a constraint.
        proxy = DrivableAreaProxy.from_lane_polygons([_strip(0.0)])
        assert not corners_in_drivable_area(
            _ego_corners(0.0, 8.0), proxy, _no_fallback)

    def test_footprint_just_past_the_buffer_is_not_drivable(self):
        proxy = DrivableAreaProxy.from_lane_polygons([_strip(0.0)])
        edge = LANE_HALF_WIDTH + DRIVABLE_AREA_BUFFER_M - EGO_WIDTH / 2
        assert corners_in_drivable_area(
            _ego_corners(0.0, edge - 0.01), proxy, _no_fallback)
        assert not corners_in_drivable_area(
            _ego_corners(0.0, edge + 0.01), proxy, _no_fallback)

    def test_centred_footprint_is_drivable_in_a_single_lane(self):
        # Un-shifted poses agreed 100% before the fix; they must still pass,
        # under both predicates.
        lanes = [_strip(0.0)]
        corners = _ego_corners(0.0, 0.0)
        assert _legacy_per_piece(corners, lanes)
        proxy = DrivableAreaProxy.from_lane_polygons(lanes)
        assert corners_in_drivable_area(corners, proxy, _no_fallback)


class TestFallback:
    """Behaviour when no union can be built — unchanged from before."""

    def test_no_lanes_means_not_drivable(self):
        # A scorer with an empty map scores dac=0; that is load-bearing
        # (tests/evaluation/test_epdms_live_scorer.py pins the gate).
        assert not corners_in_drivable_area(
            _ego_corners(0.0, 0.0), None, lambda: [])

    def test_per_piece_containment_still_accepts_a_centred_footprint(self):
        assert corners_in_drivable_area(
            _ego_corners(0.0, 0.0), None, lambda: [_strip(0.0)])

    def test_per_piece_containment_rejects_an_overhanging_footprint(self):
        # The documented weakness of the fallback, pinned so the reason the
        # union exists stays visible: this is the pose the union accepts.
        assert not corners_in_drivable_area(
            _ego_corners(0.0, 1.0), None, lambda: [_strip(0.0)])

    def test_fallback_polygons_are_only_requested_once(self):
        calls = []

        def _polys():
            calls.append(1)
            return [_strip(0.0)]

        corners_in_drivable_area(_ego_corners(0.0, 0.0), None, _polys)
        assert len(calls) == 1
