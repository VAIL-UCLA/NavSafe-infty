# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A route's heading must come from motion, never from localization noise.

The artifact this pins down: a `Route` seeded from a logged track carries the
stretches where that vehicle was STOPPED. At 10 Hz those samples differ by
centimetres of noise, not motion, so `heading_at` read off such a pair returns
a random angle and the vehicle's box is rendered — and simulated — pointing
anywhere. Measured on bundle ``01a58976a2e45a3d``: six of 43 takeover-eligible
agents had segments of 0.0000-0.0055 m, and sampling along their routes swung
the heading by up to 179.3 deg. Nothing raised; the junction simply filled with
splayed, crosswise cars.

Two mechanisms are under test, and each covers what the other cannot:

* `_drop_noise_points` in the constructor, which removes the sub-`MIN_SEGMENT_M`
  points so the segment under `s` normally carries real motion;
* the outward search in `heading_at`, for the two cases the filter leaves —
  a short LAST segment (the final point is always kept, however close it fell)
  and a polyline that is entirely one cluster.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navsafe.traffic.geometry import MIN_SEGMENT_M, ROUTE_WIDTH_M, Route, corners


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _straight(n: int = 40, step: float = 1.0, heading: float = 0.0) -> np.ndarray:
    """``n`` points of real motion at ``step`` metres, pointing at ``heading``."""
    t = np.arange(n, dtype=np.float64) * step
    return np.stack([t * math.cos(heading), t * math.sin(heading)], axis=1)


def _noise(n: int, at, scale: float = 0.005, seed: int = 0) -> np.ndarray:
    """``n`` samples of a vehicle STOPPED at ``at``, jittered by ``scale`` metres."""
    rng = np.random.default_rng(seed)
    return np.asarray(at, dtype=np.float64) + rng.normal(0.0, scale, size=(n, 2))


def _headings_per_segment(route: Route) -> np.ndarray:
    """`heading_at` sampled once inside every segment the route actually has.

    Uniform sampling in ``s`` would miss the artifact: a noise segment is
    centimetres long, so it occupies almost none of the arc and almost no
    sample lands on it. It is still a segment the vehicle passes through.
    """
    mid = 0.5 * (route.s[:-1] + route.s[1:])
    return np.array([route.heading_at(s) for s in mid])


# --- the filter ----------------------------------------------------------

def test_a_stopped_stretch_collapses_to_one_point():
    """The 30 noise samples of a stop leave no segments behind."""
    xy = np.vstack([_straight(10), _noise(30, (9.0, 0.0)), _straight(10) + [30.0, 0.0]])
    route = Route(xy)

    # 10 + 10: the vehicle stopped where the last kept point already is, so the
    # cluster contributes nothing at all. A stop that begins further along
    # keeps its first sample and drops the rest.
    assert len(route.xy) == 20
    seg = np.linalg.norm(np.diff(route.xy, axis=0), axis=1)
    assert seg.min() >= MIN_SEGMENT_M


def test_the_endpoints_survive_so_the_route_keeps_its_extent():
    """The last point is kept even when it falls inside the floor."""
    xy = np.vstack([_straight(10), [[9.01, 0.0]]])   # final step 1 cm
    route = Route(xy)

    assert np.allclose(route.xy[0], xy[0])
    assert np.allclose(route.xy[-1], xy[-1])
    assert route.length == pytest.approx(9.01, abs=1e-9)


def test_a_real_lane_polyline_is_left_alone():
    """Nothing is dropped from a polyline whose every step clears the floor."""
    xy = _straight(40, step=0.5, heading=0.3)
    route = Route(xy)

    assert len(route.xy) == len(xy)
    assert np.allclose(route.xy, xy)


def test_filtering_does_not_move_the_path():
    """Arc length and sampled positions are preserved to noise scale."""
    clean = np.vstack([_straight(20), _straight(20) + [20.0, 0.0]])
    dirty = np.vstack([clean[:20], _noise(25, (19.0, 0.0)), clean[20:]])

    a, b = Route(clean), Route(dirty)
    assert b.length == pytest.approx(a.length, abs=0.05)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(a.position_at(frac * a.length),
                           b.position_at(frac * b.length), atol=0.05)


# --- the heading ---------------------------------------------------------

def test_heading_is_stable_across_a_stopped_stretch():
    """The regression itself: no flip where the vehicle stood still.

    Same polyline built with the filter off reproduces the artifact, so this
    asserts the fix rather than a property the input happened to have.
    """
    xy = np.vstack([_straight(15), _noise(40, (14.0, 0.0), seed=7),
                    _straight(15) + [40.0, 0.0]])

    fixed = np.abs(_headings_per_segment(Route(xy)))
    assert fixed.max() < math.radians(1.0)

    unfiltered = np.abs(_headings_per_segment(Route(xy, min_segment_m=0.0)))
    assert unfiltered.max() > math.radians(150.0)   # the box flips end-for-end


def test_heading_near_the_end_ignores_a_short_final_segment():
    """The kept-anyway last point must not become the heading at ``s = length``.

    The filter cannot help here — it is the point it is obliged to keep — so
    this is the outward search in `heading_at` doing the work.
    """
    xy = np.vstack([_straight(20, heading=math.pi / 4), [[19.0, 19.005]]])
    route = Route(xy)

    assert _wrap(route.heading_at(route.length) - math.pi / 4) == pytest.approx(
        0.0, abs=math.radians(1.0))


def test_a_pure_noise_cluster_reports_its_axis_not_due_east():
    """Every segment under the floor: the gross first-to-last direction.

    The old fallback was 0.0 rad — due east — which is a specific wrong heading
    that looks like a real one.
    """
    xy = np.array([[0.0, 0.0], [0.02, 0.02], [0.01, 0.03], [0.0, 0.06]])
    route = Route(xy)

    assert len(route.xy) == 2                    # collapsed to its endpoints
    assert route.heading_at(0.0) == pytest.approx(math.pi / 2, abs=1e-9)


def test_two_coincident_points_do_not_raise():
    """A degenerate route still answers, with the one honest value left."""
    route = Route([[3.0, 4.0], [3.0, 4.0]])

    assert route.length == 0.0
    assert route.heading_at(0.0) == 0.0


def test_opting_out_of_the_floor_is_honoured_by_heading_at():
    """``min_segment_m=0`` means "use my points", including for the heading."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.05]])
    route = Route(xy, min_segment_m=0.0)

    assert len(route.xy) == 3
    assert route.heading_at(route.length) == pytest.approx(math.pi / 2, abs=1e-9)


def test_heading_is_clamped_to_the_route_at_both_ends():
    xy = _straight(10, heading=-0.4)
    route = Route(xy)

    for s in (-100.0, 0.0, route.length, route.length + 100.0):
        assert route.heading_at(s) == pytest.approx(-0.4, abs=1e-9)


# --- the rest of the frame, unchanged by the filter ----------------------

def test_a_route_needs_two_points():
    with pytest.raises(ValueError):
        Route([[0.0, 0.0]])


def test_local_coordinates_are_left_positive():
    route = Route(_straight(10))

    long, lat = route.local_coordinates([3.0, 2.0])
    assert long == pytest.approx(3.0, abs=1e-9)
    assert lat == pytest.approx(2.0, abs=1e-9)      # left of due east
    assert route.local_coordinates([3.0, -2.0])[1] == pytest.approx(-2.0, abs=1e-9)


def test_point_on_lane_is_a_lateral_test_against_the_clamped_projection():
    """Half a width either side — and note the longitudinal end is NOT a wall.

    `local_coordinates` clamps to the polyline, so a point off either end
    projects onto the nearest endpoint and its ``long`` lands inside the
    extent by construction. A car behind the start therefore reads as on-lane;
    both call sites gate on ``gap > 0`` afterwards, which is what keeps that
    from mattering.
    """
    route = Route(_straight(10))                    # width ROUTE_WIDTH_M = 2.0

    assert route.point_on_lane([4.0, 0.4 * ROUTE_WIDTH_M])
    assert not route.point_on_lane([4.0, 0.6 * ROUTE_WIDTH_M])
    assert route.point_on_lane([-50.0, 0.0])
    assert not route.point_on_lane([-50.0, 0.6 * ROUTE_WIDTH_M])


def test_corners_are_the_footprint_of_a_rotated_box():
    pts = corners(0.0, 0.0, math.pi / 2, length=4.0, width=2.0)

    xs, ys = zip(*pts)
    assert max(xs) == pytest.approx(1.0, abs=1e-9)  # width now spans x
    assert max(ys) == pytest.approx(2.0, abs=1e-9)  # length now spans y
