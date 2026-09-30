"""Bit-parity gate for the vectorized ``LaneProxy.local_coordinates``.

The 2026-08 perf pass replaced the per-segment Python loop (measured at
~93% of the batch scorer's hot-path time on the banked loop-2 corpus)
with a numpy-vectorized implementation that must be BIT-IDENTICAL — not
merely close: EPDMS candidate argmaxes flip on 1-ulp drift, and the
metric already took its two authorized comparability cuts on 2026-08-18.

``_reference_local_coordinates`` below is a frozen copy of the
pre-vectorization loop. Every query in this module requires exact float
equality (compared via ``float.hex``, so NaN also compares equal) of both
``s`` and ``r``. The offline evidence for the change additionally covers
163k queries against 181 real banked-scenario lanes and a 74-frame
banked-corpus bit-replay; this in-tree sweep is the regression tripwire.
"""

import math
import zlib

import numpy as np
import pytest

from navsafe.evaluation.utils.lane_proxy import LaneProxy
from navsafe.evaluation.utils.lane_proxy_fast import FastLaneProxy


def _reference_local_coordinates(lane: LaneProxy, point) -> tuple:
    """Frozen copy of the pre-vectorization ``LaneProxy.local_coordinates``.

    Do not edit to track the live implementation — its whole value is
    being the original scalar loop the vectorized code must reproduce
    bit-for-bit.
    """
    pt = np.asarray(point, dtype=np.float64)[:2]
    best_s = 0.0
    best_r = float('inf')
    best_dist = float('inf')
    for i in range(len(lane._polyline) - 1):
        a = lane._polyline[i]
        b = lane._polyline[i + 1]
        ab = b - a
        seg_len = lane._seg_lengths[i]
        if seg_len < 1e-12:
            continue
        ap = pt - a
        t = np.dot(ap, ab) / (seg_len * seg_len)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        diff = pt - proj
        dist = float(np.linalg.norm(diff))
        if dist < best_dist:
            s_along = lane._cum_lengths[i] + t * seg_len
            tangent = lane._tangents[i]
            normal_left = np.array([-tangent[1], tangent[0]])
            signed_r = float(np.dot(diff, normal_left))
            best_s = s_along
            best_r = math.copysign(dist, signed_r) if dist > 0.0 else 0.0
            best_dist = dist
    return (best_s, best_r)


def _assert_bit_equal(lane: LaneProxy, point) -> None:
    s_ref, r_ref = _reference_local_coordinates(lane, point)
    s_new, r_new = lane.local_coordinates(point)
    assert float(s_ref).hex() == float(s_new).hex(), (
        f"s drift at {point!r}: {s_ref!r} != {s_new!r}")
    assert float(r_ref).hex() == float(r_new).hex(), (
        f"r drift at {point!r}: {r_ref!r} != {r_new!r}")


def _synthetic_lanes() -> list:
    rng = np.random.default_rng(7)
    lanes = []
    # Straight, diagonal, curved, and jittered random-walk polylines at
    # several scales and orientations.
    lanes.append(LaneProxy("straight", np.array([[0.0, 0.0], [100.0, 0.0]])))
    lanes.append(LaneProxy("diagonal", np.array([[0.0, 0.0], [-30.0, -40.0]])))
    theta = np.linspace(0.0, np.pi / 2, 25)
    lanes.append(LaneProxy(
        "arc", np.stack([50.0 * np.cos(theta), 50.0 * np.sin(theta)], axis=1)))
    walk = np.cumsum(rng.normal(scale=3.0, size=(40, 2)), axis=0) + 1e4
    lanes.append(LaneProxy("walk_offset", walk))
    # Huge coordinate offset: catches vectorized forms whose rounding
    # drifts from the scalar ops when the mantissa is dominated by the
    # offset (e.g. hypot- or fma-based distance).
    walk_far = np.cumsum(rng.normal(scale=3.0, size=(40, 2)), axis=0)
    lanes.append(LaneProxy("walk_far_offset", walk_far + np.array([1e8, -1e8])))
    # Degenerate: repeated vertices (zero-length segments) mixed with real
    # ones, and a sub-threshold segment.
    lanes.append(LaneProxy(
        "degen_mixed",
        np.array([[0.0, 0.0], [0.0, 0.0], [1e-13, 0.0], [5.0, 0.0],
                  [5.0, 5.0]])))
    lanes.append(LaneProxy("degen_only", np.array([[2.0, 3.0], [2.0, 3.0]])))
    return lanes


@pytest.mark.parametrize("lane", _synthetic_lanes(), ids=lambda ln: ln.index)
def test_random_points_bit_equal(lane: LaneProxy) -> None:
    # crc32, not hash(): str hashing is salted per interpreter, which would
    # make every run test a different point set and CI failures unreplayable.
    rng = np.random.default_rng(zlib.crc32(lane.index.encode()))
    pl = lane.polyline
    lo = pl.min(axis=0) - 30.0
    hi = pl.max(axis=0) + 30.0
    for _ in range(400):
        _assert_bit_equal(lane, rng.uniform(lo, hi))


@pytest.mark.parametrize("lane", _synthetic_lanes(), ids=lambda ln: ln.index)
def test_adversarial_points_bit_equal(lane: LaneProxy) -> None:
    pl = lane.polyline
    points = [pl[i].copy() for i in range(len(pl))]              # exact vertices
    points += [(pl[i] + pl[i + 1]) / 2.0 for i in range(len(pl) - 1)]
    points += [pl[0] + (pl[0] - pl[1]) * 3.0,                    # beyond start
               pl[-1] + (pl[-1] - pl[-2]) * 3.0]                 # beyond end
    points += [np.array([0.0, 0.0]), np.array([-0.0, -0.0]),
               pl.mean(axis=0) + 1e6,                            # far away
               pl[0] + np.array([1e-14, -1e-14])]                # ulp nudge
    for point in points:
        _assert_bit_equal(lane, point)


def test_nonfinite_points_bit_equal() -> None:
    lane = LaneProxy("straight", np.array([[0.0, 0.0], [100.0, 0.0]]))
    for point in [np.array([np.nan, 1.0]), np.array([np.inf, 0.0]),
                  np.array([-np.inf, np.nan])]:
        _assert_bit_equal(lane, point)


def test_all_degenerate_returns_untouched_incumbents() -> None:
    lane = LaneProxy("degen_only", np.array([[2.0, 3.0], [2.0, 3.0]]))
    s, r = lane.local_coordinates(np.array([10.0, 10.0]))
    assert s == 0.0
    assert r == float('inf')


# ── FastLaneProxy: the same sweep, against its ACTUAL contract ───────────────
#
# `FastLaneProxy` (navsafe/evaluation/utils/lane_proxy_fast.py) is what the
# registered `pdm_closed_fast` policy — still the default `expert_model_type`
# — puts under the batch scorer.  It had no test at all: its module docstring
# claimed `test_lane_proxy.py` verified it, and that file never imported it.
#
# It is NOT held to the bit-equality above, because it is not bit-equal: it
# computes the point-to-segment distance with `np.hypot` where the frozen loop
# uses `np.linalg.norm`, and those differ at 1 ulp on ~17% of inputs.  That
# ulp is invisible almost everywhere and decisive in two places, so the
# assertions below are graded rather than uniform:
#
#   * unique nearest segment  -> full parity (s bit-exact, r sign, |r| 1 ulp)
#   * segments tied within an ulp -> the argmin is a coin flip; only |r| holds
#   * non-finite query -> NO invariant holds; the two disagree by construction
#     (`_reference_local_coordinates`'s `dist < best_dist` is never true for a
#     NaN, so it keeps its (0.0, inf) incumbents, while `np.argmin` RETURNS
#     the first NaN index).  Pinned below rather than hidden, because the
#     divergence is a fail-BEST inversion in a default code path.


def _reference_segment_distances(lane: LaneProxy, point) -> np.ndarray:
    """Per-segment point-to-segment distances, the frozen loop's way.

    Used to detect a tie *before* asserting, so the strict invariant applies
    everywhere a tie does not — rather than being weakened for every point
    because a handful are ambiguous.
    """
    pt = np.asarray(point, dtype=np.float64)[:2]
    distances = []
    for i in range(len(lane._polyline) - 1):
        seg_len = lane._seg_lengths[i]
        if seg_len < 1e-12:
            distances.append(float('inf'))
            continue
        a = lane._polyline[i]
        ab = lane._polyline[i + 1] - a
        t = max(0.0, min(1.0, np.dot(pt - a, ab) / (seg_len * seg_len)))
        distances.append(float(np.linalg.norm(pt - (a + t * ab))))
    return np.asarray(distances, dtype=np.float64)


def _has_distance_tie(lane: LaneProxy, point) -> bool:
    """True when >1 segment sits within an ulp of the nearest — argmin is luck."""
    distances = _reference_segment_distances(lane, point)
    finite = distances[np.isfinite(distances)]
    if finite.size < 2:
        return False
    best = finite.min()
    return int((finite <= best + math.ulp(best)).sum()) > 1


def _fast_lane(lane: LaneProxy) -> FastLaneProxy:
    """The FastLaneProxy over the same polyline. Build once, reuse per point."""
    return FastLaneProxy(lane.index, lane.polyline)


def _assert_distance_parity(lane: LaneProxy, fast: FastLaneProxy, point) -> None:
    """Weakest invariant, and the only one a tie preserves: |r| agrees.

    Tolerance is 2 ulp, not 1: at a tie the two implementations measure
    genuinely different segments, so the gap carries the distance function's
    own ulp *plus* up to an ulp of real difference between the two segments.
    """
    _, r_ref = _reference_local_coordinates(lane, point)
    _, r_fast = fast.local_coordinates(point)
    if math.isnan(r_ref) or math.isinf(r_ref) or r_ref == 0.0:
        assert float(r_ref).hex() == float(r_fast).hex(), (
            f"r drift at {point!r}: {r_ref!r} != {r_fast!r}")
        return
    gap = abs(abs(r_ref) - abs(r_fast))
    assert gap <= 2.0 * math.ulp(abs(r_ref)), (
        f"|r| drifted more than 2 ulp at {point!r}: {r_ref!r} vs {r_fast!r} "
        f"(gap {gap!r}, ulp {math.ulp(abs(r_ref))!r})")


def _assert_full_parity(lane: LaneProxy, fast: FastLaneProxy, point) -> None:
    """Strongest invariant: same segment won, so s is exact and r keeps its side.

    Holds wherever the nearest segment is unique; |r| is held to 1 ulp here
    (measured worst case over 14k queries is exactly 1.000 ulp) because both
    implementations are measuring the same vector.
    """
    s_ref, r_ref = _reference_local_coordinates(lane, point)
    s_fast, r_fast = fast.local_coordinates(point)
    assert float(s_ref).hex() == float(s_fast).hex(), (
        f"s drift at {point!r}: {s_ref!r} != {s_fast!r}")
    if math.isnan(r_ref) or math.isinf(r_ref) or r_ref == 0.0:
        assert float(r_ref).hex() == float(r_fast).hex(), (
            f"r drift at {point!r}: {r_ref!r} != {r_fast!r}")
        return
    assert math.copysign(1.0, r_ref) == math.copysign(1.0, r_fast), (
        f"r changed sign at {point!r}: {r_ref!r} vs {r_fast!r}")
    gap = abs(abs(r_ref) - abs(r_fast))
    assert gap <= math.ulp(abs(r_ref)), (
        f"|r| drifted more than 1 ulp at {point!r}: {r_ref!r} vs {r_fast!r} "
        f"(gap {gap!r}, ulp {math.ulp(abs(r_ref))!r})")


def _assert_parity(lane: LaneProxy, fast: FastLaneProxy, point) -> None:
    """Full parity where the nearest segment is unique, |r| parity at a tie."""
    if _has_distance_tie(lane, point):
        _assert_distance_parity(lane, fast, point)
    else:
        _assert_full_parity(lane, fast, point)


@pytest.mark.parametrize("lane", _synthetic_lanes(), ids=lambda ln: ln.index)
def test_fast_lane_proxy_parity_on_random_points(lane: LaneProxy) -> None:
    rng = np.random.default_rng(zlib.crc32(lane.index.encode()))
    fast = _fast_lane(lane)
    pl = lane.polyline
    lo = pl.min(axis=0) - 30.0
    hi = pl.max(axis=0) + 30.0
    for _ in range(400):
        _assert_parity(lane, fast, rng.uniform(lo, hi))


@pytest.mark.parametrize("lane", _synthetic_lanes(), ids=lambda ln: ln.index)
def test_fast_lane_proxy_parity_on_adversarial_points(lane: LaneProxy) -> None:
    """Vertices, midpoints, extrapolations, ulp nudges — full parity unless tied.

    Only the arc's centre ties (1 of 267 points across the seven lanes), so the
    tie branch is an exemption the code detects, not a blanket weakening: an
    inverted sign, a wrong tangent or an off-by-one on ``cum_lengths`` still
    fails here, at exactly the vertices and clamped endpoints where such bugs
    live.
    """
    fast = _fast_lane(lane)
    pl = lane.polyline
    points = [pl[i].copy() for i in range(len(pl))]
    points += [(pl[i] + pl[i + 1]) / 2.0 for i in range(len(pl) - 1)]
    points += [pl[0] + (pl[0] - pl[1]) * 3.0,
               pl[-1] + (pl[-1] - pl[-2]) * 3.0]
    points += [np.array([0.0, 0.0]), np.array([-0.0, -0.0]),
               pl.mean(axis=0) + 1e6,
               pl[0] + np.array([1e-14, -1e-14])]
    for point in points:
        _assert_parity(lane, fast, point)


def test_fast_lane_proxy_breaks_near_ties_differently() -> None:
    """Where segments tie within an ulp, the two pick different ones.

    The arc's centre is equidistant from all 24 of its segments (total spread
    7.1e-15 == 1 ulp), so the argmin is decided purely by `np.hypot` vs
    `np.linalg.norm` rounding and the reported longitudinal position moves by
    ~26 m. This is why `FastLaneProxy` is not a drop-in for the reference.

    Asserted over a NEIGHBOURHOOD rather than the single centre point: which
    individual query flips is libm/BLAS/CPU-dependent and would fire spuriously
    on another machine, whereas "somewhere in this neighbourhood they still
    disagree" is the stable fact the two docstrings rest on.
    """
    theta = np.linspace(0.0, np.pi / 2, 25)
    lane = LaneProxy("arc", np.stack([50.0 * np.cos(theta), 50.0 * np.sin(theta)], axis=1))
    fast = _fast_lane(lane)
    rng = np.random.default_rng(11)
    neighbourhood = [np.zeros(2)] + [rng.normal(scale=1e-9, size=2) for _ in range(200)]

    diverged = [
        point for point in neighbourhood
        if float(_reference_local_coordinates(lane, point)[0]).hex()
        != float(fast.local_coordinates(point)[0]).hex()
    ]
    assert diverged, (
        "FastLaneProxy no longer breaks the arc-centre near-tie differently "
        "anywhere in this neighbourhood. Re-point or delete this test, and "
        "update the divergence claims in lane_proxy_fast.py and "
        "PDMClosedFastAdapter to match.")

    # ...and throughout the neighbourhood the distance is still invariant.
    for point in neighbourhood:
        _assert_distance_parity(lane, fast, point)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_fast_lane_proxy_fails_worst_on_non_finite_points() -> None:
    """Non-finite queries must fail WORST, identically to the reference.

    Until the guard in `FastLaneProxy.local_coordinates` this inverted: the
    NaN propagated into `dist`, `np.argmin` returned the FIRST NaN index
    (NaN compares False against everything), and `best_dist > 0.0` is False
    for a NaN, so `r` collapsed to `0.0` — "exactly on the centre-line", the
    most in-lane value there is. `+inf` produced the other shape of the same
    defect, a finite `s` with `r = -inf`.

    That path runs through the default `expert_model_type`
    (`pdm_closed_fast`), so a non-finite pose scored lane-keeping and
    driving-direction compliance as perfect rather than unusable — against
    the fail-worst policy established in 995a03a.
    """
    lane = LaneProxy("straight", np.array([[0.0, 0.0], [100.0, 0.0]]))
    fast = _fast_lane(lane)

    for point in [np.array([np.nan, 1.0]), np.array([-np.inf, np.nan]),
                  np.array([1.0, np.nan]), np.array([np.inf, 0.0])]:
        ref = _reference_local_coordinates(lane, point)
        assert ref == (0.0, float('inf'))
        assert fast.local_coordinates(point) == ref, (
            f"non-finite query {point!r} must fail worst like the reference")


def test_fast_lane_proxy_degenerate_lane_matches_reference() -> None:
    """The all-degenerate escape hatch returns the untouched incumbents."""
    lane = _fast_lane(LaneProxy("degen_only", np.array([[2.0, 3.0], [2.0, 3.0]])))
    s, r = lane.local_coordinates(np.array([10.0, 10.0]))
    assert s == 0.0
    assert r == float('inf')
