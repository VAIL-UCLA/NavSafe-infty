"""
LaneProxy — lightweight lane wrapper for scenario map_features data.

Provides the same interface as MetaDrive lane objects (local_coordinates,
heading_at, shapely_polygon, distance, length, index) so that the EPDMS scorers
can work with lane data extracted directly from scenario_data['map_features']
without requiring a MetaDrive engine/road_network.

Also owns :class:`DrivableAreaProxy` and :func:`corners_in_drivable_area` —
THE drivable-area predicate, shared by both EPDMS scorers (the live metric of
record and the batch proposal engine) so the two cannot drift apart again.
"""

import logging
import math
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# Default lane half-width used when no polygon is provided
_DEFAULT_LANE_HALF_WIDTH = 1.75  # 3.5m lane

# Tolerance the lane-polygon union is grown by before DAC containment is
# tested. Lane pieces are butt-jointed (and, with no map polygon, are
# flat-capped centerline buffers), so adjacent pieces meet on a zero-width
# seam that a corner can land exactly on. NavSim's DAC tests a single
# drivable-surface layer with no such seams; this buffer approximates it.
DRIVABLE_AREA_BUFFER_M = 0.3


class LaneProxy:
    """Wraps a scenario lane polyline to provide the MetaDrive lane interface.

    Required interface methods used by the EPDMS scorers:
        - local_coordinates(point) -> (s, r)
        - heading_at(s) -> np.ndarray  (unit vector)
        - distance(point) -> float
        - length -> float
        - index -> str
        - shapely_polygon -> Polygon
    """

    def __init__(self, lane_id: str, polyline: np.ndarray,
                 polygon: Optional[np.ndarray] = None, *,
                 is_intersection: bool = False,
                 lane_group_id: Optional[str] = None):
        """
        Args:
            lane_id: Unique identifier for this lane.
            polyline: Center-line points, shape (N, 2) or (N, 3). Only XY used.
            polygon: Optional boundary polygon, shape (M, 2) or (M, 3).
        """
        self.index = lane_id
        # Semantic-map adapter used by CaRL's TTC eligibility rule. py123d
        # retains the source lane-group/intersection relationship; the
        # ScenarioDescription conversion forwards it as this explicit flag.
        self.is_intersection = bool(is_intersection)
        # Closest ScenarioNet/py123d analogue of nuPlan's parent roadblock.
        # Route-loop removal must operate on this topology level rather than
        # treating crossing/parallel connector lanes as separate roadblocks.
        self.lane_group_id = (
            None if lane_group_id is None else str(lane_group_id)
        )
        self._polyline = np.asarray(polyline, dtype=np.float64)[:, :2]

        if len(self._polyline) < 2:
            raise ValueError(f"Lane {lane_id} polyline must have >= 2 points")

        # Pre-compute cumulative arc-length along the center-line
        diffs = np.diff(self._polyline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._seg_lengths = seg_lengths
        self._cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.length = float(self._cum_lengths[-1])

        # Pre-compute per-segment unit tangent vectors
        with np.errstate(divide='ignore', invalid='ignore'):
            self._tangents = np.where(
                seg_lengths[:, None] > 1e-12,
                diffs / seg_lengths[:, None],
                np.zeros_like(diffs),
            )

        # Build the center-line as a Shapely LineString for distance queries
        self._center_line = LineString(self._polyline)

        # Build shapely polygon
        if polygon is not None and len(polygon) >= 3:
            poly_pts = np.asarray(polygon, dtype=np.float64)[:, :2]
            self._shapely_polygon = Polygon(poly_pts)
            if not self._shapely_polygon.is_valid:
                self._shapely_polygon = self._shapely_polygon.buffer(0)
        else:
            # Extrude center-line into a polygon using default lane width
            self._shapely_polygon = self._center_line.buffer(
                _DEFAULT_LANE_HALF_WIDTH, cap_style=2  # flat caps
            )

    @property
    def shapely_polygon(self) -> Polygon:
        return self._shapely_polygon

    @property
    def polyline(self) -> np.ndarray:
        """Return the lane center-line as ``(N, 2)`` float64 (read-only view).

        This is the raw center-line that was passed to the constructor.
        Callers must not mutate the returned array; ``LaneProxy`` caches
        derived quantities (cumulative arc-length, segment tangents,
        the Shapely polygon) at construction time and a mutation here
        would silently invalidate them.
        """
        view = self._polyline.view()
        view.flags.writeable = False
        return view

    @property
    def cum_lengths(self) -> np.ndarray:
        """Cumulative arc-length per polyline vertex, shape ``(N,)`` (read-only)."""
        view = self._cum_lengths.view()
        view.flags.writeable = False
        return view

    def local_coordinates(self, point: np.ndarray) -> Tuple[float, float]:
        """Project *point* onto the lane center-line.

        Returns:
            (s, r) where s is the longitudinal distance along the lane and
            r is the signed lateral offset (positive = left of travel direction).

        Vectorised over all segments at once (2026-08 perf pass: the
        per-segment Python loop this replaces was ~93% of the batch
        scorer's hot-path time on the banked loop-2 corpus). BIT-IDENTICAL
        to that loop by construction, not merely close: every element-wise
        form below was chosen because it reproduces the scalar op exactly
        on this platform (``x0*y0 + x1*y1`` == 2-vector ``np.dot``;
        ``sqrt(dx*dx + dy*dy)`` == ``np.linalg.norm`` — NOT ``np.hypot``,
        which differs at 1 ulp on ~17% of inputs; ``maximum(0, minimum(1,
        t)) + 0.0`` == Python ``max(0.0, min(1.0, t))`` including the
        ``t == -0.0`` case; first-index ``argmin`` == the loop's strict
        ``<`` incumbent update), verified by the exact-equality sweep in
        ``tests/evaluation/test_lane_proxy_vectorized_parity.py`` and a
        74-frame banked-corpus bit-replay. Do not "simplify" these forms
        without re-running that evidence.

        Parity bound: the guarantee holds while ``seg_len * seg_len`` and
        the dot product stay finite (segment lengths below ~1.3e154 —
        i.e. any physical map, where coordinates are metres). Past that
        overflow point ``t`` becomes inf/inf = NaN and the two clamps
        legitimately diverge (Python's ``min(1.0, nan)`` returns 1.0;
        ``np.minimum`` propagates the NaN, which the inf guard then
        discards).
        """
        pt = np.asarray(point, dtype=np.float64)[:2]

        # Compare true point-to-segment distances (NOT the signed lateral
        # component of the incumbent: at a clamped endpoint the offset
        # vector is mostly tangential, so its normal component is near
        # zero even when the point is far away — mixing the two metrics
        # let far segments win and made points beyond the lane's extent
        # report r ≈ 0).
        a = self._polyline[:-1]                       # (N-1, 2)
        ab = self._polyline[1:] - a                   # (N-1, 2)
        seg_len = self._seg_lengths                   # (N-1,)
        # The scalar loop skips seg_len < 1e-12 segments entirely.
        valid = seg_len >= 1e-12

        apx = pt[0] - a[:, 0]
        apy = pt[1] - a[:, 1]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            t = (apx * ab[:, 0] + apy * ab[:, 1]) / (seg_len * seg_len)
            # ``+ 0.0`` maps the -0.0 that numpy's maximum keeps for a
            # t == -0.0 input to the +0.0 Python's max(0.0, ...) returns.
            t = np.maximum(0.0, np.minimum(1.0, t)) + 0.0
            dx = pt[0] - (a[:, 0] + t * ab[:, 0])
            dy = pt[1] - (a[:, 1] + t * ab[:, 1])
            dist = np.sqrt(dx * dx + dy * dy)
        dist = np.where(valid, dist, np.inf)
        # NaN distances (non-finite inputs) never beat the incumbent in
        # the scalar loop's strict-< scan; +inf reproduces that under
        # argmin, which would otherwise select the first NaN.
        dist = np.where(np.isnan(dist), np.inf, dist)

        # First minimal index == the loop's strict "<" incumbent update.
        i = int(np.argmin(dist))
        best_dist = float(dist[i])
        if not best_dist < np.inf:
            # No selectable segment (all degenerate / unscoreable): the
            # loop leaves its (0.0, inf) incumbents untouched.
            return (0.0, float('inf'))

        best_s = float(self._cum_lengths[i] + t[i] * seg_len[i])
        # Signed lateral: positive = left of travel direction. Magnitude
        # is the true distance so |r| stays honest for points beyond the
        # lane's longitudinal extent.
        tangent = self._tangents[i]
        normal_left = np.array([-tangent[1], tangent[0]])
        signed_r = float(np.dot(np.array([dx[i], dy[i]]), normal_left))
        best_r = math.copysign(best_dist, signed_r) if best_dist > 0.0 else 0.0

        return (best_s, best_r)

    def heading_at(self, s: float) -> np.ndarray:
        """Return the unit tangent vector at longitudinal position *s*.

        Returns:
            np.ndarray of shape (2,) — the heading direction vector.
        """
        s = max(0.0, min(s, self.length))

        # Binary search for the segment containing s
        idx = int(np.searchsorted(self._cum_lengths, s, side='right')) - 1
        idx = max(0, min(idx, len(self._tangents) - 1))

        return self._tangents[idx].copy()

    def distance(self, point: np.ndarray) -> float:
        """Return the perpendicular distance from *point* to the center-line."""
        pt = Point(float(point[0]), float(point[1]))
        return self._center_line.distance(pt)


class DrivableAreaProxy:
    """Prepared union of every lane polygon, grown by a seam tolerance.

    This is the drivable-area surface the DAC term is scored against. It
    exists because testing corners against lane polygons ONE PIECE AT A TIME
    has no tolerance for the seams between pieces or for a footprint that
    legitimately overhangs its lane: the pieces are butt-jointed (and, with no
    map polygon, are 3.5 m flat-capped centerline strips), so a corner landing
    on a shared boundary, in the gap between two abutting pieces, or a few
    tens of centimetres outside a strip is inside no piece at all. Unioning
    dissolves the seams; the buffer supplies the tolerance.

    Build once per scenario (the union is the expensive part) and reuse: the
    prepared geometry makes each subsequent containment query cheap."""

    __slots__ = ("_union", "_prepared")

    def __init__(self, union: Any) -> None:
        from shapely.prepared import prep
        self._union = union
        self._prepared = prep(union)

    @classmethod
    def from_lane_polygons(
        cls,
        polygons: Iterable[Polygon],
        *,
        buffer_m: float = DRIVABLE_AREA_BUFFER_M,
    ) -> Optional['DrivableAreaProxy']:
        """Union *polygons*, grow by *buffer_m*, and prepare the result.

        Returns ``None`` when there is nothing to union or the geometry
        operation fails, which is the caller's signal to fall back to the
        per-lane containment test.
        """
        polys = [p for p in polygons if p is not None]
        if not polys:
            return None
        try:
            from shapely.ops import unary_union
            return cls(unary_union(polys).buffer(buffer_m))
        except Exception:
            # Degenerate / self-intersecting map geometry: the caller falls
            # back to per-lane containment rather than losing DAC entirely.
            return None

    @property
    def geometry(self) -> Any:
        """The unprepared union geometry (for area/plotting/debug queries)."""
        return self._union

    def contains_point(self, x: float, y: float) -> bool:
        """Is ``(x, y)`` strictly inside the drivable surface?"""
        return bool(self._prepared.contains(Point(x, y)))

    def __getstate__(self) -> Any:
        # shapely's PreparedGeometry is not picklable; carry the union and
        # re-prepare on load so a scorer holding one stays serializable.
        return self._union

    def __setstate__(self, state: Any) -> None:
        from shapely.prepared import prep
        self._union = state
        self._prepared = prep(state)


def corners_in_drivable_area(
    corners: Sequence[Tuple[float, float]],
    drivable: Optional[DrivableAreaProxy],
    fallback_polygons: Callable[[], Iterable[Polygon]],
) -> bool:
    """Are ALL *corners* of an ego footprint inside the drivable area?

    THE shared DAC predicate. Both EPDMS scorers call this; do not inline a
    second copy — the per-piece test below used to be the batch scorer's only
    implementation while the live metric used the union, so the teacher was
    optimizing a different constraint from the one it was graded on.

    Args:
        corners: the footprint's corner coordinates (the polygon exterior
            without its repeated closing vertex).
        drivable: the prepared union proxy, or ``None`` when one could not be
            built (no lanes, or degenerate geometry).
        fallback_polygons: called only when *drivable* is ``None`` — yields
            the lane polygons to test corners against one piece at a time.
            A callable so the caller's nearby-lane query is skipped entirely
            on the normal path.
    """
    if drivable is not None:
        return all(drivable.contains_point(cx, cy) for cx, cy in corners)

    polygons = list(fallback_polygons())
    for cx, cy in corners:
        pt = Point(cx, cy)
        if not any(poly.contains(pt) for poly in polygons):
            return False
    return True


def build_lanes_from_scenario(scenario_data: dict) -> List[Tuple['LaneProxy', Polygon]]:
    """Extract lane features from scenario_data['map_features'] and return
    a list of (LaneProxy, shapely_polygon) tuples compatible with
    EPDMSLiveScorer.all_lanes.

    Only features whose type passes MetaDriveType.is_lane() are included.
    """
    from navsafe.scenario.type import MetaDriveType

    lanes = []
    map_features = scenario_data.get('map_features', {})

    for feature_id, feature_data in map_features.items():
        feature_type = feature_data.get('type', '')
        if not MetaDriveType.is_lane(feature_type):
            continue

        polyline = feature_data.get('polyline')
        if polyline is None or len(polyline) < 2:
            continue

        polygon = feature_data.get('polygon')

        try:
            proxy = LaneProxy(
                lane_id=str(feature_id),
                polyline=np.asarray(polyline),
                polygon=np.asarray(polygon) if polygon is not None else None,
                is_intersection=bool(feature_data.get("is_intersection", False)),
                lane_group_id=feature_data.get("lane_group_id"),
            )
            lanes.append((proxy, proxy.shapely_polygon))
        except Exception:
            # Malformed lane data. Observed failure modes: ValueError /
            # IndexError from ragged or mis-shaped arrays, and
            # shapely GEOSException from degenerate map polygons — the
            # last is why the catch stays broad. Skipping the lane is
            # correct (one bad feature must not lose the whole map), but
            # it must not be silent: enough skipped lanes leave the
            # scorer lane-less, which gates every candidate's dac to 0.
            logger.warning(
                "build_lanes_from_scenario: skipping malformed lane "
                "feature %r", feature_id, exc_info=True)
            continue

    return lanes
