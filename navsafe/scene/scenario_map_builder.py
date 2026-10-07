# Copyright (c) 2022-2025, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Universal Map Builder for ScenarioDescription format in IsaacSim.
Converts map features (lanes, markings, crosswalks) to USD prims.
"""

import logging

import numpy as np
from typing import Dict, List, Tuple, Optional

from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.type import MetaDriveType
from navsafe.scene.usd_mesh import define_flat_mesh

# IsaacSim / USD imports are deferred to build() to avoid module-load failures
# when IsaacSim is not yet fully initialized.

logger = logging.getLogger(__name__)


class _CoverIndex:
    """Spatially indexed "already painted" region for lane-line dedup.

    The dedup rule is unchanged: a boundary is skipped when most of it already
    lies under previously painted geometry. Only the bookkeeping changed.

    The previous accumulator was a single shapely geometry rebuilt with
    ``covered.union(buf)`` once per drawn boundary, and each candidate was
    tested with ``seg.intersection(covered)`` against that growing union. Both
    costs scale with the accumulated geometry, so the pass is quadratic --
    fine for a Real2Sim clip, fatal for a whole-log nuPlan map (9,166 boundary
    polylines over 626,492 segments), where map building never finished inside
    a 15-minute watchdog.

    Here the buffers are kept as a list behind an STRtree, and a candidate is
    intersected only against the entries whose envelopes overlap it. That is
    exactly equivalent: geometry the tree does not return cannot intersect the
    segment, so it contributes nothing to the intersection length either way.
    The tree is immutable in shapely 2, so it is rebuilt on doubling, which
    amortizes to O(1) insertions.
    """

    #: Entries added since the last rebuild are scanned linearly, so this caps
    #: that scan. Rebuilding on doubling instead would leave a tail of up to
    #: half the entries unindexed, which is the quadratic scan this class
    #: exists to remove.
    _MAX_PENDING = 64

    __slots__ = ("_indexed", "_pending", "_tree")

    def __init__(self, initial=None) -> None:
        self._indexed: list = []
        self._pending: list = []
        self._tree = None
        if initial is not None:
            self.add(initial)

    def add(self, geometry) -> None:
        if geometry is None or geometry.is_empty:
            return
        # Index the PARTS of a multi-part geometry, never the whole. The
        # painted-line seed arrives as a union over every explicit road_line on
        # the map, so its envelope spans the map: indexed whole, the tree
        # returns it for every query and prunes nothing, and each query pays an
        # intersection against the entire union. That defeats the index for the
        # one entry most likely to be huge.
        parts = getattr(geometry, "geoms", None)
        if parts is not None:
            for part in parts:
                if not part.is_empty:
                    self._pending.append(part)
        else:
            self._pending.append(geometry)
        if len(self._pending) >= self._MAX_PENDING:
            self._rebuild()

    def _rebuild(self) -> None:
        self._indexed.extend(self._pending)
        self._pending.clear()
        try:
            from shapely import STRtree
        except Exception:  # pragma: no cover - shapely <2 lacks STRtree here
            self._tree = None
            return
        self._tree = STRtree(self._indexed)

    def covers(self, line) -> bool:
        """True when more than 60% of ``line`` lies under the painted region."""
        if not self._indexed and not self._pending:
            return False
        if self._tree is None and self._indexed:  # pragma: no cover - no STRtree
            candidates = list(self._indexed)
        elif self._tree is not None:
            candidates = [self._indexed[i] for i in self._tree.query(line)]
        else:
            candidates = []
        # Everything added since the last rebuild is invisible to the tree, so
        # it is checked directly. Missing these would let a boundary be drawn
        # twice over geometry that was painted moments earlier. The bbox
        # prefilter is vectorized so the pending scan costs one shapely call,
        # not one per entry -- the tree's query already prefilters the rest.
        if self._pending:
            try:
                import shapely

                hits = shapely.intersects(np.asarray(self._pending, dtype=object), line)
                candidates.extend(
                    geometry for geometry, hit in zip(self._pending, hits) if hit
                )
            except Exception:  # pragma: no cover - non-vectorized fallback
                candidates.extend(self._pending)
        if not candidates:
            return False
        threshold = 0.6 * line.length

        import shapely
        from shapely.ops import unary_union

        # Clip against each candidate first. The results are 1-D pieces lying
        # on `line`; unioning THOSE is equivalent to unioning the 2-D buffers
        # and then clipping once, but enormously cheaper -- the buffers are
        # dense overlapping polygons around every painted boundary, and
        # unioning them per query is what kept map building past its watchdog
        # even after the tree removed the quadratic scan.
        pieces = shapely.intersection(np.asarray(candidates, dtype=object), line)
        lengths = shapely.length(pieces)
        # Two bounds decide almost every query without any union at all:
        # the union is never longer than the sum, and never shorter than the
        # largest single piece.
        total = float(lengths.sum())
        if total <= threshold:
            return False
        if float(lengths.max()) > threshold:
            return True
        keep = [piece for piece, length in zip(pieces, lengths) if length > 0.0]
        if not keep:
            return False
        return unary_union(keep).length > threshold


class ScenarioMapBuilder:
    """
    Build IsaacSim USD scene from ScenarioDescription map features.

    Creates visual and collision geometry for:
    - Road lanes (surfaces)
    - Lane markings (white/yellow lines)
    - Crosswalks
    - Sidewalks
    - Road boundaries
    """

    def __init__(self, scenario: SD, stage):
        """Build simulation map geometry from a scenario on the given USD stage."""
        self.scenario = scenario
        self.stage = stage
        self.map_features = scenario[SD.MAP_FEATURES]

        # Configuration
        self.lane_height = 0.01  # Height of lane surface above ground (1cm)
        # Markings sit 4cm above the asphalt (was 1cm) so the dark road surface
        # never wins the depth test and hides the paint (z-fighting). 4cm is too
        # small to read as floating from a car camera but big enough for stable
        # depth at far range.
        self.marking_height = 0.05
        # Lane lines are physically ~15cm, but at a driving camera with the
        # 0.5 resolution scale a 15cm ribbon is sub-pixel beyond a few metres and
        # MIP-averages into the asphalt — the line is "there" but invisible. Draw
        # them wider so they actually read; the yellow centre line gets an extra
        # bump (``centerline_width``) since it is the primary divider.
        self.marking_width = 0.22  # white edge/lane lines
        self.centerline_width = 0.30  # yellow centre line (more prominent)
        self.crosswalk_height = 0.015  # Height of crosswalk
        # Draw every lane's left/right boundary polyline as a white lane line.
        # AV2 only paints a handful of explicit road_line markings, but each lane
        # carries boundary polylines (the geometric lane edges); rendering them
        # gives the road proper lane structure. Deduped against each other and
        # against the explicit road_line markings so nothing is drawn twice.
        self.draw_lane_boundaries = True
        self.lane_line_width = 0.15  # lane-edge lines (thinner than painted lines)
        # Lanes shorter than this are intersection connectors — their boundaries
        # are skipped (real junctions carry no painted lane lines).
        self.junction_lane_m = 9.0
        # No-paint junction zone = lane-centreline crossing points buffered by
        # this radius; lane lines are clipped to outside it so nothing is drawn
        # across the intersection interior.
        self.junction_buffer_m = 8.0
        # A crossing within this distance of either lane's endpoint is treated as
        # an end-to-end connectivity joint, not an intersection (so it does not
        # create a no-paint gap in an otherwise-straight road).
        self.junction_endpoint_eps = 1.0
        # Perpendicular probe distance for the interior-vs-edge test: asphalt on
        # both sides → interior divider (dashed); one side → road edge (solid).
        self.lane_line_probe_m = 1.6

        # Created prims (for cleanup)
        self.created_prims: list[str] = []
        # Drivable polygon (shapely), set by _build_unified_road; consumed by
        # grass scatter to avoid the asphalt. None until the road is built.
        self.road_polygon = None


    def build_map(self, env_id: int, parent_prim_path: str = "/World") -> Dict[str, List]:
        """
        Build complete map for one environment.

        Args:
            env_id: Environment index
            parent_prim_path: Parent prim path for map elements

        Returns:
            Dict mapping feature type to list of created prim paths
        """
        # Lazy Isaac/USD imports — only available after IsaacSim is initialized
        from pxr import UsdGeom, Gf, Sdf  # noqa: F401

        created: dict[str, list[str]] = {
            'lanes': [],
            'lane_markings': [],
            'crosswalks': [],
            'sidewalks': [],
            'boundaries': []
        }

        # Create parent scope for this map (side effect: defines the prim).
        map_path = f"{parent_prim_path}/envs/env_{env_id}/map"
        UsdGeom.Scope.Define(self.stage, map_path)

        # Build each feature type. Lanes are collected and merged into a single
        # asphalt surface (below) instead of built one ribbon at a time —
        # per-lane ribbons leave grass showing through the seams between
        # adjacent lanes and across the intersection interior.
        lane_features = []
        # Polylines of the explicit painted lines we draw, so lane-boundary lines
        # (below) skip any boundary that is geometrically already painted.
        painted_polylines: List[np.ndarray] = []
        for feature_id, feature_data in self.map_features.items():
            feature_type = feature_data[SD.TYPE]

            try:
                if MetaDriveType.is_lane(feature_type):
                    lane_features.append((feature_id, feature_data))

                elif MetaDriveType.is_road_line(feature_type):
                    prim_path = self._build_lane_marking(feature_id, feature_data, env_id, map_path)
                    if prim_path:
                        created['lane_markings'].append(prim_path)
                        pl = feature_data.get(SD.POLYLINE)
                        if pl is not None and len(np.asarray(pl)) >= 2:
                            painted_polylines.append(np.asarray(pl, float)[:, :2])

                elif MetaDriveType.is_crosswalk(feature_type):
                    prim_path = self._build_crosswalk(feature_id, feature_data, env_id, map_path)
                    if prim_path:
                        created['crosswalks'].append(prim_path)

                elif MetaDriveType.is_sidewalk(feature_type):
                    prim_path = self._build_sidewalk(feature_id, feature_data, env_id, map_path)
                    if prim_path:
                        created['sidewalks'].append(prim_path)

                elif MetaDriveType.is_road_boundary_line(feature_type):
                    prim_path = self._build_boundary(feature_id, feature_data, env_id, map_path)
                    if prim_path:
                        created['boundaries'].append(prim_path)

            except Exception as e:
                logger.warning(f"[Warning] Failed to build feature {feature_id}: {e}")
                continue

        # Merge all lanes into one continuous asphalt surface.
        try:
            road_prim = self._build_unified_road(lane_features, env_id, map_path)
            if road_prim:
                created['lanes'].append(road_prim)
        except Exception as e:
            logger.warning(f"[Warning] Failed to build unified road: {e}")
            for fid, fd in lane_features:
                p = self._build_lane(fid, fd, env_id, map_path)
                if p:
                    created['lanes'].append(p)

        # Lane-edge lines from the per-lane boundary polylines (AV2 paints only a
        # few explicit road_lines, but every lane carries boundary geometry).
        if self.draw_lane_boundaries:
            try:
                lines = self._build_lane_boundary_lines(
                    lane_features, env_id, map_path,
                    painted_polylines=painted_polylines)
                created['lane_markings'].extend(lines)
            except Exception as e:
                logger.warning(f"[Warning] Failed to build lane boundary lines: {e}")

        self.created_prims.extend([p for prims in created.values() for p in prims])

        return created

    @staticmethod
    def _boundary_signature(arr: np.ndarray):
        """Order-independent key for a polyline (rounded to 0.1 m).

        Adjacent lanes may store a shared boundary in opposite directions, so the
        signature is the lexicographic min of the point tuple and its reverse —
        the same edge maps to one key, so it is deduped to a single drawn line."""
        pts = tuple((round(float(x), 1), round(float(y), 1)) for x, y in arr)
        return min(pts, tuple(reversed(pts)))

    def _junction_region(self, lane_features):
        """No-paint intersection zone: lane-centreline crossing points buffered.

        Where two lane centrelines genuinely *cross* is an intersection interior;
        real roads carry no painted lane lines across a junction, so lane lines
        are clipped to outside this region (removing the spaghetti knot where
        every approach's boundaries overlap). Crossings that sit on a lane
        endpoint are skipped — those are mere end-to-end connectivity joints, not
        intersections, and buffering them would punch gaps into straight roads.
        Returns a shapely geometry, or ``None`` (shapely missing / no crossings).
        """
        try:
            from shapely.geometry import LineString, MultiPoint, Point
        except Exception:
            return None
        centrelines = []
        for _fid, fd in lane_features:
            arr = self._as_xy(fd.get(SD.POLYLINE))
            if arr is not None:
                centrelines.append(LineString(arr))
        ends = [(Point(cl.coords[0]), Point(cl.coords[-1])) for cl in centrelines]
        eps = self.junction_endpoint_eps
        xpts = []
        for i in range(len(centrelines)):
            for j in range(i + 1, len(centrelines)):
                inter = centrelines[i].intersection(centrelines[j])
                if inter.is_empty:
                    continue
                for g in getattr(inter, "geoms", [inter]):
                    if g.geom_type != "Point":
                        continue  # collinear overlaps aren't crossings
                    if any(g.distance(e) < eps for e in (*ends[i], *ends[j])):
                        continue  # endpoint connectivity joint, not a crossing
                    xpts.append((g.x, g.y))
        if not xpts:
            return None
        return MultiPoint(xpts).buffer(self.junction_buffer_m)

    def _prepared_road(self):
        """``road_polygon`` with a spatial index built once and reused.

        ``road_polygon`` is a ``unary_union`` over every lane, so on a
        whole-log map (4583 lanes for a nuPlan-val scene) each unprepared
        ``contains`` walks the entire multipolygon. This predicate runs twice
        per segment of every boundary polyline, which is tens of millions of
        scans — measured as a map build that never finished inside a 15-minute
        watchdog. Preparing indexes the geometry once; the predicate results
        are identical.
        """
        road = self.road_polygon
        if road is None:
            return None
        cached = getattr(self, "_road_polygon_prepared", None)
        if cached is not None and cached[0] is road:
            return cached[1]
        try:
            from shapely import prepare
        except Exception:  # pragma: no cover - shapely <2 lacks the top-level API
            self._road_polygon_prepared = (road, road)
            return road
        prepare(road)  # in place; subsequent predicates use the index
        self._road_polygon_prepared = (road, road)
        return road

    def _is_interior_divider(self, arr: np.ndarray) -> bool:
        """True when there is asphalt on BOTH sides of the line.

        Interior dividers (between two lanes) → dashed white; road edges (asphalt
        on one side only) → solid white. Uses :attr:`road_polygon`; falls back to
        ``False`` (solid) when it is unavailable."""
        road = self._prepared_road()
        if road is None:
            return False
        try:
            import shapely
            from shapely import points as shapely_points
        except Exception:
            return False
        if len(arr) < 2:
            return False
        # One vectorized predicate call, not two per segment. shapely 2 takes
        # arrays; probing point-by-point from Python spends nearly all its time
        # in call overhead, and on a whole-log map this polyline set is ~626k
        # segments -> ~1.25M round trips, which is what pushed a map build past
        # its 15-minute watchdog. The geometry and the verdict are unchanged.
        start, end = arr[:-1], arr[1:]
        delta = end - start
        normals = np.stack([-delta[:, 1], delta[:, 0]], axis=1)
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 1e-6
        if not np.any(keep):
            return False
        normals = normals[keep] / lengths[keep, None]
        midpoints = ((start[keep] + end[keep]) / 2.0)
        off = self.lane_line_probe_m
        left = midpoints + normals * off
        right = midpoints - normals * off
        probes = shapely_points(np.concatenate([left, right], axis=0))
        inside = shapely.contains(road, probes)
        half = len(left)
        both = int(np.count_nonzero(inside[:half] & inside[half:]))
        return bool(half) and both / half > 0.6

    def _clip_outside_junction(self, arr: np.ndarray, junction) -> List[np.ndarray]:
        """Return the parts of polyline ``arr`` that lie OUTSIDE ``junction``."""
        if junction is None:
            return [arr]
        try:
            from shapely.geometry import LineString
        except Exception:
            return [arr]
        diff = LineString(arr).difference(junction)
        out = []
        for g in getattr(diff, "geoms", [diff]):
            if g.is_empty or g.length < 1.0:
                continue
            out.append(np.asarray(g.coords))
        return out

    def _build_ribbon_prim(self, prim_path: str, polyline, *, width: float,
                           dashed: bool, color, env_id: int) -> Optional[str]:
        """Build one marking mesh — a solid ribbon or a dashed line — and bind
        emissive paint. Shared by the explicit road_line markings and the derived
        lane-boundary lines (one source of truth for the ribbon/mesh/material).
        Returns the prim path, or ``None`` when the geometry is empty.
        """
        from pxr import UsdGeom  # noqa: lazy
        if dashed:
            dashes = self._make_broken_line(polyline)
            vertices: list = []
            faces: list = []
            for k in range(0, len(dashes) - 1, 2):
                piece = dashes[k:k + 2]
                if len(piece) < 2:
                    continue
                v, f = self._polyline_to_ribbon(
                    piece, width=width, height=self.marking_height)
                base = len(vertices)
                vertices.extend(v)
                faces.extend([[idx + base for idx in face] for face in f])
            if not vertices:
                return None
        else:
            vertices, faces = self._polyline_to_ribbon(
                polyline, width=width, height=self.marking_height)
        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])
        self._apply_material(prim_path, color=color, roughness=0.5,
                             surface_type="marking", env_id=env_id)
        return prim_path

    @staticmethod
    def _as_xy(arr) -> Optional[np.ndarray]:
        """Coerce a polyline payload to an ``(N>=2, 2)`` float array, or ``None``.

        Guards against schemas where boundaries are not plain coordinate arrays
        (e.g. generic ScenarioNet lists-of-dicts), which would otherwise raise."""
        if arr is None:
            return None
        try:
            a = np.asarray(arr, dtype=float)
        except (ValueError, TypeError):
            return None
        if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
            return None
        return a[:, :2]

    def _painted_cover(self, polylines):
        """Buffered union of the already-painted lines, for geometric dedup.

        Exact point-equality dedup is fragile (the painted road_lines and the
        lane boundaries are sampled differently), so we test boundaries for
        overlap against this buffered region instead. Returns a geometry or None.
        """
        if not polylines:
            return None
        try:
            from shapely.geometry import LineString
            from shapely.ops import unary_union
        except Exception:
            return None
        bufs = [LineString(p).buffer(self._COVER_BUFFER_M)
                for p in polylines if len(p) >= 2]
        return unary_union(bufs) if bufs else None

    @staticmethod
    def _overlaps_cover(seg, covered) -> bool:
        """True if most of ``seg`` already lies under ``covered`` (painted or a
        near-duplicate of an already-drawn line) — so it is skipped to avoid a
        second coplanar ribbon (z-fighting)."""
        if covered is None:
            return False
        try:
            from shapely.geometry import LineString
        except Exception:
            return False
        ls = LineString(seg)
        if ls.length < 1e-6:
            return True
        if isinstance(covered, _CoverIndex):
            return covered.covers(ls)
        return ls.intersection(covered).length > 0.6 * ls.length

    def _extend_cover(self, covered, seg):
        """Add ``seg``'s buffer to the covered region (for duplicate suppression)."""
        try:
            from shapely.geometry import LineString
        except Exception:
            return covered
        buf = LineString(seg).buffer(self._COVER_BUFFER_M)
        if isinstance(covered, _CoverIndex):
            covered.add(buf)
            return covered
        return buf if covered is None else covered.union(buf)

    def _build_lane_boundary_lines(self, lane_features, env_id: int,
                                   parent_path: str, *,
                                   painted_polylines=None) -> List[str]:
        """Draw lane boundary polylines as structured white lane lines.

        Pipeline: drop intersection-connector lanes → exact-dedup boundaries →
        clip each out of the junction zone → skip parts already painted or
        duplicated (geometric overlap test) → classify interior(dashed)/edge
        (solid) geometrically → emit. Yields clean approach lane lines with no
        paint across the junction, instead of an overlapping tangle. A single
        malformed boundary is skipped without aborting the rest.
        """
        junction = self._junction_region(lane_features)
        covered = _CoverIndex(self._painted_cover(painted_polylines))

        # Exact self-dedup of boundary geometry from non-connector lanes.
        geoms: Dict = {}
        for _fid, fd in lane_features:
            cl = fd.get(SD.POLYLINE)
            if cl is not None and self._polyline_length(cl) < self.junction_lane_m:
                continue  # intersection connector — skip its boundaries
            for key in (SD.LEFT_BOUNDARIES, SD.RIGHT_BOUNDARIES):
                arr = self._as_xy(fd.get(key))
                if arr is not None:
                    geoms.setdefault(self._boundary_signature(arr), arr)

        created: List[str] = []
        for i, arr in enumerate(geoms.values()):
            try:
                dashed = self._is_interior_divider(arr)
                for j, seg in enumerate(self._clip_outside_junction(arr, junction)):
                    if self._overlaps_cover(seg, covered):
                        continue  # already painted / duplicate edge
                    path = f"{parent_path}/markings/lane_line_{i}_{j}"
                    if self._build_ribbon_prim(path, seg, width=self.lane_line_width,
                                               dashed=dashed, color=(1.0, 1.0, 1.0),
                                               env_id=env_id):
                        created.append(path)
                        covered = self._extend_cover(covered, seg)
            except Exception as exc:  # one bad boundary must not drop the rest
                logger.warning(f"[Warning] lane boundary line {i} failed: {exc}")
                continue
        return created

    # Buffer (m) for the geometric line-dedup cover (skip boundaries that lie
    # under an already-painted or already-drawn line).
    _COVER_BUFFER_M = 0.4

    @staticmethod
    def _polyline_length(arr) -> float:
        """Total planar length (m) of a polyline."""
        a = np.asarray(arr, float)[:, :2]
        if len(a) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())

    # Buffer radius (m) applied to each lane centreline before union: ~half a
    # lane width plus an overlap margin so adjacent lanes merge with no seam and
    # crossing lanes fill the intersection. Slight spill onto the shoulder is an
    # acceptable trade for a continuous road.
    _ROAD_BUFFER_M = 2.2

    def _build_unified_road(self, lane_features, env_id: int,
                            parent_path: str) -> Optional[str]:
        """Build ONE asphalt surface = buffered union of all lane geometries.

        Closes the grass-through-the-seams gaps that per-lane ribbons leave.
        Falls back (via the caller) to per-lane ribbons if shapely is missing.
        """
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union, triangulate

        polys = []
        for fid, fd in lane_features:
            # Centreline ribbon (>= 2 * _ROAD_BUFFER_M wide). Built whenever a
            # polyline exists: on its own for polyline-only features, and
            # UNIONED with the true footprint when both are present so the
            # road surface never narrows below the pre-polygon ribbon (true
            # footprints of adjacent lanes need not overlap, so footprint-only
            # coverage would reopen seams/holes in the asphalt and collision).
            line_geom = None
            if SD.POLYLINE in fd and fd[SD.POLYLINE] is not None:
                arr = np.asarray(fd[SD.POLYLINE], dtype=float)[:, :2]
                if len(arr) >= 2:
                    line_geom = LineString(arr).buffer(
                        self._ROAD_BUFFER_M, cap_style=1, join_style=1)
            # True lane footprint (slightly buffered to close hairline seams).
            poly_geom = None
            has_polygon = SD.POLYGON in fd and fd[SD.POLYGON] is not None
            if has_polygon:
                arr = np.asarray(fd[SD.POLYGON], dtype=float)[:, :2]
                if len(arr) >= 3:
                    try:
                        cand = Polygon(arr).buffer(0.3)
                        if not cand.is_empty:
                            poly_geom = cand
                    except Exception as exc:
                        # Invalid (e.g. self-intersecting) footprint: fall back
                        # to the centreline ribbon instead of dropping the lane.
                        print(f"[Warning] lane {fid}: invalid polygon "
                              f"({exc}); "
                              + ("using centreline ribbon instead"
                                 if line_geom is not None
                                 else "no polyline either — lane dropped "
                                      "from road surface"))
            if poly_geom is not None:
                polys.append(poly_geom)
            if line_geom is not None:
                polys.append(line_geom)
            if poly_geom is None and line_geom is None:
                print(f"[Warning] lane {fid}: no usable polygon or polyline; "
                      f"skipped in unified road")
        if not polys:
            return None
        road = unary_union(polys)
        # Expose the drivable polygon so grass scatter can avoid the asphalt.
        self.road_polygon = road

        # Local origin = road centroid, so the mesh's object-space coords stay
        # small. OmniPBR projects the asphalt texture in object space; at world
        # magnitudes (~5000 m) the per-pixel UV derivative loses float precision
        # and the texture renders flat in the near field.
        ox, oy = float(road.centroid.x), float(road.centroid.y)
        verts: list[tuple[float, float, float]] = []
        counts: list[int] = []
        indices: list[int] = []
        for tri in triangulate(road):
            if not road.contains(tri.centroid):
                continue
            base = len(verts)
            for x, y in list(tri.exterior.coords)[:3]:
                verts.append((float(x) - ox, float(y) - oy, float(self.lane_height)))
            indices.extend([base, base + 1, base + 2])
            counts.append(3)
        if not counts:
            return None

        prim_path = f"{parent_path}/lanes/road_surface"
        mesh = define_flat_mesh(self.stage, prim_path, verts, counts, indices,
                                translate=(ox, oy, 0.0))
        try:
            from pxr import UsdPhysics
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        except Exception:
            pass
        # OmniPBR object-space projection needs no UVs (uses object-local XYZ).
        self._apply_material(prim_path, color=(0.08, 0.08, 0.08),
                             roughness=0.9, surface_type="road", env_id=env_id)
        return prim_path

    def _build_lane(self, feature_id: str, feature_data: Dict, env_id: int, parent_path: str) -> Optional[str]:
        """Build lane surface from polyline or polygon."""
        from pxr import UsdGeom  # noqa: lazy — only available post-IsaacSim init
        if SD.POLYGON in feature_data and feature_data[SD.POLYGON] is not None:
            points = np.array(feature_data[SD.POLYGON])
        elif SD.POLYLINE in feature_data and feature_data[SD.POLYLINE] is not None:
            # Extrude polyline to polygon (assume 3.5m width)
            polyline = np.array(feature_data[SD.POLYLINE])
            points = self._extrude_polyline_to_polygon(polyline, width=3.5)
        else:
            return None

        if len(points) < 3:
            return None

        # Create mesh from polygon
        prim_path = f"{parent_path}/lanes/lane_{feature_id}"
        vertices, faces = self._polygon_to_mesh(points, height=self.lane_height)

        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])
        self._set_planar_uvs(mesh, vertices, tiles_per_metre=0.25)

        # Add collision so lane surfaces participate in physics
        try:
            from pxr import UsdPhysics
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        except Exception:
            pass  # Non-fatal if physics extensions not loaded

        # Material: asphalt road surface — plain grey
        self._apply_material(prim_path, color=(0.4, 0.4, 0.4), roughness=0.9,
                             surface_type="road", env_id=env_id)

        # Collision (optional - can disable for performance)
        # UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

        return prim_path

    def _build_lane_marking(self, feature_id: str, feature_data: Dict, env_id: int, parent_path: str) -> Optional[str]:
        """Build an explicit painted lane marking from a road_line polyline."""
        if SD.POLYLINE not in feature_data or feature_data[SD.POLYLINE] is None:
            return None

        polyline = np.array(feature_data[SD.POLYLINE])
        if len(polyline) < 2:
            return None

        feature_type = feature_data[SD.TYPE]
        is_yellow = MetaDriveType.is_yellow_line(feature_type)
        # Honour the source tag: a line is only dashed when the dataset itself
        # tags it broken. A SOLID single yellow centre line stays solid — a
        # continuous line is far more visible from a driving camera than dashes,
        # and AV2 tags real centre dividers SOLID_SINGLE_YELLOW, so this also
        # keeps us faithful to the map instead of inventing gaps.
        is_broken = MetaDriveType.is_broken_line(feature_type)

        # Color + width. The yellow centre line is the primary divider, so draw
        # it wider than white edge/lane lines for prominence at distance.
        color = (1.0, 1.0, 0.0) if is_yellow else (1.0, 1.0, 1.0)
        width = self.centerline_width if is_yellow else self.marking_width

        prim_path = f"{parent_path}/markings/marking_{feature_id}"
        return self._build_ribbon_prim(prim_path, polyline, width=width,
                                       dashed=is_broken, color=color, env_id=env_id)

    def _build_crosswalk(self, feature_id: str, feature_data: Dict, env_id: int, parent_path: str) -> Optional[str]:
        """Build crosswalk from polygon."""
        from pxr import UsdGeom  # noqa: lazy
        if SD.POLYGON not in feature_data or feature_data[SD.POLYGON] is None:
            return None

        points = np.array(feature_data[SD.POLYGON])
        if len(points) < 3:
            return None

        prim_path = f"{parent_path}/crosswalks/crosswalk_{feature_id}"
        vertices, faces = self._polygon_to_mesh(points, height=self.crosswalk_height)

        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])
        self._set_planar_uvs(mesh, vertices, tiles_per_metre=0.5)

        # Material: white with stripes (simplified as white for now)
        self._apply_material(prim_path, color=(1.0, 1.0, 1.0), roughness=0.6,
                             surface_type="crosswalk", env_id=env_id)

        return prim_path

    def _build_sidewalk(self, feature_id: str, feature_data: Dict, env_id: int, parent_path: str) -> Optional[str]:
        """Build sidewalk from polygon."""
        if SD.POLYGON not in feature_data or feature_data[SD.POLYGON] is None:
            return None

        points = np.array(feature_data[SD.POLYGON], dtype=float)
        if len(points) < 3:
            return None

        # Centre on the polygon's centroid + xform translate so the textured
        # (OmniPBR object-space) sidewalk keeps precision — world-magnitude
        # coords render the texture flat (see usd_mesh.define_flat_mesh).
        ox = float(points[:, 0].mean())
        oy = float(points[:, 1].mean())
        local = points.copy()
        local[:, 0] -= ox
        local[:, 1] -= oy
        vertices, faces = self._polygon_to_mesh(local, height=0.15)  # 15cm

        prim_path = f"{parent_path}/sidewalks/sidewalk_{feature_id}"
        mesh = define_flat_mesh(
            self.stage, prim_path, vertices,
            [len(f) for f in faces], [idx for face in faces for idx in face],
            translate=(ox, oy, 0.0))
        self._set_planar_uvs(mesh, vertices, tiles_per_metre=0.5)

        # Material: light gray concrete
        self._apply_material(prim_path, color=(0.7, 0.7, 0.7), roughness=0.7,
                             surface_type="sidewalk", env_id=env_id)

        return prim_path

    def _build_boundary(self, feature_id: str, feature_data: Dict, env_id: int, parent_path: str) -> Optional[str]:
        """Build road boundary from polyline."""
        from pxr import UsdGeom  # noqa: lazy
        if SD.POLYLINE not in feature_data or feature_data[SD.POLYLINE] is None:
            return None

        polyline = np.array(feature_data[SD.POLYLINE])
        if len(polyline) < 2:
            return None

        prim_path = f"{parent_path}/boundaries/boundary_{feature_id}"
        vertices, faces = self._polyline_to_ribbon(polyline, width=0.2, height=0.02)

        mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([idx for face in faces for idx in face])

        # Material: gray. Use surface_type "boundary" (not "road") so it maps to
        # no library kind and gets the flat grey fallback rather than the dark
        # asphalt material.
        self._apply_material(prim_path, color=(0.5, 0.5, 0.5), roughness=0.6,
                             surface_type="boundary", env_id=env_id)

        return prim_path

    # ========== Geometry Helper Functions ==========

    def _extrude_polyline_to_polygon(self, polyline: np.ndarray, width: float) -> np.ndarray:
        """
        Extrude a polyline to a polygon with given width.

        Args:
            polyline: [N, 2] or [N, 3] array of points
            width: Width of the polygon

        Returns:
            [2*N, 2] array representing polygon vertices
        """
        polyline_2d = polyline[:, :2]  # Use only x, y

        # Compute perpendicular vectors
        left_points = []
        right_points = []

        for i in range(len(polyline_2d)):
            if i == 0:
                tangent = polyline_2d[i + 1] - polyline_2d[i]
            elif i == len(polyline_2d) - 1:
                tangent = polyline_2d[i] - polyline_2d[i - 1]
            else:
                tangent = polyline_2d[i + 1] - polyline_2d[i - 1]

            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
            perpendicular = np.array([-tangent[1], tangent[0]])

            left_points.append(polyline_2d[i] + perpendicular * width / 2)
            right_points.append(polyline_2d[i] - perpendicular * width / 2)

        # Combine left and right (left forward, right backward for proper polygon ordering)
        polygon = np.vstack([left_points, right_points[::-1]])

        return polygon

    def _polygon_to_mesh(self, points: np.ndarray, height: float = 0.0) -> Tuple[List, List]:
        """
        Convert 2D polygon to 3D mesh (extruded upward).

        Args:
            points: [N, 2] or [N, 3] array of polygon vertices
            height: Height to extrude to

        Returns:
            (vertices, faces) where vertices is list of Gf.Vec3f and faces is list of triangles
        """
        from pxr import Gf  # noqa: lazy
        points_2d = points[:, :2]

        # Triangulate polygon using simple fan triangulation
        # (For complex polygons, should use proper triangulation library)
        vertices = [Gf.Vec3f(float(p[0]), float(p[1]), float(height)) for p in points_2d]

        # Simple fan triangulation from first vertex
        faces = []
        for i in range(1, len(vertices) - 1):
            faces.append([0, i, i + 1])

        return vertices, faces

    def _set_planar_uvs(self, mesh, vertices, tiles_per_metre: float = 0.25):
        """Set a per-vertex ``st`` primvar from world XY so textures map/tile.

        Procedurally-built meshes ship with NO UVs, so a UsdUVTexture has
        nothing to sample and the surface renders as flat colour. We project
        world (x, y) to UV with a real-world tiling rate (``tiles_per_metre``)
        so the texture repeats consistently regardless of mesh size — e.g.
        0.25 → one texture tile every 4 m.
        """
        try:
            from pxr import UsdGeom, Sdf, Vt, Gf  # noqa: lazy
            uvs = [
                Gf.Vec2f(float(v[0]) * tiles_per_metre, float(v[1]) * tiles_per_metre)
                for v in vertices
            ]
            primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
            st = primvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex,
            )
            st.Set(Vt.Vec2fArray(uvs))
        except Exception as exc:  # pragma: no cover - depends on USD
            logger.warning(f"[ScenarioMapBuilder] failed to set UVs: {exc}")

    def _polyline_to_ribbon(self, polyline: np.ndarray, width: float, height: float) -> Tuple[List, List]:
        """
        Convert polyline to ribbon mesh.

        Args:
            polyline: [N, 2] or [N, 3] array of points
            width: Width of ribbon
            height: Height above ground

        Returns:
            (vertices, faces)
        """
        from pxr import Gf  # noqa: lazy
        polygon = self._extrude_polyline_to_polygon(polyline, width)

        # Create top surface
        vertices_top = [Gf.Vec3f(float(p[0]), float(p[1]), float(height)) for p in polygon]

        # Create triangles for the ribbon
        n = len(polyline)
        faces = []

        # Top surface triangulation
        # Left strip
        for i in range(n - 1):
            faces.append([i, i + 1, 2 * n - i - 1])
            faces.append([i + 1, 2 * n - i - 2, 2 * n - i - 1])

        return vertices_top, faces

    def _make_broken_line(self, polyline: np.ndarray, dash_length: float = 3.0, gap_length: float = 3.0) -> np.ndarray:
        """
        Convert solid polyline to broken line by sampling dashes.

        Args:
            polyline: [N, 2] array
            dash_length: Length of each dash
            gap_length: Length of gap between dashes

        Returns:
            Sampled polyline representing dashes
        """
        # Compute cumulative length along polyline
        diffs = np.diff(polyline, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.cumsum(np.concatenate([[0], segment_lengths]))
        total_length = cumulative_length[-1]

        # Sample points at dash intervals
        dash_points = []
        current_dist = 0.0
        pattern_length = dash_length + gap_length

        while current_dist < total_length:
            # Start of dash
            start_dist = current_dist
            end_dist = min(current_dist + dash_length, total_length)

            # Interpolate start and end points
            start_pt = self._interpolate_along_polyline(polyline, cumulative_length, start_dist)
            end_pt = self._interpolate_along_polyline(polyline, cumulative_length, end_dist)

            dash_points.extend([start_pt, end_pt])

            current_dist += pattern_length

        return np.array(dash_points) if dash_points else polyline

    def _interpolate_along_polyline(self, polyline: np.ndarray, cumulative_length: np.ndarray, distance: float) -> np.ndarray:
        """Interpolate point at given distance along polyline."""
        idx = np.searchsorted(cumulative_length, distance)
        if idx == 0:
            return polyline[0]
        if idx >= len(polyline):
            return polyline[-1]

        # Linear interpolation between polyline[idx-1] and polyline[idx]
        segment_start_dist = cumulative_length[idx - 1]
        segment_end_dist = cumulative_length[idx]
        segment_length = segment_end_dist - segment_start_dist

        if segment_length < 1e-6:
            return polyline[idx]

        t = (distance - segment_start_dist) / segment_length
        return polyline[idx - 1] + t * (polyline[idx] - polyline[idx - 1])


    def _apply_material(
        self,
        prim_path: str,
        color: Tuple[float, float, float],
        roughness: float = 0.5,
        surface_type: str = "road",
        env_id: int = 0,
    ):
        """Apply material to mesh.

        Uses a flat UsdPreviewSurface.
        """
        from pxr import UsdGeom, Gf, UsdShade, Sdf  # noqa: lazy

        # Set displayColor as a baseline (visible in wireframe / fallback)
        mesh = UsdGeom.Mesh.Get(self.stage, prim_path)
        if mesh:
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        # Fallback: plain UsdPreviewSurface
        cr, cg, cb = (round(c, 2) for c in color)
        mat_key = f"mat_{cr}_{cg}_{cb}_{round(roughness, 1)}".replace(".", "d")
        mat_prim_path = f"/World/Looks/{mat_key}"

        if not self.stage.GetPrimAtPath(mat_prim_path).IsValid():
            mat = UsdShade.Material.Define(self.stage, mat_prim_path)
            sh = UsdShade.Shader.Define(self.stage, f"{mat_prim_path}/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(cr, cg, cb))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

        # Bind material
        mesh_prim = self.stage.GetPrimAtPath(prim_path)
        if mesh_prim.IsValid():
            try:
                UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
                    UsdShade.Material.Get(self.stage, mat_prim_path))
            except Exception:
                pass  # Non-fatal: displayColor already set



    def cleanup(self):
        """Remove all created map prims."""
        for prim_path in self.created_prims:
            self.stage.RemovePrim(prim_path)
        self.created_prims.clear()
