"""Geospatial visual priors — PriorEye's memory retrieval over the scenario map.

PriorEye (Yeon et al., ECCV 2026, `ori-mrg/PriorEye`) augments an end-to-end
planner with *street-view* imagery embeddings retrieved along the route the
ego is about to drive. The priors ship as one pickle,
``siglip2_embedding_all.pkl``, keyed

    map_name -> lane_token -> node_index -> {x, y, heading, embedding, ...}

where ``x``/``y`` are absolute nuPlan UTM metres and ``embedding`` is the
SigLIP2 encoding of the Google Street View panorama nearest that lane node.
Retrieval walks the lane graph ahead of the ego, picks the branch matching the
driving command, and bins the nodes it passes into fixed 5 m distance bins —
20 bins, so 100 m of context.

**Why this is a re-implementation rather than a port.** Upstream reaches the
lane graph through the nuPlan devkit (``get_maps_api`` +
``get_proximal_map_objects`` over ``SemanticMapLayer.LANE`` /
``LANE_CONNECTOR``), which means the gpkg maps and the devkit on the eval
path. NexusSim's eval path carries neither, and does not need to: the py123d
ScenarioDescription a NavSafe bundle projects already holds that lane graph —
centreline polylines, lane polygons, and ``entry_lanes``/``exit_lanes``
successor links — keyed by the *same* nuPlan object ids the prior pickle uses.
Measured on bundle ``061e7e1700945b03`` (us-ma-boston): the bundle's arrow map
carries 20 020 objects and 3 373 lanes, and **all 2 484** boston lane tokens in
the prior pickle are present among them, at the same ids. So the retrieval
below reads ``scenario_data["map_features"]`` and reproduces the upstream
algorithm step for step.

Three deliberate deviations, each because the devkit primitive has no exact
map-features equivalent:

* **Proximity query.** ``get_proximal_map_objects(point, 5.0, ...)`` selects by
  map-object *geometry*. Here a lane is a candidate when any of its polygon or
  centreline vertices lies within 5 m of the ego. Ranking then uses the
  centreline only, exactly as upstream does.
* **Lane heading.** Upstream reads ``discrete_path[i].heading``. py123d
  polylines carry positions only, so the heading at a vertex is the finite
  difference to the next one (the last vertex reuses the previous segment).
  Both describe the same discretised centreline, to within its own spacing.
* **Path enumeration cap.** Upstream recurses over successors with no bound on
  the number of paths. A pathological junction could therefore stall a replan,
  so :data:`MAX_PATHS` caps enumeration; hitting it is reported through
  :attr:`GeospatialPriorRetriever.last_diagnostics` rather than swallowed.

Coordinates: a NavSafe bundle's scenario is ``coordinate == "local_frame0"``,
i.e. every xy has had ``metadata["scenario_origin_xy"]`` subtracted
(``utm = local + origin``). Rather than shift thousands of map polylines per
replan, the retriever shifts the handful of *prior nodes* it touches into the
scenario frame. Headings are unaffected by the recentring, so the ego heading
needs no correction.
"""

from __future__ import annotations

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from navsafe.modelzoo.navsim.prioreye.memory import MEMORY_EMBEDDING_DIMS

#: Metres per memory bin, and the number of bins. Upstream
#: ``memory_config.DISTANCE_BIN_SIZE`` / ``MEMORY_NODE_NUM``: 20 bins x 5 m =
#: 100 m of route context, which is also the recursion's distance budget.
DISTANCE_BIN_SIZE = 5.0
MEMORY_NODE_NUM = 20

#: Radius of the "which lane am I on" proximity query, and the heading
#: agreement a candidate lane must satisfy (upstream ``search_radius`` /
#: ``MAX_ANGLE_THRESHOLD``).
SEARCH_RADIUS = 5.0
MAX_HEADING_DIFF = math.pi / 3.0

#: Successor-path enumeration bound; see the module note.
MAX_PATHS = 512

#: Embedding widths per encoder. Re-exported from the model module, which is
#: where the width is checkpoint contract (it sets ``memory_in_dim``); this
#: alias keeps the retrieval code reading one name.
EMBEDDING_DIMS: Dict[str, int] = MEMORY_EMBEDDING_DIMS

#: The four nuPlan maps the prior pickle covers (upstream
#: ``MAP_NAMES_NUPLAN``).
MAP_NAMES_NUPLAN = (
    "sg-one-north",
    "us-ma-boston",
    "us-nv-las-vegas-strip",
    "us-pa-pittsburgh-hazelwood",
)

#: py123d lane types that stand in for nuPlan's ``LANE`` + ``LANE_CONNECTOR``.
#: Bike lanes are included because the prior pickle carries them — 10 of
#: boston's 2 484 tokens are ``LANE_BIKE_LANE`` — so excluding them would drop
#: priors the checkpoint was trained with. ``CROSSWALK`` shares the unprefixed
#: key space and is not a lane, hence the prefix test rather than "not a line".
LANE_TYPE_PREFIX = "LANE"

#: Feature-id namespace for lanes NexusSim synthesised, which must be excluded.
#: ``_add_logged_drivable_support`` (py123d_scenario_description.py) adds
#: ``__logged_drivable_support_<i>`` polygons, typed as lanes, wherever the
#: source map has a hole the logged ego drives through. They are repairs, not
#: nuPlan lanes: they carry no nuPlan object id (so the prior pickle can never
#: key on them) and no ``exit_lanes`` (so a path starting there cannot be
#: walked). Measured on ``0ebb578555b25ab2`` (us-pa-pittsburgh-hazelwood):
#: without this exclusion, ``closest_lane`` returned
#: ``__logged_drivable_support_0`` on every replan, the path was that one
#: feature, and retrieval produced **0 of 20 bins on all 41 replans** — while
#: all three of the scenario's route lanes DO carry priors. The failure is
#: silent by construction: an empty memory is a legal state, so the episode
#: runs on persistent memory alone and scores.
SYNTHETIC_ID_PREFIX = "__"


def _normalize_angle_rad(angle: float) -> float:
    """Wrap to ``(-pi, pi]`` — upstream ``memory_util._normalize_angle_rad``."""
    angle = float(angle)
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def command_string(one_hot: "Sequence[float] | np.ndarray") -> str:
    """NAVSIM driving-command one-hot -> upstream's command string.

    Upstream takes ``argmax(driving_command)`` over the 4-wide NAVSIM one-hot
    and maps ``{0: turn_left, 1: straight, 2: turn_right, 3: straight}`` — the
    "unknown" slot drives straight. Passing the one-hot (rather than an int)
    keeps this adapter reading the same vector the model's ``ego_status``
    carries, so the two can never disagree about the command.
    """
    idx = int(np.argmax(np.asarray(one_hot, dtype=np.float64)))
    return {0: "turn_left", 1: "straight", 2: "turn_right", 3: "straight"}.get(idx, "straight")


@lru_cache(maxsize=4)
def _load_prior_pickle(path: str) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """Load and cache the prior pickle (248 MB for siglip2; ~4 s cold).

    Cached because every replan retrieves from it and an eval sweep runs
    hundreds of replans per scenario. Keyed by path so a corruption-test
    variant can be loaded alongside the real one.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Geospatial prior pickle not found: {p}. Download it with "
            "`gdown 1hCZtWxHwqWbjrHhpq7b2Eq_Mi0PiY3Q7` and point "
            "NAVSAFE_PRIOREYE_EMBEDDING at the file."
        )
    with p.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Geospatial prior pickle {p} is not a non-empty dict")
    return data


class GeospatialPriorIndex:
    """The prior pickle, plus the per-map extent needed to identify a scenario.

    A NavSafe bundle's ScenarioDescription metadata does not name its nuPlan
    map (checked: ``metadata`` carries ``dataset`` / ``split`` / ``source`` /
    ``scenario_id`` / ``coordinate`` / ``scenario_origin_xy`` /
    ``route_lane_ids`` and no location). The map is recovered from the ego's
    UTM position instead: the four nuPlan maps sit in four different UTM zones
    and cities, so their node clouds do not overlap — boston is around
    (331 k, 4 691 k) and las-vegas around (664 k, 3 998 k). Identification is
    therefore exact and needs no extra input, and :meth:`map_for_utm` fails
    loudly rather than guessing when a position matches none or several.
    """

    def __init__(self, pickle_path: str, embedding_model: str = "siglip2") -> None:
        self.pickle_path = str(pickle_path)
        self.embedding_model = embedding_model.lower()
        if self.embedding_model not in EMBEDDING_DIMS:
            raise ValueError(
                f"Unknown prior encoder {embedding_model!r}; "
                f"expected one of {sorted(EMBEDDING_DIMS)}"
            )
        self.embedding_dim = EMBEDDING_DIMS[self.embedding_model]
        self._data = _load_prior_pickle(self.pickle_path)
        self._extents = {name: self._extent(nodes) for name, nodes in self._data.items()}

    @staticmethod
    def _extent(lanes: Dict[str, Dict[int, Dict[str, Any]]]) -> Tuple[float, float, float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for nodes in lanes.values():
            for node in nodes.values():
                xs.append(float(node["x"]))
                ys.append(float(node["y"]))
        if not xs:
            return (math.inf, math.inf, -math.inf, -math.inf)
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def map_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._data))

    def lanes(self, map_name: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """Prior nodes for ``map_name``, ``{}`` when the map is not covered."""
        return self._data.get(map_name, {})

    def map_for_utm(self, x: float, y: float, margin_m: float = 2000.0) -> str:
        """Which nuPlan map contains UTM ``(x, y)``.

        Raises:
            LookupError: when no map's node extent (grown by ``margin_m``)
                contains the point, or when more than one does. Both mean the
                retrieval would read the wrong city's priors, which is a silent
                accuracy loss rather than a crash — so it is made a crash.
        """
        hits = [
            name
            for name, (x0, y0, x1, y1) in self._extents.items()
            if x0 - margin_m <= x <= x1 + margin_m and y0 - margin_m <= y <= y1 + margin_m
        ]
        if len(hits) == 1:
            return hits[0]
        where = ", ".join(
            f"{n}: x[{e[0]:.0f}, {e[2]:.0f}] y[{e[1]:.0f}, {e[3]:.0f}]"
            for n, e in sorted(self._extents.items())
        )
        raise LookupError(
            f"Ego UTM ({x:.1f}, {y:.1f}) matches {len(hits)} prior maps "
            f"({hits or 'none'}); cannot identify the nuPlan map. Extents: {where}. "
            "Set NAVSAFE_PRIOREYE_MAP to name the map explicitly."
        )


class LaneGraph:
    """Lane centrelines + successor links, read from ``map_features``.

    Built once per scenario. The vertex table backing :meth:`closest_lane` is
    ~135 k points for a boston bundle, so the query is a single vectorised
    distance evaluation per replan (sub-millisecond) rather than a per-lane
    loop.
    """

    def __init__(
        self,
        centrelines: Dict[str, np.ndarray],
        exits: Dict[str, Tuple[str, ...]],
        points: np.ndarray,
        point_lane: np.ndarray,
        lane_ids: Tuple[str, ...],
    ) -> None:
        self._centrelines = centrelines
        self._exits = exits
        self._points = points
        self._point_lane = point_lane
        self._lane_ids = lane_ids

    @classmethod
    def from_map_features(cls, map_features: Dict[str, Any]) -> "LaneGraph":
        centrelines: Dict[str, np.ndarray] = {}
        exits: Dict[str, Tuple[str, ...]] = {}
        lane_ids: List[str] = []
        chunks: List[np.ndarray] = []
        index: List[np.ndarray] = []

        for lane_id, feature in (map_features or {}).items():
            if not isinstance(feature, dict):
                continue
            if str(lane_id).startswith(SYNTHETIC_ID_PREFIX):
                continue  # NexusSim map repair, not a nuPlan lane — see above
            if not str(feature.get("type", "")).startswith(LANE_TYPE_PREFIX):
                continue
            polyline = feature.get("polyline")
            if polyline is None:
                continue
            centre = np.asarray(polyline, dtype=np.float64)
            if centre.ndim != 2 or centre.shape[0] < 2:
                continue
            centre = centre[:, :2]

            key = str(lane_id)
            slot = len(lane_ids)
            lane_ids.append(key)
            centrelines[key] = centre
            exits[key] = tuple(str(x) for x in (feature.get("exit_lanes") or ()))

            # Candidacy geometry: centreline + polygon, standing in for
            # nuPlan's geometry-based proximal query (module note).
            verts = [centre]
            polygon = feature.get("polygon")
            if polygon is not None:
                poly = np.asarray(polygon, dtype=np.float64)
                if poly.ndim == 2 and poly.shape[0] >= 3:
                    verts.append(poly[:, :2])
            stacked = np.concatenate(verts, axis=0)
            chunks.append(stacked)
            index.append(np.full(stacked.shape[0], slot, dtype=np.int32))

        if chunks:
            points = np.concatenate(chunks, axis=0)
            point_lane = np.concatenate(index, axis=0)
        else:
            points = np.zeros((0, 2), dtype=np.float64)
            point_lane = np.zeros((0,), dtype=np.int32)
        return cls(centrelines, exits, points, point_lane, tuple(lane_ids))

    def __len__(self) -> int:
        return len(self._lane_ids)

    def centreline(self, lane_id: str) -> Optional[np.ndarray]:
        return self._centrelines.get(lane_id)

    def exits(self, lane_id: str) -> Tuple[str, ...]:
        return self._exits.get(lane_id, ())

    @staticmethod
    def _headings(centre: np.ndarray) -> np.ndarray:
        """Per-vertex heading from forward differences (module note)."""
        deltas = np.diff(centre, axis=0)
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        return np.concatenate([headings, headings[-1:]])

    def closest_lane(self, x: float, y: float, yaw: float) -> Optional[str]:
        """The nearest lane whose heading agrees with ``yaw``, or ``None``.

        Mirrors upstream ``find_closest_lane_by_pose``: candidates within
        :data:`SEARCH_RADIUS`, drop any whose centreline heading at the closest
        vertex differs from the ego's by more than :data:`MAX_HEADING_DIFF`,
        then take the closest of what remains.
        """
        if self._points.shape[0] == 0:
            return None
        query = np.array([float(x), float(y)], dtype=np.float64)
        near = np.linalg.norm(self._points - query, axis=1) <= SEARCH_RADIUS
        if not bool(near.any()):
            return None

        yaw = _normalize_angle_rad(yaw)
        best_id: Optional[str] = None
        best_distance = math.inf
        for slot in np.unique(self._point_lane[near]):
            lane_id = self._lane_ids[int(slot)]
            centre = self._centrelines[lane_id]
            distances = np.linalg.norm(centre - query, axis=1)
            closest = int(np.argmin(distances))
            if float(distances[closest]) >= best_distance:
                continue
            lane_yaw = _normalize_angle_rad(float(self._headings(centre)[closest]))
            if abs(_normalize_angle_rad(yaw - lane_yaw)) > MAX_HEADING_DIFF:
                continue
            best_distance = float(distances[closest])
            best_id = lane_id
        return best_id

    def _lengths(
        self, lane_id: str, x: Optional[float] = None, y: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """``(total, remaining)`` centreline length — upstream ``_get_lane_helpers``."""
        centre = self._centrelines.get(lane_id)
        if centre is None:
            return None, None
        segments = np.linalg.norm(np.diff(centre, axis=0), axis=1)
        total = float(segments.sum())
        if x is None or y is None:
            return total, 0.0
        distances = np.linalg.norm(centre - np.array([x, y], dtype=np.float64), axis=1)
        closest = int(np.argmin(distances))
        if closest == centre.shape[0] - 1:
            return total, 0.0
        return total, float(segments[closest:].sum())

    def paths_until_distance(
        self, x: float, y: float, yaw: float, max_distance: float
    ) -> Tuple[List[List[str]], bool]:
        """Successor paths from the ego's lane covering ``max_distance``.

        Returns ``(paths, truncated)``; ``truncated`` is True when
        :data:`MAX_PATHS` bounded the enumeration.
        """
        start = self.closest_lane(x, y, yaw)
        if start is None:
            return [], False
        _, remaining = self._lengths(start, x, y)
        if remaining is None:
            return [], False
        if remaining >= max_distance:
            return [[start]], False

        paths: List[List[str]] = []
        truncated = self._recurse(start, [start], remaining, max_distance, paths)
        return paths, truncated

    def _recurse(
        self,
        lane_id: str,
        path: List[str],
        distance: float,
        max_distance: float,
        paths: List[List[str]],
    ) -> bool:
        """Upstream ``_find_paths_recursive``, plus the :data:`MAX_PATHS` bound."""
        if len(paths) >= MAX_PATHS:
            return True
        outgoing = self.exits(lane_id)
        if not outgoing:
            paths.append(list(path))
            return False

        truncated = False
        for next_id in outgoing:
            if len(paths) >= MAX_PATHS:
                return True
            if next_id in path:  # cycle: close the path here, as upstream does
                paths.append(list(path) + [next_id])
                continue
            length, _ = self._lengths(next_id)
            if length is None:
                continue
            extended = list(path) + [next_id]
            if distance + length >= max_distance:
                paths.append(extended)
            else:
                truncated |= self._recurse(
                    next_id, extended, distance + length, max_distance, paths
                )
        return truncated

    def path_for_command(
        self, command: str, x: float, y: float, yaw: float, max_distance: float
    ) -> Tuple[Optional[List[str]], bool]:
        """Pick the successor path matching ``command`` — upstream ``find_lane_via_command``.

        The turn is scored by the signed angle between the first lane's chord
        and the last lane's chord: ``turn_left`` takes the most positive,
        ``turn_right`` the most negative, ``straight`` the smallest magnitude.
        """
        candidates, truncated = self.paths_until_distance(x, y, yaw, max_distance)
        if not candidates:
            return None, truncated

        angles: Dict[Tuple[str, ...], float] = {}
        for path in candidates:
            first = self._centrelines.get(path[0])
            last = self._centrelines.get(path[-1])
            if first is None or last is None:
                continue
            vec_a = first[-1] - first[0]
            vec_b = last[-1] - last[0]
            vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-6)
            vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-6)
            dot = float(np.dot(vec_a, vec_b))
            cross = float(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])
            angles[tuple(path)] = math.degrees(math.atan2(cross, dot))

        if not angles:
            return None, truncated
        if command == "turn_left":
            best = max(angles, key=lambda k: angles[k])
        elif command == "turn_right":
            best = min(angles, key=lambda k: angles[k])
        else:
            best = min(angles, key=lambda k: abs(angles[k]))
        return list(best), truncated


class GeospatialPriorRetriever:
    """PriorEye's ``MemoryFeatureBuilder``, over a py123d scenario.

    One instance per scenario (the lane graph is scenario-specific); the prior
    pickle behind ``index`` is process-cached and shared.
    """

    def __init__(
        self,
        index: GeospatialPriorIndex,
        lane_graph: LaneGraph,
        origin_xy: Tuple[float, float],
        map_name: str,
    ) -> None:
        self.index = index
        self.lane_graph = lane_graph
        self.origin_xy = (float(origin_xy[0]), float(origin_xy[1]))
        self.map_name = map_name
        self.max_distance = MEMORY_NODE_NUM * DISTANCE_BIN_SIZE
        #: Per-replan retrieval facts, for the adapter to log/assert on.
        self.last_diagnostics: Dict[str, Any] = {}

    def retrieve(
        self, x: float, y: float, heading: float, command: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``(embedding (20, D), position (20, 2))`` for a scenario-frame pose.

        ``x``/``y``/``heading`` are the ego pose in the *scenario* frame; the
        prior nodes are shifted into that frame by ``origin_xy``. Bins with no
        prior stay all-zero, which is exactly what
        :class:`~navsafe.modelzoo.navsim.prioreye.memory.MemoryAugmentationModule`
        masks out (``memory_embedding.abs().sum(-1) == 0``), so an unmapped
        stretch of road degrades to persistent memory only.
        """
        dim = self.index.embedding_dim
        embedding = np.zeros((MEMORY_NODE_NUM, dim), dtype=np.float64)
        position = np.zeros((MEMORY_NODE_NUM, 2), dtype=np.float32)

        lane_ids, truncated = self.lane_graph.path_for_command(
            command, x, y, heading, self.max_distance
        )
        self.last_diagnostics = {
            "map_name": self.map_name,
            "command": command,
            "path_len": 0 if lane_ids is None else len(lane_ids),
            "paths_truncated": truncated,
            "bins_filled": 0,
        }
        if lane_ids is None:
            return embedding.astype(np.float32), position

        priors = self.index.lanes(self.map_name)
        if not priors:
            return embedding.astype(np.float32), position

        heading_x, heading_y = math.cos(heading), math.sin(heading)
        cumulative = 0.0
        last_x, last_y = float(x), float(y)
        filled = [False] * MEMORY_NODE_NUM
        nodes_used: List[Optional[Dict[str, Any]]] = [None] * MEMORY_NODE_NUM
        count = 0

        for lane_id in lane_ids:
            if count >= MEMORY_NODE_NUM:
                break
            lane_nodes = priors.get(lane_id)
            if not lane_nodes:
                continue
            for node_index in sorted(lane_nodes):
                if count >= MEMORY_NODE_NUM:
                    break
                node = lane_nodes[node_index]
                node_x = float(node["x"]) - self.origin_xy[0]
                node_y = float(node["y"]) - self.origin_xy[1]
                if heading_x * (node_x - x) + heading_y * (node_y - y) < 0.0:
                    continue  # behind the ego
                cumulative += math.hypot(node_x - last_x, node_y - last_y)
                last_x, last_y = node_x, node_y
                if cumulative > self.max_distance:
                    break
                bin_index = int(cumulative // DISTANCE_BIN_SIZE)
                if not 0 <= bin_index < MEMORY_NODE_NUM or filled[bin_index]:
                    continue
                embedding[bin_index] = np.asarray(node["embedding"]).squeeze()
                nodes_used[bin_index] = node
                filled[bin_index] = True
                count += 1

        cos_h, sin_h = math.cos(-heading), math.sin(-heading)
        for i, binned in enumerate(nodes_used):
            if binned is None:
                continue
            dx = (float(binned["x"]) - self.origin_xy[0]) - x
            dy = (float(binned["y"]) - self.origin_xy[1]) - y
            position[i, 0] = dx * cos_h - dy * sin_h
            position[i, 1] = dx * sin_h + dy * cos_h

        self.last_diagnostics["bins_filled"] = count
        return embedding.astype(np.float32), position


def scenario_origin_xy(scenario_data: Dict[str, Any]) -> Tuple[float, float]:
    """The scenario's UTM origin, so ``utm = local + origin``.

    ``coordinate == "world"`` scenarios are already absolute and get a zero
    origin. A ``local_frame0`` scenario without a recorded
    ``scenario_origin_xy`` cannot be placed on the map at all — retrieval would
    silently read priors from a point kilometres away — so that raises.
    """
    metadata = (scenario_data or {}).get("metadata", {}) or {}
    coordinate = str(metadata.get("coordinate", "")).lower()
    origin = metadata.get("scenario_origin_xy")
    if coordinate == "local_frame0":
        if origin is None or len(origin) < 2:
            raise ValueError(
                "Scenario declares coordinate='local_frame0' but carries no "
                "scenario_origin_xy; the ego's UTM position is unknown and "
                "geospatial priors cannot be retrieved."
            )
        return (float(origin[0]), float(origin[1]))
    if coordinate in ("world", ""):
        return (0.0, 0.0)
    raise ValueError(
        f"Unsupported scenario coordinate frame {coordinate!r} for geospatial "
        "prior retrieval; expected 'local_frame0' or 'world'."
    )


def build_retriever(
    scenario_data: Dict[str, Any],
    ego_xy: Iterable[float],
    *,
    pickle_path: str,
    embedding_model: str = "siglip2",
    map_name: Optional[str] = None,
) -> GeospatialPriorRetriever:
    """Assemble a retriever for one scenario.

    ``ego_xy`` is any scenario-frame ego position (frame 0 is fine); it is used
    only to identify the nuPlan map, which ``map_name`` overrides.
    """
    index = GeospatialPriorIndex(pickle_path, embedding_model=embedding_model)
    origin = scenario_origin_xy(scenario_data)
    xy = np.asarray(list(ego_xy), dtype=np.float64)[:2]
    resolved = map_name or index.map_for_utm(float(xy[0] + origin[0]), float(xy[1] + origin[1]))
    graph = LaneGraph.from_map_features((scenario_data or {}).get("map_features", {}) or {})
    if len(graph) == 0:
        raise ValueError(
            "Scenario carries no lane features; geospatial prior retrieval "
            "needs the lane graph (polyline + exit_lanes) to walk the route."
        )
    return GeospatialPriorRetriever(index, graph, origin, resolved)


__all__ = [
    "DISTANCE_BIN_SIZE",
    "EMBEDDING_DIMS",
    "GeospatialPriorIndex",
    "GeospatialPriorRetriever",
    "LaneGraph",
    "MAP_NAMES_NUPLAN",
    "MAX_PATHS",
    "MEMORY_NODE_NUM",
    "SYNTHETIC_ID_PREFIX",
    "build_retriever",
    "command_string",
    "scenario_origin_xy",
]
