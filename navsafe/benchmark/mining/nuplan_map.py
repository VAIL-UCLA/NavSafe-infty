# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""nuPlan log + map access for the leaf miners.

Selection predicates read the ego's EXPERT trajectory and the map's road
graph, both straight from nuPlan — no conversion, no reconstruction. That is
what makes mining cheap enough to sweep a whole split before deciding which
clips are worth the GPU hours.

Three quirks of the map API are handled here once, because each of them
produced a wrong result before it was understood:

* ``get_one_map_object`` **raises** when several objects contain the point,
  which is normal for connectors. Caught once, it silently skipped 355 of 407
  scenarios and the sweep reported "no merges anywhere".
* ``ROADBLOCK`` is not point-queryable on every map version ("Object
  representation for layer ROADBLOCK is unavailable"), while ``LANE`` is
  everywhere — and a lane carries ``.parent``, its roadblock.
* nuPlan never joins two roadblocks directly: every transition runs
  roadblock -> ROADBLOCK_CONNECTOR -> roadblock. A predicate that asks for
  plain-roadblock predecessors therefore can never fire; the feeding ROADS are
  one hop further back.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from navsafe.benchmark import config as cfg
from navsafe.errors import NexusSimError

DATA_ROOT = str(cfg.NUPLAN_ROOT)
MAPS_ROOT = str(cfg.NUPLAN_MAPS_DEVKIT)
MAP_VERSION = "nuplan-maps-v1.0"
SPLITS = ("test", "val", "train_boston", "train_pittsburgh", "train_singapore",
          "train_vegas", "mini")


class MapAccessError(NexusSimError, RuntimeError):
    """The log or its map could not be opened."""


def find_log_db(log: str, *, data_root: str = DATA_ROOT) -> Path:
    """Locate a log's ``.db`` across the split directories."""
    for split in SPLITS:
        p = Path(data_root) / "nuplan-v1.1" / "splits" / split / f"{log}.db"
        if p.is_file():
            return p
    raise MapAccessError(f"log db not found for {log!r} under {data_root}")


def ego_expert_track(db_path: Path, t0: int, t1: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(timestamps_us, xy)`` of the logged (expert) ego between t0 and t1."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT lp.timestamp, ep.x, ep.y FROM lidar_pc lp "
            "JOIN ego_pose ep ON lp.ego_pose_token = ep.token "
            "WHERE lp.timestamp BETWEEN ? AND ? ORDER BY lp.timestamp",
            (int(t0), int(t1)),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return np.zeros(0, np.int64), np.zeros((0, 2), np.float64)
    return (np.array([r[0] for r in rows], np.int64),
            np.array([[r[1], r[2]] for r in rows], np.float64))


def log_location(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return str(con.execute("SELECT location FROM log LIMIT 1").fetchone()[0])
    finally:
        con.close()


def map_api_for(db_path: Path, *, maps_root: str = MAPS_ROOT):
    """``(map_api, location)`` for a log."""
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    location = log_location(db_path)
    return get_maps_api(maps_root, MAP_VERSION, location), location


def objects_at(map_api, x: float, y: float, layer) -> List:
    """Every map object of ``layer`` containing the point — never raises.

    ``get_all_map_objects`` is the plural form ``get_one_map_object`` should
    have been; on layers a map version does not carry it raises, which here
    means "this layer cannot answer", not "no objects".
    """
    from nuplan.common.actor_state.state_representation import Point2D

    try:
        return list(map_api.get_all_map_objects(Point2D(float(x), float(y)), layer) or [])
    except Exception:  # noqa: BLE001 — an unavailable layer is a "no", not a crash
        return []


def roadblock_at(map_api, x: float, y: float):
    """``(roadblock, is_connector)`` under a point, via ``LANE.parent``.

    Returns ``(None, False)`` off the drivable graph.
    """
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    for layer, is_conn in ((SemanticMapLayer.LANE, False),
                           (SemanticMapLayer.LANE_CONNECTOR, True)):
        for obj in objects_at(map_api, x, y, layer):
            parent = getattr(obj, "parent", None)
            if parent is not None:
                return parent, is_conn
    return None, False


def feeding_roads(map_api, block) -> Dict[str, tuple]:
    """The ROADS feeding ``block``, looked up through its connectors.

    Returns ``{roadblock_id: (roadblock, connector_or_None)}``.
    """
    out: Dict[str, tuple] = {}
    for conn in {c.id: c for c in block.incoming_edges}.values():
        ups = {r.id: r for r in getattr(conn, "incoming_edges", [])}
        if ups:
            for road in ups.values():
                out.setdefault(str(road.id), (road, conn))
        else:  # already a plain roadblock — some maps do join directly
            out.setdefault(str(conn.id), (conn, None))
    return out


def block_heading(block, *, at_end: bool = True) -> Optional[float]:
    """Travel direction of a roadblock, averaged over its lane baselines."""
    vecs = []
    for lane in getattr(block, "interior_edges", []) or []:
        try:
            pts = np.asarray([[s.x, s.y] for s in lane.baseline_path.discrete_path])
        except Exception:  # noqa: BLE001
            continue
        if len(pts) < 2:
            continue
        seg = pts[-1] - pts[-2] if at_end else pts[1] - pts[0]
        n = float(np.linalg.norm(seg))
        if n > 1e-6:
            vecs.append(seg / n)
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return float(np.arctan2(v[1], v[0]))


def heading_spread_deg(headings: List[float]) -> float:
    """Largest pairwise angle between travel directions, in degrees."""
    spread = 0.0
    for i in range(len(headings)):
        for j in range(i + 1, len(headings)):
            d = abs((headings[i] - headings[j] + np.pi) % (2 * np.pi) - np.pi)
            spread = max(spread, float(np.rad2deg(d)))
    return spread


def in_intersection(map_api, x: float, y: float) -> bool:
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    return bool(objects_at(map_api, x, y, SemanticMapLayer.INTERSECTION))


__all__ = [
    "DATA_ROOT", "MAPS_ROOT", "MAP_VERSION", "MapAccessError",
    "block_heading", "ego_expert_track", "feeding_roads", "find_log_db",
    "heading_spread_deg", "in_intersection", "log_location", "map_api_for",
    "objects_at", "roadblock_at",
]
