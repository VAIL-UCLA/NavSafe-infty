# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box dimensions of ScenarioNet tracks.

Single home for the "where do a track's length/width live" logic so the
planner's agent predictions, the EPDMS scorer's collision boxes, and any
other consumer stay bit-identical — the emergency brake and the scorer
must agree on collision geometry. ScenarioNet stores dims in per-frame
arrays under ``track["state"]["length"]`` / ``["width"]`` (scalars in
some packs); synthetic fixtures sometimes put them at the track top
level. Unknown tracks fall back to type-based defaults — never to ego
dims, which would turn pedestrians into car-sized phantom obstacles.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Fallback dims (metres) per ScenarioNet track type, used when a track
#: carries no dimension data (mirrors ``scenario_replay_manager``).
AGENT_DEFAULT_DIMS: Dict[str, Tuple[float, float]] = {
    "VEHICLE": (4.5, 1.8),
    "CYCLIST": (1.8, 0.6),
    "PEDESTRIAN": (0.6, 0.6),
}


def track_dims(
    track: Dict[str, Any],
    frame_id: int,
    *,
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    """(length, width) of a ScenarioNet track at ``frame_id``.

    Resolution order per dimension: per-frame/scalar value under
    ``track["state"]`` → track top level → type default from
    :data:`AGENT_DEFAULT_DIMS` → ``fallback`` (for unknown types).
    """
    state = track.get("state", {})

    def _lookup(key: str) -> Optional[float]:
        val = state.get(key)
        if val is None:
            val = track.get(key)
        if val is None:
            return None
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return None
        out = float(arr[min(int(frame_id), arr.size - 1)])
        return out if out > 0.0 else None

    default_l, default_w = AGENT_DEFAULT_DIMS.get(
        str(track.get("type", "")), fallback
    )
    length = _lookup("length")
    width = _lookup("width")
    return (
        length if length is not None else default_l,
        width if width is not None else default_w,
    )


def track_height(
    track: Dict[str, Any],
    frame_id: int,
    *,
    fallback: float,
) -> float:
    """Bbox height of a ScenarioNet track at ``frame_id``.

    The height analog of :func:`track_dims`, same resolution order:
    per-frame/scalar value under ``track["state"]`` → track top level →
    ``fallback`` (callers pass their type default; there is no height
    column in :data:`AGENT_DEFAULT_DIMS`).
    """
    val = track.get("state", {}).get("height")
    if val is None:
        val = track.get("height")
    if val is None:
        return fallback
    arr = np.asarray(val, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return fallback
    h = float(arr[min(int(frame_id), arr.size - 1)])
    return h if h > 0.0 else fallback


__all__ = ["AGENT_DEFAULT_DIMS", "track_dims", "track_height"]
