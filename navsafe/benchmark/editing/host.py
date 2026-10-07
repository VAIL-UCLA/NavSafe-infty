# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Load a host scenario for authoring, without booting a simulator.

Authoring, freezing and reviewing placement numbers are all CPU work on the
scenario dict. Only the render needs IsaacSim, so this takes the same path the
env's py123d loader does — scene enumeration, adapter, ScenarioDescription —
and stops there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_host_scenario(
    data_root: "str | Path",
    *,
    scene_id: Optional[str] = None,
    scene_index: int = 0,
    require_map: bool = True,
    load_map_objects: bool = True,
) -> Tuple[dict, str]:
    """Read one py123d Arrow scene as a ScenarioDescription.

    Args:
        data_root: converted py123d Arrow root.
        scene_id: exact scene uuid / log name. Wins over ``scene_index``.
        scene_index: positional index into the id-sorted scene list.
        load_map_objects: set False for a cheap state-only preflight; then
            require_map must also be False. Defaults preserve normal authoring.
        require_map: fail rather than author against a scene with no map —
            lane references and cross-sections need one.

    Returns:
        ``(sd, scene_id)``.
    """
    from navsafe.scenario.py123d_adapter import (
        Py123DAdapterConfig,
        scenario_from_py123d_scene,
    )
    from navsafe.scenario.py123d_scenario_description import py123d_to_scenario_description
    from navsafe.scenario.py123d_scenes import enumerate_scenes, scene_id as scene_id_of

    if require_map and not load_map_objects:
        raise ValueError("require_map=True conflicts with load_map_objects=False")
    data_root = Path(data_root)
    scenes = enumerate_scenes(data_root)
    if not scenes:
        raise FileNotFoundError(f"no py123d scenes under {data_root}")
    if scene_id:
        matches = [s for s in scenes if scene_id_of(s) == str(scene_id)]
        if not matches:
            raise KeyError(
                f"scene {scene_id!r} not found under {data_root} "
                f"({len(scenes)} scenes, e.g. {scene_id_of(scenes[0])})"
            )
        scene = matches[0]
    else:
        if not (0 <= int(scene_index) < len(scenes)):
            raise IndexError(f"scene_index {scene_index} outside 0..{len(scenes) - 1}")
        scene = scenes[int(scene_index)]

    scenario = scenario_from_py123d_scene(
        scene,
        Py123DAdapterConfig(
            load_state_payloads=True,
            load_custom_payloads=False,
            load_sensor_payloads=False,
            load_map_objects=bool(load_map_objects),
            data_root=str(data_root),
            require_map=bool(require_map),
        ),
    )
    sd = py123d_to_scenario_description(scenario)
    resolved = scene_id_of(scene)
    logger.info(
        "load_host_scenario: %s (%d tracks, %d map features)",
        resolved,
        len(sd.get("tracks", {})),
        len(sd.get("map_features", {})),
    )
    return sd, resolved


def describe_tracks(sd: dict, *, moving_only: bool = False) -> list:
    """Non-ego tracks with the numbers an author picks a relocation target by."""
    import numpy as np

    meta = sd.get("metadata", {}) or {}
    sdc_id = str(meta.get("sdc_id", "ego"))
    ego = (sd.get("tracks") or {}).get(sdc_id, {})
    ego_pos = np.asarray(ego.get("state", {}).get("position"), np.float64)
    rows = []
    for tid, track in (sd.get("tracks") or {}).items():
        if tid == sdc_id:
            continue
        state = track.get("state", {})
        pos = np.asarray(state.get("position"), np.float64)
        if pos.ndim != 2 or pos.shape[0] < 1:
            continue
        valid = np.asarray(state.get("valid", np.ones(len(pos), bool))).astype(bool)
        if not valid.any():
            continue
        travelled = float(np.linalg.norm(pos[valid][-1, :2] - pos[valid][0, :2]))
        if moving_only and travelled < 1.0:
            continue
        n = min(len(pos), len(ego_pos)) if ego_pos.ndim == 2 else 0
        nearest = (
            float(np.min(np.linalg.norm(pos[:n, :2] - ego_pos[:n, :2], axis=1))) if n else float("nan")
        )
        dims = [
            float(np.asarray(state.get(key, [0.0]))[0]) for key in ("length", "width", "height")
        ]
        rows.append(
            {
                "track_id": tid,
                "type": str(track.get("type", "")),
                "valid_frames": int(valid.sum()),
                "travelled_m": round(travelled, 1),
                "nearest_to_ego_m": round(nearest, 1),
                "dims": [round(d, 2) for d in dims],
            }
        )
    rows.sort(key=lambda r: r["nearest_to_ego_m"])
    return rows
