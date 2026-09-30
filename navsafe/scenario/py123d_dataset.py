# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-level py123d read helpers: multi-scene listing + load-by-index.

Replaces the legacy ScenarioNet ``ScenarioLoader`` disk-pickle utility. Tooling
that resolved scenario ids/indices from a pickle dataset (eval entry points,
BEV/front visualizers) now reads py123d Arrow logs through these, built on
:func:`navsafe.scenario.py123d_scenes.enumerate_scenes`.

py123d is imported lazily (via the helpers below / ``enumerate_scenes``) so this
module stays import-safe when py123d is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from navsafe.scenario.py123d_scenes import enumerate_scenes, scene_id


def list_scene_ids(data_root: str | Path) -> List[str]:
    """Scene ids under a py123d Arrow data root, in discovery order."""
    return [scene_id(scene) for scene in enumerate_scenes(data_root)]


def scenario_description_by_index(data_root: str | Path, index: int, *, require_map: bool = True):
    """Load one py123d scene by positional index → ``ScenarioDescription``.

    Raises ``FileNotFoundError`` when the root holds no scenes and ``IndexError``
    when ``index`` is out of range.
    """
    from navsafe.scenario.py123d_adapter import Py123DAdapterConfig, scenario_from_py123d_scene
    from navsafe.scenario.py123d_scenario_description import py123d_to_scenario_description

    scenes = enumerate_scenes(data_root)
    if not scenes:
        raise FileNotFoundError(f"No py123d scenes found under: {data_root}")
    if not 0 <= index < len(scenes):
        raise IndexError(f"scene index {index} out of range [0, {len(scenes)})")
    scenario = scenario_from_py123d_scene(
        scenes[index], Py123DAdapterConfig(data_root=str(data_root), require_map=require_map)
    )
    return py123d_to_scenario_description(scenario)


__all__ = ["scene_id", "list_scene_ids", "scenario_description_by_index"]
