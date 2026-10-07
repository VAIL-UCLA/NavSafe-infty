# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared py123d Arrow-scene enumeration.

A single place that fetches the filtered scene list from a converted py123d
data root, with optional capping and a seeded deterministic shuffle. The
runtime ``Py123DLoader``, the online-RL sampler, and the BC cache builder all
build on this so the fetch/cap/shuffle logic lives once.

py123d is imported lazily so this module stays import-safe when py123d is
absent (doc builds, schema-only tests).
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any, List

import numpy as np


def scene_id(scene: Any) -> str:
    """Stable id for a py123d scene handle (scene uuid, else log name)."""
    return str(getattr(scene, "scene_uuid", None) or getattr(scene, "log_name", "") or "")


def enumerate_scenes(
    data_root: str | Path,
    *,
    scene_uuids: Sequence[str] | None = None,
    max_scenes: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> List[Any]:
    """Return py123d Arrow scenes under ``data_root``.

    Scenes are fetched via py123d's ``SceneFilter`` (which has no count cap),
    capped to the first ``max_scenes`` when set, then reordered by a seeded RNG
    when ``shuffle`` is true. Returns ``[]`` for a root with no scenes —
    empty-handling is the caller's policy, not this helper's.

    ``scene_uuids`` restricts the fetch to named scenes, pushed down into
    ``SceneFilter`` rather than filtered afterwards so a targeted selection
    does not pay for reading the whole root. The caller's index addresses
    this filtered list. A uuid
    the root does not hold is an error, not a silent short list — a curriculum
    that quietly collected the wrong scenes would be unauditable.
    """
    try:
        from py123d.api import SceneFilter, get_filtered_scenes
    except ImportError as exc:  # pragma: no cover - optional runtime dep
        raise ImportError(
            "py123d is required to read Arrow scenes. Install it with `uv sync` "
            "(it is a core dependency)."
        ) from exc

    wanted = [str(value) for value in scene_uuids] if scene_uuids is not None else None
    if wanted is not None and not wanted:
        return []
    scene_filter = SceneFilter(scene_uuids=wanted) if wanted else SceneFilter()
    scenes = list(get_filtered_scenes(scene_filter, data_root=str(data_root)))
    if wanted is not None:
        missing = sorted(set(wanted) - {scene_id(scene) for scene in scenes})
        if missing:
            raise ValueError(
                f"{len(missing)} requested scene uuid(s) are not in {data_root}: "
                f"{missing[:5]}"
            )
    # Stable ordering: py123d's filter order is an implementation detail, so
    # sort by scene id to make index → scene reproducible across runs/machines.
    scenes.sort(key=scene_id)
    if max_scenes is not None:
        scenes = scenes[: int(max_scenes)]
    if shuffle:
        rng = np.random.default_rng(int(seed))
        order = np.arange(len(scenes))
        rng.shuffle(order)
        scenes = [scenes[i] for i in order]
    return scenes


__all__ = ["enumerate_scenes", "scene_id"]
