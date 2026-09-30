# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Universal ``map_features`` / ``dynamic_map_states`` for a reconstructed clip.

NuRec emits an OpenDRIVE **stub** -- a header element and nothing else -- so a
real2sim scenario has no road unless the map is taken from somewhere else.
py123d is that somewhere: every dataset it parses carries lanes and traffic
lights in one representation, and the eval path already projects them to
universal ScenarioNet polylines (``py123d_to_scenario_description``).  This
module provides that shared projection for reconstruction and evaluation.

Why an empty map is not a cosmetic problem: the drivable surface *is*
``map_features``.  Without it the top-down view has no lane markings, EPDMS
scores ``drivable_area_compliance = 0`` for every policy on every frame,
Bench2Drive's ``outside_route_lanes`` -- its only proportional Driving Score
penalty -- cannot be computed at all, and traffic-light compliance passes
trivially because there are no lights to run.  Three metrics and one picture,
all wrong for the same missing field, and none of them says so.

**Frames.**  A reconstructed scenario recenters its ego to (0, 0) at frame 0,
so the map has to land in that frame too.  Recent py123d projections already
declare ``coordinate == "local_frame0"`` and are recentred the same way, in
which case no offset is applied; older ones are in dataset-world coordinates
and need ``origin_offset`` (``= -ego_world[0]``) added.  Applying the offset to
an already-local map moves it roughly one UTM origin away -- about 1.2 km,
which reads downstream as "every candidate is off-road" rather than as a
transform bug, so the contract is read from the data instead of assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MapSource:
    """What was found, so callers can record provenance rather than guess."""

    map_features: Dict[str, Any]
    dynamic_map_states: Dict[str, Any]
    coordinate: str                 # the py123d projection's declared frame
    applied_offset: Tuple[float, float, float]
    clip_id: str
    root: str

    def summary(self) -> str:
        return (f"{len(self.map_features)} map features, "
                f"{len(self.dynamic_map_states)} traffic-light lanes from "
                f"{self.root} (coordinate={self.coordinate or 'unspecified'}, "
                f"offset={list(self.applied_offset)})")


def load_py123d_sd(py123d_root: str, clip_id: str) -> dict:
    """Project the py123d scene matching ``clip_id`` to a ScenarioDescription.

    Uses the same projection the ``Py123DLoader`` eval path uses, so the map and
    traffic-light representation are identical to a normal py123d scenario.
    """
    from navsafe.scenario.py123d_adapter import (
        Py123DAdapterConfig, scenario_from_py123d_scene,
    )
    from navsafe.scenario.py123d_scenario_description import (
        py123d_to_scenario_description,
    )
    from navsafe.scenario.py123d_scenes import enumerate_scenes, scene_id

    scenes = enumerate_scenes(py123d_root)
    if not scenes:
        raise FileNotFoundError(f"No py123d scenes under {py123d_root!r}")

    def _ids(sc) -> list:
        # The reconstruction clip id matches py123d's ``log_name``;
        # ``scene_uuid`` is a separately generated id. Match on either.
        return [i for i in (str(getattr(sc, "log_name", "") or ""),
                            str(scene_id(sc) or "")) if i]

    match = next((sc for sc in scenes
                  if any(i == clip_id or clip_id in i or i in clip_id
                         for i in _ids(sc))), None)
    if match is None:
        raise LookupError(
            f"No py123d scene matches clip_id={clip_id!r} under {py123d_root!r}. "
            f"Available (log_name, uuid), first 10: {[_ids(s) for s in scenes][:10]}")

    scenario = scenario_from_py123d_scene(
        match, Py123DAdapterConfig(load_map_objects=True, data_root=str(py123d_root)))
    return py123d_to_scenario_description(scenario)


def recenter_map_features(map_features: Dict[str, Any],
                          origin_offset) -> Dict[str, Any]:
    """``map_sd = map_world + origin_offset`` on every coordinate array.

    Every ``(N, 2|3)`` float array in a feature (polyline, boundaries, polygon)
    is shifted; scalars and strings (``type``) pass through.
    """
    off = np.asarray(origin_offset, np.float64)

    def _shift(arr) -> np.ndarray:
        a = np.asarray(arr, np.float64)
        if a.ndim == 2 and a.shape[1] in (2, 3):
            a = a.copy()
            k = min(a.shape[1], 3)
            a[:, :k] += off[:k]
        return a.astype(np.float32)

    out: Dict[str, Any] = {}
    for fid, feat in map_features.items():
        if not isinstance(feat, dict):
            out[fid] = feat
            continue
        nf: Dict[str, Any] = {}
        for k, v in feat.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[-1] in (2, 3):
                nf[k] = _shift(v)
            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                nf[k] = [_shift(x) for x in v]
            else:
                nf[k] = v
        out[fid] = nf
    return out


def recenter_dynamic_map_states(dynamic_map_states: Dict[str, Any],
                                origin_offset) -> Dict[str, Any]:
    """Shift each traffic light's ``stop_point``; per-frame states pass through."""
    off = np.asarray(origin_offset, np.float64)

    out: Dict[str, Any] = {}
    for lane_id, entry in dynamic_map_states.items():
        if not isinstance(entry, dict):
            out[lane_id] = entry
            continue
        ne = dict(entry)
        sp = entry.get("stop_point")
        if isinstance(sp, np.ndarray) and sp.shape[-1] in (2, 3):
            shifted = sp.astype(np.float64).copy()
            k = min(sp.shape[-1], 3)
            shifted[:k] += off[:k]
            ne["stop_point"] = shifted.astype(np.float32)
        out[lane_id] = ne
    return out


def map_for_clip(py123d_root: str, clip_id: str,
                 origin_offset) -> MapSource:
    """Map + traffic lights for ``clip_id``, in the scenario's own frame."""
    sd = load_py123d_sd(py123d_root, clip_id)
    coordinate = str((sd.get("metadata") or {}).get("coordinate", ""))
    # Already recentred by the projection? Then applying the offset again would
    # translate the map off the scene (see the module note).
    applied = (np.zeros(3, np.float64) if coordinate == "local_frame0"
               else np.asarray(origin_offset, np.float64))
    return MapSource(
        map_features=recenter_map_features(
            dict(sd.get("map_features", {}) or {}), applied),
        dynamic_map_states=recenter_dynamic_map_states(
            dict(sd.get("dynamic_map_states", {}) or {}), applied),
        coordinate=coordinate,
        # `applied` is a 3-vector by construction (np.zeros(3) or the caller's
        # xyz offset), which a generator-built tuple cannot express statically.
        applied_offset=cast(
            "Tuple[float, float, float]", tuple(float(v) for v in applied)),
        clip_id=clip_id,
        root=str(py123d_root),
    )


def map_for_clip_or_none(py123d_root: Optional[str], clip_id: str,
                         origin_offset) -> Optional[MapSource]:
    """``map_for_clip`` that reports failure instead of raising.

    A reconstruction pipeline should not die because a map root is missing --
    but it must not stay quiet either, because the resulting scenario looks
    complete and scores every policy 0 on drivable area. Every path through
    here logs what happened.
    """
    if not py123d_root:
        logger.warning(
            "no py123d root given: map_features/dynamic_map_states stay EMPTY "
            "for clip %s. Downstream this reads as drivable_area_compliance=0 "
            "for every policy, no outside_route_lanes penalty, no lane "
            "markings in the top-down view, and trivially-passing traffic "
            "lights. Pass py123d_root to load the map from Arrow.", clip_id)
        return None
    try:
        src = map_for_clip(py123d_root, clip_id, origin_offset)
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
        logger.warning("map lookup failed for clip %s under %s: %s -- "
                       "map_features/dynamic_map_states stay EMPTY",
                       clip_id, py123d_root, exc)
        return None
    if not src.map_features:
        logger.warning("py123d scene for clip %s carries no map features", clip_id)
    logger.info("map for %s: %s", clip_id, src.summary())
    return src
