"""Read-path fidelity guards for py123d → ScenarioDescription projection.

Covers two correctness fixes:

* Map features must not collide across layers. py123d numbers map-object ids
  per-layer from 0, so a lane, a road edge, and a road line can all be id ``0``;
  the projection must keep all three (it used to key them in one flat dict, so
  the last one silently overwrote the others — dropping lanes for real datasets).
* Unknown actor labels must not be silently mislabeled as ``VEHICLE``; they map
  to ``OTHER`` (honest), with a strict mode that raises for baseline-grade builds.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import navsafe.scenario.py123d_scenario_description as proj
from navsafe.scenario.py123d_scenario_description import _map_agent_type, _map_features_from_map_state
from navsafe.scenario.type import MetaDriveType


def _line(y: float) -> SimpleNamespace:
    poly = np.array([[0.0, y, 0.0], [1.0, y, 0.0]], dtype=np.float32)
    return SimpleNamespace(polyline=poly, semantic_type="solid_single_white", layer=None)


def _lane(y: float) -> SimpleNamespace:
    cl = np.array([[0.0, y, 0.0], [1.0, y, 0.0]], dtype=np.float32)
    return SimpleNamespace(lane_type="surface_street", centerline=cl, left_boundary=None, right_boundary=None)


# ─────────────────────────── map-feature collision ───────────────────────────

def test_map_features_survive_cross_layer_id_collision() -> None:
    """A lane, road edge, and road line all numbered ``0`` must all survive."""
    map_state = SimpleNamespace(
        lanes={"0": _lane(0.0)},
        road_edges={"0": _line(2.0)},
        road_lines={"0": _line(4.0)},
    )
    features = _map_features_from_map_state(map_state)
    assert len(features) == 3, f"cross-layer id collision dropped features: {features}"
    kinds = [f["type"] for f in features.values()]
    # Exactly one lane survives (the other two are line/edge features).
    assert sum(MetaDriveType.is_lane(k) for k in kinds) == 1


def test_map_feature_lane_keeps_raw_id_for_traffic_light_crossref() -> None:
    """Lane keys stay the raw id so dynamic_map_states traffic lights resolve."""
    map_state = SimpleNamespace(lanes={"lane_7": _lane(0.0)}, road_edges={}, road_lines={})
    features = _map_features_from_map_state(map_state)
    assert "lane_7" in features


# ─────────────────────────── label integrity ───────────────────────────

def test_known_and_keyword_labels_map_correctly() -> None:
    assert _map_agent_type("regular_vehicle") == MetaDriveType.VEHICLE
    assert _map_agent_type("pedestrian") == MetaDriveType.PEDESTRIAN
    assert _map_agent_type("some_bicycle_thing") == MetaDriveType.CYCLIST  # keyword fallback


def test_unknown_label_maps_to_other_not_vehicle() -> None:
    assert _map_agent_type("flying_saucer") == MetaDriveType.OTHER
    assert _map_agent_type(None) == MetaDriveType.OTHER


def test_strict_mode_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        _map_agent_type("flying_saucer", strict=True)
    # strict mode must not raise for a recognized label
    assert _map_agent_type("regular_vehicle", strict=True) == MetaDriveType.VEHICLE


def test_unknown_label_warns_once(caplog) -> None:
    proj._WARNED_UNKNOWN_LABELS.discard("zzz_unknown_label")
    with caplog.at_level(logging.WARNING, logger="navsafe.scenario.py123d_scenario_description"):
        _map_agent_type("zzz_unknown_label")
        _map_agent_type("zzz_unknown_label")  # second call must not warn again
    hits = [r for r in caplog.records if "zzz_unknown_label" in r.getMessage()]
    assert len(hits) == 1
