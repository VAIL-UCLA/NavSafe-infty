# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Lane features must carry their true polygon.

py123d ships each lane's footprint (``Lane.shapely_polygon`` ->
``NexusLaneState.polygon``), but the ScenarioDescription converter used to drop
it. Every downstream consumer then saw ZERO polygons and synthesised one
instead: ``LaneProxy`` extrudes ``centerline.buffer(1.75)`` — a uniform 3.5 m
ribbon. Real lanes in this data are 2.4-7.5 m wide (median ~3.9) and flare
through turns, so the fallback was up to 2x too narrow and the wrong shape.

EPDMS ``dac`` tests all four ego corners against these polygons, so a car
legitimately tracking a wide/flaring lane scored as off-road. Measured at
bbb77289 f40: every proposal dac=0.250 (EPDMS 0.184, gate FAIL) with the
synthetic ribbon vs dac=1.000 (EPDMS 0.734, PASS) with the real polygons —
without changing the trajectory at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.scenario.scenario_description import ScenarioDescription as SD


def test_converter_emits_lane_polygon_when_source_has_one():
    """The converter must forward ``NexusLaneState.polygon`` to SD.POLYGON."""
    from navsafe.scenario.py123d_scenario_description import (
        _map_features_from_map_state,
    )

    ring = np.array([[0.0, -1.9, 0.0], [10.0, -1.9, 0.0],
                     [10.0, 1.9, 0.0], [0.0, 1.9, 0.0],
                     [0.0, -1.9, 0.0]], dtype=np.float32)

    class _Lane:
        lane_type = "SURFACE_STREET"
        centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                              dtype=np.float32)
        left_boundary = np.array([[0.0, 1.9, 0.0], [10.0, 1.9, 0.0]],
                                 dtype=np.float32)
        right_boundary = np.array([[0.0, -1.9, 0.0], [10.0, -1.9, 0.0]],
                                  dtype=np.float32)
        polygon = ring

    class _MapState:
        lanes = {"lane_0": _Lane()}
        road_edges: dict = {}
        road_lines: dict = {}

    feats = _map_features_from_map_state(_MapState())
    assert "lane_0" in feats
    poly = feats["lane_0"].get(SD.POLYGON)
    assert poly is not None, (
        "lane polygon dropped — consumers will fall back to a synthetic "
        "centerline.buffer(1.75) ribbon and dac will score real driving as "
        "off-road")
    assert np.asarray(poly).shape[0] >= 3


def test_degenerate_polygon_is_not_emitted():
    """A <3-point 'polygon' is not a polygon; omit rather than emit garbage."""
    from navsafe.scenario.py123d_scenario_description import (
        _map_features_from_map_state,
    )

    class _Lane:
        lane_type = "SURFACE_STREET"
        centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                              dtype=np.float32)
        left_boundary = None
        right_boundary = None
        polygon = np.zeros((0, 3), dtype=np.float32)

    class _MapState:
        lanes = {"lane_0": _Lane()}
        road_edges: dict = {}
        road_lines: dict = {}

    feats = _map_features_from_map_state(_MapState())
    assert feats["lane_0"].get(SD.POLYGON) is None


@pytest.mark.skipif(
    not __import__("pathlib").Path("data").exists(),
    reason="needs the py123d data root",
)
def test_real_scene_lanes_carry_polygons_and_realistic_widths():
    """End-to-end against real data: polygons present, widths not uniform.

    A uniform width across every lane is the fingerprint of the synthetic
    ribbon fallback.
    """
    shapely = pytest.importorskip("shapely.geometry")
    pytest.importorskip("py123d")
    from navsafe.scenario.py123d_dataset import scenario_description_by_index

    sd = scenario_description_by_index("data", 20)
    lanes = [v for v in sd[SD.MAP_FEATURES].values()
             if "LANE" in str(v.get(SD.TYPE, ""))]
    assert lanes, "no lane features"
    with_poly = [v for v in lanes
                 if v.get(SD.POLYGON) is not None
                 and len(np.asarray(v[SD.POLYGON])) >= 3]
    assert len(with_poly) == len(lanes), (
        f"only {len(with_poly)}/{len(lanes)} lanes carry a polygon")

    widths = []
    for v in with_poly:
        pg = shapely.Polygon(np.asarray(v[SD.POLYGON])[:, :2])
        cl = np.asarray(v[SD.POLYLINE])[:, :2]
        length = float(np.sum(np.hypot(*np.diff(cl, axis=0).T)))
        if pg.is_valid and pg.area > 0 and length > 1.0:
            widths.append(pg.area / length)
    assert widths
    # The synthetic fallback would make every lane exactly 3.5 m.
    assert np.ptp(widths) > 0.5, (
        "lane widths are ~uniform — still on the synthetic ribbon fallback")
