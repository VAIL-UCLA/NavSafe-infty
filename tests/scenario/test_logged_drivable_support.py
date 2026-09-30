"""Regression tests for expert-supported holes in source drivable maps."""

import numpy as np
from shapely.geometry import LineString

from navsafe.evaluation.utils.lane_proxy import (
    DrivableAreaProxy,
    corners_in_drivable_area,
)
from navsafe.scenario.py123d_scenario_description import (
    _add_logged_drivable_support,
)


def _scenario(xy):
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    lane = LineString([(-20.0, 0.0), (20.0, 0.0)]).buffer(
        1.75, cap_style=2)
    return {
        "metadata": {"sdc_id": "ego", "route_lane_ids": ["lane"],
                     "dataset": "nuplan"},
        "map_features": {
            "lane": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array([[-20.0, 0.0], [20.0, 0.0]]),
                "polygon": np.asarray(lane.exterior.coords[:-1]),
            }
        },
        "tracks": {"ego": {"state": {
            "position": np.column_stack([xy, np.zeros(n)]),
            "heading": np.zeros(n),
            "velocity": np.tile([4.0, 0.0], (n, 1)),
            "length": np.full(n, 4.8),
            "width": np.full(n, 1.852),
            "valid": np.ones(n, dtype=bool),
        }}},
    }


def test_does_not_change_a_complete_map():
    sd = _scenario([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    assert _add_logged_drivable_support(sd, "ego") == 0
    assert set(sd["map_features"]) == {"lane"}
    assert "logged_drivable_support" not in sd["metadata"]


def test_adds_only_expert_swept_surface_for_a_map_hole():
    # Like 891953...: leave the mapped lane, turn, and return.  The source map
    # has no connector polygon, so the unmodified DAC surface cannot contain
    # the logged vehicle at y=-8 m.
    sd = _scenario([[0.0, 0.0], [0.0, -3.0], [0.0, -6.0], [0.0, -8.0],
                    [2.0, -8.0], [4.0, -6.0], [4.0, -3.0], [4.0, 0.0]])
    route_before = list(sd["metadata"]["route_lane_ids"])
    assert _add_logged_drivable_support(sd, "ego") >= 1
    assert sd["metadata"]["route_lane_ids"] == route_before

    polys = []
    for feature in sd["map_features"].values():
        if str(feature["type"]).startswith("LANE"):
            from shapely.geometry import Polygon
            polys.append(Polygon(feature["polygon"]))
    proxy = DrivableAreaProxy.from_lane_polygons(polys)
    support = sd["map_features"]["__logged_drivable_support_0"]
    assert all(f["provenance"] == "logged_ego_standard_lane_connector"
               for key, f in sd["map_features"].items()
               if key.startswith("__logged_drivable_support_"))
    assert support["speed_limit_mps"] == 4.0
    assert support["speed_limit_source"] == "logged_drivable_support_p85"
    support_meta = sd["metadata"]["logged_drivable_support"]
    expected_width = 2.0 * (np.hypot(2.4, .926) + .75)
    assert support_meta["version"] == 5
    assert support_meta["connector_width_m"] == expected_width
    assert support_meta["controller_tracking_tolerance_m"] == .75
    assert support_meta["repair_mode"] == "missing_turning_envelope"
    assert proxy is not None

    # Every logged ego footprint now passes, while a point 5 m beyond the exact
    # supported turn remains off-road (the repair is not a blanket buffer).
    from shapely.geometry import Polygon
    for x, y in np.asarray(sd["tracks"]["ego"]["state"]["position"])[:, :2]:
        ego = Polygon([(x + 2.4, y + .926), (x + 2.4, y - .926),
                       (x - 2.4, y - .926), (x - 2.4, y + .926)])
        assert corners_in_drivable_area(
            ego.exterior.coords[:-1], proxy, lambda: ())
    assert not proxy.contains_point(0.0, -14.0)


def test_fills_tracking_tolerance_seam_even_when_exact_log_fits():
    """An acute connector needs support around, not only on, the expert."""
    sd = _scenario([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    # The exact 1.852 m-wide expert fits, but there is less than the declared
    # 0.75 m closed-loop tracking margin on this side of the source lane.
    from shapely.geometry import Polygon
    narrow = LineString([(-20.0, 0.0), (20.0, 0.0)]).buffer(
        1.05, cap_style=2)
    sd["map_features"]["lane"]["polygon"] = np.asarray(
        narrow.exterior.coords[:-1])

    assert _add_logged_drivable_support(sd, "ego") >= 1
    support_meta = sd["metadata"]["logged_drivable_support"]
    assert support_meta["repair_mode"] == "tracking_tolerance_seam"
    assert support_meta["connector_width_m"] == 2.0 * (.926 + .75)
    polys = [Polygon(f["polygon"]) for f in sd["map_features"].values()
             if str(f["type"]).startswith("LANE")]
    proxy = DrivableAreaProxy.from_lane_polygons(polys)
    # Closed-loop center offset by 0.6 m is inside the expert-supported margin.
    ego = Polygon([(2.4, 1.526), (2.4, -.326),
                   (-2.4, -.326), (-2.4, 1.526)])
    assert corners_in_drivable_area(
        ego.exterior.coords[:-1], proxy, lambda: ())


def test_missing_map_remains_a_data_error():
    sd = _scenario([[0.0, 0.0], [0.0, -3.0]])
    sd["map_features"] = {}
    assert _add_logged_drivable_support(sd, "ego") == 0
    assert sd["map_features"] == {}


def test_generated_scenario_preserves_exact_map():
    sd = _scenario([[0.0, 0.0], [0.0, -8.0]])
    sd["metadata"]["dataset"] = "navsafe_pg"
    assert _add_logged_drivable_support(sd, "ego") == 0
    assert set(sd["map_features"]) == {"lane"}
