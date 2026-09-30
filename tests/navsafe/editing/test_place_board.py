"""The board's numbers have to mean what the leaf's numbers mean.

The board reads a point off a map and reports ``(arc, lateral)``; the leaf then
freezes those and `Polyline.offset` turns them back into a point. If those two
disagree the author places a sign in one spot and the bake stands it in
another -- which is exactly what happened once before, when the solver measured
arcs from the hand-off while `bake` measured them from the junction entry, and
ten recipes froze positions nobody had chosen. So the round trip is pinned.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from navsafe.benchmark.editing.place_board import (
    build_payload, host_payload, project_to_route, render_board)
from navsafe.benchmark.editing.placement.solve import Polyline


def _straight_route(length: float = 60.0) -> Polyline:
    pts = np.stack([np.linspace(0.0, length, 61),
                    np.zeros(61), np.zeros(61)], axis=1)
    return Polyline.from_points(pts, name="straight")


def _bent_route() -> Polyline:
    """A right-angle bend: the case where a tangent taken from the wrong
    segment silently flips which side `lateral` means."""
    a = np.stack([np.linspace(0.0, 30.0, 31), np.zeros(31), np.zeros(31)], axis=1)
    b = np.stack([np.full(30, 30.0), np.linspace(1.0, 30.0, 30), np.zeros(30)], axis=1)
    return Polyline.from_points(np.concatenate([a, b]), name="bent")


class TestRoundTrip:
    @pytest.mark.parametrize("route", [_straight_route(), _bent_route()])
    @pytest.mark.parametrize("arc,lateral", [
        (10.0, 3.0), (10.0, -3.0), (25.0, 0.0), (40.0, 5.5), (5.0, -1.25),
    ])
    def test_offset_then_project_returns_what_went_in(self, route, arc, lateral):
        handoff = 4.0
        x, y, _z, _h = route.offset(handoff + arc, lateral)
        got = project_to_route(route, (float(x[0]), float(y[0])), handoff_arc=handoff)
        assert got["arc"] == pytest.approx(arc, abs=0.35)
        assert got["lateral"] == pytest.approx(lateral, abs=0.05)

    def test_positive_lateral_is_the_left(self):
        """`Polyline.offset` documents itself as stepping to the LEFT, and the
        board's readout has to agree -- a sign is placed on the side the turn
        goes, so the sign of this number decides which kerb it stands on."""
        route = _straight_route()          # runs along +x, so left is +y
        x, y, _z, _h = route.offset(20.0, 3.0)
        assert float(y[0]) > 0
        got = project_to_route(route, (float(x[0]), float(y[0])), handoff_arc=0.0)
        assert got["lateral"] > 0

    def test_tangent_is_the_local_one_not_the_route_average(self):
        route = _bent_route()
        got = project_to_route(route, (32.0, 20.0), handoff_arc=0.0)
        assert math.degrees(got["tangent"]) == pytest.approx(90.0, abs=5.0)


class TestPayload:
    def test_a_recipe_without_actors_is_skipped(self, tmp_path):
        """A board of blanks is a board nobody reads, and half the frozen set
        has an empty cast on purpose."""
        (tmp_path / "V-8.deadbeefdeadbeef.yaml").write_text("actors: {}\n")
        assert build_payload([tmp_path / "V-8.deadbeefdeadbeef.yaml"],
                             corpus=tmp_path) == {}

    def test_host_payload_carries_what_the_board_draws(self, monkeypatch):
        probe = _FakeProbe()
        recipe = {"actors": {"sign": {
            "spawn": {"position": [3.0, -4.0, 1.0], "heading": 1.5},
            "asset": {"registry_key": "sign_no_left_turn"}}}}
        monkeypatch.setattr(
            "navsafe.benchmark.editing.placement.anchors.find_anchors",
            lambda probe, after_frame=None: [])
        d = host_payload(probe, recipe, after_frame=2)
        assert len(d["ego"]) == len(probe.ego_position)
        assert len(d["ego"][0]) == 3            # x, y, heading
        assert d["sign"]["xy"] == [3.0, -4.0]
        assert d["sign"]["key"] == "sign_no_left_turn"
        assert d["after_frame"] == 2
        assert len(d["route_arc"]) == len(d["route"])

    def test_render_embeds_the_data_and_the_title(self):
        html = render_board({"tok": {"ego": [[0, 0, 0]]}}, title="Board X")
        assert "__DATA__" not in html and "__TITLE__" not in html
        assert "Board X" in html
        assert '"tok"' in html


class _FakeProbe:
    """Enough of HostProbe for the payload builder, without a data root.

    `find_anchors` walks the route's lane ids looking for junctions, which
    needs far more of a scenario than a lane-filtering test should have to
    stand up -- so the stub answers it with nothing and the anchor search
    finds none.
    """
    def route_lane_ids(self):
        return []

    def __init__(self):
        self.ego_position = np.array([[float(k), 0.0, 0.0] for k in range(12)])
        self.after_frame = 2
        self.map_features = {
            "lane_1": {"polyline": [[0, 2, 0], [10, 2, 0]],
                       "polygon": [[0, 1, 0], [10, 1, 0], [10, 3, 0], [0, 3, 0]]},
            "far_lane": {"polyline": [[0, 900, 0], [10, 900, 0]], "polygon": None},
        }
        self.ego_route = _straight_route(11.0)

    def anchor_arc(self, route, frame):
        return float(frame)


def test_the_far_lane_is_dropped(monkeypatch):
    """A bounding box round the route drags in whole city blocks; the radius is
    measured to the ego PATH.

    The anchor search is stubbed out rather than satisfied: it walks a real
    route's lane graph, and growing the fake probe until it can is a test about
    HostProbe, not about which lanes the board keeps.
    """
    monkeypatch.setattr(
        "navsafe.benchmark.editing.placement.anchors.find_anchors",
        lambda probe, after_frame=None: [])
    d = host_payload(_FakeProbe(), {"actors": {"a": {"spawn": {}, "asset": {}}}},
                     after_frame=2)
    assert [lane["id"] for lane in d["lanes"]] == ["lane_1"]
