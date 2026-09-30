# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Where the stop line is, for the leaves that score a hold behind it.

nuPlan publishes no stop-line layer, so the line is the signalled connector's
entry -- the same geometry the evaluator's red-light rule uses. Two ways to get
that wrong, both measured on real bundles:

* Matching the signal by its ENTRY VERTEX rejects the right one. On
  ``00c1e4eb4a045f20`` the route runs straight down lane 53726 (closest approach
  0.00 m) while that lane's first vertex is 6.38 m off the path, so a 6 m
  tolerance on the vertex threw away the only light the scenario has.
* Accepting an entry that projects to the start of the route invents a line the
  ego was never in front of. That is the same bundle again: its connector entry
  lands on frame 0, which is why it cannot host a hold without a re-cut window.
"""

from __future__ import annotations

import numpy as np

from navsafe.benchmark.trace import from_eval as fe


def _sd(lanes: dict) -> dict:
    return {"dynamic_map_states": {k: {} for k in lanes}}


def _straight(n: int = 50, x0: float = 0.0) -> np.ndarray:
    return np.stack([np.linspace(x0, x0 + n - 1, n), np.zeros(n)], axis=1)


def test_the_entry_arc_is_the_stop_line():
    route = _straight(50)
    lanes = {"L": _straight(10, x0=20.0)}
    assert fe.stopline_arc(_sd(lanes), route, lanes) == 20.0


def test_a_connector_on_the_route_counts_even_when_its_entry_is_offset():
    """`00c1e4eb4a045f20`: route down the lane, entry vertex 6.4 m to the side."""
    route = _straight(50)
    centre = _straight(10, x0=20.0)
    centre[0] = [20.0, 6.4]                       # entry vertex off the path
    lanes = {"L": centre}
    assert fe.stopline_arc(_sd(lanes), route, lanes) is not None


def test_a_signal_nowhere_near_the_route_is_not_this_ego_s():
    route = _straight(50)
    lanes = {"L": _straight(10, x0=20.0) + np.array([0.0, 40.0])}
    assert fe.stopline_arc(_sd(lanes), route, lanes) is None


def test_an_entry_at_the_window_start_is_no_stop_line():
    """Level with the line on frame 0 means there is no approach to hold."""
    route = _straight(50)
    lanes = {"L": _straight(10, x0=0.0)}
    assert fe.stopline_arc(_sd(lanes), route, lanes) is None


def test_the_nearest_signal_ahead_wins():
    route = _straight(80)
    lanes = {"far": _straight(10, x0=60.0), "near": _straight(10, x0=25.0)}
    assert fe.stopline_arc(_sd(lanes), route, lanes) == 25.0


def test_no_lights_or_no_centrelines_resolves_nothing():
    route = _straight(50)
    assert fe.stopline_arc({"dynamic_map_states": {}}, route, {"L": _straight(5)}) is None
    assert fe.stopline_arc(_sd({"L": None}), route, None) is None
    assert fe.stopline_arc(_sd({"L": None}), route, {}) is None


def test_the_column_is_signed_and_positive_before_the_line():
    frames = fe.build_frames(
        _straight(30), _straight(30), agents=[], per_frame={},
        warmup_frames=0, dt=0.1, drivable_known=False, stopline_s=20.0)
    assert frames[0]["dist_to_stopline_m"] == 20.0
    assert frames[20]["dist_to_stopline_m"] == 0.0
    assert frames[25]["dist_to_stopline_m"] == -5.0


def test_without_a_stop_line_the_column_stays_not_checked():
    frames = fe.build_frames(
        _straight(10), _straight(10), agents=[], per_frame={},
        warmup_frames=0, dt=0.1, drivable_known=False)
    value = frames[0]["dist_to_stopline_m"]
    assert value != value, "absent must be NaN, never a line at the origin"
