# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The post-hoc trace carries lane columns, so the C-3 gate is not a no-op.

``scenario_rules.lane_change_frames`` reads ``lane_id`` and the signed
``lateral_offset_m``. Only ``trace/writer.py`` (the opt-in live trace, enabled
by ``NAVSAFE_TRACE``) ever wrote them; the path every real eval actually scores
through -- ``trace/from_eval.build_frames`` over a finished eval directory --
left them empty. ``has_lane_data`` then correctly reported "not checked", which
meant the C-3 lane-change gate silently did nothing in production while its
unit tests passed on synthetic frames.

Both paths now compute the lane frame with the same
``writer.nearest_lane``/``lane_centerlines`` pair, so they cannot disagree about
which lane the ego is in, or about the sign that makes a crossing detectable.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark import scenario_rules as SR
from navsafe.benchmark import termination as T
from navsafe.benchmark.trace.from_eval import build_frames
from navsafe.benchmark.trace.writer import lane_centerlines, nearest_lane

LANE_W = 3.5


def _two_lane_map() -> dict:
    """Two straight parallel lanes, 3.5 m apart, along +x."""
    return {"map_features": {
        "L1": {"type": "LANE_SURFACE_STREET",
               "polyline": np.array([[float(x), 0.0, 0.0] for x in range(0, 120, 5)])},
        "L2": {"type": "LANE_SURFACE_STREET",
               "polyline": np.array([[float(x), LANE_W, 0.0] for x in range(0, 120, 5)])},
    }}


def _frames(ys: np.ndarray, lane_center=None) -> list[dict]:
    xs = np.arange(len(ys), dtype=float)
    ego = np.stack([xs, ys], axis=1)
    route = np.stack([xs, np.zeros_like(xs)], axis=1)
    return build_frames(ego, route, [], {}, warmup_frames=0, dt=0.1,
                        drivable_known=True, lane_center=lane_center)


def test_the_offset_sign_flips_across_the_boundary():
    """The assumption the whole detector rests on, on real geometry code."""
    lc = lane_centerlines(_two_lane_map())
    _, _, near_l1 = nearest_lane(5.0, 1.4, lc)
    lane_hi, _, near_l2 = nearest_lane(5.0, 2.1, lc)
    assert near_l1 > 0 and near_l2 < 0, (
        f"offsets {near_l1} / {near_l2} do not straddle the boundary")
    assert lane_hi == "L2"


def test_a_lane_change_is_found_on_eval_path_frames():
    ys = np.clip((np.arange(60.0) - 30) * 0.5, 0, LANE_W)
    frames = _frames(ys, lane_centerlines(_two_lane_map()))
    assert SR.has_lane_data(frames)
    changes = SR.lane_change_frames(frames)
    assert len(changes) == 1
    # The crossing is geometrically at y = LANE_W/2, i.e. ~frame 33-35.
    assert 30 <= changes[0] <= 40, changes


def test_holding_the_lane_is_not_a_lane_change():
    frames = _frames(np.zeros(60), lane_centerlines(_two_lane_map()))
    assert SR.has_lane_data(frames)
    assert SR.lane_change_frames(frames) == []


def test_without_a_map_the_columns_stay_empty_and_the_gate_abstains():
    """No map_features (or an older artifact) must not fail every C-3 episode."""
    frames = _frames(np.clip((np.arange(60.0) - 30) * 0.5, 0, LANE_W))
    assert not SR.has_lane_data(frames)
    assert all(f["lane_id"] == "" for f in frames)


@pytest.mark.parametrize("changes_lane", [True, False])
def test_the_c3_gate_decides_the_goal_end_to_end(changes_lane):
    """classify() over eval-path frames, with the real C-3 rules.

    This is the join the unit tests could not cover: synthetic frames proved the
    detector, and this proves the detector sees what a scored run actually
    carries.
    """
    ys = (np.clip((np.arange(60.0) - 30) * 0.5, 0, LANE_W) if changes_lane
          else np.zeros(60))
    frames = _frames(ys, lane_centerlines(_two_lane_map()))
    rules = SR.rules_for_scenario({
        "taxonomy_leaves": ["C-3"],
        "scenario_types": ["changing_lane_with_trail"],
    })
    assert rules.require_lane_change

    term = T.classify(frames, t_max_s=60.0, goal_reached=True,
                      goal_frame=len(frames) - 1, rules=rules)
    if changes_lane:
        assert term.reason is T.TerminationReason.GOAL_REACHED
        assert "lane change" in term.detail
    else:
        assert term.reason is not T.TerminationReason.GOAL_REACHED

    # The default rules must still credit the goal either way, or the gate is
    # changing more than the one leaf it is scoped to.
    plain = T.classify(frames, t_max_s=60.0, goal_reached=True,
                       goal_frame=len(frames) - 1)
    assert plain.reason is T.TerminationReason.GOAL_REACHED


def _map_with_support_lane() -> dict:
    """The real two-lane map plus a synthesised support lane on the ego's path.

    `_add_logged_drivable_support` adds `__logged_drivable_support_<i>`
    polygons, typed as lanes, wherever the source map has a hole the logged ego
    drove through. They follow the LOGGED PATH, so on a lane-change scenario the
    support lane is nearest for most frames and `lane_id` tracks the log rather
    than the lane graph.
    """
    m = _two_lane_map()
    ys = np.clip((np.arange(0, 120, 5).astype(float) - 30) * 0.5, 0, LANE_W)
    m["map_features"]["__logged_drivable_support_0"] = {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.stack(
            [np.arange(0, 120, 5).astype(float), ys, np.zeros(len(ys))], axis=1),
    }
    return m


def test_synthesised_support_lanes_are_not_lanes():
    """Measured on 4528c271d89c53e1: 4 support polygons, nearest for 155/190
    frames, and `lateral_offset_m` never left +/-0.26 m -- which makes a lane
    change undetectable and would withhold C-3's goal from every policy."""
    lc = lane_centerlines(_map_with_support_lane())
    assert not [k for k in lc if k.startswith("__")]
    assert set(lc) == {"L1", "L2"}


def test_a_lane_change_survives_a_support_lane_on_the_ego_path():
    """The regression: with the support lane included, this returns []."""
    ys = np.clip((np.arange(60.0) - 30) * 0.5, 0, LANE_W)
    frames = _frames(ys, lane_centerlines(_map_with_support_lane()))
    assert SR.has_lane_data(frames)
    assert len(SR.lane_change_frames(frames)) == 1
