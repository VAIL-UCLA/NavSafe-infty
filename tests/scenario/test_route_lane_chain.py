# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Traversed-lane chain derivation (``metadata['route_lane_ids']``).

The chain is the branch-intent analog of nuPlan's
``route_roadblock_ids``, reconstructed from the log trajectory because
AV2 ships no route field. The failure modes worth guarding sit at
intersections: overlapping lanes make bare nearest-lane projection
jitter (A-B-A flicker), and crossing lanes can be the *nearest* lane
while the ego drives over them.
"""

from __future__ import annotations

import numpy as np

from navsafe.evaluation.utils.lane_proxy import build_lanes_from_scenario
from navsafe.scenario.route_lane_chain import (
    _collapse_shared_origin_alternatives,
    derive_route_lane_ids,
)


def _lane(points_xy: np.ndarray) -> dict:
    pts = np.asarray(points_xy, dtype=np.float64)
    return {
        "type": "LANE_SURFACE_STREET",
        "polyline": np.column_stack([pts, np.zeros(len(pts))]),
        "entry_lanes": [],
        "exit_lanes": [],
    }


def _straight(x0: float, x1: float, y: float) -> dict:
    n = max(2, int(abs(x1 - x0)) + 1)
    xs = np.linspace(x0, x1, n)
    return _lane(np.column_stack([xs, np.full(n, y)]))


def _scenario(map_features: dict, positions: np.ndarray,
              headings: np.ndarray, scenario_id: str | None = None) -> dict:
    n = len(positions)
    metadata: dict = {"sdc_id": "sdc",
                      "ts": np.arange(n, dtype=np.float64) * 0.1}
    if scenario_id is not None:
        # Converter-stamped identity: enables the memoization path.
        metadata.update({"scenario_id": scenario_id,
                         "dataset": "av2", "split": "train"})
    return {
        "metadata": metadata,
        "length": n,
        "tracks": {
            "sdc": {
                "type": "VEHICLE",
                "state": {
                    "position": np.column_stack(
                        [positions, np.zeros(n)]),
                    "heading": headings,
                    "valid": np.ones(n, dtype=bool),
                },
            },
        },
        "map_features": map_features,
        "dynamic_map_states": {},
    }


def test_chain_follows_log_through_fork() -> None:
    """Log drives main road then the left turn: chain records the turn."""
    theta = np.linspace(0.0, np.pi / 2.0, 20)
    turn = np.column_stack([100.0 + 50.0 * np.sin(theta),
                            50.0 * (1.0 - np.cos(theta))])
    features = {
        "lane_main": _straight(0.0, 100.0, 0.0),
        "lane_straight": _straight(100.0, 200.0, 0.0),
        "lane_turn": _lane(turn),
    }
    # Log ego: along lane_main, then along the turn arc.
    main_xy = np.column_stack([np.linspace(0.0, 99.0, 50), np.zeros(50)])
    main_h = np.zeros(50)
    turn_h = theta + 1e-3  # tangent of the arc at each sample
    positions = np.concatenate([main_xy, turn], axis=0)
    headings = np.concatenate([main_h, turn_h])
    chain = derive_route_lane_ids(_scenario(features, positions, headings))
    assert chain == ["lane_main", "lane_turn"], (
        f"chain must record the branch the log actually took, got {chain}"
    )


def test_dwell_suppresses_projection_flicker() -> None:
    """Single-frame blips onto an overlapping lane never enter the chain."""
    features = {
        "lane_a": _straight(0.0, 100.0, 0.0),
        "lane_b": _straight(0.0, 100.0, 0.6),
    }
    n = 40
    xs = np.linspace(0.0, 99.0, n)
    ys = np.full(n, 0.1)     # nearest = lane_a...
    ys[10] = 0.5             # ...except single-frame blips toward lane_b
    ys[25] = 0.5
    chain = derive_route_lane_ids(
        _scenario(features, np.column_stack([xs, ys]), np.zeros(n))
    )
    assert chain == ["lane_a"], (
        f"one-frame nearest-lane flicker must not enter the chain, got {chain}"
    )


def test_crossing_lane_never_wins() -> None:
    """A perpendicular lane at an intersection is nearest but not driven."""
    cross = np.column_stack([np.full(41, 50.0), np.linspace(-20.0, 20.0, 41)])
    features = {
        "lane_a": _straight(0.0, 100.0, 0.0),
        "lane_cross": _lane(cross),
    }
    n = 50
    xs = np.linspace(0.0, 99.0, n)
    chain = derive_route_lane_ids(
        _scenario(features, np.column_stack([xs, np.zeros(n)]), np.zeros(n))
    )
    assert chain == ["lane_a"], (
        f"the heading gate must exclude the crossing lane, got {chain}"
    )


def test_shared_origin_branch_oscillation_collapses_to_final_winner() -> None:
    """Persistent A-B-A-B votes at one fork must not become a route loop."""
    features = {
        "upstream": _straight(-50.0, 0.0, 0.0),
        "straight_arm": _straight(0.0, 50.0, 0.0),
        "turn_arm": _lane(np.array([[0.0, 0.0], [5.0, 0.0],
                                     [10.0, -2.0], [15.0, -7.0]])),
        "downstream": _straight(15.0, 60.0, -7.0),
    }
    scenario = _scenario(features, np.array([[0.0, 0.0], [1.0, 0.0]]),
                         np.zeros(2))
    lanes = [lane for lane, _polygon in build_lanes_from_scenario(scenario)]
    chain = _collapse_shared_origin_alternatives(
        ["upstream", "straight_arm", "turn_arm", "straight_arm",
         "turn_arm", "downstream"],
        lanes,
    )
    assert chain == ["upstream", "turn_arm", "downstream"]


def test_real_loop_with_distinct_lane_origins_is_preserved() -> None:
    """A-B-A is meaningful when B is a spatially distinct detour."""
    features = {
        "lane_a": _straight(0.0, 100.0, 0.0),
        "lane_b": _lane(np.array([[100.0, 0.0], [120.0, 20.0],
                                   [80.0, 20.0], [50.0, 0.0]])),
    }
    scenario = _scenario(features, np.array([[0.0, 0.0], [1.0, 0.0]]),
                         np.zeros(2))
    lanes = [lane for lane, _polygon in build_lanes_from_scenario(scenario)]
    assert _collapse_shared_origin_alternatives(
        ["lane_a", "lane_b", "lane_a"], lanes
    ) == ["lane_a", "lane_b", "lane_a"]


def test_empty_inputs_yield_empty_chain() -> None:
    chain = derive_route_lane_ids({"metadata": {}, "tracks": {},
                                   "map_features": {}})
    assert chain == []


def _counting_derivation(monkeypatch):
    """Route the module through a call-counting wrapper of the real impl."""
    import navsafe.scenario.route_lane_chain as rlc

    calls = {"n": 0}
    real = rlc._derive_route_lane_ids_uncached

    def counting(scenario_data):
        calls["n"] += 1
        return real(scenario_data)

    monkeypatch.setattr(rlc, "_derive_route_lane_ids_uncached", counting)
    rlc._route_chain_cache.clear()
    return calls


def test_cache_reuses_result_for_same_scene(monkeypatch) -> None:
    """Reconverting the same scene (one per env reset) derives only once."""
    calls = _counting_derivation(monkeypatch)
    features = {"lane_a": _straight(0.0, 100.0, 0.0)}
    n = 40
    xy = np.column_stack([np.linspace(0.0, 99.0, n), np.zeros(n)])

    first = derive_route_lane_ids(
        _scenario(features, xy, np.zeros(n), scenario_id="scene-001"))
    second = derive_route_lane_ids(
        _scenario(features, xy, np.zeros(n), scenario_id="scene-001"))
    assert calls["n"] == 1, "second conversion of the same scene must hit"
    assert first == second == ["lane_a"]

    # A cached hit hands back a copy: mutating it must not poison the cache.
    second.append("garbage")
    third = derive_route_lane_ids(
        _scenario(features, xy, np.zeros(n), scenario_id="scene-001"))
    assert third == ["lane_a"]


def test_cache_misses_for_different_scene(monkeypatch) -> None:
    calls = _counting_derivation(monkeypatch)
    features = {"lane_a": _straight(0.0, 100.0, 0.0)}
    n = 40
    xy = np.column_stack([np.linspace(0.0, 99.0, n), np.zeros(n)])

    derive_route_lane_ids(
        _scenario(features, xy, np.zeros(n), scenario_id="scene-001"))
    derive_route_lane_ids(
        _scenario(features, xy, np.zeros(n), scenario_id="scene-002"))
    assert calls["n"] == 2, "a different scenario_id must be a cache miss"


def test_no_scenario_id_bypasses_cache(monkeypatch) -> None:
    """Hand-built dicts without converter identity are never cached."""
    calls = _counting_derivation(monkeypatch)
    features = {"lane_a": _straight(0.0, 100.0, 0.0)}
    n = 40
    xy = np.column_stack([np.linspace(0.0, 99.0, n), np.zeros(n)])

    derive_route_lane_ids(_scenario(features, xy, np.zeros(n)))
    derive_route_lane_ids(_scenario(features, xy, np.zeros(n)))
    assert calls["n"] == 2, "no stable identity -> derive every time"
