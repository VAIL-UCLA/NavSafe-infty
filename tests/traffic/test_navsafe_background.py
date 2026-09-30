# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Who drives what, under ``--traffic-mode navsafe``.

The split is by ROLE and the two halves must not overlap: the recipe's actors
run the policies it authored, everything else is reactive background. Leaving
the background on the log is what let a replayed car rear-end a closed-loop ego
that slowed for the event — six runs on one host ended that way, at 2.6 s,
before the scenario had been tested at all.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from navsafe.traffic.navsafe import NavSafeTraffic


class _FakeBackground:
    """Stands in for SemiReactiveTraffic: records what it was told and asked."""

    def __init__(self) -> None:
        self.exclude_ids: set = set()
        self.pose_overrides: Dict[Any, Dict[str, Any]] = {}
        self.reset_calls = 0
        self.step_calls = 0
        self.seen_excludes: list = []

    def reset(self, env: Any) -> None:
        self.reset_calls += 1
        self.seen_excludes.append(set(self.exclude_ids))

    def step(self, env: Any, dt: float) -> None:
        self.step_calls += 1
        self.seen_excludes.append(set(self.exclude_ids))


def _spec(name: str, kind: str = "static", **spawn):
    return {
        "name": name,
        "track_id": f"navsafe_{name}",
        "policy": {"kind": kind},
        "spawn": {"position": (0.0, 0.0, 0.0), "heading": 0.0, **spawn},
    }


@pytest.fixture
def manager():
    bg = _FakeBackground()
    tm = NavSafeTraffic([_spec("blocker_1"), _spec("blocker_2")], background=bg)
    tm.reset(env=None)
    return tm, bg


def test_the_recipe_s_actors_are_driven_here(manager):
    tm, _ = manager
    assert set(tm.drivers) == {"navsafe_blocker_1", "navsafe_blocker_2"}


def test_the_background_is_reset_and_stepped_with_the_ego(manager):
    tm, bg = manager
    assert bg.reset_calls == 1
    tm.step(env=None, dt=0.1)
    assert bg.step_calls == 1


def test_the_background_may_not_adopt_the_recipe_s_actors(manager):
    # A recipe's inserted car is a track like any other, so the geometric
    # takeover test would drive the actor the scenario is ABOUT as background.
    tm, bg = manager
    tm.step(env=None, dt=0.1)
    assert bg.exclude_ids == {"navsafe_blocker_1", "navsafe_blocker_2"}
    assert all(e == {"navsafe_blocker_1", "navsafe_blocker_2"} for e in bg.seen_excludes)


def test_both_halves_reach_the_metrics(manager):
    tm, bg = manager
    bg.pose_overrides = {"logged_car": {"position": (1.0, 2.0, 0.0), "heading": 0.0}}
    tm.step(env=None, dt=0.1)
    # The env merges `pose_overrides` over logged track state; a background
    # vehicle absent from it is scored where the log put it, not where IDM
    # drove it.
    assert "logged_car" in tm.pose_overrides
    assert {"navsafe_blocker_1", "navsafe_blocker_2"} <= set(tm.pose_overrides)


def test_the_recipe_wins_a_collision_of_ids(manager):
    # Cannot happen while exclude_ids holds, which is the point: if the
    # invariant ever breaks, the leaf's own actor must not be overwritten.
    tm, bg = manager
    bg.pose_overrides = {"navsafe_blocker_1": {"position": (99.0, 99.0, 0.0),
                                               "heading": 0.0}}
    tm.step(env=None, dt=0.1)
    assert tm.pose_overrides["navsafe_blocker_1"]["position"][0] != 99.0


def test_a_mined_recipe_still_gets_reactive_background():
    # Zero actors is the mined case. StageFreeTraffic.step returns early with
    # no drivers, so the background has to be stepped independently of it.
    bg = _FakeBackground()
    tm = NavSafeTraffic([], background=bg)
    tm.reset(env=None)
    bg.pose_overrides = {"logged_car": {"position": (3.0, 4.0, 0.0), "heading": 0.0}}
    tm.step(env=None, dt=0.1)
    assert bg.step_calls == 1
    assert tm.pose_overrides == bg.pose_overrides


def test_background_can_be_switched_off_explicitly():
    tm = NavSafeTraffic([_spec("a")], background=False)
    tm.reset(env=None)
    tm.step(env=None, dt=0.1)          # must not raise looking for a delegate
    assert tm._background is None
