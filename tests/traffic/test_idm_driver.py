# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""What an IDM-driven recipe actor must and must not do.

Two behaviours carry the leaves that use this driver, and they pull in
opposite directions:

* it **yields** — an inserted car in the ego's lane is background traffic and
  must not rear-end whatever is in front of it;
* when the leaf's whole point is the consequence of the ego's own mistake, it
  **does not yield to the ego** — because IDM is a safe following model, and a
  safe model handed the ego as a leader simply brakes, deleting the event the
  leaf is named after.

Both are pinned here, because either one silently regressing produces a
scenario that still runs, still renders, and no longer measures anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navsafe.traffic.idm_driver import IDMDriver
from navsafe.traffic.stage_free import WorldView

DT = 0.1
#: A straight 200 m road along +x.
STRAIGHT = [[float(x), 0.0] for x in range(0, 201, 5)]
#: Long enough that a full acceleration transient fits on it.
LONG_STRAIGHT = [[float(x), 0.0] for x in range(0, 2001, 5)]


def _driver(**policy) -> IDMDriver:
    policy.setdefault("path_polyline", STRAIGHT)
    return IDMDriver("actor", {"position": (0.0, 0.0, 0.0)},
                     policy=policy, v0=10.0)


def _world(*agents, ego=None, t=0):
    return WorldView(t=t, dt=DT, ego=ego or {"position": np.zeros(3), "heading": 0.0,
                                             "velocity": np.zeros(2)},
                     agents=list(agents), ego_id="ego")


def _agent(agent_id, x, y=0.0, speed=0.0, heading=0.0):
    return {"id": agent_id, "position": np.array([x, y, 0.0]),
            "heading": heading, "length": 4.5, "width": 1.8,
            "velocity": np.array([speed * math.cos(heading), speed * math.sin(heading)])}


def _run(driver, world_fn, steps):
    for k in range(steps):
        driver.step(world_fn(k), DT)
    return driver


class TestFreeRoad:
    def test_it_accelerates_toward_the_desired_speed(self):
        # a_max is 1.0 m/s^2 and IDM tapers as v approaches v0, so reaching
        # 10 m/s takes tens of seconds, not a few. The road has to be long
        # enough to hold the transient or the actor runs out of road first.
        d = _driver(path_polyline=LONG_STRAIGHT)
        d.v = 0.0
        _run(d, lambda k: _world(), 400)
        assert d.v == pytest.approx(10.0, abs=0.5)

    def test_it_never_exceeds_its_desired_speed(self):
        d = _driver(path_polyline=LONG_STRAIGHT)
        d.v = 0.0
        for k in range(400):
            d.step(_world(), DT)
            assert d.v <= 10.0 + 1e-6

    def test_it_spawns_at_cruise_not_at_rest(self):
        # An actor that has been driving down this road already is at speed.
        # Starting every one from zero turns "a car is coming" into "a car
        # pulls away from the kerb".
        assert _driver().v == pytest.approx(10.0)

    def test_it_stops_at_the_end_of_its_road(self):
        # Extrapolating past the reconstruction's edge is a rendering
        # artefact, not a scenario.
        d = _run(_driver(), lambda k: _world(), 400)
        assert d.s == pytest.approx(d.route.length)
        assert d.v == 0.0


class TestYielding:
    def test_it_slows_for_a_stopped_car_in_its_lane(self):
        d = _driver()
        _run(d, lambda k: _world(_agent("blocker", 40.0)), 40)
        assert d.v < 6.0, "IDM did not react to a stationary leader"

    def test_it_does_not_drive_through_the_leader(self):
        d = _driver()
        _run(d, lambda k: _world(_agent("blocker", 40.0)), 120)
        gap = 40.0 - d.route.position_at(d.s)[0]
        assert gap > 0.0, f"drove past the leader (gap {gap:.2f} m)"

    def test_an_object_outside_the_corridor_is_not_a_leader(self):
        # Being nearby is not being in my lane. Without the corridor test a
        # car on the next carriageway would brake this one to a halt.
        d = _driver()
        _run(d, lambda k: _world(_agent("beside", 30.0, y=6.0)), 40)
        assert d.v == pytest.approx(10.0, abs=0.5)


class TestBlindToEgo:
    """C-10 / C-7: the oncoming car is not expecting a wrong-way ego."""

    def _closing_ego(self, k):
        # Ego sits square in the actor's lane, 50 m ahead, closing.
        return _world(ego={"position": np.array([50.0 - k * 0.5, 0.0, 0.0]),
                           "heading": math.pi, "velocity": np.array([-5.0, 0.0]),
                           "length": 4.5, "width": 1.8})

    def test_a_sighted_driver_yields_to_the_ego(self):
        d = _driver(blind_to_ego=False)
        _run(d, self._closing_ego, 40)
        assert d.v < 9.0, "an ordinary IDM actor should slow for a car in its lane"

    def test_a_blind_driver_holds_its_speed(self):
        # This is the leaf. If this ever starts failing, C-10 will still run,
        # still render, and quietly stop being a wrong-way CRASH scenario.
        d = _driver(blind_to_ego=True)
        _run(d, self._closing_ego, 40)
        assert d.v == pytest.approx(10.0, abs=0.5)

    def test_blindness_is_only_about_the_ego(self):
        # It still behaves like traffic towards everything else.
        d = _driver(blind_to_ego=True)
        _run(d, lambda k: _world(_agent("other", 40.0), ego={"position": np.zeros(3)}), 40)
        assert d.v < 6.0


class TestContract:
    def test_it_refuses_a_policy_with_no_resolved_path(self):
        # Naming 'opposing_lane_chain' is a query against one host's map;
        # the resolved polyline is what makes the recipe rebuildable.
        with pytest.raises(ValueError, match="path_polyline"):
            IDMDriver("a", {"position": (0.0, 0.0, 0.0)}, policy={})

    def test_pose_carries_velocity_for_ttc(self):
        # TTC projects agents forward at constant velocity; without this the
        # actor looks parked and TTC scores a clean 1.0 against moving traffic.
        pose = _driver().pose()
        assert pose["velocity"][0] == pytest.approx(10.0, abs=0.1)

    def test_it_is_registered_under_its_policy_name(self):
        from navsafe.traffic.navsafe import DRIVERS
        import navsafe.traffic.idm_driver  # noqa: F401

        assert DRIVERS["idm"] is IDMDriver

    def test_two_identical_runs_agree_bit_for_bit(self):
        # No RNG anywhere in IDM: same recipe and same ego behaviour must give
        # the same rollout, or a frozen scenario means nothing.
        a = _run(_driver(), lambda k: _world(_agent("b", 40.0)), 50)
        b = _run(_driver(), lambda k: _world(_agent("b", 40.0)), 50)
        assert a.s == b.s and a.v == b.v
