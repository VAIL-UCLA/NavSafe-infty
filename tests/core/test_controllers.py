# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Closed-loop tests for the trajectory-tracking controllers (CPU, hermetic).

Each tracker drives the real ``EgoDynamics`` bicycle model against a
reference path and must converge onto it — the acceptance behavior for
controller-in-the-loop execution mode.
"""

from __future__ import annotations

import numpy as np
import pytest

# controllers.py itself is pure numpy, but importing it triggers
# navsafe.policy.state's package __init__, which eagerly imports the
# torch-dependent ego_mlp adapter — the same chain that heavy-gates the
# pdm_closed test suite. Skip on light (CI) environments.
pytest.importorskip("torch")

from navsafe.core.controllers import (
    LQRTracker,
    PurePursuitTracker,
    _speed_control,
    create_tracker,
)
from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg


def _drive(tracker, path, *, start=(0.0, -1.0, 0.0, 3.0), target_speed=3.0,
           steps=60, dt=0.1):
    """Closed loop: tracker + bicycle model. Returns final lateral error and speed."""
    ego = EgoDynamics(EgoDynamicsCfg(dt=dt))
    ego.reset(x=start[0], y=start[1], heading=start[2], speed=start[3])
    tracker.reset()
    for _ in range(steps):
        state = ego.get_state()
        steer, accel = tracker.compute(state, path, target_speed)
        assert -1.0 <= steer <= 1.0 and -1.0 <= accel <= 1.0
        ego.step(steer, accel)
    # Lateral error = distance to the nearest path vertex (0.1 m spacing).
    dists = np.linalg.norm(path - np.array([ego.x, ego.y]), axis=1)
    return float(dists.min()), float(ego.speed)


def _straight_path(length=60.0, spacing=0.1):
    xs = np.arange(0.0, length, spacing)
    return np.stack([xs, np.zeros_like(xs)], axis=1)


def _curved_path(radius=25.0, spacing=0.1):
    # Quarter circle turning left, starting along +x.
    s = np.arange(0.0, 0.5 * np.pi * radius, spacing)
    theta = s / radius
    return np.stack([radius * np.sin(theta), radius * (1 - np.cos(theta))], axis=1)


@pytest.mark.parametrize("tracker_cls", [PurePursuitTracker, LQRTracker])
class TestTrackersConverge:
    def test_straight_path_convergence(self, tracker_cls) -> None:
        # Start 1 m right of the lane and re-center. Pure pursuit converges
        # within 3 s; the LQR's gains (Q favors heading error, tuned for PDM
        # proposal simulation) converge monotonically but slower — give it 10 s.
        steps = 30 if tracker_cls is PurePursuitTracker else 100
        lat_err, speed = _drive(tracker_cls(), _straight_path(),
                                start=(0.0, -1.0, 0.0, 3.0), steps=steps)
        assert lat_err < 0.3
        assert speed == pytest.approx(3.0, rel=0.1)

    def test_curved_path_convergence(self, tracker_cls) -> None:
        lat_err, speed = _drive(tracker_cls(), _curved_path(),
                                start=(0.0, 0.0, 0.0, 3.0), steps=60)
        assert lat_err < 0.5  # sustained curve: small steady-state lag allowed
        assert speed == pytest.approx(3.0, rel=0.1)

    def test_stationary_target_brakes(self, tracker_cls) -> None:
        # The official longitudinal LQR approaches the reference smoothly;
        # it needs a little longer than the legacy saturated speed P loop.
        steps = 40 if tracker_cls is LQRTracker else 30
        _, speed = _drive(tracker_cls(), _straight_path(),
                          start=(0.0, 0.0, 0.0, 5.0), target_speed=0.0,
                          steps=steps)
        assert speed < 0.2

    def test_speed_ramps_up_from_stop(self, tracker_cls) -> None:
        _, speed = _drive(tracker_cls(), _straight_path(),
                          start=(0.0, 0.0, 0.0, 0.0), target_speed=4.0, steps=50)
        assert speed == pytest.approx(4.0, rel=0.15)


class TestLQRSteadyStateBias:
    """The LQR's measurement and its A-matrix gain are a matched pair.

    ``_lqr_steer`` linearises around the *course* heading
    ``theta + atan(0.5 * tan(delta))`` and uses the symmetric-bicycle gain
    ``v/L`` — both facts about :class:`EgoDynamics`. Change one
    without the other and the slip angle becomes an un-modelled
    disturbance, which on a sustained curve integrates into a constant
    lateral offset rather than a decaying transient.

    :class:`TestTrackersConverge` drives only 6 s of a quarter circle,
    so it sees the transient and the bias mixed together. These drive a
    half circle for 12 s and read the *settled* error, which is the
    quantity the mismatch actually moves: measured 1.459 m (3 m/s) and
    1.257 m (5 m/s) with a body-heading measurement against the
    symmetric-bicycle gain, versus small error when the pair matches.
    """

    @staticmethod
    def _settled_lateral_error(speed: float, radius: float = 25.0,
                               steps: int = 120, dt: float = 0.1) -> float:
        s = np.arange(0.0, np.pi * radius, 0.1)
        path = np.stack(
            [radius * np.sin(s / radius), radius * (1.0 - np.cos(s / radius))],
            axis=1,
        )
        # The ego must not run off the end of the path, or "distance to
        # the nearest vertex" stops measuring tracking error.
        assert speed * steps * dt < np.pi * radius
        ego = EgoDynamics(EgoDynamicsCfg(dt=dt))
        ego.reset(x=0.0, y=0.0, heading=0.0, speed=speed)
        tracker = LQRTracker()
        tracker.reset()
        errors = []
        for _ in range(steps):
            steer, accel = tracker.compute(ego.get_state(), path, speed)
            ego.step(steer, accel)
            errors.append(
                float(np.linalg.norm(path - np.array([ego.x, ego.y]),
                                     axis=1).min())
            )
        # Settled = the last third of the drive, well past the transient.
        return float(np.max(errors[-steps // 3:]))

    @pytest.mark.parametrize("speed", [3.0, 5.0])
    def test_sustained_curve_has_no_steady_state_offset(self, speed) -> None:
        assert self._settled_lateral_error(speed) < 0.30


@pytest.mark.parametrize("tracker_cls", [PurePursuitTracker, LQRTracker])
class TestDegeneratePlans:
    def test_single_point_plan_does_not_crash(self, tracker_cls) -> None:
        tracker = tracker_cls()
        state = {"position": np.zeros(3), "heading": 0.0, "speed": 3.0}
        steer, accel = tracker.compute(
            state, np.array([[1.0, 0.0]]), target_speed=0.0)
        assert steer == 0.0
        assert accel == -1.0  # stop plan → brake

    def test_empty_plan_does_not_crash(self, tracker_cls) -> None:
        tracker = tracker_cls()
        state = {"position": np.zeros(3), "heading": 0.0, "speed": 1.0}
        steer, accel = tracker.compute(
            state, np.zeros((0, 2)), target_speed=0.0)
        assert steer == 0.0 and accel == -1.0


class TestSpeedControl:
    def test_brake_on_near_zero_target(self) -> None:
        assert _speed_control(3.0, 0.2) == -1.0

    def test_positive_from_rest_ramp_is_not_converted_to_a_stop(self) -> None:
        assert _speed_control(0.0, 0.05) > 0.0
        assert _speed_control(0.03, 0.05) > 0.0
        assert _speed_control(0.10, 0.05) == -1.0
        assert _speed_control(0.0, 0.0) == -1.0

    def test_positive_tight_turn_crawl_is_not_converted_to_a_stop(self) -> None:
        assert _speed_control(0.0, 0.29) > 0.0
        assert -1.0 < _speed_control(0.4, 0.29) < 0.0

    def test_brake_on_large_overshoot(self) -> None:
        assert _speed_control(5.0, 2.0) == -1.0

    def test_moderate_speed_reduction_is_not_ignored(self) -> None:
        # NexusSim's bicycle has no passive drag. Returning zero here (the
        # historical BridgeSim copy) leaves 6.12 m/s unchanged forever even
        # when the policy/harness requests 4.5 m/s for an upcoming turn.
        action = _speed_control(6.12, 4.5)
        assert -1.0 < action < 0.0
        assert action == pytest.approx(-0.81)

    @pytest.mark.parametrize("tracker_cls", [PurePursuitTracker, LQRTracker])
    def test_tracker_converges_down_to_nonzero_target(self, tracker_cls) -> None:
        steps = 25 if tracker_cls is LQRTracker else 20
        _, speed = _drive(
            tracker_cls(), _straight_path(),
            start=(0.0, 0.0, 0.0, 6.12), target_speed=4.5, steps=steps)
        assert speed == pytest.approx(4.5, abs=0.15)

    def test_throttle_saturates(self) -> None:
        assert _speed_control(0.0, 10.0) == 1.0

    def test_zero_error_zero_accel(self) -> None:
        assert _speed_control(3.0, 3.0) == 0.0


class TestFactory:
    def test_known_types(self) -> None:
        assert isinstance(create_tracker("pure_pursuit"), PurePursuitTracker)
        assert isinstance(create_tracker("pid"), PurePursuitTracker)
        assert isinstance(create_tracker("lqr"), LQRTracker)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="controller_type"):
            create_tracker("teleport")
