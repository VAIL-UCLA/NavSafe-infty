# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``--traffic-takeover`` must reach the traffic manager, not just parse.

The flag, the ``EnvCfg`` field and the manager that reads it all existed while
``build_eval_env_cfg`` quietly failed to connect them, so
``--traffic-takeover spawn`` (and ``NAVSAFE_TAKEOVER=spawn``, which
``run_bundle_eval.sh`` documents as the way to reproduce cells recorded before
2026-08-11) parsed cleanly and ran continuous-takeover traffic anyway. A
silently ignored knob is worse than a rejected one: every run looks fine and
the traffic is not what was asked for.

These tests pin the wiring end of it. The takeover *rules* are covered by
tests/test_semi_reactive_traffic.py.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from navsafe.evaluation.eval_env_config import _episode_cap, build_eval_env_cfg


def _py123d_args(**kw) -> Namespace:
    base = dict(
        scenario_source="py123d",
        py123d_data_root="data",
        py123d_scene_index=0,
        render_backend="nurec_grpc",
        sim_dt=0.1,
        execution_mode="teleport",
        no_urbanverse=False,
        asset_path="assets/urbanverse_sdk",
        contact_dynamics=None,
        traffic_mode="semi_reactive",
    )
    base.update(kw)
    return Namespace(**base)


@pytest.mark.parametrize("takeover", ["spawn", "continuous"])
def test_py123d_forwards_traffic_takeover(takeover: str) -> None:
    cfg = build_eval_env_cfg(_py123d_args(traffic_takeover=takeover))
    assert cfg.semi_reactive_takeover == takeover


def test_py123d_forwards_instance_window_and_eval_seed() -> None:
    cfg = build_eval_env_cfg(_py123d_args(py123d_frame_window=[300, 450], eval_seed=123))
    assert cfg.py123d_frame_window == (300, 450)
    assert cfg.seed == 123


def test_py123d_without_window_preserves_full_log() -> None:
    assert build_eval_env_cfg(_py123d_args()).py123d_frame_window is None


def test_default_is_continuous_when_the_caller_sets_nothing() -> None:
    # An older caller with no such attribute must keep the behaviour it had,
    # which is the EnvCfg default rather than an error.
    cfg = build_eval_env_cfg(_py123d_args())
    assert cfg.semi_reactive_takeover == "continuous"


def test_the_manager_accepts_exactly_what_the_config_can_carry() -> None:
    # The two ends agree on the vocabulary: anything build_eval_env_cfg can
    # produce, SemiReactiveTraffic must take, and nothing else.
    from navsafe.traffic.semi_reactive import SemiReactiveTraffic

    for takeover in ("spawn", "continuous"):
        assert SemiReactiveTraffic(takeover=takeover).takeover == takeover
    with pytest.raises(ValueError, match="takeover must be"):
        SemiReactiveTraffic(takeover="sometimes")


def test_unlimited_route_clock_does_not_inherit_the_sixty_second_env_cap() -> None:
    args = _py123d_args(
        ego_replay_frames=20, eval_frames=None, route_time_limit_s=0.0)
    assert _episode_cap(args) == 2**31 - 1


def test_finite_route_clock_and_frame_window_still_bound_the_env() -> None:
    finite = _py123d_args(
        ego_replay_frames=20, eval_frames=None, route_time_limit_s=30.0)
    window = _py123d_args(
        ego_replay_frames=20, eval_frames=180, route_time_limit_s=0.0)
    assert _episode_cap(finite) == 320
    assert _episode_cap(window) == 200
