# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Guards for ``navsafe.core.steering`` (the core→policy cycle fix).

The path followers ``_project_point_on_path`` / ``_pure_pursuit_steer_norm`` /
``_lqr_steer`` moved verbatim from
``navsafe/policy/state/pdm_closed_planner/forward_sim.py`` into
``navsafe/core/steering.py`` so ``core.controllers`` no longer imports policy
code. Two invariants keep that refactor honest:

1. ``LateralLQRConfig`` (core) and ``PDMConfig`` (policy) carry the SAME LQR
   tuning defaults — ``LQRTracker`` used to build a default ``PDMConfig``, so
   any drift between the two default sets silently changes tracker behaviour.
2. ``forward_sim`` re-exports the very same function objects, so the
   faithfulness pins in ``tests/policy/pdm_closed/test_faithfulness.py`` and
   PDM-Closed's own calls keep hitting one implementation, not a fork.
"""

from __future__ import annotations

import dataclasses

import pytest

from navsafe.core.steering import LateralLQRConfig


def test_lqr_defaults_match_pdm_config() -> None:
    """LateralLQRConfig's LQR defaults are pinned to PDMConfig's.

    LQRTracker previously read these values off a default-constructed
    ``PDMConfig``; the core-side dataclass must not drift from it.
    """
    # ``navsafe.policy.state`` eagerly imports the torch-backed ego_mlp
    # adapter, so even the planner's pure-dataclass config module is not
    # importable on the stock CI runner — this drift pin therefore only
    # runs on GPU machines (the documented cadence for torch-gated tests).
    pytest.importorskip("torch", reason="pdm_closed_planner.config imports the torch-backed planner package")
    from navsafe.policy.state.pdm_closed_planner.config import PDMConfig

    # The execution tracker's lateral-position weight was retuned on the 270
    # navsafe-loop bundles (2026-09-02, see the field comment in steering.py);
    # PDMConfig keeps the faithful tuplan_garage value for the planner's
    # internal forward simulation. Pin both sides of that divergence so a
    # change to either is a deliberate edit here, not drift.
    deliberate = {"lqr_q_lateral": ((10.0, 10.0, 0.0), (1.0, 10.0, 0.0))}

    pdm_defaults = {f.name: f.default for f in dataclasses.fields(PDMConfig)}
    for f in dataclasses.fields(LateralLQRConfig):
        if not f.name.startswith("lqr_"):
            continue  # sim_dt/wheelbase/max_steering_angle_rad are ctor args
        assert f.name in pdm_defaults, (
            f"LateralLQRConfig.{f.name} has no PDMConfig counterpart"
        )
        if f.name in deliberate:
            core_expected, pdm_expected = deliberate[f.name]
            assert f.default == core_expected, (
                f"LateralLQRConfig.{f.name} default {f.default!r} != the "
                f"documented retune {core_expected!r}")
            assert pdm_defaults[f.name] == pdm_expected, (
                f"PDMConfig.{f.name} default {pdm_defaults[f.name]!r} != the "
                f"faithful tuplan_garage value {pdm_expected!r}")
            continue
        assert f.default == pdm_defaults[f.name], (
            f"LateralLQRConfig.{f.name} default {f.default!r} drifted from "
            f"PDMConfig.{f.name} default {pdm_defaults[f.name]!r} — these are "
            f"a pinned pair (see navsafe/core/steering.py)."
        )


def test_forward_sim_reexports_same_objects() -> None:
    """forward_sim's names are the core functions, not a diverged copy."""
    pytest.importorskip("torch", reason="pdm_closed_planner imports the torch-backed planner package")
    from navsafe.core import steering
    from navsafe.policy.state.pdm_closed_planner import forward_sim

    assert forward_sim._lqr_steer is steering._lqr_steer
    assert forward_sim._pure_pursuit_steer_norm is steering._pure_pursuit_steer_norm
    assert forward_sim._project_point_on_path is steering._project_point_on_path
