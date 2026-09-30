"""
Tests for the harness-parity patches on ``Evaluator``:

1. ``_invoke_adapter_perceive`` is called exactly once per frame with the
   current frame id.
2. ``_enrich_ego_state`` adds ``acceleration`` + ``angular_velocity`` with
   the right values on frame 0 and on subsequent frames (finite-diff).
3. PDM-Closed's ``_score_extended_comfort`` no longer KeyErrors on an
   ego_state dict missing the new keys.

The tests do *not* import ``isaaclab`` — they invoke the Evaluator helpers
as unbound methods against a lightweight stub so they run on Apple Silicon.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/<subsystem>/<file> -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ──────────────────────────────────────────────────────────────────────
# _invoke_adapter_perceive
# ──────────────────────────────────────────────────────────────────────

def _stub_with_env_and_adapter():
    """Build a SimpleNamespace that has the attributes the helpers read."""
    from navsafe.evaluation.evaluator import Evaluator

    stub = SimpleNamespace()
    stub.env = SimpleNamespace()
    stub.adapter = SimpleNamespace()
    stub.scenario_id = "test_scenario"
    stub.config = SimpleNamespace(sim_dt=0.1)
    stub._prev_ego_velocity = None
    stub._prev_ego_heading = None
    return stub, Evaluator


def test_invoke_adapter_perceive_calls_each_frame():
    stub, Evaluator = _stub_with_env_and_adapter()
    calls = []
    stub.adapter.perceive = lambda env, frame_id: calls.append((env, int(frame_id)))

    Evaluator._invoke_adapter_perceive(stub, 0)
    Evaluator._invoke_adapter_perceive(stub, 1)
    Evaluator._invoke_adapter_perceive(stub, 7)

    assert len(calls) == 3
    envs, frames = zip(*calls)
    assert frames == (0, 1, 7)
    assert all(e is stub.env for e in envs)


def test_invoke_adapter_perceive_is_noop_when_adapter_has_no_hook():
    stub, Evaluator = _stub_with_env_and_adapter()
    # No .perceive attribute on the adapter → should do nothing silently.
    Evaluator._invoke_adapter_perceive(stub, 0)


def test_invoke_adapter_perceive_swallows_exceptions():
    stub, Evaluator = _stub_with_env_and_adapter()

    def boom(env, frame_id):
        raise RuntimeError("camera pipe stalled")

    stub.adapter.perceive = boom
    # Must not propagate; evaluator logs a warning and continues.
    Evaluator._invoke_adapter_perceive(stub, 0)


# ──────────────────────────────────────────────────────────────────────
# _enrich_ego_state
# ──────────────────────────────────────────────────────────────────────

def _base_ego_state(vx=1.0, vy=0.0, heading=0.0):
    return {
        "position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "heading": float(heading),
        "speed": float(np.sqrt(vx * vx + vy * vy)),
        "velocity": np.array([vx, vy, 0.0], dtype=np.float32),
        "timestep": 0,
    }


def test_enrich_ego_state_frame_zero_defaults():
    stub, Evaluator = _stub_with_env_and_adapter()
    es = _base_ego_state()
    out = Evaluator._enrich_ego_state(stub, es)

    # Spec §4.1: at frame 0, acceleration = [0, 0, 9.8] and angular_velocity zeros.
    np.testing.assert_allclose(out["acceleration"], [0.0, 0.0, 9.8])
    np.testing.assert_allclose(out["angular_velocity"], [0.0, 0.0, 0.0])
    # Prev caches primed for next frame.
    assert stub._prev_ego_velocity is not None
    assert stub._prev_ego_heading == 0.0


def test_enrich_ego_state_finite_difference_next_frame():
    stub, Evaluator = _stub_with_env_and_adapter()
    Evaluator._enrich_ego_state(stub, _base_ego_state(vx=1.0, heading=0.0))

    # 2nd frame: velocity went 1 → 3 m/s along +x; heading 0 → 0.02 rad.
    # At dt=0.1, expect accel[0] = 20 m/s², yaw_rate = 0.2 rad/s.
    out = Evaluator._enrich_ego_state(
        stub,
        {
            "position": np.array([0.1, 0.0, 0.0], dtype=np.float32),
            "heading": 0.02,
            "speed": 3.0,
            "velocity": np.array([3.0, 0.0, 0.0], dtype=np.float32),
            "timestep": 1,
        },
    )
    assert out["acceleration"][0] == pytest.approx(20.0, rel=1e-6)
    assert out["angular_velocity"][2] == pytest.approx(0.2, rel=1e-6)


def test_enrich_ego_state_wraps_heading_jumps():
    """Heading flip across ±π must not emit a giant yaw-rate spike."""
    stub, Evaluator = _stub_with_env_and_adapter()
    # Frame 0: heading near +π
    Evaluator._enrich_ego_state(
        stub,
        {
            "position": np.zeros(3, dtype=np.float32),
            "heading": np.pi - 0.1,
            "velocity": np.zeros(3, dtype=np.float32),
            "timestep": 0,
        },
    )
    # Frame 1: wraps to -π + 0.1 — a raw diff would read ~-2π/dt.
    out = Evaluator._enrich_ego_state(
        stub,
        {
            "position": np.zeros(3, dtype=np.float32),
            "heading": -np.pi + 0.1,
            "velocity": np.zeros(3, dtype=np.float32),
            "timestep": 1,
        },
    )
    assert abs(out["angular_velocity"][2]) == pytest.approx(2.0, abs=0.05)


# Note: The three `pdm_closed_*_no_keyerror_on_bare_ego_state` tests that
# previously lived here were removed when the `pdm_closed_adapter` module
# was retired during the NexusSim package reorg (§12.6 — no shims, hard
# cut). The defensive-path coverage they provided is now redundant
# because the new ego_state contract guarantees the keys those tests
# exercised. Restore via a fresh test if a regression appears.



