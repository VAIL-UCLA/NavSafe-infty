# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A mid-step evaluator crash must not end the episode as if it completed.

``Evaluator._step`` wraps the whole frame in ``except Exception`` and returns
``False`` — the run() loop then finalizes normally. Before this suite, that
produced a truncated ``metrics.json`` classified by the live monitor as
``budget_expired`` / ``scorable: true``: a broken run that downstream means
counted as a healthy episode (the "eval scene-drop bias" family of failures,
inverted — a silent *inclusion*).

The contract these tests pin:

* a step-level exception is recorded and finalize() classifies the episode
  ``termination_reason: "infra_failure"`` / ``scorable: false`` — the NavSafe
  taxonomy's "excluded from every denominator" ending;
* the exception class+message is stored in ``metrics.json`` itself
  (``infra_failure_error``), because that file is the only artifact
  aggregators read;
* a live termination recorded *before* the crash keeps precedence — the
  episode had already ended for a real reason;
* a healthy episode's metrics.json carries NO ``infra_failure_error`` key
  (additive-only: byte-identical output for clean runs).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from navsafe.evaluation.evaluator import EPISODE_SCORE_KEYS, Evaluator
from navsafe.benchmark.termination import (
    LiveMonitor,
    Termination,
    TerminationReason,
)


class _CrashingEnv:
    """Env facade whose per-frame state read raises (mid-step infra failure)."""

    current_scenario = None

    def get_ego_state(self):
        raise RuntimeError("injected mid-step infra failure")


def _live_monitor(frames: int) -> LiveMonitor:
    monitor = LiveMonitor(np.stack([np.arange(50.0), np.zeros(50)], axis=1),
                          warmup=0, dt=0.1, t_max_s=4.0)
    for i in range(frames):
        monitor.update(ego_xy=np.array([float(i), 0.0]), ego_speed=1.0)
    return monitor


def _make_evaluator(tmp_path: Path, *, frames: int = 3) -> Evaluator:
    """An Evaluator that can run _step + finalize without booting IsaacSim."""
    ev = object.__new__(Evaluator)
    ev._reactivity_trace = None
    ev.scenario_id = "crash_scene"
    ev.frame = frames
    ev.start_time = datetime(2026, 8, 18, 0, 0, 0)
    ev.end_time = datetime(2026, 8, 18, 0, 0, 1)
    ev.env = _CrashingEnv()
    ev.adapter = SimpleNamespace()  # no `perceive` attr -> hook skipped
    ev.config = SimpleNamespace(
        output_dir=tmp_path, ego_replay_frames=0, eval_frames=40,
        enable_vis=False, traffic_mode="log_replay", eval_mode="closed_loop",
        controller_type="pure_pursuit", sim_dt=0.1, replan_rate=1,
        execution_mode="teleport",
    )
    ev._done = False
    ev._step_crash = None
    ev._history = {
        "trajectories": [],
        "vehicle_states": [
            {"position": np.array([float(i), 0.0, 0.0])} for i in range(frames)
        ],
        "metrics": [{"velocity": 1.0, "collision": False}] * frames,
        "actions": [],
        "timestamps": [],
    }
    ev._epdms_scorer = None
    ev._epdms_results = [
        {"frame": i, "valid": True, "score": 0.5,
         "no_at_fault_collisions": 1.0, "drivable_area_compliance": 1.0,
         "driving_direction_compliance": 1.0, "traffic_light_compliance": 1.0,
         "time_to_collision_within_bound": 1.0, "lane_keeping": 1.0,
         "history_comfort": 1.0, "extended_comfort": 1.0}
        for i in range(frames)
    ]
    ev._epdms_score_failures = 0
    ev._navsafe_trace = None
    ev._termination = _live_monitor(frames)
    ev._terminated_reason = None
    ev._route = SimpleNamespace(goal_latched=False)
    return ev


def _metrics(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "metrics.json").read_text())


def test_step_crash_is_recorded_not_masked(tmp_path):
    """The exception is caught, the episode ends, and the crash is on record."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path)

    keep_going = ev._step()

    assert keep_going is False, "_step must still end the episode"
    assert ev._step_crash is not None
    assert ev._step_crash["frame"] == 3
    assert "RuntimeError" in ev._step_crash["error"]
    assert "injected mid-step infra failure" in ev._step_crash["error"]


def test_crashed_episode_is_unscorable_infra_failure(tmp_path):
    """finalize() after a crash: infra_failure, not a healthy budget_expired."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path)
    assert ev._step() is False

    results = ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["termination_reason"] == "infra_failure", (
        "the live monitor reads a truncated trace as budget_expired; the "
        "recorded crash must preempt that misclassification"
    )
    assert metrics["scorable"] is False
    # The exception class+message lives in metrics.json itself — the only
    # artifact aggregators read.
    assert "RuntimeError" in metrics["infra_failure_error"]
    assert "injected mid-step infra failure" in metrics["infra_failure_error"]
    # Exclusion is decided by `scorable`; every composite stays present.
    assert all(key in metrics for key in EPISODE_SCORE_KEYS)
    assert results["termination"]["reason"] == "infra_failure"
    assert results["termination"]["frame"] == 3
    assert results["termination"]["policy_attributed"] is False


def test_live_termination_keeps_precedence_over_crash(tmp_path):
    """A real ending recorded before the crash stays the episode's ending."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path)
    ev._terminated_reason = Termination(
        TerminationReason.CONTACT_AT_FAULT, frame=2, detail="hit agent a17")
    ev._step_crash = {"frame": 3, "error": "RuntimeError: post-ending crash"}

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["termination_reason"] == "contact_at_fault"
    assert metrics["scorable"] is True
    # The crash is still surfaced, so the record stays complete.
    assert "post-ending crash" in metrics["infra_failure_error"]


def test_crash_record_is_cleared_by_reset_for_the_next_scenario(tmp_path):
    """Batch runs reuse one Evaluator across scenarios via setup() →
    _reset_state(); a crash in scene N must not label scene N+1."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path)
    ev._route = SimpleNamespace(goal_latched=False, reset=lambda: None)
    assert ev._step() is False
    assert ev._step_crash is not None

    ev._reset_state()

    assert ev._step_crash is None
    # Restore the harness fields _reset_state cleared, then finalize a
    # healthy episode: no crash keys may leak from the previous scenario.
    ev2 = _make_evaluator(tmp_path)
    ev2._step_crash = ev._step_crash
    ev2.finalize()
    metrics = _metrics(tmp_path)
    assert "infra_failure_error" not in metrics
    assert metrics["scorable"] is True


def test_healthy_episode_has_no_crash_key(tmp_path):
    """Additive-only: clean runs must not grow a new metrics.json key."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path)
    # No _step call: finalize over the healthy 3-frame history.

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert "infra_failure_error" not in metrics
    assert metrics["scorable"] is True
