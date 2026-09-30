# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""An episode that could not be scored must say so, not go silent.

The contract these tests pin:

* every composite score is present in ``metrics.json``, defaulting to ``0.0``;
* the EPDMS sub-terms are NOT defaulted — a 0.0 there is a verdict about the
  ego (at-fault collision, off drivable) that admission gates read as measured
  evidence, so imputing one would manufacture an infraction;
* "cannot be scored" is stated by ``scorable`` (from the NavSafe termination
  taxonomy), never implied by an absent key."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from navsafe.evaluation.evaluator import (
    EPDMS_SUBSCORE_KEYS,
    EPISODE_SCORE_KEYS,
    Evaluator,
)
from navsafe.benchmark.termination import (
    LiveMonitor,
    Termination,
    TerminationReason,
)
from navsafe.scenario.scenario_description import ScenarioDescription as SD


class _ScenarioWithBrokenRoute(dict):
    """A scenario whose track lookup raises, as a bad projection would."""

    def get(self, key, default=None):  # type: ignore[override]
        if key == SD.TRACKS:
            raise RuntimeError("route projection failed")
        return super().get(key, default)


def _scenario_with_route() -> dict:
    positions = np.stack([np.arange(12.0), np.zeros(12)], axis=1)
    return {
        SD.METADATA: {SD.SDC_ID: "ego"},
        SD.TRACKS: {"ego": {SD.STATE: {"position": positions}}},
    }


def _live_monitor(frames: int) -> LiveMonitor:
    """A monitor fed the same frames the evaluator would have fed it.

    The real classification runs through ``LiveMonitor.final`` — driving it
    (rather than injecting a ``Termination``) is what makes these tests able
    to fail when the production wiring changes.
    """
    monitor = LiveMonitor(np.stack([np.arange(12.0), np.zeros(12)], axis=1),
                          warmup=1, dt=0.1, t_max_s=10.0)
    for i in range(frames):
        monitor.update(ego_xy=np.array([float(i), 0.0]), ego_speed=1.0)
    return monitor


def _make_evaluator(tmp_path, scenario, *, termination=None, frames=6,
                    monitor_frames=None):
    """A finalize-able Evaluator without booting IsaacSim.

    ``finalize`` is pure aggregation over already-collected history, so the
    fields it reads are set directly; nothing else in the class is exercised.
    ``_termination`` is always a real ``LiveMonitor`` — as in production, where
    ``_terminated_reason`` is only ever assigned while the monitor exists.
    """
    ev = object.__new__(Evaluator)
    ev._reactivity_trace = None
    ev.scenario_id = "scene_a"
    ev.frame = frames
    ev.start_time = datetime(2026, 8, 12, 0, 0, 0)
    ev.end_time = datetime(2026, 8, 12, 0, 0, 1)
    ev.env = SimpleNamespace(current_scenario=scenario)
    ev.config = SimpleNamespace(
        output_dir=tmp_path,
        ego_replay_frames=1,
        eval_frames=frames,
        enable_vis=False,
        traffic_mode="replay",
        eval_mode="teleport",
        controller_type="none",
    )
    ev._history = {
        "trajectories": [],
        "vehicle_states": [
            {"position": np.array([float(i), 0.0, 0.0])} for i in range(frames)
        ],
        "metrics": [{"velocity": 1.0, "collision": False} for _ in range(frames)],
        "actions": [],
        "timestamps": [],
    }
    ev._epdms_results = [
        {
            "frame": i,
            "valid": True,
            "score": 0.5,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "extended_comfort": 1.0,
        }
        for i in range(1, frames)
    ]
    ev._navsafe_trace = None
    ev._termination = _live_monitor(
        frames if monitor_frames is None else monitor_frames)
    ev._terminated_reason = termination
    ev._route = SimpleNamespace(goal_latched=False)
    return ev


def _metrics(tmp_path) -> dict:
    # Evaluator writes artifacts directly into its configured output_dir;
    # scenario identity is carried inside metrics.json.
    return json.loads((tmp_path / "metrics.json").read_text())


def test_unscored_episode_reports_zero_instead_of_omitting_the_key(tmp_path):
    """A thrown route projection scores 0.0 — it does not vanish."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path, _ScenarioWithBrokenRoute())

    results = ev.finalize()

    metrics = _metrics(tmp_path)
    missing = [key for key in EPISODE_SCORE_KEYS if key not in metrics]
    assert not missing, (
        f"metrics.json omits {missing}; a consumer that requires every key "
        "numeric will drop this scene from its mean instead of scoring it 0"
    )
    assert metrics["epdms"] == 0.0
    assert metrics["driving_score"] == 0.0
    assert results["metrics"]["epdms"] == 0.0
    # The whole RC-family is COUNTED, not fabricated: the EP/RC block threw,
    # so driving_score / route_completion_fraction / ego_progress must appear
    # in defaulted_metrics alongside epdms instead of being published as
    # measured zeros (the historical `except: pass` fabrication). epdms_no_ep
    # was genuinely measured, so it must NOT be listed.
    assert metrics["defaulted_metrics"] == [
        "epdms", "driving_score", "route_completion_fraction", "ego_progress"]
    assert metrics["route_completion_fraction"] == 0.0
    assert metrics["ego_progress"] == 0.0
    # The episode still ran, so it is the policy's zero to own.
    assert metrics["scorable"] is True


def test_measured_subterms_are_never_imputed(tmp_path):
    """A 0.0 sub-term is a verdict about the ego and must be earned.

    ``no_at_fault_collisions: 0.0`` means "the ego caused a collision" to the
    report card and the admission gate.  An episode that was never scored has
    no such verdict, so the key stays absent — those consumers already refuse
    loudly on absence, and a fabricated zero would silently charge the policy
    with an infraction it was never observed to commit.
    """
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path, _scenario_with_route())
    ev._epdms_results = []  # nothing scored at all

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert all(key in metrics for key in EPISODE_SCORE_KEYS)
    assert not any(key in metrics for key in EPDMS_SUBSCORE_KEYS)


def test_scored_episode_keeps_its_computed_values(tmp_path):
    """The defaulting must not overwrite a real score."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path, _scenario_with_route())

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["epdms_no_ep"] == pytest.approx(0.5)
    assert metrics["route_completion_fraction"] > 0.0
    assert metrics["epdms"] > 0.0
    assert metrics["scorable"] is True
    assert "defaulted_metrics" not in metrics
    # Classified by the live monitor, not by an injected verdict: nothing
    # ended this episode, so the frame cap did -- which is `trace_exhausted`,
    # not `budget_expired`. The latter now means the ego used the whole 60 s
    # safety ceiling.
    assert metrics["termination_reason"] == "trace_exhausted"


def test_goal_reached_is_exactly_complete_in_live_metrics(tmp_path):
    """Goal tolerance is part of completion, not a 2.4% score penalty.

    The post-run scorer has always promoted GOAL_REACHED to 100%.  Pin the
    same rule in ``Evaluator.finalize`` so an episode cannot terminate as a
    successful route yet publish a fractional route-completion metric.
    """
    pytest.importorskip("pandas")
    ev = _make_evaluator(
        tmp_path,
        _scenario_with_route(),
        termination=Termination(
            TerminationReason.GOAL_REACHED, frame=5, detail="within goal tolerance"
        ),
    )

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["termination_reason"] == "goal_reached"
    assert metrics["route_completion_fraction"] == 1.0
    assert metrics["ego_progress"] == 1.0
    assert metrics["driving_score"] == pytest.approx(metrics["epdms_no_ep"])


def test_episode_with_no_scored_frames_is_unscorable(tmp_path):
    """The recorded ``total_frames: 0`` failure mode, through the real monitor.

    14 ag20 scenes published a metrics.json with zero frames and no scores.
    Nothing about the policy was measured, so the episode is excluded by
    declaration — visible in ``scorable`` — instead of silently missing a key.
    """
    pytest.importorskip("pandas")
    ev = _make_evaluator(tmp_path, _scenario_with_route(), monitor_frames=0)
    ev._epdms_results = []
    ev._history["vehicle_states"] = []
    ev._history["metrics"] = []
    ev.frame = 0

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["total_frames"] == 0
    assert metrics["scorable"] is False
    assert metrics["termination_reason"] == "infra_failure"
    assert metrics["driving_score"] == 0.0


@pytest.mark.parametrize(
    "reason", [TerminationReason.ENVELOPE_EXIT, TerminationReason.INFRA_FAILURE]
)
def test_benchmark_ended_episode_declares_itself_unscorable(tmp_path, reason):
    """envelope_exit / infra_failure are the benchmark's endings, not the policy's."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(
        tmp_path,
        _scenario_with_route(),
        termination=Termination(reason, frame=3, detail="left the envelope"),
    )

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["scorable"] is False, (
        "an episode the benchmark ended must be excluded by declaration, not "
        "by an absent key"
    )
    assert metrics["termination_reason"] == reason.value
    # The composites are still present: exclusion is decided by `scorable`, so
    # no consumer has to infer intent from a KeyError.
    assert all(key in metrics for key in EPISODE_SCORE_KEYS)


def test_policy_attributed_ending_stays_scorable(tmp_path):
    """A deadlock is the policy's zero; it is scored, never excluded."""
    pytest.importorskip("pandas")
    ev = _make_evaluator(
        tmp_path,
        _scenario_with_route(),
        termination=Termination(
            TerminationReason.DEADLOCK, frame=3, detail="< 0.1 m/s for 5 s"
        ),
    )

    ev.finalize()

    metrics = _metrics(tmp_path)
    assert metrics["scorable"] is True
    assert metrics["termination_reason"] == "deadlock"
