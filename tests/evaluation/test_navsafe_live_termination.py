# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A benchmark failure must not be published as a policy score.

Scoring reconstructs the ending from the stored trace, where an episode the
renderer killed looks exactly like one that ran out of frames: the samples
just stop. Measured on ``00c1e4eb4a045f20`` — ``nurec_grpc render failed for
CAM_F0`` at frame 117, which the evaluator recorded as ``infra_failure`` and
the post-hoc pass re-read as ``trace_exhausted``, writing ``status: scored``
and ``driving_score: 45.816`` for an episode no policy was responsible for.
In a sweep that number lands in the mean.

``score_run(live_termination=...)`` is the channel that closes it: when the
evaluator's own reason is benchmark-attributed the run is excluded and the
reason recorded, whatever the trace looks like.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely")

from navsafe.benchmark.scoring.from_run import score_run  # noqa: E402
from navsafe.benchmark.trace.from_eval import EvalRun  # noqa: E402
from navsafe.benchmark.trace.schema import (  # noqa: E402
    PHASE_SCORED,
    PHASE_WARMUP,
    empty_frame,
)

WARMUP = 20


def _run(n: int = 120) -> EvalRun:
    """A clean straight-line episode whose samples simply stop at ``n``."""
    t = np.arange(n) * 0.1
    ego = np.stack([2.0 + 5.0 * t, np.zeros(n)], axis=1)
    route = np.stack([np.linspace(0.0, 400.0, 400), np.zeros(400)], axis=1)
    frames = []
    for i in range(n):
        row = empty_frame()
        row.update({
            "frame": i,
            "phase": PHASE_SCORED if i >= WARMUP else PHASE_WARMUP,
            "ego_speed": 5.0,
            "ego_x": float(ego[i, 0]),
            "ego_y": float(ego[i, 1]),
            "contacts": [],
        })
        frames.append(row)
    return EvalRun(
        frames=frames, route_xy=route, ego_xy=ego, dt=0.1,
        warmup_frames=WARMUP, drivable_known=True, collision_count=0)


def _score(**kw):
    return score_run(_run(), name="probe", warmup_frames=WARMUP, dt=0.1, **kw)


class TestLiveTermination:
    def test_unlimited_route_clock_is_recorded_without_json_infinity(self):
        out = _score(t_max=float("inf"))
        assert out["frames"]["t_max_s"] is None
        assert out["termination"]["reason"] == "trace_exhausted"
        assert not any("run with --eval-frames" in n for n in out["notes"])

    def test_infra_failure_excludes_the_cell(self):
        out = _score(live_termination="infra_failure")
        assert out["status"] == "excluded"
        assert out["termination"]["reason"] == "infra_failure"
        assert any("evaluator" in n for n in out["notes"])

    def test_without_it_trace_exhaustion_is_excluded(self):
        """A source/harness ending is not a completed policy outcome."""
        out = _score()
        assert out["status"] == "excluded"
        assert out["termination"]["reason"] == "trace_exhausted"

    def test_live_policy_reason_overrides_posthoc_trace_fallback(self):
        """The live evaluator knows why stepping stopped."""
        out = _score(live_termination="goal_reached")
        assert out["status"] == "scored"
        assert out["termination"]["reason"] == "goal_reached"

    def test_unknown_reason_is_noted_not_raised(self):
        out = _score(live_termination="something_new")
        assert out["status"] == "excluded"
        assert any("unknown termination" in n for n in out["notes"])

    def test_trace_exhausted_is_excluded(self):
        out = _score(live_termination="trace_exhausted")
        assert out["status"] == "excluded"

    def test_not_at_fault_contact_stays_scored(self):
        """A replayed follower rear-ending the ego is scored WITH the ego's
        own infractions (from_run's design note); it is not a benchmark
        failure, however un-attributable the contact is."""
        out = _score(live_termination="contact_not_at_fault")
        assert out["status"] == "scored"

    def test_v8_pre_intersection_at_fault_contact_is_scored(self):
        """A decisive V-8 crash is a policy failure even before the junction."""
        run = _run()
        run.frames[WARMUP]["contacts"] = [{
            "agent_id": "crossing-vru", "kind": "front", "at_fault": True,
        }]
        run.collision_count = 1

        out = score_run(
            run, name="v8-crash", warmup_frames=WARMUP, dt=0.1,
            scenario_meta={"leaf": "V-8"},
            live_termination="contact_at_fault")

        assert out["status"] == "scored"
        assert out["termination"]["reason"] == "contact_at_fault"
        assert out["metrics"]["success"] is False
        assert out["driving_score_breakdown"]["penalty"] < 1.0

    def test_v8_pre_intersection_not_at_fault_contact_is_scored(self):
        """A replay actor ending V-8 early is an outcome, not a false hold."""
        run = _run()
        run.frames[WARMUP]["contacts"] = [{
            "agent_id": "rear-follower", "kind": "rear_end", "at_fault": False,
        }]
        run.collision_count = 1

        out = score_run(
            run, name="v8-rear-ended", warmup_frames=WARMUP, dt=0.1,
            scenario_meta={"leaf": "V-8"},
            live_termination="contact_not_at_fault")

        assert out["status"] == "scored"
        assert out["termination"]["reason"] == "contact_not_at_fault"
        assert out["metrics"]["success"] is False
        assert out["driving_score_breakdown"]["penalty"] == 1.0
