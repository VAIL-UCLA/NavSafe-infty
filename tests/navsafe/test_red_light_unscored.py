"""`red_light` is never priced from the logged TL column.

The logged light states are not reliable enough to carry a 0.70 multiplier: a
wrong `red` turns a clean episode into a 70 % run and nothing distinguishes
that from a genuine violation after the fact. So the channel is SKIPPED on
every run rather than scored — and, because no infraction is raised, it also
stops blocking `RouteResult.success()`.

What must keep working: the authored hold leaves. `RED_LIGHT_RUN` comes from
`scenario_rules.hold_violation` against a per-bundle `hold_region`, not from
this column, so an authored stop line still ends the episode.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.scoring.from_run import score_run
from navsafe.benchmark.scoring import metrics as navsafe_metrics
from navsafe.benchmark.trace.from_eval import EvalRun
from navsafe.benchmark.trace.schema import PHASE_SCORED, PHASE_WARMUP, empty_frame

WARMUP = 20
N = 120
DT = 0.1


def _run(signal: str | None) -> EvalRun:
    """Straight clean episode; every scored frame carries ``signal``."""
    t = np.arange(N) * DT
    ego = np.stack([2.0 + 5.0 * t, np.zeros(N)], axis=1)
    route = np.stack([np.linspace(0.0, 400.0, 400), np.zeros(400)], axis=1)
    frames = []
    for i in range(N):
        row = empty_frame()
        row.update({
            "frame": i,
            "phase": PHASE_SCORED if i >= WARMUP else PHASE_WARMUP,
            "ego_speed": 5.0,
            "ego_x": float(ego[i, 0]),
            "ego_y": float(ego[i, 1]),
            "contacts": [],
        })
        if signal is not None:
            row["signal_id"] = "sig_0"
            row["signal_state"] = signal
        frames.append(row)
    return EvalRun(
        frames=frames, route_xy=route, ego_xy=ego, dt=DT,
        warmup_frames=WARMUP, drivable_known=True, collision_count=0)


def _score(signal):
    return score_run(_run(signal), name="probe", warmup_frames=WARMUP, dt=DT)


def _channels(out):
    return {c["channel"] for c in out["driving_score_breakdown"]["channels"]}


class TestRedLightUnscored:
    def test_red_frames_do_not_reduce_the_penalty(self):
        red = _score("red")
        assert red["driving_score_breakdown"]["penalty"] == pytest.approx(1.0)

    def test_red_scores_identically_to_green(self):
        red = _score("red")
        green = _score("green")
        assert (red["metrics"]["driving_score"]
                == pytest.approx(green["metrics"]["driving_score"]))
        assert red["metrics"]["success"] == green["metrics"]["success"]

    def test_red_light_is_reported_skipped_not_clean(self):
        """A zero-count row would read as "checked and clean"; it was not."""
        out = _score("red")
        assert "red_light" not in _channels(out)
        assert "red_light" in out["driving_score_breakdown"][
            "penalty_channels_skipped"]

    def test_skipped_even_when_the_column_is_absent(self):
        out = _score(None)
        assert "red_light" not in _channels(out)
        assert "red_light" in out["driving_score_breakdown"][
            "penalty_channels_skipped"]

    def test_success_is_not_blocked_by_a_red_frame(self):
        """No infraction is raised, so success() cannot see one."""
        out = _score("red")
        assert out["metrics"]["success"] == _score("green")["metrics"]["success"]

    def test_the_coefficient_table_is_untouched(self):
        """Bench2Drive's reference table still prices a red light that some
        other caller hands to RouteResult directly -- only NavSafe's derivation
        from the logs is gone."""
        assert navsafe_metrics.PENALTY_COEFFICIENTS["red_light"] == 0.70
        r = navsafe_metrics.RouteResult("r", 100.0, True, {"red_light": 1})
        assert r.penalty() == pytest.approx(0.70)
        assert r.success() is False
