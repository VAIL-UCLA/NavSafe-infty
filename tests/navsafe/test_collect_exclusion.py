"""Benchmark endings must leave the denominators (runner/collect.py)."""

from __future__ import annotations

import json
from pathlib import Path

from navsafe.benchmark.runner import collect


def _verdict(seed_id: str, score: float, reason: str, attributed: bool) -> dict:
    return {
        "rubric": {"passed": False, "gate_violated": False, "family": "F3",
                   "seed_id": seed_id, "regime": "log_replay",
                   "success": [], "gates": [], "diagnostics": {},
                   "coverage": {"n_frames": 100, "n_scored": 80,
                                "n_render_invalid": 0, "scored_fraction": 0.8}},
        "outcome": {"labels": [], "evidence": {}, "crash_cells": []},
        "score": {"episode_id": f"{seed_id}-log_replay", "seed_id": seed_id,
                  "family": "F3", "regime": "log_replay", "policy": "p",
                  "rubric_passed": False, "safety_gate": 1.0,
                  "progress": 0.5, "rules": 0.5, "comfort": 0.5,
                  "recovery": 1.0, "score": score, "labels": [],
                  "weights": {"progress": 0.25, "rules": 0.25,
                              "comfort": 0.25, "recovery": 0.25},
                  "coverage": {"n_frames": 100, "n_scored": 80,
                               "n_render_invalid": 0}},
        "termination": {"reason": reason, "frame": 50, "detail": "",
                        "policy_attributed": attributed, "t_max_s": 12.0},
    }


def _write(root: Path, seed_id: str, verdict: dict) -> None:
    d = root / seed_id / "run" / "trace"
    d.mkdir(parents=True)
    (d / "verdict.json").write_text(json.dumps(verdict))


def test_envelope_exit_is_excluded_from_the_aggregate(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "s1", _verdict("s1", 0.6, "budget_expired", True))
    _write(tmp_path, "s2", _verdict("s2", 0.0, "envelope_exit", False))

    monkeypatch.setattr("sys.argv",
                        ["collect", "--eval-root", str(tmp_path),
                         "--json", str(tmp_path / "out.json")])
    assert collect.main() == 0

    result = json.loads((tmp_path / "out.json").read_text())
    agg = result["aggregate"]
    # The excluded episode is not in the mean and not in the count ...
    assert agg["n_episodes"] == 1
    assert agg["nss"] == 60.0
    assert agg["n_excluded"] == 1
    assert agg["excluded_reasons"] == ["envelope_exit"]
    # ... but it is still reported, with its reason.
    assert [e["seed_id"] for e in result["excluded_episodes"]] == ["s2"]
    assert "excluded from all denominators" in capsys.readouterr().out


def test_all_excluded_is_an_error_not_an_empty_mean(tmp_path, monkeypatch):
    _write(tmp_path, "s1", _verdict("s1", 0.0, "infra_failure", False))
    monkeypatch.setattr("sys.argv", ["collect", "--eval-root", str(tmp_path)])
    assert collect.main() == 1


def test_verdicts_without_termination_still_aggregate(tmp_path, monkeypatch):
    """Traces written before the taxonomy existed carry no termination block;
    they must keep aggregating rather than silently vanishing."""
    v = _verdict("s1", 0.4, "budget_expired", True)
    v.pop("termination")
    _write(tmp_path, "s1", v)
    monkeypatch.setattr("sys.argv",
                        ["collect", "--eval-root", str(tmp_path),
                         "--json", str(tmp_path / "out.json")])
    assert collect.main() == 0
    agg = json.loads((tmp_path / "out.json").read_text())["aggregate"]
    assert agg["n_episodes"] == 1 and agg["n_excluded"] == 0
