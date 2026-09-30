# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Batch consumers must count unscorable episodes — never average them in.

Two consumers are pinned here:

* ``navsafe.cli.batch_eval`` (the ``navsafe-batch-eval`` wrapper): a scene
  whose evaluator crashed mid-episode exits 0, so the wrapper reads each
  scene's ``metrics.json`` after the run — ``infra_failure`` / missing
  metrics is a scene FAILURE (non-zero exit), a benchmark-ended episode
  (``scorable: false``) is a reported unscorable count.
* ``navsafe.evaluation.batch_evaluator.BatchEvaluator``: an episode whose
  metrics say ``scorable: false`` is excluded from the success set every
  aggregation reads and counted in the summary — as status ``"error"`` when
  the evaluator itself broke (``termination_reason: "infra_failure"``; a
  harness failure, retried on resume) and ``"unscorable"`` for benchmark
  endings (``envelope_exit``: a valid run, excluded from every mean).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navsafe.cli.batch_eval import _scene_verdict


# ---------------------------------------------------------------------------
# _scene_verdict: metrics.json -> (verdict, detail)
# ---------------------------------------------------------------------------

def _write_metrics(scene_dir: Path, payload: dict) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "metrics.json").write_text(json.dumps(payload))


def test_verdict_ok_for_scorable_scene(tmp_path):
    _write_metrics(tmp_path, {"scorable": True,
                              "termination_reason": "budget_expired",
                              "driving_score": 0.7})
    assert _scene_verdict(tmp_path) == ("ok", "")


def test_verdict_infra_failure_carries_the_error(tmp_path):
    _write_metrics(tmp_path, {
        "scorable": False,
        "termination_reason": "infra_failure",
        "infra_failure_error": "frame 12: RuntimeError: boom",
    })
    verdict, detail = _scene_verdict(tmp_path)
    assert verdict == "infra_failure"
    assert "RuntimeError: boom" in detail


def test_verdict_unscorable_for_benchmark_ended_scene(tmp_path):
    _write_metrics(tmp_path, {"scorable": False,
                              "termination_reason": "envelope_exit"})
    assert _scene_verdict(tmp_path) == ("unscorable", "envelope_exit")


def test_verdict_missing_metrics_is_a_failure(tmp_path):
    verdict, _ = _scene_verdict(tmp_path)  # no metrics.json written
    assert verdict == "no_metrics"


def test_verdict_unreadable_metrics_is_a_failure(tmp_path):
    (tmp_path / "metrics.json").write_text("{not json")
    verdict, _ = _scene_verdict(tmp_path)
    assert verdict == "no_metrics"


def test_verdict_non_dict_metrics_is_a_failure(tmp_path):
    """Valid JSON that is not an object must classify, not crash the batch."""
    (tmp_path / "metrics.json").write_text("[1, 2, 3]")
    verdict, _ = _scene_verdict(tmp_path)
    assert verdict == "no_metrics"


# ---------------------------------------------------------------------------
# Wrapper end-to-end (stub eval entry, real subprocess loop)
# ---------------------------------------------------------------------------

_STUB_EVAL = """\
import json, pathlib, sys

args = sys.argv[1:]
idx = args[args.index("--py123d-scene-index") + 1]
out = pathlib.Path(args[args.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
payloads = {
    "0": {"scorable": True, "termination_reason": "budget_expired"},
    "1": {"scorable": False, "termination_reason": "envelope_exit"},
    "2": {"scorable": False, "termination_reason": "infra_failure",
          "infra_failure_error": "frame 7: ValueError: stub crash"},
}
if idx == "3":
    sys.exit(0)  # exits 0 but writes NO metrics.json
(out / "metrics.json").write_text(json.dumps(payloads[idx]))
"""


@pytest.fixture()
def stub_entry(tmp_path, monkeypatch):
    import navsafe.cli.batch_eval as batch_eval_mod

    entry = tmp_path / "stub_eval.py"
    entry.write_text(_STUB_EVAL)
    monkeypatch.setattr(batch_eval_mod, "_EVAL_ENTRY", entry)
    return batch_eval_mod


def test_batch_all_healthy_exits_zero(stub_entry, tmp_path, capsys):
    rc = stub_entry.main(["--scene-indices", "0",
                          "--output-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 0
    assert "1/1 scene(s) succeeded" in err


def test_batch_unscorable_scene_is_counted_not_averaged(stub_entry, tmp_path,
                                                        capsys):
    rc = stub_entry.main(["--scene-indices", "0 1",
                          "--output-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    # A benchmark-ended episode is a valid run: no batch failure...
    assert rc == 0
    # ...but it is loudly counted and named, never folded into "succeeded".
    assert "1/2 scene(s) succeeded" in err
    assert "1 unscorable" in err
    assert "envelope_exit" in err


def test_batch_masked_crash_is_a_failure(stub_entry, tmp_path, capsys):
    """rc=0 + termination_reason infra_failure == a crash the exit code hid."""
    rc = stub_entry.main(["--scene-indices", "2",
                          "--output-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "ValueError: stub crash" in err


def test_batch_missing_metrics_is_a_failure(stub_entry, tmp_path, capsys):
    rc = stub_entry.main(["--scene-indices", "3",
                          "--output-dir", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no_metrics" in err


# ---------------------------------------------------------------------------
# BatchEvaluator (imports torch transitively -> lazy import + skip on stock CI)
# ---------------------------------------------------------------------------

def _batch_evaluator_cls():
    pytest.importorskip("torch")
    from navsafe.evaluation.batch_evaluator import BatchEvaluator
    return BatchEvaluator


def test_batch_evaluator_refuses_dead_score_start_frame(tmp_path):
    BatchEvaluator = _batch_evaluator_cls()
    with pytest.raises(ValueError, match="ego_replay_frames"):
        BatchEvaluator(
            model_type="pdm_closed", checkpoint_path="none",
            scenario_root=str(tmp_path), output_root=str(tmp_path / "out"),
            score_start_frame=10,
        )


def test_batch_evaluator_wires_enable_vis(tmp_path):
    BatchEvaluator = _batch_evaluator_cls()
    be = BatchEvaluator(
        model_type="pdm_closed", checkpoint_path="none",
        scenario_root=str(tmp_path), output_root=str(tmp_path / "out"),
        enable_vis=True,
    )
    assert be.eval_config.enable_vis is True


def _stubbed_batch_evaluator(tmp_path, monkeypatch, metrics: dict):
    BatchEvaluator = _batch_evaluator_cls()
    be = BatchEvaluator(
        model_type="pdm_closed", checkpoint_path="none",
        scenario_root=str(tmp_path), output_root=str(tmp_path / "out"),
    )

    class _StubEvaluator:
        config = be.eval_config

        def setup(self, scenario_path):
            pass

        def run(self):
            return {"metrics": dict(metrics)}

    monkeypatch.setattr(be, "_get_evaluator", lambda: _StubEvaluator())
    return be


def test_batch_evaluator_marks_crash_as_error_not_unscorable(tmp_path,
                                                             monkeypatch):
    """infra_failure is a HARNESS failure: status 'error' (retried on resume),
    never the benign 'unscorable' that reads as a valid benchmark ending."""
    be = _stubbed_batch_evaluator(tmp_path, monkeypatch, {
        "scorable": False,
        "termination_reason": "infra_failure",
        "infra_failure_error": "frame 4: RuntimeError: stub",
    })
    scene = tmp_path / "scene_a"
    scene.mkdir()

    record = be.evaluate_scenario(scene)

    assert record["status"] == "error"
    assert record["termination_reason"] == "infra_failure"
    assert "RuntimeError: stub" in record["error"]

    be.results["scenarios"]["scene_a"] = record
    be.save_results()
    summary = json.loads(be.results_file.read_text())["summary"]
    assert summary["error"] == 1
    assert summary["unscorable"] == 0
    assert summary["success"] == 0


def test_batch_evaluator_marks_benchmark_ending_unscorable(tmp_path,
                                                           monkeypatch):
    be = _stubbed_batch_evaluator(tmp_path, monkeypatch, {
        "scorable": False,
        "termination_reason": "envelope_exit",
    })
    scene = tmp_path / "scene_a"
    scene.mkdir()

    record = be.evaluate_scenario(scene)

    assert record["status"] == "unscorable"
    assert record["termination_reason"] == "envelope_exit"

    be.results["scenarios"]["scene_a"] = record
    be.save_results()
    summary = json.loads(be.results_file.read_text())["summary"]
    assert summary["unscorable"] == 1
    assert summary["error"] == 0
    assert summary["success"] == 0


def test_batch_evaluator_scorable_scene_stays_success(tmp_path, monkeypatch):
    BatchEvaluator = _batch_evaluator_cls()
    be = BatchEvaluator(
        model_type="pdm_closed", checkpoint_path="none",
        scenario_root=str(tmp_path), output_root=str(tmp_path / "out"),
    )

    class _StubEvaluator:
        config = be.eval_config

        def setup(self, scenario_path):
            pass

        def run(self):
            return {"metrics": {"scorable": True,
                                "termination_reason": "budget_expired",
                                "driving_score": 0.5}}

    monkeypatch.setattr(be, "_get_evaluator", lambda: _StubEvaluator())
    scene = tmp_path / "scene_b"
    scene.mkdir()

    record = be.evaluate_scenario(scene)

    assert record["status"] == "success"
    assert "termination_reason" not in record
