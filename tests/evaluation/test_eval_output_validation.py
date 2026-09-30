"""Completion validation for scored, excluded, partial, and visualized episodes."""
import json
import pytest
from PIL import Image
from navsafe.tools.output_validation import main, validate_episode


def episode(path, **updates):
    path.mkdir(parents=True, exist_ok=True)
    data = dict(status="scored", frames=dict(total=12, scored=10, warmup_excluded=2),
                termination=dict(reason="contact_at_fault"),
                metrics=dict(driving_score=0.0, success=False, efficiency_pct=None, comfort=None))
    data.update(updates)
    (path / "navsafe_metrics.json").write_text(json.dumps(data))
    return path


def test_no_vis_zero_score_and_null_optional_metrics_are_valid(tmp_path):
    assert validate_episode(episode(tmp_path))["passed"]


@pytest.mark.parametrize("updates", [dict(status="excluded"), dict(frames={}), dict(termination=None),
    dict(metrics=dict(driving_score=float("nan"), success=False, efficiency_pct=None, comfort=None)),
    dict(frames=dict(total=5, scored=10, warmup_excluded=2)), dict(metrics={})])
def test_invalid_or_excluded_episode_fails(tmp_path, updates):
    assert not validate_episode(episode(tmp_path, **updates))["passed"]


def test_missing_and_malformed_final_metrics(tmp_path):
    assert not validate_episode(tmp_path)["passed"]
    (tmp_path / "navsafe_metrics.json").write_text('[]')
    assert not validate_episode(tmp_path)["passed"]


def test_visualization_is_explicit_and_decoded(tmp_path):
    episode(tmp_path)
    assert not validate_episode(tmp_path, require_vis=True)["passed"]
    frames=tmp_path/'frames'/'00000';frames.mkdir(parents=True)
    vis=tmp_path/'visualization';vis.mkdir()
    for p in [frames/'cam_f0.jpg', frames/'topdown.png', vis/'cam_f0.gif', vis/'topdown.gif']:
        Image.new('RGB',(16,12)).save(p)
    assert validate_episode(tmp_path, require_vis=True)["passed"]
    (frames/'cam_f0.jpg').write_bytes(b'not an image')
    assert not validate_episode(tmp_path)["passed"]


def test_recursive_scan_includes_partial_runs_and_arbitrary_names(tmp_path):
    episode(tmp_path/'model'/'seed1'/'token')
    assert main(['--model-dir',str(tmp_path)])==0
    partial=tmp_path/'partial';partial.mkdir();(partial/'eval.log').write_text('failed')
    assert main(['--model-dir',str(tmp_path)])==1


def test_empty_scan_fails(tmp_path):
    assert main(['--model-dir',str(tmp_path)])==1
