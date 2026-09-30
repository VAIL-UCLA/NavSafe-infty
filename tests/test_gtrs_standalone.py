"""Standalone GTRS dependency and evaluation-contract checks."""
import ast
import hashlib
from pathlib import Path
import numpy as np
import pytest

from navsafe.modelzoo.gtrs_dense.hydra_config import HydraConfig
from navsafe.policy.sensor.gtrs_dense import GTRSDenseAdapter

ROOT = Path(__file__).resolve().parents[1] / "navsafe" / "modelzoo" / "gtrs_dense"

def test_inference_has_no_external_simscale_or_nuplan_imports():
    for path in ROOT.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] in {"navsim", "nuplan", "nexussim"} for n in names), path

@pytest.mark.parametrize("size", [8192, 16384])
def test_official_vocabulary_contract(size):
    expected = {
        8192: "cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35",
        16384: "e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d",
    }
    assert hashlib.sha256((ROOT / f"{size}.npy").read_bytes()).hexdigest() == expected[size]
    vocab = np.load(ROOT / f"{size}.npy", allow_pickle=False)
    assert vocab.shape == (size, HydraConfig().trajectory_sampling.num_poses, 3)
    assert vocab.dtype == np.float32
    assert np.isfinite(vocab).all()

def test_invalid_vocabulary_fails_before_checkpoint_loading(tmp_path, monkeypatch):
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros((3, 40, 3), dtype=np.float32))
    monkeypatch.setenv("NAVSAFE_GTRS_VOCAB", str(p))
    monkeypatch.setenv("NAVSAFE_GTRS_VOCAB_SIZE", "8192")
    a = GTRSDenseAdapter("missing-checkpoint.ckpt")
    with pytest.raises(ValueError, match="8192 finite trajectories"):
        a.load_model()

def test_inference_camera_and_waypoint_contract():
    a = GTRSDenseAdapter("unused.ckpt")
    assert list(a.get_camera_configs()) == ["CAM_F0", "CAM_L0", "CAM_R0"]
    assert a.get_waypoint_dt() == 0.1
