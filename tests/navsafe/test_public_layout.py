"""Public dataset layout and relocation contracts."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from navsafe.benchmark import config as cfg
from navsafe.benchmark.editing.recipe.schema import load_recipe, RecipeError
from navsafe.benchmark.editing.recipe.replay import _rebase_asset
from navsafe.benchmark.editing.assets.registry import AssetEntry
from navsafe.benchmark.world import deployment


def test_data_root_defaults_and_override(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NAVSAFE_")}
    env["NAVSAFE_DATA_ROOT"] = str(tmp_path)
    code = "import json; from navsafe.benchmark import config as c; print(json.dumps([str(c.BUNDLES),str(c.ASSET_BANK),str(c.GAIT_BANK),str(c.MODEL_ZOO)]))"
    got = json.loads(subprocess.check_output([sys.executable, "-c", code], env=env))
    assert got == [str(tmp_path / p) for p in ("full_test", "asset", "gait_bank", "model_zoo")]
    env["NAVSAFE_ASSET_BANK"] = str(tmp_path / "custom")
    got = json.loads(subprocess.check_output([sys.executable, "-c", code], env=env))
    assert got[1] == str(tmp_path / "custom")


def test_fetch_preserves_hf_layout_and_unrelated_files(tmp_path, monkeypatch):
    from navsafe.benchmark.eval.fetch_bundle import fetch
    token = "a2c2e046132e5596"
    dest = tmp_path / "full_test" / token
    dest.mkdir(parents=True)
    (dest / "local-note").write_text("keep")
    def download(repo, **kwargs):
        assert kwargs["local_dir"] == str(tmp_path)
        (dest / "manifest.json").write_text("{}")
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=download))
    assert fetch("example/dataset", token, tmp_path) == dest
    assert (dest / "local-note").read_text() == "keep"
    assert not (tmp_path / token).exists()


def test_asset_relocation_checks_content(tmp_path, monkeypatch):
    bank = tmp_path / "asset"
    bank.mkdir()
    (bank / "example.ply").write_bytes(b"geometry")
    monkeypatch.setattr(cfg, "ASSET_BANK", bank)
    spec = {"nurec_asset_id": "asset/example.ply",
            "asset_sha256": hashlib.sha256(b"geometry").hexdigest()}
    _rebase_asset(spec, "actor")
    assert spec["nurec_asset_id"] == str(bank / "example.ply")
    assert AssetEntry(key="example", ply="asset/example.ply").path == bank / "example.ply"
    (bank / "example.ply").write_bytes(b"different geometry")
    with pytest.raises(RecipeError, match="not the asset"):
        _rebase_asset({"nurec_asset_id": "asset/example.ply",
                       "asset_sha256": hashlib.sha256(b"geometry").hexdigest()}, "actor")


def test_kubernetes_defaults_are_cluster_independent(monkeypatch):
    for key in ("NAVSAFE_PVC_MOUNTS", "NAVSAFE_NODES", "NAVSAFE_TOLERATIONS",
                "NAVSAFE_IMAGE_PULL_SECRET"):
        monkeypatch.delenv(key, raising=False)
    from navsafe.benchmark.world.k8s_jobs import _pod_spec
    pod = _pod_spec("example/image", args=["run"])
    assert "affinity" not in pod and "tolerations" not in pod
    assert all("persistentVolumeClaim" not in v for v in pod["volumes"])
    monkeypatch.setenv("NAVSAFE_PVC_MOUNTS", json.dumps([{"claim": "test-data", "mountPath": "/example"}]))
    monkeypatch.setenv("NAVSAFE_NODES", "node-a,node-b")
    pod = _pod_spec("example/image", args=["run"])
    assert pod["volumes"][-1]["persistentVolumeClaim"]["claimName"] == "test-data"
    assert pod["containers"][0]["volumeMounts"][-1]["mountPath"] == "/example"
    values = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    assert values == ["node-a", "node-b"]


def test_published_recipes_are_integrity_checked_and_relative():
    root = Path(__file__).parents[2] / "navsafe/benchmark/recipes"
    paths = sorted(root.rglob("*.yaml"))
    assert len(paths) >= 202
    for path in paths:
        recipe = load_recipe(path)
        for actor in recipe.actors.values():
            if actor.nurec_asset_id:
                assert actor.nurec_asset_id.startswith("asset/"), path
                assert len(Path(actor.nurec_asset_id).parts) == 2, path
            if actor.nurec_pose_bank:
                assert actor.nurec_pose_bank.startswith("gait_bank/"), path


def test_freeze_does_not_publish_authoring_bank_paths(tmp_path, monkeypatch):
    from navsafe.benchmark.editing.recipe.freeze import freeze_recipe
    source = Path(__file__).parents[2] / "navsafe/benchmark/recipes/benchmark/R-4.a2c2e046132e5596.yaml"
    recipe = load_recipe(source)
    monkeypatch.setattr(cfg, "ASSET_BANK", tmp_path / "asset")
    monkeypatch.setattr(cfg, "GAIT_BANK", tmp_path / "gait_bank")
    for actor in recipe.actors.values():
        if actor.nurec_asset_id:
            actor.nurec_asset_id = str(tmp_path / actor.nurec_asset_id)
        if actor.nurec_pose_bank:
            actor.nurec_pose_bank = str(tmp_path / actor.nurec_pose_bank)
    out = freeze_recipe(recipe, tmp_path / "recipe.yaml")
    actual = load_recipe(out)
    assert str(tmp_path) not in out.read_text()
    for name, actor in actual.actors.items():
        assert actor.asset.sha256 == recipe.actors[name].asset.sha256


def test_renderer_and_harvest_use_user_cluster_settings(tmp_path, monkeypatch, capsys):
    from navsafe.benchmark.world.serve_grpc import manifest
    from navsafe.benchmark.harvest.cli import cmd_batch
    import yaml
    monkeypatch.setattr(cfg, "NAMESPACE", "user-namespace")
    monkeypatch.setenv("NAVSAFE_PVC_MOUNTS", '[{"claim":"dataset","mountPath":"/custom-data"}]')
    docs = manifest("renderer", "/custom-data/full_test/*.usdz", enable_harmonizer=False)
    assert all(d["metadata"]["namespace"] == "user-namespace" for d in docs)
    assert docs[0]["spec"]["template"]["spec"]["volumes"][-1]["persistentVolumeClaim"]["claimName"] == "dataset"
    args = types.SimpleNamespace(scenes=["token"], scenes_file=None, workers=1,
        name="harvest", image="example/image", max_assets=2, nodes="node-a", out="-")
    assert cmd_batch(args) == 0
    doc = yaml.safe_load(capsys.readouterr().out)
    assert doc["metadata"]["namespace"] == "user-namespace"
    pod = doc["spec"]["template"]["spec"]
    assert pod["containers"][0]["volumeMounts"][-1]["mountPath"] == "/custom-data"
