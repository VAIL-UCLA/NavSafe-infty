"""Tests for navsafe.scenario.py123d_dataset (py123d-backed dataset helpers).

These replace the legacy ScenarioNet ``ScenarioLoader`` list/load-by-index API.
py123d is faked via ``sys.modules`` (the suite runs without it installed).
"""

from __future__ import annotations

import sys
import types

import pytest

from navsafe.scenario.py123d_dataset import list_scene_ids, scenario_description_by_index, scene_id


class _Scene:
    def __init__(self, uuid=None, log_name=None):
        if uuid is not None:
            self.scene_uuid = uuid
        if log_name is not None:
            self.log_name = log_name


def _fake_py123d(monkeypatch, scenes) -> None:
    api = types.ModuleType("py123d.api")
    api.SceneFilter = lambda *a, **k: object()
    api.get_filtered_scenes = lambda *a, **k: list(scenes)
    pkg = types.ModuleType("py123d")
    pkg.api = api
    monkeypatch.setitem(sys.modules, "py123d", pkg)
    monkeypatch.setitem(sys.modules, "py123d.api", api)


def test_scene_id_prefers_uuid_then_log_name() -> None:
    assert scene_id(_Scene(uuid="abc")) == "abc"
    assert scene_id(_Scene(log_name="log_7")) == "log_7"
    assert scene_id(_Scene()) == ""


def test_list_scene_ids_in_discovery_order(monkeypatch) -> None:
    _fake_py123d(monkeypatch, [_Scene(uuid="s0"), _Scene(log_name="s1"), _Scene(uuid="s2")])
    assert list_scene_ids("/data") == ["s0", "s1", "s2"]


def test_scenario_description_by_index_bounds(monkeypatch) -> None:
    _fake_py123d(monkeypatch, [_Scene(uuid="s0")])
    with pytest.raises(IndexError):
        scenario_description_by_index("/data", 5)


def test_scenario_description_by_index_empty_root(monkeypatch) -> None:
    _fake_py123d(monkeypatch, [])
    with pytest.raises(FileNotFoundError):
        scenario_description_by_index("/data", 0)
