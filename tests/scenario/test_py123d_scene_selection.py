# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""UUID selection filters Arrow scenes without reading the entire root.

Unknown UUIDs must raise rather than silently returning fewer scenes.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from navsafe.scenario import py123d_scenes


class _Scene:
    def __init__(self, uuid: str) -> None:
        self.scene_uuid = uuid


def _install_fake_py123d(monkeypatch: pytest.MonkeyPatch, uuids: list[str]) -> dict[str, Any]:
    """Stand in for py123d.api, recording the filter it was handed."""
    seen: dict[str, Any] = {}

    class SceneFilter:
        def __init__(self, *, scene_uuids: list[str] | None = None) -> None:
            self.scene_uuids = scene_uuids

    def get_filtered_scenes(scene_filter: Any, *, data_root: str) -> list[_Scene]:
        seen["scene_uuids"] = scene_filter.scene_uuids
        seen["data_root"] = data_root
        wanted = scene_filter.scene_uuids
        return [_Scene(u) for u in uuids if wanted is None or u in wanted]

    module = types.ModuleType("py123d.api")
    module.SceneFilter = SceneFilter  # type: ignore[attr-defined]
    module.get_filtered_scenes = get_filtered_scenes  # type: ignore[attr-defined]
    package = types.ModuleType("py123d")
    package.api = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "py123d", package)
    monkeypatch.setitem(sys.modules, "py123d.api", module)
    return seen


ALL = ["aaa", "bbb", "ccc", "ddd"]


def test_no_uuids_reads_the_whole_root(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_fake_py123d(monkeypatch, ALL)
    scenes = py123d_scenes.enumerate_scenes("/root")
    assert [py123d_scenes.scene_id(s) for s in scenes] == ALL
    assert seen["scene_uuids"] is None


def test_selection_is_pushed_into_the_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not a post-hoc filter: a targeted round must not read the whole root."""
    seen = _install_fake_py123d(monkeypatch, ALL)
    scenes = py123d_scenes.enumerate_scenes("/root", scene_uuids=["ccc", "aaa"])
    assert seen["scene_uuids"] == ["ccc", "aaa"]
    # Sorted, so episode index -> scene is reproducible across machines.
    assert [py123d_scenes.scene_id(s) for s in scenes] == ["aaa", "ccc"]


def test_a_uuid_the_root_lacks_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A curriculum that quietly collected the wrong scenes is unauditable."""
    _install_fake_py123d(monkeypatch, ALL)
    with pytest.raises(ValueError, match="not in /root"):
        py123d_scenes.enumerate_scenes("/root", scene_uuids=["aaa", "zzz"])


def test_an_empty_selection_selects_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct from None: [] is 'no scenes', not 'every scene'."""
    seen = _install_fake_py123d(monkeypatch, ALL)
    assert py123d_scenes.enumerate_scenes("/root", scene_uuids=[]) == []
    assert "scene_uuids" not in seen, "an empty selection must not query the root"


def test_max_scenes_and_shuffle_still_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_py123d(monkeypatch, ALL)
    capped = py123d_scenes.enumerate_scenes("/root", scene_uuids=ALL, max_scenes=2)
    assert [py123d_scenes.scene_id(s) for s in capped] == ["aaa", "bbb"]
    shuffled = py123d_scenes.enumerate_scenes("/root", scene_uuids=ALL, shuffle=True, seed=7)
    assert sorted(py123d_scenes.scene_id(s) for s in shuffled) == ALL
