"""Tests for navsafe.scenario.py123d_scenes.enumerate_scenes.

The helper centralizes the py123d "fetch filtered scenes → cap → seeded
shuffle" logic shared by the runtime loader, the online-RL sampler, and the
BC cache builder. py123d is imported lazily inside the helper, so these tests
monkeypatch ``py123d.api`` to drive it with fake scene handles.
"""

from __future__ import annotations

import sys
import types

from navsafe.scenario.py123d_scenes import enumerate_scenes


def _fake_py123d(monkeypatch, scenes) -> None:
    """Inject a fake ``py123d.api`` so the helper's lazy import yields fakes.

    The suite runs without the real py123d installed, so we register a stub
    module rather than patch attributes on an importable one.
    """
    api = types.ModuleType("py123d.api")
    api.SceneFilter = lambda *a, **k: object()
    api.get_filtered_scenes = lambda *a, **k: list(scenes)
    pkg = types.ModuleType("py123d")
    pkg.api = api
    monkeypatch.setitem(sys.modules, "py123d", pkg)
    monkeypatch.setitem(sys.modules, "py123d.api", api)


def test_returns_all_scenes_in_order(monkeypatch) -> None:
    _fake_py123d(monkeypatch, ["s0", "s1", "s2"])
    assert enumerate_scenes("/data") == ["s0", "s1", "s2"]


def test_cap_keeps_prefix(monkeypatch) -> None:
    _fake_py123d(monkeypatch, ["s0", "s1", "s2", "s3"])
    assert enumerate_scenes("/data", max_scenes=2) == ["s0", "s1"]


def test_no_shuffle_preserves_order(monkeypatch) -> None:
    _fake_py123d(monkeypatch, list("abcde"))
    assert enumerate_scenes("/data", shuffle=False) == list("abcde")


def test_shuffle_is_deterministic_under_seed(monkeypatch) -> None:
    _fake_py123d(monkeypatch, list(range(20)))
    first = enumerate_scenes("/data", shuffle=True, seed=7)
    second = enumerate_scenes("/data", shuffle=True, seed=7)
    assert first == second                      # reproducible
    assert sorted(first) == list(range(20))     # permutation, nothing lost
    assert first != list(range(20))             # actually reordered


def test_empty_root_returns_empty_no_raise(monkeypatch) -> None:
    """Empty-handling is the caller's policy; the helper just returns []."""
    _fake_py123d(monkeypatch, [])
    assert enumerate_scenes("/data") == []
