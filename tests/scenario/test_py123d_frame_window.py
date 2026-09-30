# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A windowed load must be a SPARSE scene, not a different one.

This conversion emits one py123d scene per nuPlan log (650-5320 iterations),
so loading every iteration to simulate 156 of them made worker RSS scale with
the log drawn rather than the work done -- measured at 34-50 GB, and the direct
cause of the host OOMs. The window fixes that, but only if a frame loaded
inside it is byte-identical to the same frame loaded from a full scene, and
still keyed by its ABSOLUTE iteration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from navsafe.scenario.py123d_adapter import (
    Py123DAdapterConfig,
    scenario_from_py123d_scene,
)


class _FakeScene:
    """Minimal duck-typed SceneAPI; counts payload reads."""

    def __init__(self, n: int = 500) -> None:
        self.n = n
        self.reads = 0
        self.number_of_iterations = n
        self.number_of_history_iterations = 0
        self.log_name = "2021.01.01.00.00.00_veh-01_00000_00001"
        self.scene_uuid = "uuid-1"

    # --- metadata surface -------------------------------------------------
    def get_scene_metadata(self):
        return SimpleNamespace(initial_uuid="uuid-1")

    def get_log_metadata(self):
        return SimpleNamespace(log_name=self.log_name, dataset="nuplan", split="val")

    def get_map_metadata(self):
        return SimpleNamespace(map_name="us-nv-las-vegas-strip")

    def get_all_iteration_timestamps(self, *_a, **_k):
        return [SimpleNamespace(time_us=i * 100_000) for i in range(self.n)]

    def get_all_modality_metadatas(self, *_a, **_k):
        return []

    # --- payload surface --------------------------------------------------
    def get_ego_state_se3_at_iteration(self, iteration):
        self.reads += 1
        return SimpleNamespace(timestamp=SimpleNamespace(time_us=iteration * 100_000),
                               iteration=iteration)

    def get_ego_state_se3_metadata(self):
        return SimpleNamespace(modality="ego_state_se3")

    def get_all_ego_state_se3_timestamps(self, *_a, **_k):
        return [SimpleNamespace(time_us=i * 100_000) for i in range(self.n)]

    def __getattr__(self, name):           # tolerate the adapter's optional probes
        if name.startswith(("get_", "available_")):
            return lambda *a, **k: None
        raise AttributeError(name)


def _ego_frames(scenario):
    for record in scenario.modalities.values():
        if record.modality_type == "ego_state_se3":
            return record.frames
    return {}


def _window_cfg(window):
    return Py123DAdapterConfig(load_map_objects=False, require_map=False,
                               frame_window=window)


def test_window_loads_only_its_iterations():
    scene = _FakeScene(500)
    scenario = scenario_from_py123d_scene(scene, _window_cfg((100, 156)))
    frames = _ego_frames(scenario)
    assert set(frames) == set(range(100, 156))
    assert min(frames) == 100 and max(frames) == 155


def test_frames_keep_absolute_iteration_keys():
    """A windowed scene must not renumber -- consumers index absolutely."""
    scene = _FakeScene(500)
    frames = _ego_frames(scenario_from_py123d_scene(scene, _window_cfg((300, 320))))
    assert 0 not in frames, "window was renumbered to zero-based"
    assert frames[300].iteration == 300
    assert frames[300].data.iteration == 300


def test_windowed_frame_matches_the_full_load():
    """The whole point: identical content, fewer of them."""
    full = scenario_from_py123d_scene(
        _FakeScene(500),
        Py123DAdapterConfig(load_map_objects=False, require_map=False))
    windowed = scenario_from_py123d_scene(_FakeScene(500), _window_cfg((200, 210)))
    f_full, f_win = _ego_frames(full), _ego_frames(windowed)
    for i in range(200, 210):
        assert f_win[i].timestamp_us == f_full[i].timestamp_us
        assert f_win[i].data.iteration == f_full[i].data.iteration


def test_window_cuts_payload_reads_proportionally():
    """The memory/IO win, asserted rather than assumed."""
    full_scene = _FakeScene(500)
    scenario_from_py123d_scene(
        full_scene, Py123DAdapterConfig(load_map_objects=False, require_map=False))
    win_scene = _FakeScene(500)
    scenario_from_py123d_scene(win_scene, _window_cfg((0, 50)))
    assert win_scene.reads * 5 <= full_scene.reads, (
        f"expected ~10x fewer reads, got {full_scene.reads} vs {win_scene.reads}")


def test_no_window_is_the_previous_behaviour():
    scene = _FakeScene(500)
    scenario = scenario_from_py123d_scene(
        scene, Py123DAdapterConfig(load_map_objects=False, require_map=False))
    assert set(_ego_frames(scenario)) == set(range(500))


def test_window_is_clamped_to_the_scene():
    """A mined window comes from nuPlan's clock; it must never index past."""
    scene = _FakeScene(100)
    frames = _ego_frames(scenario_from_py123d_scene(scene, _window_cfg((80, 5000))))
    assert max(frames) == 99


def test_an_empty_window_fails_loudly():
    with pytest.raises(ValueError, match="selects no iterations"):
        scenario_from_py123d_scene(_FakeScene(100), _window_cfg((500, 600)))
