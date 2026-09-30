# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""EnvCfg.py123d_frame_window must actually reach the adapter.

The window is only worth anything if it survives the trip from config to
``Py123DAdapterConfig``; a silently-dropped field would leave the full-log
load (and the OOM) in place while every unit test still passed.
"""

from __future__ import annotations

import navsafe.env.loaders.py123d as loader_mod
from navsafe.env.env_cfg import EnvCfg


class _Recorder:
    """Captures the Py123DAdapterConfig the loader builds."""

    def __init__(self):
        self.seen = None

    def __call__(self, scene, config):
        self.seen = config
        raise _Stop()


class _Stop(Exception):
    pass


def _run(monkeypatch, cfg_window):
    rec = _Recorder()
    monkeypatch.setattr(loader_mod, "scenario_from_py123d_scene", rec)
    monkeypatch.setattr(loader_mod.Py123DLoader, "_resolve_data_root",
                        lambda self, cfg: "/nonexistent")
    monkeypatch.setattr(loader_mod.Py123DLoader, "_load_scenes",
                        lambda self, cfg, root: ["scene"])
    monkeypatch.setattr(loader_mod.Py123DLoader, "_select_scene",
                        lambda self, cfg, scenes: "scene")
    cfg = EnvCfg(py123d_frame_window=cfg_window)
    try:
        loader_mod.Py123DLoader().load(cfg)
    except _Stop:
        pass
    return rec.seen


def test_window_reaches_the_adapter(monkeypatch):
    seen = _run(monkeypatch, (100, 256))
    assert seen is not None, "loader never called the adapter"
    assert seen.frame_window == (100, 256)


def test_absent_window_keeps_the_full_load(monkeypatch):
    seen = _run(monkeypatch, None)
    assert seen.frame_window is None


def test_env_cfg_defaults_to_no_window():
    assert EnvCfg().py123d_frame_window is None
