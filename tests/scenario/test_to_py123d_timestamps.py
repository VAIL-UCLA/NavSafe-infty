"""Unit tests for ``_timestamps_us`` timestamp-unit handling.

``SD.TIMESTEP`` ("ts") is overloaded by three producers (scalar dt seconds,
absolute microsecond stamps, relative second stamps — see the convention table
in ``scenario_description``). The writer used to treat EVERY per-frame array as
microseconds, collapsing scenic-family second-valued series to 0,0,…,1,1.
These tests pin the unit classification for all three families. They need no
py123d install (the writer imports it lazily), so they run on the light CI
runner unlike the arrow round-trip suite.
"""

from __future__ import annotations

import numpy as np

from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.to_py123d_arrow import _timestamps_us


def _sd_with_ts(value) -> dict:
    return {SD.METADATA: {SD.TIMESTEP: value}}


def test_scenic_relative_seconds_array_scales_to_microseconds() -> None:
    # Relative timestamps: [0, dt, 2dt, ...] in seconds, float32.
    ts = np.array([i * 0.1 for i in range(15)], dtype=np.float32)
    out = _timestamps_us(_sd_with_ts(ts), 15)
    assert out == [i * 100_000 for i in range(15)]


def test_scenic_seconds_array_with_float32_jitter_stays_monotonic() -> None:
    ts = np.array([i * 0.1 for i in range(600)], dtype=np.float32)
    out = _timestamps_us(_sd_with_ts(ts), 600)
    assert all(b > a for a, b in zip(out, out[1:]))
    assert max(abs(v - i * 100_000) for i, v in enumerate(out)) <= 10  # sub-frame jitter only


def test_reader_absolute_microsecond_stamps_pass_through() -> None:
    # py123d_scenario_description family: absolute microsecond stamps.
    base = 316_000_000_000_000
    ts = np.array([base + i * 100_197 for i in range(5)], dtype=np.float64)
    out = _timestamps_us(_sd_with_ts(ts), 5)
    assert out == [base + i * 100_197 for i in range(5)]


def test_scalar_dt_seconds_expands_to_uniform_microsecond_steps() -> None:
    out = _timestamps_us(_sd_with_ts(0.1), 5)
    assert out == [0, 100_000, 200_000, 300_000, 400_000]


def test_missing_metadata_falls_back_to_default_step() -> None:
    out = _timestamps_us({SD.METADATA: {}}, 3)
    assert out == [0, 100_000, 200_000]


def test_single_frame_scalar_dt_is_not_misread_as_a_stamp() -> None:
    assert _timestamps_us(_sd_with_ts(0.1), 1) == [0]


def test_single_frame_absolute_stamp_passes_through() -> None:
    assert _timestamps_us(_sd_with_ts(np.array([316_000_000_000_000.0])), 1) == [316_000_000_000_000]


def test_zero_frames_yields_empty() -> None:
    assert _timestamps_us(_sd_with_ts(np.array([0.0, 0.1])), 0) == []


def test_short_stamp_array_falls_back_to_default_step() -> None:
    # Fewer stamps than frames: cannot be used as a per-frame series.
    out = _timestamps_us(_sd_with_ts(np.array([0.0, 0.1])), 4)
    assert out == [0, 100_000, 200_000, 300_000]


def test_dropped_frame_gap_does_not_flip_unit_classification() -> None:
    # A seconds series with one dropped frame: median positive spacing rules.
    ts = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6], dtype=np.float64)
    out = _timestamps_us(_sd_with_ts(ts), 6)
    assert out == [0, 100_000, 200_000, 400_000, 500_000, 600_000]
