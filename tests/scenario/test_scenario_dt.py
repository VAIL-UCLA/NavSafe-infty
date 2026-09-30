# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``SD.TIMESTEP`` parsing — the field carries three incompatible conventions.

Producers in this repo write, all into ``metadata['ts']``:

  1. a scalar dt in seconds      -- procgen, from_scenario_state
  2. ABSOLUTE microsecond stamps -- py123d (``log.timestamps_us``)
  3. RELATIVE second stamps      -- from_scenic (``[0, dt, 2dt, ...]``)

Consumers used to read ``ts[0]`` as a dt. That is right for (1), accidentally
survivable for (3), and catastrophic for (2) -- the dataset we run on -- where
it produced ``scenario_dt = 3.16e14``, hence ``gt_stride = 0``, hence EPDMS
scoring every candidate pose against agents FROZEN at the current frame, and
``route_horizon_s=8.0`` silently acting as 0.2 s.

These fixtures use the REAL shapes. An earlier version of this work tested
against an invented ``np.array([0.1])`` and passed while the real path was
broken -- hence the explicit absolute-timestamp regression below.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.scenario.scenario_description import scenario_dt_seconds


def _py123d_ts(n=157, dt_us=100197.0, t0=315969524359750.0):
    """py123d: absolute microsecond timestamps (real values from the data)."""
    return np.arange(n, dtype=np.float64) * dt_us + t0


def _scenic_ts(n=100, dt=0.1):
    """from_scenic: relative seconds, starting at 0."""
    return np.array([i * dt for i in range(n)], dtype=np.float32)


def test_py123d_absolute_microseconds():
    """The regression that mattered: absolute us stamps must give 0.1 s."""
    dt = scenario_dt_seconds({"ts": _py123d_ts()})
    assert dt == pytest.approx(0.100197, rel=1e-3)


def test_py123d_ts0_is_never_used_as_dt():
    """Guard the exact bug: ts[0] is an ORIGIN, not an interval."""
    ts = _py123d_ts()
    dt = scenario_dt_seconds({"ts": ts})
    assert dt < 1.0, (
        f"dt={dt:.6g} looks like ts[0]={ts[0]:.6g} -- reading the origin as a "
        f"timestep is what produced gt_stride=0 and froze agents")


def test_py123d_dt_yields_correct_gt_stride():
    """The consequence the scorer cares about: 0.5s / dt == 5, never 0."""
    dt = scenario_dt_seconds({"ts": _py123d_ts()})
    assert int(round(0.5 / dt)) == 5


def test_scenic_relative_seconds():
    assert scenario_dt_seconds({"ts": _scenic_ts()}) == pytest.approx(0.1, rel=1e-3)


def test_procgen_scalar_dt():
    assert scenario_dt_seconds({"ts": 0.1}) == pytest.approx(0.1)


def test_size_one_array_scalar_dt():
    assert scenario_dt_seconds({"ts": np.array([0.05])}) == pytest.approx(0.05)


def test_timestep_key_wins_over_ts():
    """Prior consumers preferred 'timestep'; preserve that precedence."""
    assert scenario_dt_seconds(
        {"timestep": 0.2, "ts": _py123d_ts()}) == pytest.approx(0.2)


@pytest.mark.parametrize("scale,unit", [
    (1.0, "s"), (1e3, "ms"), (1e6, "us"), (1e9, "ns"),
])
def test_unit_autodetection(scale, unit):
    """A 0.1 s cadence expressed in any of these units resolves to 0.1 s."""
    ts = np.arange(50, dtype=np.float64) * (0.1 * scale)
    assert scenario_dt_seconds({"ts": ts}) == pytest.approx(0.1, rel=1e-6), unit


def test_median_resists_a_dropped_frame():
    """One gap must not distort dt (why median, not mean or first-diff)."""
    ts = np.arange(50, dtype=np.float64) * 0.1
    ts[30:] += 5.0  # a 5 s hole
    assert scenario_dt_seconds({"ts": ts}) == pytest.approx(0.1, rel=1e-6)


@pytest.mark.parametrize("bad", [
    None, {}, {"ts": None}, {"ts": np.array([])},
    {"ts": np.zeros(10)},                    # stationary clock, no spacing
    {"ts": np.array([np.nan, np.nan])},
])
def test_degenerate_inputs_fall_back(bad):
    md = bad if isinstance(bad, dict) or bad is None else {"ts": bad}
    assert scenario_dt_seconds(md, default=0.1) == pytest.approx(0.1)


def test_fallback_warns_rather_than_silently_defaulting(caplog):
    """Silence is what let this survive; a bad dt must be loud."""
    import logging
    with caplog.at_level(logging.WARNING):
        scenario_dt_seconds({"ts": np.zeros(10)})
    assert any("scenario_dt" in r.message for r in caplog.records), (
        "a bad timestep must emit a warning")


def test_implausible_scalar_warns_and_defaults(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        dt = scenario_dt_seconds({"ts": 3.16e14})   # an absolute stamp as scalar
    assert dt == pytest.approx(0.1)
    assert any("scenario_dt" in r.message for r in caplog.records)


# --- Regressions from the Phase-1 code review -----------------------------
# Both were bugs the review caught in the FIRST version of this parser: it
# assumed "outside the plausible window => wrong units" and rescaled, which
# re-created the same silent-plausible-corruption it exists to prevent.

@pytest.mark.parametrize("dt_s", [2.0, 5.0, 1.5])
def test_seconds_above_one_are_not_silently_rescaled(dt_s):
    """A legitimate slow clock must survive.

    The first version rejected dt>1.0 s, then accepted scale=1e-3 -> a 2.0 s
    frame interval silently became 0.002 s (1000x error, no warning).
    """
    ts = np.arange(50, dtype=np.float64) * dt_s
    assert scenario_dt_seconds({"ts": ts}) == pytest.approx(dt_s), (
        "a seconds-valued dt above 1.0 s was rescaled as if it were ms")


def test_irregular_spacing_warns(caplog):
    """Held/duplicate stamps still consume a FRAME index.

    gt_stride indexes frames, so the clock-update interval can overstate the
    per-frame interval. Don't silently pick one -- say so.
    """
    import logging
    ts = np.repeat(np.arange(20, dtype=np.float64) * 0.1, 3)  # each stamp held x3
    with caplog.at_level(logging.WARNING):
        scenario_dt_seconds({"ts": ts})
    assert any("irregular" in r.message for r in caplog.records), (
        "duplicated stamps must warn: per-frame dt != clock-update interval")


def test_regular_series_does_not_warn(caplog):
    """The common path must stay quiet or the warning is worthless."""
    import logging
    with caplog.at_level(logging.WARNING):
        scenario_dt_seconds({"ts": _py123d_ts()})
    assert not [r for r in caplog.records if "scenario_dt" in r.message]


def test_unit_ladder_has_no_overlap():
    """Pin the invariant the ladder relies on.

    Units step by 1000x and only values above _DT_UNIT_THRESHOLD are
    converted, so no raw spacing can resolve under two different scales.
    """
    from navsafe.scenario import scenario_description as m
    assert m._DT_UNIT_THRESHOLD >= m._DT_MAX_S
    for scale in m._DT_UNIT_SCALES:
        assert scale <= 1e-3


def test_missing_metadata_warns(caplog):
    """Fallback must never be silent -- silence is why gt_stride=0 survived."""
    import logging
    with caplog.at_level(logging.WARNING):
        scenario_dt_seconds({"ts": None})
    assert any("scenario_dt" in r.message for r in caplog.records)
