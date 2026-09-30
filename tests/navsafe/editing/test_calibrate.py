# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Asset calibration: size by measured visual extent, not by eye."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.assets.calibrate import (
    CalibrationError,
    apply_calibration,
    calibrate_file,
    measure_visual_extent,
    solve_scale,
)
from navsafe.benchmark.editing.assets.ply_io import read_3dgs_ply, write_3dgs_ply
from navsafe.benchmark.editing.assets.registry import AssetRegistry


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _gaussians(
    n: int = 400,
    *,
    extent=(4.0, 1.5, 1.6),  # x (length), y (height), z (width) in file frame
    sigma: float = 0.01,
    opacity: float = 0.9,
    seed: int = 0,
) -> dict:
    """A dense box of small, opaque gaussians whose visual size ≈ its centre bbox."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, 1.0, size=(n, 3)) * np.asarray(extent)
    xyz[:, 1] -= 0.0  # base at y=0
    # Guarantee the corners exist so the extent is exactly `extent`.
    xyz[0] = (0.0, 0.0, 0.0)
    xyz[1] = extent
    return {
        "xyz": xyz.astype(np.float32),
        "normals": np.zeros((n, 3), np.float32),
        "f_dc": np.zeros((n, 3), np.float32),
        "f_rest": np.zeros((n, 0), np.float32),
        "opacity": np.full(n, _logit(opacity), np.float32),
        "scale": np.full((n, 3), np.log(sigma), np.float32),
        "rot": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
    }


def _with_haze(gaussians: dict, *, m: int = 60, sigma: float = 0.5, reach: float = 3.0) -> dict:
    """Surround the body with big, nearly transparent gaussians (harvested haze)."""
    rng = np.random.default_rng(1)
    haze_xyz = rng.uniform(-reach, reach, size=(m, 3)).astype(np.float32)
    out = {}
    for key, arr in gaussians.items():
        if key == "xyz":
            add = haze_xyz
        elif key == "opacity":
            add = np.full(m, _logit(0.02), np.float32)
        elif key == "scale":
            add = np.full((m, 3), np.log(sigma), np.float32)
        elif key == "rot":
            add = np.tile(np.array([1, 0, 0, 0], np.float32), (m, 1))
        else:
            add = np.zeros((m,) + arr.shape[1:], np.float32)
        out[key] = np.concatenate([arr, add], axis=0)
    return out


class TestMeasurement:
    def test_visual_extent_of_a_clean_body(self):
        m = measure_visual_extent(_gaussians())
        # length = x, width = z, height = y, plus a little ±2σ pad.
        assert m.dims[0] == pytest.approx(4.0, abs=0.1)
        assert m.dims[1] == pytest.approx(1.6, abs=0.1)
        assert m.dims[2] == pytest.approx(1.5, abs=0.1)
        assert m.base_y == pytest.approx(0.0, abs=0.05)

    def test_haze_is_excluded_by_the_opacity_floor(self):
        clean = measure_visual_extent(_gaussians())
        hazed = measure_visual_extent(_with_haze(_gaussians()))
        assert hazed.dims == pytest.approx(clean.dims, abs=0.05)
        assert hazed.kept_gaussians == clean.kept_gaussians

    def test_sigma_padding_grows_the_blobby_measurement(self):
        # Same centres, fat gaussians: the centre bbox under-reads exactly the
        # way the harvested car did; the visual extent must not.
        thin = measure_visual_extent(_gaussians(sigma=0.01))
        fat = measure_visual_extent(_gaussians(sigma=0.15))
        assert fat.dims[1] > thin.dims[1] + 0.4

    def test_all_haze_refuses(self):
        g = _gaussians(opacity=0.02)
        with pytest.raises(CalibrationError, match="haze end to end"):
            measure_visual_extent(g)


class TestSolve:
    def test_uniform_scale_is_the_median_ratio(self):
        m = measure_visual_extent(_gaussians(extent=(8.0, 3.0, 3.2)))  # 2x a 4.0/1.5/1.6 car
        cal = solve_scale(m, [4.0, 1.6, 1.5])
        assert cal.scale == pytest.approx(0.5, abs=0.02)
        assert not cal.warnings

    def test_disagreeing_proportions_warn_rather_than_average_silently(self):
        m = measure_visual_extent(_gaussians(extent=(8.0, 1.5, 1.6)))  # stretched only in x
        cal = solve_scale(m, [4.0, 1.6, 1.5])
        assert any("proportions" in w for w in cal.warnings)


class TestApplyAndFile:
    def test_apply_scales_prunes_and_rebases(self):
        g = _with_haze(_gaussians(extent=(8.0, 3.0, 3.2)))
        cal = solve_scale(measure_visual_extent(g), [4.0, 1.6, 1.5])
        out = apply_calibration(g, cal)
        m = measure_visual_extent(out)
        assert m.dims[0] == pytest.approx(4.0, abs=0.1)
        assert m.base_y == pytest.approx(0.0, abs=1e-3)
        assert len(out["xyz"]) < len(g["xyz"])  # haze pruned

    def test_calibrate_file_round_trip(self, tmp_path):
        src = tmp_path / "src.ply"
        dst = tmp_path / "calibrated.ply"
        write_3dgs_ply(_gaussians(extent=(8.0, 3.0, 3.2)), src)
        cal = calibrate_file(src, [4.0, 1.6, 1.5], out_path=dst)
        assert dst.is_file()
        m = measure_visual_extent(read_3dgs_ply(dst))
        assert m.dims == pytest.approx([d for d in cal.dims_after], abs=0.05)

    def test_refuses_to_overwrite_the_source(self, tmp_path):
        src = tmp_path / "src.ply"
        write_3dgs_ply(_gaussians(), src)
        with pytest.raises(CalibrationError, match="refusing to overwrite"):
            calibrate_file(src, [4.0, 1.6, 1.5], out_path=src)


class TestFamilies:
    def test_the_shipped_registry_carries_canonical_family_dims(self):
        registry = AssetRegistry.load()
        assert registry.family_dims("car") == [4.6, 1.85, 1.5]

    def test_heterogeneous_families_are_absent_on_purpose(self):
        registry = AssetRegistry.load()
        with pytest.raises(Exception, match="target-dims"):
            registry.family_dims("animal")
