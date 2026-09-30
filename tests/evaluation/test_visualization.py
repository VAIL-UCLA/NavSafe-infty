"""Tests for BEV visualization with scorer overlays.

Verifies PNG and MP4 file creation, non-empty output, and multi-scorer grid.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from navsafe.evaluation.visualization.bev_scorer_overlay import (
    ScorerOverlayData,
    render_multi_scorer_grid,
    render_scorer_overlay,
    save_overlay_mp4,
    save_overlay_png,
)


def _make_overlay_data(
    scorer_name: str = "test_scorer",
    n_candidates: int = 5,
    timesteps: int = 10,
) -> ScorerOverlayData:
    """Create synthetic ScorerOverlayData for testing."""
    rng = np.random.RandomState(42)
    candidates = rng.randn(n_candidates, timesteps, 3).astype(np.float32)
    scores = rng.rand(n_candidates).astype(np.float32)
    best_idx = int(scores.argmax())
    return ScorerOverlayData(
        scenario_id="test_scenario_001",
        scorer_name=scorer_name,
        candidates=candidates,
        scores=scores,
        best_idx=best_idx,
        ego_state={"position": [0.0, 0.0, 0.0], "heading": 0.0},
        agent_states=[],
        road_boundaries=[],
        lane_markings=[],
    )


class TestRenderScorerOverlay:
    """Tests for render_scorer_overlay."""

    def test_returns_valid_image(self):
        data = _make_overlay_data()
        img = render_scorer_overlay(data)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_custom_image_size(self):
        data = _make_overlay_data()
        img = render_scorer_overlay(data, image_size=400)
        assert img.shape[:2] == (400, 400)

    def test_empty_candidates(self):
        data = ScorerOverlayData(
            scenario_id="empty",
            scorer_name="test",
            candidates=np.empty((0, 10, 3), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            best_idx=0,
            ego_state={"position": [0.0, 0.0, 0.0], "heading": 0.0},
        )
        img = render_scorer_overlay(data)
        assert img.shape[2] == 3


class TestMultiScorerGrid:
    """Tests for render_multi_scorer_grid."""

    def test_grid_shape(self):
        data_list = [_make_overlay_data(f"scorer_{i}") for i in range(4)]
        grid = render_multi_scorer_grid(data_list, max_cols=2, cell_size=200)
        assert grid.shape == (400, 400, 3)

    def test_single_scorer(self):
        data_list = [_make_overlay_data()]
        grid = render_multi_scorer_grid(data_list, max_cols=2, cell_size=200)
        assert grid.shape == (200, 200, 3)

    def test_empty_list(self):
        grid = render_multi_scorer_grid([], cell_size=200)
        assert grid.shape == (200, 200, 3)


class TestSavePNG:
    """Tests for save_overlay_png — verifies PNG file creation."""

    def test_png_created(self, tmp_path):
        data = _make_overlay_data()
        img = render_scorer_overlay(data)
        out_path = str(tmp_path / "overlay.png")
        save_overlay_png(img, out_path)
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 0

    def test_invalid_extension_raises(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported format"):
            save_overlay_png(img, str(tmp_path / "bad.jpg"))


class TestSaveMP4:
    """Tests for save_overlay_mp4 — verifies MP4 file creation."""

    def test_mp4_created(self, tmp_path):
        data = _make_overlay_data()
        frames = [render_scorer_overlay(data) for _ in range(5)]
        out_path = str(tmp_path / "overlay.mp4")
        save_overlay_mp4(frames, out_path, fps=10)
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 0

    def test_invalid_extension_raises(self, tmp_path):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match="Unsupported format"):
            save_overlay_mp4(frames, str(tmp_path / "bad.avi"))

    def test_empty_frames_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty frame list"):
            save_overlay_mp4([], str(tmp_path / "empty.mp4"))


# ── Per-Scenario Evaluation Result Visualization Tests ──────────────────────

from navsafe.evaluation.visualization.scenario_results import (
    ScenarioResult,
    filter_results,
    get_output_filename,
    render_heatmap,
    render_metric_histograms,
    render_summary_table,
    sort_results,
)


def _make_scenario_results(n: int = 5, seed: int = 0) -> list[ScenarioResult]:
    """Create a list of synthetic ScenarioResult objects for testing."""
    rng = np.random.RandomState(seed)
    scorer_types = ["gt", "tta", "cls"]
    metric_keys = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]
    results = []
    for i in range(n):
        passed = rng.rand() > 0.3
        results.append(
            ScenarioResult(
                scenario_id=f"scenario_{i:03d}",
                scorer_type=scorer_types[i % len(scorer_types)],
                metrics={k: float(rng.rand()) for k in metric_keys},
                best_idx=int(rng.randint(0, 5)),
                passed=bool(passed),
                failure_reason=None if passed else rng.choice(["collision", "off_road"]),
            )
        )
    return results


class TestSummaryTable:
    """Tests for render_summary_table."""

    def test_creates_png(self, tmp_path):
        results = _make_scenario_results(5)
        out = str(tmp_path / "summary.png")
        render_summary_table(results, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_empty_results(self, tmp_path):
        out = str(tmp_path / "empty_summary.png")
        render_summary_table([], out)
        assert os.path.isfile(out)

    def test_single_result(self, tmp_path):
        results = _make_scenario_results(1)
        out = str(tmp_path / "single_summary.png")
        render_summary_table(results, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0


class TestHeatmap:
    """Tests for render_heatmap."""

    def test_creates_png(self, tmp_path):
        results = _make_scenario_results(5)
        out = str(tmp_path / "heatmap.png")
        render_heatmap(results, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_empty_results(self, tmp_path):
        out = str(tmp_path / "empty_heatmap.png")
        render_heatmap([], out)
        assert os.path.isfile(out)


class TestMetricHistograms:
    """Tests for render_metric_histograms."""

    def test_creates_png(self, tmp_path):
        results = _make_scenario_results(10)
        out = str(tmp_path / "histograms.png")
        render_metric_histograms(results, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_empty_results(self, tmp_path):
        out = str(tmp_path / "empty_hist.png")
        render_metric_histograms([], out)
        assert os.path.isfile(out)


class TestFilterResults:
    """Tests for filter_results."""

    def test_filter_by_scorer_type(self):
        results = _make_scenario_results(9)
        filtered = filter_results(results, scorer_type="gt")
        assert all(r.scorer_type == "gt" for r in filtered)
        assert len(filtered) == sum(1 for r in results if r.scorer_type == "gt")

    def test_filter_by_passed(self):
        results = _make_scenario_results(10)
        filtered = filter_results(results, passed=True)
        assert all(r.passed for r in filtered)

    def test_filter_by_metric_range(self):
        results = _make_scenario_results(10)
        filtered = filter_results(
            results, metric_name="NC", min_value=0.3, max_value=0.7
        )
        for r in filtered:
            assert 0.3 <= r.metrics["NC"] <= 0.7

    def test_combined_filters(self):
        results = _make_scenario_results(20)
        filtered = filter_results(results, scorer_type="gt", passed=True)
        assert all(r.scorer_type == "gt" and r.passed for r in filtered)

    def test_no_match_returns_empty(self):
        results = _make_scenario_results(5)
        filtered = filter_results(results, scorer_type="nonexistent")
        assert filtered == []


class TestSortResults:
    """Tests for sort_results."""

    def test_sort_by_metric_ascending(self):
        results = _make_scenario_results(10)
        sorted_r = sort_results(results, by="NC", ascending=True)
        values = [r.metrics["NC"] for r in sorted_r]
        assert values == sorted(values)

    def test_sort_by_metric_descending(self):
        results = _make_scenario_results(10)
        sorted_r = sort_results(results, by="NC", ascending=False)
        values = [r.metrics["NC"] for r in sorted_r]
        assert values == sorted(values, reverse=True)

    def test_sort_by_scenario_id(self):
        results = _make_scenario_results(5)
        sorted_r = sort_results(results, by="scenario_id", ascending=True)
        ids = [r.scenario_id for r in sorted_r]
        assert ids == sorted(ids)

    def test_sort_by_passed(self):
        results = _make_scenario_results(10)
        sorted_r = sort_results(results, by="passed", ascending=True)
        vals = [r.passed for r in sorted_r]
        assert vals == sorted(vals)


class TestGetOutputFilename:
    """Tests for get_output_filename."""

    def test_basic_naming(self):
        name = get_output_filename("scen_001", "gt", "summary", "png")
        assert name == "scen_001_gt_summary.png"

    def test_different_ext(self):
        name = get_output_filename("scen_002", "tta", "heatmap", "pdf")
        assert name == "scen_002_tta_heatmap.pdf"

    def test_components_in_name(self):
        name = get_output_filename("abc", "cls", "histogram", "png")
        assert "abc" in name
        assert "cls" in name
        assert "histogram" in name
        assert name.endswith(".png")


# ── Property-Based Tests for Scenario Results ───────────────────────────────

from hypothesis import given, settings, assume
import hypothesis.strategies as st


def _scenario_result_strategy():
    """Hypothesis strategy for generating ScenarioResult objects."""
    scorer_types = st.sampled_from(["gt", "tta", "cls", "learned"])
    metric_keys = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]

    return st.builds(
        ScenarioResult,
        scenario_id=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
            min_size=1,
            max_size=20,
        ),
        scorer_type=scorer_types,
        metrics=st.fixed_dictionaries(
            {k: st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False) for k in metric_keys}
        ),
        best_idx=st.integers(min_value=0, max_value=10),
        passed=st.booleans(),
        failure_reason=st.one_of(st.none(), st.sampled_from(["collision", "off_road"])),
    )


# Feature: bridgesim-to-navsafe-migration, Property 14: Scenario result filtering and sorting
# **Validates: Requirements 13.5**
@given(
    results=st.lists(_scenario_result_strategy(), min_size=0, max_size=30),
    filter_scorer=st.one_of(st.none(), st.sampled_from(["gt", "tta", "cls", "learned"])),
    filter_passed=st.one_of(st.none(), st.booleans()),
    sort_metric=st.sampled_from(["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]),
    ascending=st.booleans(),
)
@settings(max_examples=100)
def test_scenario_filter_sort_property(
    results, filter_scorer, filter_passed, sort_metric, ascending
):
    """Property 14: Filtered output contains exactly matching results,
    and sorted output is in the specified order."""

    # Apply filter
    filtered = filter_results(
        results, scorer_type=filter_scorer, passed=filter_passed
    )

    # Verify filter correctness: every result in filtered matches predicates
    for r in filtered:
        if filter_scorer is not None:
            assert r.scorer_type == filter_scorer
        if filter_passed is not None:
            assert r.passed is filter_passed

    # Verify completeness: every result in original that matches is in filtered
    expected_count = sum(
        1 for r in results
        if (filter_scorer is None or r.scorer_type == filter_scorer)
        and (filter_passed is None or r.passed is filter_passed)
    )
    assert len(filtered) == expected_count

    # Apply sort
    sorted_r = sort_results(filtered, by=sort_metric, ascending=ascending)

    # Verify sort order
    assert len(sorted_r) == len(filtered)
    values = [r.metrics.get(sort_metric, float("inf") if ascending else float("-inf")) for r in sorted_r]
    if ascending:
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]
    else:
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    # Verify no elements lost or gained
    assert set(id(r) for r in sorted_r) == set(id(r) for r in filtered)


# Feature: bridgesim-to-navsafe-migration, Property 15: Visualization file naming convention
# **Validates: Requirements 13.6**
@given(
    scenario_id=st.text(
        alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_"),
        min_size=1,
        max_size=30,
    ),
    scorer_type=st.sampled_from(["gt", "tta", "cls", "learned"]),
    viz_type=st.sampled_from(["summary", "heatmap", "histogram"]),
    ext=st.sampled_from(["png", "pdf", "svg"]),
)
@settings(max_examples=100)
def test_viz_naming_convention_property(scenario_id, scorer_type, viz_type, ext):
    """Property 15: Output filename follows {scenario_id}_{scorer_type}_{viz_type}.{ext}."""

    filename = get_output_filename(scenario_id, scorer_type, viz_type, ext)

    # Verify exact format
    expected = f"{scenario_id}_{scorer_type}_{viz_type}.{ext}"
    assert filename == expected

    # Verify components are recoverable from the filename
    assert filename.startswith(scenario_id + "_")
    assert filename.endswith(f".{ext}")
    # The middle part should contain scorer_type and viz_type
    stem = filename[: -len(f".{ext}")]
    parts = stem.split("_", 1)  # split off scenario_id prefix
    assert len(parts) >= 2
