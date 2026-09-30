"""Visualization subpackage for evaluation result rendering.

Provides BEV scorer overlays and per-scenario result visualizations.
"""

from navsafe.evaluation.visualization.bev_scorer_overlay import (
    ScorerOverlayData,
    render_scorer_overlay,
    render_multi_scorer_grid,
    save_overlay_png,
    save_overlay_mp4,
)
from navsafe.evaluation.visualization.scenario_results import (
    ScenarioResult,
    get_output_filename,
    filter_results,
    sort_results,
    render_summary_table,
    render_heatmap,
    render_metric_histograms,
)

__all__ = [
    "ScorerOverlayData",
    "render_scorer_overlay",
    "render_multi_scorer_grid",
    "save_overlay_png",
    "save_overlay_mp4",
    "ScenarioResult",
    "get_output_filename",
    "filter_results",
    "sort_results",
    "render_summary_table",
    "render_heatmap",
    "render_metric_histograms",
]
