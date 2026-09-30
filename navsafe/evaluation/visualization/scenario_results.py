"""Per-scenario evaluation result visualization.

Provides summary tables, heatmaps, metric histograms, filtering/sorting,
and consistent file naming for per-scenario EPDMS evaluation results.

All plots use matplotlib with the ``Agg`` backend for headless rendering.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ── Data Model ──────────────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    """Per-scenario evaluation result with EPDMS metrics."""

    scenario_id: str
    scorer_type: str
    metrics: Dict[str, float]  # EPDMS metric values
    best_idx: int
    passed: bool  # no collision and on-road
    failure_reason: Optional[str] = None  # "collision" or "off_road"


# ── File Naming ─────────────────────────────────────────────────────────────


def get_output_filename(
    scenario_id: str,
    scorer_type: str,
    viz_type: str,
    ext: str,
) -> str:
    """Build a consistent output filename.

    Args:
        scenario_id: Scenario identifier.
        scorer_type: Scorer type string (e.g. ``"gt"``, ``"tta"``).
        viz_type: Visualization type (e.g. ``"summary"``, ``"heatmap"``).
        ext: File extension without leading dot (e.g. ``"png"``).

    Returns:
        Filename string ``{scenario_id}_{scorer_type}_{viz_type}.{ext}``.
    """
    return f"{scenario_id}_{scorer_type}_{viz_type}.{ext}"


# ── Filtering and Sorting ───────────────────────────────────────────────────


def filter_results(
    results: List[ScenarioResult],
    *,
    scorer_type: Optional[str] = None,
    passed: Optional[bool] = None,
    metric_name: Optional[str] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> List[ScenarioResult]:
    """Filter scenario results by predicate.

    All provided filters are combined with logical AND.

    Args:
        results: Input list of scenario results.
        scorer_type: Keep only results with this scorer type.
        passed: Keep only results matching this pass/fail status.
        metric_name: Metric key to filter on (requires *min_value* or
            *max_value*).
        min_value: Minimum metric value (inclusive).
        max_value: Maximum metric value (inclusive).

    Returns:
        Filtered list of :class:`ScenarioResult`.
    """
    filtered = list(results)

    if scorer_type is not None:
        filtered = [r for r in filtered if r.scorer_type == scorer_type]

    if passed is not None:
        filtered = [r for r in filtered if r.passed is passed]

    if metric_name is not None:
        if min_value is not None:
            filtered = [
                r for r in filtered
                if metric_name in r.metrics and r.metrics[metric_name] >= min_value
            ]
        if max_value is not None:
            filtered = [
                r for r in filtered
                if metric_name in r.metrics and r.metrics[metric_name] <= max_value
            ]

    return filtered


def sort_results(
    results: List[ScenarioResult],
    by: str,
    ascending: bool = True,
) -> List[ScenarioResult]:
    """Sort scenario results by a metric value or attribute.

    Args:
        results: Input list of scenario results.
        by: Sort key — either a metric name present in ``metrics`` dict,
            or one of ``"scenario_id"``, ``"scorer_type"``, ``"passed"``.
        ascending: Sort direction.

    Returns:
        Sorted list of :class:`ScenarioResult`.
    """
    if by in ("scenario_id", "scorer_type", "passed"):
        key_fn = lambda r: getattr(r, by)
    else:
        # Sort by metric value; results missing the metric go to the end
        key_fn = lambda r: r.metrics.get(by, float("inf") if ascending else float("-inf"))

    return sorted(results, key=key_fn, reverse=not ascending)


# ── Summary Table ───────────────────────────────────────────────────────────


def render_summary_table(
    results: List[ScenarioResult],
    output_path: str,
) -> None:
    """Render a summary table of per-scenario results as a PNG image.

    Columns: scenario_id, scorer_type, each EPDMS metric, best_idx,
    passed, failure_reason.  Failed rows are highlighted in red.

    Args:
        results: Scenario results to tabulate.
        output_path: File path for the saved PNG.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Collect all metric keys across results
    all_metric_keys = sorted(
        {k for r in results for k in r.metrics}
    )

    # Build column headers and cell data
    col_labels = (
        ["scenario_id", "scorer_type"]
        + all_metric_keys
        + ["best_idx", "passed", "failure_reason"]
    )

    cell_text = []
    cell_colors = []
    for r in results:
        row = [r.scenario_id, r.scorer_type]
        for mk in all_metric_keys:
            val = r.metrics.get(mk)
            row.append(f"{val:.3f}" if val is not None else "—")
        row.extend([str(r.best_idx), str(r.passed), r.failure_reason or ""])
        cell_text.append(row)

        # Highlight failed rows
        if not r.passed:
            cell_colors.append(["#ffcccc"] * len(col_labels))
        else:
            cell_colors.append(["#ffffff"] * len(col_labels))

    fig_width = max(8, len(col_labels) * 1.2)
    fig_height = max(2, (len(results) + 1) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#d9e2f3"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    ax.set_title("Per-Scenario Evaluation Summary", fontsize=11, pad=10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary table → %s", output_path)


# ── Heatmap ─────────────────────────────────────────────────────────────────


def render_heatmap(
    results: List[ScenarioResult],
    output_path: str,
) -> None:
    """Render a heatmap of per-metric scores across scenarios.

    Rows are scenarios, columns are EPDMS metrics.  Failed scenarios
    are annotated with an ``X`` marker.

    Args:
        results: Scenario results to visualize.
        output_path: File path for the saved PNG.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    all_metric_keys = sorted({k for r in results for k in r.metrics})
    n_scenarios = len(results)
    n_metrics = len(all_metric_keys)

    data = np.full((n_scenarios, n_metrics), np.nan)
    scenario_labels = []
    for i, r in enumerate(results):
        scenario_labels.append(r.scenario_id)
        for j, mk in enumerate(all_metric_keys):
            if mk in r.metrics:
                data[i, j] = r.metrics[mk]

    fig_width = max(6, n_metrics * 0.8 + 2)
    fig_height = max(3, n_scenarios * 0.4 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(all_metric_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(scenario_labels, fontsize=8)

    # Mark failed scenarios
    for i, r in enumerate(results):
        if not r.passed:
            for j in range(n_metrics):
                ax.text(j, i, "X", ha="center", va="center",
                        color="red", fontsize=8, fontweight="bold")

    ax.set_title("Per-Metric Scores Heatmap", fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap → %s", output_path)


# ── Metric Histograms ───────────────────────────────────────────────────────


def render_metric_histograms(
    results: List[ScenarioResult],
    output_path: str,
    bins: int = 20,
) -> None:
    """Render histogram distributions of each EPDMS metric.

    Creates a grid of subplots, one histogram per metric.

    Args:
        results: Scenario results to visualize.
        output_path: File path for the saved PNG.
        bins: Number of histogram bins.
    """
    if not results:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No results", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    all_metric_keys = sorted({k for r in results for k in r.metrics})
    n_metrics = len(all_metric_keys)

    if n_metrics == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, mk in enumerate(all_metric_keys):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        values = [r.metrics[mk] for r in results if mk in r.metrics]
        if values:
            ax.hist(values, bins=bins, color="#4a90d9", edgecolor="white", alpha=0.8)
        ax.set_title(mk, fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle("EPDMS Metric Distributions", fontsize=12)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metric histograms → %s", output_path)
