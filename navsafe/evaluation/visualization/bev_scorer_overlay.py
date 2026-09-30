"""BEV visualization with scorer result overlays.

Renders bird's-eye-view images with candidate trajectories color-coded
by normalized score, best-trajectory highlighting, text annotations,
and multi-scorer grid layouts.  Supports PNG and MP4 output.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from navsafe.evaluation.vis_utils import (
    BEV_PPM,
    BEV_SIZE,
    render_bev,
)

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class ScorerOverlayData:
    """Input data for a single scorer overlay on a BEV image."""

    scenario_id: str
    scorer_name: str
    candidates: np.ndarray  # (N, T, 3) — x, y, heading per timestep
    scores: np.ndarray  # (N,)
    best_idx: int
    ego_state: Dict[str, Any]
    agent_states: List[Dict] = field(default_factory=list)
    road_boundaries: List[np.ndarray] = field(default_factory=list)
    lane_markings: List[np.ndarray] = field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range. Returns 0.5 for constant scores."""
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (scores - s_min) / (s_max - s_min)
    return np.full_like(scores, 0.5, dtype=np.float64)


def _score_color_bgr(t: float) -> Tuple[int, int, int]:
    """Map normalized score *t* ∈ [0, 1] to BGR colour (red=low, green=high)."""
    # Red (0,0,255) → Green (0,255,0)
    r = int(255 * (1.0 - t))
    g = int(255 * t)
    return (0, g, r)  # BGR


def _world_to_px(
    wx: float, wy: float, ego_x: float, ego_y: float
) -> Tuple[int, int]:
    """Convert world coordinates to BEV pixel coordinates."""
    cx, cy = BEV_SIZE // 2, BEV_SIZE // 2
    dx = wx - ego_x
    dy = wy - ego_y
    return int(cx + dx * BEV_PPM), int(cy - dy * BEV_PPM)


# ── Core rendering ──────────────────────────────────────────────────────────

def render_scorer_overlay(
    data: ScorerOverlayData,
    image_size: int = BEV_SIZE,
) -> np.ndarray:
    """Render a BEV image with scored candidate trajectory overlays.

    - Candidates are color-coded red (low) → green (high) by normalized score.
    - The best trajectory is drawn thick & solid; others thin & dashed.
    - Text annotations show scorer name, best_idx, and best score.

    Args:
        data: Scorer overlay data containing candidates, scores, and scene info.
        image_size: Output image size (square). Defaults to ``BEV_SIZE``.

    Returns:
        ``(H, W, 3)`` uint8 BGR image.
    """
    ego_pos = np.asarray(data.ego_state.get("position", [0.0, 0.0, 0.0]),
                         dtype=np.float64)[:2]
    ego_heading = float(data.ego_state.get("heading", 0.0))

    # Base BEV (map + agents, no trajectory)
    img = render_bev(
        ego_position=ego_pos,
        ego_heading=ego_heading,
        planned_traj_world=None,
        vehicle_states=None,
        frame_id=0,
        model_name=data.scorer_name,
    )

    n_cands = len(data.candidates)
    if n_cands == 0:
        return img

    norm_scores = _normalize_scores(data.scores.astype(np.float64))

    # Draw non-selected candidates first (thin dashed)
    for ci in range(n_cands):
        if ci == data.best_idx:
            continue
        _draw_trajectory(
            img, data.candidates[ci], ego_pos,
            color=_score_color_bgr(float(norm_scores[ci])),
            thickness=1, dashed=True,
        )

    # Draw best trajectory on top (thick solid)
    best_color = _score_color_bgr(float(norm_scores[data.best_idx]))
    _draw_trajectory(
        img, data.candidates[data.best_idx], ego_pos,
        color=best_color, thickness=3, dashed=False,
    )

    # Text annotations
    _draw_annotations(img, data)

    # Resize if needed
    if image_size != BEV_SIZE:
        img = cv2.resize(img, (image_size, image_size))

    return img


def _draw_trajectory(
    img: np.ndarray,
    traj: np.ndarray,
    ego_pos: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 1,
    dashed: bool = False,
) -> None:
    """Draw a single trajectory on the BEV image."""
    pts = []
    for wp in traj:
        px, py = _world_to_px(float(wp[0]), float(wp[1]),
                               float(ego_pos[0]), float(ego_pos[1]))
        pts.append((px, py))

    if len(pts) < 2:
        return

    if dashed:
        # Draw every other segment for dashed effect
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, pts[i], pts[min(i + 1, len(pts) - 1)],
                     color, thickness, cv2.LINE_AA)
    else:
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], False, color, thickness, cv2.LINE_AA)


def _draw_annotations(img: np.ndarray, data: ScorerOverlayData) -> None:
    """Draw scorer name, best_idx, and best score as text annotations."""
    best_score = float(data.scores[data.best_idx])

    # Semi-transparent background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (5, BEV_SIZE - 70), (300, BEV_SIZE - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.putText(img, f"Scorer: {data.scorer_name}",
                (10, BEV_SIZE - 50), _FONT, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"Best idx: {data.best_idx}  Score: {best_score:.4f}",
                (10, BEV_SIZE - 30), _FONT, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"Candidates: {len(data.candidates)}  Red=Low Green=High",
                (10, BEV_SIZE - 12), _FONT, 0.35, (200, 200, 200), 1)


# ── Multi-scorer grid ───────────────────────────────────────────────────────

def render_multi_scorer_grid(
    data_list: List[ScorerOverlayData],
    max_cols: int = 2,
    cell_size: int = 400,
) -> np.ndarray:
    """Render a grid of BEV scorer overlays for side-by-side comparison.

    Args:
        data_list: List of scorer overlay data (one per scorer).
        max_cols: Maximum columns in the grid.
        cell_size: Size of each cell (square) in pixels.

    Returns:
        ``(H, W, 3)`` uint8 BGR image containing the grid.
    """
    n = len(data_list)
    if n == 0:
        return np.full((cell_size, cell_size, 3), 30, dtype=np.uint8)

    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    grid_h = rows * cell_size
    grid_w = cols * cell_size
    grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

    for idx, data in enumerate(data_list):
        r, c = divmod(idx, cols)
        cell_img = render_scorer_overlay(data, image_size=cell_size)
        y0 = r * cell_size
        x0 = c * cell_size
        grid[y0 : y0 + cell_size, x0 : x0 + cell_size] = cell_img

    return grid


# ── Output saving ───────────────────────────────────────────────────────────

def save_overlay_png(image: np.ndarray, path: str) -> None:
    """Save a BEV overlay image as PNG.

    Args:
        image: ``(H, W, 3)`` uint8 BGR image.
        path: Output file path (should end with ``.png``).

    Raises:
        ValueError: If *path* does not end with ``.png``.
    """
    if not path.lower().endswith(".png"):
        raise ValueError(f"Unsupported format: {os.path.splitext(path)[1]}. Use png")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)
    logger.info("Saved PNG overlay → %s", path)


def save_overlay_mp4(
    frames: List[np.ndarray],
    path: str,
    fps: int = 10,
) -> None:
    """Save a sequence of BEV overlay frames as an MP4 video.

    Args:
        frames: List of ``(H, W, 3)`` uint8 BGR images.
        path: Output file path (should end with ``.mp4``).
        fps: Frames per second.

    Raises:
        ValueError: If *path* does not end with ``.mp4`` or *frames* is empty.
    """
    if not path.lower().endswith(".mp4"):
        raise ValueError(f"Unsupported format: {os.path.splitext(path)[1]}. Use mp4")
    if not frames:
        raise ValueError("Cannot create MP4 from empty frame list")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # cv2 has no stubs
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    logger.info("Saved MP4 overlay (%d frames) → %s", len(frames), path)
