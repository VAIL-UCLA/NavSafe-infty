# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Size an asset by measurement, not by eye.

The historical workflow was: insert the PLY, render an episode, squint at the
gif, adjust a scale factor, repeat. This module replaces that loop with one
measurement and one solve:

1. **Measure the visual extent.** A gaussian is not a point: on screen it
   covers roughly ±2σ around its centre, and harvested assets carry a haze of
   large, near-transparent gaussians around the body. Sizing by the centre
   bounding box under-reads (the documented harvested-car failure: registry
   said 2.03 m wide, the render showed ~2.5 m). So the measurement prunes
   gaussians below an opacity floor, trims stray floaters by centre
   percentile, and then pads every survivor by ``sigma_k`` times its own σ.

2. **Solve one uniform scale** against target real-world dims — either given
   explicitly or looked up from the registry's per-family canonical table.
   The per-axis ratios are reported separately: if they disagree beyond
   ``PROPORTION_TOL`` the asset's *proportions* are off (wrong family, haze
   not pruned, or a clipped reconstruction) and no uniform scale can fix it —
   that is a finding, not something to average away silently.

The camera remains the final judge — a calibrated asset still gets one look in
a rendered frame — but it is one confirming look, not a search loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from navsafe.benchmark.editing.assets.ply_io import (
    read_3dgs_ply,
    reorient_gaussians,
    transform_gaussians,
    write_3dgs_ply,
)
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

# Rendered opacity below this is haze, not body: a harvested asset's faint
# outer gaussians read as nothing on screen but set the bounding box, which is
# what made a harvested car measure 2.03 m and render ~2.5 m wide. ``opacity``
# in the file is a logit.
DEFAULT_OPACITY_MIN = 0.1

# A gaussian reads on screen out to about ±2σ.
DEFAULT_SIGMA_K = 2.0

# Centres outside this two-sided percentile are stray floaters — reconstruction
# debris far from the body that would otherwise set the bounding box.
DEFAULT_CENTER_PERCENTILE = 99.5

# Per-axis scale ratios spreading beyond this mean the asset's proportions
# disagree with the target and a uniform scale cannot reconcile them.
PROPORTION_TOL = 1.15


class CalibrationError(NexusSimError, ValueError):
    """The asset cannot be calibrated with the information given."""


@dataclass
class Measurement:
    """The visual extent of one asset, in the y-up file frame."""

    dims: List[float]  # [length(x), width(z), height(y)] in file units
    base_y: float  # lowest visual point; 0 means base-origin, as inserts need
    total_gaussians: int
    kept_gaussians: int
    opacity_min: float
    sigma_k: float
    center_percentile: float

    @property
    def kept_fraction(self) -> float:
        return self.kept_gaussians / max(1, self.total_gaussians)

    def describe(self) -> str:
        return (
            f"visual extent l={self.dims[0]:.3f} w={self.dims[1]:.3f} h={self.dims[2]:.3f} "
            f"base_y={self.base_y:+.3f}  "
            f"({self.kept_gaussians}/{self.total_gaussians} gaussians kept: "
            f"opacity>{self.opacity_min:g}, centres within p{self.center_percentile:g}, "
            f"±{self.sigma_k:g}σ pad)"
        )


@dataclass
class Calibration:
    """The uniform scale that takes a measurement to its target dims."""

    measured: Measurement
    target_dims: List[float]
    scale: float
    axis_ratios: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def dims_after(self) -> List[float]:
        return [round(d * self.scale, 3) for d in self.measured.dims]

    def describe(self) -> str:
        lines = [
            self.measured.describe(),
            f"target dims      l={self.target_dims[0]:.3f} w={self.target_dims[1]:.3f} "
            f"h={self.target_dims[2]:.3f}",
            "axis ratios      "
            + "  ".join(f"{axis}={ratio:.3f}" for axis, ratio in self.axis_ratios.items()),
            f"uniform scale    {self.scale:.4f}",
            f"dims after       {self.dims_after}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def measure_visual_extent(
    gaussians: Dict[str, np.ndarray],
    *,
    opacity_min: float = DEFAULT_OPACITY_MIN,
    sigma_k: float = DEFAULT_SIGMA_K,
    center_percentile: float = DEFAULT_CENTER_PERCENTILE,
) -> Measurement:
    """Measure what the asset will actually cover on screen.

    Raises:
        CalibrationError: pruning leaves nothing to measure — the opacity floor
            ate the whole asset, which means the file is haze end to end.
    """
    xyz = np.asarray(gaussians["xyz"], np.float64)
    total = int(len(xyz))
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(gaussians["opacity"], np.float64)))
    keep = opacity >= float(opacity_min)
    if not np.any(keep):
        raise CalibrationError(
            f"no gaussian clears opacity {opacity_min:g}; the file is haze end to end "
            f"(max rendered opacity {float(opacity.max()):.3f})"
        )
    xyz = xyz[keep]
    sigma = np.exp(np.asarray(gaussians["scale"], np.float64))[keep]

    if center_percentile < 100.0:
        low = (100.0 - center_percentile) / 2.0
        lo, hi = np.percentile(xyz, [low, 100.0 - low], axis=0)
        inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        if np.any(inside):  # never trim into nothing
            xyz, sigma = xyz[inside], sigma[inside]

    pad = float(sigma_k) * sigma
    lo = (xyz - pad).min(axis=0)
    hi = (xyz + pad).max(axis=0)
    # File frame is y-up, +x forward: length is x, width is z, height is y.
    return Measurement(
        dims=[float(hi[0] - lo[0]), float(hi[2] - lo[2]), float(hi[1] - lo[1])],
        base_y=float(lo[1]),
        total_gaussians=total,
        kept_gaussians=int(len(xyz)),
        opacity_min=float(opacity_min),
        sigma_k=float(sigma_k),
        center_percentile=float(center_percentile),
    )


FIT_AXES = ("median", "length", "width", "height")


def solve_scale(measured: Measurement, target_dims: List[float],
                *, fit_axis: str = "median") -> Calibration:
    """One uniform scale from measured to target dims, warning on disagreement.

    ``fit_axis="median"`` takes the median of the three per-axis ratios, so one
    corrupted axis (a ground plane fattening the height, say) does not drag the
    other two. The disagreement itself is reported, not hidden.

    For an asset whose proportions genuinely differ from the canonical body,
    the median is the wrong anchor and the right one is domain knowledge:
    people vary little in HEIGHT and a lot in girth and pose, so a harvested
    pedestrian captured mid-stride (or with its legs clipped) should be sized
    by height — sizing it by the median rendered a 0.96 m adult, which reads
    as a child standing next to a car. Vehicles are the opposite: length is
    the stable dimension.
    """
    if len(target_dims) != 3 or any(d <= 0 for d in target_dims):
        raise CalibrationError(f"target dims must be three positive metres, got {target_dims}")
    if any(d <= 0 for d in measured.dims):
        raise CalibrationError(f"measured dims are degenerate: {measured.dims}")
    ratios = {
        axis: float(t) / float(m)
        for axis, t, m in zip(("length", "width", "height"), target_dims, measured.dims)
    }
    if fit_axis not in FIT_AXES:
        raise CalibrationError(f"fit_axis must be one of {list(FIT_AXES)}, got {fit_axis!r}")
    scale = float(np.median(list(ratios.values()))) if fit_axis == "median" else ratios[fit_axis]
    warnings: List[str] = []
    spread = max(ratios.values()) / min(ratios.values())
    if spread > PROPORTION_TOL:
        worst = max(ratios, key=lambda a: abs(np.log(ratios[a] / scale)))
        warnings.append(
            f"per-axis ratios disagree by {spread:.2f}x ({worst} is the outlier): the asset's "
            f"proportions do not match the target, so no uniform scale is right on every axis. "
            f"Check the family, or re-measure with a stricter opacity floor."
        )
    if measured.kept_fraction < 0.5:
        warnings.append(
            f"only {measured.kept_fraction:.0%} of gaussians clear the opacity floor — this "
            f"asset is mostly haze, and pruning it on write (--prune) is strongly advised."
        )
    if fit_axis != "median":
        warnings.append(
            f"scaled to match {fit_axis} exactly (fit_axis={fit_axis}); the other axes keep "
            f"the asset's own proportions")
    return Calibration(
        measured=measured, target_dims=[float(d) for d in target_dims], scale=scale,
        axis_ratios=ratios, warnings=warnings,
    )


def apply_calibration(
    gaussians: Dict[str, np.ndarray],
    calibration: Calibration,
    *,
    prune: bool = True,
    rebase: bool = True,
) -> Dict[str, np.ndarray]:
    """Produce the calibrated gaussians: prune, scale, and ground the base.

    Args:
        gaussians: the original asset.
        calibration: from :func:`solve_scale`.
        prune: drop the gaussians below the measurement's opacity floor, so the
            haze that was excluded from the measurement is also absent from the
            render.
        rebase: translate so the *visual* base sits at y=0 — the base-origin
            convention inserted assets need (the sim places the asset's origin
            on the road surface).
    """
    out = gaussians
    if prune:
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(out["opacity"], np.float64)))
        keep = opacity >= calibration.measured.opacity_min
        out = {k: np.asarray(v)[keep] for k, v in out.items()}
    out = transform_gaussians(out, scale=calibration.scale)
    if rebase:
        # Re-measure after prune+scale: the base moved with both.
        base_y = measure_visual_extent(
            out,
            opacity_min=calibration.measured.opacity_min,
            sigma_k=calibration.measured.sigma_k,
            center_percentile=calibration.measured.center_percentile,
        ).base_y
        out = transform_gaussians(out, translate=(0.0, -base_y, 0.0))
    return out


AXIS_CHOICES = ("+x", "-x", "+y", "-y", "+z", "-z")


def choose_axes(gaussians, target_dims, *, opacity_min=DEFAULT_OPACITY_MIN,
                sigma_k=DEFAULT_SIGMA_K, center_percentile=DEFAULT_CENTER_PERCENTILE):
    """Pick the (forward, up) that makes the asset's PROPORTIONS match the target.

    Harvested assets do not share one convention — the harvester writes each
    object in the frame its capture had, so a batch can carry cars with their
    length along +z, bicycles along +y, and animals along either. Guessing per
    family gets some of them silently sideways.

    The shape decides it instead: of the 24 proper (forward, up) pairs, the
    right one is the one under which length:width:height agrees with the
    target's. The runner-up's score is returned too — when the best two are
    close the asset is near-symmetric (a pedestrian is), and the choice is
    genuinely arbitrary rather than confidently right.

    Returns ``(forward, up, spread, runner_up_spread)`` where spread is the
    max/min axis-ratio (1.0 = perfect agreement).
    """
    from navsafe.benchmark.editing.assets.ply_io import reorient_gaussians

    scored = []
    for fwd in AXIS_CHOICES:
        for up in AXIS_CHOICES:
            if fwd[1] == up[1]:  # parallel axes cannot span a frame
                continue
            try:
                g = reorient_gaussians(gaussians, forward=fwd, up=up, sh_policy="keep")
                m = measure_visual_extent(g, opacity_min=opacity_min, sigma_k=sigma_k,
                                          center_percentile=center_percentile)
                ratios = [t / d for t, d in zip(target_dims, m.dims) if d > 0]
                if len(ratios) != 3:
                    continue
                scored.append((max(ratios) / min(ratios), fwd, up))
            except Exception:  # noqa: BLE001 - a degenerate orientation is just a bad candidate
                continue
    if not scored:
        raise CalibrationError("no orientation could be measured")
    scored.sort()
    best = scored[0]
    runner = next((s for s in scored[1:] if (s[1], s[2]) != (best[1], best[2])), best)
    return best[1], best[2], best[0], runner[0]


def calibrate_file(
    ply_path: "str | Path",
    target_dims: List[float],
    *,
    opacity_min: float = DEFAULT_OPACITY_MIN,
    sigma_k: float = DEFAULT_SIGMA_K,
    center_percentile: float = DEFAULT_CENTER_PERCENTILE,
    out_path: "str | Path | None" = None,
    prune: bool = True,
    rebase: bool = True,
    forward: "str | None" = None,
    up: "str | None" = None,
    auto_axes: bool = False,
    fit_axis: str = "median",
) -> Calibration:
    """Measure one PLY, solve its scale, and (optionally) write the result.

    Without ``out_path`` this only measures and reports — the dry run an author
    reads before deciding. With it, the calibrated PLY is written and then
    **re-measured from the file**, so the reported final dims are the file's
    own, not an extrapolation.

    Returns:
        The :class:`Calibration`; when written, ``calibration.measured`` still
        describes the ORIGINAL file and the log carries the re-measurement.
    """
    gaussians = read_3dgs_ply(ply_path)
    if auto_axes:
        forward, up, spread, runner = choose_axes(
            gaussians, target_dims, opacity_min=opacity_min, sigma_k=sigma_k,
            center_percentile=center_percentile)
        logger.info("calibrate: auto-axes picked forward=%s up=%s (proportion spread %.3f; "
                    "next best %.3f)", forward, up, spread, runner)
        if runner - spread < 0.02:
            logger.warning(
                "calibrate: the best two orientations score within 0.02 — this asset is "
                "near-symmetric, so its FACING is not recoverable from shape. Check the "
                "render, or set --forward-axis/--up-axis explicitly.")
    if forward or up:
        if not (forward and up):
            raise CalibrationError(
                "re-orienting needs BOTH --forward-axis and --up-axis: one axis alone "
                "does not determine the frame")
        gaussians = reorient_gaussians(gaussians, forward=forward, up=up)
        logger.info("calibrate: re-oriented (forward=%s, up=%s) into y-up/+x-forward",
                    forward, up)
    measured = measure_visual_extent(
        gaussians,
        opacity_min=opacity_min,
        sigma_k=sigma_k,
        center_percentile=center_percentile,
    )
    calibration = solve_scale(measured, target_dims, fit_axis=fit_axis)
    if out_path is None:
        return calibration
    out_path = Path(out_path)
    if out_path.resolve() == Path(ply_path).resolve():
        raise CalibrationError(
            "refusing to overwrite the source PLY: recipes pin assets by sha256, so mutating "
            "a file in place would orphan every recipe that referenced it. Write a new file "
            "and register it."
        )
    calibrated = apply_calibration(gaussians, calibration, prune=prune, rebase=rebase)
    write_3dgs_ply(calibrated, out_path)
    check = measure_visual_extent(
        calibrated, opacity_min=opacity_min, sigma_k=sigma_k, center_percentile=center_percentile
    )
    logger.info("calibrate: wrote %s — %s", out_path, check.describe())
    return calibration


__all__ = [
    "Calibration",
    "CalibrationError",
    "Measurement",
    "apply_calibration",
    "calibrate_file",
    "choose_axes",
    "measure_visual_extent",
    "solve_scale",
]
