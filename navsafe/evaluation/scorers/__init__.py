"""Trajectory scorers.

Scorer classes are re-exported lazily (PEP 562): most scorer modules import
torch at module level, but this package also hosts import-light shared
helpers (``gt_stride``, ``at_fault``) that must stay importable on machines
without the heavy stack (the CI quality workflow runs without torch).
``from navsafe.evaluation.scorers import EPDMSTrajectoryScorer_Fast`` still
works unchanged; the torch import simply happens at first attribute access
instead of at package-import time.
"""

from importlib import import_module

_SCORER_MODULES = {
    "BaseTrajectoryScorer": "base_scorer",
    "ConfidenceScorer": "confidence_scorer",
    "ClsScorer": "cls_scorer",
    "CoarseTopKScorer": "coarse_topk_scorer",
    # The legacy EPDMSTrajectoryScorer and EPDMSEgoScorer were deleted
    # (simplify.md, scorer consolidation Phase 0): dormant all campaign,
    # each carrying a divergent copy of every EPDMS term. Phase 2 absorbed
    # the live metric of record (formerly utils/epdms_scorer_md.py) into
    # this module, as EPDMSLiveScorer, beside the batch engine: one file
    # owns both public behaviors of the metric.
    "EPDMSTrajectoryScorer_Fast": "epdms_trajectory_scorer_fast",
    "EPDMSLiveScorer": "epdms_trajectory_scorer_fast",
    "GtScorer": "gt_scorer",
    "LearnedScorer": "learned_scorer",
    "TtaScorer": "tta_scorer",
}

__all__ = list(_SCORER_MODULES)


def __getattr__(name: str):
    module_name = _SCORER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module_name}"), name)
