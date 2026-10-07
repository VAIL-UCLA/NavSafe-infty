"""
navsafe.evaluation

Core evaluation framework for autonomous driving scenarios.

Components:
- Evaluator / EvaluationConfig: single-scenario evaluation orchestrator
- BatchEvaluator: multi-scenario batch evaluation with resume support
- models/: model adapter interface + concrete adapters (UniAD, VAD, TCP, RAP, ...)
- scorers/: trajectory candidate scorers (EPDMS, Confidence, CoarseTopK)
- scorers/epdms_trajectory_scorer_fast.py: EPDMSLiveScorer, the live per-frame metric scorer (BridgeSim parity)
- utils/: shared constants, comfort metrics, controller utilities
"""

from navsafe.evaluation.evaluator import Evaluator, EvaluationConfig
from navsafe.evaluation.batch_evaluator import BatchEvaluator
from navsafe.evaluation.evaluate import evaluate

__all__ = [
    "BatchEvaluator",
    "evaluate",
    "Evaluator",
    "EvaluationConfig",
]
