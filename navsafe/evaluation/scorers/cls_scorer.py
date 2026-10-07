import torch
from typing import Dict, Any

from navsafe.evaluation.scorers.base_scorer import BaseTrajectoryScorer


class ClsScorer(BaseTrajectoryScorer):
    """
    Scorer that uses the model's classification confidence scores to select
    the best trajectory candidate. Selects the candidate with the highest
    confidence score per batch element via argmax.
    """

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Select the trajectory with the highest confidence score.

        Args:
            model_output: dict containing:
                - "confidence_scores": (B, N) confidence scores per candidate
                - "all_candidates": (B, N, T, 3) trajectory candidates

        Returns:
            dict with:
                - "trajectory": (B, T, 3) selected best trajectory
                - "scores": (B, N) all candidate scores
                - "best_idx": (B,) index of selected candidate

        Raises:
            ValueError: If 'confidence_scores' is not found in model_output.
        """
        scores = model_output.get("confidence_scores")
        if scores is None:
            raise ValueError("confidence_scores not found in model_output")

        candidates = model_output["all_candidates"]  # (B, N, T, 3)
        bs = candidates.shape[0]
        device = candidates.device

        best_idx = scores.argmax(dim=-1)  # (B,)
        best_traj = candidates[
            torch.arange(bs, device=device), best_idx
        ]  # (B, T, 3)

        return {
            "trajectory": best_traj,
            "scores": scores,
            "best_idx": best_idx,
        }
