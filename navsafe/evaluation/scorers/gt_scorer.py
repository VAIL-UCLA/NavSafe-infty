"""
Ground-truth EPDMS trajectory scorer for inference scaling.

Scores each candidate trajectory using the full EPDMS metric suite
(NC, DAC, DDC, TLC, EP, TTC, LK, HC, EC) against scenario ground-truth data,
then selects the candidate with the highest composite score.

Scoring runs through the one EPDMS batch engine
(:class:`~navsafe.evaluation.scorers.epdms_trajectory_scorer_fast.EPDMSTrajectoryScorer_Fast`);
the legacy per-candidate ``EPDMSScorer.score_frame`` window path was deleted
with ``evaluation/utils/epdms_scorer_md.py`` (scorer consolidation,
simplify.md Phase 2). Candidates are interpreted in the model/ego frame, the
same convention the fast engine's ``select_best`` applies to
``all_candidates``. Scores are NOT comparable to pre-consolidation
GtScorer output: the old path read candidates as world-frame waypoints,
forgave terms the logged human also violated (the "human filter"),
carried EC state across candidates, and ignored pedestrians for NC.
"""

import torch
from typing import Dict, Any, Optional

from navsafe.evaluation.scorers.base_scorer import BaseTrajectoryScorer
from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSTrajectoryScorer_Fast,
)

# Mapping from the fast engine's short metric keys to the reported EPDMS keys
_METRIC_KEY_MAP = {
    "nc": "NC",
    "dac": "DAC",
    "ddc": "DDC",
    "tlc": "TLC",
    "ep": "EP",
    "ttc": "TTC",
    "lk": "LK",
    "hc": "HC",
    "ec": "EC",
}

EPDMS_METRIC_KEYS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "EC"]

REQUIRED_SCENARIO_FIELDS = ["tracks", "map_features", "metadata"]


class GtScorer(BaseTrajectoryScorer):
    """
    Ground-truth scorer that evaluates trajectory candidates using the full
    EPDMS metric suite against scenario data. Selects the candidate with the
    highest composite score per batch element.
    """

    def __init__(self):
        self.epdms: Optional[EPDMSTrajectoryScorer_Fast] = None

    def initialize(self, scenario_data: dict, env) -> None:
        """
        Initialize the EPDMS batch engine with scenario ground-truth data.

        Args:
            scenario_data: Scenario dict containing tracks, map_features, metadata.
            env: The simulation environment instance.

        Raises:
            ValueError: If scenario_data is missing required fields.
        """
        missing = [f for f in REQUIRED_SCENARIO_FIELDS if f not in scenario_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        self.epdms = EPDMSTrajectoryScorer_Fast(verbose=False)
        self.epdms.initialize(scenario_data, env)

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Score each candidate via EPDMS metrics and select the highest composite.

        Args:
            model_output: dict with 'all_candidates' of shape (B, N, T, 3).
            **kwargs: Must include 'ego_state' and 'frame_idx'.

        Returns:
            dict with:
                - 'trajectory': (B, T, 3) selected best trajectory
                - 'scores': (B, N) composite scores per candidate
                - 'best_idx': (B,) index of selected candidate
                - 'per_metric_scores': dict mapping each EPDMS metric key
                  to a tensor of shape (B, N)

        Raises:
            ValueError: If scenario data was not provided via initialize().
            KeyError: If ``ego_state`` or ``frame_idx`` is missing from kwargs.
        """
        if self.epdms is None:
            raise ValueError(
                "GtScorer not initialized. Call initialize(scenario_data, env) first."
            )

        ego_state = kwargs["ego_state"]
        frame_idx = kwargs["frame_idx"]

        all_candidates = model_output["all_candidates"]  # (B, N, T, 3)
        B, N, T, _ = all_candidates.shape
        device = all_candidates.device

        all_scores = torch.zeros(B, N, device=device)
        per_metric = {k: torch.zeros(B, N, device=device) for k in EPDMS_METRIC_KEYS}

        for b in range(B):
            candidates_np = all_candidates[b].cpu().numpy()  # (N, T, 3)
            scores_np, metrics = self.epdms.score_candidates(
                candidates_np, ego_state, frame_idx, return_metrics=True
            )
            all_scores[b] = torch.from_numpy(scores_np).to(device)
            for n, m in enumerate(metrics):
                for short_key, report_key in _METRIC_KEY_MAP.items():
                    per_metric[report_key][b, n] = m.get(short_key, 0.0)

        best_idx = all_scores.argmax(dim=-1)  # (B,)
        best_traj = all_candidates[
            torch.arange(B, device=device), best_idx
        ]  # (B, T, 3)

        return {
            "trajectory": best_traj,
            "scores": all_scores,
            "best_idx": best_idx,
            "per_metric_scores": per_metric,
        }
