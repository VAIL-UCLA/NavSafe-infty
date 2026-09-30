"""
Test-Time Adaptation (TTA) trajectory scorer for inference scaling.

Extends GT-based EPDMS scoring (via the one EPDMS batch engine,
``EPDMSTrajectoryScorer_Fast``; scores are not comparable to
pre-consolidation TtaScorer output — see gt_scorer.py) with:
1. Plan continuity penalty: L2 distance between current candidate and previous plan
2. Extrapolated-agent collision scoring: forward-project agent positions and check
   spatial overlap with candidate trajectories

The final ranking incorporates all three signal types:
    composite = epdms_score - continuity_weight * continuity_penalty - collision_weight * collision_score
"""

import numpy as np
import torch
from typing import Dict, Any, Optional

from navsafe.evaluation.scorers.base_scorer import BaseTrajectoryScorer
from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSTrajectoryScorer_Fast,
)

REQUIRED_SCENARIO_FIELDS = ["tracks", "map_features", "metadata"]


class TtaScorer(BaseTrajectoryScorer):
    """
    TTA scorer that augments GT-based EPDMS scoring with plan continuity
    penalties and extrapolated-agent collision scores.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        continuity_weight: float = 1.0,
        collision_weight: float = 1.0,
    ):
        self.gamma = gamma
        self.continuity_weight = continuity_weight
        self.collision_weight = collision_weight
        self.epdms: Optional[EPDMSTrajectoryScorer_Fast] = None
        self.prev_plan: Optional[torch.Tensor] = None
        self.agent_predictions: Optional[Dict[str, Any]] = None

    def initialize(self, scenario_data: dict, env) -> None:
        """
        Initialize the EPDMS batch engine with scenario ground-truth data
        and store agent prediction data for collision scoring.

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
        self.prev_plan = None
        # Store tracks for agent extrapolation
        self.agent_predictions = scenario_data.get("tracks", {})

    def _compute_continuity_penalty(
        self, candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute plan continuity penalty as L2 distance to previous plan.

        Args:
            candidates: (B, N, T, 3) candidate trajectories.

        Returns:
            (B, N) continuity penalty per candidate. Zero if no previous plan
            (first timestep).
        """
        B, N, T, _ = candidates.shape
        device = candidates.device

        if self.prev_plan is None:
            # First timestep: no previous plan → zero penalty
            return torch.zeros(B, N, device=device)

        # prev_plan shape: (T, 3) — broadcast over batch and candidates
        prev = self.prev_plan.to(device)  # (T, 3)

        # Compute L2 norm of difference: ||candidate - prev_plan||
        # candidates: (B, N, T, 3), prev: (T, 3) → broadcast to (1, 1, T, 3)
        diff = candidates - prev.unsqueeze(0).unsqueeze(0)  # (B, N, T, 3)
        # L2 norm over time and coordinate dimensions
        penalty = torch.norm(diff.reshape(B, N, -1), p=2, dim=-1)  # (B, N)

        return penalty

    def _compute_collision_scores(
        self,
        candidates: torch.Tensor,
        agent_states: Optional[Dict[str, Any]] = None,
        frame_idx: int = 0,
        dt: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute extrapolated-agent collision scores by forward-projecting
        agent positions and checking spatial overlap with candidate trajectories.

        Args:
            candidates: (B, N, T, 3) candidate trajectories (x, y, heading).
            agent_states: Dict with agent position/velocity data for extrapolation.
                Expected keys: 'positions' (M, 2), 'velocities' (M, 2), 'sizes' (M, 2)
                where M is the number of agents.
            frame_idx: Current frame index.
            dt: Time step between trajectory points.

        Returns:
            (B, N) collision score per candidate. Higher means more collision risk.
        """
        B, N, T, _ = candidates.shape
        device = candidates.device

        if agent_states is None:
            return torch.zeros(B, N, device=device)

        positions = agent_states.get("positions")  # (M, 2)
        velocities = agent_states.get("velocities")  # (M, 2)
        sizes = agent_states.get("sizes")  # (M, 2) — length, width

        if positions is None or velocities is None or sizes is None:
            return torch.zeros(B, N, device=device)

        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float().to(device)
        if isinstance(velocities, np.ndarray):
            velocities = torch.from_numpy(velocities).float().to(device)
        if isinstance(sizes, np.ndarray):
            sizes = torch.from_numpy(sizes).float().to(device)

        M = positions.shape[0]
        if M == 0:
            return torch.zeros(B, N, device=device)

        # Forward-project agent positions: (M, T, 2)
        time_steps = torch.arange(T, device=device, dtype=torch.float32) * dt  # (T,)
        # positions: (M, 1, 2) + velocities: (M, 1, 2) * time_steps: (1, T, 1) → (M, T, 2)
        projected = (
            positions.unsqueeze(1)
            + velocities.unsqueeze(1) * time_steps.view(1, T, 1)
        )  # (M, T, 2)

        # Agent collision radius: half-diagonal of bounding box
        agent_radius = torch.sqrt((sizes[:, 0] / 2) ** 2 + (sizes[:, 1] / 2) ** 2)  # (M,)

        # Ego collision radius (approximate)
        ego_radius = 2.5  # meters, typical half-diagonal for ego vehicle

        collision_threshold = agent_radius + ego_radius  # (M,)

        # candidates xy: (B, N, T, 2)
        cand_xy = candidates[:, :, :, :2]  # (B, N, T, 2)

        # Compute distances: (B, N, T, M)
        # cand_xy: (B, N, T, 1, 2), projected transposed: (1, 1, T, M, 2)
        projected_t = projected.permute(1, 0, 2)  # (T, M, 2)
        diff = cand_xy.unsqueeze(-2) - projected_t.unsqueeze(0).unsqueeze(0)  # (B, N, T, M, 2)
        dists = torch.norm(diff, dim=-1)  # (B, N, T, M)

        # Check overlap: distance < collision_threshold
        # collision_threshold: (M,) → (1, 1, 1, M)
        overlap = (dists < collision_threshold.unsqueeze(0).unsqueeze(0).unsqueeze(0)).float()

        # Aggregate: sum overlaps across time and agents
        collision_scores = overlap.sum(dim=(-1, -2))  # (B, N)

        return collision_scores

    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        GT-based EPDMS + plan continuity penalty + extrapolated-agent collision score.

        Args:
            model_output: dict with 'all_candidates' of shape (B, N, T, 3).
            **kwargs: Must include 'ego_state' and 'frame_idx'.
                Optional: 'agent_states', 'dt'.

        Returns:
            dict with:
                - 'trajectory': (B, T, 3) selected best trajectory
                - 'scores': (B, N) final composite scores per candidate
                - 'best_idx': (B,) index of selected candidate
                - 'continuity_penalty': (B, N) plan continuity penalties
                - 'collision_scores': (B, N) extrapolated-agent collision scores

        Raises:
            ValueError: If scorer not initialized via initialize().
            KeyError: If ``ego_state`` or ``frame_idx`` is missing from kwargs.
        """
        if self.epdms is None:
            raise ValueError(
                "TtaScorer not initialized. Call initialize(scenario_data, env) first."
            )

        ego_state = kwargs["ego_state"]
        frame_idx = kwargs["frame_idx"]
        agent_states = kwargs.get("agent_states", None)
        dt = kwargs.get("dt", 0.5)

        all_candidates = model_output["all_candidates"]  # (B, N, T, 3)
        B, N, T, _ = all_candidates.shape
        device = all_candidates.device

        # 1. Compute GT-based EPDMS scores (same engine as GtScorer)
        epdms_scores = torch.zeros(B, N, device=device)
        for b in range(B):
            candidates_np = all_candidates[b].cpu().numpy()  # (N, T, 3)
            scores_np, _ = self.epdms.score_candidates(
                candidates_np, ego_state, frame_idx
            )
            epdms_scores[b] = torch.from_numpy(scores_np).to(device)

        # 2. Compute plan continuity penalty
        continuity_penalty = self._compute_continuity_penalty(all_candidates)

        # 3. Compute extrapolated-agent collision scores
        collision_scores = self._compute_collision_scores(
            all_candidates, agent_states, frame_idx, dt
        )

        # 4. Combine into final composite score
        composite = (
            epdms_scores
            - self.continuity_weight * continuity_penalty
            - self.collision_weight * collision_scores
        )

        # 5. Select best candidate
        best_idx = composite.argmax(dim=-1)  # (B,)
        best_traj = all_candidates[
            torch.arange(B, device=device), best_idx
        ]  # (B, T, 3)

        # 6. Update previous plan for next timestep
        # Use the first batch element's best trajectory as the reference plan
        self.prev_plan = best_traj[0].detach().clone()

        return {
            "trajectory": best_traj,
            "scores": composite,
            "best_idx": best_idx,
            "continuity_penalty": continuity_penalty,
            "collision_scores": collision_scores,
        }
