"""Learned (neural network-based) two-stage coarse-to-fine trajectory scorer.

This scorer loads a pretrained checkpoint containing a coarse-stage transformer
decoder and fine-stage per-metric prediction heads.  The coarse stage scores all
N candidates and selects the top-K; the fine stage re-ranks those K candidates
using per-metric heads and aggregates to a final score.
"""

import os
from typing import Any, Dict

import torch
import torch.nn as nn

from navsafe.evaluation.scorers.base_scorer import BaseTrajectoryScorer


class LearnedScorer(BaseTrajectoryScorer):
    """Two-stage coarse-to-fine learned trajectory scorer.

    Parameters
    ----------
    ckpt_path : str
        Path to a checkpoint file containing ``coarse_decoder_state_dict``,
        ``fine_heads_state_dict``, and ``config`` entries.
    top_k : int
        Number of candidates to keep after the coarse stage.
    device : str
        Torch device for model parameters (``"cuda"`` or ``"cpu"``).

    Raises
    ------
    FileNotFoundError
        If *ckpt_path* does not point to an existing file.
    RuntimeError
        If the checkpoint is missing required keys or has an incompatible
        format.
    """

    # Keys that must be present in a valid checkpoint
    _REQUIRED_CKPT_KEYS = {"coarse_decoder_state_dict", "fine_heads_state_dict", "config"}
    # Config entries needed to build the model
    _REQUIRED_CONFIG_KEYS = {"d_model", "nhead", "num_metrics"}

    def __init__(self, ckpt_path: str, top_k: int = 5, device: str = "cuda"):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.top_k = top_k
        self.device = torch.device(device)

        # Load and validate checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint(ckpt)

        config = ckpt["config"]
        d_model = config["d_model"]
        nhead = config["nhead"]
        num_metrics = config["num_metrics"]

        # Coarse stage: a single TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
        )
        self.coarse_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.coarse_decoder.load_state_dict(ckpt["coarse_decoder_state_dict"])
        self.coarse_decoder.to(self.device).eval()

        # Fine stage: per-metric linear heads  (metric_name → Linear(d_model, 1))
        self.fine_heads = nn.ModuleDict()
        fine_sd = ckpt["fine_heads_state_dict"]
        # Discover metric names from state-dict keys (e.g. "metric_0.weight")
        metric_names = sorted(
            {k.split(".")[0] for k in fine_sd.keys()}
        )
        for name in metric_names:
            head = nn.Linear(d_model, 1)
            head_sd = {
                k.split(".", 1)[1]: v
                for k, v in fine_sd.items()
                if k.startswith(f"{name}.")
            }
            head.load_state_dict(head_sd)
            self.fine_heads[name] = head
        self.fine_heads.to(self.device).eval()

        self.num_metrics = num_metrics

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_checkpoint(cls, ckpt: Any) -> None:
        """Raise ``RuntimeError`` when the checkpoint format is wrong."""
        if not isinstance(ckpt, dict):
            raise RuntimeError(
                f"Checkpoint format mismatch: expected dict, got {type(ckpt).__name__}"
            )
        missing = cls._REQUIRED_CKPT_KEYS - set(ckpt.keys())
        if missing:
            raise RuntimeError(
                f"Checkpoint format mismatch: expected keys {cls._REQUIRED_CKPT_KEYS}, "
                f"missing {missing}"
            )
        config = ckpt.get("config", {})
        if not isinstance(config, dict):
            raise RuntimeError(
                f"Checkpoint format mismatch: expected config to be dict, "
                f"got {type(config).__name__}"
            )
        missing_cfg = cls._REQUIRED_CONFIG_KEYS - set(config.keys())
        if missing_cfg:
            raise RuntimeError(
                f"Checkpoint format mismatch: expected config keys "
                f"{cls._REQUIRED_CONFIG_KEYS}, missing {missing_cfg}"
            )

    # ------------------------------------------------------------------
    # Two-stage scoring pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_best(self, model_output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the coarse → top-K → fine scoring pipeline.

        Parameters
        ----------
        model_output : dict
            Must contain ``"all_candidates"`` of shape ``(B, N, T, 3)``.

        Returns
        -------
        dict
            ``trajectory`` (B, T, 3), ``scores`` (B, N), ``best_idx`` (B,),
            and ``stage_scores`` with ``coarse`` and ``fine`` sub-dicts.
        """
        candidates = model_output["all_candidates"]  # (B, N, T, 3)
        B, N, T, _ = candidates.shape
        device = self.device
        candidates_dev = candidates.to(device)

        # Flatten trajectory features for the decoder: (B, N, T*3)
        flat = candidates_dev.reshape(B, N, T * 3)

        # Project to d_model via a simple linear expansion (zero-pad / truncate)
        d_model = self.coarse_decoder.layers[0].self_attn.embed_dim
        if flat.shape[-1] < d_model:
            pad = torch.zeros(B, N, d_model - flat.shape[-1], device=device)
            query = torch.cat([flat, pad], dim=-1)
        else:
            query = flat[:, :, :d_model]

        # Memory: mean-pool over candidates as a simple context
        memory = query.mean(dim=1, keepdim=True).expand(B, N, d_model)

        # --- Coarse stage ---
        coarse_features = self.coarse_decoder(query, memory)  # (B, N, d_model)
        coarse_scores = coarse_features.mean(dim=-1)  # (B, N)

        # --- Top-K selection ---
        actual_k = min(self.top_k, N)
        topk_vals, topk_idx = torch.topk(coarse_scores, actual_k, dim=-1)  # (B, K)

        # Gather top-K features
        idx_expand = topk_idx.unsqueeze(-1).expand(B, actual_k, d_model)
        topk_features = torch.gather(coarse_features, 1, idx_expand)  # (B, K, d_model)

        # --- Fine stage: per-metric heads ---
        metric_scores = {}
        for name, head in self.fine_heads.items():
            metric_scores[name] = head(topk_features).squeeze(-1)  # (B, K)

        # Aggregate fine scores: mean across metrics
        fine_scores = torch.stack(list(metric_scores.values()), dim=-1).mean(dim=-1)  # (B, K)

        # Best among top-K
        fine_best_local = fine_scores.argmax(dim=-1)  # (B,)
        best_idx_global = topk_idx[torch.arange(B, device=device), fine_best_local]  # (B,)

        # Build full-N score tensor (coarse scores for all, fine-adjusted for top-K)
        scores = coarse_scores.clone()
        # Scatter fine scores back into the full tensor for the top-K positions
        scores.scatter_(1, topk_idx, fine_scores)

        # Selected trajectory
        trajectory = candidates_dev[torch.arange(B, device=device), best_idx_global]  # (B, T, 3)

        return {
            "trajectory": trajectory.cpu(),
            "scores": scores.cpu(),
            "best_idx": best_idx_global.cpu(),
            "stage_scores": {
                "coarse": {
                    "scores": coarse_scores.cpu(),          # (B, N)
                    "topk_idx": topk_idx.cpu(),             # (B, K)
                    "topk_scores": topk_vals.cpu(),         # (B, K)
                },
                "fine": {
                    "scores": fine_scores.cpu(),            # (B, K)
                    "per_metric": {
                        k: v.cpu() for k, v in metric_scores.items()
                    },
                    "best_local_idx": fine_best_local.cpu(),  # (B,)
                },
            },
        }
