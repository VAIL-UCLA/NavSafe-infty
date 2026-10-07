"""Model adapter for DrivoR + PriorEye — the augmentation-based table row.

PriorEye (Yeon et al., "PriorEye: Geospatial Visual Priors for End-to-End
Autonomous Driving", ECCV 2026 spotlight, arXiv 2606.31830) is a *memory*
augmentation: it retrieves SigLIP2 embeddings of street-view imagery along the
route ahead and fuses them into the planner's scene tokens through a
dual-memory attention with a learned gate. Unlike the other two
augmentation-based rows — SimScale and BeyondDrive, which change *training*
only and evaluate through their base planner's own adapter — PriorEye adds
weights and an extra input at inference, so it needs an adapter of its own.

**Relationship to the `drivor` adapter.** The base planner is DrivoR
unchanged. Measured: this repo's `DrivoRModel` built from
`create_drivor_config()` loads upstream's `drivoR_baseline.ckpt` with 467/467
tensors, 0 missing and 0 unexpected, and `drivoR_prioreye.ckpt` differs from it
by exactly the 31 `_memory_module.*` tensors. So this class subclasses
`DrivoRAdapter`, flips `use_memory` on in the config, and adds the retrieval —
nothing about image preprocessing, proposal decoding, scoring or trajectory
parsing is re-implemented, and the two rows therefore differ only in the thing
the paper changes.

**Configuration** (the eval CLI passes a model name and a checkpoint, so the
rest comes from the environment):

| variable | meaning |
|---|---|
| `NAVSAFE_PRIOREYE_EMBEDDING` | **required** — path to `siglip2_embedding_all.pkl` |
| `NAVSAFE_PRIOREYE_ENCODER` | prior encoder, default `siglip2` (`dinov2`/`segformer` exist upstream but no released checkpoint uses them) |
| `NAVSAFE_PRIOREYE_MAP` | force the nuPlan map name instead of deriving it from the ego's UTM position |
| `NAVSAFE_PRIOREYE_REQUIRE_PRIORS` | `1` to fail on **any** replan that retrieves zero priors, instead of driving that replan on persistent memory alone |

Retrieval detail, and why it does not need the nuPlan devkit, is in
:mod:`navsafe.policy.sensor.utils.geospatial_priors`.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from navsafe.evaluation.utils.constants import DEFAULT_CMD, NAVSIM_CMD_MAPPING
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.drivor import DrivoRAdapter, create_drivor_config
from navsafe.policy.sensor.utils.geospatial_priors import (
    EMBEDDING_DIMS,
    GeospatialPriorRetriever,
    build_retriever,
    command_string,
)


@register_policy("prioreye")
class PriorEyeAdapter(DrivoRAdapter):
    """DrivoR with PriorEye's geospatial-prior memory module.

    The 31 memory tensors are supplied by the checkpoint, so they are NOT
    allowlisted as missing: a plain `drivor` checkpoint loaded through this
    adapter must fail rather than run a randomly-initialised memory module and
    report the result as PriorEye.
    """

    #: The checkpoint carries the memory module; a missing one is a real error.
    ALLOWED_MISSING_KEYS: tuple[str, ...] = ()
    ALLOWED_UNEXPECTED_KEYS: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        embedding_path: str | None = None,
        embedding_model: str | None = None,
        map_name: str | None = None,
        require_priors: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.embedding_path = embedding_path or os.environ.get("NAVSAFE_PRIOREYE_EMBEDDING")
        self.embedding_model = (
            embedding_model or os.environ.get("NAVSAFE_PRIOREYE_ENCODER") or "siglip2"
        ).lower()
        if self.embedding_model not in EMBEDDING_DIMS:
            raise ValueError(
                f"NAVSAFE_PRIOREYE_ENCODER={self.embedding_model!r} is not one of "
                f"{sorted(EMBEDDING_DIMS)}"
            )
        self.map_name = map_name or os.environ.get("NAVSAFE_PRIOREYE_MAP") or None
        self.require_priors = (
            bool(require_priors)
            if require_priors is not None
            else os.environ.get("NAVSAFE_PRIOREYE_REQUIRE_PRIORS", "0") == "1"
        )
        self._retriever: Optional[GeospatialPriorRetriever] = None
        self._retriever_scenario: Any = None
        self._priors_seen = 0
        self._replans = 0
        self._empty_replans = 0

    # ── model ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Build DrivoR with the memory module enabled, then load normally.

        The two memory keys are merged into whichever config the base adapter
        would have used — an operator's `config_path` if given, else
        `create_drivor_config()` — and the result is handed back through
        `config_path`. That keeps ONE code path through
        `DrivoRAdapter.load_model`: the checkpoint-key precondition, the RAM
        variant detection and the gate hooks all still run exactly as they do
        for `drivor`, and this class does not restate the (long, load-bearing)
        DrivoR config.
        """
        if self.embedding_path is None:
            raise ValueError(
                "PriorEye needs its geospatial prior embeddings: set "
                "NAVSAFE_PRIOREYE_EMBEDDING to siglip2_embedding_all.pkl "
                "(gdown 1hCZtWxHwqWbjrHhpq7b2Eq_Mi0PiY3Q7)."
            )
        if self.config_path and os.path.exists(self.config_path):
            config = OmegaConf.load(self.config_path)
        else:
            config = create_drivor_config(
                num_cameras=self.num_cameras,
                image_size=self.image_size,
                num_poses=self.num_poses,
                use_lidar=self.use_lidar,
            )
        config = OmegaConf.merge(
            config,
            OmegaConf.create(
                {
                    "use_memory": True,
                    "memory_embedding_model": self.embedding_model.upper(),
                }
            ),
        )

        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix="prioreye_config_", delete=False
        )
        with handle:
            handle.write(OmegaConf.to_yaml(config))
        original_config_path, self.config_path = self.config_path, handle.name
        try:
            super().load_model()
        finally:
            self.config_path = original_config_path
            os.unlink(handle.name)

    # ── retrieval ────────────────────────────────────────────────────────

    def _ensure_retriever(
        self, ego_state: Dict[str, Any], scenario_data: Dict[str, Any]
    ) -> GeospatialPriorRetriever:
        """Build (once per scenario) the retriever for this scenario's map.

        Keyed on the scenario dict's identity: an eval process drives one
        scenario per `Evaluator`, but a sweep reuses the loaded model across
        bundles, and the lane graph and UTM origin are per-scenario.
        """
        if self._retriever is not None and self._retriever_scenario is scenario_data:
            return self._retriever
        self._retriever = build_retriever(
            scenario_data,
            np.asarray(ego_state["position"], dtype=np.float64)[:2],
            pickle_path=str(self.embedding_path),
            embedding_model=self.embedding_model,
            map_name=self.map_name,
        )
        self._retriever_scenario = scenario_data
        self._priors_seen = 0
        self._replans = 0
        self._empty_replans = 0
        print(
            f"PriorEye: map={self._retriever.map_name} "
            f"lanes={len(self._retriever.lane_graph)} "
            f"origin=({self._retriever.origin_xy[0]:.1f}, {self._retriever.origin_xy[1]:.1f})"
        )
        return self._retriever

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        inputs = super().prepare_input(images, ego_state, scenario_data, frame_id)

        retriever = self._ensure_retriever(ego_state, scenario_data)
        command = int(ego_state.get("command", 3))
        one_hot = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD)
        embedding, position = retriever.retrieve(
            float(np.asarray(ego_state["position"], dtype=np.float64)[0]),
            float(np.asarray(ego_state["position"], dtype=np.float64)[1]),
            float(ego_state["heading"]),
            command_string(one_hot),
        )

        filled = int(retriever.last_diagnostics.get("bins_filled", 0))
        self._replans += 1
        self._priors_seen += filled
        if filled == 0:
            # Checked on EVERY replan, not just the first. An empty memory is a
            # legal state the module masks out, so a retrieval that goes blank
            # mid-episode — the ego leaves the annotated lanes, or a map-repair
            # feature wins the lane query — produces a scored episode driven by
            # the 16 persistent tokens alone. That defect was real: before the
            # `__` exclusion, pittsburgh retrieved 0/20 on all 41 replans and
            # scored anyway.
            self._empty_replans += 1
            message = (
                f"PriorEye retrieved 0 of 20 prior bins at replan "
                f"{self._replans} (frame {frame_id}, map={retriever.map_name}, "
                f"path_len={retriever.last_diagnostics.get('path_len')}). This "
                "replan ran on persistent memory alone, which is not the "
                "published policy."
            )
            if self.require_priors:
                raise RuntimeError(message)
            if self._empty_replans <= 3 or self._empty_replans % 20 == 0:
                print(
                    f"WARNING: {message} ({self._empty_replans} empty so far.) "
                    "Set NAVSAFE_PRIOREYE_REQUIRE_PRIORS=1 to fail instead."
                )

        key = self.embedding_model
        inputs[f"memory_embedding_{key}"] = (
            torch.from_numpy(embedding).unsqueeze(0).to(self.device)
        )
        inputs[f"memory_pos_{key}"] = torch.from_numpy(position).unsqueeze(0).to(self.device)
        return inputs

    def prior_coverage(self) -> float:
        """Mean filled prior bins per replan — 0.0 before the first replan.

        An episode's retrieval health in one number: 20.0 means every bin of
        the 100 m window carried a street-view prior on every replan, 0.0 means
        the memory module ran on persistent tokens only.
        """
        return 0.0 if self._replans == 0 else self._priors_seen / self._replans

    def empty_replan_count(self) -> int:
        """Replans that retrieved nothing — must be 0 for a published-policy run."""
        return self._empty_replans


__all__ = ["PriorEyeAdapter"]
