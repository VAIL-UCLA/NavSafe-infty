# Copyright (c) 2022-2025, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""One-shot evaluation helper.

Design §8 (Public API Surface) specifies ``evaluate`` as a top-level
public function: "One-shot helper: takes a policy + EnvCfg (+ optional
scorer), returns metrics."

This module provides that helper as a thin convenience wrapper around
:class:`~navsafe.evaluation.evaluator.Evaluator`. It is re-exported
from ``navsafe/__init__.py`` so users can write::

    from navsafe import evaluate
    metrics = evaluate(policy, env_cfg, scenario_path="/path/to/scenario")

Requirements: 5.1, 5.2 (public API surface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from navsafe.env.env_cfg import EnvCfg
    from navsafe.policy.base import BasePolicyAdapter


def evaluate(
    policy: "BasePolicyAdapter",
    env_cfg: "Optional[EnvCfg]" = None,
    *,
    scenario_path: "str | Path | None" = None,
    scorer: "Optional[str]" = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a single closed-loop evaluation and return metrics.

    This is the public one-shot helper from design §8. For advanced
    use (iteration, callbacks, hooks), use :class:`Evaluator` directly.

    Args:
        policy: A loaded policy adapter instance (subclass of
            :class:`~navsafe.policy.BasePolicyAdapter`).
        env_cfg: Optional :class:`~navsafe.env.EnvCfg` selecting the
            environment configuration axes. When ``None``, the default
            replay preset is used.
        scenario_path: Path to the scenario to evaluate. Required
            unless ``env_cfg`` specifies a scenario source that does
            not need an explicit path.
        scorer: Optional scorer name to resolve from the registry.
            When ``None``, the default EPDMS scorer is used.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~navsafe.evaluation.evaluator.EvaluationConfig`.

    Returns:
        A dictionary of evaluation metrics (EPDMS, driving_score,
        route_completion_fraction, per-frame details, etc.).
    """
    import dataclasses

    from navsafe.env import NavSafeEnv
    from navsafe.env.presets import REPLAY
    from navsafe.evaluation.evaluator import EvaluationConfig, Evaluator

    if scenario_path is None:
        raise ValueError(
            "evaluate() requires a scenario_path to the scenario to evaluate."
        )
    if scorer is not None:
        # The trajectory-scorer name→class mapping is not exposed through the
        # plugin registry, so there is no robust way to resolve a scorer name
        # here. Construct an Evaluator directly with a `trajectory_scorer` for
        # custom scoring; the default scoring path is used otherwise.
        raise NotImplementedError(
            "evaluate(): named-scorer selection is not yet wired. Build an "
            "Evaluator directly with a trajectory_scorer for custom scoring."
        )

    path = Path(scenario_path)

    # Resolve the environment: default to the REPLAY preset, mapping the
    # scenario path onto the env's data_directory/scenario_path fields. This
    # mirrors the wiring in unified_evaluator.main().
    if env_cfg is None:
        env_cfg = dataclasses.replace(
            REPLAY,
            data_directory=str(path.parent),
            scenario_path=str(path),
            loop_replay=False,
            spawn_ego_vehicle=True,
            required_bundles=[],
        )

    env = NavSafeEnv(env_cfg)
    eval_config = EvaluationConfig(**kwargs)
    evaluator = Evaluator(env=env, model_adapter=policy, config=eval_config)
    evaluator.setup(scenario_path=path)
    return evaluator.run()
