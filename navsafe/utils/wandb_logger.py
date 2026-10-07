# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Thin optional Weights & Biases wrapper shared by training and eval scripts.

All entry points go through :func:`init_wandb`, which returns ``None`` (with a
warning) when wandb is not installed or init fails — callers guard on the
handle, so W&B stays a soft dependency and offline runs keep working
(``WANDB_MODE=offline`` is respected as usual).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


def init_wandb(*, project: str, run_name: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None,
               entity: Optional[str] = None,
               group: Optional[str] = None,
               job_type: Optional[str] = None,
               tags: Optional[Sequence[str]] = None,
               run_id: Optional[str] = None,
               shared_label: Optional[str] = None,
               shared_primary: bool = False) -> Optional[Any]:
    """Start a W&B run; return the run handle or ``None`` on any failure."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb requested but not installed — logging disabled "
                       "(pip install wandb)")
        return None
    try:
        shared = bool(shared_label)
        settings = None
        if shared:
            settings = wandb.Settings(
                mode="shared",
                x_label=str(shared_label),
                x_primary=bool(shared_primary),
                x_update_finish_state=bool(shared_primary),
            )
        return wandb.init(
            project=project,
            name=run_name or None,
            id=run_id or None,
            entity=entity or None,
            group=group or None,
            job_type=job_type,
            tags=list(tags) if tags else None,
            config=config or {},
            settings=settings,
        )
    except Exception as exc:  # init failures must never kill a run
        logger.warning("wandb.init failed (%s) — logging disabled", exc)
        return None


def log_metrics(run: Optional[Any], metrics: Dict[str, Any],
                step: Optional[int] = None) -> None:
    """``run.log`` guarded against a disabled/None run and non-scalar values."""
    if run is None:
        return
    scalars = {k: v for k, v in metrics.items()
               if isinstance(v, (int, float, bool)) and v == v}  # drop NaN
    try:
        run.log(scalars, step=step)
    except Exception as exc:
        logger.warning("wandb.log failed (%s)", exc)


def finish(run: Optional[Any]) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


__all__ = ["init_wandb", "log_metrics", "finish"]
