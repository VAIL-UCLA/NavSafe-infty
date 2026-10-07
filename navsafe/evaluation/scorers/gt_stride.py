# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared ``gt_stride`` derivation for the EPDMS scorers.

``gt_stride`` is how many scenario frames the world advances per scored pose
(``sim_frame = frame_idx + t * gt_stride``). A stride of 0 freezes the world
for the whole scored horizon, making nc/ttc judge candidates against agents
that never move -- which is exactly what happened on every py123d run while
``scenario_dt`` was misparsed as an absolute timestamp (3.16e14).

Kept in one place because the floor was previously reimplemented per scorer
and a caller (``pdm_closed_planner.scoring.PDMScorer.initialize``) recomputed
the division inline afterwards, silently undoing it on the planner's path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def gt_stride(planner_dt: float, scenario_dt: float,
              *, default: int = 5) -> int:
    """Frames advanced per scored pose. Always >= 1."""
    if scenario_dt <= 0.0:
        return default
    stride = int(round(planner_dt / scenario_dt))
    if stride < 1:
        logger.warning(
            "EPDMS: planner_dt=%.4g s < scenario_dt=%.4g s gives "
            "gt_stride=%d; agents would not advance between scored poses. "
            "Clamping to 1.", planner_dt, scenario_dt, stride)
        return 1
    return stride
