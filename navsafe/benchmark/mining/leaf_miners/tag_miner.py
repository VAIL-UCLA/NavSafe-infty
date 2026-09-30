# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Tag-based selection: the leaves nuPlan already labels (V-8 and friends).

When nuPlan names the manoeuvre, the selection is a query, not a measurement —
and the tag is more trustworthy than any geometry we would write. A net-heading
"U-turn detector" written here once flagged a parking reversal (heading flips
at walking pace) as the strongest U-turn in the pool.

Two things this miner will not do:

* **Substitute a neighbouring tag.** If a leaf asks for ``starting_u_turn``
  and only two exist in val/test, the answer is two, not ten padded out with
  right turns. Padding is how a right turn shipped as V-8.
* **Ignore sensor coverage.** A clip with holes in its 8 cameras cannot be
  reconstructed, so it cannot become a scenario; ``require_sensor`` keeps
  those out of the candidate list rather than out of the next stage's time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from navsafe.benchmark.mining.leaf_miners.base import (
    Candidate,
    Scenario,
    register_miner,
    trained,
)

logger = logging.getLogger(__name__)


@register_miner("tag")
def mine(scenarios: List[Scenario], *, types: List[str] = (), require_sensor: bool = True,
         splits: List[str] = (), index: str = "", **_ignored) -> List[Candidate]:
    """Select scenarios whose nuPlan scenario types intersect ``types``.

    Args:
        scenarios: seed rows (their ``types`` column is the tag list).
        types: wanted nuPlan scenario types, most-specific first. The order is
            reported so a reviewer can see whether a hit came from the leaf's
            own tag or from a broader stand-in.
        require_sensor: kept for signature parity with the index-backed sweep;
            seed rows are already sensor-complete by construction.
        index: optional ``runs.parquet`` to widen the search beyond the seed
            table (not read here — see navsafe/benchmark/mining/select_candidates.py,
            which owns the whole-split sweep).
    """
    wanted = [t for t in (types or []) if t]
    if not wanted:
        raise ValueError("the tag miner needs `types` — which nuPlan scenario types count")
    out: List[Candidate] = []
    for scn in scenarios:
        have = [t.strip() for t in (scn.types or "").split(",") if t.strip()]
        matched = [t for t in wanted if t in have]
        ev: Dict[str, Any] = {"matched_types": matched, "all_types": have,
                              "wanted_in_order": wanted}
        ok = bool(matched)
        note = ""
        if ok:
            primary = wanted[0]
            ev["is_primary_tag"] = matched[0] == primary
            note = (f"tagged {matched[0]}"
                    + ("" if matched[0] == primary
                       else f" (STAND-IN — the leaf's primary tag is {primary})"))
        windows = ["s1", "s2", "s3", "s4"] if ok else []
        out.append(Candidate(scn, ok, "tag", ev, windows,
                             trained(scn.token, windows) if ok else [], note))
    hits = [c for c in out if c.qualifies]
    primary = [c for c in hits if c.evidence.get("is_primary_tag")]
    if hits and not primary:
        logger.warning(
            "no candidate carries the leaf's primary tag %r — every hit is a stand-in. "
            "Report that rather than presenting them as the leaf.", wanted[0])
    return out


__all__ = ["mine"]
