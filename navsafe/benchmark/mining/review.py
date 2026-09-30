# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The human tier of selection: a reviewer's verdict, read from their sheet.

Two of the three selection tiers are code. The pool tier reads the raw nuPlan
log and map; the host tier runs geometry predicates on a converted host. Both
measure the ROAD, and neither can answer the question that actually decides
whether a clip carries a leaf: *does this look like the scenario?* That is a
person watching 500 renders, and it was the one tier with nowhere to live —
the verdict existed in a spreadsheet, and `bake` never saw it.

This reads that spreadsheet. What it does NOT do is override the other two:
a human verdict and a geometry verdict are different questions, and the
interesting rows are the ones where they disagree. So the verdict is stamped
onto :class:`~navsafe.benchmark.mining.mine.MineRow` as its own axis, next to
the pool and host evidence, and every tier's answer stays visible.

The sheet is exported as CSV with these columns (extra columns are ignored)::

    Scenario ID (token)   the 16-hex nuPlan token
    For Editing 90        which leaf the reviewer assigned it to, blank if none
    Keep? (Y/N)           Y | N | Maybe | blank
    Bundle Status         Published | Not ready | ...
    Reviewer Notes        free text, e.g. "no crosswalk"

``For Editing 90`` may name two leaves at once (``R-3/R-4``): they share a
template and a reviewer judging "something crosses the road here" is judging
both. Such a row is stamped onto each.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

#: Sheet column -> what this module needs. Kept explicit rather than guessed,
#: because a renamed column should fail loudly rather than silently review
#: nothing.
TOKEN_COL = "Scenario ID (token)"
LEAF_COL = "For Editing 90"
KEEP_COL = "Keep? (Y/N)"
STATUS_COL = "Bundle Status"
NOTES_COL = "Reviewer Notes"

#: One cell, two leaves. R-3 and R-4 are the same template with a different
#: asset family, so a reviewer picks a host for "something crosses here" and
#: both leaves can use it.
LEAF_ALIASES: Dict[str, tuple] = {
    "R-3/R-4": ("R-3", "R-4"),
    "R-4/R-3": ("R-3", "R-4"),
}

#: A verdict of ``Maybe`` counts as selected, with the reason carried through.
#: Dropping it would silently shrink the pool; promoting it to ``Y`` would hide
#: that someone hesitated, and the note usually says why ("quality").
KEEP_SELECTED = ("Y", "MAYBE")
KEEP_REJECTED = ("N",)


class ReviewError(NexusSimError, ValueError):
    """The review sheet cannot be read as one."""


@dataclass
class ReviewRow:
    """One reviewer verdict, for one leaf."""

    token: str
    leaf: str
    keep: str          # "Y" | "N" | "Maybe" | ""
    bundle_status: str
    note: str = ""
    source: str = ""

    @property
    def selected(self) -> bool:
        return self.keep.strip().upper() in KEEP_SELECTED

    @property
    def rejected(self) -> bool:
        return self.keep.strip().upper() in KEEP_REJECTED

    def to_dict(self) -> dict:
        return {"keep": self.keep, "selected": self.selected, "rejected": self.rejected,
                "bundle_status": self.bundle_status, "note": self.note,
                "source": self.source}


def read_review(path: "str | Path") -> Dict[str, List[ReviewRow]]:
    """Every assigned verdict in the sheet, grouped by leaf.

    Rows with no leaf in ``For Editing 90`` are skipped: the sheet lists the
    whole corpus and most of it is unassigned, which is not a finding.

    Raises:
        ReviewError: the file is missing the columns this reads. A silently
            empty review is worse than a refusal — it looks exactly like
            "nobody has reviewed anything yet".
    """
    path = Path(path)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        # `fieldnames`, not a row's keys: a row with more cells than the header
        # gets a ``None`` key, which is not a column and does not sort.
        header = [c for c in (reader.fieldnames or []) if c]
        rows = list(reader)
    missing = [c for c in (TOKEN_COL, LEAF_COL, KEEP_COL) if c not in header]
    if missing:
        raise ReviewError(
            f"{path}: missing column(s) {missing}. Found: {header}. If the sheet was "
            f"renamed, update the *_COL constants rather than guessing here — a mis-read "
            f"column reviews nothing and looks like an unreviewed corpus."
        )
    if not rows:
        raise ReviewError(f"{path}: header only, no rows")

    out: Dict[str, List[ReviewRow]] = {}
    unknown: Dict[str, int] = {}
    for raw in rows:
        cell = (raw.get(LEAF_COL) or "").strip()
        if not cell:
            continue
        leaves = LEAF_ALIASES.get(cell, (cell,))
        for leaf in leaves:
            if not _looks_like_a_leaf(leaf):
                unknown[cell] = unknown.get(cell, 0) + 1
                continue
            out.setdefault(leaf, []).append(ReviewRow(
                token=(raw.get(TOKEN_COL) or "").strip(),
                leaf=leaf,
                keep=(raw.get(KEEP_COL) or "").strip(),
                bundle_status=(raw.get(STATUS_COL) or "").strip(),
                note=(raw.get(NOTES_COL) or "").strip(),
                source=path.name,
            ))
    if unknown:
        logger.warning("review: %s names no known leaf and was skipped: %s",
                       LEAF_COL, dict(unknown))
    logger.info("review: %s -> %s", path.name,
                {leaf: len(rs) for leaf, rs in sorted(out.items())})
    return out


def _looks_like_a_leaf(name: str) -> bool:
    from navsafe.benchmark.leaves import available

    return name in available()


def review_for(path: "str | Path", leaf: str) -> Dict[str, ReviewRow]:
    """``{token: ReviewRow}`` for one leaf."""
    rows = read_review(path).get(leaf, [])
    return {r.token: r for r in rows if r.token}


__all__ = ["LEAF_ALIASES", "ReviewError", "ReviewRow", "read_review", "review_for"]
