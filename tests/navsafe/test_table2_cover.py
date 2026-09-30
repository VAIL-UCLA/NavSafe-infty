# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Table-2 scenario selection must stay leaf-balanced and reproducible.

The trap this pins: greedy set cover was the right instrument for the old
417-bundle corpus, where a bundle could carry four taxonomy leaves and 17
scenarios covered all of them. The kept allocation published 2026-08-30 carries
exactly one leaf per bundle, so cover has no overlap to exploit and silently
degenerates into "one scenario per leaf" — fine for k=1, but its tie-breaking
decides which scenario, and a re-run that reshuffled the picks would make two
halves of the table incomparable.

Stratified selection is the instrument for that corpus: k per leaf, lowest
tokens first, so the same k always names the same scenarios and a leaf with
fewer than k bundles contributes what it has instead of dragging in another
leaf's.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "navsafe_table2_cover", REPO / "scripts/tools/navsafe_table2_cover.py")
cover = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(cover)


def _rows(spec: dict[str, int]) -> list[dict]:
    """One single-leaf bundle per count, tokens ordered within a leaf."""
    rows = []
    for leaf, count in spec.items():
        for index in range(count):
            rows.append({"token": f"{leaf}-{index:02d}", "leaves": [leaf]})
    return rows


def test_stratified_takes_k_per_leaf() -> None:
    rows = _rows({"C-1": 10, "V-1": 10, "R-1": 10})
    picked = cover.stratified(rows, 2)
    assert picked == ["C-1-00", "C-1-01", "R-1-00", "R-1-01", "V-1-00", "V-1-01"]


def test_stratified_is_deterministic_under_input_order() -> None:
    rows = _rows({"C-1": 4, "V-1": 4})
    assert cover.stratified(rows, 2) == cover.stratified(list(reversed(rows)), 2)


def test_stratified_gives_a_short_leaf_what_it_has() -> None:
    """C-10 publishes 8 bundles, so k=10 must not borrow from other leaves."""
    rows = _rows({"C-10": 8, "V-1": 10})
    picked = cover.stratified(rows, 10)
    assert sum(1 for token in picked if token.startswith("C-10")) == 8
    assert len(picked) == 18


def test_greedy_cover_still_exploits_multi_leaf_bundles() -> None:
    rows = [
        {"token": "multi", "leaves": ["C-2", "C-5", "V-8"]},
        {"token": "single-a", "leaves": ["C-2"]},
        {"token": "single-b", "leaves": ["C-5"]},
        {"token": "single-c", "leaves": ["V-8"]},
    ]
    picked, freq = cover.greedy_cover(rows, 1)
    assert picked == ["multi"]
    assert freq == {"C-2": 2, "C-5": 2, "V-8": 2}


def test_greedy_cover_caps_need_at_corpus_frequency() -> None:
    """A leaf with one scenario must not loop forever chasing k=2."""
    rows = [
        {"token": "a", "leaves": ["V-8"]},
        {"token": "b", "leaves": ["C-5"]},
        {"token": "c", "leaves": ["C-5"]},
    ]
    picked, _ = cover.greedy_cover(rows, 2)
    assert sorted(picked) == ["a", "b", "c"]
