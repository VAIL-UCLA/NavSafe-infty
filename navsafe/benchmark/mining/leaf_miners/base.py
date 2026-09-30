# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""What every leaf miner returns, and how one is chosen.

A miner answers one question — *which logged clips carry this leaf's event?* —
and must answer it with EVIDENCE, not a verdict. Every rejection this pipeline
had to walk back (a junction turn-fan called a merge, a right turn called a
U-turn) was a verdict without numbers attached; a candidate that carries the
predicate's own measurements can be argued with, drawn, and re-checked.

A miner never trains, converts or renders. It reports which windows a hit
falls in and which of those already have a reconstruction, so the expensive
stages are aimed rather than swept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from navsafe.benchmark import config as cfg

RECON_ROOT = str(cfg.CORPUS)
SEG_US = 5_000_000  # the 5 s sub-windows the recon pipeline trains on


@dataclass
class Scenario:
    """One row of the seed table — a 20 s window of one nuPlan log."""

    token: str
    log: str
    t0: int
    t1: int
    types: str = ""
    leaves: str = ""

    @classmethod
    def from_tsv_line(cls, line: str) -> Optional["Scenario"]:
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            return None
        return cls(token=f[0], log=f[1], t0=int(f[2]), t1=int(f[3]),
                   types=f[4] if len(f) > 4 else "", leaves=f[5] if len(f) > 5 else "")


@dataclass
class Candidate:
    """A clip this leaf can be built from, with the numbers that chose it."""

    scenario: Scenario
    qualifies: bool
    predicate: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    windows: List[str] = field(default_factory=list)      # s1..s4 the event falls in
    trained_windows: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        s = self.scenario
        return {
            "token": s.token, "log": s.log, "t0": s.t0, "t1": s.t1,
            "types": s.types, "leaves": s.leaves,
            "qualifies": bool(self.qualifies), "predicate": self.predicate,
            "evidence": self.evidence, "windows": self.windows,
            "trained_windows": self.trained_windows, "note": self.note,
        }

    def describe(self) -> str:
        head = "HIT " if self.qualifies else "    "
        return (f"{head}{self.scenario.token}  {self.predicate}  "
                f"windows={self.windows or '-'} trained={self.trained_windows or '-'}  "
                f"{self.note}")


def window_of(t_us: int, t0: int) -> str:
    """Which 5 s sub-window a timestamp falls in (``s1``..``s4``)."""
    return f"s{min(3, max(0, (int(t_us) - int(t0)) // SEG_US)) + 1}"


def trained(token: str, windows: List[str], *, recon_root: str = RECON_ROOT) -> List[str]:
    """Of ``windows``, the ones whose reconstruction is already on disk."""
    out = []
    for w in windows:
        clip = f"{token}{w}"
        if Path(recon_root, clip, "output_5cam", clip, "artifacts", "last.usdz").is_file():
            out.append(w)
    return out


# miner name -> callable(scenarios, **params) -> list[Candidate]
_MINERS: Dict[str, Callable] = {}


def register_miner(name: str):
    def deco(fn):
        _MINERS[name] = fn
        return fn
    return deco


def get_miner(name: str) -> Callable:
    try:
        return _MINERS[name]
    except KeyError:
        raise KeyError(
            f"unknown miner {name!r}; declared miners: {sorted(_MINERS)}. A leaf's "
            f"manifest names its miner under `mine.miner`."
        ) from None


def available_miners() -> List[str]:
    return sorted(_MINERS)


__all__ = [
    "Candidate", "RECON_ROOT", "SEG_US", "Scenario", "available_miners",
    "get_miner", "register_miner", "trained", "window_of",
]
