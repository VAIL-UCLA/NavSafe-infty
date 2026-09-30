"""Validate the shipped family rubrics against the scenario-program schema.

Also proves the regime story: the same rubric file must compile under all three
interaction regimes once its actor policies are swapped, because the whole
point of the three-regime design is that the *same event* is scored by the
*same rubric* while only the background traffic changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from navsafe.benchmark.rubric.schema import (  # noqa: E402
    ProgramError, ScenarioProgram,
)

REGIME_SWAP = {
    "log_replay": "replay",
    "reactive": "idm",
    "safety_critical": "idm",   # plus >=1 adversarial actor, added below
}


def main() -> int:
    here = Path(__file__).resolve().parent / "families"
    files = sorted(here.glob("*.yaml"))
    if not files:
        print(f"no rubrics under {here}", file=sys.stderr)
        return 1
    bad = 0
    for f in files:
        raw = yaml.safe_load(f.read_text())
        for regime, policy in REGIME_SWAP.items():
            d = dict(raw)
            d["actors"] = {sel: policy for sel in raw["actors"]}
            if regime == "safety_critical":
                d["actors"]["adversary_0"] = "adversarial"
            try:
                p = ScenarioProgram.from_dict(d, seed_id="<template>", regime=regime)
            except ProgramError as e:
                print(f"FAIL {f.name} [{regime}]: {e}")
                bad += 1
                continue
            if regime == "log_replay":
                print(f"OK   {f.name:<34} {p.family:<26} "
                      f"success={len(p.success)} gates={len(p.gates)} "
                      f"init={len(p.initialization)} reject={len(p.reject)} "
                      f"t_max={p.t_max_s}s")
        # a rubric must not silently accept a policy its regime forbids
        d = dict(raw, actors={sel: "adversarial" for sel in raw["actors"]})
        try:
            ScenarioProgram.from_dict(d, seed_id="<t>", regime="log_replay")
            print(f"FAIL {f.name}: log_replay accepted an adversarial actor")
            bad += 1
        except ProgramError:
            pass
    print(f"\n{len(files)} rubrics, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
