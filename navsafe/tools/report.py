#!/usr/bin/env python3
# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Summarize NavSafe metrics and compare shared scored scenarios.

Writes report.tsv, report.json and report.md into --run. Unscored cells
are excluded from metric means; undefined efficiency values are omitted
from the efficiency mean. --baseline compares only shared scored tokens.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_cells(run: Path) -> Dict[str, Dict[str, Any]]:
    """One record per evaluated scenario, keyed by token.

    Two layouts are read, because a sweep directory outlives the script that
    wrote it: cells directly under the run (``<token>_<model>/``, the original
    shape) and cells under ``scenarios/`` (the current one, which keeps them
    separate from logs).
    """
    cells: Dict[str, Dict[str, Any]] = {}
    metrics_files = sorted(
        list(run.glob("*/navsafe_metrics.json"))
        + list(run.glob("scenarios/*/navsafe_metrics.json")))
    for metrics_path in metrics_files:
        try:
            data = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] unreadable {metrics_path}: {exc}")
            continue
        scenario = data.get("scenario") or {}
        token = str(scenario.get("token") or metrics_path.parent.name.split("_")[0])
        metrics = data.get("metrics") or {}
        termination = data.get("termination") or {}
        cells[token] = {
            "token": token,
            "dir": metrics_path.parent.name,
            "status": data.get("status"),
            "driving_score": metrics.get("driving_score"),
            "success": metrics.get("success"),
            "efficiency_pct": metrics.get("efficiency_pct"),
            "comfort": metrics.get("comfort"),
            "termination": termination.get("reason"),
            "policy_attributed": termination.get("policy_attributed"),
            "collisions": termination.get("collision_count"),
            "taxonomy": ",".join(scenario.get("taxonomy_leaves") or []),
            "frames_scored": (data.get("frames") or {}).get("scored"),
        }
    return cells


def _aggregate(cells: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    scored = [c for c in cells.values() if c.get("status") == "scored"]
    excluded = [c for c in cells.values() if c.get("status") != "scored"]
    scores = [float(c["driving_score"]) for c in scored
              if c.get("driving_score") is not None]
    effs = [float(c["efficiency_pct"]) for c in scored
            if c.get("efficiency_pct") is not None]
    comforts = [float(c["comfort"]) for c in scored if c.get("comfort") is not None]
    successes = [bool(c["success"]) for c in scored if c.get("success") is not None]
    terminations: Dict[str, int] = {}
    for cell in scored:
        reason = str(cell.get("termination") or "unknown")
        terminations[reason] = terminations.get(reason, 0) + 1
    return {
        "cells": len(cells),
        "scored": len(scored),
        "excluded": len(excluded),
        "excluded_tokens": {c["token"]: c.get("status") for c in excluded},
        "driving_score_mean": (statistics.fmean(scores) if scores else None),
        "success_rate": (statistics.fmean([float(s) for s in successes])
                         if successes else None),
        "efficiency_pct_mean": (statistics.fmean(effs) if effs else None),
        "efficiency_cells": len(effs),
        "comfort_mean": (statistics.fmean(comforts) if comforts else None),
        "terminations": terminations,
    }


def _fmt(value: Any, spec: str = ".2f") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="sweep directory (holds <token>_<model>/ cells)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="a second sweep to compare against, over shared tokens")
    args = ap.parse_args()

    run: Path = args.run
    cells = _load_cells(run)
    if not cells:
        print(f"no navsafe_metrics.json under {run}")
        return 1
    rows = [cells[t] for t in sorted(cells)]
    header = ("token", "status", "score", "success", "eff%", "comfort",
              "termination")
    lines = ["\t".join(header)]
    for cell in rows:
        lines.append("\t".join([
            cell["token"], str(cell.get("status")),
            _fmt(cell.get("driving_score"), ".3f"),
            _fmt(cell.get("success")),
            _fmt(cell.get("efficiency_pct"), ".1f"),
            _fmt(cell.get("comfort"), ".3f"),
            str(cell.get("termination")),
        ]))
    table = "\n".join(lines)
    print(table)

    agg = _aggregate(cells)
    print()
    print(f"scored {agg['scored']}/{agg['cells']} "
          f"(excluded: {agg['excluded']} {agg['excluded_tokens'] or ''})")
    print(f"driving_score mean : {_fmt(agg['driving_score_mean'], '.3f')}")
    print(f"success rate       : {_fmt(agg['success_rate'], '.3f')}")
    print(f"efficiency_pct mean: {_fmt(agg['efficiency_pct_mean'], '.1f')} "
          f"(over {agg['efficiency_cells']} cells with a defined value)")
    print(f"comfort mean       : {_fmt(agg['comfort_mean'], '.3f')}")
    print(f"terminations       : {agg['terminations']}")

    comparison: Optional[Dict[str, Any]] = None
    if args.baseline is not None:
        base_cells = _load_cells(args.baseline)
        shared = sorted(
            t for t in cells
            if t in base_cells
            and cells[t].get("status") == "scored"
            and base_cells[t].get("status") == "scored")
        if shared:
            def _mean(source: Dict[str, Dict[str, Any]], key: str) -> Optional[float]:
                vals = [float(source[t][key]) for t in shared
                        if source[t].get(key) is not None]
                return statistics.fmean(vals) if vals else None

            comparison = {
                "shared_tokens": shared,
                "run_driving_score": _mean(cells, "driving_score"),
                "baseline_driving_score": _mean(base_cells, "driving_score"),
                "run_success": _mean(cells, "success"),
                "baseline_success": _mean(base_cells, "success"),
            }
            delta = None
            if (comparison["run_driving_score"] is not None
                    and comparison["baseline_driving_score"] is not None):
                delta = (comparison["run_driving_score"]
                         - comparison["baseline_driving_score"])
            comparison["driving_score_delta"] = delta
            print()
            print(f"vs baseline over {len(shared)} shared scored tokens: "
                  f"{_fmt(comparison['run_driving_score'], '.3f')} vs "
                  f"{_fmt(comparison['baseline_driving_score'], '.3f')} "
                  f"(delta {_fmt(delta, '+.3f')})")
        else:
            print("\nno token is scored in BOTH runs — nothing comparable yet")

    (run / "report.tsv").write_text(table + "\n")
    (run / "report.json").write_text(json.dumps({
        "run": str(run), "cells": rows, "aggregate": agg,
        "comparison": comparison,
    }, indent=2, default=str) + "\n")

    md = ["# NavSafe run: " + run.name, "",
          f"- scored **{agg['scored']}/{agg['cells']}** cells",
          f"- driving score **{_fmt(agg['driving_score_mean'], '.3f')}**",
          f"- success rate **{_fmt(agg['success_rate'], '.3f')}**",
          f"- efficiency **{_fmt(agg['efficiency_pct_mean'], '.1f')}%** "
          f"over {agg['efficiency_cells']} cells",
          f"- comfort **{_fmt(agg['comfort_mean'], '.3f')}**",
          "",
          "| token | status | score | success | eff% | comfort | termination |",
          "|---|---|---|---|---|---|---|"]
    for cell in rows:
        md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            cell["token"], cell.get("status"),
            _fmt(cell.get("driving_score"), ".3f"), _fmt(cell.get("success")),
            _fmt(cell.get("efficiency_pct"), ".1f"),
            _fmt(cell.get("comfort"), ".3f"), cell.get("termination")))
    if comparison:
        md += ["", "## Against the baseline", "",
               f"Over the {len(comparison['shared_tokens'])} tokens scored in "
               f"both runs: **{_fmt(comparison['run_driving_score'], '.3f')}** "
               f"vs {_fmt(comparison['baseline_driving_score'], '.3f')} "
               f"(delta {_fmt(comparison['driving_score_delta'], '+.3f')})."]
    (run / "report.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {run}/report.{{tsv,json,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
