"""Calibrate each family's episode time budget from the mined event pool.

The taxonomy doc leaves ``t_max`` per family as an open question and offers
"left turn <= 60 s" as a placeholder.  A guessed budget is a benchmark-design
choice hiding as a constant: too loose and a badly slow policy still passes,
too tight and a merely cautious one is scored as a deadlock.

So derive it.  Every mined candidate carries ``event_duration_s`` -- how long a
*human* took to complete this family's maneuver on the real log.  The budget is
the high quantile of that distribution times a slack factor, which states the
rule plainly: a policy has as long as a slow human, plus a margin, and no more.

The slack exists because closed-loop control is strictly harder than the log
(the policy also has to recover from its own deviations), not because the
number needed rounding.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from navsafe.benchmark import config as cfg


# Budget = QUANTILE of human completion time x SLACK.
QUANTILE = 0.95
SLACK = 1.3
# Floor so a family whose events are all very short does not get a budget too
# tight to absorb a single hesitation.
MIN_BUDGET_S = 10.0


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(cfg.CANDIDATES))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = {}
    for f in sorted(Path(args.candidates).glob("*.jsonl")):
        durs = sorted(json.loads(l)["event_duration_s"] for l in f.open())
        if not durs:
            continue
        p95 = quantile(durs, QUANTILE)
        budget = max(MIN_BUDGET_S, round(p95 * SLACK, 1))
        rows[f.stem] = {
            "n_events": len(durs),
            "median_s": round(statistics.median(durs), 2),
            "p95_s": round(p95, 2),
            "max_s": round(durs[-1], 2),
            "quantile": QUANTILE,
            "slack": SLACK,
            "t_max_s": budget,
        }

    hdr = f"{'family':<28}{'n':>6}{'median':>9}{'p95':>8}{'max':>8}{'t_max':>9}"
    print(hdr)
    print("-" * len(hdr))
    for fam, r in rows.items():
        print(f"{fam:<28}{r['n_events']:>6}{r['median_s']:>9.1f}"
              f"{r['p95_s']:>8.1f}{r['max_s']:>8.1f}{r['t_max_s']:>9.1f}")
    print(f"\nt_max = p{int(QUANTILE * 100)}(human completion time) x {SLACK}, "
          f"floor {MIN_BUDGET_S}s")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
