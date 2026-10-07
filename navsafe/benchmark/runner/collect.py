"""Collect every episode verdict into one benchmark result.

Each eval writes a ``verdict.json`` beside its trace. That is per-episode; a
benchmark result is what you get by reading all of them together, and this is
the only place that happens -- so it is also the only place that decides what
"the result" means.

Three reporting rules, carried over from the scoring module because they are the
substance of the benchmark rather than presentation choices:

* the **rubric pass rate is the primary** and is reported per family, never
  averaged across scenario types -- an average over families would hide exactly
  the per-competence attribution the benchmark exists to provide;
* the **worst populated group** is reported next to the mean, because a mean
  hides brittle subgroups;
* **thin cells are suppressed, not silently averaged**, and the suppression is
  counted in the output.

Usage:
    python3 -m navsafe.benchmark.runner.collect                 # human table
    python3 -m navsafe.benchmark.runner.collect --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg
from navsafe.benchmark.scoring.episode import (
    EpisodeScore, aggregate, paired_delta,
)


def load_verdicts(root: Path, tag: str = "") -> list[tuple[dict, EpisodeScore]]:
    """Verdicts under root, as (raw, EpisodeScore), optionally one run only.

    ``tag`` selects by the run directory ``eval/<seed>/<tag>/trace/`` that
    run_eval.py names after the episode shape. Without it every run ever done
    under this tree is pooled -- including runs from before a fix -- and the
    aggregate silently mixes them. A benchmark result has to be one coherent
    run, so pass the tag.
    """
    out = []
    for f in sorted(root.rglob("verdict.json")):
        # eval/<seed>/<tag>/trace/verdict.json
        if tag and f.parent.parent.name != tag:
            continue
        try:
            raw = json.loads(f.read_text())
            sc = EpisodeScore(**raw["score"])
        except Exception as exc:  # noqa: BLE001
            print(f"  !! unreadable {f}: {exc}", file=sys.stderr)
            continue
        out.append((raw, sc))
    return out


def _table(rows: list[tuple[str, ...]], head: tuple[str, ...]) -> str:
    w = [max(len(str(r[i])) for r in (list(rows) + [head])) for i in range(len(head))]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(head))
    sep = "  ".join("-" * w[i] for i in range(len(head)))
    body = "\n".join("  ".join(str(r[i]).ljust(w[i]) for i in range(len(head)))
                     for r in rows)
    return f"{line}\n{sep}\n{body}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(cfg.EVAL))
    ap.add_argument("--tag", default="",
                    help="run directory to report on (eval/<seed>/<tag>/). "
                         "Omit only when the tree holds exactly one run: "
                         "otherwise runs from before a fix are pooled in.")
    ap.add_argument("--min-group", type=int, default=1,
                    help="cells with fewer episodes are excluded from the "
                         "worst-group statistic and counted separately")
    ap.add_argument("--json", default="", help="also write the full result here")
    ap.add_argument("--baseline-regime", default="log_replay")
    ap.add_argument("--shifted-regime", default="",
                    help="regime to compare against the baseline (paired delta)")
    args = ap.parse_args()

    root = Path(args.eval_root)
    found = load_verdicts(root, args.tag)
    if not found:
        print(f"no verdict.json under {root}"
              + (f" with tag {args.tag!r}" if args.tag else "") + "\n"
              "run an eval with NAVSAFE_TRACE set (run_eval.py does this)",
              file=sys.stderr)
        return 1
    # Benchmark endings (a simulator or renderer failure; historically also
    # the retired envelope exit) are not policy results: the taxonomy reports
    # them '—' and keeps them out of every denominator, because averaging them
    # in would charge the policy for the benchmark's own limits. They are counted and listed, never silently
    # dropped -- a shrinking denominator is itself a finding.
    def _benchmark_ending(raw: dict) -> bool:
        t = raw.get("termination") or {}
        return bool(t) and not t.get("policy_attributed", True)

    excluded = [(raw, sc) for raw, sc in found if _benchmark_ending(raw)]
    keep = [(raw, sc) for raw, sc in found if not _benchmark_ending(raw)]
    scores = [sc for _, sc in keep]
    if not scores:
        print(f"all {len(found)} episode(s) ended for benchmark reasons "
              f"(simulator/renderer failure); nothing to aggregate",
              file=sys.stderr)
        return 1
    agg = aggregate(scores, min_group=args.min_group)
    agg["n_excluded"] = len(excluded)
    agg["excluded_reasons"] = sorted(
        {raw["termination"]["reason"] for raw, _ in excluded})

    print(f"\nNavSafe result — {agg['n_episodes']} episode(s) under {root}"
          + (f"  [run {args.tag}]" if args.tag else "  [ALL runs pooled]"))
    print(f"NSS = {agg['nss']:.2f}   (weights {agg['weights']})\n")
    if excluded:
        print(f"{len(excluded)} episode(s) excluded from all denominators "
              f"({', '.join(agg['excluded_reasons'])}):")
        for raw, sc in excluded:
            t = raw["termination"]
            print(f"    —  {sc.seed_id}  {sc.family}  {sc.regime}  "
                  f"{sc.policy or '-'}  {t['reason']} @ frame {t['frame']}")
        print()

    print("per family — rubric pass rate is the primary, never averaged across families")
    print(_table(
        [(fam, v["n"], f"{v['rubric_pass_rate']:.2f}",
          f"{v['gate_violation_rate']:.2f}", f"{v['nss']:.1f}")
         for fam, v in agg["per_family"].items()],
        ("family", "n", "rubric_pass", "gate_viol", "NSS")))

    print("\nper regime")
    print(_table(
        [(reg, v["n"], f"{v['rubric_pass_rate']:.2f}", f"{v['nss']:.1f}")
         for reg, v in agg["per_regime"].items()],
        ("regime", "n", "rubric_pass", "NSS")))

    if agg["worst_group"]:
        w = agg["worst_group"]
        print(f"\nworst group: {w['group']}  n={w['n']}  NSS={w['nss']:.1f}")
    if agg["n_groups_below_min"]:
        print(f"({agg['n_groups_below_min']} cell(s) below --min-group "
              f"{args.min_group}, excluded from worst-group)")

    if agg["outcome_label_counts"]:
        print("\noutcome labels (multi-label, read off the trace)")
        print(_table([(k, v) for k, v in agg["outcome_label_counts"].items()],
                     ("label", "episodes")))

    print("\nper episode")
    print(_table(
        [(sc.seed_id, sc.family, sc.regime, sc.policy or "-",
          "PASS" if sc.rubric_passed else ("GATE" if sc.safety_gate == 0 else "FAIL"),
          f"{sc.score:.3f}",
          (raw.get("termination") or {}).get("reason", "?"),
          f"{sc.coverage.get('n_scored', 0)}/{sc.coverage.get('n_frames', 0)}",
          )
         for raw, sc in keep],
        ("seed", "family", "regime", "policy", "verdict", "score", "ended",
         "scored/frames")))

    result = {"aggregate": agg,
              "episodes": [{**sc.to_dict(),
                            "termination": raw.get("termination")}
                           for raw, sc in keep],
              "excluded_episodes": [{**sc.to_dict(),
                                     "termination": raw.get("termination")}
                                    for raw, sc in excluded]}

    if args.shifted_regime:
        base = [s for s in scores if s.regime == args.baseline_regime]
        shift = [s for s in scores if s.regime == args.shifted_regime]
        # Matched on episode_id, which embeds the regime, so re-key on the seed
        # to pair the same event across regimes.
        for s in base + shift:
            s.episode_id = s.seed_id
        d = paired_delta(base, shift)
        result["paired_delta"] = d
        print(f"\npaired delta {args.baseline_regime} -> {args.shifted_regime}: "
              f"n_pairs={d['n_pairs']} delta={d['delta']} "
              f"(unmatched {d.get('n_unmatched', 0)})")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
