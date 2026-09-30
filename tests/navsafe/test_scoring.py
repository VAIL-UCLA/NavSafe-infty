"""Behavioural tests for the outcome labeller and the scoring chain.

Run:  python3 -m tests.navsafe.test_scoring
"""

from __future__ import annotations

import sys

from navsafe.benchmark.outcome.labeler import (
    CONTACT, DEADLOCK, ROAD_RULE_VIOLATION, label_episode,
)
from navsafe.benchmark.rubric.evaluator import evaluate
from tests.navsafe.trace_fixtures import (
    ProgramBuilder, straight_traversal_trace,
)
from navsafe.benchmark.scoring.episode import (
    Weights, aggregate, paired_delta, score_episode,
)


def _check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  (got {got!r}, want {want!r})"))
    return 0 if ok else 1


def main() -> int:
    fails = 0
    prog = ProgramBuilder().build()

    # --- labelling reads the trace, not the intent -------------------------
    print("labels come from behaviour")
    t = straight_traversal_trace()
    fails += _check("clean run has no labels", label_episode(t)["labels"], [])

    t = straight_traversal_trace(at_fault_collision=True)
    lab = label_episode(t)
    fails += _check("collision labelled", CONTACT in lab["labels"], True)
    fails += _check("mapped to a crash cell", lab["crash_cells"], ["traffic_crashes"])

    t = straight_traversal_trace(runs_red=True)
    lab = label_episode(t)
    fails += _check("red running labelled",
                    ROAD_RULE_VIOLATION in lab["labels"], True)

    # A policy that never moves is safe and useless: it must be labelled, and
    # as a deadlock rather than a mere unnecessary stop, since it never
    # resolves.
    print("never-moving policy")
    t = straight_traversal_trace(never_moves=True)
    lab = label_episode(t)
    fails += _check("labelled deadlock", DEADLOCK in lab["labels"], True)
    fails += _check("not labelled contact", CONTACT in lab["labels"], False)

    # --- the safety gate is multiplicative ---------------------------------
    print("safety gate zeroes the score")
    t = straight_traversal_trace(at_fault_collision=True)
    r = evaluate(t, prog)
    s = score_episode(t, r, label_episode(t))
    fails += _check("gate closed", s.safety_gate, 0.0)
    fails += _check("score is zero regardless of behaviour", s.score, 0.0)

    # --- a safe-but-useless policy scores low but is not gated -------------
    print("safe but useless scores low without a gate violation")
    t = straight_traversal_trace(never_moves=True)
    r = evaluate(t, prog)
    s = score_episode(t, r, label_episode(t))
    fails += _check("gate open", s.safety_gate, 1.0)
    fails += _check("no progress", s.progress, 0.0)
    fails += _check("rubric failed", s.rubric_passed, False)
    fails += _check("scores below a clean run", s.score < 0.6, True)

    # --- weights are inputs, and must be sane ------------------------------
    print("weight validation")
    try:
        Weights(0.5, 0.5, 0.5, 0.5).validate()
        fails += _check("rejects weights that do not sum to 1", False, True)
    except ValueError:
        fails += _check("rejects weights that do not sum to 1", True, True)

    # --- aggregation -------------------------------------------------------
    print("aggregation")
    good = straight_traversal_trace()
    bad = straight_traversal_trace(at_fault_collision=True)
    scores = []
    for i in range(8):
        tr = good if i % 2 == 0 else bad
        r = evaluate(tr, prog, episode_id=f"ep{i}")
        scores.append(score_episode(tr, r, label_episode(tr), policy="test"))
    agg = aggregate(scores, min_group=4)
    fails += _check("counts episodes", agg["n_episodes"], 8)
    fails += _check("half violate the gate",
                    agg["per_family"]["F3_straight_traversal"]["gate_violation_rate"], 0.5)
    fails += _check("rubric pass rate reported per family",
                    agg["per_family"]["F3_straight_traversal"]["rubric_pass_rate"], 0.5)
    fails += _check("worst group identified", agg["worst_group"]["n"], 8)

    # thin cells must not be reported as the worst group
    agg2 = aggregate(scores[:2], min_group=5)
    fails += _check("thin group suppressed", agg2["worst_group"], None)
    fails += _check("suppression is reported", agg2["n_groups_below_min"], 1)

    # --- paired delta needs matched episodes -------------------------------
    print("paired delta")
    base = [s for s in scores if s.episode_id in {"ep0", "ep2", "ep4"}]
    shift = []
    for eid in ("ep0", "ep2", "ep6"):
        r = evaluate(bad, prog, episode_id=eid)
        shift.append(score_episode(bad, r, label_episode(bad), policy="test"))
    d = paired_delta(base, shift)
    fails += _check("only matched episodes are paired", d["n_pairs"], 2)
    fails += _check("unmatched are reported", d["n_unmatched"], 2)
    fails += _check("degradation is positive", d["delta"] > 0, True)

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
