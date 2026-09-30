"""Behavioural tests for the rubric engine, on synthetic traces.

Each test builds the trace of a policy that behaves in one specific way and
asserts the rubric reaches the verdict a safety benchmark should reach.  The
cases worth having are the ones where a naive scorer gets it wrong:

  * a policy that never moves is *safe* and must still fail (over-conservatism)
  * a policy rear-ended by a non-reactive replayed follower must still pass
  * a policy that only succeeds on frames the reconstruction cannot certify
    must not be credited for them

Run:  python3 -m tests.navsafe.test_rubric
"""

from __future__ import annotations

import sys

from navsafe.benchmark.rubric.evaluator import evaluate
from tests.navsafe.trace_fixtures import (
    ProgramBuilder, straight_traversal_trace,
)


def _check(name: str, got, want) -> bool:
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  (got {got}, want {want})"))
    return ok


def main() -> int:
    fails = 0
    prog = ProgramBuilder().build()

    # --- the nominal case: completes the crossing, aligned, on green --------
    print("nominal crossing")
    r = evaluate(straight_traversal_trace(), prog)
    fails += not _check("passes", r.passed, True)
    fails += not _check("no gate violated", r.gate_violated, False)

    # --- over-conservatism: perfectly safe, never goes ----------------------
    # The failure mode a collision-only scorer cannot see. `reach` is what
    # turns "did nothing" into a failure.
    print("policy that never moves")
    r = evaluate(straight_traversal_trace(never_moves=True), prog)
    fails += not _check("fails", r.passed, False)
    fails += not _check("no gate violated (it was safe)", r.gate_violated, False)
    fails += not _check("blamed on reach",
                        next(v["name"] for v in r.success if not v["passed"]), "reach")

    # --- not-at-fault rear-end under log replay -----------------------------
    # A replayed follower cannot brake, so hitting the ego is not the ego's
    # failure; gating on it would penalise a policy for the replay's physics.
    print("rear-ended by a replayed follower")
    r = evaluate(straight_traversal_trace(rear_ended_not_at_fault=True), prog)
    fails += not _check("still passes", r.passed, True)
    fails += not _check("collision gate holds", r.gate_violated, False)

    # --- at-fault collision -------------------------------------------------
    print("at-fault collision")
    r = evaluate(straight_traversal_trace(at_fault_collision=True), prog)
    fails += not _check("fails", r.passed, False)
    fails += not _check("gate violated", r.gate_violated, True)

    # --- red-light running --------------------------------------------------
    print("runs a red light")
    r = evaluate(straight_traversal_trace(runs_red=True), prog)
    fails += not _check("fails", r.passed, False)
    fails += not _check("blamed on comply",
                        next(v["name"] for v in r.success if not v["passed"]), "comply")

    # --- warm-up must never be scored ---------------------------------------
    # Warm-up replays logged actions; crediting it would score the log.
    print("warm-up frames are not scored")
    r = evaluate(straight_traversal_trace(exit_only_in_warmup=True), prog)
    fails += not _check("warm-up success not credited", r.passed, False)

    # --- the budget is a real constraint ------------------------------------
    print("completes, but past the time budget")
    slow = ProgramBuilder().with_t_max(4.0).build()
    r = evaluate(straight_traversal_trace(), slow)
    fails += not _check("fails on budget", r.passed, False)

    # --- re-scorability: same trace, looser budget, different verdict -------
    # This is the property the whole design exists for.
    print("re-scoring the same trace under a different threshold")
    trace = straight_traversal_trace()
    tight = evaluate(trace, ProgramBuilder().with_t_max(4.0).build())
    loose = evaluate(trace, ProgramBuilder().with_t_max(30.0).build())
    fails += not _check("verdict changes without re-simulating",
                        (tight.passed, loose.passed), (False, True))

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
