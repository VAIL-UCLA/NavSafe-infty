#!/usr/bin/env python
# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Score a finished Evaluator run with the NavSafe termination taxonomy and
the four NavSafe metrics.

Reads an evaluator output directory and its py123d Arrow scene,
rebuilds trace-shaped frames (``navsafe.benchmark.trace.from_eval``), and runs
the *same* scoring chain a NavSafe seed run uses:

    frames -> termination.classify        why the episode ended (one reason)
           -> infractions_from_trace      fault-respecting contact counts
           -> route.route_progress        monotone route completion
           -> termination.to_route_result Bench2Drive RouteResult, or None
           -> DS / SR / Efficiency / Comfort

Pure post-processing: no simulator, no GPU — re-scoring a stored run takes
seconds, per the NavSafe architecture rule.

Three consequences of doing it this way, all of which change numbers produced
by the earlier ad-hoc version of this script:

1. **Route completion is monotone** (``RouteCompletionTest``), so a policy that
   leaves the route and ends up near its far end no longer collects 100 %.
2. **Episodes the BENCHMARK ended are excluded, not scored.**  A simulator or
   renderer failure comes back with ``driving_score``/``success`` ``null`` and
   the termination reason, per the taxonomy's two-cause rule — a benchmark
   limitation is never folded into a policy's mean as a 0.  Driving far from the
   logged path is not one of these: the render-validity envelope was removed
   2026-08-19, and an ego that leaves the log is scored for where it went.
3. **Route completion is taken at the termination frame**, not at the last
   frame simulated: the ego keeps being stepped after the event that ended the
   episode, and crediting that motion would reward driving on after a crash.

Caveats recorded in the output rather than hidden: ego kinematics are
Savitzky-Golay derivatives of a position grid (see ``navsafe_metrics_probe.py``),
and on a scenario with no map the drivable-area terms are reported ``null``
instead of 0.

Usage:
    # a bundle (what the HF dataset ships)
    python scripts/tools/navsafe_from_eval.py \
        --eval-dir /path/to/eval/output/<scene> \
        --py123d-data-root /path/to/<token>/arrow \
        --warmup-frames 20

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from navsafe.benchmark.scoring.from_run import (
    T_MAX_FLOOR_S,  # noqa: F401  (kept: external callers import these)
    T_MAX_SLACK,    # noqa: F401
    scenario_meta_for,
    score_run,
)
from navsafe.benchmark.trace import from_eval


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True, type=Path)
    ap.add_argument("--py123d-data-root", type=Path, required=True,
                    help="Arrow scene directory, e.g. <bundle>/arrow")
    ap.add_argument("--py123d-scene-index", type=int, default=0,
                    help="which scene under --py123d-data-root (default 0)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--warmup-frames", type=int, default=0,
                    help="GT ego-replay frames at the start; never scored (the "
                         "policy was not driving)")
    ap.add_argument("--t-max", type=float, default=None,
                    help="episode time budget in seconds; default is the "
                         "logged human's remaining drive x1.3, floor 10 s")
    args = ap.parse_args()

    run = from_eval.load_from_arrow(
        args.eval_dir, args.py123d_data_root,
        scene_index=args.py123d_scene_index,
        warmup_frames=args.warmup_frames, dt=args.dt)
    try:
        out = score_run(
            run, name=args.eval_dir.name,
            warmup_frames=args.warmup_frames, dt=args.dt,
            t_max=args.t_max, source=str(args.eval_dir),
            scenario_meta=(scenario_meta_for(args.py123d_data_root)
                           if args.py123d_data_root else None))
    except ValueError as exc:
        raise SystemExit(str(exc))

    _fr = out["frames"]
    # t_max_s serializes as None for an infinite budget (clock-free protocol);
    # there is no budget to compare the window against.
    if _fr["t_max_s"] is not None and _fr["scored_window_s"] < _fr["t_max_s"] - 0.05:
        print(f"[navsafe] WARNING: scored window is {_fr['scored_window_s']} s but "
              f"t_max is {_fr['t_max_s']} s — no policy was tested against its "
              "budget here.", flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (args.eval_dir / "navsafe_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
