"""Build a trace from the logged human drive, then score it with the rubric.

No simulator, no renderer: the logged ego trajectory is fed through the same
``TraceWriter`` a rollout would use.  That makes this two useful things at once.

**A test of the machinery.**  Region resolution, lane lookup, signal decoding
all run against real map geometry rather than a fixture, so a
mistake in any of them shows up here instead of after a GPU rollout.

**A solvability check with teeth.**  The human demonstrably completed the
maneuver -- that is why the event was mined.  So the log *must* pass the
family's rubric.  If it does not, the rubric and the seed disagree, and every
policy would be failed for reasons that are not the policy's.  The paper's
solvability guarantee is exactly this claim, and this is where it gets tested.

    python3 -m navsafe.benchmark.trace.replay_log [--seeds DIR] [--only ID,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from navsafe.benchmark.outcome.labeler import label_episode
from navsafe.benchmark.rubric.evaluator import evaluate
from navsafe.benchmark.rubric.schema import ScenarioProgram
from navsafe.benchmark.scoring.episode import score_episode
from navsafe.benchmark.seeds.resolve_regions import _load_scenario
from navsafe.benchmark.trace.schema import EpisodeMeta
from navsafe.benchmark.trace.writer import TraceWriter

from navsafe.benchmark import config as cfg


RUBRICS = Path(__file__).resolve().parents[1] / "rubric" / "families"


def build_trace(seed: dict, regions: dict, out_dir: Path | None) -> list[dict]:
    sd = _load_scenario(seed)
    md = sd["metadata"]
    sdc = md["sdc_id"]
    st = sd["tracks"][sdc]["state"]
    pos = np.asarray(st["position"], dtype=np.float64)[:, :2]
    head = np.asarray(st.get("heading"), dtype=np.float64).reshape(-1)
    vel = np.asarray(st.get("velocity"), dtype=np.float64)[:, :2]
    ts = np.asarray(md["ts"], dtype=np.int64)

    scored_t0 = seed["window"]["scored_t0_us"]
    t_trig = seed["event"]["t_trig_us"]
    keep = np.flatnonzero(ts >= scored_t0)
    warmup = int(np.sum(ts[keep] < t_trig))
    dt = float(np.median(np.diff(ts[keep])) / 1e6) if keep.size > 1 else 0.1

    meta = EpisodeMeta(
        episode_id=f"{seed['seed_id']}-logreplay",
        seed_id=seed["seed_id"], family=seed["family"],
        regime="log_replay", policy="logged_human",
        sim_dt=dt, warmup_frames=warmup,
        world_version="n/a (no rendering)",
        scenario_origin_xy=tuple(md.get("scenario_origin_xy") or ()) or None,
        notes={"source": "logged ego trajectory, not a rollout"},
    )
    w = TraceWriter(scenario_data=sd, regions=regions, meta=meta,
                    warmup_frames=warmup, sim_dt=dt)

    # Other road users at the same timestamps, straight from the log.
    others = {k: v for k, v in sd["tracks"].items() if k != sdc}
    for n, i in enumerate(keep):
        agents = []
        for aid, tr in others.items():
            s = tr["state"]
            if not s.get("valid", [True] * len(ts))[i]:
                continue
            p = np.asarray(s["position"], dtype=np.float64)[i][:2]
            v = np.asarray(s["velocity"], dtype=np.float64)[i][:2]
            sz = np.asarray(s.get("size", [[4.5, 1.9, 1.5]] * len(ts)), dtype=np.float64)[i]
            agents.append({
                "id": aid, "cls": str(tr.get("type", "")), "policy": "replay",
                "x": float(p[0]), "y": float(p[1]),
                "yaw": float(np.asarray(s["heading"]).reshape(-1)[i]),
                "vx": float(v[0]), "vy": float(v[1]),
                "length": float(sz[0]), "width": float(sz[1]),
            })
        w.on_frame(
            frame=n, t_sim_s=n * dt, t_log_us=int(ts[i]),
            ego={"x": float(pos[i][0]), "y": float(pos[i][1]),
                 "yaw": float(head[i]), "speed": float(np.hypot(*vel[i]))},
            agents=agents, contacts=(),
        )
    if out_dir is not None:
        w.close(out_dir)
    return w.rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(cfg.SEEDS))
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="", help="write trace.parquet under this dir")
    args = ap.parse_args()

    want = {s for s in args.only.split(",") if s}
    rc = 0
    for sd_dir in sorted(Path(args.seeds).iterdir()):
        if not (sd_dir / "seed.json").exists() or (want and sd_dir.name not in want):
            continue
        reg_path = sd_dir / "regions.json"
        if not reg_path.exists():
            print(f"{sd_dir.name}: no regions.json -- run seeds/resolve_regions.py")
            rc = 1
            continue
        seed = json.loads((sd_dir / "seed.json").read_text())
        regions = json.loads(reg_path.read_text())
        out = Path(args.out) / sd_dir.name if args.out else None
        frames = build_trace(seed, regions, out)

        raw = yaml.safe_load((RUBRICS / f"{seed['family']}.yaml").read_text())
        prog = ScenarioProgram.from_dict(raw, seed_id=seed["seed_id"], regime="log_replay")
        res = evaluate(frames, prog, episode_id=f"{seed['seed_id']}-logreplay")
        labels = label_episode(frames, res)
        sc = score_episode(frames, res, labels, policy="logged_human")

        head = "PASS" if res.passed else "FAIL"
        print(f"\n[{head}] {sd_dir.name}  {seed['family']}")
        print(f"    frames={res.coverage['n_frames']} scored={res.coverage['n_scored']} "
              f"score={sc.score:.3f}")
        for v in res.success:
            print(f"      {'ok  ' if v['passed'] else 'FAIL'} {v['name']:<12} {v['reason']}")
        for v in res.gates:
            print(f"      {'ok  ' if v['passed'] else 'GATE'} {v['name']:<12} {v['reason']}")
        if labels["labels"]:
            print(f"      labels: {', '.join(labels['labels'])}")
        if not res.passed:
            # The human completed this maneuver by construction, so a failure
            # here is a rubric/seed disagreement, not a driving failure.
            print("      ^ the logged human fails its own rubric: "
                  "rubric and seed disagree")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
