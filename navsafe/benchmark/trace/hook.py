"""Opt-in bridge from a running Evaluator to the NavSafe trace.

The evaluator gains six lines and no NavSafe knowledge: everything about seeds,
regions, agent-state shapes and trace schema lives here. Tracing is off unless
``NAVSAFE_TRACE`` points at a seed, so no existing eval changes behaviour.

    NAVSAFE_TRACE       seed dir (or seed.json / regions.json) to trace against
    NAVSAFE_TRACE_OUT   where trace.parquet + trace_meta.json go
    NAVSAFE_POLICY      recorded in the trace meta (name only)
    NAVSAFE_REGIME      log_replay | reactive | safety_critical

A failure here must not take the eval down -- a lost trace is recoverable, a
lost 20-minute rollout is not -- but it must not pass silently either, because
an eval that quietly produced no trace looks exactly like one that produced a
passing verdict. So the first failure is reported loudly and tracing then stops.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_regions(spec: str) -> tuple[dict, dict]:
    """(seed, regions) from a seed dir, seed.json, or regions.json path."""
    import json
    p = Path(spec)
    if p.is_dir():
        seed_p, reg_p = p / "seed.json", p / "regions.json"
    elif p.name == "regions.json":
        seed_p, reg_p = p.parent / "seed.json", p
    else:
        seed_p, reg_p = p, p.parent / "regions.json"
    if not reg_p.exists():
        raise FileNotFoundError(
            f"{reg_p} missing -- run navsafe/benchmark/seeds/resolve_regions.py; "
            "the rubric's regions must be frozen before a trace can reference them")
    return json.loads(seed_p.read_text()), json.loads(reg_p.read_text())


def _agents_for_trace(agent_states: Sequence[dict], ego_xy, ego_yaw) -> list[dict]:
    """Env agent dicts -> the trace's agent shape.

    The env carries ``position``/``heading``/``velocity``/``type``; the trace
    wants scalars plus a lead flag. ``is_lead`` is decided here rather than in a
    predicate: it needs the ego pose, which the writer has and the rubric does
    not.
    """
    out = []
    for a in agent_states or ():
        if a.get("is_ego"):
            continue
        pos = np.asarray(a.get("position", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(-1)
        vel = np.asarray(a.get("velocity", (0.0, 0.0)), dtype=np.float64).reshape(-1)
        vx = float(vel[0]) if vel.size else 0.0
        vy = float(vel[1]) if vel.size > 1 else 0.0
        dx, dy = float(pos[0]) - ego_xy[0], float(pos[1]) - ego_xy[1]
        fwd = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        lat = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
        out.append({
            "id": str(a.get("id", "")),
            "cls": str(a.get("type", "")),
            "x": float(pos[0]), "y": float(pos[1]),
            "yaw": float(a.get("heading", 0.0)),
            "vx": vx, "vy": vy,
            "length": float(a.get("length", 4.5)),
            "width": float(a.get("width", 1.9)),
            # same lane-ish and ahead: enough for the headway predicates, and
            # honest about being a proxy rather than a lane-graph query.
            "is_lead": bool(fwd > 0.0 and abs(lat) < 1.75),
        })
    return out


class TraceHook:
    """Per-episode trace accumulation, driven from the evaluator's step loop."""

    def __init__(self, seed: dict, regions: dict, out_dir: Path,
                 *, policy: str, regime: str, warmup_frames: int, sim_dt: float):
        from navsafe.benchmark.trace.schema import EpisodeMeta
        from navsafe.benchmark.trace.writer import TraceWriter

        self.seed, self.regions, self.out_dir = seed, regions, Path(out_dir)
        self._writer_cls = TraceWriter
        self._meta = EpisodeMeta(
            episode_id=f"{seed['seed_id']}-{regime}-{policy or 'policy'}",
            seed_id=seed["seed_id"], family=seed.get("family", ""),
            regime=regime, policy=policy,
            policy_checkpoint=os.environ.get("NAVSAFE_CHECKPOINT", ""),
            sim_dt=sim_dt, warmup_frames=warmup_frames,
            world_version=str(seed.get("artifacts", {}).get("recon", "")),
            scenario_origin_xy=tuple(regions.get("scenario_origin_xy") or ()) or None,
        )
        self.warmup_frames = warmup_frames
        self.sim_dt = sim_dt
        self._writer = None
        self._failed = False

    # --- construction ----------------------------------------------------

    @classmethod
    def from_env(cls, *, warmup_frames: int, sim_dt: float) -> Optional["TraceHook"]:
        spec = os.environ.get("NAVSAFE_TRACE", "")
        if not spec or spec in ("0", "false"):
            return None
        try:
            seed, regions = _resolve_regions(spec)
        except Exception as exc:  # noqa: BLE001
            logger.error("[navsafe] tracing requested but unusable: %s", exc)
            return None
        out = Path(os.environ.get("NAVSAFE_TRACE_OUT", "")
                   or Path(seed["artifacts"]["eval"]) / "trace")
        hook = cls(seed, regions, out,
                   policy=os.environ.get("NAVSAFE_POLICY", ""),
                   regime=os.environ.get("NAVSAFE_REGIME", "log_replay"),
                   warmup_frames=warmup_frames, sim_dt=sim_dt)
        logger.info("[navsafe] tracing %s (%s) -> %s",
                    seed["seed_id"], seed.get("family", ""), out)
        return hook

    # --- per-frame -------------------------------------------------------

    def on_step(self, *, env, frame: int, ego_state: dict, collision: bool,
                signal_hold: bool = False) -> None:
        """``signal_hold``: the evaluator's "red light ahead on the ego's
        lane" fact for this frame (EPDMSLiveScorer._signal_hold_live); the
        deadlock exemption in termination.py reads it from the trace."""
        if self._failed:
            return
        try:
            if self._writer is None:
                scenario = getattr(env, "current_scenario", None)
                if scenario is None:
                    raise RuntimeError("env.current_scenario is None; "
                                       "the trace needs the map and tracks")
                self._writer = self._writer_cls(
                    scenario_data=scenario, regions=self.regions, meta=self._meta,
                    warmup_frames=self.warmup_frames, sim_dt=self.sim_dt)

            pos = np.asarray(ego_state.get("position", (0.0, 0.0, 0.0)),
                             dtype=np.float64).reshape(-1)
            yaw = float(ego_state.get("heading", ego_state.get("yaw", 0.0)))
            vel = np.asarray(ego_state.get("velocity", (0.0, 0.0)),
                             dtype=np.float64).reshape(-1)
            speed = float(ego_state.get("speed", np.linalg.norm(vel[:2])))
            ego = {
                "x": float(pos[0]), "y": float(pos[1]),
                "z": float(pos[2]) if pos.size > 2 else 0.0,
                "yaw": yaw, "speed": speed,
                "yaw_rate": float(ego_state.get("angular_velocity", 0.0) or 0.0),
                "steer": float(ego_state.get("steer", 0.0) or 0.0),
            }
            agents = _agents_for_trace(getattr(env, "agent_states", ()) or (),
                                       (ego["x"], ego["y"]), yaw)
            # One contact record per colliding frame. Fault is left to the
            # writer, which decides it from who was closing.
            contacts = []
            if collision:
                near = min(agents, key=lambda a: math.dist(
                    (a["x"], a["y"]), (ego["x"], ego["y"])), default=None) \
                    if agents else None
                contacts.append({
                    "agent_id": near["id"] if near else "",
                    "kind": "rear_end" if (near and near.get("is_lead")) else "angle",
                    "rel_speed": abs(speed - (near or {}).get("vx", 0.0)),
                })
            self._writer.on_frame(
                frame=frame, t_sim_s=frame * self.sim_dt,
                t_log_us=self._t_log_us(frame),
                ego=ego, agents=agents, contacts=contacts,
                signal_hold=bool(signal_hold))
        except Exception as exc:  # noqa: BLE001
            self._failed = True
            logger.error("[navsafe] trace failed at frame %d, tracing disabled "
                         "for this episode: %s", frame, exc, exc_info=True)

    def _t_log_us(self, frame: int) -> int:
        ts = self.regions.get("path", {}).get("ts_us") or []
        if not ts:
            return 0
        return int(ts[min(frame, len(ts) - 1)])

    # --- end of episode --------------------------------------------------

    def close(self) -> Optional[Path]:
        """Write the trace, then score it with the family's rubric."""
        if self._writer is None or self._failed:
            if self._failed:
                logger.error("[navsafe] no trace written (see the error above)")
            return None
        try:
            path = self._writer.close(self.out_dir)
            self._score(self._writer.rows)
            return path
        except Exception as exc:  # noqa: BLE001
            logger.error("[navsafe] writing/scoring the trace failed: %s", exc,
                         exc_info=True)
            return None

    def _terminate(self, frames: list[dict], res, t_max_s: float) -> dict:
        """Termination reason for this episode (taxonomy §0.1).

        The rubric already decided whether the goal was reached; reusing its
        ``reach`` verdict keeps one definition of "the goal" instead of a
        second one here that could disagree with the verdict printed beside it.
        ``reach`` reports *when* it happened, so an earlier ending (a contact,
        a wrong-way exit) still masks a goal the ego reached later.
        """
        from navsafe.benchmark import termination as term

        reach = next((v for v in res.success if v["name"] == "reach"), None)
        goal_reached = bool(reach and reach["passed"])
        goal_frame = None
        if goal_reached:
            t_reach = (reach.get("evidence") or {}).get("t_reach_s")
            scored = [(i, f) for i, f in enumerate(frames)
                      if f.get("phase") == "scored"]
            if scored and t_reach is not None:
                target = scored[0][1]["t_sim_s"] + float(t_reach)
                goal_frame = next((i for i, f in scored
                                   if f["t_sim_s"] >= target - 1e-9), None)
        t = term.classify(frames, goal_reached=goal_reached,
                          goal_frame=goal_frame, t_max_s=float(t_max_s),
                          dt=self.sim_dt)
        return {"reason": t.reason.value, "frame": t.frame, "detail": t.detail,
                "policy_attributed": t.reason.policy_attributed,
                "t_max_s": float(t_max_s)}

    def _score(self, frames: list[dict]) -> None:
        """Rubric verdict + outcome labels + episode score, beside the trace."""
        import json

        import yaml

        from navsafe.benchmark.outcome.labeler import label_episode
        from navsafe.benchmark.rubric.evaluator import evaluate
        from navsafe.benchmark.rubric.schema import ScenarioProgram
        from navsafe.benchmark.scoring.episode import score_episode

        fam = self.seed.get("family", "")
        rubric = (Path(__file__).resolve().parents[1] / "rubric" / "families"
                  / f"{fam}.yaml")
        if not rubric.exists():
            logger.warning("[navsafe] no rubric for family %r; trace written "
                           "but unscored", fam)
            return
        prog = ScenarioProgram.from_dict(
            yaml.safe_load(rubric.read_text()),
            seed_id=self.seed["seed_id"], regime=self._meta.regime)
        res = evaluate(frames, prog, episode_id=self._meta.episode_id)
        comfort_measurable = os.environ.get(
            "NAVSAFE_EXECUTION_MODE", "teleport") != "teleport"
        labels = label_episode(frames, res, comfort_measurable=comfort_measurable)
        # Teleport execution places the ego at planned waypoints instead of
        # driving it, so the dynamics the comfort term measures never happen.
        sc = score_episode(frames, res, labels, policy=self._meta.policy,
                           comfort_measurable=comfort_measurable)
        if not comfort_measurable:
            print("[navsafe] comfort excluded from the score: execution_mode="
                  "teleport (the ego is placed, not driven)", flush=True)

        # Why the episode ended, in the benchmark's own taxonomy. Recorded on
        # every verdict because aggregation needs it: benchmark endings
        # (simulator/renderer failure) are reported '—' and excluded from
        # denominators rather than folded into a mean as a failing score.
        termination = self._terminate(frames, res, prog.t_max_s)

        out = {"rubric": res.to_dict(), "outcome": labels, "score": sc.to_dict(),
               "termination": termination}
        (self.out_dir / "verdict.json").write_text(json.dumps(out, indent=2))

        head = "PASS" if res.passed else ("GATE" if res.gate_violated else "FAIL")
        print(f"\n[navsafe] {head}  {fam}  {self.seed['seed_id']}  "
              f"({self._meta.regime})  score={sc.score:.3f}", flush=True)
        print(f"   ended  {termination['reason']} at frame "
              f"{termination['frame']}"
              + ("" if termination["policy_attributed"]
                 else "  (benchmark ending: episode excluded, not scored 0)")
              + (f" — {termination['detail']}" if termination["detail"] else ""))
        for v in res.success:
            print(f"   {'ok  ' if v['passed'] else 'FAIL'} {v['name']:<12} {v['reason']}")
        for v in res.gates:
            print(f"   {'ok  ' if v['passed'] else 'GATE'} {v['name']:<12} {v['reason']}")
        if labels["labels"]:
            print(f"   labels: {', '.join(labels['labels'])}")
        cov = res.coverage
        print(f"   frames={cov['n_frames']} scored={cov['n_scored']}")
        print(f"   verdict -> {self.out_dir / 'verdict.json'}", flush=True)
