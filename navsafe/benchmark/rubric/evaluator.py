"""Score one episode: (trace, scenario program) -> pass/fail + why.

The whole benchmark's re-scorability rests on this being a pure function.  It
opens no simulator, renders nothing, and reads no global state, so re-tuning a
threshold means re-running this over stored traces in seconds.

Rubric semantics (taxonomy doc IV-B):
  * every success predicate must hold, AND
  * no hard gate may be violated -- a gate violation fails the episode
    regardless of the success block.
Diagnostics are computed and reported but gate nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from navsafe.benchmark.rubric import predicates as P
from navsafe.benchmark.rubric.schema import ScenarioProgram


@dataclass
class EpisodeResult:
    episode_id: str
    seed_id: str
    family: str
    regime: str
    passed: bool
    gate_violated: bool
    success: list[dict] = field(default_factory=list)
    gates: list[dict] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        head = "PASS" if self.passed else ("GATE" if self.gate_violated else "FAIL")
        why = next((v["reason"] for v in self.gates if not v["passed"]), "") or \
              next((v["reason"] for v in self.success if not v["passed"]), "all predicates held")
        return f"[{head}] {self.family} {self.seed_id} ({self.regime}): {why}"


def diagnostics(frames: Sequence[dict], wanted: Sequence[str]) -> dict[str, Any]:
    fs = P.scorable(frames)
    if not fs:
        return {k: None for k in wanted}
    out: dict[str, Any] = {}
    for name in wanted:
        if name == "min_clearance":
            # The near-miss magnitude the gate no longer fails on.
            vals = [a.get("clearance_m") for f in fs for a in f.get("agents", ())
                    if a.get("clearance_m") is not None
                    and a.get("clearance_m") == a.get("clearance_m")]
            out[name] = round(min(vals), 3) if vals else None
        elif name == "min_ttc":
            ttcs = [a.get("ttc_s", float("inf")) for f in fs for a in f.get("agents", ())]
            finite = [t for t in ttcs if t == t and t != float("inf")]
            out[name] = round(min(finite), 3) if finite else None
        elif name == "time_in_intersection":
            out[name] = round(sum(1 for f in fs if f.get("in_intersection")) *
                              _dt(fs), 2)
        elif name == "n_stops":
            n, moving = 0, True
            for f in fs:
                if moving and f["ego_speed"] < 0.3:
                    n += 1
                    moving = False
                elif not moving and f["ego_speed"] > 1.0:
                    moving = True
            out[name] = n
        elif name == "max_abs_jerk":
            out[name] = round(max(abs(f.get("ego_jerk", 0.0)) for f in fs), 3)
        elif name == "max_lat_accel":
            out[name] = round(max(abs(f.get("ego_lat_accel", 0.0)) for f in fs), 3)
        elif name == "peak_steer_rate":
            rates = [abs(b.get("ego_steer", 0.0) - a.get("ego_steer", 0.0)) / _dt(fs)
                     for a, b in zip(fs, fs[1:])]
            out[name] = round(max(rates), 3) if rates else 0.0
        else:
            out[name] = None
    return out


def _dt(fs: Sequence[dict]) -> float:
    return (fs[1]["t_sim_s"] - fs[0]["t_sim_s"]) if len(fs) > 1 else 0.1


def evaluate(frames: Sequence[dict], program: ScenarioProgram, *,
             episode_id: str = "") -> EpisodeResult:
    # `t_max` is written in the rubric as the symbol `t_max_s`; resolve it here
    # so the YAML never repeats the number the program already carries.
    def resolve(params: dict) -> dict:
        return {k: (program.t_max_s if v == "t_max_s" else v)
                for k, v in params.items()}

    succ = [P.SUCCESS[p.name](frames, **resolve(p.params)) for p in program.success]
    gates = [P.GATES[g.name](frames, **resolve(g.params)) for g in program.gates]

    gate_violated = any(not g for g in gates)
    passed = (not gate_violated) and all(bool(s) for s in succ)

    total = len(frames)
    scored = len(P.scorable(frames))
    # Always 0 since the render-validity envelope was removed. The key stays
    # because runner/collect, show_verdict and replay_log read it, and 0 is now
    # the truth: no frame is excluded for deviating from the logged path.
    invalid = 0
    return EpisodeResult(
        episode_id=episode_id,
        seed_id=program.seed_id,
        family=program.family,
        regime=program.regime,
        passed=passed,
        gate_violated=gate_violated,
        success=[vars(v) for v in succ],
        gates=[vars(v) for v in gates],
        diagnostics=diagnostics(frames, program.diagnostics),
        coverage={
            "n_frames": total,
            "n_scored": scored,
            # Frames the reconstruction could not certify. Reported on every
            # episode so an exclusion never passes as full coverage.
            "n_render_invalid": invalid,
            "scored_fraction": round(scored / total, 3) if total else 0.0,
        },
    )
