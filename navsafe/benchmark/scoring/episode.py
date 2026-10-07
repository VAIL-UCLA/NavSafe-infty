"""Episode score and suite aggregation (paper Eq. 1 and Sec. III-E).

Per episode::

    S_i = C_i * D_i * (w_P P_i + w_R R_i + w_H H_i + w_Q Q_i)
          \\_______/   \\_________________________________________/
          safety gate            weighted behaviour term

and the suite score is ``NSS = 100 * mean(S_i)``.

Three things this module refuses to do, each on purpose:

* **It does not average the rubric away.**  Rubric pass/fail is the per-event
  primary and is reported per family; a mean over scenario types would hide
  precisely the attribution the benchmark exists to give.
* **It does not report only the mean.**  Averages hide brittle subgroups, so
  the worst populated group is reported alongside.
* **It does not tune its own weights.**  The weights are inputs, fixed by
  review, never fitted to a policy.  They are recorded in every result so a
  score can never be compared across different weightings by accident.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Sequence

from navsafe.benchmark.rubric import predicates as P


@dataclass(frozen=True)
class Weights:
    """Behaviour-term weights; must sum to 1 (paper Eq. 1)."""

    progress: float = 0.25
    rules: float = 0.25
    comfort: float = 0.25
    recovery: float = 0.25

    def validate(self) -> None:
        total = self.progress + self.rules + self.comfort + self.recovery
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1, got {total}")
        if min(self) < 0:
            raise ValueError("weights must be non-negative")

    def __iter__(self):
        return iter((self.progress, self.rules, self.comfort, self.recovery))

    def to_dict(self) -> dict:
        return asdict(self)


# TODO(review): these are placeholders until the paper's "fixed by preregistered
# review or sensitivity analysis" step happens. They are equal on purpose --
# an unjustified uneven weighting is worse than an obviously arbitrary even one.
DEFAULT_WEIGHTS = Weights()


@dataclass
class EpisodeScore:
    episode_id: str
    seed_id: str
    family: str
    regime: str
    policy: str
    rubric_passed: bool
    safety_gate: float          # C_i * D_i, in {0, 1}
    progress: float
    rules: float
    comfort: float
    recovery: float
    score: float
    labels: list[str] = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _clip01(v: float) -> float:
    return 0.0 if v != v else max(0.0, min(1.0, v))


def components(frames: Sequence[dict], rubric, labels: dict) -> dict[str, float]:
    """The four behaviour components, each normalised to [0, 1]."""
    fs = P.scorable(frames)
    if not fs:
        return {"progress": 0.0, "rules": 0.0, "comfort": 0.0, "recovery": 0.0}

    # Progress: how far along the rubric's own success criterion the ego got.
    # Binary would throw away the difference between "almost made it" and
    # "never moved", which is what a ranking needs.
    reach = next((v for v in rubric.success if v["name"] == "reach"), None)
    if reach and reach["passed"]:
        progress = 1.0
    elif reach and reach["evidence"].get("elapsed_s"):
        travelled = _path_length(fs)
        progress = _clip01(travelled / max(_path_length(frames), 1e-6))
    else:
        progress = 0.0

    # Rules: fraction of non-gate success predicates that held.
    non_gate = [v for v in rubric.success]
    rules = sum(1 for v in non_gate if v["passed"]) / len(non_gate) if non_gate else 1.0

    # Comfort: penalised by exceedance over the comfort envelope, not by a
    # hard threshold -- comfort is scored, never gating.
    from navsafe.benchmark.outcome.labeler import JERK_LIMIT, LAT_ACCEL_LIMIT
    jerk = max((abs(f.get("ego_jerk", 0.0)) for f in fs), default=0.0)
    lat = max((abs(f.get("ego_lat_accel", 0.0)) for f in fs), default=0.0)
    comfort = _clip01(1.0 - 0.5 * (max(0.0, jerk / JERK_LIMIT - 1.0)
                                   + max(0.0, lat / LAT_ACCEL_LIMIT - 1.0)))

    # Recovery: did the ego return to a safe, route-consistent state after its
    # first material deviation?
    from navsafe.benchmark.outcome.labeler import DELAYED_RECOVERY
    ev = labels.get("evidence", {}).get(DELAYED_RECOVERY)
    if ev is None:
        recovery = 1.0
    elif not ev.get("recovered", False):
        recovery = 0.0
    else:
        recovery = _clip01(1.0 - (ev.get("recovery_s", 0.0) - 3.0) / 10.0)

    return {"progress": progress, "rules": rules,
            "comfort": comfort, "recovery": recovery}


def _path_length(fs: Sequence[dict]) -> float:
    return sum(math.dist((a["ego_x"], a["ego_y"]), (b["ego_x"], b["ego_y"]))
               for a, b in zip(fs, fs[1:]))


def score_episode(frames, rubric, labels, *, policy: str = "",
                  weights: Weights = DEFAULT_WEIGHTS,
                  comfort_measurable: bool = True) -> EpisodeScore:
    """Episode score. ``comfort_measurable=False`` drops the comfort term.

    A teleported ego is placed at planned waypoints rather than driven, so it
    does not obey the dynamics the comfort term describes. Scoring it anyway
    would report precision without accuracy, and scoring it as zero would drag
    the benchmark score down for a property the run never measured -- so the
    term is dropped and its weight redistributed over the rest. The exclusion is
    recorded on the episode, because a silently renormalised score is worse than
    a missing one.
    """
    weights.validate()
    gate = 0.0 if rubric.gate_violated else 1.0
    comp = components(frames, rubric, labels)
    w = {"progress": weights.progress, "rules": weights.rules,
         "comfort": weights.comfort, "recovery": weights.recovery}
    if not comfort_measurable:
        dropped = w.pop("comfort")
        comp["comfort"] = float("nan")
        total = sum(w.values())
        w = {k: v + dropped * v / total for k, v in w.items()} if total else w
    behaviour = sum(w[k] * comp[k] for k in w)
    return EpisodeScore(
        episode_id=rubric.episode_id, seed_id=rubric.seed_id,
        family=rubric.family, regime=rubric.regime, policy=policy,
        rubric_passed=rubric.passed, safety_gate=gate,
        score=round(gate * behaviour, 6),
        weights={**{k: round(v, 6) for k, v in w.items()},
                 "comfort_excluded": not comfort_measurable},
        labels=labels.get("labels", []), coverage=rubric.coverage,
        **{k: (None if v != v else round(v, 6)) for k, v in comp.items()},
    )


# --- suite-level aggregation ----------------------------------------------

def aggregate(scores: Sequence[EpisodeScore], *, min_group: int = 5) -> dict:
    """NSS, per-family rubric pass rates, and the worst populated group.

    ``min_group`` guards the worst-group statistic: the minimum over
    thinly-populated cells is noise, not a finding.
    """
    if not scores:
        return {"n_episodes": 0}
    nss = 100.0 * statistics.fmean(s.score for s in scores)

    def group(key):
        out: dict[str, list[EpisodeScore]] = {}
        for s in scores:
            out.setdefault(key(s), []).append(s)
        return out

    per_family = {
        fam: {
            "n": len(g),
            # The rubric pass rate is the per-event primary and is never
            # averaged across families (doc VI).
            "rubric_pass_rate": round(sum(s.rubric_passed for s in g) / len(g), 4),
            "gate_violation_rate": round(sum(1 - s.safety_gate for s in g) / len(g), 4),
            "nss": round(100.0 * statistics.fmean(s.score for s in g), 3),
        }
        for fam, g in sorted(group(lambda s: s.family).items())
    }
    per_regime = {
        reg: {"n": len(g),
              "rubric_pass_rate": round(sum(s.rubric_passed for s in g) / len(g), 4),
              "nss": round(100.0 * statistics.fmean(s.score for s in g), 3)}
        for reg, g in sorted(group(lambda s: s.regime).items())
    }

    cells = {k: g for k, g in group(lambda s: f"{s.family}|{s.regime}").items()
             if len(g) >= min_group}
    worst = min(cells.items(), key=lambda kv: statistics.fmean(s.score for s in kv[1])) \
        if cells else None

    label_counts: dict[str, int] = {}
    for s in scores:
        for l in s.labels:
            label_counts[l] = label_counts.get(l, 0) + 1

    return {
        "n_episodes": len(scores),
        "nss": round(nss, 3),
        "per_family": per_family,
        "per_regime": per_regime,
        "worst_group": None if worst is None else {
            "group": worst[0], "n": len(worst[1]),
            "nss": round(100.0 * statistics.fmean(s.score for s in worst[1]), 3),
        },
        "n_groups_below_min": sum(
            1 for g in group(lambda s: f"{s.family}|{s.regime}").values()
            if len(g) < min_group),
        "outcome_label_counts": dict(sorted(label_counts.items())),
        "weights": scores[0].weights,
    }


def paired_delta(baseline: Sequence[EpisodeScore],
                 shifted: Sequence[EpisodeScore]) -> dict:
    """Paired generalisation gap (paper Eq. 2) over matched episodes.

    Only episodes present in both sets count; an unmatched episode has no
    counterfactual, so including it would report a distribution difference as
    if it were a causal effect.
    """
    b = {s.episode_id: s for s in baseline}
    s2 = {s.episode_id: s for s in shifted}
    common = sorted(set(b) & set(s2))
    if not common:
        return {"n_pairs": 0, "delta": None}
    deltas = [b[k].score - s2[k].score for k in common]
    return {
        "n_pairs": len(common),
        "n_unmatched": len(set(b) ^ set(s2)),
        "delta": round(statistics.fmean(deltas), 6),
        "delta_std": round(statistics.pstdev(deltas), 6) if len(deltas) > 1 else 0.0,
    }
