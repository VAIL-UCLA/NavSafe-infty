"""Scenario-program schema: the declarative unit an episode is generated from.

Mirrors the taxonomy doc IV: every episode comes from a program with three
mandatory blocks --

    initialization   typed distributions over initial conditions, sampled per
                     episode and frozen into the manifest
    rubric           success as an executable predicate over the simulator
                     trace, plus always-on hard safety gates
    actors           the background-traffic policy per agent, which is what
                     distinguishes the three interaction regimes

The point of keeping this declarative rather than writing a Python function per
family is that the program is *data*: it goes into the frozen episode manifest,
it diffs across benchmark versions, and -- because scoring is a pure function
of (trace, rubric) -- changing a threshold re-scores existing traces without
re-running a single simulation.

Nothing here evaluates anything; this module only defines and validates the
shape.  The evaluator lives in ``rubric/evaluator.py`` and consumes a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from navsafe.errors import NavSafeError

# --- predicate library (taxonomy doc IV-B) ---------------------------------
# Each entry: name -> (required params, one-line meaning).  Predicates are
# evaluated over the scored part of the trace; "hold" means the whole episode
# must satisfy them, not just the final frame.
PREDICATES: dict[str, tuple[tuple[str, ...], str]] = {
    "reach":        (("region", "t_max"), "ego enters region within the time budget"),
    "stabilize":    (("region", "tau"), "ego stays inside region for tau seconds"),
    "align":        (("lane", "heading_tol"), "ego heading matches the lane within tol"),
    "stop_before":  (("line", "v_eps"), "ego comes to a near-stop before the line"),
    "clear":        (("conflict_zone",), "ego traverses the zone with no simultaneous "
                                         "occupancy by a conflicting agent"),
    "maintain_gap": (("agent", "d_min"), "distance to agent stays above the floor"),
    "comply":       (("signal",), "no motion through red; go on green within t_react"),
}

# Always-on hard gates: a violation fails the episode regardless of the
# success predicates (doc IV-B "gates: violation => fail regardless").
GATES: dict[str, tuple[tuple[str, ...], str]] = {
    "no_collision":  (("fault",), "no collision attributable per the fault rule"),
    "on_drivable":   ((), "ego stays within the drivable area"),
    "maintain_gap":  (("d_min",), "absolute near-miss floor, NOT a comfort margin"),
}

# Recorded but never gating -- they explain a failure, they do not cause one.
DIAGNOSTICS = ("min_clearance", "min_ttc", "time_in_intersection", "n_stops", "max_abs_jerk",
               "max_lat_accel", "peak_steer_rate")

# Background-actor policies (doc Table V).  The regime an episode runs under is
# just which of these its actors are assigned.
POLICIES = {
    "replay":      "follow the logged trajectory verbatim (non-reactive)",
    "idm":         "route-following with IDM longitudinal control (reactive)",
    "adversarial": "parameterised maneuver template with sampled parameters",
    "scripted":    "deterministic trigger-based motion, for paired unit tests",
}

REGIME_POLICIES = {
    "log_replay":     {"replay"},
    "reactive":       {"idm"},
    "safety_critical": {"idm", "adversarial"},
}


class ProgramError(NavSafeError, ValueError):
    """A scenario program that cannot be compiled into episodes."""


@dataclass
class Distribution:
    """A typed distribution over one initial condition."""

    kind: str                    # "uniform" | "normal" | "choice" | "const"
    params: dict[str, Any]

    def validate(self, where: str) -> None:
        need = {"uniform": ("low", "high"), "normal": ("mean", "std"),
                "choice": ("values",), "const": ("value",)}
        if self.kind not in need:
            raise ProgramError(f"{where}: unknown distribution kind {self.kind!r}")
        missing = [k for k in need[self.kind] if k not in self.params]
        if missing:
            raise ProgramError(f"{where}: {self.kind} needs {missing}")
        if self.kind == "uniform" and self.params["low"] > self.params["high"]:
            raise ProgramError(f"{where}: uniform low > high")


@dataclass
class Predicate:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self, where: str, table=PREDICATES) -> None:
        if self.name not in table:
            raise ProgramError(f"{where}: unknown predicate {self.name!r}; "
                               f"known: {sorted(table)}")
        required, _ = table[self.name]
        missing = [p for p in required if p not in self.params]
        if missing:
            raise ProgramError(f"{where}: {self.name} missing params {missing}")


@dataclass
class ScenarioProgram:
    """One family's program, instantiated against one seed."""

    family: str
    seed_id: str
    regime: str
    initialization: dict[str, Distribution]
    success: list[Predicate]
    gates: list[Predicate]
    actors: dict[str, str]              # agent selector -> policy name
    diagnostics: list[str] = field(default_factory=list)
    reject: list[str] = field(default_factory=list)
    t_max_s: float | None = None

    def validate(self) -> None:
        if self.regime not in REGIME_POLICIES:
            raise ProgramError(f"unknown regime {self.regime!r}")
        for key, dist in self.initialization.items():
            dist.validate(f"initialization.{key}")
        if not self.success:
            raise ProgramError("a program with no success predicate cannot pass "
                               "or fail for a stated reason")
        for p in self.success:
            p.validate("success")
        for g in self.gates:
            g.validate("gates", GATES)
        # The hard gates are always on; a program may tighten them but never
        # drop them, or "no collision" stops being a benchmark-wide invariant.
        named = {g.name for g in self.gates}
        for required in ("no_collision", "on_drivable"):
            if required not in named:
                raise ProgramError(f"gate {required!r} is mandatory")
        allowed = REGIME_POLICIES[self.regime]
        for sel, pol in self.actors.items():
            if pol not in POLICIES:
                raise ProgramError(f"actors.{sel}: unknown policy {pol!r}")
            if pol not in allowed:
                raise ProgramError(
                    f"actors.{sel}: policy {pol!r} is not part of regime "
                    f"{self.regime!r} (allowed: {sorted(allowed)})")
        for d in self.diagnostics:
            if d not in DIAGNOSTICS:
                raise ProgramError(f"unknown diagnostic {d!r}")

    # --- construction from the YAML form ----------------------------------
    @classmethod
    def from_dict(cls, d: dict, *, seed_id: str, regime: str) -> "ScenarioProgram":
        def preds(items) -> list[Predicate]:
            out = []
            for it in items or ():
                if isinstance(it, str):
                    out.append(Predicate(it))
                else:
                    (name, params), = it.items()
                    out.append(Predicate(name, params or {}))
            return out

        init = {k: Distribution(v["kind"], {kk: vv for kk, vv in v.items() if kk != "kind"})
                for k, v in (d.get("initialization") or {}).items()}
        prog = cls(
            family=d["family"],
            seed_id=seed_id,
            regime=regime,
            initialization=init,
            success=preds(d.get("success")),
            gates=preds(d.get("gates")),
            actors=d.get("actors") or {},
            diagnostics=list(d.get("diagnostics") or ()),
            reject=list(d.get("reject") or ()),
            t_max_s=d.get("t_max_s"),
        )
        prog.validate()
        return prog


def load(path, *, seed_id: str, regime: str) -> ScenarioProgram:
    import yaml
    from pathlib import Path
    return ScenarioProgram.from_dict(
        yaml.safe_load(Path(path).read_text()), seed_id=seed_id, regime=regime)
