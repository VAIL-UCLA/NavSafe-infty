# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The per-event-type contract: one YAML per event type, loaded here.

An event type's build recipe used to live in four places at once — a prose checklist
in ``<LEAF>.md``, a predicate list hard-coded in ``qualify.py``, an authoring
spec under ``specs/``, and (for the mined event types) a script in someone's
``/tmp``. Changing how an event type is built meant finding all four. This module is
the single declaration; every stage reads it:

    navsafe mine     --event-type V-10     ->  manifest.mine   (which miner, params)
    navsafe qualify  --event-type C-10     ->  manifest.qualify (geometry predicates)
    navsafe bake     --event-type ...      ->  manifest.rule   (the `insert:` block,
                                         expanded against one host)
    the reviewer                     ->  manifest.checklist (the .md)
    scoring                          ->  manifest.metrics

Both tiers end at a recipe, so ``tier`` decides which *route* to a recipe a
event type takes, not whether it produces one:

``mined``        a miner selects a logged clip; the recipe pins host + window +
                 the selection evidence, and edits nothing.
``constructed``  the event type's ``insert:`` rule is expanded against a host into one
                 spawn pose and one controller per actor.

There used to be a third, ``mined+insert``: mined for the road, then an insert
supplied the consequence only where the log lacked one. It made an event type's
DEFINING event conditional on the clip, which is how C-7 — the head-on event type —
came to have a frozen recipe with no oncoming car in it. An event type either builds
its event or selects a host that already has it; both are honest, and which one
it is should not depend on the clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from navsafe.benchmark.event_types.rule import LeafRule, RuleError
from navsafe.errors import NavSafeError

LEAVES_DIR = Path(__file__).parent
TIERS = ("mined", "constructed")


class LeafError(NavSafeError, ValueError):
    """The event type manifest is missing, malformed, or names something unknown."""


@dataclass
class LeafManifest:
    """What one event type needs, from selection through review."""

    leaf: str
    name: str
    scenario: str
    tier: str
    summary: str = ""
    # selection
    mine: Dict[str, Any] = field(default_factory=dict)      # {miner, params}
    qualify: List[str] = field(default_factory=list)        # geometry predicates (hard gate)
    qualify_info: List[str] = field(default_factory=list)   # reported, never rejects
    # construction — the whole build rule, host-agnostic (see leaves/rule.py)
    insert: Dict[str, Any] = field(default_factory=dict)
    #: Per-host deviations from that rule, keyed by nuPlan token, merged over
    #: the cast's `authored` at expansion. The event type stays one spec; this is
    #: where a host whose geometry makes that spec mean something else says so,
    #: in the same file and in a form `bake` can reproduce. Hand-editing the
    #: frozen recipe was the alternative, and it produces a file no bake can
    #: rebuild.
    hosts: Dict[str, Any] = field(default_factory=dict)
    # judgment
    metrics: List[str] = field(default_factory=list)
    checklist: str = ""
    notes: str = ""
    # parsed from `insert:` at load time
    rule: "LeafRule" = field(default_factory=lambda: LeafRule(cast=[]))

    @property
    def is_mined(self) -> bool:
        return self.tier == "mined"

    @property
    def inserts(self) -> bool:
        return self.tier == "constructed"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeafManifest":
        missing = [k for k in ("leaf", "name", "scenario", "tier") if not d.get(k)]
        if missing:
            raise LeafError(f"event type manifest is missing {missing}")
        if d["tier"] not in TIERS:
            raise LeafError(f"event type {d['leaf']}: tier must be one of {list(TIERS)}, got {d['tier']!r}")
        man = cls(
            leaf=str(d["leaf"]), name=str(d["name"]), scenario=str(d["scenario"]),
            tier=str(d["tier"]), summary=str(d.get("summary", "")),
            mine=dict(d.get("mine") or {}),
            qualify=[str(x) for x in (d.get("qualify") or [])],
            qualify_info=[str(x) for x in (d.get("qualify_info") or [])],
            insert=dict(d.get("insert") or {}),
            hosts=dict(d.get("hosts") or {}),
            metrics=[str(x) for x in (d.get("metrics") or [])],
            checklist=str(d.get("checklist", f"{d['leaf']}.md")),
            notes=str(d.get("notes", "")),
        )
        # Parsing here rather than at bake time means a malformed rule is a
        # broken event type file, caught by the manifest test, not a surprise three
        # commands into a build.
        try:
            man.rule = LeafRule.from_dict(man.insert, leaf=man.leaf)
        except RuleError as exc:
            raise LeafError(f"event type {man.leaf}: {exc}") from None
        if man.inserts and not man.rule.cast:
            raise LeafError(
                f"event type {man.leaf}: tier {man.tier!r} inserts actors, so its `insert.cast:` "
                f"must declare at least one slot"
            )
        if not man.inserts and man.rule.cast:
            raise LeafError(
                f"event type {man.leaf}: tier {man.tier!r} is mined — it is judged on the road "
                f"itself — but `insert.cast:` declares {len(man.rule.cast)} slot(s). Either "
                f"the tier is really 'constructed' or the cast should be []."
            )
        return man

    def checklist_path(self) -> Path:
        return LEAVES_DIR / self.checklist


def load_leaf(leaf: str, *, leaves_dir: Optional[Path] = None) -> LeafManifest:
    """Load one event type's manifest.

    Raises:
        LeafError: no manifest for that event type — listing what does exist, because
            a typo and an unbuilt event type need different answers.
    """
    d = Path(leaves_dir or LEAVES_DIR)
    path = d / f"{leaf}.yaml"
    if not path.is_file():
        raise LeafError(
            f"no manifest for event type {leaf!r} at {path}. Declared leaves: "
            f"{', '.join(sorted(available(leaves_dir=d))) or '(none)'}"
        )
    return LeafManifest.from_dict(yaml.safe_load(path.read_text()) or {})


def available(*, leaves_dir: Optional[Path] = None) -> List[str]:
    """Every event type with a manifest, sorted."""
    d = Path(leaves_dir or LEAVES_DIR)
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_all(*, leaves_dir: Optional[Path] = None) -> Dict[str, LeafManifest]:
    """Every manifest, keyed by event type."""
    return {leaf: load_leaf(leaf, leaves_dir=leaves_dir) for leaf in available(leaves_dir=leaves_dir)}


__all__ = ["LEAVES_DIR", "LeafError", "LeafManifest", "TIERS", "available", "load_all", "load_leaf"]
