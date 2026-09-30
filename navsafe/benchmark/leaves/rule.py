# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A leaf's ``insert:`` rule, expanded against one host into an authoring spec.

Every leaf used to be built from a per-HOST spec YAML that someone wrote by
hand. Those files were the one irreproducible step in the pipeline: seventeen
accumulated, fourteen unreferenced, two generations of conventions side by
side, and the ones still wired in carried numbers like ``arc: 28.0`` and
``conflict_frame: 30`` that were measured against one clip and one render.

Nothing below this module needs them. ``author_recipe`` already resolves
``auto`` arcs off ``probe.straight_window()``, laterals off
``probe.cross_section()``, and timing off ``probe.ego_arrival_frame()`` — it
only ever needed a dict describing INTENT. So the rule lives once per leaf, in
``leaves/<LEAF>.yaml``, beside the predicates that select the host and the
metrics that judge it, and this module turns it into that dict.

Two mechanics carry all nine leaves, and neither is an expression language:

* ``count: all`` — one actor per matching registry key. That is how R-3 becomes
  the whole pedestrian bank and R-4 all four animals in a single scenario, so
  leaf variety is a property of the BANK rather than of who typed the spec.
* **scalar or list** — any value under ``authored:`` may be a list, indexed by
  slot and cycled if short. R-4's four animals differ only in speed
  (3.0 / 2.2 / 3.4 / 3.0); that is one line, not four near-identical blocks.

A leaf whose ``insert:`` declares an empty cast is the MINED case: no actors,
``provenance: mined``, judged on the road itself. It is written ``cast: []``
rather than omitted, because an absent block is an unfinished leaf and an empty
one is a decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)


class RuleError(NexusSimError, ValueError):
    """A leaf's ``insert:`` rule cannot be expanded for this host."""


@dataclass
class CastSlot:
    """One kind of actor in a leaf, and how many of it."""

    slot: str
    assets: Dict[str, Any] = field(default_factory=dict)
    count: Any = 1                      # int, or "all"
    variety: str = "fixed"              # "fixed" | "rotate"
    op: str = "insert"
    track: Dict[str, str] = field(default_factory=dict)
    render_class: str = ""
    leaf_variety_slot: str = ""
    source_track_id: str = ""
    keep_appearance: bool = False
    authored: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CastSlot":
        slot = str(d.get("slot", "")).strip()
        if not slot:
            raise RuleError("a cast slot must name a `slot:` — it becomes the actor name")
        count = d.get("count", 1)
        if count != "all":
            try:
                count = int(count)
            except (TypeError, ValueError):
                raise RuleError(
                    f"cast slot {slot!r}: count must be a whole number or 'all', "
                    f"got {count!r}") from None
            if count < 1:
                raise RuleError(f"cast slot {slot!r}: count must be at least 1")
        variety = str(d.get("variety", "fixed"))
        if variety not in ("fixed", "rotate"):
            raise RuleError(f"cast slot {slot!r}: variety must be 'fixed' or 'rotate'")
        return cls(
            slot=slot,
            assets=dict(d.get("assets") or {}),
            count=count,
            variety=variety,
            op=str(d.get("op", "insert")),
            track=dict(d.get("track") or {}),
            render_class=str(d.get("render_class", "")),
            leaf_variety_slot=str(d.get("leaf_variety_slot", "")),
            source_track_id=str(d.get("source_track_id", "")),
            keep_appearance=bool(d.get("keep_appearance", False)),
            authored=dict(d.get("authored") or {}),
        )


@dataclass
class LeafRule:
    """Everything a leaf declares about how it is BUILT, once, host-agnostic."""

    world_version: str = "car2sim_6cam_static@nre-ga"
    ego: Dict[str, Any] = field(default_factory=lambda: {"replay_frames": 8})
    layout: Dict[str, float] = field(default_factory=dict)
    clear_path: Dict[str, Any] = field(default_factory=dict)
    requires: Dict[str, Any] = field(default_factory=dict)
    pair_id: str = ""
    background_traffic: str = "keep"
    cast: List[CastSlot] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Optional[dict], *, leaf: str = "") -> "LeafRule":
        d = dict(d or {})
        if "cast" not in d:
            raise RuleError(
                f"leaf {leaf or '?'}: `insert:` must declare a `cast:` — write `cast: []` "
                f"for a mined leaf. An absent cast is an unfinished leaf; an empty one is "
                f"a decision, and only the second can be told apart from a typo.")
        return cls(
            world_version=str(d.get("world_version", "car2sim_6cam_static@nre-ga")),
            ego=dict(d.get("ego") or {"replay_frames": 8}),
            layout=dict(d.get("layout") or {}),
            clear_path=dict(d.get("clear_path") or {}),
            requires=dict(d.get("requires") or {}),
            pair_id=str(d.get("pair_id", "")),
            background_traffic=str(d.get("background_traffic", "keep")),
            cast=[CastSlot.from_dict(c) for c in (d.get("cast") or [])],
        )

    @property
    def inserts(self) -> bool:
        return bool(self.cast)


# ── asset selection ─────────────────────────────────────────────────────────


def select_assets(registry, selector: Dict[str, Any]) -> List[str]:
    """Registry keys matching a selector, in sorted (stable) order.

    Understood keys: ``keys`` (explicit list, used verbatim), ``family``,
    ``source``, ``leaf``, ``species``, ``present`` (default True — only assets
    whose PLY is actually on disk).

    Anything dropped for being absent is reported at WARNING and returned to the
    caller through the log, because a rule that silently drops to four animals
    is worse than one that says which are missing.
    """
    explicit = selector.get("keys")
    if explicit:
        missing = [k for k in explicit if k not in registry]
        if missing:
            raise RuleError(f"asset selector names unknown registry keys: {missing}")
        return [str(k) for k in explicit]

    want_present = bool(selector.get("present", True))
    out, absent = [], []
    for key in sorted(registry.entries):
        entry = registry.entries[key]
        if selector.get("family") and entry.family != selector["family"]:
            continue
        if selector.get("source") and entry.source != selector["source"]:
            continue
        if selector.get("species") and entry.species != selector["species"]:
            continue
        if selector.get("leaf") and selector["leaf"] not in entry.leaves:
            continue
        if want_present and not entry.present:
            absent.append(key)
            continue
        out.append(key)
    # LOUDLY, not at debug. The registry is a shopping list — most of it is
    # declared and not yet on disk, deliberately — and `count: all` combined
    # with a silent `present` filter means a leaf asking for "every animal"
    # quietly builds from whichever ones happen to exist. R-4 asked for the
    # whole bank and got lion / elephant / dog / horse, because the eight
    # UrbanVerse animals beside them had never been acquired, and nothing said
    # so: the bake succeeded, the recipe froze clean, and only the render
    # showed a 4.3 m elephant crossing a Las Vegas street. (The bank now
    # carries dog / cow / hippo / horse; the gap it hid is still there.)
    if absent:
        logger.warning(
            "asset selector %s matches %d asset(s) that are NOT on disk and were dropped: "
            "%s. Anything built from this selector is a SUBSET of what the leaf declares — "
            "run `navsafe assets acquire` to close the gap before believing a render.",
            selector, len(absent), ", ".join(absent),
        )
    if not out:
        raise RuleError(
            f"no asset in the registry matches {selector!r}. Either the bank does not "
            f"carry this family yet (`navsafe assets status` lists what is missing), or "
            f"the selector is wrong — do NOT widen it to whatever is nearest.")
    return out


# ── expansion ───────────────────────────────────────────────────────────────


def _per_slot(value: Any, index: int) -> Any:
    """A rule value for slot ``index``: lists are indexed and cycled, scalars broadcast."""
    if isinstance(value, list) and value and not _is_polyline(value):
        return value[index % len(value)]
    return value


def _is_polyline(value: list) -> bool:
    """A list of points is DATA, not a per-slot schedule.

    ``reference: [[x, y], ...]`` and ``waypoints: [[frame, lateral], ...]`` are
    both lists whose elements are lists; indexing them per slot would silently
    hand actor 2 a single waypoint.
    """
    return all(isinstance(v, (list, tuple)) for v in value)


def _fill(text: str, **kw) -> str:
    try:
        return text.format(**kw)
    except (KeyError, IndexError):
        return text


def expand_leaf_rule(man, *, scene_id: str, host, registry,
                     variety: int = 0, insert: bool = True,
                     recipe_id: str = "", selection: Optional[dict] = None) -> Dict[str, Any]:
    """A leaf manifest + one host -> the authoring spec ``author_recipe`` eats.

    ``insert=False`` suppresses the cast (C-10 on a host whose log already
    supplies oncoming traffic), which is how one rule covers both outcomes of
    a conditional leaf without a human choosing a branch.
    """
    rule = man.rule
    actors: Dict[str, Any] = {}
    names: List[str] = []

    if insert:
        for cast in rule.cast:
            keys = select_assets(registry, cast.assets)
            if cast.count == "all":
                chosen = keys
            elif cast.variety == "rotate":
                start = variety % len(keys)
                chosen = [keys[(start + i) % len(keys)] for i in range(min(cast.count, len(keys)))]
            else:
                if cast.count > len(keys):
                    raise RuleError(
                        f"cast slot {cast.slot!r} asks for {cast.count} assets but only "
                        f"{len(keys)} match {cast.assets!r}")
                chosen = keys[: cast.count]

            total = len(chosen)
            for i, key in enumerate(chosen):
                entry = registry.entries[key]
                name = cast.slot if total == 1 else f"{cast.slot}_{i + 1}"
                names.append(name)

                authored = {k: _per_slot(v, i) for k, v in cast.authored.items()}
                # PER-HOST OVERRIDES, declared in the leaf's `hosts:` block and
                # keyed by token. One leaf is one spec, and that is right until
                # a host's geometry makes the spec mean something else on it:
                # V-10's lane change wants the ego's left on a host already in
                # the rightmost lane, and its timing wants a different frame
                # where the ego is stopped in traffic. The alternative -- hand
                # editing the frozen recipe -- produces a file `bake` can no
                # longer reproduce, which is how this set drifted twice before.
                # Overrides are declarative, re-bakeable and visible in one
                # place; the recipe records which keys were overridden.
                # Keyed by TOKEN. `scene_id` here is the py123d scene UUID, not
                # the host's name -- looking the override up by it silently
                # matched nothing and the leaf's defaults were used, which reads
                # exactly like the override not being implemented.
                tok = str(getattr(host, "scene_id", "") or "").split("_")[0]
                over = dict((getattr(man, "hosts", None) or {}).get(tok, {}))
                if over:
                    authored.update({k: _per_slot(v, i) for k, v in over.items()})
                    authored["_host_override"] = sorted(over)
                for field_name, value in list(authored.items()):
                    if isinstance(value, str):
                        authored[field_name] = _fill(
                            value, i=i + 1, n=total, key=key,
                            species=entry.species or entry.family)

                track = {"type": cast.track.get("type") or entry.track_type or "VEHICLE"}
                # The render class may be pinned by the leaf (an override the
                # leaf contract makes visible) or by the asset's registry row
                # (the default for that class of object). Empty means the
                # renderer picks by track type, which is right for most assets
                # and fatal for a car — see the preflight in author.py.
                render_class = (cast.render_class or cast.track.get("semantic_class")
                                or entry.render_class)
                if render_class:
                    track["semantic_class"] = render_class

                actor: Dict[str, Any] = {
                    "op": cast.op,
                    "track": track,
                    "asset": {"registry_key": key},
                    "authored": authored,
                }
                if cast.source_track_id:
                    actor["source_track_id"] = cast.source_track_id
                if cast.keep_appearance:
                    actor["keep_appearance"] = True
                if cast.leaf_variety_slot:
                    actor["leaf_variety_slot"] = _fill(
                        cast.leaf_variety_slot, i=i + 1, n=total, key=key,
                        v=variety + 1, V=len(keys),
                        species=entry.species or entry.family)
                actors[name] = actor

    spec: Dict[str, Any] = {
        "recipe_id": recipe_id or f"{man.leaf}/{man.scenario}/{scene_id}/001",
        "leaf": man.leaf,
        "scenario": man.scenario,
        "provenance": "constructed" if actors else "mined",
        "host": {"scene": scene_id, "world_version": rule.world_version},
        # The rig is a property of how the clip was RECONSTRUCTED, not of the
        # leaf: the same leaf is built on hosts from both families.
        "ego": {**rule.ego, "cam_height": host.cam_height},
        "actors": actors,
        "background_traffic": rule.background_traffic,
        "review": {"status": "pending", "checklist": man.checklist},
    }
    if rule.layout:
        spec["layout"] = dict(rule.layout)
    if rule.clear_path:
        spec["clear_path"] = dict(rule.clear_path)
    if rule.requires:
        spec["requires"] = dict(rule.requires)
    if actors:
        # Every inserted actor is what the e0 variant removes: the pair differs
        # by the leaf's event and nothing else.
        spec["pair"] = {"e0_removes": names, "pair_id": rule.pair_id or man.scenario}
    if selection:
        spec["selection"] = dict(selection)
    return spec


__all__ = ["CastSlot", "LeafRule", "RuleError", "expand_leaf_rule", "select_assets"]
