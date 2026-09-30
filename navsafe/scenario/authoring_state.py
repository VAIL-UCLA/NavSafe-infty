# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``ScenarioState`` — the cross-stage symbolic IR.

A single dataclass survives all six pipeline stages. Each stage's designer tools
mutate ``self.state`` directly (via closures over the shared reference); between
stages we serialize to JSON for resumability (``scene_after_<stage>.json``).

The Scenic program is emitted at the end via
:func:`navsafe.text2sim.agent_utils.scenic_program_builder.build_scenic_source`.

See IMPLEMENTATION_GUIDE.md §2 for the design rationale (typed dataclass instead
of raw Scenic source).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

# ─────────────────────────── type aliases ───────────────────────────

WeatherT = Literal["clear", "rain", "fog", "snow", "overcast"]
TimeOfDayT = Literal["dawn", "morning", "noon", "afternoon", "dusk", "night"]
SeasonT = Literal["spring", "summer", "fall", "winter"]
PlacementKindT = Literal["fixed", "distribution"]
StaticKindT = Literal["traffic_signal", "sign", "parked_vehicle", "barrier", "vegetation", "debris"]
DynamicKindT = Literal["vehicle", "pedestrian", "cyclist"]
RequirementKindT = Literal["require", "require_always"]
StyleT = Literal["typical", "adversarial", "rare_event", "safety_critical"]


# ─────────────────────────── per-section sub-states ───────────────────────────


@dataclass
class WorldState:
    """Set by Stage A1 (World)."""

    weather: Optional[WeatherT] = None
    time_of_day: Optional[TimeOfDayT] = None
    season: Optional[SeasonT] = None
    lighting_profile: Optional[str] = None  # references catalogs/lighting_profiles.yaml

    def is_complete(self) -> bool:
        """A1 considers itself done when these four fields are set."""
        return all(
            v is not None
            for v in (self.weather, self.time_of_day, self.season, self.lighting_profile)
        )


@dataclass
class MapState:
    """Set by Stage A2 (Map)."""

    source: Optional[str] = None  # "carla:Town04" | "lgsvl:borregasave" | "osm:..." | "procgen:..."
    xodr_path: Optional[str] = None
    roi_bounds: Optional[tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    region_tags: dict[str, str] = field(default_factory=dict)  # name -> Scenic region expression

    def is_complete(self) -> bool:
        return self.source is not None and self.xodr_path is not None


@dataclass
class Placement:
    """Where an object lives in the world.

    A placement is either a *fixed* pose (``(x, y, heading)``) or a *distribution*
    expression that the Scenic sampler will resolve (``"on intersection_main"``,
    ``"Range(2, 5) elements on sidewalk_east"``, etc.).
    """

    kind: PlacementKindT
    on_region: Optional[str] = None  # name from MapState.region_tags
    pose: Optional[tuple[float, float, float]] = None  # (x, y, heading) for fixed
    scenic_expr: Optional[str] = None  # raw Scenic expression for distribution


@dataclass
class StaticObject:
    """An item in ``state.static_objects[]`` — set by Stage A3."""

    id: str
    kind: StaticKindT
    placement: Placement
    asset_tag: Optional[str] = None  # render-time hint (e.g. "urbanverse:semaphore_v3")
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EgoState:
    """Set by Stage B1 (Ego & task)."""

    spawn_region: Optional[str] = None  # region tag name or 'lane:lane_id'
    vehicle_class: str = "sedan"
    route: list[str] = field(default_factory=list)  # ordered lane/region references
    goal: Optional[dict[str, Any]] = None  # task spec — reach point, follow lane, etc.
    ego_replay_frames: int = 0


@dataclass
class DynamicAgent:
    """An item in ``state.dynamic_agents[]`` — set by Stage B2."""

    id: str
    kind: DynamicKindT
    vehicle_class: Optional[str] = None  # for kind="vehicle"
    pedestrian_class: Optional[str] = None
    cyclist_class: Optional[str] = None
    placement: Placement = field(default_factory=lambda: Placement(kind="distribution"))
    spatial_relations: list[str] = field(default_factory=list)  # raw Scenic specifier exprs
    count: str = "1"  # "1" | "Range(3, 7)" | "Options([2, 4, 6])"


@dataclass
class BehaviorAssignment:
    """A value in ``state.behaviors{}`` — set by Stage B3.

    Keys of ``state.behaviors`` are agent IDs (``DynamicAgent.id``).
    """

    behavior: str  # "IDM" | "LaneFollow" | "AnchorDrive" | "Stop" | "JaywalkingPedestrian" | ...
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenicRequirement:
    """An item in ``state.requirements[]`` — set by Stage B3 (and verifier feedback)."""

    kind: RequirementKindT
    expr: str
    note: Optional[str] = None  # human-readable description (also used by lint_require_always)


@dataclass
class ScenarioMetadata:
    """Pipeline-level bookkeeping. Not part of the scenario itself."""

    seed: int = 0
    style: StyleT = "typical"
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    # Each entry: {stage_id, iteration, scores: {category: score}, started_at, ended_at}
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_version: str = "0.1.0"


# ─────────────────────────── top-level container ───────────────────────────


@dataclass
class ScenarioState:
    """The cross-stage IR. Survives all six stages; serialized to JSON between them."""

    prompt: str
    world: WorldState = field(default_factory=WorldState)
    map: MapState = field(default_factory=MapState)
    static_objects: list[StaticObject] = field(default_factory=list)
    ego: EgoState = field(default_factory=EgoState)
    dynamic_agents: list[DynamicAgent] = field(default_factory=list)
    behaviors: dict[str, BehaviorAssignment] = field(default_factory=dict)
    requirements: list[ScenicRequirement] = field(default_factory=list)
    metadata: ScenarioMetadata = field(default_factory=ScenarioMetadata)

    # ─── stage progress helpers ───

    STAGE_ORDER: tuple[str, ...] = ()  # filled below to avoid mutable-default sin

    def record_stage(
        self,
        stage_id: str,
        iteration: int,
        scores: Mapping[str, float] | None,
        started_at: str | None = None,
        ended_at: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Append an entry to ``metadata.stage_history``."""
        self.metadata.stage_history.append(
            {
                "stage_id": stage_id,
                "iteration": iteration,
                "scores": scores or {},
                "started_at": started_at,
                "ended_at": ended_at or datetime.now(timezone.utc).isoformat(),
                "notes": notes,
            }
        )

    def stages_completed(self) -> set[str]:
        """The set of stage_ids that have appeared at least once in stage_history."""
        return {entry["stage_id"] for entry in self.metadata.stage_history}

    # ─── (de)serialization ───
    #
    # We roll our own ``to_dict`` / ``from_dict`` because:
    # 1. ``dataclasses.asdict`` mangles tuples (``roi_bounds``) into lists silently
    #    — which is fine for JSON but breaks round-trip type checks.
    # 2. We want ``Placement.kind`` literals to survive round-trip as plain strings
    #    (they do — but we want to validate them on load).
    # 3. We may add custom-typed fields later (e.g. typed Distribution ADT) without
    #    rewriting the serializer.

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenarioState":
        return _from_dict(cls, d)

    @classmethod
    def from_json(cls, s: str) -> "ScenarioState":
        return cls.from_dict(json.loads(s))

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioState":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# canonical stage ordering used by the pipeline runner and resume logic
ScenarioState.STAGE_ORDER = (
    "world",
    "map",
    "static_infra",
    "ego_task",
    "dynamic_agent",
    "behavior",
)


# ─────────────────────────── serialization internals ───────────────────────────

_LIST_FIELDS_OF_DATACLASS: dict[type, dict[str, type]] = {
    ScenarioState: {
        "static_objects": StaticObject,
        "dynamic_agents": DynamicAgent,
        "requirements": ScenicRequirement,
    },
}
_DICT_FIELDS_OF_DATACLASS: dict[type, dict[str, type]] = {
    ScenarioState: {"behaviors": BehaviorAssignment},
}


def _to_dict(obj: Any) -> Any:
    """Recursive dataclass→dict that preserves tuples as lists (JSON-safe)."""
    if is_dataclass(obj):
        out: dict[str, Any] = {}
        for f in fields(obj):
            out[f.name] = _to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Recursive dict→dataclass that tolerates extra keys (forward-compat)."""
    if not is_dataclass(cls):
        return d

    list_field_types = _LIST_FIELDS_OF_DATACLASS.get(cls, {})
    dict_field_types = _DICT_FIELDS_OF_DATACLASS.get(cls, {})

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if v is None:
            kwargs[f.name] = None
            continue

        # nested dataclass?
        if isinstance(f.type, type) and is_dataclass(f.type):
            kwargs[f.name] = _from_dict(f.type, v)
            continue

        # known list-of-dataclass field?
        if f.name in list_field_types:
            kwargs[f.name] = [_from_dict(list_field_types[f.name], x) for x in v]
            continue

        # known dict-of-dataclass field?
        if f.name in dict_field_types:
            inner = dict_field_types[f.name]
            kwargs[f.name] = {k: _from_dict(inner, x) for k, x in v.items()}
            continue

        # named sub-dataclasses on ScenarioState (resolved by attribute name not type)
        if cls is ScenarioState:
            sub_map: dict[str, type] = {
                "world": WorldState,
                "map": MapState,
                "ego": EgoState,
                "metadata": ScenarioMetadata,
            }
            if f.name in sub_map:
                kwargs[f.name] = _from_dict(sub_map[f.name], v)
                continue

        # nested dataclass within sub-types
        if cls in (StaticObject, DynamicAgent) and f.name == "placement":
            kwargs[f.name] = _from_dict(Placement, v)
            continue

        # restore tuples for known tuple-typed fields
        if cls is MapState and f.name == "roi_bounds" and isinstance(v, list):
            kwargs[f.name] = tuple(v)
            continue
        if cls is Placement and f.name == "pose" and isinstance(v, list):
            kwargs[f.name] = tuple(v)
            continue

        kwargs[f.name] = v

    return cls(**kwargs)


# ─────────────────────────── validation helpers ───────────────────────────


def assigned_agent_ids(state: ScenarioState) -> set[str]:
    return {agent.id for agent in state.dynamic_agents}


def unassigned_behaviors(state: ScenarioState) -> set[str]:
    """Agent IDs that have no behavior assigned — used by B3's critic feasibility."""
    return assigned_agent_ids(state) - set(state.behaviors.keys())


def assert_state_invariants(state: ScenarioState) -> list[str]:
    """Cheap structural checks; returns a list of violations (empty = ok).

    Used by tests and by the pipeline runner as a sanity gate between stages.
    """
    violations: list[str] = []

    # unique IDs
    static_ids = [s.id for s in state.static_objects]
    if len(static_ids) != len(set(static_ids)):
        violations.append(f"duplicate static_object ids: {sorted(static_ids)}")
    dyn_ids = [a.id for a in state.dynamic_agents]
    if len(dyn_ids) != len(set(dyn_ids)):
        violations.append(f"duplicate dynamic_agent ids: {sorted(dyn_ids)}")

    # behaviors reference known agents
    unknown_b = set(state.behaviors.keys()) - set(dyn_ids)
    if unknown_b:
        violations.append(f"behaviors reference unknown agent ids: {sorted(unknown_b)}")

    # placement on_region references a known map region
    known_regions = set(state.map.region_tags.keys())
    for so in state.static_objects:
        if so.placement.on_region and so.placement.on_region not in known_regions:
            # only flag if map stage has happened
            if state.map.region_tags:
                violations.append(
                    f"static_object {so.id!r} placed on unknown region "
                    f"{so.placement.on_region!r}"
                )
    for ag in state.dynamic_agents:
        if ag.placement.on_region and ag.placement.on_region not in known_regions:
            if state.map.region_tags:
                violations.append(
                    f"dynamic_agent {ag.id!r} placed on unknown region "
                    f"{ag.placement.on_region!r}"
                )

    return violations


__all__ = [
    "ScenarioState",
    "WorldState",
    "MapState",
    "StaticObject",
    "Placement",
    "EgoState",
    "DynamicAgent",
    "BehaviorAssignment",
    "ScenicRequirement",
    "ScenarioMetadata",
    "WeatherT",
    "TimeOfDayT",
    "SeasonT",
    "PlacementKindT",
    "StaticKindT",
    "DynamicKindT",
    "RequirementKindT",
    "StyleT",
    "assigned_agent_ids",
    "unassigned_behaviors",
    "assert_state_invariants",
]
