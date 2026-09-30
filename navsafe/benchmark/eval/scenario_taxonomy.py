# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""nuPlan scenario type -> NavSafe taxonomy leaf, and per-scenario provenance
metadata (taxonomy leaf, nuPlan label, inserted-actor flag) for the
`stitch_jobs.tsv` scenario set.

`LEAF_PRECURSORS` below is this repo's record of the 28-leaf taxonomy (10
crash, 11 violation, 4 VRU-crash, 3 incident) and its per-leaf nuPlan precursor
lists; the design document it was transcribed from is not checked in here.
`unknown` is not a precursor of any crash-mechanism leaf — it sits under I-3
General Incident deliberately, because it is nuPlan's second-largest type and
dropping it would misrepresent coverage.

None of the published scenarios currently have inserted synthetic actors —
every bundle is scored against what the reconstruction captured as-is. That
is a fact about today's dataset, not a taxonomy property, so it is recorded
per scenario (`has_inserted_actors`) rather than folded into the leaf lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# leaf -> the nuPlan scenario types that can precede it. The taxonomy's ★/†
# frequency and provenance markers are not encoded here.
LEAF_PRECURSORS: dict[str, tuple[str, ...]] = {
    # 1. Traffic Crashes
    "C-1": ("stopping_with_lead", "behind_long_vehicle", "near_long_vehicle",
            "stopping_at_traffic_light_with_lead", "stopping_at_stop_sign_with_lead",
            "stationary_at_traffic_light_with_lead", "accelerating_at_traffic_light_with_lead"),
    "C-2": ("crossed_by_vehicle", "starting_protected_cross_turn"),
    "C-3": ("changing_lane_with_lead", "changing_lane_with_trail", "near_high_speed_vehicle"),
    "C-4": ("near_multiple_vehicles",),
    "C-5": ("traversing_intersection", "on_intersection", "starting_left_turn",
            "starting_right_turn", "starting_low_speed_turn", "starting_high_speed_turn",
            "starting_protected_noncross_turn"),
    "C-6": ("traversing_pickup_dropoff", "on_pickup_dropoff"),
    "C-7": ("traversing_narrow_lane",),
    "C-8": ("on_carpark",),  # context only -- nuPlan mines no reverse maneuver
    "C-9": (),   # empty; reached only by escalation from V-4 / C-6
    "C-10": (),  # empty
    # 2. Traffic Violations
    "V-1": ("traversing_traffic_light_intersection", "on_traffic_light_intersection",
            "on_stopline_traffic_light", "starting_straight_traffic_light_intersection_traversal",
            "stopping_at_traffic_light_without_lead", "stationary_at_traffic_light_without_lead",
            "accelerating_at_traffic_light", "accelerating_at_traffic_light_without_lead",
            "on_stopline_stop_sign", "starting_straight_stop_sign_intersection_traversal",
            "stopping_at_stop_sign_without_lead", "stopping_at_stop_sign_no_crosswalk",
            "accelerating_at_stop_sign", "accelerating_at_stop_sign_no_crosswalk"),
    "V-2": ("starting_unprotected_cross_turn", "starting_unprotected_noncross_turn",
            "on_all_way_stop_intersection"),
    "V-3": ("changing_lane", "changing_lane_to_left", "changing_lane_to_right"),
    "V-4": ("high_lateral_acceleration", "high_magnitude_jerk"),
    "V-5": ("high_magnitude_speed", "medium_magnitude_speed", "low_magnitude_speed"),
    "V-6": ("following_lane_with_lead", "following_lane_with_slow_lead"),
    "V-7": ("following_lane_without_lead",),
    "V-8": ("starting_u_turn",),
    "V-9": ("stationary", "stationary_in_traffic"),
    "V-10": (),  # empty; nuPlan mines no on-ramp/merge type
    "V-11": (),  # empty
    # 3. Vulnerable Road User Crashes
    "R-1": ("waiting_for_pedestrian_to_cross", "near_pedestrian_on_crosswalk",
            "near_pedestrian_on_crosswalk_with_ego", "near_multiple_pedestrians",
            "behind_pedestrian_on_driveable", "behind_pedestrian_on_pickup_dropoff",
            "near_pedestrian_at_pickup_dropoff", "traversing_crosswalk", "on_stopline_crosswalk",
            "stopping_at_crosswalk", "stationary_at_crosswalk", "accelerating_at_crosswalk"),
    "R-2": ("behind_bike", "crossed_by_bike", "near_multiple_bikes"),
    "R-3": (),  # empty
    "R-4": (),  # empty
    # 4. Traffic Incidents
    "I-1": ("near_trafficcone_on_driveable", "near_barrier_on_driveable"),
    "I-2": ("near_construction_zone_sign",),
    # residual leaf: catches nuPlan's own `unknown` tag (2nd-largest type) plus
    # any type not claimed by another leaf, per the taxonomy doc's explicit
    # instruction not to drop `unknown` silently.
    "I-3": ("unknown",),
}

LEAF_NAMES: dict[str, str] = {
    "C-1": "Rear-End", "C-2": "Angle / T-Bone", "C-3": "Sideswipe",
    "C-4": "Multi-Vehicle", "C-5": "Intersection", "C-6": "Roadway Departure",
    "C-7": "Head-On", "C-8": "Backing", "C-9": "Single-Vehicle", "C-10": "Wrong-Way (crash)",
    "V-1": "Red-Light / Stop-Sign", "V-2": "Failure to Yield",
    "V-3": "Unsafe Lane Change / Passing", "V-4": "Reckless / Aggressive Driving",
    "V-5": "Speeding", "V-6": "Tailgating", "V-7": "Failure to Maintain Lane",
    "V-8": "Illegal U-Turn / Turn", "V-9": "Blocking Intersection",
    "V-10": "Unsafe Merge / Entry", "V-11": "Driving Wrong Way",
    "R-1": "Pedestrian-Involved", "R-2": "Bicycle-Involved",
    "R-3": "Micromobility-Involved", "R-4": "Animal-Involved",
    "I-1": "Obstruction / Road Hazard", "I-2": "Work-Zone Incident",
    "I-3": "General Incident",
}

# nuPlan scenario type -> taxonomy leaf, built by inverting LEAF_PRECURSORS.
# A handful of types are genuinely double-mapped in the source doc's prose
# (e.g. `starting_protected_cross_turn` is C-2's example but also appears
# grouped under C-5's "seven precursors" — the doc's own count for C-5 is 7
# and lists it there too); LEAF_PRECURSORS above follows the doc's per-leaf
# authoritative lists, and SCENARIO_TYPE_TO_LEAF takes the first assignment
# found in leaf order, so a duplicate in a later leaf is silently shadowed.
# None of the 20 scenario types in stitch_jobs.tsv hit this ambiguity.
SCENARIO_TYPE_TO_LEAF: dict[str, str] = {}
for _leaf, _types in LEAF_PRECURSORS.items():
    for _t in _types:
        SCENARIO_TYPE_TO_LEAF.setdefault(_t, _leaf)


def leaf_for_scenario_type(scenario_type: str) -> str | None:
    """The taxonomy leaf a nuPlan scenario type belongs to, or None if unmapped."""
    return SCENARIO_TYPE_TO_LEAF.get(scenario_type)


@dataclass
class ScenarioMeta:
    token: str
    scenario_types: tuple[str, ...]        # nuPlan scenario labels; a token can mine several
    taxonomy_leaves: tuple[str, ...]       # leaves for the mapped types, in scenario_types order, deduped
    taxonomy_leaf_names: tuple[str, ...]   # LEAF_NAMES[leaf] for each of taxonomy_leaves
    has_inserted_actors: bool              # True once synthetic/inserted actors exist in the bundle

    @property
    def scenario_type(self) -> str:
        """The first/primary type, for call sites that only want one."""
        return self.scenario_types[0]

    @property
    def taxonomy_leaf(self) -> str | None:
        """The first mapped leaf, or None if no type in scenario_types mapped."""
        return self.taxonomy_leaves[0] if self.taxonomy_leaves else None

    @property
    def taxonomy_leaf_name(self) -> str | None:
        return self.taxonomy_leaf_names[0] if self.taxonomy_leaf_names else None

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "scenario_types": list(self.scenario_types),
            "taxonomy_leaves": list(self.taxonomy_leaves),
            "taxonomy_leaf_names": list(self.taxonomy_leaf_names),
            "has_inserted_actors": self.has_inserted_actors,
        }


def load_scenario_meta(jobs_tsv: Path) -> dict[str, ScenarioMeta]:
    """token -> ScenarioMeta, from a `stitch_jobs.tsv`-shaped file
    (scenario_type, token, t0_us, t1_us, log_name).

    ``scenario_type`` may be several nuPlan types joined with `;` (a token can
    mine multiple types over its window — `scenes_500.tsv`'s source data
    already models this, comma-joined there); each is mapped to a leaf
    independently and the results kept in order, deduped, so a token carries
    every leaf it actually touches rather than only the first.

    `has_inserted_actors` is always False today: no published scenario has a
    synthetic/inserted actor yet. When that changes, the inserting tool should
    set it explicitly rather than this loader guessing from bundle contents.
    """
    out: dict[str, ScenarioMeta] = {}
    for line in jobs_tsv.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 2:
            continue
        types = tuple(t.strip() for t in f[0].split(";") if t.strip())
        token = f[1].strip()
        leaves: list[str] = []
        names: list[str] = []
        for t in types:
            leaf = leaf_for_scenario_type(t)
            if leaf and leaf not in leaves:
                leaves.append(leaf)
                names.append(LEAF_NAMES[leaf])
        out[token] = ScenarioMeta(
            token=token,
            scenario_types=types,
            taxonomy_leaves=tuple(leaves),
            taxonomy_leaf_names=tuple(names),
            has_inserted_actors=False,
        )
    return out


def main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, default=Path(__file__).with_name("stitch_jobs.tsv"))
    args = ap.parse_args()

    meta = load_scenario_meta(args.jobs)
    unmapped = [m.token for m in meta.values() if not m.taxonomy_leaves]
    for m in meta.values():
        leaves = ", ".join(f"{l} {n}" for l, n in zip(m.taxonomy_leaves, m.taxonomy_leaf_names)) or "UNMAPPED"
        types = ";".join(m.scenario_types)
        print(f"{m.token}  {types:60s} {leaves}")
    print(f"\n{len(meta)} scenario(s), {len(unmapped)} unmapped", file=sys.stderr)
    return 1 if unmapped else 0


if __name__ == "__main__":
    raise SystemExit(main())
