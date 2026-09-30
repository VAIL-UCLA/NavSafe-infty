# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The leaf contracts, the miner registry, and the mined-recipe route.

Both routes to a scenario end at a recipe. These tests hold that boundary:
a mined leaf must be able to freeze one (and must record why it was chosen),
a constructed leaf must not be able to claim it was mined, and every leaf's
declared miner / predicates / template must actually resolve.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.recipe.schema import (
    EgoSpec,
    Frames,
    HostSpec,
    Recipe,
    RecipeError,
)
from navsafe.benchmark.leaves import (
    LEAVES_DIR,
    LeafError,
    LeafManifest,
    available,
    load_all,
    load_leaf,
)
from navsafe.benchmark.mining.leaf_miners import Scenario, available_miners, get_miner


def _mined_recipe(**over) -> Recipe:
    kwargs = dict(
        recipe_id="V-10/unsafe_merge/syn/001",
        leaf="V-10",
        scenario="unsafe_merge",
        provenance="mined",
        host=HostSpec.from_dict({"scene": "syn", "world_version": "test@0"}),
        frames=Frames.from_dict({"T": 40, "dt_s": 0.1, "after_frame": 8}),
        ego=EgoSpec.from_dict({"replay_frames": 8}),
        selection={"miner": "roadblock_merge", "evidence": {"n_hits": 1}},
    )
    kwargs.update(over)
    return Recipe(**kwargs)


class TestLeafManifests:
    def test_every_leaf_declares_a_usable_contract(self):
        mans = load_all()
        assert len(mans) == 9, f"expected the nine constructed-or-mined leaves, got {sorted(mans)}"
        for leaf, man in mans.items():
            assert man.leaf == leaf
            assert man.metrics, f"{leaf}: no metrics declared — nothing would judge it"
            assert man.checklist_path().is_file(), f"{leaf}: checklist {man.checklist} missing"

    def test_mined_leaves_name_a_registered_miner(self):
        for leaf, man in load_all().items():
            if not man.is_mined:
                continue
            name = man.mine.get("miner")
            assert name, f"{leaf}: tier {man.tier} but no mine.miner"
            get_miner(name)  # raises if unregistered

    def test_declared_predicates_exist(self):
        from navsafe.benchmark.editing.qualify import PREDICATES

        for leaf, man in load_all().items():
            for pred in list(man.qualify) + list(man.qualify_info):
                assert pred in PREDICATES, f"{leaf} names unknown predicate {pred!r}"

    def test_inserting_leaves_declare_a_cast(self):
        # The per-host spec files are gone; a leaf that inserts says so in its
        # own `insert.cast`, which is what `navsafe bake` expands.
        for leaf, man in load_all().items():
            if not man.inserts:
                continue
            assert man.rule.cast, f"{leaf}: tier {man.tier} inserts but declares no cast"

    def test_a_mined_leaf_may_not_declare_a_cast(self):
        # The two are the same statement made twice, and they used to be able
        # to disagree: a leaf tiered `mined` with actors in its cast built a
        # scenario while claiming to be judged on the road alone.
        with pytest.raises(LeafError, match="is mined"):
            LeafManifest.from_dict({
                "leaf": "X-1", "name": "x", "scenario": "x", "tier": "mined",
                "insert": {"cast": [{"slot": "a", "assets": {"keys": ["k"]},
                                     "authored": {"template": "static"}}]},
            })

    def test_the_conditional_insert_tier_is_gone(self):
        # `mined+insert` made a leaf's DEFINING event conditional on the clip,
        # which is how C-7 came to have a frozen recipe with no oncoming car.
        with pytest.raises(LeafError, match="tier must be one of"):
            LeafManifest.from_dict({
                "leaf": "X-1", "name": "x", "scenario": "x", "tier": "mined+insert",
                "insert": {"cast": []},
            })

    def test_unknown_leaf_lists_what_exists(self):
        with pytest.raises(LeafError, match="Declared leaves"):
            load_leaf("Z-99")


class TestMinerRegistry:
    def test_the_declared_miners_are_registered(self):
        assert {"roadblock_merge", "tag", "two_way"} <= set(available_miners())

    def test_unknown_miner_names_the_known_ones(self):
        with pytest.raises(KeyError, match="declared miners"):
            get_miner("nope")

    def test_tag_miner_flags_a_stand_in_tag(self):
        rows = [Scenario("tok", "log", 0, 20_000_000, types="starting_unprotected_cross_turn")]
        got = get_miner("tag")(rows, types=["starting_u_turn", "starting_unprotected_cross_turn"])
        assert got[0].qualifies
        assert got[0].evidence["is_primary_tag"] is False
        assert "STAND-IN" in got[0].note  # never silently pass a neighbour off as the leaf

    def test_tag_miner_needs_types(self):
        with pytest.raises(ValueError, match="types"):
            get_miner("tag")([], types=[])

    def test_window_of_maps_time_to_the_5s_tiles(self):
        from navsafe.benchmark.mining.leaf_miners import window_of

        t0 = 1_000_000_000
        assert window_of(t0, t0) == "s1"
        assert window_of(t0 + 6_000_000, t0) == "s2"
        assert window_of(t0 + 19_000_000, t0) == "s4"


class TestMinedRecipe:
    def test_a_mined_recipe_needs_no_actors(self):
        _mined_recipe().validate()

    def test_a_mined_recipe_must_record_its_selection(self):
        with pytest.raises(RecipeError, match="must record `selection`"):
            _mined_recipe(selection={}).validate()

    def test_a_mined_recipe_may_not_carry_actors(self):
        from navsafe.benchmark.editing.recipe.schema import ActorRecipe, AssetRef

        actor = ActorRecipe(
            name="x", op="insert", asset=AssetRef(dims=[4.0, 2.0, 1.5]),
            spawn={"position": [0.0, 0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
            policy={"kind": "static"},
        )
        with pytest.raises(RecipeError, match="provenance 'mined' but the recipe carries actors"):
            _mined_recipe(actors={"x": actor}).validate()

    def test_a_constructed_recipe_still_needs_actors(self):
        with pytest.raises(RecipeError, match="constructed recipe with no actors"):
            _mined_recipe(provenance="constructed", selection={}).validate()

    def test_unknown_provenance_is_refused(self):
        with pytest.raises(RecipeError, match="provenance must be one of"):
            _mined_recipe(provenance="invented").validate()

    def test_selection_round_trips_through_yaml(self):
        import yaml

        original = _mined_recipe()
        back = Recipe.from_dict(yaml.safe_load(original.to_yaml()))
        assert back.provenance == "mined"
        assert back.selection["miner"] == "roadblock_merge"
        assert back.actors == {}
