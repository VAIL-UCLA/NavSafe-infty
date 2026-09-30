# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Recipe round-trip, integrity and host-mismatch tests.

These are the acceptance tests the scene-editing plan puts first, because every
later stage is defined against this file format:

* freeze, replay twice, assert identical traces;
* corrupt a state array, assert it fails loudly;
* feed a recipe whose ``T`` disagrees with the host, assert it refuses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from navsafe.benchmark.editing.recipe.schema import spec_digest
from navsafe.benchmark.editing.recipe import (
    ActorRecipe,
    AssetRef,
    EgoSpec,
    Frames,
    HostSpec,
    Recipe,
    RecipeError,
    edits_from_recipe,
    edits_from_recipe_file,
    freeze_recipe,
    load_recipe,
)
from navsafe.benchmark.editing.recipe.freeze import stamp_checksums
from navsafe.scenario.edits import apply_scenario_edits

T = 44
DT = 0.1
AFTER_FRAME = 8


def _host_sd(T_frames: int = T, dt_s: float = DT, ego_z: float = 55.0) -> dict:
    """Minimal host: a straight-line ego on its own logged route."""
    pos = np.zeros((T_frames, 3), np.float64)
    pos[:, 0] = np.arange(T_frames) * 2.0
    pos[:, 2] = ego_z
    ts = (np.arange(T_frames) * dt_s * 1e6).astype(np.float64)
    return {
        "metadata": {"sdc_id": "ego", "ts": ts, "coordinate": "local_frame0"},
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": pos,
                    "heading": np.zeros(T_frames),
                    "velocity": np.zeros((T_frames, 2)),
                    "valid": np.ones(T_frames, bool),
                },
                "metadata": {"object_id": "ego"},
            },
            "host_car": {
                "type": "VEHICLE",
                "state": {
                    "position": np.tile([90.0, 3.5, 55.0], (T_frames, 1)),
                    "length": np.full(T_frames, 4.6, np.float32),
                    "width": np.full(T_frames, 1.9, np.float32),
                    "height": np.full(T_frames, 1.6, np.float32),
                    "heading": np.zeros(T_frames),
                    "velocity": np.zeros((T_frames, 2)),
                    "valid": np.ones(T_frames, bool),
                },
                "metadata": {"object_id": "host_car"},
            },
            "spare_car": {
                "type": "VEHICLE",
                "state": {
                    "position": np.tile([120.0, -3.5, 55.0], (T_frames, 1)),
                    "heading": np.zeros(T_frames),
                    "velocity": np.zeros((T_frames, 2)),
                    "valid": np.ones(T_frames, bool),
                },
                "metadata": {"object_id": "spare_car"},
            },
        },
    }


def _oncoming_spawn(speed: float = 8.0) -> dict:
    """Where a head-on vehicle starts (C-10's shape)."""
    return {"position": [52.0, 0.11, -1.72], "heading": float(np.pi),
            "velocity": [-speed, 0.0], "length": 3.96, "width": 1.65}


def _oncoming_policy(speed: float = 8.0) -> dict:
    """How it then drives: IDM down the opposing lane, blind to the ego.

    The path is stored RESOLVED. Naming 'opposing_lane_chain' would make the
    recipe depend on re-running a map query whose answer may drift.
    """
    return {"kind": "idm", "v0": speed, "blind_to_ego": True,
            "path_polyline": [[52.0 - 2.0 * k, 0.11] for k in range(T)]}


def _recipe(**overrides) -> Recipe:
    actor = ActorRecipe(
        name="oncoming_vehicle",
        op="relocate",
        source_track_id="host_car",
        keep_appearance=True,
        track_type="VEHICLE",
        asset=AssetRef(registry_key="car_sedan_03", uid="946ec7c0", sha256="9f2c", dims=[3.96, 1.65, 1.45]),
        authored={"template": "dynamic", "reference": "ego_route", "arc": 52.0, "speed": -8.0},
        spawn=_oncoming_spawn(),
        policy=_oncoming_policy(),
    )
    recipe = Recipe(
        recipe_id="C-10/wrong_way_approach/test_host/001",
        leaf="C-10",
        scenario="wrong_way_approach",
        host=HostSpec(scene="test_host", world_version="car2sim_6cam_static@abc123"),
        frames=Frames(
            T=T,
            dt_s=DT,
            after_frame=AFTER_FRAME,
            timestamps_us=[int(round(k * DT * 1e6)) for k in range(T)],
        ),
        ego=EgoSpec(replay_frames=AFTER_FRAME, cam_height="waymo", z_to_ground=0.0),
        route_polyline=[[float(k * 2.0), 0.0] for k in range(T)],
        actors={"oncoming_vehicle": actor},
        pair={"e0_removes": ["oncoming_vehicle"], "pair_id": "c10_001"},
        review={"status": "pending", "checklist": "leaves/C-10.md"},
    )
    for key, value in overrides.items():
        setattr(recipe, key, value)
    return recipe


_PLY = b"ply"
_PLY_SHA = hashlib.sha256(_PLY).hexdigest()


def _asset_bank(tmp_path, recipe: Recipe):
    """A bank holding exactly the assets ``recipe`` pins, and say so.

    Replay has two gates on the asset, and a fixture has to satisfy both.
    ``NAVSAFE_ASSET_BANK`` must be set at all -- unset used to fall through to
    the frozen path, which resolves only on the machine that authored the
    recipe -- and the bank's copy must hash to what the recipe recorded. So the
    bytes are written first and the recipe is told what was written, rather
    than pinning a stub hash and hoping nothing re-reads the file.
    """
    for actor in recipe.actors.values():
        if not actor.nurec_asset_id:
            continue
        (tmp_path / Path(actor.nurec_asset_id).name).write_bytes(_PLY)
        if actor.asset is not None:
            actor.asset.sha256 = _PLY_SHA
    return tmp_path


def _edits(recipe: Recipe, **kwargs) -> list:
    """Stamp checksums (freeze's job) and build the edit spec, without a file."""
    stamp_checksums(recipe)
    return edits_from_recipe(recipe, **kwargs)


def _apply(edits: list, sd: dict | None = None) -> dict:
    return apply_scenario_edits(sd if sd is not None else _host_sd(), edits)


def _trace(sd: dict) -> dict:
    """Every track's per-frame state, as bytes — the thing that must match."""
    out = {}
    for tid, track in sorted(sd["tracks"].items()):
        state = track.get("state", {})
        out[tid] = {
            key: np.asarray(state[key]).tobytes()
            for key in ("position", "heading", "velocity", "valid")
            if key in state
        }
    return out


# ---------------------------------------------------------------------------
# 1. Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_freeze_load_yields_the_same_file(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml", frozen_at="2026-08-09")
        reloaded = load_recipe(path)
        again = freeze_recipe(reloaded, tmp_path / "r2.yaml", frozen_at="2026-08-09")
        assert path.read_text() == again.read_text()

    def test_replay_twice_gives_identical_traces(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        _, edits_a = edits_from_recipe_file(path)
        _, edits_b = edits_from_recipe_file(path)
        assert _trace(_apply(edits_a)) == _trace(_apply(edits_b))

    def test_the_spec_survives_the_file(self, tmp_path):
        # What has to round-trip is the SPECIFICATION, not a trajectory: the
        # spawn, the controller and its resolved geometry.
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        back = load_recipe(path)
        actor = back.actors["oncoming_vehicle"]
        np.testing.assert_allclose(actor.spawn["position"], _oncoming_spawn()["position"],
                                   atol=1e-6)
        assert actor.policy["kind"] == "idm"
        assert actor.policy["blind_to_ego"] is True
        assert len(actor.policy["path_polyline"]) == T

    def test_the_re_tasked_host_actor_is_taken_off_the_log(self, tmp_path):
        # A relocate re-tasks a logged actor: it becomes the recipe's, driven
        # by a policy. Leaving its logged track behind would put two copies of
        # the same car in the scene.
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        _, edits = edits_from_recipe_file(path)
        sd = _apply(edits)
        assert "host_car" not in sd["tracks"]
        track = sd["tracks"]["navsafe_oncoming_vehicle"]
        assert track["metadata"]["replaced_track_id"] == "host_car"
        assert track["metadata"]["navsafe_policy"] == "idm"


# ---------------------------------------------------------------------------
# 2. Integrity — a corrupted recipe must fail loudly
# ---------------------------------------------------------------------------


class TestIntegrity:
    @staticmethod
    def _corrupt(path, mutate):
        raw = yaml.safe_load(path.read_text())
        mutate(raw["actors"]["oncoming_vehicle"])
        path.write_text(yaml.safe_dump(raw, sort_keys=False))

    def test_an_edited_spawn_fails_to_load(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        self._corrupt(path, lambda a: a["spawn"]["position"].__setitem__(1, 9.0))
        with pytest.raises(RecipeError, match="sha256 mismatch"):
            load_recipe(path)

    def test_an_edited_policy_fails_to_load(self, tmp_path):
        # Retuning IDM changes the scenario as surely as moving the actor did.
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        self._corrupt(path, lambda a: a["policy"].__setitem__("v0", 30.0))
        with pytest.raises(RecipeError, match="sha256 mismatch"):
            load_recipe(path)

    def test_an_edited_path_fails_to_load(self, tmp_path):
        # The resolved lane is part of the spec: re-pointing it silently sends
        # the actor down a different road.
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        self._corrupt(path, lambda a: a["policy"]["path_polyline"][0].__setitem__(0, -999.0))
        with pytest.raises(RecipeError, match="sha256 mismatch"):
            load_recipe(path)

    def test_a_swapped_asset_fails_to_load(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        self._corrupt(path, lambda a: a["asset"].__setitem__("sha256", "deadbeef"))
        with pytest.raises(RecipeError, match="sha256 mismatch"):
            load_recipe(path)

    def test_a_stripped_checksum_is_refused(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        self._corrupt(path, lambda a: a.pop("sha256"))
        with pytest.raises(RecipeError, match="no sha256 recorded"):
            load_recipe(path)

    def test_apply_rechecks_a_hand_built_spec(self):
        # The last gate before the sim: a spec assembled by hand, bypassing
        # replay.py, still cannot slip past the checksums.
        edits = _edits(_recipe())
        edits[0]["actors"][0]["policy"]["v0"] = 99.0
        with pytest.raises(RecipeError, match="sha256 mismatch"):
            _apply(edits)

    def test_freeze_refuses_to_overwrite(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        with pytest.raises(RecipeError, match="immutable"):
            freeze_recipe(_recipe(), path)


# ---------------------------------------------------------------------------
# 3. Host mismatch — refuse rather than replay at the wrong speed
# ---------------------------------------------------------------------------


class TestHostMismatch:
    def test_frame_count_mismatch_is_refused(self):
        edits = _edits(_recipe())
        with pytest.raises(ValueError, match="baked for T=44 frames but this host has 40"):
            _apply(edits, _host_sd(T_frames=40))

    def test_frame_rate_mismatch_is_refused(self):
        edits = _edits(_recipe())
        with pytest.raises(ValueError, match="wrong speed"):
            _apply(edits, _host_sd(dt_s=0.05))

    def test_timestamp_mismatch_is_refused(self):
        edits = _edits(_recipe())
        sd = _host_sd()
        sd["metadata"]["ts"] = np.asarray(sd["metadata"]["ts"]) + 5.0e6  # a different window
        with pytest.raises(ValueError, match="different scenario window"):
            _apply(edits, sd)

    def test_missing_source_track_is_refused(self):
        edits = _edits(_recipe())
        sd = _host_sd()
        del sd["tracks"]["host_car"]
        with pytest.raises(ValueError, match="source track 'host_car' not found"):
            _apply(edits, sd)


# ---------------------------------------------------------------------------
# 4. Schema rules
# ---------------------------------------------------------------------------


class TestSchema:
    def test_absolute_utm_frame_is_refused(self):
        with pytest.raises(RecipeError, match="ego_frame0"):
            _recipe(frame="utm").validate()

    def test_e0_removes_must_name_a_real_actor(self):
        with pytest.raises(RecipeError, match="not an actor"):
            _recipe(pair={"e0_removes": ["ghost"]}).validate()

    def test_after_frame_must_lie_inside_the_episode(self):
        with pytest.raises(RecipeError, match="after_frame"):
            Frames.from_dict({"T": 44, "dt_s": 0.1, "after_frame": 44})

    def test_unknown_template_is_refused(self):
        recipe = _recipe()
        recipe.actors["oncoming_vehicle"].authored["template"] = "swerve"
        with pytest.raises(RecipeError, match="authored.template"):
            recipe.validate()

    def test_an_actor_without_a_policy_is_refused(self):
        # The one thing a reactive recipe cannot leave unsaid: an actor with a
        # spawn and no controller would stand at the kerb all episode and look
        # like a scenario that merely did not fire.
        recipe = _recipe()
        recipe.actors["oncoming_vehicle"].policy = {}
        with pytest.raises(RecipeError, match="policy.kind"):
            recipe.validate()

    def test_an_actor_without_a_spawn_is_refused(self):
        recipe = _recipe()
        recipe.actors["oncoming_vehicle"].spawn = {}
        with pytest.raises(RecipeError, match="spawn needs a `position`"):
            recipe.validate()

    def test_inserted_asset_needs_dims(self):
        recipe = _recipe()
        actor = recipe.actors["oncoming_vehicle"]
        actor.op = "insert"
        actor.keep_appearance = False
        actor.source_track_id = ""
        actor.asset.dims = None
        with pytest.raises(RecipeError, match="needs asset.dims|asset.dims"):
            recipe.validate()

    def test_keep_appearance_only_applies_to_relocate(self):
        recipe = _recipe()
        recipe.actors["oncoming_vehicle"].op = "insert"
        with pytest.raises(RecipeError, match="keep_appearance"):
            recipe.validate()


# ---------------------------------------------------------------------------
# 5. The other three ops, static expansion, and the e0 pair
# ---------------------------------------------------------------------------


class TestOpsAndVariants:
    def test_e_zero_leaves_the_host_untouched(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        _, edits = edits_from_recipe_file(path, variant="e_zero")
        assert edits == []  # the only actor defines the leaf, so e0 edits nothing
        # …and the host's own logged actor is therefore still where the log put it.
        sd = _apply(edits)
        assert float(sd["tracks"]["host_car"]["state"]["position"][0][0]) == pytest.approx(90.0)

    def test_e_zero_keeps_actors_not_named_in_the_pair(self):
        recipe = _recipe()
        recipe.actors["constrictor"] = ActorRecipe(
            name="constrictor",
            op="insert",
            track_type="TRAFFIC_CONE",
            asset=AssetRef(registry_key="jersey_barrier", dims=[3.0, 0.6, 0.8]),
            authored={"template": "static", "reference": "ego_route", "arc": 40.0},
            spawn={"position": [40.0, -2.0, -1.7], "heading": 0.0, "velocity": [0.0, 0.0]},
            policy={"kind": "static"},
        )
        edits = _edits(recipe, variant="e_zero")
        names = [a["name"] for a in edits[0]["actors"]]
        assert names == ["constrictor"]

    def test_a_static_actor_declares_a_pose_for_every_frame(self, tmp_path, monkeypatch):
        # A thing that never moves is still a policy ("static"), not a special
        # case in the file format. Its declaration spans the episode so
        # collision and the renderer see it throughout.
        recipe = _recipe()
        recipe.actors["sign"] = ActorRecipe(
            name="sign",
            op="insert",
            track_type="TRAFFIC_CONE",
            asset=AssetRef(registry_key="one_way_sign", dims=[0.15, 0.66, 2.0]),
            nurec_asset_id="/assets/one_way_3dgs.ply",
            authored={"template": "static", "reference": "ego_route", "arc": 30.0},
            spawn={"position": [30.0, -4.0, -1.7], "heading": 1.57, "velocity": [0.0, 0.0]},
            policy={"kind": "static"},
        )
        import navsafe.benchmark.config as cfg
        monkeypatch.setattr(cfg, "ASSET_BANK", _asset_bank(tmp_path, recipe))
        sd = _apply(_edits(recipe))
        state = sd["tracks"]["navsafe_sign"]["state"]
        assert np.asarray(state["position"]).shape == (T, 3)
        assert np.all(np.asarray(state["position"])[:, 0] == pytest.approx(30.0))
        # Rebased, because the bank is set: the frozen "/assets/..." is the
        # authoring machine's and replay no longer honours it.
        assert (sd["tracks"]["navsafe_sign"]["metadata"]["nurec_asset_id"]
                == str(tmp_path / "one_way_3dgs.ply"))

    def test_replace_deletes_the_source_and_carries_the_asset(self, tmp_path, monkeypatch):
        import navsafe.benchmark.config as cfg
        recipe = _recipe()
        actor = recipe.actors["oncoming_vehicle"]
        actor.op = "replace"
        actor.keep_appearance = False
        actor.nurec_asset_id = "/assets/sedan_uv_3dgs.ply"
        monkeypatch.setattr(cfg, "ASSET_BANK", _asset_bank(tmp_path, recipe))
        sd = _apply(_edits(recipe))
        assert "host_car" not in sd["tracks"]
        track = sd["tracks"]["navsafe_oncoming_vehicle"]
        assert float(track["state"]["length"][0]) == pytest.approx(3.96)
        assert track["metadata"]["nurec_asset_id"] == str(tmp_path / "sedan_uv_3dgs.ply")
        assert track["metadata"]["injected_obstacle"] is True

    def test_remove_drops_a_non_ego_agent(self):
        recipe = _recipe()
        recipe.actors["clutter"] = ActorRecipe(
            name="clutter", op="remove", source_track_id="spare_car"
        )
        sd = _apply(_edits(recipe))
        assert "spare_car" not in sd["tracks"]
        assert "ego" in sd["tracks"]

    def test_remove_refuses_the_ego(self):
        recipe = _recipe()
        recipe.actors["boom"] = ActorRecipe(name="boom", op="remove", source_track_id="ego")
        with pytest.raises(ValueError, match="refuses to delete the ego"):
            _apply(_edits(recipe))

    def test_the_declaration_spans_the_whole_episode(self):
        # Under the baked format an actor could be marked invalid for part of
        # the episode. A reactive actor exists from spawn to the end and its
        # WHEREABOUTS are the manager's business, so the declaration is valid
        # throughout — anything else would hide it from collision the moment
        # the policy did something unexpected.
        sd = _apply(_edits(_recipe()))
        valid = np.asarray(sd["tracks"]["navsafe_oncoming_vehicle"]["state"]["valid"])
        assert valid.shape == (T,) and valid.all()

    def test_idempotent_across_resets(self, tmp_path):
        path = freeze_recipe(_recipe(), tmp_path / "r.yaml")
        _, edits = edits_from_recipe_file(path)
        sd = _apply(edits)
        before = _trace(sd)
        sd = apply_scenario_edits(sd, edits)  # env reset with the same dict
        assert _trace(sd) == before


class TestAssetBankRebase:
    """A published recipe names the directory it was frozen from, which exists
    only on the machine that froze it. ``NAVSAFE_ASSET_BANK`` rebases it onto
    the reader's own bank without touching the recipe's identity."""

    @staticmethod
    def _inserting_recipe(asset_id: str) -> Recipe:
        recipe = _recipe()
        actor = recipe.actors["oncoming_vehicle"]
        actor.op = "insert"
        actor.source_track_id = ""
        actor.keep_appearance = False
        actor.nurec_asset_id = asset_id
        return recipe

    def test_unset_is_refused_rather_than_falling_back(self, monkeypatch):
        """The frozen path is the authoring machine's, so honouring it is only
        ever right on that machine -- which is ours, where it hides the
        mistake instead of reporting it."""
        monkeypatch.delenv("NAVSAFE_ASSET_BANK", raising=False)
        import navsafe.benchmark.config as cfg
        monkeypatch.setattr(cfg, "ASSET_BANK", None)
        with pytest.raises(RecipeError, match="NAVSAFE_ASSET_BANK is unset"):
            _edits(self._inserting_recipe("/frozen/elsewhere/car_1.ply"))

    def test_rebased_onto_the_local_bank(self, tmp_path, monkeypatch):
        import navsafe.benchmark.config as cfg
        recipe = self._inserting_recipe("/frozen/elsewhere/car_1.ply")
        monkeypatch.setattr(cfg, "ASSET_BANK", _asset_bank(tmp_path, recipe))
        edits = _edits(recipe)
        assert edits[0]["actors"][0]["nurec_asset_id"] == str(tmp_path / "car_1.ply")

    def test_rebase_does_not_disturb_the_checksum(self, tmp_path, monkeypatch):
        """The recipe's identity is the asset's sha256, not where it sits.

        The SAME frozen path both times -- only the bank moves. Unset is no
        longer one of the two cases (it is refused), so "not rebased" is
        expressed as a bank that happens to be the directory the recipe was
        frozen from, which is what a reader who unpacked it in place has.
        """
        import navsafe.benchmark.config as cfg
        frozen = tmp_path / "frozen"; frozen.mkdir()
        other = tmp_path / "other"; other.mkdir()
        frozen_ply = str(frozen / "car_1.ply")

        here = self._inserting_recipe(frozen_ply)
        monkeypatch.setattr(cfg, "ASSET_BANK", _asset_bank(frozen, here))
        plain = _edits(here)

        moved = self._inserting_recipe(frozen_ply)
        _asset_bank(frozen, moved)
        (other / "car_1.ply").write_bytes(_PLY)
        monkeypatch.setattr(cfg, "ASSET_BANK", other)
        rebased = _edits(moved)

        assert rebased[0]["actors"][0]["nurec_asset_id"] == str(other / "car_1.ply")
        assert rebased[0]["actors"][0]["sha256"] == plain[0]["actors"][0]["sha256"]

    def test_a_bank_missing_the_asset_says_so(self, tmp_path, monkeypatch):
        import navsafe.benchmark.config as cfg
        monkeypatch.setattr(cfg, "ASSET_BANK", tmp_path)
        with pytest.raises(RecipeError, match="NAVSAFE_ASSET_BANK"):
            _edits(self._inserting_recipe("/frozen/elsewhere/car_1.ply"))


class TestGaitBankRebase:
    """A pose bank is a directory the reader downloads, so the frozen path has
    to be relocatable — and what the recipe pins has to be the bank's CONTENT,
    or a bank baked from a different motion passes as the frozen one."""

    @staticmethod
    def _bank(root, name: str, *, motion: str = "walk"):
        import hashlib
        import json
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"phases": ["ph00.ply"], "phases_sha256": ["ab" * 32],
                           "stride_m": 1.4, "motion": motion})
        (d / "bank.json").write_text(body)
        return d, hashlib.sha256(body.encode()).hexdigest()

    @pytest.fixture(autouse=True)
    def _an_asset_bank_exists(self, tmp_path, monkeypatch):
        """These tests are about the GAIT bank, but every one of their recipes
        also carries an inserted asset, and replay refuses an unset
        ``NAVSAFE_ASSET_BANK`` -- so satisfy that gate and leave the subject of
        the test to the assertions.

        The recipe's asset path is the bank's own, so the asset is NOT rebased
        and ``FROZEN_ASSET_ID_KEY`` never appears. That matters:
        ``spec_digest`` hashes ``nurec_asset_id`` (only the pose-bank path is
        excluded), so an asset rebase legitimately moves the digest and would
        mask what these tests assert about the gait bank.
        """
        import navsafe.benchmark.config as cfg
        self._bank_root = tmp_path
        (tmp_path / "car_1.ply").write_bytes(_PLY)
        monkeypatch.setattr(cfg, "ASSET_BANK", tmp_path)

    def _walking_recipe(self, bank: str, sha: str) -> Recipe:
        recipe = _recipe()
        actor = recipe.actors["oncoming_vehicle"]
        actor.op = "insert"
        actor.source_track_id = ""
        actor.keep_appearance = False
        actor.nurec_asset_id = str(self._bank_root / "car_1.ply")
        actor.asset.sha256 = _PLY_SHA
        actor.nurec_pose_bank = bank
        actor.pose_bank_sha256 = sha
        return recipe

    def test_unset_is_refused_rather_than_falling_back(self, tmp_path, monkeypatch):
        """`cfg.GAIT_BANK` used to DEFAULT to this cluster's own library, so a
        reader who set nothing was pointed at it by the config itself."""
        import navsafe.benchmark.config as cfg
        d, sha = self._bank(tmp_path, "hb_pedestrian_1")
        monkeypatch.setattr(cfg, "GAIT_BANK", None)
        with pytest.raises(RecipeError, match="NAVSAFE_GAIT_BANK is unset"):
            _edits(self._walking_recipe(str(d), sha))

    def test_rebased_onto_the_local_bank(self, tmp_path, monkeypatch):
        import navsafe.benchmark.config as cfg
        _, sha = self._bank(tmp_path, "hb_pedestrian_1")
        monkeypatch.setattr(cfg, "GAIT_BANK", tmp_path)
        edits = _edits(self._walking_recipe(
            "/frozen/elsewhere/hb_pedestrian_1", sha))
        assert edits[0]["actors"][0]["nurec_pose_bank"] == str(tmp_path / "hb_pedestrian_1")

    def test_a_relocated_bank_still_passes_the_last_gate(self, tmp_path, monkeypatch):
        """The regression this whole design exists for.

        The digest is checked twice — here, and again in
        ``spawn_reactive_actor`` before the sim. Hashing the bank's PATH made a
        relocated bank fail that second gate with "the spawn, the policy or the
        asset was edited after freezing", which is both wrong and unfixable by
        the reader. The path is out of the payload; the content hash is in it.
        """
        import navsafe.benchmark.config as cfg
        _, sha = self._bank(tmp_path, "hb_pedestrian_1")
        monkeypatch.setattr(cfg, "GAIT_BANK", tmp_path)
        spec = _edits(self._walking_recipe(
            "/frozen/elsewhere/hb_pedestrian_1", sha))[0]["actors"][0]
        assert spec["nurec_pose_bank"] != "/frozen/elsewhere/hb_pedestrian_1"
        assert spec_digest(spec) == spec["sha256"]

    def test_a_bank_with_other_contents_is_refused(self, tmp_path, monkeypatch):
        """Same name, same place, baked from a different motion."""
        import navsafe.benchmark.config as cfg
        _, walk_sha = self._bank(tmp_path, "hb_pedestrian_1", motion="walk")
        monkeypatch.setattr(cfg, "GAIT_BANK", tmp_path)
        self._bank(tmp_path, "hb_pedestrian_1", motion="run")   # swapped underneath
        with pytest.raises(RecipeError, match="frozen with"):
            _edits(self._walking_recipe(
                "/frozen/elsewhere/hb_pedestrian_1", walk_sha))

    def test_a_missing_bank_says_where_to_get_one(self, tmp_path, monkeypatch):
        import navsafe.benchmark.config as cfg
        monkeypatch.setattr(cfg, "GAIT_BANK", tmp_path)
        with pytest.raises(RecipeError, match="NAVSAFE_GAIT_BANK"):
            _edits(self._walking_recipe("/frozen/elsewhere/hb_pedestrian_1", "ab" * 32))

    def test_an_actor_without_a_bank_is_untouched(self, tmp_path, monkeypatch):
        import navsafe.benchmark.config as cfg
        monkeypatch.setattr(cfg, "GAIT_BANK", tmp_path)
        edits = _edits(self._walking_recipe("", ""))
        assert "nurec_pose_bank" not in edits[0]["actors"][0]
