# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The asset layer: PLY I/O, the registry, composition, and acquisition plans."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from navsafe.benchmark.editing.assets import (
    AssetError,
    AssetRegistry,
    ComposeError,
    PlyError,
    acquire_asset,
    bounding_dims,
    compose_assets,
    compose_from_registry,
    concat_gaussians,
    plan_acquisition,
    ply_sha256,
    read_3dgs_ply,
    transform_gaussians,
    write_3dgs_ply,
)
from navsafe.benchmark.editing.assets.registry import DEFAULT_REGISTRY
from navsafe.benchmark.editing.author import AuthoringError, author_recipe


def _gaussians(n: int = 64, *, extent=(2.0, 1.0, 0.5), sh: int = 0, seed: int = 0) -> dict:
    """A synthetic 3DGS cloud filling a box, y-up like the file convention."""
    rng = np.random.default_rng(seed)
    xyz = (rng.random((n, 3)) - 0.5).astype(np.float32) * np.asarray(extent, np.float32)
    # Pin the corners so the extent is exact and grounding is testable.
    xyz[0] = [-extent[0] / 2, -extent[1] / 2, -extent[2] / 2]
    xyz[1] = [extent[0] / 2, extent[1] / 2, extent[2] / 2]
    n_rest = 3 * ((sh + 1) ** 2 - 1)
    return {
        "xyz": xyz,
        "normals": np.zeros((n, 3), np.float32),
        "f_dc": rng.random((n, 3)).astype(np.float32),
        "f_rest": (rng.random((n, n_rest)).astype(np.float32) if n_rest else np.zeros((n, 0), np.float32)),
        "opacity": np.full(n, 2.0, np.float32),
        "scale": np.full((n, 3), -3.0, np.float32),
        "rot": np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
    }


class TestPlyIO:
    def test_round_trip_is_exact(self, tmp_path):
        original = _gaussians(sh=1)
        path = write_3dgs_ply(original, tmp_path / "a.ply")
        back = read_3dgs_ply(path)
        for key, value in original.items():
            np.testing.assert_array_equal(back[key], value, err_msg=key)

    def test_bounding_dims_reads_the_file_frame(self):
        # y-up, +x forward: length is x, width is z, height is y.
        dims = bounding_dims(_gaussians(extent=(4.0, 1.6, 1.8)))
        assert dims["length"] == pytest.approx(4.0)
        assert dims["height"] == pytest.approx(1.6)
        assert dims["width"] == pytest.approx(1.8)

    def test_translate_and_scale(self):
        moved = transform_gaussians(_gaussians(extent=(2.0, 1.0, 1.0)), translate=(1.0, 2.0, 3.0), scale=2.0)
        dims = bounding_dims(moved)
        assert dims["length"] == pytest.approx(4.0)
        assert float(moved["xyz"][:, 1].mean()) == pytest.approx(2.0, abs=0.3)
        # ``scale`` holds log(sigma), so a 2x resize is +log(2), not x2.
        assert float(moved["scale"][0, 0]) == pytest.approx(-3.0 + np.log(2.0), abs=1e-5)

    def test_yaw_is_about_the_vertical_y_axis(self):
        turned = transform_gaussians(_gaussians(extent=(4.0, 1.0, 1.0)), yaw_deg=90.0)
        dims = bounding_dims(turned)
        # A 4 m length swung 90 deg about +y becomes 4 m of WIDTH.
        assert dims["width"] == pytest.approx(4.0, abs=1e-4)
        assert dims["length"] == pytest.approx(1.0, abs=1e-4)
        assert dims["height"] == pytest.approx(1.0, abs=1e-4)

    def test_rotating_real_sh_bands_is_refused(self):
        with pytest.raises(PlyError, match="view-dependent SH"):
            transform_gaussians(_gaussians(sh=2), yaw_deg=30.0)

    def test_sh_can_be_dropped_deliberately(self):
        out = transform_gaussians(_gaussians(sh=2), yaw_deg=30.0, sh_policy="drop")
        assert float(np.abs(out["f_rest"]).max()) == 0.0

    def test_translation_never_touches_sh(self):
        original = _gaussians(sh=2)
        out = transform_gaussians(original, translate=(1.0, 0.0, 0.0))
        np.testing.assert_array_equal(out["f_rest"], original["f_rest"])

    def test_concat_pads_mismatched_sh_widths(self):
        merged = concat_gaussians([_gaussians(n=10, sh=0), _gaussians(n=5, sh=1)])
        assert merged["xyz"].shape == (15, 3)
        assert merged["f_rest"].shape == (15, 9)
        assert float(np.abs(merged["f_rest"][:10]).max()) == 0.0

    def test_sha256_is_content_addressed(self, tmp_path):
        a = write_3dgs_ply(_gaussians(seed=1), tmp_path / "a.ply")
        b = write_3dgs_ply(_gaussians(seed=1), tmp_path / "b.ply")
        c = write_3dgs_ply(_gaussians(seed=2), tmp_path / "c.ply")
        assert ply_sha256(a) == ply_sha256(b)
        assert ply_sha256(a) != ply_sha256(c)


class TestCompose:
    def test_parts_merge_and_ground(self, tmp_path):
        bike = write_3dgs_ply(_gaussians(extent=(1.8, 1.1, 0.5)), tmp_path / "bike.ply")
        rider = write_3dgs_ply(_gaussians(extent=(0.5, 1.7, 0.5)), tmp_path / "rider.ply")
        report = compose_assets(
            [{"ply": str(bike)}, {"ply": str(rider), "translate": [-0.05, 0.62, 0.0]}],
            tmp_path / "bike_rider.ply",
        )
        assert report["composed"] is True
        assert report["gaussians"] == 128
        # Grounded: the lowest gaussian sits at y = 0, so the asset is
        # base-origin like every other inserted asset.
        assert report["base_y"] == pytest.approx(0.0, abs=1e-5)
        merged = read_3dgs_ply(report["output"])
        assert bounding_dims(merged)["height"] > 1.7  # the rider sticks up above the bike

    def test_one_part_is_not_a_composition(self, tmp_path):
        with pytest.raises(ComposeError, match="at least two parts"):
            compose_assets([{"ply": "x.ply"}], tmp_path / "out.ply")

    def test_unreadable_part_names_itself(self, tmp_path):
        good = write_3dgs_ply(_gaussians(), tmp_path / "a.ply")
        with pytest.raises(ComposeError, match="part 1"):
            compose_assets([{"ply": str(good)}, {"ply": str(tmp_path / "nope.ply")}], tmp_path / "o.ply")


def _registry(tmp_path, **extra) -> AssetRegistry:
    ply = write_3dgs_ply(_gaussians(extent=(1.8, 1.1, 0.5)), tmp_path / "present.ply")
    assets = {
        "present_thing": {
            "family": "car", "source": "urbanverse", "uid": "uid-123",
            "ply": str(ply), "dims": [1.8, 0.5, 1.1], "track_type": "VEHICLE",
            "leaves": ["C-10"],
        },
        "absent_thing": {
            "family": "animal", "source": "urbanverse", "uid": "uid-456",
            "ply": str(tmp_path / "absent.ply"), "dims": [1.0, 0.3, 0.7],
            "track_type": "PEDESTRIAN", "leaves": ["R-4"],
        },
        "a_cone": {
            "family": "corridor_constrictor", "source": "procedural",
            "ply": str(tmp_path / "cone.ply"), "dims": [0.4, 0.4, 1.0],
            "track_type": "TRAFFIC_CONE", "leaves": ["C-7"],
            "procedural": {"kind": "cone", "target_height": 1.0},
        },
        "host_actor": {"family": "car", "source": "host", "dims": [4.6, 1.9, 1.6], "leaves": ["C-10"]},
        "a_pair": {
            "family": "bicycle_rider", "source": "composed",
            "ply": str(tmp_path / "pair.ply"), "dims": [1.8, 0.6, 1.75],
            "track_type": "CYCLIST", "leaves": ["R-2"],
            "compose": {"parts": [{"asset": "present_thing"},
                                  {"asset": "present_thing", "translate": [0.0, 0.6, 0.0]}]},
        },
    }
    assets.update(extra)
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"assets": assets}))
    return AssetRegistry.load(path)


class TestRegistry:
    def test_resolve_pins_content(self, tmp_path):
        resolved = _registry(tmp_path).resolve("present_thing")
        block = resolved.to_recipe_asset()
        assert block["registry_key"] == "present_thing"
        assert block["uid"] == "uid-123"
        assert len(block["sha256"]) == 64
        assert block["dims"] == [1.8, 0.5, 1.1]

    def test_declared_but_absent_says_how_to_get_it(self, tmp_path):
        with pytest.raises(AssetError, match="not on disk.*assets acquire absent_thing"):
            _registry(tmp_path).resolve("absent_thing")

    def test_host_source_has_no_file(self, tmp_path):
        with pytest.raises(AssetError, match="keeps its baked gaussians"):
            _registry(tmp_path).resolve("host_actor")

    def test_unknown_key_suggests_neighbours(self, tmp_path):
        with pytest.raises(AssetError, match="no asset 'car_nope'"):
            _registry(tmp_path).get("car_nope")

    def test_status_counts_present_against_declared(self, tmp_path):
        status = _registry(tmp_path).status()
        assert status["total"] == 5
        assert status["present"] == 2  # the one file plus the file-less host entry
        assert status["by_leaf"]["R-4"] == {"present": 0, "declared": 1}

    def test_the_shipped_registry_loads_and_covers_all_nine_leaves(self):
        registry = AssetRegistry.load(DEFAULT_REGISTRY)
        leaves = set(registry.status()["by_leaf"])
        assert leaves == {"C-7", "C-10", "I-3", "R-2", "R-3", "R-4", "V-8", "V-10", "V-11"}
        # Most of the tree is declared-but-missing on purpose; the registry
        # exists to make that gap visible, not to hide it.
        assert registry.missing()

    def test_compose_from_registry_builds_the_declared_parts(self, tmp_path):
        registry = _registry(tmp_path)
        report = compose_from_registry("a_pair", registry)
        assert report["gaussians"] == 128
        assert registry.get("a_pair").present


class TestAcquire:
    def test_procedural_is_runnable_here(self, tmp_path):
        plan = plan_acquisition(_registry(tmp_path).get("a_cone"), _registry(tmp_path))
        assert plan["runnable_here"] is True
        assert "--procedural-cone" in plan["steps"][0]

    def test_urbanverse_needs_the_network(self, tmp_path):
        plan = plan_acquisition(_registry(tmp_path).get("absent_thing"), _registry(tmp_path))
        assert plan["runnable_here"] is False
        assert "UrbanVerse SDK" in plan["note"]
        assert len(plan["steps"]) == 2  # download, then convert

    def test_host_source_needs_nothing(self, tmp_path):
        plan = plan_acquisition(_registry(tmp_path).get("host_actor"), _registry(tmp_path))
        assert plan["steps"] == []
        assert "nothing to acquire" in plan["note"]

    def test_present_asset_is_a_no_op(self, tmp_path):
        registry = _registry(tmp_path)
        assert acquire_asset("present_thing", registry)["note"] == "already on disk"

    def test_planning_does_not_execute(self, tmp_path):
        registry = _registry(tmp_path)
        acquire_asset("a_cone", registry, execute=False)
        assert not registry.get("a_cone").present  # nothing was written


class TestAuthorUsesTheRegistry:
    @staticmethod
    def _spec(registry_key: str, **actor):
        base = {
            "recipe_id": "R-4/animal_dart/synthetic/001",
            "leaf": "R-4",
            "host": {"scene": "synthetic", "world_version": "test@0"},
            "ego": {"replay_frames": 4, "z_to_ground": 1.4},
            "actors": {
                "critter": {
                    "op": "insert",
                    "asset": {"registry_key": registry_key},
                    "authored": {"template": "static", "reference": "ego_route", "arc": 20.0},
                    **actor,
                }
            },
            "pair": {"e0_removes": ["critter"], "pair_id": "x"},
        }
        return base

    @staticmethod
    def _sd(T: int = 40):
        pos = np.zeros((T, 3), np.float64)
        pos[:, 0] = np.arange(T) * 1.5
        pos[:, 2] = 55.0
        return {
            "metadata": {"sdc_id": "ego", "ts": (np.arange(T) * 0.1 * 1e6).astype(np.float64)},
            "tracks": {
                "ego": {
                    "type": "VEHICLE",
                    "state": {
                        "position": pos, "heading": np.zeros(T),
                        "velocity": np.zeros((T, 2)), "valid": np.ones(T, bool),
                    },
                    "metadata": {"object_id": "ego"},
                }
            },
            "map_features": {},
        }

    def test_a_registry_key_becomes_a_content_pin(self, tmp_path):
        recipe, diagnostics = author_recipe(
            self._sd(), self._spec("present_thing"), registry=_registry(tmp_path).path
        )
        actor = recipe.actors["critter"]
        assert actor.asset.uid == "uid-123"
        assert len(actor.asset.sha256) == 64
        assert actor.asset.dims == [1.8, 0.5, 1.1]
        # …and the PLY the server must load is carried through to the render.
        assert actor.nurec_asset_id.endswith("present.ply")
        assert actor.track_type == "VEHICLE"
        assert diagnostics["registry"].endswith("registry.yaml")

    def test_an_absent_asset_refuses_to_freeze(self, tmp_path):
        with pytest.raises(AuthoringError, match="pin a checksum of nothing"):
            author_recipe(self._sd(), self._spec("absent_thing"), registry=_registry(tmp_path).path)

    def test_a_host_asset_demands_keep_appearance(self, tmp_path):
        spec = self._spec("host_actor")
        spec["actors"]["critter"]["op"] = "relocate"
        spec["actors"]["critter"]["source_track_id"] = "ego"
        with pytest.raises(AuthoringError, match="source 'host'"):
            author_recipe(self._sd(), spec, registry=_registry(tmp_path).path)


class TestTheHarvestedLibraryIsOneDirectory:
    """A `family` + `source` pair must name one library, not two.

    `car_harvested_01` sat at a loose path under /data/nurec_assets while
    hb_car_1..6 sat in ah_assets, and both declared `family: car, source:
    assetharvester`. The selector took the first declared, so I-3 and C-7 baked
    against the loose file and every other leaf used ah_assets — a split nobody
    chose, invisible in the leaf specs, and only findable by reading a frozen
    recipe's `nurec_asset_id`.
    """

    # No harvested sign exists in ah_assets yet, so this one entry still points
    # outside it. Delete the exemption — which forces the entry to move — as
    # soon as one is harvested; V-8 and V-11 are the leaves waiting on it.
    KNOWN_GAP = {"sign_do_not_enter"}

    def test_every_harvested_asset_is_in_public_asset_directory(self):
        from pathlib import Path

        from navsafe.benchmark.editing.assets import registry as registry_mod

        doc = yaml.safe_load(
            (Path(registry_mod.__file__).with_name("registry.yaml")).read_text())
        stray = {
            key: entry.get("ply")
            for key, entry in (doc.get("assets") or {}).items()
            if entry.get("source") == "assetharvester"
            and key not in self.KNOWN_GAP
            and not str(entry.get("ply") or "").startswith("asset/")
        }
        assert not stray, (
            f"harvested assets must come from the published asset directory, but "
            f"{stray} point elsewhere. Move the file into asset and repoint the "
            f"entry, or drop the entry if a hb_* one already supersedes it.")
