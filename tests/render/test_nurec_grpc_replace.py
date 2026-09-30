# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for harvested-actor replacement (Level C).

``_replace_harvested_assets`` re-skins a reconstruction's own logged actors from
a NavSafe asset bank, so that an actor a closed-loop policy meets from a
viewpoint the logged ego never had stops rendering as a smear. The fake proto
bundle and stub are the ones the insert tests already use.

What is worth pinning down here is everything that fails SILENTLY on a live
server:

* the per-scene intersection -- a 20 s bank covers four 5 s scenes, each holding
  a different subset of the same cars, and sending one scene's ids to another is
  an INVALID_ARGUMENT;
* the AABB coming from the server's own box rather than the harvest record;
* a bank whose ids match nothing being fatal, not a no-op, because a run that
  quietly renders the original actors is indistinguishable in the metrics from
  the run that was wanted;
* ``close()`` restoring, or the swap leaks onto the next eval sharing the warm
  server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navsafe.benchmark.harvest import manifest
from navsafe.render.nurec_grpc import NuRecGrpcSceneRenderer
from tests.render.test_nurec_grpc_insert import _fake_bundle, _FakeStub


def _bank(tmp_path, tracks):
    entries = {}
    for tid in tracks:
        ply = tmp_path / "lifted" / "vehicle" / tid / "gaussians.ply"
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.write_bytes(b"ply\n")
        entries[tid] = {"ply": str(ply), "label_class": "vehicle",
                        "cuboids_dims": [4.5, 1.9, 1.6]}
    manifest.write(tmp_path, "tok_20s", entries)
    return str(manifest.manifest_path(tmp_path))


def _renderer(scenes):
    """A renderer with ``scenes`` already enumerated, as setup would leave it."""
    r = NuRecGrpcSceneRenderer()
    r._g = _fake_bundle()
    r._stub = _FakeStub()
    r._scene_id = next(iter(scenes))
    r._dyn_track_ids_by_scene = {s: set(v) for s, v in scenes.items()}
    r._dyn_track_size_by_scene = {
        s: {t: (4.5, 1.9, 1.6) for t in v} for s, v in scenes.items()}
    r._dyn_track_ids = set().union(*(set(v) for v in scenes.values()))
    return r


def test_no_env_no_rpc(monkeypatch):
    """The default path must not touch the server at all."""
    monkeypatch.delenv("NUREC_GRPC_ASSET_REPLACE", raising=False)
    r = _renderer({"s1": ["a"]})
    r._replace_harvested_assets()
    assert r._stub.edit_requests == []
    assert r._replaced_asset_ids == set()


def test_replaces_only_the_tracks_a_scene_holds(tmp_path, monkeypatch):
    """A 20 s bank covers four 5 s scenes; each takes only its own actors.

    Sending one scene's ids to another is an INVALID_ARGUMENT on a live server,
    and a bank track present in no scene is normal, not an error.
    """
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a", "b", "c"]))
    r = _renderer({"s1": ["a", "x"], "s2": ["b", "c", "y"]})
    r._handoff = [["s1", None, None, None], ["s2", None, None, None]]
    r._replace_harvested_assets()

    by_scene = {req.scene_id: req for req in r._stub.edit_requests}
    assert set(by_scene) == {"s1", "s2"}
    assert [a.original_id for a in by_scene["s1"].replace] == ["a"]
    assert [a.original_id for a in by_scene["s2"].replace] == ["b", "c"]
    assert r._replaced_asset_ids == {"a", "b", "c"}


def test_replacement_id_is_an_absolute_ply_path(tmp_path, monkeypatch):
    """The server opens the file itself, so it must get an absolute path.

    The manifest on disk stores the path RELATIVE to itself, so a bank can be
    moved or published; resolving happens once on read. Both halves matter: a
    relative path reaching the server would be resolved against the server's own
    working directory, which is not the bank.
    """
    bank = _bank(tmp_path, ["a"])
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", bank)
    r = _renderer({"s1": ["a"]})
    r._replace_harvested_assets()
    action = r._stub.edit_requests[0].replace[0]
    assert Path(action.replacement_id).is_absolute()
    assert Path(action.replacement_id).is_file()
    assert action.replacement_id.endswith("/lifted/vehicle/a/gaussians.ply")
    on_disk = json.loads(open(bank).read())["assets"]["a"]["ply"]
    assert not Path(on_disk).is_absolute()


def test_aabb_comes_from_the_server_not_the_harvest(tmp_path, monkeypatch):
    """The server's box is the one the reconstruction is actually posed against."""
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a"]))
    r = _renderer({"s1": ["a"]})
    r._dyn_track_size_by_scene["s1"]["a"] = (6.1, 2.4, 3.0)   # a van, not the 4.5 in the bank
    r._replace_harvested_assets()
    size = r._stub.edit_requests[0].replace[0].object_size
    assert (size.size_x, size.size_y, size.size_z) == (6.1, 2.4, 3.0)


def test_missing_server_box_sends_no_aabb(tmp_path, monkeypatch):
    """Rather than inventing one: an omitted object_size lets the server decide."""
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a"]))
    r = _renderer({"s1": ["a"]})
    r._dyn_track_size_by_scene["s1"] = {}
    r._replace_harvested_assets()
    assert r._stub.edit_requests[0].replace[0].object_size is None


def test_a_bank_for_another_reconstruction_is_fatal(tmp_path, monkeypatch):
    """Not one id matching means the bank is not this scenario's.

    Degrading here would render the original smeared actors and score them as if
    they had been replaced -- indistinguishable, in the metrics, from the run
    that was asked for.
    """
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a", "b"]))
    r = _renderer({"s1": ["p", "q"]})
    with pytest.raises(RuntimeError, match="different reconstruction"):
        r._replace_harvested_assets()


def test_a_rejected_edit_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a"]))
    r = _renderer({"s1": ["a"]})

    class _Reject:
        success = False
        message = "no such asset"

    r._stub.edit_assets = lambda req, timeout=None: _Reject()
    with pytest.raises(RuntimeError, match="replace rejected"):
        r._replace_harvested_assets()


def test_a_vanished_asset_is_fatal(tmp_path, monkeypatch):
    """The harvester and the render container must see one path for the bank."""
    bank = _bank(tmp_path, ["a"])
    (tmp_path / "lifted" / "vehicle" / "a" / "gaussians.ply").unlink()
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", bank)
    r = _renderer({"s1": ["a"]})
    with pytest.raises(manifest.ManifestError):
        r._replace_harvested_assets()


def test_close_restores_a_replace_only_run(tmp_path, monkeypatch):
    """A replaced actor's original gaussians live in the same server-side
    snapshot as an inserted track, so a run that only replaced still has to
    restore -- or the next eval on this warm server renders THIS run's assets.
    """
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a", "b"]))
    r = _renderer({"s1": ["a"], "s2": ["b"]})
    r._handoff = [["s1", None, None, None], ["s2", None, None, None]]
    r._replace_harvested_assets()
    assert r._inserted_asset_ids == set()          # nothing was inserted
    r.close()
    assert {req.scene_id for req in r._stub.restore_requests} == {"s1", "s2"}
    assert r._replaced_asset_ids == set()


def test_reset_rolls_back_every_scene_the_run_will_touch():
    """Setup starts from the reconstruction, not from whatever was left behind.

    edit_assets state lives in the SERVER, so a killed episode, a crash before
    close(), or an ad-hoc probe leaves its assets applied and the next run
    renders them with nothing in the output to say so — including a run that
    asked for no replacement at all, which is then scored as the baseline it is
    not. Caught doing exactly that on 2026-08-28, one minute before it would
    have produced the "before" half of a comparison.
    """
    r = _renderer({"s1": ["a"], "s2": ["b"], "s3": ["c"]})
    assert r._reset_server_scenes(["s1", "s2", "s3"]) == ["s1", "s2", "s3"]
    assert [q.scene_id for q in r._stub.restore_requests] == ["s1", "s2", "s3"]


def test_reset_survives_a_server_that_refuses_it():
    """Logged, not raised: failing to reset must not kill a render, but it does
    mean an earlier run's edits are still in the scene."""
    r = _renderer({"s1": ["a"]})

    def _boom(req, timeout=None):
        raise RuntimeError("no")

    r._stub.restore_model_parameters = _boom
    assert r._reset_server_scenes(["s1"]) == []


def test_max_keeps_the_nearest_actors(tmp_path, monkeypatch):
    """The VRAM cap spends its budget on the actors that fill the frame.

    Measured 2026-08-28 on a 24 GB card: all 26 of a scenario's assets fit
    statically with 0.5 GiB spare, then killed the episode at frame 151 on a
    1.29 GiB render allocation. Nearest-first is the ranking the manifest
    already carries -- a car 50 m up the road costs the same VRAM as one at 8 m
    and almost no pixels.
    """
    entries = {}
    for tid, dist in (("far", 40.0), ("near", 7.0), ("mid", 20.0)):
        ply = tmp_path / "lifted" / "vehicle" / tid / "gaussians.ply"
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.write_bytes(b"ply\n")
        entries[tid] = {"ply": str(ply), "label_class": "vehicle",
                        "cuboids_dims": [4.5, 1.9, 1.6], "min_ego_dist_m": dist}
    manifest.write(tmp_path, "tok_20s", entries)
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE",
                       str(manifest.manifest_path(tmp_path)))
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE_MAX", "2")
    r = _renderer({"s1": ["near", "mid", "far"]})
    r._replace_harvested_assets()
    assert sorted(a.original_id for a in r._stub.edit_requests[0].replace) == ["mid", "near"]


def test_max_counts_per_scene_instances_not_manifest_entries(tmp_path, monkeypatch):
    """The budget is resident copies, not manifest rows.

    A 20 s scenario is served as four 5 s scenes and a car in three of them is
    loaded three times. Counting rows let a cap of 12 mean up to 48 resident
    assets, and the episode died mid-run on a 24 GB card while its un-replaced
    half finished clean (measured 2026-08-29 across 9 of 51 scenarios).

    Here ``near`` is in both scenes and ``mid`` in one: a budget of 2 buys the
    nearest car only, and stops there rather than skipping ahead to ``mid``.
    """
    entries = {}
    for tid, dist in (("near", 7.0), ("mid", 20.0)):
        ply = tmp_path / "lifted" / "vehicle" / tid / "gaussians.ply"
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.write_bytes(b"ply\n")
        entries[tid] = {"ply": str(ply), "label_class": "vehicle",
                        "cuboids_dims": [4.5, 1.9, 1.6], "min_ego_dist_m": dist}
    manifest.write(tmp_path, "tok_20s", entries)
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE",
                       str(manifest.manifest_path(tmp_path)))
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE_MAX", "2")
    r = _renderer({"s1": ["near", "mid"], "s2": ["near"]})
    r._handoff = [["s1", None, None, None], ["s2", None, None, None]]
    r._replace_harvested_assets()
    replaced = sorted({a.original_id for q in r._stub.edit_requests
                       for a in q.replace})
    assert replaced == ["near"]


def test_max_unset_still_budgets(tmp_path, monkeypatch):
    """Unset is a real budget, not "no limit".

    An ordinary bank does not fit: harvest keeps up to 10 tracks and each is
    resident in every served scene that holds it. Defaulting to unlimited put
    the failure twenty minutes into an episode as ``CUDA out of memory``
    instead of in the log at start-up, so the default caps.
    """
    tracks = [f"t{i:02d}" for i in range(12)]
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, tracks))
    monkeypatch.delenv("NUREC_GRPC_ASSET_REPLACE_MAX", raising=False)
    r = _renderer({"s1": tracks})
    r._replace_harvested_assets()
    assert len(r._stub.edit_requests[0].replace) == 10


def test_max_unset_does_not_trim_a_bank_that_fits(tmp_path, monkeypatch):
    """Below the budget nothing is dropped -- the cap is a ceiling, not a quota."""
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, ["a", "b", "c"]))
    monkeypatch.delenv("NUREC_GRPC_ASSET_REPLACE_MAX", raising=False)
    r = _renderer({"s1": ["a", "b", "c"]})
    r._replace_harvested_assets()
    assert len(r._stub.edit_requests[0].replace) == 3


def test_max_zero_means_no_limit(tmp_path, monkeypatch):
    """The escape hatch for a bigger card is explicit, and it is ``0``."""
    tracks = [f"t{i:02d}" for i in range(12)]
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE", _bank(tmp_path, tracks))
    monkeypatch.setenv("NUREC_GRPC_ASSET_REPLACE_MAX", "0")
    r = _renderer({"s1": tracks})
    r._replace_harvested_assets()
    assert len(r._stub.edit_requests[0].replace) == 12
