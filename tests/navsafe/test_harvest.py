# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the harvested-actor asset pipeline (``navsafe/harvest``).

Everything here is offline: selection reads Asset Harvester's parse output,
which is JSON, and the manifest is JSON. The two expensive stages (diffusion,
lifting) are subprocess calls to a third party and are not exercised.

What is worth pinning down is the arithmetic that decides which actors get an
asset and which track each asset claims to replace, because both fail silently
in production -- a mis-ranked selection just spends the budget on the wrong
cars, and a manifest whose ids do not match the reconstruction renders exactly
what it was run to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navsafe.benchmark.harvest import manifest, select
from navsafe.benchmark.harvest.ah import (
    aspect_error,
    camera_ids,
    lifted_assets,
    promote,
)
from navsafe.benchmark.harvest.cli import windows_for


def _sample(root, window, cls, track, dists, lwh=(4.5, 1.9, 1.6), n=4):
    d = root / window / cls / track / "input_views"
    d.mkdir(parents=True, exist_ok=True)
    (d / "camera.json").write_text(json.dumps({
        "frame_filenames": [f"frame_{i:02d}.jpeg" for i in range(n)],
        "mask_filenames": [f"mask_{i:02d}.png" for i in range(n)],
        "cam_dists": list(dists),
        "object_lwh": list(lwh),
    }))
    return d.parent


# --------------------------------------------------------------------------
# selection


def test_scan_reads_every_window(tmp_path):
    _sample(tmp_path, "toks1", "vehicle", "aaa", [30.0, 25.0])
    _sample(tmp_path, "toks2", "vehicle", "bbb", [12.0])
    got = select.scan(tmp_path, ["toks1", "toks2"])
    assert {c.track_id for c in got} == {"aaa", "bbb"}
    assert {c.window for c in got} == {"toks1", "toks2"}


def test_scan_skips_unparsable(tmp_path):
    d = tmp_path / "toks1" / "vehicle" / "broken" / "input_views"
    d.mkdir(parents=True)
    (d / "camera.json").write_text("{not json")
    _sample(tmp_path, "toks1", "vehicle", "ok", [10.0])
    got = select.scan(tmp_path, ["toks1"])
    assert [c.track_id for c in got] == ["ok"]


def test_choose_ranks_by_closest_approach(tmp_path):
    _sample(tmp_path, "toks1", "vehicle", "far", [80.0, 70.0])
    _sample(tmp_path, "toks1", "vehicle", "near", [9.0, 40.0])
    _sample(tmp_path, "toks1", "vehicle", "mid", [30.0])
    got = select.choose(select.scan(tmp_path, ["toks1"]), max_assets=2)
    assert [c.track_id for c in got] == ["near", "mid"]


def test_choose_takes_one_window_per_track_the_nearest(tmp_path):
    """A car crossing a window boundary is ONE asset, from its best window.

    Harvesting it once per window would be four diffusion passes for one car,
    and the four assets would then disagree with each other across a handoff.
    """
    _sample(tmp_path, "toks1", "vehicle", "shared", [60.0])
    _sample(tmp_path, "toks2", "vehicle", "shared", [7.5])
    _sample(tmp_path, "toks3", "vehicle", "shared", [22.0])
    got = select.choose(select.scan(tmp_path, ["toks1", "toks2", "toks3"]))
    assert len(got) == 1
    assert got[0].window == "toks2"
    assert got[0].min_dist_m == pytest.approx(7.5)


def test_choose_excludes_deformables_by_default(tmp_path):
    """A rigid asset cannot walk, so a harvested pedestrian is a worse artefact
    than the smear it replaces (hence the gait banks in editing/assets)."""
    _sample(tmp_path, "toks1", "pedestrian", "walker", [4.0])
    _sample(tmp_path, "toks1", "vehicle", "car", [40.0])
    got = select.choose(select.scan(tmp_path, ["toks1"]))
    assert [c.track_id for c in got] == ["car"]


def test_choose_matches_the_dataset_s_own_label_spelling(tmp_path):
    """The class on a parsed sample is the source dataset's full label.

    Measured on 2b7bf25209dd5705: nuPlan clips parse into
    ``nuplanboxdetectionlabel.vehicle``, Waymo ones into
    ``wodperceptionboxdetectionlabel.type_vehicle``. An equality filter matched
    neither and selected nothing, which reads as "the scenario has no actors".
    """
    _sample(tmp_path, "toks1", "nuplanboxdetectionlabel.vehicle", "nuplan_car", [20.0])
    _sample(tmp_path, "toks1", "wodperceptionboxdetectionlabel.type_vehicle",
            "waymo_car", [30.0])
    _sample(tmp_path, "toks1", "nuplanboxdetectionlabel.czone_sign", "sign", [5.0])
    _sample(tmp_path, "toks1", "nuplanboxdetectionlabel.pedestrian", "walker", [3.0])
    got = select.choose(select.scan(tmp_path, ["toks1"]))
    assert [c.track_id for c in got] == ["nuplan_car", "waymo_car"]


def test_choose_drops_single_view_tracks(tmp_path):
    """One view is not multi-view: the diffusion invents a whole car from it."""
    _sample(tmp_path, "toks1", "vehicle", "oneview", [3.0], n=1)
    _sample(tmp_path, "toks1", "vehicle", "usable", [50.0], n=3)
    got = select.choose(select.scan(tmp_path, ["toks1"]))
    assert [c.track_id for c in got] == ["usable"]


def test_write_sample_paths_stages_one_root(tmp_path):
    _sample(tmp_path / "parse", "toks1", "vehicle", "aaa", [10.0])
    _sample(tmp_path / "parse", "toks2", "vehicle", "bbb", [20.0])
    chosen = select.choose(select.scan(tmp_path / "parse", ["toks1", "toks2"]))
    out = select.write_sample_paths(chosen, tmp_path / "staged")
    doc = json.loads(out.read_text())
    assert doc["samples"] == ["vehicle/aaa", "vehicle/bbb"]
    for rel in doc["samples"]:
        assert (tmp_path / "staged" / rel / "input_views" / "camera.json").is_file()


# --------------------------------------------------------------------------
# manifest


def _bank(tmp_path, tracks=("aaa", "bbb")):
    entries = {}
    for t in tracks:
        ply = tmp_path / "lifted" / "vehicle" / t / "gaussians.ply"
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.write_bytes(b"ply\n")
        entries[t] = {"ply": str(ply), "label_class": "vehicle",
                      "cuboids_dims": [4.5, 1.9, 1.6]}
    return entries


def test_manifest_roundtrip(tmp_path):
    entries = _bank(tmp_path)
    manifest.write(tmp_path, "tok_20s", entries, windows=["toks1", "toks2"])
    doc = manifest.read(tmp_path)
    assert doc["scene_id"] == "tok_20s"
    assert doc["windows"] == ["toks1", "toks2"]
    assert set(manifest.ply_by_track(doc)) == {"aaa", "bbb"}


def test_manifest_refuses_a_missing_ply_at_write(tmp_path):
    with pytest.raises(manifest.ManifestError):
        manifest.write(tmp_path, "tok_20s",
                       {"aaa": {"ply": str(tmp_path / "nope.ply")}})


def test_manifest_refuses_a_ply_that_vanished(tmp_path):
    """The render server and the harvester must see the corpus at one path.

    Read-time is where a path mismatch between the harvesting pod and the
    render container shows up, and it must be fatal: half the actors replaced
    and half not is two fidelity regimes inside one scored episode.
    """
    entries = _bank(tmp_path)
    manifest.write(tmp_path, "tok_20s", entries)
    (tmp_path / "lifted" / "vehicle" / "aaa" / "gaussians.ply").unlink()
    with pytest.raises(manifest.ManifestError):
        manifest.read(tmp_path)


def test_manifest_is_relocatable(tmp_path):
    """A bank must work after it is moved — that is what publishing means.

    Paths are stored relative to the manifest's own directory, so the same
    directory harvested under /data and downloaded into someone else's
    bundle resolves without a rebasing step and without configuration.
    """
    import shutil

    src, dst = tmp_path / "here", tmp_path / "moved"
    entries = _bank(src)
    manifest.write(src, "tok_20s", entries)
    raw = json.loads(manifest.manifest_path(src).read_text())
    assert raw["schema"] == 2
    assert not Path(raw["assets"]["aaa"]["ply"]).is_absolute()

    shutil.move(str(src), str(dst))
    doc = manifest.read(dst)
    assert Path(doc["assets"]["aaa"]["ply"]).is_file()
    assert str(dst) in doc["assets"]["aaa"]["ply"]


def test_manifest_still_reads_an_old_absolute_bank(tmp_path):
    """Schema 1 banks keep working in place rather than becoming unreadable."""
    entries = _bank(tmp_path)
    manifest.write(tmp_path, "tok_20s", entries)
    p = manifest.manifest_path(tmp_path)
    doc = json.loads(p.read_text())
    doc["schema"] = 1
    doc["assets"]["aaa"]["ply"] = entries["aaa"]["ply"]
    doc["assets"]["bbb"]["ply"] = entries["bbb"]["ply"]
    p.write_text(json.dumps(doc))
    got = manifest.read(p)
    assert Path(got["assets"]["aaa"]["ply"]).is_file()


def test_manifest_refuses_a_ply_outside_the_bank(tmp_path):
    """Relative paths only mean anything if everything lives under the bank."""
    stray = tmp_path / "elsewhere" / "gaussians.ply"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"ply\n")
    with pytest.raises(manifest.ManifestError, match="outside the bank"):
        manifest.write(tmp_path / "bank", "tok_20s", {"aaa": {"ply": str(stray)}})


def test_manifest_rejects_a_future_schema(tmp_path):
    manifest.write(tmp_path, "tok_20s", _bank(tmp_path))
    p = manifest.manifest_path(tmp_path)
    doc = json.loads(p.read_text())
    doc["schema"] = manifest.SCHEMA_VERSION + 1
    p.write_text(json.dumps(doc))
    with pytest.raises(manifest.ManifestError):
        manifest.read(p)


def test_unmatched_is_the_manifest_minus_what_a_scene_holds(tmp_path):
    manifest.write(tmp_path, "tok_20s", _bank(tmp_path))
    doc = manifest.read(tmp_path)
    assert manifest.unmatched(doc, ["aaa"]) == ["bbb"]
    assert manifest.unmatched(doc, ["aaa", "bbb", "ccc"]) == []


# --------------------------------------------------------------------------
# staging -> bank


def _lifted(root, cls, track, dims=(4.5, 1.9, 1.6)):
    d = root / cls / track
    (d / "multiview").mkdir(parents=True, exist_ok=True)
    (d / "gaussians.ply").write_bytes(b"ply\n")
    (d / "multiview" / "lwh.txt").write_text(" ".join(str(v) for v in dims))
    return d


def test_promote_moves_staged_assets_and_clears_staging(tmp_path):
    """Lifting stages, orientation runs there once, promotion banks the result.

    The staging step is not tidiness: ``orient_gaussians_for_nurec`` rotates 90
    degrees IN PLACE and is not idempotent, so a resumed harvest that re-ran it
    over the whole bank would leave every previously harvested car sideways with
    nothing in the output to say so.
    """
    staging, bank = tmp_path / "_lifting", tmp_path / "lifted"
    _lifted(staging, "vehicle", "aaa")
    _lifted(bank, "vehicle", "kept")
    assert sorted(promote(staging, bank)) == ["aaa"]
    assert not staging.exists()
    assert set(lifted_assets(bank)) == {"aaa", "kept"}


def test_lifted_assets_reads_the_tree_not_the_request(tmp_path):
    """A sample can be skipped mid-run; what shipped is what is on disk."""
    bank = tmp_path / "lifted"
    _lifted(bank, "vehicle", "aaa", dims=(4.577, 1.837, 1.522))
    (bank / "vehicle" / "no_ply").mkdir(parents=True)
    got = lifted_assets(bank)
    assert set(got) == {"aaa"}
    assert got["aaa"]["label_class"] == "vehicle"
    assert got["aaa"]["cuboids_dims"] == [4.577, 1.837, 1.522]


# --------------------------------------------------------------------------
# shape gate


def test_aspect_error_is_zero_for_a_correctly_shaped_asset():
    """Unit-scaled PLY vs a metric cuboid: the comparison is on ratios.

    Post-orientation the axes are x=length, y=height, z=width. A sedan of
    4.58 x 1.84 x 1.52 m lifted to a unit-max-extent cloud should measure the
    same proportions.
    """
    dims = [4.58, 1.84, 1.52]                       # clip cuboid: l, w, h
    extent = [1.0, 1.52 / 4.58, 1.84 / 4.58]        # ply: x, y, z
    assert aspect_error(extent, dims) < 0.01


def test_aspect_error_catches_a_mask_that_caught_two_cars():
    """Measured: aa894e4b59ab587c lifted ~2x too wide for its cuboid.

    The server scales an asset onto the track's box, so this does not render as
    a small car -- it renders as one squashed onto the right footprint, which is
    more visibly wrong than the smear it replaced.
    """
    dims = [5.31, 2.35, 2.21]
    extent = [0.891, 0.324, 0.764]
    assert aspect_error(extent, dims) > 0.5


def test_aspect_error_accepts_a_tall_vehicle():
    """A van is taller than it is wide, so y > z is not evidence of a problem."""
    dims = [6.13, 2.37, 2.98]
    extent = [0.914, 0.425, 0.347]
    assert aspect_error(extent, dims) < 0.1


def test_aspect_error_abstains_without_a_cuboid():
    assert aspect_error([1.0, 0.3, 0.4], []) == 0.0
    assert aspect_error([1.0, 0.3, 0.4], [0.0, 0.0, 0.0]) == 0.0


# --------------------------------------------------------------------------
# clip / window resolution


def test_windows_for_expands_a_stitched_host(tmp_path):
    for name in ("toks1", "toks2", "toks3", "toks4", "othertoks1"):
        (tmp_path / name).mkdir()
    assert windows_for("tok_20s", tmp_path) == ["toks1", "toks2", "toks3", "toks4"]


def test_windows_for_passes_a_plain_scene_through(tmp_path):
    assert windows_for("toks1", tmp_path) == ["toks1"]


def test_camera_ids_come_from_the_clip_not_a_default(tmp_path):
    """Upstream defaults to the Hyperion rig and our clips are nuPlan-converted;
    a wrong id list parses zero tracks and still exits 0."""
    p = tmp_path / "pai_toks1.json"
    p.write_text(json.dumps({"component_stores": [
        {"path": "a.itar", "components": {"cameras": {"camera_pcam_f0": {}}}},
        {"path": "b.itar", "components": {"cameras": {"camera_pcam_l0": {}}}},
        {"path": "c.itar", "components": {"cuboids": {"default": {}}}},
    ]}))
    assert camera_ids(p) == ["camera_pcam_f0", "camera_pcam_l0"]


# --------------------------------------------------------------------------
# moving vs parked


def test_choose_keeps_only_actors_that_drove(tmp_path):
    """Replacing a parked car is a losing trade, so it is not selected.

    The ego drives past a parked car, so the reconstruction saw it across a wide
    arc and holds real pixels for every angle a policy is likely to want — and
    a lifted asset measurably loses to that (a parked FedEx truck whose branding
    is legible in the reconstruction came back as a plain grey box truck). A
    moving car travels with or against the ego, so the reconstruction has almost
    no angular baseline on it, and that is the one that falls apart.
    """
    _sample(tmp_path, "toks1", "vehicle", "parked_close", [5.0])
    _sample(tmp_path, "toks1", "vehicle", "driving_far", [45.0])
    motion = {"parked_close": {"displacement_m": 0.7, "path_m": 3.8},
              "driving_far": {"displacement_m": 41.4, "path_m": 41.6}}
    got = select.choose(select.scan(tmp_path, ["toks1"]), motion=motion)
    assert [c.track_id for c in got] == ["driving_far"]


def test_motion_uses_net_displacement_not_path_length(tmp_path):
    """Cuboid jitter sums into metres of path; displacement does not.

    Measured on one 5 s window: a parked car accumulated 3.8 m of path against
    0.7 m of displacement, while a car driving through had 41.6 m against
    41.4 m. Ranking on path length would have harvested the parked car.
    """
    m = {"jittery_parked": {"displacement_m": 0.7, "path_m": 3.8}}
    assert select.displacement_of(m, "jittery_parked") == pytest.approx(0.7)


def test_a_track_absent_from_the_motion_table_counts_as_parked(tmp_path):
    """An unexplained gap should cost an asset, not spend one."""
    assert select.displacement_of({}, "unknown") == 0.0
    _sample(tmp_path, "toks1", "vehicle", "unknown", [5.0])
    assert select.choose(select.scan(tmp_path, ["toks1"]), motion={"other": {}}) == []


def test_include_parked_is_still_possible(tmp_path):
    """motion=None keeps the old behaviour, for diagnosing."""
    _sample(tmp_path, "toks1", "vehicle", "parked", [5.0])
    got = select.choose(select.scan(tmp_path, ["toks1"]), motion=None)
    assert [c.track_id for c in got] == ["parked"]


def test_bank_is_cumulative_but_the_manifest_is_not(tmp_path):
    """A narrower re-selection must not ship the previous run's extra assets.

    The lifted tree is kept so a re-run can skip work, so it accumulates every
    track ever chosen. The manifest is what the renderer acts on, and it has to
    mean "these actors, deliberately" — an asset from an earlier, wider
    selection sitting in it would replace an actor this run excluded, with
    nothing in the file to say why it was there.
    """
    entries = _bank(tmp_path, tracks=("mover", "parked_from_last_time"))
    manifest.write(tmp_path, "tok_20s", {"mover": entries["mover"]})
    doc = manifest.read(tmp_path)
    assert set(doc["assets"]) == {"mover"}
    assert (tmp_path / "lifted" / "vehicle" / "parked_from_last_time"
            / "gaussians.ply").is_file()


# --------------------------------------------------------------------------
# packing for publication


def test_pack_indexes_every_valid_bank_and_names_the_broken(tmp_path, capsys):
    """Packing validates rather than copies.

    A bank already lives where it should be published — inside the scenario,
    which is where the eval flag looks for it — so the useful work is proving
    each one is complete and portable and writing down what goes where. A
    staging copy would only be a second 10 GB tree to keep in step.
    """
    import argparse

    from navsafe.benchmark.harvest.cli import cmd_pack

    corpus = tmp_path / "corpus"
    good = corpus / "good_20s" / "ah_assets"
    manifest.write(good, "good_20s", _bank(good), windows=["goods1"])
    bad = corpus / "bad_20s" / "ah_assets"
    manifest.write(bad, "bad_20s", _bank(bad))
    (bad / "lifted" / "vehicle" / "aaa" / "gaussians.ply").unlink()

    out = tmp_path / "pack"
    rc = cmd_pack(argparse.Namespace(
        corpus=str(corpus), dest=str(out), repo="org/ds", prefix="full_test",
        strict=False))
    assert rc == 0

    idx = json.loads((out / "index.json").read_text())
    assert idx["n_scenarios"] == 1 and idx["n_assets"] == 2
    assert set(idx["scenarios"]) == {"good_20s"}
    assert "bad_20s" in (out / "broken.txt").read_text()
    assert not (out / "unfinished.txt").exists()

    plan = (out / "upload.sh").read_text()
    assert "org/ds" in plan and "good_20s/ah_assets" in plan
    assert "bad_20s" not in plan
    assert (out / "README.md").is_file()


def test_pack_separates_unfinished_from_broken(tmp_path):
    """Mid-fleet, most banks have no manifest yet — that is not a failure.

    Reporting "did not validate" for a harvest still in its lifting step makes a
    healthy run look like a disaster and buries the banks that really are wrong.
    """
    import argparse

    from navsafe.benchmark.harvest.cli import cmd_pack

    corpus = tmp_path / "corpus"
    (corpus / "running_20s" / "ah_assets" / "lifted").mkdir(parents=True)
    good = corpus / "good_20s" / "ah_assets"
    manifest.write(good, "good_20s", _bank(good))

    out = tmp_path / "pack"
    assert cmd_pack(argparse.Namespace(
        corpus=str(corpus), dest=str(out), repo="org/ds", prefix="full_test",
        strict=False)) == 0
    assert (out / "unfinished.txt").read_text().strip() == "running_20s"
    assert not (out / "broken.txt").exists()
    assert json.loads((out / "index.json").read_text())["n_scenarios"] == 1


def test_pack_strict_fails_on_a_broken_bank(tmp_path):
    import argparse

    from navsafe.benchmark.harvest.cli import cmd_pack

    corpus = tmp_path / "corpus"
    bad = corpus / "bad_20s" / "ah_assets"
    manifest.write(bad, "bad_20s", _bank(bad))
    (bad / "lifted" / "vehicle" / "aaa" / "gaussians.ply").unlink()
    assert cmd_pack(argparse.Namespace(
        corpus=str(corpus), dest=str(tmp_path / "p"), repo="org/ds",
        prefix="full_test", strict=True)) == 1


def test_scenarios_treats_an_older_schema_as_still_needing_harvest(tmp_path):
    """A bank predating the current rules is not done.

    Schema 1 stored absolute paths (not portable) and a bank selected before the
    moving-only rule holds parked cars. Both need the scenario re-run, so
    `scenarios` must list them — otherwise a corpus-wide re-harvest silently
    skips exactly the banks that need it most.
    """
    import argparse

    from navsafe.benchmark.harvest.cli import cmd_scenarios

    corpus = tmp_path / "corpus"
    for tok, schema in (("cur", 2), ("old", 1)):
        scene = f"{tok}_20s"
        (corpus / f"{tok}s1" / "clips" / f"{tok}s1").mkdir(parents=True)
        (corpus / f"{tok}s1" / "clips" / f"{tok}s1" / f"pai_{tok}s1.json").write_text("{}")
        art = corpus / f"{tok}s1" / "output_5cam" / f"{tok}s1" / "artifacts"
        art.mkdir(parents=True)
        (art / "last.usdz").write_bytes(b"x")
        adir = corpus / scene / "ah_assets"
        adir.mkdir(parents=True)
        (adir / "replace_manifest.json").write_text(
            json.dumps({"schema": schema, "assets": {}}))

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_scenarios(argparse.Namespace(corpus=str(corpus), all=False))
    listed = buf.getvalue().split()
    assert "old_20s" in listed
    assert "cur_20s" not in listed


def test_verify_survives_kubernetes_service_link_env(monkeypatch):
    """`NUREC_GRPC_PORT` is not necessarily a port.

    A Service named `nurec-grpc` makes the kubelet inject
    `NUREC_GRPC_PORT=tcp://<ip>:<port>` into every pod in the namespace. Read
    naively that builds `host:tcp://ip:8080` and every window reports
    UNAVAILABLE — which, before this, `verify` went on to blame on the bank.
    """
    monkeypatch.setenv("NUREC_GRPC_PORT", "tcp://10.106.223.53:8080")
    port = str(__import__("os").environ["NUREC_GRPC_PORT"]).rsplit(":", 1)[-1]
    assert port == "8080"


def test_pack_calls_an_empty_scenario_empty_not_broken(tmp_path):
    """A quiet scenario with nothing to replace is an answer, not a failure.

    Ten of 87 came back that way — no moving vehicle the log saw well enough to
    lift — and lumping them in with real failures would bury the ones that
    matter and imply work still to do.
    """
    import argparse

    from navsafe.benchmark.harvest.cli import cmd_pack

    corpus = tmp_path / "corpus"
    quiet = corpus / "quiet_20s" / "ah_assets"
    quiet.mkdir(parents=True)
    (quiet / "harvest.log").write_text("nothing to harvest\n")
    (quiet / manifest.EMPTY_MARKER).write_text("nothing to replace\n")
    good = corpus / "good_20s" / "ah_assets"
    manifest.write(good, "good_20s", _bank(good))

    out = tmp_path / "pack"
    assert cmd_pack(argparse.Namespace(
        corpus=str(corpus), dest=str(out), repo="org/ds", prefix="full_test",
        strict=False)) == 0
    assert (out / "empty.txt").read_text().strip() == "quiet_20s"
    assert not (out / "unfinished.txt").exists()
    assert not (out / "broken.txt").exists()
    assert json.loads((out / "index.json").read_text())["empty_scenarios"] == ["quiet_20s"]


def test_scenarios_enumerates_by_token_not_by_arrow_directory(tmp_path):
    """Harvesting needs the NCore clips; Arrow is an EVAL prerequisite.

    `<token>_20s` is created by the Arrow conversion, so listing scenarios by
    that directory reported 87 of 407 harvestable scenarios and hid 320 that
    were ready. The bank still belongs at `<token>_20s/ah_assets` — a later
    Arrow conversion lands in the same directory, so it is already in place.
    """
    import argparse
    import contextlib
    import io

    from navsafe.benchmark.harvest.cli import cmd_scenarios

    corpus = tmp_path / "corpus"
    for tok in ("witharrow", "noarrow"):
        for w in (1, 2):
            d = corpus / f"{tok}s{w}"
            (d / "clips" / f"{tok}s{w}").mkdir(parents=True)
            (d / "clips" / f"{tok}s{w}" / f"pai_{tok}s{w}.json").write_text("{}")
            art = d / "output_5cam" / f"{tok}s{w}" / "artifacts"
            art.mkdir(parents=True)
            (art / "last.usdz").write_bytes(b"x")
    (corpus / "witharrow_20s" / "arrow").mkdir(parents=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_scenarios(argparse.Namespace(corpus=str(corpus), all=False))
    listed = buf.getvalue().split()
    assert sorted(listed) == ["noarrow_20s", "witharrow_20s"]


def test_degenerate_ply_is_dropped_not_fatal(tmp_path):
    """TokenGS sometimes lifts an empty cloud; one asset may fail, a scenario may not.

    Orientation reads every PLY in a subprocess and dies on a header with no
    vertices, which took down a whole scenario's harvest (8b20ada64fe8512a_20s).
    """
    from navsafe.benchmark.harvest.ah import drop_degenerate, ply_vertex_count

    staging = tmp_path / "_lifting"
    good = staging / "vehicle" / "good"
    bad = staging / "vehicle" / "bad"
    for d in (good, bad):
        d.mkdir(parents=True)
    good.joinpath("gaussians.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1200\nend_header\n")
    bad.joinpath("gaussians.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")

    assert ply_vertex_count(good / "gaussians.ply") == 1200
    assert drop_degenerate(staging) == ["bad"]
    assert good.is_dir() and not bad.exists()


def test_an_empty_reharvest_removes_the_stale_manifest(tmp_path):
    """A bank whose selection is no longer valid must not keep shipping it.

    Re-running under the moving-only rule can decide a scenario has nothing to
    replace. Returning early left the previous manifest — listing parked cars
    chosen under the old rule — in place, ready to be published as current.
    """
    adir = tmp_path / "ah_assets"
    manifest.write(adir, "tok_20s", _bank(adir))
    p = manifest.manifest_path(adir)
    assert p.is_file()
    p.unlink()          # what cmd_harvest does on the "nothing to harvest" path
    assert not p.is_file()
    assert (adir / "lifted" / "vehicle" / "aaa" / "gaussians.ply").is_file()


def test_pack_reads_the_empty_marker_not_the_leftovers(tmp_path):
    """"Ran and found nothing" and "still running" are the same absence of a
    manifest, and a scenario judged empty can still hold PLYs from an earlier,
    wider selection. The marker is what tells them apart."""
    import argparse

    from navsafe.benchmark.harvest.cli import cmd_pack

    corpus = tmp_path / "corpus"
    judged = corpus / "judged_20s" / "ah_assets"
    _bank(judged)                                   # leftover PLYs, no manifest
    (judged / manifest.EMPTY_MARKER).write_text("nothing to replace\n")
    running = corpus / "running_20s" / "ah_assets"
    _bank(running)                                  # same shape, no marker

    out = tmp_path / "pack"
    assert cmd_pack(argparse.Namespace(
        corpus=str(corpus), dest=str(out), repo="org/ds", prefix="full_test",
        strict=False)) == 0
    assert (out / "empty.txt").read_text().strip() == "judged_20s"
    assert (out / "unfinished.txt").read_text().strip() == "running_20s"


def test_scenarios_counts_a_marked_empty_scenario_as_done(tmp_path):
    """Nothing to replace is a finished answer, not outstanding work.

    Without this every corpus sweep re-parses the empty scenarios forever and
    reports them as still to do.
    """
    import argparse
    import contextlib
    import io

    from navsafe.benchmark.harvest.cli import cmd_scenarios

    corpus = tmp_path / "corpus"
    for tok in ("empty", "todo"):
        d = corpus / f"{tok}s1"
        (d / "clips" / f"{tok}s1").mkdir(parents=True)
        (d / "clips" / f"{tok}s1" / f"pai_{tok}s1.json").write_text("{}")
        art = d / "output_5cam" / f"{tok}s1" / "artifacts"
        art.mkdir(parents=True)
        (art / "last.usdz").write_bytes(b"x")
    adir = corpus / "empty_20s" / "ah_assets"
    adir.mkdir(parents=True)
    (adir / manifest.EMPTY_MARKER).write_text("nothing to replace\n")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_scenarios(argparse.Namespace(corpus=str(corpus), all=False))
    assert buf.getvalue().split() == ["todo_20s"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_harvest_motion_keeps_any_moving_clip(tmp_path, monkeypatch, cached, reverse):
    """Later stops must not erase motion, for cached and fresh clip records."""
    from navsafe.benchmark.harvest import cli

    windows = ["toks1", "toks2"]
    if reverse:
        windows.reverse()
    records = {
        "toks1": {"mover": {"displacement_m": 5.0},
                  "parked": {"displacement_m": 0.5},
                  "boundary": {"displacement_m": 2.0}},
        "toks2": {"mover": {"displacement_m": 0.1},
                  "parked": {"displacement_m": 0.8},
                  "late": {"displacement_m": 3.0},
                  "boundary": {"displacement_m": 0.0}},
    }
    parse = tmp_path / "parse"
    for window, tracks in records.items():
        for tid in tracks:
            _sample(parse, window, "vehicle", tid, [10.0])
        (parse / window / "sample_paths.json").write_text("{}")
        if cached:
            (parse / window / "motion.json").write_text(json.dumps(tracks))

    monkeypatch.setattr(cli.ah, "check_install", lambda: None)
    monkeypatch.setattr(cli, "windows_for", lambda *a: windows)
    calls = []

    def read_motion(window, *a, **kw):
        calls.append(window)
        return records[window]

    monkeypatch.setattr(cli.ah, "window_motion", read_motion)
    choose = select.choose

    class SelectionChecked(Exception):
        pass

    def check_selection(candidates, **kw):
        assert kw["motion"]["mover"]["displacement_m"] == 5.0
        assert kw["motion"]["parked"]["displacement_m"] == 0.8
        assert {c.track_id for c in choose(candidates, **kw)} == {
            "mover", "late", "boundary"}
        raise SelectionChecked

    monkeypatch.setattr(cli.select, "choose", check_selection)
    args = cli.build_parser().parse_args([
        "--out", str(tmp_path / "assets"), "harvest", "tok_20s",
        "--parse-root", str(parse)])
    with pytest.raises(SelectionChecked):
        cli.cmd_harvest(args)
    assert calls == ([] if cached else windows)
