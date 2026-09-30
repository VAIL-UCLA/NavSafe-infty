# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Resolving a stitched host for Arrow conversion, from the seed table alone.

A stitched host has no ``seed.json``: it is chosen by a person watching renders,
not by the mining score. Converting one used to be a shell command someone
remembered, and the names it produces have to agree on three sides at once —
the directory ``leaves/hosts.py`` globs for, the Arrow's own log name, and the
scene id the render server answers to. That agreement is what these pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from navsafe.benchmark.world.run_arrow import _from_seed_table

LOG = "2021.09.09.14.18.22_veh-48_00322_00895"
TOKEN = "07bf0601ad425977"
T0, T1 = 1631197983499958, 1631198005000266


@pytest.fixture
def tsv(tmp_path) -> str:
    # Tab-separated, as scenes_500.tsv is: token, log, t0, t1, types, leaves, ...
    path = tmp_path / "scenes.tsv"
    path.write_text(
        f"deadbeefdeadbeef\tsome.other.log\t1\t2\tx\ty\tmined\n"
        f"{TOKEN}\t{LOG}\t{T0}\t{T1}\tstarting_left_turn\tC-2\tmined\n"
    )
    return str(path)


def test_the_scene_is_named_for_the_stitched_host(tsv, tmp_path):
    scene, prov, win, out, log_out, db = _from_seed_table(
        TOKEN, tsv=tsv, corpus=str(tmp_path / "corpus"), nuplan_root=str(tmp_path / "nuplan"))
    # `<token>_20s` on all three sides, or the render server serves a scene the
    # Arrow does not name and the fallback to raster is silent.
    assert scene == f"{TOKEN}_20s"
    assert out == tmp_path / "corpus" / f"{TOKEN}_20s" / "arrow"
    assert log_out == out / "logs" / "nuplan_test" / f"{TOKEN}_20s"


def test_the_window_is_the_table_s_own(tsv, tmp_path):
    _, prov, win, _, _, _ = _from_seed_table(
        TOKEN, tsv=tsv, corpus=str(tmp_path), nuplan_root=str(tmp_path))
    assert prov["log_name"] == LOG and prov["split"] == "test"
    assert (win["recon_t0_us"], win["recon_t1_us"]) == (T0, T1)
    # 20 s, which is what makes it a stitched host rather than one 5 s window.
    assert 20.0 <= (win["recon_t1_us"] - win["recon_t0_us"]) / 1e6 <= 22.0


def test_the_db_is_taken_from_the_test_split(tsv, tmp_path):
    *_, db = _from_seed_table(TOKEN, tsv=tsv, corpus=str(tmp_path),
                              nuplan_root=str(tmp_path / "nuplan"))
    assert db == Path(tmp_path / "nuplan" / "nuplan-v1.1" / "splits" / "test" / f"{LOG}.db")


def test_a_token_the_table_lacks_is_a_finding_not_a_guess(tsv, tmp_path):
    # A reviewer picking a scenario the pipeline cannot see is a fact about the
    # seed table. Inventing a window would convert the wrong 20 seconds and
    # nothing downstream would notice.
    with pytest.raises(LookupError, match="not in"):
        _from_seed_table("0000000000000000", tsv=tsv, corpus=str(tmp_path),
                         nuplan_root=str(tmp_path))


def test_a_prefix_match_is_not_a_match(tmp_path):
    path = tmp_path / "scenes.tsv"
    path.write_text(f"{TOKEN}extra\t{LOG}\t{T0}\t{T1}\tx\ty\tz\n")
    with pytest.raises(LookupError):
        _from_seed_table(TOKEN, tsv=str(path), corpus=str(tmp_path),
                         nuplan_root=str(tmp_path))
