# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""A V-1 red light is authored, because the corpus does not contain one.

Measured 2026-08-31 on the two V-1 bundles on disk: ``00c1e4eb4a045f20``'s one
signalled lane logs ``LANE_STATE_GO`` for 42 frames then ``LANE_STATE_UNKNOWN``
for 159, and ``05d0a1a763fc5334``'s sixteen are GO/UNKNOWN with a single STOP
frame. ``_red_lane_ids`` reads UNKNOWN as not-red, so the evaluator's ``TL``
subscore is 1.0 on every scored frame of every run and the ``red_light``
channel has never once fired. Holding the logged state would hold a green.

The case that matters most here is a lane the log never mentions at all: it has
no ``dynamic_map_states`` row, and an absent row is exactly what reads as
"not red", so the override has to CREATE the entry rather than edit one.
"""

from __future__ import annotations

import json

import pytest

from navsafe.benchmark import signal_override as so


def _sd(length: int = 5, lanes: dict | None = None) -> dict:
    return {
        "length": length,
        "dynamic_map_states": dict(lanes or {}),
        "tracks": {"ego": {"state": {"position": [[0.0, 0.0]] * length}}},
        "metadata": {"sdc_id": "ego"},
    }


def _states(sd: dict, lane: str) -> list[str]:
    return sd["dynamic_map_states"][lane]["state"]["object_state"]


def test_a_lane_the_log_never_reported_gets_an_entry():
    sd = _sd()
    assert so.apply(sd, {"lane": "52246"}) == 5
    assert _states(sd, "52246") == ["LANE_STATE_STOP"] * 5


def test_a_logged_green_is_overwritten_for_every_frame():
    sd = _sd(lanes={"53726": {"state": {"object_state":
                                        ["LANE_STATE_GO"] * 2 + ["LANE_STATE_UNKNOWN"] * 3}}})
    so.apply(sd, {"lane": "53726", "state": "LANE_STATE_STOP"})
    assert _states(sd, "53726") == ["LANE_STATE_STOP"] * 5


def test_other_lanes_are_left_alone():
    sd = _sd(lanes={"other": {"state": {"object_state": ["LANE_STATE_GO"] * 5}}})
    so.apply(sd, {"lane": "52246"})
    assert _states(sd, "other") == ["LANE_STATE_GO"] * 5


def test_no_override_is_a_no_op():
    sd = _sd()
    assert so.apply(sd, None) == 0
    assert sd["dynamic_map_states"] == {}


def test_length_falls_back_to_the_ego_track():
    sd = _sd(length=7)
    del sd["length"]
    assert so.apply(sd, {"lane": "52246"}) == 7


def test_a_lane_less_override_is_refused():
    with pytest.raises(ValueError, match="lane"):
        so.validate({"state": "LANE_STATE_STOP"})


def test_an_unknown_state_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        so.validate({"lane": "1", "state": "LANE_STATE_FLASHING"})


def test_a_frame_range_is_refused_until_the_success_rule_can_express_it():
    """The schema carries it; the scorer cannot answer it yet.

    After a switch to green, holding stops being correct and the hold fraction
    would have to mean the opposite of what it means before -- so the range is
    rejected loudly rather than silently scored as an all-red episode.
    """
    with pytest.raises(NotImplementedError):
        so.validate({"lane": "1", "frames": [0, 40]})


def test_the_manifest_is_the_source_of_truth(tmp_path):
    bundle = tmp_path / "tok"
    (bundle / "arrow").mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps(
        {"signal_override": {"lane": "52246", "state": "LANE_STATE_STOP"}}))
    assert so.from_manifest(bundle / "arrow")["lane"] == "52246"
    assert so.from_manifest(bundle)["state"] == "LANE_STATE_STOP"


def test_a_bundle_without_the_key_declares_nothing(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"token": "t"}))
    assert so.from_manifest(tmp_path) is None
    assert so.from_manifest(tmp_path / "missing") is None


def test_the_env_form_matches_the_manifest_form(monkeypatch):
    """The sim reads the env, the scorer reads the manifest; one shape."""
    monkeypatch.setenv(so.ENV_VAR, json.dumps({"lane": "52246"}))
    assert so.from_env() == {"lane": "52246", "lanes": ["52246"],
                             "state": "LANE_STATE_STOP", "frames": "all"}
    monkeypatch.setenv(so.ENV_VAR, "")
    assert so.from_env() is None


def test_several_lanes_can_be_reddened_at_once():
    """One connector is not a junction.

    The evaluator exempts a crossing when a non-red lane still offers a way
    through on the ego's heading, so reddening one connector of a multi-lane
    approach leaves the siblings green. Measured on 05d0a1a763fc5334: with only
    52246 red, the TL column stayed 1.0 on all 599 scored frames while the ego
    drove straight through.
    """
    sd = _sd()
    assert so.apply(sd, {"lane": ["52246", "47390", "47391"]}) == 15
    for lane in ("52246", "47390", "47391"):
        assert _states(sd, lane) == ["LANE_STATE_STOP"] * 5


def test_one_lane_and_a_list_of_one_mean_the_same_thing():
    assert so.validate({"lane": "52246"})["lanes"] == ["52246"]
    assert so.validate({"lane": ["52246"]})["lanes"] == ["52246"]


def test_an_empty_lane_list_is_refused():
    with pytest.raises(ValueError, match="lane"):
        so.validate({"lane": []})
