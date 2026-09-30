"""Round-trip fidelity test for the ScenarioDescription → py123d Arrow writer.

This is the acceptance gate for ``write_scenario_description_arrow``: a
ScenarioDescription is written to a py123d Arrow log, then rediscovered and
read back through the *exact runtime path* (``enumerate_scenes`` → adapter →
projection). The reloaded scenario must match the input in track count, frame
count, ego trajectory, agent types, and map geometry.

Requires a real py123d install (skipped otherwise), since the writer targets
py123d's native Arrow datatypes.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("py123d")

from navsafe.scenario.py123d_adapter import Py123DAdapterConfig, scenario_from_py123d_scene
from navsafe.scenario.py123d_scenario_description import py123d_to_scenario_description
from navsafe.scenario.py123d_scenes import enumerate_scenes
from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.to_py123d_arrow import write_scenario_description_arrow


# ─────────────────────────── test scenario builder ───────────────────────────

EGO_ID = "ego"
T = 10  # frames
DT_S = 0.1


def _track(
    obj_type: str, object_id: str, xy0: tuple[float, float], vel: tuple[float, float], heading: float
) -> dict:
    """A constant-(world)-velocity track over T frames at a fixed heading.

    Heading is intentionally decoupled from the velocity direction so the
    round-trip exercises orientation independently of motion (and catches any
    body-vs-world velocity-frame error).
    """
    vx, vy = vel
    pos = np.zeros((T, 3), dtype=np.float32)
    pos[:, 0] = xy0[0] + vx * DT_S * np.arange(T)
    pos[:, 1] = xy0[1] + vy * DT_S * np.arange(T)
    state = {
        SD.POSITION: pos,
        "length": np.full(T, 4.5, dtype=np.float32),
        "width": np.full(T, 1.9, dtype=np.float32),
        "height": np.full(T, 1.6, dtype=np.float32),
        SD.HEADING: np.full(T, heading, dtype=np.float32),
        "velocity": np.tile(np.array([vx, vy], dtype=np.float32), (T, 1)),
        "valid": np.ones(T, dtype=bool),
    }
    return {
        SD.TYPE: obj_type,
        SD.STATE: state,
        SD.METADATA: {"track_length": T, "type": obj_type, "object_id": object_id},
    }


def _polyline(y: float) -> np.ndarray:
    xs = np.linspace(0.0, 20.0, 21, dtype=np.float32)
    return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1)


# Ego carries a non-zero heading (0.6 rad) distinct from its +x motion, so the
# round-trip must preserve heading and world-frame velocity independently.
EGO_HEADING = 0.6
EGO_VEL = (5.0, 0.0)


def _make_sd() -> SD:
    sd = SD()
    sd[SD.VERSION] = "1.0"
    sd[SD.ID] = "roundtrip_0"
    sd[SD.LENGTH] = T
    sd[SD.TRACKS] = {
        EGO_ID: _track("VEHICLE", EGO_ID, (0.0, 0.0), EGO_VEL, EGO_HEADING),
        "ped_1": _track("PEDESTRIAN", "ped_1", (10.0, 3.0), (0.0, 1.0), heading=1.5),
    }
    sd[SD.DYNAMIC_MAP_STATES] = {}
    sd[SD.MAP_FEATURES] = {
        "lane_0": {SD.TYPE: "LANE_SURFACE_STREET", SD.POLYLINE: _polyline(0.0)},
        "lane_1": {SD.TYPE: "LANE_SURFACE_STREET", SD.POLYLINE: _polyline(4.0)},
        # A non-lane feature: lanes/lines no longer collide on read, so it survives.
        "line_0": {SD.TYPE: "ROAD_LINE_SOLID_SINGLE_WHITE", SD.POLYLINE: _polyline(2.0)},
    }
    sd[SD.METADATA] = {SD.SDC_ID: EGO_ID, "scenario_id": "roundtrip_0", "dataset": "navsafe_pg"}
    return SD.update_summaries(sd)


def _lane_bbox(sd: SD) -> np.ndarray:
    """[xmin, ymin, xmax, ymax] over all map_feature polylines."""
    pts = [np.asarray(f[SD.POLYLINE])[:, :2] for f in sd[SD.MAP_FEATURES].values()
           if SD.POLYLINE in f and len(f[SD.POLYLINE])]
    allpts = np.concatenate(pts, axis=0)
    return np.array([allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max()])


# ─────────────────────────────── the round trip ───────────────────────────────

def test_scenario_description_arrow_round_trip(tmp_path) -> None:
    sd_in = _make_sd()
    dataset_root = tmp_path / "dataset"
    log_dir = dataset_root / "roundtrip_0"

    write_scenario_description_arrow(sd_in, log_dir)

    # Read back through the runtime path PG/text2sim → corpus → training takes.
    scenes = enumerate_scenes(dataset_root)
    assert len(scenes) == 1, "writer must produce exactly one rediscoverable log"
    scenario = scenario_from_py123d_scene(
        scenes[0],
        Py123DAdapterConfig(data_root=str(dataset_root), require_map=True),
    )
    sd_out = py123d_to_scenario_description(scenario)

    # Frame count preserved.
    assert sd_out[SD.LENGTH] == T

    # Tracks: one ego + one non-ego, types preserved.
    tracks_out = sd_out[SD.TRACKS]
    sdc_id = sd_out[SD.METADATA][SD.SDC_ID]
    assert sdc_id in tracks_out
    non_ego = {tid: tr for tid, tr in tracks_out.items() if tid != sdc_id}
    assert len(non_ego) == 1
    assert tracks_out[sdc_id][SD.TYPE] == "VEHICLE"
    ped_out = next(iter(non_ego.values()))
    assert ped_out[SD.TYPE] == "PEDESTRIAN"

    ego_out = tracks_out[sdc_id][SD.STATE]
    ego_in = sd_in[SD.TRACKS][EGO_ID][SD.STATE]

    # Ego trajectory: float32+SE3 round-trip is accurate to ~1e-3, so a tight
    # tolerance catches offset/unit/coordinate-shift regressions.
    np.testing.assert_allclose(ego_out[SD.POSITION][:, :2], ego_in[SD.POSITION][:, :2], atol=1e-2)
    # Heading preserved independently of motion direction.
    np.testing.assert_allclose(ego_out[SD.HEADING], ego_in[SD.HEADING], atol=1e-2)
    # World-frame velocity preserved (would fail if ego velocity were body-framed).
    np.testing.assert_allclose(ego_out["velocity"], ego_in["velocity"], atol=1e-2)

    # Non-ego trajectory preserved too.
    np.testing.assert_allclose(
        ped_out[SD.STATE][SD.POSITION][:, :2],
        sd_in[SD.TRACKS]["ped_1"][SD.STATE][SD.POSITION][:, :2],
        atol=1e-2,
    )

    # Map: lanes round-trip (count + geometry bbox), and the non-lane line
    # survives (cross-layer ids no longer collide on read).
    in_lanes = [f for f in sd_in[SD.MAP_FEATURES].values() if f[SD.TYPE] == "LANE_SURFACE_STREET"]
    out_lanes = [f for f in sd_out[SD.MAP_FEATURES].values() if str(f[SD.TYPE]).startswith("LANE")]
    assert len(out_lanes) == len(in_lanes)
    assert any(not str(f[SD.TYPE]).startswith("LANE") for f in sd_out[SD.MAP_FEATURES].values())
    np.testing.assert_allclose(_lane_bbox(sd_out), _lane_bbox(sd_in), atol=0.1)


def test_scenic_second_stamps_round_trip_as_microseconds(tmp_path) -> None:
    """A scenic-family SD (relative SECOND stamps in ``ts``) must round-trip.

    The writer used to misread the per-frame seconds array as microseconds,
    collapsing every stamp to 0,0,…,1,1 — the reloaded log then had a
    stationary sub-frame clock. Timestamps must come back as [i * dt] in µs.
    """
    sd_in = _make_sd()
    # Relative timestamp convention: relative second stamps, float32.
    sd_in[SD.METADATA][SD.TIMESTEP] = np.array([i * DT_S for i in range(T)], dtype=np.float32)
    dataset_root = tmp_path / "dataset"
    write_scenario_description_arrow(sd_in, dataset_root / "scenic_ts_0")

    scenes = enumerate_scenes(dataset_root)
    assert len(scenes) == 1
    scenario = scenario_from_py123d_scene(
        scenes[0], Py123DAdapterConfig(data_root=str(dataset_root), require_map=True)
    )
    stamps = list(scenario.timestamps_us)
    assert len(stamps) == T
    expected = [round(i * DT_S * 1_000_000) for i in range(T)]
    assert max(abs(a - b) for a, b in zip(stamps, expected)) <= 10  # float32 jitter only
