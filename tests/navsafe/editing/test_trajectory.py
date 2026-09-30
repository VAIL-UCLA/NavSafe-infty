# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Placement solver and the five trajectory templates."""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.placement import (
    HostProbe,
    PlacementError,
    Polyline,
    intersect_polylines,
    solve_start_arc_for_conflict,
)
from navsafe.benchmark.editing.trajectory import (
    BakeError,
    bake_actor_state,
    build_motion,
)
from navsafe.benchmark.editing.trajectory.templates import TemplateError

T = 44
DT = 0.1
EGO_Z = 55.0
Z_TO_GROUND = 1.4


def _east_line(length: float = 200.0, y: float = 0.0, z: float = EGO_Z) -> Polyline:
    xs = np.arange(0.0, length + 1e-9, 2.0)
    pts = np.stack([xs, np.full_like(xs, y), np.full_like(xs, z)], axis=1)
    return Polyline.from_points(pts, name="east")


def _host_sd() -> dict:
    """Straight ego route east, a parallel lane, and a cross street north."""
    pos = np.zeros((T, 3), np.float64)
    pos[:, 0] = np.arange(T) * 2.0
    pos[:, 2] = EGO_Z
    lane_xs = np.arange(-20.0, 200.0, 4.0)
    mainline = np.stack(
        [lane_xs, np.full_like(lane_xs, 3.6), np.full_like(lane_xs, EGO_Z - Z_TO_GROUND)], axis=1
    )
    cross_ys = np.arange(-40.0, 40.0, 4.0)
    cross = np.stack(
        [np.full_like(cross_ys, 60.0), cross_ys, np.full_like(cross_ys, EGO_Z - Z_TO_GROUND)],
        axis=1,
    )
    ego_lane = np.stack(
        [lane_xs, np.zeros_like(lane_xs), np.full_like(lane_xs, EGO_Z - Z_TO_GROUND)], axis=1
    )
    return {
        "metadata": {
            "sdc_id": "ego",
            "ts": (np.arange(T) * DT * 1e6).astype(np.float64),
            "coordinate": "local_frame0",
            "route_lane_ids": ["ego_lane"],
        },
        "tracks": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": pos,
                    "heading": np.zeros(T),
                    "velocity": np.zeros((T, 2)),
                    "valid": np.ones(T, bool),
                },
                "metadata": {"object_id": "ego"},
            }
        },
        "map_features": {
            "ego_lane": {
                "type": "LANE_SURFACE_STREET",
                "polyline": ego_lane,
                "left_boundaries": np.stack(
                    [lane_xs, np.full_like(lane_xs, 1.8)], axis=1
                ),
                "right_boundaries": np.stack(
                    [lane_xs, np.full_like(lane_xs, -1.8)], axis=1
                ),
            },
            "mainline": {"type": "LANE_SURFACE_STREET", "polyline": mainline},
            "cross_street": {"type": "LANE_SURFACE_STREET", "polyline": cross},
            "road_edge_0": {
                "type": "ROAD_EDGE_BOUNDARY",
                "polyline": np.stack([lane_xs, np.full_like(lane_xs, 7.0)], axis=1),
            },
            "road_edge_1": {
                "type": "ROAD_EDGE_BOUNDARY",
                "polyline": np.stack([lane_xs, np.full_like(lane_xs, -5.5)], axis=1),
            },
        },
    }


# ---------------------------------------------------------------------------
# Route-frame geometry
# ---------------------------------------------------------------------------


class TestPolyline:
    def test_sample_and_offset_signs(self):
        line = _east_line()
        x, y, _, heading = line.sample([10.0])
        assert (float(x[0]), float(y[0])) == pytest.approx((10.0, 0.0))
        assert float(heading[0]) == pytest.approx(0.0)
        # +left of an eastbound reference is +y.
        x, y, _, _ = line.offset([10.0], [3.0])
        assert float(y[0]) == pytest.approx(3.0)

    def test_project_recovers_arc_and_lateral(self):
        line = _east_line()
        s, lateral = line.project([37.0, -2.5])
        assert s == pytest.approx(37.0)
        assert lateral == pytest.approx(-2.5)

    def test_sampling_extrapolates_rather_than_clamping(self):
        line = _east_line(length=50.0)
        x, _, _, _ = line.sample([60.0])
        assert float(x[0]) == pytest.approx(60.0)
        assert not line.covers([60.0])

    def test_road_z_respects_the_pose_lift(self):
        pose_line = Polyline.from_points(
            _east_line().xy, z=np.full(len(_east_line().xy), EGO_Z), z_is_road_surface=False
        )
        assert float(pose_line.road_z([10.0], ego_z_to_ground_m=Z_TO_GROUND)[0]) == pytest.approx(
            EGO_Z - Z_TO_GROUND
        )

    def test_intersect_finds_the_conflict_point(self):
        east = _east_line()
        north = Polyline.from_points([[60.0, -30.0], [60.0, 30.0]])
        hit = intersect_polylines(east, north)
        assert hit is not None
        s_east, s_north, xy = hit
        assert s_east == pytest.approx(60.0)
        assert s_north == pytest.approx(30.0)
        assert list(np.round(xy, 3)) == [60.0, 0.0]

    def test_conflict_solve_back_solves_the_spawn_arc(self):
        # 4 m/s, conflict at arc 30 on frame 25 -> starts 10 m back.
        assert solve_start_arc_for_conflict(
            _east_line(), conflict_arc=30.0, speed=4.0, conflict_frame=25, dt_s=DT
        ) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Host probe
# ---------------------------------------------------------------------------


class TestHostProbe:
    def test_frames_come_from_the_host(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        frames = probe.frames_dict(after_frame=8)
        assert frames["T"] == T
        assert frames["dt_s"] == pytest.approx(DT)
        assert frames["after_frame"] == 8
        assert len(frames["timestamps_us"]) == T

    def test_after_frame_outside_the_episode_is_refused(self):
        probe = HostProbe(_host_sd())
        with pytest.raises(PlacementError, match="outside this host's episode"):
            probe.frames_dict(after_frame=T)

    def test_anchor_is_the_hand_off_on_any_reference(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        # On the ego route the anchor is just the arc at that frame…
        assert probe.anchor_arc(probe.ego_route, 8) == pytest.approx(16.0)
        # …and on a parallel lane it is the same station, projected across.
        assert probe.anchor_arc(probe.lane("mainline"), 8) == pytest.approx(36.0)

    def test_reference_kinds_resolve(self):
        probe = HostProbe(_host_sd())
        assert probe.resolve_reference("ego_route").name == "ego_route"
        assert probe.resolve_reference("lane:cross_street").name == "lane:cross_street"
        assert probe.resolve_reference([[0.0, 0.0], [10.0, 0.0]]).total == pytest.approx(10.0)
        with pytest.raises(PlacementError, match="unknown reference"):
            probe.resolve_reference("kerb")
        with pytest.raises(PlacementError, match="no lane"):
            probe.resolve_reference("lane:nope")

    def test_cross_section_measures_this_host(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        section = probe.cross_section(probe.ego_route, 40.0)
        assert section.lane_id == "ego_lane"
        assert section.lane_width_m == pytest.approx(3.6)
        assert section.left_boundary_m == pytest.approx(1.8)
        assert section.right_boundary_m == pytest.approx(-1.8)
        # −5.0 m on THIS host is past the right road edge, which is the whole
        # point of measuring rather than assuming.
        assert section.right_road_edge_m == pytest.approx(-5.5)
        assert section.left_road_edge_m == pytest.approx(7.0)
        assert "lane=ego_lane" in section.describe()


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _motion(template: str, **params):
    return build_motion(template, T=T, dt_s=DT, anchor_arc=16.0, params=params)


class TestTemplates:
    def test_static_holds_one_pose(self):
        motion = _motion("static", arc=20.0, lateral=-2.0, yaw_offset_deg=90.0)
        assert np.all(motion.s == 36.0)
        assert np.all(motion.lateral == -2.0)
        assert np.all(motion.s_dot == 0.0)
        assert np.allclose(motion.heading_offset, np.pi / 2)

    def test_dynamic_positive_speed_travels_with_the_reference(self):
        motion = _motion("dynamic", arc=10.0, speed=8.0)
        assert motion.s[10] == pytest.approx(26.0 + 8.0)
        assert np.allclose(motion.heading_offset, 0.0)

    def test_signed_speed_turns_the_actor_around(self):
        # The one convention that makes an oncoming actor expressible at all.
        motion = _motion("dynamic", arc=52.0, speed=-8.0)
        assert motion.s[10] < motion.s[0]
        assert np.allclose(np.abs(motion.heading_offset), np.pi)

    def test_dart_out_matches_the_worked_example(self):
        # The dog from the design plan: 10 m sweep at 5 m/s over dt=0.1
        # -> 20 frames, centred on conflict_frame 22 -> crossing starts at 12.
        motion = _motion(
            "dart_out",
            arc=25.0,
            start_lateral=-5.0,
            end_lateral=5.0,
            speed=5.0,
            conflict_frame=22,
        )
        assert motion.meta["duration_frames"] == 20
        assert motion.meta["crossing_start_frame"] == 12
        assert motion.lateral[11] == pytest.approx(-5.0)  # waiting at the verge
        assert motion.lateral[22] == pytest.approx(0.0)  # the lane centre, on cue
        assert motion.lateral[32] == pytest.approx(5.0)  # off the far side
        assert motion.lateral[43] == pytest.approx(5.0)  # and it stays there
        assert np.all(motion.s == 41.0)  # holds one arc position throughout

    def test_dart_out_holds_its_heading_through_the_still_frames(self):
        motion = _motion(
            "dart_out", arc=25.0, start_lateral=-5.0, end_lateral=5.0, speed=5.0, conflict_frame=22
        )
        # No rotating on the spot when it starts and stops.
        assert np.allclose(motion.heading_offset, np.pi / 2)

    def test_dart_out_needs_a_positive_crossing_speed(self):
        with pytest.raises(TemplateError, match="crossing speed"):
            _motion(
                "dart_out", arc=25.0, start_lateral=-5.0, end_lateral=5.0, speed=-5.0,
                conflict_frame=22,
            )

    def test_dart_out_needs_somewhere_to_go(self):
        with pytest.raises(TemplateError, match="nothing crosses"):
            _motion(
                "dart_out", arc=25.0, start_lateral=2.0, end_lateral=2.0, speed=5.0,
                conflict_frame=22,
            )

    def test_unknown_template_is_refused(self):
        with pytest.raises(TemplateError, match="unknown template"):
            _motion("swerve", arc=10.0)


class TestBake:
    def test_oncoming_actor_closes_head_on(self):
        line = _east_line()
        motion = _motion("dynamic", arc=52.0, speed=-8.0)
        baked = bake_actor_state(line, motion, ego_z_to_ground_m=Z_TO_GROUND, keep_appearance=True)
        position = np.asarray(baked.state["position"])
        velocity = np.asarray(baked.state["velocity"])
        assert position[0, 0] == pytest.approx(68.0)
        assert position[-1, 0] < position[0, 0]  # coming towards the ego
        assert velocity[0, 0] == pytest.approx(-8.0)
        assert float(np.asarray(baked.state["heading"])[0]) == pytest.approx(np.pi, abs=1e-4)

    def test_keep_appearance_decides_the_z_convention(self):
        line = _east_line()  # z is the road surface here
        motion = _motion("static", arc=20.0)
        kept = bake_actor_state(line, motion, ego_z_to_ground_m=Z_TO_GROUND, keep_appearance=True)
        swapped = bake_actor_state(
            line, motion, ego_z_to_ground_m=Z_TO_GROUND, keep_appearance=False
        )
        assert float(np.asarray(kept.state["position"])[0, 2]) == pytest.approx(EGO_Z + Z_TO_GROUND)
        assert float(np.asarray(swapped.state["position"])[0, 2]) == pytest.approx(EGO_Z)

    def test_dart_out_bakes_a_perpendicular_crossing(self):
        line = _east_line()
        motion = _motion(
            "dart_out", arc=25.0, start_lateral=-5.0, end_lateral=5.0, speed=5.0, conflict_frame=22
        )
        baked = bake_actor_state(line, motion)
        position = np.asarray(baked.state["position"])
        velocity = np.asarray(baked.state["velocity"])
        assert position[11, 1] == pytest.approx(-5.0)
        assert position[22, 1] == pytest.approx(0.0, abs=1e-4)
        assert position[32, 1] == pytest.approx(5.0)
        assert np.all(np.abs(position[:, 0] - 41.0) < 1e-4)  # never advances
        assert velocity[20, 1] == pytest.approx(5.0)
        assert velocity[5, 1] == pytest.approx(0.0)  # still waiting

    def test_a_spawn_arc_off_the_reference_is_refused(self):
        line = _east_line(length=50.0)
        motion = _motion("static", arc=100.0)
        with pytest.raises(BakeError, match="spawn arc"):
            bake_actor_state(line, motion)

    def test_running_off_the_reference_is_refused(self):
        line = _east_line(length=50.0)
        motion = _motion("dynamic", arc=20.0, speed=20.0)  # 86 m of travel
        with pytest.raises(BakeError, match="Shorten the episode's reach"):
            bake_actor_state(line, motion)

    def test_lane_reference_makes_the_actor_follow_the_side_road(self):
        probe = HostProbe(_host_sd(), ego_z_to_ground_m=Z_TO_GROUND)
        cross = probe.resolve_reference("lane:cross_street")
        motion = build_motion(
            "dynamic",
            T=T,
            dt_s=DT,
            anchor_arc=0.0,
            params={"arc": 5.0, "speed": 5.0},
        )
        baked = bake_actor_state(cross, motion, ego_z_to_ground_m=Z_TO_GROUND)
        position = np.asarray(baked.state["position"])
        # It runs north up the cross street, not east along the ego's path.
        assert np.allclose(position[:, 0], 60.0, atol=1e-3)
        assert position[-1, 1] > position[0, 1]
        assert float(np.asarray(baked.state["heading"])[0]) == pytest.approx(np.pi / 2, abs=1e-4)

    def test_diagnostics_give_the_reviewer_numbers(self):
        line = _east_line()
        baked = bake_actor_state(line, _motion("dynamic", arc=52.0, speed=-8.0))
        d = baked.diagnostics
        assert d["template"] == "dynamic"
        assert d["spawn_arc_m"] == pytest.approx(68.0)
        assert d["speed_range_mps"] == [pytest.approx(8.0), pytest.approx(8.0)]
        assert d["reference"] == "east"
