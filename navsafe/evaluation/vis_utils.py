"""
navsafe.evaluation.vis_utils

Visualization utilities for the UrbanSim evaluation pipeline.

Provides:
  - BEV (bird's-eye-view) rendering with map, agents, and trajectory overlay
  - Front-camera trajectory projection and ribbon overlay
  - Fallback front-camera rendering from scenario data (when real images unavailable)
  - GIF / MP4 generation from per-frame images
  - Combined side-by-side visualization
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from navsafe.core.ego_dims import EGO_LENGTH_M, EGO_WIDTH_M

logger = logging.getLogger(__name__)


def _no_overlay() -> bool:
    """``NAVSAFE_NO_OVERLAY``: skip every camera annotation.

    Same falsey set as ``eval_artifacts._no_overlay``, and deliberately not a
    bare ``os.environ.get(...)`` truth test: the string ``"0"`` is truthy in
    Python, so an explicit ``NAVSAFE_NO_OVERLAY=0`` — the way a launcher
    pins the default — used to switch the overlay *off*.
    """
    return os.environ.get("NAVSAFE_NO_OVERLAY", "").lower() not in {
        "", "0", "false", "no", "off"}


def _no_cam_map_lines() -> bool:
    """``NAVSAFE_NO_CAM_MAP_LINES``: drop the projected map polylines from the
    front camera only.

    Distinct from ``NAVSAFE_NO_OVERLAY`` (``eval_artifacts``), which writes the
    camera frame with no annotations at all. This one keeps every trajectory
    overlay and removes only the grey road geometry, which a photoreal render
    already contains. The top-down is unaffected either way — there the map
    lines *are* the picture.
    """
    return os.environ.get("NAVSAFE_NO_CAM_MAP_LINES", "").lower() not in {
        "", "0", "false", "no", "off"}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BEV_SIZE = 800          # pixels
BEV_RANGE_M = 60.0     # meters visible in each direction from ego
BEV_PPM = BEV_SIZE / (2 * BEV_RANGE_M)  # pixels per meter
# World-space cull radius for map features: anything whose bbox lies entirely
# outside ego ± this many metres cannot land inside the BEV window (60 m
# visible + ~7.5 m for the existing ±50 px keep-margin), so it is skipped
# *before* the per-point world→pixel conversion. On whole-city maps (e.g. the
# ~4.5k-lane Las Vegas nuPlan map) this early cull is what keeps render_bev
# from re-projecting every far-away lane each frame (was ~87% of the eval loop).
_BEV_CULL_R = BEV_RANGE_M + 15.0  # metres

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colours (BGR for OpenCV)
_CLR_EGO = (51, 51, 255)        # red
# Route dots were also red — indistinguishable from the ego, which is the one
# marker a reader (or a VLM) must localise first. Magenta is unused elsewhere.
_CLR_ROUTE = (255, 0, 200)      # magenta
_CLR_VEHICLE = (255, 136, 68)   # blue
_CLR_PED = (68, 204, 68)        # green
_CLR_CYCLIST = (0, 204, 255)    # yellow
_CLR_OTHER = (204, 68, 204)     # purple
_CLR_LANE = (160, 160, 160)     # light grey lane center-lines (visible on road surface)
_CLR_MARKING_W = (200, 200, 200)
_CLR_MARKING_Y = (0, 200, 240)
_CLR_BG = (255, 255, 255)       # white background
_CLR_ROAD_SURFACE = (200, 200, 200)  # light grey road surface fill
_CLR_TRAJ = (0, 136, 255)       # orange

# Driving command constants and colors (matching BridgeSim convention)
CMD_LEFT = 0
CMD_RIGHT = 1
CMD_STRAIGHT = 2
CMD_LANEFOLLOW = 3
CMD_VOID = -1

_CMD_NAMES = {
    CMD_LEFT: "LEFT",
    CMD_RIGHT: "RIGHT",
    CMD_STRAIGHT: "STRAIGHT",
    CMD_LANEFOLLOW: "LANEFOLLOW",
    CMD_VOID: "VOID",
}
_CMD_COLORS = {
    CMD_LEFT: (0, 165, 255),       # orange
    CMD_RIGHT: (255, 0, 255),      # magenta
    CMD_STRAIGHT: (0, 255, 255),   # yellow
    CMD_LANEFOLLOW: (0, 255, 0),   # green
    CMD_VOID: (128, 128, 128),     # grey
}


def derive_driving_command(
    traj_ego: Optional[np.ndarray],
    lateral_thresh: float = 2.0,
    curvature_thresh: float = 0.02,
) -> int:
    """Derive a driving command from ego-frame trajectory curvature.

    Args:
        traj_ego: (N, 2) ego-frame trajectory [lateral, forward].
        lateral_thresh: lateral displacement (m) to classify as turn.
        curvature_thresh: mean curvature to classify as turn vs lane-follow.

    Returns:
        Command int: 0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANEFOLLOW, -1=VOID.
    """
    if traj_ego is None or len(traj_ego) < 3:
        return CMD_VOID

    # Net lateral displacement at end of trajectory
    net_lateral = float(traj_ego[-1, 0])  # col0 = lateral (left+)

    # Compute mean curvature from trajectory
    dx = np.diff(traj_ego[:, 0])
    dy = np.diff(traj_ego[:, 1])
    if len(dx) < 2:
        return CMD_LANEFOLLOW

    ddx = np.diff(dx)
    ddy = np.diff(dy)
    ds = np.sqrt(dx[:-1] ** 2 + dy[:-1] ** 2) + 1e-6
    curvature = np.abs(ddx * dy[:-1] - ddy * dx[:-1]) / (ds ** 3)
    mean_curv = float(np.mean(curvature))

    if abs(net_lateral) > lateral_thresh or mean_curv > curvature_thresh:
        return CMD_LEFT if net_lateral > 0 else CMD_RIGHT
    elif mean_curv < curvature_thresh * 0.3:
        return CMD_STRAIGHT
    else:
        return CMD_LANEFOLLOW


def _agent_color_bgr(agent_type: str) -> Tuple[int, int, int]:
    from navsafe.scenario.type import MetaDriveType
    if agent_type == MetaDriveType.VEHICLE:
        return _CLR_VEHICLE
    if agent_type == MetaDriveType.PEDESTRIAN:
        return _CLR_PED
    if agent_type == MetaDriveType.CYCLIST:
        return _CLR_CYCLIST
    return _CLR_OTHER


# ---------------------------------------------------------------------------
# BEV rendering (with map + agents from scenario data)
# ---------------------------------------------------------------------------

def _bev_ego_up() -> bool:
    """NAVSAFE_BEV_EGO_UP=1: rotate the BEV so the ego always points UP.

    Off by default. The BEV is world-axis-aligned ("north up"), so the ego's
    arrow swings around as the scenario turns; for a figure that compares
    frames, or several models on the same frame, a fixed ego heading is easier
    to read. This only changes the VIEW: every world coordinate still goes
    through one transform, so the map, the boxes and the trajectories rotate
    together.
    """
    return os.environ.get("NAVSAFE_BEV_EGO_UP", "").strip() in ("1", "true", "TRUE", "yes")


def _bev_rot(ego_heading: float) -> Optional[np.ndarray]:
    """2x2 world->view rotation putting ego heading along +y, or None."""
    if not _bev_ego_up():
        return None
    # heading is measured from +x CCW; we want it to land on +y (screen up),
    # so rotate the world by (pi/2 - heading).
    a = math.pi / 2.0 - float(ego_heading)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa], [sa, ca]])


def render_bev(
    ego_position: np.ndarray,
    ego_heading: float,
    planned_traj_world: Optional[np.ndarray] = None,
    vehicle_states: Optional[List[Dict]] = None,
    frame_id: int = 0,
    model_name: str = "",
    speed_kmh: float = 0.0,
    collision: bool = False,
    scenario_data: Optional[Dict] = None,
    driving_command: int = CMD_VOID,
    sim_dt: float = 0.1,
    route_waypoints: Optional[List] = None,
    target_waypoint: Optional[tuple] = None,
    agent_states: Optional[List[Dict]] = None,
    trajectory_dt_s: Optional[float] = None,
    first_sample_time_s: float = 0.0,
) -> np.ndarray:
    """Render a BEV top-down image centered on the ego vehicle.

    When *scenario_data* is provided (the full scenario dict with 'tracks' and
    'map_features'), the BEV includes lane polylines, road markings, and other
    agent bounding boxes — producing a rich, informative visualization.

    Args:
        driving_command: Driving command int (0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANEFOLLOW).
        sim_dt: Simulation timestep; legacy default for trajectory spacing.
        trajectory_dt_s: Actual spacing of the supplied trajectory samples.
        first_sample_time_s: Time of the first supplied sample relative to now.
        route_waypoints: list of (position_2d, command, frame_idx) — static GT route.
        target_waypoint: (position_2d, command, frame_idx) — current target waypoint.
        agent_states: the env's live agent list — the same one the collision
            check consumes. Preferred over ``scenario_data['tracks']``, which
            holds the *logged* pose: under semi-reactive traffic the IDM moves
            agents off their log, so drawing tracks shows a car touching the
            ego frames before (or after) the simulated boxes actually overlap.
            Falls back to the tracks when not supplied.

    Returns:
        (BEV_SIZE, BEV_SIZE, 3) uint8 BGR image.
    """
    from navsafe.scenario.scenario_description import ScenarioDescription as SD
    from navsafe.scenario.type import MetaDriveType

    img = np.full((BEV_SIZE, BEV_SIZE, 3), _CLR_BG, dtype=np.uint8)
    cx, cy = BEV_SIZE // 2, BEV_SIZE // 2

    _rot = _bev_rot(ego_heading)

    def world_to_px(wx: float, wy: float) -> Tuple[int, int]:
        dx = wx - ego_position[0]
        dy = wy - ego_position[1]
        if _rot is not None:
            dx, dy = float(_rot[0, 0] * dx + _rot[0, 1] * dy), \
                     float(_rot[1, 0] * dx + _rot[1, 1] * dy)
        px = int(cx + dx * BEV_PPM)
        py = int(cy - dy * BEV_PPM)  # Y inverted
        return px, py

    def in_bounds(px: int, py: int, margin: int = 0) -> bool:
        return -margin <= px < BEV_SIZE + margin and -margin <= py < BEV_SIZE + margin

    # ── Map polylines (lanes + road markings) ───────────────────────────
    if scenario_data is not None:
        map_features = scenario_data.get(SD.MAP_FEATURES, {})

        # Pass 1: Draw filled lane polygons (road surface)
        for _fid, feat in map_features.items():
            feat_type = feat.get(SD.TYPE, "")
            if not MetaDriveType.is_lane(feat_type):
                continue
            polygon = feat.get(SD.POLYGON)
            if polygon is None or len(polygon) < 3:
                continue
            pts_world = np.asarray(polygon)[:, :2]
            # Cheap world-space bbox cull before the per-point pixel conversion
            # (skips far-away features on city-scale maps without touching the
            # Python world_to_px loop for each of their points).
            if (pts_world[:, 0].max() < ego_position[0] - _BEV_CULL_R or
                    pts_world[:, 0].min() > ego_position[0] + _BEV_CULL_R or
                    pts_world[:, 1].max() < ego_position[1] - _BEV_CULL_R or
                    pts_world[:, 1].min() > ego_position[1] + _BEV_CULL_R):
                continue
            pts_px = np.array([world_to_px(float(p[0]), float(p[1])) for p in pts_world], dtype=np.int32)
            if (pts_px[:, 0].max() < -50 or pts_px[:, 0].min() > BEV_SIZE + 50 or
                    pts_px[:, 1].max() < -50 or pts_px[:, 1].min() > BEV_SIZE + 50):
                continue
            cv2.fillPoly(img, [pts_px], _CLR_ROAD_SURFACE)

        # Pass 2: Draw lane center-lines and road markings on top
        for _fid, feat in map_features.items():
            feat_type = feat.get(SD.TYPE, "")
            poly = feat.get(SD.POLYLINE)
            if poly is None or len(poly) < 2:
                continue
            pts_world = np.asarray(poly)[:, :2]

            # Cheap world-space bbox cull before the per-point pixel conversion.
            if (pts_world[:, 0].max() < ego_position[0] - _BEV_CULL_R or
                    pts_world[:, 0].min() > ego_position[0] + _BEV_CULL_R or
                    pts_world[:, 1].max() < ego_position[1] - _BEV_CULL_R or
                    pts_world[:, 1].min() > ego_position[1] + _BEV_CULL_R):
                continue

            # Convert to pixel coords
            pts_px = np.array([world_to_px(float(p[0]), float(p[1])) for p in pts_world], dtype=np.int32)

            # Skip if entirely out of view
            if (pts_px[:, 0].max() < -50 or pts_px[:, 0].min() > BEV_SIZE + 50 or
                    pts_px[:, 1].max() < -50 or pts_px[:, 1].min() > BEV_SIZE + 50):
                continue

            if MetaDriveType.is_lane(feat_type):
                cv2.polylines(img, [pts_px], False, _CLR_LANE, 1, cv2.LINE_AA)
            elif MetaDriveType.is_road_line(feat_type):
                clr = _CLR_MARKING_Y if MetaDriveType.is_yellow_line(feat_type) else _CLR_MARKING_W
                thickness = 1
                if MetaDriveType.is_broken_line(feat_type):
                    # Draw dashed: every other segment
                    for i in range(0, len(pts_px) - 1, 2):
                        cv2.line(img, tuple(pts_px[i]), tuple(pts_px[min(i + 1, len(pts_px) - 1)]),
                                 clr, thickness, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [pts_px], False, clr, thickness, cv2.LINE_AA)
            elif MetaDriveType.is_crosswalk(feat_type):
                # Draw crosswalks as hatched polygons
                polygon = feat.get(SD.POLYGON)
                if polygon is not None and len(polygon) >= 3:
                    poly_pts = np.array([world_to_px(float(p[0]), float(p[1])) for p in np.asarray(polygon)[:, :2]], dtype=np.int32)
                    cv2.polylines(img, [poly_pts], True, (180, 180, 180), 1, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [pts_px], False, (180, 180, 180), 1, cv2.LINE_AA)
            elif MetaDriveType.is_road_boundary_line(feat_type):
                cv2.polylines(img, [pts_px], False, (120, 120, 120), 1, cv2.LINE_AA)
            else:
                # Other boundaries, etc.
                cv2.polylines(img, [pts_px], False, (80, 80, 80), 1, cv2.LINE_AA)

    # ── Other agents (boxes) ────────────────────────────────────────────
    # One drawing loop over (position, heading, length, width, type), fed
    # either by the env's live agent list or, without one, by the logged
    # tracks. Anything drawn here must be what the collision check sees.
    ego_id = None
    drawables: List[tuple] = []
    if agent_states is not None:
        for agent in agent_states:
            if agent.get("is_ego") or not agent.get("valid", True):
                continue
            apos = np.asarray(agent["position"], dtype=float)[:2]
            drawables.append((apos, float(agent.get("heading", 0.0)),
                              float(agent.get("length", 4.5)),
                              float(agent.get("width", 1.8)),
                              agent.get("type", "OTHER")))
    elif scenario_data is not None:
        tracks = scenario_data.get(SD.TRACKS, {})
        metadata = scenario_data.get(SD.METADATA, {})
        ego_id = str(metadata.get(SD.SDC_ID, ""))

        for agent_id, track in tracks.items():
            if agent_id == ego_id:
                continue
            st = track.get(SD.STATE, {})
            positions = st.get("position")
            if positions is None or frame_id >= len(positions):
                continue
            valid_arr = st.get("valid")
            if valid_arr is not None and frame_id < len(valid_arr) and not valid_arr[frame_id]:
                continue

            agent_type = track.get(SD.TYPE, "OTHER")
            # Dimensions
            if "length" in st and frame_id < len(st["length"]):
                al = float(st["length"][frame_id])
                aw = float(st["width"][frame_id]) if "width" in st and frame_id < len(st["width"]) else 1.8
            else:
                al = 4.5 if agent_type == MetaDriveType.VEHICLE else 0.6
                aw = 1.8 if agent_type == MetaDriveType.VEHICLE else 0.6
            drawables.append((np.asarray(positions[frame_id])[:2],
                              float(st["heading"][frame_id]), al, aw, agent_type))

    for apos, ahead, al, aw, agent_type in drawables:
        # Compute rotated box corners in pixel space
        corners = _box_corners_px(float(apos[0]), float(apos[1]), ahead, al, aw,
                                  ego_position, cx, cy, BEV_PPM, _rot)
        if corners is None:
            continue

        color = _agent_color_bgr(agent_type)
        cv2.fillPoly(img, [corners], color)
        cv2.polylines(img, [corners], True, (0, 0, 0), 1, cv2.LINE_AA)

        # Heading arrow
        arrow_len = al * 0.7
        ax_w = float(apos[0]) + arrow_len * np.cos(ahead)
        ay_w = float(apos[1]) + arrow_len * np.sin(ahead)
        a_start = world_to_px(float(apos[0]), float(apos[1]))
        a_end = world_to_px(ax_w, ay_w)
        cv2.arrowedLine(img, a_start, a_end, (0, 0, 0), 1, tipLength=0.3)

    # ── Past ego trail ──────────────────────────────────────────────────
    if vehicle_states:
        n = len(vehicle_states)
        for i, vs in enumerate(vehicle_states):
            pos = np.asarray(vs.get("position", [0, 0, 0]), dtype=float)
            px, py = world_to_px(float(pos[0]), float(pos[1]))
            if in_bounds(px, py):
                alpha = 0.3 + 0.7 * (i / max(n - 1, 1))
                c = int(180 * alpha)
                cv2.circle(img, (px, py), 2, (c, c, c), -1)

    # ── Route waypoints (red dots — static GT route) ──────────────────
    if route_waypoints is not None:
        for i, wp_data in enumerate(route_waypoints):
            wp_pos = wp_data[0]  # (2,) position
            px, py = world_to_px(float(wp_pos[0]), float(wp_pos[1]))
            if in_bounds(px, py, margin=20):
                radius = 5 if i % 10 == 0 else 2
                cv2.circle(img, (px, py), radius, _CLR_ROUTE, -1)

    # ── Target waypoint (green dot — current navigation target) ─────
    if target_waypoint is not None:
        tw_pos = target_waypoint[0]
        px, py = world_to_px(float(tw_pos[0]), float(tw_pos[1]))
        if in_bounds(px, py, margin=20):
            cv2.circle(img, (px, py), 8, (0, 255, 0), -1)  # green

    # ── Planned trajectory (cyan → yellow gradient) ─────────────────────
    # NAVSAFE_NO_PLAN_OVERLAY drops ONLY this layer -- the model's predicted
    # path, with its per-timestamp labels. The magenta GT route dots and the
    # green target marker above are a different thing and stay: a figure often
    # wants "where the scenario goes" without "what the policy proposed",
    # because the two read as one line otherwise.
    if (planned_traj_world is not None and len(planned_traj_world) > 0
            and not _no_plan_overlay()):
        pts = []
        for wp in planned_traj_world:
            px, py = world_to_px(float(wp[0]), float(wp[1]))
            pts.append((px, py))

        for i, (px, py) in enumerate(pts):
            if not in_bounds(px, py, margin=20):
                continue
            t = i / max(len(pts) - 1, 1)
            color = (int(255 * (1 - t)), 255, int(255 * t))  # cyan → yellow
            cv2.circle(img, (px, py), 4, color, -1)

            # Per-timestamp label every 5 waypoints (0.5s at sim_dt=0.1)
            if i % 5 == 0:
                ts = first_sample_time_s + i * (
                    sim_dt if trajectory_dt_s is None else trajectory_dt_s)
                cv2.putText(img, f"{ts:.1f}s", (px + 5, py - 5),
                            _FONT, 0.3, (200, 200, 200), 1)

        if len(pts) > 1:
            for i in range(len(pts) - 1):
                t = i / max(len(pts) - 2, 1)
                color = (int(255 * (1 - t)), 255, int(255 * t))
                cv2.line(img, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

    # ── Ego box (TRUE footprint, same path as every other agent) ───────
    # Was a fixed 10 px triangle (~1.5 m long, ~1.1 m wide) regardless of
    # zoom — ~7x too small by area against the real 4.515 x 1.852 m body,
    # and the ONLY object in the frame not drawn to scale. A VLM shown this
    # BEV under-read ego half-width by ~0.39 m per side, which is ~39% of the
    # planner's entire +/-1.0 m lateral-offset budget, so "there is room to
    # squeeze past" judgements were systematically wrong.
    ego_corners = _box_corners_px(
        float(ego_position[0]), float(ego_position[1]), ego_heading,
        EGO_LENGTH_M, EGO_WIDTH_M, ego_position, cx, cy, BEV_PPM, _rot)
    if ego_corners is not None:
        cv2.fillPoly(img, [ego_corners], _CLR_EGO)
        cv2.polylines(img, [ego_corners], True, (0, 0, 0), 1, cv2.LINE_AA)
        # Heading tick so orientation stays readable at any zoom.
        nose = world_to_px(
            float(ego_position[0]) + EGO_LENGTH_M * 0.75 * np.cos(ego_heading),
            float(ego_position[1]) + EGO_LENGTH_M * 0.75 * np.sin(ego_heading))
        cv2.arrowedLine(img, (cx, cy), nose, (0, 0, 0), 1,
                        cv2.LINE_AA, tipLength=0.3)

    # ── HUD ────────────────────────────────────────────────────────────
    # NAVSAFE_NO_HUD=1 suppresses the whole overlay -- the translucent box, the
    # frame/model/speed/command readout, the legend and the collision banner --
    # leaving a clean top-down for figures. Set per-run, not a code change, so
    # scored evals keep the HUD they have always had.
    if _no_hud():
        return img

    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (240, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.putText(img, f"Frame: {frame_id}  t={frame_id * 0.1:.1f}s", (10, 22), _FONT, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"{model_name}  {speed_kmh:.0f} km/h", (10, 42), _FONT, 0.5, (255, 255, 255), 1)

    # Driving command (color-coded)
    cmd_name = _CMD_NAMES.get(driving_command, "VOID")
    cmd_color = _CMD_COLORS.get(driving_command, (128, 128, 128))
    cv2.putText(img, f"Cmd: {cmd_name}", (10, 62), _FONT, 0.5, cmd_color, 2)

    # Legend row 1
    cv2.circle(img, (15, 85), 4, _CLR_EGO, -1)
    cv2.putText(img, "Ego", (25, 89), _FONT, 0.35, (255, 255, 255), 1)
    cv2.circle(img, (60, 85), 4, (0, 255, 255), -1)
    cv2.putText(img, "Plan", (70, 89), _FONT, 0.35, (255, 255, 255), 1)
    cv2.circle(img, (110, 85), 4, _CLR_VEHICLE, -1)
    cv2.putText(img, "Vehicle", (120, 89), _FONT, 0.35, (255, 255, 255), 1)
    cv2.circle(img, (175, 85), 4, _CLR_PED, -1)
    cv2.putText(img, "Ped", (185, 89), _FONT, 0.35, (255, 255, 255), 1)
    # Legend row 2: route + target
    cv2.circle(img, (15, 105), 4, _CLR_ROUTE, -1)
    cv2.putText(img, "Route", (25, 109), _FONT, 0.35, (255, 255, 255), 1)
    cv2.circle(img, (80, 105), 6, (0, 255, 0), -1)
    cv2.putText(img, "Target", (92, 109), _FONT, 0.35, (255, 255, 255), 1)

    if collision:
        cv2.rectangle(img, (0, BEV_SIZE - 30), (BEV_SIZE, BEV_SIZE), (0, 0, 180), -1)
        cv2.putText(img, "COLLISION", (BEV_SIZE // 2 - 60, BEV_SIZE - 8),
                    _FONT, 0.7, (255, 255, 255), 2)

    return img


def _box_corners_px(
    wx: float, wy: float, heading: float, length: float, width: float,
    ego_pos: np.ndarray, cx: int, cy: int, ppm: float,
    view_rot: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Compute rotated box corners in pixel space. Returns (4,2) int32 or None if out of view.

    ``view_rot`` is the optional world->view rotation from :func:`_bev_rot`
    (ego-up mode). It is applied to the box's OFFSET from the ego, after the
    box's own heading rotation, so a box keeps its shape and only the frame
    turns.
    """
    hl, hw = length / 2, width / 2
    corners_local = np.array([
        [hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw],
    ])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners_world = (rot @ corners_local.T).T + np.array([wx, wy])

    # To pixel
    dx = corners_world[:, 0] - ego_pos[0]
    dy = corners_world[:, 1] - ego_pos[1]
    if view_rot is not None:
        d = np.stack([dx, dy], axis=1) @ view_rot.T
        dx, dy = d[:, 0], d[:, 1]
    px = (cx + dx * ppm).astype(int)
    py = (cy - dy * ppm).astype(int)

    # Skip if entirely out of view
    if px.max() < -20 or px.min() > BEV_SIZE + 20 or py.max() < -20 or py.min() > BEV_SIZE + 20:
        return None

    return np.stack([px, py], axis=1).astype(np.int32)




# ---------------------------------------------------------------------------
# BEV with candidate trajectories (DiffusionDrive, etc.)
# ---------------------------------------------------------------------------

def render_bev_candidates(
    ego_position: np.ndarray,
    ego_heading: float,
    candidates_ego: np.ndarray,
    scores: Optional[np.ndarray] = None,
    planned_traj_world: Optional[np.ndarray] = None,
    prediction_position: Optional[np.ndarray] = None,
    prediction_heading: Optional[float] = None,
    frame_id: int = 0,
    model_name: str = "",
    scenario_data: Optional[Dict] = None,
) -> np.ndarray:
    """Render BEV with candidate trajectory proposals color-coded by score.

    Args:
        ego_position: Current ego position (for view centering).
        ego_heading: Current ego heading.
        candidates_ego: (K, N, 2) ego-frame candidates [lateral, forward] at prediction time.
        scores: (K,) scores per candidate (higher = better). Blue→Red color map.
        planned_traj_world: (N, 2) selected trajectory in world coords.
        prediction_position: Ego position when candidates were predicted.
        prediction_heading: Ego heading when candidates were predicted.
        frame_id: Frame number for HUD.
        model_name: Model name for HUD.
        scenario_data: Full scenario dict for map/agent rendering.

    Returns:
        (BEV_SIZE, BEV_SIZE, 3) uint8 BGR image.
    """
    # Start with the base BEV (map + agents, no trajectory)
    img = render_bev(
        ego_position=ego_position,
        ego_heading=ego_heading,
        planned_traj_world=None,  # we'll draw trajectories ourselves
        vehicle_states=None,
        frame_id=frame_id,
        model_name=model_name,
        scenario_data=scenario_data,
    )

    cx, cy = BEV_SIZE // 2, BEV_SIZE // 2

    def world_to_px(wx: float, wy: float):
        dx = wx - ego_position[0]
        dy = wy - ego_position[1]
        return int(cx + dx * BEV_PPM), int(cy - dy * BEV_PPM)

    pred_pos = prediction_position if prediction_position is not None else ego_position
    pred_h = prediction_heading if prediction_heading is not None else ego_heading
    cos_h = np.cos(pred_h)
    sin_h = np.sin(pred_h)

    # Normalize scores for color mapping
    if scores is not None and len(scores) > 0:
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            s_norm = (scores - s_min) / (s_max - s_min)
        else:
            s_norm = np.full_like(scores, 0.5)
    else:
        s_norm = np.linspace(0, 1, len(candidates_ego))

    # Draw all candidates (thin lines, blue=low → red=high)
    for ci, cand in enumerate(candidates_ego):
        t = float(s_norm[ci])
        color = (int(255 * (1 - t)), 0, int(255 * t))  # BGR: blue→red

        pts = []
        for wp in cand:
            # Transform from ego frame at prediction time → world
            w_x = float(pred_pos[0]) + cos_h * float(wp[1]) - sin_h * float(wp[0])
            w_y = float(pred_pos[1]) + sin_h * float(wp[1]) + cos_h * float(wp[0])
            px, py = world_to_px(w_x, w_y)
            pts.append([px, py])

        pts_arr = np.array(pts, dtype=np.int32)
        if len(pts_arr) > 1:
            cv2.polylines(img, [pts_arr], False, color, 1, cv2.LINE_AA)

    # Draw selected trajectory on top (thick cyan)
    if planned_traj_world is not None and len(planned_traj_world) > 0:
        pts = []
        for wp in planned_traj_world:
            px, py = world_to_px(float(wp[0]), float(wp[1]))
            pts.append([px, py])
            cv2.circle(img, (px, py), 3, (255, 255, 0), -1)  # cyan dots
        pts_arr = np.array(pts, dtype=np.int32)
        if len(pts_arr) > 1:
            cv2.polylines(img, [pts_arr], False, (255, 255, 0), 2, cv2.LINE_AA)

    # Legend overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (5, BEV_SIZE - 45), (260, BEV_SIZE - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    n_cands = len(candidates_ego)
    cv2.putText(img, f"{n_cands} candidates  Blue=Low Red=High",
                (10, BEV_SIZE - 28), _FONT, 0.4, (200, 200, 200), 1)
    cv2.putText(img, "Cyan = selected trajectory",
                (10, BEV_SIZE - 12), _FONT, 0.4, (255, 255, 0), 1)

    return img


# ---------------------------------------------------------------------------
# Front-camera: fallback rendering from scenario data
# ---------------------------------------------------------------------------

def _no_plan_overlay() -> bool:
    """``NAVSAFE_NO_PLAN_OVERLAY``: drop the model's planned trajectory.

    The top-down draws the policy's prediction (cyan-to-yellow dots plus time
    labels) over the same map as the ground-truth route. This suppresses the
    prediction only; the GT route and target waypoint are unaffected.
    """
    return os.environ.get("NAVSAFE_NO_PLAN_OVERLAY", "").strip().lower() not in (
        "", "0", "false", "no", "off")


def _no_hud() -> bool:
    """``NAVSAFE_NO_HUD``: drop every HUD annotation, top-down and camera.

    The overlay box, the frame/model/speed/command readout, the legend and the
    collision banner. Figures want the render, not the telemetry.
    """
    return os.environ.get("NAVSAFE_NO_HUD", "").strip().lower() not in (
        "", "0", "false", "no", "off")


def _no_route_overlay() -> bool:
    """``NAVSAFE_NO_ROUTE_OVERLAY``: drop the GT route from the camera view.

    The front cam carries two independent layers: the yellow polyline is the
    MODEL's planned trajectory, the red dots and the green target pole are the
    ground-truth route the evaluator is steering toward. For a figure showing
    what the policy predicted, the GT route is a distraction -- and worse, it
    reads as if the model produced it. This drops only the route layers;
    NAVSAFE_NO_OVERLAY drops everything and gives the raw render.
    """
    return os.environ.get("NAVSAFE_NO_ROUTE_OVERLAY", "").strip().lower() not in (
        "", "0", "false", "no", "off")


def render_front_cam_from_scenario(
    scenario_data: Dict,
    frame_id: int,
    ego_position: np.ndarray,
    ego_heading: float,
    planned_traj_world: Optional[np.ndarray] = None,
    model_name: str = "",
    speed_kmh: float = 0.0,
    collision: bool = False,
    fov_deg: float = 70.0,
    img_w: int = 1280,
    img_h: int = 720,
    max_depth: float = 60.0,
    driving_command: int = CMD_VOID,
    route_waypoints: Optional[List] = None,
    target_waypoint: Optional[tuple] = None,
) -> np.ndarray:
    """Render a perspective-projection front camera view from scenario data.

    Used as a fallback when real camera images are unavailable (zero-filled).
    Projects agents and trajectory from world-frame into an approximate camera.

    Returns (img_h, img_w, 3) uint8 BGR image.
    """
    from navsafe.scenario.scenario_description import ScenarioDescription as SD
    from navsafe.scenario.type import MetaDriveType

    img = np.full((img_h, img_w, 3), 200, dtype=np.uint8)  # light grey

    focal_len = (img_w / 2) / np.tan(np.radians(fov_deg / 2))

    def world_to_screen(wx: float, wy: float):
        dx = wx - ego_position[0]
        dy = wy - ego_position[1]
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        cam_x = cos_h * dx - sin_h * dy   # lateral in camera frame
        cam_y = sin_h * dx + cos_h * dy   # forward depth
        if cam_y <= 0.5:
            return None
        u = int(img_w / 2 + focal_len * (-cam_x) / cam_y)
        v = int(img_h * 0.6 - focal_len * 1.5 / cam_y)
        return u, v, cam_y

    # ── Sky + road background ──────────────────────────────────────────
    horizon_y = int(img_h * 0.45)
    img[:horizon_y, :] = (230, 180, 130)   # light blue sky (BGR)
    img[horizon_y:, :] = (180, 180, 180)   # light grey road

    # ── Road lane markings (projected) ─────────────────────────────────
    if scenario_data is not None:
        map_features = scenario_data.get(SD.MAP_FEATURES, {})
        for _fid, feat in map_features.items():
            feat_type = feat.get(SD.TYPE, "")
            if not MetaDriveType.is_road_line(feat_type):
                continue
            poly = feat.get(SD.POLYLINE)
            if poly is None or len(poly) < 2:
                continue
            pts_world = np.asarray(poly)[:, :2]
            # Cheap world-space bbox cull before the per-point projection loop.
            if (pts_world[:, 0].max() < ego_position[0] - _BEV_CULL_R or
                    pts_world[:, 0].min() > ego_position[0] + _BEV_CULL_R or
                    pts_world[:, 1].max() < ego_position[1] - _BEV_CULL_R or
                    pts_world[:, 1].min() > ego_position[1] + _BEV_CULL_R):
                continue
            screen_pts = []
            for p in pts_world:
                result = world_to_screen(float(p[0]), float(p[1]))
                if result is not None:
                    u, v, depth = result
                    if 0 <= u < img_w and 0 <= v < img_h and depth < max_depth:
                        screen_pts.append([u, v])
            if len(screen_pts) >= 2:
                pts_arr = np.array(screen_pts, dtype=np.int32)
                clr = (0, 180, 220) if MetaDriveType.is_yellow_line(feat_type) else (180, 180, 180)
                cv2.polylines(img, [pts_arr], False, clr, 1, cv2.LINE_AA)

    # ── Project agents ────────────────────────────────────────────────
    if scenario_data is not None:
        tracks = scenario_data.get(SD.TRACKS, {})
        metadata = scenario_data.get(SD.METADATA, {})
        ego_id = str(metadata.get(SD.SDC_ID, ""))

        draw_list = []  # (depth, u, v, box_w, box_h, color)
        for agent_id, track in tracks.items():
            if agent_id == ego_id:
                continue
            st = track.get(SD.STATE, {})
            positions = st.get("position")
            if positions is None or frame_id >= len(positions):
                continue
            valid_arr = st.get("valid")
            if valid_arr is not None and frame_id < len(valid_arr) and not valid_arr[frame_id]:
                continue

            apos = np.asarray(positions[frame_id])[:2]
            result = world_to_screen(float(apos[0]), float(apos[1]))
            if result is None:
                continue
            u, v, depth = result
            if depth > max_depth or depth < 1.0:
                continue
            if u < -img_w or u > 2 * img_w:
                continue

            agent_type = track.get(SD.TYPE, "OTHER")
            color = _agent_color_bgr(agent_type)

            # Scale box with depth
            scale = focal_len / depth
            if "length" in st and frame_id < len(st.get("length", [])):
                aw = float(st["width"][frame_id]) if "width" in st and frame_id < len(st["width"]) else 1.8
                ah = float(st["height"][frame_id]) if "height" in st and frame_id < len(st.get("height", [])) else 1.5
            else:
                aw, ah = (1.8, 1.5) if agent_type == MetaDriveType.VEHICLE else (0.4, 1.7)

            box_w = max(4, int(aw * scale))
            box_h = max(4, int(ah * scale))
            draw_list.append((depth, u, v, box_w, box_h, color))

        # Draw back-to-front
        draw_list.sort(key=lambda x: -x[0])
        for depth, u, v, box_w, box_h, color in draw_list:
            x1 = u - box_w // 2
            y1 = v - box_h
            cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), color, -1)
            cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), (255, 255, 255), 1)

    # ── Planned trajectory (projected) ─────────────────────────────────
    if planned_traj_world is not None and len(planned_traj_world) > 1:
        traj_pts = []
        for wp in planned_traj_world:
            result = world_to_screen(float(wp[0]), float(wp[1]))
            if result is not None:
                u, v, depth = result
                if 0 <= u < img_w and depth < max_depth:
                    traj_pts.append([u, v])
        if len(traj_pts) >= 2:
            pts_arr = np.array(traj_pts, dtype=np.int32)
            cv2.polylines(img, [pts_arr], False, _CLR_TRAJ, 3, cv2.LINE_AA)
            for pt in traj_pts:
                cv2.circle(img, tuple(pt), 3, (0, 255, 255), -1)

    # ── Route waypoints (projected red dots) ────────────────────────────
    if route_waypoints is not None and not _no_route_overlay():
        for i, wp_data in enumerate(route_waypoints):
            wp_pos = wp_data[0]
            result = world_to_screen(float(wp_pos[0]), float(wp_pos[1]))
            if result is not None:
                u, v, depth = result
                if 0 <= u < img_w and 0 <= v < img_h and depth < max_depth:
                    radius = 4 if i % 10 == 0 else 2
                    cv2.circle(img, (u, v), radius, (0, 0, 255), -1)

    # ── Target waypoint (projected green dot) ───────────────────────────
    if target_waypoint is not None and not _no_route_overlay():
        tw_pos = target_waypoint[0]
        result = world_to_screen(float(tw_pos[0]), float(tw_pos[1]))
        if result is not None:
            u, v, depth = result
            if 0 <= u < img_w and 0 <= v < img_h and depth < max_depth:
                cv2.circle(img, (u, v), 8, (0, 255, 0), -1)
                # Draw a vertical "pole" line above the target
                cv2.line(img, (u, v), (u, max(0, v - 30)), (0, 255, 0), 2)

    # ── HUD ────────────────────────────────────────────────────────────
    # NAVSAFE_NO_HUD gates the camera HUD too, not just the top-down's: the
    # green "model | frame | km/h" line and the Cmd box are the same kind of
    # annotation and equally unwanted in a figure.
    if _no_hud():
        return img

    if collision:
        cv2.rectangle(img, (0, 0), (img_w, 35), (0, 0, 180), -1)
        cv2.putText(img, f"CRASH!  |  Frame {frame_id}  {speed_kmh:.0f}km/h",
                    (10, 25), _FONT, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(img, f"{model_name}  |  Frame {frame_id}  t={frame_id * 0.1:.1f}s  {speed_kmh:.0f}km/h",
                    (10, 25), _FONT, 0.6, (0, 220, 0), 2)
        if driving_command != CMD_VOID:
            cmd_name = _CMD_NAMES.get(driving_command, "VOID")
            cmd_color = _CMD_COLORS.get(driving_command, (128, 128, 128))
            overlay_cmd = img.copy()
            cv2.rectangle(overlay_cmd, (5, img_h - 40), (180, img_h - 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay_cmd, 0.5, img, 0.5, 0, img)
            cv2.putText(img, f"Cmd: {cmd_name}", (10, img_h - 15),
                        _FONT, 0.6, cmd_color, 2)

    return img


# ---------------------------------------------------------------------------
# Front-camera trajectory overlay (for real camera images)
# ---------------------------------------------------------------------------

def project_ego_to_camera(
    waypoints_ego: np.ndarray,
    cam_config: Dict,
    image_shape: Tuple[int, int],
    z_height: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project ego-frame waypoints onto a camera image.

    Args:
        waypoints_ego: (N, 2) — col0=lateral (left+), col1=forward.
        cam_config: dict with x, y, z, yaw, fov, width, height.
        image_shape: (H, W).
        z_height: ground-plane height offset.

    Returns:
        pixels (N, 2) and valid mask (N,).
    """
    H, W = image_shape[:2]
    # Intrinsics: focal from the HORIZONTAL fov with square pixels, exactly like
    # NuRec gRPC's CameraSpec: fx = fy = W / (2 tan(fov_h/2)). The nominal "fov" field
    # is a wider diagonal value (~70deg); using it under-zooms the projection and
    # floats the overlay a few percent of image height off the ground (worst in
    # the near field). fov_h falls back to fov when absent.
    fov_h = float(cam_config.get("fov_h", cam_config.get("fov", 70.0)))
    fx = fy = W / (2.0 * np.tan(np.radians(fov_h) / 2.0))
    cx_img, cy_img = W / 2.0, H / 2.0

    # Extrinsic: the camera's full mount pose in the ego/rig FLU frame (x-fwd,
    # y-left, z-up) — body offset (x, y, z) plus yaw/pitch/roll — matching the
    # renderers' camera pose. The old code applied yaw only (and with the wrong
    # sign), leaving a lateral offset and a small vertical error from the ignored
    # mount pitch. The optical->FLU base R_base (camera z -> rig +x forward,
    # x -> -y right, y -> -z down) was calibrated against the served WOD quat.
    bx = float(cam_config.get("x", 0.0))
    by = float(cam_config.get("y", 0.0))
    bz = float(cam_config.get("z", 0.0))
    yaw = np.radians(float(cam_config.get("yaw", 0.0)))
    pitch = np.radians(float(cam_config.get("pitch", 0.0)))
    roll = np.radians(float(cam_config.get("roll", 0.0)))
    cz, sz = np.cos(yaw), np.sin(yaw)
    cyp, syp = np.cos(pitch), np.sin(pitch)
    cxr, sxr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cyp, 0.0, syp], [0.0, 1.0, 0.0], [-syp, 0.0, cyp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cxr, -sxr], [0.0, sxr, cxr]])
    R_base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R_c2r = (Rz @ Ry @ Rx) @ R_base   # camera -> rig
    t = np.array([bx, by, bz])

    wp = np.asarray(waypoints_ego, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[0] == 0:
        return np.empty((0, 2)), np.zeros((0,), dtype=bool)
    N = wp.shape[0]
    rig = np.empty((N, 3))
    rig[:, 0] = wp[:, 1]     # forward       -> FLU x
    rig[:, 1] = wp[:, 0]     # lateral(left+) -> FLU y
    # up -> FLU z: per-point height when (N,3) input, else the flat plane
    rig[:, 2] = (wp[:, 2] + z_height) if wp.shape[1] >= 3 else z_height
    cam = (rig - t) @ R_c2r  # rig -> camera (opencv: x-right, y-down, z-fwd)

    valid = cam[:, 2] > 0.1
    denom = np.where(valid, cam[:, 2], 1.0)
    u = fx * cam[:, 0] / denom + cx_img
    v = fy * cam[:, 1] / denom + cy_img
    u = np.clip(u, -W, 2 * W)
    v = np.clip(v, -H, 2 * H)

    pixels = np.column_stack([u, v])
    pixels[~valid] = -1
    return pixels, valid


def render_front_cam(
    image: np.ndarray,
    plan_traj_ego: Optional[np.ndarray],
    cam_config: Dict,
    frame_id: int = 0,
    model_name: str = "",
    speed_kmh: float = 0.0,
    collision: bool = False,
    ribbon_color: Tuple[int, int, int] = (0, 200, 200),
    ribbon_width: float = 0.9,
    ribbon_alpha: float = 0.55,
    driving_command: int = CMD_VOID,
    sim_dt: float = 0.1,
    route_waypoints: Optional[List] = None,
    target_waypoint: Optional[tuple] = None,
    ego_position: Optional[np.ndarray] = None,
    ego_heading: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """Annotate a front-camera image with a projected trajectory ribbon + HUD.

    Also projects road lane markings from scenario data onto the camera image
    so road geometry is visible even when the 3D render lacks road textures.

    Args:
        driving_command: Driving command int for overlay text.
        sim_dt: Simulation timestep for per-waypoint timestamp labels.
        route_waypoints: list of (position_2d, command, frame_idx) — static GT route.
        target_waypoint: (position_2d, command, frame_idx) — current target waypoint.
        ego_position: (3,) ego world position (needed for route projection).
        ego_heading: ego heading in radians (needed for route projection).
    """
    if _no_overlay():
        return image  # pure grpc frame: skip lane/trajectory/HUD overlay
    img = image.copy()
    H, W = img.shape[:2]

    # ── Project road lane markings onto camera image ────────────────────
    # This ensures road geometry is visible even when the 3D render lacks
    # road textures (e.g. MDL material loading failure in IsaacSim 5.x).
    #
    # NAVSAFE_NO_CAM_MAP_LINES turns just this block off, keeping the HUD,
    # the plan ribbon and the route/target dots. On a photoreal reconstruction
    # the road is already in the pixels, so the grey polylines are redundant
    # clutter drawn over real lane markings — while the trajectory overlays
    # are still what makes the frame readable.
    if (ego_position is not None and ego_heading is not None
            and not _no_cam_map_lines()):
        try:
            from navsafe.scenario.scenario_description import ScenarioDescription as SD
            from navsafe.scenario.type import MetaDriveType

            # Get scenario data from the evaluator context (passed via kwargs or global)
            scenario_data = kwargs.get("scenario_data")
            if scenario_data is not None:
                map_features = scenario_data.get(SD.MAP_FEATURES, {})
                cos_h = np.cos(-ego_heading)
                sin_h = np.sin(-ego_heading)

                # Local ground altitude: the ego track's recorded z at this
                # frame (the live chassis z may be zeroed by the env). The map
                # polylines are 3D; projecting their true height keeps lines on
                # sloped/banked road instead of a flat plane through the ego.
                z_ego = None
                try:
                    _meta = scenario_data.get("metadata", {}) or {}
                    _sdc = _meta.get("sdc_id") or scenario_data.get("sdc_id") or "ego"
                    _zt = np.asarray(
                        scenario_data["tracks"][_sdc]["state"]["position"],
                        dtype=np.float64)
                    if _zt.ndim >= 2 and _zt.shape[-1] >= 3:
                        _zt = _zt.reshape(-1, _zt.shape[-1])[:, 2]
                        if np.any(_zt != 0.0):
                            z_ego = float(_zt[min(max(int(frame_id), 0), len(_zt) - 1)])
                            # The overlay camera lives in a ground-origin ego
                            # frame (flat plane z=0 = road, cam at cfg z), but
                            # the recorded ego pose z sits above the road (WOD
                            # ~1.4 m). Drop the reference so map polylines land
                            # on the z=0 ground plane — consistent with the
                            # ribbon/route overlays AND the nurec_grpc render
                            # (whose navsim camera got the same drop). Set by
                            # the evaluator from --obstacle-z-to-ground.
                            import os as _os
                            z_ego -= float(_os.environ.get(
                                "NAVSAFE_EGO_Z_TO_GROUND", "0.0"))
                except Exception:
                    z_ego = None

                for _fid, feat in map_features.items():
                    feat_type = feat.get(SD.TYPE, "")
                    if not MetaDriveType.is_road_line(feat_type):
                        continue
                    poly = feat.get(SD.POLYLINE)
                    if poly is None or len(poly) < 2:
                        continue
                    pw = np.asarray(poly, dtype=np.float64)

                    # Cheap world-space bbox cull: skip road-lines whose bbox is
                    # entirely outside ego ± _BEV_CULL_R before projecting (on
                    # city-scale maps thousands of far lines otherwise get
                    # projected every frame).
                    if (pw[:, 0].max() < float(ego_position[0]) - _BEV_CULL_R or
                            pw[:, 0].min() > float(ego_position[0]) + _BEV_CULL_R or
                            pw[:, 1].max() < float(ego_position[1]) - _BEV_CULL_R or
                            pw[:, 1].min() > float(ego_position[1]) + _BEV_CULL_R):
                        continue

                    # world -> ego frame (lat, fwd, dz). Keep EVERY vertex:
                    # dropping behind-camera points and re-joining the survivors
                    # used to draw false strokes across the image.
                    d0 = pw[:, 0] - float(ego_position[0])
                    d1 = pw[:, 1] - float(ego_position[1])
                    lat = sin_h * d0 + cos_h * d1
                    fwd = cos_h * d0 - sin_h * d1
                    if pw.shape[1] >= 3 and z_ego is not None:
                        dz = pw[:, 2] - z_ego
                    else:
                        dz = np.zeros_like(lat)
                    ego_arr = np.column_stack([lat, fwd, dz])

                    px, valid = project_ego_to_camera(ego_arr, cam_config, (H, W))
                    # Draw only runs of CONSECUTIVE vertices that are in front
                    # and near the frame. Far-outside points come back clipped
                    # (to [-W,2W]) and used to paint long artifact strokes
                    # across the image when joined.
                    inb = (valid & (px[:, 0] > -0.25 * W) & (px[:, 0] < 1.25 * W)
                           & (px[:, 1] > -0.25 * H) & (px[:, 1] < 1.25 * H))
                    clr = (0, 140, 180) if MetaDriveType.is_yellow_line(feat_type) else (140, 140, 140)
                    broken = MetaDriveType.is_broken_line(feat_type)
                    run = []
                    for i in range(len(inb) + 1):
                        if i < len(inb) and inb[i]:
                            run.append((int(px[i, 0]), int(px[i, 1])))
                            continue
                        if len(run) >= 2:
                            pts = np.asarray(run, dtype=np.int32)
                            if broken:
                                for j in range(0, len(pts) - 1, 2):
                                    cv2.line(img, tuple(pts[j]), tuple(pts[j + 1]),
                                             clr, 1, cv2.LINE_AA)
                            else:
                                cv2.polylines(img, [pts], False, clr, 1, cv2.LINE_AA)
                        run = []
        except Exception:
            pass  # Non-fatal — road markings are a visual enhancement

    if plan_traj_ego is not None and len(plan_traj_ego) >= 2:
        offsets_l, offsets_r = [], []
        for i in range(len(plan_traj_ego) - 1):
            p1, p2 = plan_traj_ego[i], plan_traj_ego[i + 1]
            d = p2 - p1
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            normal = np.array([-d[1], d[0]]) / n
            offsets_l.append(p1 + ribbon_width * normal)
            offsets_r.append(p1 - ribbon_width * normal)

        if len(plan_traj_ego) > 1:
            d = plan_traj_ego[-1] - plan_traj_ego[-2]
            n = np.linalg.norm(d)
            if n > 1e-6:
                normal = np.array([-d[1], d[0]]) / n
                offsets_l.append(plan_traj_ego[-1] + ribbon_width * normal)
                offsets_r.append(plan_traj_ego[-1] - ribbon_width * normal)

        if len(offsets_l) >= 2:
            pts_l, vl = project_ego_to_camera(np.array(offsets_l), cam_config, (H, W))
            pts_r, vr = project_ego_to_camera(np.array(offsets_r), cam_config, (H, W))
            valid = vl & vr
            if np.any(valid):
                pts_l, pts_r = pts_l[valid], pts_r[valid]
                anchor_l = np.array([[W / 2 - 40, H - 1]])
                anchor_r = np.array([[W / 2 + 40, H - 1]])
                pts_l = np.vstack([anchor_l, pts_l])
                pts_r = np.vstack([anchor_r, pts_r])
                pts_l[:, 0] = np.clip(pts_l[:, 0], 0, W - 1)
                pts_l[:, 1] = np.clip(pts_l[:, 1], 0, H - 1)
                pts_r[:, 0] = np.clip(pts_r[:, 0], 0, W - 1)
                pts_r[:, 1] = np.clip(pts_r[:, 1], 0, H - 1)

                polygon = np.vstack([pts_l, pts_r[::-1]]).astype(np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [polygon], ribbon_color)
                cv2.addWeighted(overlay, ribbon_alpha, img, 1 - ribbon_alpha, 0, img)

                edge_c = tuple(int(c * 0.7) for c in ribbon_color)
                cv2.polylines(img, [pts_l.astype(np.int32)], False, edge_c, 2, cv2.LINE_AA)
                cv2.polylines(img, [pts_r.astype(np.int32)], False, edge_c, 2, cv2.LINE_AA)

    # ── Waypoint dots + timestamp labels on trajectory centerline ───────
    if plan_traj_ego is not None and len(plan_traj_ego) >= 2:
        center_pts, center_valid = project_ego_to_camera(plan_traj_ego, cam_config, (H, W))
        for i in range(len(center_pts)):
            if not center_valid[i]:
                continue
            u, v = int(center_pts[i, 0]), int(center_pts[i, 1])
            if 0 <= u < W and 0 <= v < H:
                # Gradient dot: cyan → yellow
                t = i / max(len(center_pts) - 1, 1)
                dot_color = (int(255 * (1 - t)), 255, int(255 * t))
                cv2.circle(img, (u, v), 4, dot_color, -1)
                # Timestamp label every 5 waypoints
                if i % 5 == 0 and i > 0:
                    ts = i * sim_dt
                    cv2.putText(img, f"{ts:.1f}s", (u + 5, v - 5),
                                _FONT, 0.35, (255, 255, 255), 1)

    # ── Route waypoints (projected red dots on real camera) ────────────
    if (route_waypoints is not None and ego_position is not None
            and ego_heading is not None and not _no_route_overlay()):
        # Transform route waypoints from world to ego frame, then project
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        for i, wp_data in enumerate(route_waypoints):
            wp_pos = wp_data[0]
            delta = wp_pos - ego_position[:2]
            ego_lat = sin_h * delta[0] + cos_h * delta[1]
            ego_fwd = cos_h * delta[0] - sin_h * delta[1]
            if ego_fwd < 0.5:  # behind camera
                continue
            wp_ego = np.array([[ego_lat, ego_fwd]])
            px, valid = project_ego_to_camera(wp_ego, cam_config, (H, W))
            if valid[0]:
                u, v = int(px[0, 0]), int(px[0, 1])
                if 0 <= u < W and 0 <= v < H:
                    radius = 4 if i % 10 == 0 else 2
                    cv2.circle(img, (u, v), radius, (0, 0, 255), -1)

    # ── Target waypoint (projected green dot on real camera) ────────────
    if (target_waypoint is not None and ego_position is not None
            and ego_heading is not None and not _no_route_overlay()):
        tw_pos = target_waypoint[0]
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        delta = tw_pos - ego_position[:2]
        ego_lat = sin_h * delta[0] + cos_h * delta[1]
        ego_fwd = cos_h * delta[0] - sin_h * delta[1]
        if ego_fwd > 0.5:
            tw_ego = np.array([[ego_lat, ego_fwd]])
            px, valid = project_ego_to_camera(tw_ego, cam_config, (H, W))
            if valid[0]:
                u, v = int(px[0, 0]), int(px[0, 1])
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(img, (u, v), 8, (0, 255, 0), -1)
                    cv2.line(img, (u, v), (u, max(0, v - 30)), (0, 255, 0), 2)

    # ── HUD with driving command ────────────────────────────────────────
    if _no_hud():
        return img

    if collision:
        cv2.rectangle(img, (0, 0), (W, 35), (0, 0, 180), -1)
        cv2.putText(img, f"CRASH!  |  Frame {frame_id}  {speed_kmh:.0f}km/h",
                    (10, 25), _FONT, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(img, f"{model_name}  |  Frame {frame_id}  {speed_kmh:.0f}km/h",
                    (10, 25), _FONT, 0.6, (0, 220, 0), 2)
        # Driving command overlay (bottom-left)
        if driving_command != CMD_VOID:
            cmd_name = _CMD_NAMES.get(driving_command, "VOID")
            cmd_color = _CMD_COLORS.get(driving_command, (128, 128, 128))
            overlay_cmd = img.copy()
            cv2.rectangle(overlay_cmd, (5, H - 40), (180, H - 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay_cmd, 0.5, img, 0.5, 0, img)
            cv2.putText(img, f"Cmd: {cmd_name}", (10, H - 15),
                        _FONT, 0.6, cmd_color, 2)

    return img


def reanchor_candidates_to_ego(
    candidates_ego: np.ndarray,
    prediction_position: np.ndarray,
    prediction_heading: float,
    ego_position: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """Re-express prediction-time ego-frame candidates in the *current* ego frame.

    Candidates are produced at a replan frame and reused until the next one, so
    between replans their origin (the ego pose at prediction time) has drifted
    away from the ego pose the camera image was captured at. Projecting them
    directly would pin the fan to the ego rather than to the ground.

    Args:
        candidates_ego: ``(K, N, 2)`` ``[lateral, forward]`` at prediction time.
        prediction_position / prediction_heading: ego pose when predicted.
        ego_position / ego_heading: ego pose of the frame being rendered.

    Returns:
        ``(K, N, 2)`` ``[lateral, forward]`` in the current ego frame.
    """
    cands = np.asarray(candidates_ego, dtype=np.float64)
    lat, fwd = cands[..., 0], cands[..., 1]

    # prediction-time ego frame → world
    cp, sp = np.cos(float(prediction_heading)), np.sin(float(prediction_heading))
    wx = float(prediction_position[0]) + cp * fwd - sp * lat
    wy = float(prediction_position[1]) + sp * fwd + cp * lat

    # world → current ego frame
    dx = wx - float(ego_position[0])
    dy = wy - float(ego_position[1])
    ce, se = np.cos(float(ego_heading)), np.sin(float(ego_heading))
    cur_fwd = ce * dx + se * dy
    cur_lat = -se * dx + ce * dy
    return np.stack([cur_lat, cur_fwd], axis=-1)


def render_front_cam_candidates(
    image: np.ndarray,
    candidates_ego: np.ndarray,
    cam_config: Dict,
    scores: Optional[np.ndarray] = None,
    plan_traj_ego: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Overlay projected candidate trajectories onto a front-camera image.

    Draws every candidate as a thin ground-plane polyline coloured blue (low
    score) → red (high score), then re-draws the selected trajectory thick and
    cyan on top so it reads against the fan. Mirrors
    :func:`render_bev_candidates`'s colour convention.

    Args:
        image: Front-camera image to annotate (BGR). Copied, not mutated.
        candidates_ego: ``(K, N, 2)`` ``[lateral, forward]`` in the *current*
            ego frame — run :func:`reanchor_candidates_to_ego` first if the
            candidates came from an earlier replan.
        cam_config: Camera intrinsics/extrinsics (x, y, z, yaw, fov, ...).
        scores: ``(K,)`` per-candidate score, higher = better.
        plan_traj_ego: ``(N, 2)`` selected trajectory in the current ego frame.

    Returns:
        Annotated copy of ``image``.
    """
    img = image.copy()
    H, W = img.shape[:2]

    cands = np.asarray(candidates_ego, dtype=np.float64)
    if cands.ndim != 3 or cands.shape[0] == 0:
        return img

    if scores is not None and len(scores) == len(cands):
        scores = np.asarray(scores, dtype=np.float64)
        s_min, s_max = scores.min(), scores.max()
        s_norm = ((scores - s_min) / (s_max - s_min) if s_max > s_min
                  else np.full(len(cands), 0.5))
    else:
        s_norm = np.linspace(0.0, 1.0, len(cands))

    for ci, cand in enumerate(cands):
        t = float(s_norm[ci])
        color = (int(255 * (1 - t)), 0, int(255 * t))  # BGR: blue→red

        pixels, valid = project_ego_to_camera(cand, cam_config, (H, W))
        pts = pixels[valid].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)

    # Selected trajectory on top — thick cyan, matching the BEV overlay.
    if plan_traj_ego is not None and len(plan_traj_ego) > 1:
        pixels, valid = project_ego_to_camera(
            np.asarray(plan_traj_ego, dtype=np.float64)[:, :2], cam_config, (H, W))
        pts = pixels[valid].astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (255, 255, 0), 2, cv2.LINE_AA)
            for u, v in pts:
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(img, (int(u), int(v)), 3, (255, 255, 0), -1)

    # Legend overlay (bottom-right, clear of render_front_cam's Cmd box).
    overlay = img.copy()
    cv2.rectangle(overlay, (W - 265, H - 45), (W - 5, H - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.putText(img, f"{len(cands)} candidates  Blue=Low Red=High",
                (W - 260, H - 28), _FONT, 0.4, (200, 200, 200), 1)
    cv2.putText(img, "Cyan = selected trajectory",
                (W - 260, H - 12), _FONT, 0.4, (255, 255, 0), 1)
    return img


# ---------------------------------------------------------------------------
# GIF / video generation
# ---------------------------------------------------------------------------

def generate_gif(
    frame_dir: Path,
    frame_ids: List[int],
    output_path: Path,
    source_filename: str,
    fps: int = 10,
) -> bool:
    """Assemble per-frame images into an animated GIF."""
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not installed — skipping GIF generation (pip install imageio)")
        return False

    frames = []
    for fid in frame_ids:
        p = frame_dir / f"{fid:05d}" / source_filename
        if p.exists():
            bgr = cv2.imread(str(p))
            if bgr is not None:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if not frames:
        logger.warning(f"No frames found for {source_filename}")
        return False

    # Ensure all frames have the same size (use the most common size)
    from collections import Counter
    sizes = Counter(f.shape[:2] for f in frames)
    target_h, target_w = sizes.most_common(1)[0][0]
    # List[Any]: imageio's stub wants the invariant List[ArrayLike], which a
    # list[np.ndarray] does not satisfy.
    resized: List[Any] = []
    for f in frames:
        if f.shape[0] != target_h or f.shape[1] != target_w:
            f = cv2.resize(f, (target_w, target_h))
        resized.append(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), resized, fps=fps, loop=0)
    logger.info(f"Saved GIF ({len(frames)} frames) → {output_path}")
    return True


def generate_combined_gif(
    frame_dir: Path,
    frame_ids: List[int],
    output_path: Path,
    left_filename: str = "topdown.png",
    right_filename: str = "cam_f0.jpg",
    fps: int = 10,
    target_height: int = 480,
    extra_filenames: Sequence[str] = (),
) -> bool:
    """Create a side-by-side GIF (BEV left, front-cam right).

    ``extra_filenames`` appends further panels to the right (e.g. a rear view)
    at the same target height. A frame is emitted only when *every* requested
    panel exists, so a camera that dropped out mid-run shortens the GIF rather
    than producing a row that silently changes meaning partway through.
    """
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not installed — skipping combined GIF")
        return False

    panel_names = [left_filename, right_filename, *extra_filenames]
    frames = []
    for fid in frame_ids:
        paths = [frame_dir / f"{fid:05d}" / n for n in panel_names]
        if not all(p.exists() for p in paths):
            continue
        panels = [cv2.imread(str(p)) for p in paths]
        if any(p is None for p in panels):
            continue

        def _resize(im: np.ndarray) -> np.ndarray:
            h, w = im.shape[:2]
            scale = target_height / h
            return cv2.resize(im, (int(w * scale), target_height))

        combined = np.hstack([_resize(p) for p in panels])
        frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    if not frames:
        logger.warning("No combined frames produced")
        return False

    # Ensure all frames have the same size
    from collections import Counter
    sizes = Counter(f.shape[:2] for f in frames)
    target_h2, target_w2 = sizes.most_common(1)[0][0]
    # List[Any]: see note in generate_gif — imageio's stub wants List[ArrayLike].
    resized_frames: List[Any] = []
    for f in frames:
        if f.shape[0] != target_h2 or f.shape[1] != target_w2:
            f = cv2.resize(f, (target_w2, target_h2))
        resized_frames.append(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), resized_frames, fps=fps, loop=0)
    logger.info(f"Saved combined GIF ({len(frames)} frames) → {output_path}")
    return True


# ---------------------------------------------------------------------------
# Online (live) display
# ---------------------------------------------------------------------------

_WINDOW_INIT = False


def show_online(
    bev_img: Optional[np.ndarray],
    cam_img: Optional[np.ndarray],
    wait_ms: int = 1,
) -> None:
    """Display BEV and front-camera images in OpenCV windows."""
    global _WINDOW_INIT
    if not _WINDOW_INIT:
        if bev_img is not None:
            cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)
        if cam_img is not None:
            cv2.namedWindow("Front Camera", cv2.WINDOW_NORMAL)
        _WINDOW_INIT = True

    if bev_img is not None:
        cv2.imshow("BEV", bev_img)
    if cam_img is not None:
        cv2.imshow("Front Camera", cam_img)
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()


def close_online() -> None:
    """Destroy any open visualization windows."""
    global _WINDOW_INIT
    if _WINDOW_INIT:
        cv2.destroyAllWindows()
        _WINDOW_INIT = False
