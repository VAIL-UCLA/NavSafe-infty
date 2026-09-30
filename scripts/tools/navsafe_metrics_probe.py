#!/usr/bin/env python
# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Replay a py123d scenario and compute diagnostic NavSafe metrics.

Uses the NuRec gRPC renderer and the current evaluation environment config.
The diagnostic output is separate from the benchmark evaluator's scored
result; it compares the replay trace against the recorded ego trajectory.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--py123d-data-root", required=True)
ap.add_argument("--py123d-scene-id", default=None)
ap.add_argument("--py123d-scene-index", type=int, default=0)
ap.add_argument("--py123d-frame-window", type=int, nargs=2, default=None)
ap.add_argument("--nurec-work-dir", default=None)
ap.add_argument("--traffic-mode", default="semi_reactive",
                choices=["no_traffic", "log_replay", "semi_reactive"])
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--sim-dt", type=float, default=0.1)
ap.add_argument("--output-dir", required=True)
ap.set_defaults(scenario_source="py123d", render_backend="nurec_grpc", execution_mode="teleport")

import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from navsafe.cli.isaac_boot import ensure_isaac_glib_preload  # noqa: E402

if "--help" in sys.argv or "-h" in sys.argv:
    ap.parse_args()

ensure_isaac_glib_preload()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from navsafe.env import NexusSimEnv  # noqa: E402
from navsafe.evaluation.eval_env_config import build_eval_env_cfg  # noqa: E402
from navsafe.benchmark.scoring import metrics as navsafe_metrics  # noqa: E402


def main() -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_eval_env_cfg(args)
    cfg.max_episode_steps = args.frames + 10
    print(f"[probe] traffic_mode={cfg.traffic_mode} backend={cfg.render_backend}",
          flush=True)
    env = NexusSimEnv(cfg)
    env.reset()

    dt = args.sim_dt
    rec = {"speed": [], "pos": [], "heading": [], "bg_speed": [],
           "collision": 0, "takeovers": 0}
    prev_agent_pos = {}

    # Static (parked) vehicles are not "background traffic" — Bench2Drive's
    # background set is the moving BackgroundActivity spawn, and nuPlan logs
    # are full of parked cars that would otherwise crush the denominator.
    _tracks0 = env.current_scenario.get("tracks", {})
    static_ids = set()
    for _aid, _tr in _tracks0.items():
        _p = np.asarray(_tr.get("state", {}).get("position", []), dtype=np.float64)
        if _p.ndim != 2 or len(_p) < 2 or np.linalg.norm(_p[-1, :2] - _p[0, :2]) < 5.0:
            static_ids.add(_aid)

    def ego_speed(state) -> float:
        v = state.get("velocity", 0.0)
        if isinstance(v, (int, float)):
            return float(v)
        return float(np.linalg.norm(np.asarray(v, dtype=np.float64)[:2]))

    for t in range(args.frames):
        gt = env.get_gt_ego_action(t)
        action = torch.tensor(np.asarray(gt, dtype=np.float32)[np.newaxis, :],
                              dtype=torch.float32,
                              device=getattr(env, "device", "cpu"))
        obs, rew, term, trunc, info = env.step(action)

        st = env.get_ego_state()
        rec["speed"].append(ego_speed(st))
        rec["pos"].append([float(st["position"][0]), float(st["position"][1])])
        rec["heading"].append(float(st.get("heading", 0.0)))
        # Background speeds from the live world: replaying agents at their
        # logged state, taken-over agents at their IDM-integrated state
        # (pose_overrides), speed by position differencing across frames.
        tm_now = getattr(env, "_traffic_manager", None)
        overrides = getattr(tm_now, "pose_overrides", None) or {}
        tracks_now = env.current_scenario.get("tracks", {})
        ego_id_now = getattr(env.agent_manager, "ego_agent_id", None)
        bg = []
        for aid, track in tracks_now.items():
            if aid == ego_id_now:
                continue
            if str(track.get("type", "VEHICLE")).upper() not in (
                    "VEHICLE", "CAR", "TRUCK", "BUS"):
                continue
            if aid in static_ids:
                continue
            if aid in overrides:
                p_now = np.asarray(overrides[aid]["position"][:2], dtype=np.float64)
            else:
                a_state = env.agent_manager._get_state_at_timestep(track, t)
                if a_state is None or not a_state.get("valid", True):
                    prev_agent_pos.pop(aid, None)
                    continue
                p_now = np.asarray(a_state["position"][:2], dtype=np.float64)
            p_prev = prev_agent_pos.get(aid)
            prev_agent_pos[aid] = p_now
            if p_prev is not None:
                bg.append(float(np.linalg.norm(p_now - p_prev)) / dt)
        rec["bg_speed"].append(float(np.mean(bg)) if bg else float("nan"))
        if isinstance(info, dict) and info.get("collision", False):
            rec["collision"] += 1
        if bool((term | trunc).any()):
            print(f"[probe] episode ended at frame {t}", flush=True)
            break

    tm = getattr(env, "_traffic_manager", None)
    taken = getattr(tm, "_taken", {}) or {}
    rec["takeovers"] = len(taken)
    n = len(rec["speed"])
    print(f"[probe] frames={n} takeovers={rec['takeovers']} "
          f"collision_frames={rec['collision']}", flush=True)

    # ── per-frame signals ────────────────────────────────────────────────
    # Teleport execution has no dynamics, and the pkl positions are stored on
    # a ~0.08 m grid — raw frame differencing turns that quantization into
    # +-18 m/s^2 phantom accelerations. Savitzky-Golay-smooth the positions
    # before differentiating (signal derivation, distinct from the metric's
    # own per-channel smoothing; the B2D reference reads accelerations from
    # the CARLA physics API and never faces this).
    from scipy.signal import savgol_filter as _sg

    def _smooth_speed(p, dt):
        w = min(11, len(p) - (1 - len(p) % 2))   # odd, <= len
        if w < 5:
            d = np.hypot(*np.diff(p, axis=0).T)
            return np.concatenate([[d[0] / dt if len(d) else 0.0], d / dt])
        vx = _sg(p[:, 0], w, 3, deriv=1, delta=dt)
        vy = _sg(p[:, 1], w, 3, deriv=1, delta=dt)
        return np.hypot(vx, vy)

    pos = np.asarray(rec["pos"])
    speed = _smooth_speed(pos, dt)
    heading = np.unwrap(np.asarray(rec["heading"]))
    lon_acc = np.gradient(speed, dt)
    yaw_rate = np.gradient(heading, dt)
    lat_acc = speed * yaw_rate
    mag_acc = np.hypot(lon_acc, lat_acc)
    accum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pos, axis=0).T))])
    bg = np.asarray(rec["bg_speed"])

    # ── route completion vs the GT ego trajectory ────────────────────────
    tracks = env.current_scenario.get("tracks", {})
    ego_id = getattr(env.agent_manager, "ego_agent_id", None)
    gt_xy = np.asarray(tracks[ego_id]["state"]["position"], dtype=np.float64)[:, :2]
    gt_seg = np.hypot(*np.diff(gt_xy, axis=0).T)
    gt_arc = np.concatenate([[0.0], np.cumsum(gt_seg)])
    d_end = float(np.linalg.norm(pos[-1] - gt_xy[-1]))
    # progress: arc length of the nearest GT vertex to the final ego position
    completion = 100.0 * float(
        gt_arc[int(np.argmin(np.linalg.norm(gt_xy - pos[-1], axis=1)))]
        / max(gt_arc[-1], 1e-6))
    completed = completion > 99.0 and d_end < 10.0
    if completed:
        completion = 100.0

    # ── the four NavSafe metrics ─────────────────────────────────────
    infractions = {}
    if rec["collision"]:
        infractions["collisions_vehicle"] = 1   # events, conservatively 1
    route = navsafe_metrics.RouteResult("np_dense_01", completion, completed, infractions)
    ds = route.driving_score()
    sr = route.success()
    # route_efficiency bins by route completion (%), not odometer. This probe
    # has no monotone route cursor, so scale the odometer by the GT route
    # length — exact for the GT oracle below, an approximation for the sim run
    # (they diverge only if the ego drives off-route and back).
    _route_len = float(np.sum(np.hypot(*np.diff(gt_xy[:n], axis=0).T))) or 1.0
    eff = navsafe_metrics.route_efficiency(speed, bg, 100.0 * accum / _route_len)
    comfort = navsafe_metrics.route_comfort(lon_acc, lat_acc, mag_acc, yaw_rate)

    np.savez(out_dir / f"signals_{cfg.traffic_mode}.npz",
             speed=speed, lon_acc=lon_acc, lat_acc=lat_acc, mag_acc=mag_acc,
             yaw_rate=yaw_rate, bg=bg, accum=accum, pos=pos, heading=heading)
    print(f"[probe] signal ranges: speed [{speed.min():.2f},{speed.max():.2f}] "
          f"max|lon_acc|={np.abs(lon_acc).max():.2f} "
          f"max|lat_acc|={np.abs(lat_acc).max():.2f} "
          f"max|yaw_rate|={np.abs(yaw_rate).max():.3f} "
          f"max|mag_jerk|~{np.abs(np.gradient(mag_acc, dt)).max():.2f} "
          f"bg_valid_frames={int(np.isfinite(bg).sum())}", flush=True)

    # ── metric-correctness oracle: the GT LOG itself ─────────────────────
    # The human drive must be comfortable and flow-matched; feeding the raw
    # log through the metric code is the clean verification, independent of
    # teleport-execution artifacts in the sim-recorded signals.
    gt_speed = _smooth_speed(gt_xy[:n], dt)
    gt_step = np.hypot(*np.diff(gt_xy[:n], axis=0).T)
    gt_heading = np.unwrap(np.asarray(
        tracks[ego_id]["state"]["heading"], dtype=np.float64)[:n])
    gt_lon = np.gradient(gt_speed, dt)
    gt_yr = np.gradient(gt_heading, dt)
    gt_lat = gt_speed * gt_yr
    gt_mag = np.hypot(gt_lon, gt_lat)
    gt_accum = np.concatenate([[0.0], np.cumsum(gt_step)])
    comfort_gt = navsafe_metrics.route_comfort(gt_lon, gt_lat, gt_mag, gt_yr)
    eff_gt = navsafe_metrics.route_efficiency(
        gt_speed, bg, 100.0 * gt_accum / (gt_accum[-1] or 1.0))

    result = {
        "traffic_mode": cfg.traffic_mode,
        "frames": n,
        "takeovers": rec["takeovers"],
        "collision_frames": rec["collision"],
        "route_completion_pct": round(completion, 2),
        "dist_to_gt_end_m": round(d_end, 2),
        "driving_score": round(ds, 3),
        "success": bool(sr),
        "efficiency_pct": None if eff is None else round(eff, 2),
        "comfort": round(comfort, 4),
        "oracle_gt_log": {
            "comfort": round(comfort_gt, 4),
            "efficiency_pct": None if eff_gt is None else round(eff_gt, 2),
        },
    }
    print("[probe] NavSafe metrics:", json.dumps(result, indent=2), flush=True)
    (out_dir / f"navsafe_metrics_{cfg.traffic_mode}.json").write_text(
        json.dumps(result, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
