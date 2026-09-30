# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Build closed-loop evaluation configuration from py123d Arrow inputs."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from navsafe.env.env_cfg import EnvCfg
from navsafe.env.presets import VISION_TRAIN
from navsafe.benchmark.scoring.from_run import SAFETY_CEILING_S

def build_eval_env_cfg(args: Any) -> EnvCfg:
    """Build an evaluation config, rejecting unsupported input formats."""
    source = getattr(args, "scenario_source", "py123d")
    if source != "py123d":
        raise ValueError(f"Unsupported scenario_source={source!r}; use 'py123d'.")
    return _build_py123d_cfg(args)


def _build_py123d_cfg(args: Any) -> EnvCfg:
    """Build a py123d symbolic scene configuration for NuRec gRPC rendering."""
    backend: Any = getattr(args, "render_backend", "nurec_grpc")
    nurec = backend == "nurec_grpc"
    cfg = dataclasses.replace(
        VISION_TRAIN,
        scenario_source="py123d",
        traffic_mode=getattr(args, "traffic_mode", "log_replay"),
        semi_reactive_takeover=getattr(
            args, "traffic_takeover", "continuous"),
        terminate_on_collision=getattr(
            args, "terminate_on_collision", False),
        py123d_data_root=args.py123d_data_root,
        py123d_frame_window=(
            tuple(args.py123d_frame_window)
            if getattr(args, "py123d_frame_window", None) is not None else None
        ),
        # NB: never cap the scene list here — the loader indexes into the FULL
        # list with start_scenario_index (a former py123d_max_scenes=1 silently
        # truncated the list so every --py123d-scene-index loaded scene 0).
        py123d_max_scenes=None,
        loop_replay=False,
        dt=args.sim_dt,
        # The env integrates the ego with PhysX only for the evaluator's
        # "physics" execution mode; teleport/controller use the kinematic env.
        execution_mode=(
            "physics"
            if getattr(args, "execution_mode", "teleport") == "physics"
            else "kinematic"
        ),
        spawn_ego_vehicle=True,
        required_bundles=[],
        render_backend=backend,
        nurec_work_dir=getattr(args, "nurec_work_dir", None),
        # Suppress the procedural scene for the NuRec backend — the
        # reconstruction is the scene.
        build_lanes=not nurec,
        build_lane_markings=not nurec,
        build_lane_boundaries=not nurec,
        spawn_agent_visuals=not nurec,
    )
    cfg.start_scenario_index = args.py123d_scene_index
    if getattr(args, "eval_seed", None) is not None:
        cfg.seed = int(args.eval_seed)
    cfg.remove_agents = getattr(args, "remove_agents", False)
    # Tri-state pass-through (None = auto: on iff physics mode). The env's
    # ValueError enforces the explicit-True/physics pairing loudly instead
    # of a silent downgrade here.
    cfg.contact_dynamics = getattr(args, "contact_dynamics", None)
    cfg.max_episode_steps = _episode_cap(args)
    return cfg


def _episode_cap(args: Any) -> int:
    """Steps the env may run before it truncates, warm-up included.

    ``--eval-frames`` unset means an INDEFINITE episode: it ends when the
    termination taxonomy ends it, or at the optional route clock. The env
    has to agree, because it truncates on its own count -- and the preset it is
    built from carries ``max_episode_steps=200``, i.e. 18 s of scored driving
    against a 60 s ceiling. Left alone, every episode with no taxonomy event
    stopped there and was classified ``trace_exhausted``: not attributed to the
    policy, and read downstream as a truncated run rather than a result.
    """
    warmup = int(getattr(args, "ego_replay_frames", 0) or 0)
    frames = getattr(args, "eval_frames", None)
    if frames is not None:
        return warmup + max(int(frames), 0)
    route_limit = getattr(args, "route_time_limit_s", SAFETY_CEILING_S)
    if (route_limit is None or float(route_limit) <= 0.0
            or not math.isfinite(float(route_limit))):
        # EnvCfg requires an integer. This is an implementation guard roughly
        # 6.8 years away at 10 Hz, not a scored route-completion time limit.
        return 2**31 - 1
    dt = float(getattr(args, "sim_dt", 0.1) or 0.1)
    return warmup + int(round(float(route_limit) / max(dt, 1e-6)))



__all__ = ["build_eval_env_cfg"]
