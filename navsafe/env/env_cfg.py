"""EnvCfg — the single configuration dataclass for NexusSimEnv.

Phase 3 task 3.17 of the NexusSim Package Reorg spec.
Requirements: 2.2, 2.3.

``EnvCfg`` selects every plugin boundary on :class:`NexusSimEnv` along
four axes (design §3):

* ``obs_kind``        — ``"state"`` or ``"sensor"``
* ``traffic_mode``    — ``"no_traffic"``, ``"log_replay"``, ``"semi_reactive"``,
  ``"navsafe"`` (``"idm"`` and ``"learned"`` are recognised literals but refused
  at env construction: unwired / future work)
* ``scenario_source`` — ``"py123d"``
* ``render_backend`` — ``"nurec_grpc"``

Episode parameters (max steps, dt, reward weights, etc.) are
placeholder fields at this HLD level; their full schema is deferred to
the LLD pass per design §13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class EnvCfg:
    """Configuration dataclass for :class:`~navsafe.env.NexusSimEnv`.

    The four axis fields are the sole source of plugin selection
    (Requirement 2.2). Episode parameters are extracted from the
    retired env classes during tasks 3.19–3.24; until then they carry
    sensible defaults.

    Field-level schema (types, units, tensor shapes, validation rules)
    is LLD per design §13 — this HLD pass establishes the axis
    literals and placeholder episode knobs only.
    """

    # ------------------------------------------------------------------
    # Plugin-selection axes (Requirement 2.3)
    # ------------------------------------------------------------------

    obs_kind: Literal["state", "sensor"] = "state"
    """Observation modality: ``"state"`` for structured ego/agents/map
    vectors, ``"sensor"`` for camera/depth/LiDAR images."""

    traffic_mode: Literal[
        "no_traffic", "log_replay", "idm", "semi_reactive", "navsafe", "learned"
    ] = "log_replay"
    """Traffic manager selection. ``"semi_reactive"`` is MetaDrive's
    ``reactive_traffic``: vehicles that appear behind the ego and inside
    ``semi_reactive_radius_m`` laterally follow their logged trajectory under
    IDM speed control, everything else replays. ``"navsafe"`` drives the
    actors a NavSafe recipe injected — IDM for vehicles, social force for
    pedestrians/animals — and, unlike every other mode, needs no USD stage,
    so it is the only one that runs under the closed-loop evaluator.
    ``"learned"`` is M1/M2 future work."""

    scenario_source: Literal["py123d"] = "py123d"
    """Evaluation data source: converted py123d Arrow logs."""

    render_backend: Literal["nurec_grpc"] = "nurec_grpc"
    """NuRec gRPC renderer for policy camera observations."""

    # ------------------------------------------------------------------
    # Episode parameters (placeholder — full schema is LLD per §13)
    # ------------------------------------------------------------------

    max_episode_steps: int = 200
    """Maximum number of simulation steps per episode."""

    seed: Optional[int] = None
    """Simulation seed; retained when representation overrides build this cfg."""

    dt: float = 0.1
    """Simulation timestep in seconds (10 Hz default).

    Legacy mapping: the evaluator and PG/eval scripts previously set
    ``ScenarioReplayEnvCfg.sim.dt`` (often via an ``args.sim_dt`` flag).
    On the consolidated env, callers map ``sim_dt`` → ``EnvCfg.dt``;
    there is no separate ``sim_dt`` field to avoid a confusing duplicate."""

    ego_max_speed: Optional[float] = None
    """Hard cap on ego speed (m/s) for the bicycle model.

    ``None`` (default) → derive from the log ego's peak speed at reset (with
    headroom, floored at the historical 15.0). This prevents a throttled ego
    from lagging behind — or being rear-ended by — faster ``log_replay``
    traffic. Set an explicit float to override (used for training/ablations)."""

    execution_mode: str = "kinematic"
    """How the ego integrates actions: ``"kinematic"`` (pure-Python bicycle
    model — the default and the only option without IsaacLab) or
    ``"physics"`` (PhysX rigid-body proxy integrates commanded velocities
    via ``sim.step()``; requires IsaacLab)."""

    contact_dynamics: Optional[bool] = None
    """Physical contact response between the ego physics proxy and replay
    agents: invisible box colliders track every replay agent and PhysX
    resolves ego contact impulses (a replay car that hits the ego pushes it
    instead of passing through). ``None`` (default) = auto: enabled
    whenever ``execution_mode="physics"`` (the only mode with a PhysX ego
    body). ``True`` = require it (construction raises ``ValueError``
    without physics mode). ``False`` = off."""

    scenario_path: Optional[str] = None
    """Optional generic data path. Prefer ``py123d_data_root``; the py123d
    loader falls back to this (then ``data_directory``) when it is unset."""

    data_directory: Optional[str] = None
    """Optional generic data directory; secondary fallback after
    ``py123d_data_root`` and ``scenario_path``."""

    num_envs: int = 1
    """Number of parallel environments (vectorized)."""


    py123d_data_root: Optional[str] = None
    """Root directory containing converted py123d Arrow logs and maps."""

    nurec_work_dir: Optional[str] = None
    """Optional reconstruction work directory for gRPC coordinate alignment.
    The Arrow scene remains the source of symbolic state."""

    scenario_edits: Optional[list] = None
    """Optional evaluation-time scenario edits — a list of
    ``{"tool": <name>, **params}`` specs applied to the loaded
    ``ScenarioDescription`` before the scene is built (see
    :mod:`navsafe.scenario.edits`). Example: ``[{"tool":
    "place_static_obstacles", "count": 6, "seed": 42}]`` drops six static
    vehicles on the ego's logged route to stress closed-loop avoidance."""

    py123d_max_scenes: Optional[int] = None
    """Optional cap on py123d scenes loaded by online RL samplers."""

    shuffle_scenarios: bool = False
    """Whether py123d scenario sampling should be shuffled."""

    scenario_seed: int = 0
    """Seed for deterministic py123d scenario sampling."""

    target_speed_mps: float = 8.0

    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "progress": 1.0,
        "comfort": -0.05,
        "collision": -10.0,
        "off_road": -5.0,
        "lane_keeping": 0.2,
        "speed": 0.1,
    })
    """Reward component weights for RL training."""

    # ------------------------------------------------------------------
    # Asset / rendering knobs
    # ------------------------------------------------------------------


    use_real_assets: bool = True
    """Whether to spawn real USD assets for agents. When ``False`` the
    replay manager falls back to cube proxies. Mirrors
    ``ScenarioReplayEnvCfg.use_real_assets`` (default ``True``); read
    directly by :meth:`NexusSimEnv._setup_scene` under IsaacSim."""


    vehicle_assets_path: Optional[str] = None
    """Path to vehicle USD assets directory."""

    #: Half-open ``[start, stop)`` py123d iteration window to materialise, or
    #: None for the whole scene. Set from a mined nuPlan scenario window so an
    #: episode loads the tagged situation instead of the entire log.
    py123d_frame_window: Optional[tuple[int, int]] = None

    semi_reactive_radius_m: float = 15.0
    """MetaDrive's ``IDM_CREATE_SIDE_CONSTRAINT``: the lateral half-width (m),
    in the ego's frame, of the region a vehicle must be in to be taken over
    by the trajectory-IDM policy. It must also be *behind* the ego (> 1 m)
    and be heading within 90 deg of it. Only used when
    ``traffic_mode="semi_reactive"``."""

    semi_reactive_takeover: str = "continuous"
    """When the takeover test runs: ``"continuous"`` (default) re-tests every
    non-reactive vehicle every frame, so a vehicle the ego passes becomes
    reactive at the frame it falls behind; ``"spawn"`` tests once at the
    vehicle's first valid frame and never again, which is MetaDrive's own
    behaviour and reproduces NexusSim results recorded before 2026-08-11.
    Takeover is one-way under both. Only used when
    ``traffic_mode="semi_reactive"``."""

    # ------------------------------------------------------------------
    # Replay parameters
    # ------------------------------------------------------------------

    loop_replay: bool = False
    """Whether to loop the scenario when reaching the end of the
    recorded trajectory. Only used when ``traffic_mode="log_replay"``.
    Matches ``ScenarioReplayEnvCfg.loop_replay``."""

    # ------------------------------------------------------------------
    # Scenario replay parameters (merged from scenario_replay_env_cfg.py)
    # ------------------------------------------------------------------


    spawn_ego_vehicle: bool = True
    """Whether to spawn/control the ego vehicle from scenario data.
    Mirrors the legacy ``ScenarioReplayEnvCfg`` default (``True``)."""

    spawn_pedestrians: bool = True
    """Whether to spawn pedestrian agents from scenario data."""

    spawn_cyclists: bool = True
    """Whether to spawn cyclist agents from scenario data."""


    spawn_agent_visuals: bool = True
    """Whether to spawn agent *visual* prims (meshes/cubes) into the USD scene.
    Set ``False`` when NuRec renders agent visuals; the symbolic agent states
    (used by collision + EPDMS) are kept regardless."""

    build_lanes: bool = True
    """Whether to build lane geometry in the scene."""

    build_lane_markings: bool = True
    """Whether to build lane marking geometry."""

    build_lane_boundaries: bool = True
    """Whether to derive white lane lines from per-lane boundary polylines
    (structured lane edges, in addition to the explicit painted road_lines)."""

    build_crosswalks: bool = True
    """Whether to build crosswalk geometry."""

    build_sidewalks: bool = True
    """Whether to build sidewalk geometry."""

    build_boundaries: bool = True
    """Whether to build road boundary geometry."""


    ego_agent_id: Optional[str] = None
    """Ego agent track ID. If None, uses sdc_id from scenario metadata."""

    asset_selection_strategy: str = "hash"
    """Asset selection strategy: ``"hash"``, ``"random"``, ``"sequential"``,
    or ``"weighted"``."""

    asset_random_seed: Optional[int] = None
    """Random seed for reproducible asset selection."""


    camera_resolution_scale: float = 1.0
    """Multiplier applied to requested camera width/height before capture
    (``1.0`` = unchanged). Trades resolution for render speed during eval;
    intrinsics are FOV-based so framing is preserved. Read by
    the NuRec camera configuration."""




    # ------------------------------------------------------------------
    # Optional ego sensor-enable slots (legacy compat)
    #
    # Each is opt-in: leave ``None`` to disable, or assign a sensor cfg
    # instance from ``navsafe.component.sensors`` to enable. Typed as
    # ``Optional[Any]`` (not the concrete cfg classes) on purpose — those
    # classes use IsaacLab's ``@configclass`` and would pull IsaacSim in
    # at import time, breaking the pure-Python import-safety guarantee of
    # this module. The runtime resolves/validates them when IsaacSim is
    # present. Mirrors the ``ego_*`` slots on ``ScenarioReplayEnvCfg``.
    # ------------------------------------------------------------------

    ego_lidar: Optional[Any] = None
    """Optional ego LiDAR sensor cfg. ``None`` disables; assign an
    ``EgoLidarCfg`` from ``navsafe.component.sensors`` to enable."""

    ego_imu: Optional[Any] = None
    """Optional ego IMU sensor cfg. ``None`` disables; assign an
    ``EgoImuCfg`` from ``navsafe.component.sensors`` to enable."""

    ego_contact: Optional[Any] = None
    """Optional ego contact sensor cfg. ``None`` disables; assign an
    ``EgoContactSensorCfg`` from ``navsafe.component.sensors`` to enable."""


    ego_frame_transformer: Optional[Any] = None
    """Optional ego frame-transformer cfg. ``None`` disables; assign an
    ``EgoFrameTransformerCfg`` from ``navsafe.component.sensors`` to enable."""

    # ------------------------------------------------------------------
    # Eval/replay compatibility fields (merged from the retired
    # ``ScenarioReplayEnvCfg``). NexusSimEnv reads these via ``getattr`` with
    # matching defaults; declaring them here makes EnvCfg the complete,
    # explicit replacement for the retired cfg surface the eval entry points set.
    # ------------------------------------------------------------------

    scenario_id: Optional[str] = None
    """Exact py123d scene uuid / log name to load (overrides
    ``start_scenario_index`` when set). Read by ``Py123DLoader``."""

    start_scenario_index: int = 0
    """Positional index into the (filtered) py123d scene list. Read by
    ``Py123DLoader`` when ``scenario_id`` is unset."""

    remove_agents: bool = False
    """Drop all non-ego tracks at load time so removed objects vanish from
    the sim state (BEV / collision / observation). Read by ``Py123DLoader``."""

    terminate_on_collision: bool = False
    """Whether an ego collision terminates the episode (eval knob)."""




    # ------------------------------------------------------------------
    # Required asset bundles (used by NexusSimEnv.reset() to detect
    # missing assets — Requirement 11.4)
    # ------------------------------------------------------------------

    required_bundles: List[str] = field(default_factory=list)
    """Asset bundle names that must be present in the local cache for
    this configuration to function. Populated by EnvCfg presets."""
