"""NexusSimEnv — the single configurable environment for NexusSim.

Phase 3 tasks 3.17, 3.25 of the NexusSim Package Reorg spec.
Requirements: 2.1, 2.2, 2.4, 2.6, 11.3, 11.4, 11.7.

Task 3.25 folds ego-dynamics/collision/renderer wiring from
``base_driving_env.py``, training step/reset/reward logic from
``training_env.py``, and manager-based patterns from
``abstract_env.py`` / ``abstract_rl_env.py`` into this unified env.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np

if TYPE_CHECKING:
    from navsafe.scenario.scenario_description import ScenarioDescription

# ── Lazy imports: IsaacLab ────────────────────────────────────────────
try:
    from isaaclab.envs import DirectRLEnv
    _HAS_ISAACLAB = True
except ImportError:  # pragma: no cover
    DirectRLEnv = object  # type: ignore[assignment,misc]
    _HAS_ISAACLAB = False

# ── Lazy imports: gymnasium ───────────────────────────────────────────
try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
    _HAS_GYMNASIUM = True
except ImportError:  # pragma: no cover
    gymnasium = None  # type: ignore[assignment]
    gym_spaces = None  # type: ignore[assignment]
    _HAS_GYMNASIUM = False

from navsafe.env.env_cfg import EnvCfg
from navsafe.core.ego_dynamics import EgoDynamics, EgoDynamicsCfg, CollisionDetector
from navsafe.core.track_dims import track_dims, track_height

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap helpers (extracted to navsafe/env/_bootstrap.py).
#
# The stateless, env-instance-free bootstrap logic — the supported-axis table,
# the loader/traffic-manager factories, config validation, asset-bundle checks,
# the default agent-dimension table, and the IsaacLab cfg augmentation — now
# lives in `_bootstrap.py`. It is re-exported here so the historical import
# paths keep working unchanged, e.g.:
#     from navsafe.env.navsafe_env import SUPPORTED_COMBINATIONS
#     from navsafe.env.navsafe_env import _validate_cfg, _check_asset_bundles
# (used by navsafe.text2sim.agent_utils.sim_runnable and tests/env/).
# ---------------------------------------------------------------------------
from navsafe.env._bootstrap import (  # noqa: E402  (after the lazy-import guards above)
    _DEFAULT_AGENT_DIMS,
    SUPPORTED_COMBINATIONS,
    _augment_cfg_for_isaaclab,
    _check_asset_bundles,
    _format_supported_combinations,
    _get_cache_root,
    _make_loader,
    _make_traffic_manager,
    _validate_cfg,
)

# Collaborator subsystems extracted from this God class (each owns its own
# state behind a narrow interface; the env composes + delegates to them).
from navsafe.env.sensor_rig import SensorRig  # noqa: E402
from navsafe.env.camera_manager import CameraManager  # noqa: E402
from navsafe.env.scene_builder import SceneBuilder  # noqa: E402


def _derive_ego_max_speed(velocity: Any, valid: Any, default: float = 15.0,
                          margin: float = 1.15) -> float:
    """Ego speed cap from the log ego's peak speed (with headroom).

    Returns ``max(default, peak_log_speed * margin)`` so the ego can keep pace
    with — and is not overtaken by — faster ``log_replay`` traffic, while never
    dropping below the historical ``default`` for slow scenes. Falls back to
    ``default`` when velocity data is missing/degenerate.
    """
    if velocity is None:
        return default
    vel = np.asarray(velocity, dtype=np.float64)
    if vel.ndim != 2 or vel.shape[0] == 0 or vel.shape[1] < 2:
        return default
    speeds = np.hypot(vel[:, 0], vel[:, 1])
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape == speeds.shape and mask.any():
            speeds = speeds[mask]
    if speeds.size == 0 or not np.isfinite(speeds).any():
        return default
    peak = float(np.nanmax(speeds))
    return max(default, peak * margin)


# ======================================================================
# NexusSimEnv
# ======================================================================


class NexusSimEnv(DirectRLEnv):  # type: ignore[misc]
    """The single configurable NexusSim environment.

    Replaces the seven retired env classes. Folds ego dynamics,
    collision detection, renderer wiring (from base_driving_env.py),
    training step/reset/reward (from training_env.py), and
    manager-based patterns (from abstract_env.py, abstract_rl_env.py).
    """

    cfg: EnvCfg

    # Agent-id sentinel used for back-compat with the ego-only camera API.
    # Any call site that still passes ``agent_id="__ego__"`` (or omits it)
    # continues to land on the historic ``/World/EgoCam_*`` prim path; new
    # per-agent rigs use ``/World/AgentCam/{agent_id}/{cam_name}``. Ported
    # verbatim from ``ScenarioReplayEnv.EGO_AGENT_ID``.
    EGO_AGENT_ID: str = "__ego__"

    def __init__(self, cfg: EnvCfg, **kwargs: Any) -> None:
        """Initialize NexusSimEnv.

        Args:
            cfg: Environment configuration dataclass.
            **kwargs: Forwarded to DirectRLEnv.__init__ when available.

        Raises:
            ValueError: If the EnvCfg axis combination is unsupported.
        """
        _validate_cfg(cfg)
        self.cfg = cfg

        # ── Ego dynamics (from base_driving_env.py) ───────────────────
        ego_cfg = EgoDynamicsCfg(
            dt=cfg.dt,
            max_speed=cfg.ego_max_speed if cfg.ego_max_speed is not None else 15.0,
            max_steer_angle=0.6,
            max_accel=3.0,
            max_brake=5.0,
            wheelbase=2.8,
            ego_length=4.5,
            ego_width=1.8,
            ego_height=1.5,
        )
        self._ego = EgoDynamics(ego_cfg)

        # ── Agent states (from base_driving_env.py) ───────────────────
        self._agent_states: List[Dict] = []
        self._scenario_data: Optional[Dict] = None

        # ── Renderer (from base_driving_env.py) ───────────────────────
        self._renderer: Any = None
        # Identity of the scenario currently installed in the renderer.
        # This is deliberately distinct from _built_scenario_id: NuRec carries
        # scenario-specific cameras/maps/tracks that must switch on reset.
        self._renderer_scenario_id: Optional[str] = None

        # ── Training state (from training_env.py) ─────────────────────
        self._prev_position: Optional[np.ndarray] = None
        self._episode_step: int = 0
        self._collision: bool = False
        self._collision_at_fault: bool = False
        self._contact_detail: Dict[str, Any] = {}

        # ── Gymnasium spaces (from training_env.py / abstract_env.py) ─
        self._action_space: Any = None
        if _HAS_GYMNASIUM:
            self._action_space = gym_spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
            )

        # ── Episode counters (from abstract_env.py / abstract_rl_env.py)
        self._common_step_counter: int = 0

        # ── Scenario replay state (from scenario_replay_env.py) ───────
        # Tracks the current position in the scenario log for log_replay
        # traffic mode. Reset to 0 on env.reset().
        self._scenario_timestep: int = 0

        # ── Ego override state (from scenario_replay_env.py) ──────────
        # Set by the evaluator (set_ego_override) to drive the ego along a
        # model-planned trajectory instead of replaying it from the log.
        # _ego_externally_driven marks controller/physics execution modes
        # (ego driven by actions; replay + prim posing must follow self._ego).
        self._ego_externally_driven: bool = False
        self._ego_override_active: bool = False
        self._ego_override_state: Optional[Dict[str, Any]] = None

        # ── Traffic manager (selected on cfg.traffic_mode) ────────────
        # The episode-level traffic manager (NoTrafficManager /
        # LogReplayTraffic / SemiReactiveTraffic / NavSafeTraffic) replaces
        # the inline IDM controller. Constructed eagerly here — the axis
        # combination has already been validated, so the modes that raise
        # (``learned``: future work; ``idm``: recognised but unwired, see
        # _bootstrap.IDM_TRAFFIC_UNWIRED_MSG) never reach this point. The
        # managers are pure-Python and do not pull in IsaacSim, so this is
        # import-safe.
        self._traffic_manager: Any = _make_traffic_manager(cfg)

        # ── Ego sensor subsystem (LiDAR / IMU / contact / tiled-camera /
        # frame-transformer) ─────────────────────────────────────────────
        # Constructed eagerly here (before super().__init__) because
        # DirectRLEnv.__init__ calls _setup_scene() → _setup_extra_sensors()
        # during construction, so the rig must already exist. The rig owns the
        # sensor handles; the env exposes them via read-only properties below.
        self._sensors = SensorRig(self)

        # ── Scene-construction subsystem ─────────────────────────────────
        # Eager (before super().__init__): the env's _setup_scene hook runs
        # during DirectRLEnv.__init__ and delegates the USD scene-build steps
        # here, and SensorRig.setup() reaches _build_combined_lidar_mesh through
        # it. Near-stateless — the scene STATE (map_builder)
        # stays on the env (owned by _setup_scene); SceneBuilder forwards to it.
        self._scene = SceneBuilder(self)
        # NB: the camera subsystem (self._cameras) is a LAZY property below, not
        # constructed here — capture only happens post-construction, and lazy
        # init keeps the camera wrappers working on minimally-built instances
        # (e.g. ``object.__new__(NexusSimEnv)`` in the renderer-wiring tests).

        # ── Physics execution mode (PhysX-integrated ego) state ────────
        # Populated by _author_ego_physics_proxy (during _setup_scene) and
        # _init_ego_physics_view (after super().__init__).
        self._ego_phys_proxy_path: Optional[str] = None
        self._ego_phys_view: Any = None
        self._ego_phys_indices: Any = None
        # Contact-dynamics state (agent collider pool). Populated by
        # _author_agent_contact_proxies / _init_agent_contact_view.
        # cfg.contact_dynamics is tri-state: None = auto (on iff physics
        # mode), True = required (raise without physics), False = off.
        self._physical_contact: bool = False
        self._contact_view: Any = None
        self._contact_slots: List[Dict[str, Any]] = []
        _cd = getattr(cfg, "contact_dynamics", None)
        if _cd and cfg.execution_mode != "physics":
            raise ValueError(
                "EnvCfg.contact_dynamics=True requires execution_mode="
                "'physics' — teleport/controller/kinematic egos have no "
                "PhysX body to push.")
        self._contact_dynamics: bool = (
            cfg.execution_mode == "physics" if _cd is None else bool(_cd))
        if cfg.execution_mode == "physics" and not _HAS_ISAACLAB:
            raise ValueError(
                "EnvCfg.execution_mode='physics' requires IsaacLab — the ego "
                "is integrated by PhysX. Use 'kinematic' without the stack.")

        # IsaacLab 3.0's DirectRLEnv.__init__ calls self._setup_scene() *during*
        # construction (via _init_sim), so all pure-Python state the scene build
        # touches (e.g. self._scenario_data) must already exist — hence super()
        # is invoked last, after the attribute initialisation above.
        if _HAS_ISAACLAB:
            _augment_cfg_for_isaaclab(cfg)
            super().__init__(cfg, **kwargs)
            if cfg.execution_mode == "physics":
                self._init_ego_physics_view()
                if self._contact_dynamics:
                    self._init_agent_contact_view()

        logger.info(
            "NexusSimEnv initialized: obs_kind=%s, traffic_mode=%s, "
            "scenario_source=%s, render_backend=%s",
            cfg.obs_kind, cfg.traffic_mode,
            cfg.scenario_source, cfg.render_backend,
        )

    # ------------------------------------------------------------------
    # Properties (from base_driving_env.py / training_env.py)
    # ------------------------------------------------------------------

    @property
    def ego_x(self) -> float:
        return self._ego.x

    @ego_x.setter
    def ego_x(self, v: float) -> None:
        self._ego.x = v

    @property
    def ego_y(self) -> float:
        return self._ego.y

    @ego_y.setter
    def ego_y(self, v: float) -> None:
        self._ego.y = v

    @property
    def ego_heading(self) -> float:
        return self._ego.heading

    @ego_heading.setter
    def ego_heading(self, v: float) -> None:
        self._ego.heading = v

    @property
    def ego_speed(self) -> float:
        return self._ego.speed

    @ego_speed.setter
    def ego_speed(self, v: float) -> None:
        self._ego.speed = v

    @property
    def agent_states(self) -> List[Dict]:
        return self._agent_states

    @agent_states.setter
    def agent_states(self, v: List[Dict]) -> None:
        self._agent_states = v

    @property
    def scenario_data(self) -> Optional[Dict]:
        return self._scenario_data

    @scenario_data.setter
    def scenario_data(self, v: Optional[Dict]) -> None:
        self._scenario_data = v

    @property
    def current_scenario(self) -> Optional[Dict]:
        """The active scenario description (alias of :attr:`scenario_data`).

        The evaluator reads ``env.current_scenario`` to drive scoring,
        route generation and EPDMS. It is the same underlying
        ``ScenarioDescription`` as :attr:`scenario_data`, populated by the
        loader selected on ``cfg.scenario_source`` at :meth:`reset`.
        """
        return self._scenario_data

    @current_scenario.setter
    def current_scenario(self, v: Optional[Dict]) -> None:
        self._scenario_data = v

    @property
    def frame(self) -> int:
        return self._ego.frame

    @property
    def scenario_timestep(self) -> int:
        """Current replay cursor into the scenario log.

        Read by the traffic managers (e.g.
        :class:`~navsafe.traffic.log_replay.LogReplayTraffic`) to advance
        non-ego agents to the current frame. The env owns timestep
        advancement (incremented each :meth:`step`, reset to 0 each
        :meth:`reset`)."""
        return self._scenario_timestep

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in steps (from abstract_env.py)."""
        return self.cfg.max_episode_steps

    @property
    def device(self) -> Any:
        """Torch device for tensors created against this env.

        Part of the evaluator contract: ``unified_evaluator`` /
        ``Evaluator`` read ``getattr(env, "device", "cpu")`` to build the
        action tensor passed to :meth:`step`. On the IsaacLab path,
        ``DirectRLEnv`` owns the device (derived from the running sim), so
        defer to it; on the pure-Python path (no IsaacSim) there is no GPU
        sim, so report ``"cpu"``.
        """
        if _HAS_ISAACLAB:
            try:
                return super().device  # DirectRLEnv.device → self.sim.device
            except AttributeError:  # pragma: no cover — sim not yet built
                return "cpu"
        return "cpu"

    # ------------------------------------------------------------------
    # Ego sensor handles — delegate to the SensorRig subsystem.
    # Read-only: the rig is the sole writer (in SensorRig.setup()). Exposed
    # on the env so existing call sites and tests (e.g. ``env.ego_lidar_sensor
    # is None``) keep working unchanged.
    # ------------------------------------------------------------------
    @property
    def ego_lidar_sensor(self) -> Any:
        return self._sensors.ego_lidar_sensor

    @property
    def ego_imu_sensor(self) -> Any:
        return self._sensors.ego_imu_sensor

    @property
    def ego_contact_sensor(self) -> Any:
        return self._sensors.ego_contact_sensor


    @property
    def ego_frame_transformer_sensor(self) -> Any:
        return self._sensors.ego_frame_transformer_sensor


    @property
    def _cameras(self) -> CameraManager:
        """The camera / rendering subsystem (lazily constructed on first use).

        Lazy (not built in ``__init__``) so the delegating camera wrappers work
        even on partially-built instances created via ``object.__new__`` — the
        renderer-wiring tests do exactly that. Stored under a distinct
        ``__dict__`` key to avoid colliding with this property name.
        """
        mgr = self.__dict__.get("_camera_manager")
        if mgr is None:
            mgr = CameraManager(self)
            self.__dict__["_camera_manager"] = mgr
        return mgr


    # ------------------------------------------------------------------
    # Ego dynamics delegation (from base_driving_env.py)
    # ------------------------------------------------------------------

    def step_ego(self, steer: float, accel: float) -> Dict[str, Any]:
        """Advance ego one timestep using bicycle model.

        NUREC_FREEZE_EGO env var: hold the ego pose in place (do not integrate)
        while scenario time keeps advancing — a diagnostic for whether dynamic
        actors move independently of ego motion.
        """
        if os.environ.get("NUREC_FREEZE_EGO"):
            self._ego.frame += 1  # keep the frame counter truthful
            return self._ego.get_state()
        return self._ego.step(steer, accel)

    def get_ego_state(self) -> Dict[str, Any]:
        """Return current ego state."""
        return self._ego.get_state()

    # ------------------------------------------------------------------
    # Collision detection (from base_driving_env.py)
    # ------------------------------------------------------------------

    def check_collision(self) -> bool:
        """Check if ego collides with any agent."""
        return CollisionDetector.check_collision(
            ego_x=self._ego.x,
            ego_y=self._ego.y,
            ego_heading=self._ego.heading,
            ego_length=self._ego.cfg.ego_length,
            ego_width=self._ego.cfg.ego_width,
            agent_states=self._agent_states,
        )

    def check_collision_at_fault(self) -> Tuple[bool, bool]:
        """Collision check with NavSim at-fault classification.

        Returns ``(collided, at_fault)`` — a rear-end by a following agent or
        contact while the ego is stopped is a collision but not at fault.
        """
        return CollisionDetector.check_collision_at_fault(
            ego_x=self._ego.x,
            ego_y=self._ego.y,
            ego_heading=self._ego.heading,
            ego_length=self._ego.cfg.ego_length,
            ego_width=self._ego.cfg.ego_width,
            agent_states=self._agent_states,
            ego_speed=self._ego.speed,
        )

    # ------------------------------------------------------------------
    # Renderer wiring (from base_driving_env.py)
    # ------------------------------------------------------------------

    def _ensure_renderer(self) -> None:
        """Lazily initialize the renderer based on cfg.render_backend."""
        if self._renderer is None:
            from navsafe.render import create_renderer
            self._renderer = create_renderer(self.cfg.render_backend)

    @property
    def renderer(self) -> Any:
        """The active scene renderer (lazily created on first access).

        Exposed under the name ``renderer`` so the ported sensor/camera
        methods (``get_camera_images`` / ``get_agent_camera_images``) match
        the ``ScenarioReplayEnv`` source verbatim, which references
        ``self.renderer``.
        """
        self._ensure_renderer()
        return self._renderer

    @renderer.setter
    def renderer(self, value: Any) -> None:
        self._renderer = value

    def get_camera_images(self, cam_configs: Dict) -> Dict[str, np.ndarray]:
        return self._cameras.get_camera_images(cam_configs)

    def get_camera_images_at_pose(
        self, observer_pose: Dict, cam_configs: Dict,
    ) -> Dict[str, np.ndarray]:
        """Render the ego rig from a hypothetical ego pose (sim untouched).

        See :meth:`CameraManager.get_camera_images_at_pose`.
        """
        return self._cameras.get_camera_images_at_pose(observer_pose, cam_configs)

    # ------------------------------------------------------------------
    # Sensor + camera-capture methods (ported verbatim from
    # ScenarioReplayEnv — task 10). All IsaacSim/omni/replicator imports
    # are lazy (inside method bodies) so the module stays import-safe
    # without IsaacSim. Adapted only where ``ScenarioReplayEnvCfg`` field
    # names differ from the equivalent ``EnvCfg`` slots.
    # ------------------------------------------------------------------

    def get_agent_camera_images(self, cam_configs_by_agent: Dict[str, Dict[str, Dict]], capture_budget: Optional[int]=None) -> Dict[Tuple[str, str], np.ndarray]:
        return self._cameras.get_agent_camera_images(cam_configs_by_agent, capture_budget)

    def _resolve_agent_world_pose(self, agent_id: str) -> Optional[Dict]:
        """Return ``{'position': (3,), 'heading': float, ...}`` in world frame.

        ``agent_id == EGO_AGENT_ID`` dispatches to :meth:`get_ego_state` so
        any active ego override (e.g. the evaluator teleport) is honoured.
        Any other id is resolved via
        :meth:`ScenarioReplayManager.get_agent_state` at
        ``self.scenario_timestep``.

        Returns ``None`` if the agent is unknown or invalid at the current
        timestep.  The returned dict is ``get_ego_state``-compatible so it
        can be passed directly to :meth:`CameraManager._position_camera` and to
        renderer entry points that accept ``ego_state``.

        Kept on the env (not the CameraManager) because it is a data-provider
        seam — it reads ``agent_manager`` (env-owned) and is monkeypatched in
        tests alongside ``get_ego_state`` / ``_collect_agent_states_for_renderer``.
        The camera subsystem reaches it via ``__getattr__`` forwarding.
        """
        if agent_id == self.EGO_AGENT_ID:
            return self.get_ego_state()

        if not hasattr(self, "agent_manager") or self.agent_manager is None:
            return None

        state = self.agent_manager.get_agent_state(agent_id, self.scenario_timestep)
        if state is None:
            return None
        if not state.get("valid", True):
            return None

        pos = np.asarray(state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
        if len(pos) < 3:
            pos = np.array([pos[0], pos[1], 0.0], dtype=np.float32)
        heading = float(state.get("heading", 0.0))
        vel_raw = np.asarray(state.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
        vx = float(vel_raw[0]) if vel_raw.size >= 1 else 0.0
        vy = float(vel_raw[1]) if vel_raw.size >= 2 else 0.0
        vz = float(vel_raw[2]) if vel_raw.size >= 3 else 0.0
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        return {
            "position": pos,
            "heading":  heading,
            "speed":    speed,
            "velocity": np.array([vx, vy, vz], dtype=np.float32),
            "timestep": self.scenario_timestep,
        }




    def _scale_cam_configs(self, cam_configs: Dict) -> Dict:
        return self._cameras._scale_cam_configs(cam_configs)


    @staticmethod
    def _cfg_num(cfg: Any, name: str, default, cast=float):
        """Read a numeric cfg field, falling back to ``default`` when the field
        is absent OR explicitly ``None``. An explicit ``0`` is preserved so
        "disable" semantics still work (unlike an ``or default`` guard, which
        treats ``0`` as missing)."""
        val = getattr(cfg, name, default)
        return default if val is None else cast(val)


    # ------------------------------------------------------------------
    # Reward computation (from training_env.py)
    # ------------------------------------------------------------------

    def _compute_reward(self, steer: float, accel: float) -> Tuple[float, dict]:
        """Compute reward components."""
        w = self.cfg.reward_weights
        curr_pos = np.array([self._ego.x, self._ego.y])
        if self._prev_position is not None:
            progress = float(np.linalg.norm(curr_pos - self._prev_position))
        else:
            progress = 0.0

        comfort = abs(steer) * 0.5 + abs(accel) * 0.3
        collision_penalty = 1.0 if self._collision else 0.0
        off_road = 0.0
        lane_keeping = 0.0
        speed_reward = self._ego.speed / 15.0

        total = (
            w.get("progress", 1.0) * progress
            + w.get("comfort", -0.05) * comfort
            + w.get("collision", -10.0) * collision_penalty
            + w.get("off_road", -5.0) * off_road
            + w.get("lane_keeping", 0.2) * lane_keeping
            + w.get("speed", 0.1) * speed_reward
        )

        info = {
            "reward_progress": progress,
            "reward_comfort": comfort,
            "reward_collision": collision_penalty,
            "reward_off_road": off_road,
            "reward_lane_keeping": lane_keeping,
            "reward_speed": speed_reward,
            "reward_total": total,
        }
        return total, info

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def _loader_has_input(self) -> bool:
        """Whether an Arrow data root is configured."""
        return bool(
            getattr(self.cfg, "py123d_data_root", None)
            or self.cfg.scenario_path
            or self.cfg.data_directory
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Checks for missing asset bundles (Requirement 11.4).
        Initializes ego from scenario data when available.

        Args:
            seed: Optional random seed.
            options: Optional reset options. May contain "scenario_data".

        Returns:
            Tuple of (observation, info_dict).
        """
        _check_asset_bundles(self.cfg)

        if options and "scenario_data" in options:
            # Explicit injection takes precedence over loader dispatch.
            self._scenario_data = options["scenario_data"]
            self._setup_renderer_for_scenario(self._scenario_data, force=True)
        elif self._scenario_data is None:
            # No explicit scenario and none loaded yet — dispatch on
            # cfg.scenario_source to the matching loader (Req 2.4, 3.4,
            # 3.5, 4.1). _make_loader raises ValueError on an unknown
            # source. The loader only runs when the source actually has
            # input configured, preserving the default-ego reset for a bare
            # py123d cfg with no data root.
            loader = _make_loader(self.cfg)
            if self._loader_has_input():
                self._scenario_data = self._post_load_scenario(loader.load(self.cfg))

        # Keep the NuRec reconstruction aligned with the loaded scenario.
        if self._scenario_data is not None:
            self._setup_renderer_for_scenario(self._scenario_data)

        # Reset ego from scenario data (from training_env.py)
        if self._scenario_data and "tracks" in self._scenario_data:
            meta = self._scenario_data.get("metadata", {})
            sdc_id = str(meta.get("sdc_id", "ego"))
            tracks = self._scenario_data.get("tracks", {})
            ego_track = tracks.get(sdc_id, {})
            state = ego_track.get("state", {})
            pos = state.get("position")
            heading = state.get("heading")
            # Derive the ego speed cap from the log ego's peak speed so a
            # throttled ego does not lag behind (or get rear-ended by) faster
            # log_replay traffic. Skipped when ego_max_speed is set explicitly.
            if self.cfg.ego_max_speed is None:
                self._ego.cfg.max_speed = _derive_ego_max_speed(
                    state.get("velocity"), state.get("valid"), default=15.0)
            if pos is not None and len(pos) > 0:
                # Seed the ego with the LOG entry speed, not a standing
                # start: log_replay traffic is blind, so an ego spawned at
                # 0 m/s inside a moving platoon is rear-ended while driving
                # legitimately (27643710 enters at 5.6 m/s; the trailing
                # replay agent closed the gap by f25 in every 0-start run).
                # Logs that genuinely start at rest keep speed 0.
                vel = state.get("velocity")
                valid = state.get("valid")
                v0 = 0.0
                if vel is not None and len(vel) > 0 and (
                        valid is None or len(valid) == 0 or bool(valid[0])):
                    v0 = float(np.linalg.norm(np.asarray(vel[0])[:2]))
                self._ego.reset(
                    x=float(pos[0][0]), y=float(pos[0][1]),
                    heading=float(heading[0]) if heading is not None else 0.0,
                    speed=v0,
                )
            else:
                self._ego.reset()
        else:
            self._ego.reset()

        self._prev_position = np.array([self._ego.x, self._ego.y])
        self._episode_step = 0
        self._collision = False
        self._collision_at_fault = False
        self._contact_detail = {}
        self._agent_states = []
        self._scenario_timestep = 0
        # Which tracks have already been admitted (see _spawn_blocked_by_ego).
        # Per-episode: without the reset a second episode in the same env would
        # treat every agent as long since spawned and never run the guard.
        self._spawned_agents: set[str] = set()
        self._spawn_held_logged: set[str] = set()
        # A leaked ego override (e.g. an abort between set_ego_override and
        # clear_ego_override) would silently pin the ego to the previous
        # episode's pose for the whole next episode — drop it on reset.
        self.clear_ego_override()

        # Physics execution mode: sync the PhysX proxy to the reset pose
        # with zero velocity so the first sim.step integrates from rest.
        if self._ego_phys_view is not None:
            self._write_ego_proxy_state(
                self._ego.x, self._ego.y, self._ego.heading)
        self._physical_contact = False
        if getattr(self, "_contact_view", None) is not None:
            self._reset_agent_contact_proxies()

        obs = self._get_observations()
        info: Dict[str, Any] = {"frame": self.frame}
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one simulation step.

        Integrates step logic from training_env.py and base_driving_env.py.
        For log_replay traffic mode, advances the scenario timestep and
        supports loop_replay to restart from the beginning.

        Args:
            action: Agent action, shape (2,) for [steer, accel].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        if action is None:
            action = np.zeros(2, dtype=np.float32)
        # The evaluator passes a torch tensor (often on cuda); bring it to host
        # before numpy conversion.
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).flatten()
        steer = float(action[0]) if len(action) > 0 else 0.0
        accel = float(action[1]) if len(action) > 1 else 0.0

        # Step ego dynamics. Three drivers, in precedence order:
        #   1. Active override (teleport execution / GT warm-up): the override
        #      pose is authoritative — integrating on top of it would advance
        #      the ego twice. In physics mode the override also teleports the
        #      PhysX proxy so its state stays continuous at handoff.
        #   2. Physics execution mode: the bicycle model computes a velocity
        #      command, PhysX integrates it (sim.step), the pose is read back.
        #   3. Kinematic (default): the bicycle model integrates directly.
        if self._ego_override_active and self._ego_override_state is not None:
            self._sync_base_ego_state()
            # step_ego is skipped, so advance the ego frame counter here to
            # keep env.frame / info["frame"] truthful during overrides.
            self._ego.frame += 1
            if self._ego_phys_view is not None:
                ov = self._ego_override_state
                vel = np.asarray(ov.get("velocity", np.zeros(3)), dtype=np.float64)
                self._write_ego_proxy_state(
                    self._ego.x, self._ego.y, self._ego.heading,
                    lin_vel=(float(vel[0]), float(vel[1])))
            state = {"speed_kmh": self._ego.speed * 3.6}
        elif self._ego_phys_view is not None:
            state = self._step_ego_physics(steer, accel)
        else:
            state = self.step_ego(steer, accel)
        self._episode_step += 1
        self._common_step_counter += 1
        self._scenario_timestep += 1

        # Advance reactive traffic BEFORE the agent list is rebuilt, so the
        # list carries this frame's decisions rather than last frame's.
        #
        # A stage-free manager is stepped here rather than from
        # _advance_replay_agent_prims because that hook returns at its first
        # line when IsaacLab is unavailable — which, with isaaclab_contrib
        # missing from the venv, is always. Advancing an agent is a decision;
        # posing its prim is a rendering detail. The prim hook owns the
        # second, and used to silently own the first as well.
        _tm = getattr(self, "_traffic_manager", None)
        # `_tm is not None` is already implied by the getattr default below
        # (None has no `stage_free`); spelled out so the checker sees it too.
        if _tm is not None and getattr(_tm, "stage_free", False):
            _tm.step(self, self.cfg.dt)

        # Update agents (traffic manager hook)
        self._update_agents()

        # Drive the IsaacSim/USD agent prims to the current replay frame.
        # The Evaluator advances the env through step() (not the DirectRLEnv
        # _pre_physics_step hook), so without this the replay manager never
        # re-poses the non-ego prims: they freeze at their frame-0 spawn pose
        # (transform *and* visibility) while the ego camera follows the ego,
        # making actors look like a static overlay. No-op off IsaacSim.
        self._advance_replay_agent_prims()

        # Refresh the symbolic agent list every mode that is NOT log_replay.
        # _update_agents() above returns early for those modes, and the only
        # other writer is _pre_physics_step, which the Evaluator never calls —
        # so under semi_reactive the list stayed at reset()'s [] for the whole
        # episode and every agent-dependent metric (collision, TTC) scored
        # against an empty world.
        if (self.cfg.traffic_mode != "log_replay"
                and not getattr(getattr(self, "_traffic_manager", None), "stage_free", False)
                and self._scenario_data is not None):
            self._agent_states = self._collect_agent_states_for_renderer()

        # Hand the renderer this frame's agent poses. Without this the
        # renderer only ever hears from _pre_physics_step, which the
        # Evaluator does not call, so nurec_grpc falls back to the poses it
        # cached at scene load: a reactive actor would yield in the collision
        # and TTC numbers while sailing along its spawn-time rail on camera.
        renderer = getattr(self, "_renderer", None)
        if renderer is not None and hasattr(renderer, "update_agents"):
            try:
                renderer.update_agents(self._agent_states)
            except Exception:  # noqa: BLE001 — a render hiccup must not end the episode
                logger.exception("failed to hand agent states to the renderer")

        # Check collision.
        self._collision, self._collision_at_fault = self.check_collision_at_fault()
        # A physical contact impulse is a collision even if the symbolic OBB
        # check missed it (e.g. sub-frame overlap resolved by PhysX).
        # Fault attribution stays symbolic — the impulse carries no
        # relative-geometry classification.
        # getattr: minimally-built instances (object.__new__ in env tests)
        # never ran __init__ and lack the contact-dynamics attributes.
        self._collision = self._collision or bool(
            getattr(self, "_physical_contact", False))
        if self._collision:
            self._contact_detail = self._describe_contact()

        # Compute reward
        reward, reward_info = self._compute_reward(steer, accel)

        # Done conditions. terminate_on_collision widens termination to ANY
        # ego-box contact (including not-at-fault overruns by replay traffic).
        # getattr: test stubs pass minimal cfg namespaces without the field.
        terminated = self._collision_at_fault or (
            bool(getattr(self.cfg, "terminate_on_collision", False))
            and self._collision)

        # Truncation: episode ends when max steps reached.
        # For log_replay with loop_replay=True, the scenario loops
        # indefinitely (matching ScenarioReplayEnv behavior where
        # loop_replay prevents truncation).
        if self.cfg.traffic_mode == "log_replay" and self.cfg.loop_replay:
            truncated = False
            # Loop the scenario timestep when it exceeds max steps
            if self._scenario_timestep >= self.cfg.max_episode_steps:
                self._scenario_timestep = 0
        else:
            truncated = self._episode_step >= self.cfg.max_episode_steps

        # Update prev position
        self._prev_position = np.array([self._ego.x, self._ego.y])

        # Update prev position
        obs = self._get_observations()
        info: Dict[str, Any] = {
            "frame": self.frame,
            "collision": self._collision,
            "collision_at_fault": self._collision_at_fault,
            # getattr: minimally-built instances (object.__new__ in env
            # tests) never ran __init__ and lack this attribute.
            "contact_detail": getattr(self, "_contact_detail", {}),
            "speed_kmh": state["speed_kmh"],
            "scenario_timestep": self._scenario_timestep,
            **reward_info,
        }
        if getattr(self, "_contact_dynamics", False):
            info["physical_contact"] = self._physical_contact
            contact = self.get_contact_state()
            if contact is not None:
                info["contact_force"] = contact.get("net_forces_w")
        # Return terminated/truncated as (num_envs,) arrays to match the
        # DirectRLEnv contract the evaluator expects (it calls ``.any()``).
        n = getattr(self, "num_envs", 1) or 1
        terminated_arr = np.full((n,), bool(terminated), dtype=bool)
        truncated_arr = np.full((n,), bool(truncated), dtype=bool)
        return obs, reward, terminated_arr, truncated_arr, info

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _update_agents(self) -> None:
        """Update agent states via traffic manager.

        For ``traffic_mode="log_replay"``, reads agent positions from
        the scenario data at the current ``_scenario_timestep``. This
        reproduces the core loop of ``ScenarioReplayEnv._pre_physics_step``
        where agents are moved to their recorded positions each frame.

        For other traffic modes, this is a no-op at HLD level (IDM and
        learned traffic are wired in tasks 3.20–3.24).
        """
        stage_free = getattr(getattr(self, "_traffic_manager", None), "stage_free", False)
        if (self.cfg.traffic_mode == "log_replay" or stage_free) \
                and self._scenario_data is not None:
            tracks = self._scenario_data.get("tracks", {})
            meta = self._scenario_data.get("metadata", {})
            sdc_id = str(meta.get("sdc_id", "ego"))
            states: List[Dict] = []
            overrides = getattr(
                getattr(self, "_traffic_manager", None), "pose_overrides", None) or {}

            for agent_id, track in tracks.items():
                # Skip ego — ego is controlled by step_ego / bicycle model
                if agent_id == sdc_id:
                    continue

                # Extract state at current scenario timestep
                track_state = track.get("state", {})
                positions = track_state.get("position")
                headings = track_state.get("heading")
                valid = track_state.get("valid")

                if positions is None or len(positions) == 0:
                    continue

                t = min(self._scenario_timestep, len(positions) - 1)

                # Check validity at this timestep
                if valid is not None and t < len(valid) and not valid[t]:
                    continue

                pos = positions[t] if t < len(positions) else positions[-1]
                heading = float(headings[t]) if headings is not None and t < len(headings) else 0.0

                # Use the same held-tail convention as the renderer's view.
                vel_xy = self._track_velocity(
                    track, overrides.get(agent_id, {}), agent_id in overrides,
                    sampled_timestep=t)

                agent_type = track.get("type", "VEHICLE")
                # Per-frame bbox dims live under track["state"] (py123d
                # adapter output); the track top level has none. A top-level
                # lookup silently defaulted EVERY agent to a 4.5x1.8 car box,
                # so a 0.6 m pedestrian passed with >1 m of real clearance
                # registered as a phantom at-fault collision (CollisionDetector
                # and the live EPDMS scorer both consume these dicts).
                def_l, def_w, def_h = _DEFAULT_AGENT_DIMS.get(
                    agent_type, (4.5, 1.8, 1.5))
                length, width = track_dims(track, t, fallback=(def_l, def_w))
                # The z is the actor's own logged altitude, in the same
                # absolute frame the reconstruction was baked in (measured:
                # actor z 605.5 against a 605.1 sidecar on 4528c271d89c53e1,
                # 29.6 against 28.8 on 17cac31ef9135faf). Zeroing it here left
                # the renderer to rebuild it from the EGO's road height plus
                # half a car, which sits ~0.35 m high -- enough to lift a car
                # off the surface it was fitted on and smear its gaussians.
                states.append({
                    "id": agent_id,
                    "position": np.array(
                        [float(pos[0]), float(pos[1]),
                         float(pos[2]) if len(pos) > 2 else 0.0],
                        dtype=np.float32),
                    "heading": heading,
                    "velocity": vel_xy,
                    "length": length,
                    "width": width,
                    "height": track_height(track, t, fallback=def_h),
                    "type": agent_type,
                    "is_ego": False,
                })

            # A reactive agent's logged track is only its spawn condition:
            # the manager owns where it actually is. Merge its live pose over
            # the logged row so collision, TTC, the BEV and the renderer all
            # see one world. An id with no logged row at this frame (spawned
            # by an edit, or logged as invalid here) is appended rather than
            # dropped — otherwise the actor exists in the manager and nowhere
            # else.
            if overrides:
                by_id = {row["id"]: row for row in states}
                for agent_id, pose in overrides.items():
                    row = by_id.get(agent_id)
                    if row is not None:
                        row.update(pose)
                        continue
                    track = tracks.get(agent_id) or {}
                    agent_type = track.get("type", "VEHICLE")
                    def_l, def_w, def_h = _DEFAULT_AGENT_DIMS.get(
                        agent_type, (4.5, 1.8, 1.5))
                    states.append({
                        "id": agent_id,
                        "position": np.zeros(3, dtype=np.float32),
                        "heading": 0.0,
                        "velocity": np.zeros(2, dtype=np.float32),
                        "length": def_l, "width": def_w, "height": def_h,
                        "type": agent_type, "is_ego": False,
                        **pose,
                    })

            self._agent_states = states

    # ------------------------------------------------------------------
    # IsaacSim scene lifecycle (ported from ScenarioReplayEnv)
    #
    # All three hooks are no-ops when IsaacLab is not installed so the
    # pure-Python step()/_update_agents() loop (used by the unit tests and
    # any non-IsaacSim caller) keeps working unchanged. They are only
    # exercised by the IsaacLab ``DirectRLEnv`` driver, which calls
    # ``_setup_scene`` during ``__init__`` and ``_pre_physics_step`` /
    # ``_apply_action`` each physics step. Parity of the IsaacSim path is
    # validated by the task-13 parity test (IsaacSim is not installed here).
    # ------------------------------------------------------------------

    def _ensure_scenario_loaded(self) -> None:
        """Load the scenario via the configured loader if not already set.

        ``ScenarioReplayEnv`` loads its scenario in ``__init__`` (before
        ``DirectRLEnv.__init__`` triggers ``_setup_scene``). On the
        consolidated env, scenario loading normally happens at
        :meth:`reset`, but the IsaacSim ``_setup_scene`` is invoked by
        ``DirectRLEnv.__init__`` before the first ``reset()``. This helper
        bridges that gap by dispatching the loader on demand, mirroring the
        legacy load-then-build ordering. It is a no-op when scenario data is
        already present or when the source has no input configured.
        """
        if self._scenario_data is not None:
            return
        if not self._loader_has_input():
            return
        loader = _make_loader(self.cfg)
        self._scenario_data = self._post_load_scenario(loader.load(self.cfg))

    def _post_load_scenario(self, sd: Optional[Dict]) -> Optional[Dict]:
        """Hook: transform freshly-loaded scenario data before the env uses it.

        Identity in the base env. Subclasses (e.g.
        :class:`~navsafe.env.navsafe_edit_env.NexusSimEditEnv`) override it
        to apply declarative scenario edits. Called at both load sites
        (``_ensure_scenario_loaded`` during ``_setup_scene`` and the
        loader-dispatch branch of :meth:`reset`).
        """
        return sd

    def _extra_agent_manager_kwargs(self) -> Dict[str, Any]:
        """Hook: extra kwargs for the :class:`ScenarioReplayManager` build.

        Empty in the base env — subclasses use it to opt into manager
        behaviours (e.g. ``spawn_z_from_track`` for mesh insertion into the
        NuRec reconstruction) without changing the base construction.
        """
        return {}


    def _setup_scene(self) -> None:
        """Build the USD scene: road geometry, agents, and scene dressing.

        Ported from ``ScenarioReplayEnv._setup_scene``. Builds road geometry
        via :class:`~navsafe.scene.scenario_map_builder.ScenarioMapBuilder`
        and spawns/replays agents via
        :class:`~navsafe.manager.scenario_replay_manager.ScenarioReplayManager`,
        gated on the ``build_*`` / ``spawn_*`` cfg flags. All IsaacSim, USD,
        and manager calls are reached only when IsaacLab is installed; the
        method returns immediately otherwise so the pure-Python paths import
        and run without IsaacSim.
        """
        if not _HAS_ISAACLAB:
            return


        # Lazy imports — these pull in IsaacSim/USD and must never run at
        # module import time.
        from navsafe.scene.scenario_map_builder import ScenarioMapBuilder
        from navsafe.manager.scenario_replay_manager import ScenarioReplayManager

        # Scenario must be available before we can build geometry/agents.
        self._ensure_scenario_loaded()
        scenario = self._scenario_data

        # Resolve vehicle_assets_path relative to the working directory.
        veh_path = self.cfg.vehicle_assets_path
        if veh_path and not os.path.isabs(veh_path):
            veh_path = os.path.abspath(veh_path)

        # Keep the IsaacSim replay cursor in sync with the pure-Python one.
        self._scenario_timestep = 0

        # Build the symbolic scene and simulation agents.
        self._build_scenario_content(scenario, veh_path=veh_path)

        # Set up the scene renderer with scenario data and env reference.
        self._setup_renderer_for_scenario(scenario, force=True)

        # Physics execution mode: the invisible rigid-body proxy must exist
        # before the parent setup so PhysX parses it during sim.reset()'s
        # warmup (force_load_physics_from_usd). Authored before the extra
        # sensors so the contact sensor can attach to the proxy path.
        if getattr(self.cfg, "execution_mode", "kinematic") == "physics":
            self._author_ego_physics_proxy()
            # The kinematic agent collider pool shares the same constraint:
            # every PhysX actor must exist before sim.reset()'s physics
            # parse, and the pool can never grow afterwards (actor add/
            # remove invalidates the tensor views).
            if self._contact_dynamics:
                self._author_agent_contact_proxies()

        # Optional ego sensors (LiDAR / IMU / contact); each is gated on the
        # corresponding cfg slot being non-None inside _setup_extra_sensors.
        try:
            self._setup_extra_sensors()
        except Exception as exc:  # pragma: no cover — Linux+IsaacSim only
            logger.warning("[NexusSimEnv] _setup_extra_sensors failed: %s", exc)

        # Call the IsaacLab parent setup.
        super()._setup_scene()

    @staticmethod
    def _scenario_identity(scenario: Any) -> str:
        """The scenario id used to detect scenario switches ('' if unknown)."""
        if not isinstance(scenario, dict):
            return ""
        metadata = scenario.get("metadata") or {}
        return str(
            metadata.get("scenario_id")
            or metadata.get("real2sim_clip_id")
            or scenario.get("scenario_id")
            or scenario.get("id")
            or ""
        )

    def _setup_renderer_for_scenario(
        self,
        scenario: Any,
        *,
        force: bool = False,
    ) -> bool:
        """Install ``scenario`` in the renderer exactly once per switch.

        Renderers are stateful: NuRec carries the scene id, timestamps,
        origin, camera calibration and editable tracks. Close and recreate
        on a scene switch so no scene-local cache or
        server edit can leak into the next episode.
        """
        identity = self._scenario_identity(scenario)
        marker = identity or f"object:{id(scenario)}"
        previous = getattr(self, "_renderer_scenario_id", None)
        if not force and marker == previous:
            return False
        if self._renderer is not None and previous is not None:
            self._renderer.close()
            self._renderer = None
        self._ensure_renderer()
        self._renderer.setup(scenario, env=self)
        self._renderer_scenario_id = marker
        return True

    def _build_scenario_content(
        self,
        scenario: Any,
        *,
        veh_path: Optional[str] = None,
        reuse_ego_prims: Optional[Dict[int, str]] = None,
    ) -> None:
        """Build the scenario-dependent USD content: map meshes + agent prims.

        The ONE shared code path between construction (``_setup_scene``) and
        the mid-run scenario rebuild (``_maybe_rebuild_scenario_scene``).
        Assigns ``self.map_builder`` / ``self.agent_manager`` only after the
        spawn succeeds, so a failed rebuild does not strand a half-populated
        manager. Records the built scenario's id in ``_built_scenario_id``.

        Args:
            scenario: The ScenarioDescription dict to materialize.
            veh_path: Pre-resolved ``cfg.vehicle_assets_path`` (absolute);
                resolved here when omitted.
            reuse_ego_prims: ``{env_id: prim_path}`` of surviving ego prims to
                adopt instead of respawning during scene rebuilds.
        """
        from navsafe.manager.scenario_replay_manager import (
            ScenarioReplayManager,
        )
        from navsafe.scene.scenario_map_builder import ScenarioMapBuilder

        if veh_path is None:
            veh_path = self.cfg.vehicle_assets_path
            if veh_path and not os.path.isabs(veh_path):
                veh_path = os.path.abspath(veh_path)

        # ``build_map`` and ``spawn_agents`` both append
        # ``envs/env_{id}/<scope>`` to the parent they are given, so the
        # parent must be the WORLD ROOT ("/World"), not the env scope —
        # passing env_prim_paths[env_id] doubles the path
        # ("/World/envs/env_N/envs/env_N/..."), which detaches the content
        # from every consumer that expects the canonical location (e.g. the
        # lane-material pass in scene_builder._fix_scene_appearance).
        world_root = self.scene.env_prim_paths[0].rsplit("/envs/", 1)[0]

        map_builder = ScenarioMapBuilder(
            scenario=cast("ScenarioDescription", scenario),
            stage=self.sim.stage,
        )
        # Gate the derived lane-edge lines like the other build_* flags (the
        # numeric look tunables stay as builder attributes).
        map_builder.draw_lane_boundaries = self.cfg.build_lane_boundaries

        if self.cfg.build_lanes or self.cfg.build_lane_markings:
            logger.info("[NexusSimEnv] Building map...")
            for env_id in range(self.num_envs):
                map_elements = map_builder.build_map(
                    env_id=env_id, parent_prim_path=world_root)
                logger.info(
                    "[NexusSimEnv] Environment %d: Lanes=%d, Markings=%d",
                    env_id,
                    len(map_elements["lanes"]),
                    len(map_elements["lane_markings"]),
                )

        manager = ScenarioReplayManager(
            scenario=cast("ScenarioDescription", scenario),
            num_envs=self.num_envs,
            device=self.device,
            asset_selection_strategy=self.cfg.asset_selection_strategy,
            random_seed=self.cfg.asset_random_seed,
            use_real_assets=self.cfg.use_real_assets,
            vehicle_assets_path=cast(str, veh_path),
            # Upstream hook (nurec backends override this to thread extra
            # manager kwargs); the shared build path must honor it for BOTH
            # construction and the mid-run scenario rebuild.
            **self._extra_agent_manager_kwargs(),
        )

        # Spawn agents (respecting the spawn_* cfg flags). With NuRec, the OTHER agents' visuals are baked into the reconstruction's
        # temporal Gaussians, so spawn ONLY the ego chassis — it is still needed
        # as the camera mount (the ego camera rig parents under it, and
        # _resolve_ego_prim_path resolves the rig prim path against the spawned
        # ego prim). The symbolic agent states (collision + EPDMS) still come
        # from _collect_agent_states_for_renderer (scenario tracks via the
        # manager, not the spawned prims).
        ego_only = not getattr(self.cfg, "spawn_agent_visuals", True)
        logger.info(
            "[NexusSimEnv] Spawning agents%s...",
            " (ego-only: NuRec renders other agents in the reconstruction)"
            if ego_only else "",
        )
        skip_types: set = set()
        if not getattr(self.cfg, "spawn_pedestrians", True):
            skip_types.add("PEDESTRIAN")
        if not getattr(self.cfg, "spawn_cyclists", True):
            skip_types.add("CYCLIST")
        agent_info = manager.spawn_agents(
            stage=self.sim.stage,
            parent_prim_path=world_root,
            skip_types=skip_types,
            ego_only=ego_only,
            reuse_ego_prims=reuse_ego_prims,
        )
        logger.info(
            "[NexusSimEnv] Spawned agents: Vehicles=%d, Pedestrians=%d, "
            "Cyclists=%d",
            len(agent_info["vehicles"]),
            len(agent_info["pedestrians"]),
            len(agent_info["cyclists"]),
        )

        self.map_builder = map_builder
        self.agent_manager = manager
        # Identity of the scenario whose USD content is materialized on the
        # stage. reset() compares against it to rebuild map/agents/ground
        # when reset_to_scene switched scenarios (the scene used to stay
        # frozen at the construction scenario — stale roads and agents for
        # every other scene).
        self._built_scenario_id = self._scenario_identity(scenario)


    def _author_ego_physics_proxy(self) -> None:
        """Author the invisible PhysX rigid body that integrates the ego.

        The ego *visual* prim cannot own physics: with fabric enabled PhysX
        writes poses to Fabric only (updateToUsd=False) while NexusSim's
        render delegate reads transforms from USD — a physics-owned visual
        prim would render frozen. Instead a bare, invisible Xform carries
        the rigid body; the visual prim keeps following ``self._ego`` via
        :meth:`_advance_replay_agent_prims`.

        Dynamic (not kinematic) so PhysX integrates commanded velocities;
        gravity disabled and zero damping so nothing but the commands moves
        it (no ground colliders exist). Explicit mass/inertia keep PhysX
        happy with a shapeless body and pin the COM to the prim origin.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        import isaaclab.sim.schemas as schemas

        proxy_path = "/World/envs/env_0/ego_physics_proxy"
        stage = self.sim.stage
        xform = UsdGeom.Xform.Define(stage, proxy_path)
        prim = xform.GetPrim()
        schemas.define_rigid_body_properties(
            proxy_path,
            schemas.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            stage=stage,
        )
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(1500.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2000.0, 2000.0, 3000.0))
        UsdGeom.Imageable(prim).MakeInvisible()
        self._ego_phys_proxy_path = proxy_path
        if self._contact_dynamics:
            self._author_ego_contact_collider(stage, proxy_path)

    @staticmethod
    def _author_contact_box(stage: Any, parent_path: str,
                            dims: Tuple[float, float, float]) -> None:
        """Author an invisible unit-cube collider scaled to ``dims``.

        ``purpose=guide`` keeps the box out of every render pass even if
        visibility is toggled; the collider participates in PhysX only.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        cube = UsdGeom.Cube.Define(stage, parent_path + "/collider")
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*dims))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        img = UsdGeom.Imageable(cube.GetPrim())
        img.CreatePurposeAttr(UsdGeom.Tokens.guide)
        img.MakeInvisible()

    def _author_ego_contact_collider(self, stage: Any,
                                     proxy_path: str) -> None:
        """Give the ego physics proxy a box collider + contact reporting.

        The box uses the same ego dims as the symbolic shapely detector, so
        physical and symbolic contact agree. Z translation and roll/pitch
        are locked (zero-g, no ground colliders): contacts stay planar even
        against boxes of differing heights.
        """
        from pxr import PhysxSchema

        import isaaclab.sim.schemas as schemas

        ecfg = self._ego.cfg
        self._author_contact_box(
            stage, proxy_path,
            (float(ecfg.ego_length), float(ecfg.ego_width),
             float(ecfg.ego_height)))
        # PhysxContactReportAPI so the optional EgoContactSensorCfg scaffold
        # can report net forces from this body.
        schemas.activate_contact_sensors(proxy_path, threshold=0.0,
                                         stage=stage)
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(
            stage.GetPrimAtPath(proxy_path))
        body.CreateLockedPosAxisAttr(4)   # lock z translation
        body.CreateLockedRotAxisAttr(3)   # lock roll | pitch

    # Park pose for contact-proxy slots whose agent is currently invalid —
    # far outside any scene content, spaced so parked boxes never touch.
    _CONTACT_PARK_XY = (1.0e4, 1.0e4)

    # Agent contact bodies are DYNAMIC with this mass (vs the ego's 1500 kg)
    # and velocity-driven every frame: PhysX kinematic targets would be the
    # natural fit, but this build's warp/CUDA tensor backend rejects
    # ``set_kinematic_targets`` ("Failed to set ... in backend") while
    # ``set_velocities``/``set_transforms`` are proven by the ego proxy. A
    # 666:1 mass ratio makes contacts effectively one-way (the ego cannot
    # measurably deflect an agent), and the per-frame velocity command
    # ``(target - current)/dt`` self-corrects any residual drift.
    _CONTACT_AGENT_MASS = 1.0e6

    def _author_agent_contact_proxies(self) -> None:
        """Author the agent collider pool: one slot per non-ego track.

        Slots are invisible box colliders on heavy velocity-driven rigid
        bodies — they push the ego proxy but are (effectively) never pushed
        themselves, keeping replay agents log-authoritative. Agent-agent
        contacts are filtered out via a self-filtered collision group:
        replay traffic frequently overlaps slightly, and unfiltered dynamic
        boxes would jitter apart. The pool is created once, before
        ``super()._setup_scene()`` (PhysX parses USD during ``sim.reset()``)
        and is never grown/shrunk afterwards: adding or removing PhysX
        actors invalidates every omni.physics.tensors view (including
        ``_ego_phys_view``). Scenario switches rebind slots instead
        (:meth:`_rebind_agent_contact_proxies`).
        """
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        import isaaclab.sim.schemas as schemas

        stage = self.sim.stage
        scope_path = "/World/envs/env_0/contact_proxies"
        UsdGeom.Scope.Define(stage, scope_path)
        self._contact_slots = []
        ego_id = self.agent_manager.ego_agent_id
        slot = 0
        for agent_id in self.agent_manager.tracks:
            if agent_id == ego_id:
                continue
            init = self.agent_manager.get_initial_state(agent_id)
            if init is None:
                continue
            dims = (float(init.get("length", 4.5) or 4.5),
                    float(init.get("width", 1.8) or 1.8),
                    float(init.get("height", 1.5) or 1.5))
            path = f"{scope_path}/slot_{slot:04d}"
            xform = UsdGeom.Xform.Define(stage, path)
            prim = xform.GetPrim()
            UsdGeom.Xformable(xform).AddTranslateOp().Set(
                Gf.Vec3d(self._CONTACT_PARK_XY[0] + 20.0 * slot,
                         self._CONTACT_PARK_XY[1], 0.0))
            schemas.define_rigid_body_properties(
                path,
                schemas.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                stage=stage,
            )
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateMassAttr(self._CONTACT_AGENT_MASS)
            mass.CreateDiagonalInertiaAttr(
                Gf.Vec3f(1.0e7, 1.0e7, 1.0e7))
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateLockedPosAxisAttr(4)   # lock z translation
            body.CreateLockedRotAxisAttr(3)   # lock roll | pitch
            self._author_contact_box(stage, path, dims)
            img = UsdGeom.Imageable(prim)
            img.CreatePurposeAttr(UsdGeom.Tokens.guide)
            img.MakeInvisible()
            self._contact_slots.append({
                "path": path, "agent_id": agent_id, "dims": dims,
                "parked": True,
            })
            slot += 1
        # Self-filtered collision group: agent boxes never collide with each
        # other, only with the (non-member) ego proxy.
        group = UsdPhysics.CollisionGroup.Define(
            stage, f"{scope_path}/agents_group")
        group.CreateFilteredGroupsRel().AddTarget(group.GetPath())
        group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(
            scope_path)
        self._disable_foreign_colliders()
        logger.info("[NexusSimEnv] contact_dynamics: %d agent collider "
                    "slots authored", len(self._contact_slots))

    def _disable_foreign_colliders(self) -> None:
        """``collisionEnabled=False`` on every non-contact-proxy collider.

        The agent visual USDs and the road-surface mesh ship
        ``CollisionAPI``-tagged geometry. Pre-feature those static colliders
        were inert — the only dynamic body (the shapeless ego proxy) could
        not touch anything — but with contact colliders live they jam the
        simulation: a contact box at z=0 permanently penetrates the
        road-surface mesh and friction freezes it within a frame (measured).
        Colliding with visual meshes is also wrong by design: contact
        semantics belong to the box proxies (track dims), not to whatever
        geometry a renderable asset happens to ship. Re-run after every
        scenario rebuild (respawned visuals bring fresh colliders).
        """
        from pxr import UsdPhysics

        stage = self.sim.stage
        n = 0
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            if ("/contact_proxies/" in path
                    or "/ego_physics_proxy" in path):
                continue
            try:
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(
                    False)
                n += 1
            except Exception:  # pragma: no cover — instance proxies etc.
                logger.info("[NexusSimEnv] could not disable collider on "
                            "%s", path)
        if n:
            logger.info("[NexusSimEnv] contact_dynamics: disabled %d "
                        "foreign colliders (road/visual-mesh geometry)", n)

    def _init_agent_contact_view(self) -> None:
        """Create the batched PhysX tensor view over the collider pool."""
        import warp as wp

        if not self._contact_slots:
            logger.warning("[NexusSimEnv] contact_dynamics: no non-ego "
                           "tracks — contact pool empty")
            return
        view = self.sim.physics_sim_view.create_rigid_body_view(
            "/World/envs/env_0/contact_proxies/slot_*")
        n = len(self._contact_slots)
        if view.count != n:
            raise RuntimeError(
                f"contact proxy view count {view.count} != authored slots "
                f"{n} — stage/physics parse mismatch")
        # Map view row -> slot via prim paths when the view exposes them
        # (view order is not guaranteed to match authoring order). Fallback:
        # slot paths are zero-padded, so lexicographic == authoring order.
        paths = list(getattr(view, "prim_paths", None) or [])
        if paths:
            by_path = {s["path"]: s for s in self._contact_slots}
            self._contact_slots = [by_path[p] for p in paths]
        self._contact_view = view
        self._contact_tf_np = np.zeros((n, 7), dtype=np.float32)
        self._contact_tf_buf = wp.zeros((n, 7), dtype=wp.float32,
                                        device=self.sim.device)
        self._contact_vel_np = np.zeros((n, 6), dtype=np.float32)
        self._contact_vel_buf = wp.zeros((n, 6), dtype=wp.float32,
                                         device=self.sim.device)
        self._contact_all_indices = wp.array(
            list(range(n)), dtype=wp.uint32, device=self.sim.device)
        self._reset_agent_contact_proxies()

    def _slot_pose_row(self, i: int, slot: Dict[str, Any],
                       state: Optional[Dict[str, Any]]) -> None:
        """Fill staging row ``i`` with the slot's pose (park pose if None)."""
        if state is None:
            self._contact_tf_np[i] = (
                self._CONTACT_PARK_XY[0] + 20.0 * i,
                self._CONTACT_PARK_XY[1], 0.0, 0.0, 0.0, 0.0, 1.0)
            return
        half = 0.5 * float(state["heading"])
        pos = state["position"]
        self._contact_tf_np[i] = (float(pos[0]), float(pos[1]), 0.0,
                                  0.0, 0.0, np.sin(half), np.cos(half))

    def _write_agent_contact_targets(self, timestep: int) -> None:
        """Velocity-drive every collider box toward its next replay pose.

        Each active slot gets ``(target - current)/dt`` as a velocity
        command for the coming PhysX substep, so contacts carry the replay
        velocity — a 10 m/s car imparts a 10 m/s-scale impulse instead of a
        zero-velocity overlap resolution — and any post-contact drift
        self-corrects on the next frame. Park/unpark transitions teleport
        instead: a velocity command to/from the park pose would drag the
        box across the scene at enormous speed.
        """
        mgr = self.agent_manager
        timestep = min(max(timestep, 0),
                       max(0, int(mgr.scenario_length) - 1))
        dt = float(self.cfg.dt) or 0.1
        current = self._contact_view.get_transforms().numpy()
        teleport_rows = []
        self._contact_vel_np[:] = 0.0
        for i, slot in enumerate(self._contact_slots):
            state = mgr.get_agent_state(slot["agent_id"], timestep)
            valid = bool(state is not None and state.get("valid", True))
            if not valid:
                if not slot["parked"]:
                    slot["parked"] = True
                    self._slot_pose_row(i, slot, None)
                    teleport_rows.append(i)
                continue
            self._slot_pose_row(i, slot, state)
            if slot["parked"]:
                slot["parked"] = False
                teleport_rows.append(i)
                continue
            tgt = self._contact_tf_np[i]
            cur = current[i]
            self._contact_vel_np[i, 0] = (tgt[0] - cur[0]) / dt
            self._contact_vel_np[i, 1] = (tgt[1] - cur[1]) / dt
            # Yaw rate from the shortest quaternion yaw difference (both
            # quats are pure-yaw by construction / planar locks).
            cur_yaw = 2.0 * np.arctan2(cur[5], cur[6])
            tgt_yaw = 2.0 * np.arctan2(tgt[5], tgt[6])
            dyaw = np.arctan2(np.sin(tgt_yaw - cur_yaw),
                              np.cos(tgt_yaw - cur_yaw))
            self._contact_vel_np[i, 5] = dyaw / dt
        if teleport_rows:
            self._teleport_contact_rows(teleport_rows)
        self._contact_vel_buf.assign(self._contact_vel_np)
        self._contact_view.set_velocities(
            self._contact_vel_buf, self._contact_all_indices)

    def _teleport_contact_rows(self, rows: List[int]) -> None:
        """Velocity-free teleport of the given slot rows (park/unpark)."""
        import warp as wp

        self._contact_tf_buf.assign(self._contact_tf_np)
        idx = wp.array(rows, dtype=wp.uint32, device=self.sim.device)
        self._contact_view.set_transforms(self._contact_tf_buf, idx)
        self._contact_vel_np[rows] = 0.0
        self._contact_vel_buf.assign(self._contact_vel_np)
        self._contact_view.set_velocities(self._contact_vel_buf, idx)

    def _reset_agent_contact_proxies(self) -> None:
        """Teleport every slot to its timestep-0 pose (or park) at reset."""
        if self._contact_view is None:
            return
        mgr = self.agent_manager
        for i, slot in enumerate(self._contact_slots):
            state = mgr.get_agent_state(slot["agent_id"], 0)
            valid = bool(state is not None and state.get("valid", True))
            slot["parked"] = not valid
            self._slot_pose_row(i, slot, state if valid else None)
        self._teleport_contact_rows(list(range(len(self._contact_slots))))
        self._physical_contact = False

    def _rebind_agent_contact_proxies(self) -> None:
        """Rebind the fixed slot pool to a new scenario's non-ego tracks.

        Slots are matched to tracks greedily by box length (both sorted) —
        collider dims across vehicle tracks differ by decimeters, an
        acceptable approximation. The cube's scale op is updated best-effort
        (omni.physx resyncs collider geometry on shape-attribute edits).
        Surplus tracks get no collider (warning); surplus slots stay parked.
        No PhysX actors are created or destroyed, so every tensor view stays
        valid.
        """
        if self._contact_view is None:
            return
        from pxr import Gf, UsdGeom

        stage = self.sim.stage
        mgr = self.agent_manager
        ego_id = mgr.ego_agent_id
        tracks = []
        for agent_id in mgr.tracks:
            if agent_id == ego_id:
                continue
            init = mgr.get_initial_state(agent_id)
            if init is None:
                continue
            tracks.append((agent_id,
                           (float(init.get("length", 4.5) or 4.5),
                            float(init.get("width", 1.8) or 1.8),
                            float(init.get("height", 1.5) or 1.5))))
        if len(tracks) > len(self._contact_slots):
            logger.warning(
                "[NexusSimEnv] contact_dynamics: new scenario has %d "
                "non-ego tracks but only %d collider slots — %d agents "
                "will have no physical collider",
                len(tracks), len(self._contact_slots),
                len(tracks) - len(self._contact_slots))
        tracks.sort(key=lambda t: t[1][0])
        slots = sorted(self._contact_slots, key=lambda s: s["dims"][0])
        for slot, (agent_id, dims) in zip(slots, tracks):
            slot["agent_id"] = agent_id
            slot["parked"] = True
            if dims != slot["dims"]:
                slot["dims"] = dims
                try:
                    cube = UsdGeom.Cube(
                        stage.GetPrimAtPath(slot["path"] + "/collider"))
                    ops = UsdGeom.Xformable(cube).GetOrderedXformOps()
                    if ops:
                        ops[0].Set(Gf.Vec3f(*dims))
                except Exception:  # pragma: no cover — best-effort rescale
                    logger.info("[NexusSimEnv] contact collider rescale "
                                "failed for %s", slot["path"])
        for slot in slots[len(tracks):]:
            slot["agent_id"] = ""
            slot["parked"] = True
        # The rebuild respawned agent visuals (and the map) with fresh
        # CollisionAPI geometry — neutralize it again.
        self._disable_foreign_colliders()
        self._reset_agent_contact_proxies()

    def _init_ego_physics_view(self) -> None:
        """Create the PhysX tensor view over the ego proxy (post sim.reset()).

        Also pauses play-driven physics stepping: camera captures pump raw
        Kit app updates, and with ``/app/player/playSimulations`` enabled
        each capture tick would step PhysX again — double-integrating the
        ego. IsaacLab's ``sim.step()`` calls ``physx_sim.simulate()``
        directly, so manual stepping is unaffected by the pause.
        """
        import warp as wp

        assert self._ego_phys_proxy_path is not None
        self._ego_phys_view = self.sim.physics_sim_view.create_rigid_body_view(
            self._ego_phys_proxy_path)
        self._ego_phys_indices = wp.array(
            [0], dtype=wp.uint32, device=self.sim.device)
        # Preallocated staging buffers: the write helpers run on the 10 Hz
        # step path — reuse pinned numpy + device arrays instead of
        # allocating a fresh wp.array (GPU alloc + launch) every frame.
        self._ego_phys_tf_np = np.zeros((1, 7), dtype=np.float32)
        self._ego_phys_vel_np = np.zeros((1, 6), dtype=np.float32)
        self._ego_phys_tf_buf = wp.zeros((1, 7), dtype=wp.float32,
                                         device=self.sim.device)
        self._ego_phys_vel_buf = wp.zeros((1, 6), dtype=wp.float32,
                                          device=self.sim.device)
        self.set_ego_externally_driven(True)
        self.sim.set_setting("/app/player/playSimulations", False)

    def _step_ego_physics(self, steer: float, accel: float) -> Dict[str, Any]:
        """Physics-mode ego step: command velocities, integrate, read back.

        The bicycle model turns the action into a velocity command, PhysX
        integrates it for one substep of ``cfg.dt``, and the resulting pose
        is adopted as the ego state. ``cmd["speed"]`` (post-accel), NOT the
        PhysX-measured speed, is applied — the command was issued at the
        current speed by design (see ``compute_velocity_command``).

        With ``contact_dynamics``: the agent collider pool is targeted at
        the next replay frame before the substep, and a contact is detected
        by the residual between commanded and PhysX-measured velocity (in
        zero-g with zero damping they match exactly unless a contact
        impulse intervened). On contact frames the ego adopts the measured
        longitudinal speed so a shove persists into subsequent motion;
        lateral/backward components survive in the pose only (the bicycle
        model carries no such state).
        """
        cmd = self._ego.compute_velocity_command(steer, accel)
        if self._contact_view is not None:
            self._write_agent_contact_targets(self._scenario_timestep + 1)
        self._write_ego_proxy_velocity(cmd)
        self.sim.step(render=False)  # one PhysX substep of cfg.dt
        x, y, yaw = self._read_ego_proxy_pose()
        self._physical_contact = False
        speed = cmd["speed"]
        if self._contact_view is not None:
            v = self._ego_phys_view.get_velocities().numpy()[0]
            residual = float(np.hypot(v[0] - cmd["lin_vel_w"][0],
                                      v[1] - cmd["lin_vel_w"][1]))
            if residual > 0.05:
                self._physical_contact = True
                speed = max(0.0, float(v[0]) * np.cos(yaw)
                            + float(v[1]) * np.sin(yaw))
        return self._ego.apply_external_state(x, y, yaw, speed)

    def _write_ego_proxy_state(self, x: float, y: float, heading: float,
                               lin_vel: Tuple[float, float] = (0.0, 0.0),
                               yaw_rate: float = 0.0) -> None:
        """Teleport the physics proxy (pure-yaw quaternion) + set velocities."""
        half = 0.5 * float(heading)
        self._ego_phys_tf_np[0] = (x, y, 0.0, 0.0, 0.0, np.sin(half), np.cos(half))
        self._ego_phys_tf_buf.assign(self._ego_phys_tf_np)  # host→device copy
        self._ego_phys_view.set_transforms(
            self._ego_phys_tf_buf, self._ego_phys_indices)
        self._ego_phys_vel_np[0] = (lin_vel[0], lin_vel[1], 0.0, 0.0, 0.0, yaw_rate)
        self._ego_phys_vel_buf.assign(self._ego_phys_vel_np)
        self._ego_phys_view.set_velocities(
            self._ego_phys_vel_buf, self._ego_phys_indices)

    def _write_ego_proxy_velocity(self, cmd: Dict[str, Any]) -> None:
        """Write a bicycle-model velocity command for PhysX to integrate."""
        self._ego_phys_vel_np[0] = (cmd["lin_vel_w"][0], cmd["lin_vel_w"][1],
                                    0.0, 0.0, 0.0, cmd["yaw_rate"])
        self._ego_phys_vel_buf.assign(self._ego_phys_vel_np)
        self._ego_phys_view.set_velocities(
            self._ego_phys_vel_buf, self._ego_phys_indices)

    def _read_ego_proxy_pose(self) -> Tuple[float, float, float]:
        """Read back the PhysX-integrated proxy pose as (x, y, yaw)."""
        tf = self._ego_phys_view.get_transforms().numpy()[0]
        qx, qy, qz, qw = (float(v) for v in tf[3:7])
        yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy),
                               1.0 - 2.0 * (qy * qy + qz * qz)))
        return float(tf[0]), float(tf[1]), yaw

    def _pre_physics_step(self, actions: Any) -> None:
        """Update agent states before the physics step.

        Ported from ``ScenarioReplayEnv._pre_physics_step``: delegates the
        non-ego agent update to the configured traffic manager (selected on
        ``cfg.traffic_mode``), then syncs the collected agent states to the
        renderer and to the pure-Python collision state, and applies the ego
        override (model-planned trajectory) when active. No-op without
        IsaacLab.

        The traffic manager replaces the previous inline
        agent-state and reactive-traffic update block:

        * ``LogReplayTraffic`` advances every agent to the current replay
          timestep (skipping the ego when an override is active).
        * ``SemiReactiveTraffic`` updates admitted reactive followers.
        * ``NoTrafficManager`` is a no-op (agents stay at their spawn pose).

        Args:
            actions: Actions tensor (unused in pure replay mode).
        """
        if not _HAS_ISAACLAB:
            return
        if getattr(self, "agent_manager", None) is None:
            return

        # Drive the USD agent prims (non-ego via the traffic manager + the ego
        # override) to the current replay frame.
        self._advance_replay_agent_prims()

        # Collect agent states and sync to the pure-Python agent list so
        # check_collision() and the renderer see the current poses.
        agent_states = self._collect_agent_states_for_renderer()
        self._agent_states = agent_states

        # Notify the renderer of updated agent positions.
        self._ensure_renderer()
        self._renderer.update_agents(agent_states)

        # Sync the bicycle-model ego fields from scenario/override data.
        self._sync_base_ego_state()

    def _advance_replay_agent_prims(self) -> None:
        """Re-pose the IsaacSim USD agent prims to the current replay frame.

        This is the USD-side, per-frame agent update shared by both env
        drivers:

        * :meth:`_pre_physics_step` — the IsaacLab ``DirectRLEnv`` / online-RL
          path, and
        * :meth:`step` — the path the
          :class:`~navsafe.evaluation.evaluator.Evaluator` drives, which does
          **not** route through ``_pre_physics_step``.

        It delegates non-ego agents to the configured traffic manager (which
        advances every replay track to ``env.scenario_timestep`` and toggles
        per-frame visibility), then re-poses the ego prim from the active ego
        override so the ego mesh tracks the planned trajectory instead of being
        left frozen at its spawn pose.

        It mutates only USD prim transforms — never the pure-Python ego or
        collision state — and is a no-op when IsaacLab is unavailable or no
        replay manager is attached, so the pure-Python step loop is unaffected.
        """
        if not _HAS_ISAACLAB:
            return
        if getattr(self, "agent_manager", None) is None:
            return

        # Delegate non-ego agent updates to the configured traffic manager.
        # The manager reads env.scenario_timestep / env.current_scenario and
        # honours env._ego_override_active where relevant.
        #
        # A STAGE-FREE MANAGER HAS ALREADY BEEN STEPPED THIS FRAME, in step()
        # before the prims are touched, because advancing an agent is a decision
        # and posing its prim is a rendering detail. Stepping it again here
        # advanced every reactive actor by 2*dt per frame: R-3's pedestrians
        # crossed in 30-40 frames instead of the 60-80 their authored 1.2-1.6
        # m/s implies, which reads as "the walkers are sprinting" and, because
        # they were through the road before anyone met anyone, as "social force
        # does nothing".
        if not getattr(self._traffic_manager, "stage_free", False):
            self._traffic_manager.step(self, self.cfg.dt)

        # Re-pose the ego prim from whoever owns the ego this step:
        #   1. the override when active (model-planned teleport / GT warm-up),
        #   2. the bicycle-model / physics-integrated ego state when the ego
        #      is externally driven (controller / physics execution modes —
        #      set via set_ego_externally_driven; the traffic replay skips
        #      the ego there),
        #   3. otherwise nobody here — log-replay traffic already posed it.
        ego_pose: Optional[Dict[str, Any]] = None
        if self._ego_override_active and self._ego_override_state is not None:
            ego_pose = self._ego_override_state
        elif self._ego_externally_driven:
            ego_pose = {
                "position": np.array(
                    [self._ego.x, self._ego.y, 0.0], dtype=np.float32),
                "heading": float(self._ego.heading),
            }
        if ego_pose is not None:
            ego_id = getattr(self.agent_manager, "ego_agent_id", None)
            if ego_id is not None:
                key = (0, ego_id)
                prim_path = self.agent_manager.agent_prims.get(key)
                if prim_path:
                    prim = self.sim.stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        self.agent_manager._update_transform(prim, ego_pose)

    def _apply_action(self) -> None:
        """Apply actions to the simulation (no-op in replay mode).

        Ported from ``ScenarioReplayEnv._apply_action``, which is a no-op:
        in pure replay the ego/agent poses are driven by the replay manager
        and the optional ego override in :meth:`_pre_physics_step`, not by an
        action applied here.
        """
        pass

    # ------------------------------------------------------------------
    # Ego override (evaluator contract, ported from ScenarioReplayEnv)
    # ------------------------------------------------------------------

    def _sync_base_ego_state(self) -> None:
        """Sync the pure-Python ego fields from scenario/override data.

        Keeps ``ego_x``/``ego_y``/``ego_heading``/``ego_speed`` aligned with
        the authoritative ego state so :meth:`check_collision` and renderer
        delegation stay correct. Ported from
        ``ScenarioReplayEnv._sync_base_ego_state``.
        """
        if self._ego_override_active and self._ego_override_state is not None:
            ego = self._ego_override_state
            pos = ego.get("position", np.zeros(3))
            self._ego.x = float(pos[0])
            self._ego.y = float(pos[1])
            self._ego.heading = float(ego.get("heading", 0.0))
            self._ego.speed = float(ego.get("speed", 0.0))
            # --- DEBUG-ONLY (not a production feature) --------------------
            # Lateral (left/right) viewpoint stress-test for ego-replay: shift
            # the replayed ego pose sideways, perpendicular to its heading, by
            # EGO_LATERAL_SHIFT_M metres (+ = left, - = right). Lets us inspect
            # how the NuRec reconstruction holds up off the recorded trajectory
            # without touching any real config. Gated on an env var so the
            # normal (unset) path is byte-for-byte unchanged. Remove when the
            # viewpoint-robustness debugging is done.
            _dbg_shift = float(os.environ.get("EGO_LATERAL_SHIFT_M", "0") or "0")
            if _dbg_shift:
                _h = self._ego.heading
                self._ego.x += -np.sin(_h) * _dbg_shift  # left = perp. to heading
                self._ego.y += np.cos(_h) * _dbg_shift
            # -------------------------------------------------------------

    #: Extra clearance, in metres, required between the ego's footprint and a
    #: track appearing for the first time. Half a car width: enough that a
    #: spawn is refused while it would overlap or graze the ego, small enough
    #: that ordinary traffic appearing a lane away is unaffected.
    SPAWN_CLEARANCE_M = 1.0

    def _spawn_blocked_by_ego(self, agent_id, state, track) -> bool:
        """Hold a first appearance until its footprint has the spawn clearance.

        Only a first appearance is tested. Once a track has been admitted it is
        the traffic manager's or the log's business, and re-testing it every
        frame would delete an actor the ego then drove into -- which is a real
        collision, not a spawn artefact.
        """
        seen = getattr(self, "_spawned_agents", None)
        if seen is None:
            seen = self._spawned_agents = set()
        if agent_id in seen:
            return False
        try:
            ego = self._ego
            pos = np.asarray(state["position"], dtype=np.float64)
            if pos.ndim != 1 or pos.size < 2:
                raise ValueError("invalid actor position")
            # Resolve the same current/default dimensions used by the renderer
            # and collision checks; validate the resulting simulation footprint.
            dims = _DEFAULT_AGENT_DIMS.get(track.get("type", "VEHICLE"), (4.5, 1.8, 1.5))
            length, width = track_dims(track, self._scenario_timestep,
                                       fallback=(dims[0], dims[1]))
            actor_box = (float(pos[0]), float(pos[1]), float(state["heading"]),
                         float(length), float(width))
            ego_box = (float(ego.x), float(ego.y), float(ego.heading),
                       float(ego.cfg.ego_length), float(ego.cfg.ego_width))
            clearance = float(self.SPAWN_CLEARANCE_M)
            if (not np.isfinite((*actor_box, *ego_box, clearance)).all()
                    or min(*actor_box[3:], *ego_box[3:]) <= 0.0 or clearance < 0.0):
                raise ValueError("invalid footprint geometry")
            actor_poly = CollisionDetector.make_box_polygon(*actor_box)
            ego_poly = CollisionDetector.make_box_polygon(*ego_box)
            if (actor_poly.is_empty or ego_poly.is_empty
                    or not actor_poly.is_valid or not ego_poly.is_valid):
                raise ValueError("invalid footprint polygon")
            gap = float(actor_poly.distance(ego_poly))
            if not np.isfinite(gap) or gap < 0.0:
                raise ValueError("invalid footprint distance")
            if gap >= clearance and not actor_poly.intersects(ego_poly):
                seen.add(agent_id)
                return False
            reason = (f"footprint clearance {gap:.3f} m; requires at least "
                      f"{clearance:.3f} m and no contact")
        except Exception as exc:  # Bad spawn geometry remains unadmitted.
            reason = f"invalid first-appearance geometry ({type(exc).__name__}: {exc})"
        if agent_id not in getattr(self, "_spawn_held_logged", set()):
            self._spawn_held_logged = getattr(self, "_spawn_held_logged", set())
            self._spawn_held_logged.add(agent_id)
            logger.warning(
                "spawn guard: holding %s at frame %s — %s",
                agent_id, self._scenario_timestep, reason)
        return True

    def _collect_agent_states_for_renderer(self) -> List[Dict]:
        """Build the renderer/collision agent-state list at the current step.

        Ported from ``ScenarioReplayEnv._collect_agent_states_for_renderer``.
        When the IsaacSim replay manager is present it reads poses from the
        manager's track data; otherwise it falls back to the pure-Python
        log-replay agent list already maintained by :meth:`_update_agents`.
        """
        manager = getattr(self, "agent_manager", None)
        if manager is None or self._scenario_data is None:
            return self._agent_states

        tracks = self._scenario_data.get("tracks", {})
        ego_id = getattr(manager, "ego_agent_id", None)
        # Semi-reactive traffic owns some agents' motion: their live poses
        # override the logged track state so collision checks and the
        # renderer see the IDM-integrated position, not the log.
        overrides = getattr(
            getattr(self, "_traffic_manager", None), "pose_overrides", None) or {}
        states: List[Dict] = []
        # Past the end of the log every raw lookup returns None, which dropped
        # the WHOLE agent set from this list -- and with it the topdown, the
        # renderer's DynamicObjects (absent actors are relocated off-screen)
        # and every collision/TTC check. The overrides line below never ran
        # either, so the IDM poses semi_reactive computes for a taken-over
        # vehicle past the log were discarded here, one layer above the traffic
        # manager that produced them: an episode outliving its 20 s bundle drove
        # down an empty road. Clamp the read the way the replay manager already
        # clamps its prims (scenario_replay_manager.update_agents), so BEV,
        # camera and collisions agree on the same held pose; an override alone
        # is enough to keep an agent, since a taken-over vehicle's live pose
        # owes nothing to the log.
        ti = self._scenario_timestep
        _slen = int(getattr(manager, "scenario_length", 0) or 0)
        if _slen:
            ti = min(ti, _slen - 1)
        for agent_id, track in tracks.items():
            state = manager._get_state_at_timestep(track, ti)
            override = overrides.get(agent_id)
            if override is None and (state is None or not state.get("valid", True)):
                continue
            if override is not None:
                state = {**(state or {}), **override}
            # A track whose `valid` window STARTS mid-episode is placed at its
            # logged pose, and the log assumes the ego has already vacated that
            # stretch. A closed-loop ego that is slower has not: on
            # navhard421/0c49c66ca32551c2 drivor ran 44 m behind the logged ego,
            # and 04a8375d492a5682 -- valid only from frame 163 -- materialised
            # 0.87 m from it, scoring an instant `rear_end` no policy could
            # avoid and truncating the episode 38 frames early. The slower the
            # policy, the likelier this is, so it penalises exactly the runs it
            # should not.
            #
            # So a first appearance is HELD while it would overlap the ego, and
            # admitted on the first frame it does not. Holding rather than
            # dropping keeps the actor in the episode: the log gives it no
            # trajectory before this frame anyway, so nothing is lost but the
            # teleport onto the ego.
            if override is None and agent_id != ego_id:
                if self._spawn_blocked_by_ego(agent_id, state, track):
                    continue
            pos = np.asarray(state.get("position", [0, 0, 0]), dtype=np.float32)
            if len(pos) < 3:
                pos = np.array([pos[0], pos[1], 0.0], dtype=np.float32)
            # The z above is the actor's OWN logged altitude, in the same
            # absolute frame the reconstruction was baked in (29.6 m on
            # 17cac31ef9135faf, 605.5 m on 4528c271d89c53e1), and it is a box
            # CENTRE, which is where the render server places a baked actor.
            # It used to be clamped to 0 here for every agent, leaving
            # nurec_grpc._actor_to_world to rebuild it from the EGO's road
            # height plus half the actor's box -- a stand-in that runs +0.35 m
            # high at the median, which lifted a car off the surface its
            # gaussians were fitted on: the red van on 17cac31ef9135faf frame 0
            # sat 65 px above where `nre render` puts it offline, and 0.8 px
            # after this line went.
            #
            # An IDM-driven actor still arrives with no z, and still gets 0:
            # its pose comes from the override merged above, which is planar
            # (semi_reactive._objects builds `position` from
            # `state["position"][:2]`), so the `len(pos) < 3` fill catches it
            # and _actor_to_world's estimate takes over. That estimate is
            # documented there as the approximation it is.
            agent_type = track.get("type", "VEHICLE")
            def_dims = _DEFAULT_AGENT_DIMS.get(agent_type, (4.5, 1.8, 1.5))
            # Same per-frame dims resolution as _update_agents: dims live
            # under track["state"], not the track top level.
            length, width = track_dims(
                track, ti, fallback=(def_dims[0], def_dims[1]))
            states.append({
                "id": agent_id,
                "position": pos,
                # The EPDMS scorer's TTC projects agents forward at constant
                # velocity, so this list is only equivalent to _update_agents'
                # if it carries velocity too — without it every agent looks
                # parked and TTC scores a clean 1.0 against moving traffic.
                "velocity": self._track_velocity(
                    track, state, agent_id in overrides, sampled_timestep=ti),
                "heading": float(state.get("heading", 0.0)),
                "length": length,
                "width": width,
                "height": track_height(track, ti, fallback=def_dims[2]),
                "type": agent_type,
                "is_ego": agent_id == ego_id,
            })
        return states

    def _describe_contact(self) -> Dict[str, Any]:
        """Which agent the ego is touching, and where it sits relative to it.

        NavSafe's ``Termination.detail`` wants "hit agent a17 rear_end", and
        an at-fault verdict that disagrees with the picture is undebuggable
        without the geometry it was derived from — so record both.
        """
        from math import cos, sin

        ex, ey = float(self._ego.x), float(self._ego.y)
        heading = float(self._ego.heading)
        forward = np.array([cos(heading), sin(heading)])
        ego_poly = CollisionDetector.make_box_polygon(
            ex, ey, heading, self._ego.cfg.ego_length, self._ego.cfg.ego_width)
        for agent in self._agent_states:
            if agent.get("is_ego"):
                continue
            poly = CollisionDetector.make_box_polygon(
                float(agent["position"][0]), float(agent["position"][1]),
                float(agent.get("heading", 0.0)),
                float(agent.get("length", 4.5)), float(agent.get("width", 1.8)))
            if not ego_poly.intersects(poly):
                continue
            delta = np.array([float(agent["position"][0]) - ex,
                              float(agent["position"][1]) - ey])
            longitudinal = float(delta @ forward)
            return {
                "agent_id": agent.get("id"),
                "agent_type": agent.get("type"),
                "longitudinal_m": longitudinal,
                "lateral_m": float(delta @ np.array([-forward[1], forward[0]])),
                "ego_speed": float(self._ego.speed),
                "ego_heading": heading,
                "kind": "front" if longitudinal > 0.0 else "rear_end",
            }
        # OBB found nothing: the contact came from the PhysX impulse path,
        # which carries no relative geometry.
        return {"agent_id": None, "kind": "physical_impulse",
                "ego_speed": float(self._ego.speed), "ego_heading": heading}

    def _track_velocity(self, track: Dict, state: Dict,
                        overridden: bool, *,
                        sampled_timestep: Optional[int] = None) -> np.ndarray:
        """Planar velocity for one agent at the current scenario timestep.

        A semi-reactive override carries its own IDM velocity. A replay pose
        held beyond its sampled frame is stationary; at logged frames retain
        the existing finite differences, including the final logged velocity.
        ``sampled_timestep`` also accounts for a scenario-level clamp earlier
        than the end of an individual track's position array.
        """
        if overridden and state.get("velocity") is not None:
            v = np.asarray(state["velocity"], dtype=np.float32)
            if v.ndim == 0:
                # SemiReactiveTraffic stores the IDM's scalar along-route
                # speed; the scorer wants a planar vector, so resolve it on
                # the agent's own heading.
                yaw = float(state.get("heading", 0.0))
                return np.array([float(v) * np.cos(yaw),
                                 float(v) * np.sin(yaw)], dtype=np.float32)
            return v[:2]
        positions = track.get("state", {}).get("position")
        if positions is None or len(positions) == 0:
            return np.zeros(2, dtype=np.float32)
        t = min(self._scenario_timestep if sampled_timestep is None or overridden
                else sampled_timestep, len(positions) - 1)
        if not overridden and self._scenario_timestep > t:
            return np.zeros(2, dtype=np.float32)
        dt = float(self.cfg.dt)
        if t + 1 < len(positions):
            a, b = positions[t], positions[t + 1]
        elif t > 0:
            a, b = positions[t - 1], positions[t]
        else:
            return np.zeros(2, dtype=np.float32)
        return ((np.asarray(b[:2], dtype=np.float32)
                 - np.asarray(a[:2], dtype=np.float32)) / dt)

    def set_ego_override(
        self,
        position: np.ndarray,
        heading: float,
        velocity: Optional[np.ndarray] = None,
    ) -> None:
        """Override the ego pose so the next step follows the supplied pose.

        Ported from ``ScenarioReplayEnv.set_ego_override``. The evaluator
        calls this each frame to drive the ego along a model-planned
        trajectory instead of replaying it from the log. Under IsaacSim the
        override is applied to the ego prim in :meth:`_pre_physics_step`; the
        pure-Python ego fields are kept in sync via :meth:`_sync_base_ego_state`.

        Args:
            position: ``(2,)`` or ``(3,)`` world-frame position ``[x, y, (z)]``.
            heading: Yaw in radians.
            velocity: Optional ``(2,)``/``(3,)`` world-frame velocity.
        """
        pos = np.asarray(position, dtype=np.float32)
        if len(pos) == 2:
            pos = np.array([pos[0], pos[1], 0.0], dtype=np.float32)

        vel = np.zeros(3, dtype=np.float32)
        if velocity is not None:
            v = np.asarray(velocity, dtype=np.float32)
            vel[: len(v)] = v[:3]

        self._ego_override_active = True
        self._ego_override_state = {
            "position": pos,
            "heading": heading,
            "velocity": vel,
            "speed": float(np.linalg.norm(vel[:2])),
            "valid": True,
            "height": 1.5,
        }
        # Keep the pure-Python ego fields aligned immediately so a caller
        # reading get_ego_state() before the next step sees the override.
        self._sync_base_ego_state()

    def clear_ego_override(self) -> None:
        """Disable the ego override — resume log replay for the ego.

        Ported from ``ScenarioReplayEnv.clear_ego_override``.
        """
        self._ego_override_active = False
        self._ego_override_state = None

    def set_ego_externally_driven(self, active: bool = True) -> None:
        """Use the integrated ego state for action-driven execution.

        The traffic manager skips the logged ego track while this is active.
        """
        self._ego_externally_driven = bool(active)
        manager = getattr(self, "_traffic_manager", None)
        if manager is not None and hasattr(manager, "skip_ego"):
            manager.skip_ego = bool(active)

    # ------------------------------------------------------------------
    # Evaluator-driver contract (load / info / GT-replay actions).
    # The Evaluator (navsafe.evaluation.evaluator) drives the env through
    # these three entry points; they were part of the retired
    # ScenarioReplayEnv API and are reinstated here on the consolidated env.
    # ------------------------------------------------------------------

    def load_scenario_from_path(self, scenario_path: Any) -> None:
        """Load the scenario for evaluation and reset the ego to frame 0.

        For ``scenario_source="py123d"`` the Arrow scene is read directly via
        the configured loader (``scenario_path`` is informational — the data
        root comes from ``cfg.py123d_data_root``). The USD scene itself is built
        once in :meth:`_setup_scene` during ``__init__``; this populates the
        scenario data (if not already) and resets the ego to the first frame.
        """
        self._ensure_scenario_loaded()
        # Reset ego / counters to the scenario's first frame.
        self.reset()

    def get_scenario_info(self) -> Dict[str, Any]:
        """Return the active ``ScenarioDescription`` dict.

        The Evaluator reads ``.get("length")`` and passes the dict to the
        scorer; :attr:`current_scenario` is the same object.
        """
        return self._scenario_data or {}

    def get_gt_ego_action(self, frame: int) -> np.ndarray:
        """Open-loop warm-up: snap the ego to its ground-truth pose.

        During the ego-replay warm-up the Evaluator wants the ego to follow
        the recorded SDC track exactly so the policy sees realistic frames.
        We set the ego override to the GT pose for the *next* step (so
        ``env.step()`` reproduces ``GT[frame+1]``) and return a no-op control
        action — the override wins over the bicycle dynamics in
        :meth:`_pre_physics_step` / :meth:`_update_agents`.
        """
        sd = self._scenario_data or {}
        meta = sd.get("metadata", {})
        sdc_id = str(meta.get("sdc_id", "ego"))
        state = sd.get("tracks", {}).get(sdc_id, {}).get("state", {})
        pos = state.get("position")
        heading = state.get("heading")
        if pos is not None and len(pos) > 0:
            i = min(int(frame) + 1, len(pos) - 1)
            p = np.asarray(pos[i], dtype=np.float32)
            h = (
                float(heading[i])
                if heading is not None and len(heading) > i
                else float(self._ego.heading)
            )
            if i + 1 < len(pos):
                vel = (np.asarray(pos[i + 1], dtype=np.float32)[:2] - p[:2]) / float(self.cfg.dt)
            elif i > 0:
                vel = (p[:2] - np.asarray(pos[i - 1], dtype=np.float32)[:2]) / float(self.cfg.dt)
            else:
                vel = np.zeros(2, dtype=np.float32)
            self.set_ego_override(position=p, heading=h, velocity=vel)
        return np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # UrbanVerse / scene-dressing helpers (ported from ScenarioReplayEnv)
    #
    # These are only reached from the IsaacSim _setup_scene path. They are
    # defined unconditionally (so the class surface matches) but every USD
    # import is performed lazily inside the method body.
    # ------------------------------------------------------------------


    def _ego_start_xy(self) -> Optional[Tuple[float, float]]:
        """Return the ego start ``(x, y)``, or ``None`` if unavailable.

        Ported from ``ScenarioReplayEnv._ego_start_xy``.
        """
        xyh = self._ego_start_xyh()
        return (xyh[0], xyh[1]) if xyh is not None else None

    def _ego_start_xyh(self) -> Optional[Tuple[float, float, float]]:
        """Return the ego start ``(x, y, heading)``, or ``None`` if unavailable.

        Ported from ``ScenarioReplayEnv._ego_start_xyh``. Heading is derived
        from the SDC track's initial motion, falling back to the stored
        heading then ``0.0``.
        """
        try:
            scenario = self._scenario_data
            if not isinstance(scenario, dict):
                return None
            meta = scenario.get("metadata", {})
            tracks = scenario.get("tracks", {})
            ego_id = meta.get("sdc_id")
            if ego_id is None or ego_id not in tracks:
                ego_id = "ego" if "ego" in tracks else None
            if ego_id is None:
                return None
            state = tracks[ego_id].get("state", {})
            pos = state.get("position")
            if pos is None or len(pos) == 0:
                return None
            import math

            arr = np.asarray(pos)
            p0 = arr[0]
            x0, y0 = float(p0[0]), float(p0[1])
            j = min(20, len(arr) - 1)
            heading = 0.0
            if j > 0:
                dx = float(arr[j][0]) - x0
                dy = float(arr[j][1]) - y0
                if (dx * dx + dy * dy) > 1e-4:
                    heading = math.atan2(dy, dx)
                else:
                    hd = state.get("heading")
                    if hd is not None and len(hd) > 0:
                        heading = float(np.asarray(hd)[0])
            return (x0, y0, heading)
        except Exception:
            return None


    def _build_combined_lidar_mesh(self) -> 'str | None':
        return self._scene._build_combined_lidar_mesh()

    def _resolve_ego_prim_path(self, configured: str) -> str:
        return self._scene._resolve_ego_prim_path(configured)


    def _setup_extra_sensors(self) -> None:
        """Delegate to :meth:`SensorRig.setup` (the ego sensor subsystem)."""
        self._sensors.setup()

    def get_lidar_returns(self) -> Optional[np.ndarray]:
        """Return the latest ray-cast LiDAR hit positions or ``None``.

        Delegates to :meth:`SensorRig.get_lidar_returns`.
        """
        return self._sensors.get_lidar_returns()

    def get_imu_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Return ``{lin_acc_b, ang_vel_b, quat_w}`` ndarrays or ``None``.

        Delegates to :meth:`SensorRig.get_imu_state`.
        """
        return self._sensors.get_imu_state()

    def get_contact_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Return contact-sensor net forces / pose, or ``None``.

        Delegates to :meth:`SensorRig.get_contact_state`.
        """
        return self._sensors.get_contact_state()




    def get_ego_front_image_bgr(self) -> Optional[np.ndarray]:
        return self._cameras.get_ego_front_image_bgr()

    def get_frame_transforms(self) -> Optional[Dict[str, np.ndarray]]:
        """Return ego-to-target SE(3) transforms or ``None``.

        Delegates to :meth:`SensorRig.get_frame_transforms`.
        """
        return self._sensors.get_frame_transforms()

    def _get_observations(self) -> Dict[str, Any]:
        """Compute observations based on cfg.obs_kind."""
        if self.cfg.obs_kind == "sensor":
            return {"policy": np.zeros(1, dtype=np.float32)}
        else:
            ego_state = self.get_ego_state()
            return {
                "policy": np.array([
                    ego_state["x"],
                    ego_state["y"],
                    ego_state["heading"],
                    ego_state["speed"],
                ], dtype=np.float32),
            }

    def _get_rewards(self) -> Any:
        """Compute rewards (DirectRLEnv interface)."""
        return np.zeros(1, dtype=np.float32)

    def _get_dones(self) -> Tuple[Any, Any]:
        """Compute done flags (DirectRLEnv interface)."""
        return np.array([False]), np.array([False])

    def _reset_idx(self, env_ids: Any) -> None:
        """Reset specific environment indices."""
        # Reset the traffic manager for the new episode (rebuilds replay
        # cursors / IDM actor state). Only meaningful on the IsaacSim path
        # where an agent_manager exists; the pure-Python reset() resets the
        # replay cursor directly via _scenario_timestep.
        if _HAS_ISAACLAB and getattr(self, "agent_manager", None) is not None:
            self._traffic_manager.reset(self)

    # ------------------------------------------------------------------
    # Cleanup (from abstract_env.py / abstract_rl_env.py)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            renderer.close()
            self._renderer = None
            self._renderer_scenario_id = None
