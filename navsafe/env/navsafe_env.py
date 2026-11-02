"""NavSafeEnv: the simulation environment.

It holds the scenario, steps the ego (by override or bicycle model),
lets a traffic manager move the other agents, detects collisions and asks the
configured render backend for camera images. Behaviour is selected through
``EnvCfg``: scenario source, traffic mode, execution mode and render backend.

The state is symbolic and stepped in pure Python; the NuRec renderer draws the
scene, so no simulator or USD stage is involved.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np

if TYPE_CHECKING:
    from navsafe.scenario.scenario_description import ScenarioDescription

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


# Stateless setup helpers (supported configurations, loader and traffic-
# manager factories, configuration and asset checks) live in `_bootstrap.py`
# and are re-exported here.
from navsafe.env._bootstrap import (  # noqa: E402  (after the lazy-import guards above)
    _DEFAULT_AGENT_DIMS,
    SUPPORTED_COMBINATIONS,
    _check_asset_bundles,
    _format_supported_combinations,
    _get_cache_root,
    _make_loader,
    _make_traffic_manager,
    _validate_cfg,
)

# Camera subsystem, extracted from this class (it owns its own state behind a
# narrow interface; the env composes + delegates to it).
from navsafe.env.camera_manager import CameraManager  # noqa: E402


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


def _configure_seed(seed: int) -> None:
    """Seed python, numpy and torch, and turn cuDNN autotuning on.

    The same calls, in the same order, as IsaacLab's ``configure_seed(seed)``
    (``torch_deterministic=False``), which the env's former IsaacLab base
    class ran for a configured seed.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ======================================================================
# NavSafeEnv
# ======================================================================


class NavSafeEnv:
    """The single configurable NavSafe environment.

    Replaces the seven retired env classes. Folds ego dynamics,
    collision detection, renderer wiring (from base_driving_env.py),
    training step/reset/reward (from training_env.py), and
    manager-based patterns (from abstract_env.py, abstract_rl_env.py).
    """

    cfg: EnvCfg

    #: One environment per process; step() returns ``(num_envs,)`` done flags.
    num_envs: int = 1

    # Agent-id sentinel used for back-compat with the ego-only camera API.
    # Any call site that still passes ``agent_id="__ego__"`` (or omits it)
    # renders from the ego rig. Ported verbatim from
    # ``ScenarioReplayEnv.EGO_AGENT_ID``.
    EGO_AGENT_ID: str = "__ego__"

    def __init__(self, cfg: EnvCfg) -> None:
        """Initialize NavSafeEnv.

        Args:
            cfg: Environment configuration dataclass.

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
        # _ego_externally_driven marks the controller execution mode
        # (ego driven by actions; the replay must follow self._ego).
        self._ego_externally_driven: bool = False
        self._ego_override_active: bool = False
        self._ego_override_state: Optional[Dict[str, Any]] = None

        # Traffic manager, selected by cfg.traffic_mode. The
        # configuration has been validated, so unsupported modes do
        # not reach this point. The managers are pure Python.
        self._traffic_manager: Any = _make_traffic_manager(cfg)

        # NB: the camera subsystem (self._cameras) is a LAZY property below, not
        # constructed here — capture only happens post-construction, and lazy
        # init keeps the camera wrappers working on minimally-built instances
        # (e.g. ``object.__new__(NavSafeEnv)`` in the renderer-wiring tests).

        # An explicit seed reseeds every RNG here, before the scenario is
        # loaded, with cuDNN autotuning on -- what the IsaacLab base class
        # did at this point, so seeded runs reproduce their earlier results.
        if cfg.seed is not None:
            _configure_seed(int(cfg.seed))

        self._setup_scene()

        logger.info(
            "NavSafeEnv initialized: obs_kind=%s, traffic_mode=%s, "
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
        action tensor passed to :meth:`step`, which brings it back to host.
        The state is stepped on the CPU.
        """
        return "cpu"

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
            # No scenario was given and none is loaded: load one
            # from the configured source, if it has input
            # configured. Otherwise the default ego reset
            # applies.
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
                # Start the ego at its logged speed. Replayed
                # traffic does not react, so an ego starting
                # from rest inside moving traffic would be
                # hit from behind.
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
        # Clear any ego override left from the previous episode,
        # which would otherwise pin the ego to an old pose.
        self.clear_ego_override()

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

        # Step the ego. An active override sets the pose directly
        # (teleport execution and warm-up replay); otherwise the
        # bicycle model integrates the state.
        if self._ego_override_active and self._ego_override_state is not None:
            self._sync_base_ego_state()
            # step_ego is skipped, so advance the ego frame counter here to
            # keep env.frame / info["frame"] truthful during overrides.
            self._ego.frame += 1
            state = {"speed_kmh": self._ego.speed * 3.6}
        else:
            state = self.step_ego(steer, accel)
        self._episode_step += 1
        self._common_step_counter += 1
        self._scenario_timestep += 1

        # Advance a stage-free manager before the agent list is
        # rebuilt, so that the list reflects this frame (see
        # navsafe.traffic.stage_free).
        _tm = getattr(self, "_traffic_manager", None)
        # `_tm is not None` is already implied by the getattr default below
        # (None has no `stage_free`); spelled out so the checker sees it too.
        if _tm is not None and getattr(_tm, "stage_free", False):
            _tm.step(self, self.cfg.dt)

        # Update agents (traffic manager hook)
        self._update_agents()

        # Step the other traffic managers (log replay, semi-reactive)
        # after the rebuild.
        self._step_replay_traffic()

        # Refresh the symbolic agent list for every traffic mode
        # except log replay, where _update_agents() above already
        # did. Collision and time-to-collision checks read this list.
        if (self.cfg.traffic_mode != "log_replay"
                and not getattr(getattr(self, "_traffic_manager", None), "stage_free", False)
                and self._scenario_data is not None):
            self._agent_states = self._collect_agent_states_for_renderer()

        # Hand the renderer this frame's agent poses. Without this
        # nurec_grpc falls back to the poses it cached at scene load: a
        # reactive actor would yield in the collision and TTC numbers while
        # sailing along its spawn-time rail on camera.
        renderer = getattr(self, "_renderer", None)
        if renderer is not None and hasattr(renderer, "update_agents"):
            try:
                renderer.update_agents(self._agent_states)
            except Exception:  # noqa: BLE001 — a render hiccup must not end the episode
                logger.exception("failed to hand agent states to the renderer")

        # Check collision.
        self._collision, self._collision_at_fault = self.check_collision_at_fault()
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
        # Return terminated/truncated as (num_envs,) arrays; the
        # evaluator calls ``.any()`` on them.
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
        the scenario data at the current ``_scenario_timestep``, where
        agents are moved to their recorded positions each frame.

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
                # Per-frame box dimensions are stored under
                # track["state"]. Using a default car-sized
                # box for every agent would report collisions
                # with small agents that were never touched.
                def_l, def_w, def_h = _DEFAULT_AGENT_DIMS.get(
                    agent_type, (4.5, 1.8, 1.5))
                length, width = track_dims(track, t, fallback=(def_l, def_w))
                # Keep the actor's own logged altitude, which
                # is in the frame the reconstruction was
                # built in, so the renderer places it on the
                # surface it was reconstructed on.
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

            # For a reactive agent the logged track is only its
            # initial condition; the traffic manager owns its
            # pose. Merge the live pose over the logged row so
            # that collision checks, the bird's-eye view and the
            # renderer agree. An agent with no logged row at this
            # frame is appended.
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

    # Scene lifecycle.

    def _ensure_scenario_loaded(self) -> None:
        """Load the scenario via the configured loader if not already set.

        :meth:`_setup_scene` runs in ``__init__``, before the first
        :meth:`reset`, and needs the scenario; this dispatches the loader on
        demand. It is a no-op when scenario data is already present or when
        the source has no input configured.
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
        :class:`~navsafe.env.navsafe_edit_env.NavSafeEditEnv`) override it
        to apply declarative scenario edits. Called at both load sites
        (``_ensure_scenario_loaded`` during ``_setup_scene`` and the
        loader-dispatch branch of :meth:`reset`).
        """
        return sd

    def _setup_scene(self) -> None:
        """Load the scenario and index its agents; install it in the renderer.

        Runs once from ``__init__``, before the first :meth:`reset`.
        """
        self._ensure_scenario_loaded()
        scenario = self._scenario_data
        self._scenario_timestep = 0
        if scenario is None:
            return
        self._build_scenario_content(scenario)
        self._setup_renderer_for_scenario(scenario, force=True)

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

    def _build_scenario_content(self, scenario: Any) -> None:
        """Index the scenario's agents for per-timestep state queries.

        Assigns ``self.agent_manager``, which the traffic managers and
        :meth:`_collect_agent_states_for_renderer` read.
        """
        from navsafe.manager.scenario_replay_manager import (
            ScenarioReplayManager,
        )

        self.agent_manager = ScenarioReplayManager(
            scenario=cast("ScenarioDescription", scenario),
            num_envs=self.num_envs,
        )

    def _step_replay_traffic(self) -> None:
        """Step a traffic manager that is not stage-free (log replay, semi-reactive).

        A stage-free manager was already stepped in :meth:`step`; stepping it
        again here would advance its actors twice per frame.
        """
        if getattr(self, "agent_manager", None) is None:
            return
        if not getattr(self._traffic_manager, "stage_free", False):
            self._traffic_manager.step(self, self.cfg.dt)

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
            # Diagnostic only: EGO_LATERAL_SHIFT_M shifts the
            # replayed ego pose sideways (positive to the left)
            # to inspect the reconstruction away from the
            # recorded trajectory. Unset by default.
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
        When the replay manager is present it reads poses from the manager's
        track data; otherwise it falls back to the log-replay agent list
        already maintained by :meth:`_update_agents`.
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
        # Beyond the end of the log, clamp the lookup to the last
        # frame, as the replay manager does for its cursor. Otherwise
        # every agent would vanish from this list, and with it from
        # the renderer and the collision checks. An agent with a live
        # override is kept regardless.
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
            # A track that first becomes valid mid-episode
            # appears at its logged pose, where the log assumes
            # the ego has already passed. A slower closed-loop
            # ego may still be there. The first appearance is
            # therefore held back while it would overlap the ego
            # and admitted on the first frame it does not.
            if override is None and agent_id != ego_id:
                if self._spawn_blocked_by_ego(agent_id, state, track):
                    continue
            pos = np.asarray(state.get("position", [0, 0, 0]), dtype=np.float32)
            if len(pos) < 3:
                pos = np.array([pos[0], pos[1], 0.0], dtype=np.float32)
            # The z above is the actor's logged altitude at its
            # box centre, which is where the renderer places a
            # reconstructed actor. An agent driven by the traffic
            # manager has a planar pose and gets z = 0 here; the
            # renderer then estimates its height.
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
        # OBB found nothing: the collision check's geometry disagrees with
        # this one, so there is no relative geometry to report.
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
        trajectory instead of replaying it from the log. The ego fields are
        kept in sync via :meth:`_sync_base_ego_state`.

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

    # Entry points used by the evaluator: loading a scenario, per-step
    # info and replay actions.

    def load_scenario_from_path(self, scenario_path: Any) -> None:
        """Load the scenario for evaluation and reset the ego to frame 0.

        For ``scenario_source="py123d"`` the Arrow scene is read directly via
        the configured loader (``scenario_path`` is informational — the data
        root comes from ``cfg.py123d_data_root``). The scene is set up once in
        :meth:`_setup_scene` during ``__init__``; this populates the scenario
        data (if not already) and resets the ego to the first frame.
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
        :meth:`step`.
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

    def get_ego_front_image_bgr(self) -> Optional[np.ndarray]:
        return self._cameras.get_ego_front_image_bgr()

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
