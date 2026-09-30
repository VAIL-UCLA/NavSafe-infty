"""
navsafe.evaluation.evaluator

Closed-loop evaluation orchestrator wired to UrbanSim's ScenarioReplayEnv.

Data flow per step:
  ScenarioReplayEnv.get_ego_state()        → ego_state dict
  ScenarioReplayEnv.get_camera_images()    → {cam_name: (H,W,3) ndarray}
  BaseModelAdapter.prepare_input()         → model_input
  BaseModelAdapter.run_inference()         → model_output
  BaseModelAdapter.parse_output()          → {'trajectory': (N,2), ...}
  TrajectoryTracker.compute()              → (steer, accel)  [controller/physics modes]
  ScenarioReplayEnv.step(action_tensor)    → (obs, rewards, terminated, truncated, info)
  EPDMSLiveScorer.score_frame_live()       → per-frame EPDMS metrics
"""

from typing import Dict, Optional, Any, List, Sequence, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
import math
import sys
import time as _time
from collections import defaultdict as _defaultdict

import numpy as np

# ── Lazy/guarded heavy imports ────────────────────────────────────────
# torch is only needed to build the action tensor in _step(); vis_utils pulls
# cv2 and is only needed on the (opt-in) visualization path. Guarding both lets
# this module import on a system without the heavy stack — which is what makes
# the evaluator import-verifiable (the structural safety net relies on it) and
# matches the graceful-degradation contract the rest of navsafe follows
# (Req 11.1). Each is exercised only when its feature is actually used; a None
# here surfaces a clear failure at that call site, not at import time.
try:
    import torch
except ImportError:  # pragma: no cover — torch is NVIDIA-index-only
    torch = None  # type: ignore[assignment]


try:
    from navsafe.evaluation import vis_utils
except ImportError:  # pragma: no cover — vis_utils imports cv2 (viz extra)
    vis_utils = None  # type: ignore[assignment]

# Collaborator subsystem extracted from this orchestrator: all per-frame /
# end-of-run artifact + visualisation output. Import-safe (it guards vis_utils
# the same way and does not import this module back).
from navsafe.evaluation.eval_artifacts import EvalArtifactWriter
# The scorer owns the number; importing it keeps the loop's ceiling and the
# t_max it is judged against from drifting apart.
from navsafe.benchmark.scoring.from_run import SAFETY_CEILING_S
from navsafe.evaluation.route_manager import RouteManager
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
from navsafe.core.execution_clock import (
    live_execution_dt, require_cached_clock, require_configured_clock, synchronize_tracker_dt,
)

logger = logging.getLogger(__name__)


def _monotone_projected_route_arc(
        route_xy: np.ndarray, ego_xy: Sequence[np.ndarray]) -> np.ndarray:
    """Nearest-route arc reached at each frame, made completion-monotone.

    Route completion is accumulated progress: once the ego reaches an arc it
    cannot lose that credit by passing the endpoint or moving away later.
    The former final-pose-only projection made 0bca... reach the last logged
    waypoint, continue safely beyond the short log, then regress to 99.63%
    when its final pose happened to be closest to the penultimate sample.
    """
    route = np.asarray(route_xy, dtype=np.float64)[:, :2]
    ego = np.asarray(ego_xy, dtype=np.float64)[:, :2]
    if len(route) < 2 or len(ego) == 0:
        return np.zeros(len(ego), dtype=np.float64)
    arc = np.concatenate([
        [0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    projected = np.asarray([
        arc[int(np.argmin(np.linalg.norm(route - pos, axis=1)))]
        for pos in ego
    ], dtype=np.float64)
    return np.maximum.accumulate(projected)

from navsafe.core import plan_execution as _plan_exec
from navsafe.core.plan_execution import PLAN_PACING_FACTOR  # noqa: F401

# The composite scores an episode is judged by. ``finalize`` guarantees all
# of them are present in ``metrics.json``, defaulting to 0.0: each is written
# by a block that can be skipped or can throw, and a key that is sometimes
# absent is indistinguishable from "this episode cannot be scored" — the
# ambiguity that let unscored scenes disappear from published means instead of
# contributing the 0 they earned. A 0 here is the honest reading: no progress
# was demonstrated, so none is credited.
EPISODE_SCORE_KEYS: Tuple[str, ...] = (
    "epdms",
    "epdms_no_ep",
    "driving_score",
    "route_completion_fraction",
    "ego_progress",
)

# The EPDMS sub-terms. Deliberately NOT defaulted: 0.0 in one of these is a
# *verdict about the ego* ("at-fault collision", "off the drivable area"), and
# admission gates and the report card read them as measured safety evidence.
# Imputing a zero would manufacture an infraction for an episode that was
# never scored — worse than the omission this module set out to fix. Absent
# means unmeasured, and those consumers already refuse loudly on absence.
EPDMS_SUBSCORE_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "extended_comfort",
)


@dataclass
class EvaluationConfig:
    """Configuration for a single evaluation run.

    Attributes:
        traffic_mode: 'no_traffic', 'log_replay', 'semi_reactive' or
            'navsafe' (drives a recipe's inserted actors); 'idm' is refused
            (unwired silent no-op)
        eval_mode: 'closed_loop' (the only implemented mode; 'open_loop' is
            refused — use ego_replay_frames >= eval_frames for a pure
            log-replay run)
        controller_type: 'pid', 'pure_pursuit', or 'lqr' (tracker used by the
            controller/physics execution modes)
        execution_mode: 'teleport', 'controller' or 'physics'. This dataclass
            still defaults to 'teleport' for the callers that construct it
            directly; eval_py123d.py's CLI defaults to 'controller' + 'lqr'
            and always passes both explicitly
        replan_rate: Run model inference every N frames; use cached trajectory in between
        sim_dt: Simulation timestep in seconds
        ego_replay_frames: Replay ground-truth ego actions for first N frames (warm-up)
        eval_frames: Cap evaluation at this many frames (None = full scenario)
        route_time_limit_s: Maximum scored driving time in seconds. ``None``
            means no clock-based route-completion limit; semantic termination
            conditions still apply.
        save_per_frame: Persist per-frame metrics to disk
        output_dir: Root directory for all result files
        enable_vis: Save per-frame BEV / front-cam images and generate GIFs
        vis_online: Show BEV / front-cam in live OpenCV windows during eval
        vis_extra_cameras: NAVSIM camera names rendered for the artifacts only
            (e.g. ``("CAM_B0",)`` for a rear view). The policy never sees them:
            they are unioned into the capture set *after* the adapter names its
            own cameras, and adapters index ``images`` by key, so an extra entry
            is inert. Each costs one render per frame.
        ego_perturb_lateral_m / ego_perturb_longitudinal_m /
        ego_perturb_yaw_deg: a one-shot rigid displacement of the ego, applied
            on the hand-off frame — the first frame the policy owns the ego,
            i.e. ``ego_replay_frames``. See
            :meth:`Evaluator._apply_handoff_perturbation`.
    """
    traffic_mode: str = "log_replay"
    eval_mode: str = "closed_loop"
    controller_type: str = "pure_pursuit"
    # How the ego executes the planned trajectory:
    #   "teleport"   — arc-length paced teleport (BridgeSim-parity scoring;
    #                  the default: all benchmark numbers use this).
    #   "controller" — controller_type tracker computes (steer, accel);
    #                  the bicycle model integrates (tracking dynamics real).
    #   "physics"    — like "controller", but PhysX integrates the ego
    #                  (requires env execution_mode="physics").
    execution_mode: str = "teleport"
    replan_rate: int = 1
    sim_dt: float = 0.1
    ego_replay_frames: int = 0
    eval_frames: Optional[int] = None
    route_time_limit_s: Optional[float] = SAFETY_CEILING_S
    save_per_frame: bool = True
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    enable_vis: bool = False
    vis_online: bool = False
    vis_extra_cameras: tuple = ()
    #: Hand-off perturbation, in the ego's own frame at the hand-off pose:
    #: +lateral is to the ego's LEFT, +longitudinal is FORWARD, +yaw is
    #: counter-clockwise (degrees). All three are applied together, once, on
    #: frame ``ego_replay_frames``. The replay prefix is untouched, so every
    #: perturbed arm of a sweep shares the same warm-up and differs only in the
    #: pose the policy first sees.
    ego_perturb_lateral_m: float = 0.0
    ego_perturb_longitudinal_m: float = 0.0
    ego_perturb_yaw_deg: float = 0.0
    ego_perturb_history: str = "handoff"
    #: Opt-in Section-4 primitive trace. Records no images and performs no
    #: per-frame filesystem writes: the recorder buffers in RAM and emits one
    #: ZIP to ``output_dir`` during finalize.
    record_reactivity_trace: bool = False
    reactivity_condition: str = "auto"
    reactivity_group_id: Optional[str] = None
    reactivity_metadata: dict = field(default_factory=dict)
    reactivity_trace_filename: str = "reactivity_trace.zip"
    #: The bundle's ``manifest.json`` ``scenario_meta``, when the caller has it.
    #: Only its ``taxonomy_leaves`` are read, to resolve the leaf's termination
    #: rules (``navsafe.benchmark.scenario_rules``). This has to reach the LIVE
    #: monitor and not only the scorer: a leaf that suppresses a terminator --
    #: I-2, where driving the wrong way IS the work-zone bypass -- must not have
    #: its episode cut short, and no post-hoc pass can un-truncate a trace that
    #: already stopped. Empty/None keeps the default rules, i.e. today's
    #: behaviour for every leaf but I-2 and C-3.
    scenario_meta: Optional[dict] = None

    def __post_init__(self):
        if self.reactivity_condition not in (
                "auto", "hazard", "irrelevant", "no_change"):
            raise ValueError("unknown reactivity condition")
        trace_name = Path(self.reactivity_trace_filename)
        if (trace_name.name != self.reactivity_trace_filename
                or trace_name.suffix != ".zip"):
            raise ValueError(
                "reactivity trace filename must be a plain .zip filename")
        if self.ego_perturb_history not in ("handoff", "controller"):
            raise ValueError("unknown ego perturbation history mode")
        if self.ego_perturb_history == "controller" and (
                self.execution_mode != "controller" or self.ego_replay_frames < 1):
            raise ValueError("controller history requires controller execution and positive warmup")
        if self.traffic_mode == "idm":
            # The env refuses this mode (unwired silent no-op — see
            # navsafe.env._bootstrap.IDM_TRAFFIC_UNWIRED_MSG); accepting it
            # here would stamp "idm" into metrics.json/results while the run
            # scored something else. Same refusal, one layer earlier.
            raise NotImplementedError(
                "traffic_mode='idm' is not wired (it was a silent no-op — "
                "agents replayed the log). Use 'log_replay', 'semi_reactive' "
                "(the wired reactive mode), or 'no_traffic'.")
        if self.traffic_mode not in (
                "no_traffic", "log_replay", "semi_reactive", "navsafe"):
            raise ValueError(f"Invalid traffic_mode: {self.traffic_mode}")
        if self.eval_mode == "open_loop":
            # "open_loop" was accepted for years but never implemented — the
            # evaluator ran the closed loop regardless, so accepting the value
            # silently mislabelled every such run. Refuse it loudly and name
            # the control that actually exists.
            raise NotImplementedError(
                "eval_mode='open_loop' was parsed but never implemented — the "
                "evaluator always runs closed-loop. For an open-loop (pure "
                "log-replay) run, set ego_replay_frames >= eval_frames "
                "(--ego-replay-frames >= --eval-frames).")
        if self.eval_mode != "closed_loop":
            raise ValueError(f"Invalid eval_mode: {self.eval_mode}")
        if self.controller_type not in ("pid", "pure_pursuit", "lqr"):
            raise ValueError(f"Invalid controller_type: {self.controller_type}")
        if self.execution_mode not in ("teleport", "controller", "physics"):
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}")
        if self.route_time_limit_s is not None:
            self.route_time_limit_s = float(self.route_time_limit_s)
            # CLI/config convention: zero (or a negative value) explicitly
            # disables the clock while retaining semantic terminations.
            if self.route_time_limit_s <= 0.0:
                self.route_time_limit_s = None
        self.output_dir = Path(self.output_dir)
        # --vis-online implies --enable-vis
        if self.vis_online:
            self.enable_vis = True
        self.vis_extra_cameras = tuple(self.vis_extra_cameras or ())
        unknown = [c for c in self.vis_extra_cameras if c not in NAVSIM_CAM_CONFIGS]
        if unknown:
            raise ValueError(
                f"Unknown vis_extra_cameras {unknown}; "
                f"valid NAVSIM cameras: {sorted(NAVSIM_CAM_CONFIGS)}")


class Evaluator:
    """Closed-loop evaluation orchestrator for UrbanSim.

    Owns the simulation loop; delegates physics to ScenarioReplayEnv,
    planning to a BaseModelAdapter, actuation to a TrajectoryTracker
    (controller/physics execution modes) or arc-paced teleport (default),
    and metric accumulation to the EPDMS live scorer
    (scorers/epdms_trajectory_scorer_fast.py::EPDMSLiveScorer — BridgeSim parity).

    Usage (single scenario)::

        env = ScenarioReplayEnv(cfg)
        adapter = create_model_adapter("tcp", checkpoint_path="...")
        adapter.load_model()

        evaluator = Evaluator(env=env, model_adapter=adapter, config=cfg)
        evaluator.setup(scenario_path=Path("/data/scenarios/scene_001"))
        results = evaluator.run()

    Usage (batch — env reused across scenarios)::

        evaluator = Evaluator(env=env, model_adapter=adapter, config=cfg)
        for path in scenario_paths:
            evaluator.setup(scenario_path=path)
            results = evaluator.run()
    """

    def __init__(
        self,
        env: Any,                          # ScenarioReplayEnv
        model_adapter: Any,                # BaseModelAdapter subclass
        config: EvaluationConfig,
        controller: Optional[Any] = None,  # TrajectoryTracker (created from config if None)
        trajectory_scorer: Optional[Any] = None,  # optional candidate scorer (EPDMS, etc.)
        observer_agent_id: Optional[str] = None,  # sensor-only observer; None = ego
    ):
        self.env = env
        self.adapter = model_adapter
        self.config = config
        self.trajectory_scorer = trajectory_scorer
        # Non-ego observer for sensors only; the ego still executes the
        # adapter's planned trajectory. None or EGO_AGENT_ID means the
        # adapter is fed ego images (the historic behaviour).
        self.observer_agent_id: Optional[str] = observer_agent_id
        rgb_mode = os.environ.get("NEXUSSIM_EVAL_OMIT_UNUSED_RGB", "0").strip()
        if rgb_mode not in {"0", "1"}:
            raise ValueError("NEXUSSIM_EVAL_OMIT_UNUSED_RGB must be 0 or 1")
        self._omit_unused_policy_rgb = rgb_mode == "1"
        if self._omit_unused_policy_rgb:
            self._capture_cam_configs()  # Validate the actual adapter/configuration.
            logger.info("RGB requests omitted for the state-only policy; "
                        "NuRec setup and per-frame pose preparation remain enabled")

        # Trajectory tracker for controller/physics execution modes (created
        # from config if the caller didn't supply one). In teleport mode it
        # is constructed but never consulted. Supersedes the legacy
        # UnifiedController, which was instantiated here but never called.
        from navsafe.core.controllers import create_tracker

        frame_dt = live_execution_dt(self.env)
        require_configured_clock(frame_dt, config.sim_dt)
        self.controller = controller or create_tracker(
            config.controller_type, sim_dt=config.sim_dt,
        )
        if frame_dt is not None:
            synchronize_tracker_dt(self.controller, frame_dt)

        # EPDMS live scorer (EPDMSLiveScorer) — the ONLY metric
        # scorer, matching BridgeSim's base_evaluator. Initialised in setup()
        # once the scenario is loaded.
        self._epdms_scorer: Optional[Any] = None
        self._epdms_results: list = []

        # Artifact / visualisation output subsystem (saves frames, renders BEV /
        # front-cam, builds GIFs). Owns no run state — it
        # forwards reads back to this evaluator; the env keeps delegating wrappers.
        self._artifacts = EvalArtifactWriter(self)

        # Route subsystem (owns the route/full_route waypoint state). Built
        # before _reset_state(), which delegates route clearing to it.
        self._route = RouteManager(self)

        # Run-state attributes (populated by _reset_state below). Declared here
        # with their authoritative Optional types so they are well-typed
        # regardless of the order mypy visits the methods that assign them.
        self._done: bool = False
        self._current_trajectory: Optional[np.ndarray] = None
        # Display-only: the most recent warm-up plan, held between replans so
        # the vis overlay does not flash. Never read by scoring.
        self._warmup_overlay_plan: Optional[np.ndarray] = None
        self._cached_world_traj: Optional[np.ndarray] = None
        self._cached_stop_command: Optional[Dict[str, Any]] = None
        self._cached_plan_arc: Optional[np.ndarray] = None
        self._cached_plan_speed: float = 0.0
        self._plan_arc_s: float = 0.0
        self._cached_prediction_position: Optional[np.ndarray] = None
        self._cached_prediction_heading: Optional[float] = None
        self.scenario_path: Optional[Path] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self._prev_ego_velocity: Optional[np.ndarray] = None
        self._prev_ego_heading: Optional[float] = None
        self._warned_backward_plan: bool = False
        self._collision_active: bool = False
        self._warned_observe_failure: bool = False
        # Set by _step's exception handler; consumed by _termination_record /
        # finalize. None == no step-level crash this episode.
        self._step_crash: Optional[Dict[str, Any]] = None
        self._reactivity_trace: Optional[Any] = None

        self._reset_state()

    # ------------------------------------------------------------------
    # Route state — owned by the RouteManager subsystem, exposed read-only so
    # the artifact writer (reads ``full_route``) and any other reader resolve.
    # ------------------------------------------------------------------
    @property
    def route(self) -> Any:
        return self._route.route

    @property
    def full_route(self) -> Any:
        return self._route.full_route

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self, scenario_path: Path) -> None:
        """Load a scenario into the environment and prepare for evaluation.

        Args:
            scenario_path: Directory of a converted ScenarioNet scenario.
        """
        logger.info(f"Setting up scenario: {scenario_path}")

        _scenario_path = Path(scenario_path)

        # Ask the env to load this scenario (env handles USD / agent rebuild)
        self.env.load_scenario_from_path(_scenario_path)

        # Controller/physics execution modes drive the ego through actions:
        # the log-replay traffic must skip the ego track and the ego prim
        # follows the integrated ego state.
        if (self.config.execution_mode != "teleport"
                and hasattr(self.env, "set_ego_externally_driven")):
            self.env.set_ego_externally_driven(True)
        # Physics mode needs the env built with EnvCfg.execution_mode="physics"
        # (PhysX ego proxy). Fail loudly instead of silently degrading to the
        # bicycle model when the env was constructed kinematic (e.g. via a
        # path that does not plumb execution_mode, like a bare preset).
        if (self.config.execution_mode == "physics"
                and getattr(self.env, "_ego_phys_view", None) is None):
            raise ValueError(
                "EvaluationConfig.execution_mode='physics' but the env has no "
                "PhysX ego proxy — build it with EnvCfg.execution_mode="
                "'physics' (eval_py123d.py --execution-mode physics does this).")

        # Reset components
        frame_dt = live_execution_dt(self.env)
        require_configured_clock(frame_dt, self.config.sim_dt)
        if frame_dt is not None:
            synchronize_tracker_dt(self.controller, frame_dt)
        self.controller.reset()
        self._reset_state()

        # Set after _reset_state() so they aren't overwritten to defaults
        self.scenario_data = self.env.get_scenario_info()
        self.scenario_path = _scenario_path
        self.scenario_id = _scenario_path.name

        # Generate route waypoints from GT trajectory (BridgeSim-style)
        self.generate_route()

        # NavSafe termination, live. The deviation columns are measured against
        # the logged ego track, so this needs the scenario — hence here, not in
        # _reset_state. Without a logged ego the monitor still runs and the
        # deviation is nan (see LiveMonitor.deviation_m); no rule depends on it.
        self._termination = self._build_termination_monitor()
        if self.config.record_reactivity_trace:
            from navsafe.evaluation.reactivity_trace import ReactivityTraceRecorder
            self._reactivity_trace = ReactivityTraceRecorder(self)

        self.start_time = datetime.now()
        logger.info(f"Scenario ready: {self.scenario_id}  "
                    f"length={self.scenario_data.get('length', '?')} frames")

        # ── EPDMS live scorer ───────────────────────────────────────────
        # Initialise the full EPDMS scorer for lane-aware, live closed-loop
        # metrics.  Falls back gracefully if the scenario lacks map_features
        # or the env doesn't expose a MetaDrive-style agent API.
        try:
            from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
                EPDMSLiveScorer as FullEPDMSScorer,
            )

            # Create a lightweight proxy that bridges navsafe's env API
            # to the MetaDrive-style API expected by EPDMSLiveScorer
            class _EnvProxy:
                """Wraps ScenarioReplayEnv to provide MetaDrive-compatible API."""
                def __init__(self, env):
                    self._env = env
                    self.agent = self  # self acts as the agent proxy too
                    self.engine = self  # self acts as the engine proxy too
                    # Per-frame ego snapshot: score_frame_live reads
                    # position/heading/velocity back-to-back (4 property
                    # hits), each of which used to rebuild the ego-state
                    # dict. The evaluator calls begin_frame() right before
                    # each score_frame_live so the snapshot never spans an
                    # env step.
                    self._ego_state_cache = None

                def begin_frame(self):
                    """Invalidate the cached ego snapshot (call per frame)."""
                    self._ego_state_cache = None

                def _ego_state(self):
                    if self._ego_state_cache is None:
                        self._ego_state_cache = self._env.get_ego_state()
                    return self._ego_state_cache

                @property
                def position(self):
                    return self._ego_state()["position"][:2]

                @property
                def heading_theta(self):
                    return self._ego_state()["heading"]

                @property
                def velocity(self):
                    return self._ego_state()["velocity"]

                @property
                def name(self):
                    return "__ego__"

                def get_objects(self):
                    """MetaDrive-style object dict from the env's log-replay
                    agent states, so NC/TTC score against real traffic (the
                    former empty-dict stub made both vacuously 1.0)."""
                    class _Obj:
                        def __init__(self, s, fallback_name):
                            # Same fallback as the dict key below: an
                            # id-less agent must not alias with other
                            # id-less agents in the scorer's contact
                            # memory (which is keyed on .name).
                            self.name = str(s.get("id", fallback_name))
                            self.position = np.asarray(s["position"][:2], dtype=float)
                            self.heading_theta = float(s.get("heading", 0.0))
                            self.velocity = np.asarray(
                                s.get("velocity", np.zeros(2))[:2], dtype=float)
                            self.top_down_length = float(s.get("length", 4.5))
                            self.top_down_width = float(s.get("width", 1.8))
                    return {
                        str(s.get("id", i)): _Obj(s, i)
                        for i, s in enumerate(self._env.agent_states)
                        if not s.get("is_ego", False)
                    }

            env_proxy = _EnvProxy(self.env)
            self._epdms_scorer = FullEPDMSScorer(self.env.current_scenario, env_proxy)
            self._epdms_scorer.reset_live_state()
            self._epdms_results = []
            self._epdms_score_failures = 0
            logger.info("EPDMS live scorer initialised successfully")
        except Exception as e:
            # BridgeSim parity: EPDMS is the ONLY scorer — a silent fallback
            # would produce a differently-defined score. Fail loudly.
            raise RuntimeError(
                f"EPDMS scorer init failed ({e}) — cannot evaluate without "
                "the BridgeSim-parity scorer") from e

        # If the model adapter carries a trajectory scorer, wire it up
        if self.trajectory_scorer is None and hasattr(self.adapter, 'trajectory_scorer'):
            self.trajectory_scorer = self.adapter.trajectory_scorer

    def run(self) -> Dict[str, Any]:
        """Run the full evaluation loop for the current scenario.

        Returns:
            Result dict (same as returned by finalize()).
        """
        sys.stderr.write(f"[EVAL] Starting evaluation loop for {self.scenario_id}\n")
        sys.stderr.flush()
        step = 0
        while True:
            keep_going = self._step()
            if not keep_going:
                break
            step += 1
            if step % 10 == 0:
                sys.stderr.write(f"[EVAL] frame {self.frame}\n")
                sys.stderr.flush()

        self.end_time = datetime.now()
        results = self.finalize()
        sys.stderr.write(f"[EVAL] Done — {results['total_frames']} frames\n")
        # [PROFILING] per-stage wall-time breakdown of the eval loop.
        if hasattr(self, "_prof"):
            nf = max(int(results.get('total_frames', 0) or step or 1), 1)
            tot = sum(self._prof.values())
            sys.stderr.write(f"[PROFILE] eval loop stages over {nf} frames "
                             f"(sum={tot:.1f}s, {tot/nf*1000:.0f} ms/frame):\n")
            for k in sorted(self._prof, key=lambda x: -self._prof[x]):
                v = self._prof[k]
                sys.stderr.write(f"[PROFILE]   {k:14s} total={v:8.1f}s  "
                                 f"per_frame={v/nf*1000:7.1f}ms  ({v/tot*100:4.1f}%)\n")
            sys.stderr.flush()
        return results

    def finalize(self) -> Dict[str, Any]:
        """Aggregate metrics and persist result files.

        Returns:
            Dictionary with aggregated evaluation results.
        """
        # Legacy run stats (odometer, speed, collisions) computed inline —
        # all metric scoring below comes from the EPDMS live scorer.
        final_metrics: Dict[str, Any] = {}
        positions = [np.asarray(s.get("position", [0, 0, 0]))[:2]
                     for s in self._history["vehicle_states"]]
        odometer = (float(np.sum(np.linalg.norm(
            np.diff(np.asarray(positions), axis=0), axis=1)))
            if len(positions) > 1 else 0.0)
        speeds = [m.get("velocity", 0.0) for m in self._history["metrics"]]
        collision_count = sum(
            1 for m in self._history["metrics"] if m.get("collision"))
        final_metrics["route_completion"] = odometer
        final_metrics["avg_speed"] = (
            float(np.mean(speeds)) if speeds else 0.0)
        final_metrics["collision_count"] = collision_count
        final_metrics["success"] = bool(
            collision_count == 0 and odometer >= 10.0)
        final_metrics["total_frames"] = self.frame

        # Resolve the ending before route-completion aggregation.  The NavSafe
        # scoring contract defines GOAL_REACHED as complete even when the
        # terminal pose is within the goal tolerance rather than exactly on
        # the final route sample.  The artifact re-scorer already applies this
        # rule; applying it here keeps the live metrics.json scorer identical.
        termination = self._termination_record()

        # EPDMS live metrics (the scorer of record; matches BridgeSim)
        if self._epdms_results:
            import pandas as pd
            df = pd.DataFrame(self._epdms_results)
            df_valid = df[df['valid'] == True]
            if not df_valid.empty:
                # Use EPDMS metrics instead of PDMS
                final_metrics['no_at_fault_collisions'] = float(df_valid['no_at_fault_collisions'].mean())
                final_metrics['drivable_area_compliance'] = float(df_valid['drivable_area_compliance'].mean())
                final_metrics['driving_direction_compliance'] = float(df_valid['driving_direction_compliance'].mean())
                final_metrics['traffic_light_compliance'] = float(df_valid['traffic_light_compliance'].mean())
                final_metrics['time_to_collision_within_bound'] = float(df_valid['time_to_collision_within_bound'].mean())
                final_metrics['lane_keeping'] = float(df_valid['lane_keeping'].mean())
                final_metrics['history_comfort'] = float(df_valid['history_comfort'].mean())
                final_metrics['extended_comfort'] = float(df_valid.get('extended_comfort', pd.Series([1.0])).mean())

                # ── Compute mean EPDMS (no EP) — matches BridgeSim's formula ──
                # Per-frame score from score_frame_live already excludes EP:
                #   score = (NC × DAC × DDC × TLC) × (5×TTC + 2×LK + 2×HC + 2×EC) / 11
                mean_epdms_no_ep = float(df_valid['score'].mean())
                final_metrics['epdms_no_ep'] = mean_epdms_no_ep

                # ── Compute route completion as fraction of GT route ──────────
                # BridgeSim formula (base_evaluator.py finalization): RC =
                # min(1, actual_rc_delta / expected_rc_delta), where the delta
                # baseline is taken at the score-start frame (end of warm-up
                # replay — that progress is the log's, not the model's) and
                # expected_rc_delta is the GT distance over the SCORED window
                # [warmup, warmup + eval_frames) as a fraction of the log.
                # When the expected delta is 0 (or the run is full-log with no
                # warm-up), BridgeSim falls back to the raw delta — here the
                # progress as a fraction of the full GT route.
                scenario = getattr(self.env, 'current_scenario', None)
                # None == the computation THREW; the RC-family keys are then
                # NOT written here, so they flow through the defaulted/counted
                # route below (present as 0.0 AND listed in defaulted_metrics)
                # instead of being published as measured zeros. NB a missing
                # scenario or absent/short GT track still yields a "measured"
                # 0.0 — that path predates this fix and is unchanged here.
                rc_fraction: Optional[float] = 0.0
                if scenario is not None:
                    try:
                        from navsafe.scenario.scenario_description import ScenarioDescription as SD
                        metadata = scenario.get(SD.METADATA, {})
                        sdc_id = str(metadata.get(SD.SDC_ID, ''))
                        tracks = scenario.get(SD.TRACKS, {})
                        ego_track = tracks.get(sdc_id, {})
                        gt_positions = ego_track.get(SD.STATE, {}).get('position')
                        if gt_positions is not None and len(gt_positions) > 1:
                            gt_pos_arr = np.array(gt_positions)[:, :2]
                            seg = np.linalg.norm(np.diff(gt_pos_arr, axis=0), axis=1)
                            warmup = max(0, int(self.config.ego_replay_frames))
                            # Scored window: warmup → warmup + eval_frames
                            # (full log when eval_frames is None), capped at
                            # the log length like BridgeSim's num_frames.
                            end = (warmup + self.config.eval_frames
                                   if self.config.eval_frames is not None
                                   else len(seg))
                            gt_window_dist = float(np.sum(seg[warmup:end]))
                            gt_total_dist = float(np.sum(seg))
                            # Model's own progress (actual_rc_delta): project
                            # the ego position at the score-start baseline and
                            # at the final frame onto the GT route arc — the
                            # in-evaluator analogue of BridgeSim's
                            # info['route_completion'] delta (keeps the
                            # scoring path independent of PDMSScorer).
                            arc = np.concatenate([[0.0], np.cumsum(seg)])

                            ego_positions = [
                                np.asarray(s.get('position', [0, 0, 0]))[:2]
                                for s in self._history['vehicle_states']]
                            progress_dist = 0.0
                            if len(ego_positions) > warmup:
                                # Completion is monotone. A final-pose-only
                                # projection can move backwards after the ego
                                # passes a short route's endpoint, undercounting
                                # a route that was genuinely completed earlier.
                                attained_arc = _monotone_projected_route_arc(
                                    gt_pos_arr, ego_positions)
                                baseline = float(attained_arc[min(
                                    warmup, len(attained_arc) - 1)])
                                progress_dist = max(
                                    0.0, float(attained_arc[-1]) - baseline)
                            else:
                                attained_arc = np.zeros(
                                    len(ego_positions), dtype=np.float64)
                            if gt_window_dist > 0:
                                rc_fraction = min(1.0, progress_dist / gt_window_dist)
                            elif gt_total_dist > 0:
                                # BridgeSim fallback: raw RC delta (fraction
                                # of the full route), no window normalization.
                                rc_fraction = min(1.0, progress_dist / gt_total_dist)

                            if (termination.get("reason") == "goal_reached"
                                    and termination.get("scorable", True)):
                                rc_fraction = 1.0

                            # ── Per-frame EP + full EPDMS (with EP) ──────────
                            # EP_f = model progress since hand-off ÷ GT
                            # progress since hand-off, clipped to [0, 1]
                            # (1.0 while the log expects no progress yet).
                            # Full EPDMS applies the NavSim weighting
                            # (5·EP + 5·TTC + 2·LK + 2·HC + 2·EC)/16 on the
                            # same multiplicative gate as the no-EP score.
                            for r in self._epdms_results:
                                f = int(r.get('frame', 0))
                                if not r.get('valid') or f >= len(ego_positions):
                                    continue
                                gt_prog = float(
                                    arc[min(f, len(arc) - 1)] - arc[min(warmup, len(arc) - 1)])
                                if len(ego_positions) > warmup:
                                    model_prog = max(
                                        0.0, float(attained_arc[f]) - baseline)
                                else:
                                    model_prog = 0.0
                                ep = (1.0 if gt_prog <= 1e-3
                                      else float(np.clip(model_prog / gt_prog, 0.0, 1.0)))
                                r['ego_progress'] = ep
                                gate = (r.get('no_at_fault_collisions', 1.0)
                                        * r.get('drivable_area_compliance', 1.0)
                                        * r.get('driving_direction_compliance', 1.0)
                                        * r.get('traffic_light_compliance', 1.0))
                                weighted = (5.0 * ep
                                            + 5.0 * r.get('time_to_collision_within_bound', 1.0)
                                            + 2.0 * r.get('lane_keeping', 1.0)
                                            + 2.0 * r.get('history_comfort', 1.0)
                                            + 2.0 * r.get('extended_comfort', 1.0)) / 16.0
                                r['score_with_ep'] = gate * weighted
                            with_ep = [r['score_with_ep'] for r in self._epdms_results
                                       if 'score_with_ep' in r]
                            if with_ep:
                                final_metrics['epdms'] = float(np.mean(with_ep))
                    except Exception:
                        # The former `pass` here fabricated a measured-looking
                        # result: route_completion_fraction / driving_score /
                        # ego_progress were then published as 0.0 WITHOUT
                        # appearing in defaulted_metrics — indistinguishable
                        # from a car that demonstrably went nowhere. Route the
                        # failure through the counted path instead.
                        rc_fraction = None
                        logger.warning(
                            "[%s] EP/route-completion computation failed — "
                            "route_completion_fraction / driving_score / "
                            "ego_progress will be reported via "
                            "defaulted_metrics", self.scenario_id,
                            exc_info=True)
                if rc_fraction is not None:
                    final_metrics['route_completion_fraction'] = rc_fraction

                    # ── Final driving score = EPDMS_no_EP × RC ───────────────
                    final_score = mean_epdms_no_ep * rc_fraction
                    final_metrics['driving_score'] = final_score
                    # Drop PDMSScorer's composite: with EPDMS live metrics in
                    # play, a second differently-defined score is confusing.
                    final_metrics.pop('pdms_score', None)

                    # Also keep ego_progress as the RC fraction for consistency
                    final_metrics['ego_progress'] = rc_fraction

                    logger.info(
                        f"EPDMS final: DS={final_score:.4f} = "
                        f"EPDMS_no_EP({mean_epdms_no_ep:.4f}) × RC({rc_fraction:.4f})"
                    )

        # ── Episode reporting contract ───────────────────────────────────
        # Why stepping stopped, in the NavSafe taxonomy. envelope_exit and
        # infra_failure are the benchmark's endings, not the policy's: those
        # episodes are reported `—` and excluded from every denominator
        # rather than folded in as a near-zero score.
        # The declaration goes into metrics.json too, because that is the only
        # artifact aggregators read. Downstream must decide "exclude this
        # episode" from `scorable` — an explicit fact — and never from a
        # missing key, which is what an ordinary scoring failure also looks
        # like. Every scored key is therefore always present: when the EP/RC
        # block above throws, it now WITHHOLDS its keys (instead of the
        # historical fabricated `driving_score: 0.0` with `epdms` ABSENT) so
        # they arrive here, are defaulted to 0.0 and are listed in
        # `defaulted_metrics` — consumers requiring all keys numeric used to
        # drop such a scene instead of averaging its zero (per-scene
        # [0.9, 0.6, 0.3, 0.0] published 0.6, not 0.45).
        final_metrics["termination_reason"] = str(termination.get("reason", "unknown"))
        final_metrics["scorable"] = bool(termination.get("scorable", True))
        # A step-level crash carries its exception class+message into
        # metrics.json itself: aggregators read only this file, and a drop
        # record they cannot see is a drop record that does not exist.
        crash = getattr(self, "_step_crash", None)
        if crash is not None:
            final_metrics["infra_failure_error"] = (
                f"frame {crash['frame']}: {crash['error']}")
        unscored = [k for k in EPISODE_SCORE_KEYS if k not in final_metrics]
        for key in unscored:
            final_metrics[key] = 0.0
        if unscored:
            # Recorded, not just substituted: a consumer that must not average
            # an imputed value can still tell it from a measured one.
            final_metrics["defaulted_metrics"] = unscored
            logger.warning(
                "scenario %s produced no value for %s — reporting 0.0 "
                "(termination=%s, scorable=%s)",
                self.scenario_id, ", ".join(unscored),
                final_metrics["termination_reason"], final_metrics["scorable"])
        if not final_metrics["scorable"]:
            logger.warning(
                "scenario %s is UNSCORABLE (%s): the benchmark ended this "
                "episode, so it must be reported `—` and excluded from every "
                "denominator, not averaged in",
                self.scenario_id, final_metrics["termination_reason"])
        # Frame-level drop record for the live metric: consumers can tell an
        # episode scored on every frame from one that silently lost frames.
        final_metrics["epdms_unscored_frames"] = int(
            getattr(self, "_epdms_score_failures", 0))
        if final_metrics["epdms_unscored_frames"]:
            logger.warning(
                "scenario %s: %d frame(s) failed EPDMS live scoring and are "
                "missing from the episode metric",
                self.scenario_id, final_metrics["epdms_unscored_frames"])

        results = {
            "scenario_id": self.scenario_id,
            "total_frames": self.frame,
            "elapsed_time": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time is not None and self.start_time is not None
                else 0.0
            ),
            "metrics": final_metrics,
            "termination": termination,
            "trajectory_history": (
                self._safe_stack(self._history["trajectories"])
            ),
            "per_frame_metrics": self._history["metrics"],
            "config": {
                "traffic_mode": self.config.traffic_mode,
                "eval_mode": self.config.eval_mode,
                "controller_type": self.config.controller_type,
            },
        }

        self._save_results(results)
        if self._reactivity_trace is not None:
            trace_path = self._reactivity_trace.finalize(
                results=results, termination=termination,
                step_crash=self._step_crash)
            logger.info(
                "Reactivity primitives saved in one write -> %s "
                "(%d bytes, sha256=%s)",
                trace_path, self._reactivity_trace.bundle_size_bytes,
                self._reactivity_trace.bundle_sha256)
        # Trace + rubric verdict, alongside the EPDMS artifacts.
        if self._navsafe_trace is not None:
            self._navsafe_trace.close()

        # Generate GIFs / combined visualization if vis was enabled
        if self.config.enable_vis:
            self._generate_visualizations()

        return results

    # ------------------------------------------------------------------
    # Internal step logic
    # ------------------------------------------------------------------

    def _prof_add(self, key: str, t0: float) -> None:
        """[PROFILING] accumulate wall-time of a per-frame stage."""
        if not hasattr(self, "_prof"):
            self._prof: _defaultdict[str, float] = _defaultdict(float)
        self._prof[key] += _time.perf_counter() - t0

    @staticmethod
    def _scored_world_timestep(frame: int, info: Any) -> int:
        """Scenario timestep of the world the live scorer reads (step 9a).

        ``env.step()`` has already advanced the sim when the live scorer
        runs, so the ego/agent poses it reads sit at the POST-step
        timestep. Prefer the env's own counter
        (``info['scenario_timestep']``) over the ``frame + 1`` fallback —
        it is the env's own account of where the replay world sits.
        (Known limit: on a loop_replay wrap frame the env re-poses agents
        BEFORE wrapping the counter to 0, so info disagrees with the
        posed agents for that one frame; the evaluation path never runs
        loop_replay.)

        2026-08-18 TL alignment fix: passing the pre-step ``frame`` made
        ``EPDMSLiveScorer._check_tlc_live`` read the logged light state
        one frame (0.1 s) behind the world being scored, charging /
        clearing every traffic-light transition late.
        """
        if isinstance(info, dict):
            ts = info.get("scenario_timestep")
            if ts is not None:
                return int(ts)
        return frame + 1

    def _step(self) -> bool:
        """Execute one frame of the evaluation loop.

        Trajectory handling mirrors BridgeSim's process_frame():
          - Model predicts ego-frame waypoints at model_dt intervals (e.g. 0.5s)
          - Trajectory is interpolated to sim_dt intervals (e.g. 0.1s)
          - Interpolated trajectory is transformed to world coords and cached
          - Each frame consumes the next waypoint via replan_offset
          - After replay, ego is teleported to the planned waypoint position

        Returns:
            True if evaluation should continue, False if done.
        """
        if self._done:
            return False
        # BridgeSim window semantics: ``eval_frames`` counts SCORED frames —
        # the run lasts ego_replay_frames (warm-up) + eval_frames sim frames.
        if (self.config.eval_frames is not None
                and self.frame >= self.config.ego_replay_frames + self.config.eval_frames):
            return False
        # With no frame window, an optional route clock may still bound the
        # episode. ``route_time_limit_s=None`` is genuinely clock-free: only a
        # semantic taxonomy event ends it.
        route_limit = getattr(self.config, "route_time_limit_s", SAFETY_CEILING_S)
        if (self.config.eval_frames is None and route_limit is not None
                and math.isfinite(route_limit)):
            ceiling = self.config.ego_replay_frames + int(
                round(route_limit / max(self.config.sim_dt, 1e-6)))
            if self.frame >= ceiling:
                sys.stderr.write(
                    f"[EVAL] route time limit reached: {route_limit:.0f} s "
                    f"of scored driving ({ceiling} frames)\n")
                return False

        try:
            self._reactivity_control_context = {
                "execution_mode": self.config.execution_mode,
                "source": "not_set",
            }
            # ── 0. Hand-off perturbation ────────────────────────────────
            # BEFORE anything reads the ego this frame: the displaced pose has
            # to be what the cameras render from, what the adapter is asked to
            # plan from, and what the scorer measures. Applying it later would
            # plan from one pose and drive from another.
            if self.frame == self.config.ego_replay_frames:
                if self.config.ego_perturb_history == "controller":
                    self._record_controller_perturbation()
                else:
                    self._apply_handoff_perturbation()

            # ── 0b. Adapter perception hook (spec §3 step 1) ────────────
            # Called every frame so rule-based adapters (PDM-Closed) can
            # cache self._env / lane geometry before prepare_input runs.
            self._invoke_adapter_perceive(self.frame)

            # ── 1. Ego state from UrbanSim ──────────────────────────────
            ego_state = self.env.get_ego_state()

            # ── 1a. Enrich ego_state to the BridgeSim adapter contract ──
            # Spec §4.1 requires ``acceleration`` and ``angular_velocity``;
            # PDM-Closed's _score_extended_comfort reads them and would
            # KeyError at the first replan frame without this step.
            ego_state = self._enrich_ego_state(ego_state)

            # ── 1b. Get next route waypoint + command ───────────────────
            _tw = _time.perf_counter()
            wp_result = self.get_next_waypoint(
                np.asarray(ego_state["position"])[:2]
            )
            self._prof_add('waypoint', _tw)
            if wp_result is not None:
                self._current_waypoint = wp_result  # (pos, cmd, frame_idx)
                self._current_command = wp_result[1]
            else:
                self._current_waypoint = None
                self._current_command = -1  # VOID

            # Attach waypoint + command to ego_state so lane-biasing
            # adapters (see spec §4.1) can read them.
            ego_state["waypoint"] = (
                np.asarray(self._current_waypoint[0], dtype=np.float32)
                if self._current_waypoint is not None else None
            )
            ego_state["command"] = int(self._current_command)

            # ── 2. Camera images from UrbanSim ─────────────────────────
            cam_configs = self._capture_cam_configs()
            _tr = _time.perf_counter()
            images = self._capture_observer_images(cam_configs)
            self._prof_add('render_obs', _tr)

            # ── 3. Open-loop warm-up: replay ground-truth ego actions ───
            if self.frame < self.config.ego_replay_frames:
                if self.config.ego_perturb_history == "controller":
                    action_np = self._execute_augmented_warmup(ego_state)
                    self._reactivity_control_context = {
                        "execution_mode": "controller",
                        "source": "augmented_warmup",
                    }
                else:
                    gt_action = self.env.get_gt_ego_action(self.frame)
                    action_np = np.asarray(gt_action, dtype=np.float32)
                    self._reactivity_control_context = {
                        "execution_mode": "replay",
                        "source": "logged_ego_action",
                    }

                self._record_replay_pose(ego_state)

                # Still run inference during replay for warmup (matches BridgeSim)
                replan_offset = self.frame % self.config.replan_rate
                needs_replan = (replan_offset == 0)
                warmup_ok = getattr(
                    self.adapter, "supports_warmup_inference", lambda: True)()
                if needs_replan and warmup_ok:
                    self._run_inference_and_cache(ego_state, images)
                    # The plan the policy just produced is what a viewer wants
                    # to see, but _record_replay_pose above has already
                    # overwritten _current_trajectory with the observed pose
                    # (and must keep doing so -- the replay prefix is scored
                    # against the logged motion, not against a prediction).
                    # Keep a display-only copy so the overlay can hold this
                    # plan until the next replan instead of flashing on one
                    # frame in every replan_rate.
                    self._warmup_overlay_plan = (
                        None if self._cached_world_traj is None
                        else np.asarray(self._cached_world_traj,
                                        dtype=np.float32).copy())

            else:
                # Policy now owns control: _current_trajectory is the real plan
                # again, so drop the display-only hold from the replay prefix.
                self._warmup_overlay_plan = None
                # ── 4. Determine replan offset & whether to replan ───────
                if self.frame == self.config.ego_replay_frames:
                    # Transition frame: force replan to restart cycle
                    replan_offset = 0
                    needs_replan = True
                else:
                    frames_since_replay_end = self.frame - self.config.ego_replay_frames
                    replan_offset = frames_since_replay_end % self.config.replan_rate
                    needs_replan = (replan_offset == 0)

                # ── 5. Run inference when it's time to replan ────────────
                if needs_replan or self._cached_world_traj is None:
                    self._run_inference_and_cache(ego_state, images)

                # _run_inference_and_cache always populates the cache; after the
                # block above it is guaranteed non-None.
                assert self._cached_world_traj is not None

                if self.config.execution_mode == "teleport":
                    action_np, world_traj_subset = self._execute_teleport_step(
                        ego_state, images, needs_replan, replan_offset)
                else:
                    action_np, world_traj_subset = self._execute_controller_step(
                        ego_state, replan_offset)

                self._record_current_trajectory(world_traj_subset, ego_state)

            # ── 8. Step UrbanSim env ────────────────────────────────────
            action_tensor = torch.tensor(
                action_np[np.newaxis, :],
                dtype=torch.float32,
                device=getattr(self.env, "device", "cpu"),
            )
            _ts = _time.perf_counter()
            obs, rewards, terminated, truncated, info = self.env.step(action_tensor)
            self._prof_add('env_step', _ts)

            episode_done = bool((terminated | truncated).any())

            # ── 9. Per-frame metrics ────────────────────────────────────
            next_ego = self.env.get_ego_state()
            collision = info.get("collision", False) if isinstance(info, dict) else False
            # Count contact *events*, not overlapped frames: since only
            # at-fault collisions terminate, a not-at-fault rear-end can keep
            # the polygons overlapping for several frames — score it once.
            collision_event = bool(collision) and not self._collision_active
            self._collision_active = bool(collision)
            # By this point the planned trajectory has always been cached (via
            # _run_inference_and_cache or the teleport block above).
            assert self._current_trajectory is not None
            # Per-frame telemetry (velocity/collision); all metric scoring is
            # the EPDMS live scorer below — BridgeSim parity.
            vel = np.asarray(ego_state.get("velocity", [0.0, 0.0]))[:2]
            # Fault is recorded per frame, not just the contact: a replayed
            # follower rear-ending a correct ego is not the policy's
            # infraction, and that distinction is unrecoverable after the run
            # unless it is written down here (the EPDMS NC column only ever
            # reports the at-fault case).
            at_fault = (info.get("collision_at_fault", False)
                        if isinstance(info, dict) else False)
            frame_metrics: Dict[str, Any] = {
                "velocity": float(ego_state.get("speed", np.linalg.norm(vel))),
                "collision": bool(collision_event),
                "collision_at_fault": bool(collision_event and at_fault),
            }
            if collision_event:
                detail = (info.get("contact_detail") or {}
                          if isinstance(info, dict) else {})
                frame_metrics["contact_detail"] = detail
                print(f"[EVAL] frame {self.frame}: CONTACT "
                      f"{detail.get('kind', '?')} with {detail.get('agent_id', '?')} "
                      f"({detail.get('agent_type', '?')}) "
                      f"long={detail.get('longitudinal_m', float('nan')):.2f} m "
                      f"lat={detail.get('lateral_m', float('nan')):.2f} m "
                      f"ego_speed={detail.get('ego_speed', float('nan')):.2f} m/s "
                      f"at_fault={bool(at_fault)}", flush=True)

            # ── 9a. EPDMS live scoring ──────────────────────────────────
            # env.step() above already advanced the sim, so the ego pose /
            # agents the scorer reads are the POST-step world (scenario
            # timestep self.frame + 1), while ``self.frame`` (incremented
            # below) still names the pre-step frame. score_frame_live
            # therefore receives the post-step timestep
            # (_scored_world_timestep — 2026-08-18 TL alignment fix; the
            # logged-light lookup used to run one frame behind the world
            # being scored). The ``frame`` stamp added below keeps the
            # evaluator's own frame numbering, unchanged.
            if (self._epdms_scorer is not None
                    and self.frame < self.config.ego_replay_frames):
                # Warm-up (open-loop replay) frames are not scored, but
                # their kinematics are real history: feed the scorer's
                # comfort chain so the FIRST scored frame measures the
                # replay→policy handoff instead of force-passing hc/ec on
                # an empty history (2026-08-18 warm-up seeding fix). Same
                # post-step call point as scoring, so the seeded samples
                # are spaced exactly one frame apart on the scored cadence.
                try:
                    begin_frame = getattr(self._epdms_scorer.env,
                                          "begin_frame", None)
                    if begin_frame is not None:
                        begin_frame()
                    self._epdms_scorer.observe_frame_kinematics()
                except Exception:
                    # A skipped observation leaves a 2·dt gap in the
                    # velocity chain that the next observe would divide
                    # by one dt (inflated accel, then inflated jerk —
                    # poisoning exactly the handoff measurement this
                    # seeding exists to make accurate). Clear the history
                    # so it re-seeds cleanly from the remaining warm-up
                    # frames; during warm-up nothing has scored yet, so
                    # reset_live_state touches only still-empty state
                    # besides the comfort chain and the seeded previous
                    # centre (re-seeded by the next observation). Warn once.
                    self._epdms_scorer.reset_live_state()
                    if not self._warned_observe_failure:
                        self._warned_observe_failure = True
                        logger.warning(
                            "[%s] EPDMS warm-up kinematics observation "
                            "failed at frame %d — comfort history cleared; "
                            "it will re-seed from the remaining warm-up "
                            "frames (first scored frame falls back to the "
                            "legacy vacuous pass if none remain)",
                            self.scenario_id, self.frame, exc_info=True)
            if self._epdms_scorer is not None and self.frame >= self.config.ego_replay_frames:
                try:
                    _tsc = _time.perf_counter()
                    # Invalidate the proxy's per-frame ego snapshot before
                    # the scorer reads it (see _EnvProxy.begin_frame).
                    begin_frame = getattr(self._epdms_scorer.env,
                                          "begin_frame", None)
                    if begin_frame is not None:
                        begin_frame()
                    epdms_metrics = self._epdms_scorer.score_frame_live(
                        self._scored_world_timestep(self.frame, info))
                    self._prof_add('epdms_score', _tsc)
                    epdms_metrics['frame'] = self.frame
                    self._epdms_results.append(epdms_metrics)
                except Exception:
                    # A frame the metric of record cannot score is a dropped
                    # sample, not a debug detail: warn (with traceback) on the
                    # first one, count the rest into metrics.json.
                    self._epdms_score_failures += 1
                    if self._epdms_score_failures == 1:
                        logger.warning(
                            "[%s] EPDMS live scoring failed at frame %d — "
                            "frame dropped from the episode metric; further "
                            "failures this episode are counted into "
                            "metrics.json epdms_unscored_frames",
                            self.scenario_id, self.frame, exc_info=True)
                    else:
                        logger.debug(
                            "[%s] EPDMS live scoring failed at frame %d "
                            "(%d dropped so far)", self.scenario_id,
                            self.frame, self._epdms_score_failures)

            # ── 9b. Visualization ───────────────────────────────────────
            if self.config.enable_vis:
                _tv = _time.perf_counter()
                self._render_frame_vis(
                    ego_state=ego_state,
                    images=images,
                    collision=collision,
                )
                self._prof_add('render_vis', _tv)

            # ── 10. Record history ──────────────────────────────────────
            self._history["trajectories"].append(self._current_trajectory)
            self._history["vehicle_states"].append(next_ego)
            # Optional per-frame actor record, alongside the ego's own row so
            # the two are frame-aligned. The scenario's `tracks` cannot stand
            # in for it: a recipe-inserted actor is written there once, as its
            # SPAWN pose repeated for every frame, and its real path is
            # produced at run time by the traffic manager -- so where an
            # inserted hazard actually was is recorded here or lost.
            #
            # This must NOT live in the inference path: that is gated by
            # `--replan-rate`, which sampled 39 of 190 frames and would let a
            # minimum-clearance or minimum-TTC reading miss the closest frame
            # entirely. Off by default; only the controlled experiments read it.
            if not hasattr(self, "_dump_agent_states"):
                self._dump_agent_states = os.environ.get(
                    "NAVSAFE_DUMP_AGENT_STATES", "").strip() in ("1", "true", "yes")
                self._agent_state_log = []
            if self._dump_agent_states:
                self._agent_state_log.append((int(self.frame), [
                    {"id": str(st.get("id", "")),
                     "x": float(np.asarray(st.get("position", [np.nan] * 2), float).reshape(-1)[0]),
                     "y": float(np.asarray(st.get("position", [np.nan] * 2), float).reshape(-1)[1]),
                     "heading": float(st.get("heading") or np.nan),
                     "vx": float(np.asarray(st.get("velocity", [0.0, 0.0]), float).reshape(-1)[0]),
                     "vy": float(np.asarray(st.get("velocity", [0.0, 0.0]), float).reshape(-1)[1]),
                     "length": float(st.get("length") or np.nan),
                     "width": float(st.get("width") or np.nan)}
                    for st in (self.env.agent_states or [])]))
            self._history["metrics"].append(frame_metrics)
            self._history["actions"].append(action_np)
            self._history["timestamps"].append(self.frame * self.config.sim_dt)

            # A red light ahead on the ego's lane (the scorer's ``signal_hold``
            # state fact): a standstill there is the signal's, not a deadlock.
            # Absent -> False, so a scorer that does not publish it changes
            # nothing. Read once here for the live monitor AND the trace.
            signal_hold = False
            # The scorer's traffic-light verdict for this frame, as a tri-state:
            # None when it published none, so the hold rules know the column is
            # absent and fall back to the stop-line geometry. Passing False for
            # "no verdict" would have geometry overruled by a compliance the
            # scorer never asserted.
            signal_violation = None
            if self._epdms_results and self._epdms_results[-1].get("frame") == self.frame:
                signal_hold = bool(
                    float(self._epdms_results[-1].get("signal_hold", 0.0) or 0.0) > 0.0)
                tlc = self._epdms_results[-1].get("traffic_light_compliance")
                if tlc is not None:
                    signal_violation = float(tlc) < 1.0

            # ── 10a1. NavSafe termination, live ─────────────────────────
            # The taxonomy is a pure function of the trace, so it can run one
            # frame at a time. Run it here, after the metrics that feed it:
            # off-drivable and driving direction come from the EPDMS frame just
            # scored, contacts from the env, the pose from the post-step ego.
            if self._termination is not None:
                on_drivable = True
                # DDC == 0 is the full wrong-way violation (>6 m against the
                # local traffic direction) and ends the episode as the ego's
                # fault. Absent or None means the scorer had nothing to judge
                # -- no lane graph, or a frame it did not score -- so the ego is
                # given the benefit of it rather than terminated on a gap.
                direction_ok = True
                if self._epdms_results and self._epdms_results[-1].get("frame") == self.frame:
                    dac = self._epdms_results[-1].get("drivable_area_compliance")
                    if dac is not None:
                        on_drivable = bool(float(dac) > 0.0)
                    ddc = self._epdms_results[-1].get("driving_direction_compliance")
                    if ddc is not None:
                        direction_ok = bool(float(ddc) > 0.0)
                contacts = []
                if collision_event:
                    contacts.append({
                        "at_fault": bool(at_fault),
                        "kind": frame_metrics.get("contact_detail", {}).get("kind"),
                        "agent_id": frame_metrics.get("contact_detail", {}).get("agent_id"),
                    })
                # GOAL_REACHED is only reachable if something tells the monitor
                # the route is finished. Without it an episode that completed
                # its route kept driving past the end and was charged for
                # whatever it hit out there, so the run reported a collision
                # rather than a completion.
                ego_xy_now = np.asarray(next_ego.get("position", [0.0, 0.0]))[:2]
                ended = self._termination.update(
                    ego_xy=ego_xy_now,
                    ego_speed=float(next_ego.get("speed", 0.0)),
                    on_drivable=on_drivable, contacts=contacts,
                    driving_direction_ok=direction_ok,
                    goal_reached=self._route.goal_reached(ego_xy_now),
                    signal_hold=signal_hold,
                    signal_violation=signal_violation,
                    dist_to_stopline_m=self._dist_to_stopline(ego_xy_now))
                frame_metrics["ego_dev_m"] = self._termination.deviation_m(
                    np.asarray(next_ego.get("position", [0.0, 0.0]))[:2])
                if ended is not None:
                    print(f"[EVAL] frame {self.frame}: TERMINATION "
                          f"{ended.reason.value} — {ended.detail}", flush=True)
                    self._terminated_reason = ended
                    episode_done = True

            # ── 10a. NavSafe canonical trace (opt-in: NAVSAFE_TRACE) ────
            # Captured here because everything the rubric needs is in scope:
            # the post-step ego, the env's agent poses, and the contact flag.
            if self._navsafe_trace is not None:
                self._navsafe_trace.on_step(
                    env=self.env, frame=self.frame,
                    ego_state=next_ego, collision=bool(collision),
                    signal_hold=signal_hold)

            if self._reactivity_trace is not None:
                self._reactivity_trace.record_step(
                    frame=self.frame, ego_before=ego_state,
                    ego_after=next_ego, action=action_np,
                    control=self._reactivity_control_context,
                    info=info, frame_metrics=frame_metrics,
                    signal_hold=signal_hold,
                    signal_violation=signal_violation,
                    episode_done=episode_done)

            self.frame += 1
            self._done = episode_done
            return not episode_done

        except Exception as e:
            # A step-level crash must not end the episode as if it completed:
            # record the fact so finalize() classifies it infra_failure /
            # scorable=false (the NavSafe taxonomy's "excluded from every
            # denominator" ending) instead of writing a healthy-looking
            # truncated metrics.json that downstream means count as a run.
            self._step_crash = {
                "frame": int(self.frame),
                "error": f"{type(e).__name__}: {e}",
            }
            logger.error(f"[{self.scenario_id}] step {self.frame} error: {e}", exc_info=True)
            return False

    def _capture_cam_configs(self) -> Dict[str, Dict[str, float]]:
        """Cameras to render this frame: the adapter's, plus vis-only extras.

        The adapter stays authoritative — extras are appended, never allowed to
        replace a camera the policy asked for, so a policy that already renders
        ``CAM_B0`` keeps its own config (resolution/fov may differ from the
        NAVSIM default). Adapters read ``images`` by key, so the surplus entry
        is invisible to inference and only reaches the artifact writer.
        """
        if getattr(self, "_omit_unused_policy_rgb", False):
            from navsafe.policy.state.pdm_closed import PDMClosedAdapter

            eligible = type(self.adapter) is PDMClosedAdapter
            env_cfg = getattr(self.env, "cfg", None)
            renderer = getattr(self.env, "_renderer", None)
            if (not eligible
                    or getattr(env_cfg, "render_backend", None) != "nurec_grpc"
                    or self.config.enable_vis or self.config.vis_online
                    or self.config.vis_extra_cameras
                    or self.observer_agent_id not in (
                        None, getattr(self.env, "EGO_AGENT_ID", "__ego__"))
                    or getattr(env_cfg, "nurec_camera_names", None) is not None
                    or getattr(renderer, "_render_steps", None) is not None
                    or os.environ.get("NUREC_GRPC_RENDER_STEPS", "").strip()):
                raise ValueError(
                    "NEXUSSIM_EVAL_OMIT_UNUSED_RGB requires exact PDM-Closed, "
                    "nurec_grpc, and no "
                    "visualization, camera/observer overrides, or render-step filter")
            # Still call env.get_camera_images({}) below: NuRec's handoff,
            # centre-to-rig transform and dynamic-object bookkeeping precede
            # its camera loop. Only RGB RPCs/decoding are omitted.
            return {}
        cam_configs = dict(self.adapter.get_camera_configs())
        for name in self.config.vis_extra_cameras:
            cam_configs.setdefault(name, NAVSIM_CAM_CONFIGS[name])
        return cam_configs

    def _capture_observer_images(self, cam_configs: Dict) -> Dict[str, np.ndarray]:
        """Fetch the adapter's required cameras from the configured observer.

        When :attr:`observer_agent_id` is ``None`` or the ego sentinel, this
        is exactly ``env.get_camera_images(cam_configs)`` (the historic
        path — byte-identical output).  When a non-ego observer is set,
        the env's per-agent capture API is used and the result is
        unwrapped back into the ``{cam_name: image}`` shape the adapter
        expects.  Missing cameras fall back to zero arrays at the
        requested resolution so downstream shape invariants hold.
        """
        env_ego_id = getattr(self.env, "EGO_AGENT_ID", "__ego__")
        observer = self.observer_agent_id
        if observer is None or observer == env_ego_id:
            return self.env.get_camera_images(cam_configs)

        if not hasattr(self.env, "get_agent_camera_images"):
            logger.warning(
                "[Evaluator] observer_agent_id=%s requested but env lacks "
                "get_agent_camera_images; falling back to ego capture",
                observer,
            )
            return self.env.get_camera_images(cam_configs)

        multi = self.env.get_agent_camera_images({observer: cam_configs})
        images: Dict[str, np.ndarray] = {}
        for cam_name, cfg in cam_configs.items():
            img = multi.get((observer, cam_name))
            if img is None:
                h = int(cfg.get("height", 900))
                w = int(cfg.get("width", 1600))
                img = np.zeros((h, w, 3), dtype=np.uint8)
            images[cam_name] = img
        return images

    def _invoke_adapter_perceive(self, frame_id: int) -> None:
        """Call ``adapter.perceive(env, frame_id)`` if the adapter defines one.

        BridgeSim's PDM-Closed uses this hook to cache ``self._env`` and
        lane geometry on the first call.  The adapter base class's default
        ``perceive`` is a no-op returning ``None``, so calling it is safe
        for adapters that don't need it.
        """
        perceive = getattr(self.adapter, "perceive", None)
        if perceive is None:
            return
        try:
            perceive(self.env, int(frame_id))
        except Exception as e:
            logger.warning(
                "[%s] adapter.perceive(frame=%d) raised %s — continuing",
                getattr(self, "scenario_id", "?"), int(frame_id), e,
            )

    def _enrich_ego_state(self, ego_state: Dict[str, Any]) -> Dict[str, Any]:
        """Add spec §4.1 keys ``acceleration`` and ``angular_velocity``.

        Both are finite-differenced from the previous frame's ``velocity``
        and ``heading`` at ``self.config.sim_dt``.  On frame 0 (or when no
        prior state has been cached), returns the BridgeSim harness
        defaults: ``acceleration = [0, 0, 9.8]`` and
        ``angular_velocity = zeros(3)``.
        """
        dt = float(self.config.sim_dt)
        curr_vel = np.asarray(ego_state.get("velocity", np.zeros(3)), dtype=np.float32)
        if curr_vel.size < 3:
            pad: np.ndarray = np.zeros(3, dtype=np.float32)
            pad[: curr_vel.size] = curr_vel
            curr_vel = pad
        curr_heading = float(ego_state.get("heading", 0.0))

        if self._prev_ego_velocity is None or dt <= 0.0:
            acceleration: np.ndarray = np.array(
                [0.0, 0.0, 9.8], dtype=np.float32)
            angular_velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        else:
            acceleration = ((curr_vel - self._prev_ego_velocity) / dt).astype(np.float32)
            # Wrap heading diff into [-π, π] so a flip across ±π doesn't
            # register as a huge spike.
            dtheta = curr_heading - (self._prev_ego_heading or 0.0)
            dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
            angular_velocity = np.array([0.0, 0.0, dtheta / dt], dtype=np.float32)

        self._prev_ego_velocity = curr_vel.copy()
        self._prev_ego_heading = curr_heading

        enriched = dict(ego_state)
        enriched.setdefault("acceleration", acceleration)
        enriched.setdefault("angular_velocity", angular_velocity)
        # Callers may want to overwrite these if the env ever starts
        # computing them natively — setdefault preserves that.
        enriched["acceleration"] = enriched.get("acceleration", acceleration)
        enriched["angular_velocity"] = enriched.get("angular_velocity", angular_velocity)
        return enriched


    def _execute_augmented_warmup(self, ego_state):
        from navsafe.evaluation.perturbation import augmented_warmup_path
        if self.config.execution_mode != "controller":
            raise ValueError("controller perturbation history requires controller execution")
        sd = self.env.get_scenario_info()
        state = sd["tracks"][str(sd["metadata"]["sdc_id"])]["state"]
        path = augmented_warmup_path(
            state["position"], state["heading"], self.config.ego_replay_frames,
            self.config.ego_perturb_lateral_m,
            self.config.ego_perturb_longitudinal_m, self.config.ego_perturb_yaw_deg)
        self.env.clear_ego_override()
        self.env.set_ego_externally_driven(True)
        # Sample the warmup's current reference time. The normal planner's
        # one-second speed lookahead can skip an entire 0.8-second warmup,
        # erasing its longitudinal perturbation before it is ever executed.
        i = min(self.frame, len(path)-2)
        tangent = path[i+1]-path[i]
        distance = float(np.linalg.norm(tangent))
        direction = (tangent/distance if distance > 1e-6 else
                     np.array([math.cos(float(ego_state["heading"])),
                               math.sin(float(ego_state["heading"]))]))
        longitudinal_error = float(
            (path[i]-np.asarray(ego_state["position"])[:2]) @ direction)
        target_speed = max(0.0, distance/self.config.sim_dt + longitudinal_error)
        steer, accel = self.controller.compute(
            ego_state, path[max(0, self.frame-1):], target_speed)
        return np.asarray([steer, accel, 0.0], dtype=np.float32)

    def _record_controller_perturbation(self):
        if self.config.ego_replay_frames <= 0:
            raise ValueError("controller perturbation history requires warmup frames")
        sd = self.env.get_scenario_info()
        state = sd["tracks"][str(sd["metadata"]["sdc_id"])]["state"]
        k = self.config.ego_replay_frames
        actual = self.env.get_ego_state()
        h = float(state["heading"][k])
        delta = np.asarray(actual["position"])[:2]-np.asarray(state["position"])[k,:2]
        yaw = float(actual["heading"])-h
        self._perturb_record = {
            "mode": "controller_history", "frame": int(self.frame),
            "requested_lateral_m": self.config.ego_perturb_lateral_m,
            "requested_longitudinal_m": self.config.ego_perturb_longitudinal_m,
            "requested_yaw_deg": self.config.ego_perturb_yaw_deg,
            "achieved_lateral_m": float(-math.sin(h)*delta[0]+math.cos(h)*delta[1]),
            "achieved_longitudinal_m": float(math.cos(h)*delta[0]+math.sin(h)*delta[1]),
            "achieved_yaw_deg": math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))),
            "position_after": np.asarray(actual["position"]).tolist(),
            "velocity_after": np.asarray(actual["velocity"]).tolist(),
        }

    def _apply_handoff_perturbation(self) -> None:
        """Rigidly displace the ego on the hand-off frame, once.

        The question this answers: how sensitive is a policy to WHERE it is
        handed the car? The replay prefix is byte-identical across a sweep, so
        an arm that differs only in this displacement isolates the pose the
        policy first sees from everything else about the episode.

        The displacement is expressed in the ego's own frame at the hand-off
        pose -- +lateral to its LEFT, +longitudinal FORWARD, +yaw
        counter-clockwise -- so ``lateral=1.0`` means the same thing on a
        north-bound and a west-bound scenario.

        Applied through ``set_ego_override``, which syncs the pure-Python ego
        immediately, so this frame's render, plan and score all see the moved
        car. The controller execution path clears the override a few lines
        later; by then the ego state it integrates from is already the
        displaced one, which is the point -- the perturbation is a change of
        initial condition, not a force held over the episode.

        Velocity is ROTATED with the body rather than kept in the world frame.
        Left as it was, a yawed ego would be handed to the policy already
        sideslipping, and the tracker's first correction would be fighting a
        skid the perturbation invented.
        """
        dlat = float(getattr(self.config, "ego_perturb_lateral_m", 0.0) or 0.0)
        dlon = float(getattr(self.config, "ego_perturb_longitudinal_m", 0.0) or 0.0)
        dyaw_deg = float(getattr(self.config, "ego_perturb_yaw_deg", 0.0) or 0.0)
        if self._perturb_record is not None or (dlat == 0.0 and dlon == 0.0
                                                and dyaw_deg == 0.0):
            return
        if not hasattr(self.env, "set_ego_override"):
            logger.warning("[perturb] env has no set_ego_override; "
                           "hand-off perturbation NOT applied")
            return

        before = self.env.get_ego_state()
        pos = np.asarray(before["position"], dtype=np.float64).copy()
        if pos.size == 2:
            pos = np.array([pos[0], pos[1], 0.0], dtype=np.float64)
        heading = float(before["heading"])
        vel = np.asarray(before.get("velocity", [0.0, 0.0, 0.0]),
                         dtype=np.float64).reshape(-1)
        vel = np.pad(vel, (0, max(0, 3 - vel.size)))[:3]

        cos_h, sin_h = math.cos(heading), math.sin(heading)
        pos[0] += cos_h * dlon - sin_h * dlat
        pos[1] += sin_h * dlon + cos_h * dlat
        dyaw = math.radians(dyaw_deg)
        cos_y, sin_y = math.cos(dyaw), math.sin(dyaw)
        vx, vy = float(vel[0]), float(vel[1])
        vel[0] = cos_y * vx - sin_y * vy
        vel[1] = sin_y * vx + cos_y * vy

        self.env.set_ego_override(position=pos, heading=heading + dyaw,
                                  velocity=vel)
        self._perturb_record = {
            "frame": int(self.frame),
            "lateral_m": dlat,
            "longitudinal_m": dlon,
            "yaw_deg": dyaw_deg,
            "position_before": [float(v) for v in
                                np.asarray(before["position"],
                                           dtype=np.float64).reshape(-1)[:3]],
            "heading_before": heading,
            "position_after": [float(v) for v in pos],
            "heading_after": float(heading + dyaw),
        }
        sys.stderr.write(
            f"[EVAL] frame {self.frame}: hand-off perturbation "
            f"lat={dlat:+.2f} m lon={dlon:+.2f} m yaw={dyaw_deg:+.1f} deg -> "
            f"({pos[0]:.2f}, {pos[1]:.2f}) heading {heading + dyaw:.3f}\n")
        sys.stderr.flush()

    def _execute_teleport_step(
        self, ego_state: Dict, images: Dict, needs_replan: bool,
        replan_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Teleport execution: advance by arc length at the plan's pacing
        speed (BridgeSim PID parity — see _run_inference_and_cache), teleport
        the ego to the paced point, and force a replan when the cached plan
        is exhausted. Returns ``(action, remaining_plan)``.
        """
        # Only called after _run_inference_and_cache populated the caches.
        assert (self._cached_world_traj is not None
                and self._cached_plan_arc is not None)
        live_dt = live_execution_dt(self.env)
        require_configured_clock(live_dt, self.config.sim_dt)
        require_cached_clock(live_dt, self._cached_execution_dt_s)
        temporal = getattr(self, "_cached_plan_speeds", None) is not None
        if temporal:
            self._plan_time_index = min(
                int(getattr(self, "_plan_time_index", 0)) + 1,
                len(self._cached_world_traj) - 1)
            target_pos = self._cached_world_traj[self._plan_time_index]
            seg_idx = max(0, self._plan_time_index - 1)
            world_traj_subset = self._cached_world_traj[seg_idx:]
            self._plan_arc_s = float(self._cached_plan_arc[self._plan_time_index])
            next_arc_s = self._plan_arc_s
        else:
            next_arc_s = self._plan_arc_s + self._cached_plan_speed * self.config.sim_dt
        if next_arc_s >= float(self._cached_plan_arc[-1]) and not needs_replan:
            logger.warning(
                f"[{self.scenario_id}] frame {self.frame}: consumed the "
                f"cached plan ({self._cached_plan_arc[-1]:.1f} m). Forcing replan."
            )
            self._run_inference_and_cache(ego_state, images)
            temporal = getattr(self, "_cached_plan_speeds", None) is not None
            if temporal:
                self._plan_time_index = min(1, len(self._cached_world_traj) - 1)
                target_pos = self._cached_world_traj[self._plan_time_index]
                seg_idx = 0
                world_traj_subset = self._cached_world_traj
                next_arc_s = float(self._cached_plan_arc[self._plan_time_index])
            else:
                next_arc_s = self._cached_plan_speed * self.config.sim_dt
        self._plan_arc_s = next_arc_s

        if not temporal:
            target_pos, seg_idx = self.point_at_arc(
                self._cached_world_traj, self._cached_plan_arc, self._plan_arc_s
            )
            world_traj_subset = self._cached_world_traj[seg_idx:]

        self._log_execution_debug(
            replan_offset,
            f"plan_speed={self._cached_plan_speed:.2f} m/s, "
            f"arc_s={self._plan_arc_s:.2f}, "
            f"traj_len={len(world_traj_subset)}, "
            f"ego=({ego_state['position'][0]:.2f}, {ego_state['position'][1]:.2f}), "
            f"heading={ego_state['heading']:.3f}, "
            f"target=({target_pos[0]:.2f}, {target_pos[1]:.2f})")

        # Teleport the ego to the paced point along the plan; velocity is
        # estimated from the position delta for the status feature/renderer.
        target_heading = self.heading_from_plan(
            world_traj_subset, ego_state["heading"])
        prev_pos = ego_state["position"][:2]
        vel_xy = (target_pos - prev_pos) / self.config.sim_dt
        self.env.set_ego_override(
            position=target_pos,
            heading=target_heading,
            velocity=np.array([vel_xy[0], vel_xy[1], 0.0], dtype=np.float32),
        )
        self._reactivity_control_context = {
            "execution_mode": "teleport",
            "source": "plan_pacing",
            "target_position": np.asarray(target_pos).tolist(),
            "target_heading_rad": float(target_heading),
            "target_speed_mps": float(self._cached_plan_speed),
            "plan_arc_m": float(self._plan_arc_s),
            "plan_segment_index": int(seg_idx),
        }
        # Dummy action (not used — ego is teleported).
        return np.zeros(3, dtype=np.float32), world_traj_subset

    def _execute_controller_step(
        self, ego_state: Dict, replan_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Controller/physics execution: the tracker chases the cached plan
        at its pacing speed and the env integrates the action (bicycle model,
        or PhysX when the env is in physics execution mode). Returns
        ``(action, remaining_plan)``.
        """
        # Only called after _run_inference_and_cache populated the cache.
        assert self._cached_world_traj is not None
        live_dt = live_execution_dt(self.env)
        require_configured_clock(live_dt, self.config.sim_dt)
        require_cached_clock(live_dt, self._cached_execution_dt_s)
        # The GT warm-up sets an ego override — clear it so step_ego runs.
        self.env.clear_ego_override()
        # The tracker gets the FULL cached plan (it projects the ego onto it
        # internally); the recording/vis subset is the remaining plan from
        # the ego's nearest point, matching the teleport branch's semantics.
        ego_xy = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        controller_traj = getattr(
            self, "_cached_controller_world_traj", self._cached_world_traj)
        nearest = int(np.argmin(np.linalg.norm(
            controller_traj - ego_xy, axis=1)))
        speed_profile = getattr(self, "_cached_controller_plan_speeds", None)
        # A timed trajectory is indexed by elapsed plan time, not by the
        # nearest spatial waypoint.  In particular, nearest is index zero on
        # every replan, which used to discard every future deceleration ramp.
        # nuPlan's LQR samples its reference velocity N*dt (one second with
        # the official N=10, dt=.1) ahead of the current trajectory time.
        # ``replan_offset`` advances that time between replans.
        timed_index = min(max(0, int(replan_offset)), len(controller_traj) - 1)
        lookahead = int(getattr(self.controller, "speed_lookahead_steps", 0))
        speed_index = min(timed_index + lookahead, len(controller_traj) - 1)
        target_speed = (float(speed_profile[speed_index])
                        if speed_profile is not None
                        else self._cached_plan_speed)
        steer, accel = self.controller.compute(
            ego_state, controller_traj, target_speed)
        stop_command = getattr(self, "_cached_stop_command", None)
        if stop_command is not None:
            from navsafe.core.stopping import stop_accel_input
            accel = stop_accel_input(stop_command)

        self._log_execution_debug(
            replan_offset,
            f"mode=controller plan_speed={target_speed:.2f} m/s, "
            f"speed_idx={speed_index}, nearest_idx={nearest}, "
            f"steer={steer:.3f}, accel={accel:.3f}, "
            f"ego=({ego_state['position'][0]:.2f}, {ego_state['position'][1]:.2f}), "
            f"speed={ego_state.get('speed', 0.0):.2f}")

        action = np.array([steer, accel, 0.0], dtype=np.float32)
        self._reactivity_control_context = {
            "execution_mode": self.config.execution_mode,
            "source": "trajectory_tracker",
            "target_speed_mps": float(target_speed),
            "nearest_plan_index": int(nearest),
            "speed_plan_index": int(speed_index),
            "commanded_steering": float(steer),
            "commanded_acceleration": float(accel),
        }
        return action, controller_traj[nearest:]

    def _log_execution_debug(self, replan_offset: int, message: str) -> None:
        """Per-replan [EVAL] debug line (stderr survives IsaacSim's stdout
        redirection)."""
        if replan_offset == 0 or self.frame == self.config.ego_replay_frames:
            sys.stderr.write(f"[EVAL] frame {self.frame}: {message}\n")
            sys.stderr.flush()

    def _record_current_trajectory(self, world_traj_subset: np.ndarray,
                                   ego_state: Dict) -> None:
        """Build the (N, 5) trajectory for recording/visualization."""
        n = len(world_traj_subset)
        world_z = np.full(
            n, ego_state["position"][2] if len(ego_state["position"]) > 2 else 0.0)
        dx_arr = np.gradient(world_traj_subset[:, 0]) if n > 1 else np.zeros(n)
        dy_arr = np.gradient(world_traj_subset[:, 1]) if n > 1 else np.zeros(n)
        self._current_trajectory = np.stack(
            [world_traj_subset[:, 0], world_traj_subset[:, 1], world_z,
             dx_arr, dy_arr], axis=1)

    def _record_replay_pose(self, ego_state: Dict) -> None:
        """Record one observed warm-up pose without invoking the policy."""
        position = np.asarray(ego_state["position"], dtype=np.float64)
        velocity = np.asarray(
            ego_state.get("velocity", [0.0, 0.0]), dtype=np.float64)
        z = float(position[2]) if position.size > 2 else 0.0
        vx = float(velocity[0]) if velocity.size > 0 else 0.0
        vy = float(velocity[1]) if velocity.size > 1 else 0.0
        self._current_trajectory = np.asarray(
            [[float(position[0]), float(position[1]), z, vx, vy]],
            dtype=np.float64,
        )

    def _run_inference_and_cache(self, ego_state: Dict, images: Dict) -> None:
        """Run model inference, interpolate trajectory, and cache in world coords.

        Mirrors BridgeSim's process_frame steps 4-6:
          1. Run model → ego-frame trajectory at model_dt intervals
          2. Interpolate to sim_dt intervals
          3. Transform to world coordinates and cache
        """
        frame_dt = live_execution_dt(self.env)
        require_configured_clock(frame_dt, self.config.sim_dt)
        if hasattr(self, "_episode_execution_dt_s"):
            require_cached_clock(frame_dt, self._episode_execution_dt_s)
        else:
            # Low-level cache callers have no earlier episode state.
            self._episode_execution_dt_s = frame_dt
        if frame_dt is not None:
            synchronize_tracker_dt(self.controller, frame_dt)
        self._cached_execution_dt_s = frame_dt
        ego_state.update(self._route.policy_context(ego_state["position"]))
        # The policy must score the same traffic state as collision detection,
        # rendering, and live EPDMS. In semi-reactive mode these poses include
        # traffic-manager overrides and can differ materially from the static
        # scenario log. Copy the snapshot because the env replaces/mutates its
        # actor list on subsequent simulation steps.
        ego_state["_execution_agent_states"] = [
            {
                **state,
                "position": np.asarray(state.get("position", []), dtype=float).copy(),
                "velocity": np.asarray(state.get("velocity", []), dtype=float).copy(),
            }
            for state in self.env.agent_states
        ]
        steering_angle = getattr(self.controller, "_steering_angle", None)
        if steering_angle is not None and np.isfinite(float(steering_angle)):
            ego_state["_execution_steering_angle_rad"] = float(steering_angle)
        max_steer = getattr(self.controller, "max_steer_angle", None)
        if max_steer is not None and np.isfinite(float(max_steer)):
            ego_state["_execution_max_steer_angle_rad"] = float(max_steer)

        model_input = self.adapter.prepare_input(
            images=images,
            ego_state=ego_state,
            scenario_data=self.scenario_data,
            frame_id=self.frame,
        )
        import time as _query_clock
        _query_start = _query_clock.perf_counter()
        raw_output = self.adapter.run_inference(model_input)
        _query_wall_s = _query_clock.perf_counter() - _query_start

        if self.trajectory_scorer is not None:
            policy_output = raw_output
            raw_output = self.trajectory_scorer.select_best(
                policy_output, ego_state=ego_state, frame_idx=self.frame
            )
            # PDM exposes its own selected trajectory as external candidate 0.
            # If that candidate is an upstream emergency brake, a generic
            # scorer's reduced return dict otherwise strips the full-state
            # marker and zero-speed profile; parse_output would then treat the
            # raw controller-correction poses as an ordinary geometric path.
            # Preserve brake semantics only when the external scorer actually
            # retained candidate 0. A different winning proposal correctly
            # overrides the planner brake and must not inherit its metadata.
            if (isinstance(policy_output, dict)
                    and policy_output.get("emergency_brake_triggered", False)):
                chosen = raw_output.get("best_idx")
                if hasattr(chosen, "detach"):
                    chosen = chosen.detach().cpu().numpy()
                chosen_arr = (np.asarray(chosen).reshape(-1)
                              if chosen is not None else np.empty(0))
                if chosen_arr.size and int(chosen_arr[0]) == 0:
                    raw_output["emergency_brake_triggered"] = True
                    speeds = policy_output.get("trajectory_speeds_mps")
                    if speeds is not None:
                        raw_output["trajectory_speeds_mps"] = speeds
            # Direct STOP actuator metadata belongs only to candidate zero,
            # the policy's own trajectory. A generic selector must neither
            # strip it from that motion nor attach it to a different winner.
            if (isinstance(policy_output, dict)
                    and policy_output.get("stop_command") is not None):
                chosen = raw_output.get("best_idx")
                if hasattr(chosen, "detach"):
                    chosen = chosen.detach().cpu().numpy()
                chosen_arr = (np.asarray(chosen).reshape(-1)
                              if chosen is not None else np.empty(0))
                if chosen_arr.size and int(chosen_arr[0]) == 0:
                    raw_output["stop_command"] = policy_output["stop_command"]
                    for key in ("stop_reference_trajectory", "stop_reference_speeds_mps"):
                        if key in policy_output:
                            raw_output[key] = policy_output[key]

        parsed = self.adapter.parse_output(raw_output, ego_state)
        from navsafe.core.stopping import (
            validate_stop_command, validate_stop_reference, validate_stop_reference_speeds,
        )
        # Every new plan clears or replaces its command. STOP does not latch
        # across replans, handbacks, scenes, or later ordinary model outputs.
        self._cached_stop_command = validate_stop_command(parsed.get("stop_command"))
        if self._cached_stop_command is not None and self.config.execution_mode == "teleport":
            raise ValueError("brake_to_stop requires controller or physics execution")
        traj_ego = np.asarray(parsed["trajectory"])  # (N, 2) ego frame [lateral, forward]
        parsed_speeds = parsed.get("trajectory_speeds_mps")
        if parsed_speeds is not None:
            parsed_speeds = np.asarray(
                parsed_speeds, dtype=np.float64).reshape(-1)

        # Cache parsed output and prediction pose for candidate visualization
        self._cached_parsed_output = parsed
        self._cached_prediction_position = np.asarray(ego_state["position"]).copy()
        self._cached_prediction_heading = float(ego_state["heading"])

        # Interpolate from model_dt to sim_dt, then anchor the plan at the ego
        # origin. The anchor is applied HERE, at the pacing seam, rather than
        # inside interpolate_plan's model_dt <= sim_dt pass-through: index 0
        # must be the pose _plan_arc_s is measured from, and this is the line
        # that knows which pose that is (the same ego_state the ego→world
        # transform below uses). ego_plan_with_origin is idempotent, so the
        # resampled branch — which already emits the t=0 sample — is untouched.
        model_dt = self.adapter.get_waypoint_dt()
        raw_traj_ego = np.array(traj_ego, copy=True)
        controller_reference_ego = raw_traj_ego
        controller_reference_speeds = parsed_speeds
        if self._cached_stop_command is not None:
            controller_reference_ego = validate_stop_reference(
                parsed.get("stop_reference_trajectory"))
            controller_reference_speeds = validate_stop_reference_speeds(
                parsed.get("stop_reference_speeds_mps"), controller_reference_ego)
        traj_ego = self._interpolate_trajectory(
            raw_traj_ego, model_dt, self.config.sim_dt)
        # A timed PDM path is a controller reference already projected onto
        # the selected lateral proposal. Keep it separate from the origin-
        # anchored path needed by teleport pacing: joining the physical ego
        # to an offset reference creates a fake diagonal steering segment.
        controller_ref = _plan_exec.controller_reference(
            controller_reference_ego, model_dt, self.config.sim_dt,
            controller_reference_speeds)
        controller_traj_ego = controller_ref.trajectory
        if model_dt <= self.config.sim_dt:
            # This branch is contractually future-only. Always insert t=0,
            # even if the t=dt pose happens to equal (0, 0) while stationary;
            # geometry alone cannot distinguish those two timestamps.
            if traj_ego.ndim == 2 and len(traj_ego):
                traj_ego = np.vstack([
                    np.zeros((1, traj_ego.shape[1]), dtype=traj_ego.dtype),
                    traj_ego,
                ])
        else:
            # Resampling emits t=0 already; keep this idempotent for empty or
            # unusual adapter outputs.
            traj_ego = _plan_exec.ego_plan_with_origin(traj_ego)

        # Visibility into degenerate plans: a first waypoint behind the ego
        # (negative forward) usually means the model output is noise. Index 0
        # is the ego origin by construction, so probe the first waypoint after
        # it — checking index 0 here would test a hardcoded 0.0 and never fire.
        first_wp = traj_ego[1] if len(traj_ego) > 1 else None
        if first_wp is not None and first_wp[1] < -0.1 and not self._warned_backward_plan:
            self._warned_backward_plan = True
            logger.warning(
                f"[{self.scenario_id}] frame {self.frame}: plan starts behind the "
                f"ego (forward={first_wp[1]:.2f} m) — degenerate model output? "
                f"(warning shown once per scenario)"
            )

        # Transform to world coordinates
        heading = float(ego_state["heading"])
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        ego_left = traj_ego[:, 0]
        ego_forward = traj_ego[:, 1]
        world_x = ego_state["position"][0] + cos_h * ego_forward - sin_h * ego_left
        world_y = ego_state["position"][1] + sin_h * ego_forward + cos_h * ego_left
        self._cached_world_traj = np.stack([world_x, world_y], axis=1)

        controller_left = controller_traj_ego[:, 0]
        controller_forward = controller_traj_ego[:, 1]
        controller_x = (ego_state["position"][0]
                        + cos_h * controller_forward
                        - sin_h * controller_left)
        controller_y = (ego_state["position"][1]
                        + sin_h * controller_forward
                        + cos_h * controller_left)
        controller_world_traj = np.stack([controller_x, controller_y], axis=1)

        # Plan pacing: advance along the plan by arc length at
        # PLAN_PACING_FACTOR × the plan's average speed, applied immediately.
        # Consuming the plan waypoint-by-waypoint instead executes only the
        # ramp-up start of every plan (each replan restarts the ramp), locking
        # the ego at the plan's initial crawl and feeding that low speed back
        # into the model's status feature. The arc is measured from the ego
        # pose (index 0 above), so _plan_arc_s = 0 means "at the ego".
        self._cached_plan_arc, self._cached_plan_speed = self.plan_pacing(
            self._cached_world_traj, self.config.sim_dt
        )
        preserves_timing = getattr(
            self.adapter, "preserves_trajectory_timing", lambda: False)()
        preserves_timing = preserves_timing or self._cached_stop_command is not None
        self._cached_controller_plan_speeds: Optional[np.ndarray]
        if preserves_timing and len(controller_world_traj) > 0:
            self._cached_controller_world_traj = controller_world_traj
            controller_speeds = controller_ref.speeds_mps
            self._cached_controller_plan_speeds = controller_speeds
            origin_added = (
                len(self._cached_world_traj) == len(controller_speeds) + 1)
            self._cached_plan_speeds = (
                np.concatenate([controller_speeds[:1], controller_speeds])
                if origin_added else controller_speeds.copy())
            self._cached_plan_speed = float(controller_speeds[0])
        else:
            self._cached_controller_world_traj = self._cached_world_traj
            self._cached_controller_plan_speeds = None
            self._cached_plan_speeds = None
        self._plan_arc_s = 0.0
        self._plan_time_index = 0

        # Also set _current_trajectory for compatibility (full trajectory)
        n = len(self._cached_world_traj)
        pos = np.asarray(ego_state["position"], dtype=np.float32)
        world_z = np.full(n, pos[2] if len(pos) > 2 else 0.0)
        dx = np.gradient(world_x) if n > 1 else np.zeros(n)
        dy = np.gradient(world_y) if n > 1 else np.zeros(n)
        self._current_trajectory = np.stack(
            [world_x, world_y, world_z, dx, dy], axis=1
        )
        self._record_plan(
            ego_state, parsed, raw_output, raw_traj_ego,
            inference_wall_s=_query_wall_s)

    @staticmethod
    def _plan_list(value: Any, ndim: int) -> Optional[list]:
        """``value`` as nested lists of floats, or None if it is not an
        ``ndim``-dimensional numeric array.

        Adapters return candidate sets as numpy or torch, batched or not, and
        a few return nothing at all. This is the one place that has to cope
        with that, so that :meth:`_record_plan` reads as a list of fields.
        """
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            arr = np.asarray(value, dtype=np.float64)
            # A leading singleton batch axis is the common adapter shape.
            while arr.ndim > ndim and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != ndim or arr.size == 0 or not np.all(np.isfinite(arr)):
                return None
            return np.round(arr, 4).tolist()
        except Exception:                                          # noqa: BLE001
            return None

    def _record_plan(self, ego_state: Dict, parsed: Dict, raw_output: Any,
                     raw_traj_ego: np.ndarray, *,
                     inference_wall_s: float = 0.0) -> None:
        """Persist one inference: every candidate, the pick, and the plan.

        Why this exists as its own artifact rather than being read back off
        the images: ``cam_f0_candidates.png`` DRAWS the candidate set, so the
        numbers are already computed — but once they are pixels they cannot be
        re-plotted, re-scaled or compared across arms. A trajectory
        distribution over a perturbation sweep is exactly that comparison, so
        the numbers are written out beside the pictures.

        Best-effort: a recording failure must never cost a scored frame.
        """
        try:
            pos = np.asarray(ego_state["position"], dtype=np.float64).reshape(-1)
            best = raw_output.get("best_idx") if isinstance(raw_output, dict) else None
            if hasattr(best, "detach"):
                best = best.detach().cpu().numpy()
            if best is not None:
                best_arr = np.asarray(best).reshape(-1)
                best = int(best_arr[0]) if best_arr.size else None
            record = {
                "frame": int(self.frame),
                "t_s": round(self.frame * float(self.config.sim_dt), 4),
                "is_warmup": bool(self.frame < self.config.ego_replay_frames),
                "ego_position": [round(float(v), 4) for v in pos[:3]],
                "ego_heading": round(float(ego_state["heading"]), 6),
                # Ego frame, [lateral(+left), forward], at the model's own
                # waypoint spacing -- NOT resampled to sim_dt, so a reader can
                # tell the model's horizon from the simulator's.
                "selected_ego": self._plan_list(raw_traj_ego, 2),
                # World frame, resampled to sim_dt: the plan the tracker chases.
                "selected_world": self._plan_list(self._cached_world_traj, 2),
                "selected_speeds_mps": self._plan_list(
                    parsed.get("trajectory_speeds_mps"), 1),
                # The candidate set, when the policy exposes one. Ego frame,
                # anchored at THIS record's ego pose.
                "candidates_ego": self._plan_list(
                    parsed.get("trajectory_coarse"), 3),
                "candidate_scores": self._plan_list(
                    parsed.get("coarse_scores"), 1),
                "selected_index": best,
                "emergency_brake": bool(
                    isinstance(raw_output, dict)
                    and raw_output.get("emergency_brake_triggered", False)),
            }
            # VLA chain-of-thought, when the policy narrates its decision.
            # AutoVLA returns it from parse_output and nothing else reads it,
            # so a failure case cannot be attributed to what the model claimed
            # to see without keeping it here. (MTDrive only surfaces its text
            # on a parse failure, so there is nothing to record on success.)
            reasoning = parsed.get("reasoning")
            if reasoning:
                record["reasoning"] = str(reasoning)
            # Auxiliary detection head, when the policy runs one: boxes as
            # (x, y, heading, length, width) in the ego BEV frame plus their
            # per-query logits. Separates "never represented the obstacle"
            # from "represented it and planned into it anyway".
            record["agent_states"] = self._plan_list(parsed.get("agent_states"), 2)
            record["agent_labels"] = self._plan_list(parsed.get("agent_labels"), 1)
            self._plan_records.append(record)
            if self._reactivity_trace is not None:
                self._reactivity_trace.record_plan(
                    frame=self.frame, ego_state=ego_state,
                    raw_traj_ego=raw_traj_ego,
                    selected_world=self._cached_world_traj,
                    controller_world=getattr(
                        self, "_cached_controller_world_traj", None),
                    selected_speeds=parsed.get("trajectory_speeds_mps"),
                    controller_speeds=getattr(
                        self, "_cached_controller_plan_speeds", None),
                    candidates_ego=parsed.get("trajectory_coarse"),
                    candidate_scores=parsed.get("coarse_scores"),
                    selected_index=best,
                    emergency_brake=record["emergency_brake"],
                    model_dt_s=float(self.adapter.get_waypoint_dt()),
                    inference_wall_s=float(inference_wall_s))
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("[plan-record] frame %d skipped: %s", self.frame, exc)

    # Plan-execution geometry: implemented in navsafe.core.plan_execution,
    # re-exported as static methods for backward compatibility (tests and
    # older call sites reference them via the Evaluator).
    plan_pacing = staticmethod(_plan_exec.plan_pacing)
    point_at_arc = staticmethod(_plan_exec.point_at_arc)
    heading_from_plan = staticmethod(_plan_exec.heading_from_plan)
    _interpolate_trajectory = staticmethod(_plan_exec.interpolate_plan)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_imwrite(path, img) -> bool:
        return EvalArtifactWriter._safe_imwrite(path, img)

    def _save_extra_sensor_artifacts(self, frame_dir) -> None:
        return self._artifacts._save_extra_sensor_artifacts(frame_dir)

    def _save_lidar_bev(self, frame_dir, hits) -> None:
        return self._artifacts._save_lidar_bev(frame_dir, hits)

    def _save_camera_extra(self, frame_dir, data_type: str, arr) -> None:
        return self._artifacts._save_camera_extra(frame_dir, data_type, arr)

    @staticmethod
    def _dump_npz(path, data: Dict) -> None:
        return EvalArtifactWriter._dump_npz(path, data)

    def _render_frame_vis(self, ego_state: Dict, images: Dict[str, np.ndarray], collision: bool) -> None:
        return self._artifacts._render_frame_vis(ego_state, images, collision)

    def _generate_visualizations(self) -> None:
        return self._artifacts._generate_visualizations()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._episode_execution_dt_s = live_execution_dt(self.env)
        require_configured_clock(self._episode_execution_dt_s, self.config.sim_dt)
        self.frame = 0
        self._done = False
        self._current_trajectory = None
        self._warmup_overlay_plan = None
        self._cached_world_traj = None
        self._cached_stop_command = None
        self._cached_execution_dt_s = None
        self._cached_plan_arc = None
        self._cached_plan_speed = 0.0
        self._plan_arc_s = 0.0
        self._cached_parsed_output = None
        self._cached_prediction_position = None
        self._cached_prediction_heading = None
        self.scenario_id = "unknown"
        self.scenario_path = None
        self.scenario_data = {}
        self.start_time = None
        self.end_time = None
        # Route waypoints (from GT trajectory, BridgeSim-style) — the route /
        # full_route deques are OWNED by the RouteManager subsystem; clear them
        # through it. The evaluator exposes them via read-only properties below.
        self._route.reset()
        self._current_waypoint = None   # (position, command, frame_idx) — current target
        self._current_command = -1      # route-based driving command (0=LEFT,1=RIGHT,2=STRAIGHT,3=LANEFOLLOW)
        # Prev-frame ego cache for finite-differenced acceleration /
        # angular_velocity that are required by the adapter contract
        # (spec §4.1).  Initialised to None so the first frame emits the
        # BridgeSim "harness default" accel/yaw-rate instead of a bogus diff.
        self._prev_ego_velocity = None
        self._prev_ego_heading = None
        self._warned_backward_plan = False
        self._collision_active = False
        self._step_crash = None
        # Hand-off perturbation: applied once, on the hand-off frame. Recorded
        # (rather than recomputed from the config) because the pose it actually
        # displaced is what a downstream reader needs.
        self._perturb_record: Optional[Dict[str, Any]] = None
        self._reactivity_trace = None
        self._reactivity_control_context: Dict[str, Any] = {}
        #: One entry per model inference — every candidate the policy produced,
        #: which one it picked, and the plan that came out. See
        #: :meth:`_record_plan`.
        self._plan_records: List[Dict[str, Any]] = []
        # NavSafe trace: None unless NAVSAFE_TRACE names a seed. Built here so
        # the step loop is a plain attribute test rather than an env lookup.
        try:
            from navsafe.benchmark.trace.hook import TraceHook
            self._navsafe_trace = TraceHook.from_env(
                warmup_frames=int(getattr(self.config, "ego_replay_frames", 0) or 0),
                sim_dt=float(getattr(self.config, "sim_dt", 0.1) or 0.1))
        except Exception:  # navsafe absent or unusable -> plain eval
            self._navsafe_trace = None
        # NavSafe termination, evaluated live. Built in setup() once the
        # scenario (and so the logged ego path the deviation columns are
        # measured against) is loaded; None until then.
        self._termination = None
        self._terminated_reason = None
        self._history: Dict[str, List] = {
            "trajectories": [],
            "vehicle_states": [],
            "metrics": [],
            "actions": [],
            "timestamps": [],
        }
        # Reset EPDMS live scorer state
        if hasattr(self, '_epdms_scorer') and self._epdms_scorer is not None:
            self._epdms_scorer.reset_live_state()
        self._epdms_results = []
        self._epdms_score_failures = 0
        self._warned_observe_failure = False

    def _termination_record(self) -> Dict[str, Any]:
        """The episode's ending as a plain dict (json/pickle safe)."""
        # A step-level crash preempts the monitor's reading: stepping stopped
        # because the evaluator broke, and the live monitor — seeing only the
        # truncated trace — would call that "budget_expired" and scorable.
        # It does NOT preempt a live termination recorded before the crash:
        # the episode had already ended for a real reason; only the
        # bookkeeping after it broke. Reason string matches
        # navsafe.termination.TerminationReason.INFRA_FAILURE, the taxonomy's
        # "excluded from every denominator" ending.
        crash = getattr(self, "_step_crash", None)
        if crash is not None and self._terminated_reason is None:
            return {"reason": "infra_failure", "frame": int(crash["frame"]),
                    "detail": f"evaluator step crashed: {crash['error']}",
                    "policy_attributed": False, "scorable": False}
        # An ending recorded live is the episode's, whether or not the monitor
        # that produced it is still around: this record decides whether the
        # episode is scorable, so it must follow the recorded fact and not the
        # presence of a collaborator.
        t = self._terminated_reason
        if t is None and self._termination is not None:
            # Same goal flag as the live update: an episode stopped by the
            # frame cap after finishing its route is GOAL_REACHED, not
            # BUDGET_EXPIRED.
            t = self._termination.final(goal_reached=self._route.goal_latched)
        if t is None:
            return {"reason": "unknown", "frame": self.frame, "detail": "",
                    "policy_attributed": True, "scorable": True}
        return {"reason": t.reason.value, "frame": int(t.frame),
                "detail": t.detail,
                "policy_attributed": bool(t.reason.policy_attributed),
                # False == report `—`, never 0: the benchmark ended it.
                # envelope_exit is retired and never produced; it stays in the
                # test so a stored trace re-read from before 2026-08-19 keeps
                # the verdict it was given.
                "scorable": t.reason.value not in ("envelope_exit", "infra_failure")}

    def _dist_to_stopline(self, ego_xy) -> float | None:
        """Signed metres to the stop line, positive before it. ``None`` when no
        stop line was resolved, which reads as "not checked" downstream."""
        if getattr(self, "_stopline_s", None) is None:
            return None
        path = getattr(self, "_stopline_path", None)
        if path is None or self._route_arc is None:
            return None
        d = np.linalg.norm(path[:, :2] - np.asarray(ego_xy)[:2], axis=1)
        return float(self._stopline_s) - float(self._route_arc[int(np.argmin(d))])

    def _build_termination_monitor(self):
        """LiveMonitor over the logged ego track, or None if navsafe is absent."""
        try:
            from navsafe.benchmark.termination import LiveMonitor
            from navsafe.scenario.scenario_description import ScenarioDescription as SD
        except Exception:  # navsafe absent -> plain eval, as before
            return None
        path = None
        try:
            scenario = getattr(self.env, "current_scenario", None) or {}
            sdc = str(scenario.get(SD.METADATA, {}).get(SD.SDC_ID, ""))
            track = (scenario.get(SD.TRACKS, {}) or {}).get(sdc)
            if track is not None:
                pos = np.asarray(track.get(SD.STATE, {}).get("position"))
                if pos.ndim == 2 and len(pos):
                    path = pos[:, :2]
        except Exception as exc:
            logger.warning("termination: no logged ego path (%s); the ego "
                           "deviation columns will be nan for this run", exc)
        dt = float(getattr(self.config, "sim_dt", 0.1) or 0.1)
        # The monitor and step loop share one clock. A configured route limit
        # uses the same safety ceiling as post-hoc scoring; ``None`` means that
        # route completion is genuinely clock-free while all semantic
        # termination rules remain active.
        route_limit = getattr(self.config, "route_time_limit_s", SAFETY_CEILING_S)
        t_max_s = float("inf") if route_limit is None else float(route_limit)

        # The leaf's rule departures, resolved from the bundle manifest the
        # caller passed. Wrong leaf metadata must not silently change scoring,
        # so a conflicting pair raises out of rules_for_scenario rather than
        # being guessed at.
        rules = None
        try:
            from navsafe.benchmark.scenario_rules import rules_for_scenario
            rules = rules_for_scenario(getattr(self.config, "scenario_meta", None))
            if not rules.is_default:
                logger.info("termination: scenario rules for leaf %s in force "
                            "(%s)", rules.leaf, rules.note)
        except Exception as exc:      # noqa: BLE001 - default rules on any doubt
            logger.warning("termination: could not resolve scenario rules "
                           "(%s); using the defaults", exc)
        # Where the ego's signalled connector meets the logged route, for the
        # hold leaves. Without it a V-1 episode cannot see its own crossing and
        # runs to the 60 s ceiling on every cell -- correct once re-scored, but
        # three times the wall clock, and the live log never says what happened.
        self._stopline_s = None
        self._route_arc = None
        self._stopline_path = path
        if rules is not None and rules.hold_region == "stop_line" and path is not None:
            try:
                from navsafe.benchmark.trace.from_eval import _arc, stopline_arc
                from navsafe.benchmark.trace.writer import lane_centerlines

                scenario = getattr(self.env, "current_scenario", None) or {}
                self._stopline_s = stopline_arc(scenario, path,
                                                lane_centerlines(scenario))
                self._route_arc = _arc(path)
                logger.info("termination: stop line at route arc %s",
                            self._stopline_s)
            except Exception as exc:                         # noqa: BLE001
                logger.warning("termination: no stop line resolved (%s); the "
                               "hold will be judged post-hoc only", exc)
        return LiveMonitor(path,
                           warmup=int(getattr(self.config, "ego_replay_frames", 0) or 0),
                           dt=dt, t_max_s=t_max_s, rules=rules)

    def generate_route(self) -> None:
        return self._route.generate_route()

    def get_next_waypoint(self, current_position: np.ndarray):
        return self._route.get_next_waypoint(current_position)


    @staticmethod
    def _lift_trajectory(traj_xy: np.ndarray, ego_state: Dict) -> np.ndarray:
        """Convert (N, 2) ego-frame waypoints to world-frame (N, 5).

        The ego-frame convention (matching BridgeSim / NavSim):
            traj_xy[:, 0] = lateral  (left-positive)
            traj_xy[:, 1] = forward  (longitudinal)

        Columns: [world_x, world_y, world_z, vx_approx, vy_approx]
        """
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        n = len(traj_xy)

        pos = np.asarray(ego_state["position"], dtype=np.float32)  # (3,)
        heading = float(ego_state["heading"])

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        ego_left = traj_xy[:, 0]
        ego_forward = traj_xy[:, 1]

        # Rotate ego-frame (lateral, forward) → world XY
        world_x = pos[0] + cos_h * ego_forward - sin_h * ego_left
        world_y = pos[1] + sin_h * ego_forward + cos_h * ego_left
        world_z = np.full(n, pos[2] if len(pos) > 2 else 0.0)

        # Approximate velocity from finite differences
        dx = np.gradient(world_x)
        dy = np.gradient(world_y)

        return np.stack([world_x, world_y, world_z, dx, dy], axis=1)  # (N, 5)

    @staticmethod
    def _safe_stack(arrays):
        """Stack a list of arrays, handling empty lists and shape mismatches."""
        if not arrays:
            return np.array([])
        try:
            return np.stack(arrays)
        except ValueError:
            # Different shapes — return as object array
            return np.array(arrays, dtype=object)

    def _save_results(self, results: Dict[str, Any]) -> None:
        # Artifacts go DIRECTLY in output_dir. They used to nest under
        # <output_dir>/<scenario_id>/, but scenario_id is the scenario
        # path's last segment, which for a py123d root (.../<clip>/arrow)
        # is 'arrow' for every clip — a constant, meaningless directory.
        # --output-dir is required, so the caller already names the
        # destination; scenario identity still travels inside metrics.json.
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        def _to_serializable(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.integer, np.floating)):
                return v.item()
            return v

        # metrics.json
        metrics_serial = {k: _to_serializable(v) for k, v in results["metrics"].items()}
        (output_dir / "metrics.json").write_text(json.dumps(metrics_serial, indent=2))

        # trajectory.npy
        if results["trajectory_history"].size > 0:
            np.save(output_dir / "trajectory.npy", results["trajectory_history"])

        # vehicle_states.npy
        if self._history["vehicle_states"]:
            positions = np.array([s.get("position", [0, 0, 0]) for s in self._history["vehicle_states"]])
            np.save(output_dir / "vehicle_states.npy", positions)

        # agent_states.json — per-frame actor poses, when asked for. See the
        # note at the accumulation site: the scenario's `tracks` hold a
        # recipe-inserted actor's SPAWN pose repeated, never its driven path,
        # so this is the only record of where an inserted hazard actually was.
        if getattr(self, "_dump_agent_states", False) and getattr(self, "_agent_state_log", None):
            (output_dir / "agent_states.json").write_text(json.dumps(
                {"dt": float(getattr(self.config, "dt", 0.1) or 0.1),
                 "frames": [{"frame": f, "agents": rows}
                            for f, rows in self._agent_state_log]}))

        # per_frame_metrics.json is no longer written — the per-frame record
        # lives in driving_score_summary.csv below.

        # driving_score_summary.csv — BridgeSim's consolidated artifact
        # (base_evaluator._save_driving_score_summary): per-frame EPDMS
        # subscore rows + one AVERAGE row carrying DS / EPDMS_no_ep / RC.
        # metrics.json above stays as the machine-readable equivalent; the
        # former summary.json duplicated it and is no longer written.
        # Written whenever there is a per-frame record, NOT only when EPDMS
        # produced one. The file's COLL* columns are this evaluator's own
        # contact record and have nothing to do with EPDMS; gating the whole
        # artifact on EPDMS meant that when live scoring raised on every frame
        # (seen 2026-08-25, 87/87 frames), the fact that the ego had driven
        # into an inserted animal reached no artifact at all. Downstream that
        # became a `trace_exhausted` termination charged to nobody and a VRU
        # contact scored in the vehicle channel -- a clean-looking 12.18.
        if self._epdms_results or self._history["metrics"]:
            self._save_driving_score_summary(output_dir, results["metrics"])

        self._save_plan_records(output_dir)

        logger.info(f"Results saved → {output_dir}")

    def _save_plan_records(self, output_dir: Path) -> None:
        """``plan_records.json``: what the policy considered, picked and drove.

        ``trajectory.npy`` already holds the executed plan per frame, but it
        is the plan AFTER interpolation, transformation and pacing, with no
        candidate set, no pick and no record of where the ego actually ended
        up. Reading a trajectory distribution off it is guesswork. This file
        is the primary artifact for that reading, and it carries the hand-off
        perturbation that produced the arm so a plot never has to infer which
        run it is looking at from a directory name.

        Best-effort: an episode that scored must not be lost to a JSON dump.
        """
        try:
            executed = []
            metrics = self._history["metrics"]
            stamps = self._history["timestamps"]
            for i, state in enumerate(self._history["vehicle_states"]):
                pos = np.asarray(state.get("position", [0.0, 0.0, 0.0]),
                                 dtype=np.float64).reshape(-1)
                fm = metrics[i] if i < len(metrics) else {}
                executed.append({
                    "frame": i,
                    "t_s": round(float(stamps[i]), 4) if i < len(stamps) else None,
                    "position": [round(float(v), 4) for v in pos[:3]],
                    "heading": round(float(state.get("heading", 0.0)), 6),
                    "speed_mps": round(float(state.get("speed", 0.0)), 4),
                    "collision": bool((fm or {}).get("collision", False)),
                })
            payload = {
                "scenario_id": self.scenario_id,
                "model": type(self.adapter).__name__,
                "sim_dt": float(self.config.sim_dt),
                "replan_rate": int(self.config.replan_rate),
                "handoff_frame": int(self.config.ego_replay_frames),
                # Null when the arm is unperturbed -- which is itself the fact
                # a reader needs, and is not the same as "the file predates
                # perturbation support".
                "perturbation": self._perturb_record,
                "perturbation_requested": {
                    "lateral_m": float(getattr(
                        self.config, "ego_perturb_lateral_m", 0.0) or 0.0),
                    "longitudinal_m": float(getattr(
                        self.config, "ego_perturb_longitudinal_m", 0.0) or 0.0),
                    "yaw_deg": float(getattr(
                        self.config, "ego_perturb_yaw_deg", 0.0) or 0.0),
                },
                "predictions": self._plan_records,
                "executed": executed,
            }
            (output_dir / "plan_records.json").write_text(
                json.dumps(payload, separators=(",", ":")))
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("[plan-record] plan_records.json not written: %s", exc)

    _EPDMS_SUBSCORE_COLS = (
        ("no_at_fault_collisions", "NC"),
        ("drivable_area_compliance", "DAC"),
        ("driving_direction_compliance", "DDC"),
        ("traffic_light_compliance", "TL"),
        ("time_to_collision_within_bound", "TTC"),
        ("lane_keeping", "LK"),
        ("history_comfort", "HC"),
        ("extended_comfort", "EC"),
        # Not a subscore: the scorer's per-frame "red light ahead" state
        # fact, persisted so a post-hoc re-score can apply the NavSafe
        # deadlock exemption the live monitor applied (termination.py). Its
        # AVERAGE-row entry is therefore the held-frame fraction, not a mean
        # subscore.
        ("signal_hold", "TL_HOLD"),
    )

    def _save_driving_score_summary(self, output_dir: Path, metrics: Dict[str, Any]) -> None:
        """Write BridgeSim-style ``driving_score_summary.csv``: one row per
        scored frame (subscores + per-frame EP / EPDMS_no_ep / EPDMS-with-EP)
        + an AVERAGE row over valid frames carrying the composites and
        DS / RC. Extends ``bridgesim/evaluation/core/base_evaluator.py``'s
        layout with the per-frame EP and score columns.
        """
        import csv

        available = [(src, dst) for src, dst in self._EPDMS_SUBSCORE_COLS
                     if any(src in r for r in self._epdms_results)]
        # COLL / COLL_AF carry the contact and its fault attribution. EPDMS's
        # NC column is an at-fault flag, so without these a not-at-fault
        # contact leaves no trace in any artifact and the NavSafe termination
        # taxonomy can never report `contact_not_at_fault` for a stored run.
        # COLL_ID / COLL_KIND persist WHICH agent the contact was with and what
        # kind it was. Without them the only record is `contact_detail` inside
        # the stdout dump of per_frame_metrics, so a stored run forces every
        # reader to re-infer the pair from geometry -- and the nearest-agent
        # guess in navsafe/trace/from_eval.py got it wrong on
        # 05d0a1a763fc5334/drivor: the evaluator recorded a rear_end with
        # 1235f522a2cb5ade, the trace reported an angle with 5f5b0326236a5740.
        header = (["frame_id", "valid"] + [dst for _, dst in available]
                  + ["EP", "EPDMS_no_ep", "EPDMS", "COLL", "COLL_AF",
                     "COLL_ID", "COLL_KIND", "DS", "RC"])
        per_frame = self._history["metrics"]

        def _fmt(r: Dict[str, Any], key: str) -> Any:
            return round(float(r[key]), 6) if key in r else ""

        # The evaluator's OWN per-frame record drives the rows; EPDMS joins in
        # by frame id where it has something. The other way round -- iterating
        # `self._epdms_results` -- silently dropped every frame EPDMS failed on,
        # taking the contact columns with it even though they were never EPDMS's
        # to begin with. Frames EPDMS did score come out byte-identical; frames
        # it did not now appear with empty subscore cells, which is the honest
        # representation of "not scored" and what `read_per_frame_epdms` already
        # treats as unknown.
        epdms_by_frame = {r["frame"]: r for r in self._epdms_results
                          if isinstance(r.get("frame"), int)}
        rows = []
        for fid, fm in enumerate(per_frame):
            r = epdms_by_frame.get(fid, {})
            fm = fm or {}
            rows.append([fid, r.get("valid", "")]
                        + [(round(float(r[src]), 6) if src in r else "")
                           for src, _ in available]
                        + [_fmt(r, "ego_progress"), _fmt(r, "score"),
                           _fmt(r, "score_with_ep"),
                           int(bool(fm.get("collision", False))),
                           int(bool(fm.get("collision_at_fault", False))),
                           (fm.get("contact_detail") or {}).get("agent_id", ""),
                           (fm.get("contact_detail") or {}).get("kind", ""),
                           "", ""])
        valid = [r for r in self._epdms_results if r.get("valid")]

        def _mean(key: str) -> Any:
            vals = [float(r[key]) for r in valid if key in r]
            return round(sum(vals) / len(vals), 6) if vals else ""

        # Mixed str/float row (csv.writer stringifies) — annotate so
        # mypy does not pin the row to list[str] from the first two cells.
        avg: list = ["AVERAGE", ""]
        for src, _ in available:
            avg.append(_mean(src))
        avg += [_mean("ego_progress"),
                round(float(metrics.get("epdms_no_ep", 0.0)), 6),
                _mean("score_with_ep"),
                # Contact totals over the run, matching the per-frame columns.
                sum(1 for m in per_frame if m.get("collision")),
                sum(1 for m in per_frame if m.get("collision_at_fault")),
                "", "",
                round(float(metrics.get("driving_score", 0.0)), 6),
                round(float(metrics.get("route_completion_fraction", 0.0)), 6)]
        rows.append(avg)
        with (output_dir / "driving_score_summary.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
