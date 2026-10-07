"""Stateless bootstrap helpers for :class:`~navsafe.env.navsafe_env.NavSafeEnv`.

This module holds the pure, env-instance-free logic that used to live at the top
of ``navsafe_env.py``: the supported-axis table, the loader/traffic-manager
factories, config validation, asset-bundle checks, and the IsaacLab cfg
augmentation. None of it touches a live env, a sim, or a render backend — it is
plain data + functions over an :class:`~navsafe.env.env_cfg.EnvCfg`, so it
imports without IsaacSim and is trivially unit-testable in isolation.

Extracted from ``navsafe_env.py`` to shrink that module and give the bootstrap
concern its own narrow home. ``navsafe_env`` re-exports every public name here
(``from navsafe.env._bootstrap import *``-style, but explicit) so existing
imports — ``from navsafe.env.navsafe_env import SUPPORTED_COMBINATIONS``,
``_validate_cfg``, ``_check_asset_bundles``, ``_get_cache_root``,
``_format_supported_combinations`` — keep resolving unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

from navsafe.env.env_cfg import EnvCfg


# Default agent dimensions by type (length, width, height in metres).
# Ported from ``ScenarioReplayEnv._DEFAULT_DIMS``; used by
# ``NavSafeEnv._collect_agent_states_for_renderer`` when a track omits its
# bbox dims.
_DEFAULT_AGENT_DIMS: Dict[str, Tuple[float, float, float]] = {
    "VEHICLE": (4.5, 1.8, 1.5),
    "PEDESTRIAN": (0.5, 0.5, 1.7),
    "CYCLIST": (1.8, 0.6, 1.6),
    "TRAFFIC_CONE": (0.3, 0.3, 0.8),
}


# ======================================================================
# Supported axis combinations
# ======================================================================

# Keep invalid traffic modes from silently falling back to log replay.
IDM_TRAFFIC_UNWIRED_MSG = (
    "traffic_mode='idm' is unsupported. Use 'semi_reactive' for IDM takeover "
    "of nearby followers, 'log_replay' for recorded traffic, or 'no_traffic'."
)

SUPPORTED_COMBINATIONS: list[Tuple[str, str, str, str]] = [
    (obs, traffic, source, "nurec_grpc")
    for obs in ("state", "sensor")
    for traffic in ("no_traffic", "log_replay")
    for source in ("py123d",)
]

# semi_reactive is a drop-in superset of log_replay (same scenario data, same
# renderer contract — the manager just hands near-ego vehicles to IDM), so it
# is supported exactly where log_replay is.
SUPPORTED_COMBINATIONS += [
    (obs, "semi_reactive", scenario, render)
    for (obs, traffic, scenario, render) in list(SUPPORTED_COMBINATIONS)
    if traffic == "log_replay"
]

# navsafe is likewise a superset of log_replay — same scenario data, and it
# reaches the renderer through the same agent-state list. It differs from
# semi_reactive in owning no prims, which is what lets it run under the
# closed-loop evaluator; nurec_grpc is that evaluator's backend, and until
# now no reactive mode was registered against it at all.
SUPPORTED_COMBINATIONS += [
    (obs, "navsafe", scenario, render)
    for (obs, traffic, scenario, render) in list(SUPPORTED_COMBINATIONS)
    if traffic == "log_replay"
]


def _format_supported_combinations() -> str:
    """Format the supported combinations as a human-readable string."""
    lines = []
    for obs, traffic, scenario, render in SUPPORTED_COMBINATIONS:
        lines.append(
            f"  (obs_kind={obs!r}, traffic_mode={traffic!r}, "
            f"scenario_source={scenario!r}, render_backend={render!r})"
        )
    return "\n".join(lines)


def _make_loader(cfg: EnvCfg) -> Any:
    """Create the py123d Arrow loader; reject other input formats."""
    if cfg.scenario_source != "py123d":
        raise ValueError(f"Unsupported scenario_source={cfg.scenario_source!r}; use 'py123d'.")
    from navsafe.env.loaders import Py123DLoader
    return Py123DLoader()


def _make_traffic_manager(cfg: EnvCfg) -> Any:
    """Return the traffic manager instance selected by ``cfg.traffic_mode``.

    Dispatches the ``traffic_mode`` axis to the matching episode-level
    manager from ``navsafe.traffic`` (each a
    :class:`~navsafe.traffic.base.TrafficManager` subclass):

    * ``"no_traffic"`` → :class:`NoTrafficManager`
    * ``"log_replay"`` → :class:`LogReplayTraffic`
    * ``"semi_reactive"`` → :class:`SemiReactiveTraffic` (MetaDrive's
      ``reactive_traffic``: followers behind the ego and within
      ``cfg.semi_reactive_radius_m`` laterally get a trajectory-IDM policy,
      everything else replays; ``cfg.semi_reactive_takeover`` decides whether
      that test runs every frame or only at spawn)
    * ``"idm"``        → **refused** (:exc:`NotImplementedError`): the mode
      was a silent no-op — see :data:`IDM_TRAFFIC_UNWIRED_MSG`

    Managers are imported lazily here so this module stays import-safe and
    cheap to import; the traffic managers themselves are pure-Python and do
    not pull in IsaacSim.

    Args:
        cfg: The environment configuration.

    Returns:
        A :class:`~navsafe.traffic.base.TrafficManager` instance.

    Raises:
        NotImplementedError: If ``cfg.traffic_mode == "learned"`` (M1/M2
            future work) or ``"idm"`` (recognised but unwired — see
            :data:`IDM_TRAFFIC_UNWIRED_MSG`); both are excluded from
            ``SUPPORTED_COMBINATIONS``.
        ValueError: If ``cfg.traffic_mode`` is not a known mode.
    """
    mode = cfg.traffic_mode
    if mode == "no_traffic":
        from navsafe.traffic import NoTrafficManager
        return NoTrafficManager()
    if mode == "log_replay":
        from navsafe.traffic import LogReplayTraffic
        return LogReplayTraffic()
    if mode == "idm":
        raise NotImplementedError(IDM_TRAFFIC_UNWIRED_MSG)
    if mode == "semi_reactive":
        from navsafe.traffic import SemiReactiveTraffic
        # Deliberately NOT parameterised from the ``idm_*`` cfg knobs: this
        # mode is a port of MetaDrive's reactive traffic, and its IDM
        # constants (TrajectoryIDMPolicy) are part of what is being
        # reproduced. Feeding NavSafe's own IDM headway in here would make
        # the two managers differ in more than reactivity.
        return SemiReactiveTraffic(
            side_constraint_m=getattr(cfg, "semi_reactive_radius_m", 15.0),
            takeover=getattr(cfg, "semi_reactive_takeover", "continuous"),
        )
    if mode == "navsafe":
        from navsafe.traffic.navsafe import NavSafeTraffic
        return NavSafeTraffic()
    if mode == "learned":
        raise NotImplementedError(
            "traffic_mode='learned' (LearnedTraffic) is M1/M2 future work "
            "and is not yet implemented. It is excluded from "
            "SUPPORTED_COMBINATIONS."
        )
    raise ValueError(
        f"Unknown traffic_mode={mode!r}. "
        f"Valid modes are: 'no_traffic', 'log_replay', 'semi_reactive', "
        f"'navsafe'."
    )


def _validate_cfg(cfg: EnvCfg) -> None:
    """Validate that the EnvCfg axis combination is supported.

    Raises:
        ValueError: If the combination is not in SUPPORTED_COMBINATIONS, or
            the execution mode is unknown.
    """
    execution_mode = getattr(cfg, "execution_mode", "kinematic")
    if execution_mode not in ("kinematic", "physics"):
        raise ValueError(
            f"Unsupported EnvCfg.execution_mode: {execution_mode!r} "
            "(expected 'kinematic' or 'physics')")

    # idm is recognised but unwired: refuse it HERE, before the combination
    # table, so the error explains itself instead of printing the whole
    # supported-combination list. (It used to select a manager whose step
    # was a silent no-op — reactive traffic that never reacted.)
    if cfg.traffic_mode == "idm":
        raise NotImplementedError(IDM_TRAFFIC_UNWIRED_MSG)

    combo = (cfg.obs_kind, cfg.traffic_mode, cfg.scenario_source, cfg.render_backend)
    if combo not in SUPPORTED_COMBINATIONS:
        supported_str = _format_supported_combinations()
        raise ValueError(
            f"Unsupported EnvCfg axis combination: "
            f"obs_kind={cfg.obs_kind!r}, "
            f"traffic_mode={cfg.traffic_mode!r}, "
            f"scenario_source={cfg.scenario_source!r}, "
            f"render_backend={cfg.render_backend!r}.\n\n"
            f"Currently supported combinations:\n{supported_str}"
        )


# ======================================================================
# Asset bundle detection
# ======================================================================

def _get_cache_root() -> Path:
    """Return the NavSafe asset cache root directory."""
    home = os.environ.get("NAVSAFE_HOME")
    if home:
        return Path(home)
    return Path.home() / ".cache" / "navsafe"


def _check_asset_bundles(cfg: EnvCfg) -> None:
    """Check that all required asset bundles are present in the cache.

    Raises:
        FileNotFoundError: If any required bundle is missing.
    """
    if not cfg.required_bundles:
        return

    cache_root = _get_cache_root()
    missing: list[str] = []

    for bundle_name in cfg.required_bundles:
        bundle_dir = cache_root / "assets" / bundle_name
        if not bundle_dir.exists():
            missing.append(bundle_name)

    if missing:
        commands = "\n".join(f"  navsafe-pull-assets {b}" for b in missing)
        raise FileNotFoundError(
            f"Required asset bundle(s) not found in local cache "
            f"({cache_root / 'assets'}):\n"
            f"  {', '.join(missing)}\n\n"
            f"Please fetch them with:\n{commands}\n\n"
            f"See `navsafe-pull-assets --help` for details."
        )


def _augment_cfg_for_isaaclab(cfg: EnvCfg) -> None:
    """Attach the attributes IsaacLab 3.0's ``DirectRLEnvCfg`` contract requires.

    ``EnvCfg`` is intentionally a plain, IsaacSim-free dataclass (so the
    pure-Python state path imports without IsaacSim). IsaacLab 3.0's
    ``DirectRLEnv.__init__`` reads a richer config contract than the fork this
    code was first written against (it now requires ``validate()``, ``seed``,
    ``sim``, ``scene``, ``decimation``, the gym-space slots, …). Rather than
    pull IsaacSim types into ``EnvCfg`` at class-definition time, we attach the
    required attributes onto the instance here — only when IsaacLab is present,
    importing its types lazily. Existing attributes are never overwritten.
    """
    from isaaclab.sim import SimulationCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.envs import ViewerCfg

    def _default(name: str, value: Any) -> None:
        if not hasattr(cfg, name):
            setattr(cfg, name, value)

    if not hasattr(cfg, "validate"):
        cfg.validate = lambda *a, **k: None  # type: ignore[attr-defined]

    render_kwargs: Dict[str, Any] = {}

    # Single-env, render-capable simulation. NavSafeEnv builds its own road /
    # agent geometry directly on ``sim.stage`` in ``_setup_scene`` (it does not
    # register assets through InteractiveScene), so a minimal empty scene cfg
    # is sufficient for DirectRLEnv to construct.
    # Physics execution mode: zero gravity — the ego physics proxy is the
    # only dynamic body (velocity-driven, no ground colliders exist), so
    # global zero-g simply protects it during warmup substeps.
    if getattr(cfg, "execution_mode", "kinematic") == "physics":
        _default("sim", SimulationCfg(dt=float(getattr(cfg, "dt", 0.1)),
                                      render_interval=1,
                                      gravity=(0.0, 0.0, 0.0), **render_kwargs))
    else:
        _default("sim", SimulationCfg(dt=float(getattr(cfg, "dt", 0.1)),
                                      render_interval=1, **render_kwargs))
    _default("scene", InteractiveSceneCfg(num_envs=1, env_spacing=0.0))
    _default("viewer", ViewerCfg())
    _default("decimation", 1)
    _default("seed", None)
    _default("events", None)
    _default("ui_window_class_type", None)        # skip the IsaacLab UI window
    _default("action_noise_model", None)
    _default("observation_noise_model", None)
    # NavSafeEnv overrides _get_observations / _apply_action, so the exact gym
    # spaces are not load-bearing for the render path — provide valid minimals.
    _default("observation_space", 1)
    _default("action_space", 2)
    _default("state_space", 0)
    _default("num_observations", None)
    _default("num_actions", None)
    _default("num_states", None)
    _default("rerender_on_reset", False)
    _default("num_rerenders_on_reset", 0)
    _default("is_finite_horizon", False)
    _default("episode_length_s", 1.0e9)           # effectively unbounded
    _default("wait_for_textures", True)
