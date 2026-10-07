"""Authoritative execution timing shared by evaluation and collection."""
from __future__ import annotations

from dataclasses import replace
import math
from numbers import Real
from typing import Any, Optional

import numpy as np


def live_execution_dt(owner: Any) -> Optional[float]:
    """Read the plant clock, then explicit environment/representation clocks.

    An invalid higher-priority clock is unknown; another configuration cannot
    make it valid. No planner/evaluator default is evidence of elapsed time.
    """
    env = getattr(owner, "env", None)
    if env is None:
        env = owner
    plant_cfg = getattr(getattr(env, "_ego", None), "cfg", None)
    missing = object()
    candidates = ((plant_cfg, "dt"), (getattr(env, "cfg", None), "dt"),
                  (env, "dt"))
    selected = None
    for source, name in candidates:
        try:
            value = getattr(source, name, missing)
        except (TypeError, ValueError, OverflowError):
            return None
        if value is missing:
            continue
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        if selected is not None and selected != value:
            raise ValueError("ego plant and environment execution timesteps disagree")
        selected = value
    if selected is not None or owner is env:
        return selected
    # A representation clock is a fallback only when its underlying
    # environment exposes none. It cannot override contradictory world time.
    try:
        value = getattr(owner, "dt", None)
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            return None
        value = float(value)
        return value if math.isfinite(value) and value > 0.0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def synchronize_tracker_dt(tracker: Any, dt: float) -> None:
    """Set the built-in tracker clock without resetting its actuator state."""
    from navsafe.core.controllers import LQRTracker, PurePursuitTracker

    if isinstance(tracker, LQRTracker):
        if tracker._cfg.sim_dt != dt:
            tracker._cfg = replace(tracker._cfg, sim_dt=dt)
        tracker.sim_dt = dt
    elif isinstance(tracker, PurePursuitTracker):
        tracker.sim_dt = dt
    # A caller-supplied custom tracker owns its own configuration. Do not
    # mutate it or infer built-in state semantics from a matching attribute.


def require_configured_clock(live_dt: Optional[float], configured_dt: Any) -> None:
    """Keep execution, timers, and native configuration receipts consistent."""
    if (not isinstance(configured_dt, Real) or isinstance(configured_dt, (bool, np.bool_))
            or not math.isfinite(float(configured_dt)) or float(configured_dt) <= 0.0):
        raise ValueError("configured execution timestep must be finite and positive")
    if live_dt is not None and live_dt != float(configured_dt):
        raise ValueError(
            f"configured execution timestep {float(configured_dt)!r} differs from live "
            f"plant timestep {live_dt!r}; configure matching execution clocks before planning")


def require_cached_clock(live_dt: Optional[float], cached_dt: Optional[float]) -> None:
    """A cached plan/certificate cannot span a change of the execution clock."""
    if live_dt != cached_dt:
        raise ValueError(
            f"execution timestep changed during cached plan ({cached_dt!r} -> {live_dt!r}); "
            "reset the scene before executing a new plan")
