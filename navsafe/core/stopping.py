"""Physical braking commands shared by admission and live plan execution.

The command replaces only longitudinal actuator input. Steering remains the
production tracker's output for the selected reference, including its behavior
on curves and short paths. A command is valid for one cached plan, never a
persistent stop latch. Replanning must explicitly carry, replace, or release it.

This module simulates motion, not safety: reaching zero does not imply that
front, rear, surface, or traffic-signal checks pass.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np


MIN_STOP_BRAKE_FRACTION = 0.25
# Shared by continuation admission and fallback execution. Intermediate and
# gentle profiles can be the only collision-free choice between front and
# rear traffic; checking only half/full braking misses those continuations.
# This finite bank does not claim to exhaust every fraction in [0.25, 1].
STOP_BRAKE_FRACTIONS = (0.5, 0.75, 1.0, 0.25)


def make_stop_command(brake_fraction: float = 1.0) -> Dict[str, Any]:
    """Build a serializable command using a fraction of the live plant brake.

    One quarter to full brake is the supported profile range. Fractions near
    zero are deliberately excluded: they are coasting requests and would make
    a finite stop verification horizon arbitrarily long.
    """
    if isinstance(brake_fraction, (bool, np.bool_)):
        raise ValueError("stop_brake_fraction must be a number, not a boolean")
    fraction = float(brake_fraction)
    if not (math.isfinite(fraction)
            and MIN_STOP_BRAKE_FRACTION <= fraction <= 1.0):
        raise ValueError(
            f"stop_brake_fraction must be in [{MIN_STOP_BRAKE_FRACTION}, 1.0]")
    return {"kind": "brake_to_stop", "brake_fraction": fraction}


def validate_stop_command(command: Any) -> Optional[Dict[str, Any]]:
    """Validate metadata without silently turning a malformed stop into motion."""
    if command is None:
        return None
    if not isinstance(command, Mapping):
        raise ValueError("stop_command must be a mapping or None")
    if command.get("kind") != "brake_to_stop":
        raise ValueError("unknown stop_command kind")
    if set(command) != {"kind", "brake_fraction"}:
        raise ValueError("stop_command requires only kind and brake_fraction")
    return make_stop_command(command["brake_fraction"])




def validate_stop_reference(reference: Any) -> np.ndarray:
    """Keep the steering reference explicit and separate from stopped labels."""
    path = np.asarray(reference, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) == 0
            or not np.isfinite(path).all()):
        raise ValueError("stop_reference_trajectory must be finite nonempty (N, 2)")
    return path.copy()


def validate_stop_reference_speeds(speeds: Any, reference: np.ndarray) -> Optional[np.ndarray]:
    """Reject malformed explicit timing instead of silently changing steering."""
    if speeds is None:
        return None
    values = np.asarray(speeds, dtype=np.float64).reshape(-1)
    if (len(values) != len(reference) or not np.isfinite(values).all()
            or (values < 0.0).any()):
        raise ValueError("stop_reference_speeds_mps must match reference length and be finite nonnegative")
    return values.copy()


def stop_accel_input(command: Mapping[str, Any]) -> float:
    """Normalized brake input applied after computing ordinary steering.

    Keeping this negative at rest makes holding exact under EgoDynamics's
    nonnegative-speed clipping. It does not create reverse movement.
    """
    validated = validate_stop_command(command)
    if validated is None:
        raise ValueError("a stop command is required")
    return -float(validated["brake_fraction"])

