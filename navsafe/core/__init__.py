# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python primitives with no IsaacSim dependencies.

Per design.md §4 (Components and Interfaces), ``navsafe.core`` houses
pure-Python code that is unit-testable on a laptop without a GPU or
IsaacSim installation:

* Ego dynamics
* Controllers (Pure Pursuit, PID) — relocated from ``navsafe.primitives.controller``
* Geometry helpers
* Collision math
* Frame conversions

**No IsaacSim imports are allowed in this package.**
"""

from navsafe.core.unified_controller import (
    ControllerType,
    UnifiedController,
)
from navsafe.core.ego_dynamics import (
    CollisionDetector,
    EgoDynamics,
    EgoDynamicsCfg,
)

__all__ = [
    "CollisionDetector",
    "ControllerType",
    "EgoDynamics",
    "EgoDynamicsCfg",
    "UnifiedController",
]
