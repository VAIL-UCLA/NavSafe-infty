# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory: the three authoring templates, and the bake that resolves them."""

from navsafe.benchmark.editing.trajectory.bake import (
    BakeError,
    bake_actor_state,
)
from navsafe.benchmark.editing.trajectory.templates import (
    Motion,
    TemplateError,
    build_motion,
)

__all__ = [
    "BakeError",
    "Motion",
    "TemplateError",
    "bake_actor_state",
    "build_motion",
]
