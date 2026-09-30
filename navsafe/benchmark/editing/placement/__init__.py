# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Placement: authored intent -> an absolute pose on a real reference line."""

from navsafe.benchmark.editing.placement.probe import (
    CrossSection,
    HostProbe,
    PlacementError,
    resolve_reference,
)
from navsafe.benchmark.editing.placement.visibility import (
    DEFAULT_MIN_REACTION_S,
    VisibilityReport,
    occlusion_report,
    summarise as summarise_visibility,
)
from navsafe.benchmark.editing.placement.solve import (
    Polyline,
    intersect_polylines,
    solve_start_arc_for_conflict,
)

__all__ = [
    "DEFAULT_MIN_REACTION_S",
    "VisibilityReport",
    "occlusion_report",
    "summarise_visibility",
    "CrossSection",
    "HostProbe",
    "PlacementError",
    "Polyline",
    "intersect_polylines",
    "resolve_reference",
    "solve_start_arc_for_conflict",
]
