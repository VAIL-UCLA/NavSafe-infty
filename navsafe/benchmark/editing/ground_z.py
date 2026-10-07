# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Where the road is: one resolver for the ego-pose-to-ground drop.

History, because the number looks arbitrary otherwise. The loader once reported
the ego at its **bounding-box centre**, ~0.85 m above the road, so the rendered
camera and every route-anchored placement floated by that much and each caller
carried its own magic constant. That was fixed at the source:
:func:`navsafe.scenario.py123d_training_extractor._extract_ego` now takes x, y
and heading from ``center_se3`` but **z from the ground reference**
(``rear_axle_se3`` / ``imu_se3``), so **the ego z means the road under the ego**
and the correct drop is **0**.

What remains is not a rig height but a per-artifact **recon-road drift** — the
rendered road disagreeing with the real one:

* lidar-supervised artifacts -> ~0 (the recon road is the real road);
* camera-only artifacts      -> a small (±0.4 m) drift, calibrated per scene.

Precedence, matching ``navsafe/cli/eval_entry.py``: **explicit value >
per-scene registry > 0**. The registry is opt-in through the
``NUREC_GROUND_Z_CALIB`` environment variable so no site path is baked into the
repo, and it is keyed by the data root's parent directory name.

Two caveats worth knowing before trusting a 0:

* ``rear_axle_se3 ≈ imu ≈ on the road`` was established **for WOD**. On other
  sources the rear-axle pose can sit some way above the road, which shows up as
  a constant residual — measure it with
  :meth:`~navsafe.benchmark.editing.placement.probe.HostProbe.measure_road_drift`
  rather than assuming.
* The registry's eventual replacement is the **lidar-derived ground mesh** baked
  into the artifact (``checkpoint.artifact.mesh.ground.enabled``), a per-(x, y)
  ground-z oracle. It is disabled on the car2sim recipe the navhard recons use,
  so the registry is still the live mechanism there.

Everything that decides where the road is must read the SAME number — the
renderer's navsim camera (``NUREC_GRPC_EGO_Z_TO_GROUND``), the front-cam map
overlay (``NAVSAFE_EGO_Z_TO_GROUND``) and asset placement — or cameras,
overlays and objects disagree about the ground.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: Opt-in path to a ``{scene_key: drop_metres}`` JSON of per-scene recon drift.
REGISTRY_ENV = "NUREC_GROUND_Z_CALIB"

#: The environment variables every road-height consumer reads.
CONSUMER_ENV = ("NUREC_GRPC_EGO_Z_TO_GROUND", "NAVSAFE_EGO_Z_TO_GROUND")


def scene_key(source_path: "str | Path") -> str:
    """Registry key for a data root or scenario pickle: its parent directory."""
    return Path(str(source_path).rstrip("/")).parent.name


def resolve_z_to_ground(
    explicit: Optional[float],
    source_path: "str | Path",
    *,
    registry_path: "str | Path | None" = None,
) -> Tuple[float, str]:
    """Resolve the ego-pose-to-ground drop for one scene.

    Args:
        explicit: a value the caller was given directly. Wins over everything.
        source_path: the py123d data root (or scenario pickle) being run.
        registry_path: override the ``NUREC_GROUND_Z_CALIB`` path.

    Returns:
        ``(metres, reason)`` — the drop and where it came from, so a run can log
        which of the three sources it used rather than printing a bare number.
    """
    if explicit is not None:
        return float(explicit), "explicit"
    raw = registry_path if registry_path is not None else os.environ.get(REGISTRY_ENV)
    key = scene_key(source_path)
    if raw:
        path = Path(raw)
        if path.exists():
            try:
                registry = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("ground-z registry %s is unreadable (%s); using 0", path, exc)
                registry = {}
            if key in registry:
                return float(registry[key]), f"registry {path} [{key}]"
    return 0.0, (
        f"default 0 for {key!r} (the ego z is ground-referenced; a nonzero drop is only "
        f"recon-road drift)"
    )


def export_to_consumers(value: float) -> None:
    """Publish the drop so renderer and overlay agree with the placement.

    Uses ``setdefault`` semantics: an operator who has already pinned one of
    these keeps their value.
    """
    for name in CONSUMER_ENV:
        os.environ.setdefault(name, str(float(value)))
