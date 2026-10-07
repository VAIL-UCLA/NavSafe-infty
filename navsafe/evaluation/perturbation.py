"""Continuous target paths for controller-generated perturbation histories."""
from __future__ import annotations
import math
import numpy as np


def augmented_warmup_path(positions, headings, handoff_frame, lateral_m,
                          longitudinal_m, yaw_deg):
    """Blend a rigid transform into the logged path by the handoff.

    This is a reference, not an imposed ego state. Actual achieved offsets
    depend on the controller and must be recorded separately.
    """
    xy = np.asarray(positions, dtype=float)[:, :2]
    headings = np.asarray(headings, dtype=float)
    k = int(handoff_frame)
    if not 1 <= k < len(xy):
        raise ValueError("controller history needs a valid positive handoff frame")
    if not np.isfinite(xy).all() or not np.isfinite(headings).all():
        raise ValueError("nonfinite logged path")
    if not all(math.isfinite(x) for x in (lateral_m, longitudinal_m, yaw_deg)):
        raise ValueError("nonfinite perturbation")
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    h = float(headings[k])
    translation = np.array([math.cos(h)*longitudinal_m-math.sin(h)*lateral_m,
                            math.sin(h)*longitudinal_m+math.cos(h)*lateral_m])
    transformed = (xy-xy[k]) @ rotation.T + xy[k] + translation
    u = np.minimum(np.arange(len(xy), dtype=float)/k, 1.0)
    weight = 10*u**3-15*u**4+6*u**5
    return xy + weight[:, None]*(transformed-xy)
