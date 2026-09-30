"""Shared test fixtures for the Gaussian-splat renderer: minimal PLY writer and
in-memory cloud builder (used by renderer + gs3d-bridge tests)."""

from __future__ import annotations

import numpy as np

from navsafe.render.ply_utils import GaussianCloud

_PLY_PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_min_ply(path, means, dc=2.0, opacity_logit=12.0, scale_log=-0.7):
    """Write a minimal standard 3DGS binary-LE PLY (bright, opaque gaussians)."""
    means = np.asarray(means, np.float32).reshape(-1, 3)
    n = len(means)
    rows = np.zeros((n, len(_PLY_PROPS)), np.float32)
    rows[:, 0:3] = means
    rows[:, 3:6] = dc            # f_dc -> bright colour
    rows[:, 6] = opacity_logit   # opacity (pre-sigmoid) -> ~1
    rows[:, 7:10] = scale_log    # scale (pre-exp)
    rows[:, 10] = 1.0            # rot_0 (w) = 1 -> identity quaternion
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join("property float %s\n" % p for p in _PLY_PROPS)
    header += "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rows.astype("<f4").tobytes())


def tiny_cloud(means):
    """An in-memory GaussianCloud at the given means (identity rot, opaque)."""
    means = np.asarray(means, np.float32).reshape(-1, 3)
    n = len(means)
    return GaussianCloud(
        means=means,
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        scales=np.full((n, 3), 0.5, np.float32),
        opacities=np.ones(n, np.float32),
        sh=np.ones((n, 1, 3), np.float32),
        sh_degree=0,
    )
