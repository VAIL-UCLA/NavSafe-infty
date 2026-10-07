# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""CaRL PDM-Closed LQR reference-profile helpers.

This is a dependency-light port of CaRL commit ``2677d14``
``batch_lqr_utils.py``.  CaRL fits velocity/acceleration and
curvature/curvature-rate profiles from the ideal proposal poses, then tracks
those fitted profiles.  Keeping the fit here (rather than silently reading
the ideal proposal's IDM speed array) makes the NavSafe proposal simulator
use the same controller inputs as the reference implementation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


INITIAL_CURVATURE_PENALTY = 1e-10


def _batch_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bij,bjk->bik", a, b)


def generate_profile(
    initial_condition: np.ndarray,
    derivatives: np.ndarray,
    discretization_time: float,
) -> np.ndarray:
    """Integrate derivatives exactly as CaRL's batch-LQR helper does."""
    if discretization_time <= 0.0:
        raise ValueError("discretization_time must be positive")
    cumsum = np.cumsum(derivatives * discretization_time, axis=-1)
    return initial_condition[..., None] + np.pad(
        cumsum, [(0, 0), (1, 0)], mode="constant"
    )


def _normalize_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _difference_matrix(number_rows: int) -> np.ndarray:
    out = np.zeros((number_rows, number_rows + 1), dtype=np.float64)
    eye = np.eye(number_rows, dtype=np.float64)
    out[:, 1:] = eye
    out[:, :-1] = -eye
    return out


def velocity_curvature_profiles_from_poses(
    poses: np.ndarray,
    *,
    discretization_time: float,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return CaRL-fitted velocity and curvature profiles.

    Args:
        poses: ``(batch, num_poses, 3)`` arrays of ``x, y, heading``.

    Returns:
        ``(velocity, curvature)``, each shaped ``(batch, num_poses - 1)``.
    """
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 3 or values.shape[2] != 3:
        raise ValueError(
            "poses must have shape (batch, num_poses>=3, 3), "
            f"got {values.shape}"
        )
    if discretization_time <= 0.0:
        raise ValueError("discretization_time must be positive")
    if jerk_penalty <= 0.0 or curvature_rate_penalty <= 0.0:
        raise ValueError("profile regularization penalties must be positive")

    pose_differences = np.diff(values, axis=1)
    xy_displacements = pose_differences[..., :2]
    heading_displacements = _normalize_angle(pose_differences[..., 2])
    headings = values[:, :-1, 2]

    batch_size, num_displacements, _ = xy_displacements.shape
    y = xy_displacements.reshape(batch_size, -1)
    a_column = np.zeros_like(y)
    a_column[:, 0::2] = np.cos(headings)
    a_column[:, 1::2] = np.sin(headings)
    design = np.repeat(
        a_column[..., None] * discretization_time**2,
        num_displacements,
        axis=2,
    )
    design[..., 0] = a_column * discretization_time
    upper = np.triu(
        np.ones((num_displacements, num_displacements), dtype=bool), k=1
    )
    design[:, np.repeat(upper, 2, axis=0)] = 0.0

    banded = _difference_matrix(num_displacements - 2)
    regularizer = np.block(
        [np.zeros((len(banded), 1), dtype=np.float64), banded]
    )
    regularizer = np.repeat(regularizer[None, ...], batch_size, axis=0)
    design_t = np.transpose(design, (0, 2, 1))
    regularizer_t = np.transpose(regularizer, (0, 2, 1))
    intermediate = _batch_matmul(
        np.linalg.pinv(
            _batch_matmul(design_t, design)
            + jerk_penalty * _batch_matmul(regularizer_t, regularizer)
        ),
        design_t,
    )
    solution = np.einsum("bij,bj->bi", intermediate, y)
    velocity = generate_profile(
        solution[:, 0], solution[:, 1:], discretization_time
    )

    batch_dim, dim = heading_displacements.shape
    curvature_design = np.repeat(
        np.tri(dim, dtype=np.float64)[None, ...], batch_dim, axis=0
    )
    curvature_design[:, :, 0] = velocity * discretization_time
    integrated_velocity = velocity * discretization_time**2
    curvature_design[:, 1:, 1:] *= integrated_velocity[:, None, 1:].transpose(
        0, 2, 1
    )
    curvature_regularizer = curvature_rate_penalty * np.eye(
        dim, dtype=np.float64
    )
    curvature_regularizer[0, 0] = INITIAL_CURVATURE_PENALTY
    curvature_design_t = curvature_design.transpose(0, 2, 1)
    curvature_intermediate = _batch_matmul(
        np.linalg.pinv(
            _batch_matmul(curvature_design_t, curvature_design)
            + curvature_regularizer
        ),
        curvature_design_t,
    )
    curvature_solution = np.einsum(
        "bij,bj->bi", curvature_intermediate, heading_displacements
    )
    curvature = generate_profile(
        curvature_solution[:, 0],
        curvature_solution[:, 1:],
        discretization_time,
    )
    return velocity, curvature


def longitudinal_lqr_acceleration(
    current_velocity: float,
    reference_velocity: float,
    *,
    discretization_time: float,
    tracking_horizon: int,
    q_longitudinal: float,
    r_longitudinal: float,
    stopping_velocity: float,
    stopping_proportional_gain: float,
) -> float:
    """CaRL's scalar longitudinal LQR, including its stopping P branch."""
    current = float(current_velocity)
    reference = float(reference_velocity)
    if current <= stopping_velocity and reference <= stopping_velocity:
        return -stopping_proportional_gain * (current - reference)
    horizon_time = float(tracking_horizon) * float(discretization_time)
    denominator = horizon_time**2 * q_longitudinal + r_longitudinal
    return float(
        -(horizon_time * q_longitudinal * (current - reference))
        / denominator
    )


def reference_slice(
    profile: np.ndarray, current_index: int, tracking_horizon: int
) -> Tuple[float, np.ndarray]:
    """CaRL lookahead scalar and padded horizon slice for one profile."""
    values = np.asarray(profile, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("profile must be non-empty")
    reference_index = min(current_index + tracking_horizon, values.size - 1)
    reference_value = float(values[reference_index])
    length = max(0, reference_index - current_index)
    out = np.empty(tracking_horizon, dtype=np.float64)
    if length:
        out[:length] = values[current_index:reference_index]
    out[length:] = reference_value
    return reference_value, out


__all__ = [
    "generate_profile",
    "longitudinal_lqr_acceleration",
    "reference_slice",
    "velocity_curvature_profiles_from_poses",
]
