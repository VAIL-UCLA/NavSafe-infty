# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Read and transform INRIA-layout 3DGS PLYs.

The writer already lives in :mod:`navsafe.tools.convert_mesh_to_3dgs` (mesh ->
gaussians -> file); what composition needs on top is the other direction, plus a
rigid transform that moves a whole asset without breaking it.

**Frame.** Files are stored y-up, because the NuRec server rotates every loaded
PLY by a fixed y-up -> z-up matrix on insert. So in file space height is +y and
"yaw" is a rotation about **+y**, not +z. Composing a rider onto a bike means
lifting along +y and sliding along +x.

**Encodings.** ``scale`` is the natural log of the eigen-scales and ``opacity``
is a logit, per the INRIA layout — so a uniform resize adds ``log(s)`` rather
than multiplying, and opacity is left alone.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

_BASE = ["x", "y", "z", "nx", "ny", "nz"]


class PlyError(NexusSimError, ValueError):
    """The file is not a 3DGS PLY this pipeline can work with."""


def read_3dgs_ply(path: "str | Path") -> Dict[str, np.ndarray]:
    """Parse a binary-little-endian float32 3DGS PLY into a gaussian dict.

    Returns:
        ``{"xyz", "normals", "f_dc", "f_rest", "opacity", "scale", "rot"}``,
        the same shape :func:`navsafe.tools.convert_mesh_to_3dgs.write_3dgs_ply`
        consumes.

    Raises:
        PlyError: not a binary float32 PLY, or missing required properties.
    """
    path = Path(path)
    with open(path, "rb") as handle:
        raw = handle.read()
    marker = b"end_header\n"
    cut = raw.find(marker)
    if cut < 0:
        raise PlyError(f"{path}: no PLY header terminator")
    header = raw[:cut].decode("ascii", "replace").splitlines()
    body = raw[cut + len(marker) :]

    if not any(line.startswith("format binary_little_endian") for line in header):
        raise PlyError(f"{path}: only binary_little_endian PLYs are supported")
    count = None
    fields = []
    for line in header:
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property "):
            parts = line.split()
            if parts[1] != "float":
                raise PlyError(f"{path}: property {parts[-1]!r} is {parts[1]}, expected float")
            fields.append(parts[-1])
    if count is None:
        raise PlyError(f"{path}: header declares no vertex element")
    table = np.frombuffer(body, dtype="<f4", count=count * len(fields)).reshape(count, len(fields))
    index = {name: i for i, name in enumerate(fields)}

    def take(names: Sequence[str], *, required: bool = True, default=None):
        missing = [n for n in names if n not in index]
        if missing:
            if required:
                raise PlyError(f"{path}: missing properties {missing}")
            return default
        return np.ascontiguousarray(table[:, [index[n] for n in names]].astype(np.float32))

    n_rest = len([f for f in fields if f.startswith("f_rest_")])
    return {
        "xyz": take(["x", "y", "z"]),
        "normals": take(["nx", "ny", "nz"], required=False, default=np.zeros((count, 3), np.float32)),
        "f_dc": take([f"f_dc_{i}" for i in range(3)]),
        "f_rest": (
            take([f"f_rest_{i}" for i in range(n_rest)])
            if n_rest
            else np.zeros((count, 0), np.float32)
        ),
        "opacity": np.ascontiguousarray(table[:, index["opacity"]].astype(np.float32))
        if "opacity" in index
        else np.zeros(count, np.float32),
        "scale": take([f"scale_{i}" for i in range(3)]),
        "rot": take([f"rot_{i}" for i in range(4)]),
    }


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` for (w, x, y, z) rows."""
    aw, ax, ay, az = (a[..., i] for i in range(4))
    bw, bx, by, bz = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32)


def transform_gaussians(
    gaussians: Dict[str, np.ndarray],
    *,
    translate: Sequence[float] = (0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
    scale: float = 1.0,
    sh_policy: str = "strict",
) -> Dict[str, np.ndarray]:
    """Rigidly move and resize a whole asset, in the y-up file frame.

    Args:
        gaussians: a gaussian dict.
        translate: metres along file-space ``(x, y, z)``; ``y`` is up.
        yaw_deg: rotation about the vertical (**+y**) axis.
        scale: uniform resize about the origin.
        sh_policy: what to do with view-dependent bands under a rotation.
            ``"strict"`` (default) refuses, because rotating spherical
            harmonics properly means rotating each band and this is not that;
            ``"drop"`` zeroes them, turning the asset diffuse — visible as a
            loss of specular highlight, but never wrong; ``"keep"`` leaves them
            untouched, which is only correct for a pure translation.

    Raises:
        PlyError: a rotation would silently corrupt non-zero SH bands.
    """
    out = {k: np.array(v, copy=True) for k, v in gaussians.items()}
    factor = float(scale)
    if factor <= 0:
        raise PlyError(f"scale must be positive, got {scale}")

    theta = np.deg2rad(float(yaw_deg))
    if abs(theta) > 1e-9 and out["f_rest"].size and float(np.abs(out["f_rest"]).max()) > 0:
        if sh_policy == "strict":
            raise PlyError(
                "this asset carries non-zero view-dependent SH bands and rotating them "
                "correctly is not implemented. Pass sh_policy='drop' to make the asset "
                "diffuse (it loses specular highlights but stays correct), or place it "
                "with yaw_deg=0."
            )
        if sh_policy == "drop":
            out["f_rest"] = np.zeros_like(out["f_rest"])
        elif sh_policy != "keep":
            raise PlyError(f"unknown sh_policy {sh_policy!r}; expected strict | drop | keep")

    xyz = out["xyz"] * factor
    if abs(theta) > 1e-9:
        cos_t, sin_t = np.float32(np.cos(theta)), np.float32(np.sin(theta))
        # Ry: (x, y, z) -> (c*x + s*z, y, -s*x + c*z)
        xyz = np.stack(
            [cos_t * xyz[:, 0] + sin_t * xyz[:, 2], xyz[:, 1], -sin_t * xyz[:, 0] + cos_t * xyz[:, 2]],
            axis=1,
        )
        half = np.array(
            [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0], np.float32
        )  # (w, x, y, z) about +y
        out["rot"] = _quat_mul(np.broadcast_to(half, out["rot"].shape), out["rot"])
    out["xyz"] = (xyz + np.asarray(translate, np.float32)).astype(np.float32)
    # ``scale`` holds log(sigma); a uniform resize is an offset there.
    out["scale"] = (out["scale"] + np.float32(np.log(factor))).astype(np.float32)
    return out


_AXES = {
    "+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0),
}


def reorient_gaussians(
    gaussians: Dict[str, np.ndarray],
    *,
    forward: str,
    up: str,
    sh_policy: str = "drop",
) -> Dict[str, np.ndarray]:
    """Rotate an asset into the pipeline's frame: **y-up with +x forward**.

    Exporters disagree about axes — a z-up file with its length along +y is
    just as common as our convention, and the difference is invisible in the
    numbers until a bicycle renders 0.3 m long and lying on its side. State
    which file axis points along the object's nose (``forward``) and which
    points at the sky (``up``); this maps those onto +x and +y and takes the
    third axis from ``forward x up``, so the result is a proper rotation
    (never a mirror).

    ``assets calibrate --forward-axis/--up-axis`` is the CLI for this, and a
    quick way to READ the axes off an unknown file is the principal axis of
    its point cloud (the longest extent is almost always the length).

    Args:
        forward, up: one of ``+x -x +y -y +z -z``, in the FILE's frame.
        sh_policy: as :func:`transform_gaussians` — defaults to ``"drop"``
            here because a re-orientation is a large rotation and keeping
            unrotated view-dependent bands would light the asset wrongly.

    Raises:
        PlyError: unknown axis name, or forward parallel to up.
    """
    try:
        f = np.asarray(_AXES[forward.lower()], np.float64)
        u = np.asarray(_AXES[up.lower()], np.float64)
    except KeyError as exc:
        raise PlyError(f"axis must be one of {sorted(_AXES)}, got {exc}") from None
    if abs(float(np.dot(f, u))) > 1e-9:
        raise PlyError(f"forward {forward!r} and up {up!r} must be perpendicular")
    # Rows of M are the file-frame directions that become target +x, +y, +z.
    M = np.stack([f, u, np.cross(f, u)], axis=0)
    out = {k: np.array(v, copy=True) for k, v in gaussians.items()}
    if out["f_rest"].size and float(np.abs(out["f_rest"]).max()) > 0:
        if sh_policy == "strict":
            raise PlyError(
                "this asset carries non-zero view-dependent SH bands; a re-orientation "
                "rotates them out of alignment. Pass sh_policy='drop' (diffuse but "
                "correct) or 'keep' if the bands are known to be negligible."
            )
        if sh_policy == "drop":
            out["f_rest"] = np.zeros_like(out["f_rest"])
        elif sh_policy != "keep":
            raise PlyError(f"unknown sh_policy {sh_policy!r}")
    out["xyz"] = np.ascontiguousarray((out["xyz"].astype(np.float64) @ M.T).astype(np.float32))
    out["normals"] = np.ascontiguousarray(
        (out["normals"].astype(np.float64) @ M.T).astype(np.float32))
    # Compose the same rotation onto every gaussian's own orientation.
    trace = float(np.trace(M))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (M[2, 1] - M[1, 2]) / s,
                      (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s])
    else:  # pick the largest diagonal element for numerical stability
        i = int(np.argmax(np.diag(M)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + M[i, i] - M[j, j] - M[k, k]) * 2.0
        q = np.zeros(4)
        q[0] = (M[k, j] - M[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (M[j, i] + M[i, j]) / s
        q[1 + k] = (M[k, i] + M[i, k]) / s
    q = (q / np.linalg.norm(q)).astype(np.float32)
    out["rot"] = _quat_mul(np.broadcast_to(q, out["rot"].shape), out["rot"])
    return out


def concat_gaussians(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Merge several gaussian dicts into one, padding SH bands to the widest."""
    parts = [p for p in parts if len(p["xyz"])]
    if not parts:
        raise PlyError("nothing to concatenate")
    width = max(p["f_rest"].shape[1] for p in parts)
    padded = []
    for part in parts:
        if part["f_rest"].shape[1] < width:
            pad = np.zeros((len(part["xyz"]), width - part["f_rest"].shape[1]), np.float32)
            part = {**part, "f_rest": np.concatenate([part["f_rest"], pad], axis=1)}
        padded.append(part)
    return {
        key: np.concatenate([p[key] for p in padded], axis=0)
        for key in ("xyz", "normals", "f_dc", "f_rest", "opacity", "scale", "rot")
    }


def bounding_dims(gaussians: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Axis-aligned extent in file space, and the (l, w, h) it implies.

    Files are y-up with +x forward, so length is the x extent, width the z
    extent and height the y extent — the order a track's ``dims`` wants.
    """
    xyz = gaussians["xyz"]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    return {
        "length": float(hi[0] - lo[0]),
        "width": float(hi[2] - lo[2]),
        "height": float(hi[1] - lo[1]),
        "base_y": float(lo[1]),
    }


def ply_sha256(path: "str | Path", *, chunk: int = 1 << 20) -> str:
    """Content hash of an asset file — what a recipe pins its identity by."""
    import hashlib

    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_3dgs_ply(gaussians: Dict[str, np.ndarray], out_path: "str | Path") -> Path:
    """Write a gaussian dict, reusing the converter's canonical writer."""
    from navsafe.tools.convert_mesh_to_3dgs import write_3dgs_ply as _write

    out_path = Path(out_path)
    _write(gaussians, out_path)
    return out_path


__all__ = [
    "PlyError",
    "reorient_gaussians",
    "bounding_dims",
    "concat_gaussians",
    "ply_sha256",
    "read_3dgs_ply",
    "transform_gaussians",
    "write_3dgs_ply",
]
