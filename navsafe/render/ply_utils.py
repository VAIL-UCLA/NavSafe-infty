"""Gaussian PLY asset utilities used by NuRec actor insertion."""
from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np

_PLY_FLOAT_TYPES = {"float", "float32"}


class GaussianCloud:
    """In-memory 3DGS model loaded from a single PLY frame."""

    __slots__ = ("means", "quats", "scales", "opacities", "sh", "sh_degree")

    def __init__(
        self,
        means:     np.ndarray,   # (N, 3) float32
        quats:     np.ndarray,   # (N, 4) float32 [w,x,y,z] normalised
        scales:    np.ndarray,   # (N, 3) float32 already exp'd
        opacities: np.ndarray,   # (N,)   float32 already sigmoid'd
        sh:        np.ndarray,   # (N, K, 3) float32
        sh_degree: int,
    ):
        self.means     = means
        self.quats     = quats
        self.scales    = scales
        self.opacities = opacities
        self.sh        = sh
        self.sh_degree = sh_degree

    @property
    def n(self) -> int:
        return len(self.means)


def load_ply(path: "str | Path") -> GaussianCloud:
    """Load a standard 3DGS binary-little-endian PLY and return a GaussianCloud."""
    path = Path(path)
    with open(path, "rb") as f:
        n_verts, prop_names, _types = _parse_ply_header(f)
        raw = np.frombuffer(
            f.read(n_verts * len(prop_names) * 4), dtype="<f4"
        ).reshape(n_verts, len(prop_names))

    col = {name: i for i, name in enumerate(prop_names)}

    means = np.stack([raw[:, col["x"]], raw[:, col["y"]], raw[:, col["z"]]], 1)

    quats = np.stack([raw[:, col[f"rot_{i}"]] for i in range(4)], 1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(min=1e-8)

    scales = np.exp(np.stack([raw[:, col[f"scale_{i}"]] for i in range(3)], 1))

    opacities = 1.0 / (1.0 + np.exp(-raw[:, col["opacity"]]))

    dc = np.stack([raw[:, col[f"f_dc_{c}"]] for c in range(3)], 1)  # (N, 3)

    rest_names = sorted(
        [p for p in prop_names if p.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rest_names:
        rest = np.stack([raw[:, col[r]] for r in rest_names], 1)  # (N, 3*(K-1))
        n_extra   = len(rest_names) // 3
        sh_degree = int(math.sqrt(n_extra + 1)) - 1
        K         = (sh_degree + 1) ** 2
        sh        = np.zeros((n_verts, K, 3), np.float32)
        sh[:, 0, :] = dc
        # SH layout: grouped by channel (R then G then B), all non-DC bases
        for c in range(3):
            sh[:, 1:, c] = rest[:, c * n_extra:(c + 1) * n_extra]
    else:
        sh_degree = 0
        sh        = dc[:, None, :].astype(np.float32)

    return GaussianCloud(
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=opacities.astype(np.float32),
        sh=sh,
        sh_degree=sh_degree,
    )


def scale_ply(src: "str | Path", dst: "str | Path", factor: float) -> Path:
    """Write ``src`` resized by ``factor`` about the file origin.

    A uniform resize of a gaussian cloud touches exactly two things: the
    centres ``x, y, z`` multiply by the factor, and the log-sigmas
    ``scale_0..2`` gain ``log(factor)``. Rotations, opacities and spherical
    harmonics are scale-invariant. This rewrites those five columns in the raw
    float matrix and copies the header verbatim, so every other byte survives —
    going through :func:`load_ply` and re-encoding would not, since that
    exponentiates the sigmas and sigmoids the opacities.

    The write is atomic: concurrent evals share an asset cache and must never
    read a half-written PLY.
    """
    src, dst = Path(src), Path(dst)
    if not factor > 0:
        raise ValueError(f"scale_ply: factor must be positive, got {factor}")
    with open(src, "rb") as f:
        n_verts, prop_names, _types = _parse_ply_header(f)
        header_len = f.tell()
        raw = np.frombuffer(
            f.read(n_verts * len(prop_names) * 4), dtype="<f4"
        ).reshape(n_verts, len(prop_names)).copy()
    col = {name: i for i, name in enumerate(prop_names)}
    needed = ("x", "y", "z", "scale_0", "scale_1", "scale_2")
    missing = [c for c in needed if c not in col]
    if missing:
        raise ValueError(
            f"scale_ply: {src} is not a 3DGS PLY (no {missing} properties)")
    raw[:, [col["x"], col["y"], col["z"]]] *= np.float32(factor)
    raw[:, [col[f"scale_{i}"] for i in range(3)]] += np.float32(math.log(factor))
    with open(src, "rb") as f:
        header = f.read(header_len)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(raw.tobytes())
    tmp.replace(dst)
    return dst


def _parse_ply_header(f) -> tuple:
    """Return (n_verts, prop_names, prop_types). Raises on non-float32 or non-binary-LE."""
    n_verts = 0
    props, types = [], []
    fmt_line = ""
    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line.startswith("format"):
            fmt_line = line
        elif line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            types.append(parts[1])
            props.append(parts[2])
        elif line == "end_header":
            break
    if "binary_little_endian" not in fmt_line:
        raise ValueError(
            f"load_ply only supports binary_little_endian PLY; got format: {fmt_line!r}"
        )
    bad = [(p, t) for p, t in zip(props, types) if t not in _PLY_FLOAT_TYPES]
    if bad:
        raise ValueError(
            f"load_ply only supports float/float32 properties; found non-float: {bad}"
        )
    return n_verts, props, types
