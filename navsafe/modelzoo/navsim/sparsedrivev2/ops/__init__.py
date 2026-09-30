"""Deformable-aggregation CUDA op for SparseDriveV2.

The upstream repo ships this as a ``python setup.py build_ext`` step whose
``.so`` then has to sit next to the sources. That does not survive a pip
install of navsafe, and a prebuilt binary is tied to one torch/CUDA pair, so
the extension is built **on first use** with ``torch.utils.cpp_extension.load``
and cached in ``TORCH_EXTENSIONS_DIR`` (``~/.cache/torch_extensions`` by
default). The first call costs ~1 min; every later process reuses the cache.

Verified 2026-08-12 against torch 2.10.0+cu128 / nvcc 12.8 on an L40S: builds
clean and matches the upstream forward signature (b, cam, feat, C).
"""

from .deformable_aggregation import (  # noqa: F401
    deformable_aggregation_func,
    deformable_format,
    feature_maps_format,
)

__all__ = [
    "deformable_aggregation_func",
    "deformable_format",
    "feature_maps_format",
]
