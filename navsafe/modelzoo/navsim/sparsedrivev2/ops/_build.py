# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Lazily JIT-build the two deformable-aggregation CUDA extensions.

``deformable_aggregation.py`` imports these two names at module import time,
which upstream resolves against ``.so`` files produced by a manual
``setup.py build_ext``. Building at import would make *importing the adapter*
cost a minute and require a GPU box, so each extension is a lazy proxy: the
compile happens on the first attribute access, i.e. the first forward pass.

``torch.utils.cpp_extension.load`` caches by name under ``TORCH_EXTENSIONS_DIR``
(default ``~/.cache/torch_extensions``), so the cost is paid once per
torch/CUDA combination, not once per run. Set ``TORCH_EXTENSIONS_DIR`` if
``$HOME`` is full or read-only — on a shared box that is the usual failure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent / "src"

# nvcc needs the target arch. Left unset, torch infers it from the visible
# device, which is right on a single-GPU-type box and wrong when the build host
# and the run host differ — hence the override rather than a hardcoded arch.
_ARCH_ENV = "TORCH_CUDA_ARCH_LIST"


class _LazyExtension:
    """Compiles on first attribute access, then forwards everything to it."""

    def __init__(self, name: str, sources: list[str]) -> None:
        self._name = name
        self._sources = sources
        self._module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            from torch.utils.cpp_extension import load

            if not os.environ.get(_ARCH_ENV):
                # Ampere/Ada/Hopper covers every card this repo is run on; an
                # explicit list also stops nvcc rebuilding per-device.
                os.environ[_ARCH_ENV] = "8.0;8.6;8.9;9.0"
            self._module = load(
                name=self._name,
                sources=[str(_SRC / s) for s in self._sources],
                extra_cuda_cflags=[
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
                verbose=False,
            )
        return self._module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)


deformable_aggregation_ext = _LazyExtension(
    "navsafe_deformable_aggregation_ext",
    ["deformable_aggregation.cpp", "deformable_aggregation_cuda.cu"],
)

deformable_aggregation_with_depth_ext = _LazyExtension(
    "navsafe_deformable_aggregation_with_depth_ext",
    ["deformable_aggregation_with_depth.cpp",
     "deformable_aggregation_with_depth_cuda.cu"],
)
