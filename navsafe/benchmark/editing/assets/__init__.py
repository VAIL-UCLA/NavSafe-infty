# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Assets: fetch the object a recipe names, as a 3DGS PLY the server can load."""

from navsafe.benchmark.editing.assets.acquire import (
    AcquireError,
    acquire_asset,
    acquire_missing,
    plan_acquisition,
)
from navsafe.benchmark.editing.assets.compose import (
    ComposeError,
    compose_assets,
    compose_from_registry,
)
from navsafe.benchmark.editing.assets.ply_io import (
    PlyError,
    bounding_dims,
    concat_gaussians,
    ply_sha256,
    read_3dgs_ply,
    transform_gaussians,
    write_3dgs_ply,
)
from navsafe.benchmark.editing.assets.registry import (
    DEFAULT_REGISTRY,
    AssetEntry,
    AssetError,
    AssetRegistry,
    ResolvedAsset,
)

__all__ = [
    "DEFAULT_REGISTRY",
    "AcquireError",
    "AssetEntry",
    "AssetError",
    "AssetRegistry",
    "ComposeError",
    "PlyError",
    "ResolvedAsset",
    "acquire_asset",
    "acquire_missing",
    "bounding_dims",
    "compose_assets",
    "compose_from_registry",
    "concat_gaussians",
    "plan_acquisition",
    "ply_sha256",
    "read_3dgs_ply",
    "transform_gaussians",
    "write_3dgs_ply",
]
