# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The asset registry: what a ``registry_key`` in a recipe actually names.

A recipe pins an asset by content — source uid plus a sha256 — so that a
rebuild loads the same object and not merely one with the same name. The
registry is the lookup between the two: a stable key an author writes, and the
3DGS PLY on disk the NuRec server loads.

It deliberately lists assets that are **not on disk yet**. The nine leaves need
roughly ten times what has been converted so far, and "declared but missing" is
the honest state of most of the tree — a registry that only listed what exists
would make the gap invisible, which is the opposite of useful when the build
order is by asset cost.

Sources, and what each implies:

``urbanverse``     retrieval from the UrbanVerse mesh library, then mesh -> 3DGS.
                   The only source with fine-grained construction objects,
                   signs, animals and street furniture as distinct categories.
``assetharvester`` reconstruction from our own AV logs. Native 3DGS at scene
                   -matched fidelity, but only coarse classes.
``procedural``     built by the converter (cones, sign plates) — no download.
``composed``       two or more assets merged into one, e.g. bicycle + rider.
``host``           no file at all: keep the host actor's baked gaussians. The
                   cheapest and most reliable option, and why ``relocate`` with
                   ``keep_appearance`` is preferred wherever a suitable actor
                   exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from navsafe.benchmark.editing.assets.ply_io import ply_sha256
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path(__file__).with_name("registry.yaml")

SOURCES = ("urbanverse", "assetharvester", "procedural", "composed", "host")


class AssetError(NexusSimError, ValueError):
    """The registry cannot supply the asset a recipe asks for."""


@dataclass
class AssetEntry:
    """One registry row, as declared — not necessarily as present on disk."""

    key: str
    family: str = ""
    # A member of a heterogeneous family (`animal` spans squirrel to deer), so
    # this is what picks the calibration target out of `species_dims:`. It was
    # declared in the YAML and silently dropped here, which meant the table
    # could only be reached by typing `--species horse` from memory.
    species: str = ""
    source: str = "urbanverse"
    uid: str = ""
    ply: str = ""
    dims: Optional[List[float]] = None
    track_type: str = "VEHICLE"
    # The class the RENDER server loads this asset as, which is not always the
    # class it is scored as: a car2sim recon permits its actor layer only for a
    # PEDESTRIAN-class track, so a VEHICLE-class insert is rejected outright.
    # Empty means "same as track_type". See render/nurec_grpc.py.
    render_class: str = ""
    #: Directory NAME (never a path) of the gait bank that makes this asset
    #: walk instead of slide -- resolved against ``cfg.GAIT_BANK`` so the
    #: registry ships no data paths. Declaring it here rather than in the
    #: recipe is what stops a re-bake from silently dropping the gait: the
    #: field used to reach a recipe only by hand-editing a file whose header
    #: says FROZEN, do not hand-edit, so it survived exactly until the next
    #: `navsafe bake`. Empty means the asset is deliberately static.
    gait_bank: str = ""
    leaves: List[str] = field(default_factory=list)
    compose: Dict[str, Any] = field(default_factory=dict)
    procedural: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def path(self) -> Optional[Path]:
        if not self.ply:
            return None
        path = Path(self.ply)
        if path.is_absolute():
            return path
        from navsafe.benchmark import config as cfg
        if cfg.ASSET_BANK is None:
            return None
        parts = path.parts[1:] if path.parts[0] == "asset" else path.parts
        return cfg.ASSET_BANK.joinpath(*parts)

    @property
    def present(self) -> bool:
        path = self.path
        return bool(path and path.is_file())

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "AssetEntry":
        source = str(d.get("source", "urbanverse"))
        if source not in SOURCES:
            raise AssetError(f"asset {key!r}: unknown source {source!r}; expected {list(SOURCES)}")
        dims = d.get("dims")
        if dims is not None:
            dims = [float(v) for v in dims]
            if len(dims) != 3:
                raise AssetError(f"asset {key!r}: dims must be [length, width, height]")
        return cls(
            key=key,
            family=str(d.get("family", "")),
            species=str(d.get("species", "")),
            source=source,
            uid=str(d.get("uid", "")),
            ply=str(d.get("ply", "")),
            dims=dims,
            track_type=str(d.get("track_type", "VEHICLE")),
            render_class=str(d.get("render_class", "")),
            leaves=[str(x) for x in (d.get("leaves") or [])],
            compose=dict(d.get("compose") or {}),
            gait_bank=str(d.get("gait_bank", "")),
            procedural=dict(d.get("procedural") or {}),
            notes=str(d.get("notes", "")),
        )


@dataclass
class ResolvedAsset:
    """A registry entry whose file exists, hashed."""

    key: str
    entry: AssetEntry
    ply: Path
    sha256: str

    def to_recipe_asset(self) -> Dict[str, Any]:
        """The ``asset:`` block a recipe pins: key, uid, content hash, dims."""
        out: Dict[str, Any] = {"registry_key": self.key, "sha256": self.sha256}
        if self.entry.uid:
            out["uid"] = self.entry.uid
        if self.entry.dims is not None:
            out["dims"] = list(self.entry.dims)
        return out


class AssetRegistry:
    """Loaded registry, keyed by ``registry_key``."""

    def __init__(
        self,
        entries: Dict[str, AssetEntry],
        *,
        path: Optional[Path] = None,
        families: Optional[Dict[str, List[float]]] = None,
        species: Optional[Dict[str, List[float]]] = None,
    ):
        self.entries = entries
        self.path = path
        # family -> canonical [length, width, height]; the calibration target.
        self.families: Dict[str, List[float]] = dict(families or {})
        # species -> canonical [length, width, height]; the target for a member
        # of a heterogeneous family (see `families` above).
        self.species: Dict[str, List[float]] = dict(species or {})

    # -- loading ---------------------------------------------------------
    @classmethod
    def load(cls, path: "str | Path | None" = None) -> "AssetRegistry":
        path = Path(path or DEFAULT_REGISTRY)
        raw = yaml.safe_load(path.read_text()) or {}
        assets = raw.get("assets") or {}
        if not isinstance(assets, dict):
            raise AssetError(f"{path}: `assets` must be a mapping of key -> entry")
        families: Dict[str, List[float]] = {}
        for family, dims in (raw.get("families") or {}).items():
            dims = [float(v) for v in dims]
            if len(dims) != 3 or any(d <= 0 for d in dims):
                raise AssetError(
                    f"{path}: families.{family} must be three positive metres, got {dims}"
                )
            families[str(family)] = dims
        species: Dict[str, List[float]] = {}
        for name, dims in (raw.get("species_dims") or {}).items():
            dims = [float(v) for v in dims]
            if len(dims) != 3 or any(d <= 0 for d in dims):
                raise AssetError(
                    f"{path}: species_dims.{name} must be three positive metres, got {dims}")
            species[str(name)] = dims
        return cls(
            {str(key): AssetEntry.from_dict(str(key), spec or {}) for key, spec in assets.items()},
            path=path,
            families=families,
            species=species,
        )

    def species_dims(self, species: str) -> List[float]:
        """Canonical [length, width, height] for a species.

        Raises:
            AssetError: unknown species. Sizing a harvested animal by guesswork
                is how a hippo ends up 0.59 m long — add it to the table instead.
        """
        try:
            return list(self.species[str(species)])
        except KeyError:
            raise AssetError(
                f"species {species!r} has no canonical dims in {self.path}. Known: "
                f"{sorted(self.species)}. Add it to `species_dims:` rather than passing "
                f"--target-dims from memory."
            ) from None

    def family_dims(self, family: str) -> List[float]:
        """Canonical [length, width, height] for a family.

        Raises:
            AssetError: the family has no canonical dims — either it is not in
                the table or it is heterogeneous on purpose (animal, debris,
                emergency_vehicle …). Calibrating a member then needs an
                explicit target.
        """
        try:
            return list(self.families[str(family)])
        except KeyError:
            raise AssetError(
                f"family {family!r} has no canonical dims in {self.path}. Heterogeneous "
                f"families are absent on purpose — pass --target-dims L W H instead, or add "
                f"the family to the `families:` table if one size genuinely fits all members. "
                f"Known families: {sorted(self.families)}"
            ) from None

    # -- queries ---------------------------------------------------------
    def __contains__(self, key: object) -> bool:
        return str(key) in self.entries

    def get(self, key: str) -> AssetEntry:
        try:
            return self.entries[str(key)]
        except KeyError:
            raise AssetError(
                f"no asset {key!r} in {self.path}. Known keys in that family: "
                f"{sorted(k for k in self.entries if str(key).split('_')[0] in k) or sorted(self.entries)[:8]}"
            ) from None

    def resolve_gait_bank(self, key: str) -> Optional["tuple[Path, str]"]:
        """``(bank directory, sha256 of its bank.json)`` for an asset that walks.

        Returns None for an asset that declares no bank -- a static insert is a
        legitimate choice, not an omission.

        Raises:
            AssetError: a bank is declared but not on disk. Falling back to the
                static asset would be silent: a gait stuck on one phase renders
                exactly like the asset it replaced, so the failure has no visual
                tell beyond "the legs never moved".
        """
        import hashlib

        from navsafe.benchmark import config as cfg

        name = self.get(key).gait_bank
        if not name:
            return None
        if not cfg.GAIT_BANK:
            raise AssetError(
                f"asset {key!r} declares gait_bank {name!r}, but "
                f"NAVSAFE_GAIT_BANK is unset. It no longer defaults to this "
                f"cluster's own library, because that default made a missing "
                f"setting invisible here and fatal everywhere else. Set it to "
                f"the directory the pose banks live in."
            )
        root = Path(cfg.GAIT_BANK) / name
        manifest = root / "bank.json"
        if not manifest.is_file():
            raise AssetError(
                f"asset {key!r} declares gait_bank {name!r} but {manifest} is not "
                f"there. Bake it with `navsafe assets animate {key}`, point "
                f"NAVSAFE_GAIT_BANK at the banks (currently {cfg.GAIT_BANK}), or "
                f"clear `gait_bank:` to say the asset is meant to be static."
            )
        return root, hashlib.sha256(manifest.read_bytes()).hexdigest()

    def resolve(self, key: str) -> ResolvedAsset:
        """Look the key up and hash its file.

        Raises:
            AssetError: the entry is declared but its PLY is not on disk. That
                is the normal state for most of the tree — acquire it first.
        """
        entry = self.get(key)
        if entry.source == "host":
            raise AssetError(
                f"asset {key!r} is source 'host': it has no file, because the actor keeps its "
                f"baked gaussians. Use op 'relocate' with keep_appearance instead of naming a "
                f"PLY."
            )
        if not entry.present:
            raise AssetError(
                f"asset {key!r} is declared ({entry.source}) but its PLY is not on disk at "
                f"{entry.ply or '<no path>'}. Acquire it first: "
                f"`python -m navsafe.benchmark.editing.cli assets acquire {key}`."
            )
        return ResolvedAsset(key=key, entry=entry, ply=entry.path, sha256=ply_sha256(entry.path))

    def status(self) -> Dict[str, Any]:
        """Per-source and per-leaf coverage — what is built and what is not."""
        by_source: Dict[str, Dict[str, int]] = {}
        by_leaf: Dict[str, Dict[str, int]] = {}
        for entry in self.entries.values():
            bucket = by_source.setdefault(entry.source, {"present": 0, "declared": 0})
            bucket["declared"] += 1
            bucket["present"] += int(entry.present or entry.source == "host")
            for leaf in entry.leaves:
                leaf_bucket = by_leaf.setdefault(leaf, {"present": 0, "declared": 0})
                leaf_bucket["declared"] += 1
                leaf_bucket["present"] += int(entry.present or entry.source == "host")
        return {
            "registry": str(self.path),
            "total": len(self.entries),
            "present": sum(1 for e in self.entries.values() if e.present or e.source == "host"),
            "by_source": by_source,
            "by_leaf": dict(sorted(by_leaf.items())),
        }

    def missing(self) -> List[AssetEntry]:
        return [
            e for e in self.entries.values() if e.source != "host" and not e.present
        ]
