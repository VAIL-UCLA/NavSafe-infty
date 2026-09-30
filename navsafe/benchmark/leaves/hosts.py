# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Which converted hosts exist, and what an ``--scenario-id`` refers to.

``navsafe mine`` prints a scenario id; ``navsafe bake`` takes one. For that to
be two commands rather than two commands plus a path lookup, something has to
turn an id into the Arrow root, the camera rig, and how many 5 s recon windows
back it. That is this module, and nothing else needs to know the layout.

Both converted families put the scenario's Arrow beside its recon under a root
whose directory name IS the scene id -- the corpus layout defined in
``navsafe/config.py``::

    <NAVHARD_CORPUS>/<scene_id>/arrow      # 5 s navhard hosts, waymo rig
    <CORPUS>/<scene_id>/arrow              # 20 s stitched hosts, navsim rig

The rig belongs here rather than in a leaf: ``cam_height`` is a property of how
the clip was reconstructed, not of what is being tested on it. Specs used to
carry ``cam_height: waymo`` while ``bake-mined`` defaulted to ``navsim``, and
whichever ran last decided — a leaf declaration cannot answer that question,
because the same leaf is built on hosts from both families.

Discovery is one glob per pattern plus a listdir. Nothing here reads Arrow, so
a full index costs well under a second; it is cached anyway because `bake`
resolves exactly one id and should not pay even that.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from navsafe.benchmark import config as cfg
from navsafe.errors import NexusSimError

logger = logging.getLogger(__name__)

# Each entry is (glob for the arrow root, family name, camera rig).
DEFAULT_ROOTS = (
    (str(cfg.arrow_dir("*", cfg.NAVHARD_CORPUS)), "navhard421", "waymo"),
    (str(cfg.arrow_dir("*_20s")), "navsafe_20s", "navsim"),
)
INDEX_CACHE = cfg.HOST_INDEX


class HostError(NexusSimError, RuntimeError):
    """No host, or too many, for what the caller asked."""


@dataclass
class Host:
    """One converted scenario: where its Arrow is and how it was reconstructed."""

    scene_id: str
    data_root: str
    family: str
    cam_height: str
    recon_root: str = ""
    windows: List[str] = field(default_factory=list)

    @property
    def token(self) -> str:
        """The nuPlan token, i.e. the scene id without its conversion suffix."""
        name = self.scene_id
        for suffix in ("_20s", "h1", "h2"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    def describe(self) -> str:
        win = ",".join(self.windows) if self.windows else "-"
        return (f"{self.scene_id:<24} {self.family:<12} rig={self.cam_height:<7} "
                f"recon={win:<12} {self.data_root}")


def _roots_from_env(roots_glob: Optional[List[str]] = None):
    """The (glob, family, rig) triples to scan, honouring overrides.

    ``--roots-glob`` and ``NAVSAFE_HOST_ROOTS`` take ``glob[:family[:rig]]`` so
    a checkout off this cluster is not stuck with hard-coded paths.
    """
    raw = list(roots_glob or [])
    if not raw and os.environ.get("NAVSAFE_HOST_ROOTS"):
        raw = [p for p in os.environ["NAVSAFE_HOST_ROOTS"].split(",") if p.strip()]
    if not raw:
        return list(DEFAULT_ROOTS)
    out = []
    for item in raw:
        parts = item.split(":")
        pattern = parts[0]
        family = parts[1] if len(parts) > 1 else "custom"
        rig = parts[2] if len(parts) > 2 else "navsim"
        out.append((pattern, family, rig))
    return out


def _recon_windows(recon_root: Path, scene_id: str) -> List[str]:
    """Which 5 s windows of this scenario have a trained recon on disk.

    A 20 s host is four reconstructions; a render needs the ones it will hand
    off to. `mine` reports this so a scenario that qualifies but cannot yet be
    served is visibly distinct from one that can.
    """
    def _usdz(clip: str) -> bool:
        return cfg.recon_usdz(clip, recon_root) is not None

    # A 20 s host is four 5 s recons named <token>s1..s4, siblings of its Arrow.
    found = [w for w in ("s1", "s2", "s3", "s4")
             if _usdz(f"{scene_id.removesuffix('_20s')}{w}")]
    # A 5 s navhard host is a single recon under its own scene id.
    if not found and _usdz(scene_id):
        found = ["s1"]
    return found


def find_hosts(roots_glob: Optional[List[str]] = None, *, refresh: bool = False) -> Dict[str, Host]:
    """Every converted host, keyed by scene id."""
    if not refresh and roots_glob is None and INDEX_CACHE.is_file():
        try:
            cached = json.loads(INDEX_CACHE.read_text())
            return {k: Host(**v) for k, v in cached.get("hosts", {}).items()}
        except Exception as exc:  # noqa: BLE001 — a stale cache must never be fatal
            logger.debug("host index cache unreadable (%s); rescanning", exc)

    hosts: Dict[str, Host] = {}
    for pattern, family, rig in _roots_from_env(roots_glob):
        for root in sorted(_glob.glob(pattern)):
            arrow = Path(root)
            scene_id = arrow.parent.name
            recon_root = arrow.parent.parent
            hosts[scene_id] = Host(
                scene_id=scene_id,
                data_root=str(arrow),
                family=family,
                cam_height=rig,
                recon_root=str(recon_root),
                windows=_recon_windows(recon_root, scene_id),
            )
    if roots_glob is None:
        try:
            INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
            INDEX_CACHE.write_text(json.dumps(
                {"hosts": {k: asdict(v) for k, v in hosts.items()}}, indent=1))
        except Exception as exc:  # noqa: BLE001 — read-only checkout is fine
            logger.debug("could not cache the host index (%s)", exc)
    return hosts


def resolve_host(scenario_id: str, roots_glob: Optional[List[str]] = None,
                 *, refresh: bool = False) -> Host:
    """One scenario id -> its host.

    Accepts an exact scene id, or a nuPlan token that prefixes exactly one —
    so the id `mine` prints pastes straight into `bake` whether it came from
    the raw-map tier (which knows tokens) or the host tier (which knows scene
    ids). An ambiguous prefix names the candidates rather than guessing.
    """
    hosts = find_hosts(roots_glob, refresh=refresh)
    if scenario_id in hosts:
        return hosts[scenario_id]
    matches = sorted(k for k in hosts if k.startswith(scenario_id))
    if len(matches) == 1:
        return hosts[matches[0]]
    if matches:
        raise HostError(
            f"scenario id {scenario_id!r} matches {len(matches)} hosts: {matches}. "
            f"Pass the full scene id.")
    raise HostError(
        f"no converted host for {scenario_id!r}. {len(hosts)} hosts are indexed; the "
        f"scenario has to be converted to Arrow before it can be baked "
        f"(navsafe/benchmark/eval/make_arrow.sh). If it IS converted, its root is "
        f"outside the indexed patterns — pass --roots-glob or set NAVSAFE_HOST_ROOTS.")


__all__ = ["DEFAULT_ROOTS", "Host", "HostError", "find_hosts", "resolve_host"]
