# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Recipe file format — the artifact the scene-editing pipeline produces.

A recipe is not a scenario; it is a specification precise enough to rebuild an
identical scenario later. Injected actors are REACTIVE — a car follows an IDM
policy, a pedestrian a social-force one — so their trajectories depend on what
the ego does and cannot be known when the recipe is written. What the file
therefore pins is the **initial condition and the controller**, not the path.

That is a deliberate reversal. The format used to bake per-frame absolute
poses, on the grounds that a non-reactive actor's whole trajectory is knowable
in advance and freezing it made a scenario independent of the policy under
test. It also made every actor drive through a braking ego and stand still
while the world moved around it. Reactivity buys realism at the cost of that
guarantee, and the cost is real: two runs agree only if the policy code, its
parameters and the ego's behaviour all agree.

What a recipe pins:

``identity``   which asset (source uid + sha256), which host scene, which
               ``world_version`` and recon artifact.
``initial``    each actor's ``spawn`` pose, in **ego frame-0** coordinates,
               plus the route polyline it was measured against.
``policy``     which controller drives it and with what parameters —
               including any geometry the controller needs resolved against
               this host (an IDM lane chain, a social-force goal), because a
               name like ``opposing_lane_chain`` is a query whose answer must
               not be allowed to drift.
``semantics``  which leaf this populates, and the reviewer checklist that
               decides whether it does.
``integrity``  a sha256 over that specification.

The pipeline does not judge whether a built scenario is correct, plausible or
faithful to its leaf — a human does, from a rendered episode. The one thing
code verifies is file integrity, so a corrupted or hand-edited recipe fails
loudly rather than quietly replaying something else.

Two frame conventions live in one file and are easy to confuse:

* **``spawn`` and the route polyline are in ego frame-0 world coordinates**.
* **``arc`` parameters in ``authored`` are measured from the hand-off**
  (``frames.after_frame``) as arc 0.

``authored`` is documentation — the intent the initial condition was solved
from, diffable and re-authorable, but never re-executed at replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml
from navsafe.errors import NexusSimError


class RecipeError(NexusSimError, ValueError):
    """A recipe is malformed, inconsistent, or fails its integrity check."""


# --------------------------------------------------------------------------
# Canonical array encoding
# --------------------------------------------------------------------------
#
# A checksum is only meaningful if the bytes it covers are reproducible from
# the file. Two things guarantee that here:
#
#   1. Every array has ONE canonical dtype and shape (below). Hashing the
#      float64 a YAML parser happens to produce would make the digest depend
#      on the parser, not on the trajectory.
#   2. Values are quantised to ``_QUANTUM`` before they are written, so the
#      short decimal in the file round-trips text -> float64 -> float32
#      exactly. Without that, a full-precision float32 needs 17 digits and the
#      file becomes unreadable for the human who has to review it.
#
# name -> (dtype, ndim, trailing dimension or None)
_ARRAY_SPEC: Dict[str, tuple] = {
    "position": (np.float32, 2, 3),
    "heading": (np.float32, 1, None),
    "velocity": (np.float32, 2, 2),
    "valid": (np.bool_, 1, None),
}

# 1e-6 m / 1e-6 rad. Far below any physical or perceptual tolerance, and small
# enough that quantisation never moves an actor by a measurable amount.
_QUANTUM = 6

# The four edit ops. All four are render/sim-consistent: the render mirrors the
# sim state, with no render-only or score-only objects.
EDIT_OPS = ("insert", "replace", "relocate", "remove")

# The authoring templates. Defined HERE and imported by ``trajectory/templates``
# rather than the other way round: ``trajectory/bake`` imports this module, so a
# import pointing the other way closes a cycle through ``trajectory/__init__``.
# One definition either way — the two files used to keep separate copies, and
# this one still listed ``lateral_schedule`` and ``group`` long after both were
# removed, so a recipe naming either validated against a template nothing builds.
TEMPLATES = ("static", "dynamic", "dart_out")

# Poses are ALWAYS in the host's ego frame-0 coordinates, never absolute UTM:
# float32 storing a 4.69e6 m UTM easting has a 0.5 m ULP, which is what made
# ego trajectories saw-tooth and lane polylines jag.
FRAME_CONVENTION = "ego_frame0"

# How the scenario came to be. Both kinds end at a recipe, because a recipe is
# what evaluation consumes — the point of the format is that a benchmark user
# never needs to know which path produced a scene.
#
# ``constructed``  actors were inserted / re-tasked; the edit IS the event, so
#                  the recipe carries one entry per actor — its spawn, the
#                  controller that drives it, and the asset it is made of.
# ``mined``        the logged road is the event (an illegal turn, a merge, a
#                  two-way street a policy may cross). Nothing is edited, so
#                  the recipe carries NO actors — what it pins is the host, the
#                  window, the ego convention and the SELECTION EVIDENCE that
#                  made this clip qualify, so the choice is auditable and
#                  re-runnable rather than remembered.
PROVENANCES = ("constructed", "mined")

# Where a rebased spec keeps the asset path it was FROZEN with.
#
# ``replay_spec`` hashes ``nurec_asset_id``, and ``NAVSAFE_ASSET_BANK`` rewrites
# it to the reader's own bank — so the digest recomputed at the last gate
# (``spawn_reactive_actor``) would disagree with the recorded one for any
# relocated bank. The frozen path rides along under this key so that gate can
# hash what was actually frozen; ``spec_digest`` excludes it, and the gate drops
# it before the spec reaches the simulator.
FROZEN_ASSET_ID_KEY = "_frozen_nurec_asset_id"

# Spec keys that name WHERE something lives rather than WHAT it is, and so vary
# legitimately from one machine to the next. They stay in the spec (the
# simulator opens them) but out of the digest, which pins content instead:
# ``asset_sha256`` for the asset, ``pose_bank_sha256`` for the gait bank.
#
# ``nurec_asset_id`` is the exception that proves the rule — it is still hashed,
# with the frozen value carried under FROZEN_ASSET_ID_KEY, because recipes were
# frozen that way. A pose bank had no such history, so its path was excluded
# from the start rather than given a second frozen key.
_UNHASHED_PATH_KEYS = ("nurec_pose_bank",)


def spec_digest(spec: Dict[str, Any]) -> str:
    """sha256 over one actor's replay spec, canonically serialised.

    The digest target moved here from the baked arrays: an actor's path is
    produced at run time by a policy, so there is no trajectory to hash. What
    determines the scenario — and what this covers — is the spawn, the policy
    with its resolved geometry, and the asset's own content hash.
    """
    payload = {k: v for k, v in spec.items()
               if k not in ("sha256", FROZEN_ASSET_ID_KEY) + _UNHASHED_PATH_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=float).encode()
    return hashlib.sha256(blob).hexdigest()


def canonical_array(name: str, data: Any, *, length: Optional[int] = None) -> np.ndarray:
    """Return ``data`` as the canonical dtype/shape for state array ``name``.

    Raises:
        RecipeError: unknown array name, wrong rank, wrong trailing dimension,
            non-finite values, or a length that disagrees with ``length``.
    """
    if name not in _ARRAY_SPEC:
        raise RecipeError(f"unknown state array {name!r}; expected one of {list(_ARRAY_SPEC)}")
    dtype, ndim, trailing = _ARRAY_SPEC[name]
    arr = np.asarray(data)
    if arr.size == 0:
        raise RecipeError(f"state array {name!r} is empty")
    if arr.ndim != ndim:
        raise RecipeError(f"state array {name!r} must be {ndim}-D, got shape {arr.shape}")
    if trailing is not None and arr.shape[1] != trailing:
        raise RecipeError(
            f"state array {name!r} must have trailing dimension {trailing}, got {arr.shape}"
        )
    if length is not None and arr.shape[0] != length:
        raise RecipeError(
            f"state array {name!r} has {arr.shape[0]} frames, expected {length}"
        )
    if dtype is np.bool_:
        return np.ascontiguousarray(arr.astype(np.bool_))
    out = np.ascontiguousarray(arr.astype(np.float64))
    if not np.all(np.isfinite(out)):
        raise RecipeError(f"state array {name!r} contains non-finite values")
    return np.ascontiguousarray(out.astype(dtype))


def quantize_array(name: str, data: Any) -> np.ndarray:
    """Round a freshly baked array to the storage quantum, for writing.

    Returns float64 deliberately: the digest is taken over the float32 view
    (``canonical_array``), but the *file* stores this rounded float64, whose
    ``repr`` is a short decimal a reviewer can read. Writing the float32 value
    instead would put ``1000.1234130859375`` in the file — exact, unreadable,
    and no more reproducible, since text -> float64 -> float32 is deterministic
    either way.
    """
    dtype = _ARRAY_SPEC[name][0]
    if dtype is np.bool_:
        return canonical_array(name, data)
    canonical_array(name, data)  # shape / finiteness gate
    return np.ascontiguousarray(np.round(np.asarray(data, np.float64), _QUANTUM))


def _plain(value: Any) -> Any:
    """numpy -> plain Python, deeply, so ``yaml.safe_dump`` can represent it."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------
# Recipe structure
# --------------------------------------------------------------------------


@dataclass
class HostSpec:
    """Which reconstructed world this recipe is measured against."""

    scene: str
    world_version: str
    artifact: str = ""
    temporal_extent_s: Optional[float] = None

    def to_dict(self) -> dict:
        out = {"scene": self.scene, "world_version": self.world_version}
        if self.artifact:
            out["artifact"] = self.artifact
        if self.temporal_extent_s is not None:
            out["temporal_extent_s"] = float(self.temporal_extent_s)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "HostSpec":
        if not d.get("scene"):
            raise RecipeError("host.scene is required — a recipe without a host rebuilds nothing")
        if not d.get("world_version"):
            raise RecipeError(
                "host.world_version is required — the same scene reconstructed with a "
                "different recipe is a different world"
            )
        return cls(
            scene=str(d["scene"]),
            world_version=str(d["world_version"]),
            artifact=str(d.get("artifact", "")),
            temporal_extent_s=(
                float(d["temporal_extent_s"]) if d.get("temporal_extent_s") is not None else None
            ),
        )


@dataclass
class EgoSpec:
    """The ego conventions the baked poses assume."""

    replay_frames: int = 0
    cam_height: str = ""
    z_to_ground: float = 0.0

    def to_dict(self) -> dict:
        out: dict = {"replay_frames": int(self.replay_frames)}
        if self.cam_height:
            out["cam_height"] = self.cam_height
        out["z_to_ground"] = float(self.z_to_ground)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "EgoSpec":
        return cls(
            replay_frames=int(d.get("replay_frames", 0)),
            cam_height=str(d.get("cam_height", "")),
            z_to_ground=float(d.get("z_to_ground", 0.0)),
        )


@dataclass
class Frames:
    """The host's own frame grid.

    ``T`` and ``dt_s`` come from the host's frame timestamps, never from a
    default: every speed-derived quantity — a crossing's frame span, the baked
    velocity — scales with ``dt_s``, so a host whose rate disagrees must be
    refused rather than silently replayed at the wrong speed.
    """

    T: int
    dt_s: float
    after_frame: int = 0
    timestamps_us: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {"T": int(self.T), "dt_s": float(self.dt_s), "after_frame": int(self.after_frame)}
        if self.timestamps_us:
            out["timestamps_us"] = [int(t) for t in self.timestamps_us]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Frames":
        try:
            T = int(d["T"])
            dt_s = float(d["dt_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecipeError(f"frames must carry integer T and float dt_s: {exc}") from exc
        if T < 2:
            raise RecipeError(f"frames.T must be >= 2, got {T}")
        if not (0.0 < dt_s < 10.0):
            raise RecipeError(f"frames.dt_s must be a plausible frame interval in seconds, got {dt_s}")
        after = int(d.get("after_frame", 0))
        if not (0 <= after < T):
            raise RecipeError(f"frames.after_frame must lie in [0, T), got {after} with T={T}")
        ts = [int(t) for t in (d.get("timestamps_us") or [])]
        if ts and len(ts) != T:
            raise RecipeError(f"frames.timestamps_us has {len(ts)} entries, expected T={T}")
        return cls(T=T, dt_s=dt_s, after_frame=after, timestamps_us=ts)


@dataclass
class AssetRef:
    """Which asset, pinned by content."""

    registry_key: str = ""
    uid: str = ""
    sha256: str = ""
    dims: Optional[List[float]] = None  # (length, width, height) metres

    def to_dict(self) -> dict:
        out: dict = {}
        for key in ("registry_key", "uid", "sha256"):
            value = getattr(self, key)
            if value:
                out[key] = value
        if self.dims is not None:
            out["dims"] = [float(v) for v in self.dims]
        return out

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AssetRef":
        d = d or {}
        dims = d.get("dims")
        if dims is not None:
            dims = [float(v) for v in dims]
            if len(dims) != 3:
                raise RecipeError(f"asset.dims must be [length, width, height], got {dims}")
        return cls(
            registry_key=str(d.get("registry_key", "")),
            uid=str(d.get("uid", "")),
            sha256=str(d.get("sha256", "")),
            dims=dims,
        )


@dataclass
class ActorRecipe:
    """One edited actor: what it is, where it starts, and what drives it."""

    name: str
    op: str
    asset: AssetRef = field(default_factory=AssetRef)
    source_track_id: str = ""
    keep_appearance: bool = False
    track_type: str = "VEHICLE"
    semantic_class: str = ""
    nurec_asset_id: str = ""
    #: A directory of posed copies spanning one gait cycle, baked by
    #: ``editing/assets/animate.py``. Set on an actor that should WALK rather
    #: than slide: the renderer inserts each phase as its own server track and
    #: shows one per frame, clocked off distance travelled. The PATH is not
    #: hashed (a reader's bank lives somewhere else); :attr:`pose_bank_sha256`
    #: pins what it holds.
    nurec_pose_bank: str = ""
    #: sha256 of the bank's ``bank.json``, which itself records the motion, the
    #: fit, the phase count and each phase PLY's own hash. This is what the
    #: digest pins, so a relocated bank is fine and a bank baked from a
    #: different motion is not -- the same split the asset uses between
    #: ``nurec_asset_id`` (where) and ``asset.sha256`` (what).
    pose_bank_sha256: str = ""
    leaf_variety_slot: str = ""
    authored: Dict[str, Any] = field(default_factory=dict)
    #: Where the actor starts: ``{position, heading, velocity, length, width}``.
    spawn: Dict[str, Any] = field(default_factory=dict)
    #: How it then moves: ``{kind, ...}`` — see navsafe.traffic.navsafe.DRIVERS.
    #: ``kind`` is the one field the manager must have; the rest is the
    #: driver's own vocabulary (``path_polyline`` for idm, ``goal`` and
    #: ``trigger`` for social_force).
    policy: Dict[str, Any] = field(default_factory=dict)
    #: the digest read back off disk, checked by :meth:`verify`
    _recorded_sha: str = ""

    # -- validation ------------------------------------------------------
    def validate(self, frames: Frames) -> None:
        """Structural checks; :meth:`verify` covers integrity separately."""
        where = f"actor {self.name!r}"
        if self.op not in EDIT_OPS:
            raise RecipeError(f"{where}: unknown op {self.op!r}; expected one of {list(EDIT_OPS)}")
        if self.op in ("replace", "relocate", "remove") and not self.source_track_id:
            raise RecipeError(f"{where}: op {self.op!r} requires source_track_id")
        if self.op == "remove":
            if self.spawn or self.policy:
                raise RecipeError(
                    f"{where}: op 'remove' drops an agent and must carry no spawn or policy")
            return
        if "position" not in self.spawn:
            raise RecipeError(f"{where}: spawn needs a `position`")
        pos = np.asarray(self.spawn.get("position"), np.float64).reshape(-1)
        if pos.size < 2:
            raise RecipeError(f"{where}: spawn.position must be [x, y] or [x, y, z]")
        kind = str((self.policy or {}).get("kind", ""))
        if not kind:
            raise RecipeError(
                f"{where}: no `policy.kind`. An inserted actor's motion comes from a "
                f"policy at run time — this recipe records the initial condition and "
                f"which controller drives it, not a trajectory. Declare "
                f"`policy: {{kind: static}}` for something that genuinely never moves."
            )
        if self.keep_appearance and self.op not in ("relocate",):
            raise RecipeError(
                f"{where}: keep_appearance only applies to 'relocate' — there is no baked "
                f"appearance to keep for an inserted or replaced asset"
            )
        if not self.keep_appearance and self.asset.dims is None and self.op != "remove":
            raise RecipeError(
                f"{where}: a swapped-in or inserted asset needs asset.dims "
                f"[length, width, height]; only kept-appearance actors inherit the host's box"
            )
        template = (self.authored or {}).get("template")
        if template is not None and template not in TEMPLATES:
            raise RecipeError(
                f"{where}: authored.template {template!r} is not one of {list(TEMPLATES)}"
            )

    # -- integrity -------------------------------------------------------
    def replay_spec(self) -> dict:
        """The per-actor dict the env's edit tool consumes.

        Built here rather than in ``replay.py`` so there is exactly ONE shape
        to hash: the digest covers precisely what reaches the simulator, which
        is what makes the check at the last gate meaningful.
        """
        spec: Dict[str, Any] = {
            "name": self.name, "op": self.op, "track_type": self.track_type,
        }
        if self.source_track_id:
            spec["source_track_id"] = self.source_track_id
        if self.op == "relocate":
            spec["keep_appearance"] = bool(self.keep_appearance)
        if self.semantic_class:
            spec["semantic_class"] = self.semantic_class
        if self.asset.dims is not None:
            spec["dims"] = [float(v) for v in self.asset.dims]
        if self.asset.sha256:
            spec["asset_sha256"] = self.asset.sha256
        if self.nurec_asset_id:
            spec["nurec_asset_id"] = self.nurec_asset_id
        if self.nurec_pose_bank:
            spec["nurec_pose_bank"] = self.nurec_pose_bank
        if self.pose_bank_sha256:
            spec["pose_bank_sha256"] = self.pose_bank_sha256
        if self.asset.registry_key:
            spec["registry_key"] = self.asset.registry_key
        if self.op != "remove":
            spec["spawn"] = _plain(self.spawn)
            spec["policy"] = _plain(self.policy)
        return spec

    def digest(self) -> str:
        """A content hash over what this recipe actually determines.

        Under the baked design the hash covered the four per-frame arrays,
        because those *were* the scenario. They are gone: an actor's path is
        now produced at run time by a policy reacting to the ego, so there is
        no trajectory to hash and hashing one would be a lie.

        What remains determined by the file — and what therefore has to be
        pinned — is the initial condition, the controller and its parameters,
        and the asset's content hash. Same digest + same policy code + same
        ego behaviour ⇒ same rollout; a changed digest means someone edited
        the scenario, whatever the actor happened to do in any given episode.
        """
        return spec_digest(self.replay_spec())

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        out: dict = {}
        asset = self.asset.to_dict()
        if asset:
            out["asset"] = asset
        if self.leaf_variety_slot:
            out["leaf_variety_slot"] = self.leaf_variety_slot
        track: dict = {"type": self.track_type}
        if self.semantic_class:
            track["semantic_class"] = self.semantic_class
        out["track"] = track
        out["op"] = self.op
        if self.source_track_id:
            out["source_track_id"] = self.source_track_id
        if self.op == "relocate":
            out["keep_appearance"] = bool(self.keep_appearance)
        if self.nurec_asset_id:
            out["nurec_asset_id"] = self.nurec_asset_id
        if self.nurec_pose_bank:
            out["nurec_pose_bank"] = self.nurec_pose_bank
        if self.pose_bank_sha256:
            out["pose_bank_sha256"] = self.pose_bank_sha256
        if self.authored:
            out["authored"] = _plain(self.authored)
        if self.spawn:
            out["spawn"] = _plain(self.spawn)
        if self.policy:
            out["policy"] = _plain(self.policy)
        if self.op != "remove":
            out["sha256"] = self.digest()
        return out

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "ActorRecipe":
        track = d.get("track") or {}
        return cls(
            name=name,
            op=str(d.get("op", "")),
            asset=AssetRef.from_dict(d.get("asset")),
            source_track_id=str(d.get("source_track_id", "")),
            keep_appearance=bool(d.get("keep_appearance", False)),
            track_type=str(track.get("type", "VEHICLE")),
            semantic_class=str(track.get("semantic_class", "")),
            nurec_asset_id=str(d.get("nurec_asset_id", "")),
            nurec_pose_bank=str(d.get("nurec_pose_bank", "")),
            pose_bank_sha256=str(d.get("pose_bank_sha256", "")),
            leaf_variety_slot=str(d.get("leaf_variety_slot", "")),
            authored=dict(d.get("authored") or {}),
            spawn=dict(d.get("spawn") or {}),
            policy=dict(d.get("policy") or {}),
            _recorded_sha=str(d.get("sha256", "")),
        )

    def verify(self) -> None:
        """Refuse a hand-edited actor block."""
        recorded = str((self._recorded_sha or "")).strip()
        if self.op == "remove":
            return
        if not recorded:
            raise RecipeError(
                f"actor {self.name!r}: no sha256 recorded; refusing to replay an "
                f"unverified scenario spec")
        actual = self.digest()
        if actual != recorded:
            raise RecipeError(
                f"actor {self.name!r}: sha256 mismatch — recorded {recorded[:12]}…, "
                f"computed {actual[:12]}…. The spawn, the policy or the asset was "
                f"edited after freezing.")


@dataclass
class Recipe:
    """A frozen, self-contained description of one built scenario."""

    recipe_id: str
    leaf: str
    host: HostSpec
    frames: Frames
    ego: EgoSpec = field(default_factory=EgoSpec)
    scenario: str = ""
    nuplan_type: Optional[str] = None
    provenance: str = "constructed"
    frozen_at: str = ""
    frame: str = FRAME_CONVENTION
    route_polyline: List[List[float]] = field(default_factory=list)
    actors: Dict[str, ActorRecipe] = field(default_factory=dict)
    background_traffic: str = "keep"
    pair: Dict[str, Any] = field(default_factory=dict)
    review: Dict[str, Any] = field(default_factory=dict)
    # Why this clip was selected (mined recipes). The predicate that fired and
    # the numbers it fired on, so a reviewer can re-check the choice without
    # re-running the sweep — and so a wrong predicate leaves a trail. Empty for
    # constructed recipes, where `actors` carries the intent instead.
    selection: Dict[str, Any] = field(default_factory=dict)

    # -- validation ------------------------------------------------------
    def validate(self) -> "Recipe":
        if not self.recipe_id:
            raise RecipeError("recipe_id is required")
        if not self.leaf:
            raise RecipeError("leaf is required — a recipe that populates no leaf populates nothing")
        if self.frame != FRAME_CONVENTION:
            raise RecipeError(
                f"frame must be {FRAME_CONVENTION!r} (never absolute UTM: float32 at a 4.7e6 m "
                f"easting has a 0.5 m ULP), got {self.frame!r}"
            )
        if self.provenance not in PROVENANCES:
            raise RecipeError(
                f"provenance must be one of {list(PROVENANCES)}, got {self.provenance!r}"
            )
        if self.provenance == "mined":
            # A mined leaf's event is the road itself, so there is nothing to
            # edit — and an actor here would mean the scenario was quietly
            # constructed while claiming to be logged.
            if self.actors:
                raise RecipeError(
                    f"provenance 'mined' but the recipe carries actors "
                    f"{sorted(self.actors)}: a mined scenario is the untouched log. "
                    f"Use provenance 'constructed' if the actors are intended."
                )
            if not self.selection:
                raise RecipeError(
                    "a mined recipe must record `selection` — which predicate chose this "
                    "clip and on what numbers. Without it the choice cannot be audited, "
                    "and a wrong predicate leaves no trail (V-10 shipped twice on a "
                    "junction turn-fan before the evidence was written down)."
                )
        elif not self.actors:
            raise RecipeError("a constructed recipe with no actors edits nothing")
        for actor in self.actors.values():
            actor.validate(self.frames)
        for name in self.pair.get("e0_removes", []) or []:
            if name not in self.actors:
                raise RecipeError(
                    f"pair.e0_removes names {name!r}, which is not an actor in this recipe"
                )
        return self

    def verify_integrity(self) -> "Recipe":
        """Re-hash every actor's specification against its recorded digest."""
        for actor in self.actors.values():
            if actor.op == "remove":
                continue
            try:
                actor.verify()
            except RecipeError as exc:
                raise RecipeError(f"{self.recipe_id}: {exc}") from None
        return self

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        out: dict = {
            "recipe_id": self.recipe_id,
            "leaf": self.leaf,
            "scenario": self.scenario,
            "nuplan_type": self.nuplan_type,
            "provenance": self.provenance,
            "frozen_at": self.frozen_at,
            "host": self.host.to_dict(),
            "frame": self.frame,
            "ego": self.ego.to_dict(),
            "route_polyline": _plain(self.route_polyline),
            "frames": self.frames.to_dict(),
        }
        out["actors"] = {name: actor.to_dict() for name, actor in self.actors.items()}
        out["background_traffic"] = self.background_traffic
        if self.pair:
            out["pair"] = _plain(self.pair)
        if self.review:
            out["review"] = _plain(self.review)
        if self.selection:
            out["selection"] = _plain(self.selection)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        if not isinstance(d, dict):
            raise RecipeError(f"a recipe must be a mapping, got {type(d).__name__}")
        actors_raw = d.get("actors") or {}
        if not isinstance(actors_raw, dict):
            raise RecipeError("actors must be a mapping of name -> actor")
        recipe = cls(
            recipe_id=str(d.get("recipe_id", "")),
            leaf=str(d.get("leaf", "")),
            host=HostSpec.from_dict(d.get("host") or {}),
            frames=Frames.from_dict(d.get("frames") or {}),
            ego=EgoSpec.from_dict(d.get("ego") or {}),
            scenario=str(d.get("scenario", "")),
            nuplan_type=d.get("nuplan_type"),
            provenance=str(d.get("provenance", "constructed")),
            frozen_at=str(d.get("frozen_at", "")),
            frame=str(d.get("frame", FRAME_CONVENTION)),
            route_polyline=[[float(x) for x in pt] for pt in (d.get("route_polyline") or [])],
            actors={
                str(name): ActorRecipe.from_dict(str(name), spec)
                for name, spec in actors_raw.items()
            },
            background_traffic=str(d.get("background_traffic", "keep")),
            pair=dict(d.get("pair") or {}),
            review=dict(d.get("review") or {}),
            selection=dict(d.get("selection") or {}),
        )
        return recipe.validate()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None, width=100)


def load_recipe(path: "str | Path", *, verify: bool = True) -> Recipe:
    """Parse a recipe file, validate it, and (by default) check its checksums.

    Args:
        path: the frozen ``.yaml``.
        verify: re-hash every baked array. Only turn this off to inspect a
            recipe you already know is corrupt.

    Raises:
        RecipeError: malformed, inconsistent, or failing its integrity check.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise RecipeError(f"{path}: not parseable as YAML: {exc}") from exc
    try:
        recipe = Recipe.from_dict(raw)
    except RecipeError as exc:
        raise RecipeError(f"{path}: {exc}") from exc
    if verify:
        recipe.verify_integrity()
    return recipe


def bake_state(
    position: Sequence,
    heading: Sequence,
    velocity: Sequence,
    valid: Sequence,
) -> Dict[str, np.ndarray]:
    """Quantise four freshly computed arrays into a storable ``state`` block."""
    return {
        "position": quantize_array("position", position),
        "heading": quantize_array("heading", heading),
        "velocity": quantize_array("velocity", velocity),
        "valid": canonical_array("valid", valid),
    }
