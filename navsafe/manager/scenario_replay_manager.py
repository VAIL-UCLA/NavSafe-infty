# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""
Universal Replay Agent Manager for ScenarioDescription format.
Manages spawning and updating of replay agents (vehicles, pedestrians, cyclists).
"""

import logging
import numpy as np
import os
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from navsafe.scenario.scenario_description import ScenarioDescription as SD
from navsafe.scenario.type import MetaDriveType
from navsafe.utils.hashing import stable_hash

# Agent-spawn outcomes MUST go through logging, not print(): after the
# IsaacLab AppLauncher boots, stdout is routed to the carb sink which drops
# anything below Warning. A bare print() of a spawn failure is therefore
# invisible in run.log, which made a "front camera renders no agents" bug
# undiagnosable from saved artifacts. Warning+ survives the sink.
logger = logging.getLogger(__name__)


def _safe_prim_name(name) -> str:
    """Sanitize an id into a valid USD prim-name segment.

    USD prim names allow only ``[A-Za-z0-9_]`` and may not start with a digit.
    py123d/av2 track ids are UUIDs containing hyphens (e.g.
    ``1c2ea4e7-4bc8-...``); used verbatim they form an ill-formed SdfPath, so
    ``stage.DefinePrim`` rejects it with "Path must be an absolute path: <>"
    and the agent silently fails to spawn (leaving only the ego). Replace any
    illegal character with ``_``.
    """
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not s:
        return "_"
    if s[0].isdigit():
        s = "_" + s
    return s


# MetaDrive vehicle USDs — ordered by approximate length (small → large).
# Each entry: (filename, approx_length_m, approx_width_m, approx_height_m)
_METADRIVE_VEHICLES = [
    ("beetle.usd",  4.1, 1.73, 1.5),   # SVehicle  – small
    ("ferra.usd",   4.3, 1.82, 1.3),   # DefaultVehicle – standard
    ("130.usd",     4.5, 1.85, 1.5),   # MVehicle  – mid-size
    ("lada.usd",    4.8, 1.82, 1.5),   # LVehicle  – large
    ("truck.usd",   5.5, 2.10, 2.0),   # XLVehicle – truck
]

# Log-bbox length (m) thresholds for picking a mesh pool by vehicle class.
# AV2 sedans/SUVs/pickups run ~4-6.5 m; box trucks ~6.5-9 m; buses and
# articulated trucks ~10-14 m. Without the split, a 13 m track wearing a
# sedan mesh scaled to fit renders as a cartoon giant car.
_TRUCK_LENGTH_M = 6.5
_BUS_LENGTH_M = 9.0


def _vehicle_asset_class(length: float) -> str:
    """Mesh-pool class ("car" / "truck" / "bus") for a track's bbox length."""
    if length >= _BUS_LENGTH_M:
        return "bus"
    if length >= _TRUCK_LENGTH_M:
        return "truck"
    return "car"


# IsaacSim imports — deferred to avoid failures before IsaacSim is initialized.
# Imported lazily inside methods that need them.
# from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
# from isaaclab.sim import SimulationContext
# import omni.isaac.core.utils.prims as prim_utils
# from pxr import Usd, UsdGeom, Gf


class ScenarioReplayManager:
    """
    Manage replay agents in IsaacSim from ScenarioDescription tracks.

    Handles:
    - Spawning agents based on type (vehicle, pedestrian, cyclist)
    - Updating agent states each timestep
    - Handling validity (showing/hiding agents)
    - Batched operations across multiple environments
    """

    def __init__(
        self,
        scenario: SD,
        num_envs: int,
        device: str = "cuda",
        use_real_assets: bool = False,
        asset_selection_strategy: str = "hash",
        random_seed=None,
        vehicle_assets_path: str = "",
        spawn_z_from_track: bool = False,
        injected_mesh_color_scale: float = 1.0,
    ):
        """
        Args:
            scenario: ScenarioDescription object
            num_envs: Number of parallel environments
            device: Device for tensor operations
            use_real_assets: If True, attempt to use vehicle USD assets when available.
            asset_selection_strategy: Strategy for selecting assets ('hash', 'random', 'sequential')
            random_seed: Random seed for reproducible asset selection
            vehicle_assets_path: Path to directory containing MetaDrive vehicle USDs.
            spawn_z_from_track: If True, place agent prims at the track's world
                z (base at ``position[2]``) instead of the assets backend's
                flat ``z=0`` ground. Required when compositing meshes into a
                scene that lives at absolute world height — e.g. the
                NuRec Gaussian reconstruction (roads at city z), where
                a z=0 prim sits tens of metres underground and never appears
                on camera.
        """
        self.scenario = scenario
        self.num_envs = num_envs
        self.device = device
        self.use_real_assets = use_real_assets
        self.asset_selection_strategy = asset_selection_strategy
        self.random_seed = random_seed
        self.vehicle_assets_path = vehicle_assets_path
        self.spawn_z_from_track = spawn_z_from_track
        self.injected_mesh_color_scale = injected_mesh_color_scale

        # Resolve available MetaDrive vehicle USDs
        self._vehicle_usds: List[Tuple[str, float, float, float]] = []
        if vehicle_assets_path:
            md_dir = os.path.join(vehicle_assets_path, "metadrive")
            for fname, vl, vw, vh in _METADRIVE_VEHICLES:
                full = os.path.join(md_dir, fname)
                if os.path.isfile(full):
                    self._vehicle_usds.append((full, vl, vw, vh))
            if self._vehicle_usds:
                print(f"[ReplayManager] Found {len(self._vehicle_usds)} MetaDrive vehicle USDs")
            else:
                print(f"[ReplayManager] WARNING: No MetaDrive vehicle USDs found in {md_dir}")

        self.tracks = scenario[SD.TRACKS]
        self.metadata = scenario[SD.METADATA]
        self.scenario_length = scenario[SD.LENGTH]

        # Agent data structures
        self.agent_ids = list(self.tracks.keys())
        self.num_agents = len(self.agent_ids)

        # Mapping from agent_id to index
        self.agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}

        # USD prims for each agent in each environment
        self.agent_prims: dict[tuple, str] = {}  # {(env_id, agent_id): prim_path}

        # Track which agents are currently valid/visible
        self.agent_visibility: dict[tuple, bool] = {}  # {(env_id, agent_id): bool}

        # Current timestep
        self.current_timestep = 0

        # Measured native asset dims, keyed by USD path (files are
        # immutable; () = measured but unusable, fall back to estimates).
        self._measured_dims_cache: dict[str, tuple] = {}

        # SDC / ego vehicle ID from scenario metadata
        self.ego_agent_id = str(self.metadata.get(SD.SDC_ID, "")) or None

        print(f"[ReplayManager] Initialized with {self.num_agents} agents, {self.scenario_length} timesteps")

    def spawn_agents(self, stage, parent_prim_path: str = "/World", *,
                     skip_types: set | None = None, ego_only: bool = False,
                     reuse_ego_prims: dict[int, str] | None = None) -> Dict:
        """
        Spawn all agents in all environments.

        Args:
            stage: USD stage
            parent_prim_path: Parent prim path
            skip_types: Set of MetaDriveType strings to skip (e.g. {"PEDESTRIAN", "CYCLIST"})
            ego_only: When True, spawn ONLY the ego agent — every non-ego agent is
                skipped regardless of ``skip_types``. Used by the NuRec
                backend, where the reconstruction bakes the other agents into its
                temporal Gaussians but the ego chassis is still needed as the
                camera mount (the ego camera rig parents under it).
            reuse_ego_prims: ``{env_id: prim_path}`` of existing ego prims to
                adopt instead of spawning new ones. Used by the mid-run scene
                rebuild (``NexusSimEnv._maybe_rebuild_scenario_scene``): the
                ego prim carries the TiledCamera rig as USD children, so it
                must survive a scenario switch — deleting and respawning it
                would orphan the camera sensors. The adopted prim is re-posed
                to this scenario's initial ego state (its previous scenario's
                fit-scale and asset are kept — visual only).

        Returns:
            Dict of created agent prim paths
        """
        from pxr import UsdGeom, Gf  # noqa: F401 — lazy import after IsaacSim init

        skip_types = skip_types or set()

        created_agents: dict[str, list[str]] = {
            'vehicles': [],
            'pedestrians': [],
            'cyclists': [],
            'other': []
        }
        failed = 0

        for env_id in range(self.num_envs):
            env_path = f"{parent_prim_path}/envs/env_{env_id}/agents"

            # Create agents scope
            agents_scope = UsdGeom.Scope.Define(stage, env_path)

            for agent_id, track_data in self.tracks.items():
                # Get initial state (first valid timestep)
                initial_state = self._get_initial_valid_state(track_data)
                if initial_state is None:
                    continue

                # Determine agent type
                agent_type = track_data[SD.TYPE]

                # ego-only mode: skip every non-ego agent (renderer-owned agent visuals)
                # EXCEPT injected obstacles — those were never in the log, so
                # the reconstruction cannot bake them; their mesh must spawn.
                injected = bool(
                    (track_data.get(SD.METADATA) or {}).get("injected_obstacle"))
                if ego_only and agent_id != self.ego_agent_id and not injected:
                    continue

                # Track-z placement applies ONLY to injected meshes: they must
                # composite into a scene at absolute city z (NuRec). The ego /
                # log agents keep the legacy flat z=0 spawn — lifting the ego
                # mesh to camera height puts its (sometimes composite) asset
                # right in front of every capture.
                initial_state["z_from_track"] = bool(
                    self.spawn_z_from_track and injected)

                # Skip filtered types
                if agent_type in skip_types:
                    continue

                # Scene rebuild: adopt the surviving ego prim (camera mount)
                # and RE-POSE it to this scenario's initial ego state — the
                # first camera capture happens before any step/override, so
                # a stale pose here renders frame 0 from the old scenario.
                if (reuse_ego_prims and agent_id == self.ego_agent_id
                        and env_id in reuse_ego_prims):
                    adopted = reuse_ego_prims[env_id]
                    self.agent_prims[(env_id, agent_id)] = adopted
                    self.agent_visibility[(env_id, agent_id)] = bool(
                        initial_state.get('valid', True))
                    prim = stage.GetPrimAtPath(adopted)
                    if prim.IsValid():
                        try:
                            self._update_transform(prim, initial_state)
                        except Exception as e:  # noqa: BLE001 — visual only
                            print(f"[Warning] Failed to re-pose adopted ego "
                                  f"prim {adopted}: {e}")
                    continue

                # Create agent prim (remove existing prim first to avoid xformOp conflicts)
                prim_path = f"{env_path}/{agent_type.lower()}_{_safe_prim_name(agent_id)}"
                existing = stage.GetPrimAtPath(prim_path)
                if existing.IsValid():
                    stage.RemovePrim(prim_path)

                try:
                    is_ego = (agent_id == self.ego_agent_id)
                    if agent_type == MetaDriveType.VEHICLE:
                        self._spawn_vehicle(stage, prim_path, initial_state, track_data, is_ego=is_ego)
                        created_agents['vehicles'].append(prim_path)
                    elif agent_type == MetaDriveType.PEDESTRIAN:
                        self._spawn_pedestrian(stage, prim_path, initial_state, track_data)
                        created_agents['pedestrians'].append(prim_path)
                    elif agent_type == MetaDriveType.CYCLIST:
                        self._spawn_cyclist(stage, prim_path, initial_state, track_data)
                        created_agents['cyclists'].append(prim_path)
                    elif MetaDriveType.is_traffic_object(agent_type):
                        self._spawn_traffic_object(stage, prim_path, initial_state, track_data)
                        created_agents['other'].append(prim_path)
                    else:
                        self._spawn_generic(stage, prim_path, initial_state, track_data)
                        created_agents['other'].append(prim_path)

                    self.agent_prims[(env_id, agent_id)] = prim_path
                    self.agent_visibility[(env_id, agent_id)] = initial_state['valid']

                except Exception as e:
                    failed += 1
                    logger.warning(
                        "failed to spawn agent %s (type=%r): %s",
                        agent_id, agent_type, e, exc_info=True)
                    continue

        spawned = sum(len(v) for v in created_agents.values())
        # WARNING (not INFO) whenever anything was dropped OR nothing spawned:
        # below-Warning records never survive the carb sink, and "spawned 0
        # agents without raising" is precisely the failure that renders an
        # empty world to the cameras — the case that must not be silent. A
        # healthy scene stays at INFO so this does not spam normal runs.
        (logger.warning if (failed or not spawned) else logger.info)(
            "ReplayManager: spawned %d agent(s) across %d env(s); %d failed",
            spawned, self.num_envs, failed)
        return created_agents

    @staticmethod
    def _scale_asset_albedo(stage, asset_prim, scale: float) -> None:
        """Multiply every shader color/emissive factor under ``asset_prim``.

        Handles the GLTF-MDL inputs the UrbanVerse conversions use
        (``base_color_factor``, ``emissive_strength``) plus the common
        UsdPreviewSurface/OmniPBR names, on the shader's authored value or its
        input default. Missing inputs are authored from (1,1,1) so untextured
        materials dim too. Best-effort: any prim without shading is skipped.
        """
        from pxr import Gf, Usd, UsdShade

        COLOR3 = ("base_color_factor", "diffuseColor", "diffuse_color_constant",
                  "diffuse_tint")
        SCALAR = ("emissive_strength",)

        seen = set()
        for prim in Usd.PrimRange(asset_prim):
            mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not mat or not mat.GetPrim().IsValid():
                continue
            mp = mat.GetPrim().GetPath().pathString
            if mp in seen:
                continue
            seen.add(mp)
            for child in Usd.PrimRange(mat.GetPrim()):
                shader = UsdShade.Shader(child)
                if not shader:
                    continue
                for name in COLOR3:
                    inp = shader.GetInput(name)
                    if not inp and name == "base_color_factor":
                        continue
                    if not inp:
                        continue
                    cur = inp.Get()
                    if cur is None:
                        cur = Gf.Vec3f(1.0, 1.0, 1.0)
                    try:
                        if hasattr(cur, "__len__") and len(cur) == 4:
                            inp.Set(type(cur)(cur[0] * scale, cur[1] * scale,
                                              cur[2] * scale, cur[3]))
                        else:
                            inp.Set(Gf.Vec3f(cur[0] * scale, cur[1] * scale,
                                             cur[2] * scale))
                    except Exception:
                        continue
                for name in SCALAR:
                    inp = shader.GetInput(name)
                    if inp and inp.Get():
                        try:
                            inp.Set(float(inp.Get()) * scale)
                        except Exception:
                            continue
                # Kill specular bloom: the dome's specular response is not
                # scaled by albedo, so a glossy/metallic surface still blows
                # out under the Gaussian-keyed auto-exposure.
                for name, val in (("metallic_factor", 0.0),
                                  ("roughness_factor", 0.9),
                                  ("clearcoat_factor", 0.0)):
                    inp = shader.GetInput(name)
                    if inp:
                        try:
                            inp.Set(float(val))
                        except Exception:
                            continue

    def _get_initial_valid_state(self, track_data: Dict) -> Optional[Dict]:
        """Get first valid state for initialization."""
        state = track_data[SD.STATE]
        valid = state.get('valid', np.ones(len(state['position'])))

        for t in range(len(valid)):
            if valid[t]:
                return self._get_state_at_timestep(track_data, t)

        return None

    def _get_state_at_timestep(self, track_data: Dict, timestep: int) -> Optional[Dict]:
        """Extract state at given timestep."""
        state = track_data[SD.STATE]

        if timestep < 0 or timestep >= len(state['position']):
            return None

        result = {
            'position': np.array(state['position'][timestep]),
            'heading': float(state['heading'][timestep]),
            'valid': state.get('valid', np.ones(len(state['position'])))[timestep]
        }

        if 'velocity' in state:
            result['velocity'] = np.array(state['velocity'][timestep])

        if 'length' in state:
            result['length'] = float(state['length'][timestep])
        if 'width' in state:
            result['width'] = float(state['width'][timestep])
        if 'height' in state:
            result['height'] = float(state['height'][timestep])

        return result

    def get_agent_state(self, agent_id: str, timestep: int) -> Optional[Dict]:
        """Return the world-frame state dict for ``agent_id`` at ``timestep``.

        Thin public wrapper around :meth:`_get_state_at_timestep` that
        resolves the agent id to the underlying track first, so callers
        (e.g. :class:`ScenarioReplayEnv`) don't have to reach into private
        track storage.

        Args:
            agent_id: Track id as it appears in ``scenario_data['tracks']``.
            timestep: Scenario timestep.

        Returns:
            ``{'position': ndarray, 'heading': float, 'valid': bool, ...}``
            or ``None`` if the agent is unknown or the timestep is out of
            range.
        """
        track = self.tracks.get(agent_id) if hasattr(self, "tracks") else None
        if track is None:
            return None
        return self._get_state_at_timestep(track, timestep)

    def get_initial_state(self, agent_id: str) -> Optional[Dict]:
        """First-valid-timestep state for ``agent_id`` (or ``None``).

        Public wrapper around :meth:`_get_initial_valid_state` — callers
        (e.g. the env's contact-collider pool) need spawn dims without
        reaching into private track storage.
        """
        track = self.tracks.get(agent_id) if hasattr(self, "tracks") else None
        if track is None:
            return None
        return self._get_initial_valid_state(track)

    def _spawn_from_usd(
        self,
        stage,
        prim_path: str,
        usd_path: str,
        initial_state: Dict,
        target_dims: Tuple[float, float, float],
        bbox: Tuple[float, float, float],
        origin_at_center: bool = True,
        y_up_asset: bool = False,
        scale_on_child: bool = False,
    ):
        """Spawn an agent by referencing a USD file and scaling to fit track dimensions.

        Args:
            stage: USD stage.
            prim_path: Destination prim path.
            usd_path: Absolute path to USD (or GLB) asset file.
            initial_state: Agent state dict (position, heading).
            target_dims: (length, width, height) from scenario data.
            bbox: (L, W, H) fallback estimate (catalog annotations or a
                hard-coded guess) — used only when measuring the referenced
                asset's actual bounds fails; the measured bounds win.
            origin_at_center: If True (MetaDrive Z-up vehicles), offset z by half
                height so bottom sits on ground.  If False (UrbanVerse Y-up
                assets where bottom is at Y=0), z_offset = 0 since the mesh
                bottom is already at Z=0 after the Y-up→Z-up rotation.
            y_up_asset: If True, the asset uses Y-up convention (UrbanVerse).
                Applies a base rotation to convert to Z-up world frame.
            scale_on_child: Author the fit-scale on an intermediate child
                Xform instead of the transform wrapper. Used for the EGO:
                the TiledCamera rig parents under the wrapper, and a wrapper
                scale op would scale the rig's metric mount offsets (camera
                height 1.49 m became 1.49 x asset-fit-scale).
        """
        from pxr import UsdGeom, Gf, UsdPhysics  # noqa: F401 — lazy IsaacSim import

        # Guard against an empty/None asset path: AddReference("") is treated
        # as an internal reference to the empty prim path and raises the
        # cryptic "Path must be an absolute path: <>" USD error, killing the
        # spawn. Surface a clear error so the caller's fallback chain can act.
        if not usd_path:
            raise ValueError(f"empty usd_path for {prim_path}")

        # Transform ops live on a dedicated wrapper Xform; the asset is
        # referenced onto a CHILD prim. Referencing a USD whose root prim
        # already carries an xformOpOrder (UrbanVerse converted USDs ship a
        # double3 xformOp:scale) directly onto the transform-bearing prim
        # makes AddScaleOp/AddTranslateOp clash with the existing op
        # ("xformOp:scale has typeName 'double3' which does not match the
        # requested precision 'PrecisionFloat'"), which previously caused
        # EVERY non-ego agent spawn to fail. Keeping the two concerns on
        # separate prims avoids the clash.
        xform_prim = stage.DefinePrim(prim_path, "Xform")
        scale_prim = None
        if scale_on_child:
            # translate/rotate on the wrapper; fit-scale on an intermediate
            # child so wrapper children (the ego camera rig) stay metric.
            scale_prim = stage.DefinePrim(prim_path + "/scaled", "Xform")
            asset_prim = stage.DefinePrim(prim_path + "/scaled/asset", "Xform")
        else:
            asset_prim = stage.DefinePrim(prim_path + "/asset", "Xform")
        asset_prim.GetReferences().AddReference(usd_path)

        # Injected meshes composited into the NuRec Gaussian scene render
        # blown-out white: the scene's histogram auto-exposure is keyed to the
        # dim emissive Gaussians (~1-5/255 raw), so a conventionally-lit mesh
        # saturates. Scale the asset's albedo down to land in the Gaussian
        # radiance range. Must happen HERE (spawn time, before the render
        # stack snapshots the stage into Fabric) — runtime USD material/light
        # edits never reach the RTX renderer.
        if self.injected_mesh_color_scale != 1.0 and initial_state.get('z_from_track'):
            self._scale_asset_albedo(stage, asset_prim, self.injected_mesh_color_scale)

        # Strip ALL physics APIs so the prim is purely visual.
        # Kinematic rigid bodies still have transforms managed by PhysX,
        # which overwrites our pxr translate_op/rotate_op updates each step.
        # Removing the APIs entirely makes the prims invisible to the physics
        # engine, so our per-frame xform writes are the sole authority.
        try:
            for desc in [asset_prim] + list(asset_prim.GetAllChildren()):
                if desc.HasAPI(UsdPhysics.ArticulationRootAPI):
                    desc.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if desc.HasAPI(UsdPhysics.RigidBodyAPI):
                    desc.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if desc.HasAPI(UsdPhysics.CollisionAPI):
                    desc.RemoveAPI(UsdPhysics.CollisionAPI)
                # Deactivate joints (can't remove, but deactivating is enough)
                joint = UsdPhysics.Joint(desc)
                if joint.GetPrim().IsValid() and joint.GetPrim().IsA(UsdPhysics.Joint):
                    desc.SetActive(False)
        except Exception:
            pass  # Non-fatal: physics errors are cosmetic in kinematic replay

        xformable = UsdGeom.Xformable(xform_prim)
        xformable.ClearXformOpOrder()

        # Measure the referenced asset's ACTUAL native bounds and prefer them
        # over the caller-supplied bbox (which is only a fallback estimate).
        # Converted UrbanVerse USDs are not unit-normalized, and several call
        # sites passed hard-coded guesses (vehicles: 4.0x1.8x1.5; cones:
        # their own target dims → scale 1.0), which rendered mis-normalized
        # assets 10-100x too large on camera. Measured sizes are memoized per
        # USD path (immutable files; cones reuse ONE USD for every instance).
        measured = self._measured_dims_cache.get(usd_path)
        if measured is None:
            try:
                from pxr import Usd  # noqa: F811 — lazy
                cache = UsdGeom.BBoxCache(
                    Usd.TimeCode.Default(),
                    [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                rng = cache.ComputeLocalBound(asset_prim).ComputeAlignedRange()
                if not rng.IsEmpty():
                    size = rng.GetSize()
                    measured = (float(size[0]), float(size[1]),
                                float(size[2]))
            except Exception as e:  # noqa: BLE001 — fall back to estimate
                print(f"[ReplayManager] bbox measurement failed for "
                      f"{usd_path}: {e}; using caller estimate {bbox}")
            if measured is None or min(measured) <= 1e-4:
                measured = ()  # sentinel: measured-and-unusable
                print(f"[ReplayManager] empty/degenerate bounds for "
                      f"{usd_path}; using caller estimate {bbox}")
            self._measured_dims_cache[usd_path] = measured
        if measured:
            sx, sy, sz = measured
            # Y-up assets: X=width, Y=height(up), Z=length(forward);
            # Z-up assets: X=length, Y=width, Z=height.
            bbox = (sz, sx, sy) if y_up_asset else (sx, sy, sz)

        # Scale uniformly to fit the scenario's reported dimensions
        tgt_l, tgt_w, tgt_h = target_dims
        src_l, src_w, src_h = bbox
        if src_l > 0 and src_w > 0 and src_h > 0:
            scale = min(tgt_l / src_l, tgt_w / src_w, tgt_h / src_h)
        else:
            scale = 1.0

        # Position — place bottom of agent on ground (z = 0, assets backend's
        # flat scene) or at the track's world z (injected meshes under NuRec).
        position = initial_state['position']
        heading = initial_state['heading']
        if origin_at_center:
            z_offset = (src_h * scale) / 2.0
        else:
            z_offset = 0.0
        z_from_track = bool(initial_state.get('z_from_track'))
        base_z = (float(position[2])
                  if z_from_track and len(position) > 2 else 0.0)
        xform_prim.SetCustomDataByKey("navsafe:z_from_track", z_from_track)

        # Apply transforms in order: translate → heading rotation → base rotation → scale
        # USD applies ops in the order listed (left-to-right = outermost-to-innermost).
        xformable.AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), base_z + z_offset)
        )

        if y_up_asset:
            # UrbanVerse assets: X=width, Y=height(up), Z=length(forward).
            # RotX(+90): Y-up→Z-up, Z(fwd)→-Y.
            # RotZ(heading+90): aligns -Y forward with world heading.
            heading_deg = float(np.degrees(heading))
            xformable.AddRotateXYZOp().Set(Gf.Vec3d(90.0, 0.0, heading_deg - 90.0))
        else:
            # MetaDrive / Z-up assets: X=forward, Z=up. Just apply heading around Z.
            quat = self._heading_to_quaternion(heading)
            xformable.AddOrientOp().Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))

        if scale_prim is not None:
            UsdGeom.Xformable(scale_prim).AddScaleOp().Set(
                Gf.Vec3f(scale, scale, scale))
        else:
            xformable.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

        # Store the z-offset so _update_transform can reuse it
        xform_prim.SetCustomDataByKey("navsafe:z_offset", float(z_offset))
        # Store whether this is a Y-up asset for _update_transform rotation
        xform_prim.SetCustomDataByKey("navsafe:y_up_asset", y_up_asset)

    def _select_vehicle_usd(self, length: float, seed: int) -> Optional[Tuple[str, float, float, float]]:
        """Pick the MetaDrive vehicle USD whose length is closest to *length*.

        If multiple USDs are equally close, *seed* breaks the tie deterministically.
        Returns (usd_path, src_length, src_width, src_height) or None.
        """
        if not self._vehicle_usds:
            return None
        # Sort by absolute length difference, then by seed for determinism
        ranked = sorted(self._vehicle_usds, key=lambda v: abs(v[1] - length))
        return ranked[seed % min(2, len(ranked))]  # pick from top-2 closest

    def _spawn_vehicle(self, stage, prim_path: str, initial_state: Dict, track_data: Dict, is_ego: bool = False):
        """Spawn a vehicle.

        Ego vehicle uses MetaDrive USDs (distinctive model, Z-up).
        Non-ego vehicles use UrbanVerse sedan USDs (Y-up, need rotation).
        Falls back to MetaDrive → UrbanVerse → cube.
        """
        from pxr import UsdGeom, Gf  # noqa: F811
        length = initial_state.get('length', 4.5)
        width = initial_state.get('width', 2.0)
        height = initial_state.get('height', 1.5)

        if is_ego:
            # Ego: try UrbanVerse first, then MetaDrive. scale_on_child keeps
            # the camera rig's metric mount offsets unscaled (see
            # _spawn_from_usd).
            md = self._select_vehicle_usd(length, seed=stable_hash(prim_path) % (2**31))
            if md is not None:
                usd_path, src_l, src_w, src_h = md
                self._spawn_from_usd(
                    stage, prim_path, usd_path, initial_state,
                    (length, width, height), (src_l, src_w, src_h),
                    y_up_asset=False,
                    scale_on_child=True,
                )
                return
        else:
            # Non-ego fallback: MetaDrive USD (Z-up)
            md = self._select_vehicle_usd(length, seed=stable_hash(prim_path) % (2**31))
            if md is not None:
                usd_path, src_l, src_w, src_h = md
                self._spawn_from_usd(
                    stage, prim_path, usd_path, initial_state,
                    (length, width, height), (src_l, src_w, src_h),
                    y_up_asset=False,
                )
                return

        # Final fallback: colored cube wrapped in an Xform parent.  The
        # parent owns the per-frame translate/orient (canonical op order
        # so IsaacLab's XformPrimView accepts it for sensor attachment);
        # the cube child carries the dimensional scale.
        position = initial_state['position']
        heading = initial_state['heading']
        height_offset = float(height) / 2.0

        xform_prim = stage.DefinePrim(prim_path, "Xform")
        xformable = UsdGeom.Xformable(xform_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(
            Gf.Vec3d(float(position[0]), float(position[1]), height_offset)
        )
        quat = self._heading_to_quaternion(heading)
        xformable.AddOrientOp().Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))
        # IsaacLab's XformPrimView requires all three canonical ops
        # (translate, orient, scale) even when scale is identity.
        xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        xform_prim.SetCustomDataByKey("navsafe:z_offset", height_offset)

        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/body")
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(length, width, height))
        cube.CreateDisplayColorAttr([(0.2, 0.4, 0.8)])

    def _spawn_pedestrian(self, stage, prim_path: str, initial_state: Dict, track_data: Dict):
        """Spawn a pedestrian — UrbanVerse USD if available, else colored cube."""
        from pxr import UsdGeom, Gf  # noqa: F811

        height = initial_state.get('height', 1.7)
        width = initial_state.get('width', 0.6)
        length = initial_state.get('length', 0.6)

        # Try UrbanVerse pedestrian USD

        # Fallback: colored cube
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)

        position = initial_state['position']
        heading = initial_state['heading']

        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), height / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(float(length), float(width), float(height)))

        quat = self._heading_to_quaternion(heading)
        xform.AddOrientOp().Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))

        cube.CreateDisplayColorAttr([(0.2, 0.8, 0.2)])  # green

    def _spawn_cyclist(self, stage, prim_path: str, initial_state: Dict, track_data: Dict):
        """Spawn a cyclist using the pedestrian proxy geometry."""
        from pxr import UsdGeom, Gf  # noqa: F811

        height = initial_state.get('height', 1.2)
        width = initial_state.get('width', 0.6)
        length = initial_state.get('length', 1.8)

        # Reuse a pedestrian USD (person) for the rider.

        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)

        position = initial_state['position']
        heading = initial_state['heading']

        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), height / 2))
        xform.AddScaleOp().Set(Gf.Vec3f(float(length), float(width), float(height)))

        quat = self._heading_to_quaternion(heading)
        xform.AddOrientOp().Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))

        cube.CreateDisplayColorAttr([(0.8, 0.8, 0.2)])  # yellow

    def _spawn_traffic_object(self, stage, prim_path: str, initial_state: Dict, track_data: Dict):
        """Spawn a traffic object (cone, barrier, sign) using UrbanVerse assets."""
        from pxr import UsdGeom, Gf  # noqa: F811

        agent_type = track_data[SD.TYPE]
        height = initial_state.get('height', 0.7)
        width = initial_state.get('width', 0.5)
        length = initial_state.get('length', 0.5)

        # Try UrbanVerse pre-converted cone USDs (use first one for
        # consistency). Barriers are excluded: rendering a barrier as a cone
        # scaled to cone dims misrepresents a lane-blocking obstacle — they
        # fall through to the primitive barrier below.

        # Pick proxy geometry based on type
        if agent_type == MetaDriveType.TRAFFIC_CONE:
            cat = "traffic_cone"
        elif agent_type == MetaDriveType.TRAFFIC_BARRIER:
            cat = "barrier"
        else:
            cat = "traffic_cone"


        # Fallback primitives: barriers get an orange BOX at track dims (a
        # 0.15 m-radius cone would misrepresent a lane-blocking obstacle);
        # everything else gets the orange cone.
        if agent_type == MetaDriveType.TRAFFIC_BARRIER:
            box = UsdGeom.Cube.Define(stage, prim_path)
            box.CreateSizeAttr(1.0)
            box.CreateDisplayColorAttr([(1.0, 0.5, 0.0)])
            position = initial_state['position']
            xform = UsdGeom.Xformable(box.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(
                float(position[0]), float(position[1]), height / 2))
            xform.AddScaleOp().Set(Gf.Vec3f(length, width, height))
            return

        cone = UsdGeom.Cone.Define(stage, prim_path)
        cone.CreateHeightAttr(height)
        cone.CreateRadiusAttr(0.15)
        cone.CreateDisplayColorAttr([(1.0, 0.5, 0.0)])

        position = initial_state['position']
        xform = UsdGeom.Xformable(cone.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), height / 2))

    def _spawn_generic(self, stage, prim_path: str, initial_state: Dict, track_data: Dict):
        """Spawn a generic/unknown object using the best available asset.

        Tries traffic_cone assets first (most common unknown type), then
        pedestrian, then falls back to a small cube.
        """
        from pxr import UsdGeom, Gf  # noqa: F811

        height = initial_state.get('height', 1.0)
        width = initial_state.get('width', 0.5)
        length = initial_state.get('length', 0.5)


        # Final fallback: small cube
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.AddScaleOp().Set(Gf.Vec3f(length, width, height))

        position = initial_state['position']
        xform = UsdGeom.Xformable(cube)
        xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), height / 2))

        cube.CreateDisplayColorAttr([(0.8, 0.2, 0.8)])

    def update_agents(self, stage, timestep: int, skip_ego: bool = False,
                      skip_agents=None):
        """
        Update all agents to states at given timestep.

        Args:
            stage: USD stage
            timestep: Current timestep
            skip_ego: If True, skip the ego agent (it's being controlled externally)
            skip_agents: Optional set of agent ids to leave untouched (they are
                being driven by a traffic manager, e.g. the semi-reactive
                takeover set); their prims keep whatever pose was last written
                via :meth:`set_agent_pose`.
        """
        if timestep >= self.scenario_length:
            # Clamp to the final logged frame instead of returning: an early
            # return froze every prim at its last-updated pose (visually
            # indistinguishable from a live frame), which silently corrupted
            # camera captures whenever an episode ran past the log's length.
            if not getattr(self, "_overrun_logged", False):
                self._overrun_logged = True
                print(f"[ReplayManager] episode ran past the log "
                      f"(timestep {timestep} >= length {self.scenario_length});"
                      f" agents hold their final logged pose")
            timestep = max(0, self.scenario_length - 1)
        self.current_timestep = timestep

        for env_id in range(self.num_envs):
            for agent_id, track_data in self.tracks.items():
                # Skip ego agent when externally controlled
                if skip_ego and agent_id == self.ego_agent_id:
                    continue
                # Skip agents owned by a traffic manager (semi-reactive takeover)
                if skip_agents and agent_id in skip_agents:
                    continue

                key = (env_id, agent_id)
                if key not in self.agent_prims:
                    continue

                prim_path = self.agent_prims[key]
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                # Get state at this timestep
                state = self._get_state_at_timestep(track_data, timestep)
                if state is None:
                    continue

                # Update visibility
                was_visible = self.agent_visibility.get(key, False)
                is_visible = bool(state['valid'])

                if is_visible != was_visible:
                    self._set_visibility(prim, is_visible)
                    self.agent_visibility[key] = is_visible

                if not is_visible:
                    continue

                # Update position and rotation
                self._update_transform(prim, state)

    def set_agent_pose(self, stage, agent_id, position, heading: float,
                       env_id: int = 0) -> bool:
        """Pose one agent prim directly, bypassing the replay tracks.

        Used by traffic managers that own an agent's motion (e.g. the
        semi-reactive takeover set): the manager integrates the pose itself
        and writes it here each frame instead of replaying the log.

        Args:
            stage: USD stage.
            agent_id: Track id of the agent to pose.
            position: (x, y) or (x, y, z) world position.
            heading: Yaw in radians.
            env_id: Environment index (single-env default 0).

        Returns:
            True if a valid prim was posed.
        """
        key = (env_id, agent_id)
        prim_path = self.agent_prims.get(key)
        if prim_path is None:
            return False
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        if not self.agent_visibility.get(key, False):
            self._set_visibility(prim, True)
            self.agent_visibility[key] = True
        pos = list(position) + [0.0] * (3 - len(position))
        self._update_transform(prim, {"position": pos, "heading": float(heading)})
        return True

    def _update_transform(self, prim, state: Dict):
        """Update prim transform based on state."""
        from pxr import UsdGeom, Gf  # noqa: F811
        xform = UsdGeom.Xformable(prim)

        # Get position
        position = state['position']

        # Determine height offset based on prim type
        z_offset = prim.GetCustomDataByKey("navsafe:z_offset")
        if z_offset is not None:
            height_offset = float(z_offset)
        elif UsdGeom.Cube.Get(prim.GetStage(), prim.GetPath()):
            height_offset = state.get('height', 1.5) / 2
        elif UsdGeom.Capsule.Get(prim.GetStage(), prim.GetPath()):
            height_offset = 0.85
        elif UsdGeom.Sphere.Get(prim.GetStage(), prim.GetPath()):
            height_offset = 1.0
        else:
            height_offset = 0.0

        # Find ops by name
        ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}

        # Update translation
        base_z = (float(position[2])
                  if prim.GetCustomDataByKey("navsafe:z_from_track")
                  and len(position) > 2 else 0.0)
        translate_op = ops.get('xformOp:translate')
        if translate_op is not None:
            translate_op.Set(Gf.Vec3d(
                float(position[0]), float(position[1]), base_z + float(height_offset)))

        # Update rotation
        heading = state['heading']
        y_up = prim.GetCustomDataByKey("navsafe:y_up_asset")

        rotate_op = ops.get('xformOp:rotateXYZ')
        orient_op = ops.get('xformOp:orient')

        if y_up and rotate_op is not None:
            # UrbanVerse Y-up asset: RotateXYZ(90, 0, heading_deg - 90)
            heading_deg = float(np.degrees(heading))
            rotate_op.Set(Gf.Vec3d(90.0, 0.0, heading_deg - 90.0))
        elif orient_op is not None:
            # MetaDrive Z-up asset: quaternion around Z
            quat = self._heading_to_quaternion(heading)
            orient_op.Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))
        else:
            # Fallback: try orient, create if needed
            orient_op = ops.get('xformOp:orient')
            if orient_op is None:
                orient_op = xform.AddOrientOp()
            quat = self._heading_to_quaternion(heading)
            orient_op.Set(Gf.Quatf(quat[3], quat[0], quat[1], quat[2]))

    def _set_visibility(self, prim, visible: bool):
        """Set prim visibility."""
        from pxr import UsdGeom  # noqa: F811
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _heading_to_quaternion(self, heading: float) -> Tuple[float, float, float, float]:
        """
        Convert heading angle (radians) to quaternion.
        Heading is rotation around Z axis.

        Returns:
            (x, y, z, w)
        """
        half_angle = heading / 2
        return (
            0.0,
            0.0,
            np.sin(half_angle),
            np.cos(half_angle)
        )

    def get_agent_info(self) -> Dict:
        """Get information about spawned agents."""
        info = {
            'total_agents': self.num_agents,
            'scenario_length': self.scenario_length,
            'current_timestep': self.current_timestep,
            'num_envs': self.num_envs
        }

        # Count by type
        type_counts: dict[str, int] = {}
        for track_data in self.tracks.values():
            agent_type = track_data[SD.TYPE]
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

        info['agents_by_type'] = type_counts

        return info

    def reset(self, stage, timestep: int = 0):
        """Reset all agents to given timestep."""
        self.update_agents(stage, timestep)
