"""Shared LiDAR collision geometry and ego prim-path resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SceneBuilder:
    """USD scene-construction subsystem for a :class:`NavSafeEnv` (composition).

    Forwards all shared state/method access to the owning env; the env owns the
    scene state (``map_builder``) via its ``_setup_scene``
    lifecycle hook.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def __getattr__(self, name: str) -> Any:
        # Near-stateless: everything except the back-ref forwards to the env.
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)


    def _build_combined_lidar_mesh(self) -> "str | None":
        """Merge ground + lane/road meshes into a single UsdGeomMesh.

        IsaacLab's :class:`RayCaster` only accepts one mesh prim, so the
        default ``/World/ground`` plane yields a flat-disk point cloud
        with no road structure.  Walking every lane mesh in the stage
        and concatenating their vertices/faces into one combined mesh
        gives the LiDAR something to actually hit — road surface, lane
        boundaries, sidewalks — without changing the sensor stack.

        The combined prim is anchored at ``/World/lidar_scene_mesh``.
        Returns its path, or ``None`` if no candidate meshes were found
        (caller falls back to the plain ground plane).
        """
        try:
            from pxr import UsdGeom, Gf, Usd  # lazy import — needs IsaacSim init
        except Exception:
            return None

        stage = self.sim.stage
        candidates: list[str] = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if "/map/" not in path:
                continue
            if prim.GetTypeName() != "Mesh":
                continue
            candidates.append(path)
        if not candidates:
            print("[NavSafeEnv] Combined LiDAR mesh: no /map/ mesh prims found — "
                  "falling back to /World/ground")
            return None

        verts: list[tuple[float, float, float]] = []
        face_counts: list[int] = []
        face_indices: list[int] = []
        offset = 0
        for path in candidates:
            mesh_prim = stage.GetPrimAtPath(path)
            mesh = UsdGeom.Mesh(mesh_prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts or not indices:
                continue
            # Compose the prim's local-to-world transform: some map meshes (the
            # unified road surface) are built with points local to a centred
            # origin and placed via an xform translate (so their object-space
            # texture UVs stay precise). Reading raw points would put them at the
            # origin instead of the scenario location.
            xf = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            for p in pts:
                w = xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                verts.append((float(w[0]), float(w[1]), float(w[2])))
            face_counts.extend(int(c) for c in counts)
            face_indices.extend(int(i) + offset for i in indices)
            offset += len(pts)

        # Add a finite ground patch (200×200 m around origin) so rays
        # that miss the road network still get a ground return — the
        # /World/ground plane is excluded from the merge because USD
        # planes don't expose mesh attributes.
        # Size the ground patch to the lane bbox + 50 m margin so
        # scenarios far from world origin still get ground returns.
        if verts:
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            x_lo, x_hi = min(xs) - 50.0, max(xs) + 50.0
            y_lo, y_hi = min(ys) - 50.0, max(ys) + 50.0
        else:
            x_lo, x_hi, y_lo, y_hi = -200.0, 200.0, -200.0, 200.0
        ground_z = 0.0
        v0 = len(verts)
        verts.extend([
            (x_lo, y_lo, ground_z),
            (x_hi, y_lo, ground_z),
            (x_hi, y_hi, ground_z),
            (x_lo, y_hi, ground_z),
        ])
        face_counts.append(4)
        face_indices.extend([v0, v0 + 1, v0 + 2, v0 + 3])

        out_path = "/World/lidar_scene_mesh"
        if stage.GetPrimAtPath(out_path).IsValid():
            stage.RemovePrim(out_path)
        out_mesh = UsdGeom.Mesh.Define(stage, out_path)
        out_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
        out_mesh.CreateFaceVertexCountsAttr(face_counts)
        out_mesh.CreateFaceVertexIndicesAttr(face_indices)
        print(f"[NavSafeEnv] Combined LiDAR mesh: {len(candidates)} source prims, "
              f"{len(verts)} verts, {len(face_counts)} faces (out: {out_path})")
        return out_path

    def _resolve_ego_prim_path(self, configured: str) -> str:
        """Substitute the legacy ``{ENV_REGEX_NS}/ego_vehicle/chassis`` stem
        with the real ego vehicle prim path created by ``ReplayManager``.

        The sensor cfgs ship with that stem as a placeholder, but
        ``_setup_extra_sensors`` instantiates each sensor directly rather
        than registering it through ``InteractiveScene``, so ``IsaacLab``
        never expands ``{ENV_REGEX_NS}``.  We resolve it here against
        ``agent_manager.agent_prims`` instead.  Any user-supplied absolute
        path that doesn't contain the stem is returned unchanged.
        """
        legacy_stem = "{ENV_REGEX_NS}/ego_vehicle/chassis"
        if legacy_stem not in configured:
            return configured
        if getattr(self, "num_envs", 1) > 1:
            logger.warning(
                "[NavSafeEnv] Ego-prim resolution currently assumes "
                "num_envs=1; got num_envs=%d. Sensors will attach only to "
                "env_0's ego.", self.num_envs,
            )
        ego_id = getattr(self.agent_manager, "ego_agent_id", None)
        if ego_id is None:
            logger.warning(
                "[NavSafeEnv] No ego_agent_id on the scenario; "
                "leaving sensor prim_path unresolved (%s).", configured,
            )
            return configured
        real_prim = self.agent_manager.agent_prims.get((0, ego_id))
        if not real_prim:
            return configured
        return configured.replace(legacy_stem, real_prim)
