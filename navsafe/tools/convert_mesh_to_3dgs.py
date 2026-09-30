# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Convert a textured mesh (GLB/GLTF/OBJ/...) into a 3DGS PLY asset.

Why
---
The NuRec ``serve-grpc`` renderer draws 3D Gaussian splats only — its asset
loader (``Asset.from_ply_bytes``) reads PLY *vertex attributes* (``opacity``,
``f_dc_*``, ``f_rest_*``, ``scale_*``, ``rot_*``) and never looks at triangle
faces, so a mesh asset (e.g. an UrbanVerse GLB) cannot be inserted directly
via the ``edit_assets`` RPC (see ``docs/edit_scenario.md``).

This tool closes the gap without any training: surface-sample the mesh,
turn every sample into a small surface-aligned (pancake) Gaussian, and write
the standard INRIA-layout 3DGS PLY the server expects. For simple props
(traffic cones, barriers) the quality is indistinguishable from a trained
splat; for complex shiny assets a trained pipeline (e.g. EA ``mesh2splat``
or multi-view 3DGS optimization) can substitute — the output contract is
the same PLY.

Conventions applied for NexusSim's insertion path:

* y-up FILE output (default): the server's ``insert_asset`` applies a fixed
  y-up -> z-up rotation to every loaded PLY (the NVIDIA asset convention),
  so a z-up file would render lying on its side. The asset is still BUILT
  z-up internally; ``--out-up-axis z`` keeps the old z-up file for viewers;
* origin at the footprint centre with the base on z=0 — matching the
  renderer's "pose z rides the ground" convention (``_make_aabb``);
* scales stored as log, opacity as logit, colors as SH DC terms — the
  standard 3DGS PLY activation conventions.

Usage (no IsaacSim / GPU required)::

    python -m navsafe.tools.convert_mesh_to_3dgs cone.glb cone_3dgs.ply \\
        --points 60000 --target-height 0.7

    # built-in procedural traffic cone (no input mesh needed):
    python -m navsafe.tools.convert_mesh_to_3dgs --procedural-cone cone_3dgs.ply
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# SH degree-0 basis constant: color = 0.5 + C0 * f_dc.
SH_C0 = 0.28209479177387814

# INRIA 3DGS PLY vertex-attribute layout (order matters for some readers).
_BASE_FIELDS = ["x", "y", "z", "nx", "ny", "nz"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _quat_z_to(normals: np.ndarray) -> np.ndarray:
    """Quaternions (w, x, y, z) rotating +z onto each unit normal.

    Rotation axis = z x n, angle = arccos(n_z); degenerate n == -z gets a
    180-degree flip about x. Vectorized half-angle construction.
    """
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    dots = np.clip(n @ z, -1.0, 1.0)
    axes = np.cross(np.broadcast_to(z, n.shape), n)
    axis_norm = np.linalg.norm(axes, axis=1, keepdims=True)
    # Default axis for the parallel / antiparallel cases.
    axes = np.where(axis_norm > 1e-8, axes / np.maximum(axis_norm, 1e-12),
                    np.array([1.0, 0.0, 0.0]))
    half = 0.5 * np.arccos(dots)
    q = np.empty((len(n), 4), np.float32)
    q[:, 0] = np.cos(half)
    q[:, 1:] = axes * np.sin(half)[:, None]
    return q


def _sample_colors(mesh, points: np.ndarray, face_idx: np.ndarray) -> np.ndarray:
    """Per-sample RGB in [0, 1]. Tries UV texture, then vertex/face colors,
    then the material base color, then mid-grey."""
    n = len(points)
    visual = getattr(mesh, "visual", None)
    # UV-mapped texture: barycentric UV at each sample -> texture lookup.
    try:
        if visual is not None and getattr(visual, "uv", None) is not None:
            import trimesh
            bary = trimesh.triangles.points_to_barycentric(
                mesh.triangles[face_idx], points)
            uv = (visual.uv[mesh.faces[face_idx]] * bary[:, :, None]).sum(axis=1)
            material = getattr(visual, "material", None)
            image = getattr(material, "baseColorTexture", None) or getattr(
                material, "image", None)
            if image is not None:
                from trimesh.visual.color import uv_to_interpolated_color
                rgba = uv_to_interpolated_color(uv, image)
                return np.asarray(rgba[:, :3], np.float32) / 255.0
    except Exception:
        pass
    # Vertex colors (barycentric blend) or flat face colors.
    try:
        vc = getattr(visual, "vertex_colors", None)
        if vc is not None and len(vc) == len(mesh.vertices):
            import trimesh
            bary = trimesh.triangles.points_to_barycentric(
                mesh.triangles[face_idx], points)
            rgba = (np.asarray(vc, np.float32)[mesh.faces[face_idx]][:, :, :3]
                    * bary[:, :, None]).sum(axis=1)
            return rgba / 255.0
    except Exception:
        pass
    try:
        material = getattr(visual, "material", None)
        base = getattr(material, "baseColorFactor", None) or getattr(
            material, "diffuse", None)
        if base is not None:
            rgb = np.asarray(base, np.float32)[:3]
            rgb = rgb / 255.0 if rgb.max() > 1.0 else rgb
            return np.tile(rgb, (n, 1))
    except Exception:
        pass
    return np.full((n, 3), 0.5, np.float32)


def _bake_texture_to_vertex_color(g):
    """If a part carries a UV texture, sample it to per-vertex colors up front.
    Some GLB PBR materials expose a baseColorTexture that the barycentric UV
    lookup misses (the base color factor is then a flat white/grey), so the
    asset renders washed-out/near-transparent. ``visual.to_color()`` bakes the
    embedded texture into vertex colors, which _sample_colors uses directly."""
    vis = getattr(g, "visual", None)
    if vis is not None and getattr(vis, "uv", None) is not None and hasattr(vis, "to_color"):
        try:
            g.visual = vis.to_color()
        except Exception:
            pass
    return g


def _load_mesh_list(path: Path):
    """Load an input file into a list of textured trimesh.Trimesh parts."""
    import trimesh
    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        parts = []
        # Map graph NODES -> geometry (node names != geometry names in many
        # GLBs; `graph.get(geometry_name)` then raises "No path from world->...").
        try:
            for node in loaded.graph.nodes_geometry:
                # trimesh ships no stubs for SceneGraph.__getitem__, whose
                # inferred element type is an unbound TypeVar.
                graph_entry: Any = loaded.graph[node]
                transform, geom_name = graph_entry
                geom = loaded.geometry.get(geom_name)
                if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                    continue
                g = geom.copy()
                if transform is not None:
                    g.apply_transform(transform)
                parts.append(_bake_texture_to_vertex_color(g))
        except Exception:
            parts = []
        if not parts:  # fallback: raw geometry without graph transforms
            parts = [_bake_texture_to_vertex_color(g.copy())
                     for g in loaded.geometry.values()
                     if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not parts:
            raise ValueError(f"no triangle geometry in {path}")
        return parts
    if isinstance(loaded, trimesh.Trimesh):
        return [_bake_texture_to_vertex_color(loaded)]
    raise ValueError(f"unsupported mesh container {type(loaded)!r} in {path}")


def mesh_to_gaussians(
    meshes: Sequence,
    n_points: int = 60_000,
    y_up_to_z_up: bool = False,
    target_height: Optional[float] = None,
    scale_factor: float = 0.7,
    thickness_ratio: float = 0.15,
    opacity: float = 0.98,
    sh_degree: int = 3,
    seed: int = 0,
) -> dict:
    """Surface-sample ``meshes`` into 3DGS parameters (dict of arrays).

    Returns ``{"xyz", "normals", "f_dc", "f_rest", "opacity", "scale",
    "rot"}`` with the standard storage activations applied (log-scale,
    logit-opacity, SH DC colors) — ready for :func:`write_3dgs_ply`.
    """
    import trimesh
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    areas = np.array([float(m.area) for m in meshes])
    counts = np.maximum(1, np.round(n_points * areas / areas.sum()).astype(int))

    pts_all, nrm_all, rgb_all = [], [], []
    for mesh, cnt in zip(meshes, counts):
        pts, fidx = trimesh.sample.sample_surface(
            mesh, int(cnt), seed=int(rng.integers(2**31)))
        nrm = mesh.face_normals[fidx]
        rgb = _sample_colors(mesh, pts, fidx)
        pts_all.append(np.asarray(pts, np.float64))
        nrm_all.append(np.asarray(nrm, np.float64))
        rgb_all.append(rgb)
    xyz = np.concatenate(pts_all)
    normals = np.concatenate(nrm_all)
    rgb = np.concatenate(rgb_all)

    if y_up_to_z_up:
        # glTF y-up -> z-up: (x, y, z) -> (x, -z, y)
        xyz = xyz[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
        normals = normals[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])

    # Normalize placement: footprint centre at the origin, base on z=0.
    xyz -= [(xyz[:, 0].min() + xyz[:, 0].max()) / 2.0,
            (xyz[:, 1].min() + xyz[:, 1].max()) / 2.0,
            xyz[:, 2].min()]
    if target_height is not None and xyz[:, 2].max() > 0:
        xyz *= float(target_height) / float(xyz[:, 2].max())

    # Splat size from sampling density: mean 4-NN distance per point, padded
    # so neighbouring pancakes overlap into a hole-free surface.
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=min(5, len(xyz)))
    nn = dists[:, 1:].mean(axis=1) if dists.ndim == 2 and dists.shape[1] > 1 \
        else np.full(len(xyz), 0.01)
    s_tangent = np.maximum(nn * scale_factor, 1e-4)
    scale = np.stack([s_tangent, s_tangent,
                      np.maximum(s_tangent * thickness_ratio, 1e-5)], axis=1)

    n_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    return {
        "xyz": xyz.astype(np.float32),
        "normals": np.zeros_like(xyz, np.float32),  # INRIA files carry zeros
        "f_dc": ((rgb - 0.5) / SH_C0).astype(np.float32),
        "f_rest": np.zeros((len(xyz), n_rest), np.float32),
        "opacity": _logit(np.full(len(xyz), opacity, np.float32)),
        "scale": np.log(scale).astype(np.float32),
        "rot": _quat_z_to(normals).astype(np.float32),
    }


def write_3dgs_ply(gaussians: dict, out_path: Path) -> None:
    """Write the standard INRIA-layout binary 3DGS PLY (pure numpy, no deps)."""
    n = len(gaussians["xyz"])
    n_rest = gaussians["f_rest"].shape[1]
    fields = list(_BASE_FIELDS)
    fields += [f"f_dc_{i}" for i in range(3)]
    fields += [f"f_rest_{i}" for i in range(n_rest)]
    fields += ["opacity"] + [f"scale_{i}" for i in range(3)]
    fields += [f"rot_{i}" for i in range(4)]

    flat = np.concatenate([
        gaussians["xyz"], gaussians["normals"], gaussians["f_dc"],
        gaussians["f_rest"], gaussians["opacity"][:, None],
        gaussians["scale"], gaussians["rot"]], axis=1).astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {f}\n" for f in fields)
        + "end_header\n"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(flat).tobytes())


def align_length_to_x(gaussians: dict) -> dict:
    """Rotate the z-up build so the longest horizontal extent lies along +x.

    The insertion path treats asset-local +x as the track's FORWARD axis
    (``heading`` spins the asset about z), but GLB sources orient their
    length arbitrarily in the ground plane. Vehicles authored length-along-y
    would render sideways across the lane. Symmetric assets (cones) have
    equal extents and pass through untouched.
    """
    g = dict(gaussians)
    xyz = g["xyz"]
    x_ext = xyz[:, 0].max() - xyz[:, 0].min()
    y_ext = xyz[:, 1].max() - xyz[:, 1].min()
    if y_ext <= x_ext * 1.05:  # already length-along-x (or symmetric)
        return g
    # Rz(-90): (x, y, z) -> (y, -x, z)
    g["xyz"] = np.stack([xyz[:, 1], -xyz[:, 0], xyz[:, 2]], axis=1)
    c = s = np.float32(np.sqrt(0.5))  # (w, x, y, z) = (c, 0, 0, -s)
    qw, qx, qy, qz = (g["rot"][:, i] for i in range(4))
    g["rot"] = np.stack([
        c * qw + s * qz,
        c * qx + s * qy,
        c * qy - s * qx,
        c * qz - s * qw,
    ], axis=1).astype(np.float32)
    return g


def rotate_about_z(gaussians: dict, deg: float) -> dict:
    """Yaw the z-up build about the vertical axis by ``deg`` degrees (xyz +
    per-gaussian quaternions). Used to set which way a facing asset (e.g. a
    sign plate) points relative to the +x forward axis."""
    if not deg:
        return gaussians
    g = dict(gaussians)
    th = np.deg2rad(deg)
    ct, st = np.float32(np.cos(th)), np.float32(np.sin(th))
    xyz = g["xyz"]
    g["xyz"] = np.stack([ct * xyz[:, 0] - st * xyz[:, 1],
                         st * xyz[:, 0] + ct * xyz[:, 1], xyz[:, 2]], axis=1)
    c, s = np.float32(np.cos(th / 2)), np.float32(np.sin(th / 2))
    qw, qx, qy, qz = (g["rot"][:, i] for i in range(4))
    g["rot"] = np.stack([c * qw - s * qz, c * qx - s * qy,
                         c * qy + s * qx, c * qz + s * qw], axis=1).astype(np.float32)
    return g


def zup_to_yup(gaussians: dict) -> dict:
    """Rotate a z-up gaussian dict to the server's y-up file convention.

    ``insert_asset`` rotates every loaded PLY by a fixed y-up -> z-up matrix
    ``(x, y, z)_file -> (x, -z, y)``, so the file must store height along +y
    for the asset to stand upright after insertion. Positions map as
    ``(x, y, z) -> (x, z, -y)`` (= Rx(-90 deg)); per-gaussian quaternions are
    pre-multiplied by the same rotation; eigen-scales are frame-local and
    stay untouched.
    """
    g = dict(gaussians)
    xyz = g["xyz"]
    g["xyz"] = np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=1)
    c = s = np.float32(np.sqrt(0.5))  # Rx(-90): (w, x, y, z) = (c, -s, 0, 0)
    qw, qx, qy, qz = (g["rot"][:, i] for i in range(4))
    g["rot"] = np.stack([
        c * qw + s * qx,
        c * qx - s * qw,
        c * qy + s * qz,
        c * qz - s * qy,
    ], axis=1).astype(np.float32)
    return g


def make_procedural_traffic_cone(
    height: float = 0.7, base_half: float = 0.2
) -> list:
    """Textureless standard traffic cone: orange body + white band + base.

    Built from trimesh primitives with vertex colors (z-up, base at z=0), so
    a smoke-test asset exists without downloading anything.
    """
    import trimesh

    def _paint(mesh, rgb):
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile(np.array(rgb + (255,), np.uint8),
                                        (len(mesh.vertices), 1)))
        return mesh

    orange, white, dark = (255, 96, 0), (245, 245, 245), (60, 60, 60)
    base_h = 0.03 * height / 0.7
    body_h = height - base_h
    # Square base plate.
    base = trimesh.creation.box(bounds=[[-base_half, -base_half, 0.0],
                                        [base_half, base_half, base_h]])
    # Truncated cone body (linearly tapered cylinder sections), with a white
    # reflective band between 45% and 65% of the body height.
    sections = [(0.00, 0.45, orange), (0.45, 0.65, white), (0.65, 1.00, orange)]
    r_bot, r_top = 0.72 * base_half, 0.12 * base_half
    parts = [_paint(base, dark)]
    for f0, f1, rgb in sections:
        r0 = r_bot + (r_top - r_bot) * f0
        r1 = r_bot + (r_top - r_bot) * f1
        # Truncated cone: revolve the slanted profile segment around z.
        seg = trimesh.creation.revolve(
            [[r0, 0.0], [r1, (f1 - f0) * body_h]], sections=48)
        seg.apply_translation([0, 0, base_h + f0 * body_h])
        parts.append(_paint(seg, rgb))
    return parts


def make_procedural_sign(height: float = 2.0, plate: float = 0.6) -> list:
    """Textureless roadside warning sign: gray post + dark-bordered orange
    square plate (z-up, base at z=0). The plate normal is +x (the insertion
    path's forward axis), so it reads face-on as the ego approaches along the
    route. A smoke-test 'sign' asset without downloading anything.
    """
    import trimesh

    def _paint(mesh, rgb):
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile(np.array(rgb + (255,), np.uint8),
                                        (len(mesh.vertices), 1)))
        return mesh

    gray, orange, dark = (105, 105, 110), (255, 140, 0), (25, 25, 25)
    hw = plate / 2.0
    post_h = max(0.2, height - plate)          # plate stacks on top -> total = height
    half = 0.04
    post = trimesh.creation.box(
        bounds=[[-half, -half, 0.0], [half, half, post_h + 0.05]])
    # Plate faces +/-x (thin in x). Dark border box slightly larger in y/z, the
    # orange face box slightly thicker in x so it shows on both sides.
    border = trimesh.creation.box(
        bounds=[[-0.03, -hw - 0.03, -hw - 0.03], [0.03, hw + 0.03, hw + 0.03]])
    face = trimesh.creation.box(
        bounds=[[-0.045, -hw, -hw], [0.045, hw, hw]])
    for m in (border, face):
        m.apply_translation([0.0, 0.0, post_h + hw])
    return [_paint(post, gray), _paint(border, dark), _paint(face, orange)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert a mesh (GLB/OBJ/...) to a 3DGS PLY asset for "
                    "NuRec edit_assets insertion.")
    ap.add_argument("input", nargs="?", default=None,
                    help="Input mesh file (omit with --procedural-cone).")
    ap.add_argument("output", help="Output .ply path.")
    ap.add_argument("--points", type=int, default=60_000,
                    help="Number of Gaussians to sample (default 60000).")
    ap.add_argument("--target-height", type=float, default=None,
                    help="Rescale so the asset is this tall in metres "
                         "(e.g. 0.7 for a cone; UrbanVerse annotations.json "
                         "carries the real height).")
    ap.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3],
                    help="SH degree for the f_rest layout (default 3 = "
                         "f_rest_0..44, the standard INRIA file).")
    ap.add_argument("--no-yup-rotation", action="store_true",
                    help="Skip the glTF y-up -> z-up rotation (for inputs "
                         "already z-up).")
    ap.add_argument("--scale-factor", type=float, default=0.7,
                    help="Splat radius as a fraction of the mean 4-NN "
                         "distance (default 0.7).")
    ap.add_argument("--opacity", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procedural-cone", action="store_true",
                    help="Ignore input; generate a standard orange traffic "
                         "cone (0.7 m unless --target-height).")
    ap.add_argument("--procedural-sign", action="store_true",
                    help="Ignore input; generate a roadside warning sign — "
                         "orange plate on a gray post (2.0 m unless "
                         "--target-height). Use --forward-axis keep so the "
                         "plate stays face-on to the route.")
    ap.add_argument("--forward-axis", choices=["auto", "keep"], default="auto",
                    help="'auto' (default) rotates the build so the longest "
                         "horizontal extent lies along +x — the insertion "
                         "path's FORWARD axis (track heading). 'keep' trusts "
                         "the source orientation. Symmetric assets are "
                         "unaffected by 'auto'.")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="Extra yaw (deg) about the vertical axis, applied after "
                         "--forward-axis. Use to face a plate asset (sign) toward "
                         "the route: e.g. --yaw-deg 90 turns a plate that auto "
                         "aligned edge-on into face-on.")
    ap.add_argument("--out-up-axis", choices=["y", "z"], default="y",
                    help="File up-axis. 'y' (default) matches the server's "
                         "insert_asset convention (it rotates y-up -> z-up "
                         "on load); 'z' writes the raw z-up build, e.g. for "
                         "external 3DGS viewers.")
    args = ap.parse_args(argv)

    if args.procedural_cone:
        meshes = make_procedural_traffic_cone(
            height=args.target_height or 0.7)
        y2z = False   # built z-up
        target_height = None  # already exact
    elif args.procedural_sign:
        meshes = make_procedural_sign(height=args.target_height or 2.0)
        y2z = False   # built z-up
        target_height = None  # already exact
    else:
        if not args.input:
            ap.error("input mesh required unless --procedural-cone/--procedural-sign")
        path = Path(args.input)
        meshes = _load_mesh_list(path)
        y2z = (path.suffix.lower() in (".glb", ".gltf")
               and not args.no_yup_rotation)
        target_height = args.target_height

    g = mesh_to_gaussians(
        meshes, n_points=args.points, y_up_to_z_up=y2z,
        target_height=target_height, scale_factor=args.scale_factor,
        opacity=args.opacity, sh_degree=args.sh_degree, seed=args.seed)
    if args.forward_axis == "auto":
        g = align_length_to_x(g)
    if args.yaw_deg:
        g = rotate_about_z(g, args.yaw_deg)
    if args.out_up_axis == "y":
        g = zup_to_yup(g)
    write_3dgs_ply(g, Path(args.output))

    xyz = g["xyz"]
    up = {"y": 1, "z": 2}[args.out_up_axis]
    print(f"[convert_mesh_to_3dgs] wrote {len(xyz)} gaussians -> {args.output} "
          f"({args.out_up_axis}-up)")
    print(f"  bbox x [{xyz[:, 0].min():+.3f}, {xyz[:, 0].max():+.3f}] "
          f"y [{xyz[:, 1].min():+.3f}, {xyz[:, 1].max():+.3f}] "
          f"z [{xyz[:, 2].min():+.3f}, {xyz[:, 2].max():+.3f}] m "
          f"(origin = footprint centre, base at {'yz'[up - 1]}=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
