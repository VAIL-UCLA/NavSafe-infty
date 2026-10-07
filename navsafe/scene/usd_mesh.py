# Copyright (c) 2022-2025, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for flat (ground-plane) procedural USD meshes.

Single place that encodes the **local-origin + xform-translate invariant** the
assets backend depends on: a procedural ground/road/sidewalk mesh authored at
absolute world coordinates (these scenes sit ~km from the origin) makes RTX lose
float precision in the texture-MIP derivative, so an object-space OmniPBR texture
renders flat in the near field. Authoring the points **local to a centred
origin** and placing the mesh with an xform translate keeps object-space UVs
precise everywhere. This helper also factors out the common flat-mesh attribute
boilerplate (up normals, no subdivision, double-sided) that was duplicated across
the ground / road / sidewalk builders.

``pxr`` is imported lazily so the module loads where ``pxr`` is a stub (tests).
"""

from __future__ import annotations


def centroid_xy(points) -> tuple:
    """Mean ``(x, y)`` of ``(x, y, z)`` points (``(0, 0)`` for empty).

    Pure (no pxr) so callers can compute a local origin and unit-test it.
    """
    pts = list(points)
    if not pts:
        return 0.0, 0.0
    n = len(pts)
    return (sum(float(p[0]) for p in pts) / n,
            sum(float(p[1]) for p in pts) / n)


def define_flat_mesh(stage, prim_path, points, face_counts, face_indices, *,
                     translate=(0.0, 0.0, 0.0), display_color=None,
                     double_sided=True):
    """Define a ``UsdGeom.Mesh`` from **local** ``points`` + an xform translate.

    ``points`` are local to the mesh origin; ``translate`` places it in the
    world. Keeping points local (not absolute world coords) is what preserves
    object-space texture precision — see the module docstring. Returns the mesh.
    """
    from pxr import UsdGeom, Gf  # lazy
    pts = list(points)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pts])
    mesh.CreateFaceVertexCountsAttr(list(face_counts))
    mesh.CreateFaceVertexIndicesAttr(list(face_indices))
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * len(pts))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr(bool(double_sided))
    if display_color is not None:
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
    tz = float(translate[2]) if len(translate) > 2 else 0.0
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(
        Gf.Vec3d(float(translate[0]), float(translate[1]), tz))
    return mesh
