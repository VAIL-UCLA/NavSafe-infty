# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Give an inserted pedestrian a real gait instead of a rigid slide.

**The problem.** A NavSafe pedestrian is inserted into the reconstruction as one
static 3D-gaussian asset (``metadata.nurec_asset_id``), and the NRE render
server's per-frame API offers exactly one lever per actor: a rigid 6-DoF pose
(``DynamicObject.pose_pair``).  Geometry is fixed at ``edit_assets`` time.  So a
crossing pedestrian slides across the road in the frozen posture the Asset
Harvester captured -- the legs never move, whatever the social-force policy
does.  There is no skeleton channel in the proto to fix this with.

**The fix.** If limb motion cannot be sent per frame, it has to be *baked into
geometry* and the geometry swapped per frame.  This module bakes the bank; the
renderer (``navsafe/render/nurec_grpc.py``) inserts its phases as separate
server tracks and shows one at a time, parking the rest off-screen.

Three steps, each of which is checkable on its own:

``fit_pose``
    Recover the pose the harvested asset was captured in, by fitting the SOMA
    skinned body to its gaussian cloud.  Harvested pedestrians are captured
    mid-stride, so this is not a rest pose and cannot be recovered from
    anthropometric bands -- but it *is* a walking pose, which is exactly the
    manifold the walk motion sweeps, so the search initialises from the
    motion's own frames and refines from there.

``bind_gaussians``
    Push every gaussian back into the body's bind space and give it the linear
    blend weights of the body surface nearest it.  After this the asset is
    rigged: it can be posed to any frame of any motion on the same skeleton.

``bake_bank``
    Pose it to K frames spanning one gait cycle and write K PLYs.  Root travel
    and heading drift are removed relative to the fitted frame, so the bank
    holds *limb motion only* -- where the pedestrian actually is stays the
    policy's business, exactly as before.  Phase 0 is the fitted frame, so it
    reproduces the source asset and gives the bake a self-check.

Everything is expressed in the asset's own file frame (y-up, +x forward, feet
at y=0), so a baked phase is a drop-in replacement for the asset it came from
and inherits its dims, its scale convention and its registry entry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from navsafe.benchmark import config as cfg
from navsafe.benchmark.editing.assets.ply_io import (
    PlyError,
    ply_sha256,
    read_3dgs_ply,
    write_3dgs_ply,
)

logger = logging.getLogger(__name__)

# The motion's 30-joint skeleton in its own joint order, as
# ``SOMASkeleton30.bone_order_names_with_parents``.  Every name is also a joint
# of the 77-joint skinned rig, so the two are related by name and nothing has
# to be inferred from index arithmetic.
NAMES30 = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)

# Gaussians this faint are the cloud's haze, not its surface; including them in
# the fit pulls the body outward into the halo.
OPACITY_MIN = 0.35


class RigError(PlyError):
    """The rig, the motion, or a fit is not usable."""


# --------------------------------------------------------------------- rig
@dataclass
class Rig:
    """The skinned body the gait is applied through.

    ``bind`` is each joint's global transform in the bind pose, so
    ``inv(bind[j])`` is what takes a bind-space point into joint j's local
    frame -- the standard linear-blend-skinning inverse-bind matrix.
    """

    vertices: np.ndarray          # (V, 3) bind-pose body surface
    bind: np.ndarray              # (J, 4, 4) joint -> world at bind
    names: List[str]              # (J,)
    parent: np.ndarray            # (J,) -1 at the root
    lbs_idx: np.ndarray           # (V, 8) joint indices
    lbs_w: np.ndarray             # (V, 8) weights, rows sum to 1
    order: np.ndarray = field(init=False)   # joints, parents first

    def __post_init__(self) -> None:
        depth = np.array([self._depth(j) for j in range(len(self.names))])
        self.order = np.argsort(depth, kind="stable")

    def _depth(self, j: int) -> int:
        d = 0
        while self.parent[j] >= 0:
            j, d = int(self.parent[j]), d + 1
        return d

    @property
    def index(self) -> Dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    @property
    def motion_joints(self) -> np.ndarray:
        """Rig indices of the 30 joints the motion carries, in motion order."""
        idx = self.index
        return np.array([idx[n] for n in NAMES30], np.int64)


def load_rig(path: Path | None = None) -> Rig:
    """Load the SOMA skinned body shipped inside the kimodo checkout."""
    path = Path(path or cfg.SOMA_SKIN)
    if not path.is_file():
        raise RigError(
            f"no skinned body at {path}. It ships inside the kimodo checkout; "
            f"point NAVSAFE_KIMODO at one (currently {cfg.KIMODO}).")
    z = np.load(path, allow_pickle=True)
    names = [str(s) for s in z["rig_joint_names"]]
    parent = np.full(len(names), -1, np.int64)
    for p, c in z["rig_joint_connections"]:
        parent[int(c)] = int(p)
    return Rig(
        vertices=z["bind_vertices"].astype(np.float64),
        bind=z["bind_rig_transform"].astype(np.float64),
        names=names,
        parent=parent,
        lbs_idx=z["lbs_indices"].astype(np.int64),
        lbs_w=z["lbs_weights"].astype(np.float64),
    )


@dataclass
class Motion:
    """Joint motion over time, however it was produced.

    Two shapes, because two things produce motion here. A generated human
    motion arrives as kimodo writes it: GLOBAL transforms for the 30 joints its
    skeleton names, which ``rest_locals`` maps onto the rig by name. A motion
    authored against a rig directly -- a procedural quadruped gait, where the
    joints have no agreed names to map by -- arrives as ``local_rot``,
    parent-relative rotations already in the rig's own joint order.
    """

    pos: np.ndarray               # (T, 30, 3) global joint positions
    rot: np.ndarray               # (T, 30, 3, 3) global joint rotations
    contact: np.ndarray | None    # (T, k) foot contacts, one column per foot
    fps: float = 30.0
    source: str = ""
    #: (T, J, 3, 3) parent-relative rotations in RIG order. When present these
    #: are used verbatim and ``pos``/``rot`` are ignored.
    local_rot: np.ndarray | None = None
    #: Metres of travel per gait cycle, when the producer already knows it.
    #: ``gait_cycle`` measures it from the root otherwise, which a motion with
    #: no root translation (an in-place cycle) cannot support.
    stride_m: float | None = None

    def __len__(self) -> int:
        return len(self.local_rot) if self.local_rot is not None else len(self.pos)


def load_motion(path: Path | None = None, *, fps: float = 30.0) -> Motion:
    """Load a kimodo motion NPZ (the CLI's and the demo's shared layout)."""
    path = Path(path or cfg.motion_npz())
    if not path.is_file():
        raise RigError(f"no motion at {path}")
    z = np.load(path, allow_pickle=True)
    for key in ("posed_joints", "global_rot_mats"):
        if key not in z.files:
            raise RigError(f"{path}: not a kimodo motion (no {key!r})")
    return Motion(
        pos=z["posed_joints"].astype(np.float64),
        rot=z["global_rot_mats"].astype(np.float64),
        contact=z["foot_contacts"] if "foot_contacts" in z.files else None,
        fps=fps,
        source=str(path),
    )


def rest_locals(rig: Rig, motion: Motion, frame: int) -> np.ndarray:
    """Parent-relative joint transforms at one motion frame. (J, 4, 4)

    The motion carries global transforms for 30 of the rig's 77 joints.  The
    other 47 -- finger links, the head tip, the toe tips -- keep the offset
    they have in the bind pose, which leaves them rigid with their parent
    rather than dangling at the origin.
    """
    J = len(rig.names)
    binv0 = np.linalg.inv(rig.bind)
    if motion.local_rot is not None:
        # Authored against this rig: compose the rotation onto the joint's own
        # rest offset. Nothing to map by name, and nothing to infer.
        L = np.empty((J, 4, 4))
        for j in range(J):
            p = int(rig.parent[j])
            rest = rig.bind[j] if p < 0 else binv0[p] @ rig.bind[j]
            d = np.eye(4)
            d[:3, :3] = motion.local_rot[frame, j]
            L[j] = rest @ d
        return L

    G = np.tile(np.eye(4), (J, 1, 1))
    carried = np.zeros(J, bool)
    for j30, name in enumerate(NAMES30):
        j = rig.index[name]
        G[j, :3, :3] = motion.rot[frame, j30]
        G[j, :3, 3] = motion.pos[frame, j30]
        carried[j] = True

    binv = np.linalg.inv(rig.bind)
    for j in rig.order:
        p = int(rig.parent[j])
        if carried[j] or p < 0:
            continue
        G[j] = G[p] @ (binv[p] @ rig.bind[j])
        carried[j] = True

    L = np.empty_like(G)
    for j in range(J):
        p = int(rig.parent[j])
        L[j] = G[j] if p < 0 else np.linalg.inv(G[p]) @ G[j]
    return L


def forward_kinematics(rig: Rig, locals_: np.ndarray,
                       delta: np.ndarray | None = None) -> np.ndarray:
    """Global joint transforms from parent-relative ones. (J, 4, 4)

    ``delta`` is a per-motion-joint axis-angle correction applied on top, and
    it propagates: rotating a shoulder carries the whole arm, which is the
    whole reason this is FK and not an in-place twist of each joint.
    """
    L = locals_.copy()
    if delta is not None:
        # One row per rig joint means it is expressed against THIS rig; the
        # shorter form is the human motion's 30 named joints, mapped by name.
        jm = (np.arange(len(rig.names)) if len(delta) == len(rig.names)
              else rig.motion_joints)
        L[jm, :3, :3] = L[jm, :3, :3] @ rodrigues(delta)
    G = np.empty_like(L)
    for j in rig.order:
        p = int(rig.parent[j])
        G[j] = L[j] if p < 0 else G[p] @ L[j]
    return G


def rodrigues(v: np.ndarray) -> np.ndarray:
    """Axis-angle rows to rotation matrices, safe at zero. (N, 3) -> (N, 3, 3)"""
    v = np.atleast_2d(np.asarray(v, np.float64))
    ang = np.sqrt((v * v).sum(-1, keepdims=True) + 1e-24)
    ax = v / ang
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    a = ang[..., None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def blend(rig: Rig, G: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-point blended skinning transform. (N, 8)+(N, 8) -> (N, 4, 4)"""
    M = G @ np.linalg.inv(rig.bind)
    return np.einsum("nk,nkij->nij", w, M[idx])


def apply(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply per-point affine transforms to per-point positions."""
    return np.einsum("nij,nj->ni", A[:, :3, :3], p) + A[:, :3, 3]


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def orthonormal(M: np.ndarray) -> np.ndarray:
    """Nearest rotation to each 3x3 block, by polar decomposition.

    Blend skinning produces a small shear near joints; a gaussian is stored as
    a quaternion plus scales and has nowhere to put it, so the rotation part is
    what it gets.
    """
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    flip = np.linalg.det(R) < 0
    if np.any(flip):
        u[flip, :, -1] *= -1.0
        R = u @ vt
    return R


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) quaternions in (w, x, y, z), matching ply_io."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(m), 4))
    big = t > 0
    s = np.sqrt(np.maximum(t[big] + 1.0, 1e-12)) * 2.0
    q[big, 0] = 0.25 * s
    q[big, 1] = (m[big, 2, 1] - m[big, 1, 2]) / s
    q[big, 2] = (m[big, 0, 2] - m[big, 2, 0]) / s
    q[big, 3] = (m[big, 1, 0] - m[big, 0, 1]) / s
    rest = ~big
    if np.any(rest):
        idx = np.argmax(np.stack([m[rest, 0, 0], m[rest, 1, 1], m[rest, 2, 2]], 1), 1)
        sub = np.flatnonzero(rest)
        for k, row in zip(idx, sub):
            a, b, c = (k + 1) % 3, (k + 2) % 3, k
            s = np.sqrt(max(1.0 + m[row, c, c] - m[row, a, a] - m[row, b, b], 1e-12)) * 2.0
            q[row, 0] = (m[row, b, a] - m[row, a, b]) / s
            q[row, 1 + c] = 0.25 * s
            q[row, 1 + a] = (m[row, a, c] + m[row, c, a]) / s
            q[row, 1 + b] = (m[row, b, c] + m[row, c, b]) / s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """(N, 4) (w, x, y, z) -> (N, 3, 3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


# --------------------------------------------------------------------- fit
@dataclass
class Fit:
    """The pose, orientation and size that explain one harvested asset."""

    frame: int                    # motion frame the pose came from
    delta: np.ndarray             # (30, 3) per-joint axis-angle correction
    yaw: float                    # asset-frame yaw of the fitted body
    scale: float                  # body -> asset metric scale
    offset: np.ndarray            # (3,) translation into the asset's frame
    chamfer: float                # mean two-way surface distance, metres
    asset: str = ""
    #: The full body -> asset linear map, when yaw and scale cannot express it.
    #: A yaw is a rotation about the asset's UP axis, which is all a fit needs
    #: when the rig is already y-up -- SOMA is. SMAL is z-up, so its placement
    #: also turns the body upright, and rebuilding the map from yaw and scale
    #: alone drops that: the template ends up lying on its side, its feet
    #: somewhere in the torso, and the leg joints then own torso gaussians.
    #: Set this and the placement is used as solved instead of reconstructed.
    linear: np.ndarray | None = None

    def to_dict(self) -> dict:
        d = {"frame": int(self.frame), "delta": self.delta.tolist(),
             "yaw": float(self.yaw), "scale": float(self.scale),
             "offset": self.offset.tolist(), "chamfer": float(self.chamfer),
             "asset": self.asset}
        if self.linear is not None:
            d["linear"] = np.asarray(self.linear, np.float64).tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Fit":
        lin = d.get("linear")
        return Fit(frame=int(d["frame"]), delta=np.asarray(d["delta"], np.float64),
                   yaw=float(d["yaw"]), scale=float(d["scale"]),
                   offset=np.asarray(d["offset"], np.float64),
                   chamfer=float(d["chamfer"]), asset=str(d.get("asset", "")),
                   linear=None if lin is None else np.asarray(lin, np.float64))


def surface_points(gaussians: Dict[str, np.ndarray], *,
                   opacity_min: float = OPACITY_MIN,
                   max_points: int = 6000, seed: int = 0) -> np.ndarray:
    """The opaque core of a gaussian asset, subsampled, as a point cloud."""
    xyz = np.asarray(gaussians["xyz"], np.float64)
    opa = 1.0 / (1.0 + np.exp(-np.asarray(gaussians["opacity"], np.float64).reshape(-1)))
    pts = xyz[opa > opacity_min]
    if len(pts) < 200:
        raise RigError(
            f"only {len(pts)} gaussians above opacity {opacity_min}; the asset "
            f"is too faint to fit a body to")
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


def _ground_frame(pts: np.ndarray) -> np.ndarray:
    """Horizontal centre, feet on the floor -- the frame a fit is scored in."""
    return np.array([pts[:, 0].mean(), pts[:, 1].min(), pts[:, 2].mean()])


def prune_gaussians(gaussians: Dict[str, np.ndarray],
                    opacity_min: float) -> Dict[str, np.ndarray]:
    """Drop the faintest gaussians before a bank is baked.

    A pose bank is the same asset K times over, so its cost on the render
    server is K times the asset's -- and a 24 GB card holding four
    reconstructions has no room for that at full density (measured: six
    pedestrians at ten phases is 2.8 M gaussians and the render OOMs).  The
    faint tail is where the redundancy is cheapest to pay for: these gaussians
    are the cloud's haze, they are the ones a viewer at 20 m cannot resolve,
    and dropping them costs density rather than shape.

    Pruning happens ONCE, before binding, so every phase drops the same
    gaussians and the bank stays internally consistent.
    """
    if opacity_min <= 0.0:
        return gaussians
    opa = 1.0 / (1.0 + np.exp(-np.asarray(gaussians["opacity"], np.float64).reshape(-1)))
    keep = opa > opacity_min
    if keep.sum() < 500:
        raise RigError(
            f"pruning at opacity {opacity_min} leaves {int(keep.sum())} "
            f"gaussians — too few to look like anything")
    out = {}
    for k, v in gaussians.items():
        arr = np.asarray(v)
        out[k] = arr[keep] if arr.ndim >= 1 and len(arr) == len(keep) else arr
    logger.info("prune_gaussians: kept %d of %d above opacity %.2f (%.0f%%)",
                int(keep.sum()), len(keep), opacity_min,
                100.0 * keep.sum() / len(keep))
    return out


def fit_pose(gaussians: Dict[str, np.ndarray], rig: Rig, motion: Motion, *,
             candidates: int = 4, iters: int = 600, yaw_steps: int = 48,
             frame_stride: int = 2, reg: float = 0.05, seed: int = 0,
             device: str | None = None) -> Fit:
    """Recover the pose a harvested pedestrian was captured in.

    Coarse stage scores every (motion frame, yaw) pair by two-way Chamfer
    distance; the refine stage runs Adam over yaw, scale, shift and per-joint
    rotation from each of the top few coarse basins.  Several candidates are
    refined rather than one because a walking body is close to symmetric
    front-to-back at Chamfer resolution, so the coarse winner is regularly the
    mirrored yaw.
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cloud = surface_points(gaussians, seed=seed)
    cloud_c = cloud - _ground_frame(cloud)
    height = float(cloud[:, 1].max() - cloud[:, 1].min())

    # Every frame of the motion, posed, in the same ground frame.
    posed = _posed_meshes(rig, motion)

    coarse = _coarse_candidates(cloud_c, posed, height, candidates,
                                yaw_steps, frame_stride, dev, seed)
    best: Fit | None = None
    for _score, frame, yaw, scale in coarse:
        cand = _refine(cloud_c, rig, motion, frame, yaw, scale,
                       iters=iters, reg=reg, seed=seed, device=dev)
        logger.info("fit_pose: frame=%d yaw=%.1f -> chamfer=%.4f",
                    frame, np.degrees(yaw), cand.chamfer)
        if best is None or cand.chamfer < best.chamfer:
            best = cand
    assert best is not None
    # The fit was solved against a ground-framed cloud; carry the frame back so
    # the transform lands in the asset's own coordinates.
    best.offset = best.offset + _ground_frame(cloud)
    return best


def _posed_meshes(rig: Rig, motion: Motion) -> np.ndarray:
    """(T, V, 3) -- the body at every motion frame, horizontally centred with
    the feet on y=0, which is the frame a harvested asset is authored in."""
    T = len(motion)
    out = np.empty((T, len(rig.vertices), 3))
    for t in range(T):
        G = forward_kinematics(rig, rest_locals(rig, motion, t))
        A = blend(rig, G, rig.lbs_idx, rig.lbs_w)
        out[t] = apply(A, rig.vertices)
    out[..., 0] -= out[..., 0].mean(1, keepdims=True)
    out[..., 2] -= out[..., 2].mean(1, keepdims=True)
    out[..., 1] -= out[..., 1].min(1, keepdims=True)
    return out


def _chamfer(a, b):
    import torch
    d = torch.cdist(a, b)
    return d.min(1).values.mean() + d.min(0).values.mean()


def _yaw_mat_t(theta):
    import torch
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, z, s], -1),
                        torch.stack([z, o, z], -1),
                        torch.stack([-s, z, c], -1)], -2)


def _coarse_candidates(cloud, posed, height, n, yaw_steps, frame_stride,
                       dev, seed, n_mesh=700, n_cloud=700):
    import torch

    rng = np.random.default_rng(seed)
    mi = rng.choice(posed.shape[1], n_mesh, replace=False)
    ci = rng.choice(len(cloud), min(n_cloud, len(cloud)), replace=False)
    C = torch.tensor(cloud[ci], dtype=torch.float32, device=dev)
    M = torch.tensor(posed[:, mi], dtype=torch.float32, device=dev)
    yaws = torch.linspace(0, 2 * np.pi, yaw_steps + 1, device=dev)[:-1]
    R = _yaw_mat_t(yaws)
    axis = torch.tensor([1.0, 0.0, 1.0], device=dev)

    scored = []
    for t in range(0, M.shape[0], frame_stride):
        m = M[t]
        s = height / float(m[:, 1].max() - m[:, 1].min())
        rot = torch.einsum("yij,nj->yni", R, m * s)
        rot = rot - rot.mean(1, keepdim=True) * axis
        for y in range(len(yaws)):
            scored.append((_chamfer(rot[y], C).item(), t, float(yaws[y]), s))
    scored.sort(key=lambda r: r[0])

    # Spread the candidates over distinct yaw basins: n near-identical yaws
    # would all fall into the same local minimum and test nothing.
    picked, used = [], []
    for c in scored:
        sep = [abs((np.degrees(c[2] - u) + 180) % 360 - 180) for u in used]
        if all(s > 40 for s in sep):
            picked.append(c)
            used.append(c[2])
        if len(picked) == n:
            break
    return picked or scored[:n]


def _refine(cloud, rig, motion, frame, yaw0, scale0, *, iters, reg, seed,
            device, n_mesh=2200, n_cloud=2200) -> Fit:
    import torch

    rng = np.random.default_rng(seed)
    ci = rng.choice(len(cloud), min(n_cloud, len(cloud)), replace=False)
    C = torch.tensor(cloud[ci], dtype=torch.float32, device=device)

    L0 = torch.tensor(rest_locals(rig, motion, frame), dtype=torch.float32, device=device)
    binv = torch.tensor(np.linalg.inv(rig.bind), dtype=torch.float32, device=device)
    jm = torch.tensor(rig.motion_joints, device=device)
    order = [int(j) for j in rig.order]
    parent = [int(p) for p in rig.parent]

    mi = rng.choice(len(rig.vertices), n_mesh, replace=False)
    V = torch.tensor(rig.vertices[mi], dtype=torch.float32, device=device)
    widx = torch.tensor(rig.lbs_idx[mi], device=device)
    ww = torch.tensor(rig.lbs_w[mi], dtype=torch.float32, device=device)

    delta = torch.zeros(len(NAMES30), 3, device=device, requires_grad=True)
    yaw = torch.tensor(float(yaw0), device=device, requires_grad=True)
    logs = torch.tensor(float(np.log(scale0)), device=device, requires_grad=True)
    shift = torch.zeros(3, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta, yaw, logs, shift], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

    loss = torch.tensor(float("nan"))
    for _ in range(iters):
        opt.zero_grad()
        L = _apply_delta_t(L0, jm, delta)
        G = _fk_t(L, order, parent)
        A = torch.einsum("vk,vkij->vij", ww, (G @ binv)[widx])
        p = torch.einsum("vij,vj->vi", A[:, :3, :3], V) + A[:, :3, 3]
        p = torch.einsum("ij,vj->vi", _yaw_mat_t(yaw), p * torch.exp(logs))
        p = p - torch.stack([p[:, 0].mean(), p[:, 1].min(), p[:, 2].mean()]) + shift
        loss = _chamfer(p, C) + reg * (delta ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        L = _apply_delta_t(L0, jm, delta)
        G = _fk_t(L, order, parent)
        Vf = torch.tensor(rig.vertices, dtype=torch.float32, device=device)
        Af = torch.einsum("vk,vkij->vij", torch.tensor(rig.lbs_w, dtype=torch.float32, device=device),
                          (G @ binv)[torch.tensor(rig.lbs_idx, device=device)])
        pf = torch.einsum("vij,vj->vi", Af[:, :3, :3], Vf) + Af[:, :3, 3]
        pf = torch.einsum("ij,vj->vi", _yaw_mat_t(yaw), pf * torch.exp(logs))
        centre = torch.stack([pf[:, 0].mean(), pf[:, 1].min(), pf[:, 2].mean()])
        offset = (shift - centre).cpu().numpy().astype(np.float64)

    return Fit(frame=int(frame), delta=delta.detach().cpu().numpy().astype(np.float64),
               yaw=float(yaw.item()), scale=float(np.exp(logs.item())),
               offset=offset, chamfer=float(loss.item()))


def _apply_delta_t(L0, jm, delta):
    import torch
    ang = torch.sqrt((delta * delta).sum(-1, keepdim=True) + 1e-12)
    ax = delta / ang
    K = torch.zeros(len(delta), 3, 3, device=delta.device)
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    I = torch.eye(3, device=delta.device).expand_as(K)
    a = ang[..., None]
    Rd = I + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)
    base = L0[jm]
    up = torch.cat([torch.cat([base[:, :3, :3] @ Rd, base[:, :3, 3:]], 2),
                    base[:, 3:, :]], 1)
    return L0.index_copy(0, jm, up)


def _fk_t(L, order, parent):
    import torch
    G: list = [None] * len(order)
    for j in order:
        p = parent[j]
        G[j] = L[j] if p < 0 else G[p] @ L[j]
    return torch.stack(G)


# -------------------------------------------------------------------- bind
@dataclass
class Rigged:
    """A harvested asset expressed in the body's bind space.

    ``xyz`` and ``rot`` are the gaussians pushed back through the fitted pose,
    so re-posing is a forward skinning pass and nothing about the fit has to be
    redone.  Everything else is carried through untouched.
    """

    gaussians: Dict[str, np.ndarray]   # the source, verbatim
    bind_xyz: np.ndarray               # (N, 3) positions in bind space
    bind_rot: np.ndarray               # (N, 3, 3) orientations in bind space
    lbs_idx: np.ndarray                # (N, 8)
    lbs_w: np.ndarray                  # (N, 8)
    fit: Fit


def fitted_transform(rig: Rig, motion: Motion, fit: Fit,
                     frame: int | None = None) -> np.ndarray:
    """Global joint transforms of the fitted body at a motion frame. (J, 4, 4)

    The fit's per-joint correction rides along on every frame: part of it is
    this person's own posture rather than fitting noise, and carrying it keeps
    them recognisably themselves through the whole cycle -- and makes the
    fitted frame reproduce the source asset exactly.
    """
    f = fit.frame if frame is None else frame
    return forward_kinematics(rig, rest_locals(rig, motion, f), fit.delta)


def _asset_affine(rig: Rig, G: np.ndarray, idx: np.ndarray, w: np.ndarray,
                  fit: Fit, pin: np.ndarray | None = None):
    """Per-gaussian bind-space -> asset-space affine.

    ``pin`` is the (rotation, translation) that cancels the root's travel and
    heading drift relative to the fitted frame; without it the whole bank would
    walk away from the actor's own position.
    """
    A = blend(rig, G, idx, w)
    lin, tr = A[:, :3, :3], A[:, :3, 3]
    if pin is not None:
        Rp, tp = pin
        lin = Rp @ lin
        tr = tr @ Rp.T + tp
    R = fit.linear if fit.linear is not None else rot_y(fit.yaw) * fit.scale
    return R @ lin, tr @ R.T + fit.offset


def _pin_to_fit(rig: Rig, motion: Motion, fit: Fit, frame: int):
    """Cancel root travel and heading drift between ``frame`` and the fit.

    Vertical motion is deliberately kept: a walking body rises and falls, and
    flattening that is what makes a baked walk read as a slideshow.
    """
    hips = int(np.flatnonzero(np.asarray(rig.parent) < 0)[0])   # the root joint
    G0 = fitted_transform(rig, motion, fit, fit.frame)
    Gu = fitted_transform(rig, motion, fit, frame)

    def heading(R):
        return float(np.arctan2(R[0, 2], R[2, 2]))

    d = heading(Gu[hips, :3, :3]) - heading(G0[hips, :3, :3])
    Rp = rot_y(-d)
    h0 = np.array([G0[hips, 0, 3], 0.0, G0[hips, 2, 3]])
    hu = np.array([Gu[hips, 0, 3], 0.0, Gu[hips, 2, 3]])
    return Rp, h0 - Rp @ hu


def bind_gaussians(gaussians: Dict[str, np.ndarray], rig: Rig, motion: Motion,
                   fit: Fit, *, neighbours: int = 4) -> Rigged:
    """Rig a harvested asset: bind-space geometry plus per-gaussian weights.

    Weights are taken from the fitted body surface nearest each gaussian,
    averaged over a few neighbours so the seam between two body parts is a
    blend rather than a step.  Gaussians with no body under them at all -- a
    carried bag, a backpack's outline -- inherit whatever part is closest and
    ride it rigidly, which is the right answer for something being carried.
    """
    from scipy.spatial import cKDTree

    G = fitted_transform(rig, motion, fit)
    # The fitted body surface, in the asset's own frame.
    lin_v, tr_v = _asset_affine(rig, G, rig.lbs_idx, rig.lbs_w, fit)
    surface = np.einsum("nij,nj->ni", lin_v, rig.vertices) + tr_v

    xyz = np.asarray(gaussians["xyz"], np.float64)
    dist, near = cKDTree(surface).query(xyz, k=neighbours)
    dist = np.atleast_2d(dist.T).T
    near = np.atleast_2d(near.T).T
    wn = 1.0 / np.maximum(dist, 1e-4)
    wn /= wn.sum(1, keepdims=True)

    # Blend the neighbours' skinning weights, then compact back to 8 joints.
    idx = rig.lbs_idx[near].reshape(len(xyz), -1)          # (N, k*8)
    wts = (rig.lbs_w[near] * wn[..., None]).reshape(len(xyz), -1)
    g_idx, g_w = _compact_weights(idx, wts, keep=rig.lbs_idx.shape[1])

    lin, tr = _asset_affine(rig, G, g_idx, g_w, fit)
    inv = np.linalg.inv(lin)
    bind_xyz = np.einsum("nij,nj->ni", inv, xyz - tr)
    src_rot = matrix_from_quat(np.asarray(gaussians["rot"], np.float64))
    bind_rot = np.einsum("nij,njk->nik", orthonormal(lin).transpose(0, 2, 1), src_rot)
    return Rigged(gaussians=gaussians, bind_xyz=bind_xyz, bind_rot=bind_rot,
                  lbs_idx=g_idx, lbs_w=g_w, fit=fit)


def _compact_weights(idx: np.ndarray, w: np.ndarray, *, keep: int):
    """Sum duplicate joints, keep the ``keep`` largest, renormalise."""
    n = len(idx)
    out_i = np.zeros((n, keep), np.int64)
    out_w = np.zeros((n, keep), np.float64)
    for r in range(n):
        acc: Dict[int, float] = {}
        for j, weight in zip(idx[r], w[r]):
            if weight > 0:
                acc[int(j)] = acc.get(int(j), 0.0) + float(weight)
        top = sorted(acc.items(), key=lambda kv: -kv[1])[:keep]
        total = sum(v for _, v in top) or 1.0
        for c, (j, weight) in enumerate(top):
            out_i[r, c], out_w[r, c] = j, weight / total
    return out_i, out_w


def pose_gaussians(rigged: Rigged, rig: Rig, motion: Motion,
                   frame: int) -> Dict[str, np.ndarray]:
    """The rigged asset posed to one motion frame, in the asset's own frame."""
    G = fitted_transform(rig, motion, rigged.fit, frame)
    pin = _pin_to_fit(rig, motion, rigged.fit, frame)
    lin, tr = _asset_affine(rig, G, rigged.lbs_idx, rigged.lbs_w, rigged.fit, pin)
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in rigged.gaussians.items()}
    out["xyz"] = (np.einsum("nij,nj->ni", lin, rigged.bind_xyz) + tr).astype(np.float32)
    R = np.einsum("nij,njk->nik", orthonormal(lin), rigged.bind_rot)
    out["rot"] = quat_from_matrix(R).astype(np.float32)
    return out


# -------------------------------------------------------------------- bake
def gait_cycle(motion: Motion) -> tuple[int, float]:
    """One gait cycle of a walk: its length in frames, and its stride in metres.

    Read off the left heel's contact signal -- the interval between successive
    heel strikes is one full cycle (two steps), and how far the root travels
    over it is the stride the phase clock indexes by.
    """
    if motion.stride_m is not None:
        # An in-place cycle authored against the rig: it has no root travel to
        # measure, and its producer already knows what one cycle covers.
        return len(motion), float(motion.stride_m)
    if motion.contact is None:
        raise RigError(f"{motion.source}: no foot contacts, so no gait cycle")
    heel = np.asarray(motion.contact)[:, 0].astype(bool)
    strikes = np.flatnonzero(~heel[:-1] & heel[1:]) + 1
    if len(strikes) < 2:
        raise RigError(
            f"{motion.source}: {len(strikes)} heel strike(s) — not a walk, or "
            f"too short to hold a cycle")
    period = int(round(float(np.median(np.diff(strikes)))))
    root = motion.pos[:, 0, :]
    steps = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1)
    stride = float(steps.mean() * period)
    return period, stride


def phase_frames(motion: Motion, start: int, phases: int) -> List[int]:
    """``phases`` frames evenly spanning one gait cycle from ``start``.

    A frame past the end of the motion is pulled back by one whole cycle: the
    walk is periodic to within the drift this bake removes anyway, so the limb
    pose is the same and the bank stays continuous.
    """
    period, _ = gait_cycle(motion)
    out = []
    for j in range(phases):
        f = start + int(round(j * period / phases))
        while f > len(motion) - 1:
            f -= period
        out.append(max(0, f))
    return out


def bake_bank(rigged: Rigged, rig: Rig, motion: Motion, out_dir: Path, *,
              phases: int = 10, dims: Sequence[float] | None = None,
              frames: Sequence[int] | None = None) -> dict:
    """Write one gait cycle of posed PLYs, plus the manifest describing it.

    The manifest is what the renderer reads: the phase files in order and the
    stride they cover, which is what lets the phase be clocked off distance
    travelled rather than off wall time.  Clocking by distance is what keeps
    the feet from skating when the policy runs the actor faster or slower than
    the motion was generated at.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    period, stride = gait_cycle(motion)
    # An explicit list is for a motion that IS one cycle already, where walking
    # out from the fitted frame would wrap through it rather than sample it.
    frames = list(frames) if frames is not None else phase_frames(
        motion, rigged.fit.frame, phases)

    files = []
    for j, f in enumerate(frames):
        posed = pose_gaussians(rigged, rig, motion, f)
        path = out_dir / f"ph{j:02d}.ply"
        write_3dgs_ply(posed, path)
        files.append(path.name)

    manifest = {
        "phases": files,
        # Each phase's own content hash, so hashing this manifest transitively
        # pins the geometry: a recipe records one sha256 for the whole bank and
        # a swapped PLY cannot hide behind an unchanged file list.
        "phases_sha256": [ply_sha256(out_dir / n) for n in files],
        "stride_m": round(stride, 4),
        "cycle_frames": period,
        "motion": motion.source,
        "motion_frames": [int(f) for f in frames],
        "fit": rigged.fit.to_dict(),
        "dims": list(dims) if dims is not None else None,
        "gaussians": int(len(rigged.bind_xyz)),
    }
    (out_dir / "bank.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("bake_bank: %d phases, stride %.3f m, cycle %d frames -> %s",
                len(files), stride, period, out_dir)
    return manifest


def bake_asset(ply: Path, out_dir: Path, *, phases: int = 10,
               rig: Rig | None = None, motion: Motion | None = None,
               dims: Sequence[float] | None = None,
               prune_opacity: float = 0.0, **fit_kw) -> dict:
    """Fit, bind and bake one harvested asset in a single call.

    ``prune_opacity`` trades the asset's faint tail for VRAM on the render
    server, which a pose bank spends K times over; see :func:`prune_gaussians`.
    The fit is done on the pruned cloud too, so what was fitted is what ships.
    """
    rig = rig or load_rig()
    motion = motion or load_motion()
    gaussians = prune_gaussians(read_3dgs_ply(ply), prune_opacity)
    fit = fit_pose(gaussians, rig, motion, **fit_kw)
    fit.asset = str(ply)
    logger.info("bake_asset: %s fitted at frame %d, chamfer %.4f m",
                Path(ply).name, fit.frame, fit.chamfer)
    rigged = bind_gaussians(gaussians, rig, motion, fit)
    manifest = bake_bank(rigged, rig, motion, out_dir, phases=phases, dims=dims)
    manifest["round_trip_m"] = round_trip_error(rigged, rig, motion)
    manifest["prune_opacity"] = float(prune_opacity)
    (Path(out_dir) / "bank.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def round_trip_error(rigged: Rigged, rig: Rig, motion: Motion) -> float:
    """How far phase 0 has moved from the source asset, in metres.

    Phase 0 is the fitted frame, so binding and re-posing must return the
    gaussians they started from.  Anything above a millimetre means the bind
    and the pose disagree about a transform, and every other phase is wrong in
    a way no picture would show.
    """
    posed = pose_gaussians(rigged, rig, motion, rigged.fit.frame)
    d = np.linalg.norm(np.asarray(posed["xyz"], np.float64)
                       - np.asarray(rigged.gaussians["xyz"], np.float64), axis=1)
    return float(d.max())
