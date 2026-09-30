# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Give a harvested animal a gait, the way ``animate.py`` gives a person one.

The machinery is shared -- fit a skinned template to the gaussian cloud, bind
the gaussians to it, bake a cycle of posed PLYs -- but both of the inputs that
made the pedestrian path cheap are missing for animals, and this module is what
supplies them.

**Template.** The kimodo checkout ships a skinned human; it ships no quadruped.
SMAL (*3D Menagerie*, CVPR 2017) is the equivalent: 3889 vertices, 33 joints,
41 shape coefficients, and five family means -- cats, dogs, horses, cows,
hippos.  It is a gated download and it is z-up where the asset library is
y-up, both handled here.  It also does not cover elephants, whose proportions
sit outside its shape space; that is a limitation of the template, not of this
code, and it is why ``family`` has to be chosen rather than guessed.

**Motion.** There is no quadruped motion in anything we ship either.
Retargeting one would first have to solve the correspondence to SMAL's 33
UNNAMED joints -- which is the same identification a generated gait needs, and
all a generated gait needs -- so the cycle is authored here instead, from the
two things that have to be right at road distance: the footfall order, and a
stride that matches how far the actor actually travels.

Joint semantics are read off SMAL's own geometry, since the model ships none:

    j0 pelvis --> j1..j6 spine --> j15,j16,j32 neck/head
                              |--> j7..j10   front left   (hip,knee,ankle,foot)
                              |--> j11..j14  front right
    j0        --> j17..j20 hind left, j21..j24 hind right, j25..j31 tail

Two properties of the generated cycle are solved rather than assumed, because
getting either wrong is what reads as fake from a moving car:

* **the stance foot is planted** -- the knee is solved per frame so the foot
  holds its height while the hip rotates over it, and the ankle undoes the
  accumulated rotation so the sole stays flat.  Swinging the hip alone arcs the
  foot through the road surface and back out.
* **the stride is measured, not derived** -- it is the foot's fore-aft travel
  through stance, taken from the motion that was actually generated.  The
  renderer clocks phase off ``travel / stride``, so a stride that disagrees
  with the pose IS a skating foot.

The stride a hip sweep can produce tops out near 1.2x leg length, which caps
how fast an actor may move before a bank of K phases aliases.  Past about a
quarter cycle per rendered frame the legs read as flicker; at 10 Hz that means
roughly ``0.25 * stride / dt`` metres per second.  A dart-out faster than that
needs a bound or gallop -- spine flexion and a flight phase -- which this
generator does not produce.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import logging
import pickle
import sys
import types
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from navsafe.benchmark import config as cfg
from navsafe.benchmark.editing.assets.animate import (
    Fit,
    Motion,
    Rig,
    RigError,
    bake_bank,
    bind_gaussians,
    prune_gaussians,
)
from navsafe.benchmark.editing.assets.ply_io import read_3dgs_ply

logger = logging.getLogger(__name__)

# SMAL's shape space, by the families its 41 toy scans were drawn from. An
# animal outside these is not merely a different beta -- the space does not
# reach it, which is the documented limitation that rules out elephants.
FAMILY = {"cat": 0, "dog": 1, "horse": 2, "cow": 3, "hippo": 4}

# proximal -> distal, per limb: (hip/shoulder, knee/elbow, ankle/wrist, foot)
FRONT_L = (7, 8, 9, 10)
FRONT_R = (11, 12, 13, 14)
HIND_L = (17, 18, 19, 20)
HIND_R = (21, 22, 23, 24)
LIMBS = (FRONT_L, FRONT_R, HIND_L, HIND_R)
SPINE = (1, 2, 3, 4, 5, 6)
LEG_JOINTS = tuple(sorted(j for limb in LIMBS for j in limb[:3]))

# Trot: the diagonal pairs strike together, so they are half a cycle apart.
TROT = {FRONT_L: 0.0, HIND_R: 0.0, FRONT_R: 0.5, HIND_L: 0.5}
# Walk: the lateral sequence, hind then fore on each side. What a heavy animal
# uses -- an elephant never leaves the ground, and this keeps three feet down.
WALK = {HIND_L: 0.0, FRONT_L: 0.25, HIND_R: 0.5, FRONT_R: 0.75}
GAITS = {"trot": TROT, "walk": WALK}

# The asset library is y-up with +x forward; SMAL is z-up with +x forward.
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [0.0, -1.0, 0.0]])


# ------------------------------------------------------------------ template
class _StubLoader(importlib.abc.Loader):
    """Absorbs the modules a 2017 Python-2 pickle references and we lack."""

    _made: Dict[str, type] = {}

    @classmethod
    def _cls(cls, name: str) -> type:
        if name not in cls._made:
            def __new__(c, *a, **k):
                return object.__new__(c)

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
                else:
                    self._state = state
            cls._made[name] = type(name, (), {"__new__": __new__,
                                              "__setstate__": __setstate__})
        return cls._made[name]

    def create_module(self, spec):
        m = types.ModuleType(spec.name)

        def getattr_(name):
            # Dunders must behave like a real module's: `inspect` walks
            # sys.modules and chokes on a class where it expects __file__.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return _StubLoader._cls(name)

        m.__getattr__ = getattr_
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    """Last-resort finder, APPENDED to ``sys.meta_path``.

    Anything really installed -- scipy, which the bind step needs -- resolves
    normally; only chumpy, the SMPL webuser package and the pre-1.8 scipy
    sparse module paths reach a stub.
    """

    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root in {"chumpy", "smpl_webuser", "cPickle", "posemapper", "verts", "lbs"}:
            return importlib.machinery.ModuleSpec(name, _StubLoader())
        if name.startswith("scipy.sparse.") and name.count(".") == 2:
            return importlib.machinery.ModuleSpec(name, _StubLoader())
        return None


def _install_stubs() -> None:
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder())


def _unwrap(o, depth: int = 0):
    """A chumpy array keeps its value in ``.x``; everything else is itself."""
    if isinstance(o, np.ndarray) or depth > 4:
        return o
    d = getattr(o, "__dict__", None)
    if d:
        for k in ("x", "_x", "r"):
            if k in d:
                return _unwrap(d[k], depth + 1)
    return o


def _dense(sparse) -> np.ndarray:
    """The stubbed scipy CSC matrix, as a dense array."""
    d = sparse.__dict__
    shape = tuple(d["_shape"])
    out = np.zeros(shape)
    data, idx, ptr = (np.asarray(d[k]) for k in ("data", "indices", "indptr"))
    for col in range(len(ptr) - 1):
        out[idx[ptr[col]:ptr[col + 1]], col] = data[ptr[col]:ptr[col + 1]]
    return out


def load_smal(root: Path | None = None) -> dict:
    """The SMAL model and its family shape means, as plain arrays."""
    root = Path(root or cfg.SMAL)
    model, data = root / "smal_CVPR2017.pkl", root / "smal_CVPR2017_data.pkl"
    if not model.is_file():
        raise RigError(
            f"no SMAL model at {model}. It is a gated download from "
            f"https://smal.is.tue.mpg.de -- unpack smalV1.0.tgz and point "
            f"NAVSAFE_SMAL at the folder (currently {cfg.SMAL}).")
    _install_stubs()
    raw = pickle.load(open(model, "rb"), encoding="latin1")
    extra = pickle.load(open(data, "rb"), encoding="latin1")
    out = {k: _unwrap(v) for k, v in raw.items()}
    out["J_regressor"] = _dense(raw["J_regressor"])
    out["cluster_means"] = extra["cluster_means"]
    return out


def shaped(smal: dict, betas) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and joint locations for one shape vector. (V,3), (J,3)"""
    v = smal["v_template"] + smal["shapedirs"] @ np.asarray(betas, np.float64)
    return v, smal["J_regressor"] @ v


def smal_rig(smal: dict, betas, *, influences: int = 8) -> Rig:
    """SMAL at one shape, as the ``Rig`` the shared bake/pose code speaks.

    SMAL's rest pose carries no joint rotation, so a joint's bind transform is
    just a translation to its rest location. The joints have no names in the
    model, so they are numbered -- nothing downstream maps by name once
    ``Motion.local_rot`` is used.
    """
    v, J = shaped(smal, betas)
    parent = np.asarray(smal["kintree_table"][0], np.int64).copy()
    parent[parent > 10 ** 6] = -1
    parent[0] = -1
    bind = np.tile(np.eye(4), (len(J), 1, 1))
    bind[:, :3, 3] = J

    w = np.asarray(smal["weights"], np.float64)
    order = np.argsort(-w, axis=1)[:, :influences]
    wt = np.take_along_axis(w, order, axis=1)
    wt /= np.maximum(wt.sum(1, keepdims=True), 1e-12)
    return Rig(vertices=v, bind=bind, names=[f"J{i:02d}" for i in range(len(J))],
               parent=parent, lbs_idx=order.astype(np.int64), lbs_w=wt)


# ---------------------------------------------------------------------- gait
def _foot_pos(rig: Rig, limb, hip: float, knee: float, ankle: float,
              binv: np.ndarray | None = None) -> np.ndarray:
    """This limb's foot joint at these three angles. SMAL is z-up.

    ``binv`` is the rig's inverted bind transforms. It does not change during a
    solve, so the caller hoists it out; inverting 33 matrices per evaluation
    made sampling the knee too slow to be worth doing.
    """
    binv = np.linalg.inv(rig.bind) if binv is None else binv
    angles = {limb[0]: hip, limb[1]: knee, limb[2]: ankle}
    p0 = int(rig.parent[limb[0]])
    G = rig.bind[p0] if p0 >= 0 else np.eye(4)
    for j in limb:
        p = int(rig.parent[j])
        rest = binv[p] @ rig.bind[j] if p >= 0 else rig.bind[j]
        d = np.eye(4)
        d[:3, :3] = _rot_lateral(angles.get(j, 0.0))
        G = G @ rest @ d
    return G[:3, 3].copy()


def _foot_height(rig: Rig, limb, hip: float, knee: float, ankle: float,
                 binv: np.ndarray | None = None) -> float:
    """Height of this limb's foot joint at these three angles."""
    return float(_foot_pos(rig, limb, hip, knee, ankle, binv)[2])


def _limb_plane(rig: Rig, limb) -> tuple[float, float, float, float, float]:
    """The limb as a planar two-link chain: lengths, rest directions, foot drop.

    A limb swings about the body's lateral axis, and a rotation about that axis
    leaves the lateral coordinate alone. So each segment's sideways offset is a
    constant the swing never touches, and the part that moves is its projection
    into the fore-aft/vertical plane. Those projections are what the IK must
    solve with; the full 3-D lengths would ask the limb for a reach it does not
    have in the plane it moves in.

    The rest directions matter for the same reason the fold direction does: the
    knee juts off the hip-to-foot line, so a segment does not start pointing
    straight down, and a joint angle is a rotation FROM where it already is.

    Returns ``(l1, l2, phi1, phi2, drop)`` -- hip->knee and knee->ankle in the
    plane, their rest angles measured from straight down with backward
    positive, and how far the foot hangs below the ankle.
    """
    J = rig.bind[:, :3, 3]
    out = []
    for a, b in ((limb[0], limb[1]), (limb[1], limb[2])):
        d = J[b] - J[a]
        out.append((float(np.hypot(d[0], d[2])),
                    float(np.arctan2(-d[0], -d[2]))))
    d3 = J[limb[3]] - J[limb[2]]
    return out[0][0], out[1][0], out[0][1], out[1][1], float(np.hypot(d3[0], d3[2]))


def _leg_ik(l1: float, l2: float, phi1: float, phi2: float,
            x: float, z: float, sign: float) -> tuple[float, float]:
    """Hip and knee angles that put the ankle at ``(x, z)`` relative to the hip.

    ``x`` is forward and ``z`` is up in the rig's frame, so a standing leg has
    ``z`` negative. One triangle, solved by the law of cosines: closed form, so
    consecutive phases cannot land on different branches, and ``sign`` fixes
    which way the knee folds once for the whole limb.

    That is the property the height solve could not offer. It searched a single
    angle against a single constraint, so its answers came in symmetric pairs
    that traded places as the hip crossed vertical, and it clamped at the ends
    of the joint's range -- a knee that flicked, once a cycle, in every limb.

    Out of reach the target is pulled onto the nearest reachable circle rather
    than abandoned, so the angles stay continuous there too.
    """
    d = float(np.hypot(x, z))
    d = min(max(d, abs(l1 - l2) + 1e-9), l1 + l2 - 1e-9)
    gamma = float(np.arccos(np.clip(
        (l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2), -1.0, 1.0)))
    alpha = float(np.arccos(np.clip(
        (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d), -1.0, 1.0)))
    aim = float(np.arctan2(-x, -z))          # 0 is straight down, + is backward
    psi1 = aim - sign * alpha                # where each segment has to point
    psi2 = psi1 + sign * (np.pi - gamma)
    hip = psi1 - phi1                        # a joint angle turns FROM the rest
    return hip, psi2 - phi2 - hip


def _bend_sign(rig: Rig, limb, binv: np.ndarray | None = None) -> float:
    """Which way this limb's knee is built to fold, as the sign of its angle.

    SMAL's rest pose already says it: the knee sits off the line joining the
    two joints it lies between -- forward on a hind stifle, back on a front
    elbow, by 1 to 10 cm depending on the family. Folding it the other way is a
    thing a leg cannot do, and it also inverts the joint under gaussians that
    were bound around the rest shape, so the leg folds against itself.

    The reference line is hip to ANKLE, the two ends of the chain the IK
    actually solves. Measuring to the FOOT instead brings a third segment into
    a two-link question and can answer it backwards: on the horse's front limb,
    nearly straight either way, the two lines disagree in sign (-0.73 cm to the
    foot, +0.95 cm to the ankle), and the leg came out folding forwards.
    """
    J = rig.bind[:, :3, 3]
    hip, knee, ankle = J[limb[0]], J[limb[1]], J[limb[2]]
    d = ankle - hip
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return 1.0
    d = d / n
    away = float((knee - hip)[0] - np.dot(knee - hip, d) * d[0])
    if abs(away) < 1e-6:                      # a perfectly straight rig: pick one
        return 1.0
    eps = 0.1
    moved = (_foot_pos(rig, limb, 0.0, eps, -eps, binv)[0]
             - _foot_pos(rig, limb, 0.0, -eps, eps, binv)[0])
    # Folding swings the foot AWAY from the side the knee juts out to.
    return 1.0 if np.sign(moved) == -np.sign(away) else -1.0


def _rot_lateral(a: float) -> np.ndarray:
    """A limb swings about the body's lateral axis, which is y in SMAL."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def gait_rotations(rig: Rig, frames: int = 16, gait: str = "trot", *,
                   step: float = 0.60, swing: float = 0.5,
                   lift: float = 0.16, stance: float = 0.92,
                   bob: float = 0.010, stiff: str = "",
                   sched: Dict[tuple, float] | None = None):
    """One cycle of local joint rotations, plus per-foot contacts.

    The foot is driven along a path and the joints follow from it. Through
    stance the foot travels straight back by ``step`` of the animal's leg
    length at a constant height; through swing it lifts by ``lift`` and returns.
    Hip and knee come from :func:`_leg_ik`, and the ankle undoes their sum so
    the sole stays flat.

    The parameters are therefore lengths, in units of the animal's own leg, and
    they mean what they say: ``step`` IS the foot's travel, so the stride is
    ``step / (1 - swing)`` leg lengths and nothing has to be discovered by
    measuring afterwards. Real quadrupeds walk at roughly 1.2 to 1.6 of that,
    which is where the default comes from.

    This replaces sweeping the hip through a chosen ANGLE and solving the knee
    to keep the foot down. That solve had one free angle against one constraint,
    so its answers came in symmetric pairs which traded places as the hip
    crossed vertical, and it clamped at the ends of the joint's range: a knee
    that flicked once a cycle, in every limb, which is what read as an animal
    moving its feet at random. Its amplitude was also unusable as a control --
    on the zebra, hip_amp 0.25 to 0.35 moved the stride by 3.2x -- so a gait
    could not be asked for, only discovered.

    ``stance`` is how far the hip-to-ankle distance is held below straight.
    Standing flexed is what real quadrupeds do and what leaves the knee room to
    extend as the hip passes over the planted foot.

    ``stiff`` -- ``"front"``, ``"hind"`` or ``"both"`` -- carries those limbs
    along the arc their own leg traces instead of along a level line, which
    holds their knee at one angle through the whole of stance. A horse's
    foreleg works close to a strut at a walk, and bending it either way looked
    wrong in the render.

    ``sched`` replaces the named gait's footfall table with one measured off a
    captured clip, and ``swing`` with that clip's duty factor. Timing is the
    half of a capture that IS recoverable: a foot declares itself by sitting
    low and swinging far, and the phase of its height says when it lands, all
    without a template. Recovering the POSE from the same clip does not work --
    the surface is nearly all torso, so the fit is free to leave the legs
    anywhere -- which is why the angles are still generated here.
    """
    if sched is None and gait not in GAITS:
        raise RigError(f"unknown gait {gait!r}; expected one of {sorted(GAITS)}")
    if not 0.0 < swing < 1.0:
        raise RigError(f"swing must be a fraction of the cycle, got {swing}")
    sched = GAITS[gait] if sched is None else dict(sched)
    if set(sched) != set(LIMBS):
        raise RigError("a schedule needs a phase for each of the four limbs")
    n = len(rig.names)
    rot = np.tile(np.eye(3), (frames, n, 1, 1))
    contact = np.zeros((frames, len(LIMBS)), bool)
    binv = np.linalg.inv(rig.bind)

    for i, limb in enumerate(LIMBS):
        l1, l2, phi1, phi2, drop = _limb_plane(rig, limb)
        sign = _bend_sign(rig, limb, binv)
        leg = l1 + l2 + drop
        travel, clear = step * leg, lift * leg
        xs, rise, down = [], [], []
        for t in range(frames):
            p = (t / frames + sched[limb]) % 1.0
            planted = p >= swing
            if planted:                                   # foot pinned, body over it
                u = (p - swing) / (1.0 - swing)
                xs.append(0.5 * travel - travel * u)
                rise.append(0.0)
            else:
                # Swing, carried back to the front. Its ends have to match the
                # stance's VELOCITY as well as its position, or the foot
                # reverses in one frame and the knee takes the corner as a
                # flick. A cubic through both endpoints with the stance's own
                # slope at each carries the foot on a little, turns it round,
                # and brings it back down already moving at contact speed --
                # which is what a leg does. The lift uses sin squared for the
                # same reason: sine leaves the foot rising as it touches down.
                u = p / swing
                m = -travel * swing / (1.0 - swing)
                u2, u3 = u * u, u * u * u
                xs.append((2 * u3 - 3 * u2 + 1) * -0.5 * travel
                          + (u3 - 2 * u2 + u) * m
                          + (-2 * u3 + 3 * u2) * 0.5 * travel
                          + (u3 - u2) * m)
                rise.append(clear * float(np.sin(np.pi * u)) ** 2)
            down.append(planted)

        # A leg cannot both stand tall and reach far: the two-link chain has to
        # span hypot(x, reach) at every phase, and the swing's turn carries the
        # foot a little PAST the step it is reversing out of. Sizing the stance
        # off the step alone left those phases just outside the reach, where
        # the solve clamps -- the knee stalling and then catching up, which is
        # the flick this rewrite exists to remove. Real quadrupeds settle lower
        # for a longer stride; measuring the path and settling to fit it means
        # the step asked for is the step taken.
        far = max(abs(v) for v in xs)
        span = float(np.sqrt(max((l1 + l2) ** 2 - far * far, 1e-12)))
        reach = min(stance * (l1 + l2), 0.999 * span)

        # A stiff limb carries the foot along the ARC its own leg traces rather
        # than along a level line. Every stance phase then sits the same
        # distance from the hip, so the law of cosines returns the same knee
        # for all of them and the joint simply does not move -- no special case
        # in the solve, and the knee still flexes through swing, which is when
        # a horse's foreleg does flex and how the foot clears the ground.
        #
        # The cost is that the foot rises off the road at the ends of stance
        # instead of staying level. That is what a strut does; the animal
        # vaults over it. It rises rather than sinks, so nothing cuts through
        # the road surface.
        locked = (stiff in ("front", "both") and limb in (FRONT_L, FRONT_R)) or \
                 (stiff in ("hind", "both") and limb in (HIND_L, HIND_R))

        for t, (x, up, planted) in enumerate(zip(xs, rise, down)):
            base = (-float(np.sqrt(max(reach * reach - x * x, 1e-12))) if locked
                    else -reach)
            hip, knee = _leg_ik(l1, l2, phi1, phi2, x, base + up, sign)
            for j, ang in zip(limb[:3], (hip, knee, -(hip + knee))):
                rot[t, j] = _rot_lateral(ang)
            contact[t, i] = planted

    for t in range(frames):
        for j in SPINE[:2]:
            rot[t, j] = _rot_lateral(bob * np.sin(4 * np.pi * t / frames))
    return rot, contact


def measured_stride(rig: Rig, rot: np.ndarray, contact: np.ndarray,
                    *, limb_index: int = 2) -> float:
    """How far one cycle carries the body, read off the motion generated.

    The planted foot does not move; the body moves over it. So the foot's
    fore-aft excursion measures the body's travel -- but only over the part of
    the cycle the foot is down. The stride is a whole cycle, so that excursion
    has to be divided back out. Measured, never a formula, because the leg's
    path decides the excursion.

    Which frames are stance comes from the ``contact`` the generator returned,
    not from re-deriving it. Re-deriving it meant two copies of the schedule,
    and when one of them started coming from a measured clip the other went on
    using the table: the stance frames sampled here were then the wrong ones,
    and every stride came out short.

    Getting this wrong is not a small error: the renderer picks the gait phase
    as ``travel / stride``, so a stride half its true value cycles the legs at
    twice the ground speed, which reads as an animal skating along the road.
    """
    binv = np.linalg.inv(rig.bind)
    limb, frames, xs = LIMBS[limb_index], len(rot), []
    down = np.asarray(contact)[:, limb_index]
    for t in range(frames):
        if not down[t]:
            continue
        p0 = int(rig.parent[limb[0]])
        G = rig.bind[p0] if p0 >= 0 else np.eye(4)
        for j in limb:
            p = int(rig.parent[j])
            rest = binv[p] @ rig.bind[j] if p >= 0 else rig.bind[j]
            d = np.eye(4)
            d[:3, :3] = rot[t, j]
            G = G @ rest @ d
        xs.append(G[0, 3])
    duty = max(float(down.mean()), 1e-6)
    return float(max(xs) - min(xs)) / duty if xs else 0.0


def max_speed(stride_m: float, *, dt: float = 0.1, per_frame: float = 0.5) -> float:
    """Fastest an actor may move before a bank of this stride aliases.

    Past this much of a cycle per rendered frame the legs stop reading as a
    gait and start reading as flicker. This is a property of the render rate,
    not of the bank: no number of phases fixes it, because the phase count
    cancels out of ``travel / stride`` -- slicing the same distance more finely
    still skips the same cycles.

    ``per_frame`` was a guessed quarter-cycle until a reviewer watched the ten
    R-4 renders at 3.0 m/s and named which ones looked wrong. Horse (0.41
    cycles/frame), cow (0.33) and hippo (0.46) passed; both dogs (1.26) were
    reported as having no gait at all. So the boundary sits above 0.46 and well
    below 1.26, and half a cycle is where it is set -- measured against what a
    person actually sees rather than against an assumption.
    """
    return stride_m * per_frame / dt


# ---------------------------------------------------------------------- fit
def fit_animal(gaussians: Dict[str, np.ndarray], smal: dict, family: str = "dog",
               *, iters: int = 800, yaw_steps: int = 48, n_cloud: int = 2500,
               n_mesh: int = 2500, seed: int = 0, device: str | None = None):
    """Recover the pose, shape and placement of a harvested animal.

    Unlike the pedestrians, harvested animals stand with their legs under them,
    so the search needs no motion to initialise from: SMAL's own rest pose at
    the family shape is already close and only yaw is unknown.

    Shape is solved WITH pose. A dog and a horse differ by beta, not by scale,
    so freezing beta at the family mean and letting a similarity transform
    absorb the difference would stretch the wrong things.

    Returns ``(fit, betas)``. The placement rides on the fit as ``fit.linear``,
    a single affine in the ASSET's own frame, and there is deliberately no
    second copy of it: SMAL is z-up where the asset frame is y-up, so the
    placement is not a yaw and a scale, and anything that rebuilds it from
    those two drops the upright rotation. That is what laid the template on its
    side, put its feet inside the torso, and left the leg joints owning torso
    gaussians -- one leg that never moved, and a body that tore when the others
    did.
    """
    import torch

    from navsafe.benchmark.editing.assets.animate import surface_points

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if family not in FAMILY:
        raise RigError(f"unknown family {family!r}; SMAL covers {sorted(FAMILY)}")

    cloud = surface_points(gaussians, max_points=n_cloud, seed=seed)
    C = torch.tensor(cloud, dtype=torch.float32, device=dev)
    ground = lambda p: torch.stack([p[:, 0].mean(), p[:, 1].min(), p[:, 2].mean()])
    cloud_ground = ground(C).clone()
    C = C - cloud_ground
    height = float(C[:, 1].max() - C[:, 1].min())

    t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32, device=dev)
    v_template, shapedirs = t(smal["v_template"]), t(smal["shapedirs"])
    J_reg, weights = t(smal["J_regressor"]), t(smal["weights"])
    parent = [int(p) if p < 10 ** 6 else -1 for p in smal["kintree_table"][0]]
    parent[0] = -1
    n_j = weights.shape[1]
    R0 = t(Z_UP_TO_Y_UP)

    def posed(betas, theta):
        v = v_template + torch.einsum("vsb,b->vs", shapedirs, betas)
        J = J_reg @ v
        R = _rodrigues_t(theta)
        G: list = [None] * n_j
        for j in range(n_j):
            p = parent[j]
            off = J[j] if p < 0 else J[j] - J[p]
            local = torch.cat([torch.cat([R[j], off[:, None]], 1),
                               torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)], 0)
            G[j] = local if p < 0 else G[p] @ local
        G = torch.stack(G)
        eye = torch.eye(4, device=dev).expand(n_j, 4, 4).clone()
        eye[:, :3, 3] = -J
        A = torch.einsum("vj,jik->vik", weights, G @ eye)
        return torch.einsum("vij,vj->vi", A[:, :3, :3], v) + A[:, :3, 3]

    def chamfer(a, b):
        d = torch.cdist(a, b)
        return d.min(1).values.mean() + d.min(0).values.mean()

    def yaw_mat(y):
        c, s = torch.cos(y), torch.sin(y)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack([torch.stack([c, z, s], -1),
                            torch.stack([z, o, z], -1),
                            torch.stack([-s, z, c], -1)], -2)

    def place(v, y, s, sh):
        p = torch.einsum("ij,vj->vi", yaw_mat(y), torch.einsum("ij,vj->vi", R0, v) * s)
        return p - ground(p) + sh

    beta0 = t(smal["cluster_means"][FAMILY[family]])
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        rest = posed(beta0, torch.zeros(n_j, 3, device=dev))
        r = torch.einsum("ij,vj->vi", R0, rest)
        s0 = height / float(r[:, 1].max() - r[:, 1].min())
        mi = rng.choice(len(rest), min(n_mesh, len(rest)), replace=False)
        best = (1e9, 0.0)
        zero3 = torch.zeros(3, device=dev)
        for y in np.linspace(0, 2 * np.pi, yaw_steps, endpoint=False):
            d = chamfer(place(rest[mi], torch.tensor(float(y), device=dev), s0, zero3), C)
            if d.item() < best[0]:
                best = (d.item(), float(y))

    betas = beta0.clone().requires_grad_(True)
    theta = torch.zeros(n_j, 3, device=dev, requires_grad=True)
    yaw = torch.tensor(best[1], device=dev, requires_grad=True)
    logs = torch.tensor(float(np.log(s0)), device=dev, requires_grad=True)
    shift = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [theta], "lr": 0.02},
                            {"params": [betas], "lr": 0.05},
                            {"params": [yaw, logs, shift], "lr": 0.02}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    loss = torch.tensor(float("nan"))
    for _ in range(iters):
        opt.zero_grad()
        p = place(posed(betas, theta)[mi], yaw, torch.exp(logs), shift)
        loss = (chamfer(p, C) + 0.05 * (theta ** 2).sum()
                + 0.002 * ((betas - beta0) ** 2).sum())
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        v = posed(betas, theta)
        mesh = place(v, yaw, torch.exp(logs), shift)
        lin = (yaw_mat(yaw) @ R0) * torch.exp(logs)
        offset = (mesh - torch.einsum("ij,vj->vi", lin, v)).mean(0) + cloud_ground
        fit = Fit(frame=0, delta=theta.detach().cpu().numpy().astype(np.float64),
                  yaw=float(yaw), scale=float(torch.exp(logs)),
                  offset=offset.cpu().numpy().astype(np.float64),
                  chamfer=float(loss.item()),
                  linear=lin.cpu().numpy().astype(np.float64))
    logger.info("fit_animal: %s chamfer %.4f m, scale %.3f, yaw %.1f deg",
                family, fit.chamfer, fit.scale, np.degrees(fit.yaw))
    return fit, betas.detach().cpu().numpy()


def _rodrigues_t(v):
    import torch
    a = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-12)
    ax = v / a
    K = torch.zeros(len(v), 3, 3, device=v.device, dtype=v.dtype)
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
    aa = a[..., None]
    return I + torch.sin(aa) * K + (1 - torch.cos(aa)) * (K @ K)


# --------------------------------------------------------------------- bake
def _rodrigues(theta: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    out = np.tile(np.eye(3), (len(theta), 1, 1))
    n = np.linalg.norm(theta, axis=1)
    out[n > 1e-9] = Rotation.from_rotvec(theta[n > 1e-9]).as_matrix()
    return out


def gait_motion(rig: Rig, fit: Fit, *, frames: int = 16, gait: str = "trot",
                **gait_kw) -> Motion:
    """The cycle this animal will be baked from, as a :class:`Motion`.

    Frame 0 is the FITTED pose, so binding there reproduces the source asset
    exactly; frames 1..N are the cycle. The legs come from the gait and
    everything else -- spine curve, head, tail -- from the fit, so the animal
    keeps the posture it was harvested in and only walks with it.
    """
    fitted = _rodrigues(np.asarray(fit.delta, np.float64))
    rot, contact = gait_rotations(rig, frames, gait, **gait_kw)
    stride = measured_stride(rig, rot, contact) * fit.scale

    local = np.empty((frames + 1, len(rig.names), 3, 3))
    local[0] = fitted
    for t in range(frames):
        local[t + 1] = fitted
        local[t + 1, list(LEG_JOINTS)] = rot[t, list(LEG_JOINTS)]
    pad = np.concatenate([contact[:1], contact], 0)
    return Motion(pos=np.zeros((frames + 1, 30, 3)),
                  rot=np.zeros((frames + 1, 30, 3, 3)),
                  contact=pad, local_rot=local, stride_m=stride,
                  source=f"procedural:{gait}", fps=1.0 / max(frames, 1))


def bake_animal(ply: Path, out_dir: Path, *, family: str = "dog",
                phases: int = 16, gait: str = "trot",
                dims: Sequence[float] | None = None, prune_opacity: float = 0.0,
                smal: dict | None = None, **gait_kw) -> dict:
    """Fit, bind and bake one harvested animal into a gait pose bank."""
    smal = smal or load_smal()
    gaussians = prune_gaussians(read_3dgs_ply(ply), prune_opacity)
    fit, betas = fit_animal(gaussians, smal, family)
    fit.asset = str(ply)
    rig = smal_rig(smal, betas)
    motion = gait_motion(rig, fit, frames=phases, gait=gait, **gait_kw)

    # The solved pose already rode into the motion's frame 0, so the Fit that
    # goes on from here carries only the placement -- applying delta again
    # would pose the animal twice.
    placed = Fit(frame=0, delta=np.zeros_like(fit.delta), yaw=fit.yaw,
                 scale=fit.scale, offset=fit.offset, chamfer=fit.chamfer,
                 asset=fit.asset, linear=fit.linear)
    rigged = bind_gaussians(gaussians, rig, motion, placed)
    manifest = bake_bank(rigged, rig, motion, out_dir, dims=dims,
                         frames=list(range(1, phases + 1)))
    manifest["fit"] = fit.to_dict()
    from navsafe.benchmark.editing.assets.animate import round_trip_error
    manifest["round_trip_m"] = round_trip_error(rigged, rig, motion)
    manifest["prune_opacity"] = float(prune_opacity)
    manifest["family"] = family
    manifest["max_speed_ms"] = round(max_speed(manifest["stride_m"]), 3)
    (Path(out_dir) / "bank.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("bake_animal: %s stride %.3f m -> usable up to %.2f m/s",
                Path(ply).name, manifest["stride_m"], manifest["max_speed_ms"])
    return manifest
