# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The quadruped gait: the parts that fail silently.

SMAL is a gated download, so nothing here loads it -- the tests build a rig
with SMAL's joint layout and check the generator against it. What that covers
is the two properties the render actually depends on: the planted foot holds
still, and the stride reported is the stride generated. Both fail quietly. A
foot that arcs through the road looks like bad reconstruction, and a stride
that disagrees with the pose makes the renderer clock the phase wrong, which
looks like a skating animal rather than like a number being off.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.assets.animate import (
    Fit, Motion, Rig, RigError, fitted_transform, rest_locals)
from navsafe.benchmark.editing.assets.animate import _asset_affine
from navsafe.benchmark.editing.assets.quadruped import (
    FAMILY,
    GAITS,
    HIND_L,
    LEG_JOINTS,
    LIMBS,
    SPINE,
    Z_UP_TO_Y_UP,
    _bend_sign,
    _leg_ik,
    _limb_plane,
    _rot_lateral,
    gait_rotations,
    max_speed,
    measured_stride,
    smal_rig,
)

N_JOINTS = 33


def _smal_like_rig(leg: float = 0.52, seed: int = 0) -> Rig:
    """A rig with SMAL's joint layout: spine, four 4-joint limbs, neck, tail.

    Built rather than loaded, so the gait generator is testable without the
    gated model. The geometry only has to be quadruped-shaped: a body above the
    ground with four limbs hanging off it, z-up as SMAL is.
    """
    parent = np.full(N_JOINTS, -1, np.int64)
    for j in range(1, 7):                       # spine forward off the pelvis
        parent[j] = j - 1
    for limb, root in ((HIND_L, 0), ((21, 22, 23, 24), 0), (LIMBS[0], 6), (LIMBS[1], 6)):
        parent[limb[0]] = root
        for a, b in zip(limb, limb[1:]):
            parent[b] = a
    parent[15], parent[16], parent[32] = 6, 15, 16       # neck, head, nose
    parent[25] = 0
    for j in range(26, 32):                     # tail
        parent[j] = j - 1
    parent[0] = -1

    J = np.zeros((N_JOINTS, 3))
    J[:7, 0] = np.linspace(-0.30, 0.13, 7)      # spine runs forward (+x)
    J[:7, 2] = 0.16
    # The knee sits OFF the hip-to-foot line, and which side decides which way
    # the joint folds: forward on a hind stifle, back on a front elbow. SMAL
    # carries 2-8 cm of it. A limb built perfectly straight, as this was, has no
    # fold direction at all, so it cannot exercise the solve that reads one.
    for limb, x, side, stifle in ((LIMBS[0], 0.24, +1, -0.03),
                                  (LIMBS[1], 0.24, -1, -0.03),
                                  (HIND_L, -0.34, +1, +0.06),
                                  ((21, 22, 23, 24), -0.34, -1, +0.06)):
        for i, j in enumerate(limb):            # descend to the foot
            J[j] = [x + (stifle if i == 1 else 0.0), side * 0.09,
                    0.16 - leg * i / 3.0]
    for i, j in enumerate((15, 16, 32)):
        J[j] = [0.33 + 0.12 * i, 0.0, 0.20]
    for i, j in enumerate(range(25, 32)):
        J[j] = [-0.38 - 0.08 * i, 0.0, 0.11]

    bind = np.tile(np.eye(4), (N_JOINTS, 1, 1))
    bind[:, :3, 3] = J
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N_JOINTS, (60, 8))
    w = rng.random((60, 8))
    return Rig(vertices=rng.normal(0.0, 0.2, (60, 3)), bind=bind,
               names=[f"J{i:02d}" for i in range(N_JOINTS)], parent=parent,
               lbs_idx=idx, lbs_w=w / w.sum(1, keepdims=True))


def _planted_heights(rig: Rig, rot: np.ndarray, gait: str = "trot",
                     limb=HIND_L, swing: float = 0.5):
    binv = np.linalg.inv(rig.bind)
    out = []
    for t in range(len(rot)):
        if (t / len(rot) + GAITS[gait][limb]) % 1.0 < swing:
            continue
        p0 = int(rig.parent[limb[0]])
        G = rig.bind[p0] if p0 >= 0 else np.eye(4)
        for j in limb:
            p = int(rig.parent[j])
            rest = binv[p] @ rig.bind[j] if p >= 0 else rig.bind[j]
            d = np.eye(4)
            d[:3, :3] = rot[t, j]
            G = G @ rest @ d
        out.append(G[2, 3])
    return np.array(out)


# ---------------------------------------------------------------- footfall
def test_a_trot_strikes_its_diagonals_together():
    """The gait's whole identity is which feet are down when."""
    rig = _smal_like_rig()
    _, contact = gait_rotations(rig, 16, "trot")
    fl, fr, hl, hr = (contact[:, i] for i in range(4))
    assert np.array_equal(fl, hr)            # one diagonal
    assert np.array_equal(fr, hl)            # the other
    assert not np.array_equal(fl, fr)        # and they are in antiphase


def test_a_walk_keeps_at_least_two_feet_down():
    """The lateral sequence is what a heavy animal uses; it never has fewer
    than two feet on the ground, which is the point of choosing it."""
    rig = _smal_like_rig()
    _, contact = gait_rotations(rig, 16, "walk")
    assert contact.sum(1).min() >= 2


def test_every_foot_takes_a_turn():
    rig = _smal_like_rig()
    _, contact = gait_rotations(rig, 16, "trot")
    assert contact.any(0).all()              # none of the four is dead
    assert (~contact).any(0).all()           # and none is permanently planted


def test_an_unknown_gait_is_refused():
    with pytest.raises(RigError, match="unknown gait"):
        gait_rotations(_smal_like_rig(), 8, "gallop")


def test_a_measured_schedule_replaces_the_named_one():
    """Timing is the half of a captured clip that IS recoverable, so it can be
    supplied instead of chosen from the table."""
    rig = _smal_like_rig()
    measured = dict(zip(LIMBS, (0.00, 0.51, 0.30, 0.77)))
    _, contact = gait_rotations(rig, 100, "walk", sched=measured, swing=0.575)
    for i, limb in enumerate(LIMBS):
        down = np.flatnonzero(contact[:, i])
        # stance runs from the limb's phase, wrapped
        assert contact[:, i].mean() == pytest.approx(1.0 - 0.575, abs=0.02)
        assert len(down) > 0


def test_a_schedule_missing_a_limb_is_refused():
    rig = _smal_like_rig()
    with pytest.raises(RigError, match="four limbs"):
        gait_rotations(rig, 8, sched={LIMBS[0]: 0.0, LIMBS[1]: 0.5})


# ------------------------------------------------------------------ planting
def test_the_stance_foot_holds_its_height():
    """The defect this exists to prevent: a hip swung without solving the knee
    arcs the foot up out of the road and back in."""
    rig = _smal_like_rig()
    rot, _ = gait_rotations(rig, 32, "trot")
    zs = _planted_heights(rig, rot)
    leg = float(np.linalg.norm(rig.bind[HIND_L[3]][:3, 3] - rig.bind[HIND_L[0]][:3, 3]))
    assert np.ptp(zs) < 0.25 * leg


def _ankle_pos(rig: Rig, limb, hip: float, knee: float) -> np.ndarray:
    """Where the limb's ankle joint lands at these hip and knee angles."""
    binv = np.linalg.inv(rig.bind)
    G = rig.bind[int(rig.parent[limb[0]])]
    for j, ang in zip(limb[:3], (hip, knee, 0.0)):
        d = np.eye(4)
        d[:3, :3] = _rot_lateral(ang)
        G = G @ (binv[int(rig.parent[j])] @ rig.bind[j]) @ d
    return G[:3, 3]


def _knee_series(rig: Rig, rot: np.ndarray, limb=HIND_L) -> np.ndarray:
    """The knee's rotation angle at each phase, as a signed scalar."""
    return np.array([np.arctan2(rot[t, limb[1]][0, 2], rot[t, limb[1]][0, 0])
                     for t in range(len(rot))])


@pytest.mark.parametrize("limb", LIMBS)
def test_the_knee_moves_smoothly_round_the_cycle(limb):
    """A leg that flicks is a leg that reads as random rather than walking.

    The old height solve flicked three separate ways -- snapping straight when
    the plant was out of reach, trading between its two symmetric answers as
    the hip crossed vertical, and folding the knee the way a joint cannot. Two
    were patched and the third was not reachable that way, because one free
    angle against one constraint cannot be made single-valued. Driving the foot
    and taking the angles from closed-form IK removes the question.

    A cycle is closed, so the step from the last phase back to the first counts.
    """
    rig = _smal_like_rig()
    k = _knee_series(rig, gait_rotations(rig, 48, "walk")[0], limb)
    step = np.abs(np.diff(np.concatenate([k, k[:1]])))
    assert step.max() < 0.25, f"{limb}: knee jumps {step.max():.2f} rad"


def test_a_stiff_limb_holds_its_knee_through_stance():
    """A horse's foreleg is close to a strut at a walk. Carrying the foot along
    the arc the leg traces, rather than along a level line, keeps every stance
    phase the same distance from the hip -- so the knee angle that reaches them
    is the same angle, and it stops moving without the solve being special-
    cased. It still flexes through swing, which is when the foot has to clear
    the ground."""
    rig = _smal_like_rig()
    rot, contact = gait_rotations(rig, 48, "walk", stiff="front")
    for i, limb in enumerate(LIMBS):
        k = _knee_series(rig, rot, limb)
        planted = contact[:, i]
        span = float(np.ptp(k[planted]))
        if limb in (LIMBS[0], LIMBS[1]):
            assert span < 1e-6, f"{limb}: stiff knee still moves {span:.3f} rad"
            assert float(np.ptp(k)) > 0.1      # but it does flex in swing
        else:
            assert span > 0.1, f"{limb}: hind knee should still work"


def test_the_ik_puts_the_ankle_where_it_was_asked_to():
    """The whole rewrite rests on this: ask for a foot position, get joint
    angles that produce it. A sign slip anywhere in the triangle shows up here
    rather than as an animal walking oddly in a render."""
    rig = _smal_like_rig()
    l1, l2, phi1, phi2, _drop = _limb_plane(rig, HIND_L)
    sign = _bend_sign(rig, HIND_L)
    hip_pos = rig.bind[HIND_L[0]][:3, 3]
    # `_foot_pos` walks the whole limb; the ankle is that chain with the last
    # joint left at rest, so the same call reports it as its third joint.
    # Targets a two-link chain of this size can actually reach; asking for
    # more is answered by the nearest reachable point, which is a different
    # contract and has its own test.
    for x in (-0.14, -0.05, 0.0, 0.05, 0.14):
        for z in (-0.28, -0.32):
            hip, knee = _leg_ik(l1, l2, phi1, phi2, x, z, sign)
            ankle = _ankle_pos(rig, HIND_L, hip, knee) - hip_pos
            assert ankle[0] == pytest.approx(x, abs=1e-6)
            assert ankle[2] == pytest.approx(z, abs=1e-6)


def test_the_stride_is_the_step_that_was_asked_for():
    """``step`` is the foot's travel through stance, so the stride is that
    divided by the stance fraction. It is an input now, not a discovery."""
    rig = _smal_like_rig()
    leg = float(np.linalg.norm(rig.bind[HIND_L[3]][:3, 3] - rig.bind[HIND_L[0]][:3, 3]))
    for step in (0.4, 0.6, 0.8):
        rot, contact = gait_rotations(rig, 64, "walk", step=step, swing=0.5)
        got = measured_stride(rig, rot, contact)
        assert got == pytest.approx(step * leg / 0.5, rel=0.02)


# --------------------------------------------------------------- integration
def test_the_gait_drives_the_shared_posing_path():
    """A Motion carrying rig-order rotations has to work in the same
    rest_locals the human path uses -- that shared path is the reason this is
    a template plus a gait rather than a second pipeline."""
    rig = _smal_like_rig()
    rot, contact = gait_rotations(rig, 8, "trot")
    motion = Motion(pos=np.zeros((8, 30, 3)), rot=np.zeros((8, 30, 3, 3)),
                    contact=contact, local_rot=rot, stride_m=0.4, source="test")
    a = rest_locals(rig, motion, 0)
    b = rest_locals(rig, motion, 4)
    assert a.shape == (N_JOINTS, 4, 4)
    assert not np.allclose(a, b)             # the pose actually changed
    # joints the gait does not touch keep their rest offset at every frame
    untouched = [j for j in range(N_JOINTS) if j not in LEG_JOINTS and j not in (1, 2)]
    assert np.allclose(a[untouched], b[untouched])


def test_only_the_legs_and_a_little_spine_move():
    rig = _smal_like_rig()
    rot, _ = gait_rotations(rig, 16, "trot")
    moved = {j for j in range(N_JOINTS)
             if not np.allclose(rot[:, j], np.eye(3), atol=1e-12)}
    assert moved <= set(LEG_JOINTS) | {1, 2}


def _placed_joints(rig: Rig, fit: Fit) -> np.ndarray:
    """Where the fit puts each joint, in the asset's own frame."""
    n = len(rig.names)
    motion = Motion(pos=np.zeros((1, 30, 3)), rot=np.zeros((1, 30, 3, 3)),
                    contact=None, local_rot=np.tile(np.eye(3), (1, n, 1, 1)))
    G = fitted_transform(rig, motion, fit)
    lin, tr = _asset_affine(rig, G, np.arange(n)[:, None], np.ones((n, 1)), fit)
    return np.einsum("nij,nj->ni", lin, rig.bind[:, :3, 3]) + tr


def _fit(rig: Rig, linear=None) -> Fit:
    return Fit(frame=0, delta=np.zeros((len(rig.names), 3)), yaw=0.0, scale=1.0,
               offset=np.zeros(3), chamfer=0.0, linear=linear)


def test_the_placement_stands_the_body_up_rather_than_laying_it_down():
    """SMAL is z-up; the asset frame is y-up. That makes the placement a
    rotation a yaw cannot express -- a yaw turns about the UP axis, and here
    the up axis itself is changing.

    Rebuilding the placement from yaw and scale silently drops it: the template
    ends up on its side with its feet somewhere inside the torso, the leg
    joints take ownership of torso gaussians, and the render shows one leg that
    never moves while the body tears when the others swing. None of the
    existing checks see it -- the round trip stays at 1e-16, because binding
    and posing remain consistent with each other however the body is lying.
    """
    rig = _smal_like_rig()
    hip, foot = HIND_L[0], HIND_L[3]

    upright = _placed_joints(rig, _fit(rig, linear=Z_UP_TO_Y_UP.copy()))
    assert upright[foot, 1] < upright[hip, 1]        # foot below hip, in asset y

    rebuilt = _placed_joints(rig, _fit(rig))         # yaw + scale only
    assert rebuilt[foot, 1] == pytest.approx(rebuilt[hip, 1])   # neither above


def test_a_fit_carries_its_placement_through_a_round_trip():
    rig = _smal_like_rig()
    fit = _fit(rig, linear=Z_UP_TO_Y_UP * 1.7)
    back = Fit.from_dict(fit.to_dict())
    assert np.allclose(back.linear, fit.linear)
    assert Fit.from_dict(_fit(rig).to_dict()).linear is None


def test_the_z_up_to_y_up_rotation_is_a_rotation():
    """It maps SMAL's frame onto the asset library's, so it must not mirror --
    a reflection here would swap the animal's left and right legs."""
    assert np.allclose(Z_UP_TO_Y_UP @ Z_UP_TO_Y_UP.T, np.eye(3))
    assert np.linalg.det(Z_UP_TO_Y_UP) == pytest.approx(1.0)
    assert np.allclose(Z_UP_TO_Y_UP @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0])


def test_smal_rig_shapes_up(monkeypatch):
    """smal_rig only reshapes arrays, so it can be checked on a stand-in."""
    rng = np.random.default_rng(0)
    v = rng.normal(0.0, 0.2, (40, 3))
    kin = np.zeros((2, N_JOINTS), np.int64)
    kin[0] = _smal_like_rig().parent
    kin[0][0] = 2 ** 31
    smal = {"v_template": v, "shapedirs": np.zeros((40, 3, 3)),
            "J_regressor": np.eye(N_JOINTS, 40), "weights": rng.random((40, N_JOINTS)),
            "kintree_table": kin}
    rig = smal_rig(smal, np.zeros(3))
    assert len(rig.names) == N_JOINTS
    assert rig.parent[0] == -1
    assert np.allclose(rig.lbs_w.sum(1), 1.0)
    assert rig.lbs_idx.shape == (40, 8)


def test_the_families_are_the_ones_smal_was_built_from():
    """Naming a family SMAL was not fitted from is a request its shape space
    cannot answer -- elephants being the case that matters here."""
    assert set(FAMILY) == {"cat", "dog", "horse", "cow", "hippo"}
    assert "elephant" not in FAMILY
