# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Gait baking: the skinning maths an animated pedestrian rests on.

Everything here runs on a synthetic rig and a synthetic motion, so it says
nothing about how well a real body fits a real harvested cloud -- that is a
picture, not an assertion.  What it does pin down is the part that fails
silently: if binding and re-posing disagree about a transform, every baked
phase is deformed in a way no eyeball would catch, and the round-trip test is
what refuses to let that ship.
"""

from __future__ import annotations

import numpy as np
import pytest

from navsafe.benchmark.editing.assets.animate import (
    NAMES30,
    Fit,
    Motion,
    Rig,
    RigError,
    bind_gaussians,
    forward_kinematics,
    gait_cycle,
    matrix_from_quat,
    phase_frames,
    pose_gaussians,
    quat_from_matrix,
    rest_locals,
    rodrigues,
    round_trip_error,
)

# A joint that the motion does not carry, so the bind-relative inheritance in
# `rest_locals` has something to resolve.
UNMAPPED = "HeadEnd"

# Enough hierarchy that a delta on a parent has to reach a grandchild; a flat
# rig would pass an FK test that in-place rotation also passes.
_CHAIN = {
    "Spine1": "Hips", "Spine2": "Spine1", "Chest": "Spine2",
    "Neck1": "Chest", "Neck2": "Neck1", "Head": "Neck2", UNMAPPED: "Head",
    "LeftShoulder": "Chest", "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightShoulder": "Chest", "RightArm": "RightShoulder",
    "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftLeg": "Hips", "LeftShin": "LeftLeg", "LeftFoot": "LeftShin",
    "LeftToeBase": "LeftFoot",
    "RightLeg": "Hips", "RightShin": "RightLeg", "RightFoot": "RightShin",
    "RightToeBase": "RightFoot",
}


def _toy_rig(n_verts: int = 40, seed: int = 0) -> Rig:
    names = list(NAMES30) + [UNMAPPED]
    idx = {n: i for i, n in enumerate(names)}
    parent = np.full(len(names), -1, np.int64)
    for child, par in _CHAIN.items():
        parent[idx[child]] = idx[par]
    for n in names:                       # anything unplaced hangs off the root
        if n != "Hips" and parent[idx[n]] < 0:
            parent[idx[n]] = idx["Hips"]

    rng = np.random.default_rng(seed)
    bind = np.tile(np.eye(4), (len(names), 1, 1))
    bind[:, :3, 3] = rng.normal(0.0, 0.4, (len(names), 3))
    bind[idx["Hips"], :3, 3] = 0.0

    lbs_idx = rng.integers(0, len(names), (n_verts, 8))
    w = rng.random((n_verts, 8))
    return Rig(
        vertices=rng.normal(0.0, 0.5, (n_verts, 3)),
        bind=bind,
        names=names,
        parent=parent,
        lbs_idx=lbs_idx,
        lbs_w=w / w.sum(1, keepdims=True),
    )


def _toy_motion(rig: Rig, T: int = 60, period: int = 12, seed: int = 1) -> Motion:
    """Globals for the 30 carried joints, plus a clean periodic heel contact."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((T, 30, 3))
    rot = np.tile(np.eye(3), (T, 30, 1, 1))
    for t in range(T):
        ang = 2 * np.pi * t / period
        for j, name in enumerate(NAMES30):
            base = rig.bind[rig.index[name], :3, 3]
            pos[t, j] = base + np.array([0.3 * t / T, 0.02 * np.sin(ang), 0.0])
            rot[t, j] = rodrigues(np.array([[0.0, 0.15 * np.sin(ang + j), 0.0]]))[0]
        pos[t, 0] += rng.normal(0.0, 1e-6, 3)

    contact = np.zeros((T, 4), bool)
    contact[::period, 0] = True           # one left-heel strike per period
    contact[1::period, 0] = True
    return Motion(pos=pos, rot=rot, contact=contact, fps=30.0, source="toy")


def _toy_gaussians(rig: Rig, motion: Motion, fit: Fit, seed: int = 2) -> dict:
    """Gaussians sitting on the fitted body, so a bind has something to grip."""
    rng = np.random.default_rng(seed)
    G = forward_kinematics(rig, rest_locals(rig, motion, fit.frame), fit.delta)
    from navsafe.benchmark.editing.assets.animate import _asset_affine, apply

    lin, tr = _asset_affine(rig, G, rig.lbs_idx, rig.lbs_w, fit)
    xyz = np.einsum("nij,nj->ni", lin, rig.vertices) + tr
    xyz = xyz + rng.normal(0.0, 0.01, xyz.shape)
    q = rng.normal(size=(len(xyz), 4))
    return {
        "xyz": xyz.astype(np.float32),
        "normals": np.zeros((len(xyz), 3), np.float32),
        "f_dc": rng.random((len(xyz), 3)).astype(np.float32),
        "f_rest": np.zeros((len(xyz), 0), np.float32),
        "opacity": np.full(len(xyz), 3.0, np.float32),
        "scale": rng.normal(-4.0, 0.2, (len(xyz), 3)).astype(np.float32),
        "rot": (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32),
    }


def _toy_fit(seed: int = 3) -> Fit:
    rng = np.random.default_rng(seed)
    return Fit(frame=7, delta=rng.normal(0.0, 0.05, (30, 3)), yaw=0.7,
               scale=1.05, offset=np.array([0.2, 0.0, -0.3]), chamfer=0.05)


# --------------------------------------------------------------------- maths
def test_rodrigues_identity_and_known_rotation():
    assert np.allclose(rodrigues(np.zeros((1, 3)))[0], np.eye(3))
    half_turn_z = rodrigues(np.array([[0.0, 0.0, np.pi]]))[0]
    assert np.allclose(half_turn_z @ np.array([1.0, 0.0, 0.0]),
                       [-1.0, 0.0, 0.0], atol=1e-12)


def test_quaternion_matrix_round_trip():
    rng = np.random.default_rng(0)
    R = rodrigues(rng.normal(0.0, 1.0, (64, 3)))
    back = matrix_from_quat(quat_from_matrix(R))
    assert np.allclose(back, R, atol=1e-10)


def test_quaternion_round_trip_survives_a_half_turn():
    """The trace-positive branch is the easy one; the others are where a sign
    slip hides, and a half turn about each axis lands squarely in them."""
    R = rodrigues(np.array([[np.pi, 0.0, 0.0], [0.0, np.pi, 0.0], [0.0, 0.0, np.pi]]))
    assert np.allclose(matrix_from_quat(quat_from_matrix(R)), R, atol=1e-9)


def test_fk_delta_propagates_down_the_chain():
    """Rotating a shoulder must carry the hand with it.

    This is the bug the first implementation shipped: rotating each joint in
    place left every child where it was, so the optimiser saw almost no
    gradient and the fit sat still.
    """
    rig = _toy_rig()
    motion = _toy_motion(rig)
    base = rest_locals(rig, motion, 3)
    delta = np.zeros((30, 3))
    delta[NAMES30.index("LeftShoulder")] = [0.0, 0.9, 0.0]

    plain = forward_kinematics(rig, base)
    turned = forward_kinematics(rig, base, delta)
    hand = rig.index["LeftHand"]
    foot = rig.index["RightFoot"]
    assert np.linalg.norm(turned[hand, :3, 3] - plain[hand, :3, 3]) > 0.05
    assert np.allclose(turned[foot], plain[foot], atol=1e-12)


def test_unmapped_joint_keeps_its_bind_offset():
    """A joint the motion does not carry rides its parent instead of dangling."""
    rig = _toy_rig()
    motion = _toy_motion(rig)
    G = forward_kinematics(rig, rest_locals(rig, motion, 5))
    head, tip = rig.index["Head"], rig.index[UNMAPPED]
    expected = np.linalg.norm(rig.bind[tip, :3, 3] - rig.bind[head, :3, 3])
    got = np.linalg.norm(G[tip, :3, 3] - G[head, :3, 3])
    assert got == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------- bind/pose
def test_bind_then_pose_reproduces_the_source_asset():
    """Phase 0 is the fitted frame, so it must return what it was given.

    Anything else means the bind and the pose disagree about a transform, and
    every other phase is wrong in a way a picture would not reveal.
    """
    rig = _toy_rig()
    motion = _toy_motion(rig)
    fit = _toy_fit()
    g = _toy_gaussians(rig, motion, fit)

    rigged = bind_gaussians(g, rig, motion, fit)
    assert round_trip_error(rigged, rig, motion) < 1e-9

    back = pose_gaussians(rigged, rig, motion, fit.frame)
    assert np.allclose(back["xyz"], g["xyz"], atol=1e-5)
    # A quaternion and its negation are the same rotation, so compare rotations.
    assert np.allclose(matrix_from_quat(np.asarray(back["rot"], np.float64)),
                       matrix_from_quat(np.asarray(g["rot"], np.float64)), atol=1e-6)


def test_posing_to_another_frame_actually_moves_the_gaussians():
    rig = _toy_rig()
    motion = _toy_motion(rig)
    fit = _toy_fit()
    rigged = bind_gaussians(_toy_gaussians(rig, motion, fit), rig, motion, fit)
    moved = pose_gaussians(rigged, rig, motion, fit.frame + 4)
    assert np.abs(np.asarray(moved["xyz"]) - rigged.gaussians["xyz"]).max() > 1e-3


def test_pose_keeps_every_other_gaussian_attribute():
    """Only geometry is re-posed; colour, opacity and size are the asset's."""
    rig = _toy_rig()
    motion = _toy_motion(rig)
    fit = _toy_fit()
    g = _toy_gaussians(rig, motion, fit)
    rigged = bind_gaussians(g, rig, motion, fit)
    out = pose_gaussians(rigged, rig, motion, fit.frame + 3)
    for key in ("f_dc", "opacity", "scale", "f_rest"):
        assert np.array_equal(out[key], g[key]), key


def test_bind_weights_sum_to_one():
    rig = _toy_rig()
    motion = _toy_motion(rig)
    fit = _toy_fit()
    rigged = bind_gaussians(_toy_gaussians(rig, motion, fit), rig, motion, fit)
    assert np.allclose(rigged.lbs_w.sum(1), 1.0, atol=1e-9)
    assert rigged.lbs_idx.shape == rigged.lbs_w.shape


# --------------------------------------------------------------------- gait
def test_gait_cycle_reads_the_period_off_the_heel_strikes():
    rig = _toy_rig()
    motion = _toy_motion(rig, T=60, period=12)
    period, stride = gait_cycle(motion)
    assert period == 12
    assert stride > 0.0


def test_gait_cycle_refuses_a_motion_with_no_contacts():
    rig = _toy_rig()
    motion = _toy_motion(rig)
    motion.contact = None
    with pytest.raises(RigError):
        gait_cycle(motion)


def test_gait_cycle_refuses_a_motion_too_short_to_hold_one():
    rig = _toy_rig()
    motion = _toy_motion(rig, T=60, period=12)
    motion.contact = np.zeros((60, 4), bool)
    motion.contact[5, 0] = True
    with pytest.raises(RigError):
        gait_cycle(motion)


def test_phase_frames_start_at_the_fit_and_stay_in_range():
    rig = _toy_rig()
    motion = _toy_motion(rig, T=60, period=12)
    frames = phase_frames(motion, 7, 6)
    assert len(frames) == 6
    assert frames[0] == 7
    assert all(0 <= f < len(motion) for f in frames)


def test_phase_frames_wrap_back_rather_than_run_off_the_end():
    """A cycle starting near the end pulls back by a whole period, which is the
    same limb pose -- rather than clamping, which would repeat one frame."""
    rig = _toy_rig()
    motion = _toy_motion(rig, T=60, period=12)
    frames = phase_frames(motion, len(motion) - 3, 6)
    assert all(0 <= f < len(motion) for f in frames)
    assert len(set(frames)) > 1
