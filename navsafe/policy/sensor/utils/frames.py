# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Renderer channel order, and why every camera adapter has to convert.

``CameraManager`` hands adapters **BGR** uint8 frames: the replicator captures
RGB and the manager swaps it at capture time (``navsafe/env/camera_manager.py``
lines 888 / 962 / 1023) so the evaluator's artifacts can go straight through
``cv2.imwrite``, which reads its input as BGR.

Every released driving model in the zoo was trained on **RGB**. navsim and
nuPlan load frames with PIL (``Image.open(...).convert("RGB")``), the ImageNet
normalisation constants the backbones carry are RGB-ordered
(``mean = (123.675, 116.28, 103.53)``), and the VLM processors take PIL images.
An adapter that forwards the renderer array unchanged therefore hands the model
a channel-swapped picture: the sky comes out orange, foliage teal, a red
traffic light **blue**, and a bus lane's red paint blue — which is precisely
the cue a red-light or lane-marking scenario is meant to test.

The conversion is one line. It lives here anyway, because the failure is
invisible: nothing raises, the plan still looks plausible, and the only symptom
is a score that is quietly too low. Naming the conversion makes it testable.
"""

from __future__ import annotations

import numpy as np


def renderer_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert one renderer frame from BGR to the RGB the models expect.

    Args:
        image: ``(H, W, 3)`` uint8 frame as delivered in the evaluator's
            ``images`` dict.

    Returns:
        A C-contiguous ``(H, W, 3)`` uint8 copy in RGB order.

    Raises:
        ValueError: the array is not a three-channel image. A silent pass
            through would reintroduce exactly the bug this module exists to
            stop, so a malformed frame fails loudly.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) renderer frame, got shape {arr.shape}")
    return np.ascontiguousarray(arr[:, :, ::-1].astype(np.uint8))




# navsim's CAM_F0 is 1920x1080. The sim rig carries navsim's intrinsics
# (fx 1545, principal point 960, 560) but renders 1120 rows, so the extra 40
# sit at the BOTTOM: rows 0..1079 of a rendered frame are pixel-for-pixel the
# navsim frame, and cropping them off needs no resampling.
NAVSIM_CAM_W, NAVSIM_CAM_H = 1920, 1080


def crop_to_navsim_aspect(image: np.ndarray) -> np.ndarray:
    """Trim a rendered frame to the 16:9 navsim CAM_F0 aspect.

    The renderer hands back 1920x1120 where navsim logged 1920x1080. Every
    model here was trained on the latter, and each one reacts differently to
    the extra 40 rows: ReCogDrive's dynamic tiling picks a 3x2 grid instead of
    4x2, which costs 512 image tokens and leaves 18% of its 2800-token budget
    as padding the model never saw at training; the fixed-size models simply
    resize a slightly taller picture into the same box.

    Because the principal point is shared, the correct trim is the bottom rows
    -- NOT a centre crop, which would move the horizon.

    The navsim-stitching adapters (``ltf``, ``diffusiondrive``,
    ``transfuser``, ``diffusiondrivev2``) need this too, for a second reason.
    They reproduce upstream's crop as ``int(28 * h / 1080)`` /
    ``int(416 * w / 1920)``, which scales correctly with
    ``camera_resolution_scale`` but cannot repair an aspect difference.
    Upstream's comment states the intent -- "crop to ensure 4:1 aspect ratio"
    -- and on 1920x1080 the arithmetic lands exactly there: sides 1088, front
    1920, stitched 4096x1024. On a 1920x1120 frame the same code yields
    4096x1062, aspect 3.857, and the fixed ``cv2.resize(..., (1024, 256))``
    then shrinks width 4.00x against height 4.15x. Measured in the 256-row
    tensor the model receives: objects 3-4 % shorter than trained, the horizon
    5 rows high, the road surface 8. The error is scale-invariant -- at
    ``camera_resolution_scale=0.5`` the stitch is 2048x532, aspect 3.85.

    Trimming first costs no resampling and restores 4:1 exactly.

    ``drivor`` (and ``prioreye``, which subclasses it) needs it for the plain
    form of the same reason: it resizes each camera straight to 1148x672, so a
    1120-row frame is scaled 672/1120 vertically against 1148/1920
    horizontally -- ratio 1.0035, where upstream's 1080 rows give 1.0407. That
    is a 3.6 % vertical squash and a horizon at row 336 of 672 instead of 348.

    Args:
        image: ``(H, W, 3)`` frame.

    Returns:
        The top ``round(W * 1080/1920)`` rows, or ``image`` unchanged when it
        is already that tall or shorter.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"expected an (H, W, 3) renderer frame, got shape {arr.shape}")
    want_h = int(round(arr.shape[1] * NAVSIM_CAM_H / NAVSIM_CAM_W))
    if arr.shape[0] <= want_h:
        return arr
    return np.ascontiguousarray(arr[:want_h])


__all__ = ["NAVSIM_CAM_H", "NAVSIM_CAM_W", "crop_to_navsim_aspect",
           "renderer_bgr_to_rgb"]
