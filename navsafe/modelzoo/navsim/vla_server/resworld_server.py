#!/usr/bin/env python
"""ResWorld inference server: temporal-residual BEV world model + planner.

Port of the ResWorld eval path (``mengtan00/ResWorld``,
``projects/mmdet3d_plugin/resworld/resworld.py``) to a closed-loop,
one-request-per-replan server.

The reference's ``forward_test``/``simple_test``/``simple_test_pts`` cannot be
called while driving: ``simple_test_pts`` computes STP3 planning metrics inline
against ground-truth boxes, maps, attribute labels and ``ego_fut_trajs``, none
of which exist closed-loop.  This server therefore calls ``extract_feat`` and
``pts_bbox_head`` directly -- the same bypass ``drivelaw_server`` makes, for the
same reason -- and reproduces the dataset pipeline
(``PrepareImageInputs``/``mmlabNormalize``) rather than reimplementing it.

What the head actually consumes (``resworld_head.forward``):

    bev_inputs   [bev_feats, can_bus]   can_bus is (num_frames, B, 18)
    cmd          (B, 1, 1, 3)           one-hot; indexed by torch.nonzero
    ego_his_trajs / ego_lcf_feat        IGNORED -- the released config sets
                                        ego_lcf_feat_idx=None, so the decoder
                                        input dim is embed_dims alone

Three frames, not one.  ``multi_adj_frame_id_cfg = (1, 3, 1)`` gives
``num_adj = 2`` and ``num_frames = 3``: TR-World's residuals ARE the differences
between consecutive BEV token sets, so a single frame would make every residual
zero and the world-model branch a no-op.  Images stack camera-major /
frame-minor -- for each camera, current then adjacents -- because
``BEVDet4D.prepare_inputs`` reads ``view(B, N, num_frame, C, H, W)``.

Run under the ResWorld venv (``NAVSAFE_RESWORLD_PYTHON``; py3.8, torch 1.9.1,
mmcv-full 1.4.0):
    python resworld_server.py --repo <ResWorld dir> --config <mmcv cfg> \
        --checkpoint <resworld_trained.pth>

Request : {"cam_npys": {"CAM_FRONT": p, ...}, "command": int,
           "can_bus": [18 floats],
           "calibration": {"CAM_FRONT": {"intrinsics": 3x3,
                                         "sensor2lidar_rotation": 3x3,
                                         "sensor2lidar_translation": 3}, ...}}
Response: {"trajectory": [[x, y]*6]}   # cumulative, LiDAR frame, dt 0.5 s

``calibration`` is required on the first request and cached after it.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np
import torch

# resworld_config.py: fut_ts = valid_fut_ts = 6, ego_fut_mode = 3.
FUT_TS = 6
EGO_FUT_MODE = 3
CAN_BUS_DIM = 18
# data_config['cams'] -- this order IS the model's channel order.
CAM_NAMES = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]


def _mmlab_normalize(img: np.ndarray) -> torch.Tensor:
    """The reference's ``mmlabNormalize``.

    ``imnormalize(..., to_rgb=True)`` swaps channels ITSELF, so the array
    handed in here must still be **BGR** -- the renderer's native order. An
    adapter that helpfully pre-converts to RGB would have the swap applied
    twice and feed the backbone a channel-swapped picture, which is invisible
    at runtime and only shows up as a quietly worse score.
    """
    from mmcv.image.photometric import imnormalize

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    out = imnormalize(np.asarray(img), mean, std, True)
    return torch.tensor(out).float().permute(2, 0, 1).contiguous()


def _sample_augmentation(height: int, width: int, input_size, resize_test: float):
    """Test-mode branch of ``PrepareImageInputs.sample_augmentation``.

    Reproduced rather than imported because the pipeline class is bound to a
    dataset ``results`` dict. ``crop_h`` is (0.0, 0.0) in the released config,
    so the crop keeps the BOTTOM ``fH`` rows -- the road, not the sky.
    """
    fH, fW = input_size
    resize = float(fW) / float(width) + resize_test
    resize_dims = (int(width * resize), int(height * resize))
    new_w, new_h = resize_dims
    crop_h = int((1 - np.mean((0.0, 0.0))) * new_h) - fH
    crop_w = int(max(0, new_w - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return resize, resize_dims, crop


def _img_transform(pil_img, resize_dims, crop):
    """``img_transform_core`` with flip=False, rotate=0 (the test path)."""
    return pil_img.resize(resize_dims).crop(crop)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ResWorld checkout")
    ap.add_argument("--config", required=True, help="mmcv config .py")
    ap.add_argument("--checkpoint", required=True, help="trained .pth")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)

    from PIL import Image
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    import importlib

    cfg = Config.fromfile(args.config)
    # The config's model/dataset classes live in the repo's plugin package and
    # only register themselves on import; mmdet3d's registry is empty without
    # it and build_model raises KeyError('ResWorld').
    if cfg.get("plugin", False):
        importlib.import_module(cfg.get("plugin_dir", "projects.mmdet3d_plugin")
                                .rstrip("/").replace("/", "."))
    else:
        importlib.import_module("projects.mmdet3d_plugin")

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # map_location='cpu' then .cuda(): the checkpoint was saved from a 4-GPU
    # run and carries device tags that do not exist on a 1-GPU machine.
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    del checkpoint
    model.cuda().eval()

    data_config = cfg.data_config
    input_size = tuple(data_config["input_size"])
    resize_test = float(data_config.get("resize_test", 0.0))
    num_frames = model.num_frame

    # Calibration arrives in the request rather than being imported: the
    # client starts this server with `env -u PYTHONPATH`, so `navsafe` is not
    # importable here, and copying OPENSCENE_CAMERA_PARAMS into this file
    # would fork the rig definition. The adapter owns it; the rig is rigid, so
    # it is cached on first use and identical every replan after.
    calib_cache: dict = {}

    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cam_npys = req["cam_npys"]
            missing = [c for c in CAM_NAMES if c not in cam_npys]
            if missing:
                raise ValueError(f"ResWorld needs all six cameras; missing {missing}")
            if not calib_cache:
                calib = req.get("calibration")
                if not calib:
                    raise ValueError(
                        "the first request must carry `calibration`: this "
                        "server cannot import navsafe to look the rig up")
                calib_cache.update(calib)

            imgs, sensor2egos, ego2globals = [], [], []
            intrins, post_rots, post_trans = [], [], []
            for cam in CAM_NAMES:
                arr = np.load(cam_npys[cam])
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError(f"{cam}: expected (H, W, 3), got {arr.shape}")
                pil = Image.fromarray(arr.astype(np.uint8))
                resize, resize_dims, crop = _sample_augmentation(
                    pil.height, pil.width, input_size, resize_test)
                warped = _img_transform(pil, resize_dims, crop)

                post_rot = torch.eye(3)
                post_tran = torch.zeros(3)
                post_rot[:2, :2] = torch.eye(2) * resize
                post_tran[:2] = -torch.Tensor(crop[:2])

                # Camera-major, frame-minor. Closed loop there is exactly one
                # rendered frame per replan, so the adjacents repeat it: the
                # residual is then zero, which is the correct statement that
                # nothing is known to have moved -- not a fabricated motion.
                for _ in range(num_frames):
                    imgs.append(_mmlab_normalize(np.array(warped)))

                params = calib_cache[cam]
                sensor2ego = torch.eye(4)
                sensor2ego[:3, :3] = torch.tensor(
                    np.asarray(params["sensor2lidar_rotation"]), dtype=torch.float32)
                sensor2ego[:3, 3] = torch.tensor(
                    np.asarray(params["sensor2lidar_translation"]), dtype=torch.float32)
                intrin = torch.tensor(
                    np.asarray(params["intrinsics"]), dtype=torch.float32)

                for _ in range(num_frames):
                    sensor2egos.append(sensor2ego.clone())
                    # Closed loop the ego frame IS the key frame, so ego2global
                    # is identity: prepare_inputs only uses it to align
                    # adjacents onto the key ego, and identity makes that a
                    # no-op instead of applying a pose we do not have.
                    ego2globals.append(torch.eye(4))
                    intrins.append(intrin.clone())
                    post_rots.append(post_rot.clone())
                    post_trans.append(post_tran.clone())

            img_inputs = [
                torch.stack(imgs).unsqueeze(0).cuda(),
                torch.stack(sensor2egos).unsqueeze(0).cuda(),
                torch.stack(ego2globals).unsqueeze(0).cuda(),
                torch.stack(intrins).unsqueeze(0).cuda(),
                torch.stack(post_rots).unsqueeze(0).cuda(),
                torch.stack(post_trans).unsqueeze(0).cuda(),
                torch.eye(3).unsqueeze(0).cuda(),   # bda: no BEV augmentation
            ]

            can_bus = np.asarray(req.get("can_bus") or [0.0] * CAN_BUS_DIM,
                                 dtype=np.float32)
            if can_bus.shape != (CAN_BUS_DIM,):
                raise ValueError(f"can_bus must be {CAN_BUS_DIM} floats, "
                                 f"got {can_bus.shape}")
            # (num_frames, B, 18) -- head does can_bus.permute(1, 0, 2).
            can_bus_t = (torch.tensor(can_bus).view(1, 1, CAN_BUS_DIM)
                         .repeat(num_frames, 1, 1).cuda())

            cmd_idx = int(req["command"])
            if not 0 <= cmd_idx < EGO_FUT_MODE:
                raise ValueError(f"command must be 0..{EGO_FUT_MODE - 1}, got {cmd_idx}")
            cmd = torch.zeros(1, 1, 1, EGO_FUT_MODE)
            cmd[0, 0, 0, cmd_idx] = 1.0
            cmd = cmd.cuda()

            img_metas = [dict(box_type_3d=None)]
            with torch.no_grad():
                img_feats, _, _ = model.extract_feat(
                    points=None, img=img_inputs, img_metas=img_metas)
                outs = model.pts_bbox_head(
                    [img_feats[0], can_bus_t], img_metas, prev_bev=None, cmd=cmd,
                    ego_his_trajs=None, ego_lcf_feat=None)

            # (B, ego_fut_mode, fut_ts, 2) per-step deltas -> select the
            # commanded mode and cumsum, exactly as simple_test_pts does.
            preds = outs["ego_fut_preds"]
            if preds.shape[1:] != (EGO_FUT_MODE, FUT_TS, 2):
                raise ValueError(
                    f"ego_fut_preds has shape {tuple(preds.shape)}, expected "
                    f"(B, {EGO_FUT_MODE}, {FUT_TS}, 2)")
            traj = preds[0, cmd_idx].cumsum(dim=-2).float().cpu().numpy()
            print(json.dumps({"trajectory": traj.tolist()}), flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
