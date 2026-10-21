#!/usr/bin/env python
"""DriveVLA-W0 inference server: Emu3 VQ + flow-matching action expert.

Port of ``inference/vla/inference_action_navsim_flow_matching_vava.py``
(``BraveGroup/DriveVLA-W0``) to a closed-loop, one-request-per-replan server.

The reference drives ``Emu3DrivingVAVADataset`` over a pickle of *pre-computed*
VQ codes and a token yaml naming every navtest scene; neither exists while
driving. This server keeps the two things that are actually the model
(``Emu3Pi0.sample_actions`` and the FAST action tokenizer), adds the vision
tokenizer the offline pass would have run, and rebuilds the prompt the dataset
would have produced.

Prompt, from ``Emu3DrivingVAVADataset.__getitem__``:

    pre_block = bos + pre_prompt + format_video_prompt(pre_codes)
                + FAST(pre past actions)
    cur_block =       prompt     + format_video_prompt(cur_codes)
                + FAST(past actions)
    input_ids = pre_block ++ cur_block

The dataset appends ``append_action_to_sample(sample, action_ids)`` after each
block -- the FUTURE actions, i.e. the training labels. ``sample_actions`` reads
only the prefix, so those are omitted here; including them would prefix the
model with the answer it is being asked for.

FAST ids are mapped into the tail of the vocabulary exactly as the dataset does
(``last_vocab_idx - id`` with ``last_vocab_idx = pad_token_id - 1``).

Run under the VLA venv:
    python drivevla_w0_server.py --repo <DriveVLA-W0 dir> --emu-hub <ckpt> \
        --vision-tokenizer <Emu3-VisionTokenizer> --vlm-model <Emu3-Stage1> \
        --norm-stats <norm_stats.json> --action-tokenizer <configs/fast>

Request : {"pre_image_npy": p, "cur_image_npy": p, "past_actions": [[x,y,h]*3],
           "command": [4 floats]}
Response: {"trajectory": [[dx, dy, dheading]*8]}   # per-step deltas, dt 0.5 s

The response rows are relative steps, each in the previous waypoint's frame;
the adapter composes them into ego-frame poses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

ACTION_FRAMES = 8
ACTION_DIM = 3
# emu3_tokenizer_navsim.py: SIZE = (W, H) = (256, 144); the VQ downsamples by
# 8, giving the 18x32 grid the prompt format hard-codes.
VQ_IMAGE_WH = (256, 144)
VQ_GRID_HW = (18, 32)
# The dataset's own category list; index order IS the one-hot order and
# matches navsim's [left, straight, right, unknown].
TEXT_NAME_LIST = ["go left", "go straight", "go right", "unknown"]


def _format_video_prompt(tokenizer, codes: torch.Tensor, pattern: str) -> str:
    """``Emu3SFTDataset.format_video_prompt`` for a (frames, h, w) code grid."""
    frames, h, w = codes.shape
    frame_strs = []
    for frame in codes:
        frame_strs.append("".join(
            pattern.format(token_id=int(t)) for t in frame.flatten()))
    videostr = tokenizer.eof_token.join(frame_strs)
    return (tokenizer.boi_token + f"{frames}*{h}*{w}" + tokenizer.img_token
            + videostr + tokenizer.eof_token + tokenizer.eoi_token)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="DriveVLA-W0 checkout")
    ap.add_argument("--emu-hub", required=True, help="trained Emu3Pi0 dir")
    ap.add_argument("--vision-tokenizer", required=True,
                    help="Emu3-VisionTokenizer dir")
    ap.add_argument("--vlm-model", required=True,
                    help="base Emu3 VLM dir (the reference's VLA_VLM_MODEL). "
                         "Emu3Pi0 loads its VLM with Emu3MoE.from_pretrained on "
                         "THIS path, which reads that directory's own config -- "
                         "the fine-tuned checkpoint's nested vlm_config omits "
                         "eov_token_id and friends, so pointing this at the "
                         "checkpoint fails in Emu3Model.__init__.")
    ap.add_argument("--norm-stats", required=True, help="norm_stats.json")
    ap.add_argument("--action-tokenizer", required=True,
                    help="FAST action tokenizer dir (configs/fast)")
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--model-max-length", type=int, default=1400)
    ap.add_argument("--visual-token-pattern", default="<|visual token {token_id:0>6d}|>")
    args = ap.parse_args()

    # Same path set the reference's own `config.setup_paths_early` installs:
    # `emu3` is vendored under reference/Emu3, not importable from the repo
    # root, and `train/` holds the DataArguments the dataset code shares.
    # Calling setup_paths_early itself is not an option -- it lives in
    # inference/vla/config.py, which is a top-level `config` module that would
    # shadow anything else by that name.
    for path in (args.repo,
                 os.path.join(args.repo, "reference", "Emu3"),
                 os.path.join(args.repo, "train")):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    import torch.nn.functional as F  # noqa: F401  (parity with the dataset)
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    from emu3.mllm import Emu3Pi0, Emu3Pi0Config, Emu3Tokenizer

    # From the VLM path, not the checkpoint: the reference builds its tokenizer
    # with `vlm_model_path_for_tokenizer`, and the two vocabularies have to be
    # the one the visual-token pattern was written against.
    tokenizer = Emu3Tokenizer.from_pretrained(
        args.vlm_model, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False)

    model_config = Emu3Pi0Config.from_pretrained(
        os.path.join(args.emu_hub, "config.json"))
    model, loading_info = Emu3Pi0.from_pretrained(
        args.emu_hub, config=model_config, pretrain_vlm_path=args.vlm_model,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        trust_remote_code=True, output_loading_info=True)
    # The reference prints these and moves on. Here a missing key means the
    # policy is running randomly-initialised weights, which scores like a bad
    # policy rather than like a bug -- a silent failure, so it is fatal.
    missing = list(loading_info.get("missing_keys") or [])
    unexpected = list(loading_info.get("unexpected_keys") or [])
    print(f"[drivevla_w0] missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)
    if missing:
        raise RuntimeError(
            f"Emu3Pi0 checkpoint is missing {len(missing)} tensors the model "
            f"needs, e.g. {missing[:8]}. Refusing to run randomly-initialised "
            f"weights.")
    model = model.cuda().eval()

    # The vision tokenizer the offline pass would have run.
    vq_model = AutoModel.from_pretrained(
        args.vision_tokenizer, trust_remote_code=True).eval().cuda()
    vq_processor = AutoImageProcessor.from_pretrained(
        args.vision_tokenizer, trust_remote_code=True)
    vq_processor.min_pixels = VQ_IMAGE_WH[0] * VQ_IMAGE_WH[1]

    # FAST action tokenizer (configs/fast), used for the past-action tokens.
    action_tokenizer = AutoProcessor.from_pretrained(
        args.action_tokenizer, trust_remote_code=True)
    last_vocab_idx = tokenizer.pad_token_id - 1
    # Resolved the way wrap_action_sequence does (`tokenizer.encode(boa_token)[0]`)
    # rather than read off the config, so the id and the vocabulary that will be
    # embedded cannot disagree. Cross-checked against config.boa_token_id below.
    boa_token_id = tokenizer.encode(tokenizer.boa_token)[0]
    cfg_boa = getattr(model_config, "boa_token_id", None)
    if cfg_boa is not None and int(cfg_boa) != int(boa_token_id):
        raise RuntimeError(
            f"tokenizer encodes boa_token as {boa_token_id} but the model config "
            f"says {cfg_boa}; the attention mask locates action blocks by this id")

    norm_cfg = json.load(open(args.norm_stats))
    action_low = torch.tensor(norm_cfg["norm_stats"]["libero"]["q01"]).cuda()
    action_high = torch.tensor(norm_cfg["norm_stats"]["libero"]["q99"]).cuda()

    from PIL import Image

    def encode_frame(path: str) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{path}: expected (H, W, 3) RGB, got {arr.shape}")
        pil = Image.fromarray(arr.astype(np.uint8)).resize(VQ_IMAGE_WH)
        pixel = vq_processor([pil], return_tensors="pt")["pixel_values"].cuda()
        with torch.no_grad():
            codes = vq_model.encode(pixel)
        grid = codes[0].detach().cpu()
        if tuple(grid.shape[-2:]) != VQ_GRID_HW:
            raise ValueError(
                f"VQ produced a {tuple(grid.shape[-2:])} grid, expected "
                f"{VQ_GRID_HW}; the prompt format depends on this shape")
        return grid.reshape(1, *VQ_GRID_HW)

    def fast_ids(actions: np.ndarray) -> list:
        ids = action_tokenizer(actions)[0]
        return [last_vocab_idx - i for i in ids]

    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            past = np.asarray(req["past_actions"], dtype=np.float32)
            if past.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"past_actions must be (n, {ACTION_DIM}), got {past.shape}")

            cur_codes = encode_frame(req["cur_image_npy"])
            pre_codes = encode_frame(req["pre_image_npy"])

            command = list(req["command"])
            prompt = TEXT_NAME_LIST[int(np.argmax(command))]

            cur_video = _format_video_prompt(tokenizer, cur_codes,
                                             args.visual_token_pattern)
            pre_video = _format_video_prompt(tokenizer, pre_codes,
                                             args.visual_token_pattern)

            cur_text = prompt + cur_video
            pre_text = tokenizer.bos_token + prompt + pre_video

            cur = tokenizer(cur_text, padding=False,
                            return_token_type_ids=False, return_tensors="pt")
            pre = tokenizer(pre_text, padding=False,
                            return_token_type_ids=False, return_tensors="pt")

            past_ids = torch.tensor(fast_ids(past), dtype=torch.long)
            past_mask = torch.ones_like(past_ids)

            # The BOA token that opens each block's action segment. The dataset
            # emits it as the head of `wrap_action_sequence`, and it is
            # STRUCTURAL rather than a label: `create_causal_style_attention_mask`
            # asserts exactly two BOAs per sample and lets the action expert
            # attend to "VLM content before the SECOND BOA". So the marker has
            # to be here even though the action tokens it would introduce must
            # not be -- those are the answer being asked for. Dropping the whole
            # wrapper (the obvious reading of "sample_actions reads only the
            # prefix") trips the assertion with counts: tensor([0]).
            boa = torch.tensor([boa_token_id], dtype=torch.long)
            boa_mask = torch.ones_like(boa)

            def _append(sample):
                return {
                    "input_ids": torch.cat(
                        [sample["input_ids"].squeeze(0), past_ids, boa], dim=-1),
                    "attention_mask": torch.cat(
                        [sample["attention_mask"].squeeze(0), past_mask, boa_mask],
                        dim=-1),
                }

            pre_block, cur_block = _append(pre), _append(cur)
            input_ids = torch.cat(
                [pre_block["input_ids"], cur_block["input_ids"]], dim=-1)
            attention_mask = torch.cat(
                [pre_block["attention_mask"], cur_block["attention_mask"]], dim=-1)

            padded = tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                padding="max_length", return_tensors="pt")
            # `pad` given a 1-D sequence returns 1-D, and sample_actions reads
            # `batch_size = input_ids.shape[0]` -- which then comes out as 1400
            # (the pad length) instead of 1, and the run dies on
            # `pre_action.view(1400, -1)` against 9 elements. Batch it here
            # rather than trusting pad to have inferred a batch dimension.
            ids = padded["input_ids"]
            mask = padded["attention_mask"]
            if ids.dim() == 1:
                ids, mask = ids.unsqueeze(0), mask.unsqueeze(0)
            if ids.shape[-1] > args.model_max_length:
                # The prompt is two video blocks plus two FAST runs; if a
                # future resolution change pushes it past the model's
                # positions, truncating would silently drop the current
                # frame's tail rather than the padding.
                raise ValueError(
                    f"prompt is {ids.shape[-1]} tokens, over the model's "
                    f"{args.model_max_length}; the VQ grid or the action "
                    f"token count must have changed")

            cmd = torch.tensor(command, dtype=torch.float32).unsqueeze(0)
            pre_action = torch.tensor(past, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                predicted = model.sample_actions(
                    input_ids=ids.cuda(),
                    attention_mask=mask.cuda(),
                    pre_action=pre_action.cuda().to(torch.bfloat16),
                    cmd=cmd.cuda().to(torch.bfloat16),
                    num_inference_steps=args.num_inference_steps,
                    action_frames=ACTION_FRAMES,
                    action_dim=ACTION_DIM,
                )

            # The reference's denormalisation, verbatim.
            denorm = 0.5 * (predicted.squeeze(0).float() + 1) * (
                action_high - action_low) + action_low
            traj = denorm.cpu().numpy().reshape(ACTION_FRAMES, ACTION_DIM)
            print(json.dumps({"trajectory": traj.tolist()}), flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
