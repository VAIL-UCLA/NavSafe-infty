#!/usr/bin/env python
"""SimWAM inference server: the action expert of a world-action model.

Port of the SimWAM NAVSIM eval path (H-EmbodVis/SimWAM,
``experiments/navsim/eval_navsim.py`` + ``src/simwam/datasets/navsim``) to a
closed-loop, one-request-per-replan server.

SimWAM co-trains a Wan2.2-5B video expert and a lightweight action DiT with
joint flow matching, but an isolated attention mask keeps action tokens
independent of future frames -- so **the video branch is discarded at
inference** and what runs here is a self-contained planner. That is why this
adapter is cheap next to the other world-model rows: no future frames are
generated, and ``infer_mode`` on navtest is ``action_only``.

Model construction goes through the repo's own Hydra task config and factory
(``simwam.runtime.create_simwam``) rather than a hand-rolled equivalent, so a
config change upstream cannot silently desync this server. Checkpoint loading
mirrors ``eval_navsim._load_model_checkpoint`` (L122-165): ``mot`` or ``dit``
weights plus an optional ``proprio_encoder``.

Contract, taken from the dataset rather than assumed:

* **image** -- one front camera, resized to 384x672 and normalised to
  [-1, 1] (``navsim_dataset._build_video_tensor``: ``normalize(mean=.5,
  std=.5)``). ``infer_action`` requires both sides to be multiples of 16.
* **proprio** -- 8 dims, ``[velocity(2), acceleration(2), driving_command(4)]``
  (``_extract_ego_features``, L506-514). Note the order: the adapters' shared
  status vector puts the command first, so the server reorders rather than
  making the adapter special.
* **prompt** -- one fixed string (``build_prompt_fixed``; the released navsim
  data config sets ``use_dynamic_prompt: false``). It is encoded once at
  start-up and the umT5 encoder is then dropped: no encoding runs per replan,
  and ~11 GB stays off the GPU for the rest of the episode.
* **output** -- ``[8, 3]`` normalised, denormalised with ``denorm_odo`` for
  ``trajectory_mode: absolute`` (the released RL task's setting).

The Wan2.2 side weights (VAE, umT5 encoder, tokenizer) are read from
``<model zoo>/_base/diffsynth`` by default, resolved from ``NAVSAFE_MODEL_ZOO``
or ``NAVSAFE_DATA_ROOT`` as ``navsafe.data_paths`` does; ``--model-base-path``
overrides it.

Run under the VLA venv:
    python simwam_server.py --repo <SimWAM dir> --checkpoint <SimWAM-RL.pt>

Request : {"image_npy": path, "status": [cmd0..3, vx, vy, ax, ay]}
Response: {"trajectory": [[x, y, heading]*8]}   # navsim ego frame, dt 0.5 s
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# navsim_dataset: the released front-camera task trains at 384x672, and
# infer_action rejects anything not a multiple of 16.
IMAGE_HW = (384, 672)
CONTEXT_LEN = 256      # configs/data/navsim.yaml
ACTION_HORIZON = 8
# The repo's root eval config; `task=` selects the preset under configs/task/.
CONFIG_NAME = "sim_navsim"
# The supervised task, used for BOTH released checkpoints. README evaluates the
# RL checkpoint under the flowgrpo task, but that only matters while LoRA
# adapters are still separate: the released SimWAM-RL.pt is merged (its `mot`
# key set is byte-for-byte the same 1649 names as SimWAM.pt, with zero `lora`
# entries), so the plain `create_simwam` factory loads it exactly. Going through
# the grpo task instead would build a LoRA wrapper with nothing to put in it and
# would demand SIMWAM_IL_CHECKPOINT on top.
DEFAULT_TASK = "navsim_uncond_front_384x672_1e-4"


# Wan2.2-TI2V-5B side weights (VAE + umT5 encoder + tokenizer). DiffSynth's
# loader defaults to ModelScope and to a cwd-relative ./checkpoints/; both are
# wrong here, so they are pinned before the loader runs. `redirect_common_files`
# is switched off with them: its redirect target
# (DiffSynth-Studio/Wan-Series-Converted-Safetensors) is a ModelScope-only repo
# and 404s on the Hub, while the official Wan-AI repo carries the same VAE and
# encoder as .pth.
def _default_model_base() -> str:
    """``<model zoo>/_base/diffsynth``, resolved as ``navsafe.data_paths`` does.

    The server cannot import navsafe (the client strips PYTHONPATH), so the
    two environment variables are read here directly.
    """
    data_root = os.path.expanduser(
        os.environ.get("NAVSAFE_DATA_ROOT", "~/data/NavSafe"))
    model_zoo = os.path.expanduser(
        os.environ.get("NAVSAFE_MODEL_ZOO", os.path.join(data_root, "model_zoo")))
    return os.path.join(model_zoo, "_base", "diffsynth")


# Future-frame video. SimWAM's paper omits frame generation at deployment --
# the isolated attention mask lets the action expert run without it -- but the
# video expert is not removed: both released checkpoints carry the full
# ``mixtures.video.*`` weights, and ``infer_joint`` decodes them through the
# Wan2.2 VAE. Off by default so the scored path stays byte-identical; the
# trajectory it returns is NOT the one ``infer_action`` produces (upstream
# only asserts they agree to atol/rtol 1e-2), so a video run must not be
# reported as the scored run.
VIDEO_FRAMES = 17      # 4n+1 for the VAE's temporal stride


def _write_video(frames, out_dir: Path, index: int, fps: int) -> None:
    """``infer_joint``'s decoded frames -> ``wm_NNNNN.mp4``.

    Best-effort by design: this runs inside the scoring loop, so a codec that
    is missing on the eval image must degrade to PNGs rather than kill the
    episode. ``_decode_latents`` hands back PIL images; anything tensor-shaped
    is normalised here so a future upstream change does not silently write
    black frames.
    """
    if frames is None:
        return
    try:
        import numpy as _np
        from PIL import Image as _Image

        if torch.is_tensor(frames):                      # (T,C,H,W) or (C,T,H,W)
            arr = frames.detach().float().cpu()
            if arr.ndim == 4 and arr.shape[0] in (1, 3) and arr.shape[1] > 4:
                arr = arr.permute(1, 0, 2, 3)            # -> (T,C,H,W)
            if arr.numel() and float(arr.max()) <= 1.0:
                arr = arr * 255.0
            frames = [_Image.fromarray(f.permute(1, 2, 0).clamp(0, 255)
                                       .to(torch.uint8).numpy())
                      for f in arr]
        frames = [f if isinstance(f, _Image.Image)
                  else _Image.fromarray(_np.asarray(f).astype("uint8"))
                  for f in frames]
        if not frames:
            return
        path = out_dir / f"wm_{index:05d}.mp4"
        try:
            import imageio.v2 as imageio
            with imageio.get_writer(path, fps=fps, macro_block_size=None) as w:
                for f in frames:
                    w.append_data(_np.asarray(f))
        except Exception as exc:                          # noqa: BLE001
            stem = out_dir / f"wm_{index:05d}"
            stem.mkdir(exist_ok=True)
            for i, f in enumerate(frames):
                f.save(stem / f"{i:03d}.png")
            print(f"[simwam] mp4 unavailable ({exc}); wrote {len(frames)} PNGs "
                  f"to {stem}", file=sys.stderr, flush=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"[simwam] video dump skipped: {exc}", file=sys.stderr, flush=True)


def _load_checkpoint(model, ckpt: str) -> None:
    """eval_navsim._load_model_checkpoint, in brief.

    Kept non-strict for the same reason the reference is: the RL checkpoint
    carries LoRA-adapted action-DiT weights over a base the factory already
    built, so extra/missing keys are expected and are reported instead of
    raising.
    """
    # mmap: the payload is 12 GB, and reading it into RAM on top of a model that
    # is already ~24 GB there takes the process past 41 GiB -- on a machine
    # that also runs IsaacSim and the renderer.
    # Memory-mapped, the tensors are copied straight into the module's storage
    # and the file stays in reclaimable page cache. Older checkpoints that are
    # not zipfile-serialised cannot be mapped, so the plain read is kept as a
    # fallback rather than made a hard requirement.
    try:
        payload = torch.load(ckpt, map_location="cpu", mmap=True)
    except (RuntimeError, ValueError) as exc:
        print(f"[simwam] mmap load unavailable ({exc}); reading into RAM",
              file=sys.stderr, flush=True)
        payload = torch.load(ckpt, map_location="cpu")
    if "mot" in payload and hasattr(model, "mot"):
        inc = model.mot.load_state_dict(payload["mot"], strict=False)
    elif "dit" in payload and hasattr(model, "dit"):
        inc = model.dit.load_state_dict(payload["dit"], strict=False)
    else:
        raise ValueError(f"{ckpt}: no `mot`/`dit` weights in the payload "
                         f"(keys: {sorted(payload)})")
    if getattr(model, "proprio_encoder", None) is not None and "proprio_encoder" in payload:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    print(f"[simwam] loaded {ckpt} step={payload.get('step')} "
          f"missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}",
          file=sys.stderr, flush=True)
    # The weights are in the modules now; holding the payload as well doubles
    # what the process needs for no purpose.
    del payload
    gc.collect()


def _build_prompt() -> str:
    """The one prompt the released navsim task conditions on.

    ``configs/data/navsim.yaml`` ships ``use_dynamic_prompt: false`` and the
    task config does not override it, so ``build_prompt_fixed`` returns this
    single fixed string for **every** sample -- which is what "uncond" in
    ``navsim_uncond_front_384x672_1e-4`` means. Describing the ego's own motion
    here instead (speed bucket, turn direction) conditions the action expert on
    text it never saw in training, and that description is derived from the very
    state the model is being asked to extrapolate.

    Verbatim from ``navsim_dataset.build_prompt_fixed``, trailing space
    included -- it is inside the encoded string.
    """
    past_seconds, future_seconds = 2.0, 4.0
    return (
        "A high-quality, photorealistic dashboard camera view of autonomous driving. "
        f"Based on the past {past_seconds:.0f} seconds videos, "
        f"predict and generate the next {future_seconds:.0f} seconds of realistic driving continuation, "
        "Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
        "clear details, and realistic physics. "
    )


def _pad_context(context, context_mask):
    """Pad an encoding out to the task's ``context_len``.

    ``encode_prompt`` pads to the model's ``tokenizer_max_len`` (128), but the
    task was trained on cached contexts of length 256 whose mask is all ones
    (``navsim_dataset`` zeroes the pad rows, then unmasks them). Those pad rows
    are therefore *attended*, so handing the DiT 128 of them instead of 256
    changes how much softmax mass lands on the real tokens.
    """
    import torch

    have = context.shape[-2]
    if have > CONTEXT_LEN:
        raise ValueError(
            f"context is {have} tokens, longer than context_len {CONTEXT_LEN}")
    if have < CONTEXT_LEN:
        pad = torch.zeros(*context.shape[:-2], CONTEXT_LEN - have,
                          context.shape[-1],
                          dtype=context.dtype, device=context.device)
        context = torch.cat([context, pad], dim=-2)
        mask_pad = torch.ones(
            *context_mask.shape[:-1], CONTEXT_LEN - context_mask.shape[-1],
            dtype=context_mask.dtype, device=context_mask.device)
        context_mask = torch.cat([context_mask, mask_pad], dim=-1)
    return context.contiguous(), context_mask.contiguous()


def _denorm_odo(traj: torch.Tensor) -> torch.Tensor:
    """navsim_dataset.denorm_odo (trajectory_mode: absolute)."""
    x = (traj[..., 0:1] + 1) / 2 * 66.74 - 1.57
    y = (traj[..., 1:2] + 1) / 2 * 42 - 19.68
    h = (traj[..., 2:3] + 1) / 2 * 3.53 - 1.67
    return torch.cat([x, y, h], dim=-1)


def _image_tensor_from_array(arr, where: str = "<array>") -> torch.Tensor:
    """(1, 3, 384, 672) in [-1, 1] from an (H, W, 3) RGB frame.

    Split from _image_tensor so a caller that already holds the pixels can have
    THIS preprocessing rather than its own copy of it. The resize below is the
    part that must not drift between the two, since a differently-aliased
    1920->672 downscale would change what the model sees.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize as tv_resize

    path = where
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: expected an (H, W, 3) RGB frame, got {arr.shape}")
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    # The dataset resizes with torchvision's antialiased bilinear; plain
    # F.interpolate aliases badly at this ~2.9x downscale (1920 -> 672).
    t = tv_resize(t, list(IMAGE_HW),
                  interpolation=InterpolationMode.BILINEAR, antialias=True)
    return (t - 0.5) / 0.5


def _image_tensor(path: str) -> torch.Tensor:
    """As above, from the saved .npy frame the server takes on the wire."""
    return _image_tensor_from_array(np.load(path), path)


def build(args):
    """Construct the model and the cached prompt context, ready to infer.

    Split out of main() so a second consumer (for example, a script that hangs
    a forward hook on the model) can have the SAME model this server serves. A
    copy of these lines elsewhere would silently stop matching the day one of
    them changes.
    """

    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", args.model_base_path)
    os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

    repo = Path(args.repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "navsim"))

    from hydra import compose, initialize_config_dir  # noqa: E402
    from hydra.utils import instantiate  # noqa: E402

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 is what the checkpoints were saved in (`torch_dtype` in the payload)
    # and what eval_navsim runs, via mixed_precision=bf16.
    dtype = torch.bfloat16
    with initialize_config_dir(config_dir=str((repo / "configs").resolve()),
                               version_base=None):
        cfg = compose(config_name=CONFIG_NAME, overrides=[
            f"task={args.task}",
            # encode_prompt() raises without it, and the task config ships it
            # off because training reads precomputed context off the dataset.
            "model.load_text_encoder=true",
            # The checkpoint carries both experts in full, and the pretrained
            # action-DiT the config points at (checkpoints/ActionDiT_*.pt) is
            # not published, so there is nothing to skip past.
            "model.skip_dit_load_from_pretrain=true",
            "model.redirect_common_files=false",
        ])
    # BUILD ON THE CPU, THEN MOVE. The factory takes one `device` for every
    # submodule, so asking it for cuda puts the video DiT, the VAE, the action
    # DiT *and* the umT5 text encoder on the card at once -- about 23.5 GiB in
    # bf16, which overruns a 24 GB card partway through ActionDiT. The text
    # encoder is ~11 GiB of that and is used exactly once, for a prompt that
    # never changes, so it has no business being on the GPU at all. Building on
    # the CPU, encoding there and releasing it before the move leaves ~12.5 GiB
    # to transfer, which fits with room for the renderer's share.
    #
    # `self.device` is a plain attribute the constructor stores; `.to()` does
    # not update it, and it is what infer_action sends its inputs to, so it has
    # to be reassigned by hand after the move.
    model = instantiate(cfg.model, model_dtype=dtype, device="cpu")
    _load_checkpoint(model, args.checkpoint)
    model = model.to("cpu").eval()

    # The text encoder is the only piece of the video branch inference still
    # needs, and only for the prompt. The bucket space is finite, so encode all
    # of it now and release the encoder rather than caching lazily. On the CPU
    # this is a few seconds, once, at startup.
    with torch.no_grad():
        ctx_cache = _pad_context(*model.encode_prompt(_build_prompt()))
    model.text_encoder = None
    model.tokenizer = None
    gc.collect()

    model.device = torch.device(device)
    model = model.to(device).eval()
    ctx_cache = tuple(t.to(device) for t in ctx_cache)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        print(f"[simwam] on {device}: "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              file=sys.stderr, flush=True)
    print(f"[simwam] cached the fixed prompt encoding ({CONTEXT_LEN} tokens); "
          f"text encoder released", file=sys.stderr, flush=True)

    return model, ctx_cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="SimWAM checkout")
    ap.add_argument("--checkpoint", required=True,
                    help="SimWAM-RL.pt (91.5 PDMS) or SimWAM.pt (90.3)")
    ap.add_argument("--task", default=DEFAULT_TASK, help="Hydra task config name")
    # configs/sim_navsim.yaml resolves num_inference_steps to
    # eval_num_inference_steps = 10 (configs/train.yaml), with seed 0.
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="default: cuda when available, else cpu")
    ap.add_argument("--model-base-path", default=_default_model_base(),
                    help="cache for the Wan2.2 VAE / umT5 encoder / tokenizer "
                         "(default: <model zoo>/_base/diffsynth)")
    ap.add_argument("--video-dir", default=os.environ.get("NAVSAFE_SIMWAM_VIDEO_DIR"),
                    help="write the video expert's future frames here, one mp4 "
                         "per replan. Runs infer_joint instead of infer_action: "
                         "much slower, and the trajectory is not bit-identical "
                         "to the scored path, so do not reuse those numbers.")
    ap.add_argument("--video-frames", type=int, default=VIDEO_FRAMES,
                    help="frames per replan; the VAE's temporal stride wants 4n+1")
    ap.add_argument("--video-fps", type=int, default=8)
    args = ap.parse_args()
    model, ctx_cache = build(args)

    video_dir = None
    video_idx = 0
    if args.video_dir:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[simwam] video expert ON -> {video_dir} "
              f"({args.video_frames} frames @ {args.video_fps} fps per replan); "
              f"trajectories will differ from the scored run",
              file=sys.stderr, flush=True)

    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            status = [float(v) for v in req["status"]]
            cmd_onehot, vel, acc = status[0:4], status[4:6], status[6:8]
            # SimWAM's own order: velocity, acceleration, command.
            proprio = torch.tensor(vel + acc + cmd_onehot, dtype=torch.float32)
            context, context_mask = ctx_cache

            with torch.no_grad():
                if video_dir is None:
                    out = model.infer_action(
                        prompt=None, context=context, context_mask=context_mask,
                        input_image=_image_tensor(req["image_npy"]),
                        action_horizon=int(args.action_horizon),
                        proprio=proprio,
                        num_inference_steps=int(args.num_inference_steps),
                        seed=int(args.seed),
                    )
                else:
                    # test_action_with_infer_action=False: upstream's check
                    # re-runs infer_action purely to warn on divergence, which
                    # doubles the cost of every replan.
                    out = model.infer_joint(
                        prompt=None, context=context, context_mask=context_mask,
                        input_image=_image_tensor(req["image_npy"]),
                        num_video_frames=int(args.video_frames),
                        action_horizon=int(args.action_horizon),
                        proprio=proprio,
                        num_inference_steps=int(args.num_inference_steps),
                        seed=int(args.seed),
                        test_action_with_infer_action=False,
                    )
                    _write_video(out.get("video"), video_dir, video_idx,
                                 int(args.video_fps))
                    video_idx += 1
            traj = out["action"] if isinstance(out, dict) and "action" in out else out
            traj = torch.as_tensor(traj).detach().float().reshape(-1, 3).cpu()
            traj = _denorm_odo(traj)
            print(json.dumps({"trajectory": traj.numpy().tolist()}), flush=True)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
