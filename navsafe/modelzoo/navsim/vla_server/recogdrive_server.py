#!/usr/bin/env python
"""ReCogDrive inference server: InternVL3 VLM + DiT diffusion planner.

Faithful port of the ReCogDrive eval path (xiaomi-research/recogdrive,
``ReCogDriveAgent`` in no-cache mode): front-camera image -> InternVL
dynamic-tile preprocessing -> VLM forward with the trajectory-prediction
prompt -> last_hidden_state -> ``ReCogDriveDiffusionPlanner.get_action``.

Run under the VLA venv:
    python recogdrive_server.py --vlm-path <InternVL3-2B dir> \
        --planner-ckpt <ReCogDrive_Diffusion_Planner_2B_{IL,RL}.ckpt>

Request : {"image_npy": path, "history": [[x,y,h]*4], "status": [8 floats]}
Response: {"trajectory": [[x,y,heading]*8]}   # navsim ego frame, dt 0.5 s
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

_THIS_DIR = Path(__file__).resolve().parent
# Vendored planner package lives at navsafe/modelzoo/navsim/recogdrive;
# import it as the top-level package "recogdrive" so its relative imports
# (.blocks, .recogdrive_dit) resolve without importing navsafe itself.
sys.path.insert(0, str(_THIS_DIR.parent))
sys.path.insert(0, str(_THIS_DIR))

from recogdrive.recogdrive_diffusion_planner import (  # noqa: E402
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from internvl_conversation import get_conv_template  # noqa: E402

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
NUM_IMAGE_TOKEN = 256

# Verbatim from recogdrive_backbone.py.
SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios,
                                       orig_width, orig_height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image_array(arr: np.ndarray, input_size=448, max_num=12) -> torch.Tensor:
    image = Image.fromarray(arr.astype(np.uint8))
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size,
                                use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


def build_question(history, command_one_hot):
    commands = ["turn left", "go straight", "turn right"]
    command_str = "unknown"
    for i, v in enumerate(command_one_hot[:3]):
        if v == 1:
            command_str = commands[i]
            break
    history_str = " ".join(
        f"   - t-{3 - i}: ({format_number(history[i][0])}, "
        f"{format_number(history[i][1])}, {format_number(history[i][2])})"
        for i in range(4))
    prompt = (
        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Active navigation command: [{command_str.upper()}]"
    )
    output_requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    return f"{prompt}{output_requirements}"


def make_planner_config(dit_type: str, sampling_method: str = "ddim"):
    if dit_type == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,
                               "output_dim": 512}
        input_embedding_dim = 384
    elif dit_type == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,
                               "output_dim": 1536}
        input_embedding_dim = 1536
    else:
        raise ValueError(f"unknown dit_type {dit_type!r}")
    diffusion_model_cfg.update({"dropout": 0.0, "attention_bias": True,
                                "norm_eps": 1e-5, "interleave_attention": True})
    return ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=5,
        grpo=False,
        model_dtype="float16",
    )


class ReCogDriveServer:
    def __init__(self, vlm_path: str, planner_ckpt: str, dit_type: str,
                 vlm_size: str, device: str = "cuda"):
        self.device = device
        print(f"[server] loading InternVL from {vlm_path}", file=sys.stderr)
        self.model = AutoModel.from_pretrained(
            vlm_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
            device_map=device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            vlm_path, trust_remote_code=True, use_fast=False)
        self.model.system_message = SYSTEM_MESSAGE
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id

        print(f"[server] loading planner from {planner_ckpt}", file=sys.stderr)
        cfg = make_planner_config(dit_type)
        cfg.vlm_size = vlm_size
        self.planner = ReCogDriveDiffusionPlanner(cfg).cuda()
        ckpt = torch.load(planner_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # The RL checkpoint carries the trained policy AND a frozen `old_policy`
        # copy -- the GRPO reference, byte-identical to the IL checkpoint, 347
        # tensors of it. Dropped here by name rather than left to strict=False,
        # so the load reports on the tensors that actually matter: with it in,
        # the log read "694 keys, 0 missing, 347 unexpected", which is
        # indistinguishable from a checkpoint whose real weights were the ones
        # being discarded.
        planner_sd = {}
        dropped_ref = 0
        for k, v in state.items():
            k2 = k[len("agent."):] if k.startswith("agent.") else k
            if not k2.startswith("action_head."):
                continue
            k3 = k2[len("action_head."):]
            if k3.startswith("old_policy."):
                dropped_ref += 1
                continue
            planner_sd[k3] = v
        missing, unexpected = self.planner.load_state_dict(planner_sd, strict=False)
        print(f"[server] planner loaded: {len(planner_sd)} keys, "
              f"{len(missing)} missing, {len(unexpected)} unexpected"
              + (f", {dropped_ref} old_policy reference tensors ignored"
                 if dropped_ref else ""), file=sys.stderr)
        # A real mismatch must not be survivable: strict=False was hiding it.
        if missing or unexpected:
            raise RuntimeError(
                f"ReCogDrive planner state_dict mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected. First missing: {list(missing)[:5]}; "
                f"first unexpected: {list(unexpected)[:5]}")
        self.planner.eval()

    @torch.no_grad()
    def vlm_hidden_state(self, pixel_values: torch.Tensor, question: str) -> torch.Tensor:
        model_dtype = next(self.model.parameters()).dtype
        template = get_conv_template("internvl2_5")
        template.system_message = SYSTEM_MESSAGE
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        num_patches = pixel_values.size(0)
        image_tokens = (IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * NUM_IMAGE_TOKEN * num_patches
                        + IMG_END_TOKEN)
        query = query.replace("<image>", image_tokens, 1)
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer([query], return_tensors="pt",
                                      padding="max_length", max_length=2800)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        image_flags = torch.tensor([1] * num_patches, dtype=torch.long)
        outputs = self.model(
            pixel_values=pixel_values.to(self.device, model_dtype),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags.squeeze(-1),
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def infer(self, image: np.ndarray, history, status) -> np.ndarray:
        pixel_values = load_image_array(image, max_num=12)
        question = build_question(history, status[:4])
        last_hidden_state = self.vlm_hidden_state(pixel_values, question)

        model_dtype = next(self.planner.parameters()).dtype
        history_t = torch.tensor(history, dtype=torch.float32).view(1, -1).cuda()
        status_t = torch.tensor(status, dtype=torch.float32).view(1, -1).cuda()
        input_state = torch.cat([status_t, history_t], dim=1)
        action_inputs = BatchFeature({
            "state": input_state.to(model_dtype),
            "his_traj": history_t.to(model_dtype),
            "status_feature": status_t.to(model_dtype),
        })
        # No `deterministic` argument: upstream's only eval call site is
        #     recogdrive_agent.py:209
        #     return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)
        # which takes get_action's own default of False, leaving `self.eta` --
        # an EtaFixed nn.Parameter DiffGRPO trains -- live. Passing the flag at
        # all, either value, is a divergence from the reference implementation,
        # and both variants (forcing deterministic=True, or threading the flag
        # through an env var) scored worse in closed loop than this plain call
        # on the same (scenario, seed) pairs. For a GRPO-trained policy the
        # sampler is part of the policy; pool seeds to handle its variance
        # rather than reaching for the flag.
        pred = self.planner.get_action(last_hidden_state.to(model_dtype), action_inputs)
        return pred["pred_traj"].float().cpu().squeeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm-path", required=True)
    ap.add_argument("--planner-ckpt", required=True)
    ap.add_argument("--dit-type", default="small")
    ap.add_argument("--vlm-size", default="small")
    args = ap.parse_args()

    server = ReCogDriveServer(args.vlm_path, args.planner_ckpt,
                              args.dit_type, args.vlm_size)
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            image = np.load(req["image_npy"])
            traj = server.infer(image, req["history"], req["status"])
            print(json.dumps({"trajectory": traj.tolist()}), flush=True)
        except Exception as exc:  # noqa: BLE001 — report to client, keep serving
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)


if __name__ == "__main__":
    main()
