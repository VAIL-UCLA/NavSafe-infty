#!/usr/bin/env python
"""AutoVLA inference server: Qwen2.5-VL-3B + physical action tokens.

Faithful port of ``AutoVLA.predict`` (ucla-mobility/AutoVLA,
``models/autovla.py``) without importing the repo's training stack: the
base Qwen2.5-VL-3B-Instruct is loaded, the 2048 ``<action_i>`` tokens are
added via the repo's ``ActionTokenizer`` (imported standalone — its module
has no navsim deps), embeddings resized, and the released merged
checkpoint's state dict loaded on top.  The prompt/video layout follows
``get_prompt`` verbatim (3 cameras x 4 frames at 2 Hz as three videos).

Run under the VLA venv:
    python autovla_server.py --base-model <Qwen2.5-VL-3B-Instruct dir> \
        --checkpoint <AutoVLA_PDMS_89.ckpt> --repo <AutoVLA repo clone>

Request : {"frames": {"front": [4 jpg paths], "front_left": [...],
           "front_right": [...]}, "velocity": float, "acceleration": float,
           "instruction": str}
Response: {"trajectory": [[x,y,heading]*10], "cot": str}   # dt 0.5 s, 5 s
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

USE_COT = True
ACTION_START_ID = 151665
MIN_PIXELS = MAX_PIXELS = 109760  # 28*28*140 per released nuplan-grpo-cot config
GEN_MAX_LENGTH = 2048
GEN_TEMPERATURE = 0.01
GEN_TOP_P = 1.0

SYSTEM_COT = (
    "You are an Advanced Driver Assistance and Full Self-Driving System. "
    "You will receive visual observations from the ego vehicle’s cameras and dynamic information about the vehicle’s current state. "
    "Your task is to predict the optimal driving action for the next five seconds.\n\n"
    "First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.\n\n"
    "If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.\n\n"
    "Structure your reasoning as follows:\n"
    "1. **Scene Analysis**: Describe the traffic situation, including relevant environmental cues such as traffic lights, lane markings, and the behaviors of surrounding vehicles or pedestrians.\n"
    "2. **Identification of Critical Objects**: Identify two to three critical road users or obstacles, specifying their relative positions to the ego vehicle.\n"
    "3. **Prediction of Critical Object Behavior**: Predict the potential movements of the identified critical objects.\n"
    "4. **Ego Vehicle Intent Reasoning**: Based on the observed environment and current vehicle state, reason about the desired intent of the ego vehicle.\n"
    "5. **Final Action Decision**: Select one lateral action and one longitudinal action:\n"
    "- **Lateral actions** (choose exactly one): [move forward, turn left, change lane to left, turn right, change lane to right]\n"
    "- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero, maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]\n\n"
    "Present the final action clearly after your reasoning steps."
)


def build_messages(frames: dict, velocity: float, acceleration: float,
                   instruction: str):
    def video_item(paths):
        return {"type": "video", "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                "video": [f"file://{p}" for p in paths]}

    user_content = [
        {"type": "text", "text": (
            "The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment.")},
        {"type": "text", "text": "The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        video_item(frames["front"]),
        {"type": "text", "text": "The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        video_item(frames["front_left"]),
        {"type": "text", "text": "The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz."},
        video_item(frames["front_right"]),
        {"type": "text", "text": (
            f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
            f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds.")},
    ]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_COT}]},
        {"role": "user", "content": user_content},
    ]


class AutoVLAServer:
    def __init__(self, base_model: str, checkpoint: str, repo: str,
                 device: str = "cuda"):
        sys.path.insert(0, repo)
        from models.action_tokenizer import ActionTokenizer  # noqa: E402

        print(f"[server] loading base VLM from {base_model}", file=sys.stderr)
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(base_model)
        model_config = {
            "tokens": {"action_start_id": ACTION_START_ID},
            "codebook_cache_path": str(Path(repo) / "codebook_cache" / "agent_vocab.pkl"),
        }
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer, model_config)
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))

        print(f"[server] loading checkpoint {checkpoint}", file=sys.stderr)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # The Lightning ckpt was saved with transformers 4.49 key layout
        # (``visual.*``, ``model.*``, ``lm_head.*``); transformers >= 4.52
        # renamed Qwen2.5-VL submodules to ``model.visual.*`` /
        # ``model.language_model.*``.  Remap to whichever layout the loaded
        # model actually uses.
        target_keys = set(self.vlm.state_dict().keys())
        remapped = {}
        for k, v in state.items():
            k = k.replace("autovla.", "").replace("drivevla.", "")
            if not k.startswith("vlm."):
                continue
            k = k[len("vlm."):]
            if k in target_keys:
                remapped[k] = v
                continue
            if k.startswith("visual."):
                k2 = "model." + k
            elif k.startswith("model."):
                k2 = "model.language_model." + k[len("model."):]
            else:
                k2 = k
            if k2 in target_keys:
                remapped[k2] = v
        missing, unexpected = self.vlm.load_state_dict(remapped, strict=False)
        print(f"[server] ckpt: {len(remapped)} keys, {len(missing)} missing, "
              f"{len(unexpected)} unexpected", file=sys.stderr)
        self.vlm.eval()
        self.device = device

    @torch.no_grad()
    def infer(self, frames, velocity, acceleration, instruction):
        messages = build_messages(frames, velocity, acceleration, instruction)
        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.device)
        outputs = self.vlm.generate(
            **inputs,
            max_length=GEN_MAX_LENGTH,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_k=0,
            top_p=GEN_TOP_P,
        )
        trimmed = outputs[0][inputs.input_ids.shape[1]:][:-1].cpu()
        cot = self.processor.decode(trimmed)
        action_tokens = trimmed[trimmed >= ACTION_START_ID]
        if len(action_tokens) == 0:
            return None, cot
        traj = self.action_tokenizer.decode_token_ids_to_trajectory(action_tokens)[0, 1:]
        return np.asarray(traj, dtype=np.float32), cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()

    server = AutoVLAServer(args.base_model, args.checkpoint, args.repo)
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            traj, cot = server.infer(req["frames"], float(req["velocity"]),
                                     float(req["acceleration"]),
                                     str(req["instruction"]))
            resp = {"trajectory": None if traj is None else traj.tolist(),
                    "cot": cot[-2000:]}
            print(json.dumps(resp), flush=True)
        except Exception as exc:  # noqa: BLE001 — report to client, keep serving
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)


if __name__ == "__main__":
    main()
