#!/usr/bin/env python
"""MTDrive inference server: Qwen2.5-VL-7B text-out trajectory planner.

MTDrive (arXiv 2601.22930) released only checkpoints — no code.  This
server reimplements single-turn inference from the paper: the prompt is
verbatim ReCogDrive's front-camera trajectory-prediction prompt (MTDrive's
SFT data comes from the ReCogDrive dataset), the model answers with
``[PT, (x, y, heading), ... x8]``, which is regex-parsed.  The released
multi-turn refinement loop needs PDM feedback text; single-turn (turn 1)
is what runs closed-loop here.

Run under the VLA venv:
    python mtdrive_server.py --model-path <mtdrive_sft|mtdrive_rl_best dir>

Request : {"image_npy": path, "history": [[x,y,h]*4], "status": [8 floats]}
Response: {"trajectory": [[x,y,heading]*8], "text": generation}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# MTDrive released no code, so its prompt is ReCogDrive's, verbatim -- and
# that includes the system turn, which both ReCogDrive's eval agent and the
# SFT conversation set. Omitting it does not leave the slot empty: Qwen's
# chat template substitutes "You are a helpful assistant.", i.e. a different
# system prompt from the one used at training.
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


def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


def build_prompt(history, command_one_hot) -> str:
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
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
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


_POINT_RE = re.compile(
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")


def parse_trajectory(text: str, n_points: int = 8) -> np.ndarray | None:
    # Prefer the [PT ...] block; fall back to any (x, y, h) tuples in order.
    block = text
    m = re.search(r"\[\s*PT(.*?)\]", text, flags=re.S)
    if m:
        block = m.group(1)
    pts = _POINT_RE.findall(block)
    if len(pts) < n_points:
        pts = _POINT_RE.findall(text)
    if not pts:
        return None
    traj = np.array([[float(a), float(b), float(c)] for a, b, c in pts[:n_points]],
                    dtype=np.float32)
    if len(traj) < n_points:  # pad by repeating the last pose
        pad = np.repeat(traj[-1:], n_points - len(traj), axis=0)
        traj = np.concatenate([traj, pad], axis=0)
    return traj


class MTDriveServer:
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"[server] loading MTDrive from {model_path}", file=sys.stderr)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=device,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

    @torch.no_grad()
    def infer(self, image: np.ndarray, history, status):
        prompt = build_prompt(history, status[:4])
        pil = Image.fromarray(image.astype(np.uint8))
        messages = [{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        }, {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil], padding=True,
                                return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=1e-6,
            repetition_penalty=1.05,
        )
        trimmed = out[0][inputs.input_ids.shape[1]:]
        answer = self.processor.decode(trimmed, skip_special_tokens=True)
        traj = parse_trajectory(answer)
        return traj, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()

    server = MTDriveServer(args.model_path)
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            image = np.load(req["image_npy"])
            traj, answer = server.infer(image, req["history"], req["status"])
            resp = {"trajectory": None if traj is None else traj.tolist(),
                    "text": answer[-2000:]}
            print(json.dumps(resp), flush=True)
        except Exception as exc:  # noqa: BLE001 — report to client, keep serving
            print(json.dumps({"error": str(exc),
                              "traceback": traceback.format_exc()}), flush=True)


if __name__ == "__main__":
    main()
