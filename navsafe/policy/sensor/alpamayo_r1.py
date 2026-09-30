"""
Alpamayo-R1 adapter — ported from BridgeSim.
Uses persistent venv subprocess for Alpamayo inference.
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple
from collections import deque

import numpy as np
import torch

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy


def _navsafe_package_root() -> Path:
    """The inner ``navsafe`` package directory, the one holding ``modelzoo/``.

    Found by name rather than by counting ``.parent`` hops. The count here was
    written for ``navsafe/policy/alpamayo_r1.py``; once the file moved into
    ``sensor/`` it was one short, so ``parents[2]`` returned the package
    directory while the caller still appended ``"navsafe"`` to it and every
    default path came out doubled as
    ``<repo>/navsafe/navsafe/modelzoo/nvidia/...``. Nothing raised — the
    subprocess simply failed to start, a long way from the cause.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "navsafe":
            return parent
    raise RuntimeError(f"no 'navsafe' package directory above {here}")


def _default_alpamayo_root() -> Path:
    return _navsafe_package_root() / "modelzoo" / "nvidia" / "alpamayo"


def _default_alpamayo_python() -> str:
    return os.environ.get(
        "ALPAMAYO_PYTHON",
        str(_default_alpamayo_root() / "ar1_venv" / "bin" / "python"),
    )


def _default_alpamayo_script() -> str:
    return os.environ.get(
        "ALPAMAYO_SCRIPT",
        str(_navsafe_package_root() / "modelzoo" / "nvidia" / "tools" / "alpamayo_bridgesim_infer.py"),
    )


@dataclass
class AlpamayoR1AdapterConfig:
    camera_names: Tuple[str, ...] = ("CAM_CROSS_LEFT", "CAM_FRONT_WIDE", "CAM_CROSS_RIGHT", "CAM_FRONT_TELE")
    frames_per_camera: int = 4
    fallback_camera: str = "rgb_camera"
    alp_python: str = field(default_factory=_default_alpamayo_python)
    alp_script: str = field(default_factory=_default_alpamayo_script)
    top_p: float = 0.98
    temperature: float = 0.6
    num_traj_samples: int = 20
    max_generation_length: int = 256
    coord_mode: str = "x_forward_y_left"
    max_history: int = 20
    waypoint_dt: float = 0.1
    time_horizon_s: float = 6.4
    cuda_launch_blocking: bool = False


class AlpamayoSubprocessClient:
    """Persistent subprocess wrapper for Alpamayo inference."""

    def __init__(self, alp_python, alp_script, model_name, *, top_p, temperature,
                 num_traj_samples, max_generation_length, coord_mode, cuda_launch_blocking):
        self.alp_python = alp_python
        self.alp_script = alp_script
        self._proc: Optional[subprocess.Popen] = None

        cmd = ["env", "-u", "PYTHONUTF8", "-u", "PYTHONHOME", "-u", "PYTHONPATH"]
        if cuda_launch_blocking:
            cmd += ["CUDA_LAUNCH_BLOCKING=1"]
        cmd += [str(alp_python), str(alp_script),
                "--model", str(model_name), "--top-p", str(top_p),
                "--temperature", str(temperature), "--num-traj-samples", str(num_traj_samples),
                "--max-generation-length", str(max_generation_length), "--coord-mode", str(coord_mode)]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True)
        ready_line = self._proc.stdout.readline().strip()
        if ready_line != "READY":
            self._proc.kill()
            raise RuntimeError(f"Expected 'READY' from server, got: {ready_line!r}")

    def infer(self, image_stack_uint8_nhwc, ego_xyz_world, ego_rot_yaw, *, nav_cmd, num_inference_groups=0):
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Server process is not running.")
        img = image_stack_uint8_nhwc
        if img.ndim == 3: img = img[None, ...]
        if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            np.save(td / "img.npy", img)
            np.save(td / "ego_xyz.npy", ego_xyz_world)
            np.save(td / "ego_rot.npy", ego_rot_yaw)
            req = {"image_npy": str(td / "img.npy"), "ego_xyz_npy": str(td / "ego_xyz.npy"),
                   "ego_rot_npy": str(td / "ego_rot.npy"), "nav_cmd": nav_cmd}
            if num_inference_groups > 0:
                req["num_inference_groups"] = num_inference_groups
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()

        if not response_line:
            raise RuntimeError("Server process closed stdout unexpectedly.")
        result = json.loads(response_line)
        if "error" in result:
            raise RuntimeError(f"Server error: {result.get('error')}\n{result.get('traceback', '')}")
        return result

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __del__(self):
        self.close()


@register_policy("alpamayo_r1")
class AlpamayoR1Adapter(SensorPolicy):
    """Adapter for Alpamayo-R1, using a persistent Alpamayo venv subprocess."""

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 scorer=None, num_groups: int = 1, **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.scorer = scorer
        self.num_groups = num_groups
        self._current_frame_id = 0

        alp_python = kwargs.get("alp_python") or _default_alpamayo_python()
        alp_script = kwargs.get("alp_script") or _default_alpamayo_script()

        self.cfg = AlpamayoR1AdapterConfig(
            camera_names=tuple(kwargs.get("camera_names", AlpamayoR1AdapterConfig.camera_names)),
            frames_per_camera=int(kwargs.get("frames_per_camera", AlpamayoR1AdapterConfig.frames_per_camera)),
            fallback_camera=str(kwargs.get("fallback_camera", AlpamayoR1AdapterConfig.fallback_camera)),
            alp_python=str(alp_python), alp_script=str(alp_script),
            top_p=float(kwargs.get("top_p", 0.98)),
            temperature=float(kwargs.get("temperature", 0.6)),
            num_traj_samples=int(kwargs.get("num_traj_samples", 1)),
            max_generation_length=int(kwargs.get("max_generation_length", 256)),
            coord_mode=str(kwargs.get("coord_mode", "x_forward_y_left")),
            max_history=int(kwargs.get("max_history", 20)),
            waypoint_dt=float(kwargs.get("waypoint_dt", 0.1)),
            time_horizon_s=float(kwargs.get("time_horizon_s", 6.4)),
            cuda_launch_blocking=bool(kwargs.get("cuda_launch_blocking", False)),
        )

        self._img_hist_by_cam: Dict[str, Deque[np.ndarray]] = {
            cam: deque(maxlen=self.cfg.frames_per_camera) for cam in self.cfg.camera_names
        }
        self._ego_pos_hist: Deque[np.ndarray] = deque(maxlen=self.cfg.max_history)
        self._ego_yaw_hist: Deque[float] = deque(maxlen=self.cfg.max_history)
        self.client: Optional[AlpamayoSubprocessClient] = None

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        cfg: Dict[str, Dict[str, float]] = {
            "CAM_FRONT_WIDE": {"x": 0.80, "y": 0.0, "z": 1.60, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "fov": 120, "width": 1920, "height": 1080},
            "CAM_FRONT_TELE": {"x": 0.80, "y": 0.0, "z": 1.60, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "fov": 30, "width": 1920, "height": 1080},
            "CAM_CROSS_LEFT": {"x": 0.40, "y": -0.55, "z": 1.60, "yaw": -90.0, "pitch": 0.0, "roll": 0.0, "fov": 120, "width": 1920, "height": 1080},
            "CAM_CROSS_RIGHT": {"x": 0.40, "y": 0.55, "z": 1.60, "yaw": 90.0, "pitch": 0.0, "roll": 0.0, "fov": 120, "width": 1920, "height": 1080},
        }
        cfg["CAM_F0"] = cfg["CAM_FRONT_WIDE"]
        return cfg

    def load_model(self):
        print("Loading Alpamayo-R1 adapter (persistent venv subprocess)...")
        if not os.path.isfile(self.cfg.alp_python):
            raise FileNotFoundError(
                f"Alpamayo venv python not found: {self.cfg.alp_python}\n"
                f"Set ALPAMAYO_PYTHON to your alpamayo venv python.")
        if not os.path.isfile(self.cfg.alp_script):
            raise FileNotFoundError(
                f"Alpamayo glue script not found: {self.cfg.alp_script}\n"
                f"Set ALPAMAYO_SCRIPT env var.")

        self.client = AlpamayoSubprocessClient(
            alp_python=self.cfg.alp_python, alp_script=self.cfg.alp_script,
            model_name=self.checkpoint_path, top_p=self.cfg.top_p,
            temperature=self.cfg.temperature, num_traj_samples=self.cfg.num_traj_samples,
            max_generation_length=self.cfg.max_generation_length,
            coord_mode=self.cfg.coord_mode, cuda_launch_blocking=self.cfg.cuda_launch_blocking)
        print("Alpamayo-R1 adapter ready.")

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        if frame_id == 0:
            for dq in self._img_hist_by_cam.values(): dq.clear()
            self._ego_pos_hist.clear()
            self._ego_yaw_hist.clear()

        for cam in self.cfg.camera_names:
            img = images.get(cam) or images.get(self.cfg.fallback_camera)
            if img is None and len(images) > 0:
                img = next(iter(images.values()))
            if img is None:
                img = np.zeros((320, 576, 3), dtype=np.uint8)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            self._img_hist_by_cam[cam].append(img)

        frames_flat = []
        for cam in self.cfg.camera_names:
            dq = self._img_hist_by_cam[cam]
            if len(dq) == 0:
                pad = np.zeros((320, 576, 3), dtype=np.uint8)
                frames = [pad] * self.cfg.frames_per_camera
            else:
                frames = list(dq)
                if len(frames) < self.cfg.frames_per_camera:
                    frames = [frames[0]] * (self.cfg.frames_per_camera - len(frames)) + frames
                else:
                    frames = frames[-self.cfg.frames_per_camera:]
            frames_flat.extend(frames)

        img_stack = np.stack(frames_flat, axis=0)

        pos = np.array(ego_state["position"], dtype=np.float32)
        yaw = float(ego_state["heading"])
        self._ego_pos_hist.append(pos)
        self._ego_yaw_hist.append(yaw)

        T = len(self._ego_pos_hist)
        ego_hist_xyz = np.zeros((1, 1, T, 3), dtype=np.float32)
        ego_hist_yaw = np.zeros((1, 1, T, 1), dtype=np.float32)
        for i, p in enumerate(self._ego_pos_hist):
            ego_hist_xyz[0, 0, i, :] = p
        for i, y in enumerate(self._ego_yaw_hist):
            ego_hist_yaw[0, 0, i, 0] = y

        nav_cmd = None
        for key in ("route_commands", "route_cmds", "nav_cmds", "commands"):
            if isinstance(scenario_data.get(key), (list, tuple)) and frame_id < len(scenario_data[key]):
                nav_cmd = scenario_data[key][frame_id]
                break
        if nav_cmd is None:
            nav_cmd = ego_state.get("command", "STRAIGHT")
        nav_cmd = str(nav_cmd)
        if nav_cmd.isdigit():
            n = int(nav_cmd)
            nav_cmd = "LEFT" if n in (0, 1) else ("RIGHT" if n in (4, 5) else "STRAIGHT")

        return {"img_stack": img_stack, "ego_xyz_world": ego_hist_xyz,
                "ego_yaw_world": ego_hist_yaw, "nav_cmd": nav_cmd}

    def forward_inference_scaling(self, model_input, num_groups=1):
        if self.client is None:
            raise RuntimeError("AlpamayoR1Adapter: client is None. Did you call load_model()?")
        result = self.client.infer(
            image_stack_uint8_nhwc=model_input["img_stack"],
            ego_xyz_world=model_input["ego_xyz_world"],
            ego_rot_yaw=model_input["ego_yaw_world"],
            nav_cmd=model_input.get("nav_cmd", "STRAIGHT"),
            num_inference_groups=num_groups)
        if "all_candidates" not in result:
            raise RuntimeError("Server did not return all_candidates.")
        candidates_np = np.array(result["all_candidates"], dtype=np.float32)
        candidates_tensor = torch.from_numpy(candidates_np).unsqueeze(0)
        candidates_full_np = np.array(result["all_candidates_full"], dtype=np.float32)
        candidates_full_tensor = torch.from_numpy(candidates_full_np).unsqueeze(0)
        return {"all_candidates": candidates_tensor, "all_candidates_full": candidates_full_tensor,
                "confidence_scores": None, "scorer_context": {}}

    def run_inference(self, model_input: Any) -> Any:
        if self.client is None:
            raise RuntimeError("AlpamayoR1Adapter: client is None. Did you call load_model()?")
        if self.scorer is not None:
            return self.forward_inference_scaling(model_input, num_groups=self.num_groups)
        return self.client.infer(
            image_stack_uint8_nhwc=model_input["img_stack"],
            ego_xyz_world=model_input["ego_xyz_world"],
            ego_rot_yaw=model_input["ego_yaw_world"],
            nav_cmd=model_input.get("nav_cmd", "STRAIGHT"))

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        if self.scorer is not None and "all_candidates" in model_output:
            result = self.scorer.select_best(model_output, ego_state=ego_state, frame_idx=self._current_frame_id)
            best_idx = result["best_idx"][0].item()
            if "all_candidates_full" in model_output:
                trajectory = model_output["all_candidates_full"][0, best_idx].cpu().numpy()
            else:
                trajectory = result["trajectory"][0].cpu().numpy()
            traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
            parsed = {"trajectory": traj_swapped, "best_idx": best_idx,
                      "num_candidates": model_output["all_candidates"].shape[1]}
            all_cands = model_output["all_candidates"][0].cpu().numpy()
            parsed["trajectory_coarse"] = np.stack([np.column_stack([c[:, 1], c[:, 0]]) for c in all_cands])
            parsed["coarse_scores"] = result["scores"][0].cpu().numpy()
            return parsed

        traj_list = model_output.get("trajectory_lateral_forward")
        if traj_list is None:
            return {"trajectory": np.zeros((10, 2), dtype=np.float32)}
        traj = np.array(traj_list, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 2:
            traj = np.zeros((10, 2), dtype=np.float32)
        out: Dict[str, Any] = {"trajectory": traj}
        reasoning = model_output.get("reasoning")
        if reasoning is not None:
            out["reasoning"] = str(reasoning)
        return out

    def get_waypoint_dt(self) -> float:
        return float(self.cfg.waypoint_dt)

    def get_trajectory_time_horizon(self) -> float:
        return float(self.cfg.time_horizon_s)
