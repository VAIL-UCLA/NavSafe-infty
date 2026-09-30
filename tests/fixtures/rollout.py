import torch
from types import SimpleNamespace
import numpy as np
class _FakeRepresentation:
    """Minimal stand-in for a RepresentationWrapper (no env, no IsaacLab)."""

    def __init__(self) -> None:
        self.dt = 0.1
        self.env = SimpleNamespace()
        self._ego = {
            "position": np.zeros(3, dtype=np.float64),
            "heading": 0.0,
            "velocity": np.zeros(3, dtype=np.float64),
            "speed": 0.0,
        }
        self._steps = 0
        # Ego-control handover, recorded so tests can assert on it. This fake
        # previously mirrored the real wrapper's gap (set_ego_override only),
        # which is precisely why rollout.py's `hasattr` guard silently skipped
        # the handover in production and the camera rendered from the LOGGED
        # ego's pose. A fake that omits the method cannot catch that bug.
        self.externally_driven = None
        self.override_cleared = 0

    def reset_to_scene(self, scene_index: int = 0):
        self._ego = {
            "position": np.zeros(3, dtype=np.float64),
            "heading": 0.0,
            "velocity": np.zeros(3, dtype=np.float64),
            "speed": 0.0,
        }
        self._steps = 0
        return {}, {}

    def get_ego_state(self):
        out = dict(self._ego)
        out["position"] = self._ego["position"].copy()
        return out

    def get_camera_images(self, cam_configs):
        return {cam: np.zeros((1080, 1920, 3), dtype=np.uint8) for cam in cam_configs}

    def set_ego_override(self, position, heading, velocity=None):
        p = np.asarray(position, dtype=np.float64)
        self._ego["position"] = np.array([p[0], p[1], 0.0], dtype=np.float64)
        self._ego["heading"] = float(heading)
        if velocity is not None:
            v = np.asarray(velocity, dtype=np.float64)
            self._ego["velocity"] = v
            self._ego["speed"] = float(np.linalg.norm(v[:2]))

    def set_ego_externally_driven(self, active: bool = True) -> None:
        self.externally_driven = bool(active)

    def clear_ego_override(self) -> None:
        self.override_cleared += 1

    def step(self, action):
        self._steps += 1
        terminated = np.array([False])
        truncated = np.array([self._steps >= 10_000])
        return {}, 0.0, terminated, truncated, {}

    def get_scenario_info(self):
        return {"metadata": {"scenario_id": "fake_scene"}}


class _StubAdapter:
    """Stub planner returning a fixed straight-ahead-ish trajectory."""

    def __init__(self, lateral_bias: float = 0.0) -> None:
        self._lat = lateral_bias

    def get_camera_configs(self):
        return {c: {"width": 1920, "height": 1080} for c in ("CAM_F0", "CAM_L0", "CAM_R0")}

    def get_waypoint_dt(self):
        return 0.5

    def prepare_input(self, images, ego_state, scenario_data, frame_id):
        return {
            "camera_feature": torch.zeros(1, 3, 256, 1024),
            "lidar_feature": torch.zeros(1, 1, 256, 256),
            "status_feature": torch.zeros(1, 8),
        }

    def run_inference(self, model_input):
        return model_input

    def parse_output(self, model_output, ego_state):
        traj = np.zeros((8, 2), dtype=np.float64)  # [lateral, forward]
        traj[:, 0] = self._lat
        traj[:, 1] = np.arange(1, 9) * 2.0  # advance forward
        return {"trajectory": traj}


# ── Rollout collector ──────────────────────────────────────────────────────

