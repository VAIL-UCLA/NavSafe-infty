"""
OpenPilot model adapter — ported from BridgeSim.
Self-contained implementation — all necessary constants, warp computation,
YUV conversion, and output parsing are reimplemented in-file (no OpenPilot imports).
All imports use navsafe.* paths. No bridgesim/nuplan/metadrive deps.
"""

import codecs
import collections
import pickle
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy

# Constants from openpilot repo
T_IDXS = [10.0 * (i / 32) ** 2 for i in range(33)]
PLAN_MHP_N = 5
PLAN_WIDTH = 15
IDX_N = 33
FEATURE_LEN = 512
DESIRE_LEN = 8
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_FL = 910.0
MEDMODEL_CY = 47.6
SBIGMODEL_FL = 455.0
SBIGMODEL_CY = 151.8

VIEW_FRAME_FROM_DEVICE = np.array(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64
).T


def rot_from_euler(rpy):
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _build_model_intrinsics():
    med = np.array([[MEDMODEL_FL, 0, MEDMODEL_INPUT_SIZE[0] / 2.0],
                    [0, MEDMODEL_FL, MEDMODEL_CY], [0, 0, 1]], dtype=np.float64)
    sbig = np.array([[SBIGMODEL_FL, 0, MEDMODEL_INPUT_SIZE[0] / 2.0],
                     [0, SBIGMODEL_FL, SBIGMODEL_CY], [0, 0, 1]], dtype=np.float64)
    return med, sbig


def _build_calib_from_model(model_intrinsics):
    M = model_intrinsics @ VIEW_FRAME_FROM_DEVICE
    return np.linalg.inv(M)


def get_warp_matrix(device_from_calib_euler, camera_intrinsics, bigmodel_frame=False):
    med_intr, sbig_intr = _build_model_intrinsics()
    calib_from_model = _build_calib_from_model(sbig_intr if bigmodel_frame else med_intr)
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = camera_intrinsics @ VIEW_FRAME_FROM_DEVICE @ device_from_calib
    return camera_from_calib @ calib_from_model


def rgb_to_yuv6ch(rgb_512x256):
    r = rgb_512x256[:, :, 0].astype(np.int32)
    g = rgb_512x256[:, :, 1].astype(np.int32)
    b = rgb_512x256[:, :, 2].astype(np.int32)
    Y = np.clip((((b * 13 + g * 65 + r * 33) + 64) >> 7) + 16, 0, 255).astype(np.uint8)
    ch0 = Y[0::2, 0::2]; ch1 = Y[1::2, 0::2]; ch2 = Y[0::2, 1::2]; ch3 = Y[1::2, 1::2]
    r_sub = (r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2] + 2) >> 2
    g_sub = (g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2
    b_sub = (b[0::2, 0::2] + b[0::2, 1::2] + b[1::2, 0::2] + b[1::2, 1::2] + 2) >> 2
    U = np.clip((b_sub * 56 - g_sub * 37 - r_sub * 19 + 0x8080) >> 8, 0, 255).astype(np.uint8)
    V = np.clip((r_sub * 56 - g_sub * 47 - b_sub * 9 + 0x8080) >> 8, 0, 255).astype(np.uint8)
    return np.stack([ch0, ch1, ch2, ch3, U, V], axis=0)


def safe_exp(x, clip_val=88.0):
    return np.exp(np.clip(x, -clip_val, clip_val))


def softmax(x, axis=-1):
    e = safe_exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_mdn(raw, in_N, out_N, out_shape):
    raw = raw.reshape((raw.shape[0], max(in_N, 1), -1))
    n_values = (raw.shape[2] - out_N) // 2
    pred_mu = raw[:, :, :n_values]

    if in_N > 1:
        weights = np.zeros((raw.shape[0], in_N, out_N), dtype=raw.dtype)
        for i in range(out_N):
            weights[:, :, i - out_N] = softmax(raw[:, :, i - out_N], axis=-1)
        pred_mu_final = np.zeros((raw.shape[0], max(out_N, 1), n_values), dtype=raw.dtype)
        for fidx in range(weights.shape[0]):
            for hidx in range(out_N):
                idxs = np.argsort(weights[fidx, :, hidx])[::-1]
                pred_mu_final[fidx, hidx] = pred_mu[fidx, idxs[0]]
    else:
        pred_mu_final = pred_mu

    if out_N > 1:
        final_shape = tuple([raw.shape[0], out_N] + list(out_shape))
    else:
        final_shape = tuple([raw.shape[0]] + list(out_shape))
    return pred_mu_final.reshape(final_shape)


def _extract_onnx_metadata(onnx_path):
    if not _HAS_ONNX:
        raise ImportError("onnx package required for metadata extraction")
    model = onnx.load(str(onnx_path), load_external_data=False)
    props = {p.key: p.value for p in model.metadata_props}
    output_slices_b64 = props.get("output_slices")
    if output_slices_b64 is None:
        raise ValueError(f"No 'output_slices' metadata in {onnx_path}")
    output_slices = pickle.loads(codecs.decode(output_slices_b64.encode(), "base64"))
    return {"output_slices": output_slices}


def _fov_to_focal(fov_deg, pixel_width):
    return (pixel_width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


def _camera_intrinsics(fov_deg, width, height):
    f = _fov_to_focal(fov_deg, width)
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


@register_policy("openpilot")
class OpenPilotAdapter(SensorPolicy):
    """Adapter for OpenPilot's driving model (vision + policy ONNX)."""

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.ckpt_dir = Path(checkpoint_path)
        self.vision_session: Any = None
        self.policy_session: Any = None
        self.vision_meta: Any = None
        self.policy_meta: Any = None
        self.road_frames: collections.deque[Any] = collections.deque(maxlen=3)
        self.wide_frames: collections.deque[Any] = collections.deque(maxlen=3)
        self._steps_per_context = 2
        self._raw_buffer_len = 50
        self._raw_features = np.zeros((1, self._raw_buffer_len, FEATURE_LEN), dtype=np.float16)
        self._raw_desire = np.zeros((1, self._raw_buffer_len, DESIRE_LEN), dtype=np.float16)
        self.prev_desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
        self.traffic_convention = np.array([[1, 0]], dtype=np.float16)
        self.warp_road: Any = None
        self.warp_wide: Any = None
        self._warps_initialized = False
        self._cur_ego_state: Any = None
        self._last_vision_out = None

    @staticmethod
    def _load_or_extract_metadata(onnx_path: Path) -> dict:
        meta_path = onnx_path.parent / (onnx_path.stem + "_metadata.pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                return pickle.load(f)
        metadata = _extract_onnx_metadata(onnx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        return metadata

    def load_model(self):
        if not _HAS_ONNX:
            raise ImportError(
                "OpenPilot adapter requires onnx and onnxruntime. "
                "Install with: pip install onnx onnxruntime-gpu"
            )
        print("Loading OpenPilot ONNX models...")
        vision_path = self.ckpt_dir / "driving_vision.onnx"
        policy_path = self.ckpt_dir / "driving_policy.onnx"
        for p in [vision_path, policy_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required ONNX model not found: {p}")

        vision_meta_raw = self._load_or_extract_metadata(vision_path)
        policy_meta_raw = self._load_or_extract_metadata(policy_path)
        self.vision_meta = vision_meta_raw.get("output_slices", vision_meta_raw)
        self.policy_meta = policy_meta_raw.get("output_slices", policy_meta_raw)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.vision_session = ort.InferenceSession(str(vision_path), sess_options=sess_opts, providers=providers)
        self.policy_session = ort.InferenceSession(str(policy_path), sess_options=sess_opts, providers=providers)
        print("OpenPilot models loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {
            "CAM_ROAD": {"x": 0.0, "y": 0.0, "z": 1.22, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                         "fov": 40, "width": 1928, "height": 1208},
            "CAM_WIDE": {"x": 0.0, "y": 0.0, "z": 1.22, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                         "fov": 120, "width": 1928, "height": 1208},
        }

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        command = ego_state.get('command', 3)
        desire_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 3, 5: 4}
        desire_index = desire_map.get(command, 0)
        cur_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
        if 0 < desire_index < DESIRE_LEN:
            cur_desire[desire_index] = 1.0
        cur_desire[0] = 0.0

        new_desire = np.where(cur_desire - self.prev_desire_vec > 0.99, cur_desire, 0.0)
        self.prev_desire_vec[:] = cur_desire
        self._raw_desire[0, :-1] = self._raw_desire[0, 1:]
        self._raw_desire[0, -1] = new_desire.astype(np.float16)

        road_img = images["CAM_ROAD"][:, :, ::-1].copy()
        wide_img = images["CAM_WIDE"][:, :, ::-1].copy()

        if not self._warps_initialized:
            h, w = road_img.shape[:2]
            cam_configs = self.get_camera_configs()
            calib_euler = np.array([0.0, 0.0, 0.0])
            road_intr = _camera_intrinsics(fov_deg=cam_configs["CAM_ROAD"]["fov"], width=w, height=h)
            wide_intr = _camera_intrinsics(fov_deg=cam_configs["CAM_WIDE"]["fov"], width=w, height=h)
            self.warp_road = get_warp_matrix(calib_euler, road_intr, bigmodel_frame=False)
            self.warp_wide = get_warp_matrix(calib_euler, wide_intr, bigmodel_frame=True)
            self._warps_initialized = True

        road_warped = cv2.warpPerspective(road_img, self.warp_road, MEDMODEL_INPUT_SIZE,
                                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        wide_warped = cv2.warpPerspective(wide_img, self.warp_wide, MEDMODEL_INPUT_SIZE,
                                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)

        road_yuv = rgb_to_yuv6ch(road_warped)
        wide_yuv = rgb_to_yuv6ch(wide_warped)
        self.road_frames.append(road_yuv)
        self.wide_frames.append(wide_yuv)

        if len(self.road_frames) >= 3:
            road_input = np.concatenate([self.road_frames[-3], self.road_frames[-1]], axis=0)
            wide_input = np.concatenate([self.wide_frames[-3], self.wide_frames[-1]], axis=0)
        else:
            road_input = np.concatenate([road_yuv, road_yuv], axis=0)
            wide_input = np.concatenate([wide_yuv, wide_yuv], axis=0)

        img = road_input[np.newaxis].astype(np.uint8)
        big_img = wide_input[np.newaxis].astype(np.uint8)

        vision_out = self.vision_session.run(None, {"img": img, "big_img": big_img})[0]
        hs_slice = self.vision_meta["hidden_state"]
        hidden_state = vision_out[0, hs_slice].astype(np.float16)
        self._raw_features[0, :-1] = self._raw_features[0, 1:]
        self._raw_features[0, -1] = hidden_state
        self._cur_ego_state = ego_state
        return {"ego_state": ego_state}

    def run_inference(self, model_input: Any) -> Any:
        idxs = np.arange(-1, -self._raw_buffer_len - 1, -self._steps_per_context)[::-1]
        features_for_policy = self._raw_features[:, idxs]
        desire_for_policy = self._raw_desire.reshape(1, 25, self._steps_per_context, DESIRE_LEN).max(axis=2)

        policy_out = self.policy_session.run(None, {
            "features_buffer": features_for_policy,
            "desire_pulse": desire_for_policy,
            "traffic_convention": self.traffic_convention,
        })[0].astype(np.float32)

        plan_slice = self.policy_meta["plan"]
        raw_plan = policy_out[:, plan_slice]

        plan_n_values = IDX_N * PLAN_WIDTH
        if raw_plan.shape[1] == 2 * plan_n_values:
            plan_in_N, plan_out_N = 0, 0
        else:
            plan_in_N, plan_out_N = PLAN_MHP_N, 1
        plan = parse_mdn(raw_plan, in_N=plan_in_N, out_N=plan_out_N, out_shape=(IDX_N, PLAN_WIDTH))
        return {"plan": plan}

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        plan = model_output["plan"][0]
        plan_x_fwd = plan[:, 0]
        plan_y_right = plan[:, 1]
        target_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        interp_x_fwd = np.interp(target_times, T_IDXS, plan_x_fwd)
        interp_y_right = np.interp(target_times, T_IDXS, plan_y_right)
        trajectory = np.column_stack([-interp_y_right, interp_x_fwd])
        return {"trajectory": trajectory}
