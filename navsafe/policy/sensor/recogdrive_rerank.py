# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""ReCogDrive + the VLA-Jspace first-step reranker.

ReCogDrive already draws its plan from a DDIM sampler, so N candidates cost one
VLM pass and one batched sampler run. This adapter draws N, reads each
candidate's state at the **first denoising call**, scores them with a fitted
linear ridge, and drives the winner. The planner is unchanged: no retraining,
no extra forward pass, no second VLM.

The scorer is `VLA-Jspace/scripts/rerank_score.py` and the weights are
`reranker_rd_il_token.npz` (4120-d, lambda 10, sha256 fe19abf3…8dfa9032).
Measured open loop on navtest (12146 scenes, N=16): PDMS 86.07 -> 87.21,
+1.139 [+0.965, +1.316], which is 28.3% of the 4.019-point oracle headroom.
`VLA-Jspace/NAVSAFE.md` is the handoff document; §5 there is the part that
matters for reading a closed-loop result, because **82% of that open-loop gain
comes from the 11.5% of scenes where at least one candidate scores zero**.
Outside that stratum the effect is +0.23 points, which a closed-loop run of
this size cannot resolve.

Three properties of the scorer are load-bearing and are asserted rather than
described (NAVSAFE.md §3.4):

1. **The score is not a per-candidate function.** Features are centred over the
   scene's N candidates, so all N must exist before any can be scored. A
   streaming "score each as it arrives" integration is wrong, not approximate.
2. **float16 is part of the payload.** The pool was stored and fitted in
   float16, including the `dit[:, :-1] - base` subtraction. A float32 path runs,
   looks sane, and quietly moves the picks.
3. **The feature is read at the first call only.** Nothing downstream of it is
   used, which is what makes the readout free.

`--checkpoint` is the ReCogDrive planner (IL); the reranker weights come from
`NAVSAFE_RERANK_WEIGHTS`. With no weights set the adapter refuses to load
rather than silently degrading to "take candidate 0", which would score as a
policy that was never evaluated.
"""

from __future__ import annotations

import os
from navsafe.data_paths import model_path
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.sensor.vla_client import (
    EgoPoseHistory,
    VLASubprocessClient,
    local_velocity_acceleration,
    navsim_command_one_hot,
    vla_python,
    vla_server_dir,
)
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS
from navsafe.policy.sensor.utils.frames import (
    crop_to_navsim_aspect,
    renderer_bgr_to_rgb,
)

DEFAULT_VLM_PATH = model_path("recogdrive/vlm2b")

#: Candidate count the weight was fitted at. NAVSAFE.md §3.3: the centring
#: adapts to any N, but the calibration was measured at 16 and nothing
#: validates that changing it leaves the scorer's behaviour alone.
DEFAULT_N_CAND = 16


def _torch_seed() -> int:
    """The eval process's current torch seed, to hand to the server.

    the evaluator applies `--eval-seed` before building the adapter, so
    reading it here captures that value without the adapter needing to know
    the flag exists.
    """
    try:
        import torch
        return int(torch.initial_seed() % (2 ** 31))
    except Exception:                                          # pragma: no cover
        return 0


def _load_scorer(weights: str):
    """`Reranker` from the standalone numpy scorer shipped with the weights.

    Imported from the weights' own directory rather than vendored: NAVSAFE.md
    §7 makes `rerank_score.py` a single self-contained file precisely so the
    deploying repo runs the fitting pipeline's code, not a transcription of it.
    A copy here could drift from the fit and the drift would be invisible --
    the picks would just quietly change.
    """
    import importlib.util

    scorer_py = Path(weights).with_name("rerank_score.py")
    if not scorer_py.is_file():
        raise FileNotFoundError(
            f"rerank_score.py must sit beside the weights ({scorer_py}); it is "
            f"the fitting pipeline's own feature builder and must not be "
            f"reimplemented here")
    spec = importlib.util.spec_from_file_location("navsafe_rerank_score", scorer_py)
    if spec is None or spec.loader is None:                     # pragma: no cover
        raise ImportError(f"cannot load {scorer_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@register_policy("recogdrive_rerank")
class ReCogDriveRerankAdapter(SensorPolicy):
    """ReCogDrive-IL with first-step candidate reranking."""

    def __init__(self, checkpoint_path: str, config_path: str | None = None,
                 vlm_path: str | None = None, dit_type: str = "small",
                 weights: str | None = None, n_cand: int | None = None,
                 **kwargs):
        super().__init__(checkpoint_path, config_path=config_path, **kwargs)
        self.vlm_path = (vlm_path
                         or os.environ.get("NAVSAFE_RECOGDRIVE_VLM", DEFAULT_VLM_PATH))
        self.dit_type = dit_type
        self.weights = weights or os.environ.get("NAVSAFE_RERANK_WEIGHTS", "")
        self.n_cand = int(n_cand or os.environ.get("NAVSAFE_RERANK_N", DEFAULT_N_CAND))
        self.client: VLASubprocessClient | None = None
        self.reranker: Any = None
        self._scorer_mod: Any = None
        self._history = EgoPoseHistory()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="rd_rerank_")
        #: Per-episode telemetry. The pick distribution is the first thing to
        #: look at when a closed-loop result comes out flat: an argmin that
        #: never moves off one slot means the feature is not discriminating on
        #: these scenes, which is a different failure from "it ranked fine and
        #: the closed loop washed it out" (NAVSAFE.md §5.4).
        self.pick_counts: Dict[int, int] = {}
        self._n_calls = 0

    # -- load ---------------------------------------------------------------

    def load_model(self):
        if not self.weights:
            raise ValueError(
                "recogdrive_rerank needs the reranker weights: set "
                "NAVSAFE_RERANK_WEIGHTS=/path/to/reranker_rd_il_token.npz. "
                "Refusing to run unranked -- that would score plain "
                "ReCogDrive under this model's name.")
        self._scorer_mod = _load_scorer(self.weights)
        self.reranker = self._scorer_mod.Reranker(self.weights)
        if self.reranker.model != "recogdrive":
            raise ValueError(
                f"{self.weights} was fitted for {self.reranker.model!r}, not "
                f"recogdrive; the feature layouts differ (4120-d vs 8224-d)")
        if self.reranker.dim != 4120:
            raise ValueError(
                f"expected the 4120-d token tier, got {self.reranker.dim}-d")
        if self.reranker.n_cand != self.n_cand:
            # Not fatal -- the centring adapts -- but it is a departure from
            # what was measured, so it is said out loud rather than absorbed.
            print(f"[recogdrive_rerank] WARNING: weight fitted at N="
                  f"{self.reranker.n_cand}, running at N={self.n_cand}; the "
                  f"calibration was not validated at this N", flush=True)

        script = str(vla_server_dir() / "recogdrive_rerank_server.py")
        print(f"Starting ReCogDrive rerank server (planner={self.checkpoint_path}, "
              f"N={self.n_cand})...", flush=True)
        self.client = VLASubprocessClient(
            vla_python(), script,
            ["--vlm-path", self.vlm_path,
             "--planner-ckpt", self.checkpoint_path,
             "--dit-type", self.dit_type,
             "--n-cand", str(self.n_cand),
             # subprocess.Popen gives the child a FRESH interpreter, so the
             # eval process's torch.manual_seed(--eval-seed) does not reach
             # it (vla_client.py builds the argv itself and propagates no RNG
             # state). Without this the candidate draw is unseeded and a cell
             # is not reproducible, which the fit's own collector avoided by
             # seeding immediately before get_action (rd_collect.py:305).
             "--seed", str(_torch_seed())])
        self.model = self.client
        print(f"ReCogDrive rerank server ready "
              f"(reranker {self.reranker.dim}-d, lam={self.reranker.meta['lam']}).",
              flush=True)

    # -- inference ----------------------------------------------------------

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                      scenario_data: Dict[str, Any], frame_id: int) -> Any:
        self._history.update(ego_state, frame_id)
        img = images.get("CAM_F0")
        if img is None and images:
            img = next(iter(images.values()))
        if img is None:
            img = np.zeros((1120, 1920, 3), dtype=np.uint8)
        img_path = str(Path(self._tmpdir.name) / "cam_f0.npy")
        np.save(img_path, renderer_bgr_to_rgb(crop_to_navsim_aspect(img)))

        vel, acc = local_velocity_acceleration(ego_state)
        status = navsim_command_one_hot(ego_state) + [float(vel[0]), float(vel[1]),
                                                      float(acc[0]), float(acc[1])]
        return {"image_npy": img_path,
                "history": self._history.local_history(),
                "status": status}

    def run_inference(self, model_input: Any) -> Any:
        """N candidates plus the first-call tap, then pick one.

        The scoring happens HERE rather than in the server so the deployed
        selector is the published numpy scorer running on the published
        weights -- the server's only job is to produce the candidates and the
        tap, which keeps the model process free of the selection logic.
        """
        if self.client is None or self.reranker is None:
            raise RuntimeError("ReCogDriveRerankAdapter: call load_model() first")
        out = self.client.request(model_input)

        traj = np.asarray(out["trajectory"], dtype=np.float32)       # [N, 8, 3]
        x_T = np.asarray(out["x_T"], dtype=np.float16)               # [N, 8, 3]
        dit = np.asarray(out["dit"], dtype=np.float16)               # [N, 8, 512]
        if traj.ndim != 3 or traj.shape[0] < 2:
            raise RuntimeError(
                f"expected >=2 candidates, got trajectory shape {traj.shape}; "
                f"the reranker cannot select from one candidate")

        # The published feature builder, on the published dtype path.
        feats = self._scorer_mod.features_recogdrive(x_T, dit)       # [N, 4120]
        scores = self.reranker.score(feats)
        j = int(scores.argmin())
        # The evaluator already records what a policy "considered, picked and
        # drove" (evaluator.py:_save_plan_records) from `best_idx`,
        # `trajectory_coarse` and `coarse_scores`. An adapter that does not
        # populate them leaves selected_index/candidate_scores null in
        # plan_records.json -- which is exactly what made "is the reranker
        # actually selecting?" unanswerable from the artifacts. Fill them.
        self._last_scores = scores
        self._last_candidates = traj
        self.pick_counts[j] = self.pick_counts.get(j, 0) + 1
        self._n_calls += 1
        # A selector whose argmin never moves is INERT, and it scores exactly
        # like a working one -- the episode completes and the number looks
        # plausible. The fit's own picks spread over all 16 slots (max share
        # 12.1% on 12146 navtest scenes), so a collapsed distribution here is
        # the signature to catch. Logged periodically rather than per frame:
        # per frame would drown the cell log, never would leave the question
        # unanswerable after the fact.
        if self._n_calls % 20 == 0:
            spread = float(scores.max() - scores.min())
            top = sorted(self.pick_counts.items(), key=lambda kv: -kv[1])[:4]
            print(f"[rerank] {self._n_calls} picks, {len(self.pick_counts)} distinct slots, "
                  f"top={top}, last score spread={spread:.4f}", flush=True)
        return {"trajectory": traj[j], "best_idx": j,
                "candidate_scores": scores, "candidates": traj}

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        traj = np.asarray(model_output["trajectory"], dtype=np.float32)  # (8,3) x fwd, y left
        out: Dict[str, np.ndarray] = {
            "trajectory": np.column_stack([traj[:, 1], traj[:, 0]]),
            "heading": traj[:, 2],
        }
        # Same [lateral, forward] convention as the driven plan, so the
        # recorded candidate set is directly comparable with `selected_ego`.
        cands = model_output.get("candidates")
        if cands is not None:
            c = np.asarray(cands, dtype=np.float32)
            out["trajectory_coarse"] = np.stack(
                [np.column_stack([c[i, :, 1], c[i, :, 0]]) for i in range(len(c))])
        sc = model_output.get("candidate_scores")
        if sc is not None:
            out["coarse_scores"] = np.asarray(sc, dtype=np.float32)
        return out

    def get_waypoint_dt(self) -> float:
        return 0.5

    def get_trajectory_time_horizon(self) -> float:
        return 4.0


__all__ = ["ReCogDriveRerankAdapter"]
