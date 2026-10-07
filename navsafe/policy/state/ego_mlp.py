"""EgoStatusMLP — the only first-party :class:`StatePolicy` adapter.

Phase 3 task 3.6 of the NavSafe Package Reorg spec ports this
adapter from its pre-reorg location at
``navsafe/evaluation/models/ego_mlp_adapter.py`` into the new
:mod:`navsafe.policy` plugin boundary. Backs:

* Requirement 3.3 — every first-party adapter inherits from exactly
  one of :class:`StatePolicy` / :class:`SensorPolicy`.
* Requirement 3.4 — the ``ego_mlp`` adapter inherits from
  :class:`StatePolicy` (it is the *only* first-party adapter that
  does, because its :meth:`prepare_input` consumes ego status —
  velocity, acceleration, driving command — rather than camera /
  LiDAR tensors).
* Requirement 3.7 — base-class choice is determined by what
  ``prepare_input`` consumes, not by historical lineage. EgoStatusMLP
  is a blind baseline that reads only ego state, hence ``StatePolicy``.
* Requirement 10.5 — :meth:`load_model`, :meth:`prepare_input`,
  :meth:`run_inference`, :meth:`parse_output` semantics are preserved
  verbatim from the pre-reorg adapter; the reorg only changes the
  base class and adds the registry decorator.

The :func:`register_policy` decorator wires the class into the
process-wide :class:`navsafe.engine.registry.Registry` under the
name ``"ego_mlp"`` so the CLI (``navsafe eval --model-type ego_mlp``)
and the public API (``evaluate(EgoStatusMLPAdapter, ...)``) resolve
the same class against the same registry (Requirement 4.6).
"""

import torch
import numpy as np
from typing import Dict, Any

from navsafe.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from navsafe.policy.registry import register_policy
from navsafe.policy.state_policy import StatePolicy


@register_policy("ego_mlp")
class EgoStatusMLPAdapter(StatePolicy):
    """
    Adapter for EgoStatusMLP model.
    Blind baseline that ignores all sensors and only uses ego vehicle state.
    """

    def __init__(self, checkpoint_path: str, hidden_dim: int = 512, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_poses = 8  # 4 seconds at 0.5s interval

    def load_model(self):
        print("Loading EgoStatusMLP model...")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.num_poses * 3),
        )

        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        clean_sd = {}
        for k, v in state_dict.items():
            new_key = k.replace('agent._mlp.', '')
            clean_sd[new_key] = v

        self.model.load_state_dict(clean_sd, strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("EgoStatusMLP model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        return {}

    def prepare_input(self, images: Dict[str, np.ndarray], ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any], frame_id: int) -> Any:
        velocity = ego_state['velocity'][:2]
        heading = ego_state['heading']
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, s], [-s, c]])
        vel_local = R @ velocity

        KICKOFF_SPEED = 2.0
        if np.linalg.norm(vel_local) < 0.5:
            vel_local[0] = KICKOFF_SPEED

        if 'acceleration' in ego_state:
            acc = ego_state['acceleration'][:2]
            acc_local = R @ acc
        else:
            acc_local = np.array([0.0, 0.0])

        command = ego_state.get('command', 1)
        cmd_onehot = NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD).copy()

        ego_status = np.concatenate([vel_local, acc_local, cmd_onehot]).astype(np.float32)
        ego_status_tensor = torch.from_numpy(ego_status).unsqueeze(0).to(self.device)
        return {"ego_status": ego_status_tensor}

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            ego_status = model_input["ego_status"]
            poses = self.model(ego_status)
            trajectory = poses.reshape(-1, self.num_poses, 3)
        return {"trajectory": trajectory}

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        trajectory = model_output["trajectory"][0].cpu().numpy()
        traj_swapped = np.column_stack([trajectory[:, 1], trajectory[:, 0]])
        return {'trajectory': traj_swapped}


__all__ = ["EgoStatusMLPAdapter"]
