"""State-based policy adapters.

Policies in this subpackage consume structured ego/agent/map state
vectors (not raw sensor data). Their ``prepare_input`` reads ego
status, velocity, acceleration, driving commands, etc.

Currently contains:
- ``ego_mlp`` — EgoStatusMLP baseline (the only neural StatePolicy)
- ``idm_centerline`` — Single-trajectory IDM + nearest-lane centerline
  (legacy rule-based baseline; previously mis-labelled as ``pdm_closed``)
- ``pdm_closed`` — Faithful PDM-Closed (proposal/score/select rule-based
  planner per Dauner et al., CoRL 2023)
- ``scripted_path`` — replays a pre-authored world-frame path (figure
  generation, not a benchmark entry)
"""

from navsafe.policy.state.ego_mlp import EgoStatusMLPAdapter
from navsafe.policy.state.idm_centerline import IDMCenterlineAdapter
from navsafe.policy.state.pdm_closed import PDMClosedAdapter
from navsafe.policy.state.scripted_path import ScriptedPathAdapter

__all__ = [
    "EgoStatusMLPAdapter",
    "ScriptedPathAdapter",
    "IDMCenterlineAdapter",
    "PDMClosedAdapter",
]
