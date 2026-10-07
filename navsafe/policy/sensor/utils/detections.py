"""Auxiliary detection head outputs, carried through to the plan record.

TransFuser-style policies (LTF, DiffusionDrive, DiffusionDriveV2, DrivoR, RAP)
run an agent head beside the trajectory head: ``agent_states`` are 2D boxes in
the ego BEV frame and ``agent_labels`` their per-query logits. Both are dropped
by every adapter's ``parse_output``, which keeps only the trajectory, so a
closed-loop failure cannot be attributed to what the model represented.

The distinction this enables is the point: a policy that never boxed the
obstacle failed in perception, whereas one that boxed it and drove into it
anyway failed in planning. Note the second reading is about the shared BEV
features, NOT a causal claim about the plan -- the trajectory head reads those
features directly and is not conditioned on the decoded boxes.
"""

from typing import Any, Dict

import numpy as np

# (x, y, heading, length, width) -- navsafe.modelzoo.common.enums
# .BoundingBox2DIndex, the layout every TransFuser-style agent head emits.
BOX_DIM = 5


def _to_numpy(value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    return arr if arr.size else None


def attach_detections(parsed: Dict[str, Any], model_output: Any) -> Dict[str, Any]:
    """Copy ``agent_states``/``agent_labels`` into ``parsed``, when present.

    Best-effort and in place: a model without an agent head, or a malformed
    head output, must never cost the caller its trajectory.
    """
    if not isinstance(model_output, dict):
        return parsed
    try:
        states = _to_numpy(model_output.get("agent_states"))
        if states is None or states.shape[-1] != BOX_DIM:
            return parsed
        # Drop the batch axis; a (N, 5) box set is what the recorder expects.
        parsed["agent_states"] = states[0] if states.ndim == 3 else states

        labels = _to_numpy(model_output.get("agent_labels"))
        if labels is not None:
            parsed["agent_labels"] = labels[0] if labels.ndim == 2 else labels
    except Exception:                                              # noqa: BLE001
        return parsed
    return parsed


__all__ = ["attach_detections", "BOX_DIM"]
