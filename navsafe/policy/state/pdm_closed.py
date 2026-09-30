# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""PDM-Closed adapter — thin StatePolicy shim around :class:`PDMPlanner`.

The actual algorithm lives in
:mod:`navsafe.policy.state.pdm_closed_planner` (see Stage 0–4 of the
faithful PDM-Closed refactor). This file is *just* the adapter:

* Wire :class:`PDMPlanner` into the NexusSim plugin registry under
  ``"pdm_closed"`` so ``--model-type pdm_closed`` resolves to the
  faithful planner.
* Translate the
  ``(images, ego_state, scenario_data, frame_id)`` evaluator contract
  into a single ``planner.plan(...)`` call.
* Pass the planner's already-formatted ``(8, 2)`` ego-frame
  ``[lateral, forward]`` trajectory through ``parse_output``
  unchanged — no further frame transform is needed because the
  forward simulation produced output in the planner's standard
  contract.

For a single-trajectory IDM-and-centerline baseline (the historical
behaviour previously mis-labelled as ``pdm_closed``), use
``--model-type idm_centerline``
(:class:`navsafe.policy.state.idm_centerline.IDMCenterlineAdapter`).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from navsafe.policy.registry import register_policy
from navsafe.policy.state.pdm_closed_planner import (
    PDMPlanner,
    pdm_config_from_env,
    pdm_env_overrides,
    result_execution_trajectory,
)
from navsafe.policy.state_policy import StatePolicy
from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS


def _timed_position_speeds(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Fallback speed profile from consecutive future poses."""
    from navsafe.core.plan_execution import controller_waypoint_speeds

    return controller_waypoint_speeds(trajectory, dt)


def _candidate_arrays(model_output: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Expose the planner's proposal set under the evaluator's viz contract.

    ``EvalArtifacts`` renders ``frames/*/topdown_candidates.png`` (and the
    stitched ``topdown_candidates.gif``) for any policy whose ``parse_output``
    carries a ``trajectory_coarse`` key — see
    :func:`navsafe.evaluation.vis_utils.render_bev_candidates`. It expects
    ``(K, N, 2)`` ego-frame ``[lateral, forward]`` candidates plus an optional
    ``(K,)`` score vector, which is exactly what a proposal's
    ``output_xy_ego`` and :attr:`PDMPlanResult.scores` already are.

    Returns an empty dict — i.e. no candidate overlay for this frame — when
    the planner produced no proposals (emergency brake) or when the proposal
    shapes are not stackable. Never raises: this is visualization-only and
    must not take down an evaluation run.
    """
    proposals = model_output.get("proposals") or []
    if not proposals:
        return {}

    per_proposal = [np.asarray(p.output_xy_ego, dtype=np.float64) for p in proposals]
    shapes = {arr.shape for arr in per_proposal}
    if len(shapes) != 1:
        # Ragged proposal horizons would break the stack; skip the overlay
        # rather than fail the frame.
        return {}

    out: Dict[str, np.ndarray] = {"trajectory_coarse": np.stack(per_proposal)}

    scores = model_output.get("scores")
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        # The scorer writes one score per proposal_idx; a mismatch means the
        # selection ran on a different proposal set (e.g. brake fallback).
        if scores.shape == (len(per_proposal),):
            out["coarse_scores"] = scores
    return out


@register_policy("pdm_closed", overwrite=True)
class PDMClosedAdapter(StatePolicy):
    """Faithful PDM-Closed planner exposed as a NexusSim StatePolicy.

    Inputs:
        ``prepare_input(images, ego_state, scenario_data, frame_id)`` —
        consumes only ``ego_state`` + ``scenario_data`` + ``frame_id``;
        ``images`` is ignored (PDM-Closed is a state-mode planner).

    Outputs:
        ``parse_output`` returns ``{"trajectory": (N, 2)}`` in ego frame
        with column order ``[lateral, forward]``. ``N`` is the
        adapter's :meth:`get_waypoint_dt`-consistent count of poses
        (8 by default, matching every other state policy in NexusSim).

        When the planner produced proposals, ``parse_output`` also emits
        ``trajectory_coarse`` ``(K, N, 2)`` and ``coarse_scores`` ``(K,)``
        so the evaluator's candidate overlay renders every proposal the
        planner scored, coloured blue (low) → red (high). Purely
        informational: the evaluator drives the ego from ``trajectory``.

    The adapter requires no checkpoint; pass ``--checkpoint none`` on
    the CLI. ``load_model`` is a no-op (idempotent across multiple
    calls).
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self._cfg = pdm_config_from_env()
        #: Config actually in force, for run metadata. A brake-mode claim
        #: in a paper table has to be checkable against what ran.
        self.config_provenance = {
            "route_source": self._cfg.route_source,
            "emergency_brake_mode": self._cfg.emergency_brake_mode,
            # Both arms of a comparison must agree on this (docs/navsafe_eval.md).
            "traffic_light_obstacles": bool(self._cfg.traffic_light_obstacles),
            "overrides_from_env": pdm_env_overrides(),
        }
        self._planner = PDMPlanner(self._cfg)
        self._loaded = False

    # ── BasePolicyAdapter overrides ───────────────────────────────────────────

    def load_model(self) -> None:
        """No-op: PDM-Closed is rule-based.

        Idempotent so the evaluator can call it more than once
        without a noisy second print.
        """
        if self._loaded:
            return
        self._loaded = True
        print(
            "PDM-Closed: faithful proposal/score/select planner loaded "
            f"({self._cfg.num_proposals} proposals = "
            f"{len(self._cfg.lateral_offsets)} lateral × "
            f"{len(self._cfg.idm_policies)} IDM policies; "
            f"horizon={self._cfg.horizon_s}s, output_dt={self._cfg.output_dt}s)."
        )

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        # Privileged planner; cameras are visualization-only and never
        # touch prepare_input / run_inference (which ignore `images`).
        # Requesting CAM_F0 lets the evaluator render a NuRec
        # front view instead of the synthetic agent-bbox fallback.
        return {"CAM_F0": NAVSIM_CAM_CONFIGS["CAM_F0"]}

    def get_waypoint_dt(self) -> float:
        return float(self._cfg.output_dt)

    def get_trajectory_time_horizon(self) -> float:
        return float(self._cfg.trajectory_horizon_s)

    def preserves_trajectory_timing(self) -> bool:
        """PDM output samples are time-parameterized, not geometry-only."""
        return True

    def supports_warmup_inference(self) -> bool:
        """Avoid advancing the stateful planner before it controls the ego."""
        return False

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        # The planner takes the same three fields directly. We pass
        # them through as a dict so ``run_inference`` can be a single
        # ``plan`` call and ``parse_output`` only needs the trajectory.
        return {
            "ego_state": ego_state,
            "scenario_data": scenario_data,
            "frame_id": int(frame_id),
        }

    def run_inference(self, model_input: Any) -> Any:
        result = self._planner.plan(
            scenario_data=model_input["scenario_data"],
            ego_state=model_input["ego_state"],
            frame_id=model_input["frame_id"],
        )
        out: Dict[str, Any] = {
            "trajectory": result.trajectory,
            "trajectory_speeds_mps": result.trajectory_speeds_mps,
            "best_idx": result.best_idx,
            "scores": result.scores,
            "proposals": result.proposals,
            "emergency_brake_triggered": result.emergency_brake_triggered,
            "route_source": result.route_source,
        }
        # Expose the proposal set under the trajectory-scorer contract
        # (``all_candidates``: (B, K, T, >=2) model-frame [forward, lateral])
        # so ``--trajectory-scorer epdms_fast`` can re-score the planner's
        # proposals with the full EPDMS terms (privileged: the scorer reads
        # the symbolic scenario, so injected obstacles are part of the score)
        # and pick the argmax. Proposals are stored [lateral, forward];
        # swap the columns. Skipped for ragged proposal horizons.
        #
        # The planner's OWN selection (``result.trajectory`` — which is the
        # emergency-brake trajectory when the brake triggered) goes in as
        # candidate 0: when every raw proposal gates to score 0 (e.g. an
        # unavoidable obstacle blocks the whole fan), ``argmax`` returns the
        # first index, so ties resolve to the planner's safe choice instead
        # of an arbitrary colliding proposal.
        props = result.proposals or []
        if props:
            per = [np.asarray(p.output_xy_ego, dtype=np.float32) for p in props]
            own = result_execution_trajectory(result, self._cfg).astype(
                np.float32, copy=False)
            if own.ndim == 2 and own.shape[1] >= 2:
                per = [own[:, :2]] + per
            if len({a.shape for a in per}) == 1:
                import torch
                cand = np.stack(per)[..., [1, 0]]  # (K, T, 2) → [forward, lateral]
                out["all_candidates"] = torch.from_numpy(cand).unsqueeze(0)
        return out

    def parse_output(
        self, model_output: Any, ego_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Planner output is already in the contracted ``(N, 2)``
        # ego-frame ``[lateral, forward]`` shape — no further transform
        # required. Copy so the evaluator can safely retain the array.
        traj = model_output["trajectory"]
        if hasattr(traj, "cpu"):
            traj = traj.cpu().numpy()
        traj = np.asarray(traj, dtype=np.float64)
        if "proposals" not in model_output:
            # A trajectory scorer replaced the planner dict with its
            # ``select_best`` output: (1, T, >=2) model-frame
            # [forward, lateral] — squeeze and swap back to the adapter's
            # (T, 2) [lateral, forward] contract.
            if traj.ndim == 3:
                traj = traj[0]
            traj = traj[:, [1, 0]]
        emergency_brake = bool(
            model_output.get("emergency_brake_triggered", False))
        if (emergency_brake
                and self._cfg.emergency_brake_mode == "trajectory"):
            # Upstream's emergency poses are a longitudinal-controller
            # correction with current heading and zero dynamic state, not an
            # XY path. Project that full-state meaning onto NexusSim's XY-only
            # adapter contract before any generic path consumer can infer
            # reverse motion / a 180-degree heading from the raw positions.
            from navsafe.policy.state.pdm_closed_planner.planner import (
                emergency_brake_execution_trajectory,
            )
            traj = emergency_brake_execution_trajectory(traj)
        parsed: Dict[str, Any] = {
            "trajectory": traj.copy(),
            "emergency_brake_triggered": emergency_brake,
        }
        speeds = model_output.get("trajectory_speeds_mps")
        if speeds is not None:
            speeds = np.asarray(speeds, dtype=np.float64).reshape(-1)
            if len(speeds) == len(traj):
                parsed["trajectory_speeds_mps"] = speeds.copy()
        parsed.update(_candidate_arrays(model_output))
        return parsed


@register_policy("pdm_closed_fast", overwrite=True)
class PDMClosedFastAdapter(PDMClosedAdapter):
    """PDM-Closed whose proposal scorer builds ``FastLaneProxy`` lanes.

    Prefer ``pdm_closed`` for the reference evaluation path.

    Identical planner, proposals, and scoring formulas — the only change is
    that the wrapped proposal scorer builds
    :class:`~navsafe.evaluation.utils.lane_proxy_fast.FastLaneProxy` lanes
    instead of the reference
    :class:`~navsafe.evaluation.utils.lane_proxy.LaneProxy`. That was worth
    ~40x on the projection hot loop when the reference
    ``local_coordinates`` was still a per-segment Python loop. The
    2026-08 perf pass vectorised the reference itself, which erased the gap
    and then reversed it: on the 74-frame banked loop-2 corpus this class
    scores at ~186 ms/frame against the reference path's ~164 ms/frame
    (3 paired runs, same process).

    It is also the *less* exact of the two now. The reference reproduces the
    historical scalar loop bit-for-bit; ``FastLaneProxy`` uses ``np.hypot``,
    which differs from that loop at 1 ulp on ~17% of pointwise inputs. Where
    two segments tie within that ulp the ``argmin`` flips and the reported
    longitudinal position moves by metres, and on a **non-finite** query the
    two disagree outright — the reference reports infinitely off-lane, this
    class reports ``r = 0.0``, exactly on the centre-line. See
    ``navsafe/evaluation/utils/lane_proxy_fast.py`` for the full statement
    and ``tests/evaluation/test_lane_proxy_vectorized_parity.py`` for the
    tests that pin it.

    On the banked loop-2 corpus all 1110 candidates and 11100 metric terms
    replay bit-identically either way (reproduce with
    ``toolbox/epdms_consolidation_parity.py`` plus the loop-2 dumps). That
    corpus contains no tie and no non-finite query, though, so it does not
    generalise: re-score rather than comparing numbers across a switch.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        super().__init__(checkpoint_path, **kwargs)
        # PDMPlanner → PDMScorer wrapper → EPDMSTrajectoryScorer_Fast
        self._planner._scorer.scorer.use_fast_lanes = True


__all__ = ["PDMClosedAdapter", "PDMClosedFastAdapter"]
