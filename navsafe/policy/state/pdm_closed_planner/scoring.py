# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""PDM-Closed proposal scoring.

Two aggregation variants are supported:

* ``"upstream"`` (default, faithful) — matches the PDM scorer shipped by
  ``autonomousvision/CaRL`` (commit
  ``2677d1477193ecb5eb41ad8d7e4eda78057fd8a9``), which retains the
  published ``tuplan_garage`` aggregation::

      multi_prod    = NC * DAC * DDC                            # 3 metrics
      weighted_sum  = 5*Progress + 5*TTC + 2*Comfortable        # 3 metrics
      score         = multi_prod * weighted_sum / 12

  No TLC, no separate Lane-Keeping. ``Comfortable`` is computed from the
  PDM simulator's 11-channel dynamic state with CaRL's exact Savitzky-Golay
  windows and strict thresholds. It is intentionally separate from the
  generic EPDMS verifier's XY-derived HC.

* ``"navsim"`` (opt-in) — NavSim-extended PDMS::

      multi_prod    = NC * DAC * DDC * TLC                      # 4 metrics
      weighted_sum  = 5*Progress + 5*TTC + 2*LK + 2*HC          # 4 metrics
      score         = multi_prod * weighted_sum / 14

  This is NOT the metric NavSafe reports: the reported ``epdms`` keeps
  Extended Comfort (denominator 16) and anchors progress on the
  ground-truth log, while this variant is the /14 EC-free shape with an
  absolute ``min(dist/30, 1)`` progress term. Use this variant for the
  planner's internal scoring when you want the proposal selection to
  align with NavSim leaderboard semantics (e.g. proposals that violate
  red lights are immediately disqualified).

The default is ``"upstream"`` because the goal of this module is a
faithful reproduction of the CoRL 2023 PDM-Closed planner. To run the
NavSim-extended scoring, set ``cfg.scorer_variant = "navsim"`` on
:class:`PDMConfig` (or pass ``scorer_variant="navsim"`` to
:class:`PDMScorer` directly).

Both variants reuse :class:`EPDMSTrajectoryScorer_Fast`'s metric
machinery (NC, DAC, DDC, TLC, EP, TTC, LK, HC) — only the aggregation
differs. This guarantees that when a metric calculation is improved
in the EPDMS scorer, both variants benefit automatically.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import savgol_filter

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSTrajectoryScorer_Fast,
)
from navsafe.evaluation.utils.lane_proxy import LaneProxy
from navsafe.policy.state.pdm_closed_planner.config import PDMConfig
from navsafe.policy.state.pdm_closed_planner.proposals import Proposal


ScorerVariant = Literal["upstream", "navsim"]


class _CenterlineProjector:
    """Precomputed arc-length projection onto a route centerline.

    Upstream ``_calculate_progress`` semantics: project endpoints onto
    the route centerline and take the (non-negative) arc-length delta,
    so off-route motion — e.g. continuing straight where the route
    turns — earns ~zero progress instead of full chord displacement.
    Precomputes segments/cumulative lengths once per replan; the
    per-proposal cost is a single endpoint projection (all proposals
    share the same prepended start point).
    """

    def __init__(self, centerline: np.ndarray) -> None:
        self._pts = np.asarray(centerline, dtype=np.float64)
        self._seg = np.diff(self._pts, axis=0)
        self._seg_lens = np.linalg.norm(self._seg, axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._seg_lens)])
        self._denom = np.maximum(self._seg_lens**2, 1e-24)

    @property
    def degenerate(self) -> bool:
        return len(self._seg_lens) == 0

    def arc_length_of(self, p: np.ndarray) -> float:
        rel = np.asarray(p) - self._pts[:-1]
        t = np.clip(np.einsum("ij,ij->i", rel, self._seg) / self._denom, 0.0, 1.0)
        proj = self._pts[:-1] + t[:, None] * self._seg
        d = np.linalg.norm(proj - np.asarray(p), axis=1)
        i = int(np.argmin(d))
        return float(self._cum[i] + t[i] * self._seg_lens[i])


# ----------------------------------------------------------------------
# Upstream constants — pinned to ``carl_nuplan/.../pdm_scorer.py``
# (and identically to ``tuplan_garage/.../pdm_scorer.py``).
# ----------------------------------------------------------------------

#: Minimum raw progress (in metres) for the upstream scorer's
#: relative-progress normalisation to engage. When *every* proposal's
#: masked progress is below this threshold, all proposals receive
#: progress = 1.0 (then 0.0 for those with multi_prod == 0).
#:
#: Reference: ``pdm_scorer.py::PROGRESS_DISTANCE_THRESHOLD = 0.1`` (m).
PROGRESS_DISTANCE_THRESHOLD_M: float = 0.1

_UPSTREAM_AGENT_TYPES = frozenset({"VEHICLE", "PEDESTRIAN", "CYCLIST"})


def _upstream_nc_from_at_fault_ids(
    at_fault_ids: Sequence[str], actor_types: Mapping[str, str],
) -> float:
    """Upstream's no-at-fault-collision score over EVERY at-fault contact.

    ``pdm_scorer.py:331-337``::

        no_at_fault_collision_score = 0.0 if type in AGENT_TYPES else 0.5
        no_collision_scores[i] = np.minimum(no_collision_scores[i], score)

    Two properties that a single-``collision_actor_id`` remap cannot express:
    the score is the **minimum** over all at-fault contacts, so a 0.5 static
    hit never masks a later 0.0 agent hit; and an *unknown* type is not
    evidence of a static object. Upstream reads a real
    ``TrackedObjectType``, so a missing entry here means the port's own type
    map failed, and the safe reading for a published baseline is the agent
    score. Failing open to 0.5 would quietly halve the penalty.
    """
    if not at_fault_ids:
        return 1.0

    def one(actor_id: str) -> float:
        actor_type = actor_types.get(str(actor_id))
        if actor_type is None or not str(actor_type):
            return 0.0            # unknown type: fail closed
        if str(actor_type).upper() in _UPSTREAM_AGENT_TYPES:
            return 0.0            # vehicle / pedestrian / cyclist
        return 0.5                # a known non-agent object

    return min(one(actor_id) for actor_id in at_fault_ids)


def _mask_upstream_progress(
    raw_progress: np.ndarray, multiplicative_scores: np.ndarray
) -> np.ndarray:
    """CaRL's pre-normalization progress scaling, including fractions."""
    return np.asarray(raw_progress) * np.asarray(multiplicative_scores)


def _upstream_pdm_comfortable(
    states: np.ndarray, dt: float,
) -> np.ndarray:
    """Exact CoRL PDM comfort aggregate over simulator state arrays.

    This is the PDM-only equivalent of tuplan_garage's
    ``ego_is_comfortable``.  It deliberately does not change the generic
    XY-only EPDMS verifier, whose candidates have no actuator state.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 11:
        raise ValueError(
            "PDM state array must have shape (N, T, 11), "
            f"got {values.shape}")
    n_time = values.shape[1]
    if n_time < 4:
        raise ValueError(
            "PDM comfort needs at least four state samples, "
            f"got {n_time}")
    times = np.arange(n_time, dtype=np.float64) * float(dt)

    def filtered_acceleration(
        coordinate: str, *, window_length: int = 8,
    ) -> np.ndarray:
        if coordinate == "x":
            acceleration = values[..., 5]
        elif coordinate == "y":
            acceleration = values[..., 6]
        elif coordinate == "magnitude":
            acceleration = np.hypot(values[..., 5], values[..., 6])
        else:  # pragma: no cover - private closed vocabulary
            raise ValueError(coordinate)
        return np.round(savgol_filter(
            acceleration,
            polyorder=2,
            window_length=min(int(window_length), n_time),
            axis=-1,
        ), decimals=8)

    def derivative(
        array: np.ndarray, *, deriv_order: int = 1,
        poly_order: int = 2, window_length: int = 5,
    ) -> np.ndarray:
        window = min(int(window_length), n_time)
        if poly_order >= window:
            raise ValueError(
                f"comfort polynomial order {poly_order} requires more "
                f"than {window} samples")
        delta = float(np.diff(times).mean())
        return savgol_filter(
            array, polyorder=poly_order, window_length=window,
            deriv=deriv_order, delta=delta, axis=-1)

    def strictly_bounded(
        array: np.ndarray, lower: float, upper: float,
    ) -> np.ndarray:
        return np.all((array > lower) & (array < upper), axis=-1)

    # Match tuplan_garage/pdm_comfort_metrics.py, including its distinct
    # smoothing windows and strict (not inclusive) thresholds.
    lon_accel = filtered_acceleration("x", window_length=n_time)
    lat_accel = filtered_acceleration("y", window_length=n_time)
    jerk_mag = np.round(derivative(
        filtered_acceleration("magnitude"), window_length=n_time,
    ), decimals=8)
    lon_jerk = np.round(derivative(
        filtered_acceleration("x"), window_length=n_time,
    ), decimals=8)
    # Match CaRL's private ``_phase_unwrap`` literally.  In particular, its
    # yaw helper accepts ``window_length`` but does not forward it to
    # ``_approximate_derivatives``; both yaw metrics therefore use the
    # derivative helper's five-sample default.  Replacing that quirk with the
    # caller's full-horizon window changes proposal selection on real banks.
    heading = values[..., 2]
    adjustments = np.zeros_like(heading)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(heading, axis=-1) / (2.0 * np.pi)), axis=-1)
    heading = heading - (2.0 * np.pi) * adjustments
    yaw_accel = np.round(derivative(
        heading, deriv_order=2, poly_order=3, window_length=5,
    ), decimals=8)
    yaw_rate = np.round(derivative(
        heading, deriv_order=1, poly_order=2, window_length=5,
    ), decimals=8)

    terms = np.column_stack([
        strictly_bounded(lon_accel, -4.05, 2.40),
        strictly_bounded(lat_accel, -4.89, 4.89),
        strictly_bounded(jerk_mag, -8.37, 8.37),
        strictly_bounded(lon_jerk, -4.13, 4.13),
        strictly_bounded(yaw_accel, -1.93, 1.93),
        strictly_bounded(yaw_rate, -0.95, 0.95),
    ])
    return np.all(terms, axis=1).astype(np.float64)


def _apply_pdm_full_states(
    model_output: Dict[str, Any],
    derived_states: Dict[str, np.ndarray],
    world_to_sim_offset: np.ndarray,
    *,
    num_proposals: int,
    horizon: int,
    planner_dt: float,
) -> Tuple[Dict[str, np.ndarray], bool, Optional[np.ndarray]]:
    """Replace reconstructed XY dynamics with CaRL's coherent PDM states.

    The generic EPDMS interface accepts XY samples and reconstructs heading
    and speed after SavGol filtering.  CaRL's PDM scorer instead receives the
    tracker-simulated state array directly.  PDM callers provide that array
    through ``pdm_full_states``; generic verifier callers omit it and keep the
    historical XY-only behavior.
    """
    full = model_output.get("pdm_full_states")
    if full is None:
        return derived_states, False, None
    if not isinstance(full, dict):
        raise TypeError("pdm_full_states must be a dict of state arrays")
    required = ("x", "y", "heading", "speed", "state")
    expected = (num_proposals, horizon + 1)
    arrays: Dict[str, np.ndarray] = {}
    for key in required:
        if key not in full:
            raise ValueError(f"pdm_full_states missing required key {key!r}")
        value = np.asarray(full[key], dtype=np.float64)
        key_expected = (
            (num_proposals, horizon + 1, 11)
            if key == "state" else expected
        )
        if value.shape != key_expected:
            raise ValueError(
                f"pdm_full_states[{key!r}] has shape {value.shape}; "
                f"expected {key_expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"pdm_full_states[{key!r}] contains non-finite values")
        arrays[key] = value

    coherent = dict(derived_states)
    coherent["x"] = arrays["x"] + float(world_to_sim_offset[0])
    coherent["y"] = arrays["y"] + float(world_to_sim_offset[1])
    coherent["heading"] = np.unwrap(arrays["heading"], axis=1)
    # ProposalState carries signed longitudinal speed.  CaRL reads the norm
    # of its velocity vector, so reverse values remain moving rather than
    # becoming a stopped-ego exemption.
    coherent["speed"] = np.abs(arrays["speed"])
    comfort = _upstream_pdm_comfortable(arrays["state"], planner_dt)
    return coherent, True, comfort


def _snapshot_from_log(
    scorer: EPDMSTrajectoryScorer_Fast, frame_idx: int,
) -> List[Dict[str, Any]]:
    """A live-snapshot-shaped actor list read from the log at ``frame_idx``.

    Offline callers (no ``_execution_agent_states``) used to score against
    agents FROZEN at the current frame while the planner forecast them at
    constant velocity — two different worlds for one decision. This mirrors
    :func:`~.forward_sim.predict_agents_constant_velocity` (same frame clamp,
    same past-log hold with zero velocity, same dims resolver) so both roll
    the same agents forward with the same velocity convention.
    """
    assert scorer.scenario_data is not None
    tracks = scorer.scenario_data.get("tracks", {})
    out: List[Dict[str, Any]] = []
    for track_id, track in tracks.items():
        if str(track_id) == str(scorer.sdc_id):
            continue
        state = track.get("state", {})
        positions = state.get("position")
        if positions is None:
            continue
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim < 2 or positions.shape[0] == 0:
            continue
        sample = min(int(frame_idx), positions.shape[0] - 1)
        past_log = int(frame_idx) >= positions.shape[0]
        valid = state.get("valid")
        if valid is not None and not bool(np.asarray(valid, dtype=bool)[sample]):
            continue
        velocities = state.get("velocity")
        if past_log or velocities is None or len(velocities) <= sample:
            vx = vy = 0.0
        else:
            vx, vy = float(velocities[sample][0]), float(velocities[sample][1])
        headings = state.get("heading")
        heading = (float(headings[sample]) if headings is not None
                   and len(headings) > sample else math.atan2(vy, vx))
        length, width = scorer._track_dims(track, sample)
        out.append({
            "id": str(track_id), "type": str(track.get("type", "VEHICLE")),
            "position": positions[sample, :2].tolist(), "heading": heading,
            "velocity": [vx, vy], "length": length, "width": width,
            "valid": True, "is_ego": False,
        })
    return out


def _actor_types_for_scoring(scorer: EPDMSTrajectoryScorer_Fast,
                             ego_state: Dict[str, Any]) -> Dict[str, str]:
    """Actor type lookup aligned with live/log scorer object identifiers."""
    live = ego_state.get("_execution_agent_states")
    if live is not None:
        return {
            str(state.get("id", i)): str(state.get("type", "VEHICLE")).upper()
            for i, state in enumerate(live)
            if not state.get("is_ego", False)
        }
    assert scorer.scenario_data is not None
    return {
        str(track_id): str(track.get("type", "VEHICLE")).upper()
        for track_id, track in scorer.scenario_data.get("tracks", {}).items()
        if str(track_id) != str(scorer.sdc_id)
    }


# ----------------------------------------------------------------------
# Upstream weights — match tuplan_garage/CaRL pdm_scorer.py exactly.
# Reference: pdm_scorer.py:
#   WEIGHTED_METRICS_WEIGHTS[PROGRESS]    = 5.0
#   WEIGHTED_METRICS_WEIGHTS[TTC]         = 5.0
#   WEIGHTED_METRICS_WEIGHTS[COMFORTABLE] = 2.0
# ----------------------------------------------------------------------

W_PROGRESS_UPSTREAM: float = 5.0
W_TTC_UPSTREAM: float = 5.0
W_COMFORTABLE_UPSTREAM: float = 2.0
W_TOTAL_UPSTREAM: float = (
    W_PROGRESS_UPSTREAM + W_TTC_UPSTREAM + W_COMFORTABLE_UPSTREAM
)
"""``W_TOTAL_UPSTREAM == 12``. Matches tuplan_garage / CaRL exactly."""


# ----------------------------------------------------------------------
# NavSim-extended weights — current NavSim/EPDMS scoring (minus EC).
# Adds TLC to multiplicative and Lane-Keeping to weighted.
# ----------------------------------------------------------------------

W_PROGRESS_NAVSIM: float = 5.0
W_TTC_NAVSIM: float = 5.0
W_LANE_KEEPING_NAVSIM: float = 2.0
W_HISTORY_COMFORT_NAVSIM: float = 2.0
W_TOTAL_NAVSIM: float = (
    W_PROGRESS_NAVSIM
    + W_TTC_NAVSIM
    + W_LANE_KEEPING_NAVSIM
    + W_HISTORY_COMFORT_NAVSIM
)
"""``W_TOTAL_NAVSIM == 14``. Matches NavSim's PDMS (no Extended Comfort)."""


# ----------------------------------------------------------------------
# Backward-compat aliases — older code imported these as PDMS_*.
# Pinned to the upstream values (the new default).
# ----------------------------------------------------------------------

W_PROGRESS_PDMS: float = W_PROGRESS_UPSTREAM
W_TTC_PDMS: float = W_TTC_UPSTREAM
W_LANE_KEEPING_PDMS: float = W_LANE_KEEPING_NAVSIM
W_HISTORY_COMFORT_PDMS: float = W_COMFORTABLE_UPSTREAM
W_TOTAL_PDMS: float = W_TOTAL_NAVSIM
"""Deprecated: use the ``W_*_UPSTREAM`` / ``W_*_NAVSIM`` constants."""


# ======================================================================
# _UpstreamPDMSScorerFast — strict tuplan_garage/CaRL aggregation
# ======================================================================


class _UpstreamPDMSScorerFast(EPDMSTrajectoryScorer_Fast):
    """EPDMS fast scorer with strict-upstream PDMS aggregation.

    ``red_light_validity`` is an experimental compatibility switch.  The
    :class:`PDMScorer` upstream variant leaves it disabled: red connectors
    enter the observation as stationary obstacles, while TLC is absent from
    upstream aggregation.

    Aggregation::

        multi_prod    = nc * dac * ddc                                # 3
        weighted_sum  = 5 * normalized_progress + 5*ttc + 2*hc        # 3 (HC ≡ Comfortable)
        score         = multi_prod * weighted_sum / 12

    Drops TLC and Lane-Keeping from upstream-faithful aggregation;
    HC is the upstream-equivalent of ``ego_is_comfortable``.

    Relative-progress normalisation
    -------------------------------

    Faithful to ``carl_nuplan/.../pdm_scorer.py::_aggregate_scores``
    (and identically to ``tuplan_garage``'s version). The progress
    metric used in the weighted sum is *relative*, not absolute:

    1. Compute each proposal's raw progress in metres.
    2. Mask by ``multi_prod`` (proposals that violate any of NC, DAC,
       DDC contribute zero).
    3. Normalise by the maximum masked progress, when that maximum
       exceeds :data:`PROGRESS_DISTANCE_THRESHOLD_M`. When all
       proposals have near-zero progress (e.g. ego is essentially
       stopped), every proposal receives ``progress = 1.0``, then
       ``0.0`` for the proposals with ``multi_prod == 0``.
    4. Plug the normalised progress into the weighted sum.

    Progress is measured along the route **centerline** when the
    caller provides one (``centerline`` kwarg — the planner passes its
    extracted route), matching upstream's ``_calculate_progress``:
    endpoints are projected onto the centerline and the arc-length
    delta is taken, so off-route motion (continuing straight where the
    route turns) earns ~zero progress. Without a centerline the
    measurement falls back to chord distance between the first and
    last sample — adequate on straights, but it over-rewards off-route
    proposals at turns.

    The :class:`_NavSimPDMSScorerFast` variant intentionally retains
    the absolute ``ep = min(dist / 30, 1.0)`` formulation NavSim's
    leaderboard uses; only this strict-upstream variant applies the
    relative normalisation.
    """

    #: Experimental switch to fold TLC into the validity product.
    red_light_validity: bool = False

    def _get_best_lane(self, x: float, y: float, heading: float,
                       nearby_indices: List[int], speed: Optional[float] = None,
                       *, heading_gate: bool = True):
        """Skip losing heading projections for scenario-backed lanes only."""
        if type(nearby_indices) not in (list, tuple) or not nearby_indices or not heading_gate:
            return super()._get_best_lane(
                x, y, heading, nearby_indices, speed, heading_gate=heading_gate)
        # Custom/live lane methods may have side effects or raise. Delegate the
        # whole query before calling any lane method to retain their ordering.
        try:
            scenario_lanes = all(
                type(self.all_lanes[idx][0]) is LaneProxy for idx in nearby_indices)
        except (IndexError, KeyError, TypeError):
            scenario_lanes = False
        if not scenario_lanes:
            return super()._get_best_lane(
                x, y, heading, nearby_indices, speed, heading_gate=heading_gate)

        candidates = []
        best_accepted_distance = None
        is_stopped = (speed is not None) and (speed < 1.0)
        pos = np.array([x, y])
        for idx in nearby_indices:
            lane = self.all_lanes[idx][0]
            dist = lane.distance(pos)
            # Only an accepted earlier lane can exclude a farther candidate.
            # >= retains the first tie; NaN comparisons fall through as before.
            if best_accepted_distance is not None and dist >= best_accepted_distance:
                continue
            s, r = lane.local_coordinates(pos)
            s_clamped = max(0, min(s, lane.length))
            lane_heading_vec = lane.heading_at(s_clamped)
            lane_heading = math.atan2(lane_heading_vec[1], lane_heading_vec[0])
            diff = abs(heading - lane_heading)
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            if is_stopped or (abs(diff) < (np.pi / 2)):
                candidates.append((lane, dist))
                if best_accepted_distance is None or dist < best_accepted_distance:
                    best_accepted_distance = dist

        if candidates:
            return min(candidates, key=lambda c: c[1])[0]
        return None

    def select_best(  # type: ignore[override]
        self,
        model_output: Dict[str, Any],
        *,
        # Explicit keyword parameters (not ``kwargs[...]`` pulls): a missing
        # or misspelled argument is a TypeError at the call site instead of
        # a KeyError (or a silently swallowed kwarg) mid-scoring. The
        # signature narrowing vs the base's untyped ``**kwargs`` grab-bag is
        # deliberate — hence the override ignore above.
        ego_state: Dict[str, Any],
        frame_idx: int,
        centerline: Optional[np.ndarray] = None,
        # Number of leading proposals forming the stock base grid. When given,
        # progress normalisation is anchored to the base grid so appended
        # safety-brake proposals cannot rescale it — see the normalisation
        # block below. ``None`` keeps the legacy behaviour.
        n_base: Optional[int] = None,
        # Retained for base-class (**kwargs) signature compatibility only;
        # anything landing here is a caller bug and is rejected loudly.
        **unexpected: Any,
    ) -> Dict[str, Any]:
        if unexpected:
            raise TypeError(
                "select_best() got unexpected keyword arguments: "
                f"{sorted(unexpected)}")
        if not self._initialized:
            raise RuntimeError("_UpstreamPDMSScorerFast not initialized.")
        assert self.scenario_data is not None  # set in initialize()
        t_start = time.time()

        all_candidates = model_output["all_candidates"]
        candidates_np = all_candidates[0].cpu().numpy()
        N = candidates_np.shape[0]
        horizon = candidates_np.shape[1]

        # Planning must be causal.  Score against the same live snapshot the
        # proposal generator received, rolled forward with the scorer's CV
        # model.  With no live snapshot (offline/unit callers), hold the
        # current logged observation; never read future logged poses.
        #
        # The reference samples its occupancy maps at 0.2 s
        # (``observation_sample_res=2``; ``__getitem__`` maps
        # ``time_idx // 2``); ``_calculate_metrics`` reproduces that lookup
        # for this scorer (``observation_sample_res`` set in ``__init__``).
        # Without a live snapshot (offline callers) the log at ``frame_idx``
        # is read into the same snapshot shape, so the scorer rolls the same
        # agents forward as ``predict_agents_constant_velocity`` does.
        _current_agents, current_red = self._precompute_frame_data(frame_idx, 0)
        live_agent_states = ego_state.get("_execution_agent_states")
        if live_agent_states is None:
            live_agent_states = _snapshot_from_log(self, int(frame_idx))
        ttc_extra_steps = 9
        # ``_live_agents_per_t`` needs the ego position for the reference's
        # nearest-k admission; offline callers may pass a snapshot without it.
        snapshot = list(live_agent_states)
        if not any(bool(s.get("is_ego", False)) for s in snapshot):
            snapshot.append({"is_ego": True, "position": list(
                np.asarray(ego_state["position"], dtype=np.float64)[:2])})
        agents_per_t = self._live_agents_per_t(
            snapshot, horizon + ttc_extra_steps)
        agents_per_t = self._filter_agents_to_map_radius(
            agents_per_t, ego_state
        )
        active_lane_indices, active_drivable_area = (
            self._active_map_context(ego_state)
        )
        red_now = set(current_red[0]) if current_red else set()
        red_lanes_per_t = [set(red_now) for _ in range(horizon + 1)]
        candidates_world = self._ego_to_world(candidates_np, ego_state)

        sdc_track = self.scenario_data["tracks"][self.sdc_id]
        # Past the log the SDC has no recorded validity to consult -- the ego
        # is being simulated, not replayed, and `_precompute_frame_data`'s
        # `sim_frame < scenario_length` guards already yield empty agent lists
        # there. Indexing raw died with `IndexError: index 205 is out of bounds
        # for axis 0 with size 201` the first time an episode outlived its 20 s
        # bundle; early-returning instead would hand back candidate 0 unscored
        # for the whole post-log phase, which is worse than scoring against an
        # empty world. Clamp the read, branch on the true frame -- the same
        # shape traffic/semi_reactive.py uses for its own post-log reads.
        _sdc_valid = sdc_track["state"]["valid"]
        _past_log = int(frame_idx) >= len(_sdc_valid)
        if not _past_log and not _sdc_valid[int(frame_idx)]:
            best_traj = all_candidates[0, 0]
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.zeros(1, N),
                "best_idx": torch.tensor([0]),
                "collision_times": torch.full(
                    (1, N), float("inf"), dtype=torch.float64),
            }

        # Prepend the ACTUAL ego position — the frame the candidates
        # were expressed in. Prepending the GT log position corrupts
        # the first finite-difference segment of every proposal by the
        # ego's drift from the log (speed ~10x drift, jerk ~1000x),
        # zeroing HC and rotating early ego polygons for all proposals.
        current_pos = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        ego_pos_tiled = np.broadcast_to(current_pos[None, None, :], (N, 1, 2))
        all_paths_world = np.concatenate([ego_pos_tiled, candidates_world], axis=1)
        all_paths_sim = all_paths_world + self.world_to_sim_offset

        all_states = self._get_all_trajectory_states(all_paths_sim, self.planner_dt)
        all_states, has_pdm_full_states, pdm_comfort = _apply_pdm_full_states(
            model_output, all_states, self.world_to_sim_offset,
            num_proposals=N, horizon=horizon, planner_dt=self.planner_dt)

        # Optional route centerline (world frame) for upstream-faithful
        # progress. Passed through by PDMScorer.score_proposals.
        # Segments and the shared start-point projection are computed
        # once here, not once per proposal.
        projector: Optional[_CenterlineProjector] = None
        s_start = 0.0
        if centerline is not None:
            projector = _CenterlineProjector(centerline)
            if projector.degenerate:
                projector = None
            else:
                s_start = projector.arc_length_of(all_paths_world[0, 0])

        # ------------------------------------------------------------------
        # First pass: compute per-proposal multiplicative metrics, weighted
        # metric inputs (TTC, HC), and raw progress in metres.
        # ------------------------------------------------------------------
        multi_prod = np.zeros(N, dtype=np.float64)
        ttc_vals = np.zeros(N, dtype=np.float64)
        hc_vals = np.zeros(N, dtype=np.float64)
        raw_progress = np.zeros(N, dtype=np.float64)
        collision_times = np.full(N, np.inf, dtype=np.float64)
        actor_types = _actor_types_for_scoring(self, ego_state)
        for i in range(N):
            states_i = {key: arr[i] for key, arr in all_states.items()}

            metrics = self._calculate_metrics(
                states_i,
                horizon,
                frame_idx,
                ego_state=ego_state,
                # Deliberately None: inside _calculate_metrics ``n_execute``
                # only shapes the ``ec`` (extended-comfort) window, and this
                # aggregation never reads ``ec`` — the replan-pacing
                # derivation from ``prev_frame_idx`` that used to live here
                # was proven score-identical (bit-probe on the banked loop2
                # corpus) and only mutated hidden scorer state.
                n_execute=None,
                agents_per_t=agents_per_t,
                red_lanes_per_t=red_lanes_per_t,
                carl_collision_classifier=has_pdm_full_states,
                active_lane_indices=active_lane_indices,
                active_drivable_area=active_drivable_area,
            )
            if pdm_comfort is not None:
                metrics["hc"] = float(pdm_comfort[i])

            # Exact upstream gates: any off-road pose fails DAC, and an
            # at-fault collision with a static object scores 0.5 rather than
            # the 0 used for agent collisions.
            metrics["dac"] = float(float(metrics["dac"]) >= 1.0 - 1e-12)
            if float(metrics["nc"]) == 0.0:
                at_fault_ids = metrics.get("collision_at_fault_ids")
                if not at_fault_ids:
                    collision_id = metrics.get("collision_actor_id")
                    at_fault_ids = (
                        [] if collision_id is None else [str(collision_id)])
                metrics["nc"] = _upstream_nc_from_at_fault_ids(
                    at_fault_ids, actor_types)
            collision_times[i] = float(
                metrics.get("collision_time_s", float("inf")))

            # Upstream multiplicative: NC * DAC * DDC. There is no TLC term:
            # red connectors enter the current observation as stationary lead
            # obstacles, while upstream NC/TTC explicitly ignore red tokens.
            # ``red_light_validity`` is retained only for opt-in experiments;
            # the upstream wrapper below leaves it disabled.
            multi_prod[i] = float(metrics["nc"]) * float(metrics["dac"]) * float(
                metrics["ddc"]
            )
            if self.red_light_validity:
                multi_prod[i] *= float(metrics.get("tlc", 1.0))
            ttc_vals[i] = float(metrics["ttc"])
            # HC (max accel/jerk/yaw_rate within thresholds) is the
            # upstream-equivalent of ``ego_is_comfortable`` — both are
            # boolean aggregates of comfort sub-metrics.
            hc_vals[i] = float(metrics["hc"])

            # Raw progress in metres. With a route centerline, project
            # the endpoints onto it (upstream ``_calculate_progress``):
            # off-route motion earns ~zero progress. Without one, fall
            # back to chord distance between first and last sample.
            if projector is not None:
                raw_progress[i] = max(
                    0.0,
                    projector.arc_length_of(all_paths_world[i, -1]) - s_start,
                )
            else:
                dx = float(all_states["x"][i, -1]) - float(all_states["x"][i, 0])
                dy = float(all_states["y"][i, -1]) - float(all_states["y"][i, 0])
                raw_progress[i] = math.sqrt(dx * dx + dy * dy)

        # ------------------------------------------------------------------
        # Relative progress normalisation (CaRL / tuplan_garage style).
        # ------------------------------------------------------------------
        # CaRL scales raw progress by the complete multiplicative score
        # before relative normalization. NC and DDC are deliberately
        # fractional for static-object collisions and 2--6 m wrong-way
        # travel, respectively; reducing this to a boolean mask made those
        # proposals retain full progress credit and did not match the
        # reference implementation.
        masked_progress = _mask_upstream_progress(raw_progress, multi_prod)
        # Anchor progress to the stock grid so the appended safety brake
        # cannot change stock proposal scores.
        base_progress = (masked_progress[:n_base]
                         if n_base is not None and 0 < n_base <= N
                         else masked_progress)
        max_progress = float(base_progress.max()) if base_progress.size else 0.0
        if max_progress > PROGRESS_DISTANCE_THRESHOLD_M:
            normalized_progress = masked_progress / max_progress
            normalized_progress = np.minimum(normalized_progress, 1.0)
        else:
            # Every proposal made (almost) no progress — typically
            # because the ego is stopped or the horizon is very short.
            # Upstream gives all proposals progress = 1.0 in this
            # branch (then zeroes out the disqualified ones below);
            # the rationale is that with no movement to compare,
            # progress is not a useful discriminator and should be
            # neutral with respect to the other metrics.
            normalized_progress = np.ones(N, dtype=np.float64)
        normalized_progress[multi_prod == 0.0] = 0.0

        # ------------------------------------------------------------------
        # Second pass: compute final scores using normalised progress.
        # ------------------------------------------------------------------
        # Upstream weighted: 5*Progress + 5*TTC + 2*Comfortable, with
        # Progress now relative-normalised across proposals.
        weighted_sum = (
            W_PROGRESS_UPSTREAM * normalized_progress
            + W_TTC_UPSTREAM * ttc_vals
            + W_COMFORTABLE_UPSTREAM * hc_vals
        )
        scores = multi_prod * weighted_sum / W_TOTAL_UPSTREAM

        best_idx = int(np.argmax(scores))

        elapsed = time.time() - t_start
        if self.verbose:
            print(
                f"[Upstream PDMS] Frame {frame_idx}: best_idx={best_idx}/{N}, "
                f"score={scores[best_idx]:.4f}, max={scores.max():.4f}, "
                f"mean={scores.mean():.4f}, min={scores.min():.4f}, "
                f"time={elapsed:.2f}s"
            )

        best_traj = all_candidates[0, best_idx]
        return {
            "trajectory": best_traj.unsqueeze(0),
            "scores": torch.from_numpy(scores).unsqueeze(0).float(),
            "best_idx": torch.tensor([best_idx]),
            "collision_times": torch.from_numpy(collision_times).unsqueeze(0),
        }


# ======================================================================
# _NavSimPDMSScorerFast — NavSim-extended PDMS (drops EC only)
# ======================================================================


class _NavSimPDMSScorerFast(EPDMSTrajectoryScorer_Fast):
    """EPDMS fast scorer with NavSim-extended PDMS aggregation.

    Aggregation::

        multi_prod    = nc * dac * ddc * tlc                          # 4 (extra: TLC)
        weighted_sum  = 5*ep + 5*ttc + 2*lk + 2*hc                    # 4 (extra: LK)
        score         = multi_prod * weighted_sum / 14

    This is the original ``_PDMSTrajectoryScorerFast`` from before the
    upstream-strict refactor. Use it when you want the planner's
    internal scoring to align with NavSim leaderboard semantics.
    """

    def select_best(  # type: ignore[override]
        self,
        model_output: Dict[str, Any],
        *,
        ego_state: Dict[str, Any],
        frame_idx: int,
        # Accepted for call-signature parity with the upstream variant —
        # PDMScorer.score_proposals passes all three to either variant —
        # but not used by this aggregation (absolute /30 m progress needs
        # no centerline, and there is no relative normalisation for
        # n_base to anchor).
        centerline: Optional[np.ndarray] = None,
        n_base: Optional[int] = None,
        # Retained for base-class (**kwargs) signature compatibility only;
        # anything landing here is a caller bug and is rejected loudly.
        # Signature narrowing vs the base's untyped ``**kwargs`` is
        # deliberate — see the override ignore on the def line.
        **unexpected: Any,
    ) -> Dict[str, Any]:
        if unexpected:
            raise TypeError(
                "select_best() got unexpected keyword arguments: "
                f"{sorted(unexpected)}")
        if not self._initialized:
            raise RuntimeError("_NavSimPDMSScorerFast not initialized.")
        assert self.scenario_data is not None  # set in initialize()

        t_start = time.time()

        all_candidates = model_output["all_candidates"]
        candidates_np = all_candidates[0].cpu().numpy()
        N = candidates_np.shape[0]
        horizon = candidates_np.shape[1]

        current_agents, current_red = self._precompute_frame_data(frame_idx, 0)
        live_agent_states = ego_state.get("_execution_agent_states")
        ttc_extra_steps = 9
        agents_per_t = (
            self._live_agents_per_t(
                live_agent_states, horizon + ttc_extra_steps
            )
            if live_agent_states is not None
            else [
                list(current_agents[0])
                for _ in range(horizon + ttc_extra_steps + 1)
            ]
        )
        agents_per_t = self._filter_agents_to_map_radius(
            agents_per_t, ego_state
        )
        active_lane_indices, active_drivable_area = (
            self._active_map_context(ego_state)
        )
        red_now = set(current_red[0]) if current_red else set()
        red_lanes_per_t = [set(red_now) for _ in range(horizon + 1)]
        candidates_world = self._ego_to_world(candidates_np, ego_state)

        sdc_track = self.scenario_data["tracks"][self.sdc_id]
        # Past the log the SDC has no recorded validity to consult -- the ego
        # is being simulated, not replayed, and `_precompute_frame_data`'s
        # `sim_frame < scenario_length` guards already yield empty agent lists
        # there. Indexing raw died with `IndexError: index 205 is out of bounds
        # for axis 0 with size 201` the first time an episode outlived its 20 s
        # bundle; early-returning instead would hand back candidate 0 unscored
        # for the whole post-log phase, which is worse than scoring against an
        # empty world. Clamp the read, branch on the true frame -- the same
        # shape traffic/semi_reactive.py uses for its own post-log reads.
        _sdc_valid = sdc_track["state"]["valid"]
        _past_log = int(frame_idx) >= len(_sdc_valid)
        if not _past_log and not _sdc_valid[int(frame_idx)]:
            best_traj = all_candidates[0, 0]
            return {
                "trajectory": best_traj.unsqueeze(0),
                "scores": torch.zeros(1, N),
                "best_idx": torch.tensor([0]),
                "collision_times": torch.full(
                    (1, N), float("inf"), dtype=torch.float64),
            }

        # Actual ego position, not the GT log's (see upstream variant).
        current_pos = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        ego_pos_tiled = np.broadcast_to(current_pos[None, None, :], (N, 1, 2))
        all_paths_world = np.concatenate([ego_pos_tiled, candidates_world], axis=1)
        all_paths_sim = all_paths_world + self.world_to_sim_offset

        all_states = self._get_all_trajectory_states(all_paths_sim, self.planner_dt)
        all_states, has_pdm_full_states, pdm_comfort = _apply_pdm_full_states(
            model_output, all_states, self.world_to_sim_offset,
            num_proposals=N, horizon=horizon, planner_dt=self.planner_dt)

        scores = np.zeros(N)
        collision_times = np.full(N, np.inf, dtype=np.float64)
        for i in range(N):
            states_i = {key: arr[i] for key, arr in all_states.items()}

            metrics = self._calculate_metrics(
                states_i,
                horizon,
                frame_idx,
                ego_state=ego_state,
                # Deliberately None — see the upstream variant: ``n_execute``
                # only feeds the ``ec`` term, which this aggregation never
                # reads, so the old ``prev_frame_idx`` pacing derivation was
                # dead for the score and only mutated hidden state.
                n_execute=None,
                agents_per_t=agents_per_t,
                red_lanes_per_t=red_lanes_per_t,
                carl_collision_classifier=has_pdm_full_states,
                active_lane_indices=active_lane_indices,
                active_drivable_area=active_drivable_area,
            )
            if pdm_comfort is not None:
                metrics["hc"] = float(pdm_comfort[i])
            collision_times[i] = float(
                metrics.get("collision_time_s", float("inf")))

            # NavSim multiplicative: NC * DAC * DDC * TLC.
            multi_prod = (
                metrics["nc"] * metrics["dac"] * metrics["ddc"] * metrics["tlc"]
            )
            # NavSim weighted (no EC).
            weighted_sum = (
                W_PROGRESS_NAVSIM * metrics["ep"]
                + W_TTC_NAVSIM * metrics["ttc"]
                + W_LANE_KEEPING_NAVSIM * metrics["lk"]
                + W_HISTORY_COMFORT_NAVSIM * metrics["hc"]
            )
            scores[i] = multi_prod * weighted_sum / W_TOTAL_NAVSIM

        best_idx = int(np.argmax(scores))

        elapsed = time.time() - t_start
        if self.verbose:
            print(
                f"[NavSim PDMS] Frame {frame_idx}: best_idx={best_idx}/{N}, "
                f"score={scores[best_idx]:.4f}, max={scores.max():.4f}, "
                f"mean={scores.mean():.4f}, min={scores.min():.4f}, "
                f"time={elapsed:.2f}s"
            )

        best_traj = all_candidates[0, best_idx]
        return {
            "trajectory": best_traj.unsqueeze(0),
            "scores": torch.from_numpy(scores).unsqueeze(0).float(),
            "best_idx": torch.tensor([best_idx]),
            "collision_times": torch.from_numpy(collision_times).unsqueeze(0),
        }


# ======================================================================
# PDMScorer — public wrapper
# ======================================================================


class PDMScorer:
    """PDM-Closed proposal scorer.

    Wraps either :class:`_UpstreamPDMSScorerFast` (default, faithful
    to ``tuplan_garage`` / CaRL) or :class:`_NavSimPDMSScorerFast`
    (NavSim-extended, opt-in).

    Attributes:
        cfg: PDM configuration.
        variant: ``"upstream"`` or ``"navsim"``.
        scorer: The wrapped fast scorer.
        scenario_data: Last scenario the scorer was initialised against.
    """

    def __init__(
        self,
        cfg: PDMConfig,
        *,
        verbose: bool = False,
        variant: Optional[ScorerVariant] = None,
    ) -> None:
        self.cfg = cfg
        # Variant priority: explicit kwarg > cfg.scorer_variant > "upstream".
        if variant is None:
            variant = getattr(cfg, "scorer_variant", "upstream")
        if variant not in ("upstream", "navsim"):
            raise ValueError(
                f"variant must be 'upstream' or 'navsim', got {variant!r}"
            )
        self.variant: ScorerVariant = variant
        self.scorer: _UpstreamPDMSScorerFast | _NavSimPDMSScorerFast
        if variant == "upstream":
            self.scorer = _UpstreamPDMSScorerFast(verbose=verbose)
            # Upstream handles red signals only through the current
            # observation; TLC is not part of PDMScorer's multiplicative
            # product and red tokens are ignored by NC/TTC. The explicit
            # NavSafe score-brake experiment re-enables TLC validity so its
            # red-floor arm has an all-invalid signal to act on; the faithful
            # default (score arms ``none``) remains byte-for-byte upstream.
            self.scorer.red_light_validity = (
                cfg.traffic_light_obstacles
                and cfg.emergency_brake_score_arms != "none")
        else:
            self.scorer = _NavSimPDMSScorerFast(verbose=verbose)
        self.scorer.planner_dt = float(cfg.sim_dt)
        self.scorer.map_radius_m = float(cfg.map_radius_m)
        # The reference's NC/TTC angle tests are anchored at the ego REAR
        # AXLE. NavSafe's pose is the centre of a symmetric bicycle, whose
        # rear axle sits half a wheelbase behind it — the same L/2 the
        # forward-sim uses for its "agent ahead" test. The module default
        # (``PDM_REAR_AXLE_TO_CENTER_M``, Pacifica's 1.461 m) stays untouched
        # for every other consumer of the batch scorer.
        self.scorer.rear_axle_to_center_m = 0.5 * float(cfg.wheelbase)
        # ``PDMObservation(observation_sample_res=2)``: NC/TTC read the
        # 0.2 s-sampled forecast (``time_idx // 2``). Exact per-step agents
        # would be strictly more accurate; parity with the reference wins.
        self.scorer.observation_sample_res = 2
        # Score the car the planner models (its own config dims).
        self.scorer.ego_length_m = float(cfg.ego_length)
        self.scorer.ego_width_m = float(cfg.ego_width)
        self.scenario_data: Optional[dict] = None
        self.last_collision_times: np.ndarray = np.empty(
            0, dtype=np.float64)

    def initialize(
        self,
        scenario_data: dict,
        env: Optional[object] = None,
    ) -> None:
        """Bind this scorer to a scenario."""
        self.scenario_data = scenario_data
        self.scorer.initialize(scenario_data, env)
        # Re-tune to the planner's own step; the scorer's gt_stride property
        # recomputes the clamped stride from planner_dt on every access. This
        # used to divide inline, which silently undid the scorer's gt_stride>=1
        # guarantee on the planner's path (the evaluator's path kept it, so the
        # bug hid): with a misparsed scenario_dt the planner scored every
        # proposal against a frozen world.
        self.scorer.planner_dt = float(self.cfg.sim_dt)

    def score_proposals(
        self,
        proposals: List[Proposal],
        ego_state: dict,
        frame_id: int,
        *,
        centerline: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """Score every proposal and return ``(scores, best_idx)``.

        Statefulness: the wrapped scorer is stateful only in its scenario
        binding (``initialize``). Neither variant's ``select_best`` reads or
        writes ``prev_frame_idx`` any more — the replan-pacing ``n_execute``
        it used to derive only ever fed the parent scorer's ``ec`` term,
        which neither aggregation reads (proven score-identical by bit-probe
        on the banked loop2 corpus). The planner still writes
        ``prev_frame_idx`` onto the wrapped scorer before calling this;
        that write is inert here (it remains live for the parent EPDMS
        scorer the evaluator uses). Repeated calls at one frame are safe.
        """
        if self.scenario_data is None:
            raise RuntimeError(
                "PDMScorer.score_proposals called before initialize(); "
                "call initialize(scenario_data, env=None) first."
            )
        if not proposals:
            raise ValueError("proposals must be non-empty")
        # Candidate mode appends a safety brake after the stock grid.
        if len(proposals) < self.cfg.num_proposals:
            raise ValueError(
                f"expected at least {self.cfg.num_proposals} proposals "
                f"(the base grid), got {len(proposals)}"
            )

        n = len(proposals)
        t = self.cfg.num_sim_steps

        candidates = np.zeros((1, n, t, 3), dtype=np.float32)
        ego_xy = np.asarray(ego_state["position"], dtype=np.float64)[:2]
        ego_heading = float(ego_state["heading"])
        cos_h = float(np.cos(-ego_heading))
        sin_h = float(np.sin(-ego_heading))
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
        for i, prop in enumerate(proposals):
            dense_world = np.column_stack([prop.state.x[1 : t + 1], prop.state.y[1 : t + 1]])
            if dense_world.shape != (t, 2):
                raise ValueError(
                    f"proposal {i} dense rollout has shape {dense_world.shape}, "
                    f"expected ({t}, 2)"
                )
            rel = dense_world - ego_xy
            dense_ego = (rot @ rel.T).T
            candidates[0, i, :, 0] = dense_ego[:, 0]  # forward
            candidates[0, i, :, 1] = dense_ego[:, 1]  # lateral

        candidates_t = torch.from_numpy(candidates)
        full_states = {
            key: np.stack([
                np.asarray(getattr(prop.state, key), dtype=np.float64)[: t + 1]
                for prop in proposals
            ], axis=0)
            for key in ("x", "y", "heading", "speed")
        }
        missing_full_state = [
            i for i, prop in enumerate(proposals)
            if getattr(prop.state, "full_state", None) is None
        ]
        if missing_full_state:
            raise ValueError(
                "PDM proposals must carry the simulator's 11-channel "
                "full_state; missing on proposal indices "
                f"{missing_full_state}")
        full_states["state"] = np.stack([
            np.asarray(prop.state.full_state, dtype=np.float64)[: t + 1]
            for prop in proposals
        ], axis=0)
        result = self.scorer.select_best(
            {
                "all_candidates": candidates_t,
                "pdm_full_states": full_states,
            },
            ego_state=ego_state,
            frame_idx=frame_id,
            centerline=centerline,
            # The base grid is always the leading cfg.num_proposals entries
            # (the safety brake appends past it, see score_proposals' length
            # check). Anchoring progress normalisation there keeps an
            # appended safety-brake proposal from rescaling the base grid's scores.
            n_base=int(self.cfg.num_proposals),
        )
        scores = result["scores"].squeeze(0).cpu().numpy().astype(np.float64)
        collision_times = result.get("collision_times")
        self.last_collision_times = (
            collision_times.squeeze(0).cpu().numpy().astype(np.float64)
            if collision_times is not None
            else np.full(n, np.inf, dtype=np.float64)
        )
        best_idx = int(result["best_idx"].item())
        return scores, best_idx


__all__ = [
    "PDMScorer",
    "ScorerVariant",
    "PROGRESS_DISTANCE_THRESHOLD_M",
    "W_PROGRESS_UPSTREAM",
    "W_TTC_UPSTREAM",
    "W_COMFORTABLE_UPSTREAM",
    "W_TOTAL_UPSTREAM",
    "W_PROGRESS_NAVSIM",
    "W_TTC_NAVSIM",
    "W_LANE_KEEPING_NAVSIM",
    "W_HISTORY_COMFORT_NAVSIM",
    "W_TOTAL_NAVSIM",
    # Backward-compat aliases
    "W_PROGRESS_PDMS",
    "W_TTC_PDMS",
    "W_LANE_KEEPING_PDMS",
    "W_HISTORY_COMFORT_PDMS",
    "W_TOTAL_PDMS",
]
