"""Vectorised LaneProxy variant — superseded; kept for a registered policy.

``FastLaneProxy`` overrides ``local_coordinates`` with a numpy-vectorised
implementation of the same formulas — point-to-segment projection,
first-minimum tie-breaking, clamped endpoints, signed lateral convention —
though not, as it turns out, of the same arithmetic (see the caveat below).
It was written when ``LaneProxy.local_coordinates`` was still a per-segment
Python loop that profiled at ~85% of a 16 s PDM-Closed inference (~1M scalar
``np.linalg.norm`` calls).

**Superseded, but still load-bearing.** The 2026-08 perf pass vectorised the
reference itself, erasing this module's advantage and reversing it (banked
loop-2 corpus: ~186 ms/frame through this class against ~164 ms/frame through
:class:`~navsafe.evaluation.utils.lane_proxy.LaneProxy`), which is now the
more exact of the two as well. It is not marked deprecated because nothing can
migrate off it yet: ``pdm_closed_fast`` — which switches it on via the batch
scorer's ``use_fast_lanes`` flag — is a registered policy name, the default
``expert_model_type``, and the only expert
``outer.trial_plan._REGISTERED_EXPERTS`` permits. Prefer the reference
wherever the choice is free; retiring this module means changing those
surfaces first.

Exactness caveat: the reference reproduces the original scalar loop
bit-for-bit — a property pinned by
``tests/evaluation/test_lane_proxy_vectorized_parity.py``, not one that holds
by construction — and this class does NOT. Its ``np.hypot`` differs from the
loop's ``np.linalg.norm`` at 1 ulp on ~17% of pointwise inputs, which shows up
in three graded ways:

* **unique nearest segment** (almost everywhere) — ``s`` is bit-exact and
  ``|r|`` agrees within 1 ulp;
* **segments tied within an ulp** — the ulp decides the ``argmin``, so the
  reported longitudinal position can move by tens of metres (the centre of a
  curved lane is the clean case). Only ``|r|`` still agrees;
* **non-finite query** — nothing agrees, and the divergence runs the wrong
  way. The reference's ``dist < best_dist`` is never true for a NaN, so it
  keeps its ``(0.0, inf)`` incumbents (infinitely off-lane); ``np.argmin``
  returns the first NaN index and ``best_dist > 0.0`` is False, so this class
  reports ``r = 0.0`` — *exactly on the centre-line*. That is a fail-BEST
  inversion against the repo's fail-worst policy for non-finite candidates,
  in the default expert's lane path. It is pinned as a known defect by the
  test module above; fixing it is a behaviour change to a live default and
  needs its own corpus replay.

No banked-corpus EPDMS score moves as a result — that corpus contains no tie
and no non-finite query — but the two are not interchangeable in general.
"""

import math
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon

from navsafe.evaluation.utils.lane_proxy import LaneProxy


class FastLaneProxy(LaneProxy):
    """``LaneProxy`` with a vectorised ``local_coordinates``."""

    def local_coordinates(self, point: np.ndarray) -> Tuple[float, float]:
        """Project *point* onto the lane center-line (vectorised).

        Returns:
            (s, r) where s is the longitudinal distance along the lane and
            r is the signed lateral offset (positive = left of travel
            direction). Semantics identical to ``LaneProxy.local_coordinates``.
        """
        pt = np.asarray(point, dtype=np.float64)[:2]
        # A non-finite query must fail WORST, exactly as the reference does.
        # Without this guard the NaN propagates into `dist`, `np.argmin`
        # returns the first NaN index (NaN compares False against everything),
        # and `best_dist > 0.0` is False for NaN — so `best_r` collapses to
        # 0.0 and the caller reads the ego as sitting exactly on the
        # centre-line. That is a fail-BEST inversion in the lane path of the
        # DEFAULT expert (`pdm_closed_fast`): a NaN pose would score lane
        # keeping and driving-direction compliance perfect rather than
        # unusable, the same class of defect as the non-finite fail-open
        # closed in 995a03a. The reference returns (0.0, inf) here.
        if not np.isfinite(pt).all():
            return (0.0, float('inf'))

        a = self._polyline[:-1]                       # (N-1, 2)
        ab = self._polyline[1:] - a                   # (N-1, 2)
        seg_len = self._seg_lengths                   # (N-1,)
        valid = seg_len >= 1e-12
        if not np.any(valid):
            # Mirrors the reference loop: no valid segment leaves the
            # initial (0.0, inf) incumbents untouched.
            return (0.0, float('inf'))

        ap = pt[None, :] - a                          # (N-1, 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.einsum('ij,ij->i', ap, ab) / (seg_len * seg_len)
        t = np.clip(np.where(valid, t, 0.0), 0.0, 1.0)
        proj = a + t[:, None] * ab                    # (N-1, 2)
        diff = pt[None, :] - proj                     # (N-1, 2)
        dist = np.hypot(diff[:, 0], diff[:, 1])
        dist = np.where(valid, dist, np.inf)

        # argmin returns the FIRST minimal index — same tie-breaking as the
        # reference's strict "<" incumbent update.
        i = int(np.argmin(dist))
        best_dist = float(dist[i])
        best_s = float(self._cum_lengths[i] + t[i] * seg_len[i])
        # Signed lateral: positive = left of travel direction. Magnitude is
        # the true distance so |r| stays honest for points beyond the lane's
        # longitudinal extent.
        tangent = self._tangents[i]
        signed_r = float(diff[i, 0] * -tangent[1] + diff[i, 1] * tangent[0])
        best_r = math.copysign(best_dist, signed_r) if best_dist > 0.0 else 0.0

        return (best_s, best_r)


def build_fast_lanes_from_scenario(
        scenario_data: dict) -> List[Tuple[FastLaneProxy, Polygon]]:
    """``build_lanes_from_scenario`` producing :class:`FastLaneProxy` objects."""
    from navsafe.scenario.type import MetaDriveType

    lanes = []
    for feature_id, feature_data in scenario_data.get('map_features', {}).items():
        if not MetaDriveType.is_lane(feature_data.get('type', '')):
            continue
        polyline = feature_data.get('polyline')
        if polyline is None or len(polyline) < 2:
            continue
        polygon = feature_data.get('polygon')
        try:
            proxy = FastLaneProxy(
                lane_id=str(feature_id),
                polyline=np.asarray(polyline),
                polygon=np.asarray(polygon) if polygon is not None else None,
                is_intersection=bool(feature_data.get("is_intersection", False)),
            )
        except (ValueError, IndexError):
            continue
        lanes.append((proxy, proxy.shapely_polygon))
    return lanes


__all__ = ["FastLaneProxy", "build_fast_lanes_from_scenario"]
