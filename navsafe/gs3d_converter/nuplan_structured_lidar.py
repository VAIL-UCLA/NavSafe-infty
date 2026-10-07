# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Recover a spinning-lidar model for nuPlan's merged point cloud.

nuPlan ships one ``MergedPointCloud`` per 20 Hz tick: five lidars fused into a
single ego-frame cloud with ``x y z intensity lidar_info ring``.  There is no
per-point timestamp and py123d records the extrinsics as identity ("Lidar
extrinsic are unknown"), so ``NCoreBridge`` has always written it through the
flattened path — one shared stamp for the whole sweep, no lidar intrinsics.
That is what makes lidar-*supervised* training impossible on nuPlan: NRE's
``get_lidar_data_batch`` needs the spinning model plus per-ray
``model_element``, and the flattened path registers neither.

Both are recoverable from the data itself:

* ``ring`` really is the beam index.  Fitting the lidar→ego extrinsic that makes
  every ring a constant-elevation cone drives the per-ring elevation scatter of
  the TOP sensor from 4.24 deg (ego frame, i.e. measured from the wrong origin)
  to **0.002 - 0.022 deg**, at a mount of ``t ~ [1.4, -0.02, 1.64] m`` -- a roof
  position.  The residual is the calibration quality; a real spinning lidar
  cannot fit that well by accident.  Five parameters, not six: yaw is fixed at
  zero because elevation cannot see it (see :meth:`calibrate`).
* The sweep is not motion-compensated, so per-ray times derived from azimuth
  phase let NRE deskew it against the native-rate rig trajectory the bridge
  already preserves.  Evidence is a speed-stratified sharpness test (occupied
  5 cm voxels over 20 aggregated sweeps in world frame): the gain is 0.00 % with
  the car stopped (0.03 m/s), grows with speed, and at 10.4 m/s separates the two
  spin hypotheses with *opposite* signs -- ccw -0.73 %, cw +0.35 %.  A wrong sign
  making things worse is the signature of a real, uncompensated sweep.

  Be honest about the size of it: averaged over four fast windows the best
  setting gains only ~0.1 %, and the wrong phase origin costs ~0.5 %.  The
  spinning model -- which is what unlocks lidar supervision at all -- is the
  solid part; the deskew is a small correction on top, and is switchable.

Only the TOP sensor is modelled: it is the one with full 360 deg azimuth, 40
beams and 220 m range.  The other four cover partial arcs and cannot share a
single spinning model.  The merged cloud keeps its own (unstructured) component
untouched, so Gaussian initialisation sees exactly the point set it always did
and only supervision gains a new source.

Interface mirrors :mod:`wod_structured_lidar` so ``NCoreBridge`` can treat both
the same way.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# nuPlan lidar ticks at 20 Hz (measured: 50002, 50005, 50010 us between
# consecutive lidar_pc rows).
_SWEEP_US = 50000

# lidar_info -> py123d LidarID; we model LIDAR_TOP only.
_TOP_LIDAR_ID = 2  # py123d LidarID.LIDAR_TOP

# Azimuth at which a sweep starts, i.e. the point of the rotation that carries
# the frame's own timestamp. Measured in the yaw-fixed sensor frame (see
# calibrate) by scanning it over four 10 m/s windows: aggregate sharpness is best
# at -157.5 deg (-0.105 %) and worst at -45 deg (+0.469 %, i.e. worse than not
# deskewing at all). -180 deg ties with -157.5, so the origin is pinned to about
# +/-20 deg -- +/-2.8 ms, or +/-3 cm of ego motion at 10 m/s.
_PHASE_ORIGIN_RAD = -2.748894  # -157.5 deg

# Refuse to build a model worse than this: beyond it the (row, col) -> ray
# mapping we hand NRE would be fiction, and supervising against fiction is
# worse than not supervising at all.
_MAX_FIT_RESIDUAL_DEG = 0.15


def _rot(rpy: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return (np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]))


class NuplanStructuredLidar:
    """Spinning model + per-ray timing for nuPlan's TOP lidar.

    Calibrated from a single sweep (28 k points over 40 rings is ample -- the
    0.022 deg residual quoted above comes from exactly one sweep), then applied
    to every later sweep.  ``decode`` takes the py123d ``Lidar`` modality the
    bridge already receives, so nothing re-reads the ``.pcd`` files.
    """

    sweep_us = _SWEEP_US

    def __init__(self, spin_sign: int = 1,
                 phase_origin_rad: float = _PHASE_ORIGIN_RAD,
                 deskew: bool = True):
        """Per-ray timing model.

        ``spin_sign``       +1 = azimuth increases with time (ccw), -1 = cw.
        ``phase_origin_rad`` azimuth at which a sweep starts.
        ``deskew``          False emits one stamp per sweep (the flattened
                            convention) while still registering the spinning
                            model, so supervision is unlocked without betting on
                            the timing model.

        Defaults are the measured optimum, not a convention: see the module
        docstring.  Override with ``NUPLAN_LIDAR_SPIN_SIGN``,
        ``NUPLAN_LIDAR_PHASE_ORIGIN_DEG`` and ``NUPLAN_LIDAR_DESKEW=0``.
        """
        env = os.environ.get("NUPLAN_LIDAR_SPIN_SIGN")
        self.spin_sign = int(env) if env else int(spin_sign)
        if self.spin_sign not in (-1, 1):
            raise ValueError(f"spin_sign must be +/-1, got {self.spin_sign}")
        env = os.environ.get("NUPLAN_LIDAR_PHASE_ORIGIN_DEG")
        self.phase_origin = (np.radians(float(env)) if env
                             else float(phase_origin_rad))
        self.deskew = os.environ.get("NUPLAN_LIDAR_DESKEW", "1") != "0" and bool(deskew)

        self.extrinsic: Optional[np.ndarray] = None      # 4x4 lidar -> rig
        self.row_elevations: Optional[np.ndarray] = None  # (n_rows,) rad
        self.n_rows: Optional[int] = None
        self.n_columns: Optional[int] = None
        self.fit_residual_deg: Optional[float] = None
        self._R = None
        self._t = None
        self._ring_to_row = None

    # ------------------------------------------------------------------
    @staticmethod
    def _top_points(lidar_mod):
        """(xyz, ring, intensity) of the TOP sensor, or None if unavailable.

        Accepts either an in-memory py123d ``Lidar`` (arrow path) or the lazy
        ``ParsedLidar`` the raw-nuPlan reader yields, which carries only a path
        to the ``MergedPointCloud`` ``.pcd``.  ``ring`` and ``lidar_info`` survive
        py123d's loader as the CHANNEL and IDS features -- it is only
        ``NCoreBridge._load_lidar`` that drops them.
        """
        feats = getattr(lidar_mod, "point_cloud_features", None)
        xyz = getattr(lidar_mod, "xyz", None)
        if feats is None or xyz is None:
            ds_root = getattr(lidar_mod, "_dataset_root", None)
            rel = getattr(lidar_mod, "_relative_path", None)
            if ds_root is None or rel is None or not str(rel).endswith(".pcd"):
                return None
            try:
                from pathlib import Path
                from py123d.parser.nuplan.nuplan_sensor_io import (
                    load_nuplan_point_cloud_data_from_path)
                xyz, feats = load_nuplan_point_cloud_data_from_path(
                    Path(ds_root) / rel)
            except Exception as exc:  # noqa: BLE001 - fall back to flattened
                logger.warning("NuplanStructuredLidar: cannot read %s (%s)", rel, exc)
                return None
        try:
            from py123d.datatypes.sensors.lidar import LidarFeature
            k_ids = LidarFeature.IDS.serialize()
            k_chan = LidarFeature.CHANNEL.serialize()
            k_int = LidarFeature.INTENSITY.serialize()
        except Exception:  # noqa: BLE001 - py123d layout change
            return None
        if k_ids not in feats or k_chan not in feats:
            return None

        ids = np.asarray(feats[k_ids])
        m = ids == _TOP_LIDAR_ID
        if m.sum() < 1000:
            return None
        inten = feats.get(k_int)
        inten = (np.ones(int(m.sum()), np.float32) if inten is None
                 else np.asarray(inten)[m].astype(np.float32) / 255.0)
        return (np.asarray(xyz)[m].astype(np.float64),
                np.asarray(feats[k_chan])[m].astype(np.int32),
                inten.clip(0.0, 1.0))

    # ------------------------------------------------------------------
    def calibrate(self, lidar_mod) -> bool:
        """Fit the extrinsic and beam elevations from one sweep.

        Returns False when the sweep is unusable or the fit is too loose to
        trust, in which case the caller keeps the flattened path.
        """
        got = self._top_points(lidar_mod)
        if got is None:
            logger.warning("NuplanStructuredLidar: no TOP points in sweep")
            return False
        P, ring, _ = got

        # Subsample for the fit; 8 k points over 40 rings still gives ~200 per
        # ring, far more than the 6 unknowns need.
        if len(P) > 8000:
            sel = np.linspace(0, len(P) - 1, 8000).astype(int)
            Pf, rf = P[sel], ring[sel]
        else:
            Pf, rf = P, ring
        rings = [k for k in np.unique(rf) if (rf == k).sum() > 20]
        if len(rings) < 8:
            logger.warning("NuplanStructuredLidar: only %d usable rings", len(rings))
            return False

        # Five parameters, not six: YAW IS DELIBERATELY FIXED AT ZERO. Rotating
        # about the vertical axis changes no point's elevation, so the cost cannot
        # see yaw and the optimiser wanders (it returned 39, -189, -76 and -1452
        # deg on four consecutive clips of the same log). Geometry survives that --
        # the extrinsic rotates the rays back -- but azimuth does not, and azimuth
        # is what per-ray time is derived from, so a free yaw would hand each clip
        # a different random time offset of up to half a sweep. Fixing it at zero
        # costs nothing and makes the sensor frame reproducible: ego frame,
        # de-rolled and de-pitched, origin at the fitted optical centre.
        def cost(par):
            rpy = np.array([par[0], par[1], 0.0])
            v = (Pf - par[2:5]) @ _rot(rpy).T
            r = np.linalg.norm(v, axis=1)
            e = np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
            return float(np.mean([e[rf == k].std() for k in rings]))

        from scipy.optimize import minimize
        best = None
        # Mount height is the one parameter with a wide plausible range and a
        # cost surface that traps Nelder-Mead; seed a few and keep the best.
        for z0 in (0.0, 1.0, 1.8, 2.5):
            r = minimize(cost, np.array([0.0, 0.0, 0.0, 0.0, z0]),
                         method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-9,
                                  "maxiter": 6000, "maxfev": 6000})
            if best is None or r.fun < best.fun:
                best = r
        best_rpy = np.array([best.x[0], best.x[1], 0.0])

        resid_deg = float(np.degrees(best.fun))
        if not np.isfinite(resid_deg) or resid_deg > _MAX_FIT_RESIDUAL_DEG:
            logger.warning("NuplanStructuredLidar: beam fit residual %.3f deg "
                           "exceeds %.3f deg; refusing the structured path",
                           resid_deg, _MAX_FIT_RESIDUAL_DEG)
            return False

        self._R = _rot(best_rpy)
        self._t = np.asarray(best.x[2:5], dtype=np.float64)
        self.fit_residual_deg = resid_deg

        # Per-beam elevation and the azimuth sampling density, from the full sweep.
        v = (P - self._t) @ self._R.T
        r = np.linalg.norm(v, axis=1)
        el = np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
        az = np.arctan2(v[:, 1], v[:, 0])

        n_rows = int(ring.max()) + 1
        elev = np.zeros(n_rows, np.float64)
        for k in range(n_rows):
            m = ring == k
            elev[k] = np.median(el[m]) if m.any() else np.nan
        # Rings with no returns in this sweep: interpolate from their neighbours
        # so the model stays a monotone beam ladder.
        bad = ~np.isfinite(elev)
        if bad.any():
            idx = np.arange(n_rows)
            elev[bad] = np.interp(idx[bad], idx[~bad], elev[~bad])

        # NCore asserts row 0 is the highest beam ("Row elevation angles must be
        # sorted in descending order"); nuPlan numbers rings from the bottom up.
        # Keep an explicit ring->row permutation rather than assuming the ladder
        # is exactly reversed, so a sensor with shuffled ring ids still works.
        order = np.argsort(elev)[::-1]
        self._ring_to_row = np.empty(n_rows, np.int64)
        self._ring_to_row[order] = np.arange(n_rows)
        elev = elev[order]

        # Column count from the sensor's own azimuth resolution: over-resolving
        # only wastes model_element range, under-resolving collides returns.
        per_ring = np.median([np.sum(ring == k) for k in rings])
        n_columns = int(np.clip(round(per_ring), 256, 4096))

        self.row_elevations = elev.astype(np.float32)
        self.n_rows = n_rows
        self.n_columns = n_columns
        logger.info("NuplanStructuredLidar: fitted TOP beam model -- "
                    "%d rows x %d cols, residual %.4f deg, "
                    "extrinsic t=[%.3f, %.3f, %.3f] m, rpy=[%.2f, %.2f, %.2f] deg",
                    n_rows, n_columns, resid_deg, *self._t,
                    *np.degrees(best_rpy))
        # 4x4 lidar -> rig: the fitted (R, t) maps ego -> sensor, so invert.
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = self._R.T
        ext[:3, 3] = self._t
        self.extrinsic = ext
        return True

    # ------------------------------------------------------------------
    def column_azimuths(self, width: int) -> np.ndarray:
        """Azimuth of each column, following ``spin_sign`` (see __init__)."""
        ratios = (np.arange(width, dtype=np.float64) + 0.5) / width
        az = ratios * 2.0 * np.pi * self.spin_sign + self.phase_origin
        return ((az + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)

    def spinning_params(self):
        """NCore spinning model for ``store_lidar_intrinsics``."""
        from ncore.impl.data.types import (
            RowOffsetStructuredSpinningLidarModelParameters,
        )
        if self.n_columns is None:
            raise RuntimeError("calibrate() before spinning_params()")
        return RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=1e6 / _SWEEP_US,
            spinning_direction="ccw" if self.spin_sign > 0 else "cw",
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations,
            column_azimuths_rad=self.column_azimuths(self.n_columns),
            # nuPlan gives no per-beam firing offsets and they are not
            # recoverable without per-point time; zero is the honest default.
            row_azimuth_offsets_rad=np.zeros(self.n_rows, np.float32),
        )

    # ------------------------------------------------------------------
    def decode(self, lidar_mod, frame_ts_us: int) -> Optional[dict]:
        """Structured record for one sweep, keys matching WodStructuredLidar."""
        if self._R is None:
            return None
        got = self._top_points(lidar_mod)
        if got is None:
            return None
        P, ring, inten = got

        v = (P - self._t) @ self._R.T
        dist = np.linalg.norm(v, axis=1)
        ok = dist > 1e-3
        v, dist, ring, inten = v[ok], dist[ok], ring[ok], inten[ok]
        direction = (v / dist[:, None]).astype(np.float32)

        az = np.arctan2(v[:, 1], v[:, 0])
        # phase in [0, 1) along the sweep, in the sensor's spin direction
        phase = (((self.spin_sign * az - self.phase_origin)
                  / (2.0 * np.pi)) % 1.0)
        col = np.clip((phase * self.n_columns).astype(np.int64),
                      0, self.n_columns - 1)
        row = self._ring_to_row[np.clip(ring, 0, self.n_rows - 1)]

        ray_ts = (np.uint64(frame_ts_us)
                  + ((phase * _SWEEP_US).astype(np.uint64) if self.deskew
                     else np.zeros(len(phase), np.uint64)))
        return {
            "model_element": np.column_stack([row, col]).astype(np.uint16),
            "direction": direction,
            "distance_m": dist.astype(np.float32),
            "intensity": inten.astype(np.float32),
            "timestamp_us": ray_ts,
            "frame_timestamp_us": int(frame_ts_us),
            "frame_end_timestamp_us": int(frame_ts_us) + _SWEEP_US - 1,
        }
