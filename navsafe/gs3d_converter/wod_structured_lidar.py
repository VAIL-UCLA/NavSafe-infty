# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Structured (range-image) WOD TOP-lidar decoding for the NCore bridge.

Why
---
NRE's lidar-supervision dataloader (``get_lidar_data_batch``) requires a
structured spinning-lidar model (``n_rows x n_columns`` beam grid registered
via ``IntrinsicsComponent.store_lidar_intrinsics``) plus a per-ray
``model_element=(row, col)``. The py123d Arrow path flattens the WOD range
image into an unstructured ego-frame cloud, losing exactly that structure —
which is why ``nurec_runner`` historically trained camera-only
(``dataset.n_train_sample_lidar_rays=0``) and the reconstructed road's z
drifts by decimetres (see ``docs``/memory: ground-z саga).

This module re-derives the structure straight from the WOD ``.tfrecord``
(pure Python: protobuf + zlib — **no tensorflow**), mirroring the math of
Waymo's ``range_image_utils.compute_range_image_polar``:

* per-column azimuth ``(2*ratios - 1)*pi - az_correction`` with
  ``ratios = (arange(W, 0, -1) - 0.5) / W`` and
  ``az_correction = atan2(extrinsic[1, 0], extrinsic[0, 0])`` — these are
  SENSOR-frame azimuths (Waymo folds the mount yaw into the polar grid and
  then rotates by the full extrinsic; undoing the extrinsic leaves exactly
  this grid), so they pair with the real TOP-lidar->rig extrinsic the bridge
  registers;
* ``row_elevations_rad`` = reversed ``beam_inclinations`` (WOD stores them
  ascending; range-image row 0 is the top beam, and NCore wants descending);
* azimuth decreases with column → ``spinning_direction="cw"``.

Directions are evaluated from the model grid itself (the NCore template's
"polar range image" case), so stored rays and the registered model agree by
construction.
"""

from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_TOP_SWEEP_US = 100_000  # 10 Hz spin


def _import_wod_protos():
    """Import the py123d-vendored Waymo protos without tensorflow.

    ``dataset_pb2`` has a proto dependency chain (vector -> keypoint ->
    label -> ...); a fixpoint loop sidesteps caring about the exact order.
    """
    import importlib
    import pkgutil

    import py123d.parser.wod.waymo_open_dataset.protos as protos_pkg

    mods = [m.name for m in pkgutil.iter_modules(protos_pkg.__path__)
            if m.name.endswith("_pb2")]
    loaded: set = set()
    for _ in range(len(mods) + 2):
        for m in mods:
            if m in loaded:
                continue
            try:
                importlib.import_module(
                    f"py123d.parser.wod.waymo_open_dataset.protos.{m}")
                loaded.add(m)
            except Exception:  # noqa: BLE001 — dependency not loaded yet
                pass
        if len(loaded) == len(mods):
            break
    from py123d.parser.wod.waymo_open_dataset.protos import dataset_pb2
    return dataset_pb2


def _iter_tfrecord(path: Path):
    """Yield raw records from a TFRecord file (pure Python framing)."""
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            f.seek(4, 1)  # length CRC
            payload = f.read(length)
            if len(payload) < length:
                return
            f.seek(4, 1)  # data CRC
            yield payload


class WodStructuredLidar:
    """Random access to structured TOP-lidar sweeps of one WOD segment.

    Frames are indexed by ``Frame.timestamp_micros`` — the same stamps the
    py123d Arrow conversion carries, so the bridge can pair its per-frame
    loop with the tfrecord exactly.
    """

    sweep_us = _TOP_SWEEP_US

    def __init__(self, tfrecord_path: str | Path):
        self._pb = _import_wod_protos()
        self.path = Path(tfrecord_path)
        self._frames: Dict[int, bytes] = {}
        self._calib = None
        for rec in _iter_tfrecord(self.path):
            frame = self._pb.Frame()
            frame.ParseFromString(rec)
            self._frames[int(frame.timestamp_micros)] = rec
            if self._calib is None:
                for c in frame.context.laser_calibrations:
                    if c.name == self._pb.LaserName.TOP:
                        self._calib = c
        if self._calib is None:
            raise ValueError(f"no TOP laser calibration in {self.path}")

        ext = np.asarray(self._calib.extrinsic.transform,
                         np.float64).reshape(4, 4)
        incl = np.asarray(self._calib.beam_inclinations, np.float64)
        if incl.size == 0:  # non-TOP lasers only carry min/max; TOP has the list
            raise ValueError("TOP calibration lacks beam_inclinations")
        self.extrinsic = ext
        # WOD stores inclinations ascending; range-image row 0 = highest beam.
        self.row_elevations = incl[::-1].astype(np.float32).copy()
        self.n_rows = int(incl.size)
        self.n_columns: Optional[int] = None  # filled on first decode
        self._az_correction = float(np.arctan2(ext[1, 0], ext[0, 0]))

    # ------------------------------------------------------------------
    def column_azimuths(self, width: int) -> np.ndarray:
        ratios = (np.arange(width, 0, -1, dtype=np.float64) - 0.5) / width
        return ((ratios * 2.0 - 1.0) * np.pi - self._az_correction).astype(np.float32)

    def spinning_params(self):
        """NCore spinning-lidar model for ``store_lidar_intrinsics``.

        Requires at least one :meth:`decode` call (to learn ``n_columns``).
        """
        from ncore.impl.data.types import (
            RowOffsetStructuredSpinningLidarModelParameters,
        )
        if self.n_columns is None:
            raise RuntimeError("decode one frame before spinning_params()")
        return RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=1e6 / _TOP_SWEEP_US,
            # Azimuth strictly decreases with column index (see
            # column_azimuths) = clockwise viewed from +Z.
            spinning_direction="cw",
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations,
            column_azimuths_rad=self.column_azimuths(self.n_columns),
            row_azimuth_offsets_rad=np.zeros(self.n_rows, np.float32),
        )

    # ------------------------------------------------------------------
    def decode(self, timestamp_us: int, tol_us: int = 1000) -> Optional[dict]:
        """Decode the TOP first-return sweep nearest ``timestamp_us``.

        Returns ``None`` when no tfrecord frame lies within ``tol_us``.
        Result keys: ``model_element`` (N,2 u16), ``direction`` (N,3 f32,
        sensor frame, unit), ``distance_m`` (N,), ``intensity`` (N,, [0,1]),
        ``timestamp_us`` (N, u64 absolute per-ray).
        """
        stamps = np.fromiter(self._frames.keys(), np.int64)
        i = int(np.abs(stamps - int(timestamp_us)).argmin())
        ts = int(stamps[i])
        if abs(ts - int(timestamp_us)) > tol_us:
            return None

        frame = self._pb.Frame()
        frame.ParseFromString(self._frames[ts])
        top = next((l for l in frame.lasers if l.name == self._pb.LaserName.TOP), None)
        if top is None or not top.ri_return1.range_image_compressed:
            return None
        mf = self._pb.MatrixFloat()
        mf.ParseFromString(zlib.decompress(top.ri_return1.range_image_compressed))
        h, w = mf.shape.dims[0], mf.shape.dims[1]
        ri = np.asarray(mf.data, np.float32).reshape(h, w, mf.shape.dims[2])
        if self.n_columns is None:
            self.n_columns = int(w)
        assert h == self.n_rows and w == self.n_columns, "range image shape drift"

        rng = ri[..., 0]
        rows, cols = np.nonzero(rng > 0)
        dist = rng[rows, cols].astype(np.float32)
        # WOD intensity is an unbounded raw value; tanh is the conventional
        # squash (used by Waymo's own visualizers) onto [0, 1).
        intensity = np.tanh(np.clip(ri[rows, cols, 1], 0.0, None)).astype(np.float32)

        az = self.column_azimuths(w)[cols].astype(np.float64)
        el = self.row_elevations[rows].astype(np.float64)
        cos_el = np.cos(el)
        direction = np.stack([cos_el * np.cos(az), cos_el * np.sin(az),
                              np.sin(el)], axis=1).astype(np.float32)

        # Per-ray time: linear in column along the sweep (template convention).
        col_frac = cols.astype(np.float64) / max(w - 1, 1)
        ray_ts = (ts + col_frac * _TOP_SWEEP_US).astype(np.uint64)

        return {
            "model_element": np.column_stack([rows, cols]).astype(np.uint16),
            "direction": direction,
            "distance_m": dist,
            "intensity": intensity,
            "timestamp_us": ray_ts,
            "frame_timestamp_us": ts,
        }


def find_wod_tfrecord(clip_id: str) -> Optional[Path]:
    """Resolve the segment tfrecord for ``clip_id`` from env-configured dirs.

    ``R2S_WOD_TFRECORD`` — explicit file path, or ``R2S_WOD_TFRECORD_DIR`` —
    directory searched for ``segment-<clip>*.tfrecord``. Returns None when
    unset/missing so the bridge can fall back to the flattened path.
    """
    import os

    explicit = os.environ.get("R2S_WOD_TFRECORD")
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    root = os.environ.get("R2S_WOD_TFRECORD_DIR")
    if not root:
        return None
    hits = sorted(Path(root).glob(f"**/segment-{clip_id}*.tfrecord"))
    return hits[0] if hits else None
