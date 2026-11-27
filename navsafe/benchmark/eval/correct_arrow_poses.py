# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Move a 20 s bundle's logged poses into its reconstruction's world.

The NavSafe-280 reconstructions were trained on refined camera poses: for every
frame, COLMAP replaced the logged ego attitude (z, roll, pitch, yaw; x and y
kept) and every box of that frame was moved with the same rigid correction::

    W(t) = R_u(t) R_l(t)^T,     c' = p_u(t) + W(t) (c - p_l(t)),     yaw' = yaw(W(t) R)

with ``p_l, R_l`` the logged ego pose (``ego_state_se3.imu_se3``), ``p_u, R_u``
the training rig pose (the usdz's ``rig_trajectories.json``) and ``c, R`` a
logged box. The logged attitude is off by 0.1-2 deg per frame, and every logged
box inherits that error: 50 m ahead a 1 deg pitch error puts a car 0.9 m into
the road, and a parked car's logged box jumps by a median 1.0 m in z across
the frames of one clip. The reconstruction, which matches the real images,
holds the corrected world.

Before this tool the bundles shipped the logged poses, so the simulator lived
in the logged world while the renderer drew the corrected one: the camera sat
up to 0.65 m off its training height and actors were sent up to 2.5 m off
their training boxes (sunk into the road). This tool rewrites
``ego_state_se3.arrow`` and ``box_detections_se3.arrow`` so that both share the
reconstruction's world:

* ego: only z, from the training rig at each ego timestamp. x, y stay logged
  (the training rig keeps them), so routes, recipe frames (``ego_frame0`` is a
  translation) and the map are unchanged. The orientation also stays logged:
  the logged heading follows the direction of travel (median |travel -
  heading| 0.08 deg over the 280), while the training rig yaw sits a constant
  ~1-2 deg off it in some clips -- a sensor mount offset, not a body heading
  error. The renderer applies the rig's own yaw and tilt to the camera;
* boxes: the formula above, with the ego log row and the training rig sample
  nearest to the box timestamp; the orientation stays yaw-only, as in the
  training boxes; ``velocity_3d`` is rotated by ``W``. The boxes do carry the
  mount offset (they come from the LiDAR on the same rig): with it removed, a
  parked car's position spread across frames drops, e.g. 0.86 -> 0.18 m;
* ``dynamic_state_se3`` (body-frame velocity, acceleration, yaw rate),
  ``sync``, ``traffic_light_detections`` and ``custom.scenario`` are not
  touched.

On all 280 bundles the formula reproduces the usdz training boxes to 1.7 cm
in z and 8 cm in xy (1.38 M boxes). Frames without a training rig pose within
60 ms keep their logged values and are counted in the returned statistics.

Usage::

    python -m navsafe.benchmark.eval.correct_arrow_poses \\
        --src-root <full_test> --dst-root <new_full_test> [TOKEN ...]

The destination bundle hard-links every other file of the source bundle and
writes a ``manifest.json`` with ``arrow.poses`` set to ``"reconstruction"``.
The published bundles are already converted; the tool documents how, and
converts bundles built with :mod:`navsafe.benchmark.eval.bundle`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from scipy.spatial.transform import Rotation, Slerp

#: Largest gap (us) between a timestamp and the nearest training rig pose.
MAX_GAP_US = 60_000
EGO_FILE = "ego_state_se3.arrow"
BOX_FILE = "box_detections_se3.arrow"
POSES_TAG = "reconstruction"


def _read_table(path: Path) -> pa.Table:
    with pa.memory_map(str(path)) as src:
        return ipc.open_file(src).read_all()


def _write_table(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pa.OSFile(str(tmp), "wb") as sink, ipc.new_file(sink, table.schema) as w:
        w.write_table(table)
    os.replace(tmp, path)


class _Track:
    """A pose track interpolated in time: positions linearly, rotations by slerp."""

    def __init__(self, ts_us: np.ndarray, pos: np.ndarray, rot: Rotation):
        o = np.argsort(ts_us)
        self.ts = np.asarray(ts_us, np.int64)[o]
        self.pos = np.asarray(pos, np.float64)[o]
        self.rot = rot[o]
        self._slerp = Slerp(self.ts.astype(np.float64), self.rot)

    def at(self, t_us: np.ndarray) -> Tuple[np.ndarray, Rotation, np.ndarray]:
        """(positions, rotations, valid) at ``t_us``; invalid ones are clamped."""
        t = np.asarray(t_us, np.int64)
        k = np.clip(np.searchsorted(self.ts, t), 1, len(self.ts) - 1)
        gap = np.minimum(np.abs(self.ts[k] - t), np.abs(self.ts[k - 1] - t))
        valid = gap <= MAX_GAP_US
        tc = np.clip(t, self.ts[0], self.ts[-1]).astype(np.float64)
        pos = np.stack([np.interp(tc, self.ts, self.pos[:, i]) for i in range(3)], 1)
        return pos, self._slerp(tc), valid

    def nearest(self, t_us: np.ndarray) -> Tuple[np.ndarray, Rotation, np.ndarray]:
        """(positions, rotations, valid) of the nearest sample to each of ``t_us``."""
        t = np.asarray(t_us, np.int64)
        k = np.clip(np.searchsorted(self.ts, t), 1, len(self.ts) - 1)
        k = np.where(np.abs(self.ts[k - 1] - t) <= np.abs(self.ts[k] - t), k - 1, k)
        return self.pos[k], self.rot[k], np.abs(self.ts[k] - t) <= MAX_GAP_US


def _training_rig(usdz: Path) -> _Track:
    with zipfile.ZipFile(usdz) as z:
        rig = json.loads(z.read("rig_trajectories.json"))
    rt = rig["rig_trajectories"]
    if len(rt) != 1:
        raise ValueError(f"{usdz.name}: {len(rt)} rig trajectories, expected 1")
    T = np.asarray(rt[0]["T_rig_worlds"], np.float64)
    return _Track(np.asarray(rt[0]["T_rig_world_timestamps_us"], np.int64), T[:, :3, 3],
                  Rotation.from_matrix(T[:, :3, :3]))


def _wxyz(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def _to_wxyz(r: Rotation) -> np.ndarray:
    return r.as_quat()[..., [3, 0, 1, 2]]


def _yaw_only(r: Rotation) -> Rotation:
    m = r.as_matrix()
    return Rotation.from_euler("z", np.arctan2(m[..., 1, 0], m[..., 0, 0])[..., None])


def _replace_fsl(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    """Replace a fixed_size_list<double> column (one row per table row)."""
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    col = table.column(idx).combine_chunks()
    if col.null_count:
        raise ValueError(f"{name}: unexpected nulls")
    width = field.type.list_size
    arr = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), pa.float64()), width)
    return table.set_column(idx, field, arr.cast(field.type))


def _replace_list_fsl(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    """Replace a list<fixed_size_list<double>> column, keeping its row offsets."""
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    col = table.column(idx).combine_chunks()
    if col.null_count or col.values.null_count:
        raise ValueError(f"{name}: unexpected nulls")
    width = field.type.value_type.list_size
    inner = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), pa.float64()), width)
    arr = pa.ListArray.from_arrays(col.offsets, inner)
    return table.set_column(idx, field, arr.cast(field.type))


def _flat_list_fsl(table: pa.Table, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """(values (M, width), row index of each value) of a list<fixed_size_list> column."""
    col = table.column(name).combine_chunks()
    width = col.type.value_type.list_size
    off = col.offsets.to_numpy()
    start, stop = int(off[0]), int(off[-1])
    flat = col.values.values.to_numpy(zero_copy_only=False)[start * width:stop * width]
    rows = np.repeat(np.arange(len(off) - 1), np.diff(off))
    return flat.reshape(-1, width).astype(np.float64), rows


def correct_log_dir(log_dir: Path, usdz: Path, out_dir: Path) -> Dict[str, float]:
    """Write corrected ego / box arrow files of one log directory into ``out_dir``."""
    rig = _training_rig(usdz)
    ego = _read_table(log_dir / EGO_FILE)
    te = np.asarray(ego.column("ego_state_se3.timestamp_us").to_numpy(), np.int64)
    imu = np.asarray(ego.column("ego_state_se3.imu_se3").combine_chunks().values.to_numpy(), np.float64).reshape(-1, 7)
    log = _Track(te, imu[:, :3], _wxyz(imu[:, 3:7]))

    # usdz world = arrow world + o. r6 keeps the logged x, y, so o_xy is one
    # constant (the NRE centring); its spread is the xy noise of the training rig.
    pu_e, Ru_e, ok_e = rig.at(te)
    if not ok_e.any():
        raise ValueError(f"{usdz.name}: no training rig pose within {MAX_GAP_US} us of the ego log")
    d = pu_e[ok_e, :2] - imu[ok_e, :2]
    o_xy = np.median(d, axis=0)
    stats = {"ego_frames": int(len(te)), "ego_frames_kept_logged": int((~ok_e).sum()),
             "o_xy_spread_m": float(np.max(np.linalg.norm(d - o_xy, axis=1)))}

    # --- ego: training z; x, y and orientation logged (see the module docstring)
    new_imu = imu.copy()
    new_imu[ok_e, 2] = pu_e[ok_e, 2]
    dz_e = new_imu[:, 2] - imu[:, 2]
    dyaw_e = np.degrees((_yaw_only(Ru_e).as_euler("zyx")[:, 0] - _yaw_only(log.rot).as_euler("zyx")[:, 0]
                         + np.pi) % (2 * np.pi) - np.pi)
    stats.update(ego_dz_absmax_m=float(np.abs(dz_e).max()), ego_dz_median_m=float(np.median(dz_e)),
                 rig_minus_log_yaw_absmax_deg=float(np.abs(dyaw_e[ok_e]).max()))
    ego = _replace_fsl(ego, "ego_state_se3.imu_se3", new_imu)

    # --- boxes: c' = p_u + W (c - p_l), yaw' = yaw(W R), v' = W v, with the ego log row and the
    # training rig sample nearest to the box time -- what the reconstruction's box correction used
    # (interpolating both instead leaves ~1 cm against the training boxes; nearest leaves < 1 mm).
    box = _read_table(log_dir / BOX_FILE)
    tb = np.asarray(box.column("box_detections_se3.timestamp_us").to_numpy(), np.int64)
    bb, rows = _flat_list_fsl(box, "box_detections_se3.bounding_box_se3")
    vel, vrows = _flat_list_fsl(box, "box_detections_se3.velocity_3d")
    if not np.array_equal(rows, vrows):
        raise ValueError("box and velocity rows differ")
    pl, Rl, ok_l = log.nearest(tb)
    pu, Ru, ok_u = rig.nearest(tb)
    pu = pu - np.array([o_xy[0], o_xy[1], 0.0])      # training rig in the arrow world
    ok = ok_l & ok_u
    W = (Ru * Rl.inv()).as_matrix()                    # (frames, 3, 3)
    sel = ok[rows]
    r = rows[sel]
    c = bb[sel, :3]
    c_new = pu[r] + np.einsum("nij,nj->ni", W[r], c - pl[r])
    R_new = _yaw_only(Rotation.from_matrix(W[r]) * _wxyz(bb[sel, 3:7]))
    new_bb = bb.copy()
    new_bb[sel, :3] = c_new
    new_bb[sel, 3:7] = _to_wxyz(R_new)
    new_vel = vel.copy()
    new_vel[sel] = np.einsum("nij,nj->ni", W[r], vel[sel])
    box = _replace_list_fsl(box, "box_detections_se3.bounding_box_se3", new_bb)
    box = _replace_list_fsl(box, "box_detections_se3.velocity_3d", new_vel)
    dzb = c_new[:, 2] - c[:, 2] if len(c) else np.zeros(1)
    dxyb = np.linalg.norm(c_new[:, :2] - c[:, :2], axis=1) if len(c) else np.zeros(1)
    stats.update(box_frames=int(len(tb)), box_frames_kept_logged=int((~ok).sum()), boxes=int(len(bb)),
                 boxes_moved=int(sel.sum()), box_dz_absmax_m=float(np.abs(dzb).max()),
                 box_dz_p95_m=float(np.percentile(np.abs(dzb), 95)), box_dxy_p95_m=float(np.percentile(dxyb, 95)))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_table(ego, out_dir / EGO_FILE)
    _write_table(box, out_dir / BOX_FILE)
    return stats


def _link_tree(src: Path, dst: Path, skip: set) -> None:
    """Hard-link every file of ``src`` into ``dst`` (copy across devices), except ``skip``."""
    for root, _dirs, files in os.walk(src, followlinks=True):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s, t = Path(root) / f, dst / rel / f
            if (rel / f).as_posix() in skip or t.exists():
                continue
            try:
                os.link(s.resolve(), t)
            except OSError:
                shutil.copy2(s, t)


def correct_bundle(src: Path, dst: Path) -> Dict[str, float]:
    """Write a corrected copy of bundle ``src`` to ``dst``; returns the statistics."""
    man = json.loads((src / "manifest.json").read_text())
    if man.get("arrow", {}).get("poses") == POSES_TAG:
        raise ValueError(f"{src.name}: arrow poses are already corrected")
    if len(man["subclips"]) != 1:
        raise ValueError(f"{src.name}: {len(man['subclips'])} sub-clips; expected one 20 s model")
    usdz = src / man["subclips"][0]["usdz"]
    logs = sorted(p for p in (src / "arrow" / "logs").glob("*/*") if (p / EGO_FILE).exists())
    if len(logs) != 1:
        raise ValueError(f"{src.name}: {len(logs)} arrow log directories, expected 1")
    rel_log = logs[0].relative_to(src)
    skip = {(rel_log / EGO_FILE).as_posix(), (rel_log / BOX_FILE).as_posix(), "manifest.json"}
    if dst.exists():
        raise ValueError(f"{dst} exists; remove it first")
    try:
        stats = correct_log_dir(logs[0], usdz, dst / rel_log)
        _link_tree(src, dst, skip)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    man["arrow"]["poses"] = POSES_TAG
    man["arrow"]["poses_note"] = (
        "ego z = the reconstruction's training rig (x, y and orientation logged); boxes moved with "
        "the per-frame rig correction the reconstruction was trained with "
        "(navsafe.benchmark.eval.correct_arrow_poses)")
    (dst / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--dst-root", type=Path, required=True)
    ap.add_argument("--stats", type=Path, help="write per-bundle statistics (JSON lines) here")
    ap.add_argument("tokens", nargs="*", help="bundles to convert (default: all under --src-root)")
    a = ap.parse_args()
    toks = a.tokens or sorted(p.name for p in a.src_root.iterdir() if (p / "manifest.json").exists())
    bad = 0
    for tok in toks:
        try:
            st = {"token": tok, **correct_bundle(a.src_root / tok, a.dst_root / tok)}
        except Exception as e:  # noqa: BLE001 -- report every bundle, fail at the end
            st = {"token": tok, "error": repr(e)}
            bad += 1
        print(json.dumps(st), flush=True)
        if a.stats:
            with open(a.stats, "a") as f:
                f.write(json.dumps(st) + "\n")
    if bad:
        raise SystemExit(f"{bad} of {len(toks)} bundles failed")


if __name__ == "__main__":
    main()
