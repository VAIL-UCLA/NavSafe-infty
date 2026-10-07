# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a downloadable, self-contained 20 s NavSafe eval scenario.

Input is what the `NavSafe` dataset publishes under `intermediate_data/`: four 5 s NuRec artifacts
per scenario (`<token>s1..s4.usdz`). Output is one directory per scenario that
a downloader can serve and evaluate without touching the reconstruction
pipeline, the pod, or any sidecar file that lives only on `/data`.

**There is no single fused usdz, and there should not be.** Each artifact is a
separately *trained* NuRec model — its 2 GB is `checkpoint.ckpt`, not geometry
(`export-usdz-artifact` output, which `serve-grpc` loads and which renders
nothing in IsaacSim; see `nurec_runner.run_export`). Four models cannot be
merged into one model. A "20 s scenario" is therefore four models plus a
manifest saying which one owns which second and where it sits — exactly what
`NUREC_GRPC_HANDOFF` (`navsafe/render/nurec_grpc.py`) already consumes. This
tool writes that manifest; single-file delivery, if wanted, is a tar of the
directory, not a merge.

**Where the numbers come from.** Both halves of the handoff are recovered from
data anyone can obtain, so a bundle can be rebuilt from scratch:

* *Windows* — each usdz carries its own `data_info.json`
  (`sequence_timestamp_interval_us`) and `customLayerData.absoluteTimeOffsetMicroSec`.
  Read from the zip directory, so the 2 GB checkpoint is never unpacked.
* *Offsets* — **not** `rig_trajectories.json`'s `world_to_nre`, which is an
  internal NRE centering (measured: it moves 5.7 m between two sub-clips whose
  ego moved 30.7 m). The reference frame of each sub-clip is its own frame-0
  ego pose, which the source nuPlan log holds: the rig's frame-0 z
  (16.560667) matches the log's ego z at the window start (16.561) to the
  millimetre. So `offset_xy_utm` = the log's ego (x, y) at the sub-clip's
  start timestamp — the same quantity the retired `nurec_origin_offset.json`
  sidecars carried.

Usage::

    navsafe bundle \\
        --usdz-dir   "$NAVSAFE_USDZ" \\
        --nuplan-splits "$NUPLAN_ROOT/nuplan-v1.1/splits/test" \\
        --jobs       navsafe/benchmark/eval/stitch_jobs.tsv \\
        --out        "$NAVSAFE_BUNDLES" \\
        [--token 00c1e4eb4a045f20] [--verify-only]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from navsafe.benchmark.eval.scenario_taxonomy import ScenarioMeta, load_scenario_meta

# A sub-clip's frame-0 pose is looked up in the log by timestamp; nuPlan ego
# poses are ~20 Hz, so the nearest one is within ~25 ms. Anything beyond this
# means the window and the log do not belong together.
MAX_POSE_SNAP_US = 60_000
SUBCLIPS = ("s1", "s2", "s3", "s4")

# Printed into every bundle README. Deliberately stdlib-only and independent of
# NavSafe: a downloader has the bundle long before they have the repo, and
# importing `navsafe` bootstraps Omniverse (which writes an EULA prompt to
# stdout and would corrupt the captured value).
HANDOFF_SNIPPET = '''```bash
BUNDLE=$(pwd)
export NUREC_GRPC_HANDOFF="$(python3 - "$BUNDLE" <<'PY'
import json, sys
b = sys.argv[1]
m = json.load(open(b + "/manifest.json"))
print(";".join("%s,%s/offsets/%s.json,%d,%d"
               % (c["scene_id"], b, c["scene_id"],
                  c["t_start_us"], c["t_stop_us"])
               for c in m["subclips"]))
PY
)"
```'''


@dataclass
class SubClip:
    """One 5 s reconstruction: which model, when it applies, where it sits."""

    scene_id: str            # grpc scene id == the usdz's own metadata scene_id
    usdz: str                # path, relative to the bundle directory
    t_start_us: int
    t_stop_us: int
    offset_xy_utm: tuple[float, float]
    ego_z_m: float
    pose_snap_us: int        # how far the matched log pose was from the window

    def to_dict(self) -> dict:
        d = asdict(self)
        d["offset_xy_utm"] = list(self.offset_xy_utm)
        return d


def read_usdz_meta(path: Path) -> dict:
    """`scene_id` and the absolute time window, from the zip's small entries.

    Deliberately does not open `checkpoint.ckpt`: the metadata is a few KB and
    the checkpoint is 2 GB, so a whole dataset can be indexed in seconds.
    """
    with zipfile.ZipFile(path) as z:
        info = json.loads(z.read("data_info.json"))
        window = info["sequence_timestamp_interval_us"]
        scene_id = str(info.get("sequence_id") or "")
        if not scene_id:
            import re
            meta = z.read("metadata.yaml").decode()
            m = re.search(r"^scene_id:\s*(\S+)", meta, re.M)
            scene_id = m.group(1) if m else path.stem
    return {"scene_id": scene_id,
            "t_start_us": int(window["start"]),
            "t_stop_us": int(window["stop"])}


def ego_pose_at(db: Path, ts_us: int) -> tuple[float, float, float, int]:
    """(x, y, z, snap_us) of the nearest logged ego pose to ``ts_us``."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        row = con.execute(
            "SELECT timestamp, x, y, z FROM ego_pose "
            "ORDER BY ABS(timestamp - ?) LIMIT 1", (ts_us,)).fetchone()
    if row is None:
        raise LookupError(f"{db.name}: no ego_pose rows")
    t, x, y, z = row
    return float(x), float(y), float(z), int(t) - int(ts_us)


def build_scenario(token: str, log_name: str, usdz_dir: Path,
                   splits_dir: Path) -> list[SubClip]:
    """The four sub-clips of one scenario, ordered s1..s4."""
    db = splits_dir / f"{log_name}.db"
    if not db.exists():
        raise FileNotFoundError(f"{token}: source log not found: {db}")

    out: list[SubClip] = []
    for s in SUBCLIPS:
        usdz = usdz_dir / token / f"{token}{s}.usdz"
        if not usdz.exists():
            raise FileNotFoundError(f"{token}: missing sub-clip {usdz.name}")
        meta = read_usdz_meta(usdz)
        x, y, z, snap = ego_pose_at(db, meta["t_start_us"])
        if abs(snap) > MAX_POSE_SNAP_US:
            raise ValueError(
                f"{meta['scene_id']}: nearest ego pose is {snap} us from the "
                f"window start — the usdz and log {log_name} do not match")
        out.append(SubClip(scene_id=meta["scene_id"], usdz=usdz.name,
                           t_start_us=meta["t_start_us"],
                           t_stop_us=meta["t_stop_us"],
                           offset_xy_utm=(x, y), ego_z_m=z, pose_snap_us=snap))
    out.sort(key=lambda c: c.t_start_us)
    return out


def check_contiguous(clips: list[SubClip]) -> list[str]:
    """Gaps/overlaps between consecutive windows, as human-readable warnings.

    A one-frame (~100 ms) gap is what the recon windows normally have; anything
    larger is a hole in the 20 s the handoff cannot cover, and it is reported
    rather than silently stitched over.
    """
    warns = []
    for a, b in zip(clips, clips[1:]):
        gap = b.t_start_us - a.t_stop_us
        if gap < 0:
            warns.append(f"{a.scene_id} -> {b.scene_id}: windows OVERLAP by "
                         f"{-gap/1e6:.3f} s")
        elif gap > 150_000:
            warns.append(f"{a.scene_id} -> {b.scene_id}: {gap/1e6:.3f} s gap "
                         "is unrendered by any sub-clip")
    return warns


def handoff_string(clips: list[SubClip], bundle: Path) -> str:
    """The `NUREC_GRPC_HANDOFF` value: `sid,offset_json,t0,t1;...`.

    ``bundle`` must be the reader's own path to the bundle: the renderer
    ``open()``s the offsets JSON directly (``nurec_grpc.py``), so the value is
    only valid on the machine that built it. That is why no ``handoff.txt`` is
    shipped inside a bundle any more — a baked absolute path survives being
    published and then fails inside renderer init on the downloader's box,
    which reads as a reconstruction problem rather than a stale string. Derive
    it at use time with :func:`handoff_for_bundle` instead.
    """
    return ";".join(
        f"{c.scene_id},{bundle / 'offsets' / (c.scene_id + '.json')},"
        f"{c.t_start_us},{c.t_stop_us}"
        for c in clips)


def handoff_for_bundle(bundle: Path) -> str:
    """The `NUREC_GRPC_HANDOFF` value for an on-disk bundle, from its manifest.

    Everything the handoff needs is already recorded in ``manifest.json``; only
    the absolute prefix is machine-local, and it is supplied here by the
    bundle's own location. Exposed on the command line as ``--handoff``.
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    return ";".join(
        f"{c['scene_id']},{bundle / 'offsets' / (c['scene_id'] + '.json')},"
        f"{c['t_start_us']},{c['t_stop_us']}"
        for c in manifest["subclips"])


def write_bundle(token: str, log_name: str, clips: list[SubClip],
                 usdz_dir: Path, out_root: Path, *, link: bool = True,
                 meta: Optional[ScenarioMeta] = None) -> Path:
    """Write one bundle directory; usdz are hard-linked when possible."""
    bundle = out_root / token
    (bundle / "offsets").mkdir(parents=True, exist_ok=True)

    for c in clips:
        src = usdz_dir / token / c.usdz
        dst = bundle / c.usdz
        if not dst.exists():
            try:
                if link:
                    dst.hardlink_to(src)
                else:
                    raise OSError
            except OSError:                       # different filesystem
                dst.symlink_to(src.resolve())
        # The sidecar the handoff reader expects, regenerated from the log so
        # the bundle does not depend on the reconstruction workspace.
        (bundle / "offsets" / f"{c.scene_id}.json").write_text(json.dumps(
            {"offset_xy_utm": list(c.offset_xy_utm), "ego_z_m": c.ego_z_m,
             "source": "nuplan ego_pose at the sub-clip window start",
             "log": log_name, "pose_snap_us": c.pose_snap_us}, indent=2))

    arrow = _arrow_summary(bundle / "arrow", token,
                           clips[0].t_start_us, clips[-1].t_stop_us)
    manifest = {
        "token": token,
        "log": log_name,
        # nuPlan scenario label, NavSafe event type, and whether this
        # bundle contains any synthetic/inserted actors — the event type lookup is
        # `eval/scenario_taxonomy.py`. None if stitch_jobs.tsv carried no row
        # for this token, so a missing lookup is visible rather than silently
        # absent.
        "scenario_meta": meta.to_dict() if meta else None,
        "t_start_us": clips[0].t_start_us,
        "t_stop_us": clips[-1].t_stop_us,
        "duration_s": round((clips[-1].t_stop_us - clips[0].t_start_us) / 1e6, 3),
        "render_backend": "nurec_grpc",
        "subclips": [c.to_dict() for c in clips],
        "warnings": check_contiguous(clips) + _arrow_warnings(arrow),
        # Arrow is built separately (it needs py123d + the nuPlan devkit); the
        # bundle records what it must cover so a missing arrow is obvious, and
        # what it actually covers so a *short* or *long* one is obvious too.
        "arrow": arrow,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # No handoff.txt: its only original content is this machine's absolute path
    # to offsets/, which the renderer open()s directly, so a published copy
    # fails on every other box. `--handoff <bundle>` derives it from the
    # manifest at use time instead.
    (bundle / "handoff.txt").unlink(missing_ok=True)
    _write_readme(bundle, token, log_name, clips, meta)
    return bundle


def _arrow_summary(arrow: Path, token: str, t0: int, t1: int) -> dict:
    """What the Arrow scenario actually covers, next to what it should.

    A silently short Arrow renders fine and then runs out of scenario
    mid-episode, so the covered span is recorded rather than assumed.

    The span is checked from **both** ends. ``covers_window`` alone is
    one-sided and an over-long Arrow satisfies it, which is how two published
    bundles shipped 21.5 s of scenario over a 20 s reconstruction: the extra
    frames have no model behind them, and ``_pick_handoff`` answers an
    out-of-window timestamp with its *nearest* scene rather than an error, so
    the tail renders the last sub-clip frozen at its boundary in silence.
    ``overshoot_us`` records how far past the window the Arrow runs, and
    ``matches_window`` is the two-sided verdict.
    """
    out: dict = {"path": "arrow", "present": False,
                 "window_us": [t0, t1]}
    ego = arrow / "logs" / "nuplan_test" / token / "ego_state_se3.arrow"
    if not ego.exists():
        return out
    out["present"] = True
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        table = ipc.open_file(pa.memory_map(str(ego))).read_all()
        ts = table.column("ego_state_se3.timestamp_us").to_pylist()
        covers = bool(ts[0] <= t0 + MAX_POSE_SNAP_US
                      and ts[-1] >= t1 - MAX_POSE_SNAP_US)
        # Positive = frames beyond the reconstruction; negative = the Arrow
        # ends inside the window (already caught by ``covers_window``).
        overshoot = int(ts[-1] - t1)
        undershoot = int(t0 - ts[0])
        out.update(frames=table.num_rows,
                   covered_us=[ts[0], ts[-1]],
                   covered_s=round((ts[-1] - ts[0]) / 1e6, 3),
                   covers_window=covers,
                   overshoot_us=overshoot,
                   starts_early_us=undershoot,
                   matches_window=bool(covers
                                       and overshoot <= MAX_POSE_SNAP_US
                                       and undershoot <= MAX_POSE_SNAP_US))
        maps = sorted(p.name for p in (arrow / "maps" / "nuplan").glob("*.arrow"))
        out["maps"] = maps
    except Exception as exc:  # noqa: BLE001 - a summary must not break the build
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _arrow_warnings(arrow: dict) -> list[str]:
    """Manifest warnings for an Arrow that does not match its window.

    These land in ``manifest["warnings"]`` next to the sub-clip gap/overlap
    warnings, so a bad Arrow is visible in the published artifact itself
    rather than only to whoever ran the build.
    """
    if not arrow.get("present") or "covered_us" not in arrow:
        return ["arrow: missing or unreadable — the bundle has pixels but no scenario"]
    warns: list[str] = []
    over = int(arrow.get("overshoot_us", 0))
    early = int(arrow.get("starts_early_us", 0))
    if over > MAX_POSE_SNAP_US:
        warns.append(
            f"arrow runs {over / 1e6:.3f} s PAST the reconstruction window "
            f"(~{round(over / 100_000)} frames at 10 Hz with no model behind "
            "them) — rebuild it on the manifest's t_start_us/t_stop_us, not "
            "the stitch_jobs window")
    if early > MAX_POSE_SNAP_US:
        warns.append(
            f"arrow starts {early / 1e6:.3f} s BEFORE the reconstruction window")
    if not arrow.get("covers_window"):
        warns.append("arrow is SHORT of the window — an episode will run out "
                     "of scenario mid-drive")
    return warns


def _write_readme(bundle: Path, token: str, log_name: str,
                  clips: list[SubClip], meta: Optional[ScenarioMeta] = None) -> None:
    rows = "\n".join(
        f"| {c.scene_id} | {c.t_start_us} | {c.t_stop_us} | "
        f"{c.offset_xy_utm[0]:.3f}, {c.offset_xy_utm[1]:.3f} |" for c in clips)
    if meta and meta.taxonomy_leaves:
        leaves = ", ".join(f"`{l}` {n}" for l, n in
                           zip(meta.taxonomy_leaves, meta.taxonomy_leaf_names))
        types = ", ".join(f"`{t}`" for t in meta.scenario_types)
        taxonomy_line = (f"Taxonomy leaves: {leaves} — nuPlan type(s): {types}\n"
                         f"Inserted actors: {'yes' if meta.has_inserted_actors else 'no'}\n")
    else:
        taxonomy_line = ""
    (bundle / "README.md").write_text(f"""# NavSafe eval scenario `{token}`

20 s of driving as four 5 s NuRec reconstructions, served over gRPC.

Source nuPlan log: `{log_name}`
{taxonomy_line}Window: `{clips[0].t_start_us}` → `{clips[-1].t_stop_us}`
({(clips[-1].t_stop_us - clips[0].t_start_us) / 1e6:.1f} s)

| scene id | t_start_us | t_stop_us | offset_xy_utm |
|---|---|---|---|
{rows}

Each `.usdz` is a **trained NuRec model** (its bulk is `checkpoint.ckpt`), not
exported geometry: `serve-grpc` renders from it, IsaacSim cannot. The four are
separate models, so they are not merged — `NUREC_GRPC_HANDOFF` switches between
them by time window and shifts the camera by that window's `offset_xy_utm`.

## 1. Serve

```bash
docker run --rm --gpus all -p 8085:8080 -v $PWD:/artifacts \\
  nvcr.io/nvidia/nre/nre-ga serve-grpc --enable-editing-actors \\
  --artifact-glob '/artifacts/*.usdz'
```

Check the scene list it prints includes all four ids above: a renderer that
does not hold a scene **falls back to raster silently**.

## 2. Evaluate

`NUREC_GRPC_HANDOFF` names the offsets JSON by **absolute path** — the renderer
opens it directly — so it is derived from this bundle's location rather than
shipped as a file that would point at the machine that built it. Everything it
needs is in `manifest.json`; this needs no NavSafe install:

{HANDOFF_SNIPPET}

Inside the repo, `navsafe bundle --handoff $BUNDLE`
prints the same string.

```bash
NUREC_GRPC_HOST=localhost NUREC_GRPC_PORT=8085 NUREC_GRPC_CAM_RIG=recon \\
python navsafe/cli/eval_entry.py \\
  --scenario-source py123d --py123d-data-root {bundle}/arrow --py123d-scene-index 0 \\
  --render-backend nurec_grpc \\
  --model-type <policy> --checkpoint <ckpt> \\
  --traffic-mode semi_reactive --replan-rate 5 \\
  --ego-replay-frames 8 --eval-frames 180 \\
  --output-dir out/{token}
```

## 3. Score

```bash
python navsafe/tools/score_run.py \\
  --eval-dir out/{token} --py123d-data-root {bundle}/arrow --warmup-frames 8
```

`arrow/` (ego, boxes, map, route) is built from the nuPlan log for the window
above — see `docs/reconstruct_navsim_nuplan.md` §4. Without it there is no
scenario to drive, only pixels to render.
""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handoff", type=Path, default=None, metavar="BUNDLE",
                    help="print the NUREC_GRPC_HANDOFF value for an on-disk "
                         "bundle and exit (paths are resolved from BUNDLE, so "
                         "this works on any machine)")
    ap.add_argument("--usdz-dir", type=Path,
                    help="downloaded NavSafe intermediate_data tree (<token>/<token>s1..s4.usdz), e.g. via fetch_usdz.py")
    ap.add_argument("--nuplan-splits", type=Path,
                    help="dir of nuPlan .db logs (the test split for navhard)")
    ap.add_argument("--jobs", type=Path,
                    help="TSV: scenario_type, token, t0_us, t1_us, log_name")
    ap.add_argument("--out", type=Path, default=None,
                    help="bundle root; omit with --verify-only")
    ap.add_argument("--token", default=None, help="only this scenario")
    ap.add_argument("--verify-only", action="store_true",
                    help="report what would be written, touch nothing")
    args = ap.parse_args()

    # --handoff is a lookup on a finished bundle, not a build: it needs neither
    # the usdz nor the nuPlan logs, so it is answered before their checks.
    if args.handoff is not None:
        print(handoff_for_bundle(args.handoff.resolve()))
        return 0
    for required in ("usdz_dir", "nuplan_splits", "jobs"):
        if getattr(args, required) is None:
            raise SystemExit(f"--{required.replace('_', '-')} is required to build a bundle")

    scenario_meta = load_scenario_meta(args.jobs)
    logs: dict[str, str] = {}
    for line in args.jobs.read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 5:
            logs[f[1].strip()] = f[4].strip()

    # `.cache` and friends: hf_hub keeps its bookkeeping beside the payload.
    tokens = sorted(p.name for p in args.usdz_dir.iterdir()
                    if p.is_dir() and not p.name.startswith("."))
    if args.token:
        tokens = [t for t in tokens if t == args.token]
        if not tokens:
            raise SystemExit(f"{args.token}: no such scenario under {args.usdz_dir}")
    if not args.verify_only and args.out is None:
        raise SystemExit("--out is required unless --verify-only")

    ok = failed = 0
    for token in tokens:
        log_name = logs.get(token)
        if log_name is None:
            print(f"[skip] {token}: no job row (source log unknown)")
            failed += 1
            continue
        try:
            clips = build_scenario(token, log_name, args.usdz_dir,
                                   args.nuplan_splits)
        except (FileNotFoundError, ValueError, LookupError) as exc:
            print(f"[fail] {token}: {exc}")
            failed += 1
            continue

        span = (clips[-1].t_stop_us - clips[0].t_start_us) / 1e6
        print(f"[ok]   {token}  {span:.1f}s  {log_name}")
        for c in clips:
            print(f"         {c.scene_id}  {c.t_start_us}..{c.t_stop_us}  "
                  f"offset=({c.offset_xy_utm[0]:.2f}, {c.offset_xy_utm[1]:.2f})  "
                  f"snap={c.pose_snap_us:+d}us")
        for w in check_contiguous(clips):
            print(f"         WARN {w}")
        meta = scenario_meta.get(token)
        if meta and meta.taxonomy_leaves:
            leaves = ", ".join(f"{l} {n}" for l, n in
                               zip(meta.taxonomy_leaves, meta.taxonomy_leaf_names))
            print(f"         {leaves}  ({';'.join(meta.scenario_types)})")
        elif meta:
            print(f"         WARN no event type for scenario_type(s)={meta.scenario_types!r}")
        if not args.verify_only:
            print(f"         -> {write_bundle(token, log_name, clips, args.usdz_dir, args.out, meta=meta)}")
        ok += 1

    print(f"\n{ok} scenario(s) ready, {failed} skipped/failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
