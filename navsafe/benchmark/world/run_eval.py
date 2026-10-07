"""Stage 6: closed-loop evaluation of one NavSafe seed against its recon.

Runs the eval client (this repo) against a ``serve-grpc`` renderer reachable at
``NUREC_GRPC_HOST``.  Everything that defines the episode comes from
``seed.json``, so the same command is reproducible from the frozen seed:

* ``--ego-replay-frames`` = the seed's ``t_pre`` -- the policy warm-up margin of
  the event contract.  The policy takes over exactly at ``t_trig``.
* ``--eval-frames``       = the window MINUS the warm-up, capped at
  ``EPISODE_FRAMES``.  ``eval_frames`` counts *scored* frames and the run lasts
  ``warmup + eval_frames`` (``evaluator.py``), so passing the whole window would
  score the policy for ``t_pre`` seconds past ``t_term + t_post`` -- outside the
  mined event and past the extent the reconstruction was trained on.  The cap is
  what keeps every seed the same length; see ``EPISODE_FRAMES``.

Coordinate frames: the reconstruction was trained on a store re-referenced to
ego frame 0 (``NCORE_REREF_FRAME0``), so the scenario must be recentred to the
same origin (``PY123D_RECENTER=1``).  Both origins are the same nuPlan ego pose,
so they align at offset 0 -- ``NUREC_GRPC_ORIGIN_OFFSET_FILE`` must stay unset
here or the shift is subtracted twice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg

SIM_HZ = 10.0
T_PRE_S = 2.0  # keep in sync with navsafe.mining.windowing.T_PRE_S

# EVERY EPISODE IS THE SAME LENGTH. `stitch_jobs.tsv` does not declare one
# window length: of its 417 rows, 266 are 21.5 s, 148 are exactly 20.0 s and 3
# are 22.0 s, so `<token>_20s` is a name rather than a measurement. Derived
# straight from the window, a seed's scored count was therefore 195 frames on
# one host and 180 on the next, and every per-episode metric — route
# completion, ego progress, anything EPDMS normalises by episode — was averaged
# across scenarios of unequal duration. A benchmark cannot report one number
# over a mixed denominator.
#
# 20.0 s is the cap because it is also what exists to render: a stitched host is
# four 5 s reconstructions, so a 21.5 s window's last stretch has no recon
# behind it. Truncating costs nothing that was ever renderable.
EPISODE_FRAMES = int(20.0 * SIM_HZ)


def stage_py123d_root(seed: dict, tmp_root: Path) -> Path:
    """A single-scene py123d root: this seed's log + the shared maps.

    ``--py123d-scene-index 0`` then unambiguously selects this seed rather than
    whichever log happens to sort first in the shared arrow tree.
    """
    tok = seed["seed_id"]
    arrow = Path(seed["artifacts"]["arrow"])
    log_src = Path(seed["artifacts"]["arrow_log"])
    split = log_src.parent.name

    root = tmp_root / f"py123d_{tok}"
    if root.exists():
        shutil.rmtree(root)
    (root / "logs" / split).mkdir(parents=True)
    (root / "logs" / split / tok).symlink_to(log_src)
    (root / "maps").symlink_to(arrow / "maps")
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--navsafe", default=str(cfg.NAVSAFE_ROOT))
    ap.add_argument("--grpc-host", default=os.environ.get("NUREC_GRPC_HOST", "localhost"))
    ap.add_argument("--model-type", default="drivor")
    ap.add_argument("--checkpoint", default=cfg.DEFAULT_CHECKPOINT)
    ap.add_argument("--traffic-mode", default="log_replay",
                    choices=["log_replay", "idm", "none"],
                    help="NavSafe interaction regime (taxonomy doc Table V/VI)")
    ap.add_argument("--controller", default="pure_pursuit")
    ap.add_argument("--execution-mode", default="teleport",
                    help="same values as the evaluator; recorded in the trace "
                         "because comfort is unmeasurable under teleport")
    ap.add_argument("--camera-resolution-scale", type=float, default=1.0)
    ap.add_argument("--replan-rate", type=int, default=5)
    ap.add_argument("--tmp-root", default="/tmp")
    ap.add_argument("--tag", default="", help="suffix for the output dir")
    ap.add_argument("--run", default="",
                    help="REQUIRED unless --dry-run: names the campaign this "
                         "episode belongs to. Output goes to "
                         "NAVSAFE_RUNS/<today>-<slug>/eval/<seed>/<shape>/. "
                         "Without a campaign the path is named after the "
                         "episode shape alone, so re-running a seed after a "
                         "fix overwrites the earlier result in place and "
                         "collect.py silently aggregates a mixture of both.")
    # Same names and units as the evaluator -- one concept, one spelling, so a
    # value can be moved between the two commands without changing meaning.
    # Left unset, both are derived from the seed's frozen window, which is the
    # reason to use this wrapper at all.
    ap.add_argument("--ego-replay-frames", type=int, default=None,
                    help="frames of logged-ego replay before the policy takes "
                         "over (default: the seed's t_pre)")
    ap.add_argument("--eval-frames", type=int, default=None,
                    help="frames the policy drives and is scored; 0 = pure "
                         "replay (default: the seed's window less t_pre)")
    ap.add_argument("--ego-replay-all", action="store_true",
                    help="ego follows the logged trajectory for the whole clip; "
                         "the policy never drives. This is the reconstruction "
                         "check, not a policy evaluation: rendering along the "
                         "exact trajectory the recon was trained on isolates "
                         "render quality from any policy behaviour.")
    ap.add_argument("--asset-harvester-replace", nargs="?", const="auto",
                    default=None, metavar="MANIFEST_JSON",
                    help="render this seed's logged actors from harvested "
                         "per-object assets instead of the reconstruction's "
                         "baked gaussians (appearance only; no metric moves). "
                         "No value = the bank beside the seed's own scenario. "
                         "Build one with `navsafe harvest "
                         "harvest <scene_id>`")
    ap.add_argument("--no-trace", action="store_true",
                    help="skip the NavSafe trace and rubric verdict; leaves only "
                         "the EPDMS artifacts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # A scored episode has to belong to a named campaign. The old default
    # named the output after the episode shape alone, so re-running a seed
    # after a fix overwrote the earlier result and the aggregate silently
    # mixed the two. Refusing here is the only thing that actually stops it.
    if not args.run and not args.dry_run:
        ap.error("--run is required: name the campaign this episode belongs "
                 "to (e.g. --run epdms-no-extcomfort), so its output lands in "
                 "its own directory instead of overwriting an earlier one. "
                 "Use --dry-run to inspect the command without writing.")

    p = Path(args.seed)
    seed = json.loads((p / "seed.json" if p.is_dir() else p).read_text())
    tok = seed["seed_id"]

    # The window already contains t_pre, and the episode runs
    # warmup + eval_frames, so the scored count is the window less the warm-up.
    warmup_frames = int(round(T_PRE_S * SIM_HZ))
    window_frames = int(round(seed["window"]["window_duration_s"] * SIM_HZ))
    if window_frames > EPISODE_FRAMES:
        print(f"[navsafe] {tok}: window is {window_frames} frames "
              f"({window_frames / SIM_HZ:.1f}s); scoring the first {EPISODE_FRAMES} "
              f"so every seed is the same length", file=sys.stderr)
    # The cap assumes the reconstruction reaches 20 s, and not every host's
    # does: measured 20.000 s on 0bcae698fd905226 but 19.900 s on
    # 01a58976a2e45a3d, whose four windows are 4.8-5.0 s rather than 5.0 each.
    # A seed that cannot supply the full episode is NOT quietly shortened —
    # that is the mixed denominator this cap exists to remove. It is named, so
    # it can be re-reconstructed or dropped from the set as a decision.
    recon_frames = int(round(
        (seed["window"]["recon_t1_us"] - seed["window"]["recon_t0_us"]) / 1e6 * SIM_HZ))
    if recon_frames < EPISODE_FRAMES:
        print(f"[navsafe] {tok}: reconstruction covers {recon_frames} frames "
              f"({recon_frames / SIM_HZ:.1f}s), short of the {EPISODE_FRAMES}-frame "
              f"episode — the tail has no recon behind it", file=sys.stderr)
    eval_frames = min(window_frames, EPISODE_FRAMES) - warmup_frames
    if args.ego_replay_frames is not None:
        warmup_frames = int(args.ego_replay_frames)
    if args.eval_frames is not None:
        eval_frames = int(args.eval_frames)
    if args.ego_replay_all:
        # `frame < ego_replay_frames` is the replay branch and the run ends at
        # `ego_replay_frames + eval_frames`, so covering the whole
        # reconstruction window with warm-up and asking for zero scored frames
        # renders the entire clip without the policy ever taking over.
        recon_s = (seed["window"]["recon_t1_us"] - seed["window"]["recon_t0_us"]) / 1e6
        warmup_frames = int(round(recon_s * SIM_HZ))
        eval_frames = 0

    pure_replay = args.ego_replay_all or eval_frames == 0
    if args.tag:
        default_tag = args.tag
    elif pure_replay:
        default_tag = f"ego_replay_{(warmup_frames / SIM_HZ):.0f}s"
    else:
        default_tag = (f"{args.traffic_mode}_{(warmup_frames / SIM_HZ):.0f}s"
                       f"+{(eval_frames / SIM_HZ):.0f}s")
    if args.run:
        # One directory per campaign, seeds beneath it: the campaign can be
        # kept or dropped whole, and two of them cannot collide.
        out_dir = cfg.run_dir(args.run) / "eval" / tok / default_tag
    else:
        # Only reachable under --dry-run, which writes nothing anyway.
        out_dir = Path(seed["artifacts"]["eval"]) / default_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = stage_py123d_root(seed, Path(args.tmp_root))

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{cfg.NRE_STUBS}:{args.navsafe}",
        "NUREC_GRPC_HOST": args.grpc_host,
        # PINNED, not inherited. The recon rig is the reconstruction's own
        # calibrated camera; the navsim rig rebuilds a synthetic pinhole and is
        # the A/B path, not the default. Reading it from the ambient
        # environment let a worker script that exported
        # NUREC_GRPC_CAM_RIG=navsim silently render three campaigns
        # (recheck-edit80, car-rescale-check, v10-cutin) under the wrong rig,
        # with nothing in the eval command saying so. Which rig a NavSafe seed
        # renders under is a property of NavSafe, so NavSafe states it.
        "NUREC_GRPC_CAM_RIG": "recon",
        # Recentre the scenario to ego frame 0 to match the re-referenced store.
        "PY123D_RECENTER": "1",
        "NUPLAN_MAPS_ROOT": str(cfg.NUPLAN_MAPS_DEVKIT),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        # Per-run camdump: the file opens in APPEND mode, so a shared path
        # silently mixes runs and poisons any jitter analysis.
        "NUREC_GRPC_CAMDUMP": str(out_dir / "camdump.txt"),
    })
    if not args.no_trace and not pure_replay:
        # The canonical trace and the rubric verdict. Skipped for a pure replay:
        # with no scored frames there is no policy behaviour to judge, and a
        # verdict there would be a verdict on the log.
        # `traffic_mode` names a traffic manager, `regime` names an interaction
        # regime in the taxonomy -- IDM traffic is the "reactive" regime.
        regime = {"log_replay": "log_replay", "idm": "reactive"}.get(
            args.traffic_mode, args.traffic_mode)
        env.update({
            "NAVSAFE_TRACE": str(p if p.is_dir() else p.parent),
            "NAVSAFE_TRACE_OUT": str(out_dir / "trace"),
            "NAVSAFE_POLICY": args.model_type,
            "NAVSAFE_REGIME": regime,
            "NAVSAFE_CHECKPOINT": args.checkpoint,
            # the evaluator defaults to teleport; the scorer needs to know because
            # comfort is not measurable when the ego is placed rather than driven.
            "NAVSAFE_EXECUTION_MODE": args.execution_mode,
        })
    env.pop("NUREC_GRPC_ORIGIN_OFFSET_FILE", None)  # would double-subtract

    cmd = [
        sys.executable, f"{args.navsafe}/navsafe/cli/eval_entry.py",
        "--scenario-source", "py123d",
        "--py123d-data-root", str(data_root),
        "--py123d-scene-index", "0",
        "--render-backend", "nurec_grpc",
        "--model-type", args.model_type,
        "--checkpoint", args.checkpoint,
        "--traffic-mode", args.traffic_mode,
        "--controller", args.controller,
        "--execution-mode", args.execution_mode,
        "--eval-frames", str(eval_frames),
        "--ego-replay-frames", str(warmup_frames),
        "--replan-rate", str(args.replan_rate),
        "--camera-resolution-scale", str(args.camera_resolution_scale),
        "--output-dir", str(out_dir),
    ]
    if args.asset_harvester_replace:
        # `--py123d-data-root` here is a STAGED copy under --tmp-root, so
        # the evaluator's "beside the scenario" default cannot find the bank;
        # resolve it from the seed's real Arrow, which names the scenario.
        bank = (Path(args.asset_harvester_replace)
                if args.asset_harvester_replace != "auto"
                else Path(seed["artifacts"]["arrow"]).parent
                / "ah_assets" / "replace_manifest.json")
        if not bank.is_file():
            print(f"[eval] {tok}: no harvested asset bank at {bank}", file=sys.stderr)
            return 1
        cmd += ["--asset-harvester-replace", str(bank)]
    print(f"[eval] {tok} [{seed['family']}] regime={args.traffic_mode} "
          f"frames={eval_frames} warmup={warmup_frames} grpc={args.grpc_host}")
    print("[eval]", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    rc = subprocess.call(cmd, env=env, cwd=args.navsafe)
    # the evaluator nests its artifacts one level down (``py123d_<token>/``), so
    # search rather than assuming the layout -- an eval that ran fine must not
    # be reported as failed because the writer moved its output.
    if args.ego_replay_frames is not None:
        warmup_frames = int(args.ego_replay_frames)
    if args.eval_frames is not None:
        eval_frames = int(args.eval_frames)
    if pure_replay:
        # No scored frames, so no metrics -- the artifacts that matter are the
        # rendered frames and the GIF.
        gifs = sorted(out_dir.rglob("visualization/*.gif"))
        print(f"[eval] {tok}: ego-replay render done, {len(gifs)} gif(s)")
        for g in gifs:
            print(f"         {g}")
        return rc
    hits = sorted(out_dir.rglob("metrics.json"))
    if rc != 0 or not hits:
        print(f"[eval] {tok}: FAILED (rc={rc}, metrics_found={len(hits)})", file=sys.stderr)
        return rc or 1
    metrics = hits[0]
    # Also surface it at the regime root so downstream tooling has one
    # predictable path per (seed, regime).
    if metrics.parent != out_dir:
        (out_dir / "metrics.json").write_text(metrics.read_text())
    m = json.loads(metrics.read_text())
    keep = ("score", "no_at_fault_collisions", "drivable_area_compliance",
            "ego_progress", "time_to_collision_within_bound")
    print(f"[eval] {tok}: " + "  ".join(
        f"{k}={m[k]:.3f}" for k in keep if isinstance(m.get(k), (int, float))))
    verdict = out_dir / "trace" / "verdict.json"
    if verdict.exists():
        print(f"[eval] {tok}: navsafe verdict -> {verdict}")
    elif not args.no_trace:
        print(f"[eval] {tok}: WARNING no navsafe verdict written "
              f"(expected {verdict})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
