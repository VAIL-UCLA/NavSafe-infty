#!/usr/bin/env python3
"""Evaluate a policy on py123d Arrow data using NuRec gRPC observations."""

import argparse
from pathlib import Path

# The benchmark's own recipes, resolved by path rather than by importing
# navsafe.benchmark — argument parsing happens long before the Isaac app is
# launched and the heavy imports below become legal.
_AUTOSELECT_DEFAULT_DIR = (
    Path(__file__).resolve().parents[2]
    / "navsafe" / "benchmark" / "recipes" / "benchmark"
)

ap = argparse.ArgumentParser()
ap.add_argument("--scenario-source", choices=["py123d"], default="py123d",
                help="Evaluation input format: py123d Arrow (default).")
ap.add_argument(
    "--nurec-work-dir",
    default=None,
    help="gs3d/NuRec real2sim work dir (<work>/<clip>/output/<run>). "
    "With py123d Arrow input, "
    "the arrow log stays the symbolic source and the loader "
    "resolves the reconstruction for the scene from here — no "
    "SD pickle needed.",
)
ap.add_argument("--model-type", default="transfuser")
ap.add_argument(
    "--num-proposals",
    type=int,
    default=None,
    help="Log the top-N candidate trajectories and their scores per replan "
         "(drivor, diffusiondrive, diffusiondrivev2, rap). Off by default: "
         "these adapters publish trajectory_coarse/coarse_scores only when it "
         "is set, and otherwise record the selected plan alone, which cannot "
         "show whether a safe proposal existed and was outranked. Capped at "
         "the model's own proposal count.",
)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--config", default=None)
ap.add_argument("--plan-anchor-path", default=None)
ap.add_argument("--py123d-data-root", default="data")
ap.add_argument("--py123d-scene-index", type=int, default=0)
ap.add_argument(
    "--py123d-frame-window", type=int, nargs=2, metavar=("START", "STOP"),
    help="Half-open Arrow iteration window for a held-out scenario instance. "
    "STOP is exclusive.",
)
ap.add_argument(
    "--remove-agents",
    action="store_true",
    help="Drop all non-ego tracked objects from the scenario "
    "(dynamic + static) so they vanish from BOTH the render "
    "and the sim state (BEV / collision / observation) — a "
    "consistent background-only scene.",
)
ap.add_argument(
    "--eval-seed", type=int, default=None,
    help="Global RNG seed (python random, numpy, torch — CPU and CUDA) for "
    "the eval process, seeded once before env/policy construction. "
    "It makes repeated runs of the SAME bundle "
    "against the SAME policy diverge, for policies with any internal "
    "sampling (temperature/top-k decoding, dropout at inference, stochastic "
    "adapters). It does not add noise to the simulation, traffic or "
    "controller, which have no RNG of their own and remain bit-identical "
    "across seeds for a purely deterministic policy (e.g. pdm_closed). "
    "Unset (default) leaves every RNG's own default seeding untouched.")
ap.add_argument(
    "--output-dir",
    required=True,
    help="Where artifacts go. REQUIRED, and deliberately without a default: the "
    "evaluator nests everything under <output-dir>/<scenario_id>/, and "
    "scenario_id is the scenario PATH'S LAST SEGMENT (evaluator.py "
    "`self.scenario_id = _scenario_path.name`). For a py123d root like "
    ".../<clip>/arrow that is literally 'arrow' for every clip, so the output "
    "dir is the only thing that distinguishes one scene's artifacts from "
    "another's. A shared default would quietly interleave runs from different "
    "scenes in the same folder. (navsafe/world/run_eval.py dodges this by "
    "staging a per-seed root whose name IS the token.)",
)
ap.add_argument(
    "--wandb",
    action="store_true",
    help="Log the final eval metrics to Weights & Biases "
    "(soft dependency; skipped with a warning if wandb is "
    "unavailable)",
)
ap.add_argument("--wandb-project", default="navsafe-eval")
ap.add_argument("--wandb-run-name", default=None)
ap.add_argument("--wandb-entity", default=None)
ap.add_argument("--traffic-mode", default="log_replay")
ap.add_argument(
    "--traffic-takeover",
    default="continuous",
    choices=["continuous", "spawn"],
    help="For --traffic-mode semi_reactive: when the IDM-takeover test runs. "
    "'continuous' (default) re-tests every vehicle every frame, so one the "
    "ego passes becomes reactive as it falls behind; 'spawn' tests once at "
    "the vehicle's first frame, which is MetaDrive's own behaviour and "
    "reproduces NavSafe results recorded with the earlier behaviour.",
)
ap.add_argument(
    "--eval-mode",
    default="closed_loop",
    help="Only 'closed_loop' exists. The historical 'open_loop' value was "
    "parsed but never implemented (the run was closed-loop regardless) and "
    "is now refused; for a pure log-replay (open-loop) run use "
    "--ego-replay-frames >= --eval-frames.",
)
ap.add_argument(
    "--navsafe-metrics",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Score the finished run with the four NavSafe metrics "
    "(DS / SR / Efficiency / Comfort) and write <run>/navsafe_metrics.json. On by "
    "default: metrics.json alone carries the EPDMS family, whose "
    "`driving_score` is a DIFFERENT metric over a different window. Pure "
    "post-processing of stored artifacts — no extra simulation, seconds to "
    "run. --no-navsafe-metrics skips it.",
)
ap.add_argument(
    "--navsafe-prune-artifacts",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="After navsafe_metrics.json is written, delete the artifacts only the "
    "other metric families need: metrics.json, driving_score_summary.csv and "
    "run_meta.json. Off by default because those ARE the result on an EPDMS "
    "run. Only ever prunes a directory that got a navsafe_metrics.json, so a "
    "scoring failure leaves everything in place. Note the cost: those two "
    "files are also the INPUT re-scoring reads, so score_run.py on a "
    "pruned run loses the per-frame collision/drivability flags, the TL column "
    "the red_light channel is measured from, and the collision total — which "
    "is why a note saying so is written into navsafe_metrics.json.",
)
ap.add_argument(
    "--controller",
    default="lqr",
    choices=["pure_pursuit", "pid", "lqr"],
    help="Tracker used by the controller/physics execution modes. Default lqr.",
)
ap.add_argument(
    "--execution-mode",
    default="controller",
    choices=["teleport", "controller", "physics"],
    help="How the ego executes plans: tracker + bicycle model (default), "
    "arc-paced teleport along the plan, or tracker + PhysX-integrated ego. "
    "The default is 'controller' because teleport places the ego AT the "
    "planned waypoints, so a trajectory the vehicle could never track is "
    "executed exactly and the tracking error it would have cost is invisible; "
    "it also makes the comfort term meaningless (the motion is the placement "
    "cadence, not dynamics) so comfort is excluded under it. Use "
    "--execution-mode teleport --controller pure_pursuit to reproduce numbers "
    "recorded against the old defaults.",
)
ap.add_argument(
    "--contact-dynamics",
    default=None,
    action=argparse.BooleanOptionalAction,
    help="PhysX contact response between replay agents and the "
    "ego: a replay car that hits the ego physically pushes "
    "it. Default: auto — on whenever --execution-mode "
    "physics. --no-contact-dynamics disables",
)
ap.add_argument(
    "--terminate-on-collision",
    action="store_true",
    help="End the episode on ANY ego-box contact, not just at-fault "
    "ones. This is the NavSafe convention: termination.py defines "
    "CONTACT_NOT_AT_FAULT as an episode ending (the contact is not "
    "charged to the ego, but nothing after it is scorable). Off by "
    "default, which keeps the NavSim at-fault-only behaviour.",
)
ap.add_argument("--replan-rate", type=int, default=5)
ap.add_argument("--sim-dt", type=float, default=0.1)
ap.add_argument(
    "--record-reactivity-trace", action="store_true",
    help="Buffer Section-4 primitive traces in RAM and write one "
         "reactivity_trace.zip to --output-dir at episode finalization. "
         "Records no images and performs no per-frame trace I/O.")
ap.add_argument(
    "--reactivity-condition", default="auto",
    choices=("auto", "hazard", "irrelevant", "no_change"),
    help="Causal arm label stored in reactivity_trace.zip. Auto infers "
         "hazard for an enabled authored event and no_change otherwise.")
ap.add_argument("--reactivity-group-id", default=None,
                help="Pair/triplet identifier shared by matched causal arms.")
ap.add_argument("--ego-replay-frames", type=int, default=8)
ap.add_argument(
    "--keep-ego-replay-frames",
    action="store_true",
    help="Respect --ego-replay-frames even when a --recipe was baked with a "
    "different hand-off. The recipe's baked actor poses are absolute per-frame "
    "arrays and are unaffected; what changes is only WHO drives the ego after "
    "the recipe's hand-off. Use for pure log-replay renders (--ego-replay-frames "
    ">= --eval-frames), where the recipe's arrival window still holds because "
    "the replayed ego IS the trajectory it was solved against.",
)
ap.add_argument("--ego-perturb-history", choices=("handoff", "controller"),
                default="handoff", help="Controller mode renders a continuous tracked augmented warmup; no handoff teleport.")
ap.add_argument(
    "--ego-perturb-lateral", type=float, default=0.0, metavar="M",
    help="Displace the ego sideways by M metres on the HAND-OFF frame -- the "
    "first frame the policy owns the car (--ego-replay-frames, or the "
    "recipe's value when a recipe overrides it). Positive is to the ego's "
    "LEFT. The replay prefix is untouched, so two runs differing only in this "
    "flag share an identical warm-up and differ only in the pose the policy is "
    "handed. Nothing is checked for you: an offset that puts the ego inside "
    "another vehicle or off the drivable area will be simulated and scored as "
    "such.")
ap.add_argument(
    "--ego-perturb-longitudinal", type=float, default=0.0, metavar="M",
    help="Displace the ego along its own heading by M metres on the hand-off "
    "frame. Positive is FORWARD. Speed is unchanged -- this moves the car, it "
    "does not add a jump in velocity.")
ap.add_argument(
    "--ego-perturb-yaw", type=float, default=0.0, metavar="DEG",
    help="Rotate the ego by DEG degrees about its own centre on the hand-off "
    "frame. Positive is counter-clockwise. The velocity vector rotates with "
    "the body, so the policy is handed a car pointing somewhere else rather "
    "than one already sideslipping.")
ap.add_argument(
    "--eval-frames", type=int, default=None,
    help="Cap the SCORED window at this many frames. Unset (the default) runs "
    "the episode INDEFINITELY: it ends when the termination taxonomy ends it "
    "(goal, contact, off-drivable, deadlock), or at --route-time-limit-s if "
    "enabled. A frame count was the wrong bound for scoring -- an episode "
    "that merely ran out of frames was reported as a policy that blew its "
    "budget -- but it is still the right bound for collection (mining, "
    "batch), which is why the flag remains. 0 or less means a pure replay: "
    "no policy, the ego follows the log for --ego-replay-frames frames.")
ap.add_argument(
    "--route-time-limit-s", type=float, default=60.0,
    help="Maximum scored route-completion time in seconds when --eval-frames "
    "is unset. 0 disables the clock entirely; semantic endings still apply. "
    "Default: 60 seconds.")
ap.add_argument(
    "--enable-vis",
    action="store_true",
    help="Write per-frame BEV/front-cam images and end-of-run GIFs under "
         "<output_dir>/<scenario_id>/. Off by default (pure render+score; faster).",
)
ap.add_argument(
    "--log-level",
    default=None,
    help="Root log level (DEBUG/INFO/WARNING/...). Unset leaves Python's "
         "default, which is WARNING — so navsafe's INFO lines (traffic "
         "takeovers, scenario setup, GIF summaries) are dropped. Pass INFO "
         "to see them.",
)
ap.add_argument(
    "--vis-cameras",
    default=None,
    help="Comma-separated NAVSIM cameras rendered for the ARTIFACTS ONLY, on "
         "top of whatever the policy asks for (e.g. CAM_B0 for a rear view). "
         "Each writes <cam>.jpg per frame, its own GIF, and an extra panel in "
         "combined.gif. The policy's input is unchanged — it indexes images by "
         "name — so this does not alter the scored behaviour, only the render "
         "cost (one more render per camera per frame). Needs --enable-vis.",
)
ap.add_argument(
    "--trajectory-scorer",
    default=None,
    help="Candidate-trajectory selector (e.g. epdms). Leave unset for "
    "single-trajectory policies like transfuser; PDMS still comes "
    "from the per-frame PDMSScorer + the live EPDMS scorer.",
)
ap.add_argument("--render-backend", default="nurec_grpc", choices=["nurec_grpc"],
                help="NuRec gRPC renderer (default).")
ap.add_argument(
    "--no-real-assets",
    action="store_true",
    help="Spawn replay agents as primitive cubes instead of real "
    "USD meshes (debug: isolate the referenced-asset render path).",
)
ap.add_argument(
    "--no-road",
    action="store_true",
    help="Skip building the road map geometry (lanes/markings/"
    "crosswalks/sidewalks/boundaries) so only the sky + grass "
    "background renders. Baseline isolation aid.",
)
ap.add_argument(
    "--hide-ego",
    action="store_true",
    help="Make the ego vehicle's visual mesh invisible to the camera "
    "(the camera is mounted on the ego, so its own body otherwise "
    "appears as a dark wedge in the frame). Camera/physics keep "
    "working — only the visual is hidden.",
)
ap.add_argument("--camera-resolution-scale", type=float, default=0.5)
# temporal-consistency flags the adapter helper reads (unused for transfuser)
ap.add_argument("--enable-temporal-consistency", action="store_true")
ap.add_argument("--temporal-alpha", type=float, default=1.5)
ap.add_argument("--temporal-lambda", type=float, default=0.3)
ap.add_argument("--temporal-max-history", type=int, default=8)
ap.add_argument("--temporal-sigma", type=float, default=5.0)
ap.add_argument("--consensus-temperature", type=float, default=1.0)
ap.add_argument("--v2-scorer-checkpoint", default=None)
ap.add_argument(
    "--place-static-obstacles",
    type=int,
    default=0,
    help="Scenario-edit tool: drop N static vehicle-class obstacles "
    "on the ego's logged route (road centre) before the scene "
    "builds, to stress closed-loop avoidance. 0 = off.",
)
ap.add_argument(
    "--obstacle-arc-positions",
    default=None,
    help="Comma-separated distances (m) past the replay hand-off "
    "at which to place the obstacles, e.g. '25' or "
    "'20,45,80'. Deterministic — overrides the seeded "
    "arc sampling. Useful to keep obstacles on clean, "
    "well-reconstructed road (the sampler once dropped a car "
    "into the baked dust cloud at an intersection).",
)
ap.add_argument(
    "--cam-height",
    choices=["navsim", "waymo"],
    default="navsim",
    help="Virtual camera mount height above the road for the "
    "nurec_grpc navsim rig. 'navsim' = the nuPlan rig "
    "(CAM_F0 at 1.49 m) — matches the NavSim policies' "
    "training distribution but sits below the WOD "
    "reconstruction's 2.12 m training rig, so renders are "
    "softer. 'waymo' = lift the whole NavSim rig so "
    "CAM_F0 sits at the WOD roof-mount height (2.12 m) — "
    "sharpest renders, but a higher vantage than the "
    "policy saw in training.",
)
ap.add_argument(
    "--obstacle-z-to-ground",
    type=float,
    default=None,
    help="Extra downward shift (m) for injected obstacles AND the "
    "navsim camera, to compensate a reconstruction whose "
    "RENDERED road drifts from the true road. The ego z is now "
    "ground-referenced at the loader (py123d_training_extractor "
    "reads the rear-axle pose, not the bbox centre), so "
    "lidar-supervised artifacts need 0; only camera-only recon "
    "drift (~±0.4 m) needs a nonzero value. Default None -> "
    "per-scene registry (ground_z_calib.json) -> 0.",
)
ap.add_argument(
    "--obstacle-seed",
    type=int,
    default=42,
    help="RNG seed for --place-static-obstacles (archetype + position jitter).",
)
ap.add_argument(
    "--obstacle-types",
    default="car,suv,truck",
    help="Comma-separated obstacle archetype pool for "
    "--place-static-obstacles (car, suv, truck, cone, cone_tall, sedan_uv). "
    "'cone' injects a TRAFFIC_CONE-typed track.",
)
ap.add_argument(
    "--obstacle-nurec-asset",
    action="append",
    default=None,
    metavar="ARCHETYPE=ASSET_ID",
    help="NuRec server asset for an obstacle archetype, e.g. "
    "cone=/assets/cone_3dgs.ply (repeatable). On the "
    "nurec_grpc backend the injected track is registered "
    "into the served scene via edit_assets so it appears "
    "in the render as well as in sim state. ASSET_ID is an "
    "AssetBank track id or a 3DGS PLY path readable inside "
    "the sensorsim container.",
)
ap.add_argument(
    "--replace-agent-ids",
    action="append",
    default=None,
    metavar="TRACK_ID=ARCHETYPE",
    help="Replace an existing scenario agent with an injected "
    "NuRec asset (repeatable), e.g. "
    "wdPo1LwisGiYoUxDltocrg=sedan_uv. The original track is "
    "deleted (vanishes from render AND sim state) and a new "
    "track carrying the archetype's asset (see "
    "--obstacle-nurec-asset) is placed on its per-frame "
    "trajectory. Find track ids with "
    "navsafe.tools.list_scene_agents.",
)
ap.add_argument(
    "--asset-harvester-replace",
    nargs="?",
    const="auto",
    default=None,
    metavar="MANIFEST_JSON",
    help="Render the scenario's logged actors from harvested per-object 3D "
    "assets instead of the reconstruction's baked gaussians (nurec_grpc "
    "backend only). A reconstruction fits each actor to the views the LOGGED "
    "ego had, so a policy that drives differently sees angles no training view "
    "covered and the actor smears -- worst on oncoming traffic and just after "
    "a 5 s handoff. This swaps APPEARANCE ONLY: track ids, boxes, per-frame "
    "poses and therefore every metric are unchanged (contrast "
    "--replace-agent-ids, which deletes a logged actor and inserts a different "
    "one). Takes the bank's replace_manifest.json, or no value to find it "
    "beside the scenario (<py123d-data-root>/../ah_assets/). Build a bank with "
    "`navsafe harvest harvest <scene_id>`.",
)
ap.add_argument(
    "--relocate-agents",
    action="append",
    default=None,
    metavar="TRACK_ID:mode=..,arc=..,speed=..",
    help="Move an existing reconstructed actor onto a synthetic "
    "route-anchored trajectory (safety-critical authoring; "
    "repeatable). Format TRACK_ID:key=val,key=val with keys "
    "mode (static|dynamic), arc (m past the replay hand-off, "
    "default 20), speed (m/s for dynamic, default 3), lateral "
    "(+left m, default 0), yaw_offset_deg (default 0), "
    "archetype (optional). Without archetype the actor KEEPS "
    "its baked appearance (pure pose override); with archetype "
    "it is swapped for that imported asset (see "
    "--obstacle-nurec-asset) on the trajectory. E.g. "
    "wdPo..:mode=static,arc=25  or  wdPo..:mode=dynamic,arc=12,speed=4.",
)
ap.add_argument(
    "--edits-json",
    default=None,
    metavar="JSON_OR_PATH",
    help="Raw scenario_edits, as a JSON list or a path to one. For exercising "
    "the runtime (e.g. a reactive actor) without a frozen recipe: it carries "
    "no checksums, no provenance and no event type, so results from it must not be "
    "scored or shipped. Use --recipe for anything real.",
)
ap.add_argument(
    "--recipe",
    default=None,
    metavar="RECIPE_YAML",
    help="Replay a frozen NavSafe scene-editing recipe. Each actor enters as an "
    "ordinary track declaring its asset, dims and class, plus a spawn pose and "
    "the CONTROLLER that drives it — needs --traffic-mode navsafe, or the "
    "actors appear and stand at their spawn poses for the whole episode. "
    "Checksums are verified before anything reaches the env, and a host "
    "whose frame count / rate disagrees with the recipe is refused. "
    "Composes with the parameter-driven edit flags above, but normally "
    "replaces them: a recipe is self-contained by design.",
)
ap.add_argument(
    "--recipe-dir",
    default=None,
    metavar="DIR",
    help="Auto-select the recipe from the bundle's own manifest: if "
    "scenario_meta.has_inserted_actors is true, run "
    "<DIR>/<LEAF>.<token>.yaml; otherwise run the host unedited. Lets one "
    "sweep mix edited and unedited scenarios without being told which each "
    "cell is. Pass the flag with no value to use the benchmark's own recipes "
    "in the repo. An explicit --recipe always wins, and a scenario whose "
    "manifest claims inserted actors with no matching recipe is an error "
    "rather than a silently unedited run.",
    nargs="?",
    const=str(_AUTOSELECT_DEFAULT_DIR),
)
ap.add_argument(
    "--recipe-variant",
    default="e_plus",
    choices=["e_plus", "e_zero"],
    help="Which half of the recipe's pair to run. 'e_plus' is the built "
    "scenario; 'e_zero' is its counterfactual, dropping the actors named "
    "in the recipe's pair.e0_removes — the ones whose presence defines "
    "the event type.",
)

# ── IsaacSim glib preload (shared helper, must run before AppLauncher) ────────
# IsaacSim's bundled libgobject needs g_dir_unref (glib >= 2.80); on older system
# glib the GPU/USD stack fails to boot. The shared, stdlib-only helper fixes the
# load order (LD_PRELOAD re-exec) before AppLauncher. Lives outside ``navsafe``
# so it imports without pulling torch.
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from navsafe.cli.isaac_boot import ensure_isaac_glib_preload  # noqa: E402

ensure_isaac_glib_preload()

# IsaacSim app (with cameras) must launch before NavSafeEnv / omni imports.
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if args.py123d_frame_window is not None:
    start, stop = args.py123d_frame_window
    if args.scenario_source != "py123d" or not 0 <= start < stop:
        ap.error("--py123d-frame-window requires py123d and 0 <= START < STOP")
if args.log_level:
    import logging as _logging
    # force=True: IsaacSim's bootstrap installs its own root handlers, so a
    # plain basicConfig here would be a no-op.
    _logging.basicConfig(level=args.log_level.upper(), force=True)
# Reject the unimplemented mode before the slow simulator start.
if args.eval_mode != "closed_loop":
    ap.error(
        f"--eval-mode {args.eval_mode} was parsed but never implemented — the "
        "evaluator always runs closed-loop. For a pure log-replay (open-loop) "
        "run, use --ego-replay-frames >= --eval-frames.")
# Likewise refuse the unwired traffic mode pre-boot (the env raises the same
# NotImplementedError, but only after the full IsaacSim boot — through
# navsafe batch-eval that would pay one boot per scene just to fail).
# Message mirrors navsafe.env._bootstrap.IDM_TRAFFIC_UNWIRED_MSG, which
# cannot be imported here without pulling navsafe in before AppLauncher.
if args.traffic_mode == "idm":
    ap.error(
        "--traffic-mode idm is not wired (it was a silent no-op — agents "
        "replayed the log). Use log_replay, semi_reactive (the wired "
        "reactive mode), or no_traffic.")
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from navsafe.env import NavSafeEnv
from navsafe.evaluation.eval_env_config import build_eval_env_cfg
from navsafe.evaluation.evaluator import Evaluator, EvaluationConfig
from navsafe.evaluation.unified_evaluator import (
    create_model_adapter_from_args,
    create_trajectory_scorer,
)

if args.log_level:
    # Re-assert after the imports: AppLauncher installs root handlers and
    # unified_evaluator calls basicConfig at import, either of which can undo
    # the level set above. Setting it on the package logger survives both.
    import logging as _logging
    _logging.getLogger("navsafe").setLevel(args.log_level.upper())

if args.eval_seed is not None:
    # Seed the global RNGs once, before the environment and policy are
    # built. The simulator, traffic and controllers are deterministic, so
    # this is what makes runs with different --eval-seed differ.
    import random as _random
    import numpy as _np
    _random.seed(args.eval_seed)
    _np.random.seed(args.eval_seed)
    try:
        import torch as _torch
        _torch.manual_seed(args.eval_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.eval_seed)
    except ImportError:
        pass
    print(f"[navsafe-eval] eval_seed={args.eval_seed} (python random, numpy, torch)")

# Build the env config for the selected source (py123d Arrow or procedural map).
# The pure-Python builder picks the branch from --scenario-source; the assets /
# camera-rig augmentation below applies on top when render_backend == "assets".

# Resolve how far the ego pose sits above the reconstructed road. The loader
# reports the ego z at ground level, so this offset only absorbs drift of the
# reconstructed road: about zero for lidar-supervised reconstructions, up to
# a few decimetres for camera-only ones. Precedence: explicit flag, then the
# per-scene registry, then 0.
from navsafe.benchmark.editing.ground_z import export_to_consumers, resolve_z_to_ground

_source_path = args.py123d_data_root
args.obstacle_z_to_ground, _z_reason = resolve_z_to_ground(
    args.obstacle_z_to_ground, _source_path
)
print(f"[navsafe-eval] ego-z-to-ground {args.obstacle_z_to_ground} ({_z_reason})", flush=True)

# Single source of truth for the ego-pose-to-ground drop: the renderer's
# navsim camera (NUREC_GRPC_EGO_Z_TO_GROUND) and the front-cam map overlay
# (NAVSAFE_EGO_Z_TO_GROUND) must use the SAME constant as the obstacle
# placement, or cameras/overlays/obstacles disagree about where the road is.
export_to_consumers(args.obstacle_z_to_ground)
# --cam-height waymo: lift every NavSim camera by the same delta so CAM_F0
# lands at the WOD roof-mount height (baked pcam_f0 extrinsic z = 2.116 m
# above ground). The nurec_grpc renderer and the front-cam overlay both read
# cfg["z"] as height-above-road, so they stay consistent automatically.
if args.cam_height == "waymo":
    from navsafe.utils.camera_utils import NAVSIM_CAM_CONFIGS as _NCC

    _delta = 2.116 - float(_NCC["CAM_F0"]["z"])
    for _c in _NCC.values():
        _c["z"] = float(_c["z"]) + _delta
    print(
        f"[navsafe-eval] cam-height=waymo: NavSim rig lifted {_delta:+.3f} m "
        f"(CAM_F0 -> 2.116 m above road)",
        flush=True,
    )

env_cfg = build_eval_env_cfg(args)

# Harvested-asset replacement is a property of the render server session, not
# of the scenario or the env, so it travels the same way every other nurec_grpc
# option does: an environment variable the backend reads at setup. Resolved
# here rather than in the backend because "beside the scenario" is a fact about
# this invocation's data root, which the backend never sees.
if args.asset_harvester_replace:
    if args.render_backend != "nurec_grpc":
        ap.error(
            "--asset-harvester-replace needs --render-backend nurec_grpc: the "
            "swap is done by the render server, and no other backend has one")
    _bank = args.asset_harvester_replace
    if _bank == "auto":
        if not args.py123d_data_root:
            ap.error("--asset-harvester-replace with no path needs "
                     "--py123d-data-root to find the bank beside")
        _bank = str(Path(args.py123d_data_root).resolve().parent
                    / "ah_assets" / "replace_manifest.json")
    if not Path(_bank).is_file():
        ap.error(f"--asset-harvester-replace: no manifest at {_bank}. Build one "
                 f"with `navsafe harvest harvest <scene_id>`.")
    os.environ["NUREC_GRPC_ASSET_REPLACE"] = _bank
    print(f"[navsafe-eval] asset-harvester replace: {_bank}", flush=True)

# Shared archetype -> NuRec asset map (used by both obstacle placement and
# agent replacement).
_asset_map = {}
for _spec in args.obstacle_nurec_asset or []:
    _k, _, _v = _spec.partition("=")
    if not (_k.strip() and _v.strip()):
        ap.error(f"--obstacle-nurec-asset expects ARCHETYPE=ASSET_ID, got {_spec!r}")
    _asset_map[_k.strip()] = _v.strip()

_edits = []
_recipe = None
if args.place_static_obstacles > 0:
    edit = {
        "tool": "place_static_obstacles",
        "count": int(args.place_static_obstacles),
        "seed": int(args.obstacle_seed),
        # Anchor placements at the replay hand-off so the policy has a
        # reaction window (obstacles inside the replayed stretch are unavoidable).
        "after_frame": int(args.ego_replay_frames),
        "ego_z_to_ground_m": float(args.obstacle_z_to_ground),
    }
    if args.obstacle_arc_positions:
        edit["arc_positions"] = [
            float(v) for v in args.obstacle_arc_positions.split(",") if v.strip()
        ]
    edit["types"] = [t.strip() for t in args.obstacle_types.split(",") if t.strip()]
    if _asset_map:
        edit["nurec_asset_ids"] = _asset_map
    _edits.append(edit)
    print(
        f"[navsafe-eval] scenario edit: {args.place_static_obstacles} static "
        f"obstacles on route (types={edit['types']}, seed={args.obstacle_seed}"
        + (f", nurec_assets={_asset_map}" if _asset_map else "")
        + ")",
        flush=True,
    )

if args.replace_agent_ids:
    _repl = {}
    for _spec in args.replace_agent_ids:
        _k, _, _v = _spec.partition("=")
        if not (_k.strip() and _v.strip()):
            ap.error(f"--replace-agent-ids expects TRACK_ID=ARCHETYPE, got {_spec!r}")
        _repl[_k.strip()] = _v.strip()
    _edits.append(
        {
            "tool": "replace_agent_with_asset",
            "replacements": _repl,
            "nurec_asset_ids": _asset_map,
            "ego_z_to_ground_m": float(args.obstacle_z_to_ground),
        }
    )
    print(
        f"[navsafe-eval] scenario edit: replacing agents {_repl} (nurec_assets={_asset_map})",
        flush=True,
    )

if args.relocate_agents:
    _reloc = {}
    for _spec in args.relocate_agents:
        _k, _, _v = _spec.partition(":")
        if not (_k.strip() and _v.strip()):
            ap.error(f"--relocate-agents expects TRACK_ID:key=val,..., got {_spec!r}")
        _d = {}
        for _kv in _v.split(","):
            _kk, _, _vv = _kv.partition("=")
            _kk, _vv = _kk.strip(), _vv.strip()
            if not (_kk and _vv):
                ap.error(f"--relocate-agents bad key=val in {_spec!r}")
            if _kk in ("arc", "speed", "lateral", "yaw_offset_deg"):
                _d[_kk] = float(_vv)
            elif _kk in ("mode", "archetype"):
                _d[_kk] = _vv
            else:
                ap.error(f"--relocate-agents unknown key {_kk!r} in {_spec!r}")
        _reloc[_k.strip()] = _d
    _edits.append(
        {
            "tool": "relocate_agent",
            "relocations": _reloc,
            "nurec_asset_ids": _asset_map,
            "after_frame": int(args.ego_replay_frames),
            "ego_z_to_ground_m": float(args.obstacle_z_to_ground),
        }
    )
    print(
        f"[navsafe-eval] scenario edit: relocating agents {_reloc} (nurec_assets={_asset_map})",
        flush=True,
    )

if args.edits_json:
    # A raw edit list, for exercising the runtime without a recipe. It
    # has no checksums or provenance, so its results must not be scored.
    import json as _json

    _raw = Path(args.edits_json)
    _edits.extend(_json.loads(_raw.read_text()) if _raw.is_file()
                  else _json.loads(args.edits_json))
    print(f"[navsafe-eval] --edits-json: {len(_edits)} raw edit(s); UNVERIFIED — "
          f"no recipe, no checksums, not for scoring", flush=True)

if args.recipe_dir and not args.recipe:
    # Select the recipe from the bundle, so that one sweep can mix edited
    # and unedited scenarios.
    from navsafe.benchmark.editing.autoselect import resolve_recipe

    _choice = resolve_recipe(args.py123d_data_root, recipe_dir=args.recipe_dir)
    print(
        f"[navsafe-eval] recipe auto-select: token={_choice.token} "
        f"manifest.has_inserted_actors={_choice.manifest_claim!r} -> "
        f"{'EDITED ' + _choice.path.name if _choice.edited else 'UNEDITED'} "
        f"({_choice.reason})",
        flush=True,
    )
    if _choice.edited:
        args.recipe = str(_choice.path)
        # A recipe's actors move only under navsafe traffic;
        # otherwise they would stand at their spawn poses for the
        # whole episode.
        if args.traffic_mode != "navsafe":
            print(
                f"[navsafe-eval] auto-select: --traffic-mode "
                f"{args.traffic_mode} -> navsafe, required by --recipe "
                f"(inserted actors do not move otherwise)",
                flush=True,
            )
            args.traffic_mode = "navsafe"
            # env_cfg was built from args at line ~672, before this block, so
            # the switch must reach it too — otherwise the env keeps the
            # original manager, which has no adopt() for the recipe's reactive
            # actors, and they stand at their spawn poses all episode.
            env_cfg.traffic_mode = "navsafe"

if args.recipe:
    from navsafe.benchmark.editing.recipe import edits_from_recipe_file

    # Checksums are verified here, before anything reaches the env: a corrupted
    # or hand-edited recipe must fail at load, not halfway through an episode.
    _recipe, _recipe_edits = edits_from_recipe_file(args.recipe, variant=args.recipe_variant)
    _takeover_ids = [a.source_track_id for a in _recipe.actors.values()
                     if a.op == "relocate" and a.keep_appearance
                     and a.policy.get("require_harvested_asset")]
    if _takeover_ids:
        if not os.environ.get("NUREC_GRPC_ASSET_REPLACE"):
            ap.error("Original-car takeover requires --asset-harvester-replace; tracks=" + ",".join(_takeover_ids))
        os.environ["NUREC_GRPC_ASSET_REPLACE_REPORT"] = str(Path(args.output_dir) / "harvester_takeover_audit.json")
    _edits.extend(_recipe_edits)
    if args.eval_frames is not None and int(args.eval_frames) <= 0:
        # A pure log-replay render has no policy and no hand-off: the ego
        # follows the log for the whole episode. Spawn poses are absolute, so
        # the edited scene still renders — the actors simply start where the
        # recipe put them and their controllers react to a logged ego. Forcing
        # the recipe's hand-off would truncate the render to its replay_frames.
        print(
            f"[navsafe-eval] recipe {_recipe.recipe_id}: pure replay render "
            f"(--eval-frames 0), keeping --ego-replay-frames "
            f"{args.ego_replay_frames} rather than the recipe's "
            f"{_recipe.ego.replay_frames}; the arrival window does not apply.",
            flush=True,
        )
    elif int(_recipe.ego.replay_frames) != int(args.ego_replay_frames):
        # Actor placement was solved relative to the recipe's hand-
        # off frame, so a different hand-off would invalidate it. A
        # longer replay is the exception, since the replayed ego
        # follows the trajectory the placement was solved against.
        if args.keep_ego_replay_frames:
            print(
                f"[navsafe-eval] recipe {_recipe.recipe_id} was baked with "
                f"ego.replay_frames={_recipe.ego.replay_frames}; keeping "
                f"--ego-replay-frames {args.ego_replay_frames} as requested "
                f"(--keep-ego-replay-frames).",
                flush=True,
            )
        else:
            print(
                f"[navsafe-eval] WARNING: recipe {_recipe.recipe_id} was baked with "
                f"ego.replay_frames={_recipe.ego.replay_frames} but this run passed "
                f"--ego-replay-frames {args.ego_replay_frames}; using the recipe's value, "
                f"because its placements were solved against that hand-off.",
                flush=True,
            )
            args.ego_replay_frames = int(_recipe.ego.replay_frames)
    print(
        f"[navsafe-eval] scenario edit: recipe {_recipe.recipe_id} [{args.recipe_variant}] "
        f"leaf={_recipe.leaf} host={_recipe.host.scene}@{_recipe.host.world_version} "
        f"actors={len(_recipe.actors)} T={_recipe.frames.T} dt={_recipe.frames.dt_s}",
        flush=True,
    )
    if not _recipe_edits:
        print(
            "[navsafe-eval] recipe variant edits nothing — the untouched host IS the control",
            flush=True,
        )

if _edits:
    env_cfg.scenario_edits = _edits

if args.no_real_assets:
    env_cfg.use_real_assets = False

if args.no_road:
    # Disable the whole map build (the env gate is `build_lanes or
    # build_lane_markings`), leaving only the grass ground + sky.
    env_cfg.build_lanes = False
    env_cfg.build_lane_markings = False
    env_cfg.build_crosswalks = False
    env_cfg.build_sidewalks = False
    env_cfg.build_boundaries = False

# NuRec camera resolution follows the evaluation configuration.
env_cfg.camera_resolution_scale = args.camera_resolution_scale

_src_desc = f"py123d:{args.py123d_data_root}"
# Scenario edits need the edit-aware subclass; the stock env stays pristine.
_env_cls = NavSafeEnv
if getattr(env_cfg, "scenario_edits", None):
    from navsafe.env import NavSafeEditEnv

    _env_cls = NavSafeEditEnv
print(
    f"[navsafe-eval] building {_env_cls.__name__} (backend={args.render_backend}) from {_src_desc}",
    flush=True,
)
env = _env_cls(env_cfg)

if args.hide_ego:
    # Hide the ego's visual mesh (it's mounted under .../ego_vehicle/ and shows
    # as a dark wedge from the on-board camera). The camera sensor and physics
    # are unaffected — only the rendered geometry under ego_vehicle is hidden.
    from pxr import UsdGeom

    _stage = env.sim.stage
    _hidden = 0
    for _prim in _stage.Traverse():
        _p = _prim.GetPath().pathString
        # The ego visual lives under .../agents/vehicle_ego/ (the camera mount
        # "ego_vehicle/chassis" is remapped to it). Hide its geometry prims.
        if ("vehicle_ego" in _p or "/ego_vehicle/" in _p) and _prim.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(_prim).MakeInvisible()
            _hidden += 1
    print(f"[navsafe-eval] hid {_hidden} ego visual meshes", flush=True)

adapter = create_model_adapter_from_args(args)
scorer = create_trajectory_scorer(args)

def _scenario_meta_for_config(args) -> dict:
    """``scenario_meta`` for the evaluator, or ``{}`` when unavailable.

    Never raises: a missing or malformed manifest must not stop a run, it just
    means the default scoring rules apply -- which is what every event type but I-2
    and C-3 gets anyway.
    """
    try:
        from navsafe.benchmark.scoring.from_run import scenario_meta_for
        return scenario_meta_for(args.py123d_data_root) or {}
    except Exception as exc:                                   # noqa: BLE001
        print(f"[navsafe-eval] scenario_meta unavailable ({exc}); "
              "default termination rules", flush=True)
        return {}


eval_config = EvaluationConfig(
    traffic_mode=args.traffic_mode,
    eval_mode=args.eval_mode,
    controller_type=args.controller,
    execution_mode=args.execution_mode,
    replan_rate=args.replan_rate,
    sim_dt=args.sim_dt,
    ego_replay_frames=args.ego_replay_frames,
    # None == indefinite: the evaluator stops on the taxonomy or the safety
    # ceiling (see EvaluatorConfig.eval_frames / SAFETY_CEILING_S).
    eval_frames=args.eval_frames,
    route_time_limit_s=args.route_time_limit_s,
    save_per_frame=True,
    output_dir=Path(args.output_dir),
    enable_vis=args.enable_vis,
    vis_online=False,
    vis_extra_cameras=tuple(
        c.strip().upper() for c in (args.vis_cameras or "").split(",") if c.strip()
    ),
    # Applied on the hand-off frame, which by this point is the RESOLVED one:
    # a recipe that overrides --ego-replay-frames (see the recipe block above)
    # has already rewritten args.ego_replay_frames, so "the first frame the
    # policy owns the ego" means the same thing on an edited scenario as on a
    # plain one.
    ego_perturb_lateral_m=args.ego_perturb_lateral,
    ego_perturb_longitudinal_m=args.ego_perturb_longitudinal,
    ego_perturb_yaw_deg=args.ego_perturb_yaw,
    ego_perturb_history=args.ego_perturb_history,
    record_reactivity_trace=args.record_reactivity_trace,
    reactivity_condition=args.reactivity_condition,
    reactivity_group_id=args.reactivity_group_id,
    reactivity_metadata={
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "recipe_path": str(args.recipe) if args.recipe else None,
        "recipe_variant": args.recipe_variant if args.recipe else None,
        "recipe_id": getattr(_recipe, "recipe_id", None),
        "leaf": getattr(_recipe, "leaf", None),
    },
    # The event type sets the termination rules of the live monitor. It
    # must be known before stepping, because an episode ended early
    # cannot be recovered by scoring again.
    scenario_meta=(_scenario_meta_for_config(args)
                   if args.scenario_source == "py123d" else None),
)

if (args.ego_perturb_lateral or args.ego_perturb_longitudinal
        or args.ego_perturb_yaw):
    print(f"[navsafe-eval] hand-off perturbation: "
          f"lateral={args.ego_perturb_lateral:+.2f} m "
          f"longitudinal={args.ego_perturb_longitudinal:+.2f} m "
          f"yaw={args.ego_perturb_yaw:+.1f} deg, applied on frame "
          f"{args.ego_replay_frames} (the resolved hand-off)", flush=True)

evaluator = Evaluator(
    env=env,
    model_adapter=adapter,
    config=eval_config,
    trajectory_scorer=scorer,
)
_setup_path = Path(args.py123d_data_root)
evaluator.setup(scenario_path=_setup_path)
# The EPDMS trajectory scorer needs the scenario + env before select_best;
# the Evaluator doesn't call this, so do it here (post-setup, scene loaded).
if scorer is not None and hasattr(scorer, "initialize"):
    scorer.initialize(env.current_scenario or env.get_scenario_info(), env)
results = evaluator.run()
print(f"[navsafe-eval] DONE. results={results}", flush=True)
print(f"[navsafe-eval] artifacts under {args.output_dir}", flush=True)

# NavSafe metrics (driving score, success, efficiency, comfort), computed
# from the stored artifacts by the same chain as `navsafe score`. A scoring
# failure does not discard the finished rollout. `_navsafe_scored` lists the
# directories that received a navsafe_metrics.json; only those are pruned.
_navsafe_scored: list[Path] = []
if args.navsafe_metrics:
    try:
        import json as _json

        from navsafe.benchmark.scoring.from_run import score_run as _score_run
        from navsafe.benchmark.scoring.from_run import scenario_meta_for
        from navsafe.benchmark.trace import from_eval as _from_eval

        _sd = env.current_scenario
        if not _sd:
            raise RuntimeError("env has no current_scenario to score against")
        # The evaluator writes artifacts directly into --output-dir.
        # Subdirectories are also scanned, for runs written with the
        # older nested layout.
        _root = Path(args.output_dir)
        _eval_dirs = ([_root] if (_root / "vehicle_states.npy").exists()
                      else [p for p in _root.iterdir()
                            if (p / "vehicle_states.npy").exists()])
        if not _eval_dirs:
            raise RuntimeError(
                f"no vehicle_states.npy in {args.output_dir} or its subdirectories")
        _live_termination = str(
            ((results or {}).get("termination") or {}).get("reason")
            or (results or {}).get("metrics", {}).get("termination_reason")
            or "")
        for _eval_dir in _eval_dirs:
            _run = _from_eval.load_with_scenario(
                _eval_dir, _sd, warmup_frames=args.ego_replay_frames, dt=0.1)
            _navsafe_metrics = _score_run(
                _run, name=_eval_dir.name,
                warmup_frames=args.ego_replay_frames, dt=0.1,
                source=str(_eval_dir),
                # Infinity means the run had no route time
                # limit; scoring would otherwise assume 60 s.
                t_max=(float("inf") if eval_config.route_time_limit_s is None
                       else eval_config.route_time_limit_s),
                # Taxonomy travels with the result: a metrics file that cannot
                # say which event type it scored is not much use months later.
                scenario_meta=scenario_meta_for(args.py123d_data_root),
                # Pass on why the evaluator stopped. From the
                # stored trace alone, a renderer failure mid-
                # episode cannot be told apart from frames
                # running out.
                live_termination=(_live_termination
                                  if len(_eval_dirs) == 1 else ""))
            if args.navsafe_prune_artifacts:
                # Mark, never drop: the inputs this score was computed from are
                # about to go, and a later reader must not assume they were
                # never there.
                _navsafe_metrics.setdefault("notes", []).append(
                    "metrics.json, driving_score_summary.csv and run_meta.json "
                    "were pruned after scoring (--navsafe-prune-artifacts); "
                    "re-scoring this directory loses the per-frame EPDMS flags "
                    "(contacts fall back to the run total) and the TL column "
                    "the red_light channel is measured from")
            (_eval_dir / "navsafe_metrics.json").write_text(
                _json.dumps(_navsafe_metrics, indent=2))
            _navsafe_scored.append(_eval_dir)
            _m = _navsafe_metrics["metrics"]
            _ds = _m["driving_score"]
            print(f"[navsafe-eval] navsafe: DS={_ds if _ds is not None else '—'} "
                  f"SR={_m['success']} efficiency={_m['efficiency_pct']} "
                  f"comfort={_m['comfort']} ({_navsafe_metrics['status']}, "
                  f"{_navsafe_metrics['termination']['reason']}) -> {_eval_dir}/navsafe_metrics.json",
                  flush=True)
    except Exception as _metrics_exc:  # noqa: BLE001
        print(f"[navsafe-eval] WARNING: navsafe scoring failed, no "
              f"navsafe_metrics.json written: "
              f"{type(_metrics_exc).__name__}: {_metrics_exc}", flush=True)

# Provenance: git SHA + working diff + resolved args next to the results.
# Written after the run so a crash cannot leave a run_meta.json claiming a
# result that never finished; best-effort and never fatal. Skipped outright
# under --navsafe-prune-artifacts rather than written and then deleted.
if not (args.navsafe_prune_artifacts and _navsafe_scored):
    try:
        from navsafe.utils.provenance import write_provenance

        write_provenance(args.output_dir, args, {"results": results})
    except Exception as _prov_exc:  # noqa: BLE001
        print(f"[navsafe-eval] provenance capture failed: {_prov_exc}", flush=True)

# Prune the other metric families' artifacts, in the directories that were
# actually scored. Not in the list: vehicle_states.npy and trajectory.npy (the
# rollout itself), frames/ and visualization/ (the render).
if args.navsafe_prune_artifacts:
    if not _navsafe_scored:
        print("[navsafe-eval] not pruning: no navsafe_metrics.json was written, "
              "so metrics.json is the only score this run has", flush=True)
    # Scored directories lose all three; the run root additionally loses a
    # run_meta.json, because provenance is written there and the scored
    # directory can be a child of it (the pre-flattening layout).
    _to_remove = [d / n for d in _navsafe_scored
                  for n in ("metrics.json", "driving_score_summary.csv",
                            "run_meta.json")]
    if _navsafe_scored:
        _to_remove.append(Path(args.output_dir) / "run_meta.json")
    for _p in _to_remove:
        try:
            _p.unlink(missing_ok=True)
        except OSError as _rm_exc:         # read-only mount, etc.
            print(f"[navsafe-eval] could not remove {_p}: {_rm_exc}", flush=True)
    if _navsafe_scored:
        print("[navsafe-eval] pruned metrics.json, driving_score_summary.csv "
              "and run_meta.json (--navsafe-prune-artifacts)", flush=True)

if args.wandb:
    from navsafe.utils.wandb_logger import finish, init_wandb, log_metrics

    _run = init_wandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name or Path(args.output_dir).name,
        entity=args.wandb_entity,
        config=vars(args),
        job_type="eval",
        tags=[args.model_type, args.render_backend],
    )
    _metrics = results.get("metrics", {}) if isinstance(results, dict) else {}
    log_metrics(_run, {f"eval/{k}": v for k, v in _metrics.items()})
    finish(_run)

# Release env resources BEFORE the app: the renderer's close() undoes
# server-side edit_assets state (nurec_grpc restore_model_parameters) — skip
# it and inserted assets leak into later evals on the warm shared server.
try:
    env.close()
except Exception as _exc:  # noqa: BLE001 — teardown must not mask results
    print(f"[navsafe-eval] env.close() failed: {_exc}", flush=True)
simulation_app.close()
