#!/usr/bin/env python3
"""
Unified Evaluator - Single CLI entry point for all models in UrbanSim.

Adapted from BridgeSim's unified_evaluator.py to use UrbanSim's
Evaluator and model adapter infrastructure.

Usage:
    python -m navsafe.evaluation.unified_evaluator \
        --model-type tcp \
        --checkpoint /path/to/checkpoint.ckpt \
        --scenario-path /path/to/scenario \
        --output-dir /path/to/output
"""

import argparse
import dataclasses
import sys
import logging
from pathlib import Path

from navsafe.evaluation.evaluator import Evaluator, EvaluationConfig
from navsafe.policy import create_model_adapter
from navsafe.env import NavSafeEnv, EnvCfg
from navsafe.env.presets import REPLAY
from navsafe.evaluation.scorers import (
    ConfidenceScorer,
    CoarseTopKScorer,
    EPDMSTrajectoryScorer_Fast,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_trajectory_scorer(args):
    """Create trajectory scorer based on args. Returns None if no scorer requested."""
    scorer_name = args.trajectory_scorer
    if scorer_name is None:
        return None

    model_type = args.model_type.lower()

    if scorer_name == "confidence":
        return ConfidenceScorer()

    elif scorer_name == "coarse_topk":
        v2_ckpt = args.v2_scorer_checkpoint
        if model_type == "diffusiondrive" and v2_ckpt is None:
            raise ValueError(
                "--v2-scorer-checkpoint is required when using "
                "--trajectory-scorer coarse_topk with DiffusionDrive v1."
            )
        return CoarseTopKScorer(v2_scorer_checkpoint_path=v2_ckpt, device="cuda")

    elif scorer_name == "epdms_fast":
        return EPDMSTrajectoryScorer_Fast()

    elif scorer_name in ("epdms", "epdms_ego"):
        raise ValueError(
            f"trajectory scorer {scorer_name!r} was deleted in the scorer "
            "consolidation (see simplify.md) — use 'epdms_fast'"
        )

    else:
        raise ValueError(f"Unknown trajectory scorer: {scorer_name}")


#: Adapters whose ``parse_output`` publishes ``trajectory_coarse`` /
#: ``coarse_scores`` when ``num_proposals`` is set. Others take no such
#: argument and would raise on it.
_VOCABULARY_ADAPTERS = frozenset(
    {"drivor", "diffusiondrive", "diffusiondrivev2", "rap"}
)


def create_model_adapter_from_args(args):
    """Create appropriate model adapter based on parsed args."""
    model_type = args.model_type.lower()

    kwargs = {}
    if args.plan_anchor_path:
        kwargs["plan_anchor_path"] = args.plan_anchor_path
    if args.enable_temporal_consistency:
        kwargs.update(
            {
                "enable_temporal_consistency": True,
                "temporal_alpha": args.temporal_alpha,
                "temporal_lambda": args.temporal_lambda,
                "temporal_max_history": args.temporal_max_history,
                "temporal_sigma": args.temporal_sigma,
                "consensus_temperature": args.consensus_temperature,
            }
        )
    # Candidate logging is opt-in because it is the only thing that makes the
    # proposal set observable: the vocabulary adapters (drivor, diffusiondrive,
    # diffusiondrivev2, rap) publish ``trajectory_coarse``/``coarse_scores``
    # ONLY when num_proposals is set, and otherwise emit the argmax alone. A
    # failure case then cannot distinguish "no safe proposal existed" from "a
    # safe proposal was ranked below the one executed".
    num_proposals = getattr(args, "num_proposals", None)
    if num_proposals and model_type in _VOCABULARY_ADAPTERS:
        kwargs["num_proposals"] = int(num_proposals)
    elif num_proposals:
        print(f"[eval] --num-proposals ignored: {model_type} has no candidate set",
              flush=True)

    adapter = create_model_adapter(
        model_type=model_type,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        **kwargs,
    )
    adapter.load_model()
    return adapter


def main():
    parser = argparse.ArgumentParser(
        description="UrbanSim Unified Evaluator - single-scenario evaluation entry point"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=[
            "uniad",
            "vad",
            "tcp",
            "rap",
            "lead",
            "lead_navsim",
            "drivor",
            "transfuser",
            "ltf",
            "egomlp",
            "ego_mlp",
            "diffusiondrive",
            "diffusiondrivev2",
            "openpilot",
            "alpamayo_r1",
            "pdm_closed",
        ],
        help="Model type",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to model config file")
    parser.add_argument("--plan-anchor-path", type=str, default=None)
    parser.add_argument(
        "--scenario-path",
        type=str,
        default=None,
        help="Path to a ScenarioNet ScenarioDescription (pickle/dir). "
        "Mutually exclusive with --py123d-data-root.",
    )
    parser.add_argument(
        "--py123d-data-root",
        type=str,
        default=None,
        help="Root of converted py123d Arrow logs (contains logs/ and maps/). "
        "When set, the env reads the Arrow scene directly "
        "(scenario_source='py123d') — no pickle.",
    )
    parser.add_argument(
        "--py123d-scene-index",
        type=int,
        default=0,
        help="Index of the py123d scene to load (start_scenario_index).",
    )
    parser.add_argument(
        "--traffic-mode",
        type=str,
        default="log_replay",
        choices=["no_traffic", "log_replay", "semi_reactive"],
        help="'idm' was removed: it was an unwired silent no-op (agents "
             "replayed the log); semi_reactive is the wired reactive mode.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save-perframe", action="store_true", default=True)
    parser.add_argument("--no-save-perframe", dest="save_perframe", action="store_false")
    parser.add_argument(
        "--controller", type=str, default="pure_pursuit", choices=["pid", "pure_pursuit"]
    )
    parser.add_argument("--replan-rate", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.1)
    parser.add_argument("--ego-replay-frames", type=int, default=0)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--score-start-frame", type=int, default=None,
                        help="REFUSED if set: never wired to anything. "
                             "Scoring starts at the warm-up hand-off — use "
                             "--ego-replay-frames.")
    parser.add_argument(
        "--eval-mode", type=str, default="closed_loop", choices=["closed_loop"],
        help="Only closed_loop exists; the historical 'open_loop' value was "
             "parsed but never implemented (use --ego-replay-frames >= "
             "--eval-frames for a pure log-replay run).",
    )
    parser.add_argument("--enable-vis", action="store_true")

    # Trajectory scoring
    parser.add_argument(
        "--trajectory-scorer",
        type=str,
        default=None,
        choices=["confidence", "coarse_topk", "epdms_fast"],
    )
    parser.add_argument("--v2-scorer-checkpoint", type=str, default=None)

    # Temporal consistency (DiffusionDriveV2)
    parser.add_argument("--enable-temporal-consistency", action="store_true")
    parser.add_argument("--temporal-alpha", type=float, default=1.5)
    parser.add_argument("--temporal-lambda", type=float, default=0.3)
    parser.add_argument("--temporal-max-history", type=int, default=8)
    parser.add_argument("--temporal-sigma", type=float, default=5.0)
    parser.add_argument("--consensus-temperature", type=float, default=1.0)

    args = parser.parse_args()

    if args.score_start_frame is not None:
        parser.error(
            "--score-start-frame was never wired to anything; scoring starts "
            "at the warm-up hand-off — use --ego-replay-frames instead.")

    # Build UrbanSim EvaluationConfig
    eval_config = EvaluationConfig(
        traffic_mode=args.traffic_mode,
        eval_mode=args.eval_mode,
        controller_type=args.controller,
        replan_rate=args.replan_rate,
        sim_dt=args.sim_dt,
        ego_replay_frames=args.ego_replay_frames,
        eval_frames=args.eval_frames,
        save_per_frame=args.save_perframe,
        output_dir=Path(args.output_dir),
    )

    try:
        # Create UrbanSim environment from the REPLAY preset, mapping the
        # legacy ScenarioReplayEnvCfg overrides 1:1 onto EnvCfg:
        #   .data_directory -> data_directory (parent of scenario path)
        #   .loop_replay    -> loop_replay (False)
        #   .sim.dt         -> dt (args.sim_dt)
        #   .spawn_ego_vehicle -> spawn_ego_vehicle (True)
        # The legacy path supplied a user scenario directly, so it never
        # required the preset's asset bundles; clear required_bundles to
        # match that behavior and avoid a spurious FileNotFoundError.
        # Source selection: py123d Arrow (read directly, no pickle) or a
        # ScenarioNet ScenarioDescription via --scenario-path. Exactly one.
        if bool(args.py123d_data_root) == bool(args.scenario_path):
            parser.error("provide exactly one of --py123d-data-root or --scenario-path")

        if args.py123d_data_root:
            env_cfg = dataclasses.replace(
                REPLAY,
                scenario_source="py123d",
                py123d_data_root=args.py123d_data_root,
                py123d_max_scenes=1,
                loop_replay=False,
                dt=args.sim_dt,
                spawn_ego_vehicle=True,
                required_bundles=[],
            )
            # start_scenario_index is a loader getattr-default, not an EnvCfg
            # field, so it is attached to the instance rather than passed to
            # dataclasses.replace.
            env_cfg.start_scenario_index = args.py123d_scene_index
        else:
            env_cfg = dataclasses.replace(
                REPLAY,
                data_directory=str(Path(args.scenario_path).parent),
                scenario_path=str(args.scenario_path),
                loop_replay=False,
                dt=args.sim_dt,
                spawn_ego_vehicle=True,
                required_bundles=[],
            )
        env = NavSafeEnv(env_cfg)

        # Load model adapter
        model_adapter = create_model_adapter_from_args(args)

        # Optionally attach trajectory scorer
        trajectory_scorer = create_trajectory_scorer(args)

        # Run evaluation
        evaluator = Evaluator(
            env=env,
            model_adapter=model_adapter,
            config=eval_config,
            trajectory_scorer=trajectory_scorer,
        )
        evaluator.setup(scenario_path=Path(args.scenario_path))
        results = evaluator.run()
        logger.info(f"Evaluation complete. Results: {results}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
