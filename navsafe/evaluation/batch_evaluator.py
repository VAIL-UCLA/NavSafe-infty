"""
Batch evaluator for running evaluation on multiple scenarios.

Adapted from BridgeSim's batch_evaluator.py to use UrbanSim's in-process
Evaluator instead of subprocess calls.

Features:
- Process all scenarios in a directory
- Progress tracking with ETA
- Error handling and recovery
- Result aggregation
- Resume capability
"""

import json
import csv
import dataclasses
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from navsafe.evaluation.evaluator import Evaluator, EvaluationConfig
from navsafe.policy import create_model_adapter
# NavSafeEnv / EnvCfg / REPLAY imported lazily in _get_env() to avoid
# triggering IsaacSim initialization before the env is actually needed.

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """Batch evaluator for multiple scenarios using UrbanSim's Evaluator in-process."""

    def __init__(
        self,
        model_type: str,
        checkpoint_path: str,
        scenario_root: str,
        output_root: str,
        config_path: Optional[str] = None,
        plan_anchor_path: Optional[str] = None,
        traffic_mode: str = "log_replay",
        max_workers: int = 1,
        resume: bool = False,
        save_perframe: bool = True,
        controller_type: str = "pure_pursuit",
        replan_rate: int = 1,
        sim_dt: float = 0.1,
        ego_replay_frames: int = 0,
        eval_frames: Optional[int] = None,
        score_start_frame: Optional[int] = None,
        eval_mode: str = "closed_loop",
        enable_vis: bool = False,
        max_scenarios: Optional[int] = None,
        enable_temporal_consistency: bool = False,
        temporal_alpha: float = 1.5,
        temporal_lambda: float = 0.3,
        temporal_max_history: int = 8,
        temporal_sigma: float = 5.0,
        consensus_temperature: float = 1.0,
    ):
        if score_start_frame is not None:
            # Parsed for years but never wired to anything: accepting a value
            # silently would misdescribe the run. Scoring starts at the
            # warm-up hand-off.
            raise ValueError(
                "score_start_frame is not wired to anything — scoring starts "
                "at the warm-up hand-off. Use ego_replay_frames "
                "(--ego-replay-frames) to set it.")
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.plan_anchor_path = plan_anchor_path
        self.scenario_root = Path(scenario_root)
        self.output_root = Path(output_root)
        self.traffic_mode = traffic_mode
        self.max_workers = max_workers
        self.resume = resume
        self.save_perframe = save_perframe
        self.controller_type = controller_type
        self.replan_rate = replan_rate
        self.sim_dt = sim_dt
        self.ego_replay_frames = ego_replay_frames
        self.eval_frames = eval_frames
        self.score_start_frame = score_start_frame
        self.eval_mode = eval_mode
        self.enable_vis = enable_vis
        self.max_scenarios = max_scenarios
        self.enable_temporal_consistency = enable_temporal_consistency
        self.temporal_alpha = temporal_alpha
        self.temporal_lambda = temporal_lambda
        self.temporal_max_history = temporal_max_history
        self.temporal_sigma = temporal_sigma
        self.consensus_temperature = consensus_temperature

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_root / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Result tracking
        self.results: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "config": {
                "model_type": model_type,
                "checkpoint_path": checkpoint_path,
                "scenario_root": str(scenario_root),
                "output_root": str(output_root),
                "traffic_mode": traffic_mode,
                "max_workers": max_workers,
                "save_perframe": save_perframe,
                "replan_rate": replan_rate,
                "sim_dt": sim_dt,
                "ego_replay_frames": ego_replay_frames,
                "eval_frames": eval_frames,
                "score_start_frame": score_start_frame,
                "eval_mode": eval_mode,
                "enable_vis": enable_vis,
            },
            "scenarios": {},
        }

        self.results_file = self.output_root / "batch_results.json"
        if self.resume and self.results_file.exists():
            with open(self.results_file, "r") as f:
                prev_results = json.load(f)
                self.results["scenarios"] = prev_results.get("scenarios", {})
            logger.info(f"Resuming from previous run with {len(self.results['scenarios'])} completed scenarios")

        # Build EvaluationConfig for UrbanSim's Evaluator
        self.eval_config = EvaluationConfig(
            traffic_mode=traffic_mode,
            eval_mode=eval_mode,
            controller_type=controller_type,
            replan_rate=replan_rate,
            sim_dt=sim_dt,
            ego_replay_frames=ego_replay_frames,
            eval_frames=eval_frames,
            save_per_frame=save_perframe,
            output_dir=self.output_root,
            enable_vis=enable_vis,
        )

        # UrbanSim env and model adapter — created lazily, shared across scenarios
        self._env = None
        self._model_adapter = None
        self._evaluator: Optional[Evaluator] = None

    def _get_env(self):
        """Lazily create the consolidated NavSafeEnv (shared across all scenarios)."""
        if self._env is None:
            from navsafe.env import NavSafeEnv, EnvCfg
            from navsafe.env.presets import REPLAY
            # Build an EnvCfg from the REPLAY preset, overriding the same fields
            # the legacy ScenarioReplayEnvCfg path set:
            #   data_directory      <- scenario_root
            #   loop_replay = False
            #   spawn_ego_vehicle = True (Evaluator drives ego via actions)
            #   sim.dt              <- dt (sim_dt maps to EnvCfg.dt)
            # required_bundles is cleared: a user-supplied scenario root does not
            # require an asset bundle, matching the legacy behavior.
            env_cfg = dataclasses.replace(
                REPLAY,
                data_directory=str(self.scenario_root),
                loop_replay=False,
                spawn_ego_vehicle=True,
                dt=self.sim_dt,
                required_bundles=[],
                # The requested traffic mode must reach the env: it used to be
                # recorded into batch_results.json / metrics.json but never
                # passed here, so EVERY value silently ran the REPLAY preset's
                # log_replay while the artifacts claimed otherwise.
                traffic_mode=self.traffic_mode,
            )
            self._env = NavSafeEnv(env_cfg)
            logger.info("NavSafeEnv created")
        return self._env

    def _get_model_adapter(self):
        """Lazily initialize the model adapter."""
        if self._model_adapter is None:
            kwargs = {}
            if self.plan_anchor_path:
                kwargs["plan_anchor_path"] = self.plan_anchor_path
            if self.enable_temporal_consistency:
                kwargs["enable_temporal_consistency"] = True
                kwargs["temporal_alpha"] = self.temporal_alpha
                kwargs["temporal_lambda"] = self.temporal_lambda
                kwargs["temporal_max_history"] = self.temporal_max_history
                kwargs["temporal_sigma"] = self.temporal_sigma
                kwargs["consensus_temperature"] = self.consensus_temperature

            self._model_adapter = create_model_adapter(
                model_type=self.model_type,
                checkpoint_path=self.checkpoint_path,
                config_path=self.config_path,
                **kwargs,
            )
            self._model_adapter.load_model()
            logger.info(f"Loaded model adapter: {self._model_adapter}")
        return self._model_adapter

    def _get_evaluator(self) -> Evaluator:
        """Lazily build the Evaluator (env + adapter created once, shared across scenarios)."""
        if self._evaluator is None:
            self._evaluator = Evaluator(
                env=self._get_env(),
                model_adapter=self._get_model_adapter(),
                config=self.eval_config,
            )
            logger.info("Evaluator created")
        return self._evaluator

    def get_scenarios(self) -> List[Path]:
        """Get list of scenario directories."""
        scenarios = [
            d for d in sorted(self.scenario_root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]

        if self.resume:
            scenarios = [
                s for s in scenarios
                if s.name not in self.results["scenarios"]
                or self.results["scenarios"][s.name].get("status") != "success"
            ]
            if scenarios:
                logger.info(f"Found {len(scenarios)} scenarios to evaluate (after filtering completed)")

        if self.max_scenarios is not None:
            scenarios = scenarios[:self.max_scenarios]

        return scenarios

    def evaluate_scenario(self, scenario_path: Path) -> Dict[str, Any]:
        """Evaluate a single scenario using the shared UrbanSim Evaluator.

        The UrbanSim environment and model adapter are created once on the first
        call and reused for every subsequent scenario.
        """
        scenario_name = scenario_path.name
        log_file = self.log_dir / f"{scenario_name}.log"

        start_time = time.time()
        try:
            evaluator = self._get_evaluator()

            # Point the evaluator's output dir at this scenario
            evaluator.config.output_dir = self.output_root

            evaluator.setup(scenario_path=scenario_path)
            eval_results = evaluator.run()

            duration = time.time() - start_time

            # An episode the benchmark ended publishes scorable=false in its
            # metrics: it must be counted and surfaced, never averaged in as
            # a healthy run. Two distinct endings hide behind that flag:
            #   * infra_failure — the evaluator/simulator BROKE mid-episode.
            #     That is a harness failure, not a benchmark verdict: status
            #     "error" (same bucket as an exception here), and an operator
            #     reading the summary sees breakage.
            #   * everything else (envelope_exit, ...) — a valid run the
            #     benchmark ended: status "unscorable", excluded from every
            #     aggregate, reported as a count.
            # NB --resume re-runs every non-"success" scenario, so BOTH
            # statuses are retried (records overwrite idempotently); for a
            # deterministic envelope_exit that re-run is redundant compute,
            # accepted to keep resume semantics simple.
            # `success` therefore means "completed AND scorable"; aggregation
            # below only reads `success` records.
            metrics = (eval_results.get("metrics") or {}
                       if isinstance(eval_results, dict) else {})
            if metrics.get("scorable", True):
                status = "success"
            elif metrics.get("termination_reason") == "infra_failure":
                status = "error"
            else:
                status = "unscorable"

            with open(log_file, "w") as f:
                f.write(f"Scenario: {scenario_name}\nStatus: {status}\nDuration: {duration:.1f}s\n")
                f.write(json.dumps(eval_results, indent=2, default=str))

            record = {
                "status": status,
                "duration": duration,
                "log_file": str(log_file),
                "results": eval_results,
                "timestamp": datetime.now().isoformat(),
            }
            if status != "success":
                record["termination_reason"] = str(
                    metrics.get("termination_reason", "unknown"))
                if "infra_failure_error" in metrics:
                    record["error"] = str(metrics["infra_failure_error"])
                logger.warning(
                    "Scenario %s is %s (%s) — excluded from every aggregate, "
                    "reported as a count", scenario_name, status.upper(),
                    record["termination_reason"])
            return record

        except Exception as e:
            duration = time.time() - start_time
            import traceback
            tb = traceback.format_exc()
            with open(log_file, "w") as f:
                f.write(f"Scenario: {scenario_name}\nError: {e}\n\n{tb}")
            logger.error(f"Scenario {scenario_name} failed: {e}")
            return {
                "status": "error",
                "duration": duration,
                "error": str(e),
                "log_file": str(log_file),
                "timestamp": datetime.now().isoformat(),
            }

    def save_results(self):
        """Save results to JSON file."""
        self.results["end_time"] = datetime.now().isoformat()

        total = len(self.results["scenarios"])
        success = sum(1 for r in self.results["scenarios"].values() if r["status"] == "success")
        failed = sum(1 for r in self.results["scenarios"].values() if r["status"] == "failed")
        error = sum(1 for r in self.results["scenarios"].values() if r["status"] == "error")
        timeout = sum(1 for r in self.results["scenarios"].values() if r["status"] == "timeout")
        unscorable = sum(
            1 for r in self.results["scenarios"].values()
            if r["status"] == "unscorable")

        self.results["summary"] = {
            "total": total,
            "success": success,
            "failed": failed,
            "error": error,
            "timeout": timeout,
            # Episodes the benchmark ended (scorable=false in metrics.json):
            # counted here, excluded from every aggregate below.
            "unscorable": unscorable,
            "success_rate": success / total if total > 0 else 0.0,
        }

        if self.results["scenarios"]:
            total_duration = sum(r.get("duration", 0) for r in self.results["scenarios"].values())
            self.results["summary"]["total_duration_seconds"] = total_duration
            self.results["summary"]["average_duration_seconds"] = total_duration / total

        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def aggregate_all_results(self):
        """Collect and aggregate results from all successful scenarios.

        Unscorable episodes (``status == "unscorable"``) are excluded from
        every aggregate and reported as a count — never averaged in.
        """
        unscorable = [name for name, r in self.results["scenarios"].items()
                      if r["status"] == "unscorable"]
        if unscorable:
            logger.warning(
                "%d unscorable scenario(s) excluded from aggregation: %s",
                len(unscorable), ", ".join(sorted(unscorable)))
        result_files = []
        for scenario_name, result in self.results["scenarios"].items():
            if result["status"] == "success":
                results_json = self.output_root / scenario_name / "evaluation_results.json"
                if results_json.exists():
                    result_files.append(results_json)

        if not result_files:
            logger.warning("No successful scenarios to aggregate.")
        else:
            aggregated_output = self.output_root / "aggregated_results.json"
            logger.info(
                f"{'='*60}\n"
                f"Aggregating Results from {len(result_files)} Scenarios\n"
                f"{'='*60}")
            aggregate_results(result_files, str(aggregated_output))

        self.export_results_csv()

    def export_results_csv(self):
        """Export comprehensive CSV with legacy driving scores and EPDMS scores."""
        csv_path = self.output_root / "evaluation_summary.csv"

        columns = [
            "scenario_name",
            "driving_score", "route_completion", "infraction_penalty",
            "epdms_score", "no_at_fault_collisions", "drivable_area_compliance",
            "driving_direction_compliance", "traffic_light_compliance",
            "time_to_collision", "lane_keeping", "history_comfort", "extended_comfort",
        ]

        rows = []
        legacy_totals = {"driving_score": [], "route_completion": [], "infraction_penalty": []}
        epdms_totals = {
            "epdms_score": [], "no_at_fault_collisions": [], "drivable_area_compliance": [],
            "driving_direction_compliance": [], "traffic_light_compliance": [],
            "time_to_collision": [], "lane_keeping": [], "history_comfort": [], "extended_comfort": [],
        }

        for scenario_name, result in self.results["scenarios"].items():
            if result["status"] != "success":
                continue

            scenario_dir = self.output_root / scenario_name
            row = {"scenario_name": scenario_name}

            # Load legacy scores from evaluation_results.json
            results_json = scenario_dir / "evaluation_results.json"
            if results_json.exists():
                with open(results_json, "r") as f:
                    data = json.load(f)
                    if "_checkpoint" in data and "records" in data["_checkpoint"]:
                        record = data["_checkpoint"]["records"][0]
                        scores = record.get("scores", {})
                        row["driving_score"] = scores.get("score_composed", 0.0)
                        row["route_completion"] = scores.get("score_route", 0.0)
                        row["infraction_penalty"] = scores.get("score_penalty", 0.0)
                        for key in ["driving_score", "route_completion", "infraction_penalty"]:
                            if key in row:
                                legacy_totals[key].append(row[key])

            # Load EPDMS scores
            epdms_files = list(scenario_dir.glob("*_closedloop_epdms_summary.csv"))
            if epdms_files:
                with open(epdms_files[0], "r") as f:
                    reader = csv.DictReader(f)
                    for epdms_row in reader:
                        row["epdms_score"] = float(epdms_row.get("final_score", 0.0))
                        row["no_at_fault_collisions"] = float(epdms_row.get("mean_no_at_fault_collisions", 0.0))
                        row["drivable_area_compliance"] = float(epdms_row.get("mean_drivable_area_compliance", 0.0))
                        row["driving_direction_compliance"] = float(epdms_row.get("mean_driving_direction_compliance", 0.0))
                        row["traffic_light_compliance"] = float(epdms_row.get("mean_traffic_light_compliance", 0.0))
                        row["time_to_collision"] = float(epdms_row.get("mean_time_to_collision", 0.0))
                        row["lane_keeping"] = float(epdms_row.get("mean_lane_keeping", 0.0))
                        row["history_comfort"] = float(epdms_row.get("mean_history_comfort", 0.0))
                        row["extended_comfort"] = float(epdms_row.get("mean_extended_comfort", 0.0))
                        for key in epdms_totals:
                            if key in row:
                                epdms_totals[key].append(row[key])
                        break

            rows.append(row)

        avg_row = {"scenario_name": "AVERAGE"}
        for key, values in {**legacy_totals, **epdms_totals}.items():
            avg_row[key] = np.mean(values) if values else 0.0

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in columns})
            writer.writerow({k: "" for k in columns})
            writer.writerow({
                k: f"{avg_row.get(k, ''):.6f}" if isinstance(avg_row.get(k), float) else avg_row.get(k, "")
                for k in columns
            })

        logger.info(f"Exported evaluation summary CSV to: {csv_path}")
        logger.info(
            f"{'='*70}\n"
            "Summary (Averages)\n"
            f"{'='*70}\n"
            f"  Driving Score:        {avg_row.get('driving_score', 0):.4f}\n"
            f"  Route Completion:     {avg_row.get('route_completion', 0):.4f}%\n"
            f"  Infraction Penalty:   {avg_row.get('infraction_penalty', 0):.4f}\n"
            f"  EPDMS Score:          {avg_row.get('epdms_score', 0):.4f}\n"
            f"{'='*70}")

    def print_summary(self):
        """Print evaluation summary."""
        summary = self.results.get("summary", {})

        logger.info(
            f"{'='*60}\n"
            "Batch Evaluation Complete!\n"
            f"{'='*60}\n"
            f"Total scenarios: {summary.get('total', 0)}\n"
            f"  Success: {summary.get('success', 0)}\n"
            f"  Failed:  {summary.get('failed', 0)}\n"
            f"  Error:   {summary.get('error', 0)}\n"
            f"  Timeout: {summary.get('timeout', 0)}\n"
            f"  Unscorable (excluded from every mean): {summary.get('unscorable', 0)}\n"
            f"Success rate: {summary.get('success_rate', 0):.1%}")

        if "total_duration_seconds" in summary:
            total_hours = summary["total_duration_seconds"] / 3600
            avg_seconds = summary.get("average_duration_seconds", 0)
            logger.info(f"Total duration: {total_hours:.2f} hours")
            logger.info(f"Average per scenario: {avg_seconds:.1f} seconds")

        logger.info(
            f"Results saved to: {self.results_file}\n"
            f"Logs saved to: {self.log_dir}\n"
            f"{'='*60}")

    def run(self):
        """Run batch evaluation."""
        scenarios = self.get_scenarios()

        if not scenarios:
            logger.warning("No scenarios to evaluate!")
            return

        logger.info(
            f"{'='*60}\n"
            "Starting Batch Evaluation\n"
            f"{'='*60}\n"
            f"Total scenarios: {len(scenarios)}\n"
            f"Model: {self.model_type}\n"
            f"Output: {self.output_root}\n"
            f"{'='*60}")

        for scenario_path in tqdm(scenarios, desc="Evaluating scenarios"):
            scenario_name = scenario_path.name

            result = self.evaluate_scenario(scenario_path)
            self.results["scenarios"][scenario_name] = result

            self.save_results()

            status_symbol = {
                "success": "✓",
                "failed": "✗",
                "error": "⚠",
                "timeout": "⏱",
                "unscorable": "—",
            }.get(result["status"], "?")

            tqdm.write(f"{status_symbol} {scenario_name}: {result['status']} ({result.get('duration', 0):.1f}s)")

        self.print_summary()
        self.aggregate_all_results()


def aggregate_results(result_files: List[Path], output_path: str) -> None:
    """
    Aggregate individual scenario results into a final JSON.

    Computes global statistics from multiple scenario evaluation results,
    including driving scores, infractions per km, and planning metrics (L2 errors).
    """
    if not result_files:
        logger.warning("No results to aggregate")
        return

    records = []
    for idx, result_file in enumerate(result_files):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
                if "_checkpoint" in data and "records" in data["_checkpoint"]:
                    record = data["_checkpoint"]["records"][0]
                    record["index"] = idx
                    records.append(record)
        except Exception as e:
            logger.warning(f"Error loading {result_file}: {e}")

    if not records:
        logger.warning("No valid records found")
        return

    total_scenarios = len(records)

    # Aggregate infractions
    infractions_sum: Dict[str, Any] = {}
    for record in records:
        for key, value in record.get("infractions", {}).items():
            infractions_sum[key] = infractions_sum.get(key, 0) + (len(value) if isinstance(value, list) else value)

    # Aggregate scores
    score_route_values = [r["scores"]["score_route"] for r in records]
    score_penalty_values = [r["scores"]["score_penalty"] for r in records]
    score_composed_values = [r["scores"]["score_composed"] for r in records]

    score_route_mean = np.mean(score_route_values)
    score_penalty_mean = np.mean(score_penalty_values)
    score_composed_mean = np.mean(score_composed_values)

    # Planning metrics (L2 error)
    planning_keys = ["avg_l2_1s", "avg_l2_2s", "avg_l2_3s", "l2_0.5s", "l2_1.0s", "l2_1.5s", "l2_2.0s", "l2_2.5s", "l2_3.0s"]
    planning_metrics_mean = {}
    planning_metrics_std = {}
    for key in planning_keys:
        values = [r["planning_metrics"][key] for r in records if "planning_metrics" in r and key in r["planning_metrics"]]
        planning_metrics_mean[key] = round(np.mean(values), 3) if values else 0.0
        planning_metrics_std[key] = round(np.std(values), 3) if len(values) > 1 else 0.0

    # Route length / km driven
    total_route_length = sum(r["meta"]["route_length"] for r in records)
    total_duration_game = sum(r["meta"]["duration_game"] for r in records)
    km_driven = sum(r["meta"]["route_length"] / 1000.0 * r["scores"]["score_route"] / 100.0 for r in records)
    km_driven = max(km_driven, 0.001)

    global_record = {
        "index": 0,
        "route_id": -1,
        "status": f"Evaluated {total_scenarios} scenarios",
        "infractions": infractions_sum,
        "scores_mean": {
            "score_route": round(score_route_mean, 6),
            "score_penalty": round(score_penalty_mean, 6),
            "score_composed": round(score_composed_mean, 6),
        },
        "scores_std_dev": {
            "score_route": round(np.std(score_route_values), 6),
            "score_penalty": round(np.std(score_penalty_values), 6),
            "score_composed": round(np.std(score_composed_values), 6),
        },
        "planning_metrics_mean": planning_metrics_mean,
        "planning_metrics_std_dev": planning_metrics_std,
        "meta": {
            "route_length": total_route_length,
            "duration_game": round(total_duration_game, 3),
        },
    }

    final_results = {
        "entry_status": "Finished",
        "eligible": True,
        "_checkpoint": {
            "global_record": global_record,
            "progress": [total_scenarios, total_scenarios],
            "records": records,
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4, default=str)

    logger.info(
        f"Aggregated results saved to: {output_file}\n"
        f"  Total scenarios: {total_scenarios}\n"
        f"  Avg. driving score: {score_composed_mean:.6f}\n"
        f"  Avg. route completion: {score_route_mean:.6f}%\n"
        f"  L2 @ 1s: {planning_metrics_mean.get('avg_l2_1s', 0.0):.3f}\n"
        f"  L2 @ 2s: {planning_metrics_mean.get('avg_l2_2s', 0.0):.3f}\n"
        f"  L2 @ 3s: {planning_metrics_mean.get('avg_l2_3s', 0.0):.3f}")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="UrbanSim batch evaluation for multiple scenarios")

    parser.add_argument("--model-type", type=str, required=True,
                        choices=["uniad", "vad", "tcp", "rap", "lead", "lead_navsim", "drivor",
                                 "transfuser", "ltf", "egomlp", "ego_mlp", "diffusiondrive",
                                 "diffusiondrivev2", "openpilot", "alpamayo_r1", "pdm_closed"],
                        help="Model type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to model config file")
    parser.add_argument("--plan-anchor-path", type=str, default=None, help="Plan anchor file (DiffusionDrive/V2)")
    parser.add_argument("--scenario-root", type=str, required=True, help="Root directory containing scenarios")
    parser.add_argument("--output-dir", type=str, required=True, help="Root directory for outputs")
    parser.add_argument("--traffic-mode", type=str, default="log_replay",
                        choices=["no_traffic", "log_replay", "semi_reactive"],
                        help="'idm' was removed: it was an unwired silent "
                             "no-op (agents replayed the log); semi_reactive "
                             "is the wired reactive mode.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-perframe", action="store_true", default=True)
    parser.add_argument("--no-save-perframe", dest="save_perframe", action="store_false")
    parser.add_argument("--controller", type=str, default="pure_pursuit", choices=["pid", "pure_pursuit"])
    parser.add_argument("--replan-rate", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.1)
    parser.add_argument("--ego-replay-frames", type=int, default=0)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--score-start-frame", type=int, default=None,
                        help="REFUSED if set: never wired to anything. "
                             "Scoring starts at the warm-up hand-off — use "
                             "--ego-replay-frames.")
    parser.add_argument("--eval-mode", type=str, default="closed_loop",
                        choices=["closed_loop"],
                        help="Only closed_loop exists; the historical "
                             "'open_loop' value was parsed but never "
                             "implemented (use --ego-replay-frames >= "
                             "--eval-frames for a pure log-replay run).")
    parser.add_argument("--enable-vis", action="store_true")
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

    evaluator = BatchEvaluator(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        scenario_root=args.scenario_root,
        output_root=args.output_dir,
        config_path=args.config,
        plan_anchor_path=args.plan_anchor_path,
        traffic_mode=args.traffic_mode,
        max_workers=args.max_workers,
        resume=args.resume,
        save_perframe=args.save_perframe,
        controller_type=args.controller,
        replan_rate=args.replan_rate,
        sim_dt=args.sim_dt,
        ego_replay_frames=args.ego_replay_frames,
        eval_frames=args.eval_frames,
        score_start_frame=args.score_start_frame,
        eval_mode=args.eval_mode,
        enable_vis=args.enable_vis,
        enable_temporal_consistency=args.enable_temporal_consistency,
        temporal_alpha=args.temporal_alpha,
        temporal_lambda=args.temporal_lambda,
        temporal_max_history=args.temporal_max_history,
        temporal_sigma=args.temporal_sigma,
        consensus_temperature=args.consensus_temperature,
    )

    try:
        evaluator.run()
    except KeyboardInterrupt:
        print("\n\nBatch evaluation interrupted by user!")
        print("Progress has been saved. Use --resume to continue.")
        evaluator.save_results()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nBatch evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        evaluator.save_results()
        sys.exit(1)


if __name__ == "__main__":
    main()
