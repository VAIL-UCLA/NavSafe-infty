"""In-memory primitive trace for NavSafe reactivity experiments.

The recorder performs no per-frame filesystem I/O.  Every table stays in
process memory; :meth:`finalize` builds one ZIP in a BytesIO and performs one
sequential write to the evaluation output directory (normally a PVC).
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional
import io
import json
import math
import zipfile

import numpy as np


SCHEMA_VERSION = "navsafe-reactivity-primitives-v1"
_TABLES = (
    "intervention.jsonl",
    "ego_states.jsonl",
    "actor_states.jsonl",
    "model_queries.jsonl",
    "plans.jsonl",
    "controls.jsonl",
    "safety_events.jsonl",
)


def _jsonable(value: Any) -> Any:
    """Convert runtime values without rounding; non-finite numbers become null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _vector(value: Any, default: tuple[float, ...] = ()) -> list[Optional[float]]:
    if value is None:
        value = default
    try:
        return _jsonable(np.asarray(value, dtype=np.float64).reshape(-1))
    except Exception:
        return _jsonable(default)


def _json_line(row: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(row), separators=(",", ":"), allow_nan=False) + "\n"


def _driver_description(driver: Any) -> dict[str, Any]:
    motion = getattr(driver, "motion", None)
    policy = dict(getattr(motion, "policy", {}) or {})
    profile = getattr(driver, "profile", None)
    if not policy and profile is not None:
        policy = _jsonable(profile)
    kind = policy.get("event_kind") or getattr(motion, "kind", None)
    onset = policy.get("onset_s")
    # AuthoredMotion keeps the original policy today, but ``onset`` is the
    # runtime source of truth used by sample().  Reading it also keeps traces
    # correct for wrappers/minimal runtime builds that do not retain policy.
    if onset is None:
        onset = getattr(motion, "onset", getattr(motion, "onset_s", None))
    if onset is None:
        onset = getattr(profile, "onset_s", None)
    enabled = policy.get("enabled")
    if enabled is None:
        enabled = getattr(motion, "enabled", getattr(profile, "enabled", True))
    return {
        "driver": type(driver).__name__,
        "event_kind": kind,
        "configured_onset_s": _jsonable(onset),
        "enabled": bool(enabled),
        "policy": _jsonable(policy),
    }


class ReactivityTraceRecorder:
    """Collect trace primitives in RAM and emit one archive at episode end."""

    def __init__(self, evaluator: Any):
        self.evaluator = evaluator
        self.config = evaluator.config
        self.tables: dict[str, list[dict[str, Any]]] = {name: [] for name in _TABLES}
        self._query_id = 0
        self.active_query_id: Optional[int] = None
        self._previous_actor: dict[str, tuple[float, np.ndarray, float]] = {}
        self._actual_onsets: dict[str, float] = {}
        self._actor_catalog: dict[str, dict[str, Any]] = {}
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.condition = self._condition()
        self.group_id = getattr(self.config, "reactivity_group_id", None)
        self.manifest = self._manifest()
        self.scene_geometry = self._scene_geometry()
        self._catalog_drivers()

    def _manager(self) -> Any:
        return getattr(self.evaluator.env, "_traffic_manager", None)

    def _drivers(self) -> dict[str, Any]:
        return {
            str(k): v for k, v in getattr(self._manager(), "drivers", {}).items()
        }

    def _condition(self) -> str:
        requested = str(getattr(self.config, "reactivity_condition", "auto"))
        if requested != "auto":
            return requested
        descriptions = [_driver_description(d) for d in self._drivers().values()]
        return "hazard" if any(d["enabled"] for d in descriptions) else "no_change"

    def _manifest(self) -> dict[str, Any]:
        adapter = self.evaluator.adapter
        cfg = self.config
        metadata = dict(getattr(cfg, "reactivity_metadata", None) or {})
        return {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": self.started_at,
            "run_id": metadata.pop("run_id", None),
            "group_id": self.group_id,
            "condition": self.condition,
            "scenario_id": self.evaluator.scenario_id,
            "scenario_path": str(self.evaluator.scenario_path),
            "scenario_metadata": _jsonable(
                self.evaluator.scenario_data.get("metadata", {})),
            "model": {
                "adapter": type(adapter).__name__,
                "waypoint_dt_s": _jsonable(adapter.get_waypoint_dt()),
                "metadata": _jsonable(metadata),
            },
            "simulation": {
                "traffic_mode": cfg.traffic_mode,
                "execution_mode": cfg.execution_mode,
                "controller_type": cfg.controller_type,
                "sim_dt_s": cfg.sim_dt,
                "replan_rate_frames": cfg.replan_rate,
                "ego_replay_frames": cfg.ego_replay_frames,
                "eval_frames": cfg.eval_frames,
                "enable_vis": cfg.enable_vis,
            },
            "storage": {
                "buffering": "memory_until_finalize",
                "pvc_writes": 1,
                "images_recorded": False,
            },
        }

    def _scene_geometry(self) -> dict[str, Any]:
        route = []
        for item in list(getattr(self.evaluator, "full_route", []) or []):
            candidate = item[0] if isinstance(item, (list, tuple)) and item else item
            try:
                arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
            except Exception:
                continue
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                route.append(arr[:3].tolist())
        scenario = self.evaluator.scenario_data
        metadata = scenario.get("metadata", {})
        return {
            "coordinate_frame": "scenario_world",
            "route_polyline": route,
            "map_feature_reference": {
                "scenario_id": self.evaluator.scenario_id,
                "map_name": metadata.get("map_name"),
                "map_version": metadata.get("map_version"),
                "map_feature_count": len(scenario.get("map_features", {}) or {}),
            },
            "ego_dimensions": self._ego_dimensions(),
        }

    def _ego_dimensions(self) -> dict[str, Any]:
        cfg = getattr(getattr(self.evaluator.env, "_ego", None), "cfg", None)
        return {
            "length_m": _jsonable(getattr(cfg, "ego_length", None)),
            "width_m": _jsonable(getattr(cfg, "ego_width", None)),
            "wheelbase_m": _jsonable(getattr(cfg, "wheelbase", None)),
        }

    def _catalog_drivers(self) -> None:
        for actor_id, driver in self._drivers().items():
            description = _driver_description(driver)
            self._actor_catalog[actor_id] = description
            self.tables["intervention.jsonl"].append({
                "kind": "definition",
                "actor_id": actor_id,
                "condition": self.condition,
                **description,
            })

    def record_plan(
        self,
        *,
        frame: int,
        ego_state: Mapping[str, Any],
        raw_traj_ego: Any,
        selected_world: Any,
        controller_world: Any,
        selected_speeds: Any,
        controller_speeds: Any,
        candidates_ego: Any,
        candidate_scores: Any,
        selected_index: Optional[int],
        emergency_brake: bool,
        model_dt_s: float,
        inference_wall_s: float,
    ) -> int:
        query_id = self._query_id
        self._query_id += 1
        self.active_query_id = query_id
        query_time = float(frame) * float(self.config.sim_dt)
        self.tables["model_queries.jsonl"].append({
            "query_id": query_id,
            "query_frame": int(frame),
            "query_time_s": query_time,
            "plan_application_time_s": query_time,
            "inference_wall_s": float(inference_wall_s),
            "ego_position": _vector(ego_state.get("position")),
            "ego_heading_rad": _jsonable(ego_state.get("heading")),
            "selected_candidate_index": selected_index,
            "emergency_brake": bool(emergency_brake),
            "complete": True,
        })
        native = np.asarray(raw_traj_ego)
        self.tables["plans.jsonl"].append({
            "query_id": query_id,
            "coordinate_conventions": {
                "selected_ego": "ego lateral-left, forward",
                "selected_world": "scenario world XY",
                "controller_world": "scenario world XY",
            },
            "native_waypoint_dt_s": float(model_dt_s),
            "native_waypoint_times_s": (
                (np.arange(len(native), dtype=np.float64) + 1.0) * float(model_dt_s)
            ).tolist(),
            "selected_ego": _jsonable(raw_traj_ego),
            "selected_world": _jsonable(selected_world),
            "controller_world": _jsonable(controller_world),
            "selected_speeds_mps": _jsonable(selected_speeds),
            "controller_speeds_mps": _jsonable(controller_speeds),
            "candidates_ego": _jsonable(candidates_ego),
            "candidate_scores": _jsonable(candidate_scores),
            "selected_candidate_index": selected_index,
            "emergency_brake": bool(emergency_brake),
        })
        return query_id

    def record_step(
        self,
        *,
        frame: int,
        ego_before: Mapping[str, Any],
        ego_after: Mapping[str, Any],
        action: Any,
        control: Mapping[str, Any],
        info: Any,
        frame_metrics: Mapping[str, Any],
        signal_hold: bool,
        signal_violation: Optional[bool],
        episode_done: bool,
    ) -> None:
        dt = float(self.config.sim_dt)
        start = float(frame) * dt
        end = start + dt
        before_v = np.asarray(ego_before.get("velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        after_v = np.asarray(ego_after.get("velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
        n = max(len(before_v), len(after_v), 2)
        b = np.zeros(n); a = np.zeros(n)
        b[:len(before_v)] = before_v; a[:len(after_v)] = after_v
        yaw_before = float(ego_before.get("heading", 0.0))
        yaw_after = float(ego_after.get("heading", yaw_before))
        yaw_delta = math.atan2(math.sin(yaw_after-yaw_before), math.cos(yaw_after-yaw_before))
        ego_row = {
            "frame": int(frame), "time_s": start, "end_time_s": end,
            "phase": "warmup" if frame < self.config.ego_replay_frames else "scored",
            "position": _vector(ego_after.get("position")),
            "heading_rad": yaw_after,
            "velocity": _vector(after_v),
            "speed_mps": _jsonable(ego_after.get("speed", np.linalg.norm(after_v[:2]))),
            "acceleration": _jsonable((a-b)/dt),
            "yaw_rate_rad_s": yaw_delta/dt,
            "route_command": _jsonable(getattr(self.evaluator, "_current_command", None)),
            "route_deviation_m": _jsonable(frame_metrics.get("ego_dev_m")),
            "terminal": bool(episode_done),
        }
        self.tables["ego_states.jsonl"].append(ego_row)
        self.tables["controls.jsonl"].append({
            "frame": int(frame), "time_s": start,
            "query_id": self.active_query_id,
            "action": _vector(action),
            **_jsonable(dict(control)),
        })
        self._record_actors(frame=frame, time_s=end)
        info_dict = info if isinstance(info, Mapping) else {}
        if bool(frame_metrics.get("collision")):
            self.tables["safety_events.jsonl"].append({
                "kind": "contact_begin", "frame": int(frame), "time_s": end,
                "at_fault": bool(frame_metrics.get("collision_at_fault")),
                "detail": _jsonable(frame_metrics.get("contact_detail", {})),
            })
        if signal_hold or signal_violation is not None:
            self.tables["safety_events.jsonl"].append({
                "kind": "traffic_signal_state", "frame": int(frame), "time_s": end,
                "signal_hold": bool(signal_hold),
                "signal_violation": signal_violation,
            })
        if episode_done:
            self.tables["safety_events.jsonl"].append({
                "kind": "episode_terminal", "frame": int(frame), "time_s": end,
                "terminated": _jsonable(info_dict.get("terminated")),
                "truncated": _jsonable(info_dict.get("truncated")),
            })

    def _record_actors(self, *, frame: int, time_s: float) -> None:
        drivers = self._drivers()
        for index, state in enumerate(list(getattr(self.evaluator.env, "agent_states", []) or [])):
            actor_id = str(state.get("id", index))
            velocity = np.asarray(state.get("velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
            heading = float(state.get("heading", 0.0))
            previous = self._previous_actor.get(actor_id)
            acceleration = np.zeros(max(2, len(velocity)), dtype=np.float64)
            yaw_rate = 0.0
            if previous is not None and time_s > previous[0]:
                old_v = previous[1]
                n = max(len(old_v), len(velocity), 2)
                left = np.zeros(n); right = np.zeros(n)
                left[:len(old_v)] = old_v; right[:len(velocity)] = velocity
                acceleration = (right-left)/(time_s-previous[0])
                dyaw = math.atan2(math.sin(heading-previous[2]), math.cos(heading-previous[2]))
                yaw_rate = dyaw/(time_s-previous[0])
            self._previous_actor[actor_id] = (time_s, velocity.copy(), heading)
            driver = drivers.get(actor_id)
            desc = self._actor_catalog.get(actor_id, {})
            if driver is not None and not desc:
                # Some environments instantiate recipe drivers lazily on the
                # first step, after the recorder itself is constructed.
                desc = _driver_description(driver)
                self._actor_catalog[actor_id] = desc
                self.tables["intervention.jsonl"].append({
                    "kind": "definition",
                    "actor_id": actor_id,
                    "condition": self.condition,
                    **desc,
                })
                if (str(getattr(self.config, "reactivity_condition", "auto"))
                        == "auto" and desc["enabled"]):
                    self.condition = "hazard"
                    self.manifest["condition"] = "hazard"
            role = self.condition if driver is not None else "background"
            row = {
                "frame": int(frame), "time_s": float(time_s),
                "actor_id": actor_id,
                "source_track_id": _jsonable(state.get("source_track_id")),
                "render_id": _jsonable(state.get("render_id")),
                "role": role,
                "semantic_class": _jsonable(state.get("type", state.get("semantic_class"))),
                "position": _vector(state.get("position")),
                "heading_rad": heading,
                "velocity": _vector(velocity),
                "acceleration": _jsonable(acceleration),
                "yaw_rate_rad_s": yaw_rate,
                "length_m": _jsonable(state.get("length")),
                "width_m": _jsonable(state.get("width")),
                "active": bool(state.get("active", True)),
                "driver": desc.get("driver"),
                "event_kind": desc.get("event_kind"),
            }
            self.tables["actor_states.jsonl"].append(row)
            if driver is None:
                continue
            current = float(getattr(driver, "time_s", time_s))
            onset = desc.get("configured_onset_s")
            enabled = bool(desc.get("enabled", True))
            active = enabled and onset is not None and current + 1e-9 >= float(onset)
            phase = "active" if active else ("disabled" if not enabled else "before_onset")
            if active and actor_id not in self._actual_onsets:
                self._actual_onsets[actor_id] = current
                self.tables["intervention.jsonl"].append({
                    "kind": "actual_onset", "actor_id": actor_id,
                    "event_kind": desc.get("event_kind"),
                    "frame": int(frame), "actual_onset_s": current,
                    "configured_onset_s": onset,
                })
            self.tables["intervention.jsonl"].append({
                "kind": "state", "frame": int(frame), "time_s": current,
                "actor_id": actor_id, "phase": phase, "enabled": enabled,
                "event_kind": desc.get("event_kind"),
                "configured_onset_s": onset,
                "position": row["position"], "velocity": row["velocity"],
            })

    def finalize(self, *, results: Mapping[str, Any], termination: Mapping[str, Any],
                 step_crash: Any = None) -> Path:
        completion = {
            "schema_version": SCHEMA_VERSION,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_scored_frames": self.config.eval_frames,
            "recorded_frames": len(self.tables["ego_states.jsonl"]),
            "recorded_queries": len(self.tables["model_queries.jsonl"]),
            "recorded_plans": len(self.tables["plans.jsonl"]),
            "recorded_actor_rows": len(self.tables["actor_states.jsonl"]),
            "actual_onsets_s": self._actual_onsets,
            "termination": _jsonable(termination),
            "step_crash": _jsonable(step_crash),
            "metrics_snapshot": _jsonable(results.get("metrics", {})),
            "trace_complete": (
                step_crash is None
                and len(self.tables["model_queries.jsonl"]) == len(self.tables["plans.jsonl"])
                and len(self.tables["ego_states.jsonl"]) == int(results.get("total_frames", 0))
            ),
        }
        schema = {
            "schema_version": SCHEMA_VERSION,
            "time_base": "simulation seconds",
            "event_time_zero": "actual_onset_s",
            "images": "not recorded",
            "tables": list(_TABLES),
        }
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True) as archive:
            archive.writestr("schema.json", json.dumps(schema, separators=(",", ":")))
            archive.writestr("manifest.json", json.dumps(_jsonable(self.manifest), separators=(",", ":")))
            archive.writestr("scene_geometry.json", json.dumps(_jsonable(self.scene_geometry), separators=(",", ":")))
            for name in _TABLES:
                archive.writestr(name, "".join(_json_line(row) for row in self.tables[name]))
            archive.writestr("completion.json", json.dumps(_jsonable(completion), separators=(",", ":")))
        bundle.seek(0)
        with zipfile.ZipFile(bundle, "r") as verification:
            bad = verification.testzip()
            if bad is not None:
                raise RuntimeError(f"reactivity trace ZIP CRC failed: {bad}")
        payload = bundle.getbuffer()
        output = Path(self.config.output_dir) / str(
            getattr(self.config, "reactivity_trace_filename", "reactivity_trace.zip"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb", buffering=1024 * 1024) as stream:
            written = stream.write(payload)
        if written != len(payload):
            raise IOError(f"short reactivity trace write: {written} != {len(payload)}")
        self.bundle_size_bytes = len(payload)
        self.bundle_sha256 = sha256(payload).hexdigest()
        return output
