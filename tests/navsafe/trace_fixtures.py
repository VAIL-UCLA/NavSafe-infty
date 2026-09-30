"""Synthetic traces and programs, so the rubric engine is testable on its own.

The rubric is a pure function of the trace, which means it can be exercised
long before -- and independently of -- the simulator that will eventually
produce real traces.  These fixtures encode one deliberate behaviour each.
"""

from __future__ import annotations


from navsafe.benchmark.rubric.schema import ScenarioProgram
from navsafe.benchmark.trace.schema import PHASE_SCORED, PHASE_WARMUP, empty_frame

DT = 0.1
WARMUP = 20            # 2.0 s, matching the seeds' t_pre
SCORED = 80            # 8.0 s


class ProgramBuilder:
    """A minimal straight-traversal program; tweak one thing per test."""

    def __init__(self) -> None:
        self._t_max = 15.0

    def with_t_max(self, t: float) -> "ProgramBuilder":
        self._t_max = t
        return self

    def build(self) -> ScenarioProgram:
        return ScenarioProgram.from_dict(
            {
                "family": "F3_straight_traversal",
                "initialization": {},
                "success": [
                    {"reach": {"region": "exit_lane_polygon", "t_max": "t_max_s"}},
                    {"align": {"lane": "exit_lane", "heading_tol": 10}},
                    {"comply": {"signal": "signal_at_entry"}},
                ],
                "gates": [
                    {"no_collision": {"fault": "at_fault"}},
                    "on_drivable",
                    {"maintain_gap": {"d_min": 0.5}},
                ],
                "diagnostics": ["min_ttc", "time_in_intersection", "n_stops",
                                "max_abs_jerk"],
                "actors": {"*": "replay"},
                "t_max_s": self._t_max,
            },
            seed_id="synthetic", regime="log_replay",
        )


def straight_traversal_trace(
    *,
    never_moves: bool = False,
    rear_ended_not_at_fault: bool = False,
    at_fault_collision: bool = False,
    runs_red: bool = False,
    exit_only_in_warmup: bool = False,
) -> list[dict]:
    """Ego crosses a signalised intersection and settles in the exit lane."""
    frames: list[dict] = []
    speed = 0.0 if never_moves else 8.0
    x = 0.0
    for i in range(WARMUP + SCORED):
        t = i * DT
        scored = i >= WARMUP
        f = empty_frame()
        x += speed * DT
        # green from the start unless the test wants a red-light run
        state = "red" if runs_red and scored and t < (WARMUP * DT + 3.0) else "green"
        # geometry: intersection from 20-45 m, exit lane past 50 m
        in_ix = 20.0 <= x <= 45.0
        in_exit = x > 50.0
        if exit_only_in_warmup:
            in_exit = not scored          # only "arrives" during warm-up
        f.update({
            "frame": i, "t_sim_s": t, "t_log_us": int(t * 1e6),
            "phase": PHASE_SCORED if scored else PHASE_WARMUP,
            "ego_x": x, "ego_y": 0.0, "ego_z": 0.0,
            "ego_yaw": 0.0, "ego_speed": speed,
            "ego_accel": 0.0, "ego_jerk": 0.0, "ego_lat_accel": 0.0,
            "ego_steer": 0.0,
            "on_drivable": True,
            "lane_id": "exit_lane" if in_exit else "entry_lane",
            "lane_heading": 0.0, "lateral_offset_m": 0.0,
            "dist_to_stopline_m": 18.0 - x,
            "in_intersection": in_ix,
            "in_conflict_zone": in_ix,
            "in_exit_lane": in_exit,
            "signal_id": "sig_0", "signal_state": state,
            "agents": [], "contacts": [],
            "min_clearance_m": 12.0,
            "driving_direction_ok": True,
            "ego_dev_lat_m": 0.0, "ego_dev_lon_m": 0.0, "ego_dev_yaw_deg": 0.0,
        })
        # A follower that stays well clear unless the test says otherwise.
        f["agents"] = [{
            "id": "follower_0", "cls": "vehicle", "policy": "replay",
            "x": x - 12.0, "y": 0.0, "yaw": 0.0, "speed": speed,
            "length": 4.5, "width": 1.9, "dist_to_ego": 12.0,
            "clearance_m": 12.0 - 4.5,
            "is_lead": False, "in_conflict_zone": False, "ttc_s": float("inf"),
            "ego_is_closing": False,   # it is behind; the ego is not closing
        }]
        if scored and rear_ended_not_at_fault and i == WARMUP + 30:
            f["contacts"] = [{"agent_id": "follower_0", "at_fault": False,
                              "kind": "rear_end", "rel_speed": 2.0}]
            f["agents"][0]["dist_to_ego"] = 0.0
            f["agents"][0]["clearance_m"] = 0.0
        if scored and at_fault_collision and i == WARMUP + 30:
            f["contacts"] = [{"agent_id": "crosser_0", "at_fault": True,
                              "kind": "angle", "rel_speed": 7.0}]
        frames.append(f)
    return frames
