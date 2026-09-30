"""The red-light CROSSING rule, batch and live (2026-08-23).

See ``TL_STOP_LINE_ZONE_M`` in ``epdms_trajectory_scorer_fast.py``: the lanes
carrying a light state are intersection connectors whose START is the stop
line, and the charge is the crossing of that line at speed. These tests pin
the batch scorer's half of the rule (the live half is in
``test_epdms_live_scorer.py``), the sibling exemptions, the clamp past the log
end, and batch-vs-live agreement on one crossing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from navsafe.evaluation.scorers.epdms_trajectory_scorer_fast import (
    EPDMSLiveScorer,
    EPDMSTrajectoryScorer_Fast,
)

N = 100


def _lane(points, width=3.5):
    pts = np.asarray(points, dtype=np.float64)
    d = np.gradient(pts, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-d[:, 1], d[:, 0]]) * (width / 2.0)
    polygon = np.vstack([pts + normal, (pts - normal)[::-1]])
    return {"type": "LANE_SURFACE_STREET",
            "polyline": np.column_stack([pts, np.zeros(len(pts))]),
            "polygon": polygon}


def _straight(x0, x1, y=0.0):
    xs = np.linspace(x0, x1, int(abs(x1 - x0)) + 1)
    return np.column_stack([xs, np.full_like(xs, y)])


def _light(states):
    seq = [states] * N if isinstance(states, str) else list(states)
    return {"state": {"object_state": seq}}


def _world(*, conn_state="LANE_STATE_STOP", turn_state=None, merge_red=True,
           cross_red=True):
    """Approach lane x in [-50, 0]; connector x in [0, 40] starting at the
    stop line x=0; optional turn sibling forking off x=0 (curving to +y);
    a merging sibling ending at (40, 0) from (20, -15); a crossing connector
    through x=20 along +y."""
    turn = np.array([[s, 0.08 * s * s] for s in np.linspace(0.0, 15.0, 16)])
    merge = np.column_stack([np.linspace(20.0, 40.0, 21),
                             np.linspace(-15.0, 0.0, 21)])
    cross = np.column_stack([np.full(41, 20.0), np.linspace(-20.0, 20.0, 41)])
    features = {
        "approach": _lane(_straight(-50.0, 0.0)),
        "conn": _lane(_straight(0.0, 40.0)),
        "merge": _lane(merge),
        "cross": _lane(cross),
    }
    states = {"conn": _light(conn_state)}
    if merge_red:
        states["merge"] = _light("LANE_STATE_STOP")
    if cross_red:
        states["cross"] = _light("LANE_STATE_STOP")
    if turn_state is not None:
        features["turn"] = _lane(turn)
        states["turn"] = _light(turn_state)
    positions = np.zeros((N, 3))
    positions[:, 0] = np.linspace(-50.0, 40.0, N)
    return {
        "map_features": features,
        "dynamic_map_states": states,
        "tracks": {"ego": {"type": "VEHICLE", "state": {
            "position": positions, "heading": np.zeros(N),
            "valid": np.ones(N, dtype=bool),
            "velocity": np.ones((N, 2)) * np.array([5.0, 0.0])}}},
        "metadata": {"sdc_id": "ego", "ts": 0.1},
        "length": N,
    }


def _batch(sd):
    env = MagicMock()
    env.agent.position = np.array([0.0, 0.0])
    type(env).engine = PropertyMock(side_effect=AttributeError)
    scorer = EPDMSTrajectoryScorer_Fast(verbose=False)
    scorer.initialize(sd, env)
    scorer.planner_dt = 0.5
    return scorer


def _states(xs, ys=None, speed=5.0, heading=0.0):
    n = len(xs)
    return {
        "x": np.asarray(xs, dtype=float),
        "y": np.zeros(n) if ys is None else np.asarray(ys, dtype=float),
        "heading": np.full(n, heading), "speed": np.full(n, speed),
        "acceleration": np.zeros(n), "jerk": np.zeros(n),
        "yaw_rate": np.zeros(n), "lon_accel": np.zeros(n),
        "lon_jerk": np.zeros(n), "yaw_accel": np.zeros(n),
    }


def _tlc(scorer, states, frame_idx=0):
    horizon = len(states["x"]) - 1
    _, red = scorer._precompute_frame_data(frame_idx, horizon)
    return scorer._calculate_metrics(states, horizon, frame_idx,
                                     red_lanes_per_t=red)["tlc"]


# --- batch -----------------------------------------------------------------


def test_batch_crossing_the_red_stop_line_at_speed_is_charged():
    sc = _batch(_world())
    # Poses 0.5 s apart at 5 m/s: -6, -3.5, -1, 1.5 (crossing), 4, ...
    assert _tlc(sc, _states(np.arange(-6.0, 20.0, 2.5))) == 0.0


def test_batch_pose_zero_inside_the_connector_is_not_a_crossing():
    """The current pose is where the candidate STARTS; whatever it did to get
    there was charged live. A candidate launched from 1 m past the line
    must not zero every proposal of a stopped-too-far-forward ego."""
    sc = _batch(_world())
    assert _tlc(sc, _states(np.arange(1.0, 30.0, 2.5))) == 1.0


def test_batch_walking_pace_crossing_is_not_charged():
    sc = _batch(_world())
    assert _tlc(sc, _states(np.arange(-1.0, 4.0, 0.4), speed=0.8)) == 1.0


def test_batch_stopping_short_of_the_line_is_clean():
    sc = _batch(_world())
    xs = np.array([-12.0, -9.0, -6.5, -4.5, -3.2, -3.0, -3.0, -3.0])
    assert _tlc(sc, _states(xs, speed=2.0)) == 1.0


def test_batch_merging_sibling_entered_from_the_side_near_its_end_is_not_charged():
    """The ego on its own (stateless here) lane enters the red merging
    connector's polygon near ITS end -- the navhard421 artifact."""
    sd = _world(conn_state="LANE_STATE_GO")
    sc = _batch(sd)
    assert _tlc(sc, _states(np.arange(30.0, 45.0, 2.5))) == 1.0


def test_batch_crossing_connector_traversed_mid_way_is_not_charged():
    sd = _world(conn_state="LANE_STATE_GO")
    sc = _batch(sd)
    assert _tlc(sc, _states(np.arange(10.0, 32.0, 2.5))) == 1.0


def test_batch_green_sibling_sharing_the_stop_line_exempts_the_crossing():
    """Straight GO, protected-left STOP, same stop line: an ego driving
    straight also enters the red turn connector's first metres. It had a
    green way through that line, so it is not charged."""
    sc = _batch(_world(conn_state="LANE_STATE_GO", turn_state="LANE_STATE_STOP"))
    assert _tlc(sc, _states(np.arange(-6.0, 20.0, 2.5))) == 1.0
    # ... and when every signalized connector at that line is red, it is.
    sc = _batch(_world(conn_state="LANE_STATE_STOP", turn_state="LANE_STATE_STOP"))
    assert _tlc(sc, _states(np.arange(-6.0, 20.0, 2.5))) == 0.0


def test_batch_red_lanes_clamp_past_the_log_end():
    seq = ["LANE_STATE_GO"] * 50 + ["LANE_STATE_STOP"] * 50
    sd = _world(conn_state=seq, merge_red=False, cross_red=False)
    sc = _batch(sd)
    # frame 500 is past the 100-frame log: the last logged state (red) holds.
    assert _tlc(sc, _states(np.arange(-6.0, 20.0, 2.5)), frame_idx=500) == 0.0
    # Inside the log, before the transition: green.
    assert _tlc(sc, _states(np.arange(-6.0, 20.0, 2.5)), frame_idx=10) == 1.0


# --- batch vs live ----------------------------------------------------------


class _FakeEnv:
    def __init__(self):
        self.agent = self
        self.engine = self
        self.name = "__ego__"
        self.position = np.array([0.0, 0.0])
        self.heading_theta = 0.0
        self.velocity = np.array([0.0, 0.0])

    def get_objects(self):
        return {}


def _live_tlc(sd, prev_xy, xy, speed, frame_idx=0, heading=0.0):
    env = _FakeEnv()
    live = EPDMSLiveScorer(sd, env)
    live.reset_live_state()
    env.position = np.asarray(prev_xy, dtype=float)
    env.velocity = np.array([speed, 0.0])
    live.observe_frame_kinematics()
    env.position = np.asarray(xy, dtype=float)
    env.heading_theta = heading
    return live.score_frame_live(frame_idx)["traffic_light_compliance"]


@pytest.mark.parametrize("prev_x,x,speed,expected", [
    (-1.0, 1.0, 5.0, 0.0),      # crossing at speed
    (1.0, 2.0, 5.0, 1.0),       # already inside
    (-0.3, 0.1, 0.8, 1.0),      # walking pace (0.4 m per 0.5 s sample)
    (-4.0, -2.0, 5.0, 1.0),     # still short of the line
])
def test_batch_and_live_agree_on_the_crossing(prev_x, x, speed, expected):
    sd = _world(merge_red=False, cross_red=False)
    assert _live_tlc(sd, (prev_x, 0.0), (x, 0.0), speed) == expected
    sc = _batch(sd)
    assert _tlc(sc, _states([prev_x, x], speed=speed)) == expected


def test_live_green_sibling_sharing_the_stop_line_exempts_the_crossing():
    sd = _world(conn_state="LANE_STATE_GO", turn_state="LANE_STATE_STOP",
                merge_red=False, cross_red=False)
    assert _live_tlc(sd, (-1.0, 0.0), (1.0, 0.0), 5.0) == 1.0
    sd = _world(conn_state="LANE_STATE_STOP", turn_state="LANE_STATE_STOP",
                merge_red=False, cross_red=False)
    assert _live_tlc(sd, (-1.0, 0.0), (1.0, 0.0), 5.0) == 0.0
