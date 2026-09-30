# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Semi-reactive traffic: MetaDrive's ``reactive_traffic``, ported and extended.

Selected by ``EnvCfg(traffic_mode="semi_reactive")``.  The eligibility
conditions, the IDM constants, the lead-vehicle search and the batched update
are taken from MetaDrive's ``ScenarioTrafficManager`` with
``reactive_traffic=True`` (metadriverse/metadrive @ main, read 2026-08-09) and
cited per rule below.  The same constants and conditions were checked against
the MetaDrive checkouts on this box
(``/data/{metadrive,marl-metadrive,cat_yuxin}``,
``/home/user/metadrive``); all four agree with upstream.

**One deliberate departure from upstream, and it is the default**: ``takeover``
controls *when* the eligibility test runs.

* ``"continuous"`` (default) re-tests every non-reactive vehicle every frame,
  so a vehicle the ego catches up to and passes becomes reactive at the frame
  it falls behind, instead of replaying for the whole episode.
* ``"spawn"`` tests once, at the frame the vehicle first appears, and never
  again.  This is MetaDrive's own behaviour; use it to reproduce upstream (or
  any NexusSim result recorded before 2026-08-11) exactly.

Takeover is one-way under both settings.  A reactive vehicle is never handed
back to replay: its integrated state has diverged from the log, so restoring
the logged pose would teleport it.

**Who becomes reactive** (``ScenarioTrafficManager.spawn_vehicle``, L175-262).
The rules below are upstream's; ``takeover`` only decides how often they run.
Vehicle type and ``moving`` are properties of the whole track and can never
change once evaluated, so a vehicle failing either is cached as permanently
ineligible and never re-tested.  ``length_ok``, ``heading_ok`` and ``idm_ok``
depend on the current frame and the ego's pose, and are what continuous mode
re-checks::

    moving    = max(std(valid_positions[:, :2])) > STATIC_THRESHOLD (3 m)
    length_ok = |pos[t] - pos[end-1]| > IDM_CREATE_MIN_LENGTH (5 m)
    heading_ok= |wrap_to_pi(ego_heading - heading)| < pi/2
    idm_ok    = heading_dist < IDM_CREATE_FORWARD_CONSTRAINT (-1 m)
                and |side_dist| < IDM_CREATE_SIDE_CONSTRAINT (15 m)
                and heading_ok

``heading_dist``/``side_dist`` are the vehicle's position in the **ego's local
frame**, so ``heading_dist < -1`` means *behind the ego by more than a metre*.
Only **followers** are ever taken over -- a vehicle ahead of the ego fails
``idm_ok`` in either mode.  The modes differ in what "ahead" means: upstream
freezes the answer at spawn, so a lead stays replayed forever, while continuous
mode takes the same vehicle over the moment the ego actually passes it.
``length_ok`` also tightens with time (it measures the travel *remaining* in
the track), so a vehicle whose log is nearly exhausted stops being eligible.
Pedestrians and cyclists always replay
(``spawn_pedestrian``/``spawn_cyclist`` add ``ReplayTrafficParticipantPolicy``
unconditionally), and so do vehicles classified static.

**How a reactive vehicle drives** (``TrajectoryIDMPolicy``, ``idm_policy.py``
L436-503).  It follows its own remaining logged trajectory as a ``PointLane``
of width 2 m (``get_idm_route``), with IDM longitudinal control and a PID
lateral controller.  Speed control runs in batches: policy ``i`` acts on steps
where ``episode_step % IDM_ACT_BATCH_SIZE == policy_index`` and re-uses its
last acceleration otherwise (``ScenarioTrafficManager.before_step`` L71-79).
The lead is the nearest object **inside that 2 m trajectory corridor**, ahead
in trajectory coordinates, within ``IDM_MAX_DIST`` = 20 m
(``FrontBackObjects.get_find_front_back_objs_single_lane``).

Two places where this port deliberately differs, because the simulators differ:

1. **Lateral control.**  MetaDrive steers with a PID controller onto the
   trajectory and integrates a Bullet vehicle; NexusSim advances the vehicle
   along its trajectory by arc length, so lateral error is zero by
   construction.  The lateral controller has nothing to do here, and its PID
   gains are not ported.
2. **Acceleration units.**  MetaDrive's IDM computes with speeds in **km/h**
   (``speed_km_h``, ``velocity_km_h``) and treats the result as a normalized
   throttle/brake command for its vehicle dynamics.  With ``metadrive_units``
   (the default) the same km/h expression is evaluated -- including its mixed
   units in the desired gap, ``d0 [m] + v [km/h] * tau [s]``, reproduced so the
   policy behaves as MetaDrive's does -- and the result is integrated directly
   as m/s^2.  Set ``metadrive_units=False`` for the unit-consistent SI reading
   of the same constants; it accelerates and brakes harder, and is not what
   MetaDrive does.

A vehicle that reaches the end of its trajectory (within ``DEST_REGION_RADIUS``
= 2 m) is *despawned* by MetaDrive; here it holds its final pose with zero
speed, because removing a prim mid-episode churns the renderer's scene graph.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from navsafe.component.traffic_agent.idm import IDMActor, IDMParams
from navsafe.traffic.geometry import ROUTE_WIDTH_M, Route, corners
from navsafe.traffic.base import TrafficManager

logger = logging.getLogger(__name__)

# --- ScenarioTrafficManager class constants (L27-33) ----------------------
STATIC_THRESHOLD_M = 3.0            # static if max positional std <= this
IDM_ACT_BATCH_SIZE = 5              # speed control runs 1-in-N steps
IDM_CREATE_SIDE_CONSTRAINT_M = 15.0
IDM_CREATE_FORWARD_CONSTRAINT_M = -1.0
IDM_CREATE_MIN_LENGTH_M = 5.0
# get_idm_route(traj_points, width=2) -- the corridor the lead must be inside.
IDM_ROUTE_WIDTH_M = ROUTE_WIDTH_M

#: Lateral half-band, beyond the trajectory corridor, in which the EGO alone is
#: still treated as something to brake for.
#:
#: Upstream asks only "is it inside my 2 m corridor", which is right for logged
#: traffic and wrong for a closed-loop ego: the ego is by definition somewhere
#: the log never put it, so it can converge on a reactive vehicle while sitting
#: outside that corridor. Measured on 05d0a1a763fc5334 with drivor: the ego was
#: 2.72 m lateral and ~6 m ahead of 1235f522a2cb5ade, which semi_reactive HAD
#: taken over at frame 1 (7.3 m behind, 4.2 m lateral, continuous mode). The
#: corner test rejected the ego every frame, `_front_object` returned None, IDM
#: accelerated freely, and it rear-ended the ego at 7.6 m/s -- a
#: `contact_not_at_fault` the ego could not avoid and the actor never braked
#: for. Takeover was never the problem; the lead search was.
#:
#: 3.0 m is half an actor width plus half an ego width plus ~1.2 m of margin:
#: enough for a neighbouring-lane conflict, not enough to notice a car two
#: lanes over. Widening the corridor itself would also make every reactive
#: vehicle brake for ordinary overtaking traffic, a behaviour change in every
#: scenario rather than a fix for this failure.
EGO_CONFLICT_LAT_M = 3.0

# --- TrajectoryIDMPolicy / IDMPolicy constants (idm_policy.py L178-223, 436-440)
NORMAL_SPEED_KMH = 40.0             # TrajectoryIDMPolicy.NORMAL_SPEED
IDM_MAX_DIST_M = 20.0               # TrajectoryIDMPolicy.IDM_MAX_DIST
DEST_REGION_RADIUS_M = 2.0          # TrajectoryIDMPolicy.DEST_REGION_RADIUS

#: Metres of straight road appended to a reactive vehicle's logged route, so it
#: keeps driving after the log runs out instead of parking on the last frame.
#:
#: A bundle is a 20 s window and upstream despawns (here: freezes) a vehicle
#: within DEST_REGION_RADIUS_M of its final logged pose. An episode that runs
#: past 20 s therefore fills with stationary traffic, which is not a scenario --
#: the ego meets a car park. Extending the route along the final heading keeps
#: IDM in charge: the vehicle holds its speed, still brakes for whatever is in
#: front of it, and still yields to the ego.
#:
#: A straight extrapolation is deliberately dumb. It does not follow the lane
#: graph, so a vehicle whose log ended mid-turn will continue straight out of
#: its lane, and past the extension it freezes as before. Both are visible in
#: the render rather than silent, and neither pretends to be logged behaviour:
#: the extension exists so post-window traffic keeps MOVING, not so it is
#: realistic.
ROUTE_EXTENSION_M = 400.0

#: How far a vehicle must get from where it is standing before a logged sample
#: counts as travel rather than drift, in metres.
#:
#: `Route`'s own MIN_SEGMENT_M floor asks a purely geometric question — is this
#: step longer than localization noise — and that is not enough for a vehicle
#: that is STOPPED. A car waiting at a junction wanders by decimetres over
#: several seconds; the floor eventually accumulates 0.2 m of that wander and
#: emits it as a segment, indistinguishable by length from 0.2 m of real
#: travel, and pointing wherever the drift happened to go.
#:
#: Measured on bundle ``131a036a111e54f3`` (V-9 Blocking Intersection, whose
#: agents are `stationary_in_traffic` and therefore nearly all start stopped):
#: with the floor alone, 6 of 31 takeover-eligible vehicles still turned more
#: than 90 deg between adjacent kept segments, worst 170.6 deg. Every one of
#: them was the same shape — the turn at s ~ 0.2 m, both segments 0.20-0.26 m
#: (i.e. just over the floor), the vehicle doing 0.04-0.24 m/s. Raising the
#: floor only raises the amount of drift that has to accumulate.
#:
#: So the leading stopped stretch is dropped from the SEED instead, and the
#: route's first segment runs from where the vehicle stands to where it
#: actually got to. 2.0 m is far above any drift seen here and well under
#: IDM_CREATE_MIN_LENGTH_M, so a vehicle that clears the takeover gate always
#: has a departure point to aim at.
#:
#: This bounds the error rather than removing it. The chord starts where the
#: vehicle IS, so it still absorbs whatever the vehicle drifts between now and
#: departing: the same six agents came out at 2.9-17.8 deg, i.e. up to ~0.65 m
#: of drift across the 2 m chord. That is an error in the direction of travel,
#: and it shrinks as DEPARTURE_M grows; what it replaces is unbounded and
#: random. (Localization noise proper, at cm scale, contributes under 0.2 deg
#: here and is not the term that matters.)
DEPARTURE_M = 2.0

#: A speed gate on the trim was tried here and REJECTED on measurement; it is
#: recorded so it is not re-invented.
#:
#: The worry is real in principle — the trim replaces what it drops with one
#: straight chord, which cuts the corner off a vehicle that was already driving
#: through a turn. The gate was "only trim if reaching DEPARTURE_M took at
#: least 20 samples (2 s at 10 Hz, so under ~1 m/s)". On
#: ``131a036a111e54f3`` it fired on three agents the trim was helping
#: (1656599f0a9e5a6d 17.1 deg -> 80.4, b8d1163e92285179 24.0 -> 32.2,
#: 9c10f281744d50ab 2.1 -> 3.7) and did NOT fire on the one it was written for
#: (6aaf673c5b0157e3, which takes 21 samples). Departure speed does not
#: separate the two populations on this data.
#:
#: The chord's real cost is small and bounded: 2 m across a turn of radius 8 m
#: deviates ~0.06 m from the arc. What it replaces is unbounded and random.
DISTANCE_WANTED_M = 10.0            # IDMPolicy.DISTANCE_WANTED
TIME_WANTED_S = 1.5                 # IDMPolicy.TIME_WANTED
DELTA = 10.0                        # IDMPolicy.DELTA
ACC_FACTOR = 1.0                    # IDMPolicy.ACC_FACTOR
DEACC_FACTOR = 5.0                  # |IDMPolicy.DEACC_FACTOR|

KMH_PER_MS = 3.6

_VEHICLE_TYPES = {"VEHICLE", "vehicle", "CAR", "car", "TRUCK", "truck",
                  "BUS", "bus"}

# Kept for callers that still import the old names; the values are MetaDrive's.
DEFAULT_TRIGGER_RADIUS_M = IDM_CREATE_SIDE_CONSTRAINT_M
DEFAULT_MIN_MOVING_DIST_M = IDM_CREATE_MIN_LENGTH_M


def metadrive_idm_params(*, metadrive_units: bool = True) -> IDMParams:
    """MetaDrive's ``TrajectoryIDMPolicy`` constants as :class:`IDMParams`.

    The IDM expression in ``IDMPolicy.acceleration``/``desired_gap`` is the
    standard one, so the shared actor computes it; only the constants and the
    unit convention are MetaDrive's.
    """
    return IDMParams(
        v0=NORMAL_SPEED_KMH if metadrive_units else NORMAL_SPEED_KMH / KMH_PER_MS,
        s0=DISTANCE_WANTED_M, T=TIME_WANTED_S,
        a=ACC_FACTOR, b=DEACC_FACTOR, delta=DELTA,
    )


# _Route/_corners moved to navsafe.traffic.geometry so the recipe-driven
# manager reuses them instead of carrying a second copy with its own sign
# conventions. Aliased here to keep this module's internals unchanged.
_Route = Route


class _IDMVehicle:
    """One taken-over vehicle: its route, longitudinal state and policy slot."""

    __slots__ = ("route", "s", "v", "policy_index", "last_accel", "length",
                 "width", "arrived")

    def __init__(self, route: _Route, s: float, v: float, policy_index: int,
                 length: float, width: float) -> None:
        self.route = route
        self.s = float(s)
        self.v = float(v)
        self.policy_index = int(policy_index)
        self.last_accel = 0.0
        self.length = float(length)
        self.width = float(width)
        self.arrived = False

    def pose(self) -> Tuple[np.ndarray, float]:
        return self.route.position_at(self.s), self.route.heading_at(self.s)


_corners = corners



class SemiReactiveTraffic(TrafficManager):
    """Log replay, with MetaDrive's reactive-traffic takeover for followers."""

    def __init__(self, params: IDMParams | None = None, *,
                 side_constraint_m: float = IDM_CREATE_SIDE_CONSTRAINT_M,
                 forward_constraint_m: float = IDM_CREATE_FORWARD_CONSTRAINT_M,
                 min_length_m: float = IDM_CREATE_MIN_LENGTH_M,
                 act_batch_size: int = IDM_ACT_BATCH_SIZE,
                 takeover: str = "continuous",
                 metadrive_units: bool = True,
                 exclude_ids: Optional[Set[Any]] = None) -> None:
        """``params`` overrides the IDM constants; leaving it None uses
        MetaDrive's.  Passing NexusSim's own ``idm_*`` knobs here would
        silently make the two traffic modes differ in more than reactivity.

        ``takeover`` is ``"continuous"`` (re-test every frame; the default) or
        ``"spawn"`` (test once, MetaDrive's behaviour).  See the module
        docstring.

        ``exclude_ids`` names agents this manager must never adopt. It exists
        for actors another manager already owns: a NavSafe recipe's inserted
        car is a track like any other, so the geometric takeover test would
        happily adopt it and drive it as background traffic — the actor the
        scenario is ABOUT would then follow MetaDrive's plausibility rules
        instead of the policy the recipe authored. The set is mutable after
        construction because which actors a recipe owns is known only once the
        scenario edits have been applied.
        """
        if takeover not in ("continuous", "spawn"):
            raise ValueError(
                f"takeover must be 'continuous' or 'spawn', got {takeover!r}")
        self.takeover = takeover
        self.metadrive_units = bool(metadrive_units)
        self.exclude_ids: Set[Any] = set(exclude_ids or ())
        self.params = params if params is not None else metadrive_idm_params(
            metadrive_units=self.metadrive_units)
        self.actor = IDMActor(self.params)
        self.side_constraint_m = float(side_constraint_m)
        self.forward_constraint_m = float(forward_constraint_m)
        self.min_length_m = float(min_length_m)
        self.act_batch_size = max(1, int(act_batch_size))

        self._idm: Dict[Any, _IDMVehicle] = {}
        # Learned in step() from the agent manager; None until then, which the
        # lead search treats as "no ego to brake for".
        self._ego_id: Any = None
        # Vehicles that have already logged an out-of-corridor ego conflict, so
        # the line appears once per vehicle instead of once per acting step.
        self._conflict_logged: Set[Any] = set()
        self._seen: Set[Any] = set()          # tested at least once
        # Failed a whole-track rule (type / static / degenerate track): can
        # never become eligible, so continuous mode skips it without redoing
        # the per-track statistics every frame.
        self._ineligible: Set[Any] = set()
        self._static: Set[Any] = set()
        # agent_id -> (positions, valid), parsed once per episode.
        self._track_cache: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        self._idm_policy_count = 0
        self._episode_step = 0
        # Read by NexusSimEnv._collect_agent_states_for_renderer: current poses
        # of taken-over agents, overriding the logged track state.
        self.pose_overrides: Dict[Any, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # TrafficManager interface
    # ------------------------------------------------------------------

    def reset(self, env: Any) -> None:
        self._idm.clear()
        self._seen.clear()
        self._ineligible.clear()
        self._static.clear()
        self._track_cache.clear()
        self._conflict_logged.clear()
        self._ego_id = None
        self._idm_policy_count = 0
        self._episode_step = 0
        self.pose_overrides.clear()
        env.agent_manager.reset(env.sim.stage, timestep=0)

    def step(self, env: Any, dt: float) -> None:
        manager = env.agent_manager
        t = int(env.scenario_timestep)
        ego_state = env.get_ego_state()
        ego_id = getattr(manager, "ego_agent_id", None)
        # Kept for the lead search, which needs to know which object is the ego
        # before it may brake for one outside its corridor.
        self._ego_id = ego_id
        ego_pos = np.asarray(ego_state["position"][:2], dtype=np.float64)
        ego_heading = float(ego_state.get("heading", 0.0))
        tracks = env.current_scenario.get("tracks", {})

        # 1. Takeover test: at the first valid frame ("spawn"), or at every
        #    frame until it passes ("continuous"). Takeover is one-way, so a
        #    vehicle already in _idm is never re-tested under either setting.
        for agent_id, track in tracks.items():
            if agent_id == ego_id or agent_id in self._idm:
                continue
            if agent_id in self.exclude_ids:
                continue                      # owned by another manager
            if agent_id in self._ineligible:
                continue
            if self.takeover == "spawn" and agent_id in self._seen:
                continue
            # Past the end of the log there is no state at `t`: the lookup
            # returns None (or an invalid row) and every remaining vehicle was
            # skipped here, before _decide could hand it to IDM -- so an episode
            # running beyond the bundle window kept a field of frozen traffic.
            # Clamp the probe to the last logged frame; `_decide` sees the same
            # `t` and takes the past-log branch.
            arrays = self._track_arrays(agent_id, track)
            probe = t
            if arrays is not None:
                probe = min(t, len(arrays[1]) - 1)
            state = manager._get_state_at_timestep(track, probe)
            if state is None or not state.get("valid", True):
                continue
            self._seen.add(agent_id)
            self._decide(agent_id, track, t, state, ego_pos, ego_heading)

        # 2. Replay everyone the decision left on the log.
        skip_ego = bool(getattr(env, "_ego_override_active", False))
        # Excluded agents are skipped from the replay too: their owner
        # publishes a pose for them every frame, and replaying their logged
        # track underneath it would fight that owner on the USD prim.
        manager.update_agents(stage=env.sim.stage, timestep=t,
                              skip_ego=skip_ego,
                              skip_agents=set(self._idm) | set(self.exclude_ids))

        # 3. Act for the reactive vehicles.
        if self._idm:
            objects = self._world_objects(manager, tracks, t, ego_id,
                                          ego_pos, ego_heading, ego_state)
            self.pose_overrides = {}
            for agent_id, veh in self._idm.items():
                self._act(agent_id, veh, objects, dt)
                pos, heading = veh.pose()
                manager.set_agent_pose(env.sim.stage, agent_id, pos, heading)
                self.pose_overrides[agent_id] = {
                    "position": np.array([pos[0], pos[1], 0.0]),
                    "heading": heading,
                    # A 2-VECTOR, as every other driver emits (idm_driver,
                    # social_force). `veh.v` is IDM's scalar longitudinal
                    # speed, and this module's own `_speed_of` accepts either
                    # form -- but `pose_overrides` is a shared contract that
                    # leaves this module, and the EPDMS scorer's agent view
                    # does `velocity[:2]`. A scalar there raised
                    # `TypeError: 'float' object is not subscriptable` on EVERY
                    # scored frame, which dropped the whole episode from EPDMS,
                    # took driving_score_summary.csv with it, and left the
                    # NavSafe taxonomy no per-frame contact record to classify
                    # a collision from.
                    "velocity": np.array([veh.v * math.cos(heading),
                                          veh.v * math.sin(heading)],
                                         dtype=np.float64),
                    "length": veh.length,
                    "width": veh.width,
                }
        self._episode_step += 1

    # ------------------------------------------------------------------
    # spawn-time decision (ScenarioTrafficManager.spawn_vehicle)
    # ------------------------------------------------------------------

    def _track_arrays(self, agent_id: Any, track: Dict
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(positions, valid)`` for a track that passed the whole-track rules.

        Logged tracks do not change during an episode, so both the parse and
        the static classification are done once and cached -- continuous mode
        re-tests a vehicle every frame and would otherwise re-run ``std`` over
        the full track each time.  Returns None (and marks the vehicle
        permanently ineligible) if the track is degenerate or static.
        """
        cached = self._track_cache.get(agent_id)
        if cached is not None:
            return cached

        st = track.get("state", {})
        positions = np.asarray(st.get("position", []), dtype=np.float64)
        if positions.ndim != 2 or len(positions) < 2:
            self._ineligible.add(agent_id)
            return None
        valid = np.asarray(st.get("valid", [True] * len(positions)), dtype=bool)
        if not valid.any():
            self._ineligible.add(agent_id)
            return None

        # Static classification: std of the valid positions, not net travel --
        # a car that loops back to where it started is still moving.
        pts = positions[valid][:, :2]
        if float(np.max(np.std(pts, axis=0))) <= STATIC_THRESHOLD_M:
            self._static.add(agent_id)
            self._ineligible.add(agent_id)
            return None

        self._track_cache[agent_id] = (positions, valid)
        return positions, valid

    def _decide(self, agent_id: Any, track: Dict, t: int, state: Dict,
                ego_pos: np.ndarray, ego_heading: float) -> None:
        """Test one vehicle for takeover at frame ``t``.

        Failures that are properties of the whole track mark the vehicle
        ``_ineligible`` so continuous mode never re-tests it; failures that
        depend on ``t`` or the ego's pose simply return, leaving the vehicle
        eligible for a later frame.
        """
        if str(track.get("type", "VEHICLE")) not in _VEHICLE_TYPES:
            self._ineligible.add(agent_id)            # pedestrians/cyclists replay
            return

        arrays = self._track_arrays(agent_id, track)
        if arrays is None:
            return
        positions, valid = arrays

        # Past the end of its log a track has no logged behaviour left to
        # replay: replay CLAMPS to the last valid frame, so every remaining
        # vehicle stands still and an episode that runs beyond the bundle
        # window fills with parked traffic. There is nothing to be faithful to
        # out there, so the follower-only gates below (which exist to leave
        # LOGGED behaviour alone wherever possible) stop applying and every
        # eligible vehicle is handed to IDM.
        past_log = t >= len(valid) - 1
        # Every index below is into the LOGGED arrays, so past the log they must
        # be clamped to the last logged frame -- `t` itself keeps counting up
        # with the episode and would run off the end.
        ti = min(t, len(valid) - 1)

        # The contiguous valid window starting at this frame (get_max_valid_indicis).
        end = len(valid)
        for i in range(ti + 1, len(valid)):
            if not valid[i]:
                end = i
                break
        if end - ti < 2 and not past_log:
            return

        moving = positions[ti][:2] - positions[end - 1][:2]
        if (float(np.hypot(moving[0], moving[1])) <= self.min_length_m
                and not past_log):
            return                                    # length_ok

        pos = np.asarray(state.get("position", [0.0, 0.0])[:2], dtype=np.float64)
        heading = float(state.get("heading", 0.0))
        if (abs(_wrap_to_pi(ego_heading - heading)) >= math.pi / 2
                and not past_log):
            return                                    # heading_ok

        # Ego-local coordinates: forward first, then left-positive lateral.
        d = pos - ego_pos
        heading_dist = math.cos(ego_heading) * d[0] + math.sin(ego_heading) * d[1]
        side_dist = -math.sin(ego_heading) * d[0] + math.cos(ego_heading) * d[1]
        if not (heading_dist < self.forward_constraint_m
                and abs(side_dist) < self.side_constraint_m) and not past_log:
            return                                    # idm_ok

        # Past the log the remaining logged window is empty, so seed the route
        # from the last valid pose and let _extend_route supply the road ahead.
        seed = positions[ti:end][:, :2]
        if past_log or len(seed) < 2:
            tail = positions[max(0, len(positions) - 6):][:, :2]
            seed = tail if len(tail) >= 2 else positions[-2:][:, :2]
        # The logged heading is the fallback because it is the one statement
        # about this vehicle's orientation that does NOT come from the same
        # positions the route is built out of.
        route = _Route(_extend_route(_trim_leading_stop(seed)),
                       fallback_heading=heading)
        if route.length <= 0:
            return
        # The takeover frame is the one thing about this manager that is not
        # reconstructible from the artifacts afterwards, and under
        # takeover="continuous" it is no longer simply the spawn frame.
        logger.info(
            "semi_reactive: %s taken over at frame %d (%s%s) — %.1f m behind, "
            "%.1f m lateral, route %.1f m",
            agent_id, t, self.takeover, ", past log" if past_log else "",
            -heading_dist, side_dist, route.length)
        self._idm[agent_id] = _IDMVehicle(
            route=route, s=0.0, v=_speed_of(state),
            policy_index=self._idm_policy_count % self.act_batch_size,
            length=float(state.get("length", 4.5)),
            width=float(state.get("width", 1.8)),
        )
        self._idm_policy_count += 1

    # ------------------------------------------------------------------
    # per-step control (TrajectoryIDMPolicy.act)
    # ------------------------------------------------------------------

    def _world_objects(self, manager: Any, tracks: Dict, t: int, ego_id: Any,
                       ego_pos: np.ndarray, ego_heading: float,
                       ego_state: Dict) -> List[Dict[str, Any]]:
        """Everything a reactive vehicle can see: the ego, the replaying
        agents at their logged state, and the other reactive vehicles at their
        integrated state (MetaDrive's ``lidar.get_surrounding_objects``)."""
        objs: List[Dict[str, Any]] = [{
            "id": ego_id, "position": ego_pos, "heading": ego_heading,
            "speed": _speed_of(ego_state),
            "length": float(ego_state.get("length", 4.5)),
            "width": float(ego_state.get("width", 1.8)),
        }]
        for agent_id, track in tracks.items():
            if agent_id == ego_id or agent_id in self._idm:
                continue
            state = manager._get_state_at_timestep(track, t)
            if state is None or not state.get("valid", True):
                continue
            objs.append({
                "id": agent_id,
                "position": np.asarray(state["position"][:2], dtype=np.float64),
                "heading": float(state.get("heading", 0.0)),
                "speed": _speed_of(state),
                "length": float(state.get("length", 4.5)),
                "width": float(state.get("width", 1.8)),
            })
        for agent_id, veh in self._idm.items():
            pos, heading = veh.pose()
            objs.append({"id": agent_id, "position": pos, "heading": heading,
                         "speed": veh.v, "length": veh.length,
                         "width": veh.width})
        return objs

    def _front_object(self, agent_id: Any, veh: _IDMVehicle,
                      objects: List[Dict[str, Any]], *, ego_id: Any = None
                      ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Nearest object ahead inside the trajectory corridor, within 20 m.

        ``get_find_front_back_objs_single_lane``: Euclidean pre-filter at
        ``max_distance``, then the object must have a bounding-box corner on
        the lane polygon, then the smallest positive longitudinal difference
        in lane coordinates.  The distance handed to IDM is that longitudinal
        difference -- centre to centre, not bumper to bumper.
        """
        pos, _ = veh.pose()
        own_long, _ = veh.route.local_coordinates(pos)
        best, best_long = None, IDM_MAX_DIST_M
        # The ego when it is near-but-not-in the corridor, kept apart so a real
        # in-corridor lead always wins (see EGO_CONFLICT_LAT_M).
        conflict, conflict_long, conflict_lat = None, IDM_MAX_DIST_M, 0.0
        for obj in objects:
            if obj["id"] == agent_id:
                continue
            op = np.asarray(obj["position"], dtype=np.float64)[:2]
            if float(np.hypot(*(op - pos))) > IDM_MAX_DIST_M:
                continue
            long, lat = veh.route.local_coordinates(op)
            long -= own_long
            if not any(veh.route.point_on_lane(c) for c in
                       _corners(float(op[0]), float(op[1]), float(obj["heading"]),
                                float(obj["length"]), float(obj["width"]))):
                # Outside the corridor. The ego alone earns a second look, only
                # ahead, and only while CLOSING: braking for an ego that is
                # pulling away would stall the actor, and a stalled actor is its
                # own artefact. (No blind_to_ego test here — that is a recipe
                # property of navsafe.traffic.navsafe actors; every vehicle this
                # manager drives is ordinary logged background.)
                if (ego_id is not None and obj["id"] == ego_id
                        and abs(lat) <= EGO_CONFLICT_LAT_M
                        and 0 < long < conflict_long
                        and self._closing_rate(veh, obj) > 0.0):
                    conflict, conflict_lat = obj, lat
                    conflict_long = long
                continue
            if 0 < long < best_long:
                best_long, best = long, obj
        if best is None and conflict is not None:
            # Once per vehicle, not once per frame: it fires every acting step
            # for as long as the conflict lasts, and 180 identical lines would
            # bury the takeover decisions this log exists to show.
            if agent_id not in self._conflict_logged:
                self._conflict_logged.add(agent_id)
                logger.info("semi_reactive: %s braking for the ego %.1f m ahead, "
                            "%.1f m lateral (outside its %.1f m corridor)",
                            agent_id, conflict_long, conflict_lat,
                            veh.route.width)
            return conflict, conflict_long
        return best, best_long

    def _closing_rate(self, veh: _IDMVehicle, obj: Dict[str, Any]) -> float:
        """Relative velocity projected on this vehicle's heading, m/s.

        ``IDMPolicy.desired_gap``'s own quantity: positive means the gap is
        shrinking. Used both to hand IDM its closing rate and to decide whether
        an out-of-corridor ego is worth braking for at all.
        """
        heading = veh.route.heading_at(veh.s)
        oh = float(obj["heading"])
        return ((veh.v * math.cos(heading) - obj["speed"] * math.cos(oh))
                * math.cos(heading)
                + (veh.v * math.sin(heading) - obj["speed"] * math.sin(oh))
                * math.sin(heading))

    def _act(self, agent_id: Any, veh: _IDMVehicle,
             objects: List[Dict[str, Any]], dt: float) -> None:
        if veh.arrived:
            veh.v = 0.0
            return

        # Batched speed control: one policy slot acts per step, the rest hold
        # their last acceleration (ScenarioTrafficManager.before_step).
        if self._episode_step % self.act_batch_size == veh.policy_index:
            front, gap = self._front_object(agent_id, veh, objects,
                                            ego_id=self._ego_id)
            scale = KMH_PER_MS if self.metadrive_units else 1.0
            if front is None:
                accel = self.actor.compute_acceleration(veh.v * scale, 0.0,
                                                        float("inf"))
            else:
                # Projected closing rate, as in IDMPolicy.desired_gap: the
                # relative velocity vector along this vehicle's heading.
                dv = self._closing_rate(veh, front)
                accel = self.actor.compute_acceleration(
                    veh.v * scale, dv * scale, max(gap, 1e-3))
            veh.last_accel = float(accel)
        accel = veh.last_accel

        veh.v = max(0.0, veh.v + accel * dt)
        veh.s = min(veh.s + veh.v * dt, veh.route.length)
        if float(np.hypot(*(veh.route.position_at(veh.s) - veh.route.end))) \
                < DEST_REGION_RADIUS_M:
            # MetaDrive despawns here; holding the final pose keeps the prim
            # count stable for the renderer.
            veh.arrived = True
            veh.v = 0.0


def _trim_leading_stop(xy: np.ndarray,
                       departure_m: float = DEPARTURE_M) -> np.ndarray:
    """``xy`` with the samples the vehicle logged while STANDING STILL removed.

    The first point is kept — it is where the vehicle is, so it has to be where
    its route starts — and everything between it and the first sample
    ``departure_m`` away from it is dropped. The route's opening segment then
    runs from the vehicle to somewhere it demonstrably drove, instead of being
    read off the wander of a stationary car (see DEPARTURE_M).

    Returned unchanged when the track never departs: there is no travel
    direction to recover, and the floor plus the caller's fallback heading are
    what handle that. The takeover gate (``min_length_m``) normally makes it
    impossible anyway.
    """
    if len(xy) < 3 or departure_m <= 0.0:
        return xy
    away = np.linalg.norm(xy[1:] - xy[0], axis=1) >= departure_m
    if not away.any():
        return xy
    return np.vstack([xy[0], xy[int(np.argmax(away)) + 1:]])


def _extend_route(xy: np.ndarray) -> np.ndarray:
    """The logged polyline plus ROUTE_EXTENSION_M of straight road at its end.

    The heading comes from the last few metres rather than the final pair of
    points: consecutive log samples at 10 Hz can be centimetres apart, and a
    heading taken from two of them is dominated by position noise, which would
    send the extension off at a visible angle.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2 or ROUTE_EXTENSION_M <= 0:
        return xy
    tail = xy[-1] - xy[max(0, len(xy) - 6)]
    n = float(np.hypot(*tail))
    if n < 1e-6:                       # a parked track: nothing to extrapolate
        return xy
    step = tail / n
    return np.vstack([xy, xy[-1] + step * ROUTE_EXTENSION_M])


def _speed_of(state: Dict[str, Any]) -> float:
    vel = state.get("velocity", 0.0)
    if isinstance(vel, (int, float)):
        return float(vel)
    arr = np.asarray(vel, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(arr[:2])) if arr.size else 0.0


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["SemiReactiveTraffic", "metadrive_idm_params",
           "IDM_CREATE_SIDE_CONSTRAINT_M", "IDM_CREATE_FORWARD_CONSTRAINT_M",
           "IDM_CREATE_MIN_LENGTH_M", "IDM_ACT_BATCH_SIZE",
           "STATIC_THRESHOLD_M", "DEFAULT_TRIGGER_RADIUS_M",
           "DEFAULT_MIN_MOVING_DIST_M"]
