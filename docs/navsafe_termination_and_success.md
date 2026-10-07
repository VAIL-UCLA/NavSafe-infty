# NavSafe: Termination state + SR

```
frames (trace) -> termination.classify      ONE reason, the earliest event
               -> infractions_from_trace    counted contacts (fault-respecting)
               -> route.route_progress      monotone route completion
               -> termination.to_route_result   RouteResult, or None (excluded)
               -> RouteResult.success()     SR
```
[`scoring/from_run.py::score_run`](../navsafe/benchmark/scoring/from_run.py#L180)
is the only place the chain is assembled; the CLI
(`navsafe/tools/score_run.py`) and the in-run path (the evaluator)
both call it.

---

## 1. Termination

### 1.1 The reasons

[`termination.py::TerminationReason`](../navsafe/benchmark/termination.py#L57).
Exactly one per episode. It answers "why did stepping stop", never "what did
the ego do wrong" — everything countable is an infraction (§2.2).

| reason | fires when | policy-attributed | scored? |
|---|---|---|---|
| `goal_reached` | `route.at_goal()` held (§1.3) | yes | yes, `completed=True` |
| `budget_expired` | nothing else ended it and elapsed scored time ≥ `t_max` | yes | yes, adds `route_timeout` |
| `deadlock` | ego speed < 0.1 m/s for 5 s continuous | yes | yes, adds `vehicle_blocked` |
| `contact_at_fault` | a contact with `at_fault=True` | yes | yes (contact counted by §2.2) |
| `contact_not_at_fault` | a contact with `at_fault=False` | **no** | yes — ended, but the contact is not counted |
| `off_drivable` | frame `on_drivable=False` (EPDMS `DAC < 1.0`, map required) | yes | yes, adds `route_dev` |
| `wrong_way` | frame `driving_direction_ok=False` (EPDMS `DDC == 0.0`, > 6 m against traffic, map required) | yes | yes, adds `wrong_way` |
| `trace_exhausted` | nothing else ended it and elapsed < `t_max` — harness stopped, not the ego | no | yes |
| `infra_failure` | simulator/renderer error, or no scored frames | no | **no** — `RouteResult` is `None`, status `excluded` |
| `envelope_exit` | retired; appears only in older artifacts | no | no |

Thresholds:
[`DEADLOCK_SPEED_MS = 0.1`, `DEADLOCK_HOLD_S = 5.0`](../navsafe/benchmark/termination.py#L53-L54);
[`DDC_HALF_VIOLATION_M = 2.0`, `DDC_FULL_VIOLATION_M = 6.0`](../navsafe/evaluation/scorers/epdms_trajectory_scorer_fast.py#L111-L112)
(only the full violation terminates; 0.5 is a brush, not an ending).
`policy_attributed` is
[here](../navsafe/benchmark/termination.py#L101).


### 1.3 Goal

[`scoring/route.py::at_goal`](../navsafe/benchmark/scoring/route.py#L61):
remaining route arc ≤ `max(2.0 m, 1 % of total arc)` **and** distance to the
goal point < 10 m
([`COMPLETION_PCT_FOR_DONE = 99.0`, `GOAL_RADIUS_M = 10.0`, `GOAL_ARC_TOL_M = 2.0`](../navsafe/benchmark/scoring/route.py#L34-L49)).
Route completion is a monotone cursor along the logged ego path
([`route_progress`](../navsafe/benchmark/scoring/route.py#L100)).
The live evaluator calls the same predicate over the same dense path
([`route_manager.py::goal_reached`](../navsafe/evaluation/route_manager.py#L65)).
If the evaluator's `metrics.json` says `goal_reached` but the recompute
disagrees, the report is honoured with `goal_frame=None` and a note
([here](../navsafe/benchmark/scoring/from_run.py#L213-L218)).

### 1.4 Time budget

`t_max` = [`SAFETY_CEILING_S = 60.0`](../navsafe/benchmark/scoring/from_run.py#L51)
unless `--t-max` is passed. The rubric's "0.95-quantile human × 1.3, floor
10 s" (`rubric/calibrate_tmax.py`) is **not** used by this chain.
Windows shorter than `t_max` get a `scored window is … but t_max is …` note.

---

## 2. Success Rate

### 2.1 The rule

[`scoring/metrics.py::RouteResult.success`](../navsafe/benchmark/scoring/metrics.py#L117-L135)
— Bench2Drive `merge_route_json.py` L20-27, one rule for every scenario and
every event type:

```
success  =  completed
            AND every infraction count is 0, except min_speed_infractions
            AND outside_route_lanes == 0
```

`completed` is `termination.reason is GOAL_REACHED`
([`to_route_result`](../navsafe/benchmark/termination.py#L278-L281)).
Route completion percentage alone never grants success.
SR over a set =
[`success_rate`](../navsafe/benchmark/scoring/metrics.py#L155),
with an explicit denominator; excluded episodes (`RouteResult is None`) are
outside it, never 0.

### 2.2 What can be in `infractions`

| key | source | in DS penalty | fails SR |
|---|---|---|---|
| `collisions_pedestrian` | contact kind `vru`, at-fault only, once per (agent, kind) pair — [`infractions_from_trace`](../navsafe/benchmark/scoring/metrics.py#L547) | ×0.50 | yes |
| `collisions_vehicle` | kinds `rear_end` / `angle` / `sideswipe`; or the evaluator's `collision_count` when no frame places a contact ([fallback](../navsafe/benchmark/scoring/from_run.py#L256-L258)) | ×0.60 | yes |
| `collisions_layout` | kind `single` | ×0.65 | yes |
| `red_light` | one per contiguous run of `signal_state == "red"` frames ([here](../navsafe/benchmark/scoring/from_run.py#L265-L272)); **only when the artifact has a `TL` column** | ×0.70 | yes |
| `outside_route_lanes` | % of driven distance off-drivable beyond 0.5 m per excursion ([`outside_route_lanes_pct`](../navsafe/benchmark/scoring/route.py#L143)); map required | ×(1 − pct/100) | yes if > 0 |
| `route_dev` | termination `off_drivable` | no | yes |
| `vehicle_blocked` | termination `deadlock` | no | yes |
| `route_timeout` | termination `budget_expired` | no | yes |
| `wrong_way` | termination `wrong_way` (NavSafe's; no B2D coefficient) | no | yes |
| `stop_infraction`, `scenario_timeouts`, `yield_emergency_vehicle_infractions` | **no source** — listed under `penalty_channels_skipped` | — | never |
| `min_speed_infractions` | disabled upstream | — | exempt |

Reason → infraction map:
[`_REASON_TO_INFRACTION`](../navsafe/benchmark/termination.py#L240-L255).
Coefficients:
[`PENALTY_COEFFICIENTS`](../navsafe/benchmark/scoring/metrics.py#L58-L66),
[`NON_PENALTY_KEYS`](../navsafe/benchmark/scoring/metrics.py#L84).
Channels actually evaluated vs skipped:
[`_CHANNEL_SOURCES`](../navsafe/benchmark/scoring/from_run.py#L91-L101).



### 2.4 Not-at-fault contacts

`contact_not_at_fault` ends the episode (the world is no longer meaningful)
but adds no infraction and is not policy-attributed. `completed` is False
(the reason is not `GOAL_REACHED`), so SR is **false** for that run, with DS =
route completion at the contact frame × whatever penalty the run had earned.
Seen on `05d0a1a763fc5334/drivor` (replay agent hit the ego at frame 158).

---

