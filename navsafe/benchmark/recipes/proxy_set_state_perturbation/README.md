# The state-perturbation set

56 controlled events on 28 scenarios of the published `full_test` set: one
scenario per event type, two events per scenario. Each event places a single
manipulated actor on an authored path in front of the ego, so that a change in
the outcome can be attributed to that actor.

| event | cells | what the actor does |
|---|---|---|
| `lead_brake` | 22 | a lead vehicle brakes in the ego lane |
| `vru_cross` | 18 | a vulnerable road user crosses the ego route |
| `cut_in` | 16 | a vehicle cuts into the ego lane |

Every event has two frozen recipes, named
`<event-type-id>.<token>.<event>.<variant>.yaml`:

- `event` — the actor performs the event (`policy.enabled: true`).
- `baseline` — the same actor stays in the scene with the event switched off
  (`policy.enabled: false`): a `lead_brake` lead keeps its speed, a `vru_cross`
  actor waits at the start of its path, and a `cut_in` vehicle continues
  straight instead of following the cut-in curve.

A baseline is neither the logged scene nor the recipe with its actor removed.
The two recipes of a pair differ only in `policy.enabled`, except for
`I-1.6fc91bf02f225d1b.lead_brake` and `V-9.3a5278b27c87565f.lead_brake`, whose
authored paths also differ.

## Files

- `*.yaml` — the 112 recipes. They are frozen: each actor is checksum-pinned, and
  assets and gait banks resolve against `NAVSAFE_DATA_ROOT`.
- `index.json` — one entry per event: its scenario token, the bundle paths
  relative to `NAVSAFE_DATA_ROOT`, both recipes and the `prepared_sha256` of each
  recipe file. `excluded_events` lists the two events that are not part of the
  set.
- `source_manifest.json` — which events were authored for each scenario.

In 15 events the manipulated actor is a vehicle from the log, moved onto the
authored path while keeping its track id and dimensions. Those entries name the
bundle's `ah_assets/replace_manifest.json`, which supplies that vehicle's own
harvested asset so that it renders away from its logged poses. The evaluator
refuses such a recipe without the manifest and writes
`harvester_takeover_audit.json` naming the tracks it replaced in each scene.

## Run

Download the 28 bundles, `asset/` and `gait_bank/` as in the
[evaluation guide](../../../../docs/navsafe_eval.md#download-the-data), then:

```bash
export NAVSAFE_RENDER_GPU=0 NAVSAFE_GPU=1       # renderer GPU, evaluator GPU
navsafe perturbation "<model-type>" "<checkpoint-path>"
```

This runs the 56 `event` recipes. `NAVSAFE_VARIANT=baseline` runs the baselines
and `NAVSAFE_VARIANT=both` all 112. The script verifies every recipe against its
`prepared_sha256` before starting, and accepts the renderer, shard, retry and
resume variables of the
[full benchmark sweep](../../../../docs/navsafe_eval.md#full-benchmark-sweep).
Results go to `output/perturbation_<model-type>/`, one directory per cell, with
`summary.tsv` listing status, driving score, success, terminal reason and, for
the 15 takeover cells, whether the audit is valid. A takeover cell counts as
done only when it is scored and its audit shows the required track replaced.

Protocol: `navsafe` traffic, LQR tracking, 600 scored frames (60 s) after the
8 warm-up frames each recipe freezes, contact termination, seed 0
(`NAVSAFE_EVAL_SEED`), and visualization enabled with the left, right and rear
cameras. Visualization renders every simulation step, so disabling it
(`NAVSAFE_ENABLE_VIS=0`) changes the render cadence and is a different protocol.
