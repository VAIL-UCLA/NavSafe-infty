# Evaluation tools

Scripts for fetching scenario bundles, running a policy on them and rendering clips. The [evaluation guide](../../../docs/navsafe_eval.md) explains how to use them; [PIPELINE.md](PIPELINE.md) describes what a bundle contains and how the renderer moves between its reconstruction windows.

## Running evaluations

| Command | Script | Purpose |
| :--- | :--- | :--- |
| `navsafe bundle-eval` | `run_bundle_eval.sh` | Evaluate a policy on one bundle, or render its ego replay. Starts a Docker renderer or uses an existing one. |
| `navsafe benchmark` | `run_full_benchmark.sh` | Evaluate a policy on all 280 scenarios: one bundle evaluation per scenario, with the recipe chosen per scenario. Resumable and shardable. |
| `navsafe perturbation` | `run_perturbation_set.sh` | Evaluate a policy on the 56 controlled events of [`recipes/proxy_set_state_perturbation`](../recipes/proxy_set_state_perturbation/README.md). |

## Bundles

| Script | Purpose |
| :--- | :--- |
| `fetch_bundle.py` (`navsafe fetch`) | Download whole bundles from the dataset. |
| `fetch_usdz.py` | Download only the reconstruction files of a scenario. |
| `bundle.py` (`navsafe bundle`) | Build a bundle from reconstructions (windows, offsets, manifest); `--handoff <bundle>` prints the `NUREC_GRPC_HANDOFF` value for an existing bundle. |
| `upload_bundle.py` | Upload bundles to a dataset repository. |
| `make_arrow.sh` | Build a scenario's Arrow source (ego, boxes, map, route) from the nuPlan log, using the `nuplan-navhard.yaml` conversion config. |
| `scenario_taxonomy.py` | Map nuPlan scenario types to event types; writes the event type into each bundle's manifest. |

## Clips

| Script | Purpose |
| :--- | :--- |
| `render_gif_batch.sh`, `gif_batch_jobs.py` | Render ego replays of many bundles as multi-camera GIFs, locally or as Kubernetes Jobs. |
| `stitch_eval.sh`, `stitch_jobs.tsv` | Render one scenario's four windows as a single 20 s clip. |

## Scoring

Scoring is separate from simulation: it reads the artifacts an episode wrote and needs no GPU, so a finished run can be scored again.

| Concern | Module |
| :--- | :--- |
| Why an episode ended | `navsafe/benchmark/termination.py` |
| Driving score, success, efficiency, comfort | `navsafe/benchmark/scoring/metrics.py` |
| Route completion and off-route distance | `navsafe/benchmark/scoring/route.py` |
| Scoring a finished run | `navsafe/benchmark/scoring/from_run.py`; `navsafe score` |
| Summarizing a sweep | `navsafe report` |

Episodes that end for a reason outside the policy's control are marked excluded and left out of every mean. See [scoring and termination](../../../docs/navsafe_termination_and_success.md).
