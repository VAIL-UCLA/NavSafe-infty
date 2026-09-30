# NavSafe eval pipeline

Home for the end-to-end evaluation pipeline: taking a policy and a seed (or a
plain scenario) from "start a renderer" to "a scored, attributable verdict".

## Scenario bundles

**[PIPELINE.md](PIPELINE.md)** — how trained NuRec sub-clips become a
downloadable eval scenario, what the bundle contains,
and where the handoff offsets come from.

| Script | Does |
|---|---|
| `fetch_usdz.py` | pull scenario models from the HF dataset (~8 GB each) |
| `bundle.py` | windows + offsets + manifest + per-bundle README; `--handoff <bundle>` prints the `NUREC_GRPC_HANDOFF` value for an existing one |
| `scenario_taxonomy.py` | nuPlan scenario type → NavSafe taxonomy leaf + inserted-actor flag, folded into each bundle's `manifest.json` |
| `make_arrow.sh` | Arrow scenario source (ego, boxes, map, route) from the nuPlan log |
| `nuplan-navhard.yaml` | the py123d conversion config `make_arrow.sh` installs |
| `stitch_eval.sh`, `stitch_jobs.tsv` | website clip rendering (4×5 s handoff → mp4) |

## The rule this package inherits

> World building is expensive and happens once. Episode generation and scoring
> are cheap and re-runnable. **Scoring is a pure function of a stored trace.**

Anything landing here that opens a simulator to answer a scoring question is in
the wrong layer — `navsafe/scoring/` re-scores a stored trace with no GPU, and
that has to stay true.

## What already exists elsewhere (don't re-implement)

| Concern | Lives in |
|---|---|
| Run one eval on a seed | `navsafe/world/run_eval.py` |
| Stage orchestration across ncore→arrow→aux→train→export→eval | `navsafe/runner/pipeline.py` |
| What exists on disk | `navsafe/runner/status.py` |
| Trace writing (live) | `navsafe/trace/writer.py`, `navsafe/trace/hook.py` |
| Trace rebuilding (post-hoc, non-seed runs) | `navsafe/trace/from_eval.py` |
| Why the episode ended (8 reasons, one per episode) | `navsafe/termination.py` |
| Rubric verdict + evidence | `navsafe/rubric/` |
| Outcome labels | `navsafe/outcome/labeler.py` |
| Episode score, NSS, worst group, paired Δ | `navsafe/scoring/episode.py` |
| Route completion, off-route distance | `navsafe/scoring/route.py` |
| Bench2Drive DS / SR / Efficiency / Comfort | `navsafe/scoring/metrics.py` |
| Aggregation across episodes | `navsafe/runner/collect.py` |
| Score a finished evaluator run (CLI) | `scripts/tools/navsafe_from_eval.py` |

## Conventions for files added here

- **Pure functions over stored artifacts.** No simulator, no renderer, no GPU.
- **Mark, never drop.** A frame or episode that cannot be scored is recorded
  with its reason, not silently omitted — benchmark-attributed endings
  (`envelope_exit`, `infra_failure`) are reported `—` and excluded from
  denominators, never folded in as zeros.
- **No silent caps.** If something truncates, samples, or skips, say so in the
  output.
- **Paths come from `navsafe/config.py`** (environment variables), never baked
  into a module.

## TODO

- [ ] Describe the pipeline entry point once it lands
- [ ] Usage example (one policy, one seed, end to end)
- [ ] Where its outputs go, and which are expensive
