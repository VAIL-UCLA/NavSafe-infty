# Edited scenarios

An edited scenario is a bundle plus a recipe: a frozen file that lists the actors to insert or remove and the controllers that drive them. The [benchmark recipes](../navsafe/benchmark/recipes/benchmark/README.md) are named `<event-type-id>.<scenario-token>.yaml`.

Set up the data and renderer as in the [evaluation guide](navsafe_eval.md). In addition:

- Download `asset/` and `gait_bank/`; recipes reference them relative to `NAVSAFE_DATA_ROOT` and verify their hashes.
- The renderer must read those directories at the same absolute paths as the evaluator. With the Docker wrapper, set `NAVSAFE_ASSET_MOUNT="$NAVSAFE_DATA_ROOT"`.

## Run

With the Docker wrapper, pass the recipe after the checkpoint:

```bash
navsafe bundle-eval "$TOKEN" policy "<model-type>" "<checkpoint-path>" \
  --recipe-dir navsafe/benchmark/recipes/benchmark
```

With an [existing renderer](navsafe_eval.md#existing-renderer), add the same option to `navsafe eval`:

```bash
navsafe eval \
  --py123d-data-root "$BUNDLE/arrow" --render-backend nurec_grpc \
  --model-type "<model-type>" --checkpoint "<checkpoint-path>" \
  --recipe-dir navsafe/benchmark/recipes/benchmark \
  --terminate-on-collision --eval-seed "<seed>" \
  --output-dir "<unique-output-directory>"
```

| Option | Meaning |
| :--- | :--- |
| `--recipe-dir <dir>` | Select the recipe for this bundle's token from a directory. If the recipe inserts actors, the scenario runs edited and traffic switches to `navsafe`; otherwise it runs as logged. |
| `--recipe <file>` | Apply one recipe explicitly. Use it with `--traffic-mode navsafe`, or the inserted actors stay at their spawn poses. Takes precedence over `--recipe-dir`. |
| `--recipe-variant` | `e_plus` (default) applies the recipe. `e_zero` runs its counterfactual, without the recipe's actors. |

A recipe sets the scenario's warm-up length itself (`ego.replay_frames`), which overrides `--ego-replay-frames`. The recipe is refused if its checksums do not match, if an asset on disk differs from the pinned one, or if the bundle's frame count or rate differs from the one the recipe was built for.

Each bundle's `ah_assets/` directory is separate from the shared `asset/` library: it holds harvested assets of that scenario's own logged actors, used by [`--asset-harvester-replace`](navsafe_harvested_actors.md).

Use the [recipe editor](../navsafe/benchmark/editor/README.md) to inspect or author recipes.
