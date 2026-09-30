# Evaluate an edited scenario

Complete the [dataset and renderer setup](navsafe_eval.md#existing-renderer). Select a frozen recipe under `navsafe/benchmark/recipes/benchmark/<leaf-id>.<scenario-token>.yaml` that matches the bundle. Recipe hashes bind asset bytes, gait manifests, actor parameters and frame contracts.

```bash
export TOKEN="<scenario-token>" # Required here: token matching the frozen recipe and downloaded bundle.
export BUNDLE="$NAVSAFE_DATA_ROOT/full_test/$TOKEN" # Required here: selected bundle under the HF dataset root.
export NUREC_GRPC_HANDOFF="$(python -m navsafe.benchmark.eval.bundle --handoff "$BUNDLE")" # Required here: selects the bundle's reconstruction handoff specification.
```

- Embedded `python -m navsafe.benchmark.eval.bundle` — bundle utility; **`--handoff <bundle-directory>` is required for this operation** and prints the handoff specification. Recompute it when switching bundles.

```bash
navsafe-eval \
  --scenario-source py123d --py123d-data-root "$BUNDLE/arrow" \
  --py123d-scene-index 0 --render-backend nurec_grpc \
  --model-type "<model-type>" --checkpoint "<checkpoint-path>" \
  --recipe "<recipe-yaml>" --recipe-variant e_plus \
  --traffic-mode navsafe --execution-mode controller --controller lqr \
  --ego-replay-frames "<warmup-frames>" --replan-rate "<replan-interval-frames>" \
  --eval-seed "<seed>" --terminate-on-collision \
  --output-dir "<unique-output-directory>"
```

- `--scenario-source` — **Optional**, default `py123d`; the only supported input format is py123d Arrow.
- `--py123d-data-root` — **Required for this workflow**, CLI default `data`; Arrow root for the selected bundle, not the HF snapshot root.
- `--py123d-scene-index` — **Optional**, default `0`; zero-based scene index within the Arrow root. A single-scenario bundle normally uses `0`.
- `--render-backend` — **Optional**, default `nurec_grpc`; requests images from the connected NuRec service.
- `--model-type` — **Required for this workflow**, CLI default `transfuser`; registered policy adapter name. Choose the adapter that matches the checkpoint.
- `--checkpoint` — **Required**; checkpoint path expected by the selected adapter. For a checkpoint-free adapter, pass its supported sentinel, such as `none` for `pdm_closed`.
- `--traffic-mode` — **Required for the intended protocol**, CLI default `log_replay`; `semi_reactive` lets background vehicles react to ego, while `navsafe` executes recipe actor controllers. This changes traffic behavior and scores.
- `--execution-mode` — **Optional**, default `controller`; controls how plans move ego. `controller` uses a tracker and bicycle model, `physics` uses PhysX, and `teleport` places ego along the plan and excludes meaningful comfort scoring.
- `--controller` — **Optional**, default `lqr`; trajectory tracker for controller/physics execution. Alternatives are `pure_pursuit` and `pid`; changing it changes tracking behavior.
- `--ego-replay-frames` — **Optional**, default `8`; logged-ego warm-up frames before policy takeover. Changing this moves the handoff point and evaluation starting state.
- `--replan-rate` — **Optional**, default `5`; simulation frames between policy calls, **not Hz**. At the default 10 Hz simulation rate, `5` means 2 Hz replanning. Set it for your model.
- `--eval-seed` — **Optional**, default unset; seeds Python, NumPy and Torch for policy sampling. It does not add noise to deterministic traffic or control.
- `--terminate-on-collision` — **Optional CLI flag, enabled for the NavSafe protocol**; ends the episode on any ego-box contact, including not-at-fault contact. Fault attribution still determines the penalty.
- `--recipe` — **Required for an edited scenario**, otherwise optional and unset; frozen recipe YAML matching the bundle token and frame contract. Verifies resource hashes before applying edits; use with `--traffic-mode navsafe`.
- `--recipe-variant` — **Optional**, default `e_plus`; `e_plus` applies the recipe, while `e_zero` removes the actors specified by its counterfactual pair. This changes the evaluated scenario.
- `--output-dir` — **Required**; directory for this run's metrics and artifacts. Use a distinct directory per model, scenario, configuration and seed.

Optional visualization:

- `--enable-vis` — off by default; saves images/GIFs and renders at the 10 Hz simulation rate.
- `--vis-cameras "<camera-list>"` — default unset; adds comma-separated artifact cameras when visualization is enabled.

For the [Docker wrapper](navsafe_eval.md#local-docker-wrapper), forward `--recipe "<recipe-yaml>" --traffic-mode navsafe` after the checkpoint positional argument. Its `NAVSAFE_ASSET_MOUNT` must include the dataset root.

Keep the global `asset/` and `gait_bank/` libraries separate from each bundle's `ah_assets/` actor-replacement bank. Renderer and client must resolve the same absolute paths. Use the [recipe editor](../navsafe/benchmark/editor/README.md) to author or inspect recipes.
