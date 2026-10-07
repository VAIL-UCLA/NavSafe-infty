# Data layout

The dataset [c13752hz/NavSafe](https://huggingface.co/datasets/c13752hz/NavSafe) is used as downloaded. Point `NAVSAFE_DATA_ROOT` at the root of your local copy and keep its directory tree unchanged.

```text
<NAVSAFE_DATA_ROOT>/
├── full_test/<scenario-token>/    280 scenario bundles
│   ├── manifest.json              scenario metadata, reconstruction windows
│   ├── arrow/                     ego, traffic, map and route (py123d Arrow)
│   ├── offsets/                   coordinate offset of each reconstruction window
│   ├── <token>s1.usdz … s4.usdz   four 5 s NuRec reconstructions
│   └── ah_assets/                 harvested assets of this scenario's own actors
├── asset/                         actor assets inserted by recipes
├── gait_bank/                     posed copies of walking actors
├── model_zoo/<model-directory>/   policy weights
└── proxy/                         optional authoring subset
```

## Scenarios

The benchmark has 280 scenarios: 28 event types with 10 scenarios each. Every scenario is one bundle, named by its token. A bundle covers 20 s of driving as four consecutive 5 s reconstructions; the renderer switches between them as the episode advances, using the windows and offsets listed in `manifest.json`.

Each bundle is about 8 GB, almost all of it the four `.usdz` files.

## Event types in files

The paper's *event type* appears in stored files under an older name, *leaf*. The files keep that key so that existing recipes, whose contents are checksummed, and existing results stay valid:

| File | Key | Meaning |
| :--- | :--- | :--- |
| Recipe (`*.yaml`) | `leaf` | Event-type id of the scenario, for example `R-4`. |
| Bundle `manifest.json` | `scenario_meta.taxonomy_leaves`, `taxonomy_leaf_names` | Event-type ids and names of the scenario. |
| `navsafe_metrics.json` | `scenario.taxonomy_leaves`, `taxonomy_leaf_names` | The same, copied into each result. |

The command line and the documentation say *event type* throughout; `--leaf` is accepted as an alias of `--event-type`.

## Recipes

A recipe describes the actors a scenario adds, moves or removes and how they behave. Recipes live in the repository, not in the dataset:

| Directory | Recipes | Used for |
| :--- | ---: | :--- |
| [`recipes/benchmark`](../navsafe/benchmark/recipes/benchmark/README.md) | 90 | The benchmark. 62 recipes edit their scenario; the other 28 record a scenario whose event is already in the log. The remaining 190 scenarios have no recipe and run as logged. |
| [`recipes/proxy_set_state_perturbation`](../navsafe/benchmark/recipes/proxy_set_state_perturbation/README.md) | 112 | The state-perturbation set: 56 controlled events, each with an event and a baseline recipe. |

Recipes reference assets and gait banks by path relative to the dataset root (`asset/<file>.ply`, `gait_bank/<name>/`) and pin them by content hash. A recipe fails to load if the file it names has different contents, so keep the downloaded `.ply` files and `bank.json` manifests unmodified.

## Path overrides

Each resource can be placed elsewhere with its own variable. Unset variables resolve below `NAVSAFE_DATA_ROOT`.

| Variable | Default | Contents |
| :--- | :--- | :--- |
| `NAVSAFE_BUNDLES` | `<root>/full_test` | scenario bundles |
| `NAVSAFE_ASSET_BANK` | `<root>/asset` | actor assets |
| `NAVSAFE_GAIT_BANK` | `<root>/gait_bank` | gait banks |
| `NAVSAFE_MODEL_ZOO` | `<root>/model_zoo` | policy weights |

The renderer opens bundles, assets and gait banks itself, so it must see them at the same absolute paths as the evaluator.
