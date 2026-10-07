# Benchmark recipes

The benchmark has 280 scenarios in 28 event types, 10 per type. This directory
holds the recipes for the nine event types whose scenarios are defined by a
recipe: 90 files, one per scenario, named `<event-type-id>.<token>.yaml`.
The other 190 scenarios have no recipe and are evaluated as logged.

A recipe lists the actors a scenario inserts or removes, where each one starts
and the controller that drives it. Of the 90 recipes, 62 edit their scenario.
The other 28 have no actors: the event is already present in the log (a real
oncoming car, real cyclists) or is defined by the road alone, as in the two
wrong-way event types. Those scenarios run unedited.

| Event type | Scenarios | With an edit | As logged |
| :--- | ---: | ---: | ---: |
| C-7 | 10 | 8 | 2 |
| C-10 | 10 | 0 | 10 |
| I-3 | 10 | 10 | 0 |
| R-2 | 10 | 7 | 3 |
| R-3 | 10 | 10 | 0 |
| R-4 | 10 | 10 | 0 |
| V-8 | 10 | 10 | 0 |
| V-10 | 10 | 7 | 3 |
| V-11 | 10 | 0 | 10 |
| **Total** | **90** | **62** | **28** |

## Using a recipe

The evaluator selects the recipe for a scenario from this directory when given
`--recipe-dir`, or takes one explicitly with `--recipe`; see the
[edited-scenario guide](../../../../docs/navsafe_eval_edited_scene.md). The
[full benchmark sweep](../../../../docs/navsafe_eval.md#full-benchmark-sweep)
does this for all 280 scenarios.

Inside a recipe the event-type id is stored under the key `leaf`. Recipes are frozen. Each actor carries a `sha256` over its spawn pose and
controller, and each asset is pinned by the hash of its file, so a recipe that
has been edited by hand, or that finds a different asset on disk, fails to load
instead of evaluating a different scenario. Asset and gait-bank paths are
relative to `NAVSAFE_DATA_ROOT`.

## The scenarios

### C-10

| token | inserted | logged removed |
|---|---|---|
| `06a1f481118057b2` | — | — |
| `07bf0601ad425977` | — | — |
| `17cac31ef9135faf` | — | — |
| `2bd04a0902095129` | — | — |
| `2c337eb368fb54ca` | — | — |
| `2ccebcdb0da25be5` | — | — |
| `2d24100bcb1e57e2` | — | — |
| `3067f3d3d5a75989` | — | — |
| `42a20478abeb54d5` | — | — |
| `4d6456183bd056bc` | — | — |

### C-7

| token | inserted | logged removed |
|---|---|---|
| `05bcef7a11d65c6a` | — | — |
| `0dd3256035e75770` | 1 | — |
| `0ebb578555b25ab2` | 1 | — |
| `112db94505025ec5` | 1 | — |
| `1bdfacbfcff75c27` | 1 | — |
| `1e9350ac2bc25f59` | 1 | — |
| `54ae4b189e79541b` | 1 | — |
| `599c8b2bc9b252a7` | 1 | — |
| `5d12ad55fdd858e1` | 1 | — |
| `6b7c5199f84e5aac` | 1 | — |

### I-3

| token | inserted | logged removed |
|---|---|---|
| `03b66343e1ac5d68` | 2 | 6 |
| `050284885b2059e9` | 2 | 4 |
| `131a036a111e54f3` | 2 | 16 |
| `4fff5f5a86be53d5` | 2 | 7 |
| `76da778ff251508d` | 2 | 8 |
| `84acc78da95f56d3` | 2 | 1 |
| `89e02236312d5038` | 2 | 3 |
| `a43dbb1d34665f24` | 2 | 4 |
| `a6b9a83019915658` | 2 | 9 |
| `b023b7bcbab05bcb` | 2 | 5 |

### R-2

| token | inserted | logged removed |
|---|---|---|
| `0057ce5b81c35a81` | 2 | — |
| `00c1e4eb4a045f20` | 2 | — |
| `02b68b9cc51f506a` | 2 | — |
| `02f1ad081f41550e` | 2 | — |
| `054e4984e1b55ec9` | 2 | — |
| `190bf8dd22a25d1c` | — | — |
| `20cc0fdb7e2d5c3f` | 2 | — |
| `2575048779565f0b` | 2 | — |
| `99a98a7ffb075389` | — | — |
| `ac01c31d1a5a5ed2` | — | — |

### R-3

| token | inserted | logged removed |
|---|---|---|
| `02379e524f105926` | 5 | — |
| `0777f5a7263758be` | 5 | — |
| `0802a51b0a1d512c` | 5 | — |
| `087df0996ade50d3` | 5 | — |
| `0a51eb8adf8e5391` | 5 | — |
| `132307b3c1a55f97` | 5 | — |
| `17b0992157365222` | 5 | — |
| `2462c21ce2bb5f2d` | 5 | — |
| `34489cf42e005e93` | 5 | — |
| `687dc7e79cf65570` | 5 | — |

### R-4

| token | inserted | logged removed |
|---|---|---|
| `005f87dd980253a5` | 1 | — |
| `02015675e4585611` | 1 | — |
| `0371700bf65b51e0` | 1 | — |
| `05f1a5cbc0905d8e` | 1 | — |
| `061e7e1700945b03` | 1 | — |
| `173369dc059d5fe9` | 1 | — |
| `225eb6e22af55972` | 1 | — |
| `25b5cecdb3b75e7c` | 1 | — |
| `2ee44628526e524c` | 1 | — |
| `a2c2e046132e5596` | 1 | — |

### V-10

| token | inserted | logged removed |
|---|---|---|
| `07846b829b3a575e` | 1 | — |
| `13805e7752fe5685` | 1 | — |
| `13c555e68671524f` | 1 | — |
| `14d171fcb9295596` | 1 | — |
| `34ac200e359653b5` | 1 | — |
| `49a0d29c7058501c` | 1 | — |
| `63c145828c3b5fd8` | 1 | — |
| `6c9e40634f705f56` | 1 | — |
| `bbee1ab465af50c2` | 1 | — |
| `be36f75d360c502f` | 1 | — |

### V-11

| token | inserted | logged removed |
|---|---|---|
| `0093ff0188ea5b90` | — | — |
| `0317b218061b5c4d` | — | — |
| `18b5995484435fbe` | — | — |
| `2073f76964735ff7` | — | — |
| `227d62f5dfd95624` | — | — |
| `2b7bf25209dd5705` | — | — |
| `32d85d373126537e` | — | — |
| `a18d2b32f8415373` | — | — |
| `a4581d8af5f755a9` | — | — |
| `bf896d504b4356c7` | — | — |

## Rebuilding a recipe

```bash
navsafe bake --event-type "<event-type-id>" --scenario-id "<scenario-id>" --out "<recipe-output.yaml>"
```

| Option | Meaning |
| :--- | :--- |
| `--event-type` | Event type to build, for example `R-4`. Required. |
| `--scenario-id` | Host scenario: its converted identifier or token. Required. |
| `--out` | Destination file. Defaults to an automatically named recipe path. |
| `--no-insert` | Record a host whose log already contains the event, without inserting actors. |
| `--force` | Skip the check that the host's road geometry suits the event type. The result may not stage the event. |
| `--overwrite` | Replace an existing output file. |

Rebuilding changes the recipe's checksums, so results obtained with the earlier
file are not comparable with results from the new one.
