# The NavSafe edited-scenario set

All 90 frozen recipes the benchmark evaluates -- nine taxonomy
leaves, ten scenarios each -- one file per scenario, named
`<LEAF>.<token>.yaml`. This directory **is** the set: a scenario is in the
benchmark because its recipe is here, and the eval reads these files directly
(`navsafe-eval --recipe`).

These files are the benchmark definition. Resources resolve against the downloaded HF dataset via NAVSAFE_DATA_ROOT.

Each actor's `sha256` covers its spawn and policy, and the asset it names is
pinned by content hash, so editing a number here without re-freezing makes the
recipe fail loudly at replay rather than quietly evaluating something else.

`docs/navsafe_eval_edited_scene.md` is the companion: what each scenario needs
on disk (Arrow, reconstruction, asset PLYs, gait banks) and how to run one.

## What is here

| leaf | scenarios | with an edit | log alone |
|---|---|---|---|
| C-10 | 10 | 0 | 10 |
| C-7 | 10 | 8 | 2 |
| I-3 | 10 | 10 | 0 |
| R-2 | 10 | 7 | 3 |
| R-3 | 10 | 10 | 0 |
| R-4 | 10 | 10 | 0 |
| V-8 | 10 | 10 | 0 |
| V-10 | 10 | 7 | 3 |
| V-11 | 10 | 0 | 10 |
| **total** | **90** | **62** | **28** |

V-8 joined on 2026-09-08. It was the ninth leaf and the last one still judged
on the road alone: nuPlan tags the manoeuvre but nothing in the log makes it
illegal, so a policy that turned and one that did not scored the same. Each of
its ten now carries a prohibitory plate at the junction, placed by hand on a
bird's-eye board (`editing/place_board.py`) after five rounds of solving it
against proxies produced positions that rendered cleanly and looked wrong.

A recipe with **no actors** is not an omission. C-10 (wrong-way crash) and
V-11 (driving wrong way) are defined by the ROAD, and a handful of C-7 and R-2
hosts already carry the event in their own log — a real oncoming car, real
cyclists. Inserting into those would put a second one beside it. They are
scored by the same eval on the same 20 s host; there is simply nothing to add.

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

## Rebuilding one

```bash
navsafe bake --leaf "<leaf-id>" --scenario-id "<scenario-id>" --out "<recipe-output.yaml>"
```

- `bake` — **required subcommand**; builds a frozen recipe from a selected leaf and host.
- `--leaf` — **required for this operation**; target benchmark leaf ID.
- `--scenario-id` — **required for this operation**; converted host identifier or unique token returned by mining.
- `--out` — **optional**, default an automatically named recipe path; destination YAML file.
- `--no-insert` — **optional**, off by default; uses an event already present in the host without inserting actors.
- `--force` — **optional**, off by default; bypasses the host geometry gate and can produce an unsuitable scenario.
- `--overwrite` — **optional**, off by default; permits replacing an existing recipe output.


`bake` refuses a host that fails the leaf's geometry gate; `--force` skips that
gate and is a smell rather than a flag — the four leaves defined by the road
(V-8, V-10, V-11, C-7) were once all baked onto one arbitrary clip that way,
producing four recipes that were the same stretch of road wearing four labels.
`--no-insert` is the supported way to say "this host already stages the event".

A re-bake is not free: the recipe's checksums change, so the render under
`/data/runs/20260827-editing90/` no longer shows what the recipe says until
that scenario is re-run.
