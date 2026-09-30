# NavSafe: harvested actors

Why a reconstructed car falls apart when the policy drives differently, and how
to replace it with one that does not — building the assets, and switching them
on for an eval.

Execute NavSafe commands from the installed environment. For asset generation, configure the source corpus and the external Asset Harvester tool:

```bash
export NAVSAFE_CORPUS="<absolute-source-corpus>" # Required for harvesting: per-scene NCore/reconstruction corpus, not just downloaded Arrow bundles.
export NAVSAFE_WORK="<absolute-work-directory>" # Optional: scratch/index directory; defaults to the configured user cache path.
export NAVSAFE_AH_HOME="<absolute-asset-harvester-directory>" # Required for your external tool installation: Asset Harvester root.
```

For renderer verification, also configure `NUREC_GRPC_HOST` and `NUREC_GRPC_PORT` as described in the [evaluation guide](navsafe_eval.md#existing-renderer).

---

## 1. The problem this fixes

A NavSafe scenario is 20 s of a real nuPlan log reconstructed as **four 5 s
NuRec models**; the render server hands off between them as the ego drives.
Every model is fit to the camera views the **logged** ego actually had during
its own five seconds. A reconstructed actor is therefore not a car — it is a
cloud of gaussians that happens to look like one *from the directions the log
looked at it*.

Closed-loop evaluation breaks that assumption by construction. The policy brakes
earlier, takes a wider line, arrives a second late; an oncoming car the
reconstruction only ever saw head-on at 60 m is now 9 m away and 30° off axis,
and there is no training view anywhere near that geometry. What renders is a
smear.

It is worst in two places, and they compound:

- **Oncoming traffic**, which the log observed over the shortest span of angles
  and which closes fastest when the ego's speed differs.
- **Across a window boundary**, where the handoff swaps in a 5 s model that
  never saw that car from anywhere near this pose. Neither switching on time nor
  on position avoids it: drive slower and the position-based handoff arrives
  with the other traffic in the wrong place; drive faster and the time-based one
  does.

Reproduced, without any of the eval stack, on `00185dab0ba153b9s2` — track
`0b757de6a6bb5a69` (a 4.68 × 1.92 × 1.79 m car) moved to 9.5 m in front of the
camera and turned to face it:

    baked_moved.jpg   the car is a translucent wireframe blob

## 2. The fix

Stop rendering those actors from the scene reconstruction.

[NVIDIA Asset Harvester](https://github.com/NVIDIA/asset-harvester) (Apache-2.0)
lifts one actor's sparse observations into a **view-consistent** 3D gaussian
asset: SparseViewDiT synthesises 16 coherent views of the object from the crops
the log happens to contain, and TokenGS lifts those into a `gaussians.ply`. The
NRE render server can then be told to render a logged track from that asset
instead of its baked gaussians, and the actor looks right from any angle,
because its geometry no longer depends on where the logged ego drove.

**It swaps appearance only.** The track keeps its id, its box, and its per-frame
pose from sim state through the renderer's authoritative mirror. The metric geometry is unchanged, but different images can change a learned policy's decisions and thus its final scores. That is the difference from `navsafe-eval --replace-agent-ids`,
which *deletes* a logged actor and inserts a different one — an authoring edit
that changes what the scenario is.

### No USDZ is repackaged

NVIDIA's documented route bakes harvested assets into the artifact first
(`export-external-assets` → a repackaged USDZ → an `edit_assets.json`). At
NavSafe scale that is a rewrite of a **2.1 GB** `last.usdz` per 5 s window, four
per scenario — several TB for the pool, and a second copy of every
reconstruction to keep in step.

The gRPC path skips all of it. `ReplaceAssetAction.replacement_id` is resolved
by the server as **a path to a 3DGS PLY on its own filesystem**, the same
resolver `DynamicObjectTrack.asset_id` already uses for inserted assets.
Measured 2026-08-27: a nonsense id answers `Failed to edit assets:
PLYGaussianLoader provided path ... not a file`. So the assets sit in the
corpus, the artifacts are never touched, and switching the feature on is one
flag.

The consequence to remember: **the render container must be able to open the
bank.** It does not have to be the same absolute path the harvester used — the
manifest records paths relative to itself, so the eval client resolves them
against wherever the bank actually is — but the directory has to be mounted
into the renderer, and `run_bundle_eval.sh` binds `<bundle>/ah_assets` for that
reason. A bank the eval process can read and the render container cannot is the
one failure mode this arrangement still has, and it surfaces as a fatal
manifest error rather than a silent baseline.

---

## 3. Build a bank

One command per scenario. It needs a GPU (≥16 GB, or `--offload`), the Asset
Harvester install, and the scenario's **NCore clips** — not every corpus
directory still has them, and without them there is nothing to harvest from.

```bash
python -m navsafe.benchmark.harvest harvest "<scene-id>" --max-assets "<asset-count>"
```

- `harvest` — **required subcommand**; generates a bank from this scenario's source observations.
- `<scene-id>` — **required positional**; corpus scene identifier, not a fixed example token.
- `--max-assets` — **optional**, default `10`; caps selected actors, nearest first. Raising it increases harvesting work and renderer memory use.
- `--offload` — **optional**, off by default; reduces GPU memory at a latency cost.
- `--include-parked` — **optional**, off by default; includes stationary actors, spending the asset budget on them as well.
- `--force` — **optional**, off by default; re-parses and regenerates existing assets.

For a stitched `<token>_20s` host the four `<token>s1..s4` windows are found and
parsed automatically. Output lands in the scenario's own directory,
`cfg.ah_assets_dir(scene_id)`:

    <corpus>/<scene_id>/ah_assets/
    ├── lifted/<class>/<track_id>/gaussians.ply   the assets
    ├── lifted/metadata.yaml                      NVIDIA's external-assets format
    ├── replace_manifest.json                     what the renderer reads
    └── harvest.log                               why a track produced nothing

### What it does, and why in that order

| step | | |
|---|---|---|
| **parse** | each window's NCore clip → per-track 512×512 multi-view crops + masks | cheap; also the only source of the distances the next step ranks on |
| **motion** | each window's cuboids → per-track net displacement | reads cuboids, not pixels: 13 s for four windows |
| **select** | the nearest N tracks that MOVED, one entry per track | before lifting, because lifting is the only per-asset cost |
| **lift** | 16-view diffusion + TokenGS → `gaussians.ply` | the expensive step |
| **orient** | rotate 90° about Y into NuRec's convention | skipping it puts every replaced car sideways |
| **gate** | drop assets whose lifted shape disagrees with the clip's cuboid | the server scales an asset onto the track's box, so a wrong aspect renders a *stretched* car, not a small one |
| **describe** | `metadata.yaml` | not read by our renderer; keeps the directory usable by the documented offline route |

The shape gate is worth its own line because the failure it catches is worse
than the one the whole feature fixes. Diffusion can lift a bad object — a mask
that caught the neighbouring car, a track only ever seen from one angle — and
the result is not a small or dim car, because the render server rescales
whatever it is handed onto the track's AABB. It is a car squashed onto the right
footprint.

Over the first twelve harvested scenarios it rejected **7 of ~153** lifted
assets, at 39 %, 62 %, 73 %, 94 %, 99 %, 126 % and **193 %** off their cuboid —
so roughly one in twenty would have rendered visibly deformed. Rejects are left
on disk (so they can be looked at) and left out of the manifest;
`--max-aspect-error` moves the threshold.

The same measurement confirms most of the orientation offline, without a render:
in all 28 assets the longest PLY axis is X (the car's length), the shortest is
the one matching the cuboid's height, and the origin sits within 0.1 of the
centroid in unit scale.

Which *end* of X is the front cannot be measured that way, and a 180° error
there renders every replaced actor driving backwards while passing every
numeric check above. Rasterising an oriented PLY offline (`gsplat`, in the Asset
Harvester env) puts the car upright with **+Y up** — matching the y-up file
convention the render server assumes, not the "top toward −Y" the NVIDIA page
states — and its front toward **+X**, where that page says −X.

**Settled by render, 2026-08-27: the upstream 90° is correct as it stands.**
A replaced actor placed oncoming 7 m in front of the camera comes back facing
the camera, upright and on the road (§6). So the page describes a frame other
than the file's, and no correction is needed. The knob stays because the
question will recur on a new corpus or a new NRE vintage, and a wrong answer
must not cost a re-harvest:

```bash
python -m navsafe.benchmark.harvest reorient "<scene-id>" --degrees "<rotation-degrees>"
```

- `reorient` — **required subcommand**; rotates an existing bank in place without diffusion generation.
- `<scene-id>` — **required positional**; bank's corpus scene identifier.
- `--degrees` — **required**; Y-axis rotation in degrees. This changes existing asset orientation; use the rotation verified for your bank, not a fixed correction for every asset.

spins an existing bank in place; `harvest --orient-degrees` sets it for a new
one. Re-lifting to change a rotation would be ~20 min of diffusion per scenario
to fix a matrix multiply.

**Only actors that actually drove are harvested.** This is the rule that decides
whether the feature helps or hurts, and it is not obvious. A *parked* car is one
the ego drives past, so the reconstruction saw it across a wide arc and holds
real pixels for every angle a policy is likely to want -- a lifted asset can only
be worse there, and measured on `2b7bf25209dd5705` it is: a parked FedEx truck
whose branding is legible in the reconstruction comes back as a plain grey box
truck. A *moving* car travels with or against the ego, so the relative geometry
barely changes across the clip and the reconstruction has almost no angular
baseline on it. That is the actor that falls apart when a policy arrives early,
late or wide.

It also makes the budget fit. One 5 s window held 57 tracks of which **5** had
driven anywhere; over four windows `2b7bf25209dd5705` yields 8 movers and
`0bcae698fd905226` yields 16, against 26 and 23 under the old nearest-N rule.

Motion is read from the clip's own cuboids (`harvest/motion.py`) as **net
displacement, first observation to last** -- not path length. Cuboid annotations
jitter, and summing per-frame steps turns that jitter into metres: measured on
one window, a parked car accumulated 3.8 m of path against 0.7 m of
displacement, while a car driving through had 41.6 m against 41.4 m. The two
populations are far apart, so the 2 m threshold (`--min-motion-m`) is not
delicate. `--include-parked` turns the filter off, which is only useful for
diagnosing.

### "All the dynamic vehicles" is bounded by what can be cropped

Worth being precise about, because the count in a manifest is much smaller than
the number of cars that drove through a scenario. Measured on
`0093ff0188ea5b90_20s`:

| | |
|---|---|
| tracks that moved more than 2 m | 189 |
| of those, ones Asset Harvester's parser could crop at all | **27** |
| — not a vehicle (pedestrians, cyclists), excluded by class | 17 |
| — only one usable view, excluded | 1 |
| — **harvested** | **9** |
| never cropped | **162** |

The dominant filter is upstream, not ours: the parser drops a track that never
projects into a camera acceptably — too distant, too occluded, or below
`crop_min_area_ratio` (0.2 % of the frame by default). Our own `min_views`
threshold cost exactly one track out of 27.

That is the right place for the loss to fall. A car 80 m up the road is a
handful of pixels whether it is replaced or not, and nearest-first ranking would
deprioritise it anyway. But it means a bank covers **the moving vehicles the log
saw well enough to lift**, not every moving vehicle, and a manifest of 3 assets
on a busy street is normal rather than a bug.

Among the movers, **selection ranks by closest approach to the ego** and caps at
10 by default (`--max-assets`). Both halves matter:

- *Why nearest.* It is the same quantity that decides whether a mis-rendered
  actor is visible at all — a car 80 m up the road is a handful of pixels
  whether it is smeared or not — and it is the most correlated with the failure:
  a policy that drives differently changes the geometry most for the actors it
  passes closest.
- *Why capped.* Every replaced actor's PLY is resident in the render server
  beside four 5 s reconstructions on one 24 GB card, and that working set is
  already tight enough that `--cache-size` is pinned to exactly 4 with no spare.

Distances come from Asset Harvester's own parse output (`cam_dists` in each
sample's `input_views/camera.json`) rather than a second read of the log, so
selection and harvesting agree by construction: a track with no usable views is
not rankable and is not selectable, instead of being selected and then silently
failing to lift.

A car that crosses a window boundary is **one** track with one id in all four
clips. It is harvested once, from the window that saw it closest — four assets
for one car would cost four diffusion passes and then disagree with each other
across the handoff.

Deformables are excluded: a rigid asset cannot walk, so a harvested pedestrian
freezes mid-stride, which is a worse artefact than the smear it replaces. (That
is what the gait banks in `navsafe/editing/assets/animate.py` exist for.)

### What it costs

Measured on `2b7bf25209dd5705_20s`, one RTX 3090, 2026-08-27:

| | |
|---|---|
| parse, all four windows | ~2 min (37–57 tracks each) |
| tracks parsed → selected | 54 vehicle samples → **28** unique tracks with ≥2 views |
| lift | **1152 s for 28 assets** = ~41 s each |
| per asset | ~100 k gaussians, 5.5 MB PLY |
| per scenario | **185 MB** |

So roughly **20 min and 190 MB of GPU-side output per scenario**, and about
**80 GB** for a 500-scenario pool. The parse tree is scratch and is deleted
unless `--keep-parse` is passed.

**Yield varies enormously between scenarios, and that is the scenario, not a
bug.** Over the first eleven harvested: 26, 29, 29, 23, 16, 16, 14, 11, 6, 1, 1
assets. A quiet street simply has few actors that were ever seen from two usable
angles — `0ebb578555b25ab2` parsed 19 track-window pairs of which 14 projected
into no camera acceptably at all, leaving one harvestable vehicle. A scenario
with one asset is not worth much of a replace run, and `status` is what shows
that before an eval is planned around it.

### Resuming

`harvest` is resumable at the step boundary — a re-run skips any window already
parsed and any track whose `gaussians.ply` is already there. That matters
because the cluster caps a GPU pod's lifetime at ~6 h and a bank can outlast one
lease. `--force` re-does the work anyway.

### The whole corpus

`scenarios` lists the stitched hosts that *can* be harvested and do not have a
bank yet, and says why the rest cannot. Two things gate it and both are
otherwise silent: a scenario needs its **NCore clips** (some corpus directories
no longer have them — `00c1e4eb4a045f20` is one — and without them there is
nothing to harvest from) and it needs the reconstruction those clips trained.

```bash
python -m navsafe.benchmark.harvest scenarios > "<scene-list.txt>"
```

- `scenarios` — **required subcommand**; lists harvestable scenes without completed banks.
- `--all` — **optional**, off by default; also lists scenes with existing banks.
- `> "<scene-list.txt>"` — **shell redirection**, not a CLI option; writes the list and replaces an existing file at that path.

```bash
python -m navsafe.benchmark.harvest batch --scenes-file "<scene-list.txt>" \
    --workers "<worker-count>" --image "<harvester-container-image>" --out "<harvest-manifest.yaml>"
```

- `batch` — **required subcommand**; generates a Kubernetes Job without submitting it.
- `--scenes-file` — **required for this form**; newline-separated scene IDs. Positional `scenes` are the alternative.
- `--workers` — **optional**, default `4`; concurrent workers/GPUs for the batch. Raising it requests more resources.
- `--image` — **explicitly supplied here**; container image with the required tooling. Choose an image available to your cluster.
- `--out` — **optional**, default `-` (stdout); generated YAML destination.
- `--max-assets` — **optional**, default `10`; per-scene asset cap.
- `--nodes` — **optional**, defaults to `NAVSAFE_NODES` or unrestricted placement; limits allowed hostnames.

Configure mounts, namespace and scheduling using the [Kubernetes guide](../deploy/nautilus/README.md), then inspect and submit the manifest:

```bash
kubectl apply -n "<namespace>" -f "<harvest-manifest.yaml>"
```

One indexed Job, `--workers` GPUs wide, worker *N* taking every *N*th scenario
so a slow one does not leave a worker idle at the end. `backoffLimit: 0` is
deliberate: every stage is resumable, so the answer to a pre-empted worker is to
resubmit the Job and let it skip what is done, rather than to have Kubernetes
retry a container whose failure might be a real one.

It has to be a Job and not a pod per scenario for a second reason —
Nautilus caps controller-less pods at 16 cores / 32 GB, and a Job is not
controller-less.

### Checking a bank against the renderer

The manifest names logged track ids; the reconstruction names its own. If the
two ever stop matching, the whole mechanism is a silent no-op.

```bash
python -m navsafe.benchmark.harvest verify "<scene-id>"
```

- `verify` — **required subcommand**; validates the bank against the connected renderer. A live service serving this scene is required.
- `<scene-id>` — **required positional**; scene whose bank and render tracks should match.

```bash
python -m navsafe.benchmark.harvest status
```

- `status` — **required subcommand**; lists existing banks without generating assets.
- `<scene-id>` — **optional positional**, omitted here; restricts the report to one scene.

`verify` reports, per 5 s window, how many of its served tracks the bank can
replace. Some manifest tracks matching no window is normal — a bank covers 20 s,
a window is 5 s. **None** matching anywhere is fatal and means the bank belongs
to a different reconstruction.

---

### Harvesting needs the clips, not the Arrow

A scenario becomes harvestable as soon as its four **NCore clips** exist. It does
NOT need the Arrow conversion — that is what makes a scenario *evaluatable*, and
it comes later and separately.

The distinction matters because the two live in different places. The Arrow
conversion creates `<token>_20s/`, so enumerating scenarios by that directory
reported **87 of 407** harvestable and silently hid 320 that were ready.
`scenarios` therefore enumerates by token, from the `<token>s1..s4`
reconstruction windows themselves.

The bank still belongs at `<token>_20s/ah_assets`, created if it does not exist:
that is the directory a later Arrow conversion lands in, so the bank is already
in the right place when the scenario becomes evaluatable, and
`--asset-harvester-replace` finds it with no configuration.

Measured on the pool: 414 tokens, all 414 reconstructed, 407 with all four
clips, 7 without.

### Publishing a bank

A bank is already laid out the way it should be published -- inside the scenario
it belongs to, which is exactly where `--asset-harvester-replace` looks for it
with no configuration. So there is no staging copy; `pack` validates and writes
down what goes where:

```bash
python -m navsafe.benchmark.harvest pack --repo "<dataset-repository>" --prefix "<dataset-prefix>" --strict
```

- `pack` — **required subcommand**; validates banks and writes an upload plan; it does not upload automatically.
- `--repo` — **optional**, default `c13752hz/NavSafe`; destination repository named in the plan.
- `--prefix` — **optional**, default `full_test`; path inside that repository.
- `--strict` — **optional**, off by default; returns a failure if any bank fails validation.
- `--dest` — **optional**, default a dated run directory; output location for the index and plan.

into a dated run directory: `index.json` (every bank, its assets, their
provenance and sizes), `README.md` (the layout, for the dataset card),
`upload.sh` (one `hf upload` per scenario) and `broken.txt` if any bank failed
to validate.

**`replace_manifest.json` records paths relative to itself** (schema 2), which
is what makes a bank portable: harvested under `/data/...`, unpacked into
someone else's bundle, it resolves either way with nothing to rebase. Schema 1
banks stored absolute paths and are still readable in place; re-running `harvest`
rewrites them.

Roughly 82 % of the bytes are the PLYs. The rest -- the 16 synthesised views,
the real crops they were conditioned on, and the two preview MP4s -- is the
provenance for each asset and is worth publishing; `upload.sh` documents the
`--exclude` patterns that drop it if not.

---

## 4. Evaluate with it

One flag, on the eval script from
[`docs/navsafe_eval.md`](navsafe_eval.md):

`--asset-harvester-replace [MANIFEST_JSON]` is an optional evaluator argument, not a standalone command. It is off by default. Without a value it discovers the bank beside the Arrow root; with a value it uses that replacement manifest. It changes actor appearance and rendering memory use, and can therefore change a vision policy's decisions.

With no value it finds the bank beside the scenario
(`<py123d-data-root>/../ah_assets/replace_manifest.json`), which is where both a
corpus scenario and a published bundle keep it. It requires
`--render-backend nurec_grpc`: the swap is done by the render server, and no
other backend has one.

Through the wrappers:

```bash
bash navsafe/benchmark/eval/run_bundle_eval.sh "<scenario-token>" policy "<model-type>" "<checkpoint-path>" \
    --asset-harvester-replace
```

- Script path — **required positional**; Docker renderer/evaluation wrapper. First configure the [wrapper environment](navsafe_eval.md#local-docker-wrapper).
- `<scenario-token>` — **required positional**; downloaded bundle to evaluate.
- `policy` — **optional positional**, default `policy`; enables closed-loop policy evaluation.
- `<model-type>` — **explicitly supplied here**, wrapper default `drivor`; selected adapter.
- `<checkpoint-path>` — **required unless `NAVSAFE_CHECKPOINT` is set**; adapter weights.
- `--asset-harvester-replace` — **optional**, enabled here; auto-discovers the bundle's `ah_assets/replace_manifest.json`. An explicit manifest path can follow the flag.

```bash
python -m navsafe.benchmark.world.run_eval --seed "<seed-json-or-directory>" --run "<campaign-slug>" \
    --model-type "<model-type>" --checkpoint "<checkpoint-path>" \
    --execution-mode controller --controller lqr --asset-harvester-replace
```

- `--seed` — **required**; seed JSON or containing directory with scenario metadata and paths.
- `--run` — **required for an actual run**; groups outputs under a named campaign.
- `--model-type` — **explicitly supplied here**, default `drivor`; selected adapter.
- `--checkpoint` — **required here**; overrides the configured default checkpoint to match your adapter.
- `--execution-mode` — **explicitly set to `controller` here**; this wrapper otherwise defaults to `teleport`, which changes execution and comfort semantics.
- `--controller` — **explicitly set to `lqr` here**; wrapper default `pure_pursuit`.
- `--asset-harvester-replace` — **optional**, off by default; enables bank replacement with auto-discovery when no manifest value follows.
- `--grpc-host` — **optional**, default `NUREC_GRPC_HOST` or `localhost`; renderer service hostname.

`run_bundle_eval.sh` mounts `<bundle>/ah_assets` into the renderer at its own
host path automatically — the bundle itself is mounted at `/workdir/bundle`, so
without that the manifest's absolute paths would not resolve inside the
container even though the files are right there. A hand-started server needs the
same: bind the bank at an identical path inside and out.

### What it does at run time

At setup the renderer already asks each served scene for its controllable tracks
(`get_dynamic_objects`). The replace intersects that with the manifest and sends
one `edit_assets` per scene, with the AABB the **server** reported for each
track. `close()` undoes it through the same `restore_model_parameters` the
insert path uses, so a swap cannot leak onto the next eval sharing a warm
server.

**The AABB is the track's own box, with nothing added.** That is not obvious:
NRE scales an *inserted* asset by `(object_size − dims_offset).max()`, which is
why `_grpc_object_size` pads the box on the insert path (a unit-normalised PLY
in a box asking for the real size would otherwise render at
`length − 1 m`). Replace does **not** share that convention — rendered both ways
on 2026-08-27 against a 4.58 × 1.84 × 1.52 m car, the plain box came back the
right size and the padded one visibly overflowed the lane. So the server's own
box goes through verbatim.

### What it costs where the reconstruction is already good

Replacement is not an upgrade. A harvested asset is lifted from a handful of
crops by a diffusion model; a reconstructed actor is fit to the real pixels. So
wherever the reconstruction HAS coverage, it wins, and the swap loses detail.

Rendered both ways over a whole 20 s scenario, log replay with a 2 m lateral
camera offset, 12 actors replaced, 200/200 frames each (2026-08-28):

* a FedEx truck whose **branding is legible** in the reconstruction comes back as
  a plain grey box truck;
* a parked Jeep stays recognisably the same Jeep, but smoother, with a visible
  seam;
* most frames are indistinguishable.

A 2 m offset is not enough to break a baked actor, so **that comparison shows
the cost and not the benefit**. The benefit appears only where the
reconstruction has no coverage at all: relocate one actor to 7 m oncoming — a
pose the logged ego never occupied — and the reconstruction renders *nothing*
there, while the harvested asset renders a car (§6).

Which is the whole shape of the feature: it does not make actors look better, it
makes them renderable where they otherwise are not. That is why it is a per-eval
flag rather than a default, and why the VRAM budget below is spent
nearest-first.

### Renderer VRAM: how many actors can be replaced

Every replaced actor's PLY is resident in the render server **once per scene
that holds that track**, on top of a working set that is already tight — four
5 s reconstructions must stay resident for a 20 s scenario (`--cache-size 4`,
tuned with no spare) plus the harmonizer. A 26-asset scenario therefore puts
**75** asset instances on the card, not 26.

Measured on one 24 GB card, one scenario's four windows (2026-08-28):

| | VRAM | free |
|---|---|---|
| idle (CUDA context + harmonizer) | 9.8 GiB | |
| four reconstructions resident | 18.4 GiB | 6.2 GiB |
| + 8 nearest replaced (29 instances) | 21.4 GiB | 3.1 GiB |
| + 16 nearest (54) | 23.5 GiB | 1.0 GiB |
| + all 26 (75) | 24.1 GiB | 0.5 GiB |

So roughly **76 MiB per replaced instance**, ≈220 MiB per manifest asset at four
windows, against a ~6 GiB budget.

**The render server must run with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (both serve manifests set
it). Without it the ceiling is **11** replaced tracks and the 12th kills the
episode at step 0 — and the failure is fragmentation, not exhaustion: the
allocation that died wanted 1.29 GiB while 1.17 GiB was free and 1.15 GiB sat
reserved-but-unallocated. Splitting the work across several smaller
`edit_assets` calls does **not** help; the setting does.

A 20-asset working point leaves ~1.6 GiB for the per-render transients (a single
camera render allocates over 1 GiB), which is the margin worth keeping on a
24 GB card. Above that, use a bigger card.

**It fails the run rather than degrading.** An eval that quietly rendered the
original smeared actors after being asked to replace them would be
indistinguishable, in the metrics, from the run that was wanted.

---

## 5. Installing Asset Harvester

Third-party, with its own conda environment (python 3.10, torch 2.10/cu128,
`gsplat` built from a pinned commit) that cannot share the NexusSim venv. Put it
somewhere persistent — the checkpoints alone are 12 GB — and point
`NAVSAFE_AH_HOME` at it (default `~/tools/asset_harvester`).

```bash
export NAVSAFE_AH_HOME="<absolute-asset-harvester-directory>" # Required here: root for the separately maintained Asset Harvester checkout and its tool environment.
export CONDA_ENVS_DIRS="$NAVSAFE_AH_HOME/conda/envs" # Optional: directs upstream Conda environment creation into this tool directory.
export CONDA_PKGS_DIRS="$NAVSAFE_AH_HOME/conda/pkgs" # Optional: directs upstream Conda package caching into this tool directory.
git clone https://github.com/NVIDIA/asset-harvester "$NAVSAFE_AH_HOME/repo"
cd "$NAVSAFE_AH_HOME/repo"
bash setup.sh
hf download nvidia/asset-harvester --local-dir checkpoints
```

The model card is **gated**: accept it on huggingface.co and export `HF_TOKEN`,
or the download 401s. `python -m navsafe.benchmark.harvest harvest` checks all
four checkpoints and the interpreter before it starts, so a missing piece is
named up front rather than surfacing inside a subprocess ten minutes in.

Two upstream defaults do not fit our clips and are overridden in
`navsafe/harvest/ah.py`:

- **Camera ids.** Upstream assumes the Hyperion rig
  (`camera_front_wide_120fov`, …); our nuPlan-converted clips name their sensors
  `camera_pcam_f0`, `camera_pcam_l0`, … The ids are read out of the clip's own
  manifest instead. A wrong list is not an error upstream — it parses nothing
  and exits 0.
- **Class names.** A parsed sample carries the source dataset's full label
  (`nuplanboxdetectionlabel.vehicle`, `wodperceptionboxdetectionlabel.type_vehicle`),
  so the class filter matches on substrings rather than on equality.

One upstream caveat that applies to the reconstruction rather than to the
harvest: NVIDIA advises **disabling PPISP** when reconstructing a scene that
will take inserted assets, or they read over-saturated against it.

---

## 6. What it looks like

Measured end to end on `2b7bf25209dd5705_20s` (four windows, 26 harvested
assets) on 2026-08-27. One logged car, track `710b22b5891254fd`
(4.58 × 1.84 × 1.52 m), rendered four ways from the same camera pose:

| | |
|---|---|
| **baked, log pose** | the reconstruction draws it correctly — this is the trained view |
| **baked, moved to 7 m oncoming** | the car **disappears entirely**. The reconstruction holds no gaussians that can be drawn from a viewpoint the logged ego never had; the road renders empty |
| **replaced, AABB = the track box** | a complete, correctly oriented car at the same pose — facing the camera, upright, on the road |
| **replaced, AABB + dims_offset** | the same car, visibly oversized — it overflows the lane |

Rows two and three are the whole point: the failure is not "a blurrier car", it
is an actor that cannot be rendered at all from where the policy met it, and the
harvested asset renders it.

Row four is why the AABB convention had to be measured rather than inherited
from the insert path.

The id chain holds end to end — `verify` against the live server matched **26 of
26** manifest tracks, 16–20 of them per 5 s window, which is the expected shape
for a bank that covers 20 s.
