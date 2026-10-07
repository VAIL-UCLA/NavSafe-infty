# Scenario bundles and NuRec handoff

NavSafe evaluation uses py123d Arrow for ego states, traffic, maps, and routes, and a NuRec gRPC server for camera rendering. The published bundle layout is described in [Public data layout](../../../docs/public-data-layout.md); evaluation commands are in [Run evaluation](../../../docs/navsafe_eval.md).

## Bundle contents

A bundle contains its Arrow scene, NuRec USDZ reconstruction files, a `manifest.json`, and coordinate offsets for the reconstruction segments. The manifest's `subclips` entries identify each server scene and its timestamp interval. Use those entries rather than assuming a fixed number or duration of segments.

The USDZ files contain the reconstruction data consumed by the NuRec server. Consecutive segments are separate reconstructions: the client switches the active server scene as simulation time advances rather than merging them into one reconstruction.

## Coordinate alignment and handoff

The symbolic Arrow scene and each reconstructed segment must agree on time and coordinates. Bundle construction records each segment's coordinate offset and timestamp interval. During evaluation, the renderer uses the active segment's offset to transform camera and actor poses into reconstruction coordinates.

```bash
export NUREC_GRPC_HANDOFF="$(navsafe bundle --handoff "<bundle-directory>")" # Required for multi-segment rendering: scene IDs, offsets, and timestamp intervals read from the bundle manifest.
```

- `--handoff <bundle-directory>` — **required for this command**; reads an existing bundle and prints the handoff configuration without rebuilding it. The directory must contain `manifest.json` and its referenced offsets.

The server must have the corresponding reconstruction scenes loaded. For newly reconstructed clips, follow [Reconstruct NavSim / nuPlan clips](../../../docs/reconstruct_navsim_nuplan.md).

## Recipes and scoring

A recipe defines the scenario edits, including inserted actors and their behavior. Asset and gait-bank paths resolve through the configured dataset roots. The base reconstruction alone does not establish whether an evaluated episode contains inserted actors; use the applied recipe and the resulting episode metadata.

Evaluation writes `navsafe_metrics.json`. Aggregate only scored episodes and retain excluded outcomes separately. See the output-validation command in [Run evaluation](../../../docs/navsafe_eval.md#validate-completed-outputs) to check completion, metrics, and optional visualization files.
