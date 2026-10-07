# NuRec reconstruction utilities

For the supported reconstruction workflow, environment variables, and command arguments, see [Reconstruct NavSim / nuPlan clips](../../docs/reconstruct_navsim_nuplan.md).

The package provides source ingestion, NCore conversion, NuRec fitting/export utilities. Reconstruction fitting runs in NVIDIA containers; evaluation consumes a py123d Arrow scene alongside its NuRec USDZ reconstruction.

Use the NuRec gRPC renderer for evaluation. See [Run evaluation](../../docs/navsafe_eval.md) for renderer setup and policy evaluation, and [Public data layout](../../docs/public-data-layout.md) for downloaded bundle placement.
