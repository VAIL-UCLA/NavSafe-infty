# Public data layout validation

The public release preserves the c13752hz/NavSafe Hugging Face directory tree.
Set NAVSAFE_DATA_ROOT to that snapshot root. Per-resource environment overrides
remain available; no original authoring mount is required.

Validation for the cleanup:
- 202 frozen recipes load and verify.
- Scene parameters, actor policies, transforms and asset content hashes are
  unchanged; only resource paths, actor digests and authoring-path metadata change.
- 253 recipe asset references match the public HF file SHA-256 values.
- Referenced gait manifest hashes match the downloaded HF manifests.
- A real horse asset and gait manifest were relocated into a temporary dataset
  root and passed replay resource checks.
- 663 focused CPU tests and 17 subtests passed.
- Kubernetes generators use explicit user PVC, namespace, node and toleration
  settings. Generated manifest behavior is covered by CPU tests.
- Download tests preserve full_test/<token> and existing user files.

Assets and weights stay outside Git.
