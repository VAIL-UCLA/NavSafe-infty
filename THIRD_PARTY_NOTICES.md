# Third-party notices

Source file copyright and SPDX notices are preserved. The root Apache-2.0 license applies to the NavSafe/NexusSim project code; it does not override separately licensed components.

- `pysocialforce/`: vendored PySocialForce by Yuxiang Gao and contributors (MIT). See `third_party/PySocialForce-LICENSE`. Its logger is adapted to avoid configuring the application root logger or creating a working-directory file at import time.
- `nre/`: NVIDIA NRE generated protocol bindings copied from the source checkout; retain their generated headers. NRE service/container remains an external dependency under NVIDIA terms.
- `navsafe/modelzoo/`: upstream model implementations retained from NexusSim, including NAVSIM-derived policies, DrivoR, DiffusionDrive, SparseDrive and related components. Original per-file notices remain controlling. Model weights are not included; follow the upstream model's terms.
- `navsafe/policy/sensor/gtrs_dense.py`: project adapter recovered from the existing SimScale evaluation job; the upstream SimScale model code and checkpoints are external dependencies.

This extraction does not grant additional rights to datasets, checkpoints, NVIDIA images, harvested assets or external model repositories.
