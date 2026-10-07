# Third-party notices

Source file copyright and SPDX notices are preserved. The root Apache-2.0 license applies to the NavSafe project code; it does not override separately licensed components.

- `navsafe/_vendor/pysocialforce/`: PySocialForce by Yuxiang Gao and contributors (MIT). See the `LICENSE` file in that directory. Its logger is adapted to avoid configuring the application root logger or creating a working-directory file at import time.
- `navsafe/_vendor/nurec_grpc/`: generated Python bindings of the NVIDIA NRE gRPC protocol. NRE service/container remains an external dependency under NVIDIA terms.
- `navsafe/modelzoo/`: upstream model implementations, including NAVSIM-derived policies, DrivoR, DiffusionDrive, SparseDrive and related components. Original per-file notices remain controlling. Model weights are not included; follow the upstream model's terms.
- `navsafe/policy/sensor/gtrs_dense.py`: adapter for GTRS-Dense; the upstream SimScale model code and checkpoints are external dependencies.

This repository does not grant additional rights to datasets, checkpoints, NVIDIA images, harvested assets or external model repositories.
