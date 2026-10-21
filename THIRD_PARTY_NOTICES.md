# Third-party notices

Source file copyright and SPDX notices are preserved. The root Apache-2.0 license applies to the NavSafe project code; it does not override separately licensed components.

- `navsafe/_vendor/pysocialforce/`: PySocialForce by Yuxiang Gao and contributors (MIT). See the `LICENSE` file in that directory. Its logger is adapted to avoid configuring the application root logger or creating a working-directory file at import time.
- `navsafe/_vendor/nurec_grpc/`: generated Python bindings of the NVIDIA NRE gRPC protocol. NRE service/container remains an external dependency under NVIDIA terms.
- `navsafe/modelzoo/`: upstream model implementations, including NAVSIM-derived policies, DrivoR, DiffusionDrive, SparseDrive and related components. Original per-file notices remain controlling. Model weights are not included; follow the upstream model's terms.
- `navsafe/modelzoo/navsim/recogdrive/`: the ReCogDrive diffusion planner by Xiaomi Corporation (Apache-2.0, https://github.com/xiaomi-research/recogdrive); upstream derived `recogdrive_dit.py` from LightningDiT (MIT, https://github.com/hustvl/LightningDiT).
- `navsafe/modelzoo/navsim/vla_server/mtdrive_server.py`: uses the trajectory-prediction prompt of ReCogDrive (Apache-2.0, https://github.com/xiaomi-research/recogdrive).
- `navsafe/modelzoo/navsim/vla_server/internvl_conversation.py`: conversation templates from InternVL (MIT, https://github.com/OpenGVLab/InternVL), which derive from FastChat (Apache-2.0, https://github.com/lm-sys/FastChat).
- `navsafe/modelzoo/navsim/vla_server/`: inference servers that load each model from its upstream repository; that code and its checkpoints are external dependencies under their own terms.
- `third_party/AutoVLA`, `third_party/DriveLaW`, `third_party/DriveVLA-W0`, `third_party/ResWorld`, `third_party/SimWAM`: git submodules that point to the upstream repositories. Their code is not part of this repository and stays under each repository's own terms.
- `navsafe/policy/sensor/gtrs_dense.py`: adapter for GTRS-Dense; the upstream SimScale model code and checkpoints are external dependencies.

This repository does not grant additional rights to datasets, checkpoints, NVIDIA images, harvested assets or external model repositories.
