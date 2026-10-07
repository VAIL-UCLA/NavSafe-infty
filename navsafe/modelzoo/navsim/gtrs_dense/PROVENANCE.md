# GTRS-Dense inference

Source: https://github.com/OpenDriveLab/SimScale

Revision: e7eb8a0ef3bcc41e86ccfdf4d0e5c9bb4a5826a1.

Apache-2.0; upstream notices are retained. Only model inference dependencies are included. Configuration inheritance is flattened, nuPlan map enums and training feature builders are omitted, imports are local, and backbone preloads are disabled because complete policy checkpoints supply these tensors. Camera preprocessing and trajectory scoring are unchanged.

The trajectory vocabularies (`traj_final/` of the same upstream revision) are not included here; they are in the dataset's `model_zoo/gtrs_dense/`.

8192.npy: SHA256 cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35.

16384.npy: SHA256 e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d.
