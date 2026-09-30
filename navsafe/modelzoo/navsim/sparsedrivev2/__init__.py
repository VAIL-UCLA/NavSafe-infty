# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""SparseDriveV2 model code, vendored for inference.

Upstream is ``swc-17/SparseDriveV2`` (a NAVSIM fork). Only the modules the
forward pass touches are vendored — model, backbone, decoder, blocks, grid
mask, config, and the deformable-aggregation CUDA op. What is deliberately
*not* here:

* ``sparsedrive_agent.py`` — a ``pytorch_lightning`` module wrapping the model
  for training; the adapter drives ``SparseDriveModel`` directly.
* ``sparsedrive_features.py`` — its feature builder reads NAVSIM ``AgentInput``
  dataclasses and images off disk. The adapter reproduces the same tensors from
  rendered frames plus ``OPENSCENE_CAMERA_PARAMS``; see
  ``navsafe/policy/sensor/sparsedrivev2.py``.
* ``scorer/`` — PDM scoring used only to supervise training, and it needs hydra
  + omegaconf + the nuplan devkit.

The upstream sources are otherwise unmodified, so a future version bump is a
re-copy plus the same import rewrites (``navsim.agents.sparsedrive.ops`` ->
``.ops``, ``navsim.common.enums`` -> local, nuplan ``TrajectorySampling`` ->
``navsafe.modelzoo.common``).
"""
