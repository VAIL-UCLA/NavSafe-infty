"""Vendored ReCogDrive diffusion-planner modules.

Copied from https://github.com/xiaomi-research/recogdrive
(``navsim/agents/recogdrive``) with the navsim/nuplan imports removed —
those are only reached by the GRPO training paths, never by
``ReCogDriveDiffusionPlanner.get_action``.  The VLM backbone is NOT
vendored; the adapter talks to it through the VLA subprocess server
(``navsafe/modelzoo/navsim/vla_server/recogdrive_server.py``).
"""
