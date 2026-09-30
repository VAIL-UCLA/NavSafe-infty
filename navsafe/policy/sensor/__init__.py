"""Sensor-based policy adapters.

Policies in this subpackage consume camera images, depth maps, or
LiDAR point clouds. Their ``prepare_input`` processes raw sensor
observations into model-ready tensors.

Contains all first-party adapters except ego_mlp:
- transfuser, ltf, lead_navsim, drivor
- diffusiondrive, diffusiondrivev2
- tcp, rap, uniad, vad
- alpamayo_r1, openpilot
"""
