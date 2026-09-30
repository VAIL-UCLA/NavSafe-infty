# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Agent-level building blocks (robots, sensors, traffic actors).

Per design.md §4 (Components and Interfaces), ``navsafe.component`` houses
"things attached to an env, not the env itself" — robot specs, sensor specs
(camera, gaussian camera, LiDAR), and traffic actors (per-vehicle IDM
dynamics).  Subpackages land here as Phase 3 tasks complete:

* ``navsafe.component.traffic_agent`` — per-vehicle traffic actors (task 3.11).
* ``navsafe.component.sensors`` — camera / LiDAR specs (task 3.12).
* ``navsafe.component.robot``  — USD/asset robot wiring (task 3.13).

Pure-Python dynamics ship under ``navsafe.core``; this package is the home
for the agent-attached primitives that may carry sim-side resources.
"""
