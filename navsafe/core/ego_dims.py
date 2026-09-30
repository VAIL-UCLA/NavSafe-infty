# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Canonical ego footprint (metres): the MetaDrive/NexusSim default
vehicle ("ferra", 4.515 x 1.852 m — NOT nuPlan's Pacifica), shared by the
planner config, EPDMS scorers, collision classifier, and BEV renderer so
collision and drivable-area geometry stays bit-identical everywhere.
"""

EGO_LENGTH_M: float = 4.515
EGO_WIDTH_M: float = 1.852

__all__ = ["EGO_LENGTH_M", "EGO_WIDTH_M"]
