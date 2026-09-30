# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python checks for ``_map_agent_type`` (py123d actor label → MetaDriveType).

Focus: the ``two_wheeler`` label (``DefaultBoxDetectionLabel.TWO_WHEELER`` →
``label.name.lower()``) must route to ``CYCLIST`` so two-wheeler actors spawn as
cyclists in the replay manager instead of falling through to OTHER/generic.
"""

import pytest

from navsafe.scenario.py123d_scenario_description import _map_agent_type
from navsafe.scenario.type import MetaDriveType


def test_two_wheeler_maps_to_cyclist():
    # The fix under review: TWO_WHEELER enum -> "two_wheeler" -> CYCLIST.
    assert _map_agent_type("two_wheeler") == MetaDriveType.CYCLIST


def test_vehicle_labels_map_to_vehicle():
    assert _map_agent_type("vehicle") == MetaDriveType.VEHICLE
    assert _map_agent_type("regular_vehicle") == MetaDriveType.VEHICLE


def test_pedestrian_and_person_map_to_pedestrian():
    # "pedestrian" is an exact key; "person" (DefaultBoxDetectionLabel.PERSON ->
    # "person") is only caught by the keyword fallback — assert both work.
    assert _map_agent_type("pedestrian") == MetaDriveType.PEDESTRIAN
    assert _map_agent_type("person") == MetaDriveType.PEDESTRIAN


def test_case_insensitive_and_other_fallback():
    assert _map_agent_type("TWO_WHEELER") == MetaDriveType.CYCLIST
    # An unmapped label falls back to OTHER (not silently VEHICLE).
    assert _map_agent_type("totally_unknown_label_xyz") == MetaDriveType.OTHER


def test_other_cyclist_labels_still_map_to_cyclist():
    assert _map_agent_type("bicyclist") == MetaDriveType.CYCLIST
    assert _map_agent_type("motorcyclist") == MetaDriveType.CYCLIST


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
