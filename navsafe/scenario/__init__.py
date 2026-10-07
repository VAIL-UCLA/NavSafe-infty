"""
navsafe.scenario

Scenario data format, type constants, and py123d Arrow ingestion.
The ``ScenarioDescription`` dict format is ported from MetaDrive (no MetaDrive
package dependency). Datasets are read from py123d Arrow logs — see
``py123d_dataset`` (multi-scene listing / load-by-index) and ``py123d_scenes``.
"""

from navsafe.scenario.type import MetaDriveType
from navsafe.scenario.scenario_description import ScenarioDescription
from navsafe.scenario.constants import TerminationState

__all__ = ["MetaDriveType", "ScenarioDescription", "TerminationState"]

# py123d-native schema and NavSafe common scenario state for BC/RL.
try:
    from navsafe.scenario.py123d_schema import Py123DScenarioData, Py123DFrameState
    from navsafe.scenario.py123d_adapter import Py123DAdapterConfig, scenario_from_py123d_scene
    from navsafe.scenario.py123d_training_extractor import (
        training_scenario_from_py123d,
        training_frame_from_py123d,
        training_frame_from_py123d_scene_api,
    )
    from navsafe.scenario.training_schema import (
        NexusScenarioLog,
        NexusFrameState,
        NexusMapState,
        NexusRouteState,
        NexusEgoState,
        NexusActorState,
        NexusTrafficLightState,
        TrainingScenarioLog,
        TrainingFrameState,
        TrainingMapState,
        TrainingRouteState,
    )
    __all__ += [
        "Py123DScenarioData",
        "Py123DFrameState",
        "Py123DAdapterConfig",
        "scenario_from_py123d_scene",
        "training_scenario_from_py123d",
        "training_frame_from_py123d",
        "training_frame_from_py123d_scene_api",
        "NexusScenarioLog",
        "NexusFrameState",
        "NexusMapState",
        "NexusRouteState",
        "NexusEgoState",
        "NexusActorState",
        "NexusTrafficLightState",
        "TrainingScenarioLog",
        "TrainingFrameState",
        "TrainingMapState",
        "TrainingRouteState",
    ]
except ImportError:
    pass
