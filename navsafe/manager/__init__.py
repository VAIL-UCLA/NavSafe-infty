"""
navsafe.manager — Per-episode managers used by the environment.

Handles scenario lifecycle, scoring lifecycle, and episode termination.

Relocated from ``navsafe.managers`` (plural) in the NavSafe Package Reorg
(Phase 3 task 3.14).
"""

from navsafe.manager.scenario_replay_manager import ScenarioReplayManager  # noqa: F401

__all__ = ["ScenarioReplayManager"]
