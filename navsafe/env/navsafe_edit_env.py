# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Scenario editing applied before the evaluation scene is built.

``cfg.scenario_edits`` contains declarative ``{"tool": name, **params}``
specifications applied to the loaded ScenarioDescription. Without edits,
this environment behaves identically to NavSafeEnv.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from navsafe.env.navsafe_env import NavSafeEnv


class NavSafeEditEnv(NavSafeEnv):
    """NavSafeEnv with declarative scenario edits + NuRec mesh compositing."""

    def _post_load_scenario(self, sd: Optional[Dict]) -> Optional[Dict]:
        if sd is not None and getattr(self.cfg, "scenario_edits", None):
            from navsafe.scenario.edits import apply_scenario_edits
            sd = apply_scenario_edits(sd, self.cfg.scenario_edits)
        self._adopt_reactive_actors(sd)
        return sd

    def _adopt_reactive_actors(self, sd: Optional[Dict]) -> None:
        """Hand the recipe's reactive actors to the traffic manager.

        Which actors exist is a property of the RECIPE, not of the run's
        config, so the manager cannot be told at construction time — the
        edits have to be applied first. ``spawn_reactive_actor`` parks the
        specs in the scenario metadata on its way through; this is where they
        are collected.

        A recipe that declares reactive actors while the run is on a manager
        that cannot drive them is a silent no-op — the actors would stand
        still and the episode would look merely uneventful — so it is called
        out rather than ignored.
        """
        specs = ((sd or {}).get("metadata") or {}).get("navsafe_reactive") or []
        if not specs:
            return
        manager = getattr(self, "_traffic_manager", None)
        adopt = getattr(manager, "adopt", None)
        if adopt is None:
            import logging

            logging.getLogger(__name__).error(
                "This recipe declares %d reactive actor(s) but traffic_mode=%r cannot "
                "drive them: they will stand at their spawn poses for the whole "
                "episode. Run with traffic_mode='navsafe'.",
                len(specs), getattr(self.cfg, "traffic_mode", "?"))
            return
        adopt(specs)
