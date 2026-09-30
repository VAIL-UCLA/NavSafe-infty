# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The traffic manager NavSafe recipes drive their actors with.

A recipe names the actors it inserted and the policy each one runs. This
manager takes over exactly those ids — unconditionally, by name — and hands
**everything else** to :class:`~navsafe.traffic.semi_reactive.
SemiReactiveTraffic`.

That "by name" is the whole difference from semi-reactive, which decides
takeover from geometry (a vehicle, behind the ego, inside a lateral radius).
Those are MetaDrive's rules for making *background* traffic plausible, and they
are the wrong rules for an actor that exists solely to create the event under
test: a pedestrian fails the vehicle check, an oncoming car fails the
behind-the-ego check, and the actor the leaf is about would replay while the
traffic around it reacted.

They are, however, the RIGHT rules for the traffic around it, and this manager
used to leave that traffic on the log. The cost was not subtle: a closed-loop
ego that slows for the event gets driven into from behind by a car replaying a
trajectory recorded when the ego did not slow. Measured on
17b0992157365222 — six runs across two policies (drivor, transfuser), two
trackers (pure_pursuit, lqr) and two placements ended at frame 29-33 with the
same logged vehicle, ``1b60f22d4a745373``, rear-ending the ego. Nothing about
the scenario was being tested by then; the episode was over at 2.6 s.

So the split is by ROLE: the recipe's actors run their authored policies, the
rest is reactive background. Neither manager may touch the other's agents —
see ``exclude_ids``.

Stage note: this manager itself owns no USD prims (see
:mod:`navsafe.traffic.stage_free`), which is what lets it run under the
closed-loop evaluator's pure-Python loop. The delegate does touch the stage,
and is stepped from here rather than from ``_advance_replay_agent_prims`` so
that it is stepped exactly once.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from navsafe.traffic.stage_free import Driver, StageFreeTraffic, WorldView

logger = logging.getLogger(__name__)

#: recipe `policy.kind` -> driver class, filled by the drivers themselves
DRIVERS: Dict[str, type] = {}


def register_driver(kind: str):
    def deco(cls):
        DRIVERS[kind] = cls
        return cls
    return deco


@register_driver("static")
class StaticDriver(Driver):
    """Never moves. The honest encoding of a parked car or a road blockage.

    A static actor is not a degenerate reactive one — it has no goal and no
    lead, and running it through a car-following model would only produce
    numerical noise around a pose that should be exactly constant.
    """

    def __init__(self, agent_id: Any, spawn: Dict[str, Any], **_params):
        self.agent_id = agent_id
        self.spawn = dict(spawn)
        self.reset(spawn=self.spawn)

    def reset(self, *, spawn: Dict[str, Any]) -> None:
        p = np.asarray(spawn.get("position", (0.0, 0.0, 0.0)), np.float64)
        self._pos = np.array([p[0], p[1], p[2] if len(p) > 2 else 0.0])
        self._heading = float(spawn.get("heading", 0.0))
        self._dims = (float(spawn.get("length", 4.5)),
                      float(spawn.get("width", 1.8)))

    def step(self, world: WorldView, dt: float) -> None:
        return

    def pose(self) -> Dict[str, Any]:
        return {
            "position": self._pos.astype(np.float32),
            "heading": self._heading,
            "velocity": np.zeros(2, dtype=np.float32),
            "length": self._dims[0],
            "width": self._dims[1],
        }


class NavSafeTraffic(StageFreeTraffic):
    """Drives the actors a recipe injected; the rest is reactive background."""

    def __init__(self, actors: Optional[List[Dict[str, Any]]] = None,
                 background: Optional[Any] = None) -> None:
        """``background`` is the manager for everything the recipe does not own.

        Defaults to a :class:`SemiReactiveTraffic`. Pass ``False`` for the old
        behaviour — logged traffic replayed on rails — which is only honest for
        a run that deliberately wants a non-reacting world.
        """
        super().__init__()
        self._specs: List[Dict[str, Any]] = list(actors or [])
        self._built = False
        if background is None:
            from navsafe.traffic.semi_reactive import SemiReactiveTraffic
            background = SemiReactiveTraffic()
        self._background = background or None

    def use_fixed_background(self) -> None:
        """Explicit counterfactual mode: replay all non-target traffic."""
        self._background = None

    # -- construction --------------------------------------------------
    def adopt(self, actors: List[Dict[str, Any]]) -> None:
        """Take the reactive actor specs a recipe produced.

        Called by the env once the scenario edits have been applied, because
        the specs travel with the edits rather than with the config: which
        actors exist is a property of the recipe, not of the run.
        """
        self._specs = list(actors or [])
        self._built = False

    def _build(self, env: Any) -> None:
        # Importing a driver module is what registers it. Done here rather
        # than at module scope because the drivers import this module for
        # `register_driver`, and a top-level import would close the cycle.
        from navsafe.traffic import idm_driver  # noqa: F401
        from navsafe.traffic import social_force  # noqa: F401
        from navsafe.traffic import timed_braking  # noqa: F401
        from navsafe.traffic import timed_cut_in  # noqa: F401
        from navsafe.traffic import authored_path  # noqa: F401

        self.drivers = {}
        for spec in self._specs:
            kind = str((spec.get("policy") or {}).get("kind", "static"))
            # Keyed by TRACK id: `pose_overrides` is merged into the agent
            # list, which the env keys by track id. The actor's own name
            # ("oncoming_vehicle") is what a human reads; the track it became
            # ("navsafe_oncoming_vehicle") is what the world calls it.
            agent_id = spec.get("track_id") or spec.get("name") or spec.get("id")
            cls = DRIVERS.get(kind)
            if cls is None:
                logger.error(
                    "navsafe traffic: actor %r asks for policy %r, which is not "
                    "registered (%s). Leaving it on its logged track — the "
                    "scenario will run, but not the one that was authored.",
                    agent_id, kind, sorted(DRIVERS))
                continue
            params = dict((spec.get("policy") or {}).get("params") or {})
            try:
                self.drivers[agent_id] = cls(
                    agent_id, dict(spec.get("spawn") or {}),
                    policy=dict(spec.get("policy") or {}), **params)
            except Exception:  # noqa: BLE001
                if getattr(cls, "strict", False):
                    raise
                logger.exception("navsafe traffic: could not build driver for %r", agent_id)
        self._built = True
        if self.drivers:
            logger.info("navsafe traffic: driving %d actor(s): %s",
                        len(self.drivers), sorted(map(str, self.drivers)))

    def _hand_off_ownership(self) -> None:
        """Tell the background manager which agents it may not touch.

        Set every frame rather than once: which actors the recipe owns is known
        only after the scenario edits are applied, and ``adopt`` may run again
        between episodes.
        """
        if self._background is not None:
            self._background.exclude_ids = set(self.drivers)

    def _background_call(self, what: str, *args) -> bool:
        """Run one background-manager call, disabling it if it cannot run.

        The delegate needs more of the env than this manager does — an agent
        manager and a USD stage — so a harness that provides neither (or a
        future env that drops them) would otherwise turn a working episode into
        a crash. Degrading is the right answer; degrading QUIETLY is not, since
        traffic back on rails is the exact failure this delegation exists to
        remove. So it is said once, at WARNING, with the consequence spelled
        out, and the delegate is not retried.
        """
        if self._background is None:
            return False
        try:
            getattr(self._background, what)(*args)
            return True
        except Exception as exc:  # noqa: BLE001 — any env shortfall, same answer
            logger.warning(
                "navsafe traffic: background traffic could not %s (%s: %s), so the "
                "logged vehicles stay on their recorded trajectories. An ego that "
                "slows for the event can be driven into from behind by a car "
                "replaying a trajectory recorded when it did not slow.",
                what, type(exc).__name__, exc)
            self._background = None
            return False

    # -- TrafficManager ------------------------------------------------
    def reset(self, env: Any) -> None:
        self._build(env)
        super().reset(env)
        self._hand_off_ownership()
        self._background_call("reset", env)

    def step(self, env: Any, dt: float) -> None:
        if not self._built:
            self._build(env)
        # StageFreeTraffic.step returns early when there are no drivers, which
        # a mined recipe (zero actors) always has — so capture the recipe's
        # overrides explicitly rather than relying on it having written them.
        super().step(env, dt)
        recipe_poses = dict(self.pose_overrides) if self.drivers else {}
        if self._background is not None:
            self._hand_off_ownership()
            if self._background_call("step", env, dt):
                # Background first, recipe on top: the two id sets are disjoint
                # by construction (exclude_ids), so the order only matters if
                # that invariant ever breaks — and if it does, the actor the
                # leaf is about must win.
                merged = dict(getattr(self._background, "pose_overrides", {}) or {})
                merged.update(recipe_poses)
                self.pose_overrides = merged


__all__ = ["DRIVERS", "NavSafeTraffic", "StaticDriver", "register_driver"]
