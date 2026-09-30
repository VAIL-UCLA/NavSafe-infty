"""py123d Arrow scenario loading for NavSafe evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from navsafe.scenario.scenario_description import ScenarioDescription


@runtime_checkable
class ScenarioLoaderProtocol(Protocol):
    """The minimal interface every scenario loader implements.

    A loader receives the :class:`~navsafe.env.env_cfg.EnvCfg` and
    returns a runtime
    :class:`~navsafe.scenario.scenario_description.ScenarioDescription`
    (the ScenarioNet-format dict the base env consumes). The concrete
    return type is whatever the underlying source produces; loaders do
    not invent a new description type.
    """

    def load(self, cfg: Any) -> "ScenarioDescription":
        """Produce a runtime ``ScenarioDescription`` from ``cfg``."""
        ...



from navsafe.env.loaders.py123d import Py123DLoader

__all__ = ["ScenarioLoaderProtocol", "Py123DLoader"]
