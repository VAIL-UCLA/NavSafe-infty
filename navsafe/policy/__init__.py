"""NavSafe policy plugin boundary.

Phase 3 task 3.5 of the NavSafe Package Reorg spec stands up this
package. It is the canonical home of the policy-adapter abstract
hierarchy and the local re-export of the policy registration
decorator:

* :class:`BasePolicyAdapter` — common abstract parent (Requirement 3.1)
* :class:`StatePolicy` — abstract subclass for structured-state
  observations (Requirement 3.2, 3.4)
* :class:`SensorPolicy` — abstract subclass for sensor observations
  (Requirement 3.2, 3.5)
* :func:`register_policy` — the in-process registration decorator
  (re-exported from :mod:`navsafe.engine.registry` per Requirement 4.6)
* :func:`create_model_adapter` — name-based factory that resolves an
  adapter through the registry (replaces the pre-reorg
  ``navsafe.evaluation.models.create_model_adapter``)

Subsequent Phase-3 tasks port the 13 first-party adapters into this
package as concrete subclasses of either ``StatePolicy`` (task 3.6,
``ego_mlp``) or ``SensorPolicy`` (task 3.7, the 12 sensor-modality
adapters), each decorated with ``@register_policy("<name>")``.

Per the §12.6-RESOLVED no-shim policy, the pre-reorg import path
``navsafe.evaluation.models`` and its submodules are deleted
outright; callers must update imports to ``navsafe.policy.*``
directly.

Requirement 10.5 forbids semantic changes to the
``load_model`` / ``prepare_input`` / ``run_inference`` /
``parse_output`` interface; the abstract method contract here is the
verbatim contract the pre-reorg ``BaseModelAdapter`` carried.
"""

from __future__ import annotations

from navsafe.policy.base import BasePolicyAdapter
from navsafe.policy.factory import create_model_adapter
from navsafe.policy.registry import register_policy
from navsafe.policy.sensor_policy import SensorPolicy
from navsafe.policy.state_policy import StatePolicy

__all__ = [
    "BasePolicyAdapter",
    "SensorPolicy",
    "StatePolicy",
    "create_model_adapter",
    "register_policy",
]
