"""Re-export ``register_policy`` from :mod:`navsafe.engine.registry`.

Phase 3 task 3.5 of the NexusSim Package Reorg spec. This module is
the policy-package-local convenience import path:

    from navsafe.policy.registry import register_policy

The single source of truth for the registry — and the only place the
decorator is *defined* — remains :mod:`navsafe.engine.registry`
(Requirement 4.6: the CLI and the Python API resolve plugin names
against the same registry). Re-exporting here means a plugin author
who lives in ``navsafe.policy`` can decorate without reaching across
to the engine package, while everyone still hits the same singleton.
"""

from __future__ import annotations

from navsafe.engine.registry import register_policy

__all__ = ["register_policy"]
