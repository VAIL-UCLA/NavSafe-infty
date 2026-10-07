"""Factory for instantiating policy adapters by name.

Post-reorg replacement for the deleted
``navsafe.evaluation.models.create_model_adapter``. The factory is a
thin wrapper around :class:`navsafe.engine.registry.Registry`: it
resolves ``model_type`` against the ``"policies"`` registry group and
calls the resolved class with the standard adapter constructor
contract.

Per the §12.6-RESOLVED no-shim policy of the NavSafe Package Reorg
spec, this is the *only* canonical factory; there is no deprecation
shim at the old import path.

Usage::

    from navsafe.policy import create_model_adapter

    adapter = create_model_adapter("transfuser", "/path/to/ckpt.ckpt")
    adapter.load_model()

The factory accepts the same constructor kwargs each adapter accepts
(``checkpoint_path`` is mandatory; ``config_path`` is supported by the
mmcv-based adapters; per-adapter extras like ``plan_anchor_path``,
``scorer``, or ``enable_temporal_consistency`` are forwarded through
``**kwargs``).
"""

from __future__ import annotations

from typing import Any

from navsafe.engine.registry import get_registry
from navsafe.policy.base import BasePolicyAdapter


def create_model_adapter(
    model_type: str,
    checkpoint_path: str,
    config_path: str | None = None,
    **kwargs: Any,
) -> BasePolicyAdapter:
    """Instantiate a registered policy adapter by name.

    Args:
        model_type: Adapter registry key (case-insensitive). Must
            match a name registered via ``@register_policy("<name>")``.
        checkpoint_path: Path to the model checkpoint. Forwarded to
            the adapter's ``__init__`` as ``checkpoint_path``.
        config_path: Optional model config path. Forwarded to adapters
            whose constructors accept it (mmcv-based UniAD/VAD); for
            other adapters it is silently dropped if their constructor
            does not declare it.
        **kwargs: Additional adapter-specific keyword arguments
            (``plan_anchor_path``, ``scorer``,
            ``enable_temporal_consistency``, etc.).

    Returns:
        An unloaded :class:`BasePolicyAdapter` instance. Call
        ``load_model()`` on the returned object before running
        inference.

    Raises:
        ValueError: If ``model_type`` is not registered. The error
            message starts with ``"Unsupported model type"`` and lists
            the names that *are* registered.
    """
    if not isinstance(model_type, str) or not model_type:
        raise ValueError(
            f"create_model_adapter: model_type must be a non-empty string, "
            f"got {model_type!r}."
        )

    registry = get_registry()
    name = model_type.lower()

    try:
        cls = registry.lookup("policies", name)
    except KeyError as exc:
        # Re-raise as ValueError with the message contract the
        # existing tests rely on ("Unsupported model type").
        available = sorted(registry.list("policies"))
        raise ValueError(
            f"Unsupported model type: {model_type!r}. "
            f"Registered policies: {available}."
        ) from exc

    # Build the kwargs we forward. ``config_path`` is only forwarded
    # when the caller supplied a non-None value, so adapters whose
    # constructors don't declare it (the majority) don't see an
    # unexpected keyword argument.
    init_kwargs: dict[str, Any] = dict(kwargs)
    if config_path is not None:
        init_kwargs["config_path"] = config_path

    return cls(checkpoint_path=checkpoint_path, **init_kwargs)


__all__ = ["create_model_adapter"]
