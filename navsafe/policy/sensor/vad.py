"""VAD — a :class:`SensorPolicy` adapter (split from ``uniad_vad_adapter.py``).

Phase 3 task 3.7 of the NavSafe Package Reorg spec splits the
pre-reorg single-file ``uniad_vad_adapter.py`` into two modules
under :mod:`navsafe.policy`: :mod:`navsafe.policy.uniad` (UniAD)
and this file (VAD). Both adapters historically share identical
mmcv-based plumbing — only ``model_type`` and a few
trajectory-parsing branches differ — so the shared implementation
remains the :class:`~navsafe.policy.uniad.UniADVADAdapter` class
defined in :mod:`navsafe.policy.uniad`, and this module thinly
subclasses it.

Backs:

* Requirement 3.3 — every first-party adapter inherits from exactly
  one of :class:`StatePolicy` / :class:`SensorPolicy`.
* Requirement 3.5 — VAD is one of the 12 first-party
  sensor-modality adapters and inherits from
  :class:`SensorPolicy` (transitively, via
  :class:`UniADVADAdapter`).
* Requirement 3.7 — base-class choice is determined by what
  :meth:`prepare_input` consumes; VAD consumes 6-camera Bench2Drive
  imagery, hence ``SensorPolicy``.
* Requirement 10.5 — :meth:`load_model`, :meth:`prepare_input`,
  :meth:`run_inference`, :meth:`parse_output` semantics are
  preserved verbatim from the pre-reorg ``UniADVADAdapter``.

The :func:`register_policy` decorator wires :class:`VADAdapter`
into the process-wide registry under the name ``"vad"`` so the CLI
(``navsafe eval --model-type vad``) and the public API resolve the
same class against the same registry (Requirement 4.6).
"""

from __future__ import annotations

from navsafe.policy.registry import register_policy
from navsafe.policy.sensor.uniad import UniADVADAdapter


@register_policy("vad")
class VADAdapter(UniADVADAdapter):
    """VAD-specific subclass of :class:`UniADVADAdapter`.

    Pins ``model_type="vad"`` so the shared
    ``prepare_input``/``run_inference``/``parse_output`` plumbing
    routes through the VAD branches (notably the one-hot command
    encoding in ``prepare_input``). The class exists purely to
    carry the unique ``@register_policy("vad")`` name (Requirement
    3.5, 4.6) and to give downstream code a clean ``VADAdapter``
    import target separate from the UniAD subclass in
    :mod:`navsafe.policy.uniad`.
    """

    def __init__(self, checkpoint_path: str, config_path: str | None = None, **kwargs):
        # Ignore any caller-supplied ``model_type`` to keep the
        # registration name and the runtime model_type in lockstep.
        kwargs.pop("model_type", None)
        super().__init__(
            checkpoint_path,
            config_path=config_path,
            model_type="vad",
            **kwargs,
        )


__all__ = ["VADAdapter"]
