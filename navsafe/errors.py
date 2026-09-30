# Copyright (c) 2022-2025, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Shared exception hierarchy for NexusSim.

Every custom exception defined in ``navsafe`` derives from
:class:`NexusSimError` — usually alongside its original builtin base
(``ValueError``, ``RuntimeError``, …), which is preserved so existing
``except ValueError`` / ``except RuntimeError`` callers keep working.

This module is intentionally dependency-free: it imports nothing from
``navsafe`` or third-party packages, so it is safe to import from any
module — including the light, CI-safe paths that must not pull in the
heavy IsaacSim/torch stack.
"""

from __future__ import annotations


class NexusSimError(Exception):
    """Base class for all errors raised by NexusSim.

    Contract: ``except NexusSimError`` is the catch-all boundary for
    NexusSim-originated failures — anything NexusSim itself detects and
    raises derives from this class, distinguishing expected, reportable
    failures from programming bugs and third-party errors.
    """


class ScenarioError(NexusSimError):
    """Error in scenario data, conversion, or the ``ScenarioDescription`` IR."""


class RenderError(NexusSimError):
    """Error raised by the NuRec renderer."""


class EvaluationError(NexusSimError):
    """Error raised by the evaluation stack (scorers, evaluators, routes, artifacts)."""
