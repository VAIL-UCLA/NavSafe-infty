"""NavSafe plugin registry — single source of truth for policies,
scorers, renderers, and envs.

Phase 3 task 3.2 of the NavSafe Package Reorg spec; backs Requirements
4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9 and 11.6. The :class:`Registry`
holds the in-process record of every plugin registered through either
the in-process ``@register_*`` decorators (this module) or the
out-of-process ``importlib.metadata.entry_points`` mechanism
(:meth:`Registry.discover_entry_points`).

The registry has exactly four groups (Requirement 4.1):

* ``"policies"``  — entry-points group ``"navsafe.policies"``
* ``"scorers"``   — entry-points group ``"navsafe.scorers"``
* ``"renderers"`` — entry-points group ``"navsafe.renderers"``
* ``"envs"``      — entry-points group ``"navsafe.envs"``

Design choices worth flagging up front (everything else is documented
inline at the call site):

* **Stdlib only at module load.** Importing this module must not pull
  in :mod:`navsafe.policy`, :mod:`navsafe.render`, IsaacSim, or any
  third-party plugin. Base-class validation is therefore *deferred*:
  bases are looked up lazily and registrations are accepted without
  validation when the base module hasn't landed yet (Phase-3 tasks 3.5
  and 3.10 ship the bases). Once the base is importable, ``register``
  enforces a strict ``issubclass`` check and rejects duck typing
  (Requirement 4.7).
* **Lazy plugin discovery — first-party AND third-party.**
  ``discover_entry_points`` is a no-op until first ``lookup`` /
  ``list`` (or first explicit call). When it runs it first imports the
  first-party adapter/scorer/renderer modules in
  :data:`_FIRST_PARTY_MODULES` (their module-level ``@register_*``
  decorators do the registering), then enumerates third-party
  setuptools entry points. Both halves are failure-tolerant: a module
  or plugin that raises is logged-and-skipped so consulting the
  registry never crashes (Requirements 4.8, 4.9, 11.6). This used to
  happen eagerly in ``navsafe/__init__.py``; it moved here so
  ``import navsafe`` stays fast and side-effect-free (no IsaacSim /
  torch import, no kit bootstrap) while any actual registry consumer
  still sees every plugin.
* **Idempotent re-registration.** ``register(group, name, cls)`` is a
  no-op if ``(group, name)`` already maps to the same ``cls``;
  conflicting registrations raise ``ValueError`` unless
  ``overwrite=True`` (Requirements 4.4, 4.5).
* **Single source of truth.** A module-level singleton (``get_registry``)
  is what the four decorators delegate to and what
  :func:`navsafe.evaluation.evaluator` resolves names against
  (Requirement 4.6 — CLI and Python API agree).

Source of truth for the architecture is design.md §7 (Plugin/Registry
Architecture).
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group → expected base class lazy resolver
# ---------------------------------------------------------------------------
#
# Each registry group has a corresponding abstract base class (or, for
# policies, a *tuple* of acceptable bases). The bases live in modules
# that don't exist yet at task-3.2 time (``navsafe.policy.state_policy``
# arrives in 3.5; ``navsafe.render.base`` in 3.10). To keep this
# module stdlib-only and not couple our load order to the rest of the
# package, we describe each group's bases as a *tuple of
# (module_path, class_name) pairs* and resolve them lazily via
# :func:`_resolve_bases`.
#
# When the base modules are unavailable (because Phase 3 hasn't landed
# them yet, or the user is running in a stripped-down environment),
# ``register`` accepts the registration and emits a debug-level note;
# strict validation kicks in automatically once the bases become
# importable. This keeps the registry usable today without losing
# Requirement 4.7's guarantee tomorrow.

# Group name → tuple of (module_path, class_name) pairs that the
# registered class must be a subclass of.
#
# For ``policies`` the tuple has two entries: a registered class needs
# to subclass *either* ``StatePolicy`` *or* ``SensorPolicy`` (not
# both). For the other three groups, exactly one base is expected.
_BASE_LOCATIONS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "policies": (
        ("navsafe.policy.state_policy", "StatePolicy"),
        ("navsafe.policy.sensor_policy", "SensorPolicy"),
    ),
    "scorers": (("navsafe.evaluation.scorers.base", "Scorer"),),
    "renderers": (("navsafe.render.base", "SceneRenderer"),),
    # The env base is the abstract parent of NavSafeEnv. Phase-3
    # task 3.17 lands the concrete class; until then we accept
    # registrations and validate when the module appears.
    "envs": (("navsafe.env.navsafe_env", "NavSafeEnv"),),
}


def _resolve_bases(group: str) -> Optional[Tuple[type, ...]]:
    """Lazily resolve the base classes for ``group``.

    Returns the tuple of base classes if every base in the group's
    base-locations table is currently importable. Returns ``None`` if
    *any* base module is unavailable — in that case ``register`` skips
    strict validation for this registration but the next registration
    that fires after the bases land will be checked.

    The split-tuple semantics for ``policies`` (subclass of
    ``StatePolicy`` *or* ``SensorPolicy``) is handled by
    ``issubclass(cls, resolved_bases)`` where ``resolved_bases`` is the
    tuple form — that's :func:`isinstance`/`issubclass`-native.
    """
    locations = _BASE_LOCATIONS.get(group)
    if locations is None:
        # Unknown group; ``register`` will raise before this matters.
        return None

    resolved: list[type] = []
    for module_path, class_name in locations:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Phase 3 hasn't landed this base yet (or it's been
            # removed). Skip strict validation — we'll re-check on the
            # next registration once the module is importable.
            return None
        base = getattr(module, class_name, None)
        if not isinstance(base, type):
            # Module imported but the symbol is missing or isn't a
            # class. Treat the same as a missing base — defer.
            return None
        resolved.append(base)
    return tuple(resolved)


# ---------------------------------------------------------------------------
# Group → entry-points group name
# ---------------------------------------------------------------------------
#
# The registry uses short group keys (``"policies"``); the
# corresponding entry-points group names are namespaced
# (``"navsafe.policies"``). Keep both maps in lockstep.
_ENTRY_POINT_GROUPS: Dict[str, str] = {
    "policies": "navsafe.policies",
    "scorers": "navsafe.scorers",
    "renderers": "navsafe.renderers",
    "envs": "navsafe.envs",
}


# Exposed for callers that want to enumerate the four groups (e.g.
# tests, the CLI ``--list-plugins`` flag in Phase 4) without
# string-typoing them.
GROUPS: Tuple[str, ...] = tuple(_ENTRY_POINT_GROUPS.keys())


# ---------------------------------------------------------------------------
# First-party plugin modules (lazy import, once per process)
# ---------------------------------------------------------------------------
#
# Importing these modules fires their module-level ``@register_*``
# decorators (which write to the singleton returned by
# :func:`get_registry`). Historically ``navsafe/__init__.py`` imported
# them eagerly; they moved here so the imports run lazily on first
# registry consultation instead — ``import navsafe`` must not import
# IsaacSim/torch (Requirement 11.1; see the package docstring).
#
# * The policy adapters are listed individually because the policy
#   ``__init__`` only exports base classes.
# * ``navsafe.evaluation.scorers`` fires any ``@register_scorer``.
# * ``navsafe.render`` eagerly imports all backend modules, firing
#   their ``@register_renderer`` decorators.
_FIRST_PARTY_MODULES: Tuple[str, ...] = (
    "navsafe.policy.state.ego_mlp",
    "navsafe.policy.state.idm_centerline",
    "navsafe.policy.state.pdm_closed",
    "navsafe.policy.sensor.transfuser",
    "navsafe.policy.sensor.ltf",
    "navsafe.policy.sensor.lead_navsim",
    "navsafe.policy.sensor.drivor",
    "navsafe.policy.sensor.gtrs_dense",
    "navsafe.policy.sensor.prioreye",
    "navsafe.policy.sensor.diffusiondrive",
    "navsafe.policy.sensor.diffusiondrivev2",
    "navsafe.policy.sensor.sparsedrivev2",
    "navsafe.policy.sensor.tcp",
    "navsafe.policy.sensor.rap",
    "navsafe.policy.sensor.uniad",
    "navsafe.policy.sensor.vad",
    "navsafe.policy.sensor.alpamayo_r1",
    "navsafe.policy.sensor.openpilot",
    "navsafe.policy.sensor.recogdrive",
    "navsafe.policy.sensor.recogdrive_rerank",
    "navsafe.policy.sensor.mtdrive",
    "navsafe.policy.sensor.autovla",
    "navsafe.policy.sensor.drivelaw",
    "navsafe.policy.sensor.simwam",
    "navsafe.policy.sensor.resworld",
    "navsafe.policy.sensor.drivevla_w0",
    "navsafe.evaluation.scorers",
    "navsafe.render",
)

# Process-wide once-guard for the first-party imports. Module imports are
# global (sys.modules) anyway, so this is intentionally NOT per-Registry
# state: a fresh Registry() in tests re-running discovery must not pay
# for (or double-log) the first-party import pass.
#
# The lock is RE-ENTRANT: a module imported by the pass may itself
# consult the registry at import time, which re-triggers discovery on
# the same thread; the RLock lets that re-entrant call through to the
# flag check (already True — set before importing) instead of
# self-deadlocking. Other threads block until the pass completes.
_first_party_imported: bool = False
_first_party_lock = threading.RLock()


def _import_first_party_modules() -> None:
    """Import every first-party plugin module, once per process.

    Failure-tolerant: each module import is individually wrapped, so a
    missing optional dependency (e.g. IsaacSim for the asset renderer,
    or a heavy modelzoo dep for a specific adapter) logs a warning but
    never crashes the caller. ``SystemExit`` (not an ``Exception``) is
    included because a failed IsaacSim kernel bootstrap calls
    ``sys.exit()`` on import; an optional plugin must never take down a
    registry consultation.
    """
    global _first_party_imported
    with _first_party_lock:
        if _first_party_imported:
            return
        # Mark BEFORE importing: a re-entrant call from inside one of
        # these module imports (same thread, admitted by the RLock) hits
        # the flag check above and returns instead of recursing.
        _first_party_imported = True
        for mod_path in _FIRST_PARTY_MODULES:
            try:
                importlib.import_module(mod_path)
            except (Exception, SystemExit) as exc:
                logger.warning(
                    "navsafe: failed to import first-party plugin module "
                    "%r (%s: %s). Its plugins will not be available in "
                    "the registry. Install the required optional "
                    "dependencies to enable them.",
                    mod_path,
                    type(exc).__name__,
                    exc,
                )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """In-process record of every registered plugin.

    See module docstring for the four groups, idempotency contract,
    lazy base-class validation, and lazy entry-points discovery.

    Thread-safety: all mutations are guarded by a single re-entrant
    lock. The lock is re-entrant so ``discover_entry_points`` can call
    ``register`` while holding it.
    """

    def __init__(self) -> None:
        # group name → name → registered class.
        self._records: Dict[str, Dict[str, type]] = {
            group: {} for group in _ENTRY_POINT_GROUPS
        }
        # Whether ``discover_entry_points`` has run at least once.
        # Used by ``lookup`` to trigger discovery on first access
        # (Requirement 4.8 — lazy discovery).
        self._entry_points_loaded: bool = False
        # Re-entrant so ``discover_entry_points`` can call ``register``
        # while still holding the lock.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(
        self,
        group: str,
        name: str,
        cls: type,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register ``cls`` under ``(group, name)``.

        Args:
            group: One of :data:`GROUPS` (``"policies"``, ``"scorers"``,
                ``"renderers"``, ``"envs"``).
            name: Plugin name; the same key the CLI ``--model-type`` /
                ``--trajectory-scorer`` flags resolve against.
            cls: The plugin class.
            overwrite: When ``True``, replace an existing
                ``(group, name)`` registration that maps to a different
                class. When ``False`` (the default), conflicting
                registrations raise ``ValueError``.

        Raises:
            ValueError: If ``group`` is unknown; if ``cls`` is not a
                class; if the registration conflicts with an existing
                ``(group, name)`` binding and ``overwrite=False``; or
                if ``cls`` does not subclass the expected base for
                ``group`` (when the base is currently importable).

        Idempotency (Requirement 4.4): registering the same
        ``(group, name, cls)`` triple twice is a no-op. The second call
        does not run validation again, does not log, and does not
        toggle the registration's effective state.
        """
        if group not in self._records:
            raise ValueError(
                f"Unknown registry group {group!r}. "
                f"Expected one of {sorted(self._records)}."
            )
        if not isinstance(cls, type):
            raise ValueError(
                f"Registry.register expected a class for "
                f"({group!r}, {name!r}), got {cls!r} of type "
                f"{type(cls).__name__}."
            )

        with self._lock:
            existing = self._records[group].get(name)
            if existing is cls:
                # Idempotent re-registration (Requirement 4.4). No
                # validation, no log — just return.
                return
            if existing is not None and not overwrite:
                raise ValueError(
                    f"Conflicting registration for "
                    f"({group!r}, {name!r}): already bound to "
                    f"{existing!r}, refusing to rebind to {cls!r}. "
                    f"Pass overwrite=True to replace."
                )

            # Strict subclass validation (Requirement 4.7). Skipped
            # when the base modules aren't yet importable — see
            # ``_resolve_bases``.
            bases = _resolve_bases(group)
            if bases is not None and not issubclass(cls, bases):
                base_names = ", ".join(
                    f"{b.__module__}.{b.__qualname__}" for b in bases
                )
                raise ValueError(
                    f"Cannot register {cls!r} under "
                    f"({group!r}, {name!r}): class must subclass one "
                    f"of ({base_names})."
                )
            if bases is None:
                # Bases unavailable — we accept the registration but
                # leave a breadcrumb for future debugging. Use DEBUG
                # level: this is normal during early Phase-3 work.
                logger.debug(
                    "Registering (%s, %s) -> %r without strict base "
                    "validation: base classes for group %r are not "
                    "yet importable.",
                    group,
                    name,
                    cls,
                    group,
                )

            self._records[group][name] = cls

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, group: str, name: str) -> type:
        """Return the class registered under ``(group, name)``.

        Triggers :meth:`discover_entry_points` on first call so
        out-of-process plugins are discovered lazily (Requirement
        4.8 — discovery is enumerated at import but loaded on first
        lookup).

        Raises:
            KeyError: If ``(group, name)`` is not registered after
                discovery has run. The error message includes the list
                of registered names for the group, so users see what
                the closest matches are.
            ValueError: If ``group`` is unknown.
        """
        if group not in self._records:
            raise ValueError(
                f"Unknown registry group {group!r}. "
                f"Expected one of {sorted(self._records)}."
            )

        # Lazy discovery on first lookup (Requirement 4.8). Triggered
        # BEFORE taking the registry lock: discovery imports plugin
        # modules (first-party adapters, third-party entry points) whose
        # module-level ``@register_*`` decorators re-acquire this lock,
        # and holding it across those imports would set up a lock-order
        # inversion with the interpreter's per-module import locks (a
        # thread importing an adapter directly blocks in ``register`` on
        # our lock while the discovery thread blocks on that module's
        # import lock — a silent deadlock the import system cannot
        # detect). ``discover_entry_points`` serializes internally and
        # is idempotent, so this unlocked flag check is a benign race.
        if not self._entry_points_loaded:
            self.discover_entry_points()
        with self._lock:
            cls = self._records[group].get(name)

        if cls is None:
            available = sorted(self._records[group])
            raise KeyError(
                f"No plugin registered under ({group!r}, {name!r}). "
                f"Registered names in {group!r}: {available}."
            )
        return cls

    def list(self, group: str) -> list[str]:  # noqa: A003 — match design API
        """Return the sorted list of names registered under ``group``.

        Triggers :meth:`discover_entry_points` on first call for the
        same reasons as :meth:`lookup`.

        Raises:
            ValueError: If ``group`` is unknown.
        """
        if group not in self._records:
            raise ValueError(
                f"Unknown registry group {group!r}. "
                f"Expected one of {sorted(self._records)}."
            )
        # Unlocked trigger for the same lock-ordering reason as
        # ``lookup`` (see the comment there).
        if not self._entry_points_loaded:
            self.discover_entry_points()
        with self._lock:
            return sorted(self._records[group].keys())

    # ------------------------------------------------------------------
    # Entry-points discovery
    # ------------------------------------------------------------------

    def discover_entry_points(self) -> None:
        """Import first-party plugin modules, then enumerate entry points.

        First runs :func:`_import_first_party_modules` (once per
        process) so the in-tree adapters/scorers/renderers register
        before any third-party plugin is loaded — the same relative
        order the old eager ``import navsafe`` wiring had, so a
        conflicting third-party name is still the one that gets
        rejected.

        Then iterates over ``importlib.metadata.entry_points(group=...)`` for
        each of ``navsafe.policies``, ``navsafe.scorers``,
        ``navsafe.renderers``, and ``navsafe.envs``. For each entry
        point, calls ``ep.load()`` (which imports the providing module
        and resolves the attribute) and registers the resulting class
        under the registry.

        Failure tolerance (Requirements 4.9, 11.6): if any individual
        entry point raises during load — typically because a
        third-party plugin has an optional dependency that isn't
        installed — the exception is logged at WARNING level naming
        the plugin and the loop continues. ``import navsafe`` is
        therefore robust against missing optional plugin deps
        (Requirement 4.8).

        Idempotent: subsequent calls are no-ops. The marker is set
        even when no entry points are discovered, so a clean
        environment doesn't pay for repeated enumerations.
        """
        # First-party modules first (see docstring for ordering) — but
        # ONLY for the process-wide singleton, and BEFORE taking the
        # registry lock.
        #
        # Singleton-only: the first-party ``@register_*`` decorators
        # write to ``get_registry()``, so importing them on behalf of a
        # non-singleton ``Registry()`` (a test fixture, typically)
        # registers nothing that instance can see — while still running
        # heavyweight third-party imports inside whatever context the
        # fixture set up (e.g. a monkeypatched
        # ``importlib.metadata.entry_points``, which poisons libraries
        # like triton that consult entry points during their own
        # import).
        #
        # Outside the lock: the pass imports modules whose decorators
        # re-acquire ``self._lock``; holding it here would create a
        # lock-order inversion with the interpreter's per-module import
        # locks (see the comment in ``lookup``).
        # ``_import_first_party_modules`` serializes internally and is
        # once-per-process idempotent.
        if self is _REGISTRY and not self._entry_points_loaded:
            _import_first_party_modules()

        # Take the lock for the entry-points pass: callers expect
        # ``discover_entry_points()`` to leave the registry in a
        # consistent state before returning.
        with self._lock:
            if self._entry_points_loaded:
                return
            # Mark as loaded *before* iterating so a re-entrant
            # ``register`` call from inside ``ep.load()`` (the loaded
            # module may itself call ``register_*``) doesn't trigger a
            # second discovery pass.
            self._entry_points_loaded = True

            # ``importlib.metadata`` is stdlib in 3.10+; the
            # ``entry_points(group=...)`` keyword form is the
            # 3.10-compatible API.
            try:
                from importlib.metadata import entry_points
            except ImportError:  # pragma: no cover — 3.9 fallback
                logger.debug(
                    "importlib.metadata.entry_points unavailable; "
                    "skipping out-of-process plugin discovery."
                )
                return

            for group, ep_group in _ENTRY_POINT_GROUPS.items():
                try:
                    eps: Iterable[Any] = entry_points(group=ep_group)
                except Exception:  # pragma: no cover — defensive
                    # Some Python versions and metadata backends raise
                    # on unknown groups instead of returning empty.
                    # Treat as empty.
                    logger.debug(
                        "entry_points(group=%r) raised; treating as "
                        "empty.",
                        ep_group,
                        exc_info=True,
                    )
                    continue

                for ep in eps:
                    self._load_entry_point(group, ep)

    def _load_entry_point(self, group: str, ep: Any) -> None:
        """Load a single entry point and register its target.

        Split out from :meth:`discover_entry_points` so the inner
        loop's exception handling reads cleanly. Catches *any*
        exception from ``ep.load()`` — missing optional dependencies
        are typically ``ImportError``, but a third-party plugin can
        raise anything from its module-level code, and Requirement
        4.9 / 11.6 demand we keep going regardless.
        """
        ep_name = getattr(ep, "name", "<unknown>")
        try:
            cls = ep.load()
        except Exception as exc:
            # Naming the plugin is the contract from Requirement 4.9
            # ("log a warning naming the offending plugin"). We
            # include the entry-point group too so a developer can
            # locate the offending package.
            logger.warning(
                "Skipping navsafe plugin %r in group %r: failed to "
                "import (%s: %s).",
                ep_name,
                _ENTRY_POINT_GROUPS.get(group, group),
                type(exc).__name__,
                exc,
            )
            return

        if not isinstance(cls, type):
            logger.warning(
                "Skipping navsafe plugin %r in group %r: entry point "
                "did not resolve to a class (got %r).",
                ep_name,
                _ENTRY_POINT_GROUPS.get(group, group),
                cls,
            )
            return

        # Use ``register`` so subclass validation (when bases are
        # importable) and idempotency apply uniformly to in-process
        # and out-of-process registrations.
        try:
            self.register(group, ep_name, cls)
        except ValueError as exc:
            logger.warning(
                "Skipping navsafe plugin %r in group %r: "
                "registration rejected (%s).",
                ep_name,
                _ENTRY_POINT_GROUPS.get(group, group),
                exc,
            )


# ---------------------------------------------------------------------------
# Module-level singleton + decorators
# ---------------------------------------------------------------------------

# Single process-wide registry. Created eagerly at module import
# because the cost is one empty-dict allocation per group. The
# decorators below close over this instance.
_REGISTRY: Registry = Registry()


def get_registry() -> Registry:
    """Return the process-wide :class:`Registry` singleton.

    Exposed so the CLI, the public API (``navsafe/__init__.py``), and
    tests resolve plugin names against the same instance the
    decorators write to (Requirement 4.6 — single source of truth).
    """
    return _REGISTRY


# Type variable for decorator signatures; preserves the decorated
# class type for IDE/typechecker users.
_ClsT = TypeVar("_ClsT", bound=type)


# Group → singular noun used in the decorator name. Plural group keys
# don't all simplify by stripping a trailing ``s`` (``policies`` →
# ``policie`` would be wrong), so we spell it out.
_GROUP_SINGULAR: Dict[str, str] = {
    "policies": "policy",
    "scorers": "scorer",
    "renderers": "renderer",
    "envs": "env",
}


def _make_register_decorator(
    group: str,
) -> Callable[..., Callable[[_ClsT], _ClsT]]:
    """Build a ``@register_<singular>(name, *, overwrite=False)`` decorator.

    Factory used by the four module-level decorators below. Each
    returned decorator delegates to ``get_registry().register`` so
    monkey-patching ``_REGISTRY`` (mostly in tests) is honored.
    """
    singular = _GROUP_SINGULAR[group]
    decorator_name = f"register_{singular}"

    def _decorator(
        name: str, *, overwrite: bool = False
    ) -> Callable[[_ClsT], _ClsT]:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{decorator_name}: name must be a non-empty string, "
                f"got {name!r}."
            )

        def _wrap(cls: _ClsT) -> _ClsT:
            get_registry().register(group, name, cls, overwrite=overwrite)
            return cls

        return _wrap

    _decorator.__name__ = decorator_name
    _decorator.__qualname__ = decorator_name
    _decorator.__doc__ = (
        f"Register a class under the {group!r} group of the "
        f"NavSafe registry.\n\n"
        f"Usage::\n\n"
        f"    @{decorator_name}(\"my_plugin\")\n"
        f"    class MyPlugin(...):\n"
        f"        ...\n\n"
        f"Idempotent re-registration with the same class is a no-op; "
        f"a conflicting class raises ``ValueError`` unless "
        f"``overwrite=True``."
    )
    return _decorator


# The four module-level decorators (Requirement 4.2). Each wraps
# ``Registry.register`` for one group; all delegate to the singleton
# returned by :func:`get_registry`, so the CLI and Python API see a
# unified registry (Requirement 4.6).
register_policy = _make_register_decorator("policies")
register_scorer = _make_register_decorator("scorers")
register_renderer = _make_register_decorator("renderers")
register_env = _make_register_decorator("envs")


__all__ = [
    "GROUPS",
    "Registry",
    "get_registry",
    "register_policy",
    "register_scorer",
    "register_renderer",
    "register_env",
]
