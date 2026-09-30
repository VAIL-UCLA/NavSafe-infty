# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Replay: a frozen recipe in, ``scenario_edits`` out.

This is the surface a benchmark user touches::

    recipe.yaml
      │  replay.py — verify checksums, build the edit spec
      ▼
    env_cfg.scenario_edits = [{"tool": "spawn_reactive_actor", ...}]
      │  eval_py123d.py switches to NexusSimEditEnv when edits are present
      ▼  (the stock env stays pristine)
    apply_scenario_edits(sd, edits)   — once at load, idempotent
      ▼
    sd["tracks"][id]   — the actor's DECLARATION (asset, dims, class)
    sd["metadata"]["navsafe_reactive"]  — its spawn + policy, for the manager

The declaration is an ordinary track, so asset insertion, collision dims and
the renderer's track list are unchanged. Its motion is not in the track: the
traffic manager owns that and publishes it per frame into ``pose_overrides``,
which the env merges over the logged state before anything reads it.

**Pairing.** ``e_plus`` is the built scenario; ``e_zero`` is its counterfactual,
which drops the actors named in ``pair.e0_removes`` — the ones whose presence
defines the leaf. Dropping an actor at *this* level (rather than deleting a
track later) is what makes the control honest: for a ``relocate`` the host's own
logged actor simply stays as it was, and for an ``insert`` nothing is added.
Removing background traffic instead would isolate nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from navsafe.benchmark.editing.recipe.schema import (
    FROZEN_ASSET_ID_KEY,
    Recipe,
    RecipeError,
    load_recipe,
)

logger = logging.getLogger(__name__)

VARIANTS = ("e_plus", "e_zero")


def _rebase_asset(spec: Dict[str, Any], name: str) -> None:
    """Point ``nurec_asset_id`` at the local asset bank, if one is configured.

    A recipe records the absolute path its asset was baked from, so a published
    recipe names a directory that exists only on the machine that froze it.
    ``NAVSAFE_ASSET_BANK`` (``cfg.ASSET_BANK``) rebases the basename onto the
    reader's own bank, which is what lets the same frozen recipe run anywhere.

    Deliberately AFTER ``actor.digest()``: the recipe's identity is the asset's
    sha256, not where the file sits, so rebasing must not disturb the checksum
    a relocated bank would otherwise be able to launder.

    But the digest is checked TWICE — again in ``spawn_reactive_actor``, at the
    last gate before the sim — and ``replay_spec`` puts ``nurec_asset_id`` in
    the hashed payload. A rebased path therefore passed the first gate and
    failed the second, which made ``NAVSAFE_ASSET_BANK`` unusable with any
    recipe that has actors: the run died at env reset with "sha256 mismatch —
    the spawn, the policy or the asset was edited after freezing". So the
    frozen path is preserved under ``FROZEN_ASSET_ID_KEY`` for that second gate
    to hash instead; it is dropped there and never reaches the simulator.

    The existence check is only made when rebasing is switched on. An asset
    missing from a freshly unpacked bank is the likely first-run mistake, and
    without it the failure surfaces inside the render server, a long way from
    the directory that caused it.

    Leaving the bank UNSET is refused rather than honoured. The old behaviour
    -- fall through and use the frozen absolute path -- is correct on exactly
    one machine, the one that froze the recipe, and that machine is ours: the
    run resolves, renders our own library and reports nothing, so the reader
    who forgot the variable is the only one who ever finds out, and only
    indirectly. A benchmark that is published has to fail on the machine that
    is misconfigured, not on everybody else's.
    """
    from navsafe.benchmark import config as cfg

    asset_id = spec.get("nurec_asset_id")
    if not asset_id:
        return
    if not cfg.ASSET_BANK:
        raise RecipeError(
            f"actor {name!r}: this recipe inserts the asset "
            f"{Path(str(asset_id)).name!r}, but NAVSAFE_ASSET_BANK is unset. The "
            f"resource needs a local root. Set NAVSAFE_DATA_ROOT to the HF "
            f"snapshot root or NAVSAFE_ASSET_BANK to its asset folder.")
    rebased = cfg.ASSET_BANK / Path(str(asset_id)).name
    if not rebased.is_file():
        raise RecipeError(
            f"actor {name!r}: NAVSAFE_ASSET_BANK is {cfg.ASSET_BANK}, but the "
            f"recipe's asset {Path(str(asset_id)).name!r} is not there. Point it "
            f"at the unpacked NavSafe asset bank — the published `asset/` folder, "
            f"whose PLYs carry METRIC dimensions because the renderer normalises "
            f"them itself.")
    if str(rebased) == str(asset_id):
        return
    # AND IT HAS TO BE THE SAME ASSET, not just the same file name. The digest
    # pins `asset_sha256` as a recorded value; nothing re-read the file, so
    # rebasing onto a bank whose PLYs differ from the ones the recipe was frozen
    # against passed every gate and rendered something else. Measured
    # 2026-09-04: six re-calibrated cars were written to the authoring library
    # while NAVSAFE_ASSET_BANK pointed at the published copy, and an hour of
    # renders came back showing the OLD assets with nothing to say so — the
    # reviewer spotted it, not the pipeline. The gait bank one level down has
    # always re-hashed its manifest for exactly this reason.
    recorded = str(spec.get("asset_sha256", ""))
    if recorded:
        import hashlib
        actual = hashlib.sha256(rebased.read_bytes()).hexdigest()
        if actual != recorded:
            raise RecipeError(
                f"actor {name!r}: {rebased} is not the asset this recipe was "
                f"frozen with (recorded {recorded[:12]}…, found {actual[:12]}…). "
                f"Same file name, different geometry — it would render a "
                f"different object at a different size. Re-freeze the recipe "
                f"against this bank, or point NAVSAFE_ASSET_BANK at the library "
                f"it was frozen against.")
    spec[FROZEN_ASSET_ID_KEY] = str(asset_id)
    spec["nurec_asset_id"] = str(rebased)


def _rebase_gait_bank(spec: Dict[str, Any], name: str) -> None:
    """Point ``nurec_pose_bank`` at the local gait bank, if one is configured.

    Same problem and same fix as :func:`_rebase_asset`, one level up: a recipe
    records the absolute directory its pose bank was baked into, so a published
    one names a path that exists only on the machine that froze it.
    ``NAVSAFE_GAIT_BANK`` (``cfg.GAIT_BANK``) rebases the bank's own directory
    name onto the reader's download.

    Unlike the asset path this one is NOT hashed (see ``_UNHASHED_PATH_KEYS``),
    so no frozen-value key is needed: what the digest pins is
    ``pose_bank_sha256``, the hash of the bank's own manifest. Checking it here
    is what stops a bank baked from a different motion -- a running gait where
    the leaf froze a walk -- being accepted silently. The manifest records each
    phase PLY's hash in turn, so one cheap check covers the geometry too.
    """
    import hashlib

    from navsafe.benchmark import config as cfg

    bank = spec.get("nurec_pose_bank")
    if not bank:
        return
    root = Path(str(bank))
    # Unset is refused, not honoured -- see :func:`_rebase_asset`. This one was
    # worse than the asset path until 2026-09-07: ``cfg.GAIT_BANK`` carried a
    # default of ``asset/gait_banks``, so a reader who set
    # nothing was pointed at this cluster's library by the CONFIG, not merely
    # by the recipe.
    if not cfg.GAIT_BANK:
        raise RecipeError(
            f"actor {name!r}: this recipe walks on the baked pose bank "
            f"{root.name!r}, but NAVSAFE_GAIT_BANK is unset. The path frozen in "
            f"the recipe belongs to the machine that baked it. Point "
            f"NAVSAFE_GAIT_BANK at the unpacked NavSafe `gait_bank/` folder.")
    if root.parent != cfg.GAIT_BANK:
        root = cfg.GAIT_BANK / root.name
    manifest = root / "bank.json"
    if not manifest.is_file():
        raise RecipeError(
            f"actor {name!r}: the recipe wants the gait bank {root.name!r}, but "
            f"{manifest} is not there. Download the NavSafe gait banks and point "
            f"NAVSAFE_GAIT_BANK at them (currently {cfg.GAIT_BANK}), or bake one "
            f"with `navsafe assets animate {root.name}`. Dropping "
            f"`nurec_pose_bank:` from the recipe reverts that actor to the static "
            f"asset, which changes what the leaf tests.")
    recorded = str(spec.get("pose_bank_sha256", ""))
    if recorded:
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if actual != recorded:
            raise RecipeError(
                f"actor {name!r}: gait bank {root} does not match the one this "
                f"recipe was frozen with (recorded {recorded[:12]}…, found "
                f"{actual[:12]}…). It was baked from a different asset, motion or "
                f"phase count, and would put a different gait in the scenario.")
    spec["nurec_pose_bank"] = str(root)


def edits_from_recipe(recipe: Recipe, *, variant: str = "e_plus") -> List[Dict[str, Any]]:
    """Build the ``scenario_edits`` list for one variant of a recipe.

    Checksums are verified here, before anything reaches the env — a corrupted
    recipe must fail at load rather than halfway through an episode.

    Args:
        recipe: a loaded recipe.
        variant: ``"e_plus"`` (the built scenario) or ``"e_zero"`` (the paired
            counterfactual, with ``pair.e0_removes`` actors left out).

    Returns:
        A list holding at most one ``spawn_reactive_actor`` spec, carrying every
        surviving actor. Empty when the variant edits nothing at all — an
        ``e_zero`` whose every actor is dropped is just the untouched host,
        which is exactly the control.

    Raises:
        RecipeError: unknown variant, or an integrity check fails.
    """
    if variant not in VARIANTS:
        raise RecipeError(f"unknown recipe variant {variant!r}; expected one of {list(VARIANTS)}")

    recipe.validate()
    dropped = set(recipe.pair.get("e0_removes", []) or []) if variant == "e_zero" else set()

    actors: List[Dict[str, Any]] = []
    for name, actor in recipe.actors.items():
        if name in dropped:
            continue
        if actor.op != "remove":
            actor.verify()
        spec = actor.replay_spec()
        # The digest travels with the spec so the env can re-check it at the
        # last gate, after this module is out of the picture.
        if actor.op != "remove":
            spec["sha256"] = actor.digest()
        _rebase_asset(spec, name)
        _rebase_gait_bank(spec, name)
        actors.append(spec)

    if not actors:
        logger.info(
            "edits_from_recipe: %s [%s] edits nothing (all actors dropped) — "
            "the untouched host IS the control",
            recipe.recipe_id,
            variant,
        )
        return []

    edit = {
        "tool": "spawn_reactive_actor",
        "recipe_id": recipe.recipe_id,
        "variant": variant,
        "frames": recipe.frames.to_dict(),
        "actors": actors,
    }
    logger.info(
        "edits_from_recipe: %s [%s] -> %d actor(s) %s (T=%d, dt=%.3fs)",
        recipe.recipe_id,
        variant,
        len(actors),
        [a["name"] for a in actors],
        recipe.frames.T,
        recipe.frames.dt_s,
    )
    return [edit]


def edits_from_recipe_file(
    path: "str | Path", *, variant: str = "e_plus"
) -> Tuple[Recipe, List[Dict[str, Any]]]:
    """Load a frozen recipe and build its edit spec.

    Returns:
        ``(recipe, edits)`` — the recipe too, because a caller normally needs
        its ``host``/``ego``/``frames`` to configure the run it is about to
        edit.
    """
    recipe = load_recipe(Path(path), verify=True)
    return recipe, edits_from_recipe(recipe, variant=variant)
