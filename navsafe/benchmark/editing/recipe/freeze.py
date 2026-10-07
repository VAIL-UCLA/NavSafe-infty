# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Freeze: resolve authored intent into absolute numbers and write the file.

Freezing is the moment a recipe stops being editable. It stamps the checksums
that make later tampering loud, and refuses to overwrite an existing file —
a frozen recipe is immutable, so a change means a new ``recipe_id``, not an
in-place edit.

Resolved numbers are stored, not the intent that produced them. ``arc: auto``
and ``opposing_lane_chain`` are queries against one host's map and one sampled
route; freezing the query instead of its answer would let a re-import of the
map silently move the actor. So what lands in the file is the absolute spawn
pose and the controller's resolved geometry, with ``authored`` kept alongside
as documentation that is never re-executed.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from navsafe.benchmark.editing.recipe.schema import Recipe, RecipeError

logger = logging.getLogger(__name__)

_HEADER = """\
# NavSafe scene-editing recipe — FROZEN, do not hand-edit.
#
# THIS FILE HOLDS NO TRAJECTORY. Each actor is `spawn` (its pose at frame 0, in
# ego frame-0 coordinates) plus `policy` — the controller that produces its path
# at run time by reacting to the ego, to the other inserted actors and to the
# logged traffic. Where it ends up on frame k is therefore not knowable here,
# and storing it would be a lie the moment the policy under test drives
# differently from the logged ego.
#
# `authored` is documentation: HOW the spawn and the goal were solved against
# this host (template, reference, laterals, the arc the layout assigned). It is
# diffable and re-authorable, and it is never re-executed at replay. In
# particular a `template: dart_out` there does not mean a scripted dart — it
# means that template solved the placement, which then became the social-force
# goal in `policy`.
#
# Each actor's `sha256` covers its spawn + policy. Editing a number here without
# re-freezing makes the recipe fail loudly at replay, which is the point.
"""


def stamp_checksums(recipe: Recipe) -> Recipe:
    """(Re)compute each actor's specification digest.

    There are no baked arrays to hash any more: an actor's path is produced at
    run time by a policy reacting to the ego. What the file determines — and
    what a digest can therefore honestly cover — is the spawn, the policy and
    the asset it names.
    """
    for actor in recipe.actors.values():
        actor._recorded_sha = "" if actor.op == "remove" else actor.digest()
    return recipe


def freeze_recipe(
    recipe: Recipe,
    path: "str | Path",
    *,
    frozen_at: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Validate, checksum and write ``recipe`` to ``path``.

    Args:
        recipe: the assembled recipe. Each actor's ``sha256`` is (re)computed
            here, so a caller never hand-maintains one.
        path: destination ``.yaml``.
        frozen_at: ISO date; defaults to today.
        overwrite: allow replacing an existing frozen file. Off by default —
            a frozen recipe is immutable, and a changed scenario deserves a
            new ``recipe_id`` rather than a silent rewrite under the old one.

    Returns:
        The path written.

    Raises:
        RecipeError: the recipe is malformed, or the file exists and
            ``overwrite`` is not set.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise RecipeError(
            f"{path} already exists. A frozen recipe is immutable — give the changed "
            f"scenario a new recipe_id, or pass overwrite=True if you really are "
            f"re-freezing the same one."
        )
    # Store resource identities, not paths belonging to the authoring host.
    # Explicit resources outside the configured banks retain their identity.
    from navsafe.benchmark import config as cfg
    for actor in recipe.actors.values():
        for field, bank, prefix in (
            ("nurec_asset_id", cfg.ASSET_BANK, "asset"),
            ("nurec_pose_bank", cfg.GAIT_BANK, "gait_bank"),
        ):
            value = getattr(actor, field)
            if value and bank and Path(value).is_absolute():
                try:
                    relative = Path(value).resolve().relative_to(bank.resolve())
                except ValueError:
                    continue
                setattr(actor, field, (Path(prefix) / relative).as_posix())
    recipe.frozen_at = frozen_at or date.today().isoformat()
    stamp_checksums(recipe)
    recipe.validate()
    recipe.verify_integrity()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + recipe.to_yaml())
    logger.info(
        "freeze_recipe: %s -> %s (%d actor(s), T=%d, dt=%.3fs)",
        recipe.recipe_id,
        path,
        len(recipe.actors),
        recipe.frames.T,
        recipe.frames.dt_s,
    )
    return path
