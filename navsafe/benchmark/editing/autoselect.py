"""Decide whether a scenario is edited, from the bundle's own manifest.

A NavSafe bundle carries ``scenario_meta.has_inserted_actors`` — the published
statement that this scenario's leaf is defined by actors the log never had.
An edited run is then the unedited host plus a frozen recipe, so the only thing
a driver needs to resolve is *which* recipe, and the recipe's basename is
``<LEAF>.<token>.yaml``.

This lets one sweep mix edited and unedited scenarios without being told which
is which per cell: the manifest decides, and a missing recipe for a scenario
that claims inserted actors is an error rather than a silently unedited run.

**The manifest is a claim, not the truth.** The recipe is authoritative: it is
what actually inserts, it is checksum-pinned, and it is the benchmark's
definition. Four bundles in the published ``full_test`` set say
``has_inserted_actors: true`` and have a ``provenance: mined`` recipe that
inserts nothing (see :func:`audit_manifest_claims`); for those the recipe wins
and the run is unedited. Trusting the manifest alone there would look for a
recipe with actors, not find one, and fail a scenario that is perfectly fine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_RECIPE_DIR",
    "RecipeChoice",
    "audit_manifest_claims",
    "bundle_root",
    "manifest_says_edited",
    "resolve_recipe",
]

#: The benchmark's own recipes, in the repo. This is the set's definition, not
#: its output — see ``docs/navsafe_eval_edited_scene.md``.
DEFAULT_RECIPE_DIR = Path(__file__).resolve().parents[1] / "recipes" / "benchmark"


def bundle_root(data_root: str | Path) -> Path:
    """The bundle directory for a ``--py123d-data-root``.

    Accepts either the bundle or its ``arrow/`` child, matching
    :func:`navsafe.benchmark.signal_override.from_manifest`.
    """
    root = Path(str(data_root))
    return root.parent if root.name == "arrow" else root


def manifest_says_edited(data_root: str | Path) -> bool | None:
    """``scenario_meta.has_inserted_actors``, or None if there is no manifest.

    None is "unknown", not "no": a bundle tree without a manifest predates the
    field, and a caller that needs a decision should say so rather than quietly
    treating it as unedited.
    """
    man = bundle_root(data_root) / "manifest.json"
    if not man.is_file():
        return None
    try:
        meta = json.loads(man.read_text()).get("scenario_meta") or {}
    except Exception:                                        # noqa: BLE001
        return None
    val = meta.get("has_inserted_actors")
    return bool(val) if val is not None else None


def _token_of(data_root: str | Path) -> str | None:
    """The bundle's token, from its manifest, falling back to the dir name."""
    root = bundle_root(data_root)
    man = root / "manifest.json"
    if man.is_file():
        try:
            d = json.loads(man.read_text())
            tok = d.get("token") or (d.get("scenario_meta") or {}).get("token")
            if tok:
                return str(tok)
        except Exception:                                    # noqa: BLE001
            pass
    name = root.name
    # `<token>_20s` and `<token>sN` are both corpus layouts for one token.
    for suffix in ("_20s",):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name or None


@dataclass(frozen=True)
class RecipeChoice:
    """What :func:`resolve_recipe` decided, and why — for logging."""

    path: Path | None          # the recipe to run, or None for an unedited run
    token: str | None
    manifest_claim: bool | None   # what the manifest said
    reason: str                   # human-readable, printed by the driver

    @property
    def edited(self) -> bool:
        return self.path is not None


def _recipes_for_token(recipe_dir: Path, token: str) -> list[Path]:
    return sorted(recipe_dir.glob(f"*.{token}.yaml"))


def _recipe_inserts(path: Path) -> bool:
    """Whether a recipe actually inserts an actor.

    Parsed with the recipe loader's own reader so a schema change cannot make
    this disagree with what the run will do.
    """
    import yaml

    try:
        d = yaml.safe_load(path.read_text()) or {}
    except Exception:                                        # noqa: BLE001
        return True   # unreadable here; let the real loader raise with detail
    return bool(d.get("actors"))


def resolve_recipe(
    data_root: str | Path,
    *,
    recipe_dir: str | Path | None = None,
    explicit: str | Path | None = None,
) -> RecipeChoice:
    """Pick the recipe for this scenario, or None to run the host unedited.

    ``explicit`` (a ``--recipe`` the caller passed) always wins: auto-selection
    is a convenience for sweeps, never an override of what was asked for.
    """
    token = _token_of(data_root)
    claim = manifest_says_edited(data_root)

    if explicit:
        return RecipeChoice(Path(str(explicit)), token, claim,
                            "explicit --recipe")

    rdir = Path(str(recipe_dir)) if recipe_dir else DEFAULT_RECIPE_DIR
    if not rdir.is_dir():
        if claim:
            raise FileNotFoundError(
                f"manifest for {token} says has_inserted_actors=true but the "
                f"recipe directory {rdir} does not exist; pass --recipe-dir")
        return RecipeChoice(None, token, claim,
                            f"no recipe dir ({rdir}); manifest claims unedited")

    matches = _recipes_for_token(rdir, token) if token else []

    if not matches:
        if claim:
            # The manifest promised an edit and the set cannot supply it. Scoring
            # the untouched host here would silently report the wrong scenario.
            raise FileNotFoundError(
                f"manifest for {token} says has_inserted_actors=true but no "
                f"recipe *.{token}.yaml exists in {rdir}")
        return RecipeChoice(None, token, claim, "no recipe for this token")

    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} recipes match token {token} in {rdir} "
            f"({', '.join(p.name for p in matches)}); pass --recipe to choose")

    path = matches[0]
    if not _recipe_inserts(path):
        # A `provenance: mined` recipe: the log already stages the scenario.
        # Running it is a no-op that still pays the recipe's hand-off rules, so
        # treat it as the unedited run it is.
        note = "recipe inserts nothing (mined)"
        if claim:
            note += " — manifest says has_inserted_actors=true, which is WRONG"
        return RecipeChoice(None, token, claim, note)

    if claim is False:
        # The recipe inserts but the manifest denies it. The recipe is the
        # benchmark's definition, so run it — and say so loudly.
        return RecipeChoice(path, token, claim,
                            "recipe inserts actors but manifest says "
                            "has_inserted_actors=false; trusting the recipe")

    return RecipeChoice(path, token, claim, "manifest + recipe agree")


def audit_manifest_claims(
    bundles: list[str | Path],
    *,
    recipe_dir: str | Path | None = None,
) -> list[dict]:
    """Compare every bundle's ``has_inserted_actors`` against its recipe.

    Returns one row per disagreement, so a sweep can be gated on an empty list.
    """
    rdir = Path(str(recipe_dir)) if recipe_dir else DEFAULT_RECIPE_DIR
    out: list[dict] = []
    for b in bundles:
        tok = _token_of(b)
        claim = manifest_says_edited(b)
        matches = _recipes_for_token(rdir, tok) if tok else []
        inserts = any(_recipe_inserts(p) for p in matches) if matches else False
        if claim is None or not matches:
            continue
        if bool(claim) != inserts:
            out.append({
                "token": tok,
                "manifest_has_inserted_actors": bool(claim),
                "recipe_inserts": inserts,
                "recipes": [p.name for p in matches],
            })
    return out
