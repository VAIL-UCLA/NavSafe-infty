"""Choose the recipe for a scenario bundle.

A sweep mixes edited and unedited scenarios, so the recipe is resolved per
bundle: the one named ``<event-type-id>.<token>.yaml`` in the recipe
directory, if it inserts actors; otherwise the scenario runs as logged.

The bundle manifest's ``scenario_meta.has_inserted_actors`` is used as a
cross-check only. The recipe is authoritative, because it is what the
evaluation applies and what the checksums pin. A manifest that promises
inserted actors with no recipe to supply them is an error, since running the
logged scene would silently evaluate a different scenario.
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

#: The benchmark's recipes, shipped with the package.
DEFAULT_RECIPE_DIR = Path(__file__).resolve().parents[1] / "recipes" / "benchmark"


def bundle_root(data_root: str | Path) -> Path:
    """Return the bundle directory, given the bundle or its ``arrow/`` child."""
    root = Path(str(data_root))
    return root.parent if root.name == "arrow" else root


def manifest_says_edited(data_root: str | Path) -> bool | None:
    """Return the manifest's ``has_inserted_actors``, or None if it is not stated.

    None means unknown, not False: a bundle without the field makes no claim.
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
    """Return the bundle's token from its manifest, else from the directory name."""
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
    # A reconstruction corpus names the directory `<token>_20s`.
    for suffix in ("_20s",):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name or None


@dataclass(frozen=True)
class RecipeChoice:
    """The outcome of :func:`resolve_recipe`, with the reason for logging."""

    path: Path | None             # recipe to apply; None runs the logged scene
    token: str | None
    manifest_claim: bool | None   # the manifest's has_inserted_actors
    reason: str

    @property
    def edited(self) -> bool:
        return self.path is not None


def _recipes_for_token(recipe_dir: Path, token: str) -> list[Path]:
    return sorted(recipe_dir.glob(f"*.{token}.yaml"))


def _recipe_inserts(path: Path) -> bool:
    """Return whether the recipe declares any actor."""
    import yaml

    try:
        d = yaml.safe_load(path.read_text()) or {}
    except Exception:                                        # noqa: BLE001
        return True   # unreadable: let the recipe loader report the error
    return bool(d.get("actors"))


def resolve_recipe(
    data_root: str | Path,
    *,
    recipe_dir: str | Path | None = None,
    explicit: str | Path | None = None,
) -> RecipeChoice:
    """Resolve the recipe for a bundle; ``path`` is None for an unedited run.

    ``explicit`` is a recipe the caller chose and is returned unchanged.
    Raises if the manifest claims inserted actors that no recipe supplies, or
    if several recipes match the token.
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
            # Running the logged scene here would evaluate a different scenario.
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
        # The event is already in the log, so the recipe has nothing to apply.
        note = "recipe inserts nothing (mined)"
        if claim:
            note += " — manifest says has_inserted_actors=true, which is WRONG"
        return RecipeChoice(None, token, claim, note)

    if claim is False:
        # The recipe wins over the manifest; the disagreement goes in the log.
        return RecipeChoice(path, token, claim,
                            "recipe inserts actors but manifest says "
                            "has_inserted_actors=false; trusting the recipe")

    return RecipeChoice(path, token, claim, "manifest + recipe agree")


def audit_manifest_claims(
    bundles: list[str | Path],
    *,
    recipe_dir: str | Path | None = None,
) -> list[dict]:
    """Return one row per bundle whose manifest disagrees with its recipe."""
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
