# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""One sweep, mixed edited and unedited cells, decided by the bundle.

``scenario_meta.has_inserted_actors`` is the published claim; the recipe is
what actually inserts and what the checksums pin. Measured 2026-08-31 over the
270 published ``full_test`` bundles and the 80 benchmark recipes: 56 recipes
are ``provenance: constructed`` and insert, 24 are ``mined`` and insert
nothing, and **four bundles claim ``has_inserted_actors: true`` against a mined
recipe** — ``05bcef7a11d65c6a`` (C-7) and three R-2 hosts. 60 manifests say
true where only 56 recipes insert, and that difference is exactly those four.

So the manifest cannot be the sole authority: trusting it there would hunt for
an inserting recipe, not find one, and fail four scenarios that are fine. The
recipe wins, the run is unedited, and the disagreement is reported.
"""

from __future__ import annotations

import json

import pytest

from navsafe.benchmark.editing import autoselect as a


def _bundle(tmp_path, token, *, has_inserted=None, arrow=False):
    b = tmp_path / token
    (b / "arrow").mkdir(parents=True) if arrow else b.mkdir(parents=True)
    meta = {"token": token}
    if has_inserted is not None:
        meta["has_inserted_actors"] = has_inserted
    (b / "manifest.json").write_text(json.dumps({"token": token,
                                                 "scenario_meta": meta}))
    return b


def _recipe(dirpath, leaf, token, *, actors):
    dirpath.mkdir(parents=True, exist_ok=True)
    body = {"recipe_id": f"{leaf}/x/{token}/001", "leaf": leaf,
            "provenance": "constructed" if actors else "mined",
            "actors": {f"a{i}": {"asset": {}} for i in range(actors)}}
    p = dirpath / f"{leaf}.{token}.yaml"
    import yaml
    p.write_text(yaml.safe_dump(body))
    return p


def test_manifest_true_and_recipe_inserts_runs_edited(tmp_path):
    b = _bundle(tmp_path / "b", "aa11", has_inserted=True)
    rd = tmp_path / "recipes"
    p = _recipe(rd, "R-3", "aa11", actors=5)
    c = a.resolve_recipe(b, recipe_dir=rd)
    assert c.edited and c.path == p
    assert c.manifest_claim is True


def test_manifest_false_and_no_recipe_runs_unedited(tmp_path):
    b = _bundle(tmp_path / "b", "bb22", has_inserted=False)
    rd = tmp_path / "recipes"
    rd.mkdir()
    c = a.resolve_recipe(b, recipe_dir=rd)
    assert not c.edited and c.path is None


def test_mined_recipe_runs_unedited_even_when_manifest_claims_true(tmp_path):
    """The four real bundles: manifest says true, the recipe inserts nothing.

    The recipe is authoritative, so this must be an unedited run that names the
    disagreement — not an error and not an edit.
    """
    b = _bundle(tmp_path / "b", "05bcef7a11d65c6a", has_inserted=True)
    rd = tmp_path / "recipes"
    _recipe(rd, "C-7", "05bcef7a11d65c6a", actors=0)
    c = a.resolve_recipe(b, recipe_dir=rd)
    assert not c.edited
    assert "WRONG" in c.reason


def test_recipe_inserts_but_manifest_denies_it_still_runs_edited(tmp_path):
    """The opposite skew: the benchmark's definition wins over the claim."""
    b = _bundle(tmp_path / "b", "cc33", has_inserted=False)
    rd = tmp_path / "recipes"
    p = _recipe(rd, "V-10", "cc33", actors=1)
    c = a.resolve_recipe(b, recipe_dir=rd)
    assert c.edited and c.path == p
    assert "trusting the recipe" in c.reason


def test_claimed_edit_with_no_recipe_is_an_error(tmp_path):
    """Scoring the untouched host would silently report the wrong scenario."""
    b = _bundle(tmp_path / "b", "dd44", has_inserted=True)
    rd = tmp_path / "recipes"
    rd.mkdir()
    with pytest.raises(FileNotFoundError, match="no recipe"):
        a.resolve_recipe(b, recipe_dir=rd)


def test_explicit_recipe_always_wins(tmp_path):
    b = _bundle(tmp_path / "b", "ee55", has_inserted=False)
    rd = tmp_path / "recipes"
    _recipe(rd, "R-2", "ee55", actors=2)
    other = _recipe(tmp_path / "other", "R-4", "zz99", actors=1)
    c = a.resolve_recipe(b, recipe_dir=rd, explicit=other)
    assert c.path == other and c.reason == "explicit --recipe"


def test_arrow_child_resolves_to_the_bundle(tmp_path):
    """--py123d-data-root is the arrow dir, not the bundle."""
    b = _bundle(tmp_path / "b", "ff66", has_inserted=True, arrow=True)
    rd = tmp_path / "recipes"
    p = _recipe(rd, "I-3", "ff66", actors=2)
    c = a.resolve_recipe(b / "arrow", recipe_dir=rd)
    assert c.edited and c.path == p


def test_20s_corpus_suffix_resolves_to_the_token(tmp_path):
    """`<token>_20s` is the corpus layout; the recipe is named by the token."""
    b = tmp_path / "gg77_20s"
    (b / "arrow").mkdir(parents=True)
    rd = tmp_path / "recipes"
    p = _recipe(rd, "V-11", "gg77", actors=1)
    c = a.resolve_recipe(b / "arrow", recipe_dir=rd)
    assert c.edited and c.path == p


def test_missing_manifest_is_unknown_not_unedited(tmp_path):
    b = tmp_path / "hh88"
    b.mkdir()
    assert a.manifest_says_edited(b) is None


def test_two_recipes_for_one_token_refuses_to_guess(tmp_path):
    b = _bundle(tmp_path / "b", "ii99", has_inserted=True)
    rd = tmp_path / "recipes"
    _recipe(rd, "R-2", "ii99", actors=1)
    _recipe(rd, "R-3", "ii99", actors=1)
    with pytest.raises(ValueError, match="2 recipes match"):
        a.resolve_recipe(b, recipe_dir=rd)


def test_audit_reports_only_disagreements(tmp_path):
    rd = tmp_path / "recipes"
    ok = _bundle(tmp_path / "ok", "jj10", has_inserted=True)
    _recipe(rd, "R-3", "jj10", actors=3)
    bad = _bundle(tmp_path / "bad", "kk11", has_inserted=True)
    _recipe(rd, "R-2", "kk11", actors=0)
    rows = a.audit_manifest_claims([ok, bad], recipe_dir=rd)
    assert [r["token"] for r in rows] == ["kk11"]
    assert rows[0] == {"token": "kk11", "manifest_has_inserted_actors": True,
                       "recipe_inserts": False,
                       "recipes": ["R-2.kk11.yaml"]}
