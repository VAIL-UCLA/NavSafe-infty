# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The corpus layout is a contract, and it drifted once already.

``config.py`` says the module ships no data paths of its own, but three
modules had grown their own ``/data/...`` literals anyway, and ``config``
itself carried an ``ARROW = WORK/"arrow"`` that contradicted the per-scene
``<scene_id>/arrow`` every other reader assumed.  Nothing failed, because
nothing compared them.

The layout half of this file pins the shape.  The literal half is the part
that keeps it true: a new hard-coded deployment path is a test failure, not a
review comment somebody has to notice.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from navsafe.benchmark import config as cfg

NAVSAFE_SRC = Path(cfg.__file__).parent
SCENE = "0abc123def4567s1"
HOST = "0abc123def4567_20s"


class CorpusLayout(unittest.TestCase):
    """Where each per-scenario artifact lives, relative to its corpus root."""

    def test_arrow_sits_inside_the_scene_dir(self):
        # leaves/hosts.py resolves a scene id by globbing for `arrow` and
        # reading the parent's name; a parallel arrow tree would break that.
        root = Path("/corpus")
        self.assertEqual(cfg.arrow_dir(SCENE, root), root / SCENE / "arrow")
        self.assertEqual(cfg.arrow_dir(SCENE, root).parent.name, SCENE)

    def test_clips_absorb_the_repeated_scene_id(self):
        root = Path("/corpus")
        self.assertEqual(cfg.clips_dir(SCENE, root),
                         root / SCENE / "clips" / SCENE)

    def test_recon_usdz_accepts_both_layouts_on_disk(self):
        for rel in ("output_5cam/{s}/artifacts/last.usdz", "artifacts/last.usdz"):
            with TemporaryDirectory() as tmp, self.subTest(layout=rel):
                root = Path(tmp)
                usdz = root / SCENE / rel.format(s=SCENE)
                usdz.parent.mkdir(parents=True)
                usdz.touch()
                self.assertEqual(cfg.recon_usdz(SCENE, root), usdz)

    def test_recon_usdz_is_none_when_untrained(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(cfg.recon_usdz(SCENE, Path(tmp)))

    def test_stitched_host_and_its_windows_are_siblings(self):
        # A `<token>_20s` host carries only arrow; its recons are the four
        # `<token>s1..s4` directories beside it, not children of it.
        root = Path("/corpus")
        self.assertEqual(cfg.scene_dir(HOST, root).parent,
                         cfg.scene_dir(SCENE, root).parent)

    def test_corpora_are_env_overridable(self):
        # The deployment fallbacks are a convenience, not a hard-coding.
        for var in ("NAVSAFE_CORPUS", "NAVSAFE_NAVHARD_CORPUS"):
            with self.subTest(var=var):
                self.assertIn(var, Path(cfg.__file__).read_text())


class RunLayout(unittest.TestCase):
    """Per-run outputs get an identity, so two runs cannot merge silently."""

    def test_run_dir_is_dated_and_slugged(self):
        got = cfg.run_dir("epdms-no-extcomfort", date="20260815",
                          runs=Path("/runs"))
        self.assertEqual(got, Path("/runs/20260815-epdms-no-extcomfort"))

    def test_run_dir_defaults_to_today(self):
        import datetime
        today = datetime.date.today().strftime("%Y%m%d")
        self.assertTrue(cfg.run_dir("some-run").name.startswith(f"{today}-"))

    def test_counter_names_are_refused(self):
        # The whole point: `eval/<seed>/log_replay_2s+8s/` is named after the
        # episode shape, so a re-run after a fix lands on top of the old one.
        # A run needs a name that says what it was, not which number it was.
        for bad in ("j1", "B2", "b2", "ab12", "run", "two words", "Mixed-Case",
                    "trailing-", ""):
            with self.subTest(slug=bad):
                with self.assertRaises(ValueError):
                    cfg.run_dir(bad)

    def test_descriptive_names_are_accepted(self):
        for ok in ("baseline", "epdms-no-extcomfort", "c10-oncoming-insert",
                   "navhard421-logreplay-5cam"):
            with self.subTest(slug=ok):
                self.assertTrue(cfg.run_dir(ok).name.endswith(ok))

    def test_a_run_holds_the_eval_tree_collect_already_reads(self):
        # runner/collect.py takes --eval-root and rglobs verdict.json under
        # eval/<seed>/<shape>/trace/, so <run>/eval works unchanged.
        run = cfg.run_dir("shape-check", date="20260815", runs=Path("/runs"))
        verdict = run / "eval" / "0abc" / "log_replay_2s+8s" / "trace" / "verdict.json"
        self.assertEqual(verdict.parent.parent.name, "log_replay_2s+8s")
        self.assertEqual(verdict.relative_to(run).parts[0], "eval")


class ScoredEvalNeedsACampaign(unittest.TestCase):
    """A scored eval cannot be started without naming what it is for."""

    def _run_eval(self, *argv):
        from navsafe.benchmark.world import run_eval
        return subprocess.run(
            [sys.executable, run_eval.__file__, *argv],
            capture_output=True, text=True)

    def test_run_eval_refuses_without_a_campaign(self):
        r = self._run_eval("--seed", "/nonexistent/seed.json")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--run is required", r.stderr)

    def test_dry_run_is_the_only_exemption(self):
        # --dry-run writes nothing, so it needs no campaign; it must fail on
        # the missing seed instead of on the missing --run.
        r = self._run_eval("--seed", "/nonexistent/seed.json", "--dry-run")
        self.assertNotIn("--run is required", r.stderr)

    def test_k8s_eval_refuses_without_a_campaign_before_touching_disk(self):
        from navsafe.benchmark.world import k8s_jobs
        r = subprocess.run(
            [sys.executable, k8s_jobs.__file__, "--seed", "/nonexistent",
             "--stage", "eval"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("--run is required", r.stderr)

    def test_k8s_eval_jobs_carry_the_campaign(self):
        from navsafe.benchmark.world import k8s_jobs
        seed = {"seed_id": "0abc", "family": "navsafe",
                "artifacts": {"export": "/w/export/0abc",
                              "eval": "/w/eval/0abc",
                              "ncore": "/w/ncore/0abc/clips/0abc"}}
        _, _, (script,), _ = k8s_jobs.build_eval(seed, run="probe-campaign")
        self.assertIn("--run probe-campaign", script)
        # The job's own success check and run_eval's output must agree.
        out = str(cfg.run_dir("probe-campaign") / "eval" / "0abc" / "log_replay")
        self.assertIn(f"test -f {out}/metrics.json", script)
        self.assertIn("--tag log_replay", script)


class NoHardCodedDeploymentPaths(unittest.TestCase):
    """`config.py` is the only module allowed to name a deployment path."""

    # Every root a NavSafe deployment has ever been rooted at. `bigdata`,
    # `data` and `closed-loop-e2e` are here because the first version of this
    # check listed only the two this cluster uses, and a shell script and an
    # argparse default quietly kept someone else's home directory as the
    # default for months.
    ROOTS = "avl-west|hugsim-storage|closed-loop-e2e|bigdata|data|nuplan-v1.1"
    # A quoted literal in Python, or a bare one in shell -- `.sh` is scanned
    # too, for the same reason.
    LITERAL = re.compile(rf"""(?:["'=:]|\s)/(?:{ROOTS})/""")
    # `--help` text and comments quoting an example path document it, they do
    # not resolve one, so neither is a hard-coding.
    PROSE = re.compile(r"^\s*#|help\s*=|^\s*\"\"\"|^#!")

    def _offenders(self, path: Path):
        src = path.read_text().splitlines()
        return [(i, ln.strip()) for i, ln in enumerate(src, 1)
                if self.LITERAL.search(ln) and not self.PROSE.search(ln)]

    def _scan(self, pattern: str):
        bad = {}
        for f in sorted(NAVSAFE_SRC.rglob(pattern)):
            if f.name == "config.py":
                continue
            hits = self._offenders(f)
            if hits:
                bad[f.relative_to(NAVSAFE_SRC)] = hits
        return bad

    def test_navsafe_modules_route_paths_through_config(self):
        bad = self._scan("*.py")
        self.assertEqual(bad, {}, "\n".join(
            f"{f}:{n}: {ln}  -> add a constant or helper to navsafe/config.py"
            for f, hits in bad.items() for n, ln in hits))

    def test_navsafe_scripts_take_their_paths_from_the_caller(self):
        """A shell entry point is the one a benchmark user actually types."""
        bad = self._scan("*.sh")
        self.assertEqual(bad, {}, "\n".join(
            f"{f}:{n}: {ln}  -> read it from an environment variable instead"
            for f, hits in bad.items() for n, ln in hits))


if __name__ == "__main__":
    unittest.main()
