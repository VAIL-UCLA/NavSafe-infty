# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""The two bundle defects that shipped to the published dataset once already.

Both were silent: an Arrow 1.5 s longer than the reconstruction it belongs to
passed the one-sided ``covers_window`` check, and a ``handoff.txt`` baked with
the builder's absolute paths survived publication and then failed inside
renderer init on every other machine.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from navsafe.benchmark.eval.bundle import (
    MAX_POSE_SNAP_US, _arrow_warnings, handoff_for_bundle,
)

T0, T1 = 1633506278200211, 1633506298200076


def _summary(first: int, last: int, frames: int = 201) -> dict:
    """An ``_arrow_summary``-shaped dict for an Arrow spanning [first, last]."""
    return {
        "path": "arrow", "present": True, "window_us": [T0, T1],
        "frames": frames, "covered_us": [first, last],
        "covered_s": round((last - first) / 1e6, 3),
        "covers_window": bool(first <= T0 + MAX_POSE_SNAP_US
                              and last >= T1 - MAX_POSE_SNAP_US),
        "overshoot_us": int(last - T1),
        "starts_early_us": int(T0 - first),
        "matches_window": bool(
            first <= T0 + MAX_POSE_SNAP_US and last >= T1 - MAX_POSE_SNAP_US
            and int(last - T1) <= MAX_POSE_SNAP_US
            and int(T0 - first) <= MAX_POSE_SNAP_US),
    }


class TestArrowWindowWarnings(unittest.TestCase):

    def test_matching_arrow_is_clean(self) -> None:
        a = _summary(T0 + 3263, T1 + 3994)
        self.assertTrue(a["matches_window"])
        self.assertEqual(_arrow_warnings(a), [])

    def test_overlong_arrow_warns_even_though_covers_window_passes(self) -> None:
        """The exact shape that shipped: 21.5 s of Arrow over a 20 s recon."""
        a = _summary(T0 + 3263, T1 + 1_503_692, frames=216)
        # The old one-sided check is satisfied — that is why this got published.
        self.assertTrue(a["covers_window"])
        self.assertFalse(a["matches_window"])
        warns = _arrow_warnings(a)
        self.assertEqual(len(warns), 1)
        self.assertIn("PAST the reconstruction window", warns[0])
        self.assertIn("1.504 s", warns[0])

    def test_short_arrow_still_warns(self) -> None:
        a = _summary(T0 + 3263, T1 - 2_000_000)
        self.assertFalse(a["covers_window"])
        self.assertTrue(any("SHORT of the window" in w for w in _arrow_warnings(a)))

    def test_overshoot_within_pose_snap_is_not_a_warning(self) -> None:
        """One pose interval past the window is normal, not a defect."""
        a = _summary(T0 + 3263, T1 + MAX_POSE_SNAP_US)
        self.assertTrue(a["matches_window"])
        self.assertEqual(_arrow_warnings(a), [])

    def test_missing_arrow_warns(self) -> None:
        self.assertTrue(any("no scenario" in w
                            for w in _arrow_warnings({"present": False})))


class TestHandoffIsDerivedNotStored(unittest.TestCase):

    def test_handoff_uses_the_readers_own_bundle_path(self) -> None:
        """The same manifest must yield different paths in different locations.

        A bundle moved (or downloaded) elsewhere has to produce a handoff that
        points at where it actually is, or the renderer's open() fails.
        """
        manifest = {
            "token": "tok", "subclips": [
                {"scene_id": "toks1", "t_start_us": 1, "t_stop_us": 2},
                {"scene_id": "toks2", "t_start_us": 2, "t_stop_us": 3},
            ],
        }
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            for root in (a, b):
                bundle = Path(root) / "tok"
                bundle.mkdir()
                (bundle / "manifest.json").write_text(json.dumps(manifest))
            ha = handoff_for_bundle(Path(a) / "tok")
            hb = handoff_for_bundle(Path(b) / "tok")

        self.assertNotEqual(ha, hb)
        self.assertIn(f"{a}/tok/offsets/toks1.json", ha)
        self.assertIn(f"{b}/tok/offsets/toks1.json", hb)
        # sid,offset_json,t0,t1 — two sub-clips, four fields each.
        parts = ha.split(";")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].split(","),
                         ["toks1", f"{a}/tok/offsets/toks1.json", "1", "2"])


if __name__ == "__main__":
    unittest.main()


class TestTheHandoffCommandAnswersForBothKindsOfHost(unittest.TestCase):
    """A host arrives mined or downloaded, and one command has to serve both.

    The seed table only knows hosts cut here. A published bundle is absent from
    it AND ships no ``clips/`` tree, so the sidecar the mined path reads does
    not exist either — ``navsafe handoff`` printed "not in scenes_500.tsv" and
    the eval driver skipped the scenario. Two shell drivers had each grown
    their own copy of the handoff string to work around it.
    """

    def _run(self, token: str, root: Path, tsv: Path) -> str:
        import io
        from contextlib import redirect_stdout

        from navsafe.benchmark.scenario_cli import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["handoff", "--token", token, "--recon-root", str(root), "--tsv", str(tsv)])
        return buf.getvalue()

    def test_a_downloaded_bundle_resolves_without_a_seed_row(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tsv = root / "scenes.tsv"
            tsv.write_text("")                       # the table has never heard of it
            bundle = root / "tok_20s"
            (bundle / "offsets").mkdir(parents=True)
            (bundle / "manifest.json").write_text(json.dumps({
                "token": "tok",
                "subclips": [{"scene_id": "toks1", "t_start_us": 10, "t_stop_us": 20},
                             {"scene_id": "toks2", "t_start_us": 20, "t_stop_us": 30}],
            }))
            out = self._run("tok", root, tsv)
        self.assertIn("NUREC_GRPC_HANDOFF", out)
        self.assertIn("toks1,", out)
        self.assertIn("toks2,", out)
        # The reader's own path, not one baked at publication time.
        self.assertIn(str(Path("tok_20s") / "offsets" / "toks1.json"), out)

    def test_a_host_that_is_neither_still_says_so(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tsv = root / "scenes.tsv"
            tsv.write_text("")
            out = self._run("nowhere", root, tsv)
        self.assertIn("not in", out)
        self.assertNotIn("NUREC_GRPC_HANDOFF", out)
