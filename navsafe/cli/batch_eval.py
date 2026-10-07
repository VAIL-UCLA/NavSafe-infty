"""``navsafe batch-eval``: evaluate a policy on several scenes of one Arrow root.

Loops over scene indices and runs the single-scene evaluator
(:mod:`navsafe.cli.eval_entry`) once per scene, each in its own subprocess,
writing to ``<output-dir>/scene_<i>/``. For the published benchmark, where each
bundle holds one scene, use ``navsafe benchmark`` instead.

Options consumed here:

- ``--num-scenes N``          run scenes ``start .. start+N-1``
- ``--start-scene-index S``   first index for ``--num-scenes`` (default 0)
- ``--scene-indices "0 2 5"`` explicit list; takes precedence over ``--num-scenes``
- ``--output-dir DIR``        base directory (default ``outputs/batch_eval``)

Every other argument is passed to the evaluator. A failing scene does not stop
the batch; the command exits non-zero if any scene failed. A run that crashed
or wrote no metrics counts as a failure, and an episode the benchmark ended
(``scorable: false``) is reported as unscorable and left out of every mean.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# The evaluator ships inside the installed package, including wheel installs.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_ENTRY = Path(__file__).with_name("eval_entry.py")

_DEFAULT_OUTPUT_BASE = "outputs/batch_eval"


_HELP_TEXT = f"""\
usage: navsafe batch-eval (--num-scenes N [--start-scene-index S] |
                            --scene-indices "i j k ...")
                           [--output-dir DIR] [--help]
                           <evaluator options ...>

NavSafe closed-loop evaluation over multiple py123d Arrow scenes.

This wrapper loops over a set of scene indices and forwards each one to the
single-scene evaluator {_EVAL_ENTRY.name} (inside navsafe.cli), writing
each scene's artifacts under <output-dir>/scene_<i>/. Each scene runs in
its own subprocess.

Wrapper-level flags (consumed here, not forwarded):

  --num-scenes N          Run scenes start .. start+N-1 (consecutive).
  --start-scene-index S   First index for --num-scenes (default 0).
  --scene-indices "0 2 5" Explicit index list (space/comma separated);
                          takes precedence over --num-scenes.
  --output-dir DIR        Base output dir (default {_DEFAULT_OUTPUT_BASE});
                          each scene writes to DIR/scene_<i>/.
  -h, --help              Print this message and exit (no IsaacSim load).

Exactly one of --num-scenes / --scene-indices is required. Every other
argument is forwarded verbatim to each per-scene the evaluator run. To
see the full per-scene flag set, run:

    python -m navsafe.cli.eval_entry --help

A scene that fails does not abort the batch; the command exits non-zero if
any scene failed. Each scene's metrics.json is checked after the run: an
evaluator crash (termination_reason "infra_failure") or missing metrics is a
failure; a benchmark-ended episode (scorable: false) is reported as an
unscorable count and excluded from any mean.

Example:

  ACCEPT_EULA=Y navsafe batch-eval --num-scenes 3 \\
      --model-type pdm_closed --checkpoint none \\
      --py123d-data-root data --render-backend nurec_grpc \\
      --output-dir outputs/batch_eval/pdm_closed
"""


def _wants_help(argv: Sequence[str]) -> bool:
    """Return True if ``argv`` is empty or contains ``-h`` / ``--help``."""
    if not argv:
        return True
    return any(token in ("-h", "--help") for token in argv)


def _pop_valued_flag(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    """Remove ``--name VALUE`` / ``--name=VALUE`` from ``argv``.

    Returns ``(value, remaining_argv)``. If the flag appears more than
    once the last occurrence wins (mirroring argparse). A dangling
    ``--name`` with no following value is dropped with ``value=None``.
    """
    value: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == name:
            if i + 1 < len(argv):
                value = argv[i + 1]
                i += 2
            else:
                i += 1  # dangling flag, no value
            continue
        if tok.startswith(name + "="):
            value = tok[len(name) + 1 :]
            i += 1
            continue
        remaining.append(tok)
        i += 1
    return value, remaining


def _parse_scene_indices(
    num_scenes: str | None,
    start_index: str | None,
    scene_indices: str | None,
) -> list[int]:
    """Resolve the wrapper-level index selectors into a list of scene ints.

    ``--scene-indices`` (explicit list) takes precedence over
    ``--num-scenes``. Raises ``ValueError`` with an operator-facing
    message on bad / missing input.
    """
    if scene_indices is not None:
        tokens = scene_indices.replace(",", " ").split()
        if not tokens:
            raise ValueError("--scene-indices was empty")
        try:
            return [int(t) for t in tokens]
        except ValueError as exc:  # non-integer token
            raise ValueError(f"--scene-indices must be integers: {exc}") from exc

    if num_scenes is not None:
        try:
            n = int(num_scenes)
        except ValueError as exc:
            raise ValueError(f"--num-scenes must be an integer: {exc}") from exc
        if n <= 0:
            raise ValueError("--num-scenes must be positive")
        start = 0
        if start_index is not None:
            try:
                start = int(start_index)
            except ValueError as exc:
                raise ValueError(
                    f"--start-scene-index must be an integer: {exc}"
                ) from exc
        return list(range(start, start + n))

    raise ValueError(
        "missing scene selection: pass --num-scenes N (with optional "
        "--start-scene-index S) or --scene-indices \"i j k ...\""
    )


def _scene_verdict(scene_dir: Path) -> tuple[str, str]:
    """Classify a scene run that exited 0 from its ``metrics.json``.

    A mid-episode evaluator crash exits 0 (the step loop catches it and
    finalizes), so the exit code alone cannot tell a healthy scene from a
    broken one. Returns ``(verdict, detail)`` with verdict one of:

    * ``"ok"``           — scorable episode, counts toward the batch.
    * ``"unscorable"``   — the benchmark ended it (e.g. envelope_exit);
                           a valid run, but excluded from every mean.
    * ``"infra_failure"``— the evaluator/simulator broke mid-episode;
                           treated as a scene FAILURE.
    * ``"no_metrics"``   — exit 0 but no readable metrics.json; FAILURE.
    """
    path = scene_dir / "metrics.json"
    try:
        metrics = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return "no_metrics", f"cannot read {path}: {exc}"
    if not isinstance(metrics, dict):
        return "no_metrics", f"{path} is not a JSON object"
    reason = str(metrics.get("termination_reason", "unknown"))
    if reason == "infra_failure":
        return "infra_failure", str(
            metrics.get("infra_failure_error", reason))
    if not metrics.get("scorable", True):
        return "unscorable", reason
    return "ok", ""


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``navsafe batch-eval`` console script.

    Resolves the scene-index selection, then runs the evaluator once
    per scene (subprocess) with a per-scene ``--py123d-scene-index`` and
    ``--output-dir``. Returns 0 if every scene succeeded, 1 if any scene
    failed, and 2 for usage / missing-entry errors.
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])

    if _wants_help(args):
        sys.stdout.write(_HELP_TEXT)
        return 0

    if not _EVAL_ENTRY.is_file():
        sys.stderr.write(
            f"navsafe batch-eval: cannot find underlying evaluator: {_EVAL_ENTRY}\n"
            "This command requires the NavSafe source tree to be present "
            "(e.g. an editable install from the repo root).\n"
        )
        return 2

    # Consume the wrapper-level selectors + base output dir; strip any
    # user-supplied per-scene index (we re-inject one per scene).
    num_scenes, args = _pop_valued_flag(args, "--num-scenes")
    start_index, args = _pop_valued_flag(args, "--start-scene-index")
    scene_indices_arg, args = _pop_valued_flag(args, "--scene-indices")
    output_base, args = _pop_valued_flag(args, "--output-dir")
    _user_scene_idx, forwarded = _pop_valued_flag(args, "--py123d-scene-index")

    try:
        indices = _parse_scene_indices(num_scenes, start_index, scene_indices_arg)
    except ValueError as exc:
        sys.stderr.write(f"navsafe batch-eval: {exc}\n")
        return 2

    base = output_base if output_base is not None else _DEFAULT_OUTPUT_BASE

    sys.stderr.write(
        f"navsafe batch-eval: running {len(indices)} scene(s) "
        f"{indices} -> {base}/scene_<i>/\n"
    )

    failures: list[tuple[int, str]] = []     # (scene_index, why)
    unscorable: list[tuple[int, str]] = []   # (scene_index, termination_reason)
    for i in indices:
        scene_out = f"{base}/scene_{i}"
        scene_argv = [
            sys.executable,
            str(_EVAL_ENTRY),
            *forwarded,
            "--py123d-scene-index",
            str(i),
            "--output-dir",
            scene_out,
        ]
        sys.stderr.write(f"\n=== scene {i} -> {scene_out} ===\n")
        sys.stderr.flush()
        rc = subprocess.run(scene_argv).returncode
        if rc != 0:
            failures.append((i, f"rc={rc}"))
            sys.stderr.write(f"navsafe batch-eval: scene {i} FAILED (rc={rc})\n")
            continue
        # Exit 0 does not prove a healthy episode: a mid-episode evaluator
        # crash is caught, finalized as termination_reason="infra_failure"
        # and exits 0. Read the scene's metrics.json and count the drop
        # instead of silently averaging it in downstream.
        verdict, detail = _scene_verdict(Path(scene_out))
        if verdict in ("infra_failure", "no_metrics"):
            failures.append((i, f"{verdict}: {detail}"))
            sys.stderr.write(
                f"navsafe batch-eval: scene {i} FAILED ({verdict}: {detail})\n")
        elif verdict == "unscorable":
            unscorable.append((i, detail))
            sys.stderr.write(
                f"navsafe batch-eval: scene {i} UNSCORABLE ({detail}) — "
                "excluded from every mean\n")

    ok = len(indices) - len(failures) - len(unscorable)
    sys.stderr.write(
        f"\nnavsafe batch-eval: done — {ok}/{len(indices)} scene(s) succeeded"
    )
    if unscorable:
        sys.stderr.write(
            f"; {len(unscorable)} unscorable (excluded from every mean): "
            f"{[i for i, _ in unscorable]}"
        )
    if failures:
        sys.stderr.write(
            f"; failed: {[i for i, _ in failures]}\n"
        )
        return 1
    sys.stderr.write(".\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via console script
    raise SystemExit(main())
