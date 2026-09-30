"""``navsafe-eval`` console-script wrapper.

Closed-loop evaluation of a policy on a py123d Arrow scenario. This is a
thin wrapper — it does not re-implement any evaluation logic; it forwards
all arguments to the maintained evaluator ``scripts/tools/eval_py123d.py``,
which reads a py123d Arrow data root directly and builds the unified
:class:`navsafe.env.NexusSimEnv` (NuRec gRPC backend). It is the
Python-console-script analogue of ``scripts/evaluator/run_eval.sh``.

History: the original ``navsafe-eval`` delegated to
``scripts/evaluator/eval_entry.py`` (which built the retired
``ScenarioReplayEnv``). That entry script was removed in the py123d
migration; this wrapper was re-pointed at ``eval_py123d.py`` so the
``navsafe-eval`` command works again.

``--help`` short-circuit: ``eval_py123d.py`` imports
``isaaclab.app.AppLauncher`` at module top, before its argparse runs, so
``eval_py123d.py --help`` itself requires IsaacSim. To keep
``navsafe-eval --help`` cheap and IsaacSim-free, this wrapper handles
``-h`` / ``--help`` itself and points the user at the underlying script
for the full flag set.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

# ``__file__`` is ``<repo>/navsafe/cli/eval.py``; the repo root is three
# ``parents`` up (eval.py → cli/ → navsafe/ package → repo root). The
# underlying evaluator lives under the source tree, so this command needs
# the repo checkout present (the same editable-install workflow as the
# rest of NexusSim — a bare wheel without the source tree cannot run it).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_ENTRY = Path(__file__).with_name("eval_entry.py")


_HELP_TEXT = f"""\
usage: navsafe-eval [--help] <eval_py123d.py args ...>

NexusSim closed-loop evaluation (single py123d Arrow scenario).

This wrapper forwards every argument verbatim to the maintained
evaluator, {_EVAL_ENTRY.name} (inside navsafe.cli), which reads a
py123d Arrow data root directly and builds the unified NexusSimEnv. It is
the console-script equivalent of scripts/evaluator/run_eval.sh.

Wrapper-level flags:

  -h, --help  Print this message and exit (without loading IsaacSim).

All other arguments are forwarded verbatim to the underlying evaluator.
To see the full flag set, run:

    python scripts/tools/eval_py123d.py --help

(Requires IsaacSim to be installed — see README.)

Example:

  ACCEPT_EULA=Y navsafe-eval \\
      --model-type pdm_closed --checkpoint none \\
      --py123d-data-root data --render-backend nurec_grpc \\
      --output-dir outputs/eval
"""


def _wants_help(argv: Sequence[str]) -> bool:
    """Return True if ``argv`` is empty or contains ``-h`` / ``--help``."""
    if not argv:
        return True
    return any(token in ("-h", "--help") for token in argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``navsafe-eval`` console script.

    Forwards all arguments to ``scripts/tools/eval_py123d.py`` in a fresh
    subprocess (matching ``run_eval.sh``'s ``exec python eval_py123d.py``
    semantics; the subprocess fully isolates IsaacSim's global state and
    its ``simulation_app.close()`` teardown). Returns the underlying
    script's exit code, or 0/2 for the wrapper-level help / missing-entry
    paths.
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])

    if _wants_help(args):
        sys.stdout.write(_HELP_TEXT)
        return 0

    if not _EVAL_ENTRY.is_file():
        sys.stderr.write(
            f"navsafe-eval: cannot find underlying evaluator: {_EVAL_ENTRY}\n"
            "This command requires the NexusSim source tree to be present "
            "(e.g. an editable install, `pip install -e .` / `uv sync`, from "
            "the repo root).\n"
        )
        return 2

    # Forward to eval_py123d.py with the same interpreter running this
    # console script (the venv python that has IsaacSim installed). The
    # child inherits the environment (ACCEPT_EULA, PY123D_DATA_ROOT, etc.).
    completed = subprocess.run([sys.executable, str(_EVAL_ENTRY), *args])
    return completed.returncode


if __name__ == "__main__":  # pragma: no cover - exercised via console script
    raise SystemExit(main())
