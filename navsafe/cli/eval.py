"""``navsafe eval``: closed-loop evaluation of a policy on one scenario.

A thin launcher. The evaluator itself is :mod:`navsafe.cli.eval_entry`, which
starts Isaac Sim when it is imported, so it runs in a subprocess and ``--help``
is answered here without loading the simulator.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

_EVAL_ENTRY = Path(__file__).with_name("eval_entry.py")

_HELP_TEXT = """\
usage: navsafe eval [--help] <evaluator options ...>

Closed-loop evaluation of a policy on one py123d Arrow scenario.

Every option is passed to the evaluator. The common ones are described in
docs/navsafe_eval.md; for the complete list, run

    python -m navsafe.cli.eval_entry --help

which needs Isaac Sim to be installed.

Example:

  ACCEPT_EULA=Y navsafe eval \\
      --py123d-data-root "$NAVSAFE_DATA_ROOT/full_test/<token>/arrow" \\
      --render-backend nurec_grpc \\
      --model-type pdm_closed --checkpoint none \\
      --output-dir output/eval
"""


def _wants_help(argv: Sequence[str]) -> bool:
    """Return True if ``argv`` is empty or contains ``-h`` / ``--help``."""
    return not argv or any(token in ("-h", "--help") for token in argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator in a subprocess and return its exit code."""
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if _wants_help(args):
        sys.stdout.write(_HELP_TEXT)
        return 0
    return subprocess.run([sys.executable, str(_EVAL_ENTRY), *args]).returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
