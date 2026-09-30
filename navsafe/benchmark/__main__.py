# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``python -m navsafe.benchmark`` / the ``navsafe`` console script."""

import sys

from navsafe.benchmark.scenario_cli import main

if __name__ == "__main__":
    sys.exit(main())
