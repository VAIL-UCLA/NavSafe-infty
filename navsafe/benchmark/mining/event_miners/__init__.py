# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Per-leaf selection predicates, registered by name for the event type manifests."""

from navsafe.benchmark.mining.event_miners.base import (  # noqa: F401
    Candidate,
    Scenario,
    available_miners,
    get_miner,
    register_miner,
    trained,
    window_of,
)

# Importing each module is what registers its miner.
from navsafe.benchmark.mining.event_miners import tag_miner  # noqa: F401,E402
from navsafe.benchmark.mining.event_miners import two_way  # noqa: F401,E402
from navsafe.benchmark.mining.event_miners import v10_merge  # noqa: F401,E402

__all__ = [
    "Candidate", "Scenario", "available_miners", "get_miner",
    "register_miner", "trained", "window_of",
]
