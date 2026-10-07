# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Process-stable hashing for reproducible asset selection.

Python's builtin :func:`hash` is salted per process (unless ``PYTHONHASHSEED``
is pinned), so ``hash(prim_path) % N`` style asset selection picks a *different*
asset for the same agent on every run. That breaks deterministic eval and makes
render artifacts (e.g. a particular black-rendering asset) impossible to
reproduce run-to-run. :func:`stable_hash` returns the same value for the same
input across processes and machines, so asset selection becomes deterministic
while staying spread out across the pool.
"""

from __future__ import annotations

import hashlib

__all__ = ["stable_hash"]


def stable_hash(text: object) -> int:
    """Return a process-stable, non-negative 64-bit hash of ``text``.

    ``text`` is coerced to ``str`` first so callers can pass ids/paths of any
    type. The result is suitable for ``stable_hash(x) % N`` selection and as a
    deterministic integer seed.
    """
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")
