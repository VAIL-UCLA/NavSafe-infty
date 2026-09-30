# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""NavSafe scene-editing pipeline.

Builds the taxonomy leaves that mining cannot supply, by inserting or
re-tasking actors in an already-reconstructed host world. The pipeline's one
artifact is a **recipe** — a frozen file precise enough that anyone can rebuild
the identical scene later.

Scope (deliberately narrow):

* ``assets/``     — fetch the asset a recipe names, as a 3DGS PLY the NuRec
                    server can load; ``calibrate`` sizes it by measurement
                    (visual extent vs canonical dims) instead of by eye.
* ``placement/``  — resolve authored intent (arc / lateral / heading offset)
                    into an absolute pose against a host's own cross-section;
                    ``anchors`` names the host's landmarks (intersections,
                    crosswalks) so a spec can place relative to them.
* ``trajectory/`` — the five authoring templates. They solve an actor's INITIAL
                    CONDITION (and the goal geometry its controller needs); the
                    path itself is produced at run time by that controller
                    reacting to the ego, so nothing per-frame is stored.
* ``recipe/``     — schema, freeze, replay.
* ``review/``     — render what a human checks.

One subcommand per step in ``cli.py``, listed there in the order a scenario is
built.

Out of scope: the mining sweep, host-scene selection, reconstruction, running
the policy, tracing, scoring, coverage tables. Whether a built scenario is
correct, plausible or faithful to its leaf is a human judgment made from a
rendered episode — the only thing code verifies is file integrity.
"""
