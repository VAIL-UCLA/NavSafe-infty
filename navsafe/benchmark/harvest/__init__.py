# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Harvested actor assets: what they are for, and what they replace.

A NavSafe scenario is 20 s of a real log reconstructed as **four 5 s NuRec
models** the render server hands off between as the ego drives. Each model only
ever saw the other vehicles from the viewpoints the logged ego actually had
during its own five seconds, and a reconstructed actor is a set of gaussians fit
to exactly those views. So the moment a policy drives differently from the log
-- slower, wider, later -- a car that the recon only observed at 60 m head-on is
suddenly 15 m away and 30 degrees off, at a viewing angle no training view
covered, and it renders as a smear. It is worst on oncoming traffic, and worst
again across a window boundary, where the handoff swaps in a model that never
saw that car at all from here.

The fix is to stop rendering those actors from the scene reconstruction. NVIDIA
Asset Harvester lifts one actor's sparse observations into a **view-consistent**
3D gaussian asset (SparseViewDiT synthesises 16 coherent views, TokenGS lifts
them), and the NRE render server can be told to swap a track's baked gaussians
for that asset. The actor then looks right from any angle, at any time, from any
distance, because its geometry no longer depends on where the logged ego drove.

What this package does NOT change: **the actor's motion.** A replace swaps
appearance only. Every actor's per-frame pose still comes from sim state through
the renderer's authoritative mirror, exactly as before, so no metric moves
because of it.

Three modules, in pipeline order:

``select``    which tracks are worth the cost -- nearest approach to the ego,
              capped, because assets are resident in renderer VRAM.
``ah``        drives the third-party Asset Harvester CLI: parse the NCore clips,
              lift the selected tracks, orient the PLYs for NuRec.
``manifest``  the one file the renderer reads: track id -> PLY path.

Where the output lives is ``cfg.ah_assets_dir(scene_id)`` -- inside the
scenario, beside its clip and its recon, harvested once and reused by every
eval. Rendering is wired in ``navsafe/render/nurec_grpc.py`` and switched on
per eval with ``--asset-harvester-replace``.

Not to be confused with ``--replace-agent-ids`` in ``eval_py123d.py``: that is
an *authoring* edit which DELETES a logged actor from sim state and inserts a
different one in its place, changing what the scenario is. This changes only
how an unchanged actor is drawn.
"""
