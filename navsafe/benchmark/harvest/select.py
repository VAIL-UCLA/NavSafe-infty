# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Which of a scenario's actors are worth harvesting, and from which window.

Two independent reasons not to harvest everything:

**Cost.** Each asset is a 16-view diffusion pass plus a lifting pass. A 5 s
NavSafe window exposes ~80 tracks and the 20 s scenario more, so harvesting all
of them is a couple of GPU-hours per scenario before any evaluation runs.

**Renderer VRAM.** Every replaced actor's PLY is resident in the render server
alongside four 5 s reconstructions on one 24 GB card, and the working set is
already tight enough that ``--cache-size`` is pinned to exactly 4 with no spare.

So: a cap (10 by default), spent on the actors that matter. The ranking is
**closest approach to the ego over the whole scenario**, because that is the
same quantity that decides whether a mis-rendered actor is visible at all -- a
car 80 m up the road occupies a handful of pixels whether it is smeared or not,
while the one that ends up 8 m off the bumper fills the frame. It is also the
quantity most correlated with the failure this exists to fix: a policy that
drives differently changes the geometry most for the actors it passes closest.

Distance comes from Asset Harvester's own parse output rather than a second
read of the log. Its ``input_views/camera.json`` records ``cam_dists`` -- the
camera-to-object distance of every view it kept -- and the minimum over those
views is the closest the logged ego ever got with that actor in frame. Using it
means selection and harvesting agree by construction: a track with no usable
views is not rankable and is not selectable, rather than being selected and then
silently failing to lift.

The same track is normally parsed out of several of a scenario's four 5 s
windows. It gets harvested **once**, from the window where it was seen closest,
which is also the window with the largest, least occluded crops.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Vehicles only, by default: pedestrians and cyclists are deformable, and a
# rigid harvested asset would freeze one mid-stride -- a worse artefact than the
# smear it replaces (the gait-bank machinery in ``editing/assets/animate.py``
# exists precisely because a static asset cannot walk).
#
# Matched as a SUBSTRING of the lowercased class, because the class a parsed
# sample carries is the source dataset's full label rather than a tidy noun:
# nuPlan clips yield ``nuplanboxdetectionlabel.vehicle`` and Waymo ones
# ``wodperceptionboxdetectionlabel.type_vehicle``. Comparing whole strings
# matched neither, and a class filter that matches nothing selects nothing --
# which reads as "this scenario has no actors" rather than as a filter bug.
DEFAULT_CLASSES = ("vehicle", "truck", "bus", "trailer")
# Measured over the 379 harvested banks: the median scenario yields 7 moving
# vehicles and the mean 8.8, so a cap of 10 keeps the ordinary scenario whole
# and only bites on the crowded ones -- where the actors past the tenth are by
# construction the furthest away, and the ones whose smear costs the fewest
# pixels. The previous 30 almost never bound (3 banks of 379 reached it), so it
# capped nothing while still allowing a bank whose instances cannot be served.
DEFAULT_MAX_ASSETS = 10
# Metres of NET displacement (first observation to last) above which a track
# counts as having driven. Net, not path length: cuboid annotations jitter, and
# summing per-frame steps turns that jitter into metres -- measured on one 5 s
# window, a parked car accumulated 3.8 m of path against 0.7 m of displacement,
# while a car driving through had 41.6 m of path against 41.4 m. The gap
# between the two populations is wide, so the threshold is not delicate.
DEFAULT_MIN_MOTION_M = 2.0


@dataclass(frozen=True)
class Candidate:
    """One parsed track, as a harvesting candidate."""

    track_id: str
    label_class: str
    window: str          # the 5 s scene id its crops came from
    sample_dir: Path     # <parse_root>/<window>/<class>/<track_id>
    min_dist_m: float    # closest the logged ego came with it in frame
    n_views: int
    lwh: Sequence[float]

    @property
    def rel_path(self) -> str:
        """The ``sample_paths.json`` entry Asset Harvester expects."""
        return f"{self.label_class}/{self.track_id}"


def _read_candidate(sample_dir: Path, window: str) -> Optional[Candidate]:
    cam = sample_dir / "input_views" / "camera.json"
    if not cam.is_file():
        return None
    try:
        doc = json.loads(cam.read_text())
    except json.JSONDecodeError:
        logger.warning("unreadable camera.json in %s", sample_dir)
        return None
    dists = [float(d) for d in doc.get("cam_dists", []) if float(d) > 0.0]
    frames = doc.get("frame_filenames") or []
    if not dists or not frames:
        return None
    return Candidate(
        track_id=sample_dir.name,
        label_class=sample_dir.parent.name,
        window=window,
        sample_dir=sample_dir,
        min_dist_m=min(dists),
        n_views=len(frames),
        lwh=[float(v) for v in doc.get("object_lwh", (1.0, 1.0, 1.0))],
    )


def scan(parse_root: Path, windows: Sequence[str]) -> List[Candidate]:
    """Every track Asset Harvester managed to parse, across a scenario's windows.

    ``parse_root/<window>/<class>/<track_id>/input_views/camera.json`` is the
    layout ``ncore_parser`` writes; anything it could not crop (fully occluded,
    too small, never in frame) simply has no directory and cannot be selected.
    """
    out: List[Candidate] = []
    for window in windows:
        root = Path(parse_root) / window
        if not root.is_dir():
            logger.warning("no parse output for window %s under %s", window, parse_root)
            continue
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for sample_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
                cand = _read_candidate(sample_dir, window)
                if cand is not None:
                    out.append(cand)
    return out


def displacement_of(motion: Dict[str, Dict[str, object]], track_id: str) -> float:
    """Largest per-clip net displacement after merging scenario windows.

    Absent from the table means never observed as a cuboid, which cannot happen
    for something the parser cropped -- treat it as stationary rather than
    guessing, so an unexplained gap costs an asset rather than spending one.
    """
    rec = motion.get(str(track_id))
    return float(rec.get("displacement_m", 0.0)) if rec else 0.0


def choose(
    candidates: Sequence[Candidate],
    *,
    max_assets: int = DEFAULT_MAX_ASSETS,
    classes: Sequence[str] = DEFAULT_CLASSES,
    min_views: int = 2,
    motion: Optional[Dict[str, Dict[str, object]]] = None,
    min_motion_m: float = DEFAULT_MIN_MOTION_M,
) -> List[Candidate]:
    """The nearest ``max_assets`` MOVING tracks, one entry per track id.

    ``min_views`` guards the degenerate input: one view is not multi-view, and
    SparseViewDiT asked to hallucinate an entire vehicle from a single distant
    crop produces a confident, wrong car -- which is harder to notice in a
    rendered frame than the smear it replaced.

    ``motion`` (from :mod:`navsafe.benchmark.harvest.motion`) restricts the
    harvest to actors that actually drove, and it is the single biggest thing
    that decides whether this feature helps or hurts.

    Replacing a PARKED car is a losing trade. The ego drives past it, so the
    reconstruction saw it across a wide arc and holds real pixels for every
    angle a policy is likely to want; a lifted asset can only be worse there,
    and measured on 2b7bf25209dd5705 it is -- a parked FedEx truck whose
    branding is legible in the reconstruction came back as a plain grey box
    truck. A MOVING car is the opposite: it travels with or against the ego, so
    the relative geometry barely changes for the whole clip and the
    reconstruction has almost no angular coverage of it. That is the actor that
    falls apart when a policy arrives early, late or wide, and the only one
    worth an asset and its share of renderer VRAM.

    It also makes the budget fit: one 5 s window held 57 tracks of which 5 had
    driven anywhere.

    ``motion=None`` keeps parked cars, which is only useful for diagnosing.
    """
    allowed = tuple(c.lower() for c in classes) if classes else None
    best: Dict[str, Candidate] = {}
    for cand in candidates:
        cls = cand.label_class.lower()
        if allowed is not None and not any(a in cls for a in allowed):
            continue
        if cand.n_views < min_views:
            continue
        if motion is not None and displacement_of(motion, cand.track_id) < min_motion_m:
            continue
        prev = best.get(cand.track_id)
        if prev is None or cand.min_dist_m < prev.min_dist_m:
            best[cand.track_id] = cand
    ranked = sorted(best.values(), key=lambda c: (c.min_dist_m, c.track_id))
    kept = ranked[:max_assets]
    if len(ranked) > len(kept):
        logger.info(
            "selection: %d track(s) parsed, keeping the %d nearest "
            "(cut at %.1f m; furthest dropped was %.1f m)",
            len(ranked), len(kept), kept[-1].min_dist_m, ranked[-1].min_dist_m)
    else:
        logger.info("selection: %d track(s) parsed, keeping all", len(ranked))
    return kept


def write_sample_paths(selected: Sequence[Candidate], data_root: Path) -> Path:
    """Stage the chosen samples as one Asset Harvester ``--data-root``.

    The samples were parsed per 5 s window into separate trees; lifting wants a
    single root with one ``sample_paths.json``. The per-track directories are
    symlinked rather than copied -- they hold up to 16 JPEGs plus masks each,
    and the parse tree is scratch that outlives the lift by minutes.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    for cand in selected:
        link = data_root / cand.label_class / cand.track_id
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(cand.sample_dir.resolve(), target_is_directory=True)
    out = data_root / "sample_paths.json"
    out.write_text(json.dumps(
        {"samples": [c.rel_path for c in selected]}, indent=2) + "\n")
    return out
