# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Review cards — making a human verdict cheap.

The pipeline does not decide whether a built scenario is correct, plausible or
faithful to its leaf. A human does, from a rendered episode. So the pipeline's
job is to make that review fast: per candidate, render the pair side by side,
print the placement numbers a reviewer needs to sanity-check (lane width,
closing time, spawn arc), and attach that leaf's checklist.

At 83 scenarios and a few seconds of video each, one review pass is a few hours
and parallelises across people — but only if the reviewer never has to go
looking for a number.

The reviewer records a verdict plus a **reason code**. Reason codes are a fixed
vocabulary on purpose: they aggregate into template fixes rather than
per-episode patches, so ten "wrong_lane" verdicts are one placement bug, not
ten hand edits.

``topdown.gif`` is produced here, from the scenario dict alone — no GPU. The
camera view ``cam_f0.gif`` comes from an eval run against the reconstruction;
pass its path in when you have one. The card says plainly when it is missing,
because a reviewer cannot judge scale, grounding or appearance harmony from a
bird's-eye plot.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from navsafe.benchmark.editing.recipe.replay import edits_from_recipe
from navsafe.benchmark.editing.recipe.schema import Recipe
from navsafe.benchmark.leaves import LEAVES_DIR
from navsafe.scenario.edits import apply_scenario_edits

logger = logging.getLogger(__name__)

#: The reviewer's fixed vocabulary. Anything not covered belongs in the free
#: text, and if it recurs it earns a code.
REASON_CODES: Dict[str, str] = {
    "ok": "passes the leaf's checklist",
    "asset_missing": "the asset does not render at all",
    "asset_scale": "renders, but at the wrong size",
    "asset_orientation": "sideways, reversed, or not upright",
    "asset_grounding": "floating above or sunk into the road",
    "appearance_mismatch": "renders, but does not sit in the scene's lighting or resolution",
    "wrong_lane": "the actor is not in the lane the leaf requires",
    "placement_implausible": "no real object would be there",
    "unavoidable_at_spawn": "contact is already unavoidable when the episode starts",
    "no_legal_evasion": "the only way out is itself illegal",
    "event_missed": "the policy never meets the event (outside the arrival window)",
    "offscreen_pose_visible": "an invalid-frame actor is parked somewhere a camera can see",
    "kinematics_implausible": "the motion is wrong for that embodiment",
    "leaf_mismatch": "renders fine, but does not read as this leaf",
    "host_unsuitable": "the host scene cannot support this leaf",
}

# Cap the animation length; a reviewer watches a few seconds, not a minute.
_MAX_GIF_FRAMES = 60
_BOX_FALLBACK = (4.5, 1.9)


@dataclass
class ReviewCard:
    """Everything written for one candidate."""

    directory: Path
    card_path: Path
    topdown_gif: Optional[Path] = None
    cam_gif: Optional[Path] = None
    numbers: Dict[str, Any] = field(default_factory=dict)


def _fresh(sd: dict) -> dict:
    """Copy the mutable parts of a scenario, sharing the read-only map.

    ``map_features`` is thousands of polylines and nothing here edits it;
    deep-copying it per variant is the difference between a card in seconds and
    a card in minutes.
    """
    return {
        **sd,
        "tracks": copy.deepcopy(sd.get("tracks", {})),
        "metadata": copy.deepcopy(sd.get("metadata", {})),
    }


def _variant_scenarios(recipe: Recipe, sd: dict) -> Dict[str, dict]:
    """``{"e_zero": sd, "e_plus": sd}`` — the pair, side by side."""
    out = {}
    for variant in ("e_zero", "e_plus"):
        out[variant] = apply_scenario_edits(
            _fresh(sd), edits_from_recipe(recipe, variant=variant)
        )
    return out


def _track_boxes(sd: dict, frame: int):
    """(cx, cy, heading, length, width, tag) for every valid track at ``frame``."""
    sdc_id = str((sd.get("metadata") or {}).get("sdc_id", "ego"))
    rows = []
    for tid, track in (sd.get("tracks") or {}).items():
        state = track.get("state", {})
        pos = np.asarray(state.get("position"))
        if pos.ndim != 2 or frame >= pos.shape[0]:
            continue
        valid = np.asarray(state.get("valid", np.ones(pos.shape[0], bool)))
        if not bool(valid[frame]):
            continue
        heading = float(np.asarray(state.get("heading", np.zeros(pos.shape[0])))[frame])
        length = float(np.asarray(state.get("length", [_BOX_FALLBACK[0]]))[
            min(frame, len(np.asarray(state.get("length", [0]))) - 1)
        ] or _BOX_FALLBACK[0])
        width = float(np.asarray(state.get("width", [_BOX_FALLBACK[1]]))[
            min(frame, len(np.asarray(state.get("width", [0]))) - 1)
        ] or _BOX_FALLBACK[1])
        meta = track.get("metadata") or {}
        if tid == sdc_id:
            tag = "ego"
        elif meta.get("navsafe_actor"):
            tag = "edited"
        else:
            tag = "background"
        rows.append((float(pos[frame, 0]), float(pos[frame, 1]), heading, length, width, tag))
    return rows


def _corners(cx, cy, heading, length, width):
    dx, dy = length / 2.0, width / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    rot = np.array(
        [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
    )
    return local @ rot.T + np.array([cx, cy])


def render_topdown_gif(
    recipe: Recipe,
    sd: dict,
    path: "str | Path",
    *,
    margin_m: float = 25.0,
    fps: int = 10,
) -> Path:
    """Animate the e0 / e+ pair from above, side by side.

    The map is culled to a bounding box around the action **once**, before the
    loop: projecting a whole city's lanes on every frame is what made an
    earlier BEV renderer spend 87% of its time drawing polylines nobody sees.
    """
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    path = Path(path)
    variants = _variant_scenarios(recipe, sd)
    T = recipe.frames.T
    stride = max(1, int(np.ceil(T / _MAX_GIF_FRAMES)))
    frames = list(range(0, T, stride))

    # The view is framed by the EGO and the EDITED actors only, over their
    # valid frames. Background tracks are drawn but never allowed to expand it:
    # a single invalid-frame pose parked far off (the renderer's stand-in for
    # "does not exist this frame") would otherwise zoom the whole card out to
    # nothing.
    pts = []
    sdc_id = str((sd.get("metadata") or {}).get("sdc_id", "ego"))
    for variant_sd in variants.values():
        for tid, track in (variant_sd.get("tracks") or {}).items():
            meta = track.get("metadata") or {}
            if tid != sdc_id and not meta.get("navsafe_actor"):
                continue
            state = track.get("state", {})
            pos = np.asarray(state.get("position"))
            if pos.ndim != 2 or pos.shape[1] < 2:
                continue
            valid = np.asarray(state.get("valid", np.ones(pos.shape[0], bool))).astype(bool)
            if valid.any():
                pts.append(pos[valid][:, :2])
    if not pts:
        raise ValueError("review card: neither the ego nor any edited actor has a valid pose")
    span = np.concatenate(pts, axis=0)
    lo = span.min(axis=0) - margin_m
    hi = span.max(axis=0) + margin_m
    # Keep the two panels square-ish so a reviewer reads distances honestly.
    centre = 0.5 * (lo + hi)
    half = float(np.max(hi - lo)) / 2.0
    lo, hi = centre - half, centre + half

    culled: List[np.ndarray] = []
    for feature in (sd.get("map_features") or {}).values():
        line = feature.get("polyline")
        if line is None:
            continue
        line = np.asarray(line)[:, :2]
        inside = np.all((line >= lo) & (line <= hi), axis=1)
        if np.any(inside):
            culled.append(line[inside])
    logger.info(
        "render_topdown_gif: %d map features inside the action bbox (of %d)",
        len(culled),
        len(sd.get("map_features") or {}),
    )

    colours = {"ego": "#1f77b4", "edited": "#d62728", "background": "#9e9e9e"}
    images = []
    for frame in frames:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=110)
        for ax, (variant, variant_sd) in zip(axes, variants.items()):
            for line in culled:
                ax.plot(line[:, 0], line[:, 1], color="#dddddd", linewidth=0.6, zorder=0)
            for cx, cy, heading, length, width, tag in _track_boxes(variant_sd, frame):
                if not (lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]):
                    continue
                poly = _corners(cx, cy, heading, length, width)
                ax.fill(
                    poly[:, 0], poly[:, 1], color=colours[tag],
                    alpha=0.9 if tag != "background" else 0.45,
                    zorder=3 if tag == "edited" else 2,
                )
                if tag != "background":
                    nose = np.array([cx, cy]) + 0.6 * length * np.array(
                        [np.cos(heading), np.sin(heading)]
                    )
                    ax.plot([cx, nose[0]], [cy, nose[1]], color=colours[tag], linewidth=1.4,
                            zorder=4)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            label = "e⁰ (control)" if variant == "e_zero" else "e⁺ (built)"
            ax.set_title(f"{label}   t = {frame * recipe.frames.dt_s:.1f}s", fontsize=10)
        fig.suptitle(f"{recipe.recipe_id}   ·   leaf {recipe.leaf}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, images, fps=fps, loop=0)
    logger.info("render_topdown_gif: %s (%d frames)", path, len(images))
    return path


def placement_numbers(recipe: Recipe, sd: dict) -> Dict[str, Any]:
    """The numbers a reviewer sanity-checks the placement against.

    Only what the recipe DETERMINES: each actor's spawn measured against the
    host's own route and hand-off, and which controller takes over there. How
    close it actually comes is a property of an episode, not of the file, so it
    is read off the trace afterwards rather than predicted here.
    """
    variants = _variant_scenarios(recipe, sd)
    edited = variants["e_plus"]
    sdc_id = str((edited.get("metadata") or {}).get("sdc_id", "ego"))
    ego = np.asarray(edited["tracks"][sdc_id]["state"]["position"], np.float64)
    dt = recipe.frames.dt_s
    handoff_s = recipe.frames.after_frame * dt

    numbers: Dict[str, Any] = {
        "recipe_id": recipe.recipe_id,
        "leaf": recipe.leaf,
        "host": f"{recipe.host.scene}@{recipe.host.world_version}",
        "episode": f"T={recipe.frames.T} frames, dt={dt:.3f}s ({recipe.frames.T * dt:.1f}s)",
        "hand_off": f"frame {recipe.frames.after_frame} (t={handoff_s:.1f}s)",
        "actors": {},
    }

    for name, actor in recipe.actors.items():
        if actor.op == "remove":
            numbers["actors"][name] = {"op": "remove", "source_track_id": actor.source_track_id}
            continue
        # The card used to report closest approach, closing speed and a
        # reaction estimate, all read off the actor's baked trajectory against
        # the log-replay ego. A reactive actor has no trajectory until an
        # episode is run, and the ego it will meet is the policy under test,
        # so those numbers cannot be computed here without inventing them.
        # What the card can still show is what the recipe DETERMINES: where the
        # actor starts, how far the ego is from it at the hand-off, and which
        # controller takes over. The rest belongs to the episode trace.
        spawn = np.asarray(actor.spawn.get("position", (0.0, 0.0, 0.0)), np.float64)
        handoff_xy = ego[min(recipe.frames.after_frame, len(ego) - 1), :2]
        entry = {
            "op": actor.op,
            "template": (actor.authored or {}).get("template"),
            "reference": (actor.authored or {}).get("reference"),
            "authored_arc_m": (actor.authored or {}).get("arc"),
            "authored_speed_mps": (actor.authored or {}).get("speed"),
            "keep_appearance": actor.keep_appearance,
            "policy": (actor.policy or {}).get("kind"),
            "policy_params": {k: v for k, v in (actor.policy or {}).items()
                              if k not in ("kind", "path_polyline")},
            "spawn_xy": [round(float(spawn[0]), 2), round(float(spawn[1]), 2)],
            "spawn_z_m": round(float(spawn[2]), 2) if spawn.size > 2 else None,
            "spawn_heading_rad": round(float(actor.spawn.get("heading", 0.0)), 3),
            "distance_from_ego_at_handoff_m": round(
                float(np.linalg.norm(spawn[:2] - handoff_xy)), 2),
        }
        numbers["actors"][name] = entry
    return numbers


def _checklist_text(recipe: Recipe) -> str:
    """The leaf's checklist, resolved through the leaf registry.

    It used to resolve relative to the editing package, against a second,
    byte-identical copy of the nine checklists under ``editing/leaves/``. That
    made the reference's spelling load-bearing: a recipe saying
    ``checklist: leaves/R-2.md`` resolved, one saying ``checklist: R-4.md``
    did not, and R-4's card silently printed "not found". The leaf owns its
    checklist, so ask the leaf.
    """
    ref = str((recipe.review or {}).get("checklist", "") or "")
    if recipe.leaf:
        try:
            from navsafe.benchmark.leaves import load_leaf

            path = load_leaf(recipe.leaf).checklist_path()
            if path.is_file():
                return path.read_text()
        except Exception:  # noqa: BLE001 — an unknown leaf falls through to `ref`
            pass
    if not ref:
        return "_No checklist named in the recipe's `review.checklist`._"
    path = Path(ref)
    if not path.is_absolute():
        path = LEAVES_DIR / Path(ref).name
    if not path.is_file():
        return f"_Checklist `{ref}` not found at `{path}`._"
    return path.read_text()


def build_review_card(
    recipe: Recipe,
    sd: dict,
    out_dir: "str | Path",
    *,
    cam_gif: Optional["str | Path"] = None,
    topdown: bool = True,
) -> ReviewCard:
    """Write one candidate's review card, with its pair animation and numbers.

    Args:
        recipe: the frozen recipe under review.
        sd: its host scenario, unedited.
        out_dir: directory to write into (created).
        cam_gif: a camera-view gif from an eval run against the
            reconstruction, if one exists. Without it the card says so —
            scale, grounding and appearance harmony are not judgable from a
            bird's-eye plot.
        topdown: render the BEV pair animation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    numbers = placement_numbers(recipe, sd)

    topdown_path = None
    if topdown:
        topdown_path = render_topdown_gif(recipe, sd, out_dir / "topdown.gif")

    lines: List[str] = []
    lines.append(f"# Review card · {recipe.recipe_id}\n")
    lines.append(f"**Leaf {recipe.leaf}** · {recipe.scenario or '—'} · frozen {recipe.frozen_at}\n")
    lines.append("## The pair\n")
    if topdown_path:
        lines.append(f"![top-down, e⁰ vs e⁺]({topdown_path.name})\n")
    if cam_gif:
        lines.append(f"![camera, frame 0 rig]({Path(cam_gif).name})\n")
    else:
        lines.append(
            "> **No camera view.** This card was built without an eval render, so scale, "
            "grounding, orientation and appearance harmony cannot be judged from it — only "
            "geometry and timing can. Re-run with `--cam-gif` once the episode has been "
            "rendered against the reconstruction.\n"
        )

    lines.append("## Placement numbers\n")
    lines.append(f"- host: `{numbers['host']}`")
    lines.append(f"- episode: {numbers['episode']}, hand-off at {numbers['hand_off']}")
    lines.append(
    )
    for name, entry in numbers["actors"].items():
        lines.append(f"\n### `{name}` — {entry['op']}\n")
        if entry["op"] == "remove":
            lines.append(f"- drops `{entry['source_track_id']}`")
            continue
        lines.append(
            f"- authored: `{entry['template']}` on `{entry['reference']}`, "
            f"arc {entry['authored_arc_m']} m, speed {entry['authored_speed_mps']} m/s"
        )
        lines.append(
            f"- spawn: {entry['spawn_xy']} at z {entry['spawn_z_m']} m, "
            f"keep_appearance={entry['keep_appearance']}"
        )
        lines.append(f"- speed range: {entry['speed_mps']} m/s, valid {entry['valid_frames']}")
        lines.append(
            f"- closest approach to the **log-replay** ego: "
            f"**{entry['closest_approach_m']} m** at t={entry['closest_approach_s']} s "
            f"({entry['time_from_handoff_s']:+.2f} s from the hand-off)"
        )
        lines.append(f"- closing speed there: {entry['closing_speed_mps']} m/s")
        if "seconds_to_event_at_handoff" in entry:
            lines.append(
                f"- rough time-to-event at the hand-off: "
                f"{entry['seconds_to_event_at_handoff']} s"
            )
        if "inside_arrival_window" in entry:
            mark = "yes" if entry["inside_arrival_window"] else "**NO**"
            lines.append(f"- event inside the assumed arrival window: {mark}")

    lines.append("\n## Checklist\n")
    lines.append(_checklist_text(recipe))

    lines.append("\n## Verdict\n")
    lines.append("```yaml")
    lines.append(f"recipe_id: {recipe.recipe_id}")
    lines.append("verdict:     # pass | fail | rebuild")
    lines.append("reason_code: # see below; 'ok' on a pass")
    lines.append("reviewer:")
    lines.append("notes:")
    lines.append("```\n")
    lines.append("| reason code | means |")
    lines.append("|---|---|")
    for code, meaning in REASON_CODES.items():
        lines.append(f"| `{code}` | {meaning} |")
    lines.append(
        "\n_Reason codes aggregate into template fixes rather than per-episode patches: "
        "ten `wrong_lane` verdicts are one placement bug._\n"
    )

    card_path = out_dir / "card.md"
    card_path.write_text("\n".join(lines))
    logger.info("build_review_card: %s", card_path)
    return ReviewCard(
        directory=out_dir,
        card_path=card_path,
        topdown_gif=topdown_path,
        cam_gif=Path(cam_gif) if cam_gif else None,
        numbers=numbers,
    )
