# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Command line for the NavSafe scene-editing pipeline.

One subcommand per pipeline step, in the order a scenario is built:

    # 1. asset preparation — automated measurement, one visual confirm
    cli assets status                                    what exists vs declared
    cli assets acquire  KEY [--execute]                  put the PLY on disk
    cli assets compose  KEY                              build one asset from its parts
    cli assets calibrate KEY [--write out.ply]           size it by measurement, not by eye
    cli assets animate  KEY [--phases 10]                bake a gait so it walks, not slides

    # 2. host qualification + survey — automated, read the numbers
    cli qualify  --roots-glob '<pattern>' [--event-type L]     which event types can each host carry?
    cli probe    --data-root <arrow> [--arcs ...]        frame grid, cross-sections, tracks
    cli anchors  --data-root <arrow>                     named landmarks (intersections, ...)

    # 3. bake — nothing is authored per host; the event type manifest is the spec
    cli bake     --event-type L --scenario-id ID                -> a frozen recipe

    # 4. review — automated render, HUMAN judgment
    cli card     r.yaml --data-root <arrow> --out-dir cards/001

    # consumers of a frozen recipe
    cli verify   r.yaml                                  checksums
    cli replay   r.yaml --variant e_plus                 the edits it produces

    # per-data-root calibration, shared with eval
    cli calibrate-ground-z --data-root <arrow> --out calib.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from navsafe.benchmark import config as cfg
from navsafe.benchmark.editing.assets.acquire import acquire_asset, acquire_missing
from navsafe.benchmark.editing.assets.compose import compose_from_registry
from navsafe.benchmark.editing.assets.registry import AssetRegistry
from navsafe.benchmark.editing.author import author_recipe
from navsafe.benchmark.editing.ground_z import resolve_z_to_ground
from navsafe.benchmark.editing.host import describe_tracks, load_host_scenario
from navsafe.benchmark.editing.ground_z import REGISTRY_ENV, scene_key
from navsafe.benchmark.editing.placement.probe import HostProbe
from navsafe.benchmark.editing.recipe.freeze import freeze_recipe
from navsafe.benchmark.editing.recipe.replay import VARIANTS, edits_from_recipe_file
from navsafe.benchmark.editing.recipe.schema import load_recipe
from navsafe.benchmark.editing.review.cards import build_review_card


def _add_ego_z_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ego-z-to-ground",
        type=float,
        default=None,
        help="Ego-pose-to-ground drop. The loader already reports the ego z at the ground "
        "(rear-axle reference, not the bbox centre), so this only absorbs per-artifact "
        "recon-road drift and defaults to 0. Precedence: this flag > the "
        "NUREC_GROUND_Z_CALIB per-scene registry > 0 — the same resolution the evaluator uses.",
    )


def _add_host_args(parser: argparse.ArgumentParser) -> None:
    """For the verbs that take a host DIRECTLY, by Arrow root."""
    parser.add_argument("--data-root", required=True, help="converted py123d Arrow root")
    parser.add_argument("--scene", default=None, help="scene uuid / log name")
    parser.add_argument("--scene-index", type=int, default=0)
    _add_ego_z_arg(parser)


def _z_to_ground(args) -> float:
    value, reason = resolve_z_to_ground(args.ego_z_to_ground, args.data_root)
    print(f"[editing] ego-z-to-ground {value} ({reason})")
    return value


def cmd_probe(args) -> int:
    args.ego_z_to_ground = _z_to_ground(args)
    sd, scene_id = load_host_scenario(
        args.data_root, scene_id=args.scene, scene_index=args.scene_index, require_map=False
    )
    probe = HostProbe(sd, ego_z_to_ground_m=args.ego_z_to_ground)
    print(f"scene            {scene_id}")
    print(f"frames           T={probe.T} dt={probe.dt_s:.3f}s ({probe.T * probe.dt_s:.1f}s)")
    print(f"ego route        {probe.ego_route.total:.1f} m")
    print(f"hand-off arc     {probe.anchor_arc(probe.ego_route, args.after_frame):.1f} m "
          f"(after_frame={args.after_frame})")
    print(f"lanes            {len(probe.lane_ids())}")
    measured = probe.measure_road_drift()
    if measured:
        print(
            f"road drift       ego z {measured['ego_z_median']:+.2f} m vs road "
            f"{measured['actor_road_median']:+.2f} m from {measured['samples']} perception "
            f"boxes -> residual {measured['residual_m']:+.2f} m "
            f"({measured['if_box_is_base_m']:+.2f} m if the box z is the base). "
            f"Using drop {args.ego_z_to_ground:.2f} m."
        )
    else:
        print("road drift       no actor carries dimensions; cannot cross-check the road height")
    route_lanes = probe.route_lane_ids()
    if route_lanes:
        print(f"route lanes      {' -> '.join(route_lanes)}")
    anchor = probe.anchor_arc(probe.ego_route, args.after_frame)
    print("\ncross-section along the ego route (arc measured from the hand-off):")
    for arc in args.arcs:
        section = probe.cross_section(probe.ego_route, anchor + float(arc))
        print(f"  +{float(arc):6.1f} m   {section.describe()}")
    print("\ntracks, nearest approach to the ego first:")
    for row in describe_tracks(sd, moving_only=args.moving_only)[: args.max_tracks]:
        print(
            f"  {row['track_id']:<26} {row['type']:<14} "
            f"near={row['nearest_to_ego_m']:>6.1f} m  travelled={row['travelled_m']:>6.1f} m  "
            f"valid={row['valid_frames']:>3}  dims={row['dims']}"
        )
    return 0


def cmd_qualify(args) -> int:
    """Which event types can this host (or each of these hosts) carry?"""
    import glob as _glob

    from navsafe.benchmark.editing.qualify import qualify_host
    from navsafe.benchmark.event_types import available

    leaves = args.leaf or None
    if leaves:
        known = available()
        unknown = [l for l in leaves if l not in known]
        if unknown:
            print(f"error: unknown event type {unknown}; known: {known}")
            return 2
    roots = sorted(_glob.glob(args.roots_glob)) if args.roots_glob else [args.data_root]
    if not roots or roots == [None]:
        print("error: pass --data-root <arrow root> or --roots-glob '<pattern>'")
        return 2

    qualified: dict = {}
    for root in roots:
        try:
            value, _ = resolve_z_to_ground(args.ego_z_to_ground, root)
            sd, scene_id = load_host_scenario(
                root, scene_id=args.scene, scene_index=args.scene_index, require_map=False
            )
            probe = HostProbe(sd, ego_z_to_ground_m=value)
            probe.after_frame = int(args.after_frame)
            verdicts = qualify_host(probe, scene=scene_id, leaves=leaves)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop a sweep
            print(f"SKIPPED {root} ({type(exc).__name__}: {exc})")
            continue
        print(f"\n── {root}")
        for verdict in verdicts:
            print(verdict.describe())
            if verdict.ok:
                qualified.setdefault(verdict.leaf, []).append(root)
    if len(roots) > 1 and qualified:
        print("\nqualified hosts per leaf:")
        for leaf in sorted(qualified):
            print(f"  {leaf:<6} {len(qualified[leaf]):>3}  " + "  ".join(qualified[leaf][:4])
                  + ("  …" if len(qualified[leaf]) > 4 else ""))
    return 0


def cmd_anchors(args) -> int:
    """Print the named landmarks a spec's ``anchor:`` field can refer to."""
    from navsafe.benchmark.editing.placement.anchors import find_anchors

    args.ego_z_to_ground = _z_to_ground(args)
    sd, scene_id = load_host_scenario(
        args.data_root, scene_id=args.scene, scene_index=args.scene_index, require_map=False
    )
    probe = HostProbe(sd, ego_z_to_ground_m=args.ego_z_to_ground)
    probe.after_frame = int(args.after_frame)
    anchors = find_anchors(probe, after_frame=args.after_frame)
    print(f"scene   {scene_id}")
    print(f"anchors (arc measured from the hand-off, after_frame={args.after_frame}):\n")
    for anchor in anchors:
        print("  " + anchor.describe())
    kinds = {a.kind for a in anchors}
    if "intersection_entry" not in kinds:
        print(
            "\n  NOTE: no intersection found on this route. An event type whose placement rule "
            "anchors on one (wrong-way signs, crossing traffic) needs a different host."
        )
    print(
        "\nuse in a spec:   authored: {anchor: intersection_1_exit, arc: 5.0, ...}\n"
        "aliases:         first_intersection_entry / first_intersection_exit / first_crosswalk"
    )
    return 0


def _next_recipe_path(leaf: str, scenario: str, scene_id: str, label: str) -> "tuple[Path, int]":
    """``<recipes>/<LEAF>/<scenario>/<scene>/<NNN>-<label>.yaml``, NNN unused."""
    from navsafe.benchmark.scenario_cli import RECIPES_DIR

    d = RECIPES_DIR / leaf / scenario / scene_id
    used = set()
    if d.is_dir():
        for p in d.glob("*.yaml"):
            head = p.name.split("-", 1)[0]
            if head.isdigit():
                used.add(int(head))
    seq = max(used) + 1 if used else 1
    return d / f"{seq:03d}-{label}.yaml", seq


def cmd_bake(args) -> int:
    """An event type plus a scenario id -> a frozen recipe. No per-host spec involved."""
    if not (args.leaf and args.scenario_id):
        print("error: pass --event-type <L> --scenario-id <id>  (`navsafe mine --event-type <L>` "
              "prints the ids). There is no per-host spec route: what to insert is "
              "declared once in leaves/<LEAF>.yaml.")
        return 2

    from navsafe.benchmark.editing.assets.registry import AssetRegistry
    from navsafe.benchmark.editing.qualify import qualify_host
    from navsafe.benchmark.event_types import load_leaf
    from navsafe.benchmark.event_types.hosts import HostError, resolve_host
    from navsafe.benchmark.event_types.rule import expand_leaf_rule

    man = load_leaf(args.leaf)
    try:
        host = resolve_host(args.scenario_id, args.roots_glob)
    except HostError as exc:
        print(f"error: {exc}")
        return 2

    value, _ = resolve_z_to_ground(args.ego_z_to_ground, host.data_root)
    # The Arrow's internal scene id is a py123d UUID; the directory name is what
    # a human types. Take the scene by index and keep both identities.
    sd, arrow_scene = load_host_scenario(host.data_root, scene_id=None, scene_index=0,
                                         require_map=True)
    probe = HostProbe(sd, ego_z_to_ground_m=value)
    probe.after_frame = int((man.rule.ego or {}).get("replay_frames", 8))

    # Qualify HERE, not only in `mine`: baking an unqualified host used to be
    # possible and silent, and re-checking one host costs seconds.
    verdict = qualify_host(probe, scene=host.scene_id, leaves=[args.leaf])[0]
    failed = [c for c in verdict.checks if not c.ok and not c.name.endswith("(info)")]
    if failed and not args.force:
        print(f"error: {host.scene_id} does not qualify for {args.leaf}:")
        for check in failed:
            print(f"  {check.name}: {check.evidence}")
        print("  This is a finding about the host, not a flag to pass. Pick another "
              "scenario from `navsafe mine`, or --force if you are deliberately "
              "building an off-contract probe.")
        return 2

    # A person watched this render and said no. That is the one verdict a
    # geometry gate cannot argue with — the predicates measure the road, and
    # "this does not read as the scenario" is not something they can see. It is
    # still only a refusal, not a silent skip, and --force still overrides.
    human = _human_verdict(args, man, host)
    if human.get("rejected") and not args.force:
        print(f"error: the reviewer marked {host.scene_id} `{human.get('keep')}` for "
              f"{args.leaf}"
              + (f" — {human['note']}" if human.get("note") else "")
              + f"\n  ({human.get('source') or 'the review sheet'}). Predicates measure the "
                f"road; whether the clip READS as this event type is what a person watched for. "
                f"Pick another from `navsafe mine --event-type {args.leaf}`, or --force.")
        return 2

    registry = AssetRegistry.load(getattr(args, "registry", None))
    selection = {
        "scenario_id": host.scene_id,
        "arrow_scene": arrow_scene,
        "host_family": host.family,
        "recon_windows": host.windows,
        "qualify": [{"name": c.name, "ok": bool(c.ok), "evidence": c.evidence}
                    for c in verdict.checks],
        "miner": (man.mine or {}).get("miner", ""),
    }
    if args.candidates or (man.mine or {}).get("miner"):
        selection.update(_selection_from_candidates(args, man, host))

    out = Path(args.out) if args.out else None
    if out is None:
        out, seq = _next_recipe_path(man.leaf, man.scenario, host.scene_id, args.label)
    else:
        seq = 1
    spec = expand_leaf_rule(
        man, scene_id=arrow_scene, host=host, registry=registry, variety=args.variety,
        insert=man.inserts and not args.no_insert, selection=selection,
        recipe_id=args.recipe_id or f"{man.leaf}/{man.scenario}/{host.scene_id}/{seq:03d}",
    )
    spec.setdefault("ego", {})["z_to_ground"] = value

    recipe, diagnostics = author_recipe(sd, spec, registry=registry)
    freeze_recipe(recipe, out, frozen_at=args.frozen_at, overwrite=args.overwrite)
    print(f"froze {recipe.recipe_id} ({recipe.provenance}, {len(recipe.actors)} actors) -> {out}")
    print(json.dumps(diagnostics, indent=2, default=float))
    if len(host.windows) > 1:
        print(f"\nThis host is {len(host.windows)} reconstructions; the render needs the "
              f"hand-off:\n  navsafe handoff --token {host.token}")
    return 0


def _selection_from_candidates(args, man, host) -> dict:
    """The mined evidence for this scenario, if `navsafe mine` left any."""
    from navsafe.benchmark.mining.mine import load_rows
    from navsafe.benchmark.scenario_cli import CANDIDATES_DIR

    path = Path(args.candidates) if args.candidates else CANDIDATES_DIR / f"{man.leaf}.jsonl"
    if not path.is_file():
        return {}
    for row in load_rows(path):
        if row.scenario_id == host.scene_id or row.token == host.token:
            out = {"pool": row.pool, "event_t_offset_s": row.event_t_offset_s,
                   "min_eval_frames": row.min_eval_frames, "candidates_file": str(path)}
            if row.human:
                out["human"] = dict(row.human)
            return out
    return {}


def _human_verdict(args, man, host) -> dict:
    """The reviewer's verdict on this host, from `navsafe mine --review`."""
    return (_selection_from_candidates(args, man, host) or {}).get("human") or {}



def cmd_verify(args) -> int:
    recipe = load_recipe(args.recipe)
    print(
        f"OK  {recipe.recipe_id}\n"
        f"    leaf={recipe.leaf} host={recipe.host.scene}@{recipe.host.world_version}\n"
        f"    T={recipe.frames.T} dt={recipe.frames.dt_s} after_frame={recipe.frames.after_frame}\n"
        f"    actors={list(recipe.actors)}\n"
    )
    return 0


def cmd_replay(args) -> int:
    _, edits = edits_from_recipe_file(args.recipe, variant=args.variant)
    print(json.dumps(edits, indent=2, default=float))
    return 0


def cmd_card(args) -> int:
    recipe = load_recipe(args.recipe)
    sd, _ = load_host_scenario(
        args.data_root,
        scene_id=args.scene or recipe.host.scene,
        scene_index=args.scene_index,
        require_map=False,
    )
    args.ego_z_to_ground = _z_to_ground(args)
    card = build_review_card(
        recipe, sd, args.out_dir, cam_gif=args.cam_gif, topdown=not args.no_topdown
    )
    print(f"review card -> {card.card_path}")
    if card.topdown_gif:
        print(f"top-down    -> {card.topdown_gif}")
    if not card.cam_gif:
        print(
            "note: no camera view attached — scale, grounding and appearance harmony "
            "cannot be judged from this card alone."
        )
    return 0


def cmd_calibrate_ground_z(args) -> int:
    """Measure the recon-road drift for a data root and emit the calib registry.

    The ego z is already ground-referenced, so the only legitimate nonzero drop
    is a reconstruction whose rendered road drifts from the real one. This
    measures that against the host's own perception boxes and writes it in the
    format ``NUREC_GROUND_Z_CALIB`` expects, rather than inventing a second
    calibration path beside the one the evaluator already reads.
    """
    from navsafe.scenario.py123d_scenes import enumerate_scenes
    from navsafe.scenario.py123d_scenes import scene_id as scene_id_of

    scenes = enumerate_scenes(args.data_root)
    if args.max_scenes:
        scenes = scenes[: int(args.max_scenes)]
    residuals = []
    print(f"{'scene':<40} {'residual':>9} {'if base':>9} {'boxes':>7}")
    for i, scene in enumerate(scenes):
        try:
            sd, sid = load_host_scenario(
                args.data_root, scene_id=scene_id_of(scene), require_map=False
            )
            measured = HostProbe(sd, ego_z_to_ground_m=0.0).measure_road_drift()
        except Exception as exc:  # noqa: BLE001 - one bad scene must not stop a sweep
            print(f"{scene_id_of(scene):<40} SKIPPED ({type(exc).__name__}: {exc})")
            continue
        if measured is None:
            print(f"{sid:<40} {'n/a':>9}  (no actor carries dimensions)")
            continue
        residuals.append(measured["residual_m"])
        print(
            f"{sid:<40} {measured['residual_m']:>+9.3f} {measured['if_box_is_base_m']:>+9.3f} "
            f"{measured['samples']:>7}"
        )
    if not residuals:
        print("nothing measurable in this data root")
        return 1

    import statistics

    value = round(statistics.median(residuals), 3)
    spread = (min(residuals), max(residuals))
    key = scene_key(args.data_root)
    print(
        f"\nmedian residual {value:+.3f} m over {len(residuals)} scene(s) "
        f"(spread {spread[0]:+.3f} .. {spread[1]:+.3f})"
    )
    if spread[1] - spread[0] > 0.3:
        print(
            "  NOTE: the spread is wide, so one value per data root will be wrong for some "
            "scenes. The registry is keyed by the data root's parent directory, so a finer "
            "calibration means splitting the roots."
        )
    if not args.out:
        print("\n(measured only — pass --out to write the registry)")
        return 0
    out = Path(args.out)
    registry = json.loads(out.read_text()) if out.exists() else {}
    registry[key] = value
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out} [{key}] = {value}\nuse it with: export {REGISTRY_ENV}={out}")
    return 0


def cmd_assets_calibrate(args) -> int:
    """Measure a PLY's visual extent and solve the scale that fits its target dims."""
    from navsafe.benchmark.editing.assets.calibrate import CalibrationError, calibrate_file
    from navsafe.benchmark.editing.assets.registry import AssetError

    registry = AssetRegistry.load(args.registry)
    ply, family = args.key, args.family
    if args.key in registry:
        entry = registry.get(args.key)
        if not entry.present:
            print(f"asset {args.key!r} is declared but its PLY is not on disk at {entry.ply}")
            return 1
        ply = entry.ply
        family = family or entry.family
        print(f"asset    {args.key} ({entry.source})  family={family or '?'}")
    print(f"ply      {ply}")

    if args.species:
        try:
            target = registry.species_dims(args.species)
        except AssetError as exc:
            print(f"error: {exc}")
            return 1
        print(f"target   {target} (canonical dims for species {args.species!r})")
    elif args.target_dims:
        target = [float(v) for v in args.target_dims]
        print(f"target   {target} (explicit --target-dims)")
    else:
        try:
            target = registry.family_dims(family or "")
        except AssetError as exc:
            print(f"error: {exc}")
            return 1
        print(f"target   {target} (canonical dims for family {family!r})")

    try:
        calibration = calibrate_file(
            ply,
            target,
            opacity_min=args.opacity_min,
            sigma_k=args.sigma_k,
            out_path=args.write,
            prune=not args.no_prune,
            rebase=not args.no_rebase,
            forward=args.forward_axis,
            up=args.up_axis,
            auto_axes=args.auto_axes,
            fit_axis=args.fit_axis,
        )
    except CalibrationError as exc:
        print(f"error: {exc}")
        return 1
    print()
    print(calibration.describe())
    if args.write:
        print(
            f"\nwrote {args.write} (prune={not args.no_prune}, rebase={not args.no_rebase})\n"
            f"register it: point the entry's `ply:` at the new file and set\n"
            f"  dims: {calibration.dims_after}\n"
            f"then confirm once from a rendered frame (cli card / an eval render) — "
            f"measurement replaces the search loop, not the final look."
        )
    else:
        print("\n(measured only — pass --write OUT.ply to apply the scale)")
    return 0


def cmd_assets_animate(args) -> int:
    """Bake a gait pose bank so an inserted pedestrian walks instead of sliding."""
    from navsafe.benchmark.editing.assets.animate import RigError, bake_asset

    registry = AssetRegistry.load(args.registry)
    ply, key, dims = args.key, Path(args.key).stem, None
    if args.key in registry:
        entry = registry.get(args.key)
        if not entry.present:
            print(f"asset {args.key!r} is declared but its PLY is not on disk at {entry.ply}")
            return 1
        ply, key, dims = entry.ply, args.key, entry.dims
        print(f"asset    {args.key} ({entry.source})  dims={dims}")
    if args.out:
        out = Path(args.out)
    elif cfg.GAIT_BANK:
        out = cfg.GAIT_BANK / key
    else:
        print("NAVSAFE_GAIT_BANK is unset and no longer has a default; "
              "set it or pass --out")
        return 1
    print(f"ply      {ply}")
    print(f"out      {out}")

    # An animal is not a person with four legs: the kimodo checkout ships no
    # quadruped template and no quadruped motion, so the whole path -- SMAL
    # template, generated trot -- lives in `quadruped.py` and is reached by
    # FAMILY, not by a flag. Routing it here is what stops that path from
    # being a script someone has to remember; the four species we carry (dog,
    # cow, hippo, horse) are exactly four of SMAL's five family means.
    species = registry.get(args.key).species if args.key in registry else ""
    is_animal = args.key in registry and registry.get(args.key).family == "animal"
    try:
        if is_animal:
            from navsafe.benchmark.editing.assets.quadruped import bake_animal

            if not species:
                print(f"error: asset {args.key!r} is family 'animal' but declares no "
                      f"`species:`. SMAL is fitted per family and the shape space is "
                      f"not one-size-fits-all -- an out-of-family fit grows a phantom "
                      f"hind leg. Add `species:` to the registry entry.")
                return 1
            print(f"gait     procedural trot on the SMAL {species} template")
            manifest = bake_animal(
                ply, out, family=species, phases=args.phases, dims=dims,
                prune_opacity=args.prune_opacity,
            )
        else:
            print(f"motion   {cfg.motion_npz(args.motion)}")
            manifest = bake_asset(
                ply, out,
                phases=args.phases,
                motion=None if args.motion == "default" else _load_motion(args.motion),
                dims=dims,
                prune_opacity=args.prune_opacity,
            )
    except RigError as exc:
        print(f"error: {exc}")
        return 1

    fit = manifest["fit"]
    print()
    print(f"fit      motion frame {fit['frame']}, chamfer {fit['chamfer']:.4f} m "
          f"(body-to-cloud surface distance; clothing is worth a few cm)")
    print(f"bank     {len(manifest['phases'])} phases over "
          f"{manifest['cycle_frames']} motion frames, stride "
          f"{manifest['stride_m']} m, {manifest['gaussians']} gaussians each")
    print(f"check    phase 0 reproduces the source asset to "
          f"{manifest['round_trip_m']:.2e} m")
    print()
    print(f"declare it on the asset with `gait_bank: {out.name}` in registry.yaml, then\n"
          f"re-bake the event type -- `navsafe bake` pins the bank and its digest, so a later\n"
          f"re-bake cannot silently drop the gait. Confirm from a rendered frame\n"
          f"(NUREC_GRPC_DEBUG_GAIT=1 logs the phase clock; a gait stuck on phase 0\n"
          f"renders exactly like the static asset it replaced).")
    return 0


def _load_motion(name: str):
    from navsafe.benchmark.editing.assets.animate import load_motion
    return load_motion(cfg.motion_npz(name))


def cmd_assets(args) -> int:
    import shlex

    registry = AssetRegistry.load(args.registry)
    if args.assets_command == "status":
        status = registry.status()
        print(f"registry {status['registry']}")
        print(f"present  {status['present']}/{status['total']}")
        print("\nby source:")
        for source, counts in sorted(status["by_source"].items()):
            print(f"  {source:<16} {counts['present']:>3}/{counts['declared']:<3}")
        print("\nby leaf:")
        for leaf, counts in status["by_leaf"].items():
            bar = "#" * counts["present"] + "." * (counts["declared"] - counts["present"])
            print(f"  {leaf:<6} {counts['present']:>2}/{counts['declared']:<2}  {bar}")
        if args.verbose:
            print("\nmissing:")
            for entry in registry.missing():
                print(f"  {entry.key:<28} {entry.source:<14} -> {entry.ply}")
        return 0

    if args.assets_command == "compose":
        report = compose_from_registry(args.key, registry)
        print(json.dumps(report, indent=2))
        return 0

    plans = (
        [acquire_asset(args.key, registry, execute=args.execute)]
        if args.key
        else acquire_missing(registry, leaf=args.leaf, execute=args.execute)
    )
    for plan in plans:
        head = f"{plan['key']:<28} {plan['source']:<14}"
        if plan.get("note"):
            print(f"{head} {plan['note']}")
        elif not plan["steps"]:
            print(f"{head} nothing to do")
        else:
            print(head)
        for step in plan["steps"]:
            marker = "  $ " if plan["runnable_here"] else "  (elsewhere) $ "
            print(marker + shlex.join(step))
        if "ok" in plan:
            print(f"  -> {'ok' if plan['ok'] else 'FAILED: ' + plan.get('failed_step', '')}")
    if not args.execute and any(p["steps"] for p in plans):
        print("\n(planned only — pass --execute to run the runnable ones)")
    return 0


def attach_subcommands(sub) -> None:
    """Mount the construction-tier verbs onto an existing subparsers object.

    The `navsafe` front door owns selection (mine / evidence / bake) and
    mounts these, so one command covers both routes to a recipe. Kept as a
    function rather than inlined so `python -m navsafe.benchmark.editing.cli`
    keeps working for anything that already scripts it.
    """
    probe = sub.add_parser("probe", help="what this host offers (authoring aid)")
    _add_host_args(probe)
    probe.add_argument("--after-frame", type=int, default=8, help="the replay hand-off")
    probe.add_argument(
        "--arcs",
        type=float,
        nargs="+",
        default=[10.0, 25.0, 40.0, 55.0, 70.0],
        help="arcs past the hand-off to measure the cross-section at",
    )
    probe.add_argument("--max-tracks", type=int, default=20)
    probe.add_argument("--moving-only", action="store_true")
    probe.set_defaults(func=cmd_probe)

    qualify = sub.add_parser(
        "qualify",
        help="which event types can this host carry? (the geometry tier of mining)",
    )
    qualify.add_argument("--data-root", default=None, help="one converted py123d Arrow root")
    qualify.add_argument(
        "--roots-glob", default=None,
        help="sweep many roots instead, e.g. '<data-root>/full_test/*/arrow'; prints a "
        "qualified-hosts-per-leaf summary at the end",
    )
    qualify.add_argument("--scene", default=None, help="scene uuid / log name")
    qualify.add_argument("--scene-index", type=int, default=0)
    qualify.add_argument("--ego-z-to-ground", type=float, default=None,
                         help="per-artifact recon-road drift; same resolution as probe/bake")
    qualify.add_argument("--after-frame", type=int, default=8, help="the replay hand-off")
    qualify.add_argument(
        "--event-type", "--leaf", dest="leaf", action="append", default=None, metavar="EVENT_TYPE",
        help="restrict to one event type (repeatable); default = every event type in the manifest",
    )
    qualify.set_defaults(func=cmd_qualify)

    anchors = sub.add_parser(
        "anchors",
        help="named landmarks on this host (intersections, crosswalks) a spec can anchor on",
    )
    _add_host_args(anchors)
    anchors.add_argument("--after-frame", type=int, default=8, help="the replay hand-off")
    anchors.set_defaults(func=cmd_anchors)

    bake = sub.add_parser("bake", help="an event type + a scenario id -> frozen recipe")
    # NOT `_add_host_args`: `bake` names its host with --scenario-id and resolves
    # the Arrow root, family and recon windows from it. It used to take the
    # shared host args too, and --data-root / --scene / --scene-index were
    # accepted and then silently ignored — a flag that changes nothing is worse
    # than one that is absent, because the run looks configured.
    _add_ego_z_arg(bake)
    bake.add_argument("--event-type", "--leaf", dest="leaf", default=None, metavar="EVENT_TYPE",
                      help="which event type to build")
    bake.add_argument("--scenario-id", default=None,
                      help="a scene id (or unique nuPlan token) `navsafe mine` printed")
    bake.add_argument("--roots-glob", action="append", default=None,
                      metavar="GLOB[:family[:rig]]", help="where converted hosts live")
    bake.add_argument("--candidates", default=None,
                      help="the .jsonl `navsafe mine` wrote (default: its standard path)")
    bake.add_argument("--variety", type=int, default=0,
                      help="for a `variety: rotate` cast slot, which member to use")
    bake.add_argument("--label", default="leaf", help="suffix of the auto-named recipe file")
    bake.add_argument("--force", action="store_true",
                      help="bake even though the host fails the event type's geometry gate")
    bake.add_argument("--out", default=None,
                      help="where to write the frozen recipe (default: auto-named under "
                           "<recipes>/<LEAF>/<scenario>/<scene>/)")
    bake.add_argument("--recipe-id", default=None, help="override the derived recipe_id")
    bake.add_argument("--frozen-at", default=None, help="ISO date; defaults to today")
    bake.add_argument("--overwrite", action="store_true")
    bake.add_argument(
        "--no-insert", action="store_true",
        help="build the recipe with NO actors -- the host's own log is the "
             "scenario. The event type's event has to be there already: C-7 on a "
             "clip whose oncoming car is real does not need one inserted, and "
             "inserting anyway puts a second car in a lane that is already "
             "carrying the event.")
    bake.set_defaults(func=cmd_bake)

    verify = sub.add_parser("verify", help="load a frozen recipe and check its checksums")
    verify.add_argument("recipe")
    verify.set_defaults(func=cmd_verify)

    replay = sub.add_parser("replay", help="print the scenario_edits a recipe produces")
    replay.add_argument("recipe")
    replay.add_argument("--variant", default="e_plus", choices=list(VARIANTS))
    replay.set_defaults(func=cmd_replay)

    card = sub.add_parser("card", help="render the review card for a frozen recipe")
    card.add_argument("recipe")
    _add_host_args(card)
    card.add_argument("--out-dir", required=True)
    card.add_argument(
        "--cam-gif",
        default=None,
        help="camera-view gif from an eval render against the reconstruction",
    )
    card.add_argument("--no-topdown", action="store_true")
    card.set_defaults(func=cmd_card)

    calib = sub.add_parser(
        "calibrate-ground-z",
        help="measure recon-road drift across a data root and emit the calib registry",
    )
    calib.add_argument("--data-root", required=True)
    calib.add_argument("--max-scenes", type=int, default=None)
    calib.add_argument("--out", default=None, help="registry JSON to write/update")
    calib.set_defaults(func=cmd_calibrate_ground_z)

    assets = sub.add_parser(
        "assets", help="the asset registry: status, acquisition, composition, calibration"
    )
    assets.add_argument("--registry", default=None, help="registry YAML (defaults to the packaged one)")
    assets_sub = assets.add_subparsers(dest="assets_command", required=True)
    status = assets_sub.add_parser("status", help="what is on disk vs declared, per leaf")
    status.set_defaults(func=cmd_assets)
    acquire = assets_sub.add_parser("acquire", help="plan (or run) what puts an asset on disk")
    acquire.add_argument("key", nargs="?", default=None)
    acquire.add_argument("--event-type", "--leaf", dest="leaf", default=None, metavar="EVENT_TYPE",
                         help="plan every missing asset for one event type")
    acquire.add_argument(
        "--execute",
        action="store_true",
        help="actually run the runnable steps. Off by default: acquisition fetches "
        "third-party assets and writes into a shared library.",
    )
    acquire.set_defaults(func=cmd_assets)
    compose = assets_sub.add_parser("compose", help="build a composed asset from its parts")
    compose.add_argument("key")
    compose.set_defaults(func=cmd_assets)
    animate = assets_sub.add_parser(
        "animate",
        help="bake a gait pose bank so an inserted pedestrian walks, not slides",
    )
    animate.add_argument("key", help="registry key, or a path to a 3DGS PLY")
    animate.add_argument(
        "--out", default=None,
        help="bank directory (default: NAVSAFE_GAIT_BANK/<key>)",
    )
    animate.add_argument(
        "--phases", type=int, default=10,
        help="posed copies spanning one gait cycle. Each is inserted into the render "
        "server as its own track, in EVERY handoff window, so this multiplies the "
        "asset's VRAM cost by phases x windows — 10 on a 24 GB card is about three "
        "pedestrians' worth.",
    )
    animate.add_argument(
        "--motion", default="default",
        help="motion to bake from, as an example name under NAVSAFE_KIMODO's demo "
        "examples or an absolute path to a kimodo motion NPZ. The default is the "
        "shipped 10 s casual walk, which needs no model download.",
    )
    animate.add_argument(
        "--prune-opacity", type=float, default=0.0,
        help="drop gaussians fainter than this before baking. Trades the asset's haze "
        "for VRAM, which a bank spends once per phase per window; 0.2 keeps about "
        "two fifths of a harvested pedestrian.",
    )
    animate.set_defaults(func=cmd_assets_animate)
    calibrate = assets_sub.add_parser(
        "calibrate",
        help="measure a PLY's visual extent and solve one uniform scale — no eyeballing",
    )
    calibrate.add_argument("key", help="registry key, or a path to a 3DGS PLY")
    calibrate.add_argument(
        "--target-dims", type=float, nargs=3, metavar=("L", "W", "H"), default=None,
        help="target real-world dims in metres; defaults to the registry's canonical "
        "dims for the asset's family (families: table in registry.yaml)",
    )
    calibrate.add_argument(
        "--species", default=None,
        help="size against a SPECIES from the registry's `species_dims:` table (dog, "
        "cow, hippo, horse, …). Use for members of a heterogeneous family, where the "
        "family itself has no canonical size.",
    )
    calibrate.add_argument(
        "--family", default=None,
        help="family for the canonical-dims lookup when calibrating a bare PLY path",
    )
    calibrate.add_argument(
        "--opacity-min", type=float, default=0.1,
        help="gaussians whose rendered opacity is below this are haze: excluded from the "
        "measurement, and dropped from the written file unless --no-prune",
    )
    calibrate.add_argument(
        "--sigma-k", type=float, default=2.0,
        help="pad each gaussian by this many of its own sigma — the on-screen extent",
    )
    calibrate.add_argument(
        "--write", default=None, metavar="OUT_PLY",
        help="write the calibrated PLY here (never overwrites the source: recipes pin "
        "assets by sha256). Without it, measure and report only.",
    )
    calibrate.add_argument(
        "--no-prune", action="store_true",
        help="keep sub-threshold haze gaussians in the written file",
    )
    calibrate.add_argument(
        "--fit-axis", default="median", choices=("median", "length", "width", "height"),
        help="which dimension the scale must match EXACTLY. Default 'median' suits an "
        "asset whose proportions match the canonical body; use 'height' for people "
        "(harvested captures vary in girth and pose, not stature) and 'length' for "
        "vehicles and animals.",
    )
    calibrate.add_argument(
        "--auto-axes", action="store_true",
        help="pick (forward, up) by shape: try every proper orientation and keep the one "
        "whose length:width:height best matches the target. Harvested assets do not share "
        "one convention, so guessing per family puts some of them sideways.",
    )
    calibrate.add_argument(
        "--forward-axis", default=None, metavar="AXIS",
        help="which FILE axis points along the object's nose (+x -x +y -y +z -z). Give "
        "with --up-axis to re-orient an asset whose exporter used another convention "
        "into the pipeline's y-up/+x-forward frame; read them off the point cloud's "
        "principal axis when unsure.",
    )
    calibrate.add_argument(
        "--up-axis", default=None, metavar="AXIS",
        help="which FILE axis points at the sky (see --forward-axis)",
    )
    calibrate.add_argument(
        "--no-rebase", action="store_true",
        help="do not shift the visual base to y=0 (inserts need base-origin; only skip "
        "this for an asset that is deliberately not ground-mounted)",
    )
    calibrate.set_defaults(func=cmd_assets_calibrate)


