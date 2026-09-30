# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""``navsafe`` — build a NavSafe scenario, whichever way its leaf is built.

Two routes to the same artifact. Every leaf, mined or constructed, ends at a
**recipe**; evaluation consumes only recipes and never needs to know which
route produced one.

    leaves          what each leaf needs, and how it is selected
    mine            run a leaf's selection predicate over the seed table
    evidence        draw the BEV + expert trajectory behind a candidate
    qualify         which leaves a CONVERTED host can carry (geometry tier)
    probe/anchors   what a host offers: cross-sections, named landmarks
    bake            -> recipe.  `--leaf L --scenario-id ID`, both routes alike:
                                the leaf manifest says whether anything inserts
    handoff         the NUREC_GRPC_HANDOFF a multi-window scenario renders with
    verify/replay   read a frozen recipe back
    card            the reviewer's card for a recipe
    assets          registry: status / acquire / compose / calibrate

Rule of thumb for which route a leaf takes:

    the event is the ROAD          -> mine        (V-8, V-11, C-10)
    the event is INSERTED          -> construct   (C-7, I-3, R-2, R-3, R-4, V-10)

C-10 is mined rather than built because its consequence has to be traffic the
log already carries: the ego is the wrong-way driver, and a car inserted to be
hit by it is C-7 with a different label.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg

SEED_TSV = str(cfg.WORK / "scenes_500.tsv")
# `mine` writes here and `bake --scenario-id` reads here, so the two commands
# need no path between them.
CANDIDATES_DIR = cfg.WORK / "mine"
RECIPES_DIR = cfg.WORK / "recipes"


# ── selection ───────────────────────────────────────────────────────────────


def _load_scenarios(tsv: str, limit=None):
    from navsafe.benchmark.mining.leaf_miners import Scenario

    rows = []
    for line in Path(tsv).read_text().splitlines():
        scn = Scenario.from_tsv_line(line)
        if scn is not None:
            rows.append(scn)
    return rows[:limit] if limit else rows


def cmd_leaves(args) -> int:
    """Print the contract of every leaf (or one)."""
    from navsafe.benchmark.leaves import available, load_leaf

    names = [args.leaf] if args.leaf else available()
    for leaf in names:
        man = load_leaf(leaf)
        print(f"\n── {man.leaf}  {man.name}   [{man.tier}]")
        if man.summary:
            print("   " + man.summary.strip().replace("\n", "\n   "))
        if man.is_mined:
            miner = man.mine.get("miner", "(none)")
            print(f"   mine      {miner}  params={man.mine.get('params', {})}")
        if man.qualify:
            print(f"   qualify   {', '.join(man.qualify)}"
                  + (f"   (info: {', '.join(man.qualify_info)})" if man.qualify_info else ""))
        for slot in man.rule.cast:
            authored = slot.authored or {}
            print(f"   insert    {slot.slot}: {authored.get('template', '?')} on "
                  f"{authored.get('reference', 'ego_route')}"
                  f"  x{slot.count}  ({slot.assets})")
        print(f"   metrics   {', '.join(man.metrics) or '(none declared)'}")
        print(f"   review    {man.checklist}")
        if man.notes:
            print("   note      " + man.notes.strip().replace("\n", "\n             "))
    return 0


def cmd_mine(args) -> int:
    """Every scenario that can carry this leaf — pool tier and host tier."""
    from navsafe.benchmark.mining.mine import mine_leaf, write_rows

    params = {}
    for override in args.param or []:
        k, _, v = override.partition("=")
        try:
            params[k] = json.loads(v)
        except json.JSONDecodeError:
            params[k] = v

    rows = mine_leaf(args.leaf, tsv=args.tsv, limit=args.limit, params=params,
                     roots_glob=args.roots_glob, host_tier=not args.no_host_tier,
                     review=args.review, review_only=args.review_only)
    out = Path(args.out or CANDIDATES_DIR / f"{args.leaf}.jsonl")
    write_rows(rows, out)

    usable = [r for r in rows if r.ok]
    servable = [r for r in usable if r.trained_windows]
    pending = [r for r in rows if r.status == "needs_conversion"]
    picked = [r for r in rows if r.picked]

    print(f"\n{len(usable)}/{len(rows)} scenarios can carry {args.leaf}; "
          f"{len(servable)} have a reconstruction to render from.")
    if args.review:
        disputed = [r for r in rows if r.disputed]
        print(f"{len(picked)} chosen by a reviewer; "
              f"{len([r for r in picked if r.ok])} of those also pass the geometry gate.")
        if disputed:
            print(f"\n{len(disputed)} row(s) where the reviewer and the geometry DISAGREE "
                  f"— look at these before trusting either:")
            for row in disputed[: args.top]:
                print("  " + row.describe())
    agreed = [r for r in (picked or servable or usable) if not r.disputed]
    if agreed:
        print(f"\nbest {min(len(agreed), args.top)}:")
        for row in agreed[: args.top]:
            print("  " + row.describe())
    if pending:
        print(f"\n{len(pending)} passed the raw-map predicate but are not converted yet — "
              f"the geometry gate cannot run on them:")
        for row in pending[:5]:
            print(f"  {row.token}  {row.note}")
    orphans = [r for r in rows if r.status == "not_in_seed_table"]
    if orphans:
        print(f"\n{len(orphans)} reviewer pick(s) are not in {args.tsv}:")
        for row in orphans[:10]:
            print(f"  {row.token}  {row.note}")
    broken = [r for r in rows if r.status == "host_error"]
    if broken:
        print(f"\n{len(broken)} host(s) exist but would not open:")
        for row in broken[:5]:
            print(f"  {row.scenario_id}  {row.note}")
    if not usable:
        print("\n  Nothing qualified. That is a finding about the pool, not a bug to route "
              "around: widen the seed table or record the gap, do NOT relax the predicate "
              "to make a neighbouring scenario pass.")
    print(f"\ncandidates -> {out}")
    if servable or usable:
        print(f"next:  navsafe bake --leaf {args.leaf} --scenario-id "
              f"{(servable or usable)[0].scenario_id}")

    if args.bev_dir:
        from navsafe.benchmark.mining.bev_preview import render_previews

        clips = [r.scenario_id.removesuffix("_20s") for r in (usable or rows)]
        print(f"\n[bev] rendering {len(clips)} preview(s) -> {args.bev_dir}")
        _, failures = render_previews(clips, Path(args.bev_dir), workers=args.bev_workers)
        if failures:
            print(f"[bev] {len(failures)} failed; the rest are in {args.bev_dir}")
    return 0


def cmd_evidence(args) -> int:
    """Draw the BEV + expert trajectory behind a mined candidate."""
    from navsafe.benchmark.mining.evidence import draw_candidate

    row = None
    for line in Path(args.candidates).read_text().splitlines():
        d = json.loads(line)
        if d.get("token") == args.token:
            row = d
            break
    if row is None:
        print(f"token {args.token!r} not in {args.candidates}")
        return 2
    out = draw_candidate(row, args.out, leaf=args.leaf)
    print(f"evidence figure -> {out}")
    print("Look before believing: the numbers said 'merge' twice on scenes that were a "
          "junction turn-fan and a road leaving a T-junction.")
    return 0


# ── mined -> recipe ─────────────────────────────────────────────────────────



def cmd_handoff(args) -> int:
    """Print the NUREC_GRPC_HANDOFF for a scenario's reconstructions.

    A 20 s scenario is trained as four 5 s reconstructions, and the renderer
    switches between them per frame. ALWAYS arm all four, even for a short
    render: `--eval-frames` is a knob a user turns between runs, and a handoff
    covering only the windows the current run reaches renders the road from
    the wrong model the moment someone lengthens it.

    The pick is position-based — the live ego is projected onto the logged
    track and the matched frame's timestamp chooses the clip — so a policy
    that runs ahead of or behind the log still gets the road it is on.
    """
    from navsafe.benchmark.mining.leaf_miners import Scenario

    # A host arrives one of two ways, and the hand-off has to answer for both
    # or the downloaded ones are skipped in silence. A MINED host was cut here:
    # its window is a row in the seed table and its sub-clips carry the
    # `nurec_origin_offset.json` sidecars the loop below reads. A DOWNLOADED
    # host is a published bundle: the seed table has never heard of it, it
    # ships no `clips/` tree at all, and everything the hand-off needs is in
    # its own `manifest.json` beside an `offsets/` directory. That case already
    # has one resolver -- `handoff_for_bundle` -- so it is called rather than
    # rebuilt; two eval drivers had each grown their own copy of the string.
    bundle = Path(args.recon_root, f"{args.token}_20s")
    if args.t0 is None and (bundle / "manifest.json").is_file():
        from navsafe.benchmark.eval.bundle import handoff_for_bundle
        print(f'export NUREC_GRPC_HANDOFF="{handoff_for_bundle(bundle)}"')
        return 0

    t0 = args.t0
    if t0 is None:
        for line in Path(args.tsv).read_text().splitlines():
            scn = Scenario.from_tsv_line(line)
            if scn is not None and scn.token == args.token:
                t0 = scn.t0
                break
        if t0 is None:
            print(f"token {args.token!r} not in {args.tsv}; pass --t0 explicitly")
            return 2

    parts, missing = [], []
    for i in range(args.windows):
        clip = f"{args.token}s{i + 1}"
        offsets = Path(args.recon_root, clip, "clips", clip, "nurec_origin_offset.json")
        if not offsets.is_file():
            missing.append(f"{clip}: no origin-offset sidecar at {offsets}")
            continue
        # "Is this window trained" is the question worth warning about, and
        # cfg.recon_usdz answers it in both corpus layouts. It used to be asked
        # of a hand-maintained symlink pool instead, which serve-grpc stopped
        # using in favour of globbing the training tree directly -- so every
        # token warned "not in the served pool" for four windows that were in
        # fact served, and the warning taught readers to ignore it.
        if cfg.recon_usdz(clip, Path(args.recon_root)) is None:
            missing.append(f"{clip}: not reconstructed yet (no last.usdz under "
                           f"{Path(args.recon_root, clip)}) — serve-grpc can only "
                           f"serve what exists when it starts")
        w0 = int(t0) + i * args.window_us
        parts.append(f"{clip},{offsets},{w0},{w0 + args.window_us}")
    for m in missing:
        print(f"# WARNING {m}", file=sys.stderr)
    if not parts:
        return 2
    print(f'export NUREC_GRPC_HANDOFF="{";".join(parts)}"')
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="navsafe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("leaves", help="what each leaf needs, and how it is built")
    p.add_argument("--leaf", default=None)
    p.set_defaults(func=cmd_leaves)

    p = sub.add_parser("mine", help="every scenario that can carry a leaf")
    p.add_argument("--leaf", required=True)
    p.add_argument("--tsv", default=SEED_TSV, help="seed table (token/log/t0/t1/types/leaves)")
    p.add_argument("--out", default=None,
                   help=f"candidates .jsonl (default: {CANDIDATES_DIR}/<LEAF>.jsonl, which "
                        f"is where `navsafe bake --scenario-id` looks)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top", type=int, default=10, help="how many hits to print")
    p.add_argument("--param", action="append", default=None, metavar="K=V",
                   help="override a manifest mine param (repeatable, JSON values)")
    p.add_argument("--roots-glob", action="append", default=None,
                   metavar="GLOB[:family[:rig]]",
                   help="where converted hosts live (repeatable); overrides the defaults")
    p.add_argument("--no-host-tier", action="store_true",
                   help="raw-map predicate only — do not open any converted host")
    p.add_argument("--review", default=None, metavar="CSV",
                   help="a reviewer's exported sheet. Their verdict is stamped onto each "
                        "row as its own field — it does NOT override the geometry gate, "
                        "because the two answer different questions and the rows where "
                        "they disagree are the ones worth looking at.")
    p.add_argument("--review-only", action="store_true",
                   help="sweep only the reviewer's picks. The host tier opens every "
                        "converted Arrow root, so a full sweep spends half an hour "
                        "re-deriving a shortlist that already exists.")
    p.add_argument("--bev-dir", default=None,
                   help="also render one BEV preview per scenario found, into this dir")
    p.add_argument("--bev-workers", type=int, default=4)
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("evidence", help="draw the BEV + expert trajectory behind a candidate")
    p.add_argument("--leaf", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_evidence)


    p = sub.add_parser("handoff", help="print NUREC_GRPC_HANDOFF for a scenario's recons")
    p.add_argument("--token", required=True)
    p.add_argument("--tsv", default=SEED_TSV)
    p.add_argument("--t0", type=int, default=None, help="scenario start us (else read from --tsv)")
    p.add_argument("--windows", type=int, default=4, help="how many 5 s recons (default: all four)")
    p.add_argument("--window-us", type=int, default=5_000_000)
    p.add_argument("--recon-root", default=str(cfg.CORPUS))
    p.set_defaults(func=cmd_handoff)

    # The construction-tier verbs keep their existing implementations.
    from navsafe.benchmark.editing import cli as editing_cli

    editing_cli.attach_subcommands(sub)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
