#!/usr/bin/env python3
"""Author the red light V-1 needs into each published bundle's manifest.

V-1 Red-Light is scored as a hold behind the stop line, and the corpus cannot
pose that question: the logged states on the published bundles are
``LANE_STATE_GO`` / ``LANE_STATE_UNKNOWN``, never STOP. The hold rules therefore
only apply to a bundle whose ``manifest.json`` declares a ``signal_override``
(see :mod:`navsafe.benchmark.signal_override`), and this script works out what
that declaration should be, per bundle, from the bundle's own geometry.

Two things have to be got right, and both were learned the hard way on
``05d0a1a763fc5334``:

* **Redden the whole approach, not one connector.** The evaluator exempts a
  crossing while any non-red signalized connector still offers a heading-aligned
  way through (``_green_way_through``). With one lane reddened, SimWAM drove
  through and the ``TL`` column never dropped. So the override names every
  signalized connector that shares the ego's stop line.
* **The ego must start behind the line.** Where the connector's entry projects
  to the start of the route there is no approach to hold through, and the
  scenario cannot host the hold at any threshold. ``00c1e4eb4a045f20`` is that
  case; it needs a re-cut window, not an override.

Only ``arrow/`` and ``manifest.json`` are fetched -- ~55 MB a bundle, against
8.4 GB for the whole thing -- because the geometry is all that is being read.

    python scripts/tools/navsafe_bake_signal_override.py                # report
    python scripts/tools/navsafe_bake_signal_override.py --write        # patch
    python scripts/tools/navsafe_bake_signal_override.py --publish      # + HF
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

DATASET = "c13752hz/NavSafe"
SPLIT = "full_test"
DEFAULT_WORK = Path(os.environ.get("NAVSAFE_WORK", str(Path.home() / ".cache/navsafe"))) / "v1_rebake"

#: A connector counts as sharing the ego's stop line when its entry projects
#: within this much of the resolved line. Connectors fanning out of one approach
#: (straight / left / right) share the line and overlap for their first metres,
#: which is exactly the set `_green_way_through` can exempt a crossing with.
SIBLING_WINDOW_M = 8.0

#: The ego must have at least this much road in front of the line, or there is
#: no approach to hold through and no threshold can rescue the scenario.
MIN_APPROACH_M = 2.0

#: Warm-up frames the policy does NOT drive (`run_bundle_eval.sh policy`). The
#: distances that decide feasibility are measured from the HAND-OFF pose, not
#: from the window start: 20 frames of logged replay at 6.4 m/s carry the ego
#: ~12.8 m, which is past several of these stop lines. Measured from frame 0,
#: this script called scenarios usable that the policy never gets a chance at.
WARMUP_FRAMES = 20

#: nuPlan's comfort acceleration bound (``epdms_trajectory_scorer_fast.
#: MAX_ACCEL``), used here as the braking authority a policy can be asked for
#: without the manoeuvre itself costing comfort. A stop line closer than
#: ``v**2 / (2 * MAX_DECEL_MS2)`` cannot be honoured by any policy, and a row
#: where every policy scores zero measures the scenario, not the policies.
MAX_DECEL_MS2 = 4.89


def _tokens_for_leaf(leaf: str) -> list[str]:
    """Published tokens whose taxonomy leaf is ``leaf``, from the cached index."""
    index = Path(os.environ.get("NAVSAFE_WORK", str(Path.home() / ".cache/navsafe"))) / "table2_cover/leaf_index.jsonl"
    if not index.is_file():
        raise SystemExit(
            f"{index} not found — run scripts/tools/navsafe_table2_cover.py "
            f"first, it builds the leaf index this reads")
    out = []
    for line in index.read_text().splitlines():
        row = json.loads(line)
        if row["split"] == SPLIT and row["leaves"] == [leaf]:
            out.append(row["token"])
    return sorted(out)


def _is_stop_sign(meta: dict) -> bool:
    from navsafe.benchmark.scenario_rules import STOP_SIGN_TYPES

    return bool({str(t) for t in (meta.get("scenario_types") or [])} & STOP_SIGN_TYPES)


def fetch(token: str, work: Path) -> Path:
    """Bundle directory holding ``arrow/`` and ``manifest.json``.

    A bundle already on local disk is used as-is; the point of this script is
    the geometry, and re-downloading 55 MB to re-read it would be waste.
    """
    from navsafe.data_paths import data_root
    for local in (data_root() / "full_test" / token,):
        if (local / "arrow").is_dir() and (local / "manifest.json").is_file():
            return local

    from huggingface_hub import HfApi, hf_hub_download

    dest = work / token
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in HfApi().list_repo_files(DATASET, repo_type="dataset")
             if f.startswith(f"{SPLIT}/{token}/")
             and (f.endswith("manifest.json") or f"/{token}/arrow/" in f)]
    for f in files:
        rel = f.split(f"{SPLIT}/{token}/", 1)[1]
        target = dest / rel
        if target.is_file():
            continue
        src = hf_hub_download(DATASET, f, repo_type="dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(src).read_bytes())
    return dest


def feasibility(stop_s: float, handoff_arc: float, v: float) -> tuple[bool, str]:
    """Can a policy be asked to hold behind this line? Pure arithmetic.

    Both tests are about the HAND-OFF, and getting that wrong is what made this
    script's first answer wrong: measured from the window start it called six
    bundles usable, five of which put the ego past the line before the policy
    ever drove. Twenty warm-up frames at 6.4 m/s are 12.8 m of road.
    """
    remaining = float(stop_s - handoff_arc)
    if remaining < 0:
        return False, (f"the ego is already {-remaining:.1f} m past the stop "
                       f"line when the policy takes over (line at {stop_s:.1f} m, "
                       f"hand-off at {handoff_arc:.1f} m): nothing to hold "
                       f"behind. Re-cut the window.")
    if remaining < MIN_APPROACH_M:
        return False, (f"only {remaining:.1f} m in front of the line at "
                       f"hand-off, under the {MIN_APPROACH_M:.0f} m an approach "
                       f"needs. Re-cut the window.")
    braking = v * v / (2.0 * MAX_DECEL_MS2)
    if braking > remaining:
        return False, (f"unstoppable: {v:.1f} m/s at hand-off needs "
                       f"{braking:.1f} m to stop within nuPlan's comfort bound "
                       f"and only {remaining:.1f} m remain. Every policy would "
                       f"fail this row, which measures the scenario.")
    return True, ""


def resolve(bundle: Path) -> dict[str, Any]:
    """What override this bundle needs, or why it cannot have one."""
    from navsafe.benchmark.trace import from_eval as fe
    from navsafe.benchmark.trace.writer import lane_centerlines

    arrow = bundle / "arrow"
    sd = fe.scenario_from_arrow(arrow)
    ego_id = (sd.get("metadata", {}) or {}).get("sdc_id")
    route = np.asarray(sd["tracks"][ego_id]["state"]["position"],
                       dtype=np.float64)[:, :2]
    lane_center = lane_centerlines(sd)
    stop_s = fe.stopline_arc(sd, route, lane_center)

    arc = fe._arc(route)
    handoff = min(WARMUP_FRAMES, len(route) - 1)
    # Speed the policy inherits, and how much road is left in front of it. Both
    # are properties of the HAND-OFF, not of the window start.
    v = (float(np.linalg.norm(route[handoff] - route[handoff - 1]) / 0.1)
         if handoff >= 1 else 0.0)
    out: dict[str, Any] = {
        "token": bundle.name, "stop_line_arc_m": stop_s,
        "ego_speed_handoff_ms": round(v, 2),
        "signalled_lanes": len(sd.get("dynamic_map_states") or {}),
    }
    if stop_s is None:
        out["skip"] = ("no signalled connector resolves against the route — "
                       "nothing to redden")
        return out

    out["remaining_at_handoff_m"] = round(float(stop_s - arc[handoff]), 1)
    out["braking_distance_m"] = round(v * v / (2.0 * MAX_DECEL_MS2), 1)
    ok, why = feasibility(float(stop_s), float(arc[handoff]), v)
    if not ok:
        out["skip"] = why
        return out

    siblings = []
    for lane_id in (sd.get("dynamic_map_states") or {}):
        centre = lane_center.get(str(lane_id))
        if centre is None or len(centre) == 0:
            continue
        centre = np.asarray(centre, dtype=np.float64)[:, :2]
        gap = float(np.min(np.linalg.norm(
            route[:, None, :] - centre[None, :, :], axis=2)))
        if gap > fe.STOPLINE_ROUTE_TOL_M:
            continue
        s, _ = fe._arc_at(route, arc, centre[0])
        if abs(s - stop_s) <= SIBLING_WINDOW_M:
            siblings.append(str(lane_id))
    if not siblings:
        out["skip"] = "stop line resolved but no connector shares it (bug?)"
        return out
    out["override"] = {"lane": sorted(siblings),
                       "state": "LANE_STATE_STOP", "frames": "all"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", nargs="*", help="default: every V-1 bundle")
    ap.add_argument("--leaf", default="V-1")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--write", action="store_true",
                    help="patch the local manifest.json (a .pre_v1hold backup "
                         "is kept)")
    ap.add_argument("--publish", action="store_true",
                    help="also commit the patched manifests to the HF dataset "
                         "(needs a write-scoped token)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    tokens = args.tokens or _tokens_for_leaf(args.leaf)
    print(f"{len(tokens)} {args.leaf} bundle(s)\n")

    results, patched = [], []
    for token in tokens:
        bundle = fetch(token, args.work)
        meta = json.loads((bundle / "manifest.json").read_text())
        if _is_stop_sign(meta.get("scenario_meta") or {}):
            print(f"{token}  SKIP  stop-sign scenario: default goal and success "
                  f"apply, no hold to author")
            continue
        r = resolve(bundle)
        results.append(r)
        if "skip" in r:
            print(f"{token}  SKIP  {r['skip']}")
            continue
        lanes = r["override"]["lane"]
        print(f"{token}  {r['remaining_at_handoff_m']:5.1f} m to the line at "
              f"hand-off, {r['ego_speed_handoff_ms']:4.1f} m/s "
              f"(needs {r['braking_distance_m']:.1f} m), redden "
              f"{len(lanes)}: {' '.join(lanes)}")
        if args.write:
            path = bundle / "manifest.json"
            backup = path.with_suffix(".json.pre_v1hold")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
            meta["signal_override"] = r["override"]
            path.write_text(json.dumps(meta, indent=1))
            patched.append((token, path))

    usable = [r for r in results if "skip" not in r]
    print(f"\n{len(usable)} of {len(results)} can host the hold")
    if args.write:
        print(f"patched {len(patched)} manifest(s)")

    if args.publish:
        if not patched:
            print("nothing to publish (pass --write too)")
            return 1
        from huggingface_hub import CommitOperationAdd, HfApi

        ops = [CommitOperationAdd(f"{SPLIT}/{t}/manifest.json", str(p))
               for t, p in patched]
        HfApi().create_commit(
            DATASET, ops, repo_type="dataset",
            commit_message=f"V-1: author the red light in {len(ops)} manifests")
        print(f"published {len(ops)} manifest(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
