"""Stage 2 (CPU): a NavSafe scenario -> py123d Arrow scenario source.

Closed-loop evaluation needs the scenario (ego / boxes / map / route) as an
Arrow dataset, separate from the rendered pixels.  Reconstructions carry pixels
only; without this Arrow there is nothing to drive.

Two ways in, because scenarios arrive two ways:

``--seed <dir>``    a mined seed with a ``seed.json``.  Converted over the
                    seed's *reconstruction* window -- a superset of the scored
                    window -- so the evaluator can slice warm-up and scored
                    ranges out of one dataset without a second conversion.
``--token <tok>``   a **stitched host** from the seed table: one nuPlan token
                    whose four 5 s reconstructions are ``<tok>s1..s4``.  These
                    have no ``seed.json`` -- they are chosen by a reviewer
                    watching renders, not by the mining score -- and the
                    conversion used to exist only as a shell history.  The log
                    name and window come from ``scenes_500.tsv``; the Arrow
                    lands in ``<corpus>/<tok>_20s/arrow``, which is the layout
                    ``leaves/hosts.py`` globs for, and the scene is named
                    ``<tok>_20s`` so it matches the render server's scene ids.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg


PY123D_SPLIT = {"test": "nuplan_test", "val": "nuplan_val", "mini": "nuplan-mini_test"}


def _from_seed_table(token: str, *, tsv: str, corpus: str, nuplan_root: str):
    """Resolve a stitched host from the seed table, as if it had a seed.json.

    A stitched host is one nuPlan token reconstructed as four consecutive 5 s
    windows, and its Arrow covers the whole 20 s so the renderer can hand off
    between them inside one episode. The scene is named ``<token>_20s`` because
    that string has to match on three sides at once: the directory
    ``leaves/hosts.py`` globs for, the Arrow's own log name, and the scene id
    the render server answers to.

    Raises:
        LookupError: the token is not in the table. That is a finding about the
            TABLE, not about the scenario, and not something to paper over by
            guessing a window. The usual cause is that the token belongs to the
            other corpus: ``scenes_500.tsv`` covers ``navsafe_5s_500`` only, and
            a navhard host is ``<token>h1`` under ``NAVHARD_CORPUS`` — already
            converted, with nothing for this command to do.
    """
    for line in Path(tsv).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].strip() == token:
            log_name, t0, t1 = parts[1].strip(), int(parts[2]), int(parts[3])
            break
    else:
        raise LookupError(
            f"{token} is not in {tsv}, which covers the navsafe_5s_500 corpus only. "
            f"If this is a navhard scenario its host is <token>h1 and is already "
            f"converted — check `navsafe qualify --data-root $NAVSAFE_NAVHARD_CORPUS/"
            f"{token}h1/arrow` before converting anything. Otherwise add a row here "
            f"(token, log_name, t0_us, t1_us), or point --tsv at the table that has it."
        )
    scene = f"{token}_20s"
    out = Path(corpus) / scene / "arrow"
    return (
        scene,
        {"split": "test", "log_name": log_name},
        {"recon_t0_us": t0, "recon_t1_us": t1},
        out,
        out / "logs" / "nuplan_test" / scene,
        Path(nuplan_root) / "nuplan-v1.1" / "splits" / "test" / f"{log_name}.db",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", help="a mined seed directory or its seed.json")
    source.add_argument("--token", help="a stitched host's nuPlan token, from the seed table")
    ap.add_argument("--tsv", default=str(cfg.WORK / "scenes_500.tsv"),
                    help="the seed table --token is resolved against")
    ap.add_argument("--corpus", default=str(cfg.CORPUS),
                    help="where --token writes its Arrow (<corpus>/<token>_20s/arrow)")
    ap.add_argument("--work", default=str(cfg.WORK))
    ap.add_argument("--nuplan-root", default=str(cfg.NUPLAN_ROOT))
    ap.add_argument("--navsafe", default=str(cfg.NAVSAFE_ROOT))
    # Same reason as run_ncore.py: the devkit ORM hammers the log db with
    # millions of small page reads, which is pathological on CephFS.
    ap.add_argument("--nuplan-local", default=str(cfg.LOCAL_DB))
    ap.add_argument("--convert-python", default=None,
                    help="interpreter of the py123d (3.11) conversion venv; the "
                         "sibling py123d-conversion is what actually runs. "
                         "Defaults to $NAVSAFE_CONVERT_PYTHON, then PATH.")
    ap.add_argument("--threads", type=int, default=4,
                    help="ray workers. Must not exceed the container's CPU limit — see "
                         "the note where this is passed.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.token:
        try:
            tok, prov, win, out, log_out, db_src = _from_seed_table(
                args.token, tsv=args.tsv, corpus=args.corpus, nuplan_root=args.nuplan_root)
        except LookupError as exc:
            print(f"[arrow] {exc}", file=sys.stderr)
            return 2
    else:
        p = Path(args.seed)
        seed = json.loads((p / "seed.json" if p.is_dir() else p).read_text())
        tok = seed["seed_id"]
        prov, win = seed["provenance"], seed["window"]
        out = Path(seed["artifacts"]["arrow"])          # shared py123d root
        log_out = Path(seed["artifacts"]["arrow_log"])  # this seed's logs
        db_src = Path(seed["source_paths"]["db"])

    if log_out.exists() and any(log_out.glob("*.arrow")) and not args.force:
        print(f"[arrow] {tok}: already converted -> {log_out}")
        return 0

    split = PY123D_SPLIT[prov["split"]]
    split_dir = {"nuplan_test": "test", "nuplan_val": "trainval",
                 "nuplan-mini_test": "mini"}[split]
    staged = (Path(args.nuplan_local) / "nuplan-v1.1" / "splits" / split_dir
              / f"{prov['log_name']}.db")
    if not staged.exists():
        src = db_src
        staged.parent.mkdir(parents=True, exist_ok=True)
        print(f"[arrow] staging {src.name} -> {staged.parent} "
              f"({src.stat().st_size / 1e6:.0f} MB)", flush=True)
        shutil.copy2(src, staged)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": args.navsafe,
        "PY123D_DATA_ROOT": str(out),
        "NAVSAFE_WORK": args.work,
        "NUPLAN_ROOT": args.nuplan_root,
        "NUPLAN_MAPS_ROOT": f"{args.nuplan_root}/maps",
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "HYDRA_FULL_ERROR": "1",
    })
    scenes = f'[[{prov["log_name"]}, {tok}, {win["recon_t0_us"]}, {win["recon_t1_us"]}]]'
    # py123d needs Python 3.11 and the pod's main venv is 3.12, so the pod
    # provisions a SECOND interpreter and names it in NAVSAFE_CONVERT_PYTHON.
    # This used to bare-call `py123d-conversion` off PATH, which put
    # /root/navsafe-venv/bin first and ran the conversion in the wrong venv --
    # silently, because both have the entry point. It surfaced as hydra
    # reporting `dataset=nuplan-navsafe` unavailable while the config file sat
    # in the other venv's package. Resolve the sibling entry point explicitly
    # when the variable is set, and say which one is being used.
    conv = shutil.which("py123d-conversion") or "py123d-conversion"
    convert_python = args.convert_python or os.environ.get("NAVSAFE_CONVERT_PYTHON", "")
    if convert_python:
        sibling = Path(convert_python).with_name("py123d-conversion")
        if sibling.exists():
            conv = str(sibling)
        else:
            print(f"[arrow] WARNING: NAVSAFE_CONVERT_PYTHON={convert_python} has no "
                  f"py123d-conversion beside it; falling back to {conv}", file=sys.stderr)
    print(f"[arrow] converter: {conv}", flush=True)
    cmd = [
        conv,
        "dataset=nuplan-navsafe",
        f"dataset.parser.split={split}",
        f"dataset.parser.nuplan_data_root={args.nuplan_local}",
        f"dataset.parser.scenes={scenes}",
        # Ray sizes its pool from ``os.cpu_count()``, which reports the NODE's
        # cores, not the container's cgroup limit — 32 against a 4-CPU pod. The
        # oversubscribed workers then deadlock during import rather than run
        # slowly, so this is a correctness knob, not a tuning one.
        f"execution.threads_per_node={args.threads}",
    ]
    print(f"[arrow] {tok}  {' '.join(cmd)}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print(f"[arrow] {tok}: conversion failed (rc={rc})", file=sys.stderr)
        return rc
    made = sorted(str(q) for q in log_out.glob("*.arrow"))
    maps = sorted(str(q) for q in (out / "maps").rglob("*.arrow"))
    print(f"[arrow] {tok}: {len(made)} log arrows under {log_out}")
    for q in made:
        print("    log ", q)
    print(f"[arrow] {tok}: {len(maps)} map arrows under {out / 'maps'}")
    for q in maps[:4]:
        print("    map ", q)
    if not made:
        print(f"[arrow] {tok}: no log arrows written", file=sys.stderr)
        return 1
    if not maps:
        print(f"[arrow] {tok}: NO MAP arrows -- closed-loop eval needs the map",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
