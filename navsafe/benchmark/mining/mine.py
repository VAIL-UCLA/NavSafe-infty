# Copyright (c) 2022-2026, The NexusSim Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Every scenario that can carry a leaf, in one pass.

Selection has two tiers, and they used to be two commands with a stub between
them. They ask the same question of different sources:

* the **pool tier** reads the RAW nuPlan log and map (``mine.miner`` in the
  leaf manifest). It is cheap and needs no conversion, so it can sweep the
  whole seed table — but it measures the map, not the scenario the eval will
  drive.
* the **host tier** runs the leaf's geometry predicates (``qualify`` /
  ``qualify_info``) on a CONVERTED host, in the ego's own frame. It is
  authoritative and costs an Arrow conversion, so it only runs where one
  already exists.

Running them separately meant a candidate's status was split across a JSONL
and a terminal scrollback, and the bridge between them was a miner that only
printed "now go run `navsafe qualify`". Here one row carries both tiers and one
``status`` says what to do next — including ``needs_conversion``, which is a
finding about the pool, not a failure.

A leaf with no ``qualify:`` gate (R-3, R-4: a dart-out works from any shoulder)
is not an error and is not a free pass either: every converted host is returned
with ``no_gate``, ranked by whatever ``qualify_info`` reports, so the choice is
made on numbers rather than on which token someone remembered.

There is a THIRD tier, and it is a person: whether a clip reads as the scenario
is not a question either predicate can answer, and it is settled by watching
renders. ``--review <csv>`` reads that verdict (see ``mining/review.py``) and
stamps it onto the row as its OWN field rather than folding it into ``status``.
Two axes, deliberately: "the reviewer chose this for R-2 but it has no lane a
cyclist belongs in" is a finding about one of the two, and a single verdict
would hide which.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# status values, worst to best
STATUS_ORDER = ("rejected_pool", "rejected_host", "host_error", "not_in_seed_table",
                "needs_conversion", "no_gate", "qualified")


@dataclass
class MineRow:
    """One scenario's verdict for one leaf, with both tiers' evidence."""

    leaf: str
    scenario_id: str
    token: str
    status: str
    log: str = ""
    t0: int = 0
    t1: int = 0
    pool: Dict[str, Any] = field(default_factory=dict)
    host: Dict[str, Any] = field(default_factory=dict)
    windows: List[str] = field(default_factory=list)
    trained_windows: List[str] = field(default_factory=list)
    event_t_offset_s: List[float] = field(default_factory=list)
    min_eval_frames: int = 0
    note: str = ""
    #: The reviewer's verdict, from ``--review``. Empty when the sheet does not
    #: assign this scenario to this leaf — which is not the same as a rejection.
    human: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("qualified", "no_gate")

    @property
    def picked(self) -> bool:
        """A person watched this and chose it for this leaf."""
        return bool(self.human.get("selected"))

    @property
    def disputed(self) -> bool:
        """The reviewer and the geometry disagree — the row worth looking at."""
        return bool(self.human) and self.picked != self.ok

    def describe(self) -> str:
        mark = {"qualified": "HIT ", "no_gate": "ok  ", "needs_conversion": "conv",
                "host_error": "ERR "}.get(self.status, "    ")
        human = ""
        if self.human:
            human = f" [reviewer:{self.human.get('keep') or '?'}"
            human += "!" if self.disputed else ""
            human += (f" {self.human['note']}" if self.human.get("note") else "") + "]"
        return (f"{mark}{self.scenario_id:<24} {self.status:<16} "
                f"recon={','.join(self.trained_windows) or '-':<12}{human} {self.note}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for derived in ("ok", "picked", "disputed"):
            d.pop(derived, None)
        return d


def _event_timing(evidence: Dict[str, Any], t0: int, dt_s: float = 0.1):
    """When inside the scenario the predicate fired, and the eval length that reaches it.

    An eval shorter than the event scores a clean pass on an episode that never
    got there — the single most misleading way to run one of these. This is
    lifted out of the old ``bake-mined`` because it is a property of the
    SELECTION (where the event is), not of the recipe.
    """
    hits = evidence.get("hits") or []
    offsets = [float(h["t_offset_s"]) for h in hits if h.get("t_offset_s") is not None]
    if not offsets:
        return [], 0
    return sorted(offsets), int(round(max(offsets) / max(dt_s, 1e-6))) + 1


def _host_tier(man, host, *, leaf: str) -> Dict[str, Any]:
    """Run the leaf's geometry predicates on a converted host."""
    from navsafe.benchmark.editing.ground_z import resolve_z_to_ground
    from navsafe.benchmark.editing.host import load_host_scenario
    from navsafe.benchmark.editing.placement.probe import HostProbe
    from navsafe.benchmark.editing.qualify import qualify_host

    z, _ = resolve_z_to_ground(None, host.data_root)
    # The Arrow's own scene id is a py123d UUID, not the directory name the
    # user types — one converted root holds exactly one scene, so take it by
    # index and keep the two identities distinct.
    sd, scene_id = load_host_scenario(host.data_root, scene_id=None, scene_index=0,
                                      require_map=True)
    probe = HostProbe(sd, ego_z_to_ground_m=z)
    verdict = qualify_host(probe, scene=scene_id, leaves=[leaf])[0]
    checks, info = [], []
    for check in verdict.checks:
        row = {"name": check.name.replace(" (info)", ""), "ok": bool(check.ok),
               "evidence": getattr(check, "evidence", "")}
        (info if check.name.endswith("(info)") else checks).append(row)
    return {"data_root": host.data_root, "cam_height": host.cam_height,
            "family": host.family, "arrow_scene": scene_id,
            "ok": bool(verdict.ok), "checks": checks, "info": info}


def mine_leaf(leaf: str, *, tsv: str, limit: Optional[int] = None,
              params: Optional[Dict[str, Any]] = None,
              roots_glob: Optional[List[str]] = None,
              host_tier: bool = True,
              review: Optional[str] = None,
              review_only: bool = False) -> List[MineRow]:
    """Every scenario that can carry ``leaf``, across all three tiers.

    Args:
        review: a reviewer's exported sheet (see ``mining/review.py``). Its
            verdict is stamped onto each matching row, and a token the reviewer
            assigned but the seed table does not carry gets a row of its own —
            "the person picked a scenario this sweep cannot even see" is a
            finding, and silently dropping it would look like a rejection.
        review_only: restrict the sweep to the reviewer's picks. Once a person
            has shortlisted 11 scenarios there is nothing to learn from opening
            the other 396 — and the host tier opens each converted Arrow root,
            so a full sweep costs half an hour to re-derive a list that already
            exists.
    """
    from navsafe.benchmark.leaves import load_leaf
    from navsafe.benchmark.leaves.hosts import find_hosts
    from navsafe.benchmark.mining.leaf_miners import Scenario, get_miner

    man = load_leaf(leaf)
    hosts = find_hosts(roots_glob) if host_tier else {}
    by_token = {h.token: h for h in hosts.values()}

    picks: Dict[str, Any] = {}
    if review:
        from navsafe.benchmark.mining.review import review_for

        picks = review_for(review, leaf)
        logger.info("[mine] %s · review tier: %d verdict(s) from %s",
                    leaf, len(picks), Path(review).name)

    scenarios: List[Scenario] = []
    for line in Path(tsv).read_text().splitlines():
        scn = Scenario.from_tsv_line(line)
        if scn is not None:
            scenarios.append(scn)
    if review_only:
        if not picks:
            raise ValueError(
                f"--review-only needs a review that assigns something to {leaf}; "
                f"{review or '(no --review given)'} assigns none.")
        wanted = set(picks)
        scenarios = [s for s in scenarios if s.token in wanted]
        logger.info("[mine] %s · review-only: %d of the seed table's scenarios are the "
                    "reviewer's picks", leaf, len(scenarios))
    if limit:
        scenarios = scenarios[:limit]

    # ── pool tier ──────────────────────────────────────────────────────────
    miner_name = (man.mine or {}).get("miner", "")
    if miner_name:
        merged = dict((man.mine or {}).get("params") or {})
        merged.update(params or {})
        logger.info("[mine] %s · pool tier: %s over %d scenarios · %s",
                    leaf, miner_name, len(scenarios), merged)
        candidates = get_miner(miner_name)(scenarios, **merged)
    else:
        # No raw-map predicate: the leaf's gate (if any) is geometric, so every
        # seed is a pool-tier pass and the host tier decides.
        logger.info("[mine] %s · no pool-tier miner declared; every seed defers to the "
                    "host tier", leaf)
        from navsafe.benchmark.mining.leaf_miners import Candidate
        candidates = [Candidate(s, True, "(none)", {}, [], [], "no raw-map predicate")
                      for s in scenarios]

    rows: List[MineRow] = []
    for cand in candidates:
        scn = cand.scenario
        host = by_token.get(scn.token)
        offsets, min_frames = _event_timing(cand.evidence, scn.t0)
        row = MineRow(
            leaf=leaf, scenario_id=host.scene_id if host else scn.token, token=scn.token,
            status="rejected_pool", log=scn.log, t0=scn.t0, t1=scn.t1,
            pool={"miner": miner_name or "(none)", "ok": bool(cand.qualifies),
                  "note": cand.note, "evidence": cand.evidence},
            windows=list(cand.windows),
            trained_windows=list(host.windows) if host else list(cand.trained_windows),
            event_t_offset_s=offsets, min_eval_frames=min_frames, note=cand.note,
        )
        pick = picks.pop(scn.token, None)
        if pick is not None:
            row.human = pick.to_dict()
        # A REVIEWER-PICKED row is not short-circuited by the pool tier. The
        # pool tier reads the raw map; a person watched the render. When they
        # disagree, stopping at the raw map throws away the only tier that can
        # say whether the geometry actually resolves — and "the reviewer chose
        # it, the map predicate says no, and the host qualifies anyway" is the
        # single most useful row this command can produce.
        if not cand.qualifies and not row.picked:
            rows.append(row)
            continue
        if host is None:
            row.status = "needs_conversion"
            row.note = (f"{cand.note} — no Arrow root; convert it before the geometry "
                        f"gate can run (navsafe/benchmark/eval/make_arrow.sh)")
            rows.append(row)
            continue
        if not host_tier:
            row.status = "qualified" if man.qualify else "no_gate"
            rows.append(row)
            continue
        try:
            row.host = _host_tier(man, host, leaf=leaf)
        except Exception as exc:  # noqa: BLE001 — one bad host must not stop the sweep
            # A host that exists but will not open is NOT the same finding as
            # one that was never converted; conflating them sends someone off
            # to re-run a conversion that already succeeded.
            row.status = "host_error"
            row.note = f"host tier failed to load ({type(exc).__name__}: {exc})"
            rows.append(row)
            continue
        if not cand.qualifies:
            row.note = (f"pool tier said no ({cand.note}); carried to the host tier because "
                        f"a reviewer chose it")
        if not man.qualify:
            row.status = "no_gate"
            row.note = "; ".join(f"{c['name']}: {c['evidence']}" for c in row.host["info"]) \
                or "no geometry gate — every converted host qualifies"
        elif row.host["ok"]:
            row.status = "qualified"
            row.note = "; ".join(f"{c['name']}: {c['evidence']}" for c in row.host["checks"])
        else:
            row.status = "rejected_host"
            row.note = "; ".join(f"{c['name']}: {c['evidence']}"
                                 for c in row.host["checks"] if not c["ok"])
        rows.append(row)

    # Whatever is LEFT in `picks` is a scenario the reviewer assigned to this
    # leaf that the seed table does not carry. That is a fact about the TABLE,
    # not about the scenario — the reviewer's sheet spans both corpora, and
    # `scenes_500.tsv` only covers one of them. So if the host is nonetheless
    # indexed (a navhard host is), it gets the full host tier like any other
    # row; only a token with no host anywhere is genuinely unreachable.
    for token, pick in picks.items():
        host = by_token.get(token)
        row = MineRow(
            leaf=leaf, scenario_id=host.scene_id if host else token, token=token,
            status="not_in_seed_table", human=pick.to_dict(),
            trained_windows=list(host.windows) if host else [],
        )
        if host is None or not host_tier:
            row.note = (f"the reviewer assigned this to {leaf}, and no converted host is "
                        f"indexed for it under any corpus. Convert it, or point --tsv / "
                        f"--roots-glob at the tree it lives in.")
            rows.append(row)
            continue
        try:
            row.host = _host_tier(man, host, leaf=leaf)
        except Exception as exc:  # noqa: BLE001 — one bad host must not stop the sweep
            row.status = "host_error"
            row.note = f"host tier failed to load ({type(exc).__name__}: {exc})"
            rows.append(row)
            continue
        row.status = ("no_gate" if not man.qualify
                      else "qualified" if row.host["ok"] else "rejected_host")
        checks = row.host["info"] if not man.qualify else row.host["checks"]
        row.note = (f"[{host.family}] not in the seed table, but its host IS indexed; "
                    + "; ".join(f"{c['name']}: {c['evidence']}" for c in checks
                                if man.qualify is None or not man.qualify or not c["ok"]
                                or row.host["ok"]))
        rows.append(row)

    # Reviewer-picked first: a person watched the render, and that outranks a
    # predicate for "is this the scenario". Within each group, the machine tiers
    # still decide the order, because they answer the other question.
    rows.sort(key=lambda r: (not r.picked, -STATUS_ORDER.index(r.status),
                             -len(r.trained_windows), r.scenario_id))
    return rows


def write_rows(rows: List[MineRow], out: "str | Path") -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict()) + "\n")
    return path


def load_rows(path: "str | Path") -> List[MineRow]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(MineRow(**json.loads(line)))
    return rows


__all__ = ["MineRow", "STATUS_ORDER", "load_rows", "mine_leaf", "write_rows"]
