# Copyright (c) 2022-2026, The NavSafe Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Draw what a selection predicate claims, so a human can refuse it.

Every wrong scenario this pipeline shipped passed its numbers. A merge that
was a junction turn-fan, a "U-turn" that was a right turn, an oncoming car on
a different street — each had a plausible measurement and no picture. So the
figure is not decoration: it is the step where a claim becomes falsifiable.

The drawing colours the *structure the predicate asserts*, not just the road:
for a merge, the roadblock the ego enters and the roads that feed it get
distinct colours, and the ego's expert trajectory is drawn over them with the
5 s window boundaries marked — so "two roads become one, and the ego drives in
on one of them" is either visible or it is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from navsafe.benchmark.mining import nuplan_map as nm

NEAR_M = 70.0
SEG_US = 5_000_000

_FEEDER_FILL = ("#ffd6a5", "#a0c4ff", "#ffadad", "#caffbf")
_FEEDER_EDGE = ("#e07a00", "#1d5fbf", "#c1121f", "#2d6a4f")


def _poly_xy(obj) -> Optional[np.ndarray]:
    poly = getattr(obj, "polygon", None)
    if poly is None:
        return None
    return np.asarray(poly.exterior.coords)[:, :2]


def draw_candidate(row: Dict[str, Any], out_png: str, *, leaf: str = "") -> Path:
    """Render the evidence figure for one candidate row from ``navsafe mine``.

    Args:
        row: a line of the candidates ``.jsonl`` (token/log/t0/t1 + evidence).
        out_png: where to write.
        leaf: shown in the title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer

    token, log = row["token"], row["log"]
    t0, t1 = int(row["t0"]), int(row["t1"])
    db = nm.find_log_db(log)
    ts, xy = nm.ego_expert_track(db, t0, t1)
    if len(ts) < 2:
        raise ValueError(f"{token}: only {len(ts)} ego poses in the window")
    map_api, location = nm.map_api_for(db)

    hits = (row.get("evidence") or {}).get("hits") or []
    merge_block = None
    feeders: Dict[str, tuple] = {}
    merge_xy = merge_t = None
    if hits:
        h = hits[0]
        merge_t = int(h["t_us"])
        merge_xy = tuple(h.get("xy_utm", (0.0, 0.0)))
        blk, _ = nm.roadblock_at(map_api, *merge_xy)
        if blk is not None:
            merge_block = blk
            feeders = nm.feeding_roads(map_api, blk)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)
    lo, hi = xy.min(0) - NEAR_M, xy.max(0) + NEAR_M

    centre = Point2D(*xy.mean(0))
    for layer, fc in ((SemanticMapLayer.ROADBLOCK, "0.90"),
                      (SemanticMapLayer.ROADBLOCK_CONNECTOR, "#f2e6c9")):
        try:
            objs = map_api.get_proximal_map_objects(centre, 160.0, [layer])[layer]
        except Exception:  # noqa: BLE001 — context only; absence is not fatal
            continue
        for obj in objs:
            p = _poly_xy(obj)
            if p is not None:
                ax.fill(p[:, 0], p[:, 1], fc=fc, ec="0.78", lw=0.4, zorder=1)

    if merge_block is not None:
        p = _poly_xy(merge_block)
        if p is not None:
            ax.fill(p[:, 0], p[:, 1], fc="#b7e4c7", ec="#2d6a4f", lw=1.8, zorder=3,
                    label=f"road the ego enters ({len(feeders)} feeding roads)")
        for i, (rb, _conn) in enumerate(list(feeders.values())[:4]):
            pp = _poly_xy(rb)
            if pp is None:
                continue
            came_from = str(rb.id) == str(hits[0].get("from_roadblock"))
            ax.fill(pp[:, 0], pp[:, 1], fc=_FEEDER_FILL[i], ec=_FEEDER_EDGE[i], lw=1.6, zorder=2,
                    label=f"feeding road {rb.id}" + ("  (ego came off this)" if came_from else ""))

    ax.plot(xy[:, 0], xy[:, 1], color="#d62728", lw=2.6, zorder=6, label="ego expert trajectory")
    ax.annotate("", xy=xy[-1], xytext=xy[max(0, len(xy) - 8)],
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.6), zorder=7)
    ax.scatter(*xy[0], s=70, facecolors="white", edgecolors="#d62728", linewidths=2, zorder=7,
               label="ego start")
    if merge_xy is not None:
        ax.scatter(*merge_xy, s=150, marker="*", color="#2d6a4f", zorder=8,
                   label=f"predicate fires (t+{(merge_t - t0) / 1e6:.1f}s, "
                         f"window s{min(3, (merge_t - t0) // SEG_US) + 1})")
    for w in range(1, 4):
        k = int(np.searchsorted(ts, t0 + w * SEG_US))
        if 0 < k < len(xy):
            ax.scatter(*xy[k], s=28, color="black", zorder=7)
            ax.annotate(f"s{w + 1}", xy[k], textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{leaf or row.get('predicate', '')} candidate — {token}\n{location} · "
                 f"{row.get('note', '')[:110]}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


__all__ = ["draw_candidate"]
