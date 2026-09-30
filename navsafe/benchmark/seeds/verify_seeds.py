"""Cross-check frozen seeds against the raw nuPlan log.

The event contract is only worth as much as its predicates, so re-derive the
ego kinematics straight from ``ego_pose`` and confirm that what the family
claims actually happens inside the window: a turn family must turn, a straight
family must not, a stationary family must actually stop and then pull away.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

from navsafe.benchmark import config as cfg

US = 1_000_000


def track(db: str):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = []
    for t, x, y, qw, qx, qy, qz, vx, vy in con.execute(
        "SELECT timestamp,x,y,qw,qx,qy,qz,vx,vy FROM ego_pose ORDER BY timestamp"
    ):
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        rows.append((int(t), float(x), float(y), yaw, math.hypot(vx or 0, vy or 0)))
    con.close()
    return rows


def between(rows, t0, t1):
    return [r for r in rows if t0 <= r[0] <= t1]


def turn_deg(seg):
    tot = 0.0
    for a, b in zip(seg, seg[1:]):
        d = b[3] - a[3]
        tot += (d + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(tot)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.SEEDS
    ok = True
    for sd in sorted(root.iterdir()):
        seed = json.loads((sd / "seed.json").read_text())
        rows = track(seed["source_paths"]["db"])
        ev = seed["event"]
        w = seed["window"]
        event = between(rows, ev["t_trig_us"], ev["t_term_us"])
        window = between(rows, w["scored_t0_us"], w["scored_t1_us"])
        recon = between(rows, w["recon_t0_us"], w["recon_t1_us"])
        if not event or not window:
            print(f"{seed['seed_id']}: EMPTY window!")
            ok = False
            continue
        sp = [r[4] for r in event]
        dist = sum(math.dist(a[1:3], b[1:3]) for a, b in zip(window, window[1:]))
        turn = turn_deg(event)

        # what the family promises
        fam = seed["family"]
        checks = []
        if fam == "F3_left_turn":
            checks.append(("turns left 60-120 deg", 60 <= turn <= 120))
        elif fam == "F3_straight_traversal":
            checks.append(("stays straight (<25 deg)", abs(turn) < 25))
            checks.append(("keeps moving (>20 m)", dist > 20))
        elif fam == "F1_stationary_light_lead":
            checks.append(("actually stops (min speed <0.5)", min(sp) < 0.5))
            checks.append(("pulls away (max speed >1.5)", max(sp) > 1.5))
        checks.append(("recon window covers scored window",
                       recon[0][0] <= window[0][0] and recon[-1][0] >= window[-1][0]))
        checks.append(("policy warm-up >= 1.5 s before trigger",
                       (ev["t_trig_us"] - w["scored_t0_us"]) / US >= 1.5))

        verdict = "OK " if all(c[1] for c in checks) else "FAIL"
        if verdict == "FAIL":
            ok = False
        print(f"[{verdict}] {seed['seed_id']}  {fam}  ({seed['provenance']['location']})")
        print(f"        event {ev['event_duration_s']:5.1f}s  window {w['window_duration_s']:5.1f}s"
              f"  turn {turn:+6.1f} deg  travel {dist:6.1f} m"
              f"  speed {min(sp):.1f}-{max(sp):.1f} m/s  frames {len(window)}")
        for name, passed in checks:
            print(f"          {'PASS' if passed else 'FAIL'}  {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
