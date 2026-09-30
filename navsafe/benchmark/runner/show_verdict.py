"""Print a verdict.json the way a reviewer wants to read it."""
import json
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    v = json.loads(Path(arg).read_text())
    r, o, s = v["rubric"], v["outcome"], v["score"]
    head = "PASS" if r["passed"] else ("GATE" if r["gate_violated"] else "FAIL")
    print(f"\n[{head}] {r['family']}  {r['seed_id']}  ({r['regime']})  "
          f"score={s['score']:.3f}  policy={s['policy'] or '-'}")
    print("  success predicates")
    for x in r["success"]:
        print(f"    {'ok  ' if x['passed'] else 'FAIL'} {x['name']:<12} {x['reason']}")
    print("  hard gates")
    for x in r["gates"]:
        print(f"    {'ok  ' if x['passed'] else 'GATE'} {x['name']:<12} {x['reason']}")
    t = v.get("termination")
    if t:
        mark = "" if t.get("policy_attributed", True) else \
            "   [benchmark ending — episode reported '—', excluded]"
        print(f"  ended          : {t['reason']} @ frame {t['frame']}"
              + (f" ({t['detail']})" if t.get("detail") else "") + mark)
    print(f"  outcome labels : {o['labels'] or '(none)'}")
    print(f"  crash cells    : {o['crash_cells'] or '(none)'}")
    c = r["coverage"]
    print(f"  coverage       : frames={c['n_frames']} scored={c['n_scored']} "
          f"scored_fraction={c['scored_fraction']}")
    def _f(v):
        return "n/a" if v is None else f"{v:.3f}"
    print(f"  components     : progress={_f(s['progress'])} rules={_f(s['rules'])} "
          f"comfort={_f(s['comfort'])} recovery={_f(s['recovery'])} "
          f"gate={s['safety_gate']}")
    if s.get("weights", {}).get("comfort_excluded"):
        print("                   (comfort excluded: teleport execution; "
              "weight redistributed)")
    print(f"  diagnostics    : {r['diagnostics']}")
