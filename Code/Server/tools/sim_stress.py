#!/usr/bin/env python3
"""Stress the lap in route_sim: random block layouts (one per wall, off-centre), tags often lost, a fast drive.

    python3 tools/sim_stress.py [layouts=24] [fix_loss=0.5] [dark|ir]   (dark: IR cannot see the blocks)
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import random, sys
import route_sim as rs
from sim_hardware import MotionError
def layout(seed):
    random.seed(seed); out = []
    for wall in range(4):
        c = 70 + random.uniform(-20, 20); g = random.uniform(0, 5); a, b = (c - 5) / 100, (c + 5) / 100; g /= 100
        out.append([(a, b, g, g + .2), (1.4 - g - .2, 1.4 - g, a, b), (a, b, 1.4 - g - .2, 1.4 - g), (g, g + .2, a, b)][wall])
    return out
n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
loss = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
DARK = (sys.argv[3] == "dark") if len(sys.argv) > 3 else True  # IR modules blind to the (dark) foam, the harsh case
ok = 0; coll = 0; bad = []
for k in range(n):
    r = rs.run_once(f"stress_{k}", layout(100 + k), k, [], MotionError(forward=1.06, fix_loss=loss), True, DARK, 1.0)
    ok += r["ok"]; coll += "COLLISION" in r["why"]
    if not r["ok"]: bad.append(f"stress_{k}: {r['why']}")
print("\n".join(bad)); print(f"OK {ok}/{n}, collisions {coll}")
