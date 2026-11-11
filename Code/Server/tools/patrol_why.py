#!/usr/bin/env python3
"""Look at one moment of a patrol run: what the car knew and why it did what it did.

    python3 tools/patrol_why.py DIR                 # the last moment of the run (where it stopped or hit something)
    python3 tools/patrol_why.py DIR --t 41.3        # 41.3 s after the start
    python3 tools/patrol_why.py DIR --list          # goals, stops and how long the car stood still, with their times

Prints, for that moment: the pose and its uncertainty, the command and the wheels, why it stood still if it did, what the
safety check found, how far the body was from the nearest occupied cell and from the walls ON THE MAP IT HAD THEN, how
much of the floor just ahead it had not seen, what the tags said about the pose, and the gimbal looks before it (with
the picture files). Writes why.png: that map, the body, the path, the driven trail.
If the car touched something the map shows as free or far away, the map or the pose was wrong: the looks and the tag
lines say which.
"""
import argparse
import json
import math
import pathlib
import sys

import cv2
import numpy as np

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
from patrol.drive import Safety  # noqa: E402
from patrol.world import FREE, OCCUPIED, RES_M  # noqa: E402

CELL = 10


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--t", type=float)
    ap.add_argument("--list", action="store_true")
    o = ap.parse_args()
    d = o.folder
    control, looks, plans, events, tags = (rows(d / f"{n}.jsonl") for n in ("control", "looks", "plans", "events", "tags"))
    if not control:
        raise SystemExit("no control.jsonl in " + str(d))
    if o.list:
        for e in events:
            print(f"{e['s']:7.1f} s  {e['what']}  " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("s", "what")))
        start, why = None, None
        for r in control + [dict(s=control[-1]["s"], why=None)]:
            if r["why"] != why:
                if why is not None and r["s"] - start > 0.5:
                    print(f"{start:7.1f} s  stood still {r['s'] - start:.1f} s: {why}")
                start, why = r["s"], r["why"]
        return
    t = control[-1]["s"] if o.t is None else o.t
    r = min(control, key=lambda q: abs(q["s"] - t))
    x, y, yaw = r["pose"]
    car = json.loads((SERVER / "car_nav.json").read_text())
    hl, hw = car["body_m"][0] / 2, car["body_m"][1] / 2
    maps = sorted((d / "maps").glob("*.npy"))
    before = [m for m in maps if float(m.stem) <= r["s"]]
    log = np.load((before or maps)[-1])
    nx, ny = log.shape
    W, H = (nx - 1) * RES_M, (ny - 1) * RES_M
    print(f"moment {r['s']:.2f} s (map from {float((before or maps)[-1].stem):.2f} s)")
    print(f"pose x {x * 100:.1f} cm  y {y * 100:.1f} cm  heading {math.degrees(yaw):+.1f} deg   uncertainty {r['sigma_cm']:.1f} cm")
    print(f"goal {r['goal']};  command forward {r['cmd'][0]:+.2f} left {r['cmd'][1]:+.2f} m/s turn {math.degrees(r['cmd'][2]):+.0f} deg/s;  wheels LF LR RF RR {r['pwm']}{'' if r['armed'] else ' (not armed)'}")
    print("stood still: " + r["why"] if r["why"] else "driving")
    if r.get("safety"):
        print(f"safety check: {r['safety']}")
    # the body against the map it had
    c, s = math.cos(yaw), math.sin(yaw)
    edge = [(a, b) for a in np.linspace(-hl, hl, 15) for b in (-hw, hw)] + [(a, b) for a in (-hl, hl) for b in np.linspace(-hw, hw, 9)]
    pts = np.array([(x + c * a - s * b, y + s * a + c * b) for a, b in edge])
    occ = np.argwhere(log > OCCUPIED) * RES_M
    if len(occ):
        dist = np.hypot(pts[:, None, 0] - occ[None, :, 0], pts[:, None, 1] - occ[None, :, 1])
        n = np.unravel_index(int(dist.argmin()), dist.shape)
        print(f"nearest occupied cell on the map: {dist.min() * 100:.1f} cm from the body, at ({occ[n[1]][0] * 100:.0f}, {occ[n[1]][1] * 100:.0f}) cm")
    else:
        print("the map had no occupied cell")
    print(f"body to the walls (by the pose): left {pts[:, 0].min() * 100:.1f}  right {(W - pts[:, 0].max()) * 100:.1f}  bottom {pts[:, 1].min() * 100:.1f}  top {(H - pts[:, 1].max()) * 100:.1f} cm")
    ok, detail = Safety(hl, hw, (W, H)).check(log, (x, y, yaw), r["cmd"])
    print(f"the safety rule on that map and command: {'passes' if ok else 'does not pass'} {detail}")
    near = [q for q in tags if q.get("pose") and abs(q["s"] - 0.12 - r["s"]) < 0.6]
    for q in near[-4:]:
        p = q["pose"]
        print(f"  tags ({q['camera']}, {p['tags']} tags, nearest {p['nearest']:.2f} m) at {q['s']:.2f} s: x {p['x'] * 100:.1f} y {p['y'] * 100:.1f} heading {math.degrees(p['yaw']):+.1f}"
              f"  -> differs from the pose by {math.hypot(p['x'] - x, p['y'] - y) * 100:.1f} cm")
    if not near:
        print("  no tag pose within 0.6 s of that moment")
    for q in [q for q in looks if q["s"] <= r["s"]][-5:]:
        print(f"  look {q['file']} at {q['s']:.2f} s: pan {q['pan']:+.0f} deg, echo {'none' if q['echo'] is None else f'{q['echo'] * 100:.0f} cm'}, "
              f"foot points {q['hits']}, free rays {q['free']}, close {q['close']}; {q['used']}")
    plan = [p for p in plans if p["s"] <= r["s"]]
    if plan:
        p = plan[-1]
        print(f"last plan at {p['s']:.2f} s to {p['goal']} ({p['reason']}): {'no path: ' + str(p['info'].get('why')) if not p['path'] else str(len(p['path'])) + ' steps'}, {p['info']}")
    # picture
    g = log.T[::-1]
    img = np.full(g.shape + (3,), 215, np.uint8)
    img[g < FREE] = (236, 246, 236)
    img[g > OCCUPIED] = (60, 60, 200)
    img = cv2.resize(img, None, fx=CELL, fy=CELL, interpolation=cv2.INTER_NEAREST)
    hh = img.shape[0]
    px = lambda a, b: (int(a / RES_M * CELL + CELL / 2), int(hh - b / RES_M * CELL - CELL / 2))
    trail = [q["pose"] for q in control if q["s"] <= r["s"]][::3]
    if len(trail) > 1:
        cv2.polylines(img, [np.array([px(q[0], q[1]) for q in trail], np.int32)], False, (180, 90, 20), 2, cv2.LINE_AA)
    if plan and plan[-1]["path"]:
        cv2.polylines(img, [np.array([px(q[0], q[1]) for q in plan[-1]["path"]], np.int32)], False, (0, 150, 0), 2, cv2.LINE_AA)
    box = [px(x + c * a - s * b, y + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    cv2.polylines(img, [np.array(box, np.int32)], True, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, px(x, y), px(x + c * hl, y + s * hl), (20, 20, 20), 2, cv2.LINE_AA, tipLength=0.35)
    at = (r.get("safety") or {}).get("at")
    if at:
        cv2.drawMarker(img, px(*at), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.imwrite(str(d / "why.png"), img)
    print(f"picture: {d / 'why.png'}   (looks are in {d / 'looks'})")


if __name__ == "__main__":
    main()
