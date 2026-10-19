#!/usr/bin/env python3
"""How far the drawn obstacle outlines are from the true blocks in route_sim: shift of the outline's middle and overlap (IoU).

    python3 tools/sim_map_error.py [runs=10] [fix_loss=0.3]
"""
import json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import route_sim as rs
from sim_hardware import MotionError

blocks = [(0.65, 0.75, 0.0, 0.2), (1.2, 1.4, 0.65, 0.75), (0.65, 0.75, 1.2, 1.4), (0.0, 0.2, 0.65, 0.75)]
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
loss = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
import cv2
shift, iou = [], []
for k in range(runs):
    r = rs.run_once(f"maperr{k}", blocks, k, [], MotionError(fix_loss=loss, realistic_fixes=False), True, False, 1.0)
    found = json.loads((rs.OUT / f"maperr{k}" / "obstacles.json").read_text())
    for b in found:
        if b["hits"] < 2:
            continue
        poly = np.array(b["outline_m"]) * 100.0
        mo = cv2.moments(poly.astype(np.float32))  # the area centroid: the mean of the vertices leans to where they bunch
        pc = np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]]) / 100.0
        truth = min(blocks, key=lambda t: math.hypot(pc[0] - (t[0] + t[1]) / 2, pc[1] - (t[2] + t[3]) / 2))
        rect = np.array([(truth[0], truth[2]), (truth[1], truth[2]), (truth[1], truth[3]), (truth[0], truth[3])]) * 100.0
        m = np.zeros((160, 160), np.uint8); n = np.zeros((160, 160), np.uint8)
        cv2.fillPoly(m, [np.round(poly + 10).astype(np.int32)], 1); cv2.fillPoly(n, [np.round(rect + 10).astype(np.int32)], 1)
        iou.append(float((m & n).sum() / max((m | n).sum(), 1)))
        rc = rect.mean(axis=0)
        shift.append(np.hypot(*(pc * 100.0 - rc)))
shift, iou = np.array(shift), np.array(iou)
print(f"{len(shift)} blocks seen at least twice in {runs} runs: outline shifted median {np.median(shift):.1f} cm "
      f"(p90 {np.percentile(shift, 90):.1f}), overlap with the true block IoU median {np.median(iou):.2f} (1.0 = exact)")
