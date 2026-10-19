#!/usr/bin/env python3
"""Replay the camera looks saved by route_follow.py (debug-dir/looks): foot points, walls split off, clusters.

    python3 tools/replay_looks.py [debug-dir=route_follow_debug] [x0 x1 y0 y1]   # only looks seen in that window (m)

Prints, per look, the pose, how many foot points were obstacle or wall, and every cluster as a bounding box in cm.
With no window, all looks go through ObstacleMap in order and the blocks it ends with are printed (the map the run drew).
"""
import json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from types import SimpleNamespace
from aruco_localizer import CAR_PATH, MAP_PATH, load_json
from foot_cam import FootCamera
from obstacle_map import ObstacleMap, split_walls, to_world

SERVER = pathlib.Path(__file__).resolve().parents[1]
folder = SERVER / (sys.argv[1] if len(sys.argv) > 1 else "route_follow_debug") / "looks"
window = [float(v) for v in sys.argv[2:6]] if len(sys.argv) >= 6 else None
arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
cam = FootCamera(car)
size = arena["arena_size_m"]
omap = ObstacleMap(size)
for line in (folder / "looks.jsonl").read_text().splitlines():
    look = json.loads(line)
    x, y, yaw = look["pose"]
    if window and not (window[0] <= x <= window[1] and window[2] <= y <= window[3]):
        continue
    pose = SimpleNamespace(x=x, y=y, yaw=math.radians(yaw))
    seen, near = [], []
    for pan in look["pans"]:
        img = cv2.imread(str(folder / f"{look['n']:03d}_pan{pan:+03.0f}.jpg"))
        if img is None:
            continue
        v = cam.observe(img, pan)
        seen.append(v.points[~np.isnan(v.points[:, 0])])
        if len(v.close):
            near.append(v.close)
    pts = np.vstack(seen + near) if (seen or near) else np.zeros((0, 2))
    obstacle, wall = split_walls(pts, pose, size)
    omap.add(obstacle, wall, pose)
    clusters = ObstacleMap._clusters(to_world(obstacle, pose)) if len(obstacle) else []
    desc = "; ".join(f"{len(c)} pts x {c[:,0].min()*100:.0f}~{c[:,0].max()*100:.0f} y {c[:,1].min()*100:.0f}~{c[:,1].max()*100:.0f}" for c in clusters)
    print(f"look {look['n']:3d} pose ({x*100:.0f},{y*100:.0f},{yaw:+.0f}deg): {len(obstacle)} obstacle / {len(wall)} wall pts | {desc or '-'}")

if not window:
    print("\nblocks at the end (seen at least twice):")
    for b in omap.blocks:
        if b["hits"] >= 2:
            o = np.array(b["outline"]) * 100
            print(f"  B{b['id']} hits {b['hits']}: outline x {o[:,0].min():.0f}~{o[:,0].max():.0f}, y {o[:,1].min():.0f}~{o[:,1].max():.0f} cm")
    print("  true: B1 x 65~75 y 0~20 | B2 x 120~140 y 65~75 | B3 x 65~75 y 120~140 | B4 x 0~20 y 65~75 (cm)")
