#!/usr/bin/env python3
"""Where is the stopped car? Each tag's own fit (x, y, heading) from the front and rear cameras, no motors.

    python3 tools/where_am_i.py

Single far tags are loose in range but good in heading; a tag 14 cm away fits badly (rms in px is printed).
"""
import math, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, load_json, open_grabbers

arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
loc = ArucoLocalizer(arena, car)
grabbers = open_grabbers(car, 1280, 960, rear=True)
time.sleep(1.5)
fixes = []
for name, grabber in grabbers.items():
    frame, _stamp = grabber.latest()
    if frame is not None:
        fixes += loc.observe(name, frame)[0]
for f in fixes:
    res = loc._refine([f], (f.x, f.y, f.yaw))
    print(f"{f.camera} tag{f.tag_id} {f.distance:.2f}m  x={res.x[0]:.2f} y={res.x[1]:.2f} "
          f"yaw={math.degrees(res.x[2]):+.0f}deg  rms {float(loc._per_fix_rms(res, 1)[0]):.1f}px")
joint = loc.fuse(fixes, time.monotonic())
print("joint:", joint.text() if joint else f"none ({loc.last_reject})")
head = loc.heading_only(fixes)
print("heading from far tags:", "none" if head is None else f"{math.degrees(head[0]):+.0f}deg (spread {math.degrees(head[1]):.1f}, {head[2]} tags)")
for g in grabbers.values():
    try:
        g.close()
    except Exception:
        pass
