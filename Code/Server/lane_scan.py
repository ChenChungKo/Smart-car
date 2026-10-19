#!/usr/bin/env python3
"""Stationary obstacle check for the wall-hugging lap: look from where the car stands, draw what it sees.

No motors. The car looks with what route_follow.py --hug uses: the gimbal camera (foot lines of the
foam, foot_cam.py) at several gimbal angles, plus the ultrasonic at the same angles and the IR for
comparison. The pose comes from the front/rear tags (or --pose). The foot points that are not the
walls go into the obstacle map (obstacle_map.py), and the result is told in terms of the route:
which lane each obstacle blocks, how far along it and how far off the lane line.

    python3 lane_scan.py                          # pose from the tags
    python3 lane_scan.py --pose 1.17,0.50,90      # known pose (m, m, deg)
    python3 lane_scan.py --truth 120,140,65,75    # blocks you measured (x0,x1,y0,y1 cm): scored and drawn green
    python3 lane_scan.py --replay lane_scan_debug/20261019-210000   # redo the analysis, no hardware

Output (lane_scan_debug/<time>/): overview.jpg = the map (route rectangle, car, obstacles, ultrasonic
rays) beside every camera view (front, rear, and the gimbal at each angle with its foot line),
scan.json, and raw/ with the frames for --replay.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from aruco_localizer import (CAR_PATH, MAP_PATH, ArucoLocalizer, Pose2D, draw_map, draw_overlay, load_json,
                             open_grabbers)
from foot_cam import FootCamera, FootView
from obstacle_map import ObstacleMap, lane_report, path_gap, shift_to_clear, split_walls, to_world

SERVER = Path(__file__).resolve().parent
OUT = SERVER / "lane_scan_debug"
SONIC_PIVOT_M, SONIC_ARM_M = 0.085, 0.03
PATH_MARGIN_M = 0.025
TRUTH_COLOR = (0, 170, 0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose", default=None, help="'x,y,yaw_deg' instead of the tags.")
    p.add_argument("--truth", nargs="*", default=[], metavar="X0,X1,Y0,Y1", help="Measured blocks, cm.")
    p.add_argument("--pans", type=float, nargs="+", default=[-60, -45, -30, -15, 0, 15, 30, 45, 60])
    p.add_argument("--replay", default=None, metavar="DIR")
    p.add_argument("--localize-s", type=float, default=2.5)
    return p.parse_args()


def lanes_of(arena: dict, car: dict, gap_m: float | None = None):
    """The four --hug lanes in driving order (counter-clockwise from the bottom wall), as route_follow builds them."""
    w, h = (float(v) for v in arena["arena_size_m"])
    hl, hw = (float(v) / 2.0 for v in car.get("body_m", [0.29, 0.18]))
    spot = math.hypot(hl, hw) + 0.04 + 0.02
    lane = spot if gap_m is None else hw + gap_m
    return [("bottom wall lane", (spot, lane), (w - spot, lane)),
            ("right wall lane", (w - lane, spot), (w - lane, h - spot)),
            ("top wall lane", (w - spot, h - lane), (spot, h - lane)),
            ("left wall lane", (lane, h - spot), (lane, spot))]


def localize(arena: dict, car: dict, seconds: float):
    """Pose and the front/rear frames from the CSI cameras (the cameras are closed again afterwards)."""
    loc = ArucoLocalizer(arena, car)
    grabbers = open_grabbers(car, 1280, 960, rear=True)
    frames, fixes = {}, []
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not all(g.latest()[0] is not None for g in grabbers.values()):
            time.sleep(0.1)
        seen = {n: 0.0 for n in grabbers}
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            for name, g in grabbers.items():
                frame, stamp = g.latest()
                if frame is None or stamp == seen[name]:
                    continue
                seen[name] = stamp
                found, corners, ids = loc.observe(name, frame)
                fixes.extend(found)
                frames[name] = draw_overlay(frame, corners, ids, found, name)
            time.sleep(0.05)
    finally:
        for g in grabbers.values():
            g.close()
    return loc.fuse(fixes, time.monotonic()), frames


def sense(args, car: dict, raw: Path) -> dict:
    """Gimbal camera, ultrasonic and IR at each pan angle; frames saved under raw/."""
    from foot_cam import GimbalFeed
    from safe_surround_cruise import SafetySensors
    from servo import Servo, load_gimbal_home, load_left_90_pan, load_right_90_pan, servo_pan_for_bearing

    feed = GimbalFeed().start()
    feed.wait_ready()
    servo = Servo()
    home, _ = load_gimbal_home()
    right90, left90 = load_right_90_pan(home), load_left_90_pan()
    sensors = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=False, active_high=False))
    shots = []
    try:
        time.sleep(0.5)
        for pan in args.pans:
            servo.set_pan(servo_pan_for_bearing(home, right90, pan, left90))
            time.sleep(0.5)
            frame = feed.fresh(time.monotonic())
            readings = [r for r in (sensors.read_sonic() for _ in range(3)) if r is not None]
            sonic = st.median(readings) if readings else None
            cv2.imwrite(str(raw / f"gimbal_{pan:+04.0f}.jpg"), frame)
            shots.append({"pan": pan, "sonic_cm": sonic})
            print(f"  gimbal {pan:+4.0f} deg: ultrasonic {'-' if sonic is None else f'{sonic:.0f} cm'}", flush=True)
        ir = {k: int(v) for k, v in sensors.read_ir().items() if k.startswith(("front", "wheel"))}
    finally:
        servo.set_pan(home)
        time.sleep(0.4)
        servo.release_gimbal()
        sensors.close()
        feed.close()
    return {"shots": shots, "ir": ir}


def truth_blocks(specs):
    out = []
    for k, spec in enumerate(specs, 1):
        x0, x1, y0, y1 = (float(v) / 100.0 for v in spec.split(","))
        out.append((f"T{k}", min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return out


def main():
    args = parse_args()
    arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
    size = tuple(float(v) for v in arena["arena_size_m"])
    hl, hw = (float(v) / 2.0 for v in car.get("body_m", [0.29, 0.18]))
    cam = FootCamera(car)

    if args.replay:
        run = Path(args.replay)
        info = json.loads((run / "scan.json").read_text(encoding="utf-8"))
        raw = run / "raw"
        csi = {n: cv2.imread(str(raw / f"{n}.jpg")) for n in ("front", "rear") if (raw / f"{n}.jpg").exists()}
        pose = Pose2D(*info["pose"][:2], math.radians(info["pose"][2]))
        print(f"replay {run} ({info['time']})")
    else:
        run = OUT / time.strftime("%Y%m%d-%H%M%S")
        raw = run / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        csi, pose = {}, None
        if args.pose:
            x, y, yaw = (float(v) for v in args.pose.split(","))
            pose = Pose2D(x, y, math.radians(yaw))
        else:
            print(f"pose from the tags ({args.localize_s:.0f} s)...", flush=True)
            pose, csi = localize(arena, car, args.localize_s)
            if pose is None:
                raise SystemExit("no pose from the tags: move the car or give --pose x,y,yaw_deg")
        for name, frame in csi.items():
            cv2.imwrite(str(raw / f"{name}.jpg"), frame)
        print("looking:", flush=True)
        info = sense(args, car, raw)
        info.update(time=time.strftime("%Y-%m-%d %H:%M:%S"), pose=[pose.x, pose.y, math.degrees(pose.yaw)])
        (run / "scan.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"pose x={pose.x * 100:.0f} cm y={pose.y * 100:.0f} cm heading {math.degrees(pose.yaw):+.0f} deg")

    # foot points of every angle
    views: list[FootView] = []
    for shot in info["shots"]:
        frame = cv2.imread(str(raw / f"gimbal_{shot['pan']:+04.0f}.jpg"))
        views.append(cam.observe(frame, shot["pan"], keep_frame=True))
    parts = [v.points[~np.isnan(v.points[:, 0])] for v in views] + [v.close for v in views if len(v.close)]
    points = np.vstack(parts) if parts else np.zeros((0, 2))
    obstacle, wall = split_walls(points, pose, size)
    omap = ObstacleMap(size)
    blocks = omap.add(obstacle, wall, pose)

    # what the car would decide right now, going straight ahead
    gap = path_gap(obstacle, hl, hw + PATH_MARGIN_M)
    print(f"\nfoot points {len(points)}: {len(obstacle)} obstacle, {len(wall)} on the walls (left out)")
    if gap is None:
        print("straight ahead: nothing in the car's path")
    else:
        left = shift_to_clear(obstacle, +1, hl, hw, 0.03)
        right = shift_to_clear(obstacle, -1, hl, hw, 0.03)
        print(f"straight ahead: an obstacle {gap * 100:.0f} cm in front of the nose; clear it by strafing "
              f"{left * 100:.0f} cm left or {right * 100:.0f} cm right")

    world = to_world(obstacle, pose) if len(obstacle) else np.zeros((0, 2))
    lanes = lanes_of(arena, car)
    print("\nobstacles:")
    for b in blocks:
        print(f"  {omap.describe(b)}")
    if not blocks:
        print("  none")
    print("\neffect on the route (the lap's four lanes, driving order):")
    report = lane_report(world, lanes, hw + PATH_MARGIN_M)
    for r in report:
        side = "left" if (r["across_min_m"] + r["across_max_m"]) > 0 else "right"
        print(f"  {r['lane']}: blocked from {r['first_along_m'] * 100:.0f} cm along its {r['lane_length_m'] * 100:.0f} cm, "
              f"{max(abs(r['across_min_m']), abs(r['across_max_m'])) * 100:.0f} cm to the {side} of the lane line at most")
    if not report:
        print("  no lane is blocked from here")

    truth = truth_blocks(args.truth)
    if truth:
        print("\nagainst what you measured:")
        for name, x0, x1, y0, y1 in truth:
            inside = ((world[:, 0] >= x0 - 0.04) & (world[:, 0] <= x1 + 0.04) &
                      (world[:, 1] >= y0 - 0.04) & (world[:, 1] <= y1 + 0.04)).sum() if len(world) else 0
            # distance of every foot point from the measured rectangle
            if len(world):
                dx = np.maximum.reduce([x0 - world[:, 0], world[:, 0] - x1, np.zeros(len(world))])
                dy = np.maximum.reduce([y0 - world[:, 1], world[:, 1] - y1, np.zeros(len(world))])
                near = np.hypot(dx, dy)
                close_pts = near < 0.12
                err = f"foot points within 12 cm of it sit {np.median(near[close_pts]) * 100:.1f} cm off its outline" \
                    if close_pts.any() else "no foot point near it"
            else:
                err = "no obstacle points"
            print(f"  {name} ({x0 * 100:.0f}~{x1 * 100:.0f}, {y0 * 100:.0f}~{y1 * 100:.0f} cm): {inside} foot points on it; {err}")

    # the picture: map on the left, camera views on the right
    route = [(a[0], a[1]) for _n, a, _b in lanes] + [lanes[0][1]]
    img = draw_map(arena, route=route, trail=[], pose=pose, size_px=640)
    omap.draw(img, min_hits=1)
    margin, scale = 30, (640 - 60) / max(size)
    ih = int(size[1] * scale) + 2 * margin

    def px(x, y):
        return int(round(margin + x * scale)), int(round(ih - margin - y * scale))

    for name, x0, x1, y0, y1 in truth:
        cv2.rectangle(img, px(x0, y1), px(x1, y0), TRUTH_COLOR, 2)
        cv2.putText(img, name, px(x1 + 0.01, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TRUTH_COLOR, 1, cv2.LINE_AA)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    for shot in info["shots"]:
        if shot["sonic_cm"] is None:
            continue
        a = pose.yaw + math.radians(shot["pan"])
        ox = pose.x + SONIC_PIVOT_M * c + SONIC_ARM_M * math.cos(a)
        oy = pose.y + SONIC_PIVOT_M * s + SONIC_ARM_M * math.sin(a)
        d = shot["sonic_cm"] / 100.0
        cv2.line(img, px(ox, oy), px(ox + d * math.cos(a), oy + d * math.sin(a)), (200, 180, 60), 1, cv2.LINE_AA)
        cv2.circle(img, px(ox + d * math.cos(a), oy + d * math.sin(a)), 3, (200, 120, 0), -1)
    cv2.putText(img, "red = obstacle (camera)  grey = wall  cyan line = ultrasonic  green = measured",
                (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    tiles = []
    for v, shot in zip(views, info["shots"]):
        tile = cv2.resize(cam.draw(v), (320, 240))
        sonic = "-" if shot["sonic_cm"] is None else f"{shot['sonic_cm']:.0f}cm"
        cv2.putText(tile, f"gimbal {v.pan:+.0f}  sonic {sonic}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        tiles.append(tile)
    for name in ("front", "rear"):
        if csi.get(name) is not None:
            tile = cv2.resize(csi[name], (320, 240))
            cv2.putText(tile, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    k = img.shape[0] / sheet.shape[0]
    sheet = cv2.resize(sheet, (int(sheet.shape[1] * k), img.shape[0])) if k < 1.0 else sheet
    height = max(img.shape[0], sheet.shape[0])
    pad = lambda m: np.vstack([m, np.full((height - m.shape[0], m.shape[1], 3), 240, np.uint8)])
    cv2.imwrite(str(run / "overview.jpg"), np.hstack([pad(img), pad(sheet)]))
    (run / "obstacles.json").write_text(json.dumps({"obstacles": omap.as_dict(), "lanes": report}, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"\n{run / 'overview.jpg'}")


if __name__ == "__main__":
    main()
