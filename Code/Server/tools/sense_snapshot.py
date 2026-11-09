#!/usr/bin/env python3
"""What the car senses where it stands, for a report. Nothing drives; the gimbal turns.

    python3 tools/sense_snapshot.py
    python3 tools/sense_snapshot.py --blocks 65,75,0,20 120,139,65,75     # the blocks as they are placed (cm)
    python3 tools/sense_snapshot.py --no-blocks                            # an empty arena

1. Pose from the arena tags (front and rear cameras), and which tags each camera sees.
2. One picture from each of the 5 cameras, a contact sheet, and the bird's-eye view (front, rear, left, right; and front
   and rear alone, which do not depend on the side cameras' mounting).
3. Ultrasonic sweep from -90 (right) to +90 (left) deg in 15 deg steps, 5 readings each; next to each the distance the
   map says: the wall or a block along the centre of the beam, and the nearest one inside a +-12 deg cone (the echo
   comes from the nearest thing the beam touches).
4. The 6 IR states.
Results to route_follow_debug/report/<time>/ (pictures, sonic_sweep.png, summary.json). Close the dashboard and
anything else that uses the cameras first.
"""
import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, car_from_camera, load_json, open_grabbers  # noqa: E402
from static_survey import fresh_read, open_usb, read_with_timeout, set_pan  # noqa: E402

LAP_BLOCKS = ["65,75,0,20", "120,139,65,75", "65,75,120,140", "0,20,65,75"]   # the lap layout, cm x0,x1,y0,y1
PANS = list(range(-90, 91, 15))
CONE_DEG = 12.0


def ray_hit(ox, oy, angle, width, height, blocks):
    """(distance m, what) to the first wall or block along a ray from (ox, oy)."""
    dx, dy = math.cos(angle), math.sin(angle)
    best, what = float("inf"), "wall"
    for t in ([(width - ox) / dx] if dx > 1e-9 else []) + ([-ox / dx] if dx < -1e-9 else []) + \
             ([(height - oy) / dy] if dy > 1e-9 else []) + ([-oy / dy] if dy < -1e-9 else []):
        if 0 <= t < best:
            best = t
    for i, (x0, x1, y0, y1) in enumerate(blocks):
        t0, t1 = 0.0, best
        for o, d, lo, hi in ((ox, dx, x0, x1), (oy, dy, y0, y1)):
            if abs(d) < 1e-9:
                if not lo <= o <= hi:
                    t0, t1 = 1.0, 0.0
                continue
            a, b = sorted(((lo - o) / d, (hi - o) / d))
            t0, t1 = max(t0, a), min(t1, b)
        if t0 <= t1 and t0 < best:
            best, what = t0, f"block {i + 1}"
    return best, what


def tag_pose(loc, grabbers, seconds=3.0):
    """Fused pose from several frames, the tags each camera saw, the spread of the fits."""
    poses, seen, end = [], {name: set() for name in grabbers}, time.monotonic() + seconds
    while time.monotonic() < end:
        fixes = []
        for name, grabber in grabbers.items():
            frame, _stamp = grabber.latest()
            if frame is not None:
                found = loc.observe(name, frame)[0]
                seen[name].update(f.tag_id for f in found)
                fixes += found
        pose = loc.fuse(fixes, time.monotonic()) if fixes else None
        if pose is not None:
            poses.append(pose)
        time.sleep(0.25)
    if len(poses) < 3:
        return None, seen
    ref = poses[0].yaw
    yaw = ref + st.median([(p.yaw - ref + math.pi) % (2 * math.pi) - math.pi for p in poses])
    x, y = st.median(p.x for p in poses), st.median(p.y for p in poses)
    spread = max(math.hypot(p.x - x, p.y - y) for p in poses)
    return dict(x=x, y=y, yaw=yaw, fits=len(poses), spread_m=spread,
                rms_px=st.median(p.rms_px for p in poses)), seen


def contact_sheet(frames: dict) -> np.ndarray:
    tiles = []
    for name in ("front", "rear", "gimbal", "left", "right"):
        img = frames.get(name)
        tile = np.full((360, 480, 3), 40, np.uint8) if img is None else cv2.resize(img, (480, 360))
        cv2.putText(tile, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(tile, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    tiles.append(np.full((360, 480, 3), 40, np.uint8))
    return np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])


def sweep_plot(path, pose, rows, width, height, blocks, body):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
    ax.add_patch(plt.Rectangle((0, 0), width * 100, height * 100, fill=False, lw=2, color="#555"))
    for i, (x0, x1, y0, y1) in enumerate(blocks):
        ax.add_patch(plt.Rectangle((x0 * 100, y0 * 100), (x1 - x0) * 100, (y1 - y0) * 100, color="#4a78c2", alpha=0.6))
        ax.text((x0 + x1) * 50, (y0 + y1) * 50, f"B{i + 1}", ha="center", va="center", color="white", fontsize=9)
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    hl, hw = body[0] / 2, body[1] / 2
    corners = [(pose["x"] + c * a - s * b, pose["y"] + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    ax.add_patch(plt.Polygon([(px * 100, py * 100) for px, py in corners], fill=False, color="#d23b3b", lw=2))
    for r in rows:
        ox, oy, ang = r["origin"][0], r["origin"][1], r["angle"]
        e = r["expected_centre_cm"] / 100
        ax.plot([ox * 100, (ox + e * math.cos(ang)) * 100], [oy * 100, (oy + e * math.sin(ang)) * 100], color="#bbb", lw=1)
        if r["measured_cm"] is not None:
            m = r["measured_cm"] / 100
            ax.plot((ox + m * math.cos(ang)) * 100, (oy + m * math.sin(ang)) * 100, "o",
                    color="#1c9a4a" if r["ok"] else "#d23b3b", ms=6)
    ax.set_xlim(-5, width * 100 + 5)
    ax.set_ylim(-5, height * 100 + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("ultrasonic sweep: grey = map, dot = measured (green within 5 cm)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="*", default=LAP_BLOCKS, help="x0,x1,y0,y1 in cm (default: the lap layout)")
    ap.add_argument("--no-blocks", action="store_true")
    args = ap.parse_args()
    blocks = [] if args.no_blocks else [tuple(float(v) / 100 for v in b.split(",")) for b in args.blocks]
    arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
    width, height = (float(v) for v in arena["arena_size_m"])
    body = tuple(float(v) for v in car.get("body_m", [0.275, 0.17]))
    out = SERVER / "route_follow_debug" / "report" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    loc = ArucoLocalizer(arena, car)
    grabbers = open_grabbers(car, 1280, 960, rear=True)
    caps = open_usb(load_json(SERVER / "camera_hardware.json"))
    from safe_surround_cruise import SafetySensors, read_battery_voltage
    from servo import Servo, load_gimbal_home, load_left_90_pan, load_right_90_pan, servo_pan_for_bearing
    servo = Servo()
    home, _tilt = load_gimbal_home()
    right90, left90 = load_right_90_pan(home), load_left_90_pan()
    sensors = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=False, active_high=False))
    summary = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), battery_v=read_battery_voltage(),
                   blocks_cm=[[v * 100 for v in b] for b in blocks])
    try:
        set_pan(servo, home)
        time.sleep(2.0)
        # 1. pose
        pose, seen = tag_pose(loc, grabbers)
        print("\n== 1. pose from the tags ==")
        for name, ids in seen.items():
            print(f"  {name:6s} tags seen: {sorted(ids) or 'none'}")
        if pose is None:
            print("  no pose: put the car in the arena with tags in view (the sweep needs the pose)")
        else:
            print(f"  x {pose['x'] * 100:.1f} cm  y {pose['y'] * 100:.1f} cm  heading {math.degrees(pose['yaw']):+.1f} deg"
                  f"   ({pose['fits']} fits in 3 s, spread {pose['spread_m'] * 100:.1f} cm, rms {pose['rms_px']:.1f} px)")
        summary["pose"] = None if pose is None else {**pose, "yaw_deg": math.degrees(pose["yaw"])}
        summary["tags_seen"] = {k: sorted(int(i) for i in v) for k, v in seen.items()}
        # 2. pictures and bird's-eye view
        frames = {}
        for name, grabber in grabbers.items():
            frame, _stamp = grabber.latest()
            if frame is not None:
                frames[name] = frame.copy()
        for name, cap in caps.items():
            ok, frame = fresh_read(cap)
            if ok:
                frames[name] = frame.copy()
        for name, frame in frames.items():
            cv2.imwrite(str(out / f"{name}.jpg"), frame)
        cv2.imwrite(str(out / "cameras.jpg"), contact_sheet(frames))
        from dashboard.ipm import GroundMapper
        from dashboard.live import all_mounts
        mounts = all_mounts(car)
        mapper = GroundMapper(loc, car_from_camera, body)
        bev = {}
        for label, names in (("bev_all", ("front", "rear", "left", "right")), ("bev_front_rear", ("front", "rear"))):
            items = [(n, mounts[n], frames[n]) for n in names if n in frames]
            if items:
                cv2.imwrite(str(out / f"{label}.jpg"), mapper.mosaic(items))
                bev[label] = [n for n, _m, _f in items]
        print("\n== 2. cameras ==")
        print(f"  pictures: {', '.join(sorted(frames))} ({len(frames)}/5);  bird's-eye: {bev}")
        summary["cameras"] = sorted(frames)
        # 3. ultrasonic sweep
        print("\n== 3. ultrasonic sweep (positive = left) ==")
        print("   pan   measured   map centre        map nearest in cone   measured - map")
        sonic = car.get("sonic", {})
        pivot, arm = float(sonic.get("pivot_x_m", 0.085)), float(sonic.get("arm_m", 0.03))
        rows = []
        for pan in PANS:
            set_pan(servo, servo_pan_for_bearing(home, right90, pan, left90))
            time.sleep(0.6)
            reads = [r for r in (read_with_timeout(sensors.read_sonic) for _ in range(5)) if r is not None]
            measured = st.median(reads) if reads else None
            row = dict(pan=pan, reads=reads, measured_cm=measured)
            if pose is not None:
                ang = pose["yaw"] + math.radians(pan)
                ox = pose["x"] + pivot * math.cos(pose["yaw"]) + arm * math.cos(ang)
                oy = pose["y"] + pivot * math.sin(pose["yaw"]) + arm * math.sin(ang)
                centre, what = ray_hit(ox, oy, ang, width, height, blocks)
                cone = min((ray_hit(ox, oy, ang + math.radians(d), width, height, blocks)
                            for d in np.linspace(-CONE_DEG, CONE_DEG, 13)), key=lambda h: h[0])
                err = None if measured is None else measured - cone[0] * 100
                row.update(origin=(ox, oy), angle=ang, expected_centre_cm=centre * 100, centre_hit=what,
                           expected_cone_cm=cone[0] * 100, cone_hit=cone[1], error_cm=err,
                           ok=err is not None and abs(err) <= 5.0)
                print(f"  {pan:+4d}   {'-' if measured is None else f'{measured:6.1f}'} cm   "
                      f"{centre * 100:6.1f} ({what:7s})   {cone[0] * 100:6.1f} ({cone[1]:7s})       "
                      f"{'-' if err is None else f'{err:+6.1f}'}")
            else:
                print(f"  {pan:+4d}   {'-' if measured is None else f'{measured:6.1f}'} cm")
            rows.append(row)
        set_pan(servo, home)
        if pose is not None:
            good = [r for r in rows if r.get("error_cm") is not None]
            ok = [r for r in good if r["ok"]]
            print(f"  within 5 cm of the map: {len(ok)}/{len(rows)} angles"
                  + (f"; median |error| {st.median(abs(r['error_cm']) for r in good):.1f} cm" if good else ""))
            missed = [r for r in rows if r.get("cone_hit", "").startswith("block") and not r["ok"]]
            if missed:
                print("  angles where a block is the nearest in the beam but the reading does not match: "
                      + ", ".join(f"{r['pan']:+d} ({r['measured_cm']} vs {r['expected_cone_cm']:.0f})" for r in missed))
            sweep_plot(out / "sonic_sweep.png", pose, rows, width, height, blocks, body)
        summary["sonic"] = [{k: v for k, v in r.items() if k not in ("origin", "angle")} for r in rows]
        # 4. IR
        ir = {k: int(v) for k, v in sensors.read_ir().items()}
        print(f"\n== 4. IR (1 = something within its range) ==\n  {ir}")
        summary["ir"] = ir
    finally:
        try:
            set_pan(servo, home)
        except Exception:  # noqa: BLE001
            pass
        for g in grabbers.values():
            g.close()
        for cap in caps.values():
            cap.release()
        try:
            sensors.close()
        except Exception:  # noqa: BLE001
            pass
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
