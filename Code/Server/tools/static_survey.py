#!/usr/bin/env python3
"""Static survey: at poses you measured with a ruler, record what every camera, the ultrasonic and the IR see.

    python3 tools/static_survey.py [label]          # asks for each pose; q to finish
    python3 tools/static_survey.py test --pose 50 23 0   # one capture, no questions

Put the car by hand on a spot, measure the CENTRE of the car (x from the left wall, y from the bottom wall, cm; the
inner wall faces) and the heading. Keep the heading at 0, 90, 180 or -90 by squaring the car to a wall: the distance
from the car's side to the wall must be the same at the front and the rear wheel. Heading in degrees, 0 = towards the
right wall (+x), 90 = towards the top wall. The program does not move the car.

For every pose it saves (route_follow_debug/survey/<label>/<n>_x<cm>_y<cm>_h<deg>/):
  <camera>.jpg        first frame of each of the 5 cameras (front/rear 1280x960, USB 640x480)
  obs.json            the truth you typed, the tags each camera found (raw and normalised corners, 3 frames each),
                      ultrasonic readings at 7 angles, the 6 IR states, the battery voltage
and prints, per camera, which tags it saw and how far the car position from that camera's tags is from your pose.
solve_mounts.py (next) uses obs.json to check the cameras' mounting and intrinsics.

The gimbal is turned straight ahead first. Nothing else may be using the cameras (no route_follow).
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, load_json, open_grabbers  # noqa: E402

SERVER = Path(__file__).resolve().parents[1]
OUT = SERVER / "route_follow_debug" / "survey"
USB_CAMERAS = ("gimbal", "left", "right")
SONIC_ANGLES = (90, 60, 30, 0, -30, -60, -90)
WARMUP_S = 2.0   # auto exposure and white balance settle (measured: about 1 s after opening)
FLUSH_S = 0.4    # longer than the frames the driver queues while nobody reads (4 at 30 fps)


def usb_path(bus: str) -> Path:
    """'usb-xhci-hcd.0-1.2' -> /dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1.2:1.0-video-index0 (the video node changes)."""
    ctrl, port = bus.replace("usb-", "", 1).rsplit("-", 1)
    return Path("/dev/v4l/by-path") / f"platform-{ctrl}-usb-0:{port}:1.0-video-index0"


def open_usb(hardware: dict) -> dict:
    caps = {}
    for name in USB_CAMERAS:
        path = usb_path(hardware[name]["usb_bus"])
        cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print(f"  WARNING: {name} ({path.name}) did not open", flush=True)
            continue
        fresh_read(cap, WARMUP_S)
        caps[name] = cap
    return caps


def fresh_read(cap, seconds: float = FLUSH_S):
    """Read for `seconds` and return the last frame. A USB camera nobody reads keeps the frames the driver queued
    back then (dark and off-colour right after opening, or from where the car stood before): those are read away."""
    ok, frame = False, None
    end = time.monotonic() + seconds
    while True:
        got, f = cap.read()
        if got:
            ok, frame = True, f
        if time.monotonic() >= end:
            return ok, frame


def read_with_timeout(read, timeout: float = 1.5):
    """The gpiozero ultrasonic can block for ever when no echo comes back: give it a time limit."""
    box = []
    worker = threading.Thread(target=lambda: box.append(read()), daemon=True)
    worker.start()
    worker.join(timeout)
    return box[0] if box else None


def pose_from_edges(heading: int, ahead, behind, left, right, width_cm: float, height_cm: float,
                    length_cm: float, body_w_cm: float):
    """Centre pose from distances measured from the BODY EDGES to the walls (cm; None = not measured).

    ahead = nose to the wall in front, behind = tail to the wall behind, left/right = the car's side to the wall on
    that side. Heading 0 faces the right wall (+x), 90 the top wall, 180 the left wall, -90 the bottom wall. Each axis
    gets one estimate per wall that was measured; the spread between the two is the check of your tape, the body size
    and the arena size. Returns (x, y, notes).
    """
    half_l, half_w = length_cm / 2.0, body_w_cm / 2.0
    heading = int(round(heading)) % 360
    xs, ys = [], []
    if heading == 0:
        if ahead is not None: xs.append(width_cm - ahead - half_l)
        if behind is not None: xs.append(behind + half_l)
        if left is not None: ys.append(height_cm - left - half_w)
        if right is not None: ys.append(right + half_w)
    elif heading == 90:
        if ahead is not None: ys.append(height_cm - ahead - half_l)
        if behind is not None: ys.append(behind + half_l)
        if left is not None: xs.append(left + half_w)
        if right is not None: xs.append(width_cm - right - half_w)
    elif heading == 180:
        if ahead is not None: xs.append(ahead + half_l)
        if behind is not None: xs.append(width_cm - behind - half_l)
        if left is not None: ys.append(left + half_w)
        if right is not None: ys.append(height_cm - right - half_w)
    elif heading == 270:
        if ahead is not None: ys.append(ahead + half_l)
        if behind is not None: ys.append(height_cm - behind - half_l)
        if left is not None: xs.append(width_cm - left - half_w)
        if right is not None: xs.append(right + half_w)
    else:
        raise ValueError("edge distances need heading 0, 90, 180 or -90")
    if not xs or not ys:
        raise ValueError("need at least one distance on each axis")
    notes = []
    for axis, values in (("x", xs), ("y", ys)):
        if len(values) == 2:
            notes.append(f"{axis}: the two walls give {values[0]:.1f} and {values[1]:.1f} cm (differ by {abs(values[0] - values[1]):.1f})")
    return sum(xs) / len(xs), sum(ys) / len(ys), notes


def set_pan(servo, value, tries: int = 8) -> None:
    """The servo driver sometimes answers 'Resource temporarily unavailable' on the I2C bus: try again."""
    for attempt in range(tries):
        try:
            servo.set_pan(value)
            return
        except BlockingIOError:
            time.sleep(0.05 * (attempt + 1))
    servo.set_pan(value)


def tags_json(found: dict) -> dict:
    return {str(i): {"raw": raw.tolist(), "norm": norm.tolist()} for i, (raw, norm) in found.items()}


def car_pose_from_fixes(fixes) -> tuple[float, float, float] | None:
    """Median x, y, heading of the single-tag fits of one camera."""
    if not fixes:
        return None
    ref = fixes[0].yaw
    yaw = ref + float(np.median([(f.yaw - ref + math.pi) % (2 * math.pi) - math.pi for f in fixes]))
    return float(np.median([f.x for f in fixes])), float(np.median([f.y for f in fixes])), yaw


def capture(loc, grabbers, caps, servo, sensors, pose, folder: Path, hardware_ok: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frames = {name: [] for name in ("front", "rear") + USB_CAMERAS}
    for cap in caps.values():
        fresh_read(cap)
    for _ in range(3):
        for name, grabber in grabbers.items():
            frame, _stamp = grabber.latest()
            if frame is not None:
                frames[name].append(frame.copy())
        for name, cap in caps.items():
            ok, frame = cap.read()
            if ok:
                frames[name].append(frame.copy())
        time.sleep(0.25)
    obs = {"pose_cm_deg": list(pose), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cameras": {}}
    x, y, yaw = pose[0] / 100.0, pose[1] / 100.0, math.radians(pose[2])
    print("  camera   tags seen                      position from its tags   error vs your pose", flush=True)
    for name, shots in frames.items():
        if not shots:
            print(f"  {name:8s} no frames", flush=True)
            continue
        cv2.imwrite(str(folder / f"{name}.jpg"), shots[0])
        h, w = shots[0].shape[:2]
        entry = {"size": [w, h], "frames": []}
        seen, fixes = set(), []
        for frame in shots:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, _fx = loc.detect(name, gray)
            entry["frames"].append(tags_json(found))
            seen.update(found)
            if name in loc.mounts:
                fixes += [f for f in loc.observe(name, frame)[0] if f.usable]
        obs["cameras"][name] = entry
        est = car_pose_from_fixes(fixes)
        if est is None:
            where = "-"
            err = ""
        else:
            where = f"x={est[0] * 100:5.1f} y={est[1] * 100:5.1f} h={math.degrees(est[2]):+6.1f}"
            dyaw = (est[2] - yaw + math.pi) % (2 * math.pi) - math.pi
            err = f"dx={(est[0] - x) * 100:+5.1f} dy={(est[1] - y) * 100:+5.1f} dh={math.degrees(dyaw):+5.1f}"
        print(f"  {name:8s} {str(sorted(seen)):30s} {where:24s} {err}", flush=True)
    obs["sonic_cm"], obs["ir"], obs["battery_v"] = {}, {}, None
    if hardware_ok:
        from servo import load_gimbal_home, load_left_90_pan, load_right_90_pan, servo_pan_for_bearing
        home, _tilt = load_gimbal_home()
        right90, left90 = load_right_90_pan(home), load_left_90_pan()
        for angle in SONIC_ANGLES:
            set_pan(servo, servo_pan_for_bearing(home, right90, angle, left90))
            time.sleep(0.9)
            reads = [read_with_timeout(sensors.read_sonic) for _ in range(5)]
            obs["sonic_cm"][str(angle)] = [r for r in reads if r is not None]
        set_pan(servo, home)
        obs["ir"] = {k: int(v) for k, v in sensors.read_ir().items()}
        try:
            from safe_surround_cruise import read_battery_voltage
            obs["battery_v"] = read_battery_voltage()
        except Exception:
            pass
    (folder / "obs.json").write_text(json.dumps(obs) + "\n", encoding="utf-8")
    ir = [k for k, v in obs["ir"].items() if v and k in ("front_left", "front_right", "wheel_left_front",
                                                          "wheel_left_rear", "wheel_right_front", "wheel_right_rear")]
    print(f"  sonic ahead {obs['sonic_cm'].get('0')}  IR lit: {ir or 'none'}  battery {obs['battery_v']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--pose", nargs=3, type=float, metavar=("X_CM", "Y_CM", "YAW_DEG"))
    args = ap.parse_args()
    arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
    hardware = load_json(SERVER / "camera_hardware.json")
    loc = ArucoLocalizer(arena, car)
    grabbers = open_grabbers(car, 1280, 960, rear=True)
    caps = open_usb(hardware)
    servo = sensors = None
    hardware_ok = False
    try:
        from safe_surround_cruise import SafetySensors
        from servo import Servo
        servo = Servo()
        sensors = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=False, active_high=False))
        hardware_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: servo/ultrasonic/IR unavailable ({exc}); cameras only", flush=True)
    time.sleep(2.0)
    folder_root = OUT / args.label
    n = len(list(folder_root.glob("*_x*"))) if folder_root.exists() else 0
    try:
        while True:
            if args.pose:
                pose = tuple(args.pose)
            else:
                text = input("\npose: 'e HEADING AHEAD BEHIND LEFT RIGHT' (cm from the body edges to the walls, - = not measured)"
                             "\n      or 'X_CM Y_CM HEADING' (car centre)   (q = finish) > ").strip()
                if text.lower() in ("q", "quit", ""):
                    break
                try:
                    parts = text.replace(",", " ").split()
                    if parts[0].lower() == "e":
                        vals = [None if v == "-" else float(v) for v in parts[2:6]]
                        w, h = (float(v) * 100.0 for v in arena["arena_size_m"])
                        bl, bw = (float(v) * 100.0 for v in car.get("body_m", [0.29, 0.18]))
                        cx, cy, notes = pose_from_edges(int(float(parts[1])), *vals, w, h, bl, bw)
                        pose = (cx, cy, float(parts[1]))
                        print(f"  centre of the car: x={cx:.1f} cm  y={cy:.1f} cm  heading {pose[2]:.0f}")
                        for note in notes:
                            print("  " + note)
                    else:
                        pose = tuple(float(v) for v in parts)
                        assert len(pose) == 3
                except (ValueError, AssertionError, IndexError) as exc:
                    print(f"  not understood ({exc}); e.g.  e 0 40 95 62 58   or   70 70 0")
                    continue
            n += 1
            name = f"{n:02d}_x{pose[0]:.0f}_y{pose[1]:.0f}_h{pose[2]:.0f}"
            print(f"capturing {name} ... keep the car still", flush=True)
            capture(loc, grabbers, caps, servo, sensors, pose, folder_root / name, hardware_ok)
            if args.pose:
                break
    finally:
        for g in grabbers.values():
            try:
                g.close()
            except Exception:
                pass
        for cap in caps.values():
            cap.release()
        if servo is not None:
            try:
                servo.release_gimbal()
            except Exception:
                pass
        if sensors is not None:
            sensors.close()
    print(f"saved under {folder_root}")


if __name__ == "__main__":
    main()
