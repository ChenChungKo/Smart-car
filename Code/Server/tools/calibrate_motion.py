#!/usr/bin/env python3
"""Calibrate forward, strafe and turn speed at the battery you run with. The median of several trials is written.

    python3 tools/calibrate_motion.py            # measure, print, ask before writing car_nav.json
    python3 tools/calibrate_motion.py --trials 5 --yes

Put the car in the middle of the arena facing a wall squarely with 80 cm or more of free floor in front and 40 cm or more to
its left and right, with the arena tags in view of the front camera (the turn trials read the heading from them).

Distances come from the ultrasonic against the walls, good to about 1 cm; a tag fix 1 m away is good to 5-8 cm,
which is why route_follow's one-second tag calibration was noisy. Forward and strafe are timed moves of 1.5 s,
measured as the change of the wall distance; turns are 1.0 s, measured as the change of the tag heading.
Only forward_mps, strafe_mps, turn_left_dps and turn_right_dps change; a backup of car_nav.json is kept.
"""
import argparse
import json
import math
import pathlib
import shutil
import statistics as st
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aruco_localizer import wrap_rad  # noqa: E402
from route_follow import Abort, RouteFollower, parse_args  # noqa: E402


def wall_cm(car: RouteFollower, pan: float, reads: int = 5) -> float:
    values = []
    car.look(pan)
    for _ in range(reads):
        v = car.sensors.read_sonic()
        if v is not None:
            values.append(v)
        time.sleep(0.08)
    if len(values) < 3:
        raise Abort(f"ultrasonic gave {len(values)} readings at pan {pan:+.0f}: wall too far or too close?")
    return st.median(values)


def square_up(car: RouteFollower) -> float | None:
    """Turn the car back to the nearest wall direction by the tags (a strafe turns it); returns the heading error left."""
    for _ in range(3):
        fix = car.localize(3.0)
        if fix is None:
            return None
        car.pose = fix
        target = round(fix.yaw / (math.pi / 2)) * (math.pi / 2)
        err = wrap_rad(target - fix.yaw)
        if abs(err) < math.radians(2.0):
            return math.degrees(err)
        car.turn_by(err, min_s=0.05)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--yes", action="store_true", help="write car_nav.json without asking")
    opts = ap.parse_args()
    car = RouteFollower(parse_args(["--arm", "--no-window", "--debug-dir", "route_follow_debug/calibrate"]))
    try:
        car.start()
        forward, strafe, left, right = [], [], [], []
        for n in range(1, opts.trials + 1):
            # forward then back the same time: the car stays near where it was
            if square_up(car) is None:
                print("  (no tag fix to square the car up; carrying on)", flush=True)
            d0 = wall_cm(car, 0.0)
            if d0 < 50:
                raise Abort(f"only {d0:.0f}cm to the wall ahead (the trial needs 50); move the car back")
            t, _ = car.move("forward", 1.5)
            d1 = wall_cm(car, 0.0)
            car.move("reverse", 1.5)
            forward.append((d0 - d1) / 100.0 / t)
            # strafe left, then right: the left wall distance grows then shrinks
            for kind, pan, sign in (("strafe_left", 90.0, -1.0), ("strafe_right", -90.0, -1.0)):
                s0 = wall_cm(car, pan)
                if s0 < 40:
                    raise Abort(f"only {s0:.0f}cm to the wall at {pan:+.0f} deg; move the car to the middle")
                t, _ = car.move(kind, 1.5)
                s1 = wall_cm(car, pan)
                strafe.append(abs(s1 - s0) / 100.0 / t)
                car.move("strafe_right" if kind == "strafe_left" else "strafe_left", 1.5)
                square_up(car)  # a strafe turns the car: face the wall again before the next trial
            # turns, by the tag heading
            p0 = car.localize(3.0)
            if p0 is None:
                raise Abort("no tag fix: face the car so the front camera sees tags")
            t, _ = car.move("left", 1.0)
            p1 = car.localize(3.0)
            t2, _ = car.move("right", 1.0)
            p2 = car.localize(3.0)
            if p1 is None or p2 is None:
                raise Abort("lost the tags during a turn trial")
            left.append(math.degrees(abs(wrap_rad(p1.yaw - p0.yaw))) / t)
            right.append(math.degrees(abs(wrap_rad(p2.yaw - p1.yaw))) / t2)
            print(f"trial {n}: forward {forward[-1]:.3f} m/s  strafe {strafe[-2]:.3f}/{strafe[-1]:.3f} m/s  "
                  f"left {left[-1]:.1f}  right {right[-1]:.1f} deg/s", flush=True)
        result = dict(forward_mps=round(st.median(forward), 4), strafe_mps=round(st.median(strafe), 4),
                      turn_left_dps=round(st.median(left), 2), turn_right_dps=round(st.median(right), 2))
        old = car.motion
        print("\nmedian of the trials (now in car_nav.json):")
        for k, v in result.items():
            print(f"  {k:15s} {v:8}   (was {old.get(k)})")
        spread = (max(forward) - min(forward)) / st.median(forward)
        if spread > 0.15:
            print(f"WARNING: the forward trials differ by {spread * 100:.0f} %; check the floor and repeat.")
        if not opts.yes and input("\nWrite these to car_nav.json? [y/N] ").strip().lower() != "y":
            print("not written")
            return
        path = car.car_path
        shutil.copy(path, str(path) + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        car.motion.update(result)
        car.motion["calibrated"] = True
        car.car["motion"] = car.motion
        path.write_text(json.dumps(car.car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} (backup next to it)")
    except Abort as exc:
        print(f"ABORT: {exc}")
    finally:
        car.close()


if __name__ == "__main__":
    main()
