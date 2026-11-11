#!/usr/bin/env python3
"""How well do the IMU and the cameras carry the pose? Drives a short path and compares the odometry at every stop.

    python3 tools/vio_check.py                  # 2 rounds; the car MOVES
    python3 tools/vio_check.py --rounds 3
    python3 tools/vio_check.py --replay DIR     # compute again from a saved run (route_follow_debug/vio/<time>)

Put the car in the middle of the arena, squared to a wall, no blocks: 60 cm or more ahead and 50 cm or more to each side.
One round: forward, strafe left, reverse, strafe right, turn left 90, turn right 90; about 1 s standing after each.

Every picture taken while the car moves and every gyro frame are logged. The pose of the tags while it stands (both
cameras) is the reference at each stop and is NOT given to the odometry, which starts from the first stop and runs
through the whole path without ever stopping to look:
  speed      calibrated speed x time, the heading from the turn rates (no IMU, no camera)
  gyro       the same with the heading from the gyro
  gyro+tags  the same with the tag poses of the pictures taken during the moves (vio.py)
"""
import argparse
import json
import math
import pathlib
import statistics as st
import sys
import threading
import time

import numpy as np

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from pose_kalman import wrap  # noqa: E402
from vio import VioFilter  # noqa: E402

MOVE_S = 1.5
STAND_S = 1.2
SETTLE_S = 0.4
LATENCY_S = 0.12               # a picture is stamped when it arrives; two real runs agree on 0.12-0.15 s (2026-11-09)
LATENCIES = (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.20)
PATH = (("forward", None), ("strafe_left", None), ("reverse", None), ("strafe_right", None),
        ("left", 90.0), ("right", -90.0))


class TagLogger:
    """The tag pose of every new picture of the front and rear cameras, with its time."""

    def __init__(self, car):
        self.car = car
        self.rows: list[dict] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        seen = {}
        loc = self.car.localizer
        while not self.stop.is_set():
            idle = True
            for name, grabber in self.car.grabbers.items():
                frame, stamp = grabber.latest()
                if frame is None or stamp == seen.get(name):
                    continue
                seen[name] = stamp
                idle = False
                fixes = [f for f in loc.observe(name, frame)[0] if f.usable]
                pose = loc.fuse(fixes, stamp) if fixes else None
                weak = None
                if pose is None and fixes:      # one far tag: no joint fit, but its own pose is still a (loose) measurement
                    near = min(fixes, key=lambda f: f.distance)
                    weak = dict(x=near.x, y=near.y, yaw=near.yaw, tags=1, nearest=near.distance, rms=near.err_px)
                self.rows.append(dict(t=stamp, camera=name, seen=len(fixes), weak=weak, pose=None if pose is None else dict(
                    x=pose.x, y=pose.y, yaw=pose.yaw, tags=len({f.tag_id for f in pose.fixes}),
                    nearest=min(f.distance for f in pose.fixes), rms=pose.rms_px if hasattr(pose, "rms_px") else None)))
            if idle:
                time.sleep(0.005)


def body_speed(kind: str, motion: dict) -> tuple[float, float]:
    f, s = float(motion["forward_mps"]), float(motion["strafe_mps"])
    return {"forward": (f, 0.0), "reverse": (-f, 0.0), "strafe_left": (0.0, s), "strafe_right": (0.0, -s)}.get(kind, (0.0, 0.0))


def stand_pose(log: dict, stop: dict):
    """The pose while the car stands: the median of each camera, then the mean of the cameras."""
    per = {}
    for r in log["tags"]:
        if r["pose"] and stop["t0"] <= r["t"] <= stop["t1"]:
            per.setdefault(r["camera"], []).append(r["pose"])
    if not per:
        return None
    x, y, ref, offs = [], [], None, []
    for rows in per.values():
        ref = rows[0]["yaw"] if ref is None else ref
        x.append(st.median(p["x"] for p in rows))
        y.append(st.median(p["y"] for p in rows))
        offs.append(st.median(wrap(p["yaw"] - ref) for p in rows))
    return sum(x) / len(x), sum(y) / len(y), wrap(ref + sum(offs) / len(offs))


def run(log: dict, mode: str, latency: float = LATENCY_S):
    """The pose at each stop. mode: 'speed', 'gyro', 'tags'."""
    motion = log["motion"]
    first = stand_pose(log, log["stops"][0])
    moves, stops = log["moves"], log["stops"]
    # one stream ordered by time: gyro frames, and the tag poses of the pictures taken during the moves
    events = [(row[0], 0, row[1]) for row in log["imu"]]
    if mode == "tags":
        for r in log["tags"]:
            t = r["t"] - latency
            if r["pose"] and any(m["t0"] <= t <= m["t1"] for m in moves):
                events.append((t, 1, r["pose"]))
    events.sort(key=lambda e: (e[0], e[1]))
    f = VioFilter(*first)
    bias = math.radians(log["gyro_bias_dps"])
    out, stop_i, last = [], 1, stops[0]["t1"]
    active = lambda t: next((m for m in moves if m["t0"] <= t <= m["t0"] + m["elapsed"]), None)
    for t, kind, value in events:
        if t <= stops[0]["t1"]:
            continue
        while stop_i < len(stops) and t >= stops[stop_i]["t0"]:
            out.append((float(f.x[0]), float(f.x[1]), float(f.x[2])))
            stop_i += 1
        dt = min(t - last, 0.1)
        last = t
        move = active(t)
        vx, vy = body_speed(move["kind"], motion) if move else (0.0, 0.0)
        if kind == 0:
            if mode == "speed":
                rate = 0.0
                if move and move["kind"] in ("left", "right"):
                    rate = math.radians(motion["turn_left_dps"] if move["kind"] == "left" else -motion["turn_right_dps"])
            else:
                rate = math.radians(value) - bias
            f.predict(dt, vx, vy, rate)
        else:
            f.predict(dt, vx, vy, 0.0)
            f.update_tags(value["x"], value["y"], value["yaw"], value["tags"], value["nearest"], math.hypot(vx, vy))
    while stop_i < len(stops):
        out.append((float(f.x[0]), float(f.x[1]), float(f.x[2])))
        stop_i += 1
    return out, f


def report(log: dict) -> None:
    truth = [stand_pose(log, s) for s in log["stops"][1:]]
    results = {m: run(log, m)[0] for m in ("speed", "gyro", "tags")}
    print("\nstop  after          reference (cm, deg)        position error (cm)        heading error (deg)")
    print("                                          speed  gyro  gyro+tags     speed  gyro  gyro+tags")
    err = {m: [] for m in results}
    for i, t in enumerate(truth):
        if t is None:
            continue
        cells_p, cells_h = [], []
        for m in results:
            p = results[m][i]
            ep, eh = math.hypot(p[0] - t[0], p[1] - t[1]) * 100.0, abs(math.degrees(wrap(p[2] - t[2])))
            err[m].append((ep, eh))
            cells_p.append(f"{ep:5.1f}")
            cells_h.append(f"{eh:5.1f}")
        print(f"{i + 1:3d}  {log['moves'][i]['kind']:13s} {t[0] * 100:6.1f} {t[1] * 100:6.1f} {math.degrees(t[2]):+7.1f}"
              f"   {' '.join(cells_p)}      {' '.join(cells_h)}")
    print("\nover the whole path (median / max):")
    for m, label in (("speed", "speed"), ("gyro", "gyro"), ("tags", "gyro+tags")):
        a = np.array(err[m])
        print(f"  {label:10s} position {np.median(a[:, 0]):5.1f} / {a[:, 0].max():5.1f} cm    "
              f"heading {np.median(a[:, 1]):5.1f} / {a[:, 1].max():5.1f} deg")
    moving = [r for r in log["tags"] if any(m["t0"] <= r["t"] <= m["t1"] for m in log["moves"])]
    drive = sum(m["t1"] - m["t0"] for m in log["moves"])
    print(f"\npictures during the moves: {len(moving)} in {drive:.0f} s; with a tag pose {sum(1 for r in moving if r['pose'])}")
    _out, f = run(log, "tags")
    print(f"tag poses used {f.fused}, refused {f.rejected}; speed scale at the end: forward {f.x[3]:.3f}, sideways {f.x[4]:.3f}")
    print("time of a picture before its arrival (s): position median, heading median")
    for lat in LATENCIES:
        out, _f = run(log, "tags", lat)
        e = [(math.hypot(p[0] - t[0], p[1] - t[1]) * 100.0, abs(math.degrees(wrap(p[2] - t[2])))) for p, t in zip(out, truth) if t]
        print(f"  {lat:.2f}: {st.median(v[0] for v in e):5.1f} cm  {st.median(v[1] for v in e):5.1f} deg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--replay", type=pathlib.Path)
    opts = ap.parse_args()
    if opts.replay:
        report(json.loads((opts.replay / "log.json").read_text()))
        return
    from calibrate_motion import wall_cm
    from imu_motion_check import Recorder
    from route_follow import Abort, RouteFollower, parse_args

    car = RouteFollower(parse_args(["--arm", "--no-window", "--debug-dir", "route_follow_debug/calibrate"]))
    out = SERVER / "route_follow_debug" / "vio" / time.strftime("%Y%m%d-%H%M%S")
    log = dict(moves=[], stops=[])
    rec = tags = None
    try:
        car.start()
        if car.gyro is None:
            raise Abort("no IMU: plug it in and keep the car still while the program starts")
        ahead, to_left, to_right = wall_cm(car, 0.0), wall_cm(car, 90.0), wall_cm(car, -90.0)
        car.look(0.0)
        print(f"placement: wall ahead {ahead:.0f} cm, left {to_left:.0f} cm, right {to_right:.0f} cm "
              f"(needs ahead >= 60, each side >= 50)", flush=True)
        if ahead < 60 or min(to_left, to_right) < 50:
            raise Abort("move the car: too close to a wall (nothing has moved yet)")
        rec, tags = Recorder(car.gyro.imu), TagLogger(car)

        def stand() -> None:
            time.sleep(SETTLE_S)
            t0 = time.monotonic()
            time.sleep(STAND_S)
            log["stops"].append(dict(t0=t0, t1=time.monotonic()))

        stand()
        for n in range(1, opts.rounds + 1):
            for kind, angle in PATH:
                duration = MOVE_S if angle is None else abs(angle) / car.turn_rate(kind)
                t0 = time.monotonic()
                elapsed, reason = car.move(kind, duration, None if angle is None else math.radians(angle))
                log["moves"].append(dict(kind=kind, t0=t0, t1=time.monotonic(), elapsed=elapsed, reason=reason))
                print(f"round {n} {kind:13s} {elapsed:.2f} s{' stopped: ' + reason if reason else ''}", flush=True)
                stand()
    except Abort as exc:
        print(f"ABORT: {exc}", flush=True)
    finally:
        if tags:
            tags.stop.set()
            tags.thread.join(timeout=2.0)
        if rec:
            rec.stop.set()
        bias = car.gyro.bias_dps if car.gyro is not None else 0.0
        car.close()
    if not log["moves"]:
        return
    log.update(imu=[(r[0], r[4]) for r in rec.rows], tags=tags.rows, motion=dict(car.motion, strafe_mps=car.strafe_mps), gyro_bias_dps=bias)
    out.mkdir(parents=True, exist_ok=True)
    (out / "log.json").write_text(json.dumps(log))
    print(f"saved {out}/log.json", flush=True)
    report(log)


if __name__ == "__main__":
    main()
