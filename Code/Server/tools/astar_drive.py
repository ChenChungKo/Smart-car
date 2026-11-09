#!/usr/bin/env python3
"""Plan with A* to a point and drive there without stopping, keeping the car square. Dry-run unless --arm.

    python3 tools/astar_drive.py --goal 110 110          # plans from where the car stands; nothing moves
    python3 tools/astar_drive.py --goal 110 110 --arm    # drives
    options: --no-blocks  --margin 5  --speed 0.16
    python3 tools/astar_drive.py --replay DIR            # the filter again on a saved run, with other tag weights

The heading stays at the nearest multiple of 90 deg: the mecanum wheels move the car along the path sideways and
diagonally without turning, and every picture sees the tags (they are lost while the car spins). All the time, about
50 times a second: the pose is carried by the gyro and the commanded speed and corrected by every tag pose (vio.py),
the wheels are commanded towards a point a little further along the path, and the heading error is corrected.

It stops by itself: IMU silent, no tag pose for 2 s, more than 8 cm off the path, something ahead that is not on the
map (ultrasonic: closer than 18 cm and 5 cm closer than the map), anything within 6 cm ahead, or out of time. Output: route_follow_debug/astar/<time>/ plan.png, result.png, log.json (with the tag poses of the run).
"""
import argparse
import collections
import json
import math
import pathlib
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from astar import Grid, expected_range, plan  # noqa: E402
from pose_kalman import wrap  # noqa: E402
from route_common import HOLD_FULL_DEG, HOLD_MAX, SONIC_AHEAD_M  # noqa: E402
from route_follow import Abort, RouteFollower, parse_args  # noqa: E402
from vio import VioFilter  # noqa: E402
from vio_check import LATENCY_S, TagLogger, stand_pose  # noqa: E402

LAP_BLOCKS = [(0.65, 0.75, 0.0, 0.20), (1.20, 1.39, 0.65, 0.75), (0.65, 0.75, 1.20, 1.40), (0.0, 0.20, 0.65, 0.75)]
LOOKAHEAD_M = 0.08
GOAL_TOL_M = 0.025
SLOW_M = 0.12            # slow down inside this distance of the goal
MIN_SPEED = 0.10         # m/s: below this the wheels may not turn
MAX_CROSS_M = 0.08
NO_TAGS_S = 2.0
CLOSER_THAN_MAP_M = 0.05   # ultrasonic: something this much closer than the walls and the known blocks is not on the map
STOP_UNKNOWN_M = 0.18      # ... and it stops the car when it is within this
STOP_ANY_M = 0.06          # anything this close stops the car, on the map or not
PWM_LIMIT = 2000
SCALE = 3                # picture pixels per cm
STAND_S = 1.5            # the pose of the start and of the end: this long of pictures while the car stands


def along_path(path: list, p: tuple[float, float]):
    """(index of the segment, fraction, distance from the path, distance left to the goal) of the point nearest p."""
    best = None
    for i in range(len(path) - 1):
        a, b = np.array(path[i]), np.array(path[i + 1])
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, float((np.array(p) - a) @ ab) / L2))
        d = float(np.hypot(*(np.array(p) - (a + t * ab))))
        if best is None or d < best[0]:
            best = (d, i, t)
    d, i, t = best
    left = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) * (1.0 - t)
    left += sum(math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1]) for k in range(i + 1, len(path) - 1))
    return i, t, d, left


def point_ahead(path: list, i: int, t: float, dist: float) -> tuple[float, float]:
    x, y = path[i][0] + (path[i + 1][0] - path[i][0]) * t, path[i][1] + (path[i + 1][1] - path[i][1]) * t
    while i < len(path) - 1:
        nx, ny = path[i + 1]
        seg = math.hypot(nx - x, ny - y)
        if seg >= dist and seg > 1e-9:
            return x + (nx - x) * dist / seg, y + (ny - y) * dist / seg
        dist -= seg
        x, y = nx, ny
        i += 1
    return path[-1]


def standing_pose(tags: TagLogger, seconds: float = STAND_S):
    """(x, y, yaw) from the tags while the car stands: per camera the median of its pictures, then the mean of the cameras."""
    t0 = time.monotonic()
    time.sleep(seconds)
    got = stand_pose(dict(tags=list(tags.rows)), dict(t0=t0, t1=time.monotonic()))
    return None if got is None else SimpleNamespace(x=got[0], y=got[1], yaw=got[2])


def apply_tag(f: VioFilter, hist, z: dict, at: float, speed: float) -> bool:
    """A tag pose of a picture taken at time `at`, older than the state: moved forward by what happened since."""
    then = min(hist, key=lambda h: abs(h[0] - at))
    return f.update_tags(z["x"] + f.x[0] - then[1], z["y"] + f.x[1] - then[2], wrap(z["yaw"] + float(f.x[2]) - then[3]),
                         z["tags"], z["nearest"], speed)


def replay(log: dict, tag_sigma: float, latency: float) -> VioFilter:
    """The filter of the run again from the log: the commands and the gyro as they were, other tag weight or delay."""
    rows, tag_rows = log["rows"], [r for r in log["tags"] if r["pose"]]
    f = VioFilter(*log["start"], tag_sigma=tag_sigma)
    hist = collections.deque(maxlen=120)
    last, vx, vy, n = rows[0]["t"], 0.0, 0.0, 0
    for r in rows:
        f.predict(r["t"] - last, vx, vy, r["omega"])
        last = r["t"]
        hist.append((last, float(f.x[0]), float(f.x[1]), float(f.x[2])))
        while n < len(tag_rows) and tag_rows[n]["t"] <= last:
            apply_tag(f, hist, tag_rows[n]["pose"], tag_rows[n]["t"] - latency, math.hypot(vx, vy))
            n += 1
        vx, vy = r["vx"], r["vy"]
    return f


def report_replay(log: dict) -> None:
    end, goal = log.get("end"), log["goal"]
    if not end or "tags" not in log:
        print("this log has no tag poses or no standing pose at the end")
        return
    print(f"standing at the end {end[0] * 100:.1f}, {end[1] * 100:.1f} cm; the filter at the last cycle against it (cm):")
    print("tag sigma x   delay 0.06    0.09    0.12    0.15     (scale forward, sideways at 0.06)")
    for mult in (1.0, 0.5, 0.25, 0.1):
        cells, first = [], None
        for lat in (0.06, 0.09, 0.12, 0.15):
            f = replay(log, mult, lat)
            first = first or f
            cells.append(f"{math.hypot(f.x[0] - end[0], f.x[1] - end[1]) * 100:6.1f}")
        print(f"  {mult:5.2f}      " + "  ".join(cells) + f"      {first.x[3]:.3f}, {first.x[4]:.3f}  used {first.fused} refused {first.rejected}")


def picture(grid: Grid, arena, blocks, path, trail=None) -> np.ndarray:
    k = SCALE * 100
    H = int(arena[1] * k) + 40
    img = np.full((H, int(arena[0] * k) + 40, 3), 245, np.uint8)
    px = lambda x, y: (20 + int(x * k), H - 20 - int(y * k))
    forbidden = cv2.resize(grid.blocked.T.astype(np.uint8), (int(arena[0] * k), int(arena[1] * k)), interpolation=cv2.INTER_NEAREST)
    forbidden = cv2.flip(forbidden, 0) > 0
    img[20:20 + forbidden.shape[0], 20:20 + forbidden.shape[1]][forbidden] = (225, 225, 235)
    cv2.rectangle(img, px(0, 0), px(*arena), (40, 40, 40), 2)
    for x0, x1, y0, y1 in blocks:
        cv2.rectangle(img, px(x0, y0), px(x1, y1), (120, 120, 220), -1)
    for a, b in zip(path, path[1:]):
        cv2.line(img, px(*a), px(*b), (200, 140, 30), 2, cv2.LINE_AA)
    if trail:
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, px(a[0], a[1]), px(b[0], b[1]), (40, 40, 220), 2, cv2.LINE_AA)
    cv2.circle(img, px(*path[0]), 6, (0, 160, 0), -1)
    cv2.circle(img, px(*path[-1]), 6, (0, 0, 200), 2)
    return img


def follow(car, tags: TagLogger, f: VioFilter, path: list, goal: tuple, yaw_ref: float, speed_mps: float, log: dict, trail: list,
           world: tuple, wall_guard_m: float | None = None) -> str:
    """Drive along the path without stopping, holding the heading yaw_ref; the filter is carried and corrected all the way."""
    vf, vs = float(car.motion["forward_mps"]), float(car.strafe_mps)
    fwd = np.array(car._pwm("forward"), float)
    side = np.array(car._pwm("strafe_left"), float)
    turn = np.array(car._pwm("left"), float)
    seen = len(tags.rows)
    hist = collections.deque(maxlen=120)
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    limit = time.monotonic() + 3.0 * length / speed_mps + 6.0
    last_t = time.monotonic()
    last_tag = last_t
    zero = car.gyro.peek()
    prev = zero
    vx = vy = 0.0
    while True:
        now = time.monotonic()
        if car.stop_event.is_set():
            raise Abort("stopped")
        if now > limit:
            raise Abort("out of time")
        g = car.gyro.peek()
        if g is None:
            raise Abort("IMU silent")
        dt, last_t = now - last_t, now
        omega = (g - prev) / max(dt, 1e-3)
        f.predict(dt, vx, vy, omega)
        prev = g
        hist.append((now, float(f.x[0]), float(f.x[1]), float(f.x[2])))
        while seen < len(tags.rows):
            r = tags.rows[seen]
            seen += 1
            if r["pose"] is None:
                continue
            if apply_tag(f, hist, r["pose"], r["t"] - LATENCY_S, math.hypot(vx, vy)):
                last_tag = now
        if now - last_tag > NO_TAGS_S:
            raise Abort(f"no tag pose accepted for {NO_TAGS_S:.0f} s")
        x, y, yaw = float(f.x[0]), float(f.x[1]), float(f.x[2])
        i, t, cross, left = along_path(path, (x, y))
        if cross > MAX_CROSS_M:
            raise Abort(f"{cross * 100:.0f} cm off the path")
        if wall_guard_m is not None:     # driving close along a wall: stop before the body touches it
            cy, sy = abs(math.cos(yaw_ref)), abs(math.sin(yaw_ref))
            ex, ey = cy * car.body_half[0] + sy * car.body_half[1], sy * car.body_half[0] + cy * car.body_half[1]
            gap = min(x - ex, world[0][0] - x - ex, y - ey, world[0][1] - y - ey)
            if gap < wall_guard_m:
                raise Abort(f"body only {gap * 100:.1f} cm from a wall")
        to_goal = math.hypot(goal[0] - x, goal[1] - y)
        if to_goal < GOAL_TOL_M:
            return "goal reached"
        tx, ty = point_ahead(path, i, t, min(LOOKAHEAD_M, max(left, 0.02)))
        ex, ey = tx - x, ty - y
        norm = max(math.hypot(ex, ey), 1e-6)
        speed = min(speed_mps, max(MIN_SPEED, speed_mps * min(1.0, to_goal / SLOW_M)))
        wx, wy = speed * ex / norm, speed * ey / norm
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy = c * wx + s * wy, -s * wx + c * wy
        herr = math.degrees(wrap(yaw_ref - yaw))
        hold = max(-HOLD_MAX, min(HOLD_MAX, HOLD_MAX * herr / HOLD_FULL_DEG))
        wheels = (vx / vf) * fwd + (vy / vs) * side + hold * turn
        peak = float(np.abs(wheels).max())
        if peak > PWM_LIMIT:
            wheels *= PWM_LIMIT / peak
        if vx > 0.5 * speed_mps:      # the ultrasonic looks ahead along the nose
            reading = car.sensors.read_sonic()
            if reading is not None:
                got = reading / 100.0
                map_says = expected_range(x + SONIC_AHEAD_M * c, y + SONIC_AHEAD_M * s, yaw, world[0], world[1])
                if got < STOP_ANY_M or (got < STOP_UNKNOWN_M and got < map_says - CLOSER_THAN_MAP_M):
                    raise Abort(f"ultrasonic {reading:.0f} cm ahead, the map says {map_says * 100:.0f} cm")
        car.drive(tuple(int(v) for v in wheels))
        trail.append((x, y, yaw))
        log["rows"].append(dict(t=now, omega=omega, x=x, y=y, yaw=yaw, herr=herr, cross=cross, vx=vx, vy=vy, hold=hold, pwm=[int(v) for v in wheels]))
        time.sleep(0.015)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X_CM", "Y_CM"))
    ap.add_argument("--replay", type=pathlib.Path)
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--no-blocks", action="store_true")
    ap.add_argument("--margin", type=float, default=5.0, help="cm kept between the body and walls or blocks")
    ap.add_argument("--speed", type=float, default=0.16)
    opts = ap.parse_args()
    if opts.replay:
        report_replay(json.loads((opts.replay / "log.json").read_text()))
        return
    if not opts.goal:
        ap.error("--goal X_CM Y_CM is needed")
    goal = (opts.goal[0] / 100.0, opts.goal[1] / 100.0)
    blocks = [] if opts.no_blocks else LAP_BLOCKS
    out = SERVER / "route_follow_debug" / "astar" / time.strftime("%Y%m%d-%H%M%S")
    argv = ["--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"] + (["--arm"] if opts.arm else [])
    car = RouteFollower(parse_args(argv))
    tags = None
    log = dict(rows=[], goal=goal, armed=opts.arm)
    trail, path, grid, arena, reason = [], None, None, None, None
    try:
        car.start()
        tags = TagLogger(car)
        p = standing_pose(tags)
        if p is None:
            raise Abort("no pose from the tags")
        arena = tuple(float(v) for v in car.arena["arena_size_m"])
        yaw_ref = wrap(round(p.yaw / (math.pi / 2)) * (math.pi / 2))
        grid = Grid(arena, blocks, car.body_half[0], car.body_half[1], yaw_ref, opts.margin / 100.0)
        path = plan(grid, (p.x, p.y), goal)
        print(f"pose x={p.x * 100:.1f} y={p.y * 100:.1f} yaw={math.degrees(p.yaw):+.1f}; keep heading {math.degrees(yaw_ref):+.0f}; "
              f"goal {goal[0] * 100:.0f},{goal[1] * 100:.0f}", flush=True)
        if path is None:
            raise Abort("no path: the goal or the car is too close to a wall or a block for this margin")
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        print(f"path {length * 100:.0f} cm in {len(path) - 1} straight pieces: " +
              " -> ".join(f"({a[0] * 100:.0f},{a[1] * 100:.0f})" for a in path), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "plan.png"), picture(grid, arena, blocks, path))
        print(f"plan picture: {out}/plan.png", flush=True)
        log.update(path=path, start=[p.x, p.y, p.yaw], yaw_ref=yaw_ref)
        if not opts.arm:
            print("DRY-RUN: nothing moved. Add --arm to drive.", flush=True)
            return
        if car.gyro is None:
            raise Abort("no IMU")
        f = VioFilter(p.x, p.y, p.yaw)
        reason = follow(car, tags, f, path, goal, yaw_ref, opts.speed, log, trail, (arena, blocks))
        car.stop()
        time.sleep(0.6)
        q = standing_pose(tags)
        log.update(end=None if q is None else [q.x, q.y, q.yaw], fused=f.fused, refused=f.rejected, scale=[float(f.x[3]), float(f.x[4])], reason=reason)
    except Abort as exc:
        reason = str(exc)
        print(f"ABORT: {exc}", flush=True)
    finally:
        car.stop()
        if tags is not None:
            tags.stop.set()
        car.close()
    if not log["rows"]:
        return
    R = log["rows"]
    log["reason"] = reason
    log["tags"] = tags.rows
    (out / "log.json").write_text(json.dumps(log))
    cv2.imwrite(str(out / "result.png"), picture(grid, arena, blocks, path, trail))
    e = np.array([abs(r["herr"]) for r in R]); c = np.array([r["cross"] for r in R]) * 100
    T = R[-1]["t"] - R[0]["t"]
    print(f"\n{reason}: {T:.1f} s without stopping, {len(R) / T:.0f} control cycles/s")
    print(f"off the path: median {np.median(c):.1f} cm, max {c.max():.1f} cm")
    print(f"heading off square: median {np.median(e):.1f} deg, max {e.max():.1f} deg, last second max {max(abs(r['herr']) for r in R if r['t'] > R[-1]['t'] - 1):.1f} deg")
    if "fused" in log:
        print(f"tag poses used {log['fused']}, refused {log['refused']}; speed scale forward {log['scale'][0]:.3f}, sideways {log['scale'][1]:.3f}")
    if log.get("end"):
        q = log["end"]
        print(f"standing at the end, from the tags: x={q[0] * 100:.1f} y={q[1] * 100:.1f} yaw={math.degrees(q[2]):+.1f}; "
              f"{math.hypot(q[0] - goal[0], q[1] - goal[1]) * 100:.1f} cm from the goal, {abs(math.degrees(wrap(q[2] - yaw_ref))):.1f} deg off square")
    print(f"pictures: {out}/result.png", flush=True)
    print(f"replay with other tag weights: python3 tools/astar_drive.py --replay {out}", flush=True)


if __name__ == "__main__":
    main()
