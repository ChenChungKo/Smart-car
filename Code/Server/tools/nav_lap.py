#!/usr/bin/env python3
"""Go once around whatever free space the car finds, the boundary on the right. Dry-run unless --arm.

    python3 tools/nav_lap.py                # scans from where it stands, plans; nothing moves but the gimbal
    python3 tools/nav_lap.py --arm          # drives
    options: --gap 8 (cm, right side of the body to the boundary)  --margin 5  --speed 0.13  --seconds 240

Only the wall tags of the arena are known. Pose: gyro + motor speed + tags (any picture, moving too) + ultrasonic on the
walls (vio.py). World: one occupancy grid from camera, ultrasonic and IMU tilt (nav/perception.py, nav/world.py). Task: the
line at the given distance from what is occupied, counter-clockwise (nav/mission.py). Plan: A* in x, y, heading for the
real rectangle of the body (nav/planner.py), again and again as the map grows. One safety rule: the body must not reach an
occupied cell within the distance it needs to stop; if so the car waits for a new plan.
Output: route_follow_debug/nav/<time>/ console.log (everything printed), map.png, log.json, world.npy.
"""
import argparse
import collections
import json
import math
import pathlib
import sys
import threading
import time

import cv2
import numpy as np

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from astar_drive import apply_tag, standing_pose  # noqa: E402
from nav.mission import Lap, boundary_loop  # noqa: E402
from nav.perception import Eyes  # noqa: E402
from nav.planner import ConfigSpace, heading_index, plan  # noqa: E402
from nav.world import OCCUPIED, RES_M, WorldGrid  # noqa: E402
from pose_kalman import wrap  # noqa: E402
from route_common import SONIC_ARM_M, SONIC_PIVOT_M  # noqa: E402
from route_follow import Abort, RouteFollower, parse_args  # noqa: E402
from vio import VioFilter  # noqa: E402
from vio_check import LATENCY_S, TagLogger  # noqa: E402

LOOKAHEAD_M = 0.08
PLAN_AHEAD_M = 0.45
SPIN_ONLY_DEG = 35.0      # further than this off the path's heading the car only turns
TURN_GAIN, MAX_TURN = 2.0, math.radians(45.0)
MIN_SPEED = 0.09
STOP_S, STOP_EXTRA_M = 0.4, 0.02     # how far ahead the body must be free: speed x this + this
MAX_SIGMA_M = 0.06        # the pose is not good enough to drive on: wait for tags
WAIT_S = 10.0             # no plan or no safe move for this long ends the run
PICTURE = 4               # pixels per grid cell


class Tee:
    """Whatever is printed also goes to a file."""

    def __init__(self, stream, path: pathlib.Path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        self.file.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Thinker(threading.Thread):
    """Plans again and again from the newest pose and map."""

    def __init__(self, world, half, gap, margin, pose):
        super().__init__(daemon=True)
        self.world, self.half, self.pose = world, half, pose
        self.centre_gap = half[1] + gap
        self.margin = min(margin, gap - 0.01)
        self.lap = Lap()
        self.halt = threading.Event()
        self.path, self.loop, self.hard, self.note, self.plan_s = None, None, None, "starting", 0.0

    def step(self) -> None:
        t0 = time.monotonic()
        occ = self.world.occupied()
        space = ConfigSpace(occ, self.half[0], self.half[1], self.margin)
        self.hard = ConfigSpace(occ, self.half[0], self.half[1], 0.005)
        loop = boundary_loop(space, self.centre_gap)
        if loop is None:
            self.note = "no free line at this distance"
            self.path = None
            return
        x, y, yaw = self.pose()
        n = self.lap.advance(loop, x, y)
        goal = self.lap.goal(loop, n, space, PLAN_AHEAD_M)
        path = None if goal is None else plan(space, (x, y, yaw), goal, self.centre_gap)
        self.loop, self.path = loop, path
        self.note = "ok" if path else ("no free pose ahead" if goal is None else "A* found no path")
        self.plan_s = time.monotonic() - t0

    def run(self) -> None:
        while not self.halt.is_set():
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                self.note = f"planner: {exc}"
                self.path = None
            time.sleep(0.15)


def draw(world: WorldGrid, thinker: Thinker, trail: list, start) -> np.ndarray:
    H = world.ny * PICTURE
    img = np.full((H, world.nx * PICTURE, 3), 245, np.uint8)
    log = np.clip(world.log.T[::-1], -2.0, 4.0)
    img[cv2.resize(((log < -0.5) * 1).astype(np.uint8), (img.shape[1], H), interpolation=cv2.INTER_NEAREST) > 0] = (232, 245, 232)
    img[cv2.resize(((log > OCCUPIED) * 1).astype(np.uint8), (img.shape[1], H), interpolation=cv2.INTER_NEAREST) > 0] = (60, 60, 200)
    px = lambda x, y: (int(x / RES_M * PICTURE), H - int(y / RES_M * PICTURE))
    if thinker.loop is not None:
        cv2.polylines(img, [np.array([px(*p) for p in thinker.loop], np.int32)], True, (200, 140, 30), 2, cv2.LINE_AA)
    if thinker.path:
        for a, b in zip(thinker.path, thinker.path[1:]):
            cv2.line(img, px(a[0], a[1]), px(b[0], b[1]), (0, 170, 0), 2, cv2.LINE_AA)
    for a, b in zip(trail, trail[1:]):
        cv2.line(img, px(a[0], a[1]), px(b[0], b[1]), (40, 40, 220), 2, cv2.LINE_AA)
    cv2.circle(img, px(*start[:2]), 6, (0, 130, 0), -1)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, H - 1), (30, 30, 30), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--gap", type=float, default=8.0, help="cm from the right side of the body to the boundary")
    ap.add_argument("--margin", type=float, default=5.0, help="cm the plan keeps the body off anything occupied")
    ap.add_argument("--speed", type=float, default=0.13)
    ap.add_argument("--seconds", type=float, default=240.0)
    opts = ap.parse_args()
    out = SERVER / "route_follow_debug" / "nav" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout, sys.stderr = Tee(sys.stdout, out / "console.log"), Tee(sys.stderr, out / "console.log")
    print("command: " + " ".join(sys.argv), flush=True)
    car = RouteFollower(parse_args(["--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"] + (["--arm"] if opts.arm else [])))
    tags = eyes = thinker = world = None
    hist = collections.deque(maxlen=200)
    trail, rows, reason, start = [], [], None, None
    try:
        car.start()
        tags = TagLogger(car)
        p = standing_pose(tags)
        if p is None:
            raise Abort("no pose from the tags")
        start = (p.x, p.y, p.yaw)
        now_pose = dict(x=p.x, y=p.y, yaw=p.yaw)
        hist.append((time.monotonic(), p.x, p.y, p.yaw))
        def pose_at(t):
            h = list(hist)
            if not h:
                return None
            b = min(h, key=lambda v: abs(v[0] - t))
            return b[1], b[2], b[3]
        W, H = (float(v) for v in car.arena["arena_size_m"])
        world = WorldGrid((W, H))
        eyes = Eyes(car, world, pose_at)
        print(f"pose x={p.x * 100:.1f} y={p.y * 100:.1f} yaw={math.degrees(p.yaw):+.1f}; first look around ...", flush=True)
        for pan in (-90, -60, -30, 0, 30, 60, 90, 60, 30, 0, -30, -60, -90, 0):
            hist.append((time.monotonic(), p.x, p.y, p.yaw))
            eyes.look(float(pan))
        thinker = Thinker(world, car.body_half, opts.gap / 100.0, opts.margin / 100.0, lambda: (now_pose["x"], now_pose["y"], now_pose["yaw"]))
        thinker.step()
        cells = int((world.log > OCCUPIED).sum())
        print(f"map: {cells} occupied cells from {eyes.looks} camera looks; plan took {thinker.plan_s:.2f} s: {thinker.note}", flush=True)
        if thinker.loop is not None:
            print(f"line to follow: {Lap.length(thinker.loop):.2f} m, keeping {opts.gap:.0f} cm from the boundary", flush=True)
        if thinker.path:
            print(f"first plan: {len(thinker.path)} steps to " + "({:.0f},{:.0f}) heading {:+.0f}".format(thinker.path[-1][0] * 100, thinker.path[-1][1] * 100, math.degrees(thinker.path[-1][2])), flush=True)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "map.png"), draw(world, thinker, [], start))
        print(f"map picture: {out}/map.png", flush=True)
        if not opts.arm:
            print("DRY-RUN: only the gimbal moved. Add --arm to drive.", flush=True)
            return
        if car.gyro is None:
            raise Abort("no IMU")
        vf, vs = float(car.motion["forward_mps"]), float(car.strafe_mps)
        fwd, side, turn = (np.array(car._pwm(k), float) for k in ("forward", "strafe_left", "left"))
        w_nom = math.radians(float(car.motion["turn_left_dps"]))
        f = VioFilter(p.x, p.y, p.yaw)
        seen, ranges_seen = len(tags.rows), 0
        last = time.monotonic()
        prev = car.gyro.peek()
        vx = vy = 0.0
        end = last + opts.seconds
        waiting_since = None
        eyes.start()
        thinker.start()
        t_start = last
        while True:
            t = time.monotonic()
            if car.stop_event.is_set():
                raise Abort("stopped")
            if t > end:
                raise Abort("out of time")
            g = car.gyro.peek()
            if g is None:
                raise Abort("IMU silent")
            dt, last = t - last, t
            omega_meas = (g - prev) / max(dt, 1e-3)
            prev = g
            f.predict(dt, vx, vy, omega_meas)
            hist.append((t, float(f.x[0]), float(f.x[1]), float(f.x[2])))
            while seen < len(tags.rows):
                r = tags.rows[seen]
                seen += 1
                if r["pose"]:
                    apply_tag(f, hist, r["pose"], r["t"] - LATENCY_S, math.hypot(vx, vy))
            while ranges_seen < len(eyes.ranges):
                _t, pan, rng = eyes.ranges[ranges_seen]
                ranges_seen += 1
                f.update_wall(SONIC_PIVOT_M, math.radians(pan), rng + SONIC_ARM_M, (W, H))
            x, y, yaw = float(f.x[0]), float(f.x[1]), float(f.x[2])
            now_pose.update(x=x, y=y, yaw=yaw)
            sigma = math.sqrt(max(f.P[0, 0], f.P[1, 1]))
            path, hard, loop = thinker.path, thinker.hard, thinker.loop
            if loop is not None and thinker.lap.progress_m >= Lap.length(loop) - 0.10:
                reason = f"lap finished: {thinker.lap.progress_m:.2f} m in {t - t_start:.1f} s"
                break
            cmd = (0.0, 0.0, 0.0)
            why = None
            if sigma > MAX_SIGMA_M:
                why = f"pose uncertain ({sigma * 100:.0f} cm)"
            elif not path or len(path) < 2:
                why = thinker.note
            else:
                dist = np.hypot(np.array([q[0] for q in path]) - x, np.array([q[1] for q in path]) - y)
                i = int(np.argmin(dist))
                k, along = i, 0.0
                while k < len(path) - 1 and along < LOOKAHEAD_M:
                    along += math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
                    k += 1
                tx, ty, tyaw = path[k]
                err = wrap(tyaw - yaw)
                om = max(-MAX_TURN, min(MAX_TURN, TURN_GAIN * err))
                ex, ey = tx - x, ty - y
                norm = math.hypot(ex, ey)
                scale = max(0.0, 1.0 - abs(math.degrees(err)) / SPIN_ONLY_DEG)
                speed = max(MIN_SPEED, opts.speed * scale) if scale > 0.0 and norm > 0.015 else 0.0
                wx, wy = (speed * ex / norm, speed * ey / norm) if speed > 0.0 else (0.0, 0.0)
                ahead = speed * STOP_S + STOP_EXTRA_M
                safe = True
                if hard is not None and speed > 0.0:
                    k0 = heading_index(yaw)
                    def cl(a, b):
                        ii, jj = int(round(a / RES_M)), int(round(b / RES_M))
                        return float(hard.clear[ii, jj]) if 0 <= ii < hard.nx and 0 <= jj < hard.ny else 0.0
                    for s in (0.34, 0.67, 1.0):
                        fx, fy = x + wx / speed * ahead * s, y + wy / speed * ahead * s
                        if not hard.free(int(round(fx / RES_M)), int(round(fy / RES_M)), k0) and cl(fx, fy) <= cl(x, y):
                            safe = False
                if safe:
                    c, s = math.cos(yaw), math.sin(yaw)
                    cmd = (c * wx + s * wy, -s * wx + c * wy, om)
                else:
                    why = "the body would reach something occupied"
            if why is None:
                waiting_since = None
            else:
                waiting_since = waiting_since or t
                if t - waiting_since > WAIT_S:
                    raise Abort(f"waited {WAIT_S:.0f} s: {why}")
            vx, vy, om_cmd = cmd
            eyes.heading_deg = math.degrees(math.atan2(vy, vx)) if math.hypot(vx, vy) > 0.03 else 0.0
            wheels = (vx / vf) * fwd + (vy / vs) * side + (om_cmd / w_nom) * turn
            peak = float(np.abs(wheels).max())
            if peak > 2000:
                wheels *= 2000.0 / peak
            if peak < 1.0:
                car.stop()
            else:
                car.drive(tuple(int(v) for v in wheels))
            trail.append((x, y, yaw))
            rows.append(dict(t=t, x=x, y=y, yaw=yaw, vx=vx, vy=vy, om=om_cmd, omega=omega_meas, sigma=sigma, why=why, progress=thinker.lap.progress_m))
            time.sleep(0.012)
        car.stop()
        print(reason, flush=True)
    except Abort as exc:
        reason = str(exc)
        print(f"ABORT: {exc}", flush=True)
    finally:
        car.stop()
        if thinker is not None:
            thinker.halt.set()
        if eyes is not None:
            eyes.close()
        if tags is not None:
            tags.stop.set()
        car.close()
    if rows:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "world.npy", world.log)
        (out / "log.json").write_text(json.dumps(dict(rows=rows, start=start, reason=reason, tags=tags.rows, gap=opts.gap)))
        cv2.imwrite(str(out / "map.png"), draw(world, thinker, trail, start))
        waits = collections.Counter(r["why"] for r in rows if r["why"])
        T = rows[-1]["t"] - rows[0]["t"]
        print(f"{reason}\n{T:.1f} s, {len(rows) / T:.0f} control cycles/s, camera looks {eyes.looks} (smeared {eyes.smeared}), "
              f"waiting {sum(waits.values()) / max(len(rows), 1) * 100:.0f}% of the time: {dict(waits.most_common(3))}\npictures: {out}/map.png", flush=True)


if __name__ == "__main__":
    main()
