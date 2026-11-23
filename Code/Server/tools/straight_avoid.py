#!/usr/bin/env python3
"""Go around the arena wall by wall; when about to hit something stop, go left around it and come back to the line. Dry-run unless --arm.

    python3 tools/straight_avoid.py            # shows the first line from where the car stands; only the gimbal moves
    python3 tools/straight_avoid.py --arm      # drives the four walls, counter-clockwise, wall on the right
    options: --sides 4  --gap 6 (cm, right side of the body to the wall)  --corner 22  --speed 0.13  --stop-at 8  --margin 5

Each line runs along the wall on the right at --gap, up to the wall ahead. At its end the car moves in to --corner cm from
that wall (half its diagonal is 16 cm: it cannot turn closer) and turns left; the next line starts there. After four walls
it drives back to where it started, measured from the tags while standing. Only the walls are known. Nothing is known about what stands on
a line: the ultrasonic looks ahead while driving and the gimbal camera looks for feet on the floor in the width of the body.
  drive    along the line, until something is --stop-at from the nose or the wall ahead is near
  check    the one stop per obstacle: stands about 2 s and looks again; nothing there: goes on
  left     strafes left until the ultrasonic looks past it, then by half the body and the margin
  past     drives on, the ultrasonic turned to the right, until it has seen the far end and the tail is past it
  back     strafes right to the line, watching the side
Pose: gyro, motor speed, every tag pose (vio.py). The heading is held all the time.
Output: route_follow_debug/straight/<time>/ console.log, log.json; while it runs also control.jsonl (the pose every 0.05 s)
and plans.jsonl (each line), the same format as the patrol, so the IMU viewer's map (tools/map_panel.py) follows the car.
"""
import argparse
import json
import math
import pathlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from astar_drive import apply_tag, standing_pose  # noqa: E402
from foot_cam import FootCamera, GimbalFeed  # noqa: E402
from nav_lap import Tee  # noqa: E402
from pose_kalman import ray_to_wall, wrap  # noqa: E402
from route_common import HOLD_FULL_DEG, HOLD_MAX, SONIC_ARM_M, SONIC_PIVOT_M  # noqa: E402
from route_follow import Abort, RouteFollower, parse_args  # noqa: E402
from servo import servo_pan_for_bearing  # noqa: E402
from vio import VioFilter  # noqa: E402
from vio_check import LATENCY_S, TagLogger, stand_pose  # noqa: E402

WALL_END_M = 0.10         # the line ends with the nose this far from the wall ahead
NOT_WALL_M = 0.08         # an echo this much shorter than the wall the pose expects is something else
LOOK_AGAIN_M = 0.25       # at the check: something within this of the nose is an obstacle
CLEAR_AHEAD_M = 0.40      # strafing left: the beam looks past it when the echo is longer than this
MAX_LEFT_M = 0.45
MAX_PAST_M = 0.45
BESIDE_M = 0.30           # the side beam: nearer than this and shorter than the wall = beside the block
SIDE_STOP_M = 0.10        # strafing back: the side echo this close means it is still beside the block
CAM_WIDTH_M = 0.02        # looks for feet this much wider than the body each side
LINE_GAIN = 2.0
HOME_TRIES = 1           # the tags bring it near the start; the ultrasonic then decides
SERVO_TRIES = 6
SERVO_OK_M = 0.010       # both echoes this close to what they were at the start
SERVO_MAX_STEP_M = 0.06
SERVO_MAX_ERROR_M = 0.15 # an echo further off than this is not the same thing as at the start: that direction is left alone
SERVO_SPEED = 0.09
STEADY_POSES, STEADY_MIN_S, STEADY_MAX_S = 15, 1.5, 6.0   # the pose standing: at least this many tag poses, within this long
START_SIGMA_XY = 0.04    # how well the tags alone place the car where it sees only far tags: the side echo then decides
WALL_KEEP_M = 0.03       # a place to go back to is never closer than this to a wall
HOME_REACHED_M = 0.012
HOME_MIN_SPEED = 0.08    # below this the wheels may not turn
HOME_OK_M, HOME_OK_DEG = 0.02, 2.0
SIDE_LIMIT = 0.07


class FrontCamera(threading.Thread):
    """The nearest foot in the width of the body, from the gimbal camera looking straight ahead."""

    def __init__(self, car, feed):
        super().__init__(daemon=True)
        self.car, self.feed = car, feed
        self.cam = FootCamera(car.car)
        self.cam.geometry["pitch_deg"] = float(car.car.get("nav", {}).get("gimbal_pitch_deg", self.cam.geometry["pitch_deg"]))
        self.enabled = False
        self.gap, self.at = None, 0.0
        self.halt = threading.Event()

    def run(self) -> None:
        seen = 0.0
        while not self.halt.is_set():
            frame, stamp = self.feed.latest()
            if frame is None or stamp == seen or not self.enabled:
                time.sleep(0.01)
                continue
            seen = stamp
            s = self.car.gyro.imu.latest()
            if s is None or abs(s.gyro_dps[2]) > 25.0:
                continue
            v = self.cam.observe(frame, 0.0, tilt=(s.rpy_deg[1], s.rpy_deg[0]))
            pts = np.vstack([v.points[~np.isnan(v.points[:, 0])], v.close])
            half_l, half_w = self.car.body_half
            m = (np.abs(pts[:, 1]) <= half_w + CAM_WIDTH_M) & (pts[:, 0] > half_l - 0.03)
            self.gap = float(np.percentile(pts[m, 0], 20)) - half_l if int(m.sum()) >= 5 else None
            self.at = stamp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--speed", type=float, default=0.13)
    ap.add_argument("--stop-at", type=float, default=8.0, help="cm from the nose to what is ahead")
    ap.add_argument("--margin", type=float, default=5.0, help="cm between the body and the block when passing")
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--sides", type=int, default=4, help="how many walls to drive along")
    ap.add_argument("--gap", type=float, default=6.0, help="cm from the right side of the body to the wall on a line")
    ap.add_argument("--corner", type=float, default=22.0, help="cm from the wall to the car centre where it turns")
    opts = ap.parse_args()
    out = SERVER / "route_follow_debug" / "straight" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout, sys.stderr = Tee(sys.stdout, out / "console.log"), Tee(sys.stderr, out / "console.log")
    print("command: " + " ".join(sys.argv), flush=True)
    car = RouteFollower(parse_args(["--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"] + (["--arm"] if opts.arm else [])))
    tags = feed = camera = None
    rows, events = [], []
    margin, stop_at = opts.margin / 100.0, opts.stop_at / 100.0

    # live output for the map in the IMU viewer (tools/map_panel.py): same rows as patrol/navigator.py writes
    live = dict(t0=time.monotonic(), last=0.0, note="", goal="",
                ctrl=open(out / "control.jsonl", "w", encoding="utf-8", buffering=1),
                plans=open(out / "plans.jsonl", "w", encoding="utf-8", buffering=1))

    def say(text: str) -> None:
        events.append(text)
        live["note"] = text
        print(text, flush=True)

    def live_line(ln, name: str) -> None:
        """The line the car is about to follow, as the 'plan' the map draws."""
        live["goal"] = name
        try:
            live["plans"].write(json.dumps(dict(s=time.monotonic() - live["t0"], goal=name, path=[
                [ln.x, ln.y, 0.0], [ln.x + ln.ux * ln.end, ln.y + ln.uy * ln.end, 0.0]])) + "\n")
        except (OSError, ValueError):
            pass

    def live_pose(t: float, x: float, y: float, yaw: float, sigma_m: float | None, pwm) -> None:
        if t - live["last"] < 0.05:
            return
        live["last"] = t
        try:
            live["ctrl"].write(json.dumps(dict(s=round(t - live["t0"], 3), pose=[round(x, 4), round(y, 4), round(yaw, 4)],
                                               sigma_cm=None if sigma_m is None else round(sigma_m * 100, 2),
                                               goal=live["goal"], why=None, note=live["note"][:80],
                                               pwm=pwm, armed=bool(opts.arm))) + "\n")
        except (OSError, ValueError):
            pass

    try:
        car.start()
        tags = TagLogger(car)
        W, H = (float(v) for v in car.arena["arena_size_m"])
        half_l, half_w = car.body_half
        lane = half_w + opts.gap / 100.0                # the line: this far from the wall on the right

        def new_line(x0: float, y0: float, yaw: float) -> SimpleNamespace:
            """The line along the wall on the right for this heading, starting abreast of (x0, y0)."""
            ux, uy = math.cos(yaw), math.sin(yaw)
            right = ray_to_wall(x0, y0, yaw - math.pi / 2.0, W, H) - lane          # how far the line is to the right of the car
            x, y = x0 + uy * right, y0 - ux * right
            return SimpleNamespace(x=x, y=y, ux=ux, uy=uy, yaw=yaw, end=ray_to_wall(x, y, yaw, W, H) - half_l - WALL_END_M)

        feed = GimbalFeed().start()
        feed.wait_ready()
        camera = FrontCamera(car, feed)
        camera.start()

        def point(pan: float) -> None:
            car.servo.set_pan(servo_pan_for_bearing(car.home_pan, car.right_90_pan, pan, car.left_90_pan))

        def where_standing():
            """The pose while standing: every tag pose of a few seconds, then the wall on the right measured by the ultrasonic."""
            t0 = time.monotonic()
            while time.monotonic() - t0 < STEADY_MAX_S:
                time.sleep(0.1)
                n = sum(1 for r in list(tags.rows) if r["t"] >= t0 and r["pose"])
                if n >= STEADY_POSES and time.monotonic() - t0 >= STEADY_MIN_S:
                    break
            got = stand_pose(dict(tags=list(tags.rows)), dict(t0=t0, t1=time.monotonic()))
            if got is None:
                return None, 0
            count = sum(1 for r in list(tags.rows) if r["t"] >= t0 and r["pose"])
            est = VioFilter(*got)
            est.P[0, 0] = est.P[1, 1] = START_SIGMA_XY ** 2
            point(-90.0)
            time.sleep(0.4)
            reads = [v for v in (car.sensors.read_sonic() for _ in range(5)) if v is not None]
            point(0.0)
            time.sleep(0.25)
            used = bool(reads) and est.update_wall(SONIC_PIVOT_M, -math.pi / 2.0, float(np.median(reads)) / 100.0 + SONIC_ARM_M, (W, H), gate_m=0.08)
            return SimpleNamespace(x=float(est.x[0]), y=float(est.x[1]), yaw=float(est.x[2]), tags_x=got[0], tags_y=got[1], echo=used), count

        point(0.0)
        time.sleep(0.3)
        p, count = where_standing()
        if p is None:
            raise Abort("no pose from the tags")
        line = new_line(p.x, p.y, wrap(round(p.yaw / (math.pi / 2)) * (math.pi / 2)))
        live_line(line, "side 1")
        say(f"pose x={p.x * 100:.1f} y={p.y * 100:.1f} yaw={math.degrees(p.yaw):+.1f} ({count} tag poses: x={p.tags_x * 100:.1f} y={p.tags_y * 100:.1f}; side echo {'used' if p.echo else 'not used'}); "
            f"line along {math.degrees(line.yaw):+.0f} deg, {line.end * 100:.0f} cm to the end")
        def echoes():
            """The ultrasonic straight ahead and to the right, median of several readings (m; None = no echo)."""
            out = []
            for pan, wait in ((0.0, 0.4), (-90.0, 0.5)):
                point(pan)
                time.sleep(wait)
                reads = [v for v in (car.sensors.read_sonic() for _ in range(7)) if v is not None]
                out.append(float(np.median(reads)) / 100.0 if reads else None)
            point(0.0)
            time.sleep(0.25)
            return tuple(out)

        home_echoes = echoes()
        say("at the start the ultrasonic reads: ahead " + ("none" if home_echoes[0] is None else f"{home_echoes[0] * 100:.1f} cm") +
            ", right " + ("none" if home_echoes[1] is None else f"{home_echoes[1] * 100:.1f} cm"))
        if not opts.arm:
            camera.enabled = True
            time.sleep(1.5)
            camera.enabled = False
            echo = car.sensors.read_sonic()
            say(f"DRY-RUN: ultrasonic ahead {echo} cm; camera: nearest foot {('none' if camera.gap is None else f'{camera.gap * 100:.0f} cm')} from the nose. Add --arm to drive.")
            return
        if car.gyro is None:
            raise Abort("no IMU")
        vf, vs = float(car.motion["forward_mps"]), float(car.strafe_mps)
        fwd, side, turn = (np.array(car._pwm(k), float) for k in ("forward", "strafe_left", "left"))
        f = VioFilter(p.x, p.y, p.yaw)
        hist, st = [], dict(seen=len(tags.rows), last=time.monotonic(), prev=car.gyro.peek(), vx=0.0, vy=0.0)
        end = st["last"] + opts.seconds

        def tick(vx: float, vy: float) -> tuple[float, float, float]:
            """One control cycle: pose update, heading held, wheels at the body speed (vx, vy). Returns x, y, yaw."""
            t = time.monotonic()
            if car.stop_event.is_set():
                raise Abort("stopped")
            if t > end:
                raise Abort("out of time")
            g = car.gyro.peek()
            if g is None:
                raise Abort("IMU silent")
            dt, st["last"] = t - st["last"], t
            f.predict(dt, st["vx"], st["vy"], (g - st["prev"]) / max(dt, 1e-3))
            st["prev"] = g
            hist.append((t, float(f.x[0]), float(f.x[1]), float(f.x[2])))
            del hist[:-150]
            while st["seen"] < len(tags.rows):
                r = tags.rows[st["seen"]]
                st["seen"] += 1
                if r["pose"]:
                    apply_tag(f, hist, r["pose"], r["t"] - LATENCY_S, math.hypot(st["vx"], st["vy"]))
            x, y, yaw = float(f.x[0]), float(f.x[1]), float(f.x[2])
            herr = math.degrees(wrap(line.yaw - yaw))
            hold = max(-HOLD_MAX, min(HOLD_MAX, HOLD_MAX * herr / HOLD_FULL_DEG))
            wheels = (vx / vf) * fwd + (vy / vs) * side + hold * turn
            if abs(vx) + abs(vy) < 1e-4 and abs(herr) < 1.0:
                car.stop()
            else:
                car.drive(tuple(int(v) for v in wheels))
            st["vx"], st["vy"] = vx, vy
            rows.append(dict(t=t, x=x, y=y, yaw=yaw, herr=herr, vx=vx, vy=vy, pwm=[int(v) for v in wheels], sy=float(f.x[4])))
            sent = [0, 0, 0, 0] if abs(vx) + abs(vy) < 1e-4 and abs(herr) < 1.0 else [int(v) for v in wheels]
            live_pose(t, x, y, yaw, math.sqrt(max(0.0, (f.P[0, 0] + f.P[1, 1]) / 2.0)), sent)
            time.sleep(0.012)
            return x, y, yaw

        along = lambda x, y: (x - line.x) * line.ux + (y - line.y) * line.uy
        off = lambda x, y: -(x - line.x) * line.uy + (y - line.y) * line.ux   # + = left of the line

        def hold_still(seconds: float) -> tuple[float, float, float]:
            q, stop = None, time.monotonic() + seconds
            while time.monotonic() < stop:
                q = tick(0.0, 0.0)
            return q

        def echo() -> float | None:
            r = car.sensors.read_sonic()
            return None if r is None else r / 100.0

        def wall_range(x: float, y: float, yaw: float, pan: float) -> float:
            a = math.radians(pan)
            return ray_to_wall(x + SONIC_PIVOT_M * math.cos(yaw) + SONIC_ARM_M * math.cos(yaw + a),
                               y + SONIC_PIVOT_M * math.sin(yaw) + SONIC_ARM_M * math.sin(yaw + a), yaw + a, W, H)

        def is_thing(reading: float | None, x: float, y: float, yaw: float, pan: float) -> bool:
            return reading is not None and reading < wall_range(x, y, yaw, pan) - NOT_WALL_M

        def nose_gap(x: float, y: float, yaw: float):
            """What is ahead, m from the nose (ultrasonic or camera), and which saw it; None when nothing."""
            reading = echo()
            best, who = None, None
            if is_thing(reading, x, y, yaw, 0.0):
                best, who = reading - (half_l - SONIC_PIVOT_M - SONIC_ARM_M), "ultrasonic"
            if camera.gap is not None and time.monotonic() - camera.at < 0.5 and (best is None or camera.gap < best):
                best, who = camera.gap, "camera"
            return best, who

        def to_line(target_off: float, x: float, y: float) -> float:
            return max(-SIDE_LIMIT, min(SIDE_LIMIT, LINE_GAIN * (target_off - off(x, y))))

        x, y, yaw = tick(0.0, 0.0)
        passes = 0
        camera.enabled = True
        for wall_no in range(opts.sides):
            while True:
                # drive along the line
                say(f"drive from {along(x, y) * 100:.0f} cm along the line")
                point(0.0)
                settle = time.monotonic() + 0.35         # the gimbal turns ahead while the car already moves
                while time.monotonic() < settle and along(x, y) < line.end:
                    x, y, yaw = tick(opts.speed, to_line(0.0, x, y))
                hit = None
                while along(x, y) < line.end:
                    gap, who = nose_gap(x, y, yaw)
                    if gap is not None and gap <= stop_at:
                        hit = (gap, who)
                        break
                    x, y, yaw = tick(opts.speed, to_line(0.0, x, y))
                if hit is None:
                    say(f"end of the line: {along(x, y) * 100:.0f} cm along, {off(x, y) * 100:+.1f} cm off it")
                    break
                x, y, yaw = hold_still(0.4)
                say(f"stopped {hit[0] * 100:.0f} cm from something ({hit[1]}) at {along(x, y) * 100:.0f} cm along the line")
                # look again, standing: an echo or feet near the nose?
                seen = []
                for pan in (0.0, -15.0, 15.0):
                    point(pan)
                    x, y, yaw = hold_still(0.3)
                    r = [v for v in (echo() for _ in range(3)) if v is not None]
                    if r and is_thing(float(np.median(r)), x, y, yaw, pan):
                        seen.append(float(np.median(r)) - (half_l - SONIC_PIVOT_M - SONIC_ARM_M))
                point(0.0)
                x, y, yaw = hold_still(0.4)
                if camera.gap is not None:
                    seen.append(camera.gap)
                if not seen or min(seen) > LOOK_AGAIN_M:
                    say("nothing there when looking again: going on")
                    continue
                passes += 1
                say(f"obstacle {passes}: {min(seen) * 100:.0f} cm ahead; going left around it")
                # left until the beam looks past it, then half the body and the margin more
                clear_at = None
                while True:
                    reading = echo()
                    if off(x, y) > MAX_LEFT_M:
                        raise Abort("went left as far as allowed and the way ahead is not clear")
                    if wall_range(x, y, yaw, 90.0) - half_w + SONIC_ARM_M < 0.05:
                        raise Abort("no room to go further left")
                    if clear_at is None and reading is not None and (reading > CLEAR_AHEAD_M or not is_thing(reading, x, y, yaw, 0.0)):
                        clear_at = off(x, y)                 # no echo at all says nothing: only a measured distance counts
                        say(f"beam looks past it at {clear_at * 100:.0f} cm left of the line (echo {reading * 100:.0f} cm)")
                        point(-90.0)                     # it settles during the rest of the way left
                    if clear_at is not None and off(x, y) >= clear_at + half_w + margin:
                        break
                    x, y, yaw = tick(0.0, opts.speed)
                target = off(x, y)
                # past it: the ultrasonic looks right
                camera.enabled = False
                start, beside, far = along(x, y), False, None
                say(f"passing at {target * 100:.0f} cm left of the line")
                while far is None and along(x, y) - start < MAX_PAST_M and along(x, y) < line.end:
                    reading = echo()
                    near = is_thing(reading, x, y, yaw, -90.0) and reading < BESIDE_M
                    beside = beside or near
                    if beside and not near:
                        far = along(x, y) + SONIC_PIVOT_M - 0.04
                        say(f"its far end at {far * 100:.0f} cm along the line")
                    x, y, yaw = tick(opts.speed, to_line(target, x, y))
                if far is None:
                    say("did not see its far end; going the longest it can be")
                need = (far if far is not None else along(x, y)) + half_l + margin
                while along(x, y) < min(need, line.end):
                    x, y, yaw = tick(opts.speed, to_line(target, x, y))
                # back to the line, watching the side
                for attempt in range(4):
                    while off(x, y) > 0.015:
                        reading = echo()
                        if is_thing(reading, x, y, yaw, -90.0) and reading < SIDE_STOP_M:
                            break
                        x, y, yaw = tick(0.0, -opts.speed)
                    if off(x, y) <= 0.015:
                        break
                    x, y, yaw = hold_still(0.3)
                    say("still beside it: 6 cm further on before going back")
                    goal = along(x, y) + 0.06
                    while along(x, y) < goal:
                        x, y, yaw = tick(opts.speed, 0.0)
                    x, y, yaw = hold_still(0.3)
                else:
                    raise Abort("could not get back to the line")
                say(f"back on the line: {off(x, y) * 100:+.1f} cm off it, at {along(x, y) * 100:.0f} cm along")
                camera.enabled = True
            if wall_no + 1 >= opts.sides and opts.sides != 4:   # after four walls the car goes home from the last corner
                break
            # the corner: move in from the wall on the right, turn left, and start the next line from where the tags say it is
            car.stop()
            need = max(0.0, opts.corner / 100.0 - lane)
            say(f"side {wall_no + 1} done; moving {need * 100:.0f} cm in to turn")
            while off(x, y) < need:
                x, y, yaw = tick(0.0, opts.speed)
            car.stop()
            hold_still(0.3)
            car.move("left", 90.0 / car.turn_rate("left"), math.pi / 2.0)
            camera.enabled = False
            time.sleep(0.2)
            q = standing_pose(tags, 1.0)
            if q is None:
                raise Abort("no pose from the tags after the turn")
            line = new_line(q.x, q.y, wrap(line.yaw + math.pi / 2.0))
            live_line(line, f"side {wall_no + 2}" if wall_no + 2 <= opts.sides else "back to the start")
            f.x[:3] = [q.x, q.y, q.yaw]
            f.P[:3, :3] = np.diag([f.START_SIGMA_M ** 2, f.START_SIGMA_M ** 2, math.radians(f.START_SIGMA_DEG) ** 2])
            st.update(seen=len(tags.rows), last=time.monotonic(), prev=car.gyro.peek(), vx=0.0, vy=0.0)
            say(f"turned: x={q.x * 100:.1f} y={q.y * 100:.1f} yaw={math.degrees(q.yaw):+.1f}; next line along {math.degrees(line.yaw):+.0f} deg, {line.end * 100:.0f} cm")
            x, y, yaw = tick(0.0, 0.0)
            camera.enabled = True
        if opts.sides == 4:
            # home: where it started, heading as at the start; measured standing, and again if not close enough
            camera.enabled = False
            cy_, sy_ = abs(math.cos(line.yaw)), abs(math.sin(line.yaw))
            keep_x, keep_y = cy_ * half_l + sy_ * half_w + WALL_KEEP_M, sy_ * half_l + cy_ * half_w + WALL_KEEP_M
            p.x, p.y = min(max(p.x, keep_x), W - keep_x), min(max(p.y, keep_y), H - keep_y)
            for attempt in range(HOME_TRIES):
                say(f"to the start ({p.x * 100:.0f},{p.y * 100:.0f}), try {attempt + 1}")
                while True:
                    ex, ey = p.x - x, p.y - y
                    d = math.hypot(ex, ey)
                    if d < HOME_REACHED_M:
                        break
                    v = min(opts.speed, max(HOME_MIN_SPEED, 1.5 * d)) / d
                    c, s_ = math.cos(yaw), math.sin(yaw)
                    x, y, yaw = tick(v * (c * ex + s_ * ey), v * (-s_ * ex + c * ey))
                x, y, yaw = hold_still(0.6)
                car.stop()
                q, _count = where_standing()
                if q is None:
                    raise Abort("no pose from the tags at the start point")
                d, h = math.hypot(q.x - p.x, q.y - p.y), abs(math.degrees(wrap(q.yaw - line.yaw)))
                say(f"standing: x={q.x * 100:.1f} y={q.y * 100:.1f} yaw={math.degrees(q.yaw):+.1f}; {d * 100:.1f} cm from the start, {h:.1f} deg off square")
                f.x[:3] = [q.x, q.y, q.yaw]
                st.update(seen=len(tags.rows), last=time.monotonic(), prev=car.gyro.peek(), vx=0.0, vy=0.0)
                x, y, yaw = q.x, q.y, q.yaw
                if d <= HOME_OK_M and h <= HOME_OK_DEG:
                    break
            # the last centimetres by the same measurement as at the start: move until both echoes read as they did then
            for attempt in range(1, SERVO_TRIES + 1):
                car.stop()
                x, y, yaw = hold_still(0.3)
                now_echoes = echoes()
                st.update(seen=len(tags.rows), last=time.monotonic(), prev=car.gyro.peek(), vx=0.0, vy=0.0)
                errors = [None if a is None or b is None or abs(a - b) > SERVO_MAX_ERROR_M else a - b for a, b in zip(now_echoes, home_echoes)]
                say("ultrasonic: " + ", ".join(f"{name} {'-' if e is None else f'{e * 100:+.1f} cm'}" for name, e in zip(("ahead", "right"), errors)))
                move_f = 0.0 if errors[0] is None else max(-SERVO_MAX_STEP_M, min(SERVO_MAX_STEP_M, errors[0]))     # further than at the start: forward
                move_l = 0.0 if errors[1] is None else max(-SERVO_MAX_STEP_M, min(SERVO_MAX_STEP_M, -errors[1]))    # the wall further than at the start: to the right
                if all(e is None or abs(e) <= SERVO_OK_M for e in errors):
                    break
                length = math.hypot(move_f, move_l)
                until = time.monotonic() + length / SERVO_SPEED
                while time.monotonic() < until:
                    x, y, yaw = tick(SERVO_SPEED * move_f / length, SERVO_SPEED * move_l / length)
                car.stop()
                x, y, yaw = hold_still(0.3)
            say(f"parked by the ultrasonic after {attempt} measurements")
        car.stop()
        say("finished")
        time.sleep(0.5)
        q = standing_pose(tags)
        if q is not None:
            say(f"standing at the end: x={q.x * 100:.1f} y={q.y * 100:.1f} yaw={math.degrees(q.yaw):+.1f}; {off(q.x, q.y) * 100:+.1f} cm off the line, {abs(math.degrees(wrap(q.yaw - line.yaw))):.1f} deg off square")
    except Abort as exc:
        say(f"ABORT: {exc}")
    except KeyboardInterrupt:
        say("interrupted with Ctrl+C")
    finally:
        car.stop()
        if camera is not None:
            camera.halt.set()
        if tags is not None:
            tags.stop.set()
        if feed is not None:
            feed.close()
        car.close()
        for k in ("ctrl", "plans"):
            live[k].close()
    if rows:
        (out / "log.json").write_text(json.dumps(dict(rows=rows, events=events, tags=tags.rows)))
        e = np.array([abs(r["herr"]) for r in rows])
        print(f"{rows[-1]['t'] - rows[0]['t']:.1f} s; heading off square median {np.median(e):.1f}, max {e.max():.1f} deg; saved in {out}", flush=True)


if __name__ == "__main__":
    main()
