#!/usr/bin/env python3
"""Record what the gimbal camera sees of the wall tags at every pan angle, at poses you measured with a ruler.

    python3 tools/gimbal_survey.py [label]      # asks for each pose; q to finish
    python3 tools/gimbal_survey.py [label] --auto          # no ruler: the pose comes from the front and rear cameras' tags
    python3 tools/gimbal_survey.py [label] --auto --arm    # ... and the car turns 90 deg three times by itself (4 headings)

    python3 tools/gimbal_survey.py [label] --tour --arm    # 10 places around the EMPTY arena, driven by itself

    python3 tools/gimbal_survey.py [label] --walls --arm   # 8 places close to the walls, the pose from the ultrasonic

--walls: the arena must be empty. From the middle, for each of four headings, the car drives until the ultrasonic reads
22 cm to the wall ahead, then sideways until it reads 22 cm to the wall on one side, later the other side. There its
pose comes from those two echoes (good to about 1 cm) and the heading from the gyro since the middle (tags there): no
camera is the reference. Then the gimbal sweeps.
--tour: the arena must be empty. The car drives to each place holding its heading, turns at some of them, stands, takes
its pose from the front and rear cameras' tags as it really stands (it need not be square), and sweeps the gimbal.
--auto: the front and rear cameras are calibrated, so their joint tag pose (good to 1-2 cm) stands in for the ruler. Put
the car where both see several tags (the middle of the arena) with 20 cm free all round: with --arm it turns on the spot.

Put the car by hand, squared to a wall (heading 0, 90, 180 or -90), and measure from the BODY EDGES to the walls:
    e HEADING AHEAD BEHIND LEFT RIGHT       (cm; - = not measured; one of ahead/behind and one of left/right at least)
Nothing drives; the gimbal sweeps from the right to the left in 15 deg steps (about 20 s per pose). Tall things that hide
the tags should be out of the arena. 4 or more poses, with different headings, some near a wall and some in the middle.
Saves route_follow_debug/survey/<label>/NN_x_y_h/ obs.json and the pictures. Then: python3 tools/solve_gimbal.py <label>
"""
import json
import sys
import time
from pathlib import Path

import cv2

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, load_json  # noqa: E402
from foot_cam import GimbalFeed  # noqa: E402
from servo import Servo, load_gimbal_home, load_left_90_pan, load_right_90_pan, servo_pan_for_bearing  # noqa: E402
from static_survey import pose_from_edges, tags_json  # noqa: E402

PANS = list(range(-90, 91, 15))


def sweep(loc, feed, point, folder: Path, obs: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    last = None
    for pan in PANS:
        point(float(pan))
        time.sleep(0.9 if last is None else 0.5)
        last = pan
        frame = feed.fresh(time.monotonic(), 1.5)
        found, _fx = loc.detect("gimbal", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(str(folder / f"pan{pan:+04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        obs["pans"][str(pan)] = tags_json(found)
        print(f"  pan {pan:+4d}: tags {sorted(found) or '-'}", flush=True)
    point(0.0)
    (folder / "obs.json").write_text(json.dumps(obs) + "\n")
    print(f"  saved {folder.name}: {sum(len(v) for v in obs['pans'].values())} tag views, tags {sorted({int(t) for v in obs['pans'].values() for t in v})}", flush=True)


TOUR = [((70, 70), None), ((32, 32), 0), ((105, 32), 25), ((107, 70), -25), ((105, 108), 0), ((70, 108), 25), ((32, 108), -25),
        ((32, 70), 0), ((70, 45), 30), ((70, 95), -30)]
# (x, y cm), then the heading: this many deg off the direction to the middle of the arena (None: as it stands). Facing the
# middle the front camera sees several tags; facing a wall from close by it sees one, and the pose is not good enough.


def auto(label: str, arm: bool, tour: bool = False) -> None:
    import math

    import numpy as np

    from astar import Grid, plan
    from astar_drive import follow
    from route_follow import Abort, RouteFollower, parse_args
    from vio import VioFilter
    from vio_check import TagLogger

    root = SERVER / "route_follow_debug" / "survey" / label
    car = RouteFollower(parse_args(["--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"] + (["--arm"] if arm else [])))
    feed = tags = None
    try:
        car.start()
        arena = tuple(float(v) for v in car.arena["arena_size_m"])
        tags = TagLogger(car)
        feed = GimbalFeed().start()
        feed.wait_ready()
        point = lambda pan: car.servo.set_pan(servo_pan_for_bearing(car.home_pan, car.right_90_pan, pan, car.left_90_pan))
        n = len(list(root.glob("*_x*"))) if root.exists() else 0

        def standing():
            """The pose as the car really stands, from the front and rear cameras' joint tag fits; None when not good enough."""
            time.sleep(0.8)
            t0 = time.monotonic()
            time.sleep(2.5)
            got = [r for r in list(tags.rows) if r["t"] >= t0 and r["pose"]]
            rows = [r["pose"] for r in got]
            if len(rows) < 10:
                return None, f"only {len(rows)} joint tag poses in 2.5 s"
            x, y = float(np.median([p["x"] for p in rows])), float(np.median([p["y"] for p in rows]))
            ref = rows[0]["yaw"]
            offs = [(p["yaw"] - ref + math.pi) % (2 * math.pi) - math.pi for p in rows]
            yaw = ref + float(np.median(offs))
            spread = float(np.hypot(np.std([p["x"] for p in rows]), np.std([p["y"] for p in rows]))) * 100
            turn = math.degrees(float(np.std(offs)))
            info = f"{len(rows)} poses, cameras {sorted({r['camera'] for r in got})}, {np.median([p['tags'] for p in rows]):.0f} tags, spread {spread:.1f} cm {turn:.1f} deg"
            if spread > 2.0 or turn > 1.5:
                return None, "the tag poses do not agree: " + info
            return (x, y, yaw), info

        steps = TOUR if tour else [(None, 90 if k else 0) for k in range(4 if arm else 1)]
        pose, info = standing()
        for _ in range(4):
            if pose is not None or not arm:
                break
            print(f"not good enough here ({info}); turning 45 deg and looking again", flush=True)
            car.move("left", 45.0 / car.turn_rate("left"), math.radians(45.0))
            pose, info = standing()
        if pose is None:
            raise Abort(info)
        for target, turn in steps:
            if target is not None:
                goal = (target[0] / 100.0, target[1] / 100.0)
                path = plan(Grid(arena, [], car.body_half[0], car.body_half[1], pose[2], 0.06), (pose[0], pose[1]), goal)
                if path is None:
                    print(f"no way to {target}; skipped", flush=True)
                    continue
                point(0.0)
                why = follow(car, tags, VioFilter(*pose), path, goal, pose[2], 0.13, dict(rows=[]), [], (arena, []))
                car.stop()
                print(f"drove to {target}: {why}", flush=True)
            def turn_by(deg: float) -> None:
                if abs(deg) > 4.0:
                    kind = "left" if deg > 0 else "right"
                    car.move(kind, abs(deg) / car.turn_rate(kind), math.radians(deg))

            if tour and turn is not None and target is not None:
                want = math.atan2(arena[1] / 2 - goal[1], arena[0] / 2 - goal[0]) + math.radians(turn)
                turn_by(math.degrees((want - pose[2] + math.pi) % (2 * math.pi) - math.pi))
            elif not tour and turn:
                turn_by(float(turn))
            pose_new, info = standing()
            for _ in range(3):
                if pose_new is not None:
                    break
                print(f"not good enough here ({info}); turning 35 deg and looking again", flush=True)
                turn_by(35.0)
                pose_new, info = standing()
            if pose_new is None:
                raise Abort("no pose to go on from: " + info)
            pose = pose_new
            x, y, yaw = pose
            n += 1
            print(f"pose {n} as it stands ({info}): x={x * 100:.1f} y={y * 100:.1f} heading {math.degrees(yaw):+.1f}", flush=True)
            obs = dict(pose_cm_deg=[x * 100, y * 100, math.degrees(yaw)], pose_from="front and rear cameras' tags", detail=info,
                       time=time.strftime("%Y-%m-%d %H:%M:%S"), pans={})
            sweep(loc_for(car), feed, point, root / f"{n:02d}_x{x * 100:.0f}_y{y * 100:.0f}_h{math.degrees(yaw):.0f}", obs)
    except Abort as exc:
        print(f"ABORT: {exc}", flush=True)
    finally:
        if tags is not None:
            tags.stop.set()
        if feed is not None:
            feed.close()
        car.close()
    print(f"saved under {root}\nnext: python3 tools/solve_gimbal.py {label}")


def walls(label: str) -> None:
    import math
    import threading

    import numpy as np

    from astar import Grid, plan
    from astar_drive import follow
    from patrol.drive import Wheels
    from patrol.sensing import Sonar
    from route_common import SONIC_ARM_M, SONIC_PIVOT_M
    from route_follow import Abort, RouteFollower, parse_args
    from vio import VioFilter
    from vio_check import TagLogger

    NEAR_M, SPEED, AHEAD_FACE = 0.22, 0.12, SONIC_PIVOT_M + SONIC_ARM_M
    root = SERVER / "route_follow_debug" / "survey" / label
    car = RouteFollower(parse_args(["--arm", "--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"]))
    feed = tags = sonar = None
    try:
        car.start()
        W, H = (float(v) for v in car.arena["arena_size_m"])
        tags, sonar = TagLogger(car), Sonar(car.sensors)
        feed = GimbalFeed().start()
        feed.wait_ready()
        wheels = Wheels(car, threading.Lock())
        imu, gyro = car.gyro.imu, car.gyro
        ref = dict(yaw=0.0, deg=0.0, t=0.0)
        point = lambda pan: car.servo.set_pan(servo_pan_for_bearing(car.home_pan, car.right_90_pan, pan, car.left_90_pan))

        def heading() -> float:
            deg, t = imu.yaw_integral()
            return ref["yaw"] + math.radians(((deg - ref["deg"]) - gyro.bias_dps * (t - ref["t"])) * gyro.scale)

        def set_heading(yaw: float) -> None:
            ref["deg"], ref["t"] = imu.yaw_integral()
            ref["yaw"] = yaw

        def echo(n: int = 7):
            v = [r for r in (sonar.read(time.monotonic(), 0.4) for _ in range(n)) if r is not None]
            return float(np.median(v)) if len(v) >= 3 else None

        def standing():
            time.sleep(0.8)
            t0 = time.monotonic()
            time.sleep(2.5)
            rows = [r["pose"] for r in list(tags.rows) if r["t"] >= t0 and r["pose"]]
            if len(rows) < 10:
                return None
            r0 = rows[0]["yaw"]
            offs = [(p["yaw"] - r0 + math.pi) % (2 * math.pi) - math.pi for p in rows]
            if math.degrees(float(np.std(offs))) > 1.5:
                return None
            return float(np.median([p["x"] for p in rows])), float(np.median([p["y"] for p in rows])), r0 + float(np.median(offs))

        def glide(vx: float, vy: float, want: float, seconds: float, until_m: float | None = None) -> float:
            """Move in the body frame holding the heading `want` by the gyro; stop at the time or when the echo is this near."""
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds and not car.stop_event.is_set():
                if until_m is not None:
                    _began, value = sonar.latest
                    if value is not None and time.monotonic() - _began < 0.4 and value / 100.0 <= until_m:
                        break
                err = (want - heading() + math.pi) % (2 * math.pi) - math.pi
                wheels.send(vx, vy, max(-0.6, min(0.6, 2.0 * err)))
                time.sleep(0.02)
            wheels.stop()
            time.sleep(0.5)
            if car.stop_event.is_set():
                raise Abort("stopped")
            return time.monotonic() - t0 - 0.5

        def to_middle() -> tuple:
            for _ in range(4):
                pose = standing()
                if pose is None:
                    car.move("left", 40.0 / car.turn_rate("left"), math.radians(40.0))
                    continue
                if math.hypot(pose[0] - W / 2, pose[1] - H / 2) < 0.05:
                    return pose
                path = plan(Grid((W, H), [], car.body_half[0], car.body_half[1], pose[2], 0.06), (pose[0], pose[1]), (W / 2, H / 2))
                point(0.0)
                try:
                    follow(car, tags, VioFilter(*pose), path, (W / 2, H / 2), pose[2], 0.13, dict(rows=[]), [], ((W, H), []))
                except Abort as exc:
                    print(f"  on the way to the middle: {exc}", flush=True)
                car.stop()
            raise Abort("could not get to the middle with a good pose")

        n = len(list(root.glob("*_x*"))) if root.exists() else 0
        for k in range(4):
            want = k * math.pi / 2.0
            pose = to_middle()
            turn = (want - pose[2] + math.pi) % (2 * math.pi) - math.pi
            for _ in range(3):
                if abs(math.degrees(turn)) <= 2.0:
                    break
                kind = "left" if turn > 0 else "right"
                car.move(kind, max(abs(math.degrees(turn)) / car.turn_rate(kind), 0.05), turn)
                pose = standing() or pose
                turn = (want - pose[2] + math.pi) % (2 * math.pi) - math.pi
            set_heading(pose[2])
            print(f"heading set {k + 1}: in the middle x={pose[0] * 100:.1f} y={pose[1] * 100:.1f} heading {math.degrees(pose[2]):+.1f} (tags); holding it by the gyro", flush=True)
            point(0.0)
            time.sleep(0.6)
            t_fwd = glide(SPEED, 0.0, pose[2], 6.0, NEAR_M)
            t_side = 0.0
            for side, pan in (("right", -90.0), ("left", 90.0)):
                point(pan)
                time.sleep(0.7)
                sign = -1.0 if side == "right" else 1.0
                t_side += sign * glide(0.0, sign * SPEED, pose[2], 11.0, NEAR_M)
                e_side = echo()
                point(0.0)
                time.sleep(0.7)
                e_ahead = echo()
                yaw = heading()
                if e_side is None or e_ahead is None or e_side > 0.40 or e_ahead > 0.40:
                    print(f"  {side}: echoes ahead {e_ahead} side {e_side}: not close to two walls; skipped", flush=True)
                    continue
                d = yaw - want                                   # how far off square it stands (small)
                ahead_c = e_ahead + AHEAD_FACE * math.cos(d)      # car centre to the wall ahead
                side_c = e_side + SONIC_ARM_M * math.cos(d) + sign * SONIC_PIVOT_M * math.sin(d)   # ... to the wall on that side (the nose swings towards it)
                c, s_ = math.cos(want), math.sin(want)
                # the wall ahead is at distance ahead_c along `want`; the side wall at side_c along want + sign * 90 deg
                fx, fy = c, s_
                lx, ly = -s_ * sign, c * sign
                def wall_dist(ux, uy):                            # from the arena's middle to the wall in direction u
                    return (W / 2 if abs(ux) > 0.5 else H / 2)
                mx = fx * (wall_dist(fx, fy) - ahead_c) + lx * (wall_dist(lx, ly) - side_c)
                my = fy * (wall_dist(fx, fy) - ahead_c) + ly * (wall_dist(lx, ly) - side_c)
                x, y = W / 2 + mx, H / 2 + my
                tag_now = standing()
                n += 1
                print(f"pose {n} by the ultrasonic (ahead {e_ahead * 100:.1f} cm, {side} {e_side * 100:.1f} cm) and the gyro: x={x * 100:.1f} y={y * 100:.1f} heading {math.degrees(yaw):+.1f}"
                      + ("" if tag_now is None else f"   [front/rear tags say x={tag_now[0] * 100:.1f} y={tag_now[1] * 100:.1f} heading {math.degrees(tag_now[2]):+.1f}]"), flush=True)
                obs = dict(pose_cm_deg=[x * 100, y * 100, math.degrees(yaw)], pose_from="ultrasonic to two walls, heading by the gyro since the middle",
                           echoes_cm=[e_ahead * 100, e_side * 100], side=side, tags_pose=None if tag_now is None else [tag_now[0] * 100, tag_now[1] * 100, math.degrees(tag_now[2])],
                           time=time.strftime("%Y-%m-%d %H:%M:%S"), pans={})
                sweep(loc_for(car), feed, point, root / f"{n:02d}_x{x * 100:.0f}_y{y * 100:.0f}_h{math.degrees(yaw):.0f}", obs)
            glide(0.0, -math.copysign(SPEED, t_side), pose[2], abs(t_side))
            glide(-SPEED, 0.0, pose[2], t_fwd)
    except Abort as exc:
        print(f"ABORT: {exc}", flush=True)
    finally:
        try:
            car.stop()
        except Exception:  # noqa: BLE001
            pass
        if sonar is not None:
            sonar.halt.set()
        if tags is not None:
            tags.stop.set()
        if feed is not None:
            feed.close()
        car.close()
    print(f"saved under {root}\nnext: python3 tools/solve_gimbal.py {label}")


def loc_for(car):
    return car.localizer


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = args[0] if args else "gimbal1"
    if "--walls" in sys.argv:
        walls(label)
        return
    if "--auto" in sys.argv or "--tour" in sys.argv:
        auto(label, "--arm" in sys.argv, tour="--tour" in sys.argv)
        return
    arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
    loc = ArucoLocalizer(arena, car)
    root = SERVER / "route_follow_debug" / "survey" / label
    feed = GimbalFeed().start()
    feed.wait_ready()
    servo = Servo()
    home, _tilt = load_gimbal_home()
    right90, left90 = load_right_90_pan(home), load_left_90_pan()
    n = len(list(root.glob("*_x*"))) if root.exists() else 0
    try:
        while True:
            text = input("\npose: e HEADING AHEAD BEHIND LEFT RIGHT  (cm from the body edges, - = not measured; q = finish) > ").strip()
            if text.lower() in ("q", "quit", ""):
                break
            try:
                parts = text.replace(",", " ").split()
                assert parts[0].lower() == "e"
                vals = [None if v == "-" else float(v) for v in parts[2:6]]
                w, h = (float(v) * 100.0 for v in arena["arena_size_m"])
                bl, bw = (float(v) * 100.0 for v in car["body_m"])
                cx, cy, notes = pose_from_edges(int(float(parts[1])), *vals, w, h, bl, bw)
            except (ValueError, AssertionError, IndexError) as exc:
                print(f"  not understood ({exc}); e.g.  e 0 40 95 62 58")
                continue
            pose = (cx, cy, float(parts[1]))
            print(f"  centre of the car: x={cx:.1f} cm  y={cy:.1f} cm  heading {pose[2]:.0f}")
            for note in notes:
                print("  " + note)
            n += 1
            folder = root / f"{n:02d}_x{cx:.0f}_y{cy:.0f}_h{pose[2]:.0f}"
            folder.mkdir(parents=True, exist_ok=True)
            obs = dict(pose_cm_deg=list(pose), time=time.strftime("%Y-%m-%d %H:%M:%S"), pans={})
            last = None
            for pan in PANS:
                servo.set_pan(servo_pan_for_bearing(home, right90, float(pan), left90))
                time.sleep(0.9 if last is None else 0.5)
                last = pan
                frame = feed.fresh(time.monotonic(), 1.5)
                found, _fx = loc.detect("gimbal", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                cv2.imwrite(str(folder / f"pan{pan:+04d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                obs["pans"][str(pan)] = tags_json(found)
                print(f"  pan {pan:+4d}: tags {sorted(found) or '-'}", flush=True)
            servo.set_pan(home)
            (folder / "obs.json").write_text(json.dumps(obs) + "\n")
            views = sum(len(v) for v in obs["pans"].values())
            print(f"  saved {folder.name}: {views} tag views, tags {sorted({int(t) for v in obs['pans'].values() for t in v})}")
    finally:
        try:
            servo.set_pan(home)
            time.sleep(0.4)
            servo.release_gimbal()
        except Exception:  # noqa: BLE001
            pass
        feed.close()
    print(f"saved under {root}\nnext: python3 tools/solve_gimbal.py {label}")


if __name__ == "__main__":
    main()
