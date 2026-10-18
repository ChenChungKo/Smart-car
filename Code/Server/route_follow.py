#!/usr/bin/env python3
"""Drive a waypoint route from arena_map.json using ArUco wall tags.

Each cycle does four things, in this order:
1. Localize (ArUco + Kalman).
2. Look: ultrasonic ±90° updates the live map with the cameras.
3. Plan: A* on that map to the real waypoint. The waypoint is never skipped.
4. Act: one in-place turn toward the next path point, or one short drive.

No sideways shuffle, no separate gap hunter, no second planner. The live map
is the only occupancy the route uses. A forward move still stops on ultrasonic
or IR. Dry-run unless --arm is given.

Dry-run unless --arm is given.
    python3 route_follow.py --route loop            # decisions only
    python3 route_follow.py --route loop --arm      # drive
    python3 route_follow.py --calibrate-motion --arm
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics as st
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from floor_scan import FloorScanner, to_world
from live_map import LiveMap
from pose_kalman import PoseKalman
from path_plan import choose_exit, lookahead, plan_path
from aruco_localizer import (
    CAR_PATH,
    MAP_PATH,
    ArucoLocalizer,
    Pose2D,
    draw_map,
    draw_overlay,
    load_json,
    open_grabbers,
    wrap_rad,
)
from full_car_test import estimate_battery_percent
from safe_surround_cruise import SafetySensors, read_battery_voltage
from vision_detector import classic_corridor_score

SERVER = Path(__file__).resolve().parent


def _default_route_name() -> str:
    try:
        return load_json(Path(MAP_PATH)).get("default_route", "rect")
    except OSError:
        return "rect"


def parse_args():
    p = argparse.ArgumentParser(description="ArUco waypoint route following.")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--car", default=str(CAR_PATH))
    p.add_argument("--route", default=_default_route_name(), help="Route name in arena_map.json routes.")
    p.add_argument("--repeat", action="store_true", help="Loop the route until Ctrl+C.")
    p.add_argument("--arm", action="store_true", help="Enable motors. Without it nothing moves.")
    p.add_argument("--calibrate-motion", action="store_true",
                   help="Measure forward m/s and turn deg/s with tags, write car_nav.json.")
    p.add_argument("--measure-drift", action="store_true",
                   help="Drive straight once and print lateral offset and yaw change. Does not steer or save.")
    p.add_argument("--drift-m", type=float, default=0.80, help="How far --measure-drift tries to drive.")
    p.add_argument("--wheel-pwm", default=None,
                   help="Forward duties left-front,left-rear,right-front,right-rear. "
                        "Default: motion.forward_wheel_pwm, or four times forward_pwm.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--no-rear", action="store_true", help="Localize with the front CSI only.")
    p.add_argument("--arrive-m", type=float, default=0.10)
    p.add_argument("--align-deg", type=float, default=12.0)
    p.add_argument("--max-segment-m", type=float, default=0.35)
    p.add_argument("--wall-margin-m", type=float, default=0.04,
                   help="Keep every body corner at least this far from the arena walls.")
    p.add_argument("--start-at-first", action="store_true",
                   help="Drive to the first waypoint even on a closed route (default: join nearest side).")
    p.add_argument("--return-home", action="store_true",
                   help="Treat the start pose as home: run the closed route from there and park back on it "
                        "(same heading), fine-tuning with mecanum strafes.")
    p.add_argument("--hug", action="store_true",
                   help="Mecanum wall-hugging lap: strafe into a lane --hug-gap-m from each wall, drive along it, "
                        "and turn only at corner spots far enough from both walls (ignores --route).")
    p.add_argument("--hug-gap-m", type=float, default=0.10,
                   help="Body side to wall gap in --hug lanes (8 cm is tight on 1.4 m arena).")
    p.add_argument("--lane-tol-m", type=float, default=0.02, help="Strafe back onto the lane beyond this.")
    p.add_argument("--max-jump-m", type=float, default=0.15,
                   help="Ignore a tag fix this far from the dead-reckoned pose (up to 2 in a row).")
    p.add_argument("--max-jump-deg", type=float, default=20.0)
    p.add_argument("--home", default=None,
                   help="Known home pose 'x,y,yaw_deg' (overrides arena_map.json \"home\").")
    p.add_argument("--home-tol-m", type=float, default=0.03)
    p.add_argument("--home-tol-deg", type=float, default=4.0)
    p.add_argument("--lookahead-m", type=float, default=0.15,
                   help="Steer at a point this far ahead on the route line.")
    p.add_argument("--min-turn-s", type=float, default=0.15)
    p.add_argument("--max-turn-s", type=float, default=1.2)
    p.add_argument("--settle-s", type=float, default=0.35, help="Wait after stopping before trusting frames.")
    p.add_argument("--fix-wait-s", type=float, default=1.5)
    p.add_argument("--max-blind-segments", type=int, default=2,
                   help="Segments allowed on dead reckoning before searching for tags.")
    p.add_argument("--search-step-deg", type=float, default=30.0)
    p.add_argument("--stop-cm", type=float, default=18.0)
    p.add_argument("--creep-cm", type=float, default=32.0)
    p.add_argument("--near-score", type=float, default=0.70)
    p.add_argument("--camera-safety", action="store_true",
                   help="Also stop on the front-camera corridor score (foam walls fill the fisheye, so off by default).")
    p.add_argument("--front-ir-gate-cm", type=float, default=18.0)
    p.add_argument("--blocked-wait-s", type=float, default=6.0)
    p.add_argument("--no-avoid", action="store_true",
                   help="Stop and wait at obstacles instead of going around them.")
    p.add_argument("--avoid-cm", type=float, default=28.0,
                   help="Ultrasonic range that can count as an obstacle.")
    p.add_argument("--avoid-margin-cm", type=float, default=12.0,
                   help="A reading must be this much closer than the mapped wall to be an obstacle.")
    p.add_argument("--avoid-side-m", type=float, default=0.22, help="Sideways step around an obstacle.")
    p.add_argument("--avoid-max-side-m", type=float, default=0.70,
                   help="Widest sideways step allowed when the obstacle is wide.")
    p.add_argument("--avoid-pass-m", type=float, default=0.80, help="Longest drive alongside an obstacle.")
    p.add_argument("--scan-test", action="store_true",
                   help="Localize, print the camera obstacle scan and the detour plan, then exit (no driving).")
    p.add_argument("--start-anywhere", action="store_true",
                   help="Locate the start pose from tags (turning to find them) and park on the map home.")
    p.add_argument("--sonic-x-m", type=float, default=0.10, help="Ultrasonic distance ahead of the car centre.")
    p.add_argument("--avoid-look-m", type=float, default=0.50,
                   help="Go around a camera-seen obstacle once its near face is this close (from the car centre).")
    p.add_argument("--avoid-gap-m", type=float, default=0.06, help="Side clearance kept from an obstacle edge.")
    p.add_argument("--gap-margin-m", type=float, default=0.025,
                   help="Clearance each side of the body when squeezing through a gap the gimbal camera measured.")
    p.add_argument("--no-gimbal-gap", action="store_true",
                   help="Do not measure gaps with the gimbal camera before driving through a corridor.")
    p.add_argument("--avoid-depth-m", type=float, default=0.20,
                   help="Assumed obstacle depth (the camera only sees its near face).")
    p.add_argument("--wall-band-m", type=float, default=0.06,
                   help="Foam feet this close to a mapped wall count as the wall, not an obstacle.")
    p.add_argument("--min-battery-v", type=float, default=7.0)
    p.add_argument("--no-ir", action="store_true")
    p.add_argument("--no-sonic", action="store_true")
    p.set_defaults(active_high=False)
    p.add_argument("--ir-active-high", action="store_true", dest="active_high")
    p.add_argument("--debug-dir", default="route_follow_debug")
    p.add_argument("--no-window", action="store_true", help="Do not open the live trajectory window.")
    p.add_argument("--record", nargs="?", const="route_follow_debug/run.mp4", default=None,
                   help="Save the live view (trajectory plus both cameras) to this mp4 "
                        "(default route_follow_debug/run.mp4).")
    return p.parse_args()


class Abort(Exception):
    pass


class RouteFollower:
    def __init__(self, args):
        self.args = args
        self.arena = load_json(Path(args.map))
        self.car_path = Path(args.car)
        self.car = load_json(self.car_path)
        self.motion = self.car["motion"]
        self.localizer = ArucoLocalizer(self.arena, self.car)
        routes = self.arena.get("routes", {})
        if not args.calibrate_motion and not args.measure_drift and args.route not in routes:
            raise SystemExit(f"route '{args.route}' not in {args.map}; have {sorted(routes)}")
        self.route = [tuple(map(float, wp)) for wp in routes.get(args.route, [])]
        self.grabbers = {}
        self.sensors = None
        self.motor = None
        self.servo = None
        self.home_pan, self.home_tilt = 90, 90
        self.right_90_pan = 180
        self.left_90_pan = None
        self.pan_now = 90
        self.pose: Pose2D | None = None
        self.block_yaw: float | None = None
        self.block_at: tuple[float, float] | None = None
        self.trail: list[tuple[float, float, float]] = []
        self.stop_event = threading.Event()
        self.sonic_misses = 0
        self.front_risk = 0.0
        self.move_started: float | None = None
        self.move_kind: str | None = None
        self.debug_dir = SERVER / args.debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.window = None
        self.writer = None
        self.record_path = Path(args.record) if args.record else None
        if not args.no_window:
            try:
                cv2.namedWindow("route", cv2.WINDOW_NORMAL)
                self.window = "route"
            except cv2.error as exc:
                print(f"WARNING no display for the trajectory window: {exc}", flush=True)
        length, width = (float(v) for v in self.car.get("body_m", [0.29, 0.18]))
        self.body_half = (length / 2.0, width / 2.0)
        arena_w, arena_h = (float(v) for v in self.arena["arena_size_m"])
        self.live_map = LiveMap(arena_w, arena_h)
        self.kalman = PoseKalman(arena_w, arena_h, self.body_half)
        self.guide_path: list[tuple[float, float]] = []
        self.home: Pose2D | None = None
        self.strafe_mps = float(self.motion.get("strafe_mps", 0.6 * float(self.motion["forward_mps"])))
        self.strafe_base = self.strafe_mps
        self.strafe_flip = False
        self.strafe_wrong = 0
        self.jumps = 0
        self.jump_pose: Pose2D | None = None
        self.parking = False
        self.lane_yaw: float | None = None
        self.lane_check = False
        self.allow_gap: float | None = None
        self.target_ahead: float | None = None
        self.world_obstacles: list[np.ndarray] = []
        self.floor = FloorScanner(self.localizer, self.car["camera_mounts"]["front"], "front")
        self.side_floors = {}
        for s_name in ("left", "right"):
            if s_name in self.car.get("camera_mounts", {}):
                self.side_floors[s_name] = FloorScanner(
                    self.localizer, self.car["camera_mounts"][s_name], s_name, hsv_lo=(90, 110, 40)
                )
        self.gimbal_cam = None
        if not args.no_gimbal_gap and self.car.get("gimbal_camera", {}).get("calibrated"):
            from gimbal_gap import GimbalCamera

            self.gimbal_cam = GimbalCamera(self.car)
        self.gap_drive = False
        self.hug_poses: list[tuple[float, float, float]] = []
        if args.hug:
            self.hug_poses = self.build_hug_poses()
            self.route = [(x, y) for x, y, _ in self.hug_poses] + [self.hug_poses[0][:2]]

    # ---- hardware -------------------------------------------------------

    def start(self):
        from servo import Servo

        try:
            from servo import load_gimbal_home, load_left_90_pan, load_right_90_pan

            self.servo = Servo()  # applies gimbal_home.json: ultrasonic straight ahead
            self.home_pan, self.home_tilt = load_gimbal_home()
            self.right_90_pan = load_right_90_pan(self.home_pan)
            self.left_90_pan = load_left_90_pan()
            self.pan_now = self.home_pan
        except Exception as exc:
            print(f"WARNING gimbal: {exc}", flush=True)
        self.grabbers = open_grabbers(
            self.car, self.args.width, self.args.height, rear=not self.args.no_rear, sides=True
        )
        self.sensors = SafetySensors(
            SimpleNamespace(no_ir=self.args.no_ir, no_sonic=self.args.no_sonic, active_high=self.args.active_high)
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not all(g.latest()[0] is not None for g in self.grabbers.values()):
            time.sleep(0.1)
        missing = [name for name, g in self.grabbers.items() if g.latest()[0] is None]
        required = [name for name in missing if name in ("front", "rear")]
        if required:
            raise Abort(f"cameras not ready: {required}")
        for name in missing:
            print(f"WARNING {name} camera not ready; localizing without it", flush=True)
            self.grabbers[name].close()
            del self.grabbers[name]
        if self.args.arm:
            battery = read_battery_voltage()
            if battery is None or battery < self.args.min_battery_v:
                raise Abort(f"ARM refused: battery {battery}V < {self.args.min_battery_v}V")
            print(f"ARMED battery={battery:.2f}V ({estimate_battery_percent(battery)}%). Starting in 3 s.", flush=True)
            for n in (3, 2, 1):
                print(n, flush=True)
                if self.stop_event.wait(1.0):
                    raise Abort("stopped")
            from motor import Ordinary_Car

            self.motor = Ordinary_Car()
        else:
            print("DRY-RUN: no motor object; decisions are printed only.", flush=True)

    def drive(self, pwm):
        if self.motor is not None:
            self.motor.set_motor_model(*pwm)

    def stop(self):
        if self.motor is not None:
            try:
                self.motor.set_motor_model(0, 0, 0, 0)
            except Exception:
                pass

    def request_stop(self, *_args):
        self.stop_event.set()
        self.stop()

    def close(self):
        self.stop_event.set()
        self.stop()
        if self.motor is not None and abs(self.strafe_mps - self.strafe_base) > 0.005:
            self.motion["strafe_mps"] = round(self.strafe_mps, 3)
            self.car["motion"] = self.motion
            self.car_path.write_text(json.dumps(self.car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"saved learned strafe speed {self.strafe_mps:.3f} m/s to {self.car_path.name}", flush=True)
        if self.motor is not None:
            try:
                self.motor.close()
            except Exception:
                pass
        if self.sensors is not None:
            self.sensors.close()
        if self.servo is not None and hasattr(self.servo, "pwm_servo"):
            try:
                self.servo.pwm_servo.set_pwm(8, 0, 0)
                self.servo.pwm_servo.set_pwm(9, 0, 0)
            except Exception:
                pass
        for grabber in self.grabbers.values():
            grabber.close()
        self.save_map()
        if self.writer is not None:
            self.writer.release()
            print(f"saved recording {self.record_path}", flush=True)
        if self.window is not None:
            cv2.destroyWindow(self.window)

    # ---- sensing --------------------------------------------------------

    def pan_to(self, angle: int):
        """Turn the ultrasonic toward `angle` in 15° steps instead of one snap."""
        if self.servo is None:
            return
        angle = int(round(angle))
        while abs(angle - self.pan_now) > 1:
            step = max(-15, min(15, angle - self.pan_now))
            self.pan_now += step
            self.servo.set_servo_pwm("0", self.pan_now)
            if self.pan_now == angle:
                break
            if self.stop_event.wait(0.12):
                raise Abort("stopped")
        self.pan_now = angle

    def look(self, pan_deg: float):
        """Point the ultrasonic at `pan_deg` relative to straight ahead (positive = left) and read it."""
        if self.servo is None or self.sensors is None:
            return None
        from servo import servo_pan_for_bearing

        self.pan_to(servo_pan_for_bearing(
            self.home_pan, self.right_90_pan, pan_deg, self.left_90_pan))
        if self.stop_event.wait(0.35):
            raise Abort("stopped")
        cm = self.sensors.read_sonic()
        self.note_range(pan_deg, cm)
        return cm

    def sweep_correct(self):
        """Replace the sideways part of the pose with what the ultrasonic sweep measures."""
        found = self.sweep()
        if self.pose is None or len(found) < 2:
            return
        near = {angle: cm for angle, cm in found.items() if abs(angle) >= 30 and cm <= 40.0}
        if not near:
            print("    sweep saw no nearby side wall; keeping the camera pose", flush=True)
            return
        angle = min(near, key=near.get)
        opposite = [cm for other, cm in found.items() if other * angle < 0]
        if opposite and min(opposite) < near[angle] + 15.0:
            print("    sweep sides look alike; keeping the camera pose", flush=True)
            return
        found = {angle: near[angle]}
        w, h = (float(v) for v in self.arena["arena_size_m"])
        yaw = self.pose.yaw
        best = None
        for angle, cm in found.items():
            bearing = yaw + math.radians(angle)
            reach = cm / 100.0 + 0.02
            dx, dy = math.cos(bearing), math.sin(bearing)
            dist = min(
                ((w - self.pose.x) / dx) if dx > 0.5 else float("inf"),
                (self.pose.x / -dx) if dx < -0.5 else float("inf"),
                ((h - self.pose.y) / dy) if dy > 0.5 else float("inf"),
                (self.pose.y / -dy) if dy < -0.5 else float("inf"),
            )
            # The ultrasonic range is the trusted measurement; the camera pose only decides which
            # wall the beam is pointing at.
            if dist < float("inf") and (best is None or abs(angle) < abs(best[0])):
                best = (angle, dist - reach, dx, dy)
        if best is None:
            print("    sweep did not match a wall; keeping the camera pose", flush=True)
            return
        _angle, gap, dx, dy = best
        # Slide the pose sideways (perpendicular to the heading) so the wall sits where the
        # ultrasonic says it is.
        sideways = math.copysign(gap, -dx * math.sin(yaw) + dy * math.cos(yaw) or 1.0)
        self.pose.x -= sideways * math.sin(yaw)
        self.pose.y += sideways * math.cos(yaw)
        shift = sideways
        print(f"    sweep moved the pose {shift * 100:+.0f}cm sideways -> {self.pose.text()}", flush=True)

    def sweep(self) -> dict[int, float]:
        """Range left, ahead and right; returns the angles that answered, in degrees."""
        found = {}
        try:
            for angle in (60, 30, 0, -30, -60):
                cm = self.look(angle)
                if cm is not None:
                    found[angle] = cm
                print(f"    sweep {angle:+d}deg: {cm if cm is not None else '-'}cm", flush=True)
        finally:
            if self.servo is not None:
                self.pan_to(self.home_pan)
        return found

    def safety_reason(self) -> str | None:
        """Why a forward segment must stop right now, or None."""
        sonic = self.sensors.read_sonic()
        if sonic is None and not self.args.no_sonic:
            self.sonic_misses += 1
            if self.sonic_misses >= 5:
                return "sonic_missing"
        else:
            self.sonic_misses = 0
        travelled = 0.0
        if self.move_kind == "forward" and self.move_started is not None:
            travelled = float(self.motion["forward_mps"]) * (time.monotonic() - self.move_started)
        # A camera-measured gap is wide enough for the body. The ultrasonic cone hits the
        # foam corners and would otherwise stop the drive at 24 cm; only the hard stop remains.
        if self.gap_drive and self.move_kind == "forward":
            # 10 cm, not the usual 18 cm: a confirmed gap's corners read about 17 cm while the body still fits.
            if sonic is not None and sonic < 10.0:
                return f"sonic_{sonic:.0f}cm"
            return None
        # The ultrasonic faces forward: it must not stop a reverse or strafe that moves away from what it sees.
        if self.move_kind not in ("reverse", "strafe_left", "strafe_right"):
            if self.avoid_active() and self.is_obstacle(sonic, 0.0, travelled):
                return f"obstacle_{sonic:.0f}cm"
            if sonic is not None and sonic < self.args.stop_cm:
                return f"sonic_{sonic:.0f}cm"
        ir = self.sensors.read_ir()
        if self.move_kind == "reverse":
            if any(ir.get(k) for k in ("rear_left", "rear_mid", "rear_right")):
                return "ir_rear"
            return None
        if self.move_kind == "strafe_left":
            if any(ir.get(k) for k in ("front_left", "wheel_left_front", "wheel_left_rear")):
                return "ir_left"
            return None
        if self.move_kind == "strafe_right":
            if any(ir.get(k) for k in ("front_right", "wheel_right_front", "wheel_right_rear")):
                return "ir_right"
            return None
        if self.avoid_active() and ir["front"] and self.wall_range_cm(0.0, travelled) > 25.0:
            return "obstacle_ir"
        near = sonic is None or sonic < self.args.front_ir_gate_cm
        if near and (ir["front"] or ir["front_left"] or ir["front_right"]) \
                and not (sonic is not None and sonic > self.args.front_ir_gate_cm - 6.0):
            return "ir_front"
        if not self.args.camera_safety:
            return None
        frame, stamp = self.grabbers["front"].latest()
        if frame is not None and time.monotonic() - stamp < 1.0:
            mid = classic_corridor_score(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA))["mid"]
            self.front_risk = 0.4 * mid + 0.6 * self.front_risk
            # A far foam wall also fills the fisheye; only trust the camera near the wall.
            if self.front_risk >= self.args.near_score and sonic is not None and sonic < self.args.creep_cm:
                return f"camera_{self.front_risk:.2f}"
        return None

    def localize(self, timeout: float | None = None, gate: bool = True) -> Pose2D | None:
        self.stop()
        t0 = time.monotonic()
        if self.stop_event.wait(self.args.settle_s):
            raise Abort("stopped")
        fresh_after = t0 + self.args.settle_s
        deadline = fresh_after + (self.args.fix_wait_s if timeout is None else timeout)
        seen = {name: 0.0 for name in self.grabbers}
        fixes = []
        views = 0
        while time.monotonic() < deadline and not self.stop_event.is_set():
            for name, grabber in self.grabbers.items():
                frame, stamp = grabber.latest()
                if frame is None or stamp <= fresh_after or stamp == seen[name]:
                    continue
                seen[name] = stamp
                found, corners, ids = self.localizer.observe(name, frame)
                fixes.extend(found)
                if name == "front":
                    cv2.imwrite(str(self.debug_dir / "last_front.jpg"),
                                draw_overlay(frame, corners, ids, found, "front"))
                views += 1
            if fixes and views >= 2 * len(self.grabbers):
                break
            time.sleep(0.03)
        pose = self.localizer.fuse_available(fixes, time.monotonic())
        if pose is None and fixes:
            print(f"  no fix: {self.localizer.last_reject}; dead-reckoning", flush=True)
        if pose is not None:
            pose = self.filter_pose(pose)
        return self.accept_fix(pose) if gate else pose

    def filter_pose(self, pose: Pose2D) -> Pose2D:
        """Kalman-fuse a tag pose, then keep the whole body inside the arena."""
        nearest = min((fix.distance for fix in pose.fixes), default=1.0)
        tags = len({(fix.camera, fix.tag_id) for fix in pose.fixes}) or 1
        raw_x, raw_y = pose.x, pose.y
        clipped = self.kalman.update_tag(pose.x, pose.y, pose.yaw, tags, nearest)
        pose.x, pose.y, pose.yaw = (float(v) for v in self.kalman.x)
        if clipped:
            print(f"  kalman kept the body inside the arena: tag ({raw_x:.2f},{raw_y:.2f}) "
                  f"-> ({pose.x:.2f},{pose.y:.2f})", flush=True)
        return pose

    def accept_fix(self, pose: Pose2D | None) -> Pose2D | None:
        """Drop a weak fix that jumps away from the dead-reckoned pose.

        Strong fixes (3+ tags, or one close tag) always pass. Weak ones pass only after
        4 in a row agree on the same new pose, and never while parking.
        """
        if pose is None or self.pose is None:
            return pose
        jump = math.hypot(pose.x - self.pose.x, pose.y - self.pose.y)
        turn = abs(math.degrees(wrap_rad(pose.yaw - self.pose.yaw)))
        if self.lane_check and jump > 0.10 and self.clearance(self.pose.x, self.pose.y, self.pose.yaw) < 0.06:
            print(f"  fix {pose.text()} jumped {jump * 100:.0f}cm while close to a wall; sweeping", flush=True)
            self.sweep_correct()
            return None
        if jump > self.args.max_jump_m or turn > self.args.max_jump_deg:
            strong = len({(f.camera, f.tag_id) for f in pose.fixes}) >= 2 or \
                any(f.distance < self.localizer.max_single_tag_m for f in pose.fixes) or \
                (not self.pose.fixes and len(pose.fixes) >= 1 and pose.rms_px <= 6.0)
            if not strong:
                last = self.jump_pose
                same = last is not None and math.hypot(pose.x - last.x, pose.y - last.y) < self.args.max_jump_m
                self.jumps = self.jumps + 1 if same else 1
                self.jump_pose = pose
                if self.parking or self.jumps < 4:
                    print(f"  ignoring fix {pose.text()}: jumped {jump * 100:.0f}cm / {turn:.0f}deg "
                          f"from the expected pose", flush=True)
                    return None
                print("  far tags keep agreeing on a new pose; trusting them", flush=True)
        self.jumps, self.jump_pose = 0, None
        # If only a single far tag (> 85 cm) is seen and it jumps position by > 10 cm, blend position
        # so a single far edge-on tag doesn't teleport the car 14 cm during an in-place turn.
        uniq_tags = {(f.camera, f.tag_id) for f in pose.fixes}
        min_d = min((f.distance for f in pose.fixes), default=0.0)
        if len(uniq_tags) == 1 and min_d > 0.85 and jump > 0.10:
            pose.x = 0.4 * pose.x + 0.6 * self.pose.x
            pose.y = 0.4 * pose.y + 0.6 * self.pose.y
        if self.lane_check and self.lane_yaw is not None \
                and abs(wrap_rad(pose.yaw - self.lane_yaw)) > math.radians(25.0):
            # Far down a lane the tags still locate the car well but the heading flips; keep the
            # position and the lane heading instead of steering off at the bad yaw.
            print(f"  fix {pose.text()}: heading {abs(math.degrees(wrap_rad(pose.yaw - self.lane_yaw))):.0f}deg "
                  f"off the lane; keeping its position, not its heading", flush=True)
            pose.yaw = self.pose.yaw
            return pose
        return pose

    # ---- motion ---------------------------------------------------------

    def _forward_pwm(self) -> tuple[int, int, int, int]:
        wheels = self.motion.get("forward_wheel_pwm")
        if wheels and len(wheels) == 4:
            return tuple(int(v) for v in wheels)
        fwd = int(self.motion["forward_pwm"])
        return (fwd, fwd, fwd, fwd)

    def _pwm(self, kind: str):
        fwd = self._forward_pwm()
        turn = int(self.motion["turn_pwm"])
        side = int(self.motion.get("strafe_pwm", 1200)) * (-1 if self.strafe_flip else 1)
        return {
            "forward": fwd,
            "reverse": tuple(-v for v in fwd),
            "left": (-turn, -turn, turn, turn),
            "right": (turn, turn, -turn, -turn),
            "strafe_left": (-side, side, side, -side),
            "strafe_right": (side, -side, -side, side),
        }[kind]

    def move(self, kind: str, duration: float) -> tuple[float, str | None]:
        """Timed motion; forward segments stop early on a safety reason."""
        if self.motor is None:
            print(f"    would {kind} {duration:.2f}s", flush=True)
            if self.stop_event.wait(0.5):
                raise Abort("stopped")
            return 0.0, None
        reason = None
        started = time.monotonic()
        self.move_started, self.move_kind = started, kind
        self.drive(self._pwm(kind))
        try:
            while not self.stop_event.is_set():
                elapsed = time.monotonic() - started
                if elapsed >= duration:
                    break
                if kind in ("forward", "reverse", "strafe_left", "strafe_right"):
                    reason = self.safety_reason()
                    if reason:
                        break
                time.sleep(0.04)
        finally:
            self.move_started, self.move_kind = None, None
        self.stop()
        elapsed = time.monotonic() - started
        if self.stop_event.is_set():
            raise Abort("stopped")
        self.dead_reckon(kind, elapsed)
        return elapsed, reason

    def dead_reckon(self, kind: str, elapsed: float):
        if self.pose is None or elapsed <= 0.0:
            return
        if kind in ("forward", "reverse"):
            step = float(self.motion["forward_mps"]) * elapsed * (1.0 if kind == "forward" else -1.0)
            # The car drifts left while driving straight; fold that into the expected heading.
            self.pose.yaw = wrap_rad(
                self.pose.yaw + math.radians(float(self.motion.get("drift_left_deg_per_m", 0.0))) * step)
            self.pose.x += step * math.cos(self.pose.yaw)
            self.pose.y += step * math.sin(self.pose.yaw)
        elif kind in ("strafe_left", "strafe_right"):
            side = self.strafe_mps * elapsed * (1.0 if kind == "strafe_left" else -1.0)
            self.pose.x -= side * math.sin(self.pose.yaw)
            self.pose.y += side * math.cos(self.pose.yaw)
        elif kind == "left":
            self.pose.yaw = wrap_rad(self.pose.yaw + math.radians(float(self.motion["turn_left_dps"]) * elapsed))
        elif kind == "right":
            self.pose.yaw = wrap_rad(self.pose.yaw - math.radians(float(self.motion["turn_right_dps"]) * elapsed))
        w, h = (float(v) for v in self.arena["arena_size_m"])
        hl, hw = self.body_half
        self.pose.x = min(max(self.pose.x, hl), w - hl)
        self.pose.y = min(max(self.pose.y, hw), h - hw)
        self.kalman.follow_motion(self.pose.x, self.pose.y, self.pose.yaw)
        self.pose.x, self.pose.y, self.pose.yaw = (float(v) for v in self.kalman.x)
        self.pose.fixes = []

    # ---- wall geometry ------------------------------------------------------

    def clearance(self, x: float, y: float, yaw: float) -> float:
        """Smallest gap between a body corner and an arena wall (negative = through it)."""
        w, h = (float(v) for v in self.arena["arena_size_m"])
        hl, hw = self.body_half
        c, s = math.cos(yaw), math.sin(yaw)
        gap = float("inf")
        for lx, ly in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
            cx, cy = x + lx * c - ly * s, y + lx * s + ly * c
            gap = min(gap, cx, w - cx, cy, h - cy)
        return gap

    def sweep_clearance(self, x: float, y: float, yaw: float, turn: float = 0.0, step: float = 0.0,
                        side: float = 0.0) -> float:
        """Clearance over a turn in place by `turn` rad, a straight move by `step` m or a strafe by `side` m."""
        n = max(2, int(abs(math.degrees(turn)) / 5.0) + 1, int(math.hypot(step, side) / 0.02) + 1)
        dx = step * math.cos(yaw) - side * math.sin(yaw)
        dy = step * math.sin(yaw) + side * math.cos(yaw)
        return min(
            self.clearance(x + dx * k / (n - 1), y + dy * k / (n - 1), yaw + turn * k / (n - 1))
            for k in range(n)
        )

    def wall_limit(self) -> float:
        """Required corner gap: the margin, or (already inside it) no closer than now."""
        p = self.pose
        limit = min(self.args.wall_margin_m, self.clearance(p.x, p.y, p.yaw) - 0.005)
        return limit if self.allow_gap is None else min(limit, self.allow_gap)

    def safe_step(self, step: float, sideways: bool = False) -> float:
        """Shorten a straight move (or strafe) so no body corner gets closer than the margin to a wall."""
        p, margin = self.pose, self.wall_limit()

        drift = math.radians(3.0)

        def gap(d):
            if sideways:
                return self.sweep_clearance(p.x, p.y, p.yaw, side=d)
            return min(self.sweep_clearance(p.x, p.y, p.yaw + e, step=d) for e in (0.0, drift, -drift))

        if gap(step) >= margin:
            return step
        lo, hi = 0.0, step
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if gap(mid) >= margin:
                lo = mid
            else:
                hi = mid
        return lo

    def turn_swing_extra_m(self, turn_rad: float) -> float:
        """Extra lateral clearance (beyond body half-width `hw`) needed by the 4 body corners when rotating by `turn_rad`."""
        hl, hw = self.body_half
        ang = abs(turn_rad)
        if ang <= 1e-4:
            return 0.0
        n = max(4, int(math.degrees(ang) / 4.0) + 1)
        max_y = hw
        for k in range(1, n):
            a = ang * k / (n - 1)
            ca, sa = math.cos(a), math.sin(a)
            for lx, ly in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
                max_y = max(max_y, abs(lx * sa + ly * ca))
        return max(0.0, max_y - hw)

    def side_zone_gaps(self, sign: float, sonic: bool) -> tuple[float | None, float | None]:
        """(front-half gap, rear-half gap) in metres outside the body on side `sign` (+1 = left).

        The ultrasonic sits at the nose, so its ±65°/±90° readings only describe the front half;
        the side camera and the wheel IR modules cover both halves. None = nothing measured there.
        """
        hl, hw = self.body_half
        name = "left" if sign > 0 else "right"
        front, rear = [], []
        if sonic and self.servo is not None:
            cm_90 = self.look(90.0 * sign)
            cm_65 = self.look(65.0 * sign)
            self.center_sonic()
            if cm_90 is not None:
                front.append((cm_90 + 4.0) / 100.0 - hw)
            if cm_65 is not None:
                front.append((cm_65 * math.sin(math.radians(65.0)) + 3.5) / 100.0 - hw)
        grabber = self.grabbers.get(name)
        scanner = self.side_floors.get(name)
        if grabber is not None and scanner is not None:
            frame, _ = grabber.latest()
            if frame is not None:
                pts = scanner.scan(frame)
                if len(pts):
                    f_pts = pts[(pts[:, 0] >= 0.0) & (pts[:, 0] <= hl + 0.10)]
                    r_pts = pts[(pts[:, 0] < 0.0) & (pts[:, 0] >= -hl - 0.06)]
                    if len(f_pts):
                        front.append(float(np.abs(f_pts[:, 1]).min()) - hw)
                    if len(r_pts):
                        rear.append(float(np.abs(r_pts[:, 1]).min()) - hw)
        if self.sensors is not None:
            ir = self.sensors.read_ir()
            if ir.get(f"front_{name}") or ir.get(f"wheel_{name}_front"):
                front.append(0.0)
            if ir.get(f"wheel_{name}_rear"):
                rear.append(0.0)
        return (min(front) if front else None), (min(rear) if rear else None)

    def ensure_turn_side_clearance(self, turn_rad: float):
        """Before rotating in place, make room for the two body corners that swing outward.

        A left turn swings the front-left and rear-right corners out (the other two tuck in);
        a right turn swings the front-right and rear-left corners out. The car only strafes
        away from a tight swinging corner when the ultrasonic has measured real room on the
        side it strafes toward.
        """
        if abs(math.degrees(turn_rad)) < 14.0 or self.motor is None:
            return
        s = 1.0 if turn_rad > 0 else -1.0
        need = min(max(self.turn_swing_extra_m(turn_rad) + 0.03, 0.05), 0.10)
        front_s, _ = self.side_zone_gaps(s, sonic=False)
        _, rear_o = self.side_zone_gaps(-s, sonic=False)
        front_tight = front_s is not None and front_s < need
        rear_tight = rear_o is not None and rear_o < need
        if not front_tight and not rear_tight:
            return
        turn_name = "left" if s > 0 else "right"
        if front_tight and rear_tight:
            print(f"    turn clearance: front-{turn_name} {front_s * 100:.0f}cm and rear-"
                  f"{'right' if s > 0 else 'left'} {rear_o * 100:.0f}cm both tight; not strafing", flush=True)
            return
        # Strafe toward `toward`, away from the tight corner.
        if front_tight:
            toward, tight_gap, what = -s, front_s, f"front-{turn_name}"
        else:
            toward, tight_gap, what = s, rear_o, f"rear-{'right' if s > 0 else 'left'}"
        room_front, room_rear = self.side_zone_gaps(toward, sonic=True)
        if room_front is None:
            print(f"    turn clearance: {what} {tight_gap * 100:.0f}cm, but no ultrasonic reading on the "
                  f"{'left' if toward > 0 else 'right'}; not strafing", flush=True)
            return
        room = min(g for g in (room_front, room_rear) if g is not None)
        push = min(need - tight_gap + 0.02, 0.08, room - need)
        push = min(push, abs(self.safe_step(toward * push, sideways=True))) if push > 0 else 0.0
        side_name = "left" if toward > 0 else "right"
        if push < 0.02:
            print(f"    turn clearance: {what} {tight_gap * 100:.0f}cm, {side_name} side only "
                  f"{room * 100:.0f}cm; not strafing", flush=True)
            return
        print(f"    turn clearance: {what} {tight_gap * 100:.0f}cm < {need * 100:.0f}cm; "
              f"strafing {side_name} {push * 100:.0f}cm ({side_name} room {room * 100:.0f}cm)", flush=True)
        self.move("strafe_left" if toward > 0 else "strafe_right", push / self.strafe_mps)

    def side_sees_foam(self, sign: float) -> bool:
        """That side's camera is looking at foam, even when it is still a body-width away."""
        name = "left" if sign > 0 else "right"
        grabber = self.grabbers.get(name)
        scanner = self.side_floors.get(name)
        if grabber is None or scanner is None:
            return False
        frame = grabber.latest()[0]
        return frame is not None and scanner.sees_foam(frame)

    def side_blocked(self, sign: float) -> bool:
        """The body is already against foam on this side. A 90° beam can miss that block."""
        name = "left" if sign > 0 else "right"
        if self.sensors is not None:
            ir = self.sensors.read_ir()
            keys = (("front_left", "wheel_left_front", "wheel_left_rear") if sign > 0
                    else ("front_right", "wheel_right_front", "wheel_right_rear"))
            if any(ir.get(key) for key in keys):
                return True
        grabber = self.grabbers.get(name)
        scanner = self.side_floors.get(name)
        frame = grabber.latest()[0] if grabber is not None else None
        return (frame is not None and scanner is not None and scanner.foam_against_body(frame))

    def turn_by(self, err: float, min_s: float | None = None, shuffle: bool = True) -> bool:
        """Turn in place. `shuffle` may strafe or creep first so a corner clears a wall.

        The ultrasonic exit turn passes shuffle=False: the open direction was already
        measured, and a sideways shuffle there drives into the foam beside the car.
        Returns False when the turn would swing the front corner into foam already
        filling that side's camera.
        """
        p, margin = self.pose, self.wall_limit()
        kind = "left" if err > 0 else "right"
        if (shuffle and abs(err) > math.radians(self.args.align_deg)
                and self.side_blocked(1.0 if err > 0 else -1.0)):
            print(f"    not turning {kind}; the {kind} side is already against the foam",
                  flush=True)
            return False
        rate = float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"])
        min_s = self.args.min_turn_s if min_s is None else min_s
        duration = min(max(math.degrees(abs(err)) / rate, min_s), self.args.max_turn_s)
        turn = math.copysign(math.radians(rate * duration), err)
        if not shuffle:
            print(f"    turning {kind} {math.degrees(abs(turn)):.0f}deg toward the exit, without strafing",
                  flush=True)
            self.move(kind, duration)
            return True
        # Check live left/right obstacle clearance before turning in place
        self.ensure_turn_side_clearance(turn)
        p, margin = self.pose, self.wall_limit()
        steps = [s * k * 0.02 for k in range(1, 11) for s in (1, -1)]
        shifts = [(0.0, 0.0)] + [(d, 0.0) for d in steps] + [(0.0, d) for d in steps if abs(d) <= 0.10]
        reachable = [s for s in shifts
                     if s == (0.0, 0.0) or self.sweep_clearance(p.x, p.y, p.yaw, step=s[0], side=s[1]) >= margin]
        reachable.sort(key=lambda s: abs(s[0]) + abs(s[1]))

        def turn_gap(s, angle):
            c, sn = math.cos(p.yaw), math.sin(p.yaw)
            return self.sweep_clearance(p.x + s[0] * c - s[1] * sn, p.y + s[0] * sn + s[1] * c, p.yaw, turn=angle)

        shift = next((s for s in reachable if turn_gap(s, turn) >= margin), None)
        if shift is None:
            # No spot allows the whole turn: take the biggest safe part and re-localize.
            best = None
            for s in reachable:
                for frac in (0.75, 0.5, 0.35, 0.2):
                    if math.degrees(abs(turn) * frac) >= 2.0 and turn_gap(s, turn * frac) >= margin:
                        if best is None or frac > best[1]:
                            best = (s, frac)
                        break
            if best is None:
                gap = self.sweep_clearance(p.x, p.y, p.yaw, turn=turn)
                raise Abort(f"turning here would hit a wall (corner gap {gap * 100:.0f}cm); move the car inward")
            shift, frac = best
            print(f"    only {math.degrees(abs(turn) * frac):.0f}deg of the turn is safe here", flush=True)
            duration *= frac
        step, side = shift
        if step:
            print(f"    wall too close to turn; {'forward' if step > 0 else 'reverse'} {abs(step) * 100:.0f}cm first",
                  flush=True)
            self.move("forward" if step > 0 else "reverse", abs(step) / float(self.motion["forward_mps"]))
        elif side:
            if self.side_blocked(1.0 if side > 0 else -1.0):
                print(f"    not strafing {'left' if side > 0 else 'right'}; that side is against the foam",
                      flush=True)
                side = 0.0
            else:
                side = self.cap_strafe(side)
            if abs(side) >= 0.02:
                print(f"    wall too close to turn; strafe {'left' if side > 0 else 'right'} {abs(side) * 100:.0f}cm first",
                      flush=True)
                self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
        self.move(kind, duration)
        return True

    def search(self) -> Pose2D | None:
        self.lane_check = False
        steps = int(math.ceil(360.0 / self.args.search_step_deg))
        for i in range(steps):
            print(f"  search {i + 1}/{steps}: turning left {self.args.search_step_deg:.0f}deg", flush=True)
            if self.pose is None:
                self.move("left", self.args.search_step_deg / float(self.motion["turn_left_dps"]))
            else:
                self.turn_by(math.radians(self.args.search_step_deg))
            fix = self.localize()
            if fix is not None:
                return fix
        return None

    # ---- obstacles ------------------------------------------------------

    def avoid_active(self) -> bool:
        return not self.args.no_avoid and not self.parking and self.motor is not None and self.pose is not None

    def wall_range_cm(self, pan_deg: float = 0.0, travelled: float = 0.0) -> float:
        """Range from the ultrasonic to the nearest mapped wall along a beam `pan_deg` off the heading."""
        p = self.pose
        reach = travelled + self.args.sonic_x_m
        return 100.0 * self.ray_to_wall(p.x + reach * math.cos(p.yaw), p.y + reach * math.sin(p.yaw),
                                        p.yaw + math.radians(pan_deg))

    def ray_to_wall(self, ox: float, oy: float, angle: float) -> float:
        w, h = (float(v) for v in self.arena["arena_size_m"])
        dx, dy = math.cos(angle), math.sin(angle)
        hits = []
        if dx > 1e-6:
            hits.append((w - ox) / dx)
        if dx < -1e-6:
            hits.append(-ox / dx)
        if dy > 1e-6:
            hits.append((h - oy) / dy)
        if dy < -1e-6:
            hits.append(-oy / dy)
        return max(0.0, min(hits))

    def is_obstacle(self, cm: float | None, pan_deg: float = 0.0, travelled: float = 0.0,
                    limit_cm: float | None = None) -> bool:
        """Something in the beam that is clearly closer than the arena wall it should hit."""
        limit_cm = self.args.avoid_cm if limit_cm is None else limit_cm
        if cm is None or self.pose is None or cm > limit_cm:
            return False
        return cm < self.wall_range_cm(pan_deg, travelled) - self.args.avoid_margin_cm

    def side_gap_m(self, sign: float, fast_only: bool = False) -> float | None:
        """Metres outside the body on this side.

        Fuses 3 sources so side obstacles are never missed:
        1) Ultrasonic at ±90° AND ±65° (covers both nose-side and mid-body; skipped when `fast_only=True`).
        2) Left/Right USB camera floor scan (`side_floors["left" / "right"]`) covering the entire 29 cm body length.
        3) Side infrared modules (front_left/right, wheel_left/right_front, wheel_left/right_rear) inside 8 cm.
        """
        name = "left" if sign > 0 else "right"
        cm_90, cm_65 = None, None
        if not fast_only:
            cm_90 = self.look(90.0 * sign)
            cm_65 = self.look(65.0 * sign)
            self.center_sonic()
        gaps = []
        if cm_90 is not None:
            g90 = (cm_90 + 4.0) / 100.0 - self.body_half[1]
            gaps.append(g90)
        if cm_65 is not None:
            # Projected lateral distance at 65 deg
            g65 = (cm_65 * math.sin(math.radians(65.0)) + 3.5) / 100.0 - self.body_half[1]
            gaps.append(g65)
        # Check side USB camera floor scan for any foam block along the car body (-18 cm .. +22 cm)
        cam_gap = None
        grabber = self.grabbers.get(name)
        scanner = self.side_floors.get(name)
        if grabber is not None and scanner is not None:
            frame, _ = grabber.latest()
            if frame is not None:
                s_pts = scanner.scan(frame)
                if len(s_pts):
                    along_body = s_pts[(s_pts[:, 0] >= -self.body_half[0] - 0.04) & (s_pts[:, 0] <= self.body_half[0] + 0.12)]
                    use_pts = along_body if len(along_body) else s_pts
                    cam_gap = float(np.abs(use_pts[:, 1]).min()) - self.body_half[1]
                    sonic_gap = min(gaps) if gaps else None
                    if sonic_gap is not None and sonic_gap > cam_gap + 0.20 and cam_gap < 0.15:
                        print(f"    {name} camera {cam_gap * 100:.0f}cm ignored; ultrasonic is "
                              f"{sonic_gap * 100:.0f}cm clear", flush=True)
                        cam_gap = None
                    elif cam_gap is not None:
                        gaps.append(cam_gap)
        gap = min(gaps) if gaps else None
        if gap is not None and not fast_only:
            cam_txt = f", cam={cam_gap * 100:.0f}cm" if cam_gap is not None else ""
            s65_txt = f", 65deg={cm_65:.0f}cm" if cm_65 is not None else ""
            print(f"    {name} 90deg={'-' if cm_90 is None else f'{cm_90:.0f}'}cm{s65_txt}{cam_txt} -> "
                  f"{gap * 100:.0f}cm outside the body", flush=True)
        if self.sensors is not None:
            ir = self.sensors.read_ir()
            keys = (("front_left", "wheel_left_front", "wheel_left_rear") if sign > 0
                    else ("front_right", "wheel_right_front", "wheel_right_rear"))
            hit = [key for key in keys if ir.get(key)]
            if hit:
                if not fast_only:
                    print(f"    {name} IR on ({', '.join(hit)}), under 8cm", flush=True)
                gap = 0.0 if gap is None else min(gap, 0.0)
        return gap

    def cap_strafe(self, side: float) -> float:
        """Shorten a strafe so it stops 5 cm before whatever the 90° ultrasonic sees."""
        gap = self.side_gap_m(1.0 if side > 0 else -1.0)
        if gap is None:
            return side
        allowed = max(gap - 0.05, 0.0)
        name = "left" if side > 0 else "right"
        if allowed < 0.02:
            print(f"    strafe {name} skipped ({gap * 100:.0f}cm outside the body)", flush=True)
            return 0.0
        if abs(side) > allowed:
            print(f"    strafe {name} cut from {abs(side) * 100:.0f}cm to {allowed * 100:.0f}cm", flush=True)
            return math.copysign(allowed, side)
        return side

    def side_room(self, sign: float) -> float:
        """Strafe room toward `sign` (+1 = left) before the body side reaches the wall margin.

        Only the sideways wall counts, so a dead-reckoned pose that is off along the lane cannot
        open up room that is not there.
        """
        p = self.pose
        hl, hw = self.body_half
        angle = p.yaw + sign * math.pi / 2
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        reach = min(self.ray_to_wall(p.x + lx * c, p.y + lx * s, angle) for lx in (hl, -hl))
        return reach - hw - self.args.wall_margin_m

    def obstacle_extent(self, sign: float) -> float:
        """How far (m from the car centreline) the obstacle reaches toward `sign`, from servo readings."""
        widest = 0.0
        for angle in (25, 45, 70, 90):
            pan = angle * sign
            cm = self.look(pan)
            if self.is_obstacle(cm, pan, limit_cm=60.0):
                widest = max(widest, cm / 100.0 * math.sin(math.radians(angle)))
        return widest

    def note_range(self, pan_deg: float, cm: float | None) -> None:
        """Add one ultrasonic reading to the live map. A wall return is clear space, not an obstacle."""
        if self.pose is None or cm is None:
            return
        if self.kalman.update_wall(self.args.sonic_x_m, math.radians(pan_deg), cm / 100.0):
            self.pose.x, self.pose.y, self.pose.yaw = (float(v) for v in self.kalman.x)
        reach = self.args.sonic_x_m
        ox = self.pose.x + reach * math.cos(self.pose.yaw)
        oy = self.pose.y + reach * math.sin(self.pose.yaw)
        self.live_map.mark_ray(ox, oy, self.pose.yaw + math.radians(pan_deg), cm / 100.0)

    def absorb_views(self) -> None:
        """Paint blue-foam floor contact from the cameras that already have a frame."""
        if self.pose is None:
            return
        chunks = []
        front = self.grabbers.get("front")
        frame = front.latest()[0] if front is not None else None
        if frame is not None:
            pts = self.floor.scan(frame)
            if len(pts):
                chunks.append(pts)
        for name, scanner in self.side_floors.items():
            grabber = self.grabbers.get(name)
            frame = grabber.latest()[0] if grabber is not None else None
            if frame is None:
                continue
            pts = scanner.scan(frame)
            if len(pts):
                chunks.append(pts)
        if chunks:
            pts = np.vstack(chunks)
            reach = np.hypot(pts[:, 0], pts[:, 1])
            hl, hw = self.body_half
            on_body = (np.abs(pts[:, 0]) < hl) & (np.abs(pts[:, 1]) < hw)
            near = pts[(reach > 0.12) & (reach < 0.55) & ~on_body]
            if len(near):
                self.live_map.mark_points(to_world(near, self.pose.x, self.pose.y, self.pose.yaw))
        self.confirm_side_foam()

    def confirm_side_foam(self) -> None:
        """A close ultrasonic hit is an obstacle when that side's camera is full of foam.

        The floor scan misses pale blue, so a block filling the left or right frame never
        becomes a floor point. The ultrasonic range is still where the block is.
        """
        if self.pose is None:
            return
        fresh: list[tuple[float, float]] = []
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        for name, sign in (("left", 1.0), ("right", -1.0)):
            grabber = self.grabbers.get(name)
            scanner = self.side_floors.get(name)
            if grabber is None or scanner is None:
                continue
            frame = grabber.latest()[0]
            if frame is None or not scanner.sees_foam(frame):
                continue
            beside = []
            for hx, hy in self.live_map.sonic_hits:
                dx, dy = hx - self.pose.x, hy - self.pose.y
                local_x = dx * c + dy * s
                local_y = -dx * s + dy * c
                if local_y * sign < 0.08 or abs(local_x) > 0.22 or math.hypot(dx, dy) > 0.45:
                    continue
                beside.append((math.hypot(dx, dy), hx, hy))
            if not beside:
                continue
            _reach, hx, hy = min(beside)
            fresh.append((hx, hy))
            print(f"    {name} camera sees foam; ultrasonic obstacle at ({hx:.2f},{hy:.2f})", flush=True)
        self.live_map.confirmed_hits = fresh

    def center_sonic(self):
        """Point the ultrasonic straight ahead again and let the servo settle before driving."""
        if self.servo is None:
            return
        self.pan_to(self.home_pan)
        if self.stop_event.wait(0.3):
            raise Abort("stopped")

    def obstacle_points(self, fresh: bool = False) -> np.ndarray:
        """Car-frame floor points of blue foam standing clearly inside the arena (not a wall)."""
        none = np.zeros((0, 2))
        grabber = self.grabbers.get("front")
        if grabber is None or self.pose is None:
            return none
        after = time.monotonic() + (self.args.settle_s if fresh else -1.0)
        if fresh and self.stop_event.wait(self.args.settle_s):
            raise Abort("stopped")
        deadline = time.monotonic() + 1.0
        frame, stamp = grabber.latest()
        while (frame is None or stamp <= after) and time.monotonic() < deadline:
            time.sleep(0.03)
            frame, stamp = grabber.latest()
        if frame is None or stamp <= after:
            return none
        pts = self.floor.scan(frame)
        if not len(pts):
            return pts
        p = self.pose
        world = to_world(pts, p.x, p.y, p.yaw)
        w, h = (float(v) for v in self.arena["arena_size_m"])
        inner = np.minimum.reduce([world[:, 0], w - world[:, 0], world[:, 1], h - world[:, 1]])
        reach = np.hypot(pts[:, 0], pts[:, 1])
        # Wall feet land near the map wall; how near depends on the pose error, which grows with range.
        keep = inner > self.args.wall_band_m + 0.04 * reach
        kept_pts = pts[keep]
        # Only cache near-field world obstacle points (< 55 cm from car) when pose is backed by ArUco tag fixes;
        # far fisheye floor points (> 55 cm) have large projection error and would falsely taint distant route segments.
        near_keep = keep & (reach < 0.55)
        if near_keep.any() and self.pose.fixes:
            self.world_obstacles.append(world[near_keep])
            if len(self.world_obstacles) > 12:
                self.world_obstacles = self.world_obstacles[-12:]
            self.live_map.mark_points(world[near_keep])
        return kept_pts

    def known_world_obstacles(self) -> np.ndarray:
        if not self.world_obstacles:
            return np.zeros((0, 2))
        return np.vstack(self.world_obstacles)

    def obstacle_box(self, pts: np.ndarray, reach: float):
        """(near x, far x, right y, left y) in the car frame of the obstacle in the car's path, or None."""
        band = self.body_half[1] + self.args.avoid_gap_m
        hit = (pts[:, 0] > 0) & (pts[:, 0] < reach) & (np.abs(pts[:, 1]) < band)
        if not hit.any():
            return None
        member = hit
        for _ in range(len(pts)):
            # One foam block's floor line is uneven in the fisheye, so points 25cm apart still belong together.
            gap = np.hypot(pts[:, None, 0] - pts[None, member, 0], pts[:, None, 1] - pts[None, member, 1]).min(axis=1)
            grown = gap < 0.25
            if grown.sum() == member.sum():
                break
            member = grown
        c = pts[member]
        return float(c[:, 0].min()), float(c[:, 0].max()), float(c[:, 1].min()), float(c[:, 1].max())

    def look_reach(self) -> float:
        reach = self.args.avoid_look_m
        if self.target_ahead is not None:
            reach = min(reach, max(self.target_ahead, 0.0) + 0.10)
        return reach

    def confirm_sonic(self, box) -> float | None:
        """Point the ultrasonic at the obstacle the camera found and report both ranges."""
        x0, _x1, y0, y1 = box
        mid = 0.5 * (y0 + y1)
        pan = max(-60.0, min(60.0, math.degrees(math.atan2(mid, x0 - self.args.sonic_x_m))))
        cm = self.look(pan)
        camera_cm = 100.0 * math.hypot(x0 - self.args.sonic_x_m, mid)
        print(f"    servo {pan:+.0f}deg: sonic {'-' if cm is None else f'{cm:.0f}'}cm, "
              f"camera {camera_cm:.0f}cm", flush=True)
        return cm

    def choose_side(self, pts: np.ndarray, box, only: float | None = None):
        """(sign, strafe m) that puts the body `avoid_gap_m` clear of the obstacle edge, or None."""
        x0, x1, y0, y1 = box
        hl, hw = self.body_half
        gap = self.args.avoid_gap_m
        p = self.pose
        w, h = (float(v) for v in self.arena["arena_size_m"])
        toward_centre = -(w / 2 - p.x) * math.sin(p.yaw) + (h / 2 - p.y) * math.cos(p.yaw)
        best = None
        for sign in (1.0, -1.0) if only is None else (only,):
            name = "left" if sign > 0 else "right"
            need = max(0.03, (y1 + hw + gap) if sign > 0 else (hw + gap - y0))
            room = min(self.side_room(sign), self.args.avoid_max_side_m)
            measured = self.side_gap_m(sign)
            if measured is not None:
                room = min(room, max(measured - 0.05, 0.0))
            lo, hi = sign * need - hw - gap, sign * need + hw + gap
            others = pts[(pts[:, 0] > -hl) & (pts[:, 0] < x1 + self.args.avoid_depth_m + 2 * hl)
                         & (pts[:, 1] > lo) & (pts[:, 1] < hi)]
            ok = need <= room and not len(others)
            print(f"    {name}: strafe {need * 100:.0f}cm, wall room {max(room, 0.0) * 100:.0f}cm"
                  f"{f', {len(others)} other obstacle points' if len(others) else ''}"
                  f"{'' if ok else ' -> no'}", flush=True)
            if ok:
                cost = need + (0.0 if sign * toward_centre >= 0 else 0.03)
                if best is None or cost < best[0]:
                    best = (cost, sign, need)
        return None if best is None else best[1:]

    def handle_probe(self, prefer_deg: float = 0.0) -> bool:
        """True when a camera-seen obstacle was dealt with instead of driving this step."""
        if not self.avoid_active():
            return False
        box = self.obstacle_box(self.obstacle_points(), self.look_reach())
        if box is None:
            return False
        print(f"    camera: obstacle {box[0] * 100:.0f}cm ahead, from {box[2] * 100:+.0f} to "
              f"{box[3] * 100:+.0f}cm sideways", flush=True)
        if not self.bypass(prefer_deg=prefer_deg):
            self.wait_clear("obstacle")
        return True

    def bypass(self, depth: int = 0, prefer_deg: float = 0.0) -> bool:
        """Strafe until the front camera sees the path clear of the obstacle, then drive past it.

        The camera gives the obstacle's near face and both edges, the servo-pointed ultrasonic
        confirms the range; obstacles the camera cannot see fall back to the ultrasonic detour.
        """
        if not self.avoid_active():
            return False
        hl, hw = self.body_half
        try:
            sign, last = None, None
            for _attempt in range(4):
                pts = self.obstacle_points(fresh=True)
                box = self.obstacle_box(pts, self.look_reach())
                if box is None:
                    break
                last = box
                x0, x1, y0, y1 = box
                print(f"    camera: obstacle {x0 * 100:.0f}-{x1 * 100:.0f}cm ahead, "
                      f"{y0 * 100:+.0f}..{y1 * 100:+.0f}cm sideways", flush=True)
                if self.servo is not None:
                    self.confirm_sonic(box)
                    self.center_sonic()
                if x0 < hl + 0.04:
                    back = -self.safe_step(-(hl + 0.10 - x0))
                    rear_ir = self.sensors.read_ir() if self.sensors is not None else {}
                    if back >= 0.02 and not any(rear_ir.get(k) for k in ("rear_left", "rear_mid", "rear_right")):
                        print(f"    backing up {back * 100:.0f}cm to strafe clear", flush=True)
                        self.move("reverse", back / float(self.motion["forward_mps"]))
                        continue
                    print("    no room behind to back up; escaping via side strafe or angled corridor", flush=True)
                choice = self.choose_side(pts, box, only=sign)
                # Prefer rotating into an open corridor whenever the required strafe is large (> 22 cm),
                # because wide sideways strafes in a cluttered arena easily clip obstacles beside the body!
                if choice is None or choice[1] > 0.22:
                    if choice is not None:
                        print(f"    strafe {choice[1] * 100:.0f}cm is too wide (>22cm); preferring angled corridor turn",
                              flush=True)
                    else:
                        print("    no room to strafe around; checking angled ultrasonic corridor", flush=True)
                    if self.bypass_corridor(box, prefer_deg=prefer_deg):
                        return True
                    if choice is None:
                        return False
                sign, need = choice
                print(f"    strafing {'left' if sign > 0 else 'right'} {need * 100:.0f}cm", flush=True)
                self.move("strafe_left" if sign > 0 else "strafe_right", need / self.strafe_mps)
            else:
                print("    camera still sees the obstacle in the path after 4 strafes", flush=True)
                return False
            if last is None:
                # Something stopped the car, but no blue foam in the path: try the ultrasonic detour.
                return self.bypass_sonic() if depth == 0 else True
            speed = float(self.motion["forward_mps"])
            length = last[1] + self.args.avoid_depth_m + hl
            print(f"    path clear; driving {length * 100:.0f}cm past the obstacle", flush=True)
            passed = 0.0
            while passed < length - 0.02:
                step = self.safe_step(min(0.12, length - passed))
                if step < 0.02:
                    break
                elapsed, reason = self.move("forward", step / speed)
                passed += speed * elapsed
                if reason:
                    print(f"    blocked while passing ({reason})", flush=True)
                    if reason.startswith("obstacle") and depth < 2:
                        return self.bypass(depth + 1, prefer_deg=prefer_deg)
                    return passed > 0.05
            return True
        finally:
            self.center_sonic()

    def find_open_corridor(self, prefer_deg: float = 0.0, max_off_deg: float = 45.0) -> tuple[int, float] | None:
        """Sweep ultrasonic across ±60 deg to find an open corridor (> 55 cm) biased toward `prefer_deg`."""
        if self.servo is None:
            return None
        angles = (30, 40, 20, 10, 0, -15, -30, -45, 50)
        readings: dict[int, float] = {}
        try:
            for angle in angles:
                cm = self.look(angle)
                if cm is None or cm < 55.0:
                    cm2 = self.look(angle)
                    if cm2 is not None and (cm is None or cm2 > cm):
                        cm = cm2
                print(f"    corridor check {angle:+d}deg: {'-' if cm is None else f'{cm:.0f}'}cm", flush=True)
                if cm is not None:
                    readings[angle] = cm
        finally:
            self.center_sonic()
        best = None
        best_score = -1e9
        candidates = [
            (angle, cm) for angle, cm in readings.items()
            if cm >= 55.0 and abs(float(angle) - prefer_deg) <= max_off_deg
        ]
        if not candidates:
            print(f"    no open corridor within {max_off_deg:.0f}deg of route direction {prefer_deg:+.0f}deg", flush=True)
            return None
        for angle, cm in candidates:
            left_nb = readings.get(angle + 10, readings.get(angle + 15, cm))
            right_nb = readings.get(angle - 10, readings.get(angle - 15, cm))
            # Penalize angles that turn away from the target waypoint (`prefer_deg`)
            ang_diff = abs(float(angle) - prefer_deg)
            score = (0.6 * min(cm, 120.0) + 0.2 * min(left_nb, 100.0)
                     + 0.2 * min(right_nb, 100.0) - 1.2 * ang_diff)
            if score > best_score:
                best_score = score
                best = (angle, cm)
        if best is not None:
            print(f"    corridor choice {best[0]:+d}deg ({best[1]:.0f}cm), "
                  f"route direction {prefer_deg:+.0f}deg", flush=True)
        return best

    def gimbal_gap(self, angle_deg: float, max_off_deg: float = 25.0, max_reach_m: float = 0.6):
        """Nearby gap the gimbal camera sees when panned to `angle_deg`, or None.

        A gap farther than `max_reach_m` is not the opening in front of the car, just one seen through it.

        The camera is opened only for this look: with the two side USB cameras already
        streaming, a third full-time stream risks dropping one of them.
        """
        if self.gimbal_cam is None or self.servo is None:
            return None
        from aruco_localizer import UsbGrabber
        from gimbal_gap import draw_frame

        grabber = UsbGrabber("gimbal").start()
        try:
            deadline = time.monotonic() + 6.0
            while grabber.latest()[0] is None and time.monotonic() < deadline:
                if self.stop_event.wait(0.1):
                    raise Abort("stopped")
            self.look(angle_deg)
            after = time.monotonic() + 0.15
            frame, stamp = grabber.latest()
            while (frame is None or stamp <= after) and time.monotonic() < deadline + 2.0:
                if self.stop_event.wait(0.05):
                    raise Abort("stopped")
                frame, stamp = grabber.latest()
            if frame is None or stamp <= after:
                print(f"    gimbal camera: no frame ({grabber.error or 'timeout'})", flush=True)
                return None
            cam = self.gimbal_cam
            cols, rows, pts, too_close = cam.floor_profile(frame, angle_deg)
            gaps = cam.find_gaps(cols, rows, pts, too_close, angle_deg, cam.floor_mask(frame))
            need = 2.0 * self.body_half[1] + 2.0 * self.args.gap_margin_m
            cv2.imwrite(str(self.debug_dir / "gimbal_gap.jpg"),
                        draw_frame(frame, cam, cols, rows, pts, gaps, angle_deg, need))
        finally:
            grabber.close()
        for g in gaps:
            far = float(np.hypot(*g.centre)) > max_reach_m
            print(f"    gimbal camera at {angle_deg:+.0f}deg: gap {g.straight_width_m * 100:.0f}cm, centre "
                  f"{float(np.hypot(*g.centre)) * 100:.0f}cm away at {g.bearing_deg:+.0f}deg"
                  f"{' (too far, ignored)' if far else ''}", flush=True)
        gaps = [g for g in gaps if abs(g.bearing_deg - angle_deg) <= max_off_deg
                and float(np.hypot(*g.centre)) <= max_reach_m]
        return min(gaps, key=lambda g: abs(g.bearing_deg - angle_deg)) if gaps else None

    def bypass_corridor(self, box, prefer_deg: float = 0.0) -> bool:
        """Escape a boxed-in corner by shifting into the pocket, turning into the open corridor, and driving out
        while actively steering/adjusting if an obstacle edge gets close."""
        hl, _hw = self.body_half
        speed = float(self.motion["forward_mps"])
        need = 2.0 * self.body_half[1] + 2.0 * self.args.gap_margin_m
        length = min(0.46, max(box[1] + self.args.avoid_depth_m, 0.34))
        clear_after = 0.24
        # 0) Already facing a gap wide enough: the front camera sees both edges as one obstacle, but the
        # gimbal camera can tell they leave room. Drive straight through without shifting or rescanning.
        gap = self.gimbal_gap(0.0, max_off_deg=12.0)
        if gap is not None and gap.straight_width_m >= need:
            print(f"    gimbal camera: already facing a {gap.straight_width_m * 100:.0f}cm gap "
                  f"(need {need * 100:.0f}cm); driving straight through", flush=True)
            return self.drive_corridor(int(round(gap.bearing_deg)), None, gap, length, clear_after)
        in_start_corner = self.pose is not None and self.pose.x < 0.34 and self.pose.y < 0.34
        if not in_start_corner:
            # Away from the start pocket, never follow an unverified ultrasonic opening for 46 cm.
            # Move one short leg, re-localize from tags, then recompute the direction to the waypoint.
            length = min(length, 0.20)
        # 1) Only do the blind (+9, +9) cm corner pocket shift when boxed in the start corner!
        if in_start_corner:
            sonic_0 = self.look(0)
            self.center_sonic()
            true_ahead_m = (sonic_0 / 100.0) if sonic_0 is not None else box[0]
            fwd_shift = min(0.09, max(true_ahead_m - hl - 0.08, 0.0))
            if fwd_shift >= 0.02:
                print(f"    pocket step 1a: forward {fwd_shift * 100:.0f}cm (sonic 0deg={sonic_0:.0f}cm)", flush=True)
                old_avoid = self.args.no_avoid
                if sonic_0 is not None and sonic_0 >= 28.0:
                    self.args.no_avoid = True
                try:
                    self.move("forward", fwd_shift / speed)
                finally:
                    self.args.no_avoid = old_avoid
            left_gap = self.side_gap_m(1.0)
            side_shift = 0.09 if left_gap is None else min(0.09, max(left_gap - 0.08, 0.0))
            if side_shift >= 0.02:
                print(f"    pocket step 1b: strafe left {side_shift * 100:.0f}cm", flush=True)
                self.move("strafe_left", side_shift / self.strafe_mps)
        # 2) Scan ultrasonic from current position. Outside the start pocket, stay within 20 deg
        # of the waypoint heading; a 50 deg opening is the wrong way even if it reads 300 cm.
        corridor_window = 45.0 if in_start_corner else 20.0
        corridor = self.find_open_corridor(prefer_deg=prefer_deg, max_off_deg=corridor_window)
        if corridor is None and in_start_corner:
            sonic_now = self.look(0)
            self.center_sonic()
            if sonic_now is not None and sonic_now >= 24.0:
                print(f"    corridor still occluded; stepping forward 6cm (sonic 0deg={sonic_now:.0f}cm) and rescanning",
                      flush=True)
                old_avoid = self.args.no_avoid
                self.args.no_avoid = True
                try:
                    self.move("forward", 0.06 / speed)
                finally:
                    self.args.no_avoid = old_avoid
                corridor = self.find_open_corridor(prefer_deg=prefer_deg, max_off_deg=corridor_window)
        if corridor is None:
            print("    no open angled corridor found", flush=True)
            return False
        angle, cm = corridor
        # Make room for the swinging corners first: the gap is measured from where the car turns.
        if abs(angle) >= 8:
            self.ensure_turn_side_clearance(math.radians(angle))
        gap = self.gimbal_gap(angle)
        if gap is not None:
            if gap.straight_width_m < need:
                print(f"    gimbal camera: gap toward {gap.bearing_deg:+.0f}deg is only {gap.straight_width_m * 100:.0f}cm "
                      f"wide (need {need * 100:.0f}cm); not squeezing through", flush=True)
                self.center_sonic()
                return False
            print(f"    gimbal camera: aiming at the gap centre ({gap.bearing_deg:+.0f}deg) "
                  f"instead of the ultrasonic {angle:+d}deg", flush=True)
            angle = int(round(gap.bearing_deg))
        elif self.gimbal_cam is not None:
            print(f"    gimbal camera: no nearby gap with two corners toward {angle:+d}deg; using the ultrasonic corridor",
                  flush=True)
        return self.drive_corridor(angle, cm, gap, length, clear_after)

    def drive_corridor(self, angle: int, cm: float | None, gap, length: float, clear_after: float) -> bool:
        """Turn by `angle` and drive out; with a measured `gap`, drive until the tail is through it."""
        hl, _hw = self.body_half
        speed = float(self.motion["forward_mps"])
        verified = gap is not None
        if verified:
            clear_after = float(np.hypot(*gap.centre)) + hl
            length = min(0.65, max(clear_after + 0.06, length))
        self.center_sonic()
        if abs(angle) >= 8:
            print(f"    pocket step 2: rotate {angle:+d}deg toward the {'gap centre' if verified else f'{cm:.0f}cm open corridor'}",
                  flush=True)
            kind = "left" if angle > 0 else "right"
            rate = float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"])
            dur = min(max(abs(angle) / rate, self.args.min_turn_s), self.args.max_turn_s)
            self.move(kind, dur)
        # 3) Drive out in 15 cm steps along the confirmed ultrasonic corridor.
        # Ignore false `obstacle_ir` triggers during the corridor escape unless ultrasonic confirms < 22 cm.
        print(f"    pocket step 3: drive {length * 100:.0f}cm out through the "
              f"{'measured gap' if verified else 'corridor (with live steering correction)'}", flush=True)
        passed = 0.0
        adjusts = 0
        while passed < length - 0.02:
            # 1) Always localize first so the car knows where it is on the map
            fix = self.localize(0.25)
            if fix is not None:
                self.pose = fix
            # 2) If we have cleared the corner pocket (x >= 0.48 m or y >= 0.32 m), stop driving into the center
            # and immediately return to `go_to()` so the car turns toward the route waypoint!
            if passed >= clear_after and self.pose is not None and (self.pose.x >= 0.48 or self.pose.y >= 0.34):
                print(f"    cleared corner obstacle at ({self.pose.x:.2f},{self.pose.y:.2f}); "
                      f"returning to route navigation", flush=True)
                break
            ahead_cm = self.sensors.read_sonic() if self.sensors is not None else None
            # Inside a measured gap an 18 deg swing would put a corner into the foam; only stop for a real block.
            if verified and ahead_cm is not None and ahead_cm < 15.0:
                print(f"    something {ahead_cm:.0f}cm ahead inside the gap; stopping", flush=True)
                break
            if not verified and ahead_cm is not None and ahead_cm < 30.0 and adjusts < 3:
                adjusts += 1
                l_cm = self.look(25) or 0.0
                r_cm = self.look(-25) or 0.0
                self.center_sonic()
                best_side = 1.0 if l_cm >= r_cm else -1.0
                if max(l_cm, r_cm) < 32.0:
                    break
                nudge = int(18 * best_side)
                print(f"    live adjust: ahead {ahead_cm:.0f}cm (L25={l_cm:.0f}cm R25={r_cm:.0f}cm) -> turn {nudge:+d}deg",
                      flush=True)
                kind = "left" if nudge > 0 else "right"
                rate = float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"])
                self.move(kind, abs(nudge) / rate)
            self.nudge_off_side_contact()
            step = min(0.14, length - passed)
            # Temporarily suppress `obstacle_ir` while driving down an ultrasonically verified open corridor
            old_avoid = self.args.no_avoid
            # The wide ultrasonic cone catches the gap's own edges; --stop-cm and the front IR still stop the car.
            if ahead_cm is not None and (ahead_cm > 45.0 or (verified and ahead_cm >= 15.0)):
                self.args.no_avoid = True
            try:
                elapsed, reason = self.move("forward", step / speed)
            finally:
                self.args.no_avoid = old_avoid
            passed += speed * elapsed
            if reason:
                if not verified and adjusts < 3 and (reason.startswith("obstacle") or reason.startswith("ir")
                                                     or reason.startswith("sonic")):
                    adjusts += 1
                    l_cm = self.look(25) or 0.0
                    r_cm = self.look(-25) or 0.0
                    self.center_sonic()
                    if max(l_cm, r_cm) >= 38.0:
                        nudge = 18 if l_cm >= r_cm else -18
                        print(f"    live adjust after {reason}: L25={l_cm:.0f}cm R25={r_cm:.0f}cm -> turn {nudge:+d}deg",
                              flush=True)
                        kind = "left" if nudge > 0 else "right"
                        rate = float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"])
                        self.move(kind, abs(nudge) / rate)
                        continue
                print(f"    corridor drive stopped after {passed * 100:.0f}cm ({reason})", flush=True)
                break
        fix = self.localize(0.5)
        if fix is not None:
            self.pose = fix
        return passed > 0.08

    def nudge_off_side_contact(self):
        """Strafe 3 cm away when one side's IR sees something under 8 cm and the other side is clear."""
        if self.sensors is None:
            return
        ir = self.sensors.read_ir()
        left = [k for k in ("front_left", "wheel_left_front", "wheel_left_rear") if ir.get(k)]
        right = [k for k in ("front_right", "wheel_right_front", "wheel_right_rear") if ir.get(k)]
        if bool(left) == bool(right):
            return
        toward = -1.0 if left else 1.0
        step = abs(self.safe_step(toward * 0.03, sideways=True))
        if step < 0.015:
            return
        name = "right" if toward < 0 else "left"
        print(f"    side IR on ({', '.join(left or right)}); strafing {name} {step * 100:.0f}cm", flush=True)
        self.move("strafe_left" if toward > 0 else "strafe_right", step / self.strafe_mps)

    def bypass_sonic(self) -> bool:
        """Ultrasonic-only detour for obstacles the camera cannot see (not blue foam)."""
        if not self.avoid_active() or self.servo is None:
            return False
        try:
            ahead = self.look(0)
            if not self.is_obstacle(ahead):
                print(f"    obstacle gone (sonic {'-' if ahead is None else f'{ahead:.0f}'}cm)", flush=True)
                return True
            p = self.pose
            w, h = (float(v) for v in self.arena["arena_size_m"])
            toward_centre = -(w / 2 - p.x) * math.sin(p.yaw) + (h / 2 - p.y) * math.cos(p.yaw)
            hw = self.body_half[1]
            side = None
            for sign in ((1.0, -1.0) if toward_centre >= 0 else (-1.0, 1.0)):
                name = "left" if sign > 0 else "right"
                room = min(self.side_room(sign), self.args.avoid_max_side_m)
                measured = self.side_gap_m(sign)
                if measured is not None:
                    room = min(room, max(measured - 0.05, 0.0))
                if room < self.args.avoid_side_m:
                    print(f"    {name}: only {max(room, 0.0) * 100:.0f}cm to the wall", flush=True)
                    continue
                need = max(self.args.avoid_side_m, self.obstacle_extent(sign) + hw + 0.05)
                if need > room:
                    print(f"    {name}: obstacle needs {need * 100:.0f}cm, room {room * 100:.0f}cm", flush=True)
                    continue
                side = sign * need
                break
            if side is None:
                print("    no room to go around the obstacle", flush=True)
                return False
            sign = 1.0 if side > 0 else -1.0
            print(f"    obstacle {ahead:.0f}cm ahead; going round on the {'left' if side > 0 else 'right'} "
                  f"({abs(side) * 100:.0f}cm)", flush=True)
            self.center_sonic()
            self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
            speed = float(self.motion["forward_mps"])
            passed, clear_at = 0.0, None
            # The ultrasonic is at the nose; the rest of the body is still beside the obstacle when it clears.
            tail = 2.0 * self.body_half[0]
            while passed < self.args.avoid_pass_m:
                left_to_go = self.args.avoid_pass_m - passed
                if clear_at is not None:
                    left_to_go = min(left_to_go, clear_at + tail - passed)
                    if left_to_go < 0.02:
                        break
                step = self.safe_step(min(0.12, left_to_go))
                if step < 0.02:
                    break
                elapsed, reason = self.move("forward", step / speed)
                passed += speed * elapsed
                if reason:
                    print(f"    still blocked while passing ({reason})", flush=True)
                    return passed > 0.05
                if clear_at is None and passed >= 0.10:
                    beside = self.look(-70 * sign)
                    self.center_sonic()
                    if not self.is_obstacle(beside, -70 * sign, limit_cm=60.0):
                        clear_at = passed
                        print(f"    nose past the obstacle after {passed * 100:.0f}cm; "
                              f"{tail * 100:.0f}cm more for the body", flush=True)
            return True
        finally:
            self.center_sonic()

    def wait_clear(self, reason: str):
        back = -self.safe_step(-0.08)
        rear_ir = self.sensors.read_ir() if self.sensors is not None else {}
        if back >= 0.02 and not any(rear_ir.get(k) for k in ("rear_left", "rear_mid", "rear_right")):
            print(f"  blocked ({reason}); reversing {back * 100:.0f}cm", flush=True)
            self.move("reverse", back / float(self.motion["forward_mps"]))
        else:
            print(f"  blocked ({reason}); rear wall close, skipping reverse", flush=True)
        print(f"  still blocked? waiting up to {self.args.blocked_wait_s:.0f}s", flush=True)
        deadline = time.monotonic() + self.args.blocked_wait_s
        while time.monotonic() < deadline:
            if self.stop_event.wait(0.3):
                raise Abort("stopped")
            reason = self.safety_reason()
            if reason is None:
                print("  path clear again", flush=True)
                return
        raise Abort(f"route blocked: {reason}")

    # ---- route ----------------------------------------------------------

    def record(self):
        if self.pose is not None:
            self.trail.append((self.pose.x, self.pose.y, self.pose.yaw))

    def save_map(self, extra=None):
        if not self.arena.get("arena_size_m"):
            return
        self.absorb_views()
        live = self.live_map.render(
            pose=self.pose, route=self.route, trail=self.trail, guide=self.guide_path,
            body_m=(self.body_half[0] * 2.0, self.body_half[1] * 2.0),
        )
        if self.home is not None:
            w_m, h_m = (float(v) for v in self.arena["arena_size_m"])
            margin = 28
            scale = (640 - 2 * margin) / max(w_m, h_m)
            img_h = int(h_m * scale) + 2 * margin
            hx = int(round(margin + self.home.x * scale))
            hy = int(round(img_h - margin - self.home.y * scale))
            cv2.drawMarker(live, (hx, hy), (0, 160, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.imwrite(str(self.debug_dir / "live_map.png"), live)
        img = live
        cv2.imwrite(str(self.debug_dir / "trajectory.png"),
                    draw_map(self.arena, route=self.route, trail=self.trail, pose=self.pose, size_px=640))
        if self.window is not None:
            views = []
            for name in ("front", "rear"):
                grabber = self.grabbers.get(name)
                frame = grabber.latest()[0] if grabber is not None else None
                if frame is None:
                    continue
                view = cv2.resize(frame, (img.shape[0] * frame.shape[1] // frame.shape[0], img.shape[0]))
                cv2.putText(view, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                views.append(view)
            if extra is not None:
                view = cv2.resize(extra, (img.shape[0] * extra.shape[1] // extra.shape[0], img.shape[0]))
                cv2.putText(view, "gimbal", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                views.append(view)
            if views:
                img = np.hstack([img, *views])
            cv2.imshow(self.window, img)
            cv2.waitKey(1)
        elif self.record_path is not None:
            for name in ("front", "rear"):
                grabber = self.grabbers.get(name)
                frame = grabber.latest()[0] if grabber is not None else None
                if frame is None:
                    continue
                view = cv2.resize(frame, (img.shape[0] * frame.shape[1] // frame.shape[0], img.shape[0]))
                cv2.putText(view, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                img = np.hstack([img, view])
        self.write_frame(img)

    def write_frame(self, img):
        if self.record_path is None:
            return
        if self.writer is None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.record_size = (img.shape[1] // 2 * 2, img.shape[0] // 2 * 2)
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self.writer = cv2.VideoWriter(str(self.record_path), fourcc, 1.0, self.record_size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = cv2.VideoWriter(str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                              1.0, self.record_size)
            print(f"recording {self.record_size[0]}x{self.record_size[1]} to {self.record_path}", flush=True)
        if img.shape[1::-1] != self.record_size:
            img = cv2.resize(img, self.record_size)
        self.writer.write(img)

    def refresh_pose(self, blind: int, full_turn: bool = True) -> int:
        fix = self.localize()
        if fix is not None:
            self.pose = fix
            return 0
        blind += 1
        if blind <= self.args.max_blind_segments and self.pose is not None:
            return blind
        if not full_turn and self.pose is not None:
            # Close to the target a wrong heading is costly: glance left and right instead of
            # spinning a full turn.
            print("  no tag for too long; glancing left and right", flush=True)
            held = self.pose.yaw
            for glance in (1, -2, 1):
                self.turn_by(glance * math.radians(15.0), min_s=0.05)
                fix = self.localize()
                if fix is not None:
                    self.pose = fix
                    return 0
            self.turn_by(wrap_rad(held - self.pose.yaw), min_s=0.05)
            return blind
        print("  no tag for too long; searching", flush=True)
        fix = self.search()
        if fix is None:
            raise Abort("lost: no tag visible after a full turn")
        self.pose = fix
        return 0

    def aim_point(self, prev, target) -> tuple[tuple[float, float], float, float]:
        """Point to steer at, plus along-track progress and cross-track error.

        Steering at a point a little ahead on the prev->target line keeps the car on
        the line (e.g. along a wall) instead of cutting straight at the corner.
        """
        if prev is None:
            return target, 0.0, 0.0
        sx, sy = target[0] - prev[0], target[1] - prev[1]
        length = math.hypot(sx, sy)
        if length < 1e-6:
            return target, 0.0, 0.0
        ux, uy = sx / length, sy / length
        px, py = self.pose.x - prev[0], self.pose.y - prev[1]
        along = px * ux + py * uy
        cross = -px * uy + py * ux
        ahead = along + self.args.lookahead_m
        if ahead >= length:
            return target, along - length, cross
        ahead = max(ahead, 0.0)
        cand = (prev[0] + ux * ahead, prev[1] + uy * ahead)
        obs = self.known_world_obstacles()
        if len(obs):
            # If the nominal point on the line is pinched by an obstacle (< 16 cm), slide forward along
            # the leg until the line clears the obstacle, or offset laterally into the parallel lane.
            safe_r = self.body_half[1] + self.args.avoid_gap_m
            for s_try in np.linspace(ahead, length, 8):
                pt = (prev[0] + ux * float(s_try), prev[1] + uy * float(s_try))
                if float(np.hypot(obs[:, 0] - pt[0], obs[:, 1] - pt[1]).min()) >= safe_r:
                    return pt, along - length, cross
            # Entire remaining segment is narrow: shift aim laterally at a point AT LEAST 28 cm ahead of the car
            # along the leg, AND keep the point inside the arena walls (`margin=0.22`) so we never steer toward
            # the outer wall / corner block (`offset = -0.18` on `y = 0.35` puts `y = 0.17` right into the block!).
            nx, ny = -uy, ux
            ahead_parallel = min(max(along + 0.28, ahead), length)
            base_pt = (prev[0] + ux * ahead_parallel, prev[1] + uy * ahead_parallel)
            w, h = (float(v) for v in self.arena["arena_size_m"])
            wall_keep = 0.22
            for offset in (0.18, -0.18, 0.26, -0.26):
                pt = (base_pt[0] + nx * offset, base_pt[1] + ny * offset)
                if not (wall_keep <= pt[0] <= w - wall_keep and wall_keep <= pt[1] <= h - wall_keep):
                    continue
                if float(np.hypot(obs[:, 0] - pt[0], obs[:, 1] - pt[1]).min()) >= safe_r:
                    return pt, along - length, cross - offset
        return cand, along - length, cross

    def waypoint_covered_by_obstacle(self, target: tuple[float, float], pts: np.ndarray) -> bool:
        """True when target sits inside or right next to a detected interior foam obstacle."""
        if self.pose is None or not len(pts):
            return False
        # Only use points <= 45 cm from the car so far fisheye distortion doesn't falsely cover a waypoint
        near_pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 0.45]
        if not len(near_pts):
            return False
        obs = to_world(near_pts, self.pose.x, self.pose.y, self.pose.yaw)
        d = np.hypot(obs[:, 0] - target[0], obs[:, 1] - target[1])
        return bool((d <= 0.18).any())

    def waypoint_inside_box(self, target: tuple[float, float], box) -> bool:
        """True when the obstacle box seen ahead starts before the waypoint and spans it sideways."""
        x0, x1, y0, y1 = box
        dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        tx, ty = dx * c + dy * s, -dx * s + dy * c
        return x0 < tx and x1 >= tx - 0.05 and y0 - 0.06 <= ty <= y1 + 0.06

    def planning_obstacles(self, sweep: bool) -> np.ndarray:
        """World points that the planner treats as solid: camera foam, remembered foam, and ultrasonic hits."""
        chunks = []
        pts = self.obstacle_points(fresh=sweep)
        if len(pts):
            near = pts[np.hypot(pts[:, 0], pts[:, 1]) <= 0.55]
            if len(near):
                chunks.append(to_world(near, self.pose.x, self.pose.y, self.pose.yaw))
        known = self.known_world_obstacles()
        if len(known):
            chunks.append(known)
        if sweep and self.servo is not None and self.sensors is not None:
            hits = []
            for angle in range(-90, 91, 15):
                cm = self.look(float(angle))
                if not self.is_obstacle(cm, float(angle), limit_cm=80.0):
                    continue
                bearing = self.pose.yaw + math.radians(angle)
                ox = (self.pose.x + self.args.sonic_x_m * math.cos(self.pose.yaw)
                      + cm / 100.0 * math.cos(bearing))
                oy = (self.pose.y + self.args.sonic_x_m * math.sin(self.pose.yaw)
                      + cm / 100.0 * math.sin(bearing))
                hits.append((ox, oy))
                print(f"    ultrasonic {angle:+d}deg hits at {cm:.0f}cm", flush=True)
            self.center_sonic()
            if hits:
                chunks.append(np.asarray(hits, dtype=np.float64))
        fused = self.live_map.obstacle_points()
        if len(fused):
            chunks.append(fused)
        if not chunks:
            return np.zeros((0, 2))
        return np.vstack(chunks)

    def sweep_exit(self, prefer_deg: float, held_deg: int | None) -> tuple[int, float] | None:
        """Face the open ultrasonic angle closest to the waypoint. Returns `(angle_deg, clear_room_m)`."""
        if self.servo is None or self.sensors is None:
            return None
        readings = []
        try:
            for angle in range(-90, 91, 15):
                cm = self.look(float(angle))
                wall = self.wall_range_cm(float(angle))
                if cm is None:
                    room, kind = 0.0, "no reading"
                elif self.is_obstacle(cm, float(angle), limit_cm=90.0):
                    room, kind = max(0.0, cm / 100.0 - 0.20), "obstacle"
                else:
                    room = max(0.0, min(cm, wall) / 100.0 - 0.18)
                    kind = "clear" if room >= 0.30 else "wall"
                readings.append((angle, room))
                seen = "-" if cm is None else f"{cm:.0f}cm"
                print(f"    ultrasonic {angle:+d}deg: {seen}, {kind}, clear {room * 100:.0f}cm", flush=True)
        finally:
            self.center_sonic()
        angle = choose_exit(readings, prefer_deg, held_deg=held_deg)
        if angle is None:
            print(f"    no exit with 30cm clear room near waypoint bearing {prefer_deg:+.0f}deg", flush=True)
            return None
        room = next(item[1] for item in readings if item[0] == angle)
        print(f"    exit {angle:+d}deg, clear {room * 100:.0f}cm; waypoint bearing {prefer_deg:+.0f}deg", flush=True)
        return angle, room

    def confirm_exit(self, prefer_deg: float, held_deg: int | None):
        """The servo camera sweeps the front and picks a gap wide enough for the body.

        Each view measures both foam corners in one frame. The car faces the gap centre
        closest to the waypoint, not a single ultrasonic ray.
        """
        del held_deg
        if self.gimbal_cam is None or self.servo is None:
            print("    gimbal camera is not available; not guessing an exit from the ultrasonic", flush=True)
            return None
        from aruco_localizer import UsbGrabber
        from gimbal_gap import draw_frame

        need = 2.0 * self.body_half[1] + 2.0 * self.args.gap_margin_m
        pans = tuple(range(90, -91, -15))
        seen = []
        ranges: list[tuple[int, float | None]] = []
        grabber = UsbGrabber("gimbal").start()
        try:
            deadline = time.monotonic() + 6.0
            while grabber.latest()[0] is None and time.monotonic() < deadline:
                if self.stop_event.wait(0.1):
                    raise Abort("stopped")
            if grabber.latest()[0] is None:
                print(f"    gimbal camera: no frame ({grabber.error or 'timeout'})", flush=True)
                return None
            for pan in pans:
                sonic = self.look(float(pan))
                ranges.append((pan, sonic))
                after = time.monotonic()
                frame, stamp = grabber.latest()
                pan_deadline = time.monotonic() + 2.5
                while (frame is None or stamp <= after) and time.monotonic() < pan_deadline:
                    if self.stop_event.wait(0.05):
                        raise Abort("stopped")
                    frame, stamp = grabber.latest()
                sonic_txt = "-" if sonic is None else f"{sonic:.0f}cm"
                if frame is None or stamp <= after:
                    print(f"    gimbal {pan:+d}deg: sonic {sonic_txt}, no frame", flush=True)
                    continue
                cam = self.gimbal_cam
                cols, rows, pts, too_close = cam.floor_profile(frame, float(pan))
                gaps = cam.find_gaps(cols, rows, pts, too_close, float(pan), cam.floor_mask(frame))
                self.save_map(draw_frame(frame, cam, cols, rows, pts, gaps, float(pan), need))
                near = [g for g in gaps if float(np.hypot(*g.centre)) <= 0.60
                        and abs(g.bearing_deg - pan) <= 25.0]
                if not near:
                    print(f"    gimbal {pan:+d}deg: sonic {sonic_txt}, no gap in front", flush=True)
                    continue
                for gap in near:
                    reach = float(np.hypot(*gap.centre))
                    wide = gap.straight_width_m >= need
                    print(f"    gimbal {pan:+d}deg: sonic {sonic_txt}, gap {gap.straight_width_m * 100:.0f}cm "
                          f"at {gap.bearing_deg:+.0f}deg, {reach * 100:.0f}cm away"
                          f"{'' if wide else f' (need {need * 100:.0f}cm)'}", flush=True)
                    if wide:
                        seen.append((gap, draw_frame(frame, cam, cols, rows, pts, gaps, float(pan), need)))
        finally:
            grabber.close()
            self.center_sonic()
        if not seen:
            open_pans = [(pan, sonic) for pan, sonic in ranges if sonic is not None and sonic >= 45.0]
            if open_pans:
                pan, sonic = min(open_pans, key=lambda item: abs(item[0] - prefer_deg))
                if abs(pan - prefer_deg) <= 30.0:
                    print(f"    no foam corners, but ultrasonic {pan:+d}deg is open for {sonic:.0f}cm "
                          f"(planned bearing {prefer_deg:+.0f}deg)", flush=True)
                    return ("open", float(pan), float(sonic))
            print(f"    gimbal camera: no gap at least {need * 100:.0f}cm wide toward the waypoint", flush=True)
            return None
        gap, image = min(seen, key=lambda item: (
            abs(item[0].bearing_deg - prefer_deg), abs(item[0].bearing_deg - item[0].pan_deg)))
        cv2.imwrite(str(self.debug_dir / "gimbal_gap.jpg"), image)
        print(f"    gimbal camera chooses the {gap.straight_width_m * 100:.0f}cm gap at {gap.bearing_deg:+.0f}deg "
              f"(waypoint bearing {prefer_deg:+.0f}deg)", flush=True)
        return gap

    def clearance_bearing(self, gap) -> float:
        """Aim a little toward the farther corner so a 25 cm gap does not rub the nearer foam."""
        left_r = float(np.hypot(*gap.left.point))
        right_r = float(np.hypot(*gap.right.point))
        reach = float(np.hypot(*gap.centre))
        spare = float(gap.straight_width_m) - 2.0 * self.body_half[1]
        shift = min(0.03, max(0.0, spare / 2.0 - 0.015))
        if abs(left_r - right_r) < 0.04 or shift < 0.01:
            return float(gap.bearing_deg)
        sign = 1.0 if right_r < left_r else -1.0
        bias = math.degrees(math.atan2(sign * shift, max(reach, 0.25)))
        side = "right" if sign > 0 else "left"
        print(f"    nearer {side} corner; aim {bias:+.0f}deg off the gap centre so that side stays clear",
              flush=True)
        return float(gap.bearing_deg) + bias

    def drive_measured_gap(self, gap, dist: float) -> float:
        """Turn to the gap centre and drive through it. Returns metres actually moved."""
        bearing = self.clearance_bearing(gap)
        if abs(bearing) > self.args.align_deg and not self.turn_by(math.radians(bearing), shuffle=False):
            return 0.0
        self.center_sonic()
        reach = float(np.hypot(gap.centre[0], gap.centre[1]))
        length = min(dist, max(0.18, min(0.28, reach + 0.08)))
        speed = float(self.motion["forward_mps"])
        print(f"    driving {length * 100:.0f}cm through the {gap.straight_width_m * 100:.0f}cm gap", flush=True)
        passed = 0.0
        while passed < length - 0.02 and not self.stop_event.is_set():
            ahead = self.sensors.read_sonic() if self.sensors is not None else None
            if ahead is not None and ahead < 10.0:
                print(f"    gap drive stopped (sonic {ahead:.0f}cm)", flush=True)
                break
            step = min(0.15, length - passed)
            self.gap_drive = True
            try:
                elapsed, reason = self.move("forward", step / speed)
            finally:
                self.gap_drive = False
            passed += speed * elapsed
            if reason:
                print(f"    gap drive stopped ({reason}) after {passed * 100:.0f}cm", flush=True)
                break
        print(f"    drove {passed * 100:.0f}cm through the gap", flush=True)
        return passed

    def line_blocked(self, target: tuple[float, float], obstacles: np.ndarray, inflate_m: float) -> bool:
        if not len(obstacles):
            return False
        dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
        dist = math.hypot(dx, dy)
        if dist < 0.08:
            return False
        for scale in np.linspace(0.18, 1.0, 16):
            x = self.pose.x + dx * float(scale)
            y = self.pose.y + dy * float(scale)
            if math.hypot(x - self.pose.x, y - self.pose.y) < 0.16:
                continue
            if float(np.hypot(obstacles[:, 0] - x, obstacles[:, 1] - y).min()) < inflate_m:
                return True
        return False

    def look_around(self) -> None:
        """One left-to-right ultrasonic sweep. Each reading updates the live map."""
        if self.servo is None or self.sensors is None:
            return
        print("    look", flush=True)
        self.live_map.begin_sweep()
        try:
            for angle in range(90, -91, -15):
                cm = self.look(float(angle))
                print(f"    {angle:+d}deg {('-' if cm is None else f'{cm:.0f}cm')}", flush=True)
        finally:
            self.live_map.commit_sweep()
        openings = []
        if self.pose is not None:
            openings = [math.degrees(wrap_rad(angle - self.pose.yaw))
                        for angle, _dist, kind in self.live_map.last_beams if kind == "gap"]
        if openings:
            text = ", ".join(f"{pan:+.0f}deg" for pan in openings)
            print(f"    opening {text}", flush=True)
        self.center_sonic()

    def heading_clear(self, world_angle: float) -> bool:
        """True when this heading lies inside an opening, not merely beside one."""
        beams = self.live_map.last_beams
        if not beams:
            return True
        gaps = sorted(angle for angle, _dist, kind in beams if kind == "gap")
        runs: list[list[float]] = []
        for angle in gaps:
            if runs and angle - runs[-1][-1] <= math.radians(22):
                runs[-1].append(angle)
            else:
                runs.append([angle])
        for run in runs:
            mid = (run[0] + run[-1]) / 2.0
            half = (run[-1] - run[0]) / 2.0
            if len(run) == 1:
                half = math.radians(8)
            if abs(wrap_rad(world_angle - mid)) <= half + 1e-6:
                return True
        angle, dist, kind = min(beams, key=lambda beam: abs(wrap_rad(beam[0] - world_angle)))
        if abs(wrap_rad(angle - world_angle)) > math.radians(22):
            return True
        return dist >= 0.45

    def path_aim(self, path: list[tuple[float, float]], target_dist: float) -> tuple[float, float]:
        """Point along the path that looks through the middle of an opening."""
        chosen = lookahead(path, min(0.25, target_dist))
        best = None
        best_err = math.pi
        best_reach = math.inf
        reach = 0.20
        while reach <= min(0.80, target_dist):
            cand = lookahead(path, reach)
            world = math.atan2(cand[1] - self.pose.y, cand[0] - self.pose.x)
            if self.heading_clear(world):
                runs = []
                gaps = sorted(angle for angle, _dist, kind in self.live_map.last_beams if kind == "gap")
                for angle in gaps:
                    if runs and angle - runs[-1][-1] <= math.radians(22):
                        runs[-1].append(angle)
                    else:
                        runs.append([angle])
                err = min(
                    (abs(wrap_rad(world - (run[0] + run[-1]) / 2.0)) for run in runs),
                    default=0.0,
                )
                if best is None or err + 1e-3 < best_err or (abs(err - best_err) <= math.radians(2) and reach < best_reach):
                    best, best_err, best_reach = cand, err, reach
            reach += 0.05
        return best if best is not None else chosen

    def note_bump(self, reason: str) -> None:
        """Remember a heading the bumper hit, including when the ultrasonic still reads it as open."""
        if self.pose is None or not (reason.startswith("obstacle") or reason.startswith("sonic") or reason.startswith("ir_")):
            return
        reach = 0.18
        self.live_map.remember_block(
            self.pose.x + reach * math.cos(self.pose.yaw),
            self.pose.y + reach * math.sin(self.pose.yaw),
            self.pose.x, self.pose.y,
        )
        self.block_yaw = self.pose.yaw
        self.block_at = (self.pose.x, self.pose.y)

    def heading_just_blocked(self) -> bool:
        if self.block_yaw is None or self.block_at is None or self.pose is None:
            return False
        if math.hypot(self.pose.x - self.block_at[0], self.pose.y - self.block_at[1]) > 0.25:
            self.block_yaw = None
            return False
        return abs(wrap_rad(self.pose.yaw - self.block_yaw)) < math.radians(25)

    def forward_arc_m(self) -> float | None:
        """Shortest of the beams 15° either side of the nose."""
        if self.pose is None or not self.live_map.last_beams:
            return None
        dists = []
        for pan in (-15, 0, 15):
            world = self.pose.yaw + math.radians(pan)
            angle, dist, _kind = min(self.live_map.last_beams, key=lambda beam: abs(wrap_rad(beam[0] - world)))
            if abs(wrap_rad(angle - world)) <= math.radians(20):
                dists.append(dist)
        return min(dists) if dists else None

    def detour_aim(self, target: tuple[float, float]) -> tuple[float, float] | None:
        """Point along an open beam, off the heading that just hit, and as near the waypoint as that allows."""
        if self.pose is None or not self.live_map.last_beams:
            return None
        waypoint = math.atan2(target[1] - self.pose.y, target[0] - self.pose.x)
        best = None
        best_score = math.inf
        for angle, dist, _kind in self.live_map.last_beams:
            if dist < 0.45:
                continue
            pan = wrap_rad(angle - self.pose.yaw)
            if abs(pan) < math.radians(35) or abs(pan) > math.radians(100):
                continue
            if self.block_yaw is not None and abs(wrap_rad(angle - self.block_yaw)) < math.radians(30):
                continue
            score = abs(wrap_rad(angle - waypoint))
            if score < best_score:
                best_score = score
                best = (self.pose.x + 0.40 * math.cos(angle), self.pose.y + 0.40 * math.sin(angle))
        return best

    def side_ir_on(self, sign: float) -> bool:
        if self.sensors is None:
            return False
        ir = self.sensors.read_ir()
        keys = (("front_left", "wheel_left_front", "wheel_left_rear") if sign > 0
                else ("front_right", "wheel_right_front", "wheel_right_rear"))
        return any(ir.get(key) for key in keys)

    def go_to(self, index: int, target: tuple[float, float], prev: tuple[float, float] | None = None):
        """Sense, plan on the live map, then one turn or one short drive."""
        blind = 0
        arena = tuple(float(v) for v in self.arena["arena_size_m"])
        inflate = self.body_half[1] + self.args.gap_margin_m
        while not self.stop_event.is_set():
            self.live_map.prune_blocks(self.pose.x, self.pose.y)
            dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
            dist = math.hypot(dx, dy)
            self.target_ahead = dist
            err_target = wrap_rad(math.atan2(dy, dx) - self.pose.yaw)
            _aim, past, cross = self.aim_point(prev, target)
            print(f"wp{index} ({target[0]:.2f},{target[1]:.2f}) dist={dist:.2f}m "
                  f"err={math.degrees(err_target):+.0f}deg off-line={cross * 100:+.0f}cm "
                  f"pose {self.pose.text()}", flush=True)
            self.record()
            overshot = dist < 2.5 * self.args.arrive_m and abs(err_target) > math.pi / 2
            passed = prev is not None and past >= -self.args.arrive_m and abs(cross) < 2.5 * self.args.arrive_m
            if dist <= self.args.arrive_m or overshot or passed:
                print(f"reached wp{index}", flush=True)
                self.save_map()
                return
            self.look_around()
            self.save_map()
            obstacles = self.live_map.obstacle_points()
            plan = plan_path((self.pose.x, self.pose.y), target, arena, obstacles, inflate)
            if not plan:
                raise Abort(f"no path to wp{index}; the live map closes every route")
            near_end = math.hypot(plan.path[-1][0] - self.pose.x, plan.path[-1][1] - self.pose.y) < 0.08
            if plan.goal_blocked:
                short = math.hypot(plan.path[-1][0] - target[0], plan.path[-1][1] - target[1])
                print(f"    waypoint still occupied {short * 100:.0f}cm ahead; keep approaching", flush=True)
            aim = target if plan.goal_blocked and near_end else self.path_aim(plan.path, dist)
            self.guide_path = list(plan.path)
            err = wrap_rad(math.atan2(aim[1] - self.pose.y, aim[0] - self.pose.x) - self.pose.yaw)
            print(f"    plan {len(plan.path)} cells -> ({aim[0]:.2f},{aim[1]:.2f}) "
                  f"turn {math.degrees(err):+.0f}deg", flush=True)
            arc = self.forward_arc_m()
            force_drive = False
            if (arc is not None and arc >= 0.40 and not self.heading_just_blocked()
                    and abs(err_target) < math.radians(80)
                    and abs(err) > math.radians(self.args.align_deg)):
                print(f"    ahead is open ({arc * 100:.0f}cm); drive", flush=True)
                force_drive = True
                err = 0.0
            elif self.heading_just_blocked():
                aim_world = math.atan2(aim[1] - self.pose.y, aim[0] - self.pose.x)
                if self.block_yaw is not None and abs(wrap_rad(aim_world - self.block_yaw)) < math.radians(35):
                    alt = self.detour_aim(target)
                    if alt is not None:
                        aim = alt
                        err = wrap_rad(math.atan2(aim[1] - self.pose.y, aim[0] - self.pose.x) - self.pose.yaw)
                        print(f"    that heading just hit; turn {math.degrees(err):+.0f}deg", flush=True)
            if not force_drive and not self.heading_clear(math.atan2(aim[1] - self.pose.y, aim[0] - self.pose.x)):
                print("    no open heading on the path", flush=True)
                back = -self.safe_step(-0.06)
                if back >= 0.02:
                    print(f"    reverse {back * 100:.0f}cm", flush=True)
                    self.move("reverse", back / float(self.motion["forward_mps"]))
                    blind = self.refresh_pose(blind)
                    self.record()
                    self.save_map()
                    continue
                raise Abort(f"no room to reach wp{index}")
            self.save_map()
            if abs(err) > math.radians(self.args.align_deg):
                if self.side_ir_on(1.0 if err > 0 else -1.0):
                    back = -self.safe_step(-0.06)
                    if back >= 0.02:
                        print(f"    that side is touching; reverse {back * 100:.0f}cm before the turn",
                              flush=True)
                        self.move("reverse", back / float(self.motion["forward_mps"]))
                self.turn_by(err, shuffle=False)
            else:
                segment = min(0.20, dist, math.hypot(aim[0] - self.pose.x, aim[1] - self.pose.y))
                segment = self.safe_step(segment)
                if segment < 0.02:
                    back = -self.safe_step(-0.06)
                    if back >= 0.02:
                        print(f"    no room forward; reverse {back * 100:.0f}cm", flush=True)
                        self.move("reverse", back / float(self.motion["forward_mps"]))
                    else:
                        raise Abort(f"no room to reach wp{index}")
                else:
                    print(f"    drive {segment * 100:.0f}cm", flush=True)
                    _elapsed, reason = self.move("forward", segment / float(self.motion["forward_mps"]))
                    if reason:
                        print(f"    drive stopped ({reason})", flush=True)
                        self.note_bump(reason)
            blind = self.refresh_pose(blind)
            self.record()
            self.save_map()

    def scan_test(self):
        """Print what the front camera sees as obstacles from where the car stands; never drives."""
        self.pose = self.localize(3.0)
        if self.pose is None:
            raise Abort("scan test: no tag fix here; a camera must see at least one tag")
        print(f"pose {self.pose.text()}", flush=True)
        self.target_ahead = None
        pts = self.obstacle_points(fresh=True)
        print(f"{len(pts)} obstacle floor points (car frame, cm):", flush=True)
        for x, y in pts:
            print(f"  ahead {x * 100:+4.0f}  left {y * 100:+4.0f}", flush=True)
        frame, _ = self.grabbers["front"].latest()
        cv2.imwrite(str(self.debug_dir / "scan_mask.png"), self.floor.mask(frame))
        box = self.obstacle_box(pts, self.args.avoid_look_m)
        if box is None:
            print(f"path clear for {self.args.avoid_look_m * 100:.0f}cm", flush=True)
            return
        print(f"obstacle in path: {box[0] * 100:.0f}-{box[1] * 100:.0f}cm ahead, "
              f"{box[2] * 100:+.0f}..{box[3] * 100:+.0f}cm sideways", flush=True)
        if self.servo is not None:
            self.confirm_sonic(box)
            self.center_sonic()
        choice = self.choose_side(pts, box)
        if choice is not None:
            print(f"plan: strafe {'left' if choice[0] > 0 else 'right'} {choice[1] * 100:.0f}cm", flush=True)
        else:
            corridor = self.find_open_corridor()
            if corridor is not None:
                ang, cm = corridor
                print(f"plan: shift (+9, +9)cm, rotate {ang:+d}deg into {cm:.0f}cm open corridor, "
                      f"then rejoin route", flush=True)
            else:
                print("plan: no way round", flush=True)

    def follow(self):
        if self.args.scan_test:
            self.scan_test()
            return
        self.pose = self.localize(3.0)
        if self.args.return_home:
            self.set_home()
        if self.pose is None:
            print("no tag at start; searching", flush=True)
            self.pose = self.search()
            if self.pose is None:
                raise Abort("no tag visible; check tags, lighting and arena_map.json")
        print(f"start pose {self.pose.text()}", flush=True)
        self.record()
        self.save_map()
        if self.args.hug:
            self.follow_hug()
            return
        while True:
            legs = self.legs()
            for i, (prev, wp) in enumerate(legs):
                self.go_to(i, wp, prev)
            if self.stop_event.is_set():
                raise Abort("stopped")
            if self.home is not None:
                self.dock()
            if not self.args.repeat:
                break
        print("route finished", flush=True)

    def body_inside(self, pose: Pose2D) -> bool:
        w, h = (float(v) for v in self.arena["arena_size_m"])
        margin = self.body_half[1]
        return margin <= pose.x <= w - margin and margin <= pose.y <= h - margin

    def consensus_pose(self, samples: list[Pose2D]) -> Pose2D | None:
        """One start pose. A fix that puts the body through a wall loses to one inside the arena.

        When the samples split into two clusters, the median can sit on the worse cluster.
        Pick the inside fix with the most front/rear tags instead.
        """
        if not samples:
            return None
        inside = [pose for pose in samples if self.body_inside(pose)]
        pool = inside or samples
        span = max(math.hypot(a.x - b.x, a.y - b.y) for a in pool for b in pool)

        def evidence(pose: Pose2D) -> tuple[int, int, float]:
            csi = sum(1 for fix in pose.fixes if fix.camera in ("front", "rear"))
            return (csi, len(pose.fixes), -pose.rms_px)

        if span > 0.15:
            best = max(pool, key=evidence)
            print(f"  start fixes disagree by {span * 100:.0f}cm; using {best.text()}", flush=True)
            return best
        ref = pool[0].yaw
        return Pose2D(
            st.median(pose.x for pose in pool),
            st.median(pose.y for pose in pool),
            wrap_rad(ref + st.median(wrap_rad(pose.yaw - ref) for pose in pool)),
            pool[-1].stamp,
            list(pool[-1].fixes),
            st.median(pose.rms_px for pose in pool),
        )

    def set_home(self):
        """Home = where the car was placed; it must be known before the car moves.

        With a map home (arena_map.json "home" or --home) a nearby start fix is used as the
        parking target, since the tags then read the same way again when the car is back there.
        """
        raw = self.args.home or self.arena.get("home")
        known = None
        if raw is not None:
            x, y, yaw = (float(v) for v in (raw.split(",") if isinstance(raw, str) else raw))
            known = Pose2D(x, y, math.radians(yaw))
        if self.pose is None:
            for attempt in range(5 if known is None else 2):
                print(f"no tag at start; looking again without moving ({attempt + 1})", flush=True)
                self.pose = self.localize(3.0)
                if self.pose is not None:
                    break
        if self.args.start_anywhere:
            if known is None:
                raise Abort("--start-anywhere needs \"home\" in arena_map.json or --home x,y,yaw")
            if self.pose is None:
                print("start anywhere: no tag fix yet; turning in place to find tags", flush=True)
                self.pose = self.search()
            if self.pose is None:
                raise Abort("start anywhere: no tag fix after a full turn; place the car where a camera "
                            "sees at least one tag")
        measured = None
        if self.pose is not None:
            first, self.pose = self.pose, None
            samples = [first] + [f for f in (self.localize() for _ in range(4)) if f is not None]
            measured = self.consensus_pose(samples)
            self.pose = measured
        if known is None:
            if measured is None:
                raise Abort("--return-home needs a tag fix at the start pose; place the car so a camera "
                            "sees at least one tag, or set \"home\" in arena_map.json")
            self.pose, self.home = measured, Pose2D(measured.x, measured.y, measured.yaw)
            source = "measured"
        elif measured is None:
            self.pose, self.home = Pose2D(known.x, known.y, known.yaw), known
            source = "map home (no tag fix here)"
        else:
            off = math.hypot(measured.x - known.x, measured.y - known.y)
            turn = abs(math.degrees(wrap_rad(measured.yaw - known.yaw)))
            print(f"map home x={known.x:.2f} y={known.y:.2f} yaw={known.yaw_deg:+.0f}deg; tags read "
                  f"{off * 100:.0f}cm / {turn:.0f}deg away", flush=True)
            if off <= 0.15 and turn <= 20.0:
                self.pose, self.home = measured, Pose2D(measured.x, measured.y, measured.yaw)
                source = "measured (matches map home)"
            elif self.args.start_anywhere:
                self.pose, self.home = measured, known
                source = f"map home; starting from tags at {measured.text()}"
            else:
                self.pose, self.home = Pose2D(known.x, known.y, known.yaw), known
                source = "map home (start fix disagrees; ignoring it)"
        print(f"home pose x={self.home.x:.2f} y={self.home.y:.2f} yaw={self.home.yaw_deg:+.0f}deg "
              f"[{source}]", flush=True)

    def averaged_fix(self, n: int) -> Pose2D | None:
        fixes = [f for f in (self.localize() for _ in range(n)) if f is not None]
        if not fixes:
            return None
        yaw = math.atan2(sum(math.sin(f.yaw) for f in fixes), sum(math.cos(f.yaw) for f in fixes))
        return Pose2D(sum(f.x for f in fixes) / len(fixes), sum(f.y for f in fixes) / len(fixes), yaw,
                      fixes[-1].stamp, fixes[-1].fixes, fixes[-1].rms_px)

    def strafe(self, left: float) -> bool:
        """Mecanum sideways move by `left` m (negative = right); learns the strafe speed from the next fix."""
        side = math.copysign(min(abs(left), 0.06), left)
        # Strafing along the lane swings the nose or the tail sideways into a wall the pose can
        # easily underestimate, so cap that direction with the ultrasonic reading.
        if self.lane_yaw is not None:
            along = abs(math.cos(wrap_rad(self.pose.yaw - self.lane_yaw)))
            if along > 0.7:
                sonic = self.sensors.read_sonic() if self.sensors is not None else None
                if sonic is not None:
                    side = math.copysign(min(abs(side), max(sonic / 100.0 - 0.04, 0.0) / along), side)
        side = self.safe_step(side, sideways=True)
        if abs(side) < 0.005:
            print("    wall beside the car; cannot strafe", flush=True)
            return False
        before = Pose2D(self.pose.x, self.pose.y, self.pose.yaw)
        # Command from the faster speed estimate so a slow guess cannot drive past the request.
        command_speed = max(self.strafe_mps, self.strafe_base, 0.18)
        elapsed, _ = self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / command_speed)
        fix = self.averaged_fix(2)
        if fix is None:
            return True
        self.pose = fix
        moved = -(fix.x - before.x) * math.sin(before.yaw) + (fix.y - before.y) * math.cos(before.yaw)
        if elapsed <= 0.0 or abs(moved) < 0.01:
            return True
        if moved * side < 0:
            if abs(side) >= 0.04 and abs(moved) >= 0.5 * abs(side):
                self.strafe_wrong += 1
                if self.strafe_wrong >= 2:
                    self.strafe_flip = not self.strafe_flip
                    self.strafe_wrong = 0
                    print("    strafe went the other way twice; flipping strafe direction", flush=True)
            return True
        self.strafe_wrong = 0
        if abs(side) < 0.03 or abs(abs(moved) - abs(side)) > 0.03:
            print(f"    strafed {moved * 100:+.0f}cm (asked {side * 100:+.0f}cm); keeping strafe speed "
                  f"{self.strafe_mps:.3f} m/s", flush=True)
            return True
        speed = min(max(abs(moved) / elapsed, 0.5 * self.strafe_base), 2.0 * self.strafe_base)
        self.strafe_mps = 0.5 * self.strafe_mps + 0.5 * speed
        print(f"    strafed {moved * 100:+.0f}cm (asked {side * 100:+.0f}cm); strafe speed now "
              f"{self.strafe_mps:.3f} m/s", flush=True)
        return True

    def go_pose(self, label: str, target: tuple[float, float, float], tol: float, tol_deg: float,
                attempts: int = 40, parking: bool = False) -> bool:
        """Reach (x, y, yaw) holonomically: turn to the heading, strafe sideways, drive along it.

        When parking, a blocked sensor ends the attempt instead of the run.
        """
        tol_yaw = math.radians(tol_deg)
        lane = max(self.args.lane_tol_m, tol / 2)
        blind = 0
        self.lane_yaw = target[2]
        self.lane_check = True
        try:
            return self._go_pose(label, target, tol, tol_deg, attempts, parking, tol_yaw, lane, blind)
        finally:
            self.lane_yaw = None
            self.lane_check = False

    def _go_pose(self, label, target, tol, tol_deg, attempts, parking, tol_yaw, lane, blind) -> bool:
        def blocked(reason):
            if not parking:
                if not (reason.startswith("obstacle") and self.bypass()):
                    self.wait_clear(reason)
                return False
            print(f"    blocked ({reason}) while parking; stopping here", flush=True)
            return True

        for attempt in range(1, attempts + 1):
            if self.stop_event.is_set():
                raise Abort("stopped")
            p = self.pose
            dyaw = wrap_rad(target[2] - p.yaw)
            dx, dy = target[0] - p.x, target[1] - p.y
            ahead = dx * math.cos(p.yaw) + dy * math.sin(p.yaw)
            left = -dx * math.sin(p.yaw) + dy * math.cos(p.yaw)
            print(f"{label} ({target[0]:.2f},{target[1]:.2f},{math.degrees(target[2]):+.0f}deg) "
                  f"ahead={ahead * 100:+.0f}cm left={left * 100:+.0f}cm heading err={math.degrees(dyaw):+.0f}deg "
                  f"pose {p.text()}", flush=True)
            if math.hypot(ahead, left) <= tol and abs(dyaw) <= tol_yaw:
                return True
            # Overshot along the lane (common at wall-hug ends); do not reverse/strafe into the wall.
            if ahead < -tol and abs(left) <= max(lane, tol) and abs(dyaw) <= max(tol_yaw, math.radians(6.0)):
                print("    passed target along the lane; accepting", flush=True)
                return True
            if attempt == 1:
                start_dist = math.hypot(ahead, left)
            if parking and math.hypot(ahead, left) > start_dist + 0.20:
                print("    pose drifted 20cm further from home; localization is unreliable here, stopping",
                      flush=True)
                return False
            near = math.hypot(ahead, left) < 0.15
            bearing = wrap_rad(math.atan2(dy, dx) - p.yaw)
            in_lane = abs(dyaw) <= math.radians(20.0) and ahead > 0 and abs(left) <= 0.5 * ahead
            self.lane_check = in_lane
            self.target_ahead = ahead if in_lane else math.hypot(dx, dy)
            if not near and not in_lane:
                # Far and not already lined up along the target heading: face the spot and drive there.
                if abs(bearing) > math.radians(self.args.align_deg):
                    self.turn_by(bearing)
                else:
                    step = self.safe_step(min(math.hypot(dx, dy), self.args.max_segment_m))
                    if step < 0.01:
                        self.turn_by(bearing if abs(bearing) > math.radians(3.0) else dyaw, min_s=0.05)
                    elif not self.handle_probe():
                        _elapsed, reason = self.move("forward", step / float(self.motion["forward_mps"]))
                        if reason and blocked(reason):
                            return False
            elif abs(dyaw) > tol_yaw:
                self.turn_by(dyaw, min_s=0.05)
            elif abs(left) > lane and (abs(left) >= abs(ahead) or abs(left) > self.args.lane_tol_m) \
                    and self.strafe(left):
                self.record()
                self.save_map()
                continue
            else:
                step = self.safe_step(max(-self.args.max_segment_m, min(self.args.max_segment_m, ahead)))
                if abs(step) < 0.01:
                    print("    wall reached; taking this as the spot", flush=True)
                    return False
                kind = "forward" if step > 0 else "reverse"
                if not (kind == "forward" and self.handle_probe()):
                    _elapsed, reason = self.move(kind, abs(step) / float(self.motion["forward_mps"]))
                    if reason and blocked(reason):
                        return False
            if near:
                fix = self.averaged_fix(2)
                self.pose, blind = (fix, 0) if fix is not None else (self.pose, self.refresh_pose(blind, False))
            else:
                blind = self.refresh_pose(blind, full_turn=not in_lane)
            self.record()
            self.save_map()
        print(f"WARNING: {label} not reached within {tol * 100:.0f}cm / {tol_deg:.0f}deg", flush=True)
        return False

    def dock(self):
        """Park on the home pose; the walls may be as close as they were when the car was placed there."""
        home = self.home
        print(f"parking at home x={home.x:.2f} y={home.y:.2f} yaw={home.yaw_deg:+.0f}deg", flush=True)
        self.sweep_correct()
        self.allow_gap = max(self.clearance(home.x, home.y, home.yaw) - 0.01, -0.03)
        self.parking = True
        try:
            ok = self.go_pose("park", (home.x, home.y, home.yaw), self.args.home_tol_m,
                              self.args.home_tol_deg, attempts=30, parking=True)
        finally:
            self.allow_gap = None
            self.parking = False
        p = self.pose
        off = math.hypot(home.x - p.x, home.y - p.y)
        dyaw = math.degrees(wrap_rad(home.yaw - p.yaw))
        print(f"{'parked at home' if ok else 'WARNING: parked near home'} "
              f"(off by {off * 100:.1f}cm, {dyaw:+.1f}deg)", flush=True)
        self.save_map()

    def build_hug_poses(self) -> list[tuple[float, float, float]]:
        """Counter-clockwise wall-hugging lap as (x, y, yaw) poses.

        Each wall: lane start, lane end, then the corner turning spot before and after the turn.
        """
        w, h = (float(v) for v in self.arena["arena_size_m"])
        hl, hw = self.body_half
        lane = hw + self.args.hug_gap_m
        spot = math.hypot(hl, hw) + self.args.wall_margin_m + 0.02
        walls = [
            (0.0, (spot, lane), (w - spot, lane), (w - spot, spot)),
            (90.0, (w - lane, spot), (w - lane, h - spot), (w - spot, h - spot)),
            (180.0, (w - spot, h - lane), (spot, h - lane), (spot, h - spot)),
            (-90.0, (lane, h - spot), (lane, spot), (spot, spot)),
        ]
        poses = []
        for i, (yaw, start, end, corner) in enumerate(walls):
            nxt = math.radians(walls[(i + 1) % 4][0])
            yaw = math.radians(yaw)
            poses += [(*start, yaw), (*end, yaw), (*corner, yaw), (*corner, nxt)]
        return poses

    def follow_hug(self):
        poses = self.hug_poses
        n = len(poses)

        def nearest(anchor):
            def cost(q):
                return math.hypot(q[0] - anchor.x, q[1] - anchor.y) + 0.1 * abs(wrap_rad(q[2] - anchor.yaw))
            return min((i for i in range(n) if i % 4 in (0, 1)), key=lambda i: cost(poses[i]))

        if self.args.start_anywhere and self.home is not None:
            k, end = nearest(self.pose), nearest(self.home)
            count = n + (end - k) % n + 1
        else:
            k = nearest(self.home or self.pose)
            count = n + 1
        print(f"hug lane {self.args.hug_gap_m * 100:.0f}cm from the walls; starting at pose {k}, "
              f"{count} poses before parking", flush=True)
        while True:
            for j in range(count):
                i = (k + j) % n
                kind = ("lane start", "lane end", "corner", "turned")[i % 4]
                self.go_pose(f"p{i} {kind}", poses[i], 0.04, 4.0)
            if self.stop_event.is_set():
                raise Abort("stopped")
            if self.home is not None:
                self.dock()
                k, count = nearest(self.home), n + 1
            if not self.args.repeat:
                break
        print("route finished", flush=True)

    def legs(self) -> list[tuple[tuple[float, float] | None, tuple[float, float]]]:
        """(previous point, target) pairs; a closed route is joined at the side nearest the car.

        With a home pose the lap starts and ends at the point of that side nearest home.
        """
        route = self.route
        closed = len(route) > 2 and math.dist(route[0], route[-1]) < 1e-6
        if not closed or self.args.start_at_first:
            return [(route[i - 1] if i else None, wp) for i, wp in enumerate(route)]
        ring = route[:-1]
        n = len(ring)
        anchor = self.home or self.pose

        def nearest(a, b):
            sx, sy = b[0] - a[0], b[1] - a[1]
            t = ((anchor.x - a[0]) * sx + (anchor.y - a[1]) * sy) / max(sx * sx + sy * sy, 1e-9)
            t = min(max(t, 0.0), 1.0)
            return a[0] + t * sx, a[1] + t * sy

        k = min(range(n), key=lambda i: math.dist((anchor.x, anchor.y), nearest(ring[i], ring[(i + 1) % n])))
        if self.home is not None:
            join = nearest(ring[k], ring[(k + 1) % n])
            order = [ring[(k + 1 + j) % n] for j in range(n)] + [join]
            prevs = [join] + order[:-1]
        else:
            order = [ring[(k + 1 + j) % n] for j in range(n)] + [ring[(k + 1) % n]]
            prevs = [ring[k]] + order[:-1]
        print(f"joining route between ({ring[k][0]:.2f},{ring[k][1]:.2f}) and "
              f"({ring[(k + 1) % n][0]:.2f},{ring[(k + 1) % n][1]:.2f})", flush=True)
        return list(zip(prevs, order))

    def calibrate_motion(self):
        if self.motor is None:
            raise Abort("--calibrate-motion needs --arm")

        def fix_or_abort(label: str) -> Pose2D:
            fix = self.localize(3.0)
            if fix is None:
                raise Abort(f"calibration: no tag at '{label}'; face the car toward the tags")
            print(f"  {label}: {fix.text()}", flush=True)
            return fix

        p0 = fix_or_abort("start")
        t_fwd, reason = self.move("forward", 1.0)
        if reason:
            raise Abort(f"calibration forward stopped early: {reason}")
        p1 = fix_or_abort("after forward")
        t_left, _ = self.move("left", 1.2)
        p2 = fix_or_abort("after left turn")
        t_right, _ = self.move("right", 1.2)
        p3 = fix_or_abort("after right turn")

        mps = math.hypot(p1.x - p0.x, p1.y - p0.y) / t_fwd
        left = math.degrees(abs(wrap_rad(p2.yaw - p1.yaw))) / t_left
        right = math.degrees(abs(wrap_rad(p3.yaw - p2.yaw))) / t_right
        print(f"forward {mps:.3f} m/s  left {left:.1f} deg/s  right {right:.1f} deg/s", flush=True)
        self.motion.update(
            forward_mps=round(mps, 4),
            turn_left_dps=round(left, 2),
            turn_right_dps=round(right, 2),
            calibrated=True,
        )
        self.car["motion"] = self.motion
        self.car_path.write_text(json.dumps(self.car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {self.car_path}", flush=True)

    def measure_drift(self):
        """Drive straight with fixed wheel PWM and report how far the pose left the line."""
        if self.motor is None:
            raise Abort("--measure-drift needs --arm")
        if self.args.wheel_pwm:
            pwm = tuple(int(v) for v in self.args.wheel_pwm.split(","))
            if len(pwm) != 4:
                raise Abort("--wheel-pwm needs four integers: left-front,left-rear,right-front,right-rear")
        else:
            pwm = self._forward_pwm()
        distance = float(self.args.drift_m)
        speed = float(self.motion["forward_mps"])
        if distance <= 0.0 or speed <= 0.0:
            raise Abort("--drift-m and forward_mps must be positive")

        def sample(label: str) -> Pose2D:
            fixes = [f for f in (self.localize(3.0, gate=False) for _ in range(4)) if f is not None]
            if len(fixes) < 2:
                raise Abort(f"no reliable tag fix at {label}; move the car until two tags are visible")
            ref = fixes[0].yaw
            pose = Pose2D(
                st.median(f.x for f in fixes),
                st.median(f.y for f in fixes),
                wrap_rad(ref + st.median(wrap_rad(f.yaw - ref) for f in fixes)),
                fixes[-1].stamp,
                fixes[-1].fixes,
                st.median(f.rms_px for f in fixes),
            )
            print(f"  {label}: {pose.text()} ({len(fixes)} fixes)", flush=True)
            return pose

        print(f"straight run {distance:.2f} m, wheel PWM {pwm}", flush=True)
        start = sample("start")
        self.pose = Pose2D(start.x, start.y, start.yaw)
        duration = distance / speed
        elapsed, reason = self._drive_pwm(pwm, duration)
        travelled_cmd = speed * elapsed
        if reason:
            print(f"  stopped early: {reason} after {elapsed:.2f}s ({travelled_cmd:.2f} m commanded)", flush=True)
        end = sample("end")
        dx, dy = end.x - start.x, end.y - start.y
        along = dx * math.cos(start.yaw) + dy * math.sin(start.yaw)
        left = -dx * math.sin(start.yaw) + dy * math.cos(start.yaw)
        dyaw = math.degrees(wrap_rad(end.yaw - start.yaw))
        per_m = dyaw / along if along > 0.2 else float("nan")
        print(
            f"result: along={along * 100:+.1f}cm  left={left * 100:+.1f}cm  "
            f"yaw={dyaw:+.1f}deg  ({per_m:+.1f} deg/m)",
            flush=True,
        )
        print("positive left/yaw means the car drifted left. PWM was not saved.", flush=True)

    def _drive_pwm(self, pwm: tuple[int, int, int, int], duration: float) -> tuple[float, str | None]:
        """Timed four-wheel command. Forward safety still applies; the pose is not steered."""
        reason = None
        started = time.monotonic()
        self.drive(pwm)
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - started
            if elapsed >= duration:
                break
            reason = self.safety_reason()
            if reason:
                break
            time.sleep(0.04)
        self.stop()
        elapsed = time.monotonic() - started
        if self.stop_event.is_set():
            raise Abort("stopped")
        return elapsed, reason


def main():
    args = parse_args()
    follower = RouteFollower(args)
    signal.signal(signal.SIGINT, follower.request_stop)
    signal.signal(signal.SIGTERM, follower.request_stop)
    if not follower.motion.get("calibrated") and not args.calibrate_motion and not args.measure_drift:
        print("NOTE: car_nav.json motion is not calibrated; run --calibrate-motion --arm first.", flush=True)
    try:
        follower.start()
        if args.calibrate_motion:
            follower.calibrate_motion()
        elif args.measure_drift:
            follower.measure_drift()
        else:
            follower.follow()
    except Abort as exc:
        print(f"ABORT: {exc}", flush=True)
    finally:
        follower.close()
        print(f"Stopped; motors off. Map: {follower.debug_dir / 'trajectory.png'}", flush=True)


if __name__ == "__main__":
    main()
