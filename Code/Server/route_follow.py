#!/usr/bin/env python3
"""Drive a waypoint route from arena_map.json using ArUco wall tags.

Stop-and-look: the car stops, localizes from the tags (front + rear CSI),
turns in place toward the next waypoint, drives a short straight segment, and
repeats. Between tag fixes the pose is dead-reckoned from the commanded motion
rates in car_nav.json. Ultrasonic, front IR and the front camera risk score
stop a forward segment at any time.

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
from obstacle_map import ObstacleMap, path_gap, shift_to_clear, split_walls, to_world
from pose_kalman import PoseKalman
from vision_detector import classic_corridor_score

SERVER = Path(__file__).resolve().parent
SONIC_PIVOT_M = 0.085  # gimbal pan axis ahead of the car centre (the ultrasonic turns about it)
SONIC_ARM_M = 0.03     # ultrasonic face ahead of the pan axis
SONIC_AHEAD_M = SONIC_PIVOT_M + SONIC_ARM_M  # ultrasonic face ahead of the car centre, facing forward
BLOCK_LONG_M = 0.20    # longest side of an obstacle block (arena_map.json obstacle_block_m)
SONIC_HALF_DEG = 8.0   # assumed half width of the ultrasonic beam; the HC-SR04's is wider
SCAN_STEP_M = 0.15     # longest lane step between two looks with the ultrasonic only
BLIND_GAP_M = 0.24     # anything nearer than this (from the ultrasonic face) is in the camera blind zone: it then reports the
                       # bottom row of the frame, 15-17 cm ahead of the nose on the car, not the distance
CAM_STEP_M = 0.20      # ... with the camera, which sees the lane 35 cm ahead of the nose
CAM_PANS = (-30.0, 30.0, 0.0)  # gimbal angles of one camera look (+ = left); ends facing ahead
PASS_MARGIN_M = 0.08   # the tail clears the block's far face by this much before strafing back (pose drifts a few cm)
WALL_ECHO_M = 0.16     # what the camera sees this close to where the pose puts the wall ahead is the wall
PATH_MARGIN_M = 0.025  # an obstacle within this of the body's width counts as in the path
CLEAR_MARGIN_M = 0.03  # a detour strafes until the path is clear by this much
EDGE_PAN_DEG = 25.0    # the ultrasonic also looks this far to each side for the edges of the path (15 missed a block 3 cm in)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="ArUco waypoint route following.")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--car", default=str(CAR_PATH))
    p.add_argument("--route", default="loop", help="Route name in arena_map.json routes.")
    p.add_argument("--repeat", action="store_true", help="Loop the route until Ctrl+C.")
    p.add_argument("--arm", action="store_true", help="Enable motors. Without it nothing moves.")
    p.add_argument("--calibrate-motion", action="store_true",
                   help="Measure forward m/s and turn deg/s with tags, write car_nav.json.")
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
    p.add_argument("--hug-gap-m", type=float, default=None,
                   help="Body side to wall gap in --hug lanes. Default: the lanes run through the corner turning "
                        "spots, so the lap is a plain rectangle (about 14 cm). Smaller gaps hug the walls closer "
                        "but step inward at every corner to turn."),
    p.add_argument("--no-detour", dest="detour", action="store_false",
                   help="In --hug lanes, stop at an obstacle instead of strafing around it.")
    p.add_argument("--no-pose-filter", dest="pose_filter", action="store_false",
                   help="In --hug, accept or ignore each tag fix with fixed jump limits instead of the pose filter.")
    p.add_argument("--no-save-looks", dest="save_looks", action="store_false",
                   help="Do not save every camera look (frames and pose) under debug-dir/looks for replay.")
    p.add_argument("--no-camera-obstacles", dest="camera_obstacles", action="store_false",
                   help="Find obstacles with the ultrasonic and IR only (the gimbal camera is the default).")
    p.add_argument("--approach-cm", type=float, default=5.0,
                   help="Close in to this far in front of the car's nose before strafing around an obstacle. "
                        "The camera sees the floor only from 13 cm out, so the last stretch is driven by "
                        "dead reckoning from the last look; 3 cm or less risks touching it.")
    p.add_argument("--detour-max-m", type=float, default=0.40,
                   help="Give up when the obstacle is still ahead after strafing this far off the lane.")
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
    p.add_argument("--min-battery-v", type=float, default=7.0,
                   help="Only a warning below this: the run starts anyway.")
    p.add_argument("--no-ir", action="store_true")
    p.add_argument("--no-sonic", action="store_true")
    p.set_defaults(active_high=False)
    p.add_argument("--ir-active-high", action="store_true", dest="active_high")
    p.add_argument("--debug-dir", default="route_follow_debug")
    p.add_argument("--no-window", action="store_true", help="Do not open the live trajectory window.")
    p.add_argument("--record", nargs="?", const="route_follow_debug/run.mp4", default=None,
                   help="Save the live view (trajectory plus both cameras) to this mp4 "
                        "(default route_follow_debug/run.mp4).")
    return p.parse_args(argv)


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
        if not args.calibrate_motion and args.route not in routes:
            raise SystemExit(f"route '{args.route}' not in {args.map}; have {sorted(routes)}")
        self.route = [tuple(map(float, wp)) for wp in routes.get(args.route, [])]
        self.grabbers = {}
        self.sensors = None
        self.motor = None
        self.servo = None
        self.home_pan, self.home_tilt = 90, 90
        self.right_90_pan, self.left_90_pan = 180, 0
        self.pose: Pose2D | None = None
        self.trail: list[tuple[float, float, float]] = []
        self.stop_event = threading.Event()
        self.sonic_misses = 0
        self.travel = 0.0  # forward distance of the move in progress, not yet in the pose
        self.turn_scale = {"left": 1.0, "right": 1.0}  # learned from tag fixes after turns (battery changes the rate)
        self.last_turn: tuple[str, float, float] | None = None  # (kind, yaw before, commanded turn rad), for learning
        self.closing_cm: float | None = None  # ultrasonic stop while closing in on an obstacle to go around
        self.closing_ir_after: float | None = None  # while closing in: the front IR may light after this much travel
        self.closed_in: float | None = None   # gap (m, from the ultrasonic face) expected after the last closing move
        self.detouring = False                # True while going around an obstacle (tag headings are trusted)
        self.use_filter = bool(args.hug and args.pose_filter)
        self.kalman = PoseKalman(*(float(v) for v in self.arena["arena_size_m"]),
                                 tuple(float(v) / 2.0 for v in self.car.get("body_m", [0.29, 0.18])),
                                 float(self.motion.get("strafe_yaw_deg_per_cm", 0.6)))
        self.scanner = None                   # camera obstacle finder (foot_cam.CameraScanner) or None
        self.obstacles = ObstacleMap(self.arena["arena_size_m"])
        self.last_views: list = []
        self.look_no = 0
        self.front_risk = 0.0
        self.debug_dir = SERVER / args.debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        if args.save_looks:  # a fresh run keeps only its own looks
            import shutil

            shutil.rmtree(self.debug_dir / "looks", ignore_errors=True)
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
        self.hug_poses: list[tuple[float, float, float]] = []
        if args.hug:
            self.hug_poses = self.build_hug_poses()
            self.route = [(x, y) for x, y, _ in self.hug_poses] + [self.hug_poses[0][:2]]

    # ---- hardware -------------------------------------------------------

    def start(self):
        # Hardware modules load here, so the simulators (route_sim.py, Webots) can import this file anywhere.
        from full_car_test import estimate_battery_percent
        from safe_surround_cruise import SafetySensors, read_battery_voltage
        from servo import Servo

        try:
            from servo import load_gimbal_home, load_left_90_pan, load_right_90_pan

            self.servo = Servo()  # applies gimbal_home.json: ultrasonic straight ahead
            self.home_pan, self.home_tilt = load_gimbal_home()
            self.right_90_pan, self.left_90_pan = load_right_90_pan(self.home_pan), load_left_90_pan()
        except Exception as exc:
            print(f"WARNING gimbal: {exc}", flush=True)
        self.grabbers = open_grabbers(self.car, self.args.width, self.args.height, rear=not self.args.no_rear)
        self.sensors = SafetySensors(
            SimpleNamespace(no_ir=self.args.no_ir, no_sonic=self.args.no_sonic, active_high=self.args.active_high)
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not all(g.latest()[0] is not None for g in self.grabbers.values()):
            time.sleep(0.1)
        missing = [name for name, g in self.grabbers.items() if g.latest()[0] is None]
        if missing:
            raise Abort(f"cameras not ready: {missing}")
        if self.args.hug and self.args.detour and self.args.camera_obstacles and self.servo is not None:
            try:  # one USB camera (the gimbal's) at a time works; the side USB cameras stay closed
                from foot_cam import CameraScanner, FootCamera, GimbalFeed

                feed = GimbalFeed().start()
                feed.wait_ready()
                self.scanner = CameraScanner(FootCamera(self.car), feed)
                print("obstacles: gimbal camera foot lines + ultrasonic + IR", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING gimbal camera unavailable ({exc}); obstacles from the ultrasonic and IR only",
                      flush=True)
        if self.args.arm:
            battery = read_battery_voltage()
            if battery is None:
                print("WARNING battery voltage unreadable; starting anyway", flush=True)
            else:
                low = f" (below {self.args.min_battery_v}V: motors run slower than calibrated)" \
                    if battery < self.args.min_battery_v else ""
                print(f"ARMED battery={battery:.2f}V ({estimate_battery_percent(battery)}%){low}. "
                      f"Starting in 3 s.", flush=True)
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
        if self.scanner is not None:
            try:
                self.scanner.close()
            except Exception:
                pass
        if self.obstacles.blocks:
            (self.debug_dir / "obstacles.json").write_text(
                json.dumps(self.obstacles.as_dict(), indent=2) + "\n", encoding="utf-8")
        if self.servo is not None and hasattr(self.servo, "release_gimbal"):
            try:  # face ahead, then stop the pulses so the gimbal does not buzz after the run
                self.servo.set_pan(self.home_pan)
                time.sleep(0.4)
                self.servo.release_gimbal()
            except Exception:
                pass
        if self.sensors is not None:
            self.sensors.close()
        for grabber in self.grabbers.values():
            grabber.close()
        self.save_map()
        if self.writer is not None:
            self.writer.release()
            print(f"saved recording {self.record_path}", flush=True)
        if self.window is not None:
            cv2.destroyWindow(self.window)

    # ---- sensing --------------------------------------------------------

    def look(self, pan_deg: float):
        """Point the ultrasonic at `pan_deg` relative to straight ahead (positive = left) and read it."""
        if self.servo is None or self.sensors is None:
            return None
        from servo import servo_pan_for_bearing

        # The servo is not 1:1 and differs left and right (gimbal_home.json right_90_pan / left_90_pan).
        self.servo.set_pan(servo_pan_for_bearing(self.home_pan, self.right_90_pan, pan_deg, self.left_90_pan))
        if self.stop_event.wait(0.35):
            raise Abort("stopped")
        return self.sensors.read_sonic()

    def sweep_correct(self):
        """Replace the sideways part of the pose with what the ultrasonic sweep measures.

        Only beams that look sideways and land on a wall running along the heading measure the
        sideways error: moving the car left by d shortens such a beam at angle a by d / sin(a).
        A reading far from the wall the pose predicts is an obstacle (or a glancing echo) and is
        left out.
        """
        found = self.sweep()
        if self.pose is None or len(found) < 2:
            return
        p = self.pose
        along_x = abs(math.cos(p.yaw)) > math.cos(math.radians(30.0))  # heading runs along a y=const wall
        shifts = []
        for angle, cm in found.items():
            if abs(angle) < 45:
                continue
            beam = p.yaw + math.radians(angle)
            ox = p.x + SONIC_PIVOT_M * math.cos(p.yaw) + SONIC_ARM_M * math.cos(beam)
            oy = p.y + SONIC_PIVOT_M * math.sin(p.yaw) + SONIC_ARM_M * math.sin(beam)
            predicted, axis = self.wall_hit(ox, oy, beam)
            if axis != ("y" if along_x else "x") or abs(predicted - cm / 100.0) > 0.12:
                continue
            shifts.append((predicted - cm / 100.0) * math.sin(math.radians(angle)))
        if not shifts:
            print("    sweep did not match a side wall; keeping the camera pose", flush=True)
            return
        shift = st.median(shifts)
        p.x -= shift * math.sin(p.yaw)
        p.y += shift * math.cos(p.yaw)
        print(f"    sweep moved the pose {shift * 100:+.0f}cm sideways -> {p.text()}", flush=True)

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
                self.servo.set_pan(self.home_pan)
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
        stop_cm = self.args.stop_cm if self.closing_cm is None else min(self.args.stop_cm, self.closing_cm)
        if sonic is not None and sonic < stop_cm and not self.wall_echo(sonic):
            return f"sonic_{sonic:.0f}cm"
        ir = self.sensors.read_ir()
        if self.closing_ir_after is not None and self.travel >= self.closing_ir_after:
            ir = {**ir, "front": 0, "front_left": 0, "front_right": 0}  # expected to light now: closing in on purpose
        near = sonic is None or sonic < self.args.front_ir_gate_cm
        if near and (ir["front"] or ir["front_left"] or ir["front_right"]) \
                and not (sonic is not None and sonic > self.args.front_ir_gate_cm - 6.0):
            return "ir_front"
        if self.args.hug and self.args.detour and (ir["front_left"] or ir["front_right"]):
            # A block that covers only the edge of the path is missed by the centre beam; the corner
            # IR is all that sees it. A false trigger at the wall ahead is taken as arriving there.
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

    def wall_echo(self, sonic_cm: float) -> bool:
        """In --hug lanes with detours, a reading that matches the wall straight ahead is not a reason to
        stop until 8 cm: safe_step already keeps the body off that wall, and backing away from it
        could put the car into an obstacle it has just gone around."""
        if not (self.args.hug and self.args.detour) or self.pose is None or sonic_cm < 8.0:
            return False
        return abs(sonic_cm / 100.0 - (self.wall_ahead() - self.travel)) < 0.06

    def wall_ahead(self) -> float:
        """Distance from the ultrasonic face to the wall straight ahead, by the pose."""
        p = self.pose
        return self.wall_distance(p.x + SONIC_AHEAD_M * math.cos(p.yaw), p.y + SONIC_AHEAD_M * math.sin(p.yaw), p.yaw)

    def localize(self, timeout: float | None = None) -> Pose2D | None:
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
        pose = self.localizer.fuse(fixes, time.monotonic())
        if pose is None and fixes:
            print(f"  no fix: {self.localizer.last_reject}; dead-reckoning", flush=True)
        return self.accept_fix(pose)

    def accept_fix(self, pose: Pose2D | None) -> Pose2D | None:
        """Fuse a tag fix into the pose: the filter (--hug), or the fixed jump limits of `accept_fix_hard`."""
        if not self.use_filter:
            return self.accept_fix_hard(pose)
        if pose is None:
            return None
        pairs = {(f.camera, f.tag_id) for f in pose.fixes}
        nearest = min((f.distance for f in pose.fixes), default=9.9)
        if self.pose is None or not self.kalman.ready:
            self.kalman.start(pose.x, pose.y, pose.yaw, len(pairs), nearest)
            self.jumps, self.jump_pose = 0, None
            return pose
        jump = math.hypot(pose.x - self.pose.x, pose.y - self.pose.y)
        if self.lane_check and jump > 0.10 and self.clearance(self.pose.x, self.pose.y, self.pose.yaw) < 0.06:
            print(f"  fix {pose.text()} jumped {jump * 100:.0f}cm while close to a wall; sweeping", flush=True)
            self.sweep_correct()
            return None
        self.kalman.sync(self.pose.x, self.pose.y, self.pose.yaw)
        before = self.pose
        if self.kalman.update_tag(pose.x, pose.y, pose.yaw, len(pairs), nearest) == "outlier":
            strong = len(pairs) >= 3 or any(f.distance < self.localizer.max_single_tag_m for f in pose.fixes)
            last = self.jump_pose
            same = last is not None and math.hypot(pose.x - last.x, pose.y - last.y) < self.args.max_jump_m
            self.jumps = self.jumps + 1 if same else 1
            self.jump_pose = pose
            if not strong and (self.parking or self.jumps < 4):
                sigma_xy, sigma_yaw = self.kalman.sigma()
                print(f"  ignoring fix {pose.text()}: outside what the filter expects "
                      f"(+-{sigma_xy * 100:.0f}cm, +-{sigma_yaw:.0f}deg)", flush=True)
                return None
            print("  far tags keep agreeing on a new pose; trusting them" if not strong
                  else f"  strong fix {pose.text()} far from the expected pose; trusting it", flush=True)
            self.kalman.start(pose.x, pose.y, pose.yaw, len(pairs), nearest)
            fused = pose
            if len(pairs) >= 2:
                self.learn_turn(pose.yaw)
        else:
            fused = Pose2D(float(self.kalman.x[0]), float(self.kalman.x[1]), float(self.kalman.x[2]), pose.stamp,
                           pose.fixes, pose.rms_px)
            if len(pairs) >= 2:
                self.learn_turn(pose.yaw)
        self.jumps, self.jump_pose = 0, None
        moved = math.hypot(fused.x - before.x, fused.y - before.y)
        turned = abs(math.degrees(wrap_rad(fused.yaw - before.yaw)))
        if moved > 0.05 or turned > 5.0:
            print(f"  fix moved the pose {moved * 100:.0f}cm / {turned:.0f}deg -> {fused.text()}", flush=True)
        # Two or more tags agreeing on a heading far from the lane are believed: after a corner the car can really be
        # 25 deg off (a turn that went too far), and forcing the lane heading back drives it into the wall.
        if self.lane_check and self.lane_yaw is not None and not self.detouring and len(pairs) < 2 \
                and abs(wrap_rad(fused.yaw - self.lane_yaw)) > math.radians(25.0):
            print(f"  fix {fused.text()}: heading {abs(math.degrees(wrap_rad(fused.yaw - self.lane_yaw))):.0f}deg "
                  f"off the lane; keeping its position, not its heading", flush=True)
            fused.yaw = before.yaw
            self.kalman.x[2] = before.yaw
        return fused

    def accept_fix_hard(self, pose: Pose2D | None) -> Pose2D | None:
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
            strong = len({(f.camera, f.tag_id) for f in pose.fixes}) >= (2 if self.detouring else 3) or \
                any(f.distance < self.localizer.max_single_tag_m for f in pose.fixes)
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
        if self.lane_check and self.lane_yaw is not None and not self.detouring \
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
        """Left-front, left-rear, right-front, right-rear duties for driving straight (car_nav.json)."""
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
        if kind not in ("left", "right"):
            self.last_turn = None
        if self.motor is None:
            print(f"    would {kind} {duration:.2f}s", flush=True)
            if self.stop_event.wait(0.5):
                raise Abort("stopped")
            return 0.0, None
        reason = None
        started = time.monotonic()
        self.drive(self._pwm(kind))
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - started
            if elapsed >= duration:
                break
            if kind == "forward":
                self.travel = float(self.motion["forward_mps"]) * elapsed  # the pose is only updated after
                reason = self.safety_reason()
                if reason:
                    break
            time.sleep(0.04)
        self.stop()
        self.travel = 0.0
        elapsed = time.monotonic() - started
        if self.stop_event.is_set():
            raise Abort("stopped")
        self.dead_reckon(kind, elapsed)
        return elapsed, reason

    def turn_rate(self, kind: str) -> float:
        """Turn rate (deg/s) of "left" or "right": car_nav.json, scaled by what the last turns really did."""
        return float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"]) * self.turn_scale[kind]

    def learn_turn(self, fix_yaw: float) -> None:
        """A tag fix right after a turn says how far it really turned: correct the rate used for the next turns.

        The calibrated rate is for one battery state; a fresh or a tired battery turns faster or slower. Only
        turns of 25 deg or more teach (a small one is as noisy as the fix), and a rate moves half way to what
        was seen, within 0.7 to 1.4 times the calibration.
        """
        if self.last_turn is None:
            return
        kind, before, commanded = self.last_turn
        self.last_turn = None
        if abs(commanded) < math.radians(25.0):
            return
        done = wrap_rad(fix_yaw - before)
        ratio = done / commanded
        if not 0.4 < ratio < 1.8:  # turned the wrong way or not at all: not this rate's fault
            return
        scale = min(max(self.turn_scale[kind] * (0.5 + 0.5 * ratio), 0.7), 1.4)
        if abs(scale - self.turn_scale[kind]) >= 0.03:
            print(f"    turn {kind}: asked {math.degrees(commanded):+.0f}deg, did {math.degrees(done):+.0f}deg; "
                  f"rate now {self.turn_rate(kind) * scale / self.turn_scale[kind]:.0f}deg/s", flush=True)
        self.turn_scale[kind] = scale

    def dead_reckon(self, kind: str, elapsed: float):
        if self.pose is None or elapsed <= 0.0:
            return
        if self.use_filter and self.kalman.ready:
            amount = {"forward": float(self.motion["forward_mps"]), "reverse": float(self.motion["forward_mps"]),
                      "strafe_left": self.strafe_mps, "strafe_right": self.strafe_mps,
                      "left": math.radians(self.turn_rate("left")),
                      "right": math.radians(self.turn_rate("right"))}[kind] * elapsed
            self.kalman.sync(self.pose.x, self.pose.y, self.pose.yaw)
            self.kalman.predict(kind, amount)
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
            self.pose.yaw = wrap_rad(self.pose.yaw + math.radians(self.turn_rate("left") * elapsed))
        elif kind == "right":
            self.pose.yaw = wrap_rad(self.pose.yaw - math.radians(self.turn_rate("right") * elapsed))
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

    def turn_by(self, err: float, min_s: float | None = None):
        """Turn in place, first shuffling along the heading if a corner would hit a wall."""
        p, margin = self.pose, self.wall_limit()
        kind = "left" if err > 0 else "right"
        rate = self.turn_rate(kind)
        min_s = self.args.min_turn_s if min_s is None else min_s
        duration = min(max(math.degrees(abs(err)) / rate, min_s), self.args.max_turn_s)
        turn = math.copysign(math.radians(rate * duration), err)
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
        self.last_turn = (kind, self.pose.yaw, turn)
        step, side = shift
        if step:
            print(f"    wall too close to turn; {'forward' if step > 0 else 'reverse'} {abs(step) * 100:.0f}cm first",
                  flush=True)
            self.move("forward" if step > 0 else "reverse", abs(step) / float(self.motion["forward_mps"]))
        elif side:
            print(f"    wall too close to turn; strafe {'left' if side > 0 else 'right'} {abs(side) * 100:.0f}cm first",
                  flush=True)
            self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
        self.last_turn = (kind, self.pose.yaw, turn)
        self.move(kind, duration)

    def search(self) -> Pose2D | None:
        self.lane_check = False
        steps = int(math.ceil(360.0 / self.args.search_step_deg))
        for i in range(steps):
            print(f"  search {i + 1}/{steps}: turning left {self.args.search_step_deg:.0f}deg", flush=True)
            if self.pose is None:
                self.move("left", self.args.search_step_deg / self.turn_rate("left"))
            else:
                self.turn_by(math.radians(self.args.search_step_deg))
            fix = self.localize()
            if fix is not None:
                return fix
        return None

    def wait_clear(self, reason: str):
        print(f"  blocked ({reason}); reversing 8cm", flush=True)
        self.move("reverse", 0.08 / float(self.motion["forward_mps"]))
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

    def save_map(self):
        if not self.arena.get("arena_size_m"):
            return
        img = draw_map(self.arena, route=self.route, trail=self.trail, pose=self.pose, size_px=640)
        if self.home is not None:
            w_m, h_m = (float(v) for v in self.arena["arena_size_m"])
            margin, scale = 30, (640 - 2 * 30) / max(w_m, h_m)
            img_h = int(h_m * scale) + 2 * margin
            hx = int(round(margin + self.home.x * scale))
            hy = int(round(img_h - margin - self.home.y * scale))
            cv2.drawMarker(img, (hx, hy), (0, 140, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
        self.obstacles.draw(img)
        cv2.imwrite(str(self.debug_dir / "trajectory.png"), img)
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
        return (prev[0] + ux * ahead, prev[1] + uy * ahead), along - length, cross

    def go_to(self, index: int, target: tuple[float, float], prev: tuple[float, float] | None = None):
        blind = 0
        while not self.stop_event.is_set():
            dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
            dist = math.hypot(dx, dy)
            aim, past, cross = self.aim_point(prev, target)
            ax, ay = aim[0] - self.pose.x, aim[1] - self.pose.y
            err = wrap_rad(math.atan2(ay, ax) - self.pose.yaw)
            err_target = wrap_rad(math.atan2(dy, dx) - self.pose.yaw)
            print(f"wp{index} ({target[0]:.2f},{target[1]:.2f}) dist={dist:.2f}m "
                  f"err={math.degrees(err):+.0f}deg off-line={cross * 100:+.0f}cm "
                  f"pose {self.pose.text()}", flush=True)
            overshot = dist < 2.5 * self.args.arrive_m and abs(err_target) > math.pi / 2
            passed = prev is not None and past >= -self.args.arrive_m and abs(cross) < 2.0 * self.args.arrive_m
            if dist <= self.args.arrive_m or overshot or passed:
                print(f"reached wp{index}", flush=True)
                return
            if abs(err) > math.radians(self.args.align_deg):
                self.turn_by(err)
            else:
                segment = self.safe_step(min(math.hypot(ax, ay), dist, self.args.max_segment_m))
                if segment < 0.02:
                    back = -self.safe_step(-0.06)
                    if abs(err) > math.radians(3.0):
                        print("    wall ahead; straightening first", flush=True)
                        self.turn_by(err)
                    elif back >= 0.02:
                        print(f"    wall ahead; reverse {back * 100:.0f}cm", flush=True)
                        self.move("reverse", back / float(self.motion["forward_mps"]))
                    else:
                        raise Abort(f"wall ahead too close to reach wp{index}; move the route inward")
                    blind = self.refresh_pose(blind)
                    continue
                _elapsed, reason = self.move("forward", segment / float(self.motion["forward_mps"]))
                if reason:
                    self.wait_clear(reason)
            blind = self.refresh_pose(blind)
            self.record()
            self.save_map()

    def follow(self):
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
        measured = None
        if self.pose is not None:
            first, self.pose = self.pose, None
            fixes = [first] + [f for f in (self.localize() for _ in range(4)) if f is not None]
            ref = fixes[0].yaw
            measured = Pose2D(st.median(f.x for f in fixes), st.median(f.y for f in fixes),
                              wrap_rad(ref + st.median(wrap_rad(f.yaw - ref) for f in fixes)))
            self.pose = measured
        if known is None:
            if measured is None:
                raise Abort("--return-home needs a tag fix at the start pose; place the car so a camera "
                            "sees two tags (or one closer than 0.8 m), or set \"home\" in arena_map.json")
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

    def strafe(self, left: float, ahead_cap: bool = True) -> bool:
        """Mecanum sideways move by `left` m (negative = right); learns the strafe speed from the next fix.

        `ahead_cap=False` for going around an obstacle: the ultrasonic then reads the obstacle itself,
        right in front, and the cap below would forbid any strafe at all.
        """
        side = math.copysign(min(abs(left), 0.06), left)
        # Strafing along the lane swings the nose or the tail sideways into a wall the pose can
        # easily underestimate, so cap that direction with the ultrasonic reading.
        if self.lane_yaw is not None and ahead_cap:
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
        elapsed, _ = self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
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
        if abs(side) < 0.03:
            return True
        speed = min(max(abs(moved) / elapsed, 0.5 * self.strafe_base), 2.0 * self.strafe_base)
        self.strafe_mps = 0.5 * self.strafe_mps + 0.5 * speed
        print(f"    strafed {moved * 100:+.0f}cm (asked {side * 100:+.0f}cm); strafe speed now "
              f"{self.strafe_mps:.3f} m/s", flush=True)
        return True

    # ---- obstacle detour ------------------------------------------------

    def wall_hit(self, x: float, y: float, angle: float) -> tuple[float, str]:
        """Distance from (x, y) along `angle` to the arena wall it points at, and that wall: "x" for
        x=0 / x=w, "y" for y=0 / y=h."""
        w, h = (float(v) for v in self.arena["arena_size_m"])
        dx, dy = math.cos(angle), math.sin(angle)
        hits = [(t, axis) for t, axis in ((((w - x) / dx) if dx > 1e-6 else None, "x"),
                                         ((-x / dx) if dx < -1e-6 else None, "x"),
                                         (((h - y) / dy) if dy > 1e-6 else None, "y"),
                                         ((-y / dy) if dy < -1e-6 else None, "y"))
                if t is not None and t >= 0.0]
        return min(hits) if hits else (float("inf"), "")

    def wall_distance(self, x: float, y: float, angle: float) -> float:
        """Distance from (x, y) along `angle` to the arena wall it points at."""
        return self.wall_hit(x, y, angle)[0]

    def aim(self, pan_deg: float):
        """Turn the gimbal to `pan_deg` (+ = left) through its calibrated angles and wait until it is there."""
        from servo import servo_pan_for_bearing

        self.servo.set_pan(servo_pan_for_bearing(self.home_pan, self.right_90_pan, pan_deg, self.left_90_pan))
        if self.stop_event.wait(0.4):
            raise Abort("stopped")

    def look_around(self, pans=CAM_PANS) -> tuple[np.ndarray, np.ndarray]:
        """One camera look from where the car stands: (obstacle points, wall points), car frame.

        The gimbal turns to each angle, the foot line of the foam is found in a fresh frame and put on
        the floor (foot_cam.py). Points on the arena walls are split off (the pose says where they are);
        the rest go into the obstacle map, which draw_map shows.
        """
        views = self.scanner.scan(self.aim, pans)
        parts = [v.points[~np.isnan(v.points[:, 0])] for v in views] + [v.close for v in views if len(v.close)]
        points = np.vstack(parts) if parts else np.zeros((0, 2))
        obstacle, wall = split_walls(points, self.pose, self.arena["arena_size_m"])
        for block in self.obstacles.add(obstacle, wall, self.pose):
            if block["hits"] == 1:
                print(f"    new obstacle {self.obstacles.describe(block)} (pose {self.pose.text()})", flush=True)
        self.last_views = views
        self.save_look(views)
        self.log_look(views)
        return obstacle, wall

    def log_look(self, views) -> None:
        """Every look kept for replay: the frames, and the pose they were taken at (debug_dir/looks)."""
        if not self.args.save_looks or not any(v.frame is not None for v in views):
            return
        self.look_no += 1
        folder = self.debug_dir / "looks"
        folder.mkdir(exist_ok=True)
        for v in views:
            if v.frame is not None:
                cv2.imwrite(str(folder / f"{self.look_no:03d}_pan{v.pan:+03.0f}.jpg"), v.frame)
        p = self.pose
        with (folder / "looks.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"n": self.look_no, "pose": [round(p.x, 4), round(p.y, 4), round(math.degrees(p.yaw), 2)],
                                "pans": [v.pan for v in views], "time": time.strftime("%H:%M:%S")}) + "\n")

    def save_look(self, views) -> None:
        """The last look's frames with the foot lines on them: debug_dir/last_look.jpg."""
        frames = [v for v in views if v.frame is not None]
        if not frames or not hasattr(self.scanner, "cam"):
            return
        tiles = []
        for v in frames:
            tile = cv2.resize(self.scanner.cam.draw(v), (320, 240))
            cv2.putText(tile, f"pan {v.pan:+.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(tile)
        cv2.imwrite(str(self.debug_dir / "last_look.jpg"), np.hstack(tiles))

    def camera_gap(self, reason: str | None = None, exclude: np.ndarray | None = None) -> float | None:
        """Gap (m, from the ultrasonic face) to the nearest thing in the car's path that is not the wall.

        The camera looks at the path ahead (obstacle_map.path_gap: the body's width plus PATH_MARGIN_M,
        from the nose on); the ultrasonic straight ahead and a front IR stop count too. The pose says
        where the wall ahead is: what the camera sees there is the wall, whatever the filter said.
        """
        if self.sensors is None or self.pose is None:
            return None
        nose, half = self.body_half[0], self.body_half[1] + PATH_MARGIN_M
        obstacle, _wall = self.look_around()
        if exclude is not None and len(exclude) and len(obstacle):
            # the block being gone around is still beside-ahead of the car, only a few cm clear of its path
            world = to_world(obstacle, self.pose)
            near = np.hypot(world[:, None, 0] - exclude[None, :, 0], world[:, None, 1] - exclude[None, :, 1]).min(axis=1)
            obstacle = obstacle[near > 0.10]
        ahead = path_gap(obstacle, nose, half)
        gap = None if ahead is None else ahead + (nose - SONIC_AHEAD_M)
        readings = [r for r in (self.sensors.read_sonic() for _ in range(3)) if r is not None]
        if readings:
            sonic = st.median(readings) / 100.0
            gap = sonic if gap is None else min(gap, sonic)
        if reason == "ir_front":
            gap = min(gap if gap is not None else 1.0, 0.08)
        if gap is not None and self.wall_ahead() - gap <= WALL_ECHO_M:
            gap = None  # the wall ahead; the pose is off by up to this much after a stretch without tags
        return gap

    def beyond_spot(self, gap: float | None, target) -> bool:
        """True when the obstacle `gap` ahead (from the ultrasonic face) lies past `target`: the body fits at it."""
        if gap is None:
            return False
        p = self.pose
        ahead = (target[0] - p.x) * math.cos(p.yaw) + (target[1] - p.y) * math.sin(p.yaw)
        return gap + SONIC_AHEAD_M - self.body_half[0] > ahead + 0.02

    def snap_to_wall(self):
        """At a lane end the wall is dead ahead, and the ultrasonic reads the true distance to it.

        Dead reckoning drifts along the lane (5 % of the distance, and tags are often too far to fix it);
        the wall does not move. Heading square to a wall and a reading within 30 cm that is within 12 cm of
        what the pose predicts: move the pose along the heading to agree with the ultrasonic.
        """
        if self.sensors is None or self.pose is None:
            return
        p = self.pose
        if max(abs(math.cos(p.yaw)), abs(math.sin(p.yaw))) < 0.95:
            return
        readings = [r for r in (self.sensors.read_sonic() for _ in range(3)) if r is not None]
        if not readings:
            return
        measured = st.median(readings) / 100.0
        err = self.wall_ahead() - measured
        if measured > 0.30 or abs(err) > 0.12:
            return
        if self.use_filter and self.kalman.ready:
            self.kalman.sync(p.x, p.y, p.yaw)
            if self.kalman.update_wall(SONIC_AHEAD_M, 0.0, measured):
                moved = math.hypot(self.kalman.x[0] - p.x, self.kalman.x[1] - p.y)
                p.x, p.y, p.yaw = (float(v) for v in self.kalman.x)
                print(f"    wall {measured * 100:.0f}cm ahead; the filter moved the pose {moved * 100:.0f}cm -> "
                      f"{p.text()}", flush=True)
            return
        p.x += err * math.cos(p.yaw)
        p.y += err * math.sin(p.yaw)
        print(f"    wall {measured * 100:.0f}cm ahead; pose moved {err * 100:+.0f}cm along the heading -> {p.text()}",
              flush=True)

    def obstacle_ahead(self, reason: str | None = None, edges: bool = True) -> float | None:
        """Gap (m, from the ultrasonic face) to something in the car's path that is not the wall, or None.

        The pose says where the wall ahead should be; a reading 10 cm or more short of it is an
        obstacle. The beam straight ahead misses a block that covers only the edge of the path (a
        block standing out from the wall reaches only the wall-side 6 cm of the lane), and the corner
        IR does not see dark foam, so with `edges` the ultrasonic also looks EDGE_PAN_DEG to each side: a
        return inside the body's width (plus 2 cm) counts. A slanted block can bounce the beam away,
        so a front IR stop counts as one 8 cm off.
        """
        if self.scanner is not None:
            return self.camera_gap(reason)
        if self.sensors is None or self.pose is None:
            return None
        readings = [r for r in (self.sensors.read_sonic() for _ in range(3)) if r is not None]
        gap = st.median(readings) / 100.0 if readings else None
        if reason == "ir_front":
            gap = min(gap if gap is not None else 1.0, 0.08)
        if gap is not None and self.wall_ahead() - gap <= 0.10:
            gap = None  # the wall ahead (the IR also lights at the end of every lane)
        if edges and self.servo is not None:
            for pan in (EDGE_PAN_DEG, -EDGE_PAN_DEG):
                reading, wall = self.side_range(pan)
                if reading is None or abs(reading - wall) < 0.06:
                    continue  # nothing, or the wall that beam points at
                reach = SONIC_ARM_M + reading
                side = reach * math.sin(math.radians(EDGE_PAN_DEG))
                if side <= self.body_half[1] + 0.02:
                    edge_gap = SONIC_PIVOT_M + reach * math.cos(math.radians(pan)) - SONIC_AHEAD_M
                    gap = edge_gap if gap is None else min(gap, edge_gap)
        return gap

    def face(self, yaw: float, tol_deg: float = 6.0):
        err = wrap_rad(yaw - self.pose.yaw)
        if abs(err) > math.radians(tol_deg):
            self.turn_by(err, min_s=0.05)

    def side_range(self, pan: float) -> tuple[float | None, float]:
        """Ultrasonic turned to `pan` (deg, +left) and back: (reading m or None, distance to the wall there)."""
        try:
            cm = self.look(pan)
        finally:
            if self.servo is not None:
                self.servo.set_pan(self.home_pan)
                if self.stop_event.wait(0.35):  # facing ahead again before anything drives
                    raise Abort("stopped")
        p, a = self.pose, self.pose.yaw + math.radians(pan)
        ox = p.x + SONIC_PIVOT_M * math.cos(p.yaw) + SONIC_ARM_M * math.cos(a)
        oy = p.y + SONIC_PIVOT_M * math.sin(p.yaw) + SONIC_ARM_M * math.sin(a)
        return (None if cm is None else cm / 100.0), self.wall_distance(ox, oy, a)

    def outer_corridor(self, out: float) -> float | None:
        """How far the car can drive ahead before its outer side meets something, or None if nothing.

        Neither the forward ultrasonic (centre line) nor the corner IR (7 cm out, 8 cm range) covers the
        last few cm of the body's outer side, where a neighbouring wall's block can stand. The ultrasonic
        turned 15 and 30 deg to that side does: a return within the body width (plus 3 cm) on that side
        is in the path.
        """
        limit = None
        for pan in (15.0 * out, 30.0 * out):
            reading, _wall = self.side_range(pan)
            if reading is None:
                continue
            reach = SONIC_ARM_M + reading
            ahead = SONIC_PIVOT_M + reach * math.cos(math.radians(pan))
            side = reach * math.sin(math.radians(abs(pan)))
            if side <= self.body_half[1] + 0.03:
                room = max(ahead - self.body_half[0] - 0.02, 0.0)
                limit = room if limit is None else min(limit, room)
        return limit

    def detour(self, target: tuple[float, float, float], gap: float):
        """Go around an obstacle (see `_detour`); tag headings are trusted meanwhile.

        A mecanum strafe turns the car a little (the wheels are not equally strong). The lane rule that
        refuses a tag heading far from the lane is there for far tags that flip; here the car really
        can be 25 deg off after a long strafe, and strafing on at the lane heading drives it diagonally.
        """
        self.detouring = True
        try:
            self._detour(target, gap)
        finally:
            self.detouring = False

    def _detour(self, target: tuple[float, float, float], gap: float):
        """Go around an obstacle in a --hug lane with mecanum strafes, keeping the lane heading.

        1. Strafe toward the arena middle until the ultrasonic no longer sees it, then half the body
           width more: the beam runs down the car's centre line.
        2. Drive past it. Its far side is found with the ultrasonic turned to the wall side (a block
           is 10 or 20 cm deep); then the tail, which trails the ultrasonic by 26 cm, must clear it.
        3. Strafe back onto the lane, checking the side IR on the way.
        Beside the block the car does not turn: the corners would swing into it.
        """
        yaw = target[2]
        w, h = (float(v) for v in self.arena["arena_size_m"])
        c, s = math.cos(yaw), math.sin(yaw)
        middle = -(w / 2 - self.pose.x) * s + (h / 2 - self.pose.y) * c
        out = 1.0 if middle > 0 else -1.0  # +1: the arena middle is to the left of the lane
        side_name = "left" if out > 0 else "right"
        wall_side = "right" if out > 0 else "left"
        print(f"  obstacle {gap * 100:.0f}cm ahead in the lane; going around it on the {side_name}", flush=True)

        def off_lane() -> float:
            dx, dy = self.pose.x - target[0], self.pose.y - target[1]
            return out * (-dx * s + dy * c)

        def along() -> float:
            return (self.pose.x - target[0]) * c + (self.pose.y - target[1]) * s

        out_keys = ("wheel_left_front", "wheel_left_rear") if out > 0 else ("wheel_right_front", "wheel_right_rear")

        def side_blocked() -> bool:
            return self.sensors is not None and any(self.sensors.read_ir().get(k) for k in out_keys)

        def sidestep(distance: float):
            for _ in range(8):
                left = distance - off_lane()
                if abs(left) < 0.01:
                    return
                if side_blocked():
                    print(f"    {side_name} IR lit; staying {off_lane() * 100:.0f}cm off the lane", flush=True)
                    return
                # Short steps: an overshoot here swings the body further into the middle of the arena.
                if not self.strafe(out * math.copysign(min(abs(left), 0.03), left), ahead_cap=False):
                    raise Abort("detour: a wall blocks the strafe")
                self.record()
                self.save_map()

        passing = np.zeros((0, 2))  # world points of the block(s) this detour goes around

        def forward_to(goal: float) -> bool:
            """Drive along the lane to `goal`; False when a sensor or wall stopped it short."""
            for _ in range(8):
                self.face(yaw, 3.0)  # 10 deg over a 50 cm pass is 9 cm sideways
                step = self.safe_step(min(self.args.max_segment_m, goal - along()))
                if step < 0.01:
                    break
                if self.scanner is not None:
                    # Out beside the obstacle the car is no longer on its lane: look before each step, another
                    # block (e.g. one standing out from the next wall) may be in the way of the shifted path.
                    ahead_gap = self.camera_gap(exclude=passing)
                    if ahead_gap is not None:
                        step = min(step, ahead_gap - 0.12, CAM_STEP_M)
                        if step < 0.02:
                            print(f"    another obstacle {ahead_gap * 100:.0f}cm ahead while passing; stopping", flush=True)
                            break
                _elapsed, reason = self.move("forward", step / float(self.motion["forward_mps"]))
                self.refresh_pose(0, full_turn=False)
                self.record()
                self.save_map()
                if reason:
                    print(f"    stopped while passing ({reason})", flush=True)
                    break
            return along() >= goal - 0.02

        # 0. Right up against it a strafe would scrape the nose along it: back off to 6 cm first.
        if gap < 0.05:
            back = self.safe_step(-(0.06 - gap))
            if back < -0.005:
                print(f"    only {gap * 100:.0f}cm in front; reversing {-back * 100:.0f}cm first", flush=True)
                self.move("reverse", -back / float(self.motion["forward_mps"]))
                gap -= back

        start, detected = off_lane(), along()
        if self.scanner is not None:
            # 1. The camera gives the obstacle's edge: strafe exactly as far as the body needs to clear it by
            # CLEAR_MARGIN_M, look again, and repeat if the first estimate was short.
            nose, half = self.body_half[0], self.body_half[1]
            for attempt in range(6):
                self.face(yaw, 2.0)  # a strafe turns the car; the path ahead is judged square to the lane
                obstacle, _wall = self.look_around()
                if attempt == 0:
                    for block in self.obstacles.blocks:
                        if block.get("last_scan") == self.obstacles.scans:
                            print(f"    {self.obstacles.describe(block, target)}", flush=True)
                            passing = np.vstack([passing, block["points"]])
                need = shift_to_clear(obstacle, out, nose, half, CLEAR_MARGIN_M)
                if need <= 0.012:
                    break
                if off_lane() - start + need > self.args.detour_max_m:
                    raise Abort(f"detour: the path is blocked up to {(off_lane() - start + need) * 100:.0f}cm "
                                f"off the lane; no way around")
                # The strafe sweeps the side of the car; the camera looks ahead. Ask the ultrasonic, turned to
                # the side the car is about to move to, whether something stands there.
                reading, wall = self.side_range(90.0 * out)
                room = reading + SONIC_ARM_M if reading is not None else None  # from the car's centre line
                if room is not None and reading < wall - 0.08 and room < half + need + 0.05:
                    raise Abort(f"detour: something {room * 100:.0f}cm to the {side_name}, where the strafe of "
                                f"{need * 100:.0f}cm would go; no way around")
                print(f"    path blocked; strafing {need * 100:.0f}cm {side_name}", flush=True)
                goal = off_lane() + need
                while goal - off_lane() > 0.01:
                    if side_blocked():
                        raise Abort(f"detour: something on the {side_name} side too; no way around")
                    self.face(yaw, 2.0)
                    if not self.strafe(out * min(goal - off_lane(), 0.04), ahead_cap=False):
                        raise Abort("detour: a wall blocks the strafe")
                    self.record()
                    self.save_map()
            else:
                raise Abort("detour: the camera still sees the path blocked after 6 looks")
            print(f"    path clear, {(off_lane() - start) * 100:.0f}cm off the lane", flush=True)
        else:
            ahead = gap
            while True:
                if off_lane() - start > self.args.detour_max_m:
                    raise Abort(f"detour: obstacle still ahead {self.args.detour_max_m * 100:.0f}cm off the lane")
                if side_blocked():
                    raise Abort(f"detour: something on the {side_name} side too; no way around")
                self.face(yaw)
                # Once the beam starts to slip off the obstacle's edge, smaller steps: every cm out here
                # is a cm the car swings into the middle of the arena.
                if not self.strafe(out * (0.04 if ahead is None or ahead < gap + 0.03 else 0.03), ahead_cap=False):
                    raise Abort("detour: a wall blocks the strafe")
                self.record()
                self.save_map()
                ahead = self.obstacle_ahead(edges=False)  # the side beams would keep finding this very block
                print(f"    {(off_lane() - start) * 100:.0f}cm out; ahead "
                      f"{'clear' if ahead is None else f'{ahead * 100:.0f}cm'}", flush=True)
                if ahead is None or ahead > gap + 0.15:
                    break
            # Clear means the edge of the beam has left the obstacle, so its edge is at least that far off the
            # centre line. SONIC_HALF_DEG is narrower than the real cone, which only ever makes this cautious.
            need = off_lane() - gap * math.tan(math.radians(SONIC_HALF_DEG)) + self.body_half[1] + 0.02
            sidestep(need)
            if off_lane() < need - 0.015:
                raise Abort(f"detour: only {off_lane() * 100:.0f}cm of the {need * 100:.0f}cm needed beside the "
                            "obstacle is free; no way around")

        # 2. Past it. `near` is its near face along the lane; the side beam is SONIC_PIVOT_M ahead of the
        # centre and a few cm wide, so "clear" there puts the far face at most 4 cm behind the beam.
        near = detected + SONIC_AHEAD_M + gap
        pan = -90.0 * out
        far = None
        probe = near + 0.10 + 0.04
        while far is None and probe <= near + BLOCK_LONG_M + 0.08:
            if not forward_to(probe - SONIC_PIVOT_M):
                break
            reading, wall = self.side_range(pan)
            beside = reading is not None and reading < wall - 0.08
            print(f"    {wall_side} side at +{(along() + SONIC_PIVOT_M - near) * 100:.0f}cm past its near face: "
                  f"ultrasonic {'-' if reading is None else f'{reading * 100:.0f}cm'}, wall {wall * 100:.0f}cm"
                  f"{' (still beside it)' if beside else ''}", flush=True)
            if not beside:
                far = along() + SONIC_PIVOT_M - 0.04
            probe += 0.05
        if far is None:
            far = near + BLOCK_LONG_M
            print(f"    could not see its far side; assuming {BLOCK_LONG_M * 100:.0f}cm deep", flush=True)
        clear = far + self.body_half[0] + PASS_MARGIN_M
        room = self.outer_corridor(out) if self.scanner is None else None  # the camera covers it
        if room is not None and clear - along() > room:
            raise Abort(f"detour: something {room * 100:.0f}cm ahead on the {side_name} edge of the path; "
                        f"passing needs {(clear - along()) * 100:.0f}cm")
        if not forward_to(clear):
            raise Abort(f"detour: stopped {(clear - along()) * 100:.0f}cm short of clearing the obstacle; "
                        "strafing back would hit it")

        # 3. Back onto the lane, whichever side of it the car is on now; a side IR lighting up while still
        # well off the lane is the obstacle, so go a little further first.
        ir_keys = ("wheel_right_front", "wheel_right_rear") if out > 0 else ("wheel_left_front", "wheel_left_rear")
        for _ in range(14):
            err = off_lane()  # + = toward the arena middle
            if abs(err) <= self.args.lane_tol_m:
                break
            p = self.pose
            wall_gap = self.wall_distance(p.x, p.y, p.yaw - out * math.pi / 2.0) - self.body_half[1]  # body edge to wall
            if err > 0.10 and wall_gap > 0.12 and self.sensors is not None \
                    and any(self.sensors.read_ir().get(k) for k in ir_keys):
                print(f"    {wall_side} IR sees the obstacle; forward 6cm before strafing back", flush=True)
                if not forward_to(along() + 0.06):
                    raise Abort("detour: obstacle beside the car and no room to go further")
                continue
            self.face(yaw, 2.0)
            if not self.strafe(-out * max(min(err, 0.06), -0.06), ahead_cap=False):
                break
            self.record()
            self.save_map()
        print(f"    back on the lane ({off_lane() * 100:+.0f}cm)", flush=True)

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
            # Heading already right and the spot close: strafe and drive to it rather than turning beside
            # the walls (the start, joining the first lane from home, is like this).
            square = abs(dyaw) <= math.radians(20.0) and math.hypot(ahead, left) < 0.30
            if not near and not in_lane and not square:
                # Far and not already lined up along the target heading: face the spot and drive there.
                if abs(bearing) > math.radians(self.args.align_deg):
                    self.turn_by(bearing)
                else:
                    step = self.safe_step(min(math.hypot(dx, dy), self.args.max_segment_m))
                    if step < 0.01:
                        self.turn_by(bearing if abs(bearing) > math.radians(3.0) else dyaw, min_s=0.05)
                    else:
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
                detour = self.args.hug and self.args.detour and not parking and step > 0
                if detour:
                    # The side looks see the edges of the path only up to ~24 cm ahead, and the beam straight
                    # ahead misses a block that covers just an edge: never drive past what was looked at.
                    step = min(step, CAM_STEP_M if self.scanner is not None else SCAN_STEP_M)
                    gap = self.obstacle_ahead()
                    short = self.beyond_spot(gap, target)
                    if short:
                        gap = None  # the spot this move goes to is short of it: no need to get past it
                    approach = self.args.approach_cm / 100.0 + (self.body_half[0] - SONIC_AHEAD_M)
                    if gap is not None:
                        # After a closing move the obstacle is inside the camera's blind zone: it reports the
                        # nearest floor it sees, not the distance. Believe the dead-reckoned gap instead, and
                        # go around: closing in again on that report would drive into it.
                        blind = self.scanner is not None and gap <= BLIND_GAP_M
                        if gap <= approach + 0.03 or blind:
                            # Seen only in the blind zone and never closed in on: the real gap is somewhere
                            # between 0 and what is reported, so no closing in either; go around from here.
                            true_gap = self.closed_in if (blind and self.closed_in is not None) else gap
                            self.closed_in = None
                            self.detour(target, true_gap)
                            continue
                        step = min(step, gap - approach)
                        print(f"    obstacle {(gap + SONIC_AHEAD_M - self.body_half[0]) * 100:.0f}cm ahead of the nose; "
                              f"closing to {self.args.approach_cm:.0f}cm", flush=True)
                        self.closed_in = gap - step
                    else:
                        self.closed_in = None
                if abs(step) < 0.01:
                    print("    wall reached; taking this as the spot", flush=True)
                    if detour:
                        self.snap_to_wall()
                    return False
                kind = "forward" if step > 0 else "reverse"
                if detour and (gap is not None or short):
                    self.closing_cm = (approach - 0.03) * 100.0
                    if gap is not None:
                        self.closing_ir_after = max(step - 0.04, 0.0)
                try:
                    _elapsed, reason = self.move(kind, abs(step) / float(self.motion["forward_mps"]))
                finally:
                    self.closing_cm = None
                    self.closing_ir_after = None
                if reason:
                    self.closed_in = None
                if reason and detour:
                    gap = self.obstacle_ahead(reason)
                    if self.beyond_spot(gap, target):
                        # Stopped by what lies past the spot (not in the way of it): this is as close as we go.
                        print(f"    obstacle past the spot ({reason}); taking this as the spot", flush=True)
                        return False
                    if gap is not None:
                        self.detour(target, gap)
                        continue
                    if self.wall_ahead() < 0.20:
                        # Backing off the wall here could reverse into an obstacle just passed.
                        print(f"    wall ahead ({reason}); taking this as the spot", flush=True)
                        self.snap_to_wall()
                        return False
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
        # A turn in place sweeps a circle of the half diagonal; `spot` keeps that circle off both walls.
        spot = math.hypot(hl, hw) + self.args.wall_margin_m + 0.02
        lane = spot if self.args.hug_gap_m is None else hw + self.args.hug_gap_m
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
        anchor = self.home or self.pose

        def cost(q):
            return math.hypot(q[0] - anchor.x, q[1] - anchor.y) + 0.1 * abs(wrap_rad(q[2] - anchor.yaw))

        k = min((i for i in range(n) if i % 4 in (0, 1)), key=lambda i: cost(poses[i]))
        hw = self.body_half[1]
        lane = abs(poses[0][1])  # the first lane runs along the y=0 wall
        print(f"hug lane: body {(lane - hw) * 100:.0f}cm from the walls; starting at pose {k}", flush=True)
        while True:
            for j in range(n + 1):
                i = (k + j) % n
                kind = ("lane start", "lane end", "corner", "turned")[i % 4]
                self.go_pose(f"p{i} {kind}", poses[i], 0.04, 4.0)
            if self.stop_event.is_set():
                raise Abort("stopped")
            if self.home is not None:
                self.dock()
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


def main():
    args = parse_args()
    follower = RouteFollower(args)
    signal.signal(signal.SIGINT, follower.request_stop)
    signal.signal(signal.SIGTERM, follower.request_stop)
    if not follower.motion.get("calibrated") and not args.calibrate_motion:
        print("NOTE: car_nav.json motion is not calibrated; run --calibrate-motion --arm first.", flush=True)
    try:
        follower.start()
        if args.calibrate_motion:
            follower.calibrate_motion()
        else:
            follower.follow()
    except Abort as exc:
        print(f"ABORT: {exc}", flush=True)
    finally:
        follower.close()
        print(f"Stopped; motors off. Map: {follower.debug_dir / 'trajectory.png'}", flush=True)


if __name__ == "__main__":
    main()
