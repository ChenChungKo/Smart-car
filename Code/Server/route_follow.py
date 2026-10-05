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
from full_car_test import estimate_battery_percent
from safe_surround_cruise import SafetySensors, read_battery_voltage
from vision_detector import classic_corridor_score

SERVER = Path(__file__).resolve().parent


def parse_args():
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
    p.add_argument("--hug-gap-m", type=float, default=0.08, help="Body side to wall gap in --hug lanes.")
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
        if not args.calibrate_motion and args.route not in routes:
            raise SystemExit(f"route '{args.route}' not in {args.map}; have {sorted(routes)}")
        self.route = [tuple(map(float, wp)) for wp in routes.get(args.route, [])]
        self.grabbers = {}
        self.sensors = None
        self.motor = None
        self.servo = None
        self.home_pan, self.home_tilt = 90, 90
        self.pose: Pose2D | None = None
        self.trail: list[tuple[float, float, float]] = []
        self.stop_event = threading.Event()
        self.sonic_misses = 0
        self.front_risk = 0.0
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
        from servo import Servo

        try:
            from servo import load_gimbal_home

            self.servo = Servo()  # applies gimbal_home.json: ultrasonic straight ahead
            self.home_pan, self.home_tilt = load_gimbal_home()
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
        self.servo.set_servo_pwm("0", int(round(self.home_pan - pan_deg)))
        if self.stop_event.wait(0.35):
            raise Abort("stopped")
        return self.sensors.read_sonic()

    def sweep_correct(self):
        """Replace the sideways part of the pose with what the ultrasonic sweep measures."""
        found = self.sweep()
        if self.pose is None or len(found) < 2:
            return
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
                self.servo.set_servo_pwm("0", self.home_pan)
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
        if sonic is not None and sonic < self.args.stop_cm:
            return f"sonic_{sonic:.0f}cm"
        ir = self.sensors.read_ir()
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
            strong = len({(f.camera, f.tag_id) for f in pose.fixes}) >= 3 or \
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

    def _pwm(self, kind: str):
        fwd = int(self.motion["forward_pwm"])
        turn = int(self.motion["turn_pwm"])
        side = int(self.motion.get("strafe_pwm", 1200)) * (-1 if self.strafe_flip else 1)
        return {
            "forward": (fwd, fwd, fwd, fwd),
            "reverse": (-fwd, -fwd, -fwd, -fwd),
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
        self.drive(self._pwm(kind))
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - started
            if elapsed >= duration:
                break
            if kind == "forward":
                reason = self.safety_reason()
                if reason:
                    break
            time.sleep(0.04)
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
        rate = float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"])
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
        step, side = shift
        if step:
            print(f"    wall too close to turn; {'forward' if step > 0 else 'reverse'} {abs(step) * 100:.0f}cm first",
                  flush=True)
            self.move("forward" if step > 0 else "reverse", abs(step) / float(self.motion["forward_mps"]))
        elif side:
            print(f"    wall too close to turn; strafe {'left' if side > 0 else 'right'} {abs(side) * 100:.0f}cm first",
                  flush=True)
            self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
        self.move(kind, duration)

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
            if not near and not in_lane:
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
                if abs(step) < 0.01:
                    print("    wall reached; taking this as the spot", flush=True)
                    return False
                kind = "forward" if step > 0 else "reverse"
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
        anchor = self.home or self.pose

        def cost(q):
            return math.hypot(q[0] - anchor.x, q[1] - anchor.y) + 0.1 * abs(wrap_rad(q[2] - anchor.yaw))

        k = min((i for i in range(n) if i % 4 in (0, 1)), key=lambda i: cost(poses[i]))
        print(f"hug lane {self.args.hug_gap_m * 100:.0f}cm from the walls; starting at pose {k}", flush=True)
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
