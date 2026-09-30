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
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2

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
    p.add_argument("--front-ir-gate-cm", type=float, default=18.0)
    p.add_argument("--blocked-wait-s", type=float, default=6.0)
    p.add_argument("--min-battery-v", type=float, default=7.0)
    p.add_argument("--no-ir", action="store_true")
    p.add_argument("--no-sonic", action="store_true")
    p.set_defaults(active_high=False)
    p.add_argument("--ir-active-high", action="store_true", dest="active_high")
    p.add_argument("--debug-dir", default="route_follow_debug")
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
        self.pose: Pose2D | None = None
        self.trail: list[tuple[float, float, float]] = []
        self.stop_event = threading.Event()
        self.sonic_misses = 0
        self.front_risk = 0.0
        self.debug_dir = SERVER / args.debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ---- hardware -------------------------------------------------------

    def start(self):
        from servo import Servo

        try:
            self.servo = Servo()  # applies gimbal_home.json: ultrasonic straight ahead
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

    # ---- sensing --------------------------------------------------------

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
        if near and (ir["front"] or ir["front_left"] or ir["front_right"]):
            return "ir_front"
        frame, stamp = self.grabbers["front"].latest()
        if frame is not None and time.monotonic() - stamp < 1.0:
            mid = classic_corridor_score(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA))["mid"]
            self.front_risk = 0.4 * mid + 0.6 * self.front_risk
            # A far foam wall also fills the fisheye; only trust the camera near the wall.
            if self.front_risk >= self.args.near_score and (sonic is None or sonic < self.args.creep_cm):
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
            print(f"  tags seen but fit rejected (rms {self.localizer.last_rms_px:.1f}px); "
                  "run aruco_localizer.py --calibrate-pitch 20", flush=True)
        return pose

    # ---- motion ---------------------------------------------------------

    def _pwm(self, kind: str):
        fwd = int(self.motion["forward_pwm"])
        turn = int(self.motion["turn_pwm"])
        return {
            "forward": (fwd, fwd, fwd, fwd),
            "left": (-turn, -turn, turn, turn),
            "right": (turn, turn, -turn, -turn),
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
        if kind == "forward":
            step = float(self.motion["forward_mps"]) * elapsed
            self.pose.x += step * math.cos(self.pose.yaw)
            self.pose.y += step * math.sin(self.pose.yaw)
        elif kind == "left":
            self.pose.yaw = wrap_rad(self.pose.yaw + math.radians(float(self.motion["turn_left_dps"]) * elapsed))
        elif kind == "right":
            self.pose.yaw = wrap_rad(self.pose.yaw - math.radians(float(self.motion["turn_right_dps"]) * elapsed))
        self.pose.fixes = []

    def turn_by(self, err: float):
        kind = "left" if err > 0 else "right"
        rate = float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"])
        duration = min(max(math.degrees(abs(err)) / rate, self.args.min_turn_s), self.args.max_turn_s)
        self.move(kind, duration)

    def search(self) -> Pose2D | None:
        step_s = self.args.search_step_deg / float(self.motion["turn_left_dps"])
        steps = int(math.ceil(360.0 / self.args.search_step_deg))
        for i in range(steps):
            print(f"  search {i + 1}/{steps}: turning left {self.args.search_step_deg:.0f}deg", flush=True)
            self.move("left", step_s)
            fix = self.localize()
            if fix is not None:
                return fix
        return None

    def wait_clear(self, reason: str):
        print(f"  blocked ({reason}); waiting up to {self.args.blocked_wait_s:.0f}s", flush=True)
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
        img = draw_map(self.arena, route=self.route, trail=self.trail, pose=self.pose, size_px=700)
        cv2.imwrite(str(self.debug_dir / "trajectory.png"), img)

    def refresh_pose(self, blind: int) -> int:
        fix = self.localize()
        if fix is not None:
            self.pose = fix
            return 0
        blind += 1
        if blind > self.args.max_blind_segments or self.pose is None:
            print("  no tag for too long; searching", flush=True)
            fix = self.search()
            if fix is None:
                raise Abort("lost: no tag visible after a full turn")
            self.pose = fix
            return 0
        return blind

    def go_to(self, index: int, target: tuple[float, float]):
        blind = 0
        while not self.stop_event.is_set():
            dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
            dist = math.hypot(dx, dy)
            err = wrap_rad(math.atan2(dy, dx) - self.pose.yaw)
            print(f"wp{index} ({target[0]:.2f},{target[1]:.2f}) dist={dist:.2f}m "
                  f"err={math.degrees(err):+.0f}deg pose {self.pose.text()}", flush=True)
            overshot = dist < 2.5 * self.args.arrive_m and abs(err) > math.pi / 2
            if dist <= self.args.arrive_m or overshot:
                print(f"reached wp{index}", flush=True)
                return
            if abs(err) > math.radians(self.args.align_deg):
                self.turn_by(err)
            else:
                segment = min(dist, self.args.max_segment_m)
                _elapsed, reason = self.move("forward", segment / float(self.motion["forward_mps"]))
                if reason:
                    self.wait_clear(reason)
            blind = self.refresh_pose(blind)
            self.record()
            self.save_map()

    def follow(self):
        self.pose = self.localize(3.0)
        if self.pose is None:
            print("no tag at start; searching", flush=True)
            self.pose = self.search()
            if self.pose is None:
                raise Abort("no tag visible; check tags, lighting and arena_map.json")
        print(f"start pose {self.pose.text()}", flush=True)
        self.record()
        while True:
            for i, wp in enumerate(self.route):
                self.go_to(i, wp)
            if not self.args.repeat:
                break
        print("route finished", flush=True)

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
