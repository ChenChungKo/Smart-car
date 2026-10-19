#!/usr/bin/env python3
"""Simulated hardware under route_follow.py: the route logic runs unchanged, only the car is fake.

`SimFollower` is a RouteFollower whose motors, ultrasonic, IR, gimbal servo and tag localization
come from a world backend. Two backends exist:
- route_sim.py: flat 2D geometry, runs anywhere (also on the Pi), no extra software.
- webots/controllers/route_follow_webots: the Webots scene, its sonar/IR rays and 3D view.

A backend provides (arena metres, origin at the bottom-left corner, yaw 0 = +x):
    truth -> [x, y, yaw]        the real pose
    place(x, y, yaw)            move the car there (kinematic: a mecanum strafe needs no wheel physics)
    tick(dt)                    let `dt` seconds of simulated time pass
    sonic_m(pan_deg) -> m|None  the ultrasonic, turned `pan_deg` (positive = left)
    ir() -> dict                the IR sensors under route_follow's names, 1 = something within 8 cm
    hits(x, y, yaw) -> str|None what the body would touch at that pose (an obstacle or a wall)

The motion is the calibrated motion in car_nav.json times `MotionError`, so dead reckoning is a
little wrong, as on the real car. Localization is the true pose plus noise.
"""

from __future__ import annotations

import json
import math
import random
from types import SimpleNamespace
from dataclasses import dataclass

from aruco_localizer import Pose2D
from route_follow import Abort, RouteFollower

IR_RANGE_M = 0.08
# Where each IR sensor sits (car frame, m) and looks (deg, + = left), under route_follow's names.
IR_LAYOUT = {
    "front": (0.145, 0.0, 0.0),
    "front_left": (0.145, 0.07, 0.0),
    "front_right": (0.145, -0.07, 0.0),
    "wheel_left_front": (0.08, 0.09, 90.0),
    "wheel_left_rear": (-0.08, 0.09, 90.0),
    "wheel_right_front": (0.08, -0.09, -90.0),
    "wheel_right_rear": (-0.08, -0.09, -90.0),
}


@dataclass
class MotionError:
    """How the simulated car differs from car_nav.json (1.0 = exactly as calibrated)."""
    forward: float = 1.05
    strafe: float = 0.90
    turn: float = 0.97
    drift_deg_per_m: float = 4.0   # the real car drifts left while driving straight
    strafe_yaw_deg_per_m: float = 0.0  # a strafe to the left turns the car this much clockwise (unequal wheels)
    corner_bias: tuple = (0.0, 0.0, 0.0)  # tag fixes at home itself (6 cm) are off by this (x m, y m, yaw deg): close tags
    fix_outlier: float = 0.0       # share of tag fixes that are wrong: 12-20 cm off, heading up to 35 deg off
    realistic_fixes: bool = True   # tag noise grows with the nearest tag's distance (else 1 cm / 1 deg)
    fix_loss: float = 0.0          # share of tag looks that give no fix (far tags only): position is dead-reckoned
    fix_sigma_m: float = 0.01      # tag fix noise
    fix_sigma_deg: float = 1.0


def body_outline(x: float, y: float, yaw: float, half_l: float = 0.145, half_w: float = 0.09, n: int = 30):
    """Points around the body rectangle, for contact checks."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for k in range(n):
        t = -1.0 + 2.0 * k / (n - 1)
        for lx, ly in ((half_l * t, half_w), (half_l * t, -half_w), (half_l, half_w * t), (-half_l, half_w * t)):
            pts.append((x + lx * c - ly * s, y + lx * s + ly * c))
    return pts


class _Clock:
    """Stands in for route_follow's stop_event: waiting lets simulated time pass."""

    def __init__(self, world):
        self.world = world
        self.flag = False

    def wait(self, seconds: float | None = None) -> bool:
        if seconds:
            self.world.tick(seconds)
        return self.flag

    def set(self):
        self.flag = True

    def is_set(self) -> bool:
        return self.flag


class _Servo:
    def __init__(self, car: "SimFollower"):
        self.car = car

    def set_servo_pwm(self, channel: str, value: float):
        if channel == "0":  # pan: route_follow sends home_pan - pan_deg
            self.car.pan_deg = self.car.home_pan - float(value)

    def set_pan(self, value: float):
        self.set_servo_pwm("0", value)


class _Sensors:
    def __init__(self, car: "SimFollower"):
        self.car = car

    def read_sonic(self) -> float | None:
        m = self.car.world.sonic_m(self.car.pan_deg)
        return None if m is None else round(m * 100.0 + random.gauss(0.0, 0.5), 1)

    def read_ir(self) -> dict[str, int]:
        return self.car.world.ir()

    def close(self):
        pass


class SimScanner:
    """foot_cam.CameraScanner on a simulated world: the foot-line finder's report, by ray casting."""

    def __init__(self, world):
        self.world = world

    def scan(self, aim, pans):
        views = []
        for pan in pans:
            aim(pan)
            views.append(self.world.foot_view(pan))
        return views

    def close(self):
        pass


class SimFollower(RouteFollower):
    def __init__(self, args, world, error: MotionError | None = None, max_moves: int = 2000):
        super().__init__(args)
        self.world = world
        self.error = error or MotionError()
        self.max_moves = max_moves
        self.moves = 0
        self.pose_errors: list[tuple[float, float]] = []
        self.pan_deg = 0.0
        self.stop_event = _Clock(world)
        self.sensors = _Sensors(self)
        self.servo = _Servo(self)
        if args.hug and args.detour and args.camera_obstacles and hasattr(world, "foot_view"):
            self.scanner = SimScanner(world)

    # ---- hardware ---------------------------------------------------------

    def start(self):
        self.motor = object()  # motion is on; drive()/stop() below never touch it
        print("SIMULATION: motors, ultrasonic, IR and tag fixes come from the simulated world", flush=True)

    def drive(self, pwm):
        pass

    def stop(self):
        pass

    def close(self):
        self.save_map()
        if self.obstacles.blocks:
            (self.debug_dir / "obstacles.json").write_text(
                json.dumps(self.obstacles.as_dict(), indent=2) + "\n", encoding="utf-8")
        if self.writer is not None:
            self.writer.release()

    def look(self, pan_deg: float):
        self.pan_deg = pan_deg
        self.world.tick(0.35)  # the servo turns
        return self.sensors.read_sonic()

    def localize(self, timeout: float | None = None) -> Pose2D | None:
        self.world.tick(self.args.settle_s)
        x, y, yaw = self.world.truth
        e = self.error
        if e.fix_loss and random.random() < e.fix_loss:
            return self.accept_fix(None)
        home = self.arena["home"]
        if any(e.corner_bias) and math.hypot(x - home[0], y - home[1]) < 0.06:
            x, y, yaw = x + e.corner_bias[0], y + e.corner_bias[1], yaw + math.radians(e.corner_bias[2])
        if not e.realistic_fixes:  # equal noise at any distance, as a fix of two tags 80 cm away
            fixes = [SimpleNamespace(camera="front", tag_id=k, distance=0.8 + 0.1 * k) for k in range(2)]
            return self.accept_fix(Pose2D(x + random.gauss(0.0, e.fix_sigma_m), y + random.gauss(0.0, e.fix_sigma_m),
                                          yaw + math.radians(random.gauss(0.0, e.fix_sigma_deg)), fixes=fixes))
        d = random.uniform(0.45, 1.35)                      # nearest tag of the fix
        n = 3 if d < 0.9 else 2                             # tags in it
        sxy, syaw = 0.02 + 0.06 * d, math.radians(1.0 + 3.0 * d)
        px, py, pyaw = x + random.gauss(0.0, sxy), y + random.gauss(0.0, sxy), yaw + random.gauss(0.0, syaw)
        if e.fix_outlier and random.random() < e.fix_outlier:
            jump, angle = random.uniform(0.12, 0.20), random.uniform(0.0, 2.0 * math.pi)
            px, py = px + jump * math.cos(angle), py + jump * math.sin(angle)
            pyaw += math.radians(random.uniform(-35.0, 35.0))
        fixes = [SimpleNamespace(camera="front", tag_id=k, distance=d + 0.12 * k) for k in range(n)]
        return self.accept_fix(Pose2D(px, py, pyaw, fixes=fixes))

    def move(self, kind: str, duration: float) -> tuple[float, str | None]:
        """The same timed motion as on the car: forward stops early for a safety reason."""
        m, e = self.motion, self.error
        if kind not in ("left", "right"):
            self.last_turn = None
        if kind not in ("strafe_left", "strafe_right"):
            self.strafe_acc = None
        vx, vy, wz = {
            "forward": (m["forward_mps"] * e.forward, 0.0, 0.0),
            "reverse": (-m["forward_mps"] * e.forward, 0.0, 0.0),
            "strafe_left": (0.0, self.strafe_base * e.strafe, 0.0),
            "strafe_right": (0.0, -self.strafe_base * e.strafe, 0.0),
            "left": (0.0, 0.0, math.radians(m["turn_left_dps"]) * e.turn),
            "right": (0.0, 0.0, -math.radians(m["turn_right_dps"]) * e.turn),
        }[kind]
        if self.strafe_flip:  # route_follow learned the strafe the wrong way round: the wiring follows
            vy = -vy
        dt, t, reason = 0.02, 0.0, None
        while t < duration - 1e-9:
            x, y, yaw = self.world.truth
            yaw += wz * dt + (math.radians(e.drift_deg_per_m) * vx * dt if kind == "forward" else 0.0)
            yaw -= math.radians(e.strafe_yaw_deg_per_m) * vy * dt
            x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
            y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
            what = self.world.hits(x, y, yaw)
            if what:
                raise Abort(f"SIM COLLISION: {kind} into {what} at x={x:.3f} y={y:.3f}")
            self.world.place(x, y, yaw)
            self.world.tick(dt)
            t += dt
            if kind == "forward" and int(round(t / dt)) % 2 == 0:
                self.travel = m["forward_mps"] * t
                reason = self.safety_reason()
                if reason:
                    break
        self.travel = 0.0
        self.moves += 1
        if self.moves > self.max_moves:
            raise Abort(f"SIM: more than {self.max_moves} moves; stuck")
        self.dead_reckon(kind, t)
        if self.pose is not None:  # how wrong the car's idea of where it is, against the truth
            tx, ty, tyaw = self.world.truth
            self.pose_errors.append((math.hypot(self.pose.x - tx, self.pose.y - ty),
                                     abs(math.degrees(math.atan2(math.sin(self.pose.yaw - tyaw),
                                                                 math.cos(self.pose.yaw - tyaw))))))
        return t, reason
