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

import math
import random
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


class SimFollower(RouteFollower):
    def __init__(self, args, world, error: MotionError | None = None, max_moves: int = 2000):
        super().__init__(args)
        self.world = world
        self.error = error or MotionError()
        self.max_moves = max_moves
        self.moves = 0
        self.pan_deg = 0.0
        self.stop_event = _Clock(world)
        self.sensors = _Sensors(self)
        self.servo = _Servo(self)

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
        fix = Pose2D(x + random.gauss(0.0, e.fix_sigma_m), y + random.gauss(0.0, e.fix_sigma_m),
                     yaw + math.radians(random.gauss(0.0, e.fix_sigma_deg)))
        return self.accept_fix(fix)

    def move(self, kind: str, duration: float) -> tuple[float, str | None]:
        """The same timed motion as on the car: forward stops early for a safety reason."""
        m, e = self.motion, self.error
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
        return t, reason
