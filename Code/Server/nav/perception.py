"""Camera, ultrasonic and IMU together: the gimbal looks where the car is going and every look updates the map.

The camera finds where things meet the floor; the IMU's tilt corrects that projection and its turn rate rejects smeared
pictures; the ultrasonic on the same gimbal measures the distance directly and clears what is not there (a reflection on
the floor has no echo).
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from foot_cam import FootCamera, GimbalFeed
from nav.world import WorldGrid
from route_common import SONIC_ARM_M, SONIC_PIVOT_M
from servo import servo_pan_for_bearing

SETTLE_S = 0.16           # the gimbal has stopped moving and shaking
MAX_TURN_DPS = 25.0       # faster than this a picture is smeared
SCAN_OFFSETS = (0.0, -22.0, 22.0, 0.0, -60.0)   # relative to the direction of travel: ahead, the sides of the path, the wall side


class Eyes(threading.Thread):
    def __init__(self, car, world: WorldGrid, pose_at):
        super().__init__(daemon=True)
        self.car, self.world, self.pose_at = car, world, pose_at
        self.cam = FootCamera(car.car)
        self.cam.geometry["pitch_deg"] = float(car.car.get("nav", {}).get("gimbal_pitch_deg", self.cam.geometry["pitch_deg"]))
        self.feed = GimbalFeed().start()
        self.feed.wait_ready()
        self.heading_deg = 0.0     # direction of travel relative to the nose, set by the controller
        self.halt = threading.Event()
        self.ranges: list = []     # (t, pan_deg, m) for the pose filter
        self.looks = 0
        self.smeared = 0

    def look(self, pan: float) -> None:
        car = self.car
        pan = max(-90.0, min(90.0, pan))
        car.servo.set_pan(servo_pan_for_bearing(car.home_pan, car.right_90_pan, pan, car.left_90_pan))
        time.sleep(SETTLE_S)
        t = time.monotonic()
        echo = car.sensors.read_sonic()
        frame = self.feed.fresh(t, 0.5)
        pose = self.pose_at(t)
        if pose is None:
            return
        px, py, yaw = pose
        sample = car.gyro.imu.latest() if car.gyro is not None else None
        tilt = (sample.rpy_deg[1], sample.rpy_deg[0]) if sample is not None else (0.0, 0.0)   # nose up, right side up
        c, s = math.cos(yaw), math.sin(yaw)
        world = lambda p: np.column_stack([px + c * p[:, 0] - s * p[:, 1], py + s * p[:, 0] + c * p[:, 1]])
        a = math.radians(pan)
        if echo is not None:
            ox, oy = SONIC_PIVOT_M + SONIC_ARM_M * math.cos(a), SONIC_ARM_M * math.sin(a)
            self.world.add_sonic(px + c * ox - s * oy, py + s * ox + c * oy, yaw + a, echo / 100.0)
            self.ranges.append((t, pan, echo / 100.0))
        if sample is not None and abs(sample.gyro_dps[2]) > MAX_TURN_DPS:
            self.smeared += 1
            return
        view = self.cam.observe(frame, pan, tilt=tilt)
        points = view.points[~np.isnan(view.points[:, 0])]
        m = self.cam.mount(pan)
        self.world.add_foot((px + c * m["x_m"] - s * m["y_m"], py + s * m["x_m"] + c * m["y_m"]), world(points), world(view.close))
        self.looks += 1

    def run(self) -> None:
        while not self.halt.is_set():
            for off in SCAN_OFFSETS:
                if self.halt.is_set():
                    return
                try:
                    self.look(self.heading_deg + off)
                except Exception as exc:  # noqa: BLE001  a missed look must not stop the car's control loop
                    print(f"look failed: {exc}", flush=True)

    def close(self) -> None:
        self.halt.set()
        if self.is_alive():
            self.join(timeout=2.0)
        self.feed.close()
