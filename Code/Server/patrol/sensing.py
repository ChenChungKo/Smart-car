"""The gimbal looks where the car is going: camera foot line (IMU tilt applied) and ultrasonic into the map, every look kept.

The same picture is searched for wall tags: the gimbal camera sees tags the fixed cameras do not (things in the arena hide
them), and where it is at each pan angle is known (gimbal_mount.py, car_nav.json "gimbal_tags")."""
from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np

from aruco_localizer import ArucoLocalizer
from foot_cam import HEIGHT, MAX_RANGE_M, FootCamera, GimbalFeed
from gimbal_mount import car_T_camera
from route_common import SONIC_ARM_M, SONIC_PIVOT_M
from servo import PAN_APPROACH, PAN_MIN, servo_pan_for_bearing
from vio_check import LATENCY_S

SETTLE_S = 0.10                     # after the gimbal has arrived
TRAVEL_S_PER_DEG = 0.14 / 60.0      # how long it takes to get there (MG90S, with slack)
MAX_TURN_DPS = 25.0                 # faster than this the picture is smeared: only the echo is used
SCAN_OFFSETS = (0.0, -25.0, 25.0, 0.0, -50.0, 50.0)   # about the direction of travel
FREE_RANGE_M = min(0.45, MAX_RANGE_M)


class Sonar(threading.Thread):
    """Reads the ultrasonic all the time on its own thread: with nothing in range the driver's read can wait for ever,
    and nothing else may wait with it. A reading counts only if it began after the moment asked for."""

    def __init__(self, sensors):
        super().__init__(daemon=True)
        self.sensors = sensors
        self.latest = (0.0, None)          # (when the reading began, cm or None)
        self.halt = threading.Event()
        self.start()

    def run(self) -> None:
        while not self.halt.is_set():
            began = time.monotonic()
            try:
                value = self.sensors.read_sonic()
            except Exception:  # noqa: BLE001
                value = None
            self.latest = (began, value)
            time.sleep(0.005)

    def read(self, after: float, timeout: float) -> float | None:
        """Metres of a reading begun after `after`, or None: no echo within the timeout (nothing in range)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            began, value = self.latest
            if began >= after:
                return None if value is None else value / 100.0
            time.sleep(0.005)
        return None


class Eyes(threading.Thread):
    def __init__(self, car, world, pose_at, i2c_lock, recorder):
        super().__init__(daemon=True)
        self.car, self.world, self.pose_at, self.i2c, self.rec = car, world, pose_at, i2c_lock, recorder
        self.cam = FootCamera(car.car)
        self.cam.geometry["pitch_deg"] = float(car.car.get("nav", {}).get("gimbal_pitch_deg", self.cam.geometry["pitch_deg"]))
        self.feed = GimbalFeed().start()
        self.feed.wait_ready()
        self.sonar = Sonar(car.sensors)
        self.loc = ArucoLocalizer(car.arena, car.car)      # its own: the front and rear cameras' thread has the car's
        self.gimbal_model = car.car.get("gimbal_tags")
        if self.gimbal_model is not None:
            self.loc.kd("gimbal", 640, 480)
        self.tag_rows: list = []     # tag poses of looks, not yet given to the estimator
        self.aim_deg = 0.0           # direction of travel relative to the nose (set by the navigator)
        self.halt = threading.Event()
        self.ranges: list = []       # (pan deg, m) not yet given to the estimator
        self.looks = self.smeared = 0
        self.last = None             # the last look, for the display
        self.pan_now = None
        self.pan_deg = None

    def point(self, pan_deg: float) -> float:
        """Turn the gimbal and wait until it has arrived."""
        car = self.car
        before = self.pan_deg
        pan_deg = max(-90.0, min(90.0, pan_deg))
        angle = servo_pan_for_bearing(car.home_pan, car.right_90_pan, pan_deg, car.left_90_pan)
        if angle != self.pan_now:
            if (self.pan_now is None or angle < self.pan_now) and angle - PAN_APPROACH >= PAN_MIN:
                with self.i2c:
                    car.servo.set_servo_pwm("0", angle - PAN_APPROACH)
                time.sleep(0.12)
            with self.i2c:
                car.servo.set_servo_pwm("0", angle)
            self.pan_now = car.servo.pan_now = angle
            time.sleep(abs(pan_deg - (before if before is not None else -pan_deg)) * TRAVEL_S_PER_DEG)
        self.pan_deg = pan_deg
        return pan_deg

    def tag_fixes(self, frame, pan: float) -> list:
        """The wall tags in a gimbal picture taken at this pan, each with the pose of the car it implies."""
        if self.gimbal_model is None:
            return []
        name = f"gimbal{pan:+.1f}"
        if name not in self.loc.cam_T_car:
            self.loc.base_kd[name] = self.loc.base_kd["gimbal"]
            self.loc.cam_T_car[name] = np.linalg.inv(car_T_camera(self.gimbal_model, pan))
        return [f for f in self.loc.observe(name, frame)[0] if f.usable]

    def locate(self, pans=(-90, -60, -30, 0, 30, 60, 90)) -> dict | None:
        """Standing: one pose from every tag the gimbal sees around and the front and rear cameras see (a joint fit)."""
        fixes, seen = [], {}
        for pan in pans:
            p = self.point(float(pan))
            time.sleep(SETTLE_S + 0.15)
            got = self.tag_fixes(self.feed.fresh(time.monotonic(), 1.0), p)
            fixes += got
            seen[f"gimbal {p:+.0f}"] = sorted(f.tag_id for f in got)
        for name, grabber in self.car.grabbers.items():
            frame = grabber.latest()[0]
            if frame is not None:
                got = [f for f in self.loc.observe(name, frame)[0] if f.usable]
                fixes += got
                seen[name] = sorted(f.tag_id for f in got)
        pose = self.loc.fuse(fixes, time.monotonic()) if fixes else None
        if pose is None:
            return dict(pose=None, seen=seen, why=self.loc.last_reject)
        return dict(pose=(pose.x, pose.y, pose.yaw), seen=seen, tags=len({f.tag_id for f in pose.fixes}), views=len(pose.fixes), rms_px=pose.rms_px)

    def echo(self, reads: int = 1, timeout: float = 0.3) -> float | None:
        """Median of `reads` fresh readings (m); None when there was no echo."""
        v = []
        for _ in range(reads):
            r = self.sonar.read(time.monotonic(), timeout)
            if r is not None:
                v.append(r)
        return float(np.median(v)) if v else None

    def look(self, pan_deg: float) -> dict:
        pan = self.point(pan_deg)
        time.sleep(SETTLE_S)
        t = time.monotonic()
        echo = self.echo()
        frame = self.feed.fresh(t, 0.6)
        px, py, yaw = self.pose_at(t)
        s = self.car.gyro.imu.latest()
        tilt = (s.rpy_deg[1], s.rpy_deg[0]) if s is not None else (0.0, 0.0)
        c, sn = math.cos(yaw), math.sin(yaw)
        to_world = lambda p: np.column_stack([px + c * p[:, 0] - sn * p[:, 1], py + sn * p[:, 0] + c * p[:, 1]]) if len(p) else np.zeros((0, 2))
        a = math.radians(pan)
        rec = dict(t=t, pan=pan, echo=echo, pose=[px, py, yaw], tilt=list(tilt), hits=0, free=0, close=0, used="echo only")
        if echo is not None:
            ox, oy = SONIC_PIVOT_M + SONIC_ARM_M * math.cos(a), SONIC_ARM_M * math.sin(a)
            self.world.add_sonic((px + c * ox - sn * oy, py + sn * ox + c * oy), yaw + a, echo)
            self.ranges.append((pan, echo))
        view = None
        if s is not None and abs(s.gyro_dps[2]) > MAX_TURN_DPS:
            self.smeared += 1
            rec["used"] = "echo only: turning too fast for the picture"
        else:
            fixes = self.tag_fixes(frame, pan)
            joint = self.loc.fuse(fixes, t) if fixes else None
            rec["tags"] = sorted(f.tag_id for f in fixes)
            if joint is not None:
                z = dict(x=joint.x, y=joint.y, yaw=joint.yaw, tags=len({f.tag_id for f in joint.fixes}), nearest=min(f.distance for f in joint.fixes))
                self.tag_rows.append(dict(t=t + LATENCY_S, camera="gimbal", pose=z))
                rec["tag_pose"] = [z["x"], z["y"], z["yaw"]]
            view = self.cam.observe(frame, pan, keep_frame=True, tilt=tilt)
            m = self.cam.mount(pan)
            cam_xy = np.array([m["x_m"], m["y_m"]])
            ok = view.usable & (view.rows < HEIGHT)
            rows = np.where(view.rows >= 0, view.rows, HEIGHT - 1)        # no foot in the column: the floor at the bottom gives the direction
            floor = self.cam.project(np.column_stack([view.cols[ok], rows[ok]]), pan, tilt)
            d = floor - cam_xy
            rng = np.hypot(d[:, 0], d[:, 1])
            good = ~np.isnan(rng)
            is_hit = good & (view.rows[ok] >= 0) & (rng < MAX_RANGE_M)
            hits = floor[is_hit]
            far = good & ~is_hit
            ends = cam_xy + d[far] / np.maximum(rng[far], 1e-6)[:, None] * FREE_RANGE_M
            origin = to_world(cam_xy.reshape(1, 2))[0]
            self.world.add_camera(origin, to_world(hits), to_world(ends), to_world(view.close))
            rec.update(hits=int(len(hits)), free=int(len(ends)), close=int(len(view.close)), used="camera and echo")
            self.looks += 1
        rec["file"] = self.rec.look(rec, frame)
        self.last = dict(rec, view=view, frame=frame)
        return rec

    def run(self) -> None:
        while not self.halt.is_set():
            for off in SCAN_OFFSETS:
                if self.halt.is_set():
                    return
                try:
                    self.look(self.aim_deg + off)
                except Exception as exc:  # noqa: BLE001  a missed look must not end the run
                    self.rec.event("look failed", error=str(exc))

    def take_tag_rows(self) -> list:
        out, self.tag_rows = self.tag_rows, []
        return out

    def take_ranges(self) -> list:
        out, self.ranges = self.ranges, []
        return out

    def picture(self) -> np.ndarray | None:
        last = self.last
        if last is None:
            return None
        if last["view"] is not None:
            return self.cam.draw(last["view"])
        return cv2.resize(last["frame"], (640, 480))

    def close(self) -> None:
        self.halt.set()
        self.sonar.halt.set()
        if self.is_alive():
            self.join(timeout=2.0)
        self.feed.close()
