"""Where the car is: gyro and commanded speed carry the pose, tag poses and wall echoes correct it (vio.py)."""
from __future__ import annotations

import collections
import math
import time

import numpy as np

from pose_kalman import wrap
from route_common import SONIC_ARM_M, SONIC_PIVOT_M
from vio import VioFilter
from vio_check import LATENCY_S, stand_pose

STEADY_POSES, STEADY_MIN_S, STEADY_MAX_S = 15, 1.5, 6.0
STAND_SIGMA_XY = 0.04
WEAK_SIGMA_XY = 0.08          # standing on single far tags only
WALL_SQUARE_DEG = 20.0        # an echo measures a wall only when the beam meets it this squarely
GIMBAL_LOOK_SCALE = 1.0       # one gimbal picture scatters 1.6 cm and 1.7 deg (measured standing), twice the fixed cameras
ECHO_SIGMA = (0.01, 0.04)     # m, and per metre of range: 1 cm close by, about 3 cm short at 65 cm (sweeps of 2026-11-09)
FLOOR_XY_M, FLOOR_YAW = 0.01, math.radians(0.5)   # the tags and echoes are off by this much the same way every time (calibration):
                                                   # many of them standing still must not make the pose surer than that


def tag_pose(row: dict):
    """The joint tag pose of a picture, or the pose from its one far tag (looser; tags=1 says so)."""
    return row["pose"] or row.get("weak")


class Estimator:
    def __init__(self, car, tags, arena_m, world):
        self.car, self.tags, self.arena, self.world = car, tags, arena_m, world
        self.f = None
        self.hist = collections.deque(maxlen=300)
        self.seen = 0
        self.last = self.prev_gyro = None
        self.cmd = (0.0, 0.0)
        self.omega = 0.0
        self.tag_used = self.tag_refused = self.wall_used = 0

    def measure_standing(self, side_echo_m: float | None, joint: dict | None = None):
        """(x, y, yaw) while standing: every tag pose of a few seconds, then the wall on the right by the ultrasonic."""
        if joint is not None and joint.get("pose") is not None:      # the gimbal's sweep with the fixed cameras: the best there is
            est = VioFilter(*joint["pose"])
            est.P[0, 0] = est.P[1, 1] = STAND_SIGMA_XY ** 2
            used = side_echo_m is not None and est.update_wall(SONIC_PIVOT_M, -math.pi / 2.0, side_echo_m + SONIC_ARM_M, self.arena, gate_m=0.08)
            return dict(x=float(est.x[0]), y=float(est.x[1]), yaw=float(est.x[2]), tags_xy=list(joint["pose"][:2]), side_echo_used=bool(used),
                        tags=f"{joint['tags']} tags in {joint['views']} views, gimbal sweep with the front and rear cameras", sigma_m=float(math.sqrt(max(est.P[0, 0], est.P[1, 1]))))
        t0 = time.monotonic()
        while time.monotonic() - t0 < STEADY_MAX_S:
            time.sleep(0.1)
            n = sum(1 for r in list(self.tags.rows) if r["t"] >= t0 and tag_pose(r))
            if n >= STEADY_POSES and time.monotonic() - t0 >= STEADY_MIN_S:
                break
        rows = [r for r in list(self.tags.rows) if r["t"] >= t0]
        strong = any(r["pose"] for r in rows)
        got = stand_pose(dict(tags=[dict(r, pose=r["pose"] if strong else tag_pose(r)) for r in rows]), dict(t0=t0, t1=time.monotonic()))
        if got is None:
            return None
        est = VioFilter(*got)
        est.P[0, 0] = est.P[1, 1] = (STAND_SIGMA_XY if strong else WEAK_SIGMA_XY) ** 2
        used = side_echo_m is not None and est.update_wall(SONIC_PIVOT_M, -math.pi / 2.0, side_echo_m + SONIC_ARM_M, self.arena, gate_m=0.08)
        return dict(x=float(est.x[0]), y=float(est.x[1]), yaw=float(est.x[2]), tags_xy=[got[0], got[1]], side_echo_used=bool(used),
                    tags="several" if strong else "one far tag only", sigma_m=float(math.sqrt(max(est.P[0, 0], est.P[1, 1]))))

    def begin(self, x: float, y: float, yaw: float, keep_scale: bool = False, sigma_m: float | None = None) -> None:
        scale = self.f.x[3:].copy() if (keep_scale and self.f is not None) else None
        self.f = VioFilter(x, y, yaw)
        if sigma_m is not None:
            self.f.P[0, 0] = self.f.P[1, 1] = sigma_m ** 2
        if scale is not None:
            self.f.x[3:] = scale
        self.seen = len(self.tags.rows)
        self.last, self.prev_gyro = time.monotonic(), self.car.gyro.peek()
        self.cmd = (0.0, 0.0)
        self.hist.append((self.last, x, y, yaw))

    def advance(self, sent_vx: float, sent_vy: float, ranges: list, more_tags: tuple = ()) -> tuple[float, float, float]:
        """One step: predict with what the wheels were told since the last step, then the measurements that came in."""
        t = time.monotonic()
        g = self.car.gyro.peek()
        if g is None:
            raise RuntimeError("IMU silent")
        dt, self.last = t - self.last, t
        self.omega = (g - self.prev_gyro) / max(dt, 1e-3)
        self.prev_gyro = g
        self.f.predict(dt, self.cmd[0], self.cmd[1], self.omega)
        self.hist.append((t, float(self.f.x[0]), float(self.f.x[1]), float(self.f.x[2])))
        speed = math.hypot(*self.cmd)
        rows = list(more_tags)                    # the gimbal's looks
        while self.seen < len(self.tags.rows):    # the front and rear cameras
            rows.append(self.tags.rows[self.seen])
            self.seen += 1
        for r in sorted(rows, key=lambda q: q["t"]):
            z = r["pose"]          # only a joint fit of several tags: one far tag is off by several cm the same way every
            weak = False             # picture (here 6 cm), and many of those would pull the pose there
            if z:
                then = self.pose_at(r["t"] - LATENCY_S)
                ok = self.f.update_tags(z["x"] + self.f.x[0] - then[0], z["y"] + self.f.x[1] - then[1],
                                        wrap(z["yaw"] + float(self.f.x[2]) - then[2]), z["tags"], z["nearest"], speed,
                                        GIMBAL_LOOK_SCALE if r["camera"] == "gimbal" else None)
                self.tag_used += ok
                self.tag_refused += not ok
        for pan_deg, range_m in ranges:
            self.wall_used += self.wall_echo(pan_deg, range_m)
        P = self.f.P
        P[0, 0], P[1, 1], P[2, 2] = max(P[0, 0], FLOOR_XY_M ** 2), max(P[1, 1], FLOOR_XY_M ** 2), max(P[2, 2], FLOOR_YAW ** 2)
        self.cmd = (sent_vx, sent_vy)
        return self.pose()

    def wall_echo(self, pan_deg: float, range_m: float) -> bool:
        """Use an echo as a distance to a wall: only square onto the wall, and with nothing on the map in between."""
        x, y, yaw = self.pose()
        a = yaw + math.radians(pan_deg)
        if abs((math.degrees(a) + 45.0) % 90.0 - 45.0) > WALL_SQUARE_DEG:
            return False
        origin = (x + SONIC_PIVOT_M * math.cos(yaw), y + SONIC_PIVOT_M * math.sin(yaw))
        if self.world.something_before(origin, a, range_m + SONIC_ARM_M):
            return False
        return bool(self.f.update_wall(SONIC_PIVOT_M, math.radians(pan_deg), range_m + SONIC_ARM_M, self.arena,
                                       sigma_m=ECHO_SIGMA[0] + ECHO_SIGMA[1] * range_m))

    def pose(self) -> tuple[float, float, float]:
        return float(self.f.x[0]), float(self.f.x[1]), float(self.f.x[2])

    def pose_at(self, t: float) -> tuple[float, float, float]:
        h = min(list(self.hist), key=lambda v: abs(v[0] - t))
        return h[1], h[2], h[3]

    def sigma_m(self) -> float:
        return float(math.sqrt(max(self.f.P[0, 0], self.f.P[1, 1])))
