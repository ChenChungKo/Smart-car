"""From a path to wheel commands, and the one safety rule."""
from __future__ import annotations

import math

import numpy as np

from pose_kalman import wrap
from patrol.world import FREE, OCCUPIED, RES_M

LOOKAHEAD_M = 0.08
SPIN_ONLY_DEG = 35.0          # further than this off the path's heading the car only turns
TURN_GAIN, MAX_TURN = 2.0, math.radians(45.0)
MIN_SPEED = 0.09              # m/s: below this the wheels may not turn
STOP_S, STOP_EXTRA_M = 0.4, 0.03   # the body must be on seen-free floor this far ahead: speed x STOP_S + STOP_EXTRA_M
BODY_SLACK_M = 0.01
PWM_LIMIT = 2000
HEADING_OK = math.radians(1.0)   # standing: a heading this close is left alone (the wheels cannot make smaller turns)


class Follower:
    def __init__(self, speed: float):
        self.speed = speed

    def command(self, pose, path, goal):
        """(vx, vy, omega) in the body frame towards a point a little further along the path, and what it aimed at."""
        x, y, yaw = pose
        d = np.hypot(np.array([q[0] for q in path]) - x, np.array([q[1] for q in path]) - y)
        i = k = int(np.argmin(d))
        along = 0.0
        while k < len(path) - 1 and along < LOOKAHEAD_M:
            along += math.hypot(path[k + 1][0] - path[k][0], path[k + 1][1] - path[k][1])
            k += 1
        tx, ty, tyaw = path[k]
        if k == len(path) - 1:
            tx, ty = goal.x, goal.y
            tyaw = goal.yaw if goal.yaw is not None else tyaw
        err = wrap(tyaw - yaw)
        om = max(-MAX_TURN, min(MAX_TURN, TURN_GAIN * err))
        ex, ey = tx - x, ty - y
        dist = math.hypot(ex, ey)
        to_goal = math.hypot(goal.x - x, goal.y - y)
        scale = max(0.0, 1.0 - abs(math.degrees(err)) / SPIN_ONLY_DEG)
        speed = 0.0
        if scale > 0.0 and dist > 0.008:
            speed = max(MIN_SPEED, min(self.speed * scale, 2.0 * to_goal))
        wx, wy = (speed * ex / dist, speed * ey / dist) if speed > 0.0 else (0.0, 0.0)
        if speed == 0.0 and abs(err) < HEADING_OK:
            om = 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        info = dict(path_index=i, target=[tx, ty, tyaw], off_path_m=float(d[i]), heading_err_deg=math.degrees(err))
        return (c * wx + s * wy, -s * wx + c * wy, om), info


class Safety:
    """The body, where it will be within its stopping distance, must stand on floor that was seen free."""

    def __init__(self, half_l: float, half_w: float, arena_m):
        a = np.arange(-half_l - BODY_SLACK_M, half_l + BODY_SLACK_M + 1e-6, RES_M * 0.8)
        b = np.arange(-half_w - BODY_SLACK_M, half_w + BODY_SLACK_M + 1e-6, RES_M * 0.8)
        A, B = np.meshgrid(a, b, indexing="ij")
        self.body = np.column_stack([A.ravel(), B.ravel()])
        self.arena = arena_m

    def cells(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = x + c * self.body[:, 0] - s * self.body[:, 1], y + s * self.body[:, 0] + c * self.body[:, 1]
        return np.rint(px / RES_M).astype(int), np.rint(py / RES_M).astype(int), px, py

    def check(self, log: np.ndarray, pose, cmd):
        """(ok, detail). detail says what stands in the way and where, for the log."""
        x, y, yaw = pose
        vx, vy, om = cmd
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = c * vx - s * vy, s * vx + c * vy
        speed = math.hypot(wx, wy)
        if speed < 1e-4 and abs(om) < 1e-3:
            return True, dict(checked_m=0.0)
        ahead = speed * STOP_S + (STOP_EXTRA_M if speed > 1e-4 else 0.0)
        nx, ny = log.shape
        i0, j0, _px, _py = self.cells(x, y, yaw)
        here = np.zeros((nx, ny), bool)
        ok0 = (i0 >= 0) & (i0 < nx) & (j0 >= 0) & (j0 < ny)
        here[i0[ok0], j0[ok0]] = True
        ux, uy = (wx / speed, wy / speed) if speed > 1e-4 else (0.0, 0.0)
        for part in (0.5, 1.0):
            fx, fy, fyaw = x + ux * ahead * part, y + uy * ahead * part, yaw + om * STOP_S * part
            i, j, px, py = self.cells(fx, fy, fyaw)
            i, j = np.clip(i, 0, nx - 1), np.clip(j, 0, ny - 1)
            new = ~here[i, j]                     # where the body already stands is not in question
            wall = new & ((i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1))
            if wall.any():
                n = int(np.flatnonzero(wall)[0])
                return False, dict(kind="wall", at=[float(px[n]), float(py[n])], checked_m=ahead, points=int(wall.sum()))
            v = log[i, j]
            for kind, bad in (("occupied", new & (v > OCCUPIED)), ("unseen", new & (v >= FREE) & (v <= OCCUPIED))):
                if bad.any():
                    n = int(np.flatnonzero(bad)[0])
                    return False, dict(kind=kind, at=[float(px[n]), float(py[n])], log_odds=float(v[n]), checked_m=ahead, points=int(bad.sum()))
        return True, dict(checked_m=ahead)


class Wheels:
    def __init__(self, car, i2c_lock):
        self.car, self.i2c = car, i2c_lock
        self.vf, self.vs = float(car.motion["forward_mps"]), float(car.strafe_mps)
        self.fwd, self.side, self.turn = (np.array(car._pwm(k), float) for k in ("forward", "strafe_left", "left"))
        self.w_nom = math.radians(float(car.motion["turn_left_dps"]))
        curve = car.motion.get("turn_curve_pwm_dps")     # measured with the gyro: below about 500 the car does not turn at all
        self.curve = None if not curve else (np.array([c[1] for c in curve], float), np.array([c[0] for c in curve], float))
        self.pwm = [0, 0, 0, 0]

    def mix(self, vx: float, vy: float, om: float) -> list:
        """Wheel duties. Turning on the spot uses the measured curve (rate -> duty); added to a move, the turn is linear."""
        if self.curve is not None and abs(vx) < 1e-4 and abs(vy) < 1e-4:
            rate = abs(math.degrees(om))
            if rate < 0.5:
                return [0, 0, 0, 0]
            duty = float(np.interp(rate, self.curve[0], self.curve[1]))
            if rate < self.curve[0][1]:
                duty = float(self.curve[1][1])            # the slowest turn the wheels can make
            return [int(v) for v in np.sign(self.turn) * duty * (1 if om > 0 else -1)]
        w = (vx / self.vf) * self.fwd + (vy / self.vs) * self.side + (om / self.w_nom) * self.turn
        peak = float(np.abs(w).max())
        if peak > PWM_LIMIT:
            w *= PWM_LIMIT / peak
        return [int(v) for v in w]

    def send(self, vx: float, vy: float, om: float) -> list:
        pwm = self.mix(vx, vy, om)
        self.pwm = pwm
        if self.car.motor is not None:
            with self.i2c:
                if max(abs(v) for v in pwm) < 1:
                    self.car.stop()
                else:
                    self.car.drive(tuple(pwm))
        return pwm

    def stop(self) -> None:
        self.pwm = [0, 0, 0, 0]
        with self.i2c:
            self.car.stop()
