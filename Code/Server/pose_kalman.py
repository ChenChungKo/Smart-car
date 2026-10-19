#!/usr/bin/env python3
"""Planar pose filter: dead reckoning corrected by tag fixes and by the wall ahead.

State x, y, yaw (arena frame) with a covariance P. route_follow.py dead-reckons the pose after every move and
calls `predict`, which says how much each kind of move is to be trusted; a tag fix is a measurement with its own
covariance, and the filter takes the right share of it. This replaces a hard accept-or-ignore rule, which both let
a bad far-tag fix in and threw away a good one:

- A mecanum strafe turns the car (the wheels are not equally strong), up to ~1 deg per cm. The uncertainty of
  the heading therefore grows with every strafe; a tag fix 28 deg off the dead-reckoned heading after a 25 cm
  strafe is then 2 sigma, and believed, where a fixed "20 deg is a jump" rule threw it away.
- A fix is accepted when its Mahalanobis distance from the prediction is small. Outliers (a single far tag read
  the wrong way) are rejected; the caller believes a run of outliers that agree with each other (`start`).
- After every update the body is kept inside the arena: a 29 cm car whose centre is reported 5 cm from the wall
  is moved back until the whole body fits.
- The ultrasonic straight at a wall measures the distance along the heading very well; used only when it
  agrees with the wall the pose predicts (foam, which is closer, does not move the pose).
"""

from __future__ import annotations

import math

import numpy as np

CHI2_3DOF_99 = 11.34   # innovation gate: 99 % of good fixes are inside


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def ray_to_wall(ox: float, oy: float, angle: float, width: float, height: float) -> float:
    dx, dy = math.cos(angle), math.sin(angle)
    hits = []
    if dx > 1e-6:
        hits.append((width - ox) / dx)
    if dx < -1e-6:
        hits.append(-ox / dx)
    if dy > 1e-6:
        hits.append((height - oy) / dy)
    if dy < -1e-6:
        hits.append(-oy / dy)
    return max(0.0, min(hits)) if hits else 0.0


def tag_noise(tag_count: int, nearest_m: float) -> tuple[float, float]:
    """(sigma_xy m, sigma_yaw rad) of a tag fix: tighter with several tags, looser the farther the nearest is."""
    far = nearest_m > 0.85 or tag_count < 2
    sigma_xy = 0.03 if tag_count >= 2 and nearest_m <= 0.85 else (0.12 if far else 0.05)
    sigma_yaw = math.radians(4.0 if tag_count >= 2 else 12.0)
    return sigma_xy, sigma_yaw


class PoseKalman:
    def __init__(self, width_m: float, height_m: float, body_half: tuple[float, float],
                 strafe_yaw_deg_per_cm: float = 0.6):
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.body_half = body_half
        self.strafe_yaw = math.radians(strafe_yaw_deg_per_cm)  # per cm of strafe, 1 sigma
        self.ready = False
        self.x = np.zeros(3)
        self.P = np.diag([0.08 ** 2, 0.08 ** 2, math.radians(10.0) ** 2])

    # ---- state ------------------------------------------------------------

    def start(self, x: float, y: float, yaw: float, tag_count: int = 2, nearest_m: float = 0.8) -> None:
        """Begin (or begin again) at a measured pose."""
        sigma_xy, sigma_yaw = tag_noise(tag_count, nearest_m)
        self.x = np.array([x, y, wrap(yaw)], dtype=float)
        self.P = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        self.ready = True
        self._project()

    def sync(self, x: float, y: float, yaw: float) -> None:
        """Take the mean from the caller's pose (it moves it too: sweeps, parking); keep the covariance."""
        self.x = np.array([x, y, wrap(yaw)], dtype=float)
        if not self.ready:
            self.ready = True

    def sigma(self) -> tuple[float, float]:
        """(position sigma m, heading sigma deg) now."""
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1]))), math.degrees(math.sqrt(self.P[2, 2]))

    def _project(self) -> bool:
        """Pull the centre in until every body corner is inside the arena."""
        half_l, half_w = self.body_half
        yaw = float(self.x[2])
        c, s = math.cos(yaw), math.sin(yaw)
        extent_x = abs(c) * half_l + abs(s) * half_w
        extent_y = abs(s) * half_l + abs(c) * half_w
        before = self.x[:2].copy()
        self.x[0] = min(max(float(self.x[0]), extent_x), self.width_m - extent_x)
        self.x[1] = min(max(float(self.x[1]), extent_y), self.height_m - extent_y)
        clipped = float(np.hypot(self.x[0] - before[0], self.x[1] - before[1])) > 1e-4
        if clipped:
            self.P[0, 0] = max(float(self.P[0, 0]), 0.08 ** 2)
            self.P[1, 1] = max(float(self.P[1, 1]), 0.08 ** 2)
        return clipped

    # ---- prediction: how far each kind of move is to be trusted ------------

    def predict(self, kind: str, amount: float) -> None:
        """Grow the covariance after a move: `amount` is metres for drives and strafes, radians for turns."""
        yaw = float(self.x[2])
        amount = abs(amount)
        if kind in ("forward", "reverse"):
            along, across, turn = 0.04 * amount + 0.004, 0.025 * amount + 0.003, math.radians(2.0) * amount + 0.002
        elif kind in ("strafe_left", "strafe_right"):
            along, across = 0.06 * amount + 0.003, 0.10 * amount + 0.004
            turn = self.strafe_yaw * amount * 100.0 + 0.002
        else:  # turns in place
            along = across = 0.004 + 0.01 * amount
            turn = 0.15 * amount + math.radians(0.5)  # the turn rate is only good to ~15 % (battery, floor)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        local = np.diag([along ** 2, across ** 2])
        self.P[:2, :2] += rot @ local @ rot.T
        self.P[2, 2] += turn ** 2

    # ---- measurements ------------------------------------------------------

    def update_tag(self, x: float, y: float, yaw: float, tag_count: int, nearest_m: float) -> str:
        """Fuse one tag pose. Returns "fused", or "outlier" (the state is left as it was)."""
        sigma_xy, sigma_yaw = tag_noise(tag_count, nearest_m)
        measure = np.array([x, y, wrap(yaw)], dtype=float)
        noise = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        innovation = measure - self.x
        innovation[2] = wrap(float(innovation[2]))
        s_matrix = self.P + noise
        if float(innovation @ np.linalg.solve(s_matrix, innovation)) > CHI2_3DOF_99:
            return "outlier"
        gain = self.P @ np.linalg.inv(s_matrix)
        self.x = self.x + gain @ innovation
        self.x[2] = wrap(float(self.x[2]))
        self.P = (np.eye(3) - gain) @ self.P
        self._project()
        return "fused"

    def update_wall(self, sonic_x_m: float, pan_rad: float, distance_m: float, max_step_m: float = 0.12) -> bool:
        """A range that matches the arena wall moves the pose. Foam, which is much closer, does not."""
        if not self.ready or distance_m <= 0.02 or distance_m > 1.5:
            return False

        def predict(state: np.ndarray) -> float:
            x, y, yaw = (float(v) for v in state)
            ox = x + sonic_x_m * math.cos(yaw)
            oy = y + sonic_x_m * math.sin(yaw)
            return ray_to_wall(ox, oy, yaw + pan_rad, self.width_m, self.height_m)

        predicted = predict(self.x)
        if abs(distance_m - predicted) > max_step_m:
            return False
        jacobian = np.zeros(3)
        for index in range(3):
            nudged = self.x.copy()
            nudged[index] += 0.01
            jacobian[index] = (predict(nudged) - predicted) / 0.01
        variance = float(jacobian @ self.P @ jacobian) + 0.02 ** 2
        if variance < 1e-9:
            return False
        gain = self.P @ jacobian / variance
        correction = gain * (distance_m - predicted)
        if max(abs(float(correction[0])), abs(float(correction[1]))) > max_step_m or \
                abs(float(correction[2])) > math.radians(6.0):
            return False
        self.x = self.x + correction
        self.x[2] = wrap(float(self.x[2]))
        self.P = (np.eye(3) - np.outer(gain, jacobian)) @ self.P
        self._project()
        return True
