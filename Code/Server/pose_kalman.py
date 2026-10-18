#!/usr/bin/env python3
"""Planar pose Kalman filter.

The state is x, y and yaw. A tag fix is the measurement. A single far tag is
given a large variance, so it cannot drag the car through the arena wall.
After every update the body is kept inside the arena: a 29 cm car whose centre
is reported 5 cm from the wall is moved back until the whole body fits.
An ultrasonic return is used only when it agrees with the arena wall; a foam
hit does not move the pose.
"""

from __future__ import annotations

import math

import numpy as np


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


class PoseKalman:
    def __init__(self, width_m: float, height_m: float, body_half: tuple[float, float]):
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.body_half = body_half
        self.ready = False
        self.x = np.zeros(3)
        self.P = np.diag([0.08 ** 2, 0.08 ** 2, math.radians(10.0) ** 2])

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

    def update_tag(self, x: float, y: float, yaw: float, tag_count: int, nearest_m: float) -> bool:
        """Fuse one tag pose. Returns whether the body had to be kept inside the wall."""
        far = nearest_m > 0.85 or tag_count < 2
        sigma_xy = 0.03 if tag_count >= 2 and nearest_m <= 0.85 else (0.12 if far else 0.05)
        sigma_yaw = math.radians(4.0 if tag_count >= 2 else 12.0)
        measure = np.array([x, y, wrap(yaw)], dtype=float)
        noise = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        if not self.ready:
            self.x = measure.copy()
            self.P = noise.copy()
            self.ready = True
            return self._project()
        innovation = measure - self.x
        innovation[2] = wrap(float(innovation[2]))
        gain = self.P @ np.linalg.inv(self.P + noise)
        self.x = self.x + gain @ innovation
        self.x[2] = wrap(float(self.x[2]))
        self.P = (np.eye(3) - gain) @ self.P
        return self._project()

    def follow_motion(self, x: float, y: float, yaw: float) -> None:
        """Dead reckoning between tag fixes. Grows the covariance by how far the car moved."""
        if not self.ready:
            self.x = np.array([x, y, wrap(yaw)], dtype=float)
            self.ready = True
            self._project()
            return
        step = math.hypot(x - self.x[0], y - self.x[1])
        turn = abs(wrap(yaw - float(self.x[2])))
        self.x[:] = [x, y, wrap(yaw)]
        self.P = self.P + np.diag([
            (0.02 + 0.15 * step) ** 2,
            (0.02 + 0.15 * step) ** 2,
            (math.radians(3.0) + 0.3 * turn) ** 2,
        ])
        self._project()

    def update_wall(self, sonic_x_m: float, pan_rad: float, distance_m: float) -> bool:
        """A range that matches the arena wall moves the pose. Foam, which is much closer, does not."""
        if not self.ready or distance_m <= 0.02 or distance_m > 1.5:
            return False

        def predict(state: np.ndarray) -> float:
            x, y, yaw = (float(v) for v in state)
            ox = x + sonic_x_m * math.cos(yaw)
            oy = y + sonic_x_m * math.sin(yaw)
            return ray_to_wall(ox, oy, yaw + pan_rad, self.width_m, self.height_m)

        predicted = predict(self.x)
        if abs(distance_m - predicted) > 0.12:
            return False
        jacobian = np.zeros(3)
        for index in range(3):
            nudged = self.x.copy()
            nudged[index] += 0.01
            jacobian[index] = (predict(nudged) - predicted) / 0.01
        variance = float(jacobian @ self.P @ jacobian) + 0.03 ** 2
        if variance < 1e-9:
            return False
        gain = self.P @ jacobian / variance
        correction = gain * (distance_m - predicted)
        # A wall reading may only nudge the pose. A corner makes the ray equation jump, so a large step is rejected.
        if max(abs(float(correction[0])), abs(float(correction[1]))) > 0.05 or abs(float(correction[2])) > math.radians(5.0):
            return False
        self.x = self.x + correction
        self.x[2] = wrap(float(self.x[2]))
        self.P = (np.eye(3) - np.outer(gain, jacobian)) @ self.P
        self._project()
        return True
