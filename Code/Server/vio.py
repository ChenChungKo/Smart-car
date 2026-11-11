#!/usr/bin/env python3
"""Odometry from the IMU and the cameras: the z gyro and the motor speed carry the pose, the wall tags correct it.

State: x, y, yaw (arena frame) and the scale of the calibrated forward and of the sideways speed (the battery changes
them, and driving diagonally does not change them equally). Between tag
pictures the heading comes from the gyro and the distance from speed x time; every tag pose, also from a picture taken
while the car moves, is a measurement with its own uncertainty. The floor of the arena has no texture to follow and the
accelerometer cannot give a distance (tools/imu_motion_check.py), so the visual part is the tags.
"""
from __future__ import annotations

import math

import numpy as np

from pose_kalman import CHI2_3DOF_99, ray_to_wall, tag_noise, wrap


class VioFilter:
    SPEED_NOISE = 0.05                # of the speed, per sqrt(s): slip, the start and the stop of a move
    GYRO_SCALE_NOISE = 0.015          # of the angle turned
    GYRO_BIAS_NOISE = math.radians(0.03)   # rad/s, per sqrt(s)
    SCALE_DRIFT = 0.01                # of the scale, per sqrt(s)
    TIMING_S = 0.03                   # how well the time of a picture is known
    TAG_SIGMA = 0.5                   # of pose_kalman.tag_noise: while driving, front-camera poses scatter 2-3 cm, not 12 (2026-11-09)
    START_SIGMA_M = 0.02              # the pose the car starts from: tags while it stands
    START_SIGMA_DEG = 3.0             # single-camera headings differ by 1-3 deg: let the tags correct it during the drive

    def __init__(self, x: float, y: float, yaw: float, tag_sigma: float | None = None):
        self.x = np.array([x, y, wrap(yaw), 1.0, 1.0], float)
        self.P = np.diag([self.START_SIGMA_M ** 2, self.START_SIGMA_M ** 2, math.radians(self.START_SIGMA_DEG) ** 2, 0.10 ** 2, 0.10 ** 2])
        self.tag_sigma = self.TAG_SIGMA if tag_sigma is None else tag_sigma
        self.fused = self.rejected = 0

    def predict(self, dt: float, vx: float, vy: float, omega: float) -> None:
        """Advance by dt: vx, vy = calibrated body speed (m/s, forward and left), omega = measured turn rate (rad/s)."""
        yaw, sx, sy = float(self.x[2]), float(self.x[3]), float(self.x[4])
        c, s = math.cos(yaw), math.sin(yaw)
        wx, wy = c * sx * vx - s * sy * vy, s * sx * vx + c * sy * vy
        self.x += [wx * dt, wy * dt, omega * dt, 0.0, 0.0]
        self.x[2] = wrap(float(self.x[2]))
        F = np.eye(5)
        F[0, 2], F[1, 2] = -wy * dt, wx * dt
        F[0, 3], F[1, 3] = c * vx * dt, s * vx * dt
        F[0, 4], F[1, 4] = -s * vy * dt, c * vy * dt
        speed = math.hypot(vx, vy)
        q_pos = (self.SPEED_NOISE * speed) ** 2 * dt
        q_yaw = (self.GYRO_SCALE_NOISE * omega * dt) ** 2 + self.GYRO_BIAS_NOISE ** 2 * dt
        self.P = F @ self.P @ F.T + np.diag([q_pos, q_pos, q_yaw, self.SCALE_DRIFT ** 2 * dt, self.SCALE_DRIFT ** 2 * dt])

    def update_tags(self, x: float, y: float, yaw: float, tag_count: int, nearest_m: float, speed: float = 0.0,
                    sigma_scale: float | None = None) -> bool:
        """One pose from the tags of one picture. False: too far from the prediction, not used."""
        sigma_xy, sigma_yaw = tag_noise(tag_count, nearest_m)
        scale = self.tag_sigma if sigma_scale is None else sigma_scale
        sigma_xy = math.hypot(sigma_xy * scale, speed * self.TIMING_S)
        sigma_yaw *= scale
        H = np.hstack([np.eye(3), np.zeros((3, 2))])
        R = np.diag([sigma_xy ** 2, sigma_xy ** 2, sigma_yaw ** 2])
        y_ = np.array([x, y, yaw], float) - self.x[:3]
        y_[2] = wrap(float(y_[2]))
        S = H @ self.P @ H.T + R
        if float(y_ @ np.linalg.solve(S, y_)) > CHI2_3DOF_99:
            self.rejected += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y_
        self.x[2] = wrap(float(self.x[2]))
        self.P = (np.eye(5) - K @ H) @ self.P
        self.fused += 1
        return True

    def update_wall(self, sensor_ahead_m: float, pan: float, range_m: float, arena_m: tuple, gate_m: float = 0.06,
                    sigma_m: float = 0.02) -> bool:
        """An ultrasonic echo that matches the wall the pose expects corrects the pose; anything else says nothing here."""
        def expected(st) -> float:
            ox, oy = st[0] + sensor_ahead_m * math.cos(st[2]), st[1] + sensor_ahead_m * math.sin(st[2])
            return ray_to_wall(ox, oy, st[2] + pan, arena_m[0], arena_m[1])
        z = expected(self.x)
        if abs(range_m - z) > gate_m:
            return False
        H = np.zeros(5)
        for n in range(3):
            d = self.x.copy()
            d[n] += 1e-3
            H[n] = (expected(d) - z) / 1e-3
        S = float(H @ self.P @ H) + sigma_m ** 2
        K = self.P @ H / S
        self.x += K * (range_m - z)
        self.x[2] = wrap(float(self.x[2]))
        self.P = (np.eye(5) - np.outer(K, H)) @ self.P
        return True
