"""Where the gimbal camera is on the car at a pan angle: it swings on an arm about the pan axis.

Parameters (car_nav.json "gimbal_tags", solved by tools/solve_gimbal.py):
  pivot_x_m, pivot_y_m  the pan axis in the car frame
  radius_m, phase_deg   the lens is this far from the axis, at this angle from the direction it looks
  z_m                   height of the lens
  scale_left, scale_right   real turn per commanded degree, to the left and to the right
  yaw0_deg              where it looks when commanded straight ahead
  pitch_deg, roll_deg   tilt of the camera (pitch + = down), the same at every pan
"""
from __future__ import annotations

import math

import numpy as np

NAMES = ("pivot_x_m", "pivot_y_m", "radius_m", "phase_deg", "z_m", "scale_left", "scale_right", "yaw0_deg", "pitch_deg", "roll_deg")
START = dict(pivot_x_m=0.085, pivot_y_m=0.0, radius_m=0.03, phase_deg=0.0, z_m=0.105, scale_left=1.0, scale_right=1.0,
             yaw0_deg=0.0, pitch_deg=0.0, roll_deg=0.0)


def car_T_camera(p: dict, pan_deg: float) -> np.ndarray:
    """Camera frame (x right, y down, z forward) in the car frame (x forward, y left, z up) at this commanded pan (+ = left)."""
    yaw = math.radians(p["yaw0_deg"] + pan_deg * (p["scale_left"] if pan_deg >= 0 else p["scale_right"]))
    pitch, roll = math.radians(p["pitch_deg"]), math.radians(p["roll_deg"])
    forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.array([-math.sin(pitch), 0.0, -math.cos(pitch)])
    c, s = math.cos(roll), math.sin(roll)
    right, down = c * right + s * down, -s * right + c * down          # turned about the optical axis
    cz, sz = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    a = yaw + math.radians(p["phase_deg"])
    T = np.eye(4)
    T[:3, :3] = rz @ np.column_stack([right, down, forward])
    T[:3, 3] = [p["pivot_x_m"] + p["radius_m"] * math.cos(a), p["pivot_y_m"] + p["radius_m"] * math.sin(a), p["z_m"]]
    return T
