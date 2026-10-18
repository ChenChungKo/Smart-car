"""Range scan from one fisheye camera: blue foam (walls and obstacles) standing on a light floor.

In every image column the lowest strongly blue pixel is where the foam meets the floor; the
fisheye intrinsics and the camera mount turn that pixel into a point on the floor (car frame,
x forward, y left). The glossy floor mirrors the foam in a pale blue, so saturation must be high.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from aruco_localizer import CALIB, car_from_camera

EXTRINSIC = CALIB / "bev_extrinsic_metric_auto"
SUMMARY = EXTRINSIC / "metric_extrinsic_auto_summary.json"


class FloorScanner:
    def __init__(self, localizer, mount: dict, camera: str, width: int = 640, height: int = 480,
                 hsv_lo=(90, 170, 40), hsv_hi=(130, 255, 255), column_step: int = 8,
                 max_range_m: float = 1.2):
        self.camera = camera
        self.k, self.d = localizer.kd(camera, width, height)
        T = car_from_camera(mount)
        self.rotation, self.origin = T[:3, :3], T[:3, 3]
        self.size = (width, height)
        self.lo, self.hi = np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
        self.columns = np.arange(column_step // 2, width, column_step)
        self.max_range_m = max_range_m
        self.homography = None
        h_path = EXTRINSIC / f"camera_{camera}_H.npy"
        if camera in ("left", "right") and h_path.exists() and SUMMARY.exists():
            info = json.loads(SUMMARY.read_text(encoding="utf-8"))
            self.homography = np.load(h_path).astype(np.float64)
            self.car_uv = np.array(info["car_mask_px"]["center"], dtype=np.float64)
            self.px_per_m = float(info["px_per_mm"]) * 1000.0
            self.p_bev = self.k.copy()
            self.p_bev[0, 2] = width / 2.0 * 2.0
            self.p_bev[1, 2] = height / 2.0 * 2.0

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        """Normalized image coordinates. Front and rear use the fisheye model; USB uses the standard model."""
        pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        if self.d.shape[0] <= 4:
            return cv2.fisheye.undistortPoints(pts, self.k, self.d).reshape(-1, 2)
        return cv2.undistortPoints(pts, self.k, self.d).reshape(-1, 2)

    def foam_mask(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array((85, 25, 40), np.uint8), np.array((130, 255, 255), np.uint8))
        return cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def sees_foam(self, bgr: np.ndarray) -> bool:
        """True when this camera is looking at a foam block, including the pale blue the floor scan misses."""
        return float((self.foam_mask(bgr) > 0).mean()) >= 0.12

    def foam_against_body(self, bgr: np.ndarray) -> bool:
        """The block fills the near part of this side camera, not a foam wall farther ahead."""
        blue = self.foam_mask(bgr) > 0
        near = blue[int(blue.shape[0] * 0.55):]
        return float(blue.mean()) >= 0.20 and float(near.mean()) >= 0.22

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        blue = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), self.lo, self.hi)
        return cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((7, 3), np.uint8))

    def scan(self, bgr: np.ndarray) -> np.ndarray:
        """Nx2 floor points (car frame, m) where blue foam meets the floor."""
        sub = self.mask(bgr)[:, self.columns] > 0
        has = sub.any(axis=0)
        if not has.any():
            return np.zeros((0, 2))
        foot = sub.shape[0] - np.argmax(sub[::-1], axis=0)
        pixels = np.column_stack([self.columns[has], foot[has]]).astype(np.float64)
        return self.project(pixels)

    def project(self, pixels: np.ndarray) -> np.ndarray:
        """Floor points (car frame, m) for image pixels where a wall meets the floor."""
        if len(pixels) == 0:
            return np.zeros((0, 2))
        pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        if self.homography is not None:
            # Pixels near the horizon sit above the calibrated ground mat and blow up under H.
            pts = pts[pts[:, 0, 1] >= self.size[1] * 0.50]
            if len(pts) == 0:
                return np.zeros((0, 2))
            und = cv2.fisheye.undistortPoints(pts, self.k, self.d, np.eye(3), self.p_bev)
            bev = cv2.perspectiveTransform(und.astype(np.float32), self.homography).reshape(-1, 2)
            fwd = (self.car_uv[1] - bev[:, 1]) / self.px_per_m
            left = (self.car_uv[0] - bev[:, 0]) / self.px_per_m
            points = np.column_stack([fwd, left])
            side_ok = points[:, 1] > 0.09 if self.camera == "left" else points[:, 1] < -0.09
            reach = np.hypot(points[:, 0], points[:, 1])
            return points[side_ok & (reach < self.max_range_m)]
        norm = self.undistort(pts)
        limit = 2.7 if self.d.shape[0] <= 4 else 1.15
        norm = norm[np.hypot(norm[:, 0], norm[:, 1]) < limit]
        if len(norm) == 0:
            return np.zeros((0, 2))
        rays = (self.rotation @ np.column_stack([norm, np.ones(len(norm))]).T).T
        rays = rays[rays[:, 2] < -1e-3]
        if len(rays) == 0:
            return np.zeros((0, 2))
        points = self.origin[:2] + (-self.origin[2] / rays[:, 2])[:, None] * rays[:, :2]
        reach = np.hypot(points[:, 0] - self.origin[0], points[:, 1] - self.origin[1])
        return points[reach < self.max_range_m]

    def plain_contact(self, bgr: np.ndarray) -> np.ndarray:
        """Where a plain grey wall meets a lighter floor. Blue foam uses scan() instead."""
        img = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
        height, width = gray.shape
        top, bottom = int(height * 0.35), int(height * 0.72)
        change = gray[top + 1:bottom] - gray[top:bottom - 1]
        kernel = np.ones(15) / 15.0
        pixels = []
        for x in range(16, width - 16, 8):
            column = np.convolve(change[:, x].astype(float), kernel, mode="same")
            index = int(np.argmax(column))
            if column[index] < 4.0:
                continue
            pixels.append((x, top + index + 1))
        return self.project(np.array(pixels, np.float64) if pixels else np.zeros((0, 2)))


def to_world(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([x + points[:, 0] * c - points[:, 1] * s, y + points[:, 0] * s + points[:, 1] * c])
