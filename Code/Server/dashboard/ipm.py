"""Ground-plane (bird's-eye) projection of the cameras: every pixel of a car-centred floor grid is looked up in the
fisheye image of each camera (its mount and intrinsics), so the floor appears as seen from above and tags, walls and
blocks stretch away where they stand up."""
from __future__ import annotations

import math

import cv2
import numpy as np

X_RANGE = (-0.45, 0.90)    # m ahead of the car centre: bottom to top of the picture
Y_RANGE = (-0.70, 0.70)    # m to the left: right edge to left edge
RES_M = 0.01


class GroundMapper:
    def __init__(self, loc, car_from_camera):
        self.loc = loc
        self.car_from_camera = car_from_camera
        self.maps: dict = {}
        xs = np.arange(X_RANGE[1], X_RANGE[0], -RES_M)          # top row = furthest ahead
        ys = np.arange(Y_RANGE[1], Y_RANGE[0], -RES_M)          # left column = most left
        gy, gx = np.meshgrid(ys, xs)                            # (rows, cols)
        self.shape = gx.shape
        self.ground = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).reshape(-1, 3)

    def _build(self, camera: str, mount: dict, w: int, h: int):
        k, d = self.loc.kd(camera, w, h)
        cam_T_car = np.linalg.inv(self.car_from_camera(mount))
        pts = (cam_T_car[:3, :3] @ self.ground.T).T + cam_T_car[:3, 3]
        ok = pts[:, 2] > 0.03
        angle = np.degrees(np.arctan2(np.hypot(pts[:, 0], pts[:, 1]), np.maximum(pts[:, 2], 1e-6)))
        ok &= angle < 75.0                                      # the fisheye rim is not trusted on the floor
        proj = np.zeros((len(pts), 2))
        if ok.any():
            p, _ = cv2.fisheye.projectPoints(pts[ok].reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), k, d)
            proj[ok] = p.reshape(-1, 2)
        ok &= (proj[:, 0] >= 0) & (proj[:, 0] < w - 1) & (proj[:, 1] >= 0) & (proj[:, 1] < h - 1)
        # nearer to the camera = sharper: weight by distance from the camera's foot on the floor
        cam_xy = self.car_from_camera(mount)[:2, 3]
        dist = np.hypot(self.ground[:, 0] - cam_xy[0], self.ground[:, 1] - cam_xy[1])
        weight = np.where(ok, 1.0 / (0.05 + dist), 0.0).reshape(self.shape).astype(np.float32)
        mx = proj[:, 0].reshape(self.shape).astype(np.float32)
        my = proj[:, 1].reshape(self.shape).astype(np.float32)
        return mx, my, weight

    def warp(self, camera: str, mount: dict, frame: np.ndarray):
        h, w = frame.shape[:2]
        key = (camera, w, h, tuple(sorted(mount.items())))
        if key not in self.maps:
            self.maps[key] = self._build(camera, mount, w, h)
        mx, my, weight = self.maps[key]
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), weight

    def mosaic(self, items: list, body: tuple[float, float]) -> np.ndarray:
        """items: [(camera, mount, frame)]; the sharpest camera wins each floor cell; the car is drawn on top."""
        out = np.full(self.shape + (3,), 40, np.uint8)
        best = np.zeros(self.shape, np.float32)
        for camera, mount, frame in items:
            img, weight = self.warp(camera, mount, frame)
            take = weight > best
            out[take] = img[take]
            best = np.maximum(best, weight)
        # car outline: x forward = up, y left = left
        length, width = body
        def cell(x, y):
            return int(round((Y_RANGE[1] - y) / RES_M)), int(round((X_RANGE[1] - x) / RES_M))
        poly = np.array([cell(length / 2, width / 2), cell(length / 2, -width / 2), cell(-length / 2, -width / 2),
                         cell(-length / 2, width / 2)], np.int32)
        cv2.polylines(out, [poly], True, (80, 80, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(out, cell(0, 0), cell(length / 2 + 0.08, 0), (80, 80, 255), 2, tipLength=0.3)
        for r in (0.25, 0.5, 0.75):                              # distance rings every 25 cm ahead
            cv2.line(out, cell(r, Y_RANGE[1]), cell(r, Y_RANGE[0]), (110, 110, 110), 1)
            cv2.putText(out, f"{int(r * 100)}", (4, cell(r, 0)[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return out
