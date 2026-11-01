"""Ground-plane (bird's-eye) projection of the cameras: every pixel of a car-centred floor grid is looked up in the
fisheye image of each camera (its mount and intrinsics), so the floor appears as seen from above and tags, walls and
blocks stretch away where they stand up."""
from __future__ import annotations

import math

import cv2
import numpy as np

SHOW_SCALE = 3
FEATHER_PX = 12            # a camera counts fully only this many cells (2.4 cm) inside the edge of what it sees
SHARP_POWER = 6            # blend weight: (sharpness / best sharpness) to this power
MIN_OVERLAP_CELLS = 300
MARGIN_M = 0.10            # the picture reaches this far beyond the body edge on every side
MIN_DEPRESSION_DEG = 12.0  # a camera shows the floor only where it looks down at it by this much: lower, one pixel covers many cm
RES_M = 0.002              # 3 mm per picture pixel: finer than the cameras see the floor


class GroundMapper:
    def __init__(self, loc, car_from_camera, body: tuple[float, float]):
        self.loc = loc
        self.car_from_camera = car_from_camera
        self.body = body
        self.x_range = (-body[0] / 2 - MARGIN_M, body[0] / 2 + MARGIN_M)   # m ahead of the car centre: bottom to top
        self.y_range = (-body[1] / 2 - MARGIN_M, body[1] / 2 + MARGIN_M)   # m to the left: right edge to left edge
        self.maps: dict = {}
        xs = np.arange(self.x_range[1], self.x_range[0], -RES_M)          # top row = furthest ahead
        ys = np.arange(self.y_range[1], self.y_range[0], -RES_M)          # left column = most left
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
        # the floor is sharp where the camera is high and close: cm of floor per pixel grows like range^2 / height
        cam_xy = self.car_from_camera(mount)[:2, 3]
        height = float(self.car_from_camera(mount)[2, 3])
        dist = np.hypot(self.ground[:, 0] - cam_xy[0], self.ground[:, 1] - cam_xy[1])
        ok &= np.degrees(np.arctan2(height, dist)) >= MIN_DEPRESSION_DEG
        weight = np.where(ok, height / (dist ** 2 + height ** 2), 0.0).reshape(self.shape).astype(np.float32)
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

    @staticmethod
    def _colour_gains(images: list, valid: np.ndarray) -> list:
        """Per-camera, per-channel gain so that cameras agree on the floor they both see (least squares on the log of the
        mean colour of each overlap, mean gain 1)."""
        n = len(images)
        rows, rhs = [], []
        for i in range(n):
            for j in range(i + 1, n):
                both = valid[i] & valid[j]
                if both.sum() < MIN_OVERLAP_CELLS:
                    continue
                mi, mj = images[i][both].mean(axis=0) + 1.0, images[j][both].mean(axis=0) + 1.0
                row = np.zeros(n)
                row[i], row[j] = 1.0, -1.0
                rows.append(row)
                rhs.append(np.log(mj) - np.log(mi))          # g_i - g_j = log(mean_j / mean_i)
        gains = [np.ones(3, np.float32) for _ in range(n)]
        if rows:
            rows.append(np.ones(n))
            rhs.append(np.zeros(3))
            g = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
            gains = [np.clip(np.exp(g[k]), 0.7, 1.4).astype(np.float32) for k in range(n)]
        return gains

    def mosaic(self, items: list) -> np.ndarray:
        """items: [(camera, mount, frame)]. Where cameras overlap they are blended: the sharper one counts more, and every
        camera counts less towards the edge of what it sees. Each camera's colour is scaled so that the overlaps agree."""
        warps = [self.warp(camera, mount, frame) for camera, mount, frame in items]
        images = [img.astype(np.float32) for img, _w in warps]
        quality = np.stack([w for _img, w in warps])
        best = quality.max(axis=0)
        gains = self._colour_gains(images, quality > 0)
        total = np.zeros(self.shape, np.float32)
        acc = np.zeros(self.shape + (3,), np.float32)
        for img, gain, q in zip(images, gains, quality):
            edge = np.clip(cv2.distanceTransform((q > 0).astype(np.uint8), cv2.DIST_L2, 3) / FEATHER_PX, 0.0, 1.0)
            wt = edge * (q / np.maximum(best, 1e-9)) ** SHARP_POWER
            acc += img * gain * wt[..., None]
            total += wt
        out = np.full(self.shape + (3,), 40, np.uint8)
        seen = total > 1e-6
        out[seen] = np.clip(acc[seen] / total[seen, None], 0, 255).astype(np.uint8)
        best = np.where(seen, 1.0, 0.0).astype(np.float32)
        out = cv2.resize(out, None, fx=SHOW_SCALE, fy=SHOW_SCALE, interpolation=cv2.INTER_CUBIC)   # smooth, lines stay sharp
        unseen = cv2.resize((best == 0).astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        yy, xx = np.indices(unseen.shape)
        out[unseen & (((xx + yy) // 6) % 2 == 0)] = (70, 70, 70)    # striped: no camera sees this floor
        length, width = self.body                                # x forward = up, y left = left

        def cell(x, y):
            return (int(round((self.y_range[1] - y) / RES_M * SHOW_SCALE)),
                    int(round((self.x_range[1] - x) / RES_M * SHOW_SCALE)))
        cv2.rectangle(out, cell(length / 2, width / 2), cell(-length / 2, -width / 2), (80, 80, 255), 3, cv2.LINE_AA)
        cv2.arrowedLine(out, cell(0, 0), cell(length / 2 - 0.02, 0), (80, 80, 255), 3, tipLength=0.25)
        for d in (0.05, 0.10):                                   # 5 and 10 cm from the body edge
            cv2.rectangle(out, cell(length / 2 + d, width / 2 + d), cell(-length / 2 - d, -width / 2 - d),
                          (150, 150, 150), 1, cv2.LINE_AA)
        return out
