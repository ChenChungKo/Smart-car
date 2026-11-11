"""The map: one grid of the arena, each cell the log-odds that something stands there.

Three states come out of it: occupied, seen free, and not seen yet. The car drives only on floor it has seen free.
  camera     a foot point is where something meets the floor: that cell and the one behind it gain; the floor the camera
             sees in front of it loses (it is free)
  ultrasonic an echo puts something on the arc at that range; the cells before it on the centre ray lose a little, but
             never where the camera already has something: foam at an angle throws the echo away, so a far echo is no
             proof that nothing nearer is there
  body       where the car stands, nothing else stands
Only the size of the arena is known in advance.
"""
from __future__ import annotations

import math
import threading

import numpy as np

RES_M = 0.02
OCCUPIED, FREE = 0.8, -0.3        # log-odds above / below: occupied / seen free; between: not known
LIMIT = (-3.0, 3.0)
WALL_BAND_M = 0.07                # a reading this close to the arena's edge is the wall, which is known
CAM_HIT, CAM_FREE = 0.6, -0.4     # per look: two looks make a cell occupied
CAM_DEPTH_M = 0.02
SONIC_HIT, SONIC_FREE = 0.5, -0.2
SONIC_HALF_DEG = 8.0
SONIC_KEEPS_ABOVE = 0.3           # the ultrasonic does not clear a cell the camera holds above this
BODY_FREE = -1.5


class World:
    def __init__(self, arena_m):
        self.w, self.h = float(arena_m[0]), float(arena_m[1])
        self.nx, self.ny = int(round(self.w / RES_M)) + 1, int(round(self.h / RES_M)) + 1
        self.log = np.zeros((self.nx, self.ny), np.float32)
        ii, jj = np.mgrid[0:self.nx, 0:self.ny]
        self.X, self.Y = ii * RES_M, jj * RES_M
        self.edge = (ii == 0) | (jj == 0) | (ii == self.nx - 1) | (jj == self.ny - 1)
        self.wall_band = np.minimum(np.minimum(self.X, self.w - self.X), np.minimum(self.Y, self.h - self.Y)) < WALL_BAND_M
        self.lock = threading.Lock()
        self.version = 0

    def _mask(self, pts) -> np.ndarray:
        m = np.zeros((self.nx, self.ny), bool)
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts):
            i, j = np.rint(pts[:, 0] / RES_M).astype(int), np.rint(pts[:, 1] / RES_M).astype(int)
            ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
            m[i[ok], j[ok]] = True
        return m

    @staticmethod
    def _ray_points(origin, ends, stop_short_m: float) -> np.ndarray:
        out = []
        for ex, ey in np.asarray(ends, float).reshape(-1, 2):
            length = math.hypot(ex - origin[0], ey - origin[1]) - stop_short_m
            if length <= 0:
                continue
            s = np.arange(0.0, length, RES_M * 0.6) / (length + stop_short_m)
            out.append(np.column_stack([origin[0] + (ex - origin[0]) * s, origin[1] + (ey - origin[1]) * s]))
        return np.vstack(out) if out else np.zeros((0, 2))

    def add_camera(self, origin, hits, free_ends, close) -> None:
        """One look: `hits` foot points, `free_ends` where floor was seen up to, `close` something nearer than the view."""
        hits = np.asarray(hits, float).reshape(-1, 2)
        free = self._mask(self._ray_points(origin, np.vstack([hits, np.asarray(free_ends, float).reshape(-1, 2)]), 0.03))
        behind = hits.copy()
        if len(hits):
            d = hits - np.asarray(origin)
            behind = hits + d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None] * CAM_DEPTH_M
        hit = (self._mask(hits) | self._mask(behind) | self._mask(close)) & ~self.wall_band
        with self.lock:
            self.log[free & ~hit] += CAM_FREE
            self.log[hit] += CAM_HIT
            np.clip(self.log, *LIMIT, out=self.log)
            self.version += 1

    def add_sonic(self, origin, angle: float, range_m: float, reach_m: float = 1.0) -> None:
        if not 0.03 < range_m < reach_m:
            return
        offs = np.radians(np.arange(-SONIC_HALF_DEG, SONIC_HALF_DEG + 0.1, 2.0))
        arc = np.column_stack([origin[0] + range_m * np.cos(angle + offs), origin[1] + range_m * np.sin(angle + offs)])
        hit = self._mask(arc) & ~self.wall_band
        end = (origin[0] + range_m * math.cos(angle), origin[1] + range_m * math.sin(angle))
        free = self._mask(self._ray_points(origin, [end], 0.06)) & ~hit
        with self.lock:
            self.log[hit] += SONIC_HIT
            self.log[free & (self.log < SONIC_KEEPS_ABOVE)] += SONIC_FREE
            np.clip(self.log, *LIMIT, out=self.log)
            self.version += 1

    def mark_body(self, x: float, y: float, yaw: float, half_l: float, half_w: float) -> None:
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.X - x, self.Y - y
        inside = (np.abs(dx * c + dy * s) <= half_l) & (np.abs(-dx * s + dy * c) <= half_w)
        with self.lock:
            changed = inside & (self.log > BODY_FREE)
            if changed.any():
                self.log[changed] = BODY_FREE
                self.version += 1

    def something_before(self, origin, angle: float, dist_m: float) -> bool:
        """Is a cell that is not seen free on the ray before dist_m? Then an echo along it need not be the wall."""
        end = (origin[0] + dist_m * math.cos(angle), origin[1] + dist_m * math.sin(angle))
        on_ray = self._mask(self._ray_points(origin, [end], 0.08))
        return bool((self.log[on_ray] > OCCUPIED).any())

    def snapshot(self) -> np.ndarray:
        with self.lock:
            return self.log.copy()

    def state_at(self, x: float, y: float, log: np.ndarray | None = None) -> tuple[str, float]:
        """('wall' | 'occupied' | 'unseen' | 'free', log-odds) of the cell at (x, y)."""
        i, j = int(round(x / RES_M)), int(round(y / RES_M))
        if not (0 < i < self.nx - 1 and 0 < j < self.ny - 1):
            return "wall", 9.0
        v = float((self.log if log is None else log)[i, j])
        return ("occupied" if v > OCCUPIED else "free" if v < FREE else "unseen"), v
