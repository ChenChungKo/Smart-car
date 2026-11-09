"""One occupancy grid of the arena in world coordinates; every sensor updates it with its own weight.

Log-odds per cell. The ultrasonic says the cells along its beam are free up to the echo and something is at the echo; the
camera says something stands where a foot point is and the floor before it is free. Only the size of the arena is known
in advance: it comes from the wall tags of the map. Nothing is assumed about what stands in it.
"""
from __future__ import annotations

import math
import threading

import cv2
import numpy as np

RES_M = 0.02
LIMIT = (-2.0, 4.0)
OCCUPIED = 0.85           # log-odds above this: a cell the planner must not touch
WALL_BAND_M = 0.05        # a reading this close to the arena's edge is the wall, which is known already
SONIC_HIT, SONIC_FREE = 0.9, -0.35       # the ultrasonic measures the distance: what is nearer than its echo is not there
SONIC_HALF_BEAM_DEG = 12.0
CAMERA_HIT, CAMERA_FREE = 0.35, -0.08    # the camera: an estimate on a line through the picture, a shadow looks the same
CAMERA_RANGE_SIGMA = (0.02, 0.04, 0.30)  # m: at least the first, the second at the third metres away, growing with range^2
CAMERA_CLOSE = 0.25       # "something at least this near": weak, it only says where the floor stops being visible


class WorldGrid:
    def __init__(self, arena_m: tuple[float, float]):
        self.w, self.h = float(arena_m[0]), float(arena_m[1])
        self.nx, self.ny = int(round(self.w / RES_M)) + 1, int(round(self.h / RES_M)) + 1
        self.log = np.zeros((self.nx, self.ny), np.float32)
        self.lock = threading.Lock()
        self.version = 0

    def _cell(self, x: float, y: float) -> tuple[int, int] | None:
        i, j = int(round(x / RES_M)), int(round(y / RES_M))
        return (i, j) if 0 <= i < self.nx and 0 <= j < self.ny else None

    def _near_wall(self, x: float, y: float) -> bool:
        return min(x, self.w - x, y, self.h - y) < WALL_BAND_M

    def _bump(self, x: float, y: float, value: float, skip_wall: bool = True) -> None:
        if skip_wall and self._near_wall(x, y):
            return
        c = self._cell(x, y)
        if c is not None:
            self.log[c] = np.clip(self.log[c] + value, *LIMIT)

    def _line(self, x0: float, y0: float, x1: float, y1: float, value: float, stop_before_m: float) -> None:
        length = math.hypot(x1 - x0, y1 - y0) - stop_before_m
        if length <= 0:
            return
        n = max(1, int(length / (RES_M * 0.7)))
        dx, dy = (x1 - x0) / math.hypot(x1 - x0, y1 - y0), (y1 - y0) / math.hypot(x1 - x0, y1 - y0)
        for k in range(n + 1):
            self._bump(x0 + dx * length * k / n, y0 + dy * length * k / n, value)

    def add_sonic(self, x: float, y: float, angle: float, range_m: float, reach_m: float = 1.2) -> None:
        """An echo at range_m along `angle` from the sensor at (x, y). The arc at that range is where it may have come from."""
        if not 0.02 < range_m < reach_m:
            return
        with self.lock:
            offsets = np.radians(np.arange(-SONIC_HALF_BEAM_DEG, SONIC_HALF_BEAM_DEG + 0.1, 3.0))
            for off in offsets:
                a = angle + float(off)
                ex, ey = x + range_m * math.cos(a), y + range_m * math.sin(a)
                self._line(x, y, ex, ey, SONIC_FREE, 0.04)
                weight = 1.0 - 0.7 * abs(float(off)) / math.radians(SONIC_HALF_BEAM_DEG)
                self._bump(ex, ey, SONIC_HIT * weight / 3.0)
            self.version += 1

    def add_foot(self, origin: tuple[float, float], points: np.ndarray, close: np.ndarray) -> None:
        """Foot points (world, m) of what stands on the floor, as seen from the camera at `origin`."""
        with self.lock:
            low, at, ref = CAMERA_RANGE_SIGMA
            for x, y in points:
                x, y = float(x), float(y)
                r = math.hypot(x - origin[0], y - origin[1])
                if r < 1e-3:
                    continue
                sigma = max(low, at * (r / ref) ** 2)
                ux, uy = (x - origin[0]) / r, (y - origin[1]) / r
                self._line(origin[0], origin[1], x, y, CAMERA_FREE, sigma + 0.02)
                for d, w in ((-sigma, 0.5), (0.0, 1.0), (sigma, 0.5)):
                    self._bump(x + ux * d, y + uy * d, CAMERA_HIT * w)
            for x, y in close:
                self._bump(float(x), float(y), CAMERA_CLOSE)
            self.version += 1

    def occupied(self) -> np.ndarray:
        """Bool (nx, ny): cells to keep off, with the arena's edge."""
        with self.lock:
            occ = self.log > OCCUPIED
        occ = occ.copy()
        occ[0, :] = occ[-1, :] = True
        occ[:, 0] = occ[:, -1] = True
        return occ


def clearance_m(occ: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5).astype(float) * RES_M
