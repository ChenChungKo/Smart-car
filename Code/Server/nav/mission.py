"""The task: go once around the free space, keeping a distance from its boundary, the boundary on the right.

The line to follow is the set of points at that distance from whatever the map says is occupied: along a wall it is a
straight line, around a block it bends around the block, and it changes when the map changes.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from nav.planner import ConfigSpace, heading_index
from nav.world import RES_M


def boundary_loop(space: ConfigSpace, centre_gap_m: float):
    """Closed line [(x, y), ...], counter-clockwise (the boundary on the right), or None."""
    inside = (space.clear >= centre_gap_m).astype(np.uint8)
    contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    pts = np.column_stack([c[:, 1], c[:, 0]]) * RES_M               # contours are (col, row) = (j, i)
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    return pts if area > 0 else pts[::-1]


class Lap:
    """How far along the line the car has come, and the next pose to aim for."""

    def __init__(self):
        self.progress_m = 0.0
        self.last = None        # the point of the line the car was nearest to

    def advance(self, loop: np.ndarray, x: float, y: float) -> int:
        n = int(np.argmin(np.hypot(loop[:, 0] - x, loop[:, 1] - y)))
        if self.last is not None:
            step = math.hypot(loop[n, 0] - self.last[0], loop[n, 1] - self.last[1])
            ahead = loop[(n + 8) % len(loop)] - loop[n]
            moved = np.array([loop[n, 0] - self.last[0], loop[n, 1] - self.last[1]])
            if step < 0.25 and float(moved @ ahead) > 0.0:   # forward along the line; a jump is the map changing
                self.progress_m += step
        self.last = (float(loop[n, 0]), float(loop[n, 1]))
        return n

    def goal(self, loop: np.ndarray, n: int, space: ConfigSpace, ahead_m: float):
        """(i, j, heading) of a free pose on the line about ahead_m further, tangent to the line."""
        count, step = len(loop), max(1, int(ahead_m / RES_M))
        for extra in range(0, count // 2, 3):
            a = (n + step + extra) % count
            p, q = loop[a], loop[(a + 6) % count]
            yaw = math.atan2(float(q[1] - p[1]), float(q[0] - p[0]))
            i, j, k = int(round(p[0] / RES_M)), int(round(p[1] / RES_M)), heading_index(yaw)
            if space.free(i, j, k):
                return i, j, k
        return None

    @staticmethod
    def length(loop: np.ndarray) -> float:
        return float(np.sum(np.hypot(*(np.roll(loop, -1, axis=0) - loop).T)))
