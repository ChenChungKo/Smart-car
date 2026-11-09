"""A* in (x, y, heading) for the real body: a rectangle, tested against the map at every heading.

A cell is free for a heading when the rectangle of the car at that heading, with a margin, touches nothing; so where it
cannot turn it plans to move away first, without being told. Moving straight along the nose costs least, driving sideways
a little more; a cell also costs more the further it is from the distance to keep from the boundary.
"""
from __future__ import annotations

import heapq
import math
import time

import cv2
import numpy as np

from nav.world import RES_M, clearance_m

HEADINGS = 8
STEP = 2.0 * math.pi / HEADINGS
MOVES = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
TURN_COST = 1.5           # cells of travel a 45 deg turn on the spot is worth
SIDEWAYS_COST = 0.6       # of a step, when the move is 90 deg from the nose
OFF_GAP_COST = 1.2        # of a step, at OFF_GAP_FULL_M or more from the distance to keep
OFF_GAP_FULL_M = 0.10
MAX_EXPANDED = 80000


def heading_index(yaw: float) -> int:
    return int(round(yaw / STEP)) % HEADINGS


def body_kernel(half_l: float, half_w: float, yaw: float) -> np.ndarray:
    r = int(math.ceil(math.hypot(half_l, half_w) / RES_M)) + 1
    c, s = math.cos(yaw), math.sin(yaw)
    a, b = np.mgrid[-r:r + 1, -r:r + 1]
    dx, dy = a * RES_M, b * RES_M
    return ((np.abs(dx * c + dy * s) <= half_l) & (np.abs(-dx * s + dy * c) <= half_w)).astype(np.uint8)


class ConfigSpace:
    """For each heading, the cells where the body (plus margin) would touch something."""

    def __init__(self, occ: np.ndarray, half_l: float, half_w: float, margin: float):
        self.occ = occ
        self.clear = clearance_m(occ)
        self.blocked = [cv2.dilate(occ.astype(np.uint8), body_kernel(half_l + margin, half_w + margin, k * STEP)) > 0
                        for k in range(HEADINGS)]
        self.nx, self.ny = occ.shape

    def free(self, i: int, j: int, k: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[k % HEADINGS][i, j]

    def nearest_free(self, i: int, j: int, k: int, reach_cells: int = 4):
        best = None
        for a in range(-reach_cells, reach_cells + 1):
            for b in range(-reach_cells, reach_cells + 1):
                if self.free(i + a, j + b, k) and (best is None or a * a + b * b < best[0]):
                    best = (a * a + b * b, i + a, j + b)
        return None if best is None else (best[1], best[2])


def plan(space: ConfigSpace, start: tuple[float, float, float], goal: tuple[int, int, int], gap_m: float,
         budget_s: float = 0.6):
    """[(x, y, yaw), ...] from the start pose to the goal cell (i, j, heading), or None."""
    k0 = heading_index(start[2])
    i0, j0 = int(round(start[0] / RES_M)), int(round(start[1] / RES_M))
    if not space.free(i0, j0, k0):
        near = space.nearest_free(i0, j0, k0)
        if near is None:
            return None
        i0, j0 = near
    gi, gj, gk = goal

    def h(i: int, j: int, k: int) -> float:
        dx, dy = abs(i - gi), abs(j - gj)
        turn = min((k - gk) % HEADINGS, (gk - k) % HEADINGS)
        return dx + dy + (math.sqrt(2.0) - 2.0) * min(dx, dy) + TURN_COST * turn

    s = (i0, j0, k0)
    openq = [(h(*s), 0.0, s)]
    cost, came = {s: 0.0}, {}
    end, expanded = time.monotonic() + budget_s, 0
    while openq:
        _f, g, cur = heapq.heappop(openq)
        if g > cost.get(cur, math.inf):
            continue
        i, j, k = cur
        if abs(i - gi) <= 1 and abs(j - gj) <= 1 and k == gk:
            chain = [cur]
            while cur in came:
                cur = came[cur]
                chain.append(cur)
            chain.reverse()
            return [(c[0] * RES_M, c[1] * RES_M, (c[2] * STEP + math.pi) % (2 * math.pi) - math.pi) for c in chain]
        expanded += 1
        if expanded > MAX_EXPANDED or (expanded % 500 == 0 and time.monotonic() > end):
            return None
        for dk in (-1, 1):
            n = (i, j, (k + dk) % HEADINGS)
            if space.free(*n) and g + TURN_COST < cost.get(n, math.inf):
                cost[n], came[n] = g + TURN_COST, cur
                heapq.heappush(openq, (g + TURN_COST + h(*n), g + TURN_COST, n))
        for m, (di, dj) in enumerate(MOVES):
            n = (i + di, j + dj, k)
            if not space.free(*n) or (di and dj and (not space.free(i + di, j, k) or not space.free(i, j + dj, k))):
                continue
            off = abs(float(space.clear[n[0], n[1]]) - gap_m)
            side = 0.5 * (1.0 - math.cos((m - k) * STEP))
            step = math.hypot(di, dj) * (1.0 + SIDEWAYS_COST * side + OFF_GAP_COST * min(off / OFF_GAP_FULL_M, 1.0))
            new = g + step
            if new < cost.get(n, math.inf):
                cost[n], came[n] = new, cur
                heapq.heappush(openq, (new + h(*n), new, n))
    return None
