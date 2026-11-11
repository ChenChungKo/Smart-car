"""A* in (x, y, heading) for the real rectangle of the body.

A pose is allowed when the body with a margin touches nothing occupied. Onto floor it has not seen yet the car goes nose
first: the gimbal camera sits at the nose and sees the floor ahead, not the floor beside or behind the body. Any other move
(sideways, diagonal, backwards, turning) is allowed only where the whole body will stand on floor already seen free.
Turning costs, driving sideways costs a little, floor not seen yet and floor close to
something cost more. Nothing here knows the arena or the task.
"""
from __future__ import annotations

import heapq
import math
import time

import cv2
import numpy as np

from patrol.world import FREE, OCCUPIED, RES_M

HEADINGS = 8
STEP = 2.0 * math.pi / HEADINGS
MOVES = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
TURN_COST = 5.0            # cells of travel a 45 deg turn is worth: the car keeps its heading unless turning pays
SIDE_COST = {0: 0.0, 1: 0.15, 2: 0.5, 3: 1.0, 4: 1.0}   # by how many 45 deg steps the move is off the nose
UNSEEN_COST = 0.5
NEAR_COST, NEAR_M = 1.5, 0.08
BUDGET_S = 2.0


def heading_index(yaw: float) -> int:
    return int(round(yaw / STEP)) % HEADINGS


def heading_yaw(k: int) -> float:
    return (k * STEP + math.pi) % (2.0 * math.pi) - math.pi


def body_kernel(half_l: float, half_w: float, yaw: float) -> np.ndarray:
    r = int(math.ceil(math.hypot(half_l, half_w) / RES_M)) + 1
    c, s = math.cos(yaw), math.sin(yaw)
    a, b = np.mgrid[-r:r + 1, -r:r + 1]
    dx, dy = a * RES_M, b * RES_M
    return ((np.abs(dx * c + dy * s) <= half_l) & (np.abs(-dx * s + dy * c) <= half_w)).astype(np.uint8)


class Space:
    """What the map allows, for each heading of the body."""

    def __init__(self, log: np.ndarray, edge: np.ndarray, half_l: float, half_w: float, margin: float, version: int):
        self.version = version
        occ = (log > OCCUPIED) | edge
        self.unseen = ~occ & ~(log < FREE)
        self.clear = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * RES_M
        self.blocked = [cv2.dilate(occ.astype(np.uint8), body_kernel(half_l + margin, half_w + margin, k * STEP)) > 0
                        for k in range(HEADINGS)]
        notfree = (~(log < FREE) | edge).astype(np.uint8)
        self.seen_ok = [cv2.dilate(notfree, body_kernel(half_l + 0.01, half_w + 0.01, k * STEP)) == 0 for k in range(HEADINGS)]
        self.nx, self.ny = occ.shape

    def free(self, i: int, j: int, k: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[k % HEADINGS][i, j]

    def nearest_free(self, i: int, j: int, k: int | None, reach_cells: int):
        best = None
        for a in range(-reach_cells, reach_cells + 1):
            for b in range(-reach_cells, reach_cells + 1):
                for kk in ([k] if k is not None else range(HEADINGS)):
                    if self.free(i + a, j + b, kk) and (best is None or a * a + b * b < best[0]):
                        best = (a * a + b * b, i + a, j + b, kk)
        return None if best is None else best[1:]

    def path_ok(self, path: list, start: int = 0) -> bool:
        return all(self.free(int(round(x / RES_M)), int(round(y / RES_M)), heading_index(yaw)) for x, y, yaw in path[start:])


def plan(space: Space, start, goal_xy, goal_yaw=None):
    """([(x, y, yaw), ...] or None, info) from the pose `start` to the point, with the heading goal_yaw if given."""
    t0 = time.monotonic()
    k0 = heading_index(start[2])
    i0, j0 = int(round(start[0] / RES_M)), int(round(start[1] / RES_M))
    info = dict(expanded=0, moved_start=False)
    if not space.free(i0, j0, k0):
        near = space.nearest_free(i0, j0, k0, 4)
        if near is None:
            return None, dict(info, why="the car's own pose is not allowed by the map (too close to something)")
        i0, j0, info["moved_start"] = near[0], near[1], True
    gi, gj = int(round(goal_xy[0] / RES_M)), int(round(goal_xy[1] / RES_M))
    gk = None if goal_yaw is None else heading_index(goal_yaw)

    def h(i, j, k):
        dx, dy = abs(i - gi), abs(j - gj)
        turn = 0 if gk is None else min((k - gk) % HEADINGS, (gk - k) % HEADINGS)
        return dx + dy + (math.sqrt(2.0) - 2.0) * min(dx, dy) + TURN_COST * turn

    s = (i0, j0, k0)
    openq, cost, came = [(h(*s), 0.0, s)], {s: 0.0}, {}
    while openq:
        _f, g, cur = heapq.heappop(openq)
        if g > cost.get(cur, math.inf):
            continue
        i, j, k = cur
        if i == gi and j == gj and (gk is None or k == gk):
            chain = [cur]
            while cur in came:
                cur = came[cur]
                chain.append(cur)
            chain.reverse()
            path = [(c[0] * RES_M, c[1] * RES_M, heading_yaw(c[2])) for c in chain]
            return path, dict(info, ms=int((time.monotonic() - t0) * 1000), cost=round(g, 1))
        info["expanded"] += 1
        if info["expanded"] % 400 == 0 and time.monotonic() - t0 > BUDGET_S:
            return None, dict(info, why="planning took too long")
        for dk in (-1, 1):
            n = (i, j, (k + dk) % HEADINGS)
            if space.free(*n) and space.seen_ok[n[2]][i, j] and g + TURN_COST < cost.get(n, math.inf):
                cost[n], came[n] = g + TURN_COST, cur
                heapq.heappush(openq, (g + TURN_COST + h(*n), g + TURN_COST, n))
        for m, (di, dj) in enumerate(MOVES):
            off = min((m - k) % HEADINGS, (k - m) % HEADINGS)
            n = (i + di, j + dj, k)
            if not space.free(*n) or (off > 0 and not space.seen_ok[k][n[0], n[1]]):
                continue
            if (di and dj and (not space.free(i + di, j, k) or not space.free(i, j + dj, k))):
                continue
            extra = SIDE_COST[off] + (UNSEEN_COST if space.unseen[n[0], n[1]] else 0.0) \
                + NEAR_COST * max(0.0, 1.0 - float(space.clear[n[0], n[1]]) / (NEAR_M + 0.16))
            new = g + math.hypot(di, dj) * (1.0 + extra)
            if new < cost.get(n, math.inf):
                cost[n], came[n] = new, cur
                heapq.heappush(openq, (new + h(*n), new, n))
    return None, dict(info, why="no way there on the map as it is")
