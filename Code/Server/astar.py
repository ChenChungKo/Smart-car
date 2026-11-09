#!/usr/bin/env python3
"""A* on a grid of the arena for a car that keeps one heading (a mecanum car moves sideways without turning).

A cell is free when the whole body, plus a margin, at that cell stays clear of the walls and the blocks. Cells close to
something cost more, so the path does not run along the edge of what is allowed; the path is then shortened with
straight lines that stay on free cells.
"""
from __future__ import annotations

import heapq
import math

import cv2
import numpy as np

RES_M = 0.01
NEAR_COST_M = 0.08   # within this of a forbidden cell a step costs up to twice as much
LANE_COST_PER_M = 50.0   # with a lane to follow: a step costs (1 + this x distance from the lane) times as much


class Grid:
    def __init__(self, arena_m: tuple[float, float], blocks: list, half_l: float, half_w: float, yaw: float, margin: float,
                 wall_margin: float | None = None):
        self.w, self.h = float(arena_m[0]), float(arena_m[1])
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        self.ex = c * half_l + s * half_w + margin    # half extent of the body along x and y of the arena, with the margin
        self.ey = s * half_l + c * half_w + margin
        wall = margin if wall_margin is None else wall_margin   # kept from the walls; the blocks keep `margin`
        wx, wy = self.ex - margin + wall, self.ey - margin + wall
        self.nx, self.ny = int(round(self.w / RES_M)) + 1, int(round(self.h / RES_M)) + 1
        xs, ys = np.arange(self.nx) * RES_M, np.arange(self.ny) * RES_M
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        blocked = (X < wx) | (X > self.w - wx) | (Y < wy) | (Y > self.h - wy)
        for x0, x1, y0, y1 in blocks:
            blocked |= (X > x0 - self.ex) & (X < x1 + self.ex) & (Y > y0 - self.ey) & (Y < y1 + self.ey)
        self.blocked = blocked
        self.clear = cv2.distanceTransform((~blocked).astype(np.uint8), cv2.DIST_L2, 5).astype(float) * RES_M

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x / RES_M)), int(round(y / RES_M))

    def world(self, i: int, j: int) -> tuple[float, float]:
        return i * RES_M, j * RES_M

    def free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[i, j]

    def nearest_free(self, i: int, j: int, reach_m: float):
        best, reach = None, int(reach_m / RES_M)
        for a in range(-reach, reach + 1):
            for b in range(-reach, reach + 1):
                if self.free(i + a, j + b) and (best is None or a * a + b * b < best[0]):
                    best = (a * a + b * b, i + a, j + b)
        return None if best is None else (best[1], best[2])

    def line_free(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (RES_M / 2)))
        for k in range(n + 1):
            x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
            if not self.free(*self.cell(x, y)):
                return False
        return True


def plan(grid: Grid, start: tuple[float, float], goal: tuple[float, float], snap_m: float = 0.06, lane=None):
    """The path [(x, y), ...] in metres from start to goal, or None. The start may be snapped this far to a free cell.

    lane: ((x0, y0), (x1, y1)): the path keeps to this straight line and leaves it only to go around what is in the way."""
    s, g = grid.cell(*start), grid.cell(*goal)
    if not grid.free(*s):
        s = grid.nearest_free(*s, snap_m)
    if s is None or not grid.free(*g):
        return None
    def h(c):
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)
    def off_lane(c):
        (x0, y0), (x1, y1) = lane
        x, y = grid.world(*c)
        return abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / max(math.hypot(x1 - x0, y1 - y0), 1e-9)

    openq = [(h(s), 0.0, s)]
    came, cost = {}, {s: 0.0}
    while openq:
        _f, gc, cur = heapq.heappop(openq)
        if cur == g:
            cells = [cur]
            while cur in came:
                cur = came[cur]
                cells.append(cur)
            cells.reverse()
            points = [grid.world(*c) for c in cells]
            points = [start] + points if start != points[0] else points
            return compress(points) if lane else shorten(grid, points)
        if gc > cost.get(cur, math.inf):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if not di and not dj:
                    continue
                n = (cur[0] + di, cur[1] + dj)
                if not grid.free(*n) or (di and dj and (not grid.free(cur[0] + di, cur[1]) or not grid.free(cur[0], cur[1] + dj))):
                    continue
                extra = LANE_COST_PER_M * off_lane(n) if lane else max(0.0, 1.0 - grid.clear[n] / NEAR_COST_M)
                step = math.hypot(di, dj) * (1.0 + extra)
                new = gc + step
                if new < cost.get(n, math.inf):
                    cost[n], came[n] = new, cur
                    heapq.heappush(openq, (new + h(n), new, n))
    return None


def compress(points: list) -> list:
    """The corners of a grid path: points in a straight run are dropped."""
    out = [points[0]]
    for a, b, c in zip(points, points[1:], points[2:]):
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-9:
            out.append(b)
    return out + [points[-1]]


def shorten(grid: Grid, points: list) -> list:
    """Straight lines over free cells instead of grid steps. The first point may be the (snapped) start: it is kept."""
    out, i = [points[0]], 1 if len(points) > 1 else 0
    if len(points) > 1:
        out.append(points[1])
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not grid.line_free(points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


def expected_range(x: float, y: float, yaw: float, arena_m: tuple[float, float], blocks: list, half_deg: float = 12.0) -> float:
    """Nearest wall or known block seen from (x, y) along yaw within the beam of the ultrasonic (m)."""
    w, h = float(arena_m[0]), float(arena_m[1])
    best = math.inf
    for off in np.radians(np.arange(-half_deg, half_deg + 0.1, 2.0)):
        dx, dy = math.cos(yaw + off), math.sin(yaw + off)
        rects = [(0.0, w, 0.0, h, True)] + [(b[0], b[1], b[2], b[3], False) for b in blocks]
        for x0, x1, y0, y1, inside in rects:
            hits = []
            for t, ok in (((x0 - x) / dx if abs(dx) > 1e-9 else -1.0, 0), ((x1 - x) / dx if abs(dx) > 1e-9 else -1.0, 0),
                          ((y0 - y) / dy if abs(dy) > 1e-9 else -1.0, 1), ((y1 - y) / dy if abs(dy) > 1e-9 else -1.0, 1)):
                if t > 1e-6:
                    px, py = x + t * dx, y + t * dy
                    if (ok == 0 and y0 - 1e-6 <= py <= y1 + 1e-6) or (ok == 1 and x0 - 1e-6 <= px <= x1 + 1e-6):
                        hits.append(t)
            if hits:
                best = min(best, min(hits))
    return best
