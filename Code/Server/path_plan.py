"""Grid path planning for the arena.

Walls and sensed obstacle points are inflated by the car's half-width. An 8-connected A*
search then returns a centreline the body can follow. The grid is only 5 cm, so one plan
over the 140 cm arena is a few thousand cells.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

RESOLUTION_M = 0.05


class Plan:
    def __init__(self, path: list[tuple[float, float]], goal_blocked: bool):
        self.path = path
        self.goal_blocked = goal_blocked

    def __bool__(self) -> bool:
        return bool(self.path)


def plan_path(
    start: tuple[float, float],
    goal: tuple[float, float],
    arena_m: tuple[float, float],
    obstacles: np.ndarray,
    inflate_m: float,
    resolution_m: float = RESOLUTION_M,
) -> Plan:
    """Shortest body-centre path. An occupied waypoint is not skipped: the path ends at the
    closest reachable cell and `goal_blocked` says the waypoint itself still has to be approached."""
    width, height = arena_m
    nx = max(1, int(math.ceil(width / resolution_m)))
    ny = max(1, int(math.ceil(height / resolution_m)))
    blocked = np.zeros((ny, nx), dtype=bool)
    inflate = max(inflate_m, resolution_m)

    def cell(x: float, y: float) -> tuple[int, int] | None:
        ix, iy = int(x / resolution_m), int(y / resolution_m)
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None

    def centre(ix: int, iy: int) -> tuple[float, float]:
        return (ix + 0.5) * resolution_m, (iy + 0.5) * resolution_m

    for iy in range(ny):
        for ix in range(nx):
            x, y = centre(ix, iy)
            if min(x, width - x, y, height - y) < inflate:
                blocked[iy, ix] = True
    if len(obstacles):
        reach = inflate + resolution_m
        for ox, oy in np.asarray(obstacles, dtype=np.float64).reshape(-1, 2):
            x0, x1 = int((ox - reach) / resolution_m), int((ox + reach) / resolution_m) + 1
            y0, y1 = int((oy - reach) / resolution_m), int((oy + reach) / resolution_m) + 1
            for iy in range(max(0, y0), min(ny, y1)):
                for ix in range(max(0, x0), min(nx, x1)):
                    x, y = centre(ix, iy)
                    if math.hypot(x - ox, y - oy) <= inflate:
                        blocked[iy, ix] = True

    start_cell = cell(*start)
    goal_cell = cell(*goal)
    if start_cell is None or goal_cell is None:
        return Plan([], False)
    # The car already occupies its own cell, even when wall inflation covers that cell.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            ix, iy = start_cell[0] + dx, start_cell[1] + dy
            if 0 <= ix < nx and 0 <= iy < ny:
                blocked[iy, ix] = False
    open_heap = [(0.0, 0.0, start_cell)]
    cost = {start_cell: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    while open_heap:
        _score, current_cost, current = heapq.heappop(open_heap)
        if current_cost > cost.get(current, math.inf):
            continue
        if current == goal_cell:
            break
        cx, cy = current
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nxt = cx + dx, cy + dy
                if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or blocked[nxt[1], nxt[0]]:
                    continue
                step = math.hypot(dx, dy) * resolution_m
                nxt_cost = current_cost + step
                if nxt_cost < cost.get(nxt, math.inf):
                    cost[nxt] = nxt_cost
                    came[nxt] = current
                    heuristic = math.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1]) * resolution_m
                    heapq.heappush(open_heap, (nxt_cost + heuristic, nxt_cost, nxt))
    goal_blocked = goal_cell not in cost
    if goal_blocked:
        # Keep the waypoint. Stop the grid path at the reachable cell nearest to it.
        goal_cell = min(cost, key=lambda c: math.hypot(*np.subtract(centre(*c), goal)))
    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(came[cells[-1]])
    cells.reverse()
    return Plan([centre(*c) for c in cells], goal_blocked)


def choose_exit(readings: list[tuple[int, float]], prefer_deg: float, min_room_m: float = 0.30,
                held_deg: int | None = None) -> int | None:
    """Angle to face. `readings` are `(angle_deg, clear_room_m)`.

    The chosen angle is the one closest to the waypoint bearing that has enough clear room.
    A previously chosen angle is kept while it stays clear, so the car does not reverse its turn
    on the next scan.
    """
    if held_deg is not None:
        held = next((room for angle, room in readings if angle == held_deg), 0.0)
        if held >= min_room_m:
            return held_deg
    usable = [(angle, room) for angle, room in readings if room >= min_room_m]
    if not usable:
        return None
    return min(usable, key=lambda item: (abs(item[0] - prefer_deg), -item[1]))[0]


def lookahead(path: list[tuple[float, float]], distance_m: float) -> tuple[float, float]:
    """Point `distance_m` along `path`, or the path's end when the path is shorter."""
    if len(path) < 2:
        return path[-1]
    left = distance_m
    for a, b in zip(path, path[1:]):
        step = math.hypot(b[0] - a[0], b[1] - a[1])
        if step >= left:
            scale = 0.0 if step == 0.0 else left / step
            return a[0] + (b[0] - a[0]) * scale, a[1] + (b[1] - a[1]) * scale
        left -= step
    return path[-1]
