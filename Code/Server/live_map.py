#!/usr/bin/env python3
"""Occupancy map built from the car's own sensors.

Each ultrasonic sweep paints its rays clear. A return that matches the arena
wall is drawn on that wall. A return much longer than both neighbours is an
opening: the cone stays empty. Only a return whose neighbours are similarly
close is the face of an obstacle. Cells start unknown.

The wheels are never used. ``python3 live_map.py`` looks once from where the
car is standing and writes route_follow_debug/live_map.png.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
WALL_TOL_M = 0.12


class LiveMap:
    def __init__(self, width_m: float, height_m: float, resolution_m: float = 0.02):
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.resolution_m = float(resolution_m)
        self.nx = max(1, int(round(self.width_m / self.resolution_m)))
        self.ny = max(1, int(round(self.height_m / self.resolution_m)))
        self.grid = np.zeros((self.ny, self.nx), np.uint8)
        self.sonic_hits: list[tuple[float, float]] = []
        self.wall_hits: list[tuple[float, float]] = []
        self.confirmed_hits: list[tuple[float, float]] = []
        # Points the bumper found and the ultrasonic missed: (hit_x, hit_y, pose_x, pose_y).
        self.blocks: list[tuple[float, float, float, float]] = []
        self._sweeping = False
        self._beams: list[tuple[float, float, float, float]] = []
        # World angle, range, and kind (wall, gap, graze, face) from the last sweep.
        self.last_beams: list[tuple[float, float, str]] = []

    def wall_distance(self, ox: float, oy: float, angle: float) -> float:
        dx, dy = math.cos(angle), math.sin(angle)
        hits = []
        if dx > 1e-6:
            hits.append((self.width_m - ox) / dx)
        if dx < -1e-6:
            hits.append(-ox / dx)
        if dy > 1e-6:
            hits.append((self.height_m - oy) / dy)
        if dy < -1e-6:
            hits.append(-oy / dy)
        return max(0.0, min(hits)) if hits else 0.0

    def _cell(self, x: float, y: float) -> tuple[int, int] | None:
        col = int(x / self.resolution_m)
        row = int(y / self.resolution_m)
        if 0 <= col < self.nx and 0 <= row < self.ny:
            return row, col
        return None

    def _paint(self, x: float, y: float, value: int, radius_m: float = 0.0) -> None:
        if value == FREE:
            cell = self._cell(x, y)
            if cell is not None and self.grid[cell] == UNKNOWN:
                self.grid[cell] = FREE
            return
        radius = max(self.resolution_m, radius_m)
        steps = int(math.ceil(radius / self.resolution_m))
        for drow in range(-steps, steps + 1):
            for dcol in range(-steps, steps + 1):
                px = x + dcol * self.resolution_m
                py = y + drow * self.resolution_m
                if dcol * dcol + drow * drow > (steps + 0.2) ** 2:
                    continue
                cell = self._cell(px, py)
                if cell is not None:
                    self.grid[cell] = OCCUPIED

    def mark_ray(self, ox: float, oy: float, angle: float, distance_m: float) -> None:
        """Clear the beam. A return near the arena wall is stored on the wall, not in front of the car."""
        if distance_m <= 0.02 or distance_m > 2.5:
            return
        wall = self.wall_distance(ox, oy, angle)
        dx, dy = math.cos(angle), math.sin(angle)
        on_wall = wall > 0.02 and distance_m >= wall - WALL_TOL_M
        end = min(distance_m, wall if wall > 0.02 else distance_m)
        if on_wall:
            end = max(0.0, wall - self.resolution_m)
        step = self.resolution_m
        count = max(0, int(end / step))
        for i in range(count):
            self._paint(ox + dx * step * i, oy + dy * step * i, FREE)
        if on_wall:
            inset = self.resolution_m * 0.5
            point = (
                min(max(ox + dx * wall, inset), self.width_m - inset),
                min(max(oy + dy * wall, inset), self.height_m - inset),
            )
            if all(math.hypot(point[0] - px, point[1] - py) > 0.04 for px, py in self.wall_hits):
                self.wall_hits.append(point)
        if self._sweeping:
            self._beams.append((ox, oy, angle, distance_m))

    def begin_sweep(self) -> None:
        self._sweeping = True
        self._beams = []

    def _clear_ray(self, ox: float, oy: float, angle: float, reach: float, half_width: float) -> None:
        """Drop obstacle points the beam now says are empty."""
        if reach <= 0.03:
            return
        dx, dy = math.cos(angle), math.sin(angle)

        def keep(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
            kept = []
            for hx, hy in points:
                along = (hx - ox) * dx + (hy - oy) * dy
                lateral = abs(-(hx - ox) * dy + (hy - oy) * dx)
                if 0.02 < along < reach and lateral <= half_width:
                    continue
                kept.append((hx, hy))
            return kept

        self.sonic_hits = keep(self.sonic_hits)
        self.confirmed_hits = keep(self.confirmed_hits)

    def commit_sweep(self) -> None:
        """Turn one left-to-right sweep into obstacle faces. An opening is not a face."""
        beams = list(self._beams)
        self._beams = []
        self._sweeping = False
        if not beams:
            return
        order = sorted(range(len(beams)), key=lambda i: beams[i][2])

        def neighbors(k: int) -> list[int]:
            found = []
            i = order[k]
            for slot in (k - 1, k + 1):
                if 0 <= slot < len(order):
                    j = order[slot]
                    if abs(beams[i][2] - beams[j][2]) <= math.radians(22):
                        found.append(j)
            return found

        neighbor_of = {order[k]: neighbors(k) for k in range(len(order))}
        kinds = ["face"] * len(beams)
        for i, (ox, oy, angle, dist) in enumerate(beams):
            wall = self.wall_distance(ox, oy, angle)
            if wall > 0.02 and dist >= wall - WALL_TOL_M:
                kinds[i] = "wall"
                continue
            nbr = neighbor_of.get(i, [])
            if nbr and dist >= min(beams[j][3] for j in nbr) + 0.25:
                kinds[i] = "gap"
        for i in range(len(beams)):
            if kinds[i] == "face" and any(kinds[j] == "gap" for j in neighbor_of.get(i, [])):
                kinds[i] = "graze"
        placed: list[tuple[float, float]] = []
        for i, (ox, oy, angle, dist) in enumerate(beams):
            # A far return with similar neighbours is open space, not a wall in front of the bumper.
            if dist > 0.40:
                continue
            place = None
            if kinds[i] == "face":
                place = angle
            elif kinds[i] == "graze":
                # Keep the corner, but on the side away from the opening so the gap stays wide enough.
                gaps = [beams[j][2] for j in neighbor_of.get(i, []) if kinds[j] == "gap"]
                if len(gaps) == 1 and abs(angle - gaps[0]) > 1e-4:
                    place = angle + math.copysign(math.radians(7.5), angle - gaps[0])
            if place is None:
                continue
            hx = ox + math.cos(place) * dist
            hy = oy + math.sin(place) * dist
            if not (0.02 < hx < self.width_m - 0.02 and 0.02 < hy < self.height_m - 0.02):
                continue
            if kinds[i] == "graze":
                placed = [p for p in placed if math.hypot(p[0] - hx, p[1] - hy) > 0.08]
            elif any(math.hypot(p[0] - hx, p[1] - hy) <= 0.06 for p in placed):
                continue
            placed.append((hx, hy))
        # This sweep replaces obstacles in the half it looked at. Points behind the car stay.
        ox, oy = beams[len(beams) // 2][0], beams[len(beams) // 2][1]
        center = math.atan2(
            sum(math.sin(angle) for _ox, _oy, angle, _dist in beams),
            sum(math.cos(angle) for _ox, _oy, angle, _dist in beams),
        )

        def in_view(hx: float, hy: float) -> bool:
            bearing = math.atan2(hy - oy, hx - ox)
            return abs(math.atan2(math.sin(bearing - center), math.cos(bearing - center))) <= math.radians(100)

        kept = [p for p in self.sonic_hits if not in_view(*p)]
        self.sonic_hits = self._keep_groups(kept + placed, len(placed))
        self.confirmed_hits = []
        self.last_beams = [(angle, dist, kinds[i]) for i, (_ox, _oy, angle, dist) in enumerate(beams)]

    def _keep_groups(self, points: list[tuple[float, float]], fresh_count: int) -> list[tuple[float, float]]:
        """Drop an old hit that sits alone. A real foam face has a neighbour, and this sweep's hits stay."""
        count = len(points)
        if count == 0:
            return []
        parent = list(range(count))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for i, (ax, ay) in enumerate(points):
            for j in range(i + 1, count):
                bx, by = points[j]
                if math.hypot(ax - bx, ay - by) <= 0.12:
                    parent[find(j)] = find(i)
        size: dict[int, int] = {}
        for index in range(count):
            root = find(index)
            size[root] = size.get(root, 0) + 1
        fresh_from = count - fresh_count
        return [point for index, point in enumerate(points) if index >= fresh_from or size[find(index)] >= 2]

    def remember_block(self, hx: float, hy: float, pose_x: float, pose_y: float) -> None:
        """The bumper hit something the ultrasonic read as open. Keep it until the car has moved on."""
        self.blocks.append((hx, hy, pose_x, pose_y))

    def prune_blocks(self, x: float, y: float) -> None:
        self.blocks = [b for b in self.blocks if math.hypot(x - b[2], y - b[3]) <= 0.35]

    def mark_points(self, points: np.ndarray, min_points: int = 2) -> None:
        """Keep a cell only when several floor points land in it. Sparse fisheye hits are dropped."""
        if points is None or len(points) == 0:
            return
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cols = np.floor(pts[:, 0] / self.resolution_m).astype(int)
        rows = np.floor(pts[:, 1] / self.resolution_m).astype(int)
        valid = (cols >= 0) & (cols < self.nx) & (rows >= 0) & (rows < self.ny)
        if not np.any(valid):
            return
        flat = rows[valid] * self.nx + cols[valid]
        counts = np.bincount(flat, minlength=self.nx * self.ny)
        for index in np.flatnonzero(counts >= min_points):
            self.grid[index // self.nx, index % self.nx] = OCCUPIED

    def _stamp(self, points: list[tuple[float, float]], radius_m: float) -> np.ndarray:
        mask = np.zeros((self.ny, self.nx), np.uint8)
        if radius_m <= 0:
            for x, y in points:
                cell = self._cell(x, y)
                if cell is not None:
                    mask[cell] = 1
            return mask
        radius = max(1, int(math.ceil(radius_m / self.resolution_m)))
        for x, y in points:
            cell = self._cell(x, y)
            if cell is None:
                continue
            row, col = cell
            mask[max(0, row - radius):row + radius + 1, max(0, col - radius):col + radius + 1] = 1
        return mask

    def interior_occupied(self) -> np.ndarray:
        """Foam inside the arena: camera and ultrasonic on the same spot, not the wall foot."""
        camera = (self.grid == OCCUPIED).astype(np.uint8)
        band = max(1, int(math.ceil(0.10 / self.resolution_m)))
        camera[:band, :] = 0
        camera[-band:, :] = 0
        camera[:, :band] = 0
        camera[:, -band:] = 0
        occupied = cv2.bitwise_and(camera, self._stamp(self.sonic_hits, 0.03))
        occupied = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(occupied, 8)
        kept = np.zeros_like(occupied)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= 2:
                kept[labels == index] = 1
        return kept

    def display_grid(self) -> np.ndarray:
        """One cell per hit. Walls on the arena edge, interior marks only where several returns agree."""
        obstacles = self._stamp(self.sonic_hits + [(b[0], b[1]) for b in self.blocks], 0.0)
        walls = self._stamp(self.wall_hits, 0.0)
        clear = (self.grid == FREE).astype(np.uint8)
        clear[obstacles > 0] = 0
        clear[walls > 0] = 0
        shown = np.zeros_like(self.grid)
        shown[clear > 0] = FREE
        shown[walls > 0] = OCCUPIED
        shown[obstacles > 0] = OCCUPIED
        return shown

    def obstacle_points(self) -> np.ndarray:
        """Interior foam the route planner treats as solid. Arena walls are already inflated there."""
        parts = []
        if self.sonic_hits:
            parts.append(np.asarray(self.sonic_hits, dtype=np.float64).reshape(-1, 2))
        if self.blocks:
            parts.append(np.array([(b[0], b[1]) for b in self.blocks], dtype=np.float64))
        if not parts:
            return np.zeros((0, 2))
        return np.vstack(parts)

    def render(self, pose=None, route=None, trail=None, guide=None, body_m=(0.29, 0.18), size_px: int = 640) -> np.ndarray:
        margin = 28
        scale = (size_px - 2 * margin) / max(self.width_m, self.height_m)
        width = int(self.width_m * scale) + 2 * margin
        height = int(self.height_m * scale) + 2 * margin
        image = np.full((height, width, 3), 228, np.uint8)
        colors = {
            UNKNOWN: (228, 228, 228),
            FREE: (252, 252, 252),
            OCCUPIED: (180, 90, 30),
        }

        def pixel(x: float, y: float) -> tuple[int, int]:
            return int(round(margin + x * scale)), int(round(height - margin - y * scale))

        shown = self.display_grid()
        for row in range(self.ny):
            for col in range(self.nx):
                value = int(shown[row, col])
                if value == UNKNOWN:
                    continue
                x0, y1 = pixel(col * self.resolution_m, (row + 1) * self.resolution_m)
                x1, y0 = pixel((col + 1) * self.resolution_m, row * self.resolution_m)
                cv2.rectangle(image, (x0, y0), (x1, y1), colors[value], -1)
        cv2.rectangle(image, pixel(0, self.height_m), pixel(self.width_m, 0), (40, 40, 40), 2)
        if route:
            pts = [pixel(x, y) for x, y in route]
            for a, b in zip(pts, pts[1:]):
                cv2.line(image, a, b, (200, 160, 0), 1, cv2.LINE_AA)
        if guide and len(guide) >= 2:
            pts = [pixel(x, y) for x, y in guide]
            for a, b in zip(pts, pts[1:]):
                cv2.line(image, a, b, (40, 140, 40), 2, cv2.LINE_AA)
        if trail and len(trail) >= 2:
            pts = [pixel(x, y) for x, y, _yaw in trail]
            for a, b in zip(pts, pts[1:]):
                cv2.line(image, a, b, (40, 40, 200), 1, cv2.LINE_AA)
        if pose is not None:
            length, width_body = body_m
            half_l, half_w = length / 2.0, width_body / 2.0
            c, s = math.cos(pose.yaw), math.sin(pose.yaw)
            corners = []
            for lx, ly in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
                corners.append(pixel(pose.x + c * lx - s * ly, pose.y + s * lx + c * ly))
            cv2.fillConvexPoly(image, np.array(corners, np.int32), (60, 60, 210))
            nose = pixel(pose.x + c * (half_l + 0.06), pose.y + s * (half_l + 0.06))
            cv2.arrowedLine(image, pixel(pose.x, pose.y), nose, (20, 20, 140), 2, tipLength=0.35)
        cv2.putText(image, "live map", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        return image
