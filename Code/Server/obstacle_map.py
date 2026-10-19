#!/usr/bin/env python3
"""Obstacles from camera foot points: which are walls, what is in the car's path, and the map.

All of it is plain geometry on points, so it runs the same on the car, in route_sim.py and in the
lane_scan.py check.

- `split_walls` keeps the foot points that are not the arena walls. The foam walls are foam too and the
  camera sees their feet; a point within WALL_BAND_M of a wall (the pose is good to a few cm) is the wall.
  A block standing out from a wall is still found by the rest of its face, which is 10-20 cm long.
- `in_path` / `path_gap` / `shift_to_clear` work in the car frame: the corridor the body sweeps going
  straight ahead, how far ahead the nearest point is, and how far the car must strafe to be clear.
- `ObstacleMap` keeps the blocks seen so far in arena coordinates (fitted rectangle, or the face
  seen out to the wall it stands against) and draws them over route_follow's trajectory map.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

WALL_BAND_M = 0.07   # a foot this close to a wall is the wall (more for far feet, see `wall_band`)
CAMERA_X_M = 0.10    # the gimbal camera, ahead of the car centre: far feet are less exact
DEPTH_20_FROM_M = 0.125   # a face seen this far from the wall (95th percentile) is the end of a block standing out from it
MAP_RANGE_M = 0.36   # only feet this near the camera make blocks: farther, a wall's foot is off by more than its band
OUTSIDE_M = 0.04     # a foot this far beyond a wall is something outside the arena (people, a fridge)
CLUSTER_GAP_M = 0.045
BLOCK_REACH_M = 0.22     # foot points farther than this from the middle of a cluster are not the same block
BLOCK_AREA_MAX_M2 = 0.035  # a 20 x 10 cm block is 0.02; the carried-out outline of one may not be much bigger
MERGE_M = 0.22       # two looks this close are one block: the pose drifts 10-20 cm between lanes


def to_world(points: np.ndarray, pose) -> np.ndarray:
    """Car-frame points to arena coordinates for a pose with x, y, yaw."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    p = np.asarray(points, float).reshape(-1, 2)
    return np.column_stack([pose.x + c * p[:, 0] - s * p[:, 1], pose.y + s * p[:, 0] + c * p[:, 1]])


def to_car(points: np.ndarray, pose) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    d = np.asarray(points, float).reshape(-1, 2) - [pose.x, pose.y]
    return np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])


def inner_distance(world: np.ndarray, size) -> np.ndarray:
    """Distance of each point to the nearest wall, negative beyond it."""
    w, h = size
    return np.minimum.reduce([world[:, 0], w - world[:, 0], world[:, 1], h - world[:, 1]])


def wall_band(points_car: np.ndarray) -> np.ndarray:
    """How close to a wall a foot must be to count as the wall: wider for feet far from the camera, where the
    floor point is less exact (a degree of tilt moves a foot 4 cm at 40 cm, 6 cm at 60 cm)."""
    reach = np.hypot(points_car[:, 0] - CAMERA_X_M, points_car[:, 1])
    return WALL_BAND_M + 0.25 * np.maximum(reach - 0.25, 0.0)


def split_walls(points_car: np.ndarray, pose, size) -> tuple[np.ndarray, np.ndarray]:
    """(obstacle points, wall points), both in the car frame; points outside the arena are dropped."""
    p = np.asarray(points_car, float).reshape(-1, 2)
    p = p[~np.isnan(p[:, 0])]
    if len(p) == 0:
        return p, p
    d = inner_distance(to_world(p, pose), size)
    band = wall_band(p)
    return p[d > band], p[(d <= band) & (d > -OUTSIDE_M - band)]


def in_path(points: np.ndarray, nose_x: float, half_width: float) -> np.ndarray:
    """Mask of the points the body would run into going straight ahead (car frame)."""
    p = np.asarray(points, float).reshape(-1, 2)
    return (p[:, 0] >= nose_x - 0.01) & (np.abs(p[:, 1]) <= half_width)


def path_gap(points: np.ndarray, nose_x: float, half_width: float) -> float | None:
    """Distance from the nose to the face of the nearest thing in the path, or None.

    The nearest single point is biased near: the points of a flat face scatter by a few cm, and the one that
    scatters closest is not the face. The median of the points within 5 cm of the nearest is the face.
    """
    p = np.asarray(points, float).reshape(-1, 2)
    hit = in_path(p, nose_x, half_width)
    if not hit.any():
        return None
    x = p[hit, 0]
    return float(np.median(x[x <= x.min() + 0.05]) - nose_x)


def shift_to_clear(points: np.ndarray, out: float, nose_x: float, half_width: float, margin: float) -> float:
    """How far to strafe toward `out` (+1 left, -1 right) so no point is in the path any more; 0 if clear.

    A point at lateral position u (positive toward `out`) moves to u - s when the car moves s: it is out
    of the path once u - s <= -(half_width + margin).
    """
    p = np.asarray(points, float).reshape(-1, 2)
    hit = in_path(p, nose_x, half_width + margin)
    if not hit.any():
        return 0.0
    return float((out * p[hit, 1]).max() + half_width + margin)


def lane_coords(point, lane) -> tuple[float, float]:
    """(along, across) of an arena point relative to a lane pose (x, y, yaw); across is positive to the left."""
    dx, dy = point[0] - lane[0], point[1] - lane[1]
    c, s = math.cos(lane[2]), math.sin(lane[2])
    return dx * c + dy * s, -dx * s + dy * c


def lane_report(points: np.ndarray, lanes, half_width: float) -> list[dict]:
    """Which lanes the points block. `lanes` is [(name, (x0, y0), (x1, y1))] in driving order.

    For every lane with points inside the body's swept width: how far along the lane the first one is
    (from its start) and the lateral range they cover relative to the lane line (+ = left of the way
    the car drives). The route is what the car drives; this is what the obstacles do to it.
    """
    out = []
    p = np.asarray(points, float).reshape(-1, 2)
    for name, a, b in lanes:
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.hypot(*(b - a)))
        if length < 1e-6 or len(p) == 0:
            continue
        u = (b - a) / length
        d = p - a
        along, across = d @ u, -d[:, 0] * u[1] + d[:, 1] * u[0]
        hit = (along >= 0.0) & (along <= length) & (np.abs(across) <= half_width)
        if hit.any():
            out.append({"lane": name, "first_along_m": float(along[hit].min()), "last_along_m": float(along[hit].max()),
                        "across_min_m": float(across[hit].min()), "across_max_m": float(across[hit].max()),
                        "lane_length_m": length})
    return out


class ObstacleMap:
    def __init__(self, size, block=(0.20, 0.10)):
        self.w, self.h = (float(v) for v in size)
        self.block = (max(block), min(block))  # long and short side of one foam block
        self.blocks: list[dict] = []
        self.next_id = 1
        self.scan_obstacle = np.zeros((0, 2))
        self.scan_wall = np.zeros((0, 2))
        self.scans = 0

    # ---- building the map -------------------------------------------------

    def add(self, obstacle_car: np.ndarray, wall_car: np.ndarray, pose) -> list[dict]:
        """Fold one scan (car-frame points) into the blocks; returns the blocks seen in this scan."""
        self.scans += 1
        world = to_world(obstacle_car, pose) if len(obstacle_car) else np.zeros((0, 2))
        self.scan_obstacle = world
        self.scan_wall = to_world(wall_car, pose) if len(wall_car) else np.zeros((0, 2))
        seen = []
        for cluster in self._clusters(world):
            centre = cluster.mean(axis=0)
            near = min(self.blocks, key=lambda b: float(np.hypot(*(b["centre"] - centre))), default=None)
            if near is not None and float(np.hypot(*(near["centre"] - centre))) < MERGE_M:
                block = near
                block["obs"] = (block["obs"] + [cluster])[-3:]
                block["hits"] += 1
            else:
                block = {"id": self.next_id, "obs": [cluster], "hits": 1, "first_scan": self.scans}
                self.next_id += 1
                self.blocks.append(block)
            block["last_scan"] = self.scans
            # Decisions (the detour's `passing` points) use the plain rule: the latest look if it has 8 points,
            # else all of the last three.
            pts = block["obs"][-1] if len(block["obs"]) and len(block["obs"][-1]) >= 8 else np.vstack(block["obs"])
            block["points"], block["centre"] = pts, pts.mean(axis=0)
            # The drawing uses only feet near the camera (far feet of a wall land off the wall and make corner
            # blocks) and keeps its own outline; it never changes what the car decides.
            cam = to_world(np.array([[CAMERA_X_M, 0.0]]), pose)[0]
            near_pts = cluster[np.hypot(cluster[:, 0] - cam[0], cluster[:, 1] - cam[1]) <= MAP_RANGE_M]
            if len(near_pts) >= 5:
                block["near_hits"] = block.get("near_hits", 0) + 1
                block["draw_obs"] = (block.get("draw_obs", []) + [near_pts])[-3:]
                # Only the latest look draws, unless it is thin (next to the block the camera is blind: 7 points
                # in a 2 cm bunch must not overwrite the 30 points of the look from 20 cm away).
                latest = block["draw_obs"][-1]
                fullest = max(block["draw_obs"], key=len)
                dpts = latest if (len(latest) >= 15 or len(latest) >= 0.6 * len(fullest)) else fullest
                block["draw_points"] = dpts
                block["outline"] = self._outline(dpts)
            seen.append(block)
        return seen

    @staticmethod
    def _clusters(world: np.ndarray, minimum: int = 5) -> list[np.ndarray]:
        if len(world) < minimum:
            return []
        cells = np.round(world * 100.0).astype(int)  # 1 cm cells
        cells -= cells.min(axis=0)
        grid = np.zeros((cells[:, 1].max() + 1, cells[:, 0].max() + 1), np.uint8)
        grid[cells[:, 1], cells[:, 0]] = 255
        k = int(round(CLUSTER_GAP_M * 100.0 / 2.0)) * 2 + 1
        grid = cv2.dilate(grid, np.ones((k, k), np.uint8))
        count, labels = cv2.connectedComponents(grid)
        out = []
        for label in range(1, count):
            members = labels[cells[:, 1], cells[:, 0]] == label
            if members.sum() >= minimum:
                out.append(world[members])
        return out

    def _wall_box(self, pts: np.ndarray) -> np.ndarray | None:
        """A block standing against a wall, drawn at its real size, or None when the points are not that.

        The car sees the face towards it and the foot points scatter by a few cm, so their extent is 19 cm
        wide for a block 10 cm wide, and the side towards the wall is not seen at all. What is known: the
        block is 20 x 10 cm and touches the wall. Its DEPTH (from the wall out to the face the car sees) is
        10 or 20: the nearer of the two to what was seen. Its WIDTH along the wall is then the other one, centred
        on the foot points.
        """
        if len(pts) < 5:
            return None
        mid = np.median(pts, axis=0)
        # the wall nearest the block: (axis across the wall, coordinate of the wall)
        axis, wall_at = min(((0, 0.0), (0, self.w), (1, 0.0), (1, self.h)), key=lambda t: abs(mid[t[0]] - t[1]))
        if abs(mid[axis] - wall_at) > 0.22:
            return None  # not against a wall
        seen_depth = float(np.percentile(np.abs(pts[:, axis] - wall_at), 95))
        if seen_depth > 0.28:
            return None  # deeper than a block: two blocks, or a wall
        depth = self.block[0] if seen_depth >= DEPTH_20_FROM_M else self.block[1]
        width = self.block[1] if depth == self.block[0] else self.block[0]
        along = 1 - axis
        centre = float(np.median(pts[:, along]))
        reach = self.h if along == 1 else self.w
        lo = min(max(centre - width / 2.0, 0.0), reach - width)  # kept inside the arena
        hi = lo + width
        n_lo, n_hi = (0.0, depth) if wall_at == 0.0 else (wall_at - depth, wall_at)
        if axis == 0:   # wall at x = 0 or x = w: the block runs out from it in x and spreads along y
            return np.array([[n_lo, lo], [n_hi, lo], [n_hi, hi], [n_lo, hi]])
        return np.array([[lo, n_lo], [hi, n_lo], [hi, n_hi], [lo, n_hi]])

    def _outline(self, pts: np.ndarray) -> np.ndarray:
        """Polygon of a block: the foot line seen, carried out to the wall it stands against (if close).

        A block is 20 x 10 cm. Points far from the rest (a stray foot point, a bit of wall) are dropped, and the
        carry-out to the wall is skipped when it would make an outline larger than a block could be.
        """
        centre = np.median(pts, axis=0)
        near = np.hypot(*(pts - centre).T) <= BLOCK_REACH_M
        pts = pts[near] if near.sum() >= 4 else pts
        box = self._wall_box(pts)
        if box is not None:
            return box
        plain = cv2.convexHull((pts * 100.0).astype(np.float32)).reshape(-1, 2) / 100.0
        hull_pts = pts
        d = inner_distance(pts, (self.w, self.h))
        if d.min() <= 0.22:
            nearest = pts[int(np.argmin(d))]
            # (axis, coordinate of the wall) for the four walls; take the one this block stands against
            walls = [(0, 0.0, nearest[0]), (0, self.w, self.w - nearest[0]),
                     (1, 0.0, nearest[1]), (1, self.h, self.h - nearest[1])]
            axis, wall_value, _ = min(walls, key=lambda t: t[2])
            proj = pts.copy()
            proj[:, axis] = wall_value
            hull_pts = np.vstack([pts, proj])
        hull = cv2.convexHull((hull_pts * 100.0).astype(np.float32)).reshape(-1, 2) / 100.0
        area = float(cv2.contourArea((hull * 100.0).astype(np.float32))) / 1e4
        return hull if area <= BLOCK_AREA_MAX_M2 else plain

    # ---- reading it -------------------------------------------------------

    def describe(self, block: dict, lane=None) -> str:
        cx, cy = block["centre"]
        text = f"B{block['id']} at ({cx * 100:.0f},{cy * 100:.0f}) cm, seen {block['hits']}x"
        if lane is not None:
            along, across = lane_coords((cx, cy), lane)
            text += (f"; {abs(along) * 100:.0f} cm {'before' if along < 0 else 'past'} the lane's target point, "
                     f"{abs(across) * 100:.0f} cm to the {'left' if across > 0 else 'right'} of the lane line")
        return text

    def as_dict(self) -> list[dict]:
        return [{"id": b["id"], "hits": b["hits"], "centre_m": [round(float(v), 3) for v in b["centre"]],
                 "outline_m": np.round(b["outline"], 3).tolist()} for b in self.blocks if "outline" in b]

    # ---- drawing ----------------------------------------------------------

    def draw(self, img: np.ndarray, size_px: int = 640, margin: int = 30, min_hits: int = 3,
             show_walls: bool = False) -> np.ndarray:
        """Draw onto route_follow's map image (aruco_localizer.draw_map: margin 30, 640 px)."""
        scale = (size_px - 2 * margin) / max(self.w, self.h)
        floor_h = int(self.h * scale) + 2 * margin

        def px(p):
            return int(round(margin + p[0] * scale)), int(round(floor_h - margin - p[1] * scale))

        if show_walls:  # the foot points taken for the arena walls: a check of the pose, not of the obstacles
            for p in self.scan_wall:
                cv2.circle(img, px(p), 1, (170, 170, 170), -1)
        shown = 0
        for block in self.blocks:
            if block.get("near_hits", 0) < min_hits or "outline" not in block:
                continue  # seen once: far feet and wall edges look like blocks; wait for a second look
            shown += 1  # numbered by what is drawn: ids of one-look false blocks are not shown, so no gaps
            fresh = block.get("last_scan", 0) >= self.scans - 1
            color = (0, 0, 220) if fresh else (110, 110, 200)
            poly = np.array([px(p) for p in block["outline"]], np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
            cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
            for p in block["draw_points"]:
                cv2.circle(img, px(p), 1, (0, 120, 255), -1)
            u, v = px(block["draw_points"].mean(axis=0))
            cv2.putText(img, f"B{shown}", (u - 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3,
                        cv2.LINE_AA)
            cv2.putText(img, f"B{shown}", (u - 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                        cv2.LINE_AA)
        for p in self.scan_obstacle:
            cv2.circle(img, px(p), 2, (0, 60, 255), -1)
        return img
