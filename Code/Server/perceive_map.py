#!/usr/bin/env python3
"""Stationary surroundings map: one gimbal sweep of camera + ultrasonic, placed by the tags.

The car never drives. Steps:
1. Sense. Grab the front and rear CSI frames, then pan the gimbal across the front. At each
   angle the gimbal camera gives the line where blue foam meets the floor, the ultrasonic
   (same bracket) gives one range, and any tag in the gimbal view is kept.
2. Localize. The front and rear CSI cameras are bolted to the body, so their tags give the pose.
   They also give the heading that the gimbal tags are compared with: the median difference is how
   far the gimbal really turned past the command in this sweep (the servo does not come back to the
   same angle), and every gimbal view is redone at the corrected angle. Only when the CSI cameras
   see no usable tag do the gimbal tags give the pose. When a tag within 1 m is seen, farther tags
   are dropped, and a pose that puts the body through a wall is refused.
3. Map. Camera foot points near the arena edge are wall. The rest are grouped (within 4 cm) and
   each group is fitted with the known foam block (arena_map.json obstacle_block_m): the visible
   faces lie on the rectangle's outline and the hidden part is behind them. A group wider than one
   block (two blocks touching, foam against a wall) keeps its raw points. The ultrasonic does not
   draw obstacles: a return that lands on a block confirms it, one that lands on no block is kept
   as "sonic only", and a return beyond the wall (glancing echo off a slanted face) is ignored.
4. Print each block (world cm, heading, distance from the car body) and the open headings the body
   fits through.

The front CSI floor scan is drawn for comparison only; it does not vote.

    python3 perceive_map.py
    python3 perceive_map.py --pose 0.70,0.70,0      # skip tags, use a known pose
    python3 perceive_map.py --pans -60 -30 0 30 60
    python3 perceive_map.py --replay --pose 0.185,0.14,0   # rerun on the saved raw data, no hardware
    python3 perceive_map.py --replay perceive_debug/history/20261018-203000

Output: perceive_debug/perceive_map.png, perceive_gimbal.jpg, perceive_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from aruco_localizer import (CAR_PATH, MAP_PATH, ArucoLocalizer, CsiGrabber, Pose2D, TagFix, car_from_camera,
                             load_json, wrap_rad)
from block_map import BlockMap, rect_corners, rect_distance
from floor_scan import FloorScanner, to_world
from gimbal_gap import HEIGHT, GimbalCamera, Rig, draw_frame

SERVER = Path(__file__).resolve().parent
OUT = SERVER / "perceive_debug"
RAW = OUT / "raw"          # the last run, read by --replay
HISTORY = OUT / "history"  # every live run kept by its start time, for replaying older scenes
RES = 0.02
NEAR_TAG_M = 1.0
UNSHAPED_TEXT = {"cut": "沒看完整（太近超出畫面或在畫面邊緣）", "few": "點太少",
                 "wide": "比一塊方塊大（可能兩塊相靠）", "far_one_face": "只看到一個面又太遠，方向不確定",
                 "through_wall": "套出來會穿過牆"}
SONIC_TEXT = {"wall": "打到牆", "obstacle": "比牆近，是障礙物", "long": "比牆遠，不採信",
              "far": "超過範圍，不採用", "none": "沒有讀數"}
SOLID_LINE_M = 0.08  # a foot line at least this long is a block face; shorter scatter is weak
ONE_FACE_REACH_M = 0.50  # a block seen by one face is boxed only this close to the car centre
DEFAULT_SONIC = {"pivot_x_m": 0.085, "arm_m": 0.03}
UNKNOWN, FREE, BLUE, GREY, SONIC = range(5)
WALL_COLOR = (60, 100, 140)
COLORS = {
    FREE: (250, 250, 250),
    BLUE: (205, 130, 40),
    GREY: (120, 120, 120),
    SONIC: (200, 60, 160),
}


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose", default=None, help="Known pose 'x,y,yaw_deg' instead of the tags.")
    p.add_argument("--new-map", action="store_true",
                   help="Forget the remembered blocks (perceive_debug/block_map.json) and start a new map.")
    p.add_argument("--replay", nargs="?", const=str(RAW), default=None, metavar="DIR",
                   help="No hardware: rerun the analysis on a saved run (default: the last one, perceive_debug/raw/; "
                        "older runs are in perceive_debug/history/).")
    p.add_argument("--localize-s", type=float, default=2.0, help="Seconds of CSI frames to collect tags from.")
    p.add_argument("--pans", type=int, nargs="+", default=list(range(-90, 91, 15)),
                   help="Gimbal angles (deg, positive = left).")
    p.add_argument("--settle-s", type=float, default=0.6, help="Wait after each pan before reading.")
    p.add_argument("--max-range-m", type=float, default=0.6,
                   help="Ignore foam feet farther than this from the gimbal camera; past it a degree of tilt "
                        "is over 5 cm and the points get sparse.")
    p.add_argument("--view-deg", type=float, default=40.0,
                   help="Use foam feet only this far off the gimbal camera axis (deg).")
    p.add_argument("--wall-band-m", type=float, default=0.15,
                   help="A foot this close to an arena wall, running along it, is the wall.")
    p.add_argument("--wall-tol-m", type=float, default=0.10,
                   help="An ultrasonic range within this of the predicted wall is the wall.")
    p.add_argument("--beam-deg", type=float, default=12.0, help="Ultrasonic half cone.")
    p.add_argument("--sonic-max-m", type=float, default=0.30,
                   help="Use an ultrasonic reading only up to this range (from the sensor face).")
    p.add_argument("--agree-m", type=float, default=0.06,
                   help="Camera foot and ultrasonic return within this count as the same obstacle.")
    p.add_argument("--margin-m", type=float, default=0.03, help="Clearance wanted on each side of the body.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    return p.parse_args(argv)


# ---- pose ---------------------------------------------------------------


def csi_fixes(loc: ArucoLocalizer, grabbers: dict, seconds: float) -> list[TagFix]:
    """Tags from the front and rear CSI over `seconds`; the latest fix of each tag per camera."""
    deadline = time.monotonic() + seconds
    seen = {name: 0.0 for name in grabbers}
    latest: dict[tuple[str, int], TagFix] = {}
    while time.monotonic() < deadline:
        for name, grabber in grabbers.items():
            frame, stamp = grabber.latest()
            if frame is None or stamp == seen[name]:
                continue
            seen[name] = stamp
            for fix in loc.observe(name, frame)[0]:
                latest[(fix.camera, fix.tag_id)] = fix
        time.sleep(0.05)
    return list(latest.values())


def body_inside(pose: Pose2D, body_m, w: float, h: float, tol: float = 0.02) -> bool:
    half_l, half_w = body_m[0] / 2.0, body_m[1] / 2.0
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    for lx, ly in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
        x, y = pose.x + c * lx - s * ly, pose.y + s * lx + c * ly
        if not (-tol <= x <= w + tol and -tol <= y <= h + tol):
            return False
    return True


def solve_pose(loc: ArucoLocalizer, fixes: list[TagFix], body_m, w: float, h: float) -> tuple[Pose2D | None, str]:
    """One joint fit. Close tags only when there are any; the body must fit inside the arena.

    A tag a hand's width away fills the frame, so a small mount error fails the joint fit; then the
    sharpest single tag that lands inside the arena is used, as route_follow does.
    """
    if not fixes:
        return None, "沒有看到任何標籤"
    near = [f for f in fixes if f.distance <= NEAR_TAG_M]
    pose = loc.fuse(near or fixes)
    if pose is None:
        why = loc.last_reject
        pose = loc._single_tag_pose(near or fixes, time.monotonic())
        if pose is None:
            return None, reject_text(why)
    if not body_inside(pose, body_m, w, h):
        return None, f"車身會超出場地（{pose_text(pose)}）"
    return pose, ""


CAMERA_TEXT = {"front": "車頭", "rear": "車尾", "left": "左側", "right": "右側"}
REJECT_TEXT = {"no tags": "沒有看到標籤", "tags disagree": "標籤彼此對不起來", "pose outside arena": "算出的位置在場地外"}


def camera_text(name: str) -> str:
    return CAMERA_TEXT.get(name, "雲台" if name.startswith("gimbal") else name)


def tag_list(fixes: list[TagFix]) -> str:
    return "、".join(f"{camera_text(f.camera)}看到 {f.tag_id} 號（{f.distance * 100:.0f} 公分）"
                     for f in sorted(fixes, key=lambda f: f.distance))


def pose_text(pose: Pose2D) -> str:
    return f"x {pose.x * 100:.0f} 公分、y {pose.y * 100:.0f} 公分、車頭朝 {math.degrees(pose.yaw):+.0f} 度"


def reject_text(reason: str) -> str:
    return next((text for key, text in REJECT_TEXT.items() if reason.startswith(key)), reason)


def wait_frames(grabbers: dict, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not all(g.latest()[0] is not None for g in grabbers.values()):
        time.sleep(0.1)


# ---- sensing ------------------------------------------------------------


def acquire(args, rig: Rig, raw: Path) -> list[tuple[int, np.ndarray, float | None]]:
    """Pan across `args.pans`; (pan, gimbal frame, ultrasonic cm) per angle, also saved under raw/."""
    shots = []
    for pan in args.pans:
        rig.pan(pan, settle_s=args.settle_s)
        frame = rig.frame()
        sonic_cm = rig.sonic_cm()
        cv2.imwrite(str(raw / f"gimbal_{pan:+03d}.jpg"), frame)
        shots.append((pan, frame, sonic_cm))
        print(f"  雲台 {pan:+3d} 度：超聲波 {'沒有讀數' if sonic_cm is None else f'{sonic_cm:.0f} 公分'}", flush=True)
    return shots


def column_angles(cam: GimbalCamera) -> np.ndarray:
    """Horizontal angle (deg) of each scanned column off the gimbal camera's axis."""
    pixels = np.column_stack([cam.columns, np.full(len(cam.columns), cam.k[1, 2])]).astype(np.float64)
    norm = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), cam.k, cam.d).reshape(-1, 2)
    return np.degrees(np.arctan(norm[:, 0]))


def process(args, loc: ArucoLocalizer, cam: GimbalCamera, pan: int, frame: np.ndarray, sonic_cm,
            offset: float = 0.0, pitch: float | None = None) -> dict:
    """One gimbal view in the car frame: foam foot points, gaps, tags. Nothing here depends on the pose.

    `pan` is the commanded angle; the geometry uses `pan + offset`, the angle the gimbal really turned to,
    and `pitch` when the arena walls have measured this view's tilt (see `wall_pitch`).
    Foot points come only from the middle of the frame, where the fisheye model holds best; the
    15 deg pan steps overlap enough to cover the rest.
    """
    true_pan = pan + offset
    geometry = cam.geometry if pitch is None else dict(cam.geometry, pitch_deg=pitch)
    cols, rows = cam.feet(frame)
    pts = np.full((len(cols), 2), np.nan)
    seen = (rows >= 0) & (rows < HEIGHT)
    if seen.any():
        pts[seen] = cam.project(np.column_stack([cols[seen], rows[seen]]), true_pan, geometry)
    m = cam.mount(true_pan, geometry)
    pts[~(np.hypot(pts[:, 0] - m["x_m"], pts[:, 1] - m["y_m"]) < args.max_range_m)] = np.nan
    too_close = rows == HEIGHT
    usable = np.abs(column_angles(cam)) <= args.view_deg
    pts[~usable] = np.nan
    # Foam that fills the bottom of the frame stands closer than the nearest floor the camera sees, so
    # its foot is out of view. The bottom edge's floor point marks "something at least this close".
    blind = too_close & usable
    close = np.zeros((0, 2))
    if blind.any():
        close = cam.project(np.column_stack([cols[blind], np.full(int(blind.sum()), HEIGHT - 1)]), true_pan, geometry)
        close = close[~np.isnan(close[:, 0])]
    gaps = cam.find_gaps(cols, rows, pts, too_close, true_pan, cam.floor_mask(frame))
    view = f"gimbal{pan:+d}"
    loc.add_view(view, cam.mount(true_pan, geometry), "gimbal")
    fixes = loc.observe(view, frame)[0]
    return {
        "pan": pan, "true_pan": true_pan, "pitch": pitch, "frame": frame, "cols": cols, "rows": rows,
        "pts": pts, "gaps": gaps, "sonic_cm": sonic_cm, "fixes": fixes, "too_close": int(too_close.sum()),
        "usable": usable, "too_close_mask": too_close, "close": close,
    }


def wall_pitch(cam: GimbalCamera, rec: dict, pose: Pose2D, arena_m, max_fix_deg: float = 3.0) -> float | None:
    """Tilt of this view that puts its long wall foot line on the arena wall, or None.

    The tilt servo settles a degree or two off after every pan, which moves a foot 60 cm away by
    over 10 cm. The walls are at known places, so a long foot line close to one is a ruler. Only
    wall feet 35 cm or more away are used: nearer, the tilt hardly matters.
    """
    rows, pts = rec["rows"], rec["pts"]
    m = cam.mount(rec["true_pan"])
    # Up close the tilt barely moves a foot, so a near wall measures the pose error, not the tilt.
    ok = ~np.isnan(pts[:, 0]) & (np.hypot(pts[:, 0] - m["x_m"], pts[:, 1] - m["y_m"]) >= 0.35)
    if ok.sum() < 12:
        return None
    w, h = arena_m
    segment = foot_segments(rec["cols"][ok], rows[ok])
    world = to_world(pts[ok], pose.x, pose.y, pose.yaw)
    inner = np.minimum.reduce([world[:, 0], w - world[:, 0], world[:, 1], h - world[:, 1]])
    wall = np.zeros(len(world), bool)
    for label in np.unique(segment):
        members = segment == label
        # A wall line: long, and lying near a wall even before the correction.
        if members.sum() >= 12 and np.median(inner[members]) < 0.20:
            wall |= members
    if wall.sum() < 12:
        return None
    pixels = np.column_stack([rec["cols"][ok][wall], rows[ok][wall]]).astype(np.float64)
    base = float(cam.geometry["pitch_deg"])

    def misfit(pitch: float) -> float:
        p = cam.project(pixels, rec["true_pan"], dict(cam.geometry, pitch_deg=pitch))
        p = p[~np.isnan(p[:, 0])]
        if len(p) < 0.8 * len(pixels):
            return math.inf
        q = to_world(p, pose.x, pose.y, pose.yaw)
        gap = np.minimum.reduce([q[:, 0], w - q[:, 0], q[:, 1], h - q[:, 1]])
        return float(np.median(np.abs(gap)))

    candidates = np.arange(base - max_fix_deg, base + max_fix_deg + 1e-9, 0.1)
    errors = [misfit(p) for p in candidates]
    best = int(np.argmin(errors))
    return float(candidates[best]) if math.isfinite(errors[best]) else None


def pan_offset(records: list[dict], ref: Pose2D) -> float | None:
    """How far the gimbal really turned past the command in this sweep, from the tags it saw.

    The fixed front/rear cameras give the car heading; a gimbal view's tag gives the car heading
    plus that view's pan error. The servo does not return to the same angle every time, so this is
    measured again on every sweep. Views that disagree with the rest by over 10 deg (a tag at the
    frame edge read wrong) are dropped.
    """
    errors = [math.degrees(wrap_rad(f.yaw - ref.yaw)) for rec in records for f in rec["fixes"]
              if f.distance <= NEAR_TAG_M]
    if len(errors) < 2:
        return None
    middle = float(np.median(errors))
    kept = [e for e in errors if abs(e - middle) <= 10.0]
    return float(np.median(kept)) if len(kept) >= 2 else None


def load_raw(raw: Path = RAW) -> tuple[dict, list[tuple[int, np.ndarray, float | None]]]:
    """The last run's frames and readings, for --replay."""
    info = json.loads((raw / "sweep.json").read_text(encoding="utf-8"))
    shots = [(int(s["pan"]), cv2.imread(str(raw / f"gimbal_{int(s['pan']):+03d}.jpg")), s["sonic_cm"])
             for s in info["shots"]]
    return info, shots


# ---- mapping ------------------------------------------------------------


def ray_to_wall(ox: float, oy: float, angle: float, w: float, h: float) -> float:
    dx, dy = math.cos(angle), math.sin(angle)
    hits = []
    if dx > 1e-6:
        hits.append((w - ox) / dx)
    if dx < -1e-6:
        hits.append(-ox / dx)
    if dy > 1e-6:
        hits.append((h - oy) / dy)
    if dy < -1e-6:
        hits.append(-oy / dy)
    return max(0.0, min(hits)) if hits else 0.0


class Grid:
    """World grid of free floor that was seen. Obstacles are blocks, stamped in `labels`. The walls are
    known from arena_map.json, so measured wall feet are not drawn; they only calibrate the tilt."""

    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.nx, self.ny = int(round(w / RES)), int(round(h / RES))
        self.free = np.zeros((self.ny, self.nx), np.int16)

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        col, row = int(x / RES), int(y / RES)
        if 0 <= col < self.nx and 0 <= row < self.ny:
            return row, col
        return None

    def add(self, layer: np.ndarray, x: float, y: float) -> None:
        c = self.cell(x, y)
        if c is not None:
            layer[c] += 1

    def clear_line(self, ox: float, oy: float, ex: float, ey: float, stop_short: float = 0.03) -> None:
        length = math.hypot(ex - ox, ey - oy) - stop_short
        if length <= 0:
            return
        dx, dy = (ex - ox) / (length + stop_short), (ey - oy) / (length + stop_short)
        for i in range(int(length / RES)):
            self.add(self.free, ox + dx * i * RES, oy + dy * i * RES)

    def labels(self, blocks: list[dict], sonic_only: list[tuple[float, float]]) -> np.ndarray:
        out = np.full((self.ny, self.nx), UNKNOWN, np.uint8)
        out[self.free > 0] = FREE
        for block in blocks:
            mask = np.zeros((self.ny, self.nx), np.uint8)
            if block["shaped"]:
                poly = np.round(np.asarray(block["corners"]) / RES * 16).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1, shift=4)
            for x, y in block["points"]:
                c = self.cell(x, y)
                if c is not None:
                    mask[c] = 1
            out[mask > 0] = BLUE if block["color"] == "blue" else GREY
        for x, y in sonic_only:
            c = self.cell(x, y)
            if c is not None:
                out[c] = SONIC
        return out


def sonic_origin(sonic: dict, pan: int) -> tuple[float, float]:
    """Ultrasonic face in the car frame: on an arm in front of the pan shaft, turning with it."""
    a = math.radians(pan)
    return sonic["pivot_x_m"] + sonic["arm_m"] * math.cos(a), sonic["arm_m"] * math.sin(a)


def body_gap(origin: tuple[float, float], pan: int, distance: float, body_m) -> float:
    """Distance from the body edge (where the beam leaves the car outline) to what the beam hit."""
    a = math.radians(pan)
    dx, dy = math.cos(a), math.sin(a)
    exits = []
    if abs(dx) > 1e-6:
        exits.append((math.copysign(body_m[0] / 2.0, dx) - origin[0]) / dx)
    if abs(dy) > 1e-6:
        exits.append((math.copysign(body_m[1] / 2.0, dy) - origin[1]) / dy)
    return distance - min(exits)


def foot_segments(cols: np.ndarray, rows: np.ndarray, jump_px: int = 10) -> np.ndarray:
    """Label each foot point (in column order) by the unbroken image line it belongs to.

    The foot row jumps where a nearer object hides a farther one, and the line breaks where no foot
    was found; between breaks it is one surface.
    """
    step = float(np.min(np.diff(cols))) if len(cols) > 1 else 1.0
    labels = np.zeros(len(rows), np.int32)
    for i in range(1, len(rows)):
        broken = abs(int(rows[i]) - int(rows[i - 1])) > jump_px or cols[i] - cols[i - 1] > 1.5 * step
        labels[i] = labels[i - 1] + broken
    return labels


def wall_feet(world: np.ndarray, segment: np.ndarray, w: float, h: float, band: float,
              half_window: int = 3, max_angle_deg: float = 35.0) -> np.ndarray:
    """Which foot points are the arena wall: near a wall, and running along it.

    A wall's foot is a line parallel to that wall. A block standing against the wall shows a face
    running out from it, so a foot line at right angles to the nearby wall is a block, however close.
    """
    near = {"x": (world[:, 0] < band) | (world[:, 0] > w - band),   # left/right walls run along y
            "y": (world[:, 1] < band) | (world[:, 1] > h - band)}   # bottom/top walls run along x
    wall = np.zeros(len(world), bool)
    cos_max = math.cos(math.radians(max_angle_deg))
    for i in np.flatnonzero(near["x"] | near["y"]):
        mates = np.flatnonzero(segment == segment[i])
        mates = mates[np.abs(mates - i) <= half_window]
        span = world[mates.max()] - world[mates.min()]
        if len(mates) < 3 or math.hypot(*span) < 0.01:
            wall[i] = True  # too little line to tell; near the wall it is most likely the wall
            continue
        ux, uy = span / math.hypot(*span)
        wall[i] = bool((near["x"][i] and abs(uy) >= cos_max) or (near["y"][i] and abs(ux) >= cos_max))
    return wall


def place_views(args, cam: GimbalCamera, sonic: dict, records: list[dict], pose: Pose2D, grid: Grid) -> None:
    """Put every sweep record in the world: paint free floor and wall, keep obstacle foot points and
    the ultrasonic return. Adds the world-side fields to each record."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)

    def place(p):
        return pose.x + c * p[0] - s * p[1], pose.y + s * p[0] + c * p[1]

    for rec in records:
        pan, pts = rec["true_pan"], rec["pts"]
        m = cam.mount(pan)
        ox, oy = place((m["x_m"], m["y_m"]))
        beam = pose.yaw + math.radians(pan)

        ok = ~np.isnan(pts[:, 0])
        world = to_world(pts[ok], pose.x, pose.y, pose.yaw) if ok.any() else np.zeros((0, 2))
        reach = np.hypot(pts[ok, 0] - m["x_m"], pts[ok, 1] - m["y_m"])
        # One unbroken foot line in one image is one surface. A line that runs into the edge of the
        # usable part of the frame, or into foam too close to see its foot, was not seen to its end.
        index = np.flatnonzero(ok)
        segment = foot_segments(rec["cols"][ok], rec["rows"][ok])
        near_wall = wall_feet(world, segment, grid.w, grid.h, args.wall_band_m)
        edge = {int(np.flatnonzero(rec["usable"]).min()), int(np.flatnonzero(rec["usable"]).max())}
        cut = {}
        for label in np.unique(segment):
            ends = index[segment == label][[0, -1]]
            beside = [i for i in (ends[0] - 1, ends[1] + 1) if 0 <= i < len(rec["cols"])]
            cut[int(label)] = bool(edge & set(ends.tolist())) or any(rec["too_close_mask"][i] for i in beside)
        obstacle, obstacle_seg, obstacle_cut, obstacle_px = [], [], [], []
        foot_px = np.column_stack([rec["cols"][ok], rec["rows"][ok]])
        for (wx, wy), wall, label, pixel in zip(world, near_wall, segment, foot_px):
            grid.clear_line(ox, oy, wx, wy)
            if not wall:
                obstacle.append((wx, wy))
                obstacle_seg.append(int(label))
                obstacle_cut.append(cut[int(label)])
                obstacle_px.append(pixel)

        kind, gap = "none", None
        sx, sy = place(sonic_origin(sonic, pan))
        if rec["sonic_cm"] is not None:
            d = rec["sonic_cm"] / 100.0
            gap = body_gap(sonic_origin(sonic, pan), pan, d, args.body_m)
            wall = ray_to_wall(sx, sy, beam, grid.w, grid.h)
            if d > args.sonic_max_m:
                # Far returns are where slanted faces bounce the beam away; only near ones are used.
                kind = "far"
            elif abs(d - wall) <= args.wall_tol_m:
                kind = "wall"
            elif d < wall:
                kind = "obstacle"
            else:
                kind = "long"  # beyond the wall: a glancing echo, not trusted
            if kind != "far":
                grid.clear_line(sx, sy, sx + d * math.cos(beam), sy + d * math.sin(beam))
        rec.update(origin=(sx, sy), beam=beam, sonic_kind=kind, body_gap=gap,
                   obstacle_world=np.asarray(obstacle).reshape(-1, 2), obstacle_seg=np.asarray(obstacle_seg, int),
                   obstacle_cut=np.asarray(obstacle_cut, bool), obstacle_px=np.asarray(obstacle_px).reshape(-1, 2),
                   n_obst=len(obstacle),
                   n_wall=len(world) - len(obstacle))


# ---- blocks -------------------------------------------------------------


def fit_block(points: np.ndarray, size, car_xy) -> dict | None:
    """Known-size rectangle whose outline runs through the visible foot points.

    The camera sees only the faces toward the car, so the rectangle starts on the far side of
    them. Both ways of laying the long side are tried; the better fit wins. When only one face is
    seen, both fit as well, and the face's length says which side it is: 10 cm is the short side.
    `one_face` reports that case, so the caller can refuse it when the face was too far to measure.
    """
    from scipy.optimize import least_squares

    (_cx, _cy), (w, h), angle = cv2.minAreaRect(points.astype(np.float32))
    if max(w, h) > math.hypot(*size) + 0.05:
        return None  # wider than one block: two blocks touching, or foam against a wall
    centroid = points.mean(axis=0)
    away = centroid - np.asarray(car_xy)
    away /= max(float(np.hypot(*away)), 1e-6)
    seen = max(w, h)
    fits = []
    for yaw in (math.radians(angle), math.radians(angle) + math.pi / 2):
        start = [*(centroid + away * size[1] / 2.0), yaw]
        # Each way of laying the block may only turn a little, or both would settle on the same one.
        turn = math.radians(20.0)
        res = least_squares(lambda p: rect_distance(points, p[0], p[1], p[2], size), start,
                            bounds=([-np.inf, -np.inf, yaw - turn], [np.inf, np.inf, yaw + turn]),
                            loss="soft_l1", f_scale=0.01)
        rms = float(np.sqrt(np.mean(res.fun ** 2)))
        along = float(np.ptp(points @ np.array([math.cos(res.x[2]), math.sin(res.x[2])])))
        # True when the visible line lies along the long side, which fits a line longer than halfway.
        fits.append((rms, (along >= 0.8 * seen) == (seen > sum(size) / 2.0), res.x))
    fits.sort(key=lambda f: f[0])
    one_face = fits[1][0] - fits[0][0] < 0.005
    if one_face and fits[1][1] and not fits[0][1]:
        fits.reverse()
    rms, _, (cx, cy, yaw) = fits[0]
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return {"center": (float(cx), float(cy)), "yaw": float(yaw), "rms": rms, "one_face": one_face}


def block_color(frame: np.ndarray, feet: np.ndarray) -> str:
    """'blue' or 'grey' from the face just above the block's foot line in one frame."""
    hsv = cv2.cvtColor(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    samples = [hsv[r, int(c)] for c, row in feet for r in range(max(int(row) - 25, 0), max(int(row) - 5, 0), 4)]
    if not samples:
        return "grey"
    hue, sat = np.median(np.asarray(samples)[:, 0]), np.median(np.asarray(samples)[:, 1])
    return "blue" if sat > 50 and 85 <= hue <= 130 else "grey"


def find_blocks(records: list[dict], size, car_xy, arena_m, agree_m: float, beam_deg: float) -> tuple[list[dict], list]:
    """Foam blocks from the camera foot points of every view; the ultrasonic only confirms them.

    A fitted block that pokes more than 3 cm through an arena wall is not trusted; its raw points stay.
    Returns (blocks, ultrasonic returns that no block explains).
    """
    from scipy.spatial import cKDTree

    used = [rec for rec in records if len(rec["obstacle_world"])]
    blocks = []
    if used:
        pts = np.vstack([rec["obstacle_world"] for rec in used])
        view = np.concatenate([np.full(len(rec["obstacle_world"]), k) for k, rec in enumerate(used)])
        cut = np.concatenate([rec["obstacle_cut"] for rec in used])
        pixel = np.vstack([rec["obstacle_px"] for rec in used])
        parent = list(range(len(pts)))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in cKDTree(pts).query_pairs(0.04):
            parent[root(i)] = root(j)
        groups: dict[int, list[int]] = {}
        for i in range(len(pts)):
            groups.setdefault(root(i), []).append(i)
        for members in groups.values():
            if len(members) < 6:
                continue
            group = pts[members]
            # Each view sits at its own few degrees of servo play, so several views of one face land as
            # parallel lines: the block is fitted to the view that sees it best. Within that view its
            # foot line may be broken where the nearest corner dips below the frame; the pieces are still
            # one block (wall feet were already set aside by their direction).
            members = np.asarray(members)
            best = np.bincount(view[members]).argmax()
            in_view = members[view[members] == best]
            single = pts[in_view]
            # A line cut off by the frame edge was not seen to its end, unless it already spans a whole
            # long side: then its length and direction are known and only the far side is filled in.
            long_side = len(single) >= 2 and float(np.ptp(single, axis=0).max()) >= 0.9 * size[0]
            seen_whole = not cut[in_view].any() or long_side
            why = "cut" if not seen_whole else "few" if len(single) < 6 else None
            fit = fit_block(single, size, car_xy) if why is None else None
            if why is None and fit is None:
                why = "wide"
            # With one face seen, its length decides which side it is; far off that length is a few cm
            # unsure, so the block keeps only the points it was seen by.
            if fit is not None and fit["one_face"] and \
                    np.hypot(*(single - np.asarray(car_xy)).T).min() > ONE_FACE_REACH_M:
                fit, why = None, "far_one_face"
            corners = None if fit is None else rect_corners(*fit["center"], fit["yaw"], size)
            if corners is not None and (corners.min() < -0.03 or (corners > np.asarray(arena_m) + 0.03).any()):
                fit, why = None, "through_wall"
            # A real block's foot is a straight line 10 or 20 cm long. Scattered dots (a shadow, a mark on
            # the floor) are "weak": kept, but the map does not plan around them until something confirms.
            (_c, (span_a, span_b), _a) = cv2.minAreaRect(single.astype(np.float32)) if len(single) >= 3 \
                else ((0, 0), (0.0, 0.0), 0)
            weak = len(single) < 8 or max(span_a, span_b) < SOLID_LINE_M
            block = {"points": group, "shaped": fit is not None, "confirmed": False, "why": why, "weak": weak,
                     "color": block_color(used[best]["frame"], pixel[in_view])}
            if fit is not None:
                block.update(fit, corners=corners)
            blocks.append(block)

    sonic_only = []
    for rec in records:
        if rec["sonic_kind"] != "obstacle":
            continue
        ox, oy = rec["origin"]
        d = rec["sonic_cm"] / 100.0
        half = math.radians(beam_deg)
        arc = np.array([(ox + d * math.cos(a), oy + d * math.sin(a))
                        for a in np.linspace(rec["beam"] - half, rec["beam"] + half, 9)])
        hit = False
        for block in blocks:
            if block["shaped"]:
                near = rect_distance(arc, *block["center"], block["yaw"], size).min() <= agree_m
            else:
                near = np.hypot(arc[:, None, 0] - block["points"][None, :, 0],
                                arc[:, None, 1] - block["points"][None, :, 1]).min() <= agree_m
            if near:
                block["confirmed"] = hit = True
        if not hit:
            sonic_only.append((ox + d * math.cos(rec["beam"]), oy + d * math.sin(rec["beam"])))
    return blocks, sonic_only


def body_clearance(points: np.ndarray, pose: Pose2D, body_m) -> float:
    """Shortest distance from the car outline to any of `points` (world)."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    d = points - (pose.x, pose.y)
    local = np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])
    out = np.maximum(np.abs(local) - (body_m[0] / 2.0, body_m[1] / 2.0), 0)
    return float(np.hypot(out[:, 0], out[:, 1]).min())


def outline(block: dict, step: float = 0.01) -> np.ndarray:
    """Points along a block's rectangle (or its raw foot points when it has no shape)."""
    if not block["shaped"]:
        return block["points"]
    corners = block["corners"]
    edges = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        n = max(2, int(np.hypot(*(b - a)) / step) + 1)
        edges.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(edges)


def headings(labels: np.ndarray, grid: Grid, pose: Pose2D, half_width: float,
             max_m: float = 0.9) -> list[tuple[int, float, float]]:
    """(bearing deg, clear m, seen share) for a body-wide corridor straight out of the car centre."""
    blocked = np.isin(labels, (BLUE, GREY, SONIC))
    seen = labels != UNKNOWN
    out = []
    for bearing in range(-90, 91, 5):
        a = pose.yaw + math.radians(bearing)
        dx, dy, nx, ny = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
        clear, looked, total = 0.0, 0, 0
        d = 0.0
        hit = False
        while d <= max_m and not hit:
            for k in np.linspace(-half_width, half_width, max(3, int(2 * half_width / RES) + 1)):
                x, y = pose.x + dx * d + nx * k, pose.y + dy * d + ny * k
                cell = grid.cell(x, y)
                if cell is None:
                    hit = True
                    break
                total += 1
                looked += int(seen[cell])
                if blocked[cell]:
                    hit = True
                    break
            if not hit:
                clear = d
            d += RES
        out.append((bearing, clear, looked / max(total, 1)))
    return out


def openings(rows: list[tuple[int, float, float]], need_m: float) -> list[tuple[int, int, float]]:
    """Runs of consecutive bearings with at least `need_m` clear: (from, to, smallest clear)."""
    runs, current = [], []
    for bearing, clear, _seen in rows:
        if clear >= need_m:
            current.append((bearing, clear))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [(r[0][0], r[-1][0], min(c for _b, c in r)) for r in runs]


# ---- drawing ------------------------------------------------------------


def draw_world(arena: dict, car: dict, grid: Grid, labels: np.ndarray, pose: Pose2D, records: list[dict],
               blocks: list[dict], front_world: np.ndarray, opens, size_px: int = 720,
               route=None, guide=None, trail=None) -> np.ndarray:
    margin = 30
    scale = (size_px - 2 * margin) / max(grid.w, grid.h)
    width, height = int(grid.w * scale) + 2 * margin, int(grid.h * scale) + 2 * margin + 70
    img = np.full((height, width, 3), 225, np.uint8)
    floor_h = int(grid.h * scale) + 2 * margin

    def px(x, y):
        return int(round(margin + x * scale)), int(round(floor_h - margin - y * scale))

    for row in range(grid.ny):
        for col in range(grid.nx):
            value = int(labels[row, col])
            if value == UNKNOWN:
                continue
            a = px(col * RES, (row + 1) * RES)
            b = px((col + 1) * RES, row * RES)
            cv2.rectangle(img, a, b, COLORS[value], -1)
    cv2.rectangle(img, px(0, grid.h), px(grid.w, 0), WALL_COLOR, 4)
    for tag in arena.get("tags", []):
        tx, ty = tag["center_m"][:2]
        cv2.putText(img, f"T{tag['id']}", (px(tx, ty)[0] - 10, px(tx, ty)[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    for rec in records:
        if rec["sonic_cm"] is None:
            continue
        ox, oy = rec["origin"]
        d = rec["sonic_cm"] / 100.0
        color = {"wall": (160, 160, 160), "obstacle": (200, 60, 160)}.get(rec["sonic_kind"], (210, 210, 120))
        cv2.line(img, px(ox, oy), px(ox + d * math.cos(rec["beam"]), oy + d * math.sin(rec["beam"])),
                 color, 1, cv2.LINE_AA)
    for x, y in front_world:
        cv2.drawMarker(img, px(x, y), (0, 140, 0), cv2.MARKER_CROSS, 5, 1)
    for line, color, width in ((route, (200, 160, 0), 1), (guide, (40, 140, 40), 2), (trail, (40, 40, 200), 1)):
        if line and len(line) >= 2:
            pts = [px(p[0], p[1]) for p in line]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, color, width, cv2.LINE_AA)
    half_l, half_w = (float(v) / 2.0 for v in car.get("body_m", [0.29, 0.18]))
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    corners = [px(pose.x + c * lx - s * ly, pose.y + s * lx + c * ly)
               for lx, ly in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]
    cv2.polylines(img, [np.array(corners, np.int32)], True, (40, 40, 160), 2)
    cv2.arrowedLine(img, px(pose.x, pose.y), px(pose.x + c * (half_l + 0.05), pose.y + s * (half_l + 0.05)),
                    (40, 40, 160), 2, tipLength=0.3)
    for lo, hi, clear in opens:
        mid = pose.yaw + math.radians((lo + hi) / 2.0)
        reach = min(clear, 0.5)
        cv2.arrowedLine(img, px(pose.x, pose.y), px(pose.x + reach * math.cos(mid), pose.y + reach * math.sin(mid)),
                        (0, 170, 0), 2, tipLength=0.15)
    for block in blocks:
        k = block["id"]
        for x, y in block["points"]:
            cv2.circle(img, px(x, y), 1, (0, 0, 0), -1)
        if block["shaped"]:
            cv2.polylines(img, [np.array([px(x, y) for x, y in block["corners"]], np.int32)], True, (0, 0, 0), 1)
        x, y = block["center"] if block["shaped"] else block["points"].mean(axis=0)
        u, v = px(x, y)
        cv2.rectangle(img, (u - 9, v - 11), (u + 9, v + 7), (255, 255, 255), -1)
        cv2.putText(img, str(k), (u - 6, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    legend = [("blue block", COLORS[BLUE]), ("grey block", COLORS[GREY]), ("sonic / too close", COLORS[SONIC]),
              ("wall", WALL_COLOR), ("front CSI (compare)", (0, 140, 0)), ("open heading", (0, 170, 0))]
    x0, y0 = 10, floor_h + 18
    for k, (text, color) in enumerate(legend):
        x, y = x0 + (k % 3) * 235, y0 + (k // 3) * 24
        cv2.rectangle(img, (x, y - 10), (x + 14, y + 4), color, -1)
        cv2.putText(img, text, (x + 20, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, f"pose {pose.text()}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return img


BOX_COLORS = {"blue": (255, 140, 0), "grey": (210, 210, 210)}  # drawn bright enough to show on the foam


def draw_blocks(img: np.ndarray, cam: GimbalCamera, true_pan: float, pose: Pose2D, blocks: list[dict],
                height_m: float) -> None:
    """Draw every fitted block as a box in one gimbal frame, so a wrong fit shows against the real foam."""
    cam_T_car = np.linalg.inv(car_from_camera(cam.mount(true_pan)))
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for block in blocks:
        if not block["shaped"]:
            continue
        d = np.asarray(block["corners"]) - (pose.x, pose.y)
        local = np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])
        box = np.array([[x, y, z, 1.0] for z in (0.0, height_m) for x, y in local])
        in_cam = (cam_T_car @ box.T).T[:, :3]
        if (in_cam[:, 2] < 0.03).any():
            continue
        px, _ = cv2.fisheye.projectPoints(in_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.k, cam.d)
        px = px.reshape(-1, 2).astype(int)
        if not ((px[:, 0] > -img.shape[1]) & (px[:, 0] < 2 * img.shape[1])).any():
            continue
        color = BOX_COLORS[block["color"]]
        for a, b in edges:
            cv2.line(img, tuple(px[a]), tuple(px[b]), color, 2)
        cv2.putText(img, f"#{block['id']}", tuple(px[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def contact_sheet(cam: GimbalCamera, records: list[dict], need_m: float, pose: Pose2D, blocks: list[dict],
                  height_m: float, cols: int = 4) -> np.ndarray:
    tiles = []
    for rec in records:
        tile = draw_frame(rec["frame"], cam, rec["cols"], rec["rows"], rec["pts"], rec["gaps"], rec["pan"], need_m,
                          tint=False)
        draw_blocks(tile, cam, rec["true_pan"], pose, blocks, height_m)
        sonic = "-" if rec["sonic_cm"] is None else f"{rec['sonic_cm']:.0f}cm {rec.get('sonic_kind', '')}"
        cv2.putText(tile, f"sonic {sonic}", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if rec["fixes"]:
            ids = ",".join(str(f.tag_id) for f in rec["fixes"])
            cv2.putText(tile, f"tag {ids}", (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(cv2.resize(tile, (320, 240)))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


# ---- shared pipeline ------------------------------------------------------


class Perception:
    """One scan turned into a map: used by this script and by route_follow.py --perceive.

    `scene` places the car and finds the blocks in one sweep; `merge` folds them into the
    remembered BlockMap; `obstacles` gives the planner the points to keep clear of.
    """

    def __init__(self, args, arena: dict, car: dict, loc: ArucoLocalizer | None = None):
        self.args = args
        self.arena = arena
        self.loc = loc or ArucoLocalizer(arena, car)
        self.cam = GimbalCamera(car)
        self.size_m = tuple(float(v) for v in arena["arena_size_m"])
        self.body_m = [float(v) for v in car.get("body_m", [0.29, 0.18])]
        args.body_m = self.body_m
        self.sonic = dict(DEFAULT_SONIC, **car.get("sonic", {}))
        self.block_m = [float(v) for v in arena.get("obstacle_block_m", [0.20, 0.10])][:2]
        self.block_h = float(arena.get("obstacle_block_height_m", 0.15))

    def scene(self, shots, fixes_csi: list[TagFix], pose: Pose2D | None = None, say=print) -> SimpleNamespace:
        """Place the car (unless `pose` is given) and find the blocks of one sweep.

        The fixed CSI cameras give the reference heading; it also measures how far the gimbal
        really turned, and every gimbal view is redone at that angle. The walls then measure each
        view's tilt. `pose` is None in the result when no tag could place the car.
        """
        args, loc, cam = self.args, self.loc, self.cam
        w, h = self.size_m
        records = [process(args, loc, cam, pan, frame, sonic) for pan, frame, sonic in shots]
        for rec in records:
            if rec["fixes"]:
                seen = "、".join(f"{f.tag_id} 號（{f.distance * 100:.0f} 公分）"
                                for f in sorted(rec["fixes"], key=lambda f: f.distance))
                say(f"  雲台 {rec['pan']:+d} 度看到標籤：{seen}")
        csi_pose, _ = solve_pose(loc, fixes_csi, self.body_m, w, h)
        ref = pose or csi_pose
        offset = pan_offset(records, ref) if ref is not None else None
        if offset is not None:
            say(f"\n雲台實際角度比指令多 {offset:+.1f} 度（由雲台看到的標籤對照車子朝向算出），已修正")
            records = [process(args, loc, cam, pan, frame, sonic, offset) for pan, frame, sonic in shots]
        why = ""
        if pose is not None:
            say(f"\n使用給定的位置：{pose_text(pose)}")
        elif csi_pose is not None:
            pose = csi_pose
            say(f"\n車子位置：{pose_text(pose)}（車頭/車尾鏡頭，{tag_list(pose.fixes)}，誤差 {pose.rms_px:.1f} 像素）")
        else:
            pose, why = solve_pose(loc, [f for rec in records for f in rec["fixes"]], self.body_m, w, h)
            if pose is not None:
                say(f"\n車子位置：{pose_text(pose)}（誤差 {pose.rms_px:.1f} 像素；車頭/車尾鏡頭沒有可用標籤，改用雲台鏡頭，"
                    f"雲台角度誤差未修正）")
        if pose is None:
            return SimpleNamespace(pose=None, why=why, records=records)
        if pose.fixes and len({f.tag_id for f in pose.fixes}) == 1:
            say("警告：只用到一張標籤，位置可能有錯；地圖上的障礙物位置會跟著偏。")
        for k, rec in enumerate(records):
            pitch = wall_pitch(cam, rec, pose, (w, h))
            if pitch is not None:
                records[k] = process(args, loc, cam, rec["pan"], rec["frame"], rec["sonic_cm"], offset or 0.0, pitch)
        fixed = [f"{r['pan']:+d}:{r['pitch'] - cam.geometry['pitch_deg']:+.1f}" for r in records if r["pitch"] is not None]
        say(f"雲台俯角依牆面修正（度）: {'  '.join(fixed) if fixed else '沒有看到夠長的牆腳'}")
        grid = Grid(w, h)
        place_views(args, cam, self.sonic, records, pose, grid)
        sightings, sonic_only = find_blocks(records, self.block_m, (pose.x, pose.y), (w, h), args.agree_m,
                                            args.beam_deg)
        close = [to_world(rec["close"], pose.x, pose.y, pose.yaw) for rec in records if len(rec["close"])]
        close = np.vstack(close) if close else np.zeros((0, 2))
        if len(close):
            # A wall right beside the car fills the frame bottom too; the wall is already kept clear of.
            inner = np.minimum.reduce([close[:, 0], w - close[:, 0], close[:, 1], h - close[:, 1]])
            close = close[inner > 0.06]
        if len(close):
            say(f"有東西近到腳線超出畫面下緣：{len(close)} 個方向，當成障礙物（離車心最近 "
                f"{float(np.hypot(close[:, 0] - pose.x, close[:, 1] - pose.y).min()) * 100:.0f} 公分）")
        return SimpleNamespace(pose=pose, why=why, offset=offset, records=records, grid=grid,
                               sightings=sightings, sonic_only=sonic_only, close=close)

    def merge(self, scene: SimpleNamespace, bmap: BlockMap, tags: int) -> dict:
        """Fold one scene's blocks into the remembered map."""
        pose, grid = scene.pose, scene.grid
        gimbal_x = float(self.cam.geometry["pivot_x_m"])
        camera_xy = (pose.x + gimbal_x * math.cos(pose.yaw), pose.y + gimbal_x * math.sin(pose.yaw))

        def seen_free(points: np.ndarray) -> float:
            cells = [grid.cell(x, y) for x, y in points]
            cells = [c for c in cells if c is not None]
            return float(np.mean([grid.free[c] > 0 for c in cells])) if cells else 0.0

        return bmap.update(scene.sightings, camera_xy, tags, seen_free)

    def blocks(self, bmap: BlockMap, pose: Pose2D) -> list[dict]:
        """Remembered blocks, nearest to the car body first, each with its `clearance`."""
        blocks = bmap.as_blocks()
        for block in blocks:
            block["clearance"] = body_clearance(outline(block), pose, self.body_m)
        return sorted(blocks, key=lambda b: b["clearance"])

    @staticmethod
    def obstacles(blocks: list[dict], sonic_only, close=()) -> np.ndarray:
        """Floor points the planner must keep the car body away from: remembered blocks, ultrasonic
        returns no block explains, and foam too close for the camera to see its foot."""
        parts = [outline(block) for block in blocks if not block.get("provisional")]
        for extra in (sonic_only, close):
            if len(extra):
                parts.append(np.asarray(extra, dtype=float).reshape(-1, 2))
        return np.vstack(parts) if parts else np.zeros((0, 2))


# ---- main ---------------------------------------------------------------


def main():
    args = parse_args()
    OUT.mkdir(exist_ok=True)
    arena = load_json(MAP_PATH)
    car = load_json(CAR_PATH)
    perception = Perception(args, arena, car)
    loc, cam = perception.loc, perception.cam
    if not cam.geometry.get("calibrated"):
        print("警告：雲台鏡頭高度/俯角尚未校正，先跑 gimbal_gap.py --calibrate；目前用估計值。", flush=True)
    w, h = perception.size_m
    body_m = perception.body_m
    need = body_m[1] + 2.0 * args.margin_m

    # 1. Sense (or replay the last run's raw data).
    if args.replay:
        raw = Path(args.replay)
        info, shots = load_raw(raw)
        frames = {name: cv2.imread(str(raw / f"{name}.jpg")) for name in ("front", "rear")}
        fixes_csi = [f for name, frame in frames.items() if frame is not None for f in loc.observe(name, frame)[0]]
        ir = info["ir"]
        print(f"重播 {raw}（{info['time']}）")
    else:
        raw = HISTORY / time.strftime("%Y%m%d-%H%M%S")
        raw.mkdir(parents=True, exist_ok=True)
        grabbers = {name: CsiGrabber(name, int(car["camera_mounts"][name]["camera_num"]), args.width,
                                     args.height).start() for name in ("front", "rear")}
        rig = None
        sensors = None
        try:
            wait_frames(grabbers)
            print(f"車頭/車尾鏡頭找標籤（{args.localize_s:.0f} 秒）...", flush=True)
            fixes_csi = csi_fixes(loc, grabbers, args.localize_s)
            frames = {name: g.latest()[0] for name, g in grabbers.items()}
            for name, frame in frames.items():
                if frame is not None:
                    cv2.imwrite(str(raw / f"{name}.jpg"), frame)
            for grabber in grabbers.values():
                grabber.close()
            grabbers = {}

            from safe_surround_cruise import SafetySensors

            sensors = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=False, active_high=False))
            rig = Rig(sonic=False, sensors=sensors)
            print("雲台掃描:", flush=True)
            shots = acquire(args, rig, raw)
            ir = sensors.read_ir()
        finally:
            if rig is not None:
                rig.close()
            if sensors is not None:
                sensors.close()
            for grabber in grabbers.values():
                grabber.close()
        info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "ir": {k: int(v) for k, v in ir.items()},
                "shots": [{"pan": pan, "sonic_cm": sonic} for pan, _frame, sonic in shots]}
        (raw / "sweep.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copytree(raw, RAW, dirs_exist_ok=True)
    front_frame = frames.get("front")
    print(f"車頭/車尾標籤: {tag_list(fixes_csi) or '沒有看到'}")
    given = None
    if args.pose:
        x, y, yaw = (float(v) for v in args.pose.split(","))
        given = Pose2D(x, y, math.radians(yaw))
    scene = perception.scene(shots, fixes_csi, given)
    if scene.pose is None:
        raise SystemExit(f"\n定位失敗：{scene.why}。換個位置，或用 --pose x,y,yaw 指定。")
    pose, records, grid, sonic_only = scene.pose, scene.records, scene.grid, scene.sonic_only
    offset = scene.offset
    block_m, block_h = perception.block_m, perception.block_h
    # Fold this scan into the blocks remembered from earlier scans. A replay reads the map but does not
    # save it, so the same frames are never counted twice.
    bmap = BlockMap(block_m) if args.new_map else BlockMap.load(block_m)
    tags = 2 if args.pose else len({f.tag_id for f in pose.fixes})
    counts = perception.merge(scene, bmap, tags)
    if not args.replay:
        bmap.save()
    print(f"這次看到 {len(scene.sightings)} 塊；地圖新增 {counts['new']}、更新 {counts['updated']}、"
          f"移除 {counts['removed']} 塊{'（重播不存檔）' if args.replay else ''}")
    blocks = perception.blocks(bmap, pose)
    labels = grid.labels(blocks, list(sonic_only) + [tuple(p) for p in scene.close])
    front_world = np.zeros((0, 2))
    if front_frame is not None:
        cv2.imwrite(str(OUT / "perceive_front.jpg"), front_frame)
        pts = FloorScanner(loc, car["camera_mounts"]["front"], "front").scan(front_frame)
        if len(pts):
            reach = np.hypot(pts[:, 0], pts[:, 1])
            on_body = (np.abs(pts[:, 0]) < body_m[0] / 2) & (np.abs(pts[:, 1]) < body_m[1] / 2)
            pts = pts[(reach > 0.12) & (reach < args.max_range_m) & ~on_body]
            if len(pts):
                front_world = to_world(pts, pose.x, pose.y, pose.yaw)

    # 4. Report.
    rows = headings(labels, grid, pose, need / 2.0)
    opens = openings(rows, 0.35)

    print(f"\n超聲波（只採用讀數 {args.sonic_max_m * 100:.0f} 公分內的回波）:")
    for r in records:
        if r["body_gap"] is not None:
            print(f"  雲台 {r['pan']:+4d} 度  讀數 {r['sonic_cm']:5.1f} 公分  車身邊緣到物體 {r['body_gap'] * 100:5.1f} 公分  "
                  f"{SONIC_TEXT[r['sonic_kind']]}")
    on = [k for k in ("front_left", "front_right", "wheel_left_front", "wheel_left_rear",
                      "wheel_right_front", "wheel_right_rear") if ir.get(k)]
    print(f"\n紅外線（8 公分內）：{', '.join(on) if on else '都沒觸發'}")
    print(f"\n障礙物（場地座標，原點在左下角；方塊 {block_m[0] * 100:.0f}×{block_m[1] * 100:.0f} 公分）：")
    if not blocks:
        print("  沒有偵測到場地內部的障礙物")
    for block in blocks:
        k = block["id"]
        tag = f"{'藍色' if block['color'] == 'blue' else '灰色'}，{'超聲波確認過' if block['confirmed'] else '只有鏡頭'}，" \
              f"看到 {block['hits']} 次"
        if block["shaped"]:
            cx, cy = block["center"]
            heading = {1: "方向由一個面判斷", 2: "方向由轉角確定"}[block["heading_quality"]]
            print(f"  {k} 號：中心 ({cx * 100:.0f}, {cy * 100:.0f}) 公分 ±{block['sigma'] * 100:.0f}，"
                  f"長邊朝 {math.degrees(block['yaw']):+.0f} 度（{heading}），離車身 {block['clearance'] * 100:.0f} 公分（{tag}）")
        else:
            lo, hi = block["points"].min(axis=0) * 100, block["points"].max(axis=0) * 100
            print(f"  {k} 號：只畫看到的點，{UNSHAPED_TEXT.get(block['why'], '尚未看清楚')}；x {lo[0]:.0f}~{hi[0]:.0f}、"
                  f"y {lo[1]:.0f}~{hi[1]:.0f} 公分，離車身 {block['clearance'] * 100:.0f} 公分（{tag}）")
    if sonic_only:
        text = "、".join(f"({x * 100:.0f},{y * 100:.0f})" for x, y in sonic_only)
        print(f"  只有超聲波回波、鏡頭沒看到方塊：{text} 公分（地圖上紫色）")

    print(f"\n各方向可直走距離（車寬 {body_m[1] * 100:.0f} 公分，兩側各留 {args.margin_m * 100:.0f} 公分；正值是左邊）：")
    for bearing, clear, seen in rows:
        if bearing % 15 == 0:
            bar = "#" * int(clear * 40)
            print(f"  {bearing:+4d} 度  {clear * 100:4.0f} 公分  已看過 {seen * 100:3.0f}%  {bar}")
    if opens:
        text = "、".join(f"{lo:+d}~{hi:+d} 度（至少 {clear * 100:.0f} 公分）" for lo, hi, clear in opens)
        print(f"可通行方向（至少 35 公分）：{text}")
    else:
        print("可通行方向：沒有任何方向能直走 35 公分")
    for rec in records:
        for g in rec["gaps"]:
            lx, ly = to_world(g.left.point[None], pose.x, pose.y, pose.yaw)[0] * 100
            rx, ry = to_world(g.right.point[None], pose.x, pose.y, pose.yaw)[0] * 100
            verdict = "車子過得去" if g.straight_width_m >= need else "太窄"
            print(f"  雲台 {rec['pan']:+d} 度看到縫 {g.width_m * 100:.0f} 公分（{verdict}），"
                  f"兩個角在 ({lx:.0f},{ly:.0f}) 和 ({rx:.0f},{ry:.0f}) 公分")

    cv2.imwrite(str(OUT / "perceive_map.png"),
                draw_world(arena, car, grid, labels, pose, records, blocks, front_world, opens))
    cv2.imwrite(str(OUT / "perceive_gimbal.jpg"), contact_sheet(cam, records, need, pose, blocks, block_h))
    summary = {
        "pose": [round(pose.x, 3), round(pose.y, 3), round(math.degrees(pose.yaw), 1)],
        "gimbal_pan_offset_deg": None if offset is None else round(offset, 1),
        "pose_tags": [{"camera": f.camera, "id": f.tag_id, "distance_m": round(f.distance, 3)} for f in pose.fixes],
        "blocks": [{"id": b["id"], "hits": b["hits"],
                    "center_m": [round(v, 3) for v in b["center"]] if b["shaped"] else None,
                    "yaw_deg": round(math.degrees(b["yaw"]), 1) if b["shaped"] else None,
                    "corners_m": np.round(b["corners"], 3).tolist() if b["shaped"] else None,
                    "points_range_m": [np.round(b["points"].min(axis=0), 3).tolist(),
                                       np.round(b["points"].max(axis=0), 3).tolist()],
                    "clearance_m": round(b["clearance"], 3), "confirmed": b["confirmed"], "color": b["color"],
                    "sigma_m": round(b["sigma"], 3)} for b in blocks],
        "sonic_only_m": [[round(x, 3), round(y, 3)] for x, y in sonic_only],
        "sweep": [{"pan": r["pan"], "sonic_cm": r["sonic_cm"], "sonic_kind": r["sonic_kind"],
                   "body_gap_m": None if r["body_gap"] is None else round(r["body_gap"], 3),
                   "foot_obstacle_points": r["n_obst"], "foot_wall_points": r["n_wall"],
                   "tags": [f.tag_id for f in r["fixes"]]} for r in records],
        "headings": [{"bearing_deg": b, "clear_m": round(c, 3), "seen": round(s, 2)} for b, c, s in rows],
        "openings": [{"from_deg": lo, "to_deg": hi, "clear_m": round(c, 3)} for lo, hi, c in opens],
        "ir": {k: int(v) for k, v in ir.items()},
    }
    (OUT / "perceive_map.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n輸出檔案：{OUT / 'perceive_map.png'}、{OUT / 'perceive_gimbal.jpg'}、{OUT / 'perceive_map.json'}")


if __name__ == "__main__":
    main()
