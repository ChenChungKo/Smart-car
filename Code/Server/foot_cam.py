#!/usr/bin/env python3
"""Where foam meets the floor, seen by the gimbal USB camera: the lap's obstacle sensor.

The ultrasonic misses a block that covers only the edge of the lane (its beam passes beside it,
or the echo is thrown away by a slanted face), and the IR modules only light at 8 cm. The camera
sees the block from half a metre away. In every image column, the lowest pixel that is not light
floor is where something meets the floor; the camera height and tilt (car_nav.json gimbal_camera)
and the fisheye calibration turn that pixel into a floor point in the car frame (x forward, y left).

Each column is judged from the bottom up, because a shadow is only darker floor:
- Blue foam is told by colour (S >= 60; the glossy floor mirrors the blue walls at S 30-35). The
  foot is its lowest blue pixel; a dark band under it is its own shadow.
- A dark run with light floor above it again lies flat on the floor (a shadow or a stain) unless
  it stands up: the camera is z_m high, so a thing h tall reaches h / z_m of the way from its foot
  to the horizon. Under `MIN_RISE` of that way it is skipped.
- Grey / black foam has the floor's colour ratios, so its foot is the hard brightness edge near the
  bottom of the dark run; a shadow fades out instead.
Foam that reaches the bottom of the frame is closer than the nearest floor the camera sees; such
columns give `close` points, the nearest floor seen there: "something at least this near".

Used by route_follow.py (every look before a lane step), lane_scan.py (the stationary check that
draws the result) and tested on saved frames. Opening the camera is `GimbalFeed`; one USB camera at
a time works, three at once drop out.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB_DIR = SERVER / "calibration_patterns" / "captures" / "gimbal"
HARDWARE_PATH = SERVER / "camera_hardware.json"
WIDTH, HEIGHT = 640, 480
FOAM_LO, FOAM_HI = (85, 60, 20), (125, 255, 255)
DARK_RATIO = 0.70      # darker than this share of the floor in the same column is a candidate
MIN_RISE = 0.5         # a candidate must stand this share of the way to the horizon (5 cm at 0.5)
EDGE_PX = 20           # a grey foot is the hardest brightness step in this many rows of the run's bottom
COLUMN_STEP = 4
VIEW_DEG = 40.0       # use columns this far off the camera axis; the fisheye edge is the least accurate
MAX_RANGE_M = 0.50     # beyond this a degree of tilt is over 4 cm and the points get sparse


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def car_from_camera(mount: dict) -> np.ndarray:
    """Car frame x forward, y left, z up. OpenCV camera x right, y down, z forward."""
    pitch = math.radians(float(mount["pitch_deg"]))
    forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.array([-math.sin(pitch), 0.0, -math.cos(pitch)])
    rotation = rot_z(math.radians(float(mount.get("yaw_deg", 0.0)))) @ np.column_stack([right, down, forward])
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = [mount["x_m"], mount.get("y_m", 0.0), mount["z_m"]]
    return out


@dataclass
class FootView:
    """One frame at one gimbal angle: foot points and the conservative 'at least this near' points."""
    pan: float
    cols: np.ndarray
    rows: np.ndarray
    points: np.ndarray      # (N, 2) car-frame floor points of the foot line, NaN where not seen
    close: np.ndarray       # (M, 2) nearest visible floor under columns whose foam reaches the frame bottom
    usable: np.ndarray      # which columns are inside VIEW_DEG
    frame: np.ndarray | None = None


class FootCamera:
    def __init__(self, car: dict):
        self.k = np.load(CALIB_DIR / "camera_0_K.npy").astype(np.float64)  # calibrated at 640 x 480
        self.d = np.load(CALIB_DIR / "camera_0_D.npy").astype(np.float64).reshape(-1, 1)[:4]
        self.geometry = {"pivot_x_m": 0.08, "radius_m": 0.02, "z_m": 0.10, "pitch_deg": 0.0,
                         **car.get("gimbal_camera", {})}
        self.columns = np.arange(COLUMN_STEP // 2, WIDTH, COLUMN_STEP)
        self.angles = self._column_angles()
        self.horizon = self._horizon_rows()

    # ---- geometry ---------------------------------------------------------

    def mount(self, pan_deg: float) -> dict:
        g = self.geometry
        yaw = math.radians(pan_deg)
        return {"x_m": g["pivot_x_m"] + g["radius_m"] * math.cos(yaw), "y_m": g["radius_m"] * math.sin(yaw),
                "z_m": g["z_m"], "yaw_deg": pan_deg, "pitch_deg": g["pitch_deg"]}

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        norm = cv2.fisheye.undistortPoints(pts, self.k, self.d).reshape(-1, 2)
        return np.column_stack([norm, np.ones(len(norm))])

    def project(self, pixels: np.ndarray, pan_deg: float) -> np.ndarray:
        """Floor points (car frame, m) for pixels; NaN where the ray does not reach the floor."""
        if len(pixels) == 0:
            return np.zeros((0, 2))
        t = car_from_camera(self.mount(pan_deg))
        rays = (t[:3, :3] @ self.rays(pixels).T).T
        out = np.full((len(rays), 2), np.nan)
        down = rays[:, 2] < -1e-3
        scale = -t[2, 3] / rays[down, 2]
        out[down] = t[:2, 3] + scale[:, None] * rays[down, :2]
        return out

    def _column_angles(self) -> np.ndarray:
        pixels = np.column_stack([self.columns, np.full(len(self.columns), self.k[1, 2])]).astype(np.float64)
        norm = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), self.k, self.d).reshape(-1, 2)
        return np.degrees(np.arctan(norm[:, 0]))

    def _horizon_rows(self) -> np.ndarray:
        """Per scanned column, the lowest row whose ray no longer reaches the floor."""
        rows = np.arange(HEIGHT)
        out = np.zeros(len(self.columns))
        for i, c in enumerate(self.columns):
            hit = ~np.isnan(self.project(np.column_stack([np.full(HEIGHT, c), rows]), 0.0)[:, 0])
            out[i] = rows[hit].min() if hit.any() else 0
        return out

    # ---- foot lines -------------------------------------------------------

    @staticmethod
    def floor_level(hsv: np.ndarray) -> np.ndarray:
        """Brightness of the floor in each column, from the bottom quarter of the frame (1 x WIDTH)."""
        lower = hsv[int(hsv.shape[0] * 0.75):].astype(np.float32)
        overall = float(np.median(lower[:, :, 2]))
        floor = (lower[:, :, 1] < 30) & (lower[:, :, 2] > 0.6 * overall)
        counts = floor.sum(axis=0)
        level = np.where(counts > 0, (lower[:, :, 2] * floor).sum(axis=0) / np.maximum(counts, 1), overall)
        level = np.maximum(level, 0.5 * overall)
        return cv2.blur(level.reshape(1, -1), (31, 1)).reshape(1, -1)

    def feet(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(columns, foot rows): row -1 = no foam in the column, row HEIGHT = foam to the frame bottom."""
        img = bgr if bgr.shape[1] == WIDTH else cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = cv2.GaussianBlur(hsv[:, :, 2], (1, 5), 0).astype(np.float32)
        blue = cv2.inRange(hsv, np.array(FOAM_LO, np.uint8), np.array(FOAM_HI, np.uint8)) > 0
        blue = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 3), np.uint8)) > 0
        dark = (hsv[:, :, 2] < DARK_RATIO * self.floor_level(hsv)) & ~blue
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 3), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8)) > 0
        rows = np.full(len(self.columns), -1, dtype=np.int32)
        for i, c in enumerate(self.columns):
            rows[i] = self._column_foot(blue[:, c], dark[:, c], value[:, c], self.horizon[i])
        return self.columns, rows

    @staticmethod
    def _column_foot(blue: np.ndarray, dark: np.ndarray, value: np.ndarray, horizon: float) -> int:
        solid = blue | dark
        r = HEIGHT - 1
        while r >= 0:
            if not solid[r]:
                r -= 1
                continue
            top = r
            while top > 0 and solid[top - 1]:
                top -= 1
            run_blue = np.flatnonzero(blue[top:r + 1])
            if len(run_blue):
                foot = top + int(run_blue.max())  # dark below the blue face is its shadow
            elif top == 0 or r - top + 1 >= max(8.0, MIN_RISE * (r - horizon)):
                lo = max(top, r - EDGE_PX)
                step = value[lo + 1:min(r + 3, HEIGHT)] - value[lo:min(r + 3, HEIGHT) - 1]
                foot = lo + int(np.argmax(step)) if len(step) else r
                foot = min(max(foot, top), r)
            else:
                r = top - 1  # flat on the floor with floor above: a shadow or a stain
                continue
            return HEIGHT if foot >= HEIGHT - 3 else foot
        return -1

    def observe(self, bgr: np.ndarray, pan_deg: float, keep_frame: bool = False) -> FootView:
        """Foot points of one frame, taken with the gimbal at `pan_deg` (+ = left)."""
        cols, rows = self.feet(bgr)
        usable = np.abs(self.angles) <= VIEW_DEG
        seen = (rows >= 0) & (rows < HEIGHT) & usable
        pts = np.full((len(cols), 2), np.nan)
        if seen.any():
            pts[seen] = self.project(np.column_stack([cols[seen], rows[seen]]), pan_deg)
        m = self.mount(pan_deg)
        reach = np.hypot(pts[:, 0] - m["x_m"], pts[:, 1] - m["y_m"])
        pts[~(reach < MAX_RANGE_M)] = np.nan
        blind = (rows == HEIGHT) & usable
        close = np.zeros((0, 2))
        if blind.any():
            close = self.project(np.column_stack([cols[blind], np.full(int(blind.sum()), HEIGHT - 1)]), pan_deg)
            close = close[~np.isnan(close[:, 0])]
        return FootView(pan_deg, cols, rows, pts, close, usable, bgr.copy() if keep_frame else None)

    def draw(self, view: FootView) -> np.ndarray:
        """The frame with the foot line (yellow) and the near-blind columns (magenta along the bottom)."""
        img = cv2.resize(view.frame, (WIDTH, HEIGHT)).copy()
        for c, r, ok in zip(view.cols, view.rows, view.usable):
            if 0 <= r < HEIGHT:
                cv2.circle(img, (int(c), int(r)), 2, (0, 255, 255) if ok else (120, 120, 120), -1)
            elif r == HEIGHT:
                cv2.circle(img, (int(c), HEIGHT - 4), 2, (255, 0, 255), -1)
        return img


class CameraScanner:
    """One look: for each gimbal angle, aim (the caller turns the gimbal and waits), grab a fresh frame, find the feet."""

    def __init__(self, cam: FootCamera, feed: "GimbalFeed"):
        self.cam = cam
        self.feed = feed

    def scan(self, aim, pans) -> list[FootView]:
        views = []
        for pan in pans:
            aim(pan)
            frame = self.feed.fresh(time.monotonic())
            views.append(self.cam.observe(frame, pan, keep_frame=True))
        return views

    def close(self) -> None:
        self.feed.close()


class GimbalFeed:
    """Latest frame of the gimbal USB camera, read on a thread (640 x 480 YUYV, the calibration size)."""

    def __init__(self, hardware_path: Path = HARDWARE_PATH):
        self.entry = json.loads(Path(hardware_path).read_text(encoding="utf-8"))["gimbal"]
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gimbal-feed")
        self.error = ""

    def start(self) -> "GimbalFeed":
        self._thread.start()
        return self

    def _run(self):
        from camera_devices import resolve_usb_capture_index

        force_list = False
        while not self._stop.is_set():
            cap = None
            try:
                index, _note = resolve_usb_capture_index(self.entry, force_list=force_list)
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open /dev/video{index}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.error = ""
                misses = 0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        misses += 1
                        if misses >= 4:
                            raise RuntimeError("no frames")
                        time.sleep(0.05)
                        continue
                    misses = 0
                    with self._lock:
                        self._frame, self._stamp = frame, time.monotonic()
            except Exception as exc:  # noqa: BLE001 - reopen after any camera hiccup
                self.error = str(exc)
                force_list = True
                self._stop.wait(0.5)
            finally:
                if cap is not None:
                    cap.release()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy()), self._stamp

    def fresh(self, after: float, timeout: float = 3.0) -> np.ndarray:
        """A frame captured after the monotonic time `after` (the gimbal has stopped moving by then)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, stamp = self.latest()
            if frame is not None and stamp > after:
                return frame
            time.sleep(0.02)
        raise RuntimeError(f"gimbal camera gave no fresh frame ({self.error or 'timeout'})")

    def wait_ready(self, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.latest()[0] is None:
            time.sleep(0.1)
        if self.latest()[0] is None:
            raise RuntimeError(f"gimbal camera not ready ({self.error})")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
