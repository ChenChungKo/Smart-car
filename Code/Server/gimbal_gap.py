#!/usr/bin/env python3
"""Width of the gap between two foam obstacles, seen by the gimbal (pan/tilt) USB camera.

The gimbal camera shares the pan servo with the ultrasonic. In every image column the lowest
pixel that is not light floor (blue foam, or a black/grey block or wall) is where it meets the floor; the camera height and pitch turn that pixel into
a floor point in the car frame (x forward, y left). Neighbouring columns that land close together
belong to one obstacle; where the nearer obstacle ends, its last floor point is a real corner.
The gap width is the floor distance between the two corners that face each other.

Both corners come from one frame, so pan servo backlash moves them together and barely changes
the width.

Never drives the motors. The pan/tilt PWM is released when the program ends.

    python3 gimbal_gap.py --calibrate      # fit camera height/pitch against the ultrasonic, save to car_nav.json
    python3 gimbal_gap.py                  # measure gaps at pan -60..+60
    python3 gimbal_gap.py --pans 20 30 40  # measure gaps at chosen pan angles (positive = left)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from aruco_localizer import CAR_PATH, MAP_PATH, ArucoLocalizer, UsbGrabber, car_from_camera, load_json

SERVER = Path(__file__).resolve().parent
DEBUG_DIR = SERVER / "gimbal_gap_debug"
CAMERA = "gimbal"
WIDTH, HEIGHT = 640, 480
DEFAULT_GEOMETRY = {
    "pivot_x_m": 0.08,
    "radius_m": 0.02,
    "z_m": 0.10,
    "pitch_deg": 0.0,
    "sonic_ahead_m": 0.0,
    "calibrated": False,
}


@dataclass
class Corner:
    column: int
    point: np.ndarray


@dataclass
class Gap:
    pan_deg: float
    left: Corner
    right: Corner
    width_m: float
    centre: np.ndarray

    @property
    def bearing_deg(self) -> float:
        return math.degrees(math.atan2(self.centre[1], self.centre[0]))

    @property
    def straight_width_m(self) -> float:
        """Width across the line from the car to the gap centre: what a car driving straight through gets."""
        u = self.centre / max(float(np.hypot(*self.centre)), 1e-6)
        d = self.left.point - self.right.point
        return abs(float(d[0] * u[1] - d[1] * u[0]))


class GimbalCamera:
    def __init__(self, car: dict, hsv_lo=(85, 30, 20), hsv_hi=(125, 255, 255), dark_ratio: float = 0.70,
                 column_step: int = 4):
        self.localizer = ArucoLocalizer(load_json(MAP_PATH), car)
        self.k, self.d = self.localizer.kd(CAMERA, WIDTH, HEIGHT)
        # Kept out of camera_mounts: ArucoLocalizer treats every entry there as a fixed camera.
        self.geometry = dict(DEFAULT_GEOMETRY, **car.get("gimbal_camera", {}))
        self.lo, self.hi = np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
        self.dark_ratio = dark_ratio
        self.columns = np.arange(column_step // 2, WIDTH, column_step)

    def mount(self, pan_deg: float, geometry: dict | None = None) -> dict:
        g = self.geometry if geometry is None else geometry
        yaw = math.radians(pan_deg)
        return {
            "x_m": g["pivot_x_m"] + g["radius_m"] * math.cos(yaw),
            "y_m": g["radius_m"] * math.sin(yaw),
            "z_m": g["z_m"],
            "yaw_deg": pan_deg,
            "pitch_deg": g["pitch_deg"],
        }

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        """Anything standing on the light floor: blue foam, or dark (black/grey blocks, grey walls).

        Blue foam is S > 85. Dark is judged against the floor in the same column, because the floor
        itself is dimmer in the fisheye corners and in a dim frame. Grey foam is 0.20-0.38 of the
        floor in shade but up to 0.67 facing the light, while a block's shadow on the floor is
        0.41-0.55, so brightness cannot tell them apart. `dark_ratio` keeps every block: a shadow
        taken for foam only makes a block look bigger, a missed block is driven into.
        """
        img = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        solid = cv2.inRange(hsv, self.lo, self.hi) | np.where(hsv[:, :, 2] < self.dark_ratio * self.floor_level(hsv),
                                                                255, 0).astype(np.uint8)
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((7, 3), np.uint8))
        return cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))

    @staticmethod
    def floor_level(hsv: np.ndarray) -> np.ndarray:
        """Brightness of the floor in each column, from the bottom quarter of the frame (1 x WIDTH)."""
        lower = hsv[int(hsv.shape[0] * 0.75):].astype(np.float32)
        overall = float(np.median(lower[:, :, 2]))
        # Floor is colourless and not much darker than the rest of the floor; foam there is left out.
        floor = (lower[:, :, 1] < 30) & (lower[:, :, 2] > 0.6 * overall)
        counts = floor.sum(axis=0)
        level = np.where(counts > 0, (lower[:, :, 2] * floor).sum(axis=0) / np.maximum(counts, 1), overall)
        level = np.maximum(level, 0.5 * overall)  # a column whose bottom is all foam borrows the frame's floor
        return cv2.blur(level.reshape(1, -1), (31, 1)).reshape(1, -1)

    def feet(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(columns, foot rows) where blue foam meets the floor; row -1 = no foam, row HEIGHT = foot below the frame."""
        blue = self.mask(bgr)[:, self.columns] > 0
        rows = np.full(len(self.columns), -1, dtype=np.int32)
        has = blue.any(axis=0)
        lowest = HEIGHT - 1 - np.argmax(blue[::-1], axis=0)
        rows[has] = lowest[has]
        # Foam that reaches the bottom edge is too close for its foot to be in view.
        rows[has & blue[-3:].any(axis=0)] = HEIGHT
        return self.columns, rows

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Unit-depth rays (x right, y down, z forward) in the camera frame."""
        pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        norm = cv2.fisheye.undistortPoints(pts, self.k, self.d).reshape(-1, 2)
        return np.column_stack([norm, np.ones(len(norm))])

    def project(self, pixels: np.ndarray, pan_deg: float, geometry: dict | None = None) -> np.ndarray:
        """Floor points (car frame, m) for pixels; NaN where the ray does not hit the floor."""
        if len(pixels) == 0:
            return np.zeros((0, 2))
        T = car_from_camera(self.mount(pan_deg, geometry))
        rays = (T[:3, :3] @ self.rays(pixels).T).T
        out = np.full((len(rays), 2), np.nan)
        down = rays[:, 2] < -1e-3
        scale = -T[2, 3] / rays[down, 2]
        out[down] = T[:2, 3] + scale[:, None] * rays[down, :2]
        return out

    def floor_profile(self, bgr: np.ndarray, pan_deg: float, max_range_m: float = 1.5):
        """Per column: floor point of the foam foot (NaN when none or out of range) and whether it is too close."""
        cols, rows = self.feet(bgr)
        pts = np.full((len(cols), 2), np.nan)
        ok = (rows >= 0) & (rows < HEIGHT)
        if ok.any():
            pts[ok] = self.project(np.column_stack([cols[ok], rows[ok]]), pan_deg)
        m = self.mount(pan_deg)
        reach = np.hypot(pts[:, 0] - m["x_m"], pts[:, 1] - m["y_m"])
        pts[~(reach < max_range_m)] = np.nan
        return cols, rows, pts, rows == HEIGHT

    def floor_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Lit grey floor: low saturation, not dark, not foam."""
        img = cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 55) & ~(self.mask(img) > 0)

    @staticmethod
    def floor_between(floor: np.ndarray, cols, rows, i: int, j: int, min_share: float = 0.6) -> bool:
        """Floor is visible just beyond the two corners, across the columns between them."""
        row = min(int(rows[i]), int(rows[j])) - 8
        lo, hi = int(cols[i]) + 3, int(cols[j]) - 3
        if row < 0 or hi <= lo:
            return False
        band = floor[max(row - 3, 0):row + 1, lo:hi]
        return bool(band.size) and float(band.mean()) >= min_share

    def find_gaps(self, cols, rows, pts, too_close, pan_deg: float, floor: np.ndarray | None = None,
                  max_corner_m: float = 0.9, step_m: float = 0.15, row_jump_px: int = 10,
                  pair_depth_m: float = 0.25) -> list[Gap]:
        """Gaps between two occluding corners that face each other with only farther things between.

        The column just outside a corner has to be at least 15 cm farther. A small kink along the
        arena wall is not a second foam corner.
        """

        def reach(i):
            return float(np.hypot(*pts[i]))

        # One obstacle's foot is a continuous line in the image, even where its face recedes at a
        # grazing angle; an occlusion makes the foot row jump.
        segments: list[list[int]] = []
        for i in range(len(cols)):
            if np.isnan(pts[i, 0]):
                continue
            if segments and segments[-1][-1] == i - 1 and abs(int(rows[i]) - int(rows[i - 1])) <= row_jump_px:
                segments[-1].append(i)
                continue
            segments.append([i])
        segments = [s for s in segments if len(s) >= 3]

        def beyond(i, j) -> bool:
            """Column j (next to corner i) shows nothing, or something clearly farther than corner i."""
            return np.isnan(pts[j, 0]) and not too_close[j] or (not np.isnan(pts[j, 0]) and reach(j) > reach(i) + step_m)

        gaps = []
        # Image left = car left: the right end of segment a faces the left end of a later segment b.
        for a, seg_a in enumerate(segments):
            end_a = seg_a[-1]
            if end_a + 1 >= len(cols) or not beyond(end_a, end_a + 1) or reach(end_a) > max_corner_m:
                continue
            for b in range(a + 1, len(segments)):
                start_b = segments[b][0]
                if start_b - 1 < 0 or not beyond(start_b, start_b - 1):
                    between_near = min(reach(i) for i in segments[b])
                    if between_near < reach(end_a) + step_m:
                        break
                    continue
                # The two sides of one opening stand at a similar range; a much farther corner is background
                # seen through the opening, a much nearer one closes it off.
                if reach(start_b) > reach(end_a) + pair_depth_m or reach(start_b) > max_corner_m:
                    continue
                if reach(start_b) < reach(end_a) - pair_depth_m:
                    break
                near = max(reach(end_a), reach(start_b))
                between = [i for s in segments[a + 1:b] for i in s]
                if any(reach(i) < near + step_m for i in between) or too_close[end_a + 1:start_b].any():
                    break
                if floor is not None and not self.floor_between(floor, cols, rows, end_a, start_b):
                    break
                left, right = Corner(int(cols[end_a]), pts[end_a]), Corner(int(cols[start_b]), pts[start_b])
                width = float(np.hypot(*(left.point - right.point)))
                gaps.append(Gap(pan_deg, left, right, width, 0.5 * (left.point + right.point)))
                break
        return gaps


def draw_frame(bgr, cam: GimbalCamera, cols, rows, pts, gaps: list[Gap], pan_deg: float, need_m: float,
               tint: bool = True):
    img = cv2.resize(bgr, (WIDTH, HEIGHT)).copy()
    if tint:
        solid = cam.mask(img) > 0
        img[solid] = (0.6 * img[solid] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
    for c, r, p in zip(cols, rows, pts):
        if 0 <= r < HEIGHT and not np.isnan(p[0]):
            cv2.circle(img, (int(c), int(r)), 2, (0, 255, 255), -1)
    for gap in gaps:
        ok = gap.straight_width_m >= need_m
        colour = (0, 220, 0) if ok else (0, 0, 255)
        ends = []
        for corner in (gap.left, gap.right):
            row = int(rows[list(cols).index(corner.column)])
            ends.append((corner.column, row))
            cv2.circle(img, (corner.column, row), 7, colour, 2)
        cv2.line(img, ends[0], ends[1], colour, 2)
        mid = ((ends[0][0] + ends[1][0]) // 2, min(ends[0][1], ends[1][1]) - 12)
        cv2.putText(img, f"{gap.width_m * 100:.0f}cm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(img, f"gimbal pan {pan_deg:+.0f}deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def draw_top(profiles, gaps: list[Gap], need_m: float, body_m, size_px: int = 600, span_m: float = 1.2):
    img = np.full((size_px, size_px, 3), 255, np.uint8)
    scale = size_px / span_m
    ox, oy = size_px // 2, size_px - 60

    def px(p):
        return int(ox - p[1] * scale), int(oy - p[0] * scale)

    for d in (0.2, 0.4, 0.6, 0.8, 1.0):
        cv2.circle(img, (ox, oy), int(d * scale), (225, 225, 225), 1)
        cv2.putText(img, f"{d * 100:.0f}", (ox + 3, oy - int(d * scale) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    hl, hw = body_m[0] / 2.0, body_m[1] / 2.0
    body = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    cv2.polylines(img, [np.array([px(p) for p in body]).reshape(-1, 1, 2)], True, (90, 90, 90), 2)
    palette = [(200, 120, 0), (0, 150, 200), (150, 0, 200), (0, 160, 0), (200, 0, 100), (100, 100, 0), (0, 80, 200)]
    for k, (pan, pts) in enumerate(profiles):
        for p in pts:
            if not np.isnan(p[0]):
                cv2.circle(img, px(p), 2, palette[k % len(palette)], -1)
    for gap in gaps:
        colour = (0, 200, 0) if gap.straight_width_m >= need_m else (0, 0, 255)
        a, b = px(gap.left.point), px(gap.right.point)
        cv2.line(img, a, b, colour, 2)
        cv2.circle(img, a, 5, colour, -1)
        cv2.circle(img, b, 5, colour, -1)
        cv2.putText(img, f"{gap.width_m * 100:.0f}cm", ((a[0] + b[0]) // 2 + 4, (a[1] + b[1]) // 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    cv2.putText(img, f"top view, car frame; need {need_m * 100:.0f}cm", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


class Rig:
    """Gimbal camera, pan/tilt servo and ultrasonic; releases the servo PWM on close.

    A caller that already drives the servo and ultrasonic (route_follow) passes them in, so the
    PCA9685 that also runs the wheels is not set up twice; then close() leaves the servo alone.
    """

    def __init__(self, sonic: bool, servo=None, sensors=None):
        from servo import Servo, load_gimbal_home, load_left_90_pan, load_right_90_pan

        self.own_servo = servo is None
        self.grabber = UsbGrabber(CAMERA).start()
        try:
            self.servo = Servo() if servo is None else servo
            self.home_pan, self.home_tilt = load_gimbal_home()
            self.right_90 = load_right_90_pan(self.home_pan)
            self.left_90 = load_left_90_pan()
            self.pan_now = self.home_pan
            self.sensors = sensors
            if sonic and sensors is None:
                from safe_surround_cruise import SafetySensors

                self.sensors = SafetySensors(SimpleNamespace(no_ir=True, no_sonic=False, active_high=False))
            deadline = time.monotonic() + 8.0
            while self.grabber.latest()[0] is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if self.grabber.latest()[0] is None:
                raise RuntimeError(f"gimbal camera not ready: {self.grabber.error}")
        except Exception:
            self.grabber.close()
            raise

    def pan(self, pan_deg: float, settle_s: float = 0.6):
        from servo import servo_pan_for_bearing

        target = servo_pan_for_bearing(self.home_pan, self.right_90, pan_deg, self.left_90)
        while abs(target - self.pan_now) > 1:
            self.pan_now += max(-15, min(15, target - self.pan_now))
            self.servo.set_servo_pwm("0", self.pan_now)
            time.sleep(0.12)
        self.pan_now = target
        self.servo.set_servo_pwm("0", target)
        time.sleep(settle_s)

    def frame(self) -> np.ndarray:
        _, before = self.grabber.latest()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            frame, stamp = self.grabber.latest()
            if stamp > before + 0.15:
                return frame.copy()
            time.sleep(0.03)
        raise RuntimeError("gimbal camera stopped delivering frames")

    def sonic_cm(self, n: int = 5) -> float | None:
        readings = [r for r in (self.sensors.read_sonic() for _ in range(n)) if r is not None]
        time.sleep(0.02)
        return float(np.median(readings)) if readings else None

    def close(self):
        try:
            self.pan(0.0, settle_s=0.4)
        finally:
            if self.own_servo:
                self.servo.pwm_servo.set_pwm(8, 0, 0)
                self.servo.pwm_servo.set_pwm(9, 0, 0)
            self.grabber.close()


def calibrate(args, car: dict, cam: GimbalCamera):
    """Fit camera height, pitch and the ultrasonic offset so foot pixels match ultrasonic ranges."""
    from scipy.optimize import least_squares

    rig = Rig(sonic=True)
    samples = []
    try:
        for pan in range(-80, 81, 10):
            rig.pan(pan)
            frame = rig.frame()
            sonic = rig.sonic_cm()
            cols, rows = cam.feet(frame)
            cv2.imwrite(str(DEBUG_DIR / f"cal_{pan:+03d}.jpg"), frame)
            centre = (cols >= WIDTH / 2 - 40) & (cols <= WIDTH / 2 + 40)
            band = centre & (rows >= 0) & (rows < HEIGHT)
            spread = float(np.subtract(*np.percentile(rows[band], [90, 10]))) if band.sum() else 999.0
            # The ultrasonic cone sees the nearest surface; only trust it when one flat face fills the centre.
            if sonic is None or sonic > args.max_cal_cm or band.sum() < 0.8 * centre.sum() or spread > 12:
                print(f"  pan {pan:+3d}: sonic {sonic if sonic is None else round(sonic)}cm, "
                      f"{int(band.sum())}/{int(centre.sum())} centre columns, foot spread {spread:.0f}px -> skipped", flush=True)
                continue
            row = int(np.median(rows[band]))
            samples.append((pan, sonic / 100.0, WIDTH // 2, row))
            print(f"  pan {pan:+3d}: sonic {sonic:.0f}cm, centre foot row {row} (spread {spread:.0f}px)", flush=True)
        # A gap measured with a ruler sets the scale that the short ultrasonic ranges cannot.
        gap_pixels = []
        for pan in (args.gap_pans if args.gap_cm is not None else []):
            rig.pan(pan)
            frame = rig.frame()
            cols, rows, pts, too_close = cam.floor_profile(frame, pan)
            gaps = cam.find_gaps(cols, rows, pts, too_close, pan, cam.floor_mask(frame))
            if not gaps:
                print(f"  gap pan {pan:+3d}: no gap found", flush=True)
                continue
            gap = min(gaps, key=lambda g: abs(g.width_m * 100.0 - args.gap_cm))
            ends = [(c.column, int(rows[list(cols).index(c.column)])) for c in (gap.left, gap.right)]
            gap_pixels.append((pan, ends))
            print(f"  gap pan {pan:+3d}: corners at pixels {ends[0]} and {ends[1]}, "
                  f"{gap.width_m * 100:.0f}cm with the old geometry", flush=True)
    finally:
        rig.close()
    if len(samples) + len(gap_pixels) < 3:
        raise SystemExit(f"only {len(samples)} ultrasonic and {len(gap_pixels)} gap samples; "
                         f"face flat foam 15-80 cm away and retry")
    ranges = [s[1] for s in samples]
    if args.height_cm is None and not gap_pixels and max(ranges) - min(ranges) < 0.15:
        print(f"WARNING ultrasonic ranges only span {min(ranges) * 100:.0f}-{max(ranges) * 100:.0f}cm; "
              f"height and pitch are hard to tell apart. Pass --height-cm or --gap-cm.")

    base = dict(cam.geometry)
    fixed_z = None if args.height_cm is None else args.height_cm / 100.0

    def unpack(params):
        if fixed_z is None:
            return params[0], params[1], args.sonic_ahead_cm / 100.0
        return fixed_z, params[0], args.sonic_ahead_cm / 100.0

    def residuals(params):
        z, pitch, ahead = unpack(params)
        g = dict(base, z_m=z, pitch_deg=pitch, radius_m=0.0, pivot_x_m=0.0)
        out = []
        for _pan, sonic, col, row in samples:
            p = cam.project(np.array([[col, row]]), 0.0, g)[0]
            out.append((float(np.hypot(*p)) if not np.isnan(p[0]) else 5.0) - (sonic + ahead))
        for pan, ends in gap_pixels:
            a, b = cam.project(np.array(ends, np.float64), pan, dict(base, z_m=z, pitch_deg=pitch))
            width = float(np.hypot(*(a - b))) if not np.isnan(a[0] + b[0]) else 5.0
            out.append(width - args.gap_cm / 100.0)
        return np.array(out)

    if fixed_z is None:
        res = least_squares(residuals, [base["z_m"], base["pitch_deg"]], bounds=([0.03, -25.0], [0.25, 35.0]))
    else:
        res = least_squares(residuals, [base["pitch_deg"]], bounds=([-25.0], [35.0]))
    z, pitch, ahead = (float(v) for v in unpack(res.x))
    err = residuals(res.x)
    print(f"\nfit: height {z * 100:.1f}cm, pitch {pitch:+.1f}deg (down), ultrasonic {ahead * 100:+.1f}cm ahead of the lens")
    for (pan, sonic, _c, _r), e in zip(samples, err):
        print(f"  pan {pan:+3d}: sonic {sonic * 100:.0f}cm, camera {(sonic + ahead + e) * 100:.0f}cm, error {e * 100:+.1f}cm")
    for (pan, _ends), e in zip(gap_pixels, err[len(samples):]):
        print(f"  gap pan {pan:+3d}: ruler {args.gap_cm:.0f}cm, camera {args.gap_cm + e * 100:.1f}cm, error {e * 100:+.1f}cm")
    print(f"rms error {float(np.sqrt(np.mean(err ** 2))) * 100:.1f}cm over {len(err)} samples")
    if args.dry_run:
        return
    car["gimbal_camera"] = dict(base, z_m=round(z, 4), pitch_deg=round(pitch, 2), sonic_ahead_m=round(ahead, 4),
                          calibrated=True, tilt_servo=home_tilt(),
                          note="USB 雲台鏡頭，與超音波同支架。z_m/pitch_deg 由 gimbal_gap.py --calibrate 以超音波距離"
                               "（加上 --gap-cm 尺量縫寬）擬合；pivot_x_m/radius_m 為估計值。tilt 改變後需重標。")
    CAR_PATH.write_text(json.dumps(car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved gimbal_camera to {CAR_PATH}")


def home_tilt() -> int:
    from servo import load_gimbal_home

    return load_gimbal_home()[1]


def measure(args, car: dict, cam: GimbalCamera):
    body_m = [float(v) for v in car.get("body_m", [0.29, 0.18])]
    need = body_m[1] + 2.0 * args.margin_m
    if not cam.geometry.get("calibrated"):
        print("WARNING gimbal camera height/pitch not calibrated yet; run --calibrate first. Using estimates.")
    rig = Rig(sonic=False)
    profiles, all_gaps = [], []
    try:
        for pan in args.pans:
            rig.pan(pan)
            frame = rig.frame()
            cv2.imwrite(str(DEBUG_DIR / f"raw_{pan:+03d}.jpg"), frame)
            cols, rows, pts, too_close = cam.floor_profile(frame, pan)
            gaps = cam.find_gaps(cols, rows, pts, too_close, pan, cam.floor_mask(frame))
            profiles.append((pan, pts))
            all_gaps += gaps
            cv2.imwrite(str(DEBUG_DIR / f"gap_{pan:+03d}.jpg"), draw_frame(frame, cam, cols, rows, pts, gaps, pan, need))
            seen = int((~np.isnan(pts[:, 0])).sum())
            print(f"pan {pan:+3d}: {seen} foam floor points, {int(too_close.sum())} columns too close to see the foot", flush=True)
            for gap in gaps:
                l, r = gap.left.point * 100, gap.right.point * 100
                verdict = "fits" if gap.straight_width_m >= need else "too narrow"
                print(f"    gap {gap.width_m * 100:.0f}cm corner to corner, {gap.straight_width_m * 100:.0f}cm driving "
                      f"straight in toward {gap.bearing_deg:+.0f}deg ({verdict}, need {need * 100:.0f}cm); "
                      f"left corner ({l[0]:.0f},{l[1]:+.0f})cm, right corner ({r[0]:.0f},{r[1]:+.0f})cm", flush=True)
    finally:
        rig.close()
    cv2.imwrite(str(DEBUG_DIR / "gap_top.png"), draw_top(profiles, all_gaps, need, body_m))
    print(f"images: {DEBUG_DIR}/gap_*.jpg, top view {DEBUG_DIR / 'gap_top.png'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="Fit camera height/pitch against the ultrasonic.")
    p.add_argument("--dry-run", action="store_true", help="With --calibrate: print the fit, do not save.")
    p.add_argument("--max-cal-cm", type=float, default=90.0, help="Ignore ultrasonic readings beyond this.")
    p.add_argument("--height-cm", type=float, default=None,
                   help="With --calibrate: lens centre height above the floor measured with a ruler; only pitch is fitted.")
    p.add_argument("--gap-cm", type=float, default=None,
                   help="With --calibrate: ruler width of a gap the gimbal sees at --gap-pans; sets the scale.")
    p.add_argument("--gap-pans", type=int, nargs="+", default=[20, 40, 60],
                   help="With --gap-cm: pan angles that see that gap.")
    p.add_argument("--sonic-ahead-cm", type=float, default=0.0,
                   help="With --calibrate: how far the ultrasonic face sits ahead of the lens (negative = behind).")
    p.add_argument("--pans", type=int, nargs="+", default=[-60, -40, -20, 0, 20, 40, 60],
                   help="Pan angles to measure at (deg, positive = left).")
    p.add_argument("--margin-m", type=float, default=0.025,
                   help="Clearance wanted on each side of the body: gap error ~1cm + aiming ~1cm + wobble.")
    return p.parse_args()


def main():
    args = parse_args()
    DEBUG_DIR.mkdir(exist_ok=True)
    car = load_json(CAR_PATH)
    cam = GimbalCamera(car)
    if args.calibrate:
        calibrate(args, car, cam)
    else:
        measure(args, car, cam)


if __name__ == "__main__":
    main()
