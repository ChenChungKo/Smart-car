#!/usr/bin/env python3
"""Draw the surroundings from the front camera and one ultrasonic sweep.

Stationary. Wheels are never enabled. The gimbal steps across ±90° and the
front camera's blue-foam floor contact is drawn in the same car frame.

The layout follows Ohya, Kosaka and Kak (IEEE TRA 1998): the view is split
into bearings, a close ultrasonic return (their 50 cm gate) marks a blocked
direction, the camera marks a stationary obstacle where it meets the floor,
and the remaining arc is the open passage. Bearings outside the sweep are
unknown. Distance is the ultrasonic reading when that bearing has one;
otherwise it is the camera's floor contact.

    python3 surround_map.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aruco_localizer import CAR_PATH, ArucoLocalizer, CsiGrabber, load_json, open_grabbers
from floor_scan import FloorScanner
from pca9685 import PCA9685
from safe_surround_cruise import SafetySensors
from servo import load_gimbal_home, load_left_90_pan, load_right_90_pan, servo_pan_for_bearing

SERVER = Path(__file__).resolve().parent
OUT = SERVER / "route_follow_debug"
SONIC_X_CM = 10.0
# The HC-SR04 cans sit ~4 cm in front of the pan servo shaft, so rotating the
# gimbal swings the sensor face outward by SONIC_ARM_CM in the beam direction.
SONIC_ARM_CM = 4.0
GATE_CM = 50.0
STEP_DEG = 15
FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def pan_pulse(angle: int) -> int:
    return 2500 - int((angle + 10) / 0.09)


def tilt_pulse(angle: int) -> int:
    return 500 + int((angle + 10) / 0.09)


def slew(pwm: PCA9685, pan_now: int, target: int) -> int:
    target = int(target)
    while abs(target - pan_now) > 1:
        pan_now += max(-12, min(12, target - pan_now))
        pwm.set_servo_pulse(8, pan_pulse(pan_now))
        time.sleep(0.08)
    pwm.set_servo_pulse(8, pan_pulse(target))
    return target


def release(pwm: PCA9685) -> None:
    pwm.set_pwm(8, 0, 0)
    pwm.set_pwm(9, 0, 0)
    pwm.close()


def grab_csi(car: dict, name: str):
    """One CSI still at the fixed exposure. The fisheye ring is black, so the mean stays modest."""
    mount = car["camera_mounts"][name]
    grabber = CsiGrabber(name, int(mount["camera_num"]), 1280, 960).start()
    frame = None
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        grabbed, _stamp = grabber.latest()
        if grabbed is None:
            time.sleep(0.05)
            continue
        frame = grabbed
        if float(frame.mean()) > 70.0:
            break
        time.sleep(0.05)
    if frame is not None and float(frame.mean()) < 50.0:
        # A close grey wall leaves the 40 ms frame dark. Hold a longer exposure and reshoot.
        grabber.close()
        from picamera2 import Picamera2
        from camera_devices import create_csi_still_configuration, csi_array_to_bgr

        camera = Picamera2(camera_num=int(mount["camera_num"]))
        camera.configure(create_csi_still_configuration(camera, 1280, 960))
        camera.start()
        camera.set_controls({
            "AeEnable": False, "AwbEnable": True, "ExposureTime": 80000, "AnalogueGain": 8.0,
        })
        time.sleep(0.4)
        for _ in range(6):
            frame = csi_array_to_bgr(camera.capture_array())
        camera.close()
        print(f"  {name} held at 80 ms, brightness {frame.mean():.0f}", flush=True)
        return frame
    grabber.close()
    if frame is None:
        raise RuntimeError(f"{name} camera has no frame")
    print(f"  {name} brightness {frame.mean():.0f}", flush=True)
    return frame


def grab_front(car: dict):
    return grab_csi(car, "front")


def usb_torn(frame: np.ndarray) -> bool:
    """Same tear test as the stable left-camera preview: a hard step across a row."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    row = np.abs(gray[1:] - gray[:-1]).mean(axis=1)
    if row.size < 8:
        return True
    bottom = float(row[-40:].max()) if row.size >= 40 else float(row.max())
    return bottom > 42.0 or float(row.max()) > 58.0


def grab_usb(name: str):
    """1080p YUYV, one decoded frame, then scale to 640x480.

    Native 640x480 on these cameras is a center crop and tears. Extra read()
    calls on the left USB stream also tear it. This is the single-shot path
    used by the surround preview.
    """
    from camera_devices import resolve_usb_capture_index

    hardware = json.loads((SERVER / "camera_hardware.json").read_text(encoding="utf-8"))
    index, note = resolve_usb_capture_index(hardware[name])
    subprocess.run(
        [
            "v4l2-ctl", "-d", f"/dev/video{index}",
            "--set-fmt-video=width=1920,height=1080,pixelformat=YUYV",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open /dev/video{index}")
    # FourCC before size, or the driver stays in the cropped 640 mode.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 6)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"{name}: USB {note} /dev/video{index} {got[0]}x{got[1]} YUYV", flush=True)
    frame = None
    try:
        time.sleep(0.6)
        for _ in range(8):
            cap.grab()
        for _ in range(3):
            if not cap.grab():
                continue
            ok, grabbed = cap.retrieve()
            if not ok or grabbed is None:
                continue
            small = cv2.resize(grabbed, (640, 480), interpolation=cv2.INTER_AREA)
            frame = small
            if not usb_torn(small):
                break
    finally:
        cap.release()
    if frame is None:
        raise RuntimeError(f"{name} camera has no frame")
    print(f"  {name} brightness {frame.mean():.0f} torn={usb_torn(frame)}", flush=True)
    return frame


def scanner_for(car: dict, camera: str) -> FloorScanner:
    arena = load_json(SERVER / "arena_map.json")
    # The left camera's brighter exposure washes the foam, so its blue is less saturated
    # than the front CSI. 110 still misses the grey floor.
    lo = (90, 110, 40) if camera in ("left", "right") else (90, 170, 40)
    return FloorScanner(ArucoLocalizer(arena, car), car["camera_mounts"][camera], camera, hsv_lo=lo)


def camera_points(car: dict, frame, camera: str = "front") -> np.ndarray:
    scanner = scanner_for(car, camera)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"surround_{camera}.jpg"), frame)
    if camera == "front":
        cv2.imwrite(str(OUT / "surround_mask.png"), scanner.mask(frame))
    points = scanner.scan(frame)
    if camera != "rear":
        return points
    # The rear wall in this arena is grey, so the blue-foam scan misses it.
    plain = scanner.plain_contact(frame) * 100.0
    if len(plain):
        keep = (plain[:, 0] < -BODY_CM[0]) & (plain[:, 0] > -40.0) & (np.abs(plain[:, 1]) < 20.0)
        plain = plain[keep] / 100.0
    parts = [pts for pts in (points, plain) if len(pts)]
    return np.vstack(parts) if parts else np.zeros((0, 2))


def sweep(sensors: SafetySensors) -> list[dict]:
    home, tilt = load_gimbal_home()
    right = load_right_90_pan(home)
    left = load_left_90_pan()
    pwm = PCA9685(0x40, debug=False)
    pwm.set_pwm_freq(50)
    pwm.set_servo_pulse(9, tilt_pulse(tilt))
    pan_now = home
    pwm.set_servo_pulse(8, pan_pulse(pan_now))
    time.sleep(0.3)
    rows = []
    try:
        for bearing in range(90, -91, -STEP_DEG):
            target = servo_pan_for_bearing(home, right, bearing, left)
            pan_now = slew(pwm, pan_now, target)
            time.sleep(0.45)
            cm = sensors.read_sonic()
            rows.append({"bearing_deg": bearing, "servo": target, "cm": None if cm is None else round(cm, 1)})
            print(f"  sonic {bearing:+4d} deg  {('-' if cm is None else f'{cm:.0f} cm'):>8}", flush=True)
        pan_now = slew(pwm, pan_now, home)
        time.sleep(0.2)
    finally:
        release(pwm)
    return rows


def bearing_of(x_cm: float, y_cm: float) -> float:
    return math.degrees(math.atan2(y_cm, x_cm - SONIC_X_CM))


def range_of(x_cm: float, y_cm: float) -> float:
    return math.hypot(x_cm - SONIC_X_CM, y_cm)


def fuse(sonic_rows: list[dict], points_m: np.ndarray) -> list[dict]:
    pts = points_m * 100.0
    sectors = []
    for row in sonic_rows:
        bearing = row["bearing_deg"]
        cam = None
        if len(pts):
            ang = np.vectorize(bearing_of)(pts[:, 0], pts[:, 1])
            dist = np.vectorize(range_of)(pts[:, 0], pts[:, 1])
            keep = np.abs(ang - bearing) <= STEP_DEG / 2.0
            if keep.any():
                cam = float(dist[keep].min())
        sonic = row["cm"]
        if sonic is not None and sonic <= GATE_CM:
            fused, source = sonic, "ultrasonic"
        elif cam is not None and sonic is not None:
            fused, source = sonic, "both"
        elif cam is not None:
            fused, source = cam, "camera"
        else:
            fused, source = sonic, "ultrasonic" if sonic is not None else "none"
        sectors.append({
            "bearing_deg": bearing,
            "sonic_cm": sonic,
            "camera_cm": None if cam is None else round(cam, 1),
            "fused_cm": None if fused is None else round(fused, 1),
            "source": source,
            "blocked": fused is not None and fused <= GATE_CM,
        })
    blocked = [s["blocked"] for s in sectors]
    grown = blocked[:]
    for i, flag in enumerate(blocked):
        if not flag:
            continue
        if i > 0:
            grown[i - 1] = True
        if i + 1 < len(grown):
            grown[i + 1] = True
    for sector, flag in zip(sectors, grown):
        sector["body_blocked"] = flag
    return sectors


def open_passage(sectors: list[dict]) -> tuple[int, int] | None:
    best = None
    start = None
    for i, sector in enumerate(sectors + [None]):
        free = sector is not None and not sector["body_blocked"] and sector["bearing_deg"] >= -60
        if free and start is None:
            start = i
        if not free and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    if best is None:
        return None
    return sectors[best[0]]["bearing_deg"], sectors[best[1]]["bearing_deg"]


def obstacle_overlay(frame: np.ndarray, car: dict, camera: str) -> np.ndarray:
    """White outline of the blue foam this camera treated as an obstacle."""
    scanner = scanner_for(car, camera)
    small = cv2.resize(frame, scanner.size, interpolation=cv2.INTER_AREA)
    mask = scanner.mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= 80]
    cv2.drawContours(small, kept, -1, (255, 255, 255), 2)
    return cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)


def draw(points_by_cam: dict[str, np.ndarray], sectors: list[dict], ir: dict, passage,
         frames: dict[str, np.ndarray], car: dict) -> Path:
    font = ImageFont.truetype(FONT_B, 26)
    reg = ImageFont.truetype(FONT, 16)
    bold = ImageFont.truetype(FONT_B, 16)
    thumbs = []
    labels = {"front": "Front camera", "left": "Left camera"}
    for name in ("front", "left"):
        if name not in frames:
            continue
        view = cv2.cvtColor(obstacle_overlay(frames[name], car, name), cv2.COLOR_BGR2RGB)
        photo = Image.fromarray(view)
        photo.thumbnail((620, 465), Image.Resampling.LANCZOS)
        thumbs.append((name, photo))
    photo_h = sum(photo.height + 52 for _, photo in thumbs)
    W, H = 1680, max(photo_h + 36, 760)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ink = (32, 30, 27)
    y = 8
    for name, photo in thumbs:
        d.text((24, y), labels[name], font=font, fill=ink)
        d.text((24, y + 28), "White line: obstacle this camera kept.", font=reg, fill=(90, 86, 78))
        img.paste(photo, (24, y + 50))
        y += photo.height + 62
    origin_x = 24 + max(photo.width for _, photo in thumbs) + 16
    fwd0, fwd1, left0, left1 = -40, 140, -130, 140
    pad_l, pad_t, pad_r, pad_b = origin_x, 78, 250, 36
    sc = min((W - pad_l - pad_r) / (left1 - left0), (H - pad_t - pad_b) / (fwd1 - fwd0))

    def P(fwd, left):
        return (pad_l + (left1 - left) * sc, pad_t + (fwd1 - fwd) * sc)

    ox, oy = P(SONIC_X_CM, 0)
    for radius in (25, 50, 75, 100, 125):
        d.arc([ox - radius * sc, oy - radius * sc, ox + radius * sc, oy + radius * sc],
              200, 340, fill=(214, 210, 202), width=1)
        tx, ty = P(SONIC_X_CM + radius, 0)
        d.text((tx + 4, ty - 8), str(radius), font=reg, fill=(120, 116, 108))

    def at(bearing: float, cm: float):
        return P(*sonic_hit(bearing, cm))

    if passage is not None:
        hi, lo = passage
        wedge = [P(SONIC_X_CM, 0)]
        for bearing in range(hi, lo - 1, -STEP_DEG):
            wedge.append(at(bearing, 70))
        if len(wedge) >= 3:
            d.polygon(wedge, fill=(214, 232, 214))
        d.line([P(SONIC_X_CM, 0), at(0.5 * (hi + lo), 78)], fill=(36, 120, 64), width=3)
    for sector in sectors:
        cm = sector["fused_cm"]
        if cm is None:
            continue
        color = (196, 64, 36) if sector["blocked"] else (150, 176, 160)
        d.line([P(SONIC_X_CM, 0), at(sector["bearing_deg"], min(cm, 130))], fill=color, width=5)

    point_color = {"front": (24, 96, 176), "left": (0, 140, 130)}
    for name, points_m in points_by_cam.items():
        color = point_color.get(name, (24, 96, 176))
        for x, y_cm in points_m * 100.0:
            px, py = P(float(x), float(y_cm))
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
    for sector in sectors:
        cm = sector["sonic_cm"]
        if cm is None:
            continue
        px, py = at(sector["bearing_deg"], cm)
        d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(196, 64, 36) if cm <= GATE_CM else (24, 96, 176))

    d.polygon([P(14.5, 9), P(14.5, -9), P(-14.5, -9), P(-14.5, 9)], fill=(236, 232, 224), outline=ink)
    d.polygon([P(14.5, 6), P(20, 0), P(14.5, -6)], outline=ink)
    cx, cy = P(-1, 0)
    d.text((cx - 14, cy - 8), "car", font=bold, fill=ink)
    d.ellipse((ox - 4, oy - 4, ox + 4, oy + 4), fill=ink)

    d.text((origin_x, 18), "Range", font=font, fill=ink)
    d.text((origin_x, 46), "Nose up. Red inside 50 cm.", font=reg, fill=(90, 86, 78))

    lx = W - 230
    d.rounded_rectangle((lx - 16, 90, W - 20, 90 + 48 + len(sectors) * 26), radius=8,
                        fill=(255, 255, 255), outline=(220, 216, 208))
    d.text((lx, 100), "bearing", font=bold, fill=ink)
    d.text((lx + 70, 100), "sonic", font=bold, fill=ink)
    d.text((lx + 130, 100), "cam", font=bold, fill=ink)
    for i, sector in enumerate(sectors):
        b = sector["bearing_deg"]
        name = "ahead" if b == 0 else (("L " if b > 0 else "R ") + str(abs(b)))
        yy = 128 + i * 26
        d.text((lx, yy), name, font=reg, fill=(196, 64, 36) if sector["blocked"] else ink)
        d.text((lx + 70, yy), "-" if sector["sonic_cm"] is None else f"{sector['sonic_cm']:.0f}", font=reg, fill=ink)
        d.text((lx + 130, yy), "-" if sector["camera_cm"] is None else f"{sector['camera_cm']:.0f}", font=reg, fill=ink)
    note_y = 160 + len(sectors) * 26
    if passage is None:
        passage_text = "No open passage inside the front 120 deg."
    else:
        hi, lo = passage
        def side(b):
            return "ahead" if b == 0 else (("L" if b > 0 else "R") + str(abs(b)))
        passage_text = f"Open passage {side(hi)} to {side(lo)}."
    d.text((origin_x, H - 36), passage_text, font=bold, fill=(36, 120, 64) if passage else (196, 64, 36))
    hit = [name for name, value in ir.items() if value and name in
           ("front_left", "front_right", "wheel_left_front", "wheel_left_rear",
            "wheel_right_front", "wheel_right_rear")]
    if hit:
        d.text((24, y), "IR on: " + ", ".join(hit), font=reg, fill=(90, 86, 78))
    d.ellipse((origin_x, H - 58, origin_x + 10, H - 48), fill=(24, 96, 176))
    d.text((origin_x + 16, H - 62), "front camera", font=reg, fill=ink)
    d.ellipse((origin_x + 140, H - 58, origin_x + 150, H - 48), fill=(0, 140, 130))
    d.text((origin_x + 156, H - 62), "left camera", font=reg, fill=ink)
    path = OUT / "surround_map.png"
    img.save(path)
    return path


def scan_views(car: dict, frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    points = {}
    for name, frame in frames.items():
        points[name] = camera_points(car, frame, name)
        print(f"  {name}: {len(points[name])} floor points", flush=True)
    return points


def combined(points_by_cam: dict[str, np.ndarray]) -> np.ndarray:
    parts = [pts for pts in points_by_cam.values() if len(pts)]
    if not parts:
        return np.zeros((0, 2))
    return np.vstack(parts)


BODY_CM = (14.5, 9.0)
NEAR_CM = 30.0
IR_RANGE_CM = 8.0
# On the body face, looking outward. The rear wheels look out to the side, not backward.
IR_FACE = {
    "front_left": (BODY_CM[0], 6.0, 1.0, 0.0),
    "front_right": (BODY_CM[0], -6.0, 1.0, 0.0),
    "wheel_left_front": (8.0, BODY_CM[1], 0.0, 1.0),
    "wheel_right_front": (8.0, -BODY_CM[1], 0.0, -1.0),
    "wheel_left_rear": (-8.0, BODY_CM[1], 0.0, 1.0),
    "wheel_right_rear": (-8.0, -BODY_CM[1], 0.0, -1.0),
}


def body_gap_cm(fwd: float, left: float) -> float:
    """Distance outside the car body. Zero when the point is on or inside the body."""
    dx = max(abs(fwd) - BODY_CM[0], 0.0)
    dy = max(abs(left) - BODY_CM[1], 0.0)
    return math.hypot(dx, dy)


def sonic_hit(bearing: float, cm: float) -> tuple[float, float]:
    rad = math.radians(bearing)
    r = cm + SONIC_ARM_CM
    return SONIC_X_CM + r * math.cos(rad), r * math.sin(rad)


def side_wall_left_cm(rows: list[dict], sign: float) -> float | None:
    """Lateral position of a side wall, from ultrasonic hits at 60° or more.

    The side-camera mounts are still estimates, so these returns set the wall distance.
    The camera still says how far along the car the wall runs.
    """
    vals = []
    for row in rows:
        cm = row["cm"]
        bearing = row["bearing_deg"]
        if cm is None or cm > GATE_CM:
            continue
        if sign > 0 and bearing < 60:
            continue
        if sign < 0 and bearing > -60:
            continue
        _fwd, left = sonic_hit(bearing, cm)
        vals.append(left)
    if not vals:
        return None
    return float(np.median(vals))


def at_sonic_range(fwd: float, left: float, rows: list[dict]) -> tuple[float, float]:
    """Keep the camera bearing. Use a close ultrasonic return for the distance.

    The floor contact has been running about 10 cm short of the ultrasonic. A return
    farther than the 50 cm gate is a beam that missed, so the camera point stays.
    """
    bearing = math.degrees(math.atan2(left, fwd - SONIC_X_CM))
    chosen = None
    for row in rows:
        if row["cm"] is None or row["cm"] > GATE_CM:
            continue
        delta = abs((bearing - row["bearing_deg"] + 180) % 360 - 180)
        if delta > 15.0:
            continue
        if chosen is None or delta < chosen[0]:
            chosen = (delta, row["cm"])
    if chosen is None:
        return fwd, left
    return sonic_hit(bearing, chosen[1])


def grab_gimbal_at(bearing_deg: int) -> tuple[np.ndarray, list[dict]]:
    """Turn gimbal to bearing_deg, grab one 1080p->640x480 frame, detect ArUco tags, return home and release."""
    home, tilt = load_gimbal_home()
    right = load_right_90_pan(home)
    left = load_left_90_pan()
    pwm = PCA9685(0x40, debug=False)
    pwm.set_pwm_freq(50)
    pwm.set_servo_pulse(9, tilt_pulse(tilt))
    pan_now = home
    pwm.set_servo_pulse(8, pan_pulse(pan_now))
    time.sleep(0.3)
    try:
        target = servo_pan_for_bearing(home, right, bearing_deg, left)
        pan_now = slew(pwm, pan_now, target)
        time.sleep(0.45)
        frame = grab_usb("gimbal")
        pan_now = slew(pwm, pan_now, home)
        time.sleep(0.2)
    finally:
        release(pwm)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "surround_gimbal.jpg"), frame)
    tags = gimbal_aruco_tags(frame, bearing_deg)
    return frame, tags


def gimbal_aruco_tags(frame: np.ndarray, bearing_deg: int) -> list[dict]:
    """Detect ArUco markers on a gimbal frame and solve range & car-frame bearing."""
    arena = load_json(SERVER / "arena_map.json")
    car = load_json(Path(CAR_PATH))
    loc = ArucoLocalizer(arena, car)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = loc.detector.detectMarkers(gray)
    if ids is None:
        return []
    gk = np.load(SERVER / "calibration_patterns/captures/gimbal/camera_0_K.npy")
    gd = np.load(SERVER / "calibration_patterns/captures/gimbal/camera_0_D.npy")
    vis = frame.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.imwrite(str(OUT / "surround_gimbal_aruco.jpg"), vis)
    found = []
    for quad, tid in zip(corners, ids.ravel()):
        pts = quad.reshape(-1, 1, 2).astype(np.float64)
        norm = cv2.fisheye.undistortPoints(pts, gk, gd).reshape(-1, 2)
        ok, _rvecs, tvecs, errs = cv2.solvePnPGeneric(
            loc.object_points, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        i = int(np.argmin(np.asarray(errs).reshape(-1)))
        t = tvecs[i].reshape(3)
        dist_cm = float(np.linalg.norm(t)) * 100.0
        cam_deg = math.degrees(math.atan2(t[0], t[2]))
        found.append({
            "id": int(tid),
            "gimbal_bearing_deg": int(bearing_deg),
            "distance_cm": round(dist_cm, 1),
            "car_bearing_deg": round(float(bearing_deg) - cam_deg, 1),
        })
    return found


def draw_near(points_by_cam: dict[str, np.ndarray], rows: list[dict], ir: dict,
              gimbal_frame: np.ndarray | None = None, gimbal_bearing: int = 30,
              gimbal_tags: list[dict] | None = None) -> Path:
    """Top-down picture of only what sits within 30 cm of the body."""
    font = ImageFont.truetype(FONT_B, 26)
    reg = ImageFont.truetype(FONT, 18)
    bold = ImageFont.truetype(FONT_B, 18)
    W, H = 1280, 980
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    ink = (32, 30, 27)
    sc = 7.0

    def P(fwd, left):
        # Nose up. The close walls are on the right and behind, so the car sits low and right.
        return (720 - left * sc, 700 - fwd * sc)

    right_pts = points_by_cam.get("right", np.zeros((0, 2))) * 100.0
    if len(right_pts):
        right_left = float(np.median(right_pts[:, 1]))
    else:
        right_left = side_wall_left_cm(rows, -1.0) or -16.0
    left_pts = points_by_cam.get("left", np.zeros((0, 2))) * 100.0
    left_wall = float(np.median(left_pts[:, 1])) if len(left_pts) else None
    rear_pts = points_by_cam.get("rear", np.zeros((0, 2))) * 100.0
    rear_fwd = float(np.median(rear_pts[:, 0])) if len(rear_pts) else -20.5

    hl, hw, radius = BODY_CM[0], BODY_CM[1], NEAR_CM
    raw = []
    for y in np.linspace(-hw, hw, 8):
        raw.append((hl + radius, float(y)))
    for deg in range(0, 91, 8):
        rad = math.radians(deg)
        raw.append((hl + radius * math.cos(rad), hw + radius * math.sin(rad)))
    for x in np.linspace(hl, -hl, 8):
        raw.append((float(x), hw + radius))
    for deg in range(90, 181, 8):
        rad = math.radians(deg)
        raw.append((-hl + radius * math.cos(rad), hw + radius * math.sin(rad)))
    for y in np.linspace(hw, -hw, 8):
        raw.append((-hl - radius, float(y)))
    for deg in range(180, 271, 8):
        rad = math.radians(deg)
        raw.append((-hl + radius * math.cos(rad), -hw + radius * math.sin(rad)))
    for x in np.linspace(-hl, hl, 8):
        raw.append((float(x), -hw - radius))
    for deg in range(270, 361, 8):
        rad = math.radians(deg)
        raw.append((hl + radius * math.cos(rad), -hw + radius * math.sin(rad)))
    zone = [P(fwd, left) for fwd, left in raw]
    d.polygon(zone, fill=(247, 248, 250), outline=(140, 148, 160))
    # 2 cm grid clipped to the 30 cm zone and bounded by the right and rear walls.
    obs_list = []
    for name, pts in points_by_cam.items():
        for x, y in pts * 100.0:
            f_pt, l_pt = float(x), float(y)
            if name == "front":
                f_pt, l_pt = at_sonic_range(f_pt, l_pt, rows)
            if l_pt >= right_left - 0.8 and f_pt >= rear_fwd - 0.8:
                obs_list.append((f_pt, l_pt))
    for row in rows:
        if row["cm"] is not None:
            f_pt, l_pt = sonic_hit(row["bearing_deg"], row["cm"])
            if l_pt >= right_left - 0.8 and f_pt >= rear_fwd - 0.8:
                obs_list.append((f_pt, l_pt))
    obs_arr = np.array(obs_list) if obs_list else np.zeros((0, 2))

    def cell_in_zone(f_c: float, l_c: float) -> bool:
        if f_c < rear_fwd or l_c < right_left:
            return False
        dx = max(abs(f_c) - hl, 0.0)
        dy = max(abs(l_c) - hw, 0.0)
        return math.hypot(dx, dy) <= radius

    # Shade 2 cm cells: orange-red = obstacle contact cell.
    for g_fwd in range(-20, 44, 2):
        for g_left in range(-16, 38, 2):
            fc, lc = g_fwd + 1.0, g_left + 1.0
            if not cell_in_zone(fc, lc):
                continue
            if len(obs_arr):
                d_pt = np.hypot(obs_arr[:, 0] - fc, obs_arr[:, 1] - lc).min()
            else:
                d_pt = 99.0
            if d_pt <= 1.8:
                x0, y0 = P(g_fwd + 2.0, g_left + 2.0)
                x1, y1 = P(g_fwd, g_left)
                d.rectangle((x0, y0, x1, y1), fill=(255, 204, 153))

    grid_color = (218, 222, 228)
    for g_fwd in range(-20, 46, 2):
        if g_fwd < rear_fwd:
            continue
        dx = max(abs(g_fwd) - hl, 0.0)
        if dx > radius:
            continue
        max_dy = math.sqrt(radius * radius - dx * dx)
        l_hi = hw + max_dy
        l_lo = max(-hw - max_dy, right_left)
        if l_hi > l_lo:
            y_px = P(g_fwd, 0)[1]
            d.line([P(g_fwd, l_hi)[0], y_px, P(g_fwd, l_lo)[0], y_px], fill=grid_color, width=1)
    for g_left in range(int(math.ceil(right_left)), 40, 2):
        dy = max(abs(g_left) - hw, 0.0)
        if dy > radius:
            continue
        max_dx = math.sqrt(radius * radius - dy * dy)
        f_hi = hl + max_dx
        f_lo = max(-hl - max_dx, rear_fwd)
        if f_hi > f_lo:
            x_px = P(0, g_left)[0]
            d.line([x_px, P(f_hi, g_left)[1], x_px, P(f_lo, g_left)[1]], fill=grid_color, width=1)
    # The right and rear sides are walls, so the page past them stays blank.
    d.rectangle((P(0, right_left)[0], 0, W, H), fill=(255, 255, 255))
    d.rectangle((0, P(rear_fwd, 0)[1], W, H), fill=(255, 255, 255))
    d.polygon([P(BODY_CM[0], BODY_CM[1]), P(BODY_CM[0], -BODY_CM[1]),
               P(-BODY_CM[0], -BODY_CM[1]), P(-BODY_CM[0], BODY_CM[1])],
              fill=(236, 232, 224), outline=ink)
    d.polygon([P(BODY_CM[0], 6), P(BODY_CM[0] + 6, 0), P(BODY_CM[0], -6)], outline=ink)
    cx, cy = P(-2, 0)
    d.text((cx - 16, cy - 10), "car", font=bold, fill=ink)
    # Draw a single clean target marker at (goal_f, goal_l) instead of filling grid squares.

    def inside_walls(fwd: float, left: float) -> bool:
        return left >= right_left - 0.8 and fwd >= rear_fwd - 0.8

    near_rows = []
    for row in rows:
        if row["cm"] is None:
            continue
        fwd, left = sonic_hit(row["bearing_deg"], row["cm"])
        gap = body_gap_cm(fwd, left)
        if gap <= NEAR_CM and inside_walls(fwd, left):
            near_rows.append((row, fwd, left, gap))
    for _row, fwd, left, _gap in near_rows:
        px, py = P(fwd, left)
        d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(196, 64, 36))
    colors = {
        "front": (24, 96, 176),
        "left": (0, 140, 130),
        "right": (120, 48, 160),
        "rear": (20, 120, 48),
    }
    for name, pts in points_by_cam.items():
        for x, y in pts * 100.0:
            fwd, left = float(x), float(y)
            if name == "front":
                fwd, left = at_sonic_range(fwd, left, rows)
            gap = body_gap_cm(fwd, left)
            if gap > NEAR_CM or not inside_walls(fwd, left):
                continue
            px, py = P(fwd, left)
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colors.get(name, ink))
    wall = (90, 86, 78)
    right_gap = max(abs(right_left) - BODY_CM[1], 0.0)
    d.line([P(rear_fwd, right_left), P(34, right_left)], fill=wall, width=8)
    d.line([P(rear_fwd, right_left), P(rear_fwd, 12)], fill=wall, width=8)
    d.text((P(8, right_left)[0] + 10, P(8, right_left)[1]), f"right wall ({right_gap:.0f} cm)", font=bold, fill=wall)
    rear_gap = max(abs(rear_fwd) - BODY_CM[0], 0.0)
    d.text((P(rear_fwd, -4)[0] - 90, P(rear_fwd, 0)[1] + 8), f"rear wall ({rear_gap:.0f} cm)", font=bold, fill=wall)
    if left_wall is not None:
        left_gap = max(left_wall - BODY_CM[1], 0.0)
        d.text((P(14, left_wall)[0] - 185, P(14, left_wall)[1]), f"left obstacle ({left_gap:.0f} cm)", font=bold, fill=(0, 110, 100))
    # Planned safest body position (fwd=+9 cm, left=+9 cm) rotated L30 deg to face the open exit.
    goal_f, goal_l, yaw_deg = 9.0, 9.0, 30.0
    mark = (24, 118, 56)
    rad_y = math.radians(yaw_deg)
    cy_rot, sy_rot = math.cos(rad_y), math.sin(rad_y)

    def R_pt(lf: float, ll: float) -> tuple[float, float]:
        return P(goal_f + cy_rot * lf - sy_rot * ll, goal_l + sy_rot * lf + cy_rot * ll)

    # Also show faint red corner sweep circle (r=17.1 cm) at (0,0) to show why rotating at (0,0) clips the right wall
    r_diag = math.hypot(BODY_CM[0], BODY_CM[1])
    ox0, oy0 = P(0.0, 0.0)
    d.arc([ox0 - r_diag * sc, oy0 - r_diag * sc, ox0 + r_diag * sc, oy0 + r_diag * sc],
          -40, 45, fill=(196, 64, 36), width=2)

    g_box = [
        R_pt(+BODY_CM[0], +BODY_CM[1]),
        R_pt(+BODY_CM[0], -BODY_CM[1]),
        R_pt(-BODY_CM[0], -BODY_CM[1]),
        R_pt(-BODY_CM[0], +BODY_CM[1]),
    ]
    for i in range(4):
        p0, p1 = g_box[i], g_box[(i + 1) % 4]
        for s_idx in range(0, 12, 2):
            t0, t1 = s_idx / 12.0, (s_idx + 1) / 12.0
            d.line([
                (p0[0] + (p1[0] - p0[0]) * t0, p0[1] + (p1[1] - p0[1]) * t0),
                (p0[0] + (p1[0] - p0[0]) * t1, p0[1] + (p1[1] - p0[1]) * t1),
            ], fill=mark, width=3)
    d.polygon([R_pt(BODY_CM[0], 5.5), R_pt(BODY_CM[0] + 5.5, 0.0), R_pt(BODY_CM[0], -5.5)], outline=mark, width=2)
    # Center-to-center trajectory from (0, 0) to (goal_f, goal_l)
    start, end = (0.0, 0.0), (goal_f, goal_l)
    d.ellipse((P(0, 0)[0] - 5, P(0, 0)[1] - 5, P(0, 0)[0] + 5, P(0, 0)[1] + 5), fill=ink)
    d.line([P(*start), P(*end)], fill=mark, width=5)
    df, dl = end[0] - start[0], end[1] - start[1]
    length = math.hypot(df, dl)
    uf, ul = df / length, dl / length
    pf, pl = -ul, uf
    tip = (end[0] + 1.5 * uf, end[1] + 1.5 * ul)
    base = (tip[0] - 4.5 * uf, tip[1] - 4.5 * ul)
    wing = 2.0
    d.polygon([
        P(*tip),
        P(base[0] + wing * pf, base[1] + wing * pl),
        P(base[0] - wing * pf, base[1] - wing * pl),
    ], fill=mark)
    d.text((P(goal_f - 4, goal_l + 32)[0] - 120, P(-10, 0)[1]),
           "1. shift (+9,+9) & rotate L30", font=bold, fill=mark)
    d.text((P(goal_f - 4, goal_l + 32)[0] - 120, P(-10, 0)[1] + 22),
           "   (all 4 corners >= 9.4 cm gap)", font=reg, fill=mark)
    # Step 2 arrow: drive straight ahead along L30 deg heading (163 cm open)
    s2_start = (goal_f + cy_rot * (BODY_CM[0] + 3.0), goal_l + sy_rot * (BODY_CM[0] + 3.0))
    s2_end = (goal_f + cy_rot * (BODY_CM[0] + 14.0), goal_l + sy_rot * (BODY_CM[0] + 14.0))
    d.line([P(*s2_start), P(*s2_end)], fill=mark, width=4)
    uf2, ul2 = cy_rot, sy_rot
    pf2, pl2 = -ul2, uf2
    tip2 = (s2_end[0] + 2.0 * uf2, s2_end[1] + 2.0 * ul2)
    base2 = (tip2[0] - 4.5 * uf2, tip2[1] - 4.5 * ul2)
    d.polygon([
        P(*tip2),
        P(base2[0] + wing * pf2, base2[1] + wing * pl2),
        P(base2[0] - wing * pf2, base2[1] - wing * pl2),
    ], fill=mark)
    d.text((P(*s2_end)[0] - 210, P(*s2_end)[1] - 26), "2. drive straight out (163 cm open)", font=bold, fill=mark)
    lx = 930
    d.text((lx, 90), "bearing", font=bold, fill=ink)
    d.text((lx + 115, 90), "gap", font=bold, fill=ink)
    for i, (row, _fwd, _left, gap) in enumerate(near_rows):
        b = row["bearing_deg"]
        name = "ahead" if b == 0 else (("L " if b > 0 else "R ") + str(abs(b)))
        yy = 120 + i * 26
        d.ellipse((lx, yy + 4, lx + 12, yy + 16), fill=(196, 64, 36))
        d.text((lx + 20, yy), f"{name}  {row['cm']:.0f} cm", font=reg, fill=ink)
        d.text((lx + 150, yy), f"{gap:.0f} cm", font=reg, fill=ink)

    # Global Arena Pose + Gimbal Camera inset on the right panel
    gy = 408
    arena_x = round(-rear_fwd, 1)
    arena_y = round(-right_left, 1)
    d.rounded_rectangle((lx - 14, gy, W - 22, gy + 112), radius=8, fill=(244, 247, 252), outline=(180, 192, 210))
    d.text((lx, gy + 10), "Global Pose in 140x140 Arena", font=bold, fill=ink)
    d.text((lx, gy + 36), f"x = {arena_x:.1f} cm, y = {arena_y:.1f} cm, yaw = 0 deg", font=reg, fill=ink)
    d.text((lx, gy + 60), "Walls: rear x=0 (5.7 cm), right y=0 (7.0 cm)", font=reg, fill=(80, 86, 96))
    if gimbal_tags:
        t0 = gimbal_tags[0]
        d.text((lx, gy + 84),
               f"Gimbal -> Tag {t0['id']} at {t0['distance_cm']:.0f} cm (L{t0['car_bearing_deg']:.0f} deg)",
               font=bold, fill=(24, 96, 176))
    else:
        d.text((lx, gy + 84), "Tags seen: T0 (right 15cm), T6 (left 91cm), T4 (147cm)", font=reg, fill=(24, 96, 176))

    if gimbal_frame is not None:
        vis_g = gimbal_frame.copy()
        arena = load_json(SERVER / "arena_map.json")
        car = load_json(Path(CAR_PATH))
        loc = ArucoLocalizer(arena, car)
        g_gray = cv2.cvtColor(vis_g, cv2.COLOR_BGR2GRAY) if vis_g.ndim == 3 else vis_g
        c_g, id_g, _ = loc.detector.detectMarkers(g_gray)
        if id_g is not None:
            cv2.aruco.drawDetectedMarkers(vis_g, c_g, id_g)
        g_rgb = cv2.cvtColor(vis_g, cv2.COLOR_BGR2RGB)
        g_img = Image.fromarray(g_rgb)
        g_img.thumbnail((330, 248), Image.Resampling.LANCZOS)
        iy = gy + 128
        side_lbl = f"L{gimbal_bearing}" if gimbal_bearing > 0 else (f"R{abs(gimbal_bearing)}" if gimbal_bearing < 0 else "0")
        d.text((lx, iy), f"Gimbal camera at {side_lbl} deg (exit view):", font=bold, fill=mark)
        img.paste(g_img, (lx, iy + 26))
        d.rectangle((lx, iy + 26, lx + g_img.width, iy + 26 + g_img.height), outline=(140, 148, 160), width=2)
    d.text((28, 18), "Within 30 cm of the body", font=font, fill=ink)
    d.text((28, 50), "Nose up. Car in the lower right, against the right and rear walls.", font=reg,
           fill=(90, 86, 78))
    d.text((28, 74), "Red circle = ultrasonic; small dots = cameras (blue front, teal left, purple right, green rear).", font=reg, fill=(90, 86, 78))
    hit = [k for k, v in ir.items() if v and k in
           ("front_left", "front_right", "wheel_left_front", "wheel_left_rear",
            "wheel_right_front", "wheel_right_rear")]
    d.text((28, H - 36), "IR on: " + (", ".join(hit) if hit else "none"), font=reg, fill=(90, 86, 78))
    path = OUT / "surround_near.png"
    img.save(path)
    print(f"within 30cm: {len(near_rows)} ultrasonic hits", flush=True)
    for row, _fwd, _left, gap in near_rows:
        print(f"  {row['bearing_deg']:+4d} deg  {row['cm']:.0f} cm from gimbal, "
              f"{gap:.0f} cm outside the body", flush=True)
    return path


def publish(car, frames, points_by_cam, rows, ir) -> None:
    sectors = fuse(rows, combined(points_by_cam))
    passage = open_passage(sectors)
    path = draw(points_by_cam, sectors, ir, passage, frames, car)
    payload = {
        "sonic_x_cm": SONIC_X_CM,
        "gate_cm": GATE_CM,
        "ir": ir,
        "passage_deg": passage,
        "sectors": sectors,
        "camera_cm": {
            name: [[round(float(x) * 100, 1), round(float(y) * 100, 1)] for x, y in pts]
            for name, pts in points_by_cam.items()
        },
    }
    (OUT / "surround_map.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}", flush=True)
    if passage is None:
        print("open passage: none", flush=True)
    else:
        print(f"open passage: {passage[0]} to {passage[1]} deg", flush=True)


def main() -> None:
    car = load_json(Path(CAR_PATH))
    print("front camera", flush=True)
    frames = {"front": grab_front(car)}
    print("rear camera", flush=True)
    frames["rear"] = grab_csi(car, "rear")
    print("left camera", flush=True)
    frames["left"] = grab_usb("left")
    print("right camera", flush=True)
    frames["right"] = grab_usb("right")
    points = scan_views(car, frames)
    sensors = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=False, active_high=False))
    try:
        ir = sensors.read_ir()
        print("ultrasonic sweep", flush=True)
        rows = sweep(sensors)
    finally:
        sensors.close()
    sectors = fuse(rows, combined(points))
    passage = open_passage(sectors)
    exit_bearing = int(round(0.5 * (passage[0] + passage[1]))) if passage else 30
    print(f"gimbal camera at {exit_bearing:+d} deg", flush=True)
    g_frame, g_tags = grab_gimbal_at(exit_bearing)
    publish(car, frames, points, rows, ir)
    draw_near(points, rows, ir, gimbal_frame=g_frame, gimbal_bearing=exit_bearing, gimbal_tags=g_tags)


def redraw() -> None:
    """Add a fresh left-camera frame to the last sweep. Does not move the car or the gimbal."""
    car = load_json(Path(CAR_PATH))
    payload = json.loads((OUT / "surround_map.json").read_text(encoding="utf-8"))
    frames = {"front": cv2.imread(str(OUT / "surround_front.jpg"))}
    print("left camera", flush=True)
    frames["left"] = grab_usb("left")
    points = scan_views(car, frames)
    rows = [{"bearing_deg": s["bearing_deg"], "cm": s["sonic_cm"]} for s in payload["sectors"]]
    publish(car, frames, points, rows, payload["ir"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--redraw":
        redraw()
    else:
        main()
