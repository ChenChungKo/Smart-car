#!/usr/bin/env python3
"""Car pose on arena_map.json from ArUco wall tags seen by the CSI fisheye cameras.

Corners are found on the raw fisheye frame, undistorted with that camera's
fisheye K/D, and solved with IPPE_SQUARE. The tag pose on the map plus the
camera mount on the car (car_nav.json) give the car's (x, y, yaw).

Live check (front + rear CSI, no motors):
    python3 aruco_localizer.py
Still image:
    python3 aruco_localizer.py --image some_front.jpg --camera front
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB = SERVER / "calibration_patterns"
MAP_PATH = SERVER / "arena_map.json"
CAR_PATH = SERVER / "car_nav.json"
DEBUG_DIR = SERVER / "aruco_debug"
# Every fisheye K/D in calibration_patterns/captures was solved at 640x480.
CALIB_W, CALIB_H = 640, 480


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wrap_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_T(rotation, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def world_from_tag(tag: dict) -> np.ndarray:
    """ArUco frame: x = tag right as seen from the front, y = up, z = out of the face.

    rotation_deg turns the printed tag in its own plane (180 = stuck upside down).
    """
    psi = math.radians(float(tag["facing_deg"]))
    normal = np.array([math.cos(psi), math.sin(psi), 0.0])
    right = np.array([-math.sin(psi), math.cos(psi), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    wall_R = np.column_stack([right, up, normal])
    return make_T(wall_R @ rot_z(float(tag.get("rotation_deg", 0.0))), tag["center_m"])


def car_from_camera(mount: dict) -> np.ndarray:
    """Car frame x forward, y left, z up. OpenCV camera x right, y down, z forward."""
    pitch = math.radians(float(mount["pitch_deg"]))
    forward = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.array([-math.sin(pitch), 0.0, -math.cos(pitch)])
    rotation = rot_z(float(mount.get("yaw_deg", 0.0))) @ np.column_stack([right, down, forward])
    return make_T(rotation, [mount["x_m"], mount.get("y_m", 0.0), mount["z_m"]])


@dataclass
class TagFix:
    camera: str
    tag_id: int
    x: float
    y: float
    yaw: float
    distance: float
    err_px: float
    tilt_deg: float
    usable: bool = True
    rays: np.ndarray | None = None
    fx: float = 1.0
    fit_px: float = 0.0
    off_axis_deg: float = 0.0
    view_deg: float = 0.0


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    stamp: float = 0.0
    fixes: list[TagFix] = field(default_factory=list)
    rms_px: float = 0.0

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)

    def text(self) -> str:
        ids = ",".join(f"{f.camera[0]}{f.tag_id}" for f in self.fixes) or "dead-reckon"
        return f"x={self.x:.2f} y={self.y:.2f} yaw={self.yaw_deg:+.0f}deg [{ids}]"


class ArucoLocalizer:
    def __init__(self, arena: dict, car: dict):
        self.arena = arena
        self.size = float(arena["tag_size_m"])
        half = self.size / 2.0
        self.object_points = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        self.world_T_tag = {int(tag["id"]): world_from_tag(tag) for tag in arena["tags"]}
        self.world_corners = {
            tag_id: (T[:3, :3] @ self.object_points.T).T + T[:3, 3]
            for tag_id, T in self.world_T_tag.items()
        }
        self.mounts = car["camera_mounts"]
        self.cam_T_car = {
            name: np.linalg.inv(car_from_camera(mount))
            for name, mount in self.mounts.items()
        }
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, arena.get("tag_dictionary", "DICT_4X4_50"))
        )
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Close tags bend under the fisheye; sample only the middle of each cell.
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.33
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.base_kd: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.rect_maps: dict = {}
        self.arena_size = arena.get("arena_size_m")
        self.max_distance_m = 2.5
        self.max_err_px = 3.0
        self.max_tilt_deg = 35.0
        # The fisheye intrinsics were calibrated without boards near the rim.
        self.max_off_axis_deg = float(car.get("max_off_axis_deg", 55.0))
        # Corners of a tag seen edge-on are poorly defined.
        self.max_view_deg = float(car.get("max_view_deg", 65.0))
        self.inlier_px = 5.0
        self.max_single_tag_m = 0.8
        self.last_rms_px: float | None = None
        self.last_reject = ""

    def kd(self, camera: str, width: int, height: int):
        if camera not in self.base_kd:
            folder = CALIB / "captures" / camera
            k = np.load(folder / "camera_0_K.npy").astype(np.float64)
            d = np.load(folder / "camera_0_D.npy").astype(np.float64).reshape(-1, 1)[:4]
            self.base_kd[camera] = (k, d)
        k, d = self.base_kd[camera]
        return np.diag([width / CALIB_W, height / CALIB_H, 1.0]) @ k, d

    def rectifier(self, camera: str, width: int, height: int):
        """Pinhole view of the central +-65 deg, where close tags have straight edges."""
        key = (camera, width, height)
        if key not in self.rect_maps:
            k, d = self.kd(camera, width, height)
            f = (width / 2.0) / math.tan(math.radians(65.0))
            new_k = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                k, d, np.eye(3), new_k, (width, height), cv2.CV_16SC2
            )
            self.rect_maps[key] = (m1, m2, new_k)
        return self.rect_maps[key]

    def detect(self, camera: str, gray: np.ndarray):
        """Tag id -> (corners in the raw image, normalised pinhole corners)."""
        h, w = gray.shape[:2]
        k, d = self.kd(camera, w, h)
        found = {}
        corners, ids, _ = self.detector.detectMarkers(gray)
        for quad, tag_id in zip(corners, [] if ids is None else ids.flatten()):
            norm = cv2.fisheye.undistortPoints(quad.reshape(-1, 1, 2).astype(np.float64), k, d)
            found[int(tag_id)] = (quad.reshape(4, 2), norm.reshape(4, 2))
        m1, m2, new_k = self.rectifier(camera, w, h)
        corners, ids, _ = self.detector.detectMarkers(cv2.remap(gray, m1, m2, cv2.INTER_LINEAR))
        for quad, tag_id in zip(corners, [] if ids is None else ids.flatten()):
            # Straight edges here give better corners than the curved raw ones.
            norm = (quad.reshape(4, 2).astype(np.float64) - new_k[:2, 2]) / new_k[0, 0]
            raw = cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), k, d).reshape(4, 2)
            found[int(tag_id)] = (raw.astype(np.float32), norm)
        return found, float(k[0, 0])

    def observe(self, camera: str, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        found, fx = self.detect(camera, gray)
        fixes: list[TagFix] = []
        for tag_id, (_raw, norm) in found.items():
            if tag_id not in self.world_T_tag:
                continue
            fix = self._solve(camera, tag_id, norm, fx)
            if fix is not None:
                fixes.append(fix)
        if not found:
            return fixes, (), None
        corners = tuple(raw.reshape(1, 4, 2) for raw, _norm in found.values())
        ids = np.array(list(found), dtype=np.int32).reshape(-1, 1)
        return fixes, corners, ids

    def ranges(self, camera: str, bgr: np.ndarray):
        """Lens-to-tag-centre distance and bearing for every tag, map not needed."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        found = []
        if ids is None:
            return found, corners, ids
        k, d = self.kd(camera, gray.shape[1], gray.shape[0])
        for quad, tag_id in zip(corners, ids.flatten()):
            norm = cv2.fisheye.undistortPoints(
                quad.reshape(-1, 1, 2).astype(np.float64), k, d
            ).reshape(-1, 2)
            ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                self.object_points, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue
            i = int(np.argmin(np.asarray(errs).reshape(-1)))
            t = tvecs[i].reshape(3)
            side_px = float(np.mean(np.linalg.norm(quad.reshape(4, 2) - np.roll(quad.reshape(4, 2), 1, axis=0), axis=1)))
            found.append({
                "id": int(tag_id),
                "distance_m": float(np.linalg.norm(t)),
                "bearing_deg": math.degrees(math.atan2(t[0], t[2])),
                "err_px": float(np.asarray(errs).reshape(-1)[i]) * float(k[0, 0]),
                "side_px": side_px,
            })
        return found, corners, ids

    def _solve(self, camera: str, tag_id: int, norm: np.ndarray, fx: float) -> TagFix | None:
        ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            self.object_points, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            return None
        best = None
        for rvec, tvec, err in zip(rvecs, tvecs, np.asarray(errs).reshape(-1)):
            rotation, _ = cv2.Rodrigues(rvec)
            cam_T_tag = make_T(rotation, tvec)
            world_T_car = self.world_T_tag[tag_id] @ np.linalg.inv(cam_T_tag) @ self.cam_T_car[camera]
            tilt = math.degrees(math.acos(float(np.clip(world_T_car[2, 2], -1.0, 1.0))))
            err_px = float(err) * fx
            # IPPE gives two mirror solutions; the car is upright on the floor.
            score = err_px + 0.1 * tilt
            if best is None or score < best[0]:
                t = tvec.reshape(3)
                facing = float(-rotation[:, 2] @ t / np.linalg.norm(t))
                view = math.degrees(math.acos(float(np.clip(facing, -1.0, 1.0))))
                best = (score, world_T_car, float(np.linalg.norm(t)), err_px, tilt, view)
        _score, T, distance, err_px, tilt, view = best
        if distance > self.max_distance_m or err_px > self.max_err_px:
            return None
        x, y = float(T[0, 3]), float(T[1, 3])
        # A far tag alone gives a poor pose, but its corners still help the joint fit.
        usable = tilt <= self.max_tilt_deg and self._inside(x, y, 0.2)
        rays = np.column_stack([norm, np.ones(len(norm))])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        centre = rays.mean(axis=0)
        off_axis = math.degrees(math.acos(float(centre[2] / np.linalg.norm(centre))))
        return TagFix(
            camera=camera,
            tag_id=tag_id,
            x=x,
            y=y,
            yaw=math.atan2(T[1, 0], T[0, 0]),
            distance=distance,
            err_px=err_px,
            tilt_deg=tilt,
            usable=usable,
            rays=rays,
            fx=fx,
            off_axis_deg=off_axis,
            view_deg=view,
        )

    def _central(self, fixes: list[TagFix]) -> list[TagFix]:
        """Drop rim tags and tags seen edge-on unless they are all we have."""
        central = [
            f for f in fixes
            if f.off_axis_deg <= self.max_off_axis_deg and f.view_deg <= self.max_view_deg
        ]
        return central or fixes

    def _inside(self, x: float, y: float, margin: float) -> bool:
        if self.arena_size is None:
            return True
        w, h = (float(v) for v in self.arena_size)
        return -margin <= x <= w + margin and -margin <= y <= h + margin

    def _residuals(self, params, fixes: list[TagFix], cameras: list[str] | None = None):
        """Angular corner error in pixels for a car flat on the floor at (x, y, yaw).

        With cameras given, params also carry one pitch per camera.
        """
        x, y, yaw = params[:3]
        car_T_world = np.linalg.inv(make_T(rot_z(math.degrees(yaw)), [x, y, 0.0]))
        cam_T_car = dict(self.cam_T_car)
        if cameras:
            for name, pitch in zip(cameras, params[3:]):
                mount = dict(self.mounts[name], pitch_deg=float(pitch))
                cam_T_car[name] = np.linalg.inv(car_from_camera(mount))
        out = []
        for f in fixes:
            T = cam_T_car[f.camera] @ car_T_world
            p = (T[:3, :3] @ self.world_corners[f.tag_id].T).T + T[:3, 3]
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            err = (p - f.rays) * f.fx
            if not cameras:
                # A tag's height in the image says little about range but a lot about
                # small pitch errors, so only its size, shape and bearing count here.
                err[:, 1] -= err[:, 1].mean()
            out.append(err.ravel())
        return np.concatenate(out)

    def _refine(self, fixes: list[TagFix], start, cameras=None, pitches=()):
        from scipy.optimize import least_squares

        x0 = np.array([*start, *pitches], dtype=np.float64)
        return least_squares(
            self._residuals, x0, args=(fixes, cameras), loss="soft_l1", f_scale=2.0
        )

    @staticmethod
    def _per_fix_rms(result, n: int) -> np.ndarray:
        return np.sqrt((result.fun.reshape(n, -1) ** 2).mean(axis=1) * 3.0)

    def _best_start(self, fixes: list[TagFix], cameras=None, pitches=()):
        # Single far tags flip between IPPE's two solutions; try each as a start.
        starts = [(f.x, f.y, f.yaw) for f in fixes if f.usable] or [(f.x, f.y, f.yaw) for f in fixes]
        return min((self._refine(fixes, s, cameras, pitches) for s in starts), key=lambda r: r.cost)

    def fuse(self, fixes: list[TagFix], stamp: float = 0.0) -> Pose2D | None:
        """Joint planar fit of every tag corner from every camera."""
        self.last_rms_px = None
        self.last_reject = "no tags"
        if not fixes:
            return None
        fixes = self._central(fixes)
        # Keep the largest set of tags that agree, not whichever tag fits itself best.
        candidates = []
        starts = [(f.x, f.y, f.yaw) for f in fixes if f.usable] or [(f.x, f.y, f.yaw) for f in fixes]
        for start in starts:
            res = self._refine(fixes, start)
            errs = self._per_fix_rms(res, len(fixes))
            kept = [f for f, e in zip(fixes, errs) if e <= self.inlier_px]
            if kept and len(kept) < len(fixes):
                res = self._refine(kept, res.x[:3])
                errs = self._per_fix_rms(res, len(kept))
            else:
                kept = list(fixes)
            candidates.append((len(kept), -float(np.mean(errs ** 2)), res, kept, errs))
        _n, _score, best, fixes, rms = max(candidates, key=lambda c: (c[0], c[1]))
        x, y, yaw = (float(v) for v in best.x[:3])
        total = float(np.sqrt(np.mean(rms ** 2)))
        self.last_rms_px = total
        # A large residual on a flat-floor fit usually means pitch_deg in car_nav.json is off.
        if total > 2.0 * self.max_err_px:
            self.last_reject = f"tags disagree (rms {total:.1f}px)"
            return None
        if not self._inside(x, y, 0.1):
            self.last_reject = f"pose outside arena ({x:.2f},{y:.2f})"
            return None
        # One small far tag pins range and bearing loosely; wait for a second tag.
        tags = {(f.camera, f.tag_id) for f in fixes}
        if len(tags) == 1 and min(f.distance for f in fixes) > self.max_single_tag_m:
            cam, tag_id = next(iter(tags))
            self.last_reject = f"only one far tag ({cam[0]}{tag_id} at {min(f.distance for f in fixes):.2f}m)"
            return None
        self.last_reject = ""
        for f, e in zip(fixes, rms):
            f.fit_px = float(e)
        return Pose2D(x, y, wrap_rad(yaw), stamp, list(fixes), total)

    def fit_pitch(self, fixes: list[TagFix]) -> tuple[dict[str, float], float | None]:
        """Pitch per camera that sees two or more tags, car parked on a flat floor."""
        fixes = self._central(fixes)
        cameras = sorted({f.camera for f in fixes if sum(g.camera == f.camera for g in fixes) >= 2})
        if not cameras:
            return {}, None
        pitches = [float(self.mounts[name]["pitch_deg"]) for name in cameras]
        res = self._best_start(fixes, cameras, pitches)
        rms = float(np.sqrt(np.mean(self._per_fix_rms(res, len(fixes)) ** 2)))
        return {name: float(p) for name, p in zip(cameras, res.x[3:])}, rms


class CsiGrabber:
    """Keep the latest frame from one Picamera2 CSI camera on a worker thread."""

    def __init__(self, name: str, camera_num: int, width: int, height: int):
        self.name = name
        self.camera_num = camera_num
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._frame = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"csi-{name}")
        self.error = ""

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        from picamera2 import Picamera2

        from camera_devices import create_csi_still_configuration, csi_array_to_bgr

        while not self._stop.is_set():
            camera = None
            try:
                camera = Picamera2(camera_num=self.camera_num)
                camera.configure(create_csi_still_configuration(camera, self.width, self.height))
                camera.start()
                camera.set_controls({"AeEnable": True, "AwbEnable": True})
                time.sleep(0.8)
                print(f"{self.name}: CSI camera_num={self.camera_num} {self.width}x{self.height}", flush=True)
                while not self._stop.is_set():
                    frame = csi_array_to_bgr(camera.capture_array())
                    with self._lock:
                        self._frame = frame
                        self._stamp = time.monotonic()
            except Exception as exc:
                self.error = str(exc)
                print(f"WARNING {self.name}: {exc}; retrying", flush=True)
            finally:
                if camera is not None:
                    try:
                        camera.stop()
                        camera.close()
                    except Exception:
                        pass
            self._stop.wait(2.0)

    def latest(self):
        with self._lock:
            return self._frame, self._stamp

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)


def open_grabbers(car: dict, width: int, height: int, rear: bool = True) -> dict[str, CsiGrabber]:
    names = ["front", "rear"] if rear else ["front"]
    grabbers = {}
    for name in names:
        mount = car["camera_mounts"][name]
        grabbers[name] = CsiGrabber(name, int(mount["camera_num"]), width, height).start()
    return grabbers


def pitch_text(loc: ArucoLocalizer, fixes: list[TagFix]) -> str:
    fitted, rms = loc.fit_pitch(fixes)
    if not fitted:
        return "pitch: need 2+ tags in one camera"
    return " ".join(f"{name} pitch_deg~{value:.1f}" for name, value in fitted.items()) + f" (rms {rms:.1f}px)"


def save_pitch(car_path: Path, car: dict, samples: dict[str, list[float]]) -> None:
    for name, values in samples.items():
        car["camera_mounts"][name]["pitch_deg"] = round(float(np.median(values)), 1)
    car_path.write_text(json.dumps(car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def draw_overlay(bgr, corners, ids, fixes: list[TagFix], label: str) -> np.ndarray:
    out = bgr.copy()
    if ids is not None and len(ids):
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
    scale = out.shape[1] / 640.0
    lines = [label] + [
        f"id{f.tag_id} d={f.distance:.2f}m err={f.err_px:.1f}px -> "
        f"({f.x:.2f},{f.y:.2f}) {math.degrees(f.yaw):+.0f}deg"
        for f in fixes
    ]
    for i, text in enumerate(lines):
        org = (int(10 * scale), int((24 + 22 * i) * scale))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 0), int(4 * scale), cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)
    return out


def draw_map(arena: dict, route=None, trail=None, pose: Pose2D | None = None, size_px: int = 520) -> np.ndarray:
    """Top-down arena view with +y up on screen."""
    w_m, h_m = (float(v) for v in arena["arena_size_m"])
    margin = 30
    scale = (size_px - 2 * margin) / max(w_m, h_m)
    img_w = int(w_m * scale) + 2 * margin
    img_h = int(h_m * scale) + 2 * margin
    img = np.full((img_h, img_w, 3), 245, np.uint8)

    def px(x, y):
        return int(round(margin + x * scale)), int(round(img_h - margin - y * scale))

    cv2.rectangle(img, px(0, h_m), px(w_m, 0), (60, 60, 60), 2)
    half = float(arena["tag_size_m"]) / 2.0
    for tag in arena["tags"]:
        cx, cy = tag["center_m"][:2]
        psi = math.radians(float(tag["facing_deg"]))
        rx, ry = -math.sin(psi), math.cos(psi)
        cv2.line(img, px(cx - rx * half, cy - ry * half), px(cx + rx * half, cy + ry * half), (0, 120, 255), 4)
        lx, ly = px(cx + math.cos(psi) * 0.08, cy + math.sin(psi) * 0.08)
        cv2.putText(img, str(tag["id"]), (lx - 5, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 200), 1, cv2.LINE_AA)
    if route:
        pts = [px(x, y) for x, y in route]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (200, 160, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (200, 160, 0), -1)
            cv2.putText(img, str(i), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 90, 0), 1, cv2.LINE_AA)
    if trail and len(trail) >= 2:
        pts = [px(x, y) for x, y, _yaw in trail]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (40, 40, 220), 2)
    if pose is not None:
        c = px(pose.x, pose.y)
        tip = px(pose.x + 0.15 * math.cos(pose.yaw), pose.y + 0.15 * math.sin(pose.yaw))
        cv2.circle(img, c, 8, (40, 40, 220), -1)
        cv2.arrowedLine(img, c, tip, (40, 40, 220), 2, tipLength=0.35)
    return img


def parse_args():
    p = argparse.ArgumentParser(description="Live ArUco localization check (no motors).")
    p.add_argument("--map", default=str(MAP_PATH))
    p.add_argument("--car", default=str(CAR_PATH))
    p.add_argument("--image", help="Localize one still image instead of live cameras.")
    p.add_argument("--camera", default="front", help="Camera name for --image.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--no-rear", action="store_true")
    p.add_argument("--calibrate-pitch", type=int, default=0, metavar="N",
                   help="Car parked still: fit each camera's pitch over N frames and write car_nav.json.")
    p.add_argument("--range", action="store_true",
                   help="Print lens-to-tag distance for any tag (tape-measure check, map not used).")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--hz", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    arena = load_json(Path(args.map))
    car = load_json(Path(args.car))
    loc = ArucoLocalizer(arena, car)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise SystemExit(f"cannot read {args.image}")
        fixes, corners, ids = loc.observe(args.camera, bgr)
        pose = loc.fuse(fixes)
        seen = [] if ids is None else ids.flatten().tolist()
        print(f"tags seen={seen} usable={[f.tag_id for f in fixes]}")
        print(pose.text() if pose else f"no pose (fit rms={loc.last_rms_px})")
        if fixes:
            print(pitch_text(loc, fixes))
        out = DEBUG_DIR / f"{Path(args.image).stem}_aruco.jpg"
        cv2.imwrite(str(out), draw_overlay(bgr, corners, ids, fixes, args.camera))
        print(f"overlay -> {out}")
        return

    show = not args.no_window and bool(os.environ.get("DISPLAY"))
    grabbers = open_grabbers(car, args.width, args.height, rear=not args.no_rear)
    trail: list[tuple[float, float, float]] = []
    last = {name: 0.0 for name in grabbers}
    period = 1.0 / max(args.hz, 0.5)
    pitch_samples: dict[str, list[float]] = {}
    if args.calibrate_pitch:
        print(f"Pitch calibration: keep the car still; collecting {args.calibrate_pitch} fits.")
    print("Press q in the window (or Ctrl+C) to quit, s to save frames.")
    try:
        while True:
            started = time.monotonic()
            views = []
            all_fixes: list[TagFix] = []
            for name, grabber in grabbers.items():
                frame, stamp = grabber.latest()
                if frame is None or stamp == last[name]:
                    continue
                last[name] = stamp
                if args.range:
                    found, corners, ids = loc.ranges(name, frame)
                    for r in found:
                        print(f"{name} id{r['id']}: {r['distance_m'] * 100:.1f} cm  "
                              f"bearing {r['bearing_deg']:+.0f}deg  side {r['side_px']:.0f}px  "
                              f"err {r['err_px']:.2f}px", flush=True)
                    label = f"{name}: " + "  ".join(f"id{r['id']} {r['distance_m'] * 100:.0f}cm" for r in found)
                    views.append(draw_overlay(frame, corners, ids, [], label))
                    continue
                fixes, corners, ids = loc.observe(name, frame)
                all_fixes.extend(fixes)
                views.append(draw_overlay(frame, corners, ids, fixes, name))
            if args.range:
                if show and views:
                    cv2.imshow("aruco_range", np.vstack([cv2.resize(v, (640, 480)) for v in views]))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(max(0.0, period - (time.monotonic() - started)))
                continue
            if args.calibrate_pitch:
                fitted, rms = loc.fit_pitch(all_fixes)
                if fitted and rms is not None and rms <= 2.0 * loc.max_err_px:
                    for name, value in fitted.items():
                        pitch_samples.setdefault(name, []).append(value)
                    print(" ".join(f"{n} {v:.1f}deg" for n, v in fitted.items()) + f" rms {rms:.1f}px", flush=True)
                else:
                    print("need 2+ tags in a camera" if not fitted else f"fit rejected rms {rms:.1f}px", flush=True)
                if pitch_samples and min(len(v) for v in pitch_samples.values()) >= args.calibrate_pitch:
                    save_pitch(Path(args.car), car, pitch_samples)
                    print("saved pitch_deg: " + " ".join(
                        f"{n}={car['camera_mounts'][n]['pitch_deg']}" for n in pitch_samples) + f" -> {args.car}")
                    break
                time.sleep(max(0.0, period - (time.monotonic() - started)))
                continue
            pose = loc.fuse(all_fixes, started)
            if pose is not None:
                trail.append((pose.x, pose.y, pose.yaw))
                trail = trail[-200:]
            if pose is None:
                print("no tag" if not all_fixes else f"tags seen but fit rejected (rms {loc.last_rms_px:.1f}px; "
                      "run --calibrate-pitch)", flush=True)
            else:
                print(f"{pose.text()} rms={pose.rms_px:.1f}px", flush=True)
            if show and views:
                small = [cv2.resize(v, (480, 360)) for v in views]
                strip = np.vstack(small)
                panel = draw_map(arena, trail=trail, pose=pose, size_px=strip.shape[0])
                pad = np.full((strip.shape[0], panel.shape[1], 3), 245, np.uint8)
                pad[: panel.shape[0]] = panel[: strip.shape[0]]
                cv2.imshow("aruco_localizer", np.hstack([strip, pad]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    stamp = time.strftime("%H%M%S")
                    for name, grabber in grabbers.items():
                        frame, _ = grabber.latest()
                        if frame is not None:
                            cv2.imwrite(str(DEBUG_DIR / f"{name}_{stamp}.jpg"), frame)
                    print(f"saved frames to {DEBUG_DIR}")
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        for grabber in grabbers.values():
            grabber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
