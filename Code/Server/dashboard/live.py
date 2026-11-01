"""Hardware source of the standalone dashboard: all five cameras, the ArUco localization from all of them, the
ultrasonic, the six IR modules and the battery. Used when no lap is running (the cameras can have one owner only)."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, car_from_camera, draw_overlay, load_json, open_grabbers
from dashboard.hub import Hub

SERVER = Path(__file__).resolve().parents[1]
GIMBAL_MOUNT = {"x_m": 0.1381, "y_m": 0.0158, "z_m": 0.1146, "yaw_deg": 1.514, "pitch_deg": 3.9464}  # solved 2026-10-27
IR_NAMES = ("front_left", "front_right", "wheel_left_front", "wheel_left_rear", "wheel_right_front", "wheel_right_rear")


def usb_path(bus: str) -> Path:
    ctrl, port = bus.replace("usb-", "", 1).rsplit("-", 1)
    return Path("/dev/v4l/by-path") / f"platform-{ctrl}-usb-0:{port}:1.0-video-index0"


def all_mounts(car: dict) -> dict:
    mounts = {n: dict(m) for n, m in car["camera_mounts"].items()}
    mounts["gimbal"] = dict(car.get("gimbal_mount_home") or GIMBAL_MOUNT)
    return mounts


def cpu_percent(last: list) -> float:
    a = [float(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    total, idle = sum(a), a[3] + a[4]
    prev = last[0] if last else None
    last[:] = [(total, idle)]
    if not prev:
        return 0.0
    return round(100.0 * (1.0 - (idle - prev[1]) / max(total - prev[0], 1.0)), 0)


class LiveSource:
    def __init__(self, hub: Hub, cameras: bool = True, sensors: bool = True):
        self.hub = hub
        self.arena, self.car = load_json(MAP_PATH), load_json(CAR_PATH)
        self.hardware = load_json(SERVER / "camera_hardware.json")
        self.mounts = all_mounts(self.car)
        self.loc = ArucoLocalizer(self.arena, self.car)
        self.loc.cam_T_car["gimbal"] = np.linalg.inv(car_from_camera(self.mounts["gimbal"]))
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self.latest: dict[str, tuple[np.ndarray, float]] = {}
        self.fixes: list[tuple[float, str, list]] = []
        self.tags_seen: dict[str, list[int]] = {}
        self.want_cameras, self.want_sensors = cameras, sensors
        self.grabbers: dict = {}
        self.caps: dict = {}
        self.sensors = None
        w, h = (float(v) for v in self.arena["arena_size_m"])
        off = 0.14 + float(self.car.get("body_m", [0.275, 0.17])[1]) / 2.0
        self.route = [(off, off), (w - off, off), (w - off, h - off), (off, h - off), (off, off)]
        self.hub.set_state(mode="live", home=self.arena.get("home"), route=self.route, trail=[], blocks=[],
                           body_m=self.car.get("body_m"), arena_m=self.arena["arena_size_m"])
        self.trail: list = []

    # ---- cameras -------------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self.want_cameras:
            self.grabbers = open_grabbers(self.car, 1280, 960, rear=True)
            for name in ("gimbal", "left", "right"):
                path = usb_path(self.hardware[name]["usb_bus"])
                cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if cap.isOpened():
                    self.caps[name] = cap
                    self._thread(self._read_usb, name, cap)
                else:
                    self.hub.add_log(f"camera {name} did not open")
            self._thread(self._detect_loop)
        if self.want_sensors:
            self._thread(self._sensor_loop)
        self._thread(self._pose_loop)

    def _thread(self, target, *args) -> None:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        self.threads.append(t)

    def _read_usb(self, name: str, cap) -> None:
        while not self.stop.is_set():
            ok, frame = cap.read()
            if ok:
                self.latest[name] = (frame, time.time())
            else:
                time.sleep(0.05)

    def _frame(self, name: str):
        if name in self.grabbers:
            frame, stamp = self.grabbers[name].latest()
            return None if frame is None else (frame, stamp)
        return self.latest.get(name)

    def _detect_loop(self) -> None:
        names = ["front", "rear", "left", "right", "gimbal"]
        last_stamp: dict[str, float] = {}
        while not self.stop.is_set():
            for name in names:
                got = self._frame(name)
                if got is None or got[1] == last_stamp.get(name):
                    continue
                frame, stamp = got
                last_stamp[name] = stamp
                try:
                    fixes, corners, ids = self.loc.observe(name, frame)
                except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the dashboard
                    self.hub.add_log(f"{name}: {exc}")
                    continue
                self.fixes.append((time.time(), name, fixes))
                self.tags_seen[name] = [] if ids is None else sorted(int(i) for i in ids.flatten())
                view = draw_overlay(frame, corners, ids, fixes, name)
                self.hub.set_frame(name, view)
                time.sleep(0.02)
            time.sleep(0.05)

    def _pose_loop(self) -> None:
        cpu_state: list = []
        while not self.stop.is_set():
            time.sleep(0.5)
            now = time.time()
            self.fixes = [f for f in self.fixes if now - f[0] < 1.2]
            fixes = [fx for _t, _n, fl in self.fixes for fx in fl]
            pose = None
            try:
                pose = self.loc.fuse(fixes, time.monotonic()) if fixes else None
            except Exception as exc:  # noqa: BLE001
                self.hub.add_log(f"pose: {exc}")
            fields = {"tags_seen": dict(self.tags_seen), "cpu": cpu_percent(cpu_state), "temp_c": self._temp()}
            if pose is not None:
                fields["pose"] = {"x": pose.x, "y": pose.y, "yaw_deg": pose.yaw_deg, "rms_px": round(pose.rms_px, 1),
                                  "source": "tags " + ",".join(sorted({f"{f.camera[0]}{f.tag_id}" for f in pose.fixes})),
                                  "sigma_cm": None}
                self.trail.append((pose.x, pose.y))
                self.trail = self.trail[-300:]
                fields["trail"] = list(self.trail)
            else:
                fields["pose_note"] = self.loc.last_reject or "no tag"
            self.hub.set_state(**fields)

    @staticmethod
    def _temp() -> float | None:
        try:
            return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0, 1)
        except Exception:  # noqa: BLE001
            return None

    # ---- sensors --------------------------------------------------------------------------------------------------

    def _sensor_loop(self) -> None:
        try:
            from safe_surround_cruise import SafetySensors, read_battery_voltage
            self.sensors = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=False, active_high=False))
        except Exception as exc:  # noqa: BLE001
            self.hub.add_log(f"sensors unavailable: {exc}")
            return
        tick = 0
        while not self.stop.is_set():
            fields = {}
            try:
                ir = self.sensors.read_ir()
                fields["ir"] = {k: int(bool(ir.get(k))) for k in IR_NAMES}
            except Exception as exc:  # noqa: BLE001
                fields["ir_error"] = str(exc)
            if tick % 3 == 0:
                box: list = []
                worker = threading.Thread(target=lambda: box.append(self.sensors.read_sonic()), daemon=True)
                worker.start()
                worker.join(1.0)
                fields["sonic_cm"] = box[0] if box else None
                fields["sonic_pan_deg"] = 0.0
            if tick % 10 == 0:
                fields["battery_v"] = read_battery_voltage()
            self.hub.set_state(**fields)
            tick += 1
            time.sleep(0.2)

    def close(self) -> None:
        self.stop.set()
        for g in self.grabbers.values():
            try:
                g.close()
            except Exception:  # noqa: BLE001
                pass
        for cap in self.caps.values():
            cap.release()
        if self.sensors is not None:
            try:
                self.sensors.close()
            except Exception:  # noqa: BLE001
                pass
