"""The dashboard inside a lap: route_follow.py --dashboard [PORT] starts the web server in the same process and feeds it
from the follower (its pose, trail, map of blocks, cameras, sensors and log), so nothing else has to open the cameras."""
from __future__ import annotations

import re
import sys
import threading
import time

import cv2
import numpy as np

from aruco_localizer import draw_overlay
from dashboard.hub import HUB
from dashboard.live import IR_NAMES, LiveSource, cpu_percent, usb_path
from dashboard.server import start_server


class Tee:
    """Everything the lap prints also goes to the dashboard's log."""

    def __init__(self, stream):
        self.stream = stream
        self.partial = ""

    def write(self, text):
        self.stream.write(text)
        self.partial += text
        while "\n" in self.partial:
            line, self.partial = self.partial.split("\n", 1)
            if line.strip():
                HUB.add_log(re.sub(r"\x1b\[[0-9;]*m", "", line)[:180])
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def attach(follower, port: int = 5000) -> None:
    sys.stdout = Tee(sys.stdout)
    arena, car = follower.arena, follower.car
    body = car.get("body_m", [0.275, 0.17])
    HUB.set_state(mode="run", home=[follower.home.x, follower.home.y, follower.home.yaw] if follower.home else arena.get("home"),
                  route=[list(p) for p in (follower.route or [])], body_m=body, arena_m=arena["arena_size_m"], trail=[], blocks=[])
    # the sonic reading the lap takes is remembered here (a second reader would fight it over the GPIO)
    if follower.sensors is not None:
        original = follower.sensors.read_sonic

        def read_sonic():
            value = original()
            HUB.set_state(sonic_cm=value, sonic_pan_deg=getattr(follower, "pan_deg", 0.0))
            return value
        follower.sensors.read_sonic = read_sonic
    from dashboard import imu_feed
    imu_feed.start(HUB, follower.gyro.imu if getattr(follower, "gyro", None) is not None else None)
    start_server(HUB, arena, car, port=port)
    threading.Thread(target=_feed, args=(follower,), daemon=True).start()
    print(f"dashboard: http://<the Pi's address>:{port}", flush=True)


def _feed(follower) -> None:
    caps = {}
    for name in ("left", "right"):        # the lap opens neither; their bandwidth is free
        try:
            path = usb_path(follower.args and __import__("aruco_localizer").load_json(
                __import__("pathlib").Path(__file__).resolve().parents[1] / "camera_hardware.json")[name]["usb_bus"])
            cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if cap.isOpened():
                caps[name] = cap
        except Exception:  # noqa: BLE001
            pass
    cpu_state: list = []
    tick = 0
    while not follower.stop_event.is_set():
        try:
            pose = follower.pose
            fields = {"cpu": cpu_percent(cpu_state), "temp_c": LiveSource._temp()}
            if pose is not None:
                tags = ",".join(sorted({f"{f.camera[0]}{f.tag_id}" for f in pose.fixes}))
                sigma = None
                if getattr(follower, "kalman", None) is not None and follower.kalman.ready:
                    sigma = follower.kalman.sigma()[0] * 100.0
                fields["pose"] = {"x": pose.x, "y": pose.y, "yaw_deg": pose.yaw_deg, "rms_px": None,
                                  "source": ("tags " + tags) if tags else "dead-reckon", "sigma_cm": sigma}
            if tick % 5 == 0:
                fields["trail"] = [(x, y) for x, y, _yaw in follower.trail[-400:]]
                fields["blocks"] = [b["outline"].tolist() for b in follower.obstacles.drawn(min_hits=1)]
                fields["status"] = (list(HUB.log)[-1] if HUB.log else "")[:90]
            if follower.sensors is not None:
                ir = follower.sensors.read_ir()
                fields["ir"] = {k: int(bool(ir.get(k))) for k in IR_NAMES}
            if tick % 25 == 0:
                try:
                    from safe_surround_cruise import read_battery_voltage
                    fields["battery_v"] = read_battery_voltage()
                except Exception:  # noqa: BLE001
                    pass
            HUB.set_state(**fields)
            for name, grabber in getattr(follower, "grabbers", {}).items():
                frame, _stamp = grabber.latest()
                if frame is not None and tick % 2 == 0:
                    HUB.set_frame(name, frame)
            scanner = getattr(follower, "scanner", None)
            gimbal = getattr(getattr(scanner, "feed", None), "latest", None)
            if gimbal is not None and tick % 2 == 0:
                frame, _stamp = gimbal()
                if frame is not None:
                    HUB.set_frame("gimbal", frame)
            for name, cap in caps.items():
                if tick % 3 == 0:
                    ok, frame = cap.read()
                    if ok:
                        HUB.set_frame(name, frame)
        except Exception as exc:  # noqa: BLE001 - the dashboard must never stop a lap
            HUB.add_log(f"dashboard: {exc}")
        tick += 1
        time.sleep(0.2)
    for cap in caps.values():
        cap.release()
