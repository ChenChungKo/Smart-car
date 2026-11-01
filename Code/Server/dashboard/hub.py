"""The one place the dashboard reads from: the latest state, the latest frame of every camera and a short log.

Whoever owns the hardware (dashboard/live.py, or a running route_follow) pushes into it; the web server only reads.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np


class Hub:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: dict = {}
        self.frames: dict[str, tuple[np.ndarray, float]] = {}
        self.log: deque[str] = deque(maxlen=80)
        self._jpeg: dict[str, tuple[float, bytes]] = {}

    def set_state(self, **fields) -> None:
        with self.lock:
            self.state.update(fields)
            self.state["updated"] = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            out = dict(self.state)
            out["log"] = list(self.log)
            out["frame_age_s"] = {n: round(time.time() - t, 2) for n, (_f, t) in self.frames.items()}
        return out

    def set_frame(self, name: str, bgr: np.ndarray, width: int = 640) -> None:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA) if w > width else bgr.copy()
        with self.lock:
            self.frames[name] = (small, time.time())

    def frame(self, name: str):
        with self.lock:
            return self.frames.get(name)

    def add_log(self, line: str) -> None:
        with self.lock:
            self.log.append(line)

    def jpeg(self, name: str, image: np.ndarray, stamp: float, quality: int = 70) -> bytes:
        """Encode once per frame, whoever asks."""
        with self.lock:
            cached = self._jpeg.get(name)
            if cached and cached[0] == stamp:
                return cached[1]
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        data = buf.tobytes() if ok else b""
        with self.lock:
            self._jpeg[name] = (stamp, data)
        return data


HUB = Hub()
