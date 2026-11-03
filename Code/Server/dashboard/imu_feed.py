"""Puts the GY-95T readings into the Hub for the page, and integrates the gyro so that a turn can be checked by hand."""
from __future__ import annotations

import threading
import time

from imu import Gy95t

FEED: "ImuFeed | None" = None


class ImuFeed:
    def __init__(self, hub, imu: Gy95t | None = None):
        self.hub = hub
        self.imu = imu or Gy95t().start()          # the lap's own reader when there is one: the port has one reader
        self.stop = threading.Event()
        self.gyro_yaw = 0.0           # deg, gyro z integrated since the last zero (counter-clockwise seen from above = +)
        self.last_t: float | None = None
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True, name="imu-feed")
        self.thread.start()

    def zero(self) -> None:
        self.gyro_yaw = 0.0

    def close(self) -> None:
        self.stop.set()

    def _run(self) -> None:
        last_frames, last_check = 0, time.monotonic()
        rate = 0.0
        while not self.stop.is_set():
            s = self.imu.latest()
            now = time.monotonic()
            if s is not None and s.t != self.last_t:
                if self.last_t is not None:
                    self.gyro_yaw += s.gyro_dps[2] * min(s.t - self.last_t, 0.2)
                self.last_t = s.t
            if now - last_check >= 1.0:
                rate = (self.imu.frames - last_frames) / (now - last_check)
                last_frames, last_check = self.imu.frames, now
            if s is None:
                self.hub.set_state(imu=None)
            else:
                self.hub.set_state(imu=dict(
                    acc_g=s.acc_g, gyro_dps=s.gyro_dps, rpy_deg=s.rpy_deg, temperature_c=s.temperature_c,
                    mag_gauss=s.mag_gauss, mag_level=s.mag_level, age_s=round(now - s.t, 2), rate_hz=round(rate),
                    bad_frames=self.imu.bad_frames, gyro_yaw_deg=round(self.gyro_yaw, 1)))
            time.sleep(0.1)


def start(hub, imu: Gy95t | None = None) -> ImuFeed:
    global FEED
    FEED = ImuFeed(hub, imu)
    return FEED
