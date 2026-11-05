"""GY-95T nine-axis module on a CH340 USB serial port (115200 baud, it streams by itself, about 54 frames/s).

Frame, 32 bytes:  A4 03 08 1B | 27 data bytes (registers 0x08..0x22, little endian) | checksum
(checksum = low 8 bits of the sum of the 31 bytes before it). The data bytes, in register order:

    acc x, y, z      int16 x3   raw / 2048 = g
    gyro x, y, z     int16 x3   raw / 16.4 = deg/s
    roll, pitch, yaw int16 x3   raw / 100 = deg      (the module's own fusion)
    level            uint8      magnetic calibration quality
    temperature      int16      raw / 100 = deg C
    mag x, y, z      int16 x3   raw / 1000 = gauss

Layout and scales: https://birdzhang.xyz/2025/10/31/ (no manufacturer document); confirmed here by the frame length
(27 bytes = exactly these fields) and the values at rest (z = 1 g, gyro near 0, 26.7 deg C).
"""
from __future__ import annotations

import glob
import struct
import threading
import time
from dataclasses import dataclass

import serial

HEADER = bytes([0xA4, 0x03, 0x08, 0x1B])
FRAME_LEN = 32
G = 9.80665
DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"


@dataclass(frozen=True)
class ImuSample:
    t: float                  # time.monotonic() when the frame was complete
    acc_g: tuple              # x, y, z in g (the car at rest: z = +1)
    gyro_dps: tuple           # x, y, z in deg/s
    rpy_deg: tuple            # roll, pitch, yaw from the module's own fusion
    temperature_c: float
    mag_gauss: tuple
    mag_level: int


def parse(frame: bytes) -> ImuSample | None:
    """One 32-byte frame -> sample, or None when the header or the checksum is wrong."""
    if len(frame) != FRAME_LEN or frame[:4] != HEADER or sum(frame[:31]) & 0xFF != frame[31]:
        return None
    ax, ay, az, gx, gy, gz, roll, pitch, yaw, level, temp, mx, my, mz = struct.unpack("<9hBh3h", frame[4:31])
    return ImuSample(t=time.monotonic(), acc_g=(ax / 2048, ay / 2048, az / 2048),
                     gyro_dps=(gx / 16.4, gy / 16.4, gz / 16.4), rpy_deg=(roll / 100, pitch / 100, yaw / 100),
                     temperature_c=temp / 100, mag_gauss=(mx / 1000, my / 1000, mz / 1000), mag_level=level)


class Gy95t:
    """Reads the stream on a thread; latest() is the newest good sample. Bad frames are counted, never used."""

    def __init__(self, port: str | None = None, baud: int = 115200):
        self.port = port or (DEFAULT_PORT if glob.glob(DEFAULT_PORT) else "/dev/ttyUSB0")
        self.baud = baud
        self._lock = threading.Lock()
        self._sample: ImuSample | None = None
        self.frames = 0
        self.bad_frames = 0
        self._raw_yaw_deg = 0.0       # integral of the gyro z since start, no bias removed
        self._raw_yaw_t: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gy95t")

    def start(self) -> "Gy95t":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest(self) -> ImuSample | None:
        with self._lock:
            return self._sample

    def yaw_integral(self) -> tuple[float, float | None]:
        """(deg, time.monotonic() of the last frame in it): the gyro z summed over every frame since start."""
        with self._lock:
            return self._raw_yaw_deg, self._raw_yaw_t

    def _run(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            try:
                with serial.Serial(self.port, self.baud, timeout=0.1) as port:
                    buf.clear()
                    while not self._stop.is_set():
                        buf += port.read(max(1, port.in_waiting))   # whatever has arrived, so a frame is stamped soon after its last byte
                        while True:
                            i = buf.find(HEADER)
                            if i < 0:
                                del buf[:-3]                      # keep a possible start of the next header
                                break
                            if len(buf) - i < FRAME_LEN:
                                del buf[:i]
                                break
                            sample = parse(bytes(buf[i:i + FRAME_LEN]))
                            if sample is None:
                                self.bad_frames += 1
                                del buf[:i + 1]
                                continue
                            del buf[:i + FRAME_LEN]
                            self.frames += 1
                            with self._lock:
                                if self._raw_yaw_t is not None:
                                    self._raw_yaw_deg += sample.gyro_dps[2] * min(sample.t - self._raw_yaw_t, 0.1)
                                self._raw_yaw_t = sample.t
                                self._sample = sample
            except (serial.SerialException, OSError):
                time.sleep(1.0)                                   # unplugged: try again


class YawGyro:
    """Heading change of the car from the z gyro (z up, counter-clockwise +, the same as the map's yaw).

    take() is the angle turned since the last take() or mark(), minus the bias, times the scale; the pose code calls
    take() after every move and mark() whenever the pose is replaced by a measurement, so the gyro only ever adds the
    turn the measurement has not seen yet.
    """
    STALE_S = 0.5

    def __init__(self, imu: Gy95t, scale: float = 1.0, bias_dps: float = 0.0):
        self.imu = imu
        self.scale = scale
        self.bias_dps = bias_dps
        self._deg, self._t = imu.yaw_integral()

    def calibrate(self, seconds: float = 2.0) -> tuple[float, float]:
        """Bias of the gyro z (deg/s, mean) and its noise (std) from `seconds` of standing still; sets the bias."""
        values, last, end = [], None, time.monotonic() + seconds
        while time.monotonic() < end:
            s = self.imu.latest()
            if s is not None and s is not last:
                values.append(s.gyro_dps[2])
                last = s
            time.sleep(0.005)
        if len(values) < 20:
            return self.bias_dps, float("inf")
        mean, std = sum(values) / len(values), (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5
        self.bias_dps = mean
        self.mark()
        return mean, std

    def fresh(self) -> bool:
        _deg, t = self.imu.yaw_integral()
        return t is not None and time.monotonic() - t < self.STALE_S

    def mark(self) -> None:
        self._deg, self._t = self.imu.yaw_integral()

    def peek(self) -> float | None:
        """Radians turned since the last take()/mark(), without moving the mark (for control during a move)."""
        if not self.fresh():
            return None
        deg, t = self.imu.yaw_integral()
        seconds = 0.0 if self._t is None else t - self._t
        return ((deg - self._deg) - self.bias_dps * seconds) * self.scale * 3.141592653589793 / 180.0

    def take(self) -> tuple[float, float] | None:
        """(radians turned, seconds it covers) since the last take()/mark(); None when the gyro is not sending."""
        if not self.fresh():
            return None
        deg, t = self.imu.yaw_integral()
        seconds = 0.0 if self._t is None else t - self._t
        turned = ((deg - self._deg) - self.bias_dps * seconds) * self.scale
        self._deg, self._t = deg, t
        return turned * 3.141592653589793 / 180.0, seconds

    @staticmethod
    def sigma(radians: float, seconds: float) -> float:
        """1 sigma of that angle: the scale is good to ~1.5 %, the bias to ~0.03 deg/s, plus a floor."""
        return ((0.015 * radians) ** 2 + (0.03 * 3.141592653589793 / 180.0 * seconds) ** 2 + (0.002) ** 2) ** 0.5
