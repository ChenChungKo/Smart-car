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
                                self._sample = sample
            except (serial.SerialException, OSError):
                time.sleep(1.0)                                   # unplugged: try again
