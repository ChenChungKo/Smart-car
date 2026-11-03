#!/usr/bin/env python3
"""The GY-95T values live in the terminal, refreshed in place.

    python3 tools/imu_live.py

Enter = zero the integrated gyro z (turn the car by hand and read how far it says it turned; counter-clockwise is +).
Ctrl+C = quit. The port has one reader at a time: close the dashboard and imu_check first.
"""
import math
import pathlib
import select
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu import Gy95t  # noqa: E402


def main() -> None:
    imu = Gy95t().start()
    gyro_yaw, last_t, last_frames, last_check, rate = 0.0, None, 0, time.monotonic(), 0.0
    clear = "\x1b[2J\x1b[H" if sys.stdout.isatty() else ""
    try:
        while True:
            s = imu.latest()
            now = time.monotonic()
            if s is not None and s.t != last_t:
                if last_t is not None:
                    gyro_yaw += s.gyro_dps[2] * min(s.t - last_t, 0.2)
                last_t = s.t
            if now - last_check >= 1.0:
                rate, last_frames, last_check = (imu.frames - last_frames) / (now - last_check), imu.frames, now
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                gyro_yaw = 0.0
            if s is None:
                out = "no data from the IMU (is /dev/ttyUSB0 there? is another program reading it?)"
            else:
                mag = math.hypot(*s.mag_gauss)
                out = (f"GY-95T   {rate:3.0f} frames/s   bad {imu.bad_frames}   age {now - s.t:.2f} s\n\n"
                       f"acc   x {s.acc_g[0]:+8.3f}   y {s.acc_g[1]:+8.3f}   z {s.acc_g[2]:+8.3f}   g\n"
                       f"gyro  x {s.gyro_dps[0]:+8.2f}   y {s.gyro_dps[1]:+8.2f}   z {s.gyro_dps[2]:+8.2f}   deg/s\n"
                       f"module roll {s.rpy_deg[0]:+8.2f}   pitch {s.rpy_deg[1]:+8.2f}   yaw {s.rpy_deg[2]:+8.2f}   deg\n"
                       f"mag   x {s.mag_gauss[0]:+8.3f}   y {s.mag_gauss[1]:+8.3f}   z {s.mag_gauss[2]:+8.3f}   gauss"
                       f"   |{mag:.2f}|  level {s.mag_level}\n"
                       f"temperature {s.temperature_c:.1f} C\n\n"
                       f"gyro z integrated: {gyro_yaw:+9.1f} deg   (Enter = zero, Ctrl+C = quit)")
            sys.stdout.write(clear + out + "\n")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        imu.close()


if __name__ == "__main__":
    main()
