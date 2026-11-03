#!/usr/bin/env python3
"""Read the GY-95T for a few seconds and print what it says (the car should stand still on a flat floor).

    python3 tools/imu_check.py [seconds]

Prints the rate, bad frames, and per field the mean and the noise (std). At rest: acc about (0, 0, 1) g, gyro near 0
(its mean is the bias), and the module's roll/pitch near 0. Turn the car by hand during the run to see yaw and gyro z move.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu import Gy95t  # noqa: E402


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    imu = Gy95t().start()
    rows, last = [], None
    end = time.time() + seconds
    while time.time() < end:
        s = imu.latest()
        if s is not None and s is not last:
            rows.append([*s.acc_g, *s.gyro_dps, *s.rpy_deg, s.temperature_c, *s.mag_gauss])
            last = s
        time.sleep(0.005)
    imu.close()
    if not rows:
        print("no frames: is the IMU plugged in (ls /dev/ttyUSB0) and nothing else using the port?")
        return
    a = np.array(rows)
    print(f"{len(a)} frames in {seconds:.0f} s = {len(a) / seconds:.0f}/s, bad frames {imu.bad_frames}")
    names = ["acc x g", "acc y g", "acc z g", "gyro x dps", "gyro y dps", "gyro z dps", "roll deg", "pitch deg", "yaw deg",
             "temp C", "mag x G", "mag y G", "mag z G"]
    for k, n in enumerate(names):
        print(f"  {n:11s} mean {a[:, k].mean():9.3f}   std {a[:, k].std():7.3f}   range {a[:, k].min():8.3f} .. {a[:, k].max():8.3f}")
    print(f"  |acc| = {np.linalg.norm(a[:, 0:3].mean(0)):.3f} g   |mag| = {np.linalg.norm(a[:, 10:13].mean(0)):.2f} G "
          f"(the Earth's field is 0.25-0.65 G)")


if __name__ == "__main__":
    main()
