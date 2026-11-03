#!/usr/bin/env python3
"""Record the GY-95T to a CSV (one row per frame) for a given number of seconds.

    python3 tools/imu_record.py 120 route_follow_debug/imu/turn_test.csv
"""
import csv
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu import Gy95t  # noqa: E402


def main() -> None:
    seconds = float(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    imu = Gy95t().start()
    t0, last, rows = time.monotonic(), None, 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
                    "roll_deg", "pitch_deg", "yaw_deg", "mag_x_G", "mag_y_G", "mag_z_G", "mag_level", "temp_c"])
        while time.monotonic() - t0 < seconds:
            s = imu.latest()
            if s is not None and s is not last:
                last = s
                w.writerow([round(s.t - t0, 4), *s.acc_g, *s.gyro_dps, *s.rpy_deg, *s.mag_gauss, s.mag_level, s.temperature_c])
                rows += 1
            time.sleep(0.004)
    imu.close()
    print(f"{rows} frames in {seconds:.0f} s, bad frames {imu.bad_frames} -> {out}")


if __name__ == "__main__":
    main()
