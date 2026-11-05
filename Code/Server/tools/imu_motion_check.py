#!/usr/bin/env python3
"""Can the accelerometer tell how far a move went? Moves the car and compares three numbers per move.

    python3 tools/imu_motion_check.py            # 3 trials of forward, reverse, strafe left, strafe right
    python3 tools/imu_motion_check.py --trials 5

Put the car as for tools/calibrate_motion.py: in the middle of the arena, facing a wall squarely, 80 cm or more of free
floor ahead and 40 cm or more to each side (the ultrasonic is the ruler). The car MOVES (the motors are armed).

Per move: stand still 1 s, move 1.5 s, stand still 1 s. The three distances along the move:
  ultrasonic  the change of the distance to the wall it moves along or across (good to about 1 cm)
  speed model the calibrated speed times the time (what route_follow uses now)
  imu         the accelerometer, bias taken from the still second before, integrated twice; the speed is forced to 0 at
              the end of the move (the car is standing still again), which removes the part of the bias error that grows
              like a ramp. Car axes from the IMU: forward = -X, left = -Y.
Every IMU field of every frame and the windows go to route_follow_debug/imu/motion_<time>.csv/.json for offline work.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics as st
import sys
import threading
import time

import numpy as np

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from calibrate_motion import square_up, wall_cm  # noqa: E402
from route_follow import Abort, RouteFollower, parse_args  # noqa: E402

G = 9.80665
MOVES = (("forward", 0.0, "x"), ("reverse", 0.0, "x"), ("strafe_left", 90.0, "y"), ("strafe_right", -90.0, "y"))
# kind, pan of the ultrasonic (the wall it measures against: ahead, left, right), car axis; distances: left and forward +


class Recorder:
    """Every IMU frame with its time, collected on a thread."""

    def __init__(self, imu):
        self.imu = imu
        self.rows: list[tuple] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        last = None
        while not self.stop.is_set():
            s = self.imu.latest()
            if s is not None and s is not last:
                last = s
                # car frame: forward = -X_imu, left = -Y_imu; acceleration in m/s^2; then every raw field
                self.rows.append((s.t, -s.acc_g[0] * G, -s.acc_g[1] * G, s.acc_g[2] * G, s.gyro_dps[2],
                                  *s.acc_g, *s.gyro_dps, *s.rpy_deg, *s.mag_gauss, s.mag_level, s.temperature_c))
            time.sleep(0.004)

    def window(self, t0: float, t1: float) -> np.ndarray:
        return np.array([r for r in self.rows if t0 <= r[0] <= t1], float)


def imu_distance(rec: Recorder, still0: tuple, moving: tuple, still1: tuple, axis: str) -> float | None:
    """Distance (m, signed, along the car axis) of the move from the accelerometer."""
    column = 1 if axis == "x" else 2
    before, run, after = (rec.window(*w) for w in (still0, moving, still1))
    if len(before) < 20 or len(run) < 20 or len(after) < 20:
        return None
    bias = before[:, column].mean()
    # from the end of the still window before to the start of the still window after: the whole move with its coasting
    seg = rec.window(still0[1] - 0.15, still1[0] + 0.25)
    t = seg[:, 0]
    a = seg[:, column] - bias
    dt = np.diff(t, prepend=t[0])
    v = np.cumsum(a * dt)
    ramp = np.linspace(0.0, v[-1], len(v))      # standing still again: the speed must be 0; take the straight drift away
    v -= ramp
    return float(np.sum(v * dt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    opts = ap.parse_args()
    car = RouteFollower(parse_args(["--arm", "--no-window", "--debug-dir", "route_follow_debug/calibrate"]))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = SERVER / "route_follow_debug" / "imu"
    out.mkdir(parents=True, exist_ok=True)
    try:
        car.start()
        if car.gyro is None:
            raise Abort("no IMU: plug it in (ls /dev/ttyUSB0) and keep the car still while the program starts")
        rec = Recorder(car.gyro.imu)
        ahead, to_left, to_right = wall_cm(car, 0.0), wall_cm(car, 90.0), wall_cm(car, -90.0)
        car.look(0.0)
        print(f"placement: wall ahead {ahead:.0f} cm, left {to_left:.0f} cm, right {to_right:.0f} cm "
              f"(needs ahead >= 80, each side >= 40)", flush=True)
        if ahead < 80 or min(to_left, to_right) < 40:
            raise Abort("move the car: too close to a wall for the trials (nothing has moved yet)")
        results = []
        for trial in range(1, opts.trials + 1):
            for kind, pan, axis in MOVES:
                time.sleep(0.3)
                d0 = wall_cm(car, pan)
                t_a = time.monotonic()
                time.sleep(1.0)
                t_b = time.monotonic()
                t_move0 = time.monotonic()
                elapsed, _ = car.move(kind, 1.5)
                t_move1 = time.monotonic()
                t_c = time.monotonic()
                time.sleep(1.0)
                t_d = time.monotonic()
                d1 = wall_cm(car, pan)
                speed = float(car.motion["forward_mps"] if axis == "x" else car.strafe_mps)
                direction = {"forward": 1, "reverse": -1, "strafe_left": 1, "strafe_right": -1}[kind]
                model = direction * speed * elapsed
                ultra = (d0 - d1) / 100.0 * (-1.0 if kind == "strafe_right" else 1.0)   # wall closer = moved toward it
                imu = imu_distance(rec, (t_a, t_b), (t_move0, t_move1), (t_c, t_d), axis)
                results.append(dict(trial=trial, kind=kind, ultrasonic_cm=ultra * 100, model_cm=model * 100,
                                    imu_cm=None if imu is None else imu * 100,
                                    windows=[t_a, t_b, t_move0, t_move1, t_c, t_d]))
                print(f"trial {trial} {kind:13s} ultrasonic {ultra * 100:+6.1f} cm   speed model {model * 100:+6.1f} cm   "
                      f"imu {'-' if imu is None else f'{imu * 100:+6.1f}'} cm", flush=True)
                if kind in ("strafe_left", "strafe_right"):
                    square_up(car)
        print("\nmedian absolute error against the ultrasonic (cm), per kind of move:")
        for kind, _pan, _axis in MOVES:
            rows = [r for r in results if r["kind"] == kind and r["imu_cm"] is not None]
            if rows:
                e_model = st.median(abs(r["model_cm"] - r["ultrasonic_cm"]) for r in rows)
                e_imu = st.median(abs(r["imu_cm"] - r["ultrasonic_cm"]) for r in rows)
                print(f"  {kind:13s} speed model {e_model:5.1f}   imu {e_imu:5.1f}   (move of about "
                      f"{st.median(abs(r['ultrasonic_cm']) for r in rows):.0f} cm, {len(rows)} trials)")
        with open(out / f"motion_{stamp}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_monotonic", "car_x_acc_ms2", "car_y_acc_ms2", "acc_z_ms2", "gyro_z_dps",
                        "acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps_raw", "roll_deg",
                        "pitch_deg", "yaw_deg", "mag_x_G", "mag_y_G", "mag_z_G", "mag_level", "temp_c"])
            w.writerows(rec.rows)
        (out / f"motion_{stamp}.json").write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nsaved {out}/motion_{stamp}.csv and .json", flush=True)
    except Abort as exc:
        print(f"ABORT: {exc}")
    finally:
        car.close()


if __name__ == "__main__":
    main()
