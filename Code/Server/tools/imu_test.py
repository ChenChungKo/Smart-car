#!/usr/bin/env python3
"""Tests of the parts of the GY-95T, one at a time. Results in the terminal; raw frames to route_follow_debug/imu/.

    python3 tools/imu_test.py still          # nothing moves: noise, bias and level of all 9 axes (10 s)
    python3 tools/imu_test.py still 60       # the same for 60 s (shows the drift of the gyro bias)
    python3 tools/imu_test.py tilt           # nothing moves: you tilt the car (a wheel on a block); Enter = one reading
    python3 tools/imu_test.py spin           # the car TURNS in place: is the magnetometer a compass here?
    python3 tools/imu_motion_check.py        # the car drives: the distance from the accelerometer (its own tool)

still  The car stands still on the floor. Per axis the mean and the noise; |acc| must be 1.000 g (the accelerometer's
       scale); the tilt from gravity next to the module's own roll/pitch; the gyro bias and how far the heading
       drifts when the bias is measured for 2 s at the start, as route_follow does; the strength of the magnetic field
       (the earth's is about 0.25-0.65 gauss: much more or a changing value is the car's own iron and currents).
tilt   Enter takes a 1 s reading: nose up and left side up from gravity, and the module's roll/pitch. Put the car flat,
       then a wheel or two on a block, to see how much a wheel on a block tilts it. q + Enter = quit.
spin   Put the car in the middle of the arena, 30 cm or more free around it, tags in view. It turns 8 x 45 deg left,
       then 8 x 45 deg right (gyro turns), standing still 1 s after each. From the still readings (motors off) the
       magnetometer is calibrated (hard and soft iron, an ellipse); then its heading is compared with the gyro's, at
       the stops and while turning (motors on). The tags before and after check the gyro.
Axes of the module on the car: X = rear, Y = right, Z = up.
"""
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
from imu import Gy95t  # noqa: E402

OUT = SERVER / "route_follow_debug" / "imu"
FIELDS = ["t", "acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps", "roll_deg", "pitch_deg",
          "yaw_deg", "mag_x_G", "mag_y_G", "mag_z_G", "mag_level", "temp_c"]
T, AX, AY, AZ, GX, GY, GZ, ROLL, PITCH, YAW, MX, MY, MZ, LEVEL, TEMP = range(len(FIELDS))


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Recorder:
    """Every IMU frame, all fields, collected on a thread; mark() notes an event with its time."""

    def __init__(self, imu: Gy95t):
        self.imu = imu
        self.rows: list[tuple] = []
        self.marks: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            s = self.imu.latest()
            if s is not None and s is not last:
                last = s
                self.rows.append((s.t, *s.acc_g, *s.gyro_dps, *s.rpy_deg, *s.mag_gauss, s.mag_level, s.temperature_c))
            time.sleep(0.004)

    def mark(self, label: str) -> float:
        t = time.monotonic()
        self.marks.append((t, label))
        return t

    def window(self, t0: float, t1: float) -> np.ndarray:
        return np.array([r for r in self.rows if t0 <= r[T] <= t1], float).reshape(-1, len(FIELDS))

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def save(self, name: str, summary: dict) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        t0 = self.rows[0][T] if self.rows else 0.0
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows([round(r[T] - t0, 4), *r[1:]] for r in self.rows)
        summary["marks"] = [(round(t - t0, 3), label) for t, label in self.marks]
        (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"\nsaved {OUT}/{name}.csv and .json", flush=True)


def tilt_deg(acc: np.ndarray) -> tuple[float, float]:
    """(nose up, left side up) in deg from gravity; the accelerometer reads +1 g up at rest, X = rear, Y = right."""
    norm = float(np.linalg.norm(acc))
    return math.degrees(math.asin(max(-1.0, min(1.0, -acc[0] / norm)))), \
        math.degrees(math.asin(max(-1.0, min(1.0, -acc[1] / norm))))


# ---- still ------------------------------------------------------------------

def still(seconds: float) -> None:
    imu = Gy95t().start()
    time.sleep(0.5)
    rec = Recorder(imu)
    print(f"keep the car still for {seconds:.0f} s ...", flush=True)
    time.sleep(seconds)
    rec.stop()
    imu.close()
    a = np.array(rec.rows, float)
    if len(a) < 50:
        print(f"only {len(a)} frames: is the IMU plugged in and no other program reading it?")
        return
    t = a[:, T]
    span = t[-1] - t[0]
    gaps = np.diff(t)
    print(f"\n{len(a)} frames in {span:.1f} s = {len(a) / span:.1f}/s, bad frames {imu.bad_frames}, "
          f"longest gap {gaps.max() * 1000:.0f} ms")
    summary = dict(seconds=span, frames=len(a), bad_frames=imu.bad_frames, longest_gap_ms=float(gaps.max() * 1000))
    print("\n            mean        noise (std)")
    for name, col, unit in (("acc x", AX, "g"), ("acc y", AY, "g"), ("acc z", AZ, "g"),
                            ("gyro x", GX, "deg/s"), ("gyro y", GY, "deg/s"), ("gyro z", GZ, "deg/s"),
                            ("mag x", MX, "gauss"), ("mag y", MY, "gauss"), ("mag z", MZ, "gauss")):
        mean, std = float(a[:, col].mean()), float(a[:, col].std())
        summary[name.replace(" ", "_")] = dict(mean=mean, std=std)
        print(f"  {name:7s} {mean:+10.4f}  {std:9.4f}  {unit}")
    acc = a[:, AX:AZ + 1].mean(axis=0)
    g = float(np.linalg.norm(acc))
    nose, left = tilt_deg(acc)
    print(f"\naccelerometer: |acc| = {g:.4f} g (1.0000 = scale right; {100 * (g - 1):+.2f} %)")
    print(f"  tilt from gravity: nose up {nose:+.2f} deg, left side up {left:+.2f} deg   "
          f"(module: roll {a[:, ROLL].mean():+.2f}, pitch {a[:, PITCH].mean():+.2f})")
    # the accelerometer noise turned into a distance: what a still car "travels" in 1 s by integrating twice
    acc_noise = float(np.hypot(a[:, AX].std(), a[:, AY].std())) * 9.80665
    print(f"  noise {acc_noise * 1000:.0f} mm/s^2 on the floor plane")
    summary.update(acc_norm_g=g, nose_up_deg=nose, left_up_deg=left)
    first = a[t - t[0] < 2.0]
    rest = a[t - t[0] >= 2.0]
    bias = float(first[:, GZ].mean())
    drift = float(np.sum((rest[1:, GZ] - bias) * np.diff(rest[:, T]))) if len(rest) > 2 else float("nan")
    rest_s = float(rest[-1, T] - rest[0, T]) if len(rest) > 2 else float("nan")
    print(f"\ngyro z: bias {a[:, GZ].mean():+.4f} deg/s = {a[:, GZ].mean() * 60:+.1f} deg/min if it were not removed")
    print(f"  bias from the first 2 s (as route_follow): {bias:+.4f} deg/s; the heading then drifts {drift:+.2f} deg "
          f"over the next {rest_s:.0f} s ({drift / rest_s * 60:+.2f} deg/min)")
    summary.update(gyro_z_bias_2s=bias, heading_drift_deg=drift, drift_over_s=rest_s)
    mag = np.linalg.norm(a[:, MX:MZ + 1], axis=1)
    print(f"\nmagnetometer: |field| {mag.mean():.3f} gauss (noise {mag.std():.4f}), level {int(st.median(a[:, LEVEL]))}"
          f"   (the earth's field is about 0.25-0.65)")
    print(f"temperature {a[:, TEMP].mean():.2f} C (from {a[0, TEMP]:.2f} to {a[-1, TEMP]:.2f})")
    summary.update(mag_norm_G=float(mag.mean()), temperature_c=float(a[:, TEMP].mean()))
    rec.save(f"still_{time.strftime('%Y%m%d-%H%M%S')}", summary)


# ---- tilt -------------------------------------------------------------------

def tilt() -> None:
    import select
    imu = Gy95t().start()
    time.sleep(0.5)
    rec = Recorder(imu)
    readings = []
    print("put the car flat and press Enter; then tilt it (a wheel on a block) and press Enter again; q + Enter = quit",
          flush=True)
    try:
        while True:
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            if sys.stdin.readline().strip().lower() == "q":
                break
            t0 = rec.mark(f"reading {len(readings) + 1}")
            time.sleep(1.0)
            w = rec.window(t0, t0 + 1.0)
            if len(w) < 10:
                print("no data")
                continue
            nose, left = tilt_deg(w[:, AX:AZ + 1].mean(axis=0))
            readings.append(dict(nose_up_deg=nose, left_up_deg=left, roll=float(w[:, ROLL].mean()),
                                 pitch=float(w[:, PITCH].mean())))
            rel = "" if len(readings) < 2 else \
                f"   vs the first: nose {nose - readings[0]['nose_up_deg']:+.1f}, left {left - readings[0]['left_up_deg']:+.1f}"
            print(f"#{len(readings)}  nose up {nose:+6.2f} deg  left side up {left:+6.2f} deg   "
                  f"(module roll {readings[-1]['roll']:+6.2f}, pitch {readings[-1]['pitch']:+6.2f}){rel}", flush=True)
    except KeyboardInterrupt:
        pass
    rec.stop()
    imu.close()
    if readings:
        rec.save(f"tilt_{time.strftime('%Y%m%d-%H%M%S')}", dict(readings=readings))


# ---- spin -------------------------------------------------------------------

def fit_ellipse(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned ellipse A x^2 + B y^2 + C x + D y = 1 through the points: (centre x, centre y, radius x, radius y)."""
    m = np.column_stack([x * x, y * y, x, y])
    a, b, c, d = np.linalg.lstsq(m, np.ones(len(x)), rcond=None)[0]
    cx, cy = -c / (2 * a), -d / (2 * b)
    k = 1 + a * cx * cx + b * cy * cy
    return float(cx), float(cy), float(math.sqrt(k / a)), float(math.sqrt(k / b))


def spin() -> None:
    from calibrate_motion import wall_cm
    from route_follow import Abort, RouteFollower, parse_args

    car = RouteFollower(parse_args(["--arm", "--no-window", "--debug-dir", "route_follow_debug/calibrate"]))
    rec = None
    stops: list[tuple[float, float]] = []      # still windows
    try:
        car.start()
        if car.gyro is None:
            raise Abort("no IMU: plug it in (ls /dev/ttyUSB0) and keep the car still while the program starts")
        rec = Recorder(car.gyro.imu)
        room = {pan: wall_cm(car, pan) for pan in (0.0, 90.0, -90.0)}
        car.look(0.0)
        print(f"placement: wall ahead {room[0.0]:.0f} cm, left {room[90.0]:.0f} cm, right {room[-90.0]:.0f} cm "
              f"(needs 30 or more each, and behind too)", flush=True)
        if min(room.values()) < 30:
            raise Abort("move the car toward the middle: too close to a wall to turn in place (nothing has moved yet)")
        before = car.localize(3.0)
        if before is not None:
            car.pose = before
        step = math.radians(45.0)

        def still_window() -> None:
            time.sleep(0.5)
            t0 = rec.mark("still")
            time.sleep(1.0)
            stops.append((t0, time.monotonic()))

        still_window()
        for kind, sign in (("left", 1.0), ("right", -1.0)):
            for _ in range(8):
                rec.mark(kind)
                car.move(kind, math.degrees(step) / car.turn_rate(kind), turn_rad=sign * step)
                still_window()
        after = car.localize(3.0)
    except Abort as exc:
        print(f"ABORT: {exc}")
        return
    finally:
        car.close()
        if rec is not None:
            rec.stop()
    a = np.array(rec.rows, float)
    t = a[:, T]
    bias, scale = car.gyro.bias_dps, car.gyro.scale
    gyro_head = np.concatenate([[0.0], np.cumsum((a[1:, GZ] - bias) * scale * np.minimum(np.diff(t), 0.1))])

    def at(times: tuple[float, float], column: int | None = None):
        sel = (t >= times[0]) & (t <= times[1])
        return gyro_head[sel].mean() if column is None else a[sel, column].mean()

    still_mag = np.array([[at(w, MX), at(w, MY), at(w, MZ)] for w in stops])
    still_gyro = np.array([at(w) for w in stops])
    cx, cy, rx, ry = fit_ellipse(still_mag[:, 0], still_mag[:, 1])

    def mag_angle(mx, my):
        """The field's direction in the car's plane, calibrated (deg); it turns the other way to the car."""
        return np.degrees(np.arctan2((my - cy) / ry, (mx - cx) / rx))

    phi0 = mag_angle(still_mag[0, 0], still_mag[0, 1])
    still_head = np.array([wrap_deg(-(mag_angle(m[0], m[1]) - phi0)) for m in still_mag])
    still_err = np.array([wrap_deg(h - g) for h, g in zip(still_head, still_gyro)])
    print(f"\nmagnetometer calibration from {len(stops)} still readings: centre ({cx:+.3f}, {cy:+.3f}) gauss, "
          f"radii {rx:.3f} / {ry:.3f} (hard iron = the centre, soft iron = the radii differing)")
    print("\n stop   gyro heading   compass heading   compass - gyro")
    for i, (g, h, e) in enumerate(zip(still_gyro, still_head, still_err)):
        print(f"  {i:2d}     {wrap_deg(g):+7.1f}          {h:+7.1f}           {e:+6.1f}")
    print(f"motors off: compass - gyro  rms {math.sqrt(float(np.mean(still_err ** 2))):.1f} deg, "
          f"worst {float(np.max(np.abs(still_err))):.1f} deg")
    moving = np.ones(len(t), bool)
    for w in stops:
        moving &= ~((t >= w[0] - 0.5) & (t <= w[1]))
    moving &= t > stops[0][0]
    radius_still = np.hypot((still_mag[:, 0] - cx) / rx, (still_mag[:, 1] - cy) / ry)
    summary = dict(ellipse=dict(cx=cx, cy=cy, rx=rx, ry=ry), still=dict(gyro=still_gyro.tolist(),
                   compass=still_head.tolist(), error=still_err.tolist()))
    if moving.sum() > 20:
        mm = a[moving]
        head = -(mag_angle(mm[:, MX], mm[:, MY]) - phi0)
        err = np.array([wrap_deg(h - g) for h, g in zip(head, gyro_head[moving])])
        radius = np.hypot((mm[:, MX] - cx) / rx, (mm[:, MY] - cy) / ry)
        print(f"motors on (turning, {moving.sum()} frames): compass - gyro  rms {math.sqrt(float(np.mean(err ** 2))):.1f} "
              f"deg, worst {float(np.max(np.abs(err))):.1f} deg")
        print(f"  field size (1.00 = as calibrated): still {radius_still.min():.2f}-{radius_still.max():.2f}, "
              f"turning {np.percentile(radius, 5):.2f}-{np.percentile(radius, 95):.2f} (5-95 %): "
              f"a wider range while turning is the motors' own field")
        summary["moving"] = dict(rms=float(math.sqrt(np.mean(err ** 2))), worst=float(np.max(np.abs(err))),
                                 radius_p5=float(np.percentile(radius, 5)), radius_p95=float(np.percentile(radius, 95)))
    net_gyro = wrap_deg(float(still_gyro[-1] - still_gyro[0]))
    print(f"\ngyro: 360 deg left then 360 deg right, net {net_gyro:+.1f} deg", end="")
    if before is not None and after is not None:
        net_tags = wrap_deg(math.degrees(after.yaw - before.yaw))
        print(f"; tags net {net_tags:+.1f} deg: gyro - tags {wrap_deg(net_gyro - net_tags):+.1f} deg")
        summary.update(net_gyro=net_gyro, net_tags=net_tags)
    else:
        print("; no tag fix before or after, so no check against the tags")
    rec.save(f"spin_{time.strftime('%Y%m%d-%H%M%S')}", summary)


def main() -> None:
    test = sys.argv[1] if len(sys.argv) > 1 else ""
    if test == "still":
        still(float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
    elif test == "tilt":
        tilt()
    elif test == "spin":
        spin()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
