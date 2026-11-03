#!/usr/bin/env python3
"""The GY-95T values live in the terminal, refreshed in place.

    python3 tools/imu_live.py             # raw values; Enter zeroes the integrated gyro z and the shown module yaw
    python3 tools/imu_live.py --align     # the car's heading on the arena map (0 = toward the right wall, CCW +)

--align: the car must stand still, in the arena, with arena tags in view of the front or rear camera. The program takes
a tag fix of the heading, measures the gyro bias, and from then on shows heading = that fix + the gyro's turning. Enter
takes a new tag fix and prints how far the gyro heading had drifted from it (that is the check of the gyro against the tags).
Nothing is written to the module; the module's own yaw is only shown, shifted to the heading at alignment.
Ctrl+C = quit. The IMU port and the cameras have one owner at a time: close the dashboard and other programs first.
"""
import argparse
import json
import math
import pathlib
import select
import statistics as st
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from imu import Gy95t, YawGyro  # noqa: E402


class TagHeading:
    """The car's heading on the map from the arena tags, car standing still."""

    def __init__(self):
        from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, load_json, open_grabbers
        self.car = load_json(CAR_PATH)
        self.loc = ArucoLocalizer(load_json(MAP_PATH), self.car)
        self.grabbers = open_grabbers(self.car, 1280, 960, rear=True)
        time.sleep(1.5)

    def measure(self, seconds: float = 2.5):
        """(heading deg, spread deg, number of fixes, rms px) from several joint fits, or None."""
        yaws, rms, end = [], [], time.monotonic() + seconds
        while time.monotonic() < end:
            fixes = []
            for name, grabber in self.grabbers.items():
                frame, _stamp = grabber.latest()
                if frame is not None:
                    fixes += self.loc.observe(name, frame)[0]
            pose = self.loc.fuse(fixes, time.monotonic()) if fixes else None
            if pose is not None:
                yaws.append(pose.yaw)
                rms.append(pose.rms_px)
            time.sleep(0.25)
        if len(yaws) < 3:
            return None
        ref = yaws[0]
        rel = [(y - ref + math.pi) % (2 * math.pi) - math.pi for y in yaws]
        mean = ref + st.median(rel)
        return math.degrees((mean + math.pi) % (2 * math.pi) - math.pi), math.degrees(st.pstdev(rel)), len(yaws), st.median(rms)

    def close(self):
        for g in self.grabbers.values():
            try:
                g.close()
            except Exception:  # noqa: BLE001
                pass


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", action="store_true", help="show the heading on the arena map (needs a tag fix at the start)")
    opts = ap.parse_args()
    imu = Gy95t().start()
    tags = None
    gyro = None
    heading = module_at_align = None
    note = ""
    if opts.align:
        scale = float(json.loads(pathlib.Path(__file__).resolve().parents[1].joinpath("car_nav.json").read_text())
                      .get("imu", {}).get("gyro_z_scale", 1.0))
        gyro = YawGyro(imu, scale=scale)
        print("keep the car still: gyro bias for 2 s, then a tag fix ...", flush=True)
        time.sleep(0.5)
        bias, noise = gyro.calibrate(2.0)
        tags = TagHeading()
        first = tags.measure()
        if first is None:
            print("no usable tag fix: put the car inside the arena with tags in view of the front or rear camera.")
            tags.close()
            imu.close()
            return
        heading = first[0]
        s0 = imu.latest()
        module_at_align = s0.rpy_deg[2] if s0 else 0.0
        gyro.mark()
        note = (f"aligned by tags: {heading:+.1f} deg (spread {first[1]:.1f}, {first[2]} fits, rms {first[3]:.1f} px); "
                f"gyro bias {bias:+.3f} deg/s")
    gyro_yaw, yaw_zero, last_t, last_frames, last_check, rate = 0.0, 0.0, None, 0, time.monotonic(), 0.0
    clear = "\x1b[2J\x1b[H" if sys.stdout.isatty() else ""
    try:
        while True:
            s = imu.latest()
            now = time.monotonic()
            if s is not None and s.t != last_t:
                if last_t is not None:
                    gyro_yaw += s.gyro_dps[2] * min(s.t - last_t, 0.2)
                last_t = s.t
            if gyro is not None:
                taken = gyro.take()
                if taken is not None:
                    heading = wrap_deg(heading + math.degrees(taken[0]))
            if now - last_check >= 1.0:
                rate, last_frames, last_check = (imu.frames - last_frames) / (now - last_check), imu.frames, now
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                if tags is not None:
                    again = tags.measure()
                    if again is None:
                        note = "no tag fix this time; the gyro heading is kept"
                    else:
                        error = wrap_deg(heading - again[0])
                        note = (f"tag fix {again[0]:+.1f} deg (rms {again[3]:.1f} px); gyro had {heading:+.1f}: "
                                f"gyro - tags = {error:+.1f} deg; heading re-aligned to the tags")
                        heading = again[0]
                        gyro.mark()
                else:
                    gyro_yaw = 0.0
                    yaw_zero = s.rpy_deg[2] if s is not None else 0.0
            if s is None:
                out = "no data from the IMU (is /dev/ttyUSB0 there? is another program reading it?)"
            else:
                mag = math.hypot(*s.mag_gauss)
                if tags is not None:
                    shown_yaw = wrap_deg(s.rpy_deg[2] - module_at_align + (first[0] if first else 0.0))
                    head = (f"\nHEADING on the map (gyro, aligned by tags): {heading:+8.1f} deg   "
                            f"(0 = toward the right wall, counter-clockwise +)\n{note}\n")
                    tail = "Enter = new tag fix and show the gyro's drift, Ctrl+C = quit"
                else:
                    shown_yaw = wrap_deg(s.rpy_deg[2] - yaw_zero)
                    head = "\n"
                    tail = f"gyro z integrated: {gyro_yaw:+9.1f} deg   (Enter = zero, Ctrl+C = quit)"
                out = (f"GY-95T   {rate:3.0f} frames/s   bad {imu.bad_frames}   age {now - s.t:.2f} s\n{head}\n"
                       f"acc   x {s.acc_g[0]:+8.3f}   y {s.acc_g[1]:+8.3f}   z {s.acc_g[2]:+8.3f}   g\n"
                       f"gyro  x {s.gyro_dps[0]:+8.2f}   y {s.gyro_dps[1]:+8.2f}   z {s.gyro_dps[2]:+8.2f}   deg/s\n"
                       f"module roll {s.rpy_deg[0]:+8.2f}   pitch {s.rpy_deg[1]:+8.2f}   yaw {shown_yaw:+8.2f}   deg"
                       f"{'  (module yaw, shifted to the aligned heading)' if tags is not None else ''}\n"
                       f"mag   x {s.mag_gauss[0]:+8.3f}   y {s.mag_gauss[1]:+8.3f}   z {s.mag_gauss[2]:+8.3f}   gauss"
                       f"   |{mag:.2f}|  level {s.mag_level}\n"
                       f"temperature {s.temperature_c:.1f} C\n\n{tail}")
            sys.stdout.write(clear + out + "\n")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if tags is not None:
            tags.close()
        imu.close()


if __name__ == "__main__":
    main()
