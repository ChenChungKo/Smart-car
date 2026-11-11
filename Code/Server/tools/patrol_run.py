#!/usr/bin/env python3
"""Patrol the four corners among whatever stands in the arena, then park where the car started. Dry-run unless --arm.

    python3 tools/patrol_run.py              # looks, plans and records for a few seconds; the wheels are not driven
    python3 tools/patrol_run.py --arm        # drives
    options: --inset 25 (cm, corner points from the walls)  --margin 4 (cm kept from things)  --speed 0.13
             --start-clear 8 (cm around the car that must be empty when it starts: no sensor looks there)
             --seconds 300  --no-window

Only the size of the arena and its wall tags are known; nothing about what stands in it. The parts (patrol/):
  estimator  gyro + commanded speed + tag poses + wall echoes -> pose
  sensing    the gimbal camera's foot line (IMU tilt applied) and the ultrasonic -> the map, looking where the car goes
  world      occupied / seen free / not seen
  planner    A* in x, y, heading for the body's rectangle; thinking keeps a path to the current goal
  drive      follows the path; the car moves only onto floor it has seen free within its stopping distance
  mission    four corners, then home; parking by the ultrasonic readings taken at the start
  recorder   every cycle, look, plan and map on disk; gui shows it live
A window opens on the Pi's desktop (also saved every second as gui.jpg). q in the window or Ctrl+C stops the car.
Afterwards: python3 tools/patrol_why.py <folder> shows any moment of the run.
"""
import argparse
import pathlib
import signal
import sys
import threading
import time

SERVER = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "tools"))
from patrol.gui import Gui  # noqa: E402
from patrol.navigator import Navigator  # noqa: E402
from patrol.recorder import Recorder  # noqa: E402
from route_follow import RouteFollower, parse_args  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--inset", type=float, default=25.0, help="cm from the walls to the corner points")
    ap.add_argument("--margin", type=float, default=4.0, help="cm the plan keeps the body off anything occupied")
    ap.add_argument("--speed", type=float, default=0.13)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--dry-seconds", type=float, default=8.0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--start-clear", type=float, default=8.0, help="cm around the car taken as free at the start (no sensor sees beside or behind it)")
    opts = ap.parse_args()
    out = SERVER / "route_follow_debug" / "patrol" / time.strftime("%Y%m%d-%H%M%S")
    rec = Recorder(out)
    print("command: " + " ".join(sys.argv), flush=True)
    car = RouteFollower(parse_args(["--no-window", "--no-ir", "--debug-dir", "route_follow_debug/calibrate"] + (["--arm"] if opts.arm else [])))
    nav = Navigator(car, opts, rec)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: car.stop_event.set())
    worker = threading.Thread(target=nav.run, daemon=True)
    worker.start()
    Gui(nav, out, show=not opts.no_window).loop(worker.is_alive)
    worker.join(timeout=5.0)
    r = nav.result
    print(f"\n{r.get('reason')}: {r.get('seconds', 0):.1f} s, goals reached {r.get('goals_reached')} of 5, skipped {r.get('skipped')}, "
          f"camera looks {r.get('looks')} (too fast for the picture: {r.get('smeared')})")
    if r.get("parking"):
        print(f"parking: {r['parking']['measurements']} measurements, echoes against the start (ahead, right) cm: {r['parking']['errors_cm']}")
    print(f"saved in {out}\nlook at any moment: python3 tools/patrol_why.py {out}", flush=True)


if __name__ == "__main__":
    main()
