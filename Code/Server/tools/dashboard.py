#!/usr/bin/env python3
"""Web dashboard of the car, standalone: all cameras, ArUco position, bird's-eye view, sensors, map.

    python3 tools/dashboard.py [--port 5000] [--no-cameras] [--no-sensors]

Open http://<the Pi's address>:5000 in a browser. The cameras and sensors have one owner at a time: do not run it while
route_follow.py runs (use route_follow.py --dashboard for that: the same page inside the lap).
"""
import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.hub import HUB  # noqa: E402
from dashboard.live import LiveSource  # noqa: E402
from dashboard.server import start_server  # noqa: E402


def addresses() -> list[str]:
    try:
        out = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        out = []
    return [a for a in out if not a.startswith("127.")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--no-sensors", action="store_true")
    args = ap.parse_args()
    source = LiveSource(HUB, cameras=not args.no_cameras, sensors=not args.no_sensors)
    source.start()
    start_server(HUB, source.arena, source.car, port=args.port)
    for a in addresses() or ["<the Pi's address>"]:
        print(f"dashboard: http://{a}:{args.port}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()


if __name__ == "__main__":
    main()
