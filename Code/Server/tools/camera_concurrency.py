#!/usr/bin/env python3
"""Can all three USB cameras stream at once? Opens each, reads for a few seconds in parallel, prints frames/s.

    python3 tools/camera_concurrency.py [seconds]

Run with nothing else using the cameras (no route_follow). Tries YUYV (raw, what the gimbal feed uses) and MJPG
(compressed: a fraction of the USB bandwidth) at 640 x 480, all three together.
"""
import sys, threading, time
from pathlib import Path
import cv2

PATHS = sorted(Path("/dev/v4l/by-path").glob("*usb-0:*video-index0"))
# one per physical camera: the platform-xhci-hcd.N-usb-0:PORT entries (the usbv2 aliases are the same devices)
PATHS = [p for p in PATHS if "usbv2" not in p.name]
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0


def grab(path, fourcc, out, key):
    cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        out[key] = "could not open"
        return
    n, bad, start = 0, 0, time.monotonic()
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            n += 1
        else:
            bad += 1
            time.sleep(0.02)
    out[key] = f"{n / seconds:.1f} fps, {bad} failed reads"
    cap.release()


for fourcc in ("YUYV", "MJPG"):
    out = {}
    threads = [threading.Thread(target=grab, args=(p, fourcc, out, p.name)) for p in PATHS]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f"== {fourcc}, {len(PATHS)} cameras at once, 640x480:")
    for name in sorted(out):
        print(f"   {name}: {out[name]}")
    time.sleep(1.0)
