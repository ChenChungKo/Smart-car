#!/usr/bin/env python3
"""All five cameras at once with ArUco detection on every frame: frames/s, detections/s and CPU load. Nothing moves.

    python3 tools/camera_load_test.py [seconds [csi_w csi_h [usb_detect_hz [csi_detect_hz]]]]

CSI front/rear at the size route_follow uses (1280x960, observe() with the full pose fit), the three USB cameras at
640x480 (YUYV, detection only). Reports per camera: frames read per second, frames with at least one tag, and the
CPU use of the whole Pi over the run (so it also shows what the detection costs).
"""
import sys, threading, time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aruco_localizer import ArucoLocalizer, CAR_PATH, MAP_PATH, load_json, open_grabbers

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
CSI_W = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
CSI_H = int(sys.argv[3]) if len(sys.argv) > 3 else 960
USB_HZ = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # 0 = every frame
CSI_HZ = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0   # 0 = every frame


def cpu_times():
    a = [float(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    return sum(a), a[3] + a[4]  # total, idle (+iowait)


arena, car = load_json(MAP_PATH), load_json(CAR_PATH)
loc = ArucoLocalizer(arena, car)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
stats = {}
stop = threading.Event()


def csi(name, grabber):
    frames = tagged = 0
    last = 0.0
    last_run = 0.0
    while not stop.is_set():
        frame, stamp = grabber.latest()
        if frame is None or stamp == last:
            time.sleep(0.005)
            continue
        last = stamp
        if CSI_HZ and time.monotonic() - last_run < 1.0 / CSI_HZ:
            continue
        last_run = time.monotonic()
        fixes, _c, _i = loc.observe(name, frame)
        frames += 1
        tagged += bool(fixes)
    stats[name] = (frames, tagged)


def usb(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frames = tagged = 0
    last_det = 0.0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        if USB_HZ and time.monotonic() - last_det < 1.0 / USB_HZ:
            continue
        last_det = time.monotonic()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _corners, ids, _rej = detector.detectMarkers(gray)
        frames += 1
        tagged += ids is not None
    cap.release()
    stats["usb:" + path.name.split("-usb-0:")[1][:4]] = (frames, tagged)


grabbers = open_grabbers(car, CSI_W, CSI_H, rear=True)
time.sleep(4.0)
threads = [threading.Thread(target=csi, args=(n, g)) for n, g in grabbers.items()]
paths = [p for p in sorted(Path("/dev/v4l/by-path").glob("*usb-0:*video-index0")) if "usbv2" not in p.name]
threads += [threading.Thread(target=usb, args=(p,)) for p in paths]
t0, i0 = cpu_times()
start = time.monotonic()
[t.start() for t in threads]
time.sleep(seconds)
stop.set()
[t.join() for t in threads]
t1, i1 = cpu_times()
elapsed = time.monotonic() - start
print(f"{len(grabbers)} CSI at {CSI_W}x{CSI_H} (detect {CSI_HZ or 'every frame'} Hz) + {len(paths)} USB 640x480 (detect {USB_HZ or 'every frame'} Hz), {elapsed:.0f} s")
for name in sorted(stats):
    frames, tagged = stats[name]
    print(f"  {name:12s} {frames / elapsed:5.1f} frames/s   {tagged / max(frames, 1) * 100:3.0f} % of frames with a tag")
print(f"  CPU of the whole Pi: {(1 - (i1 - i0) / (t1 - t0)) * 100:.0f} %")
for g in grabbers.values():
    try:
        g.close()
    except Exception:
        pass
