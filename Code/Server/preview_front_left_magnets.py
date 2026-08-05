#!/usr/bin/env python3
"""Live side-by-side front+left preview with purple-magnet highlight.

Goal: move the front-left purple strip until BOTH cameras see the same strip,
then press s to save for magnet-based seam alignment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camera_devices import create_csi_still_configuration, resolve_usb_capture_index

SERVER = Path(__file__).resolve().parent
HARDWARE = SERVER / "camera_hardware.json"


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def purple_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 25, 35), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    purple = cv2.inRange(hsv, (125, 55, 55), (170, 255, 255))
    near = cv2.dilate(green, np.ones((25, 25), np.uint8))
    m = cv2.bitwise_and(purple, near)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def overlay_purple(bgr: np.ndarray) -> tuple[np.ndarray, int]:
    m = purple_mask(bgr)
    vis = bgr.copy()
    vis[m > 0] = (0, 0, 255)
    out = cv2.addWeighted(bgr, 0.7, vis, 0.3, 0)
    return out, int((m > 0).sum())


def open_csi(camera_num: int = 1, size=(640, 480)):
    """Full-sensor FOV (native 640x480 mode center-crops and looks zoomed)."""
    from picamera2 import Picamera2

    cam = Picamera2(camera_num=camera_num)
    cam.configure(create_csi_still_configuration(cam, size[0], size[1]))
    cam.start()
    cam.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(0.5)
    for _ in range(4):
        cam.capture_array()
    return cam


def open_usb(index: int, capture_size=(1920, 1080)):
    """Open full-FOV capture. Native 640x480 on these UVC cams is a center crop.

    Prefer YUYV at 1080p: MJPEG often returns truncated frames (bottom corruption).
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{index}")
    # FourCC before size — otherwise some UVC drivers stay in cropped 640 mode.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"left capture mode: {got[0]}x{got[1]} YUYV (will scale to 640x480)")
    for _ in range(6):
        cap.read()
    return cap


def read_usb_full_fov(cap, out_size=(640, 480)):
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    if (frame.shape[1], frame.shape[0]) != out_size:
        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
    return True, frame


def grab_best_usb_frame(cap, out_size=(640, 480), tries=8):
    """MJPG 1080 sometimes returns truncated frames; keep highest-detail one."""
    best, best_std = None, -1.0
    for _ in range(tries):
        ok, frame = read_usb_full_fov(cap, out_size)
        if not ok:
            continue
        std = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std())
        if std > best_std:
            best_std = std
            best = frame
    return best


def main():
    parser = argparse.ArgumentParser(description="Front+left live preview for magnet placement.")
    parser.add_argument("--csi", type=int, default=1)
    parser.add_argument("--left-video", type=int, default=-1, help="-1 = resolve from hardware JSON")
    parser.add_argument("--save-dir", default=str(SERVER / "camera_labeled_corners"))
    args = parser.parse_args()

    hw = load_hardware()
    left_idx = args.left_video
    if left_idx < 0:
        left_idx, how = resolve_usb_capture_index(hw["left"])
        print(f"resolved left: {how}")

    print(f"front=CSI{args.csi} (full FOV)  left=/dev/video{left_idx} (1080->640 full FOV)")
    print("Move purple strip into OVERLAP until BOTH panels show red highlight.")
    print("Keys: s=save pair  q=quit")

    csi = open_csi(args.csi)
    usb = open_usb(left_idx)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    win = "left | front  (s save, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 360)

    while True:
        rgb = csi.capture_array()
        front = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, left = read_usb_full_fov(usb)
        if not ok:
            continue
        if front.shape[1] != 640 or front.shape[0] != 480:
            front = cv2.resize(front, (640, 480), interpolation=cv2.INTER_AREA)

        f_vis, f_px = overlay_purple(front)
        l_vis, l_px = overlay_purple(left)
        cv2.putText(f_vis, f"FRONT purple={f_px}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(l_vis, f"LEFT purple={l_px}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        status = "OK overlap" if (f_px > 200 and l_px > 200) else "move strip into both views"
        color = (0, 220, 0) if "OK" in status else (0, 165, 255)
        sheet = np.hstack([l_vis, f_vis])
        cv2.putText(sheet, status, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(win, sheet)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            # refresh left with best of several frames (avoid truncated MJPEG)
            left_best = grab_best_usb_frame(usb) or left
            front_path = save_dir / "front_csi_cam1.jpg"
            left_path = save_dir / f"left_usb_video{left_idx}.jpg"
            cv2.imwrite(str(front_path), front)
            cv2.imwrite(str(left_path), left_best)
            labeled = SERVER / "camera_labeled"
            labeled.mkdir(exist_ok=True)
            cv2.imwrite(str(labeled / "front_csi_cam1.jpg"), front)
            cv2.imwrite(str(labeled / f"left_usb_video{left_idx}.jpg"), left_best)
            print(f"saved {front_path.name} + {left_path.name}  (purple front={f_px} left={l_px})")

    csi.stop()
    csi.close()
    usb.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
