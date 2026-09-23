#!/usr/bin/env python3
"""Live front+rear CSI preview while placing extrinsic chessboards.

Green text means OpenCV locked the 7x6 inner corners. The yellow box is the
size the board should cover. Press q in the window to quit.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(SERVER / "calibration_patterns"))

from calibration_capture import detect_board_preview  # noqa: E402
from camera_devices import create_csi_still_configuration, csi_array_to_bgr  # noqa: E402

BOARD = (7, 6)
WIDTH, HEIGHT = 640, 480


def open_csi(camera_num: int):
    from picamera2 import Picamera2

    camera = Picamera2(camera_num=camera_num)
    camera.configure(create_csi_still_configuration(camera, WIDTH, HEIGHT))
    camera.start()
    camera.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(1.2)
    for _ in range(8):
        camera.capture_array()
    return camera


def grab_loop(camera, box, running):
    while running["go"]:
        frame = csi_array_to_bgr(camera.capture_array())
        if frame is not None:
            box["frame"] = frame


def main():
    print("Opening rear CSI camera_num=0 ...")
    rear = open_csi(0)
    print("Opening front CSI camera_num=1 ...")
    front = open_csi(1)

    running = {"go": True}
    boxes = {"front": {"frame": None}, "rear": {"frame": None}}
    threads = [
        threading.Thread(target=grab_loop, args=(front, boxes["front"], running), daemon=True),
        threading.Thread(target=grab_loop, args=(rear, boxes["rear"], running), daemon=True),
    ]
    for thread in threads:
        thread.start()

    window = "front | rear   q=quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 480)
    board_ok = {"front": False, "rear": False}
    last_detect = 0.0
    print("Preview open. Move each board until it covers the yellow box and the label turns green.")

    try:
        while True:
            frames = []
            now = time.time()
            detect_now = now - last_detect >= 0.35
            if detect_now:
                last_detect = now
            for name in ("front", "rear"):
                frame = boxes[name]["frame"]
                if frame is None:
                    view = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                    cv2.putText(view, f"{name} waiting", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    frames.append(view)
                    continue
                view = frame.copy()
                if detect_now:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ok, corners = detect_board_preview(gray, BOARD)
                    board_ok[name] = bool(ok)
                    if ok and corners is not None:
                        cv2.drawChessboardCorners(view, BOARD, corners, True)
                x0, y0, x1, y1 = 210, 150, 430, 330
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 2)
                label = "BOARD OK" if board_ok[name] else "move board closer"
                color = (0, 220, 0) if board_ok[name] else (0, 0, 255)
                cv2.putText(view, f"{name}: {label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                frames.append(view)
            sheet = cv2.hconcat(frames)
            cv2.imshow(window, sheet)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        running["go"] = False
        cv2.destroyAllWindows()
        front.close()
        rear.close()
        print("Preview closed.")


if __name__ == "__main__":
    main()
