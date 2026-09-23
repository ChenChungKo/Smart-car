#!/usr/bin/env python3
"""Capture one labeled camera image with live preview (open that camera alone)."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_OUTPUT = SERVER / "camera_labeled"

sys.path.insert(0, str(SERVER))
from camera_devices import (
    create_csi_still_configuration,
    csi_array_to_bgr,
    resolve_usb_capture_index,
)


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def detect_board(gray, board_size=(7, 6)):
    """Try several OpenCV detectors; CSI low-angle views often fail the default finder."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flag_sets = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
    ]
    for flags in flag_sets:
        ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            return True, corners

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb_flags |= cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, board_size, flags=sb_flags)
        if ok and corners is not None:
            return True, corners.astype(np.float32)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    ok, corners = cv2.findChessboardCorners(
        blurred,
        board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if ok:
        corners = cv2.cornerSubPix(blurred, corners, (5, 5), (-1, -1), criteria)
        return True, corners
    return False, None


def open_usb(device_index, width, height, warmup, warmup_frames):
    capture = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{device_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    time.sleep(warmup)
    for _ in range(warmup_frames):
        capture.read()
        time.sleep(0.1)
    return capture


def open_csi(camera_num, width, height, warmup, warmup_frames):
    from picamera2 import Picamera2

    camera = Picamera2(camera_num=camera_num)
    config = create_csi_still_configuration(camera, width, height)
    camera.configure(config)
    camera.start()
    camera.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(warmup)
    for _ in range(warmup_frames):
        camera.capture_array()
        time.sleep(0.1)
    return camera


def capture_with_preview(read_frame, release, output_path, camera_name, board_size, require_board):
    """Live preview window + save by pressing ENTER/f/q inside the OpenCV window (no TTY needed)."""
    window = f"labeled-{camera_name}"
    state = {"frame": None, "ok_board": False, "running": True, "quit": False, "save": None}
    lock = threading.Lock()

    def preview_loop():
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while state["running"]:
            ok, frame = read_frame()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok_board, corners = detect_board(gray, board_size)
            with lock:
                state["frame"] = frame.copy()
                state["ok_board"] = ok_board
            display = frame.copy()
            if ok_board:
                cv2.drawChessboardCorners(display, board_size, corners, True)
            status = (
                f"{camera_name} | board={'OK' if ok_board else 'not found'} | "
                "Enter=save  f=force save  q=quit"
            )
            color = (0, 255, 0) if ok_board else (0, 0, 255)
            cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                state["quit"] = True
            elif key in (ord(" "), 10, 13):
                state["save"] = "normal"
            elif key in (ord("f"), ord("F")):
                state["save"] = "force"
            time.sleep(0.01)
        cv2.destroyAllWindows()

    print(f"Live preview: {camera_name}")
    print(">>> 請在跳出的視窗上操作：Enter=存檔  f=強制存檔  q=取消")

    preview_thread = threading.Thread(target=preview_loop, daemon=True)
    preview_thread.start()
    time.sleep(0.8)

    saved = False
    try:
        while True:
            if state["quit"]:
                print("Cancelled.")
                break

            with lock:
                window_save = state["save"]
                state["save"] = None

            if window_save is None:
                time.sleep(0.05)
                continue

            force = window_save == "force"
            with lock:
                frame = None if state["frame"] is None else state["frame"].copy()
                ok_board = state["ok_board"]
            if frame is None:
                print("No frame yet — wait for preview, then try again.")
                continue
            if require_board and not ok_board and not force:
                print("Board not found — adjust board, press Enter again, or f to force-save.")
                continue

            cv2.imwrite(str(output_path), frame)
            if force or not ok_board:
                print(f"Force-saved: {output_path}")
            else:
                print(f"Saved: {output_path}")
            saved = True
            break
    finally:
        state["running"] = False
        preview_thread.join(timeout=2.0)
        release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Preview then capture one labeled camera image.")
    parser.add_argument("--camera", choices=["front", "left", "right", "rear"], required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument(
        "--allow-no-board",
        action="store_true",
        help="Enter may save even when chessboard is not detected.",
    )
    args = parser.parse_args()

    hardware = load_hardware()
    camera = hardware[args.camera]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    require_board = not args.allow_no_board

    if camera["interface"] == "usb":
        device_index, resolved = resolve_usb_capture_index(camera)
        print(f"USB device: /dev/video{device_index} ({resolved})")
        capture = open_usb(device_index, args.width, args.height, args.warmup, args.warmup_frames)
        output_path = output_dir / f"{args.camera}_usb_video{device_index}.jpg"

        def read_frame():
            return capture.read()

        def release():
            capture.release()

    elif camera["interface"] == "csi":
        camera_num = 1 if args.camera == "front" else 0
        print(f"CSI camera_num={camera_num}")
        csi = open_csi(camera_num, args.width, args.height, args.warmup, args.warmup_frames)
        output_path = output_dir / f"{args.camera}_csi_cam{camera_num}.jpg"

        def read_frame():
            return True, csi_array_to_bgr(csi.capture_array())

        def release():
            csi.close()
    else:
        raise RuntimeError(f"Unsupported interface: {camera['interface']}")

    ok = capture_with_preview(
        read_frame,
        release,
        output_path,
        args.camera,
        board_size=(7, 6),
        require_board=require_board,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
