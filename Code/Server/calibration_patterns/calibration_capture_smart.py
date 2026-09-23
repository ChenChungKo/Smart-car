#!/usr/bin/env python3
"""Guided chessboard capture for intrinsic calibration.

Solves two practical problems:
1. "Angles are hard to define" — you don't aim for exact degrees. The tool
   shows a 3x3 coverage grid (where in the frame the board has been seen)
   plus a live tilt gauge (how much perspective foreshortening the current
   pose has). Just move/tilt the board until the grid fills up and the tilt
   gauge shows both LOW and HIGH readings have been captured.
2. "Each lens detects differently" — detection runs live per-camera with the
   same detector as other capture tools, so you only ever need poses THAT
   camera can actually see; the live overlay tells you the moment it's
   detected so you know exactly when it's safe to save.

Saving is MANUAL: press ENTER in the terminal when the live preview shows
the board detected (green text) and the zone/tilt combo you want.

Usage:
    python3 calibration_capture_smart.py --camera left
    python3 calibration_capture_smart.py --camera front --board-cols 7 --board-rows 6

Terminal: ENTER = save current frame, q = quit
Window:   q = quit
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from calibration_capture import (  # noqa: E402
    CAMERA_CHOICES,
    csi_camera_num,
    detect_board_preview,
    load_hardware,
    next_index,
    open_csi,
    open_usb,
    read_csi,
)

sys.path.insert(0, str(ROOT.parent))
from camera_devices import resolve_usb_capture_index  # noqa: E402

DEFAULT_OUTPUT = ROOT / "captures"
GRID_N = 3  # 3x3 position coverage
TILT_LOW, TILT_HIGH = 0.12, 0.30  # heuristic thresholds; see compute_tilt_and_center()
DETECT_INTERVAL_SEC = 0.15  # throttle chessboard detection to keep preview smooth


def compute_tilt_and_center(corners, board_size):
    """Return (tilt_score, center_xy) from the 4 outer board corners.

    tilt_score ~0 for a fronto-parallel board (looks like a plain rectangle),
    higher when the board is rotated/tilted relative to the sensor (edges of
    unequal length in the image) — exactly the poses needed to separate fx/fy
    and get well-conditioned distortion coefficients.
    """
    cols, rows = board_size
    pts = corners.reshape(-1, 2)
    top_left = pts[0]
    top_right = pts[cols - 1]
    bottom_left = pts[(rows - 1) * cols]
    bottom_right = pts[-1]

    def edge_len(a, b):
        return float(np.linalg.norm(a - b))

    top = edge_len(top_left, top_right)
    bottom = edge_len(bottom_left, bottom_right)
    left = edge_len(top_left, bottom_left)
    right = edge_len(top_right, bottom_right)

    horiz_skew = abs(top - bottom) / max((top + bottom) / 2.0, 1e-6)
    vert_skew = abs(left - right) / max((left + right) / 2.0, 1e-6)
    tilt_score = float(horiz_skew + vert_skew)

    center = pts.mean(axis=0)
    return tilt_score, center


def tilt_bucket(score):
    if score < TILT_LOW:
        return "flat"
    if score < TILT_HIGH:
        return "mid"
    return "tilt"


def draw_coverage_grid(frame, covered_bins, frame_w, frame_h, origin=(10, 40), cell_px=28):
    ox, oy = origin
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            x0, y0 = ox + gx * cell_px, oy + gy * cell_px
            filled = covered_bins.get((gx, gy), set())
            color = (60, 60, 60)
            if "tilt" in filled and "flat" in filled:
                color = (0, 220, 0)
            elif filled:
                color = (0, 180, 220)
            cv2.rectangle(frame, (x0, y0), (x0 + cell_px - 3, y0 + cell_px - 3), color, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + cell_px - 3, y0 + cell_px - 3), (255, 255, 255), 1)
    for i in range(1, GRID_N):
        x = int(frame_w * i / GRID_N)
        y = int(frame_h * i / GRID_N)
        cv2.line(frame, (x, 0), (x, frame_h), (90, 90, 90), 1)
        cv2.line(frame, (0, y), (frame_w, y), (90, 90, 90), 1)


def main():
    parser = argparse.ArgumentParser(description="Guided diverse-pose chessboard capture (manual save).")
    parser.add_argument("--camera", choices=CAMERA_CHOICES, required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--board-cols", type=int, default=7)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--fresh", action="store_true", help="Delete existing img_raw*.jpg before capture.")
    args = parser.parse_args()

    board_size = (args.board_cols, args.board_rows)
    hardware = load_hardware()
    camera = hardware[args.camera]
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT / args.camera
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for path in output_dir.glob("img_raw*.jpg"):
            path.unlink()
        print(f"Cleared old captures in {output_dir}")

    print(f"Camera: {args.camera} ({camera['model']})  mode={camera['calibration_mode']}")

    if camera["interface"] == "usb":
        device_index, resolved_from = resolve_usb_capture_index(camera)
        print(f"USB device: /dev/video{device_index} ({resolved_from})")
        capture = open_usb(device_index, args.width, args.height, args.warmup, args.warmup_frames)

        def read_frame():
            return capture.read()

        def release():
            capture.release()

    elif camera["interface"] == "csi":
        camera_num = csi_camera_num(args.camera, camera)
        print(f"CSI camera_num={camera_num}")
        csi = open_csi(camera_num, args.width, args.height, args.warmup, args.warmup_frames)

        def read_frame():
            return read_csi(csi)

        def release():
            csi.close()

    else:
        raise RuntimeError(f"Unsupported interface: {camera['interface']}")

    window = f"smart-calib-{args.camera}"
    covered_bins: dict[tuple[int, int], set[str]] = {}
    index = next_index(output_dir)
    saved_count = 0
    total_target = GRID_N * GRID_N * 2  # flat + tilt per cell

    state = {
        "frame": None,
        "ok_board": False,
        "bin_xy": None,
        "score": None,
        "bucket": None,
        "running": True,
        "quit": False,
    }
    lock = threading.Lock()
    frame_box = {"frame": None}
    frame_lock = threading.Lock()

    def grab_loop():
        """Drain the camera as fast as possible so the driver buffer never backs up."""
        while state["running"]:
            ok_frame, frame = read_frame()
            if ok_frame and frame is not None:
                with frame_lock:
                    frame_box["frame"] = frame
            else:
                time.sleep(0.01)

    def preview_loop():
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        ok_board, corners = False, None
        last_detect_time = 0.0
        while state["running"]:
            with frame_lock:
                frame = frame_box["frame"]
            if frame is None:
                time.sleep(0.01)
                continue

            now_detect = time.time()
            if now_detect - last_detect_time >= DETECT_INTERVAL_SEC:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok_board, corners = detect_board_preview(gray, board_size)
                last_detect_time = now_detect

            display = frame.copy()
            h, w = frame.shape[:2]
            status_color = (0, 0, 255)
            hint = "board NOT found - adjust board"
            bin_xy = score = bucket = None

            if ok_board:
                cv2.drawChessboardCorners(display, board_size, corners, ok_board)
                score, center = compute_tilt_and_center(corners, board_size)
                bucket = tilt_bucket(score)
                bin_x = min(GRID_N - 1, int(center[0] / w * GRID_N))
                bin_y = min(GRID_N - 1, int(center[1] / h * GRID_N))
                bin_xy = (bin_x, bin_y)
                with lock:
                    have = covered_bins.get(bin_xy, set())
                status_color = (0, 220, 0)
                note = "NEW" if bucket not in have else "already have"
                hint = f"zone={bin_xy} tilt={score:.2f}[{bucket}] have={sorted(have)} ({note})"

            with lock:
                state["frame"] = frame
                state["ok_board"] = ok_board
                state["bin_xy"] = bin_xy
                state["score"] = score
                state["bucket"] = bucket
                cur_covered = dict(covered_bins)
                cur_saved = saved_count

            draw_coverage_grid(display, cur_covered, w, h)
            cv2.putText(
                display,
                f"{args.camera} | saved={cur_saved}/{total_target} | {hint}",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
            )
            cv2.putText(
                display,
                "TERMINAL: Enter=save  q=quit",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                with lock:
                    state["quit"] = True
        cv2.destroyAllWindows()

    print("Move the board so its center visits all 9 grid zones.")
    print("For each zone, aim for once roughly FLAT and once TILTED.")
    print(f"Target: {total_target} photos (green cell = zone done, but you decide every save).")
    print("Watch the preview window; press ENTER here to save the CURRENT frame.\n")

    grab_thread = threading.Thread(target=grab_loop, daemon=True)
    preview_thread = threading.Thread(target=preview_loop, daemon=True)
    grab_thread.start()
    preview_thread.start()
    time.sleep(0.6)

    try:
        while True:
            with lock:
                if state["quit"]:
                    print("Quit from preview window.")
                    break
            with lock:
                ok_board = state["ok_board"]
                bin_xy = state["bin_xy"]
                score = state["score"]
                bucket = state["bucket"]

            status = f"board OK zone={bin_xy} tilt={score:.2f}[{bucket}]" if ok_board else "board NOT found"
            answer = input(f"[{saved_count}/{total_target}] {status} | Enter=save, q=quit: ").strip().lower()
            if answer in ("q", "quit", "exit"):
                break

            with lock:
                if state["quit"]:
                    print("Quit from preview window.")
                    break
                frame = state["frame"]
                ok_board = state["ok_board"]
                bin_xy = state["bin_xy"]
                score = state["score"]
                bucket = state["bucket"]

            if frame is None:
                print("No frame yet — wait a moment, then try again.")
                continue
            if not ok_board:
                print("  Not saved — board not detected. Adjust and press ENTER again.")
                continue

            path = output_dir / f"img_raw{index}.jpg"
            cv2.imwrite(str(path), frame)
            with lock:
                have = covered_bins.setdefault(bin_xy, set())
                have.add(bucket)
            saved_count += 1
            index += 1
            print(f"  Saved {path.name}  zone={bin_xy} bucket={bucket} tilt={score:.2f}  total={saved_count}")
    finally:
        with lock:
            state["running"] = False
        preview_thread.join(timeout=2.0)
        grab_thread.join(timeout=2.0)
        release()

    print(f"\nDone. Saved {saved_count} images to {output_dir}")
    if saved_count < 10:
        print("WARNING: fewer than 10 usable images — intrinsic calibration may be under-constrained.")


if __name__ == "__main__":
    main()
