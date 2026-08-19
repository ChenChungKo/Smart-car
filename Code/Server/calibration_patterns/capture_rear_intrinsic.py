#!/usr/bin/env python3
"""Capture varied-pose chessboard frames from the rear USB camera for a
dedicated (non-shared) fisheye intrinsic calibration.

Usage: python3 capture_rear_intrinsic.py [--device /dev/video37] [--out DIR]
"""

import argparse
import time
from pathlib import Path

import cv2

BOARD_W, BOARD_H = 7, 6


def find_corners(img):
    ok, corners = cv2.findChessboardCorners(
        img,
        (BOARD_W, BOARD_H),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
    )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video37")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "captures" / "rear_v2"))
    parser.add_argument("--target", type=int, default=22, help="Target number of good chessboard frames.")
    parser.add_argument("--interval", type=float, default=1.8, help="Seconds between capture attempts.")
    parser.add_argument("--max-seconds", type=float, default=90.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = int(args.device.rsplit("video", 1)[1])
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.device}")

    # warmup / let AE settle
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    saved = 0
    attempts = 0
    start = time.time()
    print(f"Capturing to {out_dir} ; target={args.target} good frames, interval={args.interval}s")
    while saved < args.target and (time.time() - start) < args.max_seconds:
        ok, frame = cap.read()
        attempts += 1
        if not ok or frame is None:
            time.sleep(0.2)
            continue
        remaining = args.max_seconds - (time.time() - start)
        print(f"[t={time.time()-start:5.1f}s] attempt {attempts}: checking chessboard... (saved={saved}/{args.target}, {remaining:.0f}s left)")
        if find_corners(frame):
            fp = out_dir / f"img_raw{saved}.jpg"
            cv2.imwrite(str(fp), frame)
            print(f"  -> FOUND, saved {fp.name}. Move the board to a new pose now.")
            saved += 1
        else:
            print("  -> no board detected, retrying...")
        time.sleep(args.interval)

    cap.release()
    print(f"Done. Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
