#!/usr/bin/env python3
"""Capture front/left/right/rear for live surround stitch.

Order: USB (left, right, rear) first, then CSI front.
USB prefers YUYV with long warmup; left picks lowest-tear frame among candidates.

See BEV_LIVE_CAPTURE.md for pitfalls (left MJPG tears, right flip_v).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER))

from camera_devices import (  # noqa: E402
    create_csi_still_configuration,
    resolve_usb_capture_index,
)

DEFAULT_OUT = SERVER / "camera_labeled_live_now"
DEFAULT_BEV_OUT = SERVER / "bev_output" / "live_now_stitch"
HARDWARE = SERVER / "camera_hardware.json"
EXTRINSIC = SERVER / "calibration_patterns" / "bev_extrinsic_metric_auto"


def tear_score(bgr: np.ndarray) -> tuple[float, float, int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    row = np.abs(gray[1:] - gray[:-1]).mean(axis=1)
    mx = float(row.max())
    bottom = float(row[-50:].max()) if len(row) >= 50 else mx
    return mx, bottom, int(np.argmax(row))


def capture_usb_best(
    index: int,
    path: Path,
    tag: str,
    width: int,
    height: int,
    warmup_s: float,
    n_flush: int,
    n_candidates: int,
) -> np.ndarray:
    best = None
    best_key = None
    for fourcc in ("YUYV", "MJPG"):
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                f"/dev/video{index}",
                f"--set-fmt-video=width={width},height={height},pixelformat={fourcc}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        time.sleep(warmup_s)
        for _ in range(n_flush):
            cap.read()
            time.sleep(0.03)
        for i in range(n_candidates):
            ok, frame = cap.read()
            time.sleep(0.05)
            if not ok or frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            mx, bottom, at_y = tear_score(frame)
            key = (mx, bottom, -float(frame.mean()))
            print(
                f"  {tag} {fourcc}#{i}: tear={mx:.1f} bottom={bottom:.1f} "
                f"y={at_y} mean={frame.mean():.1f}"
            )
            if best is None or key < best_key:
                best = (frame.copy(), fourcc, mx, bottom)
                best_key = key
        cap.release()
        if best is not None and fourcc == "YUYV" and best[2] < 40 and best[3] < 35:
            break
    if best is None:
        raise RuntimeError(f"{tag}: capture failed on /dev/video{index}")
    frame, fourcc, mx, bottom = best
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    print(f"{tag}: chose {fourcc} tear={mx:.1f} bottom={bottom:.1f} -> {path}")
    return frame


def capture_csi_front(path: Path, width: int, height: int, warmup_s: float) -> np.ndarray:
    from picamera2 import Picamera2

    cam = Picamera2(camera_num=1)
    cam.configure(create_csi_still_configuration(cam, width, height))
    cam.start()
    cam.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(warmup_s)
    last = None
    for _ in range(20):
        last = cam.capture_array()
        time.sleep(0.06)
    if last is not None and float(np.mean(last)) < 70:
        cam.set_controls({"ExposureTime": 50000, "AnalogueGain": 6.0})
        for _ in range(10):
            last = cam.capture_array()
            time.sleep(0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    cam.capture_file(str(path))
    cam.stop()
    cam.close()
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"front: failed to read {path}")
    print(f"front: CSI camera_num=1 mean={img.mean():.1f} -> {path}")
    return img


def stitch(labeled_dir: Path, output_dir: Path) -> Path:
    summary = json.loads((EXTRINSIC / "metric_extrinsic_auto_summary.json").read_text(encoding="utf-8"))
    c = summary["car_mask_px"]
    subprocess.check_call(
        [
            sys.executable,
            str(SERVER / "bev_deploy.py"),
            "--extrinsic-dir",
            str(EXTRINSIC),
            "--labeled-dir",
            str(labeled_dir),
        ],
        cwd=str(SERVER),
    )
    subprocess.check_call(
        [
            sys.executable,
            str(SERVER / "bev_stitch.py"),
            "--blend",
            "--feather",
            "40",
            "--car-width",
            str(int(c["width"])),
            "--car-height",
            str(int(c["height"])),
            "--car-center-x",
            str(int(round(c["center"][0]))),
            "--car-center-y",
            str(int(round(c["center"][1]))),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(SERVER),
    )
    out = output_dir / "surround_square.jpg"
    print(f"stitched: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust 4-camera live capture for BEV surround.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--usb-warmup", type=float, default=2.5)
    parser.add_argument("--csi-warmup", type=float, default=2.0)
    parser.add_argument("--flush-frames", type=int, default=35)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--stitch", action="store_true", help="Deploy metric H and stitch after capture.")
    parser.add_argument("--bev-output", default=str(DEFAULT_BEV_OUT))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hw = json.loads(HARDWARE.read_text(encoding="utf-8"))

    print("=== USB first (left, right, rear) ===")
    for name in ("left", "right", "rear"):
        idx, how = resolve_usb_capture_index(hw[name])
        print(f"{name}: {how}")
        n_cand = args.candidates + 3 if name == "left" else args.candidates
        capture_usb_best(
            idx,
            out / f"{name}.jpg",
            name,
            args.width,
            args.height,
            args.usb_warmup,
            args.flush_frames,
            n_cand,
        )

    print("=== CSI front last ===")
    capture_csi_front(out / "front.jpg", args.width, args.height, args.csi_warmup)
    shutil.copy(out / "front.jpg", out / "front_csi_cam1.jpg")

    # optional 2x2 preview
    grid = np.zeros((args.height * 2, args.width * 2, 3), dtype=np.uint8)
    for name, (r, c) in (
        ("front", (0, 0)),
        ("left", (0, 1)),
        ("right", (1, 0)),
        ("rear", (1, 1)),
    ):
        im = cv2.imread(str(out / f"{name}.jpg"))
        if im is None:
            continue
        grid[
            r * args.height : (r + 1) * args.height,
            c * args.width : (c + 1) * args.width,
        ] = im
        cv2.putText(
            grid,
            name,
            (c * args.width + 10, r * args.height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
    preview = Path(args.bev_output)
    preview.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview / "raw_2x2.jpg"), grid)
    print(f"raw grid: {preview / 'raw_2x2.jpg'}")

    if args.stitch:
        stitch(out, Path(args.bev_output))


if __name__ == "__main__":
    main()
