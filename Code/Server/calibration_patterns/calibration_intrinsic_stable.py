#!/usr/bin/env python3
"""Self-contained intrinsic calibration (no external repo dependency).

Two fixes over the upstream IntrinsicCalibration/intrinsicCalib.py:

1. Falls back to cv2.findChessboardCornersSB when the classic detector misses
   a pose (steep angle / lower contrast) — the classic detector alone often
   only accepts a handful of the diverse poses calibration_capture_smart.py
   captures, silently starving the calibration of data.
2. For fisheye cameras, fixes k3=k4=0 (CALIB_FIX_K3 | CALIB_FIX_K4). With
   ~18-20 images from a tabletop rig, the higher-order terms are poorly
   constrained and can blow up to extreme values (seen once: k3=-2.7,
   k4=3.7) that fit the calibration points but fold/shrink the usable field
   when undistorting full frames. k1,k2 alone already fit this lens well.

Usage:
    python3 calibration_intrinsic_stable.py --camera left
    python3 calibration_intrinsic_stable.py --camera front
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SERVER = ROOT.parent
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_INPUT = ROOT / "captures"


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def detect_corners(gray, board_size):
    ok, corners = cv2.findChessboardCorners(
        gray, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    )
    if ok:
        corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        )
        return True, corners
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, board_size)
        if ok:
            return True, corners.astype(np.float32)
    return False, None


def gather_points(input_dir, board_size, square_mm):
    files = sorted(glob.glob(str(input_dir / "img_raw*.jpg")))
    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_mm

    objpoints, imgpoints, image_files = [], [], []
    img_shape = None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ok, corners = detect_corners(gray, board_size)
        if ok:
            objpoints.append(objp)
            imgpoints.append(corners.reshape(1, -1, 2).astype(np.float64))
            image_files.append(Path(f).name)
    return objpoints, imgpoints, image_files, img_shape, len(files)


def calibrate_fisheye(objpoints, imgpoints, img_shape):
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3)) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3)) for _ in objpoints]
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_FIX_K3
        | cv2.fisheye.CALIB_FIX_K4
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs, tvecs, flags, criteria
    )
    return rms, K, D, rvecs, tvecs


def fisheye_view_errors(objpoints, imgpoints, K, D, rvecs, tvecs):
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, K, D)
        delta = projected.reshape(-1, 2) - imgp.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return errors


def calibrate_normal(objpoints, imgpoints, img_shape):
    objpoints_cc = [p.reshape(-1, 1, 3).astype(np.float32) for p in objpoints]
    imgpoints_cc = [p.reshape(-1, 1, 2).astype(np.float32) for p in imgpoints]
    rms, K, D, _rvecs, _tvecs = cv2.calibrateCamera(objpoints_cc, imgpoints_cc, img_shape, None, None)
    return rms, K, D


def main():
    parser = argparse.ArgumentParser(description="Stable in-house intrinsic calibration.")
    parser.add_argument("--camera", choices=["front", "left", "right", "rear", "gimbal"], required=True)
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--output-dir", default="", help="Save K/D and summary here (default: input directory).")
    parser.add_argument("--board-cols", type=int, default=7)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument(
        "--max-view-rms",
        type=float,
        default=0.75,
        help="Reject fisheye views above this per-image RMS, then recalibrate (0 disables).",
    )
    args = parser.parse_args()

    hardware = load_hardware()
    camera = hardware[args.camera]
    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT / args.camera
    if not input_dir.exists():
        raise SystemExit(f"Missing capture directory: {input_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    board_size = (args.board_cols, args.board_rows)
    objpoints, imgpoints, image_files, img_shape, total = gather_points(input_dir, board_size, args.square_mm)
    if len(objpoints) < 5:
        raise SystemExit(f"Only {len(objpoints)}/{total} images had a detectable board; need >= 5.")

    is_fisheye = camera["calibration_mode"] == "fisheye"
    rejected_views = []
    if is_fisheye:
        rms, K, D, rvecs, tvecs = calibrate_fisheye(objpoints, imgpoints, img_shape)
        if args.max_view_rms > 0:
            view_errors = fisheye_view_errors(objpoints, imgpoints, K, D, rvecs, tvecs)
            keep = [index for index, error in enumerate(view_errors) if error <= args.max_view_rms]
            rejected = [index for index, error in enumerate(view_errors) if error > args.max_view_rms]
            if rejected and len(keep) >= 5:
                rejected_views = [
                    {"file": image_files[index], "rms_reprojection_error_px": view_errors[index]}
                    for index in rejected
                ]
                objpoints = [objpoints[index] for index in keep]
                imgpoints = [imgpoints[index] for index in keep]
                image_files = [image_files[index] for index in keep]
                rms, K, D, rvecs, tvecs = calibrate_fisheye(objpoints, imgpoints, img_shape)
    else:
        rms, K, D = calibrate_normal(objpoints, imgpoints, img_shape)

    output_k = output_dir / f"camera_{args.camera_id}_K.npy"
    output_d = output_dir / f"camera_{args.camera_id}_D.npy"
    np.save(output_k, K)
    np.save(output_d, D)

    summary = {
        "camera": args.camera,
        "calibration_mode": "fisheye" if is_fisheye else "normal",
        "image_count_total": total,
        "image_count_used": len(objpoints),
        "image_files_used": image_files,
        "rejected_views": rejected_views,
        "max_view_rms_px": args.max_view_rms if is_fisheye else None,
        "rms_reprojection_error_px": rms,
        "K": K.tolist(),
        "D": D.ravel().tolist(),
        "K_file": str(output_k),
        "D_file": str(output_d),
        "fixed_k3_k4": is_fisheye,
    }
    summary_path = output_dir / "intrinsic_result.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"{args.camera}: used {len(objpoints)}/{total} images, RMS={rms:.4f}px")
    for view in rejected_views:
        print(f"Rejected: {view['file']} (view RMS={view['rms_reprojection_error_px']:.4f}px)")
    print(f"K=\n{K}")
    print(f"D={D.ravel()}")
    print(f"Saved: {output_k}, {output_d}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
