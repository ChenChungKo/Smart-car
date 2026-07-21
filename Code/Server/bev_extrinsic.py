#!/usr/bin/env python3
"""Compute BEV homography H from plane_map_calibration.json on undistorted points."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB = SERVER / "calibration_patterns"
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_PLANE_MAP = SERVER / "plane_map_calibration.json"

# surroundBEV naming: rear -> back
CAMERA_ORDER = ("front", "left", "right", "rear")
SURROUND_NAMES = {"front": "front", "left": "left", "right": "right", "rear": "back"}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_kd(camera_name):
    if camera_name in ("left", "right", "rear"):
        k_path = CALIB / "shared" / "usb_fisheye_K.npy"
        d_path = CALIB / "shared" / "usb_fisheye_D.npy"
    else:
        k_path = CALIB / "captures" / camera_name / "camera_0_K.npy"
        d_path = CALIB / "captures" / camera_name / "camera_0_D.npy"
    if not k_path.exists() or not d_path.exists():
        raise FileNotFoundError(f"Missing K/D for {camera_name}: {k_path}, {d_path}")
    return np.load(k_path), np.load(d_path)


def camera_mat_dst(K, frame_width, frame_height, focal_scale, size_scale):
    dst = K.copy().astype(np.float64)
    dst[0, 0] *= focal_scale
    dst[1, 1] *= focal_scale
    dst[0, 2] = frame_width / 2.0 * size_scale
    dst[1, 2] = frame_height / 2.0 * size_scale
    return dst


def build_undistort_maps(K, D, calibration_mode, frame_width, frame_height, focal_scale, size_scale):
    out_w = int(frame_width * size_scale)
    out_h = int(frame_height * size_scale)
    p = camera_mat_dst(K, frame_width, frame_height, focal_scale, size_scale)
    if calibration_mode == "fisheye":
        return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), p, (out_w, out_h), cv2.CV_16SC2)
    return cv2.initUndistortRectifyMap(K, D, np.eye(3), p, (out_w, out_h), cv2.CV_16SC2)


def raw_points_to_undistorted(points, map1, map2):
    """Map raw-image corners to undistorted canvas via remap inverse lookup."""
    if map1.ndim == 3:
        src_x = map1[:, :, 0].astype(np.float32)
        src_y = map1[:, :, 1].astype(np.float32)
    else:
        src_x = map1.astype(np.float32)
        src_y = map2.astype(np.float32)

    undistorted = []
    warnings = []
    for x, y in points:
        dist = (src_x - float(x)) ** 2 + (src_y - float(y)) ** 2
        iy, ix = np.unravel_index(int(np.argmin(dist)), dist.shape)
        min_dist = float(dist[iy, ix])
        if min_dist > 900:
            warnings.append(f"({x}, {y}) loose fit min_dist={min_dist:.0f}")
        undistorted.append([float(ix), float(iy)])
    return np.array(undistorted, dtype=np.float32), warnings


def undistort_points(points, K, D, calibration_mode, frame_width, frame_height, focal_scale, size_scale):
    map1, map2 = build_undistort_maps(
        K, D, calibration_mode, frame_width, frame_height, focal_scale, size_scale
    )
    return raw_points_to_undistorted(points, map1, map2)


def mm_quad_to_bev(quad_mm, map_width_mm, map_height_mm, bev_width, bev_height):
    quad = np.array(quad_mm, dtype=np.float32)
    quad[:, 0] = quad[:, 0] / map_width_mm * bev_width
    quad[:, 1] = quad[:, 1] / map_height_mm * bev_height
    return quad


def compute_h_for_camera(
    camera_name,
    src_quad_px,
    dst_quad_mm,
    map_width_mm,
    map_height_mm,
    frame_width,
    frame_height,
    bev_width,
    bev_height,
    focal_scale,
    size_scale,
):
    hardware = load_json(HARDWARE)
    cam = hardware[camera_name]
    K, D = load_kd(camera_name)
    src_undist, corner_warnings = undistort_points(
        src_quad_px,
        K,
        D,
        cam["calibration_mode"],
        frame_width,
        frame_height,
        focal_scale,
        size_scale,
    )
    dst_bev = mm_quad_to_bev(dst_quad_mm, map_width_mm, map_height_mm, bev_width, bev_height)
    H = cv2.getPerspectiveTransform(src_undist, dst_bev)
    if abs(float(np.linalg.det(H[:2, :2]))) < 1e-8:
        raise ValueError(f"Degenerate homography for {camera_name}")
    return H, src_undist, dst_bev, corner_warnings


def main():
    parser = argparse.ArgumentParser(description="Build BEV H matrices from plane_map calibration.")
    parser.add_argument("--plane-map", default=str(DEFAULT_PLANE_MAP))
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--bev-width", type=int, default=1000)
    parser.add_argument("--bev-height", type=int, default=1000)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(CALIB / "bev_extrinsic"))
    args = parser.parse_args()

    plane_map = load_json(Path(args.plane_map))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_w = plane_map["map_width_mm"]
    map_h = plane_map["map_height_mm"]
    summary = {
        "source": str(args.plane_map),
        "frame_size": [args.frame_width, args.frame_height],
        "bev_size": [args.bev_width, args.bev_height],
        "focal_scale": args.focal_scale,
        "size_scale": args.size_scale,
        "cameras": {},
    }

    for camera in CAMERA_ORDER:
        cfg = plane_map["cameras"][camera]
        H, src_undist, dst_bev, corner_warnings = compute_h_for_camera(
            camera,
            cfg["src_quad_px"],
            cfg["dst_quad_mm"],
            map_w,
            map_h,
            args.frame_width,
            args.frame_height,
            args.bev_width,
            args.bev_height,
            args.focal_scale,
            args.size_scale,
        )
        surround_name = SURROUND_NAMES[camera]
        out_h = output_dir / f"camera_{surround_name}_H.npy"
        np.save(out_h, H)
        summary["cameras"][camera] = {
            "surround_name": surround_name,
            "H_file": str(out_h),
            "src_undist_px": src_undist.tolist(),
            "dst_bev_px": dst_bev.tolist(),
            "corner_warnings": corner_warnings,
        }
        if corner_warnings:
            print(f"  warnings: {corner_warnings}")
        print(f"{camera} -> {surround_name}: saved {out_h.name}")

    summary_path = output_dir / "bev_extrinsic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
