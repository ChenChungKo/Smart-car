#!/usr/bin/env python3
"""Deploy Smart Car K/D/H and images to CameraCalibration SurroundBirdEyeView/data."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB = SERVER / "calibration_patterns"
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_SURROUND_DATA = Path.home() / "CameraCalibration-test" / "SurroundBirdEyeView" / "data"
DEFAULT_LABELED = SERVER / "camera_labeled"
DEFAULT_EXTRINSIC = CALIB / "bev_extrinsic"

CAMERAS = (
    ("front", "front", "front_csi_cam1.jpg"),
    ("left", "left", "left_usb_video0.jpg"),
    ("right", "right", "right_usb_video10.jpg"),
    ("rear", "back", "rear_usb_video37.jpg"),
)


def load_kd(camera_name):
    # Each camera now has its own individually-calibrated K/D (captured with
    # calibration_capture_smart.py). Previously left/right/rear shared one
    # "usb_fisheye" K/D, which ignored real per-unit manufacturing variance.
    return (
        CALIB / "captures" / camera_name / "camera_0_K.npy",
        CALIB / "captures" / camera_name / "camera_0_D.npy",
    )


def find_labeled_image(labeled_dir, preferred_name, camera_name):
    preferred = labeled_dir / preferred_name
    if preferred.exists():
        return preferred
    matches = sorted(labeled_dir.glob(f"{camera_name}*.jpg"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No labeled image for {camera_name} in {labeled_dir}")


def main():
    parser = argparse.ArgumentParser(description="Deploy calibration files to SurroundBirdEyeView/data.")
    parser.add_argument("--surround-data", default=str(DEFAULT_SURROUND_DATA))
    parser.add_argument("--extrinsic-dir", default=str(DEFAULT_EXTRINSIC))
    parser.add_argument("--labeled-dir", default=str(DEFAULT_LABELED))
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    args = parser.parse_args()

    surround_data = Path(args.surround_data)
    extrinsic_dir = Path(args.extrinsic_dir)
    labeled_dir = Path(args.labeled_dir)

    if not surround_data.parent.exists():
        print(f"Missing SurroundBirdEyeView: {surround_data.parent}", file=sys.stderr)
        sys.exit(1)

    manifest = {"surround_data": str(surround_data), "cameras": {}}

    for camera_name, surround_name, labeled_file in CAMERAS:
        cam_dir = surround_data / surround_name
        cam_dir.mkdir(parents=True, exist_ok=True)

        k_src, d_src = load_kd(camera_name)
        h_src = extrinsic_dir / f"camera_{surround_name}_H.npy"
        if not h_src.exists():
            print(f"Missing H: {h_src}. Run bev_extrinsic.py first.", file=sys.stderr)
            sys.exit(1)

        k_dst = cam_dir / f"camera_{surround_name}_K.npy"
        d_dst = cam_dir / f"camera_{surround_name}_D.npy"
        h_dst = cam_dir / f"camera_{surround_name}_H.npy"
        img_dst = cam_dir / f"{surround_name}.jpg"

        shutil.copy2(k_src, k_dst)
        shutil.copy2(d_src, d_dst)
        shutil.copy2(h_src, h_dst)

        src_img = find_labeled_image(labeled_dir, labeled_file, camera_name)
        image = cv2.imread(str(src_img))
        if image is None:
            raise RuntimeError(f"Cannot read {src_img}")
        if image.shape[1] != args.frame_width or image.shape[0] != args.frame_height:
            image = cv2.resize(image, (args.frame_width, args.frame_height))
        cv2.imwrite(str(img_dst), image)

        manifest["cameras"][camera_name] = {
            "surround_name": surround_name,
            "K": str(k_dst),
            "D": str(d_dst),
            "H": str(h_dst),
            "image": str(img_dst),
            "source_image": str(src_img),
        }
        print(f"Deployed {camera_name} -> {cam_dir}")

    manifest_path = extrinsic_dir / "bev_deploy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
