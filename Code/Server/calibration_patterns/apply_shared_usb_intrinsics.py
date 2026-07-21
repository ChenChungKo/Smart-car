#!/usr/bin/env python3
"""Apply left USB fisheye K/D as shared intrinsics for left/right/rear."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "shared"
DEFAULT_SOURCE = ROOT / "captures" / "left"
USB_CAMERAS = ("left", "right", "rear")


def main():
    parser = argparse.ArgumentParser(description="Share left USB fisheye intrinsics across USB cameras.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Directory with source camera_0_K/D.npy")
    args = parser.parse_args()

    source = Path(args.source)
    source_k = source / "camera_0_K.npy"
    source_d = source / "camera_0_D.npy"
    if not source_k.exists() or not source_d.exists():
        print(f"Missing K/D in {source}", file=sys.stderr)
        sys.exit(1)

    SHARED.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_k, SHARED / "usb_fisheye_K.npy")
    shutil.copy2(source_d, SHARED / "usb_fisheye_D.npy")

    K = np.load(SHARED / "usb_fisheye_K.npy")
    D = np.load(SHARED / "usb_fisheye_D.npy")
    source_name = source.parent.name if source.parent.name in USB_CAMERAS else "left"

    summary = {
        "source_camera": source_name,
        "calibration_mode": "fisheye",
        "shared_with": list(USB_CAMERAS),
        "K_file": str(SHARED / "usb_fisheye_K.npy"),
        "D_file": str(SHARED / "usb_fisheye_D.npy"),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "D": D.ravel().tolist(),
        "notes": "三顆 BL 1080p S10 USB 魚眼共用同一組內參。",
    }
    (ROOT / "usb_fisheye_intrinsics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for camera in USB_CAMERAS:
        dest = ROOT / "captures" / camera
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SHARED / "usb_fisheye_K.npy", dest / "camera_0_K.npy")
        shutil.copy2(SHARED / "usb_fisheye_D.npy", dest / "camera_0_D.npy")
        per_camera = {
            "camera": camera,
            "calibration_mode": "fisheye",
            "intrinsics_source": f"shared_{source_name}",
            "shared_from": str(SHARED / "usb_fisheye_K.npy"),
            "K_file": str(dest / "camera_0_K.npy"),
            "D_file": str(dest / "camera_0_D.npy"),
        }
        (dest / "intrinsic_result.json").write_text(json.dumps(per_camera, indent=2) + "\n", encoding="utf-8")

    print(f"Shared intrinsics from {source_name} applied to {', '.join(USB_CAMERAS)}")
    print(f"Master files: {SHARED / 'usb_fisheye_K.npy'}, {SHARED / 'usb_fisheye_D.npy'}")


if __name__ == "__main__":
    main()
