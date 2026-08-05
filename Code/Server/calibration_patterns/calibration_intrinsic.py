#!/usr/bin/env python3
"""Run intrinsic calibration on captured chessboard images."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = ROOT.parent
HARDWARE = SERVER / "camera_hardware.json"
CALIB_REPO = Path.home() / "CameraCalibration-test" / "IntrinsicCalibration"
DEFAULT_INPUT = ROOT / "captures"


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Compute intrinsic K/D from captured images.")
    parser.add_argument("--camera", choices=["front", "left", "right", "rear"], default="left")
    parser.add_argument("--input-dir", default="", help="Directory with img_raw*.jpg files.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--board-cols", type=int, default=7)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=int, default=25)
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()

    if not CALIB_REPO.exists():
        print(f"Missing repo: {CALIB_REPO}", file=sys.stderr)
        sys.exit(1)

    hardware = load_hardware()
    camera = hardware[args.camera]
    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT / args.camera
    if not input_dir.exists():
        print(f"Missing capture directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    images = sorted(input_dir.glob("img_raw*.jpg"))
    if not images:
        print(f"No img_raw*.jpg files in {input_dir}", file=sys.stderr)
        sys.exit(1)

    calib_type = "fisheye" if camera["calibration_mode"] == "fisheye" else "normal"
    output_k = input_dir / f"camera_{args.camera_id}_K.npy"
    output_d = input_dir / f"camera_{args.camera_id}_D.npy"

    cmd = [
        sys.executable,
        "intrinsicCalib.py",
        "-type",
        calib_type,
        "-input",
        "image",
        "-path",
        str(input_dir) + "/",
        "-image",
        "img_raw",
        "-fw",
        str(args.width),
        "-fh",
        str(args.height),
        "-bw",
        str(args.board_cols),
        "-bh",
        str(args.board_rows),
        "-size",
        str(args.square_mm),
        "-id",
        str(args.camera_id),
        "-num",
        "5",
    ]

    print(f"Running intrinsic calibration for {args.camera} ({calib_type})")
    print(f"Images: {len(images)} in {input_dir}")
    result = subprocess.run(cmd, cwd=str(CALIB_REPO))
    if result.returncode != 0:
        sys.exit(result.returncode)

    generated_k = CALIB_REPO / f"camera_{args.camera_id}_K.npy"
    generated_d = CALIB_REPO / f"camera_{args.camera_id}_D.npy"
    if generated_k.exists():
        generated_k.replace(output_k)
    if generated_d.exists():
        generated_d.replace(output_d)

    summary = {
        "camera": args.camera,
        "calibration_mode": calib_type,
        "image_count": len(images),
        "K_file": str(output_k),
        "D_file": str(output_d),
    }
    summary_path = input_dir / "intrinsic_result.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved K/D to {input_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
