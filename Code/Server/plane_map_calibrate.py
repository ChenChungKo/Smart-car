#!/usr/bin/env python3
"""Interactive tool to pick mat corners for plane-map calibration."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIBRATION = Path(__file__).with_name("plane_map_calibration.json")
DEFAULT_INPUT_DIR = Path(__file__).with_name("camera_labeled")
LABELS = ["TL", "TR", "BR", "BL"]


def load_calibration(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_calibration(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def pick_quad(image, window_name, existing=None):
    points = []
    if existing:
        points = [list(map(int, point)) for point in existing]

    display = image.copy()

    def redraw():
        nonlocal display
        display = image.copy()
        if points:
            for index, point in enumerate(points):
                cv2.circle(display, tuple(point), 6, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    LABELS[index],
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            if len(points) >= 2:
                cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (0, 0, 255), 2)
            if len(points) == 4:
                cv2.polylines(display, [np.array(points, dtype=np.int32)], True, (0, 0, 255), 2)

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) >= 4:
            points.clear()
        points.append([x, y])
        redraw()

    redraw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(points) == 4:
            break
        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise SystemExit("Calibration cancelled")
        if key == ord("r"):
            points.clear()
            redraw()

    cv2.destroyWindow(window_name)
    return points


def main():
    parser = argparse.ArgumentParser(description="Pick mat corners for each camera.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--camera", choices=["front", "left", "right", "rear"], help="Calibrate one camera only.")
    args = parser.parse_args()

    calibration = load_calibration(args.calibration)
    input_dir = Path(args.input_dir)
    camera_names = [args.camera] if args.camera else list(calibration["cameras"].keys())

    for name in camera_names:
        camera = calibration["cameras"][name]
        image_path = input_dir / camera["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")

        print(f"Click TL, TR, BR, BL for {name}. Enter/Space=save, r=reset, q=quit")
        points = pick_quad(image, f"calibrate-{name}", camera.get("src_quad_px"))
        camera["src_quad_px"] = points

    save_calibration(args.calibration, calibration)
    print(f"Saved calibration to {args.calibration}")


if __name__ == "__main__":
    main()
