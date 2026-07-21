#!/usr/bin/env python3
"""Stitch four labeled camera views into one top-down plane map."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CALIBRATION = Path(__file__).with_name("plane_map_calibration.json")
DEFAULT_INPUT_DIR = Path(__file__).with_name("camera_labeled")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("plane_map_output")


def quad_area(quad):
    return float(cv2.contourArea(np.array(quad, dtype=np.float32)))


def validate_quad(quad, image_shape):
    points = np.array(quad, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] != 4:
        raise RuntimeError("A mat quad must contain exactly four points")
    if len({tuple(map(float, point)) for point in points}) != 4:
        raise RuntimeError(f"Degenerate mat quad detected: {points.tolist()}")
    height, width = image_shape[:2]
    if quad_area(points) < width * height * 0.04:
        raise RuntimeError("Detected mat quad is too small")


def build_src_mask(image, src_quad):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, src_quad.astype(np.int32), 255)
    return mask


def detect_mat_quad(image, debug_path=None):
    height, width = image.shape[:2]
    roi_top = int(height * 0.10)
    roi = image[roi_top:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_roi = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[roi_top:] = mask_roi

    row_counts = np.count_nonzero(mask, axis=1)
    active_rows = np.where(row_counts > width * 0.12)[0]
    if len(active_rows) < 10:
        raise RuntimeError("No green mat region detected")

    top_row = int(active_rows[0])
    bottom_row = int(active_rows[-1])

    def row_edges(row):
        cols = np.where(mask[row] > 0)[0]
        if len(cols) < 20:
            return None
        return float(cols[0]), float(cols[-1])

    chosen_top = None
    for row in range(top_row, bottom_row + 1):
        edges = row_edges(row)
        if edges is None:
            continue
        if edges[1] - edges[0] >= width * 0.25:
            chosen_top = row
            break
    if chosen_top is None:
        chosen_top = top_row

    top = row_edges(chosen_top)
    bottom = row_edges(bottom_row)
    if top is None or bottom is None:
        raise RuntimeError("Could not estimate mat edges")

    quad = np.array(
        [
            [top[0], float(chosen_top)],
            [top[1], float(chosen_top)],
            [bottom[1], float(bottom_row)],
            [bottom[0], float(bottom_row)],
        ],
        dtype=np.float32,
    )
    validate_quad(quad, image.shape)

    if debug_path is not None:
        preview = image.copy()
        cv2.polylines(preview, [quad.astype(np.int32)], True, (0, 0, 255), 2)
        labels = ["TL", "TR", "BR", "BL"]
        for label, point in zip(labels, quad.astype(int)):
            cv2.circle(preview, tuple(point), 6, (0, 255, 255), -1)
            cv2.putText(
                preview,
                label,
                tuple(point + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        cv2.imwrite(str(debug_path), preview)

    return quad


def load_calibration(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_src_quad(image, camera, output_dir, name, auto_detect):
    debug_path = output_dir / f"debug_{name}_mat_quad.jpg"
    if auto_detect or "src_quad_px" not in camera:
        return detect_mat_quad(image, debug_path=debug_path)

    quad = np.array(camera["src_quad_px"], dtype=np.float32)
    validate_quad(quad, image.shape)
    preview = image.copy()
    cv2.polylines(preview, [quad.astype(np.int32)], True, (0, 0, 255), 2)
    labels = ["TL", "TR", "BR", "BL"]
    for label, point in zip(labels, quad.astype(int)):
        cv2.circle(preview, tuple(point), 6, (0, 255, 255), -1)
        cv2.putText(preview, label, tuple(point + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite(str(debug_path), preview)
    return quad


def warp_camera_view(image, src_mask, src_quad, dst_quad, map_size):
    map_width, map_height = map_size
    masked = cv2.bitwise_and(image, image, mask=src_mask)
    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    if abs(float(np.linalg.det(homography[:2, :2]))) < 1e-6:
        raise RuntimeError("Invalid homography matrix")

    warped = cv2.warpPerspective(masked, homography, (map_width, map_height))
    weight = cv2.warpPerspective(src_mask.astype(np.float32), homography, (map_width, map_height))
    return warped, weight, homography


def draw_grid(image, step):
    gridded = image.copy()
    height, width = gridded.shape[:2]
    for x in range(0, width, step):
        cv2.line(gridded, (x, 0), (x, height), (0, 180, 0), 1)
    for y in range(0, height, step):
        cv2.line(gridded, (0, y), (width, y), (0, 180, 0), 1)
    return gridded


def build_layout_image(tiles, tile_size=(320, 240)):
    order = ["front", "left", "right", "rear"]
    tile_width, tile_height = tile_size
    canvas = np.zeros((tile_height * 3, tile_width * 3, 3), dtype=np.uint8)
    positions = {
        "front": (tile_width, 0),
        "left": (0, tile_height),
        "right": (tile_width * 2, tile_height),
        "rear": (tile_width, tile_height * 2),
    }
    for name in order:
        tile = cv2.resize(tiles[name], tile_size)
        x, y = positions[name]
        canvas[y : y + tile_height, x : x + tile_width] = tile
        cv2.putText(
            canvas,
            name,
            (x + 10, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return canvas


def stitch_plane_map(input_dir, output_dir, calibration_path, auto_detect=False):
    calibration = load_calibration(calibration_path)
    map_width = int(calibration["map_width_mm"])
    map_height = int(calibration["map_height_mm"])
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stitched = np.zeros((map_height, map_width, 3), dtype=np.uint8)
    tiles = {}
    draw_order = ["rear", "left", "right", "front"]

    for name in draw_order:
        camera = calibration["cameras"][name]
        image_path = input_dir / camera["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing camera image: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        src_quad = resolve_src_quad(image, camera, output_dir, name, auto_detect)
        dst_quad = np.array(camera["dst_quad_mm"], dtype=np.float32)
        validate_quad(dst_quad, (map_height, map_width, 3))
        src_mask = build_src_mask(image, src_quad)

        warped, weight, homography = warp_camera_view(
            image,
            src_mask,
            src_quad,
            dst_quad,
            (map_width, map_height),
        )
        tiles[name] = warped

        active = weight > 0.05
        stitched[active] = warped[active]

        with open(output_dir / f"{name}_homography.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "src_quad_px": src_quad.tolist(),
                    "dst_quad_mm": dst_quad.tolist(),
                    "homography": homography.tolist(),
                },
                handle,
                indent=2,
            )

    center = calibration.get("car_center_mm", [map_width // 2, map_height // 2])
    cv2.drawMarker(
        stitched,
        (int(center[0]), int(center[1])),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    grid_step = int(calibration.get("grid_step_mm", 100))
    gridded = draw_grid(stitched, grid_step)
    layout = build_layout_image(tiles)

    stitched_path = output_dir / "plane_map_stitched.jpg"
    gridded_path = output_dir / "plane_map_stitched_grid.jpg"
    layout_path = output_dir / "plane_map_layout.jpg"
    cv2.imwrite(str(stitched_path), stitched)
    cv2.imwrite(str(gridded_path), gridded)
    cv2.imwrite(str(layout_path), layout)
    return stitched_path, gridded_path, layout_path


def build_parser():
    parser = argparse.ArgumentParser(description="Combine four camera views into one plane map.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Detect src_quad_px from the green mat instead of using calibration JSON.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    stitched_path, gridded_path, layout_path = stitch_plane_map(
        args.input_dir,
        args.output_dir,
        args.calibration,
        auto_detect=args.auto_detect,
    )
    print(f"Stitched map saved to {stitched_path}")
    print(f"Gridded map saved to {gridded_path}")
    print(f"Layout preview saved to {layout_path}")


if __name__ == "__main__":
    main()
