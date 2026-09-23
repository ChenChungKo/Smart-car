#!/usr/bin/env python3
"""Auto metric extrinsic from afternoon chessboard images + known mat layout.

Uses today's camera_labeled/*.jpg, current per-camera K/D, and board placement
from metric_layout_aug3.json / the user-provided car corners. No top-down photo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB = SERVER / "calibration_patterns"
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_LABELED = SERVER / "camera_labeled"
DEFAULT_LAYOUT = CALIB / "metric_layout_aug3.json"
DEFAULT_OUTPUT = CALIB / "bev_extrinsic_metric_auto"

CAMERA_ORDER = ("front", "left", "right", "rear")
SURROUND_NAMES = {"front": "front", "left": "left", "right": "right", "rear": "back"}

# CSI front/rear still need a BEV rotation when a homography exists. USB
# left/right are disambiguated by side_orientation_ok (car-at-bottom of the
# USB frame, image-right=front on the left cam / image-left=front on the right).
MANUAL_ORIENTATION_FIX = {
    "front": [("rotate", 180.0)],
    "rear": [("flip_h", 0.0), ("rotate", -90.0)],
}


def orientation_fix_matrix(kind, angle_deg, pivot):
    px, py = float(pivot[0]), float(pivot[1])
    if kind == "rotate":
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Image/BEV coords: +Y down, so visually-CCW uses [[c,s],[-s,c]].
        R = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T1 = np.array([[1.0, 0.0, -px], [0.0, 1.0, -py], [0.0, 0.0, 1.0]])
        T2 = np.array([[1.0, 0.0, px], [0.0, 1.0, py], [0.0, 0.0, 1.0]])
        return T2 @ R @ T1
    if kind == "flip_h":
        # Mirror across vertical axis through pivot: x' = 2*px - x
        return np.array([[-1.0, 0.0, 2.0 * px], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    if kind == "flip_v":
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0 * py], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unknown orientation fix kind: {kind}")

sys.path.insert(0, str(SERVER))
from bev_extrinsic import build_undistort_maps, load_kd, undistort_points  # noqa: E402
from bev_extrinsic_chessboard import corner_orders, detect_board  # noqa: E402


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def find_labeled(labeled_dir: Path, camera_name: str) -> Path:
    matches = sorted(labeled_dir.glob(f"{camera_name}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No labeled image for {camera_name}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def board_world_corners_mm(layout, camera: str) -> np.ndarray:
    """Return (42,2) mm coords for inner corners in one canonical order.

    Canonical order matches OpenCV board_size=(7,6) reshape(rows=6, cols=7):
    row increases along board "depth" away from car, col increases toward
    car-right / board-right as defined in the layout notes.
    """
    square = float(layout["square_cm"]) * 10.0  # mm
    board = layout["boards"][camera]
    # Named corner is in cm (long, short) -> (X,Y) mm
    named = np.array(board["corner_long_short"], dtype=np.float64) * 10.0

    # Outer square counts: 8 along the long-span of the board face, 7 along depth.
    # Front/rear: 8 along short (Y), 7 along long (X).
    # Left/right: 8 along long (X), 7 along short (Y).
    if camera in ("front", "rear"):
        # named = car-facing edge, right end
        # cols (7) along +Y toward left of named? named is RIGHT end, so cols
        # increase toward -Y (leftward from right end) or +Y?
        # Outer Y: [named_y - 8*square, named_y], inner first col near left.
        # Put col=0 at left (smaller Y), col=6 near named right.
        if camera == "front":
            # depth toward smaller X; row=0 near car-facing (larger X), row=5 farther
            xs = named[0] - square - np.arange(6) * square  # away from car
            # Actually car-facing inner line is named_x - square (one square inset)
            xs = named[0] - square - np.arange(6) * square
            ys = (named[1] - 8 * square + square) + np.arange(7) * square
        else:
            # rear: depth toward larger X; row=0 near car-facing (smaller X)
            xs = named[0] + square + np.arange(6) * square
            ys = (named[1] - 8 * square + square) + np.arange(7) * square
        grid = np.zeros((6, 7, 2), dtype=np.float64)
        for r in range(6):
            for c in range(7):
                grid[r, c] = (xs[r], ys[c])
    else:
        # left/right: named = car-facing edge, front end (smaller X / front of car)
        # 8 squares along +X, 7 along short depth
        if camera == "left":
            # depth toward smaller Y
            ys = named[1] - square - np.arange(6) * square
            xs = named[0] + square + np.arange(7) * square  # col along +X from front end
            # Wait board_size cols=7 along X means 7 corners; 8 squares => 7 gaps.
            # named is FRONT end of car-facing edge. Outer X [named_x, named_x+8*square]
            # inner X starts named_x+square, 7 corners along X → use cols=7 along X
            # But reshape is (6,7): rows=6 along depth Y, cols=7 along X.
            xs = named[0] + square + np.arange(7) * square
            grid = np.zeros((6, 7, 2), dtype=np.float64)
            for r in range(6):
                for c in range(7):
                    grid[r, c] = (xs[c], ys[r])
        else:
            # right: depth toward larger Y
            ys = named[1] + square + np.arange(6) * square
            xs = named[0] + square + np.arange(7) * square
            grid = np.zeros((6, 7, 2), dtype=np.float64)
            for r in range(6):
                for c in range(7):
                    grid[r, c] = (xs[c], ys[r])
    return grid.reshape(-1, 2)


def mm_to_bev(points_mm, map_w, map_h, bev_w, bev_h, px_per_mm=None):
    """Map mat mm -> front-up BEV pixels with uniform metric scale.

    Layout uses long/short with car nose toward decreasing long (X).
    Surround stitch expects front=up, left=left, so:
      u (right)  <- short / Y
      v (down)   <- long  / X   (small X / nose at top)
    """
    pts = np.asarray(points_mm, dtype=np.float32)
    if px_per_mm is None:
        px_per_mm = min(bev_w / map_h, bev_h / map_w)
    out = np.empty_like(pts)
    # Center the mat on the BEV canvas.
    origin_u = 0.5 * (bev_w - map_h * px_per_mm)
    origin_v = 0.5 * (bev_h - map_w * px_per_mm)
    out[:, 0] = origin_u + pts[:, 1] * px_per_mm
    out[:, 1] = origin_v + pts[:, 0] * px_per_mm
    return out


def side_orientation_ok(camera, homography, src_corners, car_uv):
    """USB side cameras: image bottom is the car; front of the car is
    image-right on the left camera and image-left on the right camera.
    """
    if camera not in ("left", "right"):
        return True
    center = src_corners.reshape(-1, 2).mean(axis=0)
    probes = {
        "below": center + np.array([0.0, 50.0]),
        "above": center + np.array([0.0, -50.0]),
        "left": center + np.array([-50.0, 0.0]),
        "right": center + np.array([50.0, 0.0]),
    }
    mapped = {}
    for key, point in probes.items():
        mapped[key] = cv2.perspectiveTransform(
            point.reshape(1, 1, 2).astype(np.float32), homography
        )[0, 0]
    car = np.asarray(car_uv, dtype=np.float32)
    dist_below = float(np.linalg.norm(mapped["below"] - car))
    dist_above = float(np.linalg.norm(mapped["above"] - car))
    if dist_below >= dist_above:
        return False
    if camera == "left":
        return mapped["right"][1] < mapped["left"][1]
    return mapped["left"][1] < mapped["right"][1]


def best_h(src_undist, dst_bev, board_size=(7, 6), camera="front", car_uv=None):
    best = None
    for src_id, src in enumerate(corner_orders(src_undist, board_size)):
        for dst_id, dst in enumerate(corner_orders(dst_bev, board_size)):
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is None:
                continue
            if car_uv is not None and not side_orientation_ok(camera, H, src_undist, car_uv):
                continue
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            err = float(np.linalg.norm(proj - dst.reshape(-1, 2), axis=1).mean())
            inliers = int(mask.sum()) if mask is not None else 0
            if best is None or err < best["err"]:
                best = {
                    "H": H,
                    "err": err,
                    "inliers": inliers,
                    "src_order_id": src_id,
                    "dst_order_id": dst_id,
                }
    return best


def main():
    parser = argparse.ArgumentParser(description="Metric extrinsic from chessboard layout.")
    parser.add_argument("--layout", default=str(DEFAULT_LAYOUT))
    parser.add_argument("--labeled-dir", default=str(DEFAULT_LABELED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--bev-width", type=int, default=1000)
    parser.add_argument("--bev-height", type=int, default=1000)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    args = parser.parse_args()

    layout = load_json(Path(args.layout))
    hardware = load_json(HARDWARE)
    labeled_dir = Path(args.labeled_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    map_w = float(layout["mat_cm"][0]) * 10.0  # long axis mm
    map_h = float(layout["mat_cm"][1]) * 10.0  # short axis mm
    board_size = tuple(layout.get("board_inner_corners", [7, 6]))
    px_per_mm = min(args.bev_width / map_h, args.bev_height / map_w)
    car = layout["car"]
    car_center_mm = 0.25 * (
        np.array(car["front_left"])
        + np.array(car["front_right"])
        + np.array(car["rear_left"])
        + np.array(car["rear_right"])
    ) * 10.0
    car_center_bev = mm_to_bev(
        [car_center_mm], map_w, map_h, args.bev_width, args.bev_height, px_per_mm
    )[0]
    car_width_px = int(round(float(car["width_cm"]) * 10.0 * px_per_mm))
    car_height_px = int(round(float(car["length_cm"]) * 10.0 * px_per_mm))

    summary = {
        "method": "metric_chessboard_layout_auto",
        "layout": str(args.layout),
        "map_mm": [map_w, map_h],
        "px_per_mm": px_per_mm,
        "frame_size": [args.frame_width, args.frame_height],
        "bev_size": [args.bev_width, args.bev_height],
        "car_mask_px": {
            "width": car_width_px,
            "height": car_height_px,
            "center": [float(car_center_bev[0]), float(car_center_bev[1])],
        },
        "cameras": {},
    }

    ok_count = 0
    for camera in CAMERA_ORDER:
        image_path = find_labeled(labeled_dir, camera)
        raw = cv2.imread(str(image_path))
        if raw is None:
            raise RuntimeError(f"Cannot read {image_path}")
        if raw.shape[1] != args.frame_width or raw.shape[0] != args.frame_height:
            raw = cv2.resize(raw, (args.frame_width, args.frame_height))

        mode = hardware[camera]["calibration_mode"]
        K, D = load_kd(camera)
        map1, map2 = build_undistort_maps(
            K, D, mode, args.frame_width, args.frame_height, args.focal_scale, args.size_scale
        )
        und = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        cv2.imwrite(str(debug_dir / f"{camera}_undist.jpg"), und)

        corners = detect_board(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), board_size)
        source_kind = "undistorted"
        if corners is None:
            corners_raw = detect_board(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), board_size)
            if corners_raw is None:
                print(f"{camera}: FAILED — no chessboard in {image_path.name}")
                summary["cameras"][camera] = {"ok": False, "reason": "no_board"}
                continue
            src_undist, warnings = undistort_points(
                corners_raw.reshape(-1, 2),
                K,
                D,
                mode,
                args.frame_width,
                args.frame_height,
                args.focal_scale,
                args.size_scale,
            )
            source_kind = "raw_mapped"
        else:
            src_undist = corners.reshape(-1, 2).astype(np.float32)
            warnings = []

        world_mm = board_world_corners_mm(layout, camera)
        dst_bev = mm_to_bev(world_mm, map_w, map_h, args.bev_width, args.bev_height, px_per_mm)
        # Keep dst as Nx1x2 for corner_orders
        dst_bev_pts = dst_bev.reshape(-1, 1, 2).astype(np.float32)
        src_pts = src_undist.reshape(-1, 1, 2).astype(np.float32)

        best = best_h(src_pts, dst_bev_pts, board_size, camera, car_center_bev)
        if best is None:
            print(f"{camera}: FAILED — no homography")
            summary["cameras"][camera] = {"ok": False, "reason": "no_H"}
            continue

        H = best["H"]
        fixes = MANUAL_ORIENTATION_FIX.get(camera) or []
        if fixes:
            pivot = dst_bev.reshape(-1, 2).mean(axis=0)
            for kind, angle_deg in fixes:
                H = orientation_fix_matrix(kind, angle_deg, pivot) @ H
                print(f"{camera}: applied orientation fix {kind} {angle_deg:+.0f} about {pivot}")
        surround = SURROUND_NAMES[camera]
        h_path = output_dir / f"camera_{surround}_H.npy"
        np.save(h_path, H)

        warped = cv2.warpPerspective(und, H, (args.bev_width, args.bev_height))
        canvas = np.zeros((args.bev_height, args.bev_width, 3), dtype=np.uint8)
        # mat outline + car box
        cv2.rectangle(canvas, (0, 0), (args.bev_width - 1, args.bev_height - 1), (60, 60, 60), 2)
        for a, b in (
            ("front_left", "front_right"),
            ("front_right", "rear_right"),
            ("rear_right", "rear_left"),
            ("rear_left", "front_left"),
        ):
            p0 = mm_to_bev([np.array(car[a]) * 10.0], map_w, map_h, args.bev_width, args.bev_height, px_per_mm)[0]
            p1 = mm_to_bev([np.array(car[b]) * 10.0], map_w, map_h, args.bev_width, args.bev_height, px_per_mm)[0]
            cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 255), 2)
        mask = (warped > 0).any(axis=2)
        canvas[mask] = (canvas[mask] * 0.35 + warped[mask] * 0.65).astype(np.uint8)
        # draw expected board corners
        for p in dst_bev:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        cv2.imwrite(str(debug_dir / f"{camera}_warped.jpg"), canvas)

        print(
            f"{camera} -> {surround}: err={best['err']:.2f}px inliers={best['inliers']} "
            f"src_order={best['src_order_id']} dst_order={best['dst_order_id']} "
            f"from={source_kind} ({image_path.name})"
        )
        if warnings:
            print(f"  warnings: {warnings}")
        summary["cameras"][camera] = {
            "ok": True,
            "surround_name": surround,
            "source_image": str(image_path),
            "H_file": str(h_path),
            "reproj_err_px": best["err"],
            "inliers": best["inliers"],
            "src_order_id": best["src_order_id"],
            "dst_order_id": best["dst_order_id"],
            "source_kind": source_kind,
            "world_mm_sample": world_mm[[0, 6, 35, 41]].tolist(),
        }
        ok_count += 1

    summary_path = output_dir / "metric_extrinsic_auto_summary.json"
    save_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    if ok_count < 4:
        sys.exit(2)


if __name__ == "__main__":
    main()
