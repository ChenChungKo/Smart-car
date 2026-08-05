#!/usr/bin/env python3
"""Extrinsic H from labeled camera images + top-down reference (chessboard pairs)."""

import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB = SERVER / "calibration_patterns"
HARDWARE = SERVER / "camera_hardware.json"
DEFAULT_LABELED = SERVER / "camera_labeled"
DEFAULT_TOPDOWN = CALIB / "extrinsic_ref" / "topdown_front_up.png"
DEFAULT_OUTPUT = CALIB / "bev_extrinsic"

sys.path.insert(0, str(SERVER))
from camera_devices import bev_focal_scale, bev_size_scale

# Car front faces TOP in topdown_front_up.png.
# Regions selected via TOPDOWN_REGION_SETS below.
def corner_orders(corners, board_size=(7, 6)):
    """Generate all 8 dihedral-symmetry corner order variants (rotations + mirrors).

    A checkerboard's corner scan-order start/direction is not guaranteed to be
    consistent across two independent views of the same physical board (e.g.
    a camera looking from a very different angle than the overhead reference
    photo). The original 5-candidate set (identity/reverse/row-flip/col-flip)
    only covers 0 deg and 180 deg relabelings; it cannot express a 90/270 deg
    mismatch, which happens when the board's short/long axis is swapped
    relative to the other view. Adding the transpose-based candidates below
    covers all 8 symmetries of a rectangular grid.
    """
    w, h = board_size
    c = corners.reshape(-1, 2).astype(np.float32)
    grid = c.reshape(h, w, 2)
    grid_t = grid.transpose(1, 0, 2)
    candidates = [
        grid,
        grid[::-1, ::-1, :],
        grid[:, ::-1, :],
        grid[::-1, :, :],
        grid_t,
        grid_t[::-1, ::-1, :],
        grid_t[:, ::-1, :],
        grid_t[::-1, :, :],
    ]
    unique = []
    seen = set()
    for item in candidates:
        flat = item.reshape(-1, 2)
        key = tuple(np.round(flat.ravel(), 2))
        if key not in seen:
            seen.add(key)
            unique.append(flat.reshape(-1, 1, 2))
    return unique


def best_homography(src_corners, dst_corners, board_size=(7, 6)):
    best = None
    for order_id, src in enumerate(corner_orders(src_corners, board_size)):
        H, mask = cv2.findHomography(src, dst_corners, cv2.RANSAC, 3.0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(src, H)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - dst_corners.reshape(-1, 2), axis=1).mean())
        inliers = int(mask.sum()) if mask is not None else 0
        if best is None or err < best["err"]:
            best = {"H": H, "err": err, "inliers": inliers, "order_id": order_id, "src": src}
    return best


def idealize_board_corners(corners, board_size=(7, 6), square_px=None):
    """Rebuild destination corners as a perfect rectangle grid.

    Phone top-down photos have perspective, so detected squares are unequal.
    Forcing equal squares (true ground metric) keeps front/rear/left/right mats
    on the same BEV scale so seams line up better.
    """
    width, height = board_size
    grid = corners.reshape(height, width, 2).astype(np.float64)
    row_vec = (grid[:, -1] - grid[:, 0]).mean(axis=0)
    col_vec = (grid[-1, :] - grid[0, :]).mean(axis=0)
    row_u = row_vec / (np.linalg.norm(row_vec) + 1e-9)
    col_u = col_vec - np.dot(col_vec, row_u) * row_u
    col_norm = np.linalg.norm(col_u)
    if col_norm < 1e-9:
        col_u = np.array([-row_u[1], row_u[0]])
    else:
        col_u = col_u / col_norm
    # Keep orientation consistent with original col direction.
    if np.dot(col_u, col_vec) < 0:
        col_u = -col_u
    measured = 0.5 * (
        np.linalg.norm(row_vec) / max(width - 1, 1) + np.linalg.norm(col_vec) / max(height - 1, 1)
    )
    if square_px is None:
        square_px = measured
    center = grid.mean(axis=(0, 1))
    ideal = np.zeros((height, width, 2), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            ideal[row, col] = (
                center
                + (col - (width - 1) / 2.0) * square_px * row_u
                + (row - (height - 1) / 2.0) * square_px * col_u
            )
    return ideal.reshape(-1, 1, 2).astype(np.float32), float(square_px)


TOPDOWN_REGION_SETS = {
    # Prefer topdown_front_up.png (car nose toward TOP of image).
    # Do NOT auto-swap L/R: wrong board pairing can still get low numeric error.
    "front_up": {
        "front": [(0.20, 0.00, 0.80, 0.38)],
        "rear": [(0.20, 0.62, 0.80, 1.00)],
        "left": [(0.00, 0.25, 0.38, 0.75)],
        "right": [(0.62, 0.25, 1.00, 0.75)],
    },
}
DEFAULT_CAMERA_REGIONS = {
    "front": [(0.00, 0.00, 1.00, 1.00)],
    "left": [(0.00, 0.00, 1.00, 1.00)],
    "right": [(0.00, 0.00, 1.00, 1.00)],
    "rear": [(0.00, 0.00, 1.00, 1.00)],
}
SURROUND_NAME = {"front": "front", "left": "left", "right": "right", "rear": "back"}

# Manual orientation fix for cameras where the automatic corner-order search
# converges on a numerically-low-error but physically wrong correspondence
# (checkerboards have rotational/mirror symmetry, so reprojection error alone
# cannot always disambiguate). Values are visually-CCW degrees on the BEV
# canvas, confirmed by the user comparing warped output against the topdown
# reference. Applied as a rotation about the matched board's centroid so
# position is unaffected, only orientation.
MANUAL_ORIENTATION_FIX_DEG = {
    "left": 180.0,
    "rear": 90.0,
}


def rotation_about_point(angle_deg, pivot):
    """3x3 homogeneous rotation (visually CCW on a Y-down image) about pivot."""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array(
        [
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    px, py = pivot
    T1 = np.array([[1.0, 0.0, -px], [0.0, 1.0, -py], [0.0, 0.0, 1.0]])
    T2 = np.array([[1.0, 0.0, px], [0.0, 1.0, py], [0.0, 0.0, 1.0]])
    return T2 @ R @ T1


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def load_kd(camera_name):
    # Each camera now has its own individually-calibrated K/D; left/right/rear no
    # longer share one "usb_fisheye" set (see calibration_capture_smart.py).
    return np.load(CALIB / "captures" / camera_name / "camera_0_K.npy"), np.load(
        CALIB / "captures" / camera_name / "camera_0_D.npy"
    )


def _detect_board_once(gray, board_size=(7, 6)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for flags in (
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
    ):
        ok, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if ok:
            return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    if hasattr(cv2, "findChessboardCornersSB"):
        sb = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            sb |= cv2.CALIB_CB_EXHAUSTIVE
        if hasattr(cv2, "CALIB_CB_ACCURACY"):
            sb |= cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, board_size, flags=sb)
        if ok and corners is not None:
            return corners.astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    ok, corners = cv2.findChessboardCorners(
        blurred, board_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ok:
        return cv2.cornerSubPix(blurred, corners, (5, 5), (-1, -1), criteria)
    return None


def detect_board(gray, board_size=(7, 6)):
    corners = _detect_board_once(gray, board_size)
    if corners is not None:
        return corners
    # CSI / low-contrast boards often need local contrast boost.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return _detect_board_once(clahe, board_size)


def normalize_regions(regions):
    normalized = {}
    for camera_name, camera_regions in regions.items():
        if camera_regions and isinstance(camera_regions[0], (int, float)):
            camera_regions = [camera_regions]
        normalized[camera_name] = [tuple(map(float, region)) for region in camera_regions]
    return normalized


def load_region_config(path):
    if not path:
        return normalize_regions(TOPDOWN_REGION_SETS["front_up"]), normalize_regions(DEFAULT_CAMERA_REGIONS)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    topdown_sets = config.get("topdown_sets") or {}
    topdown_name = config.get("default_topdown_set", "front_up")
    if topdown_name not in topdown_sets:
        raise KeyError(f"Missing topdown set '{topdown_name}' in {path}")
    topdown_regions = normalize_regions(topdown_sets[topdown_name])
    camera_regions = normalize_regions(config.get("camera_regions", DEFAULT_CAMERA_REGIONS))
    return topdown_regions, camera_regions


def detect_board_in_region(image, region, board_size, gray=None):
    crop, (ox, oy) = crop_region(image, region)
    if crop.size == 0:
        return None
    if gray is None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_region(gray, region)[0]
    corners = detect_board(gray, board_size)
    if corners is None:
        return None
    corners[:, 0, 0] += ox
    corners[:, 0, 1] += oy
    return corners


def concat_boards(boards):
    return np.concatenate(boards, axis=0).astype(np.float32)


def iter_board_order_combinations(boards, board_size=(7, 6), max_combinations=1024):
    per_board_candidates = [corner_orders(board, board_size) for board in boards]
    combo_count = 1
    for candidates in per_board_candidates:
        combo_count *= len(candidates)
    if combo_count > max_combinations:
        raise RuntimeError(
            f"Too many corner-order combinations ({combo_count}) for {len(boards)} boards; "
            "reduce boards per camera or simplify region config."
        )
    for choice in itertools.product(*per_board_candidates):
        yield concat_boards(choice)


def best_homography_multi(src_boards, dst_boards, board_size=(7, 6)):
    best = None
    for src_order_id, src in enumerate(iter_board_order_combinations(src_boards, board_size)):
        for dst_order_id, dst in enumerate(iter_board_order_combinations(dst_boards, board_size)):
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is None:
                continue
            proj = cv2.perspectiveTransform(src, H)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - dst.reshape(-1, 2), axis=1).mean())
            inliers = int(mask.sum()) if mask is not None else 0
            if best is None or err < best["err"]:
                best = {
                    "H": H,
                    "err": err,
                    "inliers": inliers,
                    "src_order_id": src_order_id,
                    "dst_order_id": dst_order_id,
                }
    return best


def camera_mat_dst(K, D, frame_width, frame_height, focal_scale, size_scale, pinhole=False):
    if pinhole and size_scale == 1.0:
        # alpha=0.0 (crop to valid pixels) instead of 1.0 (keep every pixel) —
        # see bev_stitch.py SmartCamera.get_camera_mat_dst for why alpha=1.0
        # pinches/folds this wide-FOV lens's undistorted image.
        p, _roi = cv2.getOptimalNewCameraMatrix(
            K, D, (frame_width, frame_height), 0.0, (frame_width, frame_height)
        )
        p = p.astype(np.float64)
        p[0, 0] *= focal_scale
        p[1, 1] *= focal_scale
        return p
    dst = K.copy().astype(np.float64)
    dst[0, 0] *= focal_scale
    dst[1, 1] *= focal_scale
    dst[0, 2] = frame_width / 2.0 * size_scale
    dst[1, 2] = frame_height / 2.0 * size_scale
    return dst


def undistort_image(image, K, D, mode, frame_width, frame_height, focal_scale, size_scale):
    out_w = int(frame_width * size_scale)
    out_h = int(frame_height * size_scale)
    pinhole = mode != "fisheye"
    p = camera_mat_dst(K, D, frame_width, frame_height, focal_scale, size_scale, pinhole=pinhole)
    if mode == "fisheye":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), p, (out_w, out_h), cv2.CV_16SC2)
    else:
        map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), p, (out_w, out_h), cv2.CV_16SC2)
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def find_labeled_image(labeled_dir, camera_name):
    matches = sorted(labeled_dir.glob(f"{camera_name}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No image for {camera_name} in {labeled_dir}")
    # Prefer newest
    return max(matches, key=lambda p: p.stat().st_mtime)


def crop_region(image, region):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = region
    return image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)], (int(x0 * w), int(y0 * h))


def prepare_topdown(topdown, bev_width, bev_height, center=True):
    """Resize top-down to BEV canvas; optionally center-crop pad."""
    h, w = topdown.shape[:2]
    scale = min(bev_width / w, bev_height / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(topdown, (nw, nh))
    canvas = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    x0 = (bev_width - nw) // 2
    y0 = (bev_height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, (x0, y0, scale)


def main():
    parser = argparse.ArgumentParser(description="Compute surround H from chessboard + topdown.")
    parser.add_argument("--labeled-dir", default=str(DEFAULT_LABELED))
    parser.add_argument("--topdown", default=str(DEFAULT_TOPDOWN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--regions-json",
        default="",
        help="Optional JSON config for multi-board topdown/camera crop regions.",
    )
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--bev-width", type=int, default=1000)
    parser.add_argument("--bev-height", type=int, default=1000)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    parser.add_argument("--board-cols", type=int, default=7)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument(
        "--square-px",
        type=float,
        default=0.0,
        help="With --ideal-dst: force destination square size in BEV pixels (0 = auto).",
    )
    parser.add_argument(
        "--ideal-dst",
        action="store_true",
        help="Rebuild topdown board corners as equal squares (shared metric). Default off.",
    )
    parser.add_argument(
        "--no-ideal-dst",
        action="store_true",
        help="Deprecated alias; ideal-dst is already off by default.",
    )
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    topdown_path = Path(args.topdown)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board_size = (args.board_cols, args.board_rows)
    hardware = load_hardware()
    topdown_regions, camera_regions = load_region_config(args.regions_json)

    topdown_raw = cv2.imread(str(topdown_path))
    if topdown_raw is None:
        raise FileNotFoundError(topdown_path)
    topdown, (pad_x, pad_y, scale) = prepare_topdown(topdown_raw, args.bev_width, args.bev_height)
    cv2.imwrite(str(output_dir / "img_dst_topdown.jpg"), topdown)

    # Detect destination boards once; optionally idealize to equal square size.
    dst_boards = {}
    measured_squares = []
    topdown_gray = cv2.cvtColor(topdown_raw, cv2.COLOR_BGR2GRAY)
    for camera_name, regions in topdown_regions.items():
        camera_boards = []
        for region_id, region in enumerate(regions):
            dst_raw = detect_board_in_region(topdown_raw, region, board_size, gray=topdown_gray)
            if dst_raw is None:
                print(f"WARNING: no topdown board in region {camera_name}[{region_id}]")
                continue
            dst_bev = dst_raw.copy()
            dst_bev[:, 0, 0] = dst_raw[:, 0, 0] * scale + pad_x
            dst_bev[:, 0, 1] = dst_raw[:, 0, 1] * scale + pad_y
            _, measured = idealize_board_corners(dst_bev, board_size)
            measured_squares.append(measured)
            camera_boards.append(dst_bev)
        if camera_boards:
            dst_boards[camera_name] = camera_boards

    if args.square_px > 0:
        square_px = args.square_px
    elif measured_squares:
        square_px = float(np.mean(measured_squares))
    else:
        square_px = None

    use_ideal = args.ideal_dst and not args.no_ideal_dst
    if use_ideal and square_px is not None:
        print(f"Idealizing destination boards to square_px={square_px:.2f} (shared metric scale)")
        for camera_name, boards in list(dst_boards.items()):
            dst_boards[camera_name] = [
                idealize_board_corners(dst_bev, board_size, square_px=square_px)[0] for dst_bev in boards
            ]

    summary = {
        "topdown": str(topdown_path),
        "frame_size": [args.frame_width, args.frame_height],
        "bev_size": [args.bev_width, args.bev_height],
        "focal_scale": args.focal_scale,
        "size_scale": args.size_scale,
        "regions_json": str(args.regions_json) if args.regions_json else None,
        "ideal_dst": use_ideal,
        "square_px": square_px if use_ideal else None,
        "cameras": {},
    }

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    for camera_name in ("front", "left", "right", "rear"):
        mode = hardware[camera_name]["calibration_mode"]
        if mode == "fisheye":
            calib_mode = "fisheye"
        else:
            calib_mode = "normal"
        K, D = load_kd(camera_name)
        cam_scale = bev_size_scale(camera_name, args.size_scale)
        cam_focal_scale = bev_focal_scale(camera_name, args.focal_scale)
        src_path = find_labeled_image(labeled_dir, camera_name)
        raw = cv2.imread(str(src_path))
        if raw is None:
            raise RuntimeError(f"Cannot read {src_path}")
        if raw.shape[1] != args.frame_width or raw.shape[0] != args.frame_height:
            raw = cv2.resize(raw, (args.frame_width, args.frame_height))

        und = undistort_image(
            raw, K, D, calib_mode, args.frame_width, args.frame_height, cam_focal_scale, cam_scale
        )
        cv2.imwrite(str(debug_dir / f"{camera_name}_undist.jpg"), und)
        und_gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        raw_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        src_boards = []
        regions = camera_regions.get(camera_name, DEFAULT_CAMERA_REGIONS[camera_name])
        for region_id, region in enumerate(regions):
            src_corners = detect_board_in_region(und, region, board_size, gray=und_gray)
            source_kind = "undistorted image"
            if src_corners is None:
                src_raw = detect_board_in_region(raw, region, board_size, gray=raw_gray)
                if src_raw is not None:
                    pts = src_raw.reshape(-1, 1, 2).astype(np.float64)
                    p = camera_mat_dst(
                        K,
                        D,
                        args.frame_width,
                        args.frame_height,
                        cam_focal_scale,
                        cam_scale,
                        pinhole=(calib_mode != "fisheye"),
                    )
                    if calib_mode == "fisheye":
                        mapped = cv2.fisheye.undistortPoints(pts, K, D, np.eye(3), P=p)
                    else:
                        mapped = cv2.undistortPoints(pts, K, D, None, P=p)
                    src_corners = mapped.astype(np.float32)
                    source_kind = "RAW then mapped"
            if src_corners is None:
                print(f"{camera_name}: missing board in camera region {region_id}")
                continue
            src_boards.append(src_corners)
            print(f"{camera_name}: board {region_id} from {source_kind}")

        if not src_boards:
            print(f"{camera_name}: FAILED — no chessboard in camera image ({src_path.name})")
            summary["cameras"][camera_name] = {"ok": False, "reason": "no_src_corners"}
            continue

        if camera_name not in dst_boards:
            print(f"{camera_name}: FAILED — no usable topdown board match")
            summary["cameras"][camera_name] = {"ok": False, "reason": "no_dst_match"}
            continue

        board_count = min(len(src_boards), len(dst_boards[camera_name]))
        if board_count == 0:
            print(f"{camera_name}: FAILED — no common board pairs")
            summary["cameras"][camera_name] = {"ok": False, "reason": "no_common_board_pairs"}
            continue
        if len(src_boards) != len(dst_boards[camera_name]):
            print(
                f"{camera_name}: WARNING — using first {board_count} matched boards "
                f"(src={len(src_boards)} dst={len(dst_boards[camera_name])})"
            )
        src_boards = src_boards[:board_count]
        dst_camera_boards = dst_boards[camera_name][:board_count]

        best_overall = best_homography_multi(src_boards, dst_camera_boards, board_size)
        if best_overall is None:
            print(f"{camera_name}: FAILED — no usable topdown board match")
            summary["cameras"][camera_name] = {"ok": False, "reason": "no_dst_match"}
            continue
        best_overall["region_name"] = "multi_board"
        best_overall["board_count"] = board_count

        H = best_overall["H"]
        fix_deg = MANUAL_ORIENTATION_FIX_DEG.get(camera_name)
        if fix_deg:
            pivot = np.concatenate(dst_camera_boards, axis=0).reshape(-1, 2).mean(axis=0)
            H = rotation_about_point(fix_deg, pivot) @ H
            print(f"{camera_name}: applied manual orientation fix ({fix_deg:+.0f} deg about {pivot})")
        surround_name = SURROUND_NAME[camera_name]
        h_path = output_dir / f"camera_{surround_name}_H.npy"
        np.save(h_path, H)

        warped = cv2.warpPerspective(und, H, (args.bev_width, args.bev_height))
        cv2.imwrite(str(debug_dir / f"{camera_name}_warped.jpg"), warped)
        overlay = topdown.copy()
        mask_w = (warped > 0).any(axis=2)
        overlay[mask_w] = (overlay[mask_w].astype(np.float32) * 0.4 + warped[mask_w].astype(np.float32) * 0.6).astype(
            np.uint8
        )
        cv2.imwrite(str(debug_dir / f"{camera_name}_overlay.jpg"), overlay)

        print(
            f"{camera_name} -> {surround_name}: saved {h_path.name} "
            f"(err={best_overall['err']:.2f}px inliers={best_overall['inliers']} "
            f"src_order={best_overall['src_order_id']} dst_order={best_overall['dst_order_id']} "
            f"boards={board_count} region={best_overall['region_name']})"
        )
        summary["cameras"][camera_name] = {
            "ok": True,
            "surround_name": surround_name,
            "source_image": str(src_path),
            "H_file": str(h_path),
            "size_scale": cam_scale,
            "reproj_err_px": best_overall["err"],
            "inliers": best_overall["inliers"],
            "src_order_id": best_overall["src_order_id"],
            "dst_order_id": best_overall["dst_order_id"],
            "board_count": board_count,
            "region_name": best_overall["region_name"],
        }

    summary_path = output_dir / "bev_extrinsic_chessboard_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")

    ok_count = sum(1 for c in summary["cameras"].values() if c.get("ok"))
    if ok_count < 4:
        print(f"WARNING: only {ok_count}/4 cameras succeeded")
        sys.exit(2)


if __name__ == "__main__":
    main()
