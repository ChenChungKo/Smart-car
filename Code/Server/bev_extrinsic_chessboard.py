#!/usr/bin/env python3
"""Extrinsic H from labeled camera images + top-down reference (chessboard pairs)."""

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
DEFAULT_LABELED = SERVER / "camera_labeled"
DEFAULT_TOPDOWN = CALIB / "extrinsic_ref" / "topdown_front_up.png"
DEFAULT_OUTPUT = CALIB / "bev_extrinsic"

# Car front faces LEFT in the top-down photo.
# Regions selected via TOPDOWN_REGION_SETS below.
def corner_orders(corners, board_size=(7, 6)):
    """Generate plausible OpenCV chessboard corner order variants."""
    w, h = board_size
    c = corners.reshape(-1, 2).astype(np.float32)
    candidates = [
        c,
        c[::-1],
        c.reshape(h, w, 2)[:, ::-1, :].reshape(-1, 2),
        c.reshape(h, w, 2)[::-1, :, :].reshape(-1, 2),
        c.reshape(h, w, 2)[::-1, ::-1, :].reshape(-1, 2),
    ]
    unique = []
    seen = set()
    for item in candidates:
        key = tuple(np.round(item.ravel(), 2))
        if key not in seen:
            seen.add(key)
            unique.append(item.reshape(-1, 1, 2))
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


TOPDOWN_REGION_SETS = {
    # Prefer topdown_front_up.png (car nose toward TOP of image).
    # Do NOT auto-swap L/R: wrong board pairing can still get low numeric error.
    "front_up": {
        "front": (0.20, 0.00, 0.80, 0.38),
        "rear": (0.20, 0.62, 0.80, 1.00),
        "left": (0.00, 0.25, 0.38, 0.75),
        "right": (0.62, 0.25, 1.00, 0.75),
    },
}
SURROUND_NAME = {"front": "front", "left": "left", "right": "right", "rear": "back"}


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as handle:
        return json.load(handle)


def load_kd(camera_name):
    if camera_name in ("left", "right", "rear"):
        return np.load(CALIB / "shared" / "usb_fisheye_K.npy"), np.load(CALIB / "shared" / "usb_fisheye_D.npy")
    return np.load(CALIB / "captures" / "front" / "camera_0_K.npy"), np.load(
        CALIB / "captures" / "front" / "camera_0_D.npy"
    )


def detect_board(gray, board_size=(7, 6)):
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


def camera_mat_dst(K, frame_width, frame_height, focal_scale, size_scale):
    dst = K.copy().astype(np.float64)
    dst[0, 0] *= focal_scale
    dst[1, 1] *= focal_scale
    dst[0, 2] = frame_width / 2.0 * size_scale
    dst[1, 2] = frame_height / 2.0 * size_scale
    return dst


def undistort_image(image, K, D, mode, frame_width, frame_height, focal_scale, size_scale):
    out_w = int(frame_width * size_scale)
    out_h = int(frame_height * size_scale)
    p = camera_mat_dst(K, frame_width, frame_height, focal_scale, size_scale)
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
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--bev-width", type=int, default=1000)
    parser.add_argument("--bev-height", type=int, default=1000)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    parser.add_argument("--board-cols", type=int, default=7)
    parser.add_argument("--board-rows", type=int, default=6)
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    topdown_path = Path(args.topdown)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    board_size = (args.board_cols, args.board_rows)
    hardware = load_hardware()

    topdown_raw = cv2.imread(str(topdown_path))
    if topdown_raw is None:
        raise FileNotFoundError(topdown_path)
    topdown, (pad_x, pad_y, scale) = prepare_topdown(topdown_raw, args.bev_width, args.bev_height)
    cv2.imwrite(str(output_dir / "img_dst_topdown.jpg"), topdown)

    summary = {
        "topdown": str(topdown_path),
        "frame_size": [args.frame_width, args.frame_height],
        "bev_size": [args.bev_width, args.bev_height],
        "focal_scale": args.focal_scale,
        "size_scale": args.size_scale,
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
        src_path = find_labeled_image(labeled_dir, camera_name)
        raw = cv2.imread(str(src_path))
        if raw is None:
            raise RuntimeError(f"Cannot read {src_path}")
        if raw.shape[1] != args.frame_width or raw.shape[0] != args.frame_height:
            raw = cv2.resize(raw, (args.frame_width, args.frame_height))

        und = undistort_image(
            raw, K, D, calib_mode, args.frame_width, args.frame_height, args.focal_scale, args.size_scale
        )
        cv2.imwrite(str(debug_dir / f"{camera_name}_undist.jpg"), und)
        src_corners = detect_board(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), board_size)
        if src_corners is None:
            # retry on raw
            src_corners = detect_board(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), board_size)
            if src_corners is not None:
                # map raw corners via undistortPoints roughly into scaled image
                pts = src_corners.reshape(-1, 1, 2).astype(np.float64)
                p = camera_mat_dst(K, args.frame_width, args.frame_height, args.focal_scale, args.size_scale)
                if calib_mode == "fisheye":
                    mapped = cv2.fisheye.undistortPoints(pts, K, D, np.eye(3), P=p)
                else:
                    mapped = cv2.undistortPoints(pts, K, D, None, P=p)
                src_corners = mapped.astype(np.float32)
                print(f"{camera_name}: corners from RAW then mapped")
            else:
                print(f"{camera_name}: FAILED — no chessboard in camera image ({src_path.name})")
                summary["cameras"][camera_name] = {"ok": False, "reason": "no_src_corners"}
                continue
        else:
            print(f"{camera_name}: corners from undistorted image")

        best_overall = None
        for region_name, regions in TOPDOWN_REGION_SETS.items():
            region = regions[camera_name]
            crop, (ox, oy) = crop_region(topdown_raw, region)
            dst_local = detect_board(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), board_size)
            if dst_local is None:
                continue
            dst_raw = dst_local.copy()
            dst_raw[:, 0, 0] += ox
            dst_raw[:, 0, 1] += oy
            dst_bev = dst_raw.copy()
            dst_bev[:, 0, 0] = dst_raw[:, 0, 0] * scale + pad_x
            dst_bev[:, 0, 1] = dst_raw[:, 0, 1] * scale + pad_y
            candidate = best_homography(src_corners, dst_bev, board_size)
            if candidate is None:
                continue
            candidate["region_name"] = region_name
            candidate["dst_bev"] = dst_bev
            if best_overall is None or candidate["err"] < best_overall["err"]:
                best_overall = candidate

        if best_overall is None:
            print(f"{camera_name}: FAILED — no usable topdown board match")
            summary["cameras"][camera_name] = {"ok": False, "reason": "no_dst_match"}
            continue

        H = best_overall["H"]
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
            f"order={best_overall['order_id']} region={best_overall['region_name']})"
        )
        summary["cameras"][camera_name] = {
            "ok": True,
            "surround_name": surround_name,
            "source_image": str(src_path),
            "H_file": str(h_path),
            "reproj_err_px": best_overall["err"],
            "inliers": best_overall["inliers"],
            "order_id": best_overall["order_id"],
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
