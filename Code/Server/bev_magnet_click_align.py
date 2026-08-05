#!/usr/bin/env python3
"""Align front-left seam by clicking the same purple-magnet points in both views.

Workflow:
  1. Run preview_front_left_magnets.py, place strip in overlap, press s.
  2. Run this script; click matching points on FRONT then LEFT (same physical spots).
  3. Writes refined left H and a before/after BEV compare.

Keys while clicking:
  left-click = add point
  u = undo   n = finish current image   q = abort
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER))
from camera_devices import bev_focal_scale, bev_size_scale
from bev_stitch import SmartCamera

SURROUND_DATA = Path.home() / "CameraCalibration-test" / "SurroundBirdEyeView" / "data"
DEFAULT_EXT = SERVER / "calibration_patterns" / "bev_extrinsic_metric"
DEFAULT_OUT = SERVER / "bev_output" / "magnet_align"


def find_image(labeled_dir: Path, prefix: str) -> Path:
    matches = sorted(labeled_dir.glob(f"{prefix}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No {prefix}*.jpg in {labeled_dir}")
    return matches[0]


def click_points(window: str, image: np.ndarray, title: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    vis0 = image.copy()
    cv2.putText(vis0, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis0, "click pts  u=undo  n=next  q=abort", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        vis = vis0.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and pts:
            pts.pop()
        elif key == ord("n"):
            break
        elif key == ord("q"):
            cv2.destroyWindow(window)
            raise SystemExit("aborted")
    cv2.destroyWindow(window)
    return pts


def project_raw_points_to_bev(cam: SmartCamera, pts_raw: np.ndarray) -> np.ndarray:
    """Map raw pixel points -> undistorted -> BEV using cam maps/H."""
    map1, map2 = cam.undistort_maps
    # Inverse lookup on undistort maps (same idea as extrinsic tools).
    if map1.ndim == 3:
        src_x = map1[:, :, 0].astype(np.float32)
        src_y = map1[:, :, 1].astype(np.float32)
    else:
        src_x = map1.astype(np.float32)
        src_y = map2.astype(np.float32)

    und = []
    for x, y in pts_raw:
        dist = (src_x - float(x)) ** 2 + (src_y - float(y)) ** 2
        iy, ix = np.unravel_index(int(np.argmin(dist)), dist.shape)
        und.append([float(ix), float(iy)])
    und = np.array(und, dtype=np.float32).reshape(-1, 1, 2)
    bev = cv2.perspectiveTransform(und, cam.homography.astype(np.float64))
    return bev.reshape(-1, 2)


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 similarity (rotation+scale+translation) mapping src->dst."""
    assert len(src) == len(dst) and len(src) >= 2
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    src_c = src - sc
    dst_c = dst - dc
    # complex / umeyama for similarity
    n = len(src)
    var_s = (src_c**2).sum() / n
    cov = (dst_c.T @ src_c) / n
    u, _, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    scale = np.trace(r.T @ cov) / max(var_s, 1e-12)
    # Prefer near-rigid: clamp scale close to 1 for seam refine
    scale = float(np.clip(scale, 0.85, 1.15))
    t = dc - scale * (r @ sc)
    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = scale * r
    T[:2, 2] = t
    return T


def main():
    parser = argparse.ArgumentParser(description="Click purple magnets to refine left H vs front.")
    parser.add_argument("--labeled-dir", default=str(SERVER / "camera_labeled_corners"))
    parser.add_argument("--extrinsic-dir", default=str(DEFAULT_EXT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--apply", action="store_true", help="Copy refined left H into extrinsic-dir")
    parser.add_argument("--min-points", type=int, default=3)
    args = parser.parse_args()

    labeled = Path(args.labeled_dir)
    ext = Path(args.extrinsic_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Deploy metric H into surround data for SmartCamera load
    for name in ("front", "left", "right", "back"):
        src = ext / f"camera_{name}_H.npy"
        if src.exists():
            dst_dir = SURROUND_DATA / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / f"camera_{name}_H.npy")

    front_img = cv2.imread(str(find_image(labeled, "front")))
    left_img = cv2.imread(str(find_image(labeled, "left")))
    if front_img is None or left_img is None:
        raise RuntimeError("Cannot read front/left images")

    print("Click the SAME physical points on the purple strip.")
    print("Order must match: point1 front == point1 left, etc. (>=3 points).")
    front_pts = click_points("front click", front_img, "FRONT: click magnet points, then n")
    left_pts = click_points("left click", left_img, "LEFT: click SAME points in SAME order, then n")
    if len(front_pts) != len(left_pts) or len(front_pts) < args.min_points:
        raise SystemExit(f"Need >= {args.min_points} matched pairs; got front={len(front_pts)} left={len(left_pts)}")

    cam_f = SmartCamera("front", SURROUND_DATA, 640, 480, 1000, 1000, bev_focal_scale("front"), bev_size_scale("front"), True)
    cam_l = SmartCamera("left", SURROUND_DATA, 640, 480, 1000, 1000, bev_focal_scale("left"), bev_size_scale("left"), False)

    bev_f = project_raw_points_to_bev(cam_f, np.array(front_pts, np.float32))
    bev_l = project_raw_points_to_bev(cam_l, np.array(left_pts, np.float32))
    err0 = float(np.linalg.norm(bev_f - bev_l, axis=1).mean())
    print(f"BEV pair mean error before: {err0:.2f} px")

    T = estimate_similarity(bev_l, bev_f)  # map left BEV -> front BEV
    H_old = cam_l.homography.astype(np.float64)
    H_new = T @ H_old
    ones = np.ones((len(bev_l), 1), np.float64)
    bev_l2 = (T @ np.hstack([bev_l.astype(np.float64), ones]).T).T[:, :2]
    err1 = float(np.linalg.norm(bev_f - bev_l2, axis=1).mean())
    print(f"BEV pair mean error after:  {err1:.2f} px")
    print(f"SE2/scale refine:\n{T}")

    h_path = out / "camera_left_H_magnet.npy"
    np.save(str(h_path), H_new)
    meta = {
        "pair": "front-left",
        "front_image": str(find_image(labeled, "front")),
        "left_image": str(find_image(labeled, "left")),
        "front_pts": front_pts,
        "left_pts": left_pts,
        "bev_error_before_px": err0,
        "bev_error_after_px": err1,
        "T_bev": T.tolist(),
        "H_out": str(h_path),
    }
    (out / "front_left_magnet_click.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Visualize
    canvas = np.zeros((1000, 1000, 3), np.uint8)
    for (x, y) in bev_f:
        cv2.circle(canvas, (int(x), int(y)), 7, (0, 0, 255), -1)
    for (x, y) in bev_l:
        cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 0), 2)
    for (x, y) in bev_l2:
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), -1)
    for a, b in zip(bev_f, bev_l2):
        cv2.line(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 1)
    cv2.putText(canvas, "red=front  cyan=left before  green=left after", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(out / "front_left_click_bev.jpg"), canvas)
    print(f"saved {h_path}")
    print(f"saved {out / 'front_left_click_bev.jpg'}")

    if args.apply:
        dst = ext / "camera_left_H.npy"
        shutil.copy2(h_path, dst)
        print(f"applied -> {dst}")


if __name__ == "__main__":
    main()
