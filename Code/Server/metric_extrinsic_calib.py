#!/usr/bin/env python3
"""Metric ground-point extrinsic calibration (no top-down photo required).

Literature-style IPM extrinsic: click known mat-grid points (mm) in each camera
image, undistort with current K/D, estimate H by RANSAC, then stitch.

Typical flow:
  1) Fill car_* corners in metric_extrinsic_config.json (mm on the mat rulers)
  2) python3 metric_extrinsic_calib.py suggest-targets
  3) Recapture four labeled images (no board needed)
  4) python3 metric_extrinsic_calib.py click --camera front
     ... repeat for left/right/rear
  5) python3 metric_extrinsic_calib.py solve
  6) python3 bev_deploy.py --extrinsic-dir calibration_patterns/bev_extrinsic_metric_click
  7) python3 bev_stitch.py --blend --feather 40 --output-dir bev_output/metric_click_test
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
DEFAULT_CONFIG = SERVER / "metric_extrinsic_config.json"
DEFAULT_LABELED = SERVER / "camera_labeled"
DEFAULT_OUTPUT = CALIB / "bev_extrinsic_metric_click"

CAMERA_ORDER = ("front", "left", "right", "rear")
SURROUND_NAMES = {"front": "front", "left": "left", "right": "right", "rear": "back"}

sys.path.insert(0, str(SERVER))
from bev_extrinsic import undistort_points, load_kd  # noqa: E402


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_hardware():
    return load_json(HARDWARE)


def find_labeled(labeled_dir: Path, camera_name: str) -> Path:
    matches = sorted(labeled_dir.glob(f"{camera_name}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No labeled image for {camera_name} in {labeled_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def require_car_corners(cfg):
    keys = (
        "car_front_left_mm",
        "car_front_right_mm",
        "car_rear_left_mm",
        "car_rear_right_mm",
    )
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise SystemExit(
            "請先在 metric_extrinsic_config.json 填入車身四角 mm：\n"
            + "\n".join(f"  - {k}" for k in missing)
            + "\n座標系：桌墊左上為 (0,0)，+X 沿長邊向右，+Y 沿短邊向下。"
        )
    return {k: np.array(cfg[k], dtype=np.float64) for k in keys}


def car_axis_from_corners(corners):
    fl, fr, rl, rr = (
        corners["car_front_left_mm"],
        corners["car_front_right_mm"],
        corners["car_rear_left_mm"],
        corners["car_rear_right_mm"],
    )
    front_mid = 0.5 * (fl + fr)
    rear_mid = 0.5 * (rl + rr)
    left_mid = 0.5 * (fl + rl)
    right_mid = 0.5 * (fr + rr)
    center = 0.25 * (fl + fr + rl + rr)
    forward = front_mid - rear_mid
    right = right_mid - left_mid
    # Unit axes in mat mm
    f_norm = np.linalg.norm(forward)
    r_norm = np.linalg.norm(right)
    if f_norm < 1e-6 or r_norm < 1e-6:
        raise SystemExit("車身四角幾乎共線，請重新量測。")
    forward = forward / f_norm
    right = right / r_norm
    return center, forward, right, f_norm, r_norm


def clamp_mm(pt, map_w, map_h, margin=20.0):
    return np.array(
        [
            float(np.clip(pt[0], margin, map_w - margin)),
            float(np.clip(pt[1], margin, map_h - margin)),
        ]
    )


def suggest_targets_for_camera(camera, center, forward, right, map_w, map_h, step):
    """Generate ~8–12 ground targets in each camera's expected FOV region."""
    # Offsets in mm relative to car center along car axes (forward, right).
    recipes = {
        "front": [
            (120, -180),
            (120, -60),
            (120, 60),
            (120, 180),
            (220, -120),
            (220, 0),
            (220, 120),
            (320, -60),
            (320, 60),
        ],
        "rear": [
            (-120, -180),
            (-120, -60),
            (-120, 60),
            (-120, 180),
            (-220, -120),
            (-220, 0),
            (-220, 120),
            (-320, -60),
            (-320, 60),
        ],
        "left": [
            (150, -120),
            (50, -120),
            (-50, -120),
            (-150, -120),
            (100, -220),
            (0, -220),
            (-100, -220),
            (50, -300),
            (-50, -300),
        ],
        "right": [
            (150, 120),
            (50, 120),
            (-50, 120),
            (-150, 120),
            (100, 220),
            (0, 220),
            (-100, 220),
            (50, 300),
            (-50, 300),
        ],
    }
    targets = []
    for df, dr in recipes[camera]:
        # Snap to nearest mat grid intersection for easier clicking.
        world = center + df * forward + dr * right
        snapped = np.array(
            [
                round(world[0] / step) * step,
                round(world[1] / step) * step,
            ],
            dtype=np.float64,
        )
        snapped = clamp_mm(snapped, map_w, map_h)
        key = (int(snapped[0]), int(snapped[1]))
        if key not in {(int(t[0]), int(t[1])) for t in targets}:
            targets.append([float(snapped[0]), float(snapped[1])])
    return targets


def cmd_set_car(args):
    cfg = load_json(Path(args.config))
    cfg["car_front_left_mm"] = [args.fl_x, args.fl_y]
    cfg["car_front_right_mm"] = [args.fr_x, args.fr_y]
    cfg["car_rear_left_mm"] = [args.rl_x, args.rl_y]
    cfg["car_rear_right_mm"] = [args.rr_x, args.rr_y]
    save_json(Path(args.config), cfg)
    print(f"Saved car corners to {args.config}")


def cmd_suggest_targets(args):
    cfg = load_json(Path(args.config))
    corners = require_car_corners(cfg)
    center, forward, right, length, width = car_axis_from_corners(corners)
    map_w = cfg["map_width_mm"]
    map_h = cfg["map_height_mm"]
    step = cfg.get("grid_step_mm", 100)

    cfg["car_center_mm"] = [float(center[0]), float(center[1])]
    cfg["car_length_mm"] = float(length)
    cfg["car_width_mm"] = float(width)
    cfg["suggested_targets"] = {}
    for camera in CAMERA_ORDER:
        targets = suggest_targets_for_camera(camera, center, forward, right, map_w, map_h, step)
        cfg["suggested_targets"][camera] = targets
        print(f"{camera}: {len(targets)} targets")
        for i, (x, y) in enumerate(targets, 1):
            print(f"  {i:02d}: ({x:.0f}, {y:.0f}) mm")
    save_json(Path(args.config), cfg)
    print(f"Updated {args.config}")


def draw_click_ui(image, targets, clicked, current_idx):
    display = image.copy()
    for i, (x, y) in enumerate(clicked):
        cv2.circle(display, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(
            display,
            str(i + 1),
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    if current_idx < len(targets):
        tx, ty = targets[current_idx]
        msg = f"Click target {current_idx + 1}/{len(targets)}: mat ({tx:.0f}, {ty:.0f}) mm"
        color = (0, 255, 255)
    else:
        msg = "All targets clicked. Enter=save  r=reset  q=quit  u=undo"
        color = (0, 255, 0)
    cv2.putText(display, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(
        display,
        "Keys: Enter=save  u=undo  r=reset  q=quit",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return display


def cmd_click(args):
    cfg = load_json(Path(args.config))
    if "suggested_targets" not in cfg or args.camera not in cfg["suggested_targets"]:
        raise SystemExit("請先執行: python3 metric_extrinsic_calib.py suggest-targets")

    labeled_dir = Path(args.labeled_dir)
    image_path = find_labeled(labeled_dir, args.camera)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read {image_path}")
    if image.shape[1] != args.frame_width or image.shape[0] != args.frame_height:
        image = cv2.resize(image, (args.frame_width, args.frame_height))

    targets = [list(map(float, p)) for p in cfg["suggested_targets"][args.camera]]
    if args.max_points > 0:
        targets = targets[: args.max_points]
    clicked = []
    window = f"metric-click-{args.camera}"

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(clicked) >= len(targets):
            return
        clicked.append([float(x), float(y)])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(f"{args.camera}: image={image_path.name}")
    print("依序點選畫面中對應的桌墊格點交點（黃字顯示目標 mm）。")

    while True:
        display = draw_click_ui(image, targets, clicked, len(clicked))
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter / Space
            if len(clicked) < 4:
                print(f"至少需要 4 點，目前 {len(clicked)}")
                continue
            break
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise SystemExit("Cancelled")
        if key == ord("r"):
            clicked.clear()
        if key == ord("u") and clicked:
            clicked.pop()

    cv2.destroyWindow(window)
    n = min(len(clicked), len(targets))
    points = []
    for i in range(n):
        points.append(
            {
                "src_px": [float(clicked[i][0]), float(clicked[i][1])],
                "dst_mm": [float(targets[i][0]), float(targets[i][1])],
            }
        )
    cfg["cameras"][args.camera]["image"] = image_path.name
    cfg["cameras"][args.camera]["points"] = points
    save_json(Path(args.config), cfg)
    print(f"Saved {n} points for {args.camera} -> {args.config}")


def mm_to_bev(points_mm, map_w, map_h, bev_w, bev_h):
    pts = np.array(points_mm, dtype=np.float32)
    out = pts.copy()
    out[:, 0] = pts[:, 0] / map_w * bev_w
    out[:, 1] = pts[:, 1] / map_h * bev_h
    return out


def cmd_solve(args):
    cfg = load_json(Path(args.config))
    hardware = load_hardware()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    map_w = cfg["map_width_mm"]
    map_h = cfg["map_height_mm"]
    summary = {
        "method": "metric_ground_points_ransac",
        "config": str(args.config),
        "frame_size": [args.frame_width, args.frame_height],
        "bev_size": [args.bev_width, args.bev_height],
        "focal_scale": args.focal_scale,
        "size_scale": args.size_scale,
        "cameras": {},
    }

    labeled_dir = Path(args.labeled_dir)
    ok_count = 0
    for camera in CAMERA_ORDER:
        points = cfg["cameras"][camera].get("points") or []
        if len(points) < 4:
            print(f"{camera}: FAILED — need >=4 points, got {len(points)}")
            summary["cameras"][camera] = {"ok": False, "reason": "too_few_points"}
            continue

        mode = hardware[camera]["calibration_mode"]
        K, D = load_kd(camera)
        src_raw = [p["src_px"] for p in points]
        dst_mm = [p["dst_mm"] for p in points]
        src_undist, warnings = undistort_points(
            src_raw,
            K,
            D,
            mode,
            args.frame_width,
            args.frame_height,
            args.focal_scale,
            args.size_scale,
        )
        dst_bev = mm_to_bev(dst_mm, map_w, map_h, args.bev_width, args.bev_height)

        H, mask = cv2.findHomography(src_undist, dst_bev, cv2.RANSAC, args.ransac_thresh)
        if H is None:
            print(f"{camera}: FAILED — findHomography returned None")
            summary["cameras"][camera] = {"ok": False, "reason": "homography_none"}
            continue

        proj = cv2.perspectiveTransform(src_undist.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst_bev, axis=1)
        inliers = int(mask.sum()) if mask is not None else len(points)
        mean_err = float(err.mean())
        inlier_err = float(err[mask.ravel() > 0].mean()) if inliers else mean_err

        surround = SURROUND_NAMES[camera]
        h_path = output_dir / f"camera_{surround}_H.npy"
        np.save(h_path, H)

        # Debug overlay: warp labeled image
        image_name = cfg["cameras"][camera].get("image") or ""
        image_path = labeled_dir / image_name if image_name else find_labeled(labeled_dir, camera)
        raw = cv2.imread(str(image_path))
        if raw is not None:
            if raw.shape[1] != args.frame_width or raw.shape[0] != args.frame_height:
                raw = cv2.resize(raw, (args.frame_width, args.frame_height))
            from bev_extrinsic import build_undistort_maps

            map1, map2 = build_undistort_maps(
                K, D, mode, args.frame_width, args.frame_height, args.focal_scale, args.size_scale
            )
            und = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
            warped = cv2.warpPerspective(und, H, (args.bev_width, args.bev_height))
            canvas = np.zeros((args.bev_height, args.bev_width, 3), dtype=np.uint8)
            # Draw mat grid for visual check
            step = cfg.get("grid_step_mm", 100)
            for x_mm in range(0, int(map_w) + 1, int(step)):
                x = int(x_mm / map_w * args.bev_width)
                cv2.line(canvas, (x, 0), (x, args.bev_height - 1), (40, 40, 40), 1)
            for y_mm in range(0, int(map_h) + 1, int(step)):
                y = int(y_mm / map_h * args.bev_height)
                cv2.line(canvas, (0, y), (args.bev_width - 1, y), (40, 40, 40), 1)
            mask_w = (warped > 0).any(axis=2)
            canvas[mask_w] = warped[mask_w]
            for p_dst, e, inl in zip(dst_bev, err, (mask.ravel() if mask is not None else [1] * len(err))):
                color = (0, 255, 0) if inl else (0, 0, 255)
                cv2.circle(canvas, (int(p_dst[0]), int(p_dst[1])), 5, color, -1)
            cv2.imwrite(str(debug_dir / f"{camera}_warped.jpg"), canvas)

        print(
            f"{camera} -> {surround}: err_mean={mean_err:.2f}px "
            f"inlier_err={inlier_err:.2f}px inliers={inliers}/{len(points)}"
        )
        if warnings:
            print(f"  undistort warnings: {warnings}")
        summary["cameras"][camera] = {
            "ok": True,
            "surround_name": surround,
            "H_file": str(h_path),
            "point_count": len(points),
            "inliers": inliers,
            "reproj_err_mean_px": mean_err,
            "reproj_err_inlier_mean_px": inlier_err,
            "undistort_warnings": warnings,
        }
        ok_count += 1

    summary_path = output_dir / "metric_extrinsic_summary.json"
    save_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    if ok_count < 4:
        print(f"WARNING: only {ok_count}/4 cameras succeeded")
        sys.exit(2)


def build_parser():
    parser = argparse.ArgumentParser(description="Metric ground-point extrinsic calibration.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="command", required=True)

    p_car = sub.add_parser("set-car", help="Write car four-corner mm into config.")
    p_car.add_argument("--fl-x", type=float, required=True)
    p_car.add_argument("--fl-y", type=float, required=True)
    p_car.add_argument("--fr-x", type=float, required=True)
    p_car.add_argument("--fr-y", type=float, required=True)
    p_car.add_argument("--rl-x", type=float, required=True)
    p_car.add_argument("--rl-y", type=float, required=True)
    p_car.add_argument("--rr-x", type=float, required=True)
    p_car.add_argument("--rr-y", type=float, required=True)
    p_car.set_defaults(func=cmd_set_car)

    p_sug = sub.add_parser("suggest-targets", help="Generate click targets from car corners.")
    p_sug.set_defaults(func=cmd_suggest_targets)

    p_click = sub.add_parser("click", help="Click suggested targets for one camera.")
    p_click.add_argument("--camera", choices=list(CAMERA_ORDER), required=True)
    p_click.add_argument("--labeled-dir", default=str(DEFAULT_LABELED))
    p_click.add_argument("--frame-width", type=int, default=640)
    p_click.add_argument("--frame-height", type=int, default=480)
    p_click.add_argument("--max-points", type=int, default=0, help="0 = all suggested targets")
    p_click.set_defaults(func=cmd_click)

    p_solve = sub.add_parser("solve", help="Estimate H for all cameras and save.")
    p_solve.add_argument("--labeled-dir", default=str(DEFAULT_LABELED))
    p_solve.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p_solve.add_argument("--frame-width", type=int, default=640)
    p_solve.add_argument("--frame-height", type=int, default=480)
    p_solve.add_argument("--bev-width", type=int, default=1000)
    p_solve.add_argument("--bev-height", type=int, default=1000)
    p_solve.add_argument("--focal-scale", type=float, default=1.0)
    p_solve.add_argument("--size-scale", type=float, default=2.0)
    p_solve.add_argument("--ransac-thresh", type=float, default=3.0)
    p_solve.set_defaults(func=cmd_solve)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
