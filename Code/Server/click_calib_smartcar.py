#!/usr/bin/env python3
"""Click-Calib-style extrinsic refinement for Smart Car OpenCV K/D/H pipeline.

Official Click-Calib (WoodScape radial_poly + 6DoF pose) is under click_calib_upstream/.
This adapter keeps the same idea — click corresponding ground points in adjacent
camera overlaps, minimize ground/BEV distance — but works with our existing
OpenCV intrinsics + BEV homographies.

Pairs (same as Click-Calib):
  front-left, front-right, rear-left, rear-right

Typical flow:
  1) python3 click_calib_smartcar.py --click
  2) python3 click_calib_smartcar.py --optimize
     (constrained SE2 in BEV — not free affine, which can collapse views)
  3) python3 bev_deploy.py --extrinsic-dir calibration_patterns/bev_extrinsic_click
  4) python3 bev_stitch.py --auto-car-size
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
DEFAULT_EXTRINSIC = CALIB / "bev_extrinsic"
DEFAULT_OUTPUT = CALIB / "bev_extrinsic_click"
DEFAULT_KEYPOINTS = CALIB / "click_calib_keypoints.json"

sys.path.insert(0, str(SERVER))
from camera_devices import bev_size_scale  # noqa: E402
from bev_extrinsic_chessboard import camera_mat_dst as chessboard_camera_mat_dst  # noqa: E402

CAMERA_ORDER = ("front", "left", "right", "rear")
SURROUND_NAMES = {"front": "front", "left": "left", "right": "right", "rear": "back"}
PAIR_SPECS = (
    ("front_left", "front", "left"),
    ("front_right", "front", "right"),
    ("rear_left", "rear", "left"),
    ("rear_right", "rear", "right"),
)


def load_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_kd(camera_name: str):
    if camera_name in ("left", "right", "rear"):
        return np.load(CALIB / "shared" / "usb_fisheye_K.npy"), np.load(CALIB / "shared" / "usb_fisheye_D.npy")
    return np.load(CALIB / "captures" / "front" / "camera_0_K.npy"), np.load(
        CALIB / "captures" / "front" / "camera_0_D.npy"
    )


def find_labeled(labeled_dir: Path, camera_name: str) -> Path:
    matches = sorted(labeled_dir.glob(f"{camera_name}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No labeled image for {camera_name} in {labeled_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_image(path: Path, width: int, height: int):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height))
    return image


def load_homographies(extrinsic_dir: Path):
    Hs = {}
    for camera in CAMERA_ORDER:
        surround = SURROUND_NAMES[camera]
        path = extrinsic_dir / f"camera_{surround}_H.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing initial H: {path}")
        Hs[camera] = np.load(path).astype(np.float64)
    return Hs


def raw_to_undistorted(points_xy, camera_name, frame_width, frame_height, focal_scale, size_scale):
    """Map raw pixels into the same undistorted canvas used when H was estimated."""
    hardware = load_json(HARDWARE)
    mode = hardware[camera_name]["calibration_mode"]
    K, D = load_kd(camera_name)
    cam_scale = bev_size_scale(camera_name, size_scale)
    pinhole = mode != "fisheye"
    p = chessboard_camera_mat_dst(
        K, D, frame_width, frame_height, focal_scale, cam_scale, pinhole=pinhole
    )
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    if mode == "fisheye":
        mapped = cv2.fisheye.undistortPoints(pts, K, D, np.eye(3), P=p)
    else:
        mapped = cv2.undistortPoints(pts, K, D, None, P=p)
    return mapped.reshape(-1, 2).astype(np.float32), cam_scale, []


def project_to_bev(points_xy, H):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def affine_matrix(params):
    """Full 2D affine [a,b,c,d,tx,ty] → 3x3. Prefer se2_matrix for constrained refine."""
    a, b, c, d, tx, ty = params
    return np.array([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def se2_matrix(params):
    """Similarity/SE2-like: [scale, theta_rad, tx, ty] → 3x3 (no shear, no collapse)."""
    scale, theta, tx, ty = params
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def se2_to_affine_list(params):
    """Pack SE2 params into the 6-vector affine form used in summaries."""
    M = se2_matrix(params)
    return [float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1]), float(M[0, 2]), float(M[1, 2])]


def apply_affine_bev(points_bev, params):
    A = affine_matrix(params) if len(params) == 6 else se2_matrix(params)
    return project_to_bev(points_bev, A)


def pair_mean_distance(points_a, points_b):
    return float(np.linalg.norm(points_a - points_b, axis=1).mean()) if len(points_a) else 0.0


def click_pair(img_a, img_b, title_a, title_b, existing=None):
    """Side-by-side clicker. Click left then right alternately (same index = same world point)."""
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    scale_a = 720 / max(ha, 1)
    scale_b = 720 / max(hb, 1)
    scale = min(1.0, scale_a, scale_b, 900 / max(wa + wb, 1))
    disp_a = cv2.resize(img_a, (int(wa * scale), int(ha * scale)))
    disp_b = cv2.resize(img_b, (int(wb * scale), int(hb * scale)))
    gap = 8
    canvas_h = max(disp_a.shape[0], disp_b.shape[0]) + 40
    canvas_w = disp_a.shape[1] + disp_b.shape[1] + gap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[40 : 40 + disp_a.shape[0], : disp_a.shape[1]] = disp_a
    canvas[40 : 40 + disp_b.shape[0], disp_a.shape[1] + gap :] = disp_b

    pts_a = []
    pts_b = []
    if existing:
        pts_a = [list(map(float, p)) for p in existing.get(title_a, [])]
        pts_b = [list(map(float, p)) for p in existing.get(title_b, [])]

    state = {"next": "a" if len(pts_a) == len(pts_b) else ("b" if len(pts_a) > len(pts_b) else "a")}

    def redraw():
        view = canvas.copy()
        cv2.putText(
            view,
            f"{title_a} | {title_b}  next={state['next'].upper()}  "
            f"n={min(len(pts_a), len(pts_b))}  Enter=save  r=reset  u=undo  q=skip",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for idx, (pa, pb) in enumerate(zip(pts_a, pts_b), start=1):
            qa = (int(pa[0] * scale), int(pa[1] * scale) + 40)
            qb = (int(pb[0] * scale) + disp_a.shape[1] + gap, int(pb[1] * scale) + 40)
            cv2.circle(view, qa, 4, (0, 0, 255), -1)
            cv2.circle(view, qb, 4, (0, 0, 255), -1)
            cv2.putText(view, str(idx), (qa[0] + 4, qa[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 100), 1)
            cv2.putText(view, str(idx), (qb[0] + 4, qb[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 100), 1)
        # unfinished point on one side
        if len(pts_a) > len(pts_b):
            pa = pts_a[-1]
            qa = (int(pa[0] * scale), int(pa[1] * scale) + 40)
            cv2.circle(view, qa, 5, (0, 255, 255), 2)
        elif len(pts_b) > len(pts_a):
            pb = pts_b[-1]
            qb = (int(pb[0] * scale) + disp_a.shape[1] + gap, int(pb[1] * scale) + 40)
            cv2.circle(view, qb, 5, (0, 255, 255), 2)
        return view

    window = f"click-calib-{title_a}-{title_b}"

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y < 40:
            return
        if x < disp_a.shape[1]:
            if state["next"] != "a":
                return
            ox = x / scale
            oy = (y - 40) / scale
            if 0 <= ox < wa and 0 <= oy < ha:
                pts_a.append([float(ox), float(oy)])
                state["next"] = "b"
        elif x >= disp_a.shape[1] + gap:
            if state["next"] != "b":
                return
            ox = (x - disp_a.shape[1] - gap) / scale
            oy = (y - 40) / scale
            if 0 <= ox < wb and 0 <= oy < hb:
                pts_b.append([float(ox), float(oy)])
                state["next"] = "a"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter / Space
            n = min(len(pts_a), len(pts_b))
            if n == 0:
                print("No complete pairs yet — keep clicking or press q to skip.")
                continue
            pts_a[:] = pts_a[:n]
            pts_b[:] = pts_b[:n]
            break
        if key in (27, ord("q")):
            pts_a.clear()
            pts_b.clear()
            break
        if key == ord("r"):
            pts_a.clear()
            pts_b.clear()
            state["next"] = "a"
        if key == ord("u"):
            if state["next"] == "b" and pts_a:
                pts_a.pop()
                state["next"] = "a"
            elif state["next"] == "a" and pts_b:
                pts_b.pop()
                state["next"] = "b"
            elif pts_a and pts_b and len(pts_a) == len(pts_b):
                pts_a.pop()
                pts_b.pop()
    cv2.destroyWindow(window)
    return {title_a: pts_a, title_b: pts_b}


def run_click(args):
    labeled_dir = Path(args.labeled_dir)
    keypoints = {"frame_size": [args.frame_width, args.frame_height], "pairs": {}}
    existing = {}
    if Path(args.keypoints).exists():
        existing = load_json(Path(args.keypoints)).get("pairs", {})

    for pair_name, cam_a, cam_b in PAIR_SPECS:
        print(f"\n=== Pair {pair_name}: click matching ground points ===")
        print("Tip: use green-mat grid intersections visible in BOTH cameras. Aim for >=8 pairs.")
        img_a = load_image(find_labeled(labeled_dir, cam_a), args.frame_width, args.frame_height)
        img_b = load_image(find_labeled(labeled_dir, cam_b), args.frame_width, args.frame_height)
        # Click on undistorted previews for easier geometry, but store RAW coords via inverse is hard.
        # So click on RAW images (same as labeled files) — matches OpenCV pipeline inputs.
        result = click_pair(img_a, img_b, cam_a, cam_b, existing=existing.get(pair_name))
        if not result[cam_a]:
            print(f"Skipped {pair_name}")
            if pair_name in existing:
                keypoints["pairs"][pair_name] = existing[pair_name]
            continue
        keypoints["pairs"][pair_name] = result
        print(f"Saved {len(result[cam_a])} correspondences for {pair_name}")

    save_json(Path(args.keypoints), keypoints)
    print(f"\nKeypoints written: {args.keypoints}")


def prepare_pair_undistorted(keypoints_pair, cam_a, cam_b, args):
    pts_a_raw = keypoints_pair[cam_a]
    pts_b_raw = keypoints_pair[cam_b]
    und_a, scale_a, warn_a = raw_to_undistorted(
        pts_a_raw, cam_a, args.frame_width, args.frame_height, args.focal_scale, args.size_scale
    )
    und_b, scale_b, warn_b = raw_to_undistorted(
        pts_b_raw, cam_b, args.frame_width, args.frame_height, args.focal_scale, args.size_scale
    )
    return und_a, und_b, {"a": warn_a, "b": warn_b, "scale_a": scale_a, "scale_b": scale_b}


def evaluate(Hs, affines, pair_undist):
    total = 0.0
    count = 0
    details = {}
    for pair_name, cam_a, cam_b, und_a, und_b in pair_undist:
        bev_a = apply_affine_bev(project_to_bev(und_a, Hs[cam_a]), affines[cam_a])
        bev_b = apply_affine_bev(project_to_bev(und_b, Hs[cam_b]), affines[cam_b])
        dist = float(np.linalg.norm(bev_a - bev_b, axis=1).sum())
        n = len(bev_a)
        total += dist
        count += n
        details[pair_name] = pair_mean_distance(bev_a, bev_b)
    mde = total / max(count, 1)
    return mde, details


def run_optimize(args):
    """Refine H with constrained BEV similarity (scale≈1, small rot/trans).

    Unconstrained 6-DoF affine can drive det→0 and collapse side/rear views while
    still lowering click MDE — that is incorrect. We optimize SE2-like params only.
    """
    from scipy.optimize import minimize

    keypoints = load_json(Path(args.keypoints))
    Hs = load_homographies(Path(args.extrinsic_dir))
    pair_undist = []
    warnings = {}
    for pair_name, cam_a, cam_b in PAIR_SPECS:
        if pair_name not in keypoints.get("pairs", {}):
            print(f"WARNING: missing pair {pair_name}")
            continue
        pair = keypoints["pairs"][pair_name]
        if len(pair.get(cam_a, [])) < 4 or len(pair.get(cam_a, [])) != len(pair.get(cam_b, [])):
            print(f"WARNING: pair {pair_name} needs >=4 equal correspondences")
            continue
        und_a, und_b, meta = prepare_pair_undistorted(pair, cam_a, cam_b, args)
        pair_undist.append((pair_name, cam_a, cam_b, und_a, und_b))
        warnings[pair_name] = meta
        print(f"{pair_name}: {len(und_a)} points")

    if len(pair_undist) < 2:
        raise RuntimeError("Need at least 2 valid pairs to optimize. Run --click first.")

    # Front fixed (gauge). Free cams: [scale, theta, tx, ty] each.
    free_cams = ["left", "right", "rear"]
    # Bounds keep the stitch geometry intact (no collapse / huge warp).
    scale_lo, scale_hi = 0.85, 1.15
    theta_max = np.deg2rad(12.0)
    trans_max = 60.0  # px on 1000x1000 BEV
    reg_w = 0.05  # soft pull toward identity

    x0 = np.array([1.0, 0.0, 0.0, 0.0] * len(free_cams), dtype=np.float64)
    bounds = []
    for _ in free_cams:
        bounds.extend(
            [
                (scale_lo, scale_hi),
                (-theta_max, theta_max),
                (-trans_max, trans_max),
                (-trans_max, trans_max),
            ]
        )

    def unpack_se2(x):
        affines = {cam: np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64) for cam in CAMERA_ORDER}
        se2 = {cam: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) for cam in CAMERA_ORDER}
        for i, cam in enumerate(free_cams):
            p = x[i * 4 : (i + 1) * 4]
            se2[cam] = p
            affines[cam] = np.array(se2_to_affine_list(p), dtype=np.float64)
        return affines, se2

    def objective(x):
        affines, se2 = unpack_se2(x)
        mde, _ = evaluate(Hs, affines, pair_undist)
        # Penalize large deviation from identity so bad clicks don't yank cameras away.
        pen = 0.0
        for cam in free_cams:
            scale, theta, tx, ty = se2[cam]
            pen += (scale - 1.0) ** 2 + (theta / theta_max) ** 2 + (tx / trans_max) ** 2 + (ty / trans_max) ** 2
        return mde + reg_w * pen

    aff0, _ = unpack_se2(x0)
    mde0, det0 = evaluate(Hs, aff0, pair_undist)
    print(f"Initial MDE: {mde0:.3f} px  details={ {k: round(v, 2) for k, v in det0.items()} }")
    print(
        f"Constrained SE2: scale∈[{scale_lo},{scale_hi}], "
        f"|θ|≤{np.rad2deg(theta_max):.0f}°, |t|≤{trans_max:.0f}px"
    )

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-9, "disp": False},
    )
    aff_opt, se2_opt = unpack_se2(result.x)
    mde1, det1 = evaluate(Hs, aff_opt, pair_undist)
    print(f"Optimized MDE: {mde1:.3f} px  details={ {k: round(v, 2) for k, v in det1.items()} }")
    for cam in free_cams:
        scale, theta, tx, ty = se2_opt[cam]
        print(
            f"  {cam}: scale={scale:.4f} θ={np.rad2deg(theta):+.2f}° "
            f"tx={tx:+.1f} ty={ty:+.1f}"
        )

    # Safety: if somehow scale collapsed, refuse to write bad H.
    for cam in free_cams:
        scale = float(se2_opt[cam][0])
        if not (scale_lo - 1e-6 <= scale <= scale_hi + 1e-6):
            raise RuntimeError(f"Rejecting unsafe scale for {cam}: {scale}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    refined = {}
    for camera in CAMERA_ORDER:
        H_new = affine_matrix(aff_opt[camera]) @ Hs[camera]
        surround = SURROUND_NAMES[camera]
        out_h = output_dir / f"camera_{surround}_H.npy"
        np.save(out_h, H_new)
        refined[camera] = {
            "surround_name": surround,
            "H_file": str(out_h),
            "affine_bev": aff_opt[camera].tolist(),
            "se2_bev": se2_opt[camera].tolist(),
        }
        print(f"saved {out_h.name}")

    summary = {
        "method": "click_calib_smartcar_se2_bev",
        "source_extrinsic_dir": str(args.extrinsic_dir),
        "keypoints": str(args.keypoints),
        "constraints": {
            "scale": [scale_lo, scale_hi],
            "theta_deg_max": float(np.rad2deg(theta_max)),
            "trans_px_max": trans_max,
            "regularizer": reg_w,
        },
        "initial_mde_px": mde0,
        "optimized_mde_px": mde1,
        "pair_mde_before": det0,
        "pair_mde_after": det1,
        "warnings": warnings,
        "cameras": refined,
    }
    summary_path = output_dir / "click_calib_summary.json"
    save_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    return summary


def run_demo_upstream():
    """Run official Click-Calib optimize on bundled WoodScape sample keypoints."""
    upstream = SERVER / "click_calib_upstream" / "source"
    sys.path.insert(0, str(upstream))
    print("Running upstream optimize.py on WoodScape sample (may take ~10-30s)...")
    import runpy

    runpy.run_path(str(upstream / "optimize.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description="Click-Calib-style refinement for Smart Car BEV H.")
    parser.add_argument("--click", action="store_true", help="Interactively click overlap correspondences.")
    parser.add_argument("--optimize", action="store_true", help="Optimize BEV H from saved keypoints.")
    parser.add_argument(
        "--demo-upstream",
        action="store_true",
        help="Run official Click-Calib optimize on included WoodScape sample data.",
    )
    parser.add_argument("--labeled-dir", default=str(DEFAULT_LABELED))
    parser.add_argument("--extrinsic-dir", default=str(DEFAULT_EXTRINSIC))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--keypoints", default=str(DEFAULT_KEYPOINTS))
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    args = parser.parse_args()

    if not any((args.click, args.optimize, args.demo_upstream)):
        parser.print_help()
        print(
            "\nSuggested:\n"
            "  python3 click_calib_smartcar.py --click\n"
            "  python3 click_calib_smartcar.py --optimize\n"
            "  python3 bev_deploy.py --extrinsic-dir calibration_patterns/bev_extrinsic_click\n"
            "  python3 bev_stitch.py --auto-car-size\n"
        )
        return

    if args.demo_upstream:
        run_demo_upstream()
    if args.click:
        run_click(args)
    if args.optimize:
        run_optimize(args)


if __name__ == "__main__":
    main()
