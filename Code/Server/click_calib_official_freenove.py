#!/usr/bin/env python3
"""Official Click-Calib workflow adapted for Freenove Smart Car (OpenCV K/D).

Follows https://github.com/LihaoWang1991/click_calib steps:
  1) Initialize 6DoF extrinsics (+ OpenCV intrinsics)
  2) Click keypoints on adjacent camera pairs (>=10 recommended)
  3) Optimize camera poses by minimizing ground-plane MDE
  4) Generate BEV overlay for qualitative check

This does NOT tweak Homography H (that was the wrong adapter). It optimizes
the same quantity as upstream optimize.py: 3D ground distance of correspondences.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SciRot

SERVER = Path(__file__).resolve().parent
UPSTREAM = SERVER / "click_calib_upstream"
sys.path.insert(0, str(UPSTREAM / "source"))

from generate_bev_img import generate_bev_all_cams  # noqa: E402

CALIB = SERVER / "calibration_patterns"
ROOT = UPSTREAM / "freenove"
IMG_DIR = ROOT / "images"
CALIB_INIT = ROOT / "calibrations" / "initial"
CALIB_OPT = ROOT / "calibrations" / "optimized"
KEYPOINTS_PATH = ROOT / "keypoints.json"
BEV_OUT = SERVER / "bev_output" / "click_calib_official"

# Freenove body ~29x18 cm; cameras near body edges, ~12 cm above ground.
CAR_LEN = 0.29
CAR_WID = 0.18
CAM_Z = 0.12

PAIR_ORDER = (
    ("front_left", "front", "left"),
    ("front_right", "front", "right"),
    ("rear_left", "rear", "left"),
    ("rear_right", "rear", "right"),
)


class OpenCVCamera:
    """WoodScape-Camera-compatible wrapper using OpenCV pinhole/fisheye."""

    def __init__(self, K, D, width, height, translation, rotation, model="fisheye"):
        self.K = np.asarray(K, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        self.model = model
        self._size = np.array([int(width), int(height)], dtype=int)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
        pose[:3, 3] = np.asarray(translation, dtype=np.float64)
        self._pose = pose
        self._inv_pose = np.linalg.inv(pose)

    @property
    def width(self):
        return int(self._size[0])

    @property
    def height(self):
        return int(self._size[1])

    @property
    def translation(self):
        return self._pose[:3, 3]

    def get_translation(self):
        return self.translation.copy()

    def get_rotation(self):
        return self._pose[:3, :3].copy()

    def update_extr(self, translation, rotation):
        self._pose[:3, 3] = np.asarray(translation, dtype=np.float64)
        self._pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
        self._inv_pose = np.linalg.inv(self._pose)

    def project_3d_to_2d(self, world_points, do_clip=False, invalid_value=np.nan):
        pts = np.asarray(world_points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("world_points must be Nx3 or Nx4")
        if pts.shape[1] == 3:
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        else:
            pts_h = pts
        cam = (pts_h @ self._inv_pose.T)[:, :3]
        obj = cam.reshape(-1, 1, 3)
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        if self.model == "fisheye":
            img, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, self.K, self.D)
        else:
            img, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.D)
        screen = img.reshape(-1, 2)
        behind = cam[:, 2] <= 0
        screen[behind] = np.nan
        # Remap casts to int16; keep invalids as -1 to avoid overflow streaks.
        bad = ~np.isfinite(screen).all(axis=1)
        screen[bad] = -1.0
        return screen

    def project_2d_to_3d_ground(self, screen_points, do_clip=False):
        pts = np.asarray(screen_points, dtype=np.float64).reshape(-1, 1, 2)
        if self.model == "fisheye":
            und = cv2.fisheye.undistortPoints(pts, self.K, self.D)
        else:
            und = cv2.undistortPoints(pts, self.K, self.D)
        xy = und.reshape(-1, 2)
        dirs_cam = np.hstack([xy, np.ones((xy.shape[0], 1))])
        R = self._pose[:3, :3]
        t = self._pose[:3, 3]
        dirs_w = (R @ dirs_cam.T).T
        denom = dirs_w[:, 2]
        # Avoid divide-by-zero for rays parallel to ground
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        scale = (-t[2]) / denom
        return t + scale[:, None] * dirs_w


def load_kd(camera_name: str):
    if camera_name == "front":
        K = np.load(CALIB / "captures" / "front" / "camera_0_K.npy")
        D = np.load(CALIB / "captures" / "front" / "camera_0_D.npy")
        return K, D, "pinhole"
    K = np.load(CALIB / "shared" / "usb_fisheye_K.npy")
    D = np.load(CALIB / "shared" / "usb_fisheye_D.npy")
    return K, D, "fisheye"


def default_extrinsics():
    """Nominal Freenove poses (meters, degrees), WoodScape-like zxz convention.

    Start from the same Euler triples as click_calib initialize_extrins_calib.py,
    then keep Freenove-scale translations (toy-car body ~29x18 cm).
    """
    half_l, half_w = CAR_LEN / 2, CAR_WID / 2
    return {
        "front": {
            "t": [half_l + 0.02, 0.0, CAM_Z],
            "euler_zxz_deg": [180.0, 90.0, 90.0],
        },
        "left": {
            "t": [0.0, half_w + 0.02, CAM_Z],
            "euler_zxz_deg": [180.0, 180.0, -180.0],
        },
        "right": {
            "t": [0.0, -(half_w + 0.02), CAM_Z],
            "euler_zxz_deg": [-180.0, 180.0, 0.0],
        },
        "rear": {
            "t": [-(half_l + 0.02), 0.0, CAM_Z],
            "euler_zxz_deg": [180.0, 90.0, -90.0],
        },
    }


def write_calib_json(path: Path, camera_name: str, K, D, model, quat, t, width, height):
    intr = {
        "model": f"opencv_{model}",
        "width": float(width),
        "height": float(height),
        "K": K.astype(float).tolist(),
        "D": np.asarray(D, dtype=float).ravel().tolist(),
        # Keep WoodScape-compatible placeholders unused by OpenCVCamera
        "aspect_ratio": 1.0,
        "cx_offset": 0.0,
        "cy_offset": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "poly_order": 4,
    }
    payload = {
        "name": camera_name,
        "extrinsic": {"quaternion": [float(x) for x in quat], "translation": [float(x) for x in t]},
        "intrinsic": intr,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_calib_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    intr = data["intrinsic"]
    quat = data["extrinsic"]["quaternion"]
    t = data["extrinsic"]["translation"]
    return intr, quat, t


def camera_from_calib(path: Path) -> OpenCVCamera:
    intr, quat, t = read_calib_json(path)
    model = "pinhole" if "pinhole" in intr.get("model", "") else "fisheye"
    K = np.asarray(intr["K"], dtype=np.float64)
    D = np.asarray(intr["D"], dtype=np.float64)
    R = SciRot.from_quat(quat).as_matrix()
    return OpenCVCamera(K, D, int(intr["width"]), int(intr["height"]), t, R, model=model)


def find_labeled(labeled_dir: Path, name: str) -> Path:
    matches = sorted(labeled_dir.glob(f"{name}*.jpg")) + sorted(labeled_dir.glob(f"{name}*.png"))
    if not matches:
        raise FileNotFoundError(f"No image for {name} in {labeled_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def cmd_init(args):
    labeled = Path(args.labeled_dir)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    CALIB_INIT.mkdir(parents=True, exist_ok=True)
    extr = default_extrinsics()
    names = {
        "front": "front.png",
        "left": "left.png",
        "right": "right.png",
        "rear": "rear.png",
    }
    for cam, fname in names.items():
        src = find_labeled(labeled, cam)
        img = cv2.imread(str(src))
        if img is None:
            raise RuntimeError(f"Cannot read {src}")
        h, w = img.shape[:2]
        cv2.imwrite(str(IMG_DIR / fname), img)
        K, D, model = load_kd(cam)
        quat = SciRot.from_euler("zxz", extr[cam]["euler_zxz_deg"], degrees=True).as_quat()
        write_calib_json(CALIB_INIT / f"{cam}.json", cam, K, D, model, quat, extr[cam]["t"], w, h)
        print(f"init {cam}: {src.name} -> {fname}, t={extr[cam]['t']}")
    print(f"Initial calibrations: {CALIB_INIT}")


def load_images():
    imgs = {}
    for cam in ("front", "left", "right", "rear"):
        path = IMG_DIR / f"{cam}.png"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        imgs[cam] = img
    return imgs


def cmd_bev(args):
    calib_dir = Path(args.calib_dir)
    cams = {c: camera_from_calib(calib_dir / f"{c}.json") for c in ("front", "left", "right", "rear")}
    imgs = load_images()
    bev = generate_bev_all_cams(
        cams["front"],
        cams["left"],
        cams["right"],
        cams["rear"],
        imgs["front"],
        imgs["left"],
        imgs["right"],
        imgs["rear"],
        overlay_opt=args.overlay,
        bev_range=args.bev_range,
        bev_size=args.bev_size,
    )
    BEV_OUT.mkdir(parents=True, exist_ok=True)
    out = BEV_OUT / args.output_name
    cv2.imwrite(str(out), bev)
    print(f"Saved BEV: {out}")
    if args.show:
        plt.imshow(cv2.cvtColor(bev, cv2.COLOR_BGR2RGB))
        plt.title(out.name)
        plt.axis("off")
        plt.show()


def click_one_pair(img_a, img_b, title_a, title_b):
    """Official-style matplotlib clicker (same index = same world point)."""
    pts_a, pts_b = [], []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    ax1.set_title(title_a)
    ax2.set_title(title_b)
    ax1.axis("off")
    ax2.axis("off")
    fig.suptitle(
        "Click-Calib (official): same index = same world point.\n"
        "Prefer overlap-region markers / mat intersections. Scroll to zoom. Close window when done.",
        fontsize=10,
    )

    def zoom(event):
        ax = event.inaxes
        if ax is None:
            return
        xdata, ydata = event.xdata, event.ydata
        if xdata is None:
            return
        x, y = ax.get_xlim(), ax.get_ylim()
        if event.button == "up":
            ax.set_xlim(xdata - (xdata - x[0]) / 1.1, xdata + (x[1] - xdata) / 1.1)
            ax.set_ylim(ydata - (ydata - y[0]) / 1.1, ydata + (y[1] - ydata) / 1.1)
        elif event.button == "down":
            ax.set_xlim(xdata - (xdata - x[0]) * 1.1, xdata + (x[1] - xdata) * 1.1)
            ax.set_ylim(ydata - (ydata - y[0]) * 1.1, ydata + (y[1] - ydata) * 1.1)
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes == ax1 and event.xdata is not None:
            x, y = int(event.xdata), int(event.ydata)
            pts_a.append((x, y))
            ax1.plot(x, y, "ro", markersize=3)
            ax1.annotate(str(len(pts_a)), (x, y), color=(0.7, 1, 0.4), fontsize=8)
        elif event.inaxes == ax2 and event.xdata is not None:
            x, y = int(event.xdata), int(event.ydata)
            pts_b.append((x, y))
            ax2.plot(x, y, "ro", markersize=3)
            ax2.annotate(str(len(pts_b)), (x, y), color=(0.7, 1, 0.4), fontsize=8)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("scroll_event", zoom)
    plt.tight_layout()
    plt.show()
    return pts_a, pts_b


def cmd_click(args):
    imgs = load_images()
    data = {"pairs": {}}
    if KEYPOINTS_PATH.exists() and not args.reset:
        data = json.loads(KEYPOINTS_PATH.read_text(encoding="utf-8"))

    for pair_name, cam_a, cam_b in PAIR_ORDER:
        print(f"\n=== {pair_name}: click matching world points (aim >=10) ===")
        pts_a, pts_b = click_one_pair(imgs[cam_a], imgs[cam_b], cam_a, cam_b)
        if not pts_a and not pts_b:
            print(f"Skipped {pair_name}")
            continue
        if len(pts_a) != len(pts_b) or len(pts_a) == 0:
            raise SystemExit(
                f"{pair_name}: unequal/empty points ({len(pts_a)} vs {len(pts_b)}). Re-run --click."
            )
        if len(pts_a) < 4:
            print(f"WARNING: only {len(pts_a)} points (official recommends >=10)")
        data["pairs"][pair_name] = {cam_a: pts_a, cam_b: pts_b}
        print(f"Saved {len(pts_a)} correspondences for {pair_name}")
        KEYPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEYPOINTS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"\nKeypoints: {KEYPOINTS_PATH}")


def optimizer_official(calib, cams, z_fixed, pairs):
    """Same objective as upstream optimize.py (ground MDE)."""
    order = ["front", "left", "right", "rear"]
    for i, name in enumerate(order):
        base = i * 6
        t = [calib[base], calib[base + 1], z_fixed[name]]
        R = SciRot.from_quat(calib[base + 2 : base + 6]).as_matrix()
        cams[name].update_extr(t, R)

    total = 0.0
    count = 0
    details = {}
    for pair_name, cam_a, cam_b in PAIR_ORDER:
        if pair_name not in pairs:
            continue
        pa = np.asarray(pairs[pair_name][cam_a], dtype=np.float64)
        pb = np.asarray(pairs[pair_name][cam_b], dtype=np.float64)
        wa = cams[cam_a].project_2d_to_3d_ground(pa)
        wb = cams[cam_b].project_2d_to_3d_ground(pb)
        d = np.linalg.norm(wa - wb, axis=1)
        total += float(d.sum())
        count += len(d)
        details[pair_name] = float(d.mean())
    mde = total / max(count, 1)
    return mde, details


def cmd_optimize(args):
    if not KEYPOINTS_PATH.exists():
        raise SystemExit("Missing keypoints. Run --click first.")
    keypoints = json.loads(KEYPOINTS_PATH.read_text(encoding="utf-8"))
    pairs = keypoints.get("pairs", {})
    if len(pairs) < 2:
        raise SystemExit("Need at least 2 pairs in keypoints.json")

    cams = {}
    z_fixed = {}
    calib_ini = []
    for name in ("front", "left", "right", "rear"):
        intr, quat, t = read_calib_json(CALIB_INIT / f"{name}.json")
        cams[name] = camera_from_calib(CALIB_INIT / f"{name}.json")
        z_fixed[name] = float(t[2])
        calib_ini.extend([t[0], t[1], *quat])
    calib_ini = np.asarray(calib_ini, dtype=np.float64)

    mde0, det0 = optimizer_official(calib_ini, cams, z_fixed, pairs)
    print(f"Initial MDE: {mde0:.4f} m  details={ {k: round(v, 4) for k, v in det0.items()} }")

    def objective(x):
        mde, _ = optimizer_official(x, cams, z_fixed, pairs)
        return mde

    print("Optimizing (BFGS, official Click-Calib objective)...")
    res = minimize(objective, calib_ini, method="BFGS", options={"maxiter": 200, "disp": False})
    mde1, det1 = optimizer_official(res.x, cams, z_fixed, pairs)
    print(f"Optimized MDE: {mde1:.4f} m  details={ {k: round(v, 4) for k, v in det1.items()} }")

    CALIB_OPT.mkdir(parents=True, exist_ok=True)
    x = res.x.tolist()
    for i, name in enumerate(("front", "left", "right", "rear")):
        base = i * 6
        t = [x[base], x[base + 1], z_fixed[name]]
        quat = x[base + 2 : base + 6]
        intr, _, _ = read_calib_json(CALIB_INIT / f"{name}.json")
        model = "pinhole" if "pinhole" in intr["model"] else "fisheye"
        write_calib_json(
            CALIB_OPT / f"{name}.json",
            name,
            np.asarray(intr["K"]),
            np.asarray(intr["D"]),
            model,
            quat,
            t,
            int(intr["width"]),
            int(intr["height"]),
        )
        print(f"saved {name}.json  t={np.round(t, 4).tolist()}")

    summary = {
        "method": "click_calib_official_opencv",
        "keypoints": str(KEYPOINTS_PATH),
        "initial_mde_m": mde0,
        "optimized_mde_m": mde1,
        "pair_mde_before_m": det0,
        "pair_mde_after_m": det1,
        "success": bool(res.success),
        "message": str(res.message),
    }
    summary_path = CALIB_OPT / "optimize_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


def create_bev_maps_fast(source_cam: OpenCVCamera, bev_range: float, bev_size: int):
    """Vectorized BEV remap (official loop is too slow on Pi for interactive GUI)."""
    scale = bev_range / bev_size
    uu, vv = np.meshgrid(np.arange(bev_size), np.arange(bev_size))
    world_x = bev_range / 2.0 - vv * scale
    world_y = bev_range / 2.0 - uu * scale
    world = np.column_stack(
        [world_x.ravel(), world_y.ravel(), np.zeros(bev_size * bev_size, dtype=np.float64)]
    )
    screen = source_cam.project_3d_to_2d(world)
    u_map = screen[:, 0].reshape(bev_size, bev_size).astype(np.float32)
    v_map = screen[:, 1].reshape(bev_size, bev_size).astype(np.float32)
    # Invalid / behind-camera samples
    bad = ~np.isfinite(u_map) | ~np.isfinite(v_map) | (u_map < 0) | (v_map < 0)
    u_map[bad] = -1
    v_map[bad] = -1
    return cv2.convertMaps(u_map, v_map, dstmap1type=cv2.CV_16SC2, nninterpolation=False)


def generate_bev_fast(cams, imgs, overlay_opt="lr", bev_range=1.5, bev_size=320):
    bevs = {
        name: cv2.remap(imgs[name], *create_bev_maps_fast(cams[name], bev_range, bev_size), cv2.INTER_LINEAR)
        for name in ("front", "left", "right", "rear")
    }
    from projection import bev_points_world_to_img

    u_l, v_f = bev_points_world_to_img(bev_range, bev_size, cams["front"].get_translation()[:2])
    u_left, _ = bev_points_world_to_img(bev_range, bev_size, cams["left"].get_translation()[:2])
    u_right, _ = bev_points_world_to_img(bev_range, bev_size, cams["right"].get_translation()[:2])
    _, v_rear = bev_points_world_to_img(bev_range, bev_size, cams["rear"].get_translation()[:2])
    # Clamp indices into image
    v_f = int(np.clip(v_f, 0, bev_size - 1))
    v_rear = int(np.clip(v_rear, 0, bev_size - 1))
    u_left = int(np.clip(u_left, 0, bev_size - 1))
    u_right = int(np.clip(u_right, 0, bev_size - 1))

    out = np.zeros_like(bevs["front"])
    if overlay_opt == "lr":
        out[0:v_f, :] = bevs["front"][0:v_f, :]
        out[v_rear:bev_size, :] = bevs["rear"][v_rear:bev_size, :]
        out[:, 0:u_left] = bevs["left"][:, 0:u_left]
        out[:, u_right:bev_size] = bevs["right"][:, u_right:bev_size]
    elif overlay_opt == "fr":
        out[:, 0:u_left] = bevs["left"][:, 0:u_left]
        out[:, u_right:bev_size] = bevs["right"][:, u_right:bev_size]
        out[0:v_f, :] = bevs["front"][0:v_f, :]
        out[v_rear:bev_size, :] = bevs["rear"][v_rear:bev_size, :]
    else:
        acc = (
            bevs["front"].astype(np.float32)
            + bevs["left"].astype(np.float32)
            + bevs["right"].astype(np.float32)
            + bevs["rear"].astype(np.float32)
        ) / 4.0
        mx = acc.max()
        out = ((acc / mx) * 255).astype(np.uint8) if mx > 0 else acc.astype(np.uint8)
    return out


def cmd_initialize(args):
    """Official Step 1: manually adjust 6DoF until BEV looks reasonable, then Export."""
    from matplotlib.widgets import TextBox, RadioButtons, Button

    # Ensure images + initial JSONs exist
    if not (IMG_DIR / "front.png").exists():
        cmd_init(args)

    imgs = load_images()
    extr = default_extrinsics()

    cams = {}
    t = {}
    eulers = {}
    intrinsics = {}
    for name in ("front", "left", "right", "rear"):
        path = CALIB_INIT / f"{name}.json"
        if path.exists():
            intr, quat, tt = read_calib_json(path)
            cams[name] = camera_from_calib(path)
            t[name] = list(tt)
            # Avoid gimbal-lock warning exploding UI: fall back to defaults if needed
            try:
                eulers[name] = SciRot.from_quat(quat).as_euler("zxz", degrees=True).tolist()
            except Exception:
                eulers[name] = list(extr[name]["euler_zxz_deg"])
            intrinsics[name] = intr
        else:
            K, D, model = load_kd(name)
            h, w = imgs[name].shape[:2]
            quat = SciRot.from_euler("zxz", extr[name]["euler_zxz_deg"], degrees=True).as_quat()
            tt = list(extr[name]["t"])
            write_calib_json(path, name, K, D, model, quat, tt, w, h)
            cams[name] = camera_from_calib(path)
            t[name] = tt
            eulers[name] = list(extr[name]["euler_zxz_deg"])
            intrinsics[name] = read_calib_json(path)[0]

    overlay_opt = {"value": "lr"}
    # Interactive preview must stay small on Pi (official 640–960 freezes).
    bev_range = float(args.bev_range)
    bev_size = min(int(args.bev_size), 360)
    busy = {"flag": False}

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.52, right=0.98, top=0.92, bottom=0.08)
    ax.axis("off")
    ax.set_title(f"Step 1: Initialize extrinsic  (preview {bev_size}px, fast)")
    print("Building first BEV preview...", flush=True)
    topview = generate_bev_fast(cams, imgs, overlay_opt["value"], bev_range, bev_size)
    im = ax.imshow(cv2.cvtColor(topview, cv2.COLOR_BGR2RGB))
    status = fig.text(0.52, 0.02, "Ready — edit a value and press Enter", fontsize=9, color="green")

    def refresh(_=None):
        if busy["flag"]:
            return
        busy["flag"] = True
        status.set_text("Updating...")
        status.set_color("orange")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        try:
            overlay_opt["value"] = "lr" if menu_topview_opt.value_selected == "left-right" else "fr"
            vals = {
                "front": (
                    float(text_pos_x_0.text),
                    float(text_pos_y_0.text),
                    float(text_rot_z1_0.text),
                    float(text_rot_x_0.text),
                    float(text_rot_z2_0.text),
                ),
                "left": (
                    float(text_pos_x_1.text),
                    float(text_pos_y_1.text),
                    float(text_rot_z1_1.text),
                    float(text_rot_x_1.text),
                    float(text_rot_z2_1.text),
                ),
                "right": (
                    float(text_pos_x_2.text),
                    float(text_pos_y_2.text),
                    float(text_rot_z1_2.text),
                    float(text_rot_x_2.text),
                    float(text_rot_z2_2.text),
                ),
                "rear": (
                    float(text_pos_x_3.text),
                    float(text_pos_y_3.text),
                    float(text_rot_z1_3.text),
                    float(text_rot_x_3.text),
                    float(text_rot_z2_3.text),
                ),
            }
            for name, (px, py, z1, rx, z2) in vals.items():
                t[name][0], t[name][1] = px, py
                R = SciRot.from_euler("zxz", [z1, rx, z2], degrees=True).as_matrix()
                cams[name].update_extr(t[name], R)
                eulers[name] = [z1, rx, z2]
            view = generate_bev_fast(cams, imgs, overlay_opt["value"], bev_range, bev_size)
            im.set_data(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            status.set_text("Ready")
            status.set_color("green")
        except ValueError:
            status.set_text("Invalid number — fix and press Enter")
            status.set_color("red")
        except Exception as exc:
            status.set_text(f"Error: {exc}")
            status.set_color("red")
            print(f"refresh error: {exc}", flush=True)
        finally:
            busy["flag"] = False
            fig.canvas.draw_idle()

    def export_calib(_event):
        CALIB_INIT.mkdir(parents=True, exist_ok=True)
        for name in ("front", "left", "right", "rear"):
            intr = intrinsics[name]
            model = "pinhole" if "pinhole" in intr.get("model", "") else "fisheye"
            quat = SciRot.from_euler("zxz", eulers[name], degrees=True).as_quat()
            write_calib_json(
                CALIB_INIT / f"{name}.json",
                name,
                np.asarray(intr["K"]),
                np.asarray(intr["D"]),
                model,
                quat,
                t[name],
                int(intr["width"]),
                int(intr["height"]),
            )
            print(
                f"Exported {name}: t={np.round(t[name], 4).tolist()} euler={np.round(eulers[name], 2).tolist()}",
                flush=True,
            )
        BEV_OUT.mkdir(parents=True, exist_ok=True)
        preview = generate_bev_fast(cams, imgs, overlay_opt["value"], bev_range, max(bev_size, 640))
        out = BEV_OUT / "init_manual_bev.jpg"
        cv2.imwrite(str(out), preview)
        status.set_text(f"Exported → {CALIB_INIT.name} + {out.name}")
        status.set_color("blue")
        fig.canvas.draw_idle()
        print(f"Exported initial calib -> {CALIB_INIT}", flush=True)
        print(f"Preview BEV -> {out}", flush=True)
        print("Next: python3 click_calib_official_freenove.py --click", flush=True)

    box_topview_opt = plt.axes([0.08, 0.78, 0.18, 0.10], facecolor="linen")
    menu_topview_opt = RadioButtons(box_topview_opt, ("left-right", "front-rear"))
    box_export = plt.axes([0.08, 0.12, 0.16, 0.05], facecolor="linen")
    button_export = Button(box_export, "Export to files")

    box_pos_x_0 = plt.axes([0.10, 0.66, 0.06, 0.03], facecolor="linen")
    text_pos_x_0 = TextBox(box_pos_x_0, "Front x", initial=f"{t['front'][0]:.3f}")
    box_pos_y_0 = plt.axes([0.22, 0.66, 0.06, 0.03], facecolor="linen")
    text_pos_y_0 = TextBox(box_pos_y_0, "y", initial=f"{t['front'][1]:.3f}")
    box_rot_z1_0 = plt.axes([0.10, 0.62, 0.06, 0.03], facecolor="linen")
    text_rot_z1_0 = TextBox(box_rot_z1_0, "z1", initial=f"{eulers['front'][0]:.1f}")
    box_rot_x_0 = plt.axes([0.22, 0.62, 0.06, 0.03], facecolor="linen")
    text_rot_x_0 = TextBox(box_rot_x_0, "x", initial=f"{eulers['front'][1]:.1f}")
    box_rot_z2_0 = plt.axes([0.34, 0.62, 0.06, 0.03], facecolor="linen")
    text_rot_z2_0 = TextBox(box_rot_z2_0, "z2", initial=f"{eulers['front'][2]:.1f}")

    box_pos_x_1 = plt.axes([0.10, 0.54, 0.06, 0.03], facecolor="linen")
    text_pos_x_1 = TextBox(box_pos_x_1, "Left x", initial=f"{t['left'][0]:.3f}")
    box_pos_y_1 = plt.axes([0.22, 0.54, 0.06, 0.03], facecolor="linen")
    text_pos_y_1 = TextBox(box_pos_y_1, "y", initial=f"{t['left'][1]:.3f}")
    box_rot_z1_1 = plt.axes([0.10, 0.50, 0.06, 0.03], facecolor="linen")
    text_rot_z1_1 = TextBox(box_rot_z1_1, "z1", initial=f"{eulers['left'][0]:.1f}")
    box_rot_x_1 = plt.axes([0.22, 0.50, 0.06, 0.03], facecolor="linen")
    text_rot_x_1 = TextBox(box_rot_x_1, "x", initial=f"{eulers['left'][1]:.1f}")
    box_rot_z2_1 = plt.axes([0.34, 0.50, 0.06, 0.03], facecolor="linen")
    text_rot_z2_1 = TextBox(box_rot_z2_1, "z2", initial=f"{eulers['left'][2]:.1f}")

    box_pos_x_2 = plt.axes([0.10, 0.42, 0.06, 0.03], facecolor="linen")
    text_pos_x_2 = TextBox(box_pos_x_2, "Right x", initial=f"{t['right'][0]:.3f}")
    box_pos_y_2 = plt.axes([0.22, 0.42, 0.06, 0.03], facecolor="linen")
    text_pos_y_2 = TextBox(box_pos_y_2, "y", initial=f"{t['right'][1]:.3f}")
    box_rot_z1_2 = plt.axes([0.10, 0.38, 0.06, 0.03], facecolor="linen")
    text_rot_z1_2 = TextBox(box_rot_z1_2, "z1", initial=f"{eulers['right'][0]:.1f}")
    box_rot_x_2 = plt.axes([0.22, 0.38, 0.06, 0.03], facecolor="linen")
    text_rot_x_2 = TextBox(box_rot_x_2, "x", initial=f"{eulers['right'][1]:.1f}")
    box_rot_z2_2 = plt.axes([0.34, 0.38, 0.06, 0.03], facecolor="linen")
    text_rot_z2_2 = TextBox(box_rot_z2_2, "z2", initial=f"{eulers['right'][2]:.1f}")

    box_pos_x_3 = plt.axes([0.10, 0.30, 0.06, 0.03], facecolor="linen")
    text_pos_x_3 = TextBox(box_pos_x_3, "Rear x", initial=f"{t['rear'][0]:.3f}")
    box_pos_y_3 = plt.axes([0.22, 0.30, 0.06, 0.03], facecolor="linen")
    text_pos_y_3 = TextBox(box_pos_y_3, "y", initial=f"{t['rear'][1]:.3f}")
    box_rot_z1_3 = plt.axes([0.10, 0.26, 0.06, 0.03], facecolor="linen")
    text_rot_z1_3 = TextBox(box_rot_z1_3, "z1", initial=f"{eulers['rear'][0]:.1f}")
    box_rot_x_3 = plt.axes([0.22, 0.26, 0.06, 0.03], facecolor="linen")
    text_rot_x_3 = TextBox(box_rot_x_3, "x", initial=f"{eulers['rear'][1]:.1f}")
    box_rot_z2_3 = plt.axes([0.34, 0.26, 0.06, 0.03], facecolor="linen")
    text_rot_z2_3 = TextBox(box_rot_z2_3, "z2", initial=f"{eulers['rear'][2]:.1f}")

    fig.text(
        0.08,
        0.92,
        "Official Step 1 (fast preview): tune until mat looks upright.\n"
        "Edit number → Enter to refresh. Export when OK.",
        fontsize=9,
    )

    menu_topview_opt.on_clicked(refresh)
    button_export.on_clicked(export_calib)
    for box in (
        text_pos_x_0,
        text_pos_y_0,
        text_rot_z1_0,
        text_rot_x_0,
        text_rot_z2_0,
        text_pos_x_1,
        text_pos_y_1,
        text_rot_z1_1,
        text_rot_x_1,
        text_rot_z2_1,
        text_pos_x_2,
        text_pos_y_2,
        text_rot_z1_2,
        text_rot_x_2,
        text_rot_z2_2,
        text_pos_x_3,
        text_pos_y_3,
        text_rot_z1_3,
        text_rot_x_3,
        text_rot_z2_3,
    ):
        box.on_submit(refresh)

    print("Initialize GUI open (fast mode). Tune, then Export.", flush=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Official Click-Calib pipeline for Freenove (OpenCV).")
    parser.add_argument("--init", action="store_true", help="Copy images + write default initial calib JSONs")
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Step1 GUI: manually adjust extrinsic until BEV looks OK (official initialize_extrins_calib)",
    )
    parser.add_argument("--click", action="store_true", help="Step2: click keypoints (matplotlib GUI)")
    parser.add_argument("--optimize", action="store_true", help="Step3: optimize 6DoF extrinsics")
    parser.add_argument("--bev", action="store_true", help="Step4: generate BEV overlay")
    parser.add_argument("--labeled-dir", default=str(SERVER / "camera_labeled_click"))
    parser.add_argument("--calib-dir", default=str(CALIB_INIT))
    parser.add_argument("--overlay", default="all", choices=("all", "fr", "lr"))
    parser.add_argument("--bev-range", type=float, default=1.5, help="BEV range in meters")
    parser.add_argument("--bev-size", type=int, default=800)
    parser.add_argument("--output-name", default="surround_official.jpg")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Ignore existing keypoints when clicking")
    args = parser.parse_args()

    if not any((args.init, args.initialize, args.click, args.optimize, args.bev)):
        parser.print_help()
        print(
            "\nOfficial order:\n"
            "  1) python3 click_calib_official_freenove.py --initialize\n"
            "     (tune poses, click Export to files)\n"
            "  2) python3 click_calib_official_freenove.py --click\n"
            "  3) python3 click_calib_official_freenove.py --optimize\n"
            "  4) python3 click_calib_official_freenove.py --bev "
            f"--calib-dir {CALIB_OPT} --output-name optimized_bev.jpg\n"
        )
        return

    if args.init:
        cmd_init(args)
    if args.initialize:
        cmd_initialize(args)
    if args.click:
        cmd_click(args)
    if args.optimize:
        cmd_optimize(args)
    if args.bev:
        cmd_bev(args)


if __name__ == "__main__":
    main()
