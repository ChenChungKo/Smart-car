#!/usr/bin/env python3
"""Bird's-eye view by intersecting fisheye rays with the ground plane.

Front and rear CSI stay in the fisheye model: each ground pixel is projected
with cv2.fisheye.projectPoints. Left and right keep the existing homography.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
CALIB = SERVER / "calibration_patterns"
LABELED = SERVER / "camera_labeled_live_now"
LAYOUT = CALIB / "metric_layout_aug3.json"
EXTRINSIC = CALIB / "bev_extrinsic_metric_auto"
OUT = SERVER / "bev_output" / "ground_ipm"

BEV_W = BEV_H = 1000


CALIB_W, CALIB_H = 640, 480


def load_kd(name: str):
    k = np.load(CALIB / "captures" / name / "camera_0_K.npy").astype(np.float64)
    d = np.load(CALIB / "captures" / name / "camera_0_D.npy").astype(np.float64).reshape(4, 1)
    return k, d


def scale_kd(k, image):
    h, w = image.shape[:2]
    if w == CALIB_W and h == CALIB_H:
        return k.copy()
    scale = np.array(
        [[w / CALIB_W, 0.0, 0.0], [0.0, h / CALIB_H, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return scale @ k


def layout_scale(layout):
    map_w = float(layout["mat_cm"][0]) * 10.0
    map_h = float(layout["mat_cm"][1]) * 10.0
    px = min(BEV_W / map_h, BEV_H / map_w)
    origin_u = 0.5 * (BEV_W - map_h * px)
    origin_v = 0.5 * (BEV_H - map_w * px)
    return map_w, map_h, px, origin_u, origin_v


def bev_world_mm(px, origin_u, origin_v):
    vs, us = np.mgrid[0:BEV_H, 0:BEV_W]
    long_x = (vs - origin_v) / px
    short_y = (us - origin_u) / px
    return long_x.astype(np.float64), short_y.astype(np.float64)


def camera_pose(kind: str, car: dict, height_mm: float, ahead_mm: float, pitch_deg: float):
    """OpenCV camera at the bumper, pitched down, looking away from the car.

    World: +long X toward the rear, +short Y toward car-right, +Z up.
    Nose is the smaller-X edge.
    """
    pitch = np.deg2rad(pitch_deg)
    cy = 0.5 * (car["front_left"][1] + car["front_right"][1]) * 10.0
    if kind == "front":
        cx = car["front_left"][0] * 10.0 - ahead_mm
        forward = np.array([-np.cos(pitch), 0.0, -np.sin(pitch)])
        right = np.array([0.0, 1.0, 0.0])
    elif kind == "rear":
        cx = car["rear_left"][0] * 10.0 + ahead_mm
        forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
        right = np.array([0.0, -1.0, 0.0])
    else:
        raise ValueError(kind)
    down = np.cross(forward, right)
    rotation_c2w = np.column_stack([right, down, forward])
    center = np.array([cx, cy, height_mm], dtype=np.float64)
    return center, rotation_c2w


def ground_remap(
    kind, car, k, d, long_x, short_y, height_mm, ahead_mm, pitch_deg, near_mm, far_mm, img_w, img_h
):
    center, rotation_c2w = camera_pose(kind, car, height_mm, ahead_mm, pitch_deg)
    rotation_w2c = rotation_c2w.T
    rvec, _ = cv2.Rodrigues(rotation_w2c)
    tvec = (-rotation_w2c @ center).reshape(3, 1)

    world = np.stack([long_x.ravel(), short_y.ravel(), np.zeros(long_x.size)], axis=1)
    cam = (rotation_w2c @ (world - center).T).T
    ahead = cam[:, 2] > 30.0
    dist = np.hypot(world[:, 0] - center[0], world[:, 1] - center[1])
    on_ground = ahead & (dist >= near_mm) & (dist <= far_mm)

    map_x = np.full(long_x.size, -1.0, dtype=np.float32)
    map_y = np.full(long_x.size, -1.0, dtype=np.float32)
    if np.any(on_ground):
        obj = world[on_ground].reshape(-1, 1, 3)
        pixels, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)
        pixels = pixels.reshape(-1, 2)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < img_w)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < img_h)
            & np.isfinite(pixels).all(axis=1)
        )
        idx = np.flatnonzero(on_ground)
        map_x[idx[inside]] = pixels[inside, 0]
        map_y[idx[inside]] = pixels[inside, 1]
    return map_x.reshape(long_x.shape), map_y.reshape(long_x.shape), center


def warp_ground(image, map_x, map_y, interpolation=cv2.INTER_CUBIC):
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )


def snapped_world_mm(homography, origin_u, origin_v, px):
    """Final BEV pixel -> ground mm, undoing the board-snap homography."""
    inv = np.linalg.inv(homography).astype(np.float64)
    us, vs = np.meshgrid(
        np.arange(BEV_W, dtype=np.float32),
        np.arange(BEV_H, dtype=np.float32),
    )
    pts = np.stack([us, vs], axis=-1).reshape(-1, 1, 2)
    ipm = cv2.perspectiveTransform(pts, inv).reshape(BEV_H, BEV_W, 2)
    long_x = (ipm[:, :, 1] - origin_v) / px
    short_y = (ipm[:, :, 0] - origin_u) / px
    return long_x.astype(np.float64), short_y.astype(np.float64)


def warp_homography(name: str, image: np.ndarray):
    from bev_extrinsic import build_undistort_maps, load_kd

    surround = {"left": "left", "right": "right"}[name]
    k, d = load_kd(name)
    map1, map2 = build_undistort_maps(k, d, "fisheye", 640, 480, 1.0, 2.0)
    und = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    homography = np.load(EXTRINSIC / f"camera_{surround}_H.npy")
    return cv2.warpPerspective(und, homography, (BEV_W, BEV_H))


def board_contour(bev, y0, y1):
    roi = bev[y0:y1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    red = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) | cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    cand = (green == 0) & (gray > 25) & (red == 0)
    mask = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8000:
        return None
    contour = contour + np.array([[[0, y0]]])
    return contour.reshape(-1, 2).astype(np.float32)


def order_quad(points, kind: str):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if kind == "front":
        top = pts[np.argsort(pts[:, 1])[:2]]
        bot = pts[np.argsort(pts[:, 1])[-2:]]
        tl, tr = top[np.argsort(top[:, 0])]
        bl, br = bot[np.argsort(bot[:, 0])]
        return np.array([tl, tr, br, bl], dtype=np.float32)
    bottom = pts[np.argsort(pts[:, 1])[-2:]]
    top = pts[np.argsort(pts[:, 1])[:2]]
    far_l, far_r = bottom[np.argsort(bottom[:, 0])]
    near_l, near_r = top[np.argsort(top[:, 0])]
    return np.array([far_l, far_r, near_r, near_l], dtype=np.float32)


def contour_quad(contour, kind: str):
    peri = cv2.arcLength(contour.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(contour.reshape(-1, 1, 2), 0.04 * peri, True)
    if len(approx) == 4:
        return order_quad(approx, kind)
    hull = cv2.convexHull(contour.reshape(-1, 1, 2)).reshape(-1, 2)
    return order_quad(hull, kind)


def mm_uv(long_mm, short_mm, px, origin_u, origin_v):
    return np.array(
        [origin_u + short_mm * px, origin_v + long_mm * px],
        dtype=np.float32,
    )


def layout_board_quad(kind: str, gap_mm: float, car: dict, square_mm: float, px, origin_u, origin_v):
    # Outer board 8 squares along short, 7 along long. Right end at short=38 cm.
    right = 380.0
    left = right - 8 * square_mm
    if kind == "front":
        near = car["front_left"][0] * 10.0 - gap_mm
        far = near - 7 * square_mm
        return np.array(
            [
                mm_uv(far, left, px, origin_u, origin_v),
                mm_uv(far, right, px, origin_u, origin_v),
                mm_uv(near, right, px, origin_u, origin_v),
                mm_uv(near, left, px, origin_u, origin_v),
            ],
            dtype=np.float32,
        )
    near = car["rear_left"][0] * 10.0 + gap_mm
    far = near + 7 * square_mm
    return np.array(
        [
            mm_uv(far, left, px, origin_u, origin_v),
            mm_uv(far, right, px, origin_u, origin_v),
            mm_uv(near, right, px, origin_u, origin_v),
            mm_uv(near, left, px, origin_u, origin_v),
        ],
        dtype=np.float32,
    )


def inner_dst_grid(kind: str, gap_mm: float, car: dict, square_mm: float, px, origin_u, origin_v):
    right = 380.0
    left = right - 8 * square_mm
    if kind == "front":
        near = car["front_left"][0] * 10.0 - gap_mm
        far = near - 7 * square_mm
        longs = far + square_mm * np.arange(1, 7)
    else:
        near = car["rear_left"][0] * 10.0 + gap_mm
        far = near + 7 * square_mm
        longs = far - square_mm * np.arange(1, 7)
    shorts = left + square_mm * np.arange(1, 8)
    pts = []
    for long_mm in longs:
        for short_mm in shorts:
            pts.append(mm_uv(long_mm, short_mm, px, origin_u, origin_v))
    return np.array(pts, dtype=np.float32)


def corner_variants(src, cols=7, rows=6):
    pts = src.reshape(rows, cols, 2)
    grids = [
        pts,
        pts[:, ::-1],
        pts[::-1],
        pts[::-1, ::-1],
        np.transpose(pts, (1, 0, 2)),
        np.transpose(pts, (1, 0, 2))[:, ::-1],
        np.transpose(pts, (1, 0, 2))[::-1],
        np.transpose(pts, (1, 0, 2))[::-1, ::-1],
    ]
    return [g.reshape(-1, 2).astype(np.float32) for g in grids]


def detect_inner_corners(bev):
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, (7, 6))
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(
            gray, (7, 6), flags=cv2.CALIB_CB_EXHAUSTIVE
        )
    if not ok:
        return None
    return corners.reshape(-1, 2).astype(np.float32)


def car_uv(car, px, origin_u, origin_v):
    return np.array(
        [
            origin_u + 0.5 * (car["front_left"][1] + car["front_right"][1]) * 10.0 * px,
            origin_v + 0.5 * (car["front_left"][0] + car["rear_left"][0]) * 10.0 * px,
        ],
        dtype=np.float32,
    )


def snap_orientation_ok(kind, homography, ipm_corners, map_x, map_y, car_center):
    """Fisheye bottom is the car. Rear image-left is car-right; front image-left is car-left."""
    pts = np.asarray(ipm_corners, dtype=np.float32).reshape(-1, 2)
    height, width = map_x.shape[:2]
    iu = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, width - 1)
    iv = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, height - 1)
    src_x = map_x[iv, iu]
    src_y = map_y[iv, iu]
    valid = (src_x >= 0) & (src_y >= 0)
    if int(valid.sum()) < 8:
        return True
    pts, src_x, src_y = pts[valid], src_x[valid], src_y[valid]

    def mapped_mean(subset):
        return cv2.perspectiveTransform(
            np.mean(subset, axis=0).reshape(1, 1, 2).astype(np.float32), homography
        )[0, 0]

    car_pt = mapped_mean(pts[src_y >= np.median(src_y)])
    far_pt = mapped_mean(pts[src_y < np.median(src_y)])
    if np.linalg.norm(car_pt - car_center) >= np.linalg.norm(far_pt - car_center):
        return False
    left_pt = mapped_mean(pts[src_x < np.median(src_x)])
    right_pt = mapped_mean(pts[src_x >= np.median(src_x)])
    if kind == "rear":
        return left_pt[0] > right_pt[0]
    return left_pt[0] < right_pt[0]


def find_snap_h(bev, kind, car, square_mm, gap_mm, px, origin_u, origin_v, map_x=None, map_y=None, img_w=None, img_h=None):
    src = detect_inner_corners(bev)
    center = car_uv(car, px, origin_u, origin_v)
    if src is not None:
        dst = inner_dst_grid(kind, gap_mm, car, square_mm, px, origin_u, origin_v)
        best_h = None
        best_err = 1e9
        for variant in corner_variants(src):
            if variant.shape != dst.shape:
                continue
            homography, _ = cv2.findHomography(variant, dst, 0)
            if homography is None:
                continue
            if map_x is not None and not snap_orientation_ok(
                kind, homography, src, map_x, map_y, center
            ):
                continue
            pred = cv2.perspectiveTransform(variant.reshape(-1, 1, 2), homography).reshape(-1, 2)
            err = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
            if err < best_err:
                best_err = err
                best_h = homography
        if best_h is not None and best_err <= 25.0:
            print(f"{kind}: inner-corner snap rms={best_err:.2f}px")
            return best_h
        print(f"{kind}: inner snap rejected err={best_err:.2f}")
    y0, y1 = (0, 520) if kind == "front" else (480, 1000)
    contour = board_contour(bev, y0, y1)
    if contour is None:
        print(f"{kind}: no board contour, using raw IPM")
        return None
    src_q = contour_quad(contour, kind)
    dst_q = layout_board_quad(kind, gap_mm, car, square_mm, px, origin_u, origin_v)
    homography, _ = cv2.findHomography(src_q, dst_q, 0)
    if homography is None:
        print(f"{kind}: homography failed")
        return None
    if map_x is not None and src is not None and not snap_orientation_ok(
        kind, homography, src, map_x, map_y, center
    ):
        pivot = dst_q.mean(axis=0)
        flip180 = np.array(
            [[-1.0, 0.0, 2.0 * pivot[0]], [0.0, -1.0, 2.0 * pivot[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        homography = flip180 @ homography
        print(f"{kind}: flipped outer-board snap 180 for car-at-bottom")
    print(f"{kind}: snapped outer board {src_q.round(0).tolist()} -> {dst_q.round(0).tolist()}")
    return homography


def unsharp(image, amount=0.55, sigma=0.9):
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


def content_mask(bev, drop_red: bool = False):
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    mask = (gray > 16).astype(np.uint8)
    if drop_red:
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 70, 50), (14, 255, 255)) | cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
        mask[red > 0] = 0
    return mask


def feather_mask(mask, radius=28):
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return np.zeros(mask.shape, np.float32)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return np.clip(dist / float(radius), 0.0, 1.0).astype(np.float32)


def angle_weight(ang_deg, center_deg, half_deg, fade_deg):
    delta = np.abs((ang_deg - center_deg + 180.0) % 360.0 - 180.0)
    linear = np.clip((half_deg + fade_deg - delta) / fade_deg, 0.0, 1.0)
    return (linear * linear * (3.0 - 2.0 * linear)).astype(np.float32)


def gaussian_pyramid(image, levels):
    pyr = [image]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(image, levels):
    gauss = gaussian_pyramid(image.astype(np.float32), levels)
    laps = []
    for i in range(levels - 1):
        height, width = gauss[i].shape[:2]
        up = cv2.pyrUp(gauss[i + 1], dstsize=(width, height))
        laps.append(gauss[i] - up)
    laps.append(gauss[-1])
    return laps


def collapse_laplacian(laps):
    image = laps[-1]
    for i in range(len(laps) - 2, -1, -1):
        height, width = laps[i].shape[:2]
        image = cv2.pyrUp(image, dstsize=(width, height)) + laps[i]
    return image


def multiband_blend(images, weights, levels=5):
    acc = None
    for image, weight in zip(images, weights):
        weight_3 = np.repeat(weight[:, :, None], 3, axis=2)
        laps = laplacian_pyramid(image, levels)
        gains = gaussian_pyramid(weight_3, levels)
        mixed = [lap * gain for lap, gain in zip(laps, gains)]
        if acc is None:
            acc = mixed
        else:
            for i, band in enumerate(mixed):
                acc[i] += band
    return np.clip(collapse_laplacian(acc), 0, 255).astype(np.uint8)


def camera_uv(name, car, origin_u, origin_v, px):
    cu = origin_u + 0.5 * (car["front_left"][1] + car["front_right"][1]) * 10.0 * px
    cv = origin_v + 0.5 * (car["front_left"][0] + car["rear_left"][0]) * 10.0 * px
    if name == "front":
        return cu, origin_v + car["front_left"][0] * 10.0 * px
    if name == "rear":
        return cu, origin_v + car["rear_left"][0] * 10.0 * px
    if name == "left":
        return origin_u + car["front_left"][1] * 10.0 * px, cv
    return origin_u + car["front_right"][1] * 10.0 * px, cv


def blend_surround(tiles, origin_u, origin_v, px, car, drop_red_cameras=()):
    cu = origin_u + 0.5 * (car["front_left"][1] + car["front_right"][1]) * 10.0 * px
    cv = origin_v + 0.5 * (car["front_left"][0] + car["rear_left"][0]) * 10.0 * px
    ys, xs = np.mgrid[0:BEV_H, 0:BEV_W].astype(np.float32)
    ang = np.degrees(np.arctan2(xs - cu, cv - ys))
    centers = {"front": 0.0, "right": 90.0, "rear": 180.0, "left": -90.0}
    half = {"front": 40.0, "rear": 40.0, "left": 52.0, "right": 52.0}
    fade = {"front": 16.0, "rear": 16.0, "left": 16.0, "right": 16.0}
    images = []
    weights = []
    for name, bev in tiles:
        cam_u, cam_v = camera_uv(name, car, origin_u, origin_v, px)
        dist = np.hypot(xs - cam_u, ys - cam_v)
        near = 1.0 / np.power(dist + 70.0, 1.15)
        content = feather_mask(content_mask(bev, drop_red=name in drop_red_cameras), 22)
        w = angle_weight(ang, centers[name], half[name], fade[name]) * content * near
        w = cv2.GaussianBlur(w, (0, 0), 2.5)
        images.append(bev.astype(np.float32))
        weights.append(w)
    stack = np.stack(weights, axis=0)
    total = stack.sum(axis=0)
    has = total > 1e-5
    stack[:, has] /= total[has]
    stack[:, ~has] = 0.0
    surround = multiband_blend(images, [stack[i] for i in range(len(tiles))], levels=5)
    surround[~has] = 0
    return surround, (cu, cv)


def draw_car(surround, car, origin_u, origin_v, px):
    pad_long_mm = 55.0
    pad_short_mm = 42.0
    short0 = min(car["front_left"][1], car["rear_left"][1]) * 10.0 - pad_short_mm
    short1 = max(car["front_right"][1], car["rear_right"][1]) * 10.0 + pad_short_mm
    long0 = min(car["front_left"][0], car["front_right"][0]) * 10.0 - pad_long_mm
    long1 = max(car["rear_left"][0], car["rear_right"][0]) * 10.0 + pad_long_mm
    x0 = int(round(origin_u + short0 * px))
    x1 = int(round(origin_u + short1 * px))
    y0 = int(round(origin_v + long0 * px))
    y1 = int(round(origin_v + long1 * px))
    cv2.rectangle(surround, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return surround


def main():
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    car = layout["car"]
    square_mm = float(layout["square_cm"]) * 10.0
    _map_w, _map_h, px, origin_u, origin_v = layout_scale(layout)
    long_x, short_y = bev_world_mm(px, origin_u, origin_v)

    front = cv2.imread(str(LABELED / "front.jpg"))
    rear = cv2.imread(str(LABELED / "rear.jpg"))
    left = cv2.imread(str(LABELED / "left.jpg"))
    right = cv2.imread(str(LABELED / "right.jpg"))
    if any(img is None for img in (front, rear, left, right)):
        raise SystemExit(f"Missing labeled images in {LABELED}")

    height_mm = 120.0
    ahead_mm = 35.0
    gap_mm = 100.0
    pitches = {"front": 28.0, "rear": 28.0}

    def csi_tile(kind, image, k0, d, pitch):
        k = scale_kd(k0, image)
        h, w = image.shape[:2]
        preview = ground_remap(
            kind, car, k, d, long_x, short_y, height_mm, ahead_mm, pitch, 40.0, 550.0, w, h
        )
        ipm = warp_ground(image, preview[0], preview[1], interpolation=cv2.INTER_LINEAR)
        snap_h = find_snap_h(
            ipm,
            kind,
            car,
            square_mm,
            gap_mm,
            px,
            origin_u,
            origin_v,
            map_x=preview[0],
            map_y=preview[1],
            img_w=w,
            img_h=h,
        )
        if snap_h is not None:
            world_x, world_y = snapped_world_mm(snap_h, origin_u, origin_v, px)
            maps = ground_remap(
                kind, car, k, d, world_x, world_y, height_mm, ahead_mm, pitch, 40.0, 550.0, w, h
            )
        else:
            maps = preview
        sharp = warp_ground(image, maps[0], maps[1], interpolation=cv2.INTER_LANCZOS4)
        return unsharp(sharp), maps[2]

    fk, fd = load_kd("front")
    rk, rd = load_kd("rear")
    front_bev, front_center = csi_tile("front", front, fk, fd, pitches["front"])
    rear_bev, rear_center = csi_tile("rear", rear, rk, rd, pitches["rear"])
    left_bev = warp_homography("left", left)
    right_bev = warp_homography("right", right)

    tiles = [
        ("front", front_bev),
        ("rear", rear_bev),
        ("left", left_bev),
        ("right", right_bev),
    ]
    surround, _center = blend_surround(tiles, origin_u, origin_v, px, car)
    surround = draw_car(surround, car, origin_u, origin_v, px)
    filtered, _ = blend_surround(
        tiles, origin_u, origin_v, px, car, drop_red_cameras=("front", "rear")
    )
    filtered = draw_car(filtered, car, origin_u, origin_v, px)

    OUT.mkdir(parents=True, exist_ok=True)
    live = SERVER / "bev_output" / "live_now_stitch"
    live.mkdir(parents=True, exist_ok=True)
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), 97]
    cv2.imwrite(str(OUT / "front_ground.jpg"), front_bev, jpeg)
    cv2.imwrite(str(OUT / "rear_ground.jpg"), rear_bev, jpeg)
    cv2.imwrite(str(OUT / "left_ground.jpg"), left_bev, jpeg)
    cv2.imwrite(str(OUT / "right_ground.jpg"), right_bev, jpeg)
    cv2.imwrite(str(OUT / "surround_ground.jpg"), surround, jpeg)
    cv2.imwrite(str(OUT / "surround_ground_filtered.jpg"), filtered, jpeg)
    cv2.imwrite(str(live / "surround_square.jpg"), surround, jpeg)
    print(f"front camera mm {front_center.tolist()}")
    print(f"rear camera mm {rear_center.tolist()}")
    print(f"front {front.shape[1]}x{front.shape[0]} rear {rear.shape[1]}x{rear.shape[0]}")
    print(f"saved {OUT / 'surround_ground.jpg'}")
    print(f"saved {OUT / 'surround_ground_filtered.jpg'}")
    print(f"saved {live / 'surround_square.jpg'}")


if __name__ == "__main__":
    main()
