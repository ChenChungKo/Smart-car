#!/usr/bin/env python3
"""Run Smart Car BEV stitch using deployed K/D/H (pinhole front + fisheye USB)."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
SURROUND_ROOT = Path.home() / "CameraCalibration-test"
sys.path.insert(0, str(SERVER))
from camera_devices import bev_focal_scale, bev_size_scale
sys.path.insert(0, str(SURROUND_ROOT))

_argv = sys.argv[:]
sys.argv = [_argv[0]]
from SurroundBirdEyeView.surroundBEV import (  # noqa: E402
    Mask,
    color_balance,
    luminance_balance,
)

sys.argv = _argv

DEFAULT_DATA = SURROUND_ROOT / "SurroundBirdEyeView" / "data"
DEFAULT_OUTPUT = SERVER / "bev_output"


class SmartCamera:
    """Pinhole front + fisheye USB, same API as surroundBEV.Camera."""

    def __init__(self, name, data_dir, frame_width, frame_height, bev_width, bev_height, focal_scale, size_scale, pinhole=False):
        self.name = name
        self.pinhole = pinhole
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.focal_scale = focal_scale
        self.size_scale = size_scale
        base = data_dir / name
        self.camera_mat = np.load(base / f"camera_{name}_K.npy")
        self.dist_coeff = np.load(base / f"camera_{name}_D.npy")
        self.homography = np.load(base / f"camera_{name}_H.npy")
        self.camera_mat_dst = self.get_camera_mat_dst()
        self.undistort_maps = self.get_undistort_maps()
        self.bev_maps = self.get_bev_maps()

    def get_camera_mat_dst(self):
        if self.pinhole and self.size_scale == 1.0:
            # alpha=1.0 keeps every source pixel but, for a wide-FOV lens with
            # strong barrel distortion, forces a tiny focal length so the whole
            # image fits — this squeezes/pinches the useful center into a sliver
            # (and can even fold pixels, showing up as black holes). alpha=0.0
            # crops to the valid rectilinear region instead, which for this lens
            # already covers the full frame and stays clean/undistorted.
            p, _roi = cv2.getOptimalNewCameraMatrix(
                self.camera_mat,
                self.dist_coeff,
                (self.frame_width, self.frame_height),
                0.0,
                (self.frame_width, self.frame_height),
            )
            p = p.astype(np.float64)
            p[0, 0] *= self.focal_scale
            p[1, 1] *= self.focal_scale
            return p
        dst = self.camera_mat.copy()
        dst[0][0] *= self.focal_scale
        dst[1][1] *= self.focal_scale
        dst[0][2] = self.frame_width / 2 * self.size_scale
        dst[1][2] = self.frame_height / 2 * self.size_scale
        return dst

    def get_undistort_maps(self):
        out_w = int(self.frame_width * self.size_scale)
        out_h = int(self.frame_height * self.size_scale)
        if self.pinhole:
            return cv2.initUndistortRectifyMap(
                self.camera_mat,
                self.dist_coeff,
                np.eye(3),
                self.camera_mat_dst,
                (out_w, out_h),
                cv2.CV_16SC2,
            )
        return cv2.fisheye.initUndistortRectifyMap(
            self.camera_mat,
            self.dist_coeff,
            np.eye(3),
            self.camera_mat_dst,
            (out_w, out_h),
            cv2.CV_16SC2,
        )

    def get_bev_maps(self):
        # Keep legacy combined maps for callers that expect them, but prefer
        # two-step undistort+warp in raw2bev for sharper output.
        bev_size = (self.bev_width, self.bev_height)
        map1 = cv2.warpPerspective(self.undistort_maps[0], self.homography, bev_size)
        map2 = cv2.warpPerspective(self.undistort_maps[1], self.homography, bev_size)
        return map1, map2

    def raw2bev(self, img):
        # Two-step path avoids warping integer undistort maps (which blurs
        # front/rear more because those views stretch the ground heavily).
        und = cv2.remap(
            img,
            self.undistort_maps[0],
            self.undistort_maps[1],
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return cv2.warpPerspective(
            und,
            self.homography,
            (self.bev_width, self.bev_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )


class SmartBevGenerator:
    def __init__(
        self,
        data_dir,
        frame_width=640,
        frame_height=480,
        bev_width=1000,
        bev_height=1000,
        car_width=120,
        car_height=180,
        car_center_x=None,
        car_center_y=None,
        focal_scale=1.0,
        size_scale=2.0,
        blend=False,
        balance=False,
        feather=None,
    ):
        global BEV_WIDTH, BEV_HEIGHT, CAR_WIDTH, CAR_HEIGHT
        import SurroundBirdEyeView.surroundBEV as sbev

        sbev.BEV_WIDTH = bev_width
        sbev.BEV_HEIGHT = bev_height
        sbev.CAR_WIDTH = car_width
        sbev.CAR_HEIGHT = car_height
        BEV_WIDTH = bev_width
        BEV_HEIGHT = bev_height
        CAR_WIDTH = car_width
        CAR_HEIGHT = car_height

        self.blend = blend
        self.balance = balance
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.car_width = car_width
        self.car_height = car_height
        self.car_center_x = bev_width // 2 if car_center_x is None else int(round(car_center_x))
        self.car_center_y = bev_height // 2 if car_center_y is None else int(round(car_center_y))
        default_feather = max(21, min(bev_width, bev_height) // 30)
        self.feather = int(default_feather if feather is None else feather)
        if self.feather % 2 == 0:
            self.feather += 1
        camera_specs = [
            ("front", False),
            ("back", False),
            ("left", False),
            ("right", False),
        ]
        self.cameras = [
            SmartCamera(
                name,
                data_dir,
                frame_width,
                frame_height,
                bev_width,
                bev_height,
                bev_focal_scale(name, focal_scale),
                bev_size_scale(name, size_scale),
                pinhole=pinhole,
            )
            for name, pinhole in camera_specs
        ]
        # Always use hard sector masks; soft seams come from distance-feather below.
        # Upstream BlendMask hits OpenCV pointPolygonTest type errors on this Pi build.
        self.masks = [Mask("front"), Mask("back"), Mask("left"), Mask("right")]

    def __call__(self, front, back, left, right, car=None):
        images = [front, back, left, right]
        if self.balance:
            images = luminance_balance(images)
        bevs = [camera.raw2bev(img) for img, camera in zip(images, self.cameras)]

        if not self.blend:
            surround = np.zeros_like(bevs[0])
            for bev, mask in zip(bevs, self.masks):
                m = mask.mask > 0
                surround[m] = bev[m]
        else:
            # Dilate each sector so neighbors overlap, then feather by distance.
            # Weight only where the warped BEV has real content — otherwise
            # empty/black pixels darken the seam ("shadow" bands).
            feather = self.feather
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather * 2 + 1, feather * 2 + 1))
            weights = []
            for bev, mask in zip(bevs, self.masks):
                binary = (mask.mask > 0).astype(np.uint8) * 255
                dilated = cv2.dilate(binary, kernel)
                dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 3)
                w = np.clip(dist / float(feather), 0.0, 1.0).astype(np.float32)
                # Valid ground/content: not near-black after warp.
                content = (bev.astype(np.int16).sum(axis=2) > 18).astype(np.float32)
                # Slight erode to drop noisy single-pixel warp edges.
                content = cv2.erode(content, np.ones((3, 3), np.uint8), iterations=1)
                weights.append(w * content)
            weight_sum = np.sum(weights, axis=0)
            surround = np.zeros_like(bevs[0], dtype=np.float32)
            for bev, w in zip(bevs, weights):
                surround += bev.astype(np.float32) * w[:, :, None]
            # Where at least one camera has content, normalize; else keep black.
            has = weight_sum > 1e-6
            surround[has] = surround[has] / weight_sum[has][:, None]
            surround = np.clip(surround, 0, 255).astype(np.uint8)

        # Keep a clean black car body; dilation for seam blend must not fill the vehicle.
        cx, cy = self.car_center_x, self.car_center_y
        hw, hh = self.car_width // 2, self.car_height // 2
        y0, y1 = max(0, cy - hh), min(self.bev_height, cy + hh)
        x0, x1 = max(0, cx - hw), min(self.bev_width, cx + hw)
        surround[y0:y1, x0:x1] = 0

        if self.balance:
            surround = color_balance(surround)
        if car is not None:
            surround = cv2.add(surround, car)
        # Outline measured car footprint so it is distinct from FOV blind zones.
        cv2.rectangle(surround, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2)
        # Front marker (car nose toward top of BEV).
        mid_x = (x0 + x1) // 2
        cv2.arrowedLine(surround, (mid_x, y0 + 18), (mid_x, y0 + 4), (0, 220, 255), 2, tipLength=0.4)
        cv2.putText(
            surround,
            "CAR",
            (x0 + 6, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        return surround


def prepare_car_overlay(car_path, bev_width, bev_height, car_width, car_height, car_center_x, car_center_y):
    """Resize top-down car art into a BEV-sized overlay (black elsewhere)."""
    car = cv2.imread(str(car_path))
    if car is None:
        raise FileNotFoundError(car_path)
    # White studio background -> black so cv2.add only paints the vehicle.
    gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    car[gray > 245] = 0
    car = cv2.resize(car, (car_width, car_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    cx = bev_width // 2 if car_center_x is None else int(round(car_center_x))
    cy = bev_height // 2 if car_center_y is None else int(round(car_center_y))
    x0 = max(0, cx - car_width // 2)
    y0 = max(0, cy - car_height // 2)
    x1 = min(bev_width, x0 + car_width)
    y1 = min(bev_height, y0 + car_height)
    canvas[y0:y1, x0:x1] = car[: y1 - y0, : x1 - x0]
    return canvas


def load_image(path, width, height):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height))
    return image


def letterbox_to_size(image, out_width, out_height):
    """Fit image into out_width x out_height with black bars (keep aspect ratio)."""
    src_h, src_w = image.shape[:2]
    scale = min(out_width / src_w, out_height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    x0 = (out_width - new_w) // 2
    y0 = (out_height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Smart Car BEV surround stitch.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--bev-width", type=int, default=1000)
    parser.add_argument("--bev-height", type=int, default=1000)
    # Front-up BEV: width=left-right 18cm, height=front-back 29cm @ ~11.36 px/cm (metric layout).
    parser.add_argument("--car-width", type=int, default=205, help="Car body width left-right, px on BEV canvas.")
    parser.add_argument("--car-height", type=int, default=330, help="Car body length front-back, px on BEV canvas.")
    parser.add_argument(
        "--car-center-x",
        type=float,
        default=-1,
        help="Car mask center X on BEV canvas (-1 = canvas mid).",
    )
    parser.add_argument(
        "--car-center-y",
        type=float,
        default=-1,
        help="Car mask center Y on BEV canvas (-1 = canvas mid).",
    )
    parser.add_argument(
        "--car-image",
        default=str(DEFAULT_DATA / "car.jpg"),
        help="Top-down car overlay image. Empty string disables overlay art.",
    )
    parser.add_argument("--no-car-image", action="store_true", help="Do not overlay car.jpg; only draw footprint box.")
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    parser.add_argument("--blend", action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument(
        "--feather",
        type=int,
        default=0,
        help="Seam feather radius in px (0 = auto ~bev/30). Wider softens sparse FOV overlaps.",
    )
    parser.add_argument(
        "--auto-car-size",
        action="store_true",
        help="Scale default 18x29cm car mask with bev canvas size (1000px base).",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="Optional display width. If set with --display-height, letterbox square BEV into this size.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=0,
        help="Optional display height for letterboxed output.",
    )
    args = parser.parse_args()

    if args.auto_car_size:
        # Base car mask measured for 1000x1000 (~18x29cm). Scale X/Y independently
        # so non-square canvases (e.g. 640x480) still keep relative body size.
        args.car_width = int(round(205 * args.bev_width / 1000.0))
        args.car_height = int(round(330 * args.bev_height / 1000.0))
        print(f"Auto car mask: {args.car_width}x{args.car_height} px on {args.bev_width}x{args.bev_height} canvas")

    car_cx = None if args.car_center_x < 0 else args.car_center_x
    car_cy = None if args.car_center_y < 0 else args.car_center_y

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    front = load_image(data_dir / "front" / "front.jpg", args.frame_width, args.frame_height)
    back = load_image(data_dir / "back" / "back.jpg", args.frame_width, args.frame_height)
    left = load_image(data_dir / "left" / "left.jpg", args.frame_width, args.frame_height)
    right = load_image(data_dir / "right" / "right.jpg", args.frame_width, args.frame_height)

    car_overlay = None
    if not args.no_car_image and args.car_image:
        car_path = Path(args.car_image)
        if car_path.exists():
            car_overlay = prepare_car_overlay(
                car_path,
                args.bev_width,
                args.bev_height,
                args.car_width,
                args.car_height,
                car_cx,
                car_cy,
            )
            print(f"Car overlay: {car_path} -> {args.car_width}x{args.car_height} px")
        else:
            print(f"WARNING: car image missing ({car_path}); drawing footprint box only")

    bev = SmartBevGenerator(
        data_dir,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        bev_width=args.bev_width,
        bev_height=args.bev_height,
        car_width=args.car_width,
        car_height=args.car_height,
        car_center_x=car_cx,
        car_center_y=car_cy,
        focal_scale=args.focal_scale,
        size_scale=args.size_scale,
        blend=args.blend,
        balance=args.balance,
        feather=(None if args.feather <= 0 else args.feather),
    )
    if args.blend:
        print(f"Seam feather: {bev.feather} px")
    surround = bev(front, back, left, right, car=car_overlay)
    square_path = output_dir / "surround_square.jpg"
    cv2.imwrite(str(square_path), surround)
    print(f"Saved square BEV: {square_path} ({args.bev_width}x{args.bev_height})")

    if args.display_width > 0 and args.display_height > 0:
        display = letterbox_to_size(surround, args.display_width, args.display_height)
        out_path = output_dir / "surround.jpg"
        cv2.imwrite(str(out_path), display)
        print(f"Saved display: {out_path} ({args.display_width}x{args.display_height}, letterboxed)")
    else:
        out_path = output_dir / "surround.jpg"
        cv2.imwrite(str(out_path), surround)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
