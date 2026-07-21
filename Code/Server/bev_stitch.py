#!/usr/bin/env python3
"""Run Smart Car BEV stitch using deployed K/D/H (pinhole front + fisheye USB)."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parent
SURROUND_ROOT = Path.home() / "CameraCalibration-test"
sys.path.insert(0, str(SURROUND_ROOT))

_argv = sys.argv[:]
sys.argv = [_argv[0]]
from SurroundBirdEyeView.surroundBEV import (  # noqa: E402
    BlendMask,
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
        bev_size = (self.bev_width, self.bev_height)
        map1 = cv2.warpPerspective(self.undistort_maps[0], self.homography, bev_size)
        map2 = cv2.warpPerspective(self.undistort_maps[1], self.homography, bev_size)
        return map1, map2

    def raw2bev(self, img):
        return cv2.remap(img, *self.bev_maps, interpolation=cv2.INTER_LINEAR)


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
        focal_scale=1.0,
        size_scale=2.0,
        blend=False,
        balance=False,
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
        self.cameras = [
            SmartCamera("front", data_dir, frame_width, frame_height, bev_width, bev_height, focal_scale, size_scale, pinhole=True),
            SmartCamera("back", data_dir, frame_width, frame_height, bev_width, bev_height, focal_scale, size_scale),
            SmartCamera("left", data_dir, frame_width, frame_height, bev_width, bev_height, focal_scale, size_scale),
            SmartCamera("right", data_dir, frame_width, frame_height, bev_width, bev_height, focal_scale, size_scale),
        ]
        if not blend:
            self.masks = [Mask("front"), Mask("back"), Mask("left"), Mask("right")]
        else:
            self.masks = [BlendMask("front"), BlendMask("back"), BlendMask("left"), BlendMask("right")]

    def __call__(self, front, back, left, right, car=None):
        images = [front, back, left, right]
        if self.balance:
            images = luminance_balance(images)
        images = [
            mask(camera.raw2bev(img))
            for img, mask, camera in zip(images, self.masks, self.cameras)
        ]
        surround = cv2.add(images[0], images[1])
        surround = cv2.add(surround, images[2])
        surround = cv2.add(surround, images[3])
        if self.balance:
            surround = color_balance(surround)
        if car is not None:
            surround = cv2.add(surround, car)
        return surround


def load_image(path, width, height):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height))
    return image


def main():
    parser = argparse.ArgumentParser(description="Smart Car BEV surround stitch.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--bev-width", type=int, default=1000)
    parser.add_argument("--bev-height", type=int, default=1000)
    parser.add_argument("--car-width", type=int, default=120)
    parser.add_argument("--car-height", type=int, default=180)
    parser.add_argument("--focal-scale", type=float, default=1.0)
    parser.add_argument("--size-scale", type=float, default=2.0)
    parser.add_argument("--blend", action="store_true")
    parser.add_argument("--balance", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    front = load_image(data_dir / "front" / "front.jpg", args.frame_width, args.frame_height)
    back = load_image(data_dir / "back" / "back.jpg", args.frame_width, args.frame_height)
    left = load_image(data_dir / "left" / "left.jpg", args.frame_width, args.frame_height)
    right = load_image(data_dir / "right" / "right.jpg", args.frame_width, args.frame_height)

    bev = SmartBevGenerator(
        data_dir,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        bev_width=args.bev_width,
        bev_height=args.bev_height,
        car_width=args.car_width,
        car_height=args.car_height,
        focal_scale=args.focal_scale,
        size_scale=args.size_scale,
        blend=args.blend,
        balance=args.balance,
    )
    surround = bev(front, back, left, right)
    out_path = output_dir / "surround.jpg"
    cv2.imwrite(str(out_path), surround)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
