#!/usr/bin/env python3
"""Where each bird's-eye camera sees the car itself (wheels, plates, cables): saved as a fixed mask per camera.

    python3 tools/ego_masks.py                 # take the pictures now; nothing drives
    python3 tools/ego_masks.py --from DIR      # from saved front.jpg, rear.jpg, left.jpg, right.jpg

Stand the car on an empty floor: nothing dark or red near it. The parts of the car do not move in a camera's picture,
so the masks stay good until a cable or a camera is moved: then run this again.

dashboard/ipm.py leaves these pixels out of the bird's-eye view: they stand above the floor, and projected onto it they
would look like things lying next to the car. Output: dashboard/ego/<camera>.png, and an overlay to check them in
route_follow_debug/ego_masks.jpg.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
OUT = SERVER / "dashboard" / "ego"
CAMERAS = ("front", "rear", "left", "right")


def ego_mask(frame: np.ndarray) -> np.ndarray:
    """The car: dark or strongly red, in the lower half of the picture, and connected to the picture's bottom edge."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] < 12) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 70)
    part = (hsv[..., 2] < (45 if w < 1000 else 95)) | red      # the USB floor falls to 60 at the picture's edge, the car is < 30
    part[: h // 2] = False
    part = part.astype(np.uint8)
    part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, np.ones((w // 160 | 1, w // 160 | 1), np.uint8))
    round_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w // 24 | 1, w // 24 | 1))
    pad = w // 24
    part = cv2.morphologyEx(np.vstack([part, np.ones((pad, w), np.uint8)]), cv2.MORPH_OPEN, round_k)[:h]
    # opening: a thin dark streak on the floor does not reach the car; the padding keeps what lies along the bottom edge
    _n, labels = cv2.connectedComponents(part)
    keep = np.isin(labels, [i for i in np.unique(labels[-1, :]) if i != 0])
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w // 40 | 1, w // 40 | 1))
    return cv2.dilate(keep.astype(np.uint8), grow) > 0


def grab() -> dict:
    from aruco_localizer import CAR_PATH, load_json, open_grabbers
    sys.path.insert(0, str(SERVER / "tools"))
    from static_survey import fresh_read, open_usb

    grabbers = open_grabbers(load_json(CAR_PATH), 1280, 960, rear=True)
    caps = open_usb(load_json(SERVER / "camera_hardware.json"))
    try:
        time.sleep(1.5)
        frames = {n: g.latest()[0].copy() for n, g in grabbers.items()}
        for name in ("left", "right"):
            ok, frame = fresh_read(caps[name])
            if not ok:
                raise SystemExit(f"{name}: no picture")
            frames[name] = frame
        return frames
    finally:
        for g in grabbers.values():
            g.close()
        for c in caps.values():
            c.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", type=Path, help="folder with front.jpg, rear.jpg, left.jpg, right.jpg")
    args = ap.parse_args()
    frames = {n: cv2.imread(str(args.source / f"{n}.jpg")) for n in CAMERAS} if args.source else grab()
    OUT.mkdir(exist_ok=True)
    tiles = []
    for name in CAMERAS:
        mask = ego_mask(frames[name])
        cv2.imwrite(str(OUT / f"{name}.png"), mask.astype(np.uint8) * 255)
        view = frames[name].copy()
        view[mask] = (view[mask] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
        tiles.append(cv2.resize(view, (480, 360)))
        print(f"{name:6s} {100 * mask.mean():4.1f}% of the picture is the car")
    check = SERVER / "route_follow_debug" / "ego_masks.jpg"
    cv2.imwrite(str(check), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print(f"masks in {OUT}; check them in {check}")


if __name__ == "__main__":
    main()
