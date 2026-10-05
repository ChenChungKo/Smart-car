#!/usr/bin/env python3
"""Aim the gimbal ultrasonic at the wall the front CSI sees.

Front CSI (camera_num=1) only gives bearing. After the car is still, the SG90
pans the HC-SR04 toward that heading. The USB gimbal camera is a look-through
view so you can check the heading; it does not estimate centimetres or wall
feet. Distance is the ultrasonic reading.

Do not pan while driving: a side echo would be treated as a clear path ahead.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from bev_extrinsic import load_kd
from camera_devices import (
    create_csi_still_configuration,
    csi_array_to_bgr,
    resolve_usb_capture_index,
)
from servo import Servo, load_gimbal_home
from ultrasonic import Ultrasonic
from vision_detector import classic_corridor_score, draw_debug, wall_aim_point

SERVER = Path(__file__).resolve().parent
CFG_PATH = SERVER / "csi_wall_aim.json"
HW_PATH = SERVER / "camera_hardware.json"
DEBUG_DIR = SERVER / "check_front_csi_debug"

DEFAULT_CFG = {
    "pan_sign": 1,
    "csi_yaw_offset_deg": 0.0,
    "max_pan_offset_deg": 50.0,
    "settle_s": 0.5,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Point HC-SR04 at the CSI wall heading; gimbal USB is a look-through check."
    )
    p.add_argument("--follow", action="store_true", help="Keep aiming while the window is open.")
    p.add_argument("--no-gimbal-cam", action="store_true")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--hz", type=float, default=5.0)
    return p.parse_args()


def load_cfg() -> dict:
    cfg = dict(DEFAULT_CFG)
    try:
        data = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        for key, value in DEFAULT_CFG.items():
            if key in data:
                cfg[key] = type(value)(data[key])
    except Exception:
        pass
    cfg["pan_sign"] = 1 if int(cfg["pan_sign"]) >= 0 else -1
    return cfg


def save_cfg(cfg: dict) -> None:
    payload = dict(cfg)
    payload["notes"] = (
        "pan_sign +1: CSI right (positive yaw) increases servo 0. a/left decreases pan."
    )
    CFG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_front_kd():
    k, d = load_kd("front")
    k = k.astype(np.float64)
    d = d.reshape(-1, 1)[:4].astype(np.float64)
    return k, d


def pixel_yaw_deg(x: float, y: float, k: np.ndarray | None, d: np.ndarray | None, width: int) -> float:
    if k is not None and d is not None:
        pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
        try:
            xn, _yn = cv2.fisheye.undistortPoints(pts, k, d)[0, 0]
            return math.degrees(math.atan(float(xn)))
        except cv2.error:
            pass
    fx = max(80.0, float(k[0, 0]) if k is not None else width / math.pi)
    cx = float(k[0, 2]) if k is not None else width / 2.0
    return math.degrees(math.atan((float(x) - cx) / fx))


def pan_for_yaw(home_pan: int, yaw_deg: float, cfg: dict) -> int:
    signed = float(cfg["pan_sign"]) * (yaw_deg + float(cfg["csi_yaw_offset_deg"]))
    if abs(signed) < 4.0:
        signed = 0.0
    limited = max(-float(cfg["max_pan_offset_deg"]), min(float(cfg["max_pan_offset_deg"]), signed))
    return int(round(max(0, min(180, home_pan + limited))))


def read_sonic_median(sonic: Ultrasonic | None, n: int = 5) -> float | None:
    if sonic is None:
        return None
    values = []
    for _ in range(n):
        cm = sonic.get_distance()
        if cm is not None and 1.0 <= float(cm) <= 300.0:
            values.append(float(cm))
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


def open_gimbal_usb():
    hw = json.loads(HW_PATH.read_text(encoding="utf-8"))
    index, how = resolve_usb_capture_index(hw["gimbal"])
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open gimbal {how}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 8)
    time.sleep(0.4)
    for _ in range(6):
        cap.grab()
    return cap, how


def grab_usb(cap, n: int = 1) -> np.ndarray | None:
    if cap is None:
        return None
    frame = None
    for _ in range(max(1, n)):
        ok, got = cap.read()
        if ok and got is not None:
            frame = got
    if frame is None:
        return None
    if frame.shape[1] != 640 or frame.shape[0] != 480:
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    return frame


def csi_bearing(frame: np.ndarray, k, d) -> tuple[dict | None, float | None, dict]:
    scores = classic_corridor_score(frame)
    aim = wall_aim_point(frame, circle=True)
    if aim is None:
        return None, None, scores
    yaw = pixel_yaw_deg(aim["x"], aim["y"], k, d, frame.shape[1])
    return aim, yaw, scores


class AimSmoother:
    """Hold the CSI aim mark if a frame jumps off the wall foot."""

    def __init__(self, hold_px: int = 30, lock_frames: int = 3, window: int = 5):
        self.hold_px = hold_px
        self.lock_frames = lock_frames
        self.window = window
        self.held: dict | None = None
        self.ys: list[int] = []
        self.xs: list[int] = []
        self.pending: list[dict] = []

    def update(self, aim: dict | None) -> dict | None:
        if aim is None:
            return self.held
        x, y = int(aim["x"]), int(aim["y"])
        if self.held is not None and abs(y - int(self.held["y"])) > self.hold_px:
            self.pending.append(dict(aim))
            self.pending = self.pending[-self.lock_frames :]
            if len(self.pending) < self.lock_frames:
                return self.held
            ys = [int(p["y"]) for p in self.pending]
            if max(ys) - min(ys) > 12:
                return self.held
            self.held = dict(aim)
            self.ys = ys
            self.xs = [int(p["x"]) for p in self.pending]
            self.pending = []
            return self.held
        self.pending = []
        self.ys.append(y)
        self.xs.append(x)
        self.ys = self.ys[-self.window :]
        self.xs = self.xs[-self.window :]
        out = dict(aim)
        out["y"] = int(round(float(np.median(np.asarray(self.ys, dtype=np.float32)))))
        out["x"] = int(round(float(np.median(np.asarray(self.xs, dtype=np.float32)))))
        self.held = out
        return out


def annotate_csi(frame, aim, yaw, pan, home_pan, sonic_cm, scores):
    out = draw_debug(frame, [], scores, "front-csi bearing")
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (180, 180, 180), 1)
    if aim is not None:
        x, y = int(aim["x"]), int(aim["y"])
        cv2.line(out, (0, y), (w, y), (0, 0, 255), 2)
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
        cv2.line(out, (x, max(0, y - 40)), (x, min(h - 1, y + 40)), (0, 255, 255), 2)
    yaw_txt = "yaw=--" if yaw is None else f"yaw={yaw:+.1f}deg"
    sonic_txt = "--" if sonic_cm is None else f"{sonic_cm:.0f}cm"
    cv2.putText(out, yaw_txt, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"pan={pan} (home {home_pan})  sonic={sonic_txt}",
        (8, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def annotate_gimbal(frame, pan, home_pan, sonic_cm):
    """Look-through only: centre crosshair, no wall-foot tracking."""
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(out, (cx, 0), (cx, h), (0, 255, 255), 1)
    cv2.line(out, (0, cy), (w, cy), (0, 255, 255), 1)
    cv2.circle(out, (cx, cy), 8, (0, 255, 255), 2)
    heading = "HOME" if pan == home_pan else f"AIM pan={pan}"
    sonic_txt = "--" if sonic_cm is None else f"{sonic_cm:.0f}cm"
    cv2.putText(
        out,
        f"gimbal look  {heading}",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"sonic={sonic_txt}   wall should sit on the cross if heading is right",
        (8, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def compose(csi, gimbal):
    if gimbal is None:
        return csi
    if gimbal.shape != csi.shape:
        gimbal = cv2.resize(gimbal, (csi.shape[1], csi.shape[0]))
    return np.hstack([csi, gimbal])


def apply_pan(servo: Servo, pan: int, tilt: int) -> None:
    servo.set_servo_pwm("0", pan)
    servo.set_servo_pwm("1", tilt)


def aim_ultrasonic(
    servo: Servo,
    sonic: Ultrasonic,
    gimbal_cap,
    home_pan: int,
    tilt: int,
    yaw_deg: float,
    cfg: dict,
) -> dict:
    pan = pan_for_yaw(home_pan, yaw_deg, cfg)
    apply_pan(servo, pan, tilt)
    time.sleep(max(0.35, float(cfg["settle_s"])))
    apply_pan(servo, pan, tilt)
    grab_usb(gimbal_cap, n=3)
    sonic_cm = read_sonic_median(sonic)
    return {"pan": pan, "sonic_cm": sonic_cm}


def main() -> int:
    args = parse_args()
    cfg = load_cfg()
    home_pan, home_tilt = load_gimbal_home()
    front_k, front_d = load_front_kd()

    servo = Servo()
    apply_pan(servo, home_pan, home_tilt)
    time.sleep(0.35)
    sonic = Ultrasonic()

    from picamera2 import Picamera2

    camera = Picamera2(camera_num=1)
    camera.configure(create_csi_still_configuration(camera, 640, 480))
    camera.start()
    camera.set_controls({"AeEnable": True, "AwbEnable": True})
    time.sleep(0.7)
    for _ in range(5):
        camera.capture_array()

    gimbal_cap = None
    gimbal_how = "disabled"
    if not args.no_gimbal_cam:
        try:
            gimbal_cap, gimbal_how = open_gimbal_usb()
        except Exception as exc:
            print(f"雲台相機未開（{exc}），仍可用 CSI 方位 + 超音波。", flush=True)
            gimbal_cap = None

    print("左：CSI 標牆壁方位。右：雲台即時畫面，只用來確認有沒有轉對，不抓牆腳。")
    print("空白鍵轉向 CSI 方位並量超音波。h 回正前方。牆應落在右邊十字上。")
    print("p 反轉左右、[ ] 微調 CSI 與雲台的零度差。開車時不要轉雲台。")
    print(
        f"home pan={home_pan} tilt={home_tilt} sign={cfg['pan_sign']} "
        f"offset={cfg['csi_yaw_offset_deg']:.1f} gimbal={gimbal_how}",
        flush=True,
    )

    show = not args.no_window
    if show:
        try:
            cv2.namedWindow("csi-aim-sonic", cv2.WINDOW_NORMAL)
        except Exception as exc:
            print(f"無法開視窗（{exc}），改為只印數值。")
            show = False

    pan = home_pan
    last_yaw = None
    last_follow = 0.0
    last_sonic_at = 0.0
    sonic_cm: float | None = None
    csi_smooth = AimSmoother()
    period = 1.0 / max(args.hz, 0.5)

    try:
        while True:
            started = time.monotonic()
            apply_pan(servo, pan, home_tilt)
            frame = csi_array_to_bgr(camera.capture_array())
            if frame is None:
                continue
            aim, yaw, scores = csi_bearing(frame, front_k, front_d)
            aim = csi_smooth.update(aim)
            if aim is not None:
                yaw = pixel_yaw_deg(aim["x"], aim["y"], front_k, front_d, frame.shape[1])

            if args.follow and yaw is not None and started - last_follow > 1.2:
                if last_yaw is None or abs(yaw - last_yaw) >= 3.0 or pan == home_pan:
                    result = aim_ultrasonic(
                        servo, sonic, gimbal_cap, home_pan, home_tilt, yaw, cfg
                    )
                    pan = int(result["pan"])
                    sonic_cm = result["sonic_cm"]
                    last_yaw = yaw
                    last_follow = started
                    last_sonic_at = started
                    print(
                        f"follow CSI {yaw:+.1f}deg -> pan {pan}  sonic={sonic_cm}",
                        flush=True,
                    )

            live_gimbal = grab_usb(gimbal_cap)
            if started - last_sonic_at > 0.45:
                sonic_cm = read_sonic_median(sonic, n=3)
                last_sonic_at = started

            csi_view = annotate_csi(frame, aim, yaw, pan, home_pan, sonic_cm, scores)
            if live_gimbal is None:
                gimbal_view = np.zeros_like(csi_view)
                cv2.putText(
                    gimbal_view,
                    "no gimbal camera",
                    (24, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
            else:
                gimbal_view = annotate_gimbal(live_gimbal, pan, home_pan, sonic_cm)
            overlay = compose(csi_view, gimbal_view)
            cv2.putText(
                overlay,
                "SPACE aim   h home   p flip   [ ] yaw0   s save   q quit",
                (8, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            if show:
                cv2.imshow("csi-aim-sonic", overlay)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF
                print(
                    f"csi_yaw={None if yaw is None else round(yaw,1)} pan={pan}",
                    flush=True,
                )

            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                pan = home_pan
                apply_pan(servo, pan, home_tilt)
                print(f"已回正前方 pan={pan}", flush=True)
            if key == ord("p"):
                cfg["pan_sign"] = -int(cfg["pan_sign"])
                save_cfg(cfg)
                print(f"pan_sign={cfg['pan_sign']}（若轉向反了再按一次 p）", flush=True)
            if key in (ord("["), ord("-")):
                cfg["csi_yaw_offset_deg"] = float(cfg["csi_yaw_offset_deg"]) - 1.0
                save_cfg(cfg)
                print(f"csi_yaw_offset={cfg['csi_yaw_offset_deg']:.1f}", flush=True)
            if key in (ord("]"), ord("=")):
                cfg["csi_yaw_offset_deg"] = float(cfg["csi_yaw_offset_deg"]) + 1.0
                save_cfg(cfg)
                print(f"csi_yaw_offset={cfg['csi_yaw_offset_deg']:.1f}", flush=True)
            if key == ord("s"):
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                path = DEBUG_DIR / time.strftime("aim_%H%M%S.jpg")
                cv2.imwrite(str(path), overlay)
                print(f"saved {path}", flush=True)
            if key == ord(" "):
                if yaw is None:
                    print("CSI 沒看到牆，不轉雲台。", flush=True)
                else:
                    result = aim_ultrasonic(
                        servo, sonic, gimbal_cap, home_pan, home_tilt, yaw, cfg
                    )
                    pan = int(result["pan"])
                    sonic_cm = result["sonic_cm"]
                    last_sonic_at = time.monotonic()
                    print(
                        f"CSI {yaw:+.1f}deg -> pan {home_pan}->{pan}  sonic={sonic_cm}cm  "
                        "請看右邊十字是否對準牆壁。",
                        flush=True,
                    )
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nEnd of program")
    finally:
        try:
            apply_pan(servo, home_pan, home_tilt)
        except Exception:
            pass
        try:
            camera.stop()
            camera.close()
        except Exception:
            pass
        if gimbal_cap is not None:
            gimbal_cap.release()
        sonic.close()
        if show:
            cv2.destroyAllWindows()
        print(f"雲台已回正前方 pan={home_pan}。巡航開車時仍鎖在這個角度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
