#!/usr/bin/env python3
"""Live gimbal view. Drag the sliders or use keys, then Enter to save the pose."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from camera_devices import resolve_usb_capture_index
from pca9685 import PCA9685
from servo import HOME_PATH, load_gimbal_home, load_right_90_pan

HERE = Path(__file__).resolve().parent
OUT = HERE / "gimbal_manual.json"
WIN = "gimbal adjust"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right", "forward"), default="right")
    parser.add_argument("--pan", type=int, default=None, help="Start at this servo angle instead of slewing across.")
    parser.add_argument("--from-pan", type=int, default=None, help="Slew from this angle to --pan.")
    args = parser.parse_args()
    home_pan, home_tilt = load_gimbal_home()
    right_90 = load_right_90_pan(home_pan)
    start_tilt = home_tilt
    if args.pan is not None:
        start_pan = max(0, min(180, args.pan))
        from_pan = start_pan if args.from_pan is None else max(0, min(180, args.from_pan))
    elif args.side == "left":
        start_pan = max(0, home_pan - (right_90 - home_pan))
        from_pan = right_90
    else:
        start_pan, from_pan = right_90, home_pan

    hw = json.loads((HERE / "camera_hardware.json").read_text(encoding="utf-8"))
    index, note = resolve_usb_capture_index(hw["gimbal"])
    print(note, flush=True)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"cannot open gimbal camera /dev/video{index}", flush=True)
        return 1

    pwm = PCA9685(0x40, debug=False)
    pan_ch, tilt_ch = 8, 9

    def pulse(channel: str, angle: int) -> int:
        error = 10
        angle = int(angle)
        if channel == "0":
            return 2500 - int((angle + error) / 0.09)
        return 500 + int((angle + error) / 0.09)

    state = {"pan": start_pan, "tilt": start_tilt, "ready": False}

    def apply() -> None:
        pwm.set_servo_pulse(pan_ch, pulse("0", state["pan"]))
        pwm.set_servo_pulse(tilt_ch, pulse("1", state["tilt"]))

    def on_pan(value: int) -> None:
        if not state["ready"]:
            return
        state["pan"] = int(value)
        apply()

    def on_tilt(value: int) -> None:
        if not state["ready"]:
            return
        state["tilt"] = int(value)
        apply()

    def save(confirmed: bool) -> None:
        payload = {
            "side": args.side,
            "pan": state["pan"],
            "tilt": state["tilt"],
            "home_pan": home_pan,
            "confirmed": confirmed,
        }
        OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if confirmed:
            try:
                home = json.loads(HOME_PATH.read_text(encoding="utf-8"))
            except Exception:
                home = {"pan": home_pan, "tilt": home_tilt}
            key = {"left": "left_90_pan", "right": "right_90_pan", "forward": "pan"}[args.side]
            home[key] = state["pan"]
            home["tilt"] = state["tilt"]
            HOME_PATH.write_text(json.dumps(home, indent=2) + "\n", encoding="utf-8")
            print(f"saved {key}={state['pan']} tilt={state['tilt']} -> {HOME_PATH}", flush=True)
        print(f"pan={state['pan']} tilt={state['tilt']} confirmed={confirmed}", flush=True)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 600)
    cv2.createTrackbar("pan", WIN, start_pan, 180, on_pan)
    cv2.createTrackbar("tilt", WIN, start_tilt, 180, on_tilt)
    state["ready"] = True
    current = from_pan
    apply_angle = state["pan"]
    state["pan"] = from_pan
    while abs(apply_angle - state["pan"]) > 0:
        step = max(-12, min(12, apply_angle - state["pan"]))
        state["pan"] += step
        apply()
        if state["pan"] != apply_angle:
            time.sleep(0.08)
    state["pan"] = apply_angle
    apply()
    label = "straight ahead" if args.side == "forward" else f"{args.side} 90 deg"
    print(f"window open for {label}, starting pan={state['pan']}. "
          "a/d pan, w/s tilt, Enter save, q close.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        view = cv2.resize(frame, (800, 600))
        lines = [
            f"set {label}",
            f"pan {state['pan']}   tilt {state['tilt']}   home {home_pan}",
            "a/d pan 1 deg    w/s tilt 1 deg",
            "Enter save this pose     q close",
        ]
        for i, text in enumerate(lines):
            cv2.putText(view, text, (16, 36 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (16, 36 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            save(False)
            break
        if key in (13, 10, ord(" ")):
            save(True)
        elif key == ord("a"):
            state["pan"] = max(0, state["pan"] - 1)
            cv2.setTrackbarPos("pan", WIN, state["pan"])
            apply()
        elif key == ord("d"):
            state["pan"] = min(180, state["pan"] + 1)
            cv2.setTrackbarPos("pan", WIN, state["pan"])
            apply()
        elif key == ord("w"):
            state["tilt"] = min(180, state["tilt"] + 1)
            cv2.setTrackbarPos("tilt", WIN, state["tilt"])
            apply()
        elif key == ord("s"):
            state["tilt"] = max(0, state["tilt"] - 1)
            cv2.setTrackbarPos("tilt", WIN, state["tilt"])
            apply()

    cap.release()
    cv2.destroyAllWindows()
    print("closed. servo stays at the last angle.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nclosed", flush=True)
        raise SystemExit(0)
