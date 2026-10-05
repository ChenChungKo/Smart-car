#!/usr/bin/env python3
"""Hold the SG90 gimbal at straight-ahead and optionally trim/save that pose."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty

from servo import Servo, load_gimbal_home, save_gimbal_home


def _read_key(timeout_s: float) -> str | None:
    fd = sys.stdin.fileno()
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        extra = ""
        while select.select([sys.stdin], [], [], 0.0)[0]:
            extra += sys.stdin.read(1)
        if extra.startswith("[A") or extra == "A":
            return "up"
        if extra.startswith("[B") or extra == "B":
            return "down"
        if extra.startswith("[C") or extra == "C":
            return "right"
        if extra.startswith("[D") or extra == "D":
            return "left"
        return None
    return ch


def main() -> int:
    pan, tilt = load_gimbal_home()
    print("設定車頭 SG90：一開始就轉到正前方並維持。")
    print("請先關閉巡航與全車測試 GUI，避免搶 PCA9685。")
    print("若機械上不正，可趁伺服維持此角度時鬆開雲台螺絲、對準車頭再鎖緊。")
    print("也可按鍵微調：")
    print("  a / ←  左轉    d / →  右轉")
    print("  w / ↑  抬頭    s / ↓  低頭")
    print("  0      回到 90/90")
    print("  Enter  儲存為正前方")
    print("  q      離開（維持目前角度）")
    print(flush=True)

    servo = Servo()
    servo.set_servo_pwm("0", pan)
    servo.set_servo_pwm("1", tilt)
    print(f"目前正前方 pan={pan} tilt={tilt}", flush=True)

    if not sys.stdin.isatty():
        print("沒有互動終端機，已轉到正前方後結束。要微調請在終端機執行本程式。")
        return 0

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            servo.set_servo_pwm("0", pan)
            servo.set_servo_pwm("1", tilt)
            key = _read_key(0.25)
            if key is None:
                continue
            if key in ("q", "Q", "\x03"):
                break
            if key in ("\r", "\n"):
                path = save_gimbal_home(pan, tilt)
                print(f"\n已儲存正前方 pan={pan} tilt={tilt} -> {path}", flush=True)
                continue
            if key == "0":
                pan, tilt = 90, 90
            elif key in ("a", "left"):
                pan = max(0, pan - 2)
            elif key in ("d", "right"):
                pan = min(180, pan + 2)
            elif key in ("w", "up"):
                tilt = min(180, tilt + 2)
            elif key in ("s", "down"):
                tilt = max(0, tilt - 2)
            else:
                continue
            print(f"\r正前方 pan={pan} tilt={tilt}   ", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\n離開設定。伺服維持最後角度；巡航啟動時會讀取已儲存的正前方。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEnd of program")
        raise SystemExit(0)
