#!/usr/bin/env python3
"""Live ultrasonic check with the gimbal locked at the saved straight-ahead pose."""

from __future__ import annotations

import argparse
import time

from servo import Servo, load_gimbal_home
from ultrasonic import Ultrasonic


def read_burst(sonic: Ultrasonic, n: int = 3) -> tuple[list[float | None], float | None]:
    raws: list[float | None] = []
    valid: list[float] = []
    for _ in range(n):
        cm = sonic.get_distance()
        raws.append(cm)
        if cm is not None and 1.0 <= float(cm) <= 300.0:
            valid.append(float(cm))
    if not valid:
        return raws, None
    valid.sort()
    return raws, valid[len(valid) // 2]


def parse_args():
    p = argparse.ArgumentParser(
        description="Confirm HC-SR04 distance with the SG90 facing straight ahead."
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Stop after N seconds. 0 means until Ctrl+C.",
    )
    p.add_argument(
        "--expect",
        type=float,
        default=0.0,
        help="Tape-measure centimetres from the sensor face to the target.",
    )
    p.add_argument("--hz", type=float, default=4.0)
    p.add_argument("--no-servo", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pan, tilt = load_gimbal_home()
    servo = None
    if not args.no_servo:
        servo = Servo()
        servo.set_servo_pwm("0", pan)
        servo.set_servo_pwm("1", tilt)
        print(f"Gimbal locked straight-ahead pan={pan} tilt={tilt}")
        time.sleep(0.45)
    else:
        print("Servo not moved (--no-servo).")

    print("量測基準：超音波金屬探頭表面到正前方平面。請用卷尺對照。")
    print("建議先對 15、20、30、50 cm 各看幾秒。Ctrl+C 結束。")
    if args.expect > 0:
        print(f"預期距離 expect={args.expect:.1f} cm")
    print(flush=True)

    sonic = Ultrasonic()
    period = 1.0 / max(args.hz, 0.5)
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    medians: list[float] = []
    try:
        while True:
            started = time.monotonic()
            raws, median = read_burst(sonic)
            raw_txt = "/".join("--" if v is None else f"{v:.1f}" for v in raws)
            extra = ""
            if median is not None:
                medians.append(median)
                if args.expect > 0:
                    extra = f"  err={median - args.expect:+.1f}cm"
            print(
                f"raw={raw_txt}  median={median if median is not None else '--'}cm{extra}",
                flush=True,
            )
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nEnd of program")
    finally:
        sonic.close()

    if medians:
        medians.sort()
        mid = medians[len(medians) // 2]
        spread = max(medians) - min(medians)
        print(
            f"summary n={len(medians)} median={mid:.1f}cm "
            f"min={min(medians):.1f} max={max(medians):.1f} spread={spread:.1f}cm"
        )
        if args.expect > 0:
            print(f"expect={args.expect:.1f}cm  error={mid - args.expect:+.1f}cm")
            if abs(mid - args.expect) <= 3.0:
                print("結果：誤差在 ±3 cm 內，巡航可直接用。")
            elif abs(mid - args.expect) <= 8.0:
                print("結果：有偏差，先確認卷尺是從探頭量、目標是平面。")
            else:
                print("結果：偏差偏大，檢查探頭朝向、接線與是否量到斜面／地面。")
    else:
        print("summary: 沒有有效讀數。檢查車板電源、HC-SR04 接線與 Echo 腳。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
