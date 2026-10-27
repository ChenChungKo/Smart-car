#!/usr/bin/env python3
"""Point the gimbal straight ahead every time the car is switched on.

The car has two switches (Freenove tutorial, chapter 3): S1 = power main switch (the control board: 5 V, 3.3 V, the
sensors and the battery ADC), S2 = motor and servo power switch. The Pi can stay on from its own USB-C supply. A servo
that gets power with no command stays wherever it is, and gimbal-center.service only runs once, at boot.

The program reads the battery voltage through the ADC (S1 on: 5 V or more; off: no reading), two readings in a row
each way. The ADC cannot tell S2. So when S1 goes on, the gimbal is centred and the pulses are KEPT for HOLD_S seconds
(a servo powered later by S2 then goes straight to the centre), and only then released so it does not buzz. Switch
S1, then S2, within that time. If S2 is switched alone later, run gimbal_center.py by hand.

    python3 gimbal_power_watch.py            # forever (systemd/gimbal-power-watch.service)
    python3 gimbal_power_watch.py --debug    # print every reading
    python3 gimbal_power_watch.py --center-now
"""

from __future__ import annotations

import argparse
import sys
import time

from gimbal_center import main as center_gimbal
from servo import Servo
from safe_surround_cruise import read_battery_voltage

HOLD_S = 60.0    # keep the centre pulses this long after S1 goes on: time to switch S2
ON_V = 5.0       # the 2S pack is 6-8.4 V; the ADC reads about 0 (or fails) with the battery switch off
POLL_S = 2.0


def _uptime_s() -> float:
    return float(open("/proc/uptime").read().split()[0])


def is_on(voltage) -> bool:
    return voltage is not None and voltage >= ON_V


def center_and_hold(hold_s: float = HOLD_S) -> None:
    """Gimbal straight ahead, pulses kept for hold_s seconds, then released."""
    for attempt in range(10):
        try:
            servo = Servo()
            servo.apply_gimbal_home()
            time.sleep(hold_s)
            servo.release_gimbal()
            return
        except Exception as exc:  # noqa: BLE001 - the I2C bus can answer 'resource temporarily unavailable'
            print(f"gimbal not ready ({exc}); retrying", flush=True)
            time.sleep(0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--center-now", action="store_true")
    args = ap.parse_args()
    if args.center_now:
        return center_gimbal()
    if is_on(read_battery_voltage()) and _uptime_s() < 300:   # just booted with S1 already on
        print("booted with S1 on: gimbal straight ahead, held for S2", flush=True)
        center_and_hold()
    state = None          # None until two readings agree, then True/False
    pending, count = None, 0
    while True:
        voltage = read_battery_voltage()
        now_on = is_on(voltage)
        if args.debug:
            print(f"{time.strftime('%H:%M:%S')} battery {voltage} V -> {'on' if now_on else 'off'}", flush=True)
        if now_on == pending:
            count += 1
        else:
            pending, count = now_on, 1
        if count >= 2 and pending != state:
            if state is False and pending is True:
                print(f"S1 switched on ({voltage} V): gimbal straight ahead, held {HOLD_S:.0f} s for S2", flush=True)
                center_and_hold()
            elif state is not None:
                print("battery switched off", flush=True)
            state = pending
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
