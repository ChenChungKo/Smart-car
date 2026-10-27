#!/usr/bin/env python3
"""Point the gimbal straight ahead every time the car's battery is switched on.

The Pi stays up while the car's battery is switched off and on, but the servos are powered by that battery: a
servo that gets power with no command stays wherever it is. gimbal-center.service only runs once, at boot. This
watches the battery voltage (the car's ADC, 2-3 s apart) and centres the gimbal when it goes from off (no reading
or under 5 V) to on (5 V or more) -- for two readings in a row each way, so a flicker does nothing.

    python3 gimbal_power_watch.py            # forever (systemd/gimbal-power-watch.service)
    python3 gimbal_power_watch.py --debug    # print every reading
    python3 gimbal_power_watch.py --center-now
"""

from __future__ import annotations

import argparse
import sys
import time

from gimbal_center import main as center_gimbal
from safe_surround_cruise import read_battery_voltage

ON_V = 5.0       # the 2S pack is 6-8.4 V; the ADC reads about 0 (or fails) with the battery switch off
POLL_S = 2.0


def is_on(voltage) -> bool:
    return voltage is not None and voltage >= ON_V


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--center-now", action="store_true")
    args = ap.parse_args()
    if args.center_now:
        return center_gimbal()
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
                print(f"battery switched on ({voltage} V): gimbal straight ahead", flush=True)
                center_gimbal()
            elif state is not None:
                print("battery switched off", flush=True)
            state = pending
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
