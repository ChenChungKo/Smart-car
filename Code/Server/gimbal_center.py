#!/usr/bin/env python3
"""Point the gimbal (ultrasonic + gimbal camera) straight ahead and leave it there.

Run at every boot by systemd/gimbal-center.service, so the ultrasonic faces forward before any
program starts. Straight ahead is gimbal_home.json "pan"/"tilt" (set with setup_gimbal_servo.py).
The pulses are then stopped: a held SG90 keeps correcting and buzzes, and its gears hold the pose.

    python3 gimbal_center.py
"""

from __future__ import annotations

import sys
import time


def main() -> int:
    for attempt in range(15):  # the I2C bus can come up a little after this service starts
        try:
            from servo import Servo

            servo = Servo()
            pan, tilt = servo.apply_gimbal_home()
            time.sleep(0.5)  # let it get there, then stop the pulses so it does not buzz
            servo.release_gimbal()
            print(f"gimbal straight ahead: pan={pan} tilt={tilt} (servo released)", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001 - any start-up failure is worth another try
            print(f"gimbal not ready ({exc}); retrying ({attempt + 1}/15)", flush=True)
            time.sleep(1.0)
    return 1


if __name__ == "__main__":
    sys.exit(main())
