#!/usr/bin/env python3
"""Drive using ultrasonic + extra IR obstacle sensors only.

Does not use light-following or the 3 bottom line sensors.
Ctrl+C stops motors immediately.

Servo 0 pans left/right; servo 1 nods up/down.
Default 6 IR pins match full_car_test.py (LOW = obstacle).
Edit --pin-* if your physical layout differs.
"""

from __future__ import annotations

import argparse
import time

from gpiozero import DigitalInputDevice
from motor import Ordinary_Car
from servo import Servo
from ultrasonic import Ultrasonic

# BCM GPIO. Names are for the cruise logic; remap with CLI if wiring differs.
DEFAULT_PINS = {
    "front_left": 26,
    "front": 20,
    "front_right": 19,
    "left": 16,
    "right": 6,
    "rear": 12,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous cruise: ultrasonic + 6 IR obstacle sensors.")
    parser.add_argument("--speed", type=int, default=700, help="Forward PWM (about 600-1200).")
    parser.add_argument("--turn-speed", type=int, default=700, help="Turn PWM (lower = smaller yaw).")
    parser.add_argument("--back-speed", type=int, default=800, help="Reverse PWM.")
    parser.add_argument("--stop-cm", type=float, default=30.0, help="Ultrasonic stop/turn distance.")
    parser.add_argument("--slow-cm", type=float, default=50.0, help="Ultrasonic slow-down distance.")
    parser.add_argument("--scan-angles", default="30,90,150", help="Servo 0 pan angles, left/mid/right.")
    parser.add_argument("--tilt-angles", default="80,90,110", help="Servo 1 tilt angles, up/level/down.")
    parser.add_argument("--scan-settle-s", type=float, default=0.5, help="Wait after each servo move before ranging.")
    parser.add_argument("--avoid-hold-s", type=float, default=0.22, help="How long to keep a turn/reverse.")
    parser.add_argument("--stale-s", type=float, default=1.2, help="Ignore L/M/R readings older than this.")
    parser.add_argument("--loop-s", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true", help="Sweep and print sensors; do not move motors.")
    parser.set_defaults(active_high=False)
    parser.add_argument("--obstacle-active-high", action="store_true", dest="active_high")
    parser.add_argument("--obstacle-active-low", action="store_false", dest="active_high")
    for name, pin in DEFAULT_PINS.items():
        parser.add_argument(f"--pin-{name.replace('_', '-')}", type=int, default=pin)
    return parser.parse_args()


class ObstacleCruise:
    def __init__(self, args):
        self.args = args
        self.motor = None if args.dry_run else Ordinary_Car()
        self.sonic = Ultrasonic()
        self.servo = Servo()
        self.scan_angles = [int(v.strip()) for v in args.scan_angles.split(",") if v.strip()]
        self.tilt_angles = [int(v.strip()) for v in args.tilt_angles.split(",") if v.strip()]
        if len(self.scan_angles) != 3:
            raise ValueError("--scan-angles needs three values: left,mid,right")
        if len(self.tilt_angles) != 3:
            raise ValueError("--tilt-angles needs three values: up,level,down")
        self.scan_index = 1  # start at mid pan
        self.scan_dir = 1
        self.tilt_index = 1  # start at level
        self.tilt_dir = 1
        self.sonic_cm = [None, None, None]  # left, mid, right (min of non-floor tilts)
        self.sonic_age = [0.0, 0.0, 0.0]
        self.grid_cm = [[None, None, None] for _ in range(3)]  # pan x tilt
        self.down_tilt = max(range(3), key=lambda i: self.tilt_angles[i])
        self.last_pan_angle = self.scan_angles[1]
        self.last_tilt_angle = self.tilt_angles[1]
        self.hold_until = 0.0
        self.hold_pwm = (0, 0, 0, 0)
        self.hold_action = "forward"
        self.sensors = {}
        pin_map = {
            "front_left": args.pin_front_left,
            "front": args.pin_front,
            "front_right": args.pin_front_right,
            "left": args.pin_left,
            "right": args.pin_right,
            "rear": args.pin_rear,
        }
        for name, pin in pin_map.items():
            self.sensors[name] = (pin, DigitalInputDevice(pin, pull_up=False))
        self.servo.set_servo_pwm("0", self.scan_angles[1])
        self.servo.set_servo_pwm("1", self.tilt_angles[1])
        time.sleep(0.15)

    def read_ir(self) -> dict[str, int]:
        hits = {}
        for name, (_pin, device) in self.sensors.items():
            raw = int(device.value)
            hits[name] = raw if self.args.active_high else int(not raw)
        return hits

    def drive(self, fl, bl, fr, br):
        if self.motor is None:
            return
        self.motor.set_motor_model(fl, bl, fr, br)

    def stop(self):
        self.drive(0, 0, 0, 0)

    def _valid(self, cm):
        return cm is not None and 1.0 <= cm < 250.0

    def _close(self, cm):
        return self._valid(cm) and cm < self.args.stop_cm

    def _slow(self, cm):
        return self._valid(cm) and cm < self.args.slow_cm

    def _ping(self, prev):
        samples = []
        for _ in range(3):
            cm = self.sonic.get_distance()
            if self._valid(cm):
                samples.append(cm)
            time.sleep(0.02)
        if not samples:
            return None
        samples.sort()
        cm = samples[len(samples) // 2]
        # Echo often jumps from ~10cm to ~200cm when the beam misses; keep the near reading.
        if self._valid(prev) and prev < 40 and cm > 120:
            return prev
        return cm

    def _expire_stale(self):
        now = time.time()
        for i in range(3):
            if self.sonic_cm[i] is not None and now - self.sonic_age[i] > self.args.stale_s:
                self.sonic_cm[i] = None

    def _step_triangle(self, index, direction, last):
        if index <= 0:
            direction = 1
        elif index >= last:
            direction = -1
        return index + direction, direction

    def _refresh_pan_cm(self, pan):
        vals = []
        for tilt, cm in enumerate(self.grid_cm[pan]):
            if not self._valid(cm):
                continue
            if tilt == self.down_tilt:
                continue
            vals.append(cm)
        if not vals:
            down = self.grid_cm[pan][self.down_tilt]
            self.sonic_cm[pan] = down if self._valid(down) else None
        else:
            self.sonic_cm[pan] = min(vals)
        if self.sonic_cm[pan] is not None:
            self.sonic_age[pan] = time.time()

    def scan_once(self):
        pan = self.scan_index
        tilt = self.tilt_index
        self.servo.set_servo_pwm("0", self.scan_angles[pan])
        self.servo.set_servo_pwm("1", self.tilt_angles[tilt])
        time.sleep(self.args.scan_settle_s)
        prev = self.grid_cm[pan][tilt]
        self.grid_cm[pan][tilt] = self._ping(prev)
        self.last_pan_angle = self.scan_angles[pan]
        self.last_tilt_angle = self.tilt_angles[tilt]
        self._refresh_pan_cm(pan)
        self._expire_stale()
        self.scan_index, self.scan_dir = self._step_triangle(self.scan_index, self.scan_dir, 2)
        self.tilt_index, self.tilt_dir = self._step_triangle(self.tilt_index, self.tilt_dir, 2)

    def _pwm_turn_right(self, mag=None):
        mag = self.args.turn_speed if mag is None else mag
        inner = -max(150, mag // 4)
        return (mag, mag, inner, inner)

    def _pwm_turn_left(self, mag=None):
        mag = self.args.turn_speed if mag is None else mag
        inner = -max(150, mag // 4)
        return (inner, inner, mag, mag)

    def _turn_toward_open(self, left_cm, right_cm):
        left_ok = self._valid(left_cm)
        right_ok = self._valid(right_cm)
        if left_ok and right_ok:
            if left_cm >= right_cm:
                return self._pwm_turn_left(), "turn_left"
            return self._pwm_turn_right(), "turn_right"
        if left_ok and not right_ok:
            return self._pwm_turn_left(), "turn_left"
        if right_ok and not left_ok:
            return self._pwm_turn_right(), "turn_right"
        return self._pwm_turn_right(), "turn_right"

    def decide(self, ir):
        speed = self.args.speed
        back = self.args.back_speed
        left_cm, mid_cm, right_cm = self.sonic_cm
        sonic_left = self._close(left_cm)
        sonic_mid = self._close(mid_cm)
        sonic_right = self._close(right_cm)
        front_ir = bool(ir["front"] or ir["front_left"] or ir["front_right"])
        left_ir = bool(ir["left"] or ir["front_left"])
        right_ir = bool(ir["right"] or ir["front_right"])
        need_avoid = sonic_mid or sonic_left or sonic_right or front_ir
        slow = self._slow(mid_cm) or self._slow(left_cm) or self._slow(right_cm)

        if need_avoid:
            # Rear IR = do not reverse. Nudge toward the more open side instead of freezing.
            if ir["rear"]:
                pwm, action = self._turn_toward_open(left_cm, right_cm)
                return pwm, action + "_rear"
            if sonic_left and not sonic_right:
                return self._pwm_turn_right(), "turn_right"
            if sonic_right and not sonic_left:
                return self._pwm_turn_left(), "turn_left"
            if sonic_mid or front_ir:
                return (-back, -back, -back, -back), "reverse"
            return (-back, -back, -back, -back), "reverse"
        if left_ir and not right_ir:
            return self._pwm_turn_right(), "turn_right"
        if right_ir and not left_ir:
            return self._pwm_turn_left(), "turn_left"
        if slow:
            crawl = max(400, speed // 2)
            return (crawl, crawl, crawl, crawl), "slow"
        return (speed, speed, speed, speed), "forward"

    def apply_hold(self, pwm, action):
        now = time.time()
        sticky = action.startswith("reverse") or action.startswith("turn_")
        if sticky:
            self.hold_until = now + self.args.avoid_hold_s
            self.hold_pwm = pwm
            self.hold_action = action
            return pwm, action
        if now < self.hold_until and (
            self.hold_action.startswith("reverse") or self.hold_action.startswith("turn_")
        ):
            return self.hold_pwm, self.hold_action + "(hold)"
        return pwm, action

    def close(self):
        try:
            self.stop()
        except Exception:
            pass
        if self.motor is not None:
            self.motor.close()
        if self.servo is not None:
            try:
                self.servo.set_servo_pwm("0", 90)
                self.servo.set_servo_pwm("1", 90)
            except Exception:
                pass
        if self.sonic is not None:
            self.sonic.close()
        for _pin, device in self.sensors.values():
            device.close()

    def run(self):
        print("Obstacle cruise. Head pans (servo 0) and nods (servo 1). Ctrl+C to stop.")
        print("IR pins:", {k: v[0] for k, v in self.sensors.items()})
        print("Pan angles:", self.scan_angles, "Tilt angles:", self.tilt_angles)
        try:
            while True:
                self.scan_once()
                ir = self.read_ir()
                pwm, action = self.apply_hold(*self.decide(ir))
                labels = {
                    "front_left": "fl",
                    "front": "f",
                    "front_right": "fr",
                    "left": "l",
                    "right": "r",
                    "rear": "re",
                }
                ir_txt = " ".join(f"{labels[n]}={ir[n]}" for n in labels)
                l, m, r = self.sonic_cm
                print(
                    f"pan={self.last_pan_angle} tilt={self.last_tilt_angle}  "
                    f"sonic L/M/R={l}/{m}/{r}cm  {ir_txt}  {action}",
                    flush=True,
                )
                self.drive(*pwm)
                time.sleep(self.args.loop_s)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            self.close()


def main():
    args = parse_args()
    ObstacleCruise(args).run()


if __name__ == "__main__":
    main()
