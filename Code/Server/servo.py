import json
import time
from pathlib import Path

from pca9685 import PCA9685

HOME_PATH = Path(__file__).resolve().parent / "gimbal_home.json"


def load_gimbal_home() -> tuple[int, int]:
    """Return saved pan/tilt for straight-ahead. Defaults to 90/90."""
    pan, tilt = 90, 90
    try:
        data = json.loads(HOME_PATH.read_text(encoding="utf-8"))
        pan = int(data.get("pan", pan))
        tilt = int(data.get("tilt", tilt))
    except Exception:
        pass
    return max(0, min(180, pan)), max(0, min(180, tilt))


def load_right_90_pan(home_pan: int = 76) -> int:
    """Servo angle that points 90 degrees to the car's right. Defaults to a 1:1 map."""
    try:
        data = json.loads(HOME_PATH.read_text(encoding="utf-8"))
        return max(0, min(180, int(data.get("right_90_pan", home_pan + 90))))
    except Exception:
        return max(0, min(180, home_pan + 90))


def load_left_90_pan() -> int | None:
    """Servo angle that points 90 degrees to the car's left, if it has been measured."""
    try:
        data = json.loads(HOME_PATH.read_text(encoding="utf-8"))
        if "left_90_pan" not in data:
            return None
        return max(0, min(180, int(data["left_90_pan"])))
    except Exception:
        return None


def servo_pan_for_bearing(home_pan: int, right_90_pan: int, pan_deg: float,
                          left_90_pan: int | None = None) -> int:
    """Servo angle for a heading relative to straight ahead. Positive pan_deg is left."""
    if pan_deg >= 0 and left_90_pan is not None:
        span = home_pan - left_90_pan
    else:
        span = right_90_pan - home_pan
    if span == 0:
        span = 90
    angle = home_pan - float(pan_deg) * (span / 90.0)
    return max(0, min(180, int(round(angle))))


def save_gimbal_home(pan: int, tilt: int) -> Path:
    pan, tilt = max(0, min(180, int(pan))), max(0, min(180, int(tilt)))
    try:
        payload = json.loads(HOME_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload["pan"] = pan
    payload["tilt"] = tilt
    payload.setdefault(
        "notes",
        "pan/tilt is straight ahead. right_90_pan looks 90 degrees to the car's right.",
    )
    HOME_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return HOME_PATH


class Servo:
    def __init__(self):
        self.pwm_frequency = 50
        self.pwm_channel_map = {
            '0': 8,
            '1': 9,
            '2': 10,
            '3': 11,
            '4': 12,
            '5': 13,
            '6': 14,
            '7': 15
        }
        self.pwm_servo = PCA9685(0x40, debug=True)
        self.pwm_servo.set_pwm_freq(self.pwm_frequency)
        # Do not park pan/tilt at 1500us first. That is ~80° with this driver's
        # angle formula, so every Servo() used to jump off the saved heading
        # before returning to gimbal_home.json.
        self.apply_gimbal_home()

    def apply_gimbal_home(self) -> tuple[int, int]:
        pan, tilt = load_gimbal_home()
        self.set_servo_pwm("0", pan)
        self.set_servo_pwm("1", tilt)
        return pan, tilt

    def set_servo_pwm(self, channel: str, angle: int, error: int = 10) -> None:
        angle = int(angle)
        if channel not in self.pwm_channel_map:
            raise ValueError(f"Invalid channel: {channel}. Valid channels are {list(self.pwm_channel_map.keys())}.")
        pulse = 2500 - int((angle + error) / 0.09) if channel == '0' else 500 + int((angle + error) / 0.09)
        self.pwm_servo.set_servo_pulse(self.pwm_channel_map[channel], pulse)

# Main program logic follows:
if __name__ == '__main__':
    print("Please keep the program running when installing the servos.")
    print("After that, you can press ctrl-C to end the program.")
    pwm_servo = Servo()
    pan, tilt = pwm_servo.apply_gimbal_home()
    print(f"Now servos will hold straight-ahead pan={pan} tilt={tilt}.")
    try:
        while True:
            pwm_servo.set_servo_pwm('0', pan)
            pwm_servo.set_servo_pwm('1', tilt)
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nEnd of program")