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


def save_gimbal_home(pan: int, tilt: int) -> Path:
    pan, tilt = max(0, min(180, int(pan))), max(0, min(180, int(tilt)))
    payload = {
        "pan": pan,
        "tilt": tilt,
        "notes": "Straight-ahead pose used at cruise start (servo 0 pan, servo 1 tilt).",
    }
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