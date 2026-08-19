#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

from adc import ADC
from buzzer import Buzzer
from infrared import Infrared
from motor import Ordinary_Car
from servo import Servo
from ultrasonic import Ultrasonic


def estimate_battery_percent(voltage):
    voltage_table = [
        (8.40, 100),
        (8.20, 90),
        (8.00, 80),
        (7.80, 70),
        (7.60, 60),
        (7.40, 50),
        (7.20, 35),
        (7.00, 20),
        (6.80, 10),
        (6.60, 5),
        (6.40, 0),
    ]
    if voltage >= voltage_table[0][0]:
        return 100
    if voltage <= voltage_table[-1][0]:
        return 0
    for (high_voltage, high_percent), (low_voltage, low_percent) in zip(voltage_table, voltage_table[1:]):
        if low_voltage <= voltage <= high_voltage:
            ratio = (voltage - low_voltage) / (high_voltage - low_voltage)
            return round(low_percent + ratio * (high_percent - low_percent))
    return 0


class EnduranceTester:
    def __init__(self, args):
        self.args = args
        self.adc = None
        self.ultrasonic = None
        self.infrared = None
        self.servo = None
        self.motor = None
        self.buzzer = None
        self.obstacle_sensors = []
        self.start_time = None
        self.last_camera_time = 0
        self.last_audio_time = 0
        self.last_servo_time = 0
        self.last_motor_time = 0
        self.motor_step = 0
        self.servo_step = 0

    def confirm(self, message):
        if self.args.yes:
            return True
        answer = input(f"{message} [y/N]: ").strip().lower()
        return answer in ("y", "yes")

    def setup(self):
        self.adc = ADC()
        self.ultrasonic = Ultrasonic()
        self.infrared = Infrared()
        self.servo = Servo()
        self.buzzer = Buzzer()
        if self.args.enable_motor_load:
            if not self.confirm("Motor load will spin the wheels. Lift the car securely before continuing. Continue?"):
                self.args.enable_motor_load = False
            else:
                self.motor = Ordinary_Car()
        self.setup_obstacle_sensors()
        self.prepare_audio_tone()

    def setup_obstacle_sensors(self):
        from gpiozero import DigitalInputDevice

        pins = [int(value.strip()) for value in self.args.obstacle_pins.split(",") if value.strip()]
        for pin in pins:
            self.obstacle_sensors.append((pin, DigitalInputDevice(pin, pull_up=False)))

    def prepare_audio_tone(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        tone_path = self.args.output_dir / "speaker_tone.wav"
        amplitude = 10000
        frame_count = int(self.args.audio_rate * self.args.audio_tone_duration)
        with wave.open(str(tone_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.args.audio_rate)
            frames = bytearray()
            for sample_index in range(frame_count):
                sample = int(amplitude * math.sin(2 * math.pi * self.args.speaker_frequency * sample_index / self.args.audio_rate))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
            wav_file.writeframes(frames)
        self.tone_path = tone_path

    def read_battery(self):
        voltage = self.adc.read_adc(2) * (3 if self.adc.pcb_version == 1 else 2)
        return round(voltage, 2), estimate_battery_percent(voltage)

    def read_sensors(self):
        ultrasonic_cm = self.ultrasonic.get_distance()
        line_left = self.infrared.read_one_infrared(1)
        line_middle = self.infrared.read_one_infrared(2)
        line_right = self.infrared.read_one_infrared(3)
        obstacle_values = []
        for pin, sensor in self.obstacle_sensors:
            raw_value = int(sensor.value)
            obstacle = raw_value if self.args.obstacle_active_high else int(not raw_value)
            obstacle_values.append(f"GPIO{pin}:{obstacle}")
        return ultrasonic_cm, line_left, line_middle, line_right, " ".join(obstacle_values)

    def run_servo_activity(self, now):
        if now - self.last_servo_time < self.args.servo_interval:
            return
        self.last_servo_time = now
        sequence = [
            ("0", 60),
            ("0", 90),
            ("0", 120),
            ("0", 90),
            ("1", 90),
            ("1", 130),
            ("1", 90),
        ]
        channel, angle = sequence[self.servo_step % len(sequence)]
        self.servo.set_servo_pwm(channel, angle)
        self.servo_step += 1

    def run_motor_activity(self, now):
        if not self.args.enable_motor_load or self.motor is None:
            return
        if now - self.last_motor_time < self.args.motor_interval:
            return
        self.last_motor_time = now
        speed = self.args.motor_speed
        sequence = [
            (speed, speed, speed, speed),
            (0, 0, 0, 0),
            (-speed, -speed, -speed, -speed),
            (0, 0, 0, 0),
        ]
        self.motor.set_motor_model(*sequence[self.motor_step % len(sequence)])
        self.motor_step += 1

    def run_buzzer_activity(self):
        self.buzzer.set_state(True)
        time.sleep(0.05)
        self.buzzer.set_state(False)

    def run_audio_activity(self, now):
        if now - self.last_audio_time < self.args.audio_interval:
            return
        self.last_audio_time = now
        subprocess.run(
            ["aplay", "-D", self.args.speaker_device, str(self.tone_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        record_path = self.args.output_dir / "microphone_latest.wav"
        subprocess.run(
            [
                "arecord",
                "-D",
                self.args.mic_device,
                "-f",
                "S16_LE",
                "-r",
                str(self.args.audio_rate),
                "-c",
                "1",
                "-d",
                str(self.args.audio_record_duration),
                str(record_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def capture_picameras(self):
        from picamera2 import Picamera2

        for camera_index in range(self.args.picamera_count):
            camera = None
            try:
                output_path = self.args.output_dir / f"picamera_{camera_index}_latest.jpg"
                camera = Picamera2(camera_num=camera_index)
                from camera_devices import create_csi_still_configuration

                config = create_csi_still_configuration(
                    camera, self.args.camera_width, self.args.camera_height
                )
                camera.configure(config)
                camera.start()
                time.sleep(self.args.camera_warmup)
                camera.capture_file(str(output_path))
            except Exception as exc:
                print(f"Picamera {camera_index} capture failed: {exc}")
            finally:
                if camera is not None:
                    camera.close()

    def capture_usb_cameras(self):
        import cv2

        indexes = [int(value.strip()) for value in self.args.usb_camera_indexes.split(",") if value.strip()]
        for camera_number, device_index in enumerate(indexes):
            output_path = self.args.output_dir / f"usb_camera_{camera_number}_latest.jpg"
            capture = cv2.VideoCapture(device_index)
            try:
                if not capture.isOpened():
                    print(f"USB camera /dev/video{device_index} open failed")
                    continue
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
                time.sleep(self.args.camera_warmup)
                ok, frame = capture.read()
                if ok and frame is not None:
                    cv2.imwrite(str(output_path), frame)
                else:
                    print(f"USB camera /dev/video{device_index} frame read failed")
            finally:
                capture.release()

    def run_camera_activity(self, now):
        if now - self.last_camera_time < self.args.camera_interval:
            return
        self.last_camera_time = now
        self.capture_picameras()
        self.capture_usb_cameras()

    def print_progress(self, elapsed_seconds, voltage, percent):
        bar_width = 30
        filled = max(0, min(bar_width, round(bar_width * percent / 100)))
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[{bar}] battery~{percent:3d}% {voltage:.2f}V | "
            f"elapsed={elapsed_seconds / 60:.1f} min | cutoff={self.args.cutoff_voltage:.2f}V"
        )

    def run(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.args.output_dir / f"endurance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.start_time = time.time()
        print(f"Endurance log: {log_path}")
        print("LED and buzzer are skipped by default. This test exercises sensors, cameras, audio, servos, and optional motor load.")
        with open(log_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                "elapsed_seconds",
                "battery_voltage",
                "battery_percent_est",
                "ultrasonic_cm",
                "line_left",
                "line_middle",
                "line_right",
                "obstacles",
            ])
            low_voltage_count = 0
            while True:
                now = time.time()
                elapsed = now - self.start_time
                voltage, percent = self.read_battery()
                ultrasonic_cm, line_left, line_middle, line_right, obstacles = self.read_sensors()
                writer.writerow([round(elapsed, 1), voltage, percent, ultrasonic_cm, line_left, line_middle, line_right, obstacles])
                log_file.flush()
                self.print_progress(elapsed, voltage, percent)
                if voltage <= self.args.cutoff_voltage:
                    low_voltage_count += 1
                    print(
                        f"Low voltage sample {low_voltage_count}/{self.args.cutoff_samples}: "
                        f"{voltage:.2f}V <= {self.args.cutoff_voltage:.2f}V"
                    )
                    if low_voltage_count >= self.args.cutoff_samples:
                        print(
                            f"Battery voltage stayed below cutoff for {self.args.cutoff_samples} consecutive samples. "
                            "Stopping endurance test."
                        )
                        break
                else:
                    low_voltage_count = 0
                if self.args.max_minutes and elapsed >= self.args.max_minutes * 60:
                    print(f"Max runtime reached: {self.args.max_minutes} minutes")
                    break
                self.run_servo_activity(now)
                self.run_motor_activity(now)
                self.run_audio_activity(now)
                self.run_camera_activity(now)
                time.sleep(self.args.sample_interval)

    def cleanup(self):
        print("Cleaning up...")
        try:
            if self.motor is not None:
                self.motor.set_motor_model(0, 0, 0, 0)
                self.motor.close()
        except Exception as exc:
            print(f"Motor cleanup failed: {exc}")
        try:
            if self.buzzer is not None:
                self.buzzer.set_state(False)
                self.buzzer.close()
        except Exception as exc:
            print(f"Buzzer cleanup failed: {exc}")
        try:
            if self.servo is not None:
                self.servo.pwm_servo.close()
        except Exception as exc:
            print(f"Servo cleanup failed: {exc}")
        try:
            if self.ultrasonic is not None:
                self.ultrasonic.close()
        except Exception as exc:
            print(f"Ultrasonic cleanup failed: {exc}")
        try:
            if self.infrared is not None:
                self.infrared.close()
        except Exception as exc:
            print(f"Infrared cleanup failed: {exc}")
        try:
            if self.adc is not None:
                self.adc.close_i2c()
        except Exception as exc:
            print(f"ADC cleanup failed: {exc}")
        for _, sensor in self.obstacle_sensors:
            sensor.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Run a whole-car endurance test and log battery drain.")
    parser.add_argument("--output-dir", type=Path, default=Path("endurance_logs"), help="Directory for logs and latest media captures.")
    parser.add_argument("--cutoff-voltage", type=float, default=6.8, help="Stop when battery voltage drops to this value.")
    parser.add_argument("--cutoff-samples", type=int, default=3, help="Stop after this many consecutive samples at or below cutoff voltage.")
    parser.add_argument("--max-minutes", type=float, help="Optional maximum runtime in minutes.")
    parser.add_argument("--sample-interval", type=float, default=5.0, help="Seconds between battery/sensor log samples.")
    parser.add_argument("--obstacle-pins", default="26,20,19,16,6,12", help="Comma-separated BCM GPIO pins for obstacle sensors.")
    parser.set_defaults(obstacle_active_high=False)
    parser.add_argument("--obstacle-active-high", action="store_true", help="Treat HIGH as obstacle detected.")
    parser.add_argument("--obstacle-active-low", action="store_false", dest="obstacle_active_high", help="Treat LOW as obstacle detected.")
    parser.add_argument("--picamera-count", type=int, default=2, help="Number of Picamera devices to capture.")
    parser.add_argument("--usb-camera-indexes", default="0,2,37", help="Comma-separated USB /dev/video indexes.")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--camera-warmup", type=float, default=0.5, help="Seconds to wait after camera start.")
    parser.add_argument("--camera-interval", type=float, default=60.0, help="Seconds between all-camera capture cycles.")
    parser.add_argument("--speaker-device", default="plughw:2,0", help="ALSA playback device.")
    parser.add_argument("--mic-device", default="plughw:2,0", help="ALSA capture device.")
    parser.add_argument("--audio-rate", type=int, default=44100, help="Audio sample rate.")
    parser.add_argument("--speaker-frequency", type=int, default=880, help="Speaker tone frequency.")
    parser.add_argument("--audio-tone-duration", type=float, default=0.4, help="Speaker tone duration in seconds.")
    parser.add_argument("--audio-record-duration", type=int, default=1, help="Microphone record duration in seconds.")
    parser.add_argument("--audio-interval", type=float, default=60.0, help="Seconds between audio test cycles.")
    parser.add_argument("--servo-interval", type=float, default=10.0, help="Seconds between servo movement steps.")
    parser.add_argument("--enable-motor-load", action="store_true", help="Enable wheel motor load. Lift the car before using this.")
    parser.add_argument("--motor-speed", type=int, default=700, help="Motor load speed.")
    parser.add_argument("--motor-interval", type=float, default=5.0, help="Seconds between motor load steps.")
    parser.add_argument("--yes", action="store_true", help="Skip safety confirmation for motor load.")
    return parser


def main():
    args = build_parser().parse_args()
    tester = EnduranceTester(args)
    try:
        tester.setup()
        tester.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
