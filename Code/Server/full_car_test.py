#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

from adc import ADC
from buzzer import Buzzer
from camera import Camera
from infrared import Infrared
from led import Led
from motor import Ordinary_Car
from servo import Servo
from ultrasonic import Ultrasonic


class FullCarTester:
    def __init__(self, args):
        self.args = args
        self.motor = None
        self.servo = None
        self.sonic = None
        self.infrared = None
        self.adc = None
        self.led = None
        self.buzzer = None
        self.camera = None

    def get_motor(self):
        if self.motor is None:
            self.motor = Ordinary_Car()
        return self.motor

    def get_servo(self):
        if self.servo is None:
            self.servo = Servo()
        return self.servo

    def get_sonic(self):
        if self.sonic is None:
            self.sonic = Ultrasonic()
        return self.sonic

    def get_infrared(self):
        if self.infrared is None:
            self.infrared = Infrared()
        return self.infrared

    def get_adc(self):
        if self.adc is None:
            self.adc = ADC()
        return self.adc

    def get_led(self):
        if self.led is None:
            self.led = Led()
        return self.led

    def get_buzzer(self):
        if self.buzzer is None:
            self.buzzer = Buzzer()
        return self.buzzer

    def get_camera(self):
        if self.camera is None:
            self.camera = Camera(stream_size=(400, 300))
        return self.camera

    def confirm(self, message):
        if self.args.yes:
            return True
        answer = input(f"{message} [y/N]: ").strip().lower()
        return answer in ("y", "yes")

    def countdown(self, seconds=3):
        for value in range(seconds, 0, -1):
            print(f"Starting in {value}...")
            time.sleep(1)

    def stop_motion(self):
        if self.motor is not None:
            self.motor.set_motor_model(0, 0, 0, 0)

    def drive_step(self, label, duty, duration=None):
        if duration is None:
            duration = self.args.duration
        print(f"{label}: duty={duty}, duration={duration:.2f}s, gear_ratio=1:{self.args.gear_ratio:g}")
        try:
            self.get_motor().set_motor_model(*duty)
            time.sleep(duration)
        finally:
            self.stop_motion()
            time.sleep(self.args.pause)

    def test_basic_drive(self):
        if not self.confirm("This test will move the car forward/back/turn. Lift the car or clear the area first. Continue?"):
            return
        self.countdown()
        speed = self.args.speed
        turn_speed = self.args.turn_speed
        steps = [
            ("Forward", (speed, speed, speed, speed), self.args.duration),
            ("Backward", (-speed, -speed, -speed, -speed), self.args.duration),
            ("Rotate left 90 degrees", (-turn_speed, -turn_speed, turn_speed, turn_speed), self.args.left_turn_90_duration),
            ("Rotate right 90 degrees back", (turn_speed, turn_speed, -turn_speed, -turn_speed), self.args.right_turn_90_duration),
        ]
        for label, duty, duration in steps:
            self.drive_step(label, duty, duration)

    def test_mecanum_drive(self):
        if not self.confirm("This test will move in mecanum directions. Lift the car or clear the area first. Continue?"):
            return
        self.countdown()
        speed = self.args.speed
        steps = [
            ("Mecanum forward", (speed, speed, speed, speed)),
            ("Mecanum backward", (-speed, -speed, -speed, -speed)),
            ("Strafe left", (-speed, speed, speed, -speed)),
            ("Strafe right", (speed, -speed, -speed, speed)),
            ("Rotate left", (-speed, -speed, speed, speed)),
            ("Rotate right", (speed, speed, -speed, -speed)),
            ("Diagonal front-left", (0, speed, speed, 0)),
            ("Diagonal front-right", (speed, 0, 0, speed)),
        ]
        for label, duty in steps:
            self.drive_step(label, duty)

    def test_servo(self):
        if not self.confirm("This test will sweep servo channels 0 and 1. Continue?"):
            return
        servo = self.get_servo()
        channel_points = {
            "0": [self.args.servo0_min, 90, self.args.servo0_max, 90],
            "1": [self.args.servo1_min, 90, self.args.servo1_max, 90],
        }
        for channel, angles in channel_points.items():
            print(f"Testing servo channel {channel}")
            for angle in angles:
                print(f"  channel {channel} -> {angle} degrees")
                servo.set_servo_pwm(channel, angle)
                time.sleep(0.8)

    def test_ultrasonic(self):
        sonic = self.get_sonic()
        print("Reading ultrasonic distance...")
        for index in range(self.args.samples):
            distance = sonic.get_distance()
            print(f"  sample {index + 1}: {distance} cm")
            time.sleep(0.4)

    def test_infrared(self):
        infrared = self.get_infrared()
        print("Reading infrared line sensors...")
        for index in range(self.args.samples):
            left = infrared.read_one_infrared(1)
            middle = infrared.read_one_infrared(2)
            right = infrared.read_one_infrared(3)
            combined = infrared.read_all_infrared()
            print(f"  sample {index + 1}: left={left}, middle={middle}, right={right}, combined={combined}")
            time.sleep(0.3)

    def test_obstacle_sensors(self):
        from gpiozero import DigitalInputDevice

        if not self.args.obstacle_pins:
            print("No obstacle sensor GPIO pins configured.")
            print("Use BCM GPIO numbers, for example: --obstacle-pins 5,6,12,13,19,26")
            return
        pins = [int(value.strip()) for value in self.args.obstacle_pins.split(",") if value.strip()]
        sensors = []
        try:
            for pin in pins:
                sensors.append((pin, DigitalInputDevice(pin, pull_up=False)))
            active_mode = "HIGH" if self.args.obstacle_active_high else "LOW"
            print(f"Reading {len(sensors)} obstacle sensor(s). Obstacle active signal: {active_mode}")
            for sample_index in range(self.args.samples):
                values = []
                for pin, sensor in sensors:
                    raw_value = int(sensor.value)
                    obstacle = raw_value if self.args.obstacle_active_high else int(not raw_value)
                    values.append(f"GPIO{pin}: obstacle={obstacle}")
                print(f"  sample {sample_index + 1}: " + " | ".join(values))
                time.sleep(0.3)
        finally:
            for _, sensor in sensors:
                sensor.close()

    def estimate_battery_percent(self, voltage):
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

    def test_adc(self):
        adc = self.get_adc()
        print("Reading ADC channels...")
        for index in range(self.args.samples):
            left_light = adc.read_adc(0)
            right_light = adc.read_adc(1)
            battery = adc.read_adc(2) * (3 if adc.pcb_version == 1 else 2)
            battery_percent = self.estimate_battery_percent(battery)
            print(
                f"  sample {index + 1}: left_light={left_light}V, "
                f"right_light={right_light}V, battery={battery:.2f}V, battery_percent~={battery_percent}%"
            )
            time.sleep(0.5)

    def test_buzzer(self):
        if not self.confirm("This test will beep the buzzer. Continue?"):
            return
        buzzer = self.get_buzzer()
        for index in range(3):
            print(f"Beep {index + 1}")
            buzzer.set_state(True)
            time.sleep(0.15)
            buzzer.set_state(False)
            time.sleep(0.2)

    def run_audio_command(self, command, label):
        import subprocess

        print(f"\n{label}: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            return True
        except FileNotFoundError:
            print(f"{label} failed: command not found: {command[0]}")
        except subprocess.CalledProcessError as exc:
            print(f"{label} failed with exit code {exc.returncode}")
        return False

    def print_audio_devices(self):
        import subprocess

        for label, command in [
            ("Playback devices", ["aplay", "-l"]),
            ("Capture devices", ["arecord", "-l"]),
        ]:
            print(f"\n=== {label} ===")
            try:
                subprocess.run(command, check=False)
            except FileNotFoundError:
                print(f"{command[0]} is not installed.")

    def create_speaker_test_wav(self, output_path):
        import math
        import wave

        amplitude = 12000
        frequency = self.args.speaker_frequency
        frame_count = int(self.args.audio_rate * self.args.audio_duration)
        with wave.open(str(output_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.args.audio_rate)
            frames = bytearray()
            for sample_index in range(frame_count):
                sample = int(amplitude * math.sin(2 * math.pi * frequency * sample_index / self.args.audio_rate))
                frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
            wav_file.writeframes(frames)

    def test_speaker(self):
        if not self.confirm("This test will play a tone through the speaker. Set volume low first. Continue?"):
            return
        audio_dir = Path(self.args.audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_path = audio_dir / "speaker_test.wav"
        self.create_speaker_test_wav(output_path)
        command = ["aplay"]
        if self.args.speaker_device:
            command.extend(["-D", self.args.speaker_device])
        command.append(str(output_path))
        self.run_audio_command(command, "Speaker playback")

    def test_microphone(self):
        if not self.confirm("This test will record audio from the microphone, then play it back. Continue?"):
            return
        audio_dir = Path(self.args.audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_path = audio_dir / "microphone_test.wav"
        record_command = [
            "arecord",
            "-f",
            "S16_LE",
            "-r",
            str(self.args.audio_rate),
            "-c",
            "1",
            "-d",
            str(self.args.audio_duration),
        ]
        if self.args.mic_device:
            record_command.extend(["-D", self.args.mic_device])
        record_command.append(str(output_path))
        if not self.run_audio_command(record_command, "Microphone recording"):
            return
        if self.args.no_audio_playback:
            print(f"Microphone recording saved to {output_path}")
            return
        playback_command = ["aplay"]
        if self.args.speaker_device:
            playback_command.extend(["-D", self.args.speaker_device])
        playback_command.append(str(output_path))
        self.run_audio_command(playback_command, "Microphone playback")

    def test_audio(self):
        self.print_audio_devices()
        self.test_speaker()
        self.test_microphone()

    def test_led(self):
        led = self.get_led()
        if not getattr(led, "is_support_led_function", False):
            print("LED function is not supported by the configured hardware version.")
            return
        print(
            f"LED config: Connect_Version={getattr(led, 'connect_version', None)}, "
            f"Pi_Version={getattr(led, 'pi_version', None)}, "
            f"driver={type(getattr(led, 'strip', None)).__name__}"
        )
        if hasattr(led.strip, "check_spi_state"):
            print(f"SPI LED init state: {led.strip.check_spi_state()}")
            led.strip.spi_gpio_info()
        elif hasattr(led.strip, "check_rpi_ws281x_state"):
            print(f"RPI WS281x init state: {led.strip.check_rpi_ws281x_state()}")
        print("Testing all LEDs red/green/blue/white...")
        for label, color in [
            ("red", (255, 0, 0)),
            ("green", (0, 255, 0)),
            ("blue", (0, 0, 255)),
            ("white", (255, 255, 255)),
        ]:
            print(f"  all {label}")
            led.strip.set_all_led_color(*color)
            time.sleep(1)
        led.colorBlink(0)
        print("Testing individual LEDs...")
        colors = [
            (0x01, 255, 0, 0),
            (0x02, 255, 125, 0),
            (0x04, 255, 255, 0),
            (0x08, 0, 255, 0),
            (0x10, 0, 255, 255),
            (0x20, 0, 0, 255),
            (0x40, 128, 0, 128),
            (0x80, 255, 255, 255),
        ]
        for index, red, green, blue in colors:
            led.ledIndex(index, red, green, blue)
            time.sleep(0.2)
        time.sleep(1)
        print("Testing LED animations...")
        animation_steps = [
            ("following", led.following, 2.0),
            ("colorBlink", lambda: led.colorBlink(1), 2.0),
            ("rainbowbreathing", led.rainbowbreathing, 2.0),
            ("rainbowCycle", led.rainbowCycle, 2.0),
        ]
        for label, callback, duration in animation_steps:
            print(f"  {label}")
            end_time = time.time() + duration
            while time.time() < end_time:
                callback()
                time.sleep(0.01)
        led.colorBlink(0)

    def test_camera(self):
        output_path = Path(self.args.camera_file)
        print(f"Capturing camera image to {output_path}")
        camera = self.get_camera()
        try:
            if not camera.camera.started:
                camera.camera.start()
                time.sleep(1)
            metadata = camera.save_image(str(output_path))
            print(f"Camera capture complete. metadata={metadata}")
        finally:
            camera.close()
            self.camera = None

    def test_cameras(self):
        self.test_picameras()
        self.test_named_usb_cameras()

    def save_usb_frame_with_exif(self, output_path, frame, position, camera_entry):
        import cv2
        from PIL import Image

        captured_at = time.strftime("%Y:%m:%d %H:%M:%S")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        exif = Image.Exif()
        exif[270] = (
            f"{position} USB camera; "
            f"sensor={camera_entry.get('sensor', 'unknown')}; "
            f"lens={camera_entry.get('lens_type', 'unknown')}"
        )
        exif[271] = "BL"
        exif[272] = camera_entry.get("model", "USB Camera")
        exif[305] = "OpenCV / full_car_test.py"
        exif[306] = captured_at
        exif[36867] = captured_at
        image.save(
            output_path,
            format="JPEG",
            quality=95,
            subsampling=0,
            dpi=(72, 72),
            exif=exif,
        )

    def test_picameras(self):
        from picamera2 import Picamera2

        output_dir = Path(self.args.camera_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        capture_width = self.args.csi_camera_width
        capture_height = self.args.csi_camera_height
        camera_info = Picamera2.global_camera_info()
        camera_count = self.args.picamera_count
        if self.args.camera_count is not None:
            camera_count = self.args.camera_count
        print(f"Detected {len(camera_info)} Picamera device(s): {camera_info}")
        print(f"CSI capture size: {capture_width}x{capture_height}")
        for camera_index in range(camera_count):
            output_path = output_dir / f"picamera_{camera_index}.jpg"
            print(f"\nTesting Picamera {camera_index}, output={output_path}")
            camera = None
            try:
                camera = Picamera2(camera_num=camera_index)
                config = camera.create_still_configuration(
                    main={"size": (capture_width, capture_height)}
                )
                camera.configure(config)
                camera.start()
                time.sleep(self.args.camera_warmup)
                metadata = camera.capture_file(str(output_path))
                print(f"  Picamera {camera_index}: capture OK, metadata={metadata}")
            except Exception as exc:
                print(f"  Picamera {camera_index}: capture failed: {exc}")
            finally:
                if camera is not None:
                    try:
                        camera.close()
                    except Exception as exc:
                        print(f"  Picamera {camera_index}: close failed: {exc}")

    def test_named_usb_cameras(self):
        from camera_devices import resolve_usb_capture_index

        output_dir = Path(self.args.camera_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hardware = self.load_camera_hardware()
        positions = ("left", "right", "rear")

        for index, position in enumerate(positions, start=1):
            # Resolve immediately before opening. A USB camera can reconnect and
            # receive a new /dev/video index while Picamera/libcamera is running.
            entry = hardware[position]
            try:
                device_index, resolved = resolve_usb_capture_index(entry)
                capture_width = self.args.camera_width
                capture_height = self.args.camera_height
                output_path = output_dir / f"usb_{position}_video{device_index}.jpg"
                print(
                    f"\nTesting {position} USB camera "
                    f"/dev/video{device_index} ({resolved}), "
                    f"capture_size={capture_width}x{capture_height}, output={output_path}"
                )
                ok, frame = self.capture_usb_frame(
                    device_index,
                    width=capture_width,
                    height=capture_height,
                    fourcc="YUYV",
                    fps=6,
                )
                if not ok:
                    print(f"  {position}: capture failed")
                else:
                    self.save_usb_frame_with_exif(
                        output_path,
                        frame,
                        position,
                        entry,
                    )
                    print(f"  {position}: capture OK, actual_size={frame.shape[1]}x{frame.shape[0]}")
            except Exception as exc:
                print(f"  {position}: capture failed: {exc}")
            if index < len(positions):
                time.sleep(self.args.camera_settle)

    def capture_csi_still(self, device_index, output_path):
        import numpy as np
        from picamera2 import Picamera2

        camera = None
        try:
            camera = Picamera2(camera_num=device_index)
            config = camera.create_still_configuration(
                main={"size": (self.args.camera_width, self.args.camera_height)}
            )
            camera.configure(config)
            camera.start()
            camera.set_controls({"AeEnable": True, "AwbEnable": True})
            time.sleep(self.args.camera_warmup)
            last_frame = None
            for _ in range(self.args.camera_warmup_frames):
                last_frame = camera.capture_array()
                time.sleep(0.1)
            if last_frame is not None and float(np.mean(last_frame)) < 80:
                camera.set_controls({"ExposureTime": 40000, "AnalogueGain": 4.0})
                for _ in range(5):
                    last_frame = camera.capture_array()
                    time.sleep(0.1)
            camera.capture_file(str(output_path))
            return True
        finally:
            if camera is not None:
                camera.close()

    def capture_usb_frame(self, device_index, width=None, height=None, fourcc=None, fps=None):
        import cv2

        width = self.args.camera_width if width is None else width
        height = self.args.camera_height if height is None else height
        capture = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        try:
            if not capture.isOpened():
                return False, None
            if fourcc:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps is not None:
                capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            time.sleep(self.args.camera_warmup)
            for _ in range(self.args.camera_warmup_frames):
                capture.read()
                time.sleep(0.1)
            ok, frame = capture.read()
            if not ok or frame is None:
                return False, None
            return True, frame
        finally:
            capture.release()

    def test_usb_cameras(self):
        import cv2

        output_dir = Path(self.args.camera_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.args.usb_camera_indexes:
            device_indexes = [int(value.strip()) for value in self.args.usb_camera_indexes.split(",") if value.strip()]
        else:
            device_indexes = list(range(self.args.usb_camera_start, self.args.usb_camera_start + self.args.usb_camera_count))
        for camera_number, device_index in enumerate(device_indexes):
            output_path = output_dir / f"usb_camera_{camera_number}.jpg"
            print(f"\nTesting USB camera /dev/video{device_index}, output={output_path}")
            try:
                ok, frame = self.capture_usb_frame(device_index)
                if not ok:
                    print(f"  USB camera {device_index}: capture failed")
                    continue
                if cv2.imwrite(str(output_path), frame):
                    print(f"  USB camera {device_index}: capture OK")
                else:
                    print(f"  USB camera {device_index}: write failed")
            except Exception as exc:
                print(f"  USB camera {device_index}: capture failed: {exc}")

    def load_camera_hardware(self):
        hardware_path = Path(__file__).with_name("camera_hardware.json")
        with open(hardware_path, encoding="utf-8") as handle:
            return json.load(handle)

    def resolve_labeled_camera_list(self):
        hardware = self.load_camera_hardware()
        order = ["front", "left", "right", "rear"]
        if self.args.labeled_camera:
            order = [self.args.labeled_camera]

        cameras = []
        for position in order:
            entry = hardware[position]
            if entry["interface"] == "csi":
                cameras.append((position, "csi", self.args.csi_camera_num, entry["device"]))
            else:
                cameras.append((position, "usb", None, entry))
        return cameras

    def test_cameras_labeled(self):
        import cv2

        from camera_devices import resolve_usb_capture_index

        output_dir = Path(self.args.labeled_camera_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        labeled_cameras = self.resolve_labeled_camera_list()
        total = len(labeled_cameras)
        print(f"Saving labeled camera images to {output_dir}")
        print("One camera at a time: open -> warmup -> save -> close (never open all together)")
        for index, (position, camera_type, device_index, resolved) in enumerate(labeled_cameras, start=1):
            print(f"\n[{index}/{total}] {position} camera")
            if not self.args.yes:
                answer = input(f"Ready to open ONLY {position}? ENTER=capture, q=skip: ").strip().lower()
                if answer in ("q", "quit", "skip"):
                    print(f"  {position}: skipped")
                    continue
            if camera_type == "csi":
                output_path = output_dir / f"{position}_csi_cam{device_index}.jpg"
                print(f"  CSI camera_num={device_index} ({resolved}) -> {output_path.name}")
                try:
                    if self.capture_csi_still(device_index, output_path):
                        print(f"  {position}: capture OK")
                    else:
                        print(f"  {position}: capture failed")
                except Exception as exc:
                    print(f"  {position}: capture failed: {exc}")
            else:
                # Resolve immediately before capture because /dev/video indexes
                # can change after opening/closing a CSI camera.
                try:
                    device_index, resolved = resolve_usb_capture_index(resolved)
                except Exception as exc:
                    print(f"  {position}: device resolution failed: {exc}")
                    continue
                output_path = output_dir / f"{position}_usb_video{device_index}.jpg"
                print(f"  /dev/video{device_index} ({resolved}) -> {output_path.name}")
                try:
                    ok, frame = self.capture_usb_frame(device_index)
                    if not ok:
                        print(f"  {position}: capture failed")
                    elif cv2.imwrite(str(output_path), frame):
                        print(f"  {position}: capture OK")
                    else:
                        print(f"  {position}: write failed")
                except Exception as exc:
                    print(f"  {position}: capture failed: {exc}")
            if index < total:
                time.sleep(self.args.camera_settle)
        print(f"\nLabeled camera capture complete. Files in {output_dir}")

    def test_plane_map(self):
        self.test_cameras_labeled()
        from plane_map_stitch import stitch_plane_map

        calibration = Path(__file__).with_name("plane_map_calibration.json")
        output_dir = Path(__file__).with_name("plane_map_output")
        input_dir = Path(self.args.labeled_camera_dir)
        stitched_path, gridded_path, preview_path = stitch_plane_map(
            input_dir,
            output_dir,
            calibration,
        )
        print(f"\nPlane map stitched: {stitched_path}")
        print(f"Plane map with grid: {gridded_path}")
        print(f"Plane map preview: {preview_path}")

    def test_sensors(self):
        self.test_ultrasonic()
        self.test_infrared()
        self.test_obstacle_sensors()
        self.test_adc()

    def progress_bar(self, current, total, label):
        width = 30
        filled = int(width * current / total) if total else width
        bar = "#" * filled + "-" * (width - filled)
        percent = int(100 * current / total) if total else 100
        print(f"[{bar}] {percent:3d}% ({current}/{total}) {label}")

    def get_system_test_steps(self):
        steps = [
            ("Battery and light ADC", self.test_adc),
            ("Ultrasonic sensor", self.test_ultrasonic),
            ("Bottom line sensors", self.test_infrared),
            ("Extra obstacle sensors", self.test_obstacle_sensors),
            ("servo", self.test_servo),
            ("buzzer", self.test_buzzer),
            ("audio", self.test_audio),
            ("5 cameras (2 CSI + 3 USB, sequential)", self.test_cameras),
            ("basic-drive", self.test_basic_drive),
            ("mecanum", self.test_mecanum_drive),
        ]
        if self.args.include_led:
            steps.insert(5, ("LED", self.test_led))
        return steps

    def test_system(self):
        steps = self.get_system_test_steps()
        print("\nStarting full car system test.")
        if not self.args.include_led:
            print("LED test is skipped. Use --include-led if you want to run it.")
        self.progress_bar(0, len(steps), "ready")
        original_yes = self.args.yes
        self.args.yes = True
        try:
            for step_number, (label, callback) in enumerate(steps, start=1):
                print(f"\n=== Step {step_number}/{len(steps)}: {label} ===")
                try:
                    callback()
                except Exception as exc:
                    print(f"{label} test failed: {exc}")
                self.progress_bar(step_number, len(steps), f"finished {label}")
        finally:
            self.args.yes = original_yes
        print("\nFull car system test complete.")

    def test_all(self):
        self.test_system()

    def cleanup(self):
        print("Cleaning up...")
        try:
            self.stop_motion()
        except Exception as exc:
            print(f"Motor cleanup failed: {exc}")
        try:
            if self.led is not None:
                self.led.colorBlink(0)
        except Exception as exc:
            print(f"LED cleanup failed: {exc}")
        try:
            if self.buzzer is not None:
                self.buzzer.set_state(False)
                self.buzzer.close()
        except Exception as exc:
            print(f"Buzzer cleanup failed: {exc}")
        try:
            if self.camera is not None:
                self.camera.close()
        except Exception as exc:
            print(f"Camera cleanup failed: {exc}")
        try:
            if self.motor is not None:
                self.motor.close()
        except Exception as exc:
            print(f"Motor close failed: {exc}")
        try:
            if self.servo is not None:
                self.servo.pwm_servo.close()
        except Exception as exc:
            print(f"Servo close failed: {exc}")
        try:
            if self.sonic is not None:
                self.sonic.close()
        except Exception as exc:
            print(f"Ultrasonic close failed: {exc}")
        try:
            if self.infrared is not None:
                self.infrared.close()
        except Exception as exc:
            print(f"Infrared close failed: {exc}")
        try:
            if self.adc is not None:
                self.adc.close_i2c()
        except Exception as exc:
            print(f"ADC close failed: {exc}")

    def run_named_test(self, name):
        tests = {
            "system": self.test_system,
            "basic-drive": self.test_basic_drive,
            "mecanum": self.test_mecanum_drive,
            "servo": self.test_servo,
            "ultrasonic": self.test_ultrasonic,
            "infrared": self.test_infrared,
            "obstacle": self.test_obstacle_sensors,
            "adc": self.test_adc,
            "sensors": self.test_sensors,
            "buzzer": self.test_buzzer,
            "speaker": self.test_speaker,
            "microphone": self.test_microphone,
            "audio": self.test_audio,
            "led": self.test_led,
            "camera": self.test_camera,
            "cameras": self.test_cameras,
            "cameras-labeled": self.test_cameras_labeled,
            "plane-map": self.test_plane_map,
            "all": self.test_all,
        }
        tests[name]()

    def menu(self):
        menu_items = [
            ("system", "Full car system test"),
            ("basic-drive", "Basic wheel movement"),
            ("mecanum", "Mecanum movement"),
            ("servo", "Servo sweep"),
            ("ultrasonic", "Ultrasonic distance"),
            ("infrared", "Infrared line sensors"),
            ("obstacle", "Extra infrared obstacle sensors"),
            ("adc", "ADC light and battery"),
            ("sensors", "All sensors"),
            ("buzzer", "Buzzer"),
            ("speaker", "External speaker"),
            ("microphone", "External microphone"),
            ("audio", "Microphone and speaker"),
            ("led", "LED pixels"),
            ("camera", "Camera capture"),
            ("cameras", "All camera captures"),
            ("cameras-labeled", "Labeled 4-camera capture (front/left/right/rear)"),
            ("plane-map", "Capture 4 cameras and stitch plane map"),
            ("all", "Run every test"),
        ]
        while True:
            print("\nFull car test menu")
            for index, (_, label) in enumerate(menu_items, start=1):
                print(f"  {index}. {label}")
            print("  q. Quit")
            choice = input("Select a test: ").strip().lower()
            if choice in ("q", "quit", "exit"):
                return
            try:
                selected = menu_items[int(choice) - 1][0]
            except (ValueError, IndexError):
                print("Invalid choice.")
                continue
            try:
                self.run_named_test(selected)
            except Exception as exc:
                print(f"{selected} test failed: {exc}")


def build_parser():
    parser = argparse.ArgumentParser(description="Interactive full-car hardware test runner.")
    parser.add_argument(
        "--test",
        choices=[
            "basic-drive",
            "system",
            "mecanum",
            "servo",
            "ultrasonic",
            "infrared",
            "obstacle",
            "adc",
            "sensors",
            "buzzer",
            "speaker",
            "microphone",
            "audio",
            "led",
            "camera",
            "cameras",
            "cameras-labeled",
            "plane-map",
            "all",
        ],
        help="Run one test directly instead of opening the menu.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip interactive safety confirmations.")
    parser.add_argument("--include-led", action="store_true", help="Include LED test in the full system test.")
    parser.add_argument("--gear-ratio", type=float, default=120, help="Motor gearbox ratio used for movement-test notes.")
    parser.add_argument("--reference-gear-ratio", type=float, default=48, help="Original gearbox ratio for speed comparison notes.")
    parser.add_argument("--speed", type=int, default=900, help="Wheel test speed, range is normally -4095..4095.")
    parser.add_argument("--turn-speed", type=int, default=900, help="Wheel turn test speed.")
    parser.add_argument("--duration", type=float, default=1.5, help="Seconds for each movement step.")
    parser.add_argument("--turn-90-duration", type=float, help="Set both left and right 90-degree rotation durations.")
    parser.add_argument("--left-turn-90-duration", type=float, default=2.4, help="Seconds for the timed left 90-degree rotation.")
    parser.add_argument("--right-turn-90-duration", type=float, default=2.1, help="Seconds for the timed right 90-degree return rotation.")
    parser.add_argument("--pause", type=float, default=0.3, help="Pause between movement steps.")
    parser.add_argument("--samples", type=int, default=8, help="Sensor sample count.")
    parser.add_argument("--servo0-min", type=int, default=0, help="Minimum angle for servo channel 0.")
    parser.add_argument("--servo0-max", type=int, default=180, help="Maximum angle for servo channel 0.")
    parser.add_argument("--servo1-min", type=int, default=80, help="Minimum angle for servo channel 1.")
    parser.add_argument("--servo1-max", type=int, default=180, help="Maximum angle for servo channel 1.")
    parser.add_argument("--camera-file", default="full_car_test_image.jpg", help="Output path for camera capture.")
    parser.add_argument("--camera-count", type=int, help="Legacy alias for Picamera indexes to test.")
    parser.add_argument("--picamera-count", type=int, default=2, help="Number of Picamera indexes to test.")
    parser.add_argument("--usb-camera-count", type=int, default=3, help="Number of USB camera indexes to test.")
    parser.add_argument("--usb-camera-start", type=int, default=0, help="First /dev/video index for USB camera tests.")
    parser.add_argument("--usb-camera-indexes", default="0,10,37", help="Comma-separated /dev/video indexes for USB camera tests, for example 0,10,37.")
    parser.add_argument("--csi-camera-num", type=int, default=1, help="CSI Picamera index for front camera (labeled capture).")
    parser.add_argument(
        "--labeled-camera",
        choices=["front", "left", "right", "rear"],
        help="Capture only one labeled camera (open that camera alone).",
    )
    parser.add_argument("--labeled-camera-dir", default="camera_labeled", help="Directory for labeled front/left/right/rear captures.")
    parser.add_argument("--camera-dir", default="camera_test_images", help="Directory for multi-camera captures.")
    parser.add_argument("--camera-width", type=int, default=1920, help="USB capture width for multi-camera tests.")
    parser.add_argument("--camera-height", type=int, default=1080, help="USB capture height for multi-camera tests.")
    parser.add_argument("--csi-camera-width", type=int, default=1920, help="CSI capture width for multi-camera tests.")
    parser.add_argument("--csi-camera-height", type=int, default=1440, help="CSI capture height for multi-camera tests.")
    parser.add_argument("--camera-warmup", type=float, default=2.0, help="Seconds to wait after opening each camera before discarding warmup frames.")
    parser.add_argument("--camera-warmup-frames", type=int, default=15, help="Number of preview frames to discard so auto-exposure can settle.")
    parser.add_argument("--camera-settle", type=float, default=1.0, help="Seconds to wait between closing one camera and opening the next.")
    parser.add_argument("--obstacle-pins", default="26,20,19,16,6,12", help="Comma-separated BCM GPIO pins for extra infrared obstacle sensors.")
    parser.set_defaults(obstacle_active_high=False)
    parser.add_argument("--obstacle-active-high", action="store_true", help="Treat HIGH as obstacle detected.")
    parser.add_argument("--obstacle-active-low", action="store_false", dest="obstacle_active_high", help="Treat LOW as obstacle detected. This is the default.")
    parser.add_argument("--audio-dir", default="audio_test", help="Directory for generated and recorded audio files.")
    parser.add_argument("--audio-duration", type=int, default=3, help="Audio test duration in seconds.")
    parser.add_argument("--audio-rate", type=int, default=44100, help="Audio sample rate.")
    parser.add_argument("--speaker-frequency", type=int, default=880, help="Speaker test tone frequency in Hz.")
    parser.add_argument("--speaker-device", default="plughw:2,0", help="ALSA playback device, for example plughw:2,0.")
    parser.add_argument("--mic-device", default="plughw:2,0", help="ALSA capture device, for example plughw:3,0.")
    parser.add_argument("--no-audio-playback", action="store_true", help="Record microphone audio without playing it back.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.turn_90_duration is not None:
        args.left_turn_90_duration = args.turn_90_duration
        args.right_turn_90_duration = args.turn_90_duration
    if args.gear_ratio > 0 and args.reference_gear_ratio > 0:
        speed_factor = args.reference_gear_ratio / args.gear_ratio
        torque_factor = args.gear_ratio / args.reference_gear_ratio
        print(
            f"Motor gear ratio: 1:{args.gear_ratio:g} "
            f"(about {speed_factor:.2f}x wheel speed and {torque_factor:.2f}x torque vs 1:{args.reference_gear_ratio:g})."
        )
    tester = FullCarTester(args)
    try:
        if args.test:
            tester.run_named_test(args.test)
        else:
            tester.menu()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
