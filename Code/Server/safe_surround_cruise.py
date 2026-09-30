#!/usr/bin/env python3
"""Safe multi-camera cruise MVP.

Control cameras:
  front = CSI camera_num=1
  rear = CSI camera_num=0
  left/right = two fixed USB fisheye cameras, resolved by usb_bus
Gimbal camera:
  USB fisheye on servo channels 0/1. Pan stays centred while driving
  because the ultrasonic sensor is mounted on the same SG90. Front CSI
  supplies wall bearing; while stopped the gimbal may peek at that
  heading so ultrasonic can range the wall, then it returns home.

The program is dry-run by default. Motors are created only with --arm after
camera, board-power and battery preflight checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from gpiozero import DigitalInputDevice

from camera_devices import (
    csi_array_to_bgr,
    create_csi_still_configuration,
    resolve_usb_capture_index,
)
from full_car_test import estimate_battery_percent
from motor import Ordinary_Car
from servo import Servo, load_gimbal_home
from ultrasonic import Ultrasonic
from vision_detector import classic_corridor_score, draw_debug, wall_aim_point
from csi_wall_aim import load_cfg as load_aim_cfg
from csi_wall_aim import load_front_kd, pan_for_yaw, pixel_yaw_deg


SERVER = Path(__file__).resolve().parent
HARDWARE_PATH = SERVER / "camera_hardware.json"
PARAMS_PATH = SERVER / "params.json"

IR_PINS = {
    "front_left": 26,
    "front": 20,
    "front_right": 19,
    "left": 16,
    "right": 6,
    "rear": 12,
}
BOARD_SENSE_PINS = (14, 15, 23, *IR_PINS.values())


def parse_args():
    p = argparse.ArgumentParser(
        description="Safe surround cruise (dry-run unless --arm is supplied)."
    )
    p.add_argument(
        "--arm",
        action="store_true",
        help="Enable motors after preflight. Without this flag PWM is never sent.",
    )
    p.add_argument("--speed", type=int, default=600, help="Forward PWM.")
    p.add_argument("--turn-speed", type=int, default=550, help="Turn PWM.")
    p.add_argument("--back-speed", type=int, default=500, help="Reverse PWM.")
    p.add_argument(
        "--motor-min-pwm",
        type=int,
        default=900,
        help="Minimum outer-wheel PWM while moving (1:120 motors stall below about 850-900).",
    )
    p.add_argument("--block-score", type=float, default=0.36)
    p.add_argument("--slow-score", type=float, default=0.20)
    p.add_argument("--near-score", type=float, default=0.70)
    p.add_argument(
        "--side-block-score",
        type=float,
        default=0.70,
        help="Side-risk level that starts avoidance.",
    )
    p.add_argument(
        "--side-clear-score",
        type=float,
        default=0.45,
        help="Lower hysteresis level that ends side avoidance.",
    )
    p.add_argument("--stop-cm", type=float, default=18.0)
    p.add_argument(
        "--sonic-clear-cm",
        type=float,
        default=30.0,
        help="Keep the chosen avoidance turn until ultrasonic distance exceeds this.",
    )
    p.add_argument("--min-turn-s", type=float, default=0.8)
    p.add_argument(
        "--max-turn-s",
        type=float,
        default=1.6,
        help="Drop an in-place turn after this even if ultrasonic is not yet sonic-clear, as long as the nose is past --stop-cm.",
    )
    p.add_argument(
        "--turn-cooldown-s",
        type=float,
        default=1.0,
        help="After an in-place turn, wait this long before starting another one unless ultrasonic is inside --stop-cm.",
    )
    p.add_argument("--hard-stop-cm", type=float, default=12.0)
    p.add_argument(
        "--creep-cm",
        type=float,
        default=32.0,
        help="Slow to a crawl once ultrasonic is closer than this, even if vision is clear.",
    )
    p.add_argument("--reverse-clear-cm", type=float, default=25.0)
    p.add_argument("--min-reverse-s", type=float, default=0.6)
    p.add_argument(
        "--rear-block-score",
        type=float,
        default=0.85,
        help="Rear-camera risk that prohibits emergency reversing.",
    )
    p.add_argument("--min-battery-v", type=float, default=7.0)
    p.add_argument(
        "--require-board-sense",
        action="store_true",
        help="Refuse ARM unless GPIO heuristics positively confirm board power.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--usb-width",
        type=int,
        default=320,
        help="USB capture width. 320 keeps YUYV bandwidth low enough for two side cameras.",
    )
    p.add_argument("--usb-height", type=int, default=240)
    p.add_argument("--usb-fps", type=float, default=4.0)
    p.add_argument("--usb-retry-s", type=float, default=3.0)
    p.add_argument("--usb-read-fail-limit", type=int, default=12)
    p.add_argument(
        "--usb-holdoff-s",
        type=float,
        default=4.0,
        help="After a USB disconnect, wait this long before any camera reopens.",
    )
    p.add_argument("--loop-hz", type=float, default=8.0)
    p.add_argument(
        "--loop-watchdog-s",
        type=float,
        default=1.5,
        help="Stop if one control iteration exceeds this duration.",
    )
    p.add_argument("--camera-stale-s", type=float, default=2.0)
    p.add_argument("--preflight-s", type=float, default=12.0)
    p.add_argument("--max-frames", type=int, default=0)
    p.set_defaults(gimbal=True)
    p.add_argument("--gimbal", action="store_true", dest="gimbal")
    p.add_argument("--no-gimbal", action="store_false", dest="gimbal")
    p.add_argument(
        "--no-fifth",
        action="store_false",
        dest="gimbal",
        help=argparse.SUPPRESS,
    )
    home_pan, home_tilt = load_gimbal_home()
    p.add_argument(
        "--gimbal-angles",
        default=f"{max(0, home_pan - 45)},{home_pan},{min(180, home_pan + 45)}",
        help="left,centre,right pan. Centre is the saved straight-ahead pose.",
    )
    p.add_argument(
        "--gimbal-tilt",
        type=int,
        default=home_tilt,
        help="Tilt for straight-ahead. Default comes from gimbal_home.json.",
    )
    p.add_argument("--gimbal-settle-s", type=float, default=0.40)
    p.add_argument("--no-ir", action="store_true")
    p.add_argument("--no-sonic", action="store_true")
    p.add_argument(
        "--front-ir-gate-cm",
        type=float,
        default=18.0,
        help="Ignore centre front IR until ultrasonic is within this range, unless sonic is missing.",
    )
    p.add_argument(
        "--front-side-ir-gate-cm",
        type=float,
        default=28.0,
        help="Ignore front-left/front-right IR until ultrasonic is within this range, unless sonic is missing.",
    )
    p.add_argument(
        "--debug-dir",
        default="safe_surround_debug",
        help="Directory for optional front debug frames.",
    )
    p.add_argument("--debug-every", type=int, default=0)
    p.set_defaults(active_high=False)
    p.add_argument("--ir-active-high", action="store_true", dest="active_high")
    p.add_argument("--ir-active-low", action="store_false", dest="active_high")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pcb_version() -> int:
    try:
        return int(load_json(PARAMS_PATH).get("Pcb_Version", 2))
    except Exception:
        return 2


def read_battery_voltage() -> float | None:
    """Read the 2S pack voltage without ADC's unbounded stable-read loop."""
    try:
        import smbus

        pcb = pcb_version()
        coefficient = 3.3 if pcb == 1 else 5.2
        scale = 3 if pcb == 1 else 2
        bus = smbus.SMBus(1)
        try:
            channel = 2
            command = 0x84 | ((((channel << 2) | (channel >> 1)) & 0x07) << 4)
            bus.write_byte(0x48, command)
            bus.read_byte(0x48)
            raw = bus.read_byte(0x48)
            return round(raw / 255.0 * coefficient * scale, 2)
        finally:
            bus.close()
    except Exception:
        return None


def board_power_on() -> tuple[bool | None, int]:
    """Return True when GPIO proves board power, otherwise None (unknown).

    IR and line-sensor outputs may legitimately all be LOW, and before gpiozero
    configures the pins several appear as "none". Therefore zero HIGH pins is
    not evidence that the board switch is off.
    """
    try:
        proc = subprocess.run(
            ["pinctrl", "get", ",".join(str(pin) for pin in BOARD_SENSE_PINS)],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        high = sum("| hi" in line for line in proc.stdout.splitlines())
        return (True if high > 0 else None), high
    except Exception:
        return None, 0


@dataclass
class CameraSample:
    frame: np.ndarray | None = None
    scores: dict[str, float] | None = None
    timestamp: float = 0.0
    frames: int = 0
    error: str = ""


class SurroundCameraManager:
    """Own all camera devices; worker threads publish only complete frames."""

    def __init__(self, args):
        self.args = args
        self.hw = load_json(HARDWARE_PATH)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.samples = {
            role: CameraSample() for role in ("front", "left", "right", "rear", "gimbal")
        }
        self.usb_indexes: dict[str, int] = {}
        self.threads: list[threading.Thread] = []
        self.usb_open_lock = threading.Lock()
        self.usb_holdoff_until = 0.0
        self.gimbal_live = threading.Event()
        self._usb_stagger = {"left": 0.0, "right": 0.7, "gimbal": 1.4}

    def _occupied_usb_indexes(self, role: str) -> set[int]:
        with self.lock:
            return {idx for other, idx in self.usb_indexes.items() if other != role}

    def _claim_usb_index(self, role: str, index: int):
        with self.lock:
            self.usb_indexes[role] = index

    def _release_usb_index(self, role: str):
        with self.lock:
            self.usb_indexes.pop(role, None)

    def _publish(self, role: str, frame, score=True):
        scores = classic_corridor_score(frame) if score else None
        with self.lock:
            previous = self.samples[role]
            self.samples[role] = CameraSample(
                frame=frame,
                scores=scores,
                timestamp=time.monotonic(),
                frames=previous.frames + 1,
            )

    def _set_error(self, role: str, exc):
        with self.lock:
            self.samples[role].error = str(exc)

    @staticmethod
    def _torn(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        row_delta = np.abs(gray[1:] - gray[:-1]).mean(axis=1)
        if row_delta.size < 8:
            return False
        bottom = float(row_delta[-40:].max()) if row_delta.size >= 40 else 0.0
        return float(row_delta.max()) > 62.0 or bottom > 46.0

    def _note_usb_disconnect(self):
        hold = max(1.0, float(self.args.usb_holdoff_s))
        with self.lock:
            self.usb_holdoff_until = max(self.usb_holdoff_until, time.monotonic() + hold)

    def _wait_usb_holdoff(self):
        while not self.stop_event.is_set():
            with self.lock:
                remain = self.usb_holdoff_until - time.monotonic()
            if remain <= 0:
                return
            self.stop_event.wait(min(remain, 0.4))

    def _open_usb_capture(self, index: int):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cannot open /dev/video{index}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.usb_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.usb_height)
        cap.set(cv2.CAP_PROP_FPS, self.args.usb_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        return cap

    def _usb_loop(self, role: str, entry: dict):
        interval = 1.0 / max(self.args.usb_fps, 1.0)
        if role == "gimbal":
            interval = 1.0 / max(min(self.args.usb_fps, 2.0), 0.5)
        fail_limit = max(1, int(self.args.usb_read_fail_limit))
        stagger = self._usb_stagger.get(role, 0.0)
        while not self.stop_event.is_set():
            if role == "gimbal" and not self.gimbal_live.is_set():
                self.stop_event.wait(0.25)
                continue
            cap = None
            retry_s = max(0.1, self.args.usb_retry_s)
            try:
                self._wait_usb_holdoff()
                if stagger and self.stop_event.wait(stagger):
                    return
                stagger = 0.0
                with self.usb_open_lock:
                    if role == "gimbal" and not self.gimbal_live.is_set():
                        continue
                    index, resolved = resolve_usb_capture_index(
                        entry,
                        occupied=self._occupied_usb_indexes(role),
                        force_list=True,
                    )
                    print(f"{role}: {resolved}", flush=True)
                    cap = self._open_usb_capture(index)
                    self._claim_usb_index(role, index)
                    if self.stop_event.wait(0.6):
                        return
                    for _ in range(4):
                        if self.stop_event.is_set():
                            return
                        cap.grab()
                last_good = None
                read_failures = 0
                while not self.stop_event.is_set():
                    if role == "gimbal" and not self.gimbal_live.is_set():
                        break
                    started = time.monotonic()
                    if not os.path.exists(f"/dev/video{index}"):
                        self._note_usb_disconnect()
                        raise RuntimeError(f"/dev/video{index} disappeared")
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        read_failures += 1
                        if not cap.isOpened():
                            self._note_usb_disconnect()
                            raise RuntimeError(f"/dev/video{index} closed")
                        if read_failures >= fail_limit:
                            self._note_usb_disconnect()
                            raise RuntimeError(
                                f"/dev/video{index} failed {read_failures} consecutive reads"
                            )
                    else:
                        read_failures = 0
                        want = (self.args.usb_height, self.args.usb_width)
                        if frame.shape[0] != want[0] or frame.shape[1] != want[1]:
                            frame = cv2.resize(
                                frame,
                                (want[1], want[0]),
                                interpolation=cv2.INTER_AREA,
                            )
                        if not self._torn(frame) or last_good is None:
                            last_good = frame
                            self._publish(role, frame)
                    elapsed = time.monotonic() - started
                    self.stop_event.wait(max(0.0, interval - elapsed))
            except Exception as exc:
                self._set_error(role, exc)
                print(f"WARNING {role}: {exc}; retrying this USB camera only", flush=True)
                text = str(exc).lower()
                if (
                    "already used" in text
                    or "not stealing" in text
                    or "disappeared" in text
                    or "no such device" in text
                    or "cannot open" in text
                ):
                    self._note_usb_disconnect()
                    retry_s = max(retry_s, float(self.args.usb_holdoff_s))
            finally:
                self._release_usb_index(role)
                if cap is not None:
                    cap.release()
            if role == "gimbal" and not self.gimbal_live.is_set():
                continue
            self.stop_event.wait(retry_s)

    def _csi_loop(self, role: str, camera_num: int, score: bool, fps: float):
        interval = 1.0 / max(fps, 0.5)
        while not self.stop_event.is_set():
            camera = None
            try:
                from picamera2 import Picamera2

                camera = Picamera2(camera_num=camera_num)
                config = create_csi_still_configuration(
                    camera, self.args.width, self.args.height
                )
                camera.configure(config)
                camera.start()
                camera.set_controls({"AeEnable": True, "AwbEnable": True})
                time.sleep(0.8)
                for _ in range(6):
                    if self.stop_event.is_set():
                        return
                    camera.capture_array()
                print(f"{role}: CSI camera_num={camera_num}", flush=True)
                misses = 0
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    try:
                        frame = csi_array_to_bgr(camera.capture_array())
                    except Exception as exc:
                        misses += 1
                        if misses >= 3:
                            raise
                        print(
                            f"WARNING {role}: CSI frame miss {misses}/3 ({exc})",
                            flush=True,
                        )
                        self.stop_event.wait(0.2)
                        continue
                    if frame is not None:
                        misses = 0
                        self._publish(role, frame, score=score)
                    elapsed = time.monotonic() - started
                    self.stop_event.wait(max(0.0, interval - elapsed))
            except Exception as exc:
                self._set_error(role, exc)
                print(f"WARNING {role}: {exc}; retrying CSI camera", flush=True)
            finally:
                if camera is not None:
                    try:
                        camera.stop()
                    except Exception:
                        pass
                    try:
                        camera.close()
                    except Exception:
                        pass
            self.stop_event.wait(2.0)

    def start(self):
        usb_entries = {
            "left": self.hw["left"],
            "right": self.hw["right"],
        }
        if self.args.gimbal:
            usb_entries["gimbal"] = self.hw["gimbal"]
        # Resolve/open USB before Picamera2 enumerates libcamera UVC devices.
        for role, entry in usb_entries.items():
            thread = threading.Thread(
                target=self._usb_loop, args=(role, entry), daemon=True, name=f"cam-{role}"
            )
            thread.start()
            self.threads.append(thread)
        time.sleep(0.4)
        front = threading.Thread(
            target=self._csi_loop,
            args=("front", 1, True, min(10.0, self.args.loop_hz)),
            daemon=True,
            name="cam-front",
        )
        front.start()
        self.threads.append(front)
        rear = threading.Thread(
            target=self._csi_loop,
            args=("rear", 0, True, 4.0),
            daemon=True,
            name="cam-rear",
        )
        rear.start()
        self.threads.append(rear)

    def snapshot(self) -> dict[str, CameraSample]:
        with self.lock:
            return {
                role: CameraSample(
                    frame=sample.frame.copy() if sample.frame is not None else None,
                    scores=dict(sample.scores) if sample.scores else None,
                    timestamp=sample.timestamp,
                    frames=sample.frames,
                    error=sample.error,
                )
                for role, sample in self.samples.items()
            }

    def wait_ready(self, timeout: float) -> dict[str, CameraSample]:
        deadline = time.monotonic() + timeout
        required = ("front", "left", "right", "rear")
        while time.monotonic() < deadline and not self.stop_event.is_set():
            samples = self.snapshot()
            if all(samples[role].timestamp > 0 for role in required):
                return samples
            time.sleep(0.1)
        return self.snapshot()

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=3.0)


class SafetySensors:
    def __init__(self, args):
        self.args = args
        self.ir: dict[str, DigitalInputDevice] = {}
        self.sonic = None
        if not args.no_ir:
            for name, pin in IR_PINS.items():
                self.ir[name] = DigitalInputDevice(pin, pull_up=False)
        if not args.no_sonic:
            self.sonic = Ultrasonic()

    def read_ir(self) -> dict[str, int]:
        values = {name: 0 for name in IR_PINS}
        for name, device in self.ir.items():
            raw = int(device.value)
            values[name] = raw if self.args.active_high else int(not raw)
        return values

    def read_sonic(self) -> float | None:
        if self.sonic is None:
            return None
        values = []
        for _ in range(3):
            cm = self.sonic.get_distance()
            if cm is not None and 1.0 <= cm <= 300.0:
                values.append(float(cm))
        if not values:
            return None
        values.sort()
        return values[len(values) // 2]

    def close(self):
        if self.sonic is not None:
            self.sonic.close()
        for device in self.ir.values():
            device.close()


class SafeSurroundCruise:
    def __init__(self, args):
        self.args = args
        self.cameras = SurroundCameraManager(args)
        self.sensors = None
        self.motor = None
        self.servo = None
        self.stop_event = threading.Event()
        self.smooth = {role: 0.0 for role in ("front", "left", "right", "rear")}
        self.smooth_ready = set()
        self.side_blocked = {"left": False, "right": False}
        self.turn_latch: str | None = None
        self.turn_latch_since = 0.0
        self.turn_cooldown_until = 0.0
        self.reverse_latch = False
        self.reverse_latch_since = 0.0
        self.gimbal_angles = [
            int(value.strip())
            for value in self.args.gimbal_angles.split(",")
            if value.strip()
        ]
        if len(self.gimbal_angles) != 3:
            raise ValueError("--gimbal-angles needs three values: left,centre,right")
        self.gimbal_label = "centre"
        self.gimbal_angle = self.gimbal_angles[1]
        self.gimbal_moved_at = 0.0
        self.gimbal_last_frame = -1
        self.gimbal_scores = {"left": None, "centre": None, "right": None}
        self.gimbal_sonic = {"left": None, "centre": None, "right": None}
        self.gimbal_far_jumps = {"left": 0, "centre": 0, "right": 0}
        self.last_front_sonic: float | None = None
        self.wall_yaw: float | None = None
        self.peek_phase = "idle"
        self.peek_pan = 0
        self.peek_sonic: float | None = None
        self.peek_confirm: float | None = None
        self.aim_cfg = load_aim_cfg()
        try:
            self.front_k, self.front_d = load_front_kd()
        except Exception as exc:
            print(f"CSI wall bearing disabled ({exc}).", flush=True)
            self.front_k = None
            self.front_d = None
        self.last_power_check = 0.0
        self.power_ok = False
        self.debug_dir = SERVER / args.debug_dir
        if args.debug_every > 0:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def request_stop(self, *_args):
        self.stop_event.set()
        self.stop_motors()

    def stop_motors(self):
        if self.motor is not None:
            try:
                self.motor.set_motor_model(0, 0, 0, 0)
            except Exception:
                pass

    def start_gimbal(self):
        if not self.args.gimbal:
            print("SG90 gimbal disabled.", flush=True)
            return
        self.servo = Servo()
        self.servo.set_servo_pwm("0", self.gimbal_angles[1])
        self.servo.set_servo_pwm("1", self.args.gimbal_tilt)
        self.gimbal_label = "centre"
        self.gimbal_angle = self.gimbal_angles[1]
        self.gimbal_moved_at = time.monotonic()
        print(
            f"SG90 gimbal enabled: pan locked at {self.gimbal_angles[1]} while driving "
            f"(tilt={self.args.gimbal_tilt}). CSI bearing may peek only while stopped.",
            flush=True,
        )

    def update_gimbal_observation(
        self, samples, sonic: float | None, now: float
    ) -> float | None:
        if self.servo is None:
            if sonic is not None:
                self.last_front_sonic = sonic
            return sonic
        settling = now < self.gimbal_moved_at + self.args.gimbal_settle_s
        sample = samples.get("gimbal")
        if (
            not settling
            and sample is not None
            and sample.frames != self.gimbal_last_frame
            and sample.timestamp >= self.gimbal_moved_at
            and sample.scores
        ):
            mid = float(sample.scores.get("mid", 0.0))
            if self.gimbal_label in self.gimbal_scores:
                self.gimbal_scores[self.gimbal_label] = mid
            if self.gimbal_label == "peek":
                self.peek_confirm = mid
            self.gimbal_last_frame = sample.frames

        if settling or self.gimbal_label != "centre":
            # Side looks and servo motion must never replace the last forward range.
            if sonic is not None and self.gimbal_label != "centre" and not settling:
                if self.gimbal_label in self.gimbal_sonic:
                    self.gimbal_sonic[self.gimbal_label] = sonic
                if self.gimbal_label == "peek":
                    self.peek_sonic = sonic
            return self.last_front_sonic

        if sonic is not None:
            previous = self.last_front_sonic
            if previous is not None and previous < 40.0 and sonic > 120.0:
                self.gimbal_far_jumps["centre"] += 1
                if self.gimbal_far_jumps["centre"] <= 4:
                    sonic = previous
                else:
                    self.gimbal_far_jumps["centre"] = 0
            else:
                self.gimbal_far_jumps["centre"] = 0
            self.gimbal_sonic["centre"] = sonic
            self.last_front_sonic = sonic
        return self.last_front_sonic

    def _home_pan(self) -> int:
        return self.gimbal_angles[1]

    def _update_wall_yaw(self, samples) -> None:
        sample = samples.get("front")
        if (
            self.front_k is None
            or sample is None
            or sample.frame is None
            or self._age(sample, time.monotonic()) > self.args.camera_stale_s
        ):
            return
        aim = wall_aim_point(sample.frame, circle=True)
        if aim is None:
            return
        yaw = pixel_yaw_deg(
            aim["x"], aim["y"], self.front_k, self.front_d, sample.frame.shape[1]
        )
        if self.wall_yaw is None:
            self.wall_yaw = yaw
        else:
            self.wall_yaw = 0.65 * self.wall_yaw + 0.35 * yaw

    def _desired_gimbal_pose(self, action: str, now: float) -> tuple[str, int]:
        home = self._home_pan()
        moving = (
            action in ("forward", "slow")
            or action.startswith("veer")
            or action.startswith("turn")
            or action.startswith("reverse")
        )
        if moving:
            self.peek_phase = "idle"
            return "centre", home
        if self.peek_phase == "go_wall":
            return "peek", self.peek_pan
        if self.peek_phase == "go_home":
            return "centre", home
        return "centre", home

    def aim_gimbal_pan(self, pan: int, now: float, label: str) -> bool:
        if self.servo is None:
            return False
        pan = max(0, min(180, int(pan)))
        if label == self.gimbal_label and pan == self.gimbal_angle:
            return False
        self.gimbal_label = label
        self.gimbal_angle = pan
        self.servo.set_servo_pwm("0", pan)
        self.servo.set_servo_pwm("1", self.args.gimbal_tilt)
        self.gimbal_moved_at = now
        return True

    def _advance_peek(self, action: str, now: float, raw_sonic, samples) -> None:
        if self.servo is None:
            self.peek_phase = "idle"
            return
        settled = now >= self.gimbal_moved_at + self.args.gimbal_settle_s
        moving = (
            action in ("forward", "slow")
            or action.startswith("veer")
            or action.startswith("turn")
            or action.startswith("reverse")
        )
        too_close = (
            self.last_front_sonic is not None
            and self.last_front_sonic <= self.args.stop_cm
        )
        if moving or too_close or self.reverse_latch:
            if self.peek_phase in ("go_wall", "go_home"):
                self.peek_phase = "go_home"
            elif self.peek_phase == "done":
                self.peek_phase = "idle"
            if too_close or self.reverse_latch or action.startswith("reverse"):
                self.peek_phase = "idle"
            return
        boxed = "boxed" in action
        if (
            self.peek_phase == "idle"
            and boxed
            and self.wall_yaw is not None
            and abs(self.wall_yaw) >= 8.0
        ):
            self.peek_pan = pan_for_yaw(self._home_pan(), self.wall_yaw, self.aim_cfg)
            if abs(self.peek_pan - self._home_pan()) >= 4:
                self.peek_phase = "go_wall"
                self.peek_sonic = None
                self.peek_confirm = None
            return
        if self.peek_phase == "go_wall" and self.gimbal_label == "peek" and settled:
            if raw_sonic is not None:
                self.peek_sonic = float(raw_sonic)
            sample = samples.get("gimbal")
            if sample is not None and sample.scores:
                self.peek_confirm = float(sample.scores.get("mid", 0.0))
            self.peek_phase = "go_home"
            return
        if self.peek_phase == "go_home" and self.gimbal_label == "centre" and settled:
            self.peek_phase = "done"
            return
        if self.peek_phase == "done" and not boxed:
            self.peek_phase = "idle"

    def _desired_gimbal_label(self) -> str:
        return "centre"

    def aim_gimbal(self, label: str, now: float) -> bool:
        index = {"left": 0, "centre": 1, "right": 2}[label]
        return self.aim_gimbal_pan(self.gimbal_angles[index], now, label)

    def directional_sonic(self, sonic: float | None, now: float) -> float | None:
        if self.servo is None:
            return sonic
        if self.gimbal_label != "centre":
            return self.last_front_sonic
        if now < self.gimbal_moved_at + self.args.gimbal_settle_s:
            return self.last_front_sonic
        return sonic

    def _age(self, sample: CameraSample, now: float) -> float:
        return float("inf") if sample.timestamp <= 0 else now - sample.timestamp

    def _risk(self, role: str, value: float) -> float:
        alpha = 0.40
        if role not in self.smooth_ready:
            self.smooth[role] = value
            self.smooth_ready.add(role)
        else:
            self.smooth[role] = alpha * value + (1.0 - alpha) * self.smooth[role]
        return self.smooth[role]

    def camera_risks(self, samples):
        now = time.monotonic()

        def live(role: str) -> dict[str, float]:
            sample = samples[role]
            if self._age(sample, now) > self.args.camera_stale_s or not sample.scores:
                return {}
            return sample.scores

        front = live("front")
        left = live("left")
        right = live("right")
        rear = live("rear")
        raw = {
            "front": float(front["mid"]) if "mid" in front else None,
            "left": (
                0.70 * float(left["mid"]) + 0.30 * float(front.get("left", 0.0))
                if "mid" in left
                else (float(front["left"]) if "left" in front else None)
            ),
            "right": (
                0.70 * float(right["mid"]) + 0.30 * float(front.get("right", 0.0))
                if "mid" in right
                else (float(front["right"]) if "right" in front else None)
            ),
            "rear": float(rear["mid"]) if "mid" in rear else None,
        }
        risks = {}
        for role, value in raw.items():
            if value is None:
                # Keep the last live estimate. Zeroing on USB/CSI drop makes
                # a blocked corridor look clear, then the car freezes or drives in.
                risks[role] = self.smooth[role] if role in self.smooth_ready else 0.0
            else:
                risks[role] = self._risk(role, value)
        return risks

    def _turn_left(self):
        mag = self._moving_pwm(self.args.turn_speed)
        return (-mag, -mag, mag, mag), "turn_left"

    def _turn_right(self):
        mag = self._moving_pwm(self.args.turn_speed)
        return (mag, mag, -mag, -mag), "turn_right"

    def _veer_left(self):
        """Keep both sides above stall PWM; speed up the outside for a gentle arc."""
        inner = self._moving_pwm(self.args.speed)
        outer = min(4095, max(inner + 300, int(inner * 1.25)))
        return (inner, inner, outer, outer), "veer_left"

    def _veer_right(self):
        """Keep both sides above stall PWM; speed up the outside for a gentle arc."""
        inner = self._moving_pwm(self.args.speed)
        outer = min(4095, max(inner + 300, int(inner * 1.25)))
        return (outer, outer, inner, inner), "veer_right"

    def _moving_pwm(self, requested: int) -> int:
        requested = max(0, min(4095, int(requested)))
        floor = max(0, min(4095, int(self.args.motor_min_pwm)))
        return max(requested, floor)

    def _latched_turn(self):
        return self._turn_left() if self.turn_latch == "left" else self._turn_right()

    def _start_turn(self, direction: str, now: float):
        self.reverse_latch = False
        self.turn_latch = direction
        self.turn_latch_since = now
        return self._latched_turn()

    def _end_turn(self, now: float) -> None:
        self.turn_latch = None
        self.turn_cooldown_until = now + self.args.turn_cooldown_s

    def _sonic_far(self, sonic: float | None) -> bool:
        return sonic is not None and sonic >= self.args.creep_cm

    def _reverse(self):
        speed = self._moving_pwm(self.args.back_speed)
        return (-speed, -speed, -speed, -speed), "reverse_escape"

    def _side_is_blocked(self, role: str, risk: float) -> bool:
        # Separate enter/exit levels prevent 0.35/0.37 threshold chatter.
        if self.side_blocked[role]:
            if risk <= self.args.side_clear_score:
                self.side_blocked[role] = False
        elif risk >= self.args.side_block_score:
            self.side_blocked[role] = True
        return self.side_blocked[role]

    def _rear_safe(self, risks, stale, ir) -> bool:
        rear_ir = bool(ir.get("rear", 0))
        rear_camera_ok = (not stale["rear"]) and risks["rear"] < self.args.rear_block_score
        if rear_camera_ok:
            # Live rear camera with a clear corridor overrides a sticky rear IR pin.
            return True
        return stale["rear"] and not rear_ir

    def _side_flags(self, risks, stale, ir) -> tuple[bool, bool]:
        left_block = (
            not stale["left"] and self._side_is_blocked("left", risks["left"])
        ) or bool(ir.get("left", 0))
        right_block = (
            not stale["right"] and self._side_is_blocked("right", risks["right"])
        ) or bool(ir.get("right", 0))
        return left_block, right_block

    def _ir_near(self, ir, name: str, sonic: float | None, gate_cm: float) -> bool:
        if not ir.get(name, 0):
            return False
        # Missing ultrasonic must not disable IR; that combination drives into walls.
        return sonic is None or sonic <= gate_cm

    def _pick_turn(self, risks, ir, left_block: bool, right_block: bool) -> str | None:
        front_left = bool(ir.get("front_left", 0))
        front_right = bool(ir.get("front_right", 0))
        if front_left and not front_right and not right_block:
            return "right"
        if front_right and not front_left and not left_block:
            return "left"
        yaw = self.wall_yaw
        if yaw is not None:
            if yaw > 8.0 and not left_block:
                return "left"
            if yaw < -8.0 and not right_block:
                return "right"
        if not left_block and (right_block or risks["left"] <= risks["right"]):
            return "left"
        if not right_block:
            return "right"
        return None

    def _turn_or_box(self, risks, ir, left_block, right_block, now, boxed_name: str):
        direction = self._pick_turn(risks, ir, left_block, right_block)
        if (
            direction is None
            and self.last_front_sonic is not None
            and self.last_front_sonic > self.args.stop_cm
        ):
            yaw = 0.0 if self.wall_yaw is None else self.wall_yaw
            if yaw > 4.0:
                direction = "left"
            elif yaw < -4.0:
                direction = "right"
            else:
                direction = "left" if risks["left"] <= risks["right"] else "right"
        if direction:
            return self._start_turn(direction, now)
        return (0, 0, 0, 0), boxed_name

    def decide(self, risks, samples, ir, sonic):
        now = time.monotonic()
        stale = {
            role: self._age(samples[role], now) > self.args.camera_stale_s
            for role in ("front", "left", "right", "rear")
        }
        rear_safe = self._rear_safe(risks, stale, ir)
        left_block, right_block = self._side_flags(risks, stale, ir)

        # Close-range: spin the nose away first. Reverse only if both sides
        # are blocked. Sitting at PWM 0 against a wall is how the car stops turning.
        if sonic is not None and sonic < self.args.hard_stop_cm:
            pwm, action = self._turn_or_box(
                risks, ir, left_block, right_block, now, "hard_stop_boxed"
            )
            if action != "hard_stop_boxed":
                return pwm, action
            if rear_safe or not bool(ir.get("rear", 0)):
                # Foam-filled rooms keep rear-camera risk high; at ~12 cm the
                # reverse escape must not wait for a clear rear corridor.
                self.reverse_latch = True
                self.reverse_latch_since = now
                return self._reverse()
            return pwm, action

        if stale["front"] and sonic is None:
            return (0, 0, 0, 0), "fault_camera_stale"

        if self.reverse_latch:
            reverse_elapsed = now - self.reverse_latch_since
            sonic_clear = sonic is not None and sonic >= self.args.reverse_clear_cm
            if reverse_elapsed >= self.args.min_reverse_s and sonic_clear:
                self.reverse_latch = False
            elif not rear_safe:
                return self._turn_or_box(
                    risks, ir, left_block, right_block, now, "stop_reverse_blocked"
                )
            else:
                return self._reverse()

        if self.turn_latch is not None:
            turn_elapsed = now - self.turn_latch_since
            sonic_clear = sonic is not None and sonic >= self.args.sonic_clear_cm
            past_stop = sonic is not None and sonic >= self.args.stop_cm
            if (
                turn_elapsed >= self.args.max_turn_s
                and past_stop
            ):
                self._end_turn(now)
            elif turn_elapsed < self.args.min_turn_s or not sonic_clear:
                return self._latched_turn()
            else:
                self._end_turn(now)

        front_ir = (
            self._ir_near(ir, "front", sonic, self.args.front_ir_gate_cm)
            or self._ir_near(ir, "front_left", sonic, self.args.front_side_ir_gate_cm)
            or self._ir_near(ir, "front_right", sonic, self.args.front_side_ir_gate_cm)
        )
        front_block = risks["front"] >= self.args.block_score or front_ir
        front_near = risks["front"] >= self.args.near_score or (
            sonic is not None and sonic < self.args.stop_cm
        )
        sonic_block = sonic is not None and sonic < self.args.stop_cm
        sonic_far = self._sonic_far(sonic)
        # CSI fisheye still fills with a distant foam wall. Ultrasonic owns
        # centimetres: in-place turns only when the nose is actually close.
        nose_close = sonic_block or (front_block and not sonic_far)
        in_cooldown = now < self.turn_cooldown_until
        if nose_close and in_cooldown and not sonic_block:
            nose_close = False

        if nose_close:
            pwm, action = self._turn_or_box(
                risks, ir, left_block, right_block, now, "stop_boxed"
            )
            if action != "stop_boxed":
                return pwm, action
            if front_near and rear_safe:
                return self._reverse()
            return pwm, action

        if left_block and not right_block:
            return self._veer_right()
        if right_block and not left_block:
            return self._veer_left()
        if front_block and sonic_far:
            yaw = 0.0 if self.wall_yaw is None else self.wall_yaw
            if yaw > 8.0 and not left_block:
                pwm, _action = self._veer_left()
                return pwm, "veer_left_yaw"
            if yaw < -8.0 and not right_block:
                pwm, _action = self._veer_right()
                return pwm, "veer_right_yaw"
        if risks["front"] >= self.args.slow_score or (
            sonic is not None and sonic < self.args.creep_cm
        ):
            speed = self._moving_pwm(max(400, self.args.speed // 2))
            return (speed, speed, speed, speed), "slow"
        speed = self._moving_pwm(self.args.speed)
        return (speed, speed, speed, speed), "forward"

    def preflight(self, samples):
        board, highs = board_power_on()
        battery = read_battery_voltage()
        now = time.monotonic()
        ages = {
            role: self._age(samples[role], now)
            for role in ("front", "left", "right", "rear", "gimbal")
        }
        print(
            "Preflight: "
            f"board={'on' if board is True else 'unknown'}(high_pins={highs}) "
            f"battery={battery}V "
            + " ".join(
                f"{role}={'ok' if age <= self.args.camera_stale_s else 'missing'}"
                for role, age in ages.items()
            ),
            flush=True,
        )
        required = ("front", "left", "right", "rear")
        missing = [
            role
            for role in required
            if ages[role] > self.args.camera_stale_s
        ]
        if self.args.arm:
            if self.args.require_board_sense and board is not True:
                raise RuntimeError(
                    "ARM refused: car-board power could not be positively confirmed"
                )
            if board is not True:
                print(
                    "WARNING: car-board power GPIO status is unknown; "
                    "continuing because --require-board-sense was not supplied.",
                    flush=True,
                )
            if battery is None or battery < self.args.min_battery_v:
                raise RuntimeError(
                    f"ARM refused: battery {battery}V < {self.args.min_battery_v:.2f}V"
                )
            if missing:
                raise RuntimeError(f"ARM refused: required cameras missing/stale: {missing}")
        return board, battery

    def _refresh_power(self):
        now = time.monotonic()
        if now - self.last_power_check < 1.0:
            return self.power_ok
        self.last_power_check = now
        board, _highs = board_power_on()
        if board is True:
            self.power_ok = True
        return self.power_ok

    def run(self):
        print("Starting safe surround cruise. Motors are OFF unless --arm is present.")
        print("Ctrl+C always stops PWM and centres the SG90 gimbal.")
        self.start_gimbal()
        self.cameras.start()
        samples = self.cameras.wait_ready(self.args.preflight_s)
        board, battery = self.preflight(samples)
        self.power_ok = board is not False
        self.sensors = SafetySensors(self.args)
        if self.args.arm:
            print(
                f"ARMED: board={'on' if board is True else 'unknown'} "
                f"battery={battery:.2f}V "
                f"({estimate_battery_percent(battery)}%). "
                f"requested speed={self.args.speed}, minimum moving PWM={self.args.motor_min_pwm}. "
                "Starting in 3 seconds."
            )
            for value in (3, 2, 1):
                print(value, flush=True)
                if self.stop_event.wait(1.0):
                    return
            self.motor = Ordinary_Car()
        else:
            print("DRY-RUN: decisions are logged; no motor object was created.")

        frame_i = 0
        period = 1.0 / max(self.args.loop_hz, 1.0)
        while not self.stop_event.is_set():
            started = time.monotonic()
            if self.args.max_frames and frame_i >= self.args.max_frames:
                break
            samples = self.cameras.snapshot()
            self._update_wall_yaw(samples)
            risks = self.camera_risks(samples)
            ir = self.sensors.read_ir()
            raw_sonic = self.sensors.read_sonic()
            filtered_sonic = self.update_gimbal_observation(
                samples, raw_sonic, started
            )
            sonic = self.directional_sonic(filtered_sonic, started)

            control_elapsed = time.monotonic() - started
            if control_elapsed > self.args.loop_watchdog_s:
                pwm, action = (0, 0, 0, 0), "fault_loop_timeout"
                if self.args.arm:
                    self.stop_event.set()
            elif not self.args.no_sonic and sonic is None:
                pwm, action = self.decide(risks, samples, ir, None)
            elif self.args.arm and not self._refresh_power():
                pwm, action = (0, 0, 0, 0), "fault_board_power"
                self.stop_event.set()
            else:
                pwm, action = self.decide(risks, samples, ir, sonic)

            self._advance_peek(action, time.monotonic(), raw_sonic, samples)
            if self.peek_phase in ("go_wall", "go_home", "done"):
                self.cameras.gimbal_live.set()
            else:
                self.cameras.gimbal_live.clear()
            label, pan = self._desired_gimbal_pose(action, time.monotonic())
            if self.peek_phase in ("go_wall", "go_home"):
                pwm = (0, 0, 0, 0)
                action = f"{action}_peek"
            if self.aim_gimbal_pan(pan, time.monotonic(), label):
                action = f"{action}_gimbal_aim"
            # Never drive while the co-mounted ultrasonic is off-centre or moving.
            centred_and_settled = self.servo is None or (
                self.gimbal_label == "centre"
                and started >= self.gimbal_moved_at + self.args.gimbal_settle_s
            )
            if not centred_and_settled:
                pwm = (0, 0, 0, 0)
                action = f"{action}_gimbal_hold"

            # A lower continuous PWM cannot turn these 1:120 motors. Alternate one
            # powered cycle and one coast cycle to make "slow" genuinely slower.
            applied_pwm = pwm
            if action == "slow" and frame_i % 2:
                applied_pwm = (0, 0, 0, 0)
            if self.motor is not None:
                self.motor.set_motor_model(*applied_pwm)

            now = time.monotonic()
            ages = {
                role: self._age(samples[role], now)
                for role in ("front", "left", "right", "rear", "gimbal")
            }
            ir_hits = ",".join(name for name, hit in ir.items() if hit) or "-"
            gimbal_values = "/".join(
                "--" if self.gimbal_scores[name] is None else f"{self.gimbal_scores[name]:.2f}"
                for name in ("left", "centre", "right")
            )
            yaw_txt = "--" if self.wall_yaw is None else f"{self.wall_yaw:+.1f}"
            peek_txt = "--" if self.peek_sonic is None else f"{self.peek_sonic:.0f}"
            print(
                f"risk F/L/R/B={risks['front']:.2f}/{risks['left']:.2f}/"
                f"{risks['right']:.2f}/{risks['rear']:.2f} "
                f"side={int(self.side_blocked['left'])}/{int(self.side_blocked['right'])} "
                f"age={ages['front']:.1f}/{ages['left']:.1f}/"
                f"{ages['right']:.1f}/{ages['rear']:.1f}/{ages['gimbal']:.1f}s "
                f"gimbal={self.gimbal_angle}:{self.gimbal_label}:{gimbal_values} "
                f"yaw={yaw_txt} peek={self.peek_phase}/{peek_txt} "
                f"sonic={sonic}cm(raw={raw_sonic}) ir={ir_hits} "
                f"action={action} pwm={applied_pwm}",
                flush=True,
            )

            if (
                self.args.debug_every > 0
                and frame_i % self.args.debug_every == 0
                and samples["front"].frame is not None
            ):
                front_scores = samples["front"].scores or {}
                debug = draw_debug(samples["front"].frame, [], front_scores, action)
                cv2.imwrite(str(self.debug_dir / f"frame_{frame_i:06d}.jpg"), debug)

            frame_i += 1
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.0, period - elapsed))

        self.stop_motors()

    def close(self):
        self.stop_event.set()
        self.stop_motors()
        if self.servo is not None:
            try:
                self.servo.set_servo_pwm("0", self.gimbal_angles[1])
                self.servo.set_servo_pwm("1", self.args.gimbal_tilt)
            except Exception:
                pass
            try:
                self.servo.pwm_servo.close()
            except Exception:
                pass
            self.servo = None
        if self.motor is not None:
            try:
                self.motor.close()
            except Exception:
                pass
            self.motor = None
        if self.sensors is not None:
            self.sensors.close()
            self.sensors = None
        self.cameras.close()


def main():
    args = parse_args()
    cruise = SafeSurroundCruise(args)
    signal.signal(signal.SIGINT, cruise.request_stop)
    signal.signal(signal.SIGTERM, cruise.request_stop)
    try:
        cruise.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise SystemExit(2) from None
    finally:
        cruise.close()
        print("Stopped; all motor PWM set to zero.", flush=True)


if __name__ == "__main__":
    main()
