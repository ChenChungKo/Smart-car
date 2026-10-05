#!/usr/bin/env python3
"""Camera-based cruise: free-space / near-surface vision on CSI1 (+ gimbal/sides).

Primary sensing is the cameras (not ultrasonic-only). Algorithm is geometric
free-space on fisheye frames; YOLO is optional extras for known objects.
Ctrl+C stops motors.
Prefer: Code/Server/.venv/bin/python vision_cruise.py
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from camera_devices import create_csi_still_configuration, csi_array_to_bgr, resolve_usb_capture_index
from motor import Ordinary_Car
from servo import Servo
from vision_detector import (
    YoloOnnxDetector,
    classic_corridor_score,
    detections_to_lane_scores,
    draw_debug,
    default_model_path,
)

SERVER = Path(__file__).resolve().parent
HARDWARE = SERVER / "camera_hardware.json"


class _MjpegBroker:
    """Latest JPEG frame shared with a tiny HTTP MJPEG server."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._event = threading.Event()

    def publish(self, bgr) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
        self._event.set()

    def wait_jpeg(self, timeout: float = 1.0):
        if not self._event.wait(timeout):
            return None
        self._event.clear()
        with self._lock:
            return self._jpeg


def _start_mjpeg_server(broker: _MjpegBroker, port: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (
                    b"<html><head><title>front CSI1</title></head>"
                    b"<body style='margin:0;background:#111;color:#eee;font-family:sans-serif'>"
                    b"<h3 style='margin:8px'>Front camera (CSI1)</h3>"
                    b"<img src='/stream.mjpg' style='max-width:100%;height:auto'/>"
                    b"</body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path not in ("/stream.mjpg", "/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = broker.wait_jpeg(1.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def parse_args():
    p = argparse.ArgumentParser(description="Camera free-space cruise (front + left + right).")
    p.add_argument("--speed", type=int, default=950, help="Forward PWM (slow; raise if wheels barely move).")
    p.add_argument("--turn-speed", type=int, default=800, help="Turn PWM (keep modest).")
    p.add_argument("--back-speed", type=int, default=850, help="Reverse PWM.")
    p.add_argument("--avoid-hold-s", type=float, default=0.28)
    p.add_argument("--block-score", type=float, default=0.36, help="Lane score treated as blocked (turn/slow).")
    p.add_argument("--slow-score", type=float, default=0.20, help="Lane score treated as slow.")
    p.add_argument(
        "--reverse-score",
        type=float,
        default=0.70,
        help="Only reverse when mid score >= this (avoids far-wall fwd/back chatter).",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--model", default=str(default_model_path()))
    p.add_argument(
        "--use-yolo",
        action="store_true",
        help="Also run YOLO for known objects (person/chair...). Off by default; cameras still used.",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--side-interval-s",
        type=float,
        default=0.35,
        help="Left/right USB poll interval (3-camera mode).",
    )
    p.add_argument("--scan-angles", default="30,90,150", help="Gimbal pan L/M/R (only with --gimbal).")
    p.add_argument("--scan-settle-s", type=float, default=0.25)
    p.add_argument("--tilt-level", type=int, default=90)
    # Default: no SG90 gimbal; use fixed front + left + right only.
    p.set_defaults(no_gimbal=True, use_ir=False, sonic_backup=False)
    p.add_argument("--gimbal", action="store_false", dest="no_gimbal", help="Enable SG90 gimbal camera scan.")
    p.add_argument("--no-gimbal", action="store_true", dest="no_gimbal", help="Disable SG90 (default).")
    p.add_argument("--no-sides", action="store_true", help="Disable left/right USB cameras.")
    p.add_argument("--no-classic-backup", action="store_true")
    p.add_argument(
        "--sonic-backup",
        action="store_true",
        dest="sonic_backup",
        help="Enable ultrasonic hard-stop (off by default in 3-cam mode).",
    )
    p.add_argument("--no-sonic-backup", action="store_false", dest="sonic_backup")
    p.add_argument("--stop-cm", type=float, default=35.0)
    p.add_argument(
        "--use-ir",
        action="store_true",
        dest="use_ir",
        help="Enable IR near-field guards (off by default; pins can false-trigger).",
    )
    p.add_argument("--no-ir", action="store_false", dest="use_ir")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--preview",
        action="store_true",
        default=True,
        help="Show live front-camera window with overlay (default on).",
    )
    p.add_argument("--no-preview", action="store_false", dest="preview")
    p.add_argument(
        "--stream-port",
        type=int,
        default=8765,
        help="MJPEG HTTP port for live front view in a browser (0=off). Default 8765.",
    )
    p.add_argument("--debug-dir", default="vision_cruise_debug")
    p.add_argument("--debug-every", type=int, default=5, help="Save overlay every N frames (0=off).")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=forever).")
    p.add_argument("--csi-num", type=int, default=1, help="Fixed front CSI index.")
    return p.parse_args()


def load_hardware():
    with open(HARDWARE, encoding="utf-8") as f:
        return json.load(f)


class VisionCruise:
    def __init__(self, args):
        self.args = args
        self.hw = load_hardware()
        self.motor = None
        if args.dry_run:
            print("*** dry_run=True: motors will NOT move ***")
        else:
            try:
                self.motor = Ordinary_Car()
                print("Motors ready (PCA9685).")
            except Exception as exc:
                raise SystemExit(f"Motor init failed: {exc}") from exc
        self.servo = None
        if not args.no_gimbal:
            try:
                self.servo = Servo()
            except Exception as exc:
                print(f"WARNING: servo/gimbal unavailable ({exc}); continuing without gimbal.")
                args.no_gimbal = True
        else:
            print("SG90 gimbal disabled; using front + left + right cameras only.")
        self.sonic = None
        if args.sonic_backup:
            from ultrasonic import Ultrasonic

            self.sonic = Ultrasonic()
        self.ir = {}
        if args.use_ir:
            from gpiozero import DigitalInputDevice

            for name, pin in (("left", 16), ("right", 6), ("front", 20)):
                self.ir[name] = DigitalInputDevice(pin, pull_up=False)

        self.detector = None
        if args.use_yolo:
            self.detector = YoloOnnxDetector(
                model_path=args.model, conf=args.conf, imgsz=args.imgsz
            )
            print(f"YOLO enabled: {args.model}")
        else:
            print("Camera free-space mode (YOLO off).")
        self.scan_angles = [int(v.strip()) for v in args.scan_angles.split(",") if v.strip()]
        if len(self.scan_angles) != 3:
            raise ValueError("--scan-angles needs left,mid,right")
        self.scan_index = 1
        self.scan_dir = 1
        self.gimbal_scores = {"left": 0.0, "mid": 0.0, "right": 0.0}
        self.side_scores = {"left": 0.0, "right": 0.0}
        self.side_blocked = {"left": False, "right": False}
        self.side_next_t = 0.0
        self.smooth_scores = {"left": 0.0, "mid": 0.0, "right": 0.0}
        self.smooth_ready = False
        self.hold_until = 0.0
        self.hold_pwm = (0, 0, 0, 0)
        self.hold_action = "forward"
        self.frame_i = 0
        self.debug_dir = SERVER / args.debug_dir
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.preview_ok = False
        self.mjpeg = None
        self.httpd = None
        if args.stream_port and args.stream_port > 0:
            self.mjpeg = _MjpegBroker()
            self.httpd = _start_mjpeg_server(self.mjpeg, args.stream_port)
            print(f"Live front stream: http://<pi-ip>:{args.stream_port}/")

        self.csi = None
        self._open_csi()
        self.usb_caps = {}
        if not args.no_sides:
            self._open_usb_side("left")
            self._open_usb_side("right")
        if not args.no_gimbal:
            self._open_usb_side("gimbal")
            if self.servo is not None:
                self.servo.set_servo_pwm("0", self.scan_angles[1])
                self.servo.set_servo_pwm("1", args.tilt_level)
                time.sleep(0.2)

        if args.preview:
            try:
                cv2.namedWindow("front_csi1", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("front_csi1", 640, 480)
                self.preview_ok = True
                print("Live preview window: front_csi1")
            except Exception as exc:
                print(f"WARNING: cannot open preview window ({exc}). Use browser stream instead.")

    def _open_csi(self):
        from picamera2 import Picamera2

        cam = Picamera2(self.args.csi_num)
        cfg = create_csi_still_configuration(cam, self.args.width, self.args.height)
        # Prefer video config for continuous frames if available.
        try:
            cfg = cam.create_video_configuration(
                main={"size": (self.args.width, self.args.height), "format": "RGB888"}
            )
        except Exception:
            pass
        cam.configure(cfg)
        cam.start()
        cam.set_controls({"AeEnable": True, "AwbEnable": True})
        time.sleep(0.8)
        self.csi = cam

    def _open_usb_side(self, key: str):
        entry = self.hw[key]
        idx, how = resolve_usb_capture_index(entry)
        print(f"{key}: /dev/video{idx} ({how})")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"WARNING: cannot open {key} video{idx}")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.usb_caps[key] = cap

    def _read_usb(self, key: str):
        cap = self.usb_caps.get(key)
        if cap is None:
            return None
        ok, frame = False, None
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        return frame if ok else None

    def _read_csi(self):
        return csi_array_to_bgr(self.csi.capture_array())

    def _score_frame(self, bgr) -> tuple[dict[str, float], list]:
        # Cameras always drive free-space scoring; YOLO only adds known-object weight.
        classic = classic_corridor_score(bgr)
        dets = []
        if self.detector is not None:
            dets = self.detector.detect(bgr)
            scores = detections_to_lane_scores(
                dets, bgr.shape[1], bgr.shape[0], classic, classic_weight=1.0
            )
        else:
            scores = {"left": classic["left"], "mid": classic["mid"], "right": classic["right"]}
        return scores, dets

    def _step_gimbal(self):
        if self.args.no_gimbal or self.servo is None:
            return
        angle = self.scan_angles[self.scan_index]
        self.servo.set_servo_pwm("0", angle)
        time.sleep(self.args.scan_settle_s)
        frame = self._read_usb("gimbal")
        if frame is not None:
            scores, _ = self._score_frame(frame)
            lane = ("left", "mid", "right")[self.scan_index]
            self.gimbal_scores[lane] = scores["mid"] * 0.7 + max(scores["left"], scores["right"]) * 0.3
        if self.scan_index <= 0:
            self.scan_dir = 1
        elif self.scan_index >= 2:
            self.scan_dir = -1
        self.scan_index += self.scan_dir

    def _poll_sides(self):
        if self.args.no_sides:
            return
        now = time.time()
        if now < self.side_next_t:
            return
        self.side_next_t = now + self.args.side_interval_s
        for side in ("left", "right"):
            frame = self._read_usb(side)
            if frame is None:
                continue
            scores, _ = self._score_frame(frame)
            self.side_scores[side] = scores["mid"]
            block = scores["mid"] >= self.args.block_score or (
                scores["left"] + scores["right"] + scores["mid"]
            ) >= self.args.block_score * 1.4
            self.side_blocked[side] = bool(block)

    def _ir_blocked(self, name: str) -> bool:
        dev = self.ir.get(name)
        if dev is None:
            return False
        # Default active-low obstacle modules: LOW = obstacle.
        return int(dev.value) == 0

    def _pwm_turn_right(self):
        mag = self.args.turn_speed
        inner = -max(150, mag // 4)
        return (mag, mag, inner, inner)

    def _pwm_turn_left(self):
        mag = self.args.turn_speed
        inner = -max(150, mag // 4)
        return (inner, inner, mag, mag)

    def _smooth_front(self, front_scores: dict[str, float]) -> dict[str, float]:
        alpha = 0.45
        if not self.smooth_ready:
            self.smooth_scores = dict(front_scores)
            self.smooth_ready = True
            return dict(self.smooth_scores)
        for k in ("left", "mid", "right"):
            self.smooth_scores[k] = alpha * front_scores[k] + (1.0 - alpha) * self.smooth_scores[k]
        return dict(self.smooth_scores)

    def decide(self, front_scores: dict[str, float]):
        speed = self.args.speed
        back = self.args.back_speed
        thr = self.args.block_score
        slow_thr = self.args.slow_score
        rev_thr = self.args.reverse_score
        front_scores = self._smooth_front(front_scores)

        # 3-camera fuse: front CSI + left/right USB (no gimbal).
        fused = {
            "left": front_scores["left"] * 0.55 + self.side_scores["left"] * 0.45,
            "mid": front_scores["mid"],
            "right": front_scores["right"] * 0.55 + self.side_scores["right"] * 0.45,
        }
        if not self.args.no_gimbal:
            fused["left"] = fused["left"] * 0.85 + self.gimbal_scores["left"] * 0.15
            fused["mid"] = fused["mid"] * 0.85 + self.gimbal_scores["mid"] * 0.15
            fused["right"] = fused["right"] * 0.85 + self.gimbal_scores["right"] * 0.15

        # Side USB only for veer / preference — not enough alone to force reverse.
        left_block = fused["left"] >= thr or self._ir_blocked("left")
        right_block = fused["right"] >= thr or self._ir_blocked("right")
        left_side = self.side_blocked["left"]
        right_side = self.side_blocked["right"]
        mid_block = fused["mid"] >= thr or self._ir_blocked("front")
        mid_slow = fused["mid"] >= slow_thr
        mid_near = fused["mid"] >= rev_thr

        if self.sonic is not None:
            cm = self.sonic.get_distance()
            if cm is not None and 1.0 <= cm < self.args.stop_cm:
                mid_block = True
                mid_near = True

        if mid_block:
            # Prefer turning toward the clearer side; reverse only when mid is near.
            if left_block and not right_block:
                return self._pwm_turn_right(), "turn_right"
            if right_block and not left_block:
                return self._pwm_turn_left(), "turn_left"
            if left_side and not right_side:
                return self._pwm_turn_right(), "turn_right"
            if right_side and not left_side:
                return self._pwm_turn_left(), "turn_left"
            if mid_near and left_block and right_block:
                return (-back, -back, -back, -back), "reverse"
            if fused["left"] <= fused["right"]:
                return self._pwm_turn_left(), "turn_left"
            return self._pwm_turn_right(), "turn_right"

        if (left_block or left_side) and not (right_block or right_side):
            return self._pwm_turn_right(), "veer_right"
        if (right_block or right_side) and not (left_block or left_side):
            return self._pwm_turn_left(), "veer_left"
        if mid_slow:
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

    def drive(self, fl, bl, fr, br):
        if self.motor is None:
            return
        self.motor.set_motor_model(fl, bl, fr, br)

    def close(self):
        try:
            self.drive(0, 0, 0, 0)
        except Exception:
            pass
        if self.motor is not None:
            try:
                self.motor.close()
            except Exception:
                pass
        if self.servo is not None:
            try:
                self.servo.set_servo_pwm("0", 90)
                self.servo.set_servo_pwm("1", 90)
            except Exception:
                pass
        if self.csi is not None:
            try:
                self.csi.close()
            except Exception:
                pass
        for cap in self.usb_caps.values():
            try:
                cap.release()
            except Exception:
                pass
        if self.sonic is not None:
            try:
                self.sonic.close()
            except Exception:
                pass
        for dev in self.ir.values():
            try:
                dev.close()
            except Exception:
                pass
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
        if self.preview_ok:
            try:
                cv2.destroyWindow("front_csi1")
            except Exception:
                pass

    def run(self):
        print("Camera cruise: front + left + right (no SG90). Ctrl+C to stop.")
        print(f"dry_run={self.args.dry_run} motor={'on' if self.motor else 'off'} "
              f"yolo={self.detector is not None} gimbal={not self.args.no_gimbal} "
              f"sides={not self.args.no_sides} ir={self.args.use_ir} sonic={self.args.sonic_backup}")
        print(f"speed={self.args.speed} turn={self.args.turn_speed}")
        gimbal_every = 4
        try:
            while True:
                if self.args.max_frames and self.frame_i >= self.args.max_frames:
                    print("Reached --max-frames.")
                    break
                frame = self._read_csi()
                if frame is None:
                    time.sleep(0.05)
                    continue
                scores, dets = self._score_frame(frame)
                if (not self.args.no_gimbal) and self.frame_i % gimbal_every == 0:
                    self._step_gimbal()
                self._poll_sides()
                pwm, action = self.apply_hold(*self.decide(scores))
                side_txt = (
                    f"sideL={self.side_scores['left']:.2f}/{int(self.side_blocked['left'])} "
                    f"sideR={self.side_scores['right']:.2f}/{int(self.side_blocked['right'])}"
                )
                print(
                    f"fL/M/R={scores['left']:.2f}/{scores['mid']:.2f}/{scores['right']:.2f} "
                    f"{side_txt} pwm={pwm[0]} {action}",
                    flush=True,
                )
                overlay = draw_debug(frame, dets, scores, action)
                if self.args.debug_every > 0 and self.frame_i % self.args.debug_every == 0:
                    path = self.debug_dir / f"frame_{self.frame_i:05d}.jpg"
                    cv2.imwrite(str(path), overlay)
                if self.mjpeg is not None:
                    self.mjpeg.publish(overlay)
                if self.preview_ok:
                    cv2.imshow("front_csi1", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("Preview closed (q/ESC).")
                        break
                self.drive(*pwm)
                self.frame_i += 1
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            self.close()


def main():
    args = parse_args()
    VisionCruise(args).run()


if __name__ == "__main__":
    main()
