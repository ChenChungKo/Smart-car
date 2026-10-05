#!/usr/bin/env python3
"""Desktop GUI for full_car_test.py — tests, live camera preview, emergency stop."""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from camera_devices import (
    csi_array_to_bgr,
    create_csi_still_configuration,
    resolve_usb_capture_index,
)
from full_car_test import MENU_ITEMS, FullCarTester, build_parser, estimate_battery_percent

SERVER = Path(__file__).resolve().parent
HW_PATH = SERVER / "camera_hardware.json"

# Tests that open cameras — preview must release first.
CAMERA_TESTS = {
    "camera",
    "cameras",
    "cameras-labeled",
    "plane-map",
    "system",
    "all",
}

PREVIEW_CAMERAS = [
    ("front", "車頭 CSI1"),
    ("left", "左側 USB"),
    ("right", "右側 USB"),
    ("rear", "車尾 CSI0"),
    ("gimbal", "雲台 USB"),
]


class _QueueWriter:
    def __init__(self, log_q: queue.Queue):
        self.log_q = log_q

    def write(self, text: str) -> int:
        if text:
            self.log_q.put(("log", text))
        return len(text) if text else 0

    def flush(self) -> None:
        pass


def load_hardware() -> dict:
    with HW_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _pcb_version() -> int:
    try:
        with (SERVER / "params.json").open(encoding="utf-8") as f:
            return int(json.load(f).get("Pcb_Version", 2))
    except Exception:
        return 2


BOARD_SENSE_PINS = (14, 15, 23, 26, 20, 19, 16, 6, 12)


def probe_power_switches() -> dict:
    """RPi is on if this GUI runs. Board/motor switch is NOT the battery ADC.

    Freenove battery sense stays live whenever the pack is plugged in. The motor
    switch only feeds shield 5V to IR modules; those GPIOs sit LOW when unpowered.
    """
    import smbus

    voltage = None
    pcb = _pcb_version()
    coeff = 3.3 if pcb == 1 else 5.2
    scale = 3 if pcb == 1 else 2
    bus = None
    try:
        bus = smbus.SMBus(1)
        channel = 2
        command = 0x84 | ((((channel << 2) | (channel >> 1)) & 0x07) << 4)
        bus.write_byte(0x48, command)
        bus.read_byte(0x48)
        raw = bus.read_byte(0x48)
        voltage = round(raw / 255.0 * coeff * scale, 2)
    except Exception:
        voltage = None
    finally:
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass

    high = 0
    try:
        import subprocess

        proc = subprocess.run(
            ["pinctrl", "get", ",".join(str(p) for p in BOARD_SENSE_PINS)],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        for line in proc.stdout.splitlines():
            if "| hi" in line:
                high += 1
    except Exception:
        high = 0

    return {
        "pi": True,
        "board": high > 0,
        "ir_high": high,
        "voltage": voltage,
    }


class CameraPreview:
    """One live camera feed for the GUI (CSI or USB)."""

    def __init__(self, frame_q: queue.Queue, log_fn, width=640, height=480):
        self.frame_q = frame_q
        self.log_fn = log_fn
        self.width = width
        self.height = height
        self.hw = load_hardware()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._csi = None
        self._cap = None
        self._last_usb = None
        self.role: str | None = None
        self._usb_interval = 0.12

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, role: str) -> None:
        self.stop()
        self.role = role
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(role,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None
        self._close_devices()
        self.role = None

    def _close_devices(self) -> None:
        if self._csi is not None:
            try:
                self._csi.stop()
            except Exception:
                pass
            try:
                self._csi.close()
            except Exception:
                pass
            self._csi = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._last_usb = None

    def _open(self, role: str) -> None:
        entry = self.hw.get(role)
        if entry is None:
            raise RuntimeError(f"camera_hardware.json 沒有 {role}")
        iface = entry.get("interface", "")
        if iface == "csi":
            from picamera2 import Picamera2

            num = 1 if role == "front" else 0
            device = entry.get("device", "")
            if "camera_num=" in device:
                num = int(device.split("camera_num=")[1].split()[0].rstrip(",)"))
            # Match full_car_test single shots: still + 640x480 full-FOV raw.
            w, h = 640, 480
            cam = Picamera2(camera_num=num)
            cam.configure(create_csi_still_configuration(cam, w, h))
            cam.start()
            try:
                cam.set_controls({"AeEnable": True, "AwbEnable": True})
            except Exception:
                pass
            time.sleep(1.0)
            for _ in range(12):
                cam.capture_array()
                time.sleep(0.05)
            self._csi = cam
            self.log_fn(f"預覽開啟：{role} CSI camera_num={num} {w}x{h} still/full-FOV\n")
        else:
            idx, how = resolve_usb_capture_index(entry)
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"無法開啟 {role} /dev/video{idx}")
            # FourCC before size. Native 640x480 on these S10 cams crops and tears;
            # 1080p YUYV then downscale matches the stable left single-shot path.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 8)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            got = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            time.sleep(0.6)
            for _ in range(8):
                cap.grab()
            self._cap = cap
            self._last_usb = None
            self.log_fn(f"預覽開啟：{role} /dev/video{idx} {got[0]}x{got[1]} YUYV 8fps ({how})\n")

    @staticmethod
    def _usb_torn(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        row = np.abs(gray[1:] - gray[:-1]).mean(axis=1)
        if row.size < 8:
            return False
        bottom = float(row[-40:].max()) if row.size >= 40 else float(row.max())
        return bottom > 42.0 or float(row.max()) > 58.0

    def _read(self) -> np.ndarray | None:
        if self._csi is not None:
            return csi_array_to_bgr(self._csi.capture_array())
        if self._cap is not None:
            # One retrieve only — extra reads on left USB tear the YUYV stream.
            if not self._cap.grab():
                return self._last_usb
            ok, frame = self._cap.retrieve()
            if not ok or frame is None:
                return self._last_usb
            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if self._usb_torn(frame) and self._last_usb is not None:
                return self._last_usb
            self._last_usb = frame
            return frame
        return None

    def _loop(self, role: str) -> None:
        try:
            self._open(role)
        except Exception as exc:
            self.frame_q.put(("preview_err", str(exc)))
            self._close_devices()
            return
        interval = self._usb_interval if self._cap is not None else 0.04
        while not self._stop.is_set():
            frame = self._read()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                while True:
                    self.frame_q.get_nowait()
            except queue.Empty:
                pass
            self.frame_q.put(("frame", frame.copy() if self._cap is not None else frame))
            time.sleep(interval)
        self._close_devices()


class FullCarTestGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Freenove 全車測試")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.log_q: queue.Queue = queue.Queue()
        self.confirm_q: queue.Queue = queue.Queue()
        self.confirm_result_q: queue.Queue = queue.Queue()
        self.preview_q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.tester: FullCarTester | None = None
        self.running = False
        self._stop_flag = threading.Event()
        self._photo = None
        self._last_test_key = ""
        self.preview = CameraPreview(self.preview_q, self._append_log_safe)
        self._closing = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._pump_queues)
        self.root.after(200, self._refresh_power)

    def _append_log_safe(self, text: str) -> None:
        self.log_q.put(("log", text))

    def _style_power_pill(self, label: tk.Label, on: bool) -> None:
        if on:
            label.configure(bg="#1e8449", fg="white")
        else:
            label.configure(bg="#c0392b", fg="white")

    def _refresh_power(self) -> None:
        if self._closing:
            return
        status = probe_power_switches()
        self._style_power_pill(self.pi_pill, True)
        self.pi_pill.configure(text="  RPi 電源：開  ")
        volts = status["voltage"]
        if volts is not None:
            pct = estimate_battery_percent(volts)
            bat = f"  電池 {volts:.2f}V  {pct}%  "
        else:
            bat = "  電池 --  "
        self.battery_lbl.configure(text=bat)
        if status["board"]:
            self.board_pill.configure(text="  車板電源：開  ")
            self._style_power_pill(self.board_pill, True)
            self.power_hint_var.set("兩個開關都開，可以移動")
        else:
            self.board_pill.configure(text="  車板電源：關  ")
            self._style_power_pill(self.board_pill, False)
            self.power_hint_var.set("車子不會動：請打開車板上的電源開關")
        self.root.after(1500, self._refresh_power)

    def _build_ui(self) -> None:
        power_row = ttk.Frame(self.root, padding=(8, 8, 8, 0))
        power_row.pack(fill=tk.X)
        ttk.Label(power_row, text="開關").pack(side=tk.LEFT, padx=(0, 8))
        self.pi_pill = tk.Label(
            power_row,
            text="  RPi 電源：偵測中  ",
            font=("Sans", 10, "bold"),
            bg="#7f8c8d",
            fg="white",
            padx=6,
            pady=3,
        )
        self.pi_pill.pack(side=tk.LEFT, padx=(0, 8))
        self.board_pill = tk.Label(
            power_row,
            text="  車板電源：偵測中  ",
            font=("Sans", 10, "bold"),
            bg="#7f8c8d",
            fg="white",
            padx=6,
            pady=3,
        )
        self.board_pill.pack(side=tk.LEFT, padx=(0, 8))
        self.battery_lbl = ttk.Label(power_row, text="  電池 --  ")
        self.battery_lbl.pack(side=tk.LEFT, padx=(0, 8))
        self.power_hint_var = tk.StringVar(value="")
        ttk.Label(power_row, textvariable=self.power_hint_var).pack(side=tk.LEFT)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="參數").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=900)
        self.turn_var = tk.IntVar(value=900)
        self.duration_var = tk.DoubleVar(value=1.5)
        ttk.Label(top, text="速度").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(top, textvariable=self.speed_var, width=6).pack(side=tk.LEFT)
        ttk.Label(top, text="轉速").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(top, textvariable=self.turn_var, width=6).pack(side=tk.LEFT)
        ttk.Label(top, text="秒數").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(top, textvariable=self.duration_var, width=6).pack(side=tk.LEFT)

        self.skip_confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="跳過確認（--yes）", variable=self.skip_confirm_var).pack(
            side=tk.LEFT, padx=(16, 0)
        )

        self.stop_btn = tk.Button(
            top,
            text="急停 STOP",
            fg="white",
            bg="#c0392b",
            activebackground="#e74c3c",
            font=("Sans", 11, "bold"),
            command=self._emergency_stop,
            width=12,
        )
        self.stop_btn.pack(side=tk.RIGHT)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=4)
        body.add(left, weight=1)
        body.add(right, weight=3)

        ttk.Label(left, text="測試項目（點一項只跑該項）").pack(anchor=tk.W)
        hint = ttk.Label(left, text="燈號：灰=未測  黃=執行中  綠=通過  紅=失敗", font=("Sans", 8))
        hint.pack(anchor=tk.W, pady=(0, 4))

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.BOTH, expand=True)

        self.test_buttons: list[ttk.Button] = []
        self.status_lights: dict[str, tk.Canvas] = {}
        self.test_results: dict[str, str] = {}  # pending|running|pass|fail|skip
        for key, label in MENU_ITEMS:
            row = ttk.Frame(btn_frame)
            row.pack(fill=tk.X, pady=1)
            light = tk.Canvas(row, width=14, height=14, highlightthickness=0)
            light.pack(side=tk.LEFT, padx=(0, 6))
            light.create_oval(2, 2, 12, 12, fill="#888888", outline="#555555", tags="dot")
            self.status_lights[key] = light
            self.test_results[key] = "pending"
            btn = ttk.Button(
                row,
                text=label,
                command=lambda k=key, n=label: self._start_test(k, n),
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.test_buttons.append(btn)

        clear_row = ttk.Frame(left)
        clear_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(clear_row, text="清除燈號", command=self._clear_lights).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="就緒 — 點左側任一項目即可單獨測試")
        ttk.Label(left, textvariable=self.status_var, wraplength=220).pack(anchor=tk.W, pady=(8, 0))

        # Right: preview on top, log below
        right_split = ttk.Panedwindow(right, orient=tk.VERTICAL)
        right_split.pack(fill=tk.BOTH, expand=True)

        preview_box = ttk.Frame(right_split, padding=4)
        log_box = ttk.Frame(right_split, padding=4)
        right_split.add(preview_box, weight=3)
        right_split.add(log_box, weight=2)

        bar = ttk.Frame(preview_box)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="相機預覽").pack(side=tk.LEFT)
        self.cam_var = tk.StringVar(value="front")
        cam_names = [f"{k} — {label}" for k, label in PREVIEW_CAMERAS]
        self.cam_combo = ttk.Combobox(
            bar,
            values=cam_names,
            state="readonly",
            width=22,
        )
        self.cam_combo.current(0)
        self.cam_combo.pack(side=tk.LEFT, padx=8)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_cam_selected)

        self.preview_btn = ttk.Button(bar, text="開始預覽", command=self._toggle_preview)
        self.preview_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="停止預覽", command=self._stop_preview).pack(side=tk.LEFT)

        self.preview_status = tk.StringVar(value="預覽未啟動")
        ttk.Label(bar, textvariable=self.preview_status).pack(side=tk.RIGHT)

        self.video_label = tk.Label(
            preview_box,
            text="選擇相機後按「開始預覽」",
            bg="#1a1a1a",
            fg="#aaaaaa",
            width=64,
            height=18,
            relief=tk.SUNKEN,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        ttk.Label(log_box, text="執行日誌").pack(anchor=tk.W)
        self.log = scrolledtext.ScrolledText(log_box, wrap=tk.WORD, height=12, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _selected_role(self) -> str:
        text = self.cam_combo.get()
        return text.split(" — ", 1)[0].strip() if text else "front"

    def _on_cam_selected(self, _event=None) -> None:
        if self.preview.active:
            self._start_preview()

    def _toggle_preview(self) -> None:
        if self.preview.active:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self) -> None:
        if self.running and self._last_test_key in CAMERA_TESTS:
            messagebox.showinfo("忙碌中", "相機測試進行中，請先等測試結束再預覽。")
            return
        role = self._selected_role()
        self.preview_status.set(f"開啟中：{role}…")
        self.preview_btn.configure(text="停止預覽")
        self.preview.start(role)

    def _stop_preview(self) -> None:
        self.preview.stop()
        self.preview_btn.configure(text="開始預覽")
        self.preview_status.set("預覽未啟動")
        self.video_label.configure(image="", text="選擇相機後按「開始預覽」")
        self._photo = None

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w, max_h = 640, 400
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 0.99:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self._photo, text="")

    def _set_light(self, key: str, state: str) -> None:
        colors = {
            "pending": ("#888888", "#555555"),
            "running": ("#f1c40f", "#b7950b"),
            "pass": ("#2ecc71", "#1e8449"),
            "fail": ("#e74c3c", "#922b21"),
            "skip": ("#95a5a6", "#7f8c8d"),
        }
        fill, outline = colors.get(state, colors["pending"])
        light = self.status_lights.get(key)
        if light is None:
            return
        light.itemconfigure("dot", fill=fill, outline=outline)
        self.test_results[key] = state

    def _clear_lights(self) -> None:
        for key in self.status_lights:
            self._set_light(key, "pending")
        self.status_var.set("燈號已清除")

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool) -> None:
        self.running = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in self.test_buttons:
            btn.configure(state=state)

    def _make_args(self):
        parser = build_parser()
        args = parser.parse_args([])
        args.speed = int(self.speed_var.get())
        args.turn_speed = int(self.turn_var.get())
        args.duration = float(self.duration_var.get())
        args.yes = bool(self.skip_confirm_var.get())
        return args

    def _gui_confirm(self, message: str) -> bool:
        self.confirm_q.put(message)
        try:
            return bool(self.confirm_result_q.get(timeout=600))
        except queue.Empty:
            return False

    def _start_test(self, key: str, label: str) -> None:
        if self.running:
            messagebox.showinfo("忙碌中", "目前已有測試在執行。")
            return
        self._last_test_key = key
        # Camera hardware is exclusive — release preview first.
        if key in CAMERA_TESTS and self.preview.active:
            self._append_log("相機測試前先關閉預覽…\n")
            self._stop_preview()
            time.sleep(0.4)

        self._stop_flag.clear()
        self._set_busy(True)
        self._set_light(key, "running")
        self.status_var.set(f"執行中：{label}")
        self._append_log(f"\n===== 開始：{label} ({key}) =====\n")

        args = self._make_args()
        self.tester = FullCarTester(args, confirm_fn=None if args.yes else self._gui_confirm)

        def worker():
            old_out, old_err = sys.stdout, sys.stderr
            writer = _QueueWriter(self.log_q)
            sys.stdout = writer  # type: ignore[assignment]
            sys.stderr = writer  # type: ignore[assignment]
            result = "fail"
            try:
                if self._stop_flag.is_set():
                    print("已取消。")
                    result = "skip"
                else:
                    self.tester.run_named_test(key)
                    if self._stop_flag.is_set():
                        print("測試中被急停。")
                        result = "fail"
                    elif self.tester.was_skipped:
                        print(f"已跳過：{label}")
                        result = "skip"
                    else:
                        print(f"通過：{label}")
                        result = "pass"
            except Exception as exc:
                print(f"失敗：{exc}")
                result = "fail"
            finally:
                try:
                    self.tester.stop_motion()
                except Exception:
                    pass
                try:
                    if key == "led":
                        self.tester.turn_off_leds()
                except Exception:
                    pass
                sys.stdout, sys.stderr = old_out, old_err
                self.log_q.put(("done", {"key": key, "label": label, "result": result}))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _emergency_stop(self) -> None:
        self._stop_flag.set()
        self._append_log("\n*** 急停 ***\n")
        if self.tester is not None:
            try:
                self.tester.stop_motion()
            except Exception as exc:
                self._append_log(f"急停失敗：{exc}\n")
        if self._last_test_key:
            self._set_light(self._last_test_key, "fail")
        try:
            self.confirm_result_q.put_nowait(False)
        except Exception:
            pass
        self.status_var.set("已急停")

    def _pump_queues(self) -> None:
        try:
            while True:
                kind, payload = self.log_q.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._set_busy(False)
                    key = payload["key"]
                    label = payload["label"]
                    result = payload["result"]
                    self._set_light(key, result)
                    if result == "pass":
                        self.status_var.set(f"通過：{label}")
                    elif result == "skip":
                        self.status_var.set(f"已跳過：{label}")
                    else:
                        self.status_var.set(f"失敗：{label}")
        except queue.Empty:
            pass

        try:
            while True:
                message = self.confirm_q.get_nowait()
                ok = messagebox.askyesno("確認", message, parent=self.root)
                self.confirm_result_q.put(bool(ok))
        except queue.Empty:
            pass

        try:
            while True:
                kind, payload = self.preview_q.get_nowait()
                if kind == "frame":
                    self._show_frame(payload)
                    if self.preview.role:
                        self.preview_status.set(f"預覽中：{self.preview.role}")
                elif kind == "preview_err":
                    self.preview_btn.configure(text="開始預覽")
                    self.preview_status.set("預覽失敗")
                    self._append_log(f"預覽錯誤：{payload}\n")
                    messagebox.showerror("預覽失敗", payload, parent=self.root)
        except queue.Empty:
            pass

        self.root.after(40, self._pump_queues)

    def _on_close(self) -> None:
        self._closing = True
        self._stop_preview()
        self._emergency_stop()
        if self.tester is not None:
            try:
                self.tester.cleanup()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    FullCarTestGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
