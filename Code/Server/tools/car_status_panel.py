"""IMU 檢視器（tools/gy95t_viewer.py）航向圖右邊的小面板：電池即時電量、四個馬達的指令。

電池：車上電路板 ADS7830 第 2 通道（和 tools/battery_monitor.py 相同的換算），每 2 秒在背景執行緒讀 5 次取中位數。
      電路板沒電（電源供應直接接 Pi）時顯示「未接電池」。馬達出力時電壓會下降，看靜止時的讀數比較準。
馬達：這台車的輪子沒有編碼器，量不到實際轉速。這裡顯示的是正在跑的車子程式（巡邏、繞圈）送給馬達驅動板的
      PWM 指令（-4095..+4095，正 = 往前轉），來自地圖區讀到的 control.jsonl 最新一行；沒有車子程式在跑時為 0。
      順序 = motor.set_motor_model：左前、左後、右前、右後。
"""
from __future__ import annotations

import statistics
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from battery_monitor import NO_BATTERY_V, pcb_version, percent, read_battery  # noqa: E402

WHEELS = ("左前", "左後", "右前", "右後")
PWM_MAX = 4095


class CarStatusPanel:
    def __init__(self, parent, width: int, bg: str, fg: str, font, live_row=None, big: bool = False):
        """live_row()：回傳 (正在跑的程式名稱或 None, control.jsonl 最新一行 dict 或 None)。
        big = 大螢幕：字、電量條和馬達長條都放大，和旁邊的航向圖差不多高。"""
        self.w = width
        self.row_h = 34 if big else 20             # 每個馬達一列的高度
        self.bar_h = 16 if big else 10
        font = (font[0], font[1] + (2 if big else 0))
        self.live_row = live_row
        self.frame = tk.Frame(parent, bg=bg)
        tk.Label(self.frame, text="電池", fg="#9a9a9a", bg=bg, font=font).pack(anchor="w")
        self.batt = tk.Label(self.frame, text="讀取中…", fg=fg, bg=bg,
                             font=(font[0], font[1] + (10 if big else 5), "bold"), anchor="w")
        self.batt.pack(anchor="w")
        self.bar = tk.Canvas(self.frame, width=width, height=self.bar_h, bg="#333", highlightthickness=0)
        self.bar.pack(anchor="w", pady=(4 if big else 2, 0))
        tk.Label(self.frame, text="馬達", fg="#9a9a9a", bg=bg, font=font,
                 wraplength=width, justify="left").pack(anchor="w", pady=(18 if big else 10, 4 if big else 2))
        self.motors = tk.Canvas(self.frame, width=width, height=4 * self.row_h, bg=bg, highlightthickness=0)
        self.motors.pack(anchor="w")
        self.note = tk.Label(self.frame, text="", fg="#9a9a9a", bg=bg, font=font, wraplength=width, justify="left")
        self.note.pack(anchor="w", pady=(2, 0))
        self.font = font
        self._shown = {}                           # 上次畫的內容：沒變就不重畫（每次重畫都要合成畫面、VNC 重傳）
        self.volts = None
        self.error = ""
        threading.Thread(target=self._battery_loop, daemon=True).start()
        self.frame.after(300, self._tick)

    # ---- battery (background thread: the I2C read takes ~60 ms) ----------------------------------------
    def _battery_loop(self):
        try:
            import smbus
            bus = smbus.SMBus(1)
            coefficient, scale = (3.3, 3) if pcb_version() == 1 else (5.2, 2)
        except Exception as exc:  # noqa: BLE001  別的電腦沒有 I2C
            self.error = f"讀不到電池（{exc}）"
            return
        while True:
            v = read_battery(bus, coefficient, scale)
            self.volts = v
            time.sleep(2.0)

    # ---- drawing ----------------------------------------------------------------------------------
    def _tick(self):
        try:
            self._draw_battery()
            self._draw_motors()
        finally:
            self.frame.after(200, self._tick)

    def _draw_battery(self):
        v = self.volts
        key = (self.error, None if v is None else round(v, 2))
        if self._shown.get("batt") == key:
            return
        self._shown["batt"] = key
        self.bar.delete("all")
        if self.error:
            self.batt.config(text=self.error[:30], fg="#9a9a9a")
            return
        if v is None or v < NO_BATTERY_V:
            self.batt.config(text="未接電池", fg="#9a9a9a")
            return
        p = percent(v)
        color = "#59d47a" if p >= 50 else ("#ffb84d" if p >= 20 else "#ff5c5c")
        self.batt.config(text=f"{p:3d}%  {v:4.2f} V", fg=color)
        self.bar.create_rectangle(0, 0, self.w * p / 100, self.bar_h, fill=color, outline="")

    def _draw_motors(self):
        name, row = self.live_row() if self.live_row else (None, None)
        pwm = (row or {}).get("pwm") if name else None
        if not pwm or len(pwm) != 4:
            pwm = [0, 0, 0, 0]
        key = (name, tuple(round(v / PWM_MAX * 100) for v in pwm), (row or {}).get("armed"))
        if self._shown.get("motors") == key:
            return
        self._shown["motors"] = key
        c = self.motors
        c.delete("all")
        rh = self.row_h
        label_w, num_w = (46, 62) if rh > 24 else (34, 50)
        x0, x1 = label_w, self.w - num_w
        mid = (x0 + x1) / 2
        top, bot = rh * 0.18, rh * 0.78                     # 長條在這一列的上下位置
        for i, (wheel, v) in enumerate(zip(WHEELS, pwm)):
            y = i * rh
            c.create_text(0, y + rh * 0.48, text=wheel, fill="#e8e8e8", anchor="w", font=self.font)
            c.create_rectangle(x0, y + top, x1, y + bot, fill="#2a2a2a", outline="")
            frac = max(-1.0, min(1.0, v / PWM_MAX))
            if frac:
                c.create_rectangle(mid, y + top, mid + frac * (x1 - x0) / 2, y + bot,
                                   fill="#4aa3ff" if frac > 0 else "#ff7a59", outline="")
            c.create_line(mid, y + top - 2, mid, y + bot + 2, fill="#888")
            c.create_text(self.w, y + rh * 0.48, text=f"{frac * 100:+4.0f}%", fill="#e8e8e8", anchor="e", font=self.font)
        if not name:
            self.note.config(text="沒有車子程式在跑")
        elif row and row.get("armed") is False:
            self.note.config(text=f"{name}（沒有 --arm：指令沒有送出）")
        else:
            self.note.config(text=f"{name}　藍 = 往前、橘 = 往後")
