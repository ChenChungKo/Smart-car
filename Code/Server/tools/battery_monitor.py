#!/usr/bin/env python3
"""電池即時監控小視窗（Pi 桌面）：電池電壓與估計電量、Pi 的 5 V 供電、供電不足紀錄、最近 10 分鐘的電壓曲線。

    python3 tools/battery_monitor.py            # 每 2 秒更新，視窗固定在最上層、右上角
    python3 tools/battery_monitor.py --every 5

電池電壓：車上電路板的 ADS7830（I2C 0x48）第 2 通道，換算與 safe_surround_cruise.read_battery_voltage 相同
（params.json Pcb_Version 1: 3.3 V x 3，其他: 5.2 V x 2），每次讀 5 次取中位數。只讀 ADC，不碰馬達、伺服機或相機。
點一下視窗 = 切換「電源供應」模式（只顯示電壓，不換算電量）；量到低於 NO_BATTERY_V 時顯示「未接電池」。
電量是 2 串鋰電池的電壓對照（同 endurance_test.estimate_battery_percent），只是估計：馬達出力時電壓會下降、
停下來後回升，所以要看靜止時的讀數。5 V 與供電不足旗標來自 vcgencmd。
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
PERCENT_TABLE = [(8.40, 100), (8.20, 90), (8.00, 80), (7.80, 70), (7.60, 60), (7.40, 50), (7.20, 35),
                 (7.00, 20), (6.80, 10), (6.60, 5), (6.40, 0)]
HISTORY_S = 600
NO_BATTERY_V = 5.5      # 低於這個電壓 = 電池輸入端沒有電 (例如電源供應器直接接 Pi)


def pcb_version() -> int:
    try:
        return int(json.loads((SERVER / "params.json").read_text()).get("Pcb_Version", 2))
    except Exception:  # noqa: BLE001
        return 2


def read_battery(bus, coefficient: float, scale: float):
    """電池電壓 (V)：5 次讀數的中位數；讀不到為 None。"""
    vals = []
    channel = 2
    command = 0x84 | ((((channel << 2) | (channel >> 1)) & 0x07) << 4)
    for _ in range(5):
        try:
            bus.write_byte(0x48, command)
            bus.read_byte(0x48)
            vals.append(bus.read_byte(0x48) / 255.0 * coefficient * scale)
        except OSError:
            pass
        time.sleep(0.01)
    return statistics.median(vals) if vals else None


def percent(v: float) -> int:
    if v >= PERCENT_TABLE[0][0]:
        return 100
    if v <= PERCENT_TABLE[-1][0]:
        return 0
    for (hv, hp), (lv, lp) in zip(PERCENT_TABLE, PERCENT_TABLE[1:]):
        if lv <= v <= hv:
            return round(lp + (v - lv) / (hv - lv) * (hp - lp))
    return 0


def vcgencmd(*args) -> str:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=1.0).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def pi_power():
    """(5 V 輸入電壓或 None, 供電不足說明)。"""
    v5 = None
    out = vcgencmd("pmic_read_adc", "EXT5V_V")
    if "=" in out:
        try:
            v5 = float(out.split("=")[-1].rstrip("V"))
        except ValueError:
            pass
    flags = vcgencmd("get_throttled")
    note = "無紀錄"
    if "=" in flags:
        f = int(flags.split("=")[-1], 16)
        if f & 0x1:
            note = "現在供電不足！"
        elif f & 0x10000:
            note = "開機後曾供電不足"
        elif f & 0x4:
            note = "現在降頻中"
        elif f & 0x40000:
            note = "開機後曾降頻"
    return v5, note


def main():
    import tkinter as tk
    import smbus

    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=2.0)
    args = ap.parse_args()
    pcb = pcb_version()
    coefficient, scale = (3.3, 3) if pcb == 1 else (5.2, 2)
    bus = smbus.SMBus(1)

    bg, fg = "#1e1e1e", "#e8e8e8"
    root = tk.Tk()
    root.title("電池")
    root.configure(bg=bg)
    root.attributes("-topmost", True)
    W, H = 300, 210
    root.geometry(f"{W}x{H}+{root.winfo_screenwidth() - W - 12}+48")

    big = tk.Label(root, text="--", fg=fg, bg=bg, font=("DejaVu Sans Mono", 26, "bold"))
    big.pack(anchor="w", padx=10, pady=(6, 0))
    bar = tk.Canvas(root, width=W - 20, height=16, bg="#333", highlightthickness=0)
    bar.pack(padx=10)
    info = tk.Label(root, text="", fg=fg, bg=bg, font=("DejaVu Sans Mono", 10), justify="left")
    info.pack(anchor="w", padx=10, pady=(4, 0))
    graph = tk.Canvas(root, width=W - 20, height=60, bg="#111", highlightthickness=0)
    graph.pack(padx=10, pady=(4, 6))

    hist = deque()
    supply = {"on": False}

    def toggle(_event=None):
        supply["on"] = not supply["on"]
        root.title("電源供應" if supply["on"] else "電池")

    for w in (root, big, bar, info, graph):
        w.bind("<Button-1>", toggle)

    def colour(p):
        return "#59d47a" if p >= 50 else ("#ffb84d" if p >= 20 else "#ff5c5c")

    def tick():
        now = time.monotonic()
        v = read_battery(bus, coefficient, scale)
        v5, note = pi_power()
        if v is not None:
            hist.append((now, v))
        while hist and now - hist[0][0] > HISTORY_S:
            hist.popleft()
        if v is None:                      # ADC 沒回應：車上電路板沒有電 (電源供應器直接接 Pi，或車子電源關著)
            big.config(text="未接電池", fg="#9a9a9a")
            bar.delete("all")
            bar.create_text(4, 8, text="車上電路板沒有電（電源供應直接接 Pi？）", fill="#bbb", anchor="w",
                            font=("DejaVu Sans Mono", 9))
        elif v < NO_BATTERY_V:
            big.config(text="未接電池", fg="#9a9a9a")
            bar.delete("all")
            bar.create_text(4, 8, text="電源供應中（點視窗可切換模式）", fill="#bbb", anchor="w",
                            font=("DejaVu Sans Mono", 9))
        elif supply["on"]:
            big.config(text=f"供電 {v:4.2f} V", fg="#6aa9ff")
            bar.delete("all")
            bar.create_text(4, 8, text="電源供應模式（點視窗切回電池）", fill="#bbb", anchor="w",
                            font=("DejaVu Sans Mono", 9))
        else:
            p = percent(v)
            big.config(text=f"{p:3d}%  {v:4.2f} V", fg=colour(p))
            bar.delete("all")
            bar.create_rectangle(0, 0, (W - 20) * p / 100, 16, fill=colour(p), outline="")
        trend = ""
        if len(hist) >= 2 and hist[-1][0] - hist[0][0] > 60:
            rate = (hist[-1][1] - hist[0][1]) / ((hist[-1][0] - hist[0][0]) / 60)
            trend = f"   {rate:+.3f} V/分"
        info.config(text=f"Pi 5V  {'--' if v5 is None else f'{v5:4.2f} V'}   {note}\n"
                         f"更新 {time.strftime('%H:%M:%S')}{trend}",
                    fg="#ff5c5c" if "不足" in note else fg)
        graph.delete("all")
        if len(hist) >= 2:
            gw, gh = W - 20, 60
            lo = min(min(x for _, x in hist), 6.4)
            hi = max(max(x for _, x in hist), 8.4)
            pts = []
            for t, x in hist:
                pts += [gw * (1 - (now - t) / HISTORY_S), gh - 4 - (gh - 8) * (x - lo) / (hi - lo)]
            graph.create_line(*pts, fill="#6aa9ff", width=2)
            graph.create_text(4, 2, text=f"{hi:.1f}", fill="#777", anchor="nw", font=("DejaVu Sans Mono", 8))
            graph.create_text(4, gh - 2, text=f"{lo:.1f}", fill="#777", anchor="sw", font=("DejaVu Sans Mono", 8))
            graph.create_text(gw - 4, gh - 2, text="最近 10 分鐘", fill="#777", anchor="se", font=("DejaVu Sans Mono", 8))
        root.after(int(args.every * 1000), tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
