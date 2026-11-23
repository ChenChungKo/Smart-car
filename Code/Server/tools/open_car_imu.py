#!/usr/bin/env python3
"""開啟車上固定的 9 軸 IMU 檢視器（桌面圖示「車上 IMU」用的就是這支）。

    python3 tools/open_car_imu.py

1. 依實體 USB 孔找到車上那顆（/dev/serial/by-path 的 USB 1.1 孔），不受 ttyUSB 編號對調影響；
2. 模式不是 0x73 就切到 0x73（使用存在模組裡的手動磁場校正、關閉自動校正；不存檔，斷電後回到 0x7A）；
3. 用車上的安裝方向開 tools/gy95t_viewer.py（X 軸朝車尾、反轉 Yaw、反轉 Roll）。
   右側相機區一開始顯示雲台相機（tools/camera_panel.py，按鈕切換前/後/左/右/關閉）。
   注意：檢視器開著某支相機時，其他程式（巡邏、繞圈）就開不了那支；要跑車時先按「關閉」。
已經有檢視器開著這顆時只提示，不重複開。
"""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVER), str(SERVER / "tools")]
CAR_SOCKET = "usb-0:1.1:1.0"          # 車上 IMU 插的實體 USB 孔
MOUNT = ["--xdir", "車尾", "--rev-yaw", "1", "--rev-roll", "1", "--rev-pitch", "0"]


def message(text, error=False):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        (messagebox.showerror if error else messagebox.showinfo)("車上 IMU", text)
        root.destroy()
    except Exception:  # noqa: BLE001  沒有桌面時印出來
        print(text)


def main():
    paths = [p for p in glob.glob("/dev/serial/by-path/*") if CAR_SOCKET in p]
    if not paths:
        message("找不到車上的 IMU（USB 1.1 孔沒有序列裝置）。\n請確認 USB 線有插好、車子電源已開。", error=True)
        return
    tty = os.path.realpath(paths[0])
    running = subprocess.run(["pgrep", "-af", "gy95t_viewer.py"], capture_output=True, text=True).stdout
    if f"--port {tty} " in running + " ":
        message(f"車上 IMU 的檢視器已經開著了（{tty}）。\n請在上方工作列找「GY-95T 檢視器」。")
        return
    note = ""
    try:
        import serial
        from imu_regs import DEFAULT_STREAM, frame, read_regs, write_reg
        with serial.Serial(paths[0], 115200, timeout=0.05) as port:
            r = read_regs(port, 0x06, 1)
            if r and r[0] != 0x73:
                write_reg(port, 0x06, 0x73)
                r = read_regs(port, 0x06, 1)
            port.write(frame(*DEFAULT_STREAM))
            time.sleep(0.3)
        if not r or r[0] != 0x73:
            note = f"（模式切換未確認：0x06 = {'無回應' if not r else hex(r[0])}）"
    except Exception as e:  # noqa: BLE001
        note = f"（無法設定模式：{e}）"
    if note:
        print(note)
    os.chdir(SERVER)
    os.execv(sys.executable, [sys.executable, str(SERVER / "tools" / "gy95t_viewer.py"),
                              "--port", tty, "--model", "95t", *MOUNT])


if __name__ == "__main__":
    main()
