#!/usr/bin/env python3
"""Quick GY-25T (UART attitude) check over USB-TTL (CH340 → /dev/ttyUSB0).

Typical frame (binary continuous / query reply):
  AA 00 YawH YawL PitchH PitchL RollH RollL SUM
  angle_deg = int16(H,L) / 100.0
  SUM = sum(bytes[0:7]) & 0xFF

Commands used by many GY-25 / GY-25T boards:
  A5 51       query Euler once
  A5 56 02    enable continuous Euler output
  A5 55       horizontal calibrate (keep still on a flat surface)
  A5 57       save settings (board-dependent)

Usage:
  python3 check_gy25t.py
  python3 check_gy25t.py --port /dev/ttyUSB0 --baud 115200
  python3 check_gy25t.py --calibrate-flat   # send flat calibrate, then read
"""

from __future__ import annotations

import argparse
import glob
import struct
import sys
import time

import serial


def find_port(preferred: str | None) -> str:
    if preferred:
        return preferred
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        return by_id[0]
    for name in ("/dev/ttyUSB0", "/dev/ttyACM0"):
        if glob.glob(name):
            return name
    raise SystemExit("找不到 USB 序列埠。請確認 GY-25T 的 CH340 已插上（lsusb 應有 1a86:7523）。")


def parse_aa_frames(buf: bytes) -> list[tuple[float, float, float]]:
    """Return list of (yaw, pitch, roll) in degrees."""
    out = []
    i = 0
    while i + 8 <= len(buf):
        if buf[i] != 0xAA:
            i += 1
            continue
        frame = buf[i : i + 8]
        if (sum(frame[:7]) & 0xFF) != frame[7]:
            i += 1
            continue
        yaw, pitch, roll = (struct.unpack(">hhh", frame[1:7])[k] / 100.0 for k in range(3))
        # Some firmwares put a mode byte at frame[1] (0x00) and angles start at [2].
        if frame[1] == 0x00 and i + 9 <= len(buf):
            frame9 = buf[i : i + 9]
            if (sum(frame9[:8]) & 0xFF) == frame9[8]:
                yaw, pitch, roll = (struct.unpack(">hhh", frame9[2:8])[k] / 100.0 for k in range(3))
                out.append((yaw, pitch, roll))
                i += 9
                continue
        out.append((yaw, pitch, roll))
        i += 8
    return out


def try_read(ser: serial.Serial, seconds: float = 1.0) -> bytes:
    deadline = time.monotonic() + seconds
    chunks = []
    while time.monotonic() < deadline:
        n = ser.in_waiting
        if n:
            chunks.append(ser.read(n))
        else:
            time.sleep(0.02)
    return b"".join(chunks)


def probe(port: str, baud: int, calibrate: bool) -> int:
    print(f"port={port}  baud={baud}")
    ser = serial.Serial(port, baud, timeout=0.2)
    time.sleep(0.2)
    ser.reset_input_buffer()

    cmds = [
        ("query Euler  A5 51", bytes([0xA5, 0x51])),
        ("continuous   A5 56 02", bytes([0xA5, 0x56, 0x02])),
    ]
    if calibrate:
        cmds.insert(0, ("flat calib   A5 55", bytes([0xA5, 0x55])))

    any_data = False
    for name, cmd in cmds:
        ser.write(cmd)
        ser.flush()
        time.sleep(0.35)
        data = try_read(ser, 0.6)
        print(f"{name:24s} -> {len(data):4d} bytes  {data[:32].hex(' ') if data else '(empty)'}")
        if data:
            any_data = True
            poses = parse_aa_frames(data)
            if poses:
                y, p, r = poses[-1]
                print(f"  parsed yaw={y:+.2f}° pitch={p:+.2f}° roll={r:+.2f}°  ({len(poses)} frames)")

    print("listening 3 s for continuous stream …")
    stream = try_read(ser, 3.0)
    print(f"stream -> {len(stream)} bytes")
    poses = parse_aa_frames(stream)
    if poses:
        print("live (Ctrl+C to stop after a few samples shown):")
        for y, p, r in poses[-5:]:
            print(f"  yaw={y:+7.2f}°  pitch={p:+7.2f}°  roll={r:+7.2f}°")
        # keep printing fresh frames briefly
        t0 = time.monotonic()
        buf = b""
        try:
            while time.monotonic() - t0 < 8.0:
                chunk = ser.read(ser.in_waiting or 1)
                if not chunk:
                    continue
                buf += chunk
                if len(buf) > 256:
                    buf = buf[-256:]
                frames = parse_aa_frames(buf)
                if frames:
                    y, p, r = frames[-1]
                    print(f"\ryaw={y:+7.2f}°  pitch={p:+7.2f}°  roll={r:+7.2f}°   ", end="", flush=True)
                    buf = b""
        except KeyboardInterrupt:
            print()
        print()
        ser.close()
        return 0

    ser.close()
    if not any_data:
        print(
            """
沒有收到任何位元組。請依序檢查：
1. 模組是否有電：GY-25T 的 VCC/GND 是否接到 USB-TTL（通常 5V；有的板子標 3.3V）。
2. TX/RX 是否交叉：模組 TX → CH340 RX，模組 RX → CH340 TX。
3. USB 線是否只供電、資料線壞了；換線或直接看 lsusb 是否仍是 1a86:7523。
4. 鮑率：多數 GY-25T 預設 115200，少數是 9600。可加 --baud 9600 再試。
5. 若接的是 I2C 腳位而不是 UART，這支腳本讀不到（需改接 TX/RX）。
"""
        )
        return 1
    print("有資料但解不出 AA 幀；把上面 hex 貼給我，再對協定微調。")
    return 2


def main() -> None:
    p = argparse.ArgumentParser(description="Test GY-25T over USB serial.")
    p.add_argument("--port", default=None, help="default: first /dev/serial/by-id/* or ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--calibrate-flat", action="store_true", help="send A5 55 flat calibrate first")
    args = p.parse_args()
    port = find_port(args.port)
    sys.exit(probe(port, args.baud, args.calibrate_flat))


if __name__ == "__main__":
    main()
