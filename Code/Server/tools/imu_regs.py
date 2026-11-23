#!/usr/bin/env python3
"""GY-95 (SM06-GY-95, Chengdu Sensor Technology manual v1.1) configuration registers over the serial port.

    python3 tools/imu_regs.py                 # read registers 0x00-0x07 and explain them; nothing is written
    python3 tools/imu_regs.py --port /dev/ttyUSB1

Protocol (manual section 7): read = A4 03 <start> <count> <sum>, the module answers A4 03 <start> <count> <data..> <sum>.
In continuous-output mode a read frame also becomes the frame the module keeps streaming, so after reading the
configuration the default stream (registers 0x08-0x22, A4 03 08 1B CA) is asked for again. Nothing is saved
(register 0x05 is never written), so a power cycle also restores whatever the module had saved.
"""
import argparse
import glob
import time

import serial

DEFAULT_STREAM = bytes([0xA4, 0x03, 0x08, 0x1B])


def frame(*b):
    return bytes(b) + bytes([sum(b) & 0xFF])


def read_regs(port, start, count, tries=20):
    want = bytes([0xA4, 0x03, start, count])
    for _ in range(tries):
        port.reset_input_buffer()
        port.write(frame(0xA4, 0x03, start, count))
        t_end = time.monotonic() + 0.3
        buf = bytearray()
        while time.monotonic() < t_end:
            buf += port.read(max(1, port.in_waiting))
            i = buf.find(want)
            if i >= 0 and len(buf) - i >= 5 + count:
                f = bytes(buf[i:i + 5 + count])
                if sum(f[:-1]) & 0xFF == f[-1]:
                    return list(f[4:4 + count])
    return None


def explain(r):
    bauds = [2400, 4800, 9600, 19200, 38400, 57600, 115200, 230400]
    rates = ["10 Hz", "50 Hz", "100 Hz", "200 Hz"]
    methods = {0: "2D (limit values)", 1: "3D (limit values)", 2: "hard-iron ellipsoid", 3: "soft+hard-iron ellipsoid (default)",
               4: "calibration data NOT loaded"}
    c, m = r[6], r[7]
    print(f"0x00 id            0x{r[0]:02X}")
    print(f"0x01 baud          {r[1]} = {bauds[r[1]] if r[1] < 8 else '?'}")
    print(f"0x02 rate          {r[2]} = {rates[r[2]] if r[2] < 4 else '?'}")
    print(f"0x03 output        {r[3]} = {'continuous' if r[3] == 0 else 'on request' if r[3] == 1 else '?'}")
    print(f"0x04 format        {r[4]} = {'hex' if r[4] == 0 else 'ascii' if r[4] == 1 else '?'}")
    print(f"0x05 save          0x{r[5]:02X}")
    print(f"0x06 calibration   0x{c:02X} = {c:08b}")
    print(f"     bit7 mag auto-cal threshold  {(c >> 7) & 1}  ({'level 70-130' if c >> 7 & 1 else 'level 80-120'})")
    print(f"     bit6 gyro auto-cal           {(c >> 6) & 1}")
    print(f"     bit5 mag anti-interference   {(c >> 5) & 1}")
    print(f"     bit4 mag fusion              {(c >> 4) & 1}")
    print(f"     bit3 mag AUTO-CALIBRATION    {(c >> 3) & 1}")
    print(f"     bit2-0 mag cal method        {c & 7} = {methods.get(c & 7, '?')}")
    mag = ["+-2 G", "+-8 G", "+-12 G", "+-30 G"][(m >> 4) & 3]
    acc = ["+-2 g", "+-4 g", "+-8 g", "+-16 g"][(m >> 2) & 3]
    gyr = ["+-250", "+-500", "+-1000", "+-2000"][m & 3]
    print(f"0x07 mode          0x{m:02X} = {m:08b}: {'horizontal' if m >> 7 & 1 else 'vertical'}, "
          f"mag {mag}, acc {acc}, gyro {gyr} deg/s")


def write_reg(port, reg, value, tries=10):
    """A4 06 reg value sum; the module echoes the same frame when it took it. Not saved (0x05 is never written)."""
    f = frame(0xA4, 0x06, reg, value)
    for _ in range(tries):
        port.reset_input_buffer()
        port.write(f)
        t_end = time.monotonic() + 0.3
        buf = bytearray()
        while time.monotonic() < t_end:
            buf += port.read(max(1, port.in_waiting))
            if f in buf:
                return True
            if bytes([0xA4, 0x86]) in buf:
                return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--set", default=None, help="REG=VALUE, e.g. 0x06=0x74 (only 0x06 and 0x07; not saved)")
    a = ap.parse_args()
    port_name = a.port or (glob.glob("/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0") or ["/dev/ttyUSB0"])[0]
    with serial.Serial(port_name, 115200, timeout=0.05) as port:
        if a.set:
            reg, value = (int(x, 0) for x in a.set.split("="))
            if reg not in (0x06, 0x07):
                raise SystemExit("only 0x06 (calibration) and 0x07 (mode) may be set here")
            print(f"write 0x{reg:02X} = 0x{value:02X} (not saved): {'echoed, OK' if write_reg(port, reg, value) else 'NO echo'}")
        regs = read_regs(port, 0x00, 0x08)
        port.write(frame(*DEFAULT_STREAM))          # back to the default stream, whatever happened
        time.sleep(0.2)
        port.reset_input_buffer()
        time.sleep(0.3)
        tail = port.read(port.in_waiting or 64)
    print(f"port {port_name}")
    if regs is None:
        print("no answer to the read frame")
    else:
        explain(regs)
    print("stream after restoring:", "default frames OK" if DEFAULT_STREAM in tail else f"NOT the default ({tail[:16].hex(' ')})")


if __name__ == "__main__":
    main()
