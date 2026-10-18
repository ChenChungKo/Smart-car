#!/usr/bin/env python3
"""Read six obstacle IR modules (see ir_obstacle_pins.json).

  python3 ir_obstacle_probe.py
"""

from __future__ import annotations

import argparse
import time

from gpiozero import DigitalInputDevice

from ir_obstacle_config import ir_obstacle_pins, load_ir_obstacle_config


def parse_args():
    p = argparse.ArgumentParser(description="Probe obstacle IR GPIO (active LOW).")
    p.add_argument("--debounce-s", type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_ir_obstacle_config()
    pins = ir_obstacle_pins()
    labels = cfg.get("physical_labels_zh", {})
    order = list(pins.keys())
    devices = [(name, pins[name], DigitalInputDevice(pins[name], pull_up=False)) for name in order]
    print("ON = module LED / triggered. Mapping from ir_obstacle_pins.json", flush=True)
    for name, pin, _ in devices:
        print(f"  {name} GPIO{pin} ({labels.get(name, '')})", flush=True)
    print("Ctrl+C to quit.", flush=True)
    shown = pending = None
    pending_since = 0.0
    try:
        while True:
            now = tuple(int(not int(dev.value)) for _, _, dev in devices)
            t = time.monotonic()
            if now != pending:
                pending, pending_since = now, t
            elif now != shown and t - pending_since >= args.debounce_s:
                shown = now
                parts = [f"{order[i]}={'ON' if on else 'off'}" for i, on in enumerate(shown)]
                print(time.strftime("%H:%M:%S ") + "  ".join(parts), flush=True)
            time.sleep(0.03)
    finally:
        for _, _, dev in devices:
            dev.close()


if __name__ == "__main__":
    main()
