#!/usr/bin/env python3
"""Watch the IR sensors live (nothing moves): python3 tools/ir_watch.py

Prints a line whenever one changes: which are lit (1 = something within about 8 cm). Hold a hand near a sensor to see
which name it is; a sensor that stays lit with nothing near it is stuck (adjust its potentiometer, or ignore it).
"""
import sys, time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safe_surround_cruise import SafetySensors  # noqa: E402

s = SafetySensors(SimpleNamespace(no_ir=False, no_sonic=True, active_high=False))
last = None
try:
    while True:
        ir = s.read_ir()
        lit = sorted(k for k, v in ir.items() if v)
        if lit != last:
            print(time.strftime("%H:%M:%S"), "lit:", ", ".join(lit) or "none", flush=True)
            last = lit
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    s.close()
