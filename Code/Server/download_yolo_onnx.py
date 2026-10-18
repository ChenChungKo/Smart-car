#!/usr/bin/env python3
"""Download / export YOLO-nano ONNX into models/yolov8n.onnx.

Preferred: export YOLOv8n with ultralytics.
Fallback used in setup: YOLOv5n ONNX (85-dim output) also works with vision_detector.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUT = MODELS / "yolov8n.onnx"


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and OUT.stat().st_size > 1_000_000:
        print(f"Already present: {OUT} ({OUT.stat().st_size} bytes)")
        return 0
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Example:", file=sys.stderr)
        print("  cd Code/Server && python3 -m venv --system-site-packages .venv", file=sys.stderr)
        print("  .venv/bin/pip install ultralytics onnxruntime", file=sys.stderr)
        print("  .venv/bin/python download_yolo_onnx.py", file=sys.stderr)
        return 1
    print("Loading yolov8n.pt and exporting ONNX imgsz=320 ...")
    model = YOLO("yolov8n.pt")
    exported = model.export(format="onnx", imgsz=320, simplify=True)
    exported_path = Path(exported)
    if exported_path.resolve() != OUT.resolve():
        OUT.write_bytes(exported_path.read_bytes())
        print(f"Copied {exported_path} -> {OUT}")
    print(f"OK: {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
