"""Everything a run decided and saw, on disk, so that any moment can be looked at again (tools/patrol_why.py).

  console.log    what was printed
  control.jsonl  every control cycle: pose, command, wheels, state, why it waited, what the safety check found
  looks.jsonl    every look of the gimbal: angle, echo, pose it was put at, what went into the map; looks/NNNNN.jpg
  plans.jsonl    every plan: from, to, path, how long it took, or why there was none
  events.jsonl   goals reached, stops, errors
  maps/          the map whenever it changed (at most every half second): T.TTT.npy, seconds since the start
  tags.jsonl     every tag pose of the front and rear cameras
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def _plain(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (np.floating, np.integer)):
        return _plain(v.item())
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


class Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.file.write(text)
        return self.stream.write(text)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Recorder:
    def __init__(self, folder: Path):
        self.dir = folder
        (folder / "looks").mkdir(parents=True, exist_ok=True)
        (folder / "maps").mkdir(exist_ok=True)
        sys.stdout, sys.stderr = Tee(sys.stdout, folder / "console.log"), Tee(sys.stderr, folder / "console.log")
        self.t0 = time.monotonic()
        self.files = {n: open(folder / f"{n}.jsonl", "a", encoding="utf-8", buffering=1) for n in ("control", "looks", "plans", "events")}
        self.lock = threading.Lock()
        self.n_look = 0
        self.last_map = -1.0

    def rel(self, t: float | None = None) -> float:
        return (time.monotonic() if t is None else t) - self.t0

    def _write(self, name: str, row: dict) -> None:
        with self.lock:
            self.files[name].write(json.dumps(_plain(row), ensure_ascii=False) + "\n")

    def control(self, row: dict) -> None:
        self._write("control", row)

    def plan(self, row: dict) -> None:
        self._write("plans", dict(row, s=self.rel()))

    def event(self, what: str, **more) -> None:
        self._write("events", dict(s=self.rel(), what=what, **more))
        print(f"[{self.rel():6.1f} s] {what}" + ("" if not more else "  " + ", ".join(f"{k}={_plain(v)}" for k, v in more.items())), flush=True)

    def look(self, rec: dict, frame) -> str:
        self.n_look += 1
        name = f"{self.n_look:05d}.jpg"
        cv2.imwrite(str(self.dir / "looks" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self._write("looks", dict(rec, s=self.rel(rec["t"]), file=name))
        return name

    def map(self, log: np.ndarray) -> None:
        s = self.rel()
        if s - self.last_map >= 0.5:
            self.last_map = s
            np.save(self.dir / "maps" / f"{s:08.3f}.npy", log)

    def close(self, tags_rows: list, summary: dict) -> None:
        with open(self.dir / "tags.jsonl", "w", encoding="utf-8") as f:
            for r in tags_rows:
                f.write(json.dumps(_plain(dict(r, s=self.rel(r["t"])))) + "\n")
        (self.dir / "summary.json").write_text(json.dumps(_plain(summary), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for f in self.files.values():
            f.close()
