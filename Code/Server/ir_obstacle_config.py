"""Load obstacle IR GPIO map from ir_obstacle_pins.json."""

from __future__ import annotations

import json
from pathlib import Path

SERVER = Path(__file__).resolve().parent
PINS_PATH = SERVER / "ir_obstacle_pins.json"


def load_ir_obstacle_config() -> dict:
    with PINS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def ir_obstacle_pins() -> dict[str, int]:
    """Name -> BCM GPIO for each physical sensor."""
    return {k: int(v) for k, v in load_ir_obstacle_config()["pins"].items()}


def ir_logic_aliases() -> dict[str, list[str]]:
    return dict(load_ir_obstacle_config()["logic_aliases"])


def expand_ir_hits(hits: dict[str, int]) -> dict[str, int]:
    """Add legacy keys front/left/right/rear as OR of mapped physical pins."""
    out = dict(hits)
    for alias, sources in ir_logic_aliases().items():
        out[alias] = 1 if any(out.get(name, 0) for name in sources) else 0
    return out
