#!/usr/bin/env python3
"""Run route_follow.py in a flat 2D simulation of the arena, with foam blocks where you put them.

No hardware and no extra software: route_follow's own logic drives a simulated mecanum car
(sim_hardware.py). The ultrasonic and IR are rays against the walls and the blocks; every motion
step checks the body against them, and a touch ends the run as a collision.

    python3 route_sim.py                                  # one block mid-way along each wall
    python3 route_sim.py --layout none                    # empty arena
    python3 route_sim.py --blocks 65,75,8,28 112,132,60,70 --runs 4
    python3 route_sim.py --runs 8 -- --no-detour          # after --: route_follow.py options
    python3 route_sim.py --world ../../webots/worlds/arena_lap.wbt    # the obstacles of a Webots world

Blocks are x0,x1,y0,y1 in cm from the bottom-left corner. Each run writes route_sim_debug/<name>/
(run.log, trajectory.png with the blocks in red); the summary table goes to the terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import random
import re
import sys
from pathlib import Path

import cv2

from aruco_localizer import MAP_PATH, load_json
from route_follow import Abort, parse_args as route_args
from sim_hardware import IR_LAYOUT, IR_RANGE_M, MotionError, SimFollower, body_outline

SERVER = Path(__file__).resolve().parent
OUT = SERVER / "route_sim_debug"
# Ready-made scenes (cm). "lap": one block in the middle of each --hug lane, as on the real arena.
LAYOUTS = {
    "lap": ["65,75,8,28", "112,132,60,70", "60,80,118,128", "10,20,60,80"],
    "lap2": ["50,70,12,22", "115,125,60,80", "75,85,112,132", "12,32,65,75"],
    "partial": ["60,80,20,30", "120,130,70,90", "55,65,110,130", "8,18,60,80"],
    "none": [],
}


class FlatWorld:
    """The arena as rectangles on the floor; rays for the ultrasonic and IR."""

    def __init__(self, size_m, blocks_m, home, sonic_half_deg: float = 8.0):
        self.w, self.h = size_m
        self.blocks = blocks_m
        self.truth = list(home)
        self.sonic_half = sonic_half_deg
        self.time = 0.0

    def inside_block(self, x: float, y: float) -> bool:
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in self.blocks)

    def ray(self, ox: float, oy: float, angle: float, max_m: float = 2.0) -> float | None:
        dx, dy = math.cos(angle), math.sin(angle)
        d = 0.0
        while d < max_m:
            x, y = ox + dx * d, oy + dy * d
            if not (0.0 <= x <= self.w and 0.0 <= y <= self.h) or self.inside_block(x, y):
                return d
            d += 0.003
        return None

    def place(self, x: float, y: float, yaw: float):
        self.truth = [x, y, yaw]

    def tick(self, dt: float):
        self.time += dt

    def sonic_m(self, pan_deg: float) -> float | None:
        x, y, yaw = self.truth
        a = yaw + math.radians(pan_deg)
        ox = x + 0.085 * math.cos(yaw) + 0.03 * math.cos(a)
        oy = y + 0.085 * math.sin(yaw) + 0.03 * math.sin(a)
        spread = (-self.sonic_half, -self.sonic_half / 2, 0.0, self.sonic_half / 2, self.sonic_half)
        hits = [d for d in (self.ray(ox, oy, a + math.radians(e)) for e in spread) if d is not None]
        return min(hits) if hits else None

    def ir(self) -> dict[str, int]:
        x, y, yaw = self.truth
        c, s = math.cos(yaw), math.sin(yaw)
        out = {}
        for name, (lx, ly, look) in IR_LAYOUT.items():
            px, py = x + lx * c - ly * s, y + lx * s + ly * c
            out[name] = int(self.ray(px, py, yaw + math.radians(look), IR_RANGE_M) is not None)
        return out

    def hits(self, x: float, y: float, yaw: float) -> str | None:
        for px, py in body_outline(x, y, yaw):
            if not (0.0 <= px <= self.w and 0.0 <= py <= self.h):
                return "a wall"
            if self.inside_block(px, py):
                return f"a block at ({px * 100:.0f},{py * 100:.0f})cm"
        return None


def world_blocks(path: Path, origin=(-0.70, -0.70)) -> list[tuple[float, float, float, float]]:
    """Floor rectangles (arena m) of the Solids in a Webots .wbt, as Webots' route_follow_webots sees them.

    Box and IndexedFaceSet bounding objects, turned by multiples of 90 deg about z. Walls (outside the
    arena) are left out: the arena edge is already a wall here.
    """
    text = path.read_text(encoding="utf-8")
    shapes = {}  # DEF name -> local (x0, x1, y0, y1)
    for name, body in re.findall(r"DEF (\w+) IndexedFaceSet \{.*?point \[(.*?)\]", text, re.S):
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", body)]
        xs, ys = nums[0::3], nums[1::3]
        shapes[name] = (min(xs), max(xs), min(ys), max(ys))
    out = []
    for chunk in re.split(r"\n(?=Solid \{)", text)[1:]:
        t = re.search(r"^  translation (\S+) (\S+) (\S+)", chunk, re.M)
        r = re.search(r"^  rotation (\S+) (\S+) (\S+) (\S+)", chunk, re.M)
        box = re.search(r"boundingObject Box \{\s*size (\S+) (\S+) (\S+)", chunk)
        use = re.search(r"boundingObject USE (\w+)", chunk)
        if t is None or not (box or use):
            continue
        tx, ty = float(t.group(1)), float(t.group(2))
        if box:
            sx, sy = float(box.group(1)), float(box.group(2))
            local = (-sx / 2, sx / 2, -sy / 2, sy / 2)
        elif use.group(1) in shapes:
            local = shapes[use.group(1)]
        else:
            continue
        quarter = round(float(r.group(4)) * (1 if r is None or float(r.group(3)) >= 0 else -1) / (math.pi / 2)) \
            if r else 0
        x0, x1, y0, y1 = local
        for _ in range(quarter % 4):  # turn the rectangle 90 deg left
            x0, x1, y0, y1 = -y1, -y0, x0, x1
        x0, x1, y0, y1 = x0 + tx - origin[0], x1 + tx - origin[0], y0 + ty - origin[1], y1 + ty - origin[1]
        if x1 <= 0.0 or y1 <= 0.0 or x0 >= -2 * origin[0] or y0 >= -2 * origin[1]:
            continue  # a wall, outside the arena
        out.append((x0, x1, y0, y1))
    return out


def draw(path: Path, blocks_m, w: float, h: float):
    """Add the blocks in red to route_follow's trajectory.png (same 640 px mapping as draw_map)."""
    img = cv2.imread(str(path))
    if img is None:
        return
    margin, scale = 30, (640 - 60) / max(w, h)
    img_h = int(h * scale) + 2 * margin
    for x0, x1, y0, y1 in blocks_m:
        cv2.rectangle(img, (int(margin + x0 * scale), int(img_h - margin - y1 * scale)),
                      (int(margin + x1 * scale), int(img_h - margin - y0 * scale)), (0, 0, 200), 2)
    cv2.imwrite(str(path), img)


def run_once(name: str, blocks_m, seed: int, extra: list[str], error: MotionError, quiet: bool) -> dict:
    random.seed(seed)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    args = route_args(["--hug", "--return-home", "--arm", "--no-window", "--debug-dir", str(out), *extra])
    arena = load_json(MAP_PATH)
    home = arena["home"]
    world = FlatWorld([float(v) for v in arena["arena_size_m"]], blocks_m,
                      (home[0], home[1], math.radians(home[2])))
    log = io.StringIO()
    result = {"name": name, "seed": seed}
    with contextlib.redirect_stdout(log if quiet else sys.stdout):
        car = SimFollower(args, world, error)
        try:
            car.start()
            car.follow()
            off = math.hypot(world.truth[0] - car.home.x, world.truth[1] - car.home.y) if car.home else None
            result.update(ok=True, why="finished", home_cm=None if off is None else off * 100)
        except Abort as exc:
            result.update(ok=False, why=str(exc), home_cm=None)
        finally:
            car.close()
    if quiet:
        (out / "run.log").write_text(log.getvalue(), encoding="utf-8")
    draw(out / "trajectory.png", blocks_m, world.w, world.h)
    text = log.getvalue()
    result.update(detours=text.count("going around"), moves=car.moves, minutes=world.time / 60.0)
    return result


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="lap")
    p.add_argument("--blocks", nargs="*", default=None, metavar="X0,X1,Y0,Y1",
                   help="Blocks in cm instead of --layout.")
    p.add_argument("--world", default=None, help="Take the obstacles from a Webots .wbt instead.")
    p.add_argument("--runs", type=int, default=1, help="Repeat with different random noise.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="Print route_follow's log instead of saving it.")
    p.add_argument("--forward-error", type=float, default=MotionError.forward)
    p.add_argument("--strafe-error", type=float, default=MotionError.strafe)
    p.add_argument("--turn-error", type=float, default=MotionError.turn)
    a = p.parse_args(argv)
    specs = a.blocks if a.blocks is not None else LAYOUTS[a.layout]
    blocks_m = []
    for spec in specs:
        x0, x1, y0, y1 = (float(v) / 100.0 for v in spec.split(","))
        blocks_m.append((min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    if a.world:
        blocks_m = world_blocks(Path(a.world))
        print(f"{len(blocks_m)} obstacles from {a.world} (cm): " +
              ", ".join(f"x {b[0] * 100:.0f}~{b[1] * 100:.0f} y {b[2] * 100:.0f}~{b[3] * 100:.0f}" for b in blocks_m))
    error = MotionError(forward=a.forward_error, strafe=a.strafe_error, turn=a.turn_error)
    label = Path(a.world).stem if a.world else "custom" if a.blocks is not None else a.layout
    results = [run_once(f"{label}_{a.seed + k}", blocks_m, a.seed + k, extra, error, not a.verbose)
               for k in range(a.runs)]
    print(f"\n{'run':<14}{'result':<9}{'detours':>8}{'moves':>7}{'sim min':>9}{'home cm':>9}  note")
    for r in results:
        home = "-" if r["home_cm"] is None else f"{r['home_cm']:.1f}"
        print(f"{r['name']:<14}{'OK' if r['ok'] else 'STOP':<9}{r['detours']:>8}{r['moves']:>7}"
              f"{r['minutes']:>9.1f}{home:>9}  {'' if r['ok'] else r['why']}")
    ok = sum(r["ok"] for r in results)
    print(f"\n{ok}/{len(results)} laps finished; logs and trajectories in {OUT}/")


if __name__ == "__main__":
    main()
