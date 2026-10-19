#!/usr/bin/env python3
"""Webots controller: Code/Server/route_follow.py drives Freenove4WDSim in the Webots scene.

The route logic is the real one (via Code/Server/sim_hardware.py). Webots supplies the scene:
the ultrasonic and IR are its DistanceSensor rays against your walls and blocks, and the car is
moved kinematically by this supervisor, so mecanum strafes need no roller physics. Tag fixes are
the true pose plus noise (the fisheye cameras are in the PROTO for viewing).

Every Solid in the world with a Box (or IndexedFaceSet) boundingObject is an obstacle: a body
touch stops the run with "SIM COLLISION". The arena's bottom-left inner corner is at --origin in
world coordinates (default -0.70,-0.70: a 1.4 m arena centred on the world origin).

controllerArgs are route_follow.py options; with none it runs the wall-hugging lap with detours:
    --hug --return-home
Extra controller-only options: --origin=X,Y  --cameras (enable the five cameras to watch them)

Needs numpy and opencv-contrib-python in the Python that Webots uses.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[3] / "Code" / "Server"
sys.path.insert(0, str(SERVER))

from controller import Supervisor  # noqa: E402

from route_follow import Abort, parse_args  # noqa: E402
from sim_hardware import IR_LAYOUT, IR_RANGE_M, SimFollower, body_outline  # noqa: E402

CAMERAS = ("cam_front", "cam_rear", "cam_left", "cam_right", "cam_gimbal")


def footprint(node) -> list[tuple[float, float]] | None:
    """Floor outline (world x, y) of a Solid's boundingObject, or None when it has none we read."""
    field = node.getField("boundingObject")
    shape = field.getSFNode() if field is not None else None
    if shape is None:
        return None
    kind = shape.getTypeName()
    if kind == "Box":
        sx, sy, _sz = shape.getField("size").getSFVec3f()
        local = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    elif kind == "IndexedFaceSet":
        coord = shape.getField("coord").getSFNode()
        points = coord.getField("point")
        xy = [points.getMFVec3f(i)[:2] for i in range(points.getCount())]
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        local = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
    else:
        print(f"note: {node.getField('name').getSFString()} has a {kind} boundingObject; not checked for contact")
        return None
    tx, ty, _tz = node.getField("translation").getSFVec3f()
    ax, ay, az, angle = node.getField("rotation").getSFRotation()
    if abs(ax) > 1e-3 or abs(ay) > 1e-3:
        print(f"note: {node.getField('name').getSFString()} is tilted; its outline is approximate")
    yaw = angle if az >= 0 else -angle
    c, s = math.cos(yaw), math.sin(yaw)
    return [(tx + x * c - y * s, ty + x * s + y * c) for x, y in local]


def inside(poly, x: float, y: float) -> bool:
    """Point in a convex polygon (any winding)."""
    sign = 0
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(cross) < 1e-12:
            continue
        if sign == 0:
            sign = 1 if cross > 0 else -1
        elif (cross > 0) != (sign > 0):
            return False
    return True


class WebotsWorld:
    """sim_hardware's world backend on top of the Webots supervisor."""

    def __init__(self, robot: Supervisor, origin: tuple[float, float], cameras: bool):
        self.robot = robot
        self.step_ms = int(robot.getBasicTimeStep())
        self.ox, self.oy = origin
        me = robot.getSelf()
        self.trans, self.rot = me.getField("translation"), me.getField("rotation")
        x, y, self.z = self.trans.getSFVec3f()
        ax, ay, az, angle = self.rot.getSFRotation()
        self.truth = [x - self.ox, y - self.oy, angle if az >= 0 else -angle]
        self.pan = robot.getDevice("servo_pan")
        self.pan.setPosition(0.0)
        self.sonic = robot.getDevice("sonic")
        self.sonic.enable(self.step_ms)
        self.irs = {name: robot.getDevice(f"ir_{name}") for name in IR_LAYOUT}
        for device in self.irs.values():
            device.enable(self.step_ms)
        if cameras:
            for name in CAMERAS:
                robot.getDevice(name).enable(4 * self.step_ms)
        self.carry = 0.0
        self.obstacles = []
        children = robot.getRoot().getField("children")
        for i in range(children.getCount()):
            node = children.getMFNode(i)
            if node.getId() == me.getId() or node.getBaseTypeName() != "Solid":
                continue
            poly = footprint(node)
            if poly:
                name = node.getField("name").getSFString() if node.getField("name") else node.getTypeName()
                self.obstacles.append((name, poly))
        print(f"{len(self.obstacles)} obstacles from the scene: {', '.join(n for n, _ in self.obstacles)}", flush=True)
        self.tick(0.5)  # sensors and the pan servo settle

    def tick(self, dt: float):
        self.carry += dt
        step_s = self.step_ms / 1000.0
        while self.carry >= step_s - 1e-9:
            if self.robot.step(self.step_ms) == -1:
                raise Abort("Webots stopped the simulation")
            self.carry -= step_s

    def place(self, x: float, y: float, yaw: float):
        self.truth = [x, y, yaw]
        self.trans.setSFVec3f([x + self.ox, y + self.oy, self.z])
        self.rot.setSFRotation([0.0, 0.0, 1.0, yaw])

    def sonic_m(self, pan_deg: float) -> float | None:
        target = math.radians(pan_deg)
        self.pan.setPosition(target)
        value = self.sonic.getValue()
        return None if value >= 2.9 else value

    def ir(self) -> dict[str, int]:
        return {name: int(device.getValue() < IR_RANGE_M) for name, device in self.irs.items()}

    def hits(self, x: float, y: float, yaw: float) -> str | None:
        outline = [(px + self.ox, py + self.oy) for px, py in body_outline(x, y, yaw)]
        for name, poly in self.obstacles:
            if any(inside(poly, px, py) for px, py in outline):
                return name
            if any(_in_body(cx - self.ox, cy - self.oy, x, y, yaw) for cx, cy in poly):
                return name
        return None


def _in_body(px: float, py: float, x: float, y: float, yaw: float, half_l: float = 0.145, half_w: float = 0.09) -> bool:
    dx, dy = px - x, py - y
    c, s = math.cos(yaw), math.sin(yaw)
    return abs(dx * c + dy * s) <= half_l and abs(-dx * s + dy * c) <= half_w


def main():
    robot = Supervisor()
    argv = [a for a in sys.argv[1:] if a]
    origin = (-0.70, -0.70)
    cameras = "--cameras" in argv
    for a in list(argv):
        if a.startswith("--origin="):
            origin = tuple(float(v) for v in a.split("=", 1)[1].split(","))
            argv.remove(a)
    argv = [a for a in argv if a != "--cameras"] or ["--hug", "--return-home"]
    args = parse_args(argv + ["--arm", "--no-window", "--debug-dir", "route_follow_webots_debug"])
    world = WebotsWorld(robot, origin, cameras)
    car = SimFollower(args, world)
    try:
        car.start()
        car.follow()
        print("WEBOTS RUN: finished", flush=True)
    except Abort as exc:
        print(f"WEBOTS RUN: stopped: {exc}", flush=True)
    finally:
        car.close()
        print(f"trajectory: {SERVER / 'route_follow_webots_debug' / 'trajectory.png'}", flush=True)
    while robot.step(world.step_ms) != -1:  # keep the scene open where it ended
        pass


if __name__ == "__main__":
    main()
