"""Keeps a path to the current goal: plans when there is none, when the map no longer allows it, or when the car left it."""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from patrol.mission import Mission
from patrol.planner import Space, plan

REFRESH_S = 3.0          # plan again this often anyway: the map has learned more
OFF_PATH_M = 0.06
NO_PLACE_S = 6.0         # a goal with nowhere allowed near it for this long is given up


class Thinker(threading.Thread):
    def __init__(self, world, mission: Mission, pose, half, margin: float, recorder):
        super().__init__(daemon=True)
        self.world, self.mission, self.pose, self.half, self.margin, self.rec = world, mission, pose, half, margin, recorder
        self.halt = threading.Event()
        self.space = None
        self.path, self.path_goal, self.planned_at = None, None, 0.0
        self.note = "starting"
        self.no_place_since = None

    def refresh_space(self) -> None:
        if self.space is None or self.space.version != self.world.version:
            log = self.world.snapshot()
            self.space = Space(log, self.world.edge, self.half[0], self.half[1], self.margin, self.world.version)
            self.rec.map(log)

    def why_plan(self, goal, pose) -> str | None:
        if self.path is None or self.path_goal is not goal:
            return "no path to this goal yet"
        d = np.hypot(np.array([q[0] for q in self.path]) - pose[0], np.array([q[1] for q in self.path]) - pose[1])
        i = int(np.argmin(d))
        if float(d[i]) > OFF_PATH_M:
            return f"the car is {d[i] * 100:.0f} cm off the path"
        if not self.space.path_ok(self.path, i + 1):
            return "the map no longer allows the path"
        if time.monotonic() - self.planned_at > REFRESH_S:
            return "refresh"
        return None

    def step(self) -> None:
        self.refresh_space()
        goal = self.mission.current()
        if goal is None:
            self.path = None
            return
        pose = self.pose()
        why = self.why_plan(goal, pose)
        if why is None:
            return
        place = Mission.place(self.space, goal)
        if place is None:
            self.path, self.note = None, f"nowhere allowed near {goal.name}"
            self.no_place_since = self.no_place_since or time.monotonic()
            if time.monotonic() - self.no_place_since > NO_PLACE_S:
                self.rec.event("goal given up: something stands on it and around it", goal=goal.name)
                self.mission.advance(skipped=True)
                self.no_place_since = None
            return
        self.no_place_since = None
        if math.hypot(place[0] - goal.x, place[1] - goal.y) > 1e-6:
            self.rec.event("goal moved to the nearest allowed point", goal=goal.name, to=[place[0], place[1]], asked=list(goal.want))
            goal.x, goal.y = place
        path, info = plan(self.space, pose, (goal.x, goal.y), goal.yaw)
        self.planned_at = time.monotonic()
        self.rec.plan(dict(goal=goal.name, to=[goal.x, goal.y, goal.yaw], start=list(pose), reason=why, info=info, path=path, map_version=self.space.version))
        if path is None:
            self.note = info.get("why", "no path")
            self.path = None
        else:
            self.path, self.path_goal, self.note = path, goal, "ok"

    def run(self) -> None:
        while not self.halt.is_set():
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                self.note = f"planner error: {exc}"
                self.rec.event("planner error", error=str(exc))
                self.path = None
            time.sleep(0.1)
