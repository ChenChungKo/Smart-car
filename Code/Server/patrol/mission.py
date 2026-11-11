"""The task as a list of goals: the four corners, then the pose the car started from."""
from __future__ import annotations

import math
from dataclasses import dataclass

from patrol.world import RES_M


@dataclass
class Goal:
    name: str
    x: float
    y: float
    yaw: float | None = None     # None: any heading
    tol_m: float = 0.04
    want: tuple | None = None    # where it was asked to be; x, y move off it when something stands there

    def __post_init__(self):
        self.want = (self.x, self.y)


class Mission:
    def __init__(self, arena_m, start, inset_m: float):
        w, h = arena_m
        corners = [("corner bottom-left", inset_m, inset_m), ("corner bottom-right", w - inset_m, inset_m),
                   ("corner top-right", w - inset_m, h - inset_m), ("corner top-left", inset_m, h - inset_m)]
        first = min(range(4), key=lambda n: math.hypot(corners[n][1] - start[0], corners[n][2] - start[1]))
        self.goals = [Goal(*corners[(first + n) % 4]) for n in range(4)] + [Goal("home", start[0], start[1], start[2], 0.03)]
        self.index = 0
        self.skipped: list = []

    def current(self) -> Goal | None:
        return self.goals[self.index] if self.index < len(self.goals) else None

    def reached(self, pose) -> bool:
        g = self.current()
        if g is None or math.hypot(pose[0] - g.x, pose[1] - g.y) > g.tol_m:
            return False
        return g.yaw is None or abs((pose[2] - g.yaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(4.0)

    def advance(self, skipped: bool = False) -> None:
        if skipped:
            self.skipped.append(self.goals[self.index].name)
        self.index += 1

    @staticmethod
    def place(space, goal: Goal, reach_m: float = 0.16):
        """The point to plan to: the goal itself, or the nearest allowed point when something stands on it. None: none."""
        i, j = int(round(goal.want[0] / RES_M)), int(round(goal.want[1] / RES_M))
        from patrol.planner import heading_index
        k = None if goal.yaw is None else heading_index(goal.yaw)
        near = space.nearest_free(i, j, k, int(reach_m / RES_M))
        if near is None:
            return None
        return goal.want if (near[0], near[1]) == (i, j) else (near[0] * RES_M, near[1] * RES_M)
