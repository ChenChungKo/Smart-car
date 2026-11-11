"""Puts the parts together: estimate, look, plan, follow, check, record. Each part is its own module; this file only
orders them in one control step."""
from __future__ import annotations

import collections
import math
import threading
import time

from pose_kalman import wrap
from patrol import parking
from patrol.drive import Follower, Safety, Wheels
from patrol.estimator import Estimator
from patrol.mission import Mission
from patrol.sensing import Eyes
from patrol.thinking import Thinker
from patrol.world import World
from vio_check import TagLogger

MAX_SIGMA_M = 0.06
WAIT_S = 20.0                 # standing this long for the same kind of reason ends the run
FIRST_LOOK = (-90, -60, -30, 0, 30, 60, 90, 45, 0, -45, 0)
CYCLE_S = 0.012


class Stop(Exception):
    pass


class Navigator:
    def __init__(self, car, opts, rec):
        self.car, self.opts, self.rec = car, opts, rec
        self.i2c = threading.Lock()
        self.status = dict(state="starting")
        self.trail: list = []
        self.sent = (0.0, 0.0, 0.0)
        self.waiting_since = None
        self.waits = collections.Counter()
        self.world = self.eyes = self.thinker = self.mission = self.est = self.tags = None
        self.result = dict(finished=False)

    # ---- start ---------------------------------------------------------------

    def setup(self) -> None:
        car = self.car
        car.start()
        if car.gyro is None:
            raise Stop("no IMU")
        self.arena = tuple(float(v) for v in car.arena["arena_size_m"])
        self.half = car.body_half
        self.tags = TagLogger(car)
        self.world = World(self.arena)
        self.est = Estimator(car, self.tags, self.arena, self.world)
        self.eyes = Eyes(car, self.world, self.est.pose_at, self.i2c, self.rec)
        self.wheels = Wheels(car, self.i2c)
        self.follower = Follower(self.opts.speed)
        self.safety = Safety(self.half[0], self.half[1], self.arena)
        self.status = dict(state="measuring the start")
        echoes = []
        for pan, wait in ((-90.0, 0.5), (0.0, 0.45)):
            self.eyes.point(pan)
            time.sleep(wait)
            echoes.append(self.eyes.echo(7))
        self.home_echoes = (echoes[1], echoes[0])          # (ahead, right)
        sweep = self.eyes.locate()
        self.rec.event("tags seen standing", **{k: v for k, v in sweep.items() if k != "pose"})
        start = self.est.measure_standing(echoes[0], sweep)
        if start is None:
            raise Stop("no pose from the tags")
        self.start = (start["x"], start["y"], start["yaw"])
        self.est.begin(*self.start, sigma_m=start["sigma_m"])
        ring = self.opts.start_clear / 100.0     # the one thing taken on trust: nothing stands this close beside or behind the car at the start, where no sensor looks
        self.world.mark_body(*self.start, self.half[0] + ring, self.half[1] + ring)
        self.rec.event("start", pose_cm_deg=[start["x"] * 100, start["y"] * 100, math.degrees(start["yaw"])], tags_only_cm=[v * 100 for v in start["tags_xy"]],
                       side_echo_used=start["side_echo_used"], tags=start["tags"], sigma_cm=start["sigma_m"] * 100, echo_ahead_cm=None if echoes[1] is None else echoes[1] * 100,
                       echo_right_cm=None if echoes[0] is None else echoes[0] * 100)
        self.status = dict(state="first look around")
        for pan in FIRST_LOOK:
            self.check_stop()
            self.eyes.look(float(pan))
        # the start pose already is the joint fit of everything seen standing here: what the first look gathered is not fed again
        self.eyes.take_ranges()
        self.eyes.take_tag_rows()
        self.est.seen = len(self.tags.rows)
        self.est.last, self.est.prev_gyro = time.monotonic(), self.car.gyro.peek()
        self.mission = Mission(self.arena, self.start, self.opts.inset / 100.0)
        self.thinker = Thinker(self.world, self.mission, self.est.pose, self.half, self.opts.margin / 100.0, self.rec)
        self.thinker.step()
        self.rec.event("goals", order=[g.name for g in self.mission.goals], first_plan=self.thinker.note,
                       steps=None if not self.thinker.path else len(self.thinker.path))

    def check_stop(self) -> None:
        if self.car.stop_event.is_set():
            raise Stop("stopped by the user")

    # ---- one control step -----------------------------------------------------

    def step(self) -> bool:
        """False when there is no goal left."""
        self.check_stop()
        pose = self.est.advance(self.sent[0], self.sent[1], self.eyes.take_ranges(), self.eyes.take_tag_rows())
        self.world.mark_body(*pose, *self.half)
        goal = self.mission.current()
        if goal is None:
            return False
        if self.mission.reached(pose):
            self.rec.event("reached", goal=goal.name, pose_cm_deg=[pose[0] * 100, pose[1] * 100, math.degrees(pose[2])])
            self.mission.advance()
            self.act(pose, (0.0, 0.0, 0.0), "reached " + goal.name, {}, {}, goal)
            return True
        cmd, why, info, safety = self.decide(pose, goal)
        self.act(pose, cmd, why, info, safety, goal)
        return True

    def decide(self, pose, goal):
        """(command, why it stands still or None, what the follower aimed at, what the safety check found)."""
        path = self.thinker.path if self.thinker.path_goal is goal else None
        sigma = self.est.sigma_m()
        if sigma > MAX_SIGMA_M:
            return (0.0, 0.0, 0.0), f"pose uncertain ({sigma * 100:.0f} cm)", {}, {}
        if not path:
            return (0.0, 0.0, 0.0), "no path: " + self.thinker.note, {}, {}
        cmd, info = self.follower.command(pose, path, goal)
        tx, ty = info["target"][0] - pose[0], info["target"][1] - pose[1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        self.eyes.aim_deg = math.degrees(math.atan2(-s * tx + c * ty, c * tx + s * ty)) if math.hypot(tx, ty) > 0.01 else 0.0
        ok, safety = self.safety.check(self.world.log, pose, cmd)
        if ok:
            return cmd, None, info, safety
        turn_only = (0.0, 0.0, cmd[2])
        if abs(cmd[2]) > 1e-3 and self.safety.check(self.world.log, pose, turn_only)[0]:
            return turn_only, None, info, dict(safety, allowed="turning only")
        return (0.0, 0.0, 0.0), f"{safety['kind']} ahead", info, safety

    def act(self, pose, cmd, why, info, safety, goal, state: str | None = None) -> None:
        now = time.monotonic()
        if why is None:
            self.waiting_since = None
        else:
            kind = why.split(":")[0].split("(")[0].strip()
            self.waits[kind] += 1
            self.waiting_since = self.waiting_since or now
            if now - self.waiting_since > WAIT_S:
                raise Stop(f"stood still {WAIT_S:.0f} s: {why}")
        armed = self.car.motor is not None
        pwm = self.wheels.send(*cmd) if armed else self.wheels.mix(*cmd)
        self.sent = cmd if armed else (0.0, 0.0, 0.0)
        self.trail.append((pose[0], pose[1]))
        row = dict(s=self.rec.rel(now), pose=pose, sigma_cm=self.est.sigma_m() * 100, goal=None if goal is None else goal.name,
                   cmd=cmd, pwm=pwm, armed=armed, why=why, follow=info, safety=safety, gyro_dps=math.degrees(self.est.omega),
                   scale=[float(self.est.f.x[3]), float(self.est.f.x[4])], map_version=self.world.version)
        self.rec.control(row)
        self.status = dict(row, state=state or ("driving" if why is None else "standing: " + why), t=now,
                           goals_done=self.mission.index, goals=len(self.mission.goals),
                           tags_used=self.est.tag_used, tags_refused=self.est.tag_refused, looks=self.eyes.looks, smeared=self.eyes.smeared)
        time.sleep(CYCLE_S)

    # ---- small moves used by parking -------------------------------------------

    def nudge(self, vx: float, vy: float, seconds: float, state: str) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.check_stop()
            pose = self.est.advance(self.sent[0], self.sent[1], [])
            om = 0.0 if (vx == 0.0 and vy == 0.0) else max(-0.5, min(0.5, 2.0 * wrap(self.start[2] - pose[2])))
            self.waiting_since = None
            self.act(pose, (vx, vy, om), None, {}, {}, None, state)
        self.wheels.stop()
        self.sent = (0.0, 0.0, 0.0)

    def hold(self, seconds: float, state: str) -> None:
        self.nudge(0.0, 0.0, seconds, state)

    # ---- the whole run ----------------------------------------------------------

    def run(self) -> None:
        t0 = time.monotonic()
        reason = "finished"
        try:
            self.setup()
            if self.car.motor is None:
                print(f"DRY-RUN: looking, planning and recording for {self.opts.dry_seconds:.0f} s; the wheels are not driven. Add --arm to drive.", flush=True)
            self.eyes.start()
            self.thinker.start()
            t0 = time.monotonic()
            while self.step():
                if self.car.motor is None and time.monotonic() - t0 > self.opts.dry_seconds:
                    raise Stop("end of the dry run")
                if time.monotonic() - t0 > self.opts.seconds:
                    raise Stop("out of time")
            self.wheels.stop()
            self.sent = (0.0, 0.0, 0.0)
            self.eyes.halt.set()
            self.eyes.join(timeout=2.0)
            if "home" not in self.mission.skipped:
                self.result["parking"] = parking.park(self, self.home_echoes)
            self.result["finished"] = True
        except Stop as exc:
            reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            reason = f"error: {exc!r}"
            import traceback
            traceback.print_exc()
        finally:
            self.finish(reason, time.monotonic() - t0)

    def finish(self, reason: str, seconds: float) -> None:
        car = self.car
        try:
            with self.i2c:
                car.stop()
        except Exception:  # noqa: BLE001
            pass
        if self.thinker is not None:
            self.thinker.halt.set()
        if self.eyes is not None:
            self.eyes.close()
        if self.tags is not None:
            self.tags.stop.set()
        total = max(1, sum(self.waits.values()) + 1)
        self.result.update(reason=reason, seconds=seconds, start=getattr(self, "start", None),
                           goals_reached=0 if self.mission is None else self.mission.index - len(self.mission.skipped),
                           skipped=[] if self.mission is None else self.mission.skipped, looks=0 if self.eyes is None else self.eyes.looks,
                           smeared=0 if self.eyes is None else self.eyes.smeared, stood_still_cycles=dict(self.waits))
        self.rec.event("end: " + reason, seconds=seconds, goals_reached=self.result["goals_reached"], skipped=self.result["skipped"],
                       stood_still_cycles=dict(self.waits), parking=self.result.get("parking"))
        self.status = dict(self.status, state="ended: " + reason)
        self.rec.close([] if self.tags is None else self.tags.rows, self.result)
        car.close()
