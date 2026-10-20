"""Timed motion, dead reckoning, safe steps and turns.

Mixin of RouteFollower (route_follow.py); the methods use self.* of the whole follower."""

from __future__ import annotations

from route_common import *  # noqa: F401,F403


class MotionMixin:
    def drive(self, pwm):
        if self.motor is not None:
            self.motor.set_motor_model(*pwm)

    def stop(self):
        if self.motor is not None:
            try:
                self.motor.set_motor_model(0, 0, 0, 0)
            except Exception:
                pass

    # ---- motion ---------------------------------------------------------

    def _forward_pwm(self) -> tuple[int, int, int, int]:
        """Left-front, left-rear, right-front, right-rear duties for driving straight (car_nav.json)."""
        wheels = self.motion.get("forward_wheel_pwm")
        if wheels and len(wheels) == 4:
            return tuple(int(v) for v in wheels)
        fwd = int(self.motion["forward_pwm"])
        return (fwd, fwd, fwd, fwd)

    def _pwm(self, kind: str):
        fwd = self._forward_pwm()
        turn = int(self.motion["turn_pwm"])
        side = int(self.motion.get("strafe_pwm", 1200)) * (-1 if self.strafe_flip else 1)
        return {
            "forward": fwd,
            "reverse": tuple(-v for v in fwd),
            "left": (-turn, -turn, turn, turn),
            "right": (turn, turn, -turn, -turn),
            "strafe_left": (-side, side, side, -side),
            "strafe_right": (side, -side, -side, side),
        }[kind]

    def move(self, kind: str, duration: float) -> tuple[float, str | None]:
        """Timed motion; forward segments stop early on a safety reason."""
        if kind not in ("left", "right"):
            self.last_turn = None
        if kind not in ("strafe_left", "strafe_right"):
            self.strafe_acc = None
        if self.motor is None:
            print(f"    would {kind} {duration:.2f}s", flush=True)
            if self.stop_event.wait(0.5):
                raise Abort("stopped")
            return 0.0, None
        reason = None
        started = time.monotonic()
        self.drive(self._pwm(kind))
        while not self.stop_event.is_set():
            elapsed = time.monotonic() - started
            if elapsed >= duration:
                break
            if kind == "forward":
                self.travel = float(self.motion["forward_mps"]) * elapsed  # the pose is only updated after
                reason = self.safety_reason()
                if reason:
                    break
            time.sleep(0.04)
        self.stop()
        self.travel = 0.0
        elapsed = time.monotonic() - started
        if self.stop_event.is_set():
            raise Abort("stopped")
        self.dead_reckon(kind, elapsed)
        if kind in ("forward", "reverse", "strafe_left", "strafe_right"):
            print(f"      [{kind} {elapsed:.2f}s ({elapsed * float(self.motion['forward_mps']) * 100:.0f}cm at the calibrated speed)"
                  f"{' stopped: ' + reason if reason else ''} -> pose x={self.pose.x:.3f} y={self.pose.y:.3f}]", flush=True)
        return elapsed, reason

    def turn_rate(self, kind: str) -> float:
        """Turn rate (deg/s) of "left" or "right": car_nav.json, scaled by what the last turns really did."""
        return float(self.motion["turn_left_dps" if kind == "left" else "turn_right_dps"]) * self.turn_scale[kind]

    def learn_strafe(self, fix_yaw: float) -> None:
        """A tag fix after a long enough strafe says how much it turned the car: learn deg per metre.

        The wheels are not equally strong, so a strafe is also a turn (about 1 deg per cm on this car). Strafes
        add up until a fix arrives; under 8 cm in all the turn is lost in the noise of the fix.
        """
        acc, self.strafe_acc = self.strafe_acc, None
        if acc is None or abs(acc[1]) < 0.08 or not self.args.learn_motion:
            return
        sample = wrap_rad(fix_yaw - acc[0]) / acc[1]
        if abs(sample) > 3.0:  # more than 170 deg per metre: not this (a re-localization, a turn in between)
            return
        old = self.strafe_yaw
        self.strafe_yaw = 0.5 * self.strafe_yaw + 0.5 * sample
        if abs(self.strafe_yaw - old) > math.radians(10.0):
            print(f"    strafe turns the car {math.degrees(self.strafe_yaw) / 100.0:+.2f}deg/cm (was "
                  f"{math.degrees(old) / 100.0:+.2f})", flush=True)

    def learn_turn(self, fix_yaw: float) -> None:
        """A tag fix right after a turn says how far it really turned: correct the rate used for the next turns.

        The calibrated rate is for one battery state; a fresh or a tired battery turns faster or slower. Only
        turns of 25 deg or more teach (a small one is as noisy as the fix), and a rate moves half way to what
        was seen, within 0.7 to 1.4 times the calibration.
        """
        if self.last_turn is None or not self.args.learn_motion:
            self.last_turn = None
            return
        kind, before, commanded = self.last_turn
        self.last_turn = None
        if abs(commanded) < math.radians(25.0):
            return
        done = wrap_rad(fix_yaw - before)
        ratio = done / commanded
        if not 0.4 < ratio < 1.8:  # turned the wrong way or not at all: not this rate's fault
            return
        scale = min(max(self.turn_scale[kind] * (0.5 + 0.5 * ratio), 0.7), 1.4)
        if abs(scale - self.turn_scale[kind]) >= 0.03:
            print(f"    turn {kind}: asked {math.degrees(commanded):+.0f}deg, did {math.degrees(done):+.0f}deg; "
                  f"rate now {self.turn_rate(kind) * scale / self.turn_scale[kind]:.0f}deg/s", flush=True)
        self.turn_scale[kind] = scale

    def dead_reckon(self, kind: str, elapsed: float):
        if self.pose is None or elapsed <= 0.0:
            return
        if self.use_filter and self.kalman.ready:
            amount = {"forward": float(self.motion["forward_mps"]), "reverse": float(self.motion["forward_mps"]),
                      "strafe_left": self.strafe_mps, "strafe_right": self.strafe_mps,
                      "left": math.radians(self.turn_rate("left")),
                      "right": math.radians(self.turn_rate("right"))}[kind] * elapsed
            self.kalman.sync(self.pose.x, self.pose.y, self.pose.yaw)
            self.kalman.predict(kind, amount)
        if kind in ("forward", "reverse"):
            step = float(self.motion["forward_mps"]) * elapsed * (1.0 if kind == "forward" else -1.0)
            # The car drifts left while driving straight; fold that into the expected heading.
            self.pose.yaw = wrap_rad(
                self.pose.yaw + math.radians(float(self.motion.get("drift_left_deg_per_m", 0.0))) * step)
            self.pose.x += step * math.cos(self.pose.yaw)
            self.pose.y += step * math.sin(self.pose.yaw)
        elif kind in ("strafe_left", "strafe_right"):
            side = self.strafe_mps * elapsed * (1.0 if kind == "strafe_left" else -1.0)
            if self.strafe_acc is None:
                self.strafe_acc = [self.pose.yaw, 0.0]
            self.strafe_acc[1] += side
            self.pose.x -= side * math.sin(self.pose.yaw)
            self.pose.y += side * math.cos(self.pose.yaw)
            self.pose.yaw = wrap_rad(self.pose.yaw + self.strafe_yaw * side)  # the car turns while it strafes
        elif kind == "left":
            self.pose.yaw = wrap_rad(self.pose.yaw + math.radians(self.turn_rate("left") * elapsed))
        elif kind == "right":
            self.pose.yaw = wrap_rad(self.pose.yaw - math.radians(self.turn_rate("right") * elapsed))
        self.pose.fixes = []

    # ---- wall geometry ------------------------------------------------------

    def clearance(self, x: float, y: float, yaw: float) -> float:
        """Smallest gap between a body corner and an arena wall (negative = through it)."""
        w, h = (float(v) for v in self.arena["arena_size_m"])
        hl, hw = self.body_half
        c, s = math.cos(yaw), math.sin(yaw)
        gap = float("inf")
        for lx, ly in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)):
            cx, cy = x + lx * c - ly * s, y + lx * s + ly * c
            gap = min(gap, cx, w - cx, cy, h - cy)
        return gap

    def sweep_clearance(self, x: float, y: float, yaw: float, turn: float = 0.0, step: float = 0.0,
                        side: float = 0.0) -> float:
        """Clearance over a turn in place by `turn` rad, a straight move by `step` m or a strafe by `side` m."""
        n = max(2, int(abs(math.degrees(turn)) / 5.0) + 1, int(math.hypot(step, side) / 0.02) + 1)
        dx = step * math.cos(yaw) - side * math.sin(yaw)
        dy = step * math.sin(yaw) + side * math.cos(yaw)
        return min(
            self.clearance(x + dx * k / (n - 1), y + dy * k / (n - 1), yaw + turn * k / (n - 1))
            for k in range(n)
        )

    def wall_limit(self) -> float:
        """Required corner gap: the margin, or (already inside it) no closer than now."""
        p = self.pose
        limit = min(self.args.wall_margin_m, self.clearance(p.x, p.y, p.yaw) - 0.005)
        return limit if self.allow_gap is None else min(limit, self.allow_gap)

    def safe_step(self, step: float, sideways: bool = False) -> float:
        """Shorten a straight move (or strafe) so no body corner gets closer than the margin to a wall."""
        p, margin = self.pose, self.wall_limit()

        drift = math.radians(3.0)

        def gap(d):
            if sideways:
                return self.sweep_clearance(p.x, p.y, p.yaw, side=d)
            return min(self.sweep_clearance(p.x, p.y, p.yaw + e, step=d) for e in (0.0, drift, -drift))

        if gap(step) >= margin:
            return step
        lo, hi = 0.0, step
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if gap(mid) >= margin:
                lo = mid
            else:
                hi = mid
        return lo

    def turn_by(self, err: float, min_s: float | None = None):
        """Turn in place, first shuffling along the heading if a corner would hit a wall."""
        p, margin = self.pose, self.wall_limit()
        kind = "left" if err > 0 else "right"
        rate = self.turn_rate(kind)
        min_s = self.args.min_turn_s if min_s is None else min_s
        duration = min(max(math.degrees(abs(err)) / rate, min_s), self.args.max_turn_s)
        turn = math.copysign(math.radians(rate * duration), err)
        steps = [s * k * 0.02 for k in range(1, 11) for s in (1, -1)]
        shifts = [(0.0, 0.0)] + [(d, 0.0) for d in steps] + [(0.0, d) for d in steps if abs(d) <= 0.10]
        reachable = [s for s in shifts
                     if s == (0.0, 0.0) or self.sweep_clearance(p.x, p.y, p.yaw, step=s[0], side=s[1]) >= margin]
        reachable.sort(key=lambda s: abs(s[0]) + abs(s[1]))

        def turn_gap(s, angle):
            c, sn = math.cos(p.yaw), math.sin(p.yaw)
            return self.sweep_clearance(p.x + s[0] * c - s[1] * sn, p.y + s[0] * sn + s[1] * c, p.yaw, turn=angle)

        shift = next((s for s in reachable if turn_gap(s, turn) >= margin), None)
        if shift is None:
            # No spot allows the whole turn: take the biggest safe part and re-localize.
            best = None
            for s in reachable:
                for frac in (0.75, 0.5, 0.35, 0.2):
                    if math.degrees(abs(turn) * frac) >= 2.0 and turn_gap(s, turn * frac) >= margin:
                        if best is None or frac > best[1]:
                            best = (s, frac)
                        break
            if best is None:
                gap = self.sweep_clearance(p.x, p.y, p.yaw, turn=turn)
                raise Abort(f"turning here would hit a wall (corner gap {gap * 100:.0f}cm); move the car inward")
            shift, frac = best
            print(f"    only {math.degrees(abs(turn) * frac):.0f}deg of the turn is safe here", flush=True)
            duration *= frac
        self.last_turn = (kind, self.pose.yaw, turn)
        step, side = shift
        if step:
            print(f"    wall too close to turn; {'forward' if step > 0 else 'reverse'} {abs(step) * 100:.0f}cm first",
                  flush=True)
            self.move("forward" if step > 0 else "reverse", abs(step) / float(self.motion["forward_mps"]))
        elif side:
            print(f"    wall too close to turn; strafe {'left' if side > 0 else 'right'} {abs(side) * 100:.0f}cm first",
                  flush=True)
            self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
        self.last_turn = (kind, self.pose.yaw, turn)
        self.move(kind, duration)

    def strafe(self, left: float, ahead_cap: bool = True) -> bool:
        """Mecanum sideways move by `left` m (negative = right); learns the strafe speed from the next fix.

        `ahead_cap=False` for going around an obstacle: the ultrasonic then reads the obstacle itself,
        right in front, and the cap below would forbid any strafe at all.
        """
        side = math.copysign(min(abs(left), 0.06), left)
        # Strafing along the lane swings the nose or the tail sideways into a wall the pose can
        # easily underestimate, so cap that direction with the ultrasonic reading.
        if self.lane_yaw is not None and ahead_cap:
            along = abs(math.cos(wrap_rad(self.pose.yaw - self.lane_yaw)))
            if along > 0.7:
                sonic = self.sensors.read_sonic() if self.sensors is not None else None
                if sonic is not None:
                    side = math.copysign(min(abs(side), max(sonic / 100.0 - 0.04, 0.0) / along), side)
        side = self.safe_step(side, sideways=True)
        if abs(side) < 0.005:
            print("    wall beside the car; cannot strafe", flush=True)
            return False
        before = Pose2D(self.pose.x, self.pose.y, self.pose.yaw)
        elapsed, _ = self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
        fix = self.averaged_fix(2)
        if fix is None:
            return True
        self.pose = fix
        moved = -(fix.x - before.x) * math.sin(before.yaw) + (fix.y - before.y) * math.cos(before.yaw)
        if elapsed <= 0.0 or abs(moved) < 0.01:
            return True
        if moved * side < 0:
            if abs(side) >= 0.04 and abs(moved) >= 0.5 * abs(side):
                self.strafe_wrong += 1
                if self.strafe_wrong >= 2:
                    self.strafe_flip = not self.strafe_flip
                    self.strafe_wrong = 0
                    print("    strafe went the other way twice; flipping strafe direction", flush=True)
            return True
        self.strafe_wrong = 0
        if abs(side) < 0.03:
            return True
        speed = min(max(abs(moved) / elapsed, 0.5 * self.strafe_base), 2.0 * self.strafe_base)
        self.strafe_mps = 0.5 * self.strafe_mps + 0.5 * speed
        print(f"    strafed {moved * 100:+.0f}cm (asked {side * 100:+.0f}cm); strafe speed now "
              f"{self.strafe_mps:.3f} m/s", flush=True)
        return True

    # ---- obstacle detour ------------------------------------------------

    def wall_hit(self, x: float, y: float, angle: float) -> tuple[float, str]:
        """Distance from (x, y) along `angle` to the arena wall it points at, and that wall: "x" for
        x=0 / x=w, "y" for y=0 / y=h."""
        w, h = (float(v) for v in self.arena["arena_size_m"])
        dx, dy = math.cos(angle), math.sin(angle)
        hits = [(t, axis) for t, axis in ((((w - x) / dx) if dx > 1e-6 else None, "x"),
                                         ((-x / dx) if dx < -1e-6 else None, "x"),
                                         (((h - y) / dy) if dy > 1e-6 else None, "y"),
                                         ((-y / dy) if dy < -1e-6 else None, "y"))
                if t is not None and t >= 0.0]
        return min(hits) if hits else (float("inf"), "")

    def wall_distance(self, x: float, y: float, angle: float) -> float:
        """Distance from (x, y) along `angle` to the arena wall it points at."""
        return self.wall_hit(x, y, angle)[0]

    def wait_clear(self, reason: str):
        print(f"  blocked ({reason}); reversing 8cm", flush=True)
        self.move("reverse", 0.08 / float(self.motion["forward_mps"]))
        print(f"  still blocked? waiting up to {self.args.blocked_wait_s:.0f}s", flush=True)
        deadline = time.monotonic() + self.args.blocked_wait_s
        while time.monotonic() < deadline:
            if self.stop_event.wait(0.3):
                raise Abort("stopped")
            reason = self.safety_reason()
            if reason is None:
                print("  path clear again", flush=True)
                return
        raise Abort(f"route blocked: {reason}")
