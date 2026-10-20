"""The route itself: lanes and corners, reaching each pose, parking, motion calibration.

Mixin of RouteFollower (route_follow.py); the methods use self.* of the whole follower."""

from __future__ import annotations

from route_common import *  # noqa: F401,F403


class PlanMixin:
    def aim_point(self, prev, target) -> tuple[tuple[float, float], float, float]:
        """Point to steer at, plus along-track progress and cross-track error.

        Steering at a point a little ahead on the prev->target line keeps the car on
        the line (e.g. along a wall) instead of cutting straight at the corner.
        """
        if prev is None:
            return target, 0.0, 0.0
        sx, sy = target[0] - prev[0], target[1] - prev[1]
        length = math.hypot(sx, sy)
        if length < 1e-6:
            return target, 0.0, 0.0
        ux, uy = sx / length, sy / length
        px, py = self.pose.x - prev[0], self.pose.y - prev[1]
        along = px * ux + py * uy
        cross = -px * uy + py * ux
        ahead = along + self.args.lookahead_m
        if ahead >= length:
            return target, along - length, cross
        ahead = max(ahead, 0.0)
        return (prev[0] + ux * ahead, prev[1] + uy * ahead), along - length, cross

    def go_to(self, index: int, target: tuple[float, float], prev: tuple[float, float] | None = None):
        blind = 0
        while not self.stop_event.is_set():
            dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
            dist = math.hypot(dx, dy)
            aim, past, cross = self.aim_point(prev, target)
            ax, ay = aim[0] - self.pose.x, aim[1] - self.pose.y
            err = wrap_rad(math.atan2(ay, ax) - self.pose.yaw)
            err_target = wrap_rad(math.atan2(dy, dx) - self.pose.yaw)
            print(f"wp{index} ({target[0]:.2f},{target[1]:.2f}) dist={dist:.2f}m "
                  f"err={math.degrees(err):+.0f}deg off-line={cross * 100:+.0f}cm "
                  f"pose {self.pose.text()}", flush=True)
            overshot = dist < 2.5 * self.args.arrive_m and abs(err_target) > math.pi / 2
            passed = prev is not None and past >= -self.args.arrive_m and abs(cross) < 2.0 * self.args.arrive_m
            if dist <= self.args.arrive_m or overshot or passed:
                print(f"reached wp{index}", flush=True)
                return
            if abs(err) > math.radians(self.args.align_deg):
                self.turn_by(err)
            else:
                segment = self.safe_step(min(math.hypot(ax, ay), dist, self.args.max_segment_m))
                if segment < 0.02:
                    back = -self.safe_step(-0.06)
                    if abs(err) > math.radians(3.0):
                        print("    wall ahead; straightening first", flush=True)
                        self.turn_by(err)
                    elif back >= 0.02:
                        print(f"    wall ahead; reverse {back * 100:.0f}cm", flush=True)
                        self.move("reverse", back / float(self.motion["forward_mps"]))
                    else:
                        raise Abort(f"wall ahead too close to reach wp{index}; move the route inward")
                    blind = self.refresh_pose(blind)
                    continue
                _elapsed, reason = self.move("forward", segment / float(self.motion["forward_mps"]))
                if reason:
                    self.wait_clear(reason)
            blind = self.refresh_pose(blind)
            self.record()
            self.save_map()

    def follow(self):
        self.pose = self.localize(3.0)
        if self.args.return_home:
            self.set_home()
        if self.pose is None:
            print("no tag at start; searching", flush=True)
            self.pose = self.search()
            if self.pose is None:
                raise Abort("no tag visible; check tags, lighting and arena_map.json")
        print(f"start pose {self.pose.text()}", flush=True)
        self.record()
        if self.args.hug:
            self.follow_hug()
            return
        while True:
            legs = self.legs()
            for i, (prev, wp) in enumerate(legs):
                self.go_to(i, wp, prev)
            if self.stop_event.is_set():
                raise Abort("stopped")
            if self.home is not None:
                self.dock()
            if not self.args.repeat:
                break
        print("route finished", flush=True)

    def go_pose(self, label: str, target: tuple[float, float, float], tol: float, tol_deg: float,
                attempts: int = 40, parking: bool = False) -> bool:
        """Reach (x, y, yaw) holonomically: turn to the heading, strafe sideways, drive along it.

        When parking, a blocked sensor ends the attempt instead of the run.
        """
        tol_yaw = math.radians(tol_deg)
        lane = max(self.args.lane_tol_m, tol / 2)
        blind = 0
        self.lane_yaw = target[2]
        self.lane_check = True
        try:
            return self._go_pose(label, target, tol, tol_deg, attempts, parking, tol_yaw, lane, blind)
        finally:
            self.lane_yaw = None
            self.lane_check = False

    def _go_pose(self, label, target, tol, tol_deg, attempts, parking, tol_yaw, lane, blind) -> bool:
        def blocked(reason):
            if not parking:
                self.wait_clear(reason)
                return False
            print(f"    blocked ({reason}) while parking; stopping here", flush=True)
            return True

        aligns = 0  # squaring-up turns since the last drive
        for attempt in range(1, attempts + 1):
            if self.stop_event.is_set():
                raise Abort("stopped")
            p = self.pose
            dyaw = wrap_rad(target[2] - p.yaw)
            dx, dy = target[0] - p.x, target[1] - p.y
            ahead = dx * math.cos(p.yaw) + dy * math.sin(p.yaw)
            left = -dx * math.sin(p.yaw) + dy * math.cos(p.yaw)
            print(f"{label} ({target[0]:.2f},{target[1]:.2f},{math.degrees(target[2]):+.0f}deg) "
                  f"ahead={ahead * 100:+.0f}cm left={left * 100:+.0f}cm heading err={math.degrees(dyaw):+.0f}deg "
                  f"pose {p.text()}", flush=True)
            if math.hypot(ahead, left) <= tol and abs(dyaw) <= tol_yaw:
                return True
            if attempt == 1:
                start_dist = math.hypot(ahead, left)
            if parking and math.hypot(ahead, left) > start_dist + 0.20:
                print("    pose drifted 20cm further from home; localization is unreliable here, stopping",
                      flush=True)
                return False
            near = math.hypot(ahead, left) < 0.15
            bearing = wrap_rad(math.atan2(dy, dx) - p.yaw)
            in_lane = abs(dyaw) <= math.radians(20.0) and ahead > 0 and abs(left) <= 0.5 * ahead
            self.lane_check = in_lane
            # Heading already right and the spot close: strafe and drive to it rather than turning beside
            # the walls (the start, joining the first lane from home, is like this).
            square = abs(dyaw) <= math.radians(20.0) and math.hypot(ahead, left) < 0.30
            if not near and not in_lane and not square:
                # Far and not already lined up along the target heading: face the spot and drive there.
                if abs(bearing) > math.radians(self.args.align_deg):
                    self.turn_by(bearing)
                else:
                    step = self.safe_step(min(math.hypot(dx, dy), self.args.max_segment_m))
                    if step < 0.01:
                        self.turn_by(bearing if abs(bearing) > math.radians(3.0) else dyaw, min_s=0.05)
                    else:
                        _elapsed, reason = self.move("forward", step / float(self.motion["forward_mps"]))
                        if reason and blocked(reason):
                            return False
            elif abs(dyaw) > tol_yaw:
                self.turn_by(dyaw, min_s=0.05)
            elif self.args.hug and not parking and ahead > 0.30 and aligns < 3 \
                    and abs(dyaw) > math.radians(DEPART_TOL_DEG):
                # About to drive a lane: first square the car up, so the lane is not started at the 4 deg that
                # the turn tolerance lets through.
                aligns += 1
                print(f"    squaring up before the lane: heading {math.degrees(dyaw):+.1f}deg off ({aligns}/3)", flush=True)
                self.turn_by(dyaw, min_s=0.03)
            elif abs(left) > lane and (abs(left) >= abs(ahead) or abs(left) > self.args.lane_tol_m) \
                    and self.strafe(left):
                self.record()
                self.save_map()
                continue
            else:
                aligns = 0
                step = self.safe_step(max(-self.args.max_segment_m, min(self.args.max_segment_m, ahead)))
                detour = self.args.hug and self.args.detour and not parking and step > 0
                if detour:
                    # The side looks see the edges of the path only up to ~24 cm ahead, and the beam straight
                    # ahead misses a block that covers just an edge: never drive past what was looked at.
                    step = min(step, CAM_STEP_M if self.scanner is not None else SCAN_STEP_M)
                    gap = self.obstacle_ahead()
                    short = self.beyond_spot(gap, target)
                    if short:
                        gap = None  # the spot this move goes to is short of it: no need to get past it
                    approach = self.args.approach_cm / 100.0 + (self.body_half[0] - SONIC_AHEAD_M)
                    if gap is not None:
                        # After a closing move the obstacle is inside the camera's blind zone: it reports the
                        # nearest floor it sees, not the distance. Believe the dead-reckoned gap instead, and
                        # go around: closing in again on that report would drive into it.
                        blind = self.scanner is not None and gap <= BLIND_GAP_M
                        if gap <= approach + 0.03 or blind:
                            # Seen only in the blind zone and never closed in on: the real gap is somewhere
                            # between 0 and what is reported, so no closing in either; go around from here.
                            true_gap = self.closed_in if (blind and self.closed_in is not None) else gap
                            self.closed_in = None
                            self.detour(target, true_gap)
                            continue
                        step = min(step, gap - approach)
                        print(f"    obstacle {(gap + SONIC_AHEAD_M - self.body_half[0]) * 100:.0f}cm ahead of the nose; "
                              f"closing to {self.args.approach_cm:.0f}cm", flush=True)
                        self.closed_in = gap - step
                    else:
                        self.closed_in = None
                if abs(step) < 0.01:
                    print("    wall reached; taking this as the spot", flush=True)
                    if detour:
                        self.snap_to_wall()
                    return False
                kind = "forward" if step > 0 else "reverse"
                if detour and (gap is not None or short):
                    self.closing_cm = (approach - 0.03) * 100.0
                    if gap is not None:
                        self.closing_ir_after = max(step - 0.04, 0.0)
                try:
                    _elapsed, reason = self.move(kind, abs(step) / float(self.motion["forward_mps"]))
                finally:
                    self.closing_cm = None
                    self.closing_ir_after = None
                if detour and gap is not None:
                    print(f"    closing move: asked {step * 100:.0f}cm, drove {_elapsed * float(self.motion['forward_mps']) * 100:.0f}cm "
                          f"by the clock ({reason or 'no stop'}); pose x={self.pose.x:.2f} y={self.pose.y:.2f}", flush=True)
                if reason:
                    self.closed_in = None
                if reason and detour:
                    gap = self.obstacle_ahead(reason)
                    if self.beyond_spot(gap, target):
                        # Stopped by what lies past the spot (not in the way of it): this is as close as we go.
                        print(f"    obstacle past the spot ({reason}); taking this as the spot", flush=True)
                        return False
                    if gap is not None:
                        self.detour(target, gap)
                        continue
                    if self.wall_ahead() < 0.20:
                        # Backing off the wall here could reverse into an obstacle just passed.
                        print(f"    wall ahead ({reason}); taking this as the spot", flush=True)
                        self.snap_to_wall()
                        return False
                if reason and blocked(reason):
                    return False
            if near:
                fix = self.averaged_fix(2)
                self.pose, blind = (fix, 0) if fix is not None else (self.pose, self.refresh_pose(blind, False))
            else:
                blind = self.refresh_pose(blind, full_turn=not in_lane)
            self.record()
            self.save_map()
        print(f"WARNING: {label} not reached within {tol * 100:.0f}cm / {tol_deg:.0f}deg", flush=True)
        return False

    def dock(self):
        """Park on the home pose; the walls may be as close as they were when the car was placed there."""
        home = self.home
        print(f"parking at home x={home.x:.2f} y={home.y:.2f} yaw={home.yaw_deg:+.0f}deg", flush=True)
        self.sweep_correct()
        self.allow_gap = max(self.clearance(home.x, home.y, home.yaw) - 0.01, 0.0)  # never into the wall
        self.parking = True
        try:
            if self.home_open_loop and self.hug_poses:
                # The tags are good at the lane start, 10 cm out of the corner, and bad in the corner itself. Stop
                # exactly on the lane start, then do the one move from there to home by dead reckoning.
                stage = self.hug_poses[0]
                self.go_pose("park-stage", stage, 0.02, 3.0, attempts=20, parking=True)
                ok = self.back_into_home()
            else:
                ok = self.go_pose("park", (home.x, home.y, home.yaw), self.args.home_tol_m,
                                  self.args.home_tol_deg, attempts=30, parking=True)
        finally:
            self.allow_gap = None
            self.parking = False
        p = self.pose
        off = math.hypot(home.x - p.x, home.y - p.y)
        dyaw = math.degrees(wrap_rad(home.yaw - p.yaw))
        print(f"{'parked at home' if ok else 'WARNING: parked near home'} "
              f"(off by {off * 100:.1f}cm, {dyaw:+.1f}deg)", flush=True)
        self.save_map()

    def back_into_home(self) -> bool:
        """From the lane start to home in two dead-reckoned moves (strafe, then back up), never into a wall."""
        p, home = self.pose, self.home
        c, s = math.cos(p.yaw), math.sin(p.yaw)
        dx, dy = home.x - p.x, home.y - p.y
        ahead, left = dx * c + dy * s, -dx * s + dy * c
        print(f"    from the lane start to home: {abs(left) * 100:.0f}cm {'left' if left > 0 else 'right'}, "
              f"{abs(ahead) * 100:.0f}cm {'forward' if ahead > 0 else 'back'}", flush=True)
        side = self.safe_step(left, sideways=True)
        if abs(side) >= 0.005:
            self.move("strafe_left" if side > 0 else "strafe_right", abs(side) / self.strafe_mps)
            self.face(home.yaw, 2.0)  # the strafe turned the car; the rear corner is 4 cm from the wall
        ahead = (home.x - self.pose.x) * c + (home.y - self.pose.y) * s
        step = self.safe_step(ahead)
        if abs(step) >= 0.005:
            self.move("forward" if step > 0 else "reverse", abs(step) / float(self.motion["forward_mps"]))
        return abs(side - left) < 0.02 and abs(step - ahead) < 0.02

    def build_hug_poses(self) -> list[tuple[float, float, float]]:
        """Counter-clockwise wall-hugging lap as (x, y, yaw) poses.

        Each wall: lane start, lane end, then the corner turning spot before and after the turn.
        """
        w, h = (float(v) for v in self.arena["arena_size_m"])
        hl, hw = self.body_half
        # A turn in place sweeps a circle of the half diagonal; `spot` keeps that circle off both walls.
        spot = math.hypot(hl, hw) + self.args.wall_margin_m + 0.02
        lane = spot if self.args.hug_gap_m is None else hw + self.args.hug_gap_m
        walls = [
            (0.0, (spot, lane), (w - spot, lane), (w - spot, spot)),
            (90.0, (w - lane, spot), (w - lane, h - spot), (w - spot, h - spot)),
            (180.0, (w - spot, h - lane), (spot, h - lane), (spot, h - spot)),
            (-90.0, (lane, h - spot), (lane, spot), (spot, spot)),
        ]
        poses = []
        for i, (yaw, start, end, corner) in enumerate(walls):
            nxt = math.radians(walls[(i + 1) % 4][0])
            yaw = math.radians(yaw)
            poses += [(*start, yaw), (*end, yaw), (*corner, yaw), (*corner, nxt)]
        return poses

    def follow_hug(self):
        poses = self.hug_poses
        n = len(poses)
        anchor = self.home or self.pose

        def cost(q):
            return math.hypot(q[0] - anchor.x, q[1] - anchor.y) + 0.1 * abs(wrap_rad(q[2] - anchor.yaw))

        k = min((i for i in range(n) if i % 4 in (0, 1)), key=lambda i: cost(poses[i]))
        hw = self.body_half[1]
        lane = abs(poses[0][1])  # the first lane runs along the y=0 wall
        print(f"hug lane: body {(lane - hw) * 100:.0f}cm from the walls; starting at pose {k}", flush=True)
        while True:
            for j in range(n + 1):
                i = (k + j) % n
                kind = ("lane start", "lane end", "corner", "turned")[i % 4]
                self.go_pose(f"p{i} {kind}", poses[i], 0.04, 4.0)
            if self.stop_event.is_set():
                raise Abort("stopped")
            if self.home is not None:
                self.dock()
            if not self.args.repeat:
                break
        print("route finished", flush=True)

    def legs(self) -> list[tuple[tuple[float, float] | None, tuple[float, float]]]:
        """(previous point, target) pairs; a closed route is joined at the side nearest the car.

        With a home pose the lap starts and ends at the point of that side nearest home.
        """
        route = self.route
        closed = len(route) > 2 and math.dist(route[0], route[-1]) < 1e-6
        if not closed or self.args.start_at_first:
            return [(route[i - 1] if i else None, wp) for i, wp in enumerate(route)]
        ring = route[:-1]
        n = len(ring)
        anchor = self.home or self.pose

        def nearest(a, b):
            sx, sy = b[0] - a[0], b[1] - a[1]
            t = ((anchor.x - a[0]) * sx + (anchor.y - a[1]) * sy) / max(sx * sx + sy * sy, 1e-9)
            t = min(max(t, 0.0), 1.0)
            return a[0] + t * sx, a[1] + t * sy

        k = min(range(n), key=lambda i: math.dist((anchor.x, anchor.y), nearest(ring[i], ring[(i + 1) % n])))
        if self.home is not None:
            join = nearest(ring[k], ring[(k + 1) % n])
            order = [ring[(k + 1 + j) % n] for j in range(n)] + [join]
            prevs = [join] + order[:-1]
        else:
            order = [ring[(k + 1 + j) % n] for j in range(n)] + [ring[(k + 1) % n]]
            prevs = [ring[k]] + order[:-1]
        print(f"joining route between ({ring[k][0]:.2f},{ring[k][1]:.2f}) and "
              f"({ring[(k + 1) % n][0]:.2f},{ring[(k + 1) % n][1]:.2f})", flush=True)
        return list(zip(prevs, order))

    def calibrate_motion(self):
        if self.motor is None:
            raise Abort("--calibrate-motion needs --arm")

        def fix_or_abort(label: str) -> Pose2D:
            fix = self.localize(3.0)
            if fix is None:
                raise Abort(f"calibration: no tag at '{label}'; face the car toward the tags")
            print(f"  {label}: {fix.text()}", flush=True)
            return fix

        p0 = fix_or_abort("start")
        t_fwd, reason = self.move("forward", 1.0)
        if reason:
            raise Abort(f"calibration forward stopped early: {reason}")
        p1 = fix_or_abort("after forward")
        t_left, _ = self.move("left", 1.2)
        p2 = fix_or_abort("after left turn")
        t_right, _ = self.move("right", 1.2)
        p3 = fix_or_abort("after right turn")

        mps = math.hypot(p1.x - p0.x, p1.y - p0.y) / t_fwd
        left = math.degrees(abs(wrap_rad(p2.yaw - p1.yaw))) / t_left
        right = math.degrees(abs(wrap_rad(p3.yaw - p2.yaw))) / t_right
        print(f"forward {mps:.3f} m/s  left {left:.1f} deg/s  right {right:.1f} deg/s", flush=True)
        self.motion.update(
            forward_mps=round(mps, 4),
            turn_left_dps=round(left, 2),
            turn_right_dps=round(right, 2),
            calibrated=True,
        )
        self.car["motion"] = self.motion
        self.car_path.write_text(json.dumps(self.car, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {self.car_path}", flush=True)
