"""Where the car is: tag fixes, the pose filter, ultrasonic wall corrections, home.

Mixin of RouteFollower (route_follow.py); the methods use self.* of the whole follower."""

from __future__ import annotations

from route_common import *  # noqa: F401,F403


class LocalizationMixin:
    def localize(self, timeout: float | None = None) -> Pose2D | None:
        self.stop()
        t0 = time.monotonic()
        if self.stop_event.wait(self.args.settle_s):
            raise Abort("stopped")
        fresh_after = t0 + self.args.settle_s
        deadline = fresh_after + (self.args.fix_wait_s if timeout is None else timeout)
        seen = {name: 0.0 for name in self.grabbers}
        fixes = []
        views = 0
        while time.monotonic() < deadline and not self.stop_event.is_set():
            for name, grabber in self.grabbers.items():
                frame, stamp = grabber.latest()
                if frame is None or stamp <= fresh_after or stamp == seen[name]:
                    continue
                seen[name] = stamp
                found, corners, ids = self.localizer.observe(name, frame)
                fixes.extend(found)
                if name == "front":
                    cv2.imwrite(str(self.debug_dir / "last_front.jpg"),
                                draw_overlay(frame, corners, ids, found, "front"))
                views += 1
            if fixes and views >= 2 * len(self.grabbers):
                break
            time.sleep(0.03)
        pose = self.localizer.fuse(fixes, time.monotonic())
        if pose is None and fixes:
            print(f"  no fix: {self.localizer.last_reject}; dead-reckoning", flush=True)
            self.heading_from_far_tags(fixes)
        return self.accept_fix(pose)

    def heading_from_far_tags(self, fixes) -> None:
        """No joint fix, but two or more far tags agree on the heading: take the heading, keep the dead-reckoned
        position. After a corner the dead-reckoned heading was 25 deg off while two tags said so for the whole lane."""
        if self.pose is None or not self.use_filter or not self.kalman.ready:
            return
        found = self.localizer.heading_only(fixes)
        if found is None:
            return
        yaw, spread, count = found
        error = math.degrees(wrap_rad(yaw - self.pose.yaw))
        if abs(error) < 2.0:
            return
        print(f"  {count} far tags agree on the heading ({math.degrees(yaw):+.0f}deg, dead-reckoned "
              f"{math.degrees(self.pose.yaw):+.0f}deg): taking the tags' heading", flush=True)
        self.pose.yaw = yaw
        self.pose.fixes = list(fixes)
        self.kalman.sync(self.pose.x, self.pose.y, yaw)
        self.kalman.P[2, 2] = min(float(self.kalman.P[2, 2]), math.radians(4.0) ** 2)
        self.learn_turn(yaw)

    def accept_fix(self, pose: Pose2D | None) -> Pose2D | None:
        """Fuse a tag fix into the pose: the filter (--hug), or the fixed jump limits of `accept_fix_hard`."""
        if not self.use_filter:
            return self.accept_fix_hard(pose)
        if pose is None:
            return None
        pairs = {(f.camera, f.tag_id) for f in pose.fixes}
        nearest = min((f.distance for f in pose.fixes), default=9.9)
        if self.pose is None or not self.kalman.ready:
            self.kalman.start(pose.x, pose.y, pose.yaw, len(pairs), nearest)
            self.jumps, self.jump_pose = 0, None
            return pose
        jump = math.hypot(pose.x - self.pose.x, pose.y - self.pose.y)
        if self.lane_check and jump > 0.10 and self.clearance(self.pose.x, self.pose.y, self.pose.yaw) < 0.06:
            print(f"  fix {pose.text()} jumped {jump * 100:.0f}cm while close to a wall; sweeping", flush=True)
            self.sweep_correct()
            return None
        self.kalman.sync(self.pose.x, self.pose.y, self.pose.yaw)
        before = self.pose
        if self.kalman.update_tag(pose.x, pose.y, pose.yaw, len(pairs), nearest) == "outlier":
            strong = len(pairs) >= 3 or any(f.distance < self.localizer.max_single_tag_m for f in pose.fixes)
            last = self.jump_pose
            same = last is not None and math.hypot(pose.x - last.x, pose.y - last.y) < self.args.max_jump_m
            self.jumps = self.jumps + 1 if same else 1
            self.jump_pose = pose
            if not strong and (self.parking or self.jumps < 4):
                sigma_xy, sigma_yaw = self.kalman.sigma()
                print(f"  ignoring fix {pose.text()}: outside what the filter expects "
                      f"(+-{sigma_xy * 100:.0f}cm, +-{sigma_yaw:.0f}deg)", flush=True)
                return None
            print("  far tags keep agreeing on a new pose; trusting them" if not strong
                  else f"  strong fix {pose.text()} far from the expected pose; trusting it", flush=True)
            self.kalman.start(pose.x, pose.y, pose.yaw, len(pairs), nearest)
            fused = pose
            if len(pairs) >= 2:
                self.learn_turn(pose.yaw)
                self.learn_strafe(pose.yaw)
        else:
            fused = Pose2D(float(self.kalman.x[0]), float(self.kalman.x[1]), float(self.kalman.x[2]), pose.stamp,
                           pose.fixes, pose.rms_px)
            if len(pairs) >= 2:
                self.learn_turn(pose.yaw)
                self.learn_strafe(pose.yaw)
        self.jumps, self.jump_pose = 0, None
        moved = math.hypot(fused.x - before.x, fused.y - before.y)
        turned = abs(math.degrees(wrap_rad(fused.yaw - before.yaw)))
        if moved > 0.05 or turned > 5.0:
            print(f"  fix moved the pose {moved * 100:.0f}cm / {turned:.0f}deg -> {fused.text()}", flush=True)
        # Two or more tags agreeing on a heading far from the lane are believed: after a corner the car can really be
        # 25 deg off (a turn that went too far), and forcing the lane heading back drives it into the wall.
        if self.lane_check and self.lane_yaw is not None and not self.detouring and len(pairs) < 2 \
                and abs(wrap_rad(fused.yaw - self.lane_yaw)) > math.radians(25.0):
            print(f"  fix {fused.text()}: heading {abs(math.degrees(wrap_rad(fused.yaw - self.lane_yaw))):.0f}deg "
                  f"off the lane; keeping its position, not its heading", flush=True)
            fused.yaw = before.yaw
            self.kalman.x[2] = before.yaw
        return fused

    def accept_fix_hard(self, pose: Pose2D | None) -> Pose2D | None:
        """Drop a weak fix that jumps away from the dead-reckoned pose.

        Strong fixes (3+ tags, or one close tag) always pass. Weak ones pass only after
        4 in a row agree on the same new pose, and never while parking.
        """
        if pose is None or self.pose is None:
            return pose
        jump = math.hypot(pose.x - self.pose.x, pose.y - self.pose.y)
        turn = abs(math.degrees(wrap_rad(pose.yaw - self.pose.yaw)))
        if self.lane_check and jump > 0.10 and self.clearance(self.pose.x, self.pose.y, self.pose.yaw) < 0.06:
            print(f"  fix {pose.text()} jumped {jump * 100:.0f}cm while close to a wall; sweeping", flush=True)
            self.sweep_correct()
            return None
        if jump > self.args.max_jump_m or turn > self.args.max_jump_deg:
            strong = len({(f.camera, f.tag_id) for f in pose.fixes}) >= (2 if self.detouring else 3) or \
                any(f.distance < self.localizer.max_single_tag_m for f in pose.fixes)
            if not strong:
                last = self.jump_pose
                same = last is not None and math.hypot(pose.x - last.x, pose.y - last.y) < self.args.max_jump_m
                self.jumps = self.jumps + 1 if same else 1
                self.jump_pose = pose
                if self.parking or self.jumps < 4:
                    print(f"  ignoring fix {pose.text()}: jumped {jump * 100:.0f}cm / {turn:.0f}deg "
                          f"from the expected pose", flush=True)
                    return None
                print("  far tags keep agreeing on a new pose; trusting them", flush=True)
        self.jumps, self.jump_pose = 0, None
        if self.lane_check and self.lane_yaw is not None and not self.detouring \
                and abs(wrap_rad(pose.yaw - self.lane_yaw)) > math.radians(25.0):
            # Far down a lane the tags still locate the car well but the heading flips; keep the
            # position and the lane heading instead of steering off at the bad yaw.
            print(f"  fix {pose.text()}: heading {abs(math.degrees(wrap_rad(pose.yaw - self.lane_yaw))):.0f}deg "
                  f"off the lane; keeping its position, not its heading", flush=True)
            pose.yaw = self.pose.yaw
            return pose
        return pose

    def refresh_pose(self, blind: int, full_turn: bool = True) -> int:
        fix = self.localize()
        if fix is not None:
            self.pose = fix
            return 0
        blind += 1
        if blind <= self.args.max_blind_segments and self.pose is not None:
            return blind
        if not full_turn and self.pose is not None:
            # Close to the target a wrong heading is costly: glance left and right instead of
            # spinning a full turn.
            print("  no tag for too long; glancing left and right", flush=True)
            held = self.pose.yaw
            for glance in (1, -2, 1):
                self.turn_by(glance * math.radians(15.0), min_s=0.05)
                fix = self.localize()
                if fix is not None:
                    self.pose = fix
                    return 0
            self.turn_by(wrap_rad(held - self.pose.yaw), min_s=0.05)
            return blind
        print("  no tag for too long; searching", flush=True)
        fix = self.search()
        if fix is None:
            raise Abort("lost: no tag visible after a full turn")
        self.pose = fix
        return 0

    def set_home(self):
        """Home = where the car was placed; it must be known before the car moves.

        With a map home (arena_map.json "home" or --home) a nearby start fix is used as the
        parking target, since the tags then read the same way again when the car is back there.
        """
        raw = self.args.home or self.arena.get("home")
        known = None
        if raw is not None:
            x, y, yaw = (float(v) for v in (raw.split(",") if isinstance(raw, str) else raw))
            known = Pose2D(x, y, math.radians(yaw))
        if self.pose is None:
            for attempt in range(5 if known is None else 2):
                print(f"no tag at start; looking again without moving ({attempt + 1})", flush=True)
                self.pose = self.localize(3.0)
                if self.pose is not None:
                    break
        measured = None
        if self.pose is not None:
            first, self.pose = self.pose, None
            fixes = [first] + [f for f in (self.localize() for _ in range(4)) if f is not None]
            ref = fixes[0].yaw
            measured = Pose2D(st.median(f.x for f in fixes), st.median(f.y for f in fixes),
                              wrap_rad(ref + st.median(wrap_rad(f.yaw - ref) for f in fixes)))
            self.pose = measured
        if known is None:
            if measured is None:
                raise Abort("--return-home needs a tag fix at the start pose; place the car so a camera "
                            "sees two tags (or one closer than 0.8 m), or set \"home\" in arena_map.json")
            self.pose, self.home = measured, Pose2D(measured.x, measured.y, measured.yaw)
            source = "measured"
        elif measured is None:
            self.pose, self.home = Pose2D(known.x, known.y, known.yaw), known
            source = "map home (no tag fix here)"
        else:
            off = math.hypot(measured.x - known.x, measured.y - known.y)
            turn = abs(math.degrees(wrap_rad(measured.yaw - known.yaw)))
            print(f"map home x={known.x:.2f} y={known.y:.2f} yaw={known.yaw_deg:+.0f}deg; tags read "
                  f"{off * 100:.0f}cm / {turn:.0f}deg away", flush=True)
            if off <= 0.15 and turn <= 20.0:
                # The car starts where it was parked or put; the map home is where it is to come back to every lap,
                # and the wall at its right measures that to a centimetre (the corner tags do not).
                fit = self.fit_in_arena(measured, HOME_WALL_MARGIN_M)
                if math.hypot(fit.x - measured.x, fit.y - measured.y) > 0.01:  # a start fix through the wall: not believed
                    self.pose, self.home = Pose2D(known.x, known.y, known.yaw), Pose2D(known.x, known.y, known.yaw)
                else:
                    self.pose, self.home = measured, Pose2D(known.x, known.y, known.yaw)
                self.wall_home = True
                source = "map home (the car was put near it; parking measures the wall)"
            else:
                self.pose, self.home = Pose2D(known.x, known.y, known.yaw), known
                source = "map home (start fix disagrees; ignoring it)"
        # A tag fix in the corner can be 10 cm off (the tags are close and seen at a steep angle). A home that puts
        # the body through a wall is not a place the car can park: parking would back into the wall for ever.
        self.home_open_loop = self.wall_home  # corner tags are bad: stop on the lane start, then strafe and back in
        fitted = self.fit_in_arena(self.home, HOME_WALL_MARGIN_M)
        if math.hypot(fitted.x - self.home.x, fitted.y - self.home.y) > 0.01:
            if known is not None:
                print(f"home pose x={self.home.x:.2f} y={self.home.y:.2f} would put the body through a wall (the "
                      f"tags are too close in the corner); home is the map's x={known.x:.2f} y={known.y:.2f}, and "
                      f"parking goes to the lane start first and backs in from there", flush=True)
                self.home = Pose2D(known.x, known.y, known.yaw)
                self.home_open_loop = True
            else:
                print(f"home pose x={self.home.x:.2f} y={self.home.y:.2f} would put the body through a wall; "
                      f"using x={fitted.x:.2f} y={fitted.y:.2f}", flush=True)
                self.home = fitted
            if self.pose is not None:
                self.pose = Pose2D(self.home.x, self.home.y, self.pose.yaw, self.pose.stamp, self.pose.fixes,
                                   self.pose.rms_px)
        print(f"home pose x={self.home.x:.2f} y={self.home.y:.2f} yaw={self.home.yaw_deg:+.0f}deg "
              f"[{source}]", flush=True)

    def fit_in_arena(self, pose: Pose2D, margin: float) -> Pose2D:
        """`pose` moved the shortest way until the whole body is `margin` inside the arena walls."""
        w, h = (float(v) for v in self.arena["arena_size_m"])
        hl, hw = self.body_half
        c, s = abs(math.cos(pose.yaw)), abs(math.sin(pose.yaw))
        ex, ey = c * hl + s * hw + margin, s * hl + c * hw + margin
        return Pose2D(min(max(pose.x, ex), w - ex), min(max(pose.y, ey), h - ey), pose.yaw, pose.stamp, pose.fixes,
                      pose.rms_px)

    def averaged_fix(self, n: int) -> Pose2D | None:
        fixes = [f for f in (self.localize() for _ in range(n)) if f is not None]
        if not fixes:
            return None
        yaw = math.atan2(sum(math.sin(f.yaw) for f in fixes), sum(math.cos(f.yaw) for f in fixes))
        return Pose2D(sum(f.x for f in fixes) / len(fixes), sum(f.y for f in fixes) / len(fixes), yaw,
                      fixes[-1].stamp, fixes[-1].fixes, fixes[-1].rms_px)

    def sweep_correct(self):
        """Replace the sideways part of the pose with what the ultrasonic sweep measures.

        Only beams that look sideways and land on a wall running along the heading measure the
        sideways error: moving the car left by d shortens such a beam at angle a by d / sin(a).
        A reading far from the wall the pose predicts is an obstacle (or a glancing echo) and is
        left out.
        """
        found = self.sweep()
        if self.pose is None or len(found) < 2:
            return
        p = self.pose
        along_x = abs(math.cos(p.yaw)) > math.cos(math.radians(30.0))  # heading runs along a y=const wall
        shifts = []
        for angle, cm in found.items():
            if abs(angle) < 45:
                continue
            beam = p.yaw + math.radians(angle)
            ox = p.x + SONIC_PIVOT_M * math.cos(p.yaw) + SONIC_ARM_M * math.cos(beam)
            oy = p.y + SONIC_PIVOT_M * math.sin(p.yaw) + SONIC_ARM_M * math.sin(beam)
            predicted, axis = self.wall_hit(ox, oy, beam)
            if axis != ("y" if along_x else "x") or abs(predicted - cm / 100.0) > 0.12:
                continue
            shifts.append((predicted - cm / 100.0) * math.sin(math.radians(angle)))
        if not shifts:
            print("    sweep did not match a side wall; keeping the camera pose", flush=True)
            return
        shift = st.median(shifts)
        p.x -= shift * math.sin(p.yaw)
        p.y += shift * math.cos(p.yaw)
        print(f"    sweep moved the pose {shift * 100:+.0f}cm sideways -> {p.text()}", flush=True)

    def sweep(self) -> dict[int, float]:
        """Range left, ahead and right; returns the angles that answered, in degrees."""
        found = {}
        try:
            for angle in (90, 60, 30, 0, -30, -60, -90):  # +-90 hit the side wall square on: the surest echo
                cm = self.look(angle)
                if cm is not None:
                    found[angle] = cm
                print(f"    sweep {angle:+d}deg: {cm if cm is not None else '-'}cm", flush=True)
        finally:
            if self.servo is not None:
                self.servo.set_pan(self.home_pan)
        return found

    def search(self) -> Pose2D | None:
        self.lane_check = False
        steps = int(math.ceil(360.0 / self.args.search_step_deg))
        for i in range(steps):
            print(f"  search {i + 1}/{steps}: turning left {self.args.search_step_deg:.0f}deg", flush=True)
            if self.pose is None:
                self.move("left", self.args.search_step_deg / self.turn_rate("left"))
            else:
                self.turn_by(math.radians(self.args.search_step_deg))
            fix = self.localize()
            if fix is not None:
                return fix
        return None

    def snap_to_wall(self):
        """At a lane end the wall is dead ahead, and the ultrasonic reads the true distance to it.

        Dead reckoning drifts along the lane (5 % of the distance, and tags are often too far to fix it);
        the wall does not move. Heading square to a wall and a reading within 30 cm that is within 12 cm of
        what the pose predicts: move the pose along the heading to agree with the ultrasonic.
        """
        if self.sensors is None or self.pose is None:
            return
        p = self.pose
        if max(abs(math.cos(p.yaw)), abs(math.sin(p.yaw))) < 0.95:
            return
        readings = [r for r in (self.sensors.read_sonic() for _ in range(3)) if r is not None]
        if not readings:
            return
        measured = st.median(readings) / 100.0
        err = self.wall_ahead() - measured
        if measured > 0.30 or abs(err) > 0.12:
            return
        if self.use_filter and self.kalman.ready:
            self.kalman.sync(p.x, p.y, p.yaw)
            if self.kalman.update_wall(SONIC_AHEAD_M, 0.0, measured):
                moved = math.hypot(self.kalman.x[0] - p.x, self.kalman.x[1] - p.y)
                p.x, p.y, p.yaw = (float(v) for v in self.kalman.x)
                print(f"    wall {measured * 100:.0f}cm ahead; the filter moved the pose {moved * 100:.0f}cm -> "
                      f"{p.text()}", flush=True)
            return
        p.x += err * math.cos(p.yaw)
        p.y += err * math.sin(p.yaw)
        print(f"    wall {measured * 100:.0f}cm ahead; pose moved {err * 100:+.0f}cm along the heading -> {p.text()}",
              flush=True)
