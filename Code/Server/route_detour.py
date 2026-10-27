"""Going around an obstacle in a lane: look, close in, strafe, pass, return.

Mixin of RouteFollower (route_follow.py); the methods use self.* of the whole follower."""

from __future__ import annotations

from route_common import *  # noqa: F401,F403


class DetourMixin:
    def detour(self, target: tuple[float, float, float], gap: float):
        """Go around an obstacle (see `_detour`); tag headings are trusted meanwhile.

        A mecanum strafe turns the car a little (the wheels are not equally strong). The lane rule that
        refuses a tag heading far from the lane is there for far tags that flip; here the car really
        can be 25 deg off after a long strafe, and strafing on at the lane heading drives it diagonally.
        """
        self.detouring = True
        try:
            self._detour(target, gap)
        finally:
            self.detouring = False

    def _detour(self, target: tuple[float, float, float], gap: float):
        """Go around an obstacle in a --hug lane with mecanum strafes, keeping the lane heading.

        1. Strafe toward the arena middle until the ultrasonic no longer sees it, then half the body
           width more: the beam runs down the car's centre line.
        2. Drive past it. Its far side is found with the ultrasonic turned to the wall side (a block
           is 10 or 20 cm deep); then the tail, which trails the ultrasonic by 26 cm, must clear it.
        3. Strafe back onto the lane, checking the side IR on the way.
        Beside the block the car does not turn: the corners would swing into it.
        """
        yaw = target[2]
        w, h = (float(v) for v in self.arena["arena_size_m"])
        c, s = math.cos(yaw), math.sin(yaw)
        middle = -(w / 2 - self.pose.x) * s + (h / 2 - self.pose.y) * c
        out = 1.0 if middle > 0 else -1.0  # +1: the arena middle is to the left of the lane
        side_name = "left" if out > 0 else "right"
        wall_side = "right" if out > 0 else "left"
        print(f"  obstacle {gap * 100:.0f}cm ahead in the lane; going around it on the {side_name}", flush=True)

        def off_lane() -> float:
            dx, dy = self.pose.x - target[0], self.pose.y - target[1]
            return out * (-dx * s + dy * c)

        def along() -> float:
            return (self.pose.x - target[0]) * c + (self.pose.y - target[1]) * s

        out_keys = ("wheel_left_front", "wheel_left_rear") if out > 0 else ("wheel_right_front", "wheel_right_rear")

        def side_blocked() -> bool:
            return self.sensors is not None and any(self.sensors.read_ir().get(k) for k in out_keys)

        def sidestep(distance: float):
            for _ in range(8):
                left = distance - off_lane()
                if abs(left) < 0.01:
                    return
                if side_blocked():
                    print(f"    {side_name} IR lit; staying {off_lane() * 100:.0f}cm off the lane", flush=True)
                    return
                # Short steps: an overshoot here swings the body further into the middle of the arena.
                if not self.strafe(out * math.copysign(min(abs(left), 0.03), left), ahead_cap=False):
                    raise Abort("detour: a wall blocks the strafe")
                self.record()
                self.save_map()

        passing: set = set()  # ids (in the obstacle map) of the block(s) this detour goes around

        def forward_to(goal: float) -> bool:
            """Drive along the lane to `goal`; False when a sensor or wall stopped it short."""
            for _ in range(8):
                self.face(yaw, 3.0)  # 10 deg over a 50 cm pass is 9 cm sideways
                step = self.safe_step(min(self.args.max_segment_m, goal - along()))
                if step < 0.01:
                    break
                if self.scanner is not None:
                    # Out beside the obstacle the car is no longer on its lane: look before each step, another
                    # block (e.g. one standing out from the next wall) may be in the way of the shifted path.
                    ahead_gap = self.camera_gap(exclude_blocks=tuple(passing))
                    if ahead_gap is not None:
                        step = min(step, ahead_gap - 0.12, CAM_STEP_M)
                        if step < 0.02:
                            print(f"    another obstacle {ahead_gap * 100:.0f}cm ahead while passing; stopping", flush=True)
                            break
                _elapsed, reason = self.move("forward", step / float(self.motion["forward_mps"]))
                self.refresh_pose(0, full_turn=False)
                self.record()
                self.save_map()
                if reason:
                    print(f"    stopped while passing ({reason})", flush=True)
                    break
            return along() >= goal - 0.02

        # 0. Right up against it a strafe would scrape the nose along it: back off to 6 cm first.
        if gap < 0.05:
            back = self.safe_step(-(0.06 - gap))
            if back < -0.005:
                print(f"    only {gap * 100:.0f}cm in front; reversing {-back * 100:.0f}cm first", flush=True)
                self.move("reverse", -back / float(self.motion["forward_mps"]))
                gap -= back

        start, detected = off_lane(), along()
        camera_blind = False  # the camera sees nothing where the ultrasonic found the obstacle: not to be believed
        if self.scanner is not None:
            # 1. The camera gives the obstacle's edge: strafe exactly as far as the body needs to clear it by
            # CLEAR_MARGIN_M, look again, and repeat if the first estimate was short.
            nose, half = self.body_half[0], self.body_half[1]
            for attempt in range(6):
                self.face(yaw, 2.0)  # a strafe turns the car; the path ahead is judged square to the lane
                obstacle, _wall = self.look_around()
                if attempt == 0:
                    sweep_need = self.wide_sweep(out, nose, half)
                    for block in self.obstacles.blocks:
                        if block.get("last_scan") == self.obstacles.scans:
                            print(f"    {self.obstacles.describe(block, target)}", flush=True)
                            passing.add(block["id"])
                judged = obstacle
                if gap < BLIND_SLANT_GAP_M:
                    # Right up against the block the camera is blind to its foot: the slanted looks (pan +-30) only
                    # give "at least this near" points that are stretched sideways (10 cm too far, a 19 cm strafe
                    # for a 9 cm need). The straight look has the true edge.
                    straight = [v for v in self.last_views if abs(v.pan) < 1.0]
                    pts = [v.points[~np.isnan(v.points[:, 0])] for v in straight] + [v.close for v in straight if len(v.close)]
                    if pts:
                        centre_only, _w = split_walls(np.vstack(pts), self.pose, self.arena["arena_size_m"])
                        if len(centre_only):
                            judged = centre_only
                need = shift_to_clear(judged, out, nose, half, CLEAR_MARGIN_M)
                if need <= 0.012 and attempt == 0:
                    # The ultrasonic put an obstacle ahead and the looks see none (a dark grey block can drop out of the
                    # foot-line finder at 8 cm: the car then drove on into it). Take the full sweep's estimate, and
                    # with that empty too, find the edge with the ultrasonic instead.
                    if sweep_need > 0.012:
                        print(f"    the looks see a clear path but the ultrasonic does not; using the full sweep: "
                              f"{sweep_need * 100:.0f}cm", flush=True)
                        need = sweep_need
                    else:
                        print("    the camera sees nothing where the ultrasonic found an obstacle; "
                              "going around by the ultrasonic", flush=True)
                        camera_blind = True
                        break
                if need <= 0.012:
                    break
                if off_lane() - start + need > self.args.detour_max_m:
                    raise Abort(f"detour: the path is blocked up to {(off_lane() - start + need) * 100:.0f}cm "
                                f"off the lane; no way around")
                # The strafe sweeps the side of the car; the camera looks ahead. Ask the ultrasonic, turned to
                # the side the car is about to move to, whether something stands there.
                reading, wall = self.side_range(90.0 * out)
                room = reading + SONIC_ARM_M if reading is not None else None  # from the car's centre line
                if room is not None and reading < wall - 0.08 and room < half + need + 0.05:
                    raise Abort(f"detour: something {room * 100:.0f}cm to the {side_name}, where the strafe of "
                                f"{need * 100:.0f}cm would go; no way around")
                print(f"    path blocked; strafing {need * 100:.0f}cm {side_name}", flush=True)
                goal = off_lane() + need
                while goal - off_lane() > 0.01:
                    if side_blocked():
                        raise Abort(f"detour: something on the {side_name} side too; no way around")
                    self.face(yaw, 2.0)
                    if not self.strafe(out * min(goal - off_lane(), 0.04), ahead_cap=False):
                        raise Abort("detour: a wall blocks the strafe")
                    self.record()
                    self.save_map()
            else:
                raise Abort("detour: the camera still sees the path blocked after 6 looks")
            if not camera_blind:
                print(f"    path clear, {(off_lane() - start) * 100:.0f}cm off the lane", flush=True)
        if self.scanner is None or camera_blind:
            ahead = gap
            while True:
                if off_lane() - start > self.args.detour_max_m:
                    raise Abort(f"detour: obstacle still ahead {self.args.detour_max_m * 100:.0f}cm off the lane")
                if side_blocked():
                    raise Abort(f"detour: something on the {side_name} side too; no way around")
                self.face(yaw)
                # Once the beam starts to slip off the obstacle's edge, smaller steps: every cm out here
                # is a cm the car swings into the middle of the arena.
                if not self.strafe(out * (0.04 if ahead is None or ahead < gap + 0.03 else 0.03), ahead_cap=False):
                    raise Abort("detour: a wall blocks the strafe")
                self.record()
                self.save_map()
                ahead = self.obstacle_ahead(edges=False)  # the side beams would keep finding this very block
                print(f"    {(off_lane() - start) * 100:.0f}cm out; ahead "
                      f"{'clear' if ahead is None else f'{ahead * 100:.0f}cm'}", flush=True)
                if ahead is None or ahead > gap + 0.15:
                    break
            # Clear means the edge of the beam has left the obstacle, so its edge is at least that far off the
            # centre line. SONIC_HALF_DEG is narrower than the real cone, which only ever makes this cautious.
            need = off_lane() - gap * math.tan(math.radians(SONIC_HALF_DEG)) + self.body_half[1] + 0.02
            sidestep(need)
            if off_lane() < need - 0.015:
                raise Abort(f"detour: only {off_lane() * 100:.0f}cm of the {need * 100:.0f}cm needed beside the "
                            "obstacle is free; no way around")

        # 2. Past it. `near` is its near face along the lane; the side beam is SONIC_PIVOT_M ahead of the
        # centre and a few cm wide, so "clear" there puts the far face at most 4 cm behind the beam.
        near = detected + SONIC_AHEAD_M + gap
        pan = -90.0 * out
        far = None
        nudges = 0
        probe = near + 0.10 + 0.04
        while far is None and probe <= near + self.block_long + 0.08:
            if not forward_to(probe - SONIC_PIVOT_M):
                break
            reading, wall = self.side_range(pan)
            beside = reading is not None and reading < wall - 0.08
            print(f"    {wall_side} side at +{(along() + SONIC_PIVOT_M - near) * 100:.0f}cm past its near face: "
                  f"ultrasonic {'-' if reading is None else f'{reading * 100:.0f}cm'}, wall {wall * 100:.0f}cm"
                  f"{' (still beside it)' if beside else ''}", flush=True)
            if beside and reading < SIDE_CLEAR_MIN_M and nudges < 3:
                # The camera set the strafe from the block's face; beside it the ultrasonic says how close the body
                # really is (the block can stand a little crooked, or the pose be a few cm off): never pass closer
                # than 2.5 cm, strafe out the difference and measure again from the same spot.
                nudges += 1
                step_out = min(SIDE_CLEAR_MIN_M - reading, 0.04)
                print(f"    only {(reading - (self.body_half[1] - SONIC_ARM_M)) * 100:.0f}cm between the body and "
                      f"the block; strafing {step_out * 100:.0f}cm further out", flush=True)
                if not self.strafe(out * step_out, ahead_cap=False):
                    raise Abort("detour: no room to give the obstacle more room")
                continue
            if not beside:
                far = along() + SONIC_PIVOT_M - 0.04
            probe += 0.05
        if far is None:
            far = near + self.block_long
            print(f"    could not see its far side; assuming {self.block_long * 100:.0f}cm deep", flush=True)
        clear = far + self.body_half[0] + PASS_MARGIN_M
        room = self.outer_corridor(out) if self.scanner is None else None  # the camera covers it
        if room is not None and clear - along() > room:
            raise Abort(f"detour: something {room * 100:.0f}cm ahead on the {side_name} edge of the path; "
                        f"passing needs {(clear - along()) * 100:.0f}cm")
        if not forward_to(clear):
            raise Abort(f"detour: stopped {(clear - along()) * 100:.0f}cm short of clearing the obstacle; "
                        "strafing back would hit it")

        # 3. Back onto the lane, whichever side of it the car is on now; a side IR lighting up while still
        # well off the lane is the obstacle, so go a little further first.
        ir_keys = ("wheel_right_front", "wheel_right_rear") if out > 0 else ("wheel_left_front", "wheel_left_rear")
        for _ in range(14):
            err = off_lane()  # + = toward the arena middle
            if abs(err) <= self.args.lane_tol_m:
                break
            p = self.pose
            wall_gap = self.wall_distance(p.x, p.y, p.yaw - out * math.pi / 2.0) - self.body_half[1]  # body edge to wall
            if err > 0.10 and wall_gap > 0.12 and self.sensors is not None \
                    and any(self.sensors.read_ir().get(k) for k in ir_keys) and self.ir_is_obstacle(-90.0 * out):
                print(f"    {wall_side} IR sees the obstacle; forward 6cm before strafing back", flush=True)
                if not forward_to(along() + 0.06):
                    raise Abort("detour: obstacle beside the car and no room to go further")
                continue
            self.face(yaw, 2.0)
            if not self.strafe(-out * max(min(err, 0.06), -0.06), ahead_cap=False):
                break
            self.record()
            self.save_map()
        print(f"    back on the lane ({off_lane() * 100:+.0f}cm)", flush=True)

    def ir_is_obstacle(self, pan: float) -> bool:
        """A wheel IR on the wall side is lit: is something within reach of the body there, or is it the wall?

        The IR modules can see a wall 17-20 cm away (one stayed lit for good with the car 20 cm from the wall, and the
        detour took it for the block and gave up). The ultrasonic turned to that side tells: a block the IR sees
        is within ~8 cm of the body, a reading of 15 cm or more from the pivot is farther than that.
        """
        reading, _wall = self.side_range(pan)
        return reading is not None and reading < 0.15

    def beyond_spot(self, gap: float | None, target) -> bool:
        """True when the obstacle `gap` ahead (from the ultrasonic face) lies past `target`: the body fits at it."""
        if gap is None:
            return False
        p = self.pose
        ahead = (target[0] - p.x) * math.cos(p.yaw) + (target[1] - p.y) * math.sin(p.yaw)
        return gap + SONIC_AHEAD_M - self.body_half[0] > ahead + 0.02

    def face(self, yaw: float, tol_deg: float = 6.0):
        err = wrap_rad(yaw - self.pose.yaw)
        if abs(err) > math.radians(tol_deg):
            self.turn_by(err, min_s=0.05)
