"""What is around the car: ultrasonic, IR, the gimbal camera's looks, the stop rules.

Mixin of RouteFollower (route_follow.py); the methods use self.* of the whole follower."""

from __future__ import annotations

from route_common import *  # noqa: F401,F403


class PerceptionMixin:
    # ---- sensing --------------------------------------------------------

    def look(self, pan_deg: float):
        """Point the ultrasonic at `pan_deg` relative to straight ahead (positive = left) and read it."""
        if self.servo is None or self.sensors is None:
            return None
        from servo import servo_pan_for_bearing

        # The servo is not 1:1 and differs left and right (gimbal_home.json right_90_pan / left_90_pan).
        self.servo.set_pan(servo_pan_for_bearing(self.home_pan, self.right_90_pan, pan_deg, self.left_90_pan))
        if self.stop_event.wait(0.35):
            raise Abort("stopped")
        return self.sensors.read_sonic()

    def safety_reason(self) -> str | None:
        """Why a forward segment must stop right now, or None."""
        sonic = self.sensors.read_sonic()
        if sonic is None and not self.args.no_sonic:
            self.sonic_misses += 1
            if self.sonic_misses >= 5:
                return "sonic_missing"
        else:
            self.sonic_misses = 0
        stop_cm = self.args.stop_cm if self.closing_cm is None else min(self.args.stop_cm, self.closing_cm)
        if sonic is not None and sonic < stop_cm and not self.wall_echo(sonic):
            return f"sonic_{sonic:.0f}cm"
        ir = self.sensors.read_ir()
        if self.closing_ir_after is not None and self.travel >= self.closing_ir_after:
            ir = {**ir, "front": 0, "front_left": 0, "front_right": 0}  # expected to light now: closing in on purpose
        near = sonic is None or sonic < self.args.front_ir_gate_cm
        if near and (ir["front"] or ir["front_left"] or ir["front_right"]) \
                and not (sonic is not None and sonic > self.args.front_ir_gate_cm - 6.0):
            return "ir_front"
        if self.args.hug and self.args.detour and (ir["front_left"] or ir["front_right"]):
            # A block that covers only the edge of the path is missed by the centre beam; the corner
            # IR is all that sees it. A false trigger at the wall ahead is taken as arriving there.
            return "ir_front"
        if not self.args.camera_safety:
            return None
        frame, stamp = self.grabbers["front"].latest()
        if frame is not None and time.monotonic() - stamp < 1.0:
            mid = classic_corridor_score(cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA))["mid"]
            self.front_risk = 0.4 * mid + 0.6 * self.front_risk
            # A far foam wall also fills the fisheye; only trust the camera near the wall.
            if self.front_risk >= self.args.near_score and sonic is not None and sonic < self.args.creep_cm:
                return f"camera_{self.front_risk:.2f}"
        return None

    def wall_echo(self, sonic_cm: float) -> bool:
        """In --hug lanes with detours, a reading that matches the wall straight ahead is not a reason to
        stop until 8 cm: safe_step already keeps the body off that wall, and backing away from it
        could put the car into an obstacle it has just gone around."""
        if not (self.args.hug and self.args.detour) or self.pose is None or sonic_cm < 8.0:
            return False
        return abs(sonic_cm / 100.0 - (self.wall_ahead() - self.travel)) < 0.06

    def wall_ahead(self) -> float:
        """Distance from the ultrasonic face to the wall straight ahead, by the pose."""
        p = self.pose
        return self.wall_distance(p.x + SONIC_AHEAD_M * math.cos(p.yaw), p.y + SONIC_AHEAD_M * math.sin(p.yaw), p.yaw)

    def aim(self, pan_deg: float):
        """Turn the gimbal to `pan_deg` (+ = left) through its calibrated angles and wait until it is there."""
        from servo import servo_pan_for_bearing

        self.servo.set_pan(servo_pan_for_bearing(self.home_pan, self.right_90_pan, pan_deg, self.left_90_pan))
        if self.stop_event.wait(0.4):
            raise Abort("stopped")

    def look_around(self, pans=CAM_PANS) -> tuple[np.ndarray, np.ndarray]:
        """One camera look from where the car stands: (obstacle points, wall points), car frame.

        The gimbal turns to each angle, the foot line of the foam is found in a fresh frame and put on
        the floor (foot_cam.py). Points on the arena walls are split off (the pose says where they are);
        the rest go into the obstacle map, which draw_map shows.
        """
        views = self.scanner.scan(self.aim, pans)
        parts = [v.points[~np.isnan(v.points[:, 0])] for v in views] + [v.close for v in views if len(v.close)]
        points = np.vstack(parts) if parts else np.zeros((0, 2))
        obstacle, wall = split_walls(points, self.pose, self.arena["arena_size_m"])
        for block in self.obstacles.add(obstacle, wall, self.pose):
            if block["hits"] == 1:
                print(f"    new obstacle {self.obstacles.describe(block)} (pose {self.pose.text()})", flush=True)
        self.last_views = views
        self.save_look(views)
        self.log_look(views)
        return obstacle, wall

    def wide_sweep(self, out: float, nose: float, half: float) -> float:
        """Right up against the block: the gimbal sweeps from the far left to the far right once.

        Only a measurement so far: the foot points of the whole sweep (no "at least this near" points) say how far
        the block reaches sideways, which is printed beside the strafe the normal looks asked for, and every frame is
        saved (looks/). Its estimate is used only when the normal looks see a clear path where the ultrasonic found
        an obstacle.
        """
        views = self.scanner.scan(self.aim, WIDE_PANS)
        self.log_look(views)
        feet = [v.points[~np.isnan(v.points[:, 0])] for v in views]
        points = np.vstack(feet) if feet else np.zeros((0, 2))
        obstacle, _wall = split_walls(points, self.pose, self.arena["arena_size_m"])
        need = shift_to_clear(obstacle, out, nose, half, CLEAR_MARGIN_M) if len(obstacle) else 0.0
        print(f"    full sweep ({len(obstacle)} foot points on the block): would strafe {need * 100:.0f}cm", flush=True)
        return need

    def log_look(self, views) -> None:
        """Every look kept for replay: the frames, and the pose they were taken at (debug_dir/looks)."""
        if not self.args.save_looks or not any(v.frame is not None for v in views):
            return
        self.look_no += 1
        folder = self.debug_dir / "looks"
        folder.mkdir(exist_ok=True)
        for v in views:
            if v.frame is not None:
                cv2.imwrite(str(folder / f"{self.look_no:03d}_pan{v.pan:+03.0f}.jpg"), v.frame)
        p = self.pose
        with (folder / "looks.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"n": self.look_no, "pose": [round(p.x, 4), round(p.y, 4), round(math.degrees(p.yaw), 2)],
                                "pans": [v.pan for v in views], "time": time.strftime("%H:%M:%S")}) + "\n")

    def save_look(self, views) -> None:
        """The last look's frames with the foot lines on them: debug_dir/last_look.jpg."""
        frames = [v for v in views if v.frame is not None]
        if not frames or not hasattr(self.scanner, "cam"):
            return
        tiles = []
        for v in frames:
            tile = cv2.resize(self.scanner.cam.draw(v), (320, 240))
            cv2.putText(tile, f"pan {v.pan:+.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(tile)
        cv2.imwrite(str(self.debug_dir / "last_look.jpg"), np.hstack(tiles))

    def camera_gap(self, reason: str | None = None, exclude: np.ndarray | None = None) -> float | None:
        """Gap (m, from the ultrasonic face) to the nearest thing in the car's path that is not the wall.

        The camera looks at the path ahead (obstacle_map.path_gap: the body's width plus PATH_MARGIN_M,
        from the nose on); the ultrasonic straight ahead and a front IR stop count too. The pose says
        where the wall ahead is: what the camera sees there is the wall, whatever the filter said.
        """
        if self.sensors is None or self.pose is None:
            return None
        nose, half = self.body_half[0], self.body_half[1] + PATH_MARGIN_M
        obstacle, _wall = self.look_around()
        if exclude is not None and len(exclude) and len(obstacle):
            # the block being gone around is still beside-ahead of the car, only a few cm clear of its path
            world = to_world(obstacle, self.pose)
            near = np.hypot(world[:, None, 0] - exclude[None, :, 0], world[:, None, 1] - exclude[None, :, 1]).min(axis=1)
            obstacle = obstacle[near > 0.10]
        ahead = path_gap(obstacle, nose, half)
        gap = None if ahead is None else ahead + (nose - SONIC_AHEAD_M)
        readings = [r for r in (self.sensors.read_sonic() for _ in range(3)) if r is not None]
        if readings:
            sonic = st.median(readings) / 100.0
            gap = sonic if gap is None else min(gap, sonic)
        if reason == "ir_front":
            gap = min(gap if gap is not None else 1.0, 0.08)
        if gap is not None and self.wall_ahead() - gap <= WALL_ECHO_M:
            gap = None  # the wall ahead; the pose is off by up to this much after a stretch without tags
        return gap

    def obstacle_ahead(self, reason: str | None = None, edges: bool = True) -> float | None:
        """Gap (m, from the ultrasonic face) to something in the car's path that is not the wall, or None.

        The pose says where the wall ahead should be; a reading 10 cm or more short of it is an
        obstacle. The beam straight ahead misses a block that covers only the edge of the path (a
        block standing out from the wall reaches only the wall-side 6 cm of the lane), and the corner
        IR does not see dark foam, so with `edges` the ultrasonic also looks EDGE_PAN_DEG to each side: a
        return inside the body's width (plus 2 cm) counts. A slanted block can bounce the beam away,
        so a front IR stop counts as one 8 cm off.
        """
        if self.scanner is not None:
            return self.camera_gap(reason)
        if self.sensors is None or self.pose is None:
            return None
        readings = [r for r in (self.sensors.read_sonic() for _ in range(3)) if r is not None]
        gap = st.median(readings) / 100.0 if readings else None
        if reason == "ir_front":
            gap = min(gap if gap is not None else 1.0, 0.08)
        if gap is not None and self.wall_ahead() - gap <= 0.10:
            gap = None  # the wall ahead (the IR also lights at the end of every lane)
        if edges and self.servo is not None:
            for pan in (EDGE_PAN_DEG, -EDGE_PAN_DEG):
                reading, wall = self.side_range(pan)
                if reading is None or abs(reading - wall) < 0.06:
                    continue  # nothing, or the wall that beam points at
                reach = SONIC_ARM_M + reading
                side = reach * math.sin(math.radians(EDGE_PAN_DEG))
                if side <= self.body_half[1] + 0.02:
                    edge_gap = SONIC_PIVOT_M + reach * math.cos(math.radians(pan)) - SONIC_AHEAD_M
                    gap = edge_gap if gap is None else min(gap, edge_gap)
        return gap

    def side_range(self, pan: float) -> tuple[float | None, float]:
        """Ultrasonic turned to `pan` (deg, +left) and back: (reading m or None, distance to the wall there)."""
        try:
            cm = self.look(pan)
        finally:
            if self.servo is not None:
                self.servo.set_pan(self.home_pan)
                if self.stop_event.wait(0.35):  # facing ahead again before anything drives
                    raise Abort("stopped")
        p, a = self.pose, self.pose.yaw + math.radians(pan)
        ox = p.x + SONIC_PIVOT_M * math.cos(p.yaw) + SONIC_ARM_M * math.cos(a)
        oy = p.y + SONIC_PIVOT_M * math.sin(p.yaw) + SONIC_ARM_M * math.sin(a)
        return (None if cm is None else cm / 100.0), self.wall_distance(ox, oy, a)

    def outer_corridor(self, out: float) -> float | None:
        """How far the car can drive ahead before its outer side meets something, or None if nothing.

        Neither the forward ultrasonic (centre line) nor the corner IR (7 cm out, 8 cm range) covers the
        last few cm of the body's outer side, where a neighbouring wall's block can stand. The ultrasonic
        turned 15 and 30 deg to that side does: a return within the body width (plus 3 cm) on that side
        is in the path.
        """
        limit = None
        for pan in (15.0 * out, 30.0 * out):
            reading, _wall = self.side_range(pan)
            if reading is None:
                continue
            reach = SONIC_ARM_M + reading
            ahead = SONIC_PIVOT_M + reach * math.cos(math.radians(pan))
            side = reach * math.sin(math.radians(abs(pan)))
            if side <= self.body_half[1] + 0.03:
                room = max(ahead - self.body_half[0] - 0.02, 0.0)
                limit = room if limit is None else min(limit, room)
        return limit
