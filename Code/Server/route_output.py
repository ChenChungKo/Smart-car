"""The map, the trajectory and the recording.

Mixin of RouteFollower (route_follow.py); the methods use self.* of the whole follower."""

from __future__ import annotations

from route_common import *  # noqa: F401,F403


class OutputMixin:
    # ---- route ----------------------------------------------------------

    def record(self):
        if self.pose is not None:
            self.trail.append((self.pose.x, self.pose.y, self.pose.yaw))

    def save_map(self):
        if not self.arena.get("arena_size_m"):
            return
        img = draw_map(self.arena, route=None, trail=self.trail, pose=self.pose, size_px=640)
        self.draw_route(img)
        if self.home is not None:
            w_m, h_m = (float(v) for v in self.arena["arena_size_m"])
            margin, scale = 30, (640 - 2 * 30) / max(w_m, h_m)
            img_h = int(h_m * scale) + 2 * margin
            hx = int(round(margin + self.home.x * scale))
            hy = int(round(img_h - margin - self.home.y * scale))
            cv2.drawMarker(img, (hx, hy), (0, 140, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
        self.obstacles.draw(img)
        cv2.imwrite(str(self.debug_dir / "trajectory.png"), img)
        if self.window is not None:
            views = []
            for name in ("front", "rear"):
                grabber = self.grabbers.get(name)
                frame = grabber.latest()[0] if grabber is not None else None
                if frame is None:
                    continue
                view = cv2.resize(frame, (img.shape[0] * frame.shape[1] // frame.shape[0], img.shape[0]))
                cv2.putText(view, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                views.append(view)
            if views:
                img = np.hstack([img, *views])
            cv2.imshow(self.window, img)
            cv2.waitKey(1)
        elif self.record_path is not None:
            for name in ("front", "rear"):
                grabber = self.grabbers.get(name)
                frame = grabber.latest()[0] if grabber is not None else None
                if frame is None:
                    continue
                view = cv2.resize(frame, (img.shape[0] * frame.shape[1] // frame.shape[0], img.shape[0]))
                cv2.putText(view, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                img = np.hstack([img, view])
        self.write_frame(img)

    def draw_route(self, img) -> None:
        """The route over the map: a line, and ONE label per place.

        The lap's corners have four route points on one spot (lane end, corner, turned, next lane start); one
        label each drew them on top of each other. With --hug the labels are 1-4, the corners in lap order;
        other routes keep the waypoint numbers of the log.
        """
        if not self.route:
            return
        w_m, h_m = (float(v) for v in self.arena["arena_size_m"])
        margin, scale = 30, (640 - 60) / max(w_m, h_m)
        img_h = int(h_m * scale) + 2 * margin

        def px(x, y):
            return int(round(margin + x * scale)), int(round(img_h - margin - y * scale))

        layer = img.copy()
        points = [px(x, y) for x, y in self.route]
        for a, b in zip(points, points[1:]):
            cv2.line(layer, a, b, (200, 160, 0), 2, cv2.LINE_AA)
        places: dict[tuple[int, int], list[int]] = {}
        for i, p in enumerate(points[:-1] if len(points) > 1 and points[0] == points[-1] else points):
            places.setdefault((round(p[0] / 6), round(p[1] / 6)), []).append(i)
        for number, (key, indices) in enumerate(places.items(), 1):
            p = points[indices[0]]
            cv2.circle(layer, p, 5, (200, 160, 0), -1)
            if self.args.hug:
                text = str(number)  # the lap's corners, 1-4 counter-clockwise from the start (the log keeps p-numbers)
            else:
                text = str(indices[0]) if len(indices) == 1 else f"{indices[0]}-{indices[-1]}"
                if len(indices) > 1 and indices != list(range(indices[0], indices[-1] + 1)):
                    text = ",".join(str(i) for i in indices)
            # the label sits outside the arena middle, away from the lines
            dx = 8 if p[0] >= img.shape[1] // 2 else -8 - 7 * len(text)
            dy = -9 if p[1] <= img.shape[0] // 2 else 15
            cv2.putText(layer, text, (p[0] + dx, p[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 90, 0), 1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.75, img, 0.25, 0, img)

    def write_frame(self, img):
        if self.record_path is None:
            return
        if self.writer is None:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.record_size = (img.shape[1] // 2 * 2, img.shape[0] // 2 * 2)
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self.writer = cv2.VideoWriter(str(self.record_path), fourcc, 1.0, self.record_size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = cv2.VideoWriter(str(self.record_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                              1.0, self.record_size)
            print(f"recording {self.record_size[0]}x{self.record_size[1]} to {self.record_path}", flush=True)
        if img.shape[1::-1] != self.record_size:
            img = cv2.resize(img, self.record_size)
        self.writer.write(img)
