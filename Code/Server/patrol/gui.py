"""One window: the map with the car, its path and what it has seen; the cameras that are open; pose, state and wheels.

The wheels have no encoders: the bars are the commands sent to the motor driver, not measured speeds.
"""
from __future__ import annotations

import math
import os
import time

import cv2
import numpy as np

from patrol.world import FREE, OCCUPIED, RES_M

W, H = 1280, 720
CELL = 7
INK, MUTED = (40, 40, 40), (110, 110, 110)
WHEELS = ("LF", "LR", "RF", "RR")


def text(img, s, x, y, scale=0.5, color=INK, bold=False):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if bold else 1, cv2.LINE_AA)


class Gui:
    def __init__(self, nav, folder, show: bool = True):
        self.nav, self.folder = nav, folder
        self.window = None
        if show:
            os.environ.setdefault("DISPLAY", ":0")
            os.environ.setdefault("XAUTHORITY", "/home/pi/.Xauthority")
            try:
                cv2.namedWindow("patrol", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("patrol", W, H)
                self.window = "patrol"
            except cv2.error as exc:
                print(f"no display for the window ({exc}); the picture is still saved as gui.jpg", flush=True)
        self.saved = 0.0

    # ---- parts ----------------------------------------------------------------

    def map_panel(self, st: dict) -> np.ndarray:
        nav = self.nav
        world = nav.world
        if world is None:
            return np.full((500, 500, 3), 235, np.uint8)
        log = world.log.T[::-1]
        img = np.full(log.shape + (3,), 215, np.uint8)             # not seen
        img[log < FREE] = (236, 246, 236)                          # seen free
        img[log > OCCUPIED] = (60, 60, 200)                        # occupied
        img = cv2.resize(img, None, fx=CELL, fy=CELL, interpolation=cv2.INTER_NEAREST)
        hh = img.shape[0]
        px = lambda x, y: (int(x / RES_M * CELL + CELL / 2), int(hh - y / RES_M * CELL - CELL / 2))
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, hh - 1), (30, 30, 30), 2)
        if nav.mission is not None:
            for n, g in enumerate(nav.mission.goals):
                done = n < nav.mission.index
                cv2.circle(img, px(g.x, g.y), 7, (150, 150, 150) if done else (200, 120, 0), -1 if done else 2, cv2.LINE_AA)
        path = None if nav.thinker is None else nav.thinker.path
        if path:
            cv2.polylines(img, [np.array([px(q[0], q[1]) for q in path], np.int32)], False, (0, 150, 0), 2, cv2.LINE_AA)
        if len(nav.trail) > 1:
            cv2.polylines(img, [np.array([px(*q) for q in nav.trail[::3]], np.int32)], False, (180, 90, 20), 1, cv2.LINE_AA)
        pose = st.get("pose")
        if pose:
            x, y, yaw = pose
            c, s = math.cos(yaw), math.sin(yaw)
            hl, hw = nav.half
            box = [px(x + c * a - s * b, y + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            cv2.polylines(img, [np.array(box, np.int32)], True, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.arrowedLine(img, px(x, y), px(x + c * hl, y + s * hl), (20, 20, 20), 2, cv2.LINE_AA, tipLength=0.35)
            at = (st.get("safety") or {}).get("at")
            if at and st.get("why"):
                cv2.drawMarker(img, px(*at), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        return img

    @staticmethod
    def fit(img, w, h, label):
        out = np.full((h, w, 3), 30, np.uint8)
        if img is not None:
            k = min(w / img.shape[1], h / img.shape[0])
            small = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
            y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
            out[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        cv2.rectangle(out, (0, 0), (len(label) * 9 + 14, 22), (30, 30, 30), -1)
        text(out, label, 6, 16, 0.45, (255, 255, 255))
        return out

    def render(self) -> np.ndarray:
        nav = self.nav
        st = dict(nav.status)
        img = np.full((H, W, 3), 250, np.uint8)
        m = self.map_panel(st)
        img[10:10 + m.shape[0], 10:10 + m.shape[1]] = m
        mx = 10 + m.shape[1] + 14
        cw = (W - mx - 30) // 3
        ch = int(cw * 0.75)
        eye = None if nav.eyes is None else nav.eyes.picture()
        last = None if nav.eyes is None else nav.eyes.last
        label = "gimbal" if last is None else f"gimbal {last['pan']:+.0f} deg, echo " + ("none" if last["echo"] is None else f"{last['echo'] * 100:.0f} cm")
        img[10:10 + ch, mx:mx + cw] = self.fit(eye, cw, ch, label)
        grabs = getattr(nav.car, "grabbers", {})
        for n, name in enumerate(("front", "rear")):
            frame = grabs[name].latest()[0] if name in grabs else None
            x0 = mx + (n + 1) * (cw + 10)
            img[10:10 + ch, x0:x0 + cw] = self.fit(frame, cw, ch, f"{name} (tags)")
        # state
        x, y = mx, 10 + ch + 40
        text(img, st.get("state", "")[:60], x, y, 0.65, (0, 0, 160) if "standing" in st.get("state", "") or "ended" in st.get("state", "") else (0, 110, 0), True)
        lines = []
        if st.get("pose"):
            p = st["pose"]
            lines += [f"pose   x {p[0] * 100:6.1f} cm   y {p[1] * 100:6.1f} cm   heading {math.degrees(p[2]):+6.1f} deg",
                      f"pose uncertainty {st.get('sigma_cm', 0):.1f} cm    tags used {st.get('tags_used', 0)}, refused {st.get('tags_refused', 0)}",
                      f"goal   {st.get('goal') or '-'}   ({st.get('goals_done', 0)} of {st.get('goals', 0)} done)",
                      "command  forward {:+.2f}  left {:+.2f} m/s  turn {:+.0f} deg/s".format(st["cmd"][0], st["cmd"][1], math.degrees(st["cmd"][2])),
                      f"gyro {st.get('gyro_dps', 0):+6.1f} deg/s    speed scale fwd {st['scale'][0]:.2f} side {st['scale'][1]:.2f}",
                      f"looks {st.get('looks', 0)} (too fast for the picture: {st.get('smeared', 0)})    map version {st.get('map_version', 0)}"]
            sf = st.get("safety") or {}
            if sf.get("kind"):
                lines.append(f"safety: {sf['kind']} at ({sf['at'][0] * 100:.0f}, {sf['at'][1] * 100:.0f}) cm")
        for n, line in enumerate(lines):
            text(img, line, x, y + 34 + n * 27, 0.55)
        # wheels
        bx, by = 10, 10 + m.shape[0] + 34
        text(img, "wheel commands (PWM sent; no encoders)" + ("" if st.get("armed", False) else "   NOT ARMED: not sent"), bx, by - 10, 0.47, MUTED)
        pwm = st.get("pwm", [0, 0, 0, 0])
        for n, (name, v) in enumerate(zip(WHEELS, pwm)):
            y0 = by + n * 34
            text(img, name, bx, y0 + 18, 0.5, INK, True)
            cv2.rectangle(img, (bx + 40, y0 + 4), (bx + 440, y0 + 24), (225, 225, 225), -1)
            mid = bx + 240
            cv2.rectangle(img, (mid, y0 + 4), (mid + int(v / 2000 * 200), y0 + 24), (140, 122, 31) if v >= 0 else (60, 60, 200), -1)
            cv2.line(img, (mid, y0), (mid, y0 + 28), INK, 1)
            text(img, f"{v:+5d}", bx + 450, y0 + 19, 0.5)
        text(img, "map: green = seen free, grey = not seen, red = occupied; green line = plan, brown = driven; q = stop", 10, H - 10, 0.42, MUTED)
        return img

    def loop(self, alive) -> None:
        while alive():
            try:
                img = self.render()
            except Exception as exc:  # noqa: BLE001  the display must not end the run
                img = np.full((H, W, 3), 250, np.uint8)
                text(img, f"display error: {exc}", 20, 40)
            now = time.monotonic()
            if now - self.saved > 1.0:
                self.saved = now
                cv2.imwrite(str(self.folder / "gui.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if self.window:
                cv2.imshow(self.window, img)
                if cv2.waitKey(80) & 0xFF in (ord("q"), 27):
                    self.nav.car.stop_event.set()
            else:
                time.sleep(0.1)
        cv2.imwrite(str(self.folder / "gui.jpg"), self.render(), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if self.window:
            cv2.destroyAllWindows()
