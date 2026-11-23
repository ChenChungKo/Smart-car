"""IMU 檢視器（tools/gy95t_viewer.py）右欄的地圖區：跟著正在跑的車子程式畫即時地圖、車子和移動軌跡。

資料來源（只讀檔案，不碰相機、超音波或馬達，所以不會和跑車程式搶硬體）：
  巡邏 tools/patrol_run.py → route_follow_debug/patrol/<時間>/
  繞圈 tools/straight_avoid.py → route_follow_debug/straight/<時間>/（同樣的 control.jsonl、plans.jsonl；沒有佔用地圖，
       plans 的 path 是目前沿著牆走的那條線）
      control.jsonl  每次控制一行：pose = [x, y, 航向(弧度)]（公尺，場地左下角為原點）、sigma_cm、goal …
      maps/*.npy     每 0.5 秒存一次的佔用地圖（log-odds，格子 2 cm，patrol/world.py）
      plans.jsonl    每次規劃一行：path = 規劃的路線
「正在跑」= 最新一趟的 control.jsonl 在 3 秒內有更新。沒有在跑時只顯示空的場地（外框、標籤、起點）。
"""
from __future__ import annotations

import json
import math
import os
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

SERVER = Path(__file__).resolve().parents[1]
# 測試時可用 MAP_PANEL_RUNS 換成別的資料夾（可用 : 分隔多個）
RUNS = [Path(p) for p in os.environ.get("MAP_PANEL_RUNS", "").split(":") if p] or \
    [SERVER / "route_follow_debug" / "patrol", SERVER / "route_follow_debug" / "straight"]
KIND = {"patrol": "巡邏中", "straight": "繞圈中"}
ARENA = json.loads((SERVER / "arena_map.json").read_text(encoding="utf-8"))
CAR = json.loads((SERVER / "car_nav.json").read_text(encoding="utf-8"))
RES_M, OCCUPIED, FREE = 0.02, 0.8, -0.3          # 和 patrol/world.py 相同
LIVE_S = 3.0
WALL_M = 0.05
REFRESH_MS = 200

# 顏色（BGR，深色主題）
FLOOR, UNSEEN, SEEN_FREE, OCC = (58, 58, 58), (48, 48, 48), (70, 96, 70), (60, 60, 210)
GRID, TRAIL, PLAN, CARC = (75, 75, 75), (40, 160, 255), (90, 200, 90), (240, 240, 240)
WALL_BLUE, WALL_GREY = (185, 120, 55), (105, 105, 105)   # 場地實際的牆：下方和右邊藍色泡棉，上方和左邊灰色


class MapPanel:
    def __init__(self, parent, size: int, bg: str, fg: str, font):
        self.size = size
        self.frame = tk.Frame(parent, bg=bg)
        self.view = tk.Label(self.frame, bg="#111", bd=0)
        self.view.pack()
        self.status = tk.Label(self.frame, text="", fg="#9a9a9a", bg=bg, font=font, anchor="w", justify="left",
                               wraplength=size)
        self.status.pack(fill="x")
        tk.Label(self.frame, text="橘線 = 走過的軌跡　綠線 = 規劃路線　紅 = 有東西　綠底 = 看過是空的　深灰 = 還沒看過",
                 fg="#777", bg=bg, font=font, anchor="w", justify="left", wraplength=size).pack(fill="x")
        self.half = (CAR["body_m"][0] / 2, CAR["body_m"][1] / 2)
        aw, ah = ARENA["arena_size_m"]
        self.aw, self.ah = aw, ah
        self.pad = 8
        self.k = (size - 2 * self.pad) / (max(aw, ah) + 2 * WALL_M)     # 像素 / 公尺
        self._reset(None)
        self._img = None
        self._last_scan = 0.0
        self.frame.after(REFRESH_MS, self._tick)

    # ---- following a run ---------------------------------------------------------------------------
    def _reset(self, run):
        self.run = run
        self.ctrl_pos = self.plan_pos = 0
        self.trail, self.pose, self.row, self.path = [], None, None, None
        self.log, self.map_name = None, None
        self.dirty = True

    def _live_run(self):
        files = []
        for root in RUNS:
            try:
                files += list(root.glob("*/control.jsonl"))
            except OSError:
                pass
        if not files:
            return None
        newest = max(files, key=lambda p: p.stat().st_mtime)
        return newest.parent if time.time() - newest.stat().st_mtime < LIVE_S else None

    @staticmethod
    def _read_lines(path: Path, pos: int):
        """從 pos 開始讀完整的行（最後一行還沒寫完就留到下次）。"""
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                data = f.read()
        except OSError:
            return [], pos
        end = data.rfind(b"\n")
        if end < 0:
            return [], pos
        rows = []
        for line in data[:end].splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        return rows, pos + end + 1

    def _poll(self):
        now = time.monotonic()
        if now - self._last_scan > 1.0:
            self._last_scan = now
            run = self._live_run()
            if run != self.run:
                self._reset(run)
        if self.run is None:
            return
        rows, self.ctrl_pos = self._read_lines(self.run / "control.jsonl", self.ctrl_pos)
        for r in rows:
            p = r.get("pose")
            if not p:
                continue
            if not self.trail or math.hypot(p[0] - self.trail[-1][0], p[1] - self.trail[-1][1]) > 0.005:
                self.trail.append((p[0], p[1]))
            self.pose, self.row = p, r
            self.dirty = True
        rows, self.plan_pos = self._read_lines(self.run / "plans.jsonl", self.plan_pos)
        for r in rows:
            if r.get("path"):
                self.path = r["path"]
                self.dirty = True
        maps = self.run / "maps"
        try:
            newest = max((p.name for p in maps.glob("*.npy")), default=None)
        except OSError:
            newest = None
        if newest and newest != self.map_name:
            try:
                self.log = np.load(maps / newest)
                self.map_name = newest
                self.dirty = True
            except (OSError, ValueError):
                pass                       # 檔案還在寫，下次再讀

    # ---- drawing -----------------------------------------------------------------------------------
    def px(self, x, y):
        return (int(round(self.pad + (x + WALL_M) * self.k)), int(round(self.size - self.pad - (y + WALL_M) * self.k)))

    def _render(self) -> np.ndarray:
        S, k = self.size, self.k
        img = np.full((S, S, 3), 30, np.uint8)
        x0, y0 = self.px(0, self.ah)
        x1, y1 = self.px(self.aw, 0)
        if self.log is not None:
            nx, ny = self.log.shape
            cells = np.empty((ny, nx, 3), np.uint8)
            log = self.log.T[::-1]
            cells[:] = UNSEEN
            cells[log < FREE] = SEEN_FREE
            cells[log > OCCUPIED] = OCC
            # 格子 i 的中心在 x = i * RES_M
            gx0, gy0 = self.px(-RES_M / 2, (ny - 0.5) * RES_M)
            gx1, gy1 = self.px((nx - 0.5) * RES_M, -RES_M / 2)
            big = cv2.resize(cells, (max(1, gx1 - gx0), max(1, gy1 - gy0)), interpolation=cv2.INTER_NEAREST)
            img[gy0:gy0 + big.shape[0], gx0:gx0 + big.shape[1]] = big
            img[:y0] = 30; img[y1 + 1:] = 30; img[:, :x0] = 30; img[:, x1 + 1:] = 30
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), FLOOR, -1)
        for g in np.arange(0.1, max(self.aw, self.ah), 0.1):          # 每 10 cm 一條細線
            if g < self.aw:
                cv2.line(img, self.px(g, 0), self.px(g, self.ah), GRID, 1)
            if g < self.ah:
                cv2.line(img, self.px(0, g), self.px(self.aw, g), GRID, 1)
        aw, ah, w = self.aw, self.ah, WALL_M                          # 牆（實際顏色）
        for (ax, ay, bx, by), color in (((-w, -w, 0, ah + w), WALL_GREY), ((-w, ah, aw + w, ah + w), WALL_GREY),
                                        ((0, -w, aw + w, 0), WALL_BLUE), ((aw, -w, aw + w, ah), WALL_BLUE)):
            cv2.rectangle(img, self.px(ax, by), self.px(bx, ay), color, -1)
        # 牆上的標籤：每個都用同樣大小的白框（放在牆的正中央）和同樣的黑字，編號在框內
        short = max(int(round(w * k)), 18)                              # 框的短邊（牆厚，至少 18 像素）
        long_ = max(int(round(ARENA.get("tag_size_m", 0.1) * k)), short + 6)
        font, thick = cv2.FONT_HERSHEY_SIMPLEX, 2 if short >= 18 else 1
        scale = cv2.getFontScaleFromHeight(font, int(short * 0.62), thick)
        try:                                                             # 清楚的粗體字（沒有這個字型就用 OpenCV 的）
            pil_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(short * 0.78))
        except OSError:
            pil_font = None
        labels = []
        for t in ARENA.get("tags", []):
            a = math.radians(t.get("facing_deg", 0))
            nx, ny = math.cos(a), math.sin(a)                            # 朝場內
            cxp, cyp = self.px(t["center_m"][0] - nx * w / 2, t["center_m"][1] - ny * w / 2)   # 牆厚的中間
            horizontal = abs(ny) > abs(nx)                               # 上下兩面牆：框是橫的
            bw, bh = (long_, short) if horizontal else (short, long_)
            box = (cxp - bw // 2, cyp - bh // 2, cxp - bw // 2 + bw - 1, cyp - bh // 2 + bh - 1)
            cv2.rectangle(img, box[:2], box[2:], (245, 245, 245), -1)
            cv2.rectangle(img, box[:2], box[2:], (20, 20, 20), 1)
            label = str(t["id"])
            if pil_font is not None:
                labels.append((cxp, cyp, label))
            else:
                (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
                cv2.putText(img, label, (cxp - tw // 2, cyp + th // 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        if labels:
            pil = Image.fromarray(img)
            d = ImageDraw.Draw(pil)
            for cxp, cyp, label in labels:
                d.text((cxp, cyp), label, font=pil_font, fill=(0, 0, 0), anchor="mm")
            img = np.array(pil)
        if "home" in ARENA:
            hx, hy = ARENA["home"][:2]
            cv2.drawMarker(img, self.px(hx, hy), (180, 180, 180), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        if self.path:
            cv2.polylines(img, [np.array([self.px(q[0], q[1]) for q in self.path], np.int32)], False, PLAN, 2, cv2.LINE_AA)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([self.px(*q) for q in self.trail], np.int32)], False, TRAIL, 2, cv2.LINE_AA)
        if self.pose:
            x, y, yaw = self.pose
            c, s = math.cos(yaw), math.sin(yaw)
            hl, hw = self.half
            box = [self.px(x + c * a - s * b, y + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            shade = img.copy()                     # 車身半透明：底下最近走過的軌跡看得到
            cv2.fillPoly(shade, [np.array(box, np.int32)], (90, 90, 90))
            img = cv2.addWeighted(shade, 0.55, img, 0.45, 0)
            cv2.polylines(img, [np.array(box, np.int32)], True, CARC, 2, cv2.LINE_AA)
            cv2.arrowedLine(img, self.px(x, y), self.px(x + c * hl * 1.3, y + s * hl * 1.3), (0, 215, 255), 2,
                            cv2.LINE_AA, tipLength=0.35)
            sig = (self.row or {}).get("sigma_cm")
            if sig:
                cv2.circle(img, self.px(x, y), max(2, int(sig / 100 * k)), (0, 215, 255), 1, cv2.LINE_AA)
        return img

    def _tick(self):
        try:
            self._poll()
            if self.dirty or self._img is None:
                self.dirty = False
                rgb = cv2.cvtColor(self._render(), cv2.COLOR_BGR2RGB)
                self._img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.view.config(image=self._img)
                self.status.config(text=self._status_text())
        finally:
            self.frame.after(REFRESH_MS, self._tick)

    def _status_text(self) -> str:
        if self.run is None:
            return "沒有車子程式在跑：只顯示場地（藍、灰 = 牆的實際顏色，白框 = 標籤和編號，十字 = 起點）"
        r = self.row or {}
        p = self.pose
        txt = f"{KIND.get(self.run.parent.name, '執行中')} {self.run.name}  {r.get('s', 0):.0f} 秒"
        if p:
            txt += f"  x {p[0] * 100:.0f} cm  y {p[1] * 100:.0f} cm  航向 {math.degrees(p[2]):+.0f}°"
        if r.get("sigma_cm") is not None:
            txt += f"  不確定 {r['sigma_cm']:.1f} cm"
        if r.get("goal"):
            txt += f"  目標 {r['goal']}"
        if r.get("why"):
            txt += f"  停：{r['why'][:30]}"
        elif r.get("note"):
            txt += f"\n{r['note'][:70]}"
        return txt
