"""IMU 檢視器（tools/gy95t_viewer.py）右側的相機區：一次顯示一個畫面，用按鈕切換。

畫面：雲台、前 CSI、後 CSI、左 USB、右 USB，或「鳥瞰」（按一次：前後 CSI＋左右 USB 各拍一張，投影到地面拼成俯視圖，
dashboard/ipm.py 的 GroundMapper；拍完相機馬上關掉，再按一次重拍）。只顯示畫面，不處理、不轉雲台。
切換：CSI 約 0.7 秒、USB 約 1～2 秒才有畫面。USB 相機被別的程式使用中時不搶，只顯示「使用中」。

讓給跑車程式：巡邏、繞圈等程式一啟動就要開前後 CSI 和雲台相機，檢視器佔著任何一支，它們就開不了。所以每秒檢查
一次有沒有 Code/Server 裡的其他程式在跑（電池視窗、雲台電源監看、檢視器自己除外）；有就把相機全部關掉，畫面註明，
程式結束 3 秒後再自動開回原本的畫面。這也避免「3 支 USB 同時開＋轉雲台」整組 USB 斷線（鳥瞰開著左右兩支 USB，
跑車程式再開雲台就是 3 支）。

在沒有車上相機的電腦（例如 Windows）上 import 會失敗，檢視器就不顯示相機區。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from aruco_localizer import CsiGrabber          # noqa: E402  前後 CSI（Picamera2，和定位程式相同的設定）
from camera_devices import resolve_usb_capture_index   # noqa: E402
from foot_cam import GimbalFeed                  # noqa: E402  USB 相機（640x480 YUYV，和巡邏程式相同）

HARDWARE = json.loads((SERVER / "camera_hardware.json").read_text(encoding="utf-8"))
CAR = json.loads((SERVER / "car_nav.json").read_text(encoding="utf-8"))
CAMERAS = [("雲台", "gimbal"), ("前", "front"), ("後", "rear"), ("左", "left"), ("右", "right"), ("鳥瞰", "bev")]
NAMES = {r: t for t, r in CAMERAS}
REFRESH_MS = int(os.environ.get("CAM_PANEL_MS", "100"))   # 畫面每 0.1 秒看一次有沒有新的（相機本身約 15～19 張/秒）
# 鳥瞰圖：按一次拍一次（車子停著時），不是即時影像
STAGGER_S = 0.7                                            # 每支相機開啟的間隔（四支同時啟動曾讓 5V 電壓不足）
WARMUP_S = 1.5                                             # 四支都有畫面後再等這麼久（自動曝光穩定）才拍
CSI_SNAPSHOT = (1640, 1232)                                # 前後 CSI 用感測器全視角的解析度拍
BEV_RES_M = 0.0012                                         # 鳥瞰圖每個像素 1.2 mm
BEV_FRONT_BACK_M, BEV_SIDE_M = 0.18, 0.34                  # 範圍：車身前後各 18 cm、左右各 34 cm（4:3 橫的，填滿相機區）
SNAP_DIR = SERVER / "route_follow_debug" / "bev_snapshots"  # 每次拍的四張原圖和鳥瞰圖
# 車子周圍相機看不到的那一圈（側邊相機的畫面下緣只到車身旁約 5 cm）畫成車子的柔和陰影：
SHADOW_SMOOTH_M = 0.03                                     # 形狀圓滑化的範圍（去掉相機視野邊緣的鋸齒）
SHADOW_GROW_M = 0.012                                      # 再往外擴，確保蓋住鋸齒狀的邊
SHADOW_FEATHER_M = 0.012                                   # 邊緣漸層融入地面的寬度
SHADOW_DARK = 0.55                                         # 緊貼車身處比地面暗多少（0 = 不暗，1 = 全黑）
RING_CM = (2.0, 7.0)                                       # 地面顏色：取陰影外面 2～7 cm 那一圈的中位數


def bev_mapper(loc, car_from_camera, body):
    """dashboard/ipm.py 的 GroundMapper，改成給檢視器的鳥瞰拍照用：
    - 範圍是橫的（左右看寬一點），填滿 4:3 的相機區，不會兩側留黑；
    - 不先放大 3 倍（顯示時本來就會縮放）；
    - 車子周圍看不到的那一圈畫成柔和的陰影（見 _fill_unseen），不用原本的 inpaint（每張 150 ms 以上）。"""
    import numpy as np
    import dashboard.ipm as ipm
    ipm.SHOW_SCALE = 1                 # 只影響檢視器這個程式：直接用細的格子，不先算粗的再放大
    ipm.RES_M = BEV_RES_M

    class LiveMapper(ipm.GroundMapper):
        def __init__(self):
            super().__init__(loc, car_from_camera, body)
            self.x_range = (-body[0] / 2 - BEV_FRONT_BACK_M, body[0] / 2 + BEV_FRONT_BACK_M)
            self.y_range = (-body[1] / 2 - BEV_SIDE_M, body[1] / 2 + BEV_SIDE_M)
            xs = np.arange(self.x_range[1], self.x_range[0], -ipm.RES_M)
            ys = np.arange(self.y_range[1], self.y_range[0], -ipm.RES_M)
            gy, gx = np.meshgrid(ys, xs)
            self.shape = gx.shape
            self.ground = np.stack([gx, gy, np.zeros_like(gx)], axis=-1).reshape(-1, 3)
            self.maps = {}
            self._fill_key = None

        def _fill_unseen(self, out, unseen):
            """看不到的那一圈畫成車子的柔和陰影：形狀圓滑化（去掉相機視野邊緣的鋸齒）、往外擴一點蓋住邊，裡面填地面
            顏色、越靠近車身越暗，邊緣漸層融入地面。不複製紋理、不猜那裡有什麼，只讓它看起來像車子的影子。
            範圍外角（相機看不到那麼遠）不處理，維持背景色。形狀和明暗只和相機有關：算一次就好。"""
            key = (unseen.shape, int(unseen.sum()))
            if key != self._fill_key:
                res = ipm.RES_M
                hole = cv2.dilate(unseen.astype(np.uint8), np.ones((5, 5), np.uint8))
                _n, lab = cv2.connectedComponents(hole)
                r0, r1, c0, c1 = self._car_rect()
                inner = np.unique(lab[r0:r1, c0:c1])
                hole = np.isin(lab, inner[inner > 0]).astype(np.float32)          # 只要和車身連在一起的那一圈
                shape = (cv2.GaussianBlur(hole, (0, 0), SHADOW_SMOOTH_M / res / 2.5) > 0.25).astype(np.uint8)
                grow = max(3, int(round(SHADOW_GROW_M / res)))
                shape = cv2.dilate(shape, np.ones((grow, grow), np.uint8))
                self._alpha = cv2.GaussianBlur(shape.astype(np.float32), (0, 0), SHADOW_FEATHER_M / res)[..., None]
                outside_cm = cv2.distanceTransform((shape == 0).astype(np.uint8), cv2.DIST_L2, 5) * res * 100
                self._ring = np.nonzero((shape == 0) & (outside_cm >= RING_CM[0]) & (outside_cm <= RING_CM[1]))
                body_mask = np.zeros(shape.shape, np.uint8)
                body_mask[r0:r1, c0:c1] = 1
                to_body = cv2.distanceTransform(1 - body_mask, cv2.DIST_L2, 5)
                to_rim = cv2.distanceTransform(shape, cv2.DIST_L2, 5)
                t = np.clip(to_rim / np.maximum(to_rim + to_body, 1e-3), 0.0, 1.0)  # 邊緣 0 → 車身 1
                self._shade = (1.0 - SHADOW_DARK * t)[..., None].astype(np.float32)
                self._fill_key = key
            if len(self._ring[0]) < 20:
                return out
            floor = np.median(out[self._ring], axis=0).astype(np.float32)       # 中位數：零星的物品不影響
            return (out * (1.0 - self._alpha) + floor * self._shade * self._alpha).astype(np.uint8)

        def _car_rect(self):
            """車身照片在圖上的列、行範圍。"""
            res = ipm.RES_M
            length, width = body
            return (int((self.x_range[1] - length / 2) / res), int((self.x_range[1] + length / 2) / res),
                    int((self.y_range[1] - width / 2) / res), int((self.y_range[1] + width / 2) / res))

    return LiveMapper()
# 這些程式不用相機，在跑也不必讓出相機
NOT_CAR_PROGRAMS = {"gy95t_viewer.py", "open_car_imu.py", "battery_monitor.py", "gimbal_power_watch.py",
                    "camera_panel.py", "map_panel.py", "car_status_panel.py", "imu_live_two.py", "imu_regs.py"}
RESUME_AFTER_S = 3.0


def is_torn(frame, good) -> bool:
    """USB 傳輸掉了一段資料時，畫面從某一列以下會整段水平平移、繞回來（2026-11-23 左相機實測，約 1～10% 的畫格）。
    判斷：和上一張好的畫面差很多，而且變動的那一段幾乎就是上一張水平平移後的樣子。車子真的在動時整個畫面一起變，
    不會剛好是水平平移，所以不會被當成錯位。"""
    g = cv2.cvtColor(cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    p = cv2.cvtColor(cv2.resize(good, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.abs(g - p).mean(axis=1)
    changed = np.flatnonzero(rows > 15)
    if len(changed) < 12:                               # 幾乎沒變：正常畫面
        return False
    band = slice(changed[0], 120)
    plain = np.abs(g[band] - p[band]).mean()
    best = min(np.abs(g[band] - np.roll(p[band], k, axis=1)).mean() for k in range(-80, 81, 2) if abs(k) >= 4)
    return best < 0.4 * plain


class UsbFeed(GimbalFeed):
    """foot_cam.GimbalFeed 換成任一支 USB 相機（camera_hardware.json 的 gimbal / left / right）。
    錯位的畫格（見 is_torn）不交出去：latest() 給的是最近一張好的畫面。"""

    def __init__(self, role: str):
        super().__init__()
        self.entry = HARDWARE[role]
        self.torn = 0
        self._good = (None, 0.0)

    def latest(self):
        frame, stamp = super().latest()
        good, good_stamp = self._good
        if frame is None or stamp == good_stamp:
            return good if good is not None else frame, good_stamp if good is not None else stamp
        if good is not None and is_torn(frame, good):
            self.torn += 1
            return good, good_stamp
        self._good = (frame, stamp)
        return frame, stamp


def usb_users(role: str) -> str:
    """這支 USB 相機被哪些程式開著（空字串 = 沒有）。"""
    try:
        index, _ = resolve_usb_capture_index(HARDWARE[role])
    except Exception as exc:  # noqa: BLE001
        return f"找不到裝置（{exc}）"
    out = subprocess.run(["fuser", f"/dev/video{index}"], capture_output=True, text=True)
    names = []
    for pid in out.stdout.split():
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = ""
        names.append(f"{pid} {Path(cmd.split(' ')[1]).name if cmd.count(' ') else cmd}")
    return "、".join(names)


def car_program() -> str:
    """Code/Server 裡正在跑、可能要用相機的另一個 Python 程式（腳本名稱），沒有就是空字串。"""
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            args = (d / "cmdline").read_bytes().split(b"\0")
            if not args or b"python" not in Path(args[0].decode(errors="replace")).name.encode():
                continue
            script = next((a.decode(errors="replace") for a in args[1:] if a.endswith(b".py")), "")
            if not script:
                continue
            path = Path(script) if script.startswith("/") else Path(os.readlink(d / "cwd")) / script
            path = path.resolve()
        except (OSError, ValueError):
            continue
        if SERVER in path.parents and path.name not in NOT_CAR_PROGRAMS:
            return path.name
    return ""


class BevSnapshot:
    """按一次拍一次的鳥瞰圖：四支相機錯開開啟，都有畫面後再等 WARMUP_S，各拿一張，馬上全部關掉，再拼接。
    latest() 和相機一樣回傳 (畫面, 時間)；拼好之前是 (None, 0)。四張原圖和鳥瞰圖存在 SNAP_DIR/<時間>/。"""

    def __init__(self):
        self.error, self.note = "", "拍攝中…"
        self._frame, self._stamp = None, 0.0
        self.taken = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bev")

    def start(self):
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._frame, self._stamp

    def close(self):
        self._stop.set()
        self._thread.join(timeout=8.0)

    def _grab(self):
        """四支相機各一張（左右被別的程式用著就跳過）。"""
        feeds, frames = {}, {}
        try:
            busy = [s for s in ("left", "right") if usb_users(s)]
            for name in ("front", "rear", "left", "right"):
                if self._stop.is_set():
                    return frames
                if name in ("front", "rear"):
                    feeds[name] = CsiGrabber(name, int(CAR["camera_mounts"][name]["camera_num"]), *CSI_SNAPSHOT).start()
                elif name not in busy:
                    feeds[name] = UsbFeed(name).start()
                self._stop.wait(STAGGER_S)
            if busy:
                self.note = f"{'、'.join(NAMES[b] for b in busy)}相機被其他程式使用中，沒有拍"
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not self._stop.is_set() and \
                    not all(f.latest()[0] is not None for f in feeds.values()):
                self._stop.wait(0.1)
            self._stop.wait(WARMUP_S)
            for name, feed in feeds.items():
                frame = feed.latest()[0]
                if frame is not None:
                    frames[name] = frame.copy()
        finally:
            for feed in feeds.values():          # 拍完馬上關，不佔 USB、不佔 CPU
                try:
                    feed.close()
                except Exception:  # noqa: BLE001
                    pass
        return frames

    def _run(self):
        try:
            frames = self._grab()
            if self._stop.is_set():
                return
            if not frames:
                self.error = "沒有拍到任何相機的畫面"
                return
            self.note = "拼接中…"
            from aruco_localizer import ArucoLocalizer, car_from_camera
            arena = json.loads((SERVER / "arena_map.json").read_text(encoding="utf-8"))
            mapper = bev_mapper(ArucoLocalizer(arena, CAR), car_from_camera, tuple(float(v) for v in CAR["body_m"]))
            items = [(n, CAR["camera_mounts"][n], frames[n]) for n in ("front", "rear", "left", "right") if n in frames]
            img = mapper.mosaic(items, filled=len(items) == 4)
            self.taken = time.strftime("%H:%M:%S")
            folder = SNAP_DIR / time.strftime("%Y%m%d-%H%M%S")
            folder.mkdir(parents=True, exist_ok=True)
            for name, frame in frames.items():
                cv2.imwrite(str(folder / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(folder / "bev.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            missing = [NAMES[n] for n in ("front", "rear", "left", "right") if n not in frames]
            self.note = (f"拍攝於 {self.taken}" + (f"（少了{'、'.join(missing)}）" if missing else "")
                         + "　再按「鳥瞰」重拍")
            with self._lock:
                self._frame, self._stamp = img, time.monotonic()
        except Exception as exc:  # noqa: BLE001
            self.error = f"鳥瞰圖失敗：{exc}"


class CameraPanel:
    def __init__(self, parent, width: int, height: int, bg: str, fg: str, font, start: str | None = "gimbal"):
        self.w, self.h = width, height
        self.frame = tk.Frame(parent, bg=bg)
        box = tk.Frame(self.frame, bg="#000", width=width, height=height)   # 固定像素大小（Label 只有文字時寬高是字數）
        box.pack_propagate(False)
        box.pack()
        self.view = tk.Label(box, bg="#000", fg="#bbb", text="相機關閉", font=font, compound="center")
        self.view.pack(fill="both", expand=True)
        self.status = tk.Label(self.frame, text="", fg="#9a9a9a", bg=bg, font=font, anchor="w")
        self.status.pack(fill="x")
        row = tk.Frame(self.frame, bg=bg)
        row.pack(anchor="w", pady=(4, 0))
        self.choice = tk.StringVar(value="")
        per_row = 7 if width >= 400 else 4                 # 畫面窄（小螢幕）時按鈕排兩排
        for i, (text, role) in enumerate(CAMERAS + [("關閉", "")]):
            tk.Radiobutton(row, text=text, value=role, variable=self.choice, indicatoron=0, width=5,
                           command=lambda: self.select(self.choice.get()), fg=fg, bg="#333", selectcolor="#2f6db5",
                           activebackground="#444", activeforeground=fg).grid(row=i // per_row, column=i % per_row,
                                                                               padx=1, pady=1)
        self.feed = None
        self.role = ""
        self.opened_at = 0.0
        self.note = ""
        self._img = None
        self._shown = 0.0
        self._count = (0, time.monotonic(), 0.0)
        self.yielded = None          # 讓給跑車程式時原本選的畫面
        self.program = ""
        self._quiet_since = None
        self._last_check = 0.0
        if start:
            self.choice.set(start)
            self.frame.after(300, lambda: self.select(start))
        self.frame.after(REFRESH_MS, self._tick)

    # -- switching -----------------------------------------------------------------------------------
    def select(self, role: str):
        if role and self.program:                # 跑車程式在跑：不開相機，記下來，等它結束再開
            self.yielded = role
            self.choice.set("")
            self._show_yielded()
            return
        if role == self.role and self.feed is not None and role != "bev":   # 鳥瞰：再按一次 = 重拍
            return
        old, self.feed, self.role, self.note = self.feed, None, role, ""
        if old is not None:                      # 舊的相機在背景關掉，不卡住畫面
            threading.Thread(target=old.close, daemon=True).start()
        self._img = None
        self.view.config(image="", text="相機關閉" if not role else "開啟中…")
        if not role:
            return
        if role == "bev":
            delay = 0.8 if old is not None else 0.0      # 剛關的 CSI 要一點時間釋放
            self.frame.after(int(delay * 1000), lambda: self._start(role, BevSnapshot))
        elif role in ("front", "rear"):
            num = int(CAR["camera_mounts"][role]["camera_num"])
            delay = 0.6 if old is not None else 0.0
            self.frame.after(int(delay * 1000), lambda: self._start(role, lambda: CsiGrabber(role, num, 640, 480)))
        else:
            users = usb_users(role)
            if users:
                self.note = f"{NAMES[role]}相機被其他程式使用中（{users}），不搶用"
                self.view.config(text=self.note, wraplength=self.w - 20)
                return
            delay = 0.3 if old is not None else 0.0
            self.frame.after(int(delay * 1000), lambda: self._start(role, lambda: UsbFeed(role)))

    def _start(self, role, make):
        if self.role != role or self.program:
            return
        self.feed = make().start()
        self.opened_at = time.monotonic()

    def close(self):
        if self.feed is not None:
            self.feed.close()
            self.feed = None

    def _show_yielded(self):
        msg = f"{self.program} 執行中：相機讓給它用" + (f"，結束後自動開回「{NAMES[self.yielded]}」" if self.yielded else "")
        self._img = None
        self.view.config(image="", text=msg, wraplength=self.w - 20)
        self.status.config(text=msg)

    def _check_program(self):
        now = time.monotonic()
        if now - self._last_check < 1.0:
            return
        self._last_check = now
        prog = car_program()
        if prog:
            self._quiet_since = None
            if not self.program:
                self.program = prog
                if self.role:
                    self.yielded = self.role
                    self.select("")              # 全部關掉
                    self.choice.set("")
                self._show_yielded()
            self.program = prog
        elif self.program:
            self._quiet_since = self._quiet_since or now
            if now - self._quiet_since >= RESUME_AFTER_S:
                self.program, back, self.yielded = "", self.yielded, None
                self._quiet_since = None
                if back:
                    self.choice.set(back)
                    self.select(back)
                else:
                    self.view.config(text="相機關閉")

    # -- drawing -------------------------------------------------------------------------------------
    def _tick(self):
        try:
            self._check_program()
            if not self.program:
                self._draw()
        finally:
            self.frame.after(REFRESH_MS, self._tick)

    def _draw(self):
        feed = self.feed
        name = NAMES.get(self.role, "")
        if feed is None:
            if self.role and not self.note:
                self.status.config(text=f"{name}：開啟中…")
            elif not self.role:
                self.status.config(text="相機關閉（CPU 不增加）")
            return
        frame, stamp = feed.latest()
        err = getattr(feed, "error", "")
        if frame is None:
            waited = time.monotonic() - self.opened_at
            msg = f"{name}：開啟中… {waited:.1f} 秒"
            if self.role == "bev":
                msg = f"鳥瞰：{feed.note} {waited:.0f} 秒（車子請保持不動）"
            if err:
                msg = f"{name}：{'被其他程式使用中' if 'busy' in err.lower() else err[:60]}"
            self.status.config(text=msg)
            if not self._img:
                self.view.config(text=msg, wraplength=self.w - 20)
            return
        n, t0, fps = self._count
        if stamp != self._shown:
            n += 1
        now = time.monotonic()
        if now - t0 >= 1.0:
            fps, n, t0 = n / (now - t0), 0, now
        self._count = (n, t0, fps)
        if stamp == self._shown:
            return
        self._shown = stamp
        h, w = frame.shape[:2]
        s = min(self.w / w, self.h / h)
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        self._img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.view.config(image=self._img, text="")
        age = now - stamp
        extra = getattr(feed, "note", "")
        if self.role == "bev":
            self.status.config(text=f"鳥瞰  {extra}")
            return
        self.status.config(text=f"{name}  {w}x{h}  顯示 {fps:4.1f} 張/秒" + (f"  畫面延遲 {age:.1f} 秒" if age > 0.7 else "")
                           + (f"  {extra}" if extra else ""))
