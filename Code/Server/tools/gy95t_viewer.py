"""GY-95T / GY-25T 即時檢視器

用法:
    python gy95t_viewer.py                      # 自動偵測型號與波特率 (預設 COM4)
    python gy95t_viewer.py --model 25t          # 指定型號 (95t / 25t)
    python gy95t_viewer.py --port /dev/ttyUSB0  # 指定其他埠
    python gy95t_viewer.py --test               # 只在終端機印出資料
    python gy95t_viewer.py --camera 前          # 右側相機區先顯示前相機（雲台/前/後/左/右/關/無；只在車上的 Pi 有）

兩種模組共用同一套協議 (A4 03 起始寄存器 數量 DATA SUM，SUM = 前面相加低 8 位)，差異:
                GY-95T (九軸)                 GY-25T (六軸)
    預設波特率  115200                        9600
    位元組順序  小端序 (L 在低位址)           大端序 (H 在低位址)
    預設輸出    0x08~0x22 (27 位元組)         0x14~0x1B (8 位元組)
    寄存器      08 Acc / 0E Gyro / 14 Roll,Pitch,Yaw (/100)
                1A Level, 1B 溫度, 1D Mag     1A 溫度
                23 四元數 (/10000)
    加陀校準    0x05 = 0x57                   0x06 = 0x10
    其他        磁場校準 0x05 = 58/59/5A      航向歸零 0x06 = 0x01
    依據: SM06-GY-95 九軸模組說明書 v1.1、GY-25T 角度模塊使用說明 v1.0

注意: 連續輸出模式下，「讀寄存器」指令會同時改變模組之後連續輸出的內容。
本程式讀完量程 (0x07) 後會自動送出原本的讀指令把串流還原。
"""
import argparse
import csv
import math
import struct
import threading
import time

import serial

ACC_FS = (2, 4, 8, 16)           # g
GYR_FS = (250, 500, 1000, 2000)  # °/s
MAG_FS = (2, 8, 12, 30)          # Gauss

MODELS = {
    "95t": {
        "name": "GY-95T", "baud": 115200, "endian": "<", "stream": (0x08, 0x1B),
        # 欄位: (起始寄存器, struct 格式, 比例)
        "fields": {
            "acc": (0x08, "3h", 1), "gyr": (0x0E, "3h", 1), "ang": (0x14, "3h", 100),
            "level": (0x1A, "B", 1), "temp": (0x1B, "h", 100), "mag": (0x1D, "3h", 1),
            "quat": (0x23, "4h", 10000),
        },
        "cmd": {"accgyro": (0x05, 0x57), "mag_start": (0x05, 0x58), "mag_stop": (0x05, 0x59),
                "mag_save": (0x05, 0x5A)},
    },
    "25t": {
        "name": "GY-25T", "baud": 9600, "endian": ">", "stream": (0x14, 0x08),
        "fields": {
            "acc": (0x08, "3h", 1), "gyr": (0x0E, "3h", 1), "ang": (0x14, "3h", 100),
            "temp": (0x1A, "h", 100),
        },
        "cmd": {"accgyro": (0x06, 0x10), "yaw_zero": (0x06, 0x01)},
    },
}


def decode_cfg(model, v):
    """寄存器 0x07: bit3-2 加計量程, bit1-0 陀螺量程 (兩型號相同)"""
    cfg = {"raw": v, "acc_fs": ACC_FS[(v >> 2) & 3], "gyr_fs": GYR_FS[v & 3]}
    if model == "95t":
        cfg["mag_fs"] = MAG_FS[(v >> 4) & 3]
        cfg["mount"] = "水平" if v & 0x80 else "垂直"
    else:
        cfg["mount"] = "水平" if (v >> 4) & 3 == 1 else "垂直"
    return cfg


def checksum(b):
    return bytes(b) + bytes([sum(b) & 0xFF])


def parse_regs(model, start, payload):
    """把一段從 start 開始的寄存器資料解析成欄位；資料幀沒涵蓋到的欄位為 None。"""
    m = MODELS[model]
    end = start + len(payload)
    out = {}
    for key, (reg, fmt, scale) in m["fields"].items():
        size = struct.calcsize(m["endian"] + fmt)
        if start <= reg and reg + size <= end:
            vals = struct.unpack_from(m["endian"] + fmt, payload, reg - start)
            if scale != 1:
                vals = tuple(v / scale for v in vals)
            out[key] = vals if len(vals) > 1 else vals[0]
        else:
            out[key] = None
    ang = out.pop("ang")
    out["roll"], out["pitch"], out["yaw"] = ang if ang else (None, None, None)
    return out


def parse_payload(d, model="95t", start=0x08):
    """相容舊版: 解析從 0x08 開始的資料。"""
    return parse_regs(model, start, d)


def _plausible(model, start, payload):
    d = parse_regs(model, start, payload)
    if d["roll"] is None:
        return False
    ok = abs(d["roll"]) <= 180 and abs(d["pitch"]) <= 90 and abs(d["yaw"]) <= 180
    if d["temp"] is not None:
        ok = ok and -30 <= d["temp"] <= 90
    if d["acc"] is not None:
        ok = ok and 0.3 < math.sqrt(sum(a * a for a in d["acc"])) / 2048 < 3  # 靜止約 1 g (±16g 量程)
    return ok


def detect(port, bauds=(115200, 9600, 230400, 57600, 38400, 19200), secs=1.0):
    """嘗試各波特率，依資料幀內容判斷型號。回傳 (model, baud) 或 (None, None)。"""
    for baud in bauds:
        try:
            s = serial.Serial(port, baud, timeout=0.1)
        except serial.SerialException:
            raise
        s.reset_input_buffer()
        buf = bytearray()
        t = time.time()
        while time.time() - t < secs:
            buf += s.read(512)
        s.close()
        frames = []
        i = 0
        while True:
            i = buf.find(b"\xA4\x03", i)
            if i < 0 or i + 4 > len(buf):
                break
            n = buf[i + 3]
            f = bytes(buf[i:i + 5 + n])
            if len(f) == 5 + n and sum(f[:-1]) & 0xFF == f[-1]:
                frames.append((f[2], f[4:-1]))
                i += 5 + n
            else:
                i += 1
        if len(frames) < 3:
            continue
        frames = frames[-20:]
        score = {m: sum(_plausible(m, st, p) for st, p in frames) for m in MODELS}
        best = max(score, key=score.get)
        if score[best] >= len(frames) * 0.8:
            return best, baud
    return None, None


class Reader(threading.Thread):
    """背景執行緒：持續讀取序列埠並解析最新一幀。"""

    def __init__(self, port, baud, model="95t"):
        super().__init__(daemon=True)
        self.model = model
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.buf = bytearray()
        self.latest = None
        self.cfg = None
        self._last_req = 0.0
        self._stream = MODELS[model]["stream"]
        self._need_restore = False
        self._last_restore = 0.0
        self.frames = 0
        self.bad = 0
        self.last_ack = None
        self.running = True
        self.lock = threading.Lock()
        # 陀螺儀航向：角速度在「上」方向 (加速度計) 的分量積分，順時針為正。不受磁場影響，但會慢慢漂移。
        # 開啟時先量 GYRO_BIAS_S 秒的零點 (請保持靜止)。
        self.gyro_heading = 0.0
        self.gyro_bias = [0.0, 0.0, 0.0]
        self._gt = None
        self._bias_buf = []
        self._bias_until = time.monotonic() + 2.5

    def start_bias(self, seconds=2.0):
        """重新量陀螺儀零點：接下來 seconds 秒請保持靜止。"""
        with self.lock:
            self._bias_buf = []
            self._bias_until = time.monotonic() + seconds

    def _integrate(self, d):
        if d.get("gyr") is None or d.get("acc") is None:
            return
        t = time.monotonic()
        fs = self.cfg["gyr_fs"] if self.cfg else 2000
        w = [g * fs / 32768.0 for g in d["gyr"]]
        if self._bias_buf is not None:
            if t < self._bias_until:
                self._bias_buf.append(w)
                self._gt = t
                return
            n = len(self._bias_buf)
            if n >= 20:
                self.gyro_bias = [sum(v[k] for v in self._bias_buf) / n for k in range(3)]
            self._bias_buf = None
        a = d["acc"]
        na = math.sqrt(sum(v * v for v in a))
        if na < 1e-6:
            return
        rate = sum((w[k] - self.gyro_bias[k]) * a[k] / na for k in range(3))   # 繞「上」逆時針為正 (度/秒)
        if self._gt is not None:
            self.gyro_heading = (self.gyro_heading - rate * min(t - self._gt, 0.1)) % 360
        self._gt = t

    def _send(self, b):
        try:
            self.ser.write(checksum(b))
        except serial.SerialException:
            pass

    def request_cfg(self):
        # 讀寄存器 0x07 一個: A4 03 07 01 SUM (收到後由 run 迴圈還原串流)
        self._send(b"\xA4\x03\x07\x01")

    def run(self):
        while self.running:
            now = time.time()
            if self.cfg is None and self.frames and now - self._last_req > 0.5:
                self._last_req = now
                self.request_cfg()
            if self._need_restore and now - self._last_restore > 0.3:
                self._need_restore = False
                self._last_restore = now
                self._send(bytes([0xA4, 0x03, *self._stream]))
            try:
                chunk = self.ser.read(256)
            except (serial.SerialException, TypeError, OSError):   # 關閉視窗時埠被關掉
                break
            if chunk:
                self.buf += chunk
                self._parse()

    def write_reg(self, reg, val):
        """寫單一寄存器 (功能碼 0x06)。模組成功時會回傳相同的 5 位元組。"""
        with self.lock:
            self.last_ack = None
        self._send(bytes([0xA4, 0x06, reg, val]))

    def _parse(self):
        buf = self.buf
        while True:
            i = buf.find(b"\xA4")
            if i < 0:
                buf.clear()
                return
            if i:
                del buf[:i]
            if len(buf) < 2:
                return
            fn = buf[1]
            if fn in (0x06, 0x86):  # 寫入回應 / 錯誤回應
                need = 5 if fn == 0x06 else 4
                if len(buf) < need:
                    return
                f = bytes(buf[:need])
                if sum(f[:-1]) & 0xFF == f[-1]:
                    with self.lock:
                        self.last_ack = ("ok", f[2], f[3]) if fn == 0x06 else ("err", f[2], None)
                    del buf[:need]
                else:
                    del buf[:1]
                continue
            if fn != 0x03:
                del buf[:1]
                continue
            if len(buf) < 4:
                return
            n = buf[3]
            total = 4 + n + 1
            if len(buf) < total:
                return
            frame = bytes(buf[:total])
            if sum(frame[:-1]) & 0xFF != frame[-1]:
                self.bad += 1
                del buf[:1]
                continue
            start, payload = frame[2], frame[4:-1]
            with self.lock:
                if start <= 0x07 < start + n:
                    self.cfg = decode_cfg(self.model, payload[0x07 - start])
                if start == 0x07 and n == 1:
                    self._need_restore = True  # 串流被讀量程指令改掉了，需要還原
                else:
                    d = parse_regs(self.model, start, payload)
                    if d["roll"] is not None:
                        self._stream = (start, n)
                        self.latest = d
                        self.frames += 1
                        self._integrate(d)
            del buf[:total]

    def close(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


def fmt_vec(v, f="{:6d}"):
    return "(" + ", ".join(f.format(x) for x in v) + ")" if v is not None else "--"


def run_test(port, baud, model):
    r = Reader(port, baud, model)
    r.start()
    seen = 0
    t0 = time.time()
    while seen < 10 and time.time() - t0 < 5:
        time.sleep(0.1)
        with r.lock:
            d, n = r.latest, r.frames
        if d and n > seen:
            seen = n
            line = (f"Roll {d['roll']:7.2f}  Pitch {d['pitch']:7.2f}  Yaw {d['yaw']:7.2f}  "
                    f"Temp {d['temp']:6.2f}°C  Acc {fmt_vec(d['acc'])}  Gyr {fmt_vec(d['gyr'])}")
            if model == "95t":
                line += f"  Mag {fmt_vec(d['mag'])}  Level {d['level']}"
            print(line)
    print(f"[{MODELS[model]['name']} @ {baud}] 有效幀 {r.frames}，校驗失敗 {r.bad}")
    print("量程設定 (0x07):", r.cfg)
    r.close()


def run_gui(port, baud, model, mount=None, camera="gimbal", screen="auto"):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    M = MODELS[model]
    try:
        reader = Reader(port, baud, model)
    except serial.SerialException as e:
        messagebox.showerror("IMU 檢視器", f"無法開啟 {port}：\n{e}\n\n請先關閉其他佔用此埠的程式（例如 GY25T 上位機）。")
        return
    reader.start()

    root = tk.Tk()
    root.title(f"{M['name']} 檢視器 - {port} @ {baud}")
    root.configure(bg="#1e1e1e")

    fg, bg = "#e8e8e8", "#1e1e1e"
    small = root.winfo_screenwidth() < 1280 or root.winfo_screenheight() < 800   # 例如車上 1024 x 600 的小螢幕
    if screen != "auto":
        small = screen == "small"
    font_big = ("Consolas", 16 if small else 26, "bold")
    font = ("Consolas", 9 if small else 12)

    left = tk.Frame(root, bg=bg)
    left.pack(side="left", padx=12, pady=12, anchor="n")
    right = tk.Frame(root, bg=bg)
    right.pack(side="left", padx=12, pady=12, anchor="n")
    # 右欄：上面是按鈕，下面是可以收起來的「詳細資訊」（溫度、原始資料、狀態、安裝方向），預設收起
    btn_slot = tk.Frame(right, bg=bg)
    btn_slot.pack(anchor="w")
    info = tk.Frame(right, bg=bg)

    # 地圖區（tools/map_panel.py）和詳細資訊輪流放在按鈕下面：展開詳細資訊時地圖先收起來，免得視窗太高
    map_box = {"panel": None}
    INFO_CLOSED = "▸ 顯示詳細資訊（溫度、原始資料、安裝方向）"

    def toggle_info():
        mp = map_box["panel"]
        if info.winfo_ismapped():
            info.pack_forget()
            info_btn.config(text=INFO_CLOSED)
            if mp is not None:
                mp.frame.pack(anchor="w", pady=(8, 0))
        else:
            if mp is not None:
                mp.frame.pack_forget()
            info.pack(anchor="w", fill="x")
            info_btn.config(text="▾ 收起詳細資訊（回到地圖）" if mp is not None else "▾ 收起詳細資訊")

    info_btn = tk.Button(right, text=INFO_CLOSED, command=toggle_info, relief="flat",
                         fg=fg, bg="#2a2a2a", activebackground="#3a3a3a", activeforeground=fg, anchor="w")
    info_btn.pack(anchor="w", fill="x", pady=(10, 0))

    # 相機區（只在車上的 Pi 有；別的電腦 import 失敗就不顯示）。版面排好後才決定放哪裡、多大（見 place_camera）
    CameraPanel = None
    cam_panel = None
    if camera is not None:
        try:
            from camera_panel import CameraPanel
        except Exception as e:  # noqa: BLE001
            print(f"不顯示相機區：{e}")

    W = (150 if CameraPanel else 170) if small else 260   # 小螢幕有相機時縮小一點，三個儀表排一列

    # 左上：Roll / Pitch / Yaw 三個大數字排一列
    big = {}
    big_box = tk.Frame(left, bg=bg)
    big_box.pack(anchor="w", fill="x", pady=(0, 8))
    for col, (name, color) in enumerate((("Roll", "#6aa9ff"), ("Pitch", "#ff6b6b"), ("Yaw", "#59d47a"))):
        tk.Label(big_box, text=name, fg=color, bg=bg, font=font).grid(row=0, column=col, sticky="w", padx=(0, 10))
        lbl = tk.Label(big_box, text="--", fg=color, bg=bg, font=font_big, width=8, anchor="w")   # 最長 -180.00°
        lbl.grid(row=1, column=col, sticky="w", padx=(0, 10))
        big[name] = lbl

    top_row = tk.Frame(left, bg=bg)
    top_row.pack()
    horizon = tk.Canvas(top_row, width=W, height=W, bg="#000", highlightthickness=0)
    horizon.pack(side="left")
    side = tk.Canvas(top_row, width=W, height=W, bg="#000", highlightthickness=0)
    side.pack(side="left", padx=(8, 0))
    gauge_caption = tk.Label(left, text="左：Roll（外部視角從車尾看）　右：Pitch（從側面看，車頭在右）", fg=fg, bg=bg,
                             font=font)
    # 下面的說明列不顯示（儀表裡已標 Roll、Pitch）；變數留著給小螢幕版面用
    comp_row = tk.Frame(left, bg=bg)           # 航向圖（靠左）＋右邊的電池與馬達面板
    comp_row.pack(anchor="w", fill="x", pady=(10, 0))
    comp_box = tk.Frame(left, bg=bg)           # 航向儀；小螢幕加相機時會移到上面那一列
    comp_box.pack(in_=comp_row, side="left", anchor="n")
    compass = tk.Canvas(comp_box, width=W, height=W, bg="#000", highlightthickness=0)
    compass.pack()
    tk.Label(comp_box, text="航向（上方 0° = 歸零時車頭方向）", fg=fg, bg=bg, font=font).pack()

    trow = tk.Frame(info, bg=bg)
    trow.pack(anchor="w", pady=(10, 0))
    tk.Label(trow, text="溫度", fg="#ffb84d", bg=bg, font=font).pack(side="left", padx=(0, 10))
    big["溫度"] = tk.Label(trow, text="--", fg="#ffb84d", bg=bg, font=(font[0], font[1] + 4, "bold"))
    big["溫度"].pack(side="left")

    detail = tk.Label(info, text="", fg=fg, bg=bg, font=font, justify="left")
    detail.pack(anchor="w", pady=(8, 0))
    status = tk.Label(info, text="", fg="#9a9a9a", bg=bg, font=("Consolas", 10), justify="left")
    status.pack(anchor="w", pady=(8, 0))

    offsets = {"yaw": 0.0}
    logger = {"file": None, "writer": None}

    # 安裝方向選項
    # GY-95T 預設值為車上實測結果 (X 朝車尾、Yaw 逆時針為正、右傾 Roll 為負)；
    # GY-25T 尚未實測，預設全部不勾。
    is95 = model == "95t"
    # 模組 X 軸指向 → X 軸相對車頭的順時針角度
    X_DIRS = {"車頭": 0, "右側": 90, "車尾": 180, "左側": 270}
    opt_xdir = tk.StringVar(value="車尾" if is95 else "車頭")
    opt_rev_yaw = tk.BooleanVar(value=is95)    # 模組 Yaw 逆時針為正，指南針需順時針為正
    opt_rev_roll = tk.BooleanVar(value=is95)   # X 朝車尾安裝時，右傾 Roll 為負
    opt_rev_pitch = tk.BooleanVar(value=False)
    if mount:                                  # 命令列指定的安裝方向 (tools/imu_mount_check.py 量出來的)
        if mount.get("xdir") in X_DIRS:
            opt_xdir.set(mount["xdir"])
        for key, var in (("rev_yaw", opt_rev_yaw), ("rev_roll", opt_rev_roll), ("rev_pitch", opt_rev_pitch)):
            if mount.get(key) is not None:
                var.set(bool(mount[key]))
    opt_outside = tk.BooleanVar(value=True)    # 外部視角：地面固定，畫一台從車尾看的小車跟著車子傾斜
    # 航向用磁力計 (傾斜補償) 直接算：2026-11-19 車頭朝南平放時，模組 Yaw 換算出 76.5° (差約 90°，
    # 模組 Yaw 看起來是以它的 Y 軸為基準)，磁場向量算出 184°。
    # 航向來源：2026-11-20 實驗室地面的水平磁場只有約 0.13 G (正常 0.36 G)，車子轉 90° 磁力計航向只變
    # 10~20°，所以預設改用陀螺儀 (只算轉了幾度，「航向歸零」後就是相對起點的角度)。
    HEADING_SOURCES = ("陀螺儀", "磁力計", "模組 Yaw")
    opt_src = tk.StringVar(value="陀螺儀")
    NOSE = {0: (1, 0, 0), 180: (-1, 0, 0), 90: (0, 1, 0), 270: (0, -1, 0)}   # 車頭在模組座標 (Z 朝上安裝)

    def mag_heading(d):
        """車頭的磁航向 (0 = 磁北，順時針為正，0~360)：用加速度計找出「上」，把磁場投影到水平面
        得到北，東 = 北 × 上，再看車頭向量在東、北上的分量。沒有磁場或加速度資料時為 None。"""
        if d.get("mag") is None or d.get("acc") is None:
            return None
        a, m = d["acc"], d["mag"]
        na = math.sqrt(sum(v * v for v in a))
        if na < 1e-6:
            return None
        up = [v / na for v in a]
        mu = sum(mi * ui for mi, ui in zip(m, up))
        north = [mi - mu * ui for mi, ui in zip(m, up)]
        east = [north[1] * up[2] - north[2] * up[1], north[2] * up[0] - north[0] * up[2],
                north[0] * up[1] - north[1] * up[0]]
        f = NOSE[X_DIRS[opt_xdir.get()]]
        fe = sum(fi * ei for fi, ei in zip(f, east))
        fn = sum(fi * ni for fi, ni in zip(f, north))
        if abs(fe) + abs(fn) < 1e-6:
            return None
        return math.degrees(math.atan2(fe, fn)) % 360

    def vehicle(d):
        """依安裝選項把模組角度換成車身角度 (roll, pitch, heading 0~360)。
        X 軸朝左/右時，模組的 Roll 軸變成車身的俯仰軸，所以 Roll/Pitch 互換。
        正負號由三個「反轉」選項 (作用在換算後的車身角度) 決定。"""
        xoff = X_DIRS[opt_xdir.get()]
        if xoff in (90, 270):
            roll, pitch = d["pitch"], d["roll"]
        else:
            roll, pitch = d["roll"], d["pitch"]
        roll = -roll if opt_rev_roll.get() else roll
        pitch = -pitch if opt_rev_pitch.get() else pitch
        yaw = -d["yaw"] if opt_rev_yaw.get() else d["yaw"]
        heading = (yaw - xoff) % 360
        src = opt_src.get()
        if src == "磁力計":
            mh = mag_heading(d)
            if mh is not None:
                heading = mh
        elif src == "陀螺儀" and d.get("gyr") is not None:
            heading = reader.gyro_heading
        # 沒有角速度資料 (例如 GY-25T 預設只輸出角度) 時，「陀螺儀」改用模組 Yaw：六軸模組的 Yaw 本身就是陀螺儀積分
        return roll, pitch, (heading - offsets["yaw"]) % 360

    def zero_yaw():
        with reader.lock:
            d = reader.latest
        if d:
            offsets["yaw"] = 0.0
            offsets["yaw"] = vehicle(d)[2]

    def reset_zero():
        offsets["yaw"] = 0.0

    opts = tk.LabelFrame(info, text="安裝方向", fg=fg, bg=bg, font=("Consolas", 10))
    opts.pack(anchor="w", fill="x", pady=(14, 0))
    xrow = tk.Frame(opts, bg=bg)
    xrow.pack(anchor="w")
    tk.Label(xrow, text="模組 X 軸朝向:", fg=fg, bg=bg).pack(side="left")
    for name in X_DIRS:
        tk.Radiobutton(xrow, text=name, value=name, variable=opt_xdir, fg=fg, bg=bg, selectcolor="#333",
                       activebackground=bg, activeforeground=fg).pack(side="left")
    tk.Label(opts, text="（朝左/右時 Roll 與 Pitch 自動互換）", fg="#888", bg=bg).pack(anchor="w")
    for text, var in (("反轉 Yaw", opt_rev_yaw),
                      ("反轉 Roll", opt_rev_roll), ("反轉 Pitch", opt_rev_pitch)):
        tk.Checkbutton(opts, text=text, variable=var, fg=fg, bg=bg, selectcolor="#333",
                       activebackground=bg, activeforeground=fg).pack(anchor="w")
    note_wrap = 400 if small else 0
    tk.Checkbutton(opts, text="外部視角（從車尾看，畫面跟著車子傾斜；不勾 = 飛機座艙視角）", variable=opt_outside,
                   wraplength=note_wrap, justify="left",
                   fg=fg, bg=bg, selectcolor="#333", activebackground=bg, activeforeground=fg).pack(anchor="w")
    srow = tk.Frame(opts, bg=bg)
    srow.pack(anchor="w")
    tk.Label(srow, text="航向來源:", fg=fg, bg=bg).pack(side="left")
    for name in HEADING_SOURCES:
        if name == "磁力計" and not is95:
            continue
        tk.Radiobutton(srow, text=name, value=name, variable=opt_src, fg=fg, bg=bg, selectcolor="#333",
                       activebackground=bg, activeforeground=fg).pack(side="left")
    tk.Label(opts, text="（陀螺儀：只算轉了幾度，不受磁場影響，會慢慢漂移；開啟或按「陀螺儀零點」時請靜止 2 秒）",
             wraplength=note_wrap, justify="left",
             fg="#888", bg=bg).pack(anchor="w")

    def toggle_log():
        if logger["file"]:
            logger["file"].close()
            logger["file"] = logger["writer"] = None
            log_btn.config(text="開始記錄 CSV")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile=time.strftime("gy95t_%Y%m%d_%H%M%S.csv"))
        if not path:
            return
        f = open(path, "w", newline="", encoding="utf-8-sig")
        w = csv.writer(f)
        w.writerow(["time", "roll", "pitch", "heading", "yaw_raw", "temp", "acc_x", "acc_y", "acc_z",
                    "gyr_x", "gyr_y", "gyr_z", "mag_x", "mag_y", "mag_z", "level"])
        logger["file"], logger["writer"] = f, w
        log_btn.config(text="停止記錄")

    btns = tk.Frame(btn_slot, bg=bg)
    btns.pack(anchor="w")
    tk.Button(btns, text="航向歸零", command=zero_yaw, width=10).pack(side="left", padx=(0, 8))
    tk.Button(btns, text="取消歸零", command=reset_zero, width=10).pack(side="left", padx=(0, 8))
    tk.Button(btns, text="陀螺儀零點", command=lambda: reader.start_bias(2.0), width=10).pack(side="left", padx=(0, 8))
    log_btn = tk.Button(btns, text="開始記錄 CSV", command=toggle_log, width=14)
    log_btn.pack(side="left", padx=(0, 8))

    # ---------------- 校準視窗 ----------------
    cal = {"win": None, "collect": False, "pts": [], "last_frame": 0, "pending": None}
    CMD_NAMES = {"accgyro": "加陀校準", "mag_start": "磁場開始校準", "mag_stop": "磁場停止校準",
                 "mag_save": "保存磁場校準", "yaw_zero": "模組航向歸零"}

    def send_cmd(key):
        reg, val = M["cmd"][key]
        reader.write_reg(reg, val)
        cal["pending"] = (key, time.time())
        cal_status.config(text=f"已送出「{CMD_NAMES[key]}」，等待模組回應…", fg="#ffd400")

    def do_accgyro():
        note = "按確定後約 2 秒完成並自動保存。" if is95 else "按確定後開始校準（GY-25T 需另外按「保存設定」才會斷電保存）。"
        if messagebox.askokcancel("加陀校準", "請確認感測器放在水平面上並保持靜止，\n" + note,
                                  parent=cal["win"]):
            send_cmd("accgyro")

    def do_yaw_zero():
        send_cmd("yaw_zero")
        cal_hint.config(text="已送出模組航向歸零：GY-25T 會把目前方向設為 Yaw = 0。")

    def do_save_cfg():
        if messagebox.askokcancel("保存設定", "把模組目前的所有設定寫入斷電保存（0x05 = 0x55）？", parent=cal["win"]):
            reader.write_reg(0x05, 0x55)
            cal["pending"] = ("save_cfg", time.time())
            cal_status.config(text="已送出「保存設定」，等待模組回應…", fg="#ffd400")

    CMD_NAMES["save_cfg"] = "保存設定"
    M["cmd"].setdefault("save_cfg", (0x05, 0x55))

    def do_mag_start():
        cal["pts"].clear()
        cal["collect"] = True
        send_cmd("mag_start")
        cal_hint.config(text="校準中：拿起整台車，慢慢往各方向轉動\n（水平轉圈、車頭朝上/朝下、左右側翻），約 30~60 秒，\n直到下方三個圖都形成完整的圓。")

    def do_mag_stop():
        cal["collect"] = False
        send_cmd("mag_stop")
        cal_hint.config(text="已停止。若圖形是完整的圓，按「保存磁場校準」；\n不滿意可重新開始。")

    def do_mag_save():
        send_cmd("mag_save")
        cal_hint.config(text="已保存。把車放回地面，觀察 Level 是否進入 80~120（越接近 100 越好）。")

    PLOT = 180
    plots = []

    def open_cal():
        nonlocal cal_status, cal_hint, cal_level
        if cal["win"] and cal["win"].winfo_exists():
            cal["win"].lift()
            return
        w = tk.Toplevel(root)
        w.title(f"{M['name']} 校準")
        w.configure(bg=bg)
        cal["win"] = w
        top = tk.Frame(w, bg=bg)
        top.pack(fill="x", padx=10, pady=8)
        b1 = tk.Frame(top, bg=bg)
        b1.pack(side="left")
        tk.Button(b1, text="① 加陀校準", width=14, command=do_accgyro).grid(row=0, column=0, padx=3, pady=3)
        if is95:
            tk.Button(b1, text="② 磁場開始校準", width=14, command=do_mag_start).grid(row=0, column=1, padx=3, pady=3)
            tk.Button(b1, text="③ 磁場停止校準", width=14, command=do_mag_stop).grid(row=0, column=2, padx=3, pady=3)
            tk.Button(b1, text="④ 保存磁場校準", width=14, command=do_mag_save).grid(row=0, column=3, padx=3, pady=3)
            hint = "磁場校準步驟：② 開始 → 轉動整台車 → ③ 停止 → ④ 保存"
        else:
            tk.Button(b1, text="模組航向歸零", width=14, command=do_yaw_zero).grid(row=0, column=1, padx=3, pady=3)
            tk.Button(b1, text="保存設定 (0x55)", width=14, command=do_save_cfg).grid(row=0, column=2, padx=3, pady=3)
            hint = ("GY-25T 沒有磁力計，不需磁場校準。\n"
                    "① 加陀校準：水平靜止放置後按下。\n"
                    "模組航向歸零：把目前方向設為 Yaw 0（寫入模組，與主視窗的「航向歸零」不同）。")
        cal_level = tk.Label(top, text="Level --" if is95 else "", fg="#ffb84d", bg=bg,
                             font=("Consolas", 18, "bold"))
        cal_level.pack(side="right", padx=10)
        cal_hint = tk.Label(w, text=hint, fg=fg, bg=bg, font=("Microsoft JhengHei", 10), justify="left")
        cal_hint.pack(anchor="w", padx=12)
        cal_status = tk.Label(w, text="", fg="#9a9a9a", bg=bg, font=("Microsoft JhengHei", 10))
        cal_status.pack(anchor="w", padx=12, pady=(2, 6))
        plots.clear()
        if not is95:
            return
        pf = tk.Frame(w, bg=bg)
        pf.pack(padx=10, pady=(0, 10))
        for k, (title, ix, iy) in enumerate((("Y-X", 0, 1), ("Z-X", 0, 2), ("Z-Y", 1, 2))):
            f = tk.Frame(pf, bg=bg)
            f.grid(row=0, column=k, padx=4)
            c = tk.Canvas(f, width=PLOT, height=PLOT, bg="#000", highlightthickness=0)
            c.pack()
            tk.Label(f, text=title, fg="#aaa", bg=bg, font=("Consolas", 10)).pack()
            plots.append((c, ix, iy))

    cal_status = cal_hint = cal_level = None

    def update_cal(d, n):
        if not (cal["win"] and cal["win"].winfo_exists()):
            return
        lv = d["level"]
        if lv is not None:
            cal_level.config(text=f"Level {lv}", fg="#59d47a" if 80 <= lv <= 120 else "#ffb84d")
        if cal["collect"] and d["mag"] is not None and n != cal["last_frame"]:
            cal["last_frame"] = n
            cal["pts"].append(d["mag"])
            if len(cal["pts"]) > 3000:
                del cal["pts"][::2]
        with reader.lock:
            ack = reader.last_ack
        if cal["pending"]:
            key, t0 = cal["pending"]
            name = CMD_NAMES[key]
            reg, val = M["cmd"][key]
            if ack and ack[0] == "ok" and ack[1] == reg and ack[2] == val:
                cal_status.config(text=f"✔ 模組已確認「{name}」", fg="#59d47a")
                cal["pending"] = None
            elif ack and ack[0] == "err":
                cal_status.config(text=f"✘ 模組回傳錯誤 (代碼 {ack[1]:02X})：「{name}」", fg="#ff6b6b")
                cal["pending"] = None
            elif time.time() - t0 > 1.5:
                cal_status.config(text=f"⚠ 沒收到「{name}」的回應，可再按一次", fg="#ff6b6b")
                cal["pending"] = None

    def draw_plots():
        if cal["win"] and cal["win"].winfo_exists() and plots:
            pts = cal["pts"]
            step = max(1, len(pts) // 1200)
            sample = pts[::step]
            m = max([abs(v) for p in sample for v in p] + [500]) * 1.1
            for c, ix, iy in plots:
                c.delete("all")
                h = PLOT / 2
                c.create_line(h, 0, h, PLOT, fill="#333")
                c.create_line(0, h, PLOT, h, fill="#333")
                for p in sample:
                    x, y = h + p[ix] / m * h, h - p[iy] / m * h
                    c.create_rectangle(x, y, x + 1, y + 1, outline="#59d47a")
                c.create_text(4, 4, text=f"{len(pts)} 點", fill="#777", anchor="nw", font=("Consolas", 8))
        root.after(400, draw_plots)

    tk.Button(btns, text="校準…", command=open_cal, width=8).pack(side="left")

    def wrap(a):
        return (a + 180) % 360 - 180

    # ---------------- 飛機式儀表 ----------------
    SKY, GROUND, BEZEL = "#2f7fd8", "#8a5a2b", "#1a1a1a"
    PX_PER_DEG = 2.4   # 姿態儀: 每度俯仰對應的像素

    def mask_circle(cv, c, R):
        """用一圈很粗的外框把圓形以外的區域蓋掉 (Tk canvas 沒有裁切功能)。"""
        w = c * 1.5 - R
        r = R + w / 2
        cv.create_oval(c - r, c - r, c + r, c + r, outline=BEZEL, width=w)
        cv.create_oval(c - R, c - R, c + R, c + R, outline="#555", width=2)

    def draw_horizon(roll, pitch):
        """姿態儀 (Attitude Indicator)：
        地平線隨 Roll 旋轉、隨 Pitch 上下移動，中央黃色機徽固定不動 (從駕駛艙往前看)。
        右傾 (roll>0) → 地平線右側上升；抬頭 (pitch>0) → 地平線下降、天空變多。"""
        if opt_outside.get():
            draw_outside(roll, pitch)
            return
        cv = horizon
        cv.delete("all")
        c = W / 2
        R = c - 12
        th = math.radians(roll)
        ct, st = math.cos(th), math.sin(th)

        def P(u, v):
            # (u 沿地平線, v 垂直地平線向下) → 螢幕座標，整體逆時針轉 roll
            v = v + pitch * PX_PER_DEG
            return c + u * ct + v * st, c - u * st + v * ct

        L = W * 2
        cv.create_rectangle(0, 0, W, W, fill=SKY, outline="")
        cv.create_polygon(*P(-L, 0), *P(L, 0), *P(L, L), *P(-L, L), fill=GROUND, outline="")
        cv.create_line(*P(-L, 0), *P(L, 0), fill="white", width=2)

        # 俯仰刻度
        for p in range(-40, 45, 5):
            if p == 0:
                continue
            v = -p * PX_PER_DEG
            if abs(v + pitch * PX_PER_DEG) > R - 20:
                continue
            half = 28 if p % 10 == 0 else 12
            cv.create_line(*P(-half, v), *P(half, v), fill="white", width=2 if p % 10 == 0 else 1)
            if p % 10 == 0:
                for side in (-1, 1):
                    x, y = P(side * (half + 12), v)
                    cv.create_text(x, y, text=str(abs(p)), fill="white", font=("Consolas", 9, "bold"),
                                   angle=roll)

        # 傾斜刻度 (固定在上方) 與指針 (隨地平線旋轉)
        rr = R - 4
        for a in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            ar = math.radians(a)
            ln = 12 if a in (0, -30, 30, -60, 60) else 7
            x1, y1 = c + rr * math.sin(ar), c - rr * math.cos(ar)
            x2, y2 = c + (rr - ln) * math.sin(ar), c - (rr - ln) * math.cos(ar)
            cv.create_line(x1, y1, x2, y2, fill="white", width=2)
        tip = rr - 14
        px, py = P(0, -tip - pitch * PX_PER_DEG)
        bx1, by1 = P(-7, -tip + 12 - pitch * PX_PER_DEG)
        bx2, by2 = P(7, -tip + 12 - pitch * PX_PER_DEG)
        cv.create_polygon(px, py, bx1, by1, bx2, by2, fill="#ffd400", outline="")

        mask_circle(cv, c, R)

        # 固定的機徽
        y = c
        cv.create_line(c - 70, y, c - 28, y, fill="#ffd400", width=5)
        cv.create_line(c - 28, y, c - 18, y + 10, fill="#ffd400", width=5)
        cv.create_line(c + 28, y, c + 70, y, fill="#ffd400", width=5)
        cv.create_line(c + 28, y, c + 18, y + 10, fill="#ffd400", width=5)
        cv.create_rectangle(c - 3, y - 3, c + 3, y + 3, fill="#ffd400", outline="")

        # 數值
        cv.create_text(6, 6, text=f"R {roll:+6.1f}°", fill="white", anchor="nw", font=("Consolas", 10, "bold"))
        cv.create_text(W - 6, 6, text=f"P {pitch:+6.1f}°", fill="white", anchor="ne",
                       font=("Consolas", 10, "bold"))

    def draw_outside(roll, pitch):
        """外部視角：從車尾看車子。地面固定水平，小車跟著 Roll 傾斜 (右傾 = 右邊低 = 圖往右傾)，
        以著地那一側的輪子為支點轉動 (像把另一側抬起來)。上方刻度與橘色指針表示傾斜角度。"""
        cv = horizon
        cv.delete("all")
        c = W / 2
        R = c - 12
        k = W / 260                                # 圖形以 260 像素設計，小螢幕等比例縮小
        ground = c + 45 * k
        cv.create_rectangle(0, 0, W, W, fill=SKY, outline="")
        cv.create_rectangle(0, ground, W, W, fill=GROUND, outline="")
        cv.create_line(0, ground, W, ground, fill="white", width=2)

        th = math.radians(max(-80.0, min(80.0, roll)))
        ct, st = math.cos(th), math.sin(th)
        half = 62                                  # 半個車寬 (像素)
        pivot = (c + half * k, ground) if roll >= 0 else (c - half * k, ground)

        def T(x, y):
            # 車身座標 (x 向右, y 向上, 原點 = 兩輪著地點中間) → 螢幕座標，順時針轉 roll
            dx, dy = c + x * k - pivot[0], ground - y * k - pivot[1]
            return pivot[0] + dx * ct - dy * st, pivot[1] + dx * st + dy * ct

        def poly(points, **kw):
            pts = []
            for x, y in points:
                pts += T(x, y)
            cv.create_polygon(pts, **kw)

        for side in (-1, 1):                       # 輪子
            poly([(side * half - 14, 0), (side * half + 14, 0), (side * half + 14, 30), (side * half - 14, 30)],
                 fill="#111", outline="#444")
        poly([(-half - 10, 12), (half + 10, 12), (half + 10, 58), (-half - 10, 58)], fill="#d8d8d8", outline="#555")
        poly([(-half + 12, 58), (half - 12, 58), (half - 26, 86), (-half + 26, 86)], fill="#9fb8d0", outline="#555")
        for side in (-1, 1):                       # 尾燈
            poly([(side * (half - 2) - 9, 40), (side * (half - 2) + 9, 40), (side * (half - 2) + 9, 50),
                  (side * (half - 2) - 9, 50)], fill="#ff3b3b", outline="")
        cv.create_text(*T(0, 30), text="後", fill="#333", font=("Consolas", 12, "bold"))

        # 上方固定刻度與跟著車子轉的指針
        rr = R - 4
        for a in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            ar = math.radians(a)
            ln = 12 if a in (0, -30, 30, -60, 60) else 7
            cv.create_line(c + rr * math.sin(ar), c - rr * math.cos(ar),
                           c + (rr - ln) * math.sin(ar), c - (rr - ln) * math.cos(ar), fill="white", width=2)
        tip = rr - 14
        a = math.radians(max(-80.0, min(80.0, roll)))
        cv.create_polygon(c + tip * math.sin(a), c - tip * math.cos(a),
                          c + (tip - 12) * math.sin(a) - 7 * math.cos(a), c - (tip - 12) * math.cos(a) - 7 * math.sin(a),
                          c + (tip - 12) * math.sin(a) + 7 * math.cos(a), c - (tip - 12) * math.cos(a) + 7 * math.sin(a),
                          fill="#ff8c1a", outline="")

        mask_circle(cv, c, R)
        cv.create_text(6, 6, text=f"R {roll:+6.1f}°", fill="white", anchor="nw", font=("Consolas", 10, "bold"))
        cv.create_text(W - 6, 6, text=f"P {pitch:+6.1f}°", fill="white", anchor="ne",
                       font=("Consolas", 10, "bold"))
        cv.create_text(c, W - 16, text="Roll", fill="white", font=("Consolas", 10, "bold"))

    def draw_side(pitch):
        """側視圖：從車子右側看，車頭朝右。地面固定水平，車頭抬起 (pitch>0) 時小車逆時針轉、車頭往上，
        以著地那一側的輪子為支點 (車頭抬起時以後輪為支點)。右側刻度與橘色指針表示俯仰角度。"""
        cv = side
        cv.delete("all")
        c = W / 2
        R = c - 12
        k = W / 260                                # 圖形以 260 像素設計，小螢幕等比例縮小
        ground = c + 45 * k
        cv.create_rectangle(0, 0, W, W, fill=SKY, outline="")
        cv.create_rectangle(0, ground, W, W, fill=GROUND, outline="")
        cv.create_line(0, ground, W, ground, fill="white", width=2)

        th = math.radians(max(-80.0, min(80.0, pitch)))
        ct, st = math.cos(th), math.sin(th)
        axle = 52                                  # 前後輪到中間的距離 (像素)
        pivot = (c - axle * k, ground) if pitch >= 0 else (c + axle * k, ground)

        def T(x, y):
            # 車身座標 (x 朝車頭 = 向右, y 向上) → 螢幕座標，逆時針轉 pitch (車頭往上)
            dx, dy = c + x * k - pivot[0], ground - y * k - pivot[1]
            return pivot[0] + dx * ct + dy * st, pivot[1] - dx * st + dy * ct

        def poly(points, **kw):
            pts = []
            for x, y in points:
                pts += T(x, y)
            cv.create_polygon(pts, **kw)

        poly([(-88, 14), (88, 14), (90, 44), (62, 50), (40, 76), (-46, 76), (-66, 50), (-90, 46)],
             fill="#d8d8d8", outline="#555")
        poly([(-36, 52), (-6, 52), (-6, 70), (-32, 70)], fill="#9fb8d0", outline="")
        poly([(2, 52), (34, 52), (24, 70), (2, 70)], fill="#9fb8d0", outline="")
        poly([(80, 30), (90, 30), (90, 40), (80, 40)], fill="#ffe680", outline="")      # 車頭燈
        poly([(-90, 30), (-82, 30), (-82, 40), (-90, 40)], fill="#ff3b3b", outline="")  # 尾燈
        for x in (-axle, axle):                    # 輪子
            wx, wy = T(x, 15)
            cv.create_oval(wx - 15 * k, wy - 15 * k, wx + 15 * k, wy + 15 * k, fill="#111", outline="#444", width=2)
        cv.create_text(*T(52, 92), text="車頭", fill="white", font=("Consolas", 10, "bold"))

        # 右側固定刻度與跟著車子轉的指針 (0 = 水平朝右)
        rr = R - 4
        for a in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
            ar = math.radians(a)
            ln = 12 if a in (0, -30, 30, -60, 60) else 7
            cv.create_line(c + rr * math.cos(ar), c - rr * math.sin(ar),
                           c + (rr - ln) * math.cos(ar), c - (rr - ln) * math.sin(ar), fill="white", width=2)
        tip = rr - 14
        ca, sa = math.cos(th), math.sin(th)
        cv.create_polygon(c + tip * ca, c - tip * sa,
                          c + (tip - 12) * ca - 7 * sa, c - (tip - 12) * sa - 7 * ca,
                          c + (tip - 12) * ca + 7 * sa, c - (tip - 12) * sa + 7 * ca,
                          fill="#ff8c1a", outline="")

        mask_circle(cv, c, R)
        cv.create_text(6, 6, text=f"P {pitch:+6.1f}°", fill="white", anchor="nw", font=("Consolas", 10, "bold"))
        cv.create_text(c, W - 16, text="Pitch", fill="white", font=("Consolas", 10, "bold"))

    def draw_compass(heading):
        """航向圖（從上往下看）：刻度盤固定不動，上方 0° = 按「航向歸零」時車頭的方向（磁力計來源時 = 磁北）；
        中間的車子跟著航向轉，車頭（黃色箭頭）指的刻度就是現在的航向，順時針 = 往右轉。
        室內磁場不準，所以不標東西南北。"""
        cv = compass
        cv.delete("all")
        c = W / 2
        R = c - 12
        k = W / 260                                # 圖形以 260 像素設計，小螢幕等比例縮小
        cv.create_rectangle(0, 0, W, W, fill=BEZEL, outline="")
        cv.create_oval(c - R, c - R, c + R, c + R, fill="#0b0b0b", outline="#555", width=2)

        for b in range(0, 360, 5):                 # 固定刻度：每 30° 標度數
            a = math.radians(b)
            sa, ca = math.sin(a), -math.cos(a)
            ln = (14 if b % 30 == 0 else (9 if b % 10 == 0 else 5)) * max(k, 0.7)
            cv.create_line(c + R * sa, c + R * ca, c + (R - ln) * sa, c + (R - ln) * ca,
                           fill="#ff8c1a" if b == 0 else "white", width=2 if b % 10 == 0 else 1)
            if b % 30 == 0:
                rt = R - 26 * max(k, 0.7)
                cv.create_text(c + rt * sa, c + rt * ca, text=str(b), fill="#ff8c1a" if b == 0 else "#bbb",
                               font=("Consolas", max(7, int(10 * k)), "bold"))

        # 車頭方向的指示線（從圓心到外圈）
        a = math.radians(heading)
        sa, ca = math.sin(a), -math.cos(a)
        cv.create_line(c, c, c + (R - 4) * sa, c + (R - 4) * ca, fill="#ffd400", width=2, dash=(4, 3))

        # 從上往下看的車子：x 向右、y 向前（像素，以 260 設計），跟著航向順時針轉
        def P(x, y):
            x, y = x * k, y * k
            return c + x * math.cos(a) + y * math.sin(a), c + x * math.sin(a) - y * math.cos(a)

        def poly(points, **kw):
            pts = []
            for x, y in points:
                pts += P(x, y)
            cv.create_polygon(pts, **kw)

        for wx in (-27, 27):                       # 四個輪子
            for wy in (-26, 22):
                poly([(wx - 6, wy - 10), (wx + 6, wy - 10), (wx + 6, wy + 10), (wx - 6, wy + 10)],
                     fill="#111", outline="#555")
        poly([(-22, -44), (22, -44), (22, 34), (16, 42), (-16, 42), (-22, 34)], fill="#d8d8d8", outline="#555")
        poly([(-16, 10), (16, 10), (13, 24), (-13, 24)], fill="#9fb8d0", outline="")      # 前擋風玻璃
        poly([(-15, -34), (15, -34), (13, -26), (-13, -26)], fill="#9fb8d0", outline="")  # 後窗
        for hx in (-13, 13):                       # 車頭燈、尾燈
            poly([(hx - 5, 36), (hx + 5, 36), (hx + 5, 41), (hx - 5, 41)], fill="#ffe680", outline="")
            poly([(hx - 5, -44), (hx + 5, -44), (hx + 5, -40), (hx - 5, -40)], fill="#ff3b3b", outline="")
        poly([(0, 66), (-10, 50), (10, 50)], fill="#ffd400", outline="#7a6500")           # 車頭箭頭

        # 數位讀數：航向，以及相對 0° 往左或往右轉了幾度
        rel = (heading + 180) % 360 - 180
        turn = "正前" if abs(rel) < 0.05 else (f"右 {rel:.1f}°" if rel > 0 else f"左 {-rel:.1f}°")
        f = ("Consolas", max(8, int(10 * k) + 1), "bold")
        cv.create_text(6, 6, text=f"{heading % 360:05.1f}°", fill="#ff8c1a", anchor="nw", font=f)
        cv.create_text(W - 6, 6, text=turn, fill="#ffd400", anchor="ne", font=f)

    last = {"t": time.time(), "n": 0, "rate": 0.0}
    # 畫面只在有變化時才重畫：每次重畫都要合成整個畫面（GPU）並讓 VNC 重新傳送，2026-11-23 量到儀表每秒重畫 20 次
    # 讓 GPU 忙 90%。儀表：傾斜變 0.1° 以上、航向變 0.2° 以上才重畫；文字：每 0.2 秒、內容有變才更新。
    shown = {"att": None, "text_t": 0.0}
    texts = {}

    def set_text(widget, text):
        if texts.get(widget) != text:
            texts[widget] = text
            widget.config(text=text)

    def tick():
        with reader.lock:
            d, n = reader.latest, reader.frames
        now = time.time()
        if now - last["t"] >= 1:
            last["rate"] = (n - last["n"]) / (now - last["t"])
            last["t"], last["n"] = now, n
        if d:
            roll, pitch, yaw = vehicle(d)
            att = shown["att"]
            if att is None or abs(roll - att[0]) >= 0.1 or abs(pitch - att[1]) >= 0.1 or abs(wrap(yaw - att[2])) >= 0.2:
                shown["att"] = (roll, pitch, yaw)
                draw_horizon(roll, pitch)
                draw_side(pitch)
                draw_compass(yaw)
            update_cal(d, n)
            if logger["writer"]:
                vec = lambda v: list(v) if v is not None else ["", "", ""]
                logger["writer"].writerow([f"{now:.3f}", roll, pitch, f"{yaw:.2f}", d["yaw"], d["temp"],
                                           *vec(d["acc"]), *vec(d["gyr"]), *vec(d.get("mag")),
                                           "" if d.get("level") is None else d["level"]])
        if d and now - shown["text_t"] >= 0.2:
            shown["text_t"] = now
            set_text(big["Roll"], f"{roll:.2f}°")
            set_text(big["Pitch"], f"{pitch:.2f}°")
            set_text(big["Yaw"], f"{yaw:.2f}°")
        if d and now - shown["text_t"] < 0.01 and info.winfo_ismapped():   # 詳細資訊展開時才更新
            set_text(big["溫度"], f"{d['temp']:7.2f}°C" if d["temp"] is not None else "--")
            with reader.lock:
                cfg = reader.cfg
            lines = []
            if d["acc"] is not None and d["gyr"] is not None:
                ax, ay, az = d["acc"]
                gx, gy, gz = d["gyr"]
                if cfg:
                    ka, kg = cfg["acc_fs"] / 32768, cfg["gyr_fs"] / 32768
                    lines.append(f"Acc  (g)   x {ax*ka:8.3f}  y {ay*ka:8.3f}  z {az*ka:8.3f}")
                    lines.append(f"Gyro (°/s) x {gx*kg:8.2f}  y {gy*kg:8.2f}  z {gz*kg:8.2f}")
                lines.append(f"Acc  raw   x {ax:8d}  y {ay:8d}  z {az:8d}")
                lines.append(f"Gyro raw   x {gx:8d}  y {gy:8d}  z {gz:8d}")
            else:
                lines.append("Acc/Gyro   模組目前只輸出角度（預設設定）")
            if d.get("mag") is not None:
                lines.append(f"Mag  raw   x {d['mag'][0]:8d}  y {d['mag'][1]:8d}  z {d['mag'][2]:8d}")
            if d.get("quat"):
                lines.append("Quat       " + "  ".join(f"q{i} {q:7.4f}" for i, q in enumerate(d["quat"])))
            lines.append(f"模組原始   Roll {d['roll']:7.2f}  Pitch {d['pitch']:7.2f}  Yaw {d['yaw']:7.2f}")
            mh = mag_heading(d)
            yaw_based = ((-d["yaw"] if opt_rev_yaw.get() else d["yaw"]) - X_DIRS[opt_xdir.get()]) % 360
            if mh is not None:
                lines.append(f"車頭航向   磁力計 {mh:6.1f}°   模組 Yaw 換算 {yaw_based:6.1f}°（未扣歸零）")
            if d.get("gyr") is not None:
                bias_now = reader._bias_buf is not None
                lines.append(f"陀螺儀     {reader.gyro_heading:6.1f}°（未扣歸零）   零點 z {reader.gyro_bias[2]:+.3f}°/s"
                             + ("   ← 量零點中，請保持靜止" if bias_now else ""))
            else:
                lines.append("陀螺儀     模組沒有輸出角速度，航向改用模組 Yaw（六軸的 Yaw 本身就是陀螺儀算的）")
            if d.get("level") is not None:
                lines.append(f"Level      {d['level']}  (磁場校準精度，越接近 100 越好)")
            if cfg:
                mag = f"  ±{cfg['mag_fs']}Gauss" if "mag_fs" in cfg else ""
                lines.append(f"量程       ±{cfg['acc_fs']}g  ±{cfg['gyr_fs']}°/s{mag}  安裝:{cfg['mount']}")
            else:
                lines.append("量程       讀取中…")
            set_text(detail, "\n".join(lines))
            set_text(status, f"{M['name']}  {port} @ {baud}   {last['rate']:.0f} 幀/秒   "
                             f"有效 {n}   校驗失敗 {reader.bad}")
        root.after(50, tick)

    def on_close():
        if logger["file"]:
            logger["file"].close()
        if cam_panel is not None:
            cam_panel.close()
        reader.close()
        root.destroy()

    def place_camera():
        """IMU 部分排好後，看螢幕還剩多少寬度：夠就把相機放在右邊第三欄；不夠（例如 1024x600）就把航向儀
        移到兩個姿態儀旁邊，相機放在它們下面。"""
        nonlocal cam_panel
        if CameraPanel is None:
            return
        sw, sh = (1024, 600) if screen == "small" else (1920, 1080) if screen == "large" else \
            (root.winfo_screenwidth(), root.winfo_screenheight())
        root.update_idletasks()
        avail = sw - root.winfo_reqwidth() - 40
        cam_font = ("Consolas", 9 if small else 11)
        try:
            if avail >= 400:
                cw = min(640, avail) // 4 * 4
                ch = min(cw * 3 // 4, sh - 160) // 3 * 3
                col = tk.Frame(root, bg=bg)
                col.pack(side="left", padx=(0, 12), pady=12, anchor="n")
                cam_panel = CameraPanel(col, ch * 4 // 3, ch, bg, fg, cam_font, start=camera or None)
                cam_panel.frame.pack(pady=(8, 0))
            else:
                comp_box.pack_forget()
                comp_box.pack(in_=top_row, side="left", padx=(8, 0))
                gauge_caption.config(text="Roll（從車尾看）　Pitch（從側面看）　航向")
                comp_row.pack_forget()                   # 這一列空了；不拿掉會留一塊空白（Tk 的框不會自己縮回去）
                cam_row = tk.Frame(left, bg=bg)          # 相機畫面＋右邊的電池與馬達
                cam_row.pack(anchor="w", pady=(8, 0))
                sp = status_box["panel"]
                # 右欄變窄、變矮：按鈕排成兩列，狀態列換行，安裝方向裡的括號說明拿掉（展開後才放得進 600 高）
                status.config(wraplength=380, justify="left")
                for w in opts.winfo_children():
                    if isinstance(w, tk.Label) and str(w.cget("text")).startswith("（"):
                        w.pack_forget()
                info_btn.config(wraplength=380, justify="left")
                kids = btns.winfo_children()
                for k in kids:
                    k.pack_forget()
                for i, k in enumerate(kids):
                    k.config(width=11)
                    k.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky="ew")
                root.update_idletasks()
                cw = top_row.winfo_reqwidth() - (sp.frame.winfo_reqwidth() + 14 if sp is not None else 0)
                ch = min(cw * 3 // 4, sh - 110 - big_box.winfo_reqheight() - top_row.winfo_reqheight() - 60)
                cam_panel = CameraPanel(left, ch * 4 // 3, ch, bg, fg, cam_font, start=camera or None)
                cam_panel.frame.pack(in_=cam_row, side="left", anchor="n")
                if sp is not None:
                    sp.frame.pack_forget()
                    sp.frame.pack(in_=cam_row, side="left", anchor="n", padx=(14, 0))
        except Exception as e:  # noqa: BLE001
            print(f"不顯示相機區：{e}")
            cam_panel = None

    # 電池與馬達面板（tools/car_status_panel.py）：馬達指令來自地圖區讀到的車子程式即時紀錄
    status_box = {"panel": None}

    def live_row():
        mp = map_box["panel"]
        if mp is None or mp.run is None:
            return None, None
        from map_panel import KIND
        return KIND.get(mp.run.parent.name, "執行中"), mp.row

    try:
        from car_status_panel import CarStatusPanel
        status_box["panel"] = CarStatusPanel(root, 200 if small else W - 6, bg, fg,      # 大螢幕：寬到和兩個姿態儀對齊
                                             ("Consolas", 9 if small else 11), live_row, big=not small)
        status_box["panel"].frame.pack(in_=comp_row, side="left", anchor="n", padx=(14, 0), pady=(8, 0))
    except Exception as e:  # noqa: BLE001  別的電腦沒有電池讀取
        print(f"不顯示電池與馬達：{e}")

    def place_map():
        """地圖放在右欄按鈕下面：寬 = 按鈕那一排，高不超過左欄。沒有 PIL 等套件（例如別的電腦）就不放。"""
        try:
            from map_panel import MapPanel
        except Exception as e:  # noqa: BLE001
            print(f"不顯示地圖區：{e}")
            return
        root.update_idletasks()
        size = min(btn_slot.winfo_reqwidth(),
                   left.winfo_reqheight() - btn_slot.winfo_reqheight() - info_btn.winfo_reqheight() - 50, 620)
        if size < 200:
            return
        map_box["panel"] = MapPanel(right, size, bg, fg, ("Consolas", 9 if small else 10))
        map_box["panel"].frame.pack(anchor="w", pady=(8, 0))

    def place_and_center():
        # 視窗出現後才排相機區：在 mainloop 之前呼叫 update_idletasks，「模組 X 軸朝向」那一列會畫不出來
        place_camera()
        place_map()
        if cam_panel is not None or map_box["panel"] is not None:   # 視窗變大了：放到螢幕中間
            root.update_idletasks()
            ww, wh = root.winfo_reqwidth(), root.winfo_reqheight()
            root.geometry(f"+{max(0, (root.winfo_screenwidth() - ww) // 2)}"
                          f"+{max(0, (root.winfo_screenheight() - wh) // 2 - 20)}")

    root.after(200, place_and_center)
    root.protocol("WM_DELETE_WINDOW", on_close)
    draw_horizon(0, 0)
    draw_side(0)
    draw_compass(0)
    tick()
    draw_plots()
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GY-95T / GY-25T 即時檢視器")
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--model", choices=["auto", "95t", "25t"], default="auto")
    ap.add_argument("--baud", type=int, default=None, help="不指定時: 自動偵測或用型號預設值")
    ap.add_argument("--test", action="store_true", help="只在終端機印出資料")
    ap.add_argument("--xdir", choices=["車頭", "右側", "車尾", "左側"], default=None, help="模組 X 軸朝向")
    ap.add_argument("--rev-yaw", type=int, choices=[0, 1], default=None)
    ap.add_argument("--rev-roll", type=int, choices=[0, 1], default=None)
    ap.add_argument("--rev-pitch", type=int, choices=[0, 1], default=None)
    ap.add_argument("--camera", choices=["雲台", "前", "後", "左", "右", "鳥瞰", "關", "無"], default="雲台",
                    help="右側相機區一開始顯示哪支（關 = 有相機區但先不開；無 = 不要相機區）；只在車上的 Pi 有")
    ap.add_argument("--screen", choices=["auto", "small", "large"], default="auto", help="版面大小（預設依螢幕）")
    a = ap.parse_args()

    model, baud = a.model, a.baud
    if model == "auto":
        print(f"偵測 {a.port} 上的模組…")
        model, found = detect(a.port, (baud,) if baud else (115200, 9600, 230400, 57600, 38400, 19200))
        if model is None:
            raise SystemExit(f"{a.port} 上找不到 GY-95T / GY-25T 的資料。請確認接線，或用 --model / --baud 指定。")
        baud = found
        print(f"偵測到 {MODELS[model]['name']} @ {baud}")
    elif baud is None:
        baud = MODELS[model]["baud"]
    mount = {"xdir": a.xdir, "rev_yaw": a.rev_yaw, "rev_roll": a.rev_roll, "rev_pitch": a.rev_pitch}
    cam = {"雲台": "gimbal", "前": "front", "後": "rear", "左": "left", "右": "right", "鳥瞰": "bev", "關": "",
           "無": None}[a.camera]
    run_test(a.port, baud, model) if a.test else run_gui(a.port, baud, model, mount, cam, a.screen)
