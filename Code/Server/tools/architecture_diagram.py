#!/usr/bin/env python3
"""Draw the system overview of the lap (slide 3): python3 tools/architecture_diagram.py out_dir"""
import pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

TEXT = {
    "en": dict(
        title="System overview: sensing → estimating → deciding → moving",
        cols=["Sensors", "Perception and state", "Decision", "Actuators"],
        sensors=["Front + rear cameras\n(wall ArUco tags)", "Gimbal camera\n(foot line of blocks)", "Ultrasonic\n(range ahead / side)",
                 "IR sensors ×6\n(front corners ×2, wheels ×4)", "Wheel commands\n(timed moves)"],
        perc=["Tag pose fit", "Pose filter\n(dead reckoning + tag fixes\n+ learned turn rate)", "Foot points → obstacle map\n(blocks, 3 sightings)"],
        dec=["Lane follower\n4 lanes, corners, squaring up", "Detour\nlook → close in → strafe\n→ pass → return", "Parking\nmap home, back in"],
        act=["4 mecanum motors\n(forward, strafe, turn)", "Gimbal servo\n(look −80°…+80°)"],
        fb="pose and map feed the next decision", safety="IR + ultrasonic stop a move at once"),
    "zh": dict(
        title="系統架構：感測 → 估計 → 決策 → 動作",
        cols=["感測器", "感知與狀態", "決策", "執行"],
        sensors=["前後相機\n（牆上 ArUco 標籤）", "雲台相機\n（方塊腳線）", "超音波\n（前方／側邊距離）",
                 "紅外線 ×6\n（車頭左右角 2、輪邊 4）", "輪子指令\n（計時移動）"],
        perc=["標籤姿態擬合", "姿態濾波器\n（推算＋標籤修正\n＋學習轉速）", "腳線點 → 障礙物地圖\n（看到 3 次才畫）"],
        dec=["車道跟隨\n四條車道、轉角、調正", "繞行\n看→靠近→側移\n→通過→回來", "停車\n地圖家位置、倒車進入"],
        act=["四個麥克納姆輪馬達\n（前進、側移、轉向）", "雲台伺服\n（看 −80°…+80°）"],
        fb="姿態與地圖進入下一個決策", safety="紅外線＋超音波即時停止移動"),
}

def box(ax, x, y, w, h, s, fc, size=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec="#33415c", lw=1.2))
    ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=size, color="#111")

def arrow(ax, x0, y0, x1, y1, style="-|>", color="#33415c", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle=style, color=color, lw=1.3, ls=ls))

def draw(lang, out):
    t = TEXT[lang]
    if lang == "zh":
        font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
        plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]  # Latin and arrows from DejaVu, Chinese from Droid
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, t["title"], ha="center", fontsize=14, weight="bold")
    xs, ws = [0.02, 0.28, 0.58, 0.83], [0.20, 0.25, 0.22, 0.15]
    colors = ["#dbe7f5", "#e3f1df", "#fdf0d5", "#f5dde0"]
    for x, w, name, c in zip(xs, ws, t["cols"], colors):
        ax.text(x + w / 2, 0.915, name, ha="center", fontsize=11, weight="bold")
        ax.add_patch(FancyBboxPatch((x - 0.008, 0.10), w + 0.016, 0.79, boxstyle="round,pad=0.005,rounding_size=0.02", fc=c, ec="none", alpha=0.45))
    # sensor order: cams, wheel commands, ultrasonic, gimbal camera, IR
    order = [0, 4, 2, 1, 3]
    sens = [t["sensors"][i] for i in order]
    n = len(sens); h = 0.125; gap = (0.76 - n * h) / (n - 1)
    sy = [0.12 + (n - 1 - i) * (h + gap) for i in range(n)]
    for y, sx in zip(sy, sens):
        box(ax, xs[0], y, ws[0], h, sx, "white", 8.5)
    py = [0.67, 0.40, 0.14]; ph = [0.17, 0.22, 0.20]      # tag fit, pose filter, map
    for y, sx, hh in zip(py, t["perc"], ph):
        box(ax, xs[1], y, ws[1], hh, sx, "white", 9)
    dy = [0.66, 0.40, 0.14]                                 # lane, detour, parking
    for y, sx in zip(dy, t["dec"]):
        box(ax, xs[2], y, ws[2], 0.19, sx, "white", 9)
    ay = [0.58, 0.20]                                       # gimbal servo, motors
    ah = [0.20, 0.26]
    servo, motors = t["act"][1], t["act"][0]
    box(ax, xs[3], ay[0], ws[3], ah[0], servo, "white", 8.5)
    box(ax, xs[3], ay[1], ws[3], ah[1], motors, "white", 8.5)
    R0, L1 = xs[0] + ws[0], xs[1]
    mid = lambda i: sy[i] + h / 2
    arrow(ax, R0, mid(0), L1, py[0] + ph[0] / 2)                 # cameras -> tag fit
    arrow(ax, R0, mid(1), L1, py[1] + ph[1] * 0.75)              # wheel commands -> filter
    arrow(ax, R0, mid(2), L1, py[1] + ph[1] * 0.35)              # ultrasonic -> filter (wall fix)
    arrow(ax, R0, mid(3), L1, py[2] + ph[2] / 2)                 # gimbal camera -> map
    cx = xs[1] + ws[1] / 2
    arrow(ax, cx, py[0], cx, py[1] + ph[1])                      # tag fit -> filter
    arrow(ax, cx, py[1], cx, py[2] + ph[2])                      # filter -> map
    R1, L2 = xs[1] + ws[1], xs[2]
    arrow(ax, R1, py[1] + ph[1] * 0.8, L2, dy[0] + 0.095)        # filter -> lane follower
    arrow(ax, R1, py[2] + ph[2] * 0.6, L2, dy[1] + 0.07)         # map -> detour
    arrow(ax, R1, py[1] + ph[1] * 0.2, L2, dy[2] + 0.15)         # filter -> parking
    R2, L3 = xs[2] + ws[2], xs[3]
    for y, target in zip(dy, (0.40, 0.33, 0.26)):                 # decisions -> motors
        arrow(ax, R2, y + 0.095, L3, ay[1] + (target - 0.20))
    arrow(ax, R2, dy[1] + 0.15, L3, ay[0] + 0.05)                # detour -> gimbal servo (look)
    # IR and ultrasonic stop a move at once: dashed path under the boxes to the motors
    red = "#b3261e"
    arrow(ax, xs[0] + ws[0] * 0.35, sy[4], xs[0] + ws[0] * 0.35, 0.075, style="-", color=red, ls="--")
    arrow(ax, xs[0] + ws[0] * 0.35, 0.075, xs[3] + ws[3] / 2, 0.075, style="-", color=red, ls="--")
    arrow(ax, xs[3] + ws[3] / 2, 0.075, xs[3] + ws[3] / 2, ay[1], color=red, ls="--")
    ax.text(0.5, 0.045, t["safety"], ha="center", fontsize=9, color=red)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)
for lang in ("en", "zh"):
    draw(lang, out / f"architecture_{lang}.png")
    print(out / f"architecture_{lang}.png")
