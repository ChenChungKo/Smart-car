"""Top-down drawings for the dashboard: the arena map with the car, and the car-centred bird's-eye view frame."""
from __future__ import annotations

import math

import cv2
import numpy as np

# six IR modules of the car, car frame (x forward, y left, m) and the way they look (deg, + = left); see ir_obstacle_pins.json
IR_POSITIONS = {
    "front_left": (0.12, 0.07, 0.0),
    "front_right": (0.12, -0.07, 0.0),
    "wheel_left_front": (0.06, 0.085, 90.0),
    "wheel_left_rear": (-0.06, 0.085, 90.0),
    "wheel_right_front": (0.06, -0.085, -90.0),
    "wheel_right_rear": (-0.06, -0.085, -90.0),
}
WALL_COLORS = {"bottom": (200, 120, 40), "right": (200, 120, 40), "top": (90, 90, 90), "left": (90, 90, 90)}  # BGR: blue, blue, grey, grey
SONIC_PIVOT_M, SONIC_ARM_M = 0.085, 0.03


def body_polygon(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2)]
    return np.array([(x + px * c - py * s, y + px * s + py * c) for px, py in pts])


def render_map(state: dict, arena: dict, car: dict, layers: set[str], size_px: int = 640) -> np.ndarray:
    """Arena seen from above (+y up). Layers: trail, tags, blocks, lane, sensors, fov, uncertainty."""
    w_m, h_m = (float(v) for v in arena["arena_size_m"])
    margin = 34
    scale = (size_px - 2 * margin) / max(w_m, h_m)
    img = np.full((int(h_m * scale) + 2 * margin, int(w_m * scale) + 2 * margin, 3), 247, np.uint8)
    img_h = img.shape[0]

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round(margin + x * scale)), int(round(img_h - margin - y * scale))

    # the walls in their colours: the IR sensors see the blue ones and not the grey ones
    cv2.line(img, px(0, 0), px(w_m, 0), WALL_COLORS["bottom"], 5)
    cv2.line(img, px(w_m, 0), px(w_m, h_m), WALL_COLORS["right"], 5)
    cv2.line(img, px(w_m, h_m), px(0, h_m), WALL_COLORS["top"], 5)
    cv2.line(img, px(0, h_m), px(0, 0), WALL_COLORS["left"], 5)
    for k in range(1, 7):  # 20 cm grid
        gx = k * 0.2
        if gx < w_m:
            cv2.line(img, px(gx, 0), px(gx, h_m), (232, 232, 232), 1)
        if gx < h_m:
            cv2.line(img, px(0, gx), px(w_m, gx), (232, 232, 232), 1)
    if "tags" in layers:
        half = float(arena["tag_size_m"]) / 2.0
        seen = {t for ids in (state.get("tags_seen") or {}).values() for t in ids}
        for tag in arena["tags"]:
            cx, cy = tag["center_m"][:2]
            psi = math.radians(float(tag["facing_deg"]))
            rx, ry = -math.sin(psi), math.cos(psi)
            color = (0, 170, 0) if tag["id"] in seen else (0, 120, 255)
            cv2.line(img, px(cx - rx * half, cy - ry * half), px(cx + rx * half, cy + ry * half), color, 5)
            lx, ly = px(cx + math.cos(psi) * 0.065, cy + math.sin(psi) * 0.065)
            cv2.putText(img, str(tag["id"]), (lx - 5, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    home = state.get("home")
    if home:
        cv2.drawMarker(img, px(home[0], home[1]), (0, 140, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    if "lane" in layers:
        for a, b in zip(state.get("route") or [], (state.get("route") or [])[1:]):
            cv2.line(img, px(*a), px(*b), (210, 190, 120), 1, cv2.LINE_AA)
    if "blocks" in layers:
        for outline in state.get("blocks") or []:
            poly = np.array([px(x, y) for x, y in outline], np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], (90, 90, 230))
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [poly], True, (50, 50, 200), 2, cv2.LINE_AA)
    if "trail" in layers and len(state.get("trail") or []) >= 2:
        pts = np.array([px(x, y) for x, y in state["trail"]], np.int32)
        cv2.polylines(img, [pts], False, (60, 60, 220), 1, cv2.LINE_AA)
    pose = state.get("pose")
    if pose:
        x, y, yaw = pose["x"], pose["y"], math.radians(pose["yaw_deg"])
        length, width = (float(v) for v in car.get("body_m", [0.275, 0.17]))
        if "uncertainty" in layers and pose.get("sigma_cm"):
            cv2.circle(img, px(x, y), max(int(pose["sigma_cm"] / 100.0 * scale), 3), (150, 150, 255), 1, cv2.LINE_AA)
        if "fov" in layers:
            for name, ang in (("front", 0.0), ("rear", 180.0), ("left", 90.0), ("right", -90.0)):
                a0, a1 = math.radians(ang) + yaw - 0.9, math.radians(ang) + yaw + 0.9
                tri = np.array([px(x, y), px(x + 0.45 * math.cos(a0), y + 0.45 * math.sin(a0)),
                                px(x + 0.45 * math.cos(a1), y + 0.45 * math.sin(a1))], np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [tri], (200, 230, 200))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        poly = np.array([px(*p) for p in body_polygon(x, y, yaw, length, width)], np.int32)
        cv2.fillPoly(img, [poly], (225, 225, 250))
        cv2.polylines(img, [poly], True, (40, 40, 200), 2, cv2.LINE_AA)
        tip = px(x + 0.16 * math.cos(yaw), y + 0.16 * math.sin(yaw))
        cv2.arrowedLine(img, px(x, y), tip, (40, 40, 200), 2, tipLength=0.35)
        if "sensors" in layers:
            c, s = math.cos(yaw), math.sin(yaw)
            for name, (lx, ly, look) in IR_POSITIONS.items():
                wx, wy = x + lx * c - ly * s, y + lx * s + ly * c
                lit = (state.get("ir") or {}).get(name)
                cv2.circle(img, px(wx, wy), 4, (0, 0, 230) if lit else (170, 170, 170), -1)
            sonic = state.get("sonic_cm")
            pan = math.radians(state.get("sonic_pan_deg") or 0.0)
            if sonic:
                ox = x + SONIC_PIVOT_M * c
                oy = y + SONIC_PIVOT_M * s
                a = yaw + pan
                end = (ox + (SONIC_ARM_M + sonic / 100.0) * math.cos(a), oy + (SONIC_ARM_M + sonic / 100.0) * math.sin(a))
                cv2.line(img, px(ox, oy), px(*end), (0, 150, 0), 1, cv2.LINE_AA)
                cv2.circle(img, px(*end), 3, (0, 150, 0), -1)
    cv2.putText(img, "x", (px(w_m, 0)[0] + 8, px(w_m, 0)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(img, "y", (px(0, h_m)[0] - 4, px(0, h_m)[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return img
